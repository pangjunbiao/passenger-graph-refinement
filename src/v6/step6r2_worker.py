"""Exact-runtime worker for the disclosed V6.1 development recovery.

This worker deliberately reuses the already-spent validation cohort.  It does
not describe that cohort as confirmation and it never exposes test, baseline,
SOTA, or label artifacts.  The three parent fits frozen before the failed
Step-6 gate are reused without parameter updates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.v6.city_backoff import mix_with_prior
from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_evidence import (
    matched_topic_stability,
    positive_npmi_graph,
    topic_quality_metrics,
)
from src.v6.development_worker import _new_parent, _state_hash
from src.v6.graph_prototype_adapter import minimum_distortion_graph_projection
from src.v6.recovery_gate import (
    select_recovery_candidate,
    summarize_recovery_candidate,
)
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt
from src.v6.step4r2_worker import _parent_probabilities
from src.v6.step5r2_worker import _seed_everything, _top_word_support
from src.v6.step6_worker import (
    _load_training,
    _load_validation,
    _variant_completion,
)
from src.v6.transport_calibration import (
    apply_frozen_moment_calibration,
    fit_changed_column_moment_calibration,
)
from src.v6.validation_context import construct_training_only_context


IMPLEMENTATION = "v6_1_step6r2_spent_validation_development_recovery_r1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty required table: {path.name}")
    pd.DataFrame.from_records(rows).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        job.get("schema_version") != 1
        or job.get("mode") != "step6r2"
        or job.get("implementation_version") != IMPLEMENTATION
    ):
        raise ValueError("unsupported V6.1 Step-6R2 worker job")
    for role, item in job["lineage_artifacts"].items():
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != str(item["sha256"]):
            raise ValueError(f"recovery lineage changed: {role}")
    for row in job["checkpoint_manifest"]:
        source = Path(row["path"])
        if not source.is_file() or sha256_file(source) != str(row["sha256"]):
            raise ValueError("a frozen Step-6 parent checkpoint changed")
    if job["data_policy"].get("test_available_to_worker") is not False:
        raise ValueError("test access must be structurally absent")
    if job["data_policy"].get("validation_status") != "SPENT_DEVELOPMENT":
        raise ValueError("the reused validation cohort must be marked spent")
    if job["model_revision"].get("city_specific_backoff") != "RETIRED":
        raise ValueError("the failed city-specific mechanism was not retired")
    return job


def _latent_product(
    parent: Any,
    input_values: np.ndarray,
    *,
    batch_size: int,
    device: Any,
    torch: Any,
) -> np.ndarray:
    parent.eval()
    vocabulary = int(parent.vocab_size)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(input_values), int(batch_size)):
            stop = min(start + int(batch_size), len(input_values))
            batch = torch.from_numpy(
                np.asarray(input_values[start:stop], dtype=np.float32)
            ).to(device)
            local_theta, _ = parent.noise_local_encode(batch[:, :vocabulary])
            global_theta, _ = parent.global_encode(batch[:, vocabulary:])
            rows.append(
                (global_theta * local_theta)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.concatenate(rows, axis=0)


def _raw_logits(
    latent: np.ndarray,
    affinity: np.ndarray,
    *,
    device: Any,
    torch: Any,
) -> np.ndarray:
    with torch.no_grad():
        values = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(device)
        matrix = torch.from_numpy(np.asarray(affinity, dtype=np.float32)).to(device)
        return (values @ matrix).detach().cpu().numpy().astype(np.float64)


def _decode_logits(
    parent: Any,
    logits: np.ndarray,
    *,
    batch_size: int,
    device: Any,
    torch: Any,
) -> np.ndarray:
    parent.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(logits), int(batch_size)):
            stop = min(start + int(batch_size), len(logits))
            values = torch.from_numpy(
                np.asarray(logits[start:stop], dtype=np.float32)
            ).to(device)
            probability = torch.softmax(parent.decoder_bn(values), dim=-1)
            rows.append(probability.detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(rows, axis=0)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError("decoder returned invalid probabilities")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise FloatingPointError("decoder probability rows are not normalized")
    return result


def _reporting_affinity(affinity: np.ndarray) -> np.ndarray:
    values = np.asarray(affinity, dtype=np.float64)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1.0e-300)


def _candidate_id(quota: int, calibration: str) -> str:
    return f"quota_{int(quota):02d}__{calibration}"


def _completion(
    target: Any,
    variants: Mapping[str, np.ndarray],
    city_indices: np.ndarray,
    city_names: list[str],
    probability_floor: float,
    *,
    seed: int,
    candidate_id: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], float]:
    summaries: dict[str, dict[str, Any]] = {}
    city_rows: list[dict[str, Any]] = []
    maximum_probability_error = 0.0
    for variant, probability in variants.items():
        maximum_probability_error = max(
            maximum_probability_error,
            float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
        )
        overall, per_city = _variant_completion(
            target,
            probability,
            city_indices,
            city_names,
            probability_floor,
        )
        summaries[variant] = overall
        for row in per_city:
            city_rows.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "variant": variant,
                    **row,
                }
            )
    return summaries, city_rows, maximum_probability_error


def _evaluate_seed(
    job: Mapping[str, Any],
    checkpoint_row: Mapping[str, Any],
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    training_input: np.ndarray,
    validation_input: np.ndarray,
    graph: np.ndarray,
    old_metric: Mapping[str, Any],
    NewMethod: type,
    *,
    device: Any,
    torch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    payload = torch.load(
        Path(checkpoint_row["path"]), map_location="cpu", weights_only=False
    )
    seed = int(checkpoint_row["seed"])
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("implementation_version")
        != "v6_step6_one_shot_validation_freeze_gate_r1"
        or int(payload.get("seed", -1)) != seed
        or str(payload.get("parent_state_sha256"))
        != str(checkpoint_row["parent_state_sha256"])
        or payload.get("model_configuration") != job["model"]
        or payload.get("official_source_commit") != job["official_commit"]
    ):
        raise ValueError("frozen Step-6 parent payload changed identity")
    parent = _new_parent(NewMethod, training["word_embeddings"], job["model"])
    parent.load_state_dict(payload["parent_state_dict"], strict=True)
    parent = parent.to(device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    parent_before = _state_hash(parent.state_dict())
    if parent_before != str(payload["parent_state_sha256"]):
        raise ValueError("frozen parent state hash does not reproduce")

    with torch.no_grad():
        base_affinity = parent.get_beta().detach().cpu().numpy().astype(np.float64)
    aligned_cores = (
        payload["aligned_prototype_core_indices"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    pooled_prior = payload["pooled_prior"].detach().cpu().numpy().astype(np.float64)
    pooled_rows = np.repeat(
        pooled_prior[None, :], len(validation["city_indices"]), axis=0
    )
    latent_train = _latent_product(
        parent,
        training_input,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    latent_validation = _latent_product(
        parent,
        validation_input,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    base_train_logits = _raw_logits(
        latent_train, base_affinity, device=device, torch=torch
    )
    base_validation_logits = _raw_logits(
        latent_validation, base_affinity, device=device, torch=torch
    )
    base_probability = _decode_logits(
        parent,
        base_validation_logits,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    official_probability = _parent_probabilities(
        parent,
        validation_input,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
    )
    parent_probability_error = float(
        np.max(np.abs(base_probability - official_probability))
    )
    without_graph_pooled = mix_with_prior(
        base_probability,
        pooled_rows,
        mixture_weight=float(job["model_revision"]["pooled_mixture_weight"]),
    )
    control_reporting = _reporting_affinity(base_affinity)
    quality_arguments = {
        "top_words": int(job["evaluation"]["top_words"]),
        "window_size": int(job["evaluation"]["c_v_window_size"]),
        "gamma": float(job["evaluation"]["c_v_gamma"]),
    }
    control_validation_quality = topic_quality_metrics(
        control_reporting, validation["full_counts"], **quality_arguments
    )
    rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    top_words: dict[str, np.ndarray] = {}
    calibration_receipts: list[dict[str, Any]] = []
    reporting_cache: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray]] = {}

    for quota in map(int, job["candidates"]["prototype_quotas"]):
        projection = minimum_distortion_graph_projection(
            base_affinity,
            aligned_cores,
            strength=1.0,
            prototype_quota=quota,
            rank_margin=float(job["projection"]["rank_margin"]),
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        projected_train_logits = _raw_logits(
            latent_train, projection.affinity, device=device, torch=torch
        )
        projected_validation_logits = _raw_logits(
            latent_validation, projection.affinity, device=device, torch=torch
        )
        native_probability = _decode_logits(
            parent,
            projected_validation_logits,
            batch_size=int(job["evaluation"]["inference_batch_size"]),
            device=device,
            torch=torch,
        )
        reporting = _reporting_affinity(projection.affinity)
        validation_quality = topic_quality_metrics(
            reporting, validation["full_counts"], **quality_arguments
        )
        training_quality = topic_quality_metrics(
            reporting, training["counts"], **quality_arguments
        )
        selected_words = np.asarray(
            validation_quality["top_word_indices"], dtype=np.int64
        )
        support = _top_word_support(
            selected_words,
            graph,
            training["counts"],
            minimum_document_frequency=int(
                job["lexical_graph"]["minimum_document_frequency"]
            ),
        )
        reporting_cache[quota] = (
            validation_quality,
            training_quality,
            support,
            selected_words,
        )
        for calibration_name in job["candidates"]["calibrations"]:
            candidate = _candidate_id(quota, str(calibration_name))
            if calibration_name == "native_decoder":
                graph_probability = native_probability
                calibration_audit = {
                    "schema_version": 1,
                    "fit_partition": "identity_no_fit",
                    "construction": "native_decoder_identity",
                    "changed_columns": int(projection.audit["changed_columns"]),
                    "maximum_changed_mean_error": 0.0,
                    "maximum_changed_regularized_variance_error": 0.0,
                    "maximum_unchanged_identity_error": 0.0,
                    "minimum_changed_scale": 1.0,
                    "maximum_changed_scale": 1.0,
                    "maximum_absolute_changed_shift": 0.0,
                }
            elif calibration_name == "training_moment_match":
                calibration = fit_changed_column_moment_calibration(
                    base_train_logits,
                    projected_train_logits,
                    base_affinity,
                    projection.affinity,
                    variance_floor=float(job["calibration"]["variance_floor"]),
                    change_tolerance=float(job["calibration"]["change_tolerance"]),
                )
                calibrated_logits = apply_frozen_moment_calibration(
                    projected_validation_logits, calibration
                )
                graph_probability = _decode_logits(
                    parent,
                    calibrated_logits,
                    batch_size=int(job["evaluation"]["inference_batch_size"]),
                    device=device,
                    torch=torch,
                )
                calibration_audit = dict(calibration.audit)
            else:
                raise ValueError(f"unknown recovery calibration: {calibration_name}")
            calibration_receipts.append(
                {"candidate_id": candidate, "seed": seed, **calibration_audit}
            )
            rho = float(job["model_revision"]["pooled_mixture_weight"])
            full_probability = mix_with_prior(
                graph_probability, pooled_rows, mixture_weight=rho
            )
            native_full_probability = mix_with_prior(
                native_probability, pooled_rows, mixture_weight=rho
            )
            variants = {
                "full_graph_plus_pooled_backoff": full_probability,
                "without_graph_pooled_backoff_retained": without_graph_pooled,
                "without_pooled_backoff_graph_retained": graph_probability,
                "same_quota_native_decoder": native_full_probability,
                "exact_official_parent": base_probability,
            }
            completion, local_city_rows, probability_error = _completion(
                validation["target"],
                variants,
                validation["city_indices"],
                training["city_names"],
                float(job["evaluation"]["probability_floor"]),
                seed=seed,
                candidate_id=candidate,
            )
            city_rows.extend(local_city_rows)
            full_nll = float(
                completion["full_graph_plus_pooled_backoff"][
                    "macro_city_nll_per_token"
                ]
            )
            no_graph_nll = float(
                completion["without_graph_pooled_backoff_retained"][
                    "macro_city_nll_per_token"
                ]
            )
            no_backoff_nll = float(
                completion["without_pooled_backoff_graph_retained"][
                    "macro_city_nll_per_token"
                ]
            )
            native_nll = float(
                completion["same_quota_native_decoder"][
                    "macro_city_nll_per_token"
                ]
            )
            parent_nll = float(
                completion["exact_official_parent"]["macro_city_nll_per_token"]
            )
            validation_quality, training_quality, support, selected_words = reporting_cache[quota]
            recall = float(
                np.mean(
                    [
                        np.intersect1d(selected_words[k], aligned_cores[k]).size
                        / aligned_cores.shape[1]
                        for k in range(aligned_cores.shape[0])
                    ]
                )
            )
            row = {
                "candidate_id": candidate,
                "seed": seed,
                "prototype_quota": quota,
                "calibration": str(calibration_name),
                "validation_npmi_full": float(validation_quality["npmi_at_10"]),
                "validation_npmi_without_graph": float(control_validation_quality["npmi_at_10"]),
                "validation_npmi_improvement_vs_without_graph": float(
                    validation_quality["npmi_at_10"]
                    - control_validation_quality["npmi_at_10"]
                ),
                "validation_cv_full": float(validation_quality["c_v_at_10"]),
                "validation_cv_without_graph": float(control_validation_quality["c_v_at_10"]),
                "validation_cv_improvement_vs_without_graph": float(
                    validation_quality["c_v_at_10"]
                    - control_validation_quality["c_v_at_10"]
                ),
                "validation_topic_diversity_full": float(
                    validation_quality["topic_diversity_at_10"]
                ),
                "validation_top_word_redundancy_full": float(
                    validation_quality["top_word_redundancy_at_10"]
                ),
                "training_reference_npmi_full": float(training_quality["npmi_at_10"]),
                "training_reference_cv_full": float(training_quality["c_v_at_10"]),
                "full_macro_city_nll": full_nll,
                "without_graph_macro_city_nll": no_graph_nll,
                "without_backoff_macro_city_nll": no_backoff_nll,
                "native_same_quota_macro_city_nll": native_nll,
                "official_parent_macro_city_nll": parent_nll,
                "graph_nll_reduction_vs_without_graph": no_graph_nll - full_nll,
                "pooled_nll_reduction_vs_without_backoff": no_backoff_nll - full_nll,
                "calibration_nll_reduction_vs_native": native_nll - full_nll,
                "full_nll_reduction_vs_official_parent": parent_nll - full_nll,
                "prototype_core_recall": recall,
                "expected_prototype_core_recall": quota / aligned_cores.shape[1],
                "maximum_probability_sum_error": probability_error,
                "maximum_nonprototype_change": float(
                    projection.audit["maximum_nonprototype_change"]
                ),
                "projection_column_sum_error": float(
                    projection.audit["maximum_column_sum_error"]
                ),
                "projection_minimum_probability": float(
                    projection.audit["minimum_probability"]
                ),
                "projection_optimality_error": float(
                    projection.audit["maximum_projection_optimality_error"]
                ),
                "mean_vocabulary_column_total_variation": float(
                    projection.audit["mean_vocabulary_column_total_variation"]
                ),
                "calibration_unchanged_identity_error": float(
                    calibration_audit["maximum_unchanged_identity_error"]
                ),
                "calibration_mean_error": float(
                    calibration_audit["maximum_changed_mean_error"]
                ),
                "calibration_variance_error": float(
                    calibration_audit[
                        "maximum_changed_regularized_variance_error"
                    ]
                ),
                "calibration_maximum_scale": float(
                    calibration_audit["maximum_changed_scale"]
                ),
                "parent_probability_reproduction_error": parent_probability_error,
                "parent_state_change": 0,
                **support,
            }
            if quota == 10 and calibration_name == "native_decoder":
                row["failed_step6_q10_pooled_nll_reproduction_error"] = abs(
                    full_nll - float(old_metric["matched_pooling_macro_city_nll"])
                )
                row["failed_step6_parent_nll_reproduction_error"] = abs(
                    parent_nll - float(old_metric["official_parent_macro_city_nll"])
                )
                old_effective = payload["effective_affinity"].detach().cpu().numpy()
                row["failed_step6_q10_affinity_reproduction_error"] = float(
                    np.max(
                        np.abs(
                            projection.affinity.astype(np.float32) - old_effective
                        )
                    )
                )
            else:
                row["failed_step6_q10_pooled_nll_reproduction_error"] = 0.0
                row["failed_step6_parent_nll_reproduction_error"] = 0.0
                row["failed_step6_q10_affinity_reproduction_error"] = 0.0
            rows.append(row)
            top_words[candidate] = selected_words
            print(
                f"[V6.1 Step 6R2 worker] seed={seed} candidate={candidate} DONE | "
                f"dNPMI={row['validation_npmi_improvement_vs_without_graph']:+.6f} "
                f"dC_v={row['validation_cv_improvement_vs_without_graph']:+.6f} "
                f"full-vs-parent={row['full_nll_reduction_vs_official_parent']:+.6f} "
                f"calibration={row['calibration_nll_reduction_vs_native']:+.6f}",
                flush=True,
            )
    parent_after = _state_hash(parent.state_dict())
    if parent_after != parent_before:
        raise RuntimeError("recovery evaluation changed a frozen parent")
    for row in rows:
        row["parent_state_change"] = int(parent_after != parent_before)
    control_words = np.asarray(
        control_validation_quality["top_word_indices"], dtype=np.int64
    )
    del parent
    torch.cuda.empty_cache()
    return rows, city_rows, top_words, control_words, calibration_receipts


def run_worker(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V6.1 Step 6R2 requires the unchanged Step-3 CUDA runtime")
    old_job = json.loads(
        Path(job["lineage_artifacts"]["failed_step6_worker_job"]["path"])
        .read_text(encoding="utf-8-sig")
    )
    for section in ("training_inputs", "validation_inputs"):
        for role, item in old_job[section].items():
            source = Path(item["path"])
            if not source.is_file() or sha256_file(source) != str(item["sha256"]):
                raise ValueError(f"frozen Step-6 input changed: {section}/{role}")
    training = _load_training(old_job)
    validation = _load_validation(old_job)
    context = construct_training_only_context(
        training["counts"],
        training["word_embeddings"],
        clusters=int(job["global_context"]["clusters"]),
        n_init=int(job["global_context"]["n_init"]),
        seed=int(job["global_context"]["seed"]),
    )
    graph, graph_receipt = positive_npmi_graph(
        training["counts"],
        minimum_document_frequency=int(
            job["lexical_graph"]["minimum_document_frequency"]
        ),
        minimum_joint_documents=int(
            job["lexical_graph"]["minimum_joint_documents"]
        ),
        reliability_power=float(job["lexical_graph"]["reliability_power"]),
    )
    if graph_receipt["logical_sha256"] != job["expected_training_graph_sha256"]:
        raise ValueError("the Step-6 training graph does not reproduce")
    validation_input, assignment_receipt = context.input_from_observed_counts(
        validation["observed"]
    )
    validation_reuse = {
        "schema_version": 1,
        "opened_at_utc": _utc_now(),
        "status": "SPENT_DEVELOPMENT",
        "semantic_access_number": 2,
        "purpose": "transparent recovery candidate selection after failed confirmation",
        "confirmation_claim_allowed": False,
        "test_available_to_worker": False,
        "baseline_or_sota_available_to_worker": False,
        "validation_inputs": old_job["validation_inputs"],
    }
    _write_json(output / "validation_reuse_receipt.json", validation_reuse)
    _write_json(output / "validation_assignment_receipt.json", assignment_receipt)
    _write_json(output / "training_context_receipt.json", context.audit)
    _write_json(output / "training_graph_receipt.json", graph_receipt)
    _write_json(
        output / "retired_city_mechanism_receipt.json",
        {
            "schema_version": 1,
            "status": "RETIRED",
            "mechanism": "hierarchical_city_specific_lexical_backoff",
            "reason": "lost to the matched pooled-backoff control in all three frozen Step-6 validation seeds",
            "replacement": "training-only pooled lexical backoff with no city-conditioned parameter",
            "headline_city_ablation_allowed": False,
            "city_grouping_retained_only_for_macro_evaluation_and_downstream_salience_analysis": True,
        },
    )

    old_metrics = pd.read_csv(
        Path(job["lineage_artifacts"]["failed_step6_validation_metrics"]["path"]),
        encoding="utf-8-sig",
    ).set_index("seed")
    NewMethod, loaded_path = _load_official_newmethod(Path(job["official_source"]))
    device = torch.device("cuda")
    all_rows: list[dict[str, Any]] = []
    all_city_rows: list[dict[str, Any]] = []
    calibration_receipts: list[dict[str, Any]] = []
    words_by_candidate: dict[str, list[np.ndarray]] = {
        _candidate_id(q, c): []
        for q in map(int, job["candidates"]["prototype_quotas"])
        for c in job["candidates"]["calibrations"]
    }
    control_words: list[np.ndarray] = []
    for checkpoint in job["checkpoint_manifest"]:
        seed = int(checkpoint["seed"])
        if seed not in old_metrics.index:
            raise ValueError("Step-6 metrics and checkpoint seeds differ")
        rows, city_rows, seed_words, seed_control, receipts = _evaluate_seed(
            job,
            checkpoint,
            training,
            validation,
            context.train_input,
            validation_input,
            graph,
            old_metrics.loc[seed].to_dict(),
            NewMethod,
            device=device,
            torch=torch,
        )
        all_rows.extend(rows)
        all_city_rows.extend(city_rows)
        calibration_receipts.extend(receipts)
        for candidate, words in seed_words.items():
            words_by_candidate[candidate].append(words)
        control_words.append(seed_control)

    control_stability = matched_topic_stability(control_words)
    summaries: list[dict[str, Any]] = []
    for candidate in sorted(words_by_candidate):
        candidate_rows = [
            row for row in all_rows if row["candidate_id"] == candidate
        ]
        summary = summarize_recovery_candidate(
            candidate_rows,
            gates=job["gates"],
            topic_stability=matched_topic_stability(words_by_candidate[candidate]),
            graph_removed_topic_stability=control_stability,
        )
        summaries.append(summary)
        print(
            f"[V6.1 Step 6R2 worker] candidate audit | {candidate} "
            f"passed={summary['passed_gate_count']}/{summary['total_gate_count']} "
            f"eligible={summary['eligible']} "
            f"failed={'|'.join(summary['failed_gates']) or 'none'}",
            flush=True,
        )
    selected = select_recovery_candidate(summaries)
    maximum_reproduction_error = float(
        max(
            max(float(row[key]) for row in all_rows)
            for key in (
                "failed_step6_q10_pooled_nll_reproduction_error",
                "failed_step6_parent_nll_reproduction_error",
                "failed_step6_q10_affinity_reproduction_error",
                "parent_probability_reproduction_error",
            )
        )
    )
    _write_csv(output / "candidate_seed_metrics.csv", all_rows)
    _write_csv(output / "candidate_city_metrics.csv", all_city_rows)
    _write_csv(output / "candidate_summary.csv", summaries)
    _write_json(output / "candidate_summaries.json", summaries)
    _write_json(output / "calibration_receipts.json", calibration_receipts)
    if selected is not None:
        selected_configuration = {
            "schema_version": 1,
            "model_family": "V6.1 evidence-first recovery",
            "prototype_quota": int(selected["prototype_quota"]),
            "calibration": str(selected["calibration"]),
            "pooled_mixture_weight": float(
                job["model_revision"]["pooled_mixture_weight"]
            ),
            "city_specific_backoff": "RETIRED",
            "selection_surface": "spent validation development only",
            "selection_score": float(
                selected["selection_score_minimum_normalized_margin"]
            ),
            "test_status": "UNOPENED",
        }
        selected_configuration["configuration_fingerprint"] = sha256_object(
            selected_configuration
        )
        _write_json(output / "selected_configuration.json", selected_configuration)
    else:
        selected_configuration = None
    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "status": "PASS" if selected is not None else "FAIL",
        "official_class_loaded_from": str(loaded_path),
        "parent_neural_refits": 0,
        "frozen_parent_checkpoints_reused": len(job["checkpoint_manifest"]),
        "candidate_configurations_evaluated": len(summaries),
        "candidate_seed_evaluations": len(all_rows),
        "validation_status": "SPENT_DEVELOPMENT",
        "validation_documents_used": int(len(validation["rows"])),
        "validation_target_tokens": int(validation["target_tokens"]),
        "hyperparameter_selection_performed": True,
        "city_specific_backoff_retired": True,
        "selected_candidate": selected,
        "selected_configuration": selected_configuration,
        "maximum_lineage_reproduction_error": maximum_reproduction_error,
        "test_documents_used": 0,
        "labels_used": 0,
        "baseline_or_sota_outputs_used": 0,
    }
    _write_json(output / "worker_result.json", result)
    print(
        f"[V6.1 Step 6R2 worker] {result['status']} | "
        f"selected={selected['candidate_id'] if selected else 'none'} "
        f"lineage-error={maximum_reproduction_error:.3e}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = _load_job(args.job.resolve())
    output = Path(job["worker_output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch = _seed_everything(int(job["checkpoint_manifest"][0]["seed"]))
    _write_json(output / "runtime_receipt.json", _runtime_receipt(torch))
    run_worker(job, output, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
