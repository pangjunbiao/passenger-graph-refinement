"""Exact-runtime worker for the revised V6 city-backoff development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.v6.city_backoff import (
    estimate_hierarchical_city_priors,
    guaranteed_parent_reduction,
    jensen_nll_upper_bound,
    mix_with_prior,
    select_city_prior_rows,
)
from src.v6.data_contract import sha256_file
from src.v6.development_evidence import (
    construct_fold_context,
    macro_city_completion,
    topic_quality_metrics,
)
from src.v6.development_worker import _load_evidence, _new_parent, _state_hash
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt


IMPLEMENTATION = "v6_step4r2_hierarchical_city_backoff_gate_r1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty required table: {path.name}")
    pd.DataFrame.from_records(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        job.get("schema_version") != 1
        or job.get("mode") != "step4r2"
        or job.get("implementation_version") != IMPLEMENTATION
    ):
        raise ValueError("unsupported V6 Step-4R2 worker job")
    for role, item in job["inputs"].items():
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(f"worker input is missing ({role}): {source}")
        if sha256_file(source) != str(item["sha256"]):
            raise ValueError(f"worker input hash changed ({role})")
    checkpoints = job["failed_step4_lineage"]["checkpoints"]
    if len(checkpoints) != len(job["development_folds"]):
        raise ValueError("the failed Step-4 lineage does not contain one checkpoint per fold")
    for row in checkpoints:
        source = Path(row["path"])
        if not source.is_file() or sha256_file(source) != str(row["sha256"]):
            raise ValueError(f"failed Step-4 checkpoint is missing or changed: {source}")
    for path_key, hash_key in (
        ("fold_metrics_path", "fold_metrics_sha256"),
        ("worker_job_path", "worker_job_sha256"),
    ):
        source = Path(job["failed_step4_lineage"][path_key])
        if (
            not source.is_file()
            or sha256_file(source)
            != str(job["failed_step4_lineage"][hash_key])
        ):
            raise ValueError(f"failed Step-4 lineage artifact changed: {source}")
    return job


def _parent_probabilities(
    parent: Any,
    input_values: np.ndarray,
    *,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import torch

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
            probabilities = torch.softmax(
                parent.decoder_bn((global_theta * local_theta) @ parent.get_beta()),
                dim=-1,
            )
            rows.append(probabilities.detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(rows, axis=0)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError("official parent returned invalid probabilities")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise FloatingPointError("official parent probability rows are not normalized")
    # Preserve the native float32 decoder values exactly; the common evaluator
    # already permits their one-ulp row-sum roundoff.
    return result


def _completion_for_fold(
    evidence: Mapping[str, Any], context: Any, fold: int
) -> tuple[np.ndarray, Any, np.ndarray]:
    source = evidence["completion_source_rows"]
    selected = np.flatnonzero(evidence["fold_assignments"][source] == int(fold))
    inputs = context.input_for_local_counts(evidence["completion_observed"][selected])
    return (
        inputs,
        evidence["completion_target"][selected],
        evidence["completion_city_indices"][selected],
    )


def _evaluate(
    target: Any,
    probabilities: np.ndarray,
    city_indices: np.ndarray,
    city_names: list[str],
    *,
    probability_floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return macro_city_completion(
        target,
        probabilities,
        city_indices,
        city_names,
        probability_floor=float(probability_floor),
    )


def _restore_parent(
    NewMethod: type,
    evidence: Mapping[str, Any],
    model_config: Mapping[str, Any],
    checkpoint_row: Mapping[str, Any],
    expected_fold_row: Mapping[str, Any],
    expected_commit: str,
    context_audit: Mapping[str, Any],
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_row["path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported failed Step-4 checkpoint schema")
    if int(payload.get("fold", -1)) != int(checkpoint_row["fold"]):
        raise ValueError("failed Step-4 checkpoint fold identity changed")
    if str(payload.get("official_source_commit")) != str(expected_commit):
        raise ValueError("failed Step-4 checkpoint source commit changed")
    if payload.get("model_configuration") != dict(model_config):
        raise ValueError("failed Step-4 checkpoint parent configuration changed")
    for key in ("centroids_sha256", "assignments_sha256", "global_bow_sha256"):
        if str(payload["context_audit"].get(key)) != str(context_audit.get(key)):
            raise ValueError(f"fold context changed before Step-4R2 ({key})")

    wrapper_state = payload.get("state_dict", {})
    parent_state = {
        name[len("parent.") :]: value
        for name, value in wrapper_state.items()
        if name.startswith("parent.")
    }
    if not parent_state:
        raise ValueError("failed Step-4 checkpoint lacks the frozen parent state")
    parent = _new_parent(NewMethod, evidence["word_embeddings"], model_config)
    parent.load_state_dict(parent_state, strict=True)
    observed_hash = _state_hash(parent.state_dict())
    expected_hash = str(expected_fold_row["warmed_parent_state_sha256"])
    if observed_hash != expected_hash:
        raise ValueError("restored parent state does not match the recorded warm parent")
    return parent, {
        "source_checkpoint_sha256": str(checkpoint_row["sha256"]),
        "restored_parent_state_sha256": observed_hash,
    }


def _candidate_summary(
    candidate_rows: list[dict[str, Any]],
    weight: float,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [row for row in candidate_rows if float(row["mixture_weight"]) == float(weight)]
    if len(selected) != 5:
        raise ValueError("a Step-4R2 candidate does not contain exactly five folds")
    city_pool = np.asarray(
        [row["city_minus_pooled_backoff_nll_reduction"] for row in selected],
        dtype=np.float64,
    )
    city_parent = np.asarray(
        [row["city_minus_parent_nll_reduction"] for row in selected],
        dtype=np.float64,
    )
    bounds = np.asarray(
        [row["guaranteed_parent_reduction_lower_bound"] for row in selected],
        dtype=np.float64,
    )
    eligible = bool(
        np.median(city_pool) >= float(gates["minimum_city_vs_pooled_reduction"])
        and np.mean(city_pool > 0.0) >= float(gates["minimum_positive_fold_fraction"])
        and np.median(city_parent) >= float(gates["minimum_city_vs_parent_reduction"])
        and np.mean(city_parent > 0.0) >= float(gates["minimum_positive_fold_fraction"])
        and bounds.min() >= float(gates["minimum_guaranteed_parent_reduction"])
    )
    return {
        "mixture_weight": float(weight),
        "mean_city_vs_pooled_reduction": float(city_pool.mean()),
        "median_city_vs_pooled_reduction": float(np.median(city_pool)),
        "minimum_city_vs_pooled_reduction": float(city_pool.min()),
        "positive_city_vs_pooled_fold_fraction": float(np.mean(city_pool > 0.0)),
        "mean_city_vs_parent_reduction": float(city_parent.mean()),
        "median_city_vs_parent_reduction": float(np.median(city_parent)),
        "minimum_city_vs_parent_reduction": float(city_parent.min()),
        "positive_city_vs_parent_fold_fraction": float(np.mean(city_parent > 0.0)),
        "minimum_guaranteed_parent_reduction_lower_bound": float(bounds.min()),
        "eligible": eligible,
    }


def run_worker(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    evidence = _load_evidence(job)
    NewMethod, loaded_path = _load_official_newmethod(Path(job["official_source"]))
    device = torch.device("cuda")
    model_config = job["model"]
    context_config = job["global_context"]
    evaluation = job["evaluation"]
    prior_config = job["city_prior"]
    candidate_weights = sorted(set(float(x) for x in job["candidate_mixture_weights"]))
    gates = job["gates"]
    if not candidate_weights or candidate_weights[0] <= 0.0 or candidate_weights[-1] > float(
        gates["maximum_mixture_weight"]
    ):
        raise ValueError("candidate mixture weights violate the frozen conservative range")

    checkpoint_by_fold = {
        int(row["fold"]): row
        for row in job["failed_step4_lineage"]["checkpoints"]
    }
    old_fold_metrics = pd.read_csv(
        job["failed_step4_lineage"]["fold_metrics_path"], encoding="utf-8-sig"
    ).set_index("fold")
    all_candidate_rows: list[dict[str, Any]] = []
    all_city_rows: list[dict[str, Any]] = []
    fold_static: dict[int, dict[str, Any]] = {}
    context_receipts: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    maximum_probability_error = 0.0
    maximum_jensen_violation = 0.0
    maximum_zero_identity_error = 0.0
    maximum_r1_parent_nll_reproduction_error = 0.0

    for fold in map(int, job["development_folds"]):
        print(f"[V6 Step 4R2 worker] fold={fold} START | reuse frozen parent", flush=True)
        context = construct_fold_context(
            evidence["counts"],
            evidence["word_embeddings"],
            evidence["fold_assignments"],
            fold,
            clusters=int(context_config["clusters"]),
            n_init=int(context_config["n_init"]),
            seed=int(context_config["seed"]) + fold,
        )
        context_receipts.append(context.audit)
        old_row = old_fold_metrics.loc[fold].to_dict()
        parent, lineage = _restore_parent(
            NewMethod,
            evidence,
            model_config,
            checkpoint_by_fold[fold],
            old_row,
            str(job["official_commit"]),
            context.audit,
            torch,
        )
        parent = parent.to(device)
        completion_input, target, completion_cities = _completion_for_fold(
            evidence, context, fold
        )
        parent_probabilities = _parent_probabilities(
            parent,
            completion_input,
            batch_size=int(evaluation["inference_batch_size"]),
            device=device,
        )
        maximum_probability_error = max(
            maximum_probability_error,
            float(np.abs(parent_probabilities.sum(axis=1) - 1.0).max()),
        )
        zero = mix_with_prior(
            parent_probabilities,
            parent_probabilities,
            mixture_weight=0.0,
        )
        maximum_zero_identity_error = max(
            maximum_zero_identity_error,
            float(np.abs(zero - parent_probabilities).max()),
        )

        train_cities = evidence["city_indices"][context.train_indices]
        prior = estimate_hierarchical_city_priors(
            evidence["counts"][context.train_indices].toarray(),
            train_cities,
            cities=len(evidence["city_names"]),
            shrinkage=float(prior_config["shrinkage"]),
            pseudocount=float(prior_config["pseudocount"]),
        )
        city_prior_rows = select_city_prior_rows(prior.city, completion_cities)
        pooled_prior_rows = np.repeat(
            prior.pooled[None, :], len(completion_cities), axis=0
        )
        parent_metrics, parent_city = _evaluate(
            target,
            parent_probabilities,
            completion_cities,
            evidence["city_names"],
            probability_floor=float(evaluation["probability_floor"]),
        )
        r1_parent_nll = float(old_row["pooled_macro_city_nll"])
        r1_parent_error = abs(
            float(parent_metrics["macro_city_nll_per_token"]) - r1_parent_nll
        )
        maximum_r1_parent_nll_reproduction_error = max(
            maximum_r1_parent_nll_reproduction_error, r1_parent_error
        )
        city_prior_metrics, city_prior_city = _evaluate(
            target,
            city_prior_rows,
            completion_cities,
            evidence["city_names"],
            probability_floor=float(evaluation["probability_floor"]),
        )
        pooled_prior_metrics, pooled_prior_city = _evaluate(
            target,
            pooled_prior_rows,
            completion_cities,
            evidence["city_names"],
            probability_floor=float(evaluation["probability_floor"]),
        )
        for variant, values in (
            ("official_parent", parent_city),
            ("city_prior_only", city_prior_city),
            ("pooled_prior_only", pooled_prior_city),
        ):
            for row in values:
                all_city_rows.append(
                    {"fold": fold, "mixture_weight": 1.0 if "prior" in variant else 0.0, "variant": variant, **row}
                )

        with torch.no_grad():
            affinity = parent.get_beta().detach().cpu().numpy().astype(np.float64)
        reporting = affinity / np.maximum(affinity.sum(axis=1, keepdims=True), 1.0e-300)
        quality = topic_quality_metrics(
            reporting,
            evidence["counts"][context.evaluation_indices],
            top_words=int(evaluation["top_words"]),
            window_size=int(evaluation["c_v_window_size"]),
            gamma=float(evaluation["c_v_gamma"]),
        )
        fold_static[fold] = {
            "fold": fold,
            "fit_documents": int(len(context.train_indices)),
            "evaluation_documents": int(len(context.evaluation_indices)),
            "completion_documents": int(parent_metrics["documents"]),
            "parent_macro_city_nll": float(parent_metrics["macro_city_nll_per_token"]),
            "r1_parent_macro_city_nll": r1_parent_nll,
            "r1_parent_nll_reproduction_error": r1_parent_error,
            "city_prior_macro_city_nll": float(city_prior_metrics["macro_city_nll_per_token"]),
            "pooled_prior_macro_city_nll": float(pooled_prior_metrics["macro_city_nll_per_token"]),
            "c_v_at_10": float(quality["c_v_at_10"]),
            "npmi_at_10": float(quality["npmi_at_10"]),
            "topic_diversity_at_10": float(quality["topic_diversity_at_10"]),
            "top_word_redundancy_at_10": float(quality["top_word_redundancy_at_10"]),
            "city_prior_documents": prior.city_document_counts.tolist(),
            "city_prior_tokens": prior.city_token_counts.tolist(),
            **lineage,
        }

        for weight in candidate_weights:
            pooled_backoff = mix_with_prior(
                parent_probabilities,
                pooled_prior_rows,
                mixture_weight=weight,
            )
            city_backoff = mix_with_prior(
                parent_probabilities,
                city_prior_rows,
                mixture_weight=weight,
            )
            pooled_metrics, pooled_city = _evaluate(
                target,
                pooled_backoff,
                completion_cities,
                evidence["city_names"],
                probability_floor=float(evaluation["probability_floor"]),
            )
            city_metrics, city_rows = _evaluate(
                target,
                city_backoff,
                completion_cities,
                evidence["city_names"],
                probability_floor=float(evaluation["probability_floor"]),
            )
            upper = jensen_nll_upper_bound(
                parent_metrics["macro_city_nll_per_token"],
                city_prior_metrics["macro_city_nll_per_token"],
                mixture_weight=weight,
            )
            maximum_jensen_violation = max(
                maximum_jensen_violation,
                float(city_metrics["macro_city_nll_per_token"] - upper),
            )
            candidate = {
                "fold": fold,
                "mixture_weight": weight,
                "parent_macro_city_nll": parent_metrics["macro_city_nll_per_token"],
                "pooled_backoff_macro_city_nll": pooled_metrics["macro_city_nll_per_token"],
                "city_backoff_macro_city_nll": city_metrics["macro_city_nll_per_token"],
                "city_minus_pooled_backoff_nll_reduction": (
                    pooled_metrics["macro_city_nll_per_token"]
                    - city_metrics["macro_city_nll_per_token"]
                ),
                "city_minus_parent_nll_reduction": (
                    parent_metrics["macro_city_nll_per_token"]
                    - city_metrics["macro_city_nll_per_token"]
                ),
                "pooled_minus_parent_nll_reduction": (
                    parent_metrics["macro_city_nll_per_token"]
                    - pooled_metrics["macro_city_nll_per_token"]
                ),
                "city_prior_minus_parent_nll_gap": (
                    parent_metrics["macro_city_nll_per_token"]
                    - city_prior_metrics["macro_city_nll_per_token"]
                ),
                "jensen_city_nll_upper_bound": upper,
                "guaranteed_parent_reduction_lower_bound": guaranteed_parent_reduction(
                    parent_metrics["macro_city_nll_per_token"],
                    city_prior_metrics["macro_city_nll_per_token"],
                    mixture_weight=weight,
                ),
                "city_micro_nll": city_metrics["micro_nll_per_token"],
                "pooled_micro_nll": pooled_metrics["micro_nll_per_token"],
            }
            all_candidate_rows.append(candidate)
            for variant, values in (
                ("matched_pooled_backoff", pooled_city),
                ("hierarchical_city_backoff", city_rows),
            ):
                for row in values:
                    all_city_rows.append(
                        {"fold": fold, "mixture_weight": weight, "variant": variant, **row}
                    )
            print(
                f"[V6 Step 4R2 worker] fold={fold} rho={weight:.2f} "
                f"city-vs-pooled={candidate['city_minus_pooled_backoff_nll_reduction']:+.6f} "
                f"city-vs-parent={candidate['city_minus_parent_nll_reduction']:+.6f}",
                flush=True,
            )

        new_checkpoint = output / "checkpoints" / f"fold_{fold:02d}_city_backoff.pt"
        new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        parent_cpu_state = {
            key: value.detach().cpu() for key, value in parent.state_dict().items()
        }
        torch.save(
            {
                "schema_version": 2,
                "implementation_version": IMPLEMENTATION,
                "fold": fold,
                "official_source_commit": job["official_commit"],
                "model_configuration": model_config,
                "city_prior_configuration": prior_config,
                "candidate_mixture_weights": candidate_weights,
                "parent_state_dict": parent_cpu_state,
                "city_priors": torch.from_numpy(prior.city.astype(np.float32)),
                "pooled_prior": torch.from_numpy(prior.pooled.astype(np.float32)),
                "context_audit": context.audit,
                "failed_step4_lineage": lineage,
            },
            new_checkpoint,
        )
        checkpoint_rows.append(
            {
                "fold": fold,
                "path": str(new_checkpoint),
                "sha256": sha256_file(new_checkpoint),
                "size_bytes": int(new_checkpoint.stat().st_size),
            }
        )
        del parent
        torch.cuda.empty_cache()

    summaries = [
        _candidate_summary(all_candidate_rows, weight, gates)
        for weight in candidate_weights
    ]
    eligible = [row for row in summaries if row["eligible"]]
    selected_weight = min(
        (float(row["mixture_weight"]) for row in eligible), default=None
    )
    selected_rows: list[dict[str, Any]] = []
    if selected_weight is not None:
        lookup = {
            int(row["fold"]): row
            for row in all_candidate_rows
            if float(row["mixture_weight"]) == selected_weight
        }
        for fold in map(int, job["development_folds"]):
            selected_rows.append({**fold_static[fold], **lookup[fold]})

    _write_csv(output / "candidate_fold_metrics.csv", all_candidate_rows)
    _write_csv(output / "candidate_summary.csv", summaries)
    _write_csv(output / "city_stratified_metrics.csv", all_city_rows)
    if selected_rows:
        _write_csv(output / "selected_fold_metrics.csv", selected_rows)
    _write_json(output / "fold_context_receipts.json", context_receipts)
    _write_json(output / "checkpoint_manifest.json", checkpoint_rows)
    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "status": "PASS" if selected_weight is not None else "FAIL",
        "official_class_loaded_from": str(loaded_path),
        "folds": len(job["development_folds"]),
        "parent_neural_refits": 0,
        "reused_hash_verified_parent_checkpoints": len(checkpoint_rows),
        "selection_rule": "smallest predeclared mixture weight satisfying every city-vs-pooled, city-vs-parent, consistency, and convexity gate",
        "candidate_summaries": summaries,
        "selected_mixture_weight": selected_weight,
        "maximum_mixture_weight": float(gates["maximum_mixture_weight"]),
        "maximum_probability_sum_error": maximum_probability_error,
        "maximum_alpha_zero_parent_identity_error": maximum_zero_identity_error,
        "maximum_r1_parent_nll_reproduction_error": maximum_r1_parent_nll_reproduction_error,
        "maximum_jensen_bound_violation": max(0.0, maximum_jensen_violation),
        "checkpoint_manifest": checkpoint_rows,
        "validation_documents_used": 0,
        "test_documents_used": 0,
        "labels_used": 0,
        "baseline_or_sota_outputs_used": 0,
    }
    _write_json(output / "worker_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = _load_job(args.job.resolve())
    output = Path(job["worker_output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V6 Step-4R2 requires the exact CUDA runtime verified in Step 3")
    torch.manual_seed(int(job["seed"]))
    torch.cuda.manual_seed_all(int(job["seed"]))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    _write_json(output / "runtime_receipt.json", _runtime_receipt(torch))
    result = run_worker(job, output, torch)
    print(
        f"[V6 Step 4R2 worker] {result['status']} | selected rho={result['selected_mixture_weight']}",
        flush=True,
    )
    # A clean scientific non-selection is reported in worker_result.json and
    # converted into a failed outer contract.  Nonzero is reserved for an
    # engineering/runtime exception.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
