"""Exact-runtime worker for the V6.1 final development refit.

The worker fits ten predeclared parent seeds on the 620 original-training plus
75 spent-development documents.  It then deterministically freezes the
selected quota-10 graph transport, training-moment calibration, pooled prior,
and every paired ablation identity.  It has no test or comparator input.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.city_backoff import estimate_pooled_unigram
from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_evidence import (
    matched_topic_stability,
    positive_npmi_graph,
    topic_quality_metrics,
)
from src.v6.development_worker import _gradient_norm, _new_parent, _state_hash
from src.v6.final_refit import (
    combine_development_partitions,
    final_variant_registry,
    validate_final_diagnostics,
)
from src.v6.graph_prototype_adapter import (
    align_prototype_cores_to_topics,
    build_graph_prototype_bank,
    minimum_distortion_graph_projection,
)
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt
from src.v6.step5r2_worker import _array_hash, _seed_everything, _top_word_support
from src.v6.step6r2_worker import _latent_product, _raw_logits, _reporting_affinity
from src.v6.transport_calibration import fit_changed_column_moment_calibration
from src.v6.validation_context import construct_training_only_context

IMPLEMENTATION = "v6_1_step7_final_development_refit_freeze_r1"


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
    pd.DataFrame.from_records(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        job.get("schema_version") != 1
        or job.get("mode") != "step7"
        or job.get("implementation_version") != IMPLEMENTATION
    ):
        raise ValueError("unsupported V6.1 Step-7 worker job")
    for role, item in job["development_inputs"].items():
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(f"final-refit input is missing ({role}): {source}")
        if sha256_file(source) != str(item["sha256"]):
            raise ValueError(f"final-refit input hash changed ({role})")
    for role, item in job["lineage_artifacts"].items():
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != str(item["sha256"]):
            raise ValueError(f"Step-6R2 lineage changed ({role})")
    if any(
        bool(job["data_policy"].get(key, True))
        for key in (
            "test_available_to_worker",
            "labels_available_to_worker",
            "comparators_available_to_worker",
        )
    ):
        raise ValueError("Step 7 must receive no test, labels, or comparators")
    return job


def _load_development(job: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        name: Path(item["path"]) for name, item in job["development_inputs"].items()
    }
    train_counts = sparse.load_npz(paths["train_counts"]).tocsr().astype(np.int64)
    spent_counts = sparse.load_npz(paths["spent_counts"]).tocsr().astype(np.int64)
    train_rows = pd.read_csv(
        paths["train_rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    spent_rows = pd.read_csv(
        paths["spent_rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    with np.load(paths["word_embeddings"], allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    counts, rows, merge_receipt = combine_development_partitions(
        train_counts, train_rows, spent_counts, spent_rows
    )
    expected = job["final_fit"]
    if counts.shape != (
        int(expected["documents"]),
        int(expected["vocabulary"]),
    ):
        raise ValueError("the final development count shape changed")
    if int(counts.sum()) != int(expected["tokens"]):
        raise ValueError("the final development token mass changed")
    if word_embeddings.shape != (
        int(expected["vocabulary"]),
        int(job["model"]["frozen_word_feature_width"]),
    ):
        raise ValueError("the frozen vocabulary embeddings changed shape")
    expected_cities = set(map(str, job["cities"]))
    if set(rows["city"].astype(str)) != expected_cities:
        raise ValueError("the final-fit city inventory changed")
    return {
        "counts": counts,
        "rows": rows,
        "word_embeddings": word_embeddings,
        "merge_receipt": merge_receipt,
    }


def _window_mean(trace: list[dict[str, Any]], key: str, *, first: bool) -> float:
    if not trace:
        raise ValueError("cannot summarize an empty training trace")
    width = min(10, max(1, len(trace) // 4))
    selected = trace[:width] if first else trace[-width:]
    return float(np.mean([float(row[key]) for row in selected]))


def _train_parent(
    model: Any,
    train_input: np.ndarray,
    configuration: Mapping[str, Any],
    *,
    seed: int,
    device: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    from torch.utils.data import DataLoader

    tensor = torch.from_numpy(np.asarray(train_input, dtype=np.float32))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(configuration["parent_learning_rate"])
    )
    epochs = int(configuration["parent_epochs"])
    trace: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 100_003 * int(epoch)
        )
        loader = DataLoader(
            tensor,
            batch_size=int(configuration["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        sums: dict[str, float] = {}
        documents = 0
        maximum_gradient = 0.0
        for batch_cpu in loader:
            batch = batch_cpu.to(device)
            result = model(batch)
            scalar_values = {
                key: value
                for key, value in result.items()
                if hasattr(value, "detach") and getattr(value, "ndim", 1) == 0
            }
            if not scalar_values or not all(
                bool(torch.isfinite(value).all()) for value in scalar_values.values()
            ):
                raise FloatingPointError("non-finite official parent objective")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            maximum_gradient = max(maximum_gradient, _gradient_norm(model.parameters()))
            optimizer.step()
            count = int(batch.shape[0])
            documents += count
            for key, value in scalar_values.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count
        row: dict[str, Any] = {
            "seed": int(seed),
            "epoch": int(epoch),
            "documents": int(documents),
            "maximum_gradient_norm": float(maximum_gradient),
        }
        row.update({key: value / documents for key, value in sums.items()})
        trace.append(row)
        if epoch in {1, epochs} or epoch % 20 == 0:
            print(
                f"[V6.1 Step 7 worker] seed={seed} epoch={epoch}/{epochs} "
                f"loss={row['loss']:.6f}",
                flush=True,
            )
    return trace


def run_worker(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V6.1 Step 7 requires the unchanged Step-3 CUDA runtime")
    NewMethod, loaded_path = _load_official_newmethod(Path(job["official_source"]))
    device = torch.device("cuda")
    development = _load_development(job)
    counts = development["counts"]
    word_embeddings = development["word_embeddings"]
    context = construct_training_only_context(
        counts,
        word_embeddings,
        clusters=int(job["global_context"]["clusters"]),
        n_init=int(job["global_context"]["n_init"]),
        seed=int(job["global_context"]["seed"]),
        fit_partition_label="all_695_train_plus_spent_development_rows",
        original_training_documents=int(
            job["final_fit"]["original_training_documents"]
        ),
        spent_development_documents=int(
            job["final_fit"]["spent_development_documents"]
        ),
    )
    pooled_prior = estimate_pooled_unigram(
        counts.toarray(),
        pseudocount=float(job["pooled_backoff"]["pseudocount"]),
    )
    graph, graph_receipt = positive_npmi_graph(
        counts,
        minimum_document_frequency=int(
            job["lexical_graph"]["minimum_document_frequency"]
        ),
        minimum_joint_documents=int(job["lexical_graph"]["minimum_joint_documents"]),
        reliability_power=float(job["lexical_graph"]["reliability_power"]),
    )
    graph_receipt["fit_partition"] = "all_695_train_plus_spent_development_rows"
    if int(graph_receipt["active_words"]) < int(
        job["lexical_graph"]["minimum_active_words"]
    ):
        raise ValueError("the final-fit lexical graph lacks required support")

    variant_registry = final_variant_registry(
        prototype_quota=int(job["selected_configuration"]["prototype_quota"]),
        pooled_mixture_weight=float(
            job["selected_configuration"]["pooled_mixture_weight"]
        ),
        calibration=str(job["selected_configuration"]["calibration"]),
    )
    variant_manifest = {
        "schema_version": 1,
        "construction": "paired_deterministic_postfit_interventions_on_common_seed_parent",
        "why_parent_is_shared": "all V6.1 modules are deterministic post-parent-fit transforms and do not alter the parent objective",
        "variants": variant_registry,
        "test_status": "UNOPENED",
    }
    variant_manifest["variant_manifest_identity"] = sha256_object(variant_manifest)
    _write_json(output / "final_variant_manifest.json", variant_manifest)
    _write_json(output / "development_merge_receipt.json", development["merge_receipt"])
    _write_json(output / "final_context_receipt.json", context.audit)
    _write_json(output / "final_graph_receipt.json", graph_receipt)
    _write_json(
        output / "pooled_prior_receipt.json",
        {
            "schema_version": 1,
            "estimator": "additive_smoothed_pooled_unigram",
            "fit_partition": "all_695_train_plus_spent_development_rows",
            "pseudocount": float(job["pooled_backoff"]["pseudocount"]),
            "mixture_weight": float(job["pooled_backoff"]["mixture_weight"]),
            "probability_sum_error": float(abs(pooled_prior.sum() - 1.0)),
            "minimum_probability": float(pooled_prior.min()),
            "logical_sha256": _array_hash(pooled_prior),
            "city_conditioned_parameters": 0,
        },
    )

    trace_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    projection_receipts: list[dict[str, Any]] = []
    full_top_words: list[np.ndarray] = []
    parent_top_words: list[np.ndarray] = []
    seeds = list(map(int, job["training"]["seeds"]))
    for seed in seeds:
        print(
            f"[V6.1 Step 7 worker] seed={seed} FINAL REFIT START | documents={counts.shape[0]}",
            flush=True,
        )
        _seed_everything(seed)
        parent = _new_parent(NewMethod, word_embeddings, job["model"]).to(device)
        initial_hash = _state_hash(parent.state_dict())
        trace = _train_parent(
            parent,
            context.train_input,
            job["training"],
            seed=seed,
            device=device,
            torch=torch,
        )
        trace_rows.extend(trace)
        final_hash = _state_hash(parent.state_dict())
        if final_hash == initial_hash:
            raise RuntimeError("official parent did not change during final refit")
        early_loss = _window_mean(trace, "loss", first=True)
        late_loss = _window_mean(trace, "loss", first=False)
        parent.eval()
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            base_affinity = parent.get_beta().detach().cpu().numpy().astype(np.float64)

        prototype_seed = seed + int(job["graph_projection"]["prototype_seed_offset"])
        bank = build_graph_prototype_bank(
            graph,
            topics=int(job["model"]["topics"]),
            top_k=int(job["evaluation"]["top_words"]),
            seed=prototype_seed,
            centrality_weight=float(job["graph_projection"]["centrality_weight"]),
            minimum_cluster_size=int(job["graph_projection"]["minimum_cluster_size"]),
        )
        aligned_cores, alignment = align_prototype_cores_to_topics(
            base_affinity,
            bank.core_indices,
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        zero = minimum_distortion_graph_projection(
            base_affinity,
            aligned_cores,
            strength=0.0,
            prototype_quota=int(job["selected_configuration"]["prototype_quota"]),
            rank_margin=float(job["graph_projection"]["rank_margin"]),
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        zero_error = float(np.max(np.abs(zero.affinity - base_affinity)))
        projection = minimum_distortion_graph_projection(
            base_affinity,
            aligned_cores,
            strength=float(job["graph_projection"]["strength"]),
            prototype_quota=int(job["selected_configuration"]["prototype_quota"]),
            rank_margin=float(job["graph_projection"]["rank_margin"]),
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        latent = _latent_product(
            parent,
            context.train_input,
            batch_size=int(job["evaluation"]["inference_batch_size"]),
            device=device,
            torch=torch,
        )
        native_logits = _raw_logits(latent, base_affinity, device=device, torch=torch)
        transported_logits = _raw_logits(
            latent, projection.affinity, device=device, torch=torch
        )
        calibration = fit_changed_column_moment_calibration(
            native_logits,
            transported_logits,
            base_affinity,
            projection.affinity,
            variance_floor=float(job["calibration"]["variance_floor"]),
            change_tolerance=float(job["calibration"]["change_tolerance"]),
        )
        reporting = _reporting_affinity(projection.affinity)
        parent_reporting = _reporting_affinity(base_affinity)
        quality_args = {
            "top_words": int(job["evaluation"]["top_words"]),
            "window_size": int(job["evaluation"]["c_v_window_size"]),
            "gamma": float(job["evaluation"]["c_v_gamma"]),
        }
        full_quality = topic_quality_metrics(reporting, counts, **quality_args)
        parent_quality = topic_quality_metrics(parent_reporting, counts, **quality_args)
        words = np.asarray(full_quality["top_word_indices"], dtype=np.int64)
        parent_words = np.asarray(parent_quality["top_word_indices"], dtype=np.int64)
        recall = float(
            np.mean(
                [
                    np.intersect1d(words[k], aligned_cores[k]).size
                    / aligned_cores.shape[1]
                    for k in range(aligned_cores.shape[0])
                ]
            )
        )
        support = _top_word_support(
            words,
            graph,
            counts,
            minimum_document_frequency=int(
                job["lexical_graph"]["minimum_document_frequency"]
            ),
        )
        prototype_audit = dict(bank.audit)
        prototype_audit["fit_partition"] = "all_695_train_plus_spent_development_rows"
        projection_receipts.append(
            {
                "seed": seed,
                "prototype_seed": prototype_seed,
                "prototype": prototype_audit,
                "alignment": alignment,
                "projection": projection.audit,
                "zero_projection_identity_error": zero_error,
                "calibration": calibration.audit,
            }
        )
        diagnostic = {
            "seed": seed,
            "fit_documents": int(counts.shape[0]),
            "parent_early_window_mean_loss": early_loss,
            "parent_late_window_mean_loss": late_loss,
            "parent_training_loss_reduction": early_loss - late_loss,
            "development_npmi_full": float(full_quality["npmi_at_10"]),
            "development_npmi_parent": float(parent_quality["npmi_at_10"]),
            "development_npmi_graph_effect": float(
                full_quality["npmi_at_10"] - parent_quality["npmi_at_10"]
            ),
            "development_cv_full": float(full_quality["c_v_at_10"]),
            "development_cv_parent": float(parent_quality["c_v_at_10"]),
            "development_cv_graph_effect": float(
                full_quality["c_v_at_10"] - parent_quality["c_v_at_10"]
            ),
            "development_topic_diversity_full": float(
                full_quality["topic_diversity_at_10"]
            ),
            "development_top_word_redundancy_full": float(
                full_quality["top_word_redundancy_at_10"]
            ),
            "prototype_core_recall": recall,
            "zero_projection_identity_error": zero_error,
            "projection_column_sum_error": float(
                projection.audit["maximum_column_sum_error"]
            ),
            "projection_minimum_probability": float(
                projection.audit["minimum_probability"]
            ),
            "projection_optimality_error": float(
                projection.audit["maximum_projection_optimality_error"]
            ),
            "maximum_nonprototype_change": float(
                projection.audit["maximum_nonprototype_change"]
            ),
            "mean_vocabulary_column_total_variation": float(
                projection.audit["mean_vocabulary_column_total_variation"]
            ),
            "calibration_mean_error": float(
                calibration.audit["maximum_changed_mean_error"]
            ),
            "calibration_variance_error": float(
                calibration.audit["maximum_changed_regularized_variance_error"]
            ),
            "calibration_unchanged_identity_error": float(
                calibration.audit["maximum_unchanged_identity_error"]
            ),
            "calibration_maximum_scale": float(
                calibration.audit["maximum_changed_scale"]
            ),
            **support,
        }
        diagnostics.append(diagnostic)
        full_top_words.append(words)
        parent_top_words.append(parent_words)

        checkpoint = output / "checkpoints" / f"seed_{seed}_final_frozen.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "implementation_version": IMPLEMENTATION,
            "seed": seed,
            "official_source_commit": job["official_commit"],
            "selected_configuration": job["selected_configuration"],
            "model_configuration": job["model"],
            "training_configuration": job["training"],
            "final_fit_receipt": development["merge_receipt"],
            "parent_state_dict": {
                name: value.detach().cpu()
                for name, value in parent.state_dict().items()
            },
            "parent_state_sha256": final_hash,
            "parent_training_loss_reduction": early_loss - late_loss,
            "base_affinity": torch.from_numpy(base_affinity.astype(np.float32)),
            "base_affinity_sha256": _array_hash(base_affinity.astype(np.float32)),
            "effective_affinity": torch.from_numpy(
                projection.affinity.astype(np.float32)
            ),
            "effective_affinity_sha256": _array_hash(
                projection.affinity.astype(np.float32)
            ),
            "aligned_prototype_core_indices": torch.from_numpy(
                aligned_cores.astype(np.int64)
            ),
            "calibration_scale": torch.from_numpy(calibration.scale.astype(np.float32)),
            "calibration_shift": torch.from_numpy(calibration.shift.astype(np.float32)),
            "calibration_changed_mask": torch.from_numpy(calibration.changed_mask),
            "calibration_audit": calibration.audit,
            "pooled_prior": torch.from_numpy(pooled_prior.astype(np.float32)),
            "pooled_mixture_weight": float(job["pooled_backoff"]["mixture_weight"]),
            "context_centroids": torch.from_numpy(context.centroids.astype(np.float32)),
            "context_global_bow": torch.from_numpy(context.global_bow),
            "context_audit": context.audit,
            "graph_receipt": graph_receipt,
            "prototype_audit": prototype_audit,
            "alignment_audit": alignment,
            "projection_audit": projection.audit,
            "variant_manifest_identity": variant_manifest["variant_manifest_identity"],
            "confirmation_preregistration_sha256": job[
                "confirmation_preregistration_sha256"
            ],
            "test_metrics": None,
            "test_access_before_freeze": 0,
        }
        torch.save(payload, checkpoint)
        checkpoint_rows.append(
            {
                "seed": seed,
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "size_bytes": int(checkpoint.stat().st_size),
                "parent_state_sha256": final_hash,
                "base_affinity_sha256": payload["base_affinity_sha256"],
                "effective_affinity_sha256": payload["effective_affinity_sha256"],
                "variant_manifest_identity": variant_manifest[
                    "variant_manifest_identity"
                ],
                "parent_training_loss_reduction": early_loss - late_loss,
            }
        )
        print(
            f"[V6.1 Step 7 worker] seed={seed} FROZEN | "
            f"loss-reduction={early_loss - late_loss:+.6f} "
            f"NPMI={diagnostic['development_npmi_full']:+.6f} "
            f"C_v={diagnostic['development_cv_full']:.6f}",
            flush=True,
        )
        del parent
        torch.cuda.empty_cache()

    summary = validate_final_diagnostics(diagnostics, expected_seeds=seeds)
    summary["full_topic_stability"] = matched_topic_stability(full_top_words)
    summary["parent_topic_stability"] = matched_topic_stability(parent_top_words)
    _write_csv(output / "final_training_trace.csv", trace_rows)
    _write_csv(output / "final_seed_diagnostics.csv", diagnostics)
    _write_json(output / "final_diagnostic_summary.json", summary)
    _write_json(
        output / "final_projection_calibration_receipts.json", projection_receipts
    )
    _write_json(output / "checkpoint_manifest.json", checkpoint_rows)
    freeze = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "created_at_utc": _utc_now(),
        "final_fit_documents": int(counts.shape[0]),
        "original_training_documents": int(
            job["final_fit"]["original_training_documents"]
        ),
        "spent_development_documents": int(
            job["final_fit"]["spent_development_documents"]
        ),
        "seeds": seeds,
        "seed_selection_performed": False,
        "validation_early_stopping_performed": False,
        "hyperparameter_selection_performed": False,
        "test_documents_used": 0,
        "labels_used": 0,
        "comparator_outputs_used": 0,
        "checkpoint_manifest_sha256": sha256_file(output / "checkpoint_manifest.json"),
        "variant_manifest_sha256": sha256_file(output / "final_variant_manifest.json"),
        "confirmation_preregistration_sha256": job[
            "confirmation_preregistration_sha256"
        ],
        "checkpoints": checkpoint_rows,
    }
    freeze["freeze_identity"] = sha256_object(freeze)
    _write_json(output / "final_freeze_receipt.json", freeze)
    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "status": "PASS",
        "official_class_loaded_from": str(loaded_path),
        "final_fit_documents": int(counts.shape[0]),
        "parent_neural_fits": len(seeds),
        "seeds": seeds,
        "seed_selection_performed": False,
        "validation_early_stopping_performed": False,
        "hyperparameter_selection_performed": False,
        "variant_count": len(variant_registry),
        "checkpoint_manifest": checkpoint_rows,
        "freeze_identity": freeze["freeze_identity"],
        "diagnostic_summary": summary,
        "test_documents_used": 0,
        "labels_used": 0,
        "baseline_or_sota_outputs_used": 0,
    }
    _write_json(output / "worker_result.json", result)
    print(
        f"[V6.1 Step 7 worker] PASS | frozen-seeds={len(seeds)} "
        f"variants={len(variant_registry)} test=0 comparators=0",
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
    torch = _seed_everything(int(job["training"]["seeds"][0]))
    _write_json(output / "runtime_receipt.json", _runtime_receipt(torch))
    run_worker(job, output, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
