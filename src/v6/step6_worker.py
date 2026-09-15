"""Exact-runtime worker for the frozen V6 one-shot validation gate.

All three parent models are trained on the 620-row training partition and
saved with hashes before validation arrays are semantically loaded.  The
validation partition is then opened once and every predeclared seed is
evaluated; no seed, epoch, module, or hyperparameter is selected from it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.city_backoff import (
    estimate_hierarchical_city_priors,
    mix_with_prior,
    select_city_prior_rows,
)
from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_evidence import (
    macro_city_completion,
    matched_topic_stability,
    positive_npmi_graph,
    topic_quality_metrics,
)
from src.v6.development_worker import (
    _gradient_norm,
    _new_parent,
    _state_hash,
)
from src.v6.graph_prototype_adapter import (
    align_prototype_cores_to_topics,
    build_graph_prototype_bank,
    minimum_distortion_graph_projection,
)
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt
from src.v6.step4r2_worker import _parent_probabilities
from src.v6.step5r2_worker import (
    _array_hash,
    _probabilities_from_affinity,
    _seed_everything,
    _top_word_support,
)
from src.v6.validation_context import construct_training_only_context
from src.v6.validation_gate import summarize_validation_seeds


IMPLEMENTATION = "v6_step6_one_shot_validation_freeze_gate_r1"


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        job.get("schema_version") != 1
        or job.get("mode") != "step6"
        or job.get("implementation_version") != IMPLEMENTATION
    ):
        raise ValueError("unsupported V6 Step-6 worker job")
    for section in ("training_inputs", "validation_inputs"):
        for role, item in job[section].items():
            source = Path(item["path"])
            if not source.is_file():
                raise FileNotFoundError(
                    f"worker input is missing ({section}/{role}): {source}"
                )
            if sha256_file(source) != str(item["sha256"]):
                raise ValueError(
                    f"worker input hash changed ({section}/{role})"
                )
    lineage = job["step5r2_lineage"]
    for key in ("contract", "selected_metrics", "worker_result"):
        item = lineage[key]
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != str(item["sha256"]):
            raise ValueError(f"Step-5R2 lineage changed ({key})")
    for row in lineage["selected_fold_checkpoints"]:
        source = Path(row["path"])
        if not source.is_file() or sha256_file(source) != str(row["sha256"]):
            raise ValueError("a selected Step-5R2 fold checkpoint changed")
    return job


def _load_training(job: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        name: Path(item["path"])
        for name, item in job["training_inputs"].items()
    }
    counts = sparse.load_npz(paths["train_counts"]).tocsr().astype(np.int64)
    with np.load(paths["word_embeddings"], allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    rows = pd.read_csv(
        paths["train_rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    ).sort_values("matrix_row").reset_index(drop=True)
    if counts.shape != (620, 1148):
        raise ValueError("the frozen training count shape changed")
    if word_embeddings.shape != (1148, 768):
        raise ValueError("the frozen vocabulary embedding shape changed")
    if rows["matrix_row"].astype(int).tolist() != list(range(620)):
        raise ValueError("training metadata is not aligned to the count matrix")
    if set(rows["split"].astype(str)) != {"train"}:
        raise ValueError("non-training rows entered the fit partition")
    city_names = list(map(str, job["cities"]))
    lookup = {name: index for index, name in enumerate(city_names)}
    if set(rows["city"].astype(str)) != set(lookup):
        raise ValueError("the training city universe changed")
    city_indices = rows["city"].astype(str).map(lookup).to_numpy(dtype=np.int64)
    return {
        "counts": counts,
        "word_embeddings": word_embeddings,
        "rows": rows,
        "city_names": city_names,
        "city_indices": city_indices,
    }


def _load_validation(job: Mapping[str, Any]) -> dict[str, Any]:
    """This function is deliberately called only after training is frozen."""

    paths = {
        name: Path(item["path"])
        for name, item in job["validation_inputs"].items()
    }
    observed = sparse.load_npz(paths["validation_observed"]).tocsr().astype(
        np.int64
    )
    target = sparse.load_npz(paths["validation_target"]).tocsr().astype(
        np.int64
    )
    rows = pd.read_csv(
        paths["validation_rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    ).sort_values("completion_row").reset_index(drop=True)
    expected_documents = int(job["validation"]["expected_completion_documents"])
    if observed.shape != target.shape or observed.shape != (
        expected_documents,
        1148,
    ):
        raise ValueError("frozen validation completion matrices are misaligned")
    if rows["completion_row"].astype(int).tolist() != list(
        range(expected_documents)
    ):
        raise ValueError("validation rows are not aligned to completion matrices")
    if set(rows["split"].astype(str)) != {"validation"}:
        raise ValueError("a non-validation row entered the validation gate")
    full_counts = (observed + target).tocsr()
    if (full_counts != full_counts.astype(np.int64)).nnz != 0:
        raise ValueError("validation counts are not integral")
    if np.any(np.asarray(observed.sum(axis=1)).reshape(-1) <= 0.0):
        raise ValueError("a validation observed half is empty")
    if np.any(np.asarray(target.sum(axis=1)).reshape(-1) <= 0.0):
        raise ValueError("a validation target half is empty")
    target_tokens = int(target.sum())
    if target_tokens != int(job["validation"]["expected_target_tokens"]):
        raise ValueError("validation target-token count changed")
    city_names = list(map(str, job["cities"]))
    lookup = {name: index for index, name in enumerate(city_names)}
    if set(rows["city"].astype(str)) != set(lookup):
        raise ValueError("the validation city universe changed")
    city_indices = rows["city"].astype(str).map(lookup).to_numpy(dtype=np.int64)
    return {
        "observed": observed,
        "target": target,
        "full_counts": full_counts,
        "rows": rows,
        "city_indices": city_indices,
        "target_tokens": target_tokens,
    }


def _window_mean(
    trace: list[dict[str, Any]], key: str, *, first: bool
) -> float:
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
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    tensor = torch.from_numpy(np.asarray(train_input, dtype=np.float32))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(configuration["parent_learning_rate"])
    )
    trace: list[dict[str, Any]] = []
    epochs = int(configuration["parent_epochs"])
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
                bool(torch.isfinite(value).all())
                for value in scalar_values.values()
            ):
                raise FloatingPointError("non-finite official parent objective")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            maximum_gradient = max(
                maximum_gradient, _gradient_norm(model.parameters())
            )
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
                f"[V6 Step 6 worker] seed={seed} train epoch={epoch}/{epochs} "
                f"loss={row['loss']:.6f}",
                flush=True,
            )
    return trace


def _variant_completion(
    target: sparse.csr_matrix,
    probabilities: np.ndarray,
    city_indices: np.ndarray,
    city_names: list[str],
    probability_floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return macro_city_completion(
        target,
        probabilities,
        city_indices,
        city_names,
        probability_floor=float(probability_floor),
    )


def _evaluate_checkpoint(
    checkpoint_row: Mapping[str, Any],
    job: Mapping[str, Any],
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    validation_input: np.ndarray,
    graph: np.ndarray,
    NewMethod: type,
    *,
    device: Any,
    torch: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    checkpoint_path = Path(checkpoint_row["path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(payload.get("seed", -1))
    if (
        payload.get("schema_version") != 1
        or payload.get("implementation_version") != IMPLEMENTATION
        or seed != int(checkpoint_row["seed"])
        or payload.get("model_configuration") != dict(job["model"])
        or payload.get("training_configuration") != dict(job["training"])
        or payload.get("official_source_commit") != job["official_commit"]
    ):
        raise ValueError("a frozen Step-6 training checkpoint changed identity")
    parent = _new_parent(
        NewMethod, training["word_embeddings"], job["model"]
    )
    parent.load_state_dict(payload["parent_state_dict"], strict=True)
    parent_hash = _state_hash(parent.state_dict())
    if parent_hash != str(payload["parent_state_sha256"]):
        raise ValueError("frozen Step-6 parent state hash does not reproduce")
    parent = parent.to(device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    effective = payload["effective_affinity"].detach().to(device)
    effective_array = effective.detach().cpu().numpy()
    if _array_hash(effective_array) != str(payload["effective_affinity_sha256"]):
        raise ValueError("frozen projected affinity hash does not reproduce")
    city_prior = payload["city_priors"].detach().cpu().numpy().astype(np.float64)
    pooled_prior = payload["pooled_prior"].detach().cpu().numpy().astype(np.float64)
    aligned_cores = (
        payload["aligned_prototype_core_indices"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    evaluation = job["evaluation"]
    parent_probability = _parent_probabilities(
        parent,
        validation_input,
        batch_size=int(evaluation["inference_batch_size"]),
        device=device,
    )
    projected_probability = _probabilities_from_affinity(
        parent,
        validation_input,
        effective,
        batch_size=int(evaluation["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    validation_cities = validation["city_indices"]
    city_rows = select_city_prior_rows(city_prior, validation_cities)
    pooled_rows = np.repeat(
        pooled_prior[None, :], len(validation_cities), axis=0
    )
    rho = float(job["city_backoff"]["fixed_mixture_weight"])
    variants = {
        "full_graph_plus_city": mix_with_prior(
            projected_probability, city_rows, mixture_weight=rho
        ),
        "without_graph_city_retained": mix_with_prior(
            parent_probability, city_rows, mixture_weight=rho
        ),
        "without_city_graph_retained": projected_probability,
        "matched_pooled_backoff_graph_retained": mix_with_prior(
            projected_probability, pooled_rows, mixture_weight=rho
        ),
        "exact_official_parent": parent_probability,
    }
    if list(variants) != list(evaluation["variants"]):
        raise ValueError("validation variant order changed after preregistration")
    completion: dict[str, dict[str, Any]] = {}
    all_city_rows: list[dict[str, Any]] = []
    maximum_probability_error = 0.0
    for name, probabilities in variants.items():
        maximum_probability_error = max(
            maximum_probability_error,
            float(np.abs(probabilities.sum(axis=1) - 1.0).max()),
        )
        overall, per_city = _variant_completion(
            validation["target"],
            probabilities,
            validation_cities,
            training["city_names"],
            float(evaluation["probability_floor"]),
        )
        completion[name] = overall
        for city_row in per_city:
            all_city_rows.append(
                {"seed": seed, "variant": name, **city_row}
            )

    with torch.no_grad():
        base_affinity = parent.get_beta().detach().cpu().numpy().astype(np.float64)
    effective_for_reporting = effective.detach().cpu().numpy().astype(np.float64)
    full_reporting = effective_for_reporting / np.maximum(
        effective_for_reporting.sum(axis=1, keepdims=True), 1.0e-300
    )
    control_reporting = base_affinity / np.maximum(
        base_affinity.sum(axis=1, keepdims=True), 1.0e-300
    )
    quality_arguments = {
        "top_words": int(evaluation["top_words"]),
        "window_size": int(evaluation["c_v_window_size"]),
        "gamma": float(evaluation["c_v_gamma"]),
    }
    full_validation_quality = topic_quality_metrics(
        full_reporting, validation["full_counts"], **quality_arguments
    )
    control_validation_quality = topic_quality_metrics(
        control_reporting, validation["full_counts"], **quality_arguments
    )
    full_training_quality = topic_quality_metrics(
        full_reporting, training["counts"], **quality_arguments
    )
    control_training_quality = topic_quality_metrics(
        control_reporting, training["counts"], **quality_arguments
    )
    full_top_words = np.asarray(
        full_validation_quality["top_word_indices"], dtype=np.int64
    )
    control_top_words = np.asarray(
        control_validation_quality["top_word_indices"], dtype=np.int64
    )
    expected_top_words = np.argsort(
        -full_reporting, axis=1, kind="stable"
    )[:, : int(evaluation["top_words"])]
    if not np.array_equal(full_top_words, expected_top_words):
        raise AssertionError("reporting and validation evaluator ranks differ")
    recall = float(
        np.mean(
            [
                np.intersect1d(full_top_words[topic], aligned_cores[topic]).size
                / aligned_cores.shape[1]
                for topic in range(aligned_cores.shape[0])
            ]
        )
    )
    support = _top_word_support(
        full_top_words,
        graph,
        training["counts"],
        minimum_document_frequency=int(
            job["lexical_graph"]["minimum_document_frequency"]
        ),
    )
    full_nll = float(
        completion["full_graph_plus_city"]["macro_city_nll_per_token"]
    )
    no_graph_nll = float(
        completion["without_graph_city_retained"]["macro_city_nll_per_token"]
    )
    no_city_nll = float(
        completion["without_city_graph_retained"]["macro_city_nll_per_token"]
    )
    pooled_nll = float(
        completion["matched_pooled_backoff_graph_retained"][
            "macro_city_nll_per_token"
        ]
    )
    parent_nll = float(
        completion["exact_official_parent"]["macro_city_nll_per_token"]
    )
    parent_after_hash = _state_hash(parent.state_dict())
    projection = payload["projection_audit"]
    row = {
        "seed": seed,
        "fit_documents": 620,
        "validation_documents": int(len(validation["rows"])),
        "validation_target_tokens": int(validation["target_tokens"]),
        "validation_npmi_full": float(full_validation_quality["npmi_at_10"]),
        "validation_npmi_without_graph": float(
            control_validation_quality["npmi_at_10"]
        ),
        "validation_npmi_improvement_vs_without_graph": float(
            full_validation_quality["npmi_at_10"]
            - control_validation_quality["npmi_at_10"]
        ),
        "validation_cv_full": float(full_validation_quality["c_v_at_10"]),
        "validation_cv_without_graph": float(
            control_validation_quality["c_v_at_10"]
        ),
        "validation_cv_improvement_vs_without_graph": float(
            full_validation_quality["c_v_at_10"]
            - control_validation_quality["c_v_at_10"]
        ),
        "validation_zero_joint_pairs_full": int(
            full_validation_quality["zero_joint_pairs"]
        ),
        "validation_zero_joint_pairs_without_graph": int(
            control_validation_quality["zero_joint_pairs"]
        ),
        "validation_zero_joint_pair_reduction_vs_without_graph": int(
            control_validation_quality["zero_joint_pairs"]
            - full_validation_quality["zero_joint_pairs"]
        ),
        "validation_topic_diversity_full": float(
            full_validation_quality["topic_diversity_at_10"]
        ),
        "validation_top_word_redundancy_full": float(
            full_validation_quality["top_word_redundancy_at_10"]
        ),
        "training_reference_npmi_full": float(
            full_training_quality["npmi_at_10"]
        ),
        "training_reference_npmi_without_graph": float(
            control_training_quality["npmi_at_10"]
        ),
        "training_reference_cv_full": float(full_training_quality["c_v_at_10"]),
        "training_reference_cv_without_graph": float(
            control_training_quality["c_v_at_10"]
        ),
        "full_macro_city_nll": full_nll,
        "without_graph_macro_city_nll": no_graph_nll,
        "without_city_macro_city_nll": no_city_nll,
        "matched_pooling_macro_city_nll": pooled_nll,
        "official_parent_macro_city_nll": parent_nll,
        "graph_nll_reduction_vs_without_graph": no_graph_nll - full_nll,
        "city_nll_reduction_vs_without_city": no_city_nll - full_nll,
        "city_nll_reduction_vs_matched_pooling": pooled_nll - full_nll,
        "full_nll_reduction_vs_official_parent": parent_nll - full_nll,
        "prototype_core_recall": recall,
        "maximum_probability_sum_error": maximum_probability_error,
        "maximum_nonprototype_change": float(
            projection["maximum_nonprototype_change"]
        ),
        "projection_column_sum_error": float(
            projection["maximum_column_sum_error"]
        ),
        "projection_minimum_probability": float(
            projection["minimum_probability"]
        ),
        "projection_optimality_error": float(
            projection["maximum_projection_optimality_error"]
        ),
        "changed_columns": int(projection["changed_columns"]),
        "mean_vocabulary_column_total_variation": float(
            projection["mean_vocabulary_column_total_variation"]
        ),
        "maximum_changed_column_total_variation": float(
            projection["maximum_changed_column_total_variation"]
        ),
        "parent_training_loss_reduction": float(
            payload["parent_training_loss_reduction"]
        ),
        "parent_state_change": int(parent_after_hash != parent_hash),
        "same_affinity_for_likelihood_and_reporting": 1,
        "full_affinity_sha256": _array_hash(effective_array),
        "parent_affinity_sha256": _array_hash(
            base_affinity.astype(np.float32)
        ),
        **support,
    }
    del parent
    torch.cuda.empty_cache()
    return row, all_city_rows, full_top_words, control_top_words


def run_worker(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V6 Step 6 requires the exact CUDA runtime from Step 3")
    NewMethod, loaded_path = _load_official_newmethod(
        Path(job["official_source"])
    )
    device = torch.device("cuda")

    # Phase A: fit and freeze everything using original training rows only.
    training = _load_training(job)
    context = construct_training_only_context(
        training["counts"],
        training["word_embeddings"],
        clusters=int(job["global_context"]["clusters"]),
        n_init=int(job["global_context"]["n_init"]),
        seed=int(job["global_context"]["seed"]),
    )
    prior = estimate_hierarchical_city_priors(
        training["counts"].toarray(),
        training["city_indices"],
        cities=len(training["city_names"]),
        shrinkage=float(job["city_backoff"]["fixed_shrinkage"]),
        pseudocount=float(job["city_backoff"]["fixed_pseudocount"]),
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
    graph_receipt["fit_partition"] = "all_620_original_training_rows_only"
    if int(graph_receipt["active_words"]) < int(
        job["lexical_graph"]["minimum_active_words"]
    ):
        raise ValueError("the full training graph lacks required word support")

    trace_rows: list[dict[str, Any]] = []
    prototype_receipts: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    maximum_zero_projection_error = 0.0
    seeds = list(map(int, job["training"]["seeds"]))
    for seed in seeds:
        print(
            f"[V6 Step 6 worker] seed={seed} TRAIN-ONLY START | validation unopened",
            flush=True,
        )
        _seed_everything(seed)
        parent = _new_parent(
            NewMethod, training["word_embeddings"], job["model"]
        ).to(device)
        initial_parent_hash = _state_hash(parent.state_dict())
        trace = _train_parent(
            parent,
            context.train_input,
            job["training"],
            seed=seed,
            device=device,
        )
        trace_rows.extend(trace)
        final_parent_hash = _state_hash(parent.state_dict())
        if final_parent_hash == initial_parent_hash:
            raise RuntimeError("official parent did not change during training")
        early_loss = _window_mean(trace, "loss", first=True)
        late_loss = _window_mean(trace, "loss", first=False)
        parent.eval()
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            base_affinity = parent.get_beta().detach().cpu().numpy().astype(
                np.float64
            )
        prototype_seed = seed + int(
            job["graph_prototype_projection"]["prototype_seed_offset"]
        )
        prototype_bank = build_graph_prototype_bank(
            graph,
            topics=int(job["model"]["topics"]),
            top_k=int(job["evaluation"]["top_words"]),
            seed=prototype_seed,
            centrality_weight=float(
                job["graph_prototype_projection"]["centrality_weight"]
            ),
            minimum_cluster_size=int(
                job["graph_prototype_projection"]["minimum_cluster_size"]
            ),
        )
        aligned_cores, alignment = align_prototype_cores_to_topics(
            base_affinity,
            prototype_bank.core_indices,
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        zero = minimum_distortion_graph_projection(
            base_affinity,
            aligned_cores,
            strength=0.0,
            prototype_quota=int(
                job["graph_prototype_projection"]["selected_prototype_quota"]
            ),
            rank_margin=float(
                job["graph_prototype_projection"]["rank_margin"]
            ),
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        maximum_zero_projection_error = max(
            maximum_zero_projection_error,
            float(np.max(np.abs(zero.affinity - base_affinity))),
        )
        projection = minimum_distortion_graph_projection(
            base_affinity,
            aligned_cores,
            strength=float(job["graph_prototype_projection"]["strength"]),
            prototype_quota=int(
                job["graph_prototype_projection"]["selected_prototype_quota"]
            ),
            rank_margin=float(
                job["graph_prototype_projection"]["rank_margin"]
            ),
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        effective = torch.from_numpy(projection.affinity.astype(np.float32))
        prototype_audit = dict(prototype_bank.audit)
        prototype_audit["fit_partition"] = "all_620_original_training_rows_only"
        prototype_receipts.append(
            {
                "seed": seed,
                "prototype_seed": prototype_seed,
                "prototype": prototype_audit,
                "alignment": alignment,
                "projection": projection.audit,
            }
        )
        checkpoint = output / "checkpoints" / f"seed_{seed}_train_frozen.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "implementation_version": IMPLEMENTATION,
                "seed": seed,
                "official_source_commit": job["official_commit"],
                "model_configuration": job["model"],
                "training_configuration": job["training"],
                "city_backoff_configuration": job["city_backoff"],
                "lexical_graph_configuration": job["lexical_graph"],
                "projection_configuration": job[
                    "graph_prototype_projection"
                ],
                "parent_state_dict": {
                    name: value.detach().cpu()
                    for name, value in parent.state_dict().items()
                },
                "parent_state_sha256": final_parent_hash,
                "parent_training_loss_reduction": early_loss - late_loss,
                "effective_affinity": effective,
                "effective_affinity_sha256": _array_hash(
                    effective.numpy()
                ),
                "aligned_prototype_core_indices": torch.from_numpy(
                    aligned_cores.astype(np.int64)
                ),
                "city_priors": torch.from_numpy(prior.city.astype(np.float32)),
                "pooled_prior": torch.from_numpy(
                    prior.pooled.astype(np.float32)
                ),
                "context_centroids": torch.from_numpy(
                    context.centroids.astype(np.float32)
                ),
                "context_global_bow": torch.from_numpy(context.global_bow),
                "context_audit": context.audit,
                "graph_receipt": graph_receipt,
                "prototype_audit": prototype_audit,
                "alignment_audit": alignment,
                "projection_audit": projection.audit,
                "validation_metrics": None,
                "validation_access_before_checkpoint_freeze": 0,
            },
            checkpoint,
        )
        checkpoint_rows.append(
            {
                "seed": seed,
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "size_bytes": int(checkpoint.stat().st_size),
                "parent_state_sha256": final_parent_hash,
                "effective_affinity_sha256": _array_hash(effective.numpy()),
                "parent_training_loss_reduction": early_loss - late_loss,
            }
        )
        print(
            f"[V6 Step 6 worker] seed={seed} TRAIN-ONLY FROZEN | "
            f"loss-reduction={early_loss - late_loss:+.6f}",
            flush=True,
        )
        del parent
        torch.cuda.empty_cache()

    _write_csv(output / "training_trace.csv", trace_rows)
    _write_json(output / "training_context_receipt.json", context.audit)
    _write_json(output / "training_graph_receipt.json", graph_receipt)
    _write_json(output / "prototype_projection_receipts.json", prototype_receipts)
    _write_json(output / "checkpoint_manifest.json", checkpoint_rows)
    freeze_body = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "created_at_utc": _utc_now(),
        "training_documents": 620,
        "validation_semantic_accesses_before_freeze": 0,
        "seed_selection_performed": False,
        "hyperparameter_selection_performed": False,
        "checkpoint_manifest_sha256": sha256_file(
            output / "checkpoint_manifest.json"
        ),
        "checkpoints": checkpoint_rows,
    }
    freeze_body["freeze_identity"] = sha256_object(freeze_body)
    _write_json(output / "training_freeze_receipt.json", freeze_body)

    # Phase B: all model states are immutable; validation is opened once now.
    validation_open = {
        "schema_version": 1,
        "opened_at_utc": _utc_now(),
        "semantic_open_count": 1,
        "training_freeze_identity": freeze_body["freeze_identity"],
        "training_checkpoints_frozen_before_open": True,
        "validation_inputs": job["validation_inputs"],
        "test_inputs_available_to_worker": False,
        "comparator_outputs_available_to_worker": False,
    }
    _write_json(output / "validation_open_receipt.json", validation_open)
    print(
        "[V6 Step 6 worker] TRAINING FREEZE COMPLETE | opening frozen validation once",
        flush=True,
    )
    validation = _load_validation(job)
    validation_input, assignment_receipt = context.input_from_observed_counts(
        validation["observed"]
    )
    _write_json(output / "validation_assignment_receipt.json", assignment_receipt)

    seed_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    full_top_words: list[np.ndarray] = []
    control_top_words: list[np.ndarray] = []
    for checkpoint_row in checkpoint_rows:
        row, per_city, full_words, control_words = _evaluate_checkpoint(
            checkpoint_row,
            job,
            training,
            validation,
            validation_input,
            graph,
            NewMethod,
            device=device,
            torch=torch,
        )
        seed_rows.append(row)
        city_rows.extend(per_city)
        full_top_words.append(full_words)
        control_top_words.append(control_words)
        print(
            f"[V6 Step 6 worker] seed={row['seed']} VALIDATION DONE | "
            f"dNPMI={row['validation_npmi_improvement_vs_without_graph']:+.6f} "
            f"dC_v={row['validation_cv_improvement_vs_without_graph']:+.6f} "
            f"city-dNLL={row['city_nll_reduction_vs_without_city']:+.6f} "
            f"full-vs-parent={row['full_nll_reduction_vs_official_parent']:+.6f}",
            flush=True,
        )
    full_stability = matched_topic_stability(full_top_words)
    control_stability = matched_topic_stability(control_top_words)
    summary = summarize_validation_seeds(
        seed_rows,
        gates=job["gates"],
        full_topic_stability=full_stability,
        graph_removed_topic_stability=control_stability,
    )
    _write_csv(output / "validation_seed_metrics.csv", seed_rows)
    _write_csv(output / "validation_city_metrics.csv", city_rows)
    _write_json(output / "validation_summary.json", summary)
    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "status": summary["status"],
        "official_class_loaded_from": str(loaded_path),
        "training_documents": 620,
        "parent_neural_fits": len(seeds),
        "validation_documents_used": int(len(validation["rows"])),
        "validation_target_tokens": int(validation["target_tokens"]),
        "validation_semantic_open_count": 1,
        "paired_variant_evaluations": len(seeds)
        * len(job["evaluation"]["variants"]),
        "candidate_architectures_evaluated": 1,
        "seed_selection_performed": False,
        "hyperparameter_selection_performed": False,
        "validation_early_stopping_performed": False,
        "maximum_zero_projection_identity_error": maximum_zero_projection_error,
        "training_freeze_identity": freeze_body["freeze_identity"],
        "checkpoint_manifest": checkpoint_rows,
        "validation_summary": summary,
        "test_documents_used": 0,
        "labels_used": 0,
        "baseline_or_sota_outputs_used": 0,
    }
    _write_json(output / "worker_result.json", result)
    print(
        f"[V6 Step 6 worker] {result['status']} | validation gates="
        f"{summary['passed_gate_count']}/{summary['total_gate_count']} | "
        f"failed={'|'.join(summary['failed_gates']) or 'none'}",
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
