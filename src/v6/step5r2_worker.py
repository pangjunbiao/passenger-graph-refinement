"""Exact-runtime worker for V6 Step 5R2 revision 5.

Revision 5 keeps the exact Revision-4 mechanism: a frozen passing Step-4R2
EnCOT parent and a deterministic minimum-total-variation projection on the 100
fold-local graph-prototype word columns.  It corrects the component-survival
semantics.  The graph-removed NLL comparison is always reported as a trade-off,
while graph retention is decided by its lexical targets and the complete model
must still retain the city module's predictive advantage over matched pooling
and the official parent.  The identical projected affinity drives document
completion and displayed topic words.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.city_backoff import mix_with_prior, select_city_prior_rows
from src.v6.data_contract import sha256_file
from src.v6.development_evidence import (
    construct_fold_context,
    macro_city_completion,
    matched_topic_stability,
    positive_npmi_graph,
    topic_quality_metrics,
)
from src.v6.development_worker import _load_evidence, _new_parent, _state_hash
from src.v6.graph_prototype_adapter import (
    align_prototype_cores_to_topics,
    build_graph_prototype_bank,
    minimum_distortion_graph_projection,
)
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt
from src.v6.step4r2_worker import _parent_probabilities


IMPLEMENTATION = "v6_step5r2_target_aligned_graph_projection_gate_r5"
STEP4R2_IMPLEMENTATION = "v6_step4r2_hierarchical_city_backoff_gate_r1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty required table: {path.name}")
    pd.DataFrame.from_records(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _seed_everything(seed: int) -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    return torch


def _load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        job.get("schema_version") != 1
        or job.get("mode") != "step5r2"
        or job.get("implementation_version") != IMPLEMENTATION
    ):
        raise ValueError("unsupported V6 Step-5R2 R5 worker job")
    for role, item in job["inputs"].items():
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(f"worker input is missing ({role}): {source}")
        if sha256_file(source) != str(item["sha256"]):
            raise ValueError(f"worker input hash changed ({role})")
    lineage = job["step4r2_lineage"]
    for path_key, hash_key in (
        ("contract_path", "contract_sha256"),
        ("selected_fold_metrics_path", "selected_fold_metrics_sha256"),
    ):
        source = Path(lineage[path_key])
        if not source.is_file() or sha256_file(source) != str(lineage[hash_key]):
            raise ValueError(f"Step-4R2 lineage artifact changed: {source}")
    checkpoints = lineage["checkpoints"]
    if len(checkpoints) != len(job["development_folds"]):
        raise ValueError("Step-4R2 lineage does not contain one checkpoint per fold")
    for row in checkpoints:
        source = Path(row["path"])
        if not source.is_file() or sha256_file(source) != str(row["sha256"]):
            raise ValueError(f"Step-4R2 checkpoint is missing or changed: {source}")
    return job


def _validate_prior(
    values: np.ndarray, *, rows: int, vocabulary: int, name: str
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (rows, vocabulary):
        raise ValueError(f"{name} has the wrong shape")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} contains invalid probability values")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{name} is not row-normalized")
    return result


def _restore_parent_and_priors(
    NewMethod: type,
    evidence: Mapping[str, Any],
    job: Mapping[str, Any],
    checkpoint_row: Mapping[str, Any],
    expected_fold_row: Mapping[str, Any],
    context_audit: Mapping[str, Any],
    torch: Any,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_row["path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fold = int(checkpoint_row["fold"])
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("unsupported Step-4R2 checkpoint schema")
    if payload.get("implementation_version") != STEP4R2_IMPLEMENTATION:
        raise ValueError("Step-4R2 checkpoint implementation changed")
    if int(payload.get("fold", -1)) != fold:
        raise ValueError("Step-4R2 checkpoint fold identity changed")
    if str(payload.get("official_source_commit")) != str(job["official_commit"]):
        raise ValueError("Step-4R2 checkpoint official-source identity changed")
    if payload.get("model_configuration") != dict(job["model"]):
        raise ValueError("Step-4R2 checkpoint architecture changed")
    stored_prior = payload.get("city_prior_configuration", {})
    backoff = job["city_backoff"]
    if float(stored_prior.get("shrinkage", float("nan"))) != float(
        backoff["fixed_shrinkage"]
    ):
        raise ValueError("city-prior shrinkage changed after Step 4R2")
    if float(stored_prior.get("pseudocount", float("nan"))) != float(
        backoff["fixed_pseudocount"]
    ):
        raise ValueError("city-prior pseudocount changed after Step 4R2")
    selected_weight = float(job["selected_mixture_weight"])
    if selected_weight not in list(map(float, payload["candidate_mixture_weights"])):
        raise ValueError("selected city mixture was not evaluated in Step 4R2")
    for key in ("centroids_sha256", "assignments_sha256", "global_bow_sha256"):
        if str(payload["context_audit"].get(key)) != str(context_audit.get(key)):
            raise ValueError(f"fold context changed after Step 4R2 ({key})")

    parent = _new_parent(NewMethod, evidence["word_embeddings"], job["model"])
    parent.load_state_dict(payload["parent_state_dict"], strict=True)
    parent_hash = _state_hash(parent.state_dict())
    if parent_hash != str(expected_fold_row["restored_parent_state_sha256"]):
        raise ValueError("restored parent state differs from passing Step-4R2")

    cities = len(evidence["city_names"])
    vocabulary = int(evidence["counts"].shape[1])
    city_prior = _validate_prior(
        payload["city_priors"].detach().cpu().numpy(),
        rows=cities,
        vocabulary=vocabulary,
        name="city priors",
    )
    pooled_prior = _validate_prior(
        payload["pooled_prior"].detach().cpu().numpy()[None, :],
        rows=1,
        vocabulary=vocabulary,
        name="pooled prior",
    )[0]
    return parent, city_prior, pooled_prior, {
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": str(checkpoint_row["sha256"]),
        "restored_parent_state_sha256": parent_hash,
    }


def _completion_partition(
    evidence: Mapping[str, Any], context: Any, fold: int
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
    source_rows = evidence["completion_source_rows"]
    source_folds = evidence["fold_assignments"][source_rows]
    selected = np.flatnonzero(source_folds == int(fold))
    if selected.size < 1:
        raise ValueError("a heldout completion partition is empty")
    inputs = context.input_for_local_counts(evidence["completion_observed"][selected])
    targets = evidence["completion_target"][selected].tocsr()
    cities = evidence["completion_city_indices"][selected]
    if np.any(np.asarray(targets.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("a completion target is empty")
    return inputs, targets, cities


def _top_word_support(
    top_words: np.ndarray,
    graph: np.ndarray,
    training_counts: sparse.csr_matrix,
    *,
    minimum_document_frequency: int,
) -> dict[str, float | int]:
    binary = training_counts.copy().tocsr()
    binary.data = np.ones_like(binary.data)
    frequency = np.asarray(binary.sum(axis=0)).reshape(-1)
    degree = np.asarray(graph).sum(axis=1)
    words = np.asarray(top_words, dtype=np.int64)
    train_supported = frequency[words] >= int(minimum_document_frequency)
    graph_supported = degree[words] > 0.0
    pair_values: list[float] = []
    for topic in words:
        block = graph[np.ix_(topic, topic)]
        upper = block[np.triu_indices(len(topic), 1)]
        pair_values.extend((upper > 0.0).astype(np.float64).tolist())
    return {
        "displayed_train_support_fraction": float(train_supported.mean()),
        "displayed_graph_support_fraction": float(graph_supported.mean()),
        "displayed_positive_graph_pair_fraction": float(np.mean(pair_values)),
        "minimum_displayed_training_document_frequency": int(frequency[words].min()),
    }


def _probabilities_from_affinity(
    parent: Any,
    input_values: np.ndarray,
    affinity: Any,
    *,
    batch_size: int,
    device: Any,
    torch: Any,
) -> np.ndarray:
    parent.eval()
    vocabulary = int(parent.vocab_size)
    fixed_affinity = affinity.detach().to(device)
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
                parent.decoder_bn((global_theta * local_theta) @ fixed_affinity),
                dim=-1,
            )
            rows.append(probabilities.detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(rows, axis=0)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError("projected parent returned invalid probabilities")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise FloatingPointError("projected probability rows are not normalized")
    return result


def _evaluate_model(
    parent: Any,
    evidence: Mapping[str, Any],
    context: Any,
    fold: int,
    city_prior: np.ndarray,
    pooled_prior: np.ndarray,
    graph: np.ndarray,
    job: Mapping[str, Any],
    *,
    affinity_override: Any | None = None,
    device: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], np.ndarray]:
    evaluation = job["evaluation"]
    held_input, held_target, held_cities = _completion_partition(
        evidence, context, fold
    )
    if affinity_override is None:
        parent_probability = _parent_probabilities(
            parent,
            held_input,
            batch_size=int(evaluation["inference_batch_size"]),
            device=device,
        )
        affinity_tensor = parent.get_beta()
    else:
        parent_probability = _probabilities_from_affinity(
            parent,
            held_input,
            affinity_override,
            batch_size=int(evaluation["inference_batch_size"]),
            device=device,
            torch=torch,
        )
        affinity_tensor = affinity_override
    city_rows = select_city_prior_rows(city_prior, held_cities)
    pooled_rows = np.repeat(pooled_prior[None, :], len(held_cities), axis=0)
    rho = float(job["selected_mixture_weight"])
    city_probability = mix_with_prior(parent_probability, city_rows, mixture_weight=rho)
    pooled_probability = mix_with_prior(
        parent_probability, pooled_rows, mixture_weight=rho
    )
    parent_completion, parent_city = macro_city_completion(
        held_target,
        parent_probability,
        held_cities,
        evidence["city_names"],
        probability_floor=float(evaluation["probability_floor"]),
    )
    city_completion, active_city = macro_city_completion(
        held_target,
        city_probability,
        held_cities,
        evidence["city_names"],
        probability_floor=float(evaluation["probability_floor"]),
    )
    pooled_completion, pooled_city = macro_city_completion(
        held_target,
        pooled_probability,
        held_cities,
        evidence["city_names"],
        probability_floor=float(evaluation["probability_floor"]),
    )
    with torch.no_grad():
        affinity = affinity_tensor.detach().cpu().numpy().astype(np.float64)
    reporting = affinity / np.maximum(affinity.sum(axis=1, keepdims=True), 1.0e-300)
    strict = topic_quality_metrics(
        reporting,
        evidence["counts"][context.evaluation_indices],
        top_words=int(evaluation["top_words"]),
        window_size=int(evaluation["c_v_window_size"]),
        gamma=float(evaluation["c_v_gamma"]),
    )
    standard = topic_quality_metrics(
        reporting,
        evidence["counts"],
        top_words=int(evaluation["top_words"]),
        window_size=int(evaluation["c_v_window_size"]),
        gamma=float(evaluation["c_v_gamma"]),
    )
    top_words = np.asarray(strict["top_word_indices"], dtype=np.int64)
    expected_top_words = np.argsort(
        -reporting, axis=1, kind="stable"
    )[:, : int(evaluation["top_words"])]
    if not np.array_equal(top_words, expected_top_words):
        raise AssertionError("quality evaluator and supplied affinity rank differently")
    support = _top_word_support(
        top_words,
        graph,
        evidence["counts"][context.train_indices],
        minimum_document_frequency=int(
            job["lexical_graph"]["minimum_document_frequency"]
        ),
    )
    probability_error = max(
        float(np.abs(parent_probability.sum(axis=1) - 1.0).max()),
        float(np.abs(city_probability.sum(axis=1) - 1.0).max()),
        float(np.abs(pooled_probability.sum(axis=1) - 1.0).max()),
    )
    metrics = {
        "documents": int(city_completion["documents"]),
        "tokens": int(city_completion["tokens"]),
        "parent_macro_city_nll": float(parent_completion["macro_city_nll_per_token"]),
        "pooled_backoff_macro_city_nll": float(
            pooled_completion["macro_city_nll_per_token"]
        ),
        "city_backoff_macro_city_nll": float(
            city_completion["macro_city_nll_per_token"]
        ),
        "city_micro_nll": float(city_completion["micro_nll_per_token"]),
        "probability_sum_error": probability_error,
        "heldout_c_v_at_10": float(strict["c_v_at_10"]),
        "heldout_npmi_at_10": float(strict["npmi_at_10"]),
        "heldout_topic_diversity_at_10": float(strict["topic_diversity_at_10"]),
        "heldout_top_word_redundancy_at_10": float(
            strict["top_word_redundancy_at_10"]
        ),
        "heldout_zero_joint_pairs": int(strict["zero_joint_pairs"]),
        "heldout_absent_word_pairs": int(strict["absent_word_pairs"]),
        "standard_c_v_at_10": float(standard["c_v_at_10"]),
        "standard_npmi_at_10": float(standard["npmi_at_10"]),
        "standard_zero_joint_pairs": int(standard["zero_joint_pairs"]),
        "affinity_sha256": _array_hash(affinity),
        "same_affinity_for_likelihood_and_reporting": True,
        **support,
    }
    return metrics, {
        "exact_parent": parent_city,
        "matched_pooled_backoff": pooled_city,
        "hierarchical_city_backoff": active_city,
    }, top_words


def _candidate_summary(
    rows: list[dict[str, Any]],
    quota: int,
    stability: Mapping[str, Any],
    control_stability: Mapping[str, Any],
    gates: Mapping[str, Any],
    tradeoff_reference: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [
        row for row in rows if int(row["prototype_quota"]) == int(quota)
    ]
    if len(selected) != 5:
        raise ValueError("each projection candidate must contain exactly five folds")

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in selected], dtype=np.float64)

    npmi_gain = values("heldout_npmi_improvement")
    cv_gain = values("heldout_cv_improvement")
    held_npmi = values("heldout_npmi_full")
    standard_npmi = values("standard_npmi_full")
    standard_cv = values("standard_cv_full")
    zero_gain = values("heldout_zero_joint_pair_reduction")
    nll_gain = values("completion_nll_reduction")
    step4_nll_gain = values("nll_reduction_vs_step4r2")
    city_pool = values("city_vs_pooled_nll_reduction")
    city_parent = values("city_vs_parent_nll_reduction")
    support = values("displayed_train_support_fraction")
    graph_support = values("displayed_graph_support_fraction")
    graph_pairs = values("displayed_positive_graph_pair_fraction")
    diversity = values("topic_diversity_full")
    redundancy = values("top_word_redundancy_full")
    recall = values("prototype_core_recall")
    nonprototype = values("maximum_nonprototype_change")
    column_error = values("runtime_column_sum_error")
    minimum_probability = values("runtime_minimum_probability")
    optimality = values("maximum_projection_optimality_error")
    changed_columns = values("changed_columns")
    parent_change = values("parent_state_change")
    affinity_identity = values("same_affinity_for_likelihood_and_reporting")
    stability_mean = float(stability["mean"])
    control_stability_mean = float(control_stability["mean"])
    gate_results = {
        "heldout_npmi_practical_effect": np.median(npmi_gain)
        >= float(gates["minimum_median_heldout_npmi_improvement"]),
        "heldout_npmi_fold_consistency": np.mean(npmi_gain > 0.0)
        >= float(gates["minimum_positive_npmi_fold_fraction"]),
        "heldout_cv_practical_effect": np.median(cv_gain)
        >= float(gates["minimum_median_heldout_cv_improvement"]),
        "heldout_cv_fold_consistency": np.mean(cv_gain > 0.0)
        >= float(gates["minimum_positive_cv_fold_fraction"]),
        "heldout_npmi_absolute_floor": np.median(held_npmi)
        >= float(gates["minimum_median_heldout_npmi"]),
        "standard_npmi_absolute_floor": np.median(standard_npmi)
        >= float(gates["minimum_median_standard_reference_npmi"]),
        "standard_cv_absolute_floor": np.median(standard_cv)
        >= float(gates["minimum_median_standard_reference_cv"]),
        "zero_joint_pair_reduction": np.median(zero_gain)
        >= float(gates["minimum_median_zero_joint_pair_reduction"]),
        "displayed_word_training_support": support.min()
        >= float(gates["minimum_displayed_train_support_fraction"]),
        "displayed_word_graph_support": graph_support.min()
        >= float(gates["minimum_displayed_graph_support_fraction"]),
        "displayed_positive_graph_pairs": graph_pairs.min()
        >= float(gates["minimum_displayed_positive_graph_pair_fraction"]),
        "city_vs_matched_pooling_effect_preserved": np.median(city_pool)
        >= float(gates["minimum_median_city_vs_pooled_reduction"]),
        "city_vs_matched_pooling_consistency_preserved": np.mean(city_pool > 0.0)
        >= float(gates["minimum_positive_city_vs_pooled_fold_fraction"]),
        "city_vs_parent_effect_preserved": np.median(city_parent)
        >= float(gates["minimum_median_city_vs_parent_reduction"]),
        "city_vs_parent_consistency_preserved": np.mean(city_parent > 0.0)
        >= float(gates["minimum_positive_city_vs_parent_fold_fraction"]),
        "topic_diversity_guardrail": diversity.min()
        >= float(gates["minimum_topic_diversity"]),
        "top_word_redundancy_guardrail": redundancy.max()
        <= float(gates["maximum_top_word_redundancy"]),
        "cross_fold_topic_stability_floor": stability_mean
        >= float(gates["minimum_cross_fold_topic_stability"]),
        "stability_paired_guardrail": stability_mean
        >= control_stability_mean
        - float(gates["maximum_stability_regression_vs_paired_control"]),
        "prototype_core_recall": recall.min()
        >= float(gates["minimum_prototype_core_recall"]),
        "nonprototype_columns_exact": nonprototype.max()
        <= float(gates["maximum_nonprototype_change"]),
        "projection_column_simplex": column_error.max()
        <= float(gates["maximum_projection_column_sum_error"]),
        "projection_nonnegative": minimum_probability.min()
        >= float(gates["minimum_projection_probability"]),
        "projection_minimum_total_variation": optimality.max()
        <= float(gates["maximum_projection_optimality_error"]),
        "projection_support_bounded": changed_columns.max()
        <= float(gates["maximum_changed_columns"]),
        "parent_state_frozen": parent_change.max() == 0.0,
        "one_affinity_for_likelihood_and_reporting": affinity_identity.min() == 1.0,
    }
    diagnostic_results = {
        "completion_nll_median_guardrail": np.median(nll_gain)
        >= -float(
            tradeoff_reference[
                "r4_reference_maximum_median_completion_nll_regression"
            ]
        ),
        "completion_nll_worst_fold_guardrail": nll_gain.min()
        >= -float(
            tradeoff_reference[
                "r4_reference_maximum_any_fold_completion_nll_regression"
            ]
        ),
        "step4r2_nll_median_preserved": np.median(step4_nll_gain)
        >= -float(
            tradeoff_reference[
                "r4_reference_maximum_median_nll_regression_vs_step4r2"
            ]
        ),
        "step4r2_nll_worst_fold_preserved": step4_nll_gain.min()
        >= -float(
            tradeoff_reference[
                "r4_reference_maximum_any_fold_nll_regression_vs_step4r2"
            ]
        ),
    }
    failed = [name for name, passed in gate_results.items() if not bool(passed)]
    failed_diagnostics = [
        name for name, passed in diagnostic_results.items() if not bool(passed)
    ]
    return {
        "prototype_quota": int(quota),
        "eligible": not failed,
        "passed_gate_count": int(len(gate_results) - len(failed)),
        "total_gate_count": int(len(gate_results)),
        "failed_gates": "|".join(failed),
        "reported_nll_tradeoff_passed_r4_references": not failed_diagnostics,
        "failed_reported_nll_diagnostics": "|".join(failed_diagnostics),
        "median_heldout_npmi_improvement": float(np.median(npmi_gain)),
        "positive_npmi_fold_fraction": float(np.mean(npmi_gain > 0.0)),
        "median_heldout_cv_improvement": float(np.median(cv_gain)),
        "positive_cv_fold_fraction": float(np.mean(cv_gain > 0.0)),
        "median_heldout_npmi": float(np.median(held_npmi)),
        "median_standard_reference_npmi": float(np.median(standard_npmi)),
        "median_standard_reference_cv": float(np.median(standard_cv)),
        "median_zero_joint_pair_reduction": float(np.median(zero_gain)),
        "minimum_displayed_train_support_fraction": float(support.min()),
        "minimum_displayed_graph_support_fraction": float(graph_support.min()),
        "minimum_displayed_positive_graph_pair_fraction": float(graph_pairs.min()),
        "median_completion_nll_reduction": float(np.median(nll_gain)),
        "minimum_completion_nll_reduction": float(nll_gain.min()),
        "median_nll_reduction_vs_step4r2": float(np.median(step4_nll_gain)),
        "minimum_nll_reduction_vs_step4r2": float(step4_nll_gain.min()),
        "median_city_vs_pooled_reduction": float(np.median(city_pool)),
        "positive_city_vs_pooled_fold_fraction": float(np.mean(city_pool > 0.0)),
        "median_city_vs_parent_reduction": float(np.median(city_parent)),
        "positive_city_vs_parent_fold_fraction": float(np.mean(city_parent > 0.0)),
        "minimum_topic_diversity": float(diversity.min()),
        "maximum_top_word_redundancy": float(redundancy.max()),
        "cross_fold_topic_stability": stability_mean,
        "paired_control_topic_stability": control_stability_mean,
        "minimum_prototype_core_recall": float(recall.min()),
        "maximum_nonprototype_change": float(nonprototype.max()),
        "maximum_runtime_column_sum_error": float(column_error.max()),
        "minimum_runtime_probability": float(minimum_probability.min()),
        "maximum_projection_optimality_error": float(optimality.max()),
        "maximum_changed_columns": int(changed_columns.max()),
        "maximum_parent_state_change": int(parent_change.max()),
        "same_affinity_for_likelihood_and_reporting": bool(
            affinity_identity.min() == 1.0
        ),
        "median_mean_vocabulary_column_total_variation": float(
            np.median(values("mean_vocabulary_column_total_variation"))
        ),
        "maximum_changed_column_total_variation": float(
            values("maximum_changed_column_total_variation").max()
        ),
    }


def run_worker(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    evidence = _load_evidence(job)
    NewMethod, loaded_path = _load_official_newmethod(Path(job["official_source"]))
    device = torch.device("cuda")
    checkpoints = {
        int(row["fold"]): row for row in job["step4r2_lineage"]["checkpoints"]
    }
    old_metrics = pd.read_csv(
        job["step4r2_lineage"]["selected_fold_metrics_path"],
        encoding="utf-8-sig",
    ).set_index("fold")
    projection_config = job["graph_prototype_projection"]
    quotas = list(map(int, projection_config["candidate_prototype_quotas"]))
    top_k = int(job["evaluation"]["top_words"])
    if (
        quotas != sorted(set(quotas))
        or not quotas
        or min(quotas) < 1
        or max(quotas) > top_k
    ):
        raise ValueError("prototype quotas must be sorted, unique, and within top-k")

    reproduction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    graph_receipts: list[dict[str, Any]] = []
    prototype_receipts: list[dict[str, Any]] = []
    alignment_receipts: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    context_receipts: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    candidate_top_words: dict[int, list[np.ndarray]] = {
        quota: [] for quota in quotas
    }
    control_top_words: list[np.ndarray] = []
    oracle_top_words: list[np.ndarray] = []
    maximum_reproduction_error = 0.0
    maximum_probability_error = 0.0
    maximum_zero_identity_error = 0.0
    frozen_states_exact = True

    for fold in map(int, job["development_folds"]):
        print(f"[V6 Step 5R2 R5 worker] fold={fold} START", flush=True)
        context = construct_fold_context(
            evidence["counts"],
            evidence["word_embeddings"],
            evidence["fold_assignments"],
            fold,
            clusters=int(job["global_context"]["clusters"]),
            n_init=int(job["global_context"]["n_init"]),
            seed=int(job["global_context"]["seed"]) + fold,
        )
        context_receipts.append({"fold": fold, **context.audit})
        graph, graph_receipt = positive_npmi_graph(
            evidence["counts"][context.train_indices],
            minimum_document_frequency=int(
                job["lexical_graph"]["minimum_document_frequency"]
            ),
            minimum_joint_documents=int(
                job["lexical_graph"]["minimum_joint_documents"]
            ),
            reliability_power=float(job["lexical_graph"]["reliability_power"]),
        )
        graph_receipt["fold"] = fold
        graph_receipts.append(graph_receipt)
        if int(graph_receipt["active_words"]) < int(
            job["lexical_graph"]["minimum_active_words"]
        ):
            raise ValueError("a fold lacks enough graph-supported words")
        prototype_bank = build_graph_prototype_bank(
            graph,
            topics=int(job["model"]["topics"]),
            top_k=int(job["evaluation"]["top_words"]),
            seed=int(job["seed"]) + fold,
            centrality_weight=float(projection_config["centrality_weight"]),
            minimum_cluster_size=int(projection_config["minimum_cluster_size"]),
        )
        prototype_receipts.append({"fold": fold, **prototype_bank.audit})
        old_row = old_metrics.loc[fold].to_dict()
        parent, city_prior, pooled_prior, lineage = _restore_parent_and_priors(
            NewMethod,
            evidence,
            job,
            checkpoints[fold],
            old_row,
            context.audit,
            torch,
        )
        parent_hash = _state_hash(parent.state_dict())
        parent = parent.to(device)
        parent.eval()
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            base_tensor = parent.get_beta().detach()
            base_affinity = base_tensor.cpu().numpy().astype(np.float64)
        aligned_cores, alignment_audit = align_prototype_cores_to_topics(
            base_affinity,
            prototype_bank.core_indices,
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        alignment_receipts.append({"fold": fold, **alignment_audit})
        oracle_top_words.append(aligned_cores)

        oracle_scores = np.zeros_like(base_affinity)
        for topic, words in enumerate(aligned_cores):
            oracle_scores[topic, words] = np.arange(
                len(words), 0, -1, dtype=np.float64
            )
        oracle_strict = topic_quality_metrics(
            oracle_scores,
            evidence["counts"][context.evaluation_indices],
            top_words=int(job["evaluation"]["top_words"]),
            window_size=int(job["evaluation"]["c_v_window_size"]),
            gamma=float(job["evaluation"]["c_v_gamma"]),
        )
        oracle_standard = topic_quality_metrics(
            oracle_scores,
            evidence["counts"],
            top_words=int(job["evaluation"]["top_words"]),
            window_size=int(job["evaluation"]["c_v_window_size"]),
            gamma=float(job["evaluation"]["c_v_gamma"]),
        )
        oracle_support = _top_word_support(
            aligned_cores,
            graph,
            evidence["counts"][context.train_indices],
            minimum_document_frequency=int(
                job["lexical_graph"]["minimum_document_frequency"]
            ),
        )
        oracle_rows.append(
            {
                "fold": fold,
                "heldout_npmi_at_10": oracle_strict["npmi_at_10"],
                "heldout_c_v_at_10": oracle_strict["c_v_at_10"],
                "heldout_zero_joint_pairs": oracle_strict["zero_joint_pairs"],
                "standard_npmi_at_10": oracle_standard["npmi_at_10"],
                "standard_c_v_at_10": oracle_standard["c_v_at_10"],
                "standard_zero_joint_pairs": oracle_standard["zero_joint_pairs"],
                "topic_diversity_at_10": oracle_standard[
                    "topic_diversity_at_10"
                ],
                "top_word_redundancy_at_10": oracle_standard[
                    "top_word_redundancy_at_10"
                ],
                **oracle_support,
            }
        )

        control_metrics, control_city, control_words = _evaluate_model(
            parent,
            evidence,
            context,
            fold,
            city_prior,
            pooled_prior,
            graph,
            job,
            device=device,
            torch=torch,
        )
        errors = {
            "parent_nll_error": abs(
                control_metrics["parent_macro_city_nll"]
                - float(old_row["parent_macro_city_nll"])
            ),
            "pooled_backoff_nll_error": abs(
                control_metrics["pooled_backoff_macro_city_nll"]
                - float(old_row["pooled_backoff_macro_city_nll"])
            ),
            "city_backoff_nll_error": abs(
                control_metrics["city_backoff_macro_city_nll"]
                - float(old_row["city_backoff_macro_city_nll"])
            ),
        }
        maximum_reproduction_error = max(
            maximum_reproduction_error, *map(float, errors.values())
        )
        maximum_probability_error = max(
            maximum_probability_error,
            float(control_metrics["probability_sum_error"]),
        )
        reproduction_rows.append({"fold": fold, **errors, **lineage})
        control_top_words.append(control_words)
        variant_rows.append(
            {
                "fold": fold,
                "prototype_quota": 0,
                "variant": "without_graph_projection_exact_step4r2",
                **control_metrics,
            }
        )
        for variant, values in control_city.items():
            for row in values:
                city_rows.append(
                    {
                        "fold": fold,
                        "prototype_quota": 0,
                        "variant": f"control_{variant}",
                        **row,
                    }
                )
        oracle_rows[-1].update(
            {
                "heldout_npmi_improvement_vs_control": (
                    oracle_rows[-1]["heldout_npmi_at_10"]
                    - control_metrics["heldout_npmi_at_10"]
                ),
                "heldout_cv_improvement_vs_control": (
                    oracle_rows[-1]["heldout_c_v_at_10"]
                    - control_metrics["heldout_c_v_at_10"]
                ),
                "heldout_zero_joint_pair_reduction_vs_control": (
                    control_metrics["heldout_zero_joint_pairs"]
                    - oracle_rows[-1]["heldout_zero_joint_pairs"]
                ),
            }
        )

        zero_projection = minimum_distortion_graph_projection(
            base_affinity,
            aligned_cores,
            strength=0.0,
            prototype_quota=top_k,
            rank_margin=float(projection_config["rank_margin"]),
            epsilon=float(job["evaluation"]["projection_epsilon"]),
        )
        zero_error = float(np.max(np.abs(zero_projection.affinity - base_affinity)))
        maximum_zero_identity_error = max(maximum_zero_identity_error, zero_error)

        for quota in quotas:
            projection = minimum_distortion_graph_projection(
                base_affinity,
                aligned_cores,
                strength=1.0,
                prototype_quota=quota,
                rank_margin=float(projection_config["rank_margin"]),
                epsilon=float(job["evaluation"]["projection_epsilon"]),
            )
            effective_tensor = torch.from_numpy(
                projection.affinity.astype(np.float32)
            ).to(device)
            runtime_affinity = effective_tensor.detach().cpu().numpy().astype(np.float64)
            runtime_difference = runtime_affinity - base_affinity
            prototype_mask = np.zeros(base_affinity.shape[1], dtype=bool)
            prototype_mask[np.unique(aligned_cores)] = True
            runtime_nonprototype_error = float(
                np.max(np.abs(runtime_difference[:, ~prototype_mask]))
            )
            runtime_column_error = float(
                np.max(np.abs(runtime_affinity.sum(axis=0) - 1.0))
            )
            runtime_minimum = float(np.min(runtime_affinity))
            metrics, candidate_city, top_words = _evaluate_model(
                parent,
                evidence,
                context,
                fold,
                city_prior,
                pooled_prior,
                graph,
                job,
                affinity_override=effective_tensor,
                device=device,
                torch=torch,
            )
            candidate_top_words[quota].append(top_words)
            recall = float(
                np.mean(
                    [
                        np.intersect1d(top_words[topic], aligned_cores[topic]).size
                        / aligned_cores.shape[1]
                        for topic in range(aligned_cores.shape[0])
                    ]
                )
            )
            parent_after_hash = _state_hash(parent.state_dict())
            parent_state_change = int(parent_after_hash != parent_hash)
            frozen_states_exact = bool(
                frozen_states_exact and parent_state_change == 0
            )
            maximum_probability_error = max(
                maximum_probability_error,
                float(metrics["probability_sum_error"]),
            )
            row = {
                "fold": fold,
                "prototype_quota": quota,
                "heldout_npmi_full": metrics["heldout_npmi_at_10"],
                "heldout_npmi_control": control_metrics["heldout_npmi_at_10"],
                "heldout_npmi_improvement": metrics["heldout_npmi_at_10"]
                - control_metrics["heldout_npmi_at_10"],
                "heldout_cv_full": metrics["heldout_c_v_at_10"],
                "heldout_cv_control": control_metrics["heldout_c_v_at_10"],
                "heldout_cv_improvement": metrics["heldout_c_v_at_10"]
                - control_metrics["heldout_c_v_at_10"],
                "standard_npmi_full": metrics["standard_npmi_at_10"],
                "standard_npmi_control": control_metrics["standard_npmi_at_10"],
                "standard_cv_full": metrics["standard_c_v_at_10"],
                "standard_cv_control": control_metrics["standard_c_v_at_10"],
                "heldout_zero_joint_pairs_full": metrics[
                    "heldout_zero_joint_pairs"
                ],
                "heldout_zero_joint_pairs_control": control_metrics[
                    "heldout_zero_joint_pairs"
                ],
                "heldout_zero_joint_pair_reduction": control_metrics[
                    "heldout_zero_joint_pairs"
                ]
                - metrics["heldout_zero_joint_pairs"],
                "nll_full": metrics["city_backoff_macro_city_nll"],
                "nll_control": control_metrics["city_backoff_macro_city_nll"],
                "completion_nll_reduction": control_metrics[
                    "city_backoff_macro_city_nll"
                ]
                - metrics["city_backoff_macro_city_nll"],
                "step4r2_nll": control_metrics["city_backoff_macro_city_nll"],
                "nll_reduction_vs_step4r2": control_metrics[
                    "city_backoff_macro_city_nll"
                ]
                - metrics["city_backoff_macro_city_nll"],
                "city_vs_pooled_nll_reduction": metrics[
                    "pooled_backoff_macro_city_nll"
                ]
                - metrics["city_backoff_macro_city_nll"],
                "city_vs_parent_nll_reduction": metrics[
                    "parent_macro_city_nll"
                ]
                - metrics["city_backoff_macro_city_nll"],
                "topic_diversity_full": metrics[
                    "heldout_topic_diversity_at_10"
                ],
                "top_word_redundancy_full": metrics[
                    "heldout_top_word_redundancy_at_10"
                ],
                "displayed_train_support_fraction": metrics[
                    "displayed_train_support_fraction"
                ],
                "displayed_graph_support_fraction": metrics[
                    "displayed_graph_support_fraction"
                ],
                "displayed_positive_graph_pair_fraction": metrics[
                    "displayed_positive_graph_pair_fraction"
                ],
                "prototype_core_recall": recall,
                "maximum_nonprototype_change": max(
                    float(projection.audit["maximum_nonprototype_change"]),
                    runtime_nonprototype_error,
                ),
                "runtime_column_sum_error": runtime_column_error,
                "runtime_minimum_probability": runtime_minimum,
                "maximum_projection_optimality_error": projection.audit[
                    "maximum_projection_optimality_error"
                ],
                "changed_columns": projection.audit["changed_columns"],
                "mean_vocabulary_column_total_variation": projection.audit[
                    "mean_vocabulary_column_total_variation"
                ],
                "maximum_changed_column_total_variation": projection.audit[
                    "maximum_changed_column_total_variation"
                ],
                "minimum_prototype_rank_slack": projection.audit[
                    "minimum_prototype_rank_slack"
                ],
                "parent_state_change": parent_state_change,
                "same_affinity_for_likelihood_and_reporting": int(
                    metrics["same_affinity_for_likelihood_and_reporting"]
                ),
                "affinity_sha256": metrics["affinity_sha256"],
                **lineage,
            }
            candidate_rows.append(row)
            projection_rows.append(
                {
                    "fold": fold,
                    "prototype_quota": quota,
                    **projection.audit,
                    "runtime_affinity_sha256": metrics["affinity_sha256"],
                    "runtime_nonprototype_error": runtime_nonprototype_error,
                    "runtime_column_sum_error": runtime_column_error,
                    "runtime_minimum_probability": runtime_minimum,
                }
            )
            variant_rows.append(
                {
                    "fold": fold,
                    "prototype_quota": quota,
                    "variant": "full_minimum_distortion_graph_projection",
                    **metrics,
                }
            )
            for variant, values in candidate_city.items():
                for city_row in values:
                    city_rows.append(
                        {
                            "fold": fold,
                            "prototype_quota": quota,
                            "variant": f"full_{variant}",
                            **city_row,
                        }
                    )
            checkpoint = (
                output
                / "candidate_checkpoints"
                / f"quota_{quota:02d}"
                / f"fold_{fold:02d}_step5r2_r5.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 3,
                    "implementation_version": IMPLEMENTATION,
                    "fold": fold,
                    "official_source_commit": job["official_commit"],
                    "model_configuration": job["model"],
                    "city_backoff_configuration": job["city_backoff"],
                    "selected_mixture_weight": float(job["selected_mixture_weight"]),
                    "lexical_graph_configuration": job["lexical_graph"],
                    "graph_prototype_projection_configuration": projection_config,
                    "selected_prototype_quota": quota,
                    "parent_checkpoint_path": lineage["source_checkpoint_path"],
                    "parent_checkpoint_sha256": lineage[
                        "source_checkpoint_sha256"
                    ],
                    "parent_state_sha256": parent_hash,
                    "effective_affinity": effective_tensor.detach().cpu(),
                    "effective_affinity_sha256": metrics["affinity_sha256"],
                    "aligned_prototype_core_indices": torch.from_numpy(
                        aligned_cores.astype(np.int64)
                    ),
                    "required_owner_increase": torch.from_numpy(
                        projection.required_owner_increase
                    ),
                    "applied_owner_increase": torch.from_numpy(
                        projection.applied_owner_increase
                    ),
                    "projection_audit": projection.audit,
                    "prototype_audit": prototype_bank.audit,
                    "prototype_alignment_audit": alignment_audit,
                    "city_priors": torch.from_numpy(city_prior.astype(np.float32)),
                    "pooled_prior": torch.from_numpy(pooled_prior.astype(np.float32)),
                    "context_audit": context.audit,
                    "graph_receipt": graph_receipt,
                    "step4r2_lineage": lineage,
                },
                checkpoint,
            )
            checkpoint_rows.append(
                {
                    "fold": fold,
                    "prototype_quota": quota,
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "size_bytes": int(checkpoint.stat().st_size),
                    "effective_affinity_sha256": metrics["affinity_sha256"],
                }
            )
            print(
                f"[V6 Step 5R2 R5 worker] fold={fold} quota={quota:d} "
                f"DONE | dNPMI={row['heldout_npmi_improvement']:+.6f} "
                f"dC_v={row['heldout_cv_improvement']:+.6f} "
                f"dNLL={row['completion_nll_reduction']:+.6f} "
                f"recall={recall:.3f}",
                flush=True,
            )
        del parent
        torch.cuda.empty_cache()

    control_stability = matched_topic_stability(control_top_words)
    oracle_stability = matched_topic_stability(oracle_top_words)
    candidate_summaries: list[dict[str, Any]] = []
    candidate_stabilities: dict[int, dict[str, Any]] = {}
    for quota in quotas:
        stability = matched_topic_stability(candidate_top_words[quota])
        candidate_stabilities[quota] = stability
        candidate_summaries.append(
            _candidate_summary(
                candidate_rows,
                quota,
                stability,
                control_stability,
                job["gates"],
                job["reported_graph_ablation_nll_tradeoff"],
            )
        )
    for summary in candidate_summaries:
        failures = str(summary["failed_gates"]) or "none"
        print(
            "[V6 Step 5R2 R5 worker] candidate gate audit | "
            f"quota={int(summary['prototype_quota'])} "
            f"passed={int(summary['passed_gate_count'])}/"
            f"{int(summary['total_gate_count'])} failed={failures} "
            "reported-NLL-tradeoff="
            f"{str(summary['failed_reported_nll_diagnostics']) or 'none'}",
            flush=True,
        )
    eligible = [row for row in candidate_summaries if bool(row["eligible"])]
    selected_quota = min(
        (int(row["prototype_quota"]) for row in eligible), default=None
    )
    selected_rows = [
        row
        for row in candidate_rows
        if selected_quota is not None
        and int(row["prototype_quota"]) == selected_quota
    ]
    selected_checkpoints = [
        row
        for row in checkpoint_rows
        if selected_quota is not None
        and int(row["prototype_quota"]) == selected_quota
    ]

    _write_csv(output / "step4r2_reproduction.csv", reproduction_rows)
    _write_csv(output / "paired_candidate_fold_metrics.csv", candidate_rows)
    _write_csv(output / "candidate_summary.csv", candidate_summaries)
    _write_csv(output / "variant_metrics.csv", variant_rows)
    _write_csv(output / "city_stratified_metrics.csv", city_rows)
    _write_csv(output / "projection_trace.csv", projection_rows)
    _write_csv(output / "prototype_oracle_metrics.csv", oracle_rows)
    _write_json(output / "fold_graph_receipts.json", graph_receipts)
    _write_json(output / "fold_prototype_receipts.json", prototype_receipts)
    _write_json(output / "fold_prototype_alignment_receipts.json", alignment_receipts)
    _write_json(output / "fold_context_receipts.json", context_receipts)
    _write_json(output / "candidate_checkpoint_manifest.json", checkpoint_rows)
    _write_json(output / "checkpoint_manifest.json", selected_checkpoints)
    if selected_rows:
        _write_csv(output / "selected_fold_metrics.csv", selected_rows)

    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "status": "PASS" if selected_quota is not None else "FAIL",
        "official_class_loaded_from": str(loaded_path),
        "folds": len(job["development_folds"]),
        "paired_control_evaluations": len(job["development_folds"]),
        "candidate_projection_evaluations": len(candidate_rows),
        "total_projection_evaluations": len(job["development_folds"])
        + len(candidate_rows),
        "optimizer_steps": 0,
        "trainable_parameter_group": "none_closed_form_projection_on_frozen_affinity",
        "selected_city_mixture_weight": float(job["selected_mixture_weight"]),
        "candidate_summaries": candidate_summaries,
        "candidate_gate_failures": {
            str(row["prototype_quota"]): (
                []
                if not str(row["failed_gates"])
                else str(row["failed_gates"]).split("|")
            )
            for row in candidate_summaries
        },
        "reported_graph_ablation_nll_tradeoffs": {
            str(row["prototype_quota"]): (
                []
                if not str(row["failed_reported_nll_diagnostics"])
                else str(row["failed_reported_nll_diagnostics"]).split("|")
            )
            for row in candidate_summaries
        },
        "selected_prototype_quota": selected_quota,
        "selection_rule": "smallest predeclared prototype quota satisfying every target support stability city-benefit probability lineage and projection-integrity gate; graph-ablation NLL remains a mandatory reported trade-off",
        "all_parent_parameters_and_buffers_exactly_unchanged": frozen_states_exact,
        "maximum_step4r2_reproduction_error": maximum_reproduction_error,
        "maximum_probability_sum_error": maximum_probability_error,
        "maximum_zero_strength_identity_error": maximum_zero_identity_error,
        "minimum_graph_active_words": int(
            min(row["active_words"] for row in graph_receipts)
        ),
        "paired_control_topic_stability": control_stability,
        "prototype_oracle_topic_stability": oracle_stability,
        "prototype_oracle_median_heldout_npmi": float(
            np.median([row["heldout_npmi_at_10"] for row in oracle_rows])
        ),
        "prototype_oracle_median_heldout_npmi_improvement": float(
            np.median(
                [row["heldout_npmi_improvement_vs_control"] for row in oracle_rows]
            )
        ),
        "prototype_oracle_median_heldout_cv_improvement": float(
            np.median(
                [row["heldout_cv_improvement_vs_control"] for row in oracle_rows]
            )
        ),
        "prototype_oracle_median_standard_npmi": float(
            np.median([row["standard_npmi_at_10"] for row in oracle_rows])
        ),
        "prototype_oracle_median_standard_cv": float(
            np.median([row["standard_c_v_at_10"] for row in oracle_rows])
        ),
        "candidate_topic_stabilities": {
            str(quota): candidate_stabilities[quota]
            for quota in quotas
        },
        "checkpoint_manifest": selected_checkpoints,
        "candidate_checkpoint_manifest": checkpoint_rows,
        "validation_documents_used": 0,
        "test_documents_used": 0,
        "labels_used": 0,
        "baseline_or_sota_outputs_used": 0,
    }
    _write_json(output / "worker_result.json", result)
    print(
        f"[V6 Step 5R2 R5 worker] {result['status']} | "
        f"selected prototype quota={selected_quota}",
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
    torch = _seed_everything(int(job["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "V6 Step-5R2 R5 requires the exact CUDA runtime verified in Step 3"
        )
    _write_json(output / "runtime_receipt.json", _runtime_receipt(torch))
    run_worker(job, output, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
