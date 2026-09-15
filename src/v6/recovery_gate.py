"""Pure aggregation and selection for the disclosed V6.1 recovery study."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    result = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if result.size < 1 or not np.isfinite(result).all():
        raise ValueError(f"candidate metric {key} is empty or non-finite")
    return result


def summarize_recovery_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    topic_stability: Mapping[str, Any],
    graph_removed_topic_stability: Mapping[str, Any],
) -> dict[str, Any]:
    expected = int(gates["expected_seed_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} seed rows, received {len(rows)}")
    identities = {str(row["candidate_id"]) for row in rows}
    if len(identities) != 1:
        raise ValueError("candidate rows contain multiple identities")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != expected:
        raise ValueError("candidate seed identities are not unique")

    d_npmi = _values(rows, "validation_npmi_improvement_vs_without_graph")
    d_cv = _values(rows, "validation_cv_improvement_vs_without_graph")
    d_parent = _values(rows, "full_nll_reduction_vs_official_parent")
    d_backoff = _values(rows, "pooled_nll_reduction_vs_without_backoff")
    d_graph_nll = _values(rows, "graph_nll_reduction_vs_without_graph")
    d_calibration = _values(rows, "calibration_nll_reduction_vs_native")
    train_npmi = _values(rows, "training_reference_npmi_full")
    train_cv = _values(rows, "training_reference_cv_full")
    diversity = _values(rows, "validation_topic_diversity_full")
    redundancy = _values(rows, "validation_top_word_redundancy_full")
    recall = _values(rows, "prototype_core_recall")
    expected_recall = _values(rows, "expected_prototype_core_recall")
    probability_error = _values(rows, "maximum_probability_sum_error")
    nonprototype = _values(rows, "maximum_nonprototype_change")
    column_error = _values(rows, "projection_column_sum_error")
    minimum_probability = _values(rows, "projection_minimum_probability")
    optimality = _values(rows, "projection_optimality_error")
    unchanged_calibration = _values(rows, "calibration_unchanged_identity_error")
    calibration_mean = _values(rows, "calibration_mean_error")
    calibration_variance = _values(rows, "calibration_variance_error")
    calibration_scale = _values(rows, "calibration_maximum_scale")
    support_train = _values(rows, "displayed_train_support_fraction")
    support_graph = _values(rows, "displayed_graph_support_fraction")
    support_pairs = _values(rows, "displayed_positive_graph_pair_fraction")
    parent_change = _values(rows, "parent_state_change")

    positive_fraction = float(gates["minimum_positive_seed_fraction"])
    full_stability = float(topic_stability["mean"])
    control_stability = float(graph_removed_topic_stability["mean"])
    gate_results = {
        "graph_npmi_practical_effect": float(np.median(d_npmi))
        >= float(gates["minimum_median_validation_npmi_improvement"]),
        "graph_npmi_seed_consistency": float(np.mean(d_npmi > 0.0))
        >= positive_fraction,
        "graph_cv_practical_effect": float(np.median(d_cv))
        >= float(gates["minimum_median_validation_cv_improvement"]),
        "graph_cv_seed_consistency": float(np.mean(d_cv > 0.0))
        >= positive_fraction,
        "pooled_backoff_practical_effect": float(np.median(d_backoff))
        >= float(gates["minimum_median_pooled_backoff_reduction"]),
        "pooled_backoff_seed_consistency": float(np.mean(d_backoff > 0.0))
        >= positive_fraction,
        "complete_model_vs_parent_practical_effect": float(np.median(d_parent))
        >= float(gates["minimum_median_full_vs_parent_reduction"]),
        "complete_model_vs_parent_seed_consistency": float(np.mean(d_parent > 0.0))
        >= positive_fraction,
        "training_reference_npmi_floor": float(np.median(train_npmi))
        >= float(gates["minimum_median_training_reference_npmi"]),
        "training_reference_cv_floor": float(np.median(train_cv))
        >= float(gates["minimum_median_training_reference_cv"]),
        "validation_topic_diversity_guardrail": float(diversity.min())
        >= float(gates["minimum_validation_topic_diversity"]),
        "validation_redundancy_guardrail": float(redundancy.max())
        <= float(gates["maximum_validation_top_word_redundancy"]),
        "topic_stability_floor": full_stability
        >= float(gates["minimum_full_topic_stability"]),
        "topic_stability_paired_guardrail": full_stability
        >= control_stability
        - float(gates["maximum_stability_regression_vs_graph_removed"]),
        "prototype_recall_meets_declared_quota": bool(
            np.all(recall + 1.0e-12 >= expected_recall)
        ),
        "displayed_word_training_support": float(support_train.min())
        >= float(gates["minimum_displayed_train_support_fraction"]),
        "displayed_word_graph_support": float(support_graph.min())
        >= float(gates["minimum_displayed_graph_support_fraction"]),
        "displayed_positive_graph_pairs": float(support_pairs.min())
        >= float(gates["minimum_displayed_positive_graph_pair_fraction"]),
        "probability_simplex": float(probability_error.max())
        <= float(gates["maximum_probability_sum_error"]),
        "nonprototype_columns_exact": float(nonprototype.max())
        <= float(gates["maximum_nonprototype_change"]),
        "projection_column_simplex": float(column_error.max())
        <= float(gates["maximum_projection_column_sum_error"]),
        "projection_nonnegative": float(minimum_probability.min())
        >= float(gates["minimum_projection_probability"]),
        "projection_minimum_total_variation": float(optimality.max())
        <= float(gates["maximum_projection_optimality_error"]),
        "calibration_unchanged_columns_exact": float(unchanged_calibration.max())
        <= float(gates["maximum_calibration_unchanged_identity_error"]),
        "calibration_training_mean_identity": float(calibration_mean.max())
        <= float(gates["maximum_calibration_moment_error"]),
        "calibration_training_variance_identity": float(calibration_variance.max())
        <= float(gates["maximum_calibration_moment_error"]),
        "calibration_scale_bounded": float(calibration_scale.max())
        <= float(gates["maximum_calibration_scale"]),
        "parent_state_frozen": bool(np.all(parent_change == 0.0)),
    }
    failed = [name for name, passed in gate_results.items() if not bool(passed)]
    threshold_margins = np.asarray(
        [
            float(np.median(d_npmi))
            / float(gates["minimum_median_validation_npmi_improvement"]),
            float(np.median(d_cv))
            / float(gates["minimum_median_validation_cv_improvement"]),
            float(np.median(d_backoff))
            / float(gates["minimum_median_pooled_backoff_reduction"]),
            float(np.median(d_parent))
            / float(gates["minimum_median_full_vs_parent_reduction"]),
        ],
        dtype=np.float64,
    )
    return {
        "schema_version": 1,
        "candidate_id": next(iter(identities)),
        "prototype_quota": int(rows[0]["prototype_quota"]),
        "calibration": str(rows[0]["calibration"]),
        "eligible": not failed,
        "failed_gates": failed,
        "passed_gate_count": int(len(gate_results) - len(failed)),
        "total_gate_count": int(len(gate_results)),
        "gate_results": gate_results,
        "selection_score_minimum_normalized_margin": float(threshold_margins.min()),
        "median_validation_npmi_improvement_vs_without_graph": float(np.median(d_npmi)),
        "positive_npmi_seed_fraction": float(np.mean(d_npmi > 0.0)),
        "median_validation_cv_improvement_vs_without_graph": float(np.median(d_cv)),
        "positive_cv_seed_fraction": float(np.mean(d_cv > 0.0)),
        "median_pooled_nll_reduction_vs_without_backoff": float(np.median(d_backoff)),
        "positive_pooled_backoff_seed_fraction": float(np.mean(d_backoff > 0.0)),
        "median_full_nll_reduction_vs_official_parent": float(np.median(d_parent)),
        "positive_full_vs_parent_seed_fraction": float(np.mean(d_parent > 0.0)),
        "median_graph_nll_reduction_vs_without_graph": float(np.median(d_graph_nll)),
        "median_calibration_nll_reduction_vs_native": float(np.median(d_calibration)),
        "median_training_reference_npmi_full": float(np.median(train_npmi)),
        "median_training_reference_cv_full": float(np.median(train_cv)),
        "minimum_validation_topic_diversity": float(diversity.min()),
        "maximum_validation_top_word_redundancy": float(redundancy.max()),
        "topic_stability": dict(topic_stability),
        "graph_removed_topic_stability": dict(graph_removed_topic_stability),
        "median_mean_vocabulary_column_total_variation": float(
            np.median(_values(rows, "mean_vocabulary_column_total_variation"))
        ),
    }


def select_recovery_candidate(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Choose the most robust eligible candidate with deterministic ties."""

    eligible = [dict(row) for row in summaries if bool(row["eligible"])]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            -float(row["selection_score_minimum_normalized_margin"]),
            -float(row["median_validation_npmi_improvement_vs_without_graph"]),
            float(row["median_mean_vocabulary_column_total_variation"]),
            str(row["candidate_id"]),
        )
    )
    return eligible[0]
