"""Pure aggregation logic for the frozen V6 one-shot validation gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError(f"validation metric {key} is empty or non-finite")
    return values


def summarize_validation_seeds(
    rows: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    full_topic_stability: Mapping[str, Any],
    graph_removed_topic_stability: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply target-aligned gates without choosing a seed or hyperparameter."""

    expected = int(gates["expected_seed_count"])
    if len(rows) != expected:
        raise ValueError(
            f"expected {expected} frozen seed rows, received {len(rows)}"
        )
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != expected:
        raise ValueError("validation seed identities are not unique")

    graph_npmi = _values(rows, "validation_npmi_improvement_vs_without_graph")
    graph_cv = _values(rows, "validation_cv_improvement_vs_without_graph")
    zero_pairs = _values(
        rows, "validation_zero_joint_pair_reduction_vs_without_graph"
    )
    city_no_city = _values(rows, "city_nll_reduction_vs_without_city")
    city_pooling = _values(rows, "city_nll_reduction_vs_matched_pooling")
    full_parent = _values(rows, "full_nll_reduction_vs_official_parent")
    graph_nll = _values(rows, "graph_nll_reduction_vs_without_graph")
    diversity = _values(rows, "validation_topic_diversity_full")
    redundancy = _values(rows, "validation_top_word_redundancy_full")
    standard_npmi = _values(rows, "training_reference_npmi_full")
    standard_cv = _values(rows, "training_reference_cv_full")
    train_support = _values(rows, "displayed_train_support_fraction")
    graph_support = _values(rows, "displayed_graph_support_fraction")
    graph_pairs = _values(rows, "displayed_positive_graph_pair_fraction")
    recall = _values(rows, "prototype_core_recall")
    probability_error = _values(rows, "maximum_probability_sum_error")
    nonprototype = _values(rows, "maximum_nonprototype_change")
    column_error = _values(rows, "projection_column_sum_error")
    minimum_probability = _values(rows, "projection_minimum_probability")
    optimality = _values(rows, "projection_optimality_error")
    changed_columns = _values(rows, "changed_columns")
    parent_change = _values(rows, "parent_state_change")
    affinity_identity = _values(
        rows, "same_affinity_for_likelihood_and_reporting"
    )
    training_loss_reduction = _values(rows, "parent_training_loss_reduction")

    positive_fraction = float(gates["minimum_positive_seed_fraction"])
    full_stability = float(full_topic_stability["mean"])
    control_stability = float(graph_removed_topic_stability["mean"])
    gate_results = {
        "graph_validation_npmi_practical_effect": float(np.median(graph_npmi))
        >= float(gates["minimum_median_validation_npmi_improvement"]),
        "graph_validation_npmi_seed_consistency": float(np.mean(graph_npmi > 0.0))
        >= positive_fraction,
        "graph_validation_cv_practical_effect": float(np.median(graph_cv))
        >= float(gates["minimum_median_validation_cv_improvement"]),
        "graph_validation_cv_seed_consistency": float(np.mean(graph_cv > 0.0))
        >= positive_fraction,
        "graph_validation_zero_joint_pairs_noninferior": float(
            np.median(zero_pairs)
        )
        >= float(gates["minimum_median_zero_joint_pair_reduction"]),
        "city_validation_nll_practical_effect": float(np.median(city_no_city))
        >= float(gates["minimum_median_city_vs_without_city_reduction"]),
        "city_validation_nll_seed_consistency": float(
            np.mean(city_no_city > 0.0)
        )
        >= positive_fraction,
        "city_vs_matched_pooling_practical_effect": float(
            np.median(city_pooling)
        )
        >= float(gates["minimum_median_city_vs_pooled_reduction"]),
        "city_vs_matched_pooling_seed_consistency": float(
            np.mean(city_pooling > 0.0)
        )
        >= positive_fraction,
        "complete_model_vs_official_parent_practical_effect": float(
            np.median(full_parent)
        )
        >= float(gates["minimum_median_full_vs_parent_reduction"]),
        "complete_model_vs_official_parent_seed_consistency": float(
            np.mean(full_parent > 0.0)
        )
        >= positive_fraction,
        "training_reference_npmi_floor": float(np.median(standard_npmi))
        >= float(gates["minimum_median_training_reference_npmi"]),
        "training_reference_cv_floor": float(np.median(standard_cv))
        >= float(gates["minimum_median_training_reference_cv"]),
        "validation_topic_diversity_guardrail": float(diversity.min())
        >= float(gates["minimum_validation_topic_diversity"]),
        "validation_top_word_redundancy_guardrail": float(redundancy.max())
        <= float(gates["maximum_validation_top_word_redundancy"]),
        "full_topic_stability_floor": full_stability
        >= float(gates["minimum_full_topic_stability"]),
        "stability_paired_guardrail": full_stability
        >= control_stability
        - float(gates["maximum_stability_regression_vs_graph_removed"]),
        "displayed_word_training_support": float(train_support.min())
        >= float(gates["minimum_displayed_train_support_fraction"]),
        "displayed_word_graph_support": float(graph_support.min())
        >= float(gates["minimum_displayed_graph_support_fraction"]),
        "displayed_positive_graph_pairs": float(graph_pairs.min())
        >= float(gates["minimum_displayed_positive_graph_pair_fraction"]),
        "complete_prototype_core_recall": float(recall.min())
        >= float(gates["minimum_prototype_core_recall"]),
        "parent_training_loss_decreased_every_seed": bool(
            np.all(training_loss_reduction > 0.0)
        ),
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
        "projection_support_bounded": float(changed_columns.max())
        <= float(gates["maximum_changed_columns"]),
        "trained_parent_frozen_during_projection_and_validation": bool(
            np.all(parent_change == 0.0)
        ),
        "one_affinity_for_likelihood_and_reporting": bool(
            np.all(affinity_identity == 1.0)
        ),
    }
    failed = [name for name, passed in gate_results.items() if not bool(passed)]
    return {
        "schema_version": 1,
        "status": "PASS" if not failed else "FAIL",
        "seed_count": expected,
        "seed_selection_performed": False,
        "hyperparameter_selection_performed": False,
        "gate_results": gate_results,
        "failed_gates": failed,
        "passed_gate_count": int(len(gate_results) - len(failed)),
        "total_gate_count": int(len(gate_results)),
        "median_validation_npmi_improvement_vs_without_graph": float(
            np.median(graph_npmi)
        ),
        "positive_npmi_seed_fraction": float(np.mean(graph_npmi > 0.0)),
        "median_validation_cv_improvement_vs_without_graph": float(
            np.median(graph_cv)
        ),
        "positive_cv_seed_fraction": float(np.mean(graph_cv > 0.0)),
        "median_validation_zero_joint_pair_reduction": float(
            np.median(zero_pairs)
        ),
        "median_city_nll_reduction_vs_without_city": float(
            np.median(city_no_city)
        ),
        "positive_city_vs_without_city_seed_fraction": float(
            np.mean(city_no_city > 0.0)
        ),
        "median_city_nll_reduction_vs_matched_pooling": float(
            np.median(city_pooling)
        ),
        "positive_city_vs_pooling_seed_fraction": float(
            np.mean(city_pooling > 0.0)
        ),
        "median_full_nll_reduction_vs_official_parent": float(
            np.median(full_parent)
        ),
        "positive_full_vs_parent_seed_fraction": float(
            np.mean(full_parent > 0.0)
        ),
        "median_graph_nll_reduction_vs_without_graph": float(
            np.median(graph_nll)
        ),
        "graph_nll_tradeoff_is_report_only": True,
        "median_validation_npmi_full": float(
            np.median(_values(rows, "validation_npmi_full"))
        ),
        "median_validation_cv_full": float(
            np.median(_values(rows, "validation_cv_full"))
        ),
        "median_training_reference_npmi_full": float(np.median(standard_npmi)),
        "median_training_reference_cv_full": float(np.median(standard_cv)),
        "minimum_validation_topic_diversity": float(diversity.min()),
        "maximum_validation_top_word_redundancy": float(redundancy.max()),
        "full_topic_stability": full_topic_stability,
        "graph_removed_topic_stability": graph_removed_topic_stability,
    }
