from copy import deepcopy

from src.v6.recovery_gate import (
    select_recovery_candidate,
    summarize_recovery_candidate,
)


GATES = {
    "expected_seed_count": 3,
    "minimum_positive_seed_fraction": 2.0 / 3.0,
    "minimum_median_validation_npmi_improvement": 0.05,
    "minimum_median_validation_cv_improvement": 0.02,
    "minimum_median_pooled_backoff_reduction": 0.05,
    "minimum_median_full_vs_parent_reduction": 0.05,
    "minimum_median_training_reference_npmi": -0.05,
    "minimum_median_training_reference_cv": 0.48,
    "minimum_validation_topic_diversity": 0.85,
    "maximum_validation_top_word_redundancy": 0.15,
    "minimum_full_topic_stability": 0.2,
    "maximum_stability_regression_vs_graph_removed": 0.02,
    "minimum_displayed_train_support_fraction": 0.85,
    "minimum_displayed_graph_support_fraction": 0.8,
    "minimum_displayed_positive_graph_pair_fraction": 0.15,
    "maximum_probability_sum_error": 1.0e-6,
    "maximum_nonprototype_change": 0.0,
    "maximum_projection_column_sum_error": 1.0e-6,
    "minimum_projection_probability": 0.0,
    "maximum_projection_optimality_error": 1.0e-12,
    "maximum_calibration_unchanged_identity_error": 0.0,
    "maximum_calibration_moment_error": 1.0e-12,
    "maximum_calibration_scale": 10.0,
}


def _rows(candidate="quota_09__native_decoder", parent_effect=0.10):
    result = []
    for index, seed in enumerate((20260910, 20270910, 20280910)):
        result.append(
            {
                "candidate_id": candidate,
                "seed": seed,
                "prototype_quota": 9,
                "calibration": "native_decoder",
                "validation_npmi_improvement_vs_without_graph": 0.08 + 0.01 * index,
                "validation_cv_improvement_vs_without_graph": 0.05 + 0.01 * index,
                "full_nll_reduction_vs_official_parent": parent_effect,
                "pooled_nll_reduction_vs_without_backoff": 0.20,
                "graph_nll_reduction_vs_without_graph": -0.04,
                "calibration_nll_reduction_vs_native": 0.0,
                "training_reference_npmi_full": 0.03,
                "training_reference_cv_full": 0.53,
                "validation_topic_diversity_full": 1.0,
                "validation_top_word_redundancy_full": 0.0,
                "prototype_core_recall": 0.9,
                "expected_prototype_core_recall": 0.9,
                "maximum_probability_sum_error": 1.0e-7,
                "maximum_nonprototype_change": 0.0,
                "projection_column_sum_error": 1.0e-7,
                "projection_minimum_probability": 0.001,
                "projection_optimality_error": 1.0e-16,
                "calibration_unchanged_identity_error": 0.0,
                "calibration_mean_error": 0.0,
                "calibration_variance_error": 0.0,
                "calibration_maximum_scale": 1.0,
                "displayed_train_support_fraction": 1.0,
                "displayed_graph_support_fraction": 1.0,
                "displayed_positive_graph_pair_fraction": 0.7,
                "parent_state_change": 0,
                "mean_vocabulary_column_total_variation": 0.02,
            }
        )
    return result


def _summarize(rows):
    return summarize_recovery_candidate(
        rows,
        gates=GATES,
        topic_stability={"mean": 0.8, "minimum": 0.7, "pairs": 3},
        graph_removed_topic_stability={"mean": 0.3, "minimum": 0.2, "pairs": 3},
    )


def test_recovery_candidate_passes_only_target_aligned_gates():
    summary = _summarize(_rows())
    assert summary["eligible"] is True
    assert summary["failed_gates"] == []
    assert summary["median_graph_nll_reduction_vs_without_graph"] < 0.0


def test_small_full_parent_effect_remains_a_failure():
    summary = _summarize(_rows(parent_effect=0.01))
    assert summary["eligible"] is False
    assert "complete_model_vs_parent_practical_effect" in summary["failed_gates"]


def test_selection_uses_robust_normalized_margin_then_distortion():
    weaker = _summarize(_rows(candidate="weaker", parent_effect=0.08))
    stronger_rows = deepcopy(_rows(candidate="stronger", parent_effect=0.14))
    for row in stronger_rows:
        row["mean_vocabulary_column_total_variation"] = 0.03
    stronger = _summarize(stronger_rows)
    selected = select_recovery_candidate([weaker, stronger])
    assert selected is not None
    assert selected["candidate_id"] == "stronger"


def test_no_candidate_is_selected_when_central_gate_fails():
    failed = _summarize(_rows(parent_effect=0.01))
    assert select_recovery_candidate([failed]) is None
