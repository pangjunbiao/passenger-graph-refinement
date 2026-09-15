from copy import deepcopy

from src.v6.step6 import EXPECTED_GATES
from src.v6.validation_gate import summarize_validation_seeds


def _row(seed: int):
    return {
        "seed": seed,
        "validation_npmi_improvement_vs_without_graph": 0.08,
        "validation_cv_improvement_vs_without_graph": 0.04,
        "validation_zero_joint_pair_reduction_vs_without_graph": 12,
        "city_nll_reduction_vs_without_city": 0.09,
        "city_nll_reduction_vs_matched_pooling": 0.01,
        "full_nll_reduction_vs_official_parent": 0.12,
        # The graph NLL loss is deliberately negative and report-only.
        "graph_nll_reduction_vs_without_graph": -0.11,
        "validation_topic_diversity_full": 1.0,
        "validation_top_word_redundancy_full": 0.0,
        "training_reference_npmi_full": 0.12,
        "training_reference_cv_full": 0.56,
        "displayed_train_support_fraction": 1.0,
        "displayed_graph_support_fraction": 1.0,
        "displayed_positive_graph_pair_fraction": 0.60,
        "prototype_core_recall": 1.0,
        "maximum_probability_sum_error": 2.0e-7,
        "maximum_nonprototype_change": 0.0,
        "projection_column_sum_error": 2.0e-7,
        "projection_minimum_probability": 0.001,
        "projection_optimality_error": 1.0e-16,
        "changed_columns": 100,
        "parent_state_change": 0,
        "same_affinity_for_likelihood_and_reporting": 1,
        "parent_training_loss_reduction": 0.1,
        "validation_npmi_full": -0.70,
        "validation_cv_full": 0.45,
    }


def _summary(rows):
    return summarize_validation_seeds(
        rows,
        gates=EXPECTED_GATES,
        full_topic_stability={"pairs": 3, "mean": 0.7, "minimum": 0.6},
        graph_removed_topic_stability={
            "pairs": 3,
            "mean": 0.2,
            "minimum": 0.1,
        },
    )


def test_negative_graph_nll_tradeoff_does_not_fake_a_validation_failure():
    rows = [_row(seed) for seed in (20260910, 20270910, 20280910)]
    summary = _summary(rows)
    assert summary["status"] == "PASS"
    assert summary["median_graph_nll_reduction_vs_without_graph"] == -0.11
    assert summary["graph_nll_tradeoff_is_report_only"] is True


def test_graph_target_failure_blocks_validation_even_when_nll_is_good():
    rows = [_row(seed) for seed in (20260910, 20270910, 20280910)]
    for row in rows:
        row["validation_npmi_improvement_vs_without_graph"] = -0.01
        row["graph_nll_reduction_vs_without_graph"] = 0.5
    summary = _summary(rows)
    assert summary["status"] == "FAIL"
    assert "graph_validation_npmi_practical_effect" in summary["failed_gates"]
    assert "graph_validation_npmi_seed_consistency" in summary["failed_gates"]


def test_seed_selection_is_never_returned_by_the_aggregator():
    rows = [_row(seed) for seed in (20260910, 20270910, 20280910)]
    rows[0]["validation_cv_improvement_vs_without_graph"] = 0.01
    summary = _summary(deepcopy(rows))
    assert summary["seed_selection_performed"] is False
    assert "selected_seed" not in summary
