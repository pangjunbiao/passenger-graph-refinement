import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.v6.final_refit import (
    FINAL_VARIANTS,
    combine_development_partitions,
    final_variant_registry,
    validate_final_diagnostics,
)


def _rows(split: str, count: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "post_id": [f"{prefix}:{index}" for index in range(count)],
            "city": ["beijing"] * count,
            "split": [split] * count,
            "matrix_row": list(range(count)),
        }
    )


def test_final_fit_merge_is_stable_complete_and_audited():
    train = sparse.csr_matrix([[2, 0, 1], [0, 3, 0]])
    spent = sparse.csr_matrix([[1, 1, 0]])
    counts, rows, receipt = combine_development_partitions(
        train, _rows("train", 2, "t"), spent, _rows("validation", 1, "v")
    )
    np.testing.assert_array_equal(counts.toarray(), [[2, 0, 1], [0, 3, 0], [1, 1, 0]])
    assert rows["post_id"].tolist() == ["t:0", "t:1", "v:0"]
    assert rows["development_row"].tolist() == [0, 1, 2]
    assert receipt["final_fit_documents"] == 3
    assert receipt["final_fit_tokens"] == 8
    assert receipt["test_documents_used"] == 0


def test_final_fit_merge_rejects_overlap_and_wrong_split():
    train = sparse.csr_matrix([[1, 0]])
    spent = sparse.csr_matrix([[0, 1]])
    left = _rows("train", 1, "same")
    right = _rows("validation", 1, "same")
    with pytest.raises(ValueError, match="crosses"):
        combine_development_partitions(train, left, spent, right)
    with pytest.raises(ValueError, match="not validation"):
        combine_development_partitions(
            train, left, spent, _rows("test", 1, "different")
        )


def test_final_variant_registry_contains_exact_target_aligned_removals():
    registry = final_variant_registry(
        prototype_quota=10,
        pooled_mixture_weight=0.35,
        calibration="training_moment_match",
    )
    assert [row["variant"] for row in registry] == list(FINAL_VARIANTS)
    assert registry[1]["graph_transport"] == "off_exact_parent_affinity"
    assert registry[2]["pooled_mixture_weight"] == 0.0
    assert registry[3]["decoder_calibration"] == "identity_native_decoder"
    assert registry[4]["pooled_mixture_weight"] == 0.0
    assert all(row["test_status"] == "UNOPENED" for row in registry)


def test_final_variant_registry_rejects_post_selection_changes():
    with pytest.raises(ValueError, match="changed"):
        final_variant_registry(
            prototype_quota=9,
            pooled_mixture_weight=0.35,
            calibration="training_moment_match",
        )


def _diagnostic(seed: int) -> dict[str, float | int]:
    return {
        "seed": seed,
        "parent_training_loss_reduction": 1.0,
        "prototype_core_recall": 1.0,
        "projection_column_sum_error": 1e-8,
        "projection_minimum_probability": 0.0,
        "projection_optimality_error": 1e-16,
        "maximum_nonprototype_change": 0.0,
        "calibration_mean_error": 1e-16,
        "calibration_variance_error": 1e-16,
        "calibration_unchanged_identity_error": 0.0,
        "calibration_maximum_scale": 1.5,
        "development_npmi_full": 0.1,
        "development_cv_full": 0.5,
        "development_topic_diversity_full": 1.0,
        "development_top_word_redundancy_full": 0.0,
    }


def test_final_diagnostics_preserve_all_seeds_without_selection():
    seeds = [101, 211, 307]
    result = validate_final_diagnostics(
        [_diagnostic(seed) for seed in seeds], expected_seeds=seeds
    )
    assert result["seed_count"] == 3
    assert result["seed_selection_performed"] is False
    assert result["all_parent_losses_decreased"] is True
    assert result["minimum_prototype_core_recall"] == 1.0


def test_final_diagnostics_reject_seed_reordering_or_nonfinite_values():
    with pytest.raises(ValueError, match="seed"):
        validate_final_diagnostics(
            [_diagnostic(211), _diagnostic(101)], expected_seeds=[101, 211]
        )
    bad = _diagnostic(101)
    bad["development_cv_full"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_final_diagnostics([bad], expected_seeds=[101])
