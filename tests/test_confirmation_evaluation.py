import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.v6.confirmation_evaluation import (
    construct_frozen_variants,
    reporting_family,
    validate_test_partition,
)
from src.v6.final_refit import FINAL_VARIANTS


def _probabilities():
    base = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    native = np.asarray([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])
    calibrated = np.asarray([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    pooled = np.asarray([0.2, 0.3, 0.5])
    return base, native, calibrated, pooled


def test_frozen_variants_are_exact_and_keep_registered_order():
    base, native, calibrated, pooled = _probabilities()
    variants = construct_frozen_variants(
        base, native, calibrated, pooled, mixture_weight=0.35
    )
    assert tuple(variants) == FINAL_VARIANTS
    np.testing.assert_allclose(
        variants[FINAL_VARIANTS[0]], 0.65 * calibrated + 0.35 * pooled
    )
    np.testing.assert_allclose(variants[FINAL_VARIANTS[1]], 0.65 * base + 0.35 * pooled)
    np.testing.assert_array_equal(variants[FINAL_VARIANTS[2]], calibrated)
    np.testing.assert_allclose(
        variants[FINAL_VARIANTS[3]], 0.65 * native + 0.35 * pooled
    )
    np.testing.assert_array_equal(variants[FINAL_VARIANTS[4]], base)


def test_reporting_family_follows_affinity_not_decoder_only_modules():
    assert reporting_family(FINAL_VARIANTS[0]) == "graph_transported_affinity"
    assert reporting_family(FINAL_VARIANTS[2]) == "graph_transported_affinity"
    assert reporting_family(FINAL_VARIANTS[3]) == "graph_transported_affinity"
    assert reporting_family(FINAL_VARIANTS[1]) == "official_parent_affinity"
    assert reporting_family(FINAL_VARIANTS[4]) == "official_parent_affinity"


def test_test_partition_contract_checks_alignment_and_development_overlap():
    counts = sparse.csr_matrix(np.asarray([[1, 1, 0], [0, 1, 2], [1, 0, 1]]))
    rows = pd.DataFrame(
        {
            "matrix_row": [0, 1, 2],
            "post_id": ["a", "b", "c"],
            "city": ["beijing", "shanghai", "xiamen"],
            "split": ["test"] * 3,
        }
    )
    matrix, ordered, report = validate_test_partition(
        counts,
        rows,
        expected_documents=3,
        vocabulary=3,
        cities=["beijing", "shanghai", "xiamen"],
        development_post_ids=["d", "e"],
    )
    assert matrix.sum() == 7
    assert ordered["post_id"].tolist() == ["a", "b", "c"]
    assert report["development_test_post_id_overlap"] == 0
    with pytest.raises(ValueError, match="crosses development and test"):
        validate_test_partition(
            counts,
            rows,
            expected_documents=3,
            vocabulary=3,
            cities=["beijing", "shanghai", "xiamen"],
            development_post_ids=["b"],
        )
