"""Pure evaluation contracts for the frozen V6.1 Step-8 variants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.city_backoff import mix_with_prior
from src.v6.data_contract import sha256_object, sparse_logical_hash
from src.v6.final_refit import FINAL_VARIANTS

GRAPH_REPORTING_VARIANTS = frozenset(
    {
        "full_graph_calibration_pooled",
        "without_pooled_graph_calibration_retained",
        "without_calibration_graph_pooled_retained",
    }
)


def construct_frozen_variants(
    base_probability: np.ndarray,
    graph_native_probability: np.ndarray,
    graph_calibrated_probability: np.ndarray,
    pooled_prior: np.ndarray,
    *,
    mixture_weight: float,
) -> dict[str, np.ndarray]:
    """Apply the five predeclared paired interventions exactly."""

    base = np.asarray(base_probability, dtype=np.float64)
    native = np.asarray(graph_native_probability, dtype=np.float64)
    calibrated = np.asarray(graph_calibrated_probability, dtype=np.float64)
    prior = np.asarray(pooled_prior, dtype=np.float64)
    if base.ndim != 2 or min(base.shape) < 1:
        raise ValueError("base probabilities must be a nonempty matrix")
    if native.shape != base.shape or calibrated.shape != base.shape:
        raise ValueError("the frozen decoder probability matrices are misaligned")
    if prior.shape != (base.shape[1],):
        raise ValueError("the pooled prior does not match the vocabulary")
    pooled_rows = np.repeat(prior[None, :], base.shape[0], axis=0)
    variants = {
        "full_graph_calibration_pooled": mix_with_prior(
            calibrated, pooled_rows, mixture_weight=float(mixture_weight)
        ),
        "without_graph_pooled_retained": mix_with_prior(
            base, pooled_rows, mixture_weight=float(mixture_weight)
        ),
        "without_pooled_graph_calibration_retained": calibrated.copy(),
        "without_calibration_graph_pooled_retained": mix_with_prior(
            native, pooled_rows, mixture_weight=float(mixture_weight)
        ),
        "exact_official_parent": base.copy(),
    }
    if tuple(variants) != FINAL_VARIANTS:
        raise AssertionError("the Step-8 variant identity or order changed")
    return variants


def reporting_family(variant: str) -> str:
    """Identify the affinity that determines a variant's displayed words."""

    value = str(variant)
    if value not in FINAL_VARIANTS:
        raise ValueError(f"unknown frozen variant: {value}")
    return (
        "graph_transported_affinity"
        if value in GRAPH_REPORTING_VARIANTS
        else "official_parent_affinity"
    )


def validate_test_partition(
    counts: sparse.spmatrix,
    rows: pd.DataFrame,
    *,
    expected_documents: int,
    vocabulary: int,
    cities: Sequence[str],
    development_post_ids: Sequence[str],
) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, Any]]:
    """Validate the fixed test count/row alignment after the one-shot opening."""

    matrix = sparse.csr_matrix(counts)
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if matrix.shape != (int(expected_documents), int(vocabulary)):
        raise ValueError("the test count matrix has an unexpected shape")
    if matrix.data.size:
        rounded = np.rint(matrix.data)
        if (
            np.any(matrix.data < 0.0)
            or not np.isfinite(matrix.data).all()
            or not np.allclose(matrix.data, rounded, atol=0.0, rtol=0.0)
        ):
            raise ValueError("test counts must be finite nonnegative integers")
        matrix.data = rounded.astype(np.int64)
    matrix = matrix.astype(np.int64)
    if np.any(np.asarray(matrix.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("the test count matrix contains an empty document")
    required = {"matrix_row", "post_id", "city", "split"}
    if required - set(rows.columns):
        raise ValueError("the test row metadata is incomplete")
    ordered = rows.sort_values("matrix_row").reset_index(drop=True).copy()
    if len(ordered) != matrix.shape[0]:
        raise ValueError("test rows and counts are misaligned")
    if ordered["matrix_row"].astype(int).tolist() != list(range(matrix.shape[0])):
        raise ValueError("test matrix_row is not contiguous")
    if not ordered["split"].astype(str).eq("test").all():
        raise ValueError("a non-test row entered confirmation")
    if ordered["post_id"].astype(str).duplicated().any():
        raise ValueError("test post IDs are not unique")
    city_order = list(map(str, cities))
    if set(ordered["city"].astype(str)) != set(city_order):
        raise ValueError("the test city inventory changed")
    test_ids = set(ordered["post_id"].astype(str))
    overlap = test_ids & set(map(str, development_post_ids))
    if overlap:
        raise ValueError("a post ID crosses development and test partitions")
    report = {
        "schema_version": 1,
        "documents": int(matrix.shape[0]),
        "tokens": int(matrix.sum()),
        "vocabulary": int(matrix.shape[1]),
        "city_documents": {
            city: int(ordered["city"].astype(str).eq(city).sum()) for city in city_order
        },
        "logical_sha256": sparse_logical_hash(matrix),
        "post_id_sequence_sha256": sha256_object(
            ordered["post_id"].astype(str).tolist()
        ),
        "development_test_post_id_overlap": 0,
    }
    return matrix, ordered, report
