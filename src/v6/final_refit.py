"""Pure contracts for the V6.1 final development refit and freeze.

This module deliberately contains no file discovery and no test-data API.  It
defines the exact fit-partition merge and the paired post-fit interventions
that Step 8 is later allowed to evaluate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.data_contract import sha256_object, sparse_logical_hash

FINAL_VARIANTS = (
    "full_graph_calibration_pooled",
    "without_graph_pooled_retained",
    "without_pooled_graph_calibration_retained",
    "without_calibration_graph_pooled_retained",
    "exact_official_parent",
)


def combine_development_partitions(
    train_counts: sparse.spmatrix,
    train_rows: pd.DataFrame,
    spent_counts: sparse.spmatrix,
    spent_rows: pd.DataFrame,
) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, Any]]:
    """Combine original train and spent validation without reordering either."""

    left = sparse.csr_matrix(train_counts, dtype=np.int64)
    right = sparse.csr_matrix(spent_counts, dtype=np.int64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("development count partitions are not column-aligned")
    if len(train_rows) != left.shape[0] or len(spent_rows) != right.shape[0]:
        raise ValueError("development rows and count matrices are misaligned")
    required = {"post_id", "city", "split", "matrix_row"}
    if not required.issubset(train_rows.columns) or not required.issubset(
        spent_rows.columns
    ):
        raise ValueError("development row metadata is incomplete")
    left_rows = train_rows.sort_values("matrix_row").reset_index(drop=True).copy()
    right_rows = spent_rows.sort_values("matrix_row").reset_index(drop=True).copy()
    if left_rows["matrix_row"].astype(int).tolist() != list(range(left.shape[0])):
        raise ValueError("original-training row order changed")
    if right_rows["matrix_row"].astype(int).tolist() != list(range(right.shape[0])):
        raise ValueError("spent-development row order changed")
    if set(left_rows["split"].astype(str)) != {"train"}:
        raise ValueError("the original-training partition contains another split")
    if set(right_rows["split"].astype(str)) != {"validation"}:
        raise ValueError("the spent-development partition is not validation")
    left_ids = set(left_rows["post_id"].astype(str))
    right_ids = set(right_rows["post_id"].astype(str))
    if len(left_ids) != len(left_rows) or len(right_ids) != len(right_rows):
        raise ValueError("a development partition contains duplicate post IDs")
    if left_ids & right_ids:
        raise ValueError("a post ID crosses final-fit partitions")
    combined = sparse.vstack((left, right), format="csr", dtype=np.int64)
    rows = pd.concat((left_rows, right_rows), ignore_index=True)
    rows["source_matrix_row"] = rows["matrix_row"].astype(int)
    rows["development_row"] = np.arange(len(rows), dtype=np.int64)
    if np.any(np.asarray(combined.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("the final fit set contains an empty document")
    receipt = {
        "schema_version": 1,
        "construction": "original_train_then_spent_validation_stable_concatenation",
        "original_training_documents": int(left.shape[0]),
        "spent_development_documents": int(right.shape[0]),
        "final_fit_documents": int(combined.shape[0]),
        "vocabulary": int(combined.shape[1]),
        "original_training_tokens": int(left.sum()),
        "spent_development_tokens": int(right.sum()),
        "final_fit_tokens": int(combined.sum()),
        "logical_sha256": sparse_logical_hash(combined),
        "post_id_sequence_sha256": sha256_object(rows["post_id"].astype(str).tolist()),
        "test_documents_used": 0,
    }
    return combined, rows, receipt


def final_variant_registry(
    *, prototype_quota: int, pooled_mixture_weight: float, calibration: str
) -> list[dict[str, Any]]:
    """Return the immutable paired intervention registry for Step 8."""

    quota = int(prototype_quota)
    rho = float(pooled_mixture_weight)
    if quota != 10 or calibration != "training_moment_match" or rho != 0.35:
        raise ValueError("the selected V6.1 configuration changed")
    rows = [
        {
            "variant": FINAL_VARIANTS[0],
            "graph_transport": f"quota_{quota}",
            "decoder_calibration": calibration,
            "pooled_mixture_weight": rho,
            "primary_target": "joint",
        },
        {
            "variant": FINAL_VARIANTS[1],
            "graph_transport": "off_exact_parent_affinity",
            "decoder_calibration": "identity",
            "pooled_mixture_weight": rho,
            "primary_target": "npmi_and_c_v",
        },
        {
            "variant": FINAL_VARIANTS[2],
            "graph_transport": f"quota_{quota}",
            "decoder_calibration": calibration,
            "pooled_mixture_weight": 0.0,
            "primary_target": "completion_nll",
        },
        {
            "variant": FINAL_VARIANTS[3],
            "graph_transport": f"quota_{quota}",
            "decoder_calibration": "identity_native_decoder",
            "pooled_mixture_weight": rho,
            "primary_target": "completion_nll",
        },
        {
            "variant": FINAL_VARIANTS[4],
            "graph_transport": "off_exact_parent_affinity",
            "decoder_calibration": "identity",
            "pooled_mixture_weight": 0.0,
            "primary_target": "joint_parent_comparison",
        },
    ]
    for row in rows:
        row["neural_parent_fit"] = "paired_common_seed_parent"
        row["test_status"] = "UNOPENED"
    return rows


def validate_final_diagnostics(
    rows: Sequence[Mapping[str, Any]], *, expected_seeds: Sequence[int]
) -> dict[str, Any]:
    """Aggregate only fit/integrity diagnostics; never select a seed."""

    if [int(row["seed"]) for row in rows] != list(map(int, expected_seeds)):
        raise ValueError("final diagnostic seeds differ from the frozen order")
    if len(rows) != len(set(map(int, expected_seeds))):
        raise ValueError("final diagnostic seeds are not unique")
    numeric = (
        "parent_training_loss_reduction",
        "prototype_core_recall",
        "projection_column_sum_error",
        "projection_minimum_probability",
        "projection_optimality_error",
        "maximum_nonprototype_change",
        "calibration_mean_error",
        "calibration_variance_error",
        "calibration_unchanged_identity_error",
        "calibration_maximum_scale",
        "development_npmi_full",
        "development_cv_full",
        "development_topic_diversity_full",
        "development_top_word_redundancy_full",
    )
    matrix = np.asarray(
        [[float(row[key]) for key in numeric] for row in rows], dtype=np.float64
    )
    if not np.isfinite(matrix).all():
        raise ValueError("a final-fit diagnostic is non-finite")
    return {
        "schema_version": 1,
        "seed_count": len(rows),
        "seed_selection_performed": False,
        "all_parent_losses_decreased": bool(
            all(float(row["parent_training_loss_reduction"]) > 0.0 for row in rows)
        ),
        "minimum_prototype_core_recall": float(
            min(float(row["prototype_core_recall"]) for row in rows)
        ),
        "maximum_projection_column_sum_error": float(
            max(float(row["projection_column_sum_error"]) for row in rows)
        ),
        "minimum_projection_probability": float(
            min(float(row["projection_minimum_probability"]) for row in rows)
        ),
        "maximum_projection_optimality_error": float(
            max(float(row["projection_optimality_error"]) for row in rows)
        ),
        "maximum_nonprototype_change": float(
            max(float(row["maximum_nonprototype_change"]) for row in rows)
        ),
        "maximum_calibration_mean_error": float(
            max(float(row["calibration_mean_error"]) for row in rows)
        ),
        "maximum_calibration_variance_error": float(
            max(float(row["calibration_variance_error"]) for row in rows)
        ),
        "maximum_calibration_unchanged_identity_error": float(
            max(float(row["calibration_unchanged_identity_error"]) for row in rows)
        ),
        "maximum_calibration_scale": float(
            max(float(row["calibration_maximum_scale"]) for row in rows)
        ),
    }
