"""Deterministic token-level document-completion partitions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class CompletionPartition:
    observed: sparse.csr_matrix
    target: sparse.csr_matrix
    retained_rows: pd.DataFrame
    excluded_rows: pd.DataFrame
    report: dict[str, Any]


def _row_seed(seed: int, post_id: str) -> int:
    payload = f"SC-HTM-V6|completion|{int(seed)}|{post_id}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little", signed=False)


def _integer_counts(counts: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = counts.tocsr(copy=True)
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if matrix.data.size:
        rounded = np.rint(matrix.data)
        if (
            np.any(matrix.data < 0)
            or not np.isfinite(matrix.data).all()
            or not np.allclose(matrix.data, rounded, atol=0.0, rtol=0.0)
        ):
            raise ValueError("document completion requires finite integer counts")
        matrix.data = rounded.astype(np.int64)
    return matrix.astype(np.int64)


def _append_row(
    row: int,
    indices: np.ndarray,
    values: np.ndarray,
    output_rows: list[int],
    output_columns: list[int],
    output_values: list[int],
) -> None:
    for column, value in zip(indices, values):
        if int(value) > 0:
            output_rows.append(row)
            output_columns.append(int(column))
            output_values.append(int(value))


def make_document_completion(
    counts: sparse.spmatrix,
    rows: pd.DataFrame,
    *,
    seed: int,
    observed_fraction: float,
    minimum_total_tokens: int = 2,
) -> CompletionPartition:
    """Split each eligible document's token multiplicities reproducibly.

    A document-specific hash seed makes the result invariant to DataFrame row
    order.  A one-token repair guarantees positive observed and target mass for
    every retained document while preserving the source count vector exactly.
    """

    matrix = _integer_counts(counts)
    if matrix.shape[0] != len(rows):
        raise ValueError("count rows and row metadata are misaligned")
    if "post_id" not in rows or rows["post_id"].astype(str).duplicated().any():
        raise ValueError("completion metadata needs unique post_id values")
    if not 0.0 < float(observed_fraction) < 1.0:
        raise ValueError("observed_fraction must lie strictly between zero and one")
    if int(minimum_total_tokens) < 2:
        raise ValueError("minimum_total_tokens must be at least two")

    observed_rows: list[int] = []
    observed_columns: list[int] = []
    observed_values: list[int] = []
    target_rows: list[int] = []
    target_columns: list[int] = []
    target_values: list[int] = []
    retained_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []

    completion_row = 0
    for source_row in range(matrix.shape[0]):
        start, end = matrix.indptr[source_row], matrix.indptr[source_row + 1]
        indices = matrix.indices[start:end]
        frequencies = matrix.data[start:end]
        total = int(frequencies.sum())
        metadata = rows.iloc[source_row].to_dict()
        post_id = str(metadata["post_id"])
        if total < int(minimum_total_tokens):
            excluded_records.append(
                {
                    "post_id": post_id,
                    "source_matrix_row": int(source_row),
                    "total_tokens": total,
                    "reason": "fewer_than_minimum_completion_tokens",
                }
            )
            continue
        rng = np.random.Generator(np.random.PCG64(_row_seed(seed, post_id)))
        observed = rng.binomial(frequencies, float(observed_fraction)).astype(np.int64)
        target = frequencies - observed
        if int(observed.sum()) == 0:
            candidates = np.flatnonzero(target > 0)
            chosen = int(candidates[np.argmax(target[candidates])])
            observed[chosen] += 1
            target[chosen] -= 1
        if int(target.sum()) == 0:
            candidates = np.flatnonzero(observed > 0)
            chosen = int(candidates[np.argmax(observed[candidates])])
            observed[chosen] -= 1
            target[chosen] += 1
        if np.any(observed < 0) or np.any(target < 0) or not np.array_equal(
            observed + target, frequencies
        ):
            raise AssertionError("document-completion partition lost token mass")
        _append_row(
            completion_row,
            indices,
            observed,
            observed_rows,
            observed_columns,
            observed_values,
        )
        _append_row(
            completion_row,
            indices,
            target,
            target_rows,
            target_columns,
            target_values,
        )
        retained_records.append(
            {
                "completion_row": completion_row,
                "source_matrix_row": int(source_row),
                "post_id": post_id,
                "city": str(metadata.get("city", "")),
                "split": str(metadata.get("split", "")),
                "group_sha256": str(metadata.get("group_sha256", "")),
                "total_tokens": total,
                "observed_tokens": int(observed.sum()),
                "target_tokens": int(target.sum()),
                "row_seed_sha256": sha256(
                    str(_row_seed(seed, post_id)).encode("ascii")
                ).hexdigest(),
            }
        )
        completion_row += 1

    shape = (completion_row, matrix.shape[1])
    observed_matrix = sparse.csr_matrix(
        (observed_values, (observed_rows, observed_columns)), shape=shape, dtype=np.int64
    )
    target_matrix = sparse.csr_matrix(
        (target_values, (target_rows, target_columns)), shape=shape, dtype=np.int64
    )
    retained = pd.DataFrame.from_records(retained_records)
    excluded = pd.DataFrame.from_records(
        excluded_records,
        columns=["post_id", "source_matrix_row", "total_tokens", "reason"],
    )
    if completion_row == 0:
        raise ValueError("no document is eligible for document completion")
    source_rows = retained["source_matrix_row"].to_numpy(dtype=np.int64)
    difference = observed_matrix + target_matrix - matrix[source_rows]
    reconstruction_error = 0 if difference.nnz == 0 else int(np.max(np.abs(difference.data)))
    observed_mass = np.asarray(observed_matrix.sum(axis=1)).reshape(-1)
    target_mass = np.asarray(target_matrix.sum(axis=1)).reshape(-1)
    report = {
        "schema_version": 1,
        "construction": "post_id_hashed_token_binomial_with_nonempty_repair",
        "seed": int(seed),
        "observed_fraction": float(observed_fraction),
        "minimum_total_tokens": int(minimum_total_tokens),
        "source_documents": int(matrix.shape[0]),
        "retained_documents": int(completion_row),
        "excluded_documents": int(len(excluded)),
        "source_tokens_retained": int((observed_matrix + target_matrix).sum()),
        "observed_tokens": int(observed_matrix.sum()),
        "target_tokens": int(target_matrix.sum()),
        "minimum_observed_tokens": int(observed_mass.min()),
        "minimum_target_tokens": int(target_mass.min()),
        "maximum_reconstruction_error": reconstruction_error,
        "all_retained_sides_nonempty": bool(
            np.all(observed_mass > 0) and np.all(target_mass > 0)
        ),
    }
    return CompletionPartition(
        observed=observed_matrix,
        target=target_matrix,
        retained_rows=retained,
        excluded_rows=excluded,
        report=report,
    )
