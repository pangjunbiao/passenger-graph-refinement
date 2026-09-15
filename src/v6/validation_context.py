"""Training-only global context used by the one-shot V6 validation gate.

The fitted centroids and global bags of words depend only on the original
training count matrix.  A validation document is assigned from its frozen
observed completion half; its target half is never used as model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from src.v6.development_evidence import dense_hash


def _semantic_rows(
    counts: sparse.spmatrix | np.ndarray,
    vocabulary_embeddings: np.ndarray,
    *,
    name: str,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    features = np.asarray(vocabulary_embeddings, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"{name} counts must be a nonempty matrix")
    if features.ndim != 2 or features.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} counts and vocabulary embeddings are misaligned")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0.0):
        raise ValueError(f"{name} counts contain invalid values")
    token_mass = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if np.any(token_mass <= 0.0):
        raise ValueError(f"{name} contains an empty observed document")
    semantic = np.asarray(matrix @ features, dtype=np.float64)
    semantic /= token_mass[:, None]
    norms = np.linalg.norm(semantic, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError(f"{name} contains a degenerate semantic summary")
    semantic /= norms
    return matrix, semantic


@dataclass(frozen=True)
class TrainingOnlyContext:
    """Frozen context map fitted without validation or confirmation data."""

    train_input: np.ndarray
    centroids: np.ndarray
    global_bow: np.ndarray
    vocabulary_embeddings: np.ndarray
    audit: dict[str, Any]

    def assignments_from_observed_counts(
        self, observed_counts: sparse.spmatrix | np.ndarray
    ) -> np.ndarray:
        _, semantic = _semantic_rows(
            observed_counts,
            self.vocabulary_embeddings,
            name="evaluation observed-half",
        )
        distances = (
            np.sum(semantic * semantic, axis=1, keepdims=True)
            + np.sum(self.centroids * self.centroids, axis=1)[None, :]
            - 2.0 * semantic @ self.centroids.T
        )
        return np.argmin(distances, axis=1).astype(np.int64)

    def input_from_observed_counts(
        self, observed_counts: sparse.spmatrix | np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        matrix, _ = _semantic_rows(
            observed_counts,
            self.vocabulary_embeddings,
            name="evaluation observed-half",
        )
        assignments = self.assignments_from_observed_counts(matrix)
        local = matrix.toarray().astype(np.float32, copy=False)
        model_input = np.concatenate(
            (local, self.global_bow[assignments]), axis=1
        )
        receipt = {
            "schema_version": 1,
            "documents_assigned": len(assignments),
            "assignment_input": "observed_completion_half_only",
            "target_half_used_for_assignment": False,
            "full_document_embedding_used": False,
            "assignments_sha256": dense_hash(assignments),
            "model_input_sha256": dense_hash(model_input),
        }
        return model_input, receipt


def construct_training_only_context(
    training_counts: sparse.spmatrix | np.ndarray,
    vocabulary_embeddings: np.ndarray,
    *,
    clusters: int,
    n_init: int,
    seed: int,
    fit_partition_label: str = "all_620_original_training_rows_only",
    original_training_documents: int | None = None,
    spent_development_documents: int = 0,
) -> TrainingOnlyContext:
    """Fit KMeans and global bags of words on an explicitly named fit set.

    The additional audit arguments are metadata only and keep the historical
    Step-6 behavior unchanged by default.  Step 7 uses them to state openly
    that the already-spent validation rows have joined the final fit set.
    """

    from sklearn.cluster import KMeans

    counts, semantic = _semantic_rows(
        training_counts, vocabulary_embeddings, name="training"
    )
    cluster_count = int(clusters)
    if not 1 < cluster_count < counts.shape[0]:
        raise ValueError("invalid training-only global-context cluster count")
    estimator = KMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        n_init=int(n_init),
        algorithm="lloyd",
    )
    assignments = estimator.fit_predict(semantic.astype(np.float32)).astype(
        np.int64
    )
    dense_counts = counts.toarray().astype(np.float32)
    global_bow = np.zeros(
        (cluster_count, counts.shape[1]), dtype=np.float32
    )
    np.add.at(global_bow, assignments, dense_counts)
    sizes = np.bincount(assignments, minlength=cluster_count)
    if np.any(sizes <= 0) or np.any(global_bow.sum(axis=1) <= 0.0):
        raise ValueError("a training-only global context is empty")
    train_input = np.concatenate(
        (dense_counts, global_bow[assignments]), axis=1
    )
    centroids = np.asarray(estimator.cluster_centers_, dtype=np.float64)
    original_count = (
        int(counts.shape[0])
        if original_training_documents is None
        else int(original_training_documents)
    )
    spent_count = int(spent_development_documents)
    if original_count < 0 or spent_count < 0:
        raise ValueError("context partition document counts must be nonnegative")
    if original_count + spent_count != int(counts.shape[0]):
        raise ValueError("context partition counts do not match fitted rows")
    if not str(fit_partition_label).strip():
        raise ValueError("fit_partition_label must be nonempty")
    audit = {
        "schema_version": 1,
        "algorithm": "sklearn.cluster.KMeans",
        "algorithm_parameter": "lloyd",
        "clusters": cluster_count,
        "n_init": int(n_init),
        "seed": int(seed),
        "fit_partition": str(fit_partition_label),
        "documents_fit": int(counts.shape[0]),
        "original_training_documents_fit": original_count,
        "spent_development_documents_fit": spent_count,
        "validation_documents_fit": spent_count,
        "test_documents_fit": 0,
        "representation": "l2_normalized_count_weighted_frozen_vocabulary_embeddings",
        "minimum_cluster_size": int(sizes.min()),
        "maximum_cluster_size": int(sizes.max()),
        "centroids_sha256": dense_hash(
            np.asarray(estimator.cluster_centers_, dtype=np.float32)
        ),
        "assignments_sha256": dense_hash(assignments),
        "global_bow_sha256": dense_hash(global_bow),
        "train_input_sha256": dense_hash(train_input),
    }
    return TrainingOnlyContext(
        train_input=train_input,
        centroids=centroids,
        global_bow=global_bow,
        vocabulary_embeddings=np.asarray(
            vocabulary_embeddings, dtype=np.float64
        ),
        audit=audit,
    )
