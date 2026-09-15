"""Fold-local evidence construction and independent development metrics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from src.v6.evaluation import (
    counts_to_token_documents,
    cv_coherence,
    document_npmi,
    mean_top_word_jaccard,
    stable_top_word_indices,
    topic_diversity,
)
from src.v6.predictive_evaluation import (
    document_completion_nll_from_probabilities,
)


def dense_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FoldContext:
    train_indices: np.ndarray
    evaluation_indices: np.ndarray
    train_input: np.ndarray
    centroids: np.ndarray
    global_bow: np.ndarray
    vocabulary_embeddings: np.ndarray
    audit: dict[str, Any]

    def assignments_from_local_counts(
        self, local_counts: sparse.spmatrix | np.ndarray
    ) -> np.ndarray:
        local = sparse.csr_matrix(local_counts, dtype=np.float64)
        token_mass = np.asarray(local.sum(axis=1)).reshape(-1)
        if np.any(token_mass <= 0.0):
            raise ValueError("context assignment received an empty observed document")
        semantic = np.asarray(local @ self.vocabulary_embeddings, dtype=np.float64)
        semantic /= token_mass[:, None]
        norms = np.linalg.norm(semantic, axis=1, keepdims=True)
        if np.any(norms <= 1.0e-12):
            raise ValueError("an observed document has a degenerate semantic summary")
        semantic /= norms
        distances = (
            np.sum(semantic * semantic, axis=1, keepdims=True)
            + np.sum(self.centroids * self.centroids, axis=1)[None, :]
            - 2.0 * semantic @ self.centroids.T
        )
        return np.argmin(distances, axis=1).astype(np.int64)

    def input_for_local_counts(
        self, local_counts: sparse.spmatrix | np.ndarray
    ) -> np.ndarray:
        local = (
            local_counts.toarray()
            if sparse.issparse(local_counts)
            else np.asarray(local_counts)
        ).astype(np.float32, copy=False)
        # There is deliberately no source-row or full-document-embedding input
        # here.  This makes it structurally impossible for a completion target
        # to influence its global-context assignment.
        assigned = self.assignments_from_local_counts(local)
        return np.concatenate((local, self.global_bow[assigned]), axis=1)


def construct_fold_context(
    counts: sparse.csr_matrix,
    vocabulary_embeddings: np.ndarray,
    fold_assignments: np.ndarray,
    heldout_fold: int,
    *,
    clusters: int,
    n_init: int,
    seed: int,
) -> FoldContext:
    """Fit KMeans/global BoW only on the training side of one frozen fold."""

    from sklearn.cluster import KMeans

    fold_values = np.asarray(fold_assignments, dtype=np.int64)
    train_indices = np.flatnonzero(fold_values != int(heldout_fold))
    evaluation_indices = np.flatnonzero(fold_values == int(heldout_fold))
    if min(len(train_indices), len(evaluation_indices)) < 1:
        raise ValueError("a development fold has an empty side")
    if not 1 < int(clusters) < len(train_indices):
        raise ValueError("invalid number of fold-local global contexts")
    estimator = KMeans(
        n_clusters=int(clusters),
        random_state=int(seed),
        n_init=int(n_init),
        algorithm="lloyd",
    )
    features = np.asarray(vocabulary_embeddings, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != counts.shape[1]:
        raise ValueError("vocabulary embeddings and counts are misaligned")
    train_counts = counts[train_indices].tocsr().astype(np.float64)
    token_mass = np.asarray(train_counts.sum(axis=1)).reshape(-1)
    if np.any(token_mass <= 0.0):
        raise ValueError("a fold-training document is empty")
    semantic = np.asarray(train_counts @ features, dtype=np.float64)
    semantic /= token_mass[:, None]
    norms = np.linalg.norm(semantic, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("a fold-training document has a degenerate semantic summary")
    semantic /= norms
    estimator.fit(semantic.astype(np.float32))
    train_assignments = estimator.labels_.astype(np.int64)
    train_dense = counts[train_indices].toarray().astype(np.float32)
    global_bow = np.zeros((int(clusters), counts.shape[1]), dtype=np.float32)
    np.add.at(global_bow, train_assignments, train_dense)
    sizes = np.bincount(train_assignments, minlength=int(clusters))
    if np.any(sizes <= 0) or np.any(global_bow.sum(axis=1) <= 0.0):
        raise ValueError("a fold-local global context is empty")
    train_input = np.concatenate(
        (train_dense, global_bow[train_assignments]), axis=1
    )
    audit = {
        "schema_version": 1,
        "heldout_fold": int(heldout_fold),
        "fit_partition": "frozen_fold_training_side_only",
        "documents_fit": int(len(train_indices)),
        "documents_evaluated": int(len(evaluation_indices)),
        "validation_documents_fit": 0,
        "test_documents_fit": 0,
        "labels_used": False,
        "clusters": int(clusters),
        "n_init": int(n_init),
        "seed": int(seed),
        "minimum_cluster_size": int(sizes.min()),
        "maximum_cluster_size": int(sizes.max()),
        "training_documents_assigned": int(len(train_assignments)),
        "representation": "l2_normalized_count_weighted_frozen_vocabulary_embeddings",
        "heldout_assignment": "nearest_fold_training_centroid_from_observed_counts_only",
        "full_document_embedding_used_for_completion_assignment": False,
        "centroids_sha256": dense_hash(
            np.asarray(estimator.cluster_centers_, dtype=np.float32)
        ),
        "assignments_sha256": dense_hash(train_assignments),
        "global_bow_sha256": dense_hash(global_bow),
    }
    return FoldContext(
        train_indices=train_indices,
        evaluation_indices=evaluation_indices,
        train_input=train_input,
        centroids=np.asarray(estimator.cluster_centers_, dtype=np.float64),
        global_bow=global_bow,
        vocabulary_embeddings=features,
        audit=audit,
    )


def positive_npmi_graph(
    training_counts: sparse.spmatrix,
    *,
    minimum_document_frequency: int,
    minimum_joint_documents: int = 2,
    reliability_power: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a reliability-weighted positive-NPMI graph inside one fit fold."""

    binary = sparse.csr_matrix(training_counts, dtype=np.float64)
    binary.eliminate_zeros()
    binary.data[:] = 1.0
    documents, vocabulary = binary.shape
    frequency = np.asarray(binary.sum(axis=0)).reshape(-1)
    active = frequency >= int(minimum_document_frequency)
    if int(minimum_joint_documents) < 1 or float(reliability_power) < 0.0:
        raise ValueError("positive-NPMI support settings are invalid")
    joint = (binary.T @ binary).toarray().astype(np.float64)
    p_joint = joint / float(documents)
    p_word = frequency / float(documents)
    denominator = -np.log(np.clip(p_joint, 1.0e-300, 1.0))
    ratio = p_joint / np.clip(p_word[:, None] * p_word[None, :], 1.0e-300, None)
    graph = np.zeros((vocabulary, vocabulary), dtype=np.float64)
    valid = (
        (joint >= int(minimum_joint_documents))
        & (p_joint < 1.0)
        & active[:, None]
        & active[None, :]
    )
    graph[valid] = np.log(np.clip(ratio[valid], 1.0e-300, None)) / denominator[valid]
    graph = np.clip(graph, 0.0, 1.0)
    graph *= np.power(joint / float(documents), float(reliability_power))
    np.fill_diagonal(graph, 0.0)
    graph = 0.5 * (graph + graph.T)
    positive = graph[graph > 0.0]
    if positive.size == 0:
        raise ValueError("fold-local positive-NPMI graph has no supported edge")
    graph /= float(positive.max())
    positive = graph[graph > 0.0]
    report = {
        "schema_version": 1,
        "documents": int(documents),
        "vocabulary": int(vocabulary),
        "minimum_document_frequency": int(minimum_document_frequency),
        "minimum_joint_documents": int(minimum_joint_documents),
        "reliability_power": float(reliability_power),
        "normalization": "divide_by_maximum_reliability_weighted_edge",
        "active_words": int(active.sum()),
        "positive_directed_entries": int(positive.size),
        "maximum": float(graph.max()),
        "mean_positive": float(positive.mean()),
        "symmetric_error": float(np.max(np.abs(graph - graph.T))),
        "diagonal_error": float(np.max(np.abs(np.diag(graph)))),
        "logical_sha256": dense_hash(graph),
        "fit_partition": "corresponding_fold_training_counts_only",
    }
    return graph.astype(np.float32), report


def crossfitted_city_unigram_signal(
    counts: sparse.csr_matrix,
    cities: Sequence[str],
    fold_assignments: np.ndarray,
    *,
    shrinkage: float,
    pseudocount: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Model-free check that partial pooling is supported by training data."""

    city_values = np.asarray(list(map(str, cities)))
    folds = np.asarray(fold_assignments, dtype=np.int64)
    vocabulary = int(counts.shape[1])
    rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        fit = folds != fold
        evaluate = ~fit
        pooled_counts = np.asarray(counts[fit].sum(axis=0)).reshape(-1)
        pooled = (pooled_counts + float(pseudocount)) / (
            pooled_counts.sum() + float(pseudocount) * vocabulary
        )
        city_probabilities: dict[str, np.ndarray] = {}
        for city in sorted(np.unique(city_values)):
            selected_counts = np.asarray(
                counts[fit & (city_values == city)].sum(axis=0)
            ).reshape(-1)
            empirical = (selected_counts + float(pseudocount)) / (
                selected_counts.sum() + float(pseudocount) * vocabulary
            )
            city_probabilities[city] = (
                (1.0 - float(shrinkage)) * pooled + float(shrinkage) * empirical
            )
        pooled_nll = 0.0
        city_nll = 0.0
        tokens = 0.0
        for index in np.flatnonzero(evaluate):
            row = counts.getrow(int(index))
            pooled_nll -= float(row.data @ np.log(pooled[row.indices]))
            selected = city_probabilities[city_values[index]]
            city_nll -= float(row.data @ np.log(selected[row.indices]))
            tokens += float(row.data.sum())
        rows.append(
            {
                "fold": int(fold),
                "shrinkage": float(shrinkage),
                "tokens": int(round(tokens)),
                "pooled_nll_per_token": pooled_nll / tokens,
                "partially_pooled_city_nll_per_token": city_nll / tokens,
                "reduction_positive_favors_city": (pooled_nll - city_nll) / tokens,
            }
        )
    reductions = np.asarray(
        [row["reduction_positive_favors_city"] for row in rows], dtype=np.float64
    )
    summary = {
        "schema_version": 1,
        "role": "model_free_training_partition_signal_preflight_not_model_result",
        "shrinkage": float(shrinkage),
        "pseudocount": float(pseudocount),
        "folds": int(len(rows)),
        "mean_reduction": float(reductions.mean()),
        "median_reduction": float(np.median(reductions)),
        "minimum_reduction": float(reductions.min()),
        "positive_fold_fraction": float(np.mean(reductions > 0.0)),
    }
    return rows, summary


def macro_city_completion(
    target_counts: sparse.csr_matrix,
    probabilities: np.ndarray,
    city_indices: np.ndarray,
    city_names: Sequence[str],
    *,
    probability_floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indices = np.asarray(city_indices, dtype=np.int64)
    city_rows: list[dict[str, Any]] = []
    for city_index, name in enumerate(city_names):
        selected = np.flatnonzero(indices == city_index)
        if selected.size == 0:
            raise ValueError(f"completion cohort lacks city {name}")
        result = document_completion_nll_from_probabilities(
            target_counts[selected],
            probabilities[selected],
            probability_floor=float(probability_floor),
        )
        city_rows.append(
            {
                "city": str(name),
                "documents": int(result.documents),
                "tokens": int(result.tokens),
                "nll_per_token": float(result.nll_per_token),
                "perplexity": float(result.perplexity),
            }
        )
    overall = document_completion_nll_from_probabilities(
        target_counts,
        probabilities,
        probability_floor=float(probability_floor),
    )
    return {
        "macro_city_nll_per_token": float(
            np.mean([row["nll_per_token"] for row in city_rows])
        ),
        "micro_nll_per_token": float(overall.nll_per_token),
        "micro_perplexity": float(overall.perplexity),
        "documents": int(overall.documents),
        "tokens": int(overall.tokens),
    }, city_rows


def topic_quality_metrics(
    reporting_topics: np.ndarray,
    evaluation_counts: sparse.csr_matrix,
    *,
    top_words: int,
    window_size: int,
    gamma: float,
) -> dict[str, Any]:
    topics = stable_top_word_indices(reporting_topics, int(top_words))
    npmi = document_npmi(evaluation_counts, topics)
    cv = cv_coherence(
        counts_to_token_documents(evaluation_counts),
        topics,
        vocabulary_size=evaluation_counts.shape[1],
        window_size=int(window_size),
        gamma=float(gamma),
    )
    return {
        "c_v_at_10": float(cv.mean),
        "npmi_at_10": float(npmi.mean),
        "topic_diversity_at_10": float(topic_diversity(topics)),
        "top_word_redundancy_at_10": float(mean_top_word_jaccard(topics)),
        "zero_joint_pairs": int(npmi.zero_joint_pairs.sum()),
        "absent_word_pairs": int(npmi.absent_word_pairs.sum()),
        "c_v_absent_topic_words": int(cv.absent_topic_words.sum()),
        "c_v_windows": int(cv.windows),
        "top_word_indices": topics.tolist(),
    }


def matched_topic_stability(
    topic_index_matrices: Sequence[np.ndarray],
) -> dict[str, float | int]:
    """Mean optimal top-word Jaccard over all fold pairs."""

    matrices = [np.asarray(value, dtype=np.int64) for value in topic_index_matrices]
    values: list[float] = []
    for left_index in range(len(matrices) - 1):
        left = matrices[left_index]
        for right in matrices[left_index + 1 :]:
            if left.shape != right.shape:
                raise ValueError("topic stability matrices have unequal shapes")
            similarity = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
            for i, first in enumerate(left):
                a = set(map(int, first))
                for j, second in enumerate(right):
                    b = set(map(int, second))
                    similarity[i, j] = len(a & b) / len(a | b)
            row, column = linear_sum_assignment(-similarity)
            values.append(float(similarity[row, column].mean()))
    return {
        "pairs": int(len(values)),
        "mean": float(np.mean(values)) if values else 1.0,
        "minimum": float(np.min(values)) if values else 1.0,
    }
