"""Frozen, model-agnostic evaluators for SC-HTM V6.

The functions in this module consume only a fitted topic-word distribution and
an explicitly supplied reference partition.  They never import a V6 model,
which keeps the evaluator independent of the implementation being compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class NPMIResult:
    mean: float
    per_topic: np.ndarray
    pairs_per_topic: np.ndarray
    zero_joint_pairs: np.ndarray
    absent_word_pairs: np.ndarray


@dataclass(frozen=True)
class CVResult:
    mean: float
    per_topic: np.ndarray
    windows: int
    absent_topic_words: np.ndarray


@dataclass(frozen=True)
class CompletionResult:
    nll_per_token: float
    perplexity: float
    tokens: int
    documents: int


def stable_top_word_indices(topic_word: np.ndarray, top_n: int) -> np.ndarray:
    """Return deterministic top words, breaking ties by vocabulary index."""

    values = np.asarray(topic_word, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("topic_word must be a finite K-by-V matrix")
    if np.any(values < 0.0):
        raise ValueError("topic_word cannot contain negative values")
    if not 2 <= int(top_n) < values.shape[1]:
        raise ValueError("top_n must be at least 2 and smaller than the vocabulary")
    vocabulary_indices = np.arange(values.shape[1], dtype=np.int64)
    rows = [
        np.lexsort((vocabulary_indices, -row))[: int(top_n)] for row in values
    ]
    return np.asarray(rows, dtype=np.int64)


def _binary_counts(counts: np.ndarray | sparse.spmatrix) -> sparse.csr_matrix:
    if sparse.issparse(counts):
        matrix = counts.tocsr(copy=True)
        if matrix.ndim != 2 or (matrix.data.size and np.any(matrix.data < 0)):
            raise ValueError("count evidence must be a nonnegative matrix")
        matrix.eliminate_zeros()
        matrix.data = np.ones_like(matrix.data, dtype=np.float64)
    else:
        values = np.asarray(counts)
        if values.ndim != 2 or np.any(values < 0):
            raise ValueError("count evidence must be a nonnegative matrix")
        matrix = sparse.csr_matrix((values > 0).astype(np.float64))
    if matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("the reference corpus is empty")
    return matrix


def _validate_topics(topics: np.ndarray, vocabulary: int) -> np.ndarray:
    result = np.asarray(topics, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] < 2:
        raise ValueError("topic indices must be a K-by-T matrix with T >= 2")
    if result.min() < 0 or result.max() >= vocabulary:
        raise ValueError("a topic-word index lies outside the reference vocabulary")
    if any(len(set(map(int, row))) != len(row) for row in result):
        raise ValueError("a topic top-word list contains duplicates")
    return result


def document_npmi(
    counts: np.ndarray | sparse.spmatrix,
    topic_word_indices: np.ndarray,
) -> NPMIResult:
    """Document-level NPMI with an explicit zero-joint value of -1.

    The reference matrix must be from the fold held out from model fitting and
    from lexical-graph construction.  A pair containing an absent word also
    receives -1, which makes coverage failures visible instead of silently
    dropping difficult pairs.
    """

    binary = _binary_counts(counts)
    topics = _validate_topics(topic_word_indices, binary.shape[1])
    documents = int(binary.shape[0])
    frequency = np.asarray(binary.sum(axis=0)).reshape(-1)
    per_topic: list[float] = []
    pairs: list[int] = []
    zero_joint: list[int] = []
    absent_pairs: list[int] = []
    for words in topics:
        values: list[float] = []
        zeros = 0
        absent = 0
        for left in range(len(words) - 1):
            for right in range(left + 1, len(words)):
                first, second = int(words[left]), int(words[right])
                if frequency[first] == 0.0 or frequency[second] == 0.0:
                    values.append(-1.0)
                    zeros += 1
                    absent += 1
                    continue
                joint = float(binary[:, first].multiply(binary[:, second]).sum())
                if joint == 0.0:
                    values.append(-1.0)
                    zeros += 1
                    continue
                p_joint = joint / documents
                if p_joint == 1.0:
                    values.append(1.0)
                    continue
                p_left = frequency[first] / documents
                p_right = frequency[second] / documents
                values.append(
                    float(np.log(p_joint / (p_left * p_right)) / -np.log(p_joint))
                )
        per_topic.append(float(np.mean(values)))
        pairs.append(len(values))
        zero_joint.append(zeros)
        absent_pairs.append(absent)
    scores = np.asarray(per_topic, dtype=np.float64)
    return NPMIResult(
        mean=float(scores.mean()),
        per_topic=scores,
        pairs_per_topic=np.asarray(pairs, dtype=np.int64),
        zero_joint_pairs=np.asarray(zero_joint, dtype=np.int64),
        absent_word_pairs=np.asarray(absent_pairs, dtype=np.int64),
    )


def counts_to_token_documents(
    counts: np.ndarray | sparse.spmatrix,
) -> tuple[tuple[int, ...], ...]:
    """Expand integer bag-of-words counts using a deterministic token order."""

    matrix = sparse.csr_matrix(counts)
    if matrix.data.size:
        rounded = np.rint(matrix.data)
        if np.any(matrix.data < 0) or not np.allclose(matrix.data, rounded):
            raise ValueError("token documents require nonnegative integer counts")
    documents: list[tuple[int, ...]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        documents.append(
            tuple(
                int(index)
                for index, frequency in zip(
                    matrix.indices[start:end], matrix.data[start:end]
                )
                for _ in range(int(round(float(frequency))))
            )
        )
    return tuple(documents)


def _boolean_windows(
    documents: Iterable[Sequence[int]], window_size: int
) -> list[set[int]]:
    if int(window_size) < 2:
        raise ValueError("C_v window_size must be at least two")
    windows: list[set[int]] = []
    for document in documents:
        tokens = [int(value) for value in document]
        if not tokens:
            continue
        if len(tokens) <= int(window_size):
            windows.append(set(tokens))
        else:
            windows.extend(
                set(tokens[start : start + int(window_size)])
                for start in range(len(tokens) - int(window_size) + 1)
            )
    if not windows:
        raise ValueError("C_v requires at least one nonempty reference window")
    return windows


def cv_coherence(
    token_documents: Iterable[Sequence[int]],
    topic_word_indices: np.ndarray,
    *,
    vocabulary_size: int,
    window_size: int = 110,
    gamma: float = 1.0,
) -> CVResult:
    """One-set indirect-cosine C_v with a frozen missing-word policy.

    This retains the V4 Röder-style calculation but fixes its undefined case:
    a top word absent from the independent reference gets a zero confirmation
    vector and is counted in ``absent_topic_words``.  No word or pair is
    silently removed.
    """

    if float(gamma) <= 0.0:
        raise ValueError("C_v gamma must be positive")
    topics = _validate_topics(topic_word_indices, int(vocabulary_size))
    windows = _boolean_windows(token_documents, int(window_size))
    number_of_windows = len(windows)
    topic_scores: list[float] = []
    absent_counts: list[int] = []
    for words in topics:
        occurrence = np.asarray(
            [sum(int(word) in window for window in windows) for word in words],
            dtype=np.float64,
        )
        absent_counts.append(int(np.count_nonzero(occurrence == 0.0)))
        matrix = np.zeros((len(words), len(words)), dtype=np.float64)
        for index in range(len(words)):
            if occurrence[index] > 0.0:
                matrix[index, index] = 1.0
        for left in range(len(words) - 1):
            for right in range(left + 1, len(words)):
                if occurrence[left] == 0.0 or occurrence[right] == 0.0:
                    # C_v confirmation vectors use a frozen zero-vector policy
                    # for a word absent from the independent reference. NPMI
                    # separately assigns such pairs its lower bound of -1.
                    value = 0.0
                else:
                    joint = sum(
                        int(words[left]) in window and int(words[right]) in window
                        for window in windows
                    )
                    if joint == 0:
                        value = -1.0
                    else:
                        p_joint = joint / number_of_windows
                        p_left = occurrence[left] / number_of_windows
                        p_right = occurrence[right] / number_of_windows
                        value = (
                            1.0
                            if p_joint == 1.0
                            else float(
                                np.log(p_joint / (p_left * p_right))
                                / -np.log(p_joint)
                            )
                        )
                matrix[left, right] = matrix[right, left] = np.sign(value) * (
                    abs(value) ** float(gamma)
                )
        topic_vector = matrix.sum(axis=0)
        topic_norm = float(np.linalg.norm(topic_vector))
        confirmations: list[float] = []
        for index, vector in enumerate(matrix):
            if occurrence[index] == 0.0:
                confirmations.append(0.0)
                continue
            denominator = float(np.linalg.norm(vector)) * topic_norm
            confirmations.append(
                0.0 if denominator == 0.0 else float(vector @ topic_vector / denominator)
            )
        topic_scores.append(float(np.mean(confirmations)))
    scores = np.asarray(topic_scores, dtype=np.float64)
    return CVResult(
        mean=float(scores.mean()),
        per_topic=scores,
        windows=number_of_windows,
        absent_topic_words=np.asarray(absent_counts, dtype=np.int64),
    )


def topic_diversity(topic_word_indices: np.ndarray) -> float:
    topics = np.asarray(topic_word_indices, dtype=np.int64)
    if topics.ndim != 2 or topics.size == 0:
        raise ValueError("topic indices must be a nonempty matrix")
    return float(len(np.unique(topics)) / topics.size)


def mean_top_word_jaccard(topic_word_indices: np.ndarray) -> float:
    topics = np.asarray(topic_word_indices, dtype=np.int64)
    if topics.ndim != 2 or topics.shape[0] < 2:
        raise ValueError("redundancy requires at least two topics")
    values: list[float] = []
    for left in range(topics.shape[0] - 1):
        first = set(map(int, topics[left]))
        for right in range(left + 1, topics.shape[0]):
            second = set(map(int, topics[right]))
            values.append(len(first & second) / len(first | second))
    return float(np.mean(values))


def document_completion_nll(
    target_counts: np.ndarray | sparse.spmatrix,
    theta_from_observed: np.ndarray,
    topic_word: np.ndarray,
    city_indices: np.ndarray | None = None,
    *,
    probability_floor: float = 1.0e-12,
) -> CompletionResult:
    """Score target tokens using theta that was inferred from observed tokens.

    ``topic_word`` may be K-by-V (shared) or C-by-K-by-V (city-specific).
    This scorer never receives observed counts, preventing accidental inference
    on observed+target inside the metric implementation.
    """

    target = sparse.csr_matrix(target_counts, dtype=np.float64)
    target.eliminate_zeros()
    theta = np.asarray(theta_from_observed, dtype=np.float64)
    beta = np.asarray(topic_word, dtype=np.float64)
    if target.shape[0] != theta.shape[0] or theta.ndim != 2:
        raise ValueError("target counts and theta rows are misaligned")
    if target.data.size and (np.any(target.data < 0) or not np.isfinite(target.data).all()):
        raise ValueError("target counts must be finite and nonnegative")
    if not np.isfinite(theta).all() or np.any(theta < 0.0):
        raise ValueError("theta must be finite and nonnegative")
    if np.max(np.abs(theta.sum(axis=1) - 1.0)) > 1.0e-6:
        raise ValueError("theta rows must sum to one")
    if beta.ndim == 2:
        if beta.shape[0] != theta.shape[1] or beta.shape[1] != target.shape[1]:
            raise ValueError("shared topic-word dimensions are incompatible")
        probabilities = theta @ beta
    elif beta.ndim == 3:
        if city_indices is None:
            raise ValueError("city_indices are required for city-specific topics")
        cities = np.asarray(city_indices, dtype=np.int64)
        if cities.shape != (target.shape[0],):
            raise ValueError("city indices and target rows are misaligned")
        if cities.min() < 0 or cities.max() >= beta.shape[0]:
            raise ValueError("a city index is out of range")
        if beta.shape[1] != theta.shape[1] or beta.shape[2] != target.shape[1]:
            raise ValueError("city topic-word dimensions are incompatible")
        probabilities = np.einsum("dk,dkv->dv", theta, beta[cities])
    else:
        raise ValueError("topic_word must have two or three dimensions")
    if not np.isfinite(beta).all() or np.any(beta < 0.0):
        raise ValueError("topic-word probabilities must be finite and nonnegative")
    if np.max(np.abs(beta.sum(axis=-1) - 1.0)) > 1.0e-6:
        raise ValueError("topic-word rows must sum to one")
    probabilities = np.clip(probabilities, float(probability_floor), 1.0)
    tokens = int(round(float(target.sum())))
    if tokens <= 0:
        raise ValueError("document completion has no target tokens")
    rows, columns = target.nonzero()
    nll = -float(np.sum(target.data * np.log(probabilities[rows, columns])))
    per_token = nll / tokens
    return CompletionResult(
        nll_per_token=per_token,
        perplexity=float(np.exp(min(per_token, 700.0))),
        tokens=tokens,
        documents=int(target.shape[0]),
    )
