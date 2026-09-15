"""Typed contracts shared by every established-baseline adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class BaselineData:
    """Comparator-visible train and observed-held-out evidence only.

    The document-completion target and city/label vector deliberately do not
    belong to this interface.  They stay in the method-independent evaluator,
    so a comparator cannot inspect them accidentally (or deliberately) while
    fitting or inferring its held-out document representation.
    """

    train_counts: sparse.csr_matrix
    validation_counts: sparse.csr_matrix
    train_embeddings: np.ndarray
    validation_embeddings: np.ndarray
    train_documents: tuple[str, ...]
    validation_documents: tuple[str, ...]
    train_token_documents: tuple[tuple[int, ...], ...]
    validation_token_documents: tuple[tuple[int, ...], ...]
    vocabulary: tuple[str, ...]
    input_fingerprint: str
    heldout_partition: str = "validation"

    @property
    def vocabulary_size(self) -> int:
        return int(self.train_counts.shape[1])


@dataclass(frozen=True)
class BaselineFit:
    """Canonical output required for method-independent evaluation."""

    method_id: str
    display_name: str
    seed: int
    topic_word: np.ndarray
    validation_theta: np.ndarray | None
    likelihood_comparable: bool
    actual_topics: int
    iterations_completed: int | None = None
    converged: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validation_word_probabilities: np.ndarray | None = None


class BaselineAdapter(ABC):
    """One auditable adapter around a published baseline implementation."""

    method_id: str
    display_name: str
    dependency: str | None = None

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)

    @abstractmethod
    def fit(self, data: BaselineData, *, topics: int, seed: int) -> BaselineFit:
        """Fit on training evidence and infer validation quantities only."""


def normalize_rows(values: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("A baseline distribution must be a finite nonnegative matrix")
    if floor < 0.0:
        raise ValueError("The normalization floor cannot be negative")
    if floor:
        matrix = np.maximum(matrix, floor)
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("A baseline distribution contains an empty row")
    return matrix / totals


def counts_to_token_documents(
    counts: sparse.spmatrix,
) -> tuple[tuple[int, ...], ...]:
    """Expand sparse counts deterministically for short-text algorithms."""

    matrix = counts.tocsr()
    documents: list[tuple[int, ...]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        indices = matrix.indices[start:end]
        frequencies = matrix.data[start:end]
        documents.append(
            tuple(
                int(index)
                for index, frequency in zip(indices, frequencies)
                for _ in range(int(frequency))
            )
        )
    return tuple(documents)


def token_documents_to_text(
    documents: Sequence[Sequence[int]], vocabulary: Sequence[str]
) -> tuple[str, ...]:
    terms = tuple(map(str, vocabulary))
    return tuple(
        " ".join(terms[int(index)] for index in document) for document in documents
    )
