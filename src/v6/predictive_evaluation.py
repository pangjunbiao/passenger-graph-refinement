"""Native-decoder document-completion scoring locked before quality results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class PredictiveCompletionResult:
    nll_per_token: float
    perplexity: float
    tokens: int
    documents: int


def document_completion_nll_from_probabilities(
    target_counts: np.ndarray | sparse.spmatrix,
    word_probabilities_from_observed: np.ndarray,
    *,
    probability_floor: float = 1.0e-12,
) -> PredictiveCompletionResult:
    """Score targets from already-decoded probabilities.

    The function has no observed-count argument.  The caller must infer all
    latent quantities and decoder probabilities from the observed half only.
    This supports EnCOT's native BatchNorm-plus-softmax decoder without
    replacing it by a normalized ``theta @ beta`` approximation.
    """

    target = sparse.csr_matrix(target_counts, dtype=np.float64)
    target.sum_duplicates()
    target.eliminate_zeros()
    probabilities = np.asarray(word_probabilities_from_observed, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != target.shape:
        raise ValueError("target counts and decoded probabilities are misaligned")
    if target.data.size and (
        np.any(target.data < 0.0) or not np.isfinite(target.data).all()
    ):
        raise ValueError("target counts must be finite and nonnegative")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("decoded probabilities must be finite and in [0, 1]")
    if np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 1.0e-6:
        raise ValueError("decoded probability rows must sum to one")
    if not 0.0 < float(probability_floor) < 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one")
    tokens_float = float(target.sum())
    tokens = int(round(tokens_float))
    if tokens <= 0 or abs(tokens_float - tokens) > 1.0e-9:
        raise ValueError("document completion must contain positive integer target mass")
    rows, columns = target.nonzero()
    selected = np.clip(
        probabilities[rows, columns], float(probability_floor), 1.0
    )
    nll = -float(np.sum(target.data * np.log(selected)))
    per_token = nll / tokens
    return PredictiveCompletionResult(
        nll_per_token=per_token,
        perplexity=float(np.exp(min(per_token, 700.0))),
        tokens=tokens,
        documents=int(target.shape[0]),
    )


def hand_calculated_self_test() -> dict[str, object]:
    target = sparse.csr_matrix(
        np.asarray([[1, 1, 0], [0, 0, 2]], dtype=np.int64)
    )
    probabilities = np.asarray(
        [[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]], dtype=np.float64
    )
    result = document_completion_nll_from_probabilities(target, probabilities)
    expected = float(np.log(2.0))
    return {
        "schema_version": 1,
        "fixture": "four_target_tokens_each_assigned_probability_one_half",
        "expected_nll_per_token": expected,
        "observed_nll_per_token": result.nll_per_token,
        "absolute_error": abs(result.nll_per_token - expected),
        "passed": abs(result.nll_per_token - expected) <= 1.0e-12,
    }

