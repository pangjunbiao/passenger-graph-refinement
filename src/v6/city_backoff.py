"""Training-partition lexical priors used by the V6 family.

The module is intentionally NumPy-only.  V6.1 retains only the pooled unigram
for prediction; the older city estimator remains for exact reproduction of the
recorded V6 development experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CityPriorEstimate:
    pooled: np.ndarray
    city: np.ndarray
    city_document_counts: np.ndarray
    city_token_counts: np.ndarray


def estimate_pooled_unigram(
    counts: np.ndarray, *, pseudocount: float
) -> np.ndarray:
    """Estimate one smoothed unigram from exactly the supplied fit rows.

    This API accepts no city assignment, evaluation target, or metric.  It is
    therefore the canonical V6.1 lexical-backoff estimator after the
    unsupported city-conditioned mechanism was retired.
    """

    values = np.asarray(counts, dtype=np.float64)
    alpha = float(pseudocount)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("counts must be a nonempty N-by-V matrix")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("counts must be finite and nonnegative")
    if np.any(values.sum(axis=1) <= 0.0):
        raise ValueError("every fitted document must contain a retained token")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("pseudocount must be finite and positive")
    pooled_counts = values.sum(axis=0)
    pooled = (pooled_counts + alpha) / (
        pooled_counts.sum() + alpha * values.shape[1]
    )
    _probability_rows(pooled[None, :], name="pooled prior")
    return pooled


def _probability_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    totals = array.sum(axis=1)
    # The pinned parent emits float32 softmax values.  Its native evaluator
    # accepts the corresponding one-ulp row-sum roundoff (up to 1e-6).
    if not np.allclose(totals, 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one")
    return array


def estimate_hierarchical_city_priors(
    counts: np.ndarray,
    city_indices: np.ndarray,
    *,
    cities: int,
    shrinkage: float,
    pseudocount: float,
) -> CityPriorEstimate:
    """Estimate partially pooled city unigram distributions.

    ``shrinkage=0`` is exact complete pooling and ``shrinkage=1`` is the
    independently smoothed city empirical distribution.  Only the rows passed
    to this function can affect the estimate.
    """

    values = np.asarray(counts, dtype=np.float64)
    assignments = np.asarray(city_indices, dtype=np.int64)
    city_count = int(cities)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("counts must be a nonempty N-by-V matrix")
    if assignments.shape != (values.shape[0],):
        raise ValueError("city indices and count rows are misaligned")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("counts must be finite and nonnegative")
    if np.any(values.sum(axis=1) <= 0.0):
        raise ValueError("every fitted document must contain a retained token")
    if city_count < 2 or assignments.min() < 0 or assignments.max() >= city_count:
        raise ValueError("city indices are outside the declared city universe")
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not np.isfinite(float(pseudocount)) or float(pseudocount) <= 0.0:
        raise ValueError("pseudocount must be finite and positive")

    document_counts = np.bincount(assignments, minlength=city_count).astype(np.int64)
    if np.any(document_counts <= 0):
        raise ValueError("every city must occur on the fitted partition")

    vocabulary = int(values.shape[1])
    pooled = estimate_pooled_unigram(values, pseudocount=float(pseudocount))
    city_rows: list[np.ndarray] = []
    city_token_counts: list[int] = []
    for city in range(city_count):
        selected = values[assignments == city].sum(axis=0)
        city_token_counts.append(round(float(selected.sum())))
        empirical = (selected + float(pseudocount)) / (
            selected.sum() + float(pseudocount) * vocabulary
        )
        city_rows.append(
            (1.0 - float(shrinkage)) * pooled
            + float(shrinkage) * empirical
        )
    city_matrix = np.asarray(city_rows, dtype=np.float64)
    _probability_rows(pooled[None, :], name="pooled prior")
    _probability_rows(city_matrix, name="city priors")
    return CityPriorEstimate(
        pooled=pooled,
        city=city_matrix,
        city_document_counts=document_counts,
        city_token_counts=np.asarray(city_token_counts, dtype=np.int64),
    )


def mix_with_prior(
    parent_probabilities: np.ndarray,
    prior_probabilities: np.ndarray,
    *,
    mixture_weight: float,
) -> np.ndarray:
    """Return ``(1-rho) p_parent + rho p_prior`` row by row.

    The zero-weight branch is explicit so that the parent identity is bitwise,
    not merely numerically close.
    """

    parent = _probability_rows(parent_probabilities, name="parent probabilities")
    prior = _probability_rows(prior_probabilities, name="prior probabilities")
    if prior.shape != parent.shape:
        raise ValueError("parent and prior probability matrices must have equal shape")
    weight = float(mixture_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("mixture weight must lie in [0, 1]")
    if weight == 0.0:
        return parent.copy()
    if weight == 1.0:
        return prior.copy()
    mixed = (1.0 - weight) * parent + weight * prior
    _probability_rows(mixed, name="mixed probabilities")
    return mixed


def select_city_prior_rows(
    city_priors: np.ndarray, city_indices: np.ndarray
) -> np.ndarray:
    priors = _probability_rows(city_priors, name="city priors")
    indices = np.asarray(city_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("city indices must be a nonempty vector")
    if indices.min() < 0 or indices.max() >= priors.shape[0]:
        raise ValueError("city index outside the fitted prior matrix")
    return priors[indices]


def jensen_nll_upper_bound(
    parent_nll: float,
    prior_nll: float,
    *,
    mixture_weight: float,
) -> float:
    """Cross-entropy upper bound implied by convexity of negative log."""

    weight = float(mixture_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("mixture weight must lie in [0, 1]")
    return (1.0 - weight) * float(parent_nll) + weight * float(prior_nll)


def guaranteed_parent_reduction(
    parent_nll: float,
    prior_nll: float,
    *,
    mixture_weight: float,
) -> float:
    """Lower bound on parent-minus-mixture NLL when the prior is better."""

    return float(parent_nll) - jensen_nll_upper_bound(
        parent_nll, prior_nll, mixture_weight=mixture_weight
    )
