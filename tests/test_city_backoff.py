import numpy as np
import pytest

from src.v6.city_backoff import (
    estimate_hierarchical_city_priors,
    estimate_pooled_unigram,
    guaranteed_parent_reduction,
    jensen_nll_upper_bound,
    mix_with_prior,
    select_city_prior_rows,
)


def _counts():
    return np.asarray(
        [
            [8, 1, 1, 0],
            [5, 2, 1, 2],
            [1, 7, 1, 1],
            [2, 5, 2, 1],
            [1, 1, 7, 1],
            [2, 1, 5, 2],
        ],
        dtype=np.float64,
    )


def test_city_priors_are_positive_simplex_rows_and_training_local():
    result = estimate_hierarchical_city_priors(
        _counts(), np.asarray([0, 0, 1, 1, 2, 2]),
        cities=3, shrinkage=0.5, pseudocount=0.5,
    )
    assert result.city.shape == (3, 4)
    assert np.all(result.city > 0.0)
    np.testing.assert_allclose(result.city.sum(axis=1), 1.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result.pooled.sum(), 1.0, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(result.city_document_counts, [2, 2, 2])
    np.testing.assert_array_equal(result.city_token_counts, [20, 20, 20])


def test_zero_shrinkage_is_exact_complete_pooling():
    result = estimate_hierarchical_city_priors(
        _counts(), np.asarray([0, 0, 1, 1, 2, 2]),
        cities=3, shrinkage=0.0, pseudocount=0.5,
    )
    np.testing.assert_array_equal(
        result.city, np.repeat(result.pooled[None, :], 3, axis=0)
    )


def test_v6_1_pooled_unigram_is_positive_normalized_and_city_free():
    pooled = estimate_pooled_unigram(_counts(), pseudocount=0.5)
    assert pooled.shape == (4,)
    assert np.all(pooled > 0.0)
    np.testing.assert_allclose(pooled.sum(), 1.0, atol=1e-15, rtol=0)
    hierarchical = estimate_hierarchical_city_priors(
        _counts(), np.asarray([0, 0, 1, 1, 2, 2]),
        cities=3, shrinkage=0.5, pseudocount=0.5,
    )
    np.testing.assert_array_equal(pooled, hierarchical.pooled)


def test_v6_1_pooled_unigram_rejects_invalid_fit_data():
    with pytest.raises(ValueError):
        estimate_pooled_unigram(np.asarray([[0.0, 0.0]]), pseudocount=0.5)
    with pytest.raises(ValueError):
        estimate_pooled_unigram(_counts(), pseudocount=0.0)


def test_zero_mixture_weight_is_bitwise_parent_identity():
    parent = np.asarray([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]], dtype=np.float64)
    prior = np.asarray([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1]], dtype=np.float64)
    mixed = mix_with_prior(parent, prior, mixture_weight=0.0)
    np.testing.assert_array_equal(mixed, parent)


def test_one_mixture_weight_is_exact_prior_and_internal_weight_is_simplex():
    parent = np.asarray([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]], dtype=np.float64)
    prior = np.asarray([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1]], dtype=np.float64)
    np.testing.assert_array_equal(
        mix_with_prior(parent, prior, mixture_weight=1.0), prior
    )
    mixed = mix_with_prior(parent, prior, mixture_weight=0.35)
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0, atol=1e-15, rtol=0)
    assert np.all(mixed > 0.0)


def test_matched_pooled_control_removes_only_city_distinctions():
    result = estimate_hierarchical_city_priors(
        _counts(), np.asarray([0, 0, 1, 1, 2, 2]),
        cities=3, shrinkage=0.0, pseudocount=0.5,
    )
    parent = np.repeat(np.asarray([[0.4, 0.3, 0.2, 0.1]]), 3, axis=0)
    indices = np.asarray([0, 1, 2])
    city_rows = select_city_prior_rows(result.city, indices)
    pooled_rows = np.repeat(result.pooled[None, :], 3, axis=0)
    np.testing.assert_array_equal(
        mix_with_prior(parent, city_rows, mixture_weight=0.2),
        mix_with_prior(parent, pooled_rows, mixture_weight=0.2),
    )


def test_jensen_cross_entropy_certificate_and_reduction_bound():
    parent = np.asarray([[0.55, 0.35, 0.10]], dtype=np.float64)
    prior = np.asarray([[0.75, 0.20, 0.05]], dtype=np.float64)
    target = np.asarray([8.0, 2.0, 0.0])
    weight = 0.25
    mixed = mix_with_prior(parent, prior, mixture_weight=weight)
    parent_nll = -float(target @ np.log(parent[0])) / target.sum()
    prior_nll = -float(target @ np.log(prior[0])) / target.sum()
    mixed_nll = -float(target @ np.log(mixed[0])) / target.sum()
    upper = jensen_nll_upper_bound(
        parent_nll, prior_nll, mixture_weight=weight
    )
    assert mixed_nll <= upper + 1e-15
    assert parent_nll - mixed_nll >= guaranteed_parent_reduction(
        parent_nll, prior_nll, mixture_weight=weight
    ) - 1e-15


def test_invalid_city_or_weight_fails_closed():
    with pytest.raises(ValueError):
        estimate_hierarchical_city_priors(
            _counts(), np.asarray([0, 0, 1, 1, 3, 3]),
            cities=3, shrinkage=0.5, pseudocount=0.5,
        )
    parent = np.asarray([[0.5, 0.5]], dtype=np.float64)
    with pytest.raises(ValueError):
        mix_with_prior(parent, parent, mixture_weight=1.01)
