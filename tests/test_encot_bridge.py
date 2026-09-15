import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.v6.encot_bridge import (
    city_conditioned_affinity,
    decode_word_probabilities,
    exact_document_topic_mass,
    official_affinity_from_logits,
    reporting_topic_distribution,
)


def test_official_affinity_uses_topic_axis_not_vocabulary_axis():
    logits = torch.tensor(
        [[2.0, -1.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float64
    )
    affinity = official_affinity_from_logits(logits)
    torch.testing.assert_close(affinity.sum(dim=0), torch.ones(3, dtype=torch.float64))
    assert not torch.allclose(affinity.sum(dim=1), torch.ones(2, dtype=torch.float64))


def test_zero_city_residual_is_exact_official_parent_identity():
    generator = torch.Generator().manual_seed(307)
    logits = torch.randn((4, 17), generator=generator, dtype=torch.float64)
    official = official_affinity_from_logits(logits)
    city = city_conditioned_affinity(
        logits,
        torch.zeros((3, 4, 17), dtype=torch.float64),
        residual_scale=0.0,
    )
    torch.testing.assert_close(
        city, official.unsqueeze(0).expand_as(city), atol=0.0, rtol=0.0
    )


def test_reporting_normalization_preserves_within_topic_ranking():
    affinity = torch.tensor(
        [[0.7, 0.1, 0.4, 0.2], [0.3, 0.9, 0.6, 0.8]], dtype=torch.float64
    )
    reporting = reporting_topic_distribution(affinity)
    torch.testing.assert_close(
        reporting.sum(dim=1), torch.ones(2, dtype=torch.float64)
    )
    assert torch.equal(
        torch.argsort(affinity, dim=1, descending=True, stable=True),
        torch.argsort(reporting, dim=1, descending=True, stable=True),
    )


def test_exact_theta_product_and_decoder_probability_simplex():
    global_theta = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float64)
    local_theta = torch.tensor([[0.4, 0.6], [0.9, 0.1]], dtype=torch.float64)
    mass = exact_document_topic_mass(global_theta, local_theta)
    np.testing.assert_allclose(
        mass.numpy(), np.asarray([[0.28, 0.18], [0.18, 0.08]])
    )
    affinity = official_affinity_from_logits(
        torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    )
    batch_norm = torch.nn.BatchNorm1d(3, affine=True).double().eval()
    probabilities = decode_word_probabilities(mass, affinity, batch_norm)
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(2, dtype=torch.float64)
    )

