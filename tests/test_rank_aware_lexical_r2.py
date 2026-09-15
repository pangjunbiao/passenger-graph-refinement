import torch

from src.v6.rank_aware_lexical_r2 import (
    rank_aware_lexical_terms,
    rank_tail_gradient_receipt,
)


def _graph(vocabulary: int, supported: int) -> torch.Tensor:
    result = torch.zeros((vocabulary, vocabulary), dtype=torch.float64)
    raw = torch.arange(1, supported * supported + 1, dtype=torch.float64).reshape(
        supported, supported
    )
    block = 0.5 * (raw + raw.T)
    block.fill_diagonal_(0.0)
    block /= block.max()
    result[:supported, :supported] = block
    return result


def _terms(logits: torch.Tensor, graph: torch.Tensor):
    affinity = torch.softmax(logits, dim=0)
    return _terms_from_affinity(affinity, graph)


def _terms_from_affinity(affinity: torch.Tensor, graph: torch.Tensor):
    return rank_aware_lexical_terms(
        affinity,
        graph,
        top_k=5,
        temperature=0.25,
        overlap_weight=1.0,
        exclusive_capacity_weight=6.0,
        collision_margin_weight=4.0,
        collision_margin=0.05,
        unsupported_mass_weight=0.75,
        epsilon=1.0e-12,
    )


def test_lexical_bundle_has_exact_soft_topk_mass_and_finite_gradients():
    generator = torch.Generator().manual_seed(501)
    logits = torch.randn(
        (4, 23), generator=generator, dtype=torch.float64, requires_grad=True
    )
    terms = _terms(logits, _graph(23, 17))
    torch.testing.assert_close(
        terms.membership.sum(dim=1),
        torch.full((4,), 5.0, dtype=torch.float64),
        atol=1.0e-10,
        rtol=0.0,
    )
    gradient = torch.autograd.grad(terms.total, logits)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_unsupported_mass_detects_rank_mass_outside_graph_support():
    graph = _graph(16, 8)
    supported_affinity = torch.full((3, 16), 0.01, dtype=torch.float64)
    unsupported_affinity = torch.full((3, 16), 0.01, dtype=torch.float64)
    supported_affinity[:, :8] += torch.tensor(
        [[8, 7, 6, 5, 4, 3, 2, 1]] * 3, dtype=torch.float64
    )
    unsupported_affinity[:, 8:] += torch.tensor(
        [[8, 7, 6, 5, 4, 3, 2, 1]] * 3, dtype=torch.float64
    )
    supported = _terms_from_affinity(supported_affinity, graph)
    unsupported = _terms_from_affinity(unsupported_affinity, graph)
    assert float(supported.unsupported_mass) < float(unsupported.unsupported_mass)


def test_overlap_term_penalizes_identical_topic_rankings():
    graph = _graph(24, 24)
    identical = torch.full((4, 24), 0.01, dtype=torch.float64)
    identical[:, :6] += torch.arange(6, dtype=torch.float64)
    separated = torch.full((4, 24), 0.01, dtype=torch.float64)
    for topic in range(4):
        separated[topic, topic * 6 : (topic + 1) * 6] += torch.arange(
            6, dtype=torch.float64
        )
    assert float(_terms_from_affinity(identical, graph).overlap) > float(
        _terms_from_affinity(separated, graph).overlap
    )


def test_exclusive_capacity_and_margin_target_hard_topk_collapse():
    graph = _graph(32, 32)
    identical = torch.full((4, 32), 0.01, dtype=torch.float64)
    identical[:, :5] += torch.tensor([5, 4, 3, 2, 1], dtype=torch.float64)
    separated = torch.full((4, 32), 0.01, dtype=torch.float64)
    for topic in range(4):
        separated[topic, topic * 5 : (topic + 1) * 5] += torch.tensor(
            [5, 4, 3, 2, 1], dtype=torch.float64
        )
    collapsed = _terms_from_affinity(identical, graph)
    exclusive = _terms_from_affinity(separated, graph)
    assert int(collapsed.hard_duplicate_slots) == 15
    assert int(exclusive.hard_duplicate_slots) == 0
    assert float(collapsed.exclusive_capacity) > float(
        exclusive.exclusive_capacity
    )
    assert float(collapsed.collision_margin) > 0.0
    assert float(exclusive.collision_margin) == 0.0


def test_collision_margin_backpropagates_to_non_owner_rank_scores():
    graph = _graph(28, 28)
    logits = torch.zeros((4, 28), dtype=torch.float64, requires_grad=True)
    terms = _terms(logits, graph)
    gradient = torch.autograd.grad(terms.collision_margin, logits)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_reported_rank_six_to_ten_receives_real_gradient():
    generator = torch.Generator().manual_seed(503)
    logits = torch.randn(
        (3, 31), generator=generator, dtype=torch.float64, requires_grad=True
    )
    receipt = rank_tail_gradient_receipt(
        torch.softmax(logits, dim=0),
        _graph(31, 25),
        top_k=10,
        temperature=0.15,
        overlap_weight=1.0,
        exclusive_capacity_weight=6.0,
        collision_margin_weight=4.0,
        collision_margin=0.05,
        unsupported_mass_weight=0.75,
        epsilon=1.0e-12,
        activity_floor=1.0e-12,
        topic_parameter=logits,
    )
    assert receipt["minimum_rank_6_to_10_gradient"] > 1.0e-12
    assert receipt["positive_rank_6_to_10_fraction"] == 1.0
    assert receipt["lexical_topic_embedding_gradient_norm"] > 1.0e-12
