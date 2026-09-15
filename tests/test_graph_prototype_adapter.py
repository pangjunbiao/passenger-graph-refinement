import numpy as np
import torch

from src.v6.graph_prototype_adapter import (
    align_prototype_cores_to_topics,
    build_graph_prototype_bank,
    centred_adapter_residual,
    effective_affinity,
    minimum_distortion_graph_projection,
    prototype_gradient_receipt,
    prototype_lexical_terms,
)


def _community_graph(communities=4, width=8):
    size = communities * width
    graph = np.full((size, size), 1.0e-4, dtype=np.float64)
    np.fill_diagonal(graph, 0.0)
    for community in range(communities):
        rows = np.arange(community * width, (community + 1) * width)
        block = 0.2 + np.add.outer(
            np.arange(width, dtype=np.float64),
            np.arange(width, dtype=np.float64),
        ) / 100.0
        np.fill_diagonal(block, 0.0)
        graph[np.ix_(rows, rows)] = block
    return graph


def test_graph_prototype_bank_is_deterministic_disjoint_and_supported():
    graph = _community_graph()
    first = build_graph_prototype_bank(
        graph,
        topics=4,
        top_k=6,
        seed=17,
        centrality_weight=0.5,
        minimum_cluster_size=6,
    )
    second = build_graph_prototype_bank(
        graph,
        topics=4,
        top_k=6,
        seed=17,
        centrality_weight=0.5,
        minimum_cluster_size=6,
    )
    np.testing.assert_array_equal(first.core_indices, second.core_indices)
    assert np.unique(first.core_indices).size == 24
    assert first.audit["minimum_cluster_size"] >= 6
    assert first.audit["minimum_core_positive_pair_fraction"] == 1.0


def test_prototype_alignment_is_a_bijection_without_changing_cores():
    cores = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    affinity = np.full((3, 7), 0.01, dtype=np.float64)
    affinity[2, cores[0]] = 0.9
    affinity[0, cores[1]] = 0.9
    affinity[1, cores[2]] = 0.9
    aligned, audit = align_prototype_cores_to_topics(
        affinity, cores, epsilon=1.0e-12
    )
    np.testing.assert_array_equal(aligned[0], cores[1])
    np.testing.assert_array_equal(aligned[1], cores[2])
    np.testing.assert_array_equal(aligned[2], cores[0])
    assert sorted(row["prototype_cluster"] for row in audit["mapping"]) == [0, 1, 2]


def _column_stochastic_affinity(topics=4, vocabulary=40):
    generator = np.random.default_rng(704)
    affinity = generator.random((topics, vocabulary))
    return affinity / affinity.sum(axis=0, keepdims=True)


def test_minimum_distortion_projection_zero_is_bitwise_parent_identity():
    affinity = _column_stochastic_affinity()
    cores = np.arange(12, dtype=np.int64).reshape(4, 3)
    result = minimum_distortion_graph_projection(
        affinity,
        cores,
        strength=0.0,
        rank_margin=1.0e-5,
        epsilon=1.0e-12,
    )
    np.testing.assert_array_equal(result.affinity, affinity)
    assert result.audit["changed_columns"] == 0
    assert result.audit["maximum_nonprototype_change"] == 0.0


def test_full_projection_secures_disjoint_cores_and_preserves_simplex():
    affinity = _column_stochastic_affinity()
    cores = np.arange(12, dtype=np.int64).reshape(4, 3)
    result = minimum_distortion_graph_projection(
        affinity,
        cores,
        strength=1.0,
        rank_margin=1.0e-5,
        epsilon=1.0e-12,
    )
    assert result.audit["minimum_prototype_core_recall"] == 1.0
    assert result.audit["minimum_prototype_rank_slack"] > 0.0
    assert result.audit["changed_nonprototype_columns"] == 0
    assert result.audit["maximum_projection_optimality_error"] < 1.0e-12
    np.testing.assert_allclose(result.affinity.sum(axis=0), 1.0, atol=1.0e-12)
    assert np.min(result.affinity) >= 0.0


def test_projection_strength_is_monotone_in_total_variation():
    affinity = _column_stochastic_affinity()
    cores = np.arange(12, dtype=np.int64).reshape(4, 3)
    distortions = []
    for strength in (0.25, 0.5, 0.75, 1.0):
        result = minimum_distortion_graph_projection(
            affinity,
            cores,
            strength=strength,
            rank_margin=1.0e-5,
            epsilon=1.0e-12,
        )
        distortions.append(result.audit["mean_vocabulary_column_total_variation"])
    assert distortions == sorted(distortions)
    np.testing.assert_allclose(
        np.asarray(distortions) / distortions[-1],
        np.asarray([0.25, 0.5, 0.75, 1.0]),
        atol=1.0e-12,
    )


def test_quota_projection_guarantees_requested_core_recall_with_less_transport():
    affinity = _column_stochastic_affinity()
    cores = np.arange(12, dtype=np.int64).reshape(4, 3)
    quota_two = minimum_distortion_graph_projection(
        affinity,
        cores,
        strength=1.0,
        prototype_quota=2,
        rank_margin=1.0e-5,
        epsilon=1.0e-12,
    )
    quota_three = minimum_distortion_graph_projection(
        affinity,
        cores,
        strength=1.0,
        prototype_quota=3,
        rank_margin=1.0e-5,
        epsilon=1.0e-12,
    )
    assert quota_two.audit["minimum_prototype_core_recall"] >= 2.0 / 3.0
    assert quota_two.audit["minimum_prototype_rank_slack"] > 0.0
    assert (
        quota_two.audit["mean_vocabulary_column_total_variation"]
        <= quota_three.audit["mean_vocabulary_column_total_variation"]
    )


def test_simplex_boundary_target_is_completed_without_changing_nonprototypes():
    affinity = np.full((3, 12), 1.0 / 3.0, dtype=np.float64)
    affinity[:, 0] = np.asarray([1.0, 0.0, 0.0])
    cores = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    result = minimum_distortion_graph_projection(
        affinity,
        cores,
        strength=1.0,
        prototype_quota=2,
        rank_margin=1.0e-5,
        epsilon=1.0e-12,
    )
    assert result.audit["requested_rank_margin_feasible"] is False
    assert result.audit["simplex_boundary_target_caps"] == 2
    assert result.audit["simplex_boundary_completion_used"] is True
    assert result.audit["maximum_simplex_target_excess"] > 0.0
    assert result.audit["prototype_quota_secured"] is False
    assert result.audit["changed_nonprototype_columns"] == 0
    np.testing.assert_array_equal(result.affinity[:, 0], affinity[:, 0])
    np.testing.assert_allclose(result.affinity.sum(axis=0), 1.0, atol=1.0e-12)
    assert np.min(result.affinity) >= 0.0


def test_boundary_completion_can_secure_quota_after_other_prototype_transport():
    affinity = np.full((3, 12), 1.0 / 3.0, dtype=np.float64)
    affinity[:, 0] = np.asarray([1.0, 0.0, 0.0])
    cores = np.asarray([[1, 2], [0, 3], [4, 5]], dtype=np.int64)
    result = minimum_distortion_graph_projection(
        affinity,
        cores,
        strength=1.0,
        prototype_quota=2,
        rank_margin=1.0e-5,
        epsilon=1.0e-12,
    )
    assert result.audit["simplex_boundary_completion_used"] is True
    assert result.audit["prototype_quota_secured"] is True
    assert result.audit["minimum_prototype_core_recall"] == 1.0
    assert result.audit["minimum_prototype_rank_slack"] > 0.0


def test_zero_adapter_is_official_affinity_identity_and_nonzero_changes_it():
    generator = torch.Generator().manual_seed(41)
    logits = torch.randn((4, 29), generator=generator, dtype=torch.float64)
    official = torch.softmax(logits, dim=0)
    zero = torch.zeros_like(official)
    identity = effective_affinity(
        official, zero, max_logit_shift=12.0, epsilon=1.0e-12
    )
    torch.testing.assert_close(identity, official, atol=2.0e-15, rtol=0.0)
    residual = torch.zeros_like(official)
    residual[0, 0] = 1.0
    changed = effective_affinity(
        official, residual, max_logit_shift=12.0, epsilon=1.0e-12
    )
    assert not torch.equal(changed, official)
    torch.testing.assert_close(
        changed.sum(dim=0),
        torch.ones(29, dtype=torch.float64),
        atol=2.0e-15,
        rtol=0.0,
    )


def test_adapter_shift_is_exactly_identifiable_and_bounded():
    generator = torch.Generator().manual_seed(42)
    raw = 100.0 * torch.randn((7, 31), generator=generator, dtype=torch.float64)
    shift = centred_adapter_residual(raw, max_logit_shift=12.0)
    torch.testing.assert_close(
        shift.sum(dim=0),
        torch.zeros(31, dtype=torch.float64),
        atol=2.0e-14,
        rtol=0.0,
    )
    assert float(shift.abs().max()) <= 12.0


def test_prototype_bundle_connects_topic_and_adapter_gradients():
    generator = torch.Generator().manual_seed(43)
    topic_logits = torch.randn(
        (4, 32), generator=generator, dtype=torch.float64, requires_grad=True
    )
    raw = torch.zeros((4, 32), dtype=torch.float64, requires_grad=True)
    official = torch.softmax(topic_logits, dim=0)
    adjusted = effective_affinity(
        official, raw, max_logit_shift=12.0, epsilon=1.0e-12
    )
    graph = torch.from_numpy(_community_graph(4, 8))
    cores = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [8, 9, 10, 11, 12, 13],
            [16, 17, 18, 19, 20, 21],
            [24, 25, 26, 27, 28, 29],
        ],
        dtype=torch.long,
    )
    terms = prototype_lexical_terms(
        adjusted,
        graph,
        raw,
        cores,
        top_k=6,
        temperature=0.2,
        overlap_weight=0.25,
        prototype_margin_weight=4.0,
        prototype_margin=0.03,
        unsupported_mass_weight=0.5,
        residual_energy_weight=0.002,
        max_logit_shift=12.0,
        epsilon=1.0e-12,
    )
    topic_gradient, adapter_gradient = torch.autograd.grad(
        terms.total, (topic_logits, raw), retain_graph=True
    )
    assert torch.isfinite(topic_gradient).all()
    assert torch.isfinite(adapter_gradient).all()
    assert float(topic_gradient.norm()) > 0.0
    assert float(adapter_gradient.norm()) > 0.0
    receipt = prototype_gradient_receipt(
        official,
        adjusted,
        graph,
        raw,
        cores,
        topic_parameter=topic_logits,
        activity_floor=1.0e-14,
        top_k=6,
        temperature=0.2,
        overlap_weight=0.25,
        prototype_margin_weight=4.0,
        prototype_margin=0.03,
        unsupported_mass_weight=0.5,
        residual_energy_weight=0.002,
        max_logit_shift=12.0,
        epsilon=1.0e-12,
    )
    assert receipt["lexical_topic_embedding_gradient_norm"] > 0.0
    assert receipt["lexical_adapter_gradient_norm"] > 0.0
    assert receipt["maximum_membership_mass_error"] < 1.0e-8
