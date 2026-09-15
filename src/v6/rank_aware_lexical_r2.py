"""Exclusive rank-aware lexical regularization for V6 Step 5R2 revision 2.

The loss acts on the exact topic ranking displayed by official EnCOT.  The
first Step-5R2 development run established that positive-NPMI association was
effective, but its weak mean-overlap term allowed several topics to converge
to the same graph hubs.  This revision keeps the successful association term
and adds two complementary anti-collapse constraints:

* a smooth unit-capacity penalty on the total soft top-M membership assigned
  to each vocabulary item; and
* a local top-M margin that demotes every non-owner of a currently duplicated
  displayed word below that topic's M+1 boundary.

Association, overlap, capacity, margin, and unsupported-rank control remain
one indivisible lexical module.  The paired ablation sets the weight of this
complete bundle to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.v6.core import soft_topk_membership
from src.v6.encot_bridge import reporting_topic_distribution


@dataclass(frozen=True)
class RankAwareLexicalTerms:
    total: torch.Tensor
    association_loss: torch.Tensor
    overlap: torch.Tensor
    exclusive_capacity: torch.Tensor
    collision_margin: torch.Tensor
    unsupported_mass: torch.Tensor
    association: torch.Tensor
    membership: torch.Tensor
    rank_scores: torch.Tensor
    supported_words: torch.Tensor
    hard_duplicate_slots: torch.Tensor


def _topk_collision_margin(
    rank_scores: torch.Tensor,
    *,
    top_k: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a piecewise-smooth top-M collision margin and duplicate count.

    A vocabulary item occurring in more than one current hard top-M list is
    owned by the topic where it lies furthest above that topic's M+1 boundary.
    Each non-owner is charged a hinge loss until the item lies at least
    ``margin`` below its boundary.  Ownership and membership masks are discrete
    routing decisions; the selected scores and boundaries retain gradients.
    """

    topics, vocabulary = map(int, rank_scores.shape)
    indices = torch.argsort(
        rank_scores, dim=1, descending=True, stable=True
    )[:, : int(top_k) + 1]
    values = torch.gather(rank_scores, 1, indices)
    boundary = values[:, int(top_k)]
    selected_indices = indices[:, : int(top_k)]
    selected = torch.zeros(
        (topics, vocabulary), dtype=torch.bool, device=rank_scores.device
    )
    selected.scatter_(1, selected_indices, True)
    multiplicity = selected.sum(dim=0)
    duplicate_slots = (multiplicity - 1).clamp_min(0).sum()
    if not bool((multiplicity > 1).any()):
        return rank_scores.sum() * 0.0, duplicate_slots

    relative_margin = rank_scores - boundary[:, None]
    routed_margin = torch.where(
        selected,
        relative_margin.detach(),
        torch.full_like(relative_margin, -torch.inf),
    )
    owner = torch.argmax(routed_margin, dim=0)
    topic_index = torch.arange(topics, device=rank_scores.device)[:, None]
    loser = (
        selected
        & (multiplicity > 1)[None, :]
        & (topic_index != owner[None, :])
    )
    violations = torch.relu(
        rank_scores - boundary[:, None] + float(margin)
    )
    loss = violations[loser].mean()
    return loss, duplicate_slots


def rank_aware_lexical_terms(
    official_affinity: torch.Tensor,
    positive_npmi_graph: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    overlap_weight: float,
    exclusive_capacity_weight: float,
    collision_margin_weight: float,
    collision_margin: float,
    unsupported_mass_weight: float,
    epsilon: float,
) -> RankAwareLexicalTerms:
    """Return the complete differentiable lexical bundle.

    ``official_affinity`` is the exact matrix returned by ``NewMethod.get_beta``.
    Row normalization therefore changes only reporting scale and preserves the
    within-topic ordering used to obtain the published top words.
    """

    if official_affinity.ndim != 2 or min(official_affinity.shape) < 2:
        raise ValueError("official_affinity must be a nonempty K-by-V matrix")
    vocabulary = int(official_affinity.shape[1])
    if positive_npmi_graph.shape != (vocabulary, vocabulary):
        raise ValueError("positive-NPMI graph and affinity are incompatible")
    if not 1 < int(top_k) < vocabulary:
        raise ValueError("top_k must lie strictly between one and vocabulary size")
    if min(float(temperature), float(epsilon)) <= 0.0:
        raise ValueError("temperature and epsilon must be positive")
    if min(
        float(overlap_weight),
        float(exclusive_capacity_weight),
        float(collision_margin_weight),
        float(collision_margin),
        float(unsupported_mass_weight),
    ) < 0.0:
        raise ValueError("lexical bundle weights must be nonnegative")
    if not bool(torch.isfinite(official_affinity).all()):
        raise ValueError("official affinity contains a non-finite value")
    if bool((official_affinity < 0.0).any()) or bool(
        (official_affinity.sum(dim=1) <= 0.0).any()
    ):
        raise ValueError("official affinity must contain nonnegative row mass")
    if not bool(torch.isfinite(positive_npmi_graph).all()):
        raise ValueError("positive-NPMI graph contains a non-finite value")
    if bool((positive_npmi_graph < 0.0).any()):
        raise ValueError("positive-NPMI graph must be nonnegative")

    reporting = reporting_topic_distribution(official_affinity)
    log_reporting = torch.log(reporting.clamp_min(float(epsilon)))
    centered = log_reporting - log_reporting.mean(dim=1, keepdim=True)
    scale = torch.sqrt(
        torch.mean(centered * centered, dim=1, keepdim=True) + float(epsilon)
    )
    rank_scores = centered / scale
    membership = soft_topk_membership(
        rank_scores,
        int(top_k),
        float(temperature),
        epsilon=float(epsilon),
    )

    numerator = torch.einsum(
        "ki,ij,kj->k", membership, positive_npmi_graph, membership
    )
    mass = membership.sum(dim=1)
    denominator = (
        mass * mass - torch.sum(membership * membership, dim=1)
    ).clamp_min(float(epsilon))
    association = numerator / denominator
    association_loss = -torch.log(association + float(epsilon)).mean()

    normalized_membership = membership / mass[:, None].clamp_min(float(epsilon))
    pair_overlap = normalized_membership @ membership.T
    topics = int(membership.shape[0])
    off_diagonal = pair_overlap - torch.diag_embed(torch.diagonal(pair_overlap))
    overlap = off_diagonal.sum() / float(max(1, topics * (topics - 1)))

    # Every topic carries exactly M membership mass.  A hard, collision-free
    # top-M solution therefore has column mass <= 1.  Penalizing squared excess
    # over that unit capacity scales directly with duplicate displayed slots
    # in the sharp-temperature limit and penalizes graph-hub monopolies more
    # strongly than the old pair-average overlap alone.
    column_mass = membership.sum(dim=0)
    capacity_excess = torch.relu(column_mass - 1.0)
    exclusive_capacity = torch.sum(capacity_excess * capacity_excess) / (
        float(topics) * float(top_k)
    )
    collision_margin_loss, hard_duplicate_slots = _topk_collision_margin(
        rank_scores,
        top_k=int(top_k),
        margin=float(collision_margin),
    )

    supported_words = positive_npmi_graph.sum(dim=1) > 0.0
    unsupported_mass = (
        membership[:, ~supported_words].sum()
        / (float(topics) * float(top_k))
    )
    total = (
        association_loss
        + float(overlap_weight) * overlap
        + float(exclusive_capacity_weight) * exclusive_capacity
        + float(collision_margin_weight) * collision_margin_loss
        + float(unsupported_mass_weight) * unsupported_mass
    )
    return RankAwareLexicalTerms(
        total=total,
        association_loss=association_loss,
        overlap=overlap,
        exclusive_capacity=exclusive_capacity,
        collision_margin=collision_margin_loss,
        unsupported_mass=unsupported_mass,
        association=association,
        membership=membership,
        rank_scores=rank_scores,
        supported_words=supported_words,
        hard_duplicate_slots=hard_duplicate_slots,
    )


def rank_tail_gradient_receipt(
    official_affinity: torch.Tensor,
    positive_npmi_graph: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    overlap_weight: float,
    exclusive_capacity_weight: float,
    collision_margin_weight: float,
    collision_margin: float,
    unsupported_mass_weight: float,
    epsilon: float,
    activity_floor: float,
    topic_parameter: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Measure score-tail gradients and, optionally, parameter connectivity."""

    terms = rank_aware_lexical_terms(
        official_affinity,
        positive_npmi_graph,
        top_k=top_k,
        temperature=temperature,
        overlap_weight=overlap_weight,
        exclusive_capacity_weight=exclusive_capacity_weight,
        collision_margin_weight=collision_margin_weight,
        collision_margin=collision_margin,
        unsupported_mass_weight=unsupported_mass_weight,
        epsilon=epsilon,
    )
    differentiation_targets = (
        (terms.rank_scores,)
        if topic_parameter is None
        else (terms.rank_scores, topic_parameter)
    )
    gradients = torch.autograd.grad(
        terms.total, differentiation_targets, retain_graph=False
    )
    gradient = gradients[0].abs()
    topic_gradient_norm = (
        float("nan")
        if topic_parameter is None
        else float(gradients[1].detach().norm().cpu())
    )
    ranking = torch.argsort(
        official_affinity.detach(), dim=1, descending=True, stable=True
    )
    tail = ranking[:, 5 : int(top_k)]
    selected = torch.gather(gradient, 1, tail)
    return {
        "minimum_rank_6_to_10_gradient": float(selected.min().detach().cpu()),
        "mean_rank_6_to_10_gradient": float(selected.mean().detach().cpu()),
        "positive_rank_6_to_10_fraction": float(
            (selected > float(activity_floor)).float().mean().detach().cpu()
        ),
        "maximum_membership_mass_error": float(
            (terms.membership.sum(dim=1) - float(top_k))
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "mean_positive_graph_association": float(
            terms.association.mean().detach().cpu()
        ),
        "unsupported_soft_topk_mass_fraction": float(
            terms.unsupported_mass.detach().cpu()
        ),
        "cross_topic_soft_overlap": float(terms.overlap.detach().cpu()),
        "exclusive_capacity_violation": float(
            terms.exclusive_capacity.detach().cpu()
        ),
        "topk_collision_margin_loss": float(
            terms.collision_margin.detach().cpu()
        ),
        "hard_duplicate_topk_slots": int(
            terms.hard_duplicate_slots.detach().cpu()
        ),
        "lexical_topic_embedding_gradient_norm": topic_gradient_norm,
    }
