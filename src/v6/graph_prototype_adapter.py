"""Fold-local graph prototypes and likelihood-connected affinity transforms.

Revision 5 uses :func:`minimum_distortion_graph_projection`: a closed-form,
prototype-column-only transform of the frozen official EnCOT affinity.  It
retains the revision-4 solution exactly whenever the requested rank target is
inside the probability simplex and clips only an otherwise-infeasible target
to the owning word column's simplex boundary.  The older differentiable
residual utilities remain solely for reproducibility of the recorded
revision-3 negative experiment.  In both cases one effective affinity drives
displayed words and document completion; there is no reporting-only topic
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class GraphPrototypeBank:
    core_indices: np.ndarray
    cluster_labels: np.ndarray
    active_indices: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class PrototypeLexicalTerms:
    total: Any
    association_loss: Any
    overlap: Any
    prototype_margin: Any
    unsupported_mass: Any
    residual_energy: Any
    association: Any
    membership: Any
    rank_scores: Any
    hard_duplicate_slots: Any
    prototype_core_recall: Any


@dataclass(frozen=True)
class MinimumDistortionProjection:
    """Closed-form transport of graph prototypes on the word simplex."""

    affinity: np.ndarray
    required_owner_increase: np.ndarray
    applied_owner_increase: np.ndarray
    audit: dict[str, Any]


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_graph_prototype_bank(
    graph: np.ndarray,
    *,
    topics: int,
    top_k: int,
    seed: int,
    centrality_weight: float,
    minimum_cluster_size: int,
) -> GraphPrototypeBank:
    """Extract disjoint coherent cores from a training-only lexical graph.

    Spectral partitioning supplies globally distinct communities.  Within each
    community, deterministic greedy selection balances connection to the
    already selected core against internal weighted degree.  No topic-quality
    metric, label, validation row, test row, baseline, or SOTA output enters
    this construction.
    """

    from sklearn.cluster import SpectralClustering

    values = np.asarray(graph, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("prototype graph must be square")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("prototype graph contains invalid weights")
    if np.max(np.abs(values - values.T)) > 1.0e-6:
        raise ValueError("prototype graph must be symmetric")
    if np.max(np.abs(np.diag(values))) > 1.0e-7:
        raise ValueError("prototype graph diagonal must be zero")
    if int(topics) < 2 or int(top_k) < 2:
        raise ValueError("prototype dimensions are too small")
    if int(minimum_cluster_size) < int(top_k):
        raise ValueError("minimum cluster size cannot be below top_k")
    if float(centrality_weight) < 0.0:
        raise ValueError("centrality weight must be nonnegative")

    degree = values.sum(axis=1)
    active = np.flatnonzero(degree > 0.0)
    if active.size < int(topics) * int(minimum_cluster_size):
        raise ValueError("too few graph-supported words for prototype bank")
    active_graph = values[np.ix_(active, active)]
    estimator = SpectralClustering(
        n_clusters=int(topics),
        affinity="precomputed",
        eigen_solver="arpack",
        assign_labels="cluster_qr",
        random_state=int(seed),
        n_jobs=1,
    )
    labels = np.asarray(estimator.fit_predict(active_graph), dtype=np.int64)
    cores: list[np.ndarray] = []
    cluster_sizes: list[int] = []
    core_pair_fractions: list[float] = []
    core_mean_weights: list[float] = []
    for cluster in range(int(topics)):
        nodes = active[labels == cluster]
        cluster_sizes.append(int(nodes.size))
        if nodes.size < int(minimum_cluster_size):
            raise ValueError(
                f"spectral prototype cluster {cluster} has only {nodes.size} words"
            )
        block = values[np.ix_(nodes, nodes)]
        internal_degree = block.sum(axis=1)
        selected = [int(np.argmax(internal_degree))]
        while len(selected) < int(top_k):
            remaining = np.setdiff1d(
                np.arange(nodes.size, dtype=np.int64),
                np.asarray(selected, dtype=np.int64),
                assume_unique=False,
            )
            cohesion = block[np.ix_(remaining, selected)].mean(axis=1)
            normalized_degree = internal_degree[remaining] / max(
                float(internal_degree.max()), 1.0e-12
            )
            score = cohesion + float(centrality_weight) * normalized_degree
            selected.append(int(remaining[int(np.argmax(score))]))
        core = nodes[np.asarray(selected, dtype=np.int64)]
        cores.append(core)
        upper = values[np.ix_(core, core)][np.triu_indices(int(top_k), 1)]
        core_pair_fractions.append(float(np.mean(upper > 0.0)))
        core_mean_weights.append(float(np.mean(upper)))
    core_indices = np.stack(cores).astype(np.int64, copy=False)
    if np.unique(core_indices).size != int(topics) * int(top_k):
        raise AssertionError("spectral prototype cores are not disjoint")
    full_labels = np.full(values.shape[0], -1, dtype=np.int64)
    full_labels[active] = labels
    audit = {
        "schema_version": 1,
        "construction": "spectral_partition_then_deterministic_cohesive_core",
        "fit_partition": "corresponding_fold_training_counts_only",
        "topics": int(topics),
        "top_k": int(top_k),
        "seed": int(seed),
        "centrality_weight": float(centrality_weight),
        "active_words": int(active.size),
        "minimum_cluster_size": int(min(cluster_sizes)),
        "maximum_cluster_size": int(max(cluster_sizes)),
        "minimum_core_positive_pair_fraction": float(min(core_pair_fractions)),
        "mean_core_positive_pair_fraction": float(np.mean(core_pair_fractions)),
        "minimum_core_mean_graph_weight": float(min(core_mean_weights)),
        "core_indices_sha256": _array_hash(core_indices),
        "cluster_labels_sha256": _array_hash(full_labels),
    }
    return GraphPrototypeBank(
        core_indices=core_indices,
        cluster_labels=full_labels,
        active_indices=active,
        audit=audit,
    )


def align_prototype_cores_to_topics(
    official_affinity: np.ndarray,
    core_indices: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align unordered graph communities to official topics without labels."""

    affinity = np.asarray(official_affinity, dtype=np.float64)
    cores = np.asarray(core_indices, dtype=np.int64)
    if affinity.ndim != 2 or cores.ndim != 2:
        raise ValueError("affinity and prototype cores must be matrices")
    if affinity.shape[0] != cores.shape[0]:
        raise ValueError("prototype count does not match official topic count")
    if np.any(cores < 0) or np.any(cores >= affinity.shape[1]):
        raise ValueError("prototype index lies outside the vocabulary")
    reporting = affinity / np.maximum(
        affinity.sum(axis=1, keepdims=True), float(epsilon)
    )
    scores = np.stack(
        [reporting[:, core].mean(axis=1) for core in cores], axis=1
    )
    topic_rows, cluster_columns = linear_sum_assignment(-scores)
    aligned = np.empty_like(cores)
    mapping: list[dict[str, Any]] = []
    for topic, cluster in zip(topic_rows, cluster_columns):
        aligned[int(topic)] = cores[int(cluster)]
        mapping.append(
            {
                "topic": int(topic),
                "prototype_cluster": int(cluster),
                "initial_mean_reporting_mass": float(scores[topic, cluster]),
            }
        )
    return aligned, {
        "schema_version": 1,
        "criterion": "maximum_initial_official_reporting_mass_hungarian",
        "mapping": sorted(mapping, key=lambda row: row["topic"]),
        "aligned_core_indices_sha256": _array_hash(aligned),
    }


def minimum_distortion_graph_projection(
    official_affinity: np.ndarray,
    core_indices: np.ndarray,
    *,
    strength: float,
    prototype_quota: int | None = None,
    rank_margin: float,
    epsilon: float,
) -> MinimumDistortionProjection:
    """Project graph prototypes toward their aligned topics in closed form.

    The official EnCOT affinity is column stochastic: every word column is a
    distribution over topics.  For a prototype word owned by topic ``k``, the
    smallest possible total-variation change that raises its topic-``k`` mass
    by ``delta`` is exactly ``delta``.  This routine makes that change while
    removing the same mass proportionally from the other topics.  At full
    strength at least ``prototype_quota`` graph-core words enter the displayed
    top-k whenever the requested positive rank margin is feasible.  For a
    quota of ``q`` out of ``M``, the selected prototypes need
    only clear the ``(M-q+1)``-th strongest complement word; this avoids the
    needless likelihood distortion of pushing every selected word to rank 1.
    The subset with the smallest required transport is chosen exhaustively
    from the (small) graph core.  At zero strength the result is an exact copy
    of the official parent.  Words outside the disjoint prototype bank are
    never modified.

    The construction is deterministic and has no learned K-by-V parameter
    matrix.  The returned affinity is intended to drive both the EnCOT
    likelihood and displayed topic words.  If ``boundary + rank_margin`` lies
    above the simplex, the target is deterministically capped at the exact
    per-column simplex boundary.  This is the minimum-TV feasible completion;
    the audit records the cap and whether the requested quota was nevertheless
    secured after all prototype columns were transported.
    """

    base = np.asarray(official_affinity, dtype=np.float64)
    cores = np.asarray(core_indices, dtype=np.int64)
    if base.ndim != 2 or cores.ndim != 2:
        raise ValueError("affinity and prototype cores must be matrices")
    topics, vocabulary = base.shape
    if cores.shape[0] != topics:
        raise ValueError("prototype count does not match topic count")
    if cores.shape[1] < 1:
        raise ValueError("each topic needs at least one prototype")
    quota = cores.shape[1] if prototype_quota is None else int(prototype_quota)
    if not 1 <= quota <= cores.shape[1]:
        raise ValueError("prototype quota must lie within the displayed core")
    if np.any(cores < 0) or np.any(cores >= vocabulary):
        raise ValueError("prototype index lies outside the vocabulary")
    if np.unique(cores).size != cores.size:
        raise ValueError("prototype cores must be globally disjoint")
    if not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("official affinity contains invalid probabilities")
    if not np.allclose(base.sum(axis=0), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("official affinity must be column stochastic")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("projection strength must lie in [0, 1]")
    if float(rank_margin) <= 0.0 or float(epsilon) <= 0.0:
        raise ValueError("projection margin and epsilon must be positive")

    projected = base.copy()
    required = np.zeros_like(cores, dtype=np.float64)
    applied = np.zeros_like(cores, dtype=np.float64)
    selected_slots = np.zeros_like(cores, dtype=bool)
    targets = np.zeros(topics, dtype=np.float64)
    all_prototypes = np.unique(cores.reshape(-1))
    simplex_boundary_caps = 0
    requested_margin_feasible = True
    maximum_requested_target = 0.0
    maximum_simplex_target_excess = 0.0

    for topic in range(topics):
        best: tuple[float, tuple[int, ...], float, np.ndarray] | None = None
        allowed_complement_words = cores.shape[1] - quota
        for slots_tuple in combinations(range(cores.shape[1]), quota):
            slots = np.asarray(slots_tuple, dtype=np.int64)
            selected_words = cores[topic, slots]
            complement = np.ones(vocabulary, dtype=bool)
            complement[selected_words] = False
            complement_scores = np.sort(base[topic, complement])[::-1]
            if complement_scores.size <= allowed_complement_words:
                raise ValueError("prototype quota leaves too small a complement")
            boundary = float(complement_scores[allowed_complement_words])
            requested_target = boundary + float(rank_margin)
            column_totals = base[:, selected_words].sum(axis=0)
            feasible_targets = np.minimum(requested_target, column_totals)
            deltas = np.maximum(
                0.0, feasible_targets - base[topic, selected_words]
            )
            candidate = (
                float(deltas.sum()),
                slots_tuple,
                requested_target,
                deltas,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        assert best is not None
        _, chosen_tuple, requested_target, deltas = best
        maximum_requested_target = max(
            maximum_requested_target, float(requested_target)
        )
        chosen_slots = np.asarray(chosen_tuple, dtype=np.int64)
        selected_slots[topic, chosen_slots] = True
        targets[topic] = min(float(requested_target), 1.0)
        for local_index, slot in enumerate(chosen_slots):
            word = cores[topic, slot]
            owner_mass = float(base[topic, word])
            required_delta = float(deltas[local_index])
            column_total = float(base[:, word].sum())
            if float(requested_target) > column_total:
                simplex_boundary_caps += 1
                requested_margin_feasible = False
                maximum_simplex_target_excess = max(
                    maximum_simplex_target_excess,
                    float(requested_target) - column_total,
                )
            available_mass = column_total - owner_mass
            if required_delta > available_mass + 1.0e-12:
                raise ValueError("prototype projection requires unavailable mass")
            applied_delta = float(strength) * required_delta
            required[topic, int(slot)] = required_delta
            applied[topic, int(slot)] = applied_delta
            if applied_delta == 0.0:
                continue
            if available_mass <= float(epsilon):
                raise ValueError("prototype owner already exhausts its word column")
            retained_fraction = 1.0 - applied_delta / available_mass
            projected[:, word] = base[:, word] * retained_fraction
            projected[topic, word] = owner_mass + applied_delta

    difference = projected - base
    column_tv = 0.5 * np.abs(difference).sum(axis=0)
    prototype_mask = np.zeros(vocabulary, dtype=bool)
    prototype_mask[all_prototypes] = True
    nonprototype_error = float(
        np.max(np.abs(difference[:, ~prototype_mask]))
        if bool((~prototype_mask).any())
        else 0.0
    )
    expected_tv = np.zeros(vocabulary, dtype=np.float64)
    for topic, words in enumerate(cores):
        expected_tv[words] = applied[topic]
    optimality_error = float(np.max(np.abs(column_tv - expected_tv)))
    column_sum_error = float(np.max(np.abs(projected.sum(axis=0) - 1.0)))
    minimum_probability = float(np.min(projected))
    ranking = np.argsort(-projected, axis=1, kind="stable")[:, : cores.shape[1]]
    recalls = np.asarray(
        [
            np.intersect1d(ranking[topic], cores[topic]).size / cores.shape[1]
            for topic in range(topics)
        ],
        dtype=np.float64,
    )
    rank_slacks = []
    for topic in range(topics):
        selected_words = cores[topic, selected_slots[topic]]
        complement = np.setdiff1d(
            np.arange(vocabulary, dtype=np.int64),
            selected_words,
            assume_unique=False,
        )
        boundary_rank = cores.shape[1] - quota
        complement_scores = np.sort(projected[topic, complement])[::-1]
        rank_slacks.append(
            float(np.min(projected[topic, selected_words]))
            - float(complement_scores[boundary_rank])
        )
    rank_slacks = np.asarray(rank_slacks, dtype=np.float64)
    changed_columns = np.flatnonzero(column_tv > 1.0e-15)
    changed_outside = np.setdiff1d(
        changed_columns, all_prototypes, assume_unique=False
    )
    if column_sum_error > 1.0e-6 or minimum_probability < -1.0e-12:
        raise AssertionError("closed-form graph projection left the simplex")
    if nonprototype_error != 0.0 or changed_outside.size:
        raise AssertionError("graph projection changed a non-prototype word")
    if optimality_error > 1.0e-12:
        raise AssertionError("graph projection is not minimum-TV for its target")
    quota_secured = bool(
        float(np.min(rank_slacks)) > 0.0
        and float(np.min(recalls)) + 1.0e-12 >= quota / cores.shape[1]
    )
    if (
        float(strength) == 1.0
        and requested_margin_feasible
        and not quota_secured
    ):
        raise AssertionError("full-strength projection did not secure its quota")

    audit = {
        "schema_version": 1,
        "construction": "closed_form_minimum_total_variation_prototype_transport",
        "strength": float(strength),
        "prototype_quota": int(quota),
        "prototype_core_width": int(cores.shape[1]),
        "rank_margin": float(rank_margin),
        "requested_rank_margin_feasible": bool(requested_margin_feasible),
        "simplex_boundary_target_caps": int(simplex_boundary_caps),
        "simplex_boundary_completion_used": bool(simplex_boundary_caps > 0),
        "prototype_quota_secured": bool(quota_secured),
        "maximum_requested_target": float(maximum_requested_target),
        "maximum_simplex_target_excess": float(maximum_simplex_target_excess),
        "prototype_columns": int(all_prototypes.size),
        "changed_columns": int(changed_columns.size),
        "changed_nonprototype_columns": int(changed_outside.size),
        "maximum_nonprototype_change": nonprototype_error,
        "maximum_column_sum_error": column_sum_error,
        "minimum_probability": minimum_probability,
        "maximum_projection_optimality_error": optimality_error,
        "mean_vocabulary_column_total_variation": float(np.mean(column_tv)),
        "maximum_changed_column_total_variation": float(np.max(column_tv)),
        "mean_required_owner_increase": float(np.mean(required)),
        "maximum_required_owner_increase": float(np.max(required)),
        "mean_applied_owner_increase": float(np.mean(applied)),
        "maximum_applied_owner_increase": float(np.max(applied)),
        "minimum_prototype_core_recall": float(np.min(recalls)),
        "mean_prototype_core_recall": float(np.mean(recalls)),
        "minimum_prototype_rank_slack": float(np.min(rank_slacks)),
        "affinity_sha256": _array_hash(projected),
        "required_owner_increase_sha256": _array_hash(required),
        "applied_owner_increase_sha256": _array_hash(applied),
        "selected_prototype_slots_sha256": _array_hash(selected_slots),
    }
    return MinimumDistortionProjection(
        affinity=projected,
        required_owner_increase=required,
        applied_owner_increase=applied,
        audit=audit,
    )


def centred_adapter_residual(raw_residual: Any, *, max_logit_shift: float) -> Any:
    """Return an identifiable column-centred residual with an exact bound.

    The first centring removes the topic-common non-identifiable direction.
    Tanh then bounds an intermediate value to half the requested range, and a
    second centring restores the zero-sum identity without allowing any final
    shift to exceed ``max_logit_shift`` in magnitude.
    """

    if float(max_logit_shift) <= 0.0:
        raise ValueError("maximum adapter logit shift must be positive")
    centred = raw_residual - raw_residual.mean(dim=0, keepdim=True)
    half_range = 0.5 * float(max_logit_shift)
    bounded = half_range * __import__("torch").tanh(centred / half_range)
    return bounded - bounded.mean(dim=0, keepdim=True)


def effective_affinity(
    official_affinity: Any,
    raw_residual: Any,
    *,
    max_logit_shift: float,
    epsilon: float,
) -> Any:
    """Add a log-affinity residual and renormalize over topics per word."""

    torch = __import__("torch")
    if official_affinity.shape != raw_residual.shape:
        raise ValueError("adapter residual and official affinity are incompatible")
    if float(epsilon) <= 0.0:
        raise ValueError("adapter epsilon must be positive")
    shift = centred_adapter_residual(
        raw_residual, max_logit_shift=float(max_logit_shift)
    )
    adjusted = official_affinity.clamp_min(float(epsilon)) * torch.exp(shift)
    return adjusted / adjusted.sum(dim=0, keepdim=True).clamp_min(float(epsilon))


def _rank_scores(affinity: Any, epsilon: float) -> Any:
    torch = __import__("torch")
    reporting = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(
        float(epsilon)
    )
    log_reporting = torch.log(reporting.clamp_min(float(epsilon)))
    centred = log_reporting - log_reporting.mean(dim=1, keepdim=True)
    scale = torch.sqrt(
        torch.mean(centred * centred, dim=1, keepdim=True) + float(epsilon)
    )
    return centred / scale


def prototype_lexical_terms(
    effective_topic_affinity: Any,
    positive_npmi_graph: Any,
    raw_residual: Any,
    prototype_core_indices: Any,
    *,
    top_k: int,
    temperature: float,
    overlap_weight: float,
    prototype_margin_weight: float,
    prototype_margin: float,
    unsupported_mass_weight: float,
    residual_energy_weight: float,
    max_logit_shift: float,
    epsilon: float,
) -> PrototypeLexicalTerms:
    """Return the indivisible graph-prototype lexical module objective."""

    torch = __import__("torch")
    from src.v6.core import soft_topk_membership

    affinity = effective_topic_affinity
    if affinity.ndim != 2 or positive_npmi_graph.shape != (
        affinity.shape[1], affinity.shape[1]
    ):
        raise ValueError("lexical graph and effective affinity are incompatible")
    cores = prototype_core_indices.to(dtype=torch.long, device=affinity.device)
    if cores.shape != (affinity.shape[0], int(top_k)):
        raise ValueError("prototype cores have the wrong shape")
    if min(
        float(temperature),
        float(prototype_margin),
        float(max_logit_shift),
        float(epsilon),
    ) <= 0.0:
        raise ValueError("lexical temperatures and margins must be positive")
    if min(
        float(overlap_weight),
        float(prototype_margin_weight),
        float(unsupported_mass_weight),
        float(residual_energy_weight),
    ) < 0.0:
        raise ValueError("lexical coefficients must be nonnegative")

    rank_scores = _rank_scores(affinity, float(epsilon))
    membership = soft_topk_membership(
        rank_scores, int(top_k), float(temperature), epsilon=float(epsilon)
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
    overlap = (
        pair_overlap.sum() - torch.diagonal(pair_overlap).sum()
    ) / float(max(1, topics * (topics - 1)))

    core_scores = torch.gather(rank_scores, 1, cores)
    noncore = torch.ones_like(rank_scores, dtype=torch.bool)
    noncore.scatter_(1, cores, False)
    boundary = rank_scores.masked_fill(~noncore, -torch.inf).max(dim=1).values
    prototype_margin_loss = torch.relu(
        boundary[:, None] + float(prototype_margin) - core_scores
    ).mean()

    supported = positive_npmi_graph.sum(dim=1) > 0.0
    unsupported_mass = membership[:, ~supported].sum() / float(
        topics * int(top_k)
    )
    centred = raw_residual - raw_residual.mean(dim=0, keepdim=True)
    residual_energy = torch.mean(
        (centred / float(max_logit_shift)) ** 2
    )
    total = (
        association_loss
        + float(overlap_weight) * overlap
        + float(prototype_margin_weight) * prototype_margin_loss
        + float(unsupported_mass_weight) * unsupported_mass
        + float(residual_energy_weight) * residual_energy
    )

    hard_top = torch.argsort(
        rank_scores, dim=1, descending=True, stable=True
    )[:, : int(top_k)]
    selected = torch.zeros_like(rank_scores, dtype=torch.bool)
    selected.scatter_(1, hard_top, True)
    multiplicity = selected.sum(dim=0)
    duplicate_slots = (multiplicity - 1).clamp_min(0).sum()
    recalled = torch.gather(selected, 1, cores).to(rank_scores.dtype).mean()
    return PrototypeLexicalTerms(
        total=total,
        association_loss=association_loss,
        overlap=overlap,
        prototype_margin=prototype_margin_loss,
        unsupported_mass=unsupported_mass,
        residual_energy=residual_energy,
        association=association,
        membership=membership,
        rank_scores=rank_scores,
        hard_duplicate_slots=duplicate_slots,
        prototype_core_recall=recalled,
    )


def prototype_gradient_receipt(
    official_affinity: Any,
    effective_topic_affinity: Any,
    positive_npmi_graph: Any,
    raw_residual: Any,
    prototype_core_indices: Any,
    *,
    topic_parameter: Any,
    activity_floor: float,
    **term_arguments: Any,
) -> dict[str, float | int]:
    """Audit tail-rank, parent-topic, and adapter gradient connectivity."""

    torch = __import__("torch")
    terms = prototype_lexical_terms(
        effective_topic_affinity,
        positive_npmi_graph,
        raw_residual,
        prototype_core_indices,
        **term_arguments,
    )
    rank_gradient, topic_gradient, adapter_gradient = torch.autograd.grad(
        terms.total,
        (terms.rank_scores, topic_parameter, raw_residual),
        retain_graph=False,
    )
    ranking = torch.argsort(
        effective_topic_affinity.detach(), dim=1, descending=True, stable=True
    )
    tail = ranking[:, 5 : int(term_arguments["top_k"])]
    selected = torch.gather(rank_gradient.abs(), 1, tail)
    return {
        "minimum_rank_6_to_10_gradient": float(selected.min().detach().cpu()),
        "mean_rank_6_to_10_gradient": float(selected.mean().detach().cpu()),
        "positive_rank_6_to_10_fraction": float(
            (selected > float(activity_floor)).float().mean().detach().cpu()
        ),
        "maximum_membership_mass_error": float(
            (terms.membership.sum(dim=1) - float(term_arguments["top_k"]))
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
        "prototype_margin_loss": float(terms.prototype_margin.detach().cpu()),
        "prototype_core_recall": float(terms.prototype_core_recall.detach().cpu()),
        "hard_duplicate_topk_slots": int(
            terms.hard_duplicate_slots.detach().cpu()
        ),
        "lexical_topic_embedding_gradient_norm": float(
            topic_gradient.detach().norm().cpu()
        ),
        "lexical_adapter_gradient_norm": float(
            adapter_gradient.detach().norm().cpu()
        ),
        "adapter_residual_rms": float(
            raw_residual.detach().square().mean().sqrt().cpu()
        ),
        "zero_adapter_identity_error": float(
            (
                effective_affinity(
                    official_affinity,
                    torch.zeros_like(raw_residual),
                    max_logit_shift=float(term_arguments["max_logit_shift"]),
                    epsilon=float(term_arguments["epsilon"]),
                )
                - official_affinity
            )
            .abs()
            .max()
            .detach()
            .cpu()
        ),
    }
