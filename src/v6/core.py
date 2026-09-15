"""V6 Step-1 mathematical core.

This module implements only the two proposed extensions that must later be
attached to a reproduced EnCOT backbone:

1. an identifiable, adaptive, city-conditioned low-rank lexical residual; and
2. a rank-aware positive-NPMI objective whose forward support is concentrated
   on the reported top-M words.

The document-topic logits in :class:`V6MathCore` are a synthetic-test harness,
not a replacement for the official EnCOT encoder.  No V4/V5 code is imported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class V6Data:
    """Observed evidence required by the standalone Step-1 harness."""

    counts: torch.Tensor
    vocabulary_embeddings: torch.Tensor
    positive_npmi_graph: torch.Tensor
    city_indices: torch.Tensor
    city_weights: torch.Tensor

    @property
    def documents(self) -> int:
        return int(self.counts.shape[0])

    @property
    def vocabulary_size(self) -> int:
        return int(self.counts.shape[1])

    @property
    def cities(self) -> int:
        return int(self.city_weights.numel())

    @property
    def embedding_dimension(self) -> int:
        return int(self.vocabulary_embeddings.shape[1])

    @property
    def total_tokens(self) -> torch.Tensor:
        return self.counts.sum()

    def to(self, *, device: torch.device | str, dtype: torch.dtype) -> "V6Data":
        return V6Data(
            counts=self.counts.to(device=device, dtype=dtype),
            vocabulary_embeddings=self.vocabulary_embeddings.to(
                device=device, dtype=dtype
            ),
            positive_npmi_graph=self.positive_npmi_graph.to(
                device=device, dtype=dtype
            ),
            city_indices=self.city_indices.to(device=device, dtype=torch.long),
            city_weights=self.city_weights.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class V6Weights:
    """Regularization weights for the V6 extensions.

    ``pooling`` and ``gate`` are internal controls of one city-residual module;
    they are not advertised as separate methodological contributions.
    """

    pooling: float
    gate: float
    lexical: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "V6Weights":
        result = cls(
            pooling=float(values["lambda_pooling"]),
            gate=float(values["lambda_gate"]),
            lexical=float(values["lambda_lexical"]),
        )
        if min(result.pooling, result.gate, result.lexical) < 0.0:
            raise ValueError("V6 regularization weights must be nonnegative")
        return result

    def without_lexical(self) -> "V6Weights":
        return replace(self, lexical=0.0)


@dataclass(frozen=True)
class V6Distributions:
    theta: torch.Tensor
    shared_topics: torch.Tensor
    city_topics: torch.Tensor
    document_word_probabilities: torch.Tensor
    city_residual_vectors: torch.Tensor
    gates: torch.Tensor
    effective_logit_residuals: torch.Tensor


@dataclass(frozen=True)
class V6Objective:
    total: torch.Tensor
    reconstruction_nll_per_token: torch.Tensor
    pooling: torch.Tensor
    gate_sparsity: torch.Tensor
    lexical: torch.Tensor
    mean_topk_positive_npmi: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "reconstruction_nll_per_token": float(
                self.reconstruction_nll_per_token.detach().cpu()
            ),
            "pooling": float(self.pooling.detach().cpu()),
            "gate_sparsity": float(self.gate_sparsity.detach().cpu()),
            "lexical": float(self.lexical.detach().cpu()),
            "mean_topk_positive_npmi": float(
                self.mean_topk_positive_npmi.detach().cpu()
            ),
        }


def _row_center(values: torch.Tensor) -> torch.Tensor:
    return values - values.mean(dim=-1, keepdim=True)


def weighted_contrast_basis(city_weights: torch.Tensor) -> torch.Tensor:
    """Return a full-rank C x (C-1) basis orthogonal to city weights.

    Directly learning C residuals and then centering them leaves a flat gauge
    direction.  The contrast basis removes that non-identifiability.  The city
    with greatest training mass is used as the pivot for numerical stability.
    """

    if city_weights.ndim != 1 or city_weights.numel() < 2:
        raise ValueError("At least two one-dimensional city weights are required")
    if bool((city_weights <= 0).any()):
        raise ValueError("Every city must have positive training weight")
    weights = city_weights / city_weights.sum()
    cities = int(weights.numel())
    pivot = int(torch.argmax(weights).detach().cpu())
    nonpivot = [city for city in range(cities) if city != pivot]
    basis = torch.zeros(
        (cities, cities - 1), dtype=weights.dtype, device=weights.device
    )
    for column, city in enumerate(nonpivot):
        basis[city, column] = 1.0
        basis[pivot, column] = -weights[city] / weights[pivot]
    return basis


def soft_topk_membership(
    scores: torch.Tensor,
    top_k: int,
    temperature: float,
    *,
    bisection_iterations: int = 80,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Differentiable soft membership with exactly ``top_k`` expected mass.

    For each row, a threshold t solves

        sum_i sigmoid((score_i - t) / temperature) = top_k.

    Bisection determines the forward threshold.  Its exact first-order implicit
    derivative is then attached, so gradients account for movement of the
    threshold rather than treating it as a constant.  This avoids the
    probability-mass domination of ordinary softmax surrogates and gives the
    words near ranks M-1, M, and M+1 direct optimization signal.
    """

    if scores.ndim < 2:
        raise ValueError("soft_topk_membership expects at least a 2-D tensor")
    vocabulary = int(scores.shape[-1])
    if not 0 < top_k < vocabulary:
        raise ValueError("top_k must be strictly between zero and vocabulary size")
    if temperature <= 0.0 or epsilon <= 0.0:
        raise ValueError("temperature and epsilon must be positive")
    if bisection_iterations < 20:
        raise ValueError("At least 20 bisection iterations are required")

    width = 40.0 * float(temperature)
    with torch.no_grad():
        lower = scores.min(dim=-1, keepdim=True).values - width
        upper = scores.max(dim=-1, keepdim=True).values + width
        for _ in range(bisection_iterations):
            midpoint = 0.5 * (lower + upper)
            mass = torch.sigmoid((scores - midpoint) / temperature).sum(
                dim=-1, keepdim=True
            )
            lower = torch.where(mass > float(top_k), midpoint, lower)
            upper = torch.where(mass > float(top_k), upper, midpoint)
        solved_threshold = 0.5 * (lower + upper)

    provisional = torch.sigmoid((scores - solved_threshold) / temperature)
    slope = (provisional * (1.0 - provisional)).detach()
    implicit_weights = slope / slope.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    threshold_proxy = (implicit_weights * scores).sum(dim=-1, keepdim=True)
    threshold = solved_threshold + threshold_proxy - threshold_proxy.detach()
    return torch.sigmoid((scores - threshold) / temperature)


def city_conditioned_distributions_from_logits(
    theta_logits: torch.Tensor,
    shared_topic_logits: torch.Tensor,
    city_contrast_coefficients: torch.Tensor,
    gate_logits: torch.Tensor,
    data: V6Data,
    *,
    residual_scale: float = 1.0,
) -> V6Distributions:
    """Map free parameters to simplex-valued shared/city distributions."""

    if residual_scale < 0.0:
        raise ValueError("residual_scale must be nonnegative")
    weights = data.city_weights / data.city_weights.sum()
    basis = weighted_contrast_basis(weights)
    residual_vectors = torch.einsum(
        "cr,rkd->ckd", basis, city_contrast_coefficients
    )
    semantic_scores = torch.einsum(
        "ckd,vd->ckv", residual_vectors, data.vocabulary_embeddings
    ) / math.sqrt(float(data.embedding_dimension))
    semantic_scores = _row_center(semantic_scores)
    gates = torch.sigmoid(gate_logits)
    gated_scores = gates.unsqueeze(-1) * semantic_scores
    weighted_score_mean = torch.einsum("c,ckv->kv", weights, gated_scores)
    effective_residuals = gated_scores - weighted_score_mean.unsqueeze(0)

    centered_shared_logits = _row_center(shared_topic_logits)
    theta = torch.softmax(_row_center(theta_logits), dim=1)
    shared_topics = torch.softmax(centered_shared_logits, dim=1)
    city_topics = torch.softmax(
        centered_shared_logits.unsqueeze(0)
        + float(residual_scale) * effective_residuals,
        dim=2,
    )
    document_topics = city_topics.index_select(0, data.city_indices)
    predicted_words = torch.einsum("nk,nkv->nv", theta, document_topics)
    return V6Distributions(
        theta=theta,
        shared_topics=shared_topics,
        city_topics=city_topics,
        document_word_probabilities=predicted_words,
        city_residual_vectors=residual_vectors,
        gates=gates,
        effective_logit_residuals=effective_residuals,
    )


def rank_aware_lexical_objective(
    shared_topic_logits: torch.Tensor,
    positive_npmi_graph: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, per-topic association, and soft top-k memberships."""

    centered = _row_center(shared_topic_logits)
    scale = torch.sqrt(torch.mean(centered * centered, dim=1, keepdim=True) + epsilon)
    rank_scores = centered / scale
    memberships = soft_topk_membership(
        rank_scores,
        top_k,
        temperature,
        epsilon=epsilon,
    )
    numerator = torch.einsum(
        "ki,ij,kj->k", memberships, positive_npmi_graph, memberships
    )
    mass = memberships.sum(dim=1)
    denominator = (mass * mass - torch.sum(memberships * memberships, dim=1)).clamp_min(
        epsilon
    )
    association = numerator / denominator
    loss = -torch.log(association + epsilon).mean()
    return loss, association, memberships


def objective_from_logits(
    theta_logits: torch.Tensor,
    shared_topic_logits: torch.Tensor,
    city_contrast_coefficients: torch.Tensor,
    gate_logits: torch.Tensor,
    data: V6Data,
    weights: V6Weights,
    *,
    lexical_top_k: int,
    lexical_temperature: float,
    epsilon: float,
    residual_scale: float = 1.0,
) -> V6Objective:
    distributions = city_conditioned_distributions_from_logits(
        theta_logits,
        shared_topic_logits,
        city_contrast_coefficients,
        gate_logits,
        data,
        residual_scale=residual_scale,
    )
    nll = -torch.sum(
        data.counts
        * torch.log(distributions.document_word_probabilities.clamp_min(epsilon))
    ) / data.total_tokens
    city_weights = data.city_weights / data.city_weights.sum()
    squared_residual = torch.mean(
        distributions.city_residual_vectors
        * distributions.city_residual_vectors,
        dim=(1, 2),
    )
    pooling = torch.sum(city_weights * squared_residual)
    gate_sparsity = distributions.gates.mean()
    lexical, association, _ = rank_aware_lexical_objective(
        shared_topic_logits,
        data.positive_npmi_graph,
        top_k=lexical_top_k,
        temperature=lexical_temperature,
        epsilon=epsilon,
    )
    total = (
        nll
        + weights.pooling * pooling
        + weights.gate * gate_sparsity
        + weights.lexical * lexical
    )
    return V6Objective(
        total=total,
        reconstruction_nll_per_token=nll,
        pooling=pooling,
        gate_sparsity=gate_sparsity,
        lexical=lexical,
        mean_topk_positive_npmi=association.mean(),
    )


class V6MathCore(nn.Module):
    """Standalone differentiable harness for the two V6 extensions."""

    def __init__(
        self,
        *,
        documents: int,
        topics: int,
        vocabulary: int,
        cities: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        if min(documents, topics, vocabulary, cities, embedding_dimension) <= 0:
            raise ValueError("All V6 dimensions must be positive")
        if topics < 2 or cities < 2:
            raise ValueError("V6 requires at least two topics and two cities")
        self.theta_logits = nn.Parameter(torch.zeros(documents, topics))
        self.shared_topic_logits = nn.Parameter(torch.zeros(topics, vocabulary))
        self.city_contrast_coefficients = nn.Parameter(
            torch.zeros(cities - 1, topics, embedding_dimension)
        )
        self.gate_logits = nn.Parameter(torch.full((cities, topics), -1.5))

    @property
    def topics(self) -> int:
        return int(self.shared_topic_logits.shape[0])

    def distributions(
        self, data: V6Data, *, residual_scale: float = 1.0
    ) -> V6Distributions:
        return city_conditioned_distributions_from_logits(
            self.theta_logits,
            self.shared_topic_logits,
            self.city_contrast_coefficients,
            self.gate_logits,
            data,
            residual_scale=residual_scale,
        )


def validate_data(
    data: V6Data,
    model: V6MathCore,
    *,
    lexical_top_k: int,
    tolerance: float = 2.0e-6,
) -> None:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if data.counts.ndim != 2:
        raise ValueError("counts must be N by V")
    documents, vocabulary = data.counts.shape
    if model.theta_logits.shape[0] != documents:
        raise ValueError("document dimension mismatch")
    if model.shared_topic_logits.shape[1] != vocabulary:
        raise ValueError("vocabulary dimension mismatch")
    if data.vocabulary_embeddings.ndim != 2:
        raise ValueError("vocabulary_embeddings must be V by D")
    if data.vocabulary_embeddings.shape[0] != vocabulary:
        raise ValueError("vocabulary embedding rows are not aligned")
    if data.positive_npmi_graph.shape != (vocabulary, vocabulary):
        raise ValueError("positive_npmi_graph must be V by V")
    if data.city_indices.shape != (documents,) or data.city_indices.dtype != torch.long:
        raise ValueError("city_indices must be a length-N torch.long vector")
    if data.city_weights.ndim != 1 or data.city_weights.numel() < 2:
        raise ValueError("city_weights must be a length-C vector")
    if model.city_contrast_coefficients.shape[0] != data.cities - 1:
        raise ValueError("city contrast dimension mismatch")
    if model.gate_logits.shape != (data.cities, model.topics):
        raise ValueError("city gate dimension mismatch")
    if not 0 < lexical_top_k < vocabulary:
        raise ValueError("lexical_top_k must be smaller than vocabulary")
    finite = (
        data.counts,
        data.vocabulary_embeddings,
        data.positive_npmi_graph,
        data.city_weights,
    )
    if any(not bool(torch.isfinite(value).all()) for value in finite):
        raise ValueError("V6 evidence contains non-finite values")
    if bool((data.counts < 0).any()) or float(data.total_tokens) <= 0.0:
        raise ValueError("counts require nonnegative entries and positive token mass")
    if bool((data.counts.sum(dim=1) <= 0).any()):
        raise ValueError("every document needs positive retained token mass")
    if bool((data.city_weights <= 0).any()):
        raise ValueError("every city needs positive training mass")
    if abs(float(data.city_weights.sum()) - 1.0) > tolerance:
        raise ValueError("city_weights must sum to one")
    if int(data.city_indices.min()) < 0 or int(data.city_indices.max()) >= data.cities:
        raise ValueError("city index outside the modeled range")
    graph = data.positive_npmi_graph
    if bool((graph < 0).any()) or float(graph.max()) > 1.0 + tolerance:
        raise ValueError("positive-NPMI evidence must lie in [0,1]")
    if not torch.allclose(graph, graph.T, atol=tolerance, rtol=0.0):
        raise ValueError("positive-NPMI evidence must be symmetric")
    if float(torch.max(torch.abs(torch.diagonal(graph)))) > tolerance:
        raise ValueError("positive-NPMI diagonal must be zero")
    if int(torch.count_nonzero(graph)) == 0:
        raise ValueError("positive-NPMI evidence has no supported edge")
    norms = torch.linalg.vector_norm(data.vocabulary_embeddings, dim=1)
    if float(torch.max(torch.abs(norms - 1.0))) > 5.0 * tolerance:
        raise ValueError("vocabulary embeddings must be row-normalized")


def objective(
    model: V6MathCore,
    data: V6Data,
    weights: V6Weights,
    *,
    lexical_top_k: int,
    lexical_temperature: float,
    epsilon: float,
    residual_scale: float = 1.0,
    validate: bool = False,
) -> V6Objective:
    if epsilon <= 0.0 or lexical_temperature <= 0.0:
        raise ValueError("epsilon and lexical_temperature must be positive")
    if validate:
        validate_data(
            data,
            model,
            lexical_top_k=lexical_top_k,
        )
    return objective_from_logits(
        model.theta_logits,
        model.shared_topic_logits,
        model.city_contrast_coefficients,
        model.gate_logits,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
        residual_scale=residual_scale,
    )


def parameter_counts(*, topics: int, vocabulary: int, cities: int, dimension: int) -> dict[str, int]:
    """Compare city-specific parameter capacities, excluding the backbone."""

    low_rank = (cities - 1) * topics * dimension + cities * topics
    independent = cities * topics * vocabulary
    return {
        "v6_low_rank_city_extension": int(low_rank),
        "independent_city_decoders": int(independent),
    }
