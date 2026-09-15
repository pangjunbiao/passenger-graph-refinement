"""V6 extensions attached to, but never substituted for, official EnCOT.

The wrapper deliberately delegates the extensions-off path to the official
class.  Consequently ``residual_scale == 0`` and ``lambda_lexical == 0`` is
the exact parent computation rather than a numerically similar reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.v6.core import soft_topk_membership, weighted_contrast_basis
from src.v6.encot_bridge import (
    city_conditioned_affinity,
    exact_document_topic_mass,
    official_affinity_logits,
    reporting_topic_distribution,
)


@dataclass(frozen=True)
class ExtensionWeights:
    """Weights expressed on a per-augmented-token objective scale."""

    pooling: float = 0.0
    gate: float = 0.0
    lexical: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExtensionWeights":
        result = cls(
            pooling=float(values.get("lambda_pooling", 0.0)),
            gate=float(values.get("lambda_gate", 0.0)),
            lexical=float(values.get("lambda_lexical", 0.0)),
        )
        if min(result.pooling, result.gate, result.lexical) < 0.0:
            raise ValueError("extension weights must be nonnegative")
        return result


@dataclass(frozen=True)
class CityResidualState:
    residual_vectors: torch.Tensor
    semantic_scores: torch.Tensor
    gates: torch.Tensor
    effective_logit_residuals: torch.Tensor


def rank_aware_lexical_loss(
    official_affinity: torch.Tensor,
    positive_npmi_graph: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize the displayed EnCOT ranking using soft top-``top_k`` mass.

    The official affinity is first row-normalized exactly as it is for topic
    reporting.  Standardizing its log removes row constants but preserves the
    displayed within-topic ranking.
    """

    if official_affinity.ndim != 2:
        raise ValueError("official_affinity must be K-by-V")
    if positive_npmi_graph.shape != (
        official_affinity.shape[1],
        official_affinity.shape[1],
    ):
        raise ValueError("positive-NPMI graph and affinity are incompatible")
    if temperature <= 0.0 or epsilon <= 0.0:
        raise ValueError("temperature and epsilon must be positive")
    if not 1 < int(top_k) < official_affinity.shape[1]:
        raise ValueError("top_k must lie between one and vocabulary size")

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
    loss = -torch.log(association + float(epsilon)).mean()
    return loss, association, membership, rank_scores


class V6EnCOT(nn.Module):
    """Exact EnCOT parent plus the two declared V6 extensions."""

    def __init__(
        self,
        official_model: nn.Module,
        *,
        frozen_vocabulary_features: torch.Tensor,
        city_weights: torch.Tensor,
        gate_initial_logit: float = 0.0,
    ) -> None:
        super().__init__()
        if frozen_vocabulary_features.ndim != 2:
            raise ValueError("frozen vocabulary features must be V-by-D")
        vocabulary = int(official_model.vocab_size)
        topics = int(official_model.num_topics)
        if frozen_vocabulary_features.shape[0] != vocabulary:
            raise ValueError("official vocabulary and frozen features disagree")
        if city_weights.ndim != 1 or city_weights.numel() < 2:
            raise ValueError("at least two city weights are required")
        if torch.any(city_weights <= 0):
            raise ValueError("every city must have positive training weight")
        normalized_features = F.normalize(
            frozen_vocabulary_features.detach().float(), dim=1
        )
        normalized_weights = city_weights.detach().float()
        normalized_weights = normalized_weights / normalized_weights.sum()

        self.parent = official_model
        self.register_buffer("frozen_vocabulary_features", normalized_features)
        self.register_buffer("city_weights", normalized_weights)
        cities = int(normalized_weights.numel())
        dimension = int(normalized_features.shape[1])
        self.city_contrast_coefficients = nn.Parameter(
            torch.zeros(cities - 1, topics, dimension)
        )
        self.gate_logits = nn.Parameter(
            torch.full((cities, topics), float(gate_initial_logit))
        )

    @property
    def cities(self) -> int:
        return int(self.city_weights.numel())

    @property
    def topics(self) -> int:
        return int(self.parent.num_topics)

    @property
    def vocabulary_size(self) -> int:
        return int(self.parent.vocab_size)

    @property
    def embedding_dimension(self) -> int:
        return int(self.frozen_vocabulary_features.shape[1])

    def normalized_city_weights(self) -> torch.Tensor:
        """Normalize in the model's current dtype after every dtype/device cast.

        Buffers created in float32 can acquire a tiny non-unit sum when a test
        or audit later converts the module to float64.  Re-normalizing at the
        point of use keeps the weighted-centering identity independent of that
        representational roundoff.
        """

        return self.city_weights / self.city_weights.sum()

    def shared_logits(self) -> torch.Tensor:
        return official_affinity_logits(
            self.parent.topic_embeddings,
            self.parent.word_embeddings,
            float(self.parent.beta_temp),
        )

    def official_affinity(self) -> torch.Tensor:
        return self.parent.get_beta()

    def reporting_topics(self) -> torch.Tensor:
        return reporting_topic_distribution(self.official_affinity())

    def city_residual_state(self) -> CityResidualState:
        weights = self.normalized_city_weights()
        basis = weighted_contrast_basis(weights)
        residual_vectors = torch.einsum(
            "cr,rkd->ckd", basis, self.city_contrast_coefficients
        )
        semantic_scores = torch.einsum(
            "ckd,vd->ckv", residual_vectors, self.frozen_vocabulary_features
        ) / math.sqrt(float(self.embedding_dimension))
        gates = torch.sigmoid(self.gate_logits)
        gated = gates.unsqueeze(-1) * semantic_scores
        # The official affinity softmax is over K.  Topic centering therefore
        # removes its null direction before prevalence centering over cities.
        topic_centered = gated - gated.mean(dim=1, keepdim=True)
        city_mean = torch.einsum("c,ckv->kv", weights, topic_centered)
        effective = topic_centered - city_mean.unsqueeze(0)
        # Reapply the two commuting projections once.  Analytically this is an
        # identity; numerically it removes the last few ulps left by reductions
        # and keeps both constraints tight after float32 <-> float64 casts.
        effective = effective - effective.mean(dim=1, keepdim=True)
        city_roundoff = torch.einsum("c,ckv->kv", weights, effective)
        effective = effective - city_roundoff.unsqueeze(0)
        return CityResidualState(
            residual_vectors=residual_vectors,
            semantic_scores=semantic_scores,
            gates=gates,
            effective_logit_residuals=effective,
        )

    def city_affinities(self, *, residual_scale: float) -> torch.Tensor:
        state = self.city_residual_state()
        return city_conditioned_affinity(
            self.shared_logits(),
            state.effective_logit_residuals,
            residual_scale=float(residual_scale),
        )

    def _validate_input(
        self, input_data: torch.Tensor, city_indices: torch.Tensor | None
    ) -> None:
        if input_data.ndim != 2 or input_data.shape[1] != 2 * self.vocabulary_size:
            raise ValueError("EnCOT input must concatenate local and global BoW")
        if city_indices is not None:
            if city_indices.shape != (input_data.shape[0],):
                raise ValueError("city indices and documents are misaligned")
            if city_indices.dtype != torch.long:
                raise ValueError("city indices must use torch.long")
            if int(city_indices.min()) < 0 or int(city_indices.max()) >= self.cities:
                raise ValueError("city index outside the fitted city universe")

    def _latent_mass(
        self, input_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_x = input_data[:, : self.vocabulary_size]
        global_x = input_data[:, self.vocabulary_size :]
        local_theta, local_kl = self.parent.noise_local_encode(local_x)
        global_theta, _ = self.parent.global_encode(global_x)
        return exact_document_topic_mass(global_theta, local_theta), local_kl, global_theta

    def decode(
        self,
        document_topic_mass: torch.Tensor,
        city_indices: torch.Tensor | None,
        *,
        residual_scale: float,
    ) -> torch.Tensor:
        if float(residual_scale) == 0.0:
            affinity = self.official_affinity()
            logits = self.parent.decoder_bn(document_topic_mass @ affinity)
        else:
            if city_indices is None:
                raise ValueError("city indices are required when the residual is active")
            affinities = self.city_affinities(residual_scale=float(residual_scale))
            selected = affinities.index_select(0, city_indices)
            logits = self.parent.decoder_bn(
                torch.einsum("nk,nkv->nv", document_topic_mass, selected)
            )
        return F.softmax(logits, dim=-1)

    def regularizers(
        self,
        positive_npmi_graph: torch.Tensor | None,
        *,
        lexical_top_k: int,
        lexical_temperature: float,
        epsilon: float,
    ) -> dict[str, torch.Tensor]:
        state = self.city_residual_state()
        squared = torch.mean(
            state.residual_vectors * state.residual_vectors, dim=(1, 2)
        )
        pooling = torch.sum(self.normalized_city_weights() * squared)
        gate = state.gates.mean()
        if positive_npmi_graph is None:
            lexical = pooling.new_zeros(())
            association = pooling.new_zeros((self.topics,))
            membership = pooling.new_zeros((self.topics, self.vocabulary_size))
            rank_scores = membership
        else:
            lexical, association, membership, rank_scores = rank_aware_lexical_loss(
                self.official_affinity(),
                positive_npmi_graph,
                top_k=int(lexical_top_k),
                temperature=float(lexical_temperature),
                epsilon=float(epsilon),
            )
        return {
            "pooling": pooling,
            "gate_sparsity": gate,
            "lexical": lexical,
            "lexical_association": association,
            "lexical_membership": membership,
            "lexical_rank_scores": rank_scores,
        }

    def reconstruction_objective(
        self,
        input_data: torch.Tensor,
        city_indices: torch.Tensor,
        *,
        residual_scale: float,
        weights: ExtensionWeights,
        positive_npmi_graph: torch.Tensor | None = None,
        lexical_top_k: int = 10,
        lexical_temperature: float = 0.2,
        epsilon: float = 1.0e-12,
    ) -> dict[str, torch.Tensor]:
        """Loss used when the exact parent is frozen during adapter fitting.

        Terms in the official objective that are constant with respect to the
        city adapter are omitted here.  The reconstruction term and its
        ``alpha_augment`` target remain exactly the official definitions.
        """

        self._validate_input(input_data, city_indices)
        mass, _, _ = self._latent_mass(input_data)
        probabilities = self.decode(
            mass, city_indices, residual_scale=float(residual_scale)
        )
        local_x = input_data[:, : self.vocabulary_size]
        global_x = input_data[:, self.vocabulary_size :]
        target = local_x + float(self.parent.alpha_augment) * global_x
        reconstruction = -(target * probabilities.clamp_min(epsilon).log()).sum(
            dim=1
        ).mean()
        scale = target.sum(dim=1).mean().detach().clamp_min(1.0)
        regularizers = self.regularizers(
            positive_npmi_graph,
            lexical_top_k=lexical_top_k,
            lexical_temperature=lexical_temperature,
            epsilon=epsilon,
        )
        extension = scale * (
            weights.pooling * regularizers["pooling"]
            + weights.gate * regularizers["gate_sparsity"]
            + weights.lexical * regularizers["lexical"]
        )
        return {
            "loss": reconstruction + extension,
            "reconstruction": reconstruction,
            "regularizer_scale": scale,
            "extension_penalty": extension,
            **regularizers,
        }

    def forward(
        self,
        input_data: torch.Tensor,
        city_indices: torch.Tensor | None = None,
        *,
        residual_scale: float = 1.0,
        weights: ExtensionWeights | None = None,
        positive_npmi_graph: torch.Tensor | None = None,
        lexical_top_k: int = 10,
        lexical_temperature: float = 0.2,
        epsilon: float = 1.0e-12,
        is_ECR: bool = True,
    ) -> dict[str, torch.Tensor]:
        weights = weights or ExtensionWeights()
        self._validate_input(input_data, city_indices)

        # Bitwise parent identity: use the official implementation itself.
        if float(residual_scale) == 0.0:
            base = dict(self.parent(input_data, is_ECR=is_ECR))
            regularizers = self.regularizers(
                positive_npmi_graph,
                lexical_top_k=lexical_top_k,
                lexical_temperature=lexical_temperature,
                epsilon=epsilon,
            )
            local_x = input_data[:, : self.vocabulary_size]
            global_x = input_data[:, self.vocabulary_size :]
            scale = (
                local_x + float(self.parent.alpha_augment) * global_x
            ).sum(dim=1).mean().detach().clamp_min(1.0)
            extension = scale * weights.lexical * regularizers["lexical"]
            base["loss_parent"] = base["loss"]
            base["loss"] = base["loss"] + extension
            base["extension_penalty"] = extension
            base["regularizer_scale"] = scale
            base.update(regularizers)
            return base

        if city_indices is None:
            raise ValueError("city indices are required when the residual is active")
        local_x = input_data[:, : self.vocabulary_size]
        global_x = input_data[:, self.vocabulary_size :]
        local_theta, local_kl = self.parent.noise_local_encode(local_x)
        global_theta, _ = self.parent.global_encode(global_x)
        mass = exact_document_topic_mass(global_theta, local_theta)
        probabilities = self.decode(
            mass, city_indices, residual_scale=float(residual_scale)
        )
        target = local_x + float(self.parent.alpha_augment) * global_x
        reconstruction = -(target * probabilities.clamp_min(epsilon).log()).sum(
            dim=1
        ).mean()
        loss_tm = reconstruction + local_kl
        loss_ecr = self.parent.get_loss_ECR() if is_ECR else reconstruction.new_zeros(())
        ot_doc = self.parent.compute_ot_loss_doc_cluster(mass)
        ot_topic = self.parent.compute_ot_loss_topic_cluster()
        parent_with_city = (
            loss_tm
            + loss_ecr
            + float(self.parent.weight_ot_doc_cluster) * ot_doc
            + float(self.parent.weight_ot_topic_cluster) * ot_topic
        )
        regularizers = self.regularizers(
            positive_npmi_graph,
            lexical_top_k=lexical_top_k,
            lexical_temperature=lexical_temperature,
            epsilon=epsilon,
        )
        scale = target.sum(dim=1).mean().detach().clamp_min(1.0)
        extension = scale * (
            weights.pooling * regularizers["pooling"]
            + weights.gate * regularizers["gate_sparsity"]
            + weights.lexical * regularizers["lexical"]
        )
        return {
            "loss": parent_with_city + extension,
            "loss_parent": parent_with_city,
            "loss_TM": loss_tm,
            "loss_ECR": loss_ecr,
            "ot_loss_doc_cluster": ot_doc,
            "ot_loss_topic_cluster": ot_topic,
            "extension_penalty": extension,
            "regularizer_scale": scale,
            **regularizers,
        }

    def infer_probabilities(
        self,
        input_data: torch.Tensor,
        city_indices: torch.Tensor | None,
        *,
        residual_scale: float,
    ) -> torch.Tensor:
        self._validate_input(input_data, city_indices)
        mass, _, _ = self._latent_mass(input_data)
        return self.decode(
            mass, city_indices, residual_scale=float(residual_scale)
        )

    def initialize_city_residual_from_counts(
        self,
        counts: np.ndarray,
        city_indices: np.ndarray,
        *,
        shrinkage: float,
        pseudocount: float,
        strength: float,
        projection_ridge: float,
        log_odds_clip: float,
    ) -> dict[str, float]:
        """Fold-local empirical-Bayes initialization; never reads held-out rows."""

        values = np.asarray(counts, dtype=np.float64)
        cities = np.asarray(city_indices, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != self.vocabulary_size:
            raise ValueError("initializer counts must be N-by-V")
        if cities.shape != (values.shape[0],):
            raise ValueError("initializer city rows are misaligned")
        if not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        if min(float(pseudocount), float(projection_ridge)) <= 0.0:
            raise ValueError("pseudocount and projection ridge must be positive")
        if min(float(strength), float(log_odds_clip)) < 0.0:
            raise ValueError("initializer strength and clipping must be nonnegative")

        global_counts = values.sum(axis=0)
        global_probability = (global_counts + float(pseudocount)) / (
            global_counts.sum() + float(pseudocount) * self.vocabulary_size
        )
        city_probability = []
        for city in range(self.cities):
            selected = values[cities == city].sum(axis=0)
            empirical = (selected + float(pseudocount)) / (
                selected.sum() + float(pseudocount) * self.vocabulary_size
            )
            city_probability.append(
                (1.0 - float(shrinkage)) * global_probability
                + float(shrinkage) * empirical
            )
        log_odds = np.log(np.asarray(city_probability)) - np.log(global_probability)
        log_odds = np.clip(log_odds, -float(log_odds_clip), float(log_odds_clip))
        pi = (
            self.normalized_city_weights()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        log_odds -= np.einsum("c,cv->v", pi, log_odds)[None, :]

        with torch.no_grad():
            affinity = self.official_affinity().detach().cpu().numpy().astype(np.float64)
        signature = affinity - affinity.mean(axis=0, keepdims=True)
        targets = log_odds[:, None, :] * signature[None, :, :]
        targets -= targets.mean(axis=1, keepdims=True)
        targets -= np.einsum("c,ckv->kv", pi, targets)[None, :, :]

        features = (
            self.frozen_vocabulary_features.detach().cpu().numpy().astype(np.float64)
        )
        diagonal = np.sum(features * features, axis=0) + float(projection_ridge)
        raw_city_vectors = np.einsum("ckv,vd->ckd", targets, features) / diagonal
        gate = float(torch.sigmoid(self.gate_logits.detach()).mean().cpu())
        raw_city_vectors *= (
            float(strength) * math.sqrt(float(self.embedding_dimension)) / max(gate, 1.0e-3)
        )
        raw_city_vectors -= np.einsum(
            "c,ckd->kd", pi, raw_city_vectors
        )[None, :, :]
        basis = weighted_contrast_basis(self.city_weights).detach().cpu().numpy()
        coefficients = np.einsum(
            "rc,ckd->rkd", np.linalg.pinv(basis), raw_city_vectors
        )
        with torch.no_grad():
            self.city_contrast_coefficients.copy_(
                torch.as_tensor(
                    coefficients,
                    dtype=self.city_contrast_coefficients.dtype,
                    device=self.city_contrast_coefficients.device,
                )
            )
        state = self.city_residual_state()
        weighted_error = torch.max(
            torch.abs(
                torch.einsum(
                    "c,ckv->kv", self.city_weights, state.effective_logit_residuals
                )
            )
        )
        topic_error = torch.max(
            torch.abs(state.effective_logit_residuals.sum(dim=1))
        )
        return {
            "maximum_absolute_log_odds": float(np.max(np.abs(log_odds))),
            "coefficient_rms": float(
                torch.sqrt(
                    torch.mean(
                        self.city_contrast_coefficients.detach()
                        * self.city_contrast_coefficients.detach()
                    )
                ).cpu()
            ),
            "effective_residual_rms": float(
                torch.sqrt(
                    torch.mean(
                        state.effective_logit_residuals.detach()
                        * state.effective_logit_residuals.detach()
                    )
                ).cpu()
            ),
            "weighted_city_center_error": float(weighted_error.cpu()),
            "topic_null_direction_error": float(topic_error.cpu()),
        }


def extension_parameter_counts(
    *, cities: int, topics: int, vocabulary: int, embedding_dimension: int
) -> dict[str, int]:
    low_rank = (int(cities) - 1) * int(topics) * int(embedding_dimension)
    low_rank += int(cities) * int(topics)
    independent = int(cities) * int(topics) * int(vocabulary)
    return {
        "v6_city_extension": low_rank,
        "independent_city_affinities": independent,
        "saving": independent - low_rank,
    }
