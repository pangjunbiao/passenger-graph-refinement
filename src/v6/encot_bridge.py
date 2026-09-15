"""Exact bridge between the pinned EnCOT core and the V6 extensions.

The official implementation calls its K-by-V matrix ``beta`` but normalizes
that matrix over the topic axis (``dim=0``), not over vocabulary.  This module
preserves that behavior exactly.  A row-normalized copy is exposed only for
top-word ranking and never replaces the official reconstruction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def pairwise_squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Match the pinned official ``pairwise_euclidean_distance`` function."""

    return (
        torch.sum(x**2, dim=1, keepdim=True)
        + torch.sum(y**2, dim=1)
        - 2.0 * torch.matmul(x, y.t())
    )


def official_affinity_logits(
    topic_embeddings: torch.Tensor,
    word_embeddings: torch.Tensor,
    beta_temperature: float,
) -> torch.Tensor:
    if float(beta_temperature) <= 0.0:
        raise ValueError("beta_temperature must be positive")
    return -pairwise_squared_euclidean(topic_embeddings, word_embeddings) / float(
        beta_temperature
    )


def official_affinity_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the official K-by-V affinity, normalized over K for each word."""

    if logits.ndim != 2:
        raise ValueError("official affinity logits must have shape K-by-V")
    return F.softmax(logits, dim=0)


def reporting_topic_distribution(affinity: torch.Tensor) -> torch.Tensor:
    """Row-normalize for metrics while preserving every within-topic ranking."""

    if affinity.ndim != 2 or torch.any(affinity < 0):
        raise ValueError("affinity must be a nonnegative K-by-V matrix")
    totals = affinity.sum(dim=1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("affinity contains an empty topic row")
    return affinity / totals


def city_conditioned_affinity(
    shared_logits: torch.Tensor,
    city_logit_residuals: torch.Tensor,
    *,
    residual_scale: float,
) -> torch.Tensor:
    """Insert C-by-K-by-V city residuals before EnCOT's topic-axis softmax."""

    if shared_logits.ndim != 2:
        raise ValueError("shared_logits must have shape K-by-V")
    if city_logit_residuals.ndim != 3:
        raise ValueError("city_logit_residuals must have shape C-by-K-by-V")
    if tuple(city_logit_residuals.shape[1:]) != tuple(shared_logits.shape):
        raise ValueError("city residual and shared-logit dimensions disagree")
    # This explicit branch is part of the parent-identity contract.  Calling
    # softmax on an expanded C-by-K-by-V tensor can select a different kernel
    # from the official K-by-V call and introduce platform-dependent rounding
    # at roughly 1e-18.  Reuse the exact official result when the extension is
    # disabled so the identity is bitwise, not merely tolerance-close.
    if float(residual_scale) == 0.0:
        official = official_affinity_from_logits(shared_logits)
        return official.unsqueeze(0).expand(
            city_logit_residuals.shape[0], -1, -1
        )
    values = shared_logits.unsqueeze(0) + float(residual_scale) * city_logit_residuals
    return F.softmax(values, dim=1)


def exact_document_topic_mass(
    global_theta: torch.Tensor, local_noise_theta: torch.Tensor
) -> torch.Tensor:
    """Preserve EnCOT's unnormalized elementwise product."""

    if global_theta.shape != local_noise_theta.shape or global_theta.ndim != 2:
        raise ValueError("global and local topic tensors must share N-by-K shape")
    return global_theta * local_noise_theta


def decode_word_probabilities(
    document_topic_mass: torch.Tensor,
    affinity: torch.Tensor,
    decoder_batch_norm: nn.Module,
) -> torch.Tensor:
    """Use EnCOT's decoder BatchNorm and vocabulary softmax unchanged."""

    if document_topic_mass.ndim != 2 or affinity.ndim != 2:
        raise ValueError("document_topic_mass and affinity must be matrices")
    if document_topic_mass.shape[1] != affinity.shape[0]:
        raise ValueError("topic dimensions do not agree")
    logits = decoder_batch_norm(torch.matmul(document_topic_mass, affinity))
    return F.softmax(logits, dim=-1)


@dataclass(frozen=True)
class EnCOTInference:
    document_topic_mass: torch.Tensor
    word_probabilities: torch.Tensor


class ExactEnCOTAdapter(nn.Module):
    """Thin, attribution-preserving wrapper around the exact official class."""

    def __init__(self, official_model: nn.Module) -> None:
        super().__init__()
        self.official_model = official_model

    def forward(self, input_data: torch.Tensor, **kwargs: Any) -> Any:
        return self.official_model(input_data, **kwargs)

    def shared_affinity_logits(self) -> torch.Tensor:
        return official_affinity_logits(
            self.official_model.topic_embeddings,
            self.official_model.word_embeddings,
            float(self.official_model.beta_temp),
        )

    def official_affinity(self) -> torch.Tensor:
        return official_affinity_from_logits(self.shared_affinity_logits())

    def reporting_topics(self) -> torch.Tensor:
        return reporting_topic_distribution(self.official_affinity())

    def infer(self, input_data: torch.Tensor) -> EnCOTInference:
        vocabulary_size = int(self.official_model.vocab_size)
        if input_data.ndim != 2 or input_data.shape[1] != 2 * vocabulary_size:
            raise ValueError("EnCOT input must concatenate local and global BoW")
        local_x = input_data[:, :vocabulary_size]
        global_x = input_data[:, vocabulary_size:]
        local_theta, _ = self.official_model.noise_local_encode(local_x)
        global_theta, _ = self.official_model.global_encode(global_x)
        mass = exact_document_topic_mass(global_theta, local_theta)
        probabilities = decode_word_probabilities(
            mass, self.official_affinity(), self.official_model.decoder_bn
        )
        return EnCOTInference(
            document_topic_mass=mass,
            word_probabilities=probabilities,
        )
