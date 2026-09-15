"""Independent NumPy oracle for V6 probability calculations.

This file deliberately does not import PyTorch or :mod:`src.v6.core`.  Step 1
uses it to catch a consistent-but-wrong implementation in the production
equations.
"""

from __future__ import annotations

import math

import numpy as np


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def weighted_contrast_basis(city_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(city_weights, dtype=np.float64)
    weights = weights / weights.sum()
    cities = len(weights)
    if cities < 2 or np.any(weights <= 0.0):
        raise ValueError("At least two positive city weights are required")
    pivot = int(np.argmax(weights))
    nonpivot = [city for city in range(cities) if city != pivot]
    basis = np.zeros((cities, cities - 1), dtype=np.float64)
    for column, city in enumerate(nonpivot):
        basis[city, column] = 1.0
        basis[pivot, column] = -weights[city] / weights[pivot]
    return basis


def soft_topk_membership(
    scores: np.ndarray,
    top_k: int,
    temperature: float,
    iterations: int = 80,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 2 or not 0 < top_k < scores.shape[-1]:
        raise ValueError("Invalid scores or top_k")
    width = 40.0 * float(temperature)
    lower = scores.min(axis=-1, keepdims=True) - width
    upper = scores.max(axis=-1, keepdims=True) + width
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        logits = np.clip((scores - midpoint) / temperature, -60.0, 60.0)
        mass = (1.0 / (1.0 + np.exp(-logits))).sum(axis=-1, keepdims=True)
        lower = np.where(mass > float(top_k), midpoint, lower)
        upper = np.where(mass > float(top_k), upper, midpoint)
    threshold = 0.5 * (lower + upper)
    logits = np.clip((scores - threshold) / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def distributions(
    *,
    theta_logits: np.ndarray,
    shared_topic_logits: np.ndarray,
    city_contrast_coefficients: np.ndarray,
    gate_logits: np.ndarray,
    vocabulary_embeddings: np.ndarray,
    city_indices: np.ndarray,
    city_weights: np.ndarray,
    residual_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    weights = np.asarray(city_weights, dtype=np.float64)
    weights /= weights.sum()
    basis = weighted_contrast_basis(weights)
    residual_vectors = np.einsum(
        "cr,rkd->ckd", basis, city_contrast_coefficients
    )
    scores = np.einsum(
        "ckd,vd->ckv", residual_vectors, vocabulary_embeddings
    ) / math.sqrt(float(vocabulary_embeddings.shape[1]))
    scores -= scores.mean(axis=2, keepdims=True)
    gates = 1.0 / (1.0 + np.exp(-gate_logits))
    gated = gates[:, :, None] * scores
    effective = gated - np.einsum("c,ckv->kv", weights, gated)[None, :, :]
    centered_shared = shared_topic_logits - shared_topic_logits.mean(
        axis=1, keepdims=True
    )
    theta = softmax(
        theta_logits - theta_logits.mean(axis=1, keepdims=True), axis=1
    )
    shared = softmax(centered_shared, axis=1)
    city_topics = softmax(
        centered_shared[None, :, :] + float(residual_scale) * effective,
        axis=2,
    )
    document_topics = city_topics[np.asarray(city_indices, dtype=np.int64)]
    predicted = np.einsum("nk,nkv->nv", theta, document_topics)
    return {
        "theta": theta,
        "shared_topics": shared,
        "city_topics": city_topics,
        "document_word_probabilities": predicted,
        "city_residual_vectors": residual_vectors,
        "gates": gates,
        "effective_logit_residuals": effective,
    }


def rank_aware_association(
    shared_topic_logits: np.ndarray,
    positive_npmi_graph: np.ndarray,
    *,
    top_k: int,
    temperature: float,
    epsilon: float,
) -> dict[str, np.ndarray | float]:
    centered = shared_topic_logits - shared_topic_logits.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True) + epsilon)
    membership = soft_topk_membership(centered / scale, top_k, temperature)
    numerator = np.einsum(
        "ki,ij,kj->k", membership, positive_npmi_graph, membership
    )
    mass = membership.sum(axis=1)
    denominator = np.maximum(
        mass * mass - np.sum(membership * membership, axis=1), epsilon
    )
    association = numerator / denominator
    return {
        "membership": membership,
        "association": association,
        "loss": float(-np.log(association + epsilon).mean()),
    }

