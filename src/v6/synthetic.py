"""Deterministic synthetic fixture for V6 Step 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.v6.core import (
    V6Data,
    V6MathCore,
    city_conditioned_distributions_from_logits,
)


@dataclass(frozen=True)
class SyntheticTruth:
    theta: np.ndarray
    shared_topic_logits: np.ndarray
    shared_topics: np.ndarray
    city_contrast_coefficients: np.ndarray
    gate_logits: np.ndarray
    city_topics: np.ndarray


def _unit_rows(values: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), epsilon
    )


def build_synthetic_fixture(
    *,
    seed: int,
    documents: int,
    topics: int,
    vocabulary: int,
    cities: int,
    embedding_dimension: int,
    document_length: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[V6Data, SyntheticTruth]:
    if vocabulary < max(12, topics * 10):
        raise ValueError("Synthetic vocabulary must support ten words per topic")
    if embedding_dimension < topics:
        raise ValueError("Synthetic embedding dimension must be at least K")
    rng = np.random.default_rng(seed)
    city_indices = np.arange(documents, dtype=np.int64) % cities
    rng.shuffle(city_indices)
    city_counts = np.bincount(city_indices, minlength=cities).astype(np.float64)
    city_weights = city_counts / city_counts.sum()

    anchors = _unit_rows(rng.normal(size=(topics, embedding_dimension)))
    blocks = np.array_split(np.arange(vocabulary), topics)
    vocabulary_embeddings = np.zeros(
        (vocabulary, embedding_dimension), dtype=np.float64
    )
    shared_logits = np.full((topics, vocabulary), -3.0, dtype=np.float64)
    for topic, indices in enumerate(blocks):
        vocabulary_embeddings[indices] = anchors[topic] + rng.normal(
            0.0, 0.10, size=(len(indices), embedding_dimension)
        )
        shared_logits[topic, indices] = np.linspace(
            3.4, 1.7, len(indices), dtype=np.float64
        )
    vocabulary_embeddings = _unit_rows(vocabulary_embeddings)
    shared_logits += rng.normal(0.0, 0.04, size=shared_logits.shape)

    coefficients = rng.normal(
        0.0, 0.75, size=(cities - 1, topics, embedding_dimension)
    )
    gate_logits = np.full((cities, topics), 1.1, dtype=np.float64)
    gate_logits += rng.normal(0.0, 0.15, size=gate_logits.shape)

    theta = np.zeros((documents, topics), dtype=np.float64)
    for row, city in enumerate(city_indices):
        prior = np.full(topics, 0.8, dtype=np.float64)
        prior[city % topics] += 3.0
        prior[(city + 1) % topics] += 1.0
        theta[row] = rng.dirichlet(prior)

    graph = np.zeros((vocabulary, vocabulary), dtype=np.float64)
    for indices in blocks:
        similarity = vocabulary_embeddings[indices] @ vocabulary_embeddings[indices].T
        local = np.clip(0.20 + 0.70 * similarity, 0.05, 0.95)
        np.fill_diagonal(local, 0.0)
        graph[np.ix_(indices, indices)] = local
    graph = 0.5 * (graph + graph.T)
    np.fill_diagonal(graph, 0.0)

    placeholder = V6Data(
        counts=torch.ones((documents, vocabulary), dtype=dtype),
        vocabulary_embeddings=torch.as_tensor(vocabulary_embeddings, dtype=dtype),
        positive_npmi_graph=torch.as_tensor(graph, dtype=dtype),
        city_indices=torch.as_tensor(city_indices, dtype=torch.long),
        city_weights=torch.as_tensor(city_weights, dtype=dtype),
    )
    distributions = city_conditioned_distributions_from_logits(
        torch.as_tensor(np.log(theta), dtype=dtype),
        torch.as_tensor(shared_logits, dtype=dtype),
        torch.as_tensor(coefficients, dtype=dtype),
        torch.as_tensor(gate_logits, dtype=dtype),
        placeholder,
    )
    city_topics = distributions.city_topics.detach().cpu().numpy()
    shared_topics = distributions.shared_topics.detach().cpu().numpy()
    counts = np.zeros((documents, vocabulary), dtype=np.float64)
    for row, city in enumerate(city_indices):
        word_probability = theta[row] @ city_topics[city]
        counts[row] = rng.multinomial(document_length, word_probability)

    data = V6Data(
        counts=torch.as_tensor(counts, dtype=dtype),
        vocabulary_embeddings=torch.as_tensor(vocabulary_embeddings, dtype=dtype),
        positive_npmi_graph=torch.as_tensor(graph, dtype=dtype),
        city_indices=torch.as_tensor(city_indices, dtype=torch.long),
        city_weights=torch.as_tensor(city_weights, dtype=dtype),
    )
    truth = SyntheticTruth(
        theta=theta,
        shared_topic_logits=shared_logits,
        shared_topics=shared_topics,
        city_contrast_coefficients=coefficients,
        gate_logits=gate_logits,
        city_topics=city_topics,
    )
    return data, truth


def initialize_from_perturbed_truth(
    model: V6MathCore,
    truth: SyntheticTruth,
    *,
    seed: int,
    noise_standard_deviation: float,
) -> None:
    """Create an interior test point; this is not a real-data initializer."""

    rng = np.random.default_rng(seed)
    theta_logits = np.log(np.clip(truth.theta, 1.0e-10, None))
    values = {
        "theta_logits": theta_logits,
        "shared_topic_logits": truth.shared_topic_logits,
        "city_contrast_coefficients": truth.city_contrast_coefficients,
        "gate_logits": truth.gate_logits,
    }
    with torch.no_grad():
        for name, reference in values.items():
            parameter = getattr(model, name)
            noisy = reference + rng.normal(
                0.0, noise_standard_deviation, size=reference.shape
            )
            parameter.copy_(
                torch.as_tensor(
                    noisy, dtype=parameter.dtype, device=parameter.device
                )
            )

