"""Collapsed-Gibbs Dirichlet multinomial mixture (GSDMM) baseline."""

from __future__ import annotations

import math

import numpy as np

from src.v6.comparators.base import BaselineAdapter, BaselineData, BaselineFit, normalize_rows


def _document_counts(document: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    words, counts = np.unique(np.asarray(document, dtype=np.int64), return_counts=True)
    return words, counts.astype(np.int64)


def _log_cluster_scores(
    words: np.ndarray,
    counts: np.ndarray,
    cluster_documents: np.ndarray,
    cluster_tokens: np.ndarray,
    cluster_words: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    topics, vocabulary = cluster_words.shape
    length = int(counts.sum())
    scores = np.log(cluster_documents + alpha)
    for topic in range(topics):
        score = math.lgamma(cluster_tokens[topic] + vocabulary * beta)
        score -= math.lgamma(cluster_tokens[topic] + length + vocabulary * beta)
        score += sum(
            math.lgamma(cluster_words[topic, word] + count + beta)
            - math.lgamma(cluster_words[topic, word] + beta)
            for word, count in zip(words, counts)
        )
        scores[topic] += score
    scores -= scores.max()
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum()


class GSDMMAdapter(BaselineAdapter):
    method_id = "gsdmm"
    display_name = "GSDMM"

    def fit(self, data: BaselineData, *, topics: int, seed: int) -> BaselineFit:
        alpha = float(self.config.get("alpha", 0.1))
        beta = float(self.config.get("beta", 0.01))
        maximum = int(self.config["maximum_iterations"])
        if alpha <= 0.0 or beta <= 0.0 or maximum < 1:
            raise ValueError("GSDMM priors and iteration budget must be positive")
        rng = np.random.default_rng(seed)
        documents = tuple(_document_counts(doc) for doc in data.train_token_documents)
        assignments = rng.integers(0, topics, size=len(documents), dtype=np.int64)
        cluster_documents = np.zeros(topics, dtype=np.int64)
        cluster_tokens = np.zeros(topics, dtype=np.int64)
        cluster_words = np.zeros((topics, data.vocabulary_size), dtype=np.int64)
        for assignment, (words, counts) in zip(assignments, documents):
            cluster_documents[assignment] += 1
            cluster_tokens[assignment] += int(counts.sum())
            cluster_words[assignment, words] += counts

        moved_history: list[int] = []
        stable_required = int(self.config.get("stable_iterations", 5))
        stable = 0
        for _iteration in range(1, maximum + 1):
            moved = 0
            for index, (words, counts) in enumerate(documents):
                old = int(assignments[index])
                cluster_documents[old] -= 1
                cluster_tokens[old] -= int(counts.sum())
                cluster_words[old, words] -= counts
                probabilities = _log_cluster_scores(
                    words,
                    counts,
                    cluster_documents,
                    cluster_tokens,
                    cluster_words,
                    alpha=alpha,
                    beta=beta,
                )
                new = int(rng.choice(topics, p=probabilities))
                assignments[index] = new
                cluster_documents[new] += 1
                cluster_tokens[new] += int(counts.sum())
                cluster_words[new, words] += counts
                moved += int(new != old)
            moved_history.append(moved)
            stable = stable + 1 if moved == 0 else 0
            if stable >= stable_required:
                break

        topic_word = normalize_rows(cluster_words + beta)
        validation_theta = np.vstack(
            [
                _log_cluster_scores(
                    *_document_counts(document),
                    cluster_documents,
                    cluster_tokens,
                    cluster_words,
                    alpha=alpha,
                    beta=beta,
                )
                for document in data.validation_token_documents
            ]
        )
        return BaselineFit(
            self.method_id,
            self.display_name,
            seed,
            topic_word,
            validation_theta,
            True,
            topics,
            iterations_completed=len(moved_history),
            converged=stable >= stable_required,
            metadata={
                "implementation": "project_collapsed_gibbs_dirichlet_multinomial_mixture",
                "topic_count_semantics": "fixed_maximum_clusters_with_prior_defined_empty_components",
                "occupied_topics": int(np.count_nonzero(cluster_documents)),
                "cluster_document_counts": cluster_documents.tolist(),
                "alpha": alpha,
                "beta": beta,
                "documents_moved_final_iteration": int(moved_history[-1]),
            },
        )
