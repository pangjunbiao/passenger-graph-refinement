"""Collapsed-Gibbs biterm topic model baseline for short feedback."""

from __future__ import annotations

import numpy as np

from src.v6.comparators.base import BaselineAdapter, BaselineData, BaselineFit, normalize_rows


def _biterms(document: tuple[int, ...], window_size: int) -> list[tuple[int, int]]:
    values = tuple(map(int, document))
    pairs: list[tuple[int, int]] = []
    for left in range(len(values) - 1):
        upper = min(len(values), left + window_size)
        pairs.extend(
            (min(values[left], values[right]), max(values[left], values[right]))
            for right in range(left + 1, upper)
        )
    return pairs


def _biterm_probabilities(
    pair: tuple[int, int],
    topic_biterms: np.ndarray,
    topic_words: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    first, second = pair
    vocabulary = topic_words.shape[1]
    denominator = 2.0 * topic_biterms + vocabulary * beta
    probabilities = (topic_biterms + alpha) * (
        (topic_words[:, first] + beta) / denominator
    ) * (
        (topic_words[:, second] + beta + float(first == second))
        / (denominator + 1.0)
    )
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("BTM produced an invalid conditional distribution")
    return probabilities / total


def _infer_document_theta(
    document: tuple[int, ...],
    topic_biterms: np.ndarray,
    topic_words: np.ndarray,
    *,
    alpha: float,
    beta: float,
    window_size: int,
) -> np.ndarray:
    pairs = _biterms(document, window_size)
    if pairs:
        values = np.mean(
            [
                _biterm_probabilities(
                    pair, topic_biterms, topic_words, alpha=alpha, beta=beta
                )
                for pair in pairs
            ],
            axis=0,
        )
    else:
        prior = topic_biterms + alpha
        if document:
            phi = normalize_rows(topic_words + beta)
            values = prior * phi[:, int(document[0])]
        else:
            values = prior
    return values / values.sum()


class BTMAdapter(BaselineAdapter):
    method_id = "btm"
    display_name = "BTM"

    def fit(self, data: BaselineData, *, topics: int, seed: int) -> BaselineFit:
        alpha = float(self.config.get("alpha", 50.0 / topics))
        beta = float(self.config.get("beta", 0.01))
        window = int(self.config.get("window_size", 15))
        maximum = int(self.config["maximum_iterations"])
        if alpha <= 0.0 or beta <= 0.0 or window < 2 or maximum < 1:
            raise ValueError("BTM priors, window, and iteration budget must be valid")
        corpus_biterms = [
            pair
            for document in data.train_token_documents
            for pair in _biterms(document, window)
        ]
        if not corpus_biterms:
            raise ValueError("BTM requires at least one training biterm")
        rng = np.random.default_rng(seed)
        assignments = rng.integers(0, topics, size=len(corpus_biterms), dtype=np.int64)
        topic_biterms = np.zeros(topics, dtype=np.int64)
        topic_words = np.zeros((topics, data.vocabulary_size), dtype=np.int64)
        for assignment, (first, second) in zip(assignments, corpus_biterms):
            topic_biterms[assignment] += 1
            topic_words[assignment, first] += 1
            topic_words[assignment, second] += 1

        moved_history: list[int] = []
        stable_required = int(self.config.get("stable_iterations", 3))
        stable = 0
        for _iteration in range(1, maximum + 1):
            moved = 0
            for index, pair in enumerate(corpus_biterms):
                old = int(assignments[index])
                first, second = pair
                topic_biterms[old] -= 1
                topic_words[old, first] -= 1
                topic_words[old, second] -= 1
                probabilities = _biterm_probabilities(
                    pair, topic_biterms, topic_words, alpha=alpha, beta=beta
                )
                new = int(rng.choice(topics, p=probabilities))
                assignments[index] = new
                topic_biterms[new] += 1
                topic_words[new, first] += 1
                topic_words[new, second] += 1
                moved += int(new != old)
            moved_history.append(moved)
            stable = stable + 1 if moved == 0 else 0
            if stable >= stable_required:
                break

        topic_word = normalize_rows(topic_words + beta)
        validation_theta = np.vstack(
            [
                _infer_document_theta(
                    document,
                    topic_biterms,
                    topic_words,
                    alpha=alpha,
                    beta=beta,
                    window_size=window,
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
            False,
            topics,
            iterations_completed=len(moved_history),
            converged=stable >= stable_required,
            metadata={
                "implementation": "project_collapsed_gibbs_biterm_topic_model",
                "training_biterms": len(corpus_biterms),
                "window_size": window,
                "alpha": alpha,
                "beta": beta,
                "heldout_nll_status": "not_comparable_native_biterm_likelihood",
                "biterms_moved_final_iteration": int(moved_history[-1]),
            },
        )
