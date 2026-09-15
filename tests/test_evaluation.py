import numpy as np
from scipy import sparse

from src.v6.evaluation import (
    counts_to_token_documents,
    cv_coherence,
    document_completion_nll,
    document_npmi,
    stable_top_word_indices,
)


def test_independent_coherence_fixture_and_missing_policy_are_finite():
    counts = sparse.csr_matrix(
        np.asarray(
            [[2, 2, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 2, 2, 0]],
            dtype=np.int64,
        )
    )
    topics = np.asarray([[0, 1], [2, 4]], dtype=np.int64)
    npmi = document_npmi(counts, topics)
    cv = cv_coherence(
        counts_to_token_documents(counts),
        topics,
        vocabulary_size=5,
        window_size=110,
        gamma=1.0,
    )
    assert abs(npmi.per_topic[0] - 1.0) < 1.0e-12
    assert npmi.absent_word_pairs[1] == 1
    assert cv.absent_topic_words[1] == 1
    assert np.isfinite(cv.mean)


def test_top_words_have_deterministic_ties():
    topics = np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64)
    assert stable_top_word_indices(topics, 3).tolist() == [[0, 1, 2]]


def test_completion_scores_only_supplied_target_and_observed_theta():
    target = sparse.csr_matrix(np.asarray([[1, 0], [0, 1]], dtype=np.int64))
    theta = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    beta = np.asarray([[0.75, 0.25], [0.25, 0.75]])
    result = document_completion_nll(target, theta, beta)
    assert abs(result.nll_per_token - -np.log(0.75)) < 1.0e-12
    assert result.tokens == 2
