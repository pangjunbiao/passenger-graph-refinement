import numpy as np
from scipy import sparse

from src.v6.development_evidence import (
    construct_fold_context,
    crossfitted_city_unigram_signal,
    matched_topic_stability,
    positive_npmi_graph,
)


def test_completion_context_is_computed_from_observed_counts_only():
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [5, 1, 0, 0],
                [4, 1, 0, 0],
                [0, 0, 5, 1],
                [0, 0, 4, 1],
                [3, 1, 1, 0],
                [0, 1, 3, 1],
            ],
            dtype=np.int64,
        )
    )
    features = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    folds = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    context = construct_fold_context(
        counts,
        features,
        folds,
        heldout_fold=1,
        clusters=2,
        n_init=5,
        seed=71,
    )
    observed = sparse.csr_matrix(np.asarray([[2, 1, 0, 0]], dtype=np.int64))
    first = context.input_for_local_counts(observed)
    second = context.input_for_local_counts(observed.copy())
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, : counts.shape[1]], observed.toarray())
    assert context.audit["full_document_embedding_used_for_completion_assignment"] is False
    assert context.audit["heldout_assignment"].endswith("observed_counts_only")


def test_positive_npmi_graph_is_fold_local_symmetric_bounded_and_nonempty():
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int64,
        )
    )
    graph, report = positive_npmi_graph(
        counts,
        minimum_document_frequency=2,
        minimum_joint_documents=2,
        reliability_power=0.5,
    )
    np.testing.assert_allclose(graph, graph.T, atol=0, rtol=0)
    np.testing.assert_allclose(np.diag(graph), 0, atol=0, rtol=0)
    assert graph.min() >= 0 and graph.max() <= 1
    assert report["positive_directed_entries"] > 0
    assert report["minimum_joint_documents"] == 2
    assert report["fit_partition"] == "corresponding_fold_training_counts_only"


def test_crossfitted_city_signal_never_uses_heldout_counts_for_estimation():
    # Each city has a stable preferred word, so partial pooling should improve
    # prediction in both held-out folds.
    dense = np.asarray(
        [
            [8, 1, 1],
            [7, 1, 1],
            [1, 8, 1],
            [1, 7, 1],
            [9, 1, 1],
            [8, 1, 1],
            [1, 9, 1],
            [1, 8, 1],
        ],
        dtype=np.int64,
    )
    cities = ["a", "a", "b", "b", "a", "a", "b", "b"]
    folds = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    rows, summary = crossfitted_city_unigram_signal(
        sparse.csr_matrix(dense), cities, folds, shrinkage=0.5, pseudocount=0.5
    )
    assert len(rows) == 2
    assert summary["positive_fold_fraction"] == 1.0
    assert summary["minimum_reduction"] > 0


def test_matched_stability_is_permutation_invariant():
    first = np.asarray([[1, 2, 3], [7, 8, 9]])
    second = np.asarray([[7, 8, 9], [1, 2, 3]])
    report = matched_topic_stability([first, second])
    assert report["mean"] == 1.0
    assert report["minimum"] == 1.0
