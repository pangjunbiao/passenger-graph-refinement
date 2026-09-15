import numpy as np
import pytest
from scipy import sparse

from src.v6.validation_context import construct_training_only_context


def _fixture():
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [5, 2, 0, 0, 0, 0],
                [4, 3, 0, 0, 0, 0],
                [3, 4, 0, 0, 0, 0],
                [4, 2, 1, 0, 0, 0],
                [0, 0, 0, 5, 2, 0],
                [0, 0, 0, 4, 3, 0],
                [0, 0, 0, 3, 4, 0],
                [0, 0, 0, 4, 2, 1],
            ],
            dtype=np.int64,
        )
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return counts, embeddings


def test_training_only_context_is_deterministic_and_train_fitted():
    counts, embeddings = _fixture()
    first = construct_training_only_context(
        counts, embeddings, clusters=2, n_init=5, seed=41
    )
    second = construct_training_only_context(
        counts, embeddings, clusters=2, n_init=5, seed=41
    )
    np.testing.assert_array_equal(first.train_input, second.train_input)
    assert first.audit == second.audit
    assert first.audit["documents_fit"] == 8
    assert first.audit["validation_documents_fit"] == 0
    assert first.train_input.shape == (8, 12)


def test_validation_context_input_depends_on_observed_half_only():
    counts, embeddings = _fixture()
    context = construct_training_only_context(
        counts, embeddings, clusters=2, n_init=5, seed=41
    )
    observed = sparse.csr_matrix([[2, 1, 0, 0, 0, 0], [0, 0, 0, 2, 1, 0]])
    first, receipt = context.input_from_observed_counts(observed)
    second, repeated = context.input_from_observed_counts(observed.copy())
    np.testing.assert_array_equal(first, second)
    assert receipt == repeated
    assert receipt["target_half_used_for_assignment"] is False
    assert receipt["full_document_embedding_used"] is False
    assert first.shape == (2, 12)


def test_validation_context_rejects_empty_observed_document():
    counts, embeddings = _fixture()
    context = construct_training_only_context(
        counts, embeddings, clusters=2, n_init=5, seed=41
    )
    with pytest.raises(ValueError, match="empty observed"):
        context.input_from_observed_counts(sparse.csr_matrix((1, 6)))


def test_final_refit_context_discloses_spent_development_rows():
    counts, embeddings = _fixture()
    context = construct_training_only_context(
        counts,
        embeddings,
        clusters=2,
        n_init=5,
        seed=41,
        fit_partition_label="all_8_train_plus_spent_development_rows",
        original_training_documents=6,
        spent_development_documents=2,
    )
    assert context.audit["documents_fit"] == 8
    assert context.audit["original_training_documents_fit"] == 6
    assert context.audit["spent_development_documents_fit"] == 2
    assert context.audit["validation_documents_fit"] == 2


def test_final_refit_context_partition_counts_must_sum_to_rows():
    counts, embeddings = _fixture()
    with pytest.raises(ValueError, match="do not match"):
        construct_training_only_context(
            counts,
            embeddings,
            clusters=2,
            n_init=5,
            seed=41,
            original_training_documents=6,
            spent_development_documents=1,
        )
