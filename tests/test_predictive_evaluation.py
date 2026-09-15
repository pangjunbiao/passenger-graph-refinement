import numpy as np
import pytest
from scipy import sparse

from src.v6.predictive_evaluation import (
    document_completion_nll_from_probabilities,
    hand_calculated_self_test,
)


def test_native_decoder_completion_matches_hand_calculation():
    assert hand_calculated_self_test()["passed"] is True


def test_native_decoder_completion_uses_only_supplied_probabilities():
    target = sparse.csr_matrix(np.asarray([[2, 0], [0, 1]], dtype=np.int64))
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    result = document_completion_nll_from_probabilities(target, probabilities)
    expected = -(2.0 * np.log(0.8) + np.log(0.7)) / 3.0
    assert result.nll_per_token == pytest.approx(expected, abs=1.0e-12)
    assert result.tokens == 3


def test_native_decoder_completion_rejects_non_simplex_rows():
    target = sparse.csr_matrix(np.asarray([[1, 0]], dtype=np.int64))
    with pytest.raises(ValueError, match="sum to one"):
        document_completion_nll_from_probabilities(
            target, np.asarray([[0.8, 0.8]], dtype=np.float64)
        )

