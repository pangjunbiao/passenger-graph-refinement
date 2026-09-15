import numpy as np

from src.v6 import reference


def test_numpy_soft_topk_has_requested_mass_and_order():
    scores = np.array([[2.0, 1.5, 0.7, 0.2, -0.4]], dtype=np.float64)
    membership = reference.soft_topk_membership(scores, top_k=3, temperature=0.4)
    assert np.max(np.abs(membership.sum(axis=1) - 3.0)) < 1.0e-10
    assert np.all(np.diff(membership[0]) < 0.0)


def test_numpy_contrast_basis_is_weight_centered():
    weights = np.array([0.55, 0.30, 0.15], dtype=np.float64)
    basis = reference.weighted_contrast_basis(weights)
    assert basis.shape == (3, 2)
    assert np.max(np.abs(weights @ basis)) < 1.0e-12

