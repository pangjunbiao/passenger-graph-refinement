import numpy as np
import pytest

from src.v6.transport_calibration import (
    apply_frozen_moment_calibration,
    fit_changed_column_moment_calibration,
)


def _fixture():
    generator = np.random.default_rng(6102)
    native = generator.normal(size=(41, 7))
    transported = native.copy()
    transported[:, 1] = 1.8 * native[:, 1] + 0.7
    transported[:, 5] = 0.6 * native[:, 5] - 0.4
    affinity = generator.uniform(size=(3, 7))
    affinity /= affinity.sum(axis=0, keepdims=True)
    projected = affinity.copy()
    projected[:, 1] = np.roll(projected[:, 1], 1)
    projected[:, 5] = np.roll(projected[:, 5], 2)
    return native, transported, affinity, projected


def test_moment_calibration_is_exact_identity_when_transport_is_zero():
    native, _, affinity, _ = _fixture()
    calibration = fit_changed_column_moment_calibration(
        native,
        native,
        affinity,
        affinity,
        variance_floor=1.0e-10,
    )
    assert not calibration.changed_mask.any()
    np.testing.assert_array_equal(calibration.scale, np.ones(7))
    np.testing.assert_array_equal(calibration.shift, np.zeros(7))
    np.testing.assert_array_equal(
        apply_frozen_moment_calibration(native, calibration), native
    )


def test_changed_columns_reproduce_training_means():
    native, transported, affinity, projected = _fixture()
    calibration = fit_changed_column_moment_calibration(
        native,
        transported,
        affinity,
        projected,
        variance_floor=1.0e-10,
    )
    result = apply_frozen_moment_calibration(transported, calibration)
    changed = calibration.changed_mask
    np.testing.assert_allclose(
        result[:, changed].mean(axis=0),
        native[:, changed].mean(axis=0),
        atol=2.0e-15,
        rtol=0.0,
    )


def test_unchanged_columns_are_bitwise_identity():
    native, transported, affinity, projected = _fixture()
    calibration = fit_changed_column_moment_calibration(
        native,
        transported,
        affinity,
        projected,
        variance_floor=1.0e-10,
    )
    result = apply_frozen_moment_calibration(transported, calibration)
    unchanged = ~calibration.changed_mask
    np.testing.assert_array_equal(result[:, unchanged], transported[:, unchanged])
    assert calibration.audit["maximum_unchanged_identity_error"] == 0.0


def test_frozen_map_applies_to_new_rows_without_refitting():
    native, transported, affinity, projected = _fixture()
    calibration = fit_changed_column_moment_calibration(
        native,
        transported,
        affinity,
        projected,
        variance_floor=1.0e-10,
    )
    new_rows = np.arange(14, dtype=np.float64).reshape(2, 7)
    expected = new_rows * calibration.scale + calibration.shift
    np.testing.assert_array_equal(
        apply_frozen_moment_calibration(new_rows, calibration), expected
    )


def test_calibration_rejects_nonfinite_training_values():
    native, transported, affinity, projected = _fixture()
    transported[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_changed_column_moment_calibration(
            native,
            transported,
            affinity,
            projected,
            variance_floor=1.0e-10,
        )
