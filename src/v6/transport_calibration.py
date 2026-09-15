"""Training-only decoder calibration for graph-transported affinities.

The official EnCOT decoder contains a vocabulary-wise BatchNorm layer whose
running moments are learned with the unmodified topic--word affinity.  A hard
graph transport changes a small set of affinity columns after parent fitting,
so those frozen moments no longer describe the logits entering the decoder.

This module fits a deterministic affine map on *training logits only*.  For
every affinity column changed by graph transport it maps the transported-logit
mean and variance back to the corresponding native-parent moments.  Unchanged
columns are exact identities.  No target, validation, test, label, baseline,
or SOTA value is accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrozenMomentCalibration:
    scale: np.ndarray
    shift: np.ndarray
    changed_mask: np.ndarray
    audit: dict[str, float | int | str]


def fit_changed_column_moment_calibration(
    native_training_logits: np.ndarray,
    transported_training_logits: np.ndarray,
    native_affinity: np.ndarray,
    transported_affinity: np.ndarray,
    *,
    variance_floor: float,
    change_tolerance: float = 0.0,
) -> FrozenMomentCalibration:
    """Fit a vocabulary-wise moment map using training rows only.

    If ``z`` is a transported logit, the calibrated value is ``a*z+b``.
    For changed columns, ``a`` and ``b`` align the regularized population
    moments with the native-parent logits.  For every unchanged affinity
    column, ``a=1`` and ``b=0`` exactly.
    """

    native = np.asarray(native_training_logits, dtype=np.float64)
    transported = np.asarray(transported_training_logits, dtype=np.float64)
    base = np.asarray(native_affinity, dtype=np.float64)
    projected = np.asarray(transported_affinity, dtype=np.float64)
    floor = float(variance_floor)
    tolerance = float(change_tolerance)
    if native.ndim != 2 or min(native.shape) < 1:
        raise ValueError("native training logits must be a nonempty matrix")
    if transported.shape != native.shape:
        raise ValueError("native and transported training logits must align")
    if base.ndim != 2 or projected.shape != base.shape:
        raise ValueError("native and transported affinities must align")
    if base.shape[1] != native.shape[1]:
        raise ValueError("affinity vocabulary and logit vocabulary differ")
    if not all(np.isfinite(value).all() for value in (native, transported, base, projected)):
        raise ValueError("calibration inputs must be finite")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("change_tolerance must be finite and nonnegative")

    changed = np.max(np.abs(projected - base), axis=0) > tolerance
    scale = np.ones(native.shape[1], dtype=np.float64)
    shift = np.zeros(native.shape[1], dtype=np.float64)
    native_mean = native.mean(axis=0)
    projected_mean = transported.mean(axis=0)
    native_variance = native.var(axis=0)
    projected_variance = transported.var(axis=0)
    if bool(changed.any()):
        scale[changed] = np.sqrt(
            (native_variance[changed] + floor)
            / (projected_variance[changed] + floor)
        )
        shift[changed] = (
            native_mean[changed]
            - scale[changed] * projected_mean[changed]
        )
    calibrated = transported * scale[None, :] + shift[None, :]
    if not np.isfinite(calibrated).all() or np.any(scale <= 0.0):
        raise FloatingPointError("moment calibration produced invalid values")

    if bool(changed.any()):
        calibrated_mean_error = float(
            np.max(np.abs(calibrated.mean(axis=0)[changed] - native_mean[changed]))
        )
        expected_regularized_variance = (
            scale[changed] ** 2
            * (projected_variance[changed] + floor)
        )
        native_regularized_variance = native_variance[changed] + floor
        calibrated_variance_error = float(
            np.max(
                np.abs(
                    expected_regularized_variance
                    - native_regularized_variance
                )
            )
        )
        minimum_scale = float(scale[changed].min())
        maximum_scale = float(scale[changed].max())
        maximum_shift = float(np.max(np.abs(shift[changed])))
    else:
        calibrated_mean_error = 0.0
        calibrated_variance_error = 0.0
        minimum_scale = 1.0
        maximum_scale = 1.0
        maximum_shift = 0.0
    unchanged_identity_error = float(
        max(
            np.max(np.abs(scale[~changed] - 1.0)) if bool((~changed).any()) else 0.0,
            np.max(np.abs(shift[~changed])) if bool((~changed).any()) else 0.0,
        )
    )
    return FrozenMomentCalibration(
        scale=scale,
        shift=shift,
        changed_mask=changed,
        audit={
            "schema_version": 1,
            "fit_partition": "training_rows_only",
            "construction": "changed-column regularized moment matching",
            "training_documents": int(native.shape[0]),
            "vocabulary": int(native.shape[1]),
            "changed_columns": int(changed.sum()),
            "variance_floor": floor,
            "change_tolerance": tolerance,
            "minimum_changed_scale": minimum_scale,
            "maximum_changed_scale": maximum_scale,
            "maximum_absolute_changed_shift": maximum_shift,
            "maximum_changed_mean_error": calibrated_mean_error,
            "maximum_changed_regularized_variance_error": calibrated_variance_error,
            "maximum_unchanged_identity_error": unchanged_identity_error,
        },
    )


def apply_frozen_moment_calibration(
    logits: np.ndarray, calibration: FrozenMomentCalibration
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != calibration.scale.size:
        raise ValueError("logits do not match the frozen calibration")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    result = values * calibration.scale[None, :] + calibration.shift[None, :]
    if not np.isfinite(result).all():
        raise FloatingPointError("calibrated logits are non-finite")
    return result
