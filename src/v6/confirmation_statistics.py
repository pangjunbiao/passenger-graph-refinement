"""Deterministic paired inference for the V6.1 confirmation stage.

The functions in this module are independent of model and data loading.  Every
reported ablation difference is oriented so that a positive value favors the
frozen complete model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import product
from typing import Any

import numpy as np
from scipy.stats import rankdata

REPORTED_METRICS = (
    "npmi_at_10",
    "c_v_at_10",
    "topic_diversity_at_10",
    "top_word_redundancy_at_10",
    "macro_city_nll_per_token",
    "micro_nll_per_token",
    "micro_perplexity",
)


def build_paired_effect_rows(
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    hypotheses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return every predeclared same-seed contrast on its favorable scale."""

    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in per_seed_rows:
        key = (str(row["variant"]), int(row["seed"]))
        if key in lookup:
            raise ValueError("a confirmation variant/seed pair is duplicated")
        lookup[key] = row
    effects: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        full = str(hypothesis["full_variant"])
        control = str(hypothesis["control_variant"])
        endpoint = str(hypothesis["endpoint"])
        direction = str(hypothesis["direction"])
        for seed in map(int, seeds):
            try:
                full_value = float(lookup[(full, seed)][endpoint])
                control_value = float(lookup[(control, seed)][endpoint])
            except KeyError as exc:
                raise ValueError("the paired confirmation grid is incomplete") from exc
            if direction == "full_higher":
                effect = full_value - control_value
            elif direction == "full_lower":
                effect = control_value - full_value
            else:
                raise ValueError(f"unknown confirmation direction: {direction}")
            effects.append(
                {
                    "contrast_id": str(hypothesis["contrast_id"]),
                    "family": str(hypothesis["family"]),
                    "seed": seed,
                    "full_variant": full,
                    "control_variant": control,
                    "endpoint": endpoint,
                    "direction": direction,
                    "full_value": full_value,
                    "control_value": control_value,
                    "paired_effect_positive_favors_full": float(effect),
                }
            )
    return effects


def _derived_seed(master_seed: int, label: str) -> int:
    payload = f"SC-HTM-V6.1|bootstrap|{int(master_seed)}|{label}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "little", signed=False)


def summarize_values(
    values: Sequence[float],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Return mean, sample SD, median, and deterministic bootstrap mean CI."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError(
            "summary values must be a finite vector of length at least two"
        )
    resamples = int(bootstrap_resamples)
    if resamples < 1000:
        raise ValueError("at least 1,000 bootstrap resamples are required")
    generator = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    draws = generator.integers(0, array.size, size=(resamples, array.size))
    means = array[draws].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
    }


def exact_signed_rank(differences: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided signed-rank randomization, including tied magnitudes.

    Zero differences are removed.  For the remaining observations, all sign
    assignments are enumerated conditional on the observed absolute ranks.
    With ten reporting seeds this requires at most 1,024 assignments.
    """

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("signed-rank differences must be a nonempty finite vector")
    nonzero = values != 0.0
    retained = values[nonzero]
    zeros = int((~nonzero).sum())
    if retained.size == 0:
        return {
            "defined": False,
            "p_value": None,
            "statistic": 0.0,
            "positive_rank_sum": 0.0,
            "negative_rank_sum": 0.0,
            "nonzero_pairs": 0,
            "zero_pairs": zeros,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "enumerated_assignments": 0,
        }
    if retained.size > 20:
        raise ValueError("exact signed-rank enumeration is limited to 20 pairs")
    ranks = rankdata(np.abs(retained), method="average").astype(np.float64)
    observed_positive = float(ranks[retained > 0.0].sum())
    total = float(ranks.sum())
    center = total / 2.0
    observed_distance = abs(observed_positive - center)
    exceedances = 0
    assignments = 2 ** int(retained.size)
    for signs in product((0.0, 1.0), repeat=int(retained.size)):
        positive_sum = float(np.dot(ranks, np.asarray(signs, dtype=np.float64)))
        if abs(positive_sum - center) >= observed_distance - 1.0e-12:
            exceedances += 1
    negative_sum = total - observed_positive
    return {
        "defined": True,
        "p_value": float(exceedances / assignments),
        "statistic": float(min(observed_positive, negative_sum)),
        "positive_rank_sum": observed_positive,
        "negative_rank_sum": negative_sum,
        "nonzero_pairs": int(retained.size),
        "zero_pairs": zeros,
        "positive_pairs": int((retained > 0.0).sum()),
        "negative_pairs": int((retained < 0.0).sum()),
        "enumerated_assignments": assignments,
    }


def apply_holm(rows: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    """Attach Holm-adjusted p-values without changing input row order."""

    level = float(alpha)
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    result = [dict(row) for row in rows]
    defined = [
        (index, float(row["raw_p_value"]))
        for index, row in enumerate(result)
        if row.get("raw_p_value") is not None
    ]
    defined.sort(key=lambda item: (item[1], item[0]))
    family_size = len(defined)
    running = 0.0
    for rank, (index, value) in enumerate(defined, start=1):
        running = max(running, (family_size - rank + 1) * value)
        adjusted = min(1.0, running)
        result[index]["holm_adjusted_p_value"] = float(adjusted)
        result[index]["holm_reject"] = bool(adjusted <= level)
        result[index]["holm_rank"] = int(rank)
        result[index]["holm_family_size"] = int(family_size)
    for row in result:
        if row.get("raw_p_value") is None:
            row["holm_adjusted_p_value"] = None
            row["holm_reject"] = False
            row["holm_rank"] = None
            row["holm_family_size"] = int(family_size)
    return result


def build_confirmation_statistics(
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    hypotheses: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build variant summaries and paired hypothesis tests."""

    records = [dict(row) for row in per_seed_rows]
    variant_order = list(map(str, variants))
    seed_order = list(map(int, seeds))
    if len(records) != len(variant_order) * len(seed_order):
        raise ValueError("the confirmation result grid is incomplete")
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in records:
        key = (str(row["variant"]), int(row["seed"]))
        if key in lookup:
            raise ValueError("a confirmation variant/seed pair is duplicated")
        lookup[key] = row
    expected = {(variant, seed) for variant in variant_order for seed in seed_order}
    if set(lookup) != expected:
        raise ValueError("the confirmation variant/seed identities changed")

    summaries: list[dict[str, Any]] = []
    for variant in variant_order:
        for metric in REPORTED_METRICS:
            values = [float(lookup[(variant, seed)][metric]) for seed in seed_order]
            label = f"variant|{variant}|{metric}"
            summary = summarize_values(
                values,
                bootstrap_resamples=int(bootstrap_resamples),
                bootstrap_seed=_derived_seed(bootstrap_seed, label),
            )
            summaries.append({"variant": variant, "metric": metric, **summary})

    tests: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        full = str(hypothesis["full_variant"])
        control = str(hypothesis["control_variant"])
        endpoint = str(hypothesis["endpoint"])
        direction = str(hypothesis["direction"])
        if endpoint not in REPORTED_METRICS:
            raise ValueError(f"unregistered confirmation endpoint: {endpoint}")
        full_values = np.asarray(
            [float(lookup[(full, seed)][endpoint]) for seed in seed_order],
            dtype=np.float64,
        )
        control_values = np.asarray(
            [float(lookup[(control, seed)][endpoint]) for seed in seed_order],
            dtype=np.float64,
        )
        if direction == "full_higher":
            difference = full_values - control_values
        elif direction == "full_lower":
            difference = control_values - full_values
        else:
            raise ValueError(f"unknown confirmation direction: {direction}")
        label = f"effect|{hypothesis['contrast_id']}|{endpoint}"
        effect = summarize_values(
            difference,
            bootstrap_resamples=int(bootstrap_resamples),
            bootstrap_seed=_derived_seed(bootstrap_seed, label),
        )
        signed_rank = exact_signed_rank(difference)
        tests.append(
            {
                "contrast_id": str(hypothesis["contrast_id"]),
                "family": str(hypothesis["family"]),
                "full_variant": full,
                "control_variant": control,
                "endpoint": endpoint,
                "direction": direction,
                "difference_definition": "positive_favors_frozen_complete_model",
                "mean_paired_effect": effect["mean"],
                "sample_sd_paired_effect": effect["sample_sd"],
                "median_paired_effect": effect["median"],
                "effect_ci95_low": effect["ci95_low"],
                "effect_ci95_high": effect["ci95_high"],
                "direction_met": bool(float(effect["mean"]) > 0.0),
                "raw_p_value": signed_rank["p_value"],
                "signed_rank_defined": signed_rank["defined"],
                "signed_rank_statistic": signed_rank["statistic"],
                "nonzero_pairs": signed_rank["nonzero_pairs"],
                "zero_pairs": signed_rank["zero_pairs"],
                "positive_pairs": signed_rank["positive_pairs"],
                "negative_pairs": signed_rank["negative_pairs"],
                "enumerated_assignments": signed_rank["enumerated_assignments"],
            }
        )
    families = sorted({str(row["family"]) for row in tests})
    adjusted: list[dict[str, Any]] = []
    for family in families:
        family_indices = [i for i, row in enumerate(tests) if row["family"] == family]
        family_rows = apply_holm([tests[i] for i in family_indices], alpha=alpha)
        adjusted.extend(family_rows)
    order = {
        (str(item["contrast_id"]), str(item["endpoint"])): index
        for index, item in enumerate(hypotheses)
    }
    adjusted.sort(key=lambda row: order[(row["contrast_id"], row["endpoint"])])
    for row in adjusted:
        row["claim_supported"] = bool(row["direction_met"] and row["holm_reject"])
    outcome = {
        "schema_version": 1,
        "primary_endpoint_tests": len(adjusted),
        "direction_met_count": sum(bool(row["direction_met"]) for row in adjusted),
        "holm_reject_count": sum(bool(row["holm_reject"]) for row in adjusted),
        "supported_endpoint_count": sum(
            bool(row["claim_supported"]) for row in adjusted
        ),
        "all_prespecified_primary_endpoints_supported": bool(
            adjusted and all(bool(row["claim_supported"]) for row in adjusted)
        ),
        "model_or_hyperparameter_change_allowed": False,
    }
    return summaries, adjusted, outcome
