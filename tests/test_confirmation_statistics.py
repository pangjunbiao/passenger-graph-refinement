import numpy as np

from src.v6.confirmation_statistics import (
    apply_holm,
    build_confirmation_statistics,
    build_paired_effect_rows,
    exact_signed_rank,
    summarize_values,
)


def test_exact_signed_rank_all_ten_positive_has_exact_two_sided_probability():
    result = exact_signed_rank(np.arange(1.0, 11.0))
    assert result["defined"] is True
    assert result["positive_pairs"] == 10
    assert result["negative_pairs"] == 0
    assert result["enumerated_assignments"] == 1024
    assert result["p_value"] == 2.0 / 1024.0


def test_exact_signed_rank_handles_zeros_and_tied_magnitudes():
    result = exact_signed_rank([0.0, 1.0, 1.0, -1.0, -1.0])
    assert result["defined"] is True
    assert result["zero_pairs"] == 1
    assert result["nonzero_pairs"] == 4
    assert result["p_value"] == 1.0


def test_exact_signed_rank_is_explicitly_undefined_for_all_zero_effects():
    result = exact_signed_rank([0.0] * 10)
    assert result["defined"] is False
    assert result["p_value"] is None


def test_holm_adjustment_is_monotone_in_sorted_p_value_order():
    rows = [
        {"name": "a", "raw_p_value": 0.01},
        {"name": "b", "raw_p_value": 0.04},
        {"name": "c", "raw_p_value": 0.03},
    ]
    adjusted = apply_holm(rows, alpha=0.05)
    assert [row["holm_adjusted_p_value"] for row in adjusted] == [0.03, 0.06, 0.06]
    assert [row["holm_reject"] for row in adjusted] == [True, False, False]


def test_bootstrap_summary_is_reproducible_and_uses_sample_sd():
    first = summarize_values(
        [1.0, 2.0, 3.0], bootstrap_resamples=2000, bootstrap_seed=17
    )
    second = summarize_values(
        [1.0, 2.0, 3.0], bootstrap_resamples=2000, bootstrap_seed=17
    )
    assert first == second
    assert first["mean"] == 2.0
    assert first["sample_sd"] == 1.0


def test_confirmation_grid_orients_every_effect_to_favor_full_model():
    variants = ["full", "control"]
    seeds = list(range(10))
    rows = []
    for variant in variants:
        for seed in seeds:
            base = 2.0 if variant == "full" else 1.0
            nll = 4.0 if variant == "full" else 5.0
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "npmi_at_10": base,
                    "c_v_at_10": base,
                    "topic_diversity_at_10": 1.0,
                    "top_word_redundancy_at_10": 0.0,
                    "macro_city_nll_per_token": nll,
                    "micro_nll_per_token": nll,
                    "micro_perplexity": float(np.exp(nll)),
                }
            )
    hypotheses = [
        {
            "contrast_id": "quality",
            "family": "ablation",
            "full_variant": "full",
            "control_variant": "control",
            "endpoint": "npmi_at_10",
            "direction": "full_higher",
        },
        {
            "contrast_id": "prediction",
            "family": "ablation",
            "full_variant": "full",
            "control_variant": "control",
            "endpoint": "macro_city_nll_per_token",
            "direction": "full_lower",
        },
    ]
    summaries, tests, outcome = build_confirmation_statistics(
        rows,
        variants=variants,
        seeds=seeds,
        hypotheses=hypotheses,
        bootstrap_resamples=2000,
        bootstrap_seed=19,
        alpha=0.05,
    )
    assert len(summaries) == 14
    assert all(row["mean_paired_effect"] == 1.0 for row in tests)
    assert all(row["claim_supported"] for row in tests)
    assert outcome["all_prespecified_primary_endpoints_supported"] is True


def test_paired_effect_rows_retain_each_seed_and_direction():
    rows = [
        {"variant": "full", "seed": 1, "score": 3.0, "loss": 1.0},
        {"variant": "control", "seed": 1, "score": 2.0, "loss": 2.0},
        {"variant": "full", "seed": 2, "score": 5.0, "loss": 2.0},
        {"variant": "control", "seed": 2, "score": 1.0, "loss": 4.0},
    ]
    hypotheses = [
        {
            "contrast_id": "quality",
            "family": "primary",
            "full_variant": "full",
            "control_variant": "control",
            "endpoint": "score",
            "direction": "full_higher",
        },
        {
            "contrast_id": "prediction",
            "family": "primary",
            "full_variant": "full",
            "control_variant": "control",
            "endpoint": "loss",
            "direction": "full_lower",
        },
    ]
    effects = build_paired_effect_rows(rows, seeds=[1, 2], hypotheses=hypotheses)
    assert [row["paired_effect_positive_favors_full"] for row in effects] == [
        1.0,
        4.0,
        1.0,
        2.0,
    ]
