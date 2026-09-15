from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.v6.step14 import (
    EXPECTED_CALIBRATIONS,
    EXPECTED_FOLDS,
    EXPECTED_QUOTAS,
    EXPECTED_RHOS,
    EXPECTED_SEEDS,
    IMPLEMENTATION,
    _aggregate,
    _aggregate_rho,
    _collapse_q,
    _load_protocol,
    _paper_table,
    _sensitivity_svg,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/step14.yaml"


def _q_rows():
    rows = []
    for quota in EXPECTED_QUOTAS:
        for calibration in EXPECTED_CALIBRATIONS:
            for index, seed in enumerate(EXPECTED_SEEDS):
                base = quota / 100 + index / 1000
                calibrated_gain = (
                    0.0 if calibration == "native_decoder" else 0.05 + quota / 1000
                )
                rows.append(
                    {
                        "candidate_id": f"quota_{quota:02d}__{calibration}",
                        "seed": seed,
                        "prototype_quota": quota,
                        "calibration": calibration,
                        "validation_npmi_full": base,
                        "validation_cv_full": 0.4 + base,
                        "full_macro_city_nll": 7.0 - base - calibrated_gain,
                        "validation_topic_diversity_full": 0.9,
                        "validation_top_word_redundancy_full": 0.02,
                        "validation_npmi_improvement_vs_without_graph": 0.1,
                        "validation_cv_improvement_vs_without_graph": 0.03,
                        "pooled_nll_reduction_vs_without_backoff": 0.08,
                        "calibration_nll_reduction_vs_native": calibrated_gain,
                        "full_nll_reduction_vs_official_parent": 0.10,
                    }
                )
    return rows


def _rho_rows():
    rows = []
    for fold in EXPECTED_FOLDS:
        parent = 7.0 + fold / 100
        for rho in EXPECTED_RHOS:
            reduction = rho * (0.30 + fold / 100)
            rows.append(
                {
                    "fold": fold,
                    "mixture_weight": rho,
                    "parent_macro_city_nll": parent,
                    "pooled_backoff_macro_city_nll": parent - reduction,
                    "pooled_minus_parent_nll_reduction": reduction,
                }
            )
    return rows


def test_step14_protocol_is_reporting_only_and_fail_closed():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["prerequisites"]["step04r2"]["candidate_rhos"] == EXPECTED_RHOS
    assert protocol["prerequisites"]["step06r2"]["candidate_rows"] == 18
    assert protocol["reporting"]["parameters"] == [
        "graph_core_coverage_q",
        "pooled_backoff_weight_rho",
        "decoder_calibration",
    ]
    assert protocol["policy"]["model_fit_allowed"] is False
    assert protocol["policy"]["new_evaluation_allowed"] is False
    assert protocol["policy"]["validation_reopen_allowed"] is False
    assert protocol["policy"]["test_access_allowed"] is False
    assert protocol["policy"]["hyperparameter_selection_allowed"] is False


def test_step14_aggregation_retains_q_calibration_and_rho_surfaces():
    cells = _aggregate(_q_rows())
    q_rows = _collapse_q(cells)
    rho_rows = _aggregate_rho(_rho_rows())
    assert len(cells) == 6
    assert len(q_rows) == 3
    assert q_rows[0]["npmi_at_10_median"] == 0.081
    assert q_rows[2]["selected_frozen_setting"] is True
    assert q_rows[2]["frozen_decoder_calibration"] == "training_moment_match"
    assert len(rho_rows) == 4
    assert rho_rows[0]["pooled_backoff_weight_rho"] == 0.0
    assert rho_rows[0]["nll_reduction_vs_rho_zero_median"] == 0.0
    assert rho_rows[-1]["pooled_backoff_weight_rho"] == 0.35
    assert rho_rows[-1]["selected_frozen_setting"] is True


def test_step14_figure_and_table_are_valid_and_scientifically_disclosed(tmp_path):
    q_rows = _collapse_q(_aggregate(_q_rows()))
    rho_rows = _aggregate_rho(_rho_rows())
    destination = tmp_path / "sensitivity.svg"
    _sensitivity_svg(destination, q_rows, rho_rows)
    ET.parse(destination)
    text = destination.read_text(encoding="utf-8")
    assert "topic count was fixed at K=10" in text
    assert "Pooled-backoff macro-city NLL" in text
    assert "Individual seed identifiers are intentionally omitted" in text
    assert "Separate pre-test one-axis studies" in text
    assert "no held-out-test sensitivity claim" in text

    table = _paper_table(q_rows, rho_rows)
    assert "Graph-core coverage q and decoder calibration" in table
    assert "Pooled lexical-backoff weight rho" in table
    assert "q-by-rho factorial experiment" in table
    assert "rho=0 is the exact parent-decoder identity anchor" in table


def test_main_and_runner_expose_step14_without_installing_packages():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts/run_step14.ps1").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-step14.txt").read_text(encoding="utf-8")
    assert '"step14"' in source and "run_v6_step14" in source
    assert "Step 14 R3" in script
    assert "pip install" not in script.casefold()
    assert "conda" not in script.casefold()
    assert "installs nothing" in requirements.casefold()
