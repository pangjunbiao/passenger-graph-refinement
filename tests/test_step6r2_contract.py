from pathlib import Path

import yaml

from src.v6.step6r2 import (
    EXPECTED_CALIBRATIONS,
    EXPECTED_FAILED_CHECKS,
    EXPECTED_QUOTAS,
    IMPLEMENTATION,
    SCOPE,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    document = yaml.safe_load(
        (ROOT / "configs" / "v6" / "step06r2.yaml").read_text(
            encoding="utf-8"
        )
    )
    return document["v6"]["step6r2"]


def test_recovery_is_explicitly_development_not_reconfirmation():
    protocol = _protocol()
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["scope"] == SCOPE
    assert protocol["data_policy"]["validation_status"] == "SPENT_DEVELOPMENT"
    assert protocol["data_policy"]["validation_confirmation_claim_allowed"] is False
    assert protocol["data_policy"]["test_available_to_worker"] is False
    assert protocol["prerequisite"]["exact_failed_checks"] == EXPECTED_FAILED_CHECKS


def test_recovery_candidate_family_is_small_and_frozen():
    protocol = _protocol()
    assert protocol["candidates"]["prototype_quotas"] == EXPECTED_QUOTAS
    assert protocol["candidates"]["calibrations"] == EXPECTED_CALIBRATIONS
    assert protocol["candidates"]["candidate_count"] == 6
    assert protocol["candidates"]["parent_refits"] == 0
    assert protocol["calibration"]["fit_partition"] == (
        "all_620_original_training_rows_only"
    )
    assert protocol["calibration"]["validation_fit_allowed"] is False


def test_unsupported_city_specific_backoff_is_retired_not_hidden():
    protocol = _protocol()
    assert protocol["model_revision"]["city_specific_backoff"] == "RETIRED"
    assert protocol["model_revision"]["replacement"] == (
        "training_only_pooled_lexical_backoff"
    )
    source = (ROOT / "src" / "v6" / "step6r2_worker.py").read_text(
        encoding="utf-8"
    )
    assert "retired_city_mechanism_receipt.json" in source
    assert "test_available_to_worker" in source


def test_main_exposes_recovery_without_changing_failed_step6_route():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    assert '"step6"' in source
    assert '"step6r2"' in source
    assert "run_v6_step6" in source
    assert "run_v6_step6r2" in source


def test_recovery_refuses_a_second_validation_reuse():
    source = (ROOT / "src" / "v6" / "step6r2.py").read_text(encoding="utf-8")
    assert "prior_reuse_receipts" in source
    assert "do not rerun it" in source
