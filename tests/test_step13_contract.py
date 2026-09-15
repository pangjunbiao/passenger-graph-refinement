from __future__ import annotations

from pathlib import Path

from src.v6.step13 import (
    EXPECTED_SEEDS,
    IMPLEMENTATION,
    _count_matrix,
    _load_protocol,
    _tokenize,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/step13.yaml"


def test_step13_protocol_freezes_external_split_and_all_firewalls():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["training"]["seeds"] == EXPECTED_SEEDS
    assert protocol["split"]["original_train_documents"] == 391
    assert protocol["split"]["spent_validation_documents"] == 84
    assert protocol["split"]["heldout_test_documents"] == 84
    assert protocol["policy"]["test_tokens_define_vocabulary"] is False
    assert protocol["policy"]["test_target_available_to_fit_or_inference"] is False
    assert protocol["policy"]["coordinates_available_to_models"] is False
    assert protocol["policy"][
        "exact_state_performance_claim_under_language_mismatch_allowed"
    ] is False


def test_external_tokenizer_is_deterministic_and_canonicalizes_placeholders():
    protocol = _load_protocol(CONFIG)
    observed = _tokenize(
        "@[agency name] BUS [line number] isn't late at [location name]!",
        protocol,
    )
    assert observed == ["agency", "bus", "route", "late", "location"]
    assert _tokenize("HTTPS://example.com RT @user", protocol) == ["url", "user"]


def test_external_count_matrix_uses_only_declared_vocabulary():
    matrix = _count_matrix(
        [["bus", "bus", "late"], ["train", "unknown"]],
        ["bus", "late", "train"],
    )
    assert matrix.shape == (2, 3)
    assert matrix.toarray().tolist() == [[2, 1, 0], [0, 0, 1]]


def test_main_exposes_external_replication_stage():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    assert '"step13"' in source
    assert "run_v6_step13" in source


def test_step13_runner_installs_no_python_package():
    script = (ROOT / "scripts/run_step13.ps1").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-step13.txt").read_text(encoding="utf-8")
    assert "pip install" not in script.casefold()
    assert "conda" not in script.casefold()
    assert "installs nothing" in requirements.casefold()


def test_step13_supplies_neuromax_execution_lock_and_exact_tie_handling():
    source = (ROOT / "src/v6/step13.py").read_text(encoding="utf-8")
    assert '"execution_lock_sha256"' in source
    assert "nonzero_effect = effect[effect != 0.0]" in source
    assert "external_ablation_targeted_tests.csv" in source
