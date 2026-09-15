import json
from pathlib import Path

from src.v6.data_contract import sha256_file
from src.v6.step6 import (
    EXPECTED_SEEDS,
    EXPECTED_VARIANTS,
    IMPLEMENTATION,
    _load_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_step6_protocol_and_preregistration_are_hash_locked():
    config = ROOT / "configs" / "v6" / "step06.yaml"
    protocol = _load_protocol(config)
    registration = ROOT / protocol["preregistration"]["path"]
    assert sha256_file(registration) == protocol["preregistration"]["sha256"]
    receipt = json.loads(registration.read_text(encoding="utf-8"))
    assert receipt["implementation_version"] == IMPLEMENTATION
    assert receipt["decision_boundary"]["test_access_allowed"] is False


def test_step6_has_one_candidate_three_seeds_and_all_paired_variants():
    protocol = _load_protocol(ROOT / "configs" / "v6" / "step06.yaml")
    assert protocol["validation"]["candidate_count"] == 1
    assert protocol["training"]["seeds"] == EXPECTED_SEEDS
    assert protocol["evaluation"]["variants"] == EXPECTED_VARIANTS
    assert protocol["training"]["seed_selection"] is False
    assert protocol["training"]["hyperparameter_selection"] is False


def test_worker_freezes_every_training_checkpoint_before_validation_load():
    source = (ROOT / "src" / "v6" / "step6_worker.py").read_text(
        encoding="utf-8"
    )
    freeze = source.index('"training_freeze_receipt.json"')
    semantic_open = source.index("validation = _load_validation(job)")
    assert freeze < semantic_open
    assert "_train_parent(" not in source[semantic_open:]


def test_main_exposes_step6_without_removing_prior_routes():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    for stage in ("step1", "step2", "step3", "step4r2", "step5r2", "step6"):
        assert f'"{stage}"' in source
    assert "run_v6_step6" in source
