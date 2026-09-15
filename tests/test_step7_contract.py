import hashlib
import json
from pathlib import Path

import yaml

from src.v6.data_contract import sha256_file, sha256_object
from src.v6.final_refit import FINAL_VARIANTS
from src.v6.step7 import (
    EXPECTED_CONFIGURATION_FINGERPRINT,
    EXPECTED_SEEDS,
    IMPLEMENTATION,
    SCOPE,
    _artifact_manifest,
    _completed_step7_worker_exists,
    _completed_worker_freeze_is_intact,
    _diagnostic_values_match,
    _load_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6" / "step07.yaml"


def _protocol():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["v6"]["step7"]


def test_step7_protocol_is_exactly_final_refit_not_test():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["scope"] == SCOPE
    assert protocol["final_fit"]["documents"] == 695
    assert protocol["final_fit"]["tokens"] == 6944
    assert protocol["data_policy"]["test_available_to_worker"] is False
    assert protocol["data_policy"]["comparators_available_to_worker"] is False


def test_step7_freezes_selected_configuration_and_ten_reporting_seeds():
    protocol = _protocol()
    assert (
        protocol["prerequisite"]["configuration_fingerprint"]
        == EXPECTED_CONFIGURATION_FINGERPRINT
    )
    assert protocol["selected_configuration"] == {
        "prototype_quota": 10,
        "calibration": "training_moment_match",
        "pooled_mixture_weight": 0.35,
        "city_specific_backoff": "RETIRED",
    }
    assert protocol["training"]["seeds"] == EXPECTED_SEEDS
    assert protocol["training"]["seed_selection"] is False
    assert protocol["ablations"]["variants"] == list(FINAL_VARIANTS)


def test_confirmation_preregistration_hash_and_failure_policy_are_frozen():
    protocol = _protocol()
    path = ROOT / protocol["confirmation_preregistration"]["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == protocol["confirmation_preregistration"]["sha256"]
    registration = json.loads(path.read_text(encoding="utf-8"))
    assert registration["registration_status"] == "FROZEN_BEFORE_ANY_V6_1_TEST_ACCESS"
    assert (
        registration["failure_policy"] == "REPORT_WITHOUT_TEST_CONDITIONED_MODEL_CHANGE"
    )
    assert registration["final_seeds"] == EXPECTED_SEEDS


def test_step7_worker_job_has_no_test_or_comparator_input_section():
    source = (ROOT / "src" / "v6" / "step7.py").read_text(encoding="utf-8")
    worker = (ROOT / "src" / "v6" / "step7_worker.py").read_text(encoding="utf-8")
    assert '"development_inputs"' in source
    assert '"test_inputs"' not in source
    assert '"comparator_inputs"' not in source
    assert '"test_inputs"' not in worker
    assert '"comparator_inputs"' not in worker


def test_main_preserves_step7_and_step8_entry_points():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    assert '"step7"' in source
    assert "run_v6_step7" in source
    assert '"step8"' in source
    assert "run_v6_step8" in source


def test_outer_gate_accepts_only_serialization_scale_float_roundoff():
    assert _diagnostic_values_match(2.291053533554077e-07, 2.2910535335540771e-07)
    assert _diagnostic_values_match(0.0034020166444783, 0.0034020166444783434)
    assert not _diagnostic_values_match(0.0034, 0.0035)
    assert _diagnostic_values_match(True, True)
    assert not _diagnostic_values_match(True, 1)


def test_completed_worker_freeze_is_recoverable_without_refitting(tmp_path):
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    output = tmp_path / "outputs" / "v6" / "step07_final_refit_freeze" / "run"
    worker = output / "worker"
    checkpoint_rows = []
    for seed in EXPECTED_SEEDS:
        checkpoint = worker / "checkpoints" / f"seed_{seed}_final_frozen.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"frozen-{seed}".encode())
        checkpoint_rows.append(
            {
                "seed": seed,
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            }
        )
    checkpoint_manifest = worker / "checkpoint_manifest.json"
    write_json(checkpoint_manifest, checkpoint_rows)
    protocol = _protocol()
    job = {
        "schema_version": 1,
        "mode": "step7",
        "implementation_version": IMPLEMENTATION,
        "training": {"seeds": EXPECTED_SEEDS},
        "data_policy": {
            "test_available_to_worker": False,
            "labels_available_to_worker": False,
            "comparators_available_to_worker": False,
        },
        "confirmation_preregistration_sha256": protocol["confirmation_preregistration"][
            "sha256"
        ],
    }
    write_json(output / "worker_job.json", job)
    freeze = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "seeds": EXPECTED_SEEDS,
        "test_documents_used": 0,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "confirmation_preregistration_sha256": protocol["confirmation_preregistration"][
            "sha256"
        ],
        "checkpoints": checkpoint_rows,
    }
    freeze["freeze_identity"] = sha256_object(freeze)
    write_json(worker / "final_freeze_receipt.json", freeze)
    write_json(
        worker / "worker_result.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "implementation_version": IMPLEMENTATION,
            "seeds": EXPECTED_SEEDS,
            "parent_neural_fits": len(EXPECTED_SEEDS),
            "seed_selection_performed": False,
            "validation_early_stopping_performed": False,
            "hyperparameter_selection_performed": False,
            "test_documents_used": 0,
            "labels_used": 0,
            "baseline_or_sota_outputs_used": 0,
            "checkpoint_manifest": checkpoint_rows,
            "freeze_identity": freeze["freeze_identity"],
        },
    )
    write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))

    assert _completed_worker_freeze_is_intact(output, protocol)
    assert _completed_step7_worker_exists(tmp_path)
    Path(checkpoint_rows[-1]["path"]).unlink()
    assert not _completed_worker_freeze_is_intact(output, protocol)
