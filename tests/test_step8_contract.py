import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from src.v6.data_contract import sha256_file, sha256_object, sparse_logical_hash
from src.v6.final_refit import FINAL_VARIANTS
from src.v6.step8 import (
    EXPECTED_FREEZE_IDENTITY,
    EXPECTED_SEEDS,
    EXPECTED_VARIANT_IDENTITY,
    IMPLEMENTATION,
    LEGACY_SHAPE_FAILURE,
    LEGACY_STEP8_IMPLEMENTATION,
    SCOPE,
    _audit_legacy_shape_failure,
    _build_test_metadata_contract,
    _json_values_match,
    _load_protocol,
)
from src.v6.step8_worker import (
    _load_job,
    _verify_technical_recovery,
    load_frozen_completion,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6" / "step08.yaml"


def _protocol():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["v6"]["step8"]


def test_step8_protocol_is_the_one_shot_test_not_a_refit_or_comparator_stage():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["scope"] == SCOPE
    assert protocol["data_policy"]["v6_1_evaluative_confirmation_number"] == 1
    assert protocol["data_policy"]["prior_failed_test_source_openings"] == 1
    assert protocol["data_policy"]["prior_test_metrics_computed"] == 0
    assert protocol["data_policy"]["cumulative_test_source_openings_after_success"] == 2
    assert protocol["frozen_model"]["parent_neural_fits_in_step8"] == 0
    assert protocol["frozen_model"]["optimizer_steps_in_step8"] == 0
    assert protocol["data_policy"]["comparators_available_to_worker"] is False


def test_step8_is_bound_to_the_exact_passing_step7_freeze():
    prerequisite = _protocol()["prerequisite"]
    assert prerequisite["freeze_identity"] == EXPECTED_FREEZE_IDENTITY
    assert prerequisite["variant_manifest_identity"] == EXPECTED_VARIANT_IDENTITY
    assert len(prerequisite["contract_sha256"]) == 64
    assert len(prerequisite["checkpoint_manifest_sha256"]) == 64


def test_step8_keeps_all_ten_seeds_and_five_paired_variants():
    frozen = _protocol()["frozen_model"]
    assert frozen["seeds"] == EXPECTED_SEEDS
    assert frozen["variants"] == list(FINAL_VARIANTS)
    assert frozen["pooled_mixture_weight"] == 0.35


def test_step8_discloses_that_legacy_test_scores_are_known():
    policy = _protocol()["data_policy"]
    assert policy["legacy_v4_test_scores_known_to_researcher"] is True
    assert policy["claim_globally_unseen_test_set"] is False
    assert policy["failure_policy"] == "REPORT_WITHOUT_TEST_CONDITIONED_MODEL_CHANGE"


def test_step8_corrects_assignment_count_to_modeled_matrix_count_without_model_change():
    protocol = _protocol()
    source = protocol["test_source"]
    assert source["expected_assignment_documents_before_model_exclusions"] == 78
    assert source["expected_source_documents"] == 77
    assert source["expected_pre_matrix_exclusions"] == 1
    assert len(source["expected_legacy_v4_sparse_logical_sha256"]) == 64
    addendum_path = ROOT / protocol["technical_recovery"]["addendum_relative_path"]
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    assert addendum["correction"]["model_hyperparameters_changed"] is False
    assert addendum["correction"]["hypotheses_changed"] is False
    assert addendum["access_accounting"]["prior_test_metrics_computed"] == 0


def test_step8_hypotheses_match_the_frozen_preregistration_targets():
    rows = _protocol()["hypotheses"]
    assert len(rows) == 7
    assert {row["family"] for row in rows} == {"ablation_primary_endpoints"}
    assert [row["endpoint"] for row in rows].count("npmi_at_10") == 2
    assert [row["endpoint"] for row in rows].count("c_v_at_10") == 2
    assert [row["endpoint"] for row in rows].count("macro_city_nll_per_token") == 3


def test_worker_contains_no_fit_optimizer_or_comparator_import_path():
    source = (ROOT / "src" / "v6" / "step8_worker.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "DataLoader" not in source
    assert "baseline_runner" not in source
    assert "sota_runner" not in source


def test_main_preserves_step8_entry_point():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    assert '"step8"' in source
    assert "run_v6_step8" in source


def test_json_comparison_allows_only_small_numeric_serialization_roundoff():
    expected = {"a": [0.1, {"b": 2.0}], "flag": False}
    observed = {"a": [0.10000000000001, {"b": 2.0}], "flag": False}
    assert _json_values_match(expected, observed)
    assert not _json_values_match(expected, {**observed, "flag": 0})
    assert not _json_values_match(expected, {"a": [0.2, {"b": 2.0}], "flag": False})


def test_corrected_test_row_count_is_derived_from_frozen_inclusion_metadata(tmp_path):
    assignments = pd.DataFrame(
        {
            "post_id": ["a", "b", "c", "d", "train"],
            "city": ["beijing", "shanghai", "xiamen", "beijing", "beijing"],
            "split": ["test", "test", "test", "test", "train"],
            "included_in_model": [True, True, True, False, True],
        }
    )
    split_path = tmp_path / "split_assignments.csv"
    assignments.to_csv(split_path, index=False, encoding="utf-8-sig")
    ledger = tmp_path / "input_hashes.csv"
    pd.DataFrame(
        {
            "role": ["split_metadata"],
            "relative_path": [split_path.name],
            "sha256": [sha256_file(split_path)],
        }
    ).to_csv(ledger, index=False, encoding="utf-8-sig")
    contract = _build_test_metadata_contract(
        {"source_project_root": str(tmp_path)},
        ledger,
        {
            "expected_assignment_documents_before_model_exclusions": 4,
            "expected_source_documents": 3,
            "expected_pre_matrix_exclusions": 1,
            "expected_cities": ["beijing", "shanghai", "xiamen"],
        },
        expected_split_metadata_sha256=sha256_file(split_path),
    )
    assert contract["assigned_documents_before_model_exclusions"] == 4
    assert contract["modeled_documents"] == 3
    assert contract["excluded_documents"] == 1
    assert contract["modeled_city_documents"] == {
        "beijing": 1,
        "shanghai": 1,
        "xiamen": 1,
    }


def test_legacy_shape_failure_audit_requires_no_metrics_or_completion(tmp_path):
    protocol = _protocol()
    addendum_relative = Path(
        protocol["technical_recovery"]["addendum_relative_path"]
    )
    addendum_source = ROOT / addendum_relative
    addendum_target = tmp_path / addendum_relative
    addendum_target.parent.mkdir(parents=True)
    addendum_target.write_bytes(addendum_source.read_bytes())
    lineage = {"frozen": {"path": "lineage", "sha256": "a" * 64}}
    checkpoints = [{"seed": 101, "path": "checkpoint", "sha256": "b" * 64}]
    development = {"train": {"path": "train", "sha256": "c" * 64}}
    test_sources = {
        "counts": {"path": str(tmp_path / "X_test.npz")},
        "rows": {"path": str(tmp_path / "rows_test.csv")},
    }
    source = tmp_path / "EnCOT"
    step7_job = {
        "selected_configuration": {"quota": 10},
        "model": {"topics": 10},
        "training": {"epochs": 120},
        "pooled_backoff": {"weight": 0.35},
        "calibration": {"kind": "moment_match"},
    }
    body = {
        "schema_version": 1,
        "mode": "step8",
        "implementation_version": LEGACY_STEP8_IMPLEMENTATION,
        "lineage_artifacts": lineage,
        "development_inputs": development,
        "test_sources": test_sources,
        "checkpoint_manifest": checkpoints,
        "official_source": str(source),
        "official_commit": protocol["prerequisite"]["official_commit"],
        "selected_configuration": step7_job["selected_configuration"],
        "model_configuration": step7_job["model"],
        "training_configuration": step7_job["training"],
        "pooled_backoff_configuration": step7_job["pooled_backoff"],
        "calibration_configuration": step7_job["calibration"],
        "confirmation_preregistration_sha256": protocol["prerequisite"][
            "confirmation_preregistration_sha256"
        ],
        "document_completion": protocol["document_completion"],
        "final_fit_reference": protocol["final_fit_reference"],
        "evaluation": protocol["evaluation"],
        "hypotheses": protocol["hypotheses"],
        "statistics": protocol["statistics"],
        "guardrails": protocol["guardrails"],
        "frozen_model": {
            **protocol["frozen_model"],
            "freeze_identity": EXPECTED_FREEZE_IDENTITY,
            "variant_manifest_identity": EXPECTED_VARIANT_IDENTITY,
        },
        "data_policy": {
            "v6_1_test_semantic_access_number": 1,
            "model_updates_allowed": False,
            "seed_selection_allowed": False,
            "labels_available_to_worker": False,
            "comparators_available_to_worker": False,
            "test_early_stopping_allowed": False,
            "post_test_configuration_change_allowed": False,
        },
        "test_source_contract": {
            "expected_source_documents": 78,
            "expected_vocabulary": 1148,
        },
    }
    job = {**body, "job_identity": sha256_object(body)}
    failed = tmp_path / "outputs" / "v6" / "step08_test_confirmation" / "failed"
    (failed / "worker").mkdir(parents=True)
    (failed / "worker_job.json").write_text(json.dumps(job), encoding="utf-8")
    (failed / "worker" / "test_access_state.json").write_text(
        json.dumps(
            {
                "status": "OPENING_FIXED_TEST_SOURCE",
                "job_identity": job["job_identity"],
                "cumulative_v6_1_semantic_accesses": 1,
                "technical_recovery_uses_frozen_snapshot": False,
            }
        ),
        encoding="utf-8",
    )
    (failed / "worker_failure.json").write_text(
        json.dumps(
            {
                "return_code": 1,
                "transcript": f"ValueError: {LEGACY_SHAPE_FAILURE}",
                "test_conditioned_model_change_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    receipt = _audit_legacy_shape_failure(
        tmp_path,
        protocol,
        lineage,
        checkpoints,
        development,
        test_sources,
        source,
        step7_job,
    )
    assert receipt["prior_test_source_openings"] == 1
    assert receipt["prior_test_metrics_computed"] == 0
    _verify_technical_recovery({**job, "technical_recovery": receipt})
    (failed / "worker" / "per_seed_variant_metrics.csv").write_text(
        "forbidden\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not the exact pre-evaluation"):
        _audit_legacy_shape_failure(
            tmp_path,
            protocol,
            lineage,
            checkpoints,
            development,
            test_sources,
            source,
            step7_job,
        )


def test_worker_job_cannot_contain_preopening_test_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.v6.step8_worker._verify_test_metadata_contract", lambda job: None
    )
    monkeypatch.setattr(
        "src.v6.step8_worker._verify_technical_recovery", lambda job: None
    )
    lineage = tmp_path / "lineage.json"
    development = tmp_path / "development.bin"
    checkpoint = tmp_path / "checkpoint.pt"
    test_counts = tmp_path / "X_test.npz"
    test_rows = tmp_path / "rows_test.csv"
    for path in (lineage, development, checkpoint, test_counts, test_rows):
        path.write_bytes(path.name.encode())
    body = {
        "schema_version": 1,
        "mode": "step8",
        "implementation_version": IMPLEMENTATION,
        "data_policy": {
            "v6_1_evaluative_confirmation_number": 1,
            "prior_failed_test_source_openings": 1,
            "prior_test_metrics_computed": 0,
            "prior_model_evaluations": 0,
            "cumulative_test_source_openings_after_success": 2,
            "labels_available_to_worker": False,
            "comparators_available_to_worker": False,
            "model_updates_allowed": False,
            "seed_selection_allowed": False,
            "test_early_stopping_allowed": False,
            "post_test_configuration_change_allowed": False,
        },
        "lineage_artifacts": {
            "lineage": {"path": str(lineage), "sha256": sha256_file(lineage)}
        },
        "development_inputs": {
            "development": {
                "path": str(development),
                "sha256": sha256_file(development),
            }
        },
        "checkpoint_manifest": [
            {
                "seed": 7,
                "path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            }
        ],
        "frozen_model": {"seeds": [7]},
        "test_sources": {
            "counts": {"path": str(test_counts)},
            "rows": {"path": str(test_rows)},
        },
    }
    job = {**body, "job_identity": sha256_object(body)}
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    assert _load_job(job_path)["job_identity"] == job["job_identity"]
    job["test_sources"]["counts"]["sha256"] = "forbidden"
    altered = dict(job)
    altered.pop("job_identity")
    job["job_identity"] = sha256_object(altered)
    job_path.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-opening"):
        _load_job(job_path)


def test_frozen_completion_loader_detects_matrix_tampering(tmp_path):
    output = tmp_path / "worker"
    output.mkdir()
    observed = sparse.csr_matrix(np.asarray([[1, 0], [0, 1]], dtype=np.int64))
    target = sparse.csr_matrix(np.asarray([[0, 1], [1, 0]], dtype=np.int64))
    sparse.save_npz(output / "test_completion_observed.npz", observed)
    sparse.save_npz(output / "test_completion_target.npz", target)
    rows = pd.DataFrame(
        {
            "completion_row": [0, 1],
            "post_id": ["a", "b"],
            "city": ["beijing", "shanghai"],
            "split": ["test", "test"],
        }
    )
    rows.to_csv(output / "test_completion_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["post_id", "reason"]).to_csv(
        output / "test_completion_excluded_rows.csv", index=False, encoding="utf-8-sig"
    )
    (output / "test_source_receipt.json").write_text("{}\n", encoding="utf-8")
    paths = {
        "observed": output / "test_completion_observed.npz",
        "target": output / "test_completion_target.npz",
        "rows": output / "test_completion_rows.csv",
        "excluded": output / "test_completion_excluded_rows.csv",
        "source_receipt": output / "test_source_receipt.json",
    }
    contract = {
        "schema_version": 1,
        "job_identity": "job",
        "vocabulary": 2,
        "observed_tokens": 2,
        "target_tokens": 2,
        "artifacts": {
            role: {
                "sha256": sha256_file(path),
                **(
                    {
                        "logical_sha256": sparse_logical_hash(
                            observed if role == "observed" else target
                        )
                    }
                    if role in {"observed", "target"}
                    else {}
                ),
            }
            for role, path in paths.items()
        },
    }
    contract["completion_identity"] = sha256_object(contract)
    (output / "test_completion_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    loaded = load_frozen_completion(output, expected_job_identity="job")
    assert loaded[0].shape == (2, 2)
    sparse.save_npz(
        output / "test_completion_target.npz",
        sparse.csr_matrix(np.asarray([[1, 0], [1, 0]], dtype=np.int64)),
    )
    with pytest.raises(ValueError, match="artifact changed"):
        load_frozen_completion(output, expected_job_identity="job")
