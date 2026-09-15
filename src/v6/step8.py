"""Fail-closed orchestration for V6.1 Step 8 test confirmation.

Integrity determines whether Step 8 completed.  Performance determines which
claims are supported, but an unfavorable result is never converted into a
pipeline error and may not trigger test-conditioned model changes.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.v6.confirmation_statistics import (
    build_confirmation_statistics,
    build_paired_effect_rows,
)
from src.v6.data_contract import _parse_included, sha256_file, sha256_object
from src.v6.development_stage import _verify_fingerprint
from src.v6.final_refit import FINAL_VARIANTS
from src.v6.official_encot import audit_official_source
from src.v6.step4r2 import _verify_manifest
from src.v6.step8_worker import IMPLEMENTATION, load_frozen_completion

SCOPE = "one_shot_v6_1_test_confirmation_full_plus_paired_ablations_no_comparators"
STEP7_IMPLEMENTATION = "v6_1_step7_final_development_refit_freeze_r1"
STEP7_READINESS = "READY_FOR_V6_1_STEP8_ONE_SHOT_TEST"
STEP8_READINESS = "READY_FOR_V6_1_STEP9_FAIR_COMPARATORS"
EXPECTED_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]
EXPECTED_DATA_KEY = "7774658044d3e297ed5bff31a45bcad5221228a53fe48fd5aabacebf12778c2d"
EXPECTED_OFFICIAL_COMMIT = "8ac3592165cc7851676be2776b314eb6e44e9388"
EXPECTED_FREEZE_IDENTITY = (
    "2ea097a4b05f3fd25d56bc976686a89a4c405fd7720f8929cc44310e609b11a3"
)
EXPECTED_VARIANT_IDENTITY = (
    "a03be8a99c61ee3963f410947eb903aecad64acbf13ac828b03e5047d87edb9f"
)
LEGACY_STEP8_IMPLEMENTATION = "v6_1_step8_one_shot_test_confirmation_r1"
LEGACY_SHAPE_FAILURE = "the test count matrix has an unexpected shape"
EXPECTED_STEP2_SPLIT_METADATA_SHA256 = (
    "b4277f72f613902cf068b80f190150384e1c16af78b4456e5b12bacfe0270a1b"
)
EXPECTED_TEST_SPARSE_LOGICAL_SHA256 = (
    "157d161e601b2454a1f5047ad8a735bdc227ed8c3529dc76c7dbefb0c489fb95"
)


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError(f"cannot write empty required table: {path.name}")
    fieldnames = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


class _Reporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")


def _load_protocol(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    protocol = document.get("v6", {}).get("step8", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("unsupported V6.1 Step-8 configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-8 implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-8 scope changed")
    prerequisite = protocol.get("prerequisite", {})
    if any(
        (
            prerequisite.get("implementation_version") != STEP7_IMPLEMENTATION,
            prerequisite.get("readiness") != STEP7_READINESS,
            prerequisite.get("data_key") != EXPECTED_DATA_KEY,
            prerequisite.get("official_commit") != EXPECTED_OFFICIAL_COMMIT,
            prerequisite.get("freeze_identity") != EXPECTED_FREEZE_IDENTITY,
            prerequisite.get("variant_manifest_identity") != EXPECTED_VARIANT_IDENTITY,
        )
    ):
        raise ValueError("Step-8 Step-7 prerequisite lock changed")
    frozen = protocol.get("frozen_model", {})
    if any(
        (
            list(map(int, frozen.get("seeds", []))) != EXPECTED_SEEDS,
            list(frozen.get("variants", [])) != list(FINAL_VARIANTS),
            float(frozen.get("pooled_mixture_weight", float("nan"))) != 0.35,
            int(frozen.get("parent_neural_fits_in_step8", -1)) != 0,
            int(frozen.get("optimizer_steps_in_step8", -1)) != 0,
        )
    ):
        raise ValueError("Step-8 frozen model or ablation registry changed")
    policy = protocol.get("data_policy", {})
    for key in (
        "labels_available_to_worker",
        "comparators_available_to_worker",
        "model_updates_allowed",
        "seed_selection_allowed",
        "test_early_stopping_allowed",
        "post_test_configuration_change_allowed",
    ):
        if policy.get(key) is not False:
            raise ValueError(f"Step-8 policy must keep {key}=false")
    if any(
        (
            int(policy.get("v6_1_evaluative_confirmation_number", -1)) != 1,
            int(policy.get("prior_failed_test_source_openings", -1)) != 1,
            int(policy.get("prior_test_metrics_computed", -1)) != 0,
            int(policy.get("prior_model_evaluations", -1)) != 0,
            int(policy.get("cumulative_test_source_openings_after_success", -1))
            != 2,
            policy.get("legacy_v4_test_scores_known_to_researcher") is not True,
            policy.get("claim_globally_unseen_test_set") is not False,
            policy.get("failure_policy")
            != "REPORT_WITHOUT_TEST_CONDITIONED_MODEL_CHANGE",
        )
    ):
        raise ValueError("Step-8 disclosure or failure policy changed")
    source = protocol.get("test_source", {})
    if any(
        (
            source.get("root") != "derive_from_frozen_step2_contract",
            source.get("counts_relative_path") != "data/processed/counts/X_test.npz",
            source.get("rows_relative_path") != "data/processed/counts/rows_test.csv",
            int(source.get("expected_assignment_documents_before_model_exclusions", -1))
            != 78,
            int(source.get("expected_source_documents", -1)) != 77,
            int(source.get("expected_pre_matrix_exclusions", -1)) != 1,
            int(source.get("expected_vocabulary", -1)) != 1148,
            list(map(str, source.get("expected_cities", [])))
            != ["beijing", "shanghai", "xiamen"],
            source.get("metadata_contract_source")
            != "frozen_step2_split_assignments_included_in_model",
            source.get("expected_legacy_v4_sparse_logical_sha256")
            != EXPECTED_TEST_SPARSE_LOGICAL_SHA256,
        )
    ):
        raise ValueError("the fixed Step-8 test-source identity changed")
    recovery = protocol.get("technical_recovery", {})
    if any(
        (
            recovery.get("failed_implementation") != LEGACY_STEP8_IMPLEMENTATION,
            recovery.get("failed_exception") != "ValueError",
            recovery.get("failed_message") != LEGACY_SHAPE_FAILURE,
            int(recovery.get("failed_expected_source_documents", -1)) != 78,
            recovery.get("require_no_completion_snapshot") is not True,
            recovery.get("require_no_model_evaluation_metrics") is not True,
            recovery.get("preserve_failed_output") is not True,
        )
    ):
        raise ValueError("the Step-8 technical-recovery policy changed")
    hypotheses = protocol.get("hypotheses", [])
    expected_hypotheses = [
        ("graph_value_npmi", "npmi_at_10", "full_higher"),
        ("graph_value_cv", "c_v_at_10", "full_higher"),
        ("pooled_backoff_value_nll", "macro_city_nll_per_token", "full_lower"),
        ("calibration_value_nll", "macro_city_nll_per_token", "full_lower"),
        ("joint_parent_value_npmi", "npmi_at_10", "full_higher"),
        ("joint_parent_value_cv", "c_v_at_10", "full_higher"),
        ("joint_parent_value_nll", "macro_city_nll_per_token", "full_lower"),
    ]
    observed_hypotheses = [
        (
            str(row.get("contrast_id")),
            str(row.get("endpoint")),
            str(row.get("direction")),
        )
        for row in hypotheses
    ]
    if observed_hypotheses != expected_hypotheses or any(
        row.get("family") != "ablation_primary_endpoints" for row in hypotheses
    ):
        raise ValueError("the preregistered Step-8 hypotheses changed")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step 8 may not create or modify a Python environment")
    registration = config_path.resolve().parents[2] / Path(
        "docs/V6_1_STEP08_09_CONFIRMATION_PREREGISTRATION.json"
    )
    if not registration.is_file() or sha256_file(registration) != str(
        prerequisite.get("confirmation_preregistration_sha256")
    ):
        raise ValueError("the frozen Step-8/9 preregistration changed")
    recovery_addendum = config_path.resolve().parents[2] / Path(
        str(recovery.get("addendum_relative_path", ""))
    )
    if not recovery_addendum.is_file() or sha256_file(recovery_addendum) != str(
        prerequisite.get("technical_recovery_addendum_sha256")
    ):
        raise ValueError("the Step-8 technical-recovery addendum changed")
    addendum = json.loads(recovery_addendum.read_text(encoding="utf-8-sig"))
    evidence = addendum.get("frozen_preexisting_evidence", {})
    correction = addendum.get("correction", {})
    accounting = addendum.get("access_accounting", {})
    if any(
        (
            addendum.get("event") != "STEP08_R1_EXPECTED_DOCUMENT_COUNT_MISMATCH",
            evidence.get("step2_data_key") != EXPECTED_DATA_KEY,
            evidence.get("step2_split_metadata_sha256")
            != EXPECTED_STEP2_SPLIT_METADATA_SHA256,
            int(evidence.get("test_assignments_before_model_exclusions", -1))
            != 78,
            int(evidence.get("modeled_test_assignments", -1)) != 77,
            evidence.get("legacy_v4_test_matrix_sparse_logical_sha256")
            != EXPECTED_TEST_SPARSE_LOGICAL_SHA256,
            int(correction.get("expected_source_documents_after", -1)) != 77,
            correction.get("model_hyperparameters_changed") is not False,
            int(accounting.get("prior_failed_source_openings", -1)) != 1,
            int(accounting.get("prior_test_metrics_computed", -1)) != 0,
            int(accounting.get("cumulative_source_openings_after_success", -1))
            != 2,
        )
    ):
        raise ValueError("the Step-8 technical-recovery evidence changed")
    return protocol


def _artifact_manifest(output: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "generated_artifact_manifest.json":
            continue
        files.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    result = {"schema_version": 1, "files": files}
    result["manifest_identity"] = sha256_object(result)
    return result


def _make_return_zip(root: Path, output: Path, log_path: Path) -> Path:
    destination = root / "v6_step08_results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("step08_test_confirmation") / output.name
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, "logs/v6_step08.log")
    return destination


def _json_values_match(expected: Any, observed: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and set(expected) == set(observed)
            and all(
                _json_values_match(expected[key], observed[key]) for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(expected) == len(observed)
            and all(
                _json_values_match(left, right)
                for left, right in zip(expected, observed)
            )
        )
    if isinstance(expected, (bool, np.bool_)):
        return isinstance(observed, (bool, np.bool_)) and bool(expected) == bool(
            observed
        )
    if isinstance(expected, (float, np.floating)):
        try:
            return bool(
                np.isclose(
                    float(expected),
                    float(observed),
                    rtol=1.0e-11,
                    atol=1.0e-14,
                    equal_nan=False,
                )
            )
        except (TypeError, ValueError):
            return False
    return expected == observed


def _runtime_matches(receipt: Mapping[str, Any], runtime: Path) -> bool:
    return bool(
        Path(receipt["python_executable"]).resolve() == runtime.resolve()
        and receipt.get("cuda_available") is True
        and receipt.get("packages_installed_or_changed") is False
        and receipt.get("environment_created") is False
    )


def _find_passing_step7(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    candidates = sorted(
        (root / "outputs" / "v6" / "step07_final_refit_freeze").glob(
            "*/step07_contract.json"
        ),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        output = contract_path.parent
        worker = output / "worker"
        required = {
            "worker_job": output / "worker_job.json",
            "checkpoint_manifest": worker / "checkpoint_manifest.json",
            "freeze_receipt": worker / "final_freeze_receipt.json",
            "variant_manifest": worker / "final_variant_manifest.json",
            "runtime_receipt": worker / "runtime_receipt.json",
            "source_receipt": output / "official_source_receipt.json",
            "generated_manifest": output / "generated_artifact_manifest.json",
        }
        if not all(path.is_file() for path in required.values()):
            continue
        if any(
            (
                contract.get("status") != "PASS",
                contract.get("readiness") != STEP7_READINESS,
                contract.get("implementation_version") != STEP7_IMPLEMENTATION,
                contract.get("data_key") != EXPECTED_DATA_KEY,
                contract.get("official_commit") != EXPECTED_OFFICIAL_COMMIT,
                contract.get("freeze_identity") != EXPECTED_FREEZE_IDENTITY,
                contract.get("variant_manifest_identity") != EXPECTED_VARIANT_IDENTITY,
                sha256_file(contract_path)
                != protocol["prerequisite"]["contract_sha256"],
                contract.get("contract_fingerprint")
                != protocol["prerequisite"]["contract_fingerprint"],
                not _verify_fingerprint(contract, "contract_fingerprint"),
                not _verify_manifest(output),
            )
        ):
            continue
        checkpoints = json.loads(
            required["checkpoint_manifest"].read_text(encoding="utf-8-sig")
        )
        freeze = json.loads(required["freeze_receipt"].read_text(encoding="utf-8-sig"))
        variant = json.loads(
            required["variant_manifest"].read_text(encoding="utf-8-sig")
        )
        freeze_body = dict(freeze)
        freeze_identity = str(freeze_body.pop("freeze_identity", ""))
        variant_body = dict(variant)
        variant_identity = str(variant_body.pop("variant_manifest_identity", ""))
        intact = bool(
            sha256_file(required["checkpoint_manifest"])
            == protocol["prerequisite"]["checkpoint_manifest_sha256"]
            and freeze_identity == EXPECTED_FREEZE_IDENTITY
            and sha256_object(freeze_body) == freeze_identity
            and variant_identity == EXPECTED_VARIANT_IDENTITY
            and sha256_object(variant_body) == variant_identity
            and [int(row["seed"]) for row in checkpoints] == EXPECTED_SEEDS
            and len(checkpoints) == 10
            and all(
                Path(row["path"]).is_file()
                and int(Path(row["path"]).stat().st_size) == int(row["size_bytes"])
                and sha256_file(Path(row["path"])) == str(row["sha256"])
                for row in checkpoints
            )
            and freeze.get("checkpoints") == checkpoints
            and freeze.get("test_documents_used") == 0
            and variant.get("test_status") == "UNOPENED"
        )
        if intact:
            lineage = {
                role: {
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
                for role, path in required.items()
            }
            return contract_path, contract, lineage, checkpoints, required["worker_job"]
    raise RuntimeError(
        "No exact intact passing Step-7 freeze was found. Keep the test sealed."
    )


def _source_input(
    step2_contract: Mapping[str, Any], input_hashes_path: Path, role: str
) -> dict[str, Any]:
    rows = pd.read_csv(input_hashes_path, encoding="utf-8-sig")
    selected = rows.loc[rows["role"].astype(str) == role]
    if len(selected) != 1:
        raise RuntimeError(f"the frozen Step-2 input role is not unique: {role}")
    row = selected.iloc[0]
    path = Path(str(step2_contract["source_project_root"])) / Path(
        str(row["relative_path"])
    )
    if not path.is_file() or sha256_file(path) != str(row["sha256"]):
        raise RuntimeError(f"the frozen development artifact changed: {role}")
    return {"path": str(path), "sha256": str(row["sha256"])}


def _build_test_metadata_contract(
    step2_contract: Mapping[str, Any],
    input_hashes_path: Path,
    test_source_contract: Mapping[str, Any],
    *,
    expected_split_metadata_sha256: str = EXPECTED_STEP2_SPLIT_METADATA_SHA256,
) -> dict[str, Any]:
    """Derive the test row count from already-opened frozen split metadata."""

    source = _source_input(step2_contract, input_hashes_path, "split_metadata")
    if source["sha256"] != str(expected_split_metadata_sha256):
        raise RuntimeError("the frozen Step-2 split-metadata identity changed")
    assignments = pd.read_csv(
        Path(source["path"]),
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    required = {"post_id", "city", "split", "included_in_model"}
    if required - set(assignments.columns):
        raise RuntimeError("the frozen split metadata lacks model-inclusion evidence")
    if assignments["post_id"].astype(str).duplicated().any():
        raise RuntimeError("the frozen split metadata contains duplicate post IDs")
    included = _parse_included(assignments["included_in_model"])
    assigned = assignments.loc[assignments["split"].astype(str).eq("test")].copy()
    modeled = assignments.loc[
        assignments["split"].astype(str).eq("test") & included
    ].copy()
    excluded = assignments.loc[
        assignments["split"].astype(str).eq("test") & ~included
    ].copy()
    city_order = list(map(str, test_source_contract["expected_cities"]))
    modeled_records = (
        modeled[["post_id", "city", "split"]]
        .astype(str)
        .sort_values(["post_id", "city", "split"])
        .to_dict(orient="records")
    )
    excluded_records = (
        excluded[["post_id", "city", "split"]]
        .astype(str)
        .sort_values(["post_id", "city", "split"])
        .to_dict(orient="records")
    )
    body = {
        "schema_version": 1,
        "source": source,
        "selection_rule": "split_is_test_and_included_in_model_is_true",
        "assigned_documents_before_model_exclusions": len(assigned),
        "modeled_documents": len(modeled),
        "excluded_documents": len(excluded),
        "modeled_city_documents": {
            city: int(modeled["city"].astype(str).eq(city).sum())
            for city in city_order
        },
        "modeled_rows_logical_sha256": sha256_object(modeled_records),
        "excluded_rows_logical_sha256": sha256_object(excluded_records),
    }
    if any(
        (
            body["assigned_documents_before_model_exclusions"]
            != int(
                test_source_contract[
                    "expected_assignment_documents_before_model_exclusions"
                ]
            ),
            body["modeled_documents"]
            != int(test_source_contract["expected_source_documents"]),
            body["excluded_documents"]
            != int(test_source_contract["expected_pre_matrix_exclusions"]),
            set(modeled["city"].astype(str)) != set(city_order),
        )
    ):
        raise RuntimeError(
            "the corrected Step-8 row count does not reproduce from frozen Step-2 metadata"
        )
    result = dict(body)
    result["contract_identity"] = sha256_object(body)
    return result


def _build_job_inputs(
    root: Path,
    protocol: Mapping[str, Any],
    step7_job_path: Path,
    lineage: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[str, Any],
    Path,
    Path,
    dict[str, Any],
]:
    step7_job = json.loads(step7_job_path.read_text(encoding="utf-8-sig"))
    if (
        step7_job.get("mode") != "step7"
        or step7_job.get("implementation_version") != STEP7_IMPLEMENTATION
    ):
        raise RuntimeError("the Step-7 worker-job identity changed")
    step2_item = step7_job["lineage_artifacts"]["step2_contract"]
    step2_path = Path(step2_item["path"])
    if not step2_path.is_file() or sha256_file(step2_path) != str(step2_item["sha256"]):
        raise RuntimeError("the Step-2 source-root contract changed")
    step2 = json.loads(step2_path.read_text(encoding="utf-8-sig"))
    if step2.get("data_key") != EXPECTED_DATA_KEY:
        raise RuntimeError("the Step-2 data identity changed")
    hashes_item = step7_job["lineage_artifacts"]["step2_input_hashes"]
    hashes_path = Path(hashes_item["path"])
    if not hashes_path.is_file() or sha256_file(hashes_path) != str(
        hashes_item["sha256"]
    ):
        raise RuntimeError("the Step-2 input-hash ledger changed")
    development_inputs = {
        name: dict(value) for name, value in step7_job["development_inputs"].items()
    }
    development_inputs["vocabulary"] = _source_input(step2, hashes_path, "vocabulary")
    for role, item in development_inputs.items():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"the final-fit development input changed: {role}")
    source_root = Path(str(step2["source_project_root"]))
    source_contract = protocol["test_source"]
    test_sources = {
        "counts": {
            "path": str(source_root / Path(source_contract["counts_relative_path"]))
        },
        "rows": {
            "path": str(source_root / Path(source_contract["rows_relative_path"]))
        },
    }
    # Only file existence is inspected before the one-shot worker.  No test
    # bytes, row values, shape, token count, or hash is read here.
    if not all(Path(item["path"]).is_file() for item in test_sources.values()):
        raise FileNotFoundError(
            "The fixed test count/row files are missing from the sibling SC-HTM data directory"
        )
    test_metadata = _build_test_metadata_contract(
        step2, hashes_path, source_contract
    )
    runtime_receipt_path = Path(lineage["runtime_receipt"]["path"])
    runtime_receipt = json.loads(runtime_receipt_path.read_text(encoding="utf-8-sig"))
    runtime = Path(runtime_receipt["python_executable"])
    if not runtime.is_file() or not _runtime_matches(runtime_receipt, runtime):
        raise RuntimeError("the exact Step-7 CUDA runtime is unavailable")
    source = Path(step7_job["official_source"])
    step3_specification = yaml.safe_load(
        (root / "configs" / "v6" / "step03.yaml").read_text(encoding="utf-8-sig")
    )["v6"]["step3"]["official_source"]
    source_receipt = audit_official_source(source, step3_specification)
    if (
        source_receipt["commit"] != EXPECTED_OFFICIAL_COMMIT
        or source_receipt["source_tree_clean"] is not True
    ):
        raise RuntimeError("the exact official EnCOT source changed")
    return development_inputs, test_sources, test_metadata, runtime, source, step7_job


_LEGACY_FAILURE_FORBIDDEN_ARTIFACTS = (
    "worker/test_completion_observed.npz",
    "worker/test_completion_target.npz",
    "worker/test_completion_rows.csv",
    "worker/test_completion_excluded_rows.csv",
    "worker/test_completion_contract.json",
    "worker/test_source_receipt.json",
    "worker/per_seed_variant_metrics.csv",
    "worker/per_city_variant_metrics.csv",
    "worker/top_words_by_seed.csv",
    "worker/seed_integrity.csv",
    "worker/variant_summary.csv",
    "worker/variant_summary.json",
    "worker/ablation_hypothesis_tests.csv",
    "worker/ablation_hypothesis_tests.json",
    "worker/paired_effects_by_seed.csv",
    "worker/confirmation_outcome.json",
    "worker/development_merge_receipt.json",
    "worker/step9_handoff.json",
    "worker/worker_result.json",
    "step08_contract.json",
)


def _audit_legacy_shape_failure(
    root: Path,
    protocol: Mapping[str, Any],
    lineage: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    development_inputs: Mapping[str, Any],
    test_sources: Mapping[str, Any],
    source: Path,
    step7_job: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind R2 to the single R1 shape failure without reading test data."""

    exact: list[tuple[Path, Path, Path, Path]] = []
    candidates = sorted(
        (root / "outputs" / "v6" / "step08_test_confirmation").glob("*"),
        reverse=True,
    )
    for output in candidates:
        job_path = output / "worker_job.json"
        state_path = output / "worker" / "test_access_state.json"
        failure_path = output / "worker_failure.json"
        if not job_path.is_file() or not state_path.is_file():
            continue
        try:
            old_job = json.loads(job_path.read_text(encoding="utf-8-sig"))
            state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("a prior Step-8 test-opening record is unreadable") from exc
        if old_job.get("implementation_version") != LEGACY_STEP8_IMPLEMENTATION:
            continue
        if not failure_path.is_file():
            raise RuntimeError(
                "the prior R1 test opening lacks its failure record; recovery is refused"
            )
        failure = json.loads(failure_path.read_text(encoding="utf-8-sig"))
        body = dict(old_job)
        identity = str(body.pop("job_identity", ""))
        transcript = str(failure.get("transcript", ""))
        old_policy = old_job.get("data_policy", {})
        old_test = old_job.get("test_source_contract", {})
        same_frozen_design = all(
            (
                old_job.get("schema_version") == 1,
                old_job.get("mode") == "step8",
                sha256_object(body) == identity,
                old_job.get("lineage_artifacts") == dict(lineage),
                old_job.get("development_inputs") == dict(development_inputs),
                old_job.get("test_sources") == dict(test_sources),
                old_job.get("checkpoint_manifest")
                == [dict(row) for row in checkpoints],
                old_job.get("official_source") == str(source),
                old_job.get("official_commit") == EXPECTED_OFFICIAL_COMMIT,
                old_job.get("selected_configuration")
                == step7_job.get("selected_configuration"),
                old_job.get("model_configuration") == step7_job.get("model"),
                old_job.get("training_configuration") == step7_job.get("training"),
                old_job.get("pooled_backoff_configuration")
                == step7_job.get("pooled_backoff"),
                old_job.get("calibration_configuration")
                == step7_job.get("calibration"),
                old_job.get("confirmation_preregistration_sha256")
                == protocol["prerequisite"]["confirmation_preregistration_sha256"],
                old_job.get("document_completion")
                == protocol.get("document_completion"),
                old_job.get("final_fit_reference")
                == protocol.get("final_fit_reference"),
                old_job.get("evaluation") == protocol.get("evaluation"),
                old_job.get("hypotheses") == protocol.get("hypotheses"),
                old_job.get("statistics") == protocol.get("statistics"),
                old_job.get("guardrails") == protocol.get("guardrails"),
                old_job.get("frozen_model")
                == {
                    **protocol["frozen_model"],
                    "freeze_identity": EXPECTED_FREEZE_IDENTITY,
                    "variant_manifest_identity": EXPECTED_VARIANT_IDENTITY,
                },
            )
        )
        valid_failure = all(
            (
                int(old_policy.get("v6_1_test_semantic_access_number", -1)) == 1,
                old_policy.get("model_updates_allowed") is False,
                old_policy.get("seed_selection_allowed") is False,
                old_policy.get("labels_available_to_worker") is False,
                old_policy.get("comparators_available_to_worker") is False,
                old_policy.get("test_early_stopping_allowed") is False,
                old_policy.get("post_test_configuration_change_allowed") is False,
                int(old_test.get("expected_source_documents", -1)) == 78,
                int(old_test.get("expected_vocabulary", -1)) == 1148,
                state.get("status") == "OPENING_FIXED_TEST_SOURCE",
                state.get("job_identity") == identity,
                int(state.get("cumulative_v6_1_semantic_accesses", -1)) == 1,
                state.get("technical_recovery_uses_frozen_snapshot") is False,
                int(failure.get("return_code", 0)) != 0,
                failure.get("test_conditioned_model_change_allowed") is False,
                f"ValueError: {LEGACY_SHAPE_FAILURE}" in transcript,
                not any((output / relative).exists() for relative in _LEGACY_FAILURE_FORBIDDEN_ARTIFACTS),
            )
        )
        if not same_frozen_design or not valid_failure:
            raise RuntimeError(
                "a prior R1 test opening exists but is not the exact pre-evaluation "
                "77-versus-78 shape failure authorized by the recovery addendum"
            )
        exact.append((output, job_path, state_path, failure_path))
    if len(exact) != 1:
        raise RuntimeError(
            "Step-8 R2 requires exactly one preserved R1 shape-failure opening; "
            f"found {len(exact)}. Keep all outputs and return the latest failure ZIP."
        )
    output, job_path, state_path, failure_path = exact[0]
    addendum_path = root / Path(
        str(protocol["technical_recovery"]["addendum_relative_path"])
    )
    receipt_body = {
        "schema_version": 1,
        "classification": "pre_evaluation_shape_validation_failure",
        "failed_implementation": LEGACY_STEP8_IMPLEMENTATION,
        "failed_exception": "ValueError",
        "failed_message": LEGACY_SHAPE_FAILURE,
        "failed_output": str(output),
        "evidence": {
            "worker_job": {"path": str(job_path), "sha256": sha256_file(job_path)},
            "test_access_state": {
                "path": str(state_path),
                "sha256": sha256_file(state_path),
            },
            "worker_failure": {
                "path": str(failure_path),
                "sha256": sha256_file(failure_path),
            },
            "recovery_addendum": {
                "path": str(addendum_path),
                "sha256": sha256_file(addendum_path),
            },
        },
        "absence_verified": list(_LEGACY_FAILURE_FORBIDDEN_ARTIFACTS),
        "prior_test_source_openings": 1,
        "prior_evaluative_confirmations": 0,
        "prior_test_metrics_computed": 0,
        "prior_model_evaluations": 0,
        "prior_model_updates": 0,
        "frozen_step7_lineage_matched": True,
        "failed_output_preserved": True,
    }
    receipt = dict(receipt_body)
    receipt["recovery_identity"] = sha256_object(receipt_body)
    return receipt


def _job_for_output(
    output: Path,
    config_path: Path,
    protocol: Mapping[str, Any],
    lineage: Mapping[str, Any],
    checkpoints: list[dict[str, Any]],
    development_inputs: Mapping[str, Any],
    test_sources: Mapping[str, Any],
    test_metadata_contract: Mapping[str, Any],
    technical_recovery: Mapping[str, Any],
    source: Path,
    step7_job: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "mode": "step8",
        "implementation_version": IMPLEMENTATION,
        "worker_output": str(output / "worker"),
        "config_sha256": sha256_file(config_path),
        "lineage_artifacts": dict(lineage),
        "development_inputs": dict(development_inputs),
        "test_sources": dict(test_sources),
        "test_metadata_contract": dict(test_metadata_contract),
        "technical_recovery": dict(technical_recovery),
        "checkpoint_manifest": checkpoints,
        "official_source": str(source),
        "official_commit": EXPECTED_OFFICIAL_COMMIT,
        "confirmation_preregistration_sha256": protocol["prerequisite"][
            "confirmation_preregistration_sha256"
        ],
        "selected_configuration": step7_job["selected_configuration"],
        "model_configuration": step7_job["model"],
        "training_configuration": step7_job["training"],
        "pooled_backoff_configuration": step7_job["pooled_backoff"],
        "calibration_configuration": step7_job["calibration"],
        "data_policy": protocol["data_policy"],
        "test_source_contract": protocol["test_source"],
        "document_completion": protocol["document_completion"],
        "final_fit_reference": protocol["final_fit_reference"],
        "frozen_model": {
            **protocol["frozen_model"],
            "freeze_identity": protocol["prerequisite"]["freeze_identity"],
            "variant_manifest_identity": protocol["prerequisite"][
                "variant_manifest_identity"
            ],
        },
        "evaluation": protocol["evaluation"],
        "hypotheses": protocol["hypotheses"],
        "statistics": protocol["statistics"],
        "guardrails": protocol["guardrails"],
    }
    job = dict(body)
    job["job_identity"] = sha256_object(body)
    return job


def _stream_worker(
    runtime: Path, root: Path, job_path: Path, reporter: _Reporter
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.step8_worker", "--job", str(job_path)],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    captured: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        value = line.rstrip("\r\n")
        captured.append(value)
        reporter(value)
    return int(process.wait()), "\n".join(captured)


def _find_recovery_output(
    root: Path,
    build_expected: Any,
    *,
    disclosed_legacy_failure: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        (root / "outputs" / "v6" / "step08_test_confirmation").glob("*"),
        reverse=True,
    )
    for output in candidates:
        if output.resolve() == disclosed_legacy_failure.resolve():
            continue
        if not output.is_dir() or not (output / "worker_job.json").is_file():
            continue
        worker = output / "worker"
        state = worker / "test_access_state.json"
        if not state.is_file():
            continue
        persisted = json.loads(
            (output / "worker_job.json").read_text(encoding="utf-8-sig")
        )
        expected = build_expected(output)
        if persisted != expected:
            raise RuntimeError(
                "A V6.1 test opening already exists under a different Step-8 job/code "
                "identity. Test-conditioned replacement is refused. Return that result."
            )
        return output, persisted
    return None, None


def _paper_table(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(str(row["variant"]), str(row["metric"])): row for row in summaries}
    rows: list[dict[str, Any]] = []
    for variant in FINAL_VARIANTS:
        result: dict[str, Any] = {"variant": variant}
        for metric in (
            "npmi_at_10",
            "c_v_at_10",
            "topic_diversity_at_10",
            "top_word_redundancy_at_10",
            "macro_city_nll_per_token",
            "micro_nll_per_token",
        ):
            summary = lookup[(variant, metric)]
            result[f"{metric}_mean"] = summary["mean"]
            result[f"{metric}_sample_sd"] = summary["sample_sd"]
            result[f"{metric}_ci95_low"] = summary["ci95_low"]
            result[f"{metric}_ci95_high"] = summary["ci95_high"]
        rows.append(result)
    return rows


def _existing_passing_output(root: Path) -> Path | None:
    candidates = sorted(
        (root / "outputs" / "v6" / "step08_test_confirmation").glob(
            "*/step08_contract.json"
        ),
        reverse=True,
    )
    for path in candidates:
        try:
            contract = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            contract.get("status") == "PASS"
            and contract.get("readiness") == STEP8_READINESS
            and contract.get("implementation_version") == IMPLEMENTATION
            and _verify_fingerprint(contract, "contract_fingerprint")
            and _verify_manifest(path.parent)
        ):
            return path.parent
    return None


def run_v6_step8(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError("V6.1 Step 8 requires a project directory containing 'V6'")
    protocol = _load_protocol(config_path)
    prior = _existing_passing_output(root)
    if prior is not None:
        log_path = root / "logs" / f"v6_step08_{prior.name}.log"
        return_zip = _make_return_zip(root, prior, log_path)
        print(
            "[V6.1 Step 8] RECOVERY | intact passing confirmation already exists; "
            "test source reopen=0 model updates=0",
            flush=True,
        )
        print(f"[V6.1 Step 8] return-zip={return_zip}", flush=True)
        return prior

    step7_path, step7_contract, lineage, checkpoints, step7_job_path = (
        _find_passing_step7(root, protocol)
    )
    (
        development_inputs,
        test_sources,
        test_metadata_contract,
        runtime,
        source,
        step7_job,
    ) = _build_job_inputs(root, protocol, step7_job_path, lineage)
    technical_recovery = _audit_legacy_shape_failure(
        root,
        protocol,
        lineage,
        checkpoints,
        development_inputs,
        test_sources,
        source,
        step7_job,
    )

    def build_expected(candidate: Path) -> dict[str, Any]:
        return _job_for_output(
            candidate,
            config_path,
            protocol,
            lineage,
            checkpoints,
            development_inputs,
            test_sources,
            test_metadata_contract,
            technical_recovery,
            source,
            step7_job,
        )

    recovery_output, recovery_job = _find_recovery_output(
        root,
        build_expected,
        disclosed_legacy_failure=Path(technical_recovery["failed_output"]),
    )
    recovery_mode = recovery_output is not None
    output = (
        recovery_output
        if recovery_output is not None
        else root / "outputs" / "v6" / "step08_test_confirmation" / _utc_id()
    )
    if not recovery_mode:
        output.mkdir(parents=True, exist_ok=False)
    run_id = output.name
    log_path = root / "logs" / f"v6_step08_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6.1 Step 8] START | implementation={IMPLEMENTATION}")
    report(
        "[V6.1 Step 8] scope=one evaluative confirmation after one disclosed "
        "pre-evaluation shape failure | 10 frozen seeds x 5 paired variants | "
        "fits=0 optimizer-steps=0 labels=0 comparators=0"
    )
    if recovery_mode:
        report(
            "[V6.1 Step 8] RECOVERY | current R2 opening retained; only the exact "
            "frozen completion snapshot may be reused; the earlier R1 shape failure "
            "remains preserved separately"
        )
    try:
        job = recovery_job if recovery_job is not None else build_expected(output)
        job_path = output / "worker_job.json"
        if recovery_mode:
            if json.loads(job_path.read_text(encoding="utf-8-sig")) != job:
                raise RuntimeError("the recoverable Step-8 worker job changed")
        else:
            _write_json(job_path, job)
            _write_json(output / "step7_lineage.json", lineage)
            _write_json(output / "technical_recovery_receipt.json", technical_recovery)
            (output / "resolved_v6_step08.yaml").write_text(
                yaml.safe_dump(
                    {"schema_version": 1, "v6": {"step8": protocol}}, sort_keys=False
                ),
                encoding="utf-8",
            )
        worker_output = output / "worker"
        result_path = worker_output / "worker_result.json"
        worker_complete = False
        if result_path.is_file():
            try:
                previous = json.loads(result_path.read_text(encoding="utf-8-sig"))
                worker_complete = bool(
                    previous.get("status") == "PASS"
                    and previous.get("step8_job_identity") == job["job_identity"]
                    and int(previous.get("test_source_openings_cumulative_v6_1", -1))
                    == 2
                    and int(
                        previous.get("evaluative_confirmations_cumulative_v6_1", -1)
                    )
                    == 1
                )
            except (OSError, json.JSONDecodeError):
                worker_complete = False
        if worker_complete:
            report(
                "[V6.1 Step 8] RECOVERY | completed worker reused; test source reopen=0"
            )
        else:
            code, transcript = _stream_worker(runtime, root, job_path, report)
            if code != 0:
                _write_json(
                    output / "worker_failure.json",
                    {
                        "schema_version": 1,
                        "return_code": code,
                        "transcript": transcript,
                        "test_conditioned_model_change_allowed": False,
                    },
                )
                raise RuntimeError(
                    "Step-8 exact-runtime worker failed; return the ZIP and do not change the model"
                )

        worker = json.loads(result_path.read_text(encoding="utf-8-sig"))
        completion_observed, completion_target, _completion_rows, completion = (
            load_frozen_completion(
                worker_output, expected_job_identity=job["job_identity"]
            )
        )
        metric_rows = pd.read_csv(
            worker_output / "per_seed_variant_metrics.csv", encoding="utf-8-sig"
        ).to_dict(orient="records")
        integrity_rows = pd.read_csv(
            worker_output / "seed_integrity.csv", encoding="utf-8-sig"
        ).to_dict(orient="records")
        persisted_summaries = json.loads(
            (worker_output / "variant_summary.json").read_text(encoding="utf-8-sig")
        )
        persisted_tests = json.loads(
            (worker_output / "ablation_hypothesis_tests.json").read_text(
                encoding="utf-8-sig"
            )
        )
        persisted_effect_rows = pd.read_csv(
            worker_output / "paired_effects_by_seed.csv", encoding="utf-8-sig"
        ).to_dict(orient="records")
        outcome = json.loads(
            (worker_output / "confirmation_outcome.json").read_text(
                encoding="utf-8-sig"
            )
        )
        independent_summaries, independent_tests, independent_outcome = (
            build_confirmation_statistics(
                metric_rows,
                variants=protocol["frozen_model"]["variants"],
                seeds=protocol["frozen_model"]["seeds"],
                hypotheses=protocol["hypotheses"],
                bootstrap_resamples=int(protocol["statistics"]["bootstrap_resamples"]),
                bootstrap_seed=int(protocol["statistics"]["bootstrap_seed"]),
                alpha=float(protocol["statistics"]["alpha"]),
            )
        )
        independent_effect_rows = build_paired_effect_rows(
            metric_rows,
            seeds=protocol["frozen_model"]["seeds"],
            hypotheses=protocol["hypotheses"],
        )
        access_state = json.loads(
            (worker_output / "test_access_state.json").read_text(encoding="utf-8-sig")
        )
        source_receipt = json.loads(
            (worker_output / "test_source_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        handoff_path = worker_output / "step9_handoff.json"
        handoff = json.loads(handoff_path.read_text(encoding="utf-8-sig"))
        handoff_body = dict(handoff)
        handoff_identity = str(handoff_body.pop("handoff_identity", ""))
        runtime_receipt = json.loads(
            (worker_output / "runtime_receipt.json").read_text(encoding="utf-8-sig")
        )
        metric_lookup = {
            (int(row["seed"]), str(row["variant"])): row for row in metric_rows
        }
        identity_pairs = [
            (seed, variant) for seed in EXPECTED_SEEDS for variant in FINAL_VARIANTS
        ]
        quality_identity = all(
            np.isclose(
                float(metric_lookup[(seed, FINAL_VARIANTS[0])][metric]),
                float(metric_lookup[(seed, variant)][metric]),
                atol=0.0,
                rtol=0.0,
            )
            for seed in EXPECTED_SEEDS
            for variant in (FINAL_VARIANTS[2], FINAL_VARIANTS[3])
            for metric in (
                "npmi_at_10",
                "c_v_at_10",
                "topic_diversity_at_10",
                "top_word_redundancy_at_10",
            )
        ) and all(
            np.isclose(
                float(metric_lookup[(seed, FINAL_VARIANTS[1])][metric]),
                float(metric_lookup[(seed, FINAL_VARIANTS[4])][metric]),
                atol=0.0,
                rtol=0.0,
            )
            for seed in EXPECTED_SEEDS
            for metric in (
                "npmi_at_10",
                "c_v_at_10",
                "topic_diversity_at_10",
                "top_word_redundancy_at_10",
            )
        )
        checks: list[tuple[str, bool, Any]] = [
            (
                "exact_step7_contract_and_manifest",
                sha256_file(step7_path) == protocol["prerequisite"]["contract_sha256"]
                and _verify_manifest(step7_path.parent),
                sha256_file(step7_path),
            ),
            (
                "step7_freeze_identity_exact",
                step7_contract["freeze_identity"] == EXPECTED_FREEZE_IDENTITY
                and worker["step7_freeze_identity"] == EXPECTED_FREEZE_IDENTITY,
                worker["step7_freeze_identity"],
            ),
            (
                "all_ten_checkpoint_files_unchanged",
                all(
                    Path(row["path"]).is_file()
                    and sha256_file(Path(row["path"])) == str(row["sha256"])
                    for row in checkpoints
                ),
                len(checkpoints),
            ),
            (
                "exact_runtime_reused_without_install",
                _runtime_matches(runtime_receipt, runtime),
                runtime_receipt["python_executable"],
            ),
            (
                "disclosed_source_opening_and_evaluative_confirmation_accounting",
                int(worker["test_source_openings_cumulative_v6_1"]) == 2
                and int(worker["evaluative_confirmations_cumulative_v6_1"]) == 1
                and int(access_state["cumulative_test_source_openings"]) == 2
                and int(access_state["cumulative_evaluative_confirmations"]) == 1
                and int(access_state["prior_failed_test_source_openings"]) == 1
                and access_state["status"] == "TEST_COMPLETION_FROZEN",
                {
                    "source_openings": worker[
                        "test_source_openings_cumulative_v6_1"
                    ],
                    "evaluative_confirmations": worker[
                        "evaluative_confirmations_cumulative_v6_1"
                    ],
                },
            ),
            (
                "prior_shape_failure_is_preserved_and_pre_evaluation",
                technical_recovery["prior_test_source_openings"] == 1
                and technical_recovery["prior_evaluative_confirmations"] == 0
                and technical_recovery["prior_test_metrics_computed"] == 0
                and technical_recovery["prior_model_evaluations"] == 0
                and technical_recovery["prior_model_updates"] == 0
                and sha256_object(
                    {
                        key: value
                        for key, value in technical_recovery.items()
                        if key != "recovery_identity"
                    }
                )
                == technical_recovery["recovery_identity"],
                technical_recovery["recovery_identity"],
            ),
            (
                "accurate_legacy_test_disclosure",
                protocol["data_policy"]["legacy_v4_test_scores_known_to_researcher"]
                is True
                and protocol["data_policy"]["claim_globally_unseen_test_set"] is False,
                protocol["data_policy"]["accurate_claim"],
            ),
            (
                "fixed_test_source_partition",
                int(completion["source_documents"]) == 77
                and int(completion["vocabulary"]) == 1148
                and source_receipt["validation"]["logical_sha256"]
                == EXPECTED_TEST_SPARSE_LOGICAL_SHA256,
                {
                    "source_documents": completion["source_documents"],
                    "vocabulary": completion["vocabulary"],
                },
            ),
            (
                "test_matrix_rows_reproduce_from_frozen_step2_inclusion_metadata",
                int(test_metadata_contract["assigned_documents_before_model_exclusions"])
                == 78
                and int(test_metadata_contract["modeled_documents"]) == 77
                and int(test_metadata_contract["excluded_documents"]) == 1
                and source_receipt["validation"]["modeled_rows_logical_sha256"]
                == test_metadata_contract["modeled_rows_logical_sha256"]
                and source_receipt["validation"]["city_documents"]
                == test_metadata_contract["modeled_city_documents"],
                {
                    "assigned": test_metadata_contract[
                        "assigned_documents_before_model_exclusions"
                    ],
                    "modeled": test_metadata_contract["modeled_documents"],
                    "excluded": test_metadata_contract["excluded_documents"],
                },
            ),
            (
                "completion_snapshot_reconstructs_and_has_two_nonempty_sides",
                int(completion["maximum_reconstruction_error"]) == 0
                and completion["all_retained_sides_nonempty"] is True
                and int(completion_observed.sum()) == int(completion["observed_tokens"])
                and int(completion_target.sum()) == int(completion["target_tokens"]),
                completion["completion_identity"],
            ),
            (
                "test_target_excluded_from_inference",
                completion["test_target_used_for_inference"] is False
                and all(
                    row["target_half_used_for_assignment"] in (False, "False", 0)
                    for row in integrity_rows
                ),
                False,
            ),
            (
                "quality_reference_excludes_test_counts",
                completion["quality_metrics_use_test_counts"] is False
                and all(
                    row["quality_reference"] == "same_695_document_final_fit_counts"
                    for row in metric_rows
                ),
                "695 final-fit documents",
            ),
            (
                "complete_ten_seed_five_variant_grid",
                len(metric_rows) == 50 and list(metric_lookup) == identity_pairs,
                len(metric_rows),
            ),
            (
                "paired_reporting_affinity_identities",
                quality_identity,
                "graph family and official-parent family",
            ),
            (
                "all_reported_metrics_finite",
                bool(
                    np.isfinite(
                        pd.DataFrame(metric_rows)[
                            [
                                "npmi_at_10",
                                "c_v_at_10",
                                "topic_diversity_at_10",
                                "top_word_redundancy_at_10",
                                "macro_city_nll_per_token",
                                "micro_nll_per_token",
                                "micro_perplexity",
                            ]
                        ].to_numpy(dtype=np.float64)
                    ).all()
                ),
                True,
            ),
            (
                "probability_rows_valid",
                float(worker["maximum_probability_sum_error"])
                <= float(protocol["guardrails"]["maximum_probability_sum_error"]),
                worker["maximum_probability_sum_error"],
            ),
            (
                "official_parent_probability_reproduced",
                float(worker["maximum_parent_probability_reproduction_error"])
                <= float(
                    protocol["guardrails"][
                        "maximum_parent_probability_reproduction_error"
                    ]
                ),
                worker["maximum_parent_probability_reproduction_error"],
            ),
            (
                "all_frozen_parent_states_unchanged",
                worker["all_parent_states_unchanged"] is True
                and all(
                    row["parent_state_sha256_before"]
                    == row["parent_state_sha256_after"]
                    for row in integrity_rows
                ),
                worker["all_parent_states_unchanged"],
            ),
            (
                "zero_fits_optimizer_steps_or_parameter_updates",
                int(worker["parent_neural_fits"]) == 0
                and int(worker["optimizer_steps"]) == 0
                and int(worker["parameters_updated"]) == 0,
                {
                    "fits": worker["parent_neural_fits"],
                    "steps": worker["optimizer_steps"],
                    "updates": worker["parameters_updated"],
                },
            ),
            (
                "no_seed_selection_or_test_early_stopping",
                worker["seed_selection_performed"] is False
                and worker["test_early_stopping_performed"] is False,
                False,
            ),
            (
                "no_test_conditioned_model_change",
                worker["test_conditioned_model_change_performed"] is False
                and outcome["test_conditioned_model_change_allowed"] is False,
                False,
            ),
            (
                "no_labels_baselines_or_sota_access",
                int(worker["labels_used"]) == 0
                and int(worker["baseline_outputs_used"]) == 0
                and int(worker["sota_outputs_used"]) == 0,
                0,
            ),
            (
                "variant_summaries_independently_reproduced",
                _json_values_match(independent_summaries, persisted_summaries),
                len(persisted_summaries),
            ),
            (
                "paired_tests_and_holm_adjustment_independently_reproduced",
                _json_values_match(independent_tests, persisted_tests),
                len(persisted_tests),
            ),
            (
                "all_per_seed_paired_effects_retained_and_reproduced",
                len(persisted_effect_rows) == 70
                and _json_values_match(independent_effect_rows, persisted_effect_rows),
                len(persisted_effect_rows),
            ),
            (
                "confirmation_outcome_counts_reproduced",
                all(
                    independent_outcome[key] == outcome[key]
                    for key in independent_outcome
                ),
                independent_outcome,
            ),
            (
                "step9_handoff_is_complete_and_frozen",
                handoff["status"] == "FROZEN_FOR_STEP9"
                and sha256_object(handoff_body) == handoff_identity
                and handoff_identity == worker["step9_handoff_identity"]
                and len(handoff["completion_artifacts"]) == 4
                and all(
                    Path(row["path"]).is_file()
                    and sha256_file(Path(row["path"])) == str(row["sha256"])
                    for row in handoff["completion_artifacts"]
                ),
                handoff_identity,
            ),
            (
                "failure_policy_is_report_without_retuning",
                protocol["data_policy"]["failure_policy"]
                == "REPORT_WITHOUT_TEST_CONDITIONED_MODEL_CHANGE",
                protocol["data_policy"]["failure_policy"],
            ),
        ]
        hard_rows = [
            {
                "check": name,
                "required": True,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
            for name, passed, observed in checks
        ]
        _write_csv(output / "hard_checks.csv", hard_rows)
        failures = [row["check"] for row in hard_rows if row["status"] == "FAIL"]
        status = "PASS" if not failures else "FAIL"
        paper_rows = _paper_table(persisted_summaries)
        _write_csv(output / "paper_ready_full_and_ablation_table.csv", paper_rows)
        full_summary = {
            row["metric"]: row
            for row in persisted_summaries
            if row["variant"] == FINAL_VARIANTS[0]
        }
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": SCOPE,
            "implementation_version": IMPLEMENTATION,
            "readiness": STEP8_READINESS if status == "PASS" else "BLOCKED_NO_RETUNING",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "performance_is_not_an_integrity_gate": True,
            "confirmation_conclusion": outcome["confirmation_conclusion"],
            "prespecified_endpoints_supported": outcome["supported_endpoint_count"],
            "prespecified_endpoints_total": outcome["primary_endpoint_tests"],
            "all_full_model_guardrails_passed": outcome[
                "all_full_model_guardrails_passed"
            ],
            "full_model_summary": full_summary,
            "data_key": EXPECTED_DATA_KEY,
            "official_commit": EXPECTED_OFFICIAL_COMMIT,
            "step7_contract_sha256": sha256_file(step7_path),
            "step7_freeze_identity": EXPECTED_FREEZE_IDENTITY,
            "step8_job_identity": job["job_identity"],
            "test_completion_identity": completion["completion_identity"],
            "test_source_documents": completion["source_documents"],
            "test_completion_documents": completion["retained_documents"],
            "test_completion_target_tokens": completion["target_tokens"],
            "test_source_openings_cumulative_v6_1": 2,
            "evaluative_confirmations_cumulative_v6_1": 1,
            "prior_failed_test_source_openings": 1,
            "prior_test_metrics_computed": 0,
            "technical_recovery_identity": technical_recovery[
                "recovery_identity"
            ],
            "parent_neural_fits": 0,
            "optimizer_steps": 0,
            "test_conditioned_model_changes": 0,
            "reporting_seeds": EXPECTED_SEEDS,
            "variant_count": len(FINAL_VARIANTS),
            "labels_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "step9_handoff_identity": handoff_identity,
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step08_contract.json", contract)
        _write_json(
            output / "generated_artifact_manifest.json", _artifact_manifest(output)
        )
        report(
            f"[V6.1 Step 8] {status} | hard checks="
            f"{contract['hard_checks_passed']}/{contract['hard_checks_total']}"
        )
        report(
            f"[V6.1 Step 8] full model | NPMI="
            f"{full_summary['npmi_at_10']['mean']:+.6f}±"
            f"{full_summary['npmi_at_10']['sample_sd']:.6f} "
            f"C_v={full_summary['c_v_at_10']['mean']:.6f}±"
            f"{full_summary['c_v_at_10']['sample_sd']:.6f} "
            f"macro-NLL={full_summary['macro_city_nll_per_token']['mean']:.6f}±"
            f"{full_summary['macro_city_nll_per_token']['sample_sd']:.6f}"
        )
        report(
            f"[V6.1 Step 8] ablation outcome | supported endpoints="
            f"{outcome['supported_endpoint_count']}/{outcome['primary_endpoint_tests']} | "
            f"{outcome['confirmation_conclusion']}"
        )
        report(f"[V6.1 Step 8] contract={output / 'step08_contract.json'}")
        return_zip = _make_return_zip(root, output, log_path)
        report(f"[V6.1 Step 8] return-zip={return_zip}")
        _make_return_zip(root, output, log_path)
        if failures:
            raise AssertionError(
                f"Step-8 confirmation integrity failed: {failures}; return the ZIP and do not retune"
            )
        return output
    except Exception:
        if output.exists():
            _write_json(
                output / "generated_artifact_manifest.json", _artifact_manifest(output)
            )
            _make_return_zip(root, output, log_path)
        raise
