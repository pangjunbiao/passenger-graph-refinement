"""Fail-closed orchestration for the V6.1 post-validation recovery study."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping
import zipfile

import numpy as np
import pandas as pd
import yaml

from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_stage import _verify_fingerprint
from src.v6.official_encot import audit_official_source
from src.v6.recovery_gate import (
    select_recovery_candidate,
    summarize_recovery_candidate,
)
from src.v6.step4r2 import _verify_manifest


IMPLEMENTATION = "v6_1_step6r2_spent_validation_development_recovery_r1"
SCOPE = "spent_validation_transparent_model_selection_no_test_no_labels_no_comparators"
FAILED_STEP6_IMPLEMENTATION = "v6_step6_one_shot_validation_freeze_gate_r1"
FAILED_STEP6_READINESS = "BLOCKED_DO_NOT_OPEN_TEST_OR_COMPARATORS"
EXPECTED_SEEDS = [20260910, 20270910, 20280910]
EXPECTED_FAILED_CHECKS = [
    "city_vs_matched_pooling_practical_effect",
    "city_vs_matched_pooling_seed_consistency",
    "complete_model_vs_official_parent_practical_effect",
]
EXPECTED_QUOTAS = [8, 9, 10]
EXPECTED_CALIBRATIONS = ["native_decoder", "training_moment_match"]


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty required table: {path.name}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    protocol = document.get("v6", {}).get("step6r2", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("unsupported V6.1 Step-6R2 configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-6R2 implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-6R2 scope changed")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step-6R2 may not create or alter an environment")
    prerequisite = protocol.get("prerequisite", {})
    if any(
        (
            prerequisite.get("implementation_version")
            != FAILED_STEP6_IMPLEMENTATION,
            prerequisite.get("status") != "FAIL",
            prerequisite.get("readiness") != FAILED_STEP6_READINESS,
            list(prerequisite.get("exact_failed_checks", []))
            != EXPECTED_FAILED_CHECKS,
        )
    ):
        raise ValueError("Step-6R2 failed-validation prerequisite changed")
    policy = protocol.get("data_policy", {})
    if any(
        (
            policy.get("validation_status") != "SPENT_DEVELOPMENT",
            int(policy.get("validation_semantic_access_number", -1)) != 2,
            policy.get("validation_confirmation_claim_allowed") is not False,
            policy.get("test_available_to_worker") is not False,
            policy.get("labels_available_to_worker") is not False,
            policy.get("baseline_or_sota_available_to_worker") is not False,
        )
    ):
        raise ValueError("Step-6R2 data-use disclosure changed")
    revision = protocol.get("model_revision", {})
    if any(
        (
            revision.get("city_specific_backoff") != "RETIRED",
            revision.get("replacement")
            != "training_only_pooled_lexical_backoff",
            float(revision.get("pooled_mixture_weight", float("nan"))) != 0.35,
        )
    ):
        raise ValueError("Step-6R2 evidence-driven model revision changed")
    candidates = protocol.get("candidates", {})
    if any(
        (
            list(map(int, candidates.get("prototype_quotas", [])))
            != EXPECTED_QUOTAS,
            list(candidates.get("calibrations", []))
            != EXPECTED_CALIBRATIONS,
            int(candidates.get("candidate_count", -1)) != 6,
            int(candidates.get("parent_refits", -1)) != 0,
            candidates.get("seed_selection") is not False,
        )
    ):
        raise ValueError("Step-6R2 candidate family changed")
    calibration = protocol.get("calibration", {})
    if any(
        (
            calibration.get("fit_partition")
            != "all_620_original_training_rows_only",
            calibration.get("validation_fit_allowed") is not False,
            float(calibration.get("variance_floor", 0.0)) <= 0.0,
            float(calibration.get("change_tolerance", -1.0)) != 0.0,
        )
    ):
        raise ValueError("Step-6R2 calibration contract changed")
    return protocol


def _artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
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
    destination = root / "v6_step06r2_results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("step06r2_development_recovery") / output.name
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, "logs/v6_step06r2.log")
    return destination


def _find_failed_step6(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    prerequisite = protocol["prerequisite"]
    candidates = sorted(
        (root / "outputs" / "v6" / "step06_validation_gate").glob(
            "*/step06_contract.json"
        ),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        output = contract_path.parent
        if any(
            (
                contract.get("status") != "FAIL",
                contract.get("readiness") != FAILED_STEP6_READINESS,
                contract.get("implementation_version")
                != FAILED_STEP6_IMPLEMENTATION,
                contract.get("data_key") != prerequisite["data_key"],
                contract.get("official_commit")
                != prerequisite["official_commit"],
                list(contract.get("failed_checks", []))
                != EXPECTED_FAILED_CHECKS,
                int(contract.get("test_documents_used", -1)) != 0,
                int(contract.get("baseline_outputs_used", -1)) != 0,
                int(contract.get("sota_outputs_used", -1)) != 0,
                int(contract.get("validation_semantic_open_count", -1)) != 1,
                not _verify_fingerprint(contract, "contract_fingerprint"),
                not _verify_manifest(output),
            )
        ):
            continue
        paths = {
            "failed_step6_contract": contract_path,
            "failed_step6_worker_job": output / "worker_job.json",
            "failed_step6_worker_result": output / "worker" / "worker_result.json",
            "failed_step6_validation_metrics": output
            / "worker"
            / "validation_seed_metrics.csv",
            "failed_step6_validation_summary": output
            / "worker"
            / "validation_summary.json",
            "failed_step6_checkpoint_manifest": output
            / "worker"
            / "checkpoint_manifest.json",
            "failed_step6_runtime_receipt": output
            / "worker"
            / "runtime_receipt.json",
            "failed_step6_training_graph_receipt": output
            / "worker"
            / "training_graph_receipt.json",
        }
        if not all(path.is_file() for path in paths.values()):
            continue
        checkpoints = json.loads(
            paths["failed_step6_checkpoint_manifest"].read_text(
                encoding="utf-8-sig"
            )
        )
        if (
            len(checkpoints) != 3
            or [int(row.get("seed", -1)) for row in checkpoints]
            != EXPECTED_SEEDS
            or not all(
                Path(row["path"]).is_file()
                and sha256_file(Path(row["path"])) == str(row["sha256"])
                for row in checkpoints
            )
        ):
            continue
        runtime_receipt = json.loads(
            paths["failed_step6_runtime_receipt"].read_text(encoding="utf-8-sig")
        )
        runtime = Path(runtime_receipt["python_executable"])
        if not runtime.is_file() or runtime_receipt.get("packages_installed_or_changed") is not False:
            continue
        lineage = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        }
        return contract_path, contract, lineage, checkpoints, runtime
    raise RuntimeError(
        "No intact failed Step-6 run with its three frozen local checkpoints was found"
    )


def _stream_worker(
    runtime: Path, root: Path, job_path: Path, reporter: _Reporter
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.step6r2_worker", "--job", str(job_path)],
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


def _equivalent(left: Any, right: Any, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(a, b, tolerance) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float, np.integer, np.floating))
        and not isinstance(left, (bool, np.bool_))
        and isinstance(right, (int, float, np.integer, np.floating))
        and not isinstance(right, (bool, np.bool_))
    ):
        return bool(
            np.isfinite(float(left))
            and np.isfinite(float(right))
            and abs(float(left) - float(right)) <= tolerance
        )
    return left == right


def run_v6_step6r2(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError("V6.1 Step 6R2 requires a project directory containing 'V6'")
    protocol = _load_protocol(config_path)
    prior_reuse_receipts = list(
        (root / "outputs" / "v6" / "step06r2_development_recovery").glob(
            "*/worker/validation_reuse_receipt.json"
        )
    )
    if prior_reuse_receipts:
        raise RuntimeError(
            "Step 6R2 already opened the spent validation surface; do not rerun it. "
            "Return the existing result ZIP/output instead."
        )
    run_id = _utc_id()
    output = root / "outputs" / "v6" / "step06r2_development_recovery" / run_id
    output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"v6_step06r2_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6.1 Step 6R2] START | implementation={IMPLEMENTATION}")
    report(
        "[V6.1 Step 6R2] scope=spent validation development selection | "
        "frozen-parent refits=0 test=0 labels=0 comparators=0"
    )
    try:
        step6_path, step6_contract, lineage, checkpoints, runtime = _find_failed_step6(
            root, protocol
        )
        old_job = json.loads(
            Path(lineage["failed_step6_worker_job"]["path"]).read_text(
                encoding="utf-8-sig"
            )
        )
        graph_receipt = json.loads(
            Path(lineage["failed_step6_training_graph_receipt"]["path"])
            .read_text(encoding="utf-8-sig")
        )
        if any(
            (
                old_job.get("implementation_version")
                != FAILED_STEP6_IMPLEMENTATION,
                list(map(int, old_job.get("training", {}).get("seeds", [])))
                != EXPECTED_SEEDS,
                old_job.get("official_commit")
                != protocol["prerequisite"]["official_commit"],
            )
        ):
            raise RuntimeError("the failed Step-6 worker job changed identity")
        step3_protocol = yaml.safe_load(
            (root / "configs" / "v6" / "step03.yaml").read_text(
                encoding="utf-8-sig"
            )
        )["v6"]["step3"]
        source_receipt = audit_official_source(
            Path(old_job["official_source"]), step3_protocol["official_source"]
        )
        if source_receipt["commit"] != protocol["prerequisite"]["official_commit"]:
            raise RuntimeError("the official EnCOT source commit changed")
        _write_json(output / "official_source_receipt.json", source_receipt)
        _write_json(
            output / "failed_step6_lineage.json",
            {
                "schema_version": 1,
                "contract_status": step6_contract["status"],
                "failed_checks": step6_contract["failed_checks"],
                "lineage_artifacts": lineage,
                "checkpoint_manifest": checkpoints,
            },
        )
        worker_output = output / "worker"
        job = {
            "schema_version": 1,
            "mode": "step6r2",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output),
            "lineage_artifacts": lineage,
            "checkpoint_manifest": checkpoints,
            "official_source": old_job["official_source"],
            "official_commit": old_job["official_commit"],
            "model": old_job["model"],
            "global_context": old_job["global_context"],
            "expected_training_graph_sha256": graph_receipt["logical_sha256"],
            "data_policy": protocol["data_policy"],
            "model_revision": protocol["model_revision"],
            "candidates": protocol["candidates"],
            "calibration": protocol["calibration"],
            "projection": protocol["projection"],
            "lexical_graph": protocol["lexical_graph"],
            "evaluation": protocol["evaluation"],
            "gates": protocol["gates"],
        }
        job_path = output / "worker_job.json"
        _write_json(job_path, job)
        code, transcript = _stream_worker(runtime, root, job_path, report)
        if code != 0:
            _write_json(
                output / "worker_failure.json",
                {"schema_version": 1, "return_code": code, "transcript": transcript},
            )
            raise RuntimeError("Step-6R2 exact-runtime worker failed")

        worker = json.loads(
            (worker_output / "worker_result.json").read_text(encoding="utf-8-sig")
        )
        seed_metrics = pd.read_csv(
            worker_output / "candidate_seed_metrics.csv", encoding="utf-8-sig"
        )
        worker_summaries = json.loads(
            (worker_output / "candidate_summaries.json").read_text(
                encoding="utf-8-sig"
            )
        )
        independent_summaries: list[dict[str, Any]] = []
        for summary in worker_summaries:
            candidate_rows = seed_metrics.loc[
                seed_metrics["candidate_id"] == summary["candidate_id"]
            ].to_dict(orient="records")
            independent_summaries.append(
                summarize_recovery_candidate(
                    candidate_rows,
                    gates=protocol["gates"],
                    topic_stability=summary["topic_stability"],
                    graph_removed_topic_stability=summary["graph_removed_topic_stability"],
                )
            )
        independent_selected = select_recovery_candidate(independent_summaries)
        selected_configuration_path = worker_output / "selected_configuration.json"
        selected_configuration = (
            json.loads(selected_configuration_path.read_text(encoding="utf-8-sig"))
            if selected_configuration_path.is_file()
            else None
        )
        recovery_runtime = json.loads(
            (worker_output / "runtime_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        validation_reuse = json.loads(
            (worker_output / "validation_reuse_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        checks: list[tuple[str, bool, Any]] = [
            ("failed_step6_contract_intact", sha256_file(step6_path) == lineage["failed_step6_contract"]["sha256"], lineage["failed_step6_contract"]["sha256"]),
            ("failed_step6_outcome_preserved", step6_contract["status"] == "FAIL" and step6_contract["failed_checks"] == EXPECTED_FAILED_CHECKS, step6_contract["failed_checks"]),
            ("validation_reclassified_as_spent_development", worker["validation_status"] == "SPENT_DEVELOPMENT" and validation_reuse["status"] == "SPENT_DEVELOPMENT" and int(validation_reuse["semantic_access_number"]) == 2, validation_reuse),
            ("confirmation_claim_disabled", protocol["data_policy"]["validation_confirmation_claim_allowed"] is False and validation_reuse["confirmation_claim_allowed"] is False, False),
            ("exact_runtime_reused_without_install", Path(recovery_runtime["python_executable"]).resolve() == runtime.resolve() and recovery_runtime["packages_installed_or_changed"] is False, recovery_runtime["python_executable"]),
            ("official_source_commit_unchanged", source_receipt["source_tree_clean"] is True and source_receipt["commit"] == protocol["prerequisite"]["official_commit"], source_receipt["commit"]),
            ("three_frozen_parents_reused_without_refit", int(worker["frozen_parent_checkpoints_reused"]) == 3 and int(worker["parent_neural_refits"]) == 0, {"reused": worker["frozen_parent_checkpoints_reused"], "refits": worker["parent_neural_refits"]}),
            ("six_declared_candidates_evaluated", int(worker["candidate_configurations_evaluated"]) == 6 and int(worker["candidate_seed_evaluations"]) == 18, {"candidates": worker["candidate_configurations_evaluated"], "seed_evaluations": worker["candidate_seed_evaluations"]}),
            ("independent_candidate_summaries_reproduced", _equivalent(worker_summaries, independent_summaries), sha256_object(independent_summaries)),
            ("independent_candidate_selection_reproduced", _equivalent(worker["selected_candidate"], independent_selected), independent_selected["candidate_id"] if independent_selected else None),
            ("lineage_numerics_reproduced", float(worker["maximum_lineage_reproduction_error"]) <= float(protocol["gates"]["maximum_lineage_reproduction_error"]), worker["maximum_lineage_reproduction_error"]),
            ("city_specific_backoff_retired", worker["city_specific_backoff_retired"] is True and protocol["model_revision"]["city_specific_backoff"] == "RETIRED", True),
            ("selected_candidate_passes_all_declared_gates", independent_selected is not None and not independent_selected["failed_gates"], independent_selected["failed_gates"] if independent_selected else "no candidate"),
            ("selected_configuration_frozen", selected_configuration is not None and selected_configuration.get("test_status") == "UNOPENED" and selected_configuration.get("configuration_fingerprint") == sha256_object({key: value for key, value in selected_configuration.items() if key != "configuration_fingerprint"}), selected_configuration.get("configuration_fingerprint") if selected_configuration else None),
            ("test_access_zero", int(worker["test_documents_used"]) == 0, worker["test_documents_used"]),
            ("label_access_zero", int(worker["labels_used"]) == 0, worker["labels_used"]),
            ("comparator_access_zero", int(worker["baseline_or_sota_outputs_used"]) == 0, worker["baseline_or_sota_outputs_used"]),
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
        (output / "resolved_v6_step06r2.yaml").write_text(
            yaml.safe_dump(
                {"schema_version": 1, "v6": {"step6r2": protocol}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": SCOPE,
            "implementation_version": IMPLEMENTATION,
            "readiness": "READY_FOR_V6_1_FINAL_REFIT_AND_ONE_SHOT_TEST" if status == "PASS" else "BLOCKED_KEEP_TEST_SEALED",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "failed_step6_contract_sha256": sha256_file(step6_path),
            "failed_step6_status_preserved": True,
            "validation_status": "SPENT_DEVELOPMENT",
            "validation_confirmation_claim_allowed": False,
            "validation_documents_used": worker["validation_documents_used"],
            "validation_target_tokens": worker["validation_target_tokens"],
            "hyperparameter_selection_performed": True,
            "city_specific_backoff_retired": True,
            "selected_configuration": selected_configuration,
            "selected_candidate_summary": independent_selected,
            "parent_neural_refits": 0,
            "test_documents_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "data_key": step6_contract["data_key"],
            "official_commit": step6_contract["official_commit"],
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step06r2_contract.json", contract)
        _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
        report(
            f"[V6.1 Step 6R2] {status} | hard checks="
            f"{contract['hard_checks_passed']}/{contract['hard_checks_total']}"
        )
        if independent_selected is not None:
            report(
                "[V6.1 Step 6R2] selected="
                f"{independent_selected['candidate_id']} | dNPMI="
                f"{independent_selected['median_validation_npmi_improvement_vs_without_graph']:+.6f} "
                "dC_v="
                f"{independent_selected['median_validation_cv_improvement_vs_without_graph']:+.6f} "
                "full-vs-parent="
                f"{independent_selected['median_full_nll_reduction_vs_official_parent']:+.6f}"
            )
        report(f"[V6.1 Step 6R2] contract={output / 'step06r2_contract.json'}")
        return_zip = _make_return_zip(root, output, log_path)
        report(f"[V6.1 Step 6R2] return-zip={return_zip}")
        _make_return_zip(root, output, log_path)
        if failures:
            raise AssertionError(
                f"Step-6R2 recovery failed: {failures}; return the ZIP and keep test sealed"
            )
        return output
    except Exception:
        if output.exists():
            _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
            _make_return_zip(root, output, log_path)
        raise
