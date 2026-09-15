"""Fail-closed orchestration for V6 Steps 4 and 5."""

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
from scipy import sparse
import yaml

from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_evidence import crossfitted_city_unigram_signal
from src.v6.official_encot import audit_official_source


STEP4_IMPLEMENTATION = "v6_step4_city_residual_grouped_mechanistic_gate_r1"
STEP5_IMPLEMENTATION = "v6_step5_rank_aware_lexical_grouped_mechanistic_gate_r1"
STEP4_SCOPE = "grouped_original_train_city_residual_gate_no_validation_test_or_comparators"
STEP5_SCOPE = "grouped_original_train_rank_aware_lexical_gate_no_validation_test_or_comparators"


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


def _load_protocol(config_path: Path, step: int) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1:
        raise ValueError(f"V6 Step-{step} configuration schema is unsupported")
    protocol = document.get("v6", {}).get(f"step{step}")
    if not isinstance(protocol, dict):
        raise ValueError(f"configuration lacks v6.step{step}")
    expected_scope = STEP4_SCOPE if step == 4 else STEP5_SCOPE
    expected_implementation = STEP4_IMPLEMENTATION if step == 4 else STEP5_IMPLEMENTATION
    if protocol.get("scope") != expected_scope:
        raise ValueError(f"V6 Step-{step} scope changed")
    if protocol.get("implementation_version") != expected_implementation:
        raise ValueError(f"V6 Step-{step} implementation lock changed")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step 4/5 may not create or alter a Python environment")
    return protocol


def _find_passing_contract(
    root: Path,
    relative_parent: str,
    filename: str,
    readiness: str,
) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((root / relative_parent).glob(f"*/{filename}"), reverse=True)
    for path in candidates:
        try:
            contract = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            contract.get("status") == "PASS"
            and contract.get("readiness") == readiness
            and int(contract.get("hard_checks_passed", -1))
            == int(contract.get("hard_checks_total", -2))
        ):
            return path, contract
    raise RuntimeError(f"No passing prerequisite contract found below {relative_parent}")


def _verify_fingerprint(contract: Mapping[str, Any], key: str) -> bool:
    values = dict(contract)
    expected = str(values.pop(key))
    return sha256_object(values) == expected


def _artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if (
            not path.is_file()
            or path.name == "generated_artifact_manifest.json"
            or "checkpoints" in path.relative_to(output).parts
        ):
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


def _make_return_zip(root: Path, output: Path, log_path: Path, step: int) -> Path:
    destination = root / f"v6_step{step:02d}_results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path(f"step{step:02d}_{'city_residual' if step == 4 else 'rank_aware_lexical'}") / output.name
        for path in sorted(output.rglob("*")):
            if not path.is_file() or "checkpoints" in path.relative_to(output).parts:
                continue
            archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, f"logs/v6_step{step:02d}.log")
    return destination


def _runtime_and_source(step3_output: Path, step3_config: Path) -> tuple[Path, Path, dict[str, Any]]:
    selection = json.loads(
        (step3_output / "runtime_selection.json").read_text(encoding="utf-8-sig")
    )
    receipt = json.loads(
        (step3_output / "official_source_receipt.json").read_text(encoding="utf-8-sig")
    )
    specification = yaml.safe_load(step3_config.read_text(encoding="utf-8-sig"))["v6"]["step3"]
    source = Path(receipt["source_directory"])
    runtime = Path(selection["selected_python"])
    if not source.is_dir() or not runtime.is_file():
        raise FileNotFoundError(
            "The exact source/runtime recorded by passing Step 3 is no longer present; rerun Step 3"
        )
    current_receipt = audit_official_source(source, specification["official_source"])
    if current_receipt["commit"] != receipt["commit"] or current_receipt["git_tree"] != receipt["git_tree"]:
        raise RuntimeError("official EnCOT identity changed after Step 3")
    return runtime, source, specification


def _find_step2(root: Path, expected_hash: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (root / "outputs" / "v6" / "step02_evaluation_firewall").glob(
            "*/step02_contract.json"
        ),
        reverse=True,
    )
    for path in candidates:
        if sha256_file(path) != str(expected_hash):
            continue
        contract = json.loads(path.read_text(encoding="utf-8-sig"))
        if contract.get("status") == "PASS":
            return path, contract
    raise RuntimeError("the exact Step-2 prerequisite used by Step 3 is unavailable")


def _worker_inputs(step3_output: Path, step2_output: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "train_counts": step3_output / "worker_input" / "train_counts.npz",
        "train_embeddings": step3_output / "worker_input" / "train_embeddings.npz",
        "word_embeddings": step3_output / "worker_input" / "word_embeddings.npz",
        "fold_assignments": step2_output / "training_group_folds.csv",
        "completion_rows": step2_output / "train_completion_rows.csv",
        "completion_observed": step2_output / "train_completion_observed.npz",
        "completion_target": step2_output / "train_completion_target.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required frozen development evidence is missing: " + "; ".join(missing))
    return {
        role: {"path": str(path), "sha256": sha256_file(path)}
        for role, path in paths.items()
    }


def _verify_frozen_input_artifacts(
    inputs: Mapping[str, Mapping[str, Any]],
    step3_output: Path,
    step2_output: Path,
    step2_contract: Mapping[str, Any],
) -> None:
    step3_manifest = json.loads(
        (step3_output / "training_only_input_manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )
    expected_step3 = step3_manifest["files"]
    for role in ("train_counts", "train_embeddings", "word_embeddings"):
        filename = Path(inputs[role]["path"]).name
        if str(inputs[role]["sha256"]) != str(expected_step3.get(filename)):
            raise RuntimeError(f"the passing Step-3 input changed: {filename}")

    step2_manifest_path = step2_output / "generated_evaluation_artifact_manifest.json"
    if sha256_file(step2_manifest_path) != str(
        step2_contract["generated_artifact_manifest_sha256"]
    ):
        raise RuntimeError("the Step-2 generated-evidence manifest changed")
    expected_step2 = {
        str(row["filename"]): str(row["sha256"])
        for row in json.loads(step2_manifest_path.read_text(encoding="utf-8-sig"))
    }
    for role in (
        "fold_assignments",
        "completion_rows",
        "completion_observed",
        "completion_target",
    ):
        filename = Path(inputs[role]["path"]).name
        if str(inputs[role]["sha256"]) != str(expected_step2.get(filename)):
            raise RuntimeError(f"the passing Step-2 evidence changed: {filename}")


def _stream_worker(
    runtime: Path,
    root: Path,
    job_path: Path,
    reporter: _Reporter,
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.development_worker", "--job", str(job_path)],
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


def _base_prerequisites(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, Path, dict[str, Any]]:
    step3_path, step3_contract = _find_passing_contract(
        root,
        "outputs/v6/step03_official_encot",
        "step03_contract.json",
        "READY_FOR_V6_STEP4_CITY_RESIDUAL_INTEGRATION",
    )
    if not _verify_fingerprint(step3_contract, "contract_fingerprint"):
        raise RuntimeError("Step-3 contract fingerprint does not verify")
    prerequisite = protocol.get("step3_prerequisite", protocol["prerequisite"])
    if any(
        (
            step3_contract.get("implementation_version") != prerequisite["implementation_version"],
            step3_contract.get("data_key") != prerequisite["data_key"],
            step3_contract.get("official_commit") != prerequisite["official_commit"],
        )
    ):
        raise RuntimeError("Step-3 contract does not match the Step-4/5 prerequisite lock")
    step3_output = step3_path.parent
    step2_path, step2_contract = _find_step2(root, step3_contract["step2_contract_sha256"])
    step3_config = root / "configs" / "v6" / "step03.yaml"
    if sha256_file(step3_config) != step3_contract["config_sha256"]:
        raise RuntimeError("Step-3 configuration changed after its passing run")
    runtime, source, step3_specification = _runtime_and_source(step3_output, step3_config)
    return (
        step3_path,
        step3_contract,
        step2_path,
        step2_contract,
        runtime,
        source,
        step3_specification,
    )


def _validate_runtime(worker_output: Path, step3_specification: Mapping[str, Any]) -> bool:
    receipt = json.loads(
        (worker_output / "runtime_receipt.json").read_text(encoding="utf-8-sig")
    )
    expected = step3_specification["runtime"]["packages"]
    return bool(
        receipt["cuda_available"]
        and receipt["environment_created"] is False
        and receipt["packages_installed_or_changed"] is False
        and all(
            str(receipt["packages"].get(name, "")).split("+", 1)[0] == str(version)
            for name, version in expected.items()
        )
    )


def run_v6_step4(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError("V6 Step 4 requires an isolated directory whose name contains 'V6'")
    protocol = _load_protocol(config_path, 4)
    run_id = _utc_id()
    output = root / "outputs" / "v6" / "step04_city_residual" / run_id
    output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"v6_step04_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6 Step 4] START | implementation={STEP4_IMPLEMENTATION}")
    report("[V6 Step 4] scope=original-train grouped folds | validation=0 test=0 comparators=0")
    return_zip = root / "v6_step04_results.zip"
    try:
        (
            step3_path,
            step3_contract,
            step2_path,
            step2_contract,
            runtime,
            source,
            step3_specification,
        ) = _base_prerequisites(root, protocol)
        inputs = _worker_inputs(step3_path.parent, step2_path.parent)
        _verify_frozen_input_artifacts(
            inputs, step3_path.parent, step2_path.parent, step2_contract
        )

        counts = sparse.load_npz(inputs["train_counts"]["path"]).tocsr()
        folds = (
            pd.read_csv(inputs["fold_assignments"]["path"], encoding="utf-8-sig")
            .sort_values("matrix_row")
            .reset_index(drop=True)
        )
        if folds["matrix_row"].astype(int).tolist() != list(range(counts.shape[0])):
            raise RuntimeError("Step-2 folds are not aligned to Step-3 training counts")
        initializer = protocol["city_residual"]["initializer"]
        selected_shrinkage = float(initializer["shrinkage"])
        candidate_rows: list[dict[str, Any]] = []
        candidate_summaries: list[dict[str, Any]] = []
        for candidate in initializer[
            "development_candidate_shrinkages_reviewed_before_neural_quality"
        ]:
            rows, summary = crossfitted_city_unigram_signal(
                counts,
                folds["city"].astype(str).tolist(),
                folds["fold"].to_numpy(dtype=np.int64),
                shrinkage=float(candidate),
                pseudocount=float(initializer["pseudocount"]),
            )
            candidate_rows.extend(rows)
            candidate_summaries.append(summary)
        selected = [
            summary
            for summary in candidate_summaries
            if float(summary["shrinkage"]) == selected_shrinkage
        ]
        if len(selected) != 1:
            raise RuntimeError("the selected city shrinkage is absent from its development grid")
        preflight = dict(selected[0])
        preflight["candidate_summaries"] = candidate_summaries
        preflight["selected_before_neural_v6_quality"] = True
        preflight["selected_is_best_candidate_by_frozen_metric"] = bool(
            preflight["mean_reduction"]
            == max(row["mean_reduction"] for row in candidate_summaries)
        )
        preflight_rows = candidate_rows
        _write_csv(output / "model_free_city_signal_by_fold.csv", preflight_rows)
        _write_json(output / "model_free_city_signal_summary.json", preflight)
        report(
            "[V6 Step 4] model-free preflight | "
            f"mean partial-pooling NLL reduction={preflight['mean_reduction']:.6f} "
            f"positive-folds={preflight['positive_fold_fraction']:.2f}"
        )

        worker_output = output / "worker"
        job = {
            "schema_version": 1,
            "mode": "step4",
            "worker_output": str(worker_output),
            "official_source": str(source),
            "official_commit": step3_contract["official_commit"],
            "inputs": inputs,
            "cities": protocol["cities"],
            "development_folds": protocol["development_folds"],
            "global_context": protocol["global_context"],
            "model": protocol["model"],
            "training": protocol["training"],
            "city_residual": protocol["city_residual"],
            "evaluation": protocol["evaluation"],
        }
        job_path = output / "worker_job.json"
        _write_json(job_path, job)
        code, transcript = _stream_worker(runtime, root, job_path, report)
        if code != 0:
            _write_json(
                output / "worker_failure.json",
                {"schema_version": 1, "return_code": code, "transcript": transcript},
            )
            _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
            _make_return_zip(root, output, log_path, 4)
            raise RuntimeError("Step-4 exact-runtime worker failed; return v6_step04_results.zip")
        worker = json.loads(
            (worker_output / "worker_result.json").read_text(encoding="utf-8-sig")
        )
        per_fold = pd.read_csv(worker_output / "fold_metrics.csv", encoding="utf-8-sig")
        gates = protocol["gates"]
        checks: list[tuple[str, bool, Any]] = [
            ("step3_contract_fingerprint", True, True),
            ("step3_parent_identity_passed", bool(step3_contract["exact_extensions_off_parent_identity_passed"]), step3_contract["exact_extensions_off_parent_identity_passed"]),
            ("step2_data_key_exact", step2_contract["data_key"] == prerequisite_value(protocol, "data_key"), step2_contract["data_key"]),
            ("exact_runtime_unchanged", _validate_runtime(worker_output, step3_specification), str(runtime)),
            ("official_source_unchanged", worker["official_class_loaded_from"].casefold().startswith(str(source).casefold()), worker["official_class_loaded_from"]),
            ("model_free_shrinkage_selection_reproduced", bool(preflight["selected_is_best_candidate_by_frozen_metric"]), preflight["selected_is_best_candidate_by_frozen_metric"]),
            ("model_free_partial_pooling_signal", preflight["median_reduction"] >= float(gates["minimum_model_free_signal"]), preflight["median_reduction"]),
            ("parent_training_loss_decreased_every_fold", bool((per_fold["parent_late_window_mean_loss"] < per_fold["parent_early_window_mean_loss"]).all()), float(worker["mean_parent_loss_decrease"])),
            ("adapter_reconstruction_decreased_every_fold", bool((per_fold["adapter_late_window_mean_reconstruction"] < per_fold["adapter_early_window_mean_reconstruction"]).all()), float(worker["mean_adapter_reconstruction_decrease"])),
            ("alpha_zero_exact_parent_prediction", float(worker["parent_identity_maximum_error"]) <= float(gates["identity_tolerance"]), worker["parent_identity_maximum_error"]),
            ("city_residual_nonzero_every_fold", bool((per_fold["effective_residual_rms"] > float(gates["activity_floor"])).all()), worker["minimum_effective_residual_rms"]),
            ("weighted_city_centering", float(worker["maximum_weighted_city_center_error"]) <= float(gates["centering_tolerance"]), worker["maximum_weighted_city_center_error"]),
            ("topic_null_direction_removed", float(worker["maximum_topic_null_direction_error"]) <= float(gates["centering_tolerance"]), worker["maximum_topic_null_direction_error"]),
            ("lower_capacity_than_independent_city_affinities", int(worker["parameter_counts"]["saving"]) > 0, worker["parameter_counts"]),
            ("paired_city_nll_practical_effect", float(worker["median_nll_reduction"]) >= float(gates["minimum_paired_nll_reduction"]), worker["median_nll_reduction"]),
            ("paired_city_nll_fold_consistency", float(worker["positive_fold_fraction"]) >= float(gates["minimum_positive_fold_fraction"]), worker["positive_fold_fraction"]),
            ("all_checkpoints_present_and_hashed", all(Path(row["path"]).is_file() and sha256_file(Path(row["path"])) == row["sha256"] for row in worker["checkpoint_manifest"]), len(worker["checkpoint_manifest"])),
            ("validation_access_zero", worker["validation_documents_used"] == 0, worker["validation_documents_used"]),
            ("test_access_zero", worker["test_documents_used"] == 0, worker["test_documents_used"]),
            ("label_access_zero", worker["labels_used"] == 0, worker["labels_used"]),
            ("comparator_access_zero", worker["baseline_or_sota_outputs_used"] == 0, worker["baseline_or_sota_outputs_used"]),
        ]
        hard_rows = [
            {"check": name, "required": True, "observed": observed, "status": "PASS" if passed else "FAIL"}
            for name, passed, observed in checks
        ]
        _write_csv(output / "hard_checks.csv", hard_rows)
        failures = [row["check"] for row in hard_rows if row["status"] == "FAIL"]
        status = "PASS" if not failures else "FAIL"
        resolved = output / "resolved_v6_step04.yaml"
        resolved.write_text(
            yaml.safe_dump({"schema_version": 1, "v6": {"step4": protocol}}, sort_keys=False),
            encoding="utf-8",
        )
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": STEP4_SCOPE,
            "implementation_version": STEP4_IMPLEMENTATION,
            "readiness": "READY_FOR_V6_STEP5_RANK_AWARE_LEXICAL_INTEGRATION" if status == "PASS" else "BLOCKED_REVISE_CITY_MODULE_BEFORE_STEP5",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "data_key": step3_contract["data_key"],
            "step3_contract_sha256": sha256_file(step3_path),
            "step2_contract_sha256": sha256_file(step2_path),
            "official_commit": step3_contract["official_commit"],
            "development_folds": len(protocol["development_folds"]),
            "staged_model_fits": worker["staged_model_fits"],
            "median_city_nll_reduction": worker["median_nll_reduction"],
            "mean_city_nll_reduction": worker["mean_nll_reduction"],
            "positive_city_nll_fold_fraction": worker["positive_fold_fraction"],
            "model_free_preflight_is_not_model_result": True,
            "development_hyperparameters_selected_from_model_free_training_signal": 1,
            "validation_documents_used": 0,
            "test_documents_used": 0,
            "human_labels_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step04_contract.json", contract)
        _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
        report(
            f"[V6 Step 4] {status} | hard checks={contract['hard_checks_passed']}/{contract['hard_checks_total']} "
            f"median paired macro-city NLL reduction={worker['median_nll_reduction']:.6f}"
        )
        report(f"[V6 Step 4] contract={output / 'step04_contract.json'}")
        report(f"[V6 Step 4] return-zip={return_zip}")
        _make_return_zip(root, output, log_path, 4)
        if failures:
            raise AssertionError(f"Step 4 failed gates: {failures}; return v6_step04_results.zip")
        return output
    except Exception:
        if output.exists():
            _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
            _make_return_zip(root, output, log_path, 4)
        raise


def prerequisite_value(protocol: Mapping[str, Any], key: str) -> Any:
    return protocol["prerequisite"][key]


def run_v6_step5(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError("V6 Step 5 requires an isolated directory whose name contains 'V6'")
    protocol = _load_protocol(config_path, 5)
    run_id = _utc_id()
    output = root / "outputs" / "v6" / "step05_rank_aware_lexical" / run_id
    output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"v6_step05_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6 Step 5] START | implementation={STEP5_IMPLEMENTATION}")
    report("[V6 Step 5] scope=original-train grouped folds | validation=0 test=0 comparators=0")
    return_zip = root / "v6_step05_results.zip"
    try:
        step4_path, step4_contract = _find_passing_contract(
            root,
            "outputs/v6/step04_city_residual",
            "step04_contract.json",
            "READY_FOR_V6_STEP5_RANK_AWARE_LEXICAL_INTEGRATION",
        )
        if not _verify_fingerprint(step4_contract, "contract_fingerprint"):
            raise RuntimeError("Step-4 contract fingerprint does not verify")
        if step4_contract["implementation_version"] != protocol["prerequisite"]["implementation_version"]:
            raise RuntimeError("Step-4 implementation does not match Step-5 lock")
        if step4_contract["data_key"] != protocol["prerequisite"]["data_key"]:
            raise RuntimeError("Step-4 data identity does not match Step-5 lock")
        (
            step3_path,
            step3_contract,
            step2_path,
            step2_contract,
            runtime,
            source,
            step3_specification,
        ) = _base_prerequisites(root, protocol)
        if sha256_file(step3_path) != step4_contract["step3_contract_sha256"]:
            raise RuntimeError("Step 5 found a different Step-3 prerequisite")
        step4_worker = step4_path.parent / "worker"
        checkpoint_manifest = json.loads(
            (step4_worker / "checkpoint_manifest.json").read_text(encoding="utf-8-sig")
        )
        if not all(
            Path(row["path"]).is_file()
            and sha256_file(Path(row["path"])) == str(row["sha256"])
            for row in checkpoint_manifest
        ):
            raise RuntimeError("a required Step-4 checkpoint is missing or changed")
        inputs = _worker_inputs(step3_path.parent, step2_path.parent)
        _verify_frozen_input_artifacts(
            inputs, step3_path.parent, step2_path.parent, step2_contract
        )
        worker_output = output / "worker"
        job = {
            "schema_version": 1,
            "mode": "step5",
            "worker_output": str(worker_output),
            "official_source": str(source),
            "official_commit": step3_contract["official_commit"],
            "inputs": inputs,
            "cities": protocol["cities"],
            "development_folds": protocol["development_folds"],
            "global_context": protocol["global_context"],
            "model": protocol["model"],
            "training": protocol["training"],
            "city_residual": protocol["city_residual"],
            "lexical": protocol["lexical"],
            "evaluation": protocol["evaluation"],
            "step4_checkpoints": checkpoint_manifest,
        }
        job_path = output / "worker_job.json"
        _write_json(job_path, job)
        code, transcript = _stream_worker(runtime, root, job_path, report)
        if code != 0:
            _write_json(output / "worker_failure.json", {"schema_version": 1, "return_code": code, "transcript": transcript})
            _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
            _make_return_zip(root, output, log_path, 5)
            raise RuntimeError("Step-5 exact-runtime worker failed; return v6_step05_results.zip")
        worker = json.loads((worker_output / "worker_result.json").read_text(encoding="utf-8-sig"))
        folds = pd.read_csv(worker_output / "paired_fold_differences.csv", encoding="utf-8-sig")
        gates = protocol["gates"]
        checks: list[tuple[str, bool, Any]] = [
            ("step4_contract_fingerprint", True, True),
            ("exact_runtime_unchanged", _validate_runtime(worker_output, step3_specification), str(runtime)),
            ("official_source_unchanged", worker["official_class_loaded_from"].casefold().startswith(str(source).casefold()), worker["official_class_loaded_from"]),
            ("paired_initial_states_exact", bool(worker["all_initial_states_exactly_matched"]), worker["all_initial_states_exactly_matched"]),
            ("fold_local_graphs_complete", len(json.loads((worker_output / "fold_graph_receipts.json").read_text())) == len(protocol["development_folds"]), len(protocol["development_folds"])),
            ("rank_6_to_10_gradients_active", float(worker["minimum_rank_6_to_10_positive_fraction"]) >= 1.0 and float(worker["minimum_rank_6_to_10_gradient"]) > float(gates["gradient_activity_floor"]), worker["minimum_rank_6_to_10_gradient"]),
            ("paired_npmi_practical_effect", float(worker["median_npmi_improvement"]) >= float(gates["minimum_paired_npmi_improvement"]), worker["median_npmi_improvement"]),
            ("paired_npmi_fold_consistency", float(worker["positive_npmi_fold_fraction"]) >= float(gates["minimum_positive_fold_fraction"]), worker["positive_npmi_fold_fraction"]),
            ("c_v_noninferiority", float(worker["median_c_v_difference"]) >= -float(gates["c_v_noninferiority_margin"]), worker["median_c_v_difference"]),
            ("completion_nll_guardrail", float(worker["median_nll_reduction"]) >= -float(gates["maximum_nll_regression"]), worker["median_nll_reduction"]),
            ("topic_diversity_guardrail", float(worker["minimum_full_topic_diversity"]) >= float(gates["minimum_topic_diversity"]), worker["minimum_full_topic_diversity"]),
            ("redundancy_guardrail", float(worker["maximum_full_top_word_redundancy"]) <= float(gates["maximum_top_word_redundancy"]), worker["maximum_full_top_word_redundancy"]),
            ("topic_stability_guardrail", float(worker["full_topic_stability"]["mean"]) >= float(gates["minimum_topic_stability"]), worker["full_topic_stability"]),
            ("all_checkpoints_present_and_hashed", all(Path(row["path"]).is_file() and sha256_file(Path(row["path"])) == row["sha256"] for row in worker["checkpoint_manifest"]), len(worker["checkpoint_manifest"])),
            ("all_training_metrics_finite", bool(np.isfinite(folds.select_dtypes(include=[np.number]).to_numpy()).all()), True),
            ("validation_access_zero", worker["validation_documents_used"] == 0, worker["validation_documents_used"]),
            ("test_access_zero", worker["test_documents_used"] == 0, worker["test_documents_used"]),
            ("label_access_zero", worker["labels_used"] == 0, worker["labels_used"]),
            ("comparator_access_zero", worker["baseline_or_sota_outputs_used"] == 0, worker["baseline_or_sota_outputs_used"]),
        ]
        hard_rows = [{"check": name, "required": True, "observed": observed, "status": "PASS" if passed else "FAIL"} for name, passed, observed in checks]
        _write_csv(output / "hard_checks.csv", hard_rows)
        failures = [row["check"] for row in hard_rows if row["status"] == "FAIL"]
        status = "PASS" if not failures else "FAIL"
        resolved = output / "resolved_v6_step05.yaml"
        resolved.write_text(yaml.safe_dump({"schema_version": 1, "v6": {"step5": protocol}}, sort_keys=False), encoding="utf-8")
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": STEP5_SCOPE,
            "implementation_version": STEP5_IMPLEMENTATION,
            "readiness": "READY_FOR_V6_STEP6_TWELVE_FIT_KILL_SCREEN" if status == "PASS" else "BLOCKED_REVISE_LEXICAL_MODULE_BEFORE_STEP6",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "data_key": step4_contract["data_key"],
            "step4_contract_sha256": sha256_file(step4_path),
            "step3_contract_sha256": sha256_file(step3_path),
            "paired_continuation_fits": worker["paired_continuation_fits"],
            "median_npmi_improvement": worker["median_npmi_improvement"],
            "median_c_v_difference": worker["median_c_v_difference"],
            "median_nll_reduction": worker["median_nll_reduction"],
            "validation_documents_used": 0,
            "test_documents_used": 0,
            "human_labels_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step05_contract.json", contract)
        _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
        report(
            f"[V6 Step 5] {status} | hard checks={contract['hard_checks_passed']}/{contract['hard_checks_total']} "
            f"median paired NPMI improvement={worker['median_npmi_improvement']:.6f}"
        )
        report(f"[V6 Step 5] contract={output / 'step05_contract.json'}")
        report(f"[V6 Step 5] return-zip={return_zip}")
        _make_return_zip(root, output, log_path, 5)
        if failures:
            raise AssertionError(f"Step 5 failed gates: {failures}; return v6_step05_results.zip")
        return output
    except Exception:
        if output.exists():
            _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
            _make_return_zip(root, output, log_path, 5)
        raise
