"""Fail-closed orchestration for the V6.1 final refit and pre-test freeze."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_stage import _verify_fingerprint
from src.v6.final_refit import FINAL_VARIANTS, validate_final_diagnostics
from src.v6.official_encot import audit_official_source
from src.v6.step4r2 import _verify_manifest

IMPLEMENTATION = "v6_1_step7_final_development_refit_freeze_r1"
SCOPE = "final_train_plus_spent_development_refit_pretest_freeze_no_test_labels_or_comparators"
PREREQUISITE_IMPLEMENTATION = "v6_1_step6r2_spent_validation_development_recovery_r1"
PREREQUISITE_READINESS = "READY_FOR_V6_1_FINAL_REFIT_AND_ONE_SHOT_TEST"
EXPECTED_DATA_KEY = "7774658044d3e297ed5bff31a45bcad5221228a53fe48fd5aabacebf12778c2d"
EXPECTED_OFFICIAL_COMMIT = "8ac3592165cc7851676be2776b314eb6e44e9388"
EXPECTED_CONFIGURATION_FINGERPRINT = (
    "73904363114be2afb42067bcc6103e45a1b3a1ed77492a8a531301ce8ff52403"
)
EXPECTED_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]


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
    protocol = document.get("v6", {}).get("step7", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("unsupported V6.1 Step-7 configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-7 implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-7 scope changed")
    prerequisite = protocol.get("prerequisite", {})
    if any(
        (
            prerequisite.get("implementation_version") != PREREQUISITE_IMPLEMENTATION,
            prerequisite.get("readiness") != PREREQUISITE_READINESS,
            prerequisite.get("data_key") != EXPECTED_DATA_KEY,
            prerequisite.get("official_commit") != EXPECTED_OFFICIAL_COMMIT,
            prerequisite.get("configuration_fingerprint")
            != EXPECTED_CONFIGURATION_FINGERPRINT,
        )
    ):
        raise ValueError("Step-7 Step-6R2 prerequisite lock changed")
    selected = protocol.get("selected_configuration", {})
    if any(
        (
            int(selected.get("prototype_quota", -1)) != 10,
            selected.get("calibration") != "training_moment_match",
            float(selected.get("pooled_mixture_weight", float("nan"))) != 0.35,
            selected.get("city_specific_backoff") != "RETIRED",
        )
    ):
        raise ValueError("Step-7 selected V6.1 configuration changed")
    training = protocol.get("training", {})
    if any(
        (
            list(map(int, training.get("seeds", []))) != EXPECTED_SEEDS,
            int(training.get("parent_epochs", -1)) != 120,
            float(training.get("parent_learning_rate", float("nan"))) != 0.002,
            int(training.get("batch_size", -1)) != 200,
            training.get("stopping_rule") != "fixed_epochs_only",
            training.get("validation_early_stopping") is not False,
            training.get("seed_selection") is not False,
            training.get("hyperparameter_selection") is not False,
        )
    ):
        raise ValueError("Step-7 final training policy changed")
    final_fit = protocol.get("final_fit", {})
    if any(
        (
            int(final_fit.get("original_training_documents", -1)) != 620,
            int(final_fit.get("spent_development_documents", -1)) != 75,
            int(final_fit.get("documents", -1)) != 695,
            int(final_fit.get("tokens", -1)) != 6944,
            int(final_fit.get("vocabulary", -1)) != 1148,
        )
    ):
        raise ValueError("Step-7 final-fit partition changed")
    if list(protocol.get("ablations", {}).get("variants", [])) != list(FINAL_VARIANTS):
        raise ValueError("Step-7 paired ablation registry changed")
    policy = protocol.get("data_policy", {})
    if any(
        policy.get(key) is not False
        for key in (
            "test_available_to_worker",
            "labels_available_to_worker",
            "comparators_available_to_worker",
        )
    ):
        raise ValueError("Step-7 prohibited-data policy changed")
    registration = protocol.get("confirmation_preregistration", {})
    registration_path = config_path.resolve().parents[2] / str(
        registration.get("path", "")
    )
    if (
        registration.get("frozen_before_test") is not True
        or not registration_path.is_file()
        or sha256_file(registration_path) != str(registration.get("sha256", ""))
    ):
        raise ValueError("Step-8/9 confirmation preregistration changed")
    receipt = json.loads(registration_path.read_text(encoding="utf-8-sig"))
    if any(
        (
            receipt.get("registration_status") != "FROZEN_BEFORE_ANY_V6_1_TEST_ACCESS",
            receipt.get("final_seeds") != EXPECTED_SEEDS,
            receipt.get("full_and_ablation_variants") != list(FINAL_VARIANTS),
            receipt.get("failure_policy")
            != "REPORT_WITHOUT_TEST_CONDITIONED_MODEL_CHANGE",
        )
    ):
        raise ValueError("Step-8/9 confirmation preregistration content changed")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step 7 may not create or alter a Python environment")
    return protocol


def _artifact_manifest(output: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
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


def _make_return_zip(root: Path, output: Path, log_path: Path) -> Path:
    destination = root / "v6_step07_results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("step07_final_refit_freeze") / output.name
        for path in sorted(output.rglob("*")):
            if not path.is_file() or "checkpoints" in path.relative_to(output).parts:
                continue
            archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, "logs/v6_step07.log")
    return destination


def _find_passing_step6r2(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any], Path, Path]:
    candidates = sorted(
        (root / "outputs" / "v6" / "step06r2_development_recovery").glob(
            "*/step06r2_contract.json"
        ),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        output = contract_path.parent
        selected = contract.get("selected_configuration", {})
        if any(
            (
                contract.get("status") != "PASS",
                contract.get("readiness") != PREREQUISITE_READINESS,
                contract.get("implementation_version") != PREREQUISITE_IMPLEMENTATION,
                contract.get("data_key") != protocol["prerequisite"]["data_key"],
                contract.get("official_commit")
                != protocol["prerequisite"]["official_commit"],
                selected.get("configuration_fingerprint")
                != protocol["prerequisite"]["configuration_fingerprint"],
                contract.get("validation_status") != "SPENT_DEVELOPMENT",
                contract.get("validation_confirmation_claim_allowed") is not False,
                int(contract.get("test_documents_used", -1)) != 0,
                int(contract.get("baseline_outputs_used", -1)) != 0,
                int(contract.get("sota_outputs_used", -1)) != 0,
                not _verify_fingerprint(contract, "contract_fingerprint"),
                not _verify_manifest(output),
            )
        ):
            continue
        job_path = output / "worker_job.json"
        runtime_path = output / "worker" / "runtime_receipt.json"
        selected_path = output / "worker" / "selected_configuration.json"
        if not all(path.is_file() for path in (job_path, runtime_path, selected_path)):
            continue
        lineage = {
            "step6r2_contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "step6r2_worker_job": {
                "path": str(job_path),
                "sha256": sha256_file(job_path),
            },
            "step6r2_runtime_receipt": {
                "path": str(runtime_path),
                "sha256": sha256_file(runtime_path),
            },
            "step6r2_selected_configuration": {
                "path": str(selected_path),
                "sha256": sha256_file(selected_path),
            },
            "step6r2_manifest": {
                "path": str(output / "generated_artifact_manifest.json"),
                "sha256": sha256_file(output / "generated_artifact_manifest.json"),
            },
        }
        return contract_path, contract, lineage, job_path, runtime_path
    raise RuntimeError("No intact passing Step-6R2 result was found")


def _source_input(
    step2_output: Path,
    step2_contract: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    hashes = pd.read_csv(step2_output / "input_hashes.csv", encoding="utf-8-sig")
    selected = hashes.loc[hashes["role"].astype(str) == role]
    if len(selected) != 1:
        raise RuntimeError(f"Step-2 input role is not unique: {role}")
    row = selected.iloc[0]
    source = Path(str(step2_contract["source_project_root"])) / Path(
        str(row["relative_path"])
    )
    if not source.is_file() or sha256_file(source) != str(row["sha256"]):
        raise RuntimeError(f"the frozen source artifact changed: {role}")
    return {"path": str(source), "sha256": str(row["sha256"])}


def _development_inputs(
    step6r2_job: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    failed_job_item = step6r2_job["lineage_artifacts"]["failed_step6_worker_job"]
    failed_job_path = Path(failed_job_item["path"])
    if not failed_job_path.is_file() or sha256_file(failed_job_path) != str(
        failed_job_item["sha256"]
    ):
        raise RuntimeError("the failed Step-6 worker-job lineage changed")
    failed_job = json.loads(failed_job_path.read_text(encoding="utf-8-sig"))
    failed_contract_item = step6r2_job["lineage_artifacts"]["failed_step6_contract"]
    failed_contract_path = Path(failed_contract_item["path"])
    if not failed_contract_path.is_file() or sha256_file(failed_contract_path) != str(
        failed_contract_item["sha256"]
    ):
        raise RuntimeError("the failed Step-6 contract lineage changed")
    failed_contract = json.loads(failed_contract_path.read_text(encoding="utf-8-sig"))
    step2_output = Path(failed_job["training_inputs"]["train_rows"]["path"]).parent
    step2_contract_path = step2_output / "step02_contract.json"
    if not step2_contract_path.is_file() or sha256_file(step2_contract_path) != str(
        failed_contract["step2_contract_sha256"]
    ):
        raise RuntimeError("the exact Step-2 contract used by Step 6 is unavailable")
    step2_contract = json.loads(step2_contract_path.read_text(encoding="utf-8-sig"))
    inputs = {
        "train_counts": dict(failed_job["training_inputs"]["train_counts"]),
        "train_rows": dict(failed_job["training_inputs"]["train_rows"]),
        "word_embeddings": dict(failed_job["training_inputs"]["word_embeddings"]),
        "spent_counts": _source_input(
            step2_output, step2_contract, "validation_counts"
        ),
        "spent_rows": _source_input(step2_output, step2_contract, "validation_rows"),
    }
    for role, item in inputs.items():
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != str(item["sha256"]):
            raise RuntimeError(f"final-refit input changed: {role}")
    source_lineage = {
        "failed_step6_worker_job": {
            "path": str(failed_job_path),
            "sha256": sha256_file(failed_job_path),
        },
        "failed_step6_contract": {
            "path": str(failed_contract_path),
            "sha256": sha256_file(failed_contract_path),
        },
        "step2_contract": {
            "path": str(step2_contract_path),
            "sha256": sha256_file(step2_contract_path),
        },
        "step2_input_hashes": {
            "path": str(step2_output / "input_hashes.csv"),
            "sha256": sha256_file(step2_output / "input_hashes.csv"),
        },
    }
    return inputs, source_lineage


def _stream_worker(
    runtime: Path, root: Path, job_path: Path, reporter: _Reporter
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.step7_worker", "--job", str(job_path)],
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


def _runtime_matches(receipt: Mapping[str, Any], runtime: Path) -> bool:
    return bool(
        Path(receipt["python_executable"]).resolve() == runtime.resolve()
        and receipt.get("cuda_available") is True
        and receipt.get("packages_installed_or_changed") is False
        and receipt.get("environment_created") is False
    )


def _diagnostic_values_match(expected: Any, observed: Any) -> bool:
    """Compare persisted diagnostics without demanding bitwise CSV round trips."""

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
                    rtol=1.0e-12,
                    atol=1.0e-15,
                    equal_nan=False,
                )
            )
        except (TypeError, ValueError):
            return False
    return expected == observed


def _completed_worker_freeze_is_intact(
    output: Path, protocol: Mapping[str, Any]
) -> bool:
    """Recognize a completed Step-7 worker whose outer gate may be resumed."""

    worker_output = output / "worker"
    try:
        if not _verify_manifest(output):
            return False
        worker = json.loads(
            (worker_output / "worker_result.json").read_text(encoding="utf-8-sig")
        )
        job = json.loads((output / "worker_job.json").read_text(encoding="utf-8-sig"))
        checkpoints = json.loads(
            (worker_output / "checkpoint_manifest.json").read_text(encoding="utf-8-sig")
        )
        freeze = json.loads(
            (worker_output / "final_freeze_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        freeze_body = dict(freeze)
        freeze_identity = str(freeze_body.pop("freeze_identity"))
        data_policy = job["data_policy"]
        checkpoint_seeds = [int(row["seed"]) for row in checkpoints]
        checkpoint_files_intact = all(
            Path(row["path"]).is_file()
            and int(Path(row["path"]).stat().st_size) == int(row["size_bytes"])
            and sha256_file(Path(row["path"])) == str(row["sha256"])
            for row in checkpoints
        )
        return bool(
            worker.get("status") == "PASS"
            and worker.get("implementation_version") == IMPLEMENTATION
            and job.get("mode") == "step7"
            and job.get("implementation_version") == IMPLEMENTATION
            and list(map(int, job["training"]["seeds"])) == EXPECTED_SEEDS
            and all(
                data_policy.get(key) is False
                for key in (
                    "test_available_to_worker",
                    "labels_available_to_worker",
                    "comparators_available_to_worker",
                )
            )
            and job.get("confirmation_preregistration_sha256")
            == protocol["confirmation_preregistration"]["sha256"]
            and list(map(int, worker.get("seeds", []))) == EXPECTED_SEEDS
            and int(worker.get("parent_neural_fits", -1)) == len(EXPECTED_SEEDS)
            and worker.get("seed_selection_performed") is False
            and worker.get("validation_early_stopping_performed") is False
            and worker.get("hyperparameter_selection_performed") is False
            and int(worker.get("test_documents_used", -1)) == 0
            and int(worker.get("labels_used", -1)) == 0
            and int(worker.get("baseline_or_sota_outputs_used", -1)) == 0
            and checkpoint_seeds == EXPECTED_SEEDS
            and len(checkpoints) == len(EXPECTED_SEEDS)
            and checkpoint_files_intact
            and worker.get("checkpoint_manifest") == checkpoints
            and freeze.get("checkpoints") == checkpoints
            and freeze.get("seeds") == EXPECTED_SEEDS
            and freeze.get("checkpoint_manifest_sha256")
            == sha256_file(worker_output / "checkpoint_manifest.json")
            and freeze.get("confirmation_preregistration_sha256")
            == protocol["confirmation_preregistration"]["sha256"]
            and int(freeze.get("test_documents_used", -1)) == 0
            and sha256_object(freeze_body) == freeze_identity
            and worker.get("freeze_identity") == freeze_identity
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _find_recoverable_step7_output(
    root: Path, protocol: Mapping[str, Any]
) -> Path | None:
    candidates = sorted(
        (root / "outputs" / "v6" / "step07_final_refit_freeze").glob("*"),
        reverse=True,
    )
    for output in candidates:
        if output.is_dir() and _completed_worker_freeze_is_intact(output, protocol):
            return output
    return None


def _completed_step7_worker_exists(root: Path) -> bool:
    for result_path in (root / "outputs" / "v6" / "step07_final_refit_freeze").glob(
        "*/worker/worker_result.json"
    ):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            result.get("status") == "PASS"
            and result.get("implementation_version") == IMPLEMENTATION
            and int(result.get("parent_neural_fits", -1)) == len(EXPECTED_SEEDS)
        ):
            return True
    return False


def run_v6_step7(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError("V6.1 Step 7 requires a project directory containing 'V6'")
    protocol = _load_protocol(config_path)
    prior_contracts = sorted(
        (root / "outputs" / "v6" / "step07_final_refit_freeze").glob(
            "*/step07_contract.json"
        ),
        reverse=True,
    )
    for prior_path in prior_contracts:
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            prior.get("status") == "PASS"
            and prior.get("readiness") == "READY_FOR_V6_1_STEP8_ONE_SHOT_TEST"
            and "contract_fingerprint" in prior
            and _verify_fingerprint(prior, "contract_fingerprint")
            and _verify_manifest(prior_path.parent)
            and _completed_worker_freeze_is_intact(prior_path.parent, protocol)
        ):
            raise RuntimeError(
                "Step 7 already produced an intact passing pre-test freeze; "
                "do not refit it. Return the existing result instead."
            )
    recoverable_output = _find_recoverable_step7_output(root, protocol)
    if recoverable_output is None and _completed_step7_worker_exists(root):
        raise RuntimeError(
            "A completed ten-seed Step-7 worker exists, but its checkpoint freeze "
            "did not pass recovery integrity. Automatic retraining is refused; "
            "return the existing Step-7 result and inspect the local checkpoints."
        )
    recovery_mode = recoverable_output is not None
    run_id = recoverable_output.name if recoverable_output is not None else _utc_id()
    output = (
        recoverable_output
        if recoverable_output is not None
        else root / "outputs" / "v6" / "step07_final_refit_freeze" / run_id
    )
    if not recovery_mode:
        output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"v6_step07_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6.1 Step 7] START | implementation={IMPLEMENTATION}")
    report(
        "[V6.1 Step 7] scope=695-document final refit -> ten-seed immutable freeze | "
        "test=0 labels=0 comparators=0"
    )
    if recovery_mode:
        report(
            "[V6.1 Step 7] RECOVERY | completed ten-seed worker and checkpoint "
            "freeze verified; optimizer rerun=0"
        )
    try:
        (
            step6r2_path,
            step6r2_contract,
            lineage,
            step6r2_job_path,
            runtime_receipt_path,
        ) = _find_passing_step6r2(root, protocol)
        step6r2_job = json.loads(step6r2_job_path.read_text(encoding="utf-8-sig"))
        selected_path = Path(lineage["step6r2_selected_configuration"]["path"])
        selected = json.loads(selected_path.read_text(encoding="utf-8-sig"))
        if selected != step6r2_contract["selected_configuration"]:
            raise RuntimeError("Step-6R2 selected-configuration copies differ")
        development_inputs, source_lineage = _development_inputs(step6r2_job)
        lineage.update(source_lineage)
        old_runtime_receipt = json.loads(
            runtime_receipt_path.read_text(encoding="utf-8-sig")
        )
        runtime = Path(old_runtime_receipt["python_executable"])
        if (
            not runtime.is_file()
            or old_runtime_receipt.get("packages_installed_or_changed") is not False
        ):
            raise RuntimeError("the exact Step-6R2 runtime is unavailable")
        step3_protocol = yaml.safe_load(
            (root / "configs" / "v6" / "step03.yaml").read_text(encoding="utf-8-sig")
        )["v6"]["step3"]
        source = Path(step6r2_job["official_source"])
        source_receipt = audit_official_source(
            source, step3_protocol["official_source"]
        )
        if (
            source_receipt["commit"] != protocol["prerequisite"]["official_commit"]
            or source_receipt["source_tree_clean"] is not True
        ):
            raise RuntimeError("the official EnCOT source changed")
        if recovery_mode:
            persisted_lineage = json.loads(
                (output / "step6r2_lineage.json").read_text(encoding="utf-8-sig")
            )
            if persisted_lineage != lineage:
                raise RuntimeError(
                    "recoverable Step-7 lineage differs from current lineage"
                )
        else:
            _write_json(output / "step6r2_lineage.json", lineage)
            _write_json(output / "official_source_receipt.json", source_receipt)
        preregistration_path = root / protocol["confirmation_preregistration"]["path"]
        worker_output = output / "worker"
        job = {
            "schema_version": 1,
            "mode": "step7",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output),
            "lineage_artifacts": lineage,
            "development_inputs": development_inputs,
            "official_source": str(source),
            "official_commit": protocol["prerequisite"]["official_commit"],
            "cities": protocol["cities"],
            "data_policy": protocol["data_policy"],
            "final_fit": protocol["final_fit"],
            "selected_configuration": selected,
            "global_context": protocol["global_context"],
            "model": protocol["model"],
            "training": protocol["training"],
            "pooled_backoff": protocol["pooled_backoff"],
            "lexical_graph": protocol["lexical_graph"],
            "graph_projection": protocol["graph_projection"],
            "calibration": protocol["calibration"],
            "evaluation": protocol["evaluation"],
            "confirmation_preregistration_sha256": sha256_file(preregistration_path),
        }
        job_path = output / "worker_job.json"
        if recovery_mode:
            persisted_job = json.loads(job_path.read_text(encoding="utf-8-sig"))
            if persisted_job != job:
                raise RuntimeError(
                    "recoverable Step-7 worker job differs from the frozen protocol"
                )
            report(
                "[V6.1 Step 7] RECOVERY | reusing exact completed worker artifacts; "
                "neural fits executed now=0"
            )
        else:
            _write_json(job_path, job)
            code, transcript = _stream_worker(runtime, root, job_path, report)
            if code != 0:
                _write_json(
                    output / "worker_failure.json",
                    {
                        "schema_version": 1,
                        "return_code": code,
                        "transcript": transcript,
                    },
                )
                raise RuntimeError("Step-7 exact-runtime worker failed")

        worker = json.loads(
            (worker_output / "worker_result.json").read_text(encoding="utf-8-sig")
        )
        diagnostics = pd.read_csv(
            worker_output / "final_seed_diagnostics.csv", encoding="utf-8-sig"
        )
        independent = validate_final_diagnostics(
            diagnostics.to_dict(orient="records"), expected_seeds=EXPECTED_SEEDS
        )
        worker_summary = json.loads(
            (worker_output / "final_diagnostic_summary.json").read_text(
                encoding="utf-8-sig"
            )
        )
        for key, value in independent.items():
            if not _diagnostic_values_match(value, worker_summary.get(key)):
                raise RuntimeError(f"final diagnostic summary did not reproduce: {key}")
        freeze = json.loads(
            (worker_output / "final_freeze_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        variants = json.loads(
            (worker_output / "final_variant_manifest.json").read_text(
                encoding="utf-8-sig"
            )
        )
        variant_body = dict(variants)
        variant_identity = str(variant_body.pop("variant_manifest_identity"))
        checkpoints = worker["checkpoint_manifest"]
        current_runtime = json.loads(
            (worker_output / "runtime_receipt.json").read_text(encoding="utf-8-sig")
        )
        graph_receipt = json.loads(
            (worker_output / "final_graph_receipt.json").read_text(encoding="utf-8-sig")
        )
        merge_receipt = json.loads(
            (worker_output / "development_merge_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        pooled_receipt = json.loads(
            (worker_output / "pooled_prior_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        checks: list[tuple[str, bool, Any]] = [
            (
                "step6r2_contract_and_manifest_verified",
                sha256_file(step6r2_path) == lineage["step6r2_contract"]["sha256"]
                and _verify_manifest(step6r2_path.parent),
                lineage["step6r2_contract"]["sha256"],
            ),
            (
                "selected_configuration_exact",
                selected.get("configuration_fingerprint")
                == EXPECTED_CONFIGURATION_FINGERPRINT
                and selected.get("test_status") == "UNOPENED",
                selected,
            ),
            (
                "spent_validation_disclosure_preserved",
                step6r2_contract["validation_status"] == "SPENT_DEVELOPMENT"
                and step6r2_contract["validation_confirmation_claim_allowed"] is False,
                step6r2_contract["validation_status"],
            ),
            (
                "confirmation_preregistration_frozen",
                sha256_file(preregistration_path)
                == protocol["confirmation_preregistration"]["sha256"],
                protocol["confirmation_preregistration"]["sha256"],
            ),
            (
                "exact_runtime_reused_without_install",
                _runtime_matches(current_runtime, runtime),
                current_runtime["python_executable"],
            ),
            (
                "official_source_commit_unchanged",
                source_receipt["commit"] == EXPECTED_OFFICIAL_COMMIT
                and source_receipt["source_tree_clean"] is True,
                source_receipt["commit"],
            ),
            (
                "final_fit_partition_exact",
                int(worker["final_fit_documents"]) == 695
                and int(merge_receipt["original_training_documents"]) == 620
                and int(merge_receipt["spent_development_documents"]) == 75
                and int(merge_receipt["final_fit_tokens"]) == 6944,
                merge_receipt,
            ),
            (
                "ten_predeclared_parent_fits",
                int(worker["parent_neural_fits"]) == 10
                and diagnostics["seed"].astype(int).tolist() == EXPECTED_SEEDS,
                diagnostics["seed"].astype(int).tolist(),
            ),
            (
                "no_seed_or_hyperparameter_selection",
                worker["seed_selection_performed"] is False
                and worker["hyperparameter_selection_performed"] is False,
                False,
            ),
            (
                "fixed_epoch_training_without_validation_stopping",
                worker["validation_early_stopping_performed"] is False
                and int(protocol["training"]["parent_epochs"]) == 120,
                protocol["training"]["stopping_rule"],
            ),
            (
                "parent_training_loss_decreased_every_seed",
                independent["all_parent_losses_decreased"],
                diagnostics["parent_training_loss_reduction"].astype(float).min(),
            ),
            (
                "quota10_prototype_recall_exact_every_seed",
                independent["minimum_prototype_core_recall"] >= 1.0 - 1.0e-12,
                independent["minimum_prototype_core_recall"],
            ),
            (
                "projection_integrity",
                independent["maximum_projection_column_sum_error"]
                <= float(
                    protocol["integrity_gates"]["maximum_projection_column_sum_error"]
                )
                and independent["minimum_projection_probability"] >= 0.0
                and independent["maximum_projection_optimality_error"]
                <= float(
                    protocol["integrity_gates"]["maximum_projection_optimality_error"]
                )
                and independent["maximum_nonprototype_change"] == 0.0,
                independent,
            ),
            (
                "calibration_integrity",
                independent["maximum_calibration_mean_error"]
                <= float(
                    protocol["integrity_gates"]["maximum_calibration_moment_error"]
                )
                and independent["maximum_calibration_variance_error"]
                <= float(
                    protocol["integrity_gates"]["maximum_calibration_moment_error"]
                )
                and independent["maximum_calibration_unchanged_identity_error"] == 0.0
                and independent["maximum_calibration_scale"]
                <= float(protocol["integrity_gates"]["maximum_calibration_scale"]),
                independent,
            ),
            (
                "pooled_prior_is_valid_and_not_city_conditioned",
                pooled_receipt["city_conditioned_parameters"] == 0
                and pooled_receipt["minimum_probability"] > 0.0
                and pooled_receipt["probability_sum_error"] <= 1.0e-12,
                pooled_receipt,
            ),
            (
                "graph_fit_on_final_development_only",
                graph_receipt["fit_partition"]
                == "all_695_train_plus_spent_development_rows"
                and int(graph_receipt["documents"]) == 695,
                graph_receipt["fit_partition"],
            ),
            (
                "five_paired_variant_identities_frozen",
                int(worker["variant_count"]) == 5
                and [row["variant"] for row in variants["variants"]]
                == list(FINAL_VARIANTS)
                and sha256_object(variant_body) == variant_identity,
                variant_identity,
            ),
            (
                "all_final_checkpoints_present_and_hashed",
                len(checkpoints) == 10
                and all(
                    Path(row["path"]).is_file()
                    and sha256_file(Path(row["path"])) == str(row["sha256"])
                    for row in checkpoints
                ),
                len(checkpoints),
            ),
            (
                "freeze_receipt_matches_checkpoints_and_preregistration",
                freeze["checkpoint_manifest_sha256"]
                == sha256_file(worker_output / "checkpoint_manifest.json")
                and freeze["confirmation_preregistration_sha256"]
                == sha256_file(preregistration_path)
                and freeze["test_documents_used"] == 0,
                freeze["freeze_identity"],
            ),
            (
                "all_fit_diagnostics_finite",
                bool(
                    np.isfinite(
                        diagnostics.select_dtypes(include=[np.number]).to_numpy()
                    ).all()
                ),
                True,
            ),
            (
                "test_access_zero",
                int(worker["test_documents_used"]) == 0,
                worker["test_documents_used"],
            ),
            (
                "label_access_zero",
                int(worker["labels_used"]) == 0,
                worker["labels_used"],
            ),
            (
                "comparator_access_zero",
                int(worker["baseline_or_sota_outputs_used"]) == 0,
                worker["baseline_or_sota_outputs_used"],
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
        (output / "resolved_v6_step07.yaml").write_text(
            yaml.safe_dump(
                {"schema_version": 1, "v6": {"step7": protocol}}, sort_keys=False
            ),
            encoding="utf-8",
        )
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": SCOPE,
            "implementation_version": IMPLEMENTATION,
            "readiness": "READY_FOR_V6_1_STEP8_ONE_SHOT_TEST"
            if status == "PASS"
            else "BLOCKED_KEEP_TEST_SEALED",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "data_key": EXPECTED_DATA_KEY,
            "official_commit": EXPECTED_OFFICIAL_COMMIT,
            "step6r2_contract_sha256": sha256_file(step6r2_path),
            "selected_configuration_fingerprint": EXPECTED_CONFIGURATION_FINGERPRINT,
            "confirmation_preregistration_sha256": sha256_file(preregistration_path),
            "final_fit_documents": worker["final_fit_documents"],
            "original_training_documents": 620,
            "spent_development_documents": 75,
            "parent_fit_seeds": EXPECTED_SEEDS,
            "parent_neural_fits": worker["parent_neural_fits"],
            "seed_selection_performed": False,
            "variant_count": worker["variant_count"],
            "variant_manifest_identity": variant_identity,
            "freeze_identity": worker["freeze_identity"],
            "test_documents_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step07_contract.json", contract)
        _write_json(
            output / "generated_artifact_manifest.json", _artifact_manifest(output)
        )
        report(
            f"[V6.1 Step 7] {status} | hard checks={contract['hard_checks_passed']}/{contract['hard_checks_total']}"
        )
        report(
            f"[V6.1 Step 7] frozen | documents=695 seeds=10 variants=5 test=0 | "
            f"freeze={contract['freeze_identity'][:12]}"
        )
        report(f"[V6.1 Step 7] contract={output / 'step07_contract.json'}")
        return_zip = _make_return_zip(root, output, log_path)
        report(f"[V6.1 Step 7] return-zip={return_zip}")
        _make_return_zip(root, output, log_path)
        if failures:
            raise AssertionError(
                f"Step-7 final freeze failed integrity checks: {failures}; return the ZIP and keep test sealed"
            )
        return output
    except Exception:
        if output.exists():
            _write_json(
                output / "generated_artifact_manifest.json", _artifact_manifest(output)
            )
            _make_return_zip(root, output, log_path)
        raise
