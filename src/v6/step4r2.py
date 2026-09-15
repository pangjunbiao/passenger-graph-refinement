"""Fail-closed orchestration for the revised V6 Step-4 city mechanism."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping
import zipfile

import pandas as pd
import yaml

from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_stage import (
    _base_prerequisites,
    _validate_runtime,
    _verify_fingerprint,
    _verify_frozen_input_artifacts,
    _worker_inputs,
)


IMPLEMENTATION = "v6_step4r2_hierarchical_city_backoff_gate_r1"
SCOPE = "grouped_original_train_city_backoff_revision_no_validation_test_or_comparators"
FAILED_R1_IMPLEMENTATION = "v6_step4_city_residual_grouped_mechanistic_gate_r1"
FAILED_R1_CHECKS = {
    "paired_city_nll_practical_effect",
    "paired_city_nll_fold_consistency",
}


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
    manifest = {"schema_version": 1, "files": files}
    manifest["manifest_identity"] = sha256_object(manifest)
    return manifest


def _make_return_zip(root: Path, output: Path, log_path: Path) -> Path:
    destination = root / "v6_step04r2_results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("step04r2_city_backoff") / output.name
        for path in sorted(output.rglob("*")):
            if not path.is_file() or "checkpoints" in path.relative_to(output).parts:
                continue
            archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, "logs/v6_step04r2.log")
    return destination


def _load_protocol(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    protocol = document.get("v6", {}).get("step4r2", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("unsupported V6 Step-4R2 configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-4R2 implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-4R2 scope changed")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step-4R2 may not create or alter an environment")
    weights = list(map(float, protocol["candidate_mixture_weights"]))
    if weights != sorted(set(weights)) or not weights:
        raise ValueError("candidate mixture weights must be sorted and unique")
    if max(weights) > float(protocol["gates"]["maximum_mixture_weight"]):
        raise ValueError("candidate mixture grid exceeds its dominance guardrail")
    return protocol


def _verify_manifest(output: Path) -> bool:
    path = output / "generated_artifact_manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    identity = str(manifest.get("manifest_identity", ""))
    body = dict(manifest)
    body.pop("manifest_identity", None)
    if sha256_object(body) != identity:
        return False
    for row in manifest.get("files", []):
        source = output / str(row["relative_path"])
        if (
            not source.is_file()
            or int(source.stat().st_size) != int(row["size_bytes"])
            or sha256_file(source) != str(row["sha256"])
        ):
            return False
    return True


def _find_failed_r1(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates = sorted(
        (root / "outputs" / "v6" / "step04_city_residual").glob(
            "*/step04_contract.json"
        ),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            contract.get("implementation_version") != FAILED_R1_IMPLEMENTATION
            or contract.get("status") != "FAIL"
            or set(contract.get("failed_checks", [])) != FAILED_R1_CHECKS
            or not _verify_fingerprint(contract, "contract_fingerprint")
            or not _verify_manifest(contract_path.parent)
        ):
            continue
        checkpoint_path = contract_path.parent / "worker" / "checkpoint_manifest.json"
        fold_metrics_path = contract_path.parent / "worker" / "fold_metrics.csv"
        worker_job_path = contract_path.parent / "worker_job.json"
        if not all(path.is_file() for path in (checkpoint_path, fold_metrics_path, worker_job_path)):
            continue
        checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        if len(checkpoints) != 5 or not all(
            Path(row["path"]).is_file()
            and sha256_file(Path(row["path"])) == str(row["sha256"])
            for row in checkpoints
        ):
            continue
        lineage = {
            "contract_path": str(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "manifest_identity": json.loads(
                (contract_path.parent / "generated_artifact_manifest.json").read_text(
                    encoding="utf-8-sig"
                )
            )["manifest_identity"],
            "fold_metrics_path": str(fold_metrics_path),
            "fold_metrics_sha256": sha256_file(fold_metrics_path),
            "worker_job_path": str(worker_job_path),
            "worker_job_sha256": sha256_file(worker_job_path),
            "checkpoints": checkpoints,
        }
        return contract_path, contract, lineage
    raise RuntimeError(
        "No intact failed Step-4 R1 result with all five local checkpoints was found"
    )


def _stream_worker(
    runtime: Path, root: Path, job_path: Path, reporter: _Reporter
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.step4r2_worker", "--job", str(job_path)],
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


def _configuration_matches_failed_run(
    protocol: Mapping[str, Any], lineage: Mapping[str, Any]
) -> bool:
    job = json.loads(Path(lineage["worker_job_path"]).read_text(encoding="utf-8-sig"))
    return bool(
        protocol["cities"] == job["cities"]
        and protocol["development_folds"] == job["development_folds"]
        and protocol["global_context"] == job["global_context"]
        and protocol["model"] == job["model"]
        and protocol["evaluation"] == job["evaluation"]
    )


def run_v6_step4r2(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError("V6 Step-4R2 requires an isolated directory whose name contains 'V6'")
    protocol = _load_protocol(config_path)
    run_id = _utc_id()
    output = root / "outputs" / "v6" / "step04r2_city_backoff" / run_id
    output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"v6_step04r2_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6 Step 4R2] START | implementation={IMPLEMENTATION}")
    report(
        "[V6 Step 4R2] scope=original-train grouped folds | reuse-parent=5 "
        "neural-refit=0 validation=0 test=0 comparators=0"
    )
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
        failed_path, failed_contract, lineage = _find_failed_r1(root)
        if failed_contract["step3_contract_sha256"] != sha256_file(step3_path):
            raise RuntimeError("failed Step-4 and current Step-3 lineage differ")
        if failed_contract["step2_contract_sha256"] != sha256_file(step2_path):
            raise RuntimeError("failed Step-4 and current Step-2 lineage differ")
        if not _configuration_matches_failed_run(protocol, lineage):
            raise RuntimeError("parent, context, city, fold, or evaluator settings changed after R1")
        inputs = _worker_inputs(step3_path.parent, step2_path.parent)
        _verify_frozen_input_artifacts(
            inputs, step3_path.parent, step2_path.parent, step2_contract
        )
        lineage["failed_checks"] = sorted(FAILED_R1_CHECKS)
        lineage["acknowledged_as_scientific_failure"] = True
        _write_json(output / "failed_step4r1_lineage.json", lineage)
        _write_json(
            output / "revision_rationale.json",
            {
                "schema_version": 1,
                "retired_component": "topic-normalized low-rank city-affinity residual",
                "reason": "training reconstruction improved but held-fold completion did not; the useful city signal is a stable lexical background signal and was attenuated by topic-axis centering and an objective/evaluator mismatch",
                "replacement": "partially pooled city lexical prior combined with the native parent probability through a normalized convex backoff channel",
                "matched_control": "the same backoff weight and pooled prior with all city distinctions removed",
                "confirmation_status": "development revision only; validation and test remain unopened",
            },
        )
        worker_output = output / "worker"
        job = {
            "schema_version": 1,
            "mode": "step4r2",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output),
            "official_source": str(source),
            "official_commit": step3_contract["official_commit"],
            "inputs": inputs,
            "failed_step4_lineage": lineage,
            "cities": protocol["cities"],
            "development_folds": protocol["development_folds"],
            "global_context": protocol["global_context"],
            "model": protocol["model"],
            "evaluation": protocol["evaluation"],
            "city_prior": protocol["city_prior"],
            "candidate_mixture_weights": protocol["candidate_mixture_weights"],
            "gates": protocol["gates"],
            "seed": protocol["seed"],
        }
        job_path = output / "worker_job.json"
        _write_json(job_path, job)
        code, transcript = _stream_worker(runtime, root, job_path, report)
        if code != 0:
            _write_json(
                output / "worker_failure.json",
                {"schema_version": 1, "return_code": code, "transcript": transcript},
            )
            raise RuntimeError("Step-4R2 exact-runtime worker failed")

        worker = json.loads(
            (worker_output / "worker_result.json").read_text(encoding="utf-8-sig")
        )
        summaries = pd.read_csv(
            worker_output / "candidate_summary.csv", encoding="utf-8-sig"
        )
        selected_weight = worker.get("selected_mixture_weight")
        selected = (
            summaries.loc[summaries["mixture_weight"] == float(selected_weight)].iloc[0]
            if selected_weight is not None
            else None
        )
        gates = protocol["gates"]
        checkpoint_manifest = worker["checkpoint_manifest"]
        checks: list[tuple[str, bool, Any]] = [
            ("failed_r1_preserved_and_acknowledged", bool(lineage["acknowledged_as_scientific_failure"]), lineage["contract_sha256"]),
            ("failed_r1_manifest_and_contract_verified", True, lineage["manifest_identity"]),
            ("failed_r1_only_expected_scientific_checks", set(failed_contract["failed_checks"]) == FAILED_R1_CHECKS, failed_contract["failed_checks"]),
            ("step3_contract_fingerprint", _verify_fingerprint(step3_contract, "contract_fingerprint"), step3_contract["contract_fingerprint"]),
            ("step2_data_key_exact", step2_contract["data_key"] == protocol["prerequisite"]["data_key"], step2_contract["data_key"]),
            ("parent_context_evaluator_configuration_unchanged", _configuration_matches_failed_run(protocol, lineage), True),
            ("exact_runtime_unchanged", _validate_runtime(worker_output, step3_specification), str(runtime)),
            ("official_source_unchanged", worker["official_class_loaded_from"].casefold().startswith(str(source).casefold()), worker["official_class_loaded_from"]),
            ("all_five_parent_checkpoints_reused", int(worker["reused_hash_verified_parent_checkpoints"]) == 5, worker["reused_hash_verified_parent_checkpoints"]),
            ("no_neural_parent_refit", int(worker["parent_neural_refits"]) == 0, worker["parent_neural_refits"]),
            ("candidate_selection_succeeded", selected is not None, selected_weight),
            ("city_vs_matched_pooling_practical_effect", selected is not None and float(selected["median_city_vs_pooled_reduction"]) >= float(gates["minimum_city_vs_pooled_reduction"]), None if selected is None else float(selected["median_city_vs_pooled_reduction"])),
            ("city_vs_matched_pooling_fold_consistency", selected is not None and float(selected["positive_city_vs_pooled_fold_fraction"]) >= float(gates["minimum_positive_fold_fraction"]), None if selected is None else float(selected["positive_city_vs_pooled_fold_fraction"])),
            ("city_vs_exact_parent_practical_effect", selected is not None and float(selected["median_city_vs_parent_reduction"]) >= float(gates["minimum_city_vs_parent_reduction"]), None if selected is None else float(selected["median_city_vs_parent_reduction"])),
            ("city_vs_exact_parent_fold_consistency", selected is not None and float(selected["positive_city_vs_parent_fold_fraction"]) >= float(gates["minimum_positive_fold_fraction"]), None if selected is None else float(selected["positive_city_vs_parent_fold_fraction"])),
            ("convexity_guaranteed_parent_effect", selected is not None and float(selected["minimum_guaranteed_parent_reduction_lower_bound"]) >= float(gates["minimum_guaranteed_parent_reduction"]), None if selected is None else float(selected["minimum_guaranteed_parent_reduction_lower_bound"])),
            ("jensen_certificate", float(worker["maximum_jensen_bound_violation"]) <= float(gates["jensen_tolerance"]), worker["maximum_jensen_bound_violation"]),
            ("zero_weight_exact_parent_identity", float(worker["maximum_alpha_zero_parent_identity_error"]) == 0.0, worker["maximum_alpha_zero_parent_identity_error"]),
            ("r1_parent_nll_reproduced", float(worker["maximum_r1_parent_nll_reproduction_error"]) <= float(gates["r1_parent_nll_reproduction_tolerance"]), worker["maximum_r1_parent_nll_reproduction_error"]),
            ("probability_simplex", float(worker["maximum_probability_sum_error"]) <= float(gates["probability_tolerance"]), worker["maximum_probability_sum_error"]),
            ("parent_dominance_guardrail", selected_weight is not None and float(selected_weight) <= float(gates["maximum_mixture_weight"]), selected_weight),
            ("all_revised_checkpoints_present_and_hashed", len(checkpoint_manifest) == 5 and all(Path(row["path"]).is_file() and sha256_file(Path(row["path"])) == row["sha256"] for row in checkpoint_manifest), len(checkpoint_manifest)),
            ("validation_access_zero", worker["validation_documents_used"] == 0, worker["validation_documents_used"]),
            ("test_access_zero", worker["test_documents_used"] == 0, worker["test_documents_used"]),
            ("label_access_zero", worker["labels_used"] == 0, worker["labels_used"]),
            ("comparator_access_zero", worker["baseline_or_sota_outputs_used"] == 0, worker["baseline_or_sota_outputs_used"]),
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
        (output / "resolved_v6_step04r2.yaml").write_text(
            yaml.safe_dump(
                {"schema_version": 1, "v6": {"step4r2": protocol}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": SCOPE,
            "implementation_version": IMPLEMENTATION,
            "readiness": "READY_FOR_V6_STEP5R2_RANK_AWARE_LEXICAL_INTEGRATION" if status == "PASS" else "BLOCKED_REVISE_BEFORE_STEP5",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "data_key": step3_contract["data_key"],
            "step3_contract_sha256": sha256_file(step3_path),
            "step2_contract_sha256": sha256_file(step2_path),
            "failed_step4r1_contract_sha256": sha256_file(failed_path),
            "official_commit": step3_contract["official_commit"],
            "selected_mixture_weight": selected_weight,
            "selection_is_original_train_development_only": True,
            "validation_remains_unopened": True,
            "test_remains_unopened": True,
            "median_city_vs_matched_pooling_nll_reduction": None if selected is None else float(selected["median_city_vs_pooled_reduction"]),
            "positive_city_vs_matched_pooling_fold_fraction": None if selected is None else float(selected["positive_city_vs_pooled_fold_fraction"]),
            "median_city_vs_exact_parent_nll_reduction": None if selected is None else float(selected["median_city_vs_parent_reduction"]),
            "minimum_convexity_guaranteed_parent_reduction": None if selected is None else float(selected["minimum_guaranteed_parent_reduction_lower_bound"]),
            "parent_neural_refits": worker["parent_neural_refits"],
            "reused_parent_checkpoints": worker["reused_hash_verified_parent_checkpoints"],
            "validation_documents_used": 0,
            "test_documents_used": 0,
            "human_labels_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step04r2_contract.json", contract)
        _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
        report(
            f"[V6 Step 4R2] {status} | hard checks={contract['hard_checks_passed']}/{contract['hard_checks_total']} "
            f"selected-rho={selected_weight}"
        )
        if selected is not None:
            report(
                "[V6 Step 4R2] effects | median city-vs-matched-pooling="
                f"{float(selected['median_city_vs_pooled_reduction']):+.6f} "
                "median city-vs-parent="
                f"{float(selected['median_city_vs_parent_reduction']):+.6f}"
            )
        report(f"[V6 Step 4R2] contract={output / 'step04r2_contract.json'}")
        return_zip = _make_return_zip(root, output, log_path)
        report(f"[V6 Step 4R2] return-zip={return_zip}")
        # Refresh the archive so its log contains the final path message.
        _make_return_zip(root, output, log_path)
        if failures:
            raise AssertionError(f"Step-4R2 failed gates: {failures}")
        return output
    except Exception:
        if output.exists():
            _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
            _make_return_zip(root, output, log_path)
        raise
