"""Fail-closed orchestration for the V6 one-shot validation freeze gate."""

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
from src.v6.development_stage import (
    _base_prerequisites,
    _validate_runtime,
    _verify_fingerprint,
    _verify_frozen_input_artifacts,
    _worker_inputs,
)
from src.v6.official_encot import audit_official_source
from src.v6.step4r2 import _verify_manifest
from src.v6.validation_gate import summarize_validation_seeds


IMPLEMENTATION = "v6_step6_one_shot_validation_freeze_gate_r1"
SCOPE = "frozen_three_seed_train_only_fit_single_validation_open_no_test_or_comparators"
STEP5_IMPLEMENTATION = "v6_step5r2_target_aligned_graph_projection_gate_r5"
STEP5_READINESS = "READY_FOR_V6_STEP6_VALIDATION_GATE"
EXPECTED_SEEDS = [20260910, 20270910, 20280910]
EXPECTED_VARIANTS = [
    "full_graph_plus_city",
    "without_graph_city_retained",
    "without_city_graph_retained",
    "matched_pooled_backoff_graph_retained",
    "exact_official_parent",
]
EXPECTED_GATES = {
    "expected_seed_count": 3,
    "minimum_positive_seed_fraction": 2.0 / 3.0,
    "minimum_median_validation_npmi_improvement": 0.05,
    "minimum_median_validation_cv_improvement": 0.02,
    "minimum_median_zero_joint_pair_reduction": 0.0,
    "minimum_median_city_vs_without_city_reduction": 0.05,
    "minimum_median_city_vs_pooled_reduction": 0.005,
    "minimum_median_full_vs_parent_reduction": 0.05,
    "minimum_median_training_reference_npmi": -0.05,
    "minimum_median_training_reference_cv": 0.48,
    "minimum_validation_topic_diversity": 0.85,
    "maximum_validation_top_word_redundancy": 0.15,
    "minimum_full_topic_stability": 0.2,
    "maximum_stability_regression_vs_graph_removed": 0.02,
    "minimum_displayed_train_support_fraction": 0.85,
    "minimum_displayed_graph_support_fraction": 0.8,
    "minimum_displayed_positive_graph_pair_fraction": 0.15,
    "minimum_prototype_core_recall": 1.0,
    "maximum_probability_sum_error": 1.0e-6,
    "maximum_nonprototype_change": 0.0,
    "maximum_projection_column_sum_error": 1.0e-6,
    "minimum_projection_probability": 0.0,
    "maximum_projection_optimality_error": 1.0e-12,
    "maximum_changed_columns": 100,
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


def _load_protocol(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    protocol = document.get("v6", {}).get("step6", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("unsupported V6 Step-6 configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-6 implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-6 scope changed")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step 6 may not create or alter an environment")
    prerequisite = protocol.get("prerequisite", {})
    if (
        prerequisite.get("implementation_version") != STEP5_IMPLEMENTATION
        or prerequisite.get("readiness") != STEP5_READINESS
        or int(prerequisite.get("selected_prototype_quota", -1)) != 10
        or float(prerequisite.get("selected_city_mixture_weight", float("nan")))
        != 0.35
    ):
        raise ValueError("Step-6 Step-5R2 prerequisite lock changed")
    registration = protocol.get("preregistration", {})
    registration_path = config_path.resolve().parents[2] / str(
        registration.get("path", "")
    )
    if (
        registration.get("frozen_before_validation") is not True
        or not registration_path.is_file()
        or sha256_file(registration_path) != str(registration.get("sha256"))
    ):
        raise ValueError("Step-6 validation preregistration changed")
    receipt = json.loads(registration_path.read_text(encoding="utf-8-sig"))
    if (
        receipt.get("implementation_version") != IMPLEMENTATION
        or receipt.get("registration_status")
        != "FROZEN_BEFORE_FIRST_SEMANTIC_VALIDATION_ACCESS"
        or receipt.get("thresholds") != EXPECTED_GATES
    ):
        raise ValueError("Step-6 preregistration content changed")
    if list(map(int, protocol.get("training", {}).get("seeds", []))) != EXPECTED_SEEDS:
        raise ValueError("Step-6 frozen seed set changed")
    if list(protocol.get("evaluation", {}).get("variants", [])) != EXPECTED_VARIANTS:
        raise ValueError("Step-6 paired variant set changed")
    if protocol.get("gates") != EXPECTED_GATES:
        raise ValueError("Step-6 validation thresholds changed")
    training = protocol["training"]
    if any(
        (
            int(training["parent_epochs"]) != 120,
            float(training["parent_learning_rate"]) != 0.002,
            int(training["batch_size"]) != 200,
            training.get("stopping_rule") != "fixed_epochs_only",
            training.get("validation_early_stopping") is not False,
            training.get("seed_selection") is not False,
            training.get("hyperparameter_selection") is not False,
        )
    ):
        raise ValueError("Step-6 train-only fit policy changed")
    validation = protocol["validation"]
    if any(
        (
            int(validation["semantic_open_count"]) != 1,
            int(validation["expected_source_documents"]) != 75,
            int(validation["expected_completion_documents"]) != 73,
            int(validation["expected_target_tokens"]) != 310,
            validation.get("load_after_all_training_checkpoints_frozen") is not True,
            int(validation["candidate_count"]) != 1,
            validation.get("seed_selection_allowed") is not False,
            validation.get("hyperparameter_selection_allowed") is not False,
        )
    ):
        raise ValueError("Step-6 one-shot validation policy changed")
    if protocol["evaluation"].get("graph_ablation_nll") != (
        "mandatory_reported_tradeoff_not_gate"
    ):
        raise ValueError("Step-6 graph NLL disclosure semantics changed")
    if protocol["evaluation"].get("same_affinity_for_likelihood_and_reporting") is not True:
        raise ValueError("Step 6 must use one affinity for likelihood and reporting")
    return protocol


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


def _make_return_zip(root: Path, output: Path, log_path: Path) -> Path:
    destination = root / "v6_step06_results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("step06_validation_gate") / output.name
        for path in sorted(output.rglob("*")):
            if not path.is_file() or "checkpoints" in path.relative_to(output).parts:
                continue
            archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, "logs/v6_step06.log")
    return destination


def _find_passing_step5r2(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    prerequisite = protocol["prerequisite"]
    candidates = sorted(
        (root / "outputs" / "v6" / "step05r2_rank_aware_lexical").glob(
            "*/step05r2_contract.json"
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
                contract.get("status") != "PASS",
                contract.get("readiness") != prerequisite["readiness"],
                contract.get("implementation_version")
                != prerequisite["implementation_version"],
                contract.get("data_key") != prerequisite["data_key"],
                contract.get("official_commit") != prerequisite["official_commit"],
                float(contract.get("selected_city_mixture_weight", float("nan")))
                != float(prerequisite["selected_city_mixture_weight"]),
                int(contract.get("selected_prototype_quota", -1))
                != int(prerequisite["selected_prototype_quota"]),
                int(contract.get("validation_documents_used", -1)) != 0,
                int(contract.get("test_documents_used", -1)) != 0,
                not _verify_fingerprint(contract, "contract_fingerprint"),
                not _verify_manifest(output),
            )
        ):
            continue
        selected_metrics = output / "worker" / "selected_fold_metrics.csv"
        worker_result = output / "worker" / "worker_result.json"
        checkpoint_manifest = output / "worker" / "checkpoint_manifest.json"
        if not all(
            path.is_file()
            for path in (selected_metrics, worker_result, checkpoint_manifest)
        ):
            continue
        checkpoints = json.loads(
            checkpoint_manifest.read_text(encoding="utf-8-sig")
        )
        if len(checkpoints) != 5 or not all(
            int(row.get("prototype_quota", -1))
            == int(prerequisite["selected_prototype_quota"])
            and Path(row["path"]).is_file()
            and sha256_file(Path(row["path"])) == str(row["sha256"])
            for row in checkpoints
        ):
            continue
        manifest = json.loads(
            (output / "generated_artifact_manifest.json").read_text(
                encoding="utf-8-sig"
            )
        )
        lineage = {
            "contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "selected_metrics": {
                "path": str(selected_metrics),
                "sha256": sha256_file(selected_metrics),
            },
            "worker_result": {
                "path": str(worker_result),
                "sha256": sha256_file(worker_result),
            },
            "selected_fold_checkpoints": checkpoints,
            "manifest_identity": manifest["manifest_identity"],
        }
        return contract_path, contract, lineage
    raise RuntimeError(
        "No intact passing Step-5R2 R5 result with all five selected local checkpoints was found"
    )


def _verify_method_lock(
    root: Path,
    protocol: Mapping[str, Any],
    step5_contract: Mapping[str, Any],
) -> None:
    step5_config = root / "configs" / "v6" / "step05r2.yaml"
    if sha256_file(step5_config) != str(step5_contract["config_sha256"]):
        raise RuntimeError("the passing Step-5R2 configuration changed")
    old = yaml.safe_load(step5_config.read_text(encoding="utf-8-sig"))["v6"][
        "step5r2"
    ]
    if protocol["model"] != old["model"]:
        raise RuntimeError("Step-6 official-parent architecture differs from R5")
    if any(
        (
            float(protocol["city_backoff"]["fixed_mixture_weight"])
            != float(old["city_backoff"]["fixed_mixture_weight"]),
            float(protocol["city_backoff"]["fixed_shrinkage"])
            != float(old["city_backoff"]["fixed_shrinkage"]),
            float(protocol["city_backoff"]["fixed_pseudocount"])
            != float(old["city_backoff"]["fixed_pseudocount"]),
            int(protocol["graph_prototype_projection"]["selected_prototype_quota"])
            != int(step5_contract["selected_prototype_quota"]),
            float(protocol["graph_prototype_projection"]["rank_margin"])
            != float(old["graph_prototype_projection"]["rank_margin"]),
            float(protocol["lexical_graph"]["reliability_power"])
            != float(old["lexical_graph"]["reliability_power"]),
            int(protocol["lexical_graph"]["minimum_document_frequency"])
            != int(old["lexical_graph"]["minimum_document_frequency"]),
            int(protocol["lexical_graph"]["minimum_joint_documents"])
            != int(old["lexical_graph"]["minimum_joint_documents"]),
        )
    ):
        raise RuntimeError("Step-6 selected R5 mechanism changed")


def _inputs(
    step3_output: Path,
    step2_output: Path,
    step2_contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    development = _worker_inputs(step3_output, step2_output)
    _verify_frozen_input_artifacts(
        development, step3_output, step2_output, step2_contract
    )
    training = {
        "train_counts": development["train_counts"],
        "word_embeddings": development["word_embeddings"],
        "train_rows": development["fold_assignments"],
    }
    validation_paths = {
        "validation_rows": step2_output / "validation_completion_rows.csv",
        "validation_observed": step2_output / "validation_completion_observed.npz",
        "validation_target": step2_output / "validation_completion_target.npz",
    }
    missing = [str(path) for path in validation_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "frozen validation evidence is missing: " + "; ".join(missing)
        )
    manifest_path = step2_output / "generated_evaluation_artifact_manifest.json"
    if sha256_file(manifest_path) != str(
        step2_contract["generated_artifact_manifest_sha256"]
    ):
        raise RuntimeError("the Step-2 evidence manifest changed")
    expected = {
        str(row["filename"]): str(row["sha256"])
        for row in json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    }
    validation: dict[str, dict[str, Any]] = {}
    for role, path in validation_paths.items():
        observed = sha256_file(path)
        if observed != expected.get(path.name):
            raise RuntimeError(f"frozen Step-2 validation evidence changed: {path.name}")
        validation[role] = {"path": str(path), "sha256": observed}
    return training, validation


def _stream_worker(
    runtime: Path, root: Path, job_path: Path, reporter: _Reporter
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.step6_worker", "--job", str(job_path)],
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


def _is_finite_table(frame: pd.DataFrame) -> bool:
    values = frame.select_dtypes(include=[np.number]).to_numpy()
    return bool(values.size > 0 and np.isfinite(values).all())


def _equivalent_summary(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> bool:
    """Compare independently recomputed JSON-like summaries robustly."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equivalent_summary(left[key], right[key], tolerance=tolerance)
            for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent_summary(a, b, tolerance=tolerance)
            for a, b in zip(left, right)
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


def run_v6_step6(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError(
            "V6 Step 6 requires an isolated directory whose name contains 'V6'"
        )
    protocol = _load_protocol(config_path)
    run_id = _utc_id()
    output = root / "outputs" / "v6" / "step06_validation_gate" / run_id
    output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"v6_step06_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6 Step 6] START | implementation={IMPLEMENTATION}")
    report(
        "[V6 Step 6] scope=3 train-only fits -> immutable checkpoint freeze -> "
        "one validation open | candidate=1 seed-selection=0 test=0 comparators=0"
    )
    try:
        step5_path, step5_contract, lineage = _find_passing_step5r2(
            root, protocol
        )
        _verify_method_lock(root, protocol, step5_contract)
        (
            step3_path,
            step3_contract,
            step2_path,
            step2_contract,
            runtime,
            source,
            step3_specification,
        ) = _base_prerequisites(root, protocol)
        if sha256_file(step3_path) != str(step5_contract["step3_contract_sha256"]):
            raise RuntimeError("Step 6 found a different Step-3 prerequisite")
        if sha256_file(step2_path) != str(step5_contract["step2_contract_sha256"]):
            raise RuntimeError("Step 6 found a different Step-2 prerequisite")
        if int(step2_contract["source_documents"]["validation"]) != int(
            protocol["validation"]["expected_source_documents"]
        ) or int(step2_contract["validation_completion_documents"]) != int(
            protocol["validation"]["expected_completion_documents"]
        ):
            raise RuntimeError("the frozen validation cohort size changed")
        training_inputs, validation_inputs = _inputs(
            step3_path.parent, step2_path.parent, step2_contract
        )
        _write_json(output / "step5r2_lineage.json", lineage)
        worker_output = output / "worker"
        job = {
            "schema_version": 1,
            "mode": "step6",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output),
            "official_source": str(source),
            "official_commit": step3_contract["official_commit"],
            "training_inputs": training_inputs,
            "validation_inputs": validation_inputs,
            "step5r2_lineage": lineage,
            "cities": protocol["cities"],
            "global_context": protocol["global_context"],
            "model": protocol["model"],
            "training": protocol["training"],
            "city_backoff": protocol["city_backoff"],
            "lexical_graph": protocol["lexical_graph"],
            "graph_prototype_projection": protocol[
                "graph_prototype_projection"
            ],
            "validation": protocol["validation"],
            "evaluation": protocol["evaluation"],
            "gates": protocol["gates"],
        }
        job_path = output / "worker_job.json"
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
            raise RuntimeError("Step-6 exact-runtime worker failed")

        worker = json.loads(
            (worker_output / "worker_result.json").read_text(
                encoding="utf-8-sig"
            )
        )
        metrics = pd.read_csv(
            worker_output / "validation_seed_metrics.csv", encoding="utf-8-sig"
        )
        worker_summary = json.loads(
            (worker_output / "validation_summary.json").read_text(
                encoding="utf-8-sig"
            )
        )
        independently_recomputed = summarize_validation_seeds(
            metrics.to_dict(orient="records"),
            gates=protocol["gates"],
            full_topic_stability=worker_summary["full_topic_stability"],
            graph_removed_topic_stability=worker_summary[
                "graph_removed_topic_stability"
            ],
        )
        checkpoint_manifest = worker["checkpoint_manifest"]
        freeze = json.loads(
            (worker_output / "training_freeze_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        validation_open = json.loads(
            (worker_output / "validation_open_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        assignment = json.loads(
            (worker_output / "validation_assignment_receipt.json").read_text(
                encoding="utf-8-sig"
            )
        )
        current_source_receipt = audit_official_source(
            source, step3_specification["official_source"]
        )
        gate_results = independently_recomputed["gate_results"]
        checks: list[tuple[str, bool, Any]] = [
            ("step5r2_contract_and_manifest_verified", True, lineage["manifest_identity"]),
            ("step5r2_contract_exact", sha256_file(step5_path) == lineage["contract"]["sha256"], lineage["contract"]["sha256"]),
            ("step5r2_r5_candidate_frozen", int(step5_contract["selected_prototype_quota"]) == 10 and float(step5_contract["selected_city_mixture_weight"]) == 0.35, {"quota": step5_contract["selected_prototype_quota"], "rho": step5_contract["selected_city_mixture_weight"]}),
            ("validation_preregistration_hash_verified", sha256_file(root / protocol["preregistration"]["path"]) == protocol["preregistration"]["sha256"], protocol["preregistration"]["sha256"]),
            ("exact_runtime_unchanged", _validate_runtime(worker_output, step3_specification), str(runtime)),
            ("official_source_commit_unchanged", current_source_receipt["commit"] == protocol["prerequisite"]["official_commit"], current_source_receipt["commit"]),
            ("training_partition_exact", int(worker["training_documents"]) == 620, worker["training_documents"]),
            ("three_predeclared_parent_fits", int(worker["parent_neural_fits"]) == 3 and metrics["seed"].astype(int).tolist() == EXPECTED_SEEDS, metrics["seed"].astype(int).tolist()),
            ("all_training_checkpoints_frozen_and_hashed", len(checkpoint_manifest) == 3 and all(Path(row["path"]).is_file() and sha256_file(Path(row["path"])) == str(row["sha256"]) for row in checkpoint_manifest), len(checkpoint_manifest)),
            ("checkpoint_freeze_precedes_validation_open", validation_open["training_checkpoints_frozen_before_open"] is True and validation_open["training_freeze_identity"] == freeze["freeze_identity"], freeze["freeze_identity"]),
            ("validation_opened_exactly_once", int(worker["validation_semantic_open_count"]) == 1 and int(validation_open["semantic_open_count"]) == 1, worker["validation_semantic_open_count"]),
            ("validation_cohort_exact", int(worker["validation_documents_used"]) == 73 and int(worker["validation_target_tokens"]) == 310, {"documents": worker["validation_documents_used"], "target_tokens": worker["validation_target_tokens"]}),
            ("validation_assignment_observed_only", assignment["target_half_used_for_assignment"] is False and assignment["full_document_embedding_used"] is False and int(assignment["documents_assigned"]) == 73, assignment),
            ("one_frozen_candidate_only", int(worker["candidate_architectures_evaluated"]) == 1, worker["candidate_architectures_evaluated"]),
            ("fifteen_paired_variant_evaluations", int(worker["paired_variant_evaluations"]) == 15, worker["paired_variant_evaluations"]),
            ("no_seed_selection", worker["seed_selection_performed"] is False and independently_recomputed["seed_selection_performed"] is False, False),
            ("no_hyperparameter_selection", worker["hyperparameter_selection_performed"] is False and independently_recomputed["hyperparameter_selection_performed"] is False, False),
            ("no_validation_early_stopping", worker["validation_early_stopping_performed"] is False, False),
            ("zero_projection_exact_parent_identity", float(worker["maximum_zero_projection_identity_error"]) == 0.0, worker["maximum_zero_projection_identity_error"]),
            ("independent_validation_summary_reproduced", _equivalent_summary(worker_summary, independently_recomputed), sha256_object(independently_recomputed)),
            ("graph_ablation_nll_tradeoff_reported", np.isfinite(float(independently_recomputed["median_graph_nll_reduction_vs_without_graph"])) and independently_recomputed["graph_nll_tradeoff_is_report_only"] is True, independently_recomputed["median_graph_nll_reduction_vs_without_graph"]),
            ("all_reported_numeric_metrics_finite", _is_finite_table(metrics), True),
            ("test_access_zero", int(worker["test_documents_used"]) == 0, worker["test_documents_used"]),
            ("label_access_zero", int(worker["labels_used"]) == 0, worker["labels_used"]),
            ("comparator_access_zero", int(worker["baseline_or_sota_outputs_used"]) == 0, worker["baseline_or_sota_outputs_used"]),
        ]
        checks.extend(
            (name, bool(passed), independently_recomputed.get(name, passed))
            for name, passed in gate_results.items()
        )
        hard_rows = [
            {
                "check": name,
                "required": True,
                "observed": value,
                "status": "PASS" if passed else "FAIL",
            }
            for name, passed, value in checks
        ]
        _write_csv(output / "hard_checks.csv", hard_rows)
        failures = [row["check"] for row in hard_rows if row["status"] == "FAIL"]
        status = "PASS" if not failures else "FAIL"
        (output / "resolved_v6_step06.yaml").write_text(
            yaml.safe_dump(
                {"schema_version": 1, "v6": {"step6": protocol}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": SCOPE,
            "implementation_version": IMPLEMENTATION,
            "readiness": "READY_FOR_V6_STEP7_FINAL_TRAINING_FREEZE" if status == "PASS" else "BLOCKED_DO_NOT_OPEN_TEST_OR_COMPARATORS",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "data_key": step5_contract["data_key"],
            "official_commit": step5_contract["official_commit"],
            "step5r2_contract_sha256": sha256_file(step5_path),
            "step3_contract_sha256": sha256_file(step3_path),
            "step2_contract_sha256": sha256_file(step2_path),
            "preregistration_sha256": protocol["preregistration"]["sha256"],
            "selected_city_mixture_weight": 0.35,
            "selected_prototype_quota": 10,
            "parent_fit_seeds": EXPECTED_SEEDS,
            "parent_neural_fits": worker["parent_neural_fits"],
            "validation_documents_used": worker["validation_documents_used"],
            "validation_target_tokens": worker["validation_target_tokens"],
            "validation_semantic_open_count": worker["validation_semantic_open_count"],
            "seed_selection_performed": False,
            "hyperparameter_selection_performed": False,
            "median_validation_npmi_improvement_vs_without_graph": independently_recomputed["median_validation_npmi_improvement_vs_without_graph"],
            "median_validation_cv_improvement_vs_without_graph": independently_recomputed["median_validation_cv_improvement_vs_without_graph"],
            "median_city_nll_reduction_vs_without_city": independently_recomputed["median_city_nll_reduction_vs_without_city"],
            "median_city_nll_reduction_vs_matched_pooling": independently_recomputed["median_city_nll_reduction_vs_matched_pooling"],
            "median_full_nll_reduction_vs_official_parent": independently_recomputed["median_full_nll_reduction_vs_official_parent"],
            "median_graph_nll_reduction_vs_without_graph": independently_recomputed["median_graph_nll_reduction_vs_without_graph"],
            "graph_nll_tradeoff_is_report_only": True,
            "median_validation_npmi_full": independently_recomputed["median_validation_npmi_full"],
            "median_validation_cv_full": independently_recomputed["median_validation_cv_full"],
            "median_training_reference_npmi_full": independently_recomputed["median_training_reference_npmi_full"],
            "median_training_reference_cv_full": independently_recomputed["median_training_reference_cv_full"],
            "test_documents_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step06_contract.json", contract)
        _write_json(
            output / "generated_artifact_manifest.json", _artifact_manifest(output)
        )
        report(
            f"[V6 Step 6] {status} | hard checks="
            f"{contract['hard_checks_passed']}/{contract['hard_checks_total']}"
        )
        report(
            "[V6 Step 6] validation effects | graph dNPMI="
            f"{contract['median_validation_npmi_improvement_vs_without_graph']:+.6f} "
            "dC_v="
            f"{contract['median_validation_cv_improvement_vs_without_graph']:+.6f} "
            "city dNLL="
            f"{contract['median_city_nll_reduction_vs_without_city']:+.6f} "
            "full-vs-parent="
            f"{contract['median_full_nll_reduction_vs_official_parent']:+.6f}"
        )
        report(
            "[V6 Step 6] disclosed graph NLL change vs -graph="
            f"{contract['median_graph_nll_reduction_vs_without_graph']:+.6f}"
        )
        report(f"[V6 Step 6] contract={output / 'step06_contract.json'}")
        return_zip = _make_return_zip(root, output, log_path)
        report(f"[V6 Step 6] return-zip={return_zip}")
        _make_return_zip(root, output, log_path)
        if failures:
            raise AssertionError(
                f"Step-6 validation failed frozen gates: {failures}; "
                "return v6_step06_results.zip and do not open test/comparators"
            )
        return output
    except Exception:
        if output.exists():
            _write_json(
                output / "generated_artifact_manifest.json",
                _artifact_manifest(output),
            )
            _make_return_zip(root, output, log_path)
        raise
