"""Fail-closed orchestration for V6 Step 5R2 revision 5."""

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
from src.v6.step4r2 import _verify_manifest


IMPLEMENTATION = "v6_step5r2_target_aligned_graph_projection_gate_r5"
SCOPE = "grouped_original_train_frozen_parent_target_aligned_graph_projection_development_freeze_no_validation_test_or_comparators"
STEP4R2_IMPLEMENTATION = "v6_step4r2_hierarchical_city_backoff_gate_r1"
STEP4R2_READINESS = "READY_FOR_V6_STEP5R2_RANK_AWARE_LEXICAL_INTEGRATION"


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
    protocol = document.get("v6", {}).get("step5r2", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("unsupported V6 Step-5R2 configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-5R2 R5 implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-5R2 R5 scope changed")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step-5R2 may not create or alter an environment")
    revision = protocol.get("revision_basis", {})
    if revision.get("numeric_target_threshold_changes") != "none":
        raise ValueError("R5 may not relax the lexical, city, or integrity targets")
    if revision.get("mechanism_change") != (
        "none_r5_reuses_the_exact_r4_projection_and_candidate_grid"
    ):
        raise ValueError("R5 may not change the audited R4 projection mechanism")
    if revision.get("selection_semantics_change") != (
        "graph_removed_nll_is_a_mandatory_reported_tradeoff_not_a_graph_component_survival_gate"
    ):
        raise ValueError("R5 component-survival semantics changed")
    audit_path = config_path.resolve().parents[2] / revision["audit_receipt"]
    if (
        not audit_path.is_file()
        or sha256_file(audit_path) != str(revision["audit_receipt_sha256"])
    ):
        raise ValueError("the R4-failure/R5-selection audit receipt changed")
    projection = protocol.get("graph_prototype_projection", {})
    if projection.get("optimization") != (
        "closed_form_no_gradient_no_optimizer_no_epoch_selection"
    ):
        raise ValueError("R5 must remain a closed-form projection")
    if projection.get("parent_state") != (
        "exact_frozen_passing_step4r2_checkpoint"
    ):
        raise ValueError("R5 parent checkpoint may not be continued")
    if projection.get("modified_support") != (
        "aligned_disjoint_prototype_columns_only"
    ):
        raise ValueError("R5 projection support changed")
    quotas = list(map(int, projection.get("candidate_prototype_quotas", [])))
    top_k = int(protocol["evaluation"]["top_words"])
    if (
        not quotas
        or quotas != sorted(set(quotas))
        or min(quotas) < 1
        or max(quotas) != top_k
    ):
        raise ValueError("R5 quotas must be sorted, unique, and end at top-k")
    if projection.get("selection_rule") != (
        "smallest_predeclared_quota_passing_all_target_integrity_city_and_parent_benefit_gates"
    ):
        raise ValueError("R5 selection rule changed")
    if float(projection["rank_margin"]) <= 0.0:
        raise ValueError("R5 rank margin must be positive")
    if int(projection["minimum_cluster_size"]) < int(
        protocol["evaluation"]["top_words"]
    ):
        raise ValueError("prototype cluster is smaller than displayed top-k")
    graph = protocol["lexical_graph"]
    if (
        int(graph["minimum_document_frequency"]) < 2
        or int(graph["minimum_joint_documents"]) < 1
        or float(graph["reliability_power"]) <= 0.0
        or int(graph["minimum_active_words"])
        < int(protocol["model"]["topics"])
        * int(protocol["evaluation"]["top_words"])
    ):
        raise ValueError("Step-5R2 lexical graph contract changed")
    if protocol["evaluation"].get("same_affinity_for_likelihood_and_reporting") is not True:
        raise ValueError("R5 must use one affinity for likelihood and reporting")
    if float(protocol["city_backoff"]["fixed_mixture_weight"]) != float(
        protocol["prerequisite"]["selected_mixture_weight"]
    ):
        raise ValueError("Step-4R2 mixture-weight lock changed")
    interpretation = protocol.get("ablation_interpretation", {})
    if interpretation.get("graph_removed_nll_comparison") != (
        "mandatory_reported_tradeoff_not_selection_gate"
    ):
        raise ValueError("R5 must report, but must not gate on, graph-ablation NLL")
    tradeoff = protocol.get("reported_graph_ablation_nll_tradeoff", {})
    if tradeoff.get("required") is not True or tradeoff.get(
        "affects_candidate_eligibility"
    ) is not False:
        raise ValueError("R5 graph-ablation NLL disclosure contract changed")
    expected_r4_references = {
        "r4_reference_maximum_median_completion_nll_regression": 0.030,
        "r4_reference_maximum_any_fold_completion_nll_regression": 0.080,
        "r4_reference_maximum_median_nll_regression_vs_step4r2": 0.020,
        "r4_reference_maximum_any_fold_nll_regression_vs_step4r2": 0.050,
    }
    if any(
        float(tradeoff.get(name, float("nan"))) != value
        for name, value in expected_r4_references.items()
    ):
        raise ValueError("R5 must preserve the failed R4 NLL references for disclosure")
    if float(protocol["gates"]["minimum_prototype_core_recall"]) != 1.0:
        raise ValueError("R5 requires complete top-k prototype-core recall")
    return protocol


def _artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if (
            not path.is_file()
            or path.name == "generated_artifact_manifest.json"
            or "candidate_checkpoints" in path.relative_to(output).parts
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
    destination = root / "v6_step05r2_results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("step05r2_rank_aware_lexical") / output.name
        for path in sorted(output.rglob("*")):
            if not path.is_file() or "candidate_checkpoints" in path.relative_to(
                output
            ).parts:
                continue
            archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, "logs/v6_step05r2.log")
    return destination


def _find_passing_step4r2(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates = sorted(
        (root / "outputs" / "v6" / "step04r2_city_backoff").glob(
            "*/step04r2_contract.json"
        ),
        reverse=True,
    )
    for path in candidates:
        try:
            contract = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        output = path.parent
        if any(
            (
                contract.get("status") != "PASS",
                contract.get("readiness") != STEP4R2_READINESS,
                contract.get("implementation_version") != STEP4R2_IMPLEMENTATION,
                contract.get("data_key") != protocol["prerequisite"]["data_key"],
                contract.get("official_commit")
                != protocol["prerequisite"]["official_commit"],
                float(contract.get("selected_mixture_weight", float("nan")))
                != float(protocol["prerequisite"]["selected_mixture_weight"]),
                not bool(contract.get("validation_remains_unopened")),
                not bool(contract.get("test_remains_unopened")),
                not _verify_fingerprint(contract, "contract_fingerprint"),
                not _verify_manifest(output),
            )
        ):
            continue
        checkpoint_path = output / "worker" / "checkpoint_manifest.json"
        selected_metrics_path = output / "worker" / "selected_fold_metrics.csv"
        if not checkpoint_path.is_file() or not selected_metrics_path.is_file():
            continue
        checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        if len(checkpoints) != 5 or not all(
            Path(row["path"]).is_file()
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
            "contract_path": str(path),
            "contract_sha256": sha256_file(path),
            "manifest_identity": manifest["manifest_identity"],
            "selected_fold_metrics_path": str(selected_metrics_path),
            "selected_fold_metrics_sha256": sha256_file(selected_metrics_path),
            "checkpoints": checkpoints,
        }
        return path, contract, lineage
    raise RuntimeError(
        "No intact passing Step-4R2 result with all five local checkpoints was found"
    )


def _stream_worker(
    runtime: Path, root: Path, job_path: Path, reporter: _Reporter
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.step5r2_worker", "--job", str(job_path)],
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


def _is_true(value: Any) -> bool:
    return bool(value) if isinstance(value, (bool, np.bool_)) else str(value).casefold() == "true"


def run_v6_step5r2(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError(
            "V6 Step-5R2 requires an isolated directory whose name contains 'V6'"
        )
    protocol = _load_protocol(config_path)
    run_id = _utc_id()
    output = root / "outputs" / "v6" / "step05r2_rank_aware_lexical" / run_id
    output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"v6_step05r2_{run_id}.log"
    report = _Reporter(log_path)
    report(f"[V6 Step 5R2] START | implementation={IMPLEMENTATION}")
    report(
        "[V6 Step 5R2] scope=original-train grouped folds | frozen Step-4R2 "
        "parent | target-aligned closed-form projection evaluations=20 "
        "optimizer-steps=0 "
        "validation=0 test=0 comparators=0"
    )
    try:
        step4_path, step4_contract, lineage = _find_passing_step4r2(root, protocol)
        step4_config = root / "configs" / "v6" / "step04r2.yaml"
        if not step4_config.is_file() or sha256_file(step4_config) != str(
            step4_contract["config_sha256"]
        ):
            raise RuntimeError("the passing Step-4R2 configuration changed")
        (
            step3_path,
            step3_contract,
            step2_path,
            step2_contract,
            runtime,
            source,
            step3_specification,
        ) = _base_prerequisites(root, protocol)
        if sha256_file(step3_path) != str(step4_contract["step3_contract_sha256"]):
            raise RuntimeError("Step-5R2 found a different Step-3 prerequisite")
        if sha256_file(step2_path) != str(step4_contract["step2_contract_sha256"]):
            raise RuntimeError("Step-5R2 found a different Step-2 prerequisite")
        inputs = _worker_inputs(step3_path.parent, step2_path.parent)
        _verify_frozen_input_artifacts(
            inputs, step3_path.parent, step2_path.parent, step2_contract
        )
        _write_json(output / "step4r2_lineage.json", lineage)
        worker_output = output / "worker"
        job = {
            "schema_version": 1,
            "mode": "step5r2",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output),
            "official_source": str(source),
            "official_commit": step3_contract["official_commit"],
            "inputs": inputs,
            "step4r2_lineage": lineage,
            "selected_mixture_weight": float(step4_contract["selected_mixture_weight"]),
            "cities": protocol["cities"],
            "development_folds": protocol["development_folds"],
            "global_context": protocol["global_context"],
            "model": protocol["model"],
            "city_backoff": protocol["city_backoff"],
            "lexical_graph": protocol["lexical_graph"],
            "graph_prototype_projection": protocol[
                "graph_prototype_projection"
            ],
            "evaluation": protocol["evaluation"],
            "gates": protocol["gates"],
            "reported_graph_ablation_nll_tradeoff": protocol[
                "reported_graph_ablation_nll_tradeoff"
            ],
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
            raise RuntimeError("Step-5R2 R5 exact-runtime worker failed")

        worker = json.loads(
            (worker_output / "worker_result.json").read_text(encoding="utf-8-sig")
        )
        summaries = pd.read_csv(
            worker_output / "candidate_summary.csv", encoding="utf-8-sig"
        )
        fold_metrics = pd.read_csv(
            worker_output / "paired_candidate_fold_metrics.csv",
            encoding="utf-8-sig",
        )
        selected_quota = worker.get("selected_prototype_quota")
        selected = (
            summaries.loc[
                summaries["prototype_quota"] == int(selected_quota)
            ].iloc[0]
            if selected_quota is not None
            else None
        )
        gates = protocol["gates"]
        checkpoint_manifest = worker["checkpoint_manifest"]

        def observed(name: str) -> Any:
            return None if selected is None else selected[name]

        checks: list[tuple[str, bool, Any]] = [
            ("step4r2_contract_and_manifest_verified", True, lineage["manifest_identity"]),
            ("step4r2_configuration_unchanged", sha256_file(step4_config) == step4_contract["config_sha256"], step4_contract["config_sha256"]),
            ("step4r2_selected_mixture_frozen", float(worker["selected_city_mixture_weight"]) == float(protocol["city_backoff"]["fixed_mixture_weight"]), worker["selected_city_mixture_weight"]),
            ("exact_runtime_unchanged", _validate_runtime(worker_output, step3_specification), str(runtime)),
            ("official_source_unchanged", worker["official_class_loaded_from"].casefold().startswith(str(source).casefold()), worker["official_class_loaded_from"]),
            ("step4r2_predictions_reproduced", float(worker["maximum_step4r2_reproduction_error"]) <= float(gates["step4r2_reproduction_tolerance"]), worker["maximum_step4r2_reproduction_error"]),
            ("probability_simplex", float(worker["maximum_probability_sum_error"]) <= float(gates["probability_tolerance"]), worker["maximum_probability_sum_error"]),
            ("fold_graph_support_complete", int(worker["minimum_graph_active_words"]) >= int(protocol["lexical_graph"]["minimum_active_words"]), worker["minimum_graph_active_words"]),
            ("prototype_oracle_quality_feasible", float(worker["prototype_oracle_median_heldout_npmi"]) >= float(gates["minimum_median_heldout_npmi"]) and float(worker["prototype_oracle_median_standard_npmi"]) >= float(gates["minimum_median_standard_reference_npmi"]) and float(worker["prototype_oracle_median_standard_cv"]) >= float(gates["minimum_median_standard_reference_cv"]) and float(worker["prototype_oracle_topic_stability"]["mean"]) >= float(gates["minimum_cross_fold_topic_stability"]), {"heldout_npmi": worker["prototype_oracle_median_heldout_npmi"], "standard_npmi": worker["prototype_oracle_median_standard_npmi"], "standard_cv": worker["prototype_oracle_median_standard_cv"], "stability": worker["prototype_oracle_topic_stability"]["mean"]}),
            ("zero_strength_exact_parent_identity", float(worker["maximum_zero_strength_identity_error"]) <= float(gates["maximum_zero_strength_identity_error"]), worker["maximum_zero_strength_identity_error"]),
            ("parent_parameters_and_buffers_frozen", bool(worker["all_parent_parameters_and_buffers_exactly_unchanged"]), worker["all_parent_parameters_and_buffers_exactly_unchanged"]),
            ("closed_form_no_optimizer", int(worker["optimizer_steps"]) == 0, worker["optimizer_steps"]),
            ("projection_evaluation_count_exact", int(worker["paired_control_evaluations"]) == 5 and int(worker["candidate_projection_evaluations"]) == 15 and int(worker["total_projection_evaluations"]) == 20, worker["total_projection_evaluations"]),
            ("no_trainable_parameter_group", worker["trainable_parameter_group"] == "none_closed_form_projection_on_frozen_affinity", worker["trainable_parameter_group"]),
            ("candidate_selection_succeeded", selected is not None, selected_quota),
            ("heldout_npmi_practical_effect", selected is not None and float(observed("median_heldout_npmi_improvement")) >= float(gates["minimum_median_heldout_npmi_improvement"]), observed("median_heldout_npmi_improvement")),
            ("heldout_npmi_fold_consistency", selected is not None and float(observed("positive_npmi_fold_fraction")) >= float(gates["minimum_positive_npmi_fold_fraction"]), observed("positive_npmi_fold_fraction")),
            ("heldout_cv_practical_effect", selected is not None and float(observed("median_heldout_cv_improvement")) >= float(gates["minimum_median_heldout_cv_improvement"]), observed("median_heldout_cv_improvement")),
            ("heldout_cv_fold_consistency", selected is not None and float(observed("positive_cv_fold_fraction")) >= float(gates["minimum_positive_cv_fold_fraction"]), observed("positive_cv_fold_fraction")),
            ("heldout_npmi_absolute_floor", selected is not None and float(observed("median_heldout_npmi")) >= float(gates["minimum_median_heldout_npmi"]), observed("median_heldout_npmi")),
            ("standard_npmi_absolute_floor", selected is not None and float(observed("median_standard_reference_npmi")) >= float(gates["minimum_median_standard_reference_npmi"]), observed("median_standard_reference_npmi")),
            ("standard_cv_absolute_floor", selected is not None and float(observed("median_standard_reference_cv")) >= float(gates["minimum_median_standard_reference_cv"]), observed("median_standard_reference_cv")),
            ("zero_joint_pair_reduction", selected is not None and float(observed("median_zero_joint_pair_reduction")) >= float(gates["minimum_median_zero_joint_pair_reduction"]), observed("median_zero_joint_pair_reduction")),
            ("displayed_word_training_support", selected is not None and float(observed("minimum_displayed_train_support_fraction")) >= float(gates["minimum_displayed_train_support_fraction"]), observed("minimum_displayed_train_support_fraction")),
            ("displayed_word_graph_support", selected is not None and float(observed("minimum_displayed_graph_support_fraction")) >= float(gates["minimum_displayed_graph_support_fraction"]), observed("minimum_displayed_graph_support_fraction")),
            ("displayed_positive_graph_pairs", selected is not None and float(observed("minimum_displayed_positive_graph_pair_fraction")) >= float(gates["minimum_displayed_positive_graph_pair_fraction"]), observed("minimum_displayed_positive_graph_pair_fraction")),
            (
                "graph_ablation_nll_tradeoff_reported",
                selected is not None
                and isinstance(observed("failed_reported_nll_diagnostics"), str)
                and np.isfinite(
                    [
                        float(observed("median_completion_nll_reduction")),
                        float(observed("minimum_completion_nll_reduction")),
                        float(observed("median_nll_reduction_vs_step4r2")),
                        float(observed("minimum_nll_reduction_vs_step4r2")),
                    ]
                ).all(),
                None
                if selected is None
                else {
                    "median": float(observed("median_nll_reduction_vs_step4r2")),
                    "worst_fold": float(
                        observed("minimum_nll_reduction_vs_step4r2")
                    ),
                    "failed_r4_references": observed(
                        "failed_reported_nll_diagnostics"
                    ),
                },
            ),
            ("city_vs_matched_pooling_effect_preserved", selected is not None and float(observed("median_city_vs_pooled_reduction")) >= float(gates["minimum_median_city_vs_pooled_reduction"]), observed("median_city_vs_pooled_reduction")),
            ("city_vs_matched_pooling_consistency_preserved", selected is not None and float(observed("positive_city_vs_pooled_fold_fraction")) >= float(gates["minimum_positive_city_vs_pooled_fold_fraction"]), observed("positive_city_vs_pooled_fold_fraction")),
            ("city_vs_parent_effect_preserved", selected is not None and float(observed("median_city_vs_parent_reduction")) >= float(gates["minimum_median_city_vs_parent_reduction"]), observed("median_city_vs_parent_reduction")),
            ("city_vs_parent_consistency_preserved", selected is not None and float(observed("positive_city_vs_parent_fold_fraction")) >= float(gates["minimum_positive_city_vs_parent_fold_fraction"]), observed("positive_city_vs_parent_fold_fraction")),
            ("topic_diversity_guardrail", selected is not None and float(observed("minimum_topic_diversity")) >= float(gates["minimum_topic_diversity"]), observed("minimum_topic_diversity")),
            ("top_word_redundancy_guardrail", selected is not None and float(observed("maximum_top_word_redundancy")) <= float(gates["maximum_top_word_redundancy"]), observed("maximum_top_word_redundancy")),
            ("cross_fold_topic_stability_floor", selected is not None and float(observed("cross_fold_topic_stability")) >= float(gates["minimum_cross_fold_topic_stability"]), observed("cross_fold_topic_stability")),
            ("stability_paired_guardrail", selected is not None and float(observed("cross_fold_topic_stability")) >= float(observed("paired_control_topic_stability")) - float(gates["maximum_stability_regression_vs_paired_control"]), None if selected is None else {"full": float(observed("cross_fold_topic_stability")), "control": float(observed("paired_control_topic_stability"))}),
            ("prototype_core_recall", selected is not None and float(observed("minimum_prototype_core_recall")) >= float(gates["minimum_prototype_core_recall"]), observed("minimum_prototype_core_recall")),
            ("complete_top10_projection_selected", selected_quota == int(protocol["evaluation"]["top_words"]), selected_quota),
            ("nonprototype_columns_exact", selected is not None and float(observed("maximum_nonprototype_change")) <= float(gates["maximum_nonprototype_change"]), observed("maximum_nonprototype_change")),
            ("projection_column_simplex", selected is not None and float(observed("maximum_runtime_column_sum_error")) <= float(gates["maximum_projection_column_sum_error"]), observed("maximum_runtime_column_sum_error")),
            ("projection_nonnegative", selected is not None and float(observed("minimum_runtime_probability")) >= float(gates["minimum_projection_probability"]), observed("minimum_runtime_probability")),
            ("projection_minimum_total_variation", selected is not None and float(observed("maximum_projection_optimality_error")) <= float(gates["maximum_projection_optimality_error"]), observed("maximum_projection_optimality_error")),
            ("projection_support_bounded", selected is not None and int(observed("maximum_changed_columns")) <= int(gates["maximum_changed_columns"]), observed("maximum_changed_columns")),
            ("selected_parent_state_frozen", selected is not None and int(observed("maximum_parent_state_change")) == 0, observed("maximum_parent_state_change")),
            ("one_affinity_for_likelihood_and_reporting", selected is not None and _is_true(observed("same_affinity_for_likelihood_and_reporting")), observed("same_affinity_for_likelihood_and_reporting")),
            ("selected_checkpoints_present_and_hashed", selected is not None and len(checkpoint_manifest) == 5 and all(Path(row["path"]).is_file() and sha256_file(Path(row["path"])) == str(row["sha256"]) for row in checkpoint_manifest), len(checkpoint_manifest)),
            ("all_reported_numeric_metrics_finite", bool(np.isfinite(fold_metrics.select_dtypes(include=[np.number]).to_numpy()).all()), True),
            ("validation_access_zero", int(worker["validation_documents_used"]) == 0, worker["validation_documents_used"]),
            ("test_access_zero", int(worker["test_documents_used"]) == 0, worker["test_documents_used"]),
            ("label_access_zero", int(worker["labels_used"]) == 0, worker["labels_used"]),
            ("comparator_access_zero", int(worker["baseline_or_sota_outputs_used"]) == 0, worker["baseline_or_sota_outputs_used"]),
        ]
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
        (output / "resolved_v6_step05r2.yaml").write_text(
            yaml.safe_dump(
                {"schema_version": 1, "v6": {"step5r2": protocol}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        contract = {
            "schema_version": 1,
            "status": status,
            "scope": SCOPE,
            "implementation_version": IMPLEMENTATION,
            "readiness": "READY_FOR_V6_STEP6_VALIDATION_GATE" if status == "PASS" else "BLOCKED_REVISE_STEP5R2_BEFORE_VALIDATION",
            "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
            "hard_checks_total": len(hard_rows),
            "failed_checks": failures,
            "data_key": step4_contract["data_key"],
            "official_commit": step3_contract["official_commit"],
            "step4r2_contract_sha256": sha256_file(step4_path),
            "step3_contract_sha256": sha256_file(step3_path),
            "step2_contract_sha256": sha256_file(step2_path),
            "selected_city_mixture_weight": worker["selected_city_mixture_weight"],
            "selected_prototype_quota": selected_quota,
            "candidate_gate_failures": worker["candidate_gate_failures"],
            "candidate_reported_graph_ablation_nll_tradeoffs": worker[
                "reported_graph_ablation_nll_tradeoffs"
            ],
            "selection_policy_audit_sha256": protocol["revision_basis"][
                "audit_receipt_sha256"
            ],
            "median_heldout_npmi_improvement": observed("median_heldout_npmi_improvement"),
            "median_heldout_cv_improvement": observed("median_heldout_cv_improvement"),
            "median_heldout_npmi": observed("median_heldout_npmi"),
            "median_standard_reference_npmi": observed("median_standard_reference_npmi"),
            "median_standard_reference_cv": observed("median_standard_reference_cv"),
            "median_completion_nll_reduction": observed("median_completion_nll_reduction"),
            "median_nll_reduction_vs_step4r2": observed("median_nll_reduction_vs_step4r2"),
            "reported_graph_ablation_nll_tradeoff": None if selected is None else {
                "median_nll_change_vs_graph_removed_step4r2": float(observed("median_nll_reduction_vs_step4r2")),
                "worst_fold_nll_change_vs_graph_removed_step4r2": float(observed("minimum_nll_reduction_vs_step4r2")),
                "failed_r4_reference_diagnostics": observed("failed_reported_nll_diagnostics"),
                "affects_r5_candidate_eligibility": False,
            },
            "median_city_vs_matched_pooling_reduction": observed("median_city_vs_pooled_reduction"),
            "median_mean_vocabulary_column_total_variation": observed("median_mean_vocabulary_column_total_variation"),
            "optimizer_steps": worker["optimizer_steps"],
            "validation_documents_used": 0,
            "test_documents_used": 0,
            "human_labels_used": 0,
            "baseline_outputs_used": 0,
            "sota_outputs_used": 0,
            "config_sha256": sha256_file(config_path),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(output / "step05r2_contract.json", contract)
        _write_json(output / "generated_artifact_manifest.json", _artifact_manifest(output))
        report(
            f"[V6 Step 5R2] {status} | hard checks="
            f"{contract['hard_checks_passed']}/{contract['hard_checks_total']} "
            f"selected-quota={selected_quota}"
        )
        if selected is not None:
            report(
                "[V6 Step 5R2] effects | median heldout dNPMI="
                f"{float(selected['median_heldout_npmi_improvement']):+.6f} "
                "dC_v="
                f"{float(selected['median_heldout_cv_improvement']):+.6f} "
                "dNLL="
                f"{float(selected['median_completion_nll_reduction']):+.6f}"
            )
            report(
                "[V6 Step 5R2] distortion | median vocabulary-column TV="
                f"{float(selected['median_mean_vocabulary_column_total_variation']):.6f}"
            )
        report(f"[V6 Step 5R2] contract={output / 'step05r2_contract.json'}")
        return_zip = _make_return_zip(root, output, log_path)
        report(f"[V6 Step 5R2] return-zip={return_zip}")
        _make_return_zip(root, output, log_path)
        if failures:
            raise AssertionError(
                f"Step-5R2 R5 failed gates: {failures}; return v6_step05r2_results.zip"
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
