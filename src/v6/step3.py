"""Execute V6 Step 3: exact official EnCOT source/runtime/adapter gate."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping
import zipfile

import numpy as np
from scipy import sparse
import yaml

from src.v6.data_contract import (
    dense_logical_hash,
    load_frozen_development_inputs,
    sha256_file,
    sha256_object,
    sparse_logical_hash,
)
from src.v6.official_encot import (
    locate_exact_runtime,
    locate_or_clone_official_source,
)
from src.v6.predictive_evaluation import hand_calculated_self_test


IMPLEMENTATION_VERSION = "v6_step3_official_encot_exact_bridge_r1"
SCOPE = "official_encot_source_runtime_adapter_and_training_smoke_no_quality_selection"


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty required audit table: {path.name}")
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
    if document.get("schema_version") != 1:
        raise ValueError("the V6 Step-3 configuration has an unsupported schema")
    protocol = document.get("v6", {}).get("step3")
    if not isinstance(protocol, dict):
        raise ValueError("configuration lacks v6.step3")
    if protocol.get("scope") != SCOPE:
        raise ValueError("the V6 Step-3 scope is not the frozen official-parent scope")
    if protocol.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("configuration and Step-3 implementation versions disagree")
    if protocol.get("runtime", {}).get("creation_or_installation_allowed") is not False:
        raise ValueError("Step 3 is not allowed to create or modify an environment")
    return protocol


def _find_passing_step2(v6_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (v6_root / "outputs" / "v6" / "step02_evaluation_firewall").glob(
            "*/step02_contract.json"
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
            and contract.get("readiness")
            == "READY_FOR_V6_STEP3_OFFICIAL_ENCOT_REPRODUCTION"
            and int(contract.get("hard_checks_passed", -1))
            == int(contract.get("hard_checks_total", -2))
        ):
            return path, contract
    raise RuntimeError(
        "No passing V6 Step-2 contract was found under "
        "outputs/v6/step02_evaluation_firewall"
    )


def _verify_step2_lock(
    root: Path, contract_path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    output = contract_path.parent
    source_results: dict[str, dict[str, Any]] = {}
    for relative, expected in contract["evaluator_source_sha256"].items():
        path = root / str(relative)
        observed = sha256_file(path)
        source_results[str(relative)] = {
            "expected": str(expected),
            "observed": observed,
            "matches": observed == str(expected),
        }
    config_observed = sha256_file(root / "configs" / "v6" / "step02.yaml")
    evaluator_observed = sha256_file(output / "evaluator_and_ablation_lock.json")
    fold_bytes = (output / "training_group_folds.csv").read_bytes()
    fold_observed = sha256(fold_bytes).hexdigest()
    # Step 2's fold_plan_sha256 is a logical object hash, not the CSV byte hash.
    fold_rows = []
    with (output / "training_group_folds.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        fold_rows = list(csv.DictReader(stream))
    fold_logical = sha256_object(
        [
            {
                "post_id": row["post_id"],
                "city": row["city"],
                "split": row["split"],
                "group_sha256": row["group_sha256"],
                "fold": int(row["fold"]),
            }
            for row in fold_rows
        ]
    )
    return {
        "schema_version": 1,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "step2_source_files": source_results,
        "all_step2_source_files_unchanged": all(
            item["matches"] for item in source_results.values()
        ),
        "step2_config_expected_sha256": contract["config_sha256"],
        "step2_config_observed_sha256": config_observed,
        "step2_config_unchanged": config_observed == contract["config_sha256"],
        "evaluator_lock_expected_sha256": contract["evaluator_lock_sha256"],
        "evaluator_lock_observed_sha256": evaluator_observed,
        "evaluator_lock_unchanged": evaluator_observed
        == contract["evaluator_lock_sha256"],
        "fold_plan_expected_logical_sha256": contract["fold_plan_sha256"],
        "fold_plan_observed_logical_sha256": fold_logical,
        "fold_plan_unchanged": fold_logical == contract["fold_plan_sha256"],
        "fold_csv_byte_sha256": fold_observed,
    }


def _aligned_embeddings(
    source_root: Path, train_post_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    evidence = source_root / "data" / "processed" / "evidence"
    with np.load(evidence / "E_document.npz", allow_pickle=False) as archive:
        all_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        all_ids = archive["post_ids"].astype(str).tolist()
    lookup = {post_id: index for index, post_id in enumerate(all_ids)}
    if len(lookup) != len(all_ids):
        raise ValueError("the frozen document embedding bank contains duplicate IDs")
    missing = [post_id for post_id in train_post_ids if post_id not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} training rows lack frozen embeddings")
    train_embeddings = all_embeddings[
        [lookup[post_id] for post_id in train_post_ids]
    ]
    with np.load(evidence / "S_vocabulary.npz", allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        tokens = archive["tokens"].astype(str).tolist()
    return train_embeddings, word_embeddings, tokens


def _version_matches(observed: Any, expected: Any) -> bool:
    return str(observed).split("+", maxsplit=1)[0] == str(expected)


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
    destination = root / "v6_step03_results.zip"
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("step03_official_encot") / output.name
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, (prefix / path.relative_to(output)).as_posix())
        if log_path.is_file():
            archive.write(log_path, "logs/v6_step03.log")
    return destination


def run_v6_step3(
    *,
    project_root: Path,
    config_path: Path,
    source_root: Path | None = None,
    encot_source: Path | None = None,
    encot_python: Path | None = None,
) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError(
            "V6 Step 3 refuses to run outside an isolated directory whose name contains 'V6'"
        )
    protocol = _load_protocol(config_path)
    source = (
        source_root.expanduser().resolve()
        if source_root is not None
        else (root / str(protocol["source_project_root"])).resolve()
    )
    if source == root or "v6" in source.name.casefold():
        raise RuntimeError("the trusted source must be the separate non-V6 SC-HTM project")

    output = root / "outputs" / "v6" / "step03_official_encot" / _utc_id()
    output.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / "v6_step03.log"
    report = _Reporter(log_path)
    report(f"[V6 Step 3] START | implementation={IMPLEMENTATION_VERSION}")
    report(
        "[V6 Step 3] scope=official-source+exact-runtime+adapter-parity+"
        "train-only-smoke | quality-selection=0 validation-fit=0 test=0 labels=0"
    )

    step2_path, step2_contract = _find_passing_step2(root)
    prerequisite = protocol["prerequisite"]
    if any(
        (
            step2_contract.get("status") != prerequisite["status"],
            step2_contract.get("readiness") != prerequisite["readiness"],
            step2_contract.get("implementation_version")
            != prerequisite["implementation_version"],
            step2_contract.get("data_key") != prerequisite["data_key"],
        )
    ):
        raise RuntimeError("the passing Step-2 contract does not match the Step-3 lock")
    step2_lock = _verify_step2_lock(root, step2_path, step2_contract)
    _write_json(output / "step02_prerequisite_audit.json", step2_lock)
    if not all(
        (
            step2_lock["all_step2_source_files_unchanged"],
            step2_lock["step2_config_unchanged"],
            step2_lock["evaluator_lock_unchanged"],
            step2_lock["fold_plan_unchanged"],
        )
    ):
        raise RuntimeError("a frozen Step-2 source/config/evaluator/fold artifact changed")

    data = load_frozen_development_inputs(
        source,
        expected_cities=tuple(map(str, protocol["cities"])),
        embedding_norm_tolerance=float(protocol["embedding_norm_tolerance"]),
    )
    if data.data_key != step2_contract["data_key"]:
        raise RuntimeError("current frozen development inputs do not match Step 2")
    train_ids = data.train_rows["post_id"].astype(str).tolist()
    train_embeddings, word_embeddings, embedded_tokens = _aligned_embeddings(
        source, train_ids
    )
    vocabulary_tokens = data.vocabulary["token"].astype(str).tolist()
    if embedded_tokens != vocabulary_tokens:
        raise ValueError("frozen word embeddings and vocabulary are not aligned")
    if train_embeddings.shape != (len(data.train_rows), 768):
        raise ValueError("unexpected training document embedding shape")
    if word_embeddings.shape != (len(vocabulary_tokens), 768):
        raise ValueError("unexpected vocabulary embedding shape")

    input_dir = output / "worker_input"
    input_dir.mkdir()
    sparse.save_npz(input_dir / "train_counts.npz", data.train_counts, compressed=True)
    np.savez_compressed(
        input_dir / "train_embeddings.npz",
        embeddings=train_embeddings,
        post_ids=np.asarray(train_ids, dtype="U64"),
    )
    np.savez_compressed(
        input_dir / "word_embeddings.npz",
        embeddings=word_embeddings,
        tokens=np.asarray(vocabulary_tokens, dtype=str),
    )
    input_manifest = {
        "schema_version": 1,
        "data_key": data.data_key,
        "fit_partition": "original_training_only",
        "training_documents": int(data.train_counts.shape[0]),
        "validation_documents_in_model_fit": 0,
        "test_documents_accessed": 0,
        "labels_accessed": 0,
        "vocabulary_size": int(data.train_counts.shape[1]),
        "embedding_dimension": int(word_embeddings.shape[1]),
        "train_counts_logical_sha256": sparse_logical_hash(data.train_counts),
        "train_embeddings_logical_sha256": dense_logical_hash(train_embeddings),
        "word_embeddings_logical_sha256": dense_logical_hash(word_embeddings),
        "train_post_ids_sha256": sha256_object(train_ids),
        "files": {
            path.name: sha256_file(path)
            for path in sorted(input_dir.iterdir())
            if path.is_file()
        },
    }
    input_manifest["identity"] = sha256_object(input_manifest)
    _write_json(output / "training_only_input_manifest.json", input_manifest)
    report(
        "[V6 Step 3] data | "
        f"train={len(data.train_rows)} vocabulary={len(vocabulary_tokens)} "
        f"embedding-width={word_embeddings.shape[1]} data-key={data.data_key[:12]}"
    )

    official_source, source_receipt = locate_or_clone_official_source(
        v6_root=root,
        source_root=source,
        explicit=encot_source,
        specification=protocol["official_source"],
    )
    _write_json(output / "official_source_receipt.json", source_receipt)
    report(
        "[V6 Step 3] official source | "
        f"commit={source_receipt['commit'][:12]} tree={source_receipt['git_tree'][:12]} "
        f"acquisition={source_receipt['acquisition']}"
    )

    runtime_python, runtime_selection = locate_exact_runtime(
        v6_root=root,
        source_root=source,
        explicit=encot_python,
        specification=protocol["runtime"],
    )
    _write_json(output / "runtime_selection.json", runtime_selection)
    report(
        "[V6 Step 3] runtime | REUSE exact existing interpreter | "
        f"python={runtime_python} installs=0"
    )

    worker_output = output / "worker"
    job = {
        "schema_version": 1,
        "official_source": str(official_source),
        "worker_output": str(worker_output),
        "train_counts": str(input_dir / "train_counts.npz"),
        "train_embeddings": str(input_dir / "train_embeddings.npz"),
        "word_embeddings": str(input_dir / "word_embeddings.npz"),
        "input_identity": input_manifest["identity"],
        "global_context": protocol["global_context"],
        "official_parent": protocol["official_parent"],
        "parity": protocol["parity"],
        "prohibited": protocol["prohibited"],
    }
    job_path = output / "worker_job.json"
    _write_json(job_path, job)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    completed = subprocess.run(
        [str(runtime_python), "-m", "src.v6.step3_worker", "--job", str(job_path)],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stdout.strip():
        for line in completed.stdout.strip().splitlines():
            report(line)
    if completed.returncode != 0:
        failure = {
            "schema_version": 1,
            "return_code": int(completed.returncode),
            "stderr": completed.stderr,
            "runtime_python": str(runtime_python),
        }
        _write_json(output / "worker_failure.json", failure)
        raise RuntimeError(
            "The exact-runtime EnCOT worker failed. Return worker_failure.json and "
            "logs/v6_step03.log; Step 4 must not start."
        )
    if completed.stderr.strip():
        (output / "worker_stderr.txt").write_text(completed.stderr, encoding="utf-8")

    runtime = json.loads(
        (worker_output / "runtime_receipt.json").read_text(encoding="utf-8")
    )
    interface = json.loads(
        (worker_output / "official_interface_audit.json").read_text(encoding="utf-8")
    )
    parity = json.loads(
        (worker_output / "bridge_parity.json").read_text(encoding="utf-8")
    )
    context = json.loads(
        (worker_output / "training_only_global_context_contract.json").read_text(
            encoding="utf-8"
        )
    )
    smoke = json.loads(
        (worker_output / "real_training_smoke.json").read_text(encoding="utf-8")
    )
    runtime_lock = protocol["runtime"]
    runtime_packages_match = all(
        _version_matches(runtime["packages"].get(name), expected)
        for name, expected in runtime_lock["packages"].items()
    )
    corrected = protocol["corrected_backbone_interface"]
    correction_receipt = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_V6_REAL_QUALITY_SELECTION",
        **corrected,
        "reason": (
            "The pinned official source normalizes its K-by-V affinity over topics. "
            "The Step-1 per-topic softmax was a synthetic residual harness and is "
            "not used as the EnCOT parent decoder."
        ),
        "step1_contract_retained_for": [
            "weighted_zero_centered_low_rank_residual_geometry",
            "adaptive_gate_gradient_activity",
            "rank_aware_soft_top10_gradient_and_numpy_oracle",
        ],
        "step3_superseding_interface_test": (
            "alpha_zero_city_bridge_equals_pinned_official_EnCOT_decoder"
        ),
    }
    _write_json(output / "method_interface_correction.json", correction_receipt)
    predictive_self_test = hand_calculated_self_test()
    predictive_addendum = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_V6_REAL_QUALITY_RESULTS",
        "reason": (
            "EnCOT's native predictive distribution is decoder_BN followed by a "
            "vocabulary softmax, not a row-normalized theta-times-beta surrogate."
        ),
        "inference_input": "observed_completion_counts_only",
        "scoring_input": "target_completion_counts_only",
        "scored_object": "exact_native_decoder_word_probabilities",
        "generic_step2_mixture_scorer_modified": False,
        "implementation": "src/v6/predictive_evaluation.py",
        "implementation_sha256": sha256_file(
            root / "src" / "v6" / "predictive_evaluation.py"
        ),
        "hand_calculated_self_test": predictive_self_test,
    }
    predictive_addendum["identity"] = sha256_object(predictive_addendum)
    _write_json(output / "predictive_evaluator_addendum.json", predictive_addendum)

    checks: list[tuple[str, bool]] = [
        ("step2_contract_passes", step2_contract["status"] == "PASS"),
        ("step2_data_key_exact", data.data_key == prerequisite["data_key"]),
        ("step2_source_files_unchanged", step2_lock["all_step2_source_files_unchanged"]),
        ("step2_config_unchanged", step2_lock["step2_config_unchanged"]),
        ("step2_evaluator_lock_unchanged", step2_lock["evaluator_lock_unchanged"]),
        ("step2_fold_plan_unchanged", step2_lock["fold_plan_unchanged"]),
        ("training_count_rows_exact", data.train_counts.shape[0] == 620),
        ("vocabulary_size_exact", data.train_counts.shape[1] == 1148),
        ("train_embedding_alignment_exact", train_embeddings.shape == (620, 768)),
        ("word_embedding_alignment_exact", word_embeddings.shape == (1148, 768)),
        ("official_commit_exact", source_receipt["commit"] == protocol["official_source"]["commit"]),
        ("official_git_tree_exact", source_receipt["git_tree"] == protocol["official_source"]["git_tree"]),
        ("official_required_files_exact", source_receipt["required_files_verified"] == len(protocol["official_source"]["required_files_sha256"])),
        ("official_source_clean", source_receipt["source_tree_clean"] is True),
        ("official_source_not_redistributed", source_receipt["source_redistributed_in_v6_package"] is False),
        ("official_main_not_executed", interface["official_main_executed"] is False),
        ("official_shell_scripts_not_executed", interface["official_shell_scripts_executed"] is False),
        ("official_class_loaded_from_pinned_file", interface["loaded_class_is_exact_pinned_file"] is True),
        ("official_beta_axis_disclosed", interface["official_beta_softmax_axis_is_topics_dim0"] is True),
        ("official_theta_product_disclosed", interface["official_theta_is_unrenormalized_elementwise_product"] is True),
        ("official_decoder_semantics_disclosed", interface["official_decoder_uses_vocabulary_softmax"] is True),
        ("exact_existing_runtime_selected", runtime_selection["exact_package_lock_matched"] is True),
        ("runtime_python_3_10", runtime["python"][:2] == [3, 10]),
        ("runtime_cuda_available", runtime["cuda_available"] is True),
        ("runtime_packages_exact", runtime_packages_match),
        ("no_environment_created", runtime["environment_created"] is False),
        ("no_packages_installed_or_changed", runtime["packages_installed_or_changed"] is False),
        ("all_bridge_parity_checks", parity["passed"] is True),
        ("extensions_off_exact_parent_identity", parity["checks"]["extensions_off_equals_official_affinity"] is True),
        ("adapter_one_step_state_parity", parity["checks"]["adapter_one_step_state_matches"] is True),
        ("reporting_rank_preservation", parity["checks"]["reporting_normalization_preserves_top_word_ranks"] is True),
        ("corrected_extension_interface_locked", corrected["official_affinity_normalization_axis"] == "topics_for_each_word" and corrected["city_residual_insertion"] == "before_official_topic_axis_softmax" and corrected["city_residual_identifiability"] == "center_over_topics_then_weighted_center_over_cities" and corrected["lexical_rank_score"] == "standardized_log_row_normalized_affinity" and corrected["predictive_scoring"] == "exact_native_decoder_probabilities_inferred_from_observed_half"),
        ("native_predictive_evaluator_hand_fixture", predictive_self_test["passed"] is True),
        ("global_context_training_only", context["fit_partition"] == "original_training_only"),
        ("global_context_uses_no_validation", context["validation_documents_fit"] == 0),
        ("global_context_uses_no_test", context["test_documents_fit"] == 0),
        ("global_context_uses_no_labels", context["labels_used"] is False),
        ("global_context_all_documents_assigned", context["cluster_size_sum"] == 620),
        ("real_smoke_train_only", smoke["training_documents"] == 620 and smoke["validation_documents_used_for_fit"] == 0 and smoke["test_documents_used_for_fit"] == 0),
        ("real_smoke_losses_finite", smoke["all_loss_components_finite"] is True),
        ("real_smoke_gradients_finite_nonzero", smoke["all_gradient_norms_finite"] is True and smoke["nonzero_gradient_observed"] is True),
        ("real_smoke_parameters_update", smoke["parameter_state_changed"] is True),
        ("real_smoke_probability_simplex", smoke["decoded_probability_sum_error"] <= 5.0e-6 and smoke["minimum_decoded_probability"] >= 0.0),
        ("official_beta_column_normalization", smoke["official_beta_column_sum_error"] <= 5.0e-6),
        ("reporting_beta_row_normalization", smoke["reporting_beta_row_sum_error"] <= 5.0e-6),
        ("no_quality_metrics_computed", smoke["quality_metrics_computed"] == 0),
        ("no_hyperparameters_selected", smoke["hyperparameters_selected"] == 0),
        ("validation_not_used_for_fit", protocol["prohibited"]["validation_rows_used_for_model_fit"] == 0),
        ("test_counts_not_accessed", data.leakage_report["test_count_matrix_accessed"] == 0),
        ("test_text_not_accessed", data.leakage_report["test_text_accessed"] == 0),
        ("labels_not_accessed", data.leakage_report["human_annotations_accessed"] == 0),
        ("prior_quality_and_comparators_not_accessed", data.leakage_report["previous_quality_metrics_accessed"] == 0 and data.leakage_report["baseline_outputs_accessed"] == 0 and data.leakage_report["sota_outputs_accessed"] == 0),
    ]
    hard_rows = [
        {
            "check": name,
            "required": True,
            "observed": bool(value),
            "status": "PASS" if value else "FAIL",
        }
        for name, value in checks
    ]
    _write_csv(output / "hard_checks.csv", hard_rows)
    failures = [row["check"] for row in hard_rows if row["status"] != "PASS"]
    status = "PASS" if not failures else "FAIL"
    resolved = output / "resolved_v6_step03.yaml"
    resolved.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "v6": {"step3": protocol}}, sort_keys=False
        ),
        encoding="utf-8",
    )
    contract = {
        "schema_version": 1,
        "status": status,
        "scope": SCOPE,
        "implementation_version": IMPLEMENTATION_VERSION,
        "readiness": (
            "READY_FOR_V6_STEP4_CITY_RESIDUAL_INTEGRATION"
            if status == "PASS"
            else "BLOCKED_FIX_STEP3_BEFORE_CONTINUING"
        ),
        "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
        "hard_checks_total": len(hard_rows),
        "failed_checks": failures,
        "data_key": data.data_key,
        "step2_contract_sha256": sha256_file(step2_path),
        "official_repository": protocol["official_source"]["repository"],
        "official_commit": source_receipt["commit"],
        "official_git_tree": source_receipt["git_tree"],
        "official_source_tree_clean": True,
        "official_source_redistributed": False,
        "exact_runtime_reused": True,
        "environment_created": False,
        "packages_installed_or_changed": False,
        "training_documents_used": int(smoke["training_documents"]),
        "validation_documents_used_for_fit": 0,
        "test_count_matrix_accessed": 0,
        "test_text_accessed": 0,
        "human_labels_accessed": 0,
        "prior_quality_metrics_accessed": 0,
        "baseline_outputs_accessed": 0,
        "sota_outputs_accessed": 0,
        "quality_metrics_computed": 0,
        "hyperparameters_selected": 0,
        "official_parent_real_training_smoke_fits": 1,
        "official_parent_quality_fits": 0,
        "adapter_parity_passed": bool(parity["passed"]),
        "exact_extensions_off_parent_identity_passed": bool(
            parity["checks"]["extensions_off_equals_official_affinity"]
        ),
        "method_interface_correction_locked": True,
        "native_predictive_evaluator_addendum_locked": True,
        "native_predictive_evaluator_sha256": predictive_addendum[
            "implementation_sha256"
        ],
        "config_sha256": sha256_file(config_path),
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    _write_json(output / "step03_contract.json", contract)
    manifest = _artifact_manifest(output)
    _write_json(output / "generated_artifact_manifest.json", manifest)

    if failures:
        report(
            f"[V6 Step 3] FAIL | hard checks={len(checks)-len(failures)}/{len(checks)} "
            f"failed={failures}"
        )
        raise AssertionError("V6 Step 3 failed one or more hard checks")
    report(
        "[V6 Step 3] PASS | "
        f"hard checks={len(checks)}/{len(checks)} | parity=PASS | train-only-smoke=PASS"
    )
    report(f"[V6 Step 3] contract={output / 'step03_contract.json'}")
    return_zip = root / "v6_step03_results.zip"
    report(f"[V6 Step 3] return-zip={return_zip}")
    _make_return_zip(root, output, log_path)
    return output
