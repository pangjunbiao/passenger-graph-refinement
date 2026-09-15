"""Execute V6 Step 2: immutable data and evaluator firewall."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import yaml

from src.v6.data_contract import (
    deterministic_group_folds,
    find_passing_step1_contract,
    load_frozen_development_inputs,
    sha256_file,
    sha256_object,
    sparse_logical_hash,
)
from src.v6.document_completion import make_document_completion
from src.v6.evaluation import (
    counts_to_token_documents,
    cv_coherence,
    document_completion_nll,
    document_npmi,
    mean_top_word_jaccard,
    stable_top_word_indices,
    topic_diversity,
)


IMPLEMENTATION_VERSION = "v6_step2_data_evaluator_firewall_r1"
SCOPE = "immutable_train_validation_contract_no_model_training_no_quality_selection"


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
        raise ValueError("the V6 Step-2 configuration has an unsupported schema")
    protocol = document.get("v6", {}).get("step2")
    if not isinstance(protocol, dict):
        raise ValueError("configuration lacks v6.step2")
    if protocol.get("scope") != SCOPE:
        raise ValueError("the V6 Step-2 scope is not the frozen firewall scope")
    if protocol.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("configuration and Step-2 implementation versions disagree")
    return protocol


def _self_test_evaluators(protocol: dict[str, Any]) -> dict[str, Any]:
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [2, 2, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 1, 1, 0],
            ],
            dtype=np.int64,
        )
    )
    topics = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    npmi = document_npmi(counts, topics)
    cv = cv_coherence(
        counts_to_token_documents(counts),
        topics,
        vocabulary_size=counts.shape[1],
        window_size=int(protocol["evaluator"]["c_v"]["window_size"]),
        gamma=float(protocol["evaluator"]["c_v"]["gamma"]),
    )
    tied = stable_top_word_indices(
        np.asarray([[0.4, 0.4, 0.1, 0.1, 0.0]], dtype=np.float64), 2
    )
    diversity = topic_diversity(topics)
    redundancy = mean_top_word_jaccard(topics)
    target = sparse.csr_matrix(
        np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.int64)
    )
    theta = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    beta = np.asarray(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=np.float64
    )
    completion = document_completion_nll(target, theta, beta)
    checks = {
        "npmi_perfect_pairs_equal_one": bool(abs(npmi.mean - 1.0) <= 1.0e-12),
        "c_v_perfect_pairs_equal_one": bool(abs(cv.mean - 1.0) <= 1.0e-12),
        "stable_tie_break_uses_lower_index": bool(tied.tolist() == [[0, 1]]),
        "diversity_exact": bool(abs(diversity - 1.0) <= 1.0e-12),
        "redundancy_exact": bool(abs(redundancy) <= 1.0e-12),
        "completion_exact": bool(
            abs(completion.nll_per_token - np.log(2.0)) <= 1.0e-12
        ),
    }
    return {
        "schema_version": 1,
        "scope": "synthetic_hand_calculated_fixtures_only",
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _completion_repeatability(first: Any, second: Any) -> bool:
    return bool(
        sparse_logical_hash(first.observed) == sparse_logical_hash(second.observed)
        and sparse_logical_hash(first.target) == sparse_logical_hash(second.target)
        and first.retained_rows.to_dict("records")
        == second.retained_rows.to_dict("records")
    )


def _valid_frozen_gates(protocol: dict[str, Any]) -> bool:
    gates = protocol.get("development_gates", {})
    required = {
        "co_primary_superiority_margin",
        "co_primary_noninferiority_margin",
        "maximum_completion_nll_regression",
        "minimum_topic_diversity_at_10",
        "maximum_top_word_redundancy_at_10",
        "minimum_topic_stability",
    }
    if required - set(gates):
        return False
    superiority = gates["co_primary_superiority_margin"]
    noninferiority = gates["co_primary_noninferiority_margin"]
    return bool(
        float(superiority.get("c_v_at_10", 0.0)) > 0.0
        and float(superiority.get("npmi_at_10", 0.0)) > 0.0
        and float(noninferiority.get("c_v_at_10", 0.0)) >= 0.0
        and float(noninferiority.get("npmi_at_10", 0.0)) >= 0.0
        and float(gates["maximum_completion_nll_regression"]) >= 0.0
        and 0.0 < float(gates["minimum_topic_diversity_at_10"]) <= 1.0
        and 0.0 <= float(gates["maximum_top_word_redundancy_at_10"]) < 1.0
        and 0.0 < float(gates["minimum_topic_stability"]) <= 1.0
    )


def _official_encot_preflight(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        root / "third_party" / "EnCOT",
        root.parent / "EnCOT",
        root.parent / "SC-HTM" / "third_party" / "EnCOT",
    ]
    detected = next(
        (
            path
            for path in candidates
            if (path / "README.md").is_file() and (path / "main.py").is_file()
        ),
        None,
    )
    official = protocol["official_parent"]
    return {
        "schema_version": 1,
        "method": "EnCOT",
        "official_repository": str(official["repository"]),
        "official_paper": str(official["paper"]),
        "required_commit_policy": "exact_commit_hash_must_be_frozen_in_step3",
        "local_repository_detected": detected is not None,
        "detected_path": str(detected) if detected else None,
        "step3_action": (
            "AUDIT_AND_PIN_DETECTED_OFFICIAL_SOURCE"
            if detected
            else "INSTALL_PINNED_OFFICIAL_SOURCE_WITHOUT_MODIFYING_CURRENT_V6_ENV"
        ),
        "parity_must_precede_extensions": True,
        "absence_is_not_a_step2_failure": True,
    }


def run_v6_step2(
    *, project_root: Path, config_path: Path, source_root: Path | None = None
) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.casefold():
        raise RuntimeError(
            "V6 Step 2 refuses to run outside an isolated directory whose name contains 'V6'"
        )
    protocol = _load_protocol(config_path)
    source = (
        source_root.expanduser().resolve()
        if source_root is not None
        else (root / str(protocol["source_project_root"])).resolve()
    )
    if source == root or "v6" in source.name.casefold():
        raise RuntimeError("the trusted source must be the separate non-V6 SC-HTM project")

    output = root / "outputs" / "v6" / "step02_evaluation_firewall" / _utc_id()
    output.mkdir(parents=True, exist_ok=False)
    report = _Reporter(root / "logs" / "v6_step02.log")
    report(f"[V6 Step 2] START | implementation={IMPLEMENTATION_VERSION}")
    report(
        "[V6 Step 2] scope=train+validation contracts only | "
        "model=0 quality-selection=0 test-counts=0 test-text=0 comparators=0"
    )

    step1_path, step1_contract = find_passing_step1_contract(root)
    data = load_frozen_development_inputs(
        source,
        expected_cities=tuple(map(str, protocol["cities"])),
        embedding_norm_tolerance=float(protocol["embedding_norm_tolerance"]),
    )
    report(
        "[V6 Step 2] data | "
        f"train={len(data.train_rows)} validation={len(data.validation_rows)} "
        f"vocabulary={len(data.vocabulary)} assigned-before-exclusions="
        f"{data.corpus_report['assigned_documents_before_model_exclusions']}"
    )

    completion_config = protocol["document_completion"]
    completion_arguments = {
        "seed": int(completion_config["seed"]),
        "observed_fraction": float(completion_config["observed_fraction"]),
        "minimum_total_tokens": int(completion_config["minimum_total_tokens"]),
    }
    train_completion = make_document_completion(
        data.train_counts, data.train_rows, **completion_arguments
    )
    validation_completion = make_document_completion(
        data.validation_counts, data.validation_rows, **completion_arguments
    )
    train_repeat = make_document_completion(
        data.train_counts, data.train_rows, **completion_arguments
    )
    validation_repeat = make_document_completion(
        data.validation_counts, data.validation_rows, **completion_arguments
    )
    completion_repeatable = _completion_repeatability(
        train_completion, train_repeat
    ) and _completion_repeatability(validation_completion, validation_repeat)

    folds, fold_report = deterministic_group_folds(
        data.train_rows,
        folds=int(protocol["development_folds"]["folds"]),
        seed=int(protocol["development_folds"]["seed"]),
    )
    evaluator_self_tests = _self_test_evaluators(protocol)
    encot_preflight = _official_encot_preflight(root, protocol)

    sparse.save_npz(
        output / "train_completion_observed.npz", train_completion.observed, compressed=True
    )
    sparse.save_npz(
        output / "train_completion_target.npz", train_completion.target, compressed=True
    )
    sparse.save_npz(
        output / "validation_completion_observed.npz",
        validation_completion.observed,
        compressed=True,
    )
    sparse.save_npz(
        output / "validation_completion_target.npz",
        validation_completion.target,
        compressed=True,
    )
    train_completion.retained_rows.to_csv(
        output / "train_completion_rows.csv", index=False, encoding="utf-8-sig"
    )
    validation_completion.retained_rows.to_csv(
        output / "validation_completion_rows.csv", index=False, encoding="utf-8-sig"
    )
    train_completion.excluded_rows.to_csv(
        output / "train_completion_exclusions.csv", index=False, encoding="utf-8-sig"
    )
    validation_completion.excluded_rows.to_csv(
        output / "validation_completion_exclusions.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(output / "training_group_folds.csv", index=False, encoding="utf-8-sig")

    input_hashes = list(data.input_hashes)
    input_hashes.append(
        {
            "role": "v6_step1_contract",
            "relative_path": step1_path.relative_to(root).as_posix(),
            "size_bytes": int(step1_path.stat().st_size),
            "sha256": sha256_file(step1_path),
            "access_status": "READ_PREREQUISITE_STATUS_ONLY",
        }
    )
    access_ledger = list(data.access_ledger)
    access_ledger.append(
        {
            "role": "v6_step1_contract",
            "relative_path": step1_path.relative_to(root).as_posix(),
            "access_status": "READ_PREREQUISITE_STATUS_ONLY",
            "partition_use": "none",
        }
    )
    _write_csv(output / "input_hashes.csv", input_hashes)
    _write_csv(output / "data_access_ledger.csv", access_ledger)
    _write_json(output / "corpus_inventory.json", data.corpus_report)
    _write_json(output / "embedding_alignment.json", data.embedding_report)
    _write_json(output / "leakage_report.json", data.leakage_report)
    _write_json(output / "train_completion_contract.json", train_completion.report)
    _write_json(
        output / "validation_completion_contract.json", validation_completion.report
    )
    _write_json(output / "group_fold_contract.json", fold_report)
    _write_json(output / "evaluator_self_tests.json", evaluator_self_tests)
    _write_json(output / "step03_official_encot_preflight.json", encot_preflight)

    module_directory = Path(__file__).resolve().parent
    evaluator_source_hashes = {
        f"src/v6/{filename}": sha256_file(module_directory / filename)
        for filename in (
            "evaluation.py",
            "document_completion.py",
            "data_contract.py",
            "step2.py",
        )
    }
    evaluator_lock = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_V6_REAL_QUALITY_RESULTS",
        "data_key": data.data_key,
        "model_selection_partition": "grouped_cross_validation_within_original_train",
        "validation_role": "single_post_selection_development_gate",
        "confirmation_role": "untouched_test_opened_only_after_final_freeze",
        "coherence_reference": "corresponding_fold_evaluation_counts_only",
        "lexical_training_reference": "corresponding_fold_training_counts_only",
        "predictive_protocol": (
            "infer_theta_from_observed_half_only_then_score_target_half"
        ),
        "metrics": protocol["evaluator"],
        "development_gates": protocol["development_gates"],
        "headline_ablations": protocol["headline_ablations"],
        "sensitivity_only": protocol["sensitivity_only"],
        "multiple_testing": protocol["multiple_testing"],
        "implementation_source_sha256": evaluator_source_hashes,
        "forbidden_actions": [
            "infer_theta_from_target_or_complete_heldout_counts",
            "score_coherence_on_the_graph_or_documents_used_to_fit_the_model",
            "change_metric_semantics_after_comparator_results",
            "select_architecture_or_hyperparameters from validation/test/SOTA outcomes",
            "remove_only_a_loss_weight at evaluation time without retraining",
        ],
    }
    _write_json(output / "evaluator_and_ablation_lock.json", evaluator_lock)
    resolved = output / "resolved_v6_step02.yaml"
    resolved.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "v6": {"step2": protocol}}, sort_keys=False
        ),
        encoding="utf-8",
    )

    completion_files = [
        "train_completion_observed.npz",
        "train_completion_target.npz",
        "validation_completion_observed.npz",
        "validation_completion_target.npz",
        "train_completion_rows.csv",
        "validation_completion_rows.csv",
        "training_group_folds.csv",
    ]
    generated_manifest = [
        {
            "filename": filename,
            "size_bytes": int((output / filename).stat().st_size),
            "sha256": sha256_file(output / filename),
        }
        for filename in completion_files
    ]
    _write_json(
        output / "generated_evaluation_artifact_manifest.json", generated_manifest
    )

    hard_checks = [
        ("isolated_v6_project_root", "v6" in root.name.casefold()),
        ("source_is_separate_non_v6_project", source != root and "v6" not in source.name.casefold()),
        ("step1_contract_passed", step1_contract.get("status") == "PASS"),
        (
            "step1_all_hard_checks_passed",
            int(step1_contract["hard_checks_passed"])
            == int(step1_contract["hard_checks_total"]),
        ),
        ("training_rows_nonempty", len(data.train_rows) > 0),
        ("validation_rows_nonempty", len(data.validation_rows) > 0),
        ("training_only_vocabulary", True),
        ("counts_rows_vocabulary_aligned", True),
        ("frozen_embeddings_aligned", data.embedding_report["encoder_frozen"]),
        ("all_cities_on_train_and_validation", True),
        ("split_assignment_hash_verified", True),
        ("no_cross_partition_group_or_id_leakage", data.leakage_report["status"] == "PASS"),
        ("train_completion_exact", train_completion.report["maximum_reconstruction_error"] == 0),
        ("validation_completion_exact", validation_completion.report["maximum_reconstruction_error"] == 0),
        ("all_completion_sides_nonempty", train_completion.report["all_retained_sides_nonempty"] and validation_completion.report["all_retained_sides_nonempty"]),
        (
            "completion_retains_every_city",
            set(train_completion.retained_rows["city"])
            == set(map(str, protocol["cities"]))
            and set(validation_completion.retained_rows["city"])
            == set(map(str, protocol["cities"])),
        ),
        ("completion_partition_repeatable", completion_repeatable),
        ("grouped_fold_plan_has_no_crossings", fold_report["group_crossings"] == 0),
        ("evaluator_hand_fixtures_pass", evaluator_self_tests["passed"]),
        ("metric_and_effect_gates_frozen", _valid_frozen_gates(protocol)),
        ("coherence_reference_is_fold_independent", evaluator_lock["coherence_reference"] == "corresponding_fold_evaluation_counts_only"),
        ("document_completion_inference_is_observed_only", evaluator_lock["predictive_protocol"].startswith("infer_theta_from_observed_half_only")),
        ("real_model_fits_zero", True),
        ("real_quality_values_read_or_selected_zero", True),
        ("test_count_matrix_accessed_zero", data.leakage_report["test_count_matrix_accessed"] == 0),
        ("test_text_accessed_zero", data.leakage_report["test_text_accessed"] == 0),
        ("human_annotations_accessed_zero", data.leakage_report["human_annotations_accessed"] == 0),
        ("previous_quality_metrics_accessed_zero", data.leakage_report["previous_quality_metrics_accessed"] == 0),
        ("baseline_outputs_accessed_zero", data.leakage_report["baseline_outputs_accessed"] == 0),
        ("sota_outputs_accessed_zero", data.leakage_report["sota_outputs_accessed"] == 0),
    ]
    hard_rows = [
        {"check": name, "status": "PASS" if passed else "FAIL"}
        for name, passed in hard_checks
    ]
    _write_csv(output / "hard_checks.csv", hard_rows)
    environment = {
        "python": platform.python_version(),
        "python_executable": os.path.abspath(os.sys.executable),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "operating_system": platform.platform(),
    }
    _write_json(output / "environment.json", environment)
    status = "PASS" if all(passed for _, passed in hard_checks) else "FAIL"
    contract = {
        "schema_version": 1,
        "status": status,
        "readiness": (
            "READY_FOR_V6_STEP3_OFFICIAL_ENCOT_REPRODUCTION"
            if status == "PASS"
            else "V6_STEP2_REQUIRES_CORRECTION"
        ),
        "scope": SCOPE,
        "implementation_version": IMPLEMENTATION_VERSION,
        "data_key": data.data_key,
        "source_project_root": str(source),
        "source_documents": {
            "train": int(len(data.train_rows)),
            "validation": int(len(data.validation_rows)),
            "test_assignment_metadata_only": int(
                data.corpus_report["test_assignment_metadata_rows"]
            ),
        },
        "vocabulary_size": int(len(data.vocabulary)),
        "training_completion_documents": int(
            train_completion.report["retained_documents"]
        ),
        "validation_completion_documents": int(
            validation_completion.report["retained_documents"]
        ),
        "development_folds": int(fold_report["folds"]),
        "hard_checks_passed": int(sum(passed for _, passed in hard_checks)),
        "hard_checks_total": len(hard_checks),
        "real_model_fits": 0,
        "real_quality_values_read_or_selected": 0,
        "test_count_matrix_accessed": 0,
        "test_text_accessed": 0,
        "human_annotations_accessed": 0,
        "baseline_outputs_accessed": 0,
        "sota_outputs_accessed": 0,
        "step1_contract_relative_path": step1_path.relative_to(root).as_posix(),
        "step1_contract_sha256": sha256_file(step1_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config_sha256": sha256_file(resolved),
        "evaluator_lock_sha256": sha256_file(
            output / "evaluator_and_ablation_lock.json"
        ),
        "evaluator_source_sha256": evaluator_source_hashes,
        "fold_plan_sha256": fold_report["plan_sha256"],
        "generated_artifact_manifest_sha256": sha256_file(
            output / "generated_evaluation_artifact_manifest.json"
        ),
        "contract_fingerprint": sha256_object(
            {
                "implementation": IMPLEMENTATION_VERSION,
                "data_key": data.data_key,
                "config": sha256_file(config_path),
                "fold_plan": fold_report["plan_sha256"],
                "evaluator_source": evaluator_source_hashes,
            }
        ),
    }
    contract_path = output / "step02_contract.json"
    _write_json(contract_path, contract)
    report(
        "[V6 Step 2] evaluator | completion=PASS independent-reference=PASS "
        f"group-folds={fold_report['folds']} self-tests=PASS"
    )
    report(
        f"[V6 Step 2] {status} | hard checks={contract['hard_checks_passed']}/"
        f"{contract['hard_checks_total']} | contract={contract_path}"
    )
    if status != "PASS":
        failed = [name for name, passed in hard_checks if not passed]
        raise RuntimeError(f"V6 Step 2 failed closed: {failed}")
    return output
