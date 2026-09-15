"""Exact-runtime worker for the frozen V6.1 one-shot test confirmation.

The worker restores the ten Step-7 checkpoints without fitting or updating a
parameter. After auditing the preserved R1 pre-evaluation shape failure, it
performs the first evaluative confirmation opening, freezes a deterministic
document-completion snapshot, and evaluates the complete model and four paired
interventions. Comparator code and labels are structurally absent.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.city_backoff import estimate_pooled_unigram
from src.v6.confirmation_evaluation import (
    construct_frozen_variants,
    reporting_family,
    validate_test_partition,
)
from src.v6.confirmation_statistics import (
    build_confirmation_statistics,
    build_paired_effect_rows,
)
from src.v6.data_contract import (
    _parse_included,
    sha256_file,
    sha256_object,
    sparse_logical_hash,
)
from src.v6.development_evidence import (
    dense_hash,
    macro_city_completion,
    topic_quality_metrics,
)
from src.v6.development_worker import _new_parent, _state_hash
from src.v6.document_completion import make_document_completion
from src.v6.final_refit import FINAL_VARIANTS, combine_development_partitions
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt
from src.v6.step4r2_worker import _parent_probabilities
from src.v6.step5r2_worker import _array_hash, _seed_everything
from src.v6.step6r2_worker import _decode_logits, _latent_product, _raw_logits
from src.v6.transport_calibration import (
    FrozenMomentCalibration,
    apply_frozen_moment_calibration,
)
from src.v6.validation_context import TrainingOnlyContext

IMPLEMENTATION = "v6_1_step8_metadata_contract_recovery_r2"
STEP7_IMPLEMENTATION = "v6_1_step7_final_development_refit_freeze_r1"
LEGACY_STEP8_IMPLEMENTATION = "v6_1_step8_one_shot_test_confirmation_r1"
LEGACY_SHAPE_FAILURE = "the test count matrix has an unexpected shape"
EXPECTED_STEP2_SPLIT_METADATA_SHA256 = (
    "b4277f72f613902cf068b80f190150384e1c16af78b4456e5b12bacfe0270a1b"
)
EXPECTED_TEST_SPARSE_LOGICAL_SHA256 = (
    "157d161e601b2454a1f5047ad8a735bdc227ed8c3529dc76c7dbefb0c489fb95"
)
EXPECTED_RECOVERY_ADDENDUM_SHA256 = (
    "b9ac7f586144f0bfc6bda491ad424e3be1dae365c2f2fd14d8df8976c72e43f1"
)

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError(f"cannot write empty required table: {path.name}")
    pd.DataFrame.from_records(records).to_csv(path, index=False, encoding="utf-8-sig")


def _modeled_test_metadata(assignments: pd.DataFrame) -> dict[str, Any]:
    required = {"post_id", "city", "split", "included_in_model"}
    if required - set(assignments.columns):
        raise ValueError("the frozen split metadata lacks model-inclusion evidence")
    if assignments["post_id"].astype(str).duplicated().any():
        raise ValueError("the frozen split metadata contains duplicate post IDs")
    included = _parse_included(assignments["included_in_model"])
    is_test = assignments["split"].astype(str).eq("test")
    assigned = assignments.loc[is_test].copy()
    modeled = assignments.loc[is_test & included].copy()
    excluded = assignments.loc[is_test & ~included].copy()

    def records(frame: pd.DataFrame) -> list[dict[str, str]]:
        return (
            frame[["post_id", "city", "split"]]
            .astype(str)
            .sort_values(["post_id", "city", "split"])
            .to_dict(orient="records")
        )

    return {
        "assigned_documents_before_model_exclusions": len(assigned),
        "modeled_documents": len(modeled),
        "excluded_documents": len(excluded),
        "modeled_records": records(modeled),
        "excluded_records": records(excluded),
    }


def _verify_test_metadata_contract(job: Mapping[str, Any]) -> None:
    contract = job["test_metadata_contract"]
    body = dict(contract)
    identity = str(body.pop("contract_identity", ""))
    if contract.get("schema_version") != 1 or sha256_object(body) != identity:
        raise ValueError("the frozen test-metadata contract identity changed")
    source = contract["source"]
    test_source = job["test_source_contract"]
    if any(
        (
            source.get("sha256") != EXPECTED_STEP2_SPLIT_METADATA_SHA256,
            int(contract.get("assigned_documents_before_model_exclusions", -1))
            != 78,
            int(contract.get("modeled_documents", -1)) != 77,
            int(contract.get("excluded_documents", -1)) != 1,
            int(
                test_source.get(
                    "expected_assignment_documents_before_model_exclusions", -1
                )
            )
            != 78,
            int(test_source.get("expected_source_documents", -1)) != 77,
            int(test_source.get("expected_pre_matrix_exclusions", -1)) != 1,
            int(test_source.get("expected_vocabulary", -1)) != 1148,
            test_source.get("expected_legacy_v4_sparse_logical_sha256")
            != EXPECTED_TEST_SPARSE_LOGICAL_SHA256,
        )
    ):
        raise ValueError("the corrected Step-8 test-source contract changed")
    path = Path(source["path"])
    if not path.is_file() or sha256_file(path) != str(source["sha256"]):
        raise ValueError("the frozen Step-2 split metadata changed")
    assignments = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    observed = _modeled_test_metadata(assignments)
    city_order = list(map(str, job["test_source_contract"]["expected_cities"]))
    expected_city = {
        city: int(
            sum(row["city"] == city for row in observed["modeled_records"])
        )
        for city in city_order
    }
    if any(
        (
            observed["assigned_documents_before_model_exclusions"]
            != int(contract["assigned_documents_before_model_exclusions"]),
            observed["modeled_documents"] != int(contract["modeled_documents"]),
            observed["excluded_documents"] != int(contract["excluded_documents"]),
            sha256_object(observed["modeled_records"])
            != str(contract["modeled_rows_logical_sha256"]),
            sha256_object(observed["excluded_records"])
            != str(contract["excluded_rows_logical_sha256"]),
            expected_city != contract["modeled_city_documents"],
        )
    ):
        raise ValueError("the corrected row count does not reproduce from Step-2 metadata")


def _verify_technical_recovery(job: Mapping[str, Any]) -> None:
    receipt = job["technical_recovery"]
    body = dict(receipt)
    identity = str(body.pop("recovery_identity", ""))
    if receipt.get("schema_version") != 1 or sha256_object(body) != identity:
        raise ValueError("the Step-8 technical-recovery receipt identity changed")
    if any(
        (
            receipt.get("failed_implementation") != LEGACY_STEP8_IMPLEMENTATION,
            receipt.get("failed_exception") != "ValueError",
            receipt.get("failed_message") != LEGACY_SHAPE_FAILURE,
            int(receipt.get("prior_test_source_openings", -1)) != 1,
            int(receipt.get("prior_evaluative_confirmations", -1)) != 0,
            int(receipt.get("prior_test_metrics_computed", -1)) != 0,
            int(receipt.get("prior_model_evaluations", -1)) != 0,
            int(receipt.get("prior_model_updates", -1)) != 0,
            receipt.get("frozen_step7_lineage_matched") is not True,
            receipt.get("failed_output_preserved") is not True,
            tuple(receipt.get("absence_verified", []))
            != _LEGACY_FAILURE_FORBIDDEN_ARTIFACTS,
        )
    ):
        raise ValueError("the Step-8 technical-recovery accounting changed")
    evidence = receipt["evidence"]
    for role in ("worker_job", "test_access_state", "worker_failure", "recovery_addendum"):
        item = evidence[role]
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"the preserved Step-8 recovery evidence changed: {role}")
    if evidence["recovery_addendum"].get("sha256") != EXPECTED_RECOVERY_ADDENDUM_SHA256:
        raise ValueError("the Step-8 recovery addendum identity changed")
    failed_output = Path(receipt["failed_output"])
    if any((failed_output / relative).exists() for relative in receipt["absence_verified"]):
        raise ValueError("the R1 shape failure contains post-opening evaluation artifacts")
    old_job = json.loads(
        Path(evidence["worker_job"]["path"]).read_text(encoding="utf-8-sig")
    )
    old_body = dict(old_job)
    old_identity = str(old_body.pop("job_identity", ""))
    state = json.loads(
        Path(evidence["test_access_state"]["path"]).read_text(encoding="utf-8-sig")
    )
    failure = json.loads(
        Path(evidence["worker_failure"]["path"]).read_text(encoding="utf-8-sig")
    )
    if any(
        (
            old_job.get("implementation_version") != LEGACY_STEP8_IMPLEMENTATION,
            sha256_object(old_body) != old_identity,
            old_job.get("lineage_artifacts") != job.get("lineage_artifacts"),
            old_job.get("development_inputs") != job.get("development_inputs"),
            old_job.get("test_sources") != job.get("test_sources"),
            old_job.get("checkpoint_manifest") != job.get("checkpoint_manifest"),
            old_job.get("official_source") != job.get("official_source"),
            old_job.get("official_commit") != job.get("official_commit"),
            old_job.get("selected_configuration")
            != job.get("selected_configuration"),
            old_job.get("model_configuration") != job.get("model_configuration"),
            old_job.get("training_configuration")
            != job.get("training_configuration"),
            old_job.get("pooled_backoff_configuration")
            != job.get("pooled_backoff_configuration"),
            old_job.get("calibration_configuration")
            != job.get("calibration_configuration"),
            old_job.get("confirmation_preregistration_sha256")
            != job.get("confirmation_preregistration_sha256"),
            old_job.get("document_completion") != job.get("document_completion"),
            old_job.get("final_fit_reference") != job.get("final_fit_reference"),
            old_job.get("frozen_model") != job.get("frozen_model"),
            old_job.get("evaluation") != job.get("evaluation"),
            old_job.get("hypotheses") != job.get("hypotheses"),
            old_job.get("statistics") != job.get("statistics"),
            old_job.get("guardrails") != job.get("guardrails"),
            int(old_job.get("test_source_contract", {}).get("expected_source_documents", -1))
            != 78,
            state.get("status") != "OPENING_FIXED_TEST_SOURCE",
            state.get("job_identity") != old_identity,
            state.get("technical_recovery_uses_frozen_snapshot") is not False,
            int(failure.get("return_code", 0)) == 0,
            f"ValueError: {LEGACY_SHAPE_FAILURE}"
            not in str(failure.get("transcript", "")),
            failure.get("test_conditioned_model_change_allowed") is not False,
        )
    ):
        raise ValueError("the preserved R1 failure does not match the frozen R2 job")


def _load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    body = dict(job)
    identity = str(body.pop("job_identity", ""))
    if (
        job.get("schema_version") != 1
        or job.get("mode") != "step8"
        or job.get("implementation_version") != IMPLEMENTATION
        or sha256_object(body) != identity
    ):
        raise ValueError("unsupported or altered V6.1 Step-8 worker job")
    policy = job["data_policy"]
    required_false = (
        "labels_available_to_worker",
        "comparators_available_to_worker",
        "model_updates_allowed",
        "seed_selection_allowed",
        "test_early_stopping_allowed",
        "post_test_configuration_change_allowed",
    )
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("Step-8 prohibited-access or no-update policy changed")
    if any(
        (
            int(policy.get("v6_1_evaluative_confirmation_number", -1)) != 1,
            int(policy.get("prior_failed_test_source_openings", -1)) != 1,
            int(policy.get("prior_test_metrics_computed", -1)) != 0,
            int(policy.get("prior_model_evaluations", -1)) != 0,
            int(policy.get("cumulative_test_source_openings_after_success", -1))
            != 2,
        )
    ):
        raise ValueError("Step-8 recovery access accounting changed")
    for section in ("lineage_artifacts", "development_inputs"):
        for role, item in job[section].items():
            source = Path(item["path"])
            if not source.is_file() or sha256_file(source) != str(item["sha256"]):
                raise ValueError(f"frozen Step-8 input changed ({section}:{role})")
    manifest = job["checkpoint_manifest"]
    expected_seeds = list(map(int, job["frozen_model"]["seeds"]))
    if [int(row["seed"]) for row in manifest] != expected_seeds:
        raise ValueError("the frozen checkpoint seed order changed")
    for row in manifest:
        source = Path(row["path"])
        if (
            not source.is_file()
            or int(source.stat().st_size) != int(row["size_bytes"])
            or sha256_file(source) != str(row["sha256"])
        ):
            raise ValueError(f"a frozen Step-7 checkpoint changed: {source}")
    _verify_test_metadata_contract(job)
    _verify_technical_recovery(job)
    # Existence is checked here, but bytes and semantics are deliberately not
    # read until the explicit one-shot opening below.
    for role, item in job["test_sources"].items():
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(f"fixed test source is missing ({role}): {source}")
        if set(item) != {"path"}:
            raise ValueError("a pre-opening test-source record contains test evidence")
    return job


def _load_development(job: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        name: Path(item["path"]) for name, item in job["development_inputs"].items()
    }
    train_counts = sparse.load_npz(paths["train_counts"]).tocsr().astype(np.int64)
    spent_counts = sparse.load_npz(paths["spent_counts"]).tocsr().astype(np.int64)
    train_rows = pd.read_csv(
        paths["train_rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    spent_rows = pd.read_csv(
        paths["spent_rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    with np.load(paths["word_embeddings"], allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    vocabulary = (
        pd.read_csv(paths["vocabulary"], encoding="utf-8-sig", dtype={"token": str})
        .sort_values("index")
        .reset_index(drop=True)
    )
    counts, rows, receipt = combine_development_partitions(
        train_counts, train_rows, spent_counts, spent_rows
    )
    expected = job["final_fit_reference"]
    if (
        counts.shape != (int(expected["documents"]), int(expected["vocabulary"]))
        or int(counts.sum()) != int(expected["tokens"])
        or word_embeddings.shape
        != (
            int(expected["vocabulary"]),
            int(job["model_configuration"]["frozen_word_feature_width"]),
        )
    ):
        raise ValueError("the frozen final-fit development evidence changed")
    if (
        len(vocabulary) != int(expected["vocabulary"])
        or vocabulary["index"].astype(int).tolist() != list(range(len(vocabulary)))
        or vocabulary["token"].astype(str).duplicated().any()
    ):
        raise ValueError("the frozen vocabulary is not index aligned")
    return {
        "counts": counts,
        "rows": rows,
        "word_embeddings": word_embeddings,
        "vocabulary": vocabulary,
        "receipt": receipt,
    }


def _tensor_array(value: Any, dtype: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def _load_frozen_payloads(
    job: Mapping[str, Any], torch: Any
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
    common_prior: np.ndarray | None = None
    common_context: tuple[np.ndarray, np.ndarray, Mapping[str, Any]] | None = None
    for row in job["checkpoint_manifest"]:
        payload = torch.load(Path(row["path"]), map_location="cpu", weights_only=False)
        seed = int(row["seed"])
        if any(
            (
                int(payload.get("schema_version", -1)) != 1,
                payload.get("implementation_version") != STEP7_IMPLEMENTATION,
                int(payload.get("seed", -1)) != seed,
                payload.get("official_source_commit") != job["official_commit"],
                payload.get("selected_configuration") != job["selected_configuration"],
                payload.get("model_configuration") != job["model_configuration"],
                payload.get("training_configuration") != job["training_configuration"],
                payload.get("variant_manifest_identity")
                != job["frozen_model"]["variant_manifest_identity"],
                payload.get("confirmation_preregistration_sha256")
                != job["confirmation_preregistration_sha256"],
                payload.get("test_metrics") is not None,
                int(payload.get("test_access_before_freeze", -1)) != 0,
            )
        ):
            raise ValueError(
                f"frozen checkpoint payload identity changed (seed={seed})"
            )
        if _state_hash(payload["parent_state_dict"]) != str(
            payload["parent_state_sha256"]
        ) or str(payload["parent_state_sha256"]) != str(row["parent_state_sha256"]):
            raise ValueError(f"frozen parent state hash changed (seed={seed})")
        base = _tensor_array(payload["base_affinity"], np.float32)
        effective = _tensor_array(payload["effective_affinity"], np.float32)
        expected_topics = int(job["model_configuration"]["topics"])
        expected_vocabulary = int(job["final_fit_reference"]["vocabulary"])
        if (
            base.shape != (expected_topics, expected_vocabulary)
            or effective.shape != base.shape
        ):
            raise ValueError(f"frozen affinity shape changed (seed={seed})")
        if (
            _array_hash(base) != str(payload["base_affinity_sha256"])
            or _array_hash(effective) != str(payload["effective_affinity_sha256"])
            or str(payload["base_affinity_sha256"]) != str(row["base_affinity_sha256"])
            or str(payload["effective_affinity_sha256"])
            != str(row["effective_affinity_sha256"])
        ):
            raise ValueError(f"frozen affinity hash changed (seed={seed})")
        prior = _tensor_array(payload["pooled_prior"], np.float32)
        centroids = _tensor_array(payload["context_centroids"], np.float32)
        global_bow = _tensor_array(payload["context_global_bow"], np.float32)
        scale = _tensor_array(payload["calibration_scale"], np.float32)
        shift = _tensor_array(payload["calibration_shift"], np.float32)
        changed = _tensor_array(payload["calibration_changed_mask"], bool)
        expected_changed = np.max(np.abs(effective - base), axis=0) > float(
            job["calibration_configuration"]["change_tolerance"]
        )
        if (
            prior.shape != (expected_vocabulary,)
            or centroids.shape
            != (
                int(payload["context_audit"]["clusters"]),
                int(job["model_configuration"]["frozen_word_feature_width"]),
            )
            or global_bow.shape
            != (int(payload["context_audit"]["clusters"]), expected_vocabulary)
            or scale.shape != (expected_vocabulary,)
            or shift.shape != (expected_vocabulary,)
            or changed.shape != (expected_vocabulary,)
            or not np.array_equal(changed, expected_changed)
            or not np.isfinite(scale).all()
            or not np.isfinite(shift).all()
            or np.any(scale <= 0.0)
            or not np.allclose(prior.sum(), 1.0, atol=1.0e-6, rtol=0.0)
            or np.any(prior < 0.0)
            or float(payload["pooled_mixture_weight"])
            != float(job["frozen_model"]["pooled_mixture_weight"])
        ):
            raise ValueError(f"frozen post-fit tensor contract changed (seed={seed})")
        audit = payload["context_audit"]
        if (
            dense_hash(centroids) != str(audit["centroids_sha256"])
            or dense_hash(global_bow) != str(audit["global_bow_sha256"])
            or int(audit.get("test_documents_fit", -1)) != 0
        ):
            raise ValueError(f"frozen context hash changed (seed={seed})")
        if common_prior is None:
            common_prior = prior
            common_context = (centroids, global_bow, audit)
        elif (
            not np.array_equal(prior, common_prior)
            or common_context is None
            or not np.array_equal(centroids, common_context[0])
            or not np.array_equal(global_bow, common_context[1])
            or audit != common_context[2]
        ):
            raise ValueError("a supposedly common frozen post-fit input varies by seed")
        payloads.append((dict(row), payload))
    return payloads


def _preflight_parent_restoration(
    NewMethod: type,
    job: Mapping[str, Any],
    development: Mapping[str, Any],
    payloads: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    device: Any,
    torch: Any,
) -> dict[str, float]:
    """Prove all ten parent payloads restore before the test source is opened."""

    # Step 7 materialized ``base_affinity`` from ``get_beta()`` on CUDA.  A CPU
    # reconstruction is useful as a portability diagnostic, but EnCOT's
    # pairwise-distance matrix multiplication is not bitwise device invariant.
    # The integrity gate must therefore reproduce the frozen tensor on the same
    # CUDA execution path while the state dictionary itself remains byte-exact.
    tolerance = float(
        job["guardrails"]["maximum_parent_probability_reproduction_error"]
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("invalid frozen parent-reproduction tolerance")
    maximum_cpu_difference = 0.0
    maximum_runtime_error = 0.0
    for checkpoint_row, payload in payloads:
        seed = int(checkpoint_row["seed"])
        parent = _new_parent(
            NewMethod, development["word_embeddings"], job["model_configuration"]
        )
        parent.load_state_dict(payload["parent_state_dict"], strict=True)
        if _state_hash(parent.state_dict()) != str(payload["parent_state_sha256"]):
            raise ValueError(f"preflight parent restoration failed (seed={seed})")
        stored_affinity = _tensor_array(payload["base_affinity"], np.float32)
        parent.eval()
        with torch.no_grad():
            cpu_affinity = (
                parent.get_beta()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        if (
            cpu_affinity.shape != stored_affinity.shape
            or not np.isfinite(cpu_affinity).all()
            or not np.isfinite(stored_affinity).all()
        ):
            raise ValueError(f"preflight parent affinity contract failed (seed={seed})")
        cpu_difference = float(np.max(np.abs(cpu_affinity - stored_affinity)))
        maximum_cpu_difference = max(maximum_cpu_difference, cpu_difference)

        parent = parent.to(device)
        parent.eval()
        if _state_hash(parent.state_dict()) != str(payload["parent_state_sha256"]):
            raise ValueError(
                f"preflight runtime-device state changed during transfer (seed={seed})"
            )
        with torch.no_grad():
            runtime_affinity = (
                parent.get_beta()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        if (
            runtime_affinity.shape != stored_affinity.shape
            or not np.isfinite(runtime_affinity).all()
        ):
            raise ValueError(
                f"preflight runtime-device affinity contract failed (seed={seed})"
            )
        runtime_error = float(np.max(np.abs(runtime_affinity - stored_affinity)))
        maximum_runtime_error = max(maximum_runtime_error, runtime_error)
        if runtime_error > tolerance:
            raise ValueError(
                "preflight same-runtime parent affinity reproduction failed "
                f"(seed={seed}, max_abs_error={runtime_error:.9e}, "
                f"frozen_tolerance={tolerance:.9e})"
            )
        del parent
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.empty_cache()
    return {
        "maximum_cpu_backend_difference": maximum_cpu_difference,
        "maximum_runtime_device_reproduction_error": maximum_runtime_error,
    }


def _completion_artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "observed": output / "test_completion_observed.npz",
        "target": output / "test_completion_target.npz",
        "rows": output / "test_completion_rows.csv",
        "excluded": output / "test_completion_excluded_rows.csv",
        "contract": output / "test_completion_contract.json",
        "source_receipt": output / "test_source_receipt.json",
    }


def load_frozen_completion(
    output: Path, *, expected_job_identity: str
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame, dict[str, Any]]:
    """Load and independently verify the immutable Step-8 completion snapshot."""

    paths = _completion_artifact_paths(output)
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("the frozen test-completion snapshot is incomplete")
    contract = json.loads(paths["contract"].read_text(encoding="utf-8-sig"))
    body = dict(contract)
    identity = str(body.pop("completion_identity", ""))
    if (
        contract.get("schema_version") != 1
        or contract.get("job_identity") != expected_job_identity
        or sha256_object(body) != identity
    ):
        raise ValueError("the frozen completion contract identity changed")
    for role, path in paths.items():
        if role == "contract":
            continue
        expected = contract["artifacts"][role]
        if sha256_file(path) != str(expected["sha256"]):
            raise ValueError(f"a frozen completion artifact changed ({role})")
    observed = sparse.load_npz(paths["observed"]).tocsr().astype(np.int64)
    target = sparse.load_npz(paths["target"]).tocsr().astype(np.int64)
    rows = (
        pd.read_csv(
            paths["rows"],
            encoding="utf-8-sig",
            dtype={"post_id": str, "city": str, "split": str},
        )
        .sort_values("completion_row")
        .reset_index(drop=True)
    )
    if (
        observed.shape != target.shape
        or observed.shape != (len(rows), int(contract["vocabulary"]))
        or sparse_logical_hash(observed)
        != str(contract["artifacts"]["observed"]["logical_sha256"])
        or sparse_logical_hash(target)
        != str(contract["artifacts"]["target"]["logical_sha256"])
        or np.any(np.asarray(observed.sum(axis=1)).reshape(-1) <= 0)
        or np.any(np.asarray(target.sum(axis=1)).reshape(-1) <= 0)
        or int(observed.sum()) != int(contract["observed_tokens"])
        or int(target.sum()) != int(contract["target_tokens"])
    ):
        raise ValueError("the frozen test-completion matrices do not reproduce")
    return observed, target, rows, contract


def _freeze_or_recover_completion(
    job: Mapping[str, Any], output: Path, development_rows: pd.DataFrame
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame, dict[str, Any], bool]:
    state_path = output / "test_access_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        if (
            state.get("job_identity") != job["job_identity"]
            or int(state.get("cumulative_test_source_openings", -1)) != 2
            or int(state.get("cumulative_evaluative_confirmations", -1)) != 1
            or int(state.get("prior_failed_test_source_openings", -1)) != 1
        ):
            raise RuntimeError("a prior test-access state has a different identity")
        try:
            observed, target, rows, contract = load_frozen_completion(
                output, expected_job_identity=job["job_identity"]
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "test opening began but no complete immutable snapshot is recoverable; "
                "do not change the model or silently reopen the test source"
            ) from exc
        if state.get("status") != "TEST_COMPLETION_FROZEN":
            _write_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "TEST_COMPLETION_FROZEN",
                    "job_identity": job["job_identity"],
                    "cumulative_test_source_openings": 2,
                    "cumulative_evaluative_confirmations": 1,
                    "prior_failed_test_source_openings": 1,
                    "prior_test_metrics_computed": 0,
                    "technical_recovery_uses_frozen_snapshot": True,
                    "completion_identity": contract["completion_identity"],
                    "updated_at_utc": _utc_now(),
                },
            )
        return observed, target, rows, contract, True

    _write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "OPENING_FIXED_TEST_SOURCE",
            "job_identity": job["job_identity"],
            "cumulative_test_source_openings": 2,
            "cumulative_evaluative_confirmations": 1,
            "prior_failed_test_source_openings": 1,
            "prior_test_metrics_computed": 0,
            "technical_recovery_uses_frozen_snapshot": False,
            "opened_at_utc": _utc_now(),
        },
    )
    source_paths = {
        role: Path(item["path"]) for role, item in job["test_sources"].items()
    }
    before_hashes = {role: sha256_file(path) for role, path in source_paths.items()}
    source_counts = sparse.load_npz(source_paths["counts"])
    source_rows = pd.read_csv(
        source_paths["rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    after_hashes = {role: sha256_file(path) for role, path in source_paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("the fixed test source changed while it was opened")
    test_config = job["test_source_contract"]
    source_counts, source_rows, validation = validate_test_partition(
        source_counts,
        source_rows,
        expected_documents=int(test_config["expected_source_documents"]),
        vocabulary=int(test_config["expected_vocabulary"]),
        cities=list(map(str, test_config["expected_cities"])),
        development_post_ids=development_rows["post_id"].astype(str).tolist(),
    )
    modeled_records = (
        source_rows[["post_id", "city", "split"]]
        .astype(str)
        .sort_values(["post_id", "city", "split"])
        .to_dict(orient="records")
    )
    validation["modeled_rows_logical_sha256"] = sha256_object(modeled_records)
    metadata = job["test_metadata_contract"]
    if any(
        (
            validation["logical_sha256"]
            != str(test_config["expected_legacy_v4_sparse_logical_sha256"]),
            validation["modeled_rows_logical_sha256"]
            != str(metadata["modeled_rows_logical_sha256"]),
            validation["city_documents"] != metadata["modeled_city_documents"],
            int(validation["documents"]) != int(metadata["modeled_documents"]),
        )
    ):
        raise ValueError(
            "the opened test matrix does not match the frozen Step-2/V4 identity"
        )
    completion_config = job["document_completion"]
    completion = make_document_completion(
        source_counts,
        source_rows,
        seed=int(completion_config["seed"]),
        observed_fraction=float(completion_config["observed_fraction"]),
        minimum_total_tokens=int(completion_config["minimum_total_tokens"]),
    )
    paths = _completion_artifact_paths(output)
    sparse.save_npz(paths["observed"], completion.observed, compressed=True)
    sparse.save_npz(paths["target"], completion.target, compressed=True)
    completion.retained_rows.to_csv(paths["rows"], index=False, encoding="utf-8-sig")
    completion.excluded_rows.to_csv(
        paths["excluded"], index=False, encoding="utf-8-sig"
    )
    source_receipt = {
        "schema_version": 1,
        "test_source_openings_cumulative_v6_1": 2,
        "evaluative_confirmations_cumulative_v6_1": 1,
        "prior_failed_test_source_openings": 1,
        "prior_test_metrics_computed": 0,
        "technical_recovery_identity": job["technical_recovery"][
            "recovery_identity"
        ],
        "accurate_claim": job["data_policy"]["accurate_claim"],
        "globally_unseen_claim_allowed": False,
        "legacy_v4_test_scores_known_to_researcher": True,
        "source_files": {
            role: {
                "path": str(source_paths[role]),
                "size_bytes": int(source_paths[role].stat().st_size),
                "sha256": before_hashes[role],
            }
            for role in sorted(source_paths)
        },
        "validation": validation,
    }
    _write_json(paths["source_receipt"], source_receipt)
    artifacts = {
        "observed": {
            "path": str(paths["observed"]),
            "sha256": sha256_file(paths["observed"]),
            "logical_sha256": sparse_logical_hash(completion.observed),
        },
        "target": {
            "path": str(paths["target"]),
            "sha256": sha256_file(paths["target"]),
            "logical_sha256": sparse_logical_hash(completion.target),
        },
        "rows": {
            "path": str(paths["rows"]),
            "sha256": sha256_file(paths["rows"]),
        },
        "excluded": {
            "path": str(paths["excluded"]),
            "sha256": sha256_file(paths["excluded"]),
        },
        "source_receipt": {
            "path": str(paths["source_receipt"]),
            "sha256": sha256_file(paths["source_receipt"]),
        },
    }
    contract = {
        "schema_version": 1,
        "job_identity": job["job_identity"],
        "construction": completion.report["construction"],
        "seed": int(completion.report["seed"]),
        "observed_fraction": float(completion.report["observed_fraction"]),
        "source_documents": int(completion.report["source_documents"]),
        "retained_documents": int(completion.report["retained_documents"]),
        "excluded_documents": int(completion.report["excluded_documents"]),
        "vocabulary": int(source_counts.shape[1]),
        "source_tokens_retained": int(completion.report["source_tokens_retained"]),
        "observed_tokens": int(completion.report["observed_tokens"]),
        "target_tokens": int(completion.report["target_tokens"]),
        "maximum_reconstruction_error": int(
            completion.report["maximum_reconstruction_error"]
        ),
        "all_retained_sides_nonempty": bool(
            completion.report["all_retained_sides_nonempty"]
        ),
        "quality_metrics_use_test_counts": False,
        "test_target_used_for_inference": False,
        "test_source_openings_cumulative_v6_1": 2,
        "evaluative_confirmations_cumulative_v6_1": 1,
        "prior_failed_test_source_openings": 1,
        "prior_test_metrics_computed": 0,
        "technical_recovery_identity": job["technical_recovery"][
            "recovery_identity"
        ],
        "artifacts": artifacts,
    }
    contract["completion_identity"] = sha256_object(contract)
    _write_json(paths["contract"], contract)
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "TEST_COMPLETION_FROZEN",
            "job_identity": job["job_identity"],
            "cumulative_test_source_openings": 2,
            "cumulative_evaluative_confirmations": 1,
            "prior_failed_test_source_openings": 1,
            "prior_test_metrics_computed": 0,
            "technical_recovery_uses_frozen_snapshot": False,
            "completion_identity": contract["completion_identity"],
            "updated_at_utc": _utc_now(),
        },
    )
    return (
        completion.observed,
        completion.target,
        completion.retained_rows,
        contract,
        False,
    )


def _frozen_context(
    payload: Mapping[str, Any], word_embeddings: np.ndarray
) -> TrainingOnlyContext:
    return TrainingOnlyContext(
        train_input=np.empty((0, 2 * len(word_embeddings)), dtype=np.float32),
        centroids=_tensor_array(payload["context_centroids"], np.float64),
        global_bow=_tensor_array(payload["context_global_bow"], np.float32),
        vocabulary_embeddings=np.asarray(word_embeddings, dtype=np.float64),
        audit=dict(payload["context_audit"]),
    )


def _quality_top_word_rows(
    quality: Mapping[str, Any],
    vocabulary: pd.DataFrame,
    development_counts: sparse.csr_matrix,
    *,
    seed: int,
    family: str,
) -> list[dict[str, Any]]:
    frequencies = np.asarray((development_counts > 0).sum(axis=0)).reshape(-1)
    tokens = vocabulary["token"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for topic, indices in enumerate(quality["top_word_indices"]):
        for rank, index in enumerate(indices, start=1):
            word_index = int(index)
            rows.append(
                {
                    "seed": int(seed),
                    "reporting_family": family,
                    "topic": int(topic),
                    "rank": int(rank),
                    "vocabulary_index": word_index,
                    "token": tokens[word_index],
                    "development_document_frequency": int(frequencies[word_index]),
                }
            )
    return rows


def _evaluate_seed(
    job: Mapping[str, Any],
    checkpoint_row: Mapping[str, Any],
    payload: Mapping[str, Any],
    development: Mapping[str, Any],
    observed: sparse.csr_matrix,
    target: sparse.csr_matrix,
    completion_rows: pd.DataFrame,
    NewMethod: type,
    *,
    device: Any,
    torch: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    seed = int(checkpoint_row["seed"])
    parent = _new_parent(
        NewMethod, development["word_embeddings"], job["model_configuration"]
    )
    parent.load_state_dict(payload["parent_state_dict"], strict=True)
    state_before = _state_hash(parent.state_dict())
    if state_before != str(payload["parent_state_sha256"]):
        raise ValueError(f"restored parent state differs at seed {seed}")
    parent = parent.to(device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    context = _frozen_context(payload, development["word_embeddings"])
    test_input, context_receipt = context.input_from_observed_counts(observed)
    base_affinity = _tensor_array(payload["base_affinity"], np.float32)
    effective_affinity = _tensor_array(payload["effective_affinity"], np.float32)
    latent = _latent_product(
        parent,
        test_input,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    base_logits = _raw_logits(latent, base_affinity, device=device, torch=torch)
    effective_logits = _raw_logits(
        latent, effective_affinity, device=device, torch=torch
    )
    official_probability = _parent_probabilities(
        parent,
        test_input,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
    )
    base_probability = _decode_logits(
        parent,
        base_logits,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    native_graph_probability = _decode_logits(
        parent,
        effective_logits,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    calibration = FrozenMomentCalibration(
        scale=_tensor_array(payload["calibration_scale"], np.float64),
        shift=_tensor_array(payload["calibration_shift"], np.float64),
        changed_mask=_tensor_array(payload["calibration_changed_mask"], bool),
        audit=dict(payload["calibration_audit"]),
    )
    calibrated_logits = apply_frozen_moment_calibration(effective_logits, calibration)
    calibrated_graph_probability = _decode_logits(
        parent,
        calibrated_logits,
        batch_size=int(job["evaluation"]["inference_batch_size"]),
        device=device,
        torch=torch,
    )
    prior = _tensor_array(payload["pooled_prior"], np.float64)
    variants = construct_frozen_variants(
        official_probability,
        native_graph_probability,
        calibrated_graph_probability,
        prior,
        mixture_weight=float(payload["pooled_mixture_weight"]),
    )
    if tuple(variants) != tuple(job["frozen_model"]["variants"]):
        raise ValueError("the frozen Step-8 variant order changed")

    quality_args = {
        "top_words": int(job["evaluation"]["top_words"]),
        "window_size": int(job["evaluation"]["c_v_window_size"]),
        "gamma": float(job["evaluation"]["c_v_gamma"]),
    }
    quality_by_family = {
        "official_parent_affinity": topic_quality_metrics(
            np.asarray(base_affinity, dtype=np.float64),
            development["counts"],
            **quality_args,
        ),
        "graph_transported_affinity": topic_quality_metrics(
            np.asarray(effective_affinity, dtype=np.float64),
            development["counts"],
            **quality_args,
        ),
    }
    top_rows: list[dict[str, Any]] = []
    for family, quality in quality_by_family.items():
        top_rows.extend(
            _quality_top_word_rows(
                quality,
                development["vocabulary"],
                development["counts"],
                seed=seed,
                family=family,
            )
        )

    city_names = list(map(str, job["test_source_contract"]["expected_cities"]))
    city_lookup = {city: index for index, city in enumerate(city_names)}
    mapped_cities = completion_rows["city"].astype(str).map(city_lookup)
    if mapped_cities.isna().any():
        raise ValueError("a completion city is outside the frozen inventory")
    city_indices = mapped_cities.to_numpy(dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    maximum_probability_error = 0.0
    for variant, probability in variants.items():
        probability_error = float(
            np.max(np.abs(np.asarray(probability).sum(axis=1) - 1.0))
        )
        maximum_probability_error = max(maximum_probability_error, probability_error)
        completion, by_city = macro_city_completion(
            target,
            probability,
            city_indices,
            city_names,
            probability_floor=float(job["document_completion"]["probability_floor"]),
        )
        family = reporting_family(variant)
        quality = quality_by_family[family]
        metric_rows.append(
            {
                "seed": seed,
                "variant": variant,
                "reporting_family": family,
                "npmi_at_10": float(quality["npmi_at_10"]),
                "c_v_at_10": float(quality["c_v_at_10"]),
                "topic_diversity_at_10": float(quality["topic_diversity_at_10"]),
                "top_word_redundancy_at_10": float(
                    quality["top_word_redundancy_at_10"]
                ),
                "macro_city_nll_per_token": float(
                    completion["macro_city_nll_per_token"]
                ),
                "micro_nll_per_token": float(completion["micro_nll_per_token"]),
                "micro_perplexity": float(completion["micro_perplexity"]),
                "completion_documents": int(completion["documents"]),
                "completion_target_tokens": int(completion["tokens"]),
                "maximum_probability_sum_error": probability_error,
                "quality_reference": "same_695_document_final_fit_counts",
                "predictive_reference": "same_frozen_test_completion_target",
            }
        )
        city_rows.extend(
            {
                "seed": seed,
                "variant": variant,
                **row,
            }
            for row in by_city
        )

    state_after = _state_hash(parent.state_dict())
    reproduction_error = float(np.max(np.abs(base_probability - official_probability)))
    integrity = {
        "seed": seed,
        "parent_state_sha256_before": state_before,
        "parent_state_sha256_after": state_after,
        "parent_state_unchanged": state_after == state_before,
        "base_affinity_sha256": _array_hash(base_affinity),
        "effective_affinity_sha256": _array_hash(effective_affinity),
        "context_assignments_sha256": context_receipt["assignments_sha256"],
        "context_model_input_sha256": context_receipt["model_input_sha256"],
        "target_half_used_for_assignment": context_receipt[
            "target_half_used_for_assignment"
        ],
        "maximum_probability_sum_error": maximum_probability_error,
        "maximum_parent_probability_reproduction_error": reproduction_error,
        "gradient_evaluations": 0,
        "optimizer_steps": 0,
        "parameters_updated": 0,
    }
    del parent
    torch.cuda.empty_cache()
    return metric_rows, city_rows, top_rows, integrity


def _step9_handoff(output: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    paths = _completion_artifact_paths(output)
    artifacts = []
    for role in ("observed", "target", "rows", "contract"):
        path = paths[role]
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    result = {
        "schema_version": 1,
        "status": "FROZEN_FOR_STEP9",
        "step8_job_identity": job["job_identity"],
        "completion_artifacts": artifacts,
        "comparator_fit_documents": int(job["final_fit_reference"]["documents"]),
        "model_change_from_comparator_results_allowed": False,
    }
    result["handoff_identity"] = sha256_object(result)
    return result


def run_worker(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V6.1 Step 8 requires the unchanged Step-7 CUDA runtime")
    device = torch.device("cuda")
    development = _load_development(job)
    payloads = _load_frozen_payloads(job, torch)
    NewMethod, loaded_path = _load_official_newmethod(Path(job["official_source"]))
    preflight = _preflight_parent_restoration(
        NewMethod,
        job,
        development,
        payloads,
        device=device,
        torch=torch,
    )
    print(
        "[V6.1 Step 8 worker] preflight parent restoration PASS | "
        "state-hashes=exact "
        f"cuda-affinity-error={preflight['maximum_runtime_device_reproduction_error']:.3e} "
        f"cpu-diagnostic-difference={preflight['maximum_cpu_backend_difference']:.3e} "
        "test-source-opened=0",
        flush=True,
    )
    expected_prior = estimate_pooled_unigram(
        development["counts"].toarray(),
        pseudocount=float(job["pooled_backoff_configuration"]["pseudocount"]),
    ).astype(np.float32)
    if any(
        not np.allclose(
            _tensor_array(payload["pooled_prior"], np.float32),
            expected_prior,
            atol=1.0e-8,
            rtol=1.0e-7,
        )
        for _, payload in payloads
    ):
        raise ValueError(
            "the frozen pooled prior does not reproduce from final-fit data"
        )

    observed, target, completion_rows, completion_contract, recovered = (
        _freeze_or_recover_completion(job, output, development["rows"])
    )
    print(
        "[V6.1 Step 8 worker] "
        + (
            "RECOVERY | immutable test-completion snapshot reused; source reopen=0"
            if recovered
            else "R2 EVALUATIVE OPENING COMPLETE | immutable document-completion snapshot frozen"
        ),
        flush=True,
    )
    metric_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    for checkpoint_row, payload in payloads:
        seed = int(checkpoint_row["seed"])
        print(f"[V6.1 Step 8 worker] seed={seed} FROZEN EVALUATION START", flush=True)
        seed_metrics, seed_cities, seed_top_words, seed_integrity = _evaluate_seed(
            job,
            checkpoint_row,
            payload,
            development,
            observed,
            target,
            completion_rows,
            NewMethod,
            device=device,
            torch=torch,
        )
        metric_rows.extend(seed_metrics)
        city_rows.extend(seed_cities)
        top_rows.extend(seed_top_words)
        integrity_rows.append(seed_integrity)
        by_variant = {row["variant"]: row for row in seed_metrics}
        full = by_variant[FINAL_VARIANTS[0]]
        parent = by_variant[FINAL_VARIANTS[-1]]
        print(
            f"[V6.1 Step 8 worker] seed={seed} DONE | "
            f"full NPMI={full['npmi_at_10']:+.6f} C_v={full['c_v_at_10']:.6f} "
            f"macro-NLL={full['macro_city_nll_per_token']:.6f} | "
            f"vs-parent dNPMI={full['npmi_at_10'] - parent['npmi_at_10']:+.6f} "
            f"dC_v={full['c_v_at_10'] - parent['c_v_at_10']:+.6f} "
            f"dNLL={parent['macro_city_nll_per_token'] - full['macro_city_nll_per_token']:+.6f}",
            flush=True,
        )

    summaries, hypothesis_tests, outcome = build_confirmation_statistics(
        metric_rows,
        variants=job["frozen_model"]["variants"],
        seeds=job["frozen_model"]["seeds"],
        hypotheses=job["hypotheses"],
        bootstrap_resamples=int(job["statistics"]["bootstrap_resamples"]),
        bootstrap_seed=int(job["statistics"]["bootstrap_seed"]),
        alpha=float(job["statistics"]["alpha"]),
    )
    paired_effect_rows = build_paired_effect_rows(
        metric_rows,
        seeds=job["frozen_model"]["seeds"],
        hypotheses=job["hypotheses"],
    )
    full_rows = [row for row in metric_rows if row["variant"] == FINAL_VARIANTS[0]]
    guardrails = {
        "topic_diversity_at_10_minimum_observed": float(
            min(row["topic_diversity_at_10"] for row in full_rows)
        ),
        "topic_diversity_at_10_minimum_required": float(
            job["guardrails"]["topic_diversity_at_10_minimum"]
        ),
        "topic_diversity_guardrail_passed": bool(
            min(row["topic_diversity_at_10"] for row in full_rows)
            >= float(job["guardrails"]["topic_diversity_at_10_minimum"])
        ),
        "top_word_redundancy_at_10_maximum_observed": float(
            max(row["top_word_redundancy_at_10"] for row in full_rows)
        ),
        "top_word_redundancy_at_10_maximum_required": float(
            job["guardrails"]["top_word_redundancy_at_10_maximum"]
        ),
        "top_word_redundancy_guardrail_passed": bool(
            max(row["top_word_redundancy_at_10"] for row in full_rows)
            <= float(job["guardrails"]["top_word_redundancy_at_10_maximum"])
        ),
    }
    outcome.update(
        {
            "full_model_guardrails": guardrails,
            "all_full_model_guardrails_passed": bool(
                guardrails["topic_diversity_guardrail_passed"]
                and guardrails["top_word_redundancy_guardrail_passed"]
            ),
            "confirmation_conclusion": (
                "ALL_PRESPECIFIED_ABLATION_ENDPOINTS_SUPPORTED"
                if outcome["all_prespecified_primary_endpoints_supported"]
                else "PARTIAL_OR_NULL_ABLATION_SUPPORT_REPORT_WITHOUT_RETUNING"
            ),
            "test_conditioned_model_change_allowed": False,
            "step9_comparison_allowed_after_integrity_pass": True,
        }
    )
    _write_csv(output / "per_seed_variant_metrics.csv", metric_rows)
    _write_csv(output / "per_city_variant_metrics.csv", city_rows)
    _write_csv(output / "top_words_by_seed.csv", top_rows)
    _write_csv(output / "seed_integrity.csv", integrity_rows)
    _write_csv(output / "variant_summary.csv", summaries)
    _write_csv(output / "ablation_hypothesis_tests.csv", hypothesis_tests)
    _write_csv(output / "paired_effects_by_seed.csv", paired_effect_rows)
    _write_json(output / "variant_summary.json", summaries)
    _write_json(output / "ablation_hypothesis_tests.json", hypothesis_tests)
    _write_json(output / "confirmation_outcome.json", outcome)
    _write_json(output / "development_merge_receipt.json", development["receipt"])
    handoff = _step9_handoff(output, job)
    _write_json(output / "step9_handoff.json", handoff)
    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "status": "PASS",
        "official_class_loaded_from": str(loaded_path),
        "step7_freeze_identity": job["frozen_model"]["freeze_identity"],
        "step8_job_identity": job["job_identity"],
        "completion_identity": completion_contract["completion_identity"],
        "test_source_openings_cumulative_v6_1": 2,
        "test_source_openings_this_invocation": 0 if recovered else 1,
        "evaluative_confirmations_cumulative_v6_1": 1,
        "evaluative_confirmations_this_invocation": 0 if recovered else 1,
        "prior_failed_test_source_openings": 1,
        "prior_test_metrics_computed": 0,
        "technical_recovery_identity": job["technical_recovery"][
            "recovery_identity"
        ],
        "technical_recovery_from_frozen_snapshot": recovered,
        "test_source_documents": int(completion_contract["source_documents"]),
        "test_completion_documents": int(completion_contract["retained_documents"]),
        "test_completion_target_tokens": int(completion_contract["target_tokens"]),
        "reporting_seeds": list(map(int, job["frozen_model"]["seeds"])),
        "variants": list(job["frozen_model"]["variants"]),
        "metric_rows": len(metric_rows),
        "parent_neural_fits": 0,
        "optimizer_steps": 0,
        "parameters_updated": 0,
        "seed_selection_performed": False,
        "test_early_stopping_performed": False,
        "test_conditioned_model_change_performed": False,
        "labels_used": 0,
        "baseline_outputs_used": 0,
        "sota_outputs_used": 0,
        "maximum_probability_sum_error": float(
            max(row["maximum_probability_sum_error"] for row in integrity_rows)
        ),
        "maximum_parent_probability_reproduction_error": float(
            max(
                row["maximum_parent_probability_reproduction_error"]
                for row in integrity_rows
            )
        ),
        "maximum_preflight_parent_affinity_reproduction_error": float(
            preflight["maximum_runtime_device_reproduction_error"]
        ),
        "maximum_preflight_cpu_affinity_backend_difference": float(
            preflight["maximum_cpu_backend_difference"]
        ),
        "all_parent_states_unchanged": bool(
            all(row["parent_state_unchanged"] for row in integrity_rows)
        ),
        "confirmation_outcome": outcome,
        "step9_handoff_identity": handoff["handoff_identity"],
    }
    _write_json(output / "worker_result.json", result)
    print(
        f"[V6.1 Step 8 worker] PASS | frozen evaluations={len(metric_rows)} "
        "fits=0 updates=0 source-openings=2 evaluative-confirmations=1 "
        "comparators=0",
        flush=True,
    )
    print(
        f"[V6.1 Step 8 worker] outcome | "
        f"supported-endpoints={outcome['supported_endpoint_count']}/"
        f"{outcome['primary_endpoint_tests']} | "
        f"{outcome['confirmation_conclusion']}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = _load_job(args.job.resolve())
    output = Path(job["worker_output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch = _seed_everything(int(job["frozen_model"]["seeds"][0]))
    receipt_path = output / "runtime_receipt.json"
    receipt = _runtime_receipt(torch)
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        if prior != receipt:
            raise RuntimeError("Step-8 technical recovery runtime changed")
    else:
        _write_json(receipt_path, receipt)
    run_worker(job, output, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
