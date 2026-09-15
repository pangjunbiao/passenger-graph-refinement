"""Exact-runtime workers for the preregistered Step-13 external replication.

Three explicit modes preserve the evaluation firewall:

* ``embed`` creates frozen vocabulary features from fit-vocabulary tokens only;
* ``fit`` trains all ten parents and freezes deterministic V6.1 post-fit state
  without receiving any held-out input; and
* ``evaluate`` restores those frozen states, infers from observed halves only,
  and scores the untouched target halves.

The external corpus has no labels or audited city identifier.  Consequently,
this worker never receives labels or coordinates and reports a predeclared
three-block temporal macro NLL instead of relabelling it as macro-city NLL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.city_backoff import estimate_pooled_unigram
from src.v6.confirmation_evaluation import construct_frozen_variants, reporting_family
from src.v6.data_contract import sha256_file, sha256_object
from src.v6.development_evidence import positive_npmi_graph, topic_quality_metrics
from src.v6.development_worker import _new_parent, _state_hash
from src.v6.final_refit import FINAL_VARIANTS
from src.v6.graph_prototype_adapter import (
    align_prototype_cores_to_topics,
    build_graph_prototype_bank,
    minimum_distortion_graph_projection,
)
from src.v6.predictive_evaluation import document_completion_nll_from_probabilities
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt
from src.v6.step4r2_worker import _parent_probabilities
from src.v6.step5r2_worker import _array_hash, _seed_everything
from src.v6.step6r2_worker import _decode_logits, _latent_product, _raw_logits
from src.v6.step7_worker import _train_parent
from src.v6.transport_calibration import (
    FrozenMomentCalibration,
    apply_frozen_moment_calibration,
    fit_changed_column_moment_calibration,
)
from src.v6.validation_context import TrainingOnlyContext, construct_training_only_context


IMPLEMENTATION = "v6_1_step13_external_temporal_replication_r1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty required table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _verified_job(path: Path) -> dict[str, Any]:
    job = _json(path)
    body = dict(job)
    identity = body.pop("job_identity", None)
    if (
        job.get("schema_version") != 1
        or job.get("implementation_version") != IMPLEMENTATION
        or job.get("mode") not in {"embed", "fit", "evaluate"}
        or sha256_object(body) != identity
    ):
        raise ValueError("unsupported or altered Step-13 worker job")
    for role, item in job.get("inputs", {}).items():
        source = Path(str(item["path"]))
        if not source.is_file() or sha256_file(source) != str(item["sha256"]):
            raise ValueError(f"Step-13 worker input changed: {role}")
    policy = job["policy"]
    if policy.get("labels_available_to_models") is not False:
        raise ValueError("external labels must be structurally absent")
    if policy.get("coordinates_available_to_models") is not False:
        raise ValueError("source coordinates must be structurally absent")
    return job


def _embedding_mode(job: Mapping[str, Any], output: Path) -> dict[str, Any]:
    import importlib.metadata as metadata
    from sentence_transformers import SentenceTransformer

    expected_version = str(job["embedding"]["sentence_transformers_version"])
    observed_version = metadata.version("sentence-transformers")
    if observed_version != expected_version:
        raise RuntimeError(
            f"sentence-transformers={observed_version}; expected exact {expected_version}"
        )
    vocabulary = pd.read_csv(
        Path(job["inputs"]["vocabulary"]["path"]), encoding="utf-8-sig"
    ).sort_values("index")
    tokens = vocabulary["token"].astype(str).tolist()
    if vocabulary["index"].astype(int).tolist() != list(range(len(tokens))):
        raise ValueError("external vocabulary indices are not contiguous")
    configuration = job["embedding"]
    model = SentenceTransformer(
        str(configuration["model_id"]),
        revision=str(configuration["model_revision"]),
        trust_remote_code=False,
    )
    template = str(configuration["input_template"])
    texts = [template.format(token=token) for token in tokens]
    embeddings = np.asarray(
        model.encode(
            texts,
            batch_size=int(configuration["batch_size"]),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=bool(configuration["normalize_embeddings"]),
        ),
        dtype=np.float32,
    )
    expected_shape = (len(tokens), int(configuration["expected_width"]))
    if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
        raise ValueError(
            f"external vocabulary embedding shape {embeddings.shape} != {expected_shape}"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("external vocabulary embeddings are not L2 normalized")
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / "word_embeddings.npz"
    np.savez_compressed(
        array_path,
        embeddings=embeddings,
        tokens=np.asarray(tokens),
    )
    resolved_commit = getattr(
        getattr(getattr(model, "_first_module", lambda: None)(), "auto_model", None),
        "config",
        None,
    )
    resolved_commit = getattr(resolved_commit, "_commit_hash", None)
    if resolved_commit is not None and str(resolved_commit) != str(
        configuration["model_revision"]
    ):
        raise ValueError("resolved embedding-model revision differs from the lock")
    receipt = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "mode": "embed",
        "job_identity": job["job_identity"],
        "model_id": configuration["model_id"],
        "requested_model_revision": configuration["model_revision"],
        "resolved_model_revision": resolved_commit,
        "sentence_transformers_version": observed_version,
        "tokens": len(tokens),
        "embedding_width": int(embeddings.shape[1]),
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "array_sha256": sha256_file(array_path),
        "logical_sha256": _array_hash(embeddings),
        "labels_used": 0,
        "coordinates_used": 0,
        "test_tokens_used_to_define_vocabulary": 0,
    }
    receipt["receipt_identity"] = sha256_object(receipt)
    _write_json(output / "embedding_receipt.json", receipt)
    return receipt


def _load_fit_inputs(job: Mapping[str, Any]) -> tuple[sparse.csr_matrix, np.ndarray]:
    counts = sparse.load_npz(Path(job["inputs"]["fit_counts"]["path"])).tocsr()
    counts = counts.astype(np.int64)
    with np.load(Path(job["inputs"]["word_embeddings"]["path"]), allow_pickle=False) as z:
        embeddings = np.asarray(z["embeddings"], dtype=np.float32)
    expected = (int(job["fit_documents"]), int(job["vocabulary_size"]))
    if counts.shape != expected:
        raise ValueError(f"external fit count shape {counts.shape} != {expected}")
    if embeddings.shape != (
        expected[1],
        int(job["model"]["frozen_word_feature_width"]),
    ):
        raise ValueError("external vocabulary features and counts are misaligned")
    if np.any(np.asarray(counts.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("external fit matrix contains an empty document")
    return counts, embeddings


def _fit_seed_receipt_valid(
    receipt_path: Path,
    checkpoint_path: Path,
    *,
    job_identity: str,
    seed: int,
) -> bool:
    try:
        receipt = _json(receipt_path)
        body = dict(receipt)
        identity = body.pop("receipt_identity")
        return bool(
            receipt.get("implementation_version") == IMPLEMENTATION
            and receipt.get("job_identity") == job_identity
            and int(receipt.get("seed", -1)) == int(seed)
            and sha256_object(body) == identity
            and checkpoint_path.is_file()
            and sha256_file(checkpoint_path) == receipt["checkpoint_sha256"]
            and Path(receipt["trace_path"]).is_file()
            and sha256_file(Path(receipt["trace_path"])) == receipt["trace_sha256"]
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _fit_mode(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Step-13 SC-HTM fitting requires the existing CUDA runtime")
    if "test_observed" in job.get("inputs", {}) or "test_target" in job.get("inputs", {}):
        raise RuntimeError("held-out matrices entered the fit-only worker")
    source = Path(str(job["official_source"]))
    NewMethod, loaded_path = _load_official_newmethod(source)
    counts, word_embeddings = _load_fit_inputs(job)
    device = torch.device("cuda")
    context = construct_training_only_context(
        counts,
        word_embeddings,
        clusters=int(job["global_context"]["clusters"]),
        n_init=int(job["global_context"]["n_init"]),
        seed=int(job["global_context"]["seed"]),
        fit_partition_label=str(job["global_context"]["fit_partition"]),
        original_training_documents=int(job["original_train_documents"]),
        spent_development_documents=int(job["spent_validation_documents"]),
    )
    graph, graph_receipt = positive_npmi_graph(
        counts,
        minimum_document_frequency=int(job["lexical_graph"]["minimum_document_frequency"]),
        minimum_joint_documents=int(job["lexical_graph"]["minimum_joint_documents"]),
        reliability_power=float(job["lexical_graph"]["reliability_power"]),
    )
    graph_receipt["fit_partition"] = str(job["global_context"]["fit_partition"])
    required_active = int(
        np.ceil(
            float(job["lexical_graph"]["minimum_active_fraction"])
            * int(job["vocabulary_size"])
        )
    )
    if int(graph_receipt["active_words"]) < required_active:
        raise ValueError(
            f"external graph active words={graph_receipt['active_words']} < {required_active}"
        )
    pooled_prior = estimate_pooled_unigram(
        counts.toarray(), pseudocount=float(job["pooled_backoff"]["pseudocount"])
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "context_receipt.json", context.audit)
    _write_json(output / "graph_receipt.json", graph_receipt)
    checkpoint_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for seed in map(int, job["training"]["seeds"]):
        checkpoint = output / "checkpoints" / f"seed_{seed}_external_frozen.pt"
        receipt_path = output / "seed_receipts" / f"seed_{seed}.json"
        if _fit_seed_receipt_valid(
            receipt_path,
            checkpoint,
            job_identity=str(job["job_identity"]),
            seed=seed,
        ):
            receipt = _json(receipt_path)
            checkpoint_rows.append(receipt["checkpoint"])
            diagnostic_rows.append(receipt["diagnostic"])
            print(f"[V6.1 Step 13 fit] seed={seed} frozen receipt REUSED", flush=True)
            continue
        print(f"[V6.1 Step 13 fit] seed={seed} START", flush=True)
        _seed_everything(seed)
        parent = _new_parent(NewMethod, word_embeddings, job["model"]).to(device)
        initial_hash = _state_hash(parent.state_dict())
        trace = _train_parent(
            parent,
            context.train_input,
            job["training"],
            seed=seed,
            device=device,
            torch=torch,
        )
        final_hash = _state_hash(parent.state_dict())
        if final_hash == initial_hash:
            raise RuntimeError(f"external parent state did not change (seed={seed})")
        parent.eval()
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            base = parent.get_beta().detach().cpu().numpy().astype(np.float64)
        prototype_seed = seed + int(job["graph_projection"]["prototype_seed_offset"])
        bank = build_graph_prototype_bank(
            graph,
            topics=int(job["model"]["topics"]),
            top_k=int(job["evaluation"]["top_words"]),
            seed=prototype_seed,
            centrality_weight=float(job["graph_projection"]["centrality_weight"]),
            minimum_cluster_size=int(job["graph_projection"]["minimum_cluster_size"]),
        )
        aligned, alignment = align_prototype_cores_to_topics(
            base,
            bank.core_indices,
            epsilon=float(job["evaluation"].get("projection_epsilon", 1.0e-12)),
        )
        projection = minimum_distortion_graph_projection(
            base,
            aligned,
            strength=float(job["graph_projection"]["strength"]),
            prototype_quota=int(job["graph_projection"]["prototype_quota"]),
            rank_margin=float(job["graph_projection"]["rank_margin"]),
            epsilon=float(job["evaluation"].get("projection_epsilon", 1.0e-12)),
        )
        latent = _latent_product(
            parent,
            context.train_input,
            batch_size=int(job["training"]["batch_size"]),
            device=device,
            torch=torch,
        )
        native_logits = _raw_logits(latent, base, device=device, torch=torch)
        graph_logits = _raw_logits(latent, projection.affinity, device=device, torch=torch)
        calibration = fit_changed_column_moment_calibration(
            native_logits,
            graph_logits,
            base,
            projection.affinity,
            variance_floor=float(job["calibration"]["variance_floor"]),
            change_tolerance=float(job["calibration"]["change_tolerance"]),
        )
        quality_args = {
            "top_words": int(job["evaluation"]["top_words"]),
            "window_size": int(job["evaluation"]["c_v_window_size"]),
            "gamma": float(job["evaluation"]["c_v_gamma"]),
        }
        full_quality = topic_quality_metrics(projection.affinity, counts, **quality_args)
        parent_quality = topic_quality_metrics(base, counts, **quality_args)
        trace_path = output / "training_traces" / f"seed_{seed}.csv"
        _write_csv(trace_path, trace)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "implementation_version": IMPLEMENTATION,
            "job_identity": job["job_identity"],
            "seed": seed,
            "official_source_commit": job["official_commit"],
            "model_configuration": dict(job["model"]),
            "parent_state_dict": {
                name: value.detach().cpu() for name, value in parent.state_dict().items()
            },
            "parent_state_sha256": final_hash,
            "base_affinity": torch.from_numpy(base.astype(np.float32)),
            "effective_affinity": torch.from_numpy(
                projection.affinity.astype(np.float32)
            ),
            "calibration_scale": torch.from_numpy(calibration.scale.astype(np.float32)),
            "calibration_shift": torch.from_numpy(calibration.shift.astype(np.float32)),
            "calibration_changed_mask": torch.from_numpy(calibration.changed_mask),
            "calibration_audit": calibration.audit,
            "pooled_prior": torch.from_numpy(pooled_prior.astype(np.float32)),
            "pooled_mixture_weight": float(job["pooled_backoff"]["mixture_weight"]),
            "context_centroids": torch.from_numpy(context.centroids.astype(np.float32)),
            "context_global_bow": torch.from_numpy(context.global_bow.astype(np.float32)),
            "context_audit": context.audit,
            "graph_receipt": graph_receipt,
            "prototype_audit": bank.audit,
            "alignment_audit": alignment,
            "projection_audit": projection.audit,
            "test_documents_used_for_fit": 0,
            "labels_used": 0,
            "coordinates_used": 0,
        }
        torch.save(payload, checkpoint)
        diagnostic = {
            "seed": seed,
            "fit_documents": int(counts.shape[0]),
            "parent_training_loss_reduction": float(
                np.mean([float(row["loss"]) for row in trace[:10]])
                - np.mean([float(row["loss"]) for row in trace[-10:]])
            ),
            "fit_npmi_full": float(full_quality["npmi_at_10"]),
            "fit_npmi_parent": float(parent_quality["npmi_at_10"]),
            "fit_cv_full": float(full_quality["c_v_at_10"]),
            "fit_cv_parent": float(parent_quality["c_v_at_10"]),
            "projection_column_sum_error": float(
                projection.audit["maximum_column_sum_error"]
            ),
            "maximum_nonprototype_change": float(
                projection.audit["maximum_nonprototype_change"]
            ),
            "calibration_mean_error": float(
                calibration.audit["maximum_changed_mean_error"]
            ),
            "calibration_variance_error": float(
                calibration.audit["maximum_changed_regularized_variance_error"]
            ),
        }
        checkpoint_row = {
            "seed": seed,
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "size_bytes": int(checkpoint.stat().st_size),
            "parent_state_sha256": final_hash,
            "base_affinity_sha256": _array_hash(base.astype(np.float32)),
            "effective_affinity_sha256": _array_hash(
                projection.affinity.astype(np.float32)
            ),
        }
        receipt = {
            "schema_version": 1,
            "implementation_version": IMPLEMENTATION,
            "job_identity": job["job_identity"],
            "seed": seed,
            "checkpoint": checkpoint_row,
            "checkpoint_sha256": checkpoint_row["sha256"],
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": sha256_file(trace_path),
            "diagnostic": diagnostic,
            "test_documents_used_for_fit": 0,
            "labels_used": 0,
            "coordinates_used": 0,
        }
        receipt["receipt_identity"] = sha256_object(receipt)
        _write_json(receipt_path, receipt)
        checkpoint_rows.append(checkpoint_row)
        diagnostic_rows.append(diagnostic)
        print(f"[V6.1 Step 13 fit] seed={seed} FROZEN", flush=True)
        del parent
        torch.cuda.empty_cache()
    _write_json(output / "checkpoint_manifest.json", checkpoint_rows)
    _write_csv(output / "fit_seed_diagnostics.csv", diagnostic_rows)
    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "mode": "fit",
        "status": "PASS",
        "job_identity": job["job_identity"],
        "official_class_loaded_from": str(loaded_path),
        "fit_documents": int(counts.shape[0]),
        "vocabulary": int(counts.shape[1]),
        "parent_model_fits": len(checkpoint_rows),
        "seeds": list(map(int, job["training"]["seeds"])),
        "checkpoint_manifest_sha256": sha256_file(output / "checkpoint_manifest.json"),
        "test_documents_used_for_fit": 0,
        "labels_used": 0,
        "coordinates_used": 0,
        "hyperparameter_selection_performed": False,
        "seed_selection_performed": False,
    }
    result["result_identity"] = sha256_object(result)
    _write_json(output / "worker_result.json", result)
    return result


def _tensor(value: Any, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _temporal_macro(
    target: sparse.csr_matrix,
    probability: np.ndarray,
    blocks: np.ndarray,
    *,
    probability_floor: float,
) -> float:
    values = []
    for block in sorted(np.unique(blocks)):
        selected = np.flatnonzero(blocks == block)
        result = document_completion_nll_from_probabilities(
            target[selected],
            probability[selected],
            probability_floor=probability_floor,
        )
        values.append(result.nll_per_token)
    return float(np.mean(values))


def _evaluation_mode(job: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Step-13 SC-HTM evaluation requires the existing CUDA runtime")
    fit_result = _json(Path(job["inputs"]["fit_result"]["path"]))
    if (
        fit_result.get("status") != "PASS"
        or fit_result.get("implementation_version") != IMPLEMENTATION
        or int(fit_result.get("test_documents_used_for_fit", -1)) != 0
    ):
        raise ValueError("external fit freeze is not valid")
    source = Path(str(job["official_source"]))
    NewMethod, loaded_path = _load_official_newmethod(source)
    fit_counts, word_embeddings = _load_fit_inputs(job)
    observed = sparse.load_npz(Path(job["inputs"]["test_observed"]["path"])).tocsr()
    target = sparse.load_npz(Path(job["inputs"]["test_target"]["path"])).tocsr()
    rows = pd.read_csv(
        Path(job["inputs"]["completion_rows"]["path"]), encoding="utf-8-sig"
    ).sort_values("completion_row")
    vocabulary = pd.read_csv(
        Path(job["inputs"]["vocabulary"]["path"]), encoding="utf-8-sig"
    ).sort_values("index")["token"].astype(str).tolist()
    if observed.shape != target.shape or observed.shape != (
        len(rows),
        int(job["vocabulary_size"]),
    ):
        raise ValueError("external completion matrices are misaligned")
    if len(vocabulary) != int(job["vocabulary_size"]):
        raise ValueError("external evaluation vocabulary changed")
    blocks = rows["temporal_block"].to_numpy(dtype=np.int64)
    expected_blocks = list(range(1, int(job["completion"]["temporal_blocks"]) + 1))
    if sorted(np.unique(blocks).tolist()) != expected_blocks:
        raise ValueError("external temporal-block inventory changed")
    manifest = _json(Path(job["inputs"]["checkpoint_manifest"]["path"]))
    if [int(row["seed"]) for row in manifest] != list(
        map(int, job["training"]["seeds"])
    ):
        raise ValueError("external checkpoint seed order changed")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    metric_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    quality_args = {
        "top_words": int(job["evaluation"]["top_words"]),
        "window_size": int(job["evaluation"]["c_v_window_size"]),
        "gamma": float(job["evaluation"]["c_v_gamma"]),
    }
    probability_floor = float(job["evaluation"]["probability_floor"])
    for checkpoint_row in manifest:
        seed = int(checkpoint_row["seed"])
        checkpoint = Path(str(checkpoint_row["path"]))
        if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_row["sha256"]:
            raise ValueError(f"external frozen checkpoint changed (seed={seed})")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            payload.get("implementation_version") != IMPLEMENTATION
            or payload.get("official_source_commit") != job["official_commit"]
            or payload.get("model_configuration") != dict(job["model"])
            or int(payload.get("test_documents_used_for_fit", -1)) != 0
        ):
            raise ValueError(f"external checkpoint contract changed (seed={seed})")
        parent = _new_parent(NewMethod, word_embeddings, job["model"])
        parent.load_state_dict(payload["parent_state_dict"], strict=True)
        before = _state_hash(parent.state_dict())
        if before != payload["parent_state_sha256"]:
            raise ValueError(f"external parent state hash changed (seed={seed})")
        parent = parent.to(device)
        parent.eval()
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        context = TrainingOnlyContext(
            train_input=np.empty((0, 2 * int(job["vocabulary_size"])), dtype=np.float32),
            centroids=_tensor(payload["context_centroids"], np.float64),
            global_bow=_tensor(payload["context_global_bow"], np.float32),
            vocabulary_embeddings=np.asarray(word_embeddings, dtype=np.float64),
            audit=dict(payload["context_audit"]),
        )
        test_input, context_receipt = context.input_from_observed_counts(observed)
        base = _tensor(payload["base_affinity"], np.float32)
        effective = _tensor(payload["effective_affinity"], np.float32)
        latent = _latent_product(
            parent,
            test_input,
            batch_size=int(job["training"]["batch_size"]),
            device=device,
            torch=torch,
        )
        base_logits = _raw_logits(latent, base, device=device, torch=torch)
        graph_logits = _raw_logits(latent, effective, device=device, torch=torch)
        official_probability = _parent_probabilities(
            parent,
            test_input,
            batch_size=int(job["training"]["batch_size"]),
            device=device,
        )
        native_graph_probability = _decode_logits(
            parent,
            graph_logits,
            batch_size=int(job["training"]["batch_size"]),
            device=device,
            torch=torch,
        )
        calibration = FrozenMomentCalibration(
            scale=_tensor(payload["calibration_scale"], np.float64),
            shift=_tensor(payload["calibration_shift"], np.float64),
            changed_mask=_tensor(payload["calibration_changed_mask"], bool),
            audit=dict(payload["calibration_audit"]),
        )
        calibrated_logits = apply_frozen_moment_calibration(graph_logits, calibration)
        calibrated_graph_probability = _decode_logits(
            parent,
            calibrated_logits,
            batch_size=int(job["training"]["batch_size"]),
            device=device,
            torch=torch,
        )
        variants = construct_frozen_variants(
            official_probability,
            native_graph_probability,
            calibrated_graph_probability,
            _tensor(payload["pooled_prior"], np.float64),
            mixture_weight=float(payload["pooled_mixture_weight"]),
        )
        if tuple(variants) != FINAL_VARIANTS:
            raise ValueError("external V6.1 variant order changed")
        qualities = {
            "official_parent_affinity": topic_quality_metrics(
                base, fit_counts, **quality_args
            ),
            "graph_transported_affinity": topic_quality_metrics(
                effective, fit_counts, **quality_args
            ),
        }
        for family, quality in qualities.items():
            for topic, indices in enumerate(quality["top_word_indices"]):
                for rank, index in enumerate(indices, start=1):
                    top_rows.append(
                        {
                            "seed": seed,
                            "reporting_family": family,
                            "topic": int(topic),
                            "rank": rank,
                            "vocabulary_index": int(index),
                            "token": vocabulary[int(index)],
                        }
                    )
        maximum_probability_error = 0.0
        for variant, probability in variants.items():
            error = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
            maximum_probability_error = max(maximum_probability_error, error)
            micro = document_completion_nll_from_probabilities(
                target, probability, probability_floor=probability_floor
            )
            family = reporting_family(variant)
            quality = qualities[family]
            metric_rows.append(
                {
                    "method_id": "v6_1",
                    "seed": seed,
                    "variant": variant,
                    "reporting_family": family,
                    "npmi_at_10": float(quality["npmi_at_10"]),
                    "c_v_at_10": float(quality["c_v_at_10"]),
                    "topic_diversity_at_10": float(
                        quality["topic_diversity_at_10"]
                    ),
                    "top_word_redundancy_at_10": float(
                        quality["top_word_redundancy_at_10"]
                    ),
                    "macro_temporal_block_nll_per_token": _temporal_macro(
                        target,
                        probability,
                        blocks,
                        probability_floor=probability_floor,
                    ),
                    "micro_nll_per_token": float(micro.nll_per_token),
                    "completion_documents": int(micro.documents),
                    "completion_target_tokens": int(micro.tokens),
                    "maximum_probability_sum_error": error,
                }
            )
        after = _state_hash(parent.state_dict())
        integrity_rows.append(
            {
                "seed": seed,
                "state_before_sha256": before,
                "state_after_sha256": after,
                "state_unchanged": before == after,
                "context_test_documents_fit": int(
                    payload["context_audit"]["test_documents_fit"]
                ),
                "context_target_used": False,
                "maximum_probability_sum_error": maximum_probability_error,
                "test_optimizer_steps": 0,
            }
        )
        if before != after:
            raise RuntimeError(f"external model state changed during evaluation (seed={seed})")
        del parent
        torch.cuda.empty_cache()
        print(f"[V6.1 Step 13 evaluate] seed={seed} DONE", flush=True)
    _write_csv(output / "per_seed_variant_metrics.csv", metric_rows)
    _write_csv(output / "top_words_by_seed.csv", top_rows)
    _write_csv(output / "seed_integrity.csv", integrity_rows)
    result = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "mode": "evaluate",
        "status": "PASS",
        "job_identity": job["job_identity"],
        "official_class_loaded_from": str(loaded_path),
        "seeds_evaluated": len(integrity_rows),
        "variants_evaluated": len(FINAL_VARIANTS),
        "test_documents": int(observed.shape[0]),
        "test_target_tokens": int(target.sum()),
        "optimizer_steps": 0,
        "model_state_changes": 0,
        "target_used_for_inference": False,
        "labels_used": 0,
        "coordinates_used": 0,
        "per_seed_variant_metrics_sha256": sha256_file(
            output / "per_seed_variant_metrics.csv"
        ),
        "top_words_by_seed_sha256": sha256_file(output / "top_words_by_seed.csv"),
        "seed_integrity_sha256": sha256_file(output / "seed_integrity.csv"),
    }
    result["result_identity"] = sha256_object(result)
    _write_json(output / "worker_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = _verified_job(args.job.resolve())
    output = Path(str(job["worker_output"])).resolve()
    if job["mode"] == "embed":
        _embedding_mode(job, output)
        return 0
    torch = _seed_everything(int(job["training"]["seeds"][0]))
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "runtime_receipt.json", _runtime_receipt(torch))
    if job["mode"] == "fit":
        _fit_mode(job, output, torch)
    else:
        _evaluation_mode(job, output, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
