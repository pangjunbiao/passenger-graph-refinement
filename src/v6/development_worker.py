"""Exact-runtime worker for the separate V6 Step-4 and Step-5 gates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.data_contract import sha256_file
from src.v6.development_evidence import (
    construct_fold_context,
    macro_city_completion,
    matched_topic_stability,
    positive_npmi_graph,
    topic_quality_metrics,
)
from src.v6.step3_worker import _load_official_newmethod, _runtime_receipt


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    pd.DataFrame.from_records(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _seed_everything(seed: int) -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    return torch


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _gradient_norm(parameters: Iterable[Any]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach()
            total += float((value * value).sum().cpu())
    return float(np.sqrt(total))


def _edge_window_mean(
    trace: list[dict[str, Any]], key: str, *, first: bool, maximum_width: int = 10
) -> float:
    if not trace:
        raise ValueError("cannot summarize an empty optimization trace")
    width = min(int(maximum_width), max(1, len(trace) // 4))
    selected = trace[:width] if first else trace[-width:]
    return float(np.mean([float(row[key]) for row in selected]))


def _load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    if job.get("schema_version") != 1 or job.get("mode") not in {"step4", "step5"}:
        raise ValueError("unsupported V6 development worker job")
    for role, item in job["inputs"].items():
        input_path = Path(item["path"])
        if not input_path.is_file():
            raise FileNotFoundError(f"worker input is missing ({role}): {input_path}")
        if sha256_file(input_path) != str(item["sha256"]):
            raise ValueError(f"worker input hash changed ({role})")
    return job


def _load_evidence(job: Mapping[str, Any]) -> dict[str, Any]:
    paths = {name: Path(value["path"]) for name, value in job["inputs"].items()}
    counts = sparse.load_npz(paths["train_counts"]).tocsr().astype(np.int64)
    with np.load(paths["train_embeddings"], allow_pickle=False) as archive:
        document_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        embedding_ids = archive["post_ids"].astype(str).tolist()
    with np.load(paths["word_embeddings"], allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    folds = pd.read_csv(
        paths["fold_assignments"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    ).sort_values("matrix_row").reset_index(drop=True)
    completion_rows = pd.read_csv(
        paths["completion_rows"],
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    ).sort_values("completion_row").reset_index(drop=True)
    observed = sparse.load_npz(paths["completion_observed"]).tocsr().astype(np.int64)
    target = sparse.load_npz(paths["completion_target"]).tocsr().astype(np.int64)

    if counts.shape != (620, 1148):
        raise ValueError("the frozen V6 training count shape changed")
    if document_embeddings.shape != (620, 768) or word_embeddings.shape != (1148, 768):
        raise ValueError("the frozen V6 embedding shape changed")
    if folds["matrix_row"].astype(int).tolist() != list(range(620)):
        raise ValueError("fold rows are not aligned to the count matrix")
    if folds["post_id"].astype(str).tolist() != embedding_ids:
        raise ValueError("fold rows and document embeddings have different IDs")
    if observed.shape != target.shape or observed.shape != (len(completion_rows), 1148):
        raise ValueError("completion matrices and rows are misaligned")
    source_rows = completion_rows["source_matrix_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        completion_rows["post_id"].astype(str).to_numpy(),
        folds.iloc[source_rows]["post_id"].astype(str).to_numpy(),
    ):
        raise ValueError("completion and fold post IDs are misaligned")
    if (observed + target != counts[source_rows]).nnz != 0:
        raise ValueError("observed plus target does not reconstruct source counts")
    city_names = list(map(str, job["cities"]))
    city_lookup = {name: index for index, name in enumerate(city_names)}
    if set(folds["city"].astype(str)) != set(city_lookup):
        raise ValueError("city universe differs from the frozen configuration")
    city_indices = folds["city"].astype(str).map(city_lookup).to_numpy(dtype=np.int64)
    completion_city_indices = city_indices[source_rows]
    return {
        "counts": counts,
        "document_embeddings": document_embeddings,
        "word_embeddings": word_embeddings,
        "folds": folds,
        "fold_assignments": folds["fold"].to_numpy(dtype=np.int64),
        "city_indices": city_indices,
        "city_names": city_names,
        "completion_rows": completion_rows,
        "completion_source_rows": source_rows,
        "completion_city_indices": completion_city_indices,
        "completion_observed": observed,
        "completion_target": target,
    }


def _new_parent(NewMethod: type, word_embeddings: np.ndarray, config: Mapping[str, Any]) -> Any:
    return NewMethod(
        int(word_embeddings.shape[0]),
        num_topics=int(config["topics"]),
        en_units=int(config.get("encoder_units", 200)),
        num_clusters=int(config["num_clusters"]),
        dropout=float(config["dropout"]),
        pretrained_WE=np.asarray(word_embeddings, dtype=np.float32),
        embed_size=int(word_embeddings.shape[1]),
        beta_temp=float(config["beta_temp"]),
        weight_loss_ECR=float(config["weight_loss_ECR"]),
        weight_ot_doc_cluster=float(config["weight_ot_doc_cluster"]),
        weight_ot_topic_cluster=float(config["weight_ot_topic_cluster"]),
        sinkhorn_alpha=float(config["sinkhorn_alpha"]),
        sinkhorn_max_iter=int(config["sinkhorn_max_iter"]),
        alpha_noise=float(config["alpha_noise"]),
        alpha_augment=float(config["alpha_augment"]),
    )


def _train_parent(
    model: Any,
    train_input: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: Any,
    fold: int,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    tensor = torch.from_numpy(np.asarray(train_input, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 100_003 * int(epoch)
        )
        loader = DataLoader(
            tensor,
            batch_size=int(batch_size),
            shuffle=True,
            generator=generator,
        )
        sums: dict[str, float] = {}
        documents = 0
        maximum_gradient = 0.0
        for batch_cpu in loader:
            batch = batch_cpu.to(device)
            result = model(batch)
            if not all(
                bool(torch.isfinite(value).all())
                for value in result.values()
                if hasattr(value, "detach")
            ):
                raise FloatingPointError("non-finite official parent objective")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            maximum_gradient = max(maximum_gradient, _gradient_norm(model.parameters()))
            optimizer.step()
            count = int(batch.shape[0])
            documents += count
            for key, value in result.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count
        row = {
            "fold": int(fold),
            "phase": "official_parent_warmup",
            "epoch": int(epoch),
            "documents": int(documents),
            "maximum_gradient_norm": float(maximum_gradient),
        }
        row.update({key: value / documents for key, value in sums.items()})
        trace.append(row)
        if epoch in {1, int(epochs)} or epoch % 10 == 0:
            print(
                f"[V6 Step 4 worker] fold={fold} parent epoch={epoch}/{epochs} "
                f"loss={row['loss']:.6f}",
                flush=True,
            )
    return trace


def _train_city_adapter(
    model: Any,
    train_input: np.ndarray,
    train_cities: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: Any,
    fold: int,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.v6.encot_extensions import ExtensionWeights

    for parameter in model.parent.parameters():
        parameter.requires_grad_(False)
    parameters = [model.city_contrast_coefficients, model.gate_logits]
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_input, dtype=np.float32)),
        torch.from_numpy(np.asarray(train_cities, dtype=np.int64)),
    )
    weights = ExtensionWeights.from_mapping(config)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.eval()  # freezes the exact parent RNG and BatchNorm statistics
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 100_003 * int(epoch)
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        sums: dict[str, float] = {}
        documents = 0
        maximum_gradient = 0.0
        for input_cpu, city_cpu in loader:
            input_batch = input_cpu.to(device)
            city_batch = city_cpu.to(device)
            result = model.reconstruction_objective(
                input_batch,
                city_batch,
                residual_scale=float(config["residual_scale"]),
                weights=weights,
                epsilon=float(config["epsilon"]),
            )
            scalars = {
                key: value
                for key, value in result.items()
                if hasattr(value, "ndim") and value.ndim == 0
            }
            if not all(bool(torch.isfinite(value).all()) for value in scalars.values()):
                raise FloatingPointError("non-finite city-adapter objective")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            maximum_gradient = max(maximum_gradient, _gradient_norm(parameters))
            torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clip_norm"]))
            optimizer.step()
            count = int(input_batch.shape[0])
            documents += count
            for key, value in scalars.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count
        row = {
            "fold": int(fold),
            "phase": "frozen_parent_city_adapter",
            "epoch": int(epoch),
            "documents": int(documents),
            "maximum_gradient_norm": float(maximum_gradient),
        }
        row.update({key: value / documents for key, value in sums.items()})
        trace.append(row)
        if epoch in {1, int(config["epochs"])} or epoch % 10 == 0:
            print(
                f"[V6 Step 4 worker] fold={fold} city epoch={epoch}/{config['epochs']} "
                f"recon={row['reconstruction']:.6f} grad={maximum_gradient:.3e}",
                flush=True,
            )
    return trace


def _batched_probabilities(
    model: Any,
    input_values: np.ndarray,
    city_indices: np.ndarray,
    *,
    residual_scale: float,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import torch

    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(input_values), int(batch_size)):
            stop = min(start + int(batch_size), len(input_values))
            batch = torch.from_numpy(
                np.asarray(input_values[start:stop], dtype=np.float32)
            ).to(device)
            cities = torch.from_numpy(
                np.asarray(city_indices[start:stop], dtype=np.int64)
            ).to(device)
            probability = model.infer_probabilities(
                batch,
                cities if float(residual_scale) != 0.0 else None,
                residual_scale=float(residual_scale),
            )
            rows.append(probability.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(rows, axis=0)


def _completion_for_fold(
    evidence: Mapping[str, Any], context: Any, fold: int
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray]:
    source = evidence["completion_source_rows"]
    selected = np.flatnonzero(evidence["fold_assignments"][source] == int(fold))
    input_values = context.input_for_local_counts(
        evidence["completion_observed"][selected]
    )
    return (
        input_values,
        evidence["completion_target"][selected],
        evidence["completion_city_indices"][selected],
        selected,
    )


def _fold_evaluation(
    model: Any,
    evidence: Mapping[str, Any],
    context: Any,
    fold: int,
    *,
    residual_scale: float,
    evaluation: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    completion_input, target, completion_cities, _ = _completion_for_fold(
        evidence, context, fold
    )
    probabilities = _batched_probabilities(
        model,
        completion_input,
        completion_cities,
        residual_scale=float(residual_scale),
        batch_size=int(evaluation["inference_batch_size"]),
        device=device,
    )
    completion, city_rows = macro_city_completion(
        target,
        probabilities,
        completion_cities,
        evidence["city_names"],
        probability_floor=float(evaluation["probability_floor"]),
    )
    with __import__("torch").no_grad():
        reporting = model.reporting_topics().detach().cpu().numpy().astype(np.float64)
    quality = topic_quality_metrics(
        reporting,
        evidence["counts"][context.evaluation_indices],
        top_words=int(evaluation["top_words"]),
        window_size=int(evaluation["c_v_window_size"]),
        gamma=float(evaluation["c_v_gamma"]),
    )
    metrics = {**completion, **{k: v for k, v in quality.items() if k != "top_word_indices"}}
    return metrics, city_rows, np.asarray(quality["top_word_indices"], dtype=np.int64)


def _parent_identity_error(model: Any, input_values: np.ndarray, device: Any) -> float:
    import torch

    batch = torch.from_numpy(np.asarray(input_values[:31], dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        wrapped = model.infer_probabilities(batch, None, residual_scale=0.0)
        vocabulary = int(model.parent.vocab_size)
        local_theta, _ = model.parent.noise_local_encode(batch[:, :vocabulary])
        global_theta, _ = model.parent.global_encode(batch[:, vocabulary:])
        mass = global_theta * local_theta
        direct = torch.softmax(
            model.parent.decoder_bn(mass @ model.parent.get_beta()), dim=-1
        )
    return float((wrapped - direct).abs().max().cpu())


def _city_activity(model: Any) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        state = model.city_residual_state()
        weighted = torch.einsum(
            "c,ckv->kv",
            model.normalized_city_weights(),
            state.effective_logit_residuals,
        )
        return {
            "coefficient_rms": float(
                torch.sqrt(torch.mean(model.city_contrast_coefficients**2)).cpu()
            ),
            "gate_minimum": float(state.gates.min().cpu()),
            "gate_mean": float(state.gates.mean().cpu()),
            "gate_maximum": float(state.gates.max().cpu()),
            "effective_residual_rms": float(
                torch.sqrt(torch.mean(state.effective_logit_residuals**2)).cpu()
            ),
            "effective_residual_maximum_absolute": float(
                state.effective_logit_residuals.abs().max().cpu()
            ),
            "weighted_city_center_error": float(weighted.abs().max().cpu()),
            "topic_null_direction_error": float(
                state.effective_logit_residuals.sum(dim=1).abs().max().cpu()
            ),
        }


def _run_step4(job: Mapping[str, Any], evidence: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    from src.v6.encot_extensions import V6EnCOT, extension_parameter_counts

    NewMethod, loaded_path = _load_official_newmethod(Path(job["official_source"]))
    device = torch.device("cuda")
    model_config = job["model"]
    training = job["training"]
    city_config = job["city_residual"]
    context_config = job["global_context"]
    evaluation = job["evaluation"]
    fold_rows: list[dict[str, Any]] = []
    city_metric_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    context_receipts: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    top_word_sets: list[np.ndarray] = []
    identity_errors: list[float] = []

    counts = evidence["counts"]
    city_indices = evidence["city_indices"]
    folds = list(map(int, job["development_folds"]))
    for fold in folds:
        fold_seed = int(training["seed"]) + 10_000 * int(fold)
        print(f"[V6 Step 4 worker] fold={fold} START seed={fold_seed}", flush=True)
        _seed_everything(fold_seed)
        context = construct_fold_context(
            counts,
            evidence["word_embeddings"],
            evidence["fold_assignments"],
            fold,
            clusters=int(context_config["clusters"]),
            n_init=int(context_config["n_init"]),
            seed=int(context_config["seed"]) + fold,
        )
        context_receipts.append(context.audit)
        parent = _new_parent(NewMethod, evidence["word_embeddings"], model_config).to(device)
        initial_parent_hash = _state_hash(parent.state_dict())
        parent_trace = _train_parent(
            parent,
            context.train_input,
            epochs=int(training["parent_epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["parent_learning_rate"]),
            seed=fold_seed,
            device=device,
            fold=fold,
        )
        trace_rows.extend(parent_trace)
        warmed_parent_hash = _state_hash(parent.state_dict())
        train_cities = city_indices[context.train_indices]
        city_counts = np.bincount(train_cities, minlength=len(evidence["city_names"]))
        wrapper = V6EnCOT(
            parent,
            frozen_vocabulary_features=torch.from_numpy(evidence["word_embeddings"]),
            city_weights=torch.from_numpy(city_counts.astype(np.float32)),
            gate_initial_logit=float(city_config["gate_initial_logit"]),
        ).to(device)
        initializer = wrapper.initialize_city_residual_from_counts(
            counts[context.train_indices].toarray(),
            train_cities,
            shrinkage=float(city_config["initializer"]["shrinkage"]),
            pseudocount=float(city_config["initializer"]["pseudocount"]),
            strength=float(city_config["initializer"]["strength"]),
            projection_ridge=float(city_config["initializer"]["projection_ridge"]),
            log_odds_clip=float(city_config["initializer"]["log_odds_clip"]),
        )
        identity_errors.append(_parent_identity_error(wrapper, context.train_input, device))
        adapter_trace = _train_city_adapter(
            wrapper,
            context.train_input,
            train_cities,
            city_config,
            seed=fold_seed + 1_003,
            device=device,
            fold=fold,
        )
        trace_rows.extend(adapter_trace)
        parent_early = _edge_window_mean(parent_trace, "loss", first=True)
        parent_late = _edge_window_mean(parent_trace, "loss", first=False)
        adapter_early = _edge_window_mean(
            adapter_trace, "reconstruction", first=True
        )
        adapter_late = _edge_window_mean(
            adapter_trace, "reconstruction", first=False
        )
        pooled_metrics, pooled_city, _ = _fold_evaluation(
            wrapper,
            evidence,
            context,
            fold,
            residual_scale=0.0,
            evaluation=evaluation,
            device=device,
        )
        city_metrics, active_city, top_words = _fold_evaluation(
            wrapper,
            evidence,
            context,
            fold,
            residual_scale=float(city_config["residual_scale"]),
            evaluation=evaluation,
            device=device,
        )
        top_word_sets.append(top_words)
        activity = _city_activity(wrapper)
        reduction = (
            pooled_metrics["macro_city_nll_per_token"]
            - city_metrics["macro_city_nll_per_token"]
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_documents": len(context.train_indices),
                "evaluation_documents": len(context.evaluation_indices),
                "completion_documents": city_metrics["documents"],
                "initial_parent_state_sha256": initial_parent_hash,
                "warmed_parent_state_sha256": warmed_parent_hash,
                "parent_early_window_mean_loss": parent_early,
                "parent_late_window_mean_loss": parent_late,
                "adapter_early_window_mean_reconstruction": adapter_early,
                "adapter_late_window_mean_reconstruction": adapter_late,
                "pooled_macro_city_nll": pooled_metrics["macro_city_nll_per_token"],
                "city_macro_city_nll": city_metrics["macro_city_nll_per_token"],
                "nll_reduction_positive_favors_city": reduction,
                "pooled_micro_nll": pooled_metrics["micro_nll_per_token"],
                "city_micro_nll": city_metrics["micro_nll_per_token"],
                "c_v_at_10": city_metrics["c_v_at_10"],
                "npmi_at_10": city_metrics["npmi_at_10"],
                "topic_diversity_at_10": city_metrics["topic_diversity_at_10"],
                "top_word_redundancy_at_10": city_metrics["top_word_redundancy_at_10"],
                **activity,
                "initializer_effective_residual_rms": initializer[
                    "effective_residual_rms"
                ],
            }
        )
        for variant, values in (("complete_pooling", pooled_city), ("city_residual", active_city)):
            for row in values:
                city_metric_rows.append({"fold": fold, "variant": variant, **row})
        checkpoint = output / "checkpoints" / f"fold_{fold:02d}_city_stage.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "fold": fold,
                "official_source_commit": job["official_commit"],
                "model_configuration": model_config,
                "city_configuration": city_config,
                "state_dict": wrapper.state_dict(),
                "context_audit": context.audit,
                "initializer": initializer,
            },
            checkpoint,
        )
        checkpoint_rows.append(
            {
                "fold": fold,
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "size_bytes": int(checkpoint.stat().st_size),
            }
        )
        print(
            f"[V6 Step 4 worker] fold={fold} DONE | macro-NLL reduction={reduction:.6f} "
            f"activity={activity['effective_residual_rms']:.3e}",
            flush=True,
        )

    _write_csv(output / "fold_metrics.csv", fold_rows)
    _write_csv(output / "city_stratified_metrics.csv", city_metric_rows)
    _write_csv(output / "training_trace.csv", trace_rows)
    _write_json(output / "fold_context_receipts.json", context_receipts)
    _write_json(output / "checkpoint_manifest.json", checkpoint_rows)
    reductions = np.asarray(
        [row["nll_reduction_positive_favors_city"] for row in fold_rows]
    )
    parent_decrease = np.asarray(
        [
            row["parent_early_window_mean_loss"]
            - row["parent_late_window_mean_loss"]
            for row in fold_rows
        ]
    )
    adapter_decrease = np.asarray(
        [
            row["adapter_early_window_mean_reconstruction"]
            - row["adapter_late_window_mean_reconstruction"]
            for row in fold_rows
        ]
    )
    parameter_counts = extension_parameter_counts(
        cities=len(evidence["city_names"]),
        topics=int(model_config["topics"]),
        vocabulary=counts.shape[1],
        embedding_dimension=evidence["word_embeddings"].shape[1],
    )
    summary = {
        "schema_version": 1,
        "mode": "step4",
        "status": "PASS",
        "official_class_loaded_from": str(loaded_path),
        "folds": len(folds),
        "staged_model_fits": len(folds),
        "parent_warmups": len(folds),
        "separately_fitted_pooling_comparators": 0,
        "parent_identity_maximum_error": float(max(identity_errors)),
        "mean_parent_loss_decrease": float(parent_decrease.mean()),
        "mean_adapter_reconstruction_decrease": float(adapter_decrease.mean()),
        "mean_nll_reduction": float(reductions.mean()),
        "median_nll_reduction": float(np.median(reductions)),
        "minimum_nll_reduction": float(reductions.min()),
        "positive_fold_fraction": float(np.mean(reductions > 0.0)),
        "minimum_effective_residual_rms": float(
            min(row["effective_residual_rms"] for row in fold_rows)
        ),
        "maximum_weighted_city_center_error": float(
            max(row["weighted_city_center_error"] for row in fold_rows)
        ),
        "maximum_topic_null_direction_error": float(
            max(row["topic_null_direction_error"] for row in fold_rows)
        ),
        "topic_stability": matched_topic_stability(top_word_sets),
        "parameter_counts": parameter_counts,
        "checkpoint_manifest": checkpoint_rows,
        "validation_documents_used": 0,
        "test_documents_used": 0,
        "labels_used": 0,
        "baseline_or_sota_outputs_used": 0,
    }
    _write_json(output / "worker_result.json", summary)
    return summary


def _joint_optimizer(model: Any, config: Mapping[str, Any]) -> Any:
    import torch

    model.parent.word_embeddings.requires_grad_(False)
    topic = [model.parent.topic_embeddings]
    residual = [model.city_contrast_coefficients, model.gate_logits]
    excluded = {id(value) for value in topic + residual + [model.parent.word_embeddings]}
    backbone = [
        parameter
        for parameter in model.parent.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    return torch.optim.Adam(
        [
            {"params": backbone, "lr": float(config["backbone_learning_rate"])},
            {"params": topic, "lr": float(config["topic_learning_rate"])},
            {"params": residual, "lr": float(config["residual_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )


def _train_joint_variant(
    model: Any,
    train_input: np.ndarray,
    train_cities: np.ndarray,
    graph: np.ndarray,
    config: Mapping[str, Any],
    city_config: Mapping[str, Any],
    lexical_config: Mapping[str, Any],
    *,
    lexical_active: bool,
    seed: int,
    fold: int,
    device: Any,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.v6.encot_extensions import ExtensionWeights

    _seed_everything(seed)
    optimizer = _joint_optimizer(model, config)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_input, dtype=np.float32)),
        torch.from_numpy(np.asarray(train_cities, dtype=np.int64)),
    )
    graph_tensor = torch.from_numpy(np.asarray(graph, dtype=np.float32)).to(device)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 100_003 * int(epoch)
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        if lexical_active:
            ramp = min(1.0, epoch / float(lexical_config["warmup_epochs"]))
            lexical_weight = float(lexical_config["lambda_lexical"]) * ramp
        else:
            lexical_weight = 0.0
        weights = ExtensionWeights(
            pooling=float(city_config["lambda_pooling"]),
            gate=float(city_config["lambda_gate"]),
            lexical=lexical_weight,
        )
        documents = 0
        sums: dict[str, float] = {}
        maximum_gradient = 0.0
        for input_cpu, city_cpu in loader:
            input_batch = input_cpu.to(device)
            city_batch = city_cpu.to(device)
            result = model(
                input_batch,
                city_batch,
                residual_scale=float(city_config["residual_scale"]),
                weights=weights,
                positive_npmi_graph=graph_tensor if lexical_active else None,
                lexical_top_k=int(lexical_config["top_k"]),
                lexical_temperature=float(lexical_config["temperature"]),
                epsilon=float(lexical_config["epsilon"]),
            )
            scalars = {
                key: value
                for key, value in result.items()
                if hasattr(value, "ndim") and value.ndim == 0
            }
            if not all(bool(torch.isfinite(value).all()) for value in scalars.values()):
                raise FloatingPointError("non-finite Step-5 joint objective")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            maximum_gradient = max(maximum_gradient, _gradient_norm(trainable))
            torch.nn.utils.clip_grad_norm_(
                trainable, float(config["gradient_clip_norm"])
            )
            optimizer.step()
            count = int(input_batch.shape[0])
            documents += count
            for key, value in scalars.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count
        row = {
            "fold": int(fold),
            "variant": "full_lexical" if lexical_active else "without_lexical",
            "epoch": int(epoch),
            "lambda_lexical": float(lexical_weight),
            "documents": int(documents),
            "maximum_gradient_norm": float(maximum_gradient),
        }
        row.update({key: value / documents for key, value in sums.items()})
        trace.append(row)
        if epoch in {1, int(config["epochs"])} or epoch % 10 == 0:
            lexical_value = row.get("lexical", 0.0)
            print(
                f"[V6 Step 5 worker] fold={fold} {row['variant']} "
                f"epoch={epoch}/{config['epochs']} loss={row['loss']:.6f} "
                f"Rlex={lexical_value:.6f}",
                flush=True,
            )
    return trace


def _real_tail_gradient(model: Any, graph: np.ndarray, lexical: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    from src.v6.encot_extensions import rank_aware_lexical_loss

    model.eval()
    graph_tensor = torch.from_numpy(np.asarray(graph, dtype=np.float32)).to(device)
    affinity = model.official_affinity()
    loss, association, membership, rank_scores = rank_aware_lexical_loss(
        affinity,
        graph_tensor,
        top_k=int(lexical["top_k"]),
        temperature=float(lexical["temperature"]),
        epsilon=float(lexical["epsilon"]),
    )
    gradient = torch.autograd.grad(loss, rank_scores, retain_graph=False)[0].abs()
    ranking = torch.argsort(affinity.detach(), dim=1, descending=True, stable=True)
    tail = ranking[:, 5:10]
    selected = torch.gather(gradient, 1, tail)
    return {
        "minimum_rank_6_to_10_gradient": float(selected.min().detach().cpu()),
        "mean_rank_6_to_10_gradient": float(selected.mean().detach().cpu()),
        "positive_rank_6_to_10_fraction": float(
            (selected > float(lexical["gradient_activity_floor"]))
            .float()
            .mean()
            .detach()
            .cpu()
        ),
        "maximum_membership_mass_error": float(
            (membership.sum(dim=1) - float(lexical["top_k"]))
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "mean_graph_association": float(association.mean().detach().cpu()),
    }


def _run_step5(job: Mapping[str, Any], evidence: Mapping[str, Any], output: Path, torch: Any) -> dict[str, Any]:
    from src.v6.encot_extensions import V6EnCOT

    NewMethod, loaded_path = _load_official_newmethod(Path(job["official_source"]))
    device = torch.device("cuda")
    model_config = job["model"]
    training = job["training"]
    city_config = job["city_residual"]
    lexical = job["lexical"]
    context_config = job["global_context"]
    evaluation = job["evaluation"]
    checkpoints = {int(row["fold"]): row for row in job["step4_checkpoints"]}

    fold_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    city_metric_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    graph_receipts: list[dict[str, Any]] = []
    context_receipts: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    full_top_words: list[np.ndarray] = []
    nolex_top_words: list[np.ndarray] = []

    for fold in map(int, job["development_folds"]):
        print(f"[V6 Step 5 worker] fold={fold} START", flush=True)
        context = construct_fold_context(
            evidence["counts"],
            evidence["word_embeddings"],
            evidence["fold_assignments"],
            fold,
            clusters=int(context_config["clusters"]),
            n_init=int(context_config["n_init"]),
            seed=int(context_config["seed"]) + fold,
        )
        context_receipts.append(context.audit)
        graph, graph_report = positive_npmi_graph(
            evidence["counts"][context.train_indices],
            minimum_document_frequency=int(lexical["minimum_document_frequency"]),
            minimum_joint_documents=int(lexical["minimum_joint_documents"]),
            reliability_power=float(lexical["reliability_power"]),
        )
        graph_report["fold"] = fold
        graph_receipts.append(graph_report)
        train_cities = evidence["city_indices"][context.train_indices]
        city_counts = np.bincount(
            train_cities, minlength=len(evidence["city_names"])
        ).astype(np.float32)

        checkpoint_row = checkpoints[fold]
        checkpoint_path = Path(checkpoint_row["path"])
        if sha256_file(checkpoint_path) != str(checkpoint_row["sha256"]):
            raise ValueError(f"Step-4 checkpoint hash changed for fold {fold}")

        def restored() -> Any:
            parent = _new_parent(NewMethod, evidence["word_embeddings"], model_config)
            wrapper = V6EnCOT(
                parent,
                frozen_vocabulary_features=torch.from_numpy(evidence["word_embeddings"]),
                city_weights=torch.from_numpy(city_counts),
                gate_initial_logit=float(city_config["gate_initial_logit"]),
            ).to(device)
            payload = torch.load(checkpoint_path, map_location=device)
            if int(payload.get("fold", -1)) != fold:
                raise ValueError(f"Step-4 checkpoint fold identity changed for fold {fold}")
            if payload.get("official_source_commit") != job["official_commit"]:
                raise ValueError("Step-4 checkpoint official-source identity changed")
            if payload.get("model_configuration") != model_config:
                raise ValueError("Step-4 checkpoint architecture differs from Step 5")
            stored_city = payload.get("city_configuration", {})
            for key in (
                "residual_scale",
                "gate_initial_logit",
                "lambda_pooling",
                "lambda_gate",
            ):
                if float(stored_city.get(key, float("nan"))) != float(city_config[key]):
                    raise ValueError(f"Step-4 city setting changed before Step 5: {key}")
            stored_context = payload.get("context_audit", {})
            for key in ("centroids_sha256", "assignments_sha256", "global_bow_sha256"):
                if stored_context.get(key) != context.audit.get(key):
                    raise ValueError(f"fold-local context changed before Step 5: {key}")
            wrapper.load_state_dict(payload["state_dict"], strict=True)
            return wrapper

        nolex_model = restored()
        full_model = restored()
        initial_match = _state_hash(nolex_model.state_dict()) == _state_hash(
            full_model.state_dict()
        )
        variant_seed = int(training["seed"]) + 10_000 * fold
        nolex_trace = _train_joint_variant(
            nolex_model,
            context.train_input,
            train_cities,
            graph,
            training,
            city_config,
            lexical,
            lexical_active=False,
            seed=variant_seed,
            fold=fold,
            device=device,
        )
        full_trace = _train_joint_variant(
            full_model,
            context.train_input,
            train_cities,
            graph,
            training,
            city_config,
            lexical,
            lexical_active=True,
            seed=variant_seed,
            fold=fold,
            device=device,
        )
        trace_rows.extend(nolex_trace)
        trace_rows.extend(full_trace)
        nolex_metrics, nolex_city, nolex_words = _fold_evaluation(
            nolex_model,
            evidence,
            context,
            fold,
            residual_scale=float(city_config["residual_scale"]),
            evaluation=evaluation,
            device=device,
        )
        full_metrics, full_city, full_words = _fold_evaluation(
            full_model,
            evidence,
            context,
            fold,
            residual_scale=float(city_config["residual_scale"]),
            evaluation=evaluation,
            device=device,
        )
        nolex_top_words.append(nolex_words)
        full_top_words.append(full_words)
        for variant, metrics in (
            ("without_rank_aware_lexical", nolex_metrics),
            ("full_v6_step5", full_metrics),
        ):
            variant_rows.append({"fold": fold, "variant": variant, **metrics})
        for variant, values in (
            ("without_rank_aware_lexical", nolex_city),
            ("full_v6_step5", full_city),
        ):
            for row in values:
                city_metric_rows.append({"fold": fold, "variant": variant, **row})
        tail = _real_tail_gradient(full_model, graph, lexical, device)
        tail_rows.append({"fold": fold, **tail})
        fold_rows.append(
            {
                "fold": fold,
                "initial_states_exactly_matched": initial_match,
                "npmi_full": full_metrics["npmi_at_10"],
                "npmi_without_lexical": nolex_metrics["npmi_at_10"],
                "npmi_improvement_positive_favors_full": (
                    full_metrics["npmi_at_10"] - nolex_metrics["npmi_at_10"]
                ),
                "c_v_full": full_metrics["c_v_at_10"],
                "c_v_without_lexical": nolex_metrics["c_v_at_10"],
                "c_v_difference_positive_favors_full": (
                    full_metrics["c_v_at_10"] - nolex_metrics["c_v_at_10"]
                ),
                "nll_full": full_metrics["macro_city_nll_per_token"],
                "nll_without_lexical": nolex_metrics["macro_city_nll_per_token"],
                "nll_reduction_positive_favors_full": (
                    nolex_metrics["macro_city_nll_per_token"]
                    - full_metrics["macro_city_nll_per_token"]
                ),
                "topic_diversity_full": full_metrics["topic_diversity_at_10"],
                "top_word_redundancy_full": full_metrics[
                    "top_word_redundancy_at_10"
                ],
                **tail,
            }
        )
        checkpoint = output / "checkpoints" / f"fold_{fold:02d}_full_step5.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "fold": fold,
                "official_source_commit": job["official_commit"],
                "model_configuration": model_config,
                "city_configuration": city_config,
                "lexical_configuration": lexical,
                "state_dict": full_model.state_dict(),
                "graph_receipt": graph_report,
                "context_audit": context.audit,
            },
            checkpoint,
        )
        checkpoint_rows.append(
            {
                "fold": fold,
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "size_bytes": int(checkpoint.stat().st_size),
            }
        )
        print(
            f"[V6 Step 5 worker] fold={fold} DONE | "
            f"dNPMI={fold_rows[-1]['npmi_improvement_positive_favors_full']:.6f} "
            f"dC_v={fold_rows[-1]['c_v_difference_positive_favors_full']:.6f}",
            flush=True,
        )

    _write_csv(output / "paired_fold_differences.csv", fold_rows)
    _write_csv(output / "variant_metrics.csv", variant_rows)
    _write_csv(output / "city_stratified_metrics.csv", city_metric_rows)
    _write_csv(output / "training_trace.csv", trace_rows)
    _write_csv(output / "real_rank_tail_gradients.csv", tail_rows)
    _write_json(output / "fold_graph_receipts.json", graph_receipts)
    _write_json(output / "fold_context_receipts.json", context_receipts)
    _write_json(output / "checkpoint_manifest.json", checkpoint_rows)
    npmi = np.asarray(
        [row["npmi_improvement_positive_favors_full"] for row in fold_rows]
    )
    cv = np.asarray([row["c_v_difference_positive_favors_full"] for row in fold_rows])
    nll = np.asarray(
        [row["nll_reduction_positive_favors_full"] for row in fold_rows]
    )
    summary = {
        "schema_version": 1,
        "mode": "step5",
        "status": "PASS",
        "official_class_loaded_from": str(loaded_path),
        "folds": len(fold_rows),
        "paired_continuation_fits": 2 * len(fold_rows),
        "all_initial_states_exactly_matched": all(
            bool(row["initial_states_exactly_matched"]) for row in fold_rows
        ),
        "mean_npmi_improvement": float(npmi.mean()),
        "median_npmi_improvement": float(np.median(npmi)),
        "positive_npmi_fold_fraction": float(np.mean(npmi > 0.0)),
        "mean_c_v_difference": float(cv.mean()),
        "median_c_v_difference": float(np.median(cv)),
        "mean_nll_reduction": float(nll.mean()),
        "median_nll_reduction": float(np.median(nll)),
        "minimum_rank_6_to_10_gradient": float(
            min(row["minimum_rank_6_to_10_gradient"] for row in tail_rows)
        ),
        "minimum_rank_6_to_10_positive_fraction": float(
            min(row["positive_rank_6_to_10_fraction"] for row in tail_rows)
        ),
        "full_topic_stability": matched_topic_stability(full_top_words),
        "without_lexical_topic_stability": matched_topic_stability(nolex_top_words),
        "checkpoint_manifest": checkpoint_rows,
        "minimum_full_topic_diversity": float(
            min(row["topic_diversity_full"] for row in fold_rows)
        ),
        "maximum_full_top_word_redundancy": float(
            max(row["top_word_redundancy_full"] for row in fold_rows)
        ),
        "validation_documents_used": 0,
        "test_documents_used": 0,
        "labels_used": 0,
        "baseline_or_sota_outputs_used": 0,
    }
    _write_json(output / "worker_result.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = _load_job(args.job.resolve())
    output = Path(job["worker_output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch = _seed_everything(int(job["training"]["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("V6 Step 4/5 requires the already verified CUDA runtime")
    runtime = _runtime_receipt(torch)
    _write_json(output / "runtime_receipt.json", runtime)
    evidence = _load_evidence(job)
    if job["mode"] == "step4":
        _run_step4(job, evidence, output, torch)
    else:
        _run_step5(job, evidence, output, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
