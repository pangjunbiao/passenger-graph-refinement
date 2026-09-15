"""Isolated exact-runtime worker for the V6 EnCOT reproduction gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import types
from typing import Any, Mapping

import numpy as np
from scipy import sparse


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dense_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _maximum_state_error(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        return float("inf")
    maximum = 0.0
    for name in left:
        a = left[name].detach().cpu()
        b = right[name].detach().cpu()
        if a.dtype.is_floating_point:
            error = float((a - b).abs().max())
            if not np.isfinite(error):
                return float("inf")
            maximum = max(maximum, error)
        elif not bool((a == b).all()):
            return float("inf")
    return maximum


def _load_official_newmethod(source_root: Path) -> tuple[type, Path]:
    """Load the exact file without executing topmost/__init__, main, or scripts."""

    model_dir = source_root / "topmost" / "models" / "NewMethod"
    model_path = model_dir / "NewMethod.py"
    package_name = "_sc_htm_v6_pinned_encot_core"
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    module_name = package_name + ".NewMethod"
    specification = importlib.util.spec_from_file_location(module_name, model_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load pinned EnCOT model file: {model_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != model_path.resolve():
        raise RuntimeError("Python loaded a non-pinned EnCOT NewMethod class")
    return module.NewMethod, loaded_path


def _seed_everything(seed: int) -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    return torch


def _runtime_receipt(torch: Any) -> dict[str, Any]:
    names = (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "gensim",
        "tqdm",
        "wandb",
        "topmost",
        "geomloss",
    )
    return {
        "schema_version": 1,
        "python": list(sys.version_info[:3]),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {name: importlib.metadata.version(name) for name in names},
        "environment_created": False,
        "packages_installed_or_changed": False,
    }


def _training_only_global_contexts(
    train_counts: sparse.csr_matrix,
    train_embeddings: np.ndarray,
    *,
    clusters: int,
    seed: int,
    n_init: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.cluster import KMeans

    if not 1 < int(clusters) < train_counts.shape[0]:
        raise ValueError("invalid global-context cluster count")
    estimator = KMeans(
        n_clusters=int(clusters),
        random_state=int(seed),
        n_init=int(n_init),
        algorithm="lloyd",
    )
    labels = estimator.fit_predict(train_embeddings)
    dense_counts = train_counts.toarray().astype(np.float32)
    global_bow = np.zeros((int(clusters), dense_counts.shape[1]), dtype=np.float32)
    np.add.at(global_bow, labels, dense_counts)
    sizes = np.bincount(labels, minlength=int(clusters))
    if np.any(sizes <= 0) or np.any(global_bow.sum(axis=1) <= 0):
        raise ValueError("a fitted training-only global context is empty")
    model_input = np.concatenate((dense_counts, global_bow[labels]), axis=1)
    return model_input, {
        "schema_version": 1,
        "algorithm": "sklearn.cluster.KMeans",
        "algorithm_parameter": "lloyd",
        "n_clusters": int(clusters),
        "n_init": int(n_init),
        "seed": int(seed),
        "fit_partition": "original_training_only",
        "documents_fit": int(train_counts.shape[0]),
        "validation_documents_fit": 0,
        "test_documents_fit": 0,
        "labels_used": False,
        "minimum_cluster_size": int(sizes.min()),
        "maximum_cluster_size": int(sizes.max()),
        "cluster_size_sum": int(sizes.sum()),
        "inertia": float(estimator.inertia_),
        "centroids_logical_sha256": _dense_hash(
            np.asarray(estimator.cluster_centers_, dtype=np.float32)
        ),
        "assignments_logical_sha256": _dense_hash(
            np.asarray(labels, dtype=np.int64)
        ),
        "global_bow_logical_sha256": _dense_hash(global_bow),
        "model_input_shape": list(map(int, model_input.shape)),
    }


def _maximum_mapping_error(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        return float("inf")
    maximum = 0.0
    for key in left:
        a = left[key]
        b = right[key]
        if hasattr(a, "detach") and hasattr(b, "detach"):
            error = float((a.detach() - b.detach()).abs().max())
            if not np.isfinite(error):
                return float("inf")
            maximum = max(maximum, error)
        elif float(a) != float(b):
            return float("inf")
    return maximum


def _run_parity(
    NewMethod: type,
    *,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    import torch

    from src.v6.encot_bridge import (
        ExactEnCOTAdapter,
        city_conditioned_affinity,
        official_affinity_from_logits,
        official_affinity_logits,
        reporting_topic_distribution,
    )

    rng = np.random.default_rng(int(seed))
    vocabulary = 13
    topics = 4
    dimension = 7
    clusters = 3
    documents = 6
    word_embeddings = rng.normal(size=(vocabulary, dimension)).astype(np.float32)
    word_embeddings /= np.linalg.norm(word_embeddings, axis=1, keepdims=True)
    kwargs = {
        "vocab_size": vocabulary,
        "num_topics": topics,
        "en_units": 11,
        "dropout": 0.0,
        "pretrained_WE": word_embeddings,
        "embed_size": dimension,
        "beta_temp": 0.4,
        "weight_loss_ECR": 0.7,
        "sinkhorn_alpha": 4.0,
        "sinkhorn_max_iter": 30,
        "alpha_noise": 0.1,
        "alpha_augment": 0.03,
        "num_clusters": clusters,
        "weight_ot_doc_cluster": 0.2,
        "weight_ot_topic_cluster": 0.3,
    }
    torch.manual_seed(int(seed))
    direct = NewMethod(**kwargs).cpu()
    wrapped_model = NewMethod(**kwargs).cpu()
    wrapped_model.load_state_dict(direct.state_dict())
    direct.eval()
    wrapped_model.eval()
    adapter = ExactEnCOTAdapter(wrapped_model)
    local = rng.poisson(1.7, size=(documents, vocabulary)).astype(np.float32) + 1.0
    global_counts = rng.poisson(2.4, size=(documents, vocabulary)).astype(np.float32) + 1.0
    input_data = torch.from_numpy(np.concatenate((local, global_counts), axis=1))

    with torch.no_grad():
        direct_beta = direct.get_beta()
        bridge_logits = official_affinity_logits(
            direct.topic_embeddings, direct.word_embeddings, direct.beta_temp
        )
        bridge_beta = official_affinity_from_logits(bridge_logits)
        adapter_beta = adapter.official_affinity()
        direct_theta = direct.get_theta(input_data)
        adapter_inference = adapter.infer(input_data)
        direct_forward = direct(input_data)
        adapter_forward = adapter(input_data)
        reporting = reporting_topic_distribution(direct_beta)
        zeros = torch.zeros((3, topics, vocabulary), dtype=direct_beta.dtype)
        pooled = city_conditioned_affinity(
            bridge_logits, zeros, residual_scale=0.0
        )

    beta_formula_error = float((direct_beta - bridge_beta).abs().max())
    beta_adapter_error = float((direct_beta - adapter_beta).abs().max())
    theta_adapter_error = float(
        (direct_theta - adapter_inference.document_topic_mass).abs().max()
    )
    forward_error = _maximum_mapping_error(direct_forward, adapter_forward)
    pooled_error = float(
        (pooled - direct_beta.unsqueeze(0).expand_as(pooled)).abs().max()
    )
    probability_sum_error = float(
        (adapter_inference.word_probabilities.sum(dim=1) - 1.0).abs().max()
    )
    official_column_sum_error = float(
        (direct_beta.sum(dim=0) - 1.0).abs().max()
    )
    reporting_row_sum_error = float((reporting.sum(dim=1) - 1.0).abs().max())
    rank_preservation = bool(
        torch.equal(
            torch.argsort(direct_beta, dim=1, descending=True, stable=True),
            torch.argsort(reporting, dim=1, descending=True, stable=True),
        )
    )

    direct.train()
    wrapped_model.train()
    direct_optimizer = torch.optim.Adam(direct.parameters(), lr=0.002)
    adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=0.002)
    torch.manual_seed(int(seed) + 1)
    direct_loss = direct(input_data)["loss"]
    direct_optimizer.zero_grad()
    direct_loss.backward()
    direct_optimizer.step()
    torch.manual_seed(int(seed) + 1)
    adapter_loss = adapter(input_data)["loss"]
    adapter_optimizer.zero_grad()
    adapter_loss.backward()
    adapter_optimizer.step()
    one_step_loss_error = abs(float(direct_loss.detach()) - float(adapter_loss.detach()))
    one_step_state_error = _maximum_state_error(
        direct.state_dict(), wrapped_model.state_dict()
    )

    checks = {
        "official_beta_formula_matches": beta_formula_error <= tolerance,
        "adapter_beta_matches": beta_adapter_error <= tolerance,
        "adapter_theta_mass_matches": theta_adapter_error <= tolerance,
        "adapter_forward_components_match": forward_error <= tolerance,
        "adapter_one_step_loss_matches": one_step_loss_error <= tolerance,
        "adapter_one_step_state_matches": one_step_state_error <= tolerance,
        "extensions_off_equals_official_affinity": pooled_error <= tolerance,
        "decoded_probabilities_sum_to_one": probability_sum_error <= tolerance,
        "official_affinity_is_column_normalized": official_column_sum_error <= tolerance,
        "reporting_topics_are_row_normalized": reporting_row_sum_error <= tolerance,
        "reporting_normalization_preserves_top_word_ranks": rank_preservation,
    }
    return {
        "schema_version": 1,
        "device": "cpu",
        "absolute_tolerance": float(tolerance),
        "errors": {
            "official_beta_formula": beta_formula_error,
            "adapter_beta": beta_adapter_error,
            "adapter_theta_mass": theta_adapter_error,
            "adapter_forward_components": forward_error,
            "adapter_one_step_loss": one_step_loss_error,
            "adapter_one_step_state": one_step_state_error,
            "extensions_off_parent_affinity": pooled_error,
            "decoded_probability_sum": probability_sum_error,
            "official_column_sum": official_column_sum_error,
            "reporting_row_sum": reporting_row_sum_error,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _gradient_norm(model: Any) -> float:
    squares = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            squares += float((gradient * gradient).sum().cpu())
    return float(np.sqrt(squares))


def _run_real_training_smoke(
    NewMethod: type,
    train_input: np.ndarray,
    word_embeddings: np.ndarray,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from src.v6.encot_bridge import ExactEnCOTAdapter, reporting_topic_distribution

    if not torch.cuda.is_available():
        raise RuntimeError("The locked Step-3 EnCOT smoke requires the existing CUDA runtime")
    device = torch.device("cuda")
    seed = int(configuration["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = NewMethod(
        int(word_embeddings.shape[0]),
        num_topics=int(configuration["topics"]),
        num_clusters=int(configuration["num_clusters"]),
        dropout=float(configuration["dropout"]),
        pretrained_WE=np.asarray(word_embeddings, dtype=np.float32),
        embed_size=int(word_embeddings.shape[1]),
        beta_temp=float(configuration["beta_temp"]),
        weight_loss_ECR=float(configuration["weight_loss_ECR"]),
        weight_ot_doc_cluster=float(configuration["weight_ot_doc_cluster"]),
        weight_ot_topic_cluster=float(configuration["weight_ot_topic_cluster"]),
        sinkhorn_alpha=float(configuration["sinkhorn_alpha"]),
        sinkhorn_max_iter=int(configuration["sinkhorn_max_iter"]),
        alpha_noise=float(configuration["alpha_noise"]),
        alpha_augment=float(configuration["alpha_augment"]),
    ).to(device)
    initial_state = _state_hash(model.state_dict())
    tensor = torch.from_numpy(np.asarray(train_input, dtype=np.float32))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        tensor,
        batch_size=int(configuration["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(configuration["learning_rate"])
    )
    epoch_rows: list[dict[str, float | int]] = []
    finite_gradients = True
    maximum_gradient_norm = 0.0
    for epoch in range(1, int(configuration["epochs_smoke_only"]) + 1):
        model.train()
        sums: dict[str, float] = {}
        documents = 0
        for batch_cpu in loader:
            batch = batch_cpu.to(device)
            result = model(batch)
            if not all(bool(torch.isfinite(value).all()) for value in result.values()):
                raise FloatingPointError("non-finite official EnCOT smoke loss")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            norm = _gradient_norm(model)
            finite_gradients = finite_gradients and np.isfinite(norm)
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            optimizer.step()
            batch_size = int(batch.shape[0])
            documents += batch_size
            for name, value in result.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        epoch_rows.append(
            {
                "epoch": epoch,
                "documents": documents,
                **{name: value / documents for name, value in sums.items()},
            }
        )
    final_state = _state_hash(model.state_dict())
    model.eval()
    adapter = ExactEnCOTAdapter(model)
    with torch.no_grad():
        official_beta = model.get_beta()
        reporting_beta = reporting_topic_distribution(official_beta)
        beta_column_error = float(
            (official_beta.sum(dim=0) - 1.0).abs().max().cpu()
        )
        reporting_row_error = float(
            (reporting_beta.sum(dim=1) - 1.0).abs().max().cpu()
        )
        minimum_probability = 1.0
        probability_sum_error = 0.0
        theta_mass_minimum = float("inf")
        theta_mass_maximum = 0.0
        for start in range(0, len(tensor), int(configuration["batch_size"])):
            batch = tensor[start : start + int(configuration["batch_size"])].to(device)
            inference = adapter.infer(batch)
            minimum_probability = min(
                minimum_probability,
                float(inference.word_probabilities.min().cpu()),
            )
            probability_sum_error = max(
                probability_sum_error,
                float(
                    (inference.word_probabilities.sum(dim=1) - 1.0)
                    .abs()
                    .max()
                    .cpu()
                ),
            )
            mass_sums = inference.document_topic_mass.sum(dim=1)
            theta_mass_minimum = min(theta_mass_minimum, float(mass_sums.min().cpu()))
            theta_mass_maximum = max(theta_mass_maximum, float(mass_sums.max().cpu()))
    return {
        "schema_version": 1,
        "scope": "original_training_partition_smoke_only_no_quality_metrics",
        "device": str(device),
        "training_documents": int(train_input.shape[0]),
        "validation_documents_used_for_fit": 0,
        "test_documents_used_for_fit": 0,
        "vocabulary_size": int(word_embeddings.shape[0]),
        "embedding_dimension": int(word_embeddings.shape[1]),
        "topics": int(configuration["topics"]),
        "cluster_embeddings": int(configuration["num_clusters"]),
        "epochs_completed": int(configuration["epochs_smoke_only"]),
        "batches_per_epoch": int(len(loader)),
        "epoch_loss_components": epoch_rows,
        "all_loss_components_finite": True,
        "all_gradient_norms_finite": bool(finite_gradients),
        "maximum_gradient_norm": maximum_gradient_norm,
        "nonzero_gradient_observed": bool(maximum_gradient_norm > 0.0),
        "parameter_state_changed": initial_state != final_state,
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "official_beta_shape": list(map(int, official_beta.shape)),
        "official_beta_column_sum_error": beta_column_error,
        "reporting_beta_row_sum_error": reporting_row_error,
        "decoded_probability_sum_error": probability_sum_error,
        "minimum_decoded_probability": minimum_probability,
        "document_topic_mass_row_sum_range": [theta_mass_minimum, theta_mass_maximum],
        "document_topic_mass_renormalized": False,
        "quality_metrics_computed": 0,
        "hyperparameters_selected": 0,
    }


def run_job(job_path: Path) -> Path:
    job = json.loads(job_path.read_text(encoding="utf-8-sig"))
    output = Path(job["worker_output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(job["official_source"]).resolve()
    torch = _seed_everything(int(job["official_parent"]["seed"]))
    NewMethod, loaded_path = _load_official_newmethod(source)
    train_counts = sparse.load_npz(job["train_counts"]).tocsr().astype(np.float32)
    with np.load(job["train_embeddings"], allow_pickle=False) as archive:
        train_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    with np.load(job["word_embeddings"], allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if train_embeddings.shape[0] != train_counts.shape[0]:
        raise ValueError("training counts and document embeddings are misaligned")
    if word_embeddings.shape[0] != train_counts.shape[1]:
        raise ValueError("training counts and word embeddings are misaligned")
    if word_embeddings.shape[1] != 768:
        raise ValueError("the frozen V6 vocabulary embedding width is not 768")

    model_input, context = _training_only_global_contexts(
        train_counts,
        train_embeddings,
        clusters=int(job["global_context"]["clusters"]),
        seed=int(job["global_context"]["seed"]),
        n_init=int(job["global_context"]["n_init"]),
    )
    parity = _run_parity(
        NewMethod,
        seed=int(job["official_parent"]["seed"]) + 17,
        tolerance=float(job["parity"]["absolute_tolerance"]),
    )
    smoke = _run_real_training_smoke(
        NewMethod, model_input, word_embeddings, job["official_parent"]
    )
    source_text = loaded_path.read_text(encoding="utf-8")
    interface = {
        "schema_version": 1,
        "loaded_class_file": str(loaded_path),
        "loaded_class_is_exact_pinned_file": loaded_path
        == (source / "topmost/models/NewMethod/NewMethod.py").resolve(),
        "official_beta_softmax_axis_is_topics_dim0": (
            "F.softmax(-dist / self.beta_temp, dim=0)" in source_text
        ),
        "official_theta_is_unrenormalized_elementwise_product": (
            "return global_theta * local_noise_theta" in source_text
        ),
        "official_decoder_uses_vocabulary_softmax": (
            "recon = F.softmax(self.decoder_bn" in source_text
        ),
        "official_main_executed": False,
        "official_shell_scripts_executed": False,
        "official_topmost_package_initializer_executed": False,
    }
    runtime = _runtime_receipt(torch)
    _write_json(output / "runtime_receipt.json", runtime)
    _write_json(output / "official_interface_audit.json", interface)
    _write_json(output / "bridge_parity.json", parity)
    _write_json(output / "training_only_global_context_contract.json", context)
    _write_json(output / "real_training_smoke.json", smoke)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "loaded_official_class": str(loaded_path),
        "parity_passed": bool(parity["passed"]),
        "worker_output": str(output),
    }
    if not parity["passed"]:
        result["status"] = "FAIL"
    _write_json(output / "worker_result.json", result)
    if result["status"] != "PASS":
        raise AssertionError("official EnCOT bridge parity failed")
    print(
        "[V6 Step 3 worker] PASS | exact source loaded | parity=PASS | "
        f"training-smoke epochs={smoke['epochs_completed']} device={smoke['device']}"
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run_job(arguments.job.expanduser().resolve())
