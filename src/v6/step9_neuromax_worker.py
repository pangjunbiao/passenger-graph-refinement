"""Isolated official NeuroMax worker for the signed V6.1 Step-9 comparison.

The worker consumes only the frozen comparator-visible numeric input bundle.  It never
reads raw feedback, labels, requirement names, validation scores, or outputs
from another method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import numpy as np
from scipy import sparse


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_object(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_text(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("NeuroMax output must be a finite nonnegative matrix")
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("NeuroMax output contains an empty probability row")
    return matrix / totals


def _validate_probabilities(
    values: np.ndarray, *, documents: int, vocabulary: int
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (documents, vocabulary):
        raise ValueError(
            f"NeuroMax probability shape {matrix.shape} != {(documents, vocabulary)}"
        )
    if (
        not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError("NeuroMax native probabilities are not row-simplex values")
    return matrix


def _verify_adapter_bundle(input_dir: Path, job: dict[str, Any]) -> None:
    receipt_path = input_dir / "adapter_input_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (
        receipt.get("schema_version") != 2
        or receipt.get("adapter_input_identity") != job["input_fingerprint"]
        or receipt.get("adapter_bundle_sha256") != job["input_bundle_sha256"]
        or receipt.get("target_files_present") != 0
        or receipt.get("city_or_label_files_present") != 0
    ):
        raise ValueError("Comparator-visible input receipt does not match the signed job")
    files = receipt.get("files", [])
    observed = []
    for item in files:
        path = input_dir / str(item["name"])
        if (
            not path.is_file()
            or _sha256_file(path) != item["sha256"]
            or path.stat().st_size != int(item["size_bytes"])
        ):
            raise ValueError(f"Comparator-visible input changed: {item['name']}")
        observed.append({
            "name": str(item["name"]),
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size_bytes"]),
        })
    if _sha256_object({"files": observed}) != job["input_bundle_sha256"]:
        raise ValueError("Comparator-visible input bundle identity is invalid")


def _load_inputs(input_dir: Path) -> dict[str, Any]:
    forbidden = (
        "X_target.npz", "completion_rows.csv", "input_receipt.json",
        "evaluation_input_receipt.json",
    )
    leaked = [name for name in forbidden if (input_dir / name).exists()]
    if leaked:
        raise RuntimeError(f"Comparator input firewall exposed forbidden files: {leaked}")
    with np.load(input_dir / "E_train.npz", allow_pickle=False) as archive:
        train_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    with np.load(input_dir / "E_test.npz", allow_pickle=False) as archive:
        test_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    with np.load(input_dir / "S_vocabulary.npz", allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        embedded_tokens = tuple(archive["tokens"].astype(str))

    import csv

    with (input_dir / "vocabulary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["index"]))
    vocabulary = tuple(row["token"] for row in rows)
    train_counts = sparse.load_npz(input_dir / "X_train.npz").tocsr()
    test_counts = sparse.load_npz(input_dir / "X_test.npz").tocsr()

    if embedded_tokens != vocabulary:
        raise ValueError("Vocabulary embeddings are not aligned to the vocabulary")
    if train_counts.shape[1] != len(vocabulary):
        raise ValueError("Training count and vocabulary dimensions differ")
    if test_counts.shape[1] != len(vocabulary):
        raise ValueError("Test count and vocabulary dimensions differ")
    if word_embeddings.shape[0] != len(vocabulary):
        raise ValueError("Word-embedding and vocabulary dimensions differ")
    if train_embeddings.shape[0] != train_counts.shape[0]:
        raise ValueError("Training count and embedding rows differ")
    if test_embeddings.shape[0] != test_counts.shape[0]:
        raise ValueError("Test count and embedding rows differ")
    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError("Training and test embedding dimensions differ")
    if np.any(np.asarray(train_counts.sum(axis=1)).ravel() <= 0.0):
        raise ValueError("NeuroMax received an empty training document")
    if np.any(np.asarray(test_counts.sum(axis=1)).ravel() <= 0.0):
        raise ValueError("NeuroMax received an empty test document")
    return {
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
        "word_embeddings": word_embeddings,
        "vocabulary": vocabulary,
    }


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


def _adapt_context_embeddings(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    *,
    output_dimension: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the unavoidable 768-to-384 adapter on training evidence only."""

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    input_dimension = int(train_embeddings.shape[1])
    if input_dimension != int(test_embeddings.shape[1]):
        raise ValueError("Train/test contextual dimensions differ")
    if output_dimension <= 0 or output_dimension > min(train_embeddings.shape):
        raise ValueError("The requested train-only PCA dimension is invalid")
    estimator = PCA(n_components=output_dimension, svd_solver="full")
    train = estimator.fit_transform(train_embeddings)
    test = estimator.transform(test_embeddings)
    train = normalize(train, norm="l2", axis=1).astype(np.float32, copy=False)
    test = normalize(test, norm="l2", axis=1).astype(np.float32, copy=False)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("PCA contextual adapter produced non-finite values")
    if np.any(np.linalg.norm(train, axis=1) <= 0.0):
        raise ValueError("PCA contextual adapter produced an empty training vector")
    if np.any(np.linalg.norm(test, axis=1) <= 0.0):
        raise ValueError("PCA contextual adapter produced an empty test vector")
    audit = {
        "algorithm": "sklearn.decomposition.PCA",
        "svd_solver": "full",
        "input_dimension": input_dimension,
        "output_dimension": int(output_dimension),
        "fit_partition": "train",
        "test_operation": "transform_with_frozen_training_PCA",
        "post_transform_normalization": "row_L2",
        "explained_variance_ratio_sum": float(
            estimator.explained_variance_ratio_.sum()
        ),
        "test_documents_used_to_fit_transform": 0,
        "labels_used_to_fit_transform": 0,
    }
    return train, test, audit


class _DictTensorDataset:
    def __init__(self, data: Any, contextual: Any) -> None:
        if data.shape[0] != contextual.shape[0]:
            raise ValueError("BoW and contextual rows differ")
        self.data = data
        self.contextual = contextual

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "data": self.data[index],
            "contextual_embed": self.contextual[index],
        }


class _OfficialNeuroMaxDataset:
    """Minimal dataset interface consumed by the official BasicTrainer."""

    def __init__(
        self,
        *,
        train_counts: sparse.spmatrix,
        test_counts: sparse.spmatrix,
        train_context: np.ndarray,
        test_context: np.ndarray,
        word_embeddings: np.ndarray,
        vocabulary: tuple[str, ...],
        batch_size: int,
        device: str,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader

        train_bow = np.asarray(train_counts.toarray(), dtype=np.float32)
        test_bow = np.asarray(test_counts.toarray(), dtype=np.float32)
        self.vocab = list(vocabulary)
        self.vocab_size = len(vocabulary)
        self.pretrained_WE = np.asarray(word_embeddings, dtype=np.float32)
        if self.pretrained_WE.shape[0] != self.vocab_size:
            raise ValueError("NeuroMax word embeddings are misaligned")
        self.train_data = torch.as_tensor(train_bow, device=device)
        self.test_data = torch.as_tensor(test_bow, device=device)
        train_context_tensor = torch.as_tensor(train_context, device=device)
        test_context_tensor = torch.as_tensor(test_context, device=device)
        self.train_contextual = train_context_tensor
        self.test_contextual = test_context_tensor
        self.train_dataloader = DataLoader(
            _DictTensorDataset(self.train_data, train_context_tensor),
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )


def _fit_neuromax(
    data: dict[str, Any],
    hyperparameters: dict[str, Any],
    adapter: dict[str, Any],
    *,
    topics: int,
    source_dir: Path,
    torch: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source = str(source_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from NeuroMax.NeuroMax import NeuroMax
    from basic_trainer import BasicTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapted_train, adapted_test, adapter_audit = _adapt_context_embeddings(
        data["train_embeddings"],
        data["test_embeddings"],
        output_dimension=int(adapter["required_model_dimension"]),
    )
    dataset = _OfficialNeuroMaxDataset(
        train_counts=data["train_counts"],
        test_counts=data["test_counts"],
        train_context=adapted_train,
        test_context=adapted_test,
        word_embeddings=data["word_embeddings"],
        vocabulary=data["vocabulary"],
        batch_size=int(hyperparameters["batch_size"]),
        device=device,
    )
    model = NeuroMax(
        vocab_size=dataset.vocab_size,
        num_topics=int(topics),
        num_groups=int(hyperparameters["num_groups"]),
        dropout=float(hyperparameters["dropout"]),
        pretrained_WE=dataset.pretrained_WE,
        beta_temp=float(hyperparameters["beta_temp"]),
        weight_loss_ECR=float(hyperparameters["weight_loss_ECR"]),
        weight_loss_GR=float(hyperparameters["weight_loss_GR"]),
        alpha_ECR=float(hyperparameters["alpha_ECR"]),
        alpha_GR=float(hyperparameters["alpha_GR"]),
        sinkhorn_max_iter=int(hyperparameters["sinkhorn_max_iter"]),
        weight_loss_InfoNCE=float(hyperparameters["weight_loss_InfoNCE"]),
    ).to(device)
    trainer = BasicTrainer(
        model,
        epochs=int(hyperparameters["epochs"]),
        learning_rate=float(hyperparameters["learning_rate"]),
        batch_size=int(hyperparameters["batch_size"]),
        lr_scheduler=str(hyperparameters["lr_scheduler"]),
        lr_step_size=int(hyperparameters["lr_step_size"]),
    )
    trainer.train(dataset, verbose=False)
    beta = trainer.export_beta()
    model.eval()
    theta_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    with torch.no_grad():
        native_beta = model.get_beta()
        for start in range(0, int(dataset.test_data.shape[0]), int(hyperparameters["batch_size"])):
            stop = min(
                start + int(hyperparameters["batch_size"]),
                int(dataset.test_data.shape[0]),
            )
            batch = dataset.test_data[start:stop]
            native_theta = model.get_theta(batch)
            native_probability = torch.softmax(
                model.decoder_bn(native_theta @ native_beta), dim=-1
            )
            theta_rows.append(
                native_theta.detach().cpu().numpy().astype(np.float64)
            )
            probability_rows.append(
                native_probability.detach().cpu().numpy().astype(np.float64)
            )
    theta = np.concatenate(theta_rows, axis=0)
    probability = _validate_probabilities(
        np.concatenate(probability_rows, axis=0),
        documents=data["test_counts"].shape[0],
        vocabulary=len(data["vocabulary"]),
    )
    return beta, theta, probability, {
        "device": device,
        "iterations_completed": int(hyperparameters["epochs"]),
        "converged": None,
        "frozen_count_vocabulary_supplied": True,
        "frozen_vocabulary_embeddings_supplied": True,
        "frozen_contextual_embeddings_supplied": True,
        "contextual_adapter_audit": adapter_audit,
        "official_source_modified": False,
        "predictive_interface": (
            "official_neuromax_get_theta_then_decoder_batchnorm_vocabulary_softmax"
        ),
    }


def _environment_fingerprint(torch: Any) -> tuple[str, dict[str, Any]]:
    freeze = _run_text(sys.executable, "-m", "pip", "freeze").splitlines()
    details = {
        "python": sys.version,
        "executable_name": Path(sys.executable).name,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze": sorted(freeze, key=str.casefold),
    }
    return _sha256_object(details), details


def run_job(job_path: Path) -> Path:
    job = json.loads(job_path.read_text(encoding="utf-8-sig"))
    if job.get("method_id") != "neuromax":
        raise ValueError("The NeuroMax worker received a different method")
    seed = int(job["seed"])
    topics = int(job["topics"])
    input_dir = Path(job["input_dir"]).resolve()
    source_dir = Path(job["source_dir"]).resolve()
    receipt_dir = Path(job["receipt_dir"]).resolve()
    _verify_adapter_bundle(input_dir, job)
    if _run_text("git", "status", "--porcelain", "--untracked-files=all", cwd=source_dir):
        raise RuntimeError("Official NeuroMax source tree is dirty")
    if _run_text("git", "rev-parse", "HEAD", cwd=source_dir) != str(
        job["source_revision"]
    ):
        raise RuntimeError("Official NeuroMax source revision changed")

    data = _load_inputs(input_dir)
    torch = _seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Official NeuroMax execution requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        beta, theta, word_probability, fit_metadata = _fit_neuromax(
            data,
            dict(job["hyperparameters"]),
            dict(job["compatibility_adapter"]),
            topics=topics,
            source_dir=source_dir,
            torch=torch,
        )
    finally:
        runtime_seconds = time.perf_counter() - started
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    fit_metadata.update(
        {
            "runtime_seconds": float(runtime_seconds),
            "peak_python_memory_mib": float(peak_python_bytes / (1024.0 ** 2)),
            "peak_cuda_memory_mib": float(
                torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            ),
        }
    )

    beta = _normalize_rows(beta)
    theta = _normalize_rows(theta)
    expected_beta = (topics, len(data["vocabulary"]))
    expected_theta = (data["test_counts"].shape[0], topics)
    if beta.shape != expected_beta:
        raise ValueError(f"NeuroMax beta shape {beta.shape} != {expected_beta}")
    if theta.shape != expected_theta:
        raise ValueError(f"NeuroMax test-theta shape {theta.shape} != {expected_theta}")
    word_probability = _validate_probabilities(
        word_probability,
        documents=data["test_counts"].shape[0],
        vocabulary=len(data["vocabulary"]),
    )

    receipt_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = receipt_dir / "canonical_output.npz"
    np.savez_compressed(
        arrays_path,
        topic_word=beta,
        test_theta=theta,
        test_word_probability=word_probability,
    )
    environment_fingerprint, environment = _environment_fingerprint(torch)
    resolved = {
        "number_of_topics": topics,
        "reporting_seed": seed,
        **dict(job["hyperparameters"]),
        "compatibility_adapter": dict(job["compatibility_adapter"]),
        "heldout_inference": "frozen_model_test_theta",
    }
    metadata = {
        "schema_version": 1,
        "method_id": "neuromax",
        "display_name": "NeuroMax",
        "seed": seed,
        "official_source_url": job["official_source_url"],
        "source_kind": job["source_kind"],
        "source_revision": job["source_revision"],
        "source_artifact_sha256": job["source_artifact_sha256"],
        "source_tree_clean": True,
        "environment_fingerprint": environment_fingerprint,
        "environment": environment,
        "model_configuration_id": job["model_configuration_id"],
        "input_fingerprint": job["input_fingerprint"],
        "input_bundle_sha256": job["input_bundle_sha256"],
        "canonical_output_sha256": _sha256_file(arrays_path),
        "hyperparameters_tuned_after_freeze": False,
        "winner_selected": False,
        "test_documents_scored": int(data["test_counts"].shape[0]),
        "labels_accessed": 0,
        "requirement_names_accessed": 0,
        "raw_text_accessed": 0,
        "posthoc_comparator_extension": True,
        "execution_lock_sha256": job["execution_lock_sha256"],
        "resolved_hyperparameters": resolved,
        "resolved_hyperparameters_sha256": _sha256_object(resolved),
        "adapter_source_sha256": _sha256_file(Path(__file__).resolve()),
        "training_completed": True,
        "model_frozen_during_test_inference": True,
        "likelihood_comparable": True,
        "target_available_to_worker": False,
        "city_or_label_available_to_worker": False,
        "implementation_version": job["implementation_version"],
        "receipt_protocol": job["receipt_protocol"],
        "execution_identity": job["execution_identity"],
        **fit_metadata,
    }
    metadata_path = receipt_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"NEUROMAX RECEIPT PASS | seed={seed} beta={beta.shape} "
        f"test_theta={theta.shape}"
    )
    print(f"receipt={receipt_dir}")
    return receipt_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one signed NeuroMax job")
    parser.add_argument("--job", required=True, type=Path)
    return parser


if __name__ == "__main__":
    run_job(_parser().parse_args().job)
