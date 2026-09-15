"""Isolated worker for the three official recent-SOTA implementations.

This file is executed by a method-specific virtual environment.  It deliberately
has no imports from the SC-HTM package: the only shared interface is the signed
JSON job plus frozen numeric inputs and the canonical receipt on disk.
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
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
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
        raise ValueError("Official output must be a finite nonnegative matrix")
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Official output contains an empty probability row")
    return matrix / totals


def _validate_probabilities(
    values: np.ndarray, *, documents: int, vocabulary: int
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (documents, vocabulary):
        raise ValueError(
            f"Native probability shape {matrix.shape} != {(documents, vocabulary)}"
        )
    if (
        not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError("Native word probabilities are not row-simplex values")
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
        vocabulary_rows = sorted(csv.DictReader(stream), key=lambda row: int(row["index"]))
    vocabulary = tuple(row["token"] for row in vocabulary_rows)
    train_counts = sparse.load_npz(input_dir / "X_train.npz").tocsr()
    test_counts = sparse.load_npz(input_dir / "X_test.npz").tocsr()
    if embedded_tokens != vocabulary:
        raise ValueError("Frozen vocabulary embeddings are not aligned to the vocabulary")
    if train_counts.shape[1] != len(vocabulary) or test_counts.shape[1] != len(vocabulary):
        raise ValueError("Count and vocabulary dimensions differ")
    if word_embeddings.shape[0] != len(vocabulary):
        raise ValueError("Word-embedding and vocabulary dimensions differ")
    if train_embeddings.shape[0] != train_counts.shape[0]:
        raise ValueError("Training count and embedding rows differ")
    if test_embeddings.shape[0] != test_counts.shape[0]:
        raise ValueError("Test count and embedding rows differ")
    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError("Training and test embedding dimensions differ")
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


def _training_only_global_contexts(
    data: dict[str, Any], *, clusters: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Aggregate training counts by frozen-embedding K-means clusters."""

    from sklearn.cluster import KMeans

    if not 1 < clusters < data["train_counts"].shape[0]:
        raise ValueError("The global-context cluster count is invalid")
    estimator = KMeans(n_clusters=clusters, random_state=seed, n_init=10)
    train_labels = estimator.fit_predict(data["train_embeddings"])
    test_labels = estimator.predict(data["test_embeddings"])
    dense_train = data["train_counts"].toarray().astype(np.float32)
    dense_test = data["test_counts"].toarray().astype(np.float32)
    global_bow = np.zeros((clusters, dense_train.shape[1]), dtype=np.float32)
    np.add.at(global_bow, train_labels, dense_train)
    if np.any(global_bow.sum(axis=1) <= 0.0):
        raise ValueError("A fitted global context is empty")
    train_input = np.concatenate((dense_train, global_bow[train_labels]), axis=1)
    test_input = np.concatenate((dense_test, global_bow[test_labels]), axis=1)
    audit = {
        "algorithm": "sklearn.cluster.KMeans",
        "n_clusters": int(clusters),
        "n_init": 10,
        "fit_partition": "train",
        "test_assignment": "predict_to_frozen_training_centroids",
        "train_cluster_minimum": int(np.bincount(train_labels, minlength=clusters).min()),
        "train_cluster_maximum": int(np.bincount(train_labels, minlength=clusters).max()),
        "test_evidence_used_to_fit_contexts": 0,
    }
    return train_input, test_input, audit


class _FrozenPreprocess:
    """Expose the frozen Step-3 count matrix through FASTopic's sparse API."""

    def __init__(self, train_bow: sparse.spmatrix, vocabulary: tuple[str, ...]):
        if not sparse.issparse(train_bow):
            raise TypeError("FASTopic requires a SciPy sparse training matrix")
        matrix = train_bow.tocsr().astype(np.float32, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if matrix.ndim != 2 or matrix.shape[1] != len(vocabulary):
            raise ValueError("FASTopic count and vocabulary dimensions differ")
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0.0):
            raise ValueError("FASTopic counts must be finite and nonnegative")
        if np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0.0):
            raise ValueError("FASTopic received an empty training document")
        self.train_bow = matrix
        self.vocabulary = list(vocabulary)

    def preprocess(self, docs: list[str]) -> dict[str, Any]:
        if len(docs) != self.train_bow.shape[0]:
            raise ValueError("FASTopic received an unexpected training document bank")
        return {"train_bow": self.train_bow.copy(), "vocab": list(self.vocabulary)}


class _NoDownloadEmbedder:
    def encode(self, docs: Any, **_: Any) -> np.ndarray:
        raise RuntimeError("The official worker requires preset frozen embeddings")


def _validated_word_embeddings(
    word_embeddings: np.ndarray, vocabulary_size: int
) -> np.ndarray:
    matrix = np.asarray(word_embeddings, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != vocabulary_size
        or matrix.shape[1] <= 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("The frozen vocabulary-embedding matrix is invalid")
    return matrix


def _fit_fastopic(
    data: dict[str, Any], hyperparameters: dict[str, Any], topics: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    del seed
    import torch
    from fastopic import FASTopic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FASTopic(
        num_topics=topics,
        preprocess=_FrozenPreprocess(data["train_counts"], data["vocabulary"]),
        device=device,
        normalize_embeddings=bool(hyperparameters["normalize_embeddings"]),
        doc_embed_model=_NoDownloadEmbedder(),
        DT_alpha=float(hyperparameters["DT_alpha"]),
        TW_alpha=float(hyperparameters["TW_alpha"]),
        theta_temp=float(hyperparameters["theta_temp"]),
        verbose=False,
    )
    document_ids = [f"train_{index}" for index in range(data["train_counts"].shape[0])]
    model.fit(
        document_ids,
        epochs=int(hyperparameters["epochs"]),
        learning_rate=float(hyperparameters["learning_rate"]),
        preset_doc_embeddings=data["train_embeddings"],
    )
    beta = model.get_beta()
    test_embeddings = torch.as_tensor(data["test_embeddings"], device=device)
    theta = model.transform(doc_embeddings=test_embeddings)
    # FASTopic's official reconstruction is theta @ beta.  Convert the
    # learned transport rows to topic-word simplexes before held-out scoring,
    # matching its public topic-distribution interpretation.
    probability = _normalize_rows(theta) @ _normalize_rows(beta)
    probability = _normalize_rows(probability)
    return beta, theta, probability, {
        "device": device,
        "iterations_completed": int(hyperparameters["epochs"]),
        "converged": None,
        "frozen_document_embeddings_supplied": True,
        "frozen_count_vocabulary_supplied": True,
        "training_bow_format": "scipy.sparse.csr_matrix_float32",
        "predictive_interface": (
            "official_fastopic_theta_times_row_normalized_transport_beta"
        ),
    }


class _OfficialDataset:
    def __init__(
        self,
        *,
        train_input: np.ndarray,
        test_input: np.ndarray,
        word_embeddings: np.ndarray,
        vocabulary: tuple[str, ...],
        batch_size: int,
        device: str,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader

        self.vocab = list(vocabulary)
        self.vocab_size = len(vocabulary)
        self.pretrained_WE = _validated_word_embeddings(
            word_embeddings, self.vocab_size
        )
        self.word_embedding_size = int(self.pretrained_WE.shape[1])
        self.train_data = torch.as_tensor(train_input, dtype=torch.float32, device=device)
        self.test_data = torch.as_tensor(test_input, dtype=torch.float32, device=device)
        self.train_dataloader = DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=False
        )


def _native_decoder_outputs(
    model: Any,
    input_values: Any,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the official held-out encoder and its trained native decoder."""

    import torch

    model.eval()
    theta_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    vocabulary = int(model.vocab_size)
    with torch.no_grad():
        beta = model.get_beta()
        for start in range(0, int(input_values.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(input_values.shape[0]))
            batch = input_values[start:stop]
            theta = model.get_theta(batch)
            probability = torch.softmax(model.decoder_bn(theta @ beta), dim=-1)
            theta_rows.append(theta.detach().cpu().numpy().astype(np.float64))
            probability_rows.append(
                probability.detach().cpu().numpy().astype(np.float64)
            )
    theta = np.concatenate(theta_rows, axis=0)
    probability = _validate_probabilities(
        np.concatenate(probability_rows, axis=0),
        documents=int(input_values.shape[0]),
        vocabulary=vocabulary,
    )
    return theta, probability


def _fit_glocom(
    data: dict[str, Any], hyperparameters: dict[str, Any], topics: int,
    seed: int, source_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    source = str(source_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.GloCOM.GloCOM import GloCOM
    from trainer.Trainer import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_input, test_input, context_audit = _training_only_global_contexts(
        data, clusters=int(hyperparameters["global_context_clusters"]), seed=seed
    )
    dataset = _OfficialDataset(
        train_input=train_input,
        test_input=test_input,
        word_embeddings=data["word_embeddings"],
        vocabulary=data["vocabulary"],
        batch_size=int(hyperparameters["batch_size"]),
        device=device,
    )
    model = GloCOM(
        dataset.vocab_size,
        num_topics=topics,
        pretrained_WE=dataset.pretrained_WE,
        aug_coef=float(hyperparameters["aug_coef"]),
        prior_var=float(hyperparameters["prior_var"]),
        weight_loss_ECR=float(hyperparameters["weight_loss_ECR"]),
    ).to(device)
    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=int(hyperparameters["epochs"]),
        learning_rate=float(hyperparameters["learning_rate"]),
        batch_size=int(hyperparameters["batch_size"]),
        verbose=False,
    )
    trainer.train()
    theta, probability = _native_decoder_outputs(
        model, dataset.test_data, batch_size=int(hyperparameters["batch_size"])
    )
    return trainer.get_beta(), theta, probability, {
        "device": device,
        "iterations_completed": int(hyperparameters["epochs"]),
        "converged": None,
        "global_context_audit": context_audit,
        "frozen_vocabulary_embeddings_supplied": True,
        "predictive_interface": (
            "official_glocom_get_theta_then_decoder_batchnorm_vocabulary_softmax"
        ),
    }


def _fit_encot(
    data: dict[str, Any], hyperparameters: dict[str, Any], topics: int,
    seed: int, source_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    source = str(source_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    import wandb
    from topmost.models.NewMethod.NewMethod import NewMethod
    from topmost.trainers.basic_trainer import BasicTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_input, test_input, context_audit = _training_only_global_contexts(
        data, clusters=int(hyperparameters["global_context_clusters"]), seed=seed
    )
    dataset = _OfficialDataset(
        train_input=train_input,
        test_input=test_input,
        word_embeddings=data["word_embeddings"],
        vocabulary=data["vocabulary"],
        batch_size=int(hyperparameters["batch_size"]),
        device=device,
    )
    model = NewMethod(
        dataset.vocab_size,
        num_topics=topics,
        # EnCOT initializes cluster embeddings from ``embed_size`` while its
        # topic embeddings inherit the supplied word-embedding width.  Bind
        # both spaces to the frozen Step-4 width so the two OT costs are valid.
        embed_size=dataset.word_embedding_size,
        num_clusters=int(hyperparameters["num_clusters"]),
        dropout=float(hyperparameters["dropout"]),
        beta_temp=float(hyperparameters["beta_temp"]),
        weight_loss_ECR=float(hyperparameters["weight_loss_ECR"]),
        weight_ot_doc_cluster=float(hyperparameters["weight_ot_doc_cluster"]),
        weight_ot_topic_cluster=float(hyperparameters["weight_ot_topic_cluster"]),
        sinkhorn_alpha=float(hyperparameters["sinkhorn_alpha"]),
        sinkhorn_max_iter=int(hyperparameters["sinkhorn_max_iter"]),
        alpha_noise=float(hyperparameters["alpha_noise"]),
        alpha_augment=float(hyperparameters["alpha_augment"]),
        pretrained_WE=dataset.pretrained_WE,
    ).to(device)
    run = wandb.init(mode="disabled", reinit=True)
    try:
        trainer = BasicTrainer(
            model=model,
            dataset=dataset,
            epochs=int(hyperparameters["epochs"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            batch_size=int(hyperparameters["batch_size"]),
            verbose=False,
        )
        trainer.train()
        beta = trainer.get_beta()
        theta, probability = _native_decoder_outputs(
            model, dataset.test_data,
            batch_size=int(hyperparameters["batch_size"]),
        )
    finally:
        if run is not None:
            run.finish()
    return beta, theta, probability, {
        "device": device,
        "iterations_completed": int(hyperparameters["epochs"]),
        "converged": None,
        "global_context_audit": context_audit,
        "frozen_vocabulary_embeddings_supplied": True,
        "word_topic_cluster_embedding_dimension": dataset.word_embedding_size,
        "embedding_dimension_source": "frozen_step4_vocabulary_embeddings",
        "predictive_interface": (
            "official_encot_product_theta_then_decoder_batchnorm_vocabulary_softmax"
        ),
    }


def _environment_fingerprint(torch: Any) -> tuple[str, dict[str, Any]]:
    frozen_packages = _run_text(sys.executable, "-m", "pip", "freeze").splitlines()
    details = {
        "python": sys.version,
        "executable_name": Path(sys.executable).name,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze": sorted(frozen_packages, key=str.casefold),
    }
    return _sha256_object(details), details


def run_job(job_path: Path) -> Path:
    job = json.loads(job_path.read_text(encoding="utf-8-sig"))
    method_id = str(job["method_id"])
    seed = int(job["seed"])
    topics = int(job["topics"])
    input_dir = Path(job["input_dir"]).resolve()
    source_dir = Path(job["source_dir"]).resolve() if job.get("source_dir") else None
    receipt_dir = Path(job["receipt_dir"]).resolve()
    _verify_adapter_bundle(input_dir, job)
    data = _load_inputs(input_dir)
    torch = _seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Official recent-SOTA execution requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    started = time.perf_counter()

    try:
        if method_id == "fastopic":
            beta, theta, word_probability, fit_metadata = _fit_fastopic(
                data, dict(job["hyperparameters"]), topics, seed
            )
        elif method_id == "glocom":
            if source_dir is None:
                raise ValueError("GloCOM requires its clean official source checkout")
            beta, theta, word_probability, fit_metadata = _fit_glocom(
                data, dict(job["hyperparameters"]), topics, seed, source_dir
            )
        elif method_id == "encot":
            if source_dir is None:
                raise ValueError("EnCOT requires its clean official source checkout")
            beta, theta, word_probability, fit_metadata = _fit_encot(
                data, dict(job["hyperparameters"]), topics, seed, source_dir
            )
        else:
            raise ValueError(f"Unsupported official method: {method_id}")
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
    expected_shape = (topics, len(data["vocabulary"]))
    if beta.shape != expected_shape:
        raise ValueError(f"Official beta shape {beta.shape} != {expected_shape}")
    expected_theta = (data["test_counts"].shape[0], topics)
    if theta.shape != expected_theta:
        raise ValueError(f"Official test-theta shape {theta.shape} != {expected_theta}")
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
        "count_vocabulary_adapter": "frozen_step3",
        "document_embedding_adapter": "frozen_step4_multilingual_e5",
        "heldout_inference": "frozen_model_test_theta",
    }
    metadata = {
        "schema_version": 1,
        "method_id": method_id,
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
        "execution_policy_id": job["execution_policy_id"],
        "execution_policy_sha256": job["execution_policy_sha256"],
        "allowed_overrides": job["allowed_overrides"],
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
        f"OFFICIAL SOTA RECEIPT PASS | method={method_id} seed={seed} "
        f"beta={beta.shape} test_theta={theta.shape}"
    )
    print(f"receipt={receipt_dir}")
    return receipt_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one signed official-SOTA job")
    parser.add_argument("--job", required=True, type=Path)
    return parser


if __name__ == "__main__":
    run_job(_parser().parse_args().job)
