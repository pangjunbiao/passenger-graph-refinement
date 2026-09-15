import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

import src.v6.step8_worker as worker
from src.v6.data_contract import sparse_logical_hash
from src.v6.development_evidence import dense_hash
from src.v6.development_worker import _state_hash
from src.v6.final_refit import FINAL_VARIANTS
from src.v6.step5r2_worker import _array_hash


class _FakeParent(torch.nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.vocab_size = int(beta.shape[1])
        self.register_buffer("beta", torch.as_tensor(beta, dtype=torch.float32))
        self.decoder_bn = torch.nn.Identity()

    def get_beta(self):
        return self.beta

    def noise_local_encode(self, values):
        theta = torch.softmax(values[:, :2] + 0.1, dim=1)
        return theta, theta

    def global_encode(self, values):
        theta = torch.softmax(values[:, :2] + 0.2, dim=1)
        return theta, theta


class _DeviceAwareRoundedParent(_FakeParent):
    """Emulate a CPU/CUDA difference without changing the saved state."""

    def __init__(self, beta, cpu_offset, runtime_offset):
        super().__init__(beta)
        self.cpu_offset = float(cpu_offset)
        self.runtime_offset = float(runtime_offset)
        self.runtime_transfer_performed = False

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.runtime_transfer_performed = True
        return result

    def get_beta(self):
        offset = (
            self.runtime_offset
            if self.runtime_transfer_performed
            else self.cpu_offset
        )
        return self.beta + offset


def _small_counts(rows, vocabulary):
    values = np.zeros((rows, vocabulary), dtype=np.int64)
    for row in range(rows):
        values[row, row % vocabulary] += 2
        values[row, (row + 1) % vocabulary] += 1
        values[row, (row + 4) % vocabulary] += 1
    return sparse.csr_matrix(values)


def test_worker_reproduces_78_assigned_77_modeled_metadata_contract(
    tmp_path, monkeypatch
):
    cities = ["beijing"] * 44 + ["shanghai"] * 17 + ["xiamen"] * 16
    assignments = pd.DataFrame(
        {
            "post_id": [f"test-{index:03d}" for index in range(78)],
            "city": cities + ["beijing"],
            "split": ["test"] * 78,
            "included_in_model": [True] * 77 + [False],
        }
    )
    path = tmp_path / "split_assignments.csv"
    assignments.to_csv(path, index=False, encoding="utf-8-sig")
    source_hash = worker.sha256_file(path)
    monkeypatch.setattr(worker, "EXPECTED_STEP2_SPLIT_METADATA_SHA256", source_hash)
    observed = worker._modeled_test_metadata(assignments)
    body = {
        "schema_version": 1,
        "source": {"path": str(path), "sha256": source_hash},
        "selection_rule": "split_is_test_and_included_in_model_is_true",
        "assigned_documents_before_model_exclusions": 78,
        "modeled_documents": 77,
        "excluded_documents": 1,
        "modeled_city_documents": {
            "beijing": 44,
            "shanghai": 17,
            "xiamen": 16,
        },
        "modeled_rows_logical_sha256": worker.sha256_object(
            observed["modeled_records"]
        ),
        "excluded_rows_logical_sha256": worker.sha256_object(
            observed["excluded_records"]
        ),
    }
    contract = {**body, "contract_identity": worker.sha256_object(body)}
    job = {
        "test_metadata_contract": contract,
        "test_source_contract": {
            "expected_assignment_documents_before_model_exclusions": 78,
            "expected_source_documents": 77,
            "expected_pre_matrix_exclusions": 1,
            "expected_vocabulary": 1148,
            "expected_cities": ["beijing", "shanghai", "xiamen"],
            "expected_legacy_v4_sparse_logical_sha256": (
                worker.EXPECTED_TEST_SPARSE_LOGICAL_SHA256
            ),
        },
    }
    worker._verify_test_metadata_contract(job)
    assignments.loc[0, "included_in_model"] = False
    assignments.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="split metadata changed"):
        worker._verify_test_metadata_contract(job)


def test_test_completion_freezes_once_then_recovers_without_source_reopen(tmp_path):
    vocabulary = 12
    counts = _small_counts(6, vocabulary)
    rows = pd.DataFrame(
        {
            "matrix_row": range(6),
            "post_id": [f"test-{index}" for index in range(6)],
            "city": ["beijing", "shanghai", "xiamen"] * 2,
            "split": ["test"] * 6,
        }
    )
    counts_path = tmp_path / "X_test.npz"
    rows_path = tmp_path / "rows_test.csv"
    sparse.save_npz(counts_path, counts)
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    modeled_records = (
        rows[["post_id", "city", "split"]]
        .astype(str)
        .sort_values(["post_id", "city", "split"])
        .to_dict(orient="records")
    )
    job = {
        "job_identity": "frozen-job",
        "test_sources": {
            "counts": {"path": str(counts_path)},
            "rows": {"path": str(rows_path)},
        },
        "test_source_contract": {
            "expected_source_documents": 6,
            "expected_vocabulary": vocabulary,
            "expected_cities": ["beijing", "shanghai", "xiamen"],
            "expected_legacy_v4_sparse_logical_sha256": sparse_logical_hash(counts),
        },
        "test_metadata_contract": {
            "modeled_documents": 6,
            "modeled_city_documents": {
                "beijing": 2,
                "shanghai": 2,
                "xiamen": 2,
            },
            "modeled_rows_logical_sha256": worker.sha256_object(modeled_records),
        },
        "technical_recovery": {"recovery_identity": "technical-recovery"},
        "document_completion": {
            "seed": 21,
            "observed_fraction": 0.5,
            "minimum_total_tokens": 2,
        },
        "data_policy": {
            "accurate_claim": "one frozen confirmation access",
        },
    }
    development_rows = pd.DataFrame({"post_id": ["train-a", "train-b"]})
    output = tmp_path / "worker"
    output.mkdir()
    first = worker._freeze_or_recover_completion(job, output, development_rows)
    assert first[-1] is False
    access = json.loads((output / "test_access_state.json").read_text())
    assert access["cumulative_test_source_openings"] == 2
    assert access["cumulative_evaluative_confirmations"] == 1
    counts_path.write_bytes(b"the source is no longer readable as an npz")
    rows_path.write_text("changed", encoding="utf-8")
    second = worker._freeze_or_recover_completion(job, output, development_rows)
    assert second[-1] is True
    assert (first[0] != second[0]).nnz == 0
    assert (first[1] != second[1]).nnz == 0


def test_frozen_seed_evaluation_constructs_exact_five_variant_grid(monkeypatch):
    documents = 6
    vocabulary = 12
    topics = 2
    beta = np.vstack(
        (
            np.linspace(0.05, 0.95, vocabulary),
            np.linspace(0.95, 0.05, vocabulary),
        )
    ).astype(np.float32)
    effective = beta.copy()
    effective[:, [2, 7]] = effective[::-1, [2, 7]]
    model = _FakeParent(beta)
    monkeypatch.setattr(
        worker, "_new_parent", lambda *args, **kwargs: _FakeParent(beta)
    )
    development_counts = _small_counts(18, vocabulary)
    word_embeddings = np.zeros((vocabulary, 3), dtype=np.float32)
    word_embeddings[:, 0] = np.linspace(0.1, 1.0, vocabulary)
    word_embeddings[:, 1] = np.linspace(1.0, 0.1, vocabulary)
    word_embeddings[:, 2] = 0.5
    word_embeddings /= np.linalg.norm(word_embeddings, axis=1, keepdims=True)
    centroids = word_embeddings[[0, 6]].astype(np.float32)
    global_bow = np.vstack(
        (
            np.asarray(development_counts[:9].sum(axis=0)).reshape(-1),
            np.asarray(development_counts[9:].sum(axis=0)).reshape(-1),
        )
    ).astype(np.float32)
    context_audit = {
        "centroids_sha256": dense_hash(centroids),
        "global_bow_sha256": dense_hash(global_bow),
        "test_documents_fit": 0,
    }
    prior = np.full(vocabulary, 1.0 / vocabulary, dtype=np.float32)
    payload = {
        "parent_state_dict": model.state_dict(),
        "parent_state_sha256": _state_hash(model.state_dict()),
        "base_affinity": torch.from_numpy(beta),
        "effective_affinity": torch.from_numpy(effective),
        "calibration_scale": torch.ones(vocabulary),
        "calibration_shift": torch.zeros(vocabulary),
        "calibration_changed_mask": torch.zeros(vocabulary, dtype=torch.bool),
        "calibration_audit": {},
        "pooled_prior": torch.from_numpy(prior),
        "pooled_mixture_weight": 0.35,
        "context_centroids": torch.from_numpy(centroids),
        "context_global_bow": torch.from_numpy(global_bow),
        "context_audit": context_audit,
    }
    test_counts = _small_counts(documents, vocabulary)
    test_rows = pd.DataFrame(
        {
            "post_id": [f"test-{index}" for index in range(documents)],
            "city": ["beijing", "shanghai", "xiamen"] * 2,
            "split": ["test"] * documents,
        }
    )
    job = {
        "model_configuration": {"frozen_word_feature_width": 3},
        "evaluation": {
            "inference_batch_size": 4,
            "top_words": 10,
            "c_v_window_size": 110,
            "c_v_gamma": 1.0,
        },
        "document_completion": {"probability_floor": 1.0e-12},
        "test_source_contract": {"expected_cities": ["beijing", "shanghai", "xiamen"]},
        "frozen_model": {"variants": list(FINAL_VARIANTS)},
    }
    development = {
        "counts": development_counts,
        "word_embeddings": word_embeddings,
        "vocabulary": pd.DataFrame(
            {"index": range(vocabulary), "token": [f"w{i}" for i in range(vocabulary)]}
        ),
    }
    rows, city_rows, top_rows, integrity = worker._evaluate_seed(
        job,
        {
            "seed": 101,
            "base_affinity_sha256": _array_hash(beta),
            "effective_affinity_sha256": _array_hash(effective),
        },
        payload,
        development,
        test_counts,
        test_counts,
        test_rows,
        _FakeParent,
        device=torch.device("cpu"),
        torch=torch,
    )
    assert [row["variant"] for row in rows] == list(FINAL_VARIANTS)
    assert len(city_rows) == len(FINAL_VARIANTS) * 3
    assert len(top_rows) == 2 * topics * 10
    assert integrity["parent_state_unchanged"] is True
    assert integrity["parameters_updated"] == 0
    # Equivalent float32 execution paths can differ by a few ULPs across
    # platforms/backends. Keep this unit-test tolerance ten times tighter than
    # the frozen Step-8 integrity gate (1e-6) without requiring bitwise identity.
    assert integrity["maximum_parent_probability_reproduction_error"] <= 1.0e-7


def test_parent_preflight_gates_same_runtime_not_cross_device_roundoff(monkeypatch):
    beta = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    frozen_parent = _FakeParent(beta)
    payloads = [
        (
            {"seed": 101},
            {
                "parent_state_dict": frozen_parent.state_dict(),
                "parent_state_sha256": _state_hash(frozen_parent.state_dict()),
                "base_affinity": torch.from_numpy(beta),
            },
        )
    ]
    job = {
        "model_configuration": {},
        "guardrails": {"maximum_parent_probability_reproduction_error": 1.0e-6},
    }
    development = {"word_embeddings": np.zeros((2, 2), dtype=np.float32)}

    monkeypatch.setattr(
        worker,
        "_new_parent",
        lambda *args, **kwargs: _DeviceAwareRoundedParent(
            beta, cpu_offset=4.0e-6, runtime_offset=5.0e-8
        ),
    )
    audit = worker._preflight_parent_restoration(
        _DeviceAwareRoundedParent,
        job,
        development,
        payloads,
        device=torch.device("cpu"),
        torch=torch,
    )
    assert audit["maximum_cpu_backend_difference"] > 1.0e-6
    assert 0.0 < audit["maximum_runtime_device_reproduction_error"] <= 1.0e-6

    monkeypatch.setattr(
        worker,
        "_new_parent",
        lambda *args, **kwargs: _DeviceAwareRoundedParent(
            beta, cpu_offset=0.0, runtime_offset=2.0e-6
        ),
    )
    with pytest.raises(ValueError, match="same-runtime parent affinity reproduction failed"):
        worker._preflight_parent_restoration(
            _DeviceAwareRoundedParent,
            job,
            development,
            payloads,
            device=torch.device("cpu"),
            torch=torch,
        )
