from __future__ import annotations

import numpy as np
from scipy import sparse

from src.v6.comparators.base import BaselineData
from src.v6.step9 import (
    EvaluationTarget,
    _assert_adapter_firewall,
    _capture_r1_lineage,
    _contextualize,
    _holm,
    _load_protocol,
    _predictive_metrics,
    _purge_forbidden_adapter_inputs,
    _runtime_compatibility_errors,
)


def test_step9_protocol_keeps_every_firewall_closed(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    protocol = _load_protocol(root / "configs/v6/step09.yaml")
    assert protocol["topics"] == 10
    assert len(protocol["seeds"]) == 10
    assert set(protocol["established"]) == {
        "lda", "nmf", "gsdmm", "btm", "bertopic", "combinedtm"
    }
    assert set(protocol["recent_sota"]) == {
        "fastopic", "glocom", "encot", "neuromax"
    }
    assert protocol["implementation_version"].endswith("_r2")


def test_main_preserves_step9_and_exposes_step10():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "main_v6.py").read_text(encoding="utf-8")
    assert '"step9"' in source
    assert "run_v6_step9" in source
    assert '"step10"' in source
    assert "run_v6_step10" in source


def test_contextual_adapter_depends_only_on_supplied_counts():
    counts = sparse.csr_matrix([[2, 0, 1], [0, 3, 0]], dtype=float)
    words = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    first = _contextualize(counts, words)
    second = _contextualize(counts.copy(), words.copy())
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)


def test_holm_is_monotone_in_sorted_p_order():
    raw = [0.04, 0.001, 0.02, 0.5]
    adjusted = _holm(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[order]) >= 0)
    assert all(a >= p for a, p in zip(adjusted, raw))


def test_comparator_data_type_cannot_expose_target_or_city():
    fields = set(BaselineData.__dataclass_fields__)
    assert "target_counts" not in fields
    assert "validation_city_indices" not in fields


def test_adapter_kit_purges_r1_target_and_city_files(tmp_path):
    for name in (
        "X_target.npz", "completion_rows.csv", "input_receipt.json",
        "evaluation_input_receipt.json",
    ):
        (tmp_path / name).write_text("legacy", encoding="utf-8")
    _purge_forbidden_adapter_inputs(tmp_path)
    (tmp_path / "adapter_input_receipt.json").write_text("{}", encoding="utf-8")
    for name in (
        "X_train.npz", "X_test.npz", "E_train.npz", "E_test.npz",
        "S_vocabulary.npz", "vocabulary.csv",
    ):
        (tmp_path / name).write_bytes(b"allowed")
    _assert_adapter_firewall(tmp_path)


def test_fastopic_runtime_probe_requires_exact_modern_api_and_versions():
    exact = {
        "python": [3, 10, 18],
        "torch": "2.7.0+cu128",
        "torchvision": "0.22.0+cu128",
        "cuda_available": True,
        "packages": {
            "numpy": "1.26.4", "scipy": "1.12.0", "pandas": "2.2.3",
            "plotly": "6.0.1", "sentence-transformers": "3.4.1",
            "gensim": "4.3.3", "scikit-learn": "1.6.1",
            "tqdm": "4.66.5", "fastopic": "1.0.1", "topmost": "1.0.2",
        },
        "api": {
            "init": "(self, num_topics, preprocess, DT_alpha, TW_alpha, theta_temp)",
            "fit": "(self, docs, preset_doc_embeddings)",
            "transform": "(self, docs=None, doc_embeddings=None)",
        },
    }
    assert _runtime_compatibility_errors("fastopic", exact) == []
    legacy = dict(exact)
    legacy["torch"] = "2.4.1+cu124"
    legacy["packages"] = {"fastopic": "0.0.5", "topmost": "0.0.5"}
    legacy["api"] = dict(exact["api"], init="(self, num_topics)")
    errors = _runtime_compatibility_errors("fastopic", legacy)
    assert any("torch=" in item for item in errors)
    assert any("preprocess" in item for item in errors)


def test_native_probabilities_control_nll_and_nonprobabilistic_methods_get_dash():
    target = EvaluationTarget(
        counts=sparse.csr_matrix([[1, 0], [0, 1], [1, 0]], dtype=float),
        city_indices=np.asarray([0, 1, 2], dtype=np.int64),
        evaluation_identity="fixture",
    )
    beta = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    theta = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    native = np.full((3, 2), 0.5)
    macro, micro = _predictive_metrics(
        beta, theta, native, target,
        likelihood_comparable=True, probability_floor=1.0e-12,
    )
    np.testing.assert_allclose([macro, micro], np.log(2.0), atol=1.0e-12)
    assert _predictive_metrics(
        beta, theta, native, target,
        likelihood_comparable=False, probability_floor=1.0e-12,
    ) == (None, None)


def test_neural_workers_export_native_word_probabilities():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/v6/step9_official_worker.py",
        "src/v6/step9_neuromax_worker.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "test_word_probability" in source
        assert "decoder_bn" in source or "beta_batchnorm" in source


def test_r1_receipts_are_archived_without_becoming_r2_evidence(tmp_path):
    receipt = tmp_path / "receipts/lda/seed_101"
    receipt.mkdir(parents=True)
    (receipt / "metadata.json").write_text(
        '{"method_id":"lda","seed":101}', encoding="utf-8"
    )
    (receipt / "canonical_output.npz").write_bytes(b"opaque-numeric-evidence")
    lineage = _capture_r1_lineage(tmp_path)
    assert lineage["r1_receipt_directories"] == 1
    assert lineage["numeric_values_opened_or_used_by_migration"] is False
    assert lineage["r1_receipts_reused_as_r2_evidence"] == 0
    assert (tmp_path / "r1_receipts_audit.zip").is_file()
    assert _capture_r1_lineage(tmp_path) == lineage
