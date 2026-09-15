from pathlib import Path
import inspect

import numpy as np
import pytest
from scipy import sparse
import yaml

from src.v6.official_encot import _runtime_matches
from src.v6.step3 import IMPLEMENTATION_VERSION, SCOPE


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    document = yaml.safe_load(
        (ROOT / "configs/v6/step03.yaml").read_text(encoding="utf-8")
    )
    return document["v6"]["step3"]


def test_step3_lock_pins_exact_official_commit_tree_and_files():
    protocol = _protocol()
    assert protocol["scope"] == SCOPE
    assert protocol["implementation_version"] == IMPLEMENTATION_VERSION
    official = protocol["official_source"]
    assert official["commit"] == "8ac3592165cc7851676be2776b314eb6e44e9388"
    assert official["git_tree"] == "4be831bd0ffb56a912af48cf980ec3fc9cc472dc"
    assert len(official["required_files_sha256"]) == 8
    assert all(len(value) == 64 for value in official["required_files_sha256"].values())
    assert official["redistribute_source_in_v6_zip"] is False


def test_step3_is_bound_to_the_returned_step2_data_key():
    protocol = _protocol()
    prerequisite = protocol["prerequisite"]
    assert prerequisite["status"] == "PASS"
    assert prerequisite["data_key"] == (
        "7774658044d3e297ed5bff31a45bcad5221228a53fe48fd5aabacebf12778c2d"
    )


def test_runtime_match_requires_every_exact_version_and_cuda():
    lock = _protocol()["runtime"]
    probe = {
        "python": [3, 10, 9],
        "cuda_available": True,
        "packages": {
            "torch": "2.4.1+cu124",
            "torchvision": "0.19.1+cu124",
            "numpy": "1.26.3",
            "scipy": "1.10.1",
            "scikit-learn": "1.5.1",
            "sentence-transformers": "2.7.0",
            "gensim": "4.3.3",
            "tqdm": "4.66.5",
            "wandb": "0.18.1",
            "topmost": "0.0.5",
            "geomloss": "0.2.6",
        },
    }
    assert _runtime_matches(probe, lock)
    assert not _runtime_matches({**probe, "cuda_available": False}, lock)
    wrong = {**probe, "packages": {**probe["packages"], "torch": "2.7.0+cu128"}}
    assert not _runtime_matches(wrong, lock)


def test_step3_source_contains_no_package_install_execution():
    import src.v6.official_encot as official_encot
    import src.v6.step3 as step3

    source = inspect.getsource(official_encot) + inspect.getsource(step3)
    forbidden_calls = ("pip install", "conda install", "uv pip", "poetry add")
    assert not any(value in source.casefold() for value in forbidden_calls)
    assert _protocol()["runtime"]["creation_or_installation_allowed"] is False


def test_global_context_fixture_fits_training_rows_only():
    pytest.importorskip("sklearn")
    from src.v6.step3_worker import _training_only_global_contexts

    counts = sparse.csr_matrix(
        np.asarray(
            [[2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 1, 1]], dtype=np.float32
        )
    )
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    model_input, report = _training_only_global_contexts(
        counts, embeddings, clusters=2, seed=19, n_init=2
    )
    assert model_input.shape == (4, 6)
    assert report["fit_partition"] == "original_training_only"
    assert report["validation_documents_fit"] == 0
    assert report["test_documents_fit"] == 0
    assert report["labels_used"] is False
    assert report["cluster_size_sum"] == 4
