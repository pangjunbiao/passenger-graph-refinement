"""V6.1 Step 9: frozen, resumable, fair comparator execution.

No V4 numeric result is read.  Every comparator is refit on the same 695 rows;
test inference receives only the immutable Step-8 observed half and scoring
receives only its target half.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats

from src.v6.comparators.base import BaselineData, normalize_rows
from src.v6.comparators.registry import make_adapter
from src.v6.data_contract import sha256_file, sha256_object, sparse_logical_hash
from src.v6.evaluation import (
    counts_to_token_documents,
    cv_coherence,
    document_completion_nll,
    document_npmi,
    mean_top_word_jaccard,
    stable_top_word_indices,
    topic_diversity,
)
from src.v6.predictive_evaluation import (
    document_completion_nll_from_probabilities,
)

IMPLEMENTATION = "v6_1_step9_fair_comparators_r2"
RECEIPT_PROTOCOL = "v6_1_step9_receipt_r2_native_decoder_firewall"
READINESS = "READY_FOR_V6_1_STEP10_EXTERNAL_HUMAN_AND_PAPER_OUTPUTS"
DISPLAY = {
    "v6_1": "SC-HTM V6.1",
    "lda": "LDA", "nmf": "NMF", "gsdmm": "GSDMM", "btm": "BTM",
    "bertopic": "BERTopic", "combinedtm": "CombinedTM",
    "fastopic": "FASTopic", "glocom": "GloCOM", "encot": "EnCOT",
    "neuromax": "NeuroMax",
}
ESTABLISHED = ("lda", "nmf", "gsdmm", "btm", "bertopic", "combinedtm")
SOTA = ("fastopic", "glocom", "encot", "neuromax")
NATIVE_PROBABILITY_METHODS = ("combinedtm", *SOTA)
FORBIDDEN_ADAPTER_FILES = (
    "X_target.npz",
    "completion_rows.csv",
    "input_receipt.json",
    "evaluation_input_receipt.json",
)


@dataclass(frozen=True)
class EvaluationTarget:
    """Target and strata retained exclusively by the common evaluator."""

    counts: sparse.csr_matrix
    city_indices: np.ndarray
    evaluation_identity: str


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _verify_fingerprint(value: Mapping[str, Any], key: str) -> bool:
    body = dict(value)
    expected = body.pop(key, None)
    return isinstance(expected, str) and sha256_object(body) == expected


def _latest_step8(root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((root / "outputs/v6/step08_test_confirmation").glob("*/step08_contract.json"), reverse=True)
    for path in candidates:
        contract = _json(path)
        if contract.get("status") == "PASS" and contract.get("readiness") == "READY_FOR_V6_1_STEP9_FAIR_COMPARATORS" and _verify_fingerprint(contract, "contract_fingerprint"):
            return path.parent, contract
    raise RuntimeError("No fingerprint-valid passing Step-8 result was found")


def _verify_manifest(base: Path) -> None:
    manifest = _json(base / "generated_artifact_manifest.json")
    body = dict(manifest); identity = body.pop("manifest_identity", None)
    if sha256_object(body) != identity:
        raise ValueError("Step-8 artifact-manifest identity is invalid")
    for item in manifest["files"]:
        path = base / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Step-8 artifact changed: {item['relative_path']}")


def _load_protocol(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    p = doc.get("v6", {}).get("step9", {})
    if doc.get("schema_version") != 1 or p.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-9 configuration/implementation lock changed")
    if list(map(int, p["seeds"])) != [101,211,307,401,503,601,701,809,907,1009] or int(p["topics"]) != 10:
        raise ValueError("Step-9 seed or topic lock changed")
    policy = p["policy"]
    if any(policy[k] is not False for k in ("install_or_upgrade_torch","reuse_v4_numeric_results","model_selection_from_comparators","drop_failed_seeds","target_half_available_to_fit_or_inference","labels_available","requirement_names_available","v6_model_change_allowed")):
        raise ValueError("A Step-9 firewall was relaxed")
    if any(policy[k] is not False for k in (
        "target_or_city_serialized_to_comparator_input",
        "pseudo_nll_for_nonprobabilistic_methods",
    )):
        raise ValueError("An R2 evaluator firewall was relaxed")
    if any(policy[k] is not True for k in (
        "exact_runtime_version_and_api_preflight",
        "native_decoder_probabilities_for_neural_models",
    )):
        raise ValueError("An R2 reproducibility requirement was disabled")
    return p


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_npz_embeddings(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        key = "embeddings" if "embeddings" in z.files else z.files[0]
        return np.asarray(z[key], dtype=np.float64)


def _contextualize(counts: sparse.csr_matrix, word_embeddings: np.ndarray) -> np.ndarray:
    """Observed-count-only semantic summaries; no target/full-test embedding."""
    weighted = np.asarray(counts @ word_embeddings, dtype=np.float64)
    lengths = np.asarray(counts.sum(axis=1)).reshape(-1, 1)
    if np.any(lengths <= 0):
        raise ValueError("contextual adapter received an empty document")
    weighted /= lengths
    norm = np.linalg.norm(weighted, axis=1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(weighted).all():
        raise ValueError("contextual adapter produced an invalid vector")
    return (weighted / norm).astype(np.float32)


def _tokens(counts: sparse.csr_matrix) -> tuple[tuple[int, ...], ...]:
    return counts_to_token_documents(counts)


def _texts(docs: tuple[tuple[int, ...], ...], vocabulary: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(" ".join(vocabulary[i] for i in doc) for doc in docs)


def _purge_forbidden_adapter_inputs(kit: Path) -> None:
    """Remove R1 target-bearing files before any R2 comparator can start."""

    kit.mkdir(parents=True, exist_ok=True)
    for name in FORBIDDEN_ADAPTER_FILES:
        path = kit / name
        if path.is_file():
            path.unlink()


def _assert_adapter_firewall(kit: Path) -> None:
    leaked = [name for name in FORBIDDEN_ADAPTER_FILES if (kit / name).exists()]
    if leaked:
        raise RuntimeError(f"Comparator input firewall contains forbidden files: {leaked}")
    allowed = {
        "X_train.npz", "X_test.npz", "E_train.npz", "E_test.npz",
        "S_vocabulary.npz", "vocabulary.csv", "adapter_input_receipt.json",
    }
    unexpected = sorted(path.name for path in kit.iterdir() if path.is_file() and path.name not in allowed)
    if unexpected:
        raise RuntimeError(f"Comparator input firewall contains unexpected files: {unexpected}")


def _prepare_inputs(
    root: Path,
    out: Path,
    step8: Path,
    protocol: Mapping[str, Any],
) -> tuple[BaselineData, EvaluationTarget, dict[str, Any]]:
    del root
    handoff = _json(step8 / "worker/step9_handoff.json")
    prerequisite = protocol["prerequisite"]
    if handoff.get("handoff_identity") != prerequisite["step9_handoff_identity"]:
        raise ValueError("Step-9 handoff identity changed")
    completion = _json(step8 / "worker/test_completion_contract.json")
    if completion.get("completion_identity") != prerequisite["step8_completion_identity"]:
        raise ValueError("Step-8 completion identity changed")
    job = _json(step8 / "worker_job.json")
    train_path = _resolve(job["development_inputs"]["train_counts"]["path"])
    val_path = _resolve(job["development_inputs"]["spent_counts"]["path"])
    vocab_path = _resolve(job["development_inputs"]["vocabulary"]["path"])
    word_path = _resolve(job["development_inputs"]["word_embeddings"]["path"])
    train = sparse.load_npz(train_path).tocsr().astype(np.float64)
    val = sparse.load_npz(val_path).tocsr().astype(np.float64)
    fit = sparse.vstack((train, val), format="csr")
    observed = sparse.load_npz(step8 / "worker/test_completion_observed.npz").tocsr().astype(np.float64)
    target = sparse.load_npz(step8 / "worker/test_completion_target.npz").tocsr().astype(np.float64)
    rows = pd.read_csv(step8 / "worker/test_completion_rows.csv", encoding="utf-8-sig").sort_values("completion_row")
    vf = pd.read_csv(vocab_path, encoding="utf-8-sig").sort_values("index")
    vocabulary = tuple(vf["token"].astype(str))
    word = _load_npz_embeddings(word_path)
    if fit.shape != (695,1148) or observed.shape != target.shape or observed.shape != (75,1148) or word.shape[0] != 1148 or len(vocabulary) != 1148:
        raise ValueError("Frozen Step-9 input dimensions changed")
    city_order = ("beijing","shanghai","xiamen")
    city = np.asarray([city_order.index(str(x)) for x in rows["city"]], dtype=np.int64)
    fit_e = _contextualize(fit, word); observed_e = _contextualize(observed, word)
    fit_docs = _tokens(fit); obs_docs = _tokens(observed)
    comparison_lock_sha256 = sha256_object({
        "topics": protocol["topics"], "seeds": protocol["seeds"],
        "established": protocol["established"],
        "recent_sota": protocol["recent_sota"],
        "evaluation": protocol["evaluation"],
        "statistics": protocol["statistics"],
    })
    adapter_identity_body = {
        "fit_logical_sha256": sparse_logical_hash(fit), "observed_logical_sha256": sparse_logical_hash(observed),
        "vocabulary_sha256": sha256_file(vocab_path),
        "word_embeddings_sha256": sha256_file(word_path),
        "contextual_adapter": "count_weighted_frozen_vocabulary_embeddings_L2", "target_used_by_adapter": False,
        "city_or_label_used_by_adapter": False,
        "comparison_lock_sha256": comparison_lock_sha256,
        "adapter_contract": "train_plus_observed_only_no_target_no_city_r2",
    }
    adapter_identity = sha256_object(adapter_identity_body)
    data = BaselineData(
        train_counts=fit,
        validation_counts=observed,
        train_embeddings=fit_e,
        validation_embeddings=observed_e,
        train_documents=_texts(fit_docs, vocabulary),
        validation_documents=_texts(obs_docs, vocabulary),
        train_token_documents=fit_docs,
        validation_token_documents=obs_docs,
        vocabulary=vocabulary,
        input_fingerprint=adapter_identity,
        heldout_partition="test_completion_observed_half",
    )
    evaluation_identity_body = {
        "adapter_input_identity": adapter_identity,
        "target_logical_sha256": sparse_logical_hash(target),
        "city_indices_sha256": sha256_object(city.tolist()),
        "completion_identity": completion["completion_identity"],
        "target_tokens": int(target.sum()),
        "evaluation": protocol["evaluation"],
    }
    evaluation_identity = sha256_object(evaluation_identity_body)
    evaluation_target = EvaluationTarget(target, city, evaluation_identity)

    kit = out / "input_kit"
    _purge_forbidden_adapter_inputs(kit)
    sparse.save_npz(kit/"X_train.npz",fit); sparse.save_npz(kit/"X_test.npz",observed)
    np.savez_compressed(kit/"E_train.npz",embeddings=fit_e); np.savez_compressed(kit/"E_test.npz",embeddings=observed_e)
    np.savez_compressed(kit/"S_vocabulary.npz",embeddings=word.astype(np.float32),tokens=np.asarray(vocabulary))
    vf[["index","token"]].to_csv(kit/"vocabulary.csv",index=False,encoding="utf-8-sig")
    bundle_files = []
    for name in (
        "X_train.npz", "X_test.npz", "E_train.npz", "E_test.npz",
        "S_vocabulary.npz", "vocabulary.csv",
    ):
        path = kit / name
        bundle_files.append({"name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    adapter_bundle_sha256 = sha256_object({"files": bundle_files})
    adapter_receipt = {
        "schema_version": 2,
        **adapter_identity_body,
        "adapter_input_identity": adapter_identity,
        "adapter_bundle_sha256": adapter_bundle_sha256,
        "files": bundle_files,
        "fit_documents": 695,
        "observed_completion_documents": 75,
        "target_files_present": 0,
        "city_or_label_files_present": 0,
        "test_target_used_for_fit_inference_or_context": False,
    }
    _write_json(kit/"adapter_input_receipt.json",adapter_receipt)
    _assert_adapter_firewall(kit)
    evaluator_receipt = {
        "schema_version": 2,
        **evaluation_identity_body,
        "evaluation_identity": evaluation_identity,
        "adapter_bundle_sha256": adapter_bundle_sha256,
        "target_retention": "parent_evaluator_only_never_serialized_to_adapter_input_kit",
        "city_retention": "parent_evaluator_only_never_serialized_to_adapter_input_kit",
        "labels": 0,
    }
    _write_json(out/"evaluation_input_receipt.json", evaluator_receipt)
    return data, evaluation_target, {
        "adapter": adapter_receipt,
        "evaluation": evaluator_receipt,
    }


def _save_fit(path: Path, fit: Any, metadata: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"topic_word":np.asarray(fit.topic_word,dtype=np.float64)}
    if fit.validation_theta is not None: arrays["test_theta"] = np.asarray(fit.validation_theta,dtype=np.float64)
    if fit.validation_word_probabilities is not None:
        arrays["test_word_probability"] = np.asarray(
            fit.validation_word_probabilities, dtype=np.float64
        )
    np.savez_compressed(path/"canonical_output.npz",**arrays)
    body = dict(metadata); body["canonical_output_sha256"] = sha256_file(path/"canonical_output.npz")
    body["receipt_identity"] = sha256_object(body)
    _write_json(path/"metadata.json",body)


def _valid_receipt(
    path: Path,
    method: str,
    seed: int,
    input_identity: str,
    execution_identity: str,
) -> bool:
    try:
        m=_json(path/"metadata.json"); body=dict(m); ident=body.pop("receipt_identity")
        if not (
            m["method_id"] == method
            and int(m["seed"]) == seed
            and m["input_identity"] == input_identity
            and m.get("implementation_version") == IMPLEMENTATION
            and m.get("receipt_protocol") == RECEIPT_PROTOCOL
            and m.get("execution_identity") == execution_identity
            and sha256_object(body) == ident
            and sha256_file(path/"canonical_output.npz") == m["canonical_output_sha256"]
        ):
            return False
        with np.load(path/"canonical_output.npz", allow_pickle=False) as archive:
            beta = np.asarray(archive["topic_word"], dtype=np.float64)
            theta = np.asarray(archive["test_theta"], dtype=np.float64) if "test_theta" in archive.files else None
            probabilities = (
                np.asarray(archive["test_word_probability"], dtype=np.float64)
                if "test_word_probability" in archive.files else None
            )
        if (
            beta.shape != (10, 1148)
            or not np.isfinite(beta).all()
            or np.any(beta < 0.0)
            or not np.allclose(beta.sum(1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            return False
        if theta is not None and (
            theta.shape != (75, 10)
            or not np.isfinite(theta).all()
            or np.any(theta < 0.0)
            or not np.allclose(theta.sum(1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            return False
        if method in NATIVE_PROBABILITY_METHODS:
            if probabilities is None or probabilities.shape != (75, 1148):
                return False
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0.0)
                or np.any(probabilities > 1.0)
                or not np.allclose(probabilities.sum(1), 1.0, atol=1.0e-6, rtol=0.0)
            ):
                return False
        return True
    except Exception:
        return False


def _established_execution_identity(
    root: Path, method: str, parameters: Mapping[str, Any]
) -> str:
    adapter_path = root / "src/v6/comparators" / f"{method}.py"
    body = {
        "implementation_version": IMPLEMENTATION,
        "receipt_protocol": RECEIPT_PROTOCOL,
        "method_id": method,
        "topics": 10,
        "parameters": dict(parameters),
        "adapter_sha256": sha256_file(adapter_path),
        "base_contract_sha256": sha256_file(root / "src/v6/comparators/base.py"),
        "predictive_interface": (
            "official_prodlda_native_decoder"
            if method == "combinedtm" else "declared_method_contract"
        ),
    }
    return sha256_object(body)


def _run_established(data: BaselineData, out: Path, protocol: Mapping[str, Any], report) -> None:
    root = Path(__file__).resolve().parents[2]
    for method in ESTABLISHED:
        parameters = dict(protocol["established"][method])
        execution_identity = _established_execution_identity(root, method, parameters)
        for seed in map(int,protocol["seeds"]):
            receipt=out/"receipts"/method/f"seed_{seed}"
            if _valid_receipt(receipt,method,seed,data.input_fingerprint,execution_identity): report(f"{method}/{seed} valid R2 receipt REUSED"); continue
            report(f"{method}/{seed} fit START")
            started=time.perf_counter(); adapter=make_adapter(method,parameters); fit=adapter.fit(data,topics=10,seed=seed)
            _save_fit(receipt,fit,{"schema_version":2,"implementation_version":IMPLEMENTATION,"receipt_protocol":RECEIPT_PROTOCOL,"execution_identity":execution_identity,"method_id":method,"display_name":DISPLAY[method],"family":"established","seed":seed,"input_identity":data.input_fingerprint,"topics":10,"actual_topics":int(fit.actual_topics),"fit_documents":695,"inference_documents":75,"inference_input":"observed_counts_only","target_available_to_adapter":False,"city_or_label_available_to_adapter":False,"labels":0,"winner_selected":False,"runtime_seconds":time.perf_counter()-started,"iterations_completed":fit.iterations_completed,"converged":fit.converged,"likelihood_comparable":bool(fit.likelihood_comparable),"predictive_interface":fit.metadata.get("predictive_interface", "theta_times_topic_word" if fit.likelihood_comparable else "not_comparable"),"resolved_parameters":parameters,"adapter_metadata":dict(fit.metadata)})
            report(f"{method}/{seed} DONE")


def _base_version(value: Any) -> str:
    return str(value).split("+", 1)[0]


def _runtime_candidates(source_root: Path, method: str) -> list[dict[str, Any]]:
    """Prefer signed V4 setup receipts, then inspect existing environments."""

    candidates: list[dict[str, Any]] = []
    stable = source_root / "data/interim/sota_execution"
    if method == "neuromax":
        receipt_paths = sorted(stable.glob("*/neuromax_extension/setup_receipt.json"))
        for receipt_path in receipt_paths:
            try:
                receipt = _json(receipt_path)
                environment = receipt["environment"]
                if receipt.get("status") == "PASS":
                    candidates.append({
                        "python": Path(str(environment["python"])),
                        "provenance": "signed_v4_neuromax_setup_receipt",
                        "setup_receipt": str(receipt_path.resolve()),
                    })
            except (KeyError, OSError, TypeError, ValueError):
                continue
    else:
        receipt_paths = sorted(stable.glob("*/setup_receipt.json"))
        for receipt_path in receipt_paths:
            try:
                receipt = _json(receipt_path)
                environment = receipt["environments"][method]
                if receipt.get("status") != "PASS":
                    continue
                item = {
                    "python": Path(str(environment["python"])),
                    "provenance": "signed_v4_official_sota_setup_receipt",
                    "setup_receipt": str(receipt_path.resolve()),
                }
                if method == "fastopic":
                    item["expected_distribution_sha256"] = receipt.get("sources", {}).get("fastopic", {}).get("source_artifact_sha256")
                candidates.append(item)
            except (KeyError, OSError, TypeError, ValueError):
                continue
    cache = source_root / "cache/sota_envs"
    for pattern in ("*/Scripts/python.exe", "*/bin/python"):
        for python in sorted(cache.glob(pattern)):
            candidates.append({
                "python": python,
                "provenance": "strictly_probed_existing_cached_runtime",
                "setup_receipt": None,
            })
    candidates.append({
        "python": Path(sys.executable),
        "provenance": "strictly_probed_current_runtime",
        "setup_receipt": None,
    })
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item["python"]).replace("\\", "/").casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _probe_runtime(python: Path, method: str, source: Path | None) -> dict[str, Any]:
    method_literal = json.dumps(method)
    source_literal = json.dumps(str(source.resolve())) if source else "None"
    code = f'''import hashlib, importlib.metadata as md, inspect, json, pathlib, sys
method = {method_literal}
source = {source_literal}
if source:
    sys.path.insert(0, source)
import torch, torchvision
payload = {{
    "python": list(sys.version_info[:3]),
    "torch": str(torch.__version__),
    "torchvision": str(torchvision.__version__),
    "torch_cuda": str(torch.version.cuda),
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "packages": {{}},
}}
def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None
package_names = {{
    "fastopic": ("numpy", "scipy", "pandas", "plotly", "sentence-transformers", "gensim", "scikit-learn", "tqdm", "topmost", "fastopic"),
    "glocom": ("numpy", "scipy", "sentence-transformers", "gensim", "scikit-learn", "tqdm", "wandb", "topmost", "geomloss"),
    "encot": ("numpy", "scipy", "sentence-transformers", "gensim", "scikit-learn", "tqdm", "wandb", "topmost", "geomloss"),
    "neuromax": ("numpy", "scipy", "sentence-transformers", "gensim", "scikit-learn", "tqdm", "wandb", "topmost", "geomloss", "torch-kmeans"),
}}[method]
payload["packages"] = {{name: version(name) for name in package_names}}
if method == "fastopic":
    from fastopic import FASTopic
    payload["api"] = {{
        "init": str(inspect.signature(FASTopic.__init__)),
        "fit": str(inspect.signature(FASTopic.fit)),
        "transform": str(inspect.signature(FASTopic.transform)),
    }}
    dist = md.distribution("fastopic")
    files = []
    for item in dist.files or []:
        path = pathlib.Path(dist.locate_file(item))
        if path.is_file():
            files.append((str(item), hashlib.sha256(path.read_bytes()).hexdigest()))
    distribution = {{"version": dist.version, "files": sorted(files)}}
    encoded = json.dumps(distribution, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload["distribution_sha256"] = hashlib.sha256(encoded).hexdigest()
elif method == "glocom":
    from models.GloCOM.GloCOM import GloCOM
    from trainer.Trainer import Trainer
    payload["api"] = {{"model": str(inspect.signature(GloCOM.__init__)), "trainer": str(inspect.signature(Trainer.__init__)), "has_get_beta": hasattr(Trainer, "get_beta"), "has_test": hasattr(Trainer, "test")}}
    payload["model_source"] = str(pathlib.Path(inspect.getsourcefile(GloCOM)).resolve())
elif method == "encot":
    from topmost.models.NewMethod.NewMethod import NewMethod
    from topmost.trainers.basic_trainer import BasicTrainer
    payload["api"] = {{"model": str(inspect.signature(NewMethod.__init__)), "trainer": str(inspect.signature(BasicTrainer.__init__)), "has_get_beta": hasattr(BasicTrainer, "get_beta"), "has_test": hasattr(BasicTrainer, "test")}}
    payload["model_source"] = str(pathlib.Path(inspect.getsourcefile(NewMethod)).resolve())
elif method == "neuromax":
    from NeuroMax.NeuroMax import NeuroMax
    from basic_trainer import BasicTrainer
    payload["api"] = {{"model": str(inspect.signature(NeuroMax.__init__)), "trainer": str(inspect.signature(BasicTrainer.__init__)), "has_train": hasattr(BasicTrainer, "train"), "has_test": hasattr(BasicTrainer, "test"), "has_export_beta": hasattr(BasicTrainer, "export_beta")}}
    payload["model_source"] = str(pathlib.Path(inspect.getsourcefile(NeuroMax)).resolve())
else:
    raise ValueError(method)
print(json.dumps(payload, sort_keys=True))'''
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def _runtime_compatibility_errors(
    method: str, probe: Mapping[str, Any], source: Path | None = None
) -> list[str]:
    errors: list[str] = []
    modern_packages = {
        "numpy": "1.26.4", "scipy": "1.12.0", "pandas": "2.2.3",
        "plotly": "6.0.1", "sentence-transformers": "3.4.1",
        "gensim": "4.3.3", "scikit-learn": "1.6.1", "tqdm": "4.66.5",
        "topmost": "1.0.2", "fastopic": "1.0.1",
    }
    legacy_packages = {
        "numpy": "1.26.3", "scipy": "1.10.1",
        "sentence-transformers": "2.7.0", "gensim": "4.3.3",
        "scikit-learn": "1.5.1", "tqdm": "4.66.5", "wandb": "0.18.1",
        "topmost": "0.0.5", "geomloss": "0.2.6",
    }
    expected_packages = {
        "fastopic": modern_packages,
        "glocom": legacy_packages,
        "encot": legacy_packages,
        "neuromax": {**legacy_packages, "torch-kmeans": "0.2.0"},
    }
    modern = method == "fastopic"
    if list(probe.get("python", []))[:2] != [3, 10]:
        errors.append(f"python={probe.get('python')} expected=3.10")
    if _base_version(probe.get("torch")) != ("2.7.0" if modern else "2.4.1"):
        errors.append(f"torch={probe.get('torch')}")
    if _base_version(probe.get("torchvision")) != ("0.22.0" if modern else "0.19.1"):
        errors.append(f"torchvision={probe.get('torchvision')}")
    if probe.get("cuda_available") is not True:
        errors.append("CUDA unavailable")
    observed_packages = probe.get("packages", {})
    for package, expected in expected_packages[method].items():
        if str(observed_packages.get(package)) != expected:
            errors.append(f"{package}={observed_packages.get(package)} expected={expected}")
    api = probe.get("api", {})
    required = {
        "fastopic": {
            "init": ("num_topics", "preprocess", "DT_alpha", "TW_alpha", "theta_temp"),
            "fit": ("preset_doc_embeddings",),
            "transform": ("doc_embeddings",),
        },
        "glocom": {
            "model": ("num_topics", "pretrained_WE", "aug_coef", "prior_var", "weight_loss_ECR"),
            "trainer": ("dataset", "epochs", "learning_rate", "batch_size"),
        },
        "encot": {
            "model": ("num_topics", "num_clusters", "embed_size", "weight_loss_ECR", "weight_ot_doc_cluster", "weight_ot_topic_cluster"),
            "trainer": ("dataset", "epochs", "learning_rate", "batch_size"),
        },
        "neuromax": {
            "model": ("num_topics", "num_groups", "pretrained_WE", "weight_loss_ECR", "weight_loss_GR", "weight_loss_InfoNCE"),
            "trainer": ("epochs", "learning_rate", "batch_size", "lr_scheduler"),
        },
    }
    for section, names in required[method].items():
        signature = str(api.get(section, ""))
        for name in names:
            if name not in signature:
                errors.append(f"API {section} lacks {name}")
    required_flags = {
        "glocom": ("has_get_beta", "has_test"),
        "encot": ("has_get_beta", "has_test"),
        "neuromax": ("has_train", "has_test", "has_export_beta"),
    }
    for flag in required_flags.get(method, ()):
        if api.get(flag) is not True:
            errors.append(f"API flag {flag}=False")
    if source is not None:
        try:
            Path(str(probe["model_source"])).resolve().relative_to(source.resolve())
        except (KeyError, ValueError):
            errors.append("official model import did not resolve inside the locked source tree")
    return errors


def _select_runtime(
    source_root: Path, method: str, source: Path | None
) -> dict[str, Any]:
    inspected: list[dict[str, Any]] = []
    for candidate in _runtime_candidates(source_root, method):
        python = Path(candidate["python"])
        diagnostic = {
            "python": str(python),
            "provenance": candidate["provenance"],
            "setup_receipt": candidate.get("setup_receipt"),
        }
        if not python.is_file():
            diagnostic["errors"] = ["interpreter missing"]
            inspected.append(diagnostic)
            continue
        try:
            probe = _probe_runtime(python, method, source)
            errors = _runtime_compatibility_errors(method, probe, source)
            expected_distribution = candidate.get("expected_distribution_sha256")
            if expected_distribution and probe.get("distribution_sha256") != expected_distribution:
                errors.append("FASTopic installed-distribution fingerprint changed")
            diagnostic["probe"] = probe
            diagnostic["errors"] = errors
            inspected.append(diagnostic)
            if not errors:
                selected = dict(diagnostic)
                selected["status"] = "PASS"
                selected["environment_identity"] = sha256_object({
                    "method": method,
                    "python": str(python.resolve()),
                    "probe": probe,
                })
                return selected
        except Exception as exc:
            diagnostic["errors"] = [f"{type(exc).__name__}: {exc}"]
            inspected.append(diagnostic)
    summary = [
        {"python": row["python"], "errors": row.get("errors", [])}
        for row in inspected
    ]
    raise RuntimeError(
        f"No exact existing {method} runtime passed the locked version/API/CUDA probe. "
        f"Step 9 installed nothing. Inspected={summary}"
    )


def _git_head(path: Path) -> str:
    return subprocess.run(["git","rev-parse","HEAD"],cwd=path,check=True,text=True,capture_output=True).stdout.strip()


def _git_tree_sha256(path: Path) -> str:
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--full-tree", "HEAD"],
        cwd=path, check=True, text=True, capture_output=True,
    ).stdout.strip()
    return sha256_object({"git_ls_tree": tree})


def _canonical_git_url(value: str) -> str:
    return str(value).strip().casefold().removesuffix(".git").rstrip("/")


def _locked_neuromax_source(source_root: Path) -> dict[str, Any]:
    locks = sorted(
        (source_root / "data/interim/sota_execution").glob(
            "*/neuromax_extension/neuromax_source_lock.json"
        )
    )
    valid: list[dict[str, Any]] = []
    for path in locks:
        try:
            value = _json(path)
            if value.get("method_id") == "neuromax":
                value = dict(value)
                value["lock_path"] = str(path.resolve())
                valid.append(value)
        except (OSError, TypeError, ValueError):
            continue
    identities = {
        (row.get("source_revision"), row.get("source_artifact_sha256"))
        for row in valid
    }
    if (
        not valid
        or len(identities) != 1
        or not isinstance(valid[-1].get("source_revision"), str)
        or len(valid[-1]["source_revision"]) != 40
        or not isinstance(valid[-1].get("source_artifact_sha256"), str)
        or len(valid[-1]["source_artifact_sha256"]) != 64
    ):
        raise RuntimeError(
            "A unique pre-metric V4 NeuroMax source lock was not found; "
            "Step 9 refuses to adopt the current repository HEAD post hoc"
        )
    return valid[-1]


def _preflight_sources(source_root: Path) -> dict[str, dict[str, Any]]:
    urls = {
        "fastopic": "https://github.com/bobxwu/FASTopic",
        "glocom": "https://github.com/qducnguyen/GloCOM",
        "encot": "https://github.com/manhdo249/EnCOT",
        "neuromax": "https://github.com/Fsoft-AIC/NeuroMax",
    }
    sources: dict[str, Path | None] = {
        "fastopic": None,
        "glocom": source_root / "third_party/sota/GloCOM",
        "encot": source_root / "third_party/sota/EnCOT",
        "neuromax": source_root / "third_party/sota/NeuroMax",
    }
    expected_revision = {
        "glocom": "4094055b9e2d0169b0aa75d5aed7220e9509f0de",
        "encot": "8ac3592165cc7851676be2776b314eb6e44e9388",
    }
    neuro_lock = _locked_neuromax_source(source_root)
    if _canonical_git_url(neuro_lock.get("official_source_url", "")) != _canonical_git_url(urls["neuromax"]):
        raise ValueError("The frozen NeuroMax source lock has an unexpected origin")
    expected_revision["neuromax"] = str(neuro_lock["source_revision"])
    result: dict[str, dict[str, Any]] = {}
    for method in SOTA:
        source = sources[method]
        if source is None:
            result[method] = {
                "method_id": method,
                "official_source_url": urls[method],
                "source_dir": None,
                "source_kind": "official_pypi_release",
                "source_revision": "1.0.1",
            }
            continue
        if not (source / ".git").is_dir():
            raise FileNotFoundError(f"Missing official git checkout: {source}")
        origin = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=source,
            check=True, text=True, capture_output=True,
        ).stdout.strip()
        if _canonical_git_url(origin) != _canonical_git_url(urls[method]):
            raise ValueError(f"{method} official source origin changed: {origin}")
        head = _git_head(source)
        if head != expected_revision[method]:
            raise ValueError(
                f"{method} official source revision changed: {head} != {expected_revision[method]}"
            )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source, check=True, text=True, capture_output=True,
        ).stdout.strip()
        if dirty:
            raise ValueError(f"{method} official source has tracked modifications")
        artifact = _git_tree_sha256(source)
        if method == "neuromax" and artifact != neuro_lock.get("source_artifact_sha256"):
            raise ValueError("NeuroMax source tree no longer matches its pre-metric lock")
        result[method] = {
            "method_id": method,
            "official_source_url": urls[method],
            "source_dir": str(source.resolve()),
            "source_kind": "official_git_repository",
            "source_revision": head,
            "source_artifact_sha256": artifact,
            "tracked_source_tree_clean": True,
            "source_lock_path": neuro_lock.get("lock_path") if method == "neuromax" else None,
        }
    return result


def _preflight_sota(
    source_root: Path, out: Path
) -> dict[str, dict[str, Any]]:
    sources = _preflight_sources(source_root)
    result: dict[str, dict[str, Any]] = {}
    for method in SOTA:
        source_text = sources[method].get("source_dir")
        runtime = _select_runtime(
            source_root, method, Path(source_text) if source_text else None
        )
        source_receipt = dict(sources[method])
        if method == "fastopic":
            source_receipt["source_artifact_sha256"] = runtime["probe"]["distribution_sha256"]
        result[method] = {"source": source_receipt, "runtime": runtime}
    body = {
        "schema_version": 2,
        "implementation_version": IMPLEMENTATION,
        "status": "PASS",
        "selection_rule": "exact_locked_versions_api_source_and_cuda_not_first_import",
        "torch_or_cuda_packages_installed": 0,
        "methods": result,
    }
    body["preflight_identity"] = sha256_object(body)
    _write_json(out/"runtime_source_preflight.json", body)
    return result


def _sota_execution_identity(
    root: Path,
    method: str,
    parameters: Mapping[str, Any],
    input_identity: str,
    method_preflight: Mapping[str, Any],
) -> str:
    worker = root / "src/v6/step9_official_worker.py"
    neuro = root / "src/v6/step9_neuromax_worker.py"
    source_receipt = method_preflight["source"]
    runtime_receipt = method_preflight["runtime"]
    return sha256_object({
        "implementation_version": IMPLEMENTATION,
        "receipt_protocol": RECEIPT_PROTOCOL,
        "method_id": method,
        "topics": 10,
        "parameters": dict(parameters),
        "worker_sha256": sha256_file(neuro if method == "neuromax" else worker),
        "environment_identity": runtime_receipt["environment_identity"],
        "source_revision": source_receipt["source_revision"],
        "source_artifact_sha256": source_receipt["source_artifact_sha256"],
        "adapter_input_identity": input_identity,
    })


def _run_sota(
    root: Path,
    data: BaselineData,
    out: Path,
    protocol: Mapping[str, Any],
    preflight: Mapping[str, Mapping[str, Any]],
    report,
) -> None:
    worker=root/"src/v6/step9_official_worker.py"; neuro=root/"src/v6/step9_neuromax_worker.py"
    for method in SOTA:
        method_preflight = preflight[method]
        source_receipt = method_preflight["source"]
        runtime_receipt = method_preflight["runtime"]
        source_text = source_receipt.get("source_dir")
        source = Path(source_text) if source_text else None
        python = Path(str(runtime_receipt["python"]))
        execution_identity = _sota_execution_identity(
            root, method, protocol["recent_sota"][method],
            data.input_fingerprint, method_preflight,
        )
        for seed in map(int,protocol["seeds"]):
            receipt=out/"receipts"/method/f"seed_{seed}"
            if _valid_receipt(receipt,method,seed,data.input_fingerprint,execution_identity): report(f"{method}/{seed} valid R2 receipt REUSED"); continue
            adapter_receipt = _json(out/"input_kit/adapter_input_receipt.json")
            job={"schema_version":2,"implementation_version":IMPLEMENTATION,"receipt_protocol":RECEIPT_PROTOCOL,"execution_identity":execution_identity,"method_id":method,"seed":seed,"topics":10,"input_dir":str((out/"input_kit").resolve()),"receipt_dir":str(receipt.resolve()),"source_dir":str(source.resolve()) if source else None,"official_source_url":source_receipt["official_source_url"],"source_kind":source_receipt["source_kind"],"source_revision":source_receipt["source_revision"],"source_artifact_sha256":source_receipt["source_artifact_sha256"],"model_configuration_id":data.input_fingerprint,"input_fingerprint":data.input_fingerprint,"input_bundle_sha256":adapter_receipt["adapter_bundle_sha256"],"execution_policy_id":"v6_1_step9_frozen","execution_policy_sha256":sha256_object(protocol["policy"]),"execution_lock_sha256":sha256_object(protocol["recent_sota"]),"allowed_overrides":["topic_count","seed","common_numeric_inputs"],"hyperparameters":dict(protocol["recent_sota"][method]),"compatibility_adapter":{"required_model_dimension":384,"fit_partition":"train_only","test_transform":"frozen_training_PCA"},"adapter_path":str((neuro if method=="neuromax" else worker).resolve())}
            jobs=out/"jobs"; jobs.mkdir(parents=True,exist_ok=True); job_path=jobs/f"{method}_seed_{seed}.json"; _write_json(job_path,job)
            report(f"{method}/{seed} official fit START")
            subprocess.run([str(python),str(neuro if method=="neuromax" else worker),"--job",str(job_path)],cwd=root,check=True)
            # Normalize legacy worker metadata into the strict Step-9 receipt envelope.
            legacy=_json(receipt/"metadata.json")
            legacy.update({"schema_version":2,"implementation_version":IMPLEMENTATION,"receipt_protocol":RECEIPT_PROTOCOL,"execution_identity":execution_identity,"input_identity":data.input_fingerprint,"family":"recent_sota","inference_input":"observed_counts_only","target_available_to_adapter":False,"city_or_label_available_to_adapter":False,"winner_selected":False,"labels":0,"likelihood_comparable":True})
            legacy.pop("receipt_identity",None); legacy["canonical_output_sha256"]=sha256_file(receipt/"canonical_output.npz"); legacy["receipt_identity"]=sha256_object(legacy)
            _write_json(receipt/"metadata.json",legacy)
            if not _valid_receipt(receipt,method,seed,data.input_fingerprint,execution_identity): raise RuntimeError(f"Invalid Step-9 R2 receipt: {method}/{seed}")
            report(f"{method}/{seed} DONE")


def _predictive_metrics(
    beta: np.ndarray,
    theta: np.ndarray | None,
    probabilities: np.ndarray | None,
    target: EvaluationTarget,
    *,
    likelihood_comparable: bool,
    probability_floor: float,
) -> tuple[float | None, float | None]:
    if not likelihood_comparable:
        return None, None
    if probabilities is not None:
        micro = document_completion_nll_from_probabilities(
            target.counts, probabilities, probability_floor=probability_floor
        )
        city_values = []
        for city in range(3):
            selected = np.flatnonzero(target.city_indices == city)
            city_values.append(
                document_completion_nll_from_probabilities(
                    target.counts[selected], probabilities[selected],
                    probability_floor=probability_floor,
                ).nll_per_token
            )
        return float(np.mean(city_values)), micro.nll_per_token
    if theta is None:
        raise ValueError("A comparable likelihood requires theta or native probabilities")
    normalized_theta = normalize_rows(theta)
    normalized_beta = normalize_rows(beta)
    micro = document_completion_nll(
        target.counts, normalized_theta, normalized_beta,
        probability_floor=probability_floor,
    )
    city_values = []
    for city in range(3):
        selected = np.flatnonzero(target.city_indices == city)
        city_values.append(
            document_completion_nll(
                target.counts[selected], normalized_theta[selected], normalized_beta,
                probability_floor=probability_floor,
            ).nll_per_token
        )
    return float(np.mean(city_values)), micro.nll_per_token


def _metrics(
    beta: np.ndarray,
    theta: np.ndarray | None,
    probabilities: np.ndarray | None,
    data: BaselineData,
    target: EvaluationTarget,
    protocol: Mapping[str, Any],
    *,
    likelihood_comparable: bool,
) -> dict[str,float|None]:
    beta=normalize_rows(beta); top=stable_top_word_indices(beta,int(protocol["evaluation"]["top_words"]))
    npmi=document_npmi(data.train_counts,top).mean
    cv=cv_coherence(data.train_token_documents,top,vocabulary_size=data.vocabulary_size,window_size=int(protocol["evaluation"]["c_v_window_size"]),gamma=float(protocol["evaluation"]["c_v_gamma"])).mean
    result={"npmi_at_10":npmi,"c_v_at_10":cv,"topic_diversity_at_10":topic_diversity(top),"top_word_redundancy_at_10":mean_top_word_jaccard(top),"macro_city_nll_per_token":None,"micro_nll_per_token":None}
    macro, micro = _predictive_metrics(
        beta, theta, probabilities, target,
        likelihood_comparable=likelihood_comparable,
        probability_floor=float(protocol["evaluation"]["probability_floor"]),
    )
    result["macro_city_nll_per_token"] = macro
    result["micro_nll_per_token"] = micro
    return result


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float,float]:
    rng=np.random.default_rng(seed); means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return tuple(map(float,np.quantile(means,[.025,.975])))


def _holm(p: list[float]) -> list[float]:
    order=np.argsort(p); out=np.empty(len(p)); running=0.0; m=len(p)
    for rank,i in enumerate(order): running=max(running,(m-rank)*p[i]); out[i]=min(1.0,running)
    return out.tolist()


def _evaluate(
    root: Path,
    step8: Path,
    data: BaselineData,
    target: EvaluationTarget,
    out: Path,
    protocol: Mapping[str,Any],
    preflight: Mapping[str, Mapping[str, Any]],
) -> None:
    rows=[]
    proposed=pd.read_csv(step8/"worker/per_seed_variant_metrics.csv",encoding="utf-8-sig")
    for r in proposed[proposed.variant=="full_graph_calibration_pooled"].to_dict("records"):
        rows.append({"method_id":"v6_1","display_name":DISPLAY["v6_1"],"family":"proposed","seed":int(r["seed"]),"likelihood_comparable":True,"predictive_interface":"frozen_step8_native_encot_decoder_plus_city_residual_calibration_backoff",**{k:r[k] for k in ("npmi_at_10","c_v_at_10","topic_diversity_at_10","top_word_redundancy_at_10","macro_city_nll_per_token","micro_nll_per_token")}})
    for method in (*ESTABLISHED,*SOTA):
        execution_identity = (
            _established_execution_identity(root, method, protocol["established"][method])
            if method in ESTABLISHED else
            _sota_execution_identity(
                root, method, protocol["recent_sota"][method],
                data.input_fingerprint, preflight[method],
            )
        )
        for seed in map(int,protocol["seeds"]):
            receipt=out/"receipts"/method/f"seed_{seed}"
            if not _valid_receipt(receipt,method,seed,data.input_fingerprint,execution_identity): raise RuntimeError(f"Missing valid R2 receipt {method}/{seed}")
            metadata = _json(receipt/"metadata.json")
            with np.load(receipt/"canonical_output.npz",allow_pickle=False) as z:
                beta=np.asarray(z["topic_word"],float)
                theta=np.asarray(z["test_theta"],float) if "test_theta" in z.files else None
                probabilities=np.asarray(z["test_word_probability"],float) if "test_word_probability" in z.files else None
            rows.append({"method_id":method,"display_name":DISPLAY[method],"family":"established" if method in ESTABLISHED else "recent_sota","seed":seed,"likelihood_comparable":bool(metadata.get("likelihood_comparable",False)),"predictive_interface":metadata.get("predictive_interface","not_comparable"),**_metrics(beta,theta,probabilities,data,target,protocol,likelihood_comparable=bool(metadata.get("likelihood_comparable",False)))})
    _write_csv(out/"all_method_seed_metrics.csv",rows)
    interface_rows = []
    for method, group in pd.DataFrame(rows).groupby("method_id", sort=False):
        interfaces = sorted(set(map(str, group["predictive_interface"])))
        comparable = sorted(set(map(bool, group["likelihood_comparable"])))
        if len(interfaces) != 1 or len(comparable) != 1:
            raise ValueError(f"Predictive interface changed across seeds for {method}")
        interface_rows.append({
            "method_id": method,
            "display_name": DISPLAY[method],
            "likelihood_comparable": comparable[0],
            "predictive_interface": interfaces[0],
            "nll_table_entry": "mean_sd" if comparable[0] else "dash_not_comparable",
        })
    _write_csv(out/"predictive_interface_audit.csv", interface_rows)
    metrics=("npmi_at_10","c_v_at_10","topic_diversity_at_10","top_word_redundancy_at_10","macro_city_nll_per_token","micro_nll_per_token")
    agg=[]
    for method,g in pd.DataFrame(rows).groupby("method_id",sort=False):
        for metric in metrics:
            x=pd.to_numeric(g[metric],errors="coerce").dropna().to_numpy(float)
            if not len(x): agg.append({"method_id":method,"display_name":DISPLAY[method],"metric":metric,"n":0,"mean":None,"sample_sd":None,"ci95_low":None,"ci95_high":None}); continue
            lo,hi=_bootstrap(x,20260909+int(hashlib.sha256(f"{method}:{metric}".encode()).hexdigest()[:8],16))
            agg.append({"method_id":method,"display_name":DISPLAY[method],"metric":metric,"n":len(x),"mean":float(x.mean()),"sample_sd":float(x.std(ddof=1)),"ci95_low":lo,"ci95_high":hi})
    _write_csv(out/"all_method_aggregate_metrics.csv",agg)
    frame=pd.DataFrame(rows); paired=[]
    for metric in metrics:
        base=frame[frame.method_id=="v6_1"].set_index("seed")[metric]
        tests=[]
        for method in (*ESTABLISHED,*SOTA):
            comp=frame[frame.method_id==method].set_index("seed")[metric]
            common=base.dropna().index.intersection(comp.dropna().index)
            if len(common)!=10: continue
            effect=(base.loc[common]-comp.loc[common]).to_numpy(float)
            if metric in ("top_word_redundancy_at_10","macro_city_nll_per_token","micro_nll_per_token"): effect=-effect
            p=float(stats.wilcoxon(effect,alternative="two-sided",method="exact").pvalue) if np.any(effect) else 1.0
            tests.append({"metric":metric,"comparator":method,"n":10,"mean_effect_positive_favors_v6_1":float(effect.mean()),"wins":int((effect>0).sum()),"ties":int((effect==0).sum()),"losses":int((effect<0).sum()),"raw_p":p})
        adjusted=_holm([x["raw_p"] for x in tests]) if tests else []
        for x,p in zip(tests,adjusted): x["holm_p"]=p; paired.append(x)
    _write_csv(out/"paired_comparator_tests.csv",paired)
    wide=pd.DataFrame(agg).pivot(index=["method_id","display_name"],columns="metric",values=["mean","sample_sd"]).reset_index()
    wide.columns=["_".join(x).rstrip("_") if isinstance(x,tuple) else x for x in wide.columns]
    wide.to_csv(out/"paper_ready_comparison_table.csv",index=False,encoding="utf-8-sig")


def _manifest(out: Path) -> dict[str,Any]:
    files=[]
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name!="generated_artifact_manifest.json"):
        files.append({"relative_path":path.relative_to(out).as_posix(),"sha256":sha256_file(path),"size_bytes":path.stat().st_size})
    body={"schema_version":1,"files":files}; body["manifest_identity"]=sha256_object(body); _write_json(out/"generated_artifact_manifest.json",body); return body


def _capture_r1_lineage(out: Path) -> dict[str, Any]:
    """Archive pre-R2 receipts without opening or selecting their numeric values."""

    lineage_path = out / "r1_receipt_lineage.json"
    archive_path = out / "r1_receipts_audit.zip"
    if lineage_path.is_file():
        lineage = _json(lineage_path)
        if not _verify_fingerprint(lineage, "lineage_identity"):
            raise ValueError("The preserved R1 receipt-lineage identity is invalid")
        expected_archive = lineage.get("audit_archive_sha256")
        if expected_archive and (
            not archive_path.is_file() or sha256_file(archive_path) != expected_archive
        ):
            raise ValueError("The preserved R1 receipt archive changed")
        return lineage
    files: list[dict[str, Any]] = []
    roots: set[Path] = set()
    for metadata_path in sorted((out / "receipts").glob("*/seed_*/metadata.json")):
        try:
            metadata = _json(metadata_path)
        except (OSError, TypeError, ValueError):
            continue
        if metadata.get("receipt_protocol") == RECEIPT_PROTOCOL:
            continue
        roots.add(metadata_path.parent)
    for receipt_root in sorted(roots):
        for path in sorted(item for item in receipt_root.rglob("*") if item.is_file()):
            files.append({
                "relative_path": path.relative_to(out).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    audit_archive_sha256 = None
    if files:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in files:
                path = out / item["relative_path"]
                archive.write(path, item["relative_path"])
        audit_archive_sha256 = sha256_file(archive_path)
    body = {
        "schema_version": 1,
        "status": "PRESERVED_BEFORE_R2_REFIT",
        "r1_receipt_directories": len(roots),
        "files": files,
        "audit_archive": archive_path.name if files else None,
        "audit_archive_sha256": audit_archive_sha256,
        "numeric_values_opened_or_used_by_migration": False,
        "r1_receipts_reused_as_r2_evidence": 0,
        "reason": "R2 restricted the adapter interface and corrected predictive decoding",
    }
    body["lineage_identity"] = sha256_object(body)
    _write_json(lineage_path, body)
    return body


def _valid_receipt_count(
    root: Path,
    data: BaselineData,
    out: Path,
    protocol: Mapping[str, Any],
    preflight: Mapping[str, Mapping[str, Any]],
) -> int:
    count = 0
    for method in ESTABLISHED:
        execution_identity = _established_execution_identity(
            root, method, protocol["established"][method]
        )
        for seed in map(int, protocol["seeds"]):
            count += int(_valid_receipt(
                out/"receipts"/method/f"seed_{seed}", method, seed,
                data.input_fingerprint, execution_identity,
            ))
    for method in SOTA:
        if method not in preflight:
            continue
        execution_identity = _sota_execution_identity(
            root, method, protocol["recent_sota"][method],
            data.input_fingerprint, preflight[method],
        )
        for seed in map(int, protocol["seeds"]):
            count += int(_valid_receipt(
                out/"receipts"/method/f"seed_{seed}", method, seed,
                data.input_fingerprint, execution_identity,
            ))
    return count


def run_v6_step9(*,project_root:Path,config_path:Path,source_root:Path|None=None)->Path:
    root=project_root.resolve(); protocol=_load_protocol(config_path); step8,contract=_latest_step8(root); _verify_manifest(step8)
    correction_audit_sha256 = sha256_file(
        root/"docs/V6_STEP09_R1_FAILURE_AUDIT_AND_R2_CORRECTION.md"
    )
    pre=protocol["prerequisite"]
    for field,key in (("step8_job_identity","step8_job_identity"),("test_completion_identity","step8_completion_identity"),("step7_freeze_identity","step7_freeze_identity")):
        if contract.get(field)!=pre[key]: raise ValueError(f"Step-8 prerequisite mismatch: {field}")
    source=(source_root or Path(protocol["source_project_root"])).resolve()
    # A stable run directory is deliberate: expensive method/seed receipts
    # survive an interrupted invocation and are reused only after full identity
    # validation.  This is not winner/seed selection.
    out=root/"outputs/v6/step09_fair_comparators"/"v6_1_frozen_comparison"; out.mkdir(parents=True,exist_ok=True)
    for name in (
        "all_method_seed_metrics.csv", "all_method_aggregate_metrics.csv",
        "paired_comparator_tests.csv", "paper_ready_comparison_table.csv",
        "predictive_interface_audit.csv",
        "step09_contract.json", "hard_checks.csv", "failure.json",
        "generated_artifact_manifest.json", "runtime_source_preflight.json",
        "evaluation_input_receipt.json",
    ):
        candidate = out / name
        if candidate.is_file():
            candidate.unlink()
    log=out/"step09.log"
    def report(msg):
        line=f"[V6.1 Step 9] {msg}"; print(line,flush=True)
        with log.open("a",encoding="utf8") as f:f.write(f"{datetime.now(timezone.utc).isoformat()} | {line}\n")
    report(f"START | implementation={IMPLEMENTATION} | frozen model changes=0 old V4 scores=0")
    data,target,input_receipt=_prepare_inputs(root,out,step8,protocol)
    r1_lineage = _capture_r1_lineage(out)
    report(
        "R1 lineage | archived receipts="
        f"{r1_lineage['r1_receipt_directories']} | reused as R2=0"
    )
    preflight: dict[str, dict[str, Any]] = {}
    try:
        report("preflight START | exact runtimes + official sources + adapter firewall")
        preflight = _preflight_sota(source, out)
        report("preflight PASS | FASTopic=1.0.1/API exact | legacy neural runtime exact | target/city hidden")
        _run_established(data,out,protocol,report)
        _run_sota(root,data,out,protocol,preflight,report)
        _evaluate(root,step8,data,target,out,protocol,preflight)
        status="PASS"; readiness=READINESS
    except Exception as exc:
        status="INCOMPLETE"; readiness="RESUME_STEP9_WITHOUT_DISCARDING_VALID_RECEIPTS"
        _write_json(out/"failure.json",{"exception":type(exc).__name__,"message":str(exc),"model_changes":0,"completed_receipts_preserved":True})
        report(f"INCOMPLETE | {type(exc).__name__}: {exc}")
    receipt_count=_valid_receipt_count(root,data,out,protocol,preflight)
    adapter_firewall_pass = not any(
        (out/"input_kit"/name).exists() for name in FORBIDDEN_ADAPTER_FILES
    )
    checks=[{"check":"step8_frozen_and_verified","status":"PASS"},{"check":"old_v4_numeric_results_used","observed":0,"status":"PASS"},{"check":"r1_numeric_receipts_reused_as_r2","observed":0,"status":"PASS"},{"check":"r1_receipts_preserved_before_refit","observed":r1_lineage["r1_receipt_directories"],"status":"PASS"},{"check":"target_available_to_fit_inference_context","observed":False,"status":"PASS" if adapter_firewall_pass else "FAIL"},{"check":"city_or_label_available_to_comparator","observed":False,"status":"PASS" if adapter_firewall_pass else "FAIL"},{"check":"native_decoder_probability_required","observed":list(NATIVE_PROBABILITY_METHODS),"status":"PASS"},{"check":"nonprobabilistic_pseudo_nll_reported","observed":0,"status":"PASS"},{"check":"exact_runtime_preflight","observed":len(preflight),"required":4,"status":"PASS" if len(preflight)==4 else "PENDING"},{"check":"model_changes","observed":0,"status":"PASS"},{"check":"all_100_comparator_receipts","observed":receipt_count,"required":100,"status":"PASS" if receipt_count==100 else "PENDING"}]
    _write_csv(out/"hard_checks.csv",checks)
    result={"schema_version":2,"implementation_version":IMPLEMENTATION,"receipt_protocol":RECEIPT_PROTOCOL,"status":status,"readiness":readiness,"step8_contract_sha256":sha256_file(step8/"step08_contract.json"),"step8_job_identity":contract["step8_job_identity"],"r1_correction_audit_sha256":correction_audit_sha256,"r1_receipt_lineage_identity":r1_lineage["lineage_identity"],"r1_receipts_preserved_before_refit":r1_lineage["r1_receipt_directories"],"r1_numeric_receipts_reused_as_r2":0,"adapter_input_identity":input_receipt["adapter"]["adapter_input_identity"],"adapter_bundle_sha256":input_receipt["adapter"]["adapter_bundle_sha256"],"evaluation_identity":input_receipt["evaluation"]["evaluation_identity"],"target_or_city_files_exposed_to_comparators":0,"native_probability_methods":list(NATIVE_PROBABILITY_METHODS),"nonprobabilistic_pseudo_nll_reported":0,"comparator_receipts":receipt_count,"comparator_receipts_required":100,"v6_model_changes":0,"legacy_v4_numeric_results_used":0,"labels_used":0}
    result["contract_fingerprint"]=sha256_object(result); _write_json(out/"step09_contract.json",result)
    zip_path=root/"v6_step09_results.zip"
    report(f"{status} | valid R2 receipts={receipt_count}/100 | return-zip={zip_path}")
    _manifest(out)
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(x for x in out.rglob("*") if x.is_file()): z.write(p,p.relative_to(out.parent).as_posix())
    print(f"[V6.1 Step 9] archive COMPLETE | {zip_path}", flush=True)
    if status!="PASS": raise RuntimeError("Step 9 is incomplete; rerun to resume from valid receipts. Return the ZIP and console output if assistance is needed.")
    return zip_path
