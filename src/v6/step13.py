"""Step 13: preregistered external temporal replication on transit tweets.

The stage separates two questions that must not be conflated:

1. Can the exact Chinese-vocabulary frozen state operate on an English corpus?
   This is answered by a vocabulary-coverage diagnostic, not by fabricated
   performance scores.
2. Does the SC-HTM V6.1 architecture generalize when instantiated for the new
   language with every model and training hyperparameter kept fixed?
   This is answered by a chronological, ten-seed external replication.

Raw text and coordinates never enter returned artifacts.  The fit worker is
structurally denied held-out matrices; the comparator workers receive only the
observed document-completion halves; and target halves remain in this common
method-independent evaluator.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.v6.comparators.base import BaselineData, normalize_rows
from src.v6.comparators.registry import make_adapter
from src.v6.data_contract import sha256_file, sha256_object, sparse_logical_hash
from src.v6.document_completion import make_document_completion
from src.v6.evaluation import (
    counts_to_token_documents,
    cv_coherence,
    document_completion_nll,
    document_npmi,
    mean_top_word_jaccard,
    stable_top_word_indices,
    topic_diversity,
)
from src.v6.predictive_evaluation import document_completion_nll_from_probabilities
from src.v6.step4r2 import _verify_manifest
from src.v6.step9 import _preflight_sources, _select_runtime


IMPLEMENTATION = "v6_1_step13_external_temporal_replication_r1"
SCOPE = "preregistered_external_cross_language_architecture_replication"
RECEIPT_PROTOCOL = "v6_1_step13_external_observed_half_firewall_r1"
READINESS = "READY_FOR_STEP10_R3_EXTERNAL_EVIDENCE_ACCEPTANCE"
EXPECTED_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]
ESTABLISHED = ("lda", "nmf", "gsdmm", "btm", "bertopic", "combinedtm")
SOTA = ("fastopic", "glocom", "encot", "neuromax")
NATIVE_PROBABILITY_METHODS = ("combinedtm", *SOTA)
DISPLAY = {
    "v6_1": "SC-HTM V6.1",
    "lda": "LDA",
    "nmf": "NMF",
    "gsdmm": "GSDMM",
    "btm": "BTM",
    "bertopic": "BERTopic",
    "combinedtm": "CombinedTM",
    "fastopic": "FASTopic",
    "glocom": "GloCOM",
    "encot": "EnCOT",
    "neuromax": "NeuroMax",
}
FULL_VARIANT = "full_graph_calibration_pooled"
VARIANT_DISPLAY = {
    FULL_VARIANT: "SC-HTM V6.1",
    "without_graph_pooled_retained": "Without graph transport",
    "without_pooled_graph_calibration_retained": "Without pooled backoff",
    "without_calibration_graph_pooled_retained": "Without decoder calibration",
    "exact_official_parent": "Exact EnCOT parent",
}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class PreparedExternalData:
    fit_counts: sparse.csr_matrix
    observed_counts: sparse.csr_matrix
    target_counts: sparse.csr_matrix
    fit_rows: pd.DataFrame
    completion_rows: pd.DataFrame
    vocabulary: tuple[str, ...]
    vocabulary_frame: pd.DataFrame
    preparation_identity: str
    completion_identity: str
    audit: dict[str, Any]
    paths: dict[str, Path]


@dataclass(frozen=True)
class EvaluationTarget:
    counts: sparse.csr_matrix
    temporal_blocks: np.ndarray
    evaluation_identity: str


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in records for key in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _verify_fingerprint(value: Mapping[str, Any], key: str) -> bool:
    body = dict(value)
    expected = body.pop(key, None)
    return isinstance(expected, str) and sha256_object(body) == expected


def _load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    protocol = document.get("v6", {}).get("step13", {})
    if (
        document.get("schema_version") != 1
        or protocol.get("implementation_version") != IMPLEMENTATION
        or protocol.get("scope") != SCOPE
    ):
        raise ValueError("Step-13 configuration/implementation lock changed")
    if list(map(int, protocol["training"]["seeds"])) != EXPECTED_SEEDS:
        raise ValueError("Step-13 seed set or order changed")
    if int(protocol["model"]["topics"]) != 10:
        raise ValueError("Step-13 topic count changed")
    split = protocol["split"]
    if [
        int(split["original_train_documents"]),
        int(split["spent_validation_documents"]),
        int(split["heldout_test_documents"]),
    ] != [391, 84, 84]:
        raise ValueError("Step-13 chronological split changed")
    if split["validation_used_for_selection"] is not False:
        raise ValueError("external validation selection was enabled")
    observed_stopwords = sha256_object(sorted(ENGLISH_STOP_WORDS))
    if observed_stopwords != protocol["preprocessing"]["stopword_set_sha256"]:
        raise ValueError("the frozen English stopword set changed")
    policy = protocol["policy"]
    for key in (
        "install_or_upgrade_packages",
        "labels_available_to_models",
        "coordinates_available_to_models",
        "test_tokens_define_vocabulary",
        "test_target_available_to_fit_or_inference",
        "seed_selection_allowed",
        "hyperparameter_selection_allowed",
        "drop_failed_seeds_allowed",
        "unfavorable_result_omission_allowed",
        "exact_state_performance_claim_under_language_mismatch_allowed",
    ):
        if policy.get(key) is not False:
            raise ValueError(f"Step-13 scientific firewall was relaxed: {key}")
    return protocol


def _read_dataset(protocol: Mapping[str, Any], root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    specification = protocol["dataset"]
    archive = root / str(specification["archive_path"])
    if not archive.is_file():
        raise FileNotFoundError(
            f"Copy Transit-Tweet-Data-main.zip to {archive}; the external dataset is not bundled."
        )
    if sha256_file(archive) != specification["archive_sha256"]:
        raise ValueError("external dataset archive hash differs from the audited upload")
    with zipfile.ZipFile(archive) as source:
        member = str(specification["csv_member"])
        if member not in source.namelist():
            raise ValueError("audited external CSV member is missing from the archive")
        payload = source.read(member)
    if hashlib.sha256(payload).hexdigest() != specification["csv_sha256"]:
        raise ValueError("external CSV hash differs from the audited upload")
    header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    expected_columns = {"Created_at", "Year", "Month", "Text", "latitude", "longitude"}
    if set(header) != expected_columns:
        raise ValueError("external dataset schema changed")
    frame = pd.read_csv(
        io.BytesIO(payload),
        encoding="utf-8-sig",
        usecols=["Created_at", "Year", "Month", "Text"],
    )
    frame["original_csv_row"] = np.arange(2, len(frame) + 2, dtype=np.int64)
    blank = frame[["Created_at", "Year", "Month", "Text"]].isna().all(axis=1)
    blank_rows = int(blank.sum())
    frame = frame.loc[~blank].copy()
    if len(frame) != int(specification["expected_nonblank_records"]):
        raise ValueError("external nonblank record count changed")
    if frame[["Created_at", "Text"]].isna().any().any():
        raise ValueError("external retained record lacks timestamp or text")
    frame["parsed_utc_timestamp"] = pd.to_datetime(
        frame["Created_at"], utc=True, errors="raise"
    )
    observed_min = frame["parsed_utc_timestamp"].min().isoformat()
    observed_max = frame["parsed_utc_timestamp"].max().isoformat()
    if (
        observed_min != specification["expected_date_min_utc"]
        or observed_max != specification["expected_date_max_utc"]
    ):
        raise ValueError("external timestamp range changed")
    year = pd.to_numeric(frame["Year"], errors="raise").astype(int)
    if not np.array_equal(year.to_numpy(), frame["parsed_utc_timestamp"].dt.year.to_numpy()):
        raise ValueError("external Year column conflicts with Created_at")
    month = frame["Month"].astype(str).str.casefold()
    expected_month = frame["parsed_utc_timestamp"].dt.month_name().str.casefold()
    if not month.equals(expected_month):
        raise ValueError("external Month column conflicts with Created_at")
    duplicate_texts = int(frame["Text"].astype(str).duplicated().sum())
    if duplicate_texts:
        raise ValueError("exact duplicate external texts entered the audited corpus")
    frame = frame.sort_values(
        ["parsed_utc_timestamp", "original_csv_row"], kind="stable"
    ).reset_index(drop=True)
    frame["post_id"] = [f"ETU{index:04d}" for index in range(1, len(frame) + 1)]
    train_n = int(protocol["split"]["original_train_documents"])
    validation_n = int(protocol["split"]["spent_validation_documents"])
    frame["split"] = "test"
    frame.loc[: train_n - 1, "split"] = "train"
    frame.loc[train_n : train_n + validation_n - 1, "split"] = "validation_spent"
    audit = {
        "schema_version": 1,
        "dataset_id": specification["dataset_id"],
        "archive_sha256": specification["archive_sha256"],
        "csv_sha256": specification["csv_sha256"],
        "source_rows_including_blank_tail": int(len(frame) + blank_rows),
        "blank_tail_rows_excluded": blank_rows,
        "retained_records": int(len(frame)),
        "exact_duplicate_texts": duplicate_texts,
        "date_min_utc": observed_min,
        "date_max_utc": observed_max,
        "split_documents": {
            key: int(value) for key, value in frame["split"].value_counts().items()
        },
        "label_columns_available": False,
        "coordinates_present_in_source": True,
        "coordinate_values_loaded": False,
        "coordinate_values_serialized": False,
        "raw_text_in_return_artifacts": False,
        "citation": specification["citation"],
        "license_status": "NO_EXPLICIT_LICENSE_FILE_FOUND_IN_SUPPLIED_REPOSITORY_SNAPSHOT",
        "redistribution_policy": "raw_dataset_not_bundled_or_returned",
    }
    return frame, audit


def _tokenize(text: str, protocol: Mapping[str, Any]) -> list[str]:
    value = unicodedata.normalize("NFKC", html.unescape(str(text))).casefold()
    substitutions = (
        (r"@?\[agency(?: name)?\]", " agency "),
        (r"\[(?:bus|train) number\]", " vehicle "),
        (r"\[line number\]", " route "),
        (r"\[location name\]", " location "),
        (r"https?://\S+|www\.\S+", " url "),
    )
    for pattern, replacement in substitutions:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    stopwords = set(ENGLISH_STOP_WORDS) | set(
        map(str, protocol["preprocessing"]["social_stopwords"])
    )
    minimum = int(protocol["preprocessing"]["minimum_token_characters"])
    tokens = []
    for match in TOKEN_PATTERN.findall(value):
        token = match.replace("'", "")
        if len(token) >= minimum and token not in stopwords:
            tokens.append(token)
    return tokens


def _count_matrix(
    documents: Sequence[Sequence[str]], vocabulary: Sequence[str]
) -> sparse.csr_matrix:
    index = {token: position for position, token in enumerate(vocabulary)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    for row, tokens in enumerate(documents):
        counts = Counter(index[token] for token in tokens if token in index)
        for column in sorted(counts):
            rows.append(row)
            columns.append(column)
            values.append(int(counts[column]))
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(documents), len(vocabulary)),
        dtype=np.int64,
    )


def _prepare_external(
    root: Path,
    out: Path,
    protocol: Mapping[str, Any],
) -> PreparedExternalData:
    frame, audit = _read_dataset(protocol, root)
    tokenized = [_tokenize(text, protocol) for text in frame["Text"].astype(str)]
    fit_n = int(protocol["split"]["final_fit_documents"])
    fit_tokens = tokenized[:fit_n]
    document_frequency: Counter[str] = Counter()
    term_frequency: Counter[str] = Counter()
    for tokens in fit_tokens:
        document_frequency.update(set(tokens))
        term_frequency.update(tokens)
    minimum_df = int(protocol["preprocessing"]["minimum_fit_document_frequency"])
    eligible = [token for token, frequency in document_frequency.items() if frequency >= minimum_df]
    eligible.sort(
        key=lambda token: (
            -document_frequency[token],
            -term_frequency[token],
            token,
        )
    )
    vocabulary = tuple(
        eligible[: int(protocol["preprocessing"]["maximum_vocabulary"])]
    )
    if len(vocabulary) < int(protocol["model"]["topics"]) * int(
        protocol["graph_projection"]["minimum_cluster_size"]
    ):
        raise ValueError("external fit vocabulary is too small for the frozen graph design")
    vocabulary_frame = pd.DataFrame(
        {
            "index": np.arange(len(vocabulary), dtype=np.int64),
            "token": vocabulary,
            "fit_document_frequency": [document_frequency[token] for token in vocabulary],
            "fit_term_frequency": [term_frequency[token] for token in vocabulary],
        }
    )
    fit_counts = _count_matrix(fit_tokens, vocabulary)
    test_tokens = tokenized[fit_n:]
    test_full = _count_matrix(test_tokens, vocabulary)
    if np.any(np.asarray(fit_counts.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("fit-only vocabulary produced an empty fit document")
    if np.any(np.asarray(test_full.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("fit-only vocabulary produced an empty external test document")
    fit_rows = frame.iloc[:fit_n][
        ["post_id", "parsed_utc_timestamp", "split", "original_csv_row"]
    ].copy()
    fit_rows["matrix_row"] = np.arange(fit_n, dtype=np.int64)
    test_rows = frame.iloc[fit_n:][
        ["post_id", "parsed_utc_timestamp", "split", "original_csv_row"]
    ].copy()
    test_rows["matrix_row"] = np.arange(len(test_rows), dtype=np.int64)
    completion = make_document_completion(
        test_full,
        test_rows,
        seed=int(protocol["completion"]["seed"]),
        observed_fraction=float(protocol["completion"]["observed_fraction"]),
        minimum_total_tokens=int(protocol["completion"]["minimum_total_tokens"]),
    )
    completion_rows = completion.retained_rows.copy()
    blocks = np.empty(len(completion_rows), dtype=np.int64)
    for block, indices in enumerate(
        np.array_split(
            np.arange(len(completion_rows), dtype=np.int64),
            int(protocol["completion"]["temporal_blocks"]),
        ),
        start=1,
    ):
        blocks[indices] = block
    completion_rows["temporal_block"] = blocks
    input_dir = out / "external_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fit_counts": input_dir / "fit_counts.npz",
        "test_full": input_dir / "test_full_counts.npz",
        "test_observed": input_dir / "test_completion_observed.npz",
        "test_target": input_dir / "test_completion_target.npz",
        "fit_rows": input_dir / "fit_rows_no_text_no_coordinates.csv",
        "completion_rows": input_dir / "completion_rows_no_text_no_coordinates.csv",
        "vocabulary": input_dir / "vocabulary.csv",
        "completion_exclusions": input_dir / "completion_exclusions.csv",
    }
    sparse.save_npz(paths["fit_counts"], fit_counts)
    sparse.save_npz(paths["test_full"], test_full)
    sparse.save_npz(paths["test_observed"], completion.observed)
    sparse.save_npz(paths["test_target"], completion.target)
    fit_rows.to_csv(paths["fit_rows"], index=False, encoding="utf-8-sig")
    completion_rows.to_csv(
        paths["completion_rows"], index=False, encoding="utf-8-sig"
    )
    vocabulary_frame.to_csv(paths["vocabulary"], index=False, encoding="utf-8-sig")
    completion.excluded_rows.to_csv(
        paths["completion_exclusions"], index=False, encoding="utf-8-sig"
    )
    completion_body = {
        **completion.report,
        "temporal_blocks": int(protocol["completion"]["temporal_blocks"]),
        "temporal_block_sizes": {
            str(block): int(np.sum(blocks == block)) for block in sorted(np.unique(blocks))
        },
        "observed_logical_sha256": sparse_logical_hash(completion.observed),
        "target_logical_sha256": sparse_logical_hash(completion.target),
        "target_used_to_define_vocabulary": False,
    }
    completion_identity = sha256_object(completion_body)
    completion_body["completion_identity"] = completion_identity
    _write_json(input_dir / "completion_contract.json", completion_body)
    audit.update(
        {
            "fit_documents": int(fit_counts.shape[0]),
            "test_documents_before_completion": int(test_full.shape[0]),
            "completion_documents": int(completion.observed.shape[0]),
            "fit_vocabulary": len(vocabulary),
            "fit_tokens": int(fit_counts.sum()),
            "test_in_vocabulary_tokens": int(test_full.sum()),
            "completion_observed_tokens": int(completion.observed.sum()),
            "completion_target_tokens": int(completion.target.sum()),
            "fit_empty_documents": 0,
            "test_empty_documents": 0,
            "vocabulary_fit_partition_only": True,
            "test_tokens_used_to_define_vocabulary": 0,
        }
    )
    preparation_body = {
        "dataset_csv_sha256": protocol["dataset"]["csv_sha256"],
        "preprocessing": protocol["preprocessing"],
        "split": protocol["split"],
        "fit_counts_logical_sha256": sparse_logical_hash(fit_counts),
        "test_full_counts_logical_sha256": sparse_logical_hash(test_full),
        "vocabulary_sha256": sha256_file(paths["vocabulary"]),
        "completion_identity": completion_identity,
    }
    preparation_identity = sha256_object(preparation_body)
    audit["preparation_identity"] = preparation_identity
    _write_json(out / "dataset_audit.json", audit)
    _write_json(out / "preparation_receipt.json", preparation_body | {
        "preparation_identity": preparation_identity
    })
    return PreparedExternalData(
        fit_counts=fit_counts,
        observed_counts=completion.observed,
        target_counts=completion.target,
        fit_rows=fit_rows,
        completion_rows=completion_rows,
        vocabulary=vocabulary,
        vocabulary_frame=vocabulary_frame,
        preparation_identity=preparation_identity,
        completion_identity=completion_identity,
        audit=audit,
        paths=paths,
    )


def _find_step7_runtime(root: Path) -> dict[str, Any]:
    candidates = sorted(
        (root / "outputs/v6/step07_final_refit_freeze").glob("*/step07_contract.json"),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = _json(contract_path)
            output = contract_path.parent
            if not (
                contract.get("status") == "PASS"
                and contract.get("readiness") == "READY_FOR_V6_1_STEP8_ONE_SHOT_TEST"
                and _verify_fingerprint(contract, "contract_fingerprint")
                and _verify_manifest(output)
            ):
                continue
            job = _json(output / "worker_job.json")
            runtime = _json(output / "worker/runtime_receipt.json")
            python = Path(str(runtime["python_executable"]))
            source = Path(str(job["official_source"]))
            if not python.is_file() or not source.is_dir():
                continue
            return {
                "step7_output": output,
                "contract": contract,
                "python": python,
                "official_source": source,
                "official_commit": contract["official_commit"],
                "freeze_identity": contract["freeze_identity"],
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError("No intact passing Step-7 freeze and exact EnCOT runtime were found")


def _find_internal_vocabulary(root: Path) -> Path:
    direct = root / "outputs/v6/step09_fair_comparators/v6_1_frozen_comparison/input_kit/vocabulary.csv"
    if direct.is_file():
        return direct
    for job_path in sorted(
        (root / "outputs/v6/step08_test_confirmation").glob("*/worker_job.json"),
        reverse=True,
    ):
        try:
            path = Path(str(_json(job_path)["development_inputs"]["vocabulary"]["path"]))
            if path.is_file():
                return path
        except (OSError, KeyError, TypeError, ValueError):
            continue
    raise FileNotFoundError("the frozen internal 1,148-token vocabulary was not found")


def _exact_state_coverage(
    root: Path, out: Path, data: PreparedExternalData
) -> dict[str, Any]:
    path = _find_internal_vocabulary(root)
    internal = pd.read_csv(path, encoding="utf-8-sig")["token"].astype(str).tolist()
    external = set(data.vocabulary)
    overlap = sorted(set(internal) & external)
    coverage = len(overlap) / len(external)
    state = {
        "schema_version": 1,
        "assessment": "exact_frozen_state_input_compatibility",
        "internal_vocabulary_tokens": len(internal),
        "external_fit_vocabulary_tokens": len(external),
        "direct_token_overlap": len(overlap),
        "external_vocabulary_coverage_fraction": float(coverage),
        "exact_state_performance_status": "NOT_ESTIMABLE_LANGUAGE_VOCABULARY_MISMATCH",
        "reason": "the frozen Chinese count/decoder coordinates do not define an English observation space",
        "zero_shot_performance_claimed": False,
        "architecture_replication_required": True,
        "internal_vocabulary_sha256": sha256_file(path),
    }
    _write_json(out / "exact_state_coverage_diagnostic.json", state)
    _write_csv(
        out / "exact_state_overlap_tokens.csv",
        [{"token": token} for token in overlap]
        or [{"token": "", "note": "no_direct_token_overlap"}],
    )
    return state


def _stream_worker(runtime: Path, root: Path, job_path: Path, report) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [str(runtime), "-m", "src.v6.step13_worker", "--job", str(job_path)],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    captured = []
    for line in process.stdout:
        value = line.rstrip("\r\n")
        captured.append(value)
        report(value)
    return_code = int(process.wait())
    if return_code:
        raise RuntimeError(
            f"Step-13 exact-runtime worker failed with exit code {return_code}: "
            + "\n".join(captured[-20:])
        )


def _input_item(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_job(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    body = dict(body)
    body["job_identity"] = sha256_object(body)
    _write_json(path, body)
    return body


def _ensure_embeddings(
    root: Path,
    out: Path,
    data: PreparedExternalData,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
    report,
) -> Path:
    worker_output = out / "embedding"
    job_path = out / "jobs/embedding_job.json"
    job = _write_job(
        job_path,
        {
            "schema_version": 1,
            "mode": "embed",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output.resolve()),
            "inputs": {"vocabulary": _input_item(data.paths["vocabulary"])},
            "embedding": dict(protocol["embeddings"]),
            "policy": {
                "labels_available_to_models": False,
                "coordinates_available_to_models": False,
            },
        },
    )
    array = worker_output / "word_embeddings.npz"
    receipt_path = worker_output / "embedding_receipt.json"
    valid = False
    if array.is_file() and receipt_path.is_file():
        try:
            receipt = _json(receipt_path)
            valid = bool(
                receipt.get("job_identity") == job["job_identity"]
                and receipt.get("requested_model_revision")
                == protocol["embeddings"]["model_revision"]
                and int(receipt.get("tokens", -1)) == len(data.vocabulary)
                and sha256_file(array) == receipt["array_sha256"]
                and _verify_fingerprint(receipt, "receipt_identity")
            )
        except (OSError, KeyError, TypeError, ValueError):
            valid = False
    if not valid:
        report("frozen external vocabulary embedding START")
        if worker_output.exists():
            shutil.rmtree(worker_output)
        _stream_worker(Path(runtime["python"]), root, job_path, report)
        report("frozen external vocabulary embedding PASS")
    return array


def _run_schtm_fit(
    root: Path,
    out: Path,
    data: PreparedExternalData,
    embeddings: Path,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
    report,
) -> Path:
    worker_output = out / "worker_fit"
    job_path = out / "jobs/schtm_fit_job.json"
    job = _write_job(
        job_path,
        {
            "schema_version": 1,
            "mode": "fit",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output.resolve()),
            "inputs": {
                "fit_counts": _input_item(data.paths["fit_counts"]),
                "word_embeddings": _input_item(embeddings),
            },
            "fit_documents": int(data.fit_counts.shape[0]),
            "vocabulary_size": len(data.vocabulary),
            "original_train_documents": int(
                protocol["split"]["original_train_documents"]
            ),
            "spent_validation_documents": int(
                protocol["split"]["spent_validation_documents"]
            ),
            "official_source": str(runtime["official_source"]),
            "official_commit": runtime["official_commit"],
            "model": dict(protocol["model"]),
            "training": dict(protocol["training"]),
            "global_context": dict(protocol["global_context"]),
            "lexical_graph": dict(protocol["lexical_graph"]),
            "graph_projection": dict(protocol["graph_projection"]),
            "calibration": dict(protocol["calibration"]),
            "pooled_backoff": dict(protocol["pooled_backoff"]),
            "evaluation": dict(protocol["evaluation"]) | {"projection_epsilon": 1.0e-12},
            "policy": {
                "labels_available_to_models": False,
                "coordinates_available_to_models": False,
            },
        },
    )
    result_path = worker_output / "worker_result.json"
    valid = False
    if result_path.is_file():
        try:
            result = _json(result_path)
            valid = bool(
                result.get("status") == "PASS"
                and result.get("job_identity") == job["job_identity"]
                and int(result.get("parent_model_fits", -1)) == len(EXPECTED_SEEDS)
                and int(result.get("test_documents_used_for_fit", -1)) == 0
                and (worker_output / "checkpoint_manifest.json").is_file()
                and sha256_file(worker_output / "checkpoint_manifest.json")
                == result.get("checkpoint_manifest_sha256")
                and _verify_fingerprint(result, "result_identity")
            )
        except (OSError, KeyError, TypeError, ValueError):
            valid = False
    if not valid:
        report("SC-HTM external ten-seed fit START | held-out matrices unavailable")
        _stream_worker(Path(runtime["python"]), root, job_path, report)
        report("SC-HTM external ten-seed fit FROZEN | test used=0")
    return result_path


def _run_schtm_evaluation(
    root: Path,
    out: Path,
    data: PreparedExternalData,
    embeddings: Path,
    fit_result: Path,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
    report,
) -> Path:
    worker_output = out / "worker_evaluation"
    checkpoint_manifest = fit_result.parent / "checkpoint_manifest.json"
    job_path = out / "jobs/schtm_evaluation_job.json"
    job = _write_job(
        job_path,
        {
            "schema_version": 1,
            "mode": "evaluate",
            "implementation_version": IMPLEMENTATION,
            "worker_output": str(worker_output.resolve()),
            "inputs": {
                "fit_counts": _input_item(data.paths["fit_counts"]),
                "word_embeddings": _input_item(embeddings),
                "test_observed": _input_item(data.paths["test_observed"]),
                "test_target": _input_item(data.paths["test_target"]),
                "completion_rows": _input_item(data.paths["completion_rows"]),
                "vocabulary": _input_item(data.paths["vocabulary"]),
                "fit_result": _input_item(fit_result),
                "checkpoint_manifest": _input_item(checkpoint_manifest),
            },
            "fit_documents": int(data.fit_counts.shape[0]),
            "vocabulary_size": len(data.vocabulary),
            "official_source": str(runtime["official_source"]),
            "official_commit": runtime["official_commit"],
            "model": dict(protocol["model"]),
            "training": dict(protocol["training"]),
            "completion": dict(protocol["completion"]),
            "evaluation": dict(protocol["evaluation"]),
            "policy": {
                "labels_available_to_models": False,
                "coordinates_available_to_models": False,
            },
        },
    )
    result_path = worker_output / "worker_result.json"
    valid = False
    if result_path.is_file():
        try:
            result = _json(result_path)
            valid = bool(
                result.get("status") == "PASS"
                and result.get("job_identity") == job["job_identity"]
                and int(result.get("seeds_evaluated", -1)) == len(EXPECTED_SEEDS)
                and int(result.get("optimizer_steps", -1)) == 0
                and (worker_output / "per_seed_variant_metrics.csv").is_file()
                and sha256_file(worker_output / "per_seed_variant_metrics.csv")
                == result.get("per_seed_variant_metrics_sha256")
                and (worker_output / "top_words_by_seed.csv").is_file()
                and sha256_file(worker_output / "top_words_by_seed.csv")
                == result.get("top_words_by_seed_sha256")
                and (worker_output / "seed_integrity.csv").is_file()
                and sha256_file(worker_output / "seed_integrity.csv")
                == result.get("seed_integrity_sha256")
                and _verify_fingerprint(result, "result_identity")
            )
        except (OSError, KeyError, TypeError, ValueError):
            valid = False
    if not valid:
        report("SC-HTM external one-shot evaluation START | optimizer steps=0")
        if worker_output.exists():
            shutil.rmtree(worker_output)
        _stream_worker(Path(runtime["python"]), root, job_path, report)
        report("SC-HTM external one-shot evaluation PASS")
    return result_path


def _contextualize(counts: sparse.csr_matrix, embeddings: np.ndarray) -> np.ndarray:
    weighted = np.asarray(counts @ embeddings, dtype=np.float64)
    mass = np.asarray(counts.sum(axis=1)).reshape(-1, 1)
    if np.any(mass <= 0):
        raise ValueError("contextual adapter received an empty document")
    weighted /= mass
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(weighted).all():
        raise ValueError("contextual adapter produced a degenerate vector")
    return (weighted / norms).astype(np.float32)


def _adapter_data(
    out: Path,
    data: PreparedExternalData,
    embeddings_path: Path,
) -> tuple[BaselineData, EvaluationTarget, dict[str, Any]]:
    with np.load(embeddings_path, allow_pickle=False) as archive:
        word_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        embedded_tokens = tuple(archive["tokens"].astype(str))
    if embedded_tokens != data.vocabulary:
        raise ValueError("external embedding tokens and vocabulary are misaligned")
    fit_embeddings = _contextualize(data.fit_counts, word_embeddings)
    observed_embeddings = _contextualize(data.observed_counts, word_embeddings)
    fit_documents = counts_to_token_documents(data.fit_counts)
    observed_documents = counts_to_token_documents(data.observed_counts)
    fit_texts = tuple(
        " ".join(data.vocabulary[index] for index in document)
        for document in fit_documents
    )
    observed_texts = tuple(
        " ".join(data.vocabulary[index] for index in document)
        for document in observed_documents
    )
    identity_body = {
        "preparation_identity": data.preparation_identity,
        "fit_counts_logical_sha256": sparse_logical_hash(data.fit_counts),
        "observed_counts_logical_sha256": sparse_logical_hash(data.observed_counts),
        "word_embeddings_sha256": sha256_file(embeddings_path),
        "fit_documents": int(data.fit_counts.shape[0]),
        "observed_documents": int(data.observed_counts.shape[0]),
        "vocabulary": len(data.vocabulary),
        "target_available": False,
        "coordinates_available": False,
        "labels_available": False,
    }
    input_identity = sha256_object(identity_body)
    baseline = BaselineData(
        train_counts=data.fit_counts.astype(np.float64),
        validation_counts=data.observed_counts.astype(np.float64),
        train_embeddings=fit_embeddings,
        validation_embeddings=observed_embeddings,
        train_documents=fit_texts,
        validation_documents=observed_texts,
        train_token_documents=fit_documents,
        validation_token_documents=observed_documents,
        vocabulary=data.vocabulary,
        input_fingerprint=input_identity,
        heldout_partition="external_test_completion_observed_half",
    )
    evaluation_body = {
        "adapter_input_identity": input_identity,
        "target_logical_sha256": sparse_logical_hash(data.target_counts),
        "temporal_blocks_sha256": sha256_object(
            data.completion_rows["temporal_block"].astype(int).tolist()
        ),
        "completion_identity": data.completion_identity,
    }
    target = EvaluationTarget(
        counts=data.target_counts,
        temporal_blocks=data.completion_rows["temporal_block"].to_numpy(dtype=np.int64),
        evaluation_identity=sha256_object(evaluation_body),
    )
    kit = out / "comparator_input_kit"
    if kit.exists():
        shutil.rmtree(kit)
    kit.mkdir(parents=True)
    sparse.save_npz(kit / "X_train.npz", data.fit_counts)
    sparse.save_npz(kit / "X_test.npz", data.observed_counts)
    np.savez_compressed(kit / "E_train.npz", embeddings=fit_embeddings)
    np.savez_compressed(kit / "E_test.npz", embeddings=observed_embeddings)
    np.savez_compressed(
        kit / "S_vocabulary.npz",
        embeddings=word_embeddings,
        tokens=np.asarray(data.vocabulary),
    )
    data.vocabulary_frame[["index", "token"]].to_csv(
        kit / "vocabulary.csv", index=False, encoding="utf-8-sig"
    )
    files = []
    for name in (
        "X_train.npz",
        "X_test.npz",
        "E_train.npz",
        "E_test.npz",
        "S_vocabulary.npz",
        "vocabulary.csv",
    ):
        path = kit / name
        files.append(
            {
                "name": name,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    bundle_identity = sha256_object({"files": files})
    receipt = {
        "schema_version": 2,
        **identity_body,
        "adapter_input_identity": input_identity,
        "adapter_bundle_sha256": bundle_identity,
        "files": files,
        "target_files_present": 0,
        "city_or_label_files_present": 0,
        "raw_text_files_present": 0,
    }
    _write_json(kit / "adapter_input_receipt.json", receipt)
    _write_json(
        out / "evaluation_input_receipt.json",
        evaluation_body
        | {
            "evaluation_identity": target.evaluation_identity,
            "target_retention": "common_evaluator_only",
        },
    )
    return baseline, target, receipt


def _save_fit(path: Path, fit: Any, metadata: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "topic_word": np.asarray(fit.topic_word, dtype=np.float64)
    }
    if fit.validation_theta is not None:
        arrays["test_theta"] = np.asarray(fit.validation_theta, dtype=np.float64)
    if fit.validation_word_probabilities is not None:
        arrays["test_word_probability"] = np.asarray(
            fit.validation_word_probabilities, dtype=np.float64
        )
    np.savez_compressed(path / "canonical_output.npz", **arrays)
    body = dict(metadata)
    body["canonical_output_sha256"] = sha256_file(path / "canonical_output.npz")
    body["receipt_identity"] = sha256_object(body)
    _write_json(path / "metadata.json", body)


def _execution_identity(
    root: Path,
    method: str,
    parameters: Mapping[str, Any],
    input_identity: str,
    environment_identity: str = "current_verified_project_environment",
    source_identity: str = "bundled_audited_adapter",
) -> str:
    adapter = root / "src/v6/comparators" / f"{method}.py"
    worker = (
        root / "src/v6/step9_neuromax_worker.py"
        if method == "neuromax"
        else root / "src/v6/step9_official_worker.py"
    )
    return sha256_object(
        {
            "implementation_version": IMPLEMENTATION,
            "receipt_protocol": RECEIPT_PROTOCOL,
            "method_id": method,
            "topics": 10,
            "parameters": dict(parameters),
            "adapter_sha256": sha256_file(adapter) if adapter.is_file() else None,
            "worker_sha256": sha256_file(worker) if worker.is_file() else None,
            "input_identity": input_identity,
            "environment_identity": environment_identity,
            "source_identity": source_identity,
        }
    )


def _valid_method_receipt(
    path: Path,
    method: str,
    seed: int,
    input_identity: str,
    execution_identity: str,
    *,
    topics: int,
    vocabulary: int,
    documents: int,
) -> bool:
    try:
        metadata = _json(path / "metadata.json")
        body = dict(metadata)
        identity = body.pop("receipt_identity")
        if not (
            metadata.get("implementation_version") == IMPLEMENTATION
            and metadata.get("receipt_protocol") == RECEIPT_PROTOCOL
            and metadata.get("method_id") == method
            and int(metadata.get("seed", -1)) == seed
            and metadata.get("input_identity") == input_identity
            and metadata.get("execution_identity") == execution_identity
            and sha256_object(body) == identity
            and sha256_file(path / "canonical_output.npz")
            == metadata["canonical_output_sha256"]
        ):
            return False
        with np.load(path / "canonical_output.npz", allow_pickle=False) as archive:
            beta = np.asarray(archive["topic_word"], dtype=np.float64)
            theta = (
                np.asarray(archive["test_theta"], dtype=np.float64)
                if "test_theta" in archive.files
                else None
            )
            probability = (
                np.asarray(archive["test_word_probability"], dtype=np.float64)
                if "test_word_probability" in archive.files
                else None
            )
        if (
            beta.shape != (topics, vocabulary)
            or not np.isfinite(beta).all()
            or np.any(beta < 0)
            or not np.allclose(beta.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            return False
        if theta is not None and (
            theta.shape != (documents, topics)
            or not np.isfinite(theta).all()
            or np.any(theta < 0)
            or not np.allclose(theta.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            return False
        if method in NATIVE_PROBABILITY_METHODS and (
            probability is None
            or probability.shape != (documents, vocabulary)
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            return False
        return True
    except Exception:
        return False


def _run_established(
    root: Path,
    out: Path,
    data: BaselineData,
    protocol: Mapping[str, Any],
    report,
) -> None:
    for method in ESTABLISHED:
        parameters = dict(protocol["established"][method])
        execution = _execution_identity(
            root, method, parameters, data.input_fingerprint
        )
        for seed in EXPECTED_SEEDS:
            receipt = out / "receipts" / method / f"seed_{seed}"
            if _valid_method_receipt(
                receipt,
                method,
                seed,
                data.input_fingerprint,
                execution,
                topics=10,
                vocabulary=data.vocabulary_size,
                documents=data.validation_counts.shape[0],
            ):
                report(f"{method}/{seed} external receipt REUSED")
                continue
            report(f"{method}/{seed} external fit START")
            started = time.perf_counter()
            fit = make_adapter(method, parameters).fit(data, topics=10, seed=seed)
            _save_fit(
                receipt,
                fit,
                {
                    "schema_version": 1,
                    "implementation_version": IMPLEMENTATION,
                    "receipt_protocol": RECEIPT_PROTOCOL,
                    "execution_identity": execution,
                    "method_id": method,
                    "display_name": DISPLAY[method],
                    "family": "established",
                    "seed": seed,
                    "input_identity": data.input_fingerprint,
                    "topics": 10,
                    "actual_topics": int(fit.actual_topics),
                    "fit_documents": int(data.train_counts.shape[0]),
                    "inference_documents": int(data.validation_counts.shape[0]),
                    "inference_input": "observed_completion_half_only",
                    "target_available_to_adapter": False,
                    "coordinates_available_to_adapter": False,
                    "labels_available_to_adapter": False,
                    "runtime_seconds": time.perf_counter() - started,
                    "iterations_completed": fit.iterations_completed,
                    "converged": fit.converged,
                    "likelihood_comparable": bool(fit.likelihood_comparable),
                    "predictive_interface": fit.metadata.get(
                        "predictive_interface",
                        "theta_times_topic_word"
                        if fit.likelihood_comparable
                        else "not_comparable",
                    ),
                    "resolved_parameters": parameters,
                    "adapter_metadata": dict(fit.metadata),
                },
            )
            report(f"{method}/{seed} DONE")


def _recent_preflight(
    source_root: Path, out: Path
) -> dict[str, dict[str, Any]]:
    sources = _preflight_sources(source_root)
    result: dict[str, dict[str, Any]] = {}
    for method in SOTA:
        source_text = sources[method].get("source_dir")
        selected = _select_runtime(
            source_root, method, Path(source_text) if source_text else None
        )
        source = dict(sources[method])
        if method == "fastopic":
            source["source_artifact_sha256"] = selected["probe"][
                "distribution_sha256"
            ]
        result[method] = {"source": source, "runtime": selected}
    receipt = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "status": "PASS",
        "packages_installed_or_changed": 0,
        "methods": result,
    }
    receipt["preflight_identity"] = sha256_object(receipt)
    _write_json(out / "recent_sota_runtime_source_preflight.json", receipt)
    return result


def _run_recent_sota(
    root: Path,
    out: Path,
    data: BaselineData,
    adapter_receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    preflight: Mapping[str, Mapping[str, Any]],
    report,
) -> None:
    worker = root / "src/v6/step9_official_worker.py"
    neuro = root / "src/v6/step9_neuromax_worker.py"
    for method in SOTA:
        method_state = preflight[method]
        source = method_state["source"]
        runtime = method_state["runtime"]
        parameters = dict(protocol["recent_sota"][method])
        execution = _execution_identity(
            root,
            method,
            parameters,
            data.input_fingerprint,
            environment_identity=str(runtime["environment_identity"]),
            source_identity=str(source["source_artifact_sha256"]),
        )
        for seed in EXPECTED_SEEDS:
            receipt = out / "receipts" / method / f"seed_{seed}"
            if _valid_method_receipt(
                receipt,
                method,
                seed,
                data.input_fingerprint,
                execution,
                topics=10,
                vocabulary=data.vocabulary_size,
                documents=data.validation_counts.shape[0],
            ):
                report(f"{method}/{seed} external receipt REUSED")
                continue
            receipt.mkdir(parents=True, exist_ok=True)
            executable = neuro if method == "neuromax" else worker
            source_text = source.get("source_dir")
            job = {
                "schema_version": 2,
                "implementation_version": IMPLEMENTATION,
                "receipt_protocol": RECEIPT_PROTOCOL,
                "execution_identity": execution,
                "method_id": method,
                "seed": seed,
                "topics": 10,
                "input_dir": str((out / "comparator_input_kit").resolve()),
                "receipt_dir": str(receipt.resolve()),
                "source_dir": str(Path(source_text).resolve()) if source_text else None,
                "official_source_url": source["official_source_url"],
                "source_kind": source["source_kind"],
                "source_revision": source["source_revision"],
                "source_artifact_sha256": source["source_artifact_sha256"],
                "model_configuration_id": data.input_fingerprint,
                "input_fingerprint": data.input_fingerprint,
                "input_bundle_sha256": adapter_receipt["adapter_bundle_sha256"],
                "execution_policy_id": "v6_1_step13_preregistered_external",
                "execution_policy_sha256": sha256_object(protocol["policy"]),
                "execution_lock_sha256": sha256_object(
                    {
                        "external_protocol": protocol["recent_sota"],
                        "split": protocol["split"],
                        "preprocessing": protocol["preprocessing"],
                    }
                ),
                "allowed_overrides": ["topic_count", "seed", "common_numeric_inputs"],
                "hyperparameters": parameters,
                "compatibility_adapter": {
                    "required_model_dimension": 384,
                    "fit_partition": "external_fit_only",
                    "test_transform": "frozen_training_PCA",
                },
                "adapter_path": str(executable.resolve()),
            }
            job_path = out / "jobs" / f"{method}_seed_{seed}.json"
            _write_json(job_path, job)
            report(f"{method}/{seed} official external fit START")
            subprocess.run(
                [str(runtime["python"]), str(executable), "--job", str(job_path)],
                cwd=root,
                check=True,
            )
            metadata = _json(receipt / "metadata.json")
            metadata.update(
                {
                    "schema_version": 1,
                    "implementation_version": IMPLEMENTATION,
                    "receipt_protocol": RECEIPT_PROTOCOL,
                    "execution_identity": execution,
                    "input_identity": data.input_fingerprint,
                    "family": "recent_sota",
                    "inference_input": "observed_completion_half_only",
                    "target_available_to_adapter": False,
                    "coordinates_available_to_adapter": False,
                    "labels_available_to_adapter": False,
                    "count_vocabulary_adapter": "external_fit_only_english_vocabulary",
                    "document_embedding_adapter": "count_weighted_external_frozen_vocabulary_embeddings",
                    "likelihood_comparable": True,
                }
            )
            metadata.pop("receipt_identity", None)
            metadata["canonical_output_sha256"] = sha256_file(
                receipt / "canonical_output.npz"
            )
            metadata["receipt_identity"] = sha256_object(metadata)
            _write_json(receipt / "metadata.json", metadata)
            if not _valid_method_receipt(
                receipt,
                method,
                seed,
                data.input_fingerprint,
                execution,
                topics=10,
                vocabulary=data.vocabulary_size,
                documents=data.validation_counts.shape[0],
            ):
                raise RuntimeError(f"invalid external receipt: {method}/{seed}")
            report(f"{method}/{seed} DONE")


def _macro_temporal_probability(
    target: EvaluationTarget,
    probabilities: np.ndarray,
    probability_floor: float,
) -> tuple[float, float]:
    micro = document_completion_nll_from_probabilities(
        target.counts, probabilities, probability_floor=probability_floor
    )
    by_block = []
    for block in sorted(np.unique(target.temporal_blocks)):
        selected = np.flatnonzero(target.temporal_blocks == block)
        by_block.append(
            document_completion_nll_from_probabilities(
                target.counts[selected],
                probabilities[selected],
                probability_floor=probability_floor,
            ).nll_per_token
        )
    return float(np.mean(by_block)), float(micro.nll_per_token)


def _metrics(
    beta: np.ndarray,
    theta: np.ndarray | None,
    probabilities: np.ndarray | None,
    data: BaselineData,
    target: EvaluationTarget,
    protocol: Mapping[str, Any],
    *,
    likelihood_comparable: bool,
) -> dict[str, float | None]:
    normalized_beta = normalize_rows(beta)
    top = stable_top_word_indices(
        normalized_beta, int(protocol["evaluation"]["top_words"])
    )
    result: dict[str, float | None] = {
        "npmi_at_10": float(document_npmi(data.train_counts, top).mean),
        "c_v_at_10": float(
            cv_coherence(
                data.train_token_documents,
                top,
                vocabulary_size=data.vocabulary_size,
                window_size=int(protocol["evaluation"]["c_v_window_size"]),
                gamma=float(protocol["evaluation"]["c_v_gamma"]),
            ).mean
        ),
        "topic_diversity_at_10": float(topic_diversity(top)),
        "top_word_redundancy_at_10": float(mean_top_word_jaccard(top)),
        "macro_temporal_block_nll_per_token": None,
        "micro_nll_per_token": None,
    }
    if not likelihood_comparable:
        return result
    floor = float(protocol["evaluation"]["probability_floor"])
    if probabilities is not None:
        macro, micro = _macro_temporal_probability(target, probabilities, floor)
    else:
        if theta is None:
            raise ValueError("a comparable probabilistic method lacks inference output")
        normalized_theta = normalize_rows(theta)
        micro = document_completion_nll(
            target.counts,
            normalized_theta,
            normalized_beta,
            probability_floor=floor,
        ).nll_per_token
        values = []
        for block in sorted(np.unique(target.temporal_blocks)):
            selected = np.flatnonzero(target.temporal_blocks == block)
            values.append(
                document_completion_nll(
                    target.counts[selected],
                    normalized_theta[selected],
                    normalized_beta,
                    probability_floor=floor,
                ).nll_per_token
            )
        macro = float(np.mean(values))
    result["macro_temporal_block_nll_per_token"] = float(macro)
    result["micro_nll_per_token"] = float(micro)
    return result


def _bootstrap(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[
        rng.integers(0, len(values), size=(int(replicates), len(values)))
    ].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _holm(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        result[index] = min(1.0, running)
    return result.tolist()


def _method_receipt_execution(
    root: Path,
    method: str,
    data: BaselineData,
    protocol: Mapping[str, Any],
    preflight: Mapping[str, Mapping[str, Any]],
) -> str:
    if method in ESTABLISHED:
        return _execution_identity(
            root,
            method,
            protocol["established"][method],
            data.input_fingerprint,
        )
    state = preflight[method]
    return _execution_identity(
        root,
        method,
        protocol["recent_sota"][method],
        data.input_fingerprint,
        environment_identity=str(state["runtime"]["environment_identity"]),
        source_identity=str(state["source"]["source_artifact_sha256"]),
    )


def _collect_and_report(
    root: Path,
    out: Path,
    data: BaselineData,
    target: EvaluationTarget,
    protocol: Mapping[str, Any],
    preflight: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    full = pd.read_csv(
        out / "worker_evaluation/per_seed_variant_metrics.csv", encoding="utf-8-sig"
    )
    rows: list[dict[str, Any]] = []
    for row in full[full["variant"].eq(FULL_VARIANT)].to_dict(orient="records"):
        rows.append(
            {
                "method_id": "v6_1",
                "display_name": DISPLAY["v6_1"],
                "family": "proposed",
                "seed": int(row["seed"]),
                "likelihood_comparable": True,
                "predictive_interface": "external_native_encot_decoder_plus_frozen_calibration_and_pooled_backoff",
                **{
                    key: row[key]
                    for key in (
                        "npmi_at_10",
                        "c_v_at_10",
                        "topic_diversity_at_10",
                        "top_word_redundancy_at_10",
                        "macro_temporal_block_nll_per_token",
                        "micro_nll_per_token",
                    )
                },
            }
        )
    methods = list(ESTABLISHED) + (list(SOTA) if protocol["recent_sota"]["enabled"] else [])
    for method in methods:
        execution = _method_receipt_execution(
            root, method, data, protocol, preflight
        )
        for seed in EXPECTED_SEEDS:
            receipt = out / "receipts" / method / f"seed_{seed}"
            if not _valid_method_receipt(
                receipt,
                method,
                seed,
                data.input_fingerprint,
                execution,
                topics=10,
                vocabulary=data.vocabulary_size,
                documents=data.validation_counts.shape[0],
            ):
                raise RuntimeError(f"missing valid external receipt: {method}/{seed}")
            metadata = _json(receipt / "metadata.json")
            with np.load(receipt / "canonical_output.npz", allow_pickle=False) as z:
                beta = np.asarray(z["topic_word"], dtype=np.float64)
                theta = (
                    np.asarray(z["test_theta"], dtype=np.float64)
                    if "test_theta" in z.files
                    else None
                )
                probabilities = (
                    np.asarray(z["test_word_probability"], dtype=np.float64)
                    if "test_word_probability" in z.files
                    else None
                )
            rows.append(
                {
                    "method_id": method,
                    "display_name": DISPLAY[method],
                    "family": "established" if method in ESTABLISHED else "recent_sota",
                    "seed": seed,
                    "likelihood_comparable": bool(
                        metadata.get("likelihood_comparable", False)
                    ),
                    "predictive_interface": metadata.get(
                        "predictive_interface", "not_comparable"
                    ),
                    **_metrics(
                        beta,
                        theta,
                        probabilities,
                        data,
                        target,
                        protocol,
                        likelihood_comparable=bool(
                            metadata.get("likelihood_comparable", False)
                        ),
                    ),
                }
            )
    _write_csv(out / "all_method_seed_metrics.csv", rows)
    metrics = (
        "npmi_at_10",
        "c_v_at_10",
        "topic_diversity_at_10",
        "top_word_redundancy_at_10",
        "macro_temporal_block_nll_per_token",
        "micro_nll_per_token",
    )
    aggregate: list[dict[str, Any]] = []
    frame = pd.DataFrame(rows)
    for method, group in frame.groupby("method_id", sort=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                aggregate.append(
                    {
                        "method_id": method,
                        "display_name": DISPLAY[method],
                        "metric": metric,
                        "n": 0,
                        "mean": None,
                        "sample_sd": None,
                        "ci95_low": None,
                        "ci95_high": None,
                    }
                )
                continue
            bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"]) + int(
                hashlib.sha256(f"{method}:{metric}".encode()).hexdigest()[:8], 16
            )
            low, high = _bootstrap(
                values,
                bootstrap_seed,
                int(protocol["statistics"]["bootstrap_resamples"]),
            )
            aggregate.append(
                {
                    "method_id": method,
                    "display_name": DISPLAY[method],
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    _write_csv(out / "all_method_aggregate_metrics.csv", aggregate)
    paired: list[dict[str, Any]] = []
    for metric in metrics:
        base = frame[frame["method_id"].eq("v6_1")].set_index("seed")[metric]
        tests = []
        for method in methods:
            comparator = frame[frame["method_id"].eq(method)].set_index("seed")[metric]
            common = base.dropna().index.intersection(comparator.dropna().index)
            if len(common) != len(EXPECTED_SEEDS):
                continue
            effect = (base.loc[common] - comparator.loc[common]).to_numpy(float)
            if metric in {
                "top_word_redundancy_at_10",
                "macro_temporal_block_nll_per_token",
                "micro_nll_per_token",
            }:
                effect = -effect
            # SciPy switches away from the exact distribution when zero
            # differences are passed. Remove ties explicitly so the stated
            # exact Wilcoxon calculation remains true.
            nonzero_effect = effect[effect != 0.0]
            raw_p = (
                float(
                    stats.wilcoxon(
                        nonzero_effect,
                        alternative="two-sided",
                        method="exact",
                    ).pvalue
                )
                if len(nonzero_effect)
                else 1.0
            )
            tests.append(
                {
                    "metric": metric,
                    "comparator": method,
                    "n": len(common),
                    "nonzero_pairs": len(nonzero_effect),
                    "mean_effect_positive_favors_v6_1": float(effect.mean()),
                    "wins": int(np.sum(effect > 0)),
                    "ties": int(np.sum(effect == 0)),
                    "losses": int(np.sum(effect < 0)),
                    "raw_p": raw_p,
                }
            )
        for row, adjusted in zip(tests, _holm([row["raw_p"] for row in tests])):
            row["holm_p"] = adjusted
            paired.append(row)
    _write_csv(out / "paired_external_comparator_tests.csv", paired)
    ablation_aggregate: list[dict[str, Any]] = []
    for variant, group in full.groupby("variant", sort=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            ablation_aggregate.append(
                {
                    "variant": variant,
                    "display_name": VARIANT_DISPLAY[str(variant)],
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                }
            )
    _write_csv(out / "external_ablation_aggregate.csv", ablation_aggregate)
    targeted_specifications = (
        ("Graph transport", "without_graph_pooled_retained", "npmi_at_10", False),
        ("Graph transport", "without_graph_pooled_retained", "c_v_at_10", False),
        (
            "Pooled backoff",
            "without_pooled_graph_calibration_retained",
            "macro_temporal_block_nll_per_token",
            True,
        ),
        (
            "Decoder calibration",
            "without_calibration_graph_pooled_retained",
            "macro_temporal_block_nll_per_token",
            True,
        ),
        ("Joint V6.1 vs. EnCOT", "exact_official_parent", "npmi_at_10", False),
        ("Joint V6.1 vs. EnCOT", "exact_official_parent", "c_v_at_10", False),
        (
            "Joint V6.1 vs. EnCOT",
            "exact_official_parent",
            "macro_temporal_block_nll_per_token",
            True,
        ),
    )
    targeted_rows: list[dict[str, Any]] = []
    full_by_seed = full[full["variant"].eq(FULL_VARIANT)].set_index("seed")
    for component, comparator_variant, metric, lower_is_better in targeted_specifications:
        comparator_by_seed = full[
            full["variant"].eq(comparator_variant)
        ].set_index("seed")
        common = full_by_seed.index.intersection(comparator_by_seed.index)
        if len(common) != len(EXPECTED_SEEDS):
            raise ValueError(
                f"external ablation lacks all paired seeds: {comparator_variant}/{metric}"
            )
        full_values = full_by_seed.loc[common, metric].to_numpy(dtype=float)
        comparator_values = comparator_by_seed.loc[common, metric].to_numpy(dtype=float)
        effect = (
            comparator_values - full_values
            if lower_is_better
            else full_values - comparator_values
        )
        nonzero_effect = effect[effect != 0.0]
        raw_p = (
            float(
                stats.wilcoxon(
                    nonzero_effect,
                    alternative="two-sided",
                    method="exact",
                ).pvalue
            )
            if len(nonzero_effect)
            else 1.0
        )
        targeted_rows.append(
            {
                "component_comparison": component,
                "controlled_removal_or_parent": comparator_variant,
                "primary_endpoint": metric,
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "paired_seeds": len(common),
                "nonzero_pairs": len(nonzero_effect),
                "full_v6_1_mean": float(full_values.mean()),
                "controlled_mean": float(comparator_values.mean()),
                "mean_effect_positive_favors_v6_1": float(effect.mean()),
                "wins": int(np.sum(effect > 0.0)),
                "ties": int(np.sum(effect == 0.0)),
                "losses": int(np.sum(effect < 0.0)),
                "raw_p": raw_p,
            }
        )
    adjusted = _holm([row["raw_p"] for row in targeted_rows])
    for row, holm_p in zip(targeted_rows, adjusted):
        row["holm_p"] = holm_p
    _write_csv(out / "external_ablation_targeted_tests.csv", targeted_rows)
    _paper_tables(out, aggregate, ablation_aggregate, methods)
    return {
        "methods": 1 + len(methods),
        "method_seed_rows": len(rows),
        "paired_tests": len(paired),
        "targeted_ablation_tests": len(targeted_rows),
        "ablation_seed_rows": len(full),
        "aggregate_rows": len(aggregate),
    }


def _format(mean: Any, sd: Any) -> str:
    if pd.isna(mean) or mean is None:
        return "—"
    return f"{float(mean):.4f} ± {float(sd):.4f}"


def _paper_tables(
    out: Path,
    aggregate: Sequence[Mapping[str, Any]],
    ablation: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> None:
    long = pd.DataFrame(aggregate)
    order = ["v6_1", *methods]
    metrics = [
        ("npmi_at_10", "NPMI@10 ↑"),
        ("c_v_at_10", "C_v@10 ↑"),
        ("topic_diversity_at_10", "Diversity@10 ↑"),
        ("top_word_redundancy_at_10", "Redundancy@10 ↓"),
        ("macro_temporal_block_nll_per_token", "Macro-period NLL ↓"),
        ("micro_nll_per_token", "Micro NLL ↓"),
    ]
    rows = []
    for method in order:
        record: dict[str, Any] = {
            "Method": DISPLAY[method],
            "method_id": method,
        }
        selected = long[long["method_id"].eq(method)].set_index("metric")
        for metric, label in metrics:
            item = selected.loc[metric]
            record[label] = _format(item["mean"], item["sample_sd"])
        rows.append(record)
    paper = out / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        paper / "main_table_external_generalization.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ablation_frame = pd.DataFrame(ablation)
    ablation_rows = []
    for variant in VARIANT_DISPLAY:
        selected = ablation_frame[ablation_frame["variant"].eq(variant)].set_index(
            "metric"
        )
        record = {"Variant": VARIANT_DISPLAY[variant], "variant": variant}
        for metric, label in metrics:
            item = selected.loc[metric]
            record[label] = _format(item["mean"], item["sample_sd"])
        ablation_rows.append(record)
    pd.DataFrame(ablation_rows).to_csv(
        paper / "supplement_external_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# External temporal generalization table",
        "",
        "Mean ± sample standard deviation across the ten prespecified seeds. The vocabulary and all fitted state use only the first 475 chronological records; document-completion inference uses only the observed half of each of the final 84 records. Macro-period NLL is the equal-weight mean over three prespecified contiguous chronological test blocks, not a city metric. Dashes denote methods without a comparable predictive probability interface.",
        "",
        "| Method | NPMI@10 ↑ | C_v@10 ↑ | Diversity@10 ↑ | Redundancy@10 ↓ | Macro-period NLL ↓ | Micro NLL ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    str(row["Method"]),
                    str(row["NPMI@10 ↑"]),
                    str(row["C_v@10 ↑"]),
                    str(row["Diversity@10 ↑"]),
                    str(row["Redundancy@10 ↓"]),
                    str(row["Macro-period NLL ↓"]),
                    str(row["Micro NLL ↓"]),
                ]
            ) + " |"
        )
    (paper / "main_table_external_generalization.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _artifact_manifest(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        relative = path.relative_to(out)
        if (
            path.name == "generated_artifact_manifest.json"
            or "checkpoints" in relative.parts
        ):
            continue
        files.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    manifest = {"schema_version": 1, "files": files}
    manifest["manifest_identity"] = sha256_object(manifest)
    return manifest


def _step10_submission(
    out: Path,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
    coverage: Mapping[str, Any],
    reporting: Mapping[str, Any],
) -> Path:
    destination = out / "copy_contents_to_inputs_v6_step10_external"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    source_files = {
        "main_table_external_generalization.csv": out
        / "paper/main_table_external_generalization.csv",
        "all_method_seed_metrics.csv": out / "all_method_seed_metrics.csv",
        "paired_external_comparator_tests.csv": out
        / "paired_external_comparator_tests.csv",
        "external_ablation_targeted_tests.csv": out
        / "external_ablation_targeted_tests.csv",
        "dataset_audit.json": out / "dataset_audit.json",
        "exact_state_coverage_diagnostic.json": out
        / "exact_state_coverage_diagnostic.json",
        "evaluation_input_receipt.json": out / "evaluation_input_receipt.json",
    }
    for name, source in source_files.items():
        shutil.copy2(source, destination / name)
    files = [
        {
            "relative_path": name,
            "sha256": sha256_file(destination / name),
            "size_bytes": int((destination / name).stat().st_size),
        }
        for name in sorted(source_files)
    ]
    manifest = {"schema_version": 1, "files": files}
    manifest["manifest_identity"] = sha256_object(manifest)
    _write_json(destination / "generated_artifact_manifest.json", manifest)
    optimizer_updates = (
        len(EXPECTED_SEEDS)
        * int(protocol["training"]["parent_epochs"])
        * int(np.ceil(int(protocol["split"]["final_fit_documents"]) / int(protocol["training"]["batch_size"])))
    )
    contract = {
        "schema_version": 2,
        "status": "PASS",
        "evidence_kind": "independent_external_architecture_replication_with_frozen_hyperparameters",
        "dataset_ids": [protocol["dataset"]["dataset_id"]],
        "dataset_independent_of_internal_772_document_corpus": True,
        "external_language": "English",
        "internal_language": "Chinese",
        "step7_freeze_identity": runtime["freeze_identity"],
        "exact_frozen_state_assessment": coverage[
            "exact_state_performance_status"
        ],
        "exact_state_coverage_fraction": coverage[
            "external_vocabulary_coverage_fraction"
        ],
        "exact_state_performance_claimed": False,
        "architecture_unchanged": True,
        "model_and_training_hyperparameters_frozen_before_external_outcomes": True,
        "language_compatible_fit_vocabulary_and_embeddings_reestimated": True,
        "v6_model_fits": len(EXPECTED_SEEDS),
        "v6_optimizer_update_steps": optimizer_updates,
        "v6_architecture_or_hyperparameter_changes": 0,
        "test_conditioned_preprocessing_changes": 0,
        "ten_predeclared_seeds_reported": True,
        "all_prespecified_methods_reported": reporting["methods"] == 11,
        "unfavorable_results_omitted": False,
        "ethics_privacy_license_documented": True,
        "raw_coordinates_exported": False,
        "raw_text_exported": False,
        "ground_truth_topic_labels_available": False,
        "label_based_metrics_reported": False,
        "artifact_manifest_identity": manifest["manifest_identity"],
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    _write_json(destination / "external_evaluation_contract.json", contract)
    return destination


def _return_zip(root: Path, out: Path) -> Path:
    destination = root / "v6_step13_results.zip"
    if destination.is_file():
        destination.unlink()
    prefix = Path("step13_external_generalization") / out.name
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in out.rglob("*") if item.is_file()):
            relative = path.relative_to(out)
            if "checkpoints" in relative.parts:
                continue
            archive.write(path, (prefix / relative).as_posix())
    return destination


def run_v6_step13(
    *,
    project_root: Path,
    config_path: Path,
    source_root: Path | None = None,
) -> Path:
    root = project_root.expanduser().resolve()
    protocol = _load_protocol(config_path)
    out = root / "outputs/v6/step13_external_generalization/v6_1_transit_tweet_temporal"
    out.mkdir(parents=True, exist_ok=True)
    for relative in (
        "step13_contract.json",
        "generated_artifact_manifest.json",
        "hard_checks.csv",
        "failure.json",
    ):
        path = out / relative
        if path.is_file():
            path.unlink()
    log_path = out / "step13.log"

    def report(message: str) -> None:
        line = f"[V6.1 Step 13] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat()} | {line}\n")

    report(
        f"START | implementation={IMPLEMENTATION} | external outcomes cannot tune protocol"
    )
    try:
        data = _prepare_external(root, out, protocol)
        coverage = _exact_state_coverage(root, out, data)
        step7 = _find_step7_runtime(root)
        runtime = {
            "python": step7["python"],
            "official_source": step7["official_source"],
            "official_commit": step7["official_commit"],
            "freeze_identity": step7["freeze_identity"],
        }
        embeddings = _ensure_embeddings(root, out, data, protocol, runtime, report)
        fit_result = _run_schtm_fit(
            root, out, data, embeddings, protocol, runtime, report
        )
        evaluation_result = _run_schtm_evaluation(
            root,
            out,
            data,
            embeddings,
            fit_result,
            protocol,
            runtime,
            report,
        )
        baseline, target, adapter_receipt = _adapter_data(out, data, embeddings)
        _run_established(root, out, baseline, protocol, report)
        preflight: dict[str, dict[str, Any]] = {}
        if protocol["recent_sota"]["enabled"]:
            external_source = (
                source_root
                or root / str(protocol["recent_sota"]["source_project_root"])
            ).expanduser().resolve()
            preflight = _recent_preflight(external_source, out)
            _run_recent_sota(
                root,
                out,
                baseline,
                adapter_receipt,
                protocol,
                preflight,
                report,
            )
        reporting = _collect_and_report(
            root, out, baseline, target, protocol, preflight
        )
        submission = _step10_submission(
            out, protocol, runtime, coverage, reporting
        )
        checks = [
            ("external_dataset_hash_and_schema", True, data.audit["csv_sha256"]),
            ("559_nonblank_unique_records", data.audit["retained_records"] == 559 and data.audit["exact_duplicate_texts"] == 0, data.audit["retained_records"]),
            ("chronological_391_84_84_split", data.audit["split_documents"] == {"train": 391, "validation_spent": 84, "test": 84}, data.audit["split_documents"]),
            ("fit_only_vocabulary", data.audit["test_tokens_used_to_define_vocabulary"] == 0, data.audit["fit_vocabulary"]),
            ("raw_coordinates_never_loaded_or_exported", data.audit["coordinate_values_loaded"] is False and data.audit["coordinate_values_serialized"] is False, False),
            ("exact_state_mismatch_reported_not_scored", coverage["exact_state_performance_status"] == "NOT_ESTIMABLE_LANGUAGE_VOCABULARY_MISMATCH" and coverage["zero_shot_performance_claimed"] is False, coverage["external_vocabulary_coverage_fraction"]),
            ("ten_schtm_fits_frozen_before_test", _json(fit_result)["parent_model_fits"] == 10 and _json(fit_result)["test_documents_used_for_fit"] == 0, 10),
            ("one_shot_schtm_evaluation_no_updates", _json(evaluation_result)["optimizer_steps"] == 0 and _json(evaluation_result)["model_state_changes"] == 0, 0),
            ("all_prespecified_method_seed_rows", reporting["method_seed_rows"] == 110, reporting["method_seed_rows"]),
            ("all_five_schtm_variants_reported", reporting["ablation_seed_rows"] == 50, reporting["ablation_seed_rows"]),
            ("labels_and_label_metrics_absent", protocol["dataset"]["label_columns_available"] is False, False),
            ("step10_r3_submission_envelope_created", (submission / "external_evaluation_contract.json").is_file(), str(submission)),
        ]
        failed = [name for name, passed, _ in checks if not passed]
        _write_csv(
            out / "hard_checks.csv",
            [
                {
                    "check": name,
                    "status": "PASS" if passed else "FAIL",
                    "observed": observed,
                }
                for name, passed, observed in checks
            ],
        )
        if failed:
            raise RuntimeError(f"Step-13 hard checks failed: {failed}")
        contract = {
            "schema_version": 1,
            "implementation_version": IMPLEMENTATION,
            "scope": SCOPE,
            "status": "PASS",
            "readiness": READINESS,
            "dataset_id": protocol["dataset"]["dataset_id"],
            "dataset_csv_sha256": protocol["dataset"]["csv_sha256"],
            "preparation_identity": data.preparation_identity,
            "completion_identity": data.completion_identity,
            "step7_freeze_identity": step7["freeze_identity"],
            "external_fit_documents": int(data.fit_counts.shape[0]),
            "external_test_documents": int(data.observed_counts.shape[0]),
            "external_vocabulary": len(data.vocabulary),
            "exact_state_performance_status": coverage["exact_state_performance_status"],
            "external_architecture_replication": True,
            "v6_model_fits": 10,
            "methods_reported": reporting["methods"],
            "method_seed_rows": reporting["method_seed_rows"],
            "unfavorable_results_omitted": False,
            "labels_used": 0,
            "coordinates_used": 0,
            "test_conditioned_preprocessing_changes": 0,
            "seed_selection_performed": False,
            "hyperparameter_selection_performed": False,
            "hard_checks_passed": len(checks),
            "hard_checks_total": len(checks),
            "step10_submission_folder": str(submission.resolve()),
        }
        contract["contract_fingerprint"] = sha256_object(contract)
        _write_json(out / "step13_contract.json", contract)
        _write_json(out / "generated_artifact_manifest.json", _artifact_manifest(out))
        result_zip = _return_zip(root, out)
        report(
            f"PASS | hard checks={len(checks)}/{len(checks)} | methods={reporting['methods']} seeds=10"
        )
        report(f"paper-table={out / 'paper/main_table_external_generalization.md'}")
        report(f"Step10-R3-envelope={submission}")
        report(f"return-zip={result_zip}")
        return out
    except Exception as exc:
        _write_json(
            out / "failure.json",
            {
                "exception": type(exc).__name__,
                "message": str(exc),
                "completed_receipts_preserved": True,
                "external_outcomes_may_not_change_protocol": True,
            },
        )
        _write_json(out / "generated_artifact_manifest.json", _artifact_manifest(out))
        _return_zip(root, out)
        report(f"INCOMPLETE | {type(exc).__name__}: {exc}")
        raise
