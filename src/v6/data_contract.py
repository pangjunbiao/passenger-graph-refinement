"""Read-only migration contract for trusted V4 preprocessing artifacts.

Only train/validation count matrices, row identifiers, split metadata, the
training-only vocabulary, and frozen encoder evidence are admitted.  Test
counts, test text, annotations, previous metrics, and comparator outputs have
no code path in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class FrozenDevelopmentInputs:
    train_counts: sparse.csr_matrix
    validation_counts: sparse.csr_matrix
    train_rows: pd.DataFrame
    validation_rows: pd.DataFrame
    vocabulary: pd.DataFrame
    input_hashes: list[dict[str, Any]]
    access_ledger: list[dict[str, Any]]
    corpus_report: dict[str, Any]
    embedding_report: dict[str, Any]
    leakage_report: dict[str, Any]
    source_evidence_manifest: dict[str, Any]
    data_key: str


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def dense_logical_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sparse_logical_hash(values: sparse.spmatrix) -> str:
    matrix = values.tocsr(copy=True)
    matrix.sort_indices()
    matrix.eliminate_zeros()
    digest = sha256()
    digest.update(str(matrix.dtype).encode("ascii"))
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix.data).tobytes())
    digest.update(np.asarray(matrix.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indptr, dtype=np.int64).tobytes())
    return digest.hexdigest()


def legacy_count_logical_hash(values: sparse.spmatrix) -> str:
    """Reproduce the V4 vocabulary manifest's integer matrix digest."""

    matrix = values.tocsr().astype(np.int64)
    matrix.sort_indices()
    digest = sha256()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(np.asarray(array).tobytes(order="C"))
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def find_passing_step1_contract(v6_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (v6_root / "outputs" / "v6" / "step01_math_core").glob(
            "*/step01_contract.json"
        ),
        reverse=True,
    )
    for path in candidates:
        try:
            contract = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            contract.get("status") == "PASS"
            and contract.get("readiness")
            == "READY_FOR_V6_STEP2_EVALUATION_FIREWALL"
            and int(contract.get("hard_checks_passed", -1))
            == int(contract.get("hard_checks_total", -2))
        ):
            return path, contract
    raise RuntimeError(
        "No passing V6 Step-1 contract was found under "
        "outputs/v6/step01_math_core"
    )


def _require_files(paths: Mapping[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Step 2 cannot find the trusted preprocessing assets: "
            + "; ".join(missing)
        )


def _read_rows(path: Path, split: str) -> pd.DataFrame:
    rows = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"post_id": str, "city": str, "split": str},
    )
    required = {"matrix_row", "post_id", "city", "split"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    if rows["matrix_row"].astype(int).tolist() != list(range(len(rows))):
        raise ValueError(f"{path.name} matrix_row is not contiguous")
    if not rows["split"].astype(str).eq(split).all():
        raise ValueError(f"{path.name} contains a non-{split} row")
    if rows["post_id"].astype(str).duplicated().any():
        raise ValueError(f"{path.name} contains duplicate post IDs")
    return rows[["matrix_row", "post_id", "city", "split"]].copy()


def _validate_counts(matrix: sparse.spmatrix, name: str) -> sparse.csr_matrix:
    result = matrix.tocsr(copy=True)
    result.sort_indices()
    result.eliminate_zeros()
    if result.data.size:
        rounded = np.rint(result.data)
        if (
            np.any(result.data < 0)
            or not np.isfinite(result.data).all()
            or not np.allclose(result.data, rounded, atol=0.0, rtol=0.0)
        ):
            raise ValueError(f"{name} is not a finite nonnegative integer matrix")
        result.data = rounded.astype(np.int64)
    if np.any(np.asarray(result.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError(f"{name} contains a zero-information row")
    return result.astype(np.int64)


def _parse_included(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.casefold()
    valid = {"true", "false", "1", "0"}
    if not set(normalized).issubset(valid):
        raise ValueError("included_in_model contains an unrecognized Boolean value")
    return normalized.isin({"true", "1"})


def _unit_norm_report(values: np.ndarray, name: str, tolerance: float) -> dict[str, Any]:
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    if maximum_error > float(tolerance):
        raise ValueError(
            f"{name} unit-norm error {maximum_error:.3e} exceeds tolerance"
        )
    return {
        "rows": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "maximum_unit_norm_error": maximum_error,
        "logical_sha256": dense_logical_hash(values),
    }


def _manifest_expected_hash(
    manifest: Mapping[str, Any], kind: str
) -> str | None:
    if kind == "document":
        candidates = (
            manifest.get("embeddings", {}).get("document", {}).get("logical_sha256"),
            manifest.get("audited_run", {}).get(
                "document_embeddings_logical_sha256"
            ),
            manifest.get("document_embeddings", {}).get("logical_sha256"),
        )
    else:
        candidates = (
            manifest.get("embeddings", {})
            .get("vocabulary", {})
            .get("logical_sha256"),
            manifest.get("audited_run", {}).get(
                "vocabulary_embeddings_logical_sha256"
            ),
            manifest.get("vocabulary_embeddings", {}).get("logical_sha256"),
        )
    return next((str(value) for value in candidates if value not in (None, "")), None)


def _encoder_is_frozen(manifest: Mapping[str, Any]) -> bool:
    if "encoder_fine_tuned" in manifest:
        return not bool(manifest["encoder_fine_tuned"])
    encoder = manifest.get("encoder", {})
    if isinstance(encoder, Mapping) and "fine_tuned" in encoder:
        return not bool(encoder["fine_tuned"])
    return False


def load_frozen_development_inputs(
    source_root: Path,
    *,
    expected_cities: tuple[str, ...],
    embedding_norm_tolerance: float,
) -> FrozenDevelopmentInputs:
    """Load and validate only the authorized V4-derived artifacts."""

    root = source_root.expanduser().resolve()
    processed = root / "data" / "processed"
    paths = {
        "train_counts": processed / "counts" / "X_train.npz",
        "validation_counts": processed / "counts" / "X_validation.npz",
        "train_rows": processed / "counts" / "rows_train.csv",
        "validation_rows": processed / "counts" / "rows_validation.csv",
        "vocabulary": processed / "vocabulary" / "vocabulary.csv",
        "vocabulary_contract": processed / "vocabulary" / "vocabulary.json",
        "split_metadata": processed / "splits" / "split_assignments.csv",
        "split_manifest": processed / "splits" / "split_manifest.json",
        "document_embeddings": processed / "evidence" / "E_document.npz",
        "vocabulary_embeddings": processed / "evidence" / "S_vocabulary.npz",
        "evidence_manifest": processed / "evidence" / "evidence_manifest.json",
    }
    _require_files(paths)
    input_hashes = [
        {
            "role": role,
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "access_status": "READ_AND_AUDITED",
        }
        for role, path in paths.items()
    ]

    vocabulary = pd.read_csv(
        paths["vocabulary"], encoding="utf-8-sig", dtype={"token": str}
    ).sort_values("index").reset_index(drop=True)
    if {"index", "token"} - set(vocabulary.columns):
        raise ValueError("vocabulary.csv lacks index or token")
    if vocabulary["index"].astype(int).tolist() != list(range(len(vocabulary))):
        raise ValueError("vocabulary indices are not contiguous")
    if vocabulary["token"].astype(str).duplicated().any():
        raise ValueError("vocabulary tokens are not unique")
    vocabulary_contract = json.loads(
        paths["vocabulary_contract"].read_text(encoding="utf-8-sig")
    )
    vocabulary_manifest = vocabulary_contract.get("manifest", {})
    if vocabulary_manifest.get("scope") != "training_partition_only":
        raise ValueError("the source vocabulary is not training-only")
    if int(vocabulary_manifest.get("heldout_documents_used_for_vocabulary", -1)) != 0:
        raise ValueError("held-out documents contributed to the source vocabulary")
    observed_vocabulary_hash = sha256_object(
        [
            {"index": int(row.index), "token": str(row.token)}
            for row in vocabulary[["index", "token"]].itertuples(index=False)
        ]
    )
    expected_vocabulary_hash = vocabulary_manifest.get("vocabulary_sha256")
    if expected_vocabulary_hash not in (None, "") and str(
        expected_vocabulary_hash
    ) != observed_vocabulary_hash:
        raise ValueError("vocabulary logical hash disagrees with vocabulary.json")

    train_counts = _validate_counts(
        sparse.load_npz(paths["train_counts"]), "X_train"
    )
    validation_counts = _validate_counts(
        sparse.load_npz(paths["validation_counts"]), "X_validation"
    )
    train_rows = _read_rows(paths["train_rows"], "train")
    validation_rows = _read_rows(paths["validation_rows"], "validation")
    if train_counts.shape != (len(train_rows), len(vocabulary)):
        raise ValueError("X_train, rows_train, and vocabulary are misaligned")
    if validation_counts.shape != (len(validation_rows), len(vocabulary)):
        raise ValueError("X_validation, rows_validation, and vocabulary are misaligned")
    expected_shapes = vocabulary_manifest.get("matrix_split_shapes", {})
    expected_count_hashes = vocabulary_manifest.get("matrix_split_logical_sha256", {})
    for split, matrix in (("train", train_counts), ("validation", validation_counts)):
        if split in expected_shapes and list(map(int, expected_shapes[split])) != list(
            matrix.shape
        ):
            raise ValueError(f"{split} count shape disagrees with vocabulary.json")
        if split in expected_count_hashes and str(
            expected_count_hashes[split]
        ) != legacy_count_logical_hash(matrix):
            raise ValueError(f"{split} count hash disagrees with vocabulary.json")
    development_ids = set(train_rows["post_id"]) | set(validation_rows["post_id"])
    if len(development_ids) != len(train_rows) + len(validation_rows):
        raise ValueError("a post ID crosses train and validation")

    assignment_header = pd.read_csv(
        paths["split_metadata"], encoding="utf-8-sig", nrows=0
    ).columns.tolist()
    assignment_columns = ["post_id", "city", "split", "group_sha256"]
    if "included_in_model" in assignment_header:
        assignment_columns.append("included_in_model")
    assignments = pd.read_csv(
        paths["split_metadata"],
        encoding="utf-8-sig",
        usecols=assignment_columns,
        dtype={
            "post_id": str,
            "city": str,
            "split": str,
            "group_sha256": str,
        },
    )
    if assignments["post_id"].duplicated().any():
        raise ValueError("split assignments contain duplicate post IDs")
    if assignments["group_sha256"].astype(str).str.strip().eq("").any():
        raise ValueError("a split assignment lacks duplicate-group metadata")
    split_manifest = json.loads(
        paths["split_manifest"].read_text(encoding="utf-8-sig")
    )
    assignment_records = (
        assignments[["post_id", "city", "split", "group_sha256"]]
        .sort_values("post_id")
        .to_dict("records")
    )
    observed_assignment_hash = sha256_object(assignment_records)
    expected_assignment_hash = split_manifest.get("assignment_sha256")
    if expected_assignment_hash not in (None, "") and str(
        expected_assignment_hash
    ) != observed_assignment_hash:
        raise ValueError("split assignment logical hash disagrees with split_manifest")

    assignment_lookup = assignments.set_index("post_id")
    if not development_ids.issubset(set(assignment_lookup.index)):
        raise ValueError("development rows are missing from split assignments")
    for frame, split in ((train_rows, "train"), (validation_rows, "validation")):
        aligned = assignment_lookup.loc[frame["post_id"].tolist()]
        if not aligned["split"].astype(str).eq(split).all():
            raise ValueError(f"{split} row metadata disagrees with split assignments")
        if not np.array_equal(
            aligned["city"].astype(str).to_numpy(), frame["city"].astype(str).to_numpy()
        ):
            raise ValueError(f"{split} city metadata disagrees with split assignments")
        if "included_in_model" in aligned and not _parse_included(
            aligned["included_in_model"]
        ).all():
            raise ValueError(f"an excluded assignment appears in rows_{split}.csv")
        frame["group_sha256"] = aligned["group_sha256"].astype(str).to_numpy()

    train_groups = set(train_rows["group_sha256"])
    validation_groups = set(validation_rows["group_sha256"])
    test_assignments = assignments.loc[assignments["split"].eq("test")].copy()
    test_groups = set(test_assignments["group_sha256"].astype(str))
    if train_groups & validation_groups or (train_groups | validation_groups) & test_groups:
        raise ValueError("a duplicate/author group crosses a frozen partition")
    test_ids = set(test_assignments["post_id"].astype(str))
    if development_ids & test_ids:
        raise ValueError("a post ID crosses development and test")

    source_manifest = json.loads(
        paths["evidence_manifest"].read_text(encoding="utf-8-sig")
    )
    if not str(source_manifest.get("status", "")).startswith("PASS"):
        raise ValueError("the frozen evidence manifest is not passing")
    if not _encoder_is_frozen(source_manifest):
        raise ValueError("the source encoder is not explicitly frozen")

    with np.load(paths["document_embeddings"], allow_pickle=False) as archive:
        all_document_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        embedding_post_ids = archive["post_ids"].astype(str).tolist()
    if len(embedding_post_ids) != len(set(embedding_post_ids)):
        raise ValueError("document embedding IDs are duplicated")
    if all_document_embeddings.shape[0] != len(embedding_post_ids):
        raise ValueError("document embeddings and IDs are misaligned")
    embedding_lookup = {
        post_id: index for index, post_id in enumerate(embedding_post_ids)
    }
    ordered_development_ids = (
        train_rows["post_id"].astype(str).tolist()
        + validation_rows["post_id"].astype(str).tolist()
    )
    missing_embeddings = [
        post_id for post_id in ordered_development_ids if post_id not in embedding_lookup
    ]
    if missing_embeddings:
        raise ValueError(
            f"frozen document embeddings miss {len(missing_embeddings)} development IDs"
        )
    selected_document_embeddings = all_document_embeddings[
        [embedding_lookup[post_id] for post_id in ordered_development_ids]
    ]
    document_report = _unit_norm_report(
        selected_document_embeddings,
        "development document embeddings",
        embedding_norm_tolerance,
    )
    source_document_hash = dense_logical_hash(all_document_embeddings)
    expected_document_hash = _manifest_expected_hash(source_manifest, "document")
    if expected_document_hash and source_document_hash != expected_document_hash:
        raise ValueError("document-embedding logical hash disagrees with manifest")

    with np.load(paths["vocabulary_embeddings"], allow_pickle=False) as archive:
        vocabulary_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        embedded_tokens = archive["tokens"].astype(str).tolist()
    if embedded_tokens != vocabulary["token"].astype(str).tolist():
        raise ValueError("vocabulary embeddings are not vocabulary-row aligned")
    vocabulary_report = _unit_norm_report(
        vocabulary_embeddings,
        "vocabulary embeddings",
        embedding_norm_tolerance,
    )
    expected_vocabulary_hash = _manifest_expected_hash(source_manifest, "vocabulary")
    if expected_vocabulary_hash and (
        vocabulary_report["logical_sha256"] != expected_vocabulary_hash
    ):
        raise ValueError("vocabulary-embedding logical hash disagrees with manifest")

    observed_cities = set(train_rows["city"]) | set(validation_rows["city"])
    if observed_cities != set(expected_cities):
        raise ValueError(
            f"city mismatch: observed={sorted(observed_cities)}, "
            f"expected={list(expected_cities)}"
        )
    for frame, split in ((train_rows, "train"), (validation_rows, "validation")):
        if set(frame["city"]) != set(expected_cities):
            raise ValueError(f"{split} does not contain every expected city")

    model_assignment_count = None
    modeled_test_assignment_count = None
    if "included_in_model" in assignments:
        included = _parse_included(assignments["included_in_model"])
        model_assignment_count = int(included.sum())
        modeled_test_assignment_count = int(
            (included & assignments["split"].eq("test")).sum()
        )
    corpus_report = {
        "schema_version": 1,
        "source_project_root": str(root),
        "assigned_documents_before_model_exclusions": int(len(assignments)),
        "split_manifest_documents": int(split_manifest.get("documents", len(assignments))),
        "included_in_model_assignments": model_assignment_count,
        "modeled_rows_loaded": {
            "train": int(len(train_rows)),
            "validation": int(len(validation_rows)),
        },
        "test_assignment_metadata_rows": int(len(test_assignments)),
        "modeled_test_assignment_metadata_rows": modeled_test_assignment_count,
        "test_count_rows_loaded": 0,
        "vocabulary_size": int(len(vocabulary)),
        "vocabulary_logical_sha256": observed_vocabulary_hash,
        "vocabulary_and_count_manifest_verified": True,
        "token_mass": {
            "train": int(train_counts.sum()),
            "validation": int(validation_counts.sum()),
        },
        "city_rows": {
            split: {
                city: int((frame["city"] == city).sum()) for city in expected_cities
            }
            for split, frame in (("train", train_rows), ("validation", validation_rows))
        },
        "assignment_sha256": observed_assignment_hash,
        "source_counts_logical_sha256": {
            "train": sparse_logical_hash(train_counts),
            "validation": sparse_logical_hash(validation_counts),
        },
    }
    embedding_report = {
        "schema_version": 1,
        "encoder_frozen": True,
        "source_document_embedding_rows_loaded_for_alignment": int(
            len(all_document_embeddings)
        ),
        "development_document_embeddings_selected": int(
            len(selected_document_embeddings)
        ),
        "test_document_embeddings_selected": 0,
        "document": document_report,
        "vocabulary": vocabulary_report,
        "source_document_logical_sha256": source_document_hash,
    }
    leakage_report = {
        "schema_version": 1,
        "status": "PASS",
        "train_validation_group_overlap": 0,
        "development_test_group_overlap": 0,
        "development_test_post_id_overlap": 0,
        "test_assignment_metadata_rows_read": int(len(test_assignments)),
        "test_count_matrix_accessed": 0,
        "test_text_accessed": 0,
        "human_annotations_accessed": 0,
        "previous_quality_metrics_accessed": 0,
        "baseline_outputs_accessed": 0,
        "sota_outputs_accessed": 0,
    }
    access_ledger = [
        {
            "role": record["role"],
            "relative_path": record["relative_path"],
            "access_status": "READ_AND_AUDITED",
            "partition_use": (
                "development_only"
                if record["role"] not in {"split_metadata", "split_manifest"}
                else "development_plus_test_assignment_metadata_only"
            ),
        }
        for record in input_hashes
    ]
    access_ledger.extend(
        [
            {
                "role": "test_counts",
                "relative_path": "data/processed/counts/X_test.npz",
                "access_status": "PROHIBITED_NOT_OPENED",
                "partition_use": "none",
            },
            {
                "role": "processed_documents_with_test_text",
                "relative_path": "data/processed/internal/documents.jsonl",
                "access_status": "PROHIBITED_NOT_OPENED",
                "partition_use": "none",
            },
            {
                "role": "human_annotations",
                "relative_path": "data/annotations",
                "access_status": "PROHIBITED_NOT_OPENED",
                "partition_use": "none",
            },
            {
                "role": "previous_results_and_comparators",
                "relative_path": "outputs",
                "access_status": "PROHIBITED_NOT_OPENED",
                "partition_use": "none",
            },
        ]
    )
    key_specification = {
        "schema_version": 1,
        "train_counts": corpus_report["source_counts_logical_sha256"]["train"],
        "validation_counts": corpus_report["source_counts_logical_sha256"][
            "validation"
        ],
        "train_post_ids": sha256_object(train_rows["post_id"].tolist()),
        "validation_post_ids": sha256_object(validation_rows["post_id"].tolist()),
        "assignment_sha256": observed_assignment_hash,
        "vocabulary_tokens": sha256_object(vocabulary["token"].tolist()),
        "document_embeddings": document_report["logical_sha256"],
        "vocabulary_embeddings": vocabulary_report["logical_sha256"],
    }
    return FrozenDevelopmentInputs(
        train_counts=train_counts,
        validation_counts=validation_counts,
        train_rows=train_rows,
        validation_rows=validation_rows,
        vocabulary=vocabulary,
        input_hashes=input_hashes,
        access_ledger=access_ledger,
        corpus_report=corpus_report,
        embedding_report=embedding_report,
        leakage_report=leakage_report,
        source_evidence_manifest=source_manifest,
        data_key=sha256_object(key_specification),
    )


def deterministic_group_folds(
    rows: pd.DataFrame, *, folds: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build balanced city-stratified folds without an extra dependency."""

    required = {"post_id", "city", "split", "group_sha256"}
    if required - set(rows.columns):
        raise ValueError("fold rows lack required identity/group fields")
    if int(folds) < 2:
        raise ValueError("at least two folds are required")
    plan = rows.reset_index(drop=True).copy()
    plan.insert(0, "source_matrix_row", np.arange(len(plan), dtype=np.int64))
    plan["fold"] = -1
    city_loads: dict[str, list[int]] = {}
    for city, city_frame in plan.groupby("city", sort=True):
        groups = [
            (str(group), indices.to_numpy(dtype=np.int64))
            for group, indices in city_frame.groupby("group_sha256", sort=True).groups.items()
        ]
        if len(groups) < int(folds):
            raise ValueError(f"city {city} has fewer duplicate groups than folds")
        groups.sort(
            key=lambda item: (
                -len(item[1]),
                sha256(
                    f"SC-HTM-V6|fold|{int(seed)}|{city}|{item[0]}".encode("utf-8")
                ).hexdigest(),
            )
        )
        loads = [0] * int(folds)
        for group, indices in groups:
            minimum = min(loads)
            candidates = [index for index, value in enumerate(loads) if value == minimum]
            digest = sha256(
                f"SC-HTM-V6|tie|{int(seed)}|{city}|{group}".encode("utf-8")
            ).hexdigest()
            chosen = candidates[int(digest[:8], 16) % len(candidates)]
            plan.loc[indices, "fold"] = chosen
            loads[chosen] += len(indices)
        city_loads[str(city)] = loads
    if (plan["fold"] < 0).any():
        raise AssertionError("some training rows were not assigned to a fold")
    group_fold_counts = plan.groupby("group_sha256")["fold"].nunique()
    if int(group_fold_counts.max()) != 1:
        raise AssertionError("a duplicate group was split across folds")
    cities = sorted(plan["city"].astype(str).unique())
    for fold in range(int(folds)):
        if set(plan.loc[plan["fold"].eq(fold), "city"].astype(str)) != set(cities):
            raise ValueError(f"fold {fold} does not contain every city")
    report = {
        "schema_version": 1,
        "construction": "deterministic_city_stratified_duplicate_group_greedy",
        "seed": int(seed),
        "folds": int(folds),
        "documents": int(len(plan)),
        "groups": int(plan["group_sha256"].nunique()),
        "group_crossings": 0,
        "city_fold_loads": city_loads,
        "plan_sha256": sha256_object(
            plan[
                ["post_id", "city", "split", "group_sha256", "fold"]
            ].to_dict("records")
        ),
    }
    return plan, report
