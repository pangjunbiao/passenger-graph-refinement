import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.data_contract import dense_logical_hash, sha256_object
from src.v6.step2 import run_v6_step2


def _unit_rows(rng: np.random.Generator, rows: int, dimension: int) -> np.ndarray:
    values = rng.normal(size=(rows, dimension)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _build_source(root: Path) -> None:
    counts_dir = root / "data" / "processed" / "counts"
    vocabulary_dir = root / "data" / "processed" / "vocabulary"
    split_dir = root / "data" / "processed" / "splits"
    evidence_dir = root / "data" / "processed" / "evidence"
    for directory in (counts_dir, vocabulary_dir, split_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)

    vocabulary = pd.DataFrame(
        {"index": range(12), "token": [f"token_{index}" for index in range(12)]}
    )
    vocabulary.to_csv(vocabulary_dir / "vocabulary.csv", index=False, encoding="utf-8-sig")
    (vocabulary_dir / "vocabulary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest": {
                    "scope": "training_partition_only",
                    "heldout_documents_used_for_vocabulary": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    cities = ("beijing", "shanghai", "xiamen")
    train_records = []
    validation_records = []
    test_records = []
    for city in cities:
        for index in range(5):
            post_id = f"{city}-train-{index}"
            train_records.append(
                {"post_id": post_id, "city": city, "split": "train", "group_sha256": f"g-{post_id}"}
            )
        post_id = f"{city}-validation"
        validation_records.append(
            {"post_id": post_id, "city": city, "split": "validation", "group_sha256": f"g-{post_id}"}
        )
        post_id = f"{city}-test"
        test_records.append(
            {"post_id": post_id, "city": city, "split": "test", "group_sha256": f"g-{post_id}"}
        )
    rng = np.random.default_rng(91)
    train_counts = rng.integers(0, 4, size=(len(train_records), 12), dtype=np.int64)
    validation_counts = rng.integers(0, 4, size=(len(validation_records), 12), dtype=np.int64)
    train_counts[:, 0] += 2
    validation_counts[:, 0] += 2
    sparse.save_npz(counts_dir / "X_train.npz", sparse.csr_matrix(train_counts))
    sparse.save_npz(counts_dir / "X_validation.npz", sparse.csr_matrix(validation_counts))
    for split, records in (("train", train_records), ("validation", validation_records)):
        frame = pd.DataFrame.from_records(records)[["post_id", "city", "split"]]
        frame.insert(0, "matrix_row", range(len(frame)))
        frame.to_csv(counts_dir / f"rows_{split}.csv", index=False, encoding="utf-8-sig")

    assignments = pd.DataFrame.from_records(train_records + validation_records + test_records)
    assignments["included_in_model"] = True
    assignments.to_csv(split_dir / "split_assignments.csv", index=False, encoding="utf-8-sig")
    assignment_hash = sha256_object(
        assignments[["post_id", "city", "split", "group_sha256"]]
        .sort_values("post_id")
        .to_dict("records")
    )
    (split_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": len(assignments),
                "assignment_sha256": assignment_hash,
            }
        ),
        encoding="utf-8",
    )

    all_ids = np.asarray(assignments["post_id"].astype(str).tolist(), dtype="U64")
    document_embeddings = _unit_rows(rng, len(all_ids), 5)
    vocabulary_embeddings = _unit_rows(rng, len(vocabulary), 5)
    np.savez_compressed(
        evidence_dir / "E_document.npz",
        embeddings=document_embeddings,
        post_ids=all_ids,
    )
    np.savez_compressed(
        evidence_dir / "S_vocabulary.npz",
        embeddings=vocabulary_embeddings,
        tokens=np.asarray(vocabulary["token"].astype(str).tolist(), dtype="U32"),
    )
    (evidence_dir / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "encoder_fine_tuned": False,
                "embeddings": {
                    "document": {"logical_sha256": dense_logical_hash(document_embeddings)},
                    "vocabulary": {"logical_sha256": dense_logical_hash(vocabulary_embeddings)},
                },
            }
        ),
        encoding="utf-8",
    )


def test_step2_end_to_end_synthetic_filesystem(tmp_path):
    v6_root = tmp_path / "SC-HTM-V6"
    source_root = tmp_path / "SC-HTM"
    _build_source(source_root)
    step1 = v6_root / "outputs" / "v6" / "step01_math_core" / "passing"
    step1.mkdir(parents=True)
    (step1 / "step01_contract.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "readiness": "READY_FOR_V6_STEP2_EVALUATION_FIREWALL",
                "hard_checks_passed": 20,
                "hard_checks_total": 20,
            }
        ),
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parents[1]
    output = run_v6_step2(
        project_root=v6_root,
        config_path=package_root / "configs" / "v6" / "step02.yaml",
        source_root=source_root,
    )
    contract = json.loads((output / "step02_contract.json").read_text(encoding="utf-8"))
    assert contract["status"] == "PASS"
    assert contract["readiness"] == "READY_FOR_V6_STEP3_OFFICIAL_ENCOT_REPRODUCTION"
    assert contract["hard_checks_passed"] == contract["hard_checks_total"]
    assert contract["test_count_matrix_accessed"] == 0
    assert contract["real_model_fits"] == 0
