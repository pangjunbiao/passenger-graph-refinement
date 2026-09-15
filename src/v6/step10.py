"""V6.1 Step 10: immutable reporting, human-validity, and external-evidence gate.

This stage never fits or changes a model.  It verifies the exact Step-8 and
Step-9 evidence, renders prespecified paper tables, prepares blinded human
evaluation material from a fixed seed, and prepares an external-evaluation
contract.  Missing human or external evidence is reported as pending, never as
a successful result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from src.v6.data_contract import sha256_file, sha256_object
from src.v6.evaluation import stable_top_word_indices


IMPLEMENTATION = "v6_1_step10_reporting_validity_gate_r3"
SCOPE = "frozen_reporting_blinded_human_case_study_and_external_protocol"
PENDING_BOTH_READINESS = "AWAITING_INDEPENDENT_HUMAN_AND_EXTERNAL_EVIDENCE"
PENDING_HUMAN_READINESS = "AWAITING_INDEPENDENT_HUMAN_EVIDENCE"
PENDING_EXTERNAL_READINESS = "AWAITING_INDEPENDENT_EXTERNAL_EVIDENCE"
FINAL_READINESS = "READY_FOR_MANUSCRIPT_COMPLETION_WITH_REPORTED_LIMITATIONS"

METRICS = (
    "npmi_at_10",
    "c_v_at_10",
    "topic_diversity_at_10",
    "top_word_redundancy_at_10",
    "macro_city_nll_per_token",
    "micro_nll_per_token",
)
LOWER_IS_BETTER = {
    "top_word_redundancy_at_10",
    "macro_city_nll_per_token",
    "micro_nll_per_token",
}
DISPLAY_HEADERS = {
    "npmi_at_10": "NPMI@10 ↑",
    "c_v_at_10": "C_v@10 ↑",
    "topic_diversity_at_10": "TD@10 ↑",
    "top_word_redundancy_at_10": "Redundancy@10 ↓",
    "macro_city_nll_per_token": "Macro-city NLL/token ↓",
    "micro_nll_per_token": "Micro NLL/token ↓",
}
VARIANT_DISPLAY = {
    "full_graph_calibration_pooled": "Full SC-HTM V6.1",
    "without_graph_pooled_retained": "Without graph-prototype projection",
    "without_pooled_graph_calibration_retained": "Without pooled lexical backoff",
    "without_calibration_graph_pooled_retained": "Without training-moment calibration",
    "exact_official_parent": "Exact official EnCOT parent",
}
VARIANT_TARGET = {
    "full_graph_calibration_pooled": "Complete model",
    "without_graph_pooled_retained": "NPMI@10 and C_v@10",
    "without_pooled_graph_calibration_retained": "Macro-city NLL/token",
    "without_calibration_graph_pooled_retained": "Macro-city NLL/token",
    "exact_official_parent": "Joint incremental value",
}
VARIANT_TYPE = {
    "full_graph_calibration_pooled": "Reference",
    "without_graph_pooled_retained": "Module removal",
    "without_pooled_graph_calibration_retained": "Module removal",
    "without_calibration_graph_pooled_retained": "Module removal",
    "exact_official_parent": "Architectural parent control",
}


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        if materialized:
            writer.writerows(materialized)


def _assert_locked_columns(
    returned: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    locked_columns: Sequence[str],
    label: str,
) -> None:
    """Require every non-response field to match the issued blinded form."""

    required = {"item_id", *locked_columns}
    for name, frame in (("returned", returned), ("issued", reference)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label}: {name} form lacks locked columns {sorted(missing)}")
        if frame["item_id"].astype(str).duplicated().any():
            raise ValueError(f"{label}: {name} form contains duplicate item IDs")
    returned_ids = set(returned["item_id"].astype(str))
    reference_ids = set(reference["item_id"].astype(str))
    if returned_ids != reference_ids:
        raise ValueError(f"{label}: returned item IDs differ from the issued form")
    left = returned.assign(item_id=returned["item_id"].astype(str)).set_index("item_id")
    right = reference.assign(item_id=reference["item_id"].astype(str)).set_index("item_id")
    for column in locked_columns:
        observed = left.loc[right.index, column].fillna("").astype(str)
        expected = right[column].fillna("").astype(str)
        changed = observed.ne(expected)
        if changed.any():
            items = changed[changed].index.astype(str).tolist()[:5]
            raise ValueError(
                f"{label}: locked field {column!r} changed for item(s) {items}"
            )


def _verify_fingerprint(value: Mapping[str, Any], key: str) -> bool:
    body = dict(value)
    expected = body.pop(key, None)
    return isinstance(expected, str) and sha256_object(body) == expected


def _load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1:
        raise ValueError("Step-10 configuration schema changed")
    protocol = document.get("v6", {}).get("step10", {})
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-10 implementation/configuration lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-10 scope changed")
    reporting = protocol["reporting"]
    expected_seeds = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]
    if list(map(int, reporting["seeds"])) != expected_seeds:
        raise ValueError("The ten reporting seeds changed")
    if int(reporting["descriptive_single_seed"]) not in expected_seeds:
        raise ValueError("The supplementary descriptive seed is invalid")
    human = protocol["human_evaluation"]
    if int(human["blinded_seed"]) != expected_seeds[0]:
        raise ValueError("Human evaluation must use the first prespecified seed")
    if int(human["required_annotators"]) < 3:
        raise ValueError("At least three independent annotators are required")
    policy = protocol["policy"]
    required_false = (
        "model_fit_allowed",
        "optimizer_steps_allowed",
        "architecture_change_allowed",
        "hyperparameter_change_allowed",
        "preprocessing_change_allowed",
        "comparator_refit_allowed",
        "seed_selection_for_main_tables_allowed",
        "omit_unfavorable_metric_allowed",
        "human_labels_prefilled_allowed",
        "external_result_fabrication_allowed",
        "test_documents_allowed_for_case_example_selection",
    )
    if any(policy[name] is not False for name in required_false):
        raise ValueError("A Step-10 scientific firewall was relaxed")
    if policy["delexicalized_text_only"] is not True:
        raise ValueError("The delexicalized-text privacy requirement was relaxed")
    return protocol


def _verify_manifest(base: Path) -> dict[str, Any]:
    manifest_path = base / "generated_artifact_manifest.json"
    manifest = _json(manifest_path)
    body = dict(manifest)
    identity = body.pop("manifest_identity", None)
    if not isinstance(identity, str) or sha256_object(body) != identity:
        raise ValueError(f"Artifact-manifest identity is invalid: {manifest_path}")
    for item in manifest.get("files", []):
        path = base / str(item["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen artifact is missing: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen artifact hash changed: {path}")
        if int(path.stat().st_size) != int(item["size_bytes"]):
            raise ValueError(f"Frozen artifact size changed: {path}")
    return manifest


def _latest_step8(root: Path, protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    expected = protocol["prerequisite"]
    candidates = sorted(
        (root / "outputs/v6/step08_test_confirmation").glob("*/step08_contract.json"),
        reverse=True,
    )
    for path in candidates:
        contract = _json(path)
        if (
            contract.get("status") == expected["step8_status"]
            and contract.get("readiness") == expected["step8_readiness"]
            and contract.get("contract_fingerprint")
            == expected["step8_contract_fingerprint"]
            and sha256_file(path) == expected["step8_contract_sha256"]
            and _verify_fingerprint(contract, "contract_fingerprint")
        ):
            return path.parent, contract
    raise RuntimeError("The exact frozen Step-8 result was not found")


def _step9(root: Path, protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = root / "outputs/v6/step09_fair_comparators/v6_1_frozen_comparison"
    path = base / "step09_contract.json"
    if not path.is_file():
        raise RuntimeError("The frozen Step-9 result was not found")
    contract = _json(path)
    expected = protocol["prerequisite"]
    if (
        contract.get("status") != expected["step9_status"]
        or contract.get("readiness") != expected["step9_readiness"]
        or contract.get("contract_fingerprint")
        != expected["step9_contract_fingerprint"]
        or sha256_file(path) != expected["step9_contract_sha256"]
        or not _verify_fingerprint(contract, "contract_fingerprint")
    ):
        raise ValueError("The Step-9 contract does not match the frozen lock")
    return base, contract


def _audit_prerequisites(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], list[dict[str, Any]]]:
    step8, contract8 = _latest_step8(root, protocol)
    step9, contract9 = _step9(root, protocol)
    _verify_manifest(step8)
    _verify_manifest(step9)
    expected = protocol["prerequisite"]
    checks = [
        ("step8_job_identity", contract8.get("step8_job_identity") == expected["step8_job_identity"]),
        ("step8_completion_identity", contract8.get("test_completion_identity") == expected["step8_completion_identity"]),
        ("step9_evaluation_identity", contract9.get("evaluation_identity") == expected["step9_evaluation_identity"]),
        ("step9_receipt_completion", int(contract9.get("comparator_receipts", -1)) == int(expected["step9_receipts"])),
        ("step9_receipt_requirement", contract9.get("comparator_receipts") == contract9.get("comparator_receipts_required")),
        ("step8_to_step9_lineage", contract9.get("step8_contract_sha256") == sha256_file(step8 / "step08_contract.json")),
        ("no_step8_test_conditioned_change", int(contract8.get("test_conditioned_model_changes", -1)) == 0),
        ("no_step9_model_change", int(contract9.get("v6_model_changes", -1)) == 0),
        ("no_legacy_v4_numbers", int(contract9.get("legacy_v4_numeric_results_used", -1)) == 0),
        ("no_nonprobabilistic_pseudo_nll", int(contract9.get("nonprobabilistic_pseudo_nll_reported", -1)) == 0),
        ("no_target_or_city_comparator_exposure", int(contract9.get("target_or_city_files_exposed_to_comparators", -1)) == 0),
    ]
    rows = [
        {"check": name, "status": "PASS" if passed else "FAIL"}
        for name, passed in checks
    ]
    failed = [row["check"] for row in rows if row["status"] != "PASS"]
    if failed:
        raise ValueError(f"Step-10 prerequisite audit failed: {failed}")
    return step8, contract8, step9, contract9, rows


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _formatted(mean: Any, sd: Any, decimals: int) -> str:
    m = _number(mean)
    s = _number(sd)
    if m is None or s is None:
        return "—"
    if abs(m) < 0.5 * 10 ** (-decimals):
        m = 0.0
    if abs(s) < 0.5 * 10 ** (-decimals):
        s = 0.0
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def _plain(value: Any, decimals: int) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if abs(number) < 0.5 * 10 ** (-decimals):
        number = 0.0
    return f"{number:.{decimals}f}"


def _comparison_rows(
    comparison: pd.DataFrame,
    method_ids: Sequence[str],
) -> list[dict[str, Any]]:
    indexed = comparison.set_index("method_id")
    rows: list[dict[str, Any]] = []
    for method in method_ids:
        if method not in indexed.index:
            raise ValueError(f"Step-9 paper table is missing {method}")
        row = indexed.loc[method]
        family = "Proposed" if method == "v6_1" else (
            "Established" if method in {"lda", "nmf", "gsdmm", "btm", "bertopic", "combinedtm"}
            else "Recent SOTA"
        )
        result: dict[str, Any] = {"Family": family, "Method": row["display_name"]}
        for metric in METRICS:
            decimals = 3 if metric == "topic_diversity_at_10" else 4
            result[DISPLAY_HEADERS[metric]] = _formatted(
                row.get(f"mean_{metric}"), row.get(f"sample_sd_{metric}"), decimals
            )
        rows.append(result)
    return rows


def _single_seed_rows(
    seeds: pd.DataFrame,
    method_ids: Sequence[str],
    selected_seed: int,
) -> list[dict[str, Any]]:
    selected = seeds[seeds["seed"].astype(int).eq(int(selected_seed))].set_index("method_id")
    rows: list[dict[str, Any]] = []
    for method in method_ids:
        if method not in selected.index:
            raise ValueError(f"Step-9 seed table is missing {method}/{selected_seed}")
        row = selected.loc[method]
        family = "Proposed" if method == "v6_1" else (
            "Established" if row["family"] == "established" else "Recent SOTA"
        )
        result: dict[str, Any] = {
            "Family": family,
            "Method": row["display_name"],
            "Seed": int(selected_seed),
        }
        for metric in METRICS:
            decimals = 3 if metric == "topic_diversity_at_10" else 4
            result[DISPLAY_HEADERS[metric]] = _plain(row.get(metric), decimals)
        rows.append(result)
    return rows


def _ablation_rows(ablation: pd.DataFrame, order: Sequence[str]) -> list[dict[str, Any]]:
    indexed = ablation.set_index("variant")
    rows: list[dict[str, Any]] = []
    for variant in order:
        if variant not in indexed.index:
            raise ValueError(f"Step-8 ablation table is missing {variant}")
        row = indexed.loc[variant]
        result: dict[str, Any] = {
            "Configuration": VARIANT_DISPLAY[variant],
            "Contrast type": VARIANT_TYPE[variant],
            "Prespecified target": VARIANT_TARGET[variant],
        }
        for metric in METRICS:
            decimals = 3 if metric == "topic_diversity_at_10" else 4
            result[DISPLAY_HEADERS[metric]] = _formatted(
                row.get(f"{metric}_mean"), row.get(f"{metric}_sample_sd"), decimals
            )
        rows.append(result)
    return rows


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    headers = list(rows[0])
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(map(clean, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(row.get(header, "")) for header in headers) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def _rank_profile(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    frame = comparison.set_index("method_id")
    rank_columns: dict[str, pd.Series] = {}
    for metric in METRICS:
        values = pd.to_numeric(frame[f"mean_{metric}"], errors="coerce")
        rank_columns[metric] = values.rank(
            method="min", ascending=metric in LOWER_IS_BETTER, na_option="bottom"
        )
    rows: list[dict[str, Any]] = []
    for method, source in frame.iterrows():
        ranks = {metric: int(rank_columns[metric].loc[method]) for metric in METRICS}
        available = [
            metric for metric in METRICS
            if _number(source.get(f"mean_{metric}")) is not None
        ]
        rows.append(
            {
                "method_id": method,
                "method": source["display_name"],
                **{f"rank_{metric}": ranks[metric] for metric in METRICS},
                "first_place_metrics": sum(ranks[metric] == 1 for metric in available),
                "top_three_metrics": sum(ranks[metric] <= 3 for metric in available),
                "metrics_ranked": len(available),
                "top_three_on_every_available_metric": all(
                    ranks[metric] <= 3 for metric in available
                ),
            }
        )
    return rows


def _paper_tables(
    out: Path,
    step8: Path,
    step9: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    reporting = protocol["reporting"]
    comparison = pd.read_csv(step9 / "paper_ready_comparison_table.csv", encoding="utf-8-sig")
    seed_metrics = pd.read_csv(step9 / "all_method_seed_metrics.csv", encoding="utf-8-sig")
    ablation = pd.read_csv(step8 / "paper_ready_full_and_ablation_table.csv", encoding="utf-8-sig")
    paired_comparison = pd.read_csv(step9 / "paired_comparator_tests.csv", encoding="utf-8-sig")
    paired_ablation = pd.read_csv(step8 / "worker/ablation_hypothesis_tests.csv", encoding="utf-8-sig")

    established = _comparison_rows(comparison, reporting["main_table_established"])
    sota = _comparison_rows(comparison, reporting["main_table_recent_sota"])
    ablations = _ablation_rows(ablation, reporting["main_table_ablations"])
    selected_seed = int(reporting["descriptive_single_seed"])
    single_established = _single_seed_rows(
        seed_metrics, reporting["main_table_established"], selected_seed
    )
    single_sota = _single_seed_rows(
        seed_metrics, reporting["main_table_recent_sota"], selected_seed
    )

    paper = out / "paper"
    supplement = out / "supplement"
    _write_csv(paper / "main_table_1_established_multiseed.csv", established)
    _write_csv(paper / "main_table_2_recent_sota_multiseed.csv", sota)
    _write_csv(paper / "main_table_3_ablation_multiseed.csv", ablations)
    _write_csv(
        paper / "main_table_3_ablation_targeted_statistics.csv",
        paired_ablation.to_dict(orient="records"),
    )
    _write_csv(
        paper / "paired_comparator_statistics.csv",
        paired_comparison.to_dict(orient="records"),
    )
    _write_csv(
        paper / "method_rank_profile.csv", _rank_profile(comparison)
    )
    _write_csv(
        supplement / f"single_seed_{selected_seed}_established.csv",
        single_established,
    )
    _write_csv(
        supplement / f"single_seed_{selected_seed}_recent_sota.csv",
        single_sota,
    )
    _write_csv(
        supplement / "all_method_seed_metrics.csv",
        seed_metrics.to_dict(orient="records"),
    )

    markdown = [
        "# Frozen SC-HTM V6.1 paper tables",
        "",
        "All main comparison tables report mean ± sample standard deviation over the ten predeclared seeds. The single-seed tables are post-hoc descriptive material for the supplement only and must not replace the multi-seed evidence.",
        "",
        "## Main Table 1 — established baselines",
        "",
        _markdown_table(established),
        "## Main Table 2 — recent SOTA",
        "",
        _markdown_table(sota),
        "## Main Table 3 — paired methodological ablation",
        "",
        _markdown_table(ablations),
        "The graph module is tested on NPMI@10 and C_v@10; pooled backoff and calibration are tested on macro-city NLL/token. Identical topic-quality values after removing predictive-only modules are expected by construction, not evidence that those modules are unnecessary.",
        "",
        "NLL is shown as an em dash for methods without a comparable normalized predictive distribution. TD and redundancy are related top-word diversity summaries and should not be described as independent confirmations.",
    ]
    _write_text(paper / "main_tables.md", "\n".join(markdown))
    return {
        "comparison_methods": int(len(comparison)),
        "comparison_seed_rows": int(len(seed_metrics)),
        "ablation_variants": int(len(ablation)),
        "main_tables": 3,
        "supplement_single_seed": selected_seed,
    }


def _topic_words_for_method(
    method: str,
    seed: int,
    step8: Path,
    step9: Path,
    vocabulary: Sequence[str],
) -> np.ndarray:
    if method == "v6_1":
        words = pd.read_csv(step8 / "worker/top_words_by_seed.csv", encoding="utf-8-sig")
        selected = words[
            words["seed"].astype(int).eq(seed)
            & words["reporting_family"].eq("graph_transported_affinity")
        ].copy()
        selected = selected.sort_values(["topic", "rank"])
        if len(selected) != 100:
            raise ValueError("The proposed frozen seed does not contain 10x10 topic words")
        result = selected["vocabulary_index"].to_numpy(dtype=np.int64).reshape(10, 10)
    else:
        path = step9 / "receipts" / method / f"seed_{seed}" / "canonical_output.npz"
        with np.load(path, allow_pickle=False) as archive:
            result = stable_top_word_indices(np.asarray(archive["topic_word"], dtype=float), 10)
    if result.shape != (10, 10) or np.any(result < 0) or np.any(result >= len(vocabulary)):
        raise ValueError(f"Invalid topic-word evidence for {method}/{seed}")
    return result


def _choose_intruder(
    top: np.ndarray,
    topic: int,
    document_frequency: np.ndarray,
) -> int:
    target = set(map(int, top[topic]))
    target_frequency = float(np.mean(np.log1p(document_frequency[list(target)])))
    candidates: set[int] = set()
    for other in range(top.shape[0]):
        if other != topic:
            candidates.update(map(int, top[other]))
    candidates.difference_update(target)
    if not candidates:
        raise ValueError("No valid word-intrusion candidate exists")
    return min(
        candidates,
        key=lambda index: (
            abs(float(np.log1p(document_frequency[index])) - target_frequency),
            index,
        ),
    )


def _human_packets(
    out: Path,
    step8: Path,
    step9: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    human = protocol["human_evaluation"]
    seed = int(human["blinded_seed"])
    vocabulary_frame = pd.read_csv(step9 / "input_kit/vocabulary.csv", encoding="utf-8-sig").sort_values("index")
    vocabulary = vocabulary_frame["token"].astype(str).tolist()
    counts = sparse.load_npz(step9 / "input_kit/X_train.npz").tocsr()
    document_frequency = np.asarray((counts > 0).sum(axis=0)).reshape(-1)
    if counts.shape != (695, 1148) or len(vocabulary) != 1148:
        raise ValueError("The human packet is not bound to the frozen 695x1148 corpus")

    rng = np.random.default_rng(int(human["randomization_seed"]))
    unblinded: list[dict[str, Any]] = []
    method_topics: dict[str, np.ndarray] = {}
    for method in human["methods"]:
        top = _topic_words_for_method(method, seed, step8, step9, vocabulary)
        method_topics[method] = top
        for topic in range(10):
            intruder = _choose_intruder(top, topic, document_frequency)
            displayed = [int(value) for value in top[topic, :5]] + [intruder]
            displayed = [displayed[index] for index in rng.permutation(len(displayed))]
            unblinded.append(
                {
                    "method_id": method,
                    "source_topic": topic,
                    "seed": seed,
                    "intruder_word": vocabulary[intruder],
                    "intruder_vocabulary_index": intruder,
                    "target_words": "、".join(vocabulary[int(i)] for i in top[topic, :5]),
                    "displayed_words": [vocabulary[index] for index in displayed],
                }
            )
    unblinded = [unblinded[index] for index in rng.permutation(len(unblinded))]
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for number, row in enumerate(unblinded, start=1):
        item_id = f"W{number:03d}"
        words = row.pop("displayed_words")
        public_rows.append(
            {
                "item_id": item_id,
                **{f"word_{index + 1}": word for index, word in enumerate(words)},
                "selected_intruder_word": "",
                "coherence_1_to_5": "",
                "passenger_requirement_yes_no": "",
                "short_topic_label": "",
                "notes": "",
            }
        )
        key_rows.append({"item_id": item_id, **row})

    proposed = method_topics["v6_1"]
    topic_order = rng.permutation(10)
    topic_public: list[dict[str, Any]] = []
    topic_key: list[dict[str, Any]] = []
    for number, topic_value in enumerate(topic_order, start=1):
        topic = int(topic_value)
        item_id = f"T{number:03d}"
        topic_public.append(
            {
                "item_id": item_id,
                "top_10_words": "、".join(vocabulary[int(i)] for i in proposed[topic]),
                "passenger_requirement_yes_no": "",
                "service_category": "",
                "topic_label": "",
                "coherence_1_to_5": "",
                "actionability_1_to_5": "",
                "suggested_transport_action": "",
                "notes": "",
            }
        )
        topic_key.append({"item_id": item_id, "source_topic": topic, "seed": seed})

    packet = out / "human_packet_share_only_this_folder"
    private = out / "private_do_not_share_with_annotators"
    annotators = int(human["required_annotators"])
    for annotator in range(1, annotators + 1):
        _write_csv(packet / f"annotator_{annotator:02d}_word_intrusion.csv", public_rows)
        _write_csv(packet / f"annotator_{annotator:02d}_topic_interpretation.csv", topic_public)
    _write_csv(private / "word_intrusion_answer_key.csv", key_rows)
    _write_csv(private / "topic_interpretation_key.csv", topic_key)
    _write_json(
        private / "blinding_receipt.json",
        {
            "schema_version": 1,
            "seed_rule": human["blinded_seed_rule"],
            "model_seed": seed,
            "randomization_seed": int(human["randomization_seed"]),
            "methods_blinded": list(human["methods"]),
            "items": len(public_rows),
            "answer_key_sha256": sha256_file(private / "word_intrusion_answer_key.csv"),
        },
    )
    categories = ", ".join(human["service_categories"])
    _write_text(
        packet / "README_FOR_ANNOTATORS.md",
        f"""# Independent blinded annotation instructions

Do not open any folder named `private_do_not_share_with_annotators`. Work independently and do not discuss answers with other annotators.

For each word-intrusion item, choose exactly one displayed word that least belongs with the others, rate the remaining topic's coherence from 1 (incoherent) to 5 (very coherent), state whether it expresses a passenger/service requirement (`yes` or `no`), and provide a short neutral label.

For each proposed-topic interpretation item, use one service category from: {categories}. Ratings must be integers 1–5. Do not alter item IDs, rows, or displayed words.

After completion, place the files unchanged except for response columns in `inputs\\v6\\step10\\human`. Keep each annotator's original numbered filename.
""",
    )
    return {
        "word_intrusion_items": len(public_rows),
        "proposed_topic_items": len(topic_public),
        "annotator_forms": annotators * 2,
        "model_seed": seed,
        "method_topics": method_topics,
        "vocabulary": vocabulary,
    }


def _read_documents(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            required = {
                "post_id", "city", "split", "published_date", "year_status",
                "delexicalized_text", "tokens", "model_token_count",
            }
            missing = required - set(value)
            if missing:
                raise ValueError(f"documents.jsonl line {line_number} lacks {sorted(missing)}")
            rows.append(value)
    return rows


def _case_study_packets(
    out: Path,
    root: Path,
    source_root: Path | None,
    step8: Path,
    human_state: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    human = protocol["human_evaluation"]
    cities = tuple(protocol["expected_corpus"]["cities"])
    per_city = pd.read_csv(step8 / "worker/per_city_variant_metrics.csv", encoding="utf-8-sig")
    aggregate_rows: list[dict[str, Any]] = []
    for (variant, city), group in per_city.groupby(["variant", "city"], sort=False):
        values = group["nll_per_token"].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "configuration": VARIANT_DISPLAY[str(variant)],
                "variant": variant,
                "city": city,
                "documents": int(group["documents"].iloc[0]),
                "target_tokens": int(group["tokens"].iloc[0]),
                "mean_nll_per_token": float(values.mean()),
                "sample_sd_nll_per_token": float(values.std(ddof=1)),
                "status": "exploratory_all_cities_not_a_prespecified_hypothesis",
            }
        )
    _write_csv(out / "paper/case_study_all_city_nll.csv", aggregate_rows)

    frame = pd.DataFrame(aggregate_rows)
    full = frame[frame["variant"].eq("full_graph_calibration_pooled")].set_index("city")
    effects: list[dict[str, Any]] = []
    for variant in protocol["reporting"]["main_table_ablations"]:
        if variant == "full_graph_calibration_pooled":
            continue
        selected = frame[frame["variant"].eq(variant)].set_index("city")
        for city in cities:
            effects.append(
                {
                    "city": city,
                    "contrast": f"Full vs {VARIANT_DISPLAY[variant]}",
                    "ablated_variant": variant,
                    "mean_nll_deterioration_when_removed_positive_favors_full": float(
                        selected.loc[city, "mean_nll_per_token"]
                        - full.loc[city, "mean_nll_per_token"]
                    ),
                    "status": "exploratory_descriptive_no_claim_of_significance",
                }
            )
    _write_csv(out / "paper/case_study_city_module_effects.csv", effects)

    candidate_root = source_root
    if candidate_root is None:
        source_text = Path(str(protocol["source_project_root"]))
        candidate_root = source_text if source_text.is_absolute() else root / source_text
    documents_path = candidate_root.expanduser().resolve() / "data/processed/internal/documents.jsonl"
    if not documents_path.is_file():
        _write_json(
            out / "paper/case_study_document_status.json",
            {
                "status": "PENDING_SOURCE_DOCUMENTS",
                "expected_path": str(documents_path),
                "test_documents_selected": 0,
            },
        )
        return {"source_documents_available": False, "case_items": 0}

    documents = _read_documents(documents_path)
    expected = int(protocol["expected_corpus"]["modeled_documents"])
    if len(documents) != expected:
        raise ValueError(f"Expected {expected} modeled documents, found {len(documents)}")
    development = [row for row in documents if row["split"] in {"train", "validation"}]
    if len(development) != int(protocol["expected_corpus"]["fit_documents"]):
        raise ValueError("Case-study development rows do not match the 695-row freeze")
    if any(row["split"] == "test" for row in development):
        raise AssertionError("A test document entered case selection")

    proposed = np.asarray(human_state["method_topics"]["v6_1"], dtype=np.int64)
    vocabulary = list(human_state["vocabulary"])
    candidates: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for topic in range(10):
        weighted = {
            vocabulary[int(index)]: float(10 - rank)
            for rank, index in enumerate(proposed[topic])
        }
        for city in cities:
            scored: list[tuple[float, int, str, dict[str, Any]]] = []
            for row in development:
                if str(row["city"]) != city:
                    continue
                counts = Counter(map(str, row["tokens"]))
                raw_score = sum(counts[token] * weight for token, weight in weighted.items())
                if raw_score <= 0:
                    continue
                length = max(1, int(row["model_token_count"]))
                score = raw_score / math.sqrt(length)
                overlap = sum(counts[token] > 0 for token in weighted)
                tie = hashlib.sha256(f"{topic}:{row['post_id']}".encode("utf-8")).hexdigest()
                scored.append((score, overlap, tie, row))
            if not scored:
                missing.append({"topic": topic, "city": city, "reason": "no_positive_top_word_overlap"})
                continue
            score, overlap, _, row = max(scored, key=lambda value: (value[0], value[1], value[2]))
            candidates.append(
                {
                    "source_topic": topic,
                    "city": city,
                    "published_date": str(row["published_date"]),
                    "year_status": str(row["year_status"]),
                    "top_10_words": "、".join(weighted),
                    "lexical_relevance_score": float(score),
                    "distinct_top_word_matches": int(overlap),
                    "delexicalized_text": str(row["delexicalized_text"]).replace("\r", " ").replace("\n", " ").strip(),
                    "source_id_sha256": hashlib.sha256(str(row["post_id"]).encode("utf-8")).hexdigest(),
                    "selection_partition": str(row["split"]),
                    "selection_rule": "highest_fixed_seed101_weighted_top10_overlap_within_topic_and_city_development_only",
                }
            )
    _write_csv(out / "supplement/case_study_candidate_audit.csv", candidates)
    _write_csv(out / "supplement/case_study_missing_cells.csv", missing)

    rng = np.random.default_rng(int(human["randomization_seed"]) + 1)
    candidates = [candidates[index] for index in rng.permutation(len(candidates))]
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for number, row in enumerate(candidates, start=1):
        item_id = f"C{number:03d}"
        public_rows.append(
            {
                "item_id": item_id,
                "city": row["city"],
                "topic_words": row["top_10_words"],
                "delexicalized_passenger_post": row["delexicalized_text"],
                "topic_fit_1_to_5": "",
                "service_need_present_yes_no": "",
                "case_requirement_label": "",
                "recommended_transport_action": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "item_id": item_id,
                **{key: value for key, value in row.items() if key not in {"delexicalized_text"}},
            }
        )
    packet = out / "human_packet_share_only_this_folder"
    annotators = int(human["required_annotators"])
    for annotator in range(1, annotators + 1):
        _write_csv(packet / f"annotator_{annotator:02d}_case_posts.csv", public_rows)
    _write_csv(out / "private_do_not_share_with_annotators/case_source_key.csv", key_rows)

    temporal_rows: list[dict[str, Any]] = []
    for row in documents:
        date = str(row["published_date"])
        year = date[:4] if len(date) >= 4 and date[:4].isdigit() else "unknown"
        temporal_rows.append(
            {"city": row["city"], "split": row["split"], "year": year, "year_status": row["year_status"]}
        )
    temporal = (
        pd.DataFrame(temporal_rows)
        .groupby(["city", "split", "year", "year_status"], dropna=False)
        .size()
        .reset_index(name="documents")
    )
    temporal["interpretation"] = "descriptive_coverage_only_not_temporal_generalization"
    _write_csv(out / "supplement/temporal_coverage_descriptive.csv", temporal.to_dict(orient="records"))
    return {
        "source_documents_available": True,
        "modeled_documents": len(documents),
        "development_documents_used_for_selection": len(development),
        "test_documents_used_for_selection": 0,
        "case_items": len(public_rows),
        "missing_topic_city_cells": len(missing),
        "source_documents_sha256": sha256_file(documents_path),
    }


def _coincidence_alpha(
    units: Mapping[str, Sequence[Any]],
    *,
    ordered_categories: Sequence[Any] | None = None,
) -> float | None:
    usable = [list(values) for values in units.values() if len(values) >= 2]
    if not usable:
        return None
    categories = (
        list(ordered_categories)
        if ordered_categories is not None
        else sorted({value for values in usable for value in values}, key=str)
    )
    index = {value: position for position, value in enumerate(categories)}
    coincidence = np.zeros((len(categories), len(categories)), dtype=float)
    for values in usable:
        counts = Counter(values)
        total = len(values)
        for left, left_count in counts.items():
            for right, right_count in counts.items():
                coincidence[index[left], index[right]] += (
                    left_count * (right_count - int(left == right)) / (total - 1)
                )
    marginals = coincidence.sum(axis=1)
    total = float(marginals.sum())
    if total <= 1:
        return None
    distance = np.zeros_like(coincidence)
    for left in range(len(categories)):
        for right in range(len(categories)):
            if ordered_categories is None:
                distance[left, right] = float(left != right)
            elif left != right:
                lo, hi = sorted((left, right))
                span = marginals[lo : hi + 1].sum() - 0.5 * (marginals[lo] + marginals[hi])
                distance[left, right] = float(span * span)
    observed = float((coincidence * distance).sum() / total)
    expected = float((np.outer(marginals, marginals) * distance).sum() / (total * (total - 1)))
    if expected <= 0:
        # Alpha is undefined when expected disagreement is zero.  Perfect
        # observed agreement alone must not be reported as alpha=1 when the
        # coefficient has a zero denominator.
        return None
    return float(1.0 - observed / expected)


def _normalize_yes_no(value: Any) -> str:
    text = str(value).strip().casefold()
    if text in {"yes", "y", "1", "true"}:
        return "yes"
    if text in {"no", "n", "0", "false"}:
        return "no"
    raise ValueError(f"Expected yes/no, found {value!r}")


def _rating(value: Any) -> int:
    number = int(float(value))
    if number not in {1, 2, 3, 4, 5} or float(value) != number:
        raise ValueError(f"Expected an integer rating from 1 to 5, found {value!r}")
    return number


def _human_returns(
    root: Path,
    out: Path,
    protocol: Mapping[str, Any],
    case_state: Mapping[str, Any],
) -> dict[str, Any]:
    required = int(protocol["human_evaluation"]["required_annotators"])
    input_dir = root / "inputs/v6/step10/human"
    suffixes = ["word_intrusion", "topic_interpretation"]
    if int(case_state.get("case_items", 0)) > 0:
        suffixes.append("case_posts")
    expected = [
        input_dir / f"annotator_{annotator:02d}_{suffix}.csv"
        for annotator in range(1, required + 1)
        for suffix in suffixes
    ]
    missing = [str(path.relative_to(root)) for path in expected if not path.is_file()]
    if missing:
        return {
            "status": "PENDING_INDEPENDENT_ANNOTATION",
            "complete": False,
            "required_files": len(expected),
            "missing_files": missing,
        }

    return_receipts = [
        {
            "relative_path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in expected
    ]
    _write_csv(out / "supplement/human_return_file_receipts.csv", return_receipts)

    key = pd.read_csv(
        out / "private_do_not_share_with_annotators/word_intrusion_answer_key.csv",
        encoding="utf-8-sig",
    ).set_index("item_id")
    topic_key = pd.read_csv(
        out / "private_do_not_share_with_annotators/topic_interpretation_key.csv",
        encoding="utf-8-sig",
    ).set_index("item_id")
    intrusion_rows: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    categories = set(protocol["human_evaluation"]["service_categories"])
    for annotator in range(1, required + 1):
        intrusion = pd.read_csv(
            input_dir / f"annotator_{annotator:02d}_word_intrusion.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        issued_intrusion = pd.read_csv(
            out / "human_packet_share_only_this_folder" / f"annotator_{annotator:02d}_word_intrusion.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        _assert_locked_columns(
            intrusion,
            issued_intrusion,
            locked_columns=[f"word_{position}" for position in range(1, 7)],
            label=f"annotator {annotator} word intrusion",
        )
        if set(intrusion["item_id"]) != set(key.index) or intrusion["item_id"].duplicated().any():
            raise ValueError(f"Annotator {annotator} changed word-intrusion item IDs")
        for row in intrusion.to_dict(orient="records"):
            item = str(row["item_id"])
            displayed = {str(row[f"word_{position}"]) for position in range(1, 7)}
            selected = str(row["selected_intruder_word"]).strip()
            if selected not in displayed:
                raise ValueError(f"{item}: selected intruder is not a displayed word")
            passenger = _normalize_yes_no(row["passenger_requirement_yes_no"])
            label = str(row["short_topic_label"]).strip()
            if not label:
                raise ValueError(f"{item}: short topic label is blank")
            intrusion_rows.append(
                {
                    "annotator": annotator,
                    "item_id": item,
                    "method_id": key.loc[item, "method_id"],
                    "source_topic": int(key.loc[item, "source_topic"]),
                    "correct_intruder": int(selected == str(key.loc[item, "intruder_word"])),
                    "coherence_1_to_5": _rating(row["coherence_1_to_5"]),
                    "passenger_requirement_yes_no": passenger,
                    "short_topic_label": label,
                }
            )
        topics = pd.read_csv(
            input_dir / f"annotator_{annotator:02d}_topic_interpretation.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        issued_topics = pd.read_csv(
            out / "human_packet_share_only_this_folder" / f"annotator_{annotator:02d}_topic_interpretation.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        _assert_locked_columns(
            topics,
            issued_topics,
            locked_columns=["top_10_words"],
            label=f"annotator {annotator} topic interpretation",
        )
        if set(topics["item_id"]) != set(topic_key.index) or topics["item_id"].duplicated().any():
            raise ValueError(f"Annotator {annotator} changed topic-interpretation item IDs")
        for row in topics.to_dict(orient="records"):
            item = str(row["item_id"])
            category = str(row["service_category"]).strip()
            if category not in categories:
                raise ValueError(f"{item}: service category is not in the frozen taxonomy")
            label = str(row["topic_label"]).strip()
            action = str(row["suggested_transport_action"]).strip()
            if not label or not action:
                raise ValueError(f"{item}: label/action is blank")
            topic_rows.append(
                {
                    "annotator": annotator,
                    "item_id": item,
                    "source_topic": int(topic_key.loc[item, "source_topic"]),
                    "passenger_requirement_yes_no": _normalize_yes_no(row["passenger_requirement_yes_no"]),
                    "service_category": category,
                    "topic_label": label,
                    "coherence_1_to_5": _rating(row["coherence_1_to_5"]),
                    "actionability_1_to_5": _rating(row["actionability_1_to_5"]),
                    "suggested_transport_action": action,
                }
            )
        if "case_posts" in suffixes:
            case_path = input_dir / f"annotator_{annotator:02d}_case_posts.csv"
            case = pd.read_csv(case_path, encoding="utf-8-sig", keep_default_na=False)
            expected_cases = pd.read_csv(
                out / "human_packet_share_only_this_folder" / f"annotator_{annotator:02d}_case_posts.csv",
                encoding="utf-8-sig",
                keep_default_na=False,
            )
            _assert_locked_columns(
                case,
                expected_cases,
                locked_columns=["city", "topic_words", "delexicalized_passenger_post"],
                label=f"annotator {annotator} case study",
            )
            for row in case.to_dict(orient="records"):
                label = str(row["case_requirement_label"]).strip()
                action = str(row["recommended_transport_action"]).strip()
                if not label or not action:
                    raise ValueError(f"{row['item_id']}: case label/action is blank")
                case_rows.append(
                    {
                        "annotator": annotator,
                        "item_id": str(row["item_id"]),
                        "city": str(row["city"]),
                        "topic_fit_1_to_5": _rating(row["topic_fit_1_to_5"]),
                        "service_need_present_yes_no": _normalize_yes_no(row["service_need_present_yes_no"]),
                        "case_requirement_label": label,
                        "recommended_transport_action": action,
                    }
                )

    intrusion_frame = pd.DataFrame(intrusion_rows)
    topic_frame = pd.DataFrame(topic_rows)
    case_frame = pd.DataFrame(case_rows)
    method_summary = (
        intrusion_frame.groupby("method_id", sort=False)
        .agg(
            ratings=("correct_intruder", "size"),
            word_intrusion_accuracy=("correct_intruder", "mean"),
            mean_coherence=("coherence_1_to_5", "mean"),
            passenger_requirement_present_judgment_rate=(
                "passenger_requirement_yes_no",
                lambda values: float(np.mean(np.asarray(values) == "yes")),
            ),
        )
        .reset_index()
    )
    _write_csv(out / "paper/human_method_summary.csv", method_summary.to_dict(orient="records"))
    _write_csv(out / "supplement/human_word_intrusion_long.csv", intrusion_rows)
    _write_csv(out / "supplement/human_topic_interpretation_long.csv", topic_rows)
    if case_rows:
        _write_csv(out / "supplement/human_case_study_long.csv", case_rows)

    def units(frame: pd.DataFrame, column: str) -> dict[str, list[Any]]:
        return {
            str(item): list(group[column])
            for item, group in frame.groupby("item_id", sort=False)
        }

    reliability = [
        {
            "instrument": "word_intrusion",
            "outcome": "coherence_1_to_5",
            "coefficient": "krippendorff_alpha_ordinal",
            "value": _coincidence_alpha(units(intrusion_frame, "coherence_1_to_5"), ordered_categories=[1, 2, 3, 4, 5]),
        },
        {
            "instrument": "word_intrusion",
            "outcome": "passenger_requirement_yes_no",
            "coefficient": "krippendorff_alpha_nominal",
            "value": _coincidence_alpha(units(intrusion_frame, "passenger_requirement_yes_no")),
        },
        {
            "instrument": "topic_interpretation",
            "outcome": "service_category",
            "coefficient": "krippendorff_alpha_nominal",
            "value": _coincidence_alpha(units(topic_frame, "service_category")),
        },
        {
            "instrument": "topic_interpretation",
            "outcome": "actionability_1_to_5",
            "coefficient": "krippendorff_alpha_ordinal",
            "value": _coincidence_alpha(units(topic_frame, "actionability_1_to_5"), ordered_categories=[1, 2, 3, 4, 5]),
        },
    ]
    if case_rows:
        reliability.extend(
            [
                {
                    "instrument": "case_posts",
                    "outcome": "topic_fit_1_to_5",
                    "coefficient": "krippendorff_alpha_ordinal",
                    "value": _coincidence_alpha(units(case_frame, "topic_fit_1_to_5"), ordered_categories=[1, 2, 3, 4, 5]),
                },
                {
                    "instrument": "case_posts",
                    "outcome": "service_need_present_yes_no",
                    "coefficient": "krippendorff_alpha_nominal",
                    "value": _coincidence_alpha(units(case_frame, "service_need_present_yes_no")),
                },
            ]
        )
    for row in reliability:
        if row["value"] is None:
            row["estimability"] = "not_estimable"
            row["note"] = "expected disagreement is zero because ratings have no variation"
        else:
            row["estimability"] = "estimated"
            row["note"] = ""
    _write_csv(out / "paper/human_interrater_reliability.csv", reliability)
    return {
        "status": "COMPLETE_REPORTED_AS_OBSERVED",
        "complete": True,
        "annotators": required,
        "word_intrusion_ratings": len(intrusion_rows),
        "topic_interpretation_ratings": len(topic_rows),
        "case_ratings": len(case_rows),
        "sealed_response_files": len(return_receipts),
        "sealed_response_set_identity": sha256_object(return_receipts),
        "locked_fields_verified_unchanged": True,
        "no_performance_threshold_used_as_completion_gate": True,
    }


def _external_packet(root: Path, out: Path, contract8: Mapping[str, Any]) -> dict[str, Any]:
    packet = out / "external_evaluation_packet"
    _write_text(
        packet / "README.md",
        """# External/temporal evaluation requirement

Step 10 does not accept hand-entered metric rows as external evidence. Supply a genuinely independent or later-time corpus first. For a cross-language corpus, the evaluator must separate (a) exact-state input-compatibility assessment from (b) architecture-level replication with model and training hyperparameters frozen before outcomes. It must declare coverage, ethics, privacy and licensing; prohibit test-conditioned changes; fit all prespecified comparators; and emit a hash-bound contract plus artifact manifest.

Place the resulting `external_evaluation_contract.json` and `generated_artifact_manifest.json` under `inputs\\v6\\step10\\external`. The generic Step-10 stage validates that envelope; it cannot invent a valid evaluator before the external dataset and schema are known.
""",
    )
    template = {
        "schema_version": 2,
        "status": "PASS",
        "evidence_kind": "independent_external_architecture_replication_with_frozen_hyperparameters",
        "dataset_ids": ["REPLACE_WITH_DATASET_ID"],
        "dataset_independent_of_internal_772_document_corpus": True,
        "external_language": "REPLACE",
        "internal_language": "Chinese",
        "step7_freeze_identity": contract8["step7_freeze_identity"],
        "exact_frozen_state_assessment": "NOT_ESTIMABLE_LANGUAGE_VOCABULARY_MISMATCH",
        "exact_state_coverage_fraction": 0.0,
        "exact_state_performance_claimed": False,
        "architecture_unchanged": True,
        "model_and_training_hyperparameters_frozen_before_external_outcomes": True,
        "language_compatible_fit_vocabulary_and_embeddings_reestimated": True,
        "v6_model_fits": 10,
        "v6_optimizer_update_steps": "REPORT_POSITIVE_INTEGER",
        "v6_architecture_or_hyperparameter_changes": 0,
        "test_conditioned_preprocessing_changes": 0,
        "ten_predeclared_seeds_reported": True,
        "all_prespecified_methods_reported": True,
        "unfavorable_results_omitted": False,
        "ethics_privacy_license_documented": True,
        "raw_coordinates_exported": False,
        "raw_text_exported": False,
        "ground_truth_topic_labels_available": False,
        "label_based_metrics_reported": False,
        "artifact_manifest_identity": "REPLACE_WITH_MANIFEST_IDENTITY",
        "contract_fingerprint": "COMPUTE_WITH_contract_fingerprint_REMOVED",
    }
    _write_json(packet / "external_evaluation_contract_template.json", template)
    input_dir = root / "inputs/v6/step10/external"
    contract_path = input_dir / "external_evaluation_contract.json"
    manifest_path = input_dir / "generated_artifact_manifest.json"
    if not contract_path.is_file() or not manifest_path.is_file():
        return {
            "status": "PENDING_DATASET_SPECIFIC_EXTERNAL_EVALUATION",
            "complete": False,
            "missing": [
                str(path.relative_to(root))
                for path in (contract_path, manifest_path)
                if not path.is_file()
            ],
        }
    contract = _json(contract_path)
    manifest = _json(manifest_path)
    manifest_body = dict(manifest)
    manifest_identity = manifest_body.pop("manifest_identity", None)
    if sha256_object(manifest_body) != manifest_identity:
        raise ValueError("External artifact-manifest identity is invalid")
    if not _verify_fingerprint(contract, "contract_fingerprint"):
        raise ValueError("External evaluation contract fingerprint is invalid")
    requirements = {
        "schema": int(contract.get("schema_version", -1)) == 2,
        "status": contract.get("status") == "PASS",
        "evidence_kind": contract.get("evidence_kind")
        == "independent_external_architecture_replication_with_frozen_hyperparameters",
        "independent": contract.get("dataset_independent_of_internal_772_document_corpus") is True,
        "freeze": contract.get("step7_freeze_identity") == contract8["step7_freeze_identity"],
        "cross_language_declared": bool(str(contract.get("external_language", "")).strip())
        and contract.get("internal_language") == "Chinese",
        "exact_state_assessed": contract.get("exact_frozen_state_assessment")
        == "NOT_ESTIMABLE_LANGUAGE_VOCABULARY_MISMATCH",
        "exact_state_not_misreported": contract.get("exact_state_performance_claimed") is False,
        "architecture_unchanged": contract.get("architecture_unchanged") is True,
        "hyperparameters_frozen": contract.get(
            "model_and_training_hyperparameters_frozen_before_external_outcomes"
        ) is True,
        "language_input_layer_declared": contract.get(
            "language_compatible_fit_vocabulary_and_embeddings_reestimated"
        ) is True,
        "fits": int(contract.get("v6_model_fits", -1)) == 10,
        "optimizer": int(contract.get("v6_optimizer_update_steps", -1)) > 0,
        "model_change": int(contract.get("v6_architecture_or_hyperparameter_changes", -1)) == 0,
        "preprocessing_change": int(contract.get("test_conditioned_preprocessing_changes", -1)) == 0,
        "seeds": contract.get("ten_predeclared_seeds_reported") is True,
        "methods": contract.get("all_prespecified_methods_reported") is True,
        "complete_reporting": contract.get("unfavorable_results_omitted") is False,
        "ethics": contract.get("ethics_privacy_license_documented") is True,
        "privacy": contract.get("raw_coordinates_exported") is False
        and contract.get("raw_text_exported") is False,
        "labels_honest": contract.get("ground_truth_topic_labels_available") is False
        and contract.get("label_based_metrics_reported") is False,
        "manifest_link": contract.get("artifact_manifest_identity") == manifest_identity,
    }
    failed = [name for name, passed in requirements.items() if not passed]
    if failed:
        raise ValueError(f"External evaluation contract failed: {failed}")
    for item in manifest.get("files", []):
        path = input_dir / str(item["relative_path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"External evidence file missing or changed: {path}")
    return {
        "status": "COMPLETE_REPORTED_AS_OBSERVED",
        "complete": True,
        "dataset_ids": contract.get("dataset_ids", []),
        "contract_sha256": sha256_file(contract_path),
        "manifest_identity": manifest_identity,
    }


def _manifest(out: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(
        item for item in out.rglob("*")
        if item.is_file() and item.name != "generated_artifact_manifest.json"
    ):
        files.append(
            {
                "relative_path": path.relative_to(out).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    body: dict[str, Any] = {"schema_version": 1, "files": files}
    body["manifest_identity"] = sha256_object(body)
    _write_json(out / "generated_artifact_manifest.json", body)
    return body


def _return_zip(root: Path, out: Path) -> Path:
    path = root / "v6_step10_results.zip"
    if path.is_file():
        path.unlink()
    prefix = Path("step10_reporting_validity") / out.name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(candidate for candidate in out.rglob("*") if candidate.is_file()):
            archive.write(item, (prefix / item.relative_to(out)).as_posix())
    return path


def run_v6_step10(
    *,
    project_root: Path,
    config_path: Path,
    source_root: Path | None = None,
) -> Path:
    """Run the non-mutating Step-10 preparation/finalization gate."""

    root = project_root.expanduser().resolve()
    protocol = _load_protocol(config_path)
    print(
        f"[V6.1 Step 10] START | implementation={IMPLEMENTATION} | "
        "fits=0 optimizer-steps=0 frozen-result-changes=0",
        flush=True,
    )
    step8, contract8, step9, contract9, hard_checks = _audit_prerequisites(root, protocol)
    out = root / "outputs/v6/step10_reporting_validity" / _utc_id()
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "hard_checks.csv", hard_checks)
    table_state = _paper_tables(out, step8, step9, protocol)
    human_packet_state = _human_packets(out, step8, step9, protocol)
    case_state = _case_study_packets(
        out, root, source_root, step8, human_packet_state, protocol
    )
    human_state = _human_returns(root, out, protocol, case_state)
    external_state = _external_packet(root, out, contract8)

    source_handoff = root / "docs/SC_HTM_V6_COMPLETE_HANDOFF_STEPS01_10.md"
    if source_handoff.is_file():
        _write_text(
            out / "paper/SC_HTM_V6_COMPLETE_HANDOFF_STEPS01_10.md",
            source_handoff.read_text(encoding="utf-8-sig"),
        )
    complete = bool(human_state["complete"] and external_state["complete"])
    status = "PASS" if complete else "PREPARATION_PASS"
    if complete:
        readiness = FINAL_READINESS
    elif human_state["complete"]:
        readiness = PENDING_EXTERNAL_READINESS
    elif external_state["complete"]:
        readiness = PENDING_HUMAN_READINESS
    else:
        readiness = PENDING_BOTH_READINESS
    completion_rows = [
        {
            "gate": "frozen_step8_step9_integrity",
            "status": "COMPLETE",
            "required_for_submission_readiness": True,
        },
        {
            "gate": "paper_tables",
            "status": "COMPLETE",
            "required_for_submission_readiness": True,
        },
        {
            "gate": "delexicalized_case_packet",
            "status": "COMPLETE" if case_state["source_documents_available"] else "PENDING",
            "required_for_submission_readiness": True,
        },
        {
            "gate": "independent_blinded_human_evaluation",
            "status": "COMPLETE" if human_state["complete"] else "PENDING",
            "required_for_submission_readiness": True,
        },
        {
            "gate": "independent_external_or_prospective_temporal_evaluation",
            "status": "COMPLETE" if external_state["complete"] else "PENDING",
            "required_for_submission_readiness": bool(
                protocol["external_evaluation"]["required_for_submission_readiness"]
            ),
        },
    ]
    _write_csv(out / "completion_gates.csv", completion_rows)
    lineage = {
        "step8_contract_path": str((step8 / "step08_contract.json").resolve()),
        "step8_contract_sha256": sha256_file(step8 / "step08_contract.json"),
        "step8_contract_fingerprint": contract8["contract_fingerprint"],
        "step9_contract_path": str((step9 / "step09_contract.json").resolve()),
        "step9_contract_sha256": sha256_file(step9 / "step09_contract.json"),
        "step9_contract_fingerprint": contract9["contract_fingerprint"],
        "step7_freeze_identity": contract8["step7_freeze_identity"],
        "model_or_hyperparameter_changes": 0,
        "comparator_refits": 0,
    }
    _write_json(out / "frozen_lineage.json", lineage)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "scope": SCOPE,
        "status": status,
        "readiness": readiness,
        "hard_checks_passed": len(hard_checks),
        "hard_checks_total": len(hard_checks),
        "paper_outputs": table_state,
        "human_packet": {
            key: value for key, value in human_packet_state.items()
            if key not in {"method_topics", "vocabulary"}
        },
        "case_study": case_state,
        "human_evaluation": human_state,
        "external_evaluation": external_state,
        "completion_gates": completion_rows,
        "model_fits": 0,
        "optimizer_steps": 0,
        "model_or_hyperparameter_changes": 0,
        "comparator_refits": 0,
        "unfavorable_results_omitted": False,
        "single_seed_used_in_main_tables": False,
        "test_documents_used_for_case_selection": 0,
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    _write_json(out / "step10_contract.json", contract)
    _manifest(out)
    zip_path = _return_zip(root, out)
    print(
        f"[V6.1 Step 10] {status} | hard checks={len(hard_checks)}/{len(hard_checks)} | "
        f"human={human_state['status']} | external={external_state['status']}",
        flush=True,
    )
    print(f"[V6.1 Step 10] contract={out / 'step10_contract.json'}", flush=True)
    print(f"[V6.1 Step 10] return-zip={zip_path}", flush=True)
    if not complete:
        pending = []
        if not human_state["complete"]:
            pending.append("independent human evidence")
        if not external_state["complete"]:
            pending.append("independent external evidence")
        print(
            "[V6.1 Step 10] Preparation is valid, but manuscript-submission "
            "readiness is intentionally NOT claimed while "
            + " and ".join(pending)
            + " remain pending.",
            flush=True,
        )
    return out
