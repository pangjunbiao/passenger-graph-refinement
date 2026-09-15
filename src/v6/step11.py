"""V6.1 Step 11 R2: distinct evidence from immutable Step-7--10 artifacts.

This stage is deliberately reporting-only.  It verifies the exact frozen
evaluation lineage, then answers questions not already answered by the
baseline/SOTA/ablation tables: seed stability, graph-induced lexical change,
optimization convergence, and the preregistered human case-study chain.  It
never replots the comparison table, fits a model, executes an optimizer,
selects a seed, changes a frozen result, reads test text, or invents human or
external evidence.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from src.v6.data_contract import sha256_file, sha256_object


IMPLEMENTATION = "v6_1_step11_distinct_evidence_synthesis_r2"
SCOPE = "frozen_distinct_mechanism_stability_case_preregistration_no_training"
READINESS = "READY_FOR_INDEPENDENT_HUMAN_AND_EXTERNAL_VALIDATION"

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
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(materialized)


def _verify_fingerprint(value: Mapping[str, Any], key: str) -> bool:
    body = dict(value)
    expected = body.pop(key, None)
    return isinstance(expected, str) and sha256_object(body) == expected


def _load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    protocol = document.get("v6", {}).get("step11", {})
    if document.get("schema_version") != 1:
        raise ValueError("Step-11 configuration schema changed")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-11 implementation/configuration lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-11 scope changed")
    reporting = protocol["reporting"]
    seeds = list(map(int, reporting["seeds"]))
    if seeds != [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]:
        raise ValueError("The ten frozen reporting seeds changed")
    if int(reporting["reference_seed"]) != seeds[0]:
        raise ValueError("Stability reference must be the first prespecified seed")
    if int(reporting["topics"]) != 10 or int(reporting["top_words"]) != 10:
        raise ValueError("The frozen topic or top-word count changed")
    p = float(reporting["rank_biased_overlap_p"])
    if not 0.0 < p < 1.0:
        raise ValueError("RBO persistence must lie strictly between zero and one")
    policy = protocol["policy"]
    false_locks = (
        "model_fit_allowed",
        "optimizer_steps_allowed",
        "architecture_change_allowed",
        "hyperparameter_change_allowed",
        "preprocessing_change_allowed",
        "comparator_refit_allowed",
        "frozen_result_change_allowed",
        "seed_selection_allowed",
        "unfavorable_result_omission_allowed",
        "human_response_invention_allowed",
        "external_result_invention_allowed",
        "test_text_access_allowed",
        "performance_table_replot_allowed",
    )
    if any(policy[name] is not False for name in false_locks):
        raise ValueError("A Step-11 scientific firewall was relaxed")
    if policy["descriptive_posthoc_outputs_must_be_labeled"] is not True:
        raise ValueError("The post-hoc disclosure requirement was relaxed")
    return protocol


def _verify_manifest(base: Path) -> dict[str, Any]:
    path = base / "generated_artifact_manifest.json"
    manifest = _json(path)
    body = dict(manifest)
    identity = body.pop("manifest_identity", None)
    if not isinstance(identity, str) or sha256_object(body) != identity:
        raise ValueError(f"Artifact-manifest identity is invalid: {path}")
    for item in manifest.get("files", []):
        artifact = base / str(item["relative_path"])
        if not artifact.is_file():
            raise FileNotFoundError(f"Frozen artifact is missing: {artifact}")
        if sha256_file(artifact) != item["sha256"]:
            raise ValueError(f"Frozen artifact hash changed: {artifact}")
        if int(artifact.stat().st_size) != int(item["size_bytes"]):
            raise ValueError(f"Frozen artifact size changed: {artifact}")
    return manifest


def _frozen_sources(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    expected = protocol["prerequisite"]
    step8_candidates = sorted(
        (root / "outputs/v6/step08_test_confirmation").glob("*/step08_contract.json"),
        reverse=True,
    )
    step8_path: Path | None = None
    contract8: dict[str, Any] | None = None
    for path in step8_candidates:
        value = _json(path)
        if (
            value.get("status") == expected["step8_status"]
            and value.get("readiness") == expected["step8_readiness"]
            and value.get("contract_fingerprint")
            == expected["step8_contract_fingerprint"]
            and sha256_file(path) == expected["step8_contract_sha256"]
            and _verify_fingerprint(value, "contract_fingerprint")
        ):
            step8_path, contract8 = path.parent, value
            break
    if step8_path is None or contract8 is None:
        raise RuntimeError("The exact locked Step-8 result was not found")
    _verify_manifest(step8_path)

    step9_path = root / "outputs/v6/step09_fair_comparators/v6_1_frozen_comparison"
    contract9_path = step9_path / "step09_contract.json"
    if not contract9_path.is_file():
        raise RuntimeError("The exact locked Step-9 result was not found")
    contract9 = _json(contract9_path)
    if not (
        contract9.get("status") == expected["step9_status"]
        and contract9.get("readiness") == expected["step9_readiness"]
        and contract9.get("contract_fingerprint")
        == expected["step9_contract_fingerprint"]
        and sha256_file(contract9_path) == expected["step9_contract_sha256"]
        and _verify_fingerprint(contract9, "contract_fingerprint")
    ):
        raise ValueError("The Step-9 result does not match the frozen lock")
    _verify_manifest(step9_path)

    step10_candidates = sorted(
        (root / "outputs/v6/step10_reporting_validity").glob("*/step10_contract.json"),
        reverse=True,
    )
    step10_path: Path | None = None
    contract10: dict[str, Any] | None = None
    for path in step10_candidates:
        value = _json(path)
        case = value.get("case_study", {})
        if (
            value.get("implementation_version") == expected["step10_implementation"]
            and value.get("status") in expected["allowed_step10_status"]
            and _verify_fingerprint(value, "contract_fingerprint")
            and int(value.get("hard_checks_passed", -1))
            == int(value.get("hard_checks_total", -2))
            and int(value.get("hard_checks_passed", 0))
            >= int(expected["step10_minimum_hard_checks"])
            and int(value.get("model_fits", -1)) == 0
            and int(value.get("optimizer_steps", -1)) == 0
            and int(value.get("model_or_hyperparameter_changes", -1)) == 0
            and int(value.get("comparator_refits", -1)) == 0
            and value.get("unfavorable_results_omitted") is False
            and value.get("single_seed_used_in_main_tables") is False
            and case.get("source_documents_available") is True
            and int(case.get("case_items", -1))
            == int(expected["step10_required_case_items"])
            and int(case.get("missing_topic_city_cells", -1)) == 0
            and int(case.get("test_documents_used_for_selection", -1)) == 0
        ):
            try:
                _verify_manifest(path.parent)
            except (ValueError, FileNotFoundError):
                continue
            step10_path, contract10 = path.parent, value
            break
    if step10_path is None or contract10 is None:
        raise RuntimeError(
            "No manifest-valid Step-10 result with the complete 30-cell case packet was found"
        )
    lineage = _json(step10_path / "frozen_lineage.json")
    if (
        lineage.get("step8_contract_sha256") != expected["step8_contract_sha256"]
        or lineage.get("step8_contract_fingerprint")
        != expected["step8_contract_fingerprint"]
        or lineage.get("step9_contract_sha256") != expected["step9_contract_sha256"]
        or lineage.get("step9_contract_fingerprint")
        != expected["step9_contract_fingerprint"]
        or int(lineage.get("model_or_hyperparameter_changes", -1)) != 0
        or int(lineage.get("comparator_refits", -1)) != 0
    ):
        raise ValueError("Step-10 lineage is not the exact Step-8/9 frozen lineage")
    return step8_path, contract8, step9_path, contract9, step10_path, contract10


def _rbo(left: Sequence[int], right: Sequence[int], persistence: float) -> float:
    """Finite extrapolated rank-biased overlap for equal nonempty lists."""

    if len(left) != len(right) or not left:
        raise ValueError("RBO requires equal, nonempty ranked lists")
    seen_left: set[int] = set()
    seen_right: set[int] = set()
    weighted = 0.0
    agreement = 0.0
    for depth, (item_left, item_right) in enumerate(zip(left, right), start=1):
        seen_left.add(int(item_left))
        seen_right.add(int(item_right))
        agreement = len(seen_left & seen_right) / depth
        weighted += agreement * persistence ** (depth - 1)
    value = (1.0 - persistence) * weighted + agreement * persistence ** len(left)
    return float(min(1.0, max(0.0, value)))


def _topic_stability(
    top_words: pd.DataFrame,
    *,
    seeds: Sequence[int],
    reference_seed: int,
    topics: int,
    top_n: int,
    family: str,
    persistence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    required = {
        "seed", "reporting_family", "topic", "rank", "vocabulary_index", "token"
    }
    if not required <= set(top_words.columns):
        raise ValueError("Step-8 top-word file has an unexpected schema")
    subset = top_words[
        (top_words["reporting_family"] == family)
        & (top_words["seed"].astype(int).isin(list(map(int, seeds))))
        & (top_words["rank"].astype(int) <= int(top_n))
    ].copy()
    expected_rows = len(seeds) * topics * top_n
    if len(subset) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} frozen full-model top-word rows, found {len(subset)}"
        )
    ranked: dict[tuple[int, int], list[int]] = {}
    tokens: dict[tuple[int, int], list[str]] = {}
    for (seed, topic), group in subset.groupby(["seed", "topic"], sort=True):
        ordered = group.sort_values("rank")
        ranks = ordered["rank"].astype(int).tolist()
        if ranks != list(range(1, top_n + 1)):
            raise ValueError(f"Top-word ranks are incomplete for seed={seed}, topic={topic}")
        ranked[(int(seed), int(topic))] = ordered["vocabulary_index"].astype(int).tolist()
        tokens[(int(seed), int(topic))] = ordered["token"].astype(str).tolist()
    if set(ranked) != {(int(seed), topic) for seed in seeds for topic in range(topics)}:
        raise ValueError("The frozen seed-by-topic top-word grid is incomplete")

    reference = [ranked[(reference_seed, topic)] for topic in range(topics)]
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    nonreference_scores: list[float] = []
    nonreference_jaccard: list[float] = []
    for seed in seeds:
        candidate = [ranked[(int(seed), topic)] for topic in range(topics)]
        overlaps = np.empty((topics, topics), dtype=np.float64)
        for ref_topic in range(topics):
            left = set(reference[ref_topic])
            for candidate_topic in range(topics):
                right = set(candidate[candidate_topic])
                overlaps[ref_topic, candidate_topic] = len(left & right) / len(left | right)
        row_index, column_index = linear_sum_assignment(-overlaps)
        if sorted(row_index.tolist()) != list(range(topics)) or len(set(column_index)) != topics:
            raise RuntimeError("Topic alignment did not produce a one-to-one permutation")
        score_by_topic: dict[int, float] = {}
        jaccard_by_topic: dict[int, float] = {}
        for ref_topic, matched_topic in zip(row_index.tolist(), column_index.tolist()):
            rbo = _rbo(reference[ref_topic], candidate[matched_topic], persistence)
            jaccard = float(overlaps[ref_topic, matched_topic])
            score_by_topic[ref_topic] = rbo
            jaccard_by_topic[ref_topic] = jaccard
            detail.append(
                {
                    "reference_seed": int(reference_seed),
                    "seed": int(seed),
                    "reference_topic": int(ref_topic),
                    "matched_topic": int(matched_topic),
                    "top_word_set_jaccard": jaccard,
                    "rank_biased_overlap_p_0_9": rbo,
                    "reference_top_words": " | ".join(tokens[(reference_seed, ref_topic)]),
                    "matched_top_words": " | ".join(tokens[(int(seed), matched_topic)]),
                    "analysis_status": "posthoc_descriptive_not_used_for_selection",
                }
            )
            if int(seed) != reference_seed:
                nonreference_scores.append(rbo)
                nonreference_jaccard.append(jaccard)
        values = [score_by_topic[topic] for topic in range(topics)]
        jaccards = [jaccard_by_topic[topic] for topic in range(topics)]
        summary.append(
            {
                "reference_seed": int(reference_seed),
                "seed": int(seed),
                "mean_rbo_p_0_9": float(np.mean(values)),
                "minimum_rbo_p_0_9": float(np.min(values)),
                "mean_top_word_set_jaccard": float(np.mean(jaccards)),
                "minimum_top_word_set_jaccard": float(np.min(jaccards)),
                "aligned_topics": topics,
                "analysis_status": "posthoc_descriptive_not_used_for_selection",
            }
        )
    overall = {
        "reference_seed": int(reference_seed),
        "comparison_seeds": len(seeds) - 1,
        "aligned_topic_pairs_excluding_self": len(nonreference_scores),
        "mean_rbo_p_0_9_excluding_self": float(np.mean(nonreference_scores)),
        "sample_sd_rbo_p_0_9_excluding_self": float(np.std(nonreference_scores, ddof=1)),
        "minimum_rbo_p_0_9_excluding_self": float(np.min(nonreference_scores)),
        "maximum_rbo_p_0_9_excluding_self": float(np.max(nonreference_scores)),
        "mean_top_word_set_jaccard_excluding_self": float(np.mean(nonreference_jaccard)),
        "minimum_top_word_set_jaccard_excluding_self": float(np.min(nonreference_jaccard)),
        "analysis_status": "posthoc_descriptive_not_used_for_selection",
    }
    return detail, summary, overall


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, value)):02x}" for value in rgb)


def _blend(left: tuple[int, int, int], right: tuple[int, int, int], amount: float) -> str:
    amount = min(1.0, max(0.0, amount))
    return _hex(tuple(round(a + (b - a) * amount) for a, b in zip(left, right)))


def _heat_color(value: float, *, kind: str, maximum: float) -> tuple[str, str]:
    if not math.isfinite(value):
        return "#e5e7eb", "#374151"
    if kind == "rank":
        amount = 0.0 if maximum <= 1 else (value - 1.0) / (maximum - 1.0)
        color = _blend((210, 244, 222), (249, 201, 205), amount)
        return color, "#111827"
    if kind == "stability":
        amount = (value - 0.75) / 0.25
        color = _blend((239, 246, 255), (30, 64, 175), amount)
        return color, "#ffffff" if amount > 0.62 else "#111827"
    magnitude = 0.0 if maximum <= 0 else min(1.0, abs(value) / maximum)
    if value >= 0:
        color = _blend((247, 252, 249), (22, 163, 74), magnitude)
    else:
        color = _blend((255, 251, 235), (220, 38, 38), magnitude)
    return color, "#ffffff" if magnitude > 0.58 else "#111827"


def _heatmap_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: np.ndarray,
    kind: str,
    formatter: str,
    legend_left: str,
    legend_right: str,
) -> None:
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError("SVG heatmap dimensions do not match its labels")
    left = 320
    top = 135
    cell_w = 125 if len(column_labels) <= 4 else 118
    cell_h = 42
    width = left + cell_w * len(column_labels) + 52
    if kind == "effect":
        width = max(width, 1050)
    height = top + cell_h * len(row_labels) + (128 if kind == "effect" else 110)
    finite = values[np.isfinite(values)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    if kind == "effect" and finite.size:
        maximum = float(np.max(np.abs(finite)))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)}</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#111827}'
        '.title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#4b5563}'
        '.label{font-size:13px}.cell{font-size:13px;font-weight:600}'
        '.legend{font-size:12px;fill:#4b5563}</style>',
        f'<text x="24" y="36" class="title">{html.escape(title)}</text>',
        f'<text x="24" y="62" class="subtitle">{html.escape(subtitle)}</text>',
    ]
    for column, label in enumerate(column_labels):
        x = left + column * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x:.1f}" y="112" text-anchor="middle" class="label">'
            f"{html.escape(str(label))}</text>"
        )
    for row, label in enumerate(row_labels):
        y = top + row * cell_h
        lines.append(
            f'<text x="{left - 14}" y="{y + 27}" text-anchor="end" class="label">'
            f"{html.escape(str(label))}</text>"
        )
        for column in range(len(column_labels)):
            value = float(values[row, column])
            x = left + column * cell_w
            fill, foreground = _heat_color(value, kind=kind, maximum=maximum)
            shown = "NA" if not math.isfinite(value) else format(value, formatter)
            lines.extend(
                [
                    f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" '
                    f'height="{cell_h - 4}" rx="4" fill="{fill}"/>',
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + 27}" '
                    f'text-anchor="middle" class="cell" style="fill:{foreground}">'
                    f"{html.escape(shown)}</text>",
                ]
            )
    legend_y = top + cell_h * len(row_labels) + 45
    steps = 12
    for index in range(steps):
        if kind == "effect":
            value = maximum * (2 * index / (steps - 1) - 1)
        elif kind == "stability":
            value = 0.75 + 0.25 * index / (steps - 1)
        else:
            value = 1 + (maximum - 1) * index / (steps - 1)
        fill, _ = _heat_color(value, kind=kind, maximum=maximum)
        lines.append(
            f'<rect x="{left + index * 24}" y="{legend_y}" width="25" height="13" fill="{fill}"/>'
        )
    lines.extend(
        [
            f'<text x="{left}" y="{legend_y + 32}" class="legend">{html.escape(legend_left)}</text>',
            f'<text x="{left + (steps - 1) * 24 + 25}" '
            f'y="{legend_y + (50 if kind == "effect" else 32)}" '
            f'text-anchor="end" class="legend">{html.escape(legend_right)}</text>',
            "</svg>",
        ]
    )
    _write_text(path, "\n".join(lines))
    ET.parse(path)


def _stability_outputs(
    step8: Path, out: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    reporting = protocol["reporting"]
    detail, summary, overall = _topic_stability(
        pd.read_csv(step8 / "worker/top_words_by_seed.csv"),
        seeds=list(map(int, reporting["seeds"])),
        reference_seed=int(reporting["reference_seed"]),
        topics=int(reporting["topics"]),
        top_n=int(reporting["top_words"]),
        family=str(reporting["reporting_family"]),
        persistence=float(reporting["rank_biased_overlap_p"]),
    )
    _write_csv(out / "supplement/topic_stability_aligned_pairs.csv", detail)
    _write_csv(out / "supplement/topic_stability_by_seed.csv", summary)
    _write_json(out / "supplement/topic_stability_summary.json", overall)
    seeds = list(map(int, reporting["seeds"]))
    topics = int(reporting["topics"])
    lookup = {
        (int(row["seed"]), int(row["reference_topic"])): float(
            row["rank_biased_overlap_p_0_9"]
        )
        for row in detail
    }
    summary_lookup = {int(row["seed"]): float(row["mean_rbo_p_0_9"]) for row in summary}
    values = np.asarray(
        [
            [lookup[(seed, topic)] for topic in range(topics)] + [summary_lookup[seed]]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    _heatmap_svg(
        out / "paper/figure_topic_stability.svg",
        title="Full-model topic stability across the ten prespecified seeds",
        subtitle=(
            "Topics are aligned one-to-one to seed 101 by maximum top-10 set Jaccard; cells show "
            "rank-biased overlap (p=0.9). Post-hoc descriptive analysis only."
        ),
        row_labels=[str(seed) for seed in seeds],
        column_labels=[f"T{topic + 1}" for topic in range(topics)] + ["Mean"],
        values=values,
        kind="stability",
        formatter=".3f",
        legend_left="0.75 / lower stability",
        legend_right="1.00 / identical order",
    )
    return overall


def _maximum_iteration(metadata: Mapping[str, Any]) -> float | None:
    parameters = metadata.get("resolved_parameters") or metadata.get("resolved_hyperparameters") or {}
    candidates: list[float] = []
    for key, value in parameters.items():
        normalized = str(key).lower()
        if (
            any(term in normalized for term in ("maximum", "max_", "epochs"))
            and any(term in normalized for term in ("iteration", "epoch"))
            and isinstance(value, (int, float))
        ):
            candidates.append(float(value))
    return min(candidates) if candidates else None


def _diagnostics(step9: Path, out: Path, methods: Sequence[str], seeds: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            receipt_path = step9 / "receipts" / method / f"seed_{seed}" / "metadata.json"
            output_path = receipt_path.parent / "canonical_output.npz"
            if not receipt_path.is_file() or not output_path.is_file():
                raise FileNotFoundError(f"Comparator receipt is incomplete: {method}, seed {seed}")
            metadata = _json(receipt_path)
            if int(metadata.get("seed", -1)) != int(seed) or metadata.get("method_id") != method:
                raise ValueError(f"Comparator receipt identity mismatch: {receipt_path}")
            if metadata.get("target_available_to_adapter") is not False:
                raise ValueError(f"Target firewall was not closed in {receipt_path}")
            if int(metadata.get("labels", metadata.get("labels_accessed", 0))) != 0:
                raise ValueError(f"Labels were accessed in {receipt_path}")
            converged = metadata.get("converged")
            maximum = _maximum_iteration(metadata)
            iterations = metadata.get("iterations_completed")
            ceiling = (
                bool(float(iterations) >= maximum)
                if maximum is not None and isinstance(iterations, (int, float))
                else "not_determinable"
            )
            rows.append(
                {
                    "method_id": method,
                    "method": metadata.get("display_name", DISPLAY[method]),
                    "family": metadata.get("family", ""),
                    "seed": int(seed),
                    "receipt_complete": True,
                    "runtime_seconds": float(metadata["runtime_seconds"]),
                    "iterations_completed": iterations if iterations is not None else "",
                    "reported_convergence": (
                        "true" if converged is True else "false" if converged is False else "not_reported"
                    ),
                    "iteration_or_epoch_ceiling_reached": ceiling,
                    "device": metadata.get("device", "not_recorded"),
                    "peak_cuda_memory_mib": metadata.get("peak_cuda_memory_mib", ""),
                    "peak_python_memory_mib": metadata.get("peak_python_memory_mib", ""),
                    "likelihood_comparable": metadata.get("likelihood_comparable", False),
                    "source_kind": metadata.get("source_kind", "project_or_library_implementation"),
                    "source_revision": metadata.get("source_revision", ""),
                    "environment_fingerprint": metadata.get("environment_fingerprint", ""),
                    "target_available_to_adapter": False,
                    "labels_accessed": 0,
                    "interpretation": "descriptive_only_heterogeneous_runtimes_not_headline_efficiency",
                }
            )
    _write_csv(out / "supplement/comparator_execution_diagnostics_by_seed.csv", rows)
    summary: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in rows if row["method_id"] == method]
        runtimes = [float(row["runtime_seconds"]) for row in selected]
        summary.append(
            {
                "method_id": method,
                "method": selected[0]["method"],
                "receipts": len(selected),
                "runtime_seconds_mean": statistics.fmean(runtimes),
                "runtime_seconds_sample_sd": statistics.stdev(runtimes),
                "reported_converged_count": sum(row["reported_convergence"] == "true" for row in selected),
                "reported_not_converged_count": sum(row["reported_convergence"] == "false" for row in selected),
                "convergence_not_reported_count": sum(
                    row["reported_convergence"] == "not_reported" for row in selected
                ),
                "devices": " | ".join(sorted({str(row["device"]) for row in selected})),
                "interpretation": "descriptive_only_do_not_rank_efficiency_across_heterogeneous_runtimes",
            }
        )
    _write_csv(out / "supplement/comparator_execution_diagnostics_summary.csv", summary)
    return {
        "receipt_rows": len(rows),
        "methods": len(summary),
        "seeds_per_method": len(seeds),
        "not_converged_receipts": sum(row["reported_convergence"] == "false" for row in rows),
        "convergence_not_reported_receipts": sum(
            row["reported_convergence"] == "not_reported" for row in rows
        ),
        "runtime_comparison_status": "descriptive_only_heterogeneous_runtimes",
    }


def _step7_source(
    root: Path,
    step8: Path,
    contract8: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Locate and verify the exact final-refit artifact inherited by Step 8."""

    expected = protocol["prerequisite"]
    candidates: list[Path] = []
    for parent in (
        root / "outputs/v6/step07_final_refit_freeze",
        root / "step07_final_refit_freeze",
    ):
        candidates.extend(sorted(parent.glob("*/step07_contract.json"), reverse=True))
    for contract_path in candidates:
        contract = _json(contract_path)
        if not (
            contract.get("status") == expected["step7_status"]
            and contract.get("readiness") == expected["step7_readiness"]
            and contract.get("freeze_identity") == expected["step7_freeze_identity"]
            and contract.get("freeze_identity") == contract8.get("step7_freeze_identity")
            and _verify_fingerprint(contract, "contract_fingerprint")
        ):
            continue
        base = contract_path.parent
        try:
            _verify_manifest(base)
        except (ValueError, FileNotFoundError):
            continue
        lineage = _json(step8 / "step7_lineage.json")
        relative = {
            "generated_manifest": "generated_artifact_manifest.json",
            "runtime_receipt": "worker/runtime_receipt.json",
            "source_receipt": "official_source_receipt.json",
            "variant_manifest": "worker/final_variant_manifest.json",
            "worker_job": "worker_job.json",
            "checkpoint_manifest": "worker/checkpoint_manifest.json",
            "freeze_receipt": "worker/final_freeze_receipt.json",
        }
        valid = True
        for key, relative_path in relative.items():
            item = base / relative_path
            record = lineage.get(key, {})
            if (
                not item.is_file()
                or sha256_file(item) != record.get("sha256")
                or int(item.stat().st_size) != int(record.get("size_bytes", -1))
            ):
                valid = False
                break
        if valid:
            return base, contract
    raise RuntimeError("The exact manifest-valid Step-7 frozen refit was not found")


def _training_convergence_svg(path: Path, frame: pd.DataFrame) -> None:
    width, height = 1100, 650
    left, right, top, bottom = 105, 55, 105, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    epochs = frame["epoch"].to_numpy(dtype=float)
    lower = frame["q25_normalized_loss"].to_numpy(dtype=float)
    upper = frame["q75_normalized_loss"].to_numpy(dtype=float)
    median = frame["median_normalized_loss"].to_numpy(dtype=float)
    x_min, x_max = float(epochs.min()), float(epochs.max())
    y_min = max(0.0, float(lower.min()) - 0.04)
    y_max = min(1.08, max(1.02, float(upper.max()) + 0.02))

    def xy(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_min) / (x_max - x_min) * plot_w
        py = top + (y_max - y) / (y_max - y_min) * plot_h
        return px, py

    upper_points = [xy(x, y) for x, y in zip(epochs, upper)]
    lower_points = [xy(x, y) for x, y in zip(epochs[::-1], lower[::-1])]
    polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper_points + lower_points)
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(x, y) for x, y in zip(epochs, median)))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>Frozen parent-model optimization convergence</title>",
        "<desc>Median normalized training loss and interquartile envelope over ten prespecified seeds.</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#111827}'
        '.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#4b5563}'
        '.axis{font-size:13px;fill:#374151}.legend{font-size:13px;fill:#374151}</style>',
        '<text x="28" y="38" class="title">Frozen parent-model optimization convergence</text>',
        '<text x="28" y="66" class="sub">Loss is divided by each seed’s epoch-1 loss; line = median, band = interquartile range (10 prespecified seeds).</text>',
    ]
    for y in np.linspace(y_min, y_max, 6):
        _, py = xy(x_min, float(y))
        svg.extend(
            [
                f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{py + 5:.2f}" text-anchor="end" class="axis">{y:.2f}</text>',
            ]
        )
    for epoch in (1, 20, 40, 60, 80, 100, 120):
        px, _ = xy(float(epoch), y_min)
        svg.extend(
            [
                f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<text x="{px:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="axis">{epoch}</text>',
            ]
        )
    svg.extend(
        [
            f'<polygon points="{polygon}" fill="#93c5fd" fill-opacity="0.52"/>',
            f'<polyline points="{line}" fill="none" stroke="#1d4ed8" stroke-width="4" stroke-linejoin="round"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" class="axis">Training epoch</text>',
            f'<text x="28" y="{top + plot_h / 2}" transform="rotate(-90 28 {top + plot_h / 2})" text-anchor="middle" class="axis">Normalized training loss</text>',
            f'<rect x="{width - 290}" y="{86}" width="24" height="13" fill="#93c5fd" fill-opacity="0.52"/>',
            f'<text x="{width - 258}" y="{98}" class="legend">Interquartile range</text>',
            "</svg>",
        ]
    )
    _write_text(path, "\n".join(svg))
    ET.parse(path)


def _training_convergence(step7: Path, out: Path, seeds: Sequence[int]) -> dict[str, Any]:
    trace = pd.read_csv(step7 / "worker/final_training_trace.csv")
    required = {"seed", "epoch", "documents", "loss"}
    if not required <= set(trace.columns):
        raise ValueError("Step-7 training trace has an unexpected schema")
    if set(trace["seed"].astype(int)) != set(map(int, seeds)):
        raise ValueError("Step-7 training trace does not contain the ten frozen seeds")
    epoch_sets = {
        int(seed): tuple(sorted(group["epoch"].astype(int)))
        for seed, group in trace.groupby("seed")
    }
    if any(values != tuple(range(1, 121)) for values in epoch_sets.values()):
        raise ValueError("Each frozen Step-7 seed must contain exactly epochs 1--120")
    trace = trace.copy()
    first = trace[trace["epoch"].eq(1)].set_index("seed")["loss"].to_dict()
    trace["normalized_loss"] = [
        float(loss) / float(first[int(seed)])
        for seed, loss in zip(trace["seed"], trace["loss"])
    ]
    rows: list[dict[str, Any]] = []
    for epoch, group in trace.groupby("epoch", sort=True):
        values = group["normalized_loss"].to_numpy(dtype=float)
        rows.append(
            {
                "epoch": int(epoch),
                "seeds": len(values),
                "median_normalized_loss": float(np.median(values)),
                "q25_normalized_loss": float(np.quantile(values, 0.25)),
                "q75_normalized_loss": float(np.quantile(values, 0.75)),
                "minimum_normalized_loss": float(np.min(values)),
                "maximum_normalized_loss": float(np.max(values)),
                "analysis_status": "frozen_training_diagnostic_not_test_performance",
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv(out / "supplement/training_convergence_by_epoch.csv", rows)
    _training_convergence_svg(out / "supplement/figure_parent_training_convergence.svg", frame)
    first_median = float(frame.iloc[0]["median_normalized_loss"])
    final_median = float(frame.iloc[-1]["median_normalized_loss"])
    per_seed = trace.sort_values("epoch").groupby("seed")["loss"].agg(["first", "last"])
    return {
        "seeds": len(seeds),
        "epochs_per_seed": 120,
        "trace_rows": len(trace),
        "initial_median_normalized_loss": first_median,
        "final_median_normalized_loss": final_median,
        "median_relative_reduction": 1.0 - final_median / first_median,
        "seeds_with_final_loss_below_initial": int((per_seed["last"] < per_seed["first"]).sum()),
        "interpretation": "parent_optimization_diagnostic_not_test_performance",
    }


def _graph_intervention_svg(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 1380, 740
    left, top, bar_w, row_h = 145, 120, 520, 49
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>Graph projection changes to frozen topic descriptors</title>",
        "<desc>Number of new top-ten words introduced by graph projection for each seed-101 topic.</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#111827}'
        '.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#4b5563}'
        '.label{font-size:14px}.value{font-size:14px;font-weight:700}'
        '.words{font-size:14px;fill:#374151}.axis{font-size:12px;fill:#6b7280}</style>',
        '<text x="28" y="38" class="title">What graph projection changes in each frozen topic</text>',
        '<text x="28" y="66" class="sub">First prespecified seed (101), same topic index; bars count newly introduced words among the final top 10.</text>',
    ]
    for tick in range(0, 11, 2):
        x = left + bar_w * tick / 10
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{top + row_h * len(rows)}" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="{top - 24}" text-anchor="middle" class="axis">{tick}</text>',
            ]
        )
    for index, row in enumerate(rows):
        y = top + index * row_h
        value = int(row["introduced_word_count"])
        fill_width = bar_w * value / 10
        svg.extend(
            [
                f'<text x="{left - 15}" y="{y + 28}" text-anchor="end" class="label">T{int(row["topic_id"]):02d}</text>',
                f'<rect x="{left}" y="{y + 7}" width="{bar_w}" height="28" rx="5" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y + 7}" width="{fill_width:.1f}" height="28" rx="5" fill="#2563eb"/>',
                f'<text x="{left + fill_width - 10:.1f}" y="{y + 27}" text-anchor="end" class="value" style="fill:#ffffff">{value}</text>',
                f'<text x="{left + bar_w + 28}" y="{y + 27}" class="words">New: {html.escape(str(row["introduced_words_preview"]))}</text>',
            ]
        )
    svg.extend(
        [
            f'<text x="{left + bar_w / 2}" y="{top + row_h * len(rows) + 47}" text-anchor="middle" class="axis">Introduced words in graph-transformed top 10</text>',
            '<text x="28" y="716" class="axis">Mechanistic post-hoc audit; word changes are not an additional performance endpoint.</text>',
            "</svg>",
        ]
    )
    _write_text(path, "\n".join(svg))
    ET.parse(path)


def _graph_intervention(
    step8: Path, out: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    reporting = protocol["reporting"]
    seed = int(reporting["reference_seed"])
    full_family = str(reporting["reporting_family"])
    parent_family = str(reporting["parent_reporting_family"])
    top_n = int(reporting["top_words"])
    topics = int(reporting["topics"])
    frame = pd.read_csv(step8 / "worker/top_words_by_seed.csv")
    rows: list[dict[str, Any]] = []
    for topic in range(topics):
        parent = frame[
            frame["seed"].eq(seed)
            & frame["reporting_family"].eq(parent_family)
            & frame["topic"].eq(topic)
        ].sort_values("rank").head(top_n)
        full = frame[
            frame["seed"].eq(seed)
            & frame["reporting_family"].eq(full_family)
            & frame["topic"].eq(topic)
        ].sort_values("rank").head(top_n)
        if len(parent) != top_n or len(full) != top_n:
            raise ValueError(f"Incomplete parent/full top words for frozen topic {topic}")
        parent_indices = parent["vocabulary_index"].astype(int).tolist()
        full_indices = full["vocabulary_index"].astype(int).tolist()
        parent_set, full_set = set(parent_indices), set(full_indices)
        retained_indices = parent_set & full_set
        introduced = full[~full["vocabulary_index"].astype(int).isin(retained_indices)]
        removed = parent[~parent["vocabulary_index"].astype(int).isin(retained_indices)]
        introduced_tokens = introduced["token"].astype(str).tolist()
        retained_tokens = full[full["vocabulary_index"].astype(int).isin(retained_indices)]["token"].astype(str).tolist()
        removed_tokens = removed["token"].astype(str).tolist()
        union = parent_set | full_set
        rows.append(
            {
                "topic_id": topic + 1,
                "frozen_source_seed": seed,
                "topic_alignment": "same_frozen_topic_index",
                "retained_word_count": len(retained_indices),
                "introduced_word_count": len(introduced_tokens),
                "top10_set_jaccard_parent_vs_graph": len(retained_indices) / len(union),
                "retained_words": " | ".join(retained_tokens),
                "introduced_words": " | ".join(introduced_tokens),
                "removed_parent_words": " | ".join(removed_tokens),
                "introduced_words_preview": " · ".join(introduced_tokens[:5]),
                "mean_development_document_frequency_introduced": float(
                    introduced["development_document_frequency"].astype(float).mean()
                ),
                "minimum_development_document_frequency_introduced": int(
                    introduced["development_document_frequency"].astype(int).min()
                ),
                "analysis_status": "mechanistic_posthoc_not_performance_endpoint",
            }
        )
    _write_csv(out / "supplement/graph_lexical_intervention_by_topic.csv", rows)
    _graph_intervention_svg(out / "paper/figure_graph_lexical_intervention.svg", rows)
    introduced_counts = np.asarray([row["introduced_word_count"] for row in rows], dtype=float)
    jaccards = np.asarray([row["top10_set_jaccard_parent_vs_graph"] for row in rows], dtype=float)
    return {
        "source_seed": seed,
        "source_seed_rule": "first_prespecified_seed_not_best_seed",
        "topics": len(rows),
        "mean_introduced_words_per_topic": float(introduced_counts.mean()),
        "minimum_introduced_words": int(introduced_counts.min()),
        "maximum_introduced_words": int(introduced_counts.max()),
        "mean_same_topic_top10_jaccard": float(jaccards.mean()),
        "interpretation": "mechanistic_posthoc_not_performance_endpoint",
    }


def _case_preregistration(step10: Path, out: Path) -> dict[str, Any]:
    public = pd.read_csv(
        step10 / "human_packet_share_only_this_folder/annotator_01_case_posts.csv"
    )
    key = pd.read_csv(step10 / "private_do_not_share_with_annotators/case_source_key.csv")
    if len(public) != 30 or len(key) != 30 or set(public["item_id"]) != set(key["item_id"]):
        raise ValueError("The Step-10 30-item case packet is incomplete")
    merged = key[["item_id", "source_topic", "city", "selection_partition", "selection_rule"]]
    grid = merged.groupby(["source_topic", "city"]).size()
    if len(grid) != 30 or not grid.eq(1).all():
        raise ValueError("The case packet is not an exact ten-topic by three-city grid")
    if set(merged["city"]) != {"beijing", "shanghai", "xiamen"}:
        raise ValueError("The case packet does not cover the three frozen cities")
    if not set(merged["selection_partition"]) <= {"train", "validation"}:
        raise ValueError("A test document entered the case-study packet")
    _write_csv(
        out / "supplement/case_study_design_audit.csv",
        [
            {
                "item_id": row.item_id,
                "source_topic": int(row.source_topic) + 1,
                "city": row.city,
                "selection_partition": row.selection_partition,
                "design_cell": f"T{int(row.source_topic) + 1:02d}_{row.city}",
                "one_case_per_topic_city": True,
            }
            for row in merged.itertuples(index=False)
        ],
    )
    preregistration = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_HUMAN_RESPONSES",
        "design": "10_topics_x_3_cities_x_3_independent_annotators",
        "case_items": 30,
        "ratings_expected": 90,
        "selection_partition": "development_only_train_or_validation",
        "test_documents_used_for_case_selection": 0,
        "city_requirement_map": {
            "rows": "independent_consensus_topic_labels",
            "columns": ["beijing", "shanghai", "xiamen"],
            "cell_value": "median_topic_fit_1_to_5_across_three_annotators",
            "cell_marker": "majority_service_need_present_yes_no",
            "no_consensus_policy": "display_no_consensus_never_force_a_label",
        },
        "evidence_chain_figure": {
            "panels": "one_case_per_city",
            "eligibility": "at_least_two_of_three_service_need_yes_and_median_topic_fit_at_least_4",
            "within_city_selection_order": [
                "highest_median_topic_fit",
                "highest_service_need_yes_fraction",
                "highest_frozen_lexical_relevance",
                "lexicographically_smallest_item_id",
            ],
            "chain": [
                "delexicalized_passenger_post",
                "frozen_topic_words",
                "independently_supported_requirement",
                "transport_action_after_blinded_adjudication",
            ],
            "if_no_eligible_case": "show_no_eligible_case_for_city_and_do_not_relax_rule",
        },
        "free_text_consensus": (
            "seal all nine original files first; then a senior transport-domain adjudicator "
            "reviews anonymized responses and records a consensus label/action without editing originals"
        ),
        "human_results_available": False,
        "performance_threshold_used": False,
    }
    preregistration["preregistration_identity"] = sha256_object(preregistration)
    _write_json(out / "paper/case_study_rendering_preregistration.json", preregistration)
    _write_text(
        out / "paper/CASE_STUDY_NEXT_GATE.md",
        """# Case-study evidence chain: pending independent annotation

The 10-topic × 3-city case design is complete and uses development documents only. No final passenger-requirement map or evidence-chain figure is rendered in Step 11 R2 because doing so would require inventing or pre-empting the three independent annotators' judgments.

After the nine original response files are sealed and Step 10 validates them, Step 12 will apply the frozen rendering rule in `case_study_rendering_preregistration.json`. It will create (i) a three-city requirement map and (ii) a three-panel evidence chain from delexicalized passenger post to topic evidence, independently supported requirement, and transport action. If the eligibility rule is not met, no case will be substituted post hoc.
""",
    )
    return {
        "case_items": 30,
        "topics": 10,
        "cities": 3,
        "expected_independent_ratings": 90,
        "test_documents_used_for_selection": 0,
        "status": "FROZEN_PENDING_INDEPENDENT_ANNOTATION",
        "preregistration_identity": preregistration["preregistration_identity"],
    }


def _inventory(step7: Path, step8: Path, step9: Path, step10: Path, out: Path) -> int:
    sources = [
        ("step7", step7 / "step07_contract.json", "frozen final-refit contract"),
        ("step7", step7 / "worker/final_training_trace.csv", "optimization-convergence source"),
        ("step8", step8 / "step08_contract.json", "frozen ablation/test contract"),
        ("step8", step8 / "worker/top_words_by_seed.csv", "full-model stability source"),
        ("step9", step9 / "step09_contract.json", "fair-comparator contract"),
        ("step9", step9 / "receipts/lda/seed_101/metadata.json", "comparator-diagnostic schema source"),
        ("step9", step9 / "generated_artifact_manifest.json", "comparator artifact manifest"),
        ("step10", step10 / "step10_contract.json", "validity preparation/final contract"),
        ("step10", step10 / "human_packet_share_only_this_folder/annotator_01_case_posts.csv", "blinded case-design source"),
        ("step10", step10 / "private_do_not_share_with_annotators/case_source_key.csv", "private case-design audit source"),
        ("step10", step10 / "generated_artifact_manifest.json", "reporting artifact manifest"),
    ]
    rows = []
    for stage, path, role in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "source_stage": stage,
                "artifact": path.name,
                "role": role,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "read_only_use": True,
            }
        )
    _write_csv(out / "supplement/frozen_evidence_inventory.csv", rows)
    return len(rows)


def _evidence_guide(
    out: Path,
    *,
    stability: Mapping[str, Any],
    graph: Mapping[str, Any],
    training: Mapping[str, Any],
    case: Mapping[str, Any],
    contract10: Mapping[str, Any],
) -> None:
    human_status = contract10["human_evaluation"]["status"]
    external_status = contract10["external_evaluation"]["status"]
    text = f"""# SC-HTM V6.1 frozen evidence guide

## What Step 11 does

Step 11 R2 is a reporting-only synthesis of the immutable Step-7--10 artifacts. It performs zero model fits, zero optimizer updates, zero comparator refits, zero seed selection, and zero result changes. It deliberately does not turn the baseline/SOTA/ablation tables into a second figure.

## Automatically supported observations

- After one-to-one topic alignment, the non-reference mean RBO is {stability['mean_rbo_p_0_9_excluding_self']:.4f} (sample SD {stability['sample_sd_rbo_p_0_9_excluding_self']:.4f}; minimum {stability['minimum_rbo_p_0_9_excluding_self']:.4f}). The mean matched top-10 set Jaccard is {stability['mean_top_word_set_jaccard_excluding_self']:.4f}.
- Graph projection introduces a mean of {graph['mean_introduced_words_per_topic']:.2f} words per final top-10 descriptor at the first prespecified seed; this is a mechanism audit, not a new performance endpoint.
- The frozen parent optimization reaches a final median normalized loss of {training['final_median_normalized_loss']:.4f}; all {training['seeds_with_final_loss_below_initial']} prespecified seeds finish below their initial loss.
- The case-study design contains {case['case_items']} development-only cases covering all {case['topics']} topics and {case['cities']} cities, with zero test documents used for selection. Its rendering rule was frozen before human responses.
- Human status remains `{human_status}`. External/prospective status remains `{external_status}`.

These are descriptive observations from frozen evidence. They do not establish operational causality or external generalization.

## Recommended main-paper placement

1. Dataset table (corpus, city, split, time, vocabulary, and privacy statistics).
2. Method architecture figure.
3. Established-baseline multi-seed table from Step 10.
4. Recent-SOTA multi-seed table from Step 10.
5. Prespecified ablation multi-seed table from Step 10, including all cross-metric tradeoffs.
6. `figure_topic_stability.svg` as evidence of seed robustness.
7. `figure_graph_lexical_intervention.svg` only if space permits and the mechanism discussion explains that lexical change is not itself performance.
8. After the human gate passes: one human-evaluation table, one three-city requirement map, and one compact evidence-chain case figure.

Do not put the post-hoc seed-809 tables in the main paper. Do not replace the multi-seed tables with a favorable single seed.

## Recommended supplementary placement

- Full aligned-pair stability CSV files.
- `figure_parent_training_convergence.svg` and its 120-epoch envelope CSV.
- Complete graph lexical-intervention details and comparator execution diagnostics.
- Comparator execution diagnostics, with the heterogeneous-runtime caveat.
- Step-10 single-seed tables, full per-seed metrics, paired statistics, annotation protocol, blinding receipts, and external-evaluation contract.
- A sensitivity rerun for comparators whose algorithm reported non-convergence is desirable, but it must be disclosed as a sensitivity analysis and may not replace the frozen main table.

## Case-study rule

Step 11 R2 creates no passenger-requirement result before annotation. The exact three-city map and evidence-chain selection rule is locked in `case_study_rendering_preregistration.json`. Step 12 may render those outputs only from three genuine independent response sets; it may not substitute a nicer case after observing responses.

## Remaining evidence phases

### Step 12 — independent human validity and case-study finalization

Give only the Step-10 `human_packet_share_only_this_folder` directory to three independent transport-domain annotators. Each annotator completes their own three numbered CSV files without seeing answer keys or each other's work. Copy the nine returned files to `inputs\\v6\\step10\\human`, rerun Step 10, and retain agreement, word-intrusion accuracy, coherence/actionability ratings, consensus labels, and the delexicalized 30-cell city case study. No researcher should fill missing answers on an annotator's behalf.

### Step 13 — external or prospective temporal validation

Provide one genuinely independent dataset with its schema, provenance, license/permission, language, city/time coverage, and frozen preprocessing mapping. Build the dataset-specific evaluator around the exact frozen Step-7 state; permit no refit or tuning. Return its signed contract and generated manifest to `inputs\\v6\\step10\\external`, then rerun Step 10. This code cannot be finalized responsibly before the actual external dataset is supplied.

## Submission condition

Automatic evidence is complete after Step 11. Manuscript-submission readiness is not complete until Step 10 is rerun with valid independent human and external packets and returns `PASS`. A top-ranked journal outcome can never be guaranteed; the defensible objective is complete, frozen, auditable evidence.
"""
    _write_text(out / "paper/STEP11_EVIDENCE_GUIDE.md", text)


def _manifest(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(
        candidate for candidate in out.rglob("*")
        if candidate.is_file() and candidate.name != "generated_artifact_manifest.json"
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
    destination = root / "v6_step11_results.zip"
    if destination.is_file():
        destination.unlink()
    prefix = Path("step11_frozen_evidence") / out.name
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidate for candidate in out.rglob("*") if candidate.is_file()):
            archive.write(path, (prefix / path.relative_to(out)).as_posix())
    return destination


def run_v6_step11(*, project_root: Path, config_path: Path) -> Path:
    """Create distinct, auditable evidence from the frozen Step-7--10 lineage."""

    root = project_root.expanduser().resolve()
    protocol = _load_protocol(config_path)
    print(
        f"[V6.1 Step 11] START | implementation={IMPLEMENTATION} | "
        "fits=0 optimizer-steps=0 refits=0 frozen-result-changes=0 test-text=0",
        flush=True,
    )
    step8, contract8, step9, contract9, step10, contract10 = _frozen_sources(root, protocol)
    out = root / "outputs/v6/step11_frozen_evidence" / _utc_id()
    out.mkdir(parents=True, exist_ok=False)

    step7, contract7 = _step7_source(root, step8, contract8, protocol)
    stability_state = _stability_outputs(step8, out, protocol)
    graph_state = _graph_intervention(step8, out, protocol)
    training_state = _training_convergence(
        step7,
        out,
        list(map(int, protocol["reporting"]["seeds"])),
    )
    case_state = _case_preregistration(step10, out)
    diagnostic_state = _diagnostics(
        step9,
        out,
        list(protocol["reporting"]["methods"])[1:],
        list(map(int, protocol["reporting"]["seeds"])),
    )
    inventory_count = _inventory(step7, step8, step9, step10, out)
    _evidence_guide(
        out,
        stability=stability_state,
        graph=graph_state,
        training=training_state,
        case=case_state,
        contract10=contract10,
    )

    expected_figures = [
        out / "paper/figure_topic_stability.svg",
        out / "paper/figure_graph_lexical_intervention.svg",
        out / "supplement/figure_parent_training_convergence.svg",
    ]
    for figure in expected_figures:
        ET.parse(figure)

    checks = [
        ("exact_step7_contract_and_manifest", True, contract7["freeze_identity"]),
        ("exact_step8_contract_and_manifest", True, sha256_file(step8 / "step08_contract.json")),
        ("exact_step9_contract_and_manifest", True, sha256_file(step9 / "step09_contract.json")),
        ("valid_step10_contract_manifest_and_case_packet", True, contract10["contract_fingerprint"]),
        ("step10_lineage_matches_frozen_step8_step9", True, "exact"),
        ("performance_table_not_replotted", not (out / "paper/figure_rank_profile.svg").exists(), "rank-profile figure absent by policy"),
        ("city_ablation_table_not_replotted", not (out / "supplement/figure_city_module_effects.svg").exists(), "city-effect figure absent by policy"),
        ("topic_alignment_grid_complete", stability_state["aligned_topic_pairs_excluding_self"] == 90, str(stability_state)),
        ("topic_stability_values_are_bounded", 0.0 <= stability_state["minimum_rbo_p_0_9_excluding_self"] <= stability_state["maximum_rbo_p_0_9_excluding_self"] <= 1.0, str(stability_state)),
        ("graph_intervention_uses_first_prespecified_seed", graph_state["source_seed"] == 101 and graph_state["topics"] == 10, str(graph_state)),
        ("graph_intervention_is_descriptive_not_endpoint", graph_state["interpretation"] == "mechanistic_posthoc_not_performance_endpoint", str(graph_state)),
        ("training_trace_grid_complete", training_state["trace_rows"] == 1200 and training_state["epochs_per_seed"] == 120, str(training_state)),
        ("all_frozen_seed_losses_finish_below_start", training_state["seeds_with_final_loss_below_initial"] == 10, str(training_state)),
        ("case_design_grid_complete", case_state["case_items"] == 30 and case_state["topics"] == 10 and case_state["cities"] == 3, str(case_state)),
        ("case_design_uses_no_test_documents", case_state["test_documents_used_for_selection"] == 0, str(case_state)),
        ("comparator_receipts_complete", diagnostic_state["receipt_rows"] == 100, str(diagnostic_state)),
        ("frozen_evidence_inventory_complete", inventory_count == 11, f"artifacts={inventory_count}"),
        ("all_svg_figures_are_valid_xml", True, "3/3"),
        ("no_model_fit_optimizer_refit_or_result_change", True, "0/0/0/0"),
        ("no_human_or_external_result_invented", True, "pending states inherited unchanged"),
    ]
    failed = [name for name, passed, _ in checks if not passed]
    _write_csv(
        out / "hard_checks.csv",
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}
            for name, passed, evidence in checks
        ],
    )
    if failed:
        raise RuntimeError(f"Step-11 hard checks failed: {failed}")

    lineage = {
        "step7_contract_path": str((step7 / "step07_contract.json").resolve()),
        "step7_contract_sha256": sha256_file(step7 / "step07_contract.json"),
        "step7_contract_fingerprint": contract7["contract_fingerprint"],
        "step7_freeze_identity": contract7["freeze_identity"],
        "step8_contract_path": str((step8 / "step08_contract.json").resolve()),
        "step8_contract_sha256": sha256_file(step8 / "step08_contract.json"),
        "step8_contract_fingerprint": contract8["contract_fingerprint"],
        "step9_contract_path": str((step9 / "step09_contract.json").resolve()),
        "step9_contract_sha256": sha256_file(step9 / "step09_contract.json"),
        "step9_contract_fingerprint": contract9["contract_fingerprint"],
        "step10_contract_path": str((step10 / "step10_contract.json").resolve()),
        "step10_contract_sha256": sha256_file(step10 / "step10_contract.json"),
        "step10_contract_fingerprint": contract10["contract_fingerprint"],
        "step10_status": contract10["status"],
        "model_fits": 0,
        "optimizer_steps": 0,
        "comparator_refits": 0,
        "frozen_result_changes": 0,
    }
    _write_json(out / "frozen_lineage.json", lineage)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "scope": SCOPE,
        "status": "PASS",
        "readiness": READINESS,
        "hard_checks_passed": len(checks),
        "hard_checks_total": len(checks),
        "topic_stability": stability_state,
        "graph_lexical_intervention": graph_state,
        "parent_training_convergence": training_state,
        "case_study_preregistration": case_state,
        "comparator_diagnostics": diagnostic_state,
        "frozen_inventory_artifacts": inventory_count,
        "figures": {
            "main_candidate": [
                "paper/figure_topic_stability.svg",
                "paper/figure_graph_lexical_intervention.svg",
            ],
            "supplement": [
                "supplement/figure_parent_training_convergence.svg",
            ],
        },
        "performance_table_replot_created": False,
        "baseline_or_sota_metric_values_used_for_figures": False,
        "human_evaluation_status": contract10["human_evaluation"]["status"],
        "external_evaluation_status": contract10["external_evaluation"]["status"],
        "automatic_evidence_complete": True,
        "submission_readiness_claimed": False,
        "model_fits": 0,
        "optimizer_steps": 0,
        "model_or_hyperparameter_changes": 0,
        "comparator_refits": 0,
        "frozen_result_changes": 0,
        "seed_selection": 0,
        "test_text_accesses": 0,
        "human_or_external_results_invented": 0,
        "unfavorable_results_omitted": False,
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    _write_json(out / "step11_contract.json", contract)
    _manifest(out)
    zip_path = _return_zip(root, out)
    print(f"[V6.1 Step 11] PASS | hard checks={len(checks)}/{len(checks)}", flush=True)
    print(
        "[V6.1 Step 11] distinct evidence | "
        f"graph-new-words/topic={graph_state['mean_introduced_words_per_topic']:.2f} "
        f"final-normalized-loss={training_state['final_median_normalized_loss']:.6f} "
        f"case-grid={case_state['topics']}x{case_state['cities']}",
        flush=True,
    )
    print(
        "[V6.1 Step 11] topic stability | "
        f"aligned mean RBO={stability_state['mean_rbo_p_0_9_excluding_self']:.6f} "
        f"minimum={stability_state['minimum_rbo_p_0_9_excluding_self']:.6f}",
        flush=True,
    )
    print(
        "[V6.1 Step 11] pending independent evidence | "
        f"human={contract['human_evaluation_status']} "
        f"external={contract['external_evaluation_status']}",
        flush=True,
    )
    print(f"[V6.1 Step 11] contract={out / 'step11_contract.json'}", flush=True)
    print(f"[V6.1 Step 11] return-zip={zip_path}", flush=True)
    return out
