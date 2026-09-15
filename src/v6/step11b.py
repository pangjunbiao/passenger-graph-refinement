"""V6.1 Step 11B: bilingual rendering of the frozen graph audit.

This reporting-only stage verifies a manifest-valid Step-11 R2 result and
renders its graph lexical-intervention figure in English and Chinese.  It
does not read model data, fit anything, inspect test text, change any number,
or treat an editorial translation as human evidence.
"""

from __future__ import annotations

import csv
import html
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.v6.data_contract import sha256_file, sha256_object


IMPLEMENTATION = "v6_1_step11b_bilingual_graph_figure_r1"
SCOPE = "manifest_verified_bilingual_editorial_rendering_no_model_or_evidence_change"
READINESS = "READY_FOR_BILINGUAL_FIGURE_USE_WITH_TRANSLATION_DISCLOSURE"


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


def _verify_manifest(base: Path) -> dict[str, Any]:
    manifest_path = base / "generated_artifact_manifest.json"
    manifest = _json(manifest_path)
    body = dict(manifest)
    identity = body.pop("manifest_identity", None)
    if not isinstance(identity, str) or sha256_object(body) != identity:
        raise ValueError(f"Artifact-manifest identity is invalid: {manifest_path}")
    for item in manifest.get("files", []):
        artifact = base / str(item["relative_path"])
        if not artifact.is_file():
            raise FileNotFoundError(f"Frozen artifact is missing: {artifact}")
        if sha256_file(artifact) != str(item["sha256"]):
            raise ValueError(f"Frozen artifact hash changed: {artifact}")
        if int(artifact.stat().st_size) != int(item["size_bytes"]):
            raise ValueError(f"Frozen artifact size changed: {artifact}")
    return manifest


def _load_protocol(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    protocol = document.get("v6", {}).get("step11b", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("Unsupported Step-11B configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-11B implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-11B scope changed")
    prerequisite = protocol["prerequisite"]
    if any(
        (
            prerequisite["implementation_version"]
            != "v6_1_step11_distinct_evidence_synthesis_r2",
            prerequisite["status"] != "PASS",
            int(prerequisite["source_seed"]) != 101,
            int(prerequisite["topics"]) != 10,
            int(prerequisite["frozen_result_changes"]) != 0,
        )
    ):
        raise ValueError("Step-11B prerequisite lock changed")
    mappings = protocol["translation"]["mappings"]
    if [int(row["topic_id"]) for row in mappings] != list(range(1, 11)):
        raise ValueError("Step-11B requires exactly one translation for T01--T10")
    if len({str(row["source_zh"]) for row in mappings}) != 10:
        raise ValueError("Duplicate Chinese source previews are not allowed")
    policy = protocol["policy"]
    false_locks = (
        "model_fit_allowed",
        "optimizer_steps_allowed",
        "test_text_access_allowed",
        "frozen_result_change_allowed",
        "translation_presented_as_human_evidence_allowed",
        "source_chinese_omission_allowed",
    )
    if any(policy[key] is not False for key in false_locks):
        raise ValueError("A Step-11B reporting firewall was relaxed")
    return protocol


def _latest_step11(root: Path, protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    expected = protocol["prerequisite"]
    candidates = sorted(
        (root / "outputs/v6/step11_frozen_evidence").glob("*/step11_contract.json"),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = _json(contract_path)
            graph = contract.get("graph_lexical_intervention", {})
            if not (
                contract.get("implementation_version")
                == expected["implementation_version"]
                and contract.get("status") == expected["status"]
                and _verify_fingerprint(contract, "contract_fingerprint")
                and int(contract.get("frozen_result_changes", -1))
                == int(expected["frozen_result_changes"])
                and int(contract.get("model_fits", -1)) == 0
                and int(contract.get("optimizer_steps", -1)) == 0
                and int(contract.get("test_text_accesses", -1)) == 0
                and int(graph.get("source_seed", -1)) == int(expected["source_seed"])
                and int(graph.get("topics", -1)) == int(expected["topics"])
            ):
                continue
            _verify_manifest(contract_path.parent)
            source = (
                contract_path.parent
                / "supplement/graph_lexical_intervention_by_topic.csv"
            )
            if not source.is_file():
                continue
            return contract_path.parent, contract
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError("No manifest-valid Step-11 R2 graph audit was found")


def _translation_rows(
    source_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if len(source_rows) != 10:
        raise ValueError("The frozen graph audit must contain ten topic rows")
    mapping = {
        int(row["topic_id"]): row for row in protocol["translation"]["mappings"]
    }
    translated: list[dict[str, Any]] = []
    for source in sorted(source_rows, key=lambda row: int(row["topic_id"])):
        topic_id = int(source["topic_id"])
        expected = mapping.get(topic_id)
        if expected is None:
            raise ValueError(f"Missing translation for T{topic_id:02d}")
        source_text = str(source["introduced_words_preview"]).strip()
        if source_text != str(expected["source_zh"]).strip():
            raise ValueError(
                f"Frozen Chinese preview changed for T{topic_id:02d}: {source_text!r}"
            )
        translated.append(
            {
                "topic_id": topic_id,
                "introduced_word_count": int(source["introduced_word_count"]),
                "source_zh": source_text,
                "editorial_en": str(expected["english"]).strip(),
                "translation_status": protocol["translation"]["status"],
                "numeric_evidence_changed": False,
                "counted_as_human_judgment": False,
            }
        )
    if [row["topic_id"] for row in translated] != list(range(1, 11)):
        raise ValueError("The translated topic grid is incomplete")
    return translated


def _contains_han(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def _graph_svg(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    language: str,
) -> None:
    if language not in {"en", "zh"}:
        raise ValueError("language must be 'en' or 'zh'")
    if language == "en":
        title = "What graph projection changes in each frozen topic"
        subtitle = (
            "First prespecified seed (101), same topic index; bars count newly "
            "introduced words among the final top 10."
        )
        prefix = "New:"
        axis = "Introduced words in graph-transformed top 10"
        footer = (
            "Mechanistic post-hoc audit; word changes are not an additional "
            "performance endpoint."
        )
        previews = [str(row["editorial_en"]) for row in rows]
        xml_title = "Graph projection changes to frozen topic descriptors"
        xml_desc = "New top-ten words introduced for each frozen seed-101 topic."
    else:
        title = "图投影对各冻结主题描述词的影响"
        subtitle = "首个预设随机种子（101），主题索引保持不变；条形表示最终前10个词中新引入词的数量。"
        prefix = "新增："
        axis = "图变换后前10个词中的新增词数"
        footer = "机制性事后审计；词汇变化不是额外的性能指标。"
        previews = [str(row["source_zh"]) for row in rows]
        xml_title = "图投影对冻结主题描述词的影响"
        xml_desc = "首个预设随机种子中各主题新引入的前十词数量。"
    width, height = 1620, 750
    left, top, bar_width, row_height = 145, 125, 500, 50
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(xml_title)}</title>",
        f"<desc>{html.escape(xml_desc)}</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#111827}'
        '.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#4b5563}'
        '.label{font-size:14px}.value{font-size:14px;font-weight:700}'
        '.words{font-size:14px;fill:#374151}.axis{font-size:12px;fill:#6b7280}</style>',
        f'<text x="28" y="39" class="title">{html.escape(title)}</text>',
        f'<text x="28" y="69" class="sub">{html.escape(subtitle)}</text>',
    ]
    for tick in range(0, 11, 2):
        x = left + bar_width * tick / 10
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{top + row_height * len(rows)}" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="{top - 24}" text-anchor="middle" class="axis">{tick}</text>',
            ]
        )
    for index, (row, preview) in enumerate(zip(rows, previews)):
        y = top + index * row_height
        value = int(row["introduced_word_count"])
        fill_width = bar_width * value / 10
        value_x = left + max(fill_width - 10, 18)
        lines.extend(
            [
                f'<text x="{left - 15}" y="{y + 29}" text-anchor="end" class="label">T{int(row["topic_id"]):02d}</text>',
                f'<rect x="{left}" y="{y + 7}" width="{bar_width}" height="29" rx="5" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y + 7}" width="{fill_width:.1f}" height="29" rx="5" fill="#2563eb"/>',
                f'<text x="{value_x:.1f}" y="{y + 28}" text-anchor="end" class="value" style="fill:#ffffff">{value}</text>',
                f'<text x="{left + bar_width + 30}" y="{y + 28}" class="words">{html.escape(prefix)} {html.escape(preview)}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{left + bar_width / 2}" y="{top + row_height * len(rows) + 47}" text-anchor="middle" class="axis">{html.escape(axis)}</text>',
            f'<text x="28" y="726" class="axis">{html.escape(footer)}</text>',
            "</svg>",
        ]
    )
    _write_text(path, "\n".join(lines))
    ET.parse(path)


def _manifest(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "generated_artifact_manifest.json":
            continue
        files.append(
            {
                "relative_path": path.relative_to(out).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    result: dict[str, Any] = {"schema_version": 1, "files": files}
    result["manifest_identity"] = sha256_object(result)
    _write_json(out / "generated_artifact_manifest.json", result)
    return result


def _return_zip(root: Path, out: Path) -> Path:
    destination = root / "v6_step11_bilingual_figures.zip"
    if destination.is_file():
        destination.unlink()
    prefix = Path("step11_bilingual_figures") / out.name
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, (prefix / path.relative_to(out)).as_posix())
    return destination


def run_v6_step11b(*, project_root: Path, config_path: Path) -> Path:
    """Render English and Chinese versions from immutable Step-11 evidence."""

    root = project_root.expanduser().resolve()
    protocol = _load_protocol(config_path)
    print(
        f"[V6.1 Step 11B] START | implementation={IMPLEMENTATION} | "
        "fits=0 optimizer-steps=0 frozen-result-changes=0 test-text=0",
        flush=True,
    )
    step11, contract11 = _latest_step11(root, protocol)
    source_csv = step11 / "supplement/graph_lexical_intervention_by_topic.csv"
    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    rows = _translation_rows(source_rows, protocol)

    out = root / "outputs/v6/step11_bilingual_figures" / _utc_id()
    out.mkdir(parents=True, exist_ok=False)
    en_path = out / protocol["outputs"]["english"]
    zh_path = out / protocol["outputs"]["chinese"]
    audit_path = out / protocol["outputs"]["translation_audit"]
    _graph_svg(en_path, rows, language="en")
    _graph_svg(zh_path, rows, language="zh")
    _write_csv(audit_path, rows)

    english_svg = en_path.read_text(encoding="utf-8")
    chinese_svg = zh_path.read_text(encoding="utf-8")
    checks = [
        ("source_step11_contract_and_manifest_valid", True, contract11["contract_fingerprint"]),
        ("source_graph_csv_hash_recorded", True, sha256_file(source_csv)),
        ("ten_frozen_topic_rows_retained", len(rows) == 10, len(rows)),
        ("topic_order_retained", [row["topic_id"] for row in rows] == list(range(1, 11)), [row["topic_id"] for row in rows]),
        ("all_numeric_counts_retained", [int(row["introduced_word_count"]) for row in rows] == [int(row["introduced_word_count"]) for row in source_rows], [row["introduced_word_count"] for row in rows]),
        ("english_figure_contains_no_han_characters", not _contains_han(english_svg), "English-only rendering"),
        ("chinese_figure_retains_all_source_previews", all(str(row["source_zh"]) in chinese_svg for row in rows), "10/10 previews"),
        ("both_svg_files_are_well_formed", True, "XML parsed"),
        ("translation_not_counted_as_human_evidence", all(row["counted_as_human_judgment"] is False for row in rows), False),
        ("model_fit_zero", True, 0),
        ("test_text_access_zero", True, 0),
        ("frozen_result_changes_zero", True, 0),
    ]
    hard_rows = [
        {
            "check": name,
            "required": True,
            "observed": observed,
            "status": "PASS" if passed else "FAIL",
        }
        for name, passed, observed in checks
    ]
    _write_csv(out / "hard_checks.csv", hard_rows)
    failures = [row["check"] for row in hard_rows if row["status"] == "FAIL"]
    status = "PASS" if not failures else "FAIL"
    lineage = {
        "schema_version": 1,
        "source_step11_directory": str(step11),
        "source_step11_contract_sha256": sha256_file(step11 / "step11_contract.json"),
        "source_step11_contract_fingerprint": contract11["contract_fingerprint"],
        "source_graph_csv_sha256": sha256_file(source_csv),
        "source_graph_manifest_identity": _json(
            step11 / "generated_artifact_manifest.json"
        )["manifest_identity"],
        "read_only_use": True,
    }
    _write_json(out / "frozen_lineage.json", lineage)
    _write_text(
        out / "paper/TRANSLATION_DISCLOSURE.md",
        """# Figure translation disclosure

The English keyword glosses are an AI-assisted editorial translation of the
frozen Chinese strings. The Chinese source figure and a row-level translation
audit are retained. Translation changed no topic, count, score, case selection,
or human rating and is not presented as human evidence. The authors should
fact-check the English glosses before submission and disclose the translation
method in the manuscript.
""",
    )
    contract = {
        "schema_version": 1,
        "status": status,
        "readiness": READINESS if status == "PASS" else "BLOCKED",
        "scope": SCOPE,
        "implementation_version": IMPLEMENTATION,
        "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
        "hard_checks_total": len(hard_rows),
        "failed_checks": failures,
        "source_step11_contract_fingerprint": contract11["contract_fingerprint"],
        "source_seed": 101,
        "topics": 10,
        "english_figure": protocol["outputs"]["english"],
        "chinese_figure": protocol["outputs"]["chinese"],
        "translation_status": protocol["translation"]["status"],
        "translation_is_human_evidence": False,
        "numeric_evidence_changes": 0,
        "model_fits": 0,
        "optimizer_steps": 0,
        "test_text_accesses": 0,
        "frozen_result_changes": 0,
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    _write_json(out / "step11b_contract.json", contract)
    _manifest(out)
    return_zip = _return_zip(root, out)
    print(
        f"[V6.1 Step 11B] {status} | hard checks="
        f"{contract['hard_checks_passed']}/{contract['hard_checks_total']} | "
        "English+Chinese figures complete",
        flush=True,
    )
    print(f"[V6.1 Step 11B] contract={out / 'step11b_contract.json'}", flush=True)
    print(f"[V6.1 Step 11B] return-zip={return_zip}", flush=True)
    if failures:
        raise AssertionError(f"Step 11B failed: {failures}")
    return out

