from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.v6.step11b import (
    IMPLEMENTATION,
    _graph_svg,
    _load_protocol,
    _translation_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/step11b.yaml"


def test_step11b_protocol_preserves_source_and_all_reporting_firewalls():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert len(protocol["translation"]["mappings"]) == 10
    assert protocol["outputs"]["english_is_main_paper_candidate"] is True
    assert protocol["outputs"]["chinese_is_source_record"] is True
    assert protocol["policy"]["model_fit_allowed"] is False
    assert protocol["policy"]["test_text_access_allowed"] is False
    assert protocol["policy"]["translation_presented_as_human_evidence_allowed"] is False
    assert protocol["policy"]["source_chinese_omission_allowed"] is False


def test_step11b_translation_requires_exact_frozen_chinese_previews():
    protocol = _load_protocol(CONFIG)
    source = [
        {
            "topic_id": row["topic_id"],
            "introduced_word_count": row["topic_id"],
            "introduced_words_preview": row["source_zh"],
        }
        for row in protocol["translation"]["mappings"]
    ]
    translated = _translation_rows(source, protocol)
    assert [row["topic_id"] for row in translated] == list(range(1, 11))
    assert all(row["numeric_evidence_changed"] is False for row in translated)
    assert all(row["counted_as_human_judgment"] is False for row in translated)


def test_step11b_bilingual_figures_are_valid_and_language_separated(tmp_path):
    rows = [
        {
            "topic_id": topic,
            "introduced_word_count": topic,
            "source_zh": f"中文词{topic}",
            "editorial_en": f"English term {topic}",
        }
        for topic in range(1, 11)
    ]
    english = tmp_path / "english.svg"
    chinese = tmp_path / "chinese.svg"
    _graph_svg(english, rows, language="en")
    _graph_svg(chinese, rows, language="zh")
    ET.parse(english)
    ET.parse(chinese)
    assert "中文" not in english.read_text(encoding="utf-8")
    assert "English term" not in chinese.read_text(encoding="utf-8")


def test_main_and_runner_expose_step11b_without_installing_packages():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts/run_step11b.ps1").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-step11b.txt").read_text(encoding="utf-8")
    assert '"step11b"' in source and "run_v6_step11b" in source
    assert "pip install" not in script.casefold()
    assert "conda" not in script.casefold()
    assert "installs nothing" in requirements.casefold()

