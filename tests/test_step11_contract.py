from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from src.v6.step11 import (
    IMPLEMENTATION,
    _graph_intervention_svg,
    _heatmap_svg,
    _load_protocol,
    _rbo,
    _training_convergence_svg,
    _topic_stability,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/step11.yaml"


def test_step11_protocol_is_reporting_only_and_fail_closed():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["reporting"]["reference_seed"] == 101
    assert protocol["reporting"]["seeds"] == [
        101, 211, 307, 401, 503, 601, 701, 809, 907, 1009
    ]
    assert protocol["policy"]["model_fit_allowed"] is False
    assert protocol["policy"]["seed_selection_allowed"] is False
    assert protocol["policy"]["unfavorable_result_omission_allowed"] is False
    assert protocol["policy"]["test_text_access_allowed"] is False
    assert protocol["policy"]["performance_table_replot_allowed"] is False
    assert protocol["reporting"]["parent_reporting_family"] == "official_parent_affinity"


def test_main_exposes_step11_without_removing_prior_final_stages():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    for stage in ("step7", "step8", "step9", "step10", "step11"):
        assert f'"{stage}"' in source
        assert f"run_v6_{stage}" in source


def test_rbo_has_expected_identity_and_rank_sensitivity():
    reference = [1, 2, 3, 4, 5]
    assert _rbo(reference, reference, 0.9) == 1.0
    reversed_score = _rbo(reference, list(reversed(reference)), 0.9)
    assert 0.0 < reversed_score < 1.0
    assert _rbo(reference, [2, 1, 3, 4, 5], 0.9) > reversed_score


def test_topic_stability_aligns_permuted_topics_without_using_topic_ids():
    rows = []
    seeds = [101, 211]
    reference_topics = {
        0: [10, 11, 12],
        1: [20, 21, 22],
    }
    candidate_topics = {
        0: [20, 21, 22],
        1: [10, 11, 12],
    }
    for seed, topics in ((101, reference_topics), (211, candidate_topics)):
        for topic, words in topics.items():
            for rank, word in enumerate(words, start=1):
                rows.append(
                    {
                        "seed": seed,
                        "reporting_family": "full",
                        "topic": topic,
                        "rank": rank,
                        "vocabulary_index": word,
                        "token": f"w{word}",
                    }
                )
    detail, summary, overall = _topic_stability(
        pd.DataFrame(rows),
        seeds=seeds,
        reference_seed=101,
        topics=2,
        top_n=3,
        family="full",
        persistence=0.9,
    )
    candidate = [row for row in detail if row["seed"] == 211]
    assert [(row["reference_topic"], row["matched_topic"]) for row in candidate] == [
        (0, 1), (1, 0)
    ]
    assert all(row["top_word_set_jaccard"] == 1.0 for row in candidate)
    assert all(row["rank_biased_overlap_p_0_9"] == 1.0 for row in candidate)
    assert len(summary) == 2
    assert overall["aligned_topic_pairs_excluding_self"] == 2


def test_svg_heatmap_is_valid_xml_and_retains_na(tmp_path):
    destination = tmp_path / "figure.svg"
    _heatmap_svg(
        destination,
        title="Audited figure",
        subtitle="Gray means unavailable.",
        row_labels=["one", "two"],
        column_labels=["A", "B"],
        values=np.asarray([[1.0, np.nan], [2.0, 3.0]]),
        kind="rank",
        formatter=".0f",
        legend_left="best",
        legend_right="worst",
    )
    ET.parse(destination)
    assert ">NA<" in destination.read_text(encoding="utf-8")


def test_graph_intervention_figure_is_valid_xml(tmp_path):
    destination = tmp_path / "graph.svg"
    _graph_intervention_svg(
        destination,
        [
            {
                "topic_id": topic,
                "introduced_word_count": 2 + topic % 8,
                "introduced_words_preview": f"new-{topic}-a · new-{topic}-b",
            }
            for topic in range(1, 11)
        ],
    )
    ET.parse(destination)
    assert "Mechanistic post-hoc audit" in destination.read_text(encoding="utf-8")


def test_training_convergence_figure_is_valid_xml(tmp_path):
    destination = tmp_path / "convergence.svg"
    epochs = np.arange(1, 121)
    frame = pd.DataFrame(
        {
            "epoch": epochs,
            "q25_normalized_loss": 1.0 - 0.0040 * (epochs - 1),
            "median_normalized_loss": 1.0 - 0.0035 * (epochs - 1),
            "q75_normalized_loss": 1.0 - 0.0030 * (epochs - 1),
        }
    )
    _training_convergence_svg(destination, frame)
    ET.parse(destination)
    assert "10 prespecified seeds" in destination.read_text(encoding="utf-8")


def test_step11_runner_installs_no_dependency_or_environment():
    script = (ROOT / "scripts/run_step11.ps1").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-step11.txt").read_text(encoding="utf-8")
    assert "pip install" not in script.lower()
    assert "conda" not in script.lower()
    assert "installs nothing" in requirements.lower()
