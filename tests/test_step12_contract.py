from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from src.v6.step12 import (
    IMPLEMENTATION,
    _case_cell_and_selection,
    _city_map_svg,
    _cluster_bootstrap_mean,
    _cross_city_descriptor_support,
    _load_protocol,
    _three_annotator_material,
    _two_of_three_majority,
    _verify_preregistration_bridge,
)
from src.v6.data_contract import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/step12.yaml"


def test_step12_protocol_is_reporting_only_and_requires_no_new_human_input():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["prerequisite"]["step11_preregistration_identity"] == (
        "6f65f886481ebdf06412ac24a484cae0e45b206f8c10ee3817602ce36ff62a73"
    )
    assert protocol["policy"]["model_fit_allowed"] is False
    assert protocol["policy"]["test_text_access_allowed"] is False
    assert protocol["policy"]["forced_consensus_allowed"] is False
    assert protocol["policy"]["new_human_input_allowed"] is False
    assert protocol["policy"]["synthetic_human_response_allowed"] is False
    assert protocol["policy"]["llm_presented_as_human_allowed"] is False
    assert protocol["policy"]["generated_consensus_text_allowed"] is False
    assert protocol["protocol_amendment"]["original_case_selection_rule_changed"] is False
    assert protocol["policy"]["cross_city_occurrence_presented_as_transfer_allowed"] is False


def test_main_exposes_step12_without_removing_prior_final_stages():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    for stage in ("step7", "step8", "step9", "step10", "step11", "step12"):
        assert f'"{stage}"' in source
        assert f"run_v6_{stage}" in source


def test_item_cluster_bootstrap_is_deterministic_and_bounded():
    frame = pd.DataFrame(
        {
            "item_id": ["a", "a", "a", "b", "b", "b"],
            "score": [1, 1, 1, 0, 0, 0],
        }
    )
    first = _cluster_bootstrap_mean(
        frame,
        item_column="item_id",
        value_column="score",
        seed=42,
        replicates=1000,
    )
    second = _cluster_bootstrap_mean(
        frame,
        item_column="item_id",
        value_column="score",
        seed=42,
        replicates=1000,
    )
    assert first == second
    assert first[0] == 0.5
    assert 0.0 <= first[1] <= first[0] <= first[2] <= 1.0


def test_preregistered_case_selection_uses_all_cells_and_exact_tie_break(tmp_path):
    protocol = _load_protocol(CONFIG)
    case_rows = []
    key_rows = []
    item = 0
    for topic in range(10):
        for city in ("beijing", "shanghai", "xiamen"):
            item += 1
            item_id = f"C{item:03d}"
            for annotator in (1, 2, 3):
                case_rows.append(
                    {
                        "annotator": annotator,
                        "item_id": item_id,
                        "city": city,
                        "topic_fit_1_to_5": 5,
                        "service_need_present_yes_no": "yes",
                    }
                )
            key_rows.append(
                {
                    "item_id": item_id,
                    "source_topic": topic,
                    "top_10_words": "、".join(f"w{x}" for x in range(10)),
                    "lexical_relevance_score": float(topic),
                    "selection_partition": "train",
                    "selection_rule": "frozen",
                }
            )
    summary, selected = _case_cell_and_selection(
        tmp_path,
        {"case": pd.DataFrame(case_rows), "case_key": pd.DataFrame(key_rows)},
        protocol,
    )
    assert len(summary) == 30
    assert len(selected) == 3
    assert set(selected["source_topic"]) == {9}
    assert set(selected["city"]) == {"beijing", "shanghai", "xiamen"}


def test_cross_city_descriptor_support_is_descriptive_and_complete(tmp_path):
    documents = []
    for city in ("beijing", "shanghai", "xiamen"):
        documents.append({"city": city, "tokens": ["shared", f"only_{city}"]})
    key = pd.DataFrame(
        [
            {
                "source_topic": topic,
                "top_10_words": "、".join(["shared"] + [f"t{topic}w{x}" for x in range(9)]),
            }
            for topic in range(10)
            for _ in range(3)
        ]
    )
    state = _cross_city_descriptor_support(tmp_path, documents, key)
    assert state["topics"] == 10
    assert state["descriptor_city_rows"] == 300
    assert state["claim_scope"] == "lexical_support_only_not_transfer"
    summary = pd.read_csv(tmp_path / "paper/cross_city_supported_descriptors.csv")
    assert summary["descriptors_observed_in_all_three_cities_count"].eq(1).all()


def test_city_map_svg_is_valid_and_retains_all_30_cells(tmp_path):
    rows = []
    for topic in range(1, 11):
        row = {
            "source_topic": topic,
            "majority_service_category_display": f"Category {topic}",
        }
        for city in ("beijing", "shanghai", "xiamen"):
            row[f"{city}_median_topic_fit"] = float(1 + topic % 5)
            row[f"{city}_service_need_yes_votes"] = 3
        rows.append(row)
    destination = tmp_path / "map.svg"
    _city_map_svg(destination, pd.DataFrame(rows))
    ET.parse(destination)
    text = destination.read_text(encoding="utf-8")
    assert text.count("  ●") == 30
    assert "development-only" in text


def test_two_of_three_majority_never_invents_a_category():
    assert _two_of_three_majority(["safety", "safety", "comfort"]) == (
        "safety",
        2,
    )
    assert _two_of_three_majority(["a", "b", "c"]) == ("no_consensus", 1)


def test_three_annotator_material_retains_every_original_string(tmp_path):
    protocol = _load_protocol(CONFIG)
    topic_rows = []
    topic_key = []
    topic_public = []
    case_rows = []
    case_public = []
    selected_rows = []
    for topic in range(10):
        item_id = f"T{topic + 1:03d}"
        topic_key.append({"item_id": item_id, "source_topic": topic, "seed": 101})
        topic_public.append(
            {"item_id": item_id, "top_10_words": "、".join(f"w{x}" for x in range(10))}
        )
        for annotator in (1, 2, 3):
            topic_rows.append(
                {
                    "annotator": annotator,
                    "item_id": item_id,
                    "service_category": "safety_security",
                    "topic_label": f"topic-{topic}-label-{annotator}",
                    "suggested_transport_action": f"topic-{topic}-action-{annotator}",
                }
            )
    for index, city in enumerate(("beijing", "shanghai", "xiamen"), start=1):
        item_id = f"C{index:03d}"
        case_public.append(
            {
                "item_id": item_id,
                "topic_words": "、".join(f"w{x}" for x in range(10)),
                "delexicalized_passenger_post": f"post-{city}",
            }
        )
        selected_rows.append(
            {
                "item_id": item_id,
                "city": city,
                "source_topic": index - 1,
                "median_topic_fit": 5.0,
                "mean_topic_fit": 5.0,
                "service_need_yes_votes": 3,
                "lexical_relevance_score": 10.0,
            }
        )
        for annotator in (1, 2, 3):
            case_rows.append(
                {
                    "annotator": annotator,
                    "item_id": item_id,
                    "case_requirement_label": f"case-{city}-label-{annotator}",
                    "recommended_transport_action": f"case-{city}-action-{annotator}",
                }
            )
    state, material = _three_annotator_material(
        tmp_path,
        {
            "topic": pd.DataFrame(topic_rows),
            "case": pd.DataFrame(case_rows),
            "topic_key": pd.DataFrame(topic_key),
            "topic_public": pd.DataFrame(topic_public),
            "case_public": pd.DataFrame(case_public),
        },
        pd.DataFrame(selected_rows),
        protocol,
    )
    assert state["new_human_files_required"] == 0
    assert state["generated_consensus_strings"] == 0
    assert state["original_free_text_strings_retained"] == 78
    assert len(material["topic"]) == 10
    assert len(material["case"]) == 3
    assert material["case"].loc[0, "annotator_03_transport_action_zh"].endswith("-3")


def test_step12_runner_installs_no_dependency_or_environment():
    script = (ROOT / "scripts/run_step12.ps1").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-step12.txt").read_text(encoding="utf-8")
    assert "pip install" not in script.lower()
    assert "conda" not in script.lower()
    assert "installs nothing" in requirements.lower()


def test_bridge_uses_step11_inventory_when_old_blank_form_was_completed_in_place(tmp_path):
    relative_public = "human_packet_share_only_this_folder/annotator_01_case_posts.csv"
    relative_private = "private_do_not_share_with_annotators/case_source_key.csv"
    expected = {
        relative_public: "item_id,locked,response\nC001,x,\n",
        relative_private: "item_id,source\nC001,train\n",
    }
    step11 = tmp_path / "step11"
    old_step10 = tmp_path / "old_step10"
    new_step10 = tmp_path / "new_step10"
    inventory_rows = []
    for relative, content in expected.items():
        new_path = new_step10 / relative
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(content, encoding="utf-8")
        inventory_rows.append(
            {
                "source_stage": "step10",
                "artifact": new_path.name,
                "sha256": sha256_file(new_path),
            }
        )
        old_path = old_step10 / relative
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text(
            content.replace("C001,x,", "C001,x,completed")
            if relative == relative_public
            else content,
            encoding="utf-8",
        )
    inventory_path = step11 / "supplement/frozen_evidence_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)

    rows = _verify_preregistration_bridge(step11, old_step10, new_step10)

    assert all(row["identical"] is True for row in rows)
    public = next(row for row in rows if row["relative_path"] == relative_public)
    assert public["original_step10_current_matches_inventory"] is False
    assert public["authoritative_pre_response_source"] == (
        "step11_manifest_valid_frozen_evidence_inventory"
    )
