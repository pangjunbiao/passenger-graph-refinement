from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.v6.step10 import (
    IMPLEMENTATION,
    _ablation_rows,
    _choose_intruder,
    _coincidence_alpha,
    _case_study_packets,
    _formatted,
    _external_packet,
    _human_returns,
    _assert_locked_columns,
    _load_protocol,
    _normalize_yes_no,
    _rating,
)
from src.v6.data_contract import sha256_object


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/step10.yaml"


def test_step10_protocol_is_reporting_only_and_keeps_all_firewalls_closed():
    protocol = _load_protocol(CONFIG)
    assert protocol["implementation_version"] == IMPLEMENTATION
    assert protocol["reporting"]["seeds"] == [
        101, 211, 307, 401, 503, 601, 701, 809, 907, 1009
    ]
    assert protocol["human_evaluation"]["blinded_seed"] == 101
    assert protocol["human_evaluation"]["required_annotators"] == 3
    assert protocol["policy"]["model_fit_allowed"] is False
    assert protocol["policy"]["omit_unfavorable_metric_allowed"] is False
    assert protocol["policy"]["test_documents_allowed_for_case_example_selection"] is False
    assert protocol["external_evaluation"]["required_model_state"] == (
        "exact_state_compatibility_diagnostic_plus_architecture_replication"
    )


def test_step10_r3_accepts_hash_bound_cross_language_replication_envelope(tmp_path):
    input_dir = tmp_path / "inputs/v6/step10/external"
    input_dir.mkdir(parents=True)
    evidence = input_dir / "dataset_audit.json"
    evidence.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "relative_path": evidence.name,
                "sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest(),
                "size_bytes": evidence.stat().st_size,
            }
        ],
    }
    manifest["manifest_identity"] = sha256_object(manifest)
    (input_dir / "generated_artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    contract = {
        "schema_version": 2,
        "status": "PASS",
        "evidence_kind": "independent_external_architecture_replication_with_frozen_hyperparameters",
        "dataset_ids": ["external"],
        "dataset_independent_of_internal_772_document_corpus": True,
        "external_language": "English",
        "internal_language": "Chinese",
        "step7_freeze_identity": "freeze",
        "exact_frozen_state_assessment": "NOT_ESTIMABLE_LANGUAGE_VOCABULARY_MISMATCH",
        "exact_state_coverage_fraction": 0.0,
        "exact_state_performance_claimed": False,
        "architecture_unchanged": True,
        "model_and_training_hyperparameters_frozen_before_external_outcomes": True,
        "language_compatible_fit_vocabulary_and_embeddings_reestimated": True,
        "v6_model_fits": 10,
        "v6_optimizer_update_steps": 3600,
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
        "artifact_manifest_identity": manifest["manifest_identity"],
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    (input_dir / "external_evaluation_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    state = _external_packet(
        tmp_path,
        tmp_path / "out",
        {"step7_freeze_identity": "freeze"},
    )
    assert state["complete"] is True
    assert state["status"] == "COMPLETE_REPORTED_AS_OBSERVED"


def test_main_exposes_all_final_stages():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    for stage in ("step7", "step8", "step9", "step10"):
        assert f'"{stage}"' in source
        assert f"run_v6_{stage}" in source


def test_formatting_preserves_noncomparable_dash_and_zero():
    assert _formatted(np.nan, np.nan, 4) == "—"
    assert _formatted(1.0e-18, 2.0e-18, 4) == "0.0000 ± 0.0000"
    assert _formatted(0.123456, 0.01004, 4) == "0.1235 ± 0.0100"


def test_ablation_table_keeps_cross_metric_tradeoffs_visible():
    variants = [
        "full_graph_calibration_pooled",
        "without_graph_pooled_retained",
        "without_pooled_graph_calibration_retained",
        "without_calibration_graph_pooled_retained",
        "exact_official_parent",
    ]
    rows = []
    for index, variant in enumerate(variants):
        row = {"variant": variant}
        for metric in (
            "npmi_at_10", "c_v_at_10", "topic_diversity_at_10",
            "top_word_redundancy_at_10", "macro_city_nll_per_token",
            "micro_nll_per_token",
        ):
            row[f"{metric}_mean"] = float(index)
            row[f"{metric}_sample_sd"] = 0.1
        rows.append(row)
    rendered = _ablation_rows(pd.DataFrame(rows), variants)
    assert len(rendered) == 5
    assert rendered[1]["Prespecified target"] == "NPMI@10 and C_v@10"
    assert rendered[-1]["Contrast type"] == "Architectural parent control"
    assert "Macro-city NLL/token ↓" in rendered[1]


def test_intruder_never_comes_from_the_target_topic():
    top = np.arange(100, dtype=np.int64).reshape(10, 10)
    df = np.arange(1, 101, dtype=float)
    intruder = _choose_intruder(top, 3, df)
    assert intruder not in set(top[3])
    assert intruder in set(top.reshape(-1))


def test_krippendorff_alpha_is_one_for_perfect_agreement():
    nominal = {"a": ["yes", "yes", "yes"], "b": ["no", "no", "no"]}
    ordinal = {"a": [1, 1, 1], "b": [4, 4, 4], "c": [5, 5, 5]}
    assert _coincidence_alpha(nominal) == 1.0
    assert _coincidence_alpha(ordinal, ordered_categories=[1, 2, 3, 4, 5]) == 1.0


def test_krippendorff_alpha_is_not_estimable_without_any_rating_variation():
    assert _coincidence_alpha({"a": ["yes", "yes"], "b": ["yes", "yes"]}) is None


def test_returned_form_locked_fields_must_match_issued_form():
    issued = pd.DataFrame([{"item_id": "W001", "word_1": "safe"}])
    returned = issued.copy()
    _assert_locked_columns(returned, issued, locked_columns=["word_1"], label="test")
    returned.loc[0, "word_1"] = "changed"
    try:
        _assert_locked_columns(returned, issued, locked_columns=["word_1"], label="test")
    except ValueError as exc:
        assert "locked field" in str(exc)
    else:
        raise AssertionError("A modified locked field was accepted")


def test_annotation_values_are_strictly_validated():
    assert _normalize_yes_no("YES") == "yes"
    assert _normalize_yes_no("0") == "no"
    assert _rating("5") == 5


def test_case_selection_uses_all_cities_but_never_test_documents(tmp_path):
    protocol = _load_protocol(CONFIG)
    source = tmp_path / "source/data/processed/internal"
    source.mkdir(parents=True)
    documents = []
    cities = ["beijing", "shanghai", "xiamen"]
    splits = ["train"] * 620 + ["validation"] * 75 + ["test"] * 77
    for index, split in enumerate(splits):
        topic = index % 10
        documents.append(
            {
                "post_id": f"post-{index}",
                "city": cities[index % 3],
                "split": split,
                "published_date": "2025-01-01",
                "year_status": "known",
                "delexicalized_text": f"safe delexicalized example {index}",
                "tokens": [f"w{topic * 10}", f"w{topic * 10 + 1}"],
                "model_token_count": 2,
            }
        )
    with (source / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for row in documents:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    step8 = tmp_path / "step8/worker"
    step8.mkdir(parents=True)
    variants = protocol["reporting"]["main_table_ablations"]
    city_rows = []
    for variant_number, variant in enumerate(variants):
        for seed in (101, 211):
            for city in cities:
                city_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "city": city,
                        "documents": 25,
                        "tokens": 100,
                        "nll_per_token": 6.0 + 0.1 * variant_number + seed / 100000,
                    }
                )
    pd.DataFrame(city_rows).to_csv(step8 / "per_city_variant_metrics.csv", index=False)
    human_state = {
        "method_topics": {"v6_1": np.arange(100, dtype=np.int64).reshape(10, 10)},
        "vocabulary": [f"w{index}" for index in range(1148)],
    }
    out = tmp_path / "out"
    result = _case_study_packets(
        out,
        tmp_path,
        tmp_path / "source",
        step8.parent,
        human_state,
        protocol,
    )
    assert result["source_documents_available"] is True
    assert result["case_items"] == 30
    assert result["test_documents_used_for_selection"] == 0
    audit = pd.read_csv(out / "supplement/case_study_candidate_audit.csv")
    assert set(audit["selection_partition"]) <= {"train", "validation"}
    assert set(audit["city"]) == set(cities)


def test_completed_independent_human_forms_are_validated_and_summarized(tmp_path):
    protocol = _load_protocol(CONFIG)
    out = tmp_path / "out"
    private = out / "private_do_not_share_with_annotators"
    private.mkdir(parents=True)
    pd.DataFrame(
        [
            {"item_id": "W001", "method_id": "v6_1", "source_topic": 0, "intruder_word": "six"},
            {"item_id": "W002", "method_id": "encot", "source_topic": 1, "intruder_word": "one"},
        ]
    ).to_csv(private / "word_intrusion_answer_key.csv", index=False)
    pd.DataFrame(
        [
            {"item_id": "T001", "source_topic": 0},
            {"item_id": "T002", "source_topic": 1},
        ]
    ).to_csv(private / "topic_interpretation_key.csv", index=False)
    inputs = tmp_path / "inputs/v6/step10/human"
    inputs.mkdir(parents=True)
    issued = out / "human_packet_share_only_this_folder"
    issued.mkdir(parents=True)
    for annotator in range(1, 4):
        word_return = pd.DataFrame(
            [
                {
                    "item_id": "W001", "word_1": "one", "word_2": "two",
                    "word_3": "three", "word_4": "four", "word_5": "five",
                    "word_6": "six", "selected_intruder_word": "six",
                    "coherence_1_to_5": 5, "passenger_requirement_yes_no": "yes",
                    "short_topic_label": "service need", "notes": "",
                },
                {
                    "item_id": "W002", "word_1": "one", "word_2": "two",
                    "word_3": "three", "word_4": "four", "word_5": "five",
                    "word_6": "six", "selected_intruder_word": "one",
                    "coherence_1_to_5": 4, "passenger_requirement_yes_no": "yes",
                    "short_topic_label": "another need", "notes": "",
                },
            ]
        )
        word_return.to_csv(inputs / f"annotator_{annotator:02d}_word_intrusion.csv", index=False)
        word_return[
            ["item_id", "word_1", "word_2", "word_3", "word_4", "word_5", "word_6"]
        ].to_csv(issued / f"annotator_{annotator:02d}_word_intrusion.csv", index=False)
        topic_return = pd.DataFrame(
            [
                {
                    "item_id": "T001", "top_10_words": "one、two、three",
                    "passenger_requirement_yes_no": "yes",
                    "service_category": "operations_reliability", "topic_label": "frequency",
                    "coherence_1_to_5": 5, "actionability_1_to_5": 5,
                    "suggested_transport_action": "adjust headways", "notes": "",
                },
                {
                    "item_id": "T002", "top_10_words": "four、five、six",
                    "passenger_requirement_yes_no": "yes",
                    "service_category": "safety_security", "topic_label": "safety",
                    "coherence_1_to_5": 4, "actionability_1_to_5": 4,
                    "suggested_transport_action": "audit safety", "notes": "",
                },
            ]
        )
        topic_return.to_csv(inputs / f"annotator_{annotator:02d}_topic_interpretation.csv", index=False)
        topic_return[["item_id", "top_10_words"]].to_csv(
            issued / f"annotator_{annotator:02d}_topic_interpretation.csv", index=False
        )
    state = _human_returns(tmp_path, out, protocol, {"case_items": 0})
    assert state["complete"] is True
    assert state["word_intrusion_ratings"] == 6
    summary = pd.read_csv(out / "paper/human_method_summary.csv")
    assert set(summary["method_id"]) == {"v6_1", "encot"}
    assert summary["word_intrusion_accuracy"].eq(1.0).all()
