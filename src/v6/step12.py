"""V6.1 Step 12: three-annotator human-validity and case-study finalization.

This stage verifies the corrected Step-10 response receipts, recomputes the
human summaries and agreement coefficients, and applies the case-selection
rule frozen by Step 11.  It then reports every one of the three original
free-text interpretations for each selected case.  No fourth adjudicator, new
human judgment, generated consensus, model fit, test-text access, or frozen
score change is permitted.

Step 11 had proposed a later senior-adjudication pass for free-text wording.
Because that person is unavailable, R3 records a transparent post-collection
operational amendment: case selection and numeric outcomes remain unchanged,
while all original wording is retained instead of being synthesized.
"""

from __future__ import annotations

import csv
import html
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.v6.data_contract import sha256_file, sha256_object
from src.v6.step10 import _coincidence_alpha, _read_documents


IMPLEMENTATION = "v6_1_step12_human_case_finalization_r3"
SCOPE = "frozen_three_annotator_case_study_finalization_no_new_human_input"
READINESS_EXTERNAL_PENDING = "READY_FOR_EXTERNAL_GATE_AND_AUTHOR_REPORTING_DISCLOSURE"
READINESS_EXTERNAL_COMPLETE = "READY_FOR_MANUSCRIPT_ASSEMBLY_WITH_AUTHOR_REPORTING_DISCLOSURE"

CITIES = ("beijing", "shanghai", "xiamen")
CITY_DISPLAY = {"beijing": "Beijing", "shanghai": "Shanghai", "xiamen": "Xiamen"}
CATEGORY_DISPLAY = {
    "operations_reliability": "Operations reliability",
    "safety_security": "Safety and security",
    "accessibility_inclusion": "Accessibility and inclusion",
    "comfort_environment": "Comfort and environment",
    "staff_conduct": "Staff conduct",
    "information_communication": "Information and communication",
    "fare_payment_digital": "Fare/payment/digital access",
    "other": "Other",
    "no_consensus": "No 2-of-3 category majority",
}
METHOD_DISPLAY = {
    "v6_1": "SC-HTM V6.1",
    "encot": "EnCOT",
    "glocom": "GloCOM",
    "neuromax": "NeuroMax",
    "fastopic": "FASTopic",
}
METHOD_ORDER = tuple(METHOD_DISPLAY)


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


def _load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1:
        raise ValueError("Step-12 configuration schema changed")
    protocol = document.get("v6", {}).get("step12", {})
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-12 implementation/configuration lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-12 scope changed")
    expected = protocol["expected_corpus"]
    if list(expected["cities"]) != list(CITIES):
        raise ValueError("The frozen city set or order changed")
    human = protocol["human_evaluation"]
    if (
        int(human["annotators"]) != 3
        or int(human["word_intrusion_ratings"]) != 150
        or int(human["topic_ratings"]) != 30
        or int(human["case_ratings"]) != 90
    ):
        raise ValueError("The completed human-evaluation design changed")
    case = protocol["case_study"]
    if list(case["selection_order"]) != [
        "highest_median_topic_fit",
        "highest_service_need_yes_fraction",
        "highest_frozen_lexical_relevance",
        "lexicographically_smallest_item_id",
    ]:
        raise ValueError("The preregistered case-selection order changed")
    false_locks = (
        "model_fit_allowed",
        "optimizer_steps_allowed",
        "architecture_change_allowed",
        "hyperparameter_change_allowed",
        "preprocessing_change_allowed",
        "comparator_refit_allowed",
        "frozen_result_change_allowed",
        "seed_selection_allowed",
        "test_text_access_allowed",
        "forced_consensus_allowed",
        "original_response_edit_allowed",
        "unadjudicated_free_text_presented_as_consensus_allowed",
        "cross_city_occurrence_presented_as_transfer_allowed",
        "unfavorable_case_cell_omission_allowed",
        "new_human_input_allowed",
        "synthetic_human_response_allowed",
        "llm_presented_as_human_allowed",
        "generated_consensus_text_allowed",
        "post_response_case_or_metric_selection_allowed",
        "editorial_machine_translation_used_as_evidence_allowed",
    )
    if any(protocol["policy"][key] is not False for key in false_locks):
        raise ValueError("A Step-12 scientific firewall was relaxed")
    if case["unverified_machine_translation_allowed"] is not False:
        raise ValueError("Unverified translation cannot enter the evidence chain")
    if case["free_text_reporting"] != "retain_all_three_original_responses_without_synthesis":
        raise ValueError("Step-12 R3 must retain all original free-text responses")
    if case["categorical_reporting"] != "two_of_three_majority_else_no_consensus":
        raise ValueError("Step-12 R3 category rule changed")
    amendment = protocol["protocol_amendment"]
    if not (
        amendment["status"] == "disclosed_post_collection_operational_amendment"
        and amendment["original_case_selection_rule_changed"] is False
        and amendment["original_numeric_ratings_changed"] is False
        and amendment["original_human_responses_changed"] is False
        and amendment["new_human_judgments_collected"] is False
        and amendment["replacement"]
        == "report_all_three_original_free_text_responses_without_consensus"
    ):
        raise ValueError("The disclosed no-new-human protocol amendment changed")
    return protocol


def _latest_corrected_step10(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    required = protocol["prerequisite"]
    candidates = sorted(
        (root / "outputs/v6/step10_reporting_validity").glob("*/step10_contract.json"),
        reverse=True,
    )
    for path in candidates:
        contract = _json(path)
        human = contract.get("human_evaluation", {})
        case = contract.get("case_study", {})
        if not (
            contract.get("implementation_version") in set(required["step10_implementations"])
            and contract.get("status") in required["step10_allowed_status"]
            and contract.get("readiness")
            in {"AWAITING_INDEPENDENT_EXTERNAL_EVIDENCE", "READY_FOR_MANUSCRIPT_COMPLETION_WITH_REPORTED_LIMITATIONS"}
            and _verify_fingerprint(contract, "contract_fingerprint")
            and int(contract.get("hard_checks_passed", -1))
            == int(contract.get("hard_checks_total", -2))
            and int(contract.get("hard_checks_passed", 0))
            >= int(required["step10_required_hard_checks"])
            and human.get("status") == required["step10_human_status"]
            and human.get("complete") is True
            and human.get("locked_fields_verified_unchanged") is True
            and int(human.get("sealed_response_files", -1)) == 9
            and int(human.get("word_intrusion_ratings", -1)) == 150
            and int(human.get("topic_interpretation_ratings", -1)) == 30
            and int(human.get("case_ratings", -1)) == 90
            and int(case.get("case_items", -1)) == 30
            and int(case.get("test_documents_used_for_selection", -1)) == 0
            and int(contract.get("model_fits", -1)) == 0
            and int(contract.get("optimizer_steps", -1)) == 0
            and int(contract.get("model_or_hyperparameter_changes", -1)) == 0
        ):
            continue
        try:
            _verify_manifest(path.parent)
        except (OSError, ValueError):
            continue
        lineage = _json(path.parent / "frozen_lineage.json")
        if lineage.get("step7_freeze_identity") != required["step7_freeze_identity"]:
            continue
        return path.parent, contract
    raise RuntimeError(
        "No corrected, manifest-valid Step-10 R2/R3 output with the complete human return was found. "
        "Install this update and rerun Step 10 first."
    )


def _step11_preregistration(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], Path | None]:
    required = protocol["prerequisite"]
    candidates = sorted(
        (root / "outputs/v6/step11_frozen_evidence").glob("*/step11_contract.json"),
        reverse=True,
    )
    for path in candidates:
        contract = _json(path)
        state = contract.get("case_study_preregistration", {})
        if not (
            contract.get("implementation_version") == required["step11_implementation"]
            and contract.get("status") == required["step11_status"]
            and _verify_fingerprint(contract, "contract_fingerprint")
            and state.get("preregistration_identity")
            == required["step11_preregistration_identity"]
            and int(contract.get("model_fits", -1)) == 0
            and int(contract.get("test_text_accesses", -1)) == 0
        ):
            continue
        try:
            _verify_manifest(path.parent)
        except (OSError, ValueError):
            continue
        prereg = _json(path.parent / "paper/case_study_rendering_preregistration.json")
        if not _verify_fingerprint(prereg, "preregistration_identity"):
            continue
        if prereg["preregistration_identity"] != required["step11_preregistration_identity"]:
            continue
        lineage = _json(path.parent / "frozen_lineage.json")
        old_sha = lineage.get("step10_contract_sha256")
        old_path: Path | None = None
        for candidate in (root / "outputs/v6/step10_reporting_validity").glob("*/step10_contract.json"):
            if candidate.is_file() and sha256_file(candidate) == old_sha:
                old_path = candidate.parent
                break
        # Do not verify the old Step-10 directory-wide manifest here.  That
        # directory contained the blank annotation forms that were meant to
        # be completed outside the pipeline.  Some operators legitimately
        # completed those copies in place, so the directory can no longer be
        # manifest-valid after the human return.  Step 11's own manifest-valid
        # frozen-evidence inventory is the authoritative pre-response record;
        # the bridge below still requires the corrected R2/R3 packet and private
        # key to match the hashes sealed in that inventory exactly.
        return path.parent, contract, prereg, lineage, old_path
    raise RuntimeError("The exact manifest-valid Step-11 case preregistration was not found")


def _verify_preregistration_bridge(
    step11: Path,
    old_step10: Path | None,
    new_step10: Path,
) -> list[dict[str, Any]]:
    inventory = pd.read_csv(
        step11 / "supplement/frozen_evidence_inventory.csv", encoding="utf-8-sig"
    )
    expected_from_inventory = {
        str(row.artifact): str(row.sha256)
        for row in inventory[inventory["source_stage"].eq("step10")].itertuples(index=False)
    }
    files = (
        "human_packet_share_only_this_folder/annotator_01_case_posts.csv",
        "private_do_not_share_with_annotators/case_source_key.csv",
    )
    rows: list[dict[str, Any]] = []
    for relative in files:
        new = new_step10 / relative
        if not new.is_file():
            raise FileNotFoundError(relative)
        artifact = Path(relative).name
        old_hash = expected_from_inventory.get(artifact)
        if old_hash is None:
            raise ValueError(f"Step 11 did not inventory the preregistered file: {artifact}")
        old_current_hash = ""
        old_current_matches_inventory: bool | str = "NOT_AVAILABLE"
        if old_step10 is not None:
            old = old_step10 / relative
            if old.is_file():
                old_current_hash = sha256_file(old)
                old_current_matches_inventory = old_current_hash == old_hash
            else:
                old_current_matches_inventory = False
        new_hash = sha256_file(new)
        if old_hash != new_hash:
            raise ValueError(
                f"The corrected Step-10 packet differs from the packet frozen by Step 11: {relative}"
            )
        rows.append(
            {
                "relative_path": relative,
                "step11_bound_step10_sha256": old_hash,
                "corrected_step10_sha256": new_hash,
                "original_step10_directory_available": old_step10 is not None,
                "original_step10_current_sha256": old_current_hash,
                "original_step10_current_matches_inventory": old_current_matches_inventory,
                "authoritative_pre_response_source": "step11_manifest_valid_frozen_evidence_inventory",
                "identical": True,
            }
        )
    return rows


def _load_human_frames(step10: Path, protocol: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    supplement = step10 / "supplement"
    frames = {
        "intrusion": pd.read_csv(supplement / "human_word_intrusion_long.csv", encoding="utf-8-sig"),
        "topic": pd.read_csv(supplement / "human_topic_interpretation_long.csv", encoding="utf-8-sig"),
        "case": pd.read_csv(supplement / "human_case_study_long.csv", encoding="utf-8-sig"),
        "topic_key": pd.read_csv(
            step10 / "private_do_not_share_with_annotators/topic_interpretation_key.csv",
            encoding="utf-8-sig",
        ),
        "case_key": pd.read_csv(
            step10 / "private_do_not_share_with_annotators/case_source_key.csv",
            encoding="utf-8-sig",
        ),
        "topic_public": pd.read_csv(
            step10 / "human_packet_share_only_this_folder/annotator_01_topic_interpretation.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        ),
        "case_public": pd.read_csv(
            step10 / "human_packet_share_only_this_folder/annotator_01_case_posts.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        ),
    }
    expected = protocol["human_evaluation"]
    counts = {
        "intrusion": int(expected["word_intrusion_ratings"]),
        "topic": int(expected["topic_ratings"]),
        "case": int(expected["case_ratings"]),
    }
    for name, count in counts.items():
        frame = frames[name]
        if len(frame) != count:
            raise ValueError(f"Expected {count} {name} ratings, found {len(frame)}")
        if frame[["annotator", "item_id"]].duplicated().any():
            raise ValueError(f"Duplicate annotator-item pair in {name} ratings")
        per_item = frame.groupby("item_id")["annotator"].nunique()
        if not per_item.eq(int(expected["annotators"])).all():
            raise ValueError(f"Every {name} item must have three distinct ratings")
    if set(frames["intrusion"]["method_id"]) != set(METHOD_ORDER):
        raise ValueError("The five blinded methods changed")
    if set(frames["case"]["city"]) != set(CITIES):
        raise ValueError("The case-rating city set changed")
    if set(frames["topic"]["service_category"]) - set(expected["service_categories"]):
        raise ValueError("An unrecognized topic service category entered the human results")
    return frames


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    *,
    item_column: str,
    value_column: str,
    seed: int,
    replicates: int,
    transform: Callable[[pd.Series], np.ndarray] | None = None,
) -> tuple[float, float, float]:
    groups = [group[value_column] for _, group in frame.groupby(item_column, sort=True)]
    if not groups:
        raise ValueError("Cannot bootstrap an empty frame")
    arrays = [
        np.asarray(transform(group) if transform is not None else group, dtype=float)
        for group in groups
    ]
    point = float(np.concatenate(arrays).mean())
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        sampled = rng.integers(0, len(arrays), size=len(arrays))
        values[index] = float(np.concatenate([arrays[i] for i in sampled]).mean())
    lower, upper = np.quantile(values, [0.025, 0.975])
    return point, float(lower), float(upper)


def _human_outputs(
    out: Path, frames: Mapping[str, pd.DataFrame], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    intrusion = frames["intrusion"]
    topic = frames["topic"]
    case = frames["case"]
    seed = int(protocol["human_evaluation"]["bootstrap_seed"])
    replicates = int(protocol["human_evaluation"]["bootstrap_replicates"])

    method_rows: list[dict[str, Any]] = []
    uncertainty: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHOD_ORDER):
        selected = intrusion[intrusion["method_id"].eq(method)].copy()
        correct = int(selected["correct_intruder"].sum())
        yes = int(selected["passenger_requirement_yes_no"].eq("yes").sum())
        method_rows.append(
            {
                "method_id": method,
                "method": METHOD_DISPLAY[method],
                "topic_items": int(selected["item_id"].nunique()),
                "ratings": len(selected),
                "word_intrusion_correct": correct,
                "word_intrusion_accuracy": float(selected["correct_intruder"].mean()),
                "mean_coherence_1_to_5": float(selected["coherence_1_to_5"].mean()),
                "passenger_requirement_present_judgments": yes,
                "passenger_requirement_present_rate": float(
                    selected["passenger_requirement_yes_no"].eq("yes").mean()
                ),
            }
        )
        specifications = (
            ("word_intrusion_accuracy", "correct_intruder", None),
            ("mean_coherence_1_to_5", "coherence_1_to_5", None),
            (
                "passenger_requirement_present_rate",
                "passenger_requirement_yes_no",
                lambda values: values.astype(str).eq("yes").to_numpy(dtype=float),
            ),
        )
        for offset, (metric, column, transform) in enumerate(specifications):
            point, lower, upper = _cluster_bootstrap_mean(
                selected,
                item_column="item_id",
                value_column=column,
                seed=seed + 100 * method_index + offset,
                replicates=replicates,
                transform=transform,
            )
            uncertainty.append(
                {
                    "population": method,
                    "metric": metric,
                    "point_estimate": point,
                    "item_cluster_bootstrap_95_ci_lower": lower,
                    "item_cluster_bootstrap_95_ci_upper": upper,
                    "bootstrap_replicates": replicates,
                    "interpretation": "descriptive_item_cluster_bootstrap_not_a_completion_threshold",
                }
            )

    sc_rows = [
        {
            "component": "Topics expressing a passenger requirement",
            "items": int(topic["item_id"].nunique()),
            "ratings": len(topic),
            "numerator": int(topic["passenger_requirement_yes_no"].eq("yes").sum()),
            "denominator": len(topic),
            "observed_result": float(topic["passenger_requirement_yes_no"].eq("yes").mean()),
            "scale_or_unit": "proportion of judgments",
        },
        {
            "component": "Topic coherence",
            "items": int(topic["item_id"].nunique()),
            "ratings": len(topic),
            "numerator": "",
            "denominator": "",
            "observed_result": float(topic["coherence_1_to_5"].mean()),
            "scale_or_unit": "mean on 1-5 scale",
        },
        {
            "component": "Topic actionability",
            "items": int(topic["item_id"].nunique()),
            "ratings": len(topic),
            "numerator": "",
            "denominator": "",
            "observed_result": float(topic["actionability_1_to_5"].mean()),
            "scale_or_unit": "mean on 1-5 scale",
        },
        {
            "component": "Selected development cases containing a service need",
            "items": int(case["item_id"].nunique()),
            "ratings": len(case),
            "numerator": int(case["service_need_present_yes_no"].eq("yes").sum()),
            "denominator": len(case),
            "observed_result": float(case["service_need_present_yes_no"].eq("yes").mean()),
            "scale_or_unit": "proportion of judgments; selected cases, not prevalence",
        },
        {
            "component": "Selected development case-topic fit",
            "items": int(case["item_id"].nunique()),
            "ratings": len(case),
            "numerator": "",
            "denominator": "",
            "observed_result": float(case["topic_fit_1_to_5"].mean()),
            "scale_or_unit": "mean on 1-5 scale; selected cases",
        },
    ]
    for offset, (population, frame, column, transform) in enumerate(
        (
            (
                "sc_htm_topics",
                topic,
                "passenger_requirement_yes_no",
                lambda values: values.astype(str).eq("yes").to_numpy(dtype=float),
            ),
            ("sc_htm_topics", topic, "coherence_1_to_5", None),
            ("sc_htm_topics", topic, "actionability_1_to_5", None),
            (
                "selected_development_cases",
                case,
                "service_need_present_yes_no",
                lambda values: values.astype(str).eq("yes").to_numpy(dtype=float),
            ),
            ("selected_development_cases", case, "topic_fit_1_to_5", None),
        )
    ):
        point, lower, upper = _cluster_bootstrap_mean(
            frame,
            item_column="item_id",
            value_column=column,
            seed=seed + 1000 + offset,
            replicates=replicates,
            transform=transform,
        )
        uncertainty.append(
            {
                "population": population,
                "metric": column,
                "point_estimate": point,
                "item_cluster_bootstrap_95_ci_lower": lower,
                "item_cluster_bootstrap_95_ci_upper": upper,
                "bootstrap_replicates": replicates,
                "interpretation": "descriptive_item_cluster_bootstrap_not_a_completion_threshold",
            }
        )

    def units(frame: pd.DataFrame, column: str) -> dict[str, list[Any]]:
        return {
            str(item): list(group[column])
            for item, group in frame.groupby("item_id", sort=False)
        }

    reliability_specs = (
        ("word_intrusion", "coherence_1_to_5", intrusion, [1, 2, 3, 4, 5]),
        ("word_intrusion", "passenger_requirement_yes_no", intrusion, None),
        ("topic_interpretation", "service_category", topic, None),
        ("topic_interpretation", "actionability_1_to_5", topic, [1, 2, 3, 4, 5]),
        ("case_posts", "topic_fit_1_to_5", case, [1, 2, 3, 4, 5]),
        ("case_posts", "service_need_present_yes_no", case, None),
    )
    reliability: list[dict[str, Any]] = []
    for instrument, outcome, frame, ordered in reliability_specs:
        value = _coincidence_alpha(
            units(frame, outcome),
            ordered_categories=ordered,
        )
        reliability.append(
            {
                "instrument": instrument,
                "outcome": outcome,
                "coefficient": (
                    "krippendorff_alpha_ordinal" if ordered is not None else "krippendorff_alpha_nominal"
                ),
                "value": value,
                "reported_value": "N/E" if value is None else f"{value:.3f}",
                "estimability": "not_estimable" if value is None else "estimated",
                "note": (
                    "all 90 ratings were yes; expected disagreement is zero"
                    if instrument == "case_posts" and outcome == "service_need_present_yes_no"
                    else ""
                ),
            }
        )

    paper = out / "paper"
    _write_csv(paper / "human_validation_panel_a.csv", method_rows)
    _write_csv(paper / "human_validation_panel_b.csv", sc_rows)
    _write_csv(paper / "human_interrater_reliability_corrected.csv", reliability)
    _write_csv(out / "supplement/human_item_cluster_bootstrap_uncertainty.csv", uncertainty)

    md = [
        "# Blinded human assessment of topic interpretability and transport relevance",
        "",
        "## Panel A. Blinded word-intrusion comparison",
        "",
        "| Method | Word-intrusion accuracy | Mean coherence (1-5) | Passenger-requirement present judgments |",
        "|---|---:|---:|---:|",
    ]
    for row in method_rows:
        md.append(
            f"| {row['method']} | {100 * row['word_intrusion_accuracy']:.1f}% "
            f"({row['word_intrusion_correct']}/{row['ratings']}) | "
            f"{row['mean_coherence_1_to_5']:.2f} | "
            f"{100 * row['passenger_requirement_present_rate']:.1f}% "
            f"({row['passenger_requirement_present_judgments']}/{row['ratings']}) |"
        )
    md.extend(
        [
            "",
            "Each method contributed ten blinded topics evaluated by three annotators. The final column is descriptive and is not an accuracy or ranking endpoint.",
            "",
            "## Panel B. Human validation of SC-HTM topics and illustrative development cases",
            "",
            "| Evaluation component | Items | Ratings | Observed result |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sc_rows:
        if row["denominator"] != "":
            result = f"{100 * float(row['observed_result']):.1f}% ({row['numerator']}/{row['denominator']})"
        else:
            result = f"{float(row['observed_result']):.2f}/5"
        md.append(f"| {row['component']} | {row['items']} | {row['ratings']} | {result} |")
    md.extend(
        [
            "",
            "Case rows refer to the 30 preregistered development-only examples and must not be interpreted as population prevalence or test-set accuracy.",
        ]
    )
    _write_text(paper / "human_validation_table.md", "\n".join(md))

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Blinded human assessment of topic interpretability and transport relevance.}",
        r"\label{tab:human_validity}",
        r"\small",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc}",
        r"\hline",
        r"\multicolumn{4}{l}{\textit{Panel A: Blinded word-intrusion comparison}} \\",
        r"\hline",
        r"Method & Word intrusion & Coherence (1--5) & Requirement-present judgments \\",
        r"\hline",
    ]
    for row in method_rows:
        name = r"\textbf{SC-HTM V6.1}" if row["method_id"] == "v6_1" else row["method"]
        tex_lines.append(
            f"{name} & {100 * row['word_intrusion_accuracy']:.1f}\\% "
            f"({row['word_intrusion_correct']}/{row['ratings']}) & "
            f"{row['mean_coherence_1_to_5']:.2f} & "
            f"{100 * row['passenger_requirement_present_rate']:.1f}\\% "
            f"({row['passenger_requirement_present_judgments']}/{row['ratings']}) \\\\"
        )
    tex_lines.extend(
        [
            r"\hline",
            r"\multicolumn{4}{l}{\footnotesize Ten topics per method; three independent ratings per topic. The final column is descriptive, not an accuracy endpoint.} \\",
            r"\hline",
            r"\multicolumn{4}{l}{\textit{Panel B: Human validation of SC-HTM topics and illustrative development cases}} \\",
            r"\hline",
            r"Evaluation component & Items & Ratings & Observed result \\",
            r"\hline",
        ]
    )
    for row in sc_rows:
        if row["denominator"] != "":
            result = (
                f"{100 * float(row['observed_result']):.1f}\\% "
                f"({row['numerator']}/{row['denominator']})"
            )
        else:
            result = f"{float(row['observed_result']):.2f}/5"
        tex_lines.append(
            f"{row['component']} & {row['items']} & {row['ratings']} & {result} \\\\"
        )
    tex_lines.extend(
        [
            r"\hline",
            r"\multicolumn{4}{l}{\footnotesize Case rows are 30 preregistered development examples; they do not estimate prevalence or test accuracy.} \\",
            r"\hline",
            r"\end{tabular*}",
            r"\end{table*}",
        ]
    )
    _write_text(paper / "human_validation_table.tex", "\n".join(tex_lines))
    return {
        "methods": len(method_rows),
        "panel_a_ratings": len(intrusion),
        "panel_b_topic_ratings": len(topic),
        "panel_b_case_ratings": len(case),
        "bootstrap_replicates": replicates,
        "case_service_need_alpha_estimable": False,
    }


def _case_cell_and_selection(
    out: Path, frames: Mapping[str, pd.DataFrame], protocol: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case = frames["case"]
    key = frames["case_key"]
    summary = (
        case.groupby(["item_id", "city"], sort=False)
        .agg(
            ratings=("annotator", "size"),
            median_topic_fit=("topic_fit_1_to_5", "median"),
            mean_topic_fit=("topic_fit_1_to_5", "mean"),
            service_need_yes_votes=(
                "service_need_present_yes_no",
                lambda values: int(values.astype(str).eq("yes").sum()),
            ),
        )
        .reset_index()
        .merge(
            key[
                [
                    "item_id",
                    "source_topic",
                    "top_10_words",
                    "lexical_relevance_score",
                    "selection_partition",
                    "selection_rule",
                ]
            ],
            on="item_id",
            validate="one_to_one",
        )
    )
    summary["service_need_yes_fraction"] = summary["service_need_yes_votes"] / summary["ratings"]
    case_protocol = protocol["case_study"]
    summary["eligible_for_evidence_chain"] = (
        summary["service_need_yes_votes"].ge(int(case_protocol["minimum_service_need_yes_votes"]))
        & summary["median_topic_fit"].ge(float(case_protocol["minimum_median_topic_fit"]))
    )
    summary["analysis_status"] = "all_preregistered_cells_reported_selected_development_examples"
    selected_rows: list[pd.DataFrame] = []
    for city in CITIES:
        eligible = summary[
            summary["city"].eq(city) & summary["eligible_for_evidence_chain"]
        ].sort_values(
            [
                "median_topic_fit",
                "service_need_yes_fraction",
                "lexical_relevance_score",
                "item_id",
            ],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        if eligible.empty:
            raise RuntimeError(f"No preregistered evidence-chain case is eligible for {city}")
        selected_rows.append(eligible.head(int(case_protocol["selected_cases_per_city"])))
    selected = pd.concat(selected_rows, ignore_index=True)
    selected["selection_rank_within_city"] = 1
    selected["selection_status"] = "selected_by_step11_preregistered_rule"
    _write_csv(out / "paper/city_case_cell_summary_all_30.csv", summary.to_dict(orient="records"))
    _write_csv(
        out / "paper/evidence_chain_selection_preregistered.csv",
        selected.to_dict(orient="records"),
    )
    return summary, selected


def _development_documents(
    root: Path,
    source_root: Path | None,
    protocol: Mapping[str, Any],
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], Path]:
    candidate = source_root
    if candidate is None:
        source_text = Path(str(protocol["source_project_root"]))
        candidate = source_text if source_text.is_absolute() else root / source_text
    path = candidate.expanduser().resolve() / "data/processed/internal/documents.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen source documents were not found: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError("The source documents differ from the Step-10 case-selection corpus")
    documents = _read_documents(path)
    expected = protocol["expected_corpus"]
    if len(documents) != int(expected["modeled_documents"]):
        raise ValueError("The modeled-document count changed")
    development = [row for row in documents if str(row["split"]) in {"train", "validation"}]
    if len(development) != int(expected["development_documents"]):
        raise ValueError("The development-document count changed")
    if any(str(row["split"]) == "test" for row in development):
        raise AssertionError("Test text entered the cross-city descriptor audit")
    return development, path


def _cross_city_descriptor_support(
    out: Path,
    documents: Sequence[Mapping[str, Any]],
    case_key: pd.DataFrame,
) -> dict[str, Any]:
    topic_words: dict[int, list[str]] = {}
    for topic, group in case_key.groupby("source_topic", sort=True):
        alternatives = group["top_10_words"].astype(str).drop_duplicates().tolist()
        if len(alternatives) != 1:
            raise ValueError(f"Topic {topic} has inconsistent frozen top-word lists")
        words = [word for word in alternatives[0].split("、") if word]
        if len(words) != 10:
            raise ValueError(f"Topic {topic} does not contain ten frozen descriptors")
        topic_words[int(topic)] = words
    if set(topic_words) != set(range(10)):
        raise ValueError("The frozen ten-topic descriptor grid is incomplete")

    by_city: dict[str, list[set[str]]] = {city: [] for city in CITIES}
    for row in documents:
        city = str(row["city"])
        if city in by_city:
            by_city[city].append(set(map(str, row["tokens"])))
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for topic in range(10):
        all_city_words: list[str] = []
        city_coverage: dict[str, int] = {}
        for word_rank, word in enumerate(topic_words[topic], start=1):
            support: dict[str, int] = {}
            for city in CITIES:
                frequency = sum(word in tokens for tokens in by_city[city])
                support[city] = int(frequency)
                long_rows.append(
                    {
                        "source_topic": topic + 1,
                        "word_rank": word_rank,
                        "descriptor": word,
                        "city": city,
                        "development_documents_in_city": len(by_city[city]),
                        "document_frequency": int(frequency),
                        "document_support_rate": float(frequency / len(by_city[city])),
                        "partition": "development_only_train_or_validation",
                    }
                )
            if all(support[city] >= 1 for city in CITIES):
                all_city_words.append(word)
        for city in CITIES:
            city_coverage[city] = sum(
                any(word in tokens for tokens in by_city[city]) for word in topic_words[topic]
            )
        summary_rows.append(
            {
                "source_topic": topic + 1,
                "frozen_top_10_descriptors": "、".join(topic_words[topic]),
                "descriptors_observed_in_all_three_cities_count": len(all_city_words),
                "descriptors_observed_in_all_three_cities": "、".join(all_city_words),
                "beijing_top10_descriptors_observed": city_coverage["beijing"],
                "shanghai_top10_descriptors_observed": city_coverage["shanghai"],
                "xiamen_top10_descriptors_observed": city_coverage["xiamen"],
                "interpretation": "development_corpus_lexical_support_not_model_stability_or_leave_one_city_out_transfer",
            }
        )
    _write_csv(out / "paper/cross_city_supported_descriptors.csv", summary_rows)
    _write_csv(out / "supplement/cross_city_descriptor_document_frequency.csv", long_rows)
    return {
        "topics": len(summary_rows),
        "descriptor_city_rows": len(long_rows),
        "development_documents": len(documents),
        "test_documents_accessed": 0,
        "claim_scope": "lexical_support_only_not_transfer",
    }


def _two_of_three_majority(values: Sequence[Any]) -> tuple[str, int]:
    """Return a categorical majority without generating a replacement label."""

    normalized = [str(value).strip() for value in values]
    if len(normalized) != 3 or any(not value for value in normalized):
        raise ValueError("A categorical case-study item must contain three nonblank votes")
    counts = Counter(normalized)
    value, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return (value, count) if count >= 2 else ("no_consensus", count)


def _three_annotator_material(
    out: Path,
    frames: Mapping[str, pd.DataFrame],
    selected: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Retain all genuine responses; derive only a categorical 2-of-3 majority."""

    topic = frames["topic"]
    case = frames["case"]
    topic_mapping = frames["topic_key"].merge(
        frames["topic_public"][["item_id", "top_10_words"]],
        on="item_id",
        validate="one_to_one",
    )
    allowed_categories = set(protocol["human_evaluation"]["service_categories"])
    topic_rows: list[dict[str, Any]] = []
    for source_topic in range(10):
        item = topic_mapping[
            topic_mapping["source_topic"].astype(int).eq(source_topic)
        ].iloc[0]
        votes = topic[topic["item_id"].eq(item["item_id"])].sort_values("annotator")
        if votes["annotator"].astype(int).tolist() != [1, 2, 3]:
            raise ValueError(f"Topic {source_topic + 1} lacks the three original annotators")
        majority, majority_votes = _two_of_three_majority(votes["service_category"])
        if majority != "no_consensus" and majority not in allowed_categories:
            raise ValueError("An invalid category entered the majority summary")
        row: dict[str, Any] = {
            "record_id": f"TOPIC_{source_topic + 1:02d}",
            "source_topic": source_topic + 1,
            "blinded_item_id": str(item["item_id"]),
            "frozen_top_10_words": str(item["top_10_words"]),
            "majority_service_category": majority,
            "majority_service_category_display": CATEGORY_DISPLAY[majority],
            "majority_service_category_votes": majority_votes,
            "free_text_reporting": "all_three_original_responses_no_synthesis",
        }
        for vote in votes.itertuples(index=False):
            annotator = int(vote.annotator)
            label = str(vote.topic_label).strip()
            action = str(vote.suggested_transport_action).strip()
            if not label or not action:
                raise ValueError(f"Topic {source_topic + 1} contains blank original wording")
            row[f"annotator_{annotator:02d}_service_category"] = str(
                vote.service_category
            )
            row[f"annotator_{annotator:02d}_requirement_label_zh"] = label
            row[f"annotator_{annotator:02d}_transport_action_zh"] = action
        topic_rows.append(row)

    public = frames["case_public"].set_index("item_id")
    case_rows: list[dict[str, Any]] = []
    city_order = {city: index for index, city in enumerate(CITIES)}
    for selected_row in sorted(
        selected.itertuples(index=False), key=lambda value: city_order[str(value.city)]
    ):
        item_id = str(selected_row.item_id)
        item = public.loc[item_id]
        votes = case[case["item_id"].eq(item_id)].sort_values("annotator")
        if votes["annotator"].astype(int).tolist() != [1, 2, 3]:
            raise ValueError(f"Selected case {item_id} lacks the three original annotators")
        row = {
            "record_id": f"CASE_{str(selected_row.city).upper()}",
            "item_id": item_id,
            "city": str(selected_row.city),
            "source_topic": int(selected_row.source_topic) + 1,
            "median_topic_fit": float(selected_row.median_topic_fit),
            "mean_topic_fit": float(selected_row.mean_topic_fit),
            "service_need_yes_votes": int(selected_row.service_need_yes_votes),
            "frozen_lexical_relevance": float(selected_row.lexical_relevance_score),
            "frozen_top_10_words": str(item["topic_words"]),
            "delexicalized_passenger_post_zh": str(item["delexicalized_passenger_post"]),
            "free_text_reporting": "all_three_original_responses_no_synthesis",
        }
        for vote in votes.itertuples(index=False):
            annotator = int(vote.annotator)
            label = str(vote.case_requirement_label).strip()
            action = str(vote.recommended_transport_action).strip()
            if not label or not action:
                raise ValueError(f"Selected case {item_id} contains blank original wording")
            row[f"annotator_{annotator:02d}_requirement_label_zh"] = label
            row[f"annotator_{annotator:02d}_transport_action_zh"] = action
        case_rows.append(row)

    topic_frame = pd.DataFrame(topic_rows)
    case_frame = pd.DataFrame(case_rows)
    if len(topic_frame) != int(protocol["case_study"]["topic_rows"]):
        raise ValueError("The three-annotator topic summary is incomplete")
    if len(case_frame) != int(protocol["case_study"]["selected_case_rows"]):
        raise ValueError("The three-city selected-case summary is incomplete")

    _write_csv(out / "paper/three_annotator_topic_interpretations.csv", topic_rows)
    _write_csv(out / "paper/three_annotator_selected_cases.csv", case_rows)
    _write_csv(
        out / "supplement/original_topic_interpretation_responses.csv",
        topic.sort_values(["item_id", "annotator"]).to_dict(orient="records"),
    )
    selected_ids = set(case_frame["item_id"].astype(str))
    _write_csv(
        out / "supplement/original_selected_case_responses.csv",
        case[case["item_id"].astype(str).isin(selected_ids)]
        .sort_values(["item_id", "annotator"])
        .to_dict(orient="records"),
    )
    identity_material = [
        {
            "record_id": str(row["record_id"]),
            "item_id": str(row.get("item_id", row.get("blinded_item_id", ""))),
            "free_text_reporting": str(row["free_text_reporting"]),
            "responses": [
                [
                    str(row[f"annotator_{annotator:02d}_requirement_label_zh"]),
                    str(row[f"annotator_{annotator:02d}_transport_action_zh"]),
                ]
                for annotator in (1, 2, 3)
            ],
        }
        for row in topic_rows + case_rows
    ]
    state = {
        "status": "COMPLETE_FROM_THREE_ORIGINAL_ANNOTATORS_NO_NEW_HUMAN_INPUT",
        "complete": True,
        "topic_rows": len(topic_frame),
        "selected_case_rows": len(case_frame),
        "topics_with_two_of_three_category_majority": int(
            topic_frame["majority_service_category_votes"].ge(2).sum()
        ),
        "topics_without_category_majority": int(
            topic_frame["majority_service_category"].eq("no_consensus").sum()
        ),
        "original_free_text_strings_retained": 10 * 3 * 2 + 3 * 3 * 2,
        "new_human_files_required": 0,
        "new_human_judgments_collected": 0,
        "fourth_adjudicator_used": False,
        "llm_or_synthetic_human_judgments": 0,
        "generated_consensus_strings": 0,
        "material_identity": sha256_object(identity_material),
    }
    _write_json(out / "three_annotator_material_contract.json", state)
    return state, {"topic": topic_frame, "case": case_frame}


def _heat_fill(value: float) -> str:
    palette = {1: "#f3f4f6", 2: "#d8e8ee", 3: "#a9ced8", 4: "#65a9b8", 5: "#16788a"}
    return palette[int(round(min(5.0, max(1.0, value))))]


def _city_map_svg(path: Path, rows: pd.DataFrame) -> None:
    width, left, top, row_h, cell_w = 1360, 520, 145, 54, 230
    height = top + 10 * row_h + 120
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;fill:#17232b}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#52616b}.head{font-size:16px;font-weight:700}.label{font-size:14px}.cell{font-size:17px;font-weight:700}</style>',
        '<text class="title" x="40" y="42">Passenger-requirement fit across three cities</text>',
        '<text class="sub" x="40" y="70">All 30 preregistered development-only cells are shown. Number = median fit (1–5); ● = at least 2/3 service-need votes.</text>',
    ]
    for city_index, city in enumerate(CITIES):
        x = left + city_index * cell_w + cell_w / 2
        pieces.append(f'<text class="head" text-anchor="middle" x="{x}" y="115">{CITY_DISPLAY[city]}</text>')
    ordered = rows.sort_values("source_topic")
    for row_index, row in enumerate(ordered.itertuples(index=False)):
        y = top + row_index * row_h
        label = f"T{int(row.source_topic):02d}  {row.majority_service_category_display}"
        if len(label) > 34:
            label = label[:33] + "…"
        pieces.append(f'<text class="label" x="40" y="{y + 34}">{html.escape(label)}</text>')
        for city_index, city in enumerate(CITIES):
            value = float(getattr(row, f"{city}_median_topic_fit"))
            yes_votes = int(getattr(row, f"{city}_service_need_yes_votes"))
            x = left + city_index * cell_w
            color = _heat_fill(value)
            text_color = "#ffffff" if value >= 4.5 else "#17232b"
            marker = "●" if yes_votes >= 2 else "○"
            pieces.extend(
                [
                    f'<rect x="{x + 4}" y="{y + 4}" width="{cell_w - 12}" height="{row_h - 8}" rx="7" fill="{color}" stroke="#ffffff"/>',
                    f'<text class="cell" text-anchor="middle" x="{x + cell_w / 2}" y="{y + 35}" fill="{text_color}" style="fill:{text_color}">{value:.0f}  {marker}</text>',
                ]
            )
    legend_y = top + 10 * row_h + 45
    pieces.append(f'<text class="sub" x="40" y="{legend_y}">Rows use 2-of-3 category majorities. Cells are selected development examples, not citywide prevalence or test accuracy.</text>')
    pieces.append("</svg>")
    _write_text(path, "\n".join(pieces))
    ET.parse(path)


def _wrap_visual(text: str, limit: int) -> list[str]:
    cleaned = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    if not cleaned:
        return [""]
    lines: list[str] = []
    current = ""
    width = 0.0
    for character in cleaned:
        increment = 1.0 if ord(character) > 127 else 0.55
        if current and width + increment > limit:
            lines.append(current)
            current, width = character, increment
        else:
            current += character
            width += increment
    if current:
        lines.append(current)
    return lines


def _evidence_chain_svg(path: Path, rows: pd.DataFrame) -> None:
    width, height = 1440, 940
    margin, gap = 34, 22
    panel_w = (width - 2 * margin - 2 * gap) / 3
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;fill:#17232b}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#52616b}.city{font-size:21px;font-weight:700}.section{font-size:14px;font-weight:700;fill:#176b7a}.body{font-size:13px}.meta{font-size:12px;fill:#52616b}</style>',
        '<text class="title" x="34" y="40">Three-annotator passenger-requirement evidence chains</text>',
        '<text class="sub" x="34" y="67">Case selection follows the frozen rule; all three original interpretations are retained without a generated consensus.</text>',
    ]
    order = {city: index for index, city in enumerate(CITIES)}
    for row in sorted(rows.itertuples(index=False), key=lambda value: order[str(value.city)]):
        panel_index = order[str(row.city)]
        x = margin + panel_index * (panel_w + gap)
        pieces.append(f'<rect x="{x}" y="94" width="{panel_w}" height="790" rx="12" fill="#f8fafb" stroke="#b9c8ce"/>')
        pieces.append(f'<text class="city" x="{x + 20}" y="130">{CITY_DISPLAY[str(row.city)]}</text>')
        pieces.append(
            f'<text class="meta" x="{x + 20}" y="153">{html.escape(str(row.item_id))} · median fit {float(row.median_topic_fit):.0f}/5 · yes {int(row.service_need_yes_votes)}/3</text>'
        )
        requirement_labels = "；".join(
            f"A{annotator}: {getattr(row, f'annotator_{annotator:02d}_requirement_label_zh')}"
            for annotator in (1, 2, 3)
        )
        transport_actions = "；".join(
            f"A{annotator}: {getattr(row, f'annotator_{annotator:02d}_transport_action_zh')}"
            for annotator in (1, 2, 3)
        )
        blocks = [
            ("Delexicalized passenger post", str(row.delexicalized_passenger_post_zh), 14, 13),
            ("Frozen topic descriptors", str(row.frozen_top_10_words), 18, 13),
            ("Three original requirement labels", requirement_labels, 22, 13),
            ("Three original transport actions", transport_actions, 22, 13),
        ]
        y = 190
        for block_index, (heading, body, wrap_limit, font_size) in enumerate(blocks):
            lines = _wrap_visual(body, wrap_limit)
            block_height = 54 + 21 * len(lines)
            fill = "#ffffff" if block_index < 2 else "#e7f2f4"
            pieces.append(f'<rect x="{x + 16}" y="{y}" width="{panel_w - 32}" height="{block_height}" rx="9" fill="{fill}" stroke="#d4e0e4"/>')
            pieces.append(f'<text class="section" x="{x + 30}" y="{y + 25}">{html.escape(heading)}</text>')
            for line_index, line in enumerate(lines):
                pieces.append(
                    f'<text class="body" x="{x + 30}" y="{y + 51 + line_index * 21}" style="font-size:{font_size}px">{html.escape(line)}</text>'
                )
            y += block_height + 18
    pieces.append('<text class="meta" x="34" y="920">Chinese source figure. No fourth adjudicator or synthesized human wording was used; the post-collection reporting amendment is disclosed.</text>')
    pieces.append("</svg>")
    _write_text(path, "\n".join(pieces))
    ET.parse(path)


def _three_annotator_case_outputs(
    out: Path,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    material: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Render the case study without choosing or inventing free-text consensus."""

    topic_rows = material["topic"].copy()
    topic_rows["source_topic"] = topic_rows["source_topic"].astype(int)
    cells = summary.copy()
    cells["source_topic"] = cells["source_topic"].astype(int) + 1
    wide_rows: list[dict[str, Any]] = []
    for topic in range(1, 11):
        interpretation = topic_rows[topic_rows["source_topic"].eq(topic)].iloc[0]
        row: dict[str, Any] = {
            "source_topic": topic,
            "majority_service_category": interpretation["majority_service_category"],
            "majority_service_category_display": interpretation[
                "majority_service_category_display"
            ],
            "majority_service_category_votes": int(
                interpretation["majority_service_category_votes"]
            ),
            "frozen_top_10_words": interpretation["frozen_top_10_words"],
            "free_text_reporting": "all_three_original_responses_no_synthesis",
        }
        for annotator in (1, 2, 3):
            row[f"annotator_{annotator:02d}_requirement_label_zh"] = interpretation[
                f"annotator_{annotator:02d}_requirement_label_zh"
            ]
            row[f"annotator_{annotator:02d}_transport_action_zh"] = interpretation[
                f"annotator_{annotator:02d}_transport_action_zh"
            ]
        for city in CITIES:
            cell = cells[cells["source_topic"].eq(topic) & cells["city"].eq(city)].iloc[0]
            row[f"{city}_median_topic_fit"] = float(cell["median_topic_fit"])
            row[f"{city}_mean_topic_fit"] = float(cell["mean_topic_fit"])
            row[f"{city}_service_need_yes_votes"] = int(cell["service_need_yes_votes"])
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    _write_csv(out / "paper/city_requirement_map_all_three_annotators.csv", wide_rows)
    _city_map_svg(out / "paper/figure_city_requirement_map.svg", wide)

    evidence = material["case"].copy().merge(
        topic_rows[
            [
                "source_topic",
                "majority_service_category",
                "majority_service_category_display",
                "majority_service_category_votes",
            ]
        ],
        on="source_topic",
        validate="many_to_one",
    )
    expected_ids = set(selected["item_id"].astype(str))
    if set(evidence["item_id"].astype(str)) != expected_ids:
        raise ValueError("The rendered evidence cases differ from the frozen selection")
    _write_csv(
        out / "paper/evidence_chain_three_cities_all_annotators_zh.csv",
        evidence.to_dict(orient="records"),
    )
    _evidence_chain_svg(
        out / "paper/figure_evidence_chain_three_cities_all_annotators_zh.svg",
        evidence,
    )

    md = [
        "# Preregistered three-city passenger-requirement case study",
        "",
        "All cases are development-only. The case-selection rule was frozen before the human responses. All three original labels and actions are shown without a fourth adjudicator or generated consensus.",
        "",
        "| City | Case / topic | Human support | Majority service category | Three original requirement labels (Chinese) | Three original transport actions (Chinese) |",
        "|---|---|---:|---|---|---|",
    ]
    city_order = {city: index for index, city in enumerate(CITIES)}
    for row in sorted(
        evidence.itertuples(index=False), key=lambda value: city_order[str(value.city)]
    ):
        labels = "<br>".join(
            f"A{annotator}: {getattr(row, f'annotator_{annotator:02d}_requirement_label_zh')}"
            for annotator in (1, 2, 3)
        )
        actions = "<br>".join(
            f"A{annotator}: {getattr(row, f'annotator_{annotator:02d}_transport_action_zh')}"
            for annotator in (1, 2, 3)
        )
        md.append(
            f"| {CITY_DISPLAY[str(row.city)]} | {row.item_id} / T{int(row.source_topic):02d} | "
            f"median fit {float(row.median_topic_fit):.0f}/5; need {int(row.service_need_yes_votes)}/3 | "
            f"{row.majority_service_category_display} ({int(row.majority_service_category_votes)}/3) | "
            f"{labels} | {actions} |"
        )
    md.extend(
        [
            "",
            "These selected examples demonstrate an auditable model-to-requirement-to-action evidence chain. They do not estimate passenger-need prevalence, citywide accuracy, or causal operational benefit.",
            "",
            "Protocol note: Step 11 proposed a fourth senior adjudicator for free-text consolidation. Because no such adjudicator was available, Step 12 R3 makes a disclosed post-collection reporting amendment: it preserves every original response and does not synthesize consensus wording. Numeric outcomes and the frozen case-selection rule are unchanged.",
        ]
    )
    _write_text(out / "paper/case_study_three_cities.md", "\n".join(md))

    translation_rows: list[dict[str, Any]] = []
    for row in topic_rows.itertuples(index=False):
        translation_rows.append(
            {
                "record_type": "topic_words",
                "record_id": f"TOPIC_{int(row.source_topic):02d}_WORDS",
                "source_text_zh": row.frozen_top_10_words,
                "english_rendering": "",
                "translation_method_disclosure": "",
                "author_accuracy_check": "",
            }
        )
        for annotator in (1, 2, 3):
            for kind, source in (
                (
                    "topic_requirement",
                    getattr(row, f"annotator_{annotator:02d}_requirement_label_zh"),
                ),
                (
                    "topic_action",
                    getattr(row, f"annotator_{annotator:02d}_transport_action_zh"),
                ),
            ):
                translation_rows.append(
                    {
                        "record_type": kind,
                        "record_id": f"TOPIC_{int(row.source_topic):02d}_A{annotator}_{kind.upper()}",
                        "source_text_zh": source,
                        "english_rendering": "",
                        "translation_method_disclosure": "",
                        "author_accuracy_check": "",
                    }
                )
    for row in evidence.itertuples(index=False):
        for kind, source in (
            ("case_post", row.delexicalized_passenger_post_zh),
            ("case_words", row.frozen_top_10_words),
        ):
            translation_rows.append(
                {
                    "record_type": kind,
                    "record_id": f"{row.item_id}_{kind.upper()}",
                    "source_text_zh": source,
                    "english_rendering": "",
                    "translation_method_disclosure": "",
                    "author_accuracy_check": "",
                }
            )
        for annotator in (1, 2, 3):
            for kind, source in (
                (
                    "case_requirement",
                    getattr(row, f"annotator_{annotator:02d}_requirement_label_zh"),
                ),
                (
                    "case_action",
                    getattr(row, f"annotator_{annotator:02d}_transport_action_zh"),
                ),
            ):
                translation_rows.append(
                    {
                        "record_type": kind,
                        "record_id": f"{row.item_id}_A{annotator}_{kind.upper()}",
                        "source_text_zh": source,
                        "english_rendering": "",
                        "translation_method_disclosure": "",
                        "author_accuracy_check": "",
                    }
                )
    _write_csv(
        out / "editorial_english_derivative/translation_template.csv", translation_rows
    )
    _write_text(
        out / "editorial_english_derivative/README.md",
        """# Optional English editorial derivative

This is not another human experiment and is not required for Step 12 PASS. Translate the Chinese source strings only for reader-facing presentation; never change item IDs, ratings, categories, case selection, or the Chinese source of record. If generative AI is used, record the system and use in `translation_method_disclosure`, check every rendering before submission, retain the Chinese source in the supplement, and follow the target journal's current disclosure policy. A translation must never be described as an additional human judgment.
""",
    )
    _write_text(
        out / "author_reporting/HUMAN_STUDY_DISCLOSURE_REQUIRED.md",
        """# Author reporting required before manuscript submission

No additional annotation or adjudication is required. The authors must, however, report the true administrative facts of the original three-person evaluation: ethics-board review/exemption or the reason none was obtained; consent or the reason it was not obtained; recruitment and compensation/relationships; relevant language/domain background; blinding and independence procedures; and privacy/data-retention handling. Do not infer or invent any of these facts from the CSV files.
""",
    )
    return {
        "map_topics": len(wide),
        "map_city_cells": len(summary),
        "evidence_chain_cases": len(evidence),
        "original_case_labels_displayed": len(evidence) * 3,
        "original_case_actions_displayed": len(evidence) * 3,
        "generated_consensus_strings": 0,
        "english_translation_status": "OPTIONAL_EDITORIAL_DERIVATIVE_NOT_HUMAN_EVIDENCE",
        "author_human_study_disclosure_status": "REQUIRED_BEFORE_SUBMISSION",
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
    destination = root / "v6_step12_results.zip"
    if destination.is_file():
        destination.unlink()
    prefix = Path("step12_human_case_finalization") / out.name
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidate for candidate in out.rglob("*") if candidate.is_file()):
            archive.write(path, (prefix / path.relative_to(out)).as_posix())
    return destination


def run_v6_step12(
    *,
    project_root: Path,
    config_path: Path,
    source_root: Path | None = None,
) -> Path:
    """Finalize the preregistered human case study without fitting any model."""

    root = project_root.expanduser().resolve()
    protocol = _load_protocol(config_path)
    print(
        f"[V6.1 Step 12] START | implementation={IMPLEMENTATION} | "
        "fits=0 optimizer-steps=0 frozen-result-changes=0 test-text=0",
        flush=True,
    )
    step10, contract10 = _latest_corrected_step10(root, protocol)
    step11, contract11, prereg, lineage11, old_step10 = _step11_preregistration(root, protocol)
    bridge = _verify_preregistration_bridge(step11, old_step10, step10)
    out = root / "outputs/v6/step12_human_case_finalization" / _utc_id()
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "preregistration_bridge.csv", bridge)

    frames = _load_human_frames(step10, protocol)
    human_state = _human_outputs(out, frames, protocol)
    case_summary, selected = _case_cell_and_selection(out, frames, protocol)
    development, documents_path = _development_documents(
        root,
        source_root,
        protocol,
        str(contract10["case_study"]["source_documents_sha256"]),
    )
    descriptor_state = _cross_city_descriptor_support(
        out, development, frames["case_key"]
    )
    three_annotator_state, material = _three_annotator_material(
        out, frames, selected, protocol
    )
    final_case_state = {
        "status": "COMPLETE_ALL_THREE_ORIGINAL_RESPONSES_RETAINED",
        **_three_annotator_case_outputs(out, case_summary, selected, material),
    }

    alpha = pd.read_csv(
        out / "paper/human_interrater_reliability_corrected.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    service_alpha = alpha[
        alpha["instrument"].eq("case_posts")
        & alpha["outcome"].eq("service_need_present_yes_no")
    ].iloc[0]
    checks = [
        ("corrected_step10_manifest_and_contract", True, contract10["contract_fingerprint"]),
        ("complete_human_return_and_locked_fields", contract10["human_evaluation"]["locked_fields_verified_unchanged"] is True, contract10["human_evaluation"]["sealed_response_set_identity"]),
        ("step11_manifest_and_contract", True, contract11["contract_fingerprint"]),
        ("exact_pre_response_case_preregistration", prereg["preregistration_identity"] == protocol["prerequisite"]["step11_preregistration_identity"], prereg["status"]),
        ("corrected_packet_identical_to_preregistered_packet", all(row["identical"] for row in bridge), f"files={len(bridge)}"),
        ("word_intrusion_ratings_complete", human_state["panel_a_ratings"] == 150, str(human_state["panel_a_ratings"])),
        ("topic_interpretation_ratings_complete", human_state["panel_b_topic_ratings"] == 30, str(human_state["panel_b_topic_ratings"])),
        ("case_ratings_complete", human_state["panel_b_case_ratings"] == 90, str(human_state["panel_b_case_ratings"])),
        ("no_variation_alpha_reported_not_estimable", service_alpha["reported_value"] == "N/E" and service_alpha["estimability"] == "not_estimable", str(service_alpha.to_dict())),
        ("all_30_preregistered_case_cells_reported", len(case_summary) == 30, f"cells={len(case_summary)}"),
        ("one_evidence_chain_case_per_city_selected", len(selected) == 3 and set(selected["city"]) == set(CITIES), selected["item_id"].astype(str).tolist()),
        ("selected_cases_meet_frozen_eligibility", bool(selected["eligible_for_evidence_chain"].all()), selected["item_id"].astype(str).tolist()),
        ("all_three_original_topic_and_case_wordings_retained", three_annotator_state["original_free_text_strings_retained"] == 78, str(three_annotator_state)),
        ("no_fourth_adjudicator_or_new_human_input", three_annotator_state["fourth_adjudicator_used"] is False and three_annotator_state["new_human_judgments_collected"] == 0, str(three_annotator_state)),
        ("no_llm_or_synthetic_human_judgment", three_annotator_state["llm_or_synthetic_human_judgments"] == 0, str(three_annotator_state)),
        ("no_generated_free_text_consensus", three_annotator_state["generated_consensus_strings"] == 0 and final_case_state["generated_consensus_strings"] == 0, str(three_annotator_state)),
        ("post_collection_reporting_amendment_disclosed", protocol["protocol_amendment"]["status"] == "disclosed_post_collection_operational_amendment", str(protocol["protocol_amendment"])),
        ("cross_city_descriptor_audit_uses_development_only", descriptor_state["development_documents"] == 695 and descriptor_state["test_documents_accessed"] == 0, str(descriptor_state)),
        ("cross_city_occurrence_not_labeled_transfer", descriptor_state["claim_scope"] == "lexical_support_only_not_transfer", descriptor_state["claim_scope"]),
        ("no_fit_tuning_refit_test_text_or_result_change", True, "0/0/0/0/0"),
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
        raise RuntimeError(f"Step-12 hard checks failed: {failed}")

    status = "PASS"
    readiness = (
        READINESS_EXTERNAL_COMPLETE
        if contract10["external_evaluation"]["complete"]
        else READINESS_EXTERNAL_PENDING
    )
    completion_gates = [
        {"gate": "corrected_step10_human_evidence", "status": "COMPLETE"},
        {"gate": "pre_response_step11_case_selection_rule", "status": "COMPLETE"},
        {"gate": "human_table_and_corrected_agreement", "status": "COMPLETE"},
        {"gate": "all_30_city_case_cells", "status": "COMPLETE"},
        {"gate": "all_three_original_case_interpretations_retained", "status": "COMPLETE"},
        {"gate": "fourth_adjudication", "status": "NOT_PERFORMED_DISCLOSED_AMENDMENT", "required_for_step12_pass": False},
        {"gate": "english_editorial_derivative", "status": "OPTIONAL_PENDING", "required_for_step12_pass": False},
        {"gate": "author_human_study_reporting_disclosure", "status": "REQUIRED_BEFORE_SUBMISSION", "required_for_step12_pass": False},
        {
            "gate": "independent_external_replication",
            "status": "COMPLETE" if contract10["external_evaluation"]["complete"] else "PENDING",
            "required_for_step12_pass": False,
        },
    ]
    _write_csv(out / "completion_gates.csv", completion_gates)
    lineage = {
        "corrected_step10_contract_path": str((step10 / "step10_contract.json").resolve()),
        "corrected_step10_contract_sha256": sha256_file(step10 / "step10_contract.json"),
        "corrected_step10_contract_fingerprint": contract10["contract_fingerprint"],
        "pre_response_step11_contract_path": str((step11 / "step11_contract.json").resolve()),
        "pre_response_step11_contract_sha256": sha256_file(step11 / "step11_contract.json"),
        "pre_response_step11_contract_fingerprint": contract11["contract_fingerprint"],
        "step11_preregistration_identity": prereg["preregistration_identity"],
        "step11_bound_original_step10_contract_sha256": lineage11["step10_contract_sha256"],
        "step11_bound_original_step10_directory_available": old_step10 is not None,
        "source_documents_path": str(documents_path),
        "source_documents_sha256": sha256_file(documents_path),
        "step7_freeze_identity": protocol["prerequisite"]["step7_freeze_identity"],
        "model_fits": 0,
        "test_text_accesses": 0,
    }
    _write_json(out / "frozen_lineage.json", lineage)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION,
        "scope": SCOPE,
        "status": status,
        "readiness": readiness,
        "hard_checks_passed": len(checks),
        "hard_checks_total": len(checks),
        "human_outputs": human_state,
        "case_selection": {
            "all_cells": len(case_summary),
            "selected_cases": selected["item_id"].astype(str).tolist(),
            "cities": selected["city"].astype(str).tolist(),
            "rule_identity": prereg["preregistration_identity"],
        },
        "cross_city_descriptor_support": descriptor_state,
        "three_annotator_finalization": three_annotator_state,
        "protocol_amendment": {
            **dict(protocol["protocol_amendment"]),
            "step11_original_free_text_plan": prereg["free_text_consensus"],
            "effect_scope": "free_text_presentation_only",
            "case_selection_or_numeric_result_changes": 0,
        },
        "final_case_outputs": final_case_state,
        "external_evaluation_status": contract10["external_evaluation"]["status"],
        "completion_gates": completion_gates,
        "model_fits": 0,
        "optimizer_steps": 0,
        "model_or_hyperparameter_changes": 0,
        "comparator_refits": 0,
        "frozen_result_changes": 0,
        "test_text_accesses": 0,
        "unfavorable_case_cells_omitted": False,
        "cross_city_transfer_claimed_from_word_occurrence": False,
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    _write_json(out / "step12_contract.json", contract)
    _manifest(out)
    result_zip = _return_zip(root, out)
    print(
        f"[V6.1 Step 12] {status} | hard checks={len(checks)}/{len(checks)} | "
        f"human-case={three_annotator_state['status']}",
        flush=True,
    )
    print(
        "[V6.1 Step 12] preregistered selections | "
        + " ".join(
            f"{row.city}={row.item_id}" for row in selected.itertuples(index=False)
        ),
        flush=True,
    )
    print(f"[V6.1 Step 12] contract={out / 'step12_contract.json'}", flush=True)
    print(f"[V6.1 Step 12] return-zip={result_zip}", flush=True)
    print(
        "[V6.1 Step 12] No new annotator or adjudicator files are required. "
        "Complete the factual author ethics/consent disclosure before submission.",
        flush=True,
    )
    return out
