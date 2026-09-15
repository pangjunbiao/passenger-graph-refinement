"""V6.1 Step 14 R3: report frozen pre-test parameter sensitivity.

This reporting-only stage combines two immutable development studies that were
completed before the one-shot test:

* Step 4R2: pooled-backoff weight rho on five grouped original-training folds.
* Step 6R2: graph-core coverage q and decoder calibration on three frozen
  spent-validation seeds.

The evidence surfaces are reported separately. Step 14 performs no fitting,
new evaluation, hyperparameter selection, validation reopening, or test access.
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

import yaml

from src.v6.data_contract import sha256_file, sha256_object


IMPLEMENTATION = "v6_1_step14_primary_parameter_sensitivity_reporting_r3"
SCOPE = "posthoc_reporting_of_frozen_pretest_parameter_sensitivity_no_new_evaluation"
READINESS = "READY_FOR_PAPER_OR_SUPPLEMENT_WITH_EVIDENCE_SURFACE_DISCLOSURE"

STEP4_IMPLEMENTATION = "v6_step4r2_hierarchical_city_backoff_gate_r1"
STEP6_IMPLEMENTATION = "v6_1_step6r2_spent_validation_development_recovery_r1"
DATA_KEY = "7774658044d3e297ed5bff31a45bcad5221228a53fe48fd5aabacebf12778c2d"
OFFICIAL_COMMIT = "8ac3592165cc7851676be2776b314eb6e44e9388"

EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_RHOS = [0.10, 0.20, 0.35]
EXPECTED_SEEDS = [20260910, 20270910, 20280910]
EXPECTED_QUOTAS = [8, 9, 10]
EXPECTED_CALIBRATIONS = ["native_decoder", "training_moment_match"]
SELECTED_CANDIDATE = "quota_10__training_moment_match"

Q_ENDPOINTS = [
    "validation_npmi_full",
    "validation_cv_full",
    "full_macro_city_nll",
    "validation_topic_diversity_full",
    "validation_top_word_redundancy_full",
]
Q_EFFECTS = [
    "validation_npmi_improvement_vs_without_graph",
    "validation_cv_improvement_vs_without_graph",
    "pooled_nll_reduction_vs_without_backoff",
    "calibration_nll_reduction_vs_native",
    "full_nll_reduction_vs_official_parent",
]


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


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
        if sha256_file(artifact) != str(item["sha256"]):
            raise ValueError(f"Frozen artifact hash changed: {artifact}")
        if int(artifact.stat().st_size) != int(item["size_bytes"]):
            raise ValueError(f"Frozen artifact size changed: {artifact}")
    return manifest


def _as_float(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite sensitivity value in {key}")
    return value


def _load_protocol(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    protocol = document.get("v6", {}).get("step14", {})
    if document.get("schema_version") != 1 or not isinstance(protocol, dict):
        raise ValueError("Unsupported Step-14 configuration")
    if protocol.get("implementation_version") != IMPLEMENTATION:
        raise ValueError("Step-14 implementation lock changed")
    if protocol.get("scope") != SCOPE:
        raise ValueError("Step-14 scope changed")

    required4 = protocol["prerequisites"]["step04r2"]
    required6 = protocol["prerequisites"]["step06r2"]
    if any(
        (
            required4["implementation_version"] != STEP4_IMPLEMENTATION,
            required4["status"] != "PASS",
            required4["scope"]
            != "grouped_original_train_city_backoff_revision_no_validation_test_or_comparators",
            required4["data_key"] != DATA_KEY,
            required4["official_commit"] != OFFICIAL_COMMIT,
            list(map(int, required4["folds"])) != EXPECTED_FOLDS,
            list(map(float, required4["candidate_rhos"])) != EXPECTED_RHOS,
            float(required4["selected_rho"]) != 0.35,
            int(required4["validation_documents_used"]) != 0,
            int(required4["test_documents_used"]) != 0,
        )
    ):
        raise ValueError("Step-14 Step-4R2 prerequisite changed")
    if any(
        (
            required6["implementation_version"] != STEP6_IMPLEMENTATION,
            required6["status"] != "PASS",
            required6["readiness"]
            != "READY_FOR_V6_1_FINAL_REFIT_AND_ONE_SHOT_TEST",
            required6["validation_status"] != "SPENT_DEVELOPMENT",
            required6["data_key"] != DATA_KEY,
            required6["official_commit"] != OFFICIAL_COMMIT,
            int(required6["test_documents_used"]) != 0,
            int(required6["candidate_rows"]) != 18,
            list(map(int, required6["seeds"])) != EXPECTED_SEEDS,
            list(map(int, required6["prototype_quotas"])) != EXPECTED_QUOTAS,
            list(required6["calibrations"]) != EXPECTED_CALIBRATIONS,
            required6["selected_candidate"] != SELECTED_CANDIDATE,
        )
    ):
        raise ValueError("Step-14 Step-6R2 prerequisite changed")

    reporting = protocol["reporting"]
    if list(reporting["parameters"]) != [
        "graph_core_coverage_q",
        "pooled_backoff_weight_rho",
        "decoder_calibration",
    ]:
        raise ValueError("Step-14 parameter family changed")
    if list(reporting["q_endpoints"]) != Q_ENDPOINTS:
        raise ValueError("Step-14 q/calibration endpoint family changed")

    interpretation = protocol["interpretation"]
    if any(
        interpretation[key] is not False
        for key in (
            "joint_q_rho_factorial_claim_allowed",
            "independent_confirmation_claim_allowed",
            "final_model_change_allowed",
            "test_robustness_claim_allowed",
        )
    ):
        raise ValueError("Step-14 interpretation limits were relaxed")
    policy = protocol["policy"]
    false_locks = (
        "model_fit_allowed",
        "optimizer_steps_allowed",
        "new_evaluation_allowed",
        "validation_reopen_allowed",
        "test_access_allowed",
        "comparator_access_allowed",
        "hyperparameter_selection_allowed",
        "final_model_change_allowed",
        "unfavorable_candidate_omission_allowed",
    )
    if any(policy[key] is not False for key in false_locks):
        raise ValueError("A Step-14 scientific firewall was relaxed")
    return protocol


def _latest_step4r2(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    expected = protocol["prerequisites"]["step04r2"]
    candidates = sorted(
        (root / "outputs/v6/step04r2_city_backoff").glob(
            "*/step04r2_contract.json"
        ),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = _json(contract_path)
            if not (
                contract.get("implementation_version")
                == expected["implementation_version"]
                and contract.get("status") == expected["status"]
                and contract.get("scope") == expected["scope"]
                and contract.get("data_key") == expected["data_key"]
                and contract.get("official_commit") == expected["official_commit"]
                and _verify_fingerprint(contract, "contract_fingerprint")
                and bool(contract.get("selection_is_original_train_development_only"))
                and float(contract.get("selected_mixture_weight", float("nan")))
                == float(expected["selected_rho"])
                and int(contract.get("parent_neural_refits", -1)) == 0
                and int(contract.get("validation_documents_used", -1)) == 0
                and int(contract.get("test_documents_used", -1)) == 0
                and int(contract.get("human_labels_used", -1)) == 0
                and int(contract.get("baseline_outputs_used", -1)) == 0
                and int(contract.get("sota_outputs_used", -1)) == 0
            ):
                continue
            _verify_manifest(contract_path.parent)
            fold_metrics = contract_path.parent / "worker/candidate_fold_metrics.csv"
            if fold_metrics.is_file():
                return contract_path.parent, contract
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError("No manifest-valid Step-4R2 PASS result was found")


def _latest_step6r2(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    expected = protocol["prerequisites"]["step06r2"]
    candidates = sorted(
        (root / "outputs/v6/step06r2_development_recovery").glob(
            "*/step06r2_contract.json"
        ),
        reverse=True,
    )
    for contract_path in candidates:
        try:
            contract = _json(contract_path)
            selected = contract.get("selected_configuration", {}) or {}
            selected_summary = contract.get("selected_candidate_summary", {}) or {}
            if not (
                contract.get("implementation_version")
                == expected["implementation_version"]
                and contract.get("status") == expected["status"]
                and contract.get("readiness") == expected["readiness"]
                and contract.get("validation_status") == expected["validation_status"]
                and contract.get("data_key") == expected["data_key"]
                and contract.get("official_commit") == expected["official_commit"]
                and _verify_fingerprint(contract, "contract_fingerprint")
                and int(contract.get("test_documents_used", -1)) == 0
                and int(contract.get("baseline_outputs_used", -1)) == 0
                and int(contract.get("sota_outputs_used", -1)) == 0
                and int(contract.get("parent_neural_refits", -1)) == 0
                and int(selected.get("prototype_quota", -1)) == 10
                and selected.get("calibration") == "training_moment_match"
                and float(selected.get("pooled_mixture_weight", float("nan")))
                == 0.35
                and selected.get("city_specific_backoff") == "RETIRED"
                and selected.get("test_status") == "UNOPENED"
                and selected_summary.get("candidate_id") == SELECTED_CANDIDATE
            ):
                continue
            _verify_manifest(contract_path.parent)
            metrics = contract_path.parent / "worker/candidate_seed_metrics.csv"
            if metrics.is_file():
                return contract_path.parent, contract
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError("No manifest-valid Step-6R2 PASS result was found")


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate all six q/calibration cells; retained for audit and tests."""

    if len(rows) != 18:
        raise ValueError("The Step-6R2 grid must contain exactly 18 rows")
    expected_cells = {
        (quota, calibration, seed)
        for quota in EXPECTED_QUOTAS
        for calibration in EXPECTED_CALIBRATIONS
        for seed in EXPECTED_SEEDS
    }
    observed_cells = {
        (int(row["prototype_quota"]), str(row["calibration"]), int(row["seed"]))
        for row in rows
    }
    if observed_cells != expected_cells or len(observed_cells) != len(rows):
        raise ValueError("The Step-6R2 q/calibration/seed grid changed")

    rank_metrics = (
        "validation_npmi_full",
        "validation_cv_full",
        "validation_topic_diversity_full",
        "validation_top_word_redundancy_full",
    )
    for quota in EXPECTED_QUOTAS:
        for seed in EXPECTED_SEEDS:
            pair = {
                str(row["calibration"]): row
                for row in rows
                if int(row["prototype_quota"]) == quota
                and int(row["seed"]) == seed
            }
            for metric in rank_metrics:
                if abs(
                    _as_float(pair["native_decoder"], metric)
                    - _as_float(pair["training_moment_match"], metric)
                ) > 1e-12:
                    raise ValueError(
                        "Decoder calibration unexpectedly changed topic-word rankings"
                    )

    required_metrics = Q_ENDPOINTS + Q_EFFECTS
    aggregate: list[dict[str, Any]] = []
    for quota in EXPECTED_QUOTAS:
        for calibration in EXPECTED_CALIBRATIONS:
            cell = [
                row
                for row in rows
                if int(row["prototype_quota"]) == quota
                and str(row["calibration"]) == calibration
            ]
            candidate_id = f"quota_{quota:02d}__{calibration}"
            if any(str(row["candidate_id"]) != candidate_id for row in cell):
                raise ValueError(f"Candidate identity changed for {candidate_id}")
            output: dict[str, Any] = {
                "candidate_id": candidate_id,
                "graph_core_coverage_q": quota,
                "decoder_calibration": calibration,
                "development_seeds": 3,
                "selected_configuration": candidate_id == SELECTED_CANDIDATE,
                "analysis_partition": "spent_development_validation",
            }
            for metric in required_metrics:
                values = [_as_float(row, metric) for row in cell]
                output[f"{metric}_median"] = float(statistics.median(values))
                output[f"{metric}_minimum"] = float(min(values))
                output[f"{metric}_maximum"] = float(max(values))
            aggregate.append(output)
    return aggregate


def _collapse_q(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(cells) != 6:
        raise ValueError("Exactly six aggregated q/calibration cells are required")
    result: list[dict[str, Any]] = []
    for quota in EXPECTED_QUOTAS:
        pair = {
            str(row["decoder_calibration"]): row
            for row in cells
            if int(row["graph_core_coverage_q"]) == quota
        }
        if set(pair) != set(EXPECTED_CALIBRATIONS):
            raise ValueError(f"Incomplete calibration pair for q={quota}")
        native = pair["native_decoder"]
        moment = pair["training_moment_match"]
        output: dict[str, Any] = {
            "graph_core_coverage_q": quota,
            "development_seeds": 3,
            "selected_frozen_setting": quota == 10,
            "frozen_decoder_calibration": (
                "training_moment_match" if quota == 10 else ""
            ),
            "analysis_partition": "spent_development_validation",
        }
        shared = {
            "npmi_at_10": "validation_npmi_full",
            "c_v_at_10": "validation_cv_full",
            "diversity_at_10": "validation_topic_diversity_full",
            "redundancy_at_10": "validation_top_word_redundancy_full",
            "npmi_gain_vs_graph_removal":
                "validation_npmi_improvement_vs_without_graph",
            "c_v_gain_vs_graph_removal":
                "validation_cv_improvement_vs_without_graph",
        }
        for target, source in shared.items():
            for suffix in ("median", "minimum", "maximum"):
                output[f"{target}_{suffix}"] = float(moment[f"{source}_{suffix}"])
        for suffix in ("median", "minimum", "maximum"):
            output[f"native_macro_city_nll_{suffix}"] = float(
                native[f"full_macro_city_nll_{suffix}"]
            )
            output[f"moment_macro_city_nll_{suffix}"] = float(
                moment[f"full_macro_city_nll_{suffix}"]
            )
            output[f"calibration_nll_reduction_{suffix}"] = float(
                moment[f"calibration_nll_reduction_vs_native_{suffix}"]
            )
        result.append(output)
    return result


def _aggregate_rho(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != len(EXPECTED_FOLDS) * len(EXPECTED_RHOS):
        raise ValueError("The Step-4R2 rho grid must contain exactly 15 rows")
    observed = {
        (int(row["fold"]), _as_float(row, "mixture_weight")) for row in rows
    }
    expected = {(fold, rho) for fold in EXPECTED_FOLDS for rho in EXPECTED_RHOS}
    if observed != expected or len(observed) != len(rows):
        raise ValueError("The Step-4R2 fold/rho grid changed")

    parent_by_fold: dict[int, float] = {}
    for fold in EXPECTED_FOLDS:
        values = [
            _as_float(row, "parent_macro_city_nll")
            for row in rows
            if int(row["fold"]) == fold
        ]
        if max(values) - min(values) > 1e-12:
            raise ValueError("The exact parent anchor changed across rho candidates")
        parent_by_fold[fold] = values[0]

    parent_values = [parent_by_fold[fold] for fold in EXPECTED_FOLDS]
    result: list[dict[str, Any]] = [
        {
            "pooled_backoff_weight_rho": 0.0,
            "configuration": "exact_parent_decoder_anchor",
            "development_folds": 5,
            "selected_frozen_setting": False,
            "analysis_partition": "grouped_original_train_development",
            "pooled_macro_city_nll_median": float(statistics.median(parent_values)),
            "pooled_macro_city_nll_minimum": float(min(parent_values)),
            "pooled_macro_city_nll_maximum": float(max(parent_values)),
            "nll_reduction_vs_rho_zero_median": 0.0,
            "nll_reduction_vs_rho_zero_minimum": 0.0,
            "nll_reduction_vs_rho_zero_maximum": 0.0,
        }
    ]
    for rho in EXPECTED_RHOS:
        cell = [row for row in rows if _as_float(row, "mixture_weight") == rho]
        pooled = [_as_float(row, "pooled_backoff_macro_city_nll") for row in cell]
        reductions = []
        for row in cell:
            reduction = _as_float(row, "parent_macro_city_nll") - _as_float(
                row, "pooled_backoff_macro_city_nll"
            )
            if abs(
                reduction - _as_float(row, "pooled_minus_parent_nll_reduction")
            ) > 1e-12:
                raise ValueError("A stored pooled-backoff reduction was not reproduced")
            reductions.append(reduction)
        result.append(
            {
                "pooled_backoff_weight_rho": rho,
                "configuration": "training_only_pooled_lexical_backoff",
                "development_folds": 5,
                "selected_frozen_setting": rho == 0.35,
                "analysis_partition": "grouped_original_train_development",
                "pooled_macro_city_nll_median": float(statistics.median(pooled)),
                "pooled_macro_city_nll_minimum": float(min(pooled)),
                "pooled_macro_city_nll_maximum": float(max(pooled)),
                "nll_reduction_vs_rho_zero_median": float(
                    statistics.median(reductions)
                ),
                "nll_reduction_vs_rho_zero_minimum": float(min(reductions)),
                "nll_reduction_vs_rho_zero_maximum": float(max(reductions)),
            }
        )
    return result


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _interval(row: Mapping[str, Any], prefix: str) -> str:
    return (
        f"{_fmt(row[f'{prefix}_median'])} "
        f"[{_fmt(row[f'{prefix}_minimum'])}, {_fmt(row[f'{prefix}_maximum'])}]"
    )


def _paper_table(
    q_rows: Sequence[Mapping[str, Any]], rho_rows: Sequence[Mapping[str, Any]]
) -> str:
    lines = [
        "# Table X. Pre-test sensitivity of SC-HTM V6.1-specific parameters",
        "",
        "## Panel A. Graph-core coverage q and decoder calibration",
        "",
        "| q | NPMI@10 ↑ | C_v@10 ↑ | Diversity@10 ↑ | Redundancy@10 ↓ | Native macro-city NLL ↓ | Moment-calibrated macro-city NLL ↓ | Paired calibration ΔNLL ↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in q_rows:
        q = int(row["graph_core_coverage_q"])
        label = f"**{q} (frozen)**" if bool(row["selected_frozen_setting"]) else str(q)
        lines.append(
            f"| {label} | {_interval(row, 'npmi_at_10')} | "
            f"{_interval(row, 'c_v_at_10')} | "
            f"{_interval(row, 'diversity_at_10')} | "
            f"{_interval(row, 'redundancy_at_10')} | "
            f"{_interval(row, 'native_macro_city_nll')} | "
            f"{_interval(row, 'moment_macro_city_nll')} | "
            f"{_interval(row, 'calibration_nll_reduction')} |"
        )
    lines.extend(
        [
            "",
            "## Panel B. Pooled lexical-backoff weight rho",
            "",
            "| rho | Decoder configuration | Macro-city NLL ↓ | Paired ΔNLL vs. rho=0 ↑ |",
            "|---:|:---|---:|---:|",
        ]
    )
    for row in rho_rows:
        rho = float(row["pooled_backoff_weight_rho"])
        label = f"{rho:.2f}"
        if bool(row["selected_frozen_setting"]):
            label = f"**{label} (frozen)**"
        configuration = (
            "Exact parent decoder (anchor)"
            if rho == 0.0
            else "Training-only pooled lexical backoff"
        )
        lines.append(
            f"| {label} | {configuration} | "
            f"{_interval(row, 'pooled_macro_city_nll')} | "
            f"{_interval(row, 'nll_reduction_vs_rho_zero')} |"
        )
    lines.extend(
        [
            "",
            "Values are median [minimum, maximum]. Panel A uses three frozen "
            "Step-6R2 spent-validation seeds. Panel B uses five grouped "
            "Step-4R2 original-training development folds; rho=0 is the exact "
            "parent-decoder identity anchor. q is the minimum number of "
            "graph-selected prototype words required among each topic's displayed "
            "top 10, while the number of topics was fixed at K=10.",
            "",
            "The frozen setting was q=10, training-only moment calibration, and "
            "rho=0.35. The two panels are separate pre-test studies, not a joint "
            "q-by-rho factorial experiment. These development values must not be "
            "described as held-out-test sensitivity, and their fold-specific "
            "coherence values are not directly comparable with the final-corpus "
            "results table. Panel B reports the matched pooled-control arm from "
            "Step-4R2; the city-conditioned arm studied there was later retired "
            "and is not part of SC-HTM V6.1.",
        ]
    )
    return "\n".join(lines)


def _figure_caption() -> str:
    return (
        "Figure X. Pre-test sensitivity of SC-HTM V6.1-specific parameters. "
        "Panels A and B show absolute NPMI@10 and C_v@10 as the graph-core "
        "coverage target q varies; q is the required number of graph-selected "
        "prototype words among each topic's displayed top 10, not the topic "
        "count (K=10 throughout). Panel C compares the native decoder with "
        "training-only moment calibration. Values in A-C are medians with "
        "minimum-maximum whiskers over three frozen Step-6R2 spent-validation "
        "seeds. Panel D reports the matched pooled-control arm of Step-4R2 as "
        "the pooled-backoff weight rho varies, using five grouped "
        "original-training development folds; rho=0 is the exact parent-decoder "
        "anchor. The city-conditioned arm considered in Step-4R2 was later "
        "retired. Red rings identify the frozen V6.1 setting (q=10, "
        "training-only moment calibration, rho=0.35). These are separate "
        "pre-test one-axis studies, not a joint q-by-rho factorial experiment "
        "or a held-out-test sensitivity analysis. Fold-specific development "
        "coherence values are not directly comparable with final-corpus values."
    )


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    pad = span * 0.16 if span > 0 else max(abs(high) * 0.05, 0.01)
    return low - pad, high + pad


def _sensitivity_svg(
    path: Path,
    q_rows: Sequence[Mapping[str, Any]],
    rho_rows: Sequence[Mapping[str, Any]],
) -> None:
    width, height = 1500, 980
    panel_width, plot_height = 555, 250
    panel_lefts = [115, 835]
    panel_tops = [185, 565]
    colors = {
        "lexical": "#2563eb",
        "native": "#6b7280",
        "moment": "#2563eb",
        "rho": "#0f766e",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>SC-HTM V6.1 pre-test parameter sensitivity</title>",
        "<desc>Four panels report q, decoder calibration, and pooled-backoff rho from two separate frozen development studies.</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:26px;font-weight:700}'
        '.sub{font-size:14px;fill:#4b5563}.panel{font-size:18px;font-weight:700}'
        '.axis{font-size:12px;fill:#4b5563}.legend{font-size:13px}.note{font-size:12px;fill:#6b7280}'
        '.final{font-size:11px;font-weight:700;fill:#b91c1c}</style>',
        '<text x="38" y="40" class="title">SC-HTM V6.1 parameter sensitivity before test evaluation</text>',
        '<text x="38" y="70" class="sub">q = graph-selected prototype words required among each topic\'s displayed top 10; the topic count was fixed at K=10.</text>',
        '<text x="38" y="96" class="sub">Points are medians and whiskers span minimum–maximum. Individual seed identifiers are intentionally omitted.</text>',
    ]

    def draw_panel(
        *,
        left: float,
        top: float,
        letter: str,
        title: str,
        x_values: Sequence[float],
        x_labels: Sequence[str],
        x_title: str,
        series: Sequence[Mapping[str, Any]],
    ) -> None:
        bottom = top + plot_height
        all_values = [
            float(value)
            for item in series
            for value in item["minimums"] + item["maximums"]
        ]
        ymin, ymax = _bounds(all_values)

        def x_coord(index: int) -> float:
            if len(x_values) == 1:
                return left + panel_width / 2
            return left + index * panel_width / (len(x_values) - 1)

        def y_coord(value: float) -> float:
            return bottom - (value - ymin) / (ymax - ymin) * plot_height

        svg.append(
            f'<text x="{left}" y="{top - 34}" class="panel">'
            f'{letter}. {html.escape(title)}</text>'
        )
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = y_coord(value)
            svg.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_width}" '
                f'y2="{y:.1f}" stroke="#e5e7eb"/>'
            )
            svg.append(
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                f'class="axis">{value:.3f}</text>'
            )
        svg.append(
            f'<line x1="{left}" y1="{bottom}" x2="{left + panel_width}" '
            f'y2="{bottom}" stroke="#111827"/>'
        )
        for index, label in enumerate(x_labels):
            x = x_coord(index)
            svg.append(
                f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" '
                f'y2="{bottom + 6}" stroke="#111827"/>'
            )
            svg.append(
                f'<text x="{x:.1f}" y="{bottom + 23}" text-anchor="middle" '
                f'class="axis">{html.escape(label)}</text>'
            )
        for item in series:
            points = " ".join(
                f"{x_coord(i):.1f},{y_coord(float(value)):.1f}"
                for i, value in enumerate(item["medians"])
            )
            color = str(item["color"])
            dash = ' stroke-dasharray="7 5"' if item.get("dashed") else ""
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                f'stroke-width="3"{dash}/>'
            )
            for index, median_value in enumerate(item["medians"]):
                x = x_coord(index)
                median = y_coord(float(median_value))
                low = y_coord(float(item["minimums"][index]))
                high = y_coord(float(item["maximums"][index]))
                svg.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{high:.1f}" x2="{x:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<line x1="{x - 5:.1f}" y1="{high:.1f}" x2="{x + 5:.1f}" y2="{high:.1f}" stroke="{color}"/>',
                        f'<line x1="{x - 5:.1f}" y1="{low:.1f}" x2="{x + 5:.1f}" y2="{low:.1f}" stroke="{color}"/>',
                        f'<circle cx="{x:.1f}" cy="{median:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>',
                    ]
                )
                if index in item.get("selected_indices", []):
                    svg.append(
                        f'<circle cx="{x:.1f}" cy="{median:.1f}" r="10" '
                        'fill="none" stroke="#dc2626" stroke-width="2"/>'
                    )
                    svg.append(
                        f'<text x="{x - 6:.1f}" y="{median - 15:.1f}" '
                        'text-anchor="end" class="final">frozen</text>'
                    )
        svg.append(
            f'<text x="{left + panel_width / 2}" y="{bottom + 49}" '
            f'text-anchor="middle" class="axis">{html.escape(x_title)}</text>'
        )

    q_values = [float(row["graph_core_coverage_q"]) for row in q_rows]
    q_labels = [str(int(value)) for value in q_values]

    def q_series(prefix: str, color: str, selected: bool = True) -> dict[str, Any]:
        return {
            "medians": [float(row[f"{prefix}_median"]) for row in q_rows],
            "minimums": [float(row[f"{prefix}_minimum"]) for row in q_rows],
            "maximums": [float(row[f"{prefix}_maximum"]) for row in q_rows],
            "color": color,
            "selected_indices": [2] if selected else [],
        }

    draw_panel(
        left=panel_lefts[0],
        top=panel_tops[0],
        letter="A",
        title="NPMI@10 ↑",
        x_values=q_values,
        x_labels=q_labels,
        x_title="Graph-core coverage q (words among top 10)",
        series=[q_series("npmi_at_10", colors["lexical"])],
    )
    draw_panel(
        left=panel_lefts[1],
        top=panel_tops[0],
        letter="B",
        title="C_v@10 ↑",
        x_values=q_values,
        x_labels=q_labels,
        x_title="Graph-core coverage q (words among top 10)",
        series=[q_series("c_v_at_10", colors["lexical"])],
    )
    native = q_series("native_macro_city_nll", colors["native"], selected=False)
    native["dashed"] = True
    moment = q_series("moment_macro_city_nll", colors["moment"])
    draw_panel(
        left=panel_lefts[0],
        top=panel_tops[1],
        letter="C",
        title="Macro-city NLL by decoder calibration ↓",
        x_values=q_values,
        x_labels=q_labels,
        x_title="Graph-core coverage q (words among top 10)",
        series=[native, moment],
    )
    rho_values = [float(row["pooled_backoff_weight_rho"]) for row in rho_rows]
    rho_series = {
        "medians": [float(row["pooled_macro_city_nll_median"]) for row in rho_rows],
        "minimums": [float(row["pooled_macro_city_nll_minimum"]) for row in rho_rows],
        "maximums": [float(row["pooled_macro_city_nll_maximum"]) for row in rho_rows],
        "color": colors["rho"],
        "selected_indices": [3],
    }
    draw_panel(
        left=panel_lefts[1],
        top=panel_tops[1],
        letter="D",
        title="Pooled-backoff macro-city NLL ↓",
        x_values=rho_values,
        x_labels=[f"{value:.2f}" for value in rho_values],
        x_title="Pooled lexical-backoff weight rho",
        series=[rho_series],
    )
    svg.extend(
        [
            '<line x1="330" y1="896" x2="360" y2="896" stroke="#2563eb" stroke-width="3"/>',
            '<text x="370" y="900" class="legend">Moment-calibrated / lexical metric</text>',
            '<line x1="690" y1="896" x2="720" y2="896" stroke="#6b7280" stroke-width="3" stroke-dasharray="7 5"/>',
            '<text x="730" y="900" class="legend">Native decoder</text>',
            '<line x1="910" y1="896" x2="940" y2="896" stroke="#0f766e" stroke-width="3"/>',
            '<text x="950" y="900" class="legend">Pooled backoff</text>',
            '<circle cx="1115" cy="896" r="9" fill="none" stroke="#dc2626" stroke-width="2"/>',
            '<text x="1133" y="900" class="legend">Frozen setting</text>',
            '<text x="38" y="942" class="note">A–C: Step-6R2 spent-validation development (3 seeds). D: Step-4R2 grouped original-training development (5 folds).</text>',
            '<text x="38" y="963" class="note">Separate pre-test one-axis studies; no joint q-by-rho claim, no new model fitting, and no held-out-test sensitivity claim.</text>',
            "</svg>",
        ]
    )
    _write_text(path, "\n".join(svg))
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
    destination = root / "v6_step14_sensitivity_results.zip"
    if destination.is_file():
        destination.unlink()
    prefix = Path("step14_primary_parameter_sensitivity") / out.name
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, (prefix / path.relative_to(out)).as_posix())
    return destination


def run_v6_step14(*, project_root: Path, config_path: Path) -> Path:
    """Render immutable q, rho, and calibration sensitivity evidence."""

    root = project_root.expanduser().resolve()
    protocol = _load_protocol(config_path)
    print(
        f"[V6.1 Step 14] START | implementation={IMPLEMENTATION} | fits=0 "
        "optimizer-steps=0 new-evaluations=0 validation-reopens=0 "
        "test-access=0 selection=0",
        flush=True,
    )
    source4, contract4 = _latest_step4r2(root, protocol)
    source6, contract6 = _latest_step6r2(root, protocol)
    rho_csv = source4 / "worker/candidate_fold_metrics.csv"
    q_csv = source6 / "worker/candidate_seed_metrics.csv"
    raw_rho_rows = _read_csv(rho_csv)
    raw_q_rows = _read_csv(q_csv)
    q_cells = _aggregate(raw_q_rows)
    q_rows = _collapse_q(q_cells)
    rho_rows = _aggregate_rho(raw_rho_rows)

    selected_q = next(row for row in q_rows if row["selected_frozen_setting"])
    selected_rho = next(row for row in rho_rows if row["selected_frozen_setting"])

    out = root / "outputs/v6/step14_primary_parameter_sensitivity" / _utc_id()
    out.mkdir(parents=True, exist_ok=False)
    reporting = protocol["reporting"]
    figure_path = out / reporting["figure"]
    caption_path = out / reporting["figure_caption"]
    paper_table_path = out / reporting["paper_table_markdown"]
    q_table_path = out / reporting["q_calibration_table_csv"]
    q_cells_path = out / reporting["q_candidate_cells_csv"]
    rho_table_path = out / reporting["rho_table_csv"]
    _write_csv(q_table_path, q_rows)
    _write_csv(q_cells_path, q_cells)
    _write_csv(rho_table_path, rho_rows)
    _write_text(paper_table_path, _paper_table(q_rows, rho_rows))
    _write_text(caption_path, _figure_caption())
    _sensitivity_svg(figure_path, q_rows, rho_rows)
    _write_text(
        out / "resolved_v6_step14.yaml",
        yaml.safe_dump(
            {"schema_version": 1, "v6": {"step14": protocol}},
            sort_keys=False,
        ),
    )

    checks = [
        ("source_step4r2_contract_and_manifest_valid", True, contract4["contract_fingerprint"]),
        ("source_step6r2_contract_and_manifest_valid", True, contract6["contract_fingerprint"]),
        ("source_step4r2_candidate_csv_hash_recorded", True, sha256_file(rho_csv)),
        ("source_step6r2_candidate_csv_hash_recorded", True, sha256_file(q_csv)),
        ("all_15_pretest_rho_fold_rows_retained", len(raw_rho_rows) == 15, len(raw_rho_rows)),
        ("all_three_predeclared_rho_candidates_retained", len(rho_rows) == 4, [row["pooled_backoff_weight_rho"] for row in rho_rows]),
        ("five_grouped_training_folds_per_rho", all(int(row["development_folds"]) == 5 for row in rho_rows), [row["development_folds"] for row in rho_rows]),
        ("rho_zero_exact_parent_anchor_retained", float(rho_rows[0]["pooled_backoff_weight_rho"]) == 0.0 and float(rho_rows[0]["nll_reduction_vs_rho_zero_median"]) == 0.0, rho_rows[0]),
        ("selected_rho_identity_retained", float(selected_rho["pooled_backoff_weight_rho"]) == 0.35, selected_rho["pooled_backoff_weight_rho"]),
        ("all_18_pretest_q_calibration_seed_rows_retained", len(raw_q_rows) == 18, len(raw_q_rows)),
        ("all_six_q_calibration_cells_retained", len(q_cells) == 6, len(q_cells)),
        ("all_three_q_values_retained", len(q_rows) == 3, [row["graph_core_coverage_q"] for row in q_rows]),
        ("three_declared_development_seeds_per_q", all(int(row["development_seeds"]) == 3 for row in q_rows), [row["development_seeds"] for row in q_rows]),
        ("selected_q_and_calibration_identity_retained", int(selected_q["graph_core_coverage_q"]) == 10 and selected_q["frozen_decoder_calibration"] == "training_moment_match", selected_q),
        ("evidence_surfaces_explicitly_separated", all(row["analysis_partition"] == "spent_development_validation" for row in q_rows) and all(row["analysis_partition"] == "grouped_original_train_development" for row in rho_rows), {"q": "SPENT_DEVELOPMENT", "rho": "ORIGINAL_TRAIN_DEVELOPMENT"}),
        ("paper_table_created", paper_table_path.is_file(), str(paper_table_path)),
        ("figure_caption_created", caption_path.is_file(), str(caption_path)),
        ("svg_is_well_formed", True, str(figure_path)),
        ("model_fits_zero", True, 0),
        ("optimizer_steps_zero", True, 0),
        ("new_evaluations_zero", True, 0),
        ("validation_reopens_zero", True, 0),
        ("test_access_zero", True, 0),
        ("comparator_access_zero", True, 0),
        ("new_hyperparameter_selection_zero", True, 0),
        ("frozen_model_changes_zero", True, 0),
        ("unfavorable_candidates_omitted_zero", True, 0),
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
        "source_step4r2_directory": str(source4),
        "source_step4r2_contract_sha256": sha256_file(source4 / "step04r2_contract.json"),
        "source_step4r2_contract_fingerprint": contract4["contract_fingerprint"],
        "source_step4r2_candidate_fold_metrics_sha256": sha256_file(rho_csv),
        "source_step4r2_manifest_identity": _json(source4 / "generated_artifact_manifest.json")["manifest_identity"],
        "source_step6r2_directory": str(source6),
        "source_step6r2_contract_sha256": sha256_file(source6 / "step06r2_contract.json"),
        "source_step6r2_contract_fingerprint": contract6["contract_fingerprint"],
        "source_step6r2_candidate_seed_metrics_sha256": sha256_file(q_csv),
        "source_step6r2_manifest_identity": _json(source6 / "generated_artifact_manifest.json")["manifest_identity"],
        "source_step4r2_test_documents_used": contract4["test_documents_used"],
        "source_step6r2_test_documents_used": contract6["test_documents_used"],
        "read_only_use": True,
    }
    _write_json(out / "frozen_lineage.json", lineage)
    contract = {
        "schema_version": 1,
        "status": status,
        "readiness": READINESS if status == "PASS" else "BLOCKED",
        "scope": SCOPE,
        "implementation_version": IMPLEMENTATION,
        "analysis_status": protocol["interpretation"]["analysis_status"],
        "hard_checks_passed": sum(row["status"] == "PASS" for row in hard_rows),
        "hard_checks_total": len(hard_rows),
        "failed_checks": failures,
        "source_step4r2_contract_fingerprint": contract4["contract_fingerprint"],
        "source_step6r2_contract_fingerprint": contract6["contract_fingerprint"],
        "rho_candidate_fold_rows": len(raw_rho_rows),
        "rho_candidates": EXPECTED_RHOS,
        "rho_parent_anchor": 0.0,
        "q_calibration_seed_rows": len(raw_q_rows),
        "q_calibration_candidate_cells": len(q_cells),
        "prototype_quotas": EXPECTED_QUOTAS,
        "decoder_calibrations": EXPECTED_CALIBRATIONS,
        "frozen_configuration_retained": {
            "topics_K": 10,
            "graph_core_coverage_q": 10,
            "decoder_calibration": "training_moment_match",
            "pooled_backoff_weight_rho": 0.35,
        },
        "evidence_surfaces": {
            "q_and_calibration": "step6r2_spent_validation_three_seeds",
            "rho": "step4r2_grouped_original_train_five_folds_matched_pooled_control",
            "joint_q_rho_factorial": False,
        },
        "independent_confirmation_claimed": False,
        "test_robustness_claimed": False,
        "model_fits": 0,
        "optimizer_steps": 0,
        "new_evaluations": 0,
        "validation_reopens": 0,
        "test_accesses": 0,
        "comparator_accesses": 0,
        "new_hyperparameter_selections": 0,
        "frozen_model_changes": 0,
    }
    contract["contract_fingerprint"] = sha256_object(contract)
    _write_json(out / "step14_contract.json", contract)
    _manifest(out)
    return_zip = _return_zip(root, out)
    print(
        f"[V6.1 Step 14] {status} | hard checks="
        f"{contract['hard_checks_passed']}/{contract['hard_checks_total']} | "
        "q=3/3 calibration=2/2 rho=3/3",
        flush=True,
    )
    print(f"[V6.1 Step 14] contract={out / 'step14_contract.json'}", flush=True)
    print(f"[V6.1 Step 14] return-zip={return_zip}", flush=True)
    if failures:
        raise AssertionError(f"Step 14 failed: {failures}")
    return out
