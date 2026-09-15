from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _step(name):
    document = yaml.safe_load(
        (ROOT / "configs" / "v6" / name).read_text(encoding="utf-8-sig")
    )
    key = "step4r2" if "r2" in name else "step4"
    return document["v6"][key]


def test_step4r2_preserves_parent_context_folds_and_evaluator():
    old = _step("step04.yaml")
    revised = _step("step04r2.yaml")
    for key in ("cities", "development_folds", "global_context", "model", "evaluation"):
        assert revised[key] == old[key]


def test_step4r2_uses_smallest_passing_conservative_weight():
    revised = _step("step04r2.yaml")
    assert revised["candidate_mixture_weights"] == [0.10, 0.20, 0.35]
    assert revised["selection"]["rule"] == "smallest_candidate_passing_every_gate"
    assert max(revised["candidate_mixture_weights"]) <= revised["gates"][
        "maximum_mixture_weight"
    ]


def test_step4r2_is_development_only_and_prohibits_comparison_access():
    revised = _step("step04r2.yaml")
    prohibited = revised["prohibited"]
    assert prohibited == {
        "parent_neural_refits": 0,
        "validation_rows_used": 0,
        "test_counts_or_text_used": 0,
        "human_labels_used": 0,
        "baseline_outputs_used": 0,
        "sota_outputs_used": 0,
        "package_install_commands": 0,
    }
    assert revised["selection"]["matched_control"].startswith("same_weight")


def test_step4r2_entry_point_is_exposed():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    assert '"step4r2"' in source
    assert "run_v6_step4r2" in source
