from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    document = yaml.safe_load(
        (ROOT / "configs" / "v6" / "step05r2.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    return document["v6"]["step5r2"]


def test_step5r2_consumes_passing_step4r2_and_freezes_city_weight():
    protocol = _protocol()
    assert protocol["implementation_version"] == (
        "v6_step5r2_target_aligned_graph_projection_gate_r5"
    )
    assert protocol["prerequisite"]["implementation_version"] == (
        "v6_step4r2_hierarchical_city_backoff_gate_r1"
    )
    assert protocol["prerequisite"]["selected_mixture_weight"] == 0.35
    assert protocol["city_backoff"]["fixed_mixture_weight"] == 0.35
    assert protocol["city_backoff"]["trainable_in_step5r2"] is False


def test_step5r2_is_a_paired_remove_projection_ablation():
    protocol = _protocol()
    projection = protocol["graph_prototype_projection"]
    assert projection["parent_state"] == "exact_frozen_passing_step4r2_checkpoint"
    assert projection["optimization"].startswith("closed_form_no_gradient")
    assert projection["candidate_prototype_quotas"] == [8, 9, 10]
    assert projection["selection_rule"] == (
        "smallest_predeclared_quota_passing_all_target_integrity_city_and_parent_benefit_gates"
    )


def test_step5r2_projection_is_zero_identity_and_likelihood_connected():
    protocol = _protocol()
    projection = protocol["graph_prototype_projection"]
    assert projection["zero_strength_identity"] == "exact_official_parent"
    assert projection["nonprototype_identity"] == "exact"
    assert "word_topic_simplex" in projection["placement"]
    assert protocol["evaluation"]["same_affinity_for_likelihood_and_reporting"]
    assert protocol["revision_basis"]["numeric_target_threshold_changes"] == "none"
    assert protocol["revision_basis"]["mechanism_change"].startswith("none_r5")


def test_step5r2_graph_and_prototype_contract_matches_support_audit():
    protocol = _protocol()
    graph = protocol["lexical_graph"]
    adapter = protocol["graph_prototype_projection"]
    assert graph["minimum_document_frequency"] == 2
    assert graph["minimum_joint_documents"] == 1
    assert graph["reliability_power"] == 1.25
    assert graph["minimum_active_words"] == 900
    assert adapter["construction"].startswith("spectral_partition")
    assert adapter["spectral_assign_labels"] == "cluster_qr"
    assert adapter["minimum_cluster_size"] >= protocol["evaluation"]["top_words"]


def test_step5r2_has_target_aligned_selection_gates_and_discloses_nll_tradeoff():
    protocol = _protocol()
    gates = protocol["gates"]
    required = {
        "minimum_median_heldout_npmi_improvement",
        "minimum_median_heldout_cv_improvement",
        "minimum_median_heldout_npmi",
        "minimum_median_standard_reference_npmi",
        "minimum_median_standard_reference_cv",
        "minimum_median_zero_joint_pair_reduction",
        "minimum_displayed_train_support_fraction",
        "minimum_displayed_graph_support_fraction",
        "minimum_median_city_vs_pooled_reduction",
        "minimum_median_city_vs_parent_reduction",
        "minimum_topic_diversity",
        "maximum_top_word_redundancy",
        "minimum_cross_fold_topic_stability",
        "minimum_prototype_core_recall",
        "maximum_zero_strength_identity_error",
        "maximum_nonprototype_change",
        "maximum_projection_column_sum_error",
        "minimum_projection_probability",
        "maximum_projection_optimality_error",
        "maximum_changed_columns",
    }
    assert required <= set(gates)
    assert gates["minimum_median_heldout_npmi_improvement"] > 0.01
    assert gates["minimum_prototype_core_recall"] == 1.0
    assert not any("nll_regression" in name for name in gates)
    tradeoff = protocol["reported_graph_ablation_nll_tradeoff"]
    assert tradeoff["required"] is True
    assert tradeoff["affects_candidate_eligibility"] is False
    assert tradeoff["r4_reference_maximum_median_nll_regression_vs_step4r2"] == 0.020
    assert protocol["ablation_interpretation"]["graph_removed_nll_comparison"] == (
        "mandatory_reported_tradeoff_not_selection_gate"
    )


def test_step5r2_uses_no_validation_test_labels_or_comparators():
    assert _protocol()["prohibited"] == {
        "step4r1_city_residual_reuse": 0,
        "validation_rows_used": 0,
        "test_counts_or_text_used": 0,
        "human_labels_used": 0,
        "baseline_outputs_used": 0,
        "sota_outputs_used": 0,
        "package_install_commands": 0,
    }


def test_step5r2_cli_route_is_current():
    source = (ROOT / "main_v6.py").read_text(encoding="utf-8")
    assert '"step5r2"' in source
    assert "run_v6_step5r2" in source
    assert 'args.stage == "step5"' not in source


def test_step5r2_uses_correct_non_empirical_bayes_name():
    estimator = _protocol()["city_backoff"]["estimator_name_for_manuscript"]
    assert estimator == "fixed_shrinkage_partially_pooled_city_unigram"
