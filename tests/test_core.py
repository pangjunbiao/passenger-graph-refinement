import csv

import pytest

torch = pytest.importorskip("torch")

from src.v6.contracts import numpy_reference_check, rank_tail_gradient_check
from src.v6.core import (
    V6MathCore,
    parameter_counts,
    soft_topk_membership,
    validate_data,
)
from src.v6.synthetic import build_synthetic_fixture, initialize_from_perturbed_truth
from src.v6.step1 import _write_csv


def _fixture():
    data, truth = build_synthetic_fixture(
        seed=79,
        documents=24,
        topics=3,
        vocabulary=36,
        cities=3,
        embedding_dimension=6,
        document_length=30,
        dtype=torch.float64,
    )
    model = V6MathCore(
        documents=data.documents,
        topics=3,
        vocabulary=data.vocabulary_size,
        cities=data.cities,
        embedding_dimension=data.embedding_dimension,
    ).double()
    initialize_from_perturbed_truth(
        model, truth, seed=91, noise_standard_deviation=0.2
    )
    return data, model


def test_complete_pooling_is_exact_identity():
    data, model = _fixture()
    validate_data(data, model, lexical_top_k=10)
    values = model.distributions(data, residual_scale=0.0)
    expected = values.shared_topics.unsqueeze(0).expand_as(values.city_topics)
    assert torch.max(torch.abs(values.city_topics - expected)).item() < 1.0e-12


def test_torch_soft_topk_mass():
    scores = torch.tensor(
        [[2.1, 1.4, 0.8, 0.1, -0.6]], dtype=torch.float64, requires_grad=True
    )
    membership = soft_topk_membership(scores, top_k=3, temperature=0.4)
    assert torch.max(torch.abs(membership.sum(dim=1) - 3.0)).item() < 1.0e-9
    membership[:, 2:].sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_ranks_six_to_ten_are_active():
    result = rank_tail_gradient_check(
        dtype=torch.float64,
        device=torch.device("cpu"),
        temperature=0.35,
        activity_floor=1.0e-9,
    )
    assert result["passed"], result


def test_independent_numpy_oracle_agrees():
    data, model = _fixture()
    result = numpy_reference_check(
        model,
        data,
        lexical_top_k=10,
        lexical_temperature=0.35,
        epsilon=1.0e-10,
        tolerance=2.0e-9,
    )
    assert result["passed"], result


def test_low_rank_extension_has_lower_capacity_than_no_pooling():
    values = parameter_counts(topics=10, vocabulary=650, cities=3, dimension=64)
    assert values["v6_low_rank_city_extension"] < values["independent_city_decoders"]


def test_audit_csv_supports_variant_specific_columns(tmp_path):
    destination = tmp_path / "heterogeneous_rows.csv"
    rows = [
        {"ablation": "complete_pooling", "identity_error": 0.0, "passed": True},
        {
            "ablation": "no_pooling_capacity",
            "v6_parameters": 100,
            "independent_parameters": 1000,
            "passed": True,
        },
    ]
    _write_csv(destination, rows)
    with destination.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        written = list(reader)
    assert reader.fieldnames == [
        "ablation",
        "identity_error",
        "passed",
        "v6_parameters",
        "independent_parameters",
    ]
    assert written[1]["v6_parameters"] == "100"
    assert written[0]["v6_parameters"] == ""
