"""Fail-closed contracts for the V6 Step-1 mathematical core."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.v6 import reference
from src.v6.core import (
    V6Data,
    V6MathCore,
    V6Weights,
    objective,
    objective_from_logits,
    parameter_counts,
    rank_aware_lexical_objective,
    soft_topk_membership,
)


def distribution_checks(
    model: V6MathCore,
    data: V6Data,
    *,
    tolerance: float,
) -> dict[str, float | bool]:
    full = model.distributions(data)
    pooled = model.distributions(data, residual_scale=0.0)
    weights = data.city_weights / data.city_weights.sum()
    probability_errors = torch.stack(
        (
            torch.max(torch.abs(full.theta.sum(dim=1) - 1.0)),
            torch.max(torch.abs(full.shared_topics.sum(dim=1) - 1.0)),
            torch.max(torch.abs(full.city_topics.sum(dim=2) - 1.0)),
            torch.max(
                torch.abs(full.document_word_probabilities.sum(dim=1) - 1.0)
            ),
        )
    )
    minimum_probability = torch.min(
        torch.stack(
            (
                full.theta.min(),
                full.shared_topics.min(),
                full.city_topics.min(),
                full.document_word_probabilities.min(),
            )
        )
    )
    residual_center_error = torch.max(
        torch.abs(torch.einsum("c,ckd->kd", weights, full.city_residual_vectors))
    )
    effective_center_error = torch.max(
        torch.abs(torch.einsum("c,ckv->kv", weights, full.effective_logit_residuals))
    )
    pooled_error = torch.max(
        torch.abs(
            pooled.city_topics
            - pooled.shared_topics.unsqueeze(0).expand_as(pooled.city_topics)
        )
    )
    report = {
        "maximum_probability_sum_error": float(probability_errors.max().detach().cpu()),
        "minimum_probability": float(minimum_probability.detach().cpu()),
        "weighted_residual_center_error": float(residual_center_error.detach().cpu()),
        "weighted_effective_logit_center_error": float(
            effective_center_error.detach().cpu()
        ),
        "complete_pooling_identity_error": float(pooled_error.detach().cpu()),
    }
    report["passed"] = bool(
        report["maximum_probability_sum_error"] <= tolerance
        and report["minimum_probability"] > 0.0
        and report["weighted_residual_center_error"] <= tolerance
        and report["weighted_effective_logit_center_error"] <= tolerance
        and report["complete_pooling_identity_error"] <= tolerance
    )
    return report


def component_gradient_activity(
    model: V6MathCore,
    data: V6Data,
    weights: V6Weights,
    *,
    lexical_top_k: int,
    lexical_temperature: float,
    epsilon: float,
    activity_floor: float,
) -> list[dict[str, Any]]:
    breakdown = objective(
        model,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
    )
    parameters = {
        "document_topic": model.theta_logits,
        "shared_topic": model.shared_topic_logits,
        "city_contrast": model.city_contrast_coefficients,
        "adaptive_gate": model.gate_logits,
    }
    components = {
        "reconstruction": breakdown.reconstruction_nll_per_token,
        "pooling": breakdown.pooling,
        "gate_sparsity": breakdown.gate_sparsity,
        "rank_aware_lexical": breakdown.lexical,
    }
    expected = {
        "reconstruction": {
            "document_topic",
            "shared_topic",
            "city_contrast",
            "adaptive_gate",
        },
        "pooling": {"city_contrast"},
        "gate_sparsity": {"adaptive_gate"},
        "rank_aware_lexical": {"shared_topic"},
    }
    rows: list[dict[str, Any]] = []
    for component_name, value in components.items():
        gradients = torch.autograd.grad(
            value,
            tuple(parameters.values()),
            retain_graph=True,
            allow_unused=True,
        )
        for (parameter_name, _), gradient in zip(parameters.items(), gradients):
            norm = 0.0
            if gradient is not None:
                norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
            should_be_active = parameter_name in expected[component_name]
            passed = norm > activity_floor if should_be_active else norm <= activity_floor
            rows.append(
                {
                    "component": component_name,
                    "parameter_group": parameter_name,
                    "gradient_norm": norm,
                    "expected_active": should_be_active,
                    "passed": bool(passed),
                }
            )
    return rows


def rank_tail_gradient_check(
    *,
    dtype: torch.dtype,
    device: torch.device,
    temperature: float,
    activity_floor: float,
) -> dict[str, Any]:
    vocabulary = 14
    top_k = 10
    scores = torch.linspace(
        2.6, -1.3, vocabulary, dtype=dtype, device=device
    ).unsqueeze(0)
    scores.requires_grad_(True)
    graph = torch.zeros((vocabulary, vocabulary), dtype=dtype, device=device)
    with torch.no_grad():
        for left in range(top_k):
            for right in range(left + 1, top_k):
                value = 0.12 + 0.78 * (((left + 2) * (right + 3)) % 11) / 10.0
                graph[left, right] = value
                graph[right, left] = value
    lexical, _, membership = rank_aware_lexical_objective(
        scores,
        graph,
        top_k=top_k,
        temperature=temperature,
        epsilon=1.0e-10,
    )
    gradient = torch.autograd.grad(lexical, scores)[0].abs()[0]
    ranking = torch.argsort(scores.detach()[0], descending=True)
    head = ranking[:5]
    tail = ranking[5:10]
    tail_gradients = gradient.index_select(0, tail)
    mass_error = torch.max(torch.abs(membership.sum(dim=1) - float(top_k)))
    return {
        "top_k": top_k,
        "maximum_membership_mass_error": float(mass_error.detach().cpu()),
        "mean_rank_1_to_5_gradient": float(
            gradient.index_select(0, head).mean().detach().cpu()
        ),
        "minimum_rank_6_to_10_gradient": float(
            tail_gradients.min().detach().cpu()
        ),
        "mean_rank_6_to_10_gradient": float(
            tail_gradients.mean().detach().cpu()
        ),
        "all_ranks_6_to_10_active": bool(
            torch.all(tail_gradients > activity_floor).detach().cpu()
        ),
        "passed": bool(
            float(mass_error.detach().cpu()) <= 1.0e-8
            and torch.all(tail_gradients > activity_floor).detach().cpu()
        ),
    }


def ablation_identity_checks(
    model: V6MathCore,
    data: V6Data,
    weights: V6Weights,
    *,
    lexical_top_k: int,
    lexical_temperature: float,
    epsilon: float,
    tolerance: float,
) -> list[dict[str, Any]]:
    full = objective(
        model,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
    )
    no_lexical = objective(
        model,
        data,
        weights.without_lexical(),
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
    )
    lexical_identity_error = abs(
        float(
            (
                no_lexical.total
                - (full.total - weights.lexical * full.lexical)
            )
            .detach()
            .cpu()
        )
    )
    complete_pooling = model.distributions(data, residual_scale=0.0)
    pooling_identity_error = float(
        torch.max(
            torch.abs(
                complete_pooling.city_topics
                - complete_pooling.shared_topics.unsqueeze(0).expand_as(
                    complete_pooling.city_topics
                )
            )
        )
        .detach()
        .cpu()
    )
    counts = parameter_counts(
        topics=model.topics,
        vocabulary=data.vocabulary_size,
        cities=data.cities,
        dimension=data.embedding_dimension,
    )
    backbone_weights = V6Weights(pooling=0.0, gate=0.0, lexical=0.0)
    backbone = objective(
        model,
        data,
        backbone_weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
        residual_scale=0.0,
    )
    backbone_identity_error = abs(
        float(
            (backbone.total - backbone.reconstruction_nll_per_token)
            .detach()
            .cpu()
        )
    )
    return [
        {
            "ablation": "without_rank_aware_lexical_evidence",
            "identity_error": lexical_identity_error,
            "passed": bool(lexical_identity_error <= tolerance),
        },
        {
            "ablation": "complete_pooling_no_city_residual",
            "identity_error": pooling_identity_error,
            "passed": bool(pooling_identity_error <= tolerance),
        },
        {
            "ablation": "encot_backbone_both_extensions_off",
            "identity_error": backbone_identity_error,
            "passed": bool(backbone_identity_error <= tolerance),
        },
        {
            "ablation": "no_pooling_independent_city_decoder_capacity",
            "v6_parameters": counts["v6_low_rank_city_extension"],
            "independent_parameters": counts["independent_city_decoders"],
            "passed": bool(
                counts["v6_low_rank_city_extension"]
                < counts["independent_city_decoders"]
            ),
        },
    ]


def numpy_reference_check(
    model: V6MathCore,
    data: V6Data,
    *,
    lexical_top_k: int,
    lexical_temperature: float,
    epsilon: float,
    tolerance: float,
) -> dict[str, Any]:
    torch_values = model.distributions(data)
    arrays = {
        "theta_logits": model.theta_logits.detach().cpu().numpy(),
        "shared_topic_logits": model.shared_topic_logits.detach().cpu().numpy(),
        "city_contrast_coefficients": model.city_contrast_coefficients.detach()
        .cpu()
        .numpy(),
        "gate_logits": model.gate_logits.detach().cpu().numpy(),
        "vocabulary_embeddings": data.vocabulary_embeddings.detach().cpu().numpy(),
        "city_indices": data.city_indices.detach().cpu().numpy(),
        "city_weights": data.city_weights.detach().cpu().numpy(),
    }
    oracle = reference.distributions(**arrays)
    compared = {
        "theta": torch_values.theta,
        "shared_topics": torch_values.shared_topics,
        "city_topics": torch_values.city_topics,
        "document_word_probabilities": torch_values.document_word_probabilities,
        "city_residual_vectors": torch_values.city_residual_vectors,
        "gates": torch_values.gates,
        "effective_logit_residuals": torch_values.effective_logit_residuals,
    }
    errors = {
        name: float(
            np.max(np.abs(value.detach().cpu().numpy() - oracle[name]))
        )
        for name, value in compared.items()
    }
    torch_lexical, torch_association, torch_membership = (
        rank_aware_lexical_objective(
            model.shared_topic_logits,
            data.positive_npmi_graph,
            top_k=lexical_top_k,
            temperature=lexical_temperature,
            epsilon=epsilon,
        )
    )
    oracle_lexical = reference.rank_aware_association(
        arrays["shared_topic_logits"],
        data.positive_npmi_graph.detach().cpu().numpy(),
        top_k=lexical_top_k,
        temperature=lexical_temperature,
        epsilon=epsilon,
    )
    errors["rank_membership"] = float(
        np.max(
            np.abs(
                torch_membership.detach().cpu().numpy()
                - oracle_lexical["membership"]
            )
        )
    )
    errors["rank_association"] = float(
        np.max(
            np.abs(
                torch_association.detach().cpu().numpy()
                - oracle_lexical["association"]
            )
        )
    )
    errors["rank_loss"] = abs(
        float(torch_lexical.detach().cpu()) - float(oracle_lexical["loss"])
    )
    maximum = max(errors.values())
    return {
        "maximum_absolute_error": maximum,
        "errors": errors,
        "passed": bool(maximum <= tolerance),
    }


def soft_topk_gradcheck(
    *,
    temperature: float,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(620260907)
    scores = (
        0.4
        * torch.randn((2, 7), dtype=torch.float64, generator=generator)
    ).requires_grad_(True)
    probe_weights = torch.linspace(
        0.3, 1.4, 14, dtype=torch.float64
    ).reshape(2, 7)

    def function(values: torch.Tensor) -> torch.Tensor:
        membership = soft_topk_membership(
            values,
            top_k=3,
            temperature=temperature,
            epsilon=1.0e-12,
        )
        return torch.sum(membership * probe_weights)

    try:
        passed = bool(
            torch.autograd.gradcheck(
                function,
                (scores,),
                eps=1.0e-6,
                atol=5.0e-5,
                rtol=5.0e-4,
                raise_exception=True,
                fast_mode=False,
            )
        )
        message = ""
    except Exception as error:  # the contract records the exact failure
        passed = False
        message = f"{type(error).__name__}: {error}"
    return {
        "passed": passed,
        "parameters_checked": int(scores.numel()),
        "error": message,
    }


def vocabulary_permutation_check(
    model: V6MathCore,
    data: V6Data,
    *,
    tolerance: float,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(1907)
    permutation_cpu = torch.randperm(data.vocabulary_size, generator=generator)
    permutation = permutation_cpu.to(data.counts.device)
    permuted_data = V6Data(
        counts=data.counts.index_select(1, permutation),
        vocabulary_embeddings=data.vocabulary_embeddings.index_select(0, permutation),
        positive_npmi_graph=data.positive_npmi_graph.index_select(
            0, permutation
        ).index_select(1, permutation),
        city_indices=data.city_indices,
        city_weights=data.city_weights,
    )
    permuted_model = V6MathCore(
        documents=data.documents,
        topics=model.topics,
        vocabulary=data.vocabulary_size,
        cities=data.cities,
        embedding_dimension=data.embedding_dimension,
    ).to(device=data.counts.device, dtype=data.counts.dtype)
    with torch.no_grad():
        permuted_model.theta_logits.copy_(model.theta_logits)
        permuted_model.shared_topic_logits.copy_(
            model.shared_topic_logits.index_select(1, permutation)
        )
        permuted_model.city_contrast_coefficients.copy_(
            model.city_contrast_coefficients
        )
        permuted_model.gate_logits.copy_(model.gate_logits)
    original = model.distributions(data)
    permuted = permuted_model.distributions(permuted_data)
    expected_words = original.document_word_probabilities.index_select(1, permutation)
    error = float(
        torch.max(
            torch.abs(permuted.document_word_probabilities - expected_words)
        )
        .detach()
        .cpu()
    )
    return {"maximum_absolute_error": error, "passed": bool(error <= tolerance)}


def full_objective_gradcheck(
    data: V6Data,
    weights: V6Weights,
    *,
    lexical_temperature: float,
    epsilon: float,
) -> dict[str, Any]:
    """Finite-difference check of every free parameter on a tiny problem."""

    tiny_vocabulary = 7
    tiny_documents = 4
    tiny_topics = 2
    tiny_cities = 2
    tiny_dimension = 3
    generator = torch.Generator(device="cpu").manual_seed(260907)
    embeddings = torch.randn(
        (tiny_vocabulary, tiny_dimension),
        dtype=torch.float64,
        generator=generator,
    )
    embeddings = embeddings / torch.linalg.vector_norm(
        embeddings, dim=1, keepdim=True
    )
    graph = torch.rand(
        (tiny_vocabulary, tiny_vocabulary),
        dtype=torch.float64,
        generator=generator,
    )
    graph = 0.5 * (graph + graph.T)
    graph.fill_diagonal_(0.0)
    tiny = V6Data(
        counts=torch.rand(
            (tiny_documents, tiny_vocabulary),
            dtype=torch.float64,
            generator=generator,
        )
        + 0.1,
        vocabulary_embeddings=embeddings,
        positive_npmi_graph=graph,
        city_indices=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        city_weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    arguments = (
        (0.2 * torch.randn((tiny_documents, tiny_topics), dtype=torch.float64, generator=generator)).requires_grad_(True),
        (0.3 * torch.randn((tiny_topics, tiny_vocabulary), dtype=torch.float64, generator=generator)).requires_grad_(True),
        (0.2 * torch.randn((tiny_cities - 1, tiny_topics, tiny_dimension), dtype=torch.float64, generator=generator)).requires_grad_(True),
        (0.2 * torch.randn((tiny_cities, tiny_topics), dtype=torch.float64, generator=generator)).requires_grad_(True),
    )

    def function(*values: torch.Tensor) -> torch.Tensor:
        return objective_from_logits(
            values[0],
            values[1],
            values[2],
            values[3],
            tiny,
            weights,
            lexical_top_k=3,
            lexical_temperature=lexical_temperature,
            epsilon=epsilon,
        ).total

    try:
        passed = bool(
            torch.autograd.gradcheck(
                function,
                arguments,
                eps=1.0e-6,
                atol=8.0e-5,
                rtol=8.0e-4,
                raise_exception=True,
                fast_mode=False,
            )
        )
        message = ""
    except Exception as error:
        passed = False
        message = f"{type(error).__name__}: {error}"
    return {
        "passed": passed,
        "parameters_checked": int(sum(value.numel() for value in arguments)),
        "error": message,
    }
