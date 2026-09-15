"""Small deterministic optimizer used only for V6 synthetic validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from src.v6.core import V6Data, V6MathCore, V6Weights, objective


@dataclass(frozen=True)
class FitResult:
    initial_objective: float
    final_objective: float
    relative_decrease: float
    initial_gradient_rms: float
    final_gradient_rms: float
    trace: list[dict[str, float | int | str]]


def _gradient_rms(model: V6MathCore) -> float:
    squared = 0.0
    count = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach() ** 2).cpu())
            count += parameter.numel()
    return (squared / max(count, 1)) ** 0.5


def fit_synthetic(
    model: V6MathCore,
    data: V6Data,
    weights: V6Weights,
    *,
    lexical_top_k: int,
    lexical_temperature: float,
    epsilon: float,
    config: dict[str, Any],
) -> FitResult:
    def evaluate() -> torch.Tensor:
        return objective(
            model,
            data,
            weights,
            lexical_top_k=lexical_top_k,
            lexical_temperature=lexical_temperature,
            epsilon=epsilon,
        ).total

    model.zero_grad(set_to_none=True)
    initial_tensor = evaluate()
    initial_tensor.backward()
    initial_rms = _gradient_rms(model)
    initial = float(initial_tensor.detach().cpu())
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["adam_learning_rate"])
    )
    best_value = initial
    best_state = deepcopy(model.state_dict())
    trace: list[dict[str, float | int | str]] = [
        {"phase": "initial", "iteration": 0, "objective": initial}
    ]
    iterations = int(config["adam_iterations"])
    trace_every = int(config["trace_every"])
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = evaluate()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite V6 synthetic objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        optimizer.step()
        with torch.no_grad():
            value = float(evaluate().detach().cpu())
        if value < best_value:
            best_value = value
            best_state = deepcopy(model.state_dict())
        if iteration % trace_every == 0 or iteration == iterations:
            trace.append(
                {"phase": "adam", "iteration": iteration, "objective": value}
            )
    model.load_state_dict(best_state)

    lbfgs_iterations = int(config.get("lbfgs_max_iterations", 0))
    if lbfgs_iterations > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=float(config.get("lbfgs_learning_rate", 0.5)),
            max_iter=lbfgs_iterations,
            max_eval=int(config.get("lbfgs_max_evaluations", lbfgs_iterations * 2)),
            tolerance_grad=float(config.get("lbfgs_tolerance_grad", 1.0e-8)),
            tolerance_change=float(config.get("lbfgs_tolerance_change", 1.0e-11)),
            history_size=int(config.get("lbfgs_history_size", 20)),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = evaluate()
            loss.backward()
            return loss

        lbfgs.step(closure)
        with torch.no_grad():
            value = float(evaluate().detach().cpu())
        if value < best_value:
            best_value = value
            best_state = deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        trace.append(
            {
                "phase": "lbfgs",
                "iteration": lbfgs_iterations,
                "objective": best_value,
            }
        )

    model.zero_grad(set_to_none=True)
    final_tensor = evaluate()
    final_tensor.backward()
    final_rms = _gradient_rms(model)
    final = float(final_tensor.detach().cpu())
    model.zero_grad(set_to_none=True)
    relative = (initial - final) / max(abs(initial), epsilon)
    return FitResult(
        initial_objective=initial,
        final_objective=final,
        relative_decrease=relative,
        initial_gradient_rms=initial_rms,
        final_gradient_rms=final_rms,
        trace=trace,
    )
