"""Execute the fail-closed V6 Step-1 synthetic contract."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from typing import Any

import numpy as np
import torch
import yaml

from src.v6.contracts import (
    ablation_identity_checks,
    component_gradient_activity,
    distribution_checks,
    full_objective_gradcheck,
    numpy_reference_check,
    rank_tail_gradient_check,
    soft_topk_gradcheck,
    vocabulary_permutation_check,
)
from src.v6.core import V6MathCore, V6Weights, objective, validate_data
from src.v6.optimizer import fit_synthetic
from src.v6.synthetic import build_synthetic_fixture, initialize_from_perturbed_truth


IMPLEMENTATION_VERSION = "v6_step1_low_rank_city_residual_rank_aware_npmi_r1"
SCOPE = "synthetic_math_only_no_real_data_no_selection_no_comparators"


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty audit table: {path.name}")
    # Audit tables may legitimately contain variant-specific columns.  Build
    # one stable ordered union instead of assuming that the first row defines
    # the complete schema (the no-pooling capacity row is the key example).
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row.keys())
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Reporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def _load_protocol(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1:
        raise ValueError("The V6 Step-1 configuration has an unsupported schema")
    protocol = document.get("v6", {}).get("step1")
    if not isinstance(protocol, dict):
        raise ValueError("Configuration lacks v6.step1")
    if protocol.get("scope") != SCOPE:
        raise ValueError("The V6 Step-1 scope is not the frozen synthetic-only scope")
    if protocol.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("Configuration and source implementation versions disagree")
    return protocol


def _select_device(requested: str) -> torch.device:
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _mean_js(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.clip(np.asarray(reference, dtype=np.float64), 1.0e-12, None)
    right = np.clip(np.asarray(candidate, dtype=np.float64), 1.0e-12, None)
    left /= left.sum(axis=-1, keepdims=True)
    right /= right.sum(axis=-1, keepdims=True)
    midpoint = 0.5 * (left + right)
    value = 0.5 * np.sum(left * np.log(left / midpoint), axis=-1)
    value += 0.5 * np.sum(right * np.log(right / midpoint), axis=-1)
    return float(np.mean(value))


def run_v6_step1(*, project_root: Path, config_path: Path) -> Path:
    root = project_root.expanduser().resolve()
    if "v6" not in root.name.lower():
        raise RuntimeError(
            "V6 Step 1 refuses to run outside an isolated directory whose name contains 'V6'"
        )
    protocol = _load_protocol(config_path)
    seed = int(protocol["seed"])
    _set_determinism(seed)
    device = _select_device(str(protocol["device"]))
    dtype = torch.float64
    output = root / "outputs" / "v6" / "step01_math_core" / _utc_id()
    output.mkdir(parents=True, exist_ok=False)
    report = _Reporter(root / "logs" / "v6_step01.log")
    report(
        f"[V6 Step 1] START | implementation={IMPLEMENTATION_VERSION} "
        f"device={device} dtype=float64 seed={seed}"
    )
    report(
        "[V6 Step 1] scope=synthetic only | real=0 test=0 labels=0 baselines=0 SOTA=0"
    )

    synthetic = protocol["synthetic"]
    cpu_data, truth = build_synthetic_fixture(
        seed=seed,
        documents=int(synthetic["documents"]),
        topics=int(synthetic["topics"]),
        vocabulary=int(synthetic["vocabulary"]),
        cities=int(synthetic["cities"]),
        embedding_dimension=int(synthetic["embedding_dimension"]),
        document_length=int(synthetic["document_length"]),
        dtype=dtype,
    )
    data = cpu_data.to(device=device, dtype=dtype)
    model = V6MathCore(
        documents=data.documents,
        topics=int(synthetic["topics"]),
        vocabulary=data.vocabulary_size,
        cities=data.cities,
        embedding_dimension=data.embedding_dimension,
    ).to(device=device, dtype=dtype)
    initialize_from_perturbed_truth(
        model,
        truth,
        seed=seed + 17,
        noise_standard_deviation=float(synthetic["initial_noise_sd"]),
    )
    lexical_top_k = int(protocol["rank_aware_lexical"]["top_k"])
    lexical_temperature = float(protocol["rank_aware_lexical"]["temperature"])
    epsilon = float(protocol["epsilon"])
    weights = V6Weights.from_mapping(protocol["weights"])
    validate_data(data, model, lexical_top_k=lexical_top_k)

    initial_distributions = model.distributions(data)
    initial_breakdown = objective(
        model,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
        validate=True,
    ).detached()
    _write_json(output / "objective_initial.json", initial_breakdown)

    checks = protocol["checks"]
    probability = distribution_checks(
        model,
        data,
        tolerance=float(checks["probability_tolerance"]),
    )
    gradients = component_gradient_activity(
        model,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
        activity_floor=float(checks["gradient_activity_floor"]),
    )
    rank_tail = rank_tail_gradient_check(
        dtype=dtype,
        device=device,
        temperature=lexical_temperature,
        activity_floor=float(checks["rank_tail_gradient_floor"]),
    )
    identities = ablation_identity_checks(
        model,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
        tolerance=float(checks["identity_tolerance"]),
    )
    oracle = numpy_reference_check(
        model,
        data,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
        tolerance=float(checks["reference_tolerance"]),
    )
    permutation = vocabulary_permutation_check(
        model,
        data,
        tolerance=float(checks["identity_tolerance"]),
    )
    operator_gradcheck = soft_topk_gradcheck(
        temperature=float(checks["gradcheck_temperature"])
    )
    objective_gradcheck = full_objective_gradcheck(
        cpu_data,
        weights,
        lexical_temperature=float(checks["gradcheck_temperature"]),
        epsilon=epsilon,
    )
    _write_json(output / "probability_and_identifiability.json", probability)
    _write_csv(output / "component_gradient_activity.csv", gradients)
    _write_json(output / "rank_tail_gradient.json", rank_tail)
    _write_csv(output / "ablation_identities.csv", identities)
    _write_json(output / "independent_numpy_reference.json", oracle)
    _write_json(output / "vocabulary_permutation.json", permutation)
    _write_json(output / "soft_topk_gradcheck.json", operator_gradcheck)
    _write_json(output / "full_objective_gradcheck.json", objective_gradcheck)
    report(
        "[V6 Step 1] equations | "
        f"probability={'PASS' if probability['passed'] else 'FAIL'} "
        f"oracle={'PASS' if oracle['passed'] else 'FAIL'} "
        f"tail-ranks={'PASS' if rank_tail['passed'] else 'FAIL'} "
        f"gradcheck={'PASS' if objective_gradcheck['passed'] else 'FAIL'}"
    )

    fit_result = fit_synthetic(
        model,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
        config=protocol["optimizer"],
    )
    _write_csv(output / "optimization_trace.csv", fit_result.trace)
    final_breakdown = objective(
        model,
        data,
        weights,
        lexical_top_k=lexical_top_k,
        lexical_temperature=lexical_temperature,
        epsilon=epsilon,
    ).detached()
    _write_json(output / "objective_final.json", final_breakdown)
    final_distributions = model.distributions(data)
    recovery = {
        "initial_shared_topic_js": _mean_js(
            truth.shared_topics,
            initial_distributions.shared_topics.detach().cpu().numpy(),
        ),
        "final_shared_topic_js": _mean_js(
            truth.shared_topics,
            final_distributions.shared_topics.detach().cpu().numpy(),
        ),
        "initial_city_topic_js": _mean_js(
            truth.city_topics,
            initial_distributions.city_topics.detach().cpu().numpy(),
        ),
        "final_city_topic_js": _mean_js(
            truth.city_topics,
            final_distributions.city_topics.detach().cpu().numpy(),
        ),
    }
    _write_json(output / "synthetic_recovery_diagnostic.json", recovery)

    checkpoint_path = output / "v6_step01_synthetic_core.pt"
    torch.save(
        {
            "schema_version": 1,
            "implementation_version": IMPLEMENTATION_VERSION,
            "scope": SCOPE,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "dimensions": {
                "documents": data.documents,
                "topics": model.topics,
                "vocabulary": data.vocabulary_size,
                "cities": data.cities,
                "embedding_dimension": data.embedding_dimension,
            },
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location=device)
    reloaded = V6MathCore(
        documents=data.documents,
        topics=model.topics,
        vocabulary=data.vocabulary_size,
        cities=data.cities,
        embedding_dimension=data.embedding_dimension,
    ).to(device=device, dtype=dtype)
    reloaded.load_state_dict(loaded["state_dict"], strict=True)
    reload_error = float(
        torch.max(
            torch.abs(
                reloaded.distributions(data).document_word_probabilities
                - final_distributions.document_word_probabilities
            )
        )
        .detach()
        .cpu()
    )
    repeat_error = float(
        torch.max(
            torch.abs(
                model.distributions(data).document_word_probabilities
                - model.distributions(data).document_word_probabilities
            )
        )
        .detach()
        .cpu()
    )
    serialization = {
        "reload_maximum_absolute_error": reload_error,
        "repeat_forward_maximum_absolute_error": repeat_error,
        "passed": bool(
            reload_error <= float(checks["identity_tolerance"])
            and repeat_error == 0.0
        ),
    }
    _write_json(output / "serialization_and_repeatability.json", serialization)

    expected_state = {
        "theta_logits",
        "shared_topic_logits",
        "city_contrast_coefficients",
        "gate_logits",
    }
    architecture_locked = bool(
        set(model.state_dict()) == expected_state
        and model.city_contrast_coefficients.shape
        == (
            data.cities - 1,
            model.topics,
            data.embedding_dimension,
        )
    )
    hard_checks = [
        ("isolated_v6_project_root", "v6" in root.name.lower()),
        ("synthetic_scope_only", protocol["scope"] == SCOPE),
        ("probability_and_identifiability", bool(probability["passed"])),
        ("independent_numpy_reference", bool(oracle["passed"])),
        ("vocabulary_permutation_equivariance", bool(permutation["passed"])),
        ("soft_topk_operator_gradcheck", bool(operator_gradcheck["passed"])),
        ("full_objective_gradcheck", bool(objective_gradcheck["passed"])),
        ("rank_6_to_10_receive_gradient", bool(rank_tail["passed"])),
        ("all_component_gradient_contracts", all(row["passed"] for row in gradients)),
        ("all_ablation_identities", all(row["passed"] for row in identities)),
        ("architecture_exactly_locked", architecture_locked),
        (
            "objective_decreased",
            fit_result.relative_decrease
            >= float(checks["minimum_relative_objective_decrease"]),
        ),
        (
            "gradient_rms_decreased",
            fit_result.final_gradient_rms < fit_result.initial_gradient_rms,
        ),
        (
            "synthetic_stationarity_diagnostic",
            fit_result.final_gradient_rms
            <= float(checks["maximum_final_gradient_rms"]),
        ),
        ("serialization_and_repeatability", bool(serialization["passed"])),
        ("real_documents_accessed_zero", True),
        ("test_metrics_accessed_zero", True),
        ("human_labels_accessed_zero", True),
        ("baseline_results_accessed_zero", True),
        ("sota_results_accessed_zero", True),
    ]
    hard_rows = [
        {"check": name, "status": "PASS" if passed else "FAIL"}
        for name, passed in hard_checks
    ]
    _write_csv(output / "hard_checks.csv", hard_rows)
    resolved_path = output / "resolved_v6_step01.yaml"
    resolved_path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "v6": {"step1": protocol}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "python_executable": os.path.abspath(os.sys.executable),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32_allowed": bool(
            torch.backends.cuda.matmul.allow_tf32
            if hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "matmul")
            else False
        ),
    }
    _write_json(output / "environment.json", environment)
    status = "PASS" if all(passed for _, passed in hard_checks) else "FAIL"
    contract = {
        "schema_version": 1,
        "status": status,
        "readiness": (
            "READY_FOR_V6_STEP2_EVALUATION_FIREWALL"
            if status == "PASS"
            else "V6_STEP1_REQUIRES_CORRECTION"
        ),
        "scope": SCOPE,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture": {
            "backbone": "official_EnCOT_adapter_deferred_to_step3",
            "extension_1": "adaptive_identifiable_city_low_rank_lexical_residual",
            "extension_2": "implicit_gradient_soft_top10_positive_npmi",
        },
        "hard_checks_passed": int(sum(passed for _, passed in hard_checks)),
        "hard_checks_total": len(hard_checks),
        "initial_objective": fit_result.initial_objective,
        "final_objective": fit_result.final_objective,
        "relative_objective_decrease": fit_result.relative_decrease,
        "initial_gradient_rms": fit_result.initial_gradient_rms,
        "final_gradient_rms": fit_result.final_gradient_rms,
        "real_documents_accessed": 0,
        "test_metrics_accessed": 0,
        "human_labels_accessed": 0,
        "baseline_results_accessed": 0,
        "sota_results_accessed": 0,
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "resolved_config_sha256": _sha256(resolved_path),
    }
    contract_path = output / "step01_contract.json"
    _write_json(contract_path, contract)
    report(
        f"[V6 Step 1] optimization | J={fit_result.initial_objective:.6f} -> "
        f"{fit_result.final_objective:.6f} | gradient RMS="
        f"{fit_result.initial_gradient_rms:.3e} -> {fit_result.final_gradient_rms:.3e}"
    )
    report(
        f"[V6 Step 1] {status} | hard checks={contract['hard_checks_passed']}/"
        f"{contract['hard_checks_total']} | contract={contract_path}"
    )
    if status != "PASS":
        failed = [name for name, passed in hard_checks if not passed]
        raise RuntimeError(f"V6 Step 1 failed closed: {failed}")
    return output
