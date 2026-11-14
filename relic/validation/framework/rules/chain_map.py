"""Validation rules for DEC chain maps and conservative regridding."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)


def _iter_maps(payload: Any) -> Iterable[Any]:
    if payload is None:
        return ()
    if isinstance(payload, (list, tuple)):
        return payload
    return (payload,)


def _require_tensor(payload: Mapping[str, Any], key: str) -> Tensor:
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"{key} must be a torch.Tensor.")
    if value.ndim != 2:
        raise ValueError(f"{key} must be rank-2.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{key} must be finite.")
    return value


def _require_vector(payload: Mapping[str, Any], key: str) -> Tensor:
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"{key} must be a torch.Tensor.")
    if value.ndim != 1:
        raise ValueError(f"{key} must be rank-1.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{key} must be finite.")
    return value


def _require_scalar(payload: Mapping[str, Any], key: str) -> Tensor:
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"{key} must be a torch.Tensor scalar.")
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError(f"{key} must be a finite torch.Tensor scalar.")
    return value.reshape(())


def _finite_nonnegative(payload: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite non-negative number.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be a finite non-negative number.")
    return value


_OPEN_LEDGER_ROUTES = {
    "port",
    "boundary_export",
    "managed_export",
    "diagnostic_residual",
    "observation",
}


def _validate_bihss_morphism(
    payload: Mapping[str, Any],
    *,
    state_map: Tensor,
    default_tol: float,
) -> dict[str, float]:
    """Validate the additional BIHSS morphism identities."""

    metrics: dict[str, float] = {}
    source_ledger = _require_tensor(payload, "source_ledger")
    target_ledger = _require_tensor(payload, "target_ledger")
    ledger_tol = _finite_nonnegative(payload, "ledger_tol", default_tol)
    mapped_ledger = target_ledger @ state_map
    if mapped_ledger.shape != source_ledger.shape:
        raise ValueError("BIHSS ledger preservation has incompatible matrix shapes.")

    compensation = payload.get("ledger_port_compensation")
    open_route = str(payload.get("open_port_route", "")).strip().lower()
    if compensation is not None:
        if open_route not in _OPEN_LEDGER_ROUTES:
            raise ValueError(
                "BIHSS ledger port compensation requires an explicit valid open_port_route."
            )
        if not isinstance(compensation, Tensor) or compensation.ndim != 2:
            raise TypeError("ledger_port_compensation must be a rank-2 torch.Tensor.")
        if compensation.shape != mapped_ledger.shape or not torch.isfinite(compensation).all():
            raise ValueError("ledger_port_compensation must be finite and match the ledger shape.")
        mapped_ledger = mapped_ledger + compensation
    elif open_route:
        raise ValueError("open_port_route requires ledger_port_compensation.")

    # References:
    #   - BIHSS theory, Definition of a BIHSS morphism, conditions (ii)-(v).
    # Mathematical expressions:
    #   C_target Phi + C_port = C_source; E_target(Phi x) ~= E_source(x);
    #   y_target^T Psi u = y_source^T u; sigma_target >= sigma_source - tolerance.
    # Parameter meanings:
    #   state_map is Phi; port_map is Psi; *_tol are declared approximation bounds.
    ledger_residual = float(
        torch.linalg.norm(mapped_ledger - source_ledger).detach().cpu().item()
    )
    metrics["ledger_residual_norm"] = ledger_residual
    if ledger_residual > ledger_tol:
        raise ValueError(
            f"BIHSS ledger preservation failed; residual norm = {ledger_residual}."
        )

    source_state = _require_vector(payload, "source_state")
    target_state = _require_vector(payload, "target_state")
    state_tol = _finite_nonnegative(payload, "state_tol", default_tol)
    mapped_state = state_map @ source_state
    if mapped_state.shape != target_state.shape:
        raise ValueError("BIHSS state map has incompatible state-vector shapes.")
    state_residual = float(torch.linalg.norm(mapped_state - target_state).detach().cpu().item())
    metrics["state_residual_norm"] = state_residual
    if state_residual > state_tol:
        raise ValueError(f"BIHSS target_state is not state_map @ source_state; residual = {state_residual}.")

    storage_tol = _finite_nonnegative(payload, "storage_tol", default_tol)
    source_energy = _require_scalar(payload, "source_energy")
    target_energy = _require_scalar(payload, "target_energy")
    source_entropy = _require_scalar(payload, "source_entropy")
    target_entropy = _require_scalar(payload, "target_entropy")
    energy_error = float(torch.abs(target_energy - source_energy).detach().cpu().item())
    entropy_error = float(torch.abs(target_entropy - source_entropy).detach().cpu().item())
    metrics["energy_storage_error"] = energy_error
    metrics["entropy_storage_error"] = entropy_error
    if energy_error > storage_tol:
        raise ValueError(f"BIHSS energy storage error {energy_error} exceeds storage_tol.")
    if entropy_error > storage_tol:
        raise ValueError(f"BIHSS entropy storage error {entropy_error} exceeds storage_tol.")

    port_map = _require_tensor(payload, "port_map")
    source_input = _require_vector(payload, "source_input")
    target_input = _require_vector(payload, "target_input")
    source_effort = _require_vector(payload, "source_effort")
    target_effort = _require_vector(payload, "target_effort")
    power_tol = _finite_nonnegative(payload, "power_tol", default_tol)
    mapped_input = port_map @ source_input
    if mapped_input.shape != target_input.shape:
        raise ValueError("BIHSS port map has incompatible input-vector shapes.")
    port_map_residual = float(torch.linalg.norm(mapped_input - target_input).detach().cpu().item())
    metrics["port_map_residual_norm"] = port_map_residual
    if port_map_residual > power_tol:
        raise ValueError(f"BIHSS target_input is not port_map @ source_input; residual = {port_map_residual}.")
    if source_effort.shape != source_input.shape or target_effort.shape != target_input.shape:
        raise ValueError("BIHSS effort and input vectors must have matching shapes.")
    source_power = torch.dot(source_effort, source_input)
    target_power = torch.dot(target_effort, target_input)
    power_residual = float(torch.abs(target_power - source_power).detach().cpu().item())
    metrics["port_power_residual"] = power_residual
    if power_residual > power_tol:
        raise ValueError(f"BIHSS port power preservation failed; residual = {power_residual}.")

    entropy_tol = _finite_nonnegative(payload, "entropy_tol", default_tol)
    source_sigma = _require_scalar(payload, "source_entropy_production")
    target_sigma = _require_scalar(payload, "target_entropy_production")
    entropy_gap = float((source_sigma - target_sigma).detach().cpu().item())
    metrics["entropy_production_gap"] = entropy_gap
    if entropy_gap > entropy_tol:
        raise ValueError(
            "BIHSS entropy production is systematically reduced; "
            f"gap = {entropy_gap}."
        )
    return metrics


def validate_chain_map_payload(payload: Mapping[str, Any]) -> ValidationResultModel:
    """Validate R_next @ D_source ~= D_target @ R_k."""

    name = str(payload.get("name", "chain_map"))
    metrics: dict[str, float] = {}
    try:
        source_D = _require_tensor(payload, "source_D")
        target_D = _require_tensor(payload, "target_D")
        R_k = _require_tensor(payload, "R_k")
        R_next = _require_tensor(payload, "R_next")
        tol = _finite_nonnegative(payload, "tol", 1.0e-10)
        # References:
        #   - Arnold, Falk, and Winther (2006), Acta Numerica 15:1-155.
        # Mathematical expression:
        #   R_{k+1} D_source = D_target R_k.
        # Parameter meanings:
        #   R_k and R_next are cochain maps; tol bounds the commutator norm.
        left = R_next @ source_D
        right = target_D @ R_k
        if left.shape != right.shape:
            raise ValueError(
                f"chain map '{name}' residual shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}."
            )
        residual_norm = float(torch.linalg.norm(left - right).detach().cpu().item())
        if residual_norm > tol:
            raise ValueError(
                f"chain map '{name}' does not commute; residual norm = {residual_norm}."
            )
        metrics["residual_norm"] = residual_norm
        if bool(payload.get("bihss_morphism", False)):
            state_map_raw = payload.get("state_map", R_k)
            if not isinstance(state_map_raw, Tensor) or state_map_raw.ndim != 2:
                raise TypeError("state_map must be a rank-2 torch.Tensor.")
            if not torch.isfinite(state_map_raw).all():
                raise ValueError("state_map must be finite.")
            metrics.update(
                _validate_bihss_morphism(
                    payload,
                    state_map=state_map_raw,
                    default_tol=tol,
                )
            )
    except (TypeError, ValueError, RuntimeError) as exc:
        # Structural validator boundary: the payload helpers and DEC chain-map
        # checks report violations with TypeError/ValueError, torch shape
        # mismatches raise RuntimeError; each becomes a failed result. Other
        # errors are defects and propagate.
        return ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=str(exc),
            passed=False,
            layer="L2",
            scope="chain_map_structure",
            tags=["dec", "chain_map", "structural", "naturality", "regridding"],
            details=[
                ValidationDetail(
                    variable_name=name,
                    expected="R_next @ D_source = D_target @ R_k",
                    value=str(exc),
                )
            ],
        )

    return ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.GENERIC_VALIDATION_OK,
        message=(
            f"Chain map '{name}' satisfies the BIHSS morphism contract."
            if bool(payload.get("bihss_morphism", False))
            else f"Chain map '{name}' commutes with the discrete derivative."
        ),
        passed=True,
        layer="L2",
        scope="chain_map_structure",
        tags=["dec", "chain_map", "structural", "naturality", "regridding"],
        details=[
            ValidationDetail(
                variable_name=name,
                expected="R_next @ D_source = D_target @ R_k",
                value=float(residual_norm),
            )
        ],
        metrics=metrics,
    )


@validated_rule(
    "chain_map_structure",
    "L2",
    name="DEC chain map structure",
    description="Validate commuting discrete derivative diagrams for regridding.",
    domains=["computation", "coupling"],
    tags=["dec", "chain_map", "structural", "naturality", "regridding"],
    references=[
        "Arnold, Falk, and Winther (2006), Finite Element Exterior Calculus",
        "Cotter (2023), compatible finite elements for geophysical fluid dynamics",
    ],
)
def validate_chain_map_structure(context: ValidationContext) -> ValidationResults:
    """Validate chain-map payloads in ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    payloads = metadata.get("chain_maps") or metadata.get("chain_map")
    return [
        validate_chain_map_payload(payload)
        for payload in _iter_maps(payloads)
        if isinstance(payload, Mapping)
    ]


__all__ = ["validate_chain_map_payload", "validate_chain_map_structure"]
