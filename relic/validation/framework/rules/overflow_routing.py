"""Validation rules for overflow and residual routing."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from relic.data.standards.dimensionality import dimensionality_from_unit

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)


_ALLOWED_ROUTES = {
    "storage",
    "boundary_outflow",
    "managed_export",
    "diagnostic_residual",
    "rejected_step",
}
_NONCANONICAL_PAYLOAD_KEYS = {
    "overflow_routes",
    "residual_routes",
    "residuals",
}
_NONCANONICAL_ROUTE_KEYS = {
    "boundary_port",
    "destination_type",
    "overflow",
    "port",
    "port_name",
    "residual",
    "residual_name",
    "route",
    "storage_variable",
    "target_variable",
    "unconsumed_flux",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _lower_text(value: Any) -> str:
    return _clean_text(value).lower()


def _contains_bool_or_complex(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.dtype == torch.bool or torch.is_complex(value)
    if isinstance(value, (bool, complex)):
        return True
    dtype_kind = getattr(getattr(value, "dtype", None), "kind", None)
    if dtype_kind in {"b", "c"}:
        return True
    if dtype_kind == "O":
        return any(_contains_bool_or_complex(item) for item in value.flat)
    if isinstance(value, Mapping):
        return any(_contains_bool_or_complex(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_bool_or_complex(item) for item in value)
    return False


def _as_numeric_tensor(value: Any) -> torch.Tensor | None:
    if value is None or _contains_bool_or_complex(value):
        return None
    if isinstance(value, torch.Tensor) and (
        value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
    ):
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    if (
        tensor.device.type == "meta"
        or tensor.layout != torch.strided
        or tensor.is_quantized
        or tensor.numel() == 0
        or tensor.is_complex()
        or not bool(torch.isfinite(tensor).all())
    ):
        return None
    return tensor


def _tensor_magnitude(tensor: torch.Tensor) -> float:
    magnitude = torch.sum(torch.abs(tensor))
    if not bool(torch.isfinite(magnitude).item()):
        return float("inf")
    return float(magnitude.detach().cpu().item())


def _iter_routes(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    routes = payload.get("routes", ())
    if isinstance(routes, Mapping):
        return (routes,)
    if isinstance(routes, (list, tuple)):
        return tuple(routes)
    if routes is None:
        return ()
    return (routes,)


def _route_name(route: Mapping[str, Any]) -> str:
    return _clean_text(route.get("name"))


def _route_type(route: Mapping[str, Any]) -> str:
    return _lower_text(route.get("route_type"))


def _route_amount_value(route: Mapping[str, Any]) -> Any:
    return route.get("amount")


def _validate_route_target(route: Mapping[str, Any], route_type: str) -> str:
    raw_target = route.get("target")
    if not isinstance(raw_target, str) or not raw_target.strip():
        return "route requires a non-empty string target."
    target = raw_target.strip()
    if route_type == "diagnostic_residual":
        if not target.startswith("diagnostics."):
            return "diagnostic_residual route requires target under diagnostics."
    elif route_type == "rejected_step":
        if route.get("step_rejected") is not True:
            return "rejected_step route requires step_rejected=true."
    return ""


def _detail(
    *,
    variable_name: str,
    expected: Any,
    value: Any,
    threshold: float | None = None,
) -> ValidationDetail:
    """Build a validation detail without duplicating optional model fields."""
    payload = {
        "variable_name": variable_name,
        "expected": expected,
        "value": value,
    }
    if threshold is not None:
        payload["threshold"] = threshold
    return ValidationDetail.model_validate(payload)


def validate_overflow_routing_payload(
    payload: Mapping[str, Any],
    *,
    strict: bool = False,
) -> ValidationResultModel:
    """Validate that nonzero residuals are explicitly routed.

    References:
      - Finlayson (2012), Introduction to Chemical Engineering Computing.
      - Hundsdorfer and Verwer (2003), advection-diffusion-reaction solvers.
      - van der Schaft and Maschke (1995), open-system port balances.
    Mathematical expression:
      For each declared residual r_k, require a route map rho(k) such that
      r_k -> storage, boundary_outflow, managed_export, diagnostic_residual,
      or rejected_step. Optional routed_amount must satisfy
      |routed_amount| = |r_k| within tolerance.
    Parameter meanings:
      amount: residual tensor to route; zero changes activity, not route validity.
      route_type: declared destination class for the residual.
      target: state, diagnostic, or boundary port receiving the residual.
      routed_amount: optional tensor whose absolute values must match amount.
    Evidence boundary:
      Target and route tokens are structural declarations only. Runtime residual
      coverage, StateManager/Graph port binding, and no_residuals evidence remain [N].
    """

    routes = _iter_routes(payload)
    details: list[ValidationDetail] = []
    route_count = 0
    active_count = 0
    routed_structural_count = 0
    invalid_route_count = 0
    for key in sorted(_NONCANONICAL_PAYLOAD_KEYS.intersection(payload)):
        details.append(
            _detail(
                variable_name="overflow_routing",
                expected="canonical routes field",
                value=f"noncanonical field '{key}'",
            )
        )
    if "required_residual_names" in payload:
        details.append(
            _detail(
                variable_name="required_residual_names",
                expected=(
                    "runtime residual coverage remains [N]; payload coverage "
                    "cannot self-certify completeness"
                ),
                value=str(payload.get("required_residual_names")),
            )
        )
    try:
        if isinstance(payload.get("tolerance", 1.0e-12), bool):
            raise TypeError("bool is not a tolerance")
        tolerance = float(payload.get("tolerance", 1.0e-12))
    except (TypeError, ValueError, OverflowError):
        tolerance = 1.0e-12
        details.append(
            _detail(
                variable_name="tolerance",
                expected="finite non-negative tolerance",
                value=payload.get("tolerance"),
            )
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        details.append(
            _detail(
                variable_name="tolerance",
                expected="finite non-negative tolerance",
                value=tolerance,
            )
        )
        tolerance = 1.0e-12

    no_residuals_value = payload.get("no_residuals", False)
    if "no_residuals" in payload and not isinstance(no_residuals_value, bool):
        details.append(
            _detail(
                variable_name="no_residuals",
                expected="no_residuals is bool",
                value=type(no_residuals_value).__name__,
            )
        )
    no_residuals = no_residuals_value is True
    if strict and not routes:
        details.append(
            _detail(
                variable_name="overflow_routing",
                expected=(
                    "non-empty routes; [N] no_residuals requires an "
                    "executable runtime residual audit emitted by the "
                    "canonical compile/runtime path"
                ),
                value=(
                    "unbound no_residuals claim"
                    if no_residuals
                    else "empty payload"
                ),
            )
        )
    if strict and routes and no_residuals:
        details.append(
            _detail(
                variable_name="no_residuals",
                expected="false when residual routes are declared",
                value=True,
            )
        )

    seen_names: set[str] = set()
    for index, raw_route in enumerate(routes):
        route_detail_start = len(details)
        if not isinstance(raw_route, Mapping):
            details.append(
                _detail(
                    variable_name=f"routes[{index}]",
                    expected="route item is a mapping",
                    value=type(raw_route).__name__,
                )
            )
            invalid_route_count += 1
            continue
        route_count += 1
        route = raw_route
        aliases = sorted(_NONCANONICAL_ROUTE_KEYS.intersection(route))
        if aliases:
            details.append(
                _detail(
                    variable_name=f"routes[{index}]",
                    expected=(
                        "canonical route fields: name, amount, units, route_type, "
                        "target, routed_amount, required, step_rejected"
                    ),
                    value=str(aliases),
                )
            )

        raw_name = route.get("name")
        name = _route_name(route)
        if not isinstance(raw_name, str) or not name:
            details.append(
                _detail(
                    variable_name=f"routes[{index}]",
                    expected="non-empty string route name",
                    value=raw_name,
                )
            )
            display_name = f"routes[{index}]"
        else:
            display_name = name
            if name in seen_names:
                details.append(
                    _detail(
                        variable_name=name,
                        expected="unique route name",
                        value=name,
                    )
                )
            seen_names.add(name)

        required_value = route.get("required", False)
        if "required" in route and not isinstance(required_value, bool):
            details.append(
                _detail(
                    variable_name=display_name,
                    expected="required is bool",
                    value=type(required_value).__name__,
                )
            )
        required = required_value is True

        step_rejected_value = route.get("step_rejected", False)
        if "step_rejected" in route and not isinstance(step_rejected_value, bool):
            details.append(
                _detail(
                    variable_name=display_name,
                    expected="step_rejected is bool",
                    value=type(step_rejected_value).__name__,
                )
            )

        has_explicit_amount = "amount" in route
        amount_tensor = (
            _as_numeric_tensor(_route_amount_value(route))
            if has_explicit_amount
            else None
        )
        if not has_explicit_amount:
            details.append(
                _detail(
                    variable_name=display_name,
                    expected="explicit residual amount",
                    value="missing",
                )
            )
        elif amount_tensor is None:
            details.append(
                _detail(
                    variable_name=display_name,
                    expected="non-empty finite real numeric residual amount",
                    value=str(_route_amount_value(route)),
                )
            )

        amount = 0.0
        if amount_tensor is not None:
            amount = _tensor_magnitude(amount_tensor)
            if not math.isfinite(amount):
                details.append(
                    _detail(
                        variable_name=display_name,
                        expected="finite derived residual magnitude",
                        value=amount,
                    )
                )
            if amount > tolerance or required:
                active_count += 1

        raw_units = route.get("units")
        residual_units = raw_units.strip() if isinstance(raw_units, str) else ""
        if dimensionality_from_unit(residual_units) is None:
            details.append(
                _detail(
                    variable_name=display_name,
                    expected=(
                        "non-empty parseable residual units in the canonical "
                        "SI-token subset"
                    ),
                    value=raw_units if raw_units is not None else "missing",
                )
            )

        raw_route_type = route.get("route_type")
        route_type = _route_type(route)
        if not isinstance(raw_route_type, str) or route_type not in _ALLOWED_ROUTES:
            details.append(
                _detail(
                    variable_name=display_name,
                    expected=(
                        "storage, boundary_outflow, managed_export, "
                        "diagnostic_residual, or rejected_step"
                    ),
                    value=route_type or "missing route_type",
                )
            )

        target_error = _validate_route_target(route, route_type)
        if target_error:
            details.append(
                _detail(
                    variable_name=display_name,
                    expected=target_error,
                    value=route.get("target", "missing"),
                )
            )
        if route_type != "rejected_step" and step_rejected_value is True:
            details.append(
                _detail(
                    variable_name=display_name,
                    expected="only rejected_step routes may set step_rejected=true",
                    value=route_type or "missing route_type",
                )
            )

        if "routed_amount" in route:
            routed_tensor = _as_numeric_tensor(route.get("routed_amount"))
            if routed_tensor is None:
                details.append(
                    _detail(
                        variable_name=display_name,
                        expected="non-empty finite real numeric routed_amount",
                        value=str(route.get("routed_amount")),
                    )
                )
            elif amount_tensor is None or routed_tensor.shape != amount_tensor.shape:
                details.append(
                    _detail(
                        variable_name=display_name,
                        expected="routed_amount shape equals residual amount shape",
                        value=str(tuple(routed_tensor.shape)),
                    )
                )
            elif not torch.allclose(
                torch.abs(routed_tensor),
                torch.abs(amount_tensor),
                rtol=0.0,
                atol=tolerance,
            ):
                details.append(
                    _detail(
                        variable_name=display_name,
                        expected=str(torch.abs(amount_tensor).tolist()),
                        value=str(torch.abs(routed_tensor).tolist()),
                        threshold=tolerance,
                    )
                )

        if len(details) == route_detail_start:
            routed_structural_count += 1
        else:
            invalid_route_count += 1

    passed = not details
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            "Overflow and residual routes are explicit."
            if passed
            else "Overflow or residual payload contains unrouteable residuals."
        ),
        passed=passed,
        layer="L2",
        scope="overflow_routing",
        tags=["iphs", "overflow", "residual", "conservation"],
        details=details,
        metrics={
            "route_count": route_count,
            "active_route_count": active_count,
            "routed_structural_count": routed_structural_count,
            "invalid_route_count": invalid_route_count,
            "no_residuals_declared": int(no_residuals),
        },
    )


@validated_rule(
    "overflow_routing",
    "L2",
    name="Overflow and residual routing",
    description="Validate explicit routes for overflow, clipping residuals, and unconsumed flux.",
    domains=["state", "orchestration", "computation"],
    tags=["iphs", "overflow", "residual", "conservation"],
    references=[
        "Hundsdorfer and Verwer (2003), Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations",
        "van der Schaft and Maschke (1995), port-Hamiltonian systems with external ports",
    ],
)
def validate_overflow_routing_structure(context: ValidationContext) -> ValidationResults:
    """Validate overflow routing payloads passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    if "overflow_routing" in metadata:
        raw_payload = metadata.get("overflow_routing")
    else:
        raw_payload = metadata
    payload = (
        raw_payload
        if isinstance(raw_payload, Mapping)
        else {"routes": [raw_payload]}
    )
    return [
        validate_overflow_routing_payload(
            payload,
            strict=metadata.get("strict") is True,
        )
    ]


__all__ = ["validate_overflow_routing_payload", "validate_overflow_routing_structure"]
