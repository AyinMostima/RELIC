"""Validation rules for IPHS/GENERIC entropy production."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from relic.parameters.spec import has_complete_parameter_metadata

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)


def _as_float(value: Any, field_name: str) -> float:
    """Return one finite real scalar without reducing supplied evidence."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite scalar, not a boolean.")
    try:
        scalar = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{field_name} must be a finite scalar.") from exc
    if scalar.numel() != 1 or scalar.dtype == torch.bool or scalar.is_complex():
        raise ValueError(f"{field_name} must be a finite scalar.")
    scalar = scalar.reshape(())
    if not bool(torch.isfinite(scalar).item()):
        raise ValueError(f"{field_name} must be a finite scalar.")
    return float(scalar.item())


def _invalid_result(name: str, message: str) -> ValidationResultModel:
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=message,
        passed=False,
        layer="L1",
        scope="entropy_step",
        tags=["iphs", "generic", "entropy", "thermodynamics"],
        details=[
            ValidationDetail(
                variable_name=name,
                expected="finite scalar entropy evidence and finite non-negative tolerance",
                value=message,
            )
        ],
    )


def _parameter_field(value: Any, field_name: str) -> Any:
    """Read a resolved ParameterSpec field without owning its schema."""

    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _resolve_strict_tolerance(
    payload: Mapping[str, Any],
    *,
    name: str,
    parameter_resolver: Optional[Callable[[str], Any]],
) -> tuple[float | None, ValidationResultModel | None]:
    """Resolve one bounded, non-negative tolerance from ParameterManager evidence."""

    if "tolerance" in payload or "entropy_tolerance" in payload:
        return None, _invalid_result(
            name,
            "Strict entropy validation forbids a literal tolerance override.",
        )

    raw_key = payload.get("tolerance_parameter_key")
    if not isinstance(raw_key, str) or not raw_key.strip():
        return None, _invalid_result(
            name,
            "Strict entropy validation requires a non-empty tolerance_parameter_key.",
        )
    tolerance_key = raw_key.strip()
    if not callable(parameter_resolver):
        return None, _invalid_result(
            name,
            "tolerance_parameter_key must resolve through ParameterManager to "
            "complete ParameterManager metadata and runtime value.",
        )

    # A resolver that signals an absent key with KeyError (mapping-style
    # resolvers; ParameterManager.get_parameter returns None instead) means
    # the tolerance is undeclared: a failed result naming the key. Other
    # resolver errors are defects and propagate.
    try:
        parameter_spec = parameter_resolver(tolerance_key)
    except KeyError as exc:
        return None, _invalid_result(
            name,
            f"tolerance_parameter_key '{tolerance_key}' is not declared in "
            f"ParameterManager ({exc}).",
        )
    if not has_complete_parameter_metadata(parameter_spec, require_value=True):
        return None, _invalid_result(
            name,
            "tolerance_parameter_key must resolve to complete ParameterManager "
            "metadata and runtime value.",
        )

    resolved_key = _parameter_field(parameter_spec, "key")
    if not isinstance(resolved_key, str) or resolved_key.strip() != tolerance_key:
        return None, _invalid_result(
            name,
            "The resolved parameter key must exactly match tolerance_parameter_key.",
        )

    try:
        tolerance = _as_float(_parameter_field(parameter_spec, "value"), "tolerance")
        bounds = torch.as_tensor(_parameter_field(parameter_spec, "bounds"))
        if (
            bounds.numel() != 2
            or bounds.dtype == torch.bool
            or bounds.is_complex()
            or not bool(torch.isfinite(bounds).all().item())
        ):
            raise ValueError
        lower, upper = bounds.reshape(-1)
        value = lower.new_tensor(tolerance)
        bounded = bool(
            ((lower >= 0) & (upper >= lower) & (value >= lower) & (value <= upper)).item()
        )
        if tolerance < 0.0 or not bounded:
            raise ValueError
    except (TypeError, ValueError, RuntimeError):
        return None, _invalid_result(
            name,
            "Strict entropy tolerance must be within finite non-negative bounds.",
        )
    return tolerance, None


def _iter_payloads(payload: Any) -> Iterable[Mapping[str, Any]]:
    if payload is None:
        return ()
    if isinstance(payload, Mapping):
        return (payload,)
    if isinstance(payload, (list, tuple)):
        return tuple(item for item in payload if isinstance(item, Mapping))
    return ()


def validate_entropy_step_payload(
    payload: Mapping[str, Any],
    *,
    strict: bool = False,
    parameter_resolver: Optional[Callable[[str], Any]] = None,
) -> ValidationResultModel:
    """Validate one entropy production payload.

    References:
      - Grmela and Oettinger (1997), GENERIC formalism.
      - van der Schaft and Jeltsema (2014), port-Hamiltonian systems.
    Mathematical expression:
      dS = S_after - S_before >= -tol, or sigma = entropy_production >= -tol.
      For a closed IPHS/GENERIC step, nonnegative entropy production is the
      discrete second-law audit condition.
    Parameter meanings:
      entropy_before: entropy before the candidate step.
      entropy_after: entropy after the candidate step.
      entropy_production: directly computed entropy production for the step.
      tolerance: legacy descriptive tolerance for small roundoff-level negatives.
      tolerance_parameter_key: ParameterManager key owning the strict tolerance.
      strict: caller or payload request for fail-closed ParameterManager binding.
    """

    name = str(payload.get("name", "entropy_step"))
    if "entropy_production" not in payload and not (
        "entropy_before" in payload and "entropy_after" in payload
    ):
        return ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message=(
                "Entropy step payload must provide entropy_production or "
                "entropy_before/entropy_after."
            ),
            passed=False,
            layer="L1",
            scope="entropy_step",
            tags=["iphs", "generic", "entropy", "thermodynamics"],
            details=[
                ValidationDetail(
                    variable_name=name,
                    expected="entropy_production or entropy_before/entropy_after",
                    value="missing",
                )
            ],
        )

    effective_strict = bool(strict or payload.get("strict", False))
    if effective_strict:
        tolerance, invalid = _resolve_strict_tolerance(
            payload,
            name=name,
            parameter_resolver=parameter_resolver,
        )
        if invalid is not None:
            return invalid
        assert tolerance is not None
    else:
        try:
            tolerance = _as_float(
                payload.get("tolerance", payload.get("entropy_tolerance", 0.0)),
                "tolerance",
            )
            if tolerance < 0.0:
                raise ValueError("tolerance must be a finite non-negative scalar.")
        except (TypeError, ValueError, RuntimeError) as exc:
            message = str(exc)
            if message.startswith("tolerance must be a finite scalar"):
                message = "tolerance must be a finite non-negative scalar."
            return _invalid_result(name, message)

    try:
        if "entropy_production" in payload:
            entropy_delta = _as_float(
                payload["entropy_production"],
                "entropy_production",
            )
            expected = "entropy_production >= -tolerance"
        else:
            entropy_after = _as_float(payload["entropy_after"], "entropy_after")
            entropy_before = _as_float(payload["entropy_before"], "entropy_before")
            entropy_delta = entropy_after - entropy_before
            expected = "entropy_after - entropy_before >= -tolerance"
    except (TypeError, ValueError, RuntimeError) as exc:
        return _invalid_result(name, str(exc))

    passed = entropy_delta >= -tolerance
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            f"Entropy step '{name}' satisfies nonnegative entropy production."
            if passed
            else f"Entropy step '{name}' violates nonnegative entropy production."
        ),
        passed=passed,
        layer="L1",
        scope="entropy_step",
        tags=["iphs", "generic", "entropy", "thermodynamics"],
        details=[
            ValidationDetail(
                variable_name=name,
                expected=expected,
                value=entropy_delta,
                threshold=tolerance,
            )
        ],
        metrics={"entropy_delta": entropy_delta, "tolerance": tolerance},
    )


@validated_rule(
    "entropy_step",
    "L1",
    name="Entropy production step",
    description="Validate nonnegative entropy production for IPHS/GENERIC candidate steps.",
    domains=["computation"],
    tags=["iphs", "generic", "entropy", "thermodynamics"],
    references=[
        "Grmela and Oettinger (1997), Dynamics and thermodynamics of complex fluids",
        "van der Schaft and Jeltsema (2014), Port-Hamiltonian systems theory",
    ],
)
def validate_entropy_step_structure(context: ValidationContext) -> ValidationResults:
    """Validate entropy step payloads passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    payloads = metadata.get("entropy_steps") or metadata.get("entropy_step")
    strict = bool(metadata.get("strict", metadata.get("strict_entropy", False)))
    parameter_resolver = metadata.get("parameter_resolver")
    parameter_manager = metadata.get("parameter_manager")
    if not callable(parameter_resolver) and parameter_manager is not None:
        candidate = getattr(parameter_manager, "get_parameter", None)
        if callable(candidate):
            parameter_resolver = candidate
    if strict and not callable(parameter_resolver):
        from relic.parameters import get_global_manager

        parameter_resolver = get_global_manager().get_parameter
    return [
        validate_entropy_step_payload(
            payload,
            strict=strict,
            parameter_resolver=parameter_resolver,
        )
        for payload in _iter_payloads(payloads)
    ]


__all__ = ["validate_entropy_step_payload", "validate_entropy_step_structure"]
