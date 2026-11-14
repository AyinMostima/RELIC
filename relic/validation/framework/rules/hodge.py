"""Validation rules for DEC Hodge star operators."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)


def _iter_specs(payload: Any) -> Iterable[Any]:
    if payload is None:
        return ()
    if isinstance(payload, (list, tuple)):
        return payload
    return (payload,)


def _hodge_failure(name: Any, message: str) -> ValidationResultModel:
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=message,
        passed=False,
        layer="L1",
        scope="hodge_passivity_structure",
        tags=["dec", "hodge", "structural", "passivity"],
        details=[
            ValidationDetail(
                variable_name=str(name),
                expected="shape ok, symmetric passive Hodge, PSD",
                value=message,
            )
        ],
    )


def validate_hodge_spec(
    spec: Any,
    *,
    passive: bool = True,
    strict: bool = False,
    parameter_resolver: Optional[Callable[[str], Any]] = None,
) -> ValidationResultModel:
    """Validate a HodgeStarSpec passivity contract."""

    name = getattr(spec, "name", "unknown_hodge")
    if not callable(getattr(spec, "validate", None)):
        # A missing or non-Hodge payload is a declaration violation.
        return _hodge_failure(
            name,
            "hodge_psd_check requires a HodgeStarSpec payload; "
            f"got {type(spec).__name__}.",
        )
    # Structural validator boundary: HodgeStarSpec.validate reports contract
    # violations (including unresolved tolerance parameters) with
    # TypeError/ValueError, torch shape errors with RuntimeError; each becomes
    # a failed result. Other errors are defects and propagate.
    try:
        audit = spec.validate(
            passive=passive,
            strict=strict,
            parameter_resolver=parameter_resolver,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        return _hodge_failure(name, str(exc))

    if not audit["passivity_evaluated"]:
        # passive=False: only shape checks ran; passivity is not claimed and
        # no eigenvalue metric is reported.
        return ValidationResultModel(
            level=ValidationLevel.INFO,
            code=ValidationCode.GENERIC_VALIDATION_OK,
            message=(
                f"Hodge operator '{name}' satisfies its shape checks; passivity was "
                "not evaluated (passive=False)."
            ),
            passed=True,
            layer="L1",
            scope="hodge_passivity_structure",
            tags=["dec", "hodge", "structural"],
            details=[
                ValidationDetail(
                    variable_name=str(name),
                    expected="shape ok",
                    value="passivity not evaluated",
                )
            ],
        )
    min_eig = float(audit["min_eigenvalue"])
    return ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.GENERIC_VALIDATION_OK,
        message=f"Hodge operator '{name}' satisfies passive structural checks.",
        passed=True,
        layer="L1",
        scope="hodge_passivity_structure",
        tags=["dec", "hodge", "structural", "passivity"],
        details=[
            ValidationDetail(
                variable_name=str(name),
                expected="H = H^T and min eigenvalue >= -tol",
                value=min_eig,
            )
        ],
        metrics={"min_eigenvalue": min_eig},
    )


@validated_rule(
    "hodge_passivity_structure",
    "L1",
    name="Hodge passivity structure",
    description="Validate Hodge shape, symmetry, and positive semidefiniteness.",
    domains=["computation"],
    tags=["dec", "hodge", "structural", "passivity"],
    references=[
        "Hirani (2003), Discrete Exterior Calculus",
        "Desbrun et al. (2005), Discrete Exterior Calculus",
    ],
)
def validate_hodge_passivity_structure(context: ValidationContext) -> ValidationResults:
    """Validate Hodge specs passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    specs = metadata.get("hodge_specs") or metadata.get("hodge_spec")
    passive = bool(metadata.get("passive", True))
    strict = bool(metadata.get("strict", metadata.get("strict_hodge", False)))
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
        validate_hodge_spec(
            spec,
            passive=passive,
            strict=strict,
            parameter_resolver=parameter_resolver,
        )
        for spec in _iter_specs(specs)
    ]


__all__ = ["validate_hodge_passivity_structure", "validate_hodge_spec"]
