"""Validation rules for DEC cochain complexes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from relic.parameters import get_global_manager

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)

# Parameter-database key of the D_next @ D_k = 0 residual tolerance.
DEC_RESIDUAL_TOLERANCE_KEY = "validation.dec_complex.cochain_residual_tolerance"


def resolve_dec_complex_tolerance(payload: Mapping[str, Any]) -> float:
    """Return the residual tolerance for the D_next @ D_k = 0 check.

    References: Hirani (2003), Discrete Exterior Calculus, Caltech PhD thesis;
    Desbrun et al. (2005), Discrete Exterior Calculus, arXiv:math/0508341.
    Mathematical expression: max |(D_{k+1} D_k)_{ij}| <= tol.
    Parameter meanings: D_k is the k-th coboundary (incidence) matrix; tol is
    the largest admitted absolute entry of the composed operator.
    Units: D_k and tol are dimensionless (signed incidence counts).
    Domain/constraints: tol >= 0; integer incidence matrices compose to exactly
    zero, so the database value is 0.0 (exact); a payload "tol" or "dec_tol"
    entry states a different tolerance explicitly for one check.
    Sign convention: none (absolute residual).
    """
    for key in ("tol", "dec_tol"):
        if key in payload:
            return float(payload[key])
    return float(
        get_global_manager().get(DEC_RESIDUAL_TOLERANCE_KEY, units="dimensionless", required=True)
    )


def _iter_bundles(payload: Any) -> Iterable[Any]:
    if payload is None:
        return ()
    if isinstance(payload, (list, tuple)):
        return payload
    return (payload,)


def _dec_complex_failure(name: Any, message: str) -> ValidationResultModel:
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=message,
        passed=False,
        layer="L1",
        scope="dec_complex_structure",
        tags=["dec", "cochain", "structural", "topology"],
        details=[
            ValidationDetail(
                variable_name=str(name),
                expected="D_next @ D_k = 0",
                value=message,
            )
        ],
    )


def validate_dec_complex_bundle(
    bundle: Any,
    *,
    tol: float,
    strict: bool = False,
) -> ValidationResultModel:
    """Validate one DEC bundle through its validate_complex contract."""

    name = getattr(bundle, "complex_id", "unknown_complex")
    if not callable(getattr(bundle, "validate_complex", None)):
        # A missing or non-DEC payload is a declaration violation.
        return _dec_complex_failure(
            name,
            "cochain_complex_check requires a DECComplexBundle payload; "
            f"got {type(bundle).__name__}.",
        )
    # Structural validator boundary: validate_complex reports contract
    # violations with TypeError/ValueError (torch shape errors: RuntimeError);
    # each becomes a failed result. Other errors are defects and propagate.
    try:
        residuals = bundle.validate_complex(tol=tol, strict=strict)
    except (TypeError, ValueError, RuntimeError) as exc:
        return _dec_complex_failure(name, str(exc))

    max_residual = max(residuals.values()) if residuals else 0.0
    return ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.GENERIC_VALIDATION_OK,
        message=f"DEC complex '{name}' satisfies D_next @ D_k = 0.",
        passed=True,
        layer="L1",
        scope="dec_complex_structure",
        tags=["dec", "cochain", "structural", "topology"],
        details=[
            ValidationDetail(
                variable_name=str(name),
                expected="D_next @ D_k = 0",
                value=float(max_residual),
            )
        ],
        metrics={"max_residual": float(max_residual)},
    )


@validated_rule(
    "dec_complex_structure",
    "L1",
    name="DEC cochain complex structure",
    description="Validate D_next @ D_k = 0 for DEC complex bundles.",
    domains=["computation"],
    tags=["dec", "cochain", "structural", "topology"],
    references=[
        "Desbrun et al. (2005), Discrete Exterior Calculus",
        "Arnold, Falk, and Winther (2006), Finite Element Exterior Calculus",
    ],
)
def validate_dec_complex_structure(context: ValidationContext) -> ValidationResults:
    """Validate DEC complex bundles passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    bundles = metadata.get("dec_complex_bundles") or metadata.get("dec_complex_bundle")
    tol = resolve_dec_complex_tolerance(metadata)
    strict = bool(metadata.get("strict", metadata.get("strict_complex", False)))
    return [
        validate_dec_complex_bundle(bundle, tol=tol, strict=strict)
        for bundle in _iter_bundles(bundles)
    ]


__all__ = [
    "DEC_RESIDUAL_TOLERANCE_KEY",
    "resolve_dec_complex_tolerance",
    "validate_dec_complex_bundle",
    "validate_dec_complex_structure",
]
