"""Validation rules for irreversible port-Hamiltonian structures."""

from __future__ import annotations

from typing import Any, Iterable

from relic.computation.operators.iphs_certificate import resolve_iphs_level_requirement

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)


_FORMULATION_CONTRACT = (
    "dPH: J + J^T = 0, R = R^T >= 0, dH/dt <= 0; "
    "GENERIC: L + L^T = 0, M = M^T >= 0, degeneracy and Jacobi evidence"
)


def _iter_specs(payload: Any) -> Iterable[Any]:
    if payload is None:
        return ()
    if isinstance(payload, (list, tuple)):
        return payload
    return (payload,)


def _iphs_failure(name: Any, message: str) -> ValidationResultModel:
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=message,
        passed=False,
        layer="L1",
        scope="iphs_structure",
        tags=["iphs", "structural", "passivity", "thermodynamics"],
        details=[
            ValidationDetail(
                variable_name=str(name),
                expected=_FORMULATION_CONTRACT,
                value=message,
            )
        ],
    )


def validate_iphs_spec(
    spec: Any,
    *,
    required_level: int | None = None,
) -> ValidationResultModel:
    """Validate one IPHSStructureSpec."""

    name = getattr(spec, "name", "unknown_iphs")
    if not callable(getattr(spec, "validate", None)):
        # A missing or non-IPHS payload is a declaration violation.
        return _iphs_failure(
            name,
            "iphs_structure_check requires an IPHSStructureSpec payload; "
            f"got {type(spec).__name__}.",
        )
    # Structural validator boundary: IPHSStructureSpec.validate reports
    # contract violations with TypeError/ValueError and strict initialization
    # failures with RuntimeError; each becomes a failed result. Other errors
    # are defects and propagate.
    try:
        audit = spec.validate(required_level=required_level)
    except (TypeError, ValueError, RuntimeError) as exc:
        return _iphs_failure(name, str(exc))

    min_eig_raw = audit.get("min_dissipation_eigenvalue", 0.0)
    min_eig = float(min_eig_raw) if isinstance(min_eig_raw, (float, int)) else 0.0
    metrics = {
        str(key): float(value)
        for key, value in audit.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.GENERIC_VALIDATION_OK,
        message=f"IPHS structure '{name}' satisfies structural checks.",
        passed=True,
        layer="L1",
        scope="iphs_structure",
        tags=["iphs", "structural", "passivity", "thermodynamics"],
        details=[
            ValidationDetail(
                variable_name=str(name),
                expected=_FORMULATION_CONTRACT,
                value=min_eig,
                metadata={"iphs_audit": audit},
            )
        ],
        metrics=metrics,
    )


@validated_rule(
    "iphs_structure",
    "L1",
    name="IPHS structure",
    description="Validate formulation-specific dPH and GENERIC structure.",
    domains=["computation"],
    tags=["iphs", "structural", "passivity", "thermodynamics"],
    references=[
        "van der Schaft and Maschke (1995), Port-Hamiltonian systems",
        "Ramirez et al. (2022), Irreversible port-Hamiltonian systems",
    ],
)
def validate_iphs_structure(context: ValidationContext) -> ValidationResults:
    """Validate local structure, without certifying a process runtime.

    A strict request here is specific to IPHS, unlike graph-wide content
    strictness. Runtime equivalence remains owned by the certificate rule.
    """

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    specs = metadata.get("iphs_specs") or metadata.get("iphs_spec")
    strict = metadata.get("strict", False)
    required_level = None
    failures: list[tuple[str, str, Any]] = []
    if type(strict) is not bool:
        failures.append(("strict", "strict must be bool.", strict))
    else:
        required_level, invalid_levels = resolve_iphs_level_requirement(
            {"metadata": metadata}, strict_iphs=strict,
        )
        for field_name, value in invalid_levels:
            failures.append((
                field_name,
                f"{field_name} must be an integer IPHS level from 0 through 5.",
                value,
            ))
    normalized_specs = tuple(_iter_specs(specs))
    if not failures and required_level is not None and not normalized_specs:
        failures.append((
            "iphs_specs",
            "Requested IPHS validation requires a non-empty structure payload.",
            0,
        ))
    if failures:
        return [
            ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=message,
                passed=False,
                layer="L1",
                scope="iphs_structure",
                tags=["iphs", "structural", "passivity", "thermodynamics"],
                details=[
                    ValidationDetail(
                        variable_name=field_name,
                        expected=message,
                        value=str(value),
                    )
                ],
            )
            for field_name, message, value in failures
        ]
    return [
        validate_iphs_spec(spec, required_level=required_level)
        for spec in normalized_specs
    ]


__all__ = ["validate_iphs_spec", "validate_iphs_structure"]
