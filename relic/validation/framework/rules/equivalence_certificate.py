"""Validation rules for IPHS equivalence certificates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from relic.computation.operators.iphs_certificate import (
    resolve_iphs_level_requirement,
    IPHSEquivalenceCertificate,
    load_iphs_equivalence_certificate,
)

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)

_NONCANONICAL_CERTIFICATE_KEYS = {
    "equivalence_certificate",
    "iphs_equivalence_certificate",
    "iphs_equivalence_certificates",
}


def _rule_failure(variable: str, message: str) -> ValidationResultModel:
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=message,
        passed=False,
        layer="L1",
        scope="equivalence_certificate",
        tags=["iphs", "certificate", "equivalence"],
        details=[
            ValidationDetail(
                variable_name=variable,
                expected=message,
                value="invalid",
            )
        ],
    )


def validate_equivalence_certificate_payload(
    payload: Any,
    *,
    strict: bool = True,
    required_level: int | None = None,
) -> ValidationResultModel:
    """Validate an IPHS replacement equivalence certificate.

    References:
      - van der Schaft and Maschke (1995), port-Hamiltonian systems.
      - Grmela and Ottinger (1997), GENERIC formalism.
    Mathematical expression:
      legacy_f(x, theta, t) == iphs_f(x, theta, t) with identical declared
      inputs, outputs, ParameterManager keys, and composition ports.
    Parameter meanings:
      payload: certificate mapping, path, or IPHSEquivalenceCertificate object.
      strict: when true, runtime replacement and management maps are required.
      required_level: caller-owned minimum executable IPHS evidence level.
    """

    if not isinstance(strict, bool):
        return _rule_failure("strict", "strict must be bool.")
    payload_strict = False
    if isinstance(payload, Mapping) and "strict" in payload:
        raw_payload_strict = payload.get("strict")
        if not isinstance(raw_payload_strict, bool):
            return _rule_failure(
                "strict",
                "payload strict must be bool.",
            )
        payload_strict = raw_payload_strict
    effective_strict = strict or payload_strict
    runtime_point_audit = None
    if isinstance(payload, Mapping) and "runtime_point_audit" in payload:
        runtime_point_audit = payload.get("runtime_point_audit")
        if not isinstance(runtime_point_audit, Mapping):
            return _rule_failure(
                "runtime_point_audit",
                "runtime_point_audit must be the mapping returned by "
                "IPHSStructureSpec.validate for the executed process.",
            )
        payload = {
            key: value for key, value in payload.items() if key != "runtime_point_audit"
        }
    # Certificate validator boundary: the loader reports an unreadable file
    # (OSError), a malformed payload (TypeError/ValueError), and validate()
    # reports contract violations with TypeError/ValueError; each becomes a
    # failed result. Other errors are defects and propagate.
    try:
        certificate = (
            payload
            if isinstance(payload, IPHSEquivalenceCertificate)
            else load_iphs_equivalence_certificate(payload)
        )
        audit = certificate.validate(
            strict=effective_strict,
            required_level=required_level,
            runtime_point_audit=runtime_point_audit,
        )
    except (OSError, TypeError, ValueError) as exc:
        return ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"IPHS equivalence certificate could not be loaded: {exc}",
            passed=False,
            layer="L1",
            scope="equivalence_certificate",
            tags=["iphs", "certificate", "equivalence"],
            details=[
                ValidationDetail(
                    variable_name="iphs_equivalence_certificate",
                    expected="loadable certificate mapping or path",
                    value=str(exc),
                )
            ],
        )

    issues = tuple(str(item) for item in audit.get("issues", ()) or ())
    passed = bool(audit.get("passed", False))
    contract_identity = {
        "process_id": str(audit.get("process_id", "")),
        "canonical_role": str(audit.get("canonical_role", "")),
        "formulation": str(audit.get("formulation", "")),
        "declared_iphs_level": (
            int(audit["iphs_level"])
            if isinstance(audit.get("iphs_level"), int)
            and not isinstance(audit.get("iphs_level"), bool)
            else None
        ),
        "module_name": str(audit.get("module_name", "")),
    }
    detail = ValidationDetail(
        variable_name=str(audit.get("module_name", "unknown_module")),
        expected=(
            "complete IPHS equivalence certificate"
            if issues
            else "structural certificate contract identity"
        ),
        value="; ".join(issues) if issues else contract_identity["process_id"],
        metadata={"contract_identity": contract_identity},
    )
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            f"IPHS equivalence certificate for '{audit.get('module_name')}' is valid."
            if passed
            else f"IPHS equivalence certificate for '{audit.get('module_name')}' is incomplete."
        ),
        passed=passed,
        layer="L1",
        scope="equivalence_certificate",
        tags=["iphs", "certificate", "equivalence"],
        details=[detail],
        metrics={
            "issue_count": int(audit.get("issue_count", 0)),
            "iphs_level": int(audit.get("iphs_level", -1))
            if audit.get("iphs_level") is not None
            else -1,
            "runtime_verified_level": int(audit.get("runtime_verified_level", -1))
            if audit.get("runtime_verified_level") is not None
            else -1,
            "required_runtime_level": required_level
            if isinstance(required_level, int) and not isinstance(required_level, bool)
            else -1,
            "runtime_evidence_issue_count": int(
                audit.get("runtime_evidence_issue_count", 0)
            ),
            "iphs_level_evidence_count": int(
                audit.get("iphs_level_evidence_count", 0)
            ),
            "input_count": int(audit.get("input_count", 0)),
            "output_count": int(audit.get("output_count", 0)),
            "parameter_count": int(audit.get("parameter_count", 0)),
            "composition_port_count": int(audit.get("composition_port_count", 0)),
            "reference_count": int(audit.get("reference_count", 0)),
            "validation_test_count": int(audit.get("validation_test_count", 0)),
        },
    )


@validated_rule(
    "equivalence_certificate",
    "L1",
    name="IPHS equivalence certificate",
    description="Validate IPHS replacement, parameter, composition, and interface certificates.",
    domains=["scientific", "computation", "orchestration"],
    tags=["iphs", "certificate", "equivalence"],
    references=[
        "van der Schaft and Maschke (1995), port-Hamiltonian systems",
        "Grmela and Ottinger (1997), dynamics and thermodynamics of complex fluids",
    ],
)
def validate_equivalence_certificate_structure(context: ValidationContext) -> ValidationResults:
    """Validate certificates passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    aliases = sorted(_NONCANONICAL_CERTIFICATE_KEYS.intersection(metadata))
    if aliases:
        return [
            _rule_failure(
                "equivalence_certificates",
                "Use canonical equivalence_certificates only; "
                f"noncanonical={aliases}.",
            )
        ]
    strict_value = metadata.get("strict", True)
    if not isinstance(strict_value, bool):
        return [_rule_failure("strict", "strict must be bool.")]
    strict = strict_value

    required_level, invalid_levels = resolve_iphs_level_requirement(
        {"metadata": metadata},
    )
    if invalid_levels:
        field_name = invalid_levels[0][0].rsplit(".", 1)[-1]
        return [
            _rule_failure(
                field_name,
                f"{field_name} must be an integer IPHS level from 0 through 5.",
            )
        ]

    payloads = metadata.get("equivalence_certificates")
    if payloads is None:
        certificate_items: tuple[Any, ...] = ()
    elif not isinstance(payloads, (list, tuple)):
        return [
            _rule_failure(
                "equivalence_certificates",
                "equivalence_certificates must be a list or tuple of certificate items.",
            )
        ]
    else:
        certificate_items = tuple(payloads)
    valid_item_types = (Mapping, IPHSEquivalenceCertificate, str, Path)
    if any(not isinstance(item, valid_item_types) for item in certificate_items):
        return [
            _rule_failure(
                "equivalence_certificates",
                "equivalence_certificates must be a list or tuple of certificate items.",
            )
        ]
    if (strict or required_level is not None) and not certificate_items:
        return [
            validate_equivalence_certificate_payload(
                None,
                strict=strict,
                required_level=required_level,
            )
        ]
    if not strict and not certificate_items:
        return []
    return [
        validate_equivalence_certificate_payload(
            item,
            strict=strict,
            required_level=required_level,
        )
        for item in certificate_items
    ]


__all__ = [
    "validate_equivalence_certificate_payload",
    "validate_equivalence_certificate_structure",
]
