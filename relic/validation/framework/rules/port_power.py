"""Validation rules for IPHS port power unit contracts."""

from __future__ import annotations

from typing import Any, Mapping

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


_PORT_IDENTITY_FIELDS = (
    "name",
    "effort_var",
    "flow_var",
    "power_units",
    "positive_power_convention",
    "port_kind",
    "role",
)


def _get_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _contract_identity(payload: Any) -> dict[str, str]:
    identity = {
        field_name: str(_get_value(payload, field_name, "") or "").strip()
        for field_name in _PORT_IDENTITY_FIELDS
    }
    if not identity["name"]:
        identity["name"] = str(_get_value(payload, "port_name", "port_power"))
    return identity


def _identity_complete(identity: Mapping[str, str]) -> bool:
    return all(bool(identity.get(field_name)) for field_name in _PORT_IDENTITY_FIELDS)


def _bundle_items(payload: Any) -> tuple[tuple[Any, ...], str]:
    if isinstance(payload, (list, tuple)):
        return tuple(payload), ""
    if isinstance(payload, Mapping):
        for key in ("port_power_contracts", "port_power_units", "ports"):
            if key not in payload:
                continue
            raw_items = payload.get(key)
            if raw_items is None:
                return (), _clean_unit(
                    payload.get("expected_power_units", payload.get("required_power_units"))
                )
            if isinstance(raw_items, (list, tuple)):
                items = tuple(raw_items)
            else:
                items = (raw_items,)
            expected = _clean_unit(
                payload.get("expected_power_units", payload.get("required_power_units"))
            )
            return items, expected
    return (payload,), ""


def _clean_unit(value: Any) -> str:
    return str(value or "").strip()


def _validate_single_port_power_payload(
    payload: Any,
    *,
    default_expected_units: str = "",
) -> ValidationResultModel:
    """Validate one IPHS port power unit payload.

    References:
      - van der Schaft and Maschke (1995), Port-Hamiltonian systems.
      - Ramirez et al. (2022), irreversible port-Hamiltonian systems.
    Mathematical expression:
      Power or entropy-flow pairing uses P_h = y_h.T u_h. A port contract must
      declare a parseable power_units field and, when expected_power_units is
      supplied, both supported unit strings must match after trimming.
    Parameter meanings:
      power_units: units declared by the port contract for y_h.T u_h.
      expected_power_units: units required by the connected boundary or domain.
    """

    if not isinstance(payload, Mapping) and not hasattr(payload, "power_units"):
        return ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message="Port power bundle contains an unsupported port contract object.",
            passed=False,
            layer="L1",
            scope="port_power",
            tags=["iphs", "port", "power", "units"],
            details=[
                ValidationDetail(
                    variable_name="unsupported_port_contract",
                    expected="mapping or object with a power_units attribute",
                    value=type(payload).__name__,
                    metadata={"contract_identity": {}},
                )
            ],
            metrics={
                "port_count": 1,
                "passed_port_count": 0,
                "failed_port_count": 1,
                "identity_complete_count": 0,
            },
        )
    identity = _contract_identity(payload)
    name = identity["name"]
    power_units = _clean_unit(_get_value(payload, "power_units"))
    expected_units = _clean_unit(
        _get_value(
            payload,
            "expected_power_units",
            _get_value(payload, "required_power_units", default_expected_units),
        )
    )
    identity_complete = int(_identity_complete(identity))
    result_metrics = {
        "port_count": 1,
        "passed_port_count": 0,
        "failed_port_count": 1,
        "identity_complete_count": identity_complete,
    }

    if not power_units:
        return ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message=f"Port '{name}' must declare power_units.",
            passed=False,
            layer="L1",
            scope="port_power",
            tags=["iphs", "port", "power", "units"],
            details=[
                ValidationDetail(
                    variable_name=name,
                    cf_standard_name=None,
                    units=None,
                    expected="non-empty power_units",
                    value="missing",
                    threshold=None,
                    metadata={"contract_identity": identity},
                )
            ],
            metrics=result_metrics,
        )

    unit_details: list[ValidationDetail] = []
    if dimensionality_from_unit(power_units) is None:
        unit_details.append(
            ValidationDetail(
                variable_name=f"{name}.power_units",
                cf_standard_name=None,
                units=None,
                expected="parseable power_units in the canonical SI-token subset",
                value=power_units,
                threshold=None,
                metadata={"contract_identity": identity},
            )
        )
    if expected_units and dimensionality_from_unit(expected_units) is None:
        unit_details.append(
            ValidationDetail(
                variable_name=f"{name}.expected_power_units",
                cf_standard_name=None,
                units=None,
                expected=(
                    "parseable expected_power_units in the canonical SI-token subset"
                ),
                value=expected_units,
                threshold=None,
                metadata={"contract_identity": identity},
            )
        )
    if unit_details:
        return ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Port '{name}' declares an unsupported power unit label.",
            passed=False,
            layer="L1",
            scope="port_power",
            tags=["iphs", "port", "power", "units"],
            details=unit_details,
            metrics={
                **result_metrics,
            },
        )

    passed = not expected_units or power_units == expected_units
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            f"Port '{name}' power units are valid."
            if passed
            else f"Port '{name}' power units do not match the expected units."
        ),
        passed=passed,
        layer="L1",
        scope="port_power",
        tags=["iphs", "port", "power", "units"],
        details=[
            ValidationDetail(
                variable_name=name,
                cf_standard_name=None,
                units=None,
                expected=expected_units or "declared power_units",
                value=power_units,
                threshold=None,
                metadata={"contract_identity": identity},
            )
        ],
        metrics={
            "port_count": 1,
            "passed_port_count": int(passed),
            "failed_port_count": int(not passed),
            "identity_complete_count": identity_complete,
        },
    )


def validate_port_power_payload(payload: Any) -> ValidationResultModel:
    """Validate one port or aggregate one explicit port-contract bundle."""

    items, default_expected_units = _bundle_items(payload)
    if not items:
        return ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="Port power validation requires a non-empty port contract bundle.",
            passed=False,
            layer="L1",
            scope="port_power",
            tags=["iphs", "port", "power", "units"],
            details=[
                ValidationDetail(
                    variable_name="ports",
                    expected="at least one port power contract",
                    value=0,
                )
            ],
            metrics={
                "port_count": 0,
                "passed_port_count": 0,
                "failed_port_count": 0,
                "identity_complete_count": 0,
            },
        )
    item_results = [
        _validate_single_port_power_payload(
            item,
            default_expected_units=default_expected_units,
        )
        for item in items
    ]
    passed_count = sum(int(result.passed) for result in item_results)
    failed_count = len(item_results) - passed_count
    identity_complete_count = sum(
        int(result.metrics.get("identity_complete_count", 0))
        for result in item_results
    )
    details = [detail for result in item_results for detail in result.details]
    passed = failed_count == 0
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            f"All {len(item_results)} port power contracts are valid."
            if passed
            else "; ".join(
                result.message for result in item_results if not result.passed
            )
        ),
        passed=passed,
        layer="L1",
        scope="port_power",
        tags=["iphs", "port", "power", "units"],
        details=details,
        metrics={
            "port_count": len(item_results),
            "passed_port_count": passed_count,
            "failed_port_count": failed_count,
            "identity_complete_count": identity_complete_count,
        },
    )


@validated_rule(
    "port_power_units",
    "L1",
    name="IPHS port power units",
    description="Validate declared power units for IPHS port pairings.",
    domains=["orchestration", "computation"],
    tags=["iphs", "port", "power", "units"],
    references=[
        "van der Schaft and Maschke (1995), The Hamiltonian formulation of energy conserving physical systems with external ports",
        "Ramirez et al. (2022), irreversible port-Hamiltonian systems",
    ],
)
def validate_port_power_structure(context: ValidationContext) -> ValidationResults:
    """Validate port power payloads passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    payloads = (
        metadata.get("port_power_contracts")
        or metadata.get("port_power_units")
        or metadata.get("ports")
    )
    if payloads is None:
        return []
    return [validate_port_power_payload(payloads)]


__all__ = ["validate_port_power_payload", "validate_port_power_structure"]
