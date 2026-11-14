"""Validation rules for IPHS port interconnection matching."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

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


_OPEN_BOUNDARY_ROLES = {
    "boundary",
    "boundary_inflow",
    "boundary_outflow",
    "environment",
    "external",
    "managed_export",
    "open",
    "open_boundary",
}

_COCHAIN_SPACE_FIELDS = (
    "complex_id",
    "degree",
    "primality",
    "extensivity",
    "physical_domain",
    "orientation_convention",
    "boundary_role",
    "units",
)
_SI_DIMENSION_FIELDS = ("M", "L", "T", "Theta", "Q", "N", "J")
_VALID_DEGREES = {"0", "1", "2", "3"}
_VALID_PRIMALITY = {"primal", "dual"}
_VALID_EXTENSIVITY = {"intensive", "extensive", "density", "integral"}
_VALID_COCHAIN_BOUNDARY_ROLES = {"interior", "boundary", "port", "mixed"}
_VALID_PORT_KINDS = {
    "energy",
    "entropy",
    "matter",
    "heat",
    "momentum",
    "radiation",
    "management",
}
_VALID_PORT_ROLES = {"unspecified", "internal", "external"}
_VALID_POWER_CONVENTIONS = {"into_system", "out_of_system"}
_ALLOWED_OPEN_SYSTEM_ROUTES = {
    "port",
    "boundary_export",
    "managed_export",
    "diagnostic_residual",
    "observation",
}


def _get_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _lower_text(value: Any) -> str:
    return _clean_text(value).lower()


def _space_value(space: Any, key: str) -> str:
    if space is None:
        return ""
    if isinstance(space, Mapping):
        return _clean_text(space.get(key))
    return _clean_text(getattr(space, key, ""))


def _port_name(port: Any) -> str:
    return _clean_text(_get_value(port, "name", "unnamed_port")) or "unnamed_port"


def _port_convention(port: Any) -> str:
    convention = _lower_text(
        _get_value(
            port,
            "positive_power_convention",
            _get_value(port, "power_convention", _get_value(port, "direction", "")),
        )
    )
    if convention in {"in", "input", "into", "into_system"}:
        return "into_system"
    if convention in {"out", "output", "out_of_system"}:
        return "out_of_system"
    return ""


def _is_open_boundary(port: Any) -> bool:
    route = _lower_text(_get_value(port, "open_system_route", ""))
    if route in _ALLOWED_OPEN_SYSTEM_ROUTES:
        return True
    if bool(_get_value(port, "allow_unmatched", False)):
        return True
    if bool(_get_value(port, "open_boundary", False)):
        return True
    boundary_role = _lower_text(_get_value(port, "boundary_role", ""))
    return boundary_role in _OPEN_BOUNDARY_ROLES


def _match_key(port: Any) -> str:
    name = _port_name(port)
    for key in ("connection_id", "pair_id", "matching_group", "port_group"):
        value = _clean_text(_get_value(port, key, ""))
        if value:
            return value
    conjugate = _clean_text(
        _get_value(port, "power_conjugate", _get_value(port, "conjugate_port", ""))
    )
    if conjugate:
        return "::".join(sorted((name, conjugate)))
    flow_var = _clean_text(_get_value(port, "flow_var", _get_value(port, "flow_name", "")))
    if flow_var:
        return flow_var
    return name


def _flow_weight(port: Any) -> float:
    raw = _get_value(port, "flow_weight", _get_value(port, "incidence", 1.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _space_signature(port: Any, space_kind: str = "flow") -> tuple[str, ...]:
    space = _get_value(port, f"{space_kind}_space", None)
    return (
        _space_value(space, "complex_id")
        or _clean_text(_get_value(port, f"{space_kind}_complex_id", "")),
        _space_value(space, "degree")
        or _clean_text(_get_value(port, f"{space_kind}_form_degree", "")),
        _space_value(space, "primality")
        or _clean_text(_get_value(port, f"{space_kind}_primality", "")),
        _space_value(space, "extensivity")
        or _clean_text(_get_value(port, f"{space_kind}_extensivity", "")),
        _space_value(space, "physical_domain")
        or _clean_text(_get_value(port, "physical_domain", "")),
        _space_value(space, "orientation_convention")
        or _clean_text(
            _get_value(port, f"{space_kind}_orientation_convention", "")
        ),
        _space_value(space, "boundary_role")
        or _clean_text(_get_value(port, f"{space_kind}_boundary_role", "")),
        _space_value(space, "units")
        or _clean_text(_get_value(port, f"{space_kind}_units", "")),
    )


def _space_n_dofs_raw(port: Any, space_kind: str) -> Any:
    space = _get_value(port, f"{space_kind}_space", None)
    if space is not None:
        value = _get_value(space, "n_dofs", None)
        if value is not None:
            return value
    return _get_value(port, f"{space_kind}_n_dofs", None)


def _positive_integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        scalar = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    if scalar.numel() != 1 or not bool(torch.isfinite(scalar).item()):
        return None
    scalar = scalar.reshape(())
    nearest = torch.round(scalar)
    if bool((scalar <= 0).item()) or not bool(
        torch.isclose(scalar, nearest, rtol=0.0, atol=0.0).item()
    ):
        return None
    return int(nearest.item())


def _is_finite_scalar(value: float) -> bool:
    return bool(torch.isfinite(torch.as_tensor(value, dtype=torch.float64)).item())


def _power_dimensions_match(effort_units: str, flow_units: str, power_units: str) -> bool | None:
    effort_dim = dimensionality_from_unit(effort_units)
    flow_dim = dimensionality_from_unit(flow_units)
    power_dim = dimensionality_from_unit(power_units)
    if effort_dim is None or flow_dim is None or power_dim is None:
        return None
    effort_vector = torch.tensor(
        [getattr(effort_dim, field) for field in _SI_DIMENSION_FIELDS],
        dtype=torch.float64,
    )
    flow_vector = torch.tensor(
        [getattr(flow_dim, field) for field in _SI_DIMENSION_FIELDS],
        dtype=torch.float64,
    )
    power_vector = torch.tensor(
        [getattr(power_dim, field) for field in _SI_DIMENSION_FIELDS],
        dtype=torch.float64,
    )
    return bool(
        torch.allclose(
            effort_vector + flow_vector,
            power_vector,
            rtol=0.0,
            atol=1.0e-12,
        )
    )


def _compact_nonempty(values: Iterable[str]) -> set[str]:
    return {value for value in values if value}


def _contract_identity(port: Any) -> dict[str, Any]:
    return {
        "name": _port_name(port),
        "matching_group": _match_key(port),
        "effort_var": _clean_text(_get_value(port, "effort_var", "")),
        "flow_var": _clean_text(_get_value(port, "flow_var", "")),
        "effort_space_signature": list(_space_signature(port, "effort")),
        "flow_space_signature": list(_space_signature(port, "flow")),
        "power_units": _clean_text(_get_value(port, "power_units", "")),
        "positive_power_convention": _port_convention(port),
        "port_kind": _lower_text(_get_value(port, "port_kind", "")),
        "role": _lower_text(_get_value(port, "role", "")),
        "open_system_route": _lower_text(
            _get_value(port, "open_system_route", "")
        ),
    }


def _identity_complete(identity: Mapping[str, Any]) -> bool:
    return bool(
        identity.get("name")
        and identity.get("matching_group")
        and identity.get("power_units")
        and identity.get("positive_power_convention")
        and identity.get("port_kind")
        and all(identity.get("effort_space_signature", ()))
        and all(identity.get("flow_space_signature", ()))
    )


def _detail(
    *,
    variable_name: str,
    expected: Any,
    value: Any,
    metadata: Mapping[str, Any] | None = None,
) -> ValidationDetail:
    """Build a validation detail without duplicating optional model fields."""
    return ValidationDetail.model_validate(
        {
            "variable_name": variable_name,
            "expected": expected,
            "value": value,
            "metadata": dict(metadata or {}),
        }
    )


def validate_port_matching_payload(payload: Mapping[str, Any]) -> ValidationResultModel:
    """Validate IPHS port matching for closed and open interconnections.

    References:
      - van der Schaft and Maschke (1995), port-Hamiltonian interconnection.
      - van der Schaft (2006), Port-Hamiltonian systems: an introductory survey.
    Mathematical expression:
      Each port is a power-conjugate pair (f, e) in V x V*, with
      P = <e, f>, dim(V) = dim(V*), and [P] = [e][f].
      Internal interconnections must satisfy sum_i s_i w_i = 0 for every
      matching group, where s_i = +1 for into_system and -1 for out_of_system.
      This declaration-level incidence condition is necessary for, but does
      not evaluate, the internal power balance sum_i y_i.T u_i = 0; no effort
      or flow value is read.
    Parameter meanings:
      effort_space/flow_space: typed dual cochain spaces carrying equal n_dofs.
      power_units: physical unit of the effort-flow duality pairing.
      positive_power_convention: sign convention for port power.
      flow_weight: optional incidence weight for split or merged ports.
      boundary_role/open_boundary: explicit declaration for unmatched ports.
    """

    ports_payload = payload.get("ports")
    if ports_payload is None:
        specs = payload.get("iphs_specs") or payload.get("iphs_spec")
        if specs is not None:
            if not isinstance(specs, (list, tuple)):
                specs = (specs,)
            ports_payload = []
            for spec in specs:
                ports_payload.extend(tuple(getattr(spec, "ports", ()) or ()))
    if ports_payload is None:
        ports_payload = ()
    if not isinstance(ports_payload, (list, tuple)):
        ports_payload = (ports_payload,)

    ports = tuple(ports_payload)
    details: list[ValidationDetail] = []
    identity_details: list[ValidationDetail] = []
    identity_complete_count = 0
    groups: dict[str, list[Any]] = defaultdict(list)
    missing_convention = 0
    open_boundary_count = 0
    strict = bool(payload.get("strict", False))
    closed_system = bool(payload.get("closed_system", False))
    try:
        tolerance = float(payload.get("tolerance", 1.0e-12))
    except (TypeError, ValueError):
        tolerance = float("nan")
    if not _is_finite_scalar(tolerance) or tolerance < 0.0:
        details.append(
            _detail(
                variable_name="tolerance",
                expected="finite non-negative tolerance",
                value=payload.get("tolerance"),
            )
        )
        tolerance = 0.0
    if strict and not ports and not closed_system:
        details.append(
            _detail(
                variable_name="ports",
                expected="non-empty ports or explicit closed_system=true",
                value="empty undeclared port set",
            )
        )

    for port in ports:
        identity = _contract_identity(port)
        identity_complete_count += int(_identity_complete(identity))
        identity_details.append(
            _detail(
                variable_name=_port_name(port),
                expected="validated port contract identity",
                value=_port_name(port),
                metadata={"contract_identity": identity},
            )
        )
        if strict:
            signatures = {
                kind: _space_signature(port, kind) for kind in ("effort", "flow")
            }
            for space_kind, signature in signatures.items():
                missing_fields = [
                    field_name
                    for field_name, value in zip(_COCHAIN_SPACE_FIELDS, signature)
                    if not value
                ]
                if _space_n_dofs_raw(port, space_kind) is None:
                    missing_fields.append("n_dofs")
                if missing_fields:
                    details.append(
                        _detail(
                            variable_name=_port_name(port),
                            expected=f"complete typed {space_kind} cochain metadata",
                            value=f"missing {missing_fields}",
                        )
                    )
                signature_by_field = dict(
                    zip(_COCHAIN_SPACE_FIELDS, signature, strict=True)
                )
                if signature_by_field["degree"] not in _VALID_DEGREES:
                    details.append(
                        _detail(
                            variable_name=_port_name(port),
                            expected="degree is one of 0, 1, 2, or 3",
                            value=signature_by_field["degree"],
                        )
                    )
                if signature_by_field["primality"] not in _VALID_PRIMALITY:
                    details.append(
                        _detail(
                            variable_name=_port_name(port),
                            expected="primality is primal or dual",
                            value=signature_by_field["primality"],
                        )
                    )
                if signature_by_field["extensivity"] not in _VALID_EXTENSIVITY:
                    details.append(
                        _detail(
                            variable_name=_port_name(port),
                            expected="recognized cochain extensivity",
                            value=signature_by_field["extensivity"],
                        )
                    )
                if (
                    signature_by_field["boundary_role"]
                    not in _VALID_COCHAIN_BOUNDARY_ROLES
                ):
                    details.append(
                        _detail(
                            variable_name=_port_name(port),
                            expected="recognized cochain boundary_role",
                            value=signature_by_field["boundary_role"],
                        )
                    )

            effort_units = signatures["effort"][-1]
            flow_units = signatures["flow"][-1]
            if effort_units and dimensionality_from_unit(effort_units) is None:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected=(
                            "parseable effort units in the canonical SI-token subset"
                        ),
                        value=effort_units,
                    )
                )
            if flow_units and dimensionality_from_unit(flow_units) is None:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected=(
                            "parseable flow units in the canonical SI-token subset"
                        ),
                        value=flow_units,
                    )
                )
            power_units = _clean_text(_get_value(port, "power_units", ""))
            if not power_units or dimensionality_from_unit(power_units) is None:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected=(
                            "non-empty parseable power_units in the canonical "
                            "SI-token subset"
                        ),
                        value=power_units or "missing",
                    )
                )
            effort_n_dofs = _positive_integer(
                _space_n_dofs_raw(port, "effort")
            )
            flow_n_dofs = _positive_integer(_space_n_dofs_raw(port, "flow"))
            if (
                effort_n_dofs is None
                or flow_n_dofs is None
                or effort_n_dofs != flow_n_dofs
            ):
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected=(
                            "equal positive effort_space and flow_space n_dofs"
                        ),
                        value=(
                            f"effort={_space_n_dofs_raw(port, 'effort')},"
                            f"flow={_space_n_dofs_raw(port, 'flow')}"
                        ),
                    )
                )
            power_match = _power_dimensions_match(
                effort_units,
                flow_units,
                power_units,
            )
            if power_match is False:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected=(
                            "power_units dimensionality equals effort units times "
                            "flow units"
                        ),
                        value=(
                            f"effort={effort_units},flow={flow_units},"
                            f"power={power_units}"
                        ),
                    )
                )
            raw_weight = _get_value(port, "flow_weight", _get_value(port, "incidence", 1.0))
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                weight = float("nan")
            if not _is_finite_scalar(weight):
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected="finite flow_weight",
                        value=raw_weight,
                    )
                )
            raw_convention = _lower_text(
                _get_value(port, "positive_power_convention", "")
            )
            if raw_convention not in _VALID_POWER_CONVENTIONS:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected="canonical into_system or out_of_system",
                        value=raw_convention or "missing",
                    )
                )
            port_kind = _lower_text(_get_value(port, "port_kind", ""))
            if port_kind not in _VALID_PORT_KINDS:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected="recognized port_kind",
                        value=port_kind or "missing",
                    )
                )
            role = _lower_text(_get_value(port, "role", ""))
            if role and role not in _VALID_PORT_ROLES:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected="recognized port role",
                        value=role,
                    )
                )
            route = _lower_text(_get_value(port, "open_system_route", ""))
            if route and route not in _ALLOWED_OPEN_SYSTEM_ROUTES:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected="allowed open_system_route",
                        value=route,
                    )
                )
            if role == "external" and route not in _ALLOWED_OPEN_SYSTEM_ROUTES:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected=(
                            "explicit allowed open_system_route for external role"
                        ),
                        value=route or "missing",
                    )
                )
            if role == "internal" and route:
                details.append(
                    _detail(
                        variable_name=_port_name(port),
                        expected="internal role without open_system_route",
                        value=route,
                    )
                )
        convention = _port_convention(port)
        if not convention:
            missing_convention += 1
            details.append(
                _detail(
                    variable_name=_port_name(port),
                    expected="into_system or out_of_system",
                    value="missing positive_power_convention",
                )
            )
            continue
        if _is_open_boundary(port):
            open_boundary_count += 1
        groups[_match_key(port)].append(port)

    for group_key, group_ports in sorted(groups.items()):
        closed_ports = [port for port in group_ports if not _is_open_boundary(port)]
        if not closed_ports:
            continue

        conventions = [_port_convention(port) for port in closed_ports]
        if "into_system" not in conventions or "out_of_system" not in conventions:
            details.append(
                _detail(
                    variable_name=group_key,
                    expected="at least one into_system and one out_of_system port or explicit open boundary",
                    value=",".join(_port_name(port) for port in closed_ports),
                )
            )

        signed_balance = 0.0
        for port in closed_ports:
            sign = 1.0 if _port_convention(port) == "into_system" else -1.0
            signed_balance += sign * _flow_weight(port)
        if abs(signed_balance) > tolerance:
            details.append(
                _detail(
                    variable_name=group_key,
                    expected="signed port incidence balance equals 0",
                    value=signed_balance,
                )
            )

        power_units = _compact_nonempty(
            _clean_text(_get_value(port, "power_units", "")) for port in closed_ports
        )
        if len(power_units) > 1:
            details.append(
                _detail(
                    variable_name=group_key,
                    expected="one power_units value per matched group",
                    value=",".join(sorted(power_units)),
                )
            )

        port_kinds = _compact_nonempty(
            _lower_text(_get_value(port, "port_kind", "")) for port in closed_ports
        )
        if len(port_kinds) > 1:
            details.append(
                _detail(
                    variable_name=group_key,
                    expected="one port_kind per matched group",
                    value=",".join(sorted(port_kinds)),
                )
            )

        elements = _compact_nonempty(
            _lower_text(_get_value(port, "element", "")) for port in closed_ports
        )
        if len(elements) > 1:
            details.append(
                _detail(
                    variable_name=group_key,
                    expected="one conserved element per matched group",
                    value=",".join(sorted(elements)),
                )
            )

        signatures = _compact_nonempty(
            "|".join(_space_signature(port, "flow")) for port in closed_ports
        )
        if len(signatures) > 1:
            details.append(
                _detail(
                    variable_name=group_key,
                    expected="compatible flow cochain space metadata",
                    value=";".join(sorted(signatures)),
                )
            )
        if strict:
            effort_signatures = _compact_nonempty(
                "|".join(_space_signature(port, "effort"))
                for port in closed_ports
            )
            if len(effort_signatures) > 1:
                details.append(
                    _detail(
                        variable_name=group_key,
                        expected="compatible effort cochain space metadata",
                        value=";".join(sorted(effort_signatures)),
                    )
                )

            port_dofs = {
                n_dofs
                for port in closed_ports
                if (
                    n_dofs := _positive_integer(
                        _space_n_dofs_raw(port, "flow")
                    )
                )
                is not None
            }
            if len(port_dofs) > 1:
                details.append(
                    _detail(
                        variable_name=group_key,
                        expected="one port n_dofs value per matched group",
                        value=",".join(str(value) for value in sorted(port_dofs)),
                    )
                )

    passed = not details
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            "IPHS port matching is closed or explicitly open."
            if passed
            else "IPHS port matching has unmatched or incompatible ports."
        ),
        passed=passed,
        layer="L1",
        scope="port_matching",
        tags=["iphs", "port", "interconnection", "power"],
        details=[*details, *identity_details],
        metrics={
            "port_count": len(ports),
            "group_count": len(groups),
            "open_boundary_count": open_boundary_count,
            "missing_convention_count": missing_convention,
            "identity_complete_count": identity_complete_count,
        },
    )


@validated_rule(
    "port_matching",
    "L1",
    name="IPHS port matching",
    description="Validate closed IPHS port pairs and explicit open boundaries.",
    domains=["orchestration", "computation"],
    tags=["iphs", "port", "interconnection", "power"],
    references=[
        "van der Schaft and Maschke (1995), The Hamiltonian formulation of energy conserving physical systems with external ports",
        "Ramirez et al. (2022), irreversible port-Hamiltonian systems",
    ],
)
def validate_port_matching_structure(context: ValidationContext) -> ValidationResults:
    """Validate port matching payloads passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    payload = metadata.get("port_matching") or metadata
    return [validate_port_matching_payload(payload)]


__all__ = ["validate_port_matching_payload", "validate_port_matching_structure"]
