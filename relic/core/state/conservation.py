"""Conservation helpers for StateManager."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, cast

import torch
from torch import Tensor

from ..conservation_registry import build_conservation_ledger
from ..feature_flags import feature_manager
from .types import (
    ConservationSummary,
    ElementConservationReport,
    StateVariableType,
    UpdateItem,
    UpdateMode,
)

logger = logging.getLogger(__name__)

_TRACKED_CONSERVATION_ELEMENTS = (
    "carbon",
    "nitrogen",
    "phosphorus",
    "water",
)
_FLUX_LIKE_TOKENS = (
    "flux",
    "uptake",
    "emission",
    "runoff",
    "drain",
    "precip",
    "evap",
    "transpiration",
    "storage",
    "content",
    "change",
    "delta",
    "pool",
)
_DIMENSIONLESS_MARKERS = (
    "factor",
    "modifier",
    "stress",
    "response",
    "rate",
    "input",
    "parameter",
)
_FLUX_MARKERS = (
    "flux",
    "pool",
    "storage",
    "content",
    "uptake",
    "emission",
    "change",
    "delta",
)


def _build_pool_sets(manager: Any) -> Dict[str, set[str]]:
    return {
        element: {name.lower() for name in pools}
        for element, pools in manager.element_pools.items()
    }


def resolve_declared_element_domain(
    manager: Any,
    name: str,
) -> tuple[bool, Optional[str]]:
    """Return explicit membership status and its element domain, if any."""
    contract_registry = getattr(manager, "_contract_registry", None)
    if contract_registry is not None:
        contract = contract_registry.get_contract(name)
        if contract is not None:
            domain = getattr(contract, "accumulates", None)
            element = str(domain).split(".", 1)[0].lower() if domain else None
            return True, element

    spec = manager._schema.get(name)
    if spec is None:
        return False, None
    domain = getattr(spec, "accumulates", None)
    if domain is not None:
        return True, str(domain).split(".", 1)[0].lower()
    if not bool(getattr(spec, "conserved", False)):
        return True, None
    return False, None


def normalize_element_label(raw: Any) -> Optional[str]:
    """Normalize an explicit update element label to a ledger domain."""
    if raw is None:
        return None
    label = str(raw).strip().lower()
    aliases = {
        "c": "carbon",
        "carbon": "carbon",
        "co2": "carbon",
        "n": "nitrogen",
        "nitrogen": "nitrogen",
        "nh4": "nitrogen",
        "no3": "nitrogen",
        "n2": "nitrogen",
        "n2o": "nitrogen",
        "p": "phosphorus",
        "phosphorus": "phosphorus",
        "po4": "phosphorus",
        "w": "water",
        "water": "water",
        "h2o": "water",
    }
    if label not in aliases:
        raise ValueError(f"Unsupported conservation element label '{raw}'")
    return aliases[label]


def _resolve_ledger_element_domain(
    manager: Any,
    name: str,
) -> tuple[bool, Optional[str]]:
    """Return the unique explicit element assigned by the canonical ledger."""
    name_lower = name.lower()
    ledger = getattr(manager, "_conservation_ledger", {}) or {}
    matched_elements = {
        str(domain).split(".", 1)[0].lower()
        for domain, pools in ledger.items()
        if name_lower in {str(pool).lower() for pool in pools}
    }
    if len(matched_elements) > 1:
        joined = ", ".join(sorted(matched_elements))
        raise ValueError(
            f"Conservation ledger assigns '{name}' to multiple elements: {joined}"
        )
    if not matched_elements:
        return False, None
    return True, next(iter(matched_elements))


def matches_element_membership(
    manager: Any,
    name: str,
    element: str,
    pool_set: set[str],
    element_hint: Optional[str] = None,
) -> bool:
    """Resolve membership by explicit declarations before name fallback."""
    name_lower = name.lower()
    if name_lower in pool_set:
        explicit_membership = True
    else:
        explicit_membership = False
    is_declared, declared_element = resolve_declared_element_domain(manager, name)
    if is_declared:
        if element_hint is not None and element_hint != declared_element:
            raise ValueError(
                f"UpdateItem element '{element_hint}' conflicts with declared element "
                f"'{declared_element}' for '{name}'"
            )
        return declared_element == element
    is_ledger_member, ledger_element = _resolve_ledger_element_domain(manager, name)
    if is_ledger_member:
        if element_hint is not None and element_hint != ledger_element:
            raise ValueError(
                f"UpdateItem element '{element_hint}' conflicts with ledger element "
                f"'{ledger_element}' for '{name}'"
            )
        return ledger_element == element
    if element_hint is not None:
        return element_hint == element
    if explicit_membership:
        return True
    return bool(manager._matches_element_keyword(name_lower, element))


def _match_tracked_elements(
    manager: Any,
    name: str,
    pool_sets: Dict[str, set[str]],
    element_hint: Optional[str] = None,
) -> Dict[str, bool]:
    return {
        element: matches_element_membership(
            manager,
            name,
            element,
            pool_sets.get(element, set()),
            element_hint,
        )
        for element in _TRACKED_CONSERVATION_ELEMENTS
    }


def _should_consider_update(
    spec: Any,
    name_lower: str,
    element_matches: Dict[str, bool],
) -> bool:
    spec_conserved = bool(spec.conserved) if spec is not None else False
    has_flux_token = any(token in name_lower for token in _FLUX_LIKE_TOKENS)
    if not (spec_conserved or has_flux_token or any(element_matches.values())):
        return False
    if (
        (not spec_conserved)
        and spec is not None
        and spec.variable_type == StateVariableType.CONCENTRATION
    ):
        return False
    return True


def _value_to_conservation_tensor(manager: Any, value: Any) -> Optional[Tensor]:
    value_tensor = manager._value_to_sum_tensor(value)
    if value_tensor is None:
        return None
    if isinstance(value_tensor, Tensor) and value_tensor.requires_grad:
        value_tensor = value_tensor.detach()
    return value_tensor


def _is_dimensionless_non_flux_name(name_lower: str) -> bool:
    has_dimensionless_marker = any(marker in name_lower for marker in _DIMENSIONLESS_MARKERS)
    if not has_dimensionless_marker:
        return False
    return not any(marker in name_lower for marker in _FLUX_MARKERS)


def _accumulate_with_preferred_tensor_meta(
    current: Optional[Tensor],
    delta_tensor: Tensor,
    preferred_device: Optional[torch.device],
    preferred_dtype: Optional[torch.dtype],
) -> tuple[Tensor, torch.device, torch.dtype]:
    if preferred_device is None:
        preferred_device = delta_tensor.device
    if preferred_dtype is None:
        preferred_dtype = delta_tensor.dtype
    target = delta_tensor.to(device=preferred_device, dtype=preferred_dtype)
    updated = target if current is None else current + target
    return updated, preferred_device, preferred_dtype


def _resolve_conservation_tolerance(
    manager: Any,
    dt: Optional[float],
    scale_factor: Optional[float],
) -> float:
    if dt is not None and scale_factor is not None:
        return dt * scale_factor
    return manager._abs_conservation_tol


def _tensor_to_scalar(value: Optional[Tensor]) -> float:
    if value is None:
        return float(torch.tensor(0.0))
    return float(value.detach().cpu().item())


def _collect_conservation_violations(
    totals: Dict[str, Optional[Tensor]],
    has_updates: Dict[str, bool],
    tolerance: float,
) -> Dict[str, float]:
    violations: Dict[str, float] = {}
    for element in _TRACKED_CONSERVATION_ELEMENTS:
        if not has_updates[element]:
            continue
        element_total = _tensor_to_scalar(totals[element])
        if abs(element_total) > tolerance:
            violations[element] = element_total
    return violations


def _get_variable_contract(manager: Any, name: str) -> Optional[Any]:
    """Return the canonical SCM contract for a variable, when available."""
    registry = getattr(manager, "_contract_registry", None)
    if registry is None:
        return None
    return registry.get_contract(name)


def _resolve_conservation_source_unit(
    manager: Any,
    name: str,
    value: Any,
    contract: Any,
) -> str:
    """Resolve and validate the declared unit of one budget contribution."""
    item_unit = (
        str(value.unit).strip()
        if isinstance(value, UpdateItem) and value.unit
        else ""
    )
    contract_unit = (
        str(getattr(contract, "udunits", "")).strip()
        if contract is not None
        else ""
    )
    spec = manager._schema.get(name)
    schema_unit = str(getattr(spec, "units", "") or "").strip()
    declared_unit = contract_unit or schema_unit
    if item_unit and declared_unit and item_unit != declared_unit:
        raise ValueError(
            f"Conservation update unit mismatch for '{name}': "
            f"declared '{declared_unit}', got '{item_unit}'"
        )
    source_unit = item_unit or declared_unit
    if not source_unit:
        raise ValueError(
            f"Conservation unit is required for '{name}'"
        )
    return source_unit


def _normalize_domain_value(
    manager: Any,
    name: str,
    value: Any,
    value_tensor: Tensor,
    contract: Any,
    *,
    element: str,
    is_flux: bool,
) -> Tensor:
    """Normalize one elemental contribution to the canonical budget unit.

    References:
    - E3SM/ELM CNP budget diagnostics and CF-1.8 unit conventions.
    - IUPAC standard atomic weights for elemental amount-mass conversion.

    Mathematical expression:
    - R_E = sum(Delta S_E) - dt * (sum(F_E,in) - sum(F_E,out)).
    - m_E = n_E M_E for explicit elemental amount units.

    Parameter meanings:
    - Delta S_E: storage increment in canonical g E m-2.
    - F_E: boundary flow in canonical g E m-2 s-1.
    - M_E: elemental molar mass from the physical-constant SSOT.

    Units and route:
    - C, N, and P use g E m-2 and g E m-2 s-1.
    - Water uses kg m-2 and kg m-2 s-1.
    - Conversion changes representation only and creates no source or sink.
    """
    from ...data.standards.unit_converter import unit_converter

    converter = unit_converter
    unit_spec = converter.get_element_budget_units(element)
    target_unit = unit_spec["flux" if is_flux else "storage"]
    variable_name = unit_spec[
        "flux_variable" if is_flux else "amount_variable"
    ]
    source_unit = _resolve_conservation_source_unit(
        manager,
        name,
        value,
        contract,
    )
    normalized = value_tensor.detach()
    if source_unit != target_unit:
        if not converter.can_convert(
            source_unit,
            target_unit,
            variable_name,
        ):
            raise ValueError(
                f"Conservation unit for '{name}' cannot be normalized: "
                f"'{source_unit}' -> '{target_unit}'"
            )
        converted = converter.convert(
            normalized,
            variable_name=variable_name,
            source_units=source_unit,
            target_units=target_unit,
            context=None,
        )
        normalized = (
            converted
            if isinstance(converted, Tensor)
            else torch.as_tensor(converted, device=value_tensor.device)
        )
    return normalized.to(device=value_tensor.device, dtype=torch.float64)


def _validate_boundary_flux_update(
    manager: Any,
    name: str,
    value: Any,
    contract: Any,
    dt: Optional[float],
) -> tuple[str, Tensor]:
    """Validate one explicit SCM boundary port and return its rate.

    References:
    - Manzoni and Porporato (2009), Soil Biology and Biochemistry 41, 1355-1367.
    - Holton and Hakim (2012), An Introduction to Dynamic Meteorology.

    Mathematical expression:
    - Delta S = (sum(F_in) - sum(F_out)) * dt.

    Parameter meanings:
    - value: non-negative boundary-flow magnitude.
    - dt: transaction duration in seconds.
    - flux_in/flux_out: SCM writer-direction declarations.
    - rate: non-negative boundary transfer per second.
    """
    if not isinstance(value, UpdateItem):
        raise ValueError(
            f"Boundary flux '{name}' requires an explicit UpdateItem"
        )
    if value.mode != UpdateMode.REPLACE or bool(value.per_timestep):
        raise ValueError(
            f"Boundary rate flux '{name}' requires REPLACE with per_timestep=False"
        )

    expected_unit = str(getattr(contract, "udunits", "")).strip()
    actual_unit = str(value.unit or "").strip()
    if not actual_unit or actual_unit != expected_unit:
        raise ValueError(
            f"Boundary budget unit mismatch for '{name}': "
            f"expected '{expected_unit}', got '{actual_unit or 'missing'}'"
        )

    source = str(value.source_functor or "").strip()
    whitelist = manager._contract_registry.get_flux_whitelist(name)
    allowed_in = set(whitelist.get("flux_in", []) or [])
    allowed_out = set(whitelist.get("flux_out", []) or [])
    is_input = bool(source and source in allowed_in)
    is_output = bool(source and source in allowed_out)
    if is_input == is_output:
        raise ValueError(
            f"Boundary flux '{name}' writer '{source or 'missing'}' must be "
            "declared in exactly one of SCM allowed_in or allowed_out"
        )

    value_tensor = _value_to_conservation_tensor(manager, value)
    if value_tensor is None:
        raise ValueError(f"Boundary flux '{name}' has no numeric value")
    if not bool(torch.isfinite(value_tensor).all().item()):
        raise ValueError(f"Boundary flux '{name}' must be finite")
    if bool((value_tensor < 0).any().item()):
        direction = "input" if is_input else "output"
        raise ValueError(
            f"Boundary {direction} flux '{name}' must be non-negative"
        )

    if dt is None or float(dt) <= 0.0:
        raise ValueError(
            f"Positive dt is required for boundary rate flux '{name}'"
        )
    return ("in" if is_input else "out"), value_tensor


def initialize_default_conservation_ledger(manager: Any) -> None:
    """Finalize conservation ledger using parameter and contract sources."""
    canonical_ledger = build_conservation_ledger(manager._contract_registry)
    manager.set_conservation_ledger(canonical_ledger)


def check_conservation(
    manager: Any,
    updates: Dict[str, Any],
    dt: Optional[float] = None,
    scale_factor: Optional[float] = None,
) -> Dict[str, float]:
    """Check closed and explicitly ported open-system conservation budgets."""
    pool_sets = _build_pool_sets(manager)
    storage_totals: Dict[str, Optional[Tensor]] = {
        name: None for name in _TRACKED_CONSERVATION_ELEMENTS
    }
    boundary_inputs: Dict[str, Optional[Tensor]] = {
        name: None for name in _TRACKED_CONSERVATION_ELEMENTS
    }
    boundary_outputs: Dict[str, Optional[Tensor]] = {
        name: None for name in _TRACKED_CONSERVATION_ELEMENTS
    }
    has_element_updates = {name: False for name in _TRACKED_CONSERVATION_ELEMENTS}
    storage_members: Dict[str, Dict[str, Tensor]] = {
        name: {} for name in _TRACKED_CONSERVATION_ELEMENTS
    }
    input_members: Dict[str, Dict[str, Tensor]] = {
        name: {} for name in _TRACKED_CONSERVATION_ELEMENTS
    }
    output_members: Dict[str, Dict[str, Tensor]] = {
        name: {} for name in _TRACKED_CONSERVATION_ELEMENTS
    }
    preferred_device: Optional[torch.device] = None
    preferred_dtype: Optional[torch.dtype] = None

    for name, value in updates.items():
        spec = manager._schema.get(name)
        contract = _get_variable_contract(manager, name)
        name_lower = name.lower()
        element_hint = normalize_element_label(
            value.element if isinstance(value, UpdateItem) else None
        )
        element_matches = _match_tracked_elements(
            manager,
            name,
            pool_sets,
            element_hint,
        )
        if not _should_consider_update(spec, name_lower, element_matches):
            continue

        for element in _TRACKED_CONSERVATION_ELEMENTS:
            if not element_matches[element]:
                continue
            has_element_updates[element] = True
            if contract is not None and str(contract.kind).lower() == "flux":
                direction, rate = _validate_boundary_flux_update(
                    manager,
                    name,
                    value,
                    contract,
                    dt,
                )
                rate = _normalize_domain_value(
                    manager,
                    name,
                    value,
                    rate,
                    contract,
                    element=element,
                    is_flux=True,
                )
                target = boundary_inputs if direction == "in" else boundary_outputs
                accumulated, preferred_device, preferred_dtype = _accumulate_with_preferred_tensor_meta(
                    target[element],
                    rate,
                    preferred_device,
                    preferred_dtype,
                )
                target[element] = accumulated
                member_target = input_members if direction == "in" else output_members
                member_target[element][name] = rate
                continue

            value_tensor = _value_to_conservation_tensor(manager, value)
            if value_tensor is None:
                continue
            if _is_dimensionless_non_flux_name(name_lower):
                continue
            value_tensor = _normalize_domain_value(
                manager,
                name,
                value,
                value_tensor,
                contract,
                element=element,
                is_flux=False,
            )
            accumulated, preferred_device, preferred_dtype = _accumulate_with_preferred_tensor_meta(
                storage_totals[element],
                value_tensor,
                preferred_device,
                preferred_dtype,
            )
            storage_totals[element] = accumulated
            storage_members[element][name] = value_tensor

    element_totals: Dict[str, Optional[Tensor]] = {
        name: None for name in _TRACKED_CONSERVATION_ELEMENTS
    }
    for element in _TRACKED_CONSERVATION_ELEMENTS:
        storage = storage_totals[element]
        flux_in = boundary_inputs[element]
        flux_out = boundary_outputs[element]
        if storage is None and flux_in is None and flux_out is None:
            continue
        reference = next(
            value
            for value in (storage, flux_in, flux_out)
            if value is not None
        )
        assert reference is not None
        total = torch.zeros_like(reference)
        if storage is not None:
            total = total + storage.to(device=total.device, dtype=total.dtype)
        dt_value = float(dt) if dt is not None else 0.0
        if flux_in is not None:
            total = total - flux_in.to(device=total.device, dtype=total.dtype) * dt_value
        if flux_out is not None:
            total = total + flux_out.to(device=total.device, dtype=total.dtype) * dt_value

        if flux_in is not None or flux_out is not None:
            from ...validation.conservation_guardian import ConservationGuardian

            registry = getattr(manager, "_contract_registry", None)
            if registry is None:
                raise ValueError(
                    f"ContractRegistry is required for open {element} budget"
                )
            guardian = ConservationGuardian(
                contract_registry=registry,
                default_atol=_resolve_conservation_tolerance(
                    manager,
                    dt,
                    scale_factor,
                ),
                default_rtol=0.0,
            )
            domain = f"{element}.total"
            zeros = {
                name: torch.zeros_like(value)
                for name, value in storage_members[element].items()
            }
            result = guardian.check_domain(
                domain=domain,
                flux_in=input_members[element],
                flux_out=output_members[element],
                storage_before=zeros,
                storage_after=storage_members[element],
                dt=dt_value,
                tolerance={
                    "atol": _resolve_conservation_tolerance(
                        manager,
                        dt,
                        scale_factor,
                    ),
                    "rtol": 0.0,
                },
            )
            if result.passed:
                total = torch.zeros_like(total)
        element_totals[element] = total

    tolerance = _resolve_conservation_tolerance(manager, dt, scale_factor)
    return _collect_conservation_violations(element_totals, has_element_updates, tolerance)


def _build_empty_element_result(element: str, pools: List[str]) -> Dict[str, Any]:
    empty_report = ElementConservationReport(balance_error=0.0, relative_error=0.0)
    return {
        "element": element,
        "report": empty_report,
        "pools": pools,
        "warnings": [],
        "violations": 0,
    }


def _collect_element_net_change(
    manager: Any,
    updates: Dict[str, Any],
    *,
    element: str,
    pools: List[str],
    dt: Optional[float],
) -> tuple[List[str], Optional[Tensor]]:
    """Collect one elemental budget residual in its canonical unit."""
    tracked_updates: List[str] = []
    net_change: Optional[Tensor] = None
    preferred_device: Optional[torch.device] = None
    preferred_dtype: Optional[torch.dtype] = None
    pool_set = {pool.lower() for pool in pools}

    for name, raw_value in updates.items():
        val_tensor = _value_to_conservation_tensor(manager, raw_value)
        if val_tensor is None:
            continue
        spec = manager._schema.get(name)
        element_hint = normalize_element_label(
            raw_value.element if isinstance(raw_value, UpdateItem) else None
        )
        if not matches_element_membership(
            manager,
            name,
            element,
            pool_set,
            element_hint,
        ):
            continue
        if spec is not None and not spec.conserved and element != "water":
            continue

        contract = _get_variable_contract(manager, name)
        if contract is not None and str(contract.kind).lower() == "flux":
            direction, rate = _validate_boundary_flux_update(
                manager,
                name,
                raw_value,
                contract,
                dt,
            )
            normalized = _normalize_domain_value(
                manager,
                name,
                raw_value,
                rate,
                contract,
                element=element,
                is_flux=True,
            )
            signed_value = normalized * float(dt)
            if direction == "in":
                signed_value = -signed_value
        else:
            signed_value = _normalize_domain_value(
                manager,
                name,
                raw_value,
                val_tensor,
                contract,
                element=element,
                is_flux=False,
            )

        net_change, preferred_device, preferred_dtype = _accumulate_with_preferred_tensor_meta(
            net_change,
            signed_value,
            preferred_device,
            preferred_dtype,
        )
        tracked_updates.append(name)

    return tracked_updates, net_change


def _has_registered_presence(manager: Any, tracked_updates: List[str]) -> bool:
    return any(
        (manager._schema.get(name) and manager._schema.get(name).conserved)
        or (name in manager._state_variables)
        for name in tracked_updates
    )


def _collect_negative_pools(
    manager: Any,
    *,
    evaluated_pools: List[str],
    updates: Dict[str, Any],
    tolerance: float,
) -> List[str]:
    negative_pools: List[str] = []
    for pool in evaluated_pools:
        if pool not in updates:
            continue
        delta_tensor = manager._value_to_sum_tensor(updates[pool])
        if delta_tensor is None:
            continue
        current_value = manager._state_variables.get(pool)
        base_tensor = manager._value_to_sum_tensor(current_value) if current_value is not None else torch.tensor(0.0)
        if base_tensor is None:
            base_tensor = torch.tensor(0.0)
        delta_scalar = float(delta_tensor.detach().cpu().item())
        base_scalar = float(base_tensor.detach().cpu().item())
        if (base_scalar + delta_scalar) < -tolerance:
            negative_pools.append(pool)
    return negative_pools


def _compute_normalized_pool_reference_scale(
    manager: Any,
    *,
    element: str,
    pools: List[str],
) -> float:
    """Compute the relative-error scale in the canonical elemental unit."""
    reference = 0.0
    for pool in pools:
        current_value = manager._state_variables.get(pool)
        if current_value is None:
            continue
        contract = _get_variable_contract(manager, pool)
        if contract is not None and str(contract.kind).lower() == "flux":
            continue
        spec = manager._schema.get(pool)
        declared_unit = str(
            getattr(contract, "udunits", "")
            or getattr(spec, "units", "")
            or ""
        ).strip()
        if not declared_unit:
            continue
        value_tensor = _value_to_conservation_tensor(manager, current_value)
        if value_tensor is None:
            continue
        normalized = _normalize_domain_value(
            manager,
            pool,
            current_value,
            value_tensor,
            contract,
            element=element,
            is_flux=False,
        )
        reference += float(normalized.detach().abs().sum().cpu().item())
    return max(reference, 1.0)


def check_element_conservation(
    manager: Any,
    updates: Dict[str, Any],
    element: str,
    dt: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute per-element conservation diagnostics with ledger awareness."""
    pools = manager.element_pools.get(element, [])
    tracked_updates, net_change = _collect_element_net_change(
        manager,
        updates,
        element=element,
        pools=pools,
        dt=dt,
    )

    if not tracked_updates:
        return _build_empty_element_result(element, [])

    registered_present = _has_registered_presence(manager, tracked_updates)
    if not registered_present and len(tracked_updates) < 2:
        return _build_empty_element_result(element, tracked_updates.copy())

    evaluated_pools = pools or tracked_updates
    net_change_scalar = (
        float(torch.tensor(0.0)) if net_change is None else float(net_change.detach().cpu().item())
    )
    reference_scale = _compute_normalized_pool_reference_scale(
        manager,
        element=element,
        pools=evaluated_pools,
    )
    relative_error = abs(net_change_scalar) / reference_scale if reference_scale > 0 else abs(net_change_scalar)
    tolerance = manager._abs_conservation_tol
    warnings_list: List[str] = []
    violations = 0
    if abs(net_change_scalar) > tolerance:
        warnings_list.append(
            f"{element} mass balance residual {net_change_scalar:.6f} exceeds tolerance {tolerance:.2e}"
        )
        violations += 1

    negative_pools = _collect_negative_pools(
        manager,
        evaluated_pools=evaluated_pools,
        updates=updates,
        tolerance=tolerance,
    )
    if negative_pools:
        warnings_list.append(
            f"{element} pools would become negative: {', '.join(negative_pools)}"
        )
        violations += len(negative_pools)

    report = ElementConservationReport(
        balance_error=float(abs(net_change_scalar)),
        relative_error=float(relative_error),
    )
    return {
        "element": element,
        "report": report,
        "pools": evaluated_pools,
        "warnings": warnings_list,
        "violations": violations,
    }


def compute_conservation_groups(manager: Any, updates: Dict[str, Any]) -> Dict[str, float]:
    """Compute grouped update sums for conserved categories."""
    groups = {
        "carbon": 0.0,
        "nitrogen": 0.0,
        "phosphorus": 0.0,
        "water": 0.0,
    }
    for name, upd in updates.items():
        val_float = manager._value_to_float(upd)
        if val_float is None:
            continue
        val = float(val_float)
        lname = name.lower()
        for cat in groups:
            if cat in lname:
                groups[cat] += val
    return groups


def _run_conservation_guardian_checks(
    manager: Any,
    updates: Dict[str, Any],
    dt: Optional[float],
    summary: ConservationSummary,
) -> None:
    """Run non-blocking domain budget checks via ConservationGuardian."""
    if dt is None or dt <= 0:
        return

    contract_registry = getattr(manager, "_contract_registry", None)
    if contract_registry is None:
        return

    from ...validation.conservation_guardian import ConservationGuardian

    domains = list(contract_registry.list_domains())
    if not domains:
        return

    guardian = ConservationGuardian(
        contract_registry=contract_registry,
        default_atol=float(manager._abs_conservation_tol),
        default_rtol=0.0,
    )

    guardian_results: Dict[str, Any] = {}
    dt_float = float(dt)
    for domain in domains:
        members = list(contract_registry.get_domain_members(domain) or [])
        if not members:
            continue

        element = str(domain).split(".", 1)[0].lower()
        if element not in _TRACKED_CONSERVATION_ELEMENTS:
            continue

        flux_in: Dict[str, Tensor] = {}
        flux_out: Dict[str, Tensor] = {}
        storage_before: Dict[str, Tensor] = {}
        storage_after: Dict[str, Tensor] = {}

        for member in members:
            if member not in updates:
                continue
            raw_value = updates[member]
            value_tensor = _value_to_conservation_tensor(manager, raw_value)
            if value_tensor is None:
                continue
            contract = _get_variable_contract(manager, member)
            if contract is not None and str(contract.kind).lower() == "flux":
                direction, rate = _validate_boundary_flux_update(
                    manager,
                    member,
                    raw_value,
                    contract,
                    dt,
                )
                normalized_rate = _normalize_domain_value(
                    manager,
                    member,
                    raw_value,
                    rate,
                    contract,
                    element=element,
                    is_flux=True,
                )
                target = flux_in if direction == "in" else flux_out
                target[member] = normalized_rate
                continue

            normalized_delta = _normalize_domain_value(
                manager,
                member,
                raw_value,
                value_tensor,
                contract,
                element=element,
                is_flux=False,
            )
            storage_before[member] = torch.zeros_like(normalized_delta)
            storage_after[member] = normalized_delta

        if not (flux_in or flux_out or storage_after):
            continue

        result = guardian.check_domain(
            domain=domain,
            flux_in=flux_in,
            flux_out=flux_out,
            storage_before=storage_before,
            storage_after=storage_after,
            dt=dt_float,
            tolerance={"atol": float(manager._abs_conservation_tol), "rtol": 0.0},
        )
        guardian_results[domain] = {
            "passed": bool(result.passed),
            "residual": float(result.residual),
            "threshold": float(result.tolerance_used),
            "sum_flux_in": float(result.sum_flux_in),
            "sum_flux_out": float(result.sum_flux_out),
            "delta_storage": float(result.delta_storage),
        }
        if not result.passed:
            summary.total_violations += 1
            summary.warnings.append(
                f"ConservationGuardian domain '{domain}' residual "
                f"{result.residual:.6e} exceeds threshold {result.tolerance_used:.6e}"
            )

    if guardian_results:
        summary.custom_checks["conservation_guardian"] = guardian_results


def generate_conservation_report(
    manager: Any,
    updates: Optional[Dict[str, Any]] = None,
    current_time: Optional[float] = None,
    dt: Optional[float] = None,
    store_history: bool = True,
    **kwargs: Any,
) -> ConservationSummary:
    """Generate and optionally store a conservation summary for provided updates."""
    if kwargs:
        raise TypeError(f"Unsupported keyword arguments: {', '.join(kwargs.keys())}")
    if updates is None:
        updates = {}
    if current_time is None:
        current_time = time.time()

    summary = ConservationSummary(
        total_violations=0,
        tolerance_value=(
            dt * manager._abs_conservation_tol if dt is not None else manager._abs_conservation_tol
        ),
    )
    summary.timestamp = current_time
    if manager._current_diagnostics:
        summary.custom_checks.update(copy.deepcopy(manager._current_diagnostics))

    if not manager._conservation_diagnostics_enabled:
        summary.balance_error = {}
        summary.total_violations = 0
        summary.warnings = []
        if dt is not None:
            summary.dt = dt
            summary.dynamic_tolerance_used = True
            summary.tolerance_value = dt * manager._abs_conservation_tol
        else:
            summary.dt = None
            summary.dynamic_tolerance_used = False
            summary.tolerance_value = manager._abs_conservation_tol
        manager._last_summary = summary
        if store_history:
            manager._conservation_history.append(summary)
        manager._current_diagnostics.clear()
        return summary

    balance_map = {
        "carbon": ("carbon_report", "C"),
        "nitrogen": ("nitrogen_report", "N"),
        "phosphorus": ("phosphorus_report", "P"),
    }
    summary.balance_error = {}
    for element, (attr_name, balance_key) in balance_map.items():
        result = check_element_conservation(manager, updates, element, dt=dt)
        setattr(summary, attr_name, result["report"])
        summary.balance_error[balance_key] = result["report"].balance_error
        summary.total_violations += result["violations"]
        summary.warnings.extend(result["warnings"])

    water_result = check_element_conservation(manager, updates, "water", dt=dt)
    summary.balance_error["W"] = water_result["report"].balance_error
    summary.total_violations += water_result["violations"]
    summary.warnings.extend(water_result["warnings"])

    if dt is not None:
        summary.dt = dt
        summary.dynamic_tolerance_used = True
        summary.tolerance_value = dt * manager._abs_conservation_tol
    else:
        summary.dt = None
        summary.dynamic_tolerance_used = False
        summary.tolerance_value = manager._abs_conservation_tol

    _run_conservation_guardian_checks(manager, updates, dt, summary)

    manager._last_summary = summary
    if store_history:
        manager._conservation_history.append(summary)
    manager._current_diagnostics.clear()
    return summary


def check_strict_conservation(
    manager: Any,
    updates: Dict[str, Any],
    dt: Optional[float] = None,
    store_history: bool = True,
) -> bool:
    """Evaluate strict conservation if the corresponding feature flag is enabled."""
    summary = generate_conservation_report(manager, updates, current_time=None, dt=dt, store_history=store_history)
    if not feature_manager.is_enabled("strict_conservation_enabled"):
        return True
    return bool(summary.total_violations == 0)


def set_conservation_ledger(
    manager: Any,
    ledger: Any,
    pools: Optional[List[str]] = None,
) -> None:
    """Set explicit conservation ledger for element categories."""
    if isinstance(ledger, str):
        ledger_dict: Dict[str, Any] = {ledger: pools or []}
    elif isinstance(ledger, dict):
        ledger_dict = ledger
    else:
        raise ValueError("Conservation ledger must be a dictionary or (element, list) pair")

    validated: Dict[str, List[str]] = {}
    scalar_targets: Dict[str, float] = {}

    for k, v in ledger_dict.items():
        if not isinstance(k, str):
            raise ValueError("Ledger must map strings to list of strings or scalar totals")

        lower_key = k.lower()
        entries: List[str] = []

        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    entries.append(item)
                elif isinstance(item, (int, float, Tensor)):
                    scalar_val = manager._value_to_float(item)
                    if scalar_val is not None:
                        scalar_targets[lower_key] = scalar_val
                    else:
                        raise ValueError("Ledger entries must be strings or scalar totals")
                else:
                    raise ValueError("Ledger entries must be strings or scalar totals")
        elif isinstance(v, (int, float, Tensor)):
            scalar_val = manager._value_to_float(v)
            if scalar_val is None:
                raise ValueError("Ledger values must be list of strings or scalar totals")
            scalar_targets[lower_key] = scalar_val
        else:
            raise ValueError("Ledger must map strings to list of strings or scalar totals")

        validated[lower_key] = entries

    manager._conservation_targets = scalar_targets
    manager._conservation_ledger = validated
    manager._element_pools.clear()


def get_conservation_ledger(manager: Any) -> Dict[str, List[str]]:
    """Get current conservation ledger mapping (copy)."""
    return dict(manager._conservation_ledger)


def get_last_conservation_summary(manager: Any) -> Optional[ConservationSummary]:
    """Return the last stored conservation summary if any."""
    return cast(Optional[ConservationSummary], manager._last_summary)


__all__ = [
    "initialize_default_conservation_ledger",
    "check_conservation",
    "check_element_conservation",
    "compute_conservation_groups",
    "generate_conservation_report",
    "check_strict_conservation",
    "set_conservation_ledger",
    "get_conservation_ledger",
    "get_last_conservation_summary",
]
