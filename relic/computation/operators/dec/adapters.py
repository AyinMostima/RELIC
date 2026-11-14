"""Adapters between typed DEC views and existing RELIC runtime owners."""

from __future__ import annotations

from typing import Any, Optional

from relic.core.state import StateWriteMode, UpdateItem, UpdateMode

from .forms import TypedCochain


def _coerce_update_mode(mode: Optional[UpdateMode | str], *, per_timestep: bool) -> UpdateMode:
    if mode is None:
        return UpdateMode.ACCUMULATE if per_timestep else UpdateMode.FLUX
    if isinstance(mode, UpdateMode):
        return mode
    return UpdateMode(str(mode).strip().lower())


def _check_typed_cochain_writer(
    registry: Any,
    *,
    variable_name: str,
    source_functor: str,
    mode: UpdateMode,
) -> str:
    canonical = registry.resolve_alias(variable_name) if hasattr(registry, "resolve_alias") else variable_name
    if mode in {UpdateMode.ACCUMULATE, UpdateMode.FLUX}:
        can_increment = getattr(registry, "can_increment", None)
        if not callable(can_increment):
            raise TypeError("registry must provide the ContractRegistry can_increment API.")
        if not can_increment(canonical, source_functor):
            owner = registry.get_owner(canonical) if hasattr(registry, "get_owner") else None
            raise PermissionError(
                f"Functor '{source_functor}' is not authorized to increment '{canonical}'. "
                f"Owner: {owner}"
            )
    else:
        can_write = getattr(registry, "can_write", None)
        if not callable(can_write):
            raise TypeError("registry must provide the ContractRegistry can_write API.")
        if not can_write(canonical, source_functor):
            owner = registry.get_owner(canonical) if hasattr(registry, "get_owner") else None
            raise PermissionError(
                f"Functor '{source_functor}' is not authorized to write '{canonical}'. "
                f"Owner: {owner}"
            )
    return str(canonical)


def typed_cochain_delta_to_update_item(
    cochain: TypedCochain,
    *,
    source_functor: str,
    registry: Any = None,
    variable_name: Optional[str] = None,
    mode: Optional[UpdateMode | str] = None,
    write_mode: Optional[StateWriteMode | str] = None,
    per_timestep: bool = True,
    element: Optional[str] = None,
) -> tuple[str, UpdateItem]:
    """Lower a TypedCochain delta into an existing UpdateItem payload.

    References:
      - LeVeque (2002), Finite Volume Methods for Hyperbolic Problems.
      - RELIC-IPHS-DEC architecture plan, typed cochain update binding.
    Mathematical expression:
      delta x_h -> UpdateItem(value=delta x_h, mode, per_timestep), then
      StateManager applies delta x_h or delta x_h * dt for flux mode.
    Parameter meanings:
      cochain: typed discrete delta values.
      source_functor: writer declared in ContractRegistry.
      per_timestep: True if values are already integrated over dt.
      element: optional conserved ledger label such as water, carbon, or nitrogen.
    """

    if not isinstance(cochain, TypedCochain):
        raise TypeError("cochain must be a TypedCochain.")
    if not isinstance(source_functor, str) or not source_functor.strip():
        raise ValueError("source_functor must be a non-empty string.")
    if registry is None:
        raise ValueError("registry is required for typed cochain state updates.")

    target_name = variable_name or cochain.name
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("variable_name must be a non-empty string.")

    update_mode = _coerce_update_mode(mode, per_timestep=per_timestep)
    canonical_name = _check_typed_cochain_writer(
        registry,
        variable_name=target_name,
        source_functor=source_functor,
        mode=update_mode,
    )

    resolved_write_mode: Optional[StateWriteMode]
    if write_mode is None:
        resolved_write_mode = None
    elif isinstance(write_mode, StateWriteMode):
        resolved_write_mode = write_mode
    else:
        resolved_write_mode = StateWriteMode(str(write_mode).strip().lower())

    basis = "per_step" if per_timestep else "per_second"
    item = UpdateItem(
        value=cochain.values,
        mode=update_mode,
        write_mode=resolved_write_mode,
        unit=cochain.space.units,
        basis=basis,
        per_timestep=bool(per_timestep),
        element=element,
        source_functor=source_functor,
    )
    return canonical_name, item


__all__ = ["typed_cochain_delta_to_update_item"]
