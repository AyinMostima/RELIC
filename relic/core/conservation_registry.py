"""Utilities for assembling conservation ledgers from parameters and metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from relic.parameters import get_global_manager as _resolve_global_manager


def _get_parameter_manager_required() -> Any:
    manager = _resolve_global_manager()
    if manager is None:
        raise RuntimeError("Parameter manager not initialized for conservation ledger")
    return manager

_PARAM_LEDGER_CACHE: Optional[Dict[str, List[str]]] = None

def merge_ledgers(*ledgers: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge multiple ledger dictionaries without duplicating entries."""
    merged: Dict[str, List[str]] = {}
    for ledger in ledgers:
        if not ledger:
            continue
        for element, pools in ledger.items():
            target = merged.setdefault(element, [])
            for pool in pools:
                if pool not in target:
                    target.append(pool)
    return merged


def load_parameter_ledger(force_reload: bool = False) -> Dict[str, List[str]]:
    """Load conservation ledger definitions from the parameter database."""
    global _PARAM_LEDGER_CACHE
    if _PARAM_LEDGER_CACHE is not None and not force_reload:
        return dict(_PARAM_LEDGER_CACHE)

    manager = _get_parameter_manager_required()
    namespace = "validation.conservation.ledger"
    namespace_params = manager.get_namespace_parameters(
        namespace=namespace,
        include_subnamespaces=False,
        return_specs=False,
    )
    if not namespace_params:
        raise KeyError("Missing conservation ledger entries in validation.conservation.ledger")

    ledger: Dict[str, List[str]] = {}
    for key, pools in namespace_params.items():
        if isinstance(pools, (list, tuple)):
            filtered = [str(item) for item in pools if isinstance(item, str)]
            if filtered:
                element = key.split(".")[-1]
                ledger[element] = filtered

    if not ledger:
        raise ValueError("Conservation ledger empty after loading")

    _PARAM_LEDGER_CACHE = dict(ledger)
    return dict(ledger)


def build_conservation_ledger(
    contract_registry: Any = None,
    *,
    force_reload: bool = False,
) -> Dict[str, List[str]]:
    """Build the canonical element ledger from explicit SSOT declarations.

    Contract ``accumulates`` domains define variable-to-element membership.
    ParameterManager ledger entries provide centrally configured extensions.
    Functor-level conservation claims are intentionally excluded because they
    describe process invariants, not the elemental composition of each output.
    """
    contract_ledger: Dict[str, List[str]] = {}
    if contract_registry is not None:
        for domain in contract_registry.list_domains():
            element = str(domain).split(".", 1)[0].lower()
            if not element:
                continue
            members = contract_registry.get_domain_members(domain)
            target = contract_ledger.setdefault(element, [])
            for member in members:
                member_name = str(member)
                if member_name and member_name not in target:
                    target.append(member_name)

    parameter_ledger = load_parameter_ledger(force_reload=force_reload)
    return merge_ledgers(contract_ledger, parameter_ledger)


__all__ = [
    "build_conservation_ledger",
    "load_parameter_ledger",
    "merge_ledgers",
]

