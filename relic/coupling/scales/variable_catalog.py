"""Single matcher for the YAML variable catalogs ``coupling.scales.*.variables``.

ScaleManager.get_variable_metadata, HierarchicalAggregator and the scale adapters read their
catalog namespaces through their own parameter access path (FunctorBase.get_parameter_value inside
Functors, the ParameterManager elsewhere) and resolve a variable name here, so the name matching
rules exist once.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

# Documented short names of catalog variables (exact CF-style names are always preferred).
# Only abbreviations that denote exactly one quantity are aliases. Ambiguous short names are
# rejected by every path: "precipitation"/"precip", "runoff", "et" and "transpiration" (amount
# or flux), "temperature"/"temp" (air, soil, leaf or surface), "humidity" (relative or
# specific) and "co2" (concentration or flux); a resistance is not a conductance.
CATALOG_ALIASES: Mapping[str, str] = {
    "lai": "leaf_area_index",
    "rh": "relative_humidity",
}


def _match(name: str, catalog: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Exact name, then case-insensitive name, then the documented alias of the name."""
    entry = catalog.get(name)
    if isinstance(entry, Mapping):
        return dict(entry)
    lowered = name.lower()
    for key, value in catalog.items():
        if isinstance(key, str) and key.lower() == lowered and isinstance(value, Mapping):
            return dict(value)
    alias = CATALOG_ALIASES.get(lowered)
    if alias is not None:
        for key, value in catalog.items():
            if isinstance(key, str) and key.lower() == alias and isinstance(value, Mapping):
                return dict(value)
    return None


def find_catalog_entry(
    name: str,
    namespaces: Iterable[str],
    read_catalog: Callable[[str], Optional[Mapping[str, Any]]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    First catalog entry of ``name`` over the ordered ``namespaces``.

    Parameters:
        name: Variable name as used by the caller
        namespaces: Catalog namespaces in priority order (each holds a ``variables`` mapping)
        read_catalog: Returns the ``variables`` mapping of one namespace, or None when the
            namespace declares no catalog

    Returns:
        (copy of the metadata entry, namespace), or None when no catalog declares the variable
    """
    for namespace in namespaces:
        catalog = read_catalog(namespace)
        if catalog is None:
            continue
        if not isinstance(catalog, Mapping):
            raise TypeError(f"{namespace}.variables must be a mapping, got {type(catalog).__name__}.")
        entry = _match(name, catalog)
        if entry is not None:
            return entry, namespace
    return None


__all__ = ["CATALOG_ALIASES", "find_catalog_entry"]
