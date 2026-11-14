"""Dimension consistency checks for coupling port metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .dimensionality import dimensionality_from_unit

if TYPE_CHECKING:
    from ...orchestration.port_spec import PortSpec
    from ...orchestration.ssot_catalog import SSOTCatalog


def check_port_dimension_consistency(ports: List[PortSpec], catalog: SSOTCatalog) -> List[str]:
    """
    Validate dimensionality compatibility between port units and SSOT units.

    Returns warning messages. Empty list means no inconsistencies detected.
    """
    warnings: List[str] = []
    for port in ports:
        if not port.units:
            continue

        port_dim = dimensionality_from_unit(port.units)
        if port_dim is None:
            warnings.append(
                f"Port '{port.name}' has unsupported units '{port.units}' for dimensionality parsing."
            )
            continue

        entry = catalog.get(port.name) or catalog.get_by_standard_name(port.standard_name)
        if entry is None:
            continue

        entry_dim = entry.dimensionality
        if entry_dim is None and entry.units:
            entry_dim = dimensionality_from_unit(entry.units)
        if entry_dim is None:
            continue

        if not port_dim.is_compatible(entry_dim):
            warnings.append(
                f"Port '{port.name}' units '{port.units}' are dimensionally incompatible "
                f"with SSOT units '{entry.units}' for '{entry.standard_name}'."
            )

    return warnings


__all__ = ["check_port_dimension_consistency"]
