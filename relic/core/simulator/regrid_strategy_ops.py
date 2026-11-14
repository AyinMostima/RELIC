"""Regridding strategy selection for RELICSimulator.

The spatial quantity basis of a variable has one owner,
``relic.coupling.scales.resolve_spatial_basis``: it reads the explicit
``spatial_basis`` (or ``variable_type``) of the variable's metadata entry and
raises when the entry is missing, ambiguous, or inconsistent. Variable names
never select a strategy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


def determine_regrid_strategy(var_name: str, metadata: Optional[Mapping[str, Any]]) -> str:
    """Select the spatial regridding method from the declared spatial basis.

    References:
        Jones (1999) Mon. Weather Rev. 127:2204-2210 (first-order conservative
        remapping); docs\\source\\theory\\scale_coupling.md (spatial quantity basis).
    Mathematical expression:
        method(b) = nearest       for b = categorical
        method(b) = conservative  for b in {intensive, cell_total, area_density}
        where conservative remapping forms q_t[j] = sum_i(A_ij q_s[i]) / A_t[j]
        for intensive and area-density fields and sums cell totals.
    Parameter meanings:
        b = resolve_spatial_basis(var_name, metadata); A_ij source-target
        overlap area; A_t target-cell area; q the field value.
    Units: A_ij, A_t [m2]; q in the variable's own units (unchanged).
    Domain/constraints: metadata is the variable's catalog or caller entry; a
        missing entry, a bare ``extensive`` without ``spatial_basis``, or an
        inconsistent ``variable_type``/``spatial_basis`` pair raises ValueError
        (no name-based fallback). Class labels have no weighted arithmetic.
    Sign convention: not applicable (the remap preserves signs of q).
    """
    from relic.coupling.scales import SpatialQuantityBasis, resolve_spatial_basis

    basis = resolve_spatial_basis(var_name, metadata)
    if basis is SpatialQuantityBasis.CATEGORICAL:
        return "nearest"
    return "conservative"


__all__ = ["determine_regrid_strategy"]
