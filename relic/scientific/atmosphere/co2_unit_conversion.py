"""
CO2 unit conversion utilities for atmosphere-soil coupling.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor


ScalarLike = Union[float, int, Tensor]


def _to_tensor(value: ScalarLike, *, ref: Tensor | None = None) -> Tensor:
    """Convert scalar-like values to tensors with optional dtype and device alignment."""
    if isinstance(value, Tensor):
        return value
    if ref is None:
        return torch.as_tensor(value, dtype=torch.get_default_dtype())
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def mixing_ratio_umol_mol_to_mass_conc_kg_m3(
    co2_umol_mol: ScalarLike,
    pressure_pa: ScalarLike,
    temperature_k: ScalarLike,
    mw_kg_mol: ScalarLike,
    r_j_mol_k: ScalarLike,
) -> Tensor:
    """
    Convert atmospheric CO2 mixing ratio [umol mol-1] to mass concentration [kg m-3].

    References:
    - Ideal gas law: pV = nRT
    - CF Metadata Conventions v1.8 unit consistency guidance

    Mathematical expression:
    - chi = co2_umol_mol * 1e-6
    - c_mol_m3 = chi * pressure_pa / (r_j_mol_k * temperature_k)
    - c_mass_kg_m3 = c_mol_m3 * mw_kg_mol

    Parameter meanings:
    - co2_umol_mol: CO2 mixing ratio [umol mol-1]
    - pressure_pa: Air pressure [Pa]
    - temperature_k: Air temperature [K]
    - mw_kg_mol: CO2 molar mass [kg mol-1]
    - r_j_mol_k: Universal gas constant [J mol-1 K-1]
    """
    co2_tensor = _to_tensor(co2_umol_mol)
    pressure_tensor = _to_tensor(pressure_pa, ref=co2_tensor)
    temperature_tensor = _to_tensor(temperature_k, ref=co2_tensor)
    mw_tensor = _to_tensor(mw_kg_mol, ref=co2_tensor)
    r_tensor = _to_tensor(r_j_mol_k, ref=co2_tensor)

    chi = co2_tensor * 1.0e-6
    c_mol_m3 = chi * pressure_tensor / (r_tensor * temperature_tensor)
    return c_mol_m3 * mw_tensor


__all__ = ["mixing_ratio_umol_mol_to_mass_conc_kg_m3"]
