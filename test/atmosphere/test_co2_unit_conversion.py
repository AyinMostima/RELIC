"""Unit tests for CO2 mixing-ratio to mass-concentration conversion."""

import torch

from relic.scientific.atmosphere.co2_unit_conversion import (
    mixing_ratio_umol_mol_to_mass_conc_kg_m3,
)


def test_mixing_ratio_umol_mol_to_mass_conc_kg_m3_scalar():
    """Scalar inputs should match ideal-gas-law reference calculation."""
    co2_umol_mol = 400.0
    pressure_pa = 101325.0
    temperature_k = 298.15
    mw_kg_mol = 44.01e-3
    r_j_mol_k = 8.314462618

    result = mixing_ratio_umol_mol_to_mass_conc_kg_m3(
        co2_umol_mol=co2_umol_mol,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        mw_kg_mol=mw_kg_mol,
        r_j_mol_k=r_j_mol_k,
    )

    expected = (
        (co2_umol_mol * 1.0e-6)
        * pressure_pa
        / (r_j_mol_k * temperature_k)
        * mw_kg_mol
    )
    assert torch.allclose(result, torch.tensor(expected, dtype=result.dtype))


def test_mixing_ratio_umol_mol_to_mass_conc_kg_m3_tensor_batch():
    """Tensor batch inputs should preserve shape and monotonicity."""
    co2_umol_mol = torch.tensor([280.0, 400.0, 560.0], dtype=torch.float64)
    pressure_pa = torch.tensor(101325.0, dtype=torch.float64)
    temperature_k = torch.tensor(298.15, dtype=torch.float64)
    mw_kg_mol = torch.tensor(44.01e-3, dtype=torch.float64)
    r_j_mol_k = torch.tensor(8.314462618, dtype=torch.float64)

    result = mixing_ratio_umol_mol_to_mass_conc_kg_m3(
        co2_umol_mol=co2_umol_mol,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        mw_kg_mol=mw_kg_mol,
        r_j_mol_k=r_j_mol_k,
    )

    assert result.shape == co2_umol_mol.shape
    assert result.dtype == torch.float64
    assert torch.all(result[1:] > result[:-1])
