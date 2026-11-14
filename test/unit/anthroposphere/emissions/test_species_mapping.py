# -*- coding: ascii -*-
"""Unit tests for the canonical SpeciesMapper instance API.
All comments use ASCII only for cross-platform consistency.
"""

import torch

from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper


def test_mass_molar_roundtrip():
    mapper = SpeciesMapper()
    mass_flux = torch.tensor(0.001)  # kg m-2 s-1
    species = "SO2"
    molar = mapper.mass_to_molar_flux(mass_flux, species)
    mass_back = mapper.molar_to_mass_flux(molar, species)
    assert torch.allclose(mass_flux, mass_back, rtol=1e-6, atol=1e-12)


def test_flux_to_mixing_ratio_dimensions():
    molar_flux = torch.tensor(1.0e-6)  # mol m-2 s-1
    pressure = torch.tensor(101325.0)  # Pa
    temperature = torch.tensor(300.0)  # K
    layer_height = torch.tensor(100.0)  # m
    dt = 3600.0  # s
    delta_chi = SpeciesMapper.flux_to_mixing_ratio(
        molar_flux, pressure, temperature, layer_height, dt
    )
    assert delta_chi > 0
    assert delta_chi < 1.0


def test_validate_mass_conservation():
    mapper = SpeciesMapper()
    state = {
        "sector_emission_flux": torch.arange(
            1,
            5 * 6 + 1,
            dtype=torch.float64,
        ).reshape(5, 6, 1, 1)
        * 1.0e-12,
    }
    outputs = mapper.forward(state, time=0.0, dt=60.0)
    violation = mapper.check_conservation_laws(state, outputs)["mass"]
    assert violation < 1.0e-12
