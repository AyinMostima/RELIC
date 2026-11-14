"""Dimensional closure tests for leaf boundary-layer conductance."""

import torch

from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.utils.physical_constants import get_physical_constant


def test_molar_and_velocity_conductance_forms_close_elementwise() -> None:
    module = BiosphereAtmosphereCoupling(
        name="boundary_dimensional_closure",
        config={},
    )
    wind = torch.tensor([0.2, 0.5, 1.0, 2.5], dtype=torch.float64)
    temperature = torch.tensor([280.0, 290.0, 300.0, 310.0], dtype=torch.float64)
    pressure = torch.tensor(
        [80000.0, 90000.0, 101325.0, 105000.0], dtype=torch.float64
    )

    conductance_mol = module.forward(
        {
            "friction_velocity": wind,
            "air_temperature": temperature,
            "atmospheric_pressure": pressure,
        },
        time=0.0,
        dt=60.0,
    )["boundary_layer_conductance"].value

    gas_constant = wind.new_tensor(float(get_physical_constant("gas_constant")))
    recovered_ms = conductance_mol * gas_constant * temperature / pressure
    expected_ms = module.boundary_layer_transfer_coefficient.to(wind) * torch.sqrt(
        wind / module.leaf_characteristic_dimension.to(wind)
    )
    residual = recovered_ms - expected_ms

    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=1.0e-15)


def test_transport_coefficient_declares_no_local_mass_or_energy_budget() -> None:
    module = BiosphereAtmosphereCoupling(
        name="boundary_open_port",
        config={},
    )

    assert module.metadata.conservation_laws in ({}, [])
    assert not hasattr(module, "diagnose_conservation")
    normalized_doc = " ".join((module.forward.__doc__ or "").split())
    assert "diagnostic transport coefficient" in normalized_doc
    state = {
        "friction_velocity": torch.tensor(0.5, dtype=torch.float64),
        "air_temperature": torch.tensor(298.15, dtype=torch.float64),
        "atmospheric_pressure": torch.tensor(101325.0, dtype=torch.float64),
    }
    outputs = module.forward(state, time=0.0, dt=60.0)
    assert module.check_conservation_laws(state, outputs) == {}
