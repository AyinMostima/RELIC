"""Pedosphere parameter-system integration regressions."""

import pytest
import torch

from relic.parameters import get_global_manager
from relic.scientific.pedosphere.decomposition.first_order import (
    FirstOrderDecomposition,
)
from relic.scientific.pedosphere.nutrient_cycling.denitrification import (
    DenitrificationProcess,
)
from relic.scientific.pedosphere.nutrient_cycling.nitrification import (
    NitrificationProcess,
)
from relic.scientific.pedosphere.transport.gas_diffusion import GasDiffusion1D


pytestmark = pytest.mark.usefixtures(
    "pedosphere_transport_constructor_overrides"
)


def test_parameter_imports() -> None:
    """Representative pedosphere parameters must resolve from the global SSOT."""
    manager = get_global_manager()
    physical = manager.get(
        "pedosphere.physical.reference_temperature",
        required=True,
    )
    numerical = manager.get(
        "pedosphere.numerical.epsilon_length",
        required=True,
    )
    assert physical is not None
    assert numerical is not None


def test_nitrification_module() -> None:
    """Nitrification must use ParameterManager-backed kinetics."""
    process = NitrificationProcess("test_nitrification")
    assert isinstance(process.vmax, (float, torch.Tensor))

    state = {
        "nh4_nitrogen": torch.tensor(0.1),
        "soil_temperature": torch.tensor(298.15),
        "soil_moisture": torch.tensor(0.3),
        "water_filled_pore_space": torch.tensor(0.5),
        "soil_ph": torch.tensor(7.0),
    }
    result = process.forward(state, time=0.0, dt=3600.0)

    assert "nitrification_flux" in result
    assert "n2o_flux" in result


def test_denitrification_module() -> None:
    """Denitrification must use ParameterManager-backed kinetics."""
    process = DenitrificationProcess("test_denitrification")
    assert isinstance(process.vmax, (float, torch.Tensor))

    state = {
        "no3_nitrogen": torch.tensor(0.05),
        "soil_temperature": torch.tensor(298.15),
        "water_filled_pore_space": torch.tensor(0.7),
        "soil_ph": torch.tensor(7.0),
        "labile_carbon": torch.tensor(0.1),  # g C m-2, required driver
    }
    result = process.forward(state, time=0.0, dt=3600.0)

    assert "denitrification_flux" in result
    assert "n2o_flux" in result
    assert "n2_production" in result


def test_decomposition_module() -> None:
    """First-order decomposition must expose canonical state and carbon ports."""
    process = FirstOrderDecomposition("test_decomposition")
    state = {
        "metabolic_litter_c": torch.tensor(0.5),
        "structural_litter_c": torch.tensor(1.0),
        "fast_som_c": torch.tensor(2.0),
        "slow_som_c": torch.tensor(5.0),
        "passive_som_c": torch.tensor(10.0),
            "soil_temperature": torch.tensor(298.15),
            "soil_moisture": torch.tensor(0.3),
            "metabolic_litter_input": torch.tensor(0.0),
            "structural_litter_input": torch.tensor(0.0),
            "lignin_n_ratio": torch.tensor(0.3),
            "clay_content": torch.tensor(0.25),
        }
    result = process.forward(state, time=0.0, dt=3600.0)

    assert "heterotrophic_respiration" in result
    assert "carbon_decomposition_rate" in result
    assert "metabolic_litter_c" in result


def test_gas_diffusion_module() -> None:
    """Gas diffusion must expose canonical surface ports without copied aliases."""
    process = GasDiffusion1D(
        n_layers=5,
        soil_depth=1.0,
        diffusivity_model="millington_quirk",
        enable_temperature_correction=True,
        enable_atmospheric_exchange=True,
    )
    state = {
        "soil_moisture": torch.tensor([0.25, 0.25, 0.30, 0.30, 0.35]),
        "porosity": torch.tensor([0.45, 0.45, 0.40, 0.40, 0.35]),
        "soil_temperature": torch.tensor([298.15, 297.0, 295.0, 293.0, 291.0]),
        "clay_content": torch.tensor([0.25, 0.25, 0.30, 0.35, 0.40]),
        "soil_co2_concentration": torch.tensor([2.0, 1.5, 1.0, 0.8, 0.5]),
        "n2o_concentration": torch.tensor([0.01, 0.008, 0.005, 0.003, 0.001]),
    }
    result = process.forward(state, time=0.0, dt=3600.0)

    assert set(result) == set(process.metadata.output_variables)
    assert "co2_surface_flux" in result
    assert "n2o_surface_flux" in result
    assert "heterotrophic_respiration" not in result
    assert "n2o_flux" not in result


def test_parameter_validation() -> None:
    """Canonical physical parameters must satisfy their declared bounds."""
    manager = get_global_manager()
    reference_temperature = manager.get(
        "pedosphere.physical.reference_temperature",
        required=True,
        bounds=(250.0, 350.0),
    )
    q10_nitrification = manager.get(
        "pedosphere.physical.q10_nitrification",
        required=True,
        bounds=(1.5, 3.0),
    )

    assert 250.0 <= reference_temperature <= 350.0
    assert 1.5 <= q10_nitrification <= 3.0
