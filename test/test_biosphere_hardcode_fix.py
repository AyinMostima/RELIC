"""Verify that biosphere parameters are resolved through ParameterManager."""

import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.respiration.maintenance_respiration import (
    MaintenanceRespiration,
)


def test_farquhar_photosynthesis_initialization() -> None:
    model = FarquharPhotosynthesis(name="test_farquhar")

    assert hasattr(model, "vcmax25")
    assert hasattr(model, "jmax25")
    assert hasattr(model, "kc25")

    output = model.forward(
        {
            "air_temperature": torch.tensor(298.15),
            "atmospheric_co2_ppm": torch.tensor(400.0),
            "ppfd": torch.tensor(1000.0),
            "leaf_area_index": torch.tensor(1.0),
            "vapor_pressure_deficit": torch.tensor(1000.0),
            "vcmax_scaling": torch.tensor(1.0),
        },
        time=0.0,
        dt=1.0,
    )
    assert "gross_photosynthesis" in output
    assert "net_photosynthesis" in output


def test_biomass_allocation_initialization() -> None:
    model = BiomassAllocation(name="test_allocation")

    assert hasattr(model, "f_leaf_base")
    assert hasattr(model, "f_stem_base")
    assert hasattr(model, "f_root_base")

    output = model.forward(
        {
            "net_primary_productivity": torch.tensor(1.0e-7),
            "air_temperature": torch.tensor(298.15),
            "leaf_area_index": torch.tensor(4.0),
            "phenology_phase": torch.tensor(2),
            "water_stress_factor": torch.tensor(0.8),
        },
        time=0.0,
        dt=86400.0,
    )
    assert set(output) == set(model.metadata.output_variables)

    rate_names = ("growth_rate_leaf", "growth_rate_stem", "growth_rate_root")
    rate_sum = torch.zeros(())
    for name in rate_names:
        item = output[name]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == "kg C m-2 s-1"
        rate_sum = rate_sum + item.value
    torch.testing.assert_close(rate_sum, torch.tensor(1.0e-7))

    pool_names = ("leaf_biomass_c", "wood_biomass_c", "root_biomass_c")
    delta_sum = torch.zeros(())
    for name in pool_names:
        item = output[name]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.ACCUMULATE
        assert item.per_timestep is True
        assert item.unit == "kg C m-2"
        delta_sum = delta_sum + item.value
    torch.testing.assert_close(delta_sum, torch.tensor(1.0e-7 * 86400.0))

    residual = output["diagnostics.biomass_allocation.carbon_balance_residual"]
    torch.testing.assert_close(residual.value, torch.zeros_like(residual.value))


def test_maintenance_respiration_initialization() -> None:
    model = MaintenanceRespiration(name="test_respiration", enable_acclimation=False)

    assert hasattr(model, "rm_leaf_25")
    assert hasattr(model, "q10_respiration")
    assert hasattr(model, "q10_leaf_factor")

    output = model.forward(
        {
            "air_temperature": torch.tensor(298.15),
            "biomass_leaf": torch.tensor(150.0),
            "biomass_stem": torch.tensor(500.0),
            "biomass_root": torch.tensor(200.0),
            "soluble_sugar": torch.tensor(20.0),
            "soil_water_potential": torch.tensor(-0.3),
        },
        time=0.0,
        dt=1.0,
    )
    assert set(output) == set(model.metadata.output_variables)
    assert isinstance(output["maintenance_respiration_total"], UpdateItem)
    assert output["maintenance_respiration_total"].unit == "kg C m-2 s-1"
    assert "diagnostics.maintenance_respiration.leaf_flux" in output
    assert "maintenance_respiration_leaf" not in output


def test_parameter_values_reasonable() -> None:
    farquhar = FarquharPhotosynthesis(name="test_farquhar")
    vcmax25 = float(farquhar.vcmax25.item())
    kc25 = float(farquhar.kc25.item())
    assert 10.0 <= vcmax25 <= 200.0
    assert 300.0 <= kc25 <= 500.0

    allocation = BiomassAllocation(name="test_allocation")
    fractions = (
        float(allocation.f_leaf_base.item()),
        float(allocation.f_stem_base.item()),
        float(allocation.f_root_base.item()),
    )
    assert 0.95 <= sum(fractions) <= 1.05
