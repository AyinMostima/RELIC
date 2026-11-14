import torch

from relic.scientific.anthroposphere.management.grazing import GrazingManagement
from relic.scientific.biosphere.respiration.growth_respiration import GrowthRespiration
from relic.scientific.biosphere.respiration.maintenance_respiration import (
    MaintenanceRespiration,
)


def _assert_declared_outputs(functor, outputs):
    declared = set(getattr(functor.metadata, "output_variables", []) or [])
    returned = set(outputs.keys())
    extra = sorted(returned - declared)
    assert not extra, f"{functor.__class__.__name__} returned undeclared outputs: {extra}"


def test_grazing_management_runtime_output_contract():
    functor = GrazingManagement(config={"calendar_type": "noleap"})

    scheduled_state = {
        "aboveground_biomass": torch.tensor([[2.0], [2.2]], dtype=torch.float32),
        "vegetation_quality": torch.tensor([[0.8], [0.9]], dtype=torch.float32),
        "soil_moisture": torch.tensor([[0.2], [0.25]], dtype=torch.float32),
        "temperature": torch.tensor([[293.15], [295.15]], dtype=torch.float32),
    }
    scheduled_outputs = functor.forward(
        scheduled_state,
        time=0.0,
        dt=86400.0,
        day_of_year=150,
    )
    _assert_declared_outputs(functor, scheduled_outputs)

    pressure_state = {
        "aboveground_biomass": torch.tensor([[2.0], [2.2]], dtype=torch.float32),
        "vegetation_quality": torch.tensor([[0.7], [0.8]], dtype=torch.float32),
        "soil_moisture": torch.tensor([[0.2], [0.25]], dtype=torch.float32),
        "temperature": torch.tensor([[293.15], [295.15]], dtype=torch.float32),
    }
    pressure_outputs = functor.forward(
        pressure_state,
        time=0.0,
        dt=3600.0,
        day_of_year=150,
    )
    _assert_declared_outputs(functor, pressure_outputs)
    assert set(pressure_outputs) == set(functor.metadata.output_variables)


def test_maintenance_respiration_runtime_output_contract():
    functor = MaintenanceRespiration()
    state = {
        "air_temperature": torch.tensor([298.15], dtype=torch.float32),
        "biomass_leaf": torch.tensor([200.0], dtype=torch.float32),
        "biomass_stem": torch.tensor([300.0], dtype=torch.float32),
        "biomass_root": torch.tensor([150.0], dtype=torch.float32),
        "soluble_sugar": torch.tensor([30.0], dtype=torch.float32),
        "soil_water_potential": torch.tensor([-0.5], dtype=torch.float32),
    }
    outputs = functor.forward(state=state, time=0.0, dt=3600.0)
    _assert_declared_outputs(functor, outputs)
    assert set(outputs) == set(functor.metadata.output_variables)


def test_growth_respiration_runtime_output_contract():
    functor = GrowthRespiration()
    state = {
        "gross_primary_productivity": torch.tensor([1.5e-7], dtype=torch.float32),
        "maintenance_respiration_total": torch.tensor(
            [3.6163690476190476e-8],
            dtype=torch.float32,
        ),
        "air_temperature": torch.tensor([298.15], dtype=torch.float32),
        "plant_nitrogen": torch.tensor([2.0], dtype=torch.float32),
        "plant_phosphorus": torch.tensor([0.2], dtype=torch.float32),
        "soluble_sugar": torch.tensor([10.0], dtype=torch.float32),
    }
    outputs = functor.forward(state=state, time=0.0, dt=3600.0)
    _assert_declared_outputs(functor, outputs)
    assert set(outputs) == set(functor.metadata.output_variables)
