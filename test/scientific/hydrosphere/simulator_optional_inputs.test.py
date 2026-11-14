import torch

from relic.core.simulator import RELICSimulator
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)
from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff
from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture


def _run_single_step(functor, state):
    sim = RELICSimulator(
        {
            "time_step": 3600.0,
            "max_time": 3600.0,
            "conservation_checks": False,
            "strict_scale_detection": False,
        }
    )
    sim.initialize_state(state)
    sim.register_functor(functor, execution_group="fast")
    sim.step(dt=3600.0)


def test_simulator_step_allows_optional_inputs():
    richards = RichardsSoilMoisture()
    assert set(richards.metadata.input_variables) == {
        "soil_moisture",
        "infiltration_flux",
        "evapotranspiration_flux",
        "capillary_rise_flux",
    }
    assert richards.optional_input_variables == {"capillary_rise_flux"}
    assert "surface_water_flux" not in richards.metadata.input_variables
    assert "soil_moisture_change" not in richards.metadata.output_variables
    assert "water_flux_profile" not in richards.metadata.output_variables

    cases = [
        (
            PenmanMonteithET(name="penman_monteith_et", config={"canopy_resistance_source": "state"}),
            {
                "air_temperature": torch.tensor(298.15),
                "relative_humidity": torch.tensor(0.6),
                "net_radiation": torch.tensor(200.0),
                "wind_speed": torch.tensor(2.0),
                "atmospheric_pressure": torch.tensor(101325.0),
                # A canopy resistance is required: g_c = 0.01 m s-1 is r_s = 100 s m-1,
                # the value of the removed default (canopy_conductance has an SCM unit).
                "canopy_conductance": torch.tensor(0.01),
            },
        ),
        (
            SurfaceRunoff(),
            {
                "precipitation_flux": torch.tensor(5e-5),
                "soil_moisture": torch.tensor(0.7),
            },
        ),
        (
            SubsurfaceRunoff(),
            {
                "soil_moisture": torch.tensor(0.5),
                "drainage_flux": torch.tensor(1e-6),
                "groundwater_storage": torch.tensor(100.0),
            },
        ),
        (
            richards,
            {
                "soil_moisture": torch.full((richards.soil_layers,), 0.35),
                "infiltration_flux": torch.tensor(0.0),
                "evapotranspiration_flux": torch.tensor(0.0),
            },
        ),
    ]

    for functor, state in cases:
        _run_single_step(functor, state)
