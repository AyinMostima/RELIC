"""
L1 Validation Smoke Tests for Hydrosphere Modules

This file is intentionally a lightweight, contract-focused suite:
- Verifies modules can be constructed and run with canonical SI inputs
- Verifies primary output keys exist and tensors are finite

Detailed scientific validation is covered by module-scoped unit tests under:
  test\\unit\\hydrosphere\\...
  test\\hydrosphere\\...

References:
- CF-1.8 conventions (units, naming)
"""

import torch

from relic.core.state import UpdateItem

from relic.core.functor import FunctorBase
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import PriestleyTaylorET
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff
from relic.scientific.hydrosphere.snow.albedo import SnowAlbedo
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.scientific.hydrosphere.transport.vertical_percolation import VerticalPercolationTransport


def _assert_tensor_finite_non_negative(t: torch.Tensor) -> None:
    assert isinstance(t, torch.Tensor)
    assert torch.all(torch.isfinite(t))
    assert torch.all(t >= 0.0)


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def test_functors_inherit_functor_base() -> None:
    functors = [
        PenmanMonteithET(config={"canopy_resistance_source": "state"}),
        PriestleyTaylorET(),
        BucketSoilMoisture(),
        RichardsSoilMoisture(),
        SurfaceRunoff(),
        VerticalPercolationTransport(),
        SnowAlbedo(),
        GroundwaterDynamics(),
    ]
    for functor in functors:
        assert isinstance(functor, FunctorBase)
        assert hasattr(functor, "forward")


def test_penman_monteith_smoke() -> None:
    et = PenmanMonteithET(config={"canopy_resistance_source": "state"})
    state = {
        "air_temperature": torch.tensor(298.15),
        "relative_humidity": torch.tensor(0.65),
        "wind_speed": torch.tensor(2.0),
        "net_radiation": torch.tensor(400.0),
        "atmospheric_pressure": torch.tensor(101325.0),
        # canopy_resistance_source "state": r_s = 1 / g_c from canopy_conductance [m s-1]
        "canopy_conductance": torch.tensor(0.01),
        "leaf_area_index": torch.tensor(3.0),
    }
    result = et.forward(state, time=43200.0, dt=3600.0)
    _assert_tensor_finite_non_negative(_unwrap_update(result["evapotranspiration_flux"]))


def test_priestley_taylor_smoke() -> None:
    et = PriestleyTaylorET()
    state = {
        "air_temperature": torch.tensor(298.15),
        "net_radiation": torch.tensor(400.0),
        "atmospheric_pressure": torch.tensor(101325.0),
    }
    result = et.forward(state, time=43200.0, dt=3600.0)
    _assert_tensor_finite_non_negative(_unwrap_update(result["evapotranspiration_flux"]))


def test_bucket_soil_moisture_smoke() -> None:
    sm = BucketSoilMoisture()
    state = {
        "soil_moisture": torch.tensor(0.25),
        "precipitation_flux": torch.tensor(5.0 / 3600.0),  # 5 mm/h -> kg m-2 s-1
        "evapotranspiration_flux": torch.tensor(2.0 / 3600.0),
    }
    result = sm.forward(state, time=3600.0, dt=3600.0)
    diagnostics = result["diagnostics.soil_moisture_bucket"]
    assert "soil_moisture_change" not in result
    soil_change = diagnostics["soil_moisture_change_rate"]
    assert torch.all(torch.isfinite(soil_change))


def test_surface_runoff_smoke() -> None:
    runoff = SurfaceRunoff()
    state = {
        "precipitation_flux": torch.tensor(20.0 / 3600.0),  # 20 mm/h -> kg m-2 s-1
        "soil_moisture": torch.tensor(0.45),
    }
    result = runoff.forward(state, time=3600.0, dt=3600.0)
    surface_flux = _unwrap_update(result["surface_runoff_flux"])
    _assert_tensor_finite_non_negative(surface_flux)


def test_vertical_percolation_smoke() -> None:
    transport = VerticalPercolationTransport()
    state = {
        "soil_moisture": torch.tensor(0.30),
        "soil_temperature": torch.tensor(293.15),
    }
    result = transport.forward(state, time=3600.0, dt=3600.0)
    assert "drainage_flux" in result
    assert "soil_moisture" not in result
    drainage_flux = _unwrap_update(result["drainage_flux"])
    assert torch.all(torch.isfinite(drainage_flux))


def test_snow_albedo_smoke() -> None:
    snow = SnowAlbedo()
    # Every input is explicit: SnowAlbedo substitutes no state default.
    state = {
        "snowfall": torch.tensor(0.0),
        "snow_surface_temperature": torch.tensor(268.15),
        "solar_zenith_angle": torch.tensor(1.0),
        "snow_depth": torch.tensor(0.5),
        snow.age_state_key: torch.tensor(5.0),
        "snow_density": torch.tensor(300.0),
        "air_temperature": torch.tensor(268.15),
        "snow_grain_size": torch.tensor(0.5),
    }
    result = snow.forward(state, time=3600.0, dt=3600.0)
    assert "surface_albedo" in result
    albedo = _unwrap_update(result["surface_albedo"])
    assert torch.all(torch.isfinite(albedo))


def test_groundwater_dynamics_smoke() -> None:
    gw = GroundwaterDynamics()
    state = {
        "groundwater_storage": torch.tensor(100.0),
        "water_table_depth": torch.tensor(2.0),
        "drainage_flux": torch.tensor(1.0e-5),
        "capillary_rise_flux": torch.tensor(1.0e-6),
        "lateral_inflow": torch.tensor(2.0e-6),
        "groundwater_pumping": torch.tensor(0.0),
    }
    result = gw.forward(state, time=3600.0, dt=3600.0)
    assert set(result) == {
        "groundwater_storage",
        "water_table_depth",
        "baseflow_flux",
        "diagnostics.groundwater",
    }
    for key in ("groundwater_storage", "water_table_depth", "baseflow_flux"):
        _assert_tensor_finite_non_negative(_unwrap_update(result[key]))
