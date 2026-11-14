"""
Integration test: Snow albedo output coupling with Penman-Monteith ET.

Verifies that SnowAlbedo (hydrosphere/snow) feeds into PenmanMonteithET,
producing different evapotranspiration responses for fresh snow (high albedo)
versus aged snow (reduced albedo). Ensures parameter lookups rely solely on
ParameterManager (no monkey patches) and that the coupled workflow yields
physically consistent trends.
"""

import torch

from relic.core.state import UpdateItem

from relic.scientific.hydrosphere.snow.albedo import SnowAlbedo
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)


def _build_et_state(net_radiation: float) -> dict:
    """Construct minimal ET state tensor dictionary for Penman-Monteith."""
    dtype = torch.float32
    return {
        "net_radiation": torch.tensor([net_radiation], dtype=dtype),
        "air_temperature": torch.tensor([273.15 + 2.0], dtype=dtype),  # K
        "relative_humidity": torch.tensor([0.65], dtype=dtype),
        "wind_speed": torch.tensor([2.0], dtype=dtype),
        "atmospheric_pressure": torch.tensor([101325.0], dtype=dtype),
        "surface_resistance": torch.tensor([120.0], dtype=dtype),  # s m-1
    }


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def test_snow_albedo_coupled_evapotranspiration_response():
    snow = SnowAlbedo()
    et = PenmanMonteithET(config={"canopy_resistance_source": "state"})

    dt = 86400.0  # one day
    shortwave_in = 300.0  # W m-2 incoming shortwave radiation

    # Fresh snow event: strong snowfall resets albedo to high values
    fresh_state = {
        "snowfall": torch.tensor(0.01, dtype=torch.float32),
        "snow_surface_temperature": torch.tensor(268.0, dtype=torch.float32),
        "air_temperature": torch.tensor(268.0, dtype=torch.float32),
        "solar_zenith_angle": torch.tensor(1.0, dtype=torch.float32),
        "snow_depth": torch.tensor(0.2, dtype=torch.float32),
        # The snowfall reset restarts the scheme-owned age state from this value.
        snow.age_state_key: torch.tensor(0.0, dtype=torch.float32),
    }
    fresh_output = snow.forward(fresh_state, time=0.0, dt=dt)
    albedo_fresh = float(_unwrap_update(fresh_output["surface_albedo"]))

    # Aged snow: no new snowfall, allowing surface aging to reduce albedo
    aged_state = {
        "snowfall": torch.tensor(0.0, dtype=torch.float32),
        "snow_surface_temperature": torch.tensor(270.0, dtype=torch.float32),
        "air_temperature": torch.tensor(270.0, dtype=torch.float32),
        "solar_zenith_angle": torch.tensor(1.0, dtype=torch.float32),
        "snow_depth": torch.tensor(0.2, dtype=torch.float32),
        # The aged step continues from the age state the fresh step returned.
        snow.age_state_key: _unwrap_update(fresh_output[snow.age_state_key]),
    }
    aged_output = snow.forward(aged_state, time=dt, dt=dt)
    albedo_aged = float(_unwrap_update(aged_output["surface_albedo"]))

    # Ensure snow aging reduced albedo
    assert albedo_aged < albedo_fresh

    # Convert albedo to net radiation for ET calculations
    net_radiation_fresh = shortwave_in * (1.0 - albedo_fresh)
    net_radiation_aged = shortwave_in * (1.0 - albedo_aged)

    et_state_fresh = _build_et_state(net_radiation_fresh)
    et_state_aged = _build_et_state(net_radiation_aged)

    et_flux_fresh = _unwrap_update(
        et.forward(et_state_fresh, time=0.0, dt=3600.0)["evapotranspiration_flux"]
    )
    et_flux_aged = _unwrap_update(
        et.forward(et_state_aged, time=0.0, dt=3600.0)["evapotranspiration_flux"]
    )

    assert torch.isfinite(et_flux_fresh).all()
    assert torch.isfinite(et_flux_aged).all()

    # Lower albedo (aged snow) should increase net radiation and ET
    assert et_flux_aged.item() > et_flux_fresh.item()

