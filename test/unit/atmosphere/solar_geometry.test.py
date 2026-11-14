"""FAO-56 solar geometry: declination (Eq. 24), seasonal correction (Eqs. 32-33),
solar time angle (Eq. 31) and the cosine of the solar zenith angle (the integrand
of Eq. 28), checked against FAO-56 Example 19 and geometric limits.

Reference: Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998), FAO
Irrigation and Drainage Paper 56, Chapter 3 (Eqs. 24, 28, 31-33) and Chapter 4
(Example 19, N'Diaye, Senegal, 1 October, J = 274).
"""

import math

import pytest
import torch

from relic.scientific.atmosphere.solar_geometry import (
    cosine_solar_zenith_fao56,
    seasonal_correction_fao56,
    solar_declination_fao56,
    solar_time_angle_fao56,
)


def test_fao56_example_19_declination_seasonal_correction_and_time_angle():
    # Example 19: J = 274, delta = -0.0753 rad, b = 3.3315, S_c = 0.1889 h; the
    # 14.00-15.00 h local period at L_z = 15, L_m = 16.25 degrees west is
    # 15.5 h UTC at longitude -16.25 degrees east and gives omega = 0.682 rad.
    day = torch.tensor(274.0, dtype=torch.float64)
    assert float(solar_declination_fao56(day)) == pytest.approx(-0.0753, abs=5e-5)
    assert float(seasonal_correction_fao56(day)) == pytest.approx(0.1889, abs=5e-5)
    omega = solar_time_angle_fao56(day, torch.tensor(15.5, dtype=torch.float64),
                                   torch.tensor(-16.25, dtype=torch.float64))
    assert float(omega) == pytest.approx(0.682, abs=5e-4)
    omega_night = solar_time_angle_fao56(day, torch.tensor(3.5, dtype=torch.float64),
                                         torch.tensor(-16.25, dtype=torch.float64))
    assert float(omega_night) == pytest.approx(-2.46, abs=5e-3)


def test_cosine_is_the_eq28_integrand_at_the_example_instant():
    lat, lon = torch.tensor(16.22, dtype=torch.float64), torch.tensor(-16.25, dtype=torch.float64)
    day, hour = torch.tensor(274.0, dtype=torch.float64), torch.tensor(15.5, dtype=torch.float64)
    phi, delta = math.radians(16.22), float(solar_declination_fao56(day))
    omega = float(solar_time_angle_fao56(day, hour, lon))
    expected = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(omega)
    assert float(cosine_solar_zenith_fao56(lat, lon, day, hour)) == pytest.approx(expected, rel=1e-12)


def test_equinox_solar_noon_at_the_equator_is_overhead():
    day = torch.tensor(81.0, dtype=torch.float64)
    noon_utc = 12.0 - float(seasonal_correction_fao56(day))
    value = cosine_solar_zenith_fao56(torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64),
                                      day, torch.tensor(noon_utc, dtype=torch.float64))
    assert float(value) > 0.99999


def test_polar_night_and_night_side_are_zero_not_negative():
    hours = torch.arange(0.0, 24.0, 0.5, dtype=torch.float64)
    polar = cosine_solar_zenith_fao56(torch.full_like(hours, 85.0), torch.zeros_like(hours),
                                      torch.full_like(hours, 355.0), hours)
    assert torch.all(polar == 0.0)
    midnight = cosine_solar_zenith_fao56(torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64),
                                         torch.tensor(81.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    assert float(midnight) == 0.0


def test_grid_broadcast_bounds_dtype_and_dateline_continuity():
    lat = torch.linspace(-90.0, 90.0, 37, dtype=torch.float32).reshape(-1, 1, 1)
    lon = torch.linspace(-180.0, 179.5, 72, dtype=torch.float32).reshape(1, -1, 1)
    hour = torch.arange(0.0, 24.0, 3.0, dtype=torch.float32).reshape(1, 1, -1)
    value = cosine_solar_zenith_fao56(lat, lon, torch.tensor(196.0, dtype=torch.float32), hour)
    assert value.shape == (37, 72, 8) and value.dtype == torch.float32
    assert torch.all((value >= 0.0) & (value <= 1.0))
    east = cosine_solar_zenith_fao56(torch.tensor(10.0), torch.tensor(180.0), torch.tensor(196.0), torch.tensor(1.5))
    west = cosine_solar_zenith_fao56(torch.tensor(10.0), torch.tensor(-180.0), torch.tensor(196.0), torch.tensor(1.5))
    assert torch.equal(east, west)


@pytest.mark.parametrize(
    "lat, lon, day, hour",
    [(91.0, 0.0, 10.0, 0.0), (0.0, float("nan"), 10.0, 0.0), (0.0, 0.0, 0.0, 0.0),
     (0.0, 0.0, 367.0, 0.0), (0.0, 0.0, 10.5, 0.0), (0.0, 0.0, 10.0, 24.5)],
)
def test_out_of_domain_inputs_raise(lat, lon, day, hour):
    with pytest.raises(ValueError):
        cosine_solar_zenith_fao56(torch.tensor(lat), torch.tensor(lon), torch.tensor(day), torch.tensor(hour))


def test_instant_contract_matches_day_and_hour_form_on_numpy_cells():
    import numpy as np

    from relic.scientific.atmosphere.solar_geometry import cos_solar_zenith

    lat = np.array([16.22, -33.9, 60.0], dtype=np.float64)
    lon = np.array([-16.25, 151.2, 24.9], dtype=np.float64)
    instant = np.datetime64("2023-10-01T15:30", "m")  # day 274 of 2023
    value = np.asarray(cos_solar_zenith(lat, lon, instant))
    expected = cosine_solar_zenith_fao56(torch.as_tensor(lat), torch.as_tensor(lon),
                                         torch.tensor(274.0, dtype=torch.float64), torch.tensor(15.5, dtype=torch.float64))
    assert value.shape == (3,) and value.dtype == np.float64
    np.testing.assert_array_equal(value, expected.numpy())
    leap = np.asarray(cos_solar_zenith(lat, lon, np.datetime64("2024-12-31T00:00", "h")))
    assert np.all((leap >= 0.0) & (leap <= 1.0))
    with pytest.raises(ValueError):
        cos_solar_zenith(lat, lon, np.datetime64("NaT"))
