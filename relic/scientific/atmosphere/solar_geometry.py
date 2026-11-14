"""FAO-56 solar geometry: declination, solar time angle and cos(solar zenith angle).

Owner of the ``cos_zenith`` state quantity when it is derived from position and
time instead of being read from a radiation model. Pure tensor functions; no
state, no parameters of the ParameterManager (the coefficients below belong to
the published FAO-56 equations and are not tunable).

References:
    Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998). Crop
    evapotranspiration. FAO Irrigation and Drainage Paper 56, Chapter 3:
    Eq. 24 (declination), Eq. 28 (hourly extraterrestrial radiation, whose
    integrand over the solar time angle is cos(theta_z)), Eq. 31 (solar time
    angle at the period midpoint), Eqs. 32-33 (seasonal correction).
    Equation images of Eqs. 24, 28, 31 and 33 were read from
    https://www.fao.org/4/x0490e/x0490e07.htm (2026-10-17); Chapter 4,
    Example 19 gives J = 274, delta = -0.0753 rad, S_c = 0.1889 h and
    omega = 0.682 rad.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.utils import time_constants

# FAO-56 Eq. 24: delta = 0.409 sin(2 pi J / 365 - 1.39); amplitude [rad], phase [rad]. The fixed
# 365-day period is the noleap year length of the calendar SSOT (CalendarHandler), read at call time.
_DECLINATION_AMPLITUDE_RAD = 0.409
_DECLINATION_PHASE_RAD = 1.39
_FAO56_CALENDAR = "noleap"
# FAO-56 Eqs. 32-33: S_c = 0.1645 sin(2b) - 0.1255 cos(b) - 0.025 sin(b) [h], b = 2 pi (J - 81) / 364.
_SEASONAL_SIN2B_H = 0.1645
_SEASONAL_COSB_H = 0.1255
_SEASONAL_SINB_H = 0.025
_SEASONAL_REFERENCE_DAY = 81.0
_SEASONAL_PERIOD_DAYS = 364.0
# FAO-56 Eq. 31: 0.06667 h per degree of longitude (1/15 rounded as published).
_HOURS_PER_DEGREE = 0.06667


def _common_tensors(*values: object) -> Tuple[Tensor, ...]:
    """Promote inputs to one floating dtype and device and broadcast them (no narrowing)."""
    tensors = [torch.as_tensor(value) for value in values]
    devices = {tensor.device for tensor in tensors}
    if len(devices) > 1:
        raise ValueError("Solar geometry inputs must share one device.")
    for tensor in tensors:
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise TypeError("Solar geometry inputs must be real numbers.")
    dtype = torch.get_default_dtype()
    floating = [tensor.dtype for tensor in tensors if tensor.is_floating_point()]
    if floating:
        dtype = floating[0]
        for other in floating[1:]:
            dtype = torch.promote_types(dtype, other)
    return tuple(torch.broadcast_tensors(*(tensor.to(dtype=dtype) for tensor in tensors)))


def _check_day_of_year(day: Tensor) -> None:
    if not bool(torch.isfinite(day).all()) or not torch.equal(day, torch.round(day)):
        raise ValueError("day_of_year must contain finite integral values.")
    if bool(torch.any((day < 1) | (day > 366))):
        raise ValueError("day_of_year must lie in [1, 366].")


def solar_declination_fao56(day_of_year: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eq. 24.
    # Mathematical expression: delta = 0.409 sin(2 pi J / 365 - 1.39).
    # Parameter meanings: J is the day number in the year (1 January = 1).
    # Units: J [day]; delta [rad].
    # Domain/constraints: integral J in [1, 366]; the fixed 365-day period is FAO-56's.
    (day,) = _common_tensors(day_of_year)
    _check_day_of_year(day)
    pi = torch.acos(day.new_tensor(-1.0))
    year_days = float(CalendarHandler.days_per_simulator_year(_FAO56_CALENDAR))
    return day.new_tensor(_DECLINATION_AMPLITUDE_RAD) * torch.sin(
        2 * pi * day / day.new_tensor(year_days) - day.new_tensor(_DECLINATION_PHASE_RAD))


def seasonal_correction_fao56(day_of_year: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eqs. 32 and 33 (equation of time).
    # Mathematical expression: b = 2 pi (J - 81) / 364;
    #   S_c = 0.1645 sin(2 b) - 0.1255 cos(b) - 0.025 sin(b).
    # Parameter meanings: J is the day number in the year; S_c is the seasonal
    #   correction for solar time.
    # Units: J [day]; b [rad]; S_c [h].
    (day,) = _common_tensors(day_of_year)
    _check_day_of_year(day)
    pi = torch.acos(day.new_tensor(-1.0))
    b = 2 * pi * (day - day.new_tensor(_SEASONAL_REFERENCE_DAY)) / day.new_tensor(_SEASONAL_PERIOD_DAYS)
    return (day.new_tensor(_SEASONAL_SIN2B_H) * torch.sin(2 * b)
            - day.new_tensor(_SEASONAL_COSB_H) * torch.cos(b)
            - day.new_tensor(_SEASONAL_SINB_H) * torch.sin(b))


def solar_time_angle_fao56(day_of_year: object, utc_hour: object, longitude_deg_east: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eq. 31 with the Greenwich zone.
    # Mathematical expression: omega = pi / 12 [(t + 0.06667 (L_z - L_m) + S_c) - 12]
    #   with t the UTC clock hour, L_z = 0 and L_m = -lambda, lambda being the east
    #   longitude normalized to [-180, 180), so 0.06667 (L_z - L_m) = 0.06667 lambda.
    # Parameter meanings: t is the UTC time of the instant (midpoint of a period);
    #   lambda is the east longitude; S_c is the seasonal correction (Eq. 32).
    # Units: t [h]; lambda [degree east]; S_c [h]; omega [rad].
    # Domain/constraints: t in [0, 24]; finite longitude; integral J in [1, 366].
    day, hour, longitude = _common_tensors(day_of_year, utc_hour, longitude_deg_east)
    if not bool(torch.isfinite(hour).all()) or bool(torch.any((hour < 0) | (hour > 24))):
        raise ValueError("utc_hour must lie in [0, 24].")
    if not bool(torch.isfinite(longitude).all()):
        raise ValueError("longitude must be finite.")
    longitude = torch.remainder(longitude + 180.0, 360.0) - 180.0
    pi = torch.acos(day.new_tensor(-1.0))
    solar_hour = hour + day.new_tensor(_HOURS_PER_DEGREE) * longitude + seasonal_correction_fao56(day)
    return pi / 12 * (solar_hour - 12)


def cosine_solar_zenith_fao56(latitude_deg: object, longitude_deg_east: object,
                              day_of_year: object, utc_hour: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eq. 28: R_a is proportional to the integral
    #   over omega of sin(phi) sin(delta) + cos(phi) cos(delta) cos(omega), i.e. cos(theta_z).
    # Mathematical expression: cos(theta_z) = max(sin(phi) sin(delta)
    #   + cos(phi) cos(delta) cos(omega), 0), with delta from Eq. 24 and omega from Eq. 31.
    # Parameter meanings: phi latitude; delta declination; omega solar time angle;
    #   theta_z solar zenith angle; the night side (sun below the horizon) is set to 0,
    #   as FAO-56 sets R_a to zero for omega outside the sunset hour angles.
    # Units: phi [rad] from degrees north; delta, omega, theta_z [rad]; cos(theta_z) [1].
    # Domain/constraints: latitude in [-90, 90]; result in [0, 1] (horizontal
    #   projection factor of direct solar radiation; no refraction or elevation term).
    latitude, longitude, day, hour = _common_tensors(latitude_deg, longitude_deg_east, day_of_year, utc_hour)
    _check_latitude(latitude)
    omega = solar_time_angle_fao56(day, hour, longitude)
    return cosine_solar_zenith_from_hour_angle(latitude, day, omega)


def _check_latitude(latitude: Tensor) -> None:
    if not bool(torch.isfinite(latitude).all()) or bool(torch.any(latitude.abs() > 90)):
        raise ValueError("latitude must lie in [-90, 90] degrees.")


def cosine_solar_zenith_from_hour_angle(latitude_deg: object, day_of_year: object,
                                        hour_angle_rad: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eq. 28 (integrand) with delta of Eq. 24.
    # Mathematical expression: cos(theta_z) = sin(beta) = max(sin(phi) sin(delta)
    #   + cos(phi) cos(delta) cos(omega), 0), clipped to [0, 1].
    # Parameter meanings: phi latitude; delta declination; omega the solar time (hour)
    #   angle, 0 at solar noon; beta the solar elevation; below the horizon the value is 0.
    # Units: phi [rad] from degrees north; delta, omega [rad]; result [1].
    # Domain/constraints: latitude in [-90, 90]; integral day_of_year in [1, 366]; the
    #   arguments broadcast together (omega may carry extra trailing axes).
    latitude, day, omega = _common_tensors(latitude_deg, day_of_year, hour_angle_rad)
    _check_latitude(latitude)
    phi = torch.deg2rad(latitude)
    delta = solar_declination_fao56(day)
    cosine = torch.sin(phi) * torch.sin(delta) + torch.cos(phi) * torch.cos(delta) * torch.cos(omega)
    return torch.clamp(cosine, min=0.0, max=1.0)


def sunset_hour_angle_fao56(latitude_deg: object, day_of_year: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eq. 25, omega_s = arccos[-tan(phi) tan(delta)],
    #   read from https://www.fao.org/4/x0490e/x0490e07.htm (2026-11-06); Example 9 there
    #   gives omega_s = 1.527 rad for 3 September (J = 246) at 20 degrees S.
    # Mathematical expression: c = -tan(phi) tan(delta); omega_s = arccos(c) for |c| < 1,
    #   omega_s = 0 for c >= 1 (polar night: the sun stays below the horizon) and
    #   omega_s = pi for c <= -1 (polar day: the sun stays above the horizon).
    # Parameter meanings: phi latitude; delta declination (Eq. 24); omega_s the sunset
    #   hour angle, half the daylight arc. FAO-56 does not treat |c| >= 1; the two clamps
    #   are the geometric limits of Eq. 25 (no refraction or solar-disc correction).
    # Units: phi, delta, omega_s [rad]; latitude [degree north].
    # Domain/constraints: latitude in [-90, 90]; integral day_of_year in [1, 366]; the
    #   arccos is evaluated only on |c| < 1, so no infinite derivative is formed at the
    #   polar limits; the result is constant (zero derivative) outside them.
    latitude, day = _common_tensors(latitude_deg, day_of_year)
    _check_latitude(latitude)
    phi = torch.deg2rad(latitude)
    delta = solar_declination_fao56(day)
    argument = -torch.tan(phi) * torch.tan(delta)
    inside = argument.abs() < 1.0
    safe = torch.where(inside, argument, torch.zeros_like(argument))
    pi = torch.acos(argument.new_tensor(-1.0))
    limit = torch.where(argument >= 1.0, torch.zeros_like(argument), pi.expand_as(argument))
    return torch.where(inside, torch.acos(safe), limit)


def daylight_hours_fao56(latitude_deg: object, day_of_year: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eq. 34, N = 24 omega_s / pi (same page as
    #   Eq. 25; Example 9: N = 11.7 h for omega_s = 1.527 rad).
    # Mathematical expression: N = 24 omega_s / pi with omega_s of sunset_hour_angle_fao56.
    # Parameter meanings: N the daylight hours (day length); omega_s the sunset hour angle.
    # Units: omega_s [rad]; N [h], in [0, 24] (0 polar night, 24 polar day).
    # Domain/constraints: as sunset_hour_angle_fao56.
    omega_s = sunset_hour_angle_fao56(latitude_deg, day_of_year)
    pi = torch.acos(omega_s.new_tensor(-1.0))
    hours_per_day = float(time_constants.SECONDS_PER_DAY) / float(time_constants.SECONDS_PER_HOUR)
    return omega_s * hours_per_day / pi


def _day_and_hour_of_instant(instant_utc: object) -> Tuple[float, float]:
    """Day of year [1, 366] and decimal UTC hour [0, 24) of one instant (naive values are UTC)."""
    import numpy as np

    stamp = np.datetime64(instant_utc, "ns")
    if np.isnat(stamp):
        raise ValueError("instant_utc must be a valid time.")
    day = stamp.astype("datetime64[D]")
    year = stamp.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = float((day - year).astype(np.int64) + 1)
    # Units: elapsed time of the day [ns] divided by SECONDS_PER_HOUR [s h-1] * 1e9 [ns s-1] -> [h].
    elapsed_ns = float((stamp - day).astype("timedelta64[ns]").astype(np.int64))
    hour = elapsed_ns / (float(time_constants.SECONDS_PER_HOUR) * 1.0e9)
    return day_of_year, hour


def cos_solar_zenith(latitude_deg: object, longitude_deg: object, instant_utc: object) -> Tensor:
    # References: Allen et al. (1998) FAO-56 Eqs. 24, 28, 31-33 (see cosine_solar_zenith_fao56).
    # Mathematical expression: cos(theta_z)(phi, lambda, t) = cosine_solar_zenith_fao56(phi, lambda, J(t), h(t)).
    # Parameter meanings: phi latitude and lambda east longitude of each cell; t one
    #   instantaneous UTC time (numpy datetime64, datetime or ISO text; naive means UTC);
    #   J(t) its day of year and h(t) its decimal UTC hour, used as the FAO-56 period midpoint.
    # Units: phi [degree north]; lambda [degree east]; h [h]; result [1] in [0, 1].
    # Domain/constraints: latitude and longitude broadcast together; one instant per call.
    day_of_year, hour = _day_and_hour_of_instant(instant_utc)
    latitude, longitude = _common_tensors(latitude_deg, longitude_deg)
    return cosine_solar_zenith_fao56(latitude, longitude, latitude.new_tensor(day_of_year), latitude.new_tensor(hour))
