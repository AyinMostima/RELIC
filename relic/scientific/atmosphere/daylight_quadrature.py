"""Daylight Gauss quadrature: daily means of instantaneous laws on a daily step.

Shared owner of the "daylight_gauss" temporal support. A process on a daily step
(dt = 86400 s) that evaluates an instantaneous, nonlinear law (light and temperature
responses) on 24-hour mean drivers carries the Jensen bias f(mean x) != mean f(x).
This module keeps the daily step and changes only how the law is evaluated: the law is
evaluated at n Gauss-Legendre instants of the daylight period and once for the night,
and the daily mean is the weighted sum of these samples (Goudriaan 1986; Goudriaan and
van Laar 1994; the Gauss scheme of the WOFOST/SUCROS family). Pure tensor functions,
no state and no ParameterManager parameter: the coefficients are those of the cited
equations and are not tunable.

Samples
-------
A daylight support has K = n + 1 samples on a trailing axis: k = 0..n-1 the Gauss
instants in time order (sunrise to sunset) and k = n the night. Sample weights
    W_i = (N / 24) w_i  (i < n),   W_n = 1 - N / 24,   sum_k W_k = 1,
with N the FAO-56 day length and w_i the Gauss-Legendre weights normalized to sum 1.
The daily mean of a law f is F = sum_k W_k f(x_k): the daylight term is the Gauss
estimate of (1/24) integral over the day of f, and the night term keeps dark
respiration and the light-independent emission fraction with one night value.

References
----------
- Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998). Crop evapotranspiration.
  FAO Irrigation and Drainage Paper 56: Eq. 24 (declination), Eq. 25 (sunset hour
  angle), Eq. 28 (cos theta_z integrand), Eq. 34 (daylight hours N = 24 omega_s / pi);
  read from https://www.fao.org/4/x0490e/x0490e07.htm (2026-11-06; Example 9:
  omega_s = 1.527 rad, N = 11.7 h on 3 September at 20 degrees S); implemented in
  relic.scientific.atmosphere.solar_geometry.
- Goudriaan, J. (1986). A simple and fast numerical method for the computation of daily
  totals of crop photosynthesis. Agricultural and Forest Meteorology 38, 249-254.
- Goudriaan, J. and van Laar, H.H. (1994). Modelling Potential Crop Growth Processes.
  Kluwer Academic Publishers, Dordrecht (cited for the scheme; not read here).
- WOFOST implementation of these schemes in PCSE (Wageningen Environmental Research),
  source read 2026-11-06: https://pcse.readthedocs.io/en/stable/_modules/pcse/util.html
  (function astro: DAYL = 12 (1 + 2 asin(AOB) / pi), 24 h and 0 h for AOB > 1 and
  AOB < -1; DSINBE = integral of sin(beta) (1 + 0.4 sin(beta)) over the day) and
  https://pcse.readthedocs.io/en/stable/_modules/pcse/crop/assimilation.html
  (function totass8: three Gauss points XGAUSS = 0.1127017, 0.5, 0.8872983 with
  weights 0.2777778, 0.4444444, 0.2777778 on the half day from noon to sunset,
  HOUR = 12 + 0.5 DAYL XGAUSS, SINB = max(0, SINLD + COSLD cos(2 pi (HOUR + 12) / 24)),
  PAR = 0.5 AVRAD SINB (1 + 0.4 SINB) / DSINBE, DTGA = DAYL sum_i WGAUSS_i FGROS_i).
  The 0.4 sin(beta) term is the increase of the atmospheric transmissivity with solar
  elevation used there; RELIC normalizes the shape by its own Gauss sum instead of the
  analytic DSINBE so that the daily total is conserved exactly by the quadrature.
- SWAT+ Theoretical Documentation, Section 1:1.3.2 Hourly Air Temperature,
  Eq. 1:1.3.1, T_hr = T_av + (T_mx - T_mn) / 2 cos(0.2618 (hr - 15)), maximum at
  1500 h and minimum at 0300 h (Campbell 1985); read 2026-11-06 from
  https://swatplus.gitbook.io/io-docs/theoretical-documentation/section-1-climate/untitled/1-1.3-temperature/1-1.3.2-hourly-air-temperature
  (0.2618 = 2 pi / 24 rounded; RELIC uses pi / 12 exactly). Campbell, G.S. (1985).
  Soil Physics with BASIC. Elsevier (cited by SWAT+, not read here). Parton and Logan
  (1981) was not accessible and is not used.

Units
-----
latitude [degree north]; day_of_year [1]; N [h]; omega [rad]; sin(beta) [1];
temperatures [K]; vapour pressures [Pa]; a redistributed flux keeps its own unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from relic.core.state import UpdateItem
from relic.scientific.atmosphere.solar_geometry import (
    cosine_solar_zenith_from_hour_angle,
    sunset_hour_angle_fao56,
)
from relic.scientific.hydrosphere.utils.psychrometrics import compute_saturation_vapor_pressure
from relic.utils import time_constants
from relic.utils.physical_constants import get_physical_constant

# Name of the temporal support selected by config["temporal_support"] of a process.
DAYLIGHT_GAUSS = "daylight_gauss"
# Config keys of the support: the number n of Gauss instants (3 or 5, required) and the
# daylight shape of shortwave-proportional fluxes (optional, default below).
NODE_COUNT_KEY = "daylight_gauss_nodes"
SHORTWAVE_SHAPE_KEY = "daylight_shortwave_shape"
NODE_COUNTS = (3, 5)
#   "sine_elevation": S(t) proportional to sin(beta(t)) (FAO-56 Eq. 28 integrand).
#   "sine_elevation_transmissivity": S(t) proportional to sin(beta) (1 + 0.4 sin(beta))
#     (Goudriaan and van Laar 1994; PCSE astro/totass8, see module references).
SHORTWAVE_SHAPES = ("sine_elevation", "sine_elevation_transmissivity")
DEFAULT_SHORTWAVE_SHAPE = "sine_elevation_transmissivity"
# References: PCSE astro and totass8 (module references). Mathematical expression:
#   tau(beta) proportional to 1 + c sin(beta). Parameter meanings: c the published slope.
# Units: c [1]. Domain/constraints: fixed published coefficient, not tunable.
_TRANSMISSIVITY_SLOPE = 0.4
# References: SWAT+ Eq. 1:1.3.1 (Campbell 1985). Mathematical expression: the cosine
#   course peaks at hr = 15 h. Parameter meanings: the published solar hour of the daily
#   maximum. Units: [h]. Domain/constraints: fixed published coefficient.
_TEMPERATURE_PEAK_HOUR = 15.0
# Solar noon of the solar-time convention of FAO-56 Eq. 31 (omega = 0 at 12 h) [h].
_SOLAR_NOON_HOUR = 12.0


def _hours_per_day() -> float:
    return float(time_constants.SECONDS_PER_DAY) / float(time_constants.SECONDS_PER_HOUR)


def gauss_legendre_rule(node_count: int) -> Tuple[np.ndarray, np.ndarray]:
    # References: Abramowitz, M. and Stegun, I.A. (1964). Handbook of Mathematical
    #   Functions, Section 25.4.29 (Gauss-Legendre quadrature); numpy.polynomial.legendre.
    # Mathematical expression: integral_{-1}^{1} g(x) dx ~ sum_i a_i g(x_i), exact for
    #   polynomials of degree <= 2n - 1; returned weights w_i = a_i / 2 sum to 1, so
    #   (1/2) integral_{-1}^{1} g dx ~ sum_i w_i g(x_i).
    # Parameter meanings: n the node count; x_i the roots of the Legendre polynomial P_n.
    # Units: x, w [1].
    # Domain/constraints: n in NODE_COUNTS; float64 nodes, symmetric about 0.
    if node_count not in NODE_COUNTS:
        raise ValueError(f"daylight Gauss node count must be one of {list(NODE_COUNTS)}, got {node_count!r}")
    nodes, weights = np.polynomial.legendre.leggauss(node_count)
    return nodes.astype(np.float64), (weights / 2.0).astype(np.float64)


def daylight_config(configuration: Mapping[str, Any], owner: str) -> Tuple[int, str]:
    """Validate the config contract of the daylight_gauss support; return (n, shape)."""
    node_count = configuration.get(NODE_COUNT_KEY)
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count not in NODE_COUNTS:
        raise ValueError(
            f"{owner} temporal_support '{DAYLIGHT_GAUSS}' requires config[{NODE_COUNT_KEY!r}] "
            f"as one of {list(NODE_COUNTS)}, got {node_count!r}")
    shape = configuration.get(SHORTWAVE_SHAPE_KEY, DEFAULT_SHORTWAVE_SHAPE)
    if shape not in SHORTWAVE_SHAPES:
        raise ValueError(
            f"{owner} config[{SHORTWAVE_SHAPE_KEY!r}] must be one of {list(SHORTWAVE_SHAPES)}, got {shape!r}")
    return node_count, shape


def reject_daylight_keys(configuration: Mapping[str, Any], owner: str) -> None:
    """Refuse the daylight keys on any other temporal support (no silent ignore)."""
    stray = [key for key in (NODE_COUNT_KEY, SHORTWAVE_SHAPE_KEY) if key in configuration]
    if stray:
        raise ValueError(f"{owner} config keys {stray} need temporal_support '{DAYLIGHT_GAUSS}'")


def require_daily_step(dt: Any, owner: str) -> None:
    """The daylight support integrates one calendar day: dt must equal seconds_per_day."""
    seconds_per_day = float(get_physical_constant("seconds_per_day"))
    if float(dt) != seconds_per_day:
        raise ValueError(
            f"{owner} temporal_support '{DAYLIGHT_GAUSS}' requires dt={seconds_per_day} s, got dt={float(dt)} s")


@dataclass(frozen=True)
class DaylightSupport:
    """Samples and weights of one daylight Gauss support (trailing sample axis).

    day_length [h], daylight_fraction [1] and light_fallback [bool] have the cell shape
    S; node_solar_hour [h] and shortwave_shape [1] have S + (n,); sample_weights [1]
    has S + (n + 1,) and sums to 1 over the last axis.
    """

    node_count: int
    day_length: Tensor
    daylight_fraction: Tensor
    node_solar_hour: Tensor
    shortwave_shape: Tensor
    sample_weights: Tensor
    light_fallback: Tensor

    @property
    def sample_count(self) -> int:
        return self.node_count + 1

    def constant(self, value: Tensor) -> Tensor:
        """A value constant over the day on the sample axis (S + (n + 1,))."""
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.sample_weights.dtype, device=self.sample_weights.device)
        shape = torch.broadcast_shapes(tuple(value.shape), tuple(self.day_length.shape))
        return value.expand(shape).unsqueeze(-1).expand(*shape, self.sample_count)

    def daily_mean(self, samples: Tensor) -> Tensor:
        # References: module docstring (Goudriaan 1986; Goudriaan and van Laar 1994).
        # Mathematical expression: F = sum_k W_k f_k = (N/24) sum_i w_i f(x_i)
        #   + (1 - N/24) f(x_night).
        # Parameter meanings: f_k the law at sample k; W_k the sample weights.
        # Units: F and f_k share the unit of the law.
        # Domain/constraints: samples on the trailing axis of length n + 1.
        # Sign convention: as the law.
        if samples.shape[-1] != self.sample_count:
            raise ValueError(f"daylight samples need a trailing axis of length {self.sample_count}")
        return (samples * self.sample_weights.to(dtype=samples.dtype)).sum(dim=-1)


def daylight_support(latitude_deg: Any, day_of_year: Any, node_count: int, shortwave_shape: str,
                     *, dtype: Optional[torch.dtype] = None) -> DaylightSupport:
    # References: FAO-56 Eqs. 24, 25, 28 and 34 (solar_geometry); Goudriaan (1986);
    #   PCSE totass8 (module references).
    # Mathematical expression: omega_s (Eq. 25), N = 24 omega_s / pi (Eq. 34);
    #   t_i = 12 + (N / 2) x_i (solar hour of Gauss node x_i in [-1, 1]);
    #   omega_i = omega_s x_i; sin(beta_i) = max(sin(phi) sin(delta)
    #   + cos(phi) cos(delta) cos(omega_i), 0); s_i = sin(beta_i) or
    #   sin(beta_i) (1 + 0.4 sin(beta_i)); W_i = (N/24) w_i, W_night = 1 - N/24.
    # Parameter meanings: x_i, w_i the normalized Gauss-Legendre rule; omega_s the sunset
    #   hour angle; N the day length; s_i the daylight shape of shortwave fluxes.
    # Units: latitude [degree north]; N, t [h]; omega [rad]; s, W [1].
    # Domain/constraints: latitude in [-90, 90]; integral day_of_year in [1, 366].
    #   Symmetry about solar noon is used where it holds: the nodes are symmetric and
    #   s_i is evaluated on the half x_i >= 0 and mirrored, so mirrored instants receive
    #   bitwise identical light; temperature is not symmetric (its maximum is at 15 h)
    #   and is evaluated at every node. light_fallback marks N = 0 (polar night, no
    #   daylight instant), where a positive daily flux is placed in the night sample.
    # Sign convention: not applicable (geometry and weights).
    if shortwave_shape not in SHORTWAVE_SHAPES:
        raise ValueError(f"shortwave shape must be one of {list(SHORTWAVE_SHAPES)}, got {shortwave_shape!r}")
    nodes, weights = gauss_legendre_rule(node_count)
    latitude = torch.as_tensor(latitude_deg)
    day = torch.as_tensor(day_of_year)
    if dtype is None:
        dtype = latitude.dtype if latitude.is_floating_point() else torch.get_default_dtype()
    latitude = latitude.to(dtype=dtype)
    day = day.to(dtype=dtype, device=latitude.device)
    latitude, day = torch.broadcast_tensors(latitude, day)
    omega_s = sunset_hour_angle_fao56(latitude, day)
    pi = torch.acos(omega_s.new_tensor(-1.0))
    hours_per_day = _hours_per_day()
    day_length = omega_s * hours_per_day / pi
    daylight_fraction = day_length / hours_per_day
    x = torch.as_tensor(nodes, dtype=dtype, device=latitude.device)
    w = torch.as_tensor(weights, dtype=dtype, device=latitude.device)
    node_solar_hour = _SOLAR_NOON_HOUR + 0.5 * day_length.unsqueeze(-1) * x
    # Mirror symmetry of the light: evaluate sin(beta) at |x_i| and index it back.
    mirror = np.abs(nodes)
    half, inverse = np.unique(np.round(mirror, 15), return_inverse=True)
    half_x = torch.as_tensor(half, dtype=dtype, device=latitude.device)
    half_sine = cosine_solar_zenith_from_hour_angle(
        latitude.unsqueeze(-1), day.unsqueeze(-1), omega_s.unsqueeze(-1) * half_x)
    sine_elevation = half_sine[..., torch.as_tensor(inverse.reshape(-1), device=latitude.device)]
    if shortwave_shape == "sine_elevation_transmissivity":
        shape = sine_elevation * (1.0 + _TRANSMISSIVITY_SLOPE * sine_elevation)
    else:
        shape = sine_elevation
    night_weight = (1.0 - daylight_fraction).unsqueeze(-1)
    sample_weights = torch.cat((daylight_fraction.unsqueeze(-1) * w, night_weight), dim=-1)
    return DaylightSupport(
        node_count=node_count, day_length=day_length, daylight_fraction=daylight_fraction,
        node_solar_hour=node_solar_hour, shortwave_shape=shape, sample_weights=sample_weights,
        light_fallback=day_length <= 0.0)


def daylight_flux_samples(daily_mean_flux: Tensor, support: DaylightSupport) -> Tensor:
    # References: Goudriaan and van Laar (1994); PCSE totass8 PAR = 0.5 AVRAD s / DSINBE
    #   (module references), normalized by the Gauss sum for exact conservation.
    # Mathematical expression: F_i = F_mean s_i / ((N/24) sum_j w_j s_j) for daylight
    #   nodes, F_night = 0; hence sum_k W_k F_k = F_mean exactly (daily total
    #   F_mean * 86400 s conserved). Where N = 0 (light_fallback) the daylight weights are
    #   0 and F_night = F_mean, the uniform 24-hour placement of the daily-mean support.
    # Parameter meanings: F_mean the 24-hour mean of a shortwave-proportional flux
    #   (SSRD, net shortwave or PPFD of the radiation provider); s_i the daylight shape.
    # Units: F keeps the unit of F_mean (W m-2 or umol photons m-2 s-1).
    # Domain/constraints: F_mean finite and >= 0. Near the polar-night edge (N -> 0+)
    #   a positive F_mean is concentrated on a short day of low sun and F_i can exceed
    #   the clear-sky value; FAO-56 geometry has no refraction term (declared limitation).
    # Sign convention: downward fluxes positive.
    flux = daily_mean_flux
    if not bool(torch.isfinite(flux).all()) or bool(torch.any(flux < 0.0)):
        raise ValueError("daylight flux samples need a finite, non-negative daily mean flux")
    shape = support.shortwave_shape.to(dtype=flux.dtype)
    weights = support.sample_weights[..., :-1].to(dtype=flux.dtype)
    denominator = (weights * shape).sum(dim=-1)
    lit = denominator > 0.0
    safe = torch.where(lit, denominator, torch.ones_like(denominator))
    flux, safe, lit = torch.broadcast_tensors(flux, safe, lit)
    day = flux.unsqueeze(-1) * shape / safe.unsqueeze(-1)
    day = torch.where(lit.unsqueeze(-1), day, torch.zeros_like(day))
    night = torch.where(lit, torch.zeros_like(flux), flux)
    return torch.cat((day, night.unsqueeze(-1)), dim=-1)


def diurnal_air_temperature_samples(mean_temperature: Tensor, maximum_temperature: Tensor,
                                    minimum_temperature: Tensor, support: DaylightSupport) -> Tensor:
    # References: SWAT+ Theoretical Documentation Eq. 1:1.3.1 (Campbell 1985), module
    #   references.
    # Mathematical expression: T(h) = T_av + (T_mx - T_mn)/2 cos(pi/12 (h - 15));
    #   T_i = T(t_i) at the daylight nodes; T_night = mean of T(h) over the night
    #   [12 + N/2, 36 - N/2] = T_av + (T_mx - T_mn)/2 (12 / (pi Z))
    #   [sin(pi/12 (21 - N/2)) - sin(pi/12 (N/2 - 3))], Z = 24 - N.
    # Parameter meanings: T_av the daily mean air temperature (the "average temperature
    #   on the day" of Eq. 1:1.3.1, here the daily-mean input); T_mx, T_mn the daily
    #   extremes; h, t_i solar hours; Z the night length.
    # Units: T [K]; h, N, Z [h].
    # Domain/constraints: finite T_mn <= T_av <= T_mx (ValueError otherwise). The
    #   24-hour mean of T(h) equals T_av; the course reaches T_av + (T_mx - T_mn)/2 at
    #   15 h, not necessarily T_mx. Polar day (Z = 0): the night weight is 0 and the
    #   night value is set to T_av (no division by Z). Polar night (N = 0): T_night = T_av.
    # Sign convention: not applicable (temperature).
    t_mean, t_max, t_min = torch.broadcast_tensors(mean_temperature, maximum_temperature, minimum_temperature)
    for name, value in (("air_temperature", t_mean), ("air_temperature_max", t_max),
                        ("air_temperature_min", t_min)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"daylight temperature course needs finite {name}")
    if bool(torch.any(t_min > t_mean)) or bool(torch.any(t_mean > t_max)):
        raise ValueError("daylight temperature course needs air_temperature_min <= air_temperature <= air_temperature_max")
    dtype = t_mean.dtype
    pi = torch.acos(t_mean.new_tensor(-1.0))
    angular = pi / 12.0
    amplitude = 0.5 * (t_max - t_min)
    hours = support.node_solar_hour.to(dtype=dtype)
    day = t_mean.unsqueeze(-1) + amplitude.unsqueeze(-1) * torch.cos(angular * (hours - _TEMPERATURE_PEAK_HOUR))
    day_length = support.day_length.to(dtype=dtype)
    night_length = _hours_per_day() - day_length
    has_night = night_length > 0.0
    safe_night = torch.where(has_night, night_length, torch.ones_like(night_length))
    upper = torch.sin(angular * (_hours_per_day() + _SOLAR_NOON_HOUR - 0.5 * day_length - _TEMPERATURE_PEAK_HOUR))
    lower = torch.sin(angular * (_SOLAR_NOON_HOUR + 0.5 * day_length - _TEMPERATURE_PEAK_HOUR))
    night_cosine_mean = (upper - lower) / (angular * safe_night)
    night_cosine_mean = torch.where(has_night, night_cosine_mean, torch.zeros_like(night_cosine_mean))
    night = t_mean + amplitude * night_cosine_mean
    day, night = torch.broadcast_tensors(day, night.unsqueeze(-1))
    return torch.cat((day[..., :support.node_count], night[..., :1]), dim=-1)


def tetens_constants() -> Dict[str, float]:
    """Tetens constants of the physical-constant SSOT (as LeafEnergyBalance and PM read them)."""
    return {
        "tetens_base_pressure": float(get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_base_pressure")),
        "tetens_coeff_a": float(get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_a")),
        "tetens_coeff_b": float(get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_b")),
        "kelvin_offset": float(get_physical_constant("celsius_to_kelvin")),
    }


def vapor_pressure_deficit_samples(temperature_samples: Tensor, actual_vapor_pressure: Tensor) -> Tuple[Tensor, Tensor]:
    # References: Allen et al. (1998) FAO-56 Eqs. 11 and 13 (Tetens e0(T) of the shared
    #   psychrometrics helper); constant actual vapour pressure through the day.
    # Mathematical expression: D_k = max(e0(T_k) - e_a, 0); floor_k = (e0(T_k) < e_a).
    # Parameter meanings: T_k the sample air temperature; e_a the daily actual vapour
    #   pressure (e.g. e0 of the daily dewpoint), held constant over the day.
    # Units: T [K]; e0, e_a, D [Pa].
    # Domain/constraints: e_a finite and >= 0; the floor is flagged, not silent
    #   (supersaturation of the course at cool instants, e.g. near the minimum).
    # Sign convention: D >= 0 is the saturation deficit of the air.
    vapor = actual_vapor_pressure
    if not bool(torch.isfinite(vapor).all()) or bool(torch.any(vapor < 0.0)):
        raise ValueError("actual_vapor_pressure must be finite and non-negative")
    saturation = compute_saturation_vapor_pressure(temperature_samples, **tetens_constants())
    deficit = saturation - vapor.to(dtype=saturation.dtype).unsqueeze(-1)
    floor = deficit < 0.0
    return torch.where(floor, torch.zeros_like(deficit), deficit), floor


# Inputs of the support itself; they never enter a law as constants.
SUPPORT_INPUTS = ("latitude", "day_of_year")
TEMPERATURE_INPUTS = ("air_temperature", "air_temperature_max", "air_temperature_min")
VAPOR_INPUT = "actual_vapor_pressure"


@dataclass(frozen=True)
class DaylightSamples:
    """Law inputs on the sample axis, the support, and the VPD floor mask (or None)."""

    state: Dict[str, Tensor]
    support: DaylightSupport
    vapor_pressure_floor: Optional[Tensor]


def daylight_sample_state(values: Mapping[str, Tensor], sample_inputs: Mapping[str, Tensor], *,
                          node_count: int, shortwave_shape: str, flux_keys: Tuple[str, ...] = (),
                          temperature: bool = False, vapor_pressure_deficit: bool = False,
                          dtype: Optional[torch.dtype] = None) -> DaylightSamples:
    # References: module docstring; daylight_support, daylight_flux_samples,
    #   diurnal_air_temperature_samples and vapor_pressure_deficit_samples.
    # Mathematical expression: x_k = x for daily-constant inputs; F_k from F_mean for the
    #   shortwave-proportional fluxes of flux_keys; T_k from (T_av, T_mx, T_mn) when
    #   temperature is set; D_k = max(e0(T_k) - e_a, 0) when vapor_pressure_deficit is set;
    #   inputs of sample_inputs (already on the sample axis, e.g. an upstream
    #   *_daylight_samples port) are taken as they are under their law name.
    # Parameter meanings: values the cell-support inputs of one call (latitude,
    #   day_of_year and, as selected, the temperature and vapour inputs); S the
    #   broadcast cell shape; K = n + 1 the sample count.
    # Units: unchanged per input.
    # Domain/constraints: every input finite (checked by the laws and the builders);
    #   sample inputs need a trailing axis of length K whose leading shape broadcasts to S.
    # Sign convention: as the inputs.
    for key in SUPPORT_INPUTS:
        if values.get(key) is None:
            raise ValueError(f"daylight support requires the declared input {key!r}")
    if dtype is None:
        floating = [value.dtype for value in values.values() if isinstance(value, Tensor) and value.is_floating_point()]
        dtype = floating[0] if floating else torch.get_default_dtype()
        for other in floating[1:]:
            dtype = torch.promote_types(dtype, other)
    count = node_count + 1
    tensors = {key: (value if isinstance(value, Tensor) else torch.as_tensor(value)).to(dtype=dtype)
               for key, value in values.items()}
    samples = {key: value.to(dtype=dtype) for key, value in sample_inputs.items()}
    for key, value in samples.items():
        if value.ndim == 0 or value.shape[-1] != count:
            raise ValueError(f"{key} needs a trailing daylight sample axis of length {count}")
    shape = torch.broadcast_shapes(*(tuple(value.shape) for value in tensors.values()),
                                   *(tuple(value.shape[:-1]) for value in samples.values()))
    device = next(iter(tensors.values())).device
    support = daylight_support(tensors["latitude"].expand(shape), tensors["day_of_year"].expand(shape),
                               node_count, shortwave_shape, dtype=dtype)
    law_state: Dict[str, Tensor] = {}
    consumed = set(SUPPORT_INPUTS)
    for key in flux_keys:
        law_state[key] = daylight_flux_samples(tensors[key].expand(shape), support)
        consumed.add(key)
    floor: Optional[Tensor] = None
    if temperature or vapor_pressure_deficit:
        temperature_samples = diurnal_air_temperature_samples(
            *(tensors[key].expand(shape) for key in TEMPERATURE_INPUTS), support)
        consumed.update(TEMPERATURE_INPUTS)
        if temperature:
            law_state["air_temperature"] = temperature_samples
        if vapor_pressure_deficit:
            law_state["vapor_pressure_deficit"], floor = vapor_pressure_deficit_samples(
                temperature_samples, tensors[VAPOR_INPUT].expand(shape))
            consumed.add(VAPOR_INPUT)
    for key, value in tensors.items():
        if key not in consumed:
            law_state[key] = support.constant(value.to(device=device).expand(shape))
    for key, value in samples.items():
        law_state[key] = value.expand(*shape, count)
    return DaylightSamples(state=law_state, support=support, vapor_pressure_floor=floor)


def daily_outputs(sample_outputs: Mapping[str, Any], support: DaylightSupport) -> Dict[str, Any]:
    # References: module docstring (sample weights).
    # Mathematical expression: floating outputs F = sum_k W_k f_k; boolean diagnostics
    #   b = OR_k b_k (true if the branch is active at any sample); UpdateItem envelopes
    #   keep mode, unit, per-timestep flag and source and carry the daily mean.
    # Parameter meanings: f_k, b_k the outputs of the instantaneous law at sample k.
    # Units: unchanged.
    # Domain/constraints: every tensor output carries the trailing sample axis; linear
    #   identities between outputs (net = gross - respiration, totals = sums) are
    #   preserved by the weighted mean.
    # Sign convention: as the law.
    result: Dict[str, Any] = {}
    for key, value in sample_outputs.items():
        if isinstance(value, UpdateItem):
            result[key] = UpdateItem(
                value=_daily_value(value.value, support), mode=value.mode, write_mode=value.write_mode,
                unit=value.unit, basis=value.basis, per_timestep=value.per_timestep,
                element=value.element, source_functor=value.source_functor)
        else:
            result[key] = _daily_value(value, support)
    return result


def _daily_value(value: Any, support: DaylightSupport) -> Any:
    if not isinstance(value, Tensor):
        return value
    if value.dtype == torch.bool:
        return value.any(dim=-1)
    if value.is_floating_point():
        return support.daily_mean(value)
    raise TypeError(f"daylight daily mean needs a floating or boolean tensor, got {value.dtype}")


__all__ = [
    "DAYLIGHT_GAUSS",
    "DEFAULT_SHORTWAVE_SHAPE",
    "DaylightSamples",
    "DaylightSupport",
    "NODE_COUNTS",
    "NODE_COUNT_KEY",
    "SHORTWAVE_SHAPES",
    "SHORTWAVE_SHAPE_KEY",
    "daily_outputs",
    "daylight_config",
    "daylight_flux_samples",
    "daylight_sample_state",
    "daylight_support",
    "diurnal_air_temperature_samples",
    "gauss_legendre_rule",
    "reject_daylight_keys",
    "require_daily_step",
    "tetens_constants",
    "vapor_pressure_deficit_samples",
]
