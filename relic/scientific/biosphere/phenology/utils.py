"""
Phenology Utility Functions for RELIC Framework

This module provides utility functions for phenological calculations including
GDD computation, GSI calculations, photoperiod calculations, and various
environmental response functions. All functions support PyTorch tensors
for automatic differentiation and GPU acceleration.

Mathematical Foundations:
    - Growing Degree Days: GDD = Sigma max(0, (T_max + T_min)/2 - T_base)
    - Growing Season Index: GSI = f_temp * f_vpd * f_daylen * f_swp
    - Photoperiod: Based on astronomical calculations and latitude
    - Environmental response functions: Various sigmoid and threshold functions

References:
    - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
      foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
    - Campbell, G.S. & Norman, J.M. (1998). An Introduction to Environmental Biophysics.
    - Jones, H.G. (2013). Plants and Microclimate: A Quantitative Approach to 
      Environmental Plant Physiology.
"""

from typing import Union, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F
from torch import Tensor
from datetime import datetime, timedelta
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.parameters import get_global_manager
from relic.utils import time_constants

# Type definitions
TensorLike = Union[Tensor, float, int]


def _read_parameter(key: str, units: Optional[str] = None):
    """
    Read a required parameter through UnifiedParameterManager.
    """
    return get_global_manager().get(
        key,
        units=units,
        required=True,
        expected_type=None,
    )


def compute_gdd(
    temperature_max: Tensor,
    temperature_min: Tensor, 
    base_temperature: Optional[float] = None,
    max_temperature: Optional[float] = None,
    method: str = "simple"
) -> Tensor:
    """
    Compute Growing Degree Days (GDD) for phenological modeling.
    
    Growing Degree Days represent the accumulated thermal energy available
    for plant growth above a base temperature threshold. Multiple calculation
    methods are supported to accommodate different modeling approaches.
    
    Parameters:
        temperature_max : Tensor
            Daily maximum temperature [degC]
            Shape: [batch_size] or [batch_size, spatial_dims]
        temperature_min : Tensor
            Daily minimum temperature [degC] 
            Shape: same as temperature_max
        base_temperature : Optional[float]
            Base temperature threshold for GDD calculation [degC]
            If None, the parameter database value is used
        max_temperature : Optional[float]
            Maximum temperature cap for GDD calculation [degC]
            If None, the parameter database value is used for modified method
        method : str
            Calculation method: "simple", "modified", "triangular"
            Default: "simple"
            
    Returns:
        gdd : Tensor
            Daily growing degree days [degC*day]
            Shape: same as input temperatures
            
    Mathematical Expressions:
        Simple method: GDD = max(0, (T_max + T_min)/2 - T_base)
        Modified method: Applies upper temperature threshold
        Triangular method: Assumes triangular temperature distribution
        
    References:
        - McMaster, G.S. & Wilhelm, W.W. (1997). Growing degree-days: one equation, 
          two interpretations. Agricultural and Forest Meteorology, 87(4), 291-300.
        - Cesaraccio, C., et al. (2001). An improved model for determining degree-day 
          values from daily temperature data. International Journal of Biometeorology, 45(4), 161-169.
    """
    # Input validation
    if not isinstance(temperature_max, torch.Tensor):
        temperature_max = torch.as_tensor(temperature_max, dtype=torch.float32)
    if not isinstance(temperature_min, torch.Tensor):
        temperature_min = torch.as_tensor(temperature_min, dtype=torch.float32)
    temperature_max = temperature_max.to(dtype=torch.float32)
    temperature_min = temperature_min.to(dtype=torch.float32)
        
    # Ensure tensors have the same shape
    if temperature_max.shape != temperature_min.shape:
        raise ValueError("temperature_max and temperature_min must have the same shape")
    
    if base_temperature is None:
        base_temperature = _read_parameter(
            "biosphere.phenology.gdd.base_temp",
            units="degC",
        )

    if method == "simple":
        # Simple average method
        t_avg = (temperature_max + temperature_min) / 2.0
        gdd = torch.clamp(t_avg - base_temperature, min=0.0)
        
    elif method == "modified":
        # Modified method with upper temperature limit
        if max_temperature is None:
            max_temperature = _read_parameter(
                "biosphere.phenology.gdd.max_temp_limit",
                units="degC",
            )
            
        # Apply temperature limits
        t_max_limited = torch.clamp(temperature_max, max=max_temperature)
        t_min_limited = torch.clamp(temperature_min, min=base_temperature)
        
        # Calculate GDD only when conditions are met
        t_avg = (t_max_limited + t_min_limited) / 2.0
        gdd = torch.clamp(t_avg - base_temperature, min=0.0)
        
    elif method == "triangular":
        # Triangular distribution method (more accurate for asymmetric temperature curves)
        t_max_adj = torch.clamp(temperature_max, min=base_temperature)
        t_min_adj = torch.clamp(temperature_min, min=base_temperature)
        
        # When both temperatures are above base
        both_above = (temperature_min >= base_temperature)
        
        # When only max is above base
        max_only = (temperature_max >= base_temperature) & (temperature_min < base_temperature)
        
        # Calculate GDD for each case
        gdd = torch.zeros_like(temperature_max)
        
        # Both above base temperature
        gdd[both_above] = (t_max_adj[both_above] + t_min_adj[both_above]) / 2.0 - base_temperature
        
        # Only max above base temperature (triangular calculation)
        gdd[max_only] = (t_max_adj[max_only] - base_temperature) ** 2 / (2 * (t_max_adj[max_only] - temperature_min[max_only]))
        
    else:
        raise ValueError(f"Unknown GDD calculation method: {method}")
    
    return gdd.to(dtype=temperature_max.dtype)


def compute_gsi_components(
    temperature_min: Tensor,
    vapor_pressure_deficit: Tensor,
    daylength: Tensor,
    tmin_threshold: Optional[float] = None,
    tmin_optimal: Optional[float] = None,
    vpd_min: Optional[float] = None,
    vpd_max: Optional[float] = None,
    daylen_threshold: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute individual components of Growing Season Index (GSI).
    
    GSI is a composite environmental suitability index based on the
    multiplicative limitation principle from Jolly et al. (2005).
    Each component represents limitation by a different environmental factor.
    
    Parameters:
        temperature_min : Tensor
            Daily minimum temperature [degC]
        vapor_pressure_deficit : Tensor  
            Daily vapor pressure deficit [Pa]
        daylength : Tensor
            Day length duration [seconds]
        tmin_threshold : Optional[float]
            Minimum temperature threshold [degC]
            If None, the parameter database value is used
        tmin_optimal : Optional[float]
            Optimal minimum temperature [degC] 
            If None, the parameter database value is used
        vpd_min : Optional[float]
            Minimum VPD for optimal conditions [Pa]
            If None, the parameter database value is used
        vpd_max : Optional[float]
            Maximum VPD threshold [Pa]
            If None, the parameter database value is used
        daylen_threshold : Optional[float]
            Daylight length threshold [seconds]
            If None, the parameter database value is used
    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            f_temp, f_vpd, f_daylen - Individual limitation factors [dimensionless, 0-1]
            
    Mathematical Expressions:
        Temperature limitation: f_temp = (T_min - T_min_thresh) / (T_min_opt - T_min_thresh)
        VPD limitation: f_vpd = (VPD_max - VPD) / (VPD_max - VPD_min) 
        Daylight limitation: f_daylen = (DL - DL_min) / (DL_thresh - DL_min)
        
        All factors are bounded between 0 and 1.
        
    References:
        - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
          foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
    """
    # Get parameter from parameter manager if not provided
    if tmin_threshold is None:
        tmin_threshold = _read_parameter(
            "biosphere.phenology.gsi.tmin_threshold",
            units="degC",
        )
    if tmin_optimal is None:
        tmin_optimal = _read_parameter(
            "biosphere.phenology.gsi.tmin_optimal",
            units="degC",
        )
    if vpd_min is None:
        vpd_min = _read_parameter(
            "biosphere.phenology.gsi.vpd_min",
            units="Pa",
        )
    if vpd_max is None:
        vpd_max = _read_parameter(
            "biosphere.phenology.gsi.vpd_max",
            units="Pa",
        )
    if daylen_threshold is None:
        daylen_threshold = _read_parameter(
            "biosphere.phenology.gsi.daylen_threshold",
            units="s",
        )
    # References: Jolly et al. (2005), doi:10.1111/j.1365-2486.2005.00930.x.
    # Mathematical expressions: f_T=(T_min-T_low)/(T_high-T_low),
    # f_VPD=(VPD_high-VPD)/(VPD_high-VPD_low), and
    # f_day=(daylength-daylength_low)/(daylength_high-daylength_low).
    # Parameter meanings: low/high values are ParameterManager-backed climatic
    # response thresholds; each factor is clipped to the closed interval [0, 1].
    # Units: T_min [degC], VPD [Pa], daylength [s], factors [dimensionless].
    # Open route: observed climate constrains phenology without a conserved flux.

    # Temperature limitation factor
    f_temp = torch.clamp(
        (temperature_min - tmin_threshold) / (tmin_optimal - tmin_threshold),
        min=0.0, max=1.0
    )
    
    # VPD limitation factor (inverse relationship - higher VPD is more stressful)
    f_vpd = torch.clamp(
        (vpd_max - vapor_pressure_deficit) / (vpd_max - vpd_min),
        min=0.0, max=1.0
    )
    
    # Daylight limitation factor (minimum enforced by parameter database)
    daylen_min = _read_parameter(
        "biosphere.phenology.constants.daylen_min",
    )
    f_daylen = torch.clamp(
        (daylength - daylen_min) / (daylen_threshold - daylen_min),
        min=0.0, max=1.0
    )
    
    target_dtype = torch.float32
    return (
        f_temp.to(dtype=target_dtype),
        f_vpd.to(dtype=target_dtype),
        f_daylen.to(dtype=target_dtype),
    )


def compute_gsi(
    temperature_min: Tensor,
    vapor_pressure_deficit: Tensor,
    daylength: Tensor,
    tmin_threshold: Optional[float] = None,
    tmin_optimal: Optional[float] = None,
    vpd_min: Optional[float] = None,
    vpd_max: Optional[float] = None,
    daylen_threshold: Optional[float] = None,
) -> Tensor:
    """
    Compute Growing Season Index (GSI) as composite environmental suitability.
    
    GSI combines multiple environmental limitation factors using multiplicative
    integration, following the principle that growth is limited by the most
    restrictive factor.
    
    Parameters:
        temperature_min : Tensor
            Daily minimum temperature [degC]
        vapor_pressure_deficit : Tensor
            Daily vapor pressure deficit [Pa] 
        daylength : Tensor
            Day length duration [seconds]
        tmin_threshold, tmin_optimal, vpd_min, vpd_max, daylen_threshold:
            Optional climatic thresholds. ParameterManager values are used when
            a threshold is not supplied.

    Returns:
        gsi : Tensor
            Growing Season Index [dimensionless, 0-1]
            
    Mathematical Expression:
        GSI = f_temp * f_vpd * f_daylen
        
        Where each factor ranges from 0 (completely limiting) to 1 (optimal).
        The multiplicative form ensures that any single strongly limiting 
        factor reduces the overall GSI.

    References:
        - Jolly, W.M., et al. (2005). Global Change Biology, 11(4), 619-632.
    """
    f_temp, f_vpd, f_daylen = compute_gsi_components(
        temperature_min,
        vapor_pressure_deficit,
        daylength,
        tmin_threshold=tmin_threshold,
        tmin_optimal=tmin_optimal,
        vpd_min=vpd_min,
        vpd_max=vpd_max,
        daylen_threshold=daylen_threshold,
    )

    # References: Jolly et al. (2005), doi:10.1111/j.1365-2486.2005.00930.x.
    # Mathematical expression: iGSI = f_T * f_VPD * f_daylength.
    # Parameter meanings: the three factors are climatic limitation scalars.
    # Units: every factor and iGSI are dimensionless.
    # Open route: iGSI is a diagnostic control and carries no material flux.
    gsi = f_temp * f_vpd * f_daylen
    
    return gsi.to(dtype=torch.float32)


def compute_daylength(
    day_of_year: TensorLike,
    latitude: TensorLike,
    *,
    calendar_type: str,
    include_twilight: bool = False,
    twilight_angle: TensorLike = 6.0,
) -> Tensor:
    """
    Compute day length on the explicitly selected simulator calendar.
    
    Calculates the duration of daylight for a given day of year and latitude
    using standard astronomical formulations. Optionally includes civil twilight.
    
    Parameters:
        day_of_year : TensorLike
            Integral simulator day of year in ``[1, days_per_year]``
        latitude : TensorLike
            Latitude in degrees (positive for Northern Hemisphere)
        calendar_type : str
            Simulator calendar. Operational values are ``noleap`` and
            ``360_day``.
        include_twilight : bool
            Whether to include twilight in day length calculation
        twilight_angle : float
            Sun angle below horizon for twilight definition [degrees]
            Default: 6.0 (civil twilight)
            
    Returns:
        daylength : Tensor
            Day length in seconds
            
    Mathematical Expression:
        Solar declination: delta = 0.409 sin(2 pi J / Y(C) - 1.39)
        Sunset hour angle:
            omega_s = arccos((sin(h_0) - sin(phi) sin(delta)) /
                             (cos(phi) cos(delta)))
        Day length: N = seconds_per_day * omega_s / pi

        Here J is day of year, Y(C) is the selected calendar year length,
        phi is latitude, delta is solar declination, and h_0 is zero for
        geometric daylight or a negative twilight depression angle.
        
    References:
        - Allen, R.G., Pereira, L.S., Raes, D. & Smith, M. (1998).
          FAO Irrigation and Drainage Paper 56, equations 24-25 and 34.
        - CF Conventions 1.8, Section 4.4.
    """
    days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)
    if not isinstance(include_twilight, bool):
        raise TypeError("include_twilight must be a bool.")

    tensor_inputs = [
        value for value in (day_of_year, latitude) if isinstance(value, Tensor)
    ]
    devices = {value.device for value in tensor_inputs}
    if len(devices) > 1:
        raise ValueError("day_of_year and latitude must use the same device.")
    device = tensor_inputs[0].device if tensor_inputs else torch.device("cpu")

    day_raw = torch.as_tensor(day_of_year, device=device)
    latitude_raw = torch.as_tensor(latitude, device=device)
    for name, value in (("day_of_year", day_raw), ("latitude", latitude_raw)):
        if value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"{name} must be a real numeric value.")

    floating_dtypes = [
        value.dtype
        for value in (day_raw, latitude_raw)
        if value.is_floating_point()
    ]
    dtype = floating_dtypes[0] if floating_dtypes else torch.get_default_dtype()
    for value_dtype in floating_dtypes[1:]:
        dtype = torch.promote_types(dtype, value_dtype)

    day = day_raw.to(dtype=dtype)
    latitude_degrees = latitude_raw.to(dtype=dtype)
    day, latitude_degrees = torch.broadcast_tensors(day, latitude_degrees)

    if not torch.all(torch.isfinite(day)).item():
        raise ValueError("day_of_year must be finite.")
    if not torch.equal(day, torch.round(day)):
        raise ValueError("day_of_year must contain integral values.")
    if torch.any((day < 1) | (day > days_per_year)).item():
        raise ValueError(
            f"day_of_year must be in [1, {days_per_year}] for {calendar_type}."
        )
    if not torch.all(torch.isfinite(latitude_degrees)).item():
        raise ValueError("latitude must be finite.")
    if torch.any((latitude_degrees < -90) | (latitude_degrees > 90)).item():
        raise ValueError("latitude must be in [-90, 90] degrees.")

    twilight_raw = torch.as_tensor(twilight_angle, device=device)
    if twilight_raw.dtype == torch.bool or twilight_raw.is_complex():
        raise TypeError("twilight_angle must be a real numeric value.")
    if twilight_raw.numel() != 1:
        raise ValueError("twilight_angle must be scalar.")
    twilight = twilight_raw.to(dtype=dtype)
    if not torch.isfinite(twilight).item():
        raise ValueError("twilight_angle must be finite.")
    if torch.any((twilight < 0) | (twilight > 18)).item():
        raise ValueError("twilight_angle must be in [0, 18] degrees.")

    # Reference: Allen et al. (1998), FAO-56 equations 24-25 and 34.
    # Mathematical expressions: delta = 0.409 sin(2 pi J / Y(C) - 1.39),
    # cos(omega_s) = [sin(h_0) - sin(phi) sin(delta)] /
    #                [cos(phi) cos(delta)], and N = t_day omega_s / pi.
    # Parameter meanings: J is simulator day of year, Y(C) is the exact
    # calendar year length, phi is latitude, h_0 is the horizon angle,
    # omega_s is the sunset hour angle, and t_day is seconds per day.
    # Units: delta, phi, h_0, and omega_s are rad; N is s.
    pi = torch.acos(day.new_tensor(-1.0))
    year_length = day.new_tensor(float(days_per_year))
    declination = day.new_tensor(0.409) * torch.sin(
        2 * pi * day / year_length - day.new_tensor(1.39)
    )
    latitude_radians = torch.deg2rad(latitude_degrees)
    horizon_angle = (
        -torch.deg2rad(twilight)
        if include_twilight
        else day.new_tensor(0.0)
    )

    numerator = (
        torch.sin(horizon_angle)
        - torch.sin(latitude_radians) * torch.sin(declination)
    )
    denominator = torch.cos(latitude_radians) * torch.cos(declination)
    denominator_is_zero = torch.abs(denominator) <= torch.finfo(dtype).eps
    safe_denominator = torch.where(
        denominator_is_zero,
        torch.ones_like(denominator),
        denominator,
    )
    cos_hour_angle = numerator / safe_denominator
    polar_limit = torch.where(
        numerator <= 0,
        -torch.ones_like(numerator),
        torch.ones_like(numerator),
    )
    cos_hour_angle = torch.where(
        denominator_is_zero,
        polar_limit,
        cos_hour_angle,
    )
    hour_angle = torch.acos(torch.clamp(cos_hour_angle, min=-1.0, max=1.0))
    seconds_per_day = day.new_tensor(float(time_constants.SECONDS_PER_DAY))
    return seconds_per_day * hour_angle / pi


def compute_photoperiod_factor(
    day_of_year: TensorLike,
    latitude: TensorLike,
    *,
    calendar_type: str,
    critical_daylength: TensorLike = 12.0,
    sensitivity: TensorLike = 1.0,
) -> Tensor:
    """
    Compute photoperiod response factor for phenological processes.
    
    Many plant processes are sensitive to photoperiod (day length) for timing
    of seasonal events like flowering and senescence. This function computes
    a smooth photoperiod response factor.
    
    Parameters:
        day_of_year : TensorLike
            Integral simulator day of year
        latitude : TensorLike
            Latitude in degrees
        calendar_type : str
            Simulator calendar: ``noleap`` or ``360_day``
        critical_daylength : float
            Critical day length threshold [hours]
        sensitivity : float
            Photoperiod sensitivity parameter [dimensionless]
            
    Returns:
        photoperiod_factor : Tensor
            Photoperiod response factor [dimensionless, 0-1]
            
    Mathematical Expression:
        Photoperiod factor = 1 / (1 + exp(-sensitivity * (daylength - critical)))
        
        This creates a sigmoid response around the critical day length.

    References:
        - Campbell, G.S. & Norman, J.M. (1998). An Introduction to Environmental Biophysics.
    """
    daylength_seconds = compute_daylength(
        day_of_year,
        latitude,
        calendar_type=calendar_type,
    )
    response_values = {}
    for name, value in (
        ("critical_daylength", critical_daylength),
        ("sensitivity", sensitivity),
    ):
        if isinstance(value, Tensor) and value.device != daylength_seconds.device:
            raise ValueError(f"{name} must use the daylength device.")
        raw_value = torch.as_tensor(value, device=daylength_seconds.device)
        if raw_value.dtype == torch.bool or raw_value.is_complex():
            raise TypeError(f"{name} must be a real numeric value.")
        response_value = raw_value.to(dtype=daylength_seconds.dtype)
        if not torch.all(torch.isfinite(response_value)).item():
            raise ValueError(f"{name} must be finite.")
        response_values[name] = response_value

    critical = response_values["critical_daylength"]
    if torch.any((critical < 0) | (critical > 24)).item():
        raise ValueError("critical_daylength must be in [0, 24] hours.")

    # Reference: Campbell and Norman (1998), photoperiod response framework.
    # Mathematical expression: f_P = sigmoid[k_P (N_h - N_crit)].
    # Parameter meanings: N_h is day length in hours, N_crit is the critical
    # day length, and k_P is the dimensionless photoperiod sensitivity.
    # Units: N_h and N_crit are h; f_P and k_P are dimensionless.
    seconds_per_hour = daylength_seconds.new_tensor(
        float(time_constants.SECONDS_PER_HOUR)
    )
    daylength_hours = daylength_seconds / seconds_per_hour
    return torch.sigmoid(
        response_values["sensitivity"] * (daylength_hours - critical)
    )


def julian_day_to_date(year: int, day_of_year: int) -> datetime:
    """
    Convert Julian day of year to calendar date.
    
    Parameters:
        year : int
            Calendar year
        day_of_year : int
            Julian day of year (1-366)
            
    Returns:
        datetime: Calendar date
        
    Raises:
        ValueError: If day_of_year is out of valid range
    """
    if not 1 <= day_of_year <= 366:
        raise ValueError(f"Day of year must be between 1-366, got {day_of_year}")
        
    # Handle leap years
    if day_of_year == 366:
        if not ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
            raise ValueError(f"Day 366 is only valid in leap years, {year} is not a leap year")
    
    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)


def date_to_julian_day(date: datetime) -> int:
    """
    Convert calendar date to Julian day of year.
    
    Parameters:
        date : datetime
            Calendar date
            
    Returns:
        int: Julian day of year (1-366)
    """
    return date.timetuple().tm_yday


def rolling_average(
    data: Tensor, 
    window_size: int, 
    dim: int = -1,
    mode: str = "constant",
    constant_value: float = 0.0
) -> Tensor:
    """
    Compute rolling/moving average of tensor data.
    
    Useful for smoothing environmental data and computing GSI memory effects.
    
    Parameters:
        data : Tensor
            Input data tensor
        window_size : int
            Size of the moving average window
        dim : int
            Dimension along which to compute rolling average
        mode : str
            Padding mode for edges: "constant", "reflect", "replicate"
        constant_value : float
            Value for constant padding
            
    Returns:
        Tensor: Rolling average with same shape as input
        
    Mathematical Expression:
        Rolling average: y[i] = (1/w) * Sigma(j=i-w+1 to i) x[j]
        where w is window_size
    """
    data = data.to(dtype=torch.float32)
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    if window_size == 1:
        return data
        
    # Create convolution kernel for moving average
    kernel = torch.ones(window_size, dtype=data.dtype, device=data.device) / window_size
    
    # Handle multidimensional tensors
    if data.dim() == 1:
        # Pad the data to handle edges
        if mode == "constant":
            padded_data = F.pad(data, (window_size//2, window_size//2), mode=mode, value=constant_value)
        else:
            padded_data = F.pad(data, (window_size//2, window_size//2), mode=mode)
            
        # Apply convolution
        result = F.conv1d(padded_data.unsqueeze(0).unsqueeze(0), 
                         kernel.unsqueeze(0).unsqueeze(0), 
                         padding=0).squeeze()
    else:
        # For higher dimensional tensors, apply along specified dimension
        # Handle arbitrary dimensions by permuting tensor
        if dim < 0:
            dim = data.dim() + dim
        
        # Permute tensor so target dimension is last
        perm = list(range(data.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        data_permuted = data.permute(*perm)
        
        # Apply rolling average along last dimension
        original_shape = data_permuted.shape
        data_reshaped = data_permuted.reshape(-1, original_shape[-1])
        
        results = []
        for i in range(data_reshaped.shape[0]):
            row_result = rolling_average(data_reshaped[i], window_size, dim=0, mode=mode, constant_value=constant_value)
            results.append(row_result)
        
        result = torch.stack(results).reshape(original_shape)
        
        # Permute back to original dimension order
        result = result.permute(*perm)
    
    return result.to(dtype=data.dtype)


def thermal_time_accumulation(
    temperature: Tensor,
    base_temp: float,
    optimal_temp: float,
    max_temp: float,
    method: str = "linear"
) -> Tensor:
    """
    Compute thermal time accumulation with optimal temperature response.
    
    Thermal time models account for the fact that plant development rates
    are not linear with temperature but show optimal responses.
    
    Parameters:
        temperature : Tensor
            Daily average temperature [degC]
        base_temp : float
            Base temperature threshold [degC]
        optimal_temp : float
            Optimal temperature for development [degC]
        max_temp : float
            Maximum temperature above which development ceases [degC]
        method : str
            Thermal time method: "linear", "triangular", "beta"
            
    Returns:
        thermal_time : Tensor
            Daily thermal time units [degC*day]
            
    Mathematical Expressions:
        Linear: TT = max(0, min(T - T_base, T_opt - T_base))
        Triangular: Piecewise linear with optimum
        Beta function: Smooth optimum curve

    References:
        - McMaster, G.S. & Wilhelm, W.W. (1997). Agricultural and Forest Meteorology, 87(4), 291-300.
    """
    temperature = torch.as_tensor(temperature, dtype=torch.float32)
    if method == "linear":
        # Simple linear relationship
        thermal_time = torch.clamp(temperature - base_temp, min=0.0)
        
    elif method == "triangular":
        # Triangular response with optimum
        thermal_time = torch.zeros_like(temperature)
        
        # Below optimal temperature
        below_opt = (temperature >= base_temp) & (temperature <= optimal_temp)
        thermal_time[below_opt] = temperature[below_opt] - base_temp
        
        # Above optimal temperature
        above_opt = (temperature > optimal_temp) & (temperature <= max_temp)
        thermal_time[above_opt] = (max_temp - temperature[above_opt]) / (max_temp - optimal_temp) * (optimal_temp - base_temp)
        
    elif method == "beta":
        # Beta function response (smooth optimum)
        # Normalize temperature to [0, 1]
        temp_norm = torch.clamp((temperature - base_temp) / (max_temp - base_temp), min=0.0, max=1.0)
        
        # Beta function parameters (can be adjusted)
        alpha = _read_parameter(
            "biosphere.phenology.constants.alpha_beta",
        )
        beta = alpha  # Symmetric Beta distribution
        
        # Compute beta response
        beta_response = (temp_norm ** (alpha - 1)) * ((1 - temp_norm) ** (beta - 1))
        
        # Scale to thermal time units
        max_thermal_time = optimal_temp - base_temp
        thermal_time = beta_response * max_thermal_time
        
    else:
        raise ValueError(f"Unknown thermal time method: {method}")
    
    return thermal_time.to(dtype=torch.float32)


def phenological_threshold_response(
    environmental_value: Tensor,
    threshold: float,
    sensitivity: float = 1.0,
    response_type: str = "sigmoid"
) -> Tensor:
    """
    Compute phenological response to environmental threshold.
    
    Many phenological processes are triggered when environmental variables
    cross certain thresholds. This function provides various response shapes.
    
    Parameters:
        environmental_value : Tensor
            Environmental variable value
        threshold : float
            Threshold value for response
        sensitivity : float
            Steepness of response curve
        response_type : str
            Type of response: "sigmoid", "step", "linear", "exponential"
            
    Returns:
        response : Tensor
            Response factor [0-1]
            
    Mathematical Expressions:
        Sigmoid: 1/(1 + exp(-sensitivity * (value - threshold)))
        Step: 1 if value > threshold else 0
        Linear: max(0, min(1, sensitivity * (value - threshold)))
        Exponential: 1 - exp(-sensitivity * max(0, value - threshold))

    References:
        - Jones, H.G. (2013). Plants and Microclimate: A Quantitative Approach.
    """
    if response_type == "sigmoid":
        response = torch.sigmoid(sensitivity * (environmental_value - threshold))
        
    elif response_type == "step":
        response = (environmental_value > threshold).float()
        
    elif response_type == "linear":
        response = torch.clamp(sensitivity * (environmental_value - threshold), min=0.0, max=1.0)
        
    elif response_type == "exponential":
        response = 1.0 - torch.exp(-sensitivity * torch.clamp(environmental_value - threshold, min=0.0))
        
    else:
        raise ValueError(f"Unknown response type: {response_type}")
    
    return response


def compute_chilling_hours(
    temperature: Tensor,
    chilling_threshold: Optional[float] = None,
    method: str = "simple"
) -> Tensor:
    """
    Compute chilling hours for vernalization and dormancy breaking.
    
    Many plants require exposure to cold temperatures (chilling) to break
    dormancy or fulfill vernalization requirements.
    
    Parameters:
        temperature : Tensor
            Hourly or daily temperature [degC]
        chilling_threshold : Optional[float]
            Temperature threshold for chilling [degC]
            If None, the parameter database value is used
        method : str
            Chilling model: "simple", "utah", "dynamic"
            
    Returns:
        chilling : Tensor
            Chilling units accumulated

    Mathematical Expressions:
        Simple: CH = 1 if T <= T_thresh else 0
        Dynamic (simplified): CH = exp(-0.5 * ((T - T_opt) / sigma) ** 2)
        with T_opt = 6.0 and sigma = 4.0
            
    References:
        - Richardson, E.A., et al. (1974). A model for estimating the completion 
          of rest for Red Haven and Elberta peach trees. HortScience, 9(4), 331-332.
    """
    if chilling_threshold is None:
        chilling_threshold = _read_parameter(
            "biosphere.phenology.constants.chilling_threshold",
            units="degC",
        )
    if method == "simple":
        # Simple threshold model
        chilling = (temperature <= chilling_threshold).float()
        
    elif method == "utah":
        # Utah chilling model
        chilling = torch.zeros_like(temperature)
        # References: Richardson et al. (1974)
        # Formula: piecewise weights by temperature bins for Utah model.
        # Parameters: utah_temp_bin_*_min/max/weight from ParameterManager.
        t1_min = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_1_min",
            units="degC",
        )
        t1_max = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_1_max",
            units="degC",
        )
        w1 = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_1_weight",
            units="dimensionless",
        )
        t2_min = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_2_min",
            units="degC",
        )
        t2_max = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_2_max",
            units="degC",
        )
        w2 = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_2_weight",
            units="dimensionless",
        )
        t3_min = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_3_min",
            units="degC",
        )
        t3_max = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_3_max",
            units="degC",
        )
        w3 = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_3_weight",
            units="dimensionless",
        )
        t4_min = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_4_min",
            units="degC",
        )
        t4_max = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_4_max",
            units="degC",
        )
        w4 = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_4_weight",
            units="dimensionless",
        )
        t5_min = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_5_min",
            units="degC",
        )
        t5_max = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_5_max",
            units="degC",
        )
        w5 = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_5_weight",
            units="dimensionless",
        )
        t6_min = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_6_min",
            units="degC",
        )
        w6 = _read_parameter(
            "biosphere.phenology.constants.utah_temp_bin_6_weight",
            units="dimensionless",
        )

        # Different temperature ranges contribute differently
        mask_1 = (temperature >= t1_min) & (temperature < t1_max)
        chilling[mask_1] = w1

        mask_2 = (temperature >= t2_min) & (temperature < t2_max)
        chilling[mask_2] = w2

        mask_3 = (temperature >= t3_min) & (temperature < t3_max)
        chilling[mask_3] = w3

        mask_4 = (temperature >= t4_min) & (temperature < t4_max)
        chilling[mask_4] = w4

        mask_5 = (temperature >= t5_min) & (temperature < t5_max)
        chilling[mask_5] = w5

        mask_6 = temperature >= t6_min
        chilling[mask_6] = w6
        
    elif method == "dynamic":
        # Dynamic model (simplified version)
        # More complex models exist in literature
        # References: Richardson et al. (1974)
        # Formula: CH = exp(-0.5 * ((T - T_opt) / sigma) ** 2)
        # Parameters: dynamic_optimal_temperature, dynamic_temperature_sigma.
        optimal_temp = _read_parameter(
            "biosphere.phenology.constants.dynamic_optimal_temperature",
            units="degC",
        )
        sigma_temp = _read_parameter(
            "biosphere.phenology.constants.dynamic_temperature_sigma",
            units="degC",
        )
        temp_diff = torch.abs(temperature - optimal_temp)
        chilling = torch.exp(-0.5 * (temp_diff / sigma_temp) ** 2)
        
    else:
        raise ValueError(f"Unknown chilling method: {method}")
    
    return chilling


def validate_environmental_data(
    data: Dict[str, Tensor],
    required_vars: list,
    check_ranges: bool = True
) -> Dict[str, Any]:
    """
    Validate environmental data for phenology calculations.
    
    Parameters:
        data : Dict[str, Tensor]
            Dictionary of environmental variables
        required_vars : list
            List of required variable names
        check_ranges : bool
            Whether to check realistic value ranges
            
    Returns:
        Dict[str, Any]: Validation results with warnings and errors
    """
    results = {
        'valid': True,
        'missing_vars': [],
        'range_warnings': [],
        'errors': []
    }
    
    # Check for required variables
    for var in required_vars:
        if var not in data:
            results['missing_vars'].append(var)
            results['valid'] = False
    
    if not check_ranges:
        return results
    
    # References: Jolly et al. (2005); Campbell and Norman (1998).
    # Formula: if value < min or value > max then record a range warning.
    # Parameters: validation_ranges.*.min/max from the parameter database.
    ranges = {
        'air_temperature': (
            _read_parameter(
                "biosphere.phenology.validation_ranges.air_temperature.min",
                units="degC",
            ),
            _read_parameter(
                "biosphere.phenology.validation_ranges.air_temperature.max",
                units="degC",
            ),
        ),
        'temperature_min': (
            _read_parameter(
                "biosphere.phenology.validation_ranges.temperature_min.min",
                units="degC",
            ),
            _read_parameter(
                "biosphere.phenology.validation_ranges.temperature_min.max",
                units="degC",
            ),
        ),
        'temperature_max': (
            _read_parameter(
                "biosphere.phenology.validation_ranges.temperature_max.min",
                units="degC",
            ),
            _read_parameter(
                "biosphere.phenology.validation_ranges.temperature_max.max",
                units="degC",
            ),
        ),
        'vapor_pressure_deficit': (
            _read_parameter(
                "biosphere.phenology.validation_ranges.vapor_pressure_deficit.min",
                units="Pa",
            ),
            _read_parameter(
                "biosphere.phenology.validation_ranges.vapor_pressure_deficit.max",
                units="Pa",
            ),
        ),
        'soil_water_potential': (
            _read_parameter(
                "biosphere.phenology.validation_ranges.soil_water_potential.min",
                units="MPa",
            ),
            _read_parameter(
                "biosphere.phenology.validation_ranges.soil_water_potential.max",
                units="MPa",
            ),
        ),
        'daylength': (
            _read_parameter(
                "biosphere.phenology.validation_ranges.daylength.min",
                units="s",
            ),
            _read_parameter(
                "biosphere.phenology.validation_ranges.daylength.max",
                units="s",
            ),
        ),
        'day_of_year': (
            _read_parameter(
                "biosphere.phenology.validation_ranges.day_of_year.min",
                units="day",
            ),
            _read_parameter(
                "biosphere.phenology.validation_ranges.day_of_year.max",
                units="day",
            ),
        ),
    }
    
    # Check ranges
    for var, (min_val, max_val) in ranges.items():
        if var in data:
            values = data[var]
            if torch.any(values < min_val) or torch.any(values > max_val):
                results['range_warnings'].append(
                    f"{var} contains values outside realistic range [{min_val}, {max_val}]"
                )
    
    return results


# Export all functions
__all__ = [
    'compute_gdd',
    'compute_gsi_components', 
    'compute_gsi',
    'compute_daylength',
    'compute_photoperiod_factor',
    'julian_day_to_date',
    'date_to_julian_day',
    'rolling_average',
    'thermal_time_accumulation',
    'phenological_threshold_response',
    'compute_chilling_hours',
    'validate_environmental_data'
]



