"""
Response Curve Functions for Phenological Modeling

This module provides a collection of mathematical response functions commonly
used in phenological models. All functions support PyTorch tensors for 
automatic differentiation and GPU acceleration.

Functions include sigmoid, exponential, linear threshold, and other response
types frequently used in ecological modeling.

References:
    - Thornton, P.E. & Running, S.W. (1999). An improved algorithm for estimating 
      incident daily solar radiation from measurements of temperature, humidity, 
      and precipitation. Agricultural and Forest Meteorology, 93(4), 211-228.
    - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
      foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
"""

from typing import Union
import torch
from torch import Tensor

from relic.parameters import get_global_manager

TensorLike = Union[Tensor, float, int]


def _read_parameter(key: str) -> float:
    """Read a required scalar parameter for phenology response curves."""
    return float(get_global_manager().get(
        key,
        expected_type=(float, int),
    ))


def sigmoid_response(
    x: Tensor,
    x_center: float,
    steepness: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0
) -> Tensor:
    """
    Sigmoid response function for smooth transitions.
    
    Commonly used for phenological transitions like leaf onset/offset
    where gradual changes are more realistic than abrupt switches.
    
    Parameters:
        x : Tensor
            Input variable (e.g., temperature, GDD, day of year)
        x_center : float
            Inflection point where y = (y_max + y_min) / 2
        steepness : float
            Controls transition steepness (higher = steeper)
            Default: 1.0
        y_min : float
            Minimum output value, default: 0.0
        y_max : float
            Maximum output value, default: 1.0
            
    Returns:
        Tensor: Sigmoid response values between y_min and y_max
        
    Mathematical Expression:
        y = y_min + (y_max - y_min) * [1 / (1 + exp(-steepness * (x - x_center)))]
        
    Examples:
        >>> # Temperature response for GSI calculation
        >>> temp = torch.linspace(-10, 20, 100)
        >>> response = sigmoid_response(temp, x_center=5.0, steepness=0.5)
        
        >>> # GDD-based growth factor
        >>> gdd = torch.linspace(0, 1000, 365)
        >>> growth = sigmoid_response(gdd, x_center=200, steepness=0.01)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    
    sigmoid_term = torch.sigmoid(steepness * (x - x_center))
    return y_min + (y_max - y_min) * sigmoid_term


def tanh_response(
    x: Tensor,
    x_center: float,
    steepness: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0
) -> Tensor:
    """
    Hyperbolic tangent response function.
    
    Similar to sigmoid but with different mathematical properties.
    Often used for symmetric responses around a center point.
    
    Parameters:
        x : Tensor
            Input variable
        x_center : float
            Center point where y = (y_max + y_min) / 2
        steepness : float
            Controls transition steepness, default: 1.0
        y_min : float
            Minimum output value, default: 0.0
        y_max : float
            Maximum output value, default: 1.0
            
    Returns:
        Tensor: Tanh response values between y_min and y_max
        
    Mathematical Expression:
        y = y_min + (y_max - y_min) * [0.5 * (1 + tanh(steepness * (x - x_center)))]
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    
    tanh_term = 0.5 * (1.0 + torch.tanh(steepness * (x - x_center)))
    return y_min + (y_max - y_min) * tanh_term


def linear_threshold_response(
    x: Tensor,
    x_min: float,
    x_max: float,
    y_min: float = 0.0,
    y_max: float = 1.0
) -> Tensor:
    """
    Piecewise linear threshold response function.
    
    Creates a linear ramp between x_min and x_max, with constant
    values outside this range. Commonly used for temperature
    responses in GDD calculations.
    
    Parameters:
        x : Tensor
            Input variable
        x_min : float
            Lower threshold (y = y_min for x <= x_min)
        x_max : float
            Upper threshold (y = y_max for x >= x_max)  
        y_min : float
            Minimum output value, default: 0.0
        y_max : float
            Maximum output value, default: 1.0
            
    Returns:
        Tensor: Piecewise linear response
        
    Mathematical Expression:
        y = {
            y_min                                    if x <= x_min
            y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)  if x_min < x < x_max
            y_max                                    if x >= x_max
        }
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    
    # Clamp to range and scale linearly
    # Get clamp bounds from parameter manager
    clamp_min = _read_parameter(
        "biosphere.phenology.validation.clamp_min",
    )
    clamp_max = _read_parameter(
        "biosphere.phenology.validation.clamp_max",
    )
    x_scaled = torch.clamp((x - x_min) / (x_max - x_min), min=clamp_min, max=clamp_max)
    return y_min + (y_max - y_min) * x_scaled


def exponential_response(
    x: Tensor,
    rate: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    saturating: bool = False
) -> Tensor:
    """
    Exponential response function.
    
    Used for processes that show exponential growth or decay,
    such as temperature effects on biological rates.
    
    Parameters:
        x : Tensor
            Input variable
        rate : float
            Exponential rate parameter
        x_offset : float
            Horizontal offset, default: 0.0
        y_offset : float
            Vertical offset, default: 0.0
        saturating : bool
            If True, use saturating exponential (1 - exp), default: False
            
    Returns:
        Tensor: Exponential response
        
    Mathematical Expression:
        Non-saturating: y = y_offset + exp(rate * (x - x_offset))
        Saturating: y = y_offset + (1 - exp(-rate * (x - x_offset)))
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    
    if saturating:
        return y_offset + (1.0 - torch.exp(-rate * (x - x_offset)))
    else:
        return y_offset + torch.exp(rate * (x - x_offset))


def step_response(
    x: Tensor,
    threshold: float,
    y_low: float = 0.0,
    y_high: float = 1.0
) -> Tensor:
    """
    Step function response.
    
    Creates an abrupt transition at a threshold value.
    Simple but useful for binary phenological switches.
    
    Parameters:
        x : Tensor
            Input variable
        threshold : float
            Threshold value for step transition
        y_low : float
            Output value for x < threshold, default: 0.0
        y_high : float
            Output value for x >= threshold, default: 1.0
            
    Returns:
        Tensor: Step response
        
    Mathematical Expression:
        y = {
            y_low   if x < threshold
            y_high  if x >= threshold
        }
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    
    return torch.where(x >= threshold, y_high, y_low)


def double_sigmoid_response(
    x: Tensor,
    x_onset: float,
    x_offset: float,
    steepness_onset: float = 1.0,
    steepness_offset: float = 1.0,
    y_base: float = 0.0,
    y_peak: float = 1.0
) -> Tensor:
    """
    Double sigmoid response for seasonal patterns.
    
    Combines two sigmoid functions to create a bell-shaped or 
    seasonal response pattern, commonly used in phenological models
    for growth factors that increase then decrease.
    
    Parameters:
        x : Tensor
            Input variable (e.g., day of year)
        x_onset : float
            Center of onset sigmoid
        x_offset : float  
            Center of offset sigmoid
        steepness_onset : float
            Steepness of onset transition, default: 1.0
        steepness_offset : float
            Steepness of offset transition, default: 1.0
        y_base : float
            Base value, default: 0.0
        y_peak : float
            Peak value, default: 1.0
            
    Returns:
        Tensor: Double sigmoid response
        
    Mathematical Expression:
        y = y_base + (y_peak - y_base) * sigma_1(x) * (1 - sigma_2(x))
        where:
        sigma_1(x) = sigmoid(steepness_onset * (x - x_onset))
        sigma_2(x) = sigmoid(steepness_offset * (x - x_offset))
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    
    onset_sigmoid = torch.sigmoid(steepness_onset * (x - x_onset))
    offset_sigmoid = torch.sigmoid(steepness_offset * (x - x_offset))
    
    combined = onset_sigmoid * (1.0 - offset_sigmoid)
    return y_base + (y_peak - y_base) * combined


# Convenience functions for common phenological responses
def growth_onset_response(day_of_year: Tensor, onset_doy: float, steepness: float = 0.2) -> Tensor:
    """Smooth growth onset response based on day of year."""
    return sigmoid_response(day_of_year, onset_doy, steepness, 0.0, 1.0)


def senescence_response(day_of_year: Tensor, offset_doy: float, steepness: float = 0.2) -> Tensor:
    """Smooth senescence response based on day of year.""" 
    return sigmoid_response(day_of_year, offset_doy, steepness, 1.0, 0.0)


def temperature_stress_response(temperature: Tensor, optimal_temp: float, tolerance: float = 5.0) -> Tensor:
    """Temperature stress response function."""
    temp_deviation = torch.abs(temperature - optimal_temp)
    return torch.exp(-temp_deviation / tolerance)

