"""
Unified Response Functions Library for RELIC

This module provides standardized mathematical response functions used across 
all scientific modules in the RELIC framework. All modules must import 
response functions from this unified library to ensure consistency and 
avoid code duplication.

References
----------
- Arrhenius, S. (1889). On the reaction rate of the inversion of sucrose by acids
- Farquhar, G.D., von Caemmerer, S., Berry, J.A. (1980). A biochemical model of photosynthetic CO2 assimilation
- Bernacchi, C.J. et al. (2001). Improved temperature response functions for models of Rubisco-limited photosynthesis
- Medlyn, B.E. et al. (2002). Temperature response of parameters of a biochemically based model of photosynthesis
- de Pury, D.G.G. and Farquhar, G.D. (1997). Simple scaling of photosynthesis from leaves to canopies

Mathematical Expressions
------------------------
1. Arrhenius: f(T) = f25 * exp(Ea/R * (1/298.15 - 1/T))
2. Peaked: f(T) = f25 * exp(Ea/R * (1/298.15 - 1/T)) / (1 + exp((S*T - Hd)/(R*T)))
3. Q10: f(T) = f25 * Q10^((T-298.15)/10)
4. Light response: A = (alpha*I + Amax - sqrt((alpha*I + Amax)^2 - 4*theta*alpha*I*Amax)) / (2*theta)
5. Michaelis-Menten: v = Vmax * S / (Km + S)

Units
-----
- Temperature (T): K
- Reference temperature: 298.15 K (25degC)
- Activation energy (Ea): J mol-1
- Deactivation energy (Hd): J mol-1 
- Entropy factor (S): J mol-1 K-1
- Gas constant (R): 8.314 J mol-1 K-1
- Light (I): mumol photons m-2 s-1
- Rates: Various units depending on process
"""

from typing import Union, Optional, Dict, Any, Callable, Tuple
import torch

_PARAM_MANAGER: Optional[Any] = None
_PARAM_MANAGER_ERROR: Optional[Exception] = None
_CONSTANT_CACHE: Dict[str, float] = {}


def _resolve_param_manager() -> Any:
    """Resolve the global ParameterManager lazily."""
    global _PARAM_MANAGER, _PARAM_MANAGER_ERROR
    if _PARAM_MANAGER is not None:
        return _PARAM_MANAGER
    from relic.parameters import get_global_manager

    _PARAM_MANAGER = get_global_manager()
    _PARAM_MANAGER_ERROR = None
    return _PARAM_MANAGER


def _resolve_constant(param_key: str, units: str) -> float:
    """Fetch a constant from the parameter database with caching."""
    if param_key in _CONSTANT_CACHE:
        return _CONSTANT_CACHE[param_key]
    manager = _resolve_param_manager()
    value = manager.get(param_key, units=units, required=True)
    if isinstance(value, torch.Tensor):
        resolved = float(value.detach().cpu().item())
    else:
        resolved = float(value)
    _CONSTANT_CACHE[param_key] = resolved
    return resolved


class _LazyConstant:
    """Lazy constant wrapper that resolves on first numeric use."""

    def __init__(self, name: str, resolver: Callable[[], float]):
        self._name = name
        self._resolver = resolver
        self._value: Optional[float] = None

    def _get(self) -> float:
        if self._value is None:
            self._value = float(self._resolver())
        return self._value

    def __float__(self) -> float:
        return self._get()

    def __int__(self) -> int:
        return int(self._get())

    def __repr__(self) -> str:
        return f"{self._name}({self._get()})"

    def __str__(self) -> str:
        return str(self._get())

    def __add__(self, other: Any) -> Any:
        return self._get() + other

    def __radd__(self, other: Any) -> Any:
        return other + self._get()

    def __sub__(self, other: Any) -> Any:
        return self._get() - other

    def __rsub__(self, other: Any) -> Any:
        return other - self._get()

    def __mul__(self, other: Any) -> Any:
        return self._get() * other

    def __rmul__(self, other: Any) -> Any:
        return other * self._get()

    def __truediv__(self, other: Any) -> Any:
        return self._get() / other

    def __rtruediv__(self, other: Any) -> Any:
        return other / self._get()

    def __pow__(self, other: Any) -> Any:
        return self._get() ** other

    def __rpow__(self, other: Any) -> Any:
        return other ** self._get()

    def __neg__(self) -> float:
        return -self._get()

    def __pos__(self) -> float:
        return +self._get()


def _coerce_constant(value: Any, name: str) -> float:
    """Coerce constant-like inputs (LazyConstant, Tensor, float) to float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor.")
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001
        raise TypeError(f"{name} must be convertible to float.") from exc


R_GAS = _LazyConstant(
    "R_GAS",
    lambda: _resolve_constant(
        "relic.physical.constants.universal.gas_constant",
        "J mol-1 K-1",
    ),
)
T_REF = _LazyConstant(
    "T_REF",
    lambda: _resolve_constant(
        "relic.physical.constants.parameters.t_ref",
        "K",
    ),
)
KELVIN_OFFSET = _LazyConstant(
    "KELVIN_OFFSET",
    lambda: _resolve_constant(
        "relic.physical.constants.temperature.celsius_to_kelvin",
        "K",
    ),
)


def _on_temperature_basis(
    temperature: torch.Tensor, *coefficients: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    """Place every coefficient of a temperature law on the temperature's basis.

    The input temperature decides the computation device and dtype. A tensor
    coefficient elsewhere (for example a ParameterManager-held nn.Parameter,
    whose device the manager decides) is converted with the differentiable
    ``Tensor.to``, so its gradient still reaches the original tensor; a number
    becomes a tensor. A temperature that is not a real floating tensor raises
    TypeError.
    """
    if (
        not isinstance(temperature, torch.Tensor)
        or not temperature.is_floating_point()
        or temperature.is_complex()
    ):
        raise TypeError(
            "temperature must be a real floating tensor [K], got "
            f"{getattr(temperature, 'dtype', type(temperature).__name__)}"
        )
    return tuple(
        coefficient.to(device=temperature.device, dtype=temperature.dtype)
        if isinstance(coefficient, torch.Tensor)
        else torch.tensor(coefficient, dtype=temperature.dtype, device=temperature.device)
        for coefficient in coefficients
    )


def arrhenius_temperature_response(
    param_25: Union[float, torch.Tensor], 
    temperature: torch.Tensor,
    activation_energy: Union[float, torch.Tensor],
    gas_constant: float = R_GAS,
    reference_temp: float = T_REF
) -> torch.Tensor:
    """
    Arrhenius temperature response function.
    
    Mathematical Expression
    -----------------------
    f(T) = f25 * exp(Ea/R * (1/T_ref - 1/T))
    
    References
    ----------
    - Arrhenius, S. (1889). On the reaction rate of the inversion of sucrose by acids
    - Farquhar et al. (1980). A biochemical model of photosynthetic CO2 assimilation
    
    Parameters
    ----------
    param_25 : float or torch.Tensor
        Parameter value at reference temperature (25degC)
    temperature : torch.Tensor
        Temperature [K]
    activation_energy : float or torch.Tensor
        Activation energy [J mol-1]
    gas_constant : float, optional
        Universal gas constant [J mol-1 K-1], default 8.314
    reference_temp : float, optional
        Reference temperature [K], default 298.15
        
    Returns
    -------
    torch.Tensor
        Temperature-adjusted parameter value
        
    Units
    -----
    - Input parameter: Same units as param_25
    - Output: Same units as param_25
    """
    # The temperature decides device and dtype; coefficients follow it.
    param_25, activation_energy = _on_temperature_basis(temperature, param_25, activation_energy)

    # Convert constants to tensors with same device as temperature
    gas_constant_value = _coerce_constant(gas_constant, "gas_constant")
    reference_temp_value = _coerce_constant(reference_temp, "reference_temp")
    gas_constant_tensor = torch.full_like(temperature, gas_constant_value)
    reference_temp_tensor = torch.full_like(temperature, reference_temp_value)
    
    # Calculate temperature response
    exponent = activation_energy / gas_constant_tensor * (1.0 / reference_temp_tensor - 1.0 / temperature)
    response = param_25 * torch.exp(exponent)
    
    return response


def peaked_temperature_response(
    param_25: Union[float, torch.Tensor],
    temperature: torch.Tensor, 
    activation_energy: Union[float, torch.Tensor],
    deactivation_energy: Union[float, torch.Tensor],
    entropy_factor: Union[float, torch.Tensor],
    gas_constant: float = R_GAS,
    reference_temp: float = T_REF
) -> torch.Tensor:
    """
    Peaked temperature response with high-temperature deactivation.
    
    Mathematical Expression
    -----------------------
    f(T) = f25 * exp(Ha/R * (1/T_ref - 1/T))
           * (1 + exp((S*T_ref - Hd)/(R*T_ref))) / (1 + exp((S*T - Hd)/(R*T)))
    The ratio of the two deactivation terms is normalized at T_ref so that
    f(T_ref) = f25 exactly; the denominator represents high-temperature
    inhibition and the numerator is its reference-temperature value.
    
    References
    ----------
    - Bernacchi, C.J. et al. (2001). Improved temperature response functions
    - Medlyn, B.E. et al. (2002). Temperature response of parameters
    
    Parameters
    ----------
    param_25 : float or torch.Tensor
        Parameter value at reference temperature (25degC)
    temperature : torch.Tensor
        Temperature [K]
    activation_energy : float or torch.Tensor
        Activation energy [J mol-1]
    deactivation_energy : float or torch.Tensor  
        Deactivation energy [J mol-1]
    entropy_factor : float or torch.Tensor
        Entropy term [J mol-1 K-1]
    gas_constant : float, optional
        Universal gas constant [J mol-1 K-1], default 8.314
    reference_temp : float, optional
        Reference temperature [K], default 298.15
        
    Returns
    -------
    torch.Tensor
        Temperature-adjusted parameter value
        
    Units
    -----
    - Input parameter: Same units as param_25
    - Output: Same units as param_25
    """
    # The temperature decides device and dtype; coefficients follow it.
    param_25, activation_energy, deactivation_energy, entropy_factor = _on_temperature_basis(
        temperature, param_25, activation_energy, deactivation_energy, entropy_factor,
    )

    # Convert constants to tensors with same device as temperature
    gas_constant_value = _coerce_constant(gas_constant, "gas_constant")
    reference_temp_value = _coerce_constant(reference_temp, "reference_temp")
    gas_constant_tensor = torch.full_like(temperature, gas_constant_value)
    reference_temp_tensor = torch.full_like(temperature, reference_temp_value)
    
    # Arrhenius activation term
    activation_term = torch.exp(
        activation_energy / gas_constant_tensor * (1.0 / reference_temp_tensor - 1.0 / temperature)
    )
    
    # High-temperature deactivation term
    numerator = 1.0 + torch.exp(
        (entropy_factor * reference_temp_tensor - deactivation_energy) / (gas_constant_tensor * reference_temp_tensor)
    )
    denominator = 1.0 + torch.exp(
        (entropy_factor * temperature - deactivation_energy) / (gas_constant_tensor * temperature)
    )
    
    deactivation_term = numerator / denominator
    
    response = param_25 * activation_term * deactivation_term
    
    return response


def q10_temperature_response(
    param_25: Union[float, torch.Tensor],
    temperature: torch.Tensor,
    q10: Union[float, torch.Tensor],
    reference_temp: float = T_REF
) -> torch.Tensor:
    """
    Q10 temperature response function.
    
    Mathematical Expression
    -----------------------
    f(T) = f25 * Q10^((T - T_ref)/10)
    
    References
    ----------
    - Van't Hoff temperature coefficient relationship
    - Commonly used for respiration and decomposition processes
    
    Parameters
    ----------
    param_25 : float or torch.Tensor
        Parameter value at reference temperature (25degC)
    temperature : torch.Tensor
        Temperature [K]
    q10 : float or torch.Tensor
        Q10 coefficient (factor increase per 10K temperature rise)
    reference_temp : float, optional
        Reference temperature [K], default 298.15
        
    Returns
    -------
    torch.Tensor
        Temperature-adjusted parameter value
        
    Units
    -----
    - Input parameter: Same units as param_25
    - Output: Same units as param_25
    """
    # The temperature decides device and dtype; coefficients follow it.
    param_25, q10 = _on_temperature_basis(temperature, param_25, q10)

    # Convert reference temperature to tensor with same device as temperature
    reference_temp_value = _coerce_constant(reference_temp, "reference_temp")
    reference_temp_tensor = torch.full_like(temperature, reference_temp_value)
    
    # Calculate Q10 response
    exponent = (temperature - reference_temp_tensor) / 10.0
    response = param_25 * torch.pow(q10, exponent)
    
    return response


def beer_lambert_effective_leaf_area(
    leaf_area_index: torch.Tensor,
    extinction_coefficient: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Integrate illuminated leaf area over a Beer-Lambert canopy.

    References
    ----------
    - Monsi and Saeki (1953), Japanese Journal of Botany 14, 22-52.
    - de Pury and Farquhar (1997), Plant Cell and Environment 20, 537-557.

    Mathematical expression
    -----------------------
    L_eff = integral_0^LAI exp(-k L) dL = (1 - exp(-k LAI)) / k.

    Parameters
    ----------
    leaf_area_index
        Leaf area per ground area [m2 leaf m-2 ground].
    extinction_coefficient
        Positive Beer-Lambert light extinction coefficient [dimensionless].

    Returns
    -------
    torch.Tensor
        Effective illuminated leaf area per ground area
        [m2 leaf m-2 ground].
    """
    if not isinstance(leaf_area_index, torch.Tensor):
        raise TypeError("leaf_area_index must be a torch.Tensor.")
    extinction = torch.as_tensor(
        extinction_coefficient,
        dtype=leaf_area_index.dtype,
        device=leaf_area_index.device,
    )
    if not torch.isfinite(leaf_area_index).all():
        raise ValueError("leaf_area_index must contain only finite values.")
    if torch.any(leaf_area_index < 0):
        raise ValueError("leaf_area_index must be non-negative.")
    if not torch.isfinite(extinction).all() or torch.any(extinction <= 0):
        raise ValueError(
            "extinction_coefficient must contain finite positive values."
        )
    return (1.0 - torch.exp(-extinction * leaf_area_index)) / extinction


def light_response_rectangular_hyperbola(
    light_intensity: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    amax: torch.Tensor,
    theta: Union[float, torch.Tensor] = 0.7
) -> torch.Tensor:
    """
    Rectangular hyperbola light response curve.
    
    Mathematical Expression
    -----------------------
    A = (alpha*I + Amax - sqrt((alpha*I + Amax)^2 - 4*theta*alpha*I*Amax)) / (2*theta)
    
    References
    ----------
    - Farquhar & Wong (1984). An empirical model of stomatal conductance
    - de Pury & Farquhar (1997). Simple scaling of photosynthesis
    
    Parameters
    ----------
    light_intensity : torch.Tensor
        Photosynthetically active radiation [mumol photons m-2 s-1]
    alpha : float or torch.Tensor
        Initial quantum yield [mol CO2 mol-1 photons] or [mol e? mol-1 photons]
    amax : torch.Tensor
        Light-saturated maximum rate [mumol m-2 s-1]
    theta : float or torch.Tensor, optional
        Curvature factor [dimensionless], default 0.7
        
    Returns
    -------
    torch.Tensor
        Actual rate under given light conditions [mumol m-2 s-1]
        
    Units
    -----
    - Light intensity: mumol photons m-2 s-1
    - Rate: mumol m-2 s-1 (CO2 or electrons depending on process)
    """
    # Ensure inputs are tensors
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=light_intensity.dtype, device=light_intensity.device)
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=light_intensity.dtype, device=light_intensity.device)
    
    # Calculate light response
    alpha_i = alpha * light_intensity
    sum_term = alpha_i + amax
    discriminant = sum_term * sum_term - 4.0 * theta * alpha_i * amax
    
    # Ensure discriminant is non-negative
    discriminant_safe = torch.clamp(discriminant, min=0.0)
    
    response = (sum_term - torch.sqrt(discriminant_safe)) / (2.0 * theta)
    response = torch.clamp(response, min=0.0)
    
    return response


def light_response_exponential(
    light_intensity: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    amax: torch.Tensor
) -> torch.Tensor:
    """
    Exponential light response curve.
    
    Mathematical Expression
    -----------------------
    A = Amax * (1 - exp(-alpha*I/Amax))
    
    References
    ----------
    - Alternative light response formulation
    - Often used in simplified canopy models
    
    Parameters
    ----------
    light_intensity : torch.Tensor
        Photosynthetically active radiation [mumol photons m-2 s-1]
    alpha : float or torch.Tensor
        Initial quantum yield [mol CO2 mol-1 photons]
    amax : torch.Tensor
        Light-saturated maximum rate [mumol m-2 s-1]
        
    Returns
    -------
    torch.Tensor
        Actual rate under given light conditions [mumol m-2 s-1]
        
    Units
    -----
    - Light intensity: mumol photons m-2 s-1
    - Rate: mumol m-2 s-1
    """
    # Ensure inputs are tensors
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=light_intensity.dtype, device=light_intensity.device)
    
    # Calculate exponential light response
    exponent = -alpha * light_intensity / (amax + 1e-8)  # Avoid division by zero
    response = amax * (1.0 - torch.exp(exponent))
    response = torch.clamp(response, min=0.0)
    
    return response


def michaelis_menten_response(
    substrate_concentration: torch.Tensor,
    vmax: torch.Tensor,
    km: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Michaelis-Menten kinetic response.
    
    Mathematical Expression
    -----------------------
    v = Vmax * [S] / (Km + [S])
    
    References
    ----------
    - Michaelis, L. and Menten, M.L. (1913). The kinetics of invertase action
    - Fundamental enzyme kinetics equation
    
    Parameters
    ----------
    substrate_concentration : torch.Tensor
        Substrate concentration [various units]
    vmax : torch.Tensor  
        Maximum reaction rate [various units]
    km : float or torch.Tensor
        Michaelis-Menten constant [same units as substrate]
        
    Returns
    -------
    torch.Tensor
        Reaction rate [same units as vmax]
        
    Units
    -----
    - Substrate: Depends on specific substrate (e.g., mumol mol-1 for CO2)
    - Vmax: Depends on process (e.g., mumol m-2 s-1)
    - Km: Same units as substrate
    """
    # Ensure inputs are tensors
    if not isinstance(km, torch.Tensor):
        km = torch.tensor(km, dtype=substrate_concentration.dtype, device=substrate_concentration.device)
    
    # Calculate Michaelis-Menten response
    response = vmax * substrate_concentration / (km + substrate_concentration)
    response = torch.clamp(response, min=0.0)
    
    return response


def exponential_decay_response(
    variable: torch.Tensor,
    coefficient: Union[float, torch.Tensor],
    reference_value: Union[float, torch.Tensor] = 0.0
) -> torch.Tensor:
    """
    Exponential decay response function.
    
    Mathematical Expression
    -----------------------
    f(x) = exp(-k * (x - x_ref))
    
    References
    ----------
    - Beer-Lambert law for light attenuation
    - Extinction coefficients in canopy modeling
    
    Parameters
    ----------
    variable : torch.Tensor
        Independent variable (e.g., depth, distance)
    coefficient : float or torch.Tensor
        Extinction/decay coefficient [units-1]
    reference_value : float or torch.Tensor, optional
        Reference point for decay, default 0.0
        
    Returns
    -------
    torch.Tensor
        Decay factor [dimensionless, 0-1]
        
    Units
    -----
    - Variable: Depends on application (e.g., m for depth, LAI for canopy)
    - Coefficient: Inverse of variable units
    - Output: Dimensionless
    """
    # Ensure inputs are tensors
    if not isinstance(coefficient, torch.Tensor):
        coefficient = torch.tensor(coefficient, dtype=variable.dtype, device=variable.device)
    if not isinstance(reference_value, torch.Tensor):
        reference_value = torch.tensor(reference_value, dtype=variable.dtype, device=variable.device)
    
    # Calculate exponential decay
    exponent = -coefficient * (variable - reference_value)
    response = torch.exp(exponent)
    response = torch.clamp(response, min=0.0, max=1.0)
    
    return response


def linear_response(
    variable: torch.Tensor,
    slope: Union[float, torch.Tensor],
    intercept: Union[float, torch.Tensor] = 0.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> torch.Tensor:
    """
    Linear response function with optional bounds.
    
    Mathematical Expression
    -----------------------
    f(x) = slope * x + intercept
    
    References
    ----------
    - Simple linear relationships
    - Used for approximations and empirical relationships
    
    Parameters
    ----------
    variable : torch.Tensor
        Independent variable
    slope : float or torch.Tensor
        Linear slope coefficient
    intercept : float or torch.Tensor, optional
        Y-intercept, default 0.0
    min_value : float, optional
        Minimum bound for output
    max_value : float, optional
        Maximum bound for output
        
    Returns
    -------
    torch.Tensor
        Linear response with optional bounds
        
    Units
    -----
    - Variable: Depends on application
    - Slope: Output units / variable units
    - Output: Depends on slope and intercept units
    """
    # Ensure inputs are tensors
    if not isinstance(slope, torch.Tensor):
        slope = torch.tensor(slope, dtype=variable.dtype, device=variable.device)
    if not isinstance(intercept, torch.Tensor):
        intercept = torch.tensor(intercept, dtype=variable.dtype, device=variable.device)
    
    # Calculate linear response
    response = slope * variable + intercept
    
    # Apply bounds if specified
    if min_value is not None:
        response = torch.clamp(response, min=min_value)
    if max_value is not None:
        response = torch.clamp(response, max=max_value)
        
    return response


def saturation_response(
    variable: torch.Tensor,
    half_saturation: Union[float, torch.Tensor],
    max_response: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Saturation response function (hyperbolic).
    
    Mathematical Expression
    -----------------------
    f(x) = max_response * x / (x + K_half)
    
    References
    ----------
    - Monod equation for microbial growth
    - Nutrient limitation functions
    
    Parameters
    ----------
    variable : torch.Tensor
        Independent variable (e.g., nutrient concentration)
    half_saturation : float or torch.Tensor
        Half-saturation constant (value at 50% max response)
    max_response : float or torch.Tensor, optional
        Maximum response value, default 1.0
        
    Returns
    -------
    torch.Tensor
        Saturation response [0 to max_response]
        
    Units
    -----
    - Variable: Depends on application
    - Half_saturation: Same units as variable
    - Output: Same units as max_response
    """
    # Ensure inputs are tensors
    if not isinstance(half_saturation, torch.Tensor):
        half_saturation = torch.tensor(half_saturation, dtype=variable.dtype, device=variable.device)
    if not isinstance(max_response, torch.Tensor):
        max_response = torch.tensor(max_response, dtype=variable.dtype, device=variable.device)
    
    # Calculate saturation response
    response = max_response * variable / (variable + half_saturation)
    response = torch.clamp(response, min=0.0)
    
    return response


class StressResponseFunctions:
    """
    Collection of stress response functions for environmental limitations.
    
    These functions convert environmental variables into stress factors 
    (typically 0-1 range) that multiply process rates.
    
    References
    ----------
    - Various plant stress physiology literature
    - Environmental response functions in ecosystem models
    """
    
    @staticmethod
    def water_stress_linear(
        water_potential: torch.Tensor,
        wilting_point: float = -1.5,
        field_capacity: float = -0.1
    ) -> torch.Tensor:
        """
        Linear water stress response.
        
        Mathematical Expression
        -----------------------
        stress = (psi - psi_wilt) / (psi_fc - psi_wilt)
        bounded between 0 and 1
        
        Parameters
        ----------
        water_potential : torch.Tensor
            Soil water potential [MPa]
        wilting_point : float
            Permanent wilting point [MPa], default -1.5
        field_capacity : float  
            Field capacity [MPa], default -0.1
            
        Returns
        -------
        torch.Tensor
            Water stress factor [0-1, where 1 = no stress]
            
        Units
        -----
        - Water potential: MPa
        - Output: Dimensionless [0-1]
        """
        stress_factor = (water_potential - wilting_point) / (field_capacity - wilting_point)
        stress_factor = torch.clamp(stress_factor, min=0.0, max=1.0)
        return stress_factor


# Create instance for easy access
stress_response_functions = StressResponseFunctions()


# Validation functions
def validate_temperature_range(temperature: torch.Tensor, min_temp: float = 233.15, max_temp: float = 333.15):
    """Validate temperature is within reasonable range for biological processes."""
    # Convert scalar to tensor if necessary
    if not isinstance(temperature, torch.Tensor):
        temperature = torch.tensor(temperature)
    
    # Handle both scalar and tensor inputs properly
    tolerance = 1e-5  # Tolerance for floating point comparison
    if temperature.numel() == 1:
        # Single value - use comparison with tolerance
        temp_val = temperature.item()
        if temp_val < (min_temp - tolerance) or temp_val > (max_temp + tolerance):
            raise ValueError(f"Temperature outside valid range [{min_temp}-{max_temp}] K")
    else:
        # Multiple values - use torch.any with tolerance
        if torch.any(temperature < (min_temp - tolerance)) or torch.any(temperature > (max_temp + tolerance)):
            raise ValueError(f"Temperature outside valid range [{min_temp}-{max_temp}] K")


def validate_positive_values(tensor: torch.Tensor, name: str):
    """Validate tensor contains only positive values."""
    if torch.any(tensor < 0):
        raise ValueError(f"{name} must contain only non-negative values")


def validate_unity_range(tensor: torch.Tensor, name: str):
    """Validate tensor values are in [0,1] range."""  
    if torch.any(tensor < 0) or torch.any(tensor > 1):
        raise ValueError(f"{name} values must be in range [0,1]")


def validate_tensor_shapes(tensors, name: str):
    """Validate that all tensors have consistent shapes."""
    if not tensors:
        return
    
    reference_shape = tensors[0].shape
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape != reference_shape:
            raise ValueError(f"{name}: tensor {i} shape {tensor.shape} doesn't match reference {reference_shape}")


# Convenience aliases for common use cases
temperature_response_q10 = q10_temperature_response
temperature_response_arrhenius = arrhenius_temperature_response
temperature_response_peaked = peaked_temperature_response

def lloyd_taylor_temperature_response(
    param_ref: Union[float, torch.Tensor],
    temperature: torch.Tensor,
    e0: Union[float, torch.Tensor] = 308.56,
    t0: float = 227.13,
    reference_temp: float = 283.15
) -> torch.Tensor:
    """
    Lloyd-Taylor temperature response function for respiration.
    
    Mathematical Expression
    -----------------------
    R(T) = R_ref * exp[E0 * (1/(T_ref - T0) - 1/(T - T0))]
    
    References
    ----------
    - Lloyd, J., Taylor, J.A. (1994). On the temperature dependence of soil respiration
    
    Parameters
    ----------
    param_ref : float or torch.Tensor
        Parameter value at reference temperature
    temperature : torch.Tensor
        Temperature [K]
    e0 : float or torch.Tensor
        Activation energy parameter [K], default 308.56
    t0 : float
        Lower temperature limit [K], default 227.13 (-46.02degC)
    reference_temp : float
        Reference temperature [K], default 283.15 (10degC)
        
    Returns
    -------
    torch.Tensor
        Temperature-adjusted parameter value
        
    Units
    -----
    - Temperature: K
    - E0: K
    - Output: Same units as param_ref
    """
    # Ensure inputs are tensors
    if not isinstance(param_ref, torch.Tensor):
        param_ref = torch.tensor(param_ref, dtype=temperature.dtype, device=temperature.device)
    if not isinstance(e0, torch.Tensor):
        e0 = torch.tensor(e0, dtype=temperature.dtype, device=temperature.device)
    
    # Check temperature validity (must be above T0)
    if torch.any(temperature <= t0):
        raise ValueError(f"Temperature must be above T0 ({t0} K)")
    
    # Calculate Lloyd-Taylor response
    exponent = e0 * (1.0 / (reference_temp - t0) - 1.0 / (temperature - t0))
    response = param_ref * torch.exp(exponent)
    
    return response


def moisture_response_linear(moisture: torch.Tensor, 
                            wilting_point: float = 0.1,
                            field_capacity: float = 0.8) -> torch.Tensor:
    """
    Linear moisture response between wilting point and field capacity.
    
    Parameters:
    -----------
    moisture : torch.Tensor
        Soil moisture content [m^3 m-3]
    wilting_point : float
        Minimum moisture for activity
    field_capacity : float
        Maximum effective moisture
        
    Returns:
    --------
    torch.Tensor
        Moisture response factor [0-1]
    """
    response = (moisture - wilting_point) / (field_capacity - wilting_point)
    return torch.clamp(response, min=0.0, max=1.0)

def ph_response_gaussian(ph: torch.Tensor,
                        ph_optimum: float = 6.5,
                        ph_sigma: float = 1.5) -> torch.Tensor:
    """
    Gaussian pH response around optimum.
    
    Parameters:
    -----------
    ph : torch.Tensor
        Soil pH
    ph_optimum : float
        Optimal pH
    ph_sigma : float
        pH tolerance range (std dev)
        
    Returns:
    --------
    torch.Tensor
        pH response factor [0-1]
    """
    response = torch.exp(-(ph - ph_optimum)**2 / (2 * ph_sigma**2))
    return torch.clamp(response, min=0.0, max=1.0)

# Export all functions
__all__ = [
    'lloyd_taylor_temperature_response',
    'arrhenius_temperature_response',
    'peaked_temperature_response',
    'q10_temperature_response', 
    'light_response_rectangular_hyperbola',
    'light_response_exponential',
    'michaelis_menten_response',
    'exponential_decay_response',
    'linear_response',
    'saturation_response',
    'StressResponseFunctions',
    'stress_response_functions',
    # Aliases
    'temperature_response_q10',
    'temperature_response_arrhenius', 
    'temperature_response_peaked',
    'moisture_response_linear',
    'ph_response_gaussian',
    'validate_temperature_range',
    'validate_positive_values',
    'validate_unity_range',
    'validate_tensor_shapes',
    'R_GAS',
    'T_REF',
    'KELVIN_OFFSET'
]

