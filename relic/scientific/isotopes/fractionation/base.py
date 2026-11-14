"""
Base class for isotope fractionation calculators

This module provides the abstract base class for all isotope fractionation
calculators in the RELIC multi-isotope system. It defines the common
interface and shared functionality for 13C, 15N, and 18O fractionation
calculations.

Key Features:
- Abstract base class for isotope-specific calculators
- Common validation and error handling
- Temperature bounds checking
- Standard isotope ratio conversions
- PyTorch tensor support for automatic differentiation

Mathematical Framework:
- Kinetic Isotope Effect (KIE): KIE = k_light / k_heavy
- Fractionation Factor: alpha = 1/KIE = k_heavy / k_light  
- Fractionation (epsilon): epsilon = (KIE - 1) * 1000? = (alpha?^1 - 1) * 1000?
- Delta notation: delta = (R_sample/R_standard - 1) * 1000?

References:
- Bigeleisen, J. & Wolfsberg, M. (1958). Theoretical and experimental aspects of isotope effects
- Hayes, J.M. (2001). Fractionation of carbon and hydrogen isotopes in biosynthetic processes
- Coplen et al. (2006). New guidelines for reporting stable isotope measurements
"""

import torch
from typing import Dict, Optional, Union, Tuple, Any, List
from abc import ABC, abstractmethod
import warnings
import logging

from relic.parameters import UnifiedParameterManager
from relic.utils.physical_constants import get_physical_constant


class IsotopeFractionationCalculator(ABC):
    parameter_namespace = "isotopes.fractionation"
    """
    Abstract base class for isotope fractionation calculators.
    
    Provides common functionality and interface for isotope-specific
    fractionation calculators (13C, 15N, 18O, etc.).
    
    Attributes:
        isotope_name: Name of the isotope species (e.g., "13C", "15N", "18O")
        temperature_bounds: Valid temperature range for calculations
        
    Methods:
        calculate_fractionation_factor: Calculate process-specific fractionation
        validate_temperature: Check temperature bounds
        delta_notation: Convert ratio to delta notation  
        ratio_from_delta: Convert delta notation to ratio
    """
    
    # Parameter values are required from the ParameterManager.

    def get_parameter_value(
        self,
        name: str,
        namespace: Optional[str] = None,
        required: bool = True,
        units: Optional[str] = None,
        expected_type: Optional[type] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> Any:
        """Read a formal isotope YAML key through the injected manager.

        The key is required: a missing key raises. A caller reads an optional parameter
        with required=False and handles its absence (None) explicitly; no caller-supplied
        default value exists.
        """
        return self.parameter_manager.get(
            name,
            namespace=namespace or self.parameter_namespace,
            units=units,
            expected_type=expected_type,
            required=required,
            bounds=bounds,
        )

    def __init__(
        self,
        isotope_name: str,
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        """
        Initialize base fractionation calculator.
        
        Parameters:
            isotope_name: Name of isotope species (e.g., "13C", "15N", "18O")
        """
        if parameter_manager is None:
            raise TypeError("parameter_manager is required")
        self.isotope_name = isotope_name
        self.parameter_manager = parameter_manager
        self.logger = logging.getLogger(f"{__name__}.{isotope_name}")
        
        # Temperature and numerical parameters
        min_temp = float(
            self.get_parameter_value(
                "temperature.min",
                namespace="isotopes",
                required=True)
        )
        max_temp = float(
            self.get_parameter_value(
                "temperature.max",
                namespace="isotopes",
                required=True)
        )
        self.temperature_bounds = (min_temp, max_temp)
        self.reference_temperature = float(
            self.get_parameter_value(
                "temperature.reference",
                namespace="isotopes",
                required=True)
        )
        self.gas_constant = float(
            get_physical_constant(
                "relic.physical.constants.universal.gas_constant"
            )
        )
        self.min_fractionation_factor = float(
            self.get_parameter_value(
                "alpha_min",
                namespace="isotopes.fractionation",
                required=True)
        )
        self.max_fractionation_factor = float(
            self.get_parameter_value(
                "alpha_max",
                namespace="isotopes.fractionation",
                required=True)
        )
        temp_diff_min = float(
            self.get_parameter_value(
                "temperature_diff_lower",
                namespace="isotopes.validation",
                required=True)
        )
        temp_diff_max = float(
            self.get_parameter_value(
                "temperature_diff_upper",
                namespace="isotopes.validation",
                required=True)
        )
        self.temperature_diff_bounds = (temp_diff_min, temp_diff_max)
        arrhenius_min = float(
            self.get_parameter_value(
                "arrhenius_exponent_min",
                namespace="isotopes.validation.numerical",
                required=True)
        )
        arrhenius_max = float(
            self.get_parameter_value(
                "arrhenius_exponent_max",
                namespace="isotopes.validation.numerical",
                required=True)
        )
        self.arrhenius_exponent_bounds = (arrhenius_min, arrhenius_max)
        
        self.logger.debug(f"Initialized {isotope_name} fractionation calculator")

    @abstractmethod
    def calculate_fractionation_factor(
        self,
        process_type: Union[str, Any],
        temperature: torch.Tensor,
        process_params: Optional[Any] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate fractionation factor for a specific process and conditions.
        
        Must be implemented by each isotope-specific calculator.
        
        Parameters:
            process_type: Type of process (isotope-specific enum)
            temperature: Temperature [K], shape [...]
            process_params: Process-specific parameters
            environmental_state: Environmental variables
            **kwargs: Additional parameters
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
        """
        pass

    def calculate_isotope_flux(
        self,
        process_name: Union[str, Any],
        total_flux: torch.Tensor,
        source_ratio: torch.Tensor,
        temperature: torch.Tensor,
        process_params: Optional[Any] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        component_basis: str = "amount_of_substance",
        light_relative_atomic_mass: Optional[Any] = None,
        heavy_relative_atomic_mass: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Partition an oriented binary total flux into its heavy component.

        References:
            IUPAC isotope-ratio and isotope-amount-fraction recommendations;
            Coplen (2011), Pure Appl. Chem. 83, 1873-1935.

        Mathematical expression:
            R_s = n_H / n_L, alpha = R_f / R_s, R_f = alpha R_s,
            amount basis: F_H = F_T R_f / (1 + R_f);
            element-mass basis: F_H = F_T A_H R_f / (A_L + A_H R_f).

        Parameter meanings and units:
            process_name selects alpha; total_flux F_T has caller-declared
            element-flux units; source_ratio R_s and alpha are dimensionless;
            temperature is in K; A_L and A_H are dimensionless relative atomic
            masses. The returned F_H has the units of total_flux.

        Domain and sign convention:
            F_T >= 0 follows the declared source-to-target orientation,
            R_s >= 0, alpha > 0, and every tensor must be finite. This is an
            exact binary-component partition only; it does not prove physical
            total-oxygen closure when oxygen-17 is omitted.
        """
        raw_total = torch.as_tensor(total_flux)
        raw_ratio = torch.as_tensor(source_ratio)
        raw_temperature = torch.as_tensor(temperature)
        tensors = (raw_total, raw_ratio, raw_temperature)

        device = next(
            (tensor.device for tensor in tensors if isinstance(tensor, torch.Tensor)),
            torch.device("cpu"),
        )
        floating_dtypes = [
            tensor.dtype for tensor in tensors if torch.is_floating_point(tensor)
        ]
        dtype = floating_dtypes[0] if floating_dtypes else torch.get_default_dtype()
        for candidate_dtype in floating_dtypes[1:]:
            dtype = torch.promote_types(dtype, candidate_dtype)

        total = raw_total.to(device=device, dtype=dtype)
        ratio = raw_ratio.to(device=device, dtype=dtype)
        temperature_tensor = raw_temperature.to(device=device, dtype=dtype)
        if not torch.isfinite(temperature_tensor).all():
            raise FractionationError("temperature must be finite")
        alpha = torch.as_tensor(
            self.calculate_fractionation_factor(
                process_name,
                temperature_tensor,
                process_params=process_params,
                environmental_state=environmental_state,
                **kwargs,
            ),
            device=device,
            dtype=dtype,
        )

        for name, tensor in (
            ("total_flux", total),
            ("source_ratio", ratio),
            ("fractionation_factor", alpha),
        ):
            if not torch.isfinite(tensor).all():
                raise FractionationError(f"{name} must be finite")
        if torch.any(total < 0):
            raise FractionationError(
                "total_flux must follow the non-negative source-to-target orientation"
            )
        if torch.any(ratio < 0):
            raise FractionationError("source_ratio must be non-negative")
        if torch.any(alpha <= 0):
            raise FractionationError("fractionation_factor must be positive")

        if component_basis == "amount_of_substance":
            try:
                total_b, ratio_b, alpha_b = torch.broadcast_tensors(
                    total,
                    ratio,
                    alpha,
                )
            except RuntimeError as exc:
                raise FractionationError(
                    "total_flux, source_ratio, and fractionation_factor are not "
                    "broadcastable"
                ) from exc
            transferred_ratio = alpha_b * ratio_b
            numerator = transferred_ratio
            denominator = 1.0 + transferred_ratio
        elif component_basis == "element_mass":
            if (
                light_relative_atomic_mass is None
                or heavy_relative_atomic_mass is None
            ):
                raise FractionationError(
                    "element_mass requires light_relative_atomic_mass and "
                    "heavy_relative_atomic_mass"
                )
            light_mass = torch.as_tensor(
                light_relative_atomic_mass,
                device=device,
                dtype=dtype,
            )
            heavy_mass = torch.as_tensor(
                heavy_relative_atomic_mass,
                device=device,
                dtype=dtype,
            )
            for name, mass in (
                ("light_relative_atomic_mass", light_mass),
                ("heavy_relative_atomic_mass", heavy_mass),
            ):
                if not torch.isfinite(mass).all() or torch.any(mass <= 0):
                    raise FractionationError(f"{name} must be finite and positive")
            try:
                total_b, ratio_b, alpha_b, light_mass_b, heavy_mass_b = (
                    torch.broadcast_tensors(
                        total,
                        ratio,
                        alpha,
                        light_mass,
                        heavy_mass,
                    )
                )
            except RuntimeError as exc:
                raise FractionationError(
                    "element-mass flux inputs are not broadcastable"
                ) from exc
            transferred_ratio = alpha_b * ratio_b
            numerator = heavy_mass_b * transferred_ratio
            denominator = light_mass_b + numerator
        else:
            raise FractionationError(
                "component_basis must be 'amount_of_substance' or 'element_mass'"
            )

        if not torch.isfinite(transferred_ratio).all() or not torch.isfinite(
            denominator
        ).all():
            raise FractionationError("transferred isotope ratio overflowed")

        heavy_flux = total_b * numerator / denominator
        if not torch.isfinite(heavy_flux).all():
            raise FractionationError("heavy isotope flux must be finite")
        if torch.any(heavy_flux < 0) or torch.any(heavy_flux > total_b):
            raise FractionationError("heavy isotope flux violates binary partition bounds")
        return heavy_flux
    
    
    @abstractmethod
    def delta_notation(self, isotope_ratio: torch.Tensor) -> torch.Tensor:
        """
        Convert isotope ratio to delta notation.
        
        delta = (R_sample/R_standard - 1) * 1000?
        
        Parameters:
            isotope_ratio: Heavy/light isotope ratio
            
        Returns:
            Delta value [?]
        """
        pass
    
    @abstractmethod
    def ratio_from_delta(self, delta_value: torch.Tensor) -> torch.Tensor:
        """
        Convert delta notation to isotope ratio.
        
        R_sample = R_standard * (delta/1000 + 1)
        
        Parameters:
            delta_value: Delta value [?]
            
        Returns:
            Isotope ratio [dimensionless]
        """
        pass
    
    def validate_temperature(
        self,
        temperature: torch.Tensor,
        process_name: str = "unknown"
    ) -> torch.Tensor:
        """
        Validate and clamp temperature to physical bounds.
        
        Parameters:
            temperature: Temperature tensor [K]
            process_name: Process name for error messages
            
        Returns:
            Validated temperature tensor [K]
            
        Raises:
            ValueError: For severely out-of-bounds temperatures
        """
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor(temperature, dtype=torch.float32)
        
        # Check for non-finite values
        if not torch.isfinite(temperature).all():
            raise ValueError(
                f"Non-finite temperature values for {self.isotope_name} {process_name}"
            )
        
        min_temp, max_temp = self.temperature_bounds
        
        # Check bounds and warn for extreme values
        if torch.any(temperature < min_temp - 50) or torch.any(temperature > max_temp + 50):
            temp_min_value = temperature.min().detach().reshape(-1)[0].tolist()
            temp_max_value = temperature.max().detach().reshape(-1)[0].tolist()
            raise ValueError(
                f"Temperature severely out of bounds for {self.isotope_name} {process_name}: "
                f"[{temp_min_value:.1f}, {temp_max_value:.1f}] K, "
                f"valid range: [{min_temp:.1f}, {max_temp:.1f}] K"
            )
        
        if torch.any(temperature < min_temp) or torch.any(temperature > max_temp):
            warnings.warn(
                f"Temperature outside recommended range for {self.isotope_name} {process_name}, "
                f"clamping to [{min_temp:.1f}, {max_temp:.1f}] K"
            )
        
        # Clamp to bounds
        temperature_clamped = torch.clamp(temperature, min_temp, max_temp)
        
        return temperature_clamped
    
    def validate_fractionation_factor(
        self,
        alpha: torch.Tensor,
        process_name: str = "unknown"
    ) -> torch.Tensor:
        """
        Validate and clamp fractionation factor to reasonable bounds.
        
        Parameters:
            alpha: Fractionation factor [dimensionless]
            process_name: Process name for error messages
            
        Returns:
            Validated fractionation factor [dimensionless]
        """
        if not torch.isfinite(alpha).all():
            raise ValueError(
                f"Non-finite fractionation factors for {self.isotope_name} {process_name}"
            )
        
        # Check for extreme values
        too_small = alpha < self.min_fractionation_factor
        too_large = alpha > self.max_fractionation_factor
        
        if too_small.any() or too_large.any():
            alpha_min_value = alpha.min().detach().reshape(-1)[0].tolist()
            alpha_max_value = alpha.max().detach().reshape(-1)[0].tolist()
            raise ValueError(
                f"Fractionation factors for {self.isotope_name} {process_name} are "
                f"outside configured bounds: [{alpha_min_value:.4f}, {alpha_max_value:.4f}] "
                f"not in [{self.min_fractionation_factor:.4f}, {self.max_fractionation_factor:.4f}]"
            )

        return alpha
    
    def compute_q10_temperature_dependence(
        self,
        temperature: torch.Tensor,
        alpha_0: float,
        q10: float,
        t_ref: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute Q10 temperature dependence for fractionation factor.
        
        alpha(T) = 1 + (alpha_0 - 1) * Q10^((T_ref - T)/10)

        Parameters:
            temperature: Temperature [K]
            alpha_0: Reference fractionation factor
            q10: Q10 temperature coefficient
            t_ref: Reference temperature [K]

        Returns:
            Temperature-dependent fractionation factor
        """
        if t_ref is None:
            t_ref = self.reference_temperature

        # Validate inputs
        if not (0.5 < alpha_0 < 2.0):
            raise ValueError(f"alpha_0 {alpha_0} outside valid range [0.5, 2.0]")
        if not (0.5 < q10 < 10.0):
            raise ValueError(f"Q10 {q10} outside valid range [0.5, 10.0]")
        
        # References:
        #   - Bigeleisen and Wolfsberg (1958), isotope-effect theory.
        # Mathematical expression:
        #   alpha(T) - 1 = (alpha_0 - 1) Q10^((T_ref - T) / 10).
        # Parameter meanings:
        #   alpha_0: fractionation factor at T_ref [1]; q10: temperature
        #   scaling of the isotope-effect magnitude [1]; T_ref, T [K].
        temp_diff = (t_ref - temperature) / 10.0
        
        # Clamp to prevent numerical overflow
        temp_diff_clamped = torch.clamp(
            temp_diff,
            min=self.temperature_diff_bounds[0],
            max=self.temperature_diff_bounds[1],
        )  # +/-50K range
        
        if not torch.allclose(temp_diff, temp_diff_clamped):
            warnings.warn(
                "Extreme temperature differences detected, clamping to "
                f"[{self.temperature_diff_bounds[0]:.1f}, {self.temperature_diff_bounds[1]:.1f}]"
            )
        
        # Compute Q10 dependence
        q10_factor = torch.pow(torch.tensor(q10, dtype=temperature.dtype, device=temperature.device),
                               temp_diff_clamped)
        
        alpha = 1.0 + (alpha_0 - 1.0) * q10_factor
        
        return self.validate_fractionation_factor(alpha, "Q10")
    
    def compute_arrhenius_temperature_dependence(
        self,
        temperature: torch.Tensor,
        alpha_0: float,
        activation_energy: float,
        t_ref: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute Arrhenius temperature dependence for fractionation factor.
        
        alpha(T) = 1 + (alpha_0 - 1) * exp(Ea/R * (1/T - 1/T_ref))
        
        Parameters:
            temperature: Temperature [K]
            alpha_0: Reference fractionation factor
            activation_energy: Activation energy [J/mol]
            t_ref: Reference temperature [K]
            
        Returns:
            Temperature-dependent fractionation factor
        """
        if t_ref is None:
            t_ref = self.reference_temperature

        # Validate inputs
        if not (0.5 < alpha_0 < 2.0):
            raise ValueError(f"alpha_0 {alpha_0} outside valid range [0.5, 2.0]")
        if not (1000.0 < abs(activation_energy) < 300000.0):
            raise ValueError(f"Activation energy {activation_energy} J/mol outside valid range")
        
        # Compute exponential argument
        R = torch.tensor(self.gas_constant, dtype=temperature.dtype, device=temperature.device)
        Ea = torch.tensor(activation_energy, dtype=temperature.dtype, device=temperature.device)
        T_ref = torch.tensor(t_ref, dtype=temperature.dtype, device=temperature.device)
        
        # References:
        #   - Bigeleisen and Wolfsberg (1958), isotope-effect theory.
        # Mathematical expression:
        #   alpha(T) - 1 = (alpha_0 - 1) exp[Ea/R (1/T - 1/T_ref)].
        # Parameter meanings:
        #   alpha_0: fractionation factor at T_ref [1]; Ea [J mol-1];
        #   R [J mol-1 K-1]; T_ref, T [K].
        exp_arg = (Ea / R) * (1.0 / temperature - 1.0 / T_ref)
        
        # Clamp to prevent overflow
        exp_arg_clamped = torch.clamp(
            exp_arg,
            min=self.arrhenius_exponent_bounds[0],
            max=self.arrhenius_exponent_bounds[1],
        )
        
        if not torch.allclose(exp_arg, exp_arg_clamped):
            warnings.warn(
                "Extreme Arrhenius arguments detected, clamping to "
                f"[{self.arrhenius_exponent_bounds[0]:.1f}, {self.arrhenius_exponent_bounds[1]:.1f}]"
            )
        
        # Compute Arrhenius dependence
        exp_factor = torch.exp(exp_arg_clamped)
        alpha = 1.0 + (alpha_0 - 1.0) * exp_factor
        
        return self.validate_fractionation_factor(alpha, "Arrhenius")
    
    def get_supported_processes(self) -> List[str]:
        """
        Get list of supported process types for this isotope species.
        
        Returns:
            List of supported process names
        """
        # This should be overridden by specific implementations
        return []
    
    def get_isotope_info(self) -> Dict[str, Any]:
        """
        Get information about this isotope species.
        
        Returns:
            Dictionary with isotope information
        """
        return {
            'isotope_name': self.isotope_name,
            'temperature_bounds': self.temperature_bounds,
            'supported_processes': self.get_supported_processes(),
            'fractionation_bounds': (self.min_fractionation_factor, self.max_fractionation_factor)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}('{self.isotope_name}')"


class FractionationError(ValueError):
    """Base exception for isotope fractionation calculator errors."""
    pass


class ProcessNotSupportedError(FractionationError):
    """Exception raised when a process type is not supported by the calculator."""
    pass


class FractionationParameterError(FractionationError):
    """Exception raised for invalid fractionation parameters."""
    pass


class TemperatureError(FractionationError):
    """Exception raised for temperature validation errors."""
    pass


# Export base classes and exceptions
__all__ = [
    'IsotopeFractionationCalculator',
    'FractionationError',
    'ProcessNotSupportedError', 
    'FractionationParameterError',
    'TemperatureError'
]
