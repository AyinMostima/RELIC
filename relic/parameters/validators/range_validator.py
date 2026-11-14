"""
Range Validator - Physical bounds checking for parameters

This module implements range validation for scientific parameters,
ensuring values stay within physically meaningful bounds.

Mathematical Framework:
    Range constraint: theta in [theta_min, theta_max]
    Validation function: V(theta) = (theta_min <= theta <= theta_max)
    
References:
    - Physical parameter bounds from literature
    - CF-1.8 standard ranges where applicable
"""

from typing import Any, Optional, Tuple
import logging

from ..manager import ParameterSpec, ValidationErrors
from ..types import ValidationResult

logger = logging.getLogger(__name__)


class RangeValidator:
    """
    Validates parameter values against specified bounds.
    
    Ensures parameters stay within physically meaningful ranges
    based on their scientific definitions and constraints.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize range validator.
        
        Parameters:
            strict : bool
                Whether to enforce strict bounds checking
        """
        self.strict = strict
    
    def validate(self, spec: ParameterSpec, value: Any = None) -> ValidationResult:
        """
        Validate parameter value against bounds.
        
        Parameters:
            spec : ParameterSpec
                Parameter specification with bounds
            value : Any
                Value to validate (if None, uses spec.value)
                
        Returns:
            ValidationResult: Validation result with is_valid and errors
        """
        errors = []
        warnings = []
        
        # Use spec.value if no value provided
        if value is None:
            value = spec.value
        
        # Skip if no bounds specified
        if spec.bounds == (-float('inf'), float('inf')):
            return ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Convert value to float for comparison
        try:
            if hasattr(value, 'item'):  # Handle tensors
                numeric_value = float(value.item())
            else:
                numeric_value = float(value)
        except (TypeError, ValueError) as e:
            errors.append(f"Cannot convert {spec.key} value to numeric: {e}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        min_val, max_val = spec.bounds
        
        # Check lower bound
        if numeric_value < min_val:
            if self.strict:
                errors.append(
                    f"Parameter {spec.key} = {numeric_value} below minimum {min_val} {spec.units}"
                )
            else:
                warnings.append(
                    f"Parameter {spec.key} = {numeric_value} below minimum {min_val} {spec.units}"
                )
        
        # Check upper bound
        if numeric_value > max_val:
            if self.strict:
                errors.append(
                    f"Parameter {spec.key} = {numeric_value} above maximum {max_val} {spec.units}"
                )
            else:
                warnings.append(
                    f"Parameter {spec.key} = {numeric_value} above maximum {max_val} {spec.units}"
                )
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def suggest_bounds(self, param_key: str) -> Optional[Tuple[float, float]]:
        """
        Suggest reasonable bounds for common parameter types.
        
        Parameters:
            param_key : str
                Parameter key to suggest bounds for
                
        Returns:
            Optional[Tuple[float, float]]: Suggested (min, max) bounds
        """
        key_lower = param_key.lower()
        
        # Temperature parameters
        if 'temperature' in key_lower or 'temp' in key_lower:
            if 'kelvin' in key_lower or '_k' in key_lower:
                return (0.0, 373.15)  # 0K to 100degC
            elif 'celsius' in key_lower or '_c' in key_lower:
                return (-273.15, 100.0)  # Absolute zero to boiling
            else:
                return (200.0, 350.0)  # Assume Kelvin, Earth range
        
        # Photosynthesis parameters
        if 'vcmax' in key_lower:
            return (10.0, 200.0)  # mumol m^2 s^-1
        if 'jmax' in key_lower:
            return (20.0, 400.0)  # mumol m^2 s^-1
        if 'tpu' in key_lower:
            return (5.0, 50.0)    # mumol m^2 s^-1
        
        # Respiration parameters
        if 'q10' in key_lower:
            return (1.5, 3.5)     # Dimensionless
        if 'respiration' in key_lower and 'rate' in key_lower:
            return (0.0, 10.0)    # mumol m^2 s^-1
        
        # Hydraulic parameters
        if 'conductivity' in key_lower or 'ksat' in key_lower:
            return (1e-10, 1e-3)  # m/s
        if 'theta' in key_lower or 'moisture' in key_lower:
            return (0.0, 1.0)     # Volumetric fraction
        
        # Allocation fractions
        if 'fraction' in key_lower or 'ratio' in key_lower:
            return (0.0, 1.0)     # Dimensionless fraction
        
        # Concentration parameters
        if 'concentration' in key_lower or 'density' in key_lower:
            return (0.0, float('inf'))  # Non-negative
        
        # Rate parameters
        if 'rate' in key_lower or 'flux' in key_lower:
            return (0.0, float('inf'))  # Non-negative rates
        
        # Default: no specific bounds
        return None


class PhysicalConsistencyValidator:
    """
    Validates physical consistency between related parameters.
    
    Ensures that parameter relationships maintain physical consistency,
    such as water content relationships and sum-to-one constraints.
    """
    
    def validate_water_parameters(self, params) -> ValidationResult:
        """
        Validate water content parameter relationships.
        
        Ensures: theta_wp < theta_fc < theta_sat
        where wp=wilting point, fc=field capacity, sat=saturation
        
        Parameters:
            params : list or dict
                List of ParameterSpec objects or dictionary of parameter key-value pairs
                
        Returns:
            ValidationResult: Validation result with is_valid and errors
        """
        errors = []
        warnings = []
        
        # Extract water content parameters
        theta_wp = None
        theta_fc = None
        theta_sat = None
        
        # Handle list of ParameterSpec objects
        if isinstance(params, list):
            param_dict = {}
            for spec in params:
                param_dict[spec.key] = spec.value
            params = param_dict
        
        for key, value in params.items():
            if 'theta_wp' in key or 'wilting_point' in key:
                theta_wp = float(value) if not hasattr(value, 'item') else float(value.item())
            elif 'theta_fc' in key or 'field_capacity' in key:
                theta_fc = float(value) if not hasattr(value, 'item') else float(value.item())
            elif 'theta_sat' in key or 'saturation' in key or 'porosity' in key:
                theta_sat = float(value) if not hasattr(value, 'item') else float(value.item())
        
        # Check relationships
        if theta_wp is not None and theta_fc is not None:
            if theta_wp >= theta_fc:
                errors.append(
                    f"Wilting point ({theta_wp}) must be less than field capacity ({theta_fc})"
                )
        
        if theta_fc is not None and theta_sat is not None:
            if theta_fc >= theta_sat:
                errors.append(
                    f"Field capacity ({theta_fc}) must be less than saturation ({theta_sat})"
                )
        
        if theta_wp is not None and theta_sat is not None:
            if theta_wp >= theta_sat:
                errors.append(
                    f"Wilting point ({theta_wp}) must be less than saturation ({theta_sat})"
                )
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    @staticmethod
    def _as_number(key: str, value: Any, errors: list) -> Optional[float]:
        """Return a checked parameter value as float, or record why it is not numeric.

        float() raises TypeError for None or a container and ValueError for text;
        .item() raises RuntimeError (torch) or ValueError (numpy) for more than one
        element. The recorded error makes the validation result invalid.
        """
        try:
            return float(value) if not hasattr(value, 'item') else float(value.item())
        except (TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"Parameter {key} is not a numeric scalar: {value!r} ({exc})")
            return None

    def validate_allocation_fractions(self, params: dict) -> ValidationErrors:
        """
        Validate that allocation fractions sum to 1.
        
        Parameters:
            params : dict
                Dictionary of parameter key-value pairs
                
        Returns:
            ValidationErrors: List of validation errors
        """
        errors = []
        warnings = []
        
        # Collect allocation fractions; a checked fraction that is not a numeric
        # scalar is a validation error, not a skipped entry.
        allocation_params = {}
        for key, value in params.items():
            if 'allocation' in key and 'fraction' in key:
                numeric_value = self._as_number(key, value, errors)
                if numeric_value is not None:
                    allocation_params[key] = numeric_value
        
        if allocation_params:
            total = sum(allocation_params.values())
            if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
                errors.append(
                    f"Allocation fractions sum to {total:.3f}, should sum to 1.0"
                )
                
            for key, value in allocation_params.items():
                if value < 0:
                    errors.append(f"Allocation fraction {key} cannot be negative: {value}")
                if value > 1:
                    errors.append(f"Allocation fraction {key} cannot exceed 1: {value}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def validate_temperature_responses(self, params: dict) -> ValidationErrors:
        """
        Validate temperature response parameters.
        
        Ensures optimal temperatures are within reasonable ranges
        and activation energies are positive.
        
        Parameters:
            params : dict
                Dictionary of parameter key-value pairs
                
        Returns:
            ValidationErrors: List of validation errors
        """
        errors = []
        warnings = []
        
        for key, value in params.items():
            is_temperature_optimum = 'temp_opt' in key or 'temperature_opt' in key
            is_activation_energy = 'activation_energy' in key or '_ea' in key
            if not (is_temperature_optimum or is_activation_energy):
                continue
            # A checked parameter that is not a numeric scalar is a validation error.
            numeric_value = self._as_number(key, value, errors)
            if numeric_value is None:
                continue

            # Check optimal temperatures
            if is_temperature_optimum:
                if 'kelvin' in key.lower() or '_k' in key:
                    if numeric_value < 273.15 or numeric_value > 323.15:
                        errors.append(
                            f"Optimal temperature {key} = {numeric_value}K outside reasonable range [273-323K]"
                        )
                elif 'celsius' in key.lower() or '_c' in key:
                    if numeric_value < 0 or numeric_value > 50:
                        errors.append(
                            f"Optimal temperature {key} = {numeric_value}degC outside reasonable range [0-50degC]"
                        )

            # Check activation energies
            if is_activation_energy:
                if numeric_value <= 0:
                    errors.append(
                        f"Activation energy {key} must be positive: {numeric_value}"
                    )
                if numeric_value > 200000:  # J/mol
                    errors.append(
                        f"Activation energy {key} = {numeric_value} J/mol unusually high"
                    )
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


# Export validator classes
__all__ = [
    'RangeValidator',
    'PhysicalConsistencyValidator'
]
