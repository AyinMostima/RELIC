"""
Unit Validator compatibility facade

This module delegates parameter-unit validation to ParameterValidator, which
uses the canonical dimensionality owner in relic.data.standards.

Mathematical Framework:
    Dimensional analysis: [theta] = M^a L^b T^c K^d ...
    Unit conversion: theta' = alpha theta + beta
    
References:
    - CF Conventions: https://cfconventions.org/
    - UDUNITS-2 specification
"""

from ..manager import ParameterSpec
from ..types import ValidationLevel, ValidationResult
from ..validation import ParameterValidator


class UnitValidator:
    """
    Compatibility facade over the canonical parameter validator.
    """
    
    def __init__(self):
        """Initialize the compatibility facade over ParameterValidator."""
        self._validator = ParameterValidator(level=ValidationLevel.STRICT)
    
    def validate(self, spec: ParameterSpec) -> ValidationResult:
        """
        Validate parameter units.
        
        Parameters:
            spec : ParameterSpec
                Parameter specification with units
                
        Returns:
            ValidationResult: Validation result with is_valid and errors
        """
        return self._validator.validate_value(spec.key, spec.value, spec)


__all__ = ['UnitValidator']
