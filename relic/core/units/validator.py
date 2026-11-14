"""Unit validation helpers built on top of UnitSystem."""

from __future__ import annotations

import warnings
from typing import Any, Dict

from .system import UnitSystem


class StateUnitValidator:
    """
    Validator for unit consistency across modules.
    Can be integrated with FunctorBase for automatic validation.
    """

    def __init__(self, strict: bool = True):
        """
        Initialize unit validator.

        Args:
            strict: If True, raise errors on unit violations. If False, warn only.
        """
        self.strict = strict
        self.unit_system = UnitSystem

    def validate_input(
        self,
        state: Dict[str, Any],
        expected_units: Dict[str, str],
    ) -> bool:
        """
        Validate input state units match expected units.

        Args:
            state: Input state dictionary
            expected_units: Expected units for each variable

        Returns:
            True if valid, raises or warns if not
        """
        issues = []
        for var, expected_unit in expected_units.items():
            if var not in state:
                continue
            unit_key = f"{var}_units"
            if unit_key not in state:
                issues.append(f"{var} missing unit metadata")
                continue
            actual_unit = state[unit_key]
            if actual_unit != expected_unit:
                issues.append(
                    f"{var} has unit {actual_unit}, expected {expected_unit}"
                )

        is_valid = len(issues) == 0

        if not is_valid:
            message = "Unit validation failed:\n" + "\n".join(issues)
            if self.strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)

        return is_valid

    def validate_output(
        self,
        output: Dict[str, Any],
        expected_units: Dict[str, str],
    ) -> bool:
        """
        Validate output units match expected units.

        Args:
            output: Output dictionary
            expected_units: Expected units for each variable

        Returns:
            True if valid, raises or warns if not
        """
        for var, expected_unit in expected_units.items():
            if var in output:
                unit_key = f"{var}_units"
                if unit_key in output:
                    actual_unit = output[unit_key]
                    if actual_unit != expected_unit:
                        message = (
                            f"Output {var} has unit {actual_unit}, expected "
                            f"{expected_unit}"
                        )
                        if self.strict:
                            raise ValueError(message)
                        warnings.warn(message)
                        return False
                else:
                    message = f"Output {var} missing unit metadata"
                    if self.strict:
                        raise ValueError(message)
                    warnings.warn(message)
                    return False

        return True

__all__ = ["StateUnitValidator"]
