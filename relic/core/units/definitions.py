"""Core unit definitions and physical constants."""

from __future__ import annotations

from enum import Enum

from relic.utils.physical_constants import get_physical_constant

# Standard conditions come from the physical constants SSOT
# (relic.physical.constants.temperature.standard_temperature [K] and
# relic.physical.constants.pressure.standard_atmosphere [Pa]). A lookup failure
# propagates and fails the import; it is not replaced by None.
STANDARD_TEMPERATURE: float = float(get_physical_constant("standard_temperature"))
STANDARD_PRESSURE: float = float(get_physical_constant("standard_pressure"))


class UnitCategory(Enum):
    """Categories of physical units."""

    TEMPERATURE = "temperature"
    CONCENTRATION = "concentration"
    FLUX = "flux"
    TIME = "time"
    PRESSURE = "pressure"
    LENGTH = "length"
    MASS = "mass"
    ENERGY = "energy"
    DIMENSIONLESS = "dimensionless"


__all__ = [
    "STANDARD_PRESSURE",
    "STANDARD_TEMPERATURE",
    "UnitCategory",
]
