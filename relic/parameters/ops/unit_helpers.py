from __future__ import annotations

import re
from typing import Any, Optional

from ...data.standards.cf_conventions import CFConventions
from ...data.standards.unit_converter import (
    ConversionContext,
    unit_converter as data_unit_converter,
)

_UNIT_DENOMINATOR_PATTERN = re.compile(r"([A-Za-z]+)(-?\d*)")


def normalize_unit_string(units: Optional[str]) -> Optional[str]:
    """Normalize informal unit strings (e.g., kg/m2/s) to CF-style exponents."""
    if units is None:
        return None
    cleaned = units.strip()
    cleaned = cleaned.replace(" DW", "").replace("DW ", "")
    if "/" not in cleaned:
        return cleaned
    parts = cleaned.split("/")
    numerator = parts[0].strip()
    denominators = parts[1:]
    normalized = numerator
    for denom in denominators:
        denom = denom.strip()
        if not denom:
            continue
        match = _UNIT_DENOMINATOR_PATTERN.match(denom)
        if match:
            symbol, power = match.groups()
            power = power or ""
            if power.startswith("-"):
                exponent = power
            else:
                exponent = f"-{power or '1'}"
            normalized += f" {symbol}{exponent}"
        else:
            normalized += f" {denom}"
    return normalized


def convert_units(value: Any, from_unit: str, to_unit: str) -> Any:
    """Convert units using the CF-1.8 unit converter."""
    return data_unit_converter.convert(
        value,
        "generic",
        from_unit,
        to_unit,
        ConversionContext(),
    )


def validate_unit(unit: str) -> bool:
    """Validate unit string against CF-1.8 conventions."""
    try:
        return bool(CFConventions.validate_units(None, unit))
    except (TypeError, ValueError):
        return isinstance(unit, str) and len(unit) > 0
