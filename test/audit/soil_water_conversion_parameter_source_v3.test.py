"""
Audit gate to prevent hardcoded soil texture parameter tables in coupling conversions.
"""

from __future__ import annotations

from pathlib import Path

import relic.coupling.scales.soil_water_conversions as swc


def test_soil_water_conversions_has_no_hardcoded_soil_params_table() -> None:
    """
    References:
    - van Genuchten (1980), parameterized soil-water retention equations.
    - RELIC Parameter SSOT policy in architecture overview.

    Mathematical expression:
    - m = 1 - 1/n
    - Se = [1 + (alpha * |psi|)^n]^(-m)
    - theta = theta_sat * Se

    Parameter meanings:
    - theta_sat: saturated water content.
    - alpha: inverse air-entry parameter.
    - n: pore-size distribution parameter.
    """
    assert not hasattr(swc, "SOIL_PARAMS")

    source = Path("relic/coupling/scales/soil_water_conversions.py").read_text(
        encoding="utf-8"
    )
    assert "SOIL_PARAMS" not in source

