"""Regression guards for hardcoded temperature and gas constant fallbacks."""

from __future__ import annotations

from pathlib import Path


def test_no_hardcoded_temperature_and_gas_constant_fallbacks_in_bridges() -> None:
    """Bridge temperature response must use parameter system/physical constants."""
    target = Path("relic\\core\\functor\\functor_bridges.py")
    text = target.read_text(encoding="utf-8", errors="ignore")

    assert "gas_constant = 8.314462618" not in text
    assert "ref_temp = 298.15" not in text


def test_no_hardcoded_q10_default_in_biological_abstraction_init() -> None:
    """Biological abstraction should not hardcode Q10 defaults in initializer."""
    target = Path("relic\\core\\abstractions.py")
    text = target.read_text(encoding="utf-8", errors="ignore")

    assert "self.q10_value: float = 2.0" not in text


def test_no_hardcoded_arrhenius_constants_in_compute_benchmarks() -> None:
    """Benchmark kernels must fetch Arrhenius constants via SSOT utilities."""
    target = Path("relic\\computation\\benchmarks.py")
    text = target.read_text(encoding="utf-8", errors="ignore")

    assert "temp + 273.15" not in text
    assert "(temp_k - 298.15)" not in text
    assert "* 8.314 *" not in text
