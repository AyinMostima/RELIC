"""Unit tests for data quality validators."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import xarray as xr

from relic.data.standards.unit_converter import UnitConverter, unit_converter


def _import_quality_module(monkeypatch):
    monkeypatch.setenv("RELIC_EXPERIMENTAL_DATA_QUALITY", "1")
    sys.modules.pop("relic.data.quality", None)
    return importlib.import_module("relic.data.quality")


def test_quality_validator_detects_range_violations(monkeypatch):
    quality_mod = _import_quality_module(monkeypatch)
    validator = quality_mod.QualityValidator(
        range_rules={"temperature": quality_mod.RangeRule(min_value=250.0, max_value=320.0)},
    )
    ds = xr.Dataset(
        {"temperature": ("time", np.array([260.0, 500.0]))},
        coords={"time": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[s]")},
    )
    ok, report = validator.check_data(ds)
    assert ok is False
    assert any(item["check"] == "range" for item in report["issues"])


def test_quality_validator_detects_temporal_inconsistency(monkeypatch):
    quality_mod = _import_quality_module(monkeypatch)
    validator = quality_mod.QualityValidator(max_time_gap_seconds=3600.0)
    ds = xr.Dataset(
        {"temperature": ("time", np.array([280.0, 281.0, 282.0]))},
        coords={
            "time": np.array(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T03:00:00",
                    "2026-01-01T02:00:00",
                ],
                dtype="datetime64[s]",
            )
        },
    )
    ok, report = validator.check_data(ds)
    assert ok is False
    checks = {item["check"] for item in report["issues"]}
    assert "temporal_consistency" in checks


def test_quality_validator_accepts_dimensionally_compatible_units(monkeypatch):
    quality_mod = _import_quality_module(monkeypatch)
    validator = quality_mod.QualityValidator(expected_units={"temperature": "K"})
    ds = xr.Dataset(
        {"temperature": ("time", np.array([20.0, 21.0]))},
        coords={"time": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[s]")},
    )
    ds["temperature"].attrs["units"] = "degC"

    ok, report = validator.check_data(ds)
    assert ok is True
    assert report["total_issues"] == 0


def test_quality_validator_reuses_unit_converter_owner_and_allows_injection(monkeypatch):
    """Default construction must reuse the canonical unit-conversion owner."""
    quality_mod = _import_quality_module(monkeypatch)

    default_validator = quality_mod.QualityValidator()
    assert default_validator._unit_converter is unit_converter

    injected = UnitConverter()
    injected_validator = quality_mod.QualityValidator(unit_converter=injected)
    assert injected_validator._unit_converter is injected
