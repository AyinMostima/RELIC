"""Owner tests for SWATAdapter.forward main-dataset selection (ledger D65/D69).

Only the SWAT 2012 output.hru text layout is read, so SWATConfig declares output_level="hru"
and the opener stores the table under that key. forward reads the dataset of the configured
output level through the single adapter read path (schema mapping, declared transforms, unit
conversion): outputs are catalog variables. Other keys are not selected; a miss is logged.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
import xarray as xr

from relic.coupling.external.hydrological_models.swat_adapter import SWATAdapter, SWATConfig
from relic.core.state import UpdateItem


# The SWAT reader keeps its CF time coordinate numeric (days since IYR-01-01 of the declared
# calendar); forward places the records at the simulation time with these units and calendar.
_TIME = ("time", np.arange(3.0), {"units": "days since 2000-01-01 00:00:00", "calendar": "noleap"})
_AT = {"calendar_type": "noleap", "time_origin": "2000-01-01T00:00:00"}


def _soil_temperature_dataset(celsius: float) -> xr.Dataset:
    """Three daily records of the schema-mapped SWAT column SOL_TMP [degC]."""
    return xr.Dataset(
        {"SOL_TMP": ("time", np.array([celsius, celsius + 1.0, celsius + 2.0], dtype=np.float64), {"units": "degC"})},
        coords={"time": _TIME},
    )


def _adapter() -> SWATAdapter:
    return SWATAdapter(
        config=SWATConfig(
            root_path=".", output_level="hru", time_interpolation="lower", provided_variables=["soil_temperature"]
        )
    )


def _kelvin(updates: dict) -> float:
    return float(updates["soil_temperature"].value)


def test_forward_reads_the_configured_output_level() -> None:
    adapter = _adapter()
    adapter.current_datasets = {"hru": _soil_temperature_dataset(3.0)}
    updates = adapter.forward({}, time=0.0, dt=86400.0, **_AT)
    assert set(updates) == {"soil_temperature"}
    assert all(isinstance(item, UpdateItem) for item in updates.values())
    assert _kelvin(updates) == pytest.approx(276.15)
    assert updates["soil_temperature"].value.dtype == torch.float32
    # The record of the second day at t = 1 day.
    assert _kelvin(adapter.forward({}, time=86400.0, dt=86400.0, **_AT)) == pytest.approx(277.15)


def test_adapter_needs_a_declared_config() -> None:
    with pytest.raises(ValueError, match="SWATConfig"):
        SWATAdapter(config=None)


def _forward_with_warnings(adapter: SWATAdapter) -> tuple[dict, list[logging.LogRecord]]:
    # Capture on the adapter logger itself so the assertion does not depend on the session
    # logging configuration (root handlers and levels differ between pytest invocations).
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture(level=logging.WARNING)
    adapter.logger.addHandler(handler)
    try:
        updates = adapter.forward({}, time=0.0, dt=86400.0, **_AT)
    finally:
        adapter.logger.removeHandler(handler)
    return updates, captured


def test_forward_raises_when_the_output_level_dataset_carries_no_mapped_columns() -> None:
    adapter = _adapter()
    adapter.current_datasets = {
        "hru": xr.Dataset({"VOLUME": ("time", np.array([1.0, 2.0]))}, coords={"time": np.arange(2.0)})
    }
    # The provided variables cannot be read: forward raises instead of returning no forcing.
    with pytest.raises(ValueError, match="mapped SWAT columns"):
        _forward_with_warnings(adapter)


def test_forward_does_not_select_other_dataset_keys() -> None:
    adapter = _adapter()
    adapter.current_datasets = {"reach": _soil_temperature_dataset(12.0), "output.rch": _soil_temperature_dataset(7.0)}
    with pytest.raises(ValueError, match="mapped SWAT columns"):
        _forward_with_warnings(adapter)
