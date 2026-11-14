"""Tests for lazy I/O quality metric aggregation."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import relic.coupling.external.common.io as io_module
from relic.coupling.external.common.io import IOConfig, IOHandler


def test_dask_missing_counts_are_computed_in_one_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = xr.Dataset(
        {
            "a": (
                ("x",),
                da.from_array(np.array([1.0, np.nan]), chunks=(1,)),
            ),
            "b": (
                ("x",),
                da.from_array(np.array([np.nan, 2.0]), chunks=(1,)),
            ),
        }
    )
    handler = IOHandler(
        IOConfig(
            chunks={"x": 1},
            parallel=True,
            validate_checksums=False,
            strict_cf=False,
        )
    )
    original_compute = io_module.dask.compute
    calls = []

    def counted_compute(*values, **kwargs):
        calls.append(len(values))
        return original_compute(*values, **kwargs)

    monkeypatch.setattr(io_module.dask, "compute", counted_compute)

    metrics = handler._compute_quality_metrics(dataset, Path("input.nc"))

    assert calls == [2]
    assert metrics["missing_data_fraction"] == pytest.approx(0.5)
