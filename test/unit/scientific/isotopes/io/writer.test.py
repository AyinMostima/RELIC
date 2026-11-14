"""Time-coordinate SSOT tests for the isotope NetCDF writer."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import relic.scientific.isotopes.io.writer as writer_module
from relic.scientific.isotopes.io.writer import IsotopeNetCDFWriter
from relic.utils.time_constants import SECONDS_PER_DAY, SECONDS_PER_HOUR


def _writer(time_units: str) -> IsotopeNetCDFWriter:
    writer = object.__new__(IsotopeNetCDFWriter)
    writer.output_config = SimpleNamespace(time_units=time_units)
    return writer


def test_convert_time_values_uses_declared_coordinate_unit() -> None:
    times = [0.0, float(SECONDS_PER_HOUR), float(SECONDS_PER_DAY)]

    np.testing.assert_allclose(
        _writer("hours since 2000-01-01")._convert_time_values(times),
        [0.0, 1.0, 24.0],
    )
    np.testing.assert_allclose(
        _writer("days since 2000-01-01")._convert_time_values(times),
        [0.0, 1.0 / 24.0, 1.0],
    )
    np.testing.assert_allclose(
        _writer("seconds since 2000-01-01")._convert_time_values(times),
        times,
    )


def test_hour_conversion_is_bound_to_seconds_per_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        writer_module,
        "SECONDS_PER_HOUR",
        float(SECONDS_PER_HOUR) / 2.0,
        raising=False,
    )

    converted = _writer("hours since 2000-01-01")._convert_time_values(
        [float(SECONDS_PER_HOUR)]
    )

    np.testing.assert_allclose(converted, [2.0])


def test_time_conversion_has_no_local_hour_literal() -> None:
    source = inspect.getsource(IsotopeNetCDFWriter._convert_time_values)

    assert "3600" not in source
    assert "SECONDS_PER_HOUR" in source
    assert "SECONDS_PER_DAY" in source


def _buffer() -> writer_module.IsotopeDataBuffer:
    buffer = object.__new__(writer_module.IsotopeDataBuffer)
    buffer.fill_value = -9999.0
    return buffer


def test_aggregation_rejects_an_unknown_method() -> None:
    """An unknown aggregation method raises instead of silently taking the mean."""
    with pytest.raises(writer_module.AggregationError, match="bogus"):
        _buffer()._apply_aggregation([1.0, 2.0], "bogus")


def test_aggregation_errors_propagate() -> None:
    """A failing aggregation raises instead of returning the fill value."""

    class Unsummable(float):
        def __radd__(self, other):
            raise TypeError("cannot add")

    with pytest.raises(TypeError, match="cannot add"):
        _buffer()._apply_aggregation(
            [Unsummable(1.0)], writer_module.AggregationMethod.DAILY_MEAN
        )


def test_output_validation_records_only_unreadable_files(tmp_path) -> None:
    """validate_output records an unreadable file as a validation error; a defect of the
    writer configuration propagates instead of being reported as a finding."""
    import netCDF4

    path = tmp_path / "out.nc"
    with netCDF4.Dataset(str(path), "w") as dataset:
        dataset.createDimension("time", None)
    writer = object.__new__(IsotopeNetCDFWriter)
    writer.output_file = path
    writer.output_config = None
    with pytest.raises(AttributeError):
        writer.validate_output()

    garbage = tmp_path / "garbage.nc"
    garbage.write_bytes(b"not a netcdf file")
    writer.output_file = garbage
    errors = writer.validate_output()
    assert len(errors) == 1
    assert "unreadable" in errors[0]
