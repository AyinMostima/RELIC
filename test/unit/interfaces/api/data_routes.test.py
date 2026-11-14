"""Tests for scope-honest data API CF controls, responses, and failure propagation."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
import torch

from relic.interfaces.api.routes import data as data_routes
from relic.coupling.external.common.io import DataFormat
from relic.interfaces.api.schemas import (
    DataConvertRequest,
    DataReadRequest,
    DataWriteRequest,
)


class _OfflineDataIO:
    """DataIO stand-in whose storage backend fails every operation."""

    def get_file_info(self, path):
        raise OSError(f"storage offline: {path}")

    def read(self, path, **kwargs):
        raise OSError(f"storage offline: {path}")

    def write(self, source, output_path, **kwargs):
        raise OSError(f"storage offline: {output_path}")

    def convert_format(self, **kwargs):
        raise OSError(f"storage offline: {kwargs['source_path']}")


def _read_request(tmp_path, dtype="float32"):
    return DataReadRequest.model_construct(
        path=str(tmp_path / "input.nc"),
        variables=None,
        format="auto",
        chunks=None,
        to_torch=True,
        device="cpu",
        dtype=dtype,
    )


def _write_request(tmp_path):
    return DataWriteRequest.model_construct(
        input_path=None,
        output_path=str(tmp_path / "output.nc"),
        data={"signal": [1.0]},
        format="netcdf4",
        compression="none",
        compression_level=4,
        validate_cf_variable_metadata=True,
    )


def _convert_request(tmp_path):
    return DataConvertRequest.model_construct(
        input_path=str(tmp_path / "input.nc"),
        output_path=str(tmp_path / "output.zarr"),
        target_format="zarr",
        variables=None,
        compression="zstd",
        preserve_metadata=True,
    )


@pytest.mark.parametrize(
    ("route_name", "request_factory"),
    [
        ("inspect_data_file", _read_request),
        ("read_data", _read_request),
        ("write_data", _write_request),
        ("convert_data_format", _convert_request),
    ],
)
def test_data_route_failure_reaches_the_api_error_boundary(
    tmp_path, monkeypatch, route_name, request_factory
):
    """A failing data route raises; the app error handler sets the HTTP error status."""
    monkeypatch.setattr(data_routes, "_get_data_io", lambda: _OfflineDataIO())
    route = getattr(data_routes, route_name)

    with pytest.raises(OSError, match="storage offline"):
        asyncio.run(route(request_factory(tmp_path)))


def test_data_read_rejects_unknown_torch_dtype(tmp_path, monkeypatch):
    """An unknown dtype name must not silently become float32."""
    calls = []

    class DataIO:
        def read(self, path, **kwargs):
            calls.append(kwargs)
            return {"signal": torch.zeros(2)}

    monkeypatch.setattr(data_routes, "_get_data_io", lambda: DataIO())

    with pytest.raises(ValueError, match="flaot32"):
        asyncio.run(data_routes.read_data(_read_request(tmp_path, dtype="flaot32")))
    assert calls == []


def test_payload_summary_keeps_tensor_metadata_without_a_call_time_torch_import(monkeypatch):
    """Tensor metadata is summarized from the module-level torch dependency."""
    monkeypatch.setitem(sys.modules, "torch", None)

    summary = data_routes._summarize_payload({"signal": torch.zeros(2, 3, dtype=torch.float32)})

    assert summary["variables"]["signal"]["shape"] == [2, 3]
    assert summary["variables"]["signal"]["dtype"] == "torch.float32"


def test_payload_summary_rejects_non_integer_dimension_sizes():
    """Dimension sizes that are not integers are an invalid payload, not raw data."""
    payload = SimpleNamespace(dims={"time": "unknown"})

    with pytest.raises(ValueError, match="unknown"):
        data_routes._summarize_payload(payload)


def test_data_write_route_requests_only_variable_metadata_sync(tmp_path, monkeypatch):
    calls = []

    class DataIO:
        def write(self, source, output_path, **kwargs):
            calls.append((source, output_path, kwargs))
            return SimpleNamespace(
                file_size=12,
                compression_ratio=None,
                cf_compliant=False,
            )

    monkeypatch.setattr(data_routes, "_get_data_io", lambda: DataIO())
    request = DataWriteRequest.model_construct(
        input_path=None,
        output_path=str(tmp_path / "output.nc"),
        data={"signal": [1.0]},
        format="netcdf4",
        compression="none",
        compression_level=4,
        validate_cf_variable_metadata=True,
    )

    response = asyncio.run(data_routes.write_data(request))

    assert response.status == "success"
    assert calls[0][2]["cf_sync"] is True
    assert "cf_compliant" not in calls[0][2]
    assert response.data["cf_validation_scope"] == (
        "single_variable_standard_name_and_units"
    )
    assert response.data["dataset_conformance_checked"] is False
    assert response.data["cf_compliant"] is False


def test_data_api_schemas_do_not_expose_positive_dataset_cf_switch():
    assert "cf_compliant" not in DataWriteRequest.model_fields
    assert "validate_cf_variable_metadata" in DataWriteRequest.model_fields



def test_data_write_route_requires_the_write_result_fields(tmp_path, monkeypatch):
    """A write result without file_size is a defect, not a zero-byte file."""

    class DataIO:
        def write(self, source, output_path, **kwargs):
            return SimpleNamespace(compression_ratio=None)

    monkeypatch.setattr(data_routes, "_get_data_io", lambda: DataIO())

    with pytest.raises(AttributeError, match="file_size"):
        asyncio.run(data_routes.write_data(_write_request(tmp_path)))


def test_data_convert_route_requires_the_read_result_fields(tmp_path, monkeypatch):
    class DataIO:
        def convert_format(self, **kwargs):
            return SimpleNamespace(), SimpleNamespace(write_time=0.5, file_size=10)

    monkeypatch.setattr(data_routes, "_get_data_io", lambda: DataIO())

    with pytest.raises(AttributeError, match="load_time"):
        asyncio.run(data_routes.convert_data_format(_convert_request(tmp_path)))


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("auto", DataFormat.AUTO), ("zarr", DataFormat.ZARR)],
)
def test_data_read_passes_the_requested_format_explicitly(tmp_path, monkeypatch, requested, expected):
    """The boundary default "auto" reaches the reader as an explicit format, never by omission."""
    calls = []

    class DataIO:
        def read(self, path, **kwargs):
            calls.append(kwargs)
            return {"signal": torch.zeros(2)}

    monkeypatch.setattr(data_routes, "_get_data_io", lambda: DataIO())
    request = _read_request(tmp_path)
    request.format = requested

    asyncio.run(data_routes.read_data(request))

    assert calls[0]["format"] is expected


def test_data_read_format_is_not_optional():
    assert DataReadRequest.model_fields["format"].default == "auto"
    with pytest.raises(ValueError):
        DataReadRequest.model_validate({"path": __file__, "format": None})


def test_data_convert_passes_the_requested_compression(tmp_path, monkeypatch):
    calls = []

    class DataIO:
        def convert_format(self, **kwargs):
            calls.append(kwargs)
            return (
                SimpleNamespace(load_time=0.1),
                SimpleNamespace(write_time=0.2, file_size=5),
            )

    monkeypatch.setattr(data_routes, "_get_data_io", lambda: DataIO())
    request = _convert_request(tmp_path)
    request.compression = "gzip"

    asyncio.run(data_routes.convert_data_format(request))

    assert calls[0]["compression"] == "gzip"
