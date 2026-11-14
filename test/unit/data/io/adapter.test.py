from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest
import torch
import xarray as xr

from relic.core.variable_mapper import VariableNameMapper, variable_name_mapper
from relic.data.io import adapter as adapter_module
from relic.data.io.adapter import DataIO, IOOperation
from relic.data.io.reader import DataReader, ReadResult
from relic.data.io.writer import DataWriter, WriteResult
from relic.data.standards.cf_conventions import CFConventions


class StubCFConventions:
    def get_standard_name(self, name: str) -> str:
        return {
            "temp": "air_temperature",
            "precip": "precipitation_flux",
        }.get(name, name)

    def get_standard_units(self, standard_name: str) -> str | None:
        return {
            "air_temperature": "K",
            "precipitation_flux": "kg m-2 s-1",
        }.get(standard_name)


def _mapper(aliases: dict[str, str] | None = None) -> VariableNameMapper:
    payloads = {
        "coupling.external.variables.variable_definitions": {},
        "coupling.scales.defaults.variables": {},
        "coupling.external.variables.aliases": aliases,
    }
    manager = Mock()
    manager.get.side_effect = lambda key, **_: payloads[key]
    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(manager)
    return mapper


def _data_io(mapper: VariableNameMapper) -> DataIO:
    data_io = object.__new__(DataIO)
    data_io._variable_mapper = mapper
    data_io.cf_conventions = CFConventions()
    return data_io


def test_data_io_reuses_shared_variable_mapper(monkeypatch) -> None:
    monkeypatch.setattr(adapter_module, "DataReader", Mock())
    monkeypatch.setattr(adapter_module, "DataWriter", Mock())
    monkeypatch.setattr(adapter_module, "CFConventions", Mock())

    data_io = DataIO()

    assert data_io._variable_mapper is variable_name_mapper


def test_standardize_identity_mapping_preserves_missing_optional_cf_metadata() -> None:
    data_io = _data_io(_mapper())
    source = xr.Dataset({"temp": ("time", [280.0, 281.0])})

    result = data_io.standardize_for_relic(source)

    assert list(result.data_vars) == ["temp"]
    assert result["temp"].attrs == {}
    assert source["temp"].attrs == {}
    assert result is not source


def test_standardize_maps_standard_name_to_internal_module_alias() -> None:
    data_io = _data_io(_mapper({"temp": "air_temperature"}))
    source = xr.Dataset({"air_temperature": ("time", [280.0])})
    source["air_temperature"].attrs = {
        "standard_name": "air_temperature",
        "units": "degree_Celsius",
    }

    result = data_io.standardize_for_relic(source)

    assert list(result.data_vars) == ["temp"]
    assert result["temp"].attrs["standard_name"] == "air_temperature"
    assert result["temp"].attrs["units"] == "degree_Celsius"


@pytest.mark.parametrize(
    "attrs",
    [
        {"standard_name": "totally_fake_cf_name", "units": "K"},
        {"standard_name": "air_temperature", "units": "Pa"},
        {"standard_name": "air_temperature"},
        {"units": "custom_unit"},
    ],
)
def test_standardize_rejects_invalid_explicit_metadata_before_rename(attrs) -> None:
    data_io = _data_io(_mapper({"temp": "source_signal"}))
    source = xr.Dataset({"source_signal": ("time", [1.0])})
    source["source_signal"].attrs = dict(attrs)

    with pytest.raises(ValueError, match="metadata"):
        data_io.standardize_for_relic(source)

    assert list(source.data_vars) == ["source_signal"]
    assert source["source_signal"].attrs == attrs


def test_standardize_does_not_overwrite_existing_internal_variable() -> None:
    data_io = _data_io(_mapper({"temp": "air_temperature"}))
    source = xr.Dataset(
        {
            "air_temperature": ("time", [280.0]),
            "temp": ("time", [285.0]),
        }
    )

    result = data_io.standardize_for_relic(source)

    assert set(result.data_vars) == {"air_temperature", "temp"}
    assert result["air_temperature"].item() == 280.0
    assert result["temp"].item() == 285.0


def test_standardize_only_processes_requested_variables() -> None:
    data_io = _data_io(
        _mapper(
            {
                "temp": "air_temperature",
                "precip": "precipitation_flux",
            }
        )
    )
    source = xr.Dataset(
        {
            "air_temperature": ("time", [280.0]),
            "precipitation_flux": ("time", [1.0e-5]),
        }
    )

    result = data_io.standardize_for_relic(
        source,
        target_variables=["air_temperature"],
    )

    assert set(result.data_vars) == {"temp", "precipitation_flux"}


def _io_with_mocks() -> tuple[DataIO, Mock, Mock]:
    data_io = object.__new__(DataIO)
    reader = create_autospec(DataReader, instance=True)
    writer = create_autospec(DataWriter, instance=True)
    reader.io_handler = Mock()
    data_io.reader = reader
    data_io.writer = writer
    data_io._performance_history = []
    data_io._variable_mapper = _mapper()
    data_io.cf_conventions = StubCFConventions()
    return data_io, reader, writer


def _read_result(data: xr.Dataset | dict[str, torch.Tensor]) -> ReadResult:
    return ReadResult(
        data=data,
        metadata={},
        load_time=0.25,
        file_size=2 * 1024 * 1024,
    )


def _write_result(path: str = "output.nc") -> WriteResult:
    return WriteResult(
        output_path=Path(path),
        file_size=1024 * 1024,
        write_time=0.5,
        compression_ratio=2.0,
    )


def test_read_returns_data_and_records_performance() -> None:
    data_io, reader, _ = _io_with_mocks()
    dataset = xr.Dataset({"temperature": ("time", [280.0])})
    reader.read_dataset.return_value = _read_result(dataset)

    returned = data_io.read("input.nc", chunks={"time": 1})

    assert returned is dataset
    reader.read_dataset.assert_called_once_with("input.nc", chunks={"time": 1})
    assert len(data_io._performance_history) == 1
    metric = data_io._performance_history[0]
    assert metric.operation is IOOperation.READ
    assert metric.throughput_mbps == pytest.approx(8.0)


def test_read_warns_and_preserves_reader_exception() -> None:
    data_io, reader, _ = _io_with_mocks()
    reader.read_dataset.side_effect = OSError("unreadable")

    with pytest.warns(UserWarning, match="Read operation failed"):
        with pytest.raises(OSError, match="unreadable"):
            data_io.read("missing.nc")


def test_read_variables_supports_dataset_and_tensor_results() -> None:
    data_io, reader, _ = _io_with_mocks()
    dataset = xr.Dataset(
        {
            "temperature": ("time", [280.0]),
            "precipitation": ("time", [1.0]),
        }
    )
    reader.read_dataset.return_value = _read_result(dataset)

    selected = data_io.read_variables(
        "input.nc",
        ["temperature", "absent"],
        decode_times=False,
    )

    assert list(selected) == ["temperature"]
    reader.read_dataset.assert_called_once_with(
        "input.nc",
        decode_times=False,
        variables=["temperature", "absent"],
    )

    tensors = {"temperature": torch.tensor([280.0])}
    reader.read_dataset.reset_mock()
    reader.read_dataset.return_value = _read_result(tensors)
    assert data_io.read_variables("input.nc", ["temperature"]) is tensors


def test_read_tensors_delegates_device_and_dtype() -> None:
    data_io, reader, _ = _io_with_mocks()
    expected = {"temperature": torch.tensor([280.0], dtype=torch.float64)}
    reader.read_tensors.return_value = expected

    returned = data_io.read_tensors(
        "input.nc",
        ["temperature"],
        device="cpu",
        dtype=torch.float64,
    )

    assert returned is expected
    reader.read_tensors.assert_called_once_with(
        "input.nc",
        ["temperature"],
        torch_device="cpu",
        torch_dtype=torch.float64,
    )


def test_write_tensor_dictionary_consumes_template_once() -> None:
    data_io, _, writer = _io_with_mocks()
    writer.write_tensors.return_value = _write_result()
    tensors = {"temperature": torch.tensor([280.0])}
    template = xr.Dataset(coords={"time": [0]})

    returned = data_io.write(
        tensors,
        "output.nc",
        template=template,
        compression="zlib",
    )

    assert returned is writer.write_tensors.return_value
    writer.write_tensors.assert_called_once_with(
        tensors,
        "output.nc",
        template,
        compression="zlib",
    )
    assert data_io._performance_history[0].operation is IOOperation.WRITE


def test_write_dataset_and_unsupported_data_branches() -> None:
    data_io, _, writer = _io_with_mocks()
    dataset = xr.Dataset({"temperature": ("time", [280.0])})
    writer.write_dataset.return_value = _write_result()

    assert data_io.write(dataset, "output.nc") is writer.write_dataset.return_value
    writer.write_dataset.assert_called_once_with(dataset, "output.nc")

    with pytest.warns(UserWarning, match="Write operation failed"):
        with pytest.raises(ValueError, match="Unsupported data type"):
            data_io.write(object(), "output.nc")


def test_write_tensors_builds_coordinate_template() -> None:
    data_io, _, writer = _io_with_mocks()
    writer.write_tensors.return_value = _write_result()
    tensors = {"temperature": torch.tensor([280.0])}

    data_io.write_tensors(
        tensors,
        "output.nc",
        coordinates={"time": np.array([0])},
        metadata={"global": {"title": "sample"}},
        compression="zlib",
    )

    args, kwargs = writer.write_tensors.call_args
    assert args[:2] == (tensors, "output.nc")
    assert isinstance(args[2], xr.Dataset)
    assert list(args[2].coords) == ["time"]
    assert args[2].attrs == {"title": "sample"}
    assert kwargs == {"compression": "zlib"}


def test_write_tensors_without_metadata_uses_no_template() -> None:
    data_io, _, writer = _io_with_mocks()
    writer.write_tensors.return_value = _write_result()
    tensors = {"temperature": torch.tensor([280.0])}

    data_io.write_tensors(tensors, "output.nc")

    writer.write_tensors.assert_called_once_with(
        tensors,
        "output.nc",
        None,
    )


def test_convert_format_strips_read_prefix_and_records_conversion() -> None:
    data_io, reader, writer = _io_with_mocks()
    dataset = xr.Dataset({"temperature": ("time", [280.0])})
    reader.read_dataset.return_value = _read_result(dataset)
    writer.write_dataset.return_value = _write_result("output.zarr")

    read_result, write_result = data_io.convert_format(
        "input.nc",
        "output.zarr",
        "zarr",
        read_chunks={"time": 1},
        read_decode_times=False,
        compression="zstd",
    )

    assert read_result is reader.read_dataset.return_value
    assert write_result is writer.write_dataset.return_value
    reader.read_dataset.assert_called_once_with(
        "input.nc",
        chunks={"time": 1},
        decode_times=False,
        to_torch=False,
    )
    writer.write_dataset.assert_called_once_with(
        dataset,
        "output.zarr",
        compression="zstd",
        format="zarr",
    )
    assert data_io._performance_history[0].operation is IOOperation.CONVERT


def test_convert_format_rejects_invalid_target_before_io() -> None:
    data_io, reader, writer = _io_with_mocks()

    with pytest.raises(ValueError, match="unsupported"):
        data_io.convert_format("input.nc", "output.bad", "unsupported")

    reader.read_dataset.assert_not_called()
    writer.write_dataset.assert_not_called()


def test_copy_with_processing_wraps_processing_errors() -> None:
    data_io, reader, writer = _io_with_mocks()
    dataset = xr.Dataset({"temperature": ("time", [280.0])})
    reader.read_dataset.return_value = _read_result(dataset)
    writer.write_dataset.return_value = _write_result()

    read_result, write_result = data_io.copy_with_processing(
        "input.nc",
        "output.nc",
        lambda source: source.assign(temperature=source.temperature - 273.15),
    )
    assert read_result is reader.read_dataset.return_value
    assert write_result is writer.write_dataset.return_value
    processed = writer.write_dataset.call_args.args[0]
    assert processed.temperature.item() == pytest.approx(6.85)

    with pytest.raises(ValueError, match="Processing function failed"):
        data_io.copy_with_processing(
            "input.nc",
            "output.nc",
            lambda _source: (_ for _ in ()).throw(RuntimeError("bad transform")),
        )


def test_get_file_info_handles_missing_and_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    data_io, _, _ = _io_with_mocks()
    missing = tmp_path / "missing.nc"
    with pytest.raises(FileNotFoundError, match="File not found"):
        data_io.get_file_info(missing)

    existing = tmp_path / "sample.nc"
    existing.touch()
    dataset = xr.Dataset(
        {"temperature": ("time", [280.0, 281.0])},
        coords={"time": [0, 1]},
        attrs={"Conventions": "CF-1.8"},
    )

    class DatasetContext:
        def __enter__(self):
            return dataset

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(xr, "open_dataset", lambda *_args, **_kwargs: DatasetContext())
    info = data_io.get_file_info(existing)

    assert info["format"] == "netcdf"
    assert info["variables"] == ["temperature"]
    assert info["dimensions"] == {"time": 2}
    assert "cf_compliant" not in info
    assert info["cf_conventions_declared"] is True
    assert info["dataset_conformance_checked"] is False
    assert info["cf_validation_scope"] == "not_evaluated"
    assert info["time_coverage"] == {"start": "0", "end": "1", "length": 2}


def test_get_file_info_raises_a_metadata_reader_failure(
    tmp_path,
    monkeypatch,
) -> None:
    data_io, _, _ = _io_with_mocks()
    existing = tmp_path / "invalid.nc"
    existing.touch()
    monkeypatch.setattr(
        xr,
        "open_dataset",
        Mock(side_effect=OSError("invalid netcdf")),
    )

    # A file whose metadata cannot be read raises; no info dict carrying an error is returned.
    with pytest.raises(OSError, match="invalid netcdf"):
        data_io.get_file_info(existing)


def test_introspection_performance_history_and_cache() -> None:
    data_io, reader, _ = _io_with_mocks()
    assert data_io.get_performance_stats() == {
        "message": "No performance data available"
    }
    formats = data_io.list_formats()
    assert "netcdf4" in formats["read_formats"]
    assert "zstd" in formats["compression_methods"]

    for index in range(105):
        data_io._record_performance(
            operation=IOOperation.READ if index % 2 == 0 else IOOperation.WRITE,
            file_size=1024 * 1024,
            operation_time=1.0,
            format=adapter_module.DataFormat.NETCDF4,
        )
    assert len(data_io._performance_history) == 100
    stats = data_io.get_performance_stats()
    assert stats["total_operations"] == 100
    assert stats["read_throughput_avg"] == pytest.approx(1.0)
    assert stats["write_throughput_max"] == pytest.approx(1.0)

    data_io.clear_cache()
    reader.io_handler.clear_cache.assert_called_once_with()
    assert data_io._performance_history == []


def test_performance_stats_handle_single_operation_kind() -> None:
    data_io, _, _ = _io_with_mocks()
    data_io._record_performance(
        operation=IOOperation.READ,
        file_size=2 * 1024 * 1024,
        operation_time=1.0,
        format=adapter_module.DataFormat.NETCDF4,
    )
    read_only = data_io.get_performance_stats()
    assert read_only["read_operations"] == 1
    assert read_only["write_operations"] == 0
    assert "write_throughput_avg" not in read_only

    data_io._performance_history.clear()
    data_io._record_performance(
        operation=IOOperation.WRITE,
        file_size=3 * 1024 * 1024,
        operation_time=1.0,
        format=adapter_module.DataFormat.NETCDF4,
    )
    write_only = data_io.get_performance_stats()
    assert write_only["read_operations"] == 0
    assert write_only["write_throughput_avg"] == pytest.approx(3.0)
    assert "read_throughput_avg" not in write_only


def test_standardize_preserves_official_attrs_and_leaves_unknown_unclaimed() -> None:
    data_io = _data_io(_mapper())
    source = xr.Dataset(
        {
            "temp": xr.DataArray(
                [280.0],
                dims=("time",),
                attrs={"standard_name": "air_temperature", "units": "degC"},
            ),
            "unknown": ("time", [1.0]),
        }
    )

    result = data_io.standardize_for_relic(source)

    assert result["temp"].attrs == {
        "standard_name": "air_temperature",
        "units": "degC",
    }
    assert result["unknown"].attrs == {}


def test_module_facades_use_single_lazy_default(monkeypatch) -> None:
    default = Mock()
    default.read.return_value = "read"
    default.write.return_value = "write"
    default.convert_format.return_value = ("read_result", "write_result")
    default.get_file_info.return_value = {"format": "netcdf"}
    monkeypatch.setattr(adapter_module, "_DEFAULT_IO", default)

    assert adapter_module.get_default_io() is default
    assert adapter_module.read("input.nc", chunks={}) == "read"
    assert adapter_module.write({}, "output.nc", format="netcdf4") == "write"
    assert adapter_module.convert_format(
        "input.nc",
        "output.zarr",
        "zarr",
    ) == ("read_result", "write_result")
    assert adapter_module.get_file_info("input.nc") == {"format": "netcdf"}
    default.read.assert_called_once_with("input.nc", chunks={})
    default.write.assert_called_once_with({}, "output.nc", format="netcdf4")


def test_get_default_io_constructs_once_when_cache_is_empty(monkeypatch) -> None:
    created = Mock()
    constructor = Mock(return_value=created)
    monkeypatch.setattr(adapter_module, "_DEFAULT_IO", None)
    monkeypatch.setattr(adapter_module, "DataIO", constructor)

    first = adapter_module.get_default_io()
    second = adapter_module.get_default_io()

    assert first is created
    assert second is created
    constructor.assert_called_once_with()


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
def test_write_tensors_delivers_explicit_variable_and_coordinate_contracts(tmp_path, format):
    from copy import deepcopy
    from relic.data.io.writer import WriteConfig

    data_io = DataIO(writer_config=WriteConfig(cf_sync=False, add_bounds=False))
    tensors = {"signal": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64), "scalar": torch.tensor(5.0)}
    coordinates = {"row": np.array([0, 1]), "column": np.array([10, 20])}
    metadata = {
        "global": {"title": "Explicit tensor export"},
        "signal": {"dims": ["row", "column"], "units": "1", "encoding": {"_FillValue": -9999.0}},
        "scalar": {"units": "1", "long_name": "Scalar observation"},
        "row": {"units": "m"},
    }
    original = deepcopy(metadata)
    path = tmp_path / ("adapter.zarr" if format == "zarr" else "adapter.nc")
    first = data_io.write_tensors(tensors, path, coordinates, metadata, format=format, delivery_key="adapter")
    assert first.validation_passed, first.warnings
    opener = xr.open_zarr if format == "zarr" else xr.open_dataset
    options = {"consolidated": False} if format == "zarr" else {}
    with opener(path, **options) as stored:
        assert stored.signal.dims == ("row", "column")
        np.testing.assert_array_equal(stored.signal.values, tensors["signal"].numpy())
        assert stored.signal.attrs["units"] == "1"
        assert stored.signal.encoding["_FillValue"] == -9999.0
        assert stored.row.attrs["units"] == "m"
        assert stored.scalar.attrs["long_name"] == "Scalar observation"
        assert "dims" not in stored.signal.attrs
    assert metadata == original
    second = data_io.write_tensors(tensors, path, coordinates, metadata, format=format, delivery_key="adapter")
    assert second.replayed and second.sha256 == first.sha256


@pytest.mark.parametrize("dims", [None, ["row"], ["row", "row"]])
def test_tensor_adapter_rejects_missing_or_ambiguous_variable_axes(tmp_path, dims):
    from relic.data.io.writer import WriteConfig
    data_io = DataIO(writer_config=WriteConfig(cf_sync=False, add_bounds=False))
    metadata = {"signal": {"units": "1"}}
    if dims is not None:
        metadata["signal"]["dims"] = dims
    with pytest.raises(ValueError, match="dimension|dims"):
        data_io.write_tensors(
            {"signal": torch.ones((2, 2))}, tmp_path / "invalid.nc",
            coordinates={"row": np.array([0, 1])}, metadata=metadata,
        )
    assert not (tmp_path / "invalid.nc").exists()


def test_tensor_adapter_supports_declared_axes_without_coordinate_arrays(tmp_path):
    from relic.data.io.writer import WriteConfig
    data_io = DataIO(writer_config=WriteConfig(cf_sync=False, add_bounds=False))
    path = tmp_path / "index.nc"
    result = data_io.write_tensors(
        {"signal": torch.tensor([1.0, 2.0])}, path,
        metadata={"signal": {"dims": ["cell"], "units": "1"}},
    )
    assert result.validation_passed
    with xr.open_dataset(path) as stored:
        assert stored.signal.dims == ("cell",)
        np.testing.assert_array_equal(stored.signal, [1.0, 2.0])


def test_write_numpy_dictionary_preserves_declared_template(tmp_path):
    from relic.data.io.writer import WriteConfig
    data_io = DataIO(writer_config=WriteConfig(cf_sync=False, add_bounds=False))
    values = np.array([1.0, 1.0 + 2.0 ** -40])
    values.setflags(write=False)
    template = xr.Dataset({"signal": ("cell", [0.0, 0.0])})
    template.signal.attrs["units"] = "1"
    path = tmp_path / "numpy.nc"
    result = data_io.write({"signal": values}, path, template=template, delivery_key="numpy")
    assert result.validation_passed
    with xr.open_dataset(path) as stored:
        np.testing.assert_array_equal(stored.signal, values)
        assert stored.signal.dtype == np.dtype("float64")


@pytest.mark.parametrize("operation", ["convert", "process"])
def test_dataset_operations_request_metadata_preserving_reader_mode(operation):
    data_io, reader, writer = _io_with_mocks()
    dataset = xr.Dataset({"signal": ("cell", [1.0, 2.0])})
    reader.read_dataset.side_effect = lambda *args, **kwargs: _read_result(
        {"signal": torch.tensor([1.0, 2.0])} if kwargs.get("to_torch", True) else dataset
    )
    writer.write_dataset.return_value = _write_result()
    if operation == "convert":
        data_io.convert_format("input.nc", "output.nc", "netcdf4")
    else:
        data_io.copy_with_processing("input.nc", "output.nc", lambda source: source.assign(signal=source.signal + 1.0))
    assert reader.read_dataset.call_args.kwargs["to_torch"] is False
    assert isinstance(writer.write_dataset.call_args.args[0], xr.Dataset)


def test_file_info_of_an_unreadable_file_raises(tmp_path):
    """get_file_info raises for a file it cannot read instead of returning an info dict with an error."""
    import pytest

    from relic.data.io.adapter import DataIO

    broken = tmp_path / "broken.nc"
    broken.write_bytes(b"not a netcdf file")
    with pytest.raises((OSError, ValueError)):
        DataIO().get_file_info(broken)
