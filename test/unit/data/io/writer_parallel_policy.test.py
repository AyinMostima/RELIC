"""DataWriter format, scientific content and external delivery contracts."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pytest
import torch
import xarray as xr

from relic.coupling.external.common.io import CompressionType, DataFormat
from relic.data.io.writer import DataWriter, WriteConfig, WriteResult


def test_writer_defaults_to_honest_sequential_mode() -> None:
    assert WriteConfig().parallel is False


def test_writer_rejects_unwired_parallel_mode_before_output(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = DataWriter(WriteConfig(parallel=False, validate_output=False))
    monkeypatch.setattr(
        writer,
        "_write_with_format",
        lambda *_args, **_kwargs: pytest.fail(
            "output writer must not run for an unwired parallel request"
        ),
    )

    with pytest.raises(NotImplementedError, match="parallel writing"):
        writer.write_dataset(
            xr.Dataset(),
            tmp_path / "output.nc",
            parallel=True,
        )


def _write_result() -> WriteResult:
    return WriteResult(output_path=Path("unused.nc"))


def test_writer_preserves_missing_optional_cf_metadata_without_claim() -> None:
    writer = DataWriter(WriteConfig(cf_sync=True, strict_cf=False))
    source = xr.Dataset({"mystery_signal": ("time", [1.0])})
    result = _write_result()

    processed = writer._synchronize_cf_attributes(source, writer.config, result)

    assert processed["mystery_signal"].attrs.get("standard_name") is None
    assert processed["mystery_signal"].attrs.get("units") is None
    assert "variable_name" not in processed["mystery_signal"].attrs
    assert result.cf_compliant is False
    assert any("dataset-level CF conformance" in item for item in result.warnings)


def test_writer_preserves_explicit_official_metadata() -> None:
    writer = DataWriter(WriteConfig(cf_sync=True, strict_cf=False))
    source = xr.Dataset({"temperature": ("time", [20.0])})
    source["temperature"].attrs = {
        "standard_name": "air_temperature",
        "units": "degree_Celsius",
        "provenance": "source-file",
    }
    result = _write_result()

    processed = writer._synchronize_cf_attributes(source, writer.config, result)

    assert processed["temperature"].attrs["standard_name"] == "air_temperature"
    assert processed["temperature"].attrs["units"] == "degree_Celsius"
    assert processed["temperature"].attrs["provenance"] == "source-file"
    assert processed["temperature"].item() == 20.0


@pytest.mark.parametrize(
    "attrs",
    [
        {"standard_name": "totally_fake_cf_name", "units": "K"},
        {"standard_name": "air_temperature", "units": "Pa"},
        {"standard_name": "air_temperature"},
        {"units": "custom_unit"},
    ],
)
def test_writer_non_strict_invalid_metadata_is_preserved_and_failed(attrs) -> None:
    writer = DataWriter(WriteConfig(cf_sync=True, strict_cf=False))
    source = xr.Dataset({"signal": ("time", [1.0])})
    source["signal"].attrs = dict(attrs)
    result = _write_result()

    processed = writer._synchronize_cf_attributes(source, writer.config, result)

    assert processed["signal"].attrs == attrs
    assert source["signal"].attrs == attrs
    assert result.cf_compliant is False
    assert result.validation_passed is False
    assert result.warnings


def test_writer_strict_invalid_metadata_raises_before_mutation() -> None:
    writer = DataWriter(WriteConfig(cf_sync=True, strict_cf=True))
    source = xr.Dataset({"signal": ("time", [1.0])})
    source["signal"].attrs = {
        "standard_name": "air_temperature",
        "units": "Pa",
    }

    with pytest.raises(ValueError, match="standard-name metadata"):
        writer._synchronize_cf_attributes(source, writer.config, _write_result())

    assert source["signal"].attrs == {
        "standard_name": "air_temperature",
        "units": "Pa",
    }


def test_writer_has_no_second_cf_unit_table_or_implicit_conversion_path() -> None:
    assert not hasattr(DataWriter, "_get_cf_standard_units")
    assert not hasattr(DataWriter, "_apply_unit_conversions")


def test_writer_global_metadata_does_not_self_declare_cf_conformance() -> None:
    writer = DataWriter(WriteConfig(update_history=True, title="Scoped output"))
    result = _write_result()

    processed = writer._ensure_global_attributes(
        xr.Dataset(),
        writer.config,
        result,
    )

    assert "Conventions" not in processed.attrs
    assert "CF-1.8 compliance" not in processed.attrs["history"]
    assert processed.attrs["title"] == "Scoped output"


def test_writer_global_attribute_contract_documents_no_conventions_claim() -> None:
    """The documented contract matches the implementation: no Conventions is written.

    The writer checks only the single-variable standard-name/unit subset, so it
    cannot claim dataset-level CF conformance; a Conventions value the caller
    supplied passes through unchanged as the caller's own declaration.
    """
    import inspect

    doc = inspect.getdoc(DataWriter._ensure_global_attributes) or ""
    assert "including Conventions" not in doc
    assert "CF-1.8 global attributes" not in doc

    writer = DataWriter(WriteConfig(update_history=False))
    processed = writer._ensure_global_attributes(
        xr.Dataset(attrs={"Conventions": "CF-1.10"}),
        writer.config,
        _write_result(),
    )
    assert processed.attrs["Conventions"] == "CF-1.10"


def _delivery_dataset():
    source = xr.Dataset(
        {"signal": (("row", "column"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"row": [0, 1], "column": [10, 20]},
        attrs={"history": "Original measurement", "experiment": "delivery"},
    )
    source["signal"].attrs = {"units": "1", "long_name": "Measured signal"}
    return source


def _delivery_writer(format):
    return DataWriter(WriteConfig(
        format=DataFormat(format), compression=CompressionType.NONE,
        cf_sync=False, add_bounds=False, auto_chunk=False,
    ))


def _read_output(path, format):
    opener = xr.open_zarr if format == "zarr" else xr.open_dataset
    options = {"consolidated": False} if format == "zarr" else {}
    with opener(path, **options) as source:
        return source.load()


def _output_bytes(path):
    if path.is_file():
        return {path.name: path.read_bytes()}
    return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*") if item.is_file()}


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
@pytest.mark.parametrize("compression", ["none", "zstd", "zlib"])
def test_prepared_write_freezes_data_options_and_runs_pipeline_once(tmp_path, monkeypatch, format, compression):
    from relic.core.simulator.checkpoint import _decode_checkpoint_value, _encode_checkpoint_value

    source = _delivery_dataset()
    source["signal"].values[0, 0] = 1.0 + 2.0 ** -40
    source["signal"].encoding.update(dtype=np.dtype("float64"), _FillValue=np.float64(-9999))
    source["signal"].attrs["valid_range"] = np.array([0.0, 10.0])
    expected = source.copy(deep=True)
    path = tmp_path / "uncreated" / ("prepared.zarr" if format == "zarr" else "prepared.nc")
    writer = _delivery_writer(format)
    pipeline = writer._process_write_pipeline
    encoding = writer._prepare_encoding
    calls = []

    def counted_pipeline(*args, **kwargs):
        calls.append("pipeline")
        return pipeline(*args, **kwargs)

    def counted_encoding(*args, **kwargs):
        calls.append("encoding")
        return encoding(*args, **kwargs)

    monkeypatch.setattr(writer, "_process_write_pipeline", counted_pipeline)
    monkeypatch.setattr(writer, "_prepare_encoding", counted_encoding)
    prepared = writer.prepare_dataset(source, path, delivery_key="frozen-output", compression=compression)
    assert calls == ["pipeline", "encoding"]
    assert not path.parent.exists()
    prepared = _decode_checkpoint_value(
        _encode_checkpoint_value(prepared, path="prepared"), path="prepared",
    )
    source["signal"].values[:] = 99.0
    source["signal"].attrs["units"] = "kg"
    source["signal"].encoding["_FillValue"] = -1.0
    writer = DataWriter(WriteConfig(format=DataFormat.CSV, title="Changed live title"))
    monkeypatch.setattr(writer, "_process_write_pipeline", lambda *a, **k: pytest.fail("Prepared data must not be processed again"))
    monkeypatch.setattr(writer, "_prepare_encoding", lambda *a, **k: pytest.fail("Prepared encoding must not be recomputed"))
    first = writer.write_dataset(None, path, snapshot=prepared)
    before = _output_bytes(path)
    repeated = writer.write_dataset(None, path, snapshot=prepared)
    assert first.validation_passed and first.delivery_key == "frozen-output"
    assert repeated.replayed and repeated.sha256 == first.sha256
    assert _output_bytes(path) == before
    stored = _read_output(path, format)
    np.testing.assert_array_equal(stored.signal.values, expected.signal.values)
    assert stored.signal.dtype == np.dtype("float64")
    assert stored.signal.attrs["units"] == "1"
    np.testing.assert_array_equal(stored.signal.attrs["valid_range"], [0.0, 10.0])
    assert stored.signal.encoding["_FillValue"] == -9999
    assert stored.attrs.get("title") != "Changed live title"


def test_ordinary_non_strict_write_failure_raises_and_publishes_nothing(tmp_path, monkeypatch):
    """A failed write raises in every mode; no receipt of an unwritten file is returned."""
    writer = _delivery_writer("netcdf4")
    def fail_write(*args, **kwargs):
        raise OSError("injected unkeyed disk failure")
    monkeypatch.setattr(writer, "_write_with_format", fail_write)
    with pytest.raises(OSError, match="unkeyed disk failure"):
        writer.write_dataset(_delivery_dataset(), tmp_path / "failure.nc")
    assert not (tmp_path / "failure.nc").exists()


def test_prepare_rejects_static_metadata_errors_without_creating_directories(tmp_path):
    writer = DataWriter(WriteConfig(strict_cf=False, add_bounds=False))
    source = xr.Dataset({"temperature": ("row", [1.0])})
    source.temperature.attrs.update(standard_name="air_temperature", units="Pa")
    path = tmp_path / "uncreated" / "bad.nc"
    with pytest.raises(ValueError, match="metadata|units"):
        writer.prepare_dataset(source, path)
    assert not path.parent.exists()


@pytest.mark.parametrize("conflict", ["dataset", "path", "key", "options"])
def test_prepared_write_rejects_conflicting_live_arguments(tmp_path, conflict):
    path = tmp_path / "fixed.nc"
    writer = _delivery_writer("netcdf4")
    prepared = writer.prepare_dataset(_delivery_dataset(), path, delivery_key="fixed")
    dataset = _delivery_dataset() if conflict == "dataset" else None
    target = tmp_path / "other.nc" if conflict == "path" else path
    options = {"delivery_key": "other"} if conflict == "key" else {"title": "other"} if conflict == "options" else {}
    with pytest.raises(ValueError, match="snapshot|prepared|conflict"):
        writer.write_dataset(dataset, target, snapshot=prepared, **options)
    assert not path.exists() and not target.exists()


@pytest.mark.parametrize("option", [None, "format", "compression", "cf_sync"])
def test_prepared_write_facade_preserves_binding_and_rejects_overrides(tmp_path, option):
    from relic.data.io.writer import write_dataset
    path = tmp_path / "facade.zarr"
    prepared = _delivery_writer("zarr").prepare_dataset(_delivery_dataset(), path, delivery_key="facade")
    if option is None:
        assert write_dataset(None, path, snapshot=prepared).validation_passed
        assert path.is_dir()
    else:
        overrides = {option: {"format": "netcdf4", "compression": "zstd", "cf_sync": True}[option]}
        with pytest.raises(ValueError, match="snapshot|prepared|conflict"):
            write_dataset(None, path, snapshot=prepared, **overrides)
        assert not path.exists()


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
def test_preparation_preserves_declared_chunks_before_materialization(tmp_path, format):
    writer = _delivery_writer(format)
    path = tmp_path / ("chunks.zarr" if format == "zarr" else "chunks.nc")
    prepared = writer.prepare_dataset(
        _delivery_dataset(), path, delivery_key="chunks",
        auto_chunk=True, chunks={"row": 1, "column": 1},
    )
    writer.write_dataset(None, path, snapshot=prepared)
    encoding = _read_output(path, format).signal.encoding
    assert tuple(encoding["chunks" if format == "zarr" else "chunksizes"]) == (1, 1)


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
def test_unbound_candidate_accepts_explicit_delivery_identity_without_reprocessing(tmp_path, monkeypatch, format):
    path = tmp_path / ("unbound.zarr" if format == "zarr" else "unbound.nc")
    writer = _delivery_writer(format)
    candidate = writer.prepare_dataset(_delivery_dataset(), path)
    monkeypatch.setattr(writer, "_process_write_pipeline", lambda *a, **k: pytest.fail("Do not reprocess a candidate"))
    first = writer.write_dataset(None, path, snapshot=candidate, delivery_key="accepted-request")
    before = _output_bytes(path)
    repeated = writer.write_dataset(None, path, snapshot=candidate, delivery_key="accepted-request")
    assert first.delivery_created_utc == candidate["result"]["delivery_created_utc"]
    assert first.delivery_key == "accepted-request"
    assert repeated.replayed and repeated.sha256 == first.sha256
    assert _output_bytes(path) == before
    assert candidate["result"]["delivery_key"] is None
    assert candidate["result"]["sha256"] is None


@pytest.mark.parametrize("key,format", [("", "netcdf4"), (" ", "netcdf4"), (True, "netcdf4"), ("key", "csv")])
def test_unbound_candidate_requires_valid_delivery_identity(tmp_path, key, format):
    path = tmp_path / "unbound.out"
    writer = _delivery_writer(format)
    candidate = writer.prepare_dataset(_delivery_dataset(), path)
    with pytest.raises(ValueError, match="Delivery key|NetCDF|Zarr"):
        writer.write_dataset(None, path, snapshot=candidate, delivery_key=key)
    assert not path.exists()


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
@pytest.mark.parametrize("compression", ["none", "zlib", "zstd"])
def test_writer_preserves_float64_and_uses_effective_encoding(tmp_path, format, compression):
    source = _delivery_dataset()
    source["signal"].values[0, 1] = 1.0 + 2.0 ** -40
    writer = _delivery_writer(format)
    path = tmp_path / ("precision.zarr" if format == "zarr" else "precision.nc")
    result = writer.write_dataset(source, path, compression=compression, chunks={"row": 1}, auto_chunk=True)
    assert result.validation_passed, result.warnings
    stored = _read_output(path, format)
    assert stored["signal"].dtype == np.dtype("float64")
    np.testing.assert_array_equal(stored["signal"].values, source["signal"].values)
    assert stored["signal"].values[0, 0] != stored["signal"].values[0, 1]
    assert stored["signal"].dims == ("row", "column")
    assert stored["signal"].attrs == source["signal"].attrs
    assert result.file_size == sum(len(data) for data in _output_bytes(path).values())
    if format == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(path) as file:
            filters = file["signal"].filters()
            assert bool(filters["zstd"]) == (compression == "zstd")
            assert bool(filters["zlib"]) == (compression == "zlib")
    else:
        import zarr
        codecs = zarr.open_group(path, mode="r")["signal"].compressors
        assert bool(codecs) == (compression != "none")


def test_tensor_template_preserves_explicit_dims_metadata_and_ownership():
    template = _delivery_dataset()
    template["unselected"] = ("row", [-1.0, -1.0])
    template.coords["row"].attrs["units"] = "m"
    template["signal"].encoding["_FillValue"] = -9999.0
    tensors = {"signal": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64), "scalar": torch.tensor(9.0)}
    original = template.copy(deep=True)
    result = _delivery_writer("netcdf4")._tensors_to_dataset(tensors, template)
    assert set(result.data_vars) == {"signal", "scalar"}
    assert result["signal"].dims == ("row", "column")
    assert result["signal"].attrs == template["signal"].attrs
    assert result["signal"].encoding["_FillValue"] == -9999.0
    assert result["row"].attrs == template["row"].attrs
    tensors["signal"].fill_(100.0)
    assert result["signal"].values[0, 0] == 5.0
    xr.testing.assert_identical(template, original)


@pytest.mark.parametrize("shape,template", [
    ((2, 1), _delivery_dataset()),
    ((2,), xr.Dataset(coords={"row": [0, 1], "column": [10, 20]})),
    ((2,), None),
])
def test_tensor_writer_requires_unambiguous_declared_dimensions(shape, template):
    with pytest.raises(ValueError, match="shape|dimension|template"):
        _delivery_writer("netcdf4")._tensors_to_dataset({"signal": torch.ones(shape)}, template)


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
@pytest.mark.parametrize("time_kind", ["numeric", "datetime", "noleap"])
@pytest.mark.parametrize("has_bounds", [False, True])
def test_writer_roundtrips_declared_cf_time_without_invented_bounds(tmp_path, format, time_kind, has_bounds):
    units = "days since 2001-01-01 00:00:00"
    calendar = "noleap" if time_kind == "noleap" else "standard"
    if time_kind == "datetime":
        times = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
    elif time_kind == "noleap":
        import cftime
        times = [cftime.DatetimeNoLeap(2001, 1, 1), cftime.DatetimeNoLeap(2001, 1, 2)]
    else:
        times = [0.0, 1.0]
    source = xr.Dataset({"signal": ("time", [1.0, 2.0])}, coords={"time": times})
    declaration = source.time.attrs if time_kind == "numeric" else source.time.encoding
    declaration.update(units=units, calendar=calendar)
    if has_bounds:
        if time_kind == "numeric":
            bounds = [[-0.5, 0.5], [0.5, 1.5]]
        elif time_kind == "datetime":
            bounds = np.stack([times - np.timedelta64(12, "h"), times + np.timedelta64(12, "h")], axis=1)
        else:
            from datetime import timedelta
            bounds = [[value - timedelta(hours=12), value + timedelta(hours=12)] for value in times]
        source["support"] = (("time", "nv"), bounds)
        source.time.attrs["bounds"] = "support"
        bounds_declaration = source.support.attrs if time_kind == "numeric" else source.support.encoding
        bounds_declaration.update(units=units, calendar=calendar)
    path = tmp_path / ("time.zarr" if format == "zarr" else "time.nc")
    result = _delivery_writer(format).write_dataset(source, path)
    assert result.validation_passed, result.warnings
    stored = _read_output(path, format)
    xr.testing.assert_equal(stored.time, xr.decode_cf(source).time)
    assert stored.time.encoding["units"] == units
    assert stored.time.encoding["calendar"] == ("gregorian" if calendar == "standard" else calendar)
    assert "time_bounds" not in stored
    if has_bounds:
        xr.testing.assert_equal(stored.support, xr.decode_cf(source).support)
        assert stored.time.attrs["bounds"] == "support"


@pytest.mark.parametrize("field,value", [("units", "hours since 2001-01-01"), ("calendar", "360_day")])
def test_writer_rejects_conflicting_bounds_contract_before_publication(tmp_path, field, value):
    from relic.coupling.external.common.time import ValidationError
    source = xr.Dataset({"signal": ("time", [1.0, 2.0])}, coords={"time": [0.0, 1.0]})
    source.time.attrs.update(units="days since 2001-01-01", calendar="noleap", bounds="support")
    source["support"] = (("time", "nv"), [[-0.5, 0.5], [0.5, 1.5]])
    source.support.attrs.update(units="days since 2001-01-01", calendar="noleap")
    source.support.attrs[field] = value
    with pytest.raises(ValidationError, match="bounds.*contract"):
        _delivery_writer("netcdf4").write_dataset(source, tmp_path / "bad.nc", delivery_key="bad")
    assert not (tmp_path / "bad.nc").exists()


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
def test_delivery_retries_with_new_writer_without_rewriting(tmp_path, monkeypatch, format):
    source = _delivery_dataset()
    path = tmp_path / ("delivery.zarr" if format == "zarr" else "delivery.nc")
    first = _delivery_writer(format).write_dataset(source, path, delivery_key="run:1:output")
    assert first.validation_passed, first.warnings
    assert first.delivery_key == "run:1:output"
    assert first.sha256 and not first.replayed
    before = _output_bytes(path)
    writer = _delivery_writer(format)
    monkeypatch.setattr(writer, "_write_with_format", lambda *a, **k: pytest.fail("Retry must not rewrite"))
    second = writer.write_dataset(source, path, delivery_key="run:1:output")
    assert second.replayed and second.sha256 == first.sha256
    assert _output_bytes(path) == before
    assert _read_output(path, format).attrs["history"].count("Processed by RELIC") == 1
    if format == "netcdf4":
        assert second.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    changed = source.copy(deep=True)
    changed["signal"] += 5.0
    result = _delivery_writer(format).write_dataset(changed, path, delivery_key="run:2:output")
    assert result.validation_passed and not result.replayed
    np.testing.assert_array_equal(_read_output(path, format)["signal"], changed["signal"])


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
@pytest.mark.parametrize("changed_contract", ["fill", "time_units", "bounds_fill"])
def test_delivery_checks_storage_contract_even_when_decoded_values_match(
    tmp_path, format, changed_contract,
):
    source = xr.Dataset({"signal": ("time", [1.0, 2.0])}, coords={"time": [0.0, 1.0]})
    source.time.attrs.update(units="days since 2001-01-01", calendar="noleap")
    source.signal.encoding["_FillValue"] = -9999.0
    if changed_contract == "bounds_fill":
        source["support"] = (("time", "nv"), [[-0.5, 0.5], [0.5, 1.5]])
        source.time.attrs["bounds"] = "support"
        source.support.encoding["_FillValue"] = -9999.0
    path = tmp_path / ("encoding.zarr" if format == "zarr" else "encoding.nc")
    writer = _delivery_writer(format)
    assert writer.write_dataset(source, path, delivery_key="same").validation_passed
    if changed_contract == "bounds_fill":
        assert _read_output(path, format).support.encoding["_FillValue"] == -9999.0
    before = _output_bytes(path)
    changed = source.copy(deep=True)
    if changed_contract == "fill":
        changed.signal.encoding["_FillValue"] = -8888.0
    elif changed_contract == "time_units":
        changed = changed.assign_coords(time=[0.0, 24.0])
        changed.time.attrs.update(units="hours since 2001-01-01", calendar="noleap")
    else:
        changed.support.encoding["_FillValue"] = -8888.0
    xr.testing.assert_identical(xr.decode_cf(source), xr.decode_cf(changed))
    with pytest.raises(ValueError, match="content|contract"):
        writer.write_dataset(changed, path, delivery_key="same")
    assert _output_bytes(path) == before


def test_missing_nonfill_zarr_chunk_is_not_a_successful_replay(tmp_path):
    source = _delivery_dataset()
    path = tmp_path / "missing.zarr"
    writer = _delivery_writer("zarr")
    assert writer.write_dataset(source, path, delivery_key="same").validation_passed
    chunks = [item for item in (path / "signal" / "c").rglob("*") if item.is_file()]
    assert chunks
    chunks[0].unlink()
    damaged = _read_output(path, "zarr")
    assert not damaged.signal.identical(source.signal)
    before = _output_bytes(path)
    with pytest.raises(ValueError, match="content"):
        writer.write_dataset(source, path, delivery_key="same")
    assert _output_bytes(path) == before


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
def test_cancelled_write_cleans_only_its_staging_artifact(tmp_path, monkeypatch, format):
    source = _delivery_dataset()
    path = tmp_path / ("cancel.zarr" if format == "zarr" else "cancel.nc")
    writer = _delivery_writer(format)
    assert writer.write_dataset(source, path, delivery_key="first").validation_passed
    before = _output_bytes(path)
    original = writer._write_with_format
    def cancel_after_staging(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("cancelled publication")
    monkeypatch.setattr(writer, "_write_with_format", cancel_after_staging)
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        writer.write_dataset(source, path, delivery_key="second")
    assert _output_bytes(path) == before
    assert list(tmp_path.iterdir()) == [path]


def test_writer_convenience_functions_return_delivery_receipts(tmp_path):
    from relic.data.io.writer import write_dataset, write_tensors
    first = write_dataset(
        _delivery_dataset(), tmp_path / "dataset.nc", cf_sync=False,
        delivery_key="dataset", compression="none", add_bounds=False,
    )
    second = write_tensors(
        {"scalar": torch.tensor(2.0)}, tmp_path / "scalar.nc",
        delivery_key="tensor", compression="none", add_bounds=False,
    )
    assert isinstance(first, WriteResult) and first.delivery_key == "dataset"
    assert isinstance(second, WriteResult) and second.delivery_key == "tensor"
    assert first.validation_passed and second.validation_passed


@pytest.mark.parametrize("boundary", [1, 2])
def test_cancel_after_completed_zarr_rename_retains_visible_version(tmp_path, monkeypatch, boundary):
    from relic.data.io import writer as writer_module
    source = _delivery_dataset()
    path = tmp_path / "rename.zarr"
    writer = _delivery_writer("zarr")
    assert writer.write_dataset(source, path, delivery_key="first").validation_passed
    before = _output_bytes(path)
    changed = source.copy(deep=True)
    changed.signal.values[:] += 5.0
    replace = writer_module.os.replace
    renames = []
    with monkeypatch.context() as local:
        def cancel_after_rename(source_path, target_path):
            result = replace(source_path, target_path)
            if Path(source_path) == path or Path(target_path) == path:
                renames.append((source_path, target_path))
                if len(renames) == boundary:
                    raise KeyboardInterrupt("cancel after completed rename")
            return result
        local.setattr(writer_module.os, "replace", cancel_after_rename)
        with pytest.raises(KeyboardInterrupt, match="completed rename"):
            writer.write_dataset(changed, path, delivery_key="second")
    assert path.is_dir()
    assert list(tmp_path.iterdir()) == [path]
    if boundary == 1:
        assert _output_bytes(path) == before
    else:
        np.testing.assert_array_equal(_read_output(path, "zarr").signal, changed.signal)
    receipt = writer.write_dataset(changed, path, delivery_key="second")
    assert receipt.validation_passed and receipt.replayed == (boundary == 2)


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
@pytest.mark.parametrize("change", ["data", "dtype", "dims", "coordinate", "units", "global", "options"])
def test_same_delivery_key_rejects_conflicts_and_preserves_target(tmp_path, format, change):
    source = _delivery_dataset()
    path = tmp_path / ("conflict.zarr" if format == "zarr" else "conflict.nc")
    writer = _delivery_writer(format)
    assert writer.write_dataset(source, path, delivery_key="fixed").validation_passed
    before = _output_bytes(path)
    changed = source.copy(deep=True)
    options = {}
    if change == "data":
        changed["signal"].values[0, 0] = 99.0
    elif change == "dtype":
        changed["signal"] = changed["signal"].astype("float32")
    elif change == "dims":
        changed = changed.rename({"row": "other_row"})
    elif change == "coordinate":
        changed = changed.assign_coords(row=[5, 6])
    elif change == "units":
        changed["signal"].attrs["units"] = "K"
    elif change == "global":
        changed.attrs["experiment"] = "changed"
    else:
        options["compression"] = "zstd"
    with pytest.raises(ValueError, match="conflict|content|dtype|contract"):
        writer.write_dataset(changed, path, delivery_key="fixed", **options)
    assert _output_bytes(path) == before


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
def test_delivery_does_not_replay_corrupt_data_from_marker_alone(tmp_path, format):
    source = _delivery_dataset()
    path = tmp_path / ("corrupt.zarr" if format == "zarr" else "corrupt.nc")
    writer = _delivery_writer(format)
    assert writer.write_dataset(source, path, delivery_key="fixed").validation_passed
    if format == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(path, "a") as file:
            file["signal"][0, 0] = 99.0
    else:
        import zarr
        zarr.open_group(path, mode="a")["signal"][0, 0] = 99.0
    before = _output_bytes(path)
    with pytest.raises(ValueError, match="content|conflict"):
        writer.write_dataset(source, path, delivery_key="fixed")
    assert _output_bytes(path) == before


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
@pytest.mark.parametrize("boundary", ["write", "validate", "publish"])
def test_failed_delivery_preserves_previous_artifact(tmp_path, monkeypatch, format, boundary):
    from relic.data.io import writer as writer_module
    source = _delivery_dataset()
    path = tmp_path / ("atomic.zarr" if format == "zarr" else "atomic.nc")
    writer = _delivery_writer(format)
    assert writer.write_dataset(source, path, delivery_key="first").validation_passed
    before = _output_bytes(path)
    changed = source.copy(deep=True)
    changed["signal"] += 5.0
    with monkeypatch.context() as local:
        def fail(*args, **kwargs):
            raise OSError("injected delivery failure")
        if boundary == "write":
            local.setattr(writer, "_write_with_format", fail)
        elif boundary == "validate":
            local.setattr(writer, "_validate_output", fail)
        else:
            replace = writer_module.os.replace
            calls = []
            def fail_publish(source_path, target_path):
                calls.append((source_path, target_path))
                if len(calls) == (2 if format == "zarr" else 1):
                    raise OSError("injected delivery failure")
                return replace(source_path, target_path)
            local.setattr(writer_module.os, "replace", fail_publish)
        with pytest.raises(OSError, match="delivery failure"):
            writer.write_dataset(changed, path, delivery_key="second")
    assert _output_bytes(path) == before
    assert list(tmp_path.iterdir()) == [path]
    assert writer.write_dataset(changed, path, delivery_key="second").validation_passed


@pytest.mark.parametrize("format", ["netcdf4", "zarr"])
def test_receipt_failure_after_publication_retries_existing_data(tmp_path, monkeypatch, format):
    source = _delivery_dataset()
    path = tmp_path / ("receipt.zarr" if format == "zarr" else "receipt.nc")
    writer = _delivery_writer(format)
    with monkeypatch.context() as local:
        def fail_receipt(*args, **kwargs):
            raise OSError("receipt unavailable")
        local.setattr(writer, "_artifact_receipt", fail_receipt)
        with pytest.raises(OSError, match="receipt unavailable"):
            writer.write_dataset(source, path, delivery_key="fixed")
    before = _output_bytes(path)
    monkeypatch.setattr(writer, "_write_with_format", lambda *a, **k: pytest.fail("Do not republish"))
    result = writer.write_dataset(source, path, delivery_key="fixed")
    assert result.replayed and result.validation_passed
    assert _output_bytes(path) == before


def test_writer_failures_raise_in_non_strict_mode(tmp_path, monkeypatch):
    """A failed write, a failed chunking request and failed metadata synchronization raise."""
    import numpy as np
    import pytest
    import xarray as xr

    from relic.data.io.writer import DataWriter, WriteConfig

    dataset = xr.Dataset({"tas": ("time", np.array([280.0, 281.0]), {"units": "K"})})
    writer = DataWriter(WriteConfig(strict_cf=False))

    def failing_prepare(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(writer, "_process_write_pipeline", failing_prepare)
        with pytest.raises(OSError, match="disk full"):
            writer.write_dataset(dataset, tmp_path / "out.nc")

    chunked = DataWriter(WriteConfig(strict_cf=False, chunks={"not_a_dim": 2}))
    result = type("R", (), {"warnings": []})()
    with pytest.raises((ValueError, KeyError)):
        chunked._optimize_chunking(dataset, chunked.config, result)
