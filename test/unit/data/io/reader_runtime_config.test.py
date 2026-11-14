"""Tests for DataReader per-call runtime I/O configuration."""

from __future__ import annotations

import pytest
import xarray as xr

from relic.coupling.external.common.io import DataFormat, IOResult
from relic.coupling.external.common.validation_types import ValidationReport
from relic.data.io.reader import DataReader, ReadConfig, ReadResult


def _reader() -> DataReader:
    return DataReader(
        ReadConfig(
            chunks=None,
            parallel=True,
            cache_size=128,
            process_time=False,
            cf_validate=False,
            cf_standardize=False,
            validate_data=False,
        )
    )


def test_reader_validation_does_not_treat_unchecked_cf_as_data_failure():
    reader = _reader()
    reader.validator = type(
        "Validator",
        (),
        {
            "validate": staticmethod(
                lambda _dataset: ValidationReport(
                    source_path="input.nc",
                    validation_time="now",
                )
            )
        },
    )()
    result = ReadResult(data=xr.Dataset(), metadata={})

    reader._validate_data(xr.Dataset(), reader.config, result)

    assert result.validation_passed is True
    assert result.cf_compliant is False


def test_per_call_io_overrides_reach_handler_and_are_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _reader()
    original = {
        "chunks": reader.io_handler.config.chunks,
        "parallel": reader.io_handler.config.parallel,
        "cache_size": reader.io_handler.config.cache_size,
    }
    observed = {}

    def fake_read(_path):
        observed.update(
            chunks=reader.io_handler.config.chunks,
            parallel=reader.io_handler.config.parallel,
            cache_size=reader.io_handler.config.cache_size,
        )
        return IOResult(
            data=xr.Dataset(),
            format_detected=DataFormat.NETCDF4,
            file_size=0,
            load_time=0.0,
        )

    monkeypatch.setattr(reader.io_handler, "read", fake_read)

    result = reader.read_dataset(
        "input.nc",
        chunks={"time": 4},
        parallel=False,
        cache_size=64,
        process_time=False,
        cf_validate=False,
        cf_standardize=False,
        validate_data=False,
    )

    assert result.success is True
    assert observed == {
        "chunks": {"time": 4},
        "parallel": False,
        "cache_size": 64,
    }
    assert reader.io_handler.config.chunks == original["chunks"]
    assert reader.io_handler.config.parallel == original["parallel"]
    assert reader.io_handler.config.cache_size == original["cache_size"]


def test_per_call_io_overrides_are_restored_after_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _reader()
    original_chunks = reader.io_handler.config.chunks
    original_parallel = reader.io_handler.config.parallel

    def fail_read(_path):
        assert reader.io_handler.config.chunks == {"time": 2}
        assert reader.io_handler.config.parallel is False
        raise RuntimeError("read failed")

    monkeypatch.setattr(reader.io_handler, "read", fail_read)

    with pytest.raises(RuntimeError, match="read failed"):
        reader.read_dataset(
            "input.nc",
            chunks={"time": 2},
            parallel=False,
            process_time=False,
        )

    assert reader.io_handler.config.chunks == original_chunks
    assert reader.io_handler.config.parallel == original_parallel


def _cf_result(dataset: xr.Dataset) -> ReadResult:
    return ReadResult(data=dataset, metadata={})


def test_cf_standardization_does_not_invent_name_units_or_global_claim() -> None:
    reader = DataReader(
        ReadConfig(
            process_time=False,
            validate_data=False,
            cf_standardize=True,
            cf_validate=True,
        )
    )
    source = xr.Dataset({"mystery_signal": ("time", [1.0])})
    result = _cf_result(source)

    processed = reader._standardize_cf(source, reader.config, result)

    assert processed["mystery_signal"].attrs == {}
    assert "Conventions" not in processed.attrs
    assert result.variables_standardized == {}
    assert result.cf_compliant is False
    assert any("dataset-level CF conformance" in item for item in result.warnings)


@pytest.mark.parametrize(
    "attrs",
    [
        {"standard_name": "totally_fake_cf_name", "units": "K"},
        {"standard_name": "air_temperature", "units": "Pa"},
        {"standard_name": "air_temperature"},
        {"units": "custom_unit"},
    ],
)
def test_non_strict_cf_failure_preserves_attrs_and_marks_result_failed(attrs) -> None:
    reader = DataReader(
        ReadConfig(
            process_time=False,
            validate_data=False,
            cf_standardize=True,
            cf_validate=True,
            strict_cf=False,
        )
    )
    source = xr.Dataset({"signal": ("time", [1.0])})
    source["signal"].attrs = dict(attrs)
    result = _cf_result(source)

    processed = reader._standardize_cf(source, reader.config, result)

    assert processed["signal"].attrs == attrs
    assert source["signal"].attrs == attrs
    assert result.cf_compliant is False
    assert result.validation_passed is False
    assert result.success is False
    assert result.warnings


def test_strict_cf_failure_raises_before_metadata_mutation() -> None:
    reader = DataReader(
        ReadConfig(
            process_time=False,
            validate_data=False,
            cf_standardize=True,
            cf_validate=True,
            strict_cf=True,
        )
    )
    source = xr.Dataset({"signal": ("time", [1.0])})
    source["signal"].attrs = {
        "standard_name": "air_temperature",
        "units": "Pa",
    }
    result = _cf_result(source)

    with pytest.raises(ValueError, match="standard-name metadata"):
        reader._standardize_cf(source, reader.config, result)

    assert source["signal"].attrs == {
        "standard_name": "air_temperature",
        "units": "Pa",
    }


def test_valid_variable_subset_does_not_claim_dataset_conformance() -> None:
    reader = DataReader(
        ReadConfig(
            process_time=False,
            validate_data=False,
            cf_standardize=True,
            cf_validate=True,
        )
    )
    source = xr.Dataset({"temperature": ("time", [20.0])})
    source["temperature"].attrs = {
        "standard_name": "air_temperature",
        "units": "degree_Celsius",
    }
    result = _cf_result(source)

    processed = reader._standardize_cf(source, reader.config, result)

    assert processed["temperature"].attrs == source["temperature"].attrs
    assert result.validation_passed is True
    assert result.success is True
    assert result.cf_compliant is False


def test_local_variable_query_is_not_treated_as_an_official_cf_alias() -> None:
    reader = _reader()
    dataset = xr.Dataset({"signal": ("time", [1.0])})
    config = ReadConfig(variables=["temperature"], process_time=False)
    result = _cf_result(dataset)

    processed = reader._select_variables(dataset, config, result)

    assert list(processed.data_vars) == []
    assert any("not found" in item for item in result.warnings)


def _conversion_reader(**overrides) -> DataReader:
    settings = {
        "process_time": False,
        "validate_data": False,
        "apply_unit_conversion": True,
        **overrides,
    }
    return DataReader(ReadConfig(**settings))


def _variable(values, standard_name, units) -> xr.DataArray:
    array = xr.DataArray(values, dims=("time",))
    array.attrs = {"standard_name": standard_name, "units": units}
    return array


def test_unit_conversion_reaches_cf_canonical_units_through_one_owner() -> None:
    """CF-consistent inputs are converted to the CF table's canonical unit."""
    reader = _conversion_reader()
    source = xr.Dataset(
        {
            "temperature": _variable([20.0, 21.0, 22.0], "air_temperature", "degC"),
            "precipitation": _variable(
                [8.64, 17.28, 8.64], "precipitation_flux", "kg m-2 day-1"
            ),
            "carbon_flux": _variable(
                [8.64, 17.28, 12.96],
                "gross_primary_productivity_of_carbon",
                "g m-2 day-1",
            ),
        }
    )
    result = _cf_result(source)

    processed = reader._apply_unit_conversion(source, reader.config, result)

    expected = {
        "temperature": ([293.15, 294.15, 295.15], "K", "degC"),
        "precipitation": ([1.0e-4, 2.0e-4, 1.0e-4], "kg m-2 s-1", "kg m-2 day-1"),
        "carbon_flux": ([1.0e-7, 2.0e-7, 1.5e-7], "kg m-2 s-1", "g m-2 day-1"),
    }
    for name, (values, units, original) in expected.items():
        assert processed[name].values.tolist() == pytest.approx(values, rel=1.0e-12)
        assert processed[name].attrs["units"] == units
        assert processed[name].attrs["original_units"] == original
        # The conversion keeps the variable metadata, including its identity.
        assert processed[name].attrs["standard_name"] == source[name].attrs["standard_name"]
    assert result.success is True


@pytest.mark.parametrize(
    ("standard_name", "units"),
    [
        ("precipitation_flux", "m s-1"),
        # kg C m-2 s-1 is CF-consistent with an expressed_as_carbon name (element-aware
        # check); a nitrogen label on a carbon-basis name is not.
        ("gross_primary_productivity_of_carbon", "kg N m-2 s-1"),
    ],
)
def test_unit_conversion_rejects_cf_inconsistent_pairs(standard_name, units) -> None:
    reader = _conversion_reader()
    source = xr.Dataset({"signal": _variable([1.0e-3], standard_name, units)})
    result = _cf_result(source)

    processed = reader._apply_unit_conversion(source, reader.config, result)

    assert processed["signal"].values.tolist() == [1.0e-3]
    assert processed["signal"].attrs["units"] == units
    assert "original_units" not in processed["signal"].attrs
    assert result.success is False
    assert result.validation_passed is False
    assert any("not CF-consistent" in item for item in result.warnings)

    strict = _conversion_reader(strict_cf=True)
    with pytest.raises(ValueError, match="not CF-consistent"):
        strict._apply_unit_conversion(source, strict.config, _cf_result(source))


def test_explicit_target_units_must_stay_cf_consistent() -> None:
    """mm day-1 is a depth rate, not a precipitation_flux unit (needs density)."""
    reader = _conversion_reader(target_units={"precipitation": "mm day-1"})
    source = xr.Dataset(
        {"precipitation": _variable([1.0e-4], "precipitation_flux", "kg m-2 s-1")}
    )
    result = _cf_result(source)

    processed = reader._apply_unit_conversion(source, reader.config, result)

    assert processed["precipitation"].values.tolist() == [1.0e-4]
    assert processed["precipitation"].attrs["units"] == "kg m-2 s-1"
    assert result.success is False
    assert any("mm day-1" in item and "not CF-consistent" in item for item in result.warnings)


def test_unit_conversion_uses_no_dataset_derived_thermodynamic_context(monkeypatch) -> None:
    """Dataset temperature/pressure means are never passed as conversion context."""
    import relic.data.io.reader as reader_module

    reader = _conversion_reader()
    assert not hasattr(reader, "_extract_conversion_context")
    captured = []
    original_convert = reader_module.unit_converter.convert

    def capture(data, variable_name, source_units, target_units, context=None):
        captured.append(context)
        return original_convert(data, variable_name, source_units, target_units, context)

    monkeypatch.setattr(reader_module.unit_converter, "convert", capture)
    source = xr.Dataset(
        {
            "air_temperature": _variable([280.0, 300.0], "air_temperature", "K"),
            "air_pressure": _variable([100000.0, 90000.0], "air_pressure", "Pa"),
            "precipitation": _variable(
                [8.64, 17.28], "precipitation_flux", "kg m-2 day-1"
            ),
        }
    )
    result = _cf_result(source)

    processed = reader._apply_unit_conversion(source, reader.config, result)

    assert processed["precipitation"].values.tolist() == pytest.approx([1.0e-4, 2.0e-4])
    assert captured, "the precipitation conversion must reach the unit owner"
    for context in captured:
        assert context.temperature is None
        assert context.pressure is None
        assert context.soil_depth is None
        assert context.water_density is None
        assert context.allow_defaults is False
    assert result.success is True


def test_quantity_change_needing_context_is_rejected_not_guessed() -> None:
    """ppm -> mol m-3 changes the CF quantity; T/P variables are not borrowed.

    With T = [280, 300] K and p = [1.0e5, 9.0e4] Pa, 400 ppm is
    [0.017182, 0.014433] mol m-3 pointwise; the former dataset-mean context
    (290 K, 9.5e4 Pa) would have written 0.015760 for both points.
    """
    reader = _conversion_reader(target_units={"co2": "mol m-3"})
    source = xr.Dataset(
        {
            "co2": _variable([400.0, 400.0], "mole_fraction_of_carbon_dioxide_in_air", "ppm"),
            "air_temperature": _variable([280.0, 300.0], "air_temperature", "K"),
            "air_pressure": _variable([100000.0, 90000.0], "air_pressure", "Pa"),
        }
    )
    result = _cf_result(source)

    processed = reader._apply_unit_conversion(source, reader.config, result)

    assert processed["co2"].values.tolist() == [400.0, 400.0]
    assert processed["co2"].attrs["units"] == "ppm"
    assert result.success is False
    assert any("co2" in item and "not CF-consistent" in item for item in result.warnings)


def test_unit_conversion_failure_is_not_reported_as_success() -> None:
    """degF is CF-consistent for air_temperature but has no conversion rule."""
    reader = _conversion_reader()
    source = xr.Dataset({"temperature": _variable([68.0], "air_temperature", "degF")})
    result = _cf_result(source)

    processed = reader._apply_unit_conversion(source, reader.config, result)

    assert processed["temperature"].values.tolist() == [68.0]
    assert processed["temperature"].attrs["units"] == "degF"
    assert result.success is False
    assert any("Unit conversion failed for temperature" in item for item in result.warnings)

    strict = _conversion_reader(strict_cf=True)
    with pytest.raises(ValueError, match="Unsupported conversion"):
        strict._apply_unit_conversion(source, strict.config, _cf_result(source))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, ["air_temperature", "precipitation_flux"]),
        ({"variables": ["air_temperature"]}, ["air_temperature"]),
        ({"exclude_vars": ["precipitation_flux"]}, ["air_temperature"]),
        ({"to_torch": True}, ["air_temperature", "precipitation_flux"]),
    ],
    ids=["no_selection", "variables", "exclude_vars", "tensors"],
)
def test_variables_read_lists_the_variables_the_call_returned(
    monkeypatch: pytest.MonkeyPatch, overrides, expected,
) -> None:
    """variables_read names the data variables actually returned, with or without selection."""
    reader = _reader()
    dataset = xr.Dataset(
        {
            "air_temperature": ("time", [280.0, 281.0]),
            "precipitation_flux": ("time", [0.0, 1.0e-5]),
        }
    )
    monkeypatch.setattr(
        reader.io_handler,
        "read",
        lambda _path: IOResult(
            data=dataset,
            format_detected=DataFormat.NETCDF4,
            file_size=0,
            load_time=0.0,
        ),
    )

    result = reader.read_dataset("input.nc", **overrides)

    returned = (
        list(result.data.data_vars) if isinstance(result.data, xr.Dataset) else list(result.data)
    )
    assert returned == expected
    assert result.variables_read == expected


def test_reader_processing_conversion_and_close_errors_raise_in_every_mode(tmp_path, monkeypatch):
    """Non-strict CF mode does not return raw data, drop variables, or hide close errors."""
    import numpy as np
    import pytest
    import xarray as xr

    from relic.data.io.reader import DataReader, ReadConfig

    path = tmp_path / "input.nc"
    xr.Dataset({"tas": ("time", np.array([280.0, 281.0]), {"units": "K"})}).to_netcdf(path, engine="netcdf4")
    reader = DataReader(ReadConfig(strict_cf=False))

    def failing_process(*args, **kwargs):
        raise ValueError("unit conversion failed")

    with monkeypatch.context() as patched:
        patched.setattr(reader, "_process_pipeline", failing_process)
        with pytest.raises(ValueError, match="unit conversion failed"):
            reader.read_dataset(path)

    class _Closing:
        def close(self):
            raise OSError("handle lost")

    with pytest.raises(OSError, match="handle lost"):
        reader._safe_close_dataset(_Closing())


@pytest.mark.parametrize("explicit_bounds", [False, True])
def test_variable_selection_preserves_coordinate_bounds_without_extra_data(
    tmp_path, explicit_bounds,
):
    """Selecting a forcing must retain its source support, not unrelated data."""
    import numpy as np

    dataset = xr.Dataset(
        data_vars={
            "air_temperature": ("time", [280.0, 281.0], {"units": "K"}),
            "excluded_observation": ("time", [3.0, 4.0]),
        },
        coords={
            "time": ("time", [900.0, 2700.0], {
                "units": "seconds since 2020-02-29", "calendar": "proleptic_gregorian",
                "bounds": "time_bounds", "standard_name": "time", "axis": "T",
            }),
            "bounds": [0, 1],
            "time_bounds": (("time", "bounds"), [[0.0, 1800.0], [1800.0, 3600.0]]),
        },
    )
    path = tmp_path / "coordinate_bounds.nc"
    dataset.to_netcdf(path, engine="netcdf4")
    names = ["air_temperature", "time_bounds"] if explicit_bounds else ["air_temperature"]
    reader = DataReader(ReadConfig(variables=names, parallel=False, process_time=True,
                                  time_bounds=False, cf_standardize=False, validate_data=False))
    result = reader.read_dataset(path)
    try:
        assert result.variables_read == ["air_temperature"]
        assert "excluded_observation" not in result.data
        assert "time_bounds" in result.data.coords
        assert result.data.time.attrs["calendar"] == "proleptic_gregorian"
        np.testing.assert_array_equal(result.data.time_bounds.values, dataset.time_bounds.values)
        assert not any("time_bounds' not found" in message for message in result.warnings)
    finally:
        result.data.close()


def test_excluding_declared_bounds_does_not_silently_restore_raw_dataset(tmp_path):
    from relic.coupling.scales.temporal.core import ValidationError

    dataset = xr.Dataset(
        {"air_temperature": ("time", [280.0])},
        coords={"time": ("time", [900.0], {"units": "seconds since 2020-01-01",
                "calendar": "proleptic_gregorian", "bounds": "time_bounds"}),
                "time_bounds": (("time", "bounds"), [[0.0, 1800.0]])},
    )
    path = tmp_path / "excluded_bounds.nc"
    dataset.to_netcdf(path, engine="netcdf4")
    reader = DataReader(ReadConfig(variables=["air_temperature"], exclude_vars=["time_bounds"],
                                  parallel=False, process_time=True, time_bounds=False,
                                  cf_standardize=False, validate_data=False))
    with pytest.raises(ValidationError, match="Declared CF time-bounds variable is missing"):
        reader.read_dataset(path)
