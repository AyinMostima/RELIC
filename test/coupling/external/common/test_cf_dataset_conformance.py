"""G11-CF: declared-scope dataset conformance on real NetCDF read/write products.

A conforming dataset is written with xarray/netCDF4, read back through the
I/O owner, and its declared-scope CF conformance must hold; the attributes
are cross-checked against the raw netCDF4 file. Each negative dataset breaks
exactly one CF rule inside the scope and must be reported, never defaulted
to success. Rules outside the scope stay unevaluated by construction.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from relic.coupling.external.common.io import CF_DECLARED_SCOPE, IOConfig, IOHandler, IOResult
from relic.data.standards.cf_conventions import (
    CF_CALENDARS,
    CF_DATASET_CONFORMANCE_SCOPE,
    CFConventions,
    CFDatasetConformanceResult,
)


def _conforming_dataset() -> xr.Dataset:
    time = np.array([0.0, 1800.0, 3600.0])
    time_bounds = np.stack([time, time + 1800.0], axis=1)
    dataset = xr.Dataset(
        {
            "air_temperature": ("time", np.array([280.0, 281.0, 282.5])),
            "time_bounds": (("time", "nv"), time_bounds),
        },
        coords={"time": ("time", time)},
        attrs={"Conventions": "CF-1.8", "title": "RELIC declared-scope conformance fixture"},
    )
    dataset["air_temperature"].attrs.update({
        "standard_name": "air_temperature",
        "units": "K",
        "cell_methods": "time: mean",
        "coordinates": "time",
    })
    dataset["time"].attrs.update({
        "standard_name": "time",
        "units": "seconds since 2020-01-01 00:00:00",
        "calendar": "noleap",
        "axis": "T",
        "bounds": "time_bounds",
    })
    dataset["time_bounds"].attrs.update({"units": "seconds since 2020-01-01 00:00:00"})
    return dataset


def _write(dataset: xr.Dataset, path: Path) -> Path:
    encoding = {name: {"zlib": True, "complevel": 1} for name in dataset.data_vars}
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


def _load(path: Path) -> IOResult:
    handler = IOHandler(IOConfig(decode_times=False, enable_cache=False, validate_checksums=False))
    return handler.read(path)


def test_conforming_netcdf_product_passes_the_declared_scope_and_matches_the_raw_file(tmp_path: Path):
    path = _write(_conforming_dataset(), tmp_path / "conforming.nc")
    result = _load(path)
    assert result.dataset_conformance_checked is True
    assert result.cf_validation_scope == CF_DECLARED_SCOPE
    assert result.cf_conventions_declared is True
    assert result.cf_conformance_issues == ()
    assert result.cf_compliant is True
    # Cross-check the product itself, not the in-memory object.
    with netCDF4.Dataset(path) as raw:
        assert raw.getncattr("Conventions") == "CF-1.8"
        assert raw["air_temperature"].getncattr("standard_name") == "air_temperature"
        assert raw["air_temperature"].getncattr("units") == "K"
        assert raw["time"].getncattr("calendar") == "noleap"
        assert raw["time"].getncattr("bounds") == "time_bounds"
        assert raw["time_bounds"].shape == (3, 2)
    direct = CFConventions.validate_dataset_conformance(xr.open_dataset(path, decode_times=False))
    assert direct.is_valid and direct.variables_checked == 2 and direct.coordinates_checked == 1


@pytest.mark.parametrize(
    "cell_methods",
    [
        "time: sum (interval: 1 hour)",
        "time: mean (interval: 1 hour)",
        "time: standard_deviation (interval: 1 hour comment: sampled values)",
        "time: mean (comment: ghost: nonsense is descriptive text)",
        "area: mean (comment: spatial mean) time: mean (interval: 1 hour)",
    ],
)
def test_cell_methods_comments_survive_netcdf_without_becoming_dimensions(tmp_path: Path, cell_methods):
    dataset = _conforming_dataset()
    dataset["air_temperature"].attrs["cell_methods"] = cell_methods
    expected_values = dataset["air_temperature"].values.copy()
    path = _write(dataset, tmp_path / "cell_methods_comments.nc")
    result = _load(path)
    assert result.cf_compliant is True, result.cf_conformance_issues
    assert result.cf_conformance_issues == ()
    with netCDF4.Dataset(path) as raw:
        assert raw["air_temperature"].getncattr("cell_methods") == cell_methods
        assert raw["air_temperature"].getncattr("units") == "K"
        np.testing.assert_array_equal(raw["air_temperature"][:], expected_values)
        np.testing.assert_array_equal(raw["time_bounds"][:], dataset["time_bounds"].values)


@pytest.mark.parametrize(
    "method",
    [
        "point", "sum", "anomaly_wrt", "maximum", "maximum_absolute_value",
        "median", "mid_range", "minimum", "minimum_absolute_value", "mean",
        "mean_absolute_value", "mean_of_upper_decile", "mode", "range",
        "root_mean_square", "standard_deviation", "sum_of_squares", "variance",
    ],
)
def test_cell_methods_accepts_the_official_method_vocabulary(method):
    # CF-1.13 Appendix E, Table E.1; only metadata syntax is evaluated here.
    dataset = _conforming_dataset()
    dataset["air_temperature"].attrs["cell_methods"] = f"time: {method}"
    result = CFConventions.validate_dataset_conformance(dataset)
    assert result.is_valid is True, result.issues


@pytest.mark.parametrize(
    "cell_methods, expected_fragment",
    [
        ("ghost: mean (interval: 1 hour)", "unknown dimension 'ghost'"),
        ("time: mean (interval: 1 hour) ghost: sum", "unknown dimension 'ghost'"),
        ("ghost, time: mean (interval: 1 hour)", "unknown dimension 'ghost'"),
        ("time: nonsense", "unknown method 'nonsense'"),
        ("time: nonsense (interval: 1 hour)", "unknown method 'nonsense'"),
        ("time: mean (comment: text) area: nonsense", "unknown method 'nonsense'"),
        ("time: mean (interval: 1 hour", "not 'name: method' syntax"),
        ("time: mean interval: 1 hour)", "not 'name: method' syntax"),
        ("time: mean (interval: (1 hour))", "not 'name: method' syntax"),
        ("time: mean (interval: 1 hour))", "not 'name: method' syntax"),
    ],
)
def test_cell_methods_rejects_invalid_top_level_tokens_and_malformed_comments(cell_methods, expected_fragment):
    dataset = _conforming_dataset()
    dataset["air_temperature"].attrs["cell_methods"] = cell_methods
    result = CFConventions.validate_dataset_conformance(dataset)
    assert result.is_valid is False
    assert any(expected_fragment in issue for issue in result.issues), result.issues


@pytest.mark.parametrize(
    "mutate, expected_fragment",
    [
        (lambda ds: ds.attrs.pop("Conventions"), "'Conventions' is missing"),
        (lambda ds: ds["air_temperature"].attrs.pop("units"), "has no units attribute"),
        (lambda ds: ds["air_temperature"].attrs.update({"units": "furlongs_per_fortnight_x"}), "not a recognized UDUNITS"),
        (lambda ds: ds["air_temperature"].attrs.update({"standard_name": "not_a_cf_name"}), "standard name"),
        (lambda ds: ds["air_temperature"].attrs.update({"units": "kg"}), "not physically compatible"),
        (lambda ds: ds["time"].attrs.update({"calendar": "martian"}), "is not a CF calendar"),
        (lambda ds: ds["time"].attrs.update({"bounds": "missing_bounds"}), "references missing variable"),
        (lambda ds: ds["time"].attrs.update({"axis": "Q"}), "axis"),
        (lambda ds: ds["air_temperature"].attrs.update({"coordinates": "ghost"}), "coordinates attribute references missing"),
        (lambda ds: ds["air_temperature"].attrs.update({"cell_methods": "mean"}), "cell_methods"),
        (lambda ds: ds["air_temperature"].attrs.update({"_FillValue": -999.0, "missing_value": -1.0}), "_FillValue and missing_value differ"),
    ],
)
def test_each_declared_rule_failure_is_reported(tmp_path: Path, mutate, expected_fragment):
    dataset = _conforming_dataset()
    mutate(dataset)
    result = CFConventions.validate_dataset_conformance(dataset)
    assert result.is_valid is False
    assert any(expected_fragment in issue for issue in result.issues), result.issues


def test_io_result_flags_are_consistent_with_the_declared_scope():
    with pytest.raises(ValueError, match="declared-scope check"):
        IOResult(data=xr.Dataset(), format_detected=None, file_size=0, load_time=0.0, cf_compliant=True)
    with pytest.raises(ValueError, match="exactly when"):
        IOResult(data=xr.Dataset(), format_detected=None, file_size=0, load_time=0.0, dataset_conformance_checked=True)
    with pytest.raises(ValueError, match="conformance issues are recorded"):
        IOResult(
            data=xr.Dataset(), format_detected=None, file_size=0, load_time=0.0,
            cf_compliant=True, dataset_conformance_checked=True, cf_validation_scope=CF_DECLARED_SCOPE,
            cf_conformance_issues=("x",),
        )
    checked = IOResult(
        data=xr.Dataset(), format_detected=None, file_size=0, load_time=0.0,
        dataset_conformance_checked=True, cf_validation_scope=CF_DECLARED_SCOPE, cf_compliant=True,
    )
    assert checked.cf_compliant is True


def test_bounded_scope_does_not_claim_rules_outside_it():
    dataset = _conforming_dataset()
    # A grid_mapping attribute is outside the declared scope: not evaluated,
    # therefore neither accepted nor rejected.
    dataset["air_temperature"].attrs["grid_mapping"] = "crs"
    result = CFConventions.validate_dataset_conformance(dataset)
    assert result.is_valid is True and result.scope == CF_DECLARED_SCOPE
    assert "standard" in CF_CALENDARS and "noleap" in CF_CALENDARS


def test_conformance_result_record_has_no_defaulted_counts_or_scope():
    # The counts and the scope describe what one check evaluated: the record
    # carries no default that could report "0 checked" or a scope nobody set.
    fields = CFDatasetConformanceResult.__dataclass_fields__
    for name in ("variables_checked", "coordinates_checked", "scope"):
        assert fields[name].default is dataclasses.MISSING, name
    with pytest.raises(TypeError):
        CFDatasetConformanceResult(is_valid=True)
    result = CFConventions.validate_dataset_conformance(_conforming_dataset())
    assert result.scope == CF_DATASET_CONFORMANCE_SCOPE == CF_DECLARED_SCOPE
