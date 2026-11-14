"""Calendar-aware frequency tests for the CMIP6 adapter."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

from relic.coupling.external.climate_models.cmip6_adapter import (
    CMIP6Adapter,
    CMIP6Config,
)


def _annual_coordinate(calendar_type: str) -> xr.DataArray:
    index = xr.date_range(
        "2001-01-01",
        periods=3,
        freq="YS",
        calendar=calendar_type,
        use_cftime=True,
    )
    return xr.DataArray(index, dims=("time",))


def test_annual_frequency_uses_realized_calendar_year(tmp_path) -> None:
    observed_spans = []

    for calendar_type in ("noleap", "360_day"):
        coordinate = _annual_coordinate(calendar_type)
        adapter = CMIP6Adapter(
            CMIP6Config(root_path=tmp_path, calendar_type=calendar_type)
        )

        assert adapter._infer_frequency(coordinate) == "yearly"
        observed_spans.append(np.diff(coordinate.values)[0].days)

    assert observed_spans == [365, 360]
    assert observed_spans[0] != observed_spans[1]


def test_calendar_specific_year_threshold_changes_boundary_decision(tmp_path) -> None:
    values = np.array(
        ["2001-01-01", "2001-12-29", "2002-12-26"],
        dtype="datetime64[D]",
    )
    coordinate = xr.DataArray(values, dims=("time",))

    noleap = CMIP6Adapter(CMIP6Config(root_path=tmp_path, calendar_type="noleap"))
    day360 = CMIP6Adapter(CMIP6Config(root_path=tmp_path, calendar_type="360_day"))

    assert noleap._infer_frequency(coordinate) == "monthly"
    assert day360._infer_frequency(coordinate) == "yearly"


def test_frequency_inference_requires_calendar_context(tmp_path) -> None:
    coordinate = xr.DataArray(
        np.array(
            ["2001-01-01", "2002-01-01", "2003-01-01"],
            dtype="datetime64[D]",
        ),
        dims=("time",),
    )
    adapter = CMIP6Adapter(CMIP6Config(root_path=tmp_path))

    with pytest.raises(ValueError, match="calendar_type"):
        adapter._infer_frequency(coordinate)


def test_frequency_inference_rejects_calendar_conflict(tmp_path) -> None:
    adapter = CMIP6Adapter(
        CMIP6Config(root_path=tmp_path, calendar_type="noleap")
    )

    with pytest.raises(ValueError, match="conflicts"):
        adapter._infer_frequency(_annual_coordinate("360_day"))


def test_frequency_inference_rejects_nonfinite_time_difference(tmp_path) -> None:
    coordinate = xr.DataArray(
        np.array(["2001-01-01", "NaT", "2002-01-01"], dtype="datetime64[D]"),
        dims=("time",),
    )
    adapter = CMIP6Adapter(
        CMIP6Config(root_path=tmp_path, calendar_type="noleap")
    )

    with pytest.raises(ValueError, match="finite"):
        adapter._infer_frequency(coordinate)


def test_frequency_inference_has_no_fixed_year_literal() -> None:
    source = inspect.getsource(CMIP6Adapter._infer_frequency)

    assert "365" not in source
    assert "days_per_simulator_year" in source


def _cmip6_adapter_with_rsds(tmp_path, lon_bounds=None, with_bounds=True):
    """CMIP6 adapter holding one rsds field (1 time step, 2 x 4 cells) with CF cell bounds."""
    data_vars = {
        "rsds": (
            ("time", "lat", "lon"),
            np.array([[[100.0, 200.0, 300.0, 400.0], [500.0, 600.0, 700.0, 800.0]]]),
            {"units": "W m-2", "standard_name": "shortwave_down_in_air"},
        )
    }
    lat_attrs = {"units": "degrees_north", "standard_name": "latitude"}
    lon_attrs = {"units": "degrees_east", "standard_name": "longitude"}
    if with_bounds:
        lat_attrs["bounds"] = "lat_bnds"
        lon_attrs["bounds"] = "lon_bnds"
        data_vars["lat_bnds"] = (("lat", "bnds"), np.array([[-1.0, 0.0], [0.0, 1.0]]))
        data_vars["lon_bnds"] = (
            ("lon", "bnds"),
            np.array(lon_bounds if lon_bounds is not None else [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        )
    dataset = xr.Dataset(
        data_vars,
        coords={
            "time": np.array([0.0]),
            "lat": ("lat", np.array([-0.5, 0.5]), lat_attrs),
            "lon": ("lon", np.array([0.5, 1.5, 2.5, 3.5]), lon_attrs),
        },
    )
    adapter = CMIP6Adapter(CMIP6Config(root_path=tmp_path, calendar_type="noleap"))
    adapter.current_datasets = {"rsds": dataset}
    return adapter


_TARGET_GRID = {
    "lat": np.array([0.0]),
    "lon": np.array([1.0, 3.0]),
    "lat_bounds": np.array([[-1.0, 1.0]]),
    "lon_bounds": np.array([[0.0, 2.0], [2.0, 4.0]]),
}


def test_cmip6_read_and_conservative_regrid_numbers(tmp_path) -> None:
    import torch

    adapter = _cmip6_adapter_with_rsds(tmp_path)
    tensors = adapter.read(variables=["shortwave_down"])
    transformed = adapter.transform_to_internal(tensors, target_grid=dict(_TARGET_GRID))
    field = transformed["shortwave_down"]
    # Catalog basis area_density -> conservative: (100+200+500+600)/4, (300+400+700+800)/4 W m-2
    torch.testing.assert_close(
        field.detach(), torch.tensor([[[350.0, 550.0]]], dtype=torch.float32), rtol=1e-5, atol=1e-3
    )
    meta = field._relic_meta
    assert meta["spatial_basis"] == "area_density"
    assert meta["regrid_method"] == "conservative"
    assert meta["conservation_error"] < 1e-6


def test_cmip6_regrid_rejects_undeclared_variables_instead_of_returning_raw_data(tmp_path) -> None:
    adapter = _cmip6_adapter_with_rsds(tmp_path)
    tensor = adapter.read(variables=["shortwave_down"])[
        "shortwave_down"
    ]
    tensor._relic_meta = dict(tensor._relic_meta, variable="not_in_any_catalog")
    with pytest.raises(KeyError, match="not_in_any_catalog"):
        adapter._apply_regridding(tensor, dict(_TARGET_GRID))
    source = inspect.getsource(CMIP6Adapter._apply_regridding)
    assert "Regridding failed" not in source
    assert "method='conservative'" not in source
    assert "Failed to transform" not in inspect.getsource(CMIP6Adapter.transform_to_internal)


def test_quality_control_uses_the_catalog_valid_range(tmp_path) -> None:
    """rsds valid range [0, 2200] W m-2 from the catalog (the greatest measured surface irradiance,
    2177 W m-2, rounded up); out-of-range values become missing."""
    import torch

    adapter = CMIP6Adapter(CMIP6Config(root_path=tmp_path, calendar_type="noleap"))
    field = torch.tensor([100.0, float("nan"), 2500.0, 300.0])
    checked = adapter._apply_quality_control(field, "shortwave_down")
    assert torch.isnan(checked[1]) and torch.isnan(checked[2])
    torch.testing.assert_close(checked[[0, 3]], torch.tensor([100.0, 300.0]))
    report = checked._relic_meta["quality_control"]
    assert report["missing_fraction"] == pytest.approx(0.5)
    assert report["out_of_valid_range"] == 1
    assert report["valid_range"] == (0.0, 2200.0)


def test_quality_control_rejects_fields_above_the_missing_threshold(tmp_path) -> None:
    import torch

    adapter = CMIP6Adapter(CMIP6Config(root_path=tmp_path, calendar_type="noleap"))
    # 3 of 4 values missing (0.75) exceed coupling.external.regrid.missing_threshold = 0.5.
    field = torch.tensor([float("nan"), float("nan"), -5.0, 300.0])
    with pytest.raises(ValueError, match="missing"):
        adapter._apply_quality_control(field, "shortwave_down")


def test_quality_control_has_no_hardcoded_bounds_or_zero_fill() -> None:
    from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter

    for cls in (CMIP6Adapter, WRFAdapter):
        source = inspect.getsource(cls)
        assert "zeros_like(tensor)" not in source, cls.__name__
        assert "nan_to_num" not in source, cls.__name__
    assert not hasattr(CMIP6Adapter, "_check_physical_bounds")


def test_cmip6_source_bounds_come_from_the_file_not_from_spacing(tmp_path) -> None:
    """Cell 2 spans 2.0-3.5 degrees and cell 3 3.5-4.0 (centres are not midpoints): target 1 =
    (1.5 (300 + 700) + 0.5 (400 + 800)) / 4 = 525 W m-2 (spacing-inferred bounds would give 550)."""
    import torch

    adapter = _cmip6_adapter_with_rsds(
        tmp_path, lon_bounds=[[0.0, 1.0], [1.0, 2.0], [2.0, 3.5], [3.5, 4.0]]
    )
    tensors = adapter.read(variables=["shortwave_down"])
    field = adapter.transform_to_internal(tensors, target_grid=dict(_TARGET_GRID))[
        "shortwave_down"
    ]
    torch.testing.assert_close(
        field.detach(), torch.tensor([[[350.0, 525.0]]], dtype=torch.float32), rtol=1e-4, atol=1e-2
    )


def test_cmip6_conservative_regrid_without_cell_bounds_raises(tmp_path) -> None:
    adapter = _cmip6_adapter_with_rsds(tmp_path, with_bounds=False)
    tensors = adapter.read(variables=["shortwave_down"])
    with pytest.raises(ValueError, match="cell bounds"):
        adapter.transform_to_internal(tensors, target_grid=dict(_TARGET_GRID))
