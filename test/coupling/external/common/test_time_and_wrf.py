from datetime import datetime
from pathlib import Path

import pytest
import torch
import xarray as xr

import relic.coupling.external.atmospheric_models.wrf_adapter as wrf_module
from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter, WRFConfig
import relic.coupling.external.common as external_common
from relic.coupling.external.common import time as time_module
from relic.coupling.scales.temporal.core import (
    CalendarError,
    TimeCoordinate,
    ValidationError,
)
from relic.parameters import get_global_manager


class _StubParameterManager:
    def __init__(
        self,
        time_format: str = "%Y/%m/%d_%H",
        reference_epoch: str = "2000-01-01 00:00:00",
    ):
        self.time_format = time_format
        self.reference_epoch = reference_epoch

    def get(self, key, namespace=None, **kwargs):
        full_key = f"{namespace}.{key}" if namespace else str(key)
        lookup = {
            "coupling.external.wrf.default_root_path": "./data/wrf",
            "coupling.external.wrf.conversion_namespace": "conversion_context",
            "coupling.external.wrf.time_format": self.time_format,
            "coupling.external.time.reference_epoch": self.reference_epoch,
            "conversion_context.wrf_west_east_count": 1,
            "conversion_context.wrf_south_north_count": 1,
            "conversion_context.wrf_bottom_top_count": 1,
            "conversion_context.reference_pressure": 100000.0,
            "relic.physical.constants.air.R_over_cp": 0.286,
        }
        return lookup.get(full_key)


class _ProxyParameterManager:
    def __init__(self, base_pm, override_time_format: str):
        self._base_pm = base_pm
        self._override_time_format = override_time_format

    def get(self, key, namespace=None, **kwargs):
        full_key = f"{namespace}.{key}" if namespace else str(key)
        if full_key == "coupling.external.wrf.time_format":
            return self._override_time_format
        return self._base_pm.get(key, namespace=namespace, **kwargs)

    def __getattr__(self, item):
        return getattr(self._base_pm, item)


class _PhysicalConstantGuard(_ProxyParameterManager):
    def get(self, key, namespace=None, **kwargs):
        full_key = f"{namespace}.{key}" if namespace else str(key)
        if full_key.startswith("relic.physical.constants."):
            raise AssertionError("Physical constants must use get_physical_constant")
        return super().get(key, namespace=namespace, **kwargs)


def test_time_handler_uses_parameter_epoch(monkeypatch):
    """TimeHandler should pull reference epoch from parameter system when available."""
    custom_epoch = "1999-12-31 23:59:59"
    monkeypatch.setattr(
        time_module,
        "get_global_manager",
        lambda: _StubParameterManager(reference_epoch=custom_epoch),
    )
    handler = time_module.TimeHandler()
    assert handler._reference_epoch == datetime(1999, 12, 31, 23, 59, 59)


@pytest.mark.parametrize(
    ("metadata_location", "calendar", "expected"),
    [
        ("attrs", "noleap", "noleap"),
        ("encoding", "360_day", "360_day"),
        ("attrs", "standard", "gregorian"),
    ],
)
def test_time_handler_detects_explicit_cf_calendar(
    metadata_location,
    calendar,
    expected,
):
    """External I/O must delegate explicit CF calendar normalization to the canonical owner."""
    dataset = xr.Dataset(coords={"time": ("time", [0.5, 1.5])})
    getattr(dataset.time, metadata_location)["calendar"] = calendar

    assert time_module.TimeHandler().detect_calendar(dataset) == expected


def test_time_handler_rejects_missing_invalid_and_conflicting_calendars():
    """RELIC strict I/O requires one unambiguous calendar binding."""
    handler = time_module.TimeHandler()
    missing = xr.Dataset(coords={"time": ("time", [0.5, 1.5])})
    invalid = missing.copy(deep=True)
    invalid.time.attrs["calendar"] = "martian"
    conflicting = missing.copy(deep=True)
    conflicting.time.attrs["calendar"] = "noleap"
    conflicting.time.encoding["calendar"] = "360_day"

    with pytest.raises(CalendarError, match="explicit calendar"):
        handler.detect_calendar(missing)
    with pytest.raises(CalendarError, match="Unsupported calendar"):
        handler.detect_calendar(invalid)
    with pytest.raises(CalendarError, match="Conflicting calendar"):
        handler.detect_calendar(conflicting)


def test_time_handler_requires_one_unambiguous_cf_units_declaration():
    """CF unit metadata must be explicit and consistent across attrs and encoding."""
    handler = time_module.TimeHandler()
    valid = xr.Dataset(coords={"time": ("time", [0.5, 1.5])})
    valid.time.encoding["units"] = "days since 2001-01-01"
    missing = xr.Dataset(coords={"time": ("time", [0.5, 1.5])})
    conflicting = valid.copy(deep=True)
    conflicting.time.attrs["units"] = "hours since 2001-01-01"

    assert handler.detect_time_units(valid) == "days since 2001-01-01"
    with pytest.raises(ValidationError, match="explicit CF time units"):
        handler.detect_time_units(missing)
    with pytest.raises(ValidationError, match="Conflicting CF time units"):
        handler.detect_time_units(conflicting)


def test_time_handler_returns_canonical_torch_coordinate_and_bounds():
    """All calendar arithmetic must remain in the canonical Torch temporal owner."""
    coordinate = time_module.TimeHandler().create_time_coordinate(
        values=[0.5],
        units="days since 2001-01-01",
        calendar="noleap",
        compute_bounds=True,
        frequency="D",
    )

    assert isinstance(coordinate, TimeCoordinate)
    assert coordinate.calendar == "noleap"
    assert coordinate.values.dtype == torch.float64
    assert torch.equal(coordinate.values.cpu(), torch.tensor([0.5], dtype=torch.float64))
    assert torch.equal(
        coordinate.bounds.cpu(),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    )


def test_time_handler_requires_evidence_for_singleton_bounds():
    """One center point cannot determine an interval without bounds or frequency evidence."""
    with pytest.raises(ValidationError, match="singleton"):
        time_module.TimeHandler().create_time_coordinate(
            values=[0.5],
            units="days since 2001-01-01",
            calendar="360_day",
            compute_bounds=True,
        )


def test_time_handler_rejects_ambiguous_cross_calendar_conversion():
    """Cross-calendar conversion is non-operational until a preservation policy is bound."""
    coordinate = TimeCoordinate(
        values=torch.tensor([0.5], dtype=torch.float64),
        units="days since 2001-01-01",
        calendar="noleap",
    )

    with pytest.raises(CalendarError, match="mapping policy"):
        time_module.TimeHandler().convert_calendar(coordinate, "gregorian")


def test_external_common_exports_only_the_canonical_time_coordinate():
    """The external namespace must not advertise a second calendar type system."""
    assert external_common.TimeCoordinate is TimeCoordinate
    for removed_name in ("CalendarType", "TemporalWindow", "AggregationMethod"):
        assert removed_name not in external_common.__all__
        with pytest.raises(AttributeError):
            getattr(external_common, removed_name)


def test_wrf_adapter_applies_time_format_override(monkeypatch):
    """WRFAdapter should respect time_format provided by ParameterManager."""
    base_pm = get_global_manager()
    proxy_pm = _ProxyParameterManager(base_pm, override_time_format="%Y/%m/%d_%H%M")
    adapter = WRFAdapter(parameter_manager=proxy_pm)
    assert adapter.config.time_format == proxy_pm._override_time_format
    assert Path(adapter.config.root_path).as_posix().endswith("data/wrf")


def test_wrf_open_adds_no_unread_diagnostic_fields():
    """open() with extract_diagnostics added WSPD, WSPD10, PRESSURE and TEMP to the opened
    datasets. No code read them (the read path reads only the schema-mapped WRF fields, and the
    adapter outputs are its provided catalog variables), and WSPD combined the staggered U and V
    when unstagger_grids was off, whose dimensions differ. The fields, their switch, the
    thermodynamic constants they used and conversion_context.potential_temperature_reference are
    gone; the adapter reads no physical constant through its ParameterManager."""
    import dataclasses
    import inspect

    assert not hasattr(WRFAdapter, "_compute_diagnostics")
    assert "extract_diagnostics" not in {item.name for item in dataclasses.fields(WRFConfig)}
    guarded_pm = _PhysicalConstantGuard(
        get_global_manager(),
        override_time_format="%Y-%m-%d_%H:%M:%S",
    )
    adapter = WRFAdapter(
        config=WRFConfig(root_path=Path("./test_wrf_path")),
        parameter_manager=guarded_pm,
    )
    for attribute in ("base_potential_temperature", "reference_pressure_pa", "_dry_air_R_over_cp"):
        assert not hasattr(adapter, attribute), attribute
    source = inspect.getsource(wrf_module)
    for stale in ("WSPD", "potential_temperature_reference", "extract_diagnostics", "R_over_cp"):
        assert stale not in source, stale
    with pytest.raises(KeyError):
        get_global_manager().get("conversion_context.potential_temperature_reference", required=True)


class _CapturingRegridOperator:
    """Records the RegridConfig handed over by the WRF adapter and returns the input unchanged."""

    def __init__(self):
        self.configs = []

    def regrid(self, source_data, source_grid, target_grid, config):
        self.configs.append(config)

        class _Result:
            data = source_data
            conservation_error = 0.0
            regrid_time = 0.0

        return _Result()


def _wrf_tensor(variable, attrs):
    """WRF field on a regular latitude-longitude grid whose CF coordinates and cell bounds were read."""
    tensor = torch.ones((2, 2), dtype=torch.float32)
    tensor._relic_meta = {
        "shape": tensor.shape,
        "dims": ("lat", "lon"),
        "coords": {"lat": torch.tensor([0.0, 1.0]), "lon": torch.tensor([0.0, 1.0])},
        "coord_dims": {"lat": ("lat",), "lon": ("lon",)},
        "attrs": dict(attrs),
        "domain": "d01",
        "variable": variable,
        "cell_bounds": {
            "lat_coordinate": "lat",
            "lon_coordinate": "lon",
            "lat_bounds": torch.tensor([[-0.5, 0.5], [0.5, 1.5]]),
            "lon_bounds": torch.tensor([[-0.5, 0.5], [0.5, 1.5]]),
        },
    }
    return tensor


def _wrf_adapter_with_capture():
    operator = _CapturingRegridOperator()
    adapter = WRFAdapter(
        config=WRFConfig(root_path=Path("./test_wrf_path")),
        regrid_operator=operator,
    )
    return adapter, operator


def test_wrf_regrid_basis_comes_from_the_variable_catalog():
    """Attributes that look like fluxes no longer decide the basis; the catalog does."""
    from relic.coupling.scales.types import SpatialQuantityBasis

    adapter, operator = _wrf_adapter_with_capture()
    target_grid = {"type": "regular", "lat": [0.0], "lon": [0.0]}

    # air_temperature is intensive in coupling.external.variables.variable_definitions even if
    # its attributes carry "flux"/"precipitation" tokens (the removed heuristic said extensive).
    adapter._apply_regridding(
        _wrf_tensor("air_temperature", {"standard_name": "precipitation_flux_like", "units": "kg m-2 s-1"}),
        target_grid,
    )
    # precipitation_flux is an area density in the catalog even without flux-like attributes.
    adapter._apply_regridding(
        _wrf_tensor("precipitation_flux", {"standard_name": "", "units": ""}),
        target_grid,
    )
    assert [config.spatial_basis for config in operator.configs] == [
        SpatialQuantityBasis.INTENSIVE,
        SpatialQuantityBasis.AREA_DENSITY,
    ]


def test_wrf_regrid_basis_from_caller_metadata_overrides_the_catalog_consistently():
    from relic.coupling.scales.types import SpatialQuantityBasis

    adapter, operator = _wrf_adapter_with_capture()
    target_grid = {"type": "regular", "lat": [0.0], "lon": [0.0]}
    adapter._apply_regridding(
        _wrf_tensor("basin_water_volume", {}),
        target_grid,
        variable_metadata={"spatial_basis": "cell_total", "valid_min": 0.0},
    )
    assert operator.configs[-1].spatial_basis is SpatialQuantityBasis.CELL_TOTAL


def test_wrf_regrid_rejects_an_undeclared_variable_instead_of_returning_it_unregridded():
    adapter, operator = _wrf_adapter_with_capture()
    tensor = _wrf_tensor("not_in_any_catalog", {"standard_name": "total_flux", "units": "W m-2"})
    with pytest.raises(KeyError, match="not_in_any_catalog"):
        adapter._apply_regridding(tensor, {"type": "regular", "lat": [0.0], "lon": [0.0]})
    assert operator.configs == []


def test_wrf_regrid_has_no_name_or_unit_heuristic():
    import inspect

    source = inspect.getsource(wrf_module.WRFAdapter._apply_regridding)
    assert "is_extensive = any(" not in source
    assert "'kg m-2 s-1', 'w m-2'" not in source


def _write_synthetic_wrfout(root):
    """One-time-step WRF-like file: 2 x 3 mass points with XLAT/XLONG coordinates."""
    import numpy as np

    dims = ("Time", "south_north", "west_east")
    dataset = xr.Dataset(
        {
            "Times": (("Time",), np.array([b"2020-01-01_00:00:00"], dtype="S19")),
            "T2": (dims, np.array([[[280.0, 281.0, 282.0], [283.0, 284.0, 285.0]]]), {"units": "K"}),
            "RH2": (dims, np.array([[[50.0, 60.0, 70.0], [80.0, 90.0, 100.0]]]), {"units": "%"}),
            "PSFC": (dims, np.array([[[100000.0, 99500.0, 99000.0], [98500.0, 98000.0, 97500.0]]]), {"units": "Pa"}),
        },
        coords={
            "XLAT": (("south_north", "west_east"), np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])),
            "XLONG": (("south_north", "west_east"), np.array([[20.0, 21.0, 22.0], [20.0, 21.0, 22.0]])),
        },
    )
    path = root / "wrfout_d01_2020-01-01_000000.nc"
    dataset.to_netcdf(path)
    return path


def test_wrf_read_and_unit_conversion_run_end_to_end(tmp_path):
    """discover -> open -> read with the schema mapping and the canonical unit converter."""
    _write_synthetic_wrfout(tmp_path)
    adapter = WRFAdapter(
        config=WRFConfig(root_path=tmp_path, unstagger_grids=False)
    )
    adapter.discover()
    adapter.open()
    tensors = adapter.read(variables=["air_temperature", "relative_humidity", "atmospheric_pressure"])

    assert set(tensors) == {"air_temperature", "relative_humidity", "atmospheric_pressure"}
    # RH2 [%] -> relative_humidity [1]: factor 0.01
    torch.testing.assert_close(
        tensors["relative_humidity"],
        torch.tensor([[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]], dtype=torch.float32),
    )
    rh_meta = tensors["relative_humidity"]._relic_meta
    assert rh_meta["attrs"]["units"] == "1"
    assert rh_meta["attrs"]["original_units"] == "%"
    assert rh_meta["variable"] == "relative_humidity"
    # Identical units pass through unchanged.
    torch.testing.assert_close(
        tensors["air_temperature"],
        torch.tensor([[[280.0, 281.0, 282.0], [283.0, 284.0, 285.0]]], dtype=torch.float32),
    )
    assert tensors["atmospheric_pressure"]._relic_meta["attrs"]["units"] == "Pa"


def test_unit_conversion_failures_raise_instead_of_returning_unconverted_data():
    adapter = WRFAdapter(config=WRFConfig(root_path=Path("./test_wrf_path")))
    # m3 m-3 -> kg m-2 needs a soil-layer thickness contract the converter does not declare.
    with pytest.raises(ValueError, match="m3 m-3"):
        adapter._apply_unit_conversion(
            xr.DataArray([0.3], dims=("cell",), attrs={"units": "m3 m-3"}),
            "soil_moisture_content",
        )
    with pytest.raises(ValueError, match="units"):
        adapter._apply_unit_conversion(xr.DataArray([280.0], dims=("cell",)), "air_temperature")
    with pytest.raises(KeyError, match="not_a_catalog_variable"):
        adapter._apply_unit_conversion(
            xr.DataArray([1.0], dims=("cell",), attrs={"units": "1"}), "not_a_catalog_variable"
        )


def test_adapter_read_loops_do_not_swallow_errors():
    import importlib
    import inspect

    for module_name, class_name in (
        ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter"),
        ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter"),
        ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter"),
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter"),
        ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter"),
    ):
        source = inspect.getsource(getattr(importlib.import_module(module_name), class_name).read)
        assert "get_variable_mappings" not in source, class_name
        assert "Failed to read" not in source, class_name


def test_wrf_regrid_positivity_comes_from_the_catalog():
    """valid_min >= 0 in the catalog decides positive_definite; standard-name tokens do not."""
    adapter, operator = _wrf_adapter_with_capture()
    target_grid = {"type": "regular", "lat": [0.0], "lon": [0.0]}
    # "radiation" in the standard name used to force positivity; valid_min -200 W m-2 does not.
    adapter._apply_regridding(
        _wrf_tensor("sensible_heat_flux", {"standard_name": "net_radiation_like"}),
        target_grid,
    )
    # No "precipitation"/"radiation" token, but the catalog valid_min is 0.
    adapter._apply_regridding(_wrf_tensor("precipitation_flux", {"standard_name": ""}), target_grid)
    assert [config.positive_definite for config in operator.configs] == [False, True]
    # Caller metadata must declare valid_min as well.
    with pytest.raises(ValueError, match="valid_min"):
        adapter._apply_regridding(
            _wrf_tensor("basin_water_volume", {}),
            target_grid,
            variable_metadata={"spatial_basis": "cell_total"},
        )
    import inspect

    source = inspect.getsource(wrf_module.WRFAdapter._apply_regridding)
    assert "'precipitation' in" not in source
    assert "'radiation' in" not in source


def _write_accumulating_wrfout(root, rainnc, i_rainnc=None, bucket_mm=None):
    """Three hourly records on 1 x 2 cells with accumulated precipitation and two soil layers."""
    import numpy as np

    times = np.array([b"2020-01-01_00:00:00", b"2020-01-01_01:00:00", b"2020-01-01_02:00:00"], dtype="S19")
    dims = ("Time", "south_north", "west_east")
    data_vars = {
        "Times": (("Time",), times),
        "RAINNC": (dims, np.asarray(rainnc, dtype=np.float64).reshape(3, 1, 2), {"units": "mm"}),
        "RAINC": (dims, np.array([0.0, 0.5, 1.0, 1.5, 1.5, 3.0]).reshape(3, 1, 2), {"units": "mm"}),
        "SMOIS": (
            ("Time", "soil_layers_stag", "south_north", "west_east"),
            np.tile(np.array([0.3, 0.3, 0.2, 0.25]).reshape(1, 2, 1, 2), (3, 1, 1, 1)),
            {"units": "m3 m-3"},
        ),
        "DZS": (("Time", "soil_layers_stag"), np.tile(np.array([0.1, 0.3]), (3, 1)), {"units": "m"}),
    }
    if i_rainnc is not None:
        data_vars["I_RAINNC"] = (dims, np.asarray(i_rainnc, dtype=np.float64).reshape(3, 1, 2), {"units": "1"})
    dataset = xr.Dataset(
        data_vars,
        coords={
            "XLAT": (("south_north", "west_east"), np.array([[10.0, 10.0]])),
            "XLONG": (("south_north", "west_east"), np.array([[20.0, 21.0]])),
        },
    )
    if bucket_mm is not None:
        dataset.attrs["BUCKET_MM"] = float(bucket_mm)
    path = root / "wrfout_d01_2020-01-01_000000.nc"
    dataset.to_netcdf(path)
    return path


def _read_wrf(root, variables):
    adapter = WRFAdapter(
        config=WRFConfig(root_path=root, unstagger_grids=False)
    )
    adapter.discover()
    adapter.open()
    return adapter.read(variables=variables)


def test_wrf_accumulations_are_de_accumulated_over_the_output_interval(tmp_path):
    """rate = rho_w (A_k - A_{k-1}) / (t_k - t_{k-1}); 1 mm of water = 1 kg m-2."""
    _write_accumulating_wrfout(tmp_path, rainnc=[0.0, 1.0, 2.0, 4.0, 5.0, 4.0])
    tensors = _read_wrf(tmp_path, ["large_scale_precipitation_flux", "convective_precipitation_flux"])
    grid = tensors["large_scale_precipitation_flux"]
    # Two hourly intervals; the first record is the initial accumulation (reported, no rate).
    expected = torch.tensor([[[2.0, 3.0]], [[3.0, 0.0]]], dtype=torch.float32) / 3600.0
    torch.testing.assert_close(grid, expected, rtol=1e-6, atol=1e-9)
    meta = grid._relic_meta
    assert meta["attrs"]["units"] == "kg m-2 s-1"
    assert "first record 2020-01-01T00:00:00" in meta["attrs"]["deaccumulation"]
    convective = tensors["convective_precipitation_flux"]
    torch.testing.assert_close(
        convective, torch.tensor([[[1.0, 1.0]], [[0.5, 1.5]]], dtype=torch.float32) / 3600.0, rtol=1e-6, atol=1e-9
    )


def test_wrf_bucket_resets_raise_unless_the_bucket_counters_are_applied(tmp_path):
    # 95 -> 5 mm: a bucket reset (bucket_mm = 100) without I_RAINNC in the file.
    reset_dir = tmp_path / "reset"
    reset_dir.mkdir()
    _write_accumulating_wrfout(reset_dir, rainnc=[90.0, 0.0, 95.0, 1.0, 5.0, 2.0])
    with pytest.raises(ValueError, match="decreases"):
        _read_wrf(reset_dir, ["large_scale_precipitation_flux"])
    # With I_RAINNC and BUCKET_MM the total A = RAINNC + I_RAINNC * BUCKET_MM is monotone.
    bucket_dir = tmp_path / "bucket"
    bucket_dir.mkdir()
    _write_accumulating_wrfout(
        bucket_dir,
        rainnc=[90.0, 0.0, 95.0, 1.0, 5.0, 2.0],
        i_rainnc=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        bucket_mm=100.0,
    )
    rate = _read_wrf(bucket_dir, ["large_scale_precipitation_flux"])["large_scale_precipitation_flux"]
    torch.testing.assert_close(
        rate, torch.tensor([[[5.0, 1.0]], [[10.0, 1.0]]], dtype=torch.float32) / 3600.0, rtol=1e-6, atol=1e-9
    )


def test_wrf_layered_soil_moisture_becomes_layer_water_mass(tmp_path):
    """m_k = rho_w theta_k dz_k with dz from DZS: 0.3 * 0.1 m -> 30 kg m-2, 0.2 * 0.3 m -> 60 kg m-2."""
    _write_accumulating_wrfout(tmp_path, rainnc=[0.0, 1.0, 2.0, 4.0, 5.0, 4.0])
    layers = _read_wrf(tmp_path, ["mass_content_of_water_in_soil_layer"])["mass_content_of_water_in_soil_layer"]
    torch.testing.assert_close(
        layers[0], torch.tensor([[[30.0, 30.0]], [[60.0, 75.0]]], dtype=torch.float32), rtol=1e-6, atol=1e-4
    )
    assert layers._relic_meta["attrs"]["units"] == "kg m-2"


def test_accumulated_and_layered_fields_are_declared_not_inferred():
    from relic.coupling.external.common.datasource import DataSource
    from relic.coupling.external.common.schema import InternalSchema

    schema = InternalSchema()
    assert schema.get_source_mapping(DataSource.WRF)["SMOIS"] == "mass_content_of_water_in_soil_layer"
    assert schema.get_source_transform(DataSource.WRF, "RAINNC")["kind"] == "accumulated"
    assert schema.get_source_transform(DataSource.WRF, "SMOIS")["thickness_variable"] == "DZS"
    assert schema.get_source_transform(DataSource.WRF, "T2") is None


def _write_wrfout_with_times(path, times):
    """Minimal wrfout-like file with the given Times strings (19 characters each)."""
    import numpy as np

    n_records = len(times)
    xr.Dataset(
        {
            "Times": (("Time",), np.array([t.encode() for t in times], dtype="S19")),
            "T2": (("Time", "south_north", "west_east"), np.full((n_records, 1, 2), 280.0), {"units": "K"}),
        },
        coords={
            "XLAT": (("south_north", "west_east"), np.array([[10.0, 10.0]])),
            "XLONG": (("south_north", "west_east"), np.array([[20.0, 21.0]])),
        },
    ).to_netcdf(path)
    return path


def _plain_wrf_adapter(root):
    return WRFAdapter(config=WRFConfig(root_path=root, unstagger_grids=False))


def test_wrf_unparsable_or_missing_times_raise(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    _write_wrfout_with_times(bad / "wrfout_d01_x.nc", ["2020-13-45_99:00:00", "2020-13-46_99:00:00"])
    with pytest.raises(ValueError, match="2020-13-45_99:00:00"):
        _plain_wrf_adapter(bad).discover()
    missing = tmp_path / "missing"
    missing.mkdir()
    path = missing / "wrfout_d01_y.nc"
    import numpy as np

    xr.Dataset({"T2": (("Time", "south_north", "west_east"), np.full((1, 1, 2), 280.0), {"units": "K"})}).to_netcdf(path)
    with pytest.raises(ValueError, match="Times"):
        _plain_wrf_adapter(missing).discover()


def test_wrf_files_are_ordered_by_their_parsed_times_not_names_or_mtime(tmp_path):
    import os

    import pandas as pd

    late = _write_wrfout_with_times(tmp_path / "wrfout_d01_a.nc", ["2020-01-01_02:00:00", "2020-01-01_03:00:00"])
    early = _write_wrfout_with_times(tmp_path / "wrfout_d01_b.nc", ["2020-01-01_00:00:00", "2020-01-01_01:00:00"])
    # The later file is older on disk: a modification-time order would put it first.
    os.utime(late, (1_000_000, 1_000_000))
    os.utime(early, (2_000_000, 2_000_000))
    adapter = _plain_wrf_adapter(tmp_path)
    info = adapter.discover()
    assert [path.name for path in adapter.discovered_files] == ["wrfout_d01_b.nc", "wrfout_d01_a.nc"]
    assert info["time_coverage"] == {"start": "2020-01-01T00:00:00", "end": "2020-01-01T03:00:00"}
    adapter.open()
    combined = adapter.current_datasets["domain_1"]
    assert list(pd.DatetimeIndex(combined["time"].values).strftime("%H")) == ["00", "01", "02", "03"]
    assert combined.sizes["Time"] == 4


def test_wrf_overlapping_record_times_raise(tmp_path):
    _write_wrfout_with_times(tmp_path / "wrfout_d01_a.nc", ["2020-01-01_00:00:00", "2020-01-01_01:00:00"])
    _write_wrfout_with_times(tmp_path / "wrfout_d01_b.nc", ["2020-01-01_01:00:00", "2020-01-01_02:00:00"])
    adapter = _plain_wrf_adapter(tmp_path)
    adapter.discover()
    with pytest.raises(OSError, match="strictly increasing"):
        adapter.open()
