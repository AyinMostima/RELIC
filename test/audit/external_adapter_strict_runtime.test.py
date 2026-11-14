import pytest

from relic.coupling.external.common.base_adapter import ExternalAdapter, TimeWindow, SpatialWindow
from relic.coupling.external.common.datasource import DataSource


class _DummyAdapter(ExternalAdapter):
    def discover(self, source):
        return {}

    def open(self, source, config):
        self._connected = True

    def read(self, time_window=None, spatial_window=None, variables=None, **kwargs):
        return {}

    def transform_to_internal(self, data, mapping, policy):
        return {}

    def close(self):
        self._connected = False


def test_external_adapter_forward_is_not_executable_template() -> None:
    adapter = _DummyAdapter(source=DataSource.WRF, provided_variables=["air_temperature"], name="dummy_external_adapter")
    with pytest.raises(NotImplementedError, match="template-only"):
        adapter.forward(state={}, time=0.0, dt=3600.0)


def test_external_adapter_memory_estimate_requires_override() -> None:
    adapter = _DummyAdapter(source=DataSource.WRF, provided_variables=["air_temperature"], name="dummy_external_adapter")
    with pytest.raises(NotImplementedError, match="abstract in strict runtime"):
        adapter.estimate_memory_usage(
            TimeWindow(start="2000-01-01", end="2000-01-02"),
            SpatialWindow(lon_min=0.0, lon_max=1.0, lat_min=0.0, lat_max=1.0),
            variables=["air_temperature"],
        )


_ADAPTERS = [
    ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig"),
    ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig"),
    ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter", "SWATConfig"),
]


def _adapter(module_name, class_name, config_name, root):
    import importlib

    module = importlib.import_module(module_name)
    # SWATConfig declares its output level (only the output.hru layout is read).
    extra = {"output_level": "hru"} if config_name == "SWATConfig" else {}
    return getattr(module, class_name)(getattr(module, config_name)(root_path=root, **extra))


def _rsds_field(variable="shortwave_down"):
    import torch

    field = torch.tensor([[100.0, 200.0, 300.0, 400.0], [500.0, 600.0, 700.0, 800.0]])
    field._relic_meta = {
        "shape": field.shape,
        "dims": ("lat", "lon"),
        "coords": {"lat": torch.tensor([-0.5, 0.5]), "lon": torch.tensor([0.5, 1.5, 2.5, 3.5])},
        "coord_dims": {"lat": ("lat",), "lon": ("lon",)},
        "attrs": {"units": "W m-2"},
        "variable": variable,
        "cell_bounds": {
            "lat_coordinate": "lat",
            "lon_coordinate": "lon",
            "lat_bounds": torch.tensor([[-1.0, 0.0], [0.0, 1.0]]),
            "lon_bounds": torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        },
    }
    return field


_TARGET_GRID = {
    "lat": [0.0],
    "lon": [1.0, 3.0],
    "lat_bounds": [[-1.0, 1.0]],
    "lon_bounds": [[0.0, 2.0], [2.0, 4.0]],
}


@pytest.mark.parametrize(("module_name", "class_name", "config_name"), _ADAPTERS)
def test_every_adapter_transform_regrids_and_checks_quality(tmp_path, module_name, class_name, config_name):
    """CESM, VIC and SWAT use the shared transform: regrid on the declared basis, catalog QC."""
    import torch

    adapter = _adapter(module_name, class_name, config_name, tmp_path)
    out = adapter.transform_to_internal(
        {"shortwave_down": _rsds_field()}, target_grid=dict(_TARGET_GRID)
    )
    field = out["shortwave_down"]
    torch.testing.assert_close(
        field.detach(), torch.tensor([[350.0, 550.0]], dtype=torch.float32), rtol=1e-5, atol=1e-3
    )
    assert field._relic_meta["regrid_method"] == "conservative"


@pytest.mark.parametrize(("module_name", "class_name", "config_name"), _ADAPTERS)
def test_every_adapter_transform_raises_instead_of_passing_data_through(
    tmp_path, module_name, class_name, config_name
):
    adapter = _adapter(module_name, class_name, config_name, tmp_path)
    with pytest.raises(KeyError, match="not_in_any_catalog"):
        adapter.transform_to_internal({"not_in_any_catalog": _rsds_field("not_in_any_catalog")})


def test_conversion_namespace_is_required_per_model_without_fallback() -> None:
    """Each adapter reads coupling.external.<model>.conversion_namespace; a missing entry raises."""
    import inspect

    known = _DummyAdapter(source=DataSource.WRF, provided_variables=["air_temperature"], name="dummy_external_adapter", model_id="wrf")
    assert known._get_conversion_namespace() == "conversion_context"
    unknown = _DummyAdapter(source=DataSource.WRF, provided_variables=["air_temperature"], name="dummy_external_adapter", model_id="not_a_model")
    with pytest.raises((KeyError, ValueError), match="conversion_namespace"):
        unknown._get_conversion_namespace()
    assert "except" not in inspect.getsource(ExternalAdapter._get_conversion_namespace)


def test_conversion_namespace_must_be_declared_by_the_parameter_database() -> None:
    """The conversion namespace is checked against the namespaces the parameter database declares,
    not against directory-name conventions (a relic.* name nobody declares raises)."""
    adapter = _DummyAdapter(source=DataSource.WRF, provided_variables=["air_temperature"], name="dummy_external_adapter", model_id="wrf")
    adapter.get_parameter_value = lambda *args, **kwargs: "relic.conversion_namespace_nobody_declares"
    with pytest.raises(KeyError, match="declared"):
        adapter._get_conversion_namespace()


@pytest.mark.parametrize(
    ("module_name", "class_name", "config_name", "names"),
    [
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", ("TMAX", "TMIN")),
    ],
)
def test_hydrology_adapters_read_daily_extrema_without_a_mean_alias(
    tmp_path, module_name, class_name, config_name, names
):
    """read(variables=None) returns the daily extrema as their own variables (degC -> K)."""
    import numpy as np
    import torch
    import xarray as xr

    adapter = _adapter(module_name, class_name, config_name, tmp_path)
    maximum, minimum = names
    dataset = xr.Dataset(
        {
            maximum: (("time", "cell"), np.array([[25.0, 30.0]]), {"units": "degC"}),
            minimum: (("time", "cell"), np.array([[10.0, 12.0]]), {"units": "degC"}),
        }
    )
    adapter.current_datasets = {"main": dataset}
    tensors = adapter.read(variables=None)
    # The canonical ContractRegistry names (daily_*_air_temperature are their aliases).
    assert set(tensors) == {"air_temperature_max", "air_temperature_min"}
    torch.testing.assert_close(
        tensors["air_temperature_max"], torch.tensor([[298.15, 303.15]], dtype=torch.float32)
    )
    torch.testing.assert_close(
        tensors["air_temperature_min"], torch.tensor([[283.15, 285.15]], dtype=torch.float32)
    )


def _clm_soil_water_dataset(with_dzsoi):
    """CLM5 history layout: H2OSOI on the top levsoi layers, DZSOI on the full levgrnd column."""
    import numpy as np
    import xarray as xr

    data_vars = {
        "H2OSOI": (
            ("time", "levsoi", "lat", "lon"),
            np.array([[[[0.3, 0.25]], [[0.2, 0.4]]]]),
            {"units": "mm3/mm3"},
        ),
    }
    if with_dzsoi:
        data_vars["DZSOI"] = (
            ("levgrnd", "lat", "lon"),
            np.array([[[0.1, 0.1]], [[0.3, 0.3]], [[2.0, 2.0]]]),
            {"units": "m"},
        )
    return xr.Dataset(data_vars)


def test_cesm_volumetric_soil_water_uses_the_history_layer_thickness(tmp_path):
    """m_k = rho_w theta_k dz_k with dz from DZSOI (top 2 of 3 levgrnd layers): 30, 25, 60, 120 kg m-2."""
    import torch

    adapter = _adapter(
        "relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", tmp_path
    )
    adapter.current_datasets = {"h0": _clm_soil_water_dataset(with_dzsoi=True)}
    layers = adapter.read(variables=["mass_content_of_water_in_soil_layer"])["mass_content_of_water_in_soil_layer"]
    torch.testing.assert_close(
        layers, torch.tensor([[[[30.0, 25.0]], [[60.0, 120.0]]]], dtype=torch.float32), rtol=1e-6, atol=1e-4
    )
    assert layers._relic_meta["attrs"]["units"] == "kg m-2"
    assert layers._relic_meta["attrs"]["layer_thickness_source"] == "DZSOI"


def test_cesm_soil_water_without_dzsoi_needs_the_configured_layer_thickness(tmp_path):
    """No DZSOI in the file: CESMConfig.soil_layer_thickness is required, one value per layer."""
    import importlib

    import torch

    module = importlib.import_module("relic.coupling.external.atmospheric_models.cesm_adapter")
    bare = module.CESMAdapter(module.CESMConfig(root_path=tmp_path))
    bare.current_datasets = {"h0": _clm_soil_water_dataset(with_dzsoi=False)}
    with pytest.raises(ValueError, match="DZSOI.*soil_layer_thickness"):
        bare.read(variables=["mass_content_of_water_in_soil_layer"])

    short = module.CESMAdapter(module.CESMConfig(root_path=tmp_path, soil_layer_thickness=[0.1]))
    short.current_datasets = {"h0": _clm_soil_water_dataset(with_dzsoi=False)}
    with pytest.raises(ValueError, match="2 layers"):
        short.read(variables=["mass_content_of_water_in_soil_layer"])

    configured = module.CESMAdapter(module.CESMConfig(root_path=tmp_path, soil_layer_thickness=[0.1, 0.3]))
    configured.current_datasets = {"h0": _clm_soil_water_dataset(with_dzsoi=False)}
    layers = configured.read(variables=["mass_content_of_water_in_soil_layer"])["mass_content_of_water_in_soil_layer"]
    torch.testing.assert_close(
        layers, torch.tensor([[[[30.0, 25.0]], [[60.0, 120.0]]]], dtype=torch.float32), rtol=1e-6, atol=1e-4
    )
    assert layers._relic_meta["attrs"]["layer_thickness_source"] == "configuration soil_layer_thickness"


def test_vic_layer_soil_moisture_depth_becomes_layer_water_mass(tmp_path):
    """OUT_SOIL_MOIST is water depth per layer: m_k = rho_w d_k, 25 mm -> 25 kg m-2; no thickness needed."""
    import numpy as np
    import torch
    import xarray as xr

    adapter = _adapter(
        "relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", tmp_path
    )
    adapter.current_datasets = {
        "main": xr.Dataset(
            {
                "OUT_SOIL_MOIST": (
                    ("time", "nlayer", "cell"),
                    np.array([[[25.0, 30.0], [100.0, 80.0], [400.0, 350.0]]]),
                    {"units": "mm"},
                )
            }
        )
    }
    layers = adapter.read(variables=None)["mass_content_of_water_in_soil_layer"]
    torch.testing.assert_close(
        layers,
        torch.tensor([[[25.0, 30.0], [100.0, 80.0], [400.0, 350.0]]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-4,
    )
    assert layers._relic_meta["attrs"]["units"] == "kg m-2"

    adapter.current_datasets = {
        "main": xr.Dataset({"OUT_SOIL_MOIST": (("time", "cell"), np.array([[25.0, 30.0]]), {"units": "mm"})})
    }
    with pytest.raises(ValueError, match="nlayer"):
        adapter.read(variables=["mass_content_of_water_in_soil_layer"])


def test_adapter_regridding_identifies_coordinates_by_cf_units(tmp_path):
    """Read metadata names the CF lat/lon coordinates (1-D or 2-D); without them regridding raises."""
    import numpy as np
    import xarray as xr

    adapter = _adapter(
        "relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", tmp_path
    )
    dataset = xr.Dataset(
        {"TREFHT": (("y", "x"), np.array([[280.0, 281.0], [282.0, 283.0]]), {"units": "K"})},
        coords={
            "nav_lat": (("y", "x"), np.array([[10.0, 10.0], [11.0, 11.0]]), {"units": "degrees_north"}),
            "nav_lon": (("y", "x"), np.array([[20.0, 21.0], [20.0, 21.0]]), {"units": "degrees_east"}),
        },
    )
    record = adapter._cf_horizontal_coordinates(dataset, dataset["TREFHT"])
    assert record == {
        "lat_coordinate": "nav_lat",
        "lon_coordinate": "nav_lon",
        "lat_bounds": None,
        "lon_bounds": None,
    }

    field = _rsds_field("air_temperature")
    field._relic_meta["attrs"] = {"units": "K"}
    field._relic_meta["cell_bounds"] = None
    with pytest.raises(ValueError, match="CF latitude"):
        adapter._apply_regridding(field, dict(_TARGET_GRID))


def test_adapters_do_not_change_the_gradient_flag_of_received_tensors(tmp_path):
    """The caller's requires_grad decides gradient tracking; adapters never toggle it."""
    import torch

    adapter = _adapter(
        "relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", tmp_path
    )
    frozen = torch.tensor([[280.0, 281.0]], dtype=torch.float64)
    converted = adapter._ensure_tensor_properties(frozen)
    assert converted.dtype == torch.float32 and converted is not frozen
    assert frozen.requires_grad is False and converted.requires_grad is False
    same = torch.tensor([280.0], dtype=torch.float32)
    assert adapter._ensure_tensor_properties(same) is same
    assert same.requires_grad is False

    out = adapter.transform_to_internal({"air_temperature": frozen})["air_temperature"]
    assert frozen.requires_grad is False and out.requires_grad is False
    tracked = torch.tensor([[280.0, 281.0]], dtype=torch.float64, requires_grad=True)
    out = adapter.transform_to_internal({"air_temperature": tracked})["air_temperature"]
    out.sum().backward()
    torch.testing.assert_close(tracked.grad, torch.ones_like(tracked))


def _vic_water_dataset(evap_name, evap_units, with_time_bounds):
    """Two daily records: water depths per record (or per day) and snow water equivalent."""
    import numpy as np
    import xarray as xr

    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    data_vars = {
        evap_name: (("time", "cell"), np.array([[2.16, 4.32], [0.864, 0.0]]), {"units": evap_units}),
        "OUT_SWE": (("time", "cell"), np.array([[150.0, 0.0], [140.0, 10.0]]), {"units": "mm"}),
    }
    time_attrs = {}
    if with_time_bounds:
        data_vars["time_bnds"] = (("time", "nv"), np.stack([time, time + np.timedelta64(1, "D")], axis=1))
        time_attrs = {"bounds": "time_bnds"}
    return xr.Dataset(data_vars, coords={"time": ("time", time, time_attrs)})


def test_water_depths_become_mass_fluxes_over_the_declared_record_interval(tmp_path):
    """F = rho_w d / dt: 2.16 mm over a 1-day record -> 2.5e-5 kg m-2 s-1; SWE 150 mm -> 150 kg m-2."""
    import torch

    expected_flux = torch.tensor([[2.5e-5, 5.0e-5], [1.0e-5, 0.0]], dtype=torch.float32)
    vic = _adapter("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", tmp_path)
    vic.current_datasets = {"main": _vic_water_dataset("OUT_EVAP", "mm", with_time_bounds=True)}
    tensors = vic.read(variables=["evapotranspiration_flux", "surface_snow_amount"])
    torch.testing.assert_close(tensors["evapotranspiration_flux"], expected_flux, rtol=1e-5, atol=1e-12)
    torch.testing.assert_close(
        tensors["surface_snow_amount"], torch.tensor([[150.0, 0.0], [140.0, 10.0]], dtype=torch.float32)
    )
    assert tensors["evapotranspiration_flux"]._relic_meta["attrs"]["units"] == "kg m-2 s-1"

    # A declared rate needs no record interval.
    vic.current_datasets = {"main": _vic_water_dataset("OUT_EVAP", "mm day-1", with_time_bounds=False)}
    torch.testing.assert_close(
        vic.read(variables=["evapotranspiration_flux"])["evapotranspiration_flux"],
        expected_flux,
        rtol=1e-5,
        atol=1e-12,
    )
    # An amount per record without CF time bounds has no interval: it raises.
    vic.current_datasets = {"main": _vic_water_dataset("OUT_EVAP", "mm", with_time_bounds=False)}
    with pytest.raises(ValueError, match="time bounds"):
        vic.read(variables=["evapotranspiration_flux"])

    swat = _adapter("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter", "SWATConfig", tmp_path)
    swat.current_datasets = {"hru": _vic_water_dataset("ET", "mm", with_time_bounds=True)}
    torch.testing.assert_close(
        swat.read(variables=["evapotranspiration_flux"])["evapotranspiration_flux"],
        expected_flux,
        rtol=1e-5,
        atol=1e-12,
    )


def test_adapter_tables_supply_units_only_when_the_file_declares_none(tmp_path):
    """File units win; an adapter table fills a missing units attribute and says so."""
    import numpy as np
    import xarray as xr

    vic = _adapter("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", tmp_path)
    dataset = xr.Dataset(
        {
            "OUT_SOIL_TEMP": (("time", "cell"), np.array([[280.0, 281.0]]), {"units": "K"}),
            "OUT_WDEW": (("time", "cell"), np.array([[0.5, 0.2]])),
        }
    )
    dataset = vic._add_vic_metadata(dataset)
    assert dataset["OUT_SOIL_TEMP"].attrs["units"] == "K"
    assert dataset["OUT_SOIL_TEMP"].attrs["units_source"] == "file"
    assert dataset["OUT_WDEW"].attrs["units"] == "mm"
    assert dataset["OUT_WDEW"].attrs["units_source"] == "adapter_table"

    swat = _adapter("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter", "SWATConfig", tmp_path)
    dataset = xr.Dataset(
        {
            "ET": (("time", "hru"), np.array([[1.5, 1.6]]), {"units": "cm"}),
            "PRECIP": (("time", "hru"), np.array([[3.0, 4.0]])),
        }
    )
    dataset = swat._add_swat_metadata(dataset)
    assert dataset["ET"].attrs["units"] == "cm"
    assert dataset["ET"].attrs["units_source"] == "file"
    assert dataset["PRECIP"].attrs["units"] == "mm"
    assert dataset["PRECIP"].attrs["units_source"] == "adapter_table"


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter"),
        ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter"),
        ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter"),
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter"),
        ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter"),
    ],
)
def test_every_adapter_forward_goes_through_the_single_read_path(module_name, class_name):
    import importlib
    import inspect

    cls = getattr(importlib.import_module(module_name), class_name)
    source = inspect.getsource(cls.forward)
    assert "_read_records_at(" in source, class_name
    parameters = inspect.signature(cls.forward).parameters
    # The simulator passes both controls by name (_build_integration_kwargs).
    assert {"calendar_type", "time_origin"} <= set(parameters), class_name
    assert ".isel(" not in source and "torch.tensor(" not in source and "from_numpy" not in source, class_name
    assert not hasattr(cls, "SWAT_VARIABLE_MAPPING")


def _noleap_days(n_days, hour=0):
    """Record times of n daily records from 2000-01-01 in the noleap calendar (cftime objects)."""
    import cftime

    return [cftime.DatetimeNoLeap(2000, 1, 1 + day, hour) for day in range(n_days)]


_NOLEAP = {"calendar_type": "noleap", "time_origin": "2000-01-01T00:00:00"}


def _time_config(module_name, class_name, config_name, root, rule, provided=None):
    import importlib

    module = importlib.import_module(module_name)
    extra = {"output_level": "hru"} if config_name == "SWATConfig" else {}
    return getattr(module, class_name)(
        getattr(module, config_name)(
            root_path=root, time_interpolation=rule, provided_variables=provided, **extra
        )
    )


def _vic_by_time(root, rule="lower"):
    import numpy as np
    import xarray as xr

    vic = _time_config(
        "relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", root, rule,
        provided=["surface_snow_amount", "soil_temperature"],
    )
    vic.current_datasets = {
        "main": xr.Dataset(
            {
                "OUT_SWE": (("time", "cell"), np.array([[150.0, 0.0], [140.0, 10.0], [130.0, 20.0]]), {"units": "mm"}),
                "OUT_SOIL_TEMP": (("time", "cell"), np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]), {"units": "degC"}),
            },
            coords={"time": _noleap_days(3)},
        )
    }
    return vic


def test_forward_returns_the_converted_catalog_fields_at_the_simulation_time(tmp_path):
    """VIC: SWE 150 mm -> 150 kg m-2 and soil temperature 7 degC -> 280.15 K at t = 0, keyed by catalog names."""
    import numpy as np
    import torch
    import xarray as xr

    from relic.core.state import UpdateItem

    vic = _vic_by_time(tmp_path)
    updates = vic.forward({}, time=0.0, dt=86400.0, **_NOLEAP)
    assert set(updates) == {"surface_snow_amount", "soil_temperature"}
    assert all(isinstance(item, UpdateItem) for item in updates.values())
    torch.testing.assert_close(updates["surface_snow_amount"].value, torch.tensor([150.0, 0.0], dtype=torch.float32))
    torch.testing.assert_close(updates["soil_temperature"].value, torch.tensor([280.15, 281.15], dtype=torch.float32))

    cesm = _time_config(
        "relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", tmp_path, "lower",
        provided=["air_temperature"],
    )
    cesm.current_datasets = {
        "h0": xr.Dataset(
            {
                "TREFHT": (("time", "cell"), np.array([[280.0, 281.0], [282.0, 283.0]]), {"units": "K"}),
                "NOT_MAPPED": (("time", "cell"), np.array([[1.0, 2.0], [3.0, 4.0]]), {"units": "1"}),
            },
            coords={"time": _noleap_days(2)},
        )
    }
    updates = cesm.forward({}, time=86400.0, dt=3600.0, **_NOLEAP)
    assert set(updates) == {"air_temperature"}
    torch.testing.assert_close(updates["air_temperature"].value, torch.tensor([282.0, 283.0], dtype=torch.float32))


def test_unit_conversion_reads_no_soil_depth_context() -> None:
    """No catalog conversion uses a soil depth: the adapter conversion reads none (any model id)."""
    import importlib
    import inspect

    import xarray as xr

    source = inspect.getsource(ExternalAdapter._apply_unit_conversion)
    assert "soil_depth" not in source and "source_model" not in source
    for module_name, class_name in (
        ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter"),
        ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter"),
        ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter"),
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter"),
        ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter"),
    ):
        cls = getattr(importlib.import_module(module_name), class_name)
        assert "_apply_unit_conversion" not in vars(cls), class_name
    adapter = _DummyAdapter(source=DataSource.WRF, provided_variables=["air_temperature"], name="dummy_external_adapter", model_id="not_a_model")
    converted = adapter._apply_unit_conversion(
        xr.DataArray([7.0], dims=("cell",), attrs={"units": "degC"}), "air_temperature"
    )
    assert float(converted.values[0]) == pytest.approx(280.15)
    assert converted.attrs["units"] == "K"


def test_a_declared_record_interval_converts_water_depths_only_without_file_bounds(tmp_path):
    """VIC OUT_EVAP 2.16 mm per record with record_interval = 1 day -> 2.5e-5 kg m-2 s-1."""
    import importlib

    import torch

    module = importlib.import_module("relic.coupling.external.hydrological_models.vic_adapter")
    expected = torch.tensor([[2.5e-5, 5.0e-5], [1.0e-5, 0.0]], dtype=torch.float32)
    declared = module.VICAdapter(module.VICConfig(root_path=tmp_path, record_interval=1.0, record_interval_units="day"))
    declared.current_datasets = {"main": _vic_water_dataset("OUT_EVAP", "mm", with_time_bounds=False)}
    flux = declared.read(variables=["evapotranspiration_flux"])["evapotranspiration_flux"]
    torch.testing.assert_close(flux, expected, rtol=1e-5, atol=1e-12)
    assert flux._relic_meta["attrs"]["flux_interval_source"] == "configuration record_interval 1.0 day"
    # File time bounds win over the configuration (here: 1-day bounds against a declared 2 days).
    both = module.VICAdapter(module.VICConfig(root_path=tmp_path, record_interval=2.0, record_interval_units="day"))
    both.current_datasets = {"main": _vic_water_dataset("OUT_EVAP", "mm", with_time_bounds=True)}
    torch.testing.assert_close(
        both.read(variables=["evapotranspiration_flux"])["evapotranspiration_flux"],
        expected,
        rtol=1e-5,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="record_interval_units"):
        module.VICConfig(root_path=tmp_path, record_interval=1.0)
    with pytest.raises(ValueError, match="positive"):
        module.VICConfig(root_path=tmp_path, record_interval=0.0, record_interval_units="day")


def test_cesm_near_surface_fields_with_relic_consumers_are_mapped(tmp_path):
    """PS -> atmospheric_pressure (surface_air_pressure) and U10 -> wind_speed; QREFHT has no consumer."""
    import numpy as np
    import torch
    import xarray as xr

    from relic.coupling.external.common.datasource import DataSource
    from relic.coupling.external.common.schema import InternalSchema

    cesm_mapping = InternalSchema().get_source_mapping(DataSource.CESM)
    assert cesm_mapping["PS"] == "atmospheric_pressure"
    assert cesm_mapping["U10"] == "wind_speed"
    assert "QREFHT" not in cesm_mapping
    adapter = _adapter(
        "relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", tmp_path
    )
    adapter.current_datasets = {
        "h0": xr.Dataset(
            {
                "PS": (("time", "cell"), np.array([[101325.0, 98000.0]]), {"units": "Pa"}),
                "U10": (("time", "cell"), np.array([[3.5, 0.5]]), {"units": "m/s"}),
            }
        )
    }
    tensors = adapter.read(variables=["atmospheric_pressure", "wind_speed"])
    torch.testing.assert_close(
        tensors["atmospheric_pressure"], torch.tensor([[101325.0, 98000.0]], dtype=torch.float32)
    )
    torch.testing.assert_close(tensors["wind_speed"], torch.tensor([[3.5, 0.5]], dtype=torch.float32))


_READ_CASES = [
    ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter", "WRFConfig", "d01", "T2",
     "air_temperature", "surface_temperature"),
    ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", "atm", "TS",
     "surface_temperature", "soil_temperature"),
    ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter", "CMIP6Config", "ts", "ts",
     "surface_temperature", "atmospheric_pressure"),
    ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", "main", "OUT_SOIL_TEMP",
     "soil_temperature", "wind_speed"),
    ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter", "SWATConfig", "hru", "SOL_TMP",
     "soil_temperature", "surface_runoff_flux"),
]


@pytest.mark.parametrize(
    ("module_name", "class_name", "config_name", "key", "field", "present", "absent"),
    _READ_CASES,
    ids=["wrf", "cesm", "cmip6", "vic", "swat"],
)
def test_absent_mapped_fields_raise_when_requested_and_are_reported_otherwise(
    tmp_path, module_name, class_name, config_name, key, field, present, absent
):
    """An explicitly requested variable missing from every opened dataset raises; with
    variables=None the mapped fields a dataset lacks are listed in last_read_report."""
    import numpy as np
    import xarray as xr

    adapter = _adapter(module_name, class_name, config_name, tmp_path)
    adapter.current_datasets = {
        key: xr.Dataset({field: (("time", "cell"), np.array([[280.0, 281.0]]), {"units": "K"})})
    }
    tensors = adapter.read(variables=None)
    assert present in tensors
    assert absent in adapter.last_read_report["absent_mapped_fields"][key]
    assert present not in adapter.last_read_report["absent_mapped_fields"][key]
    with pytest.raises(ValueError, match=absent):
        adapter.read(variables=[absent])
    assert set(adapter.read(variables=[present])) == {present}
    assert adapter.last_read_report["requested"] == [present]


def test_vic_ascii_and_binary_outputs_are_rejected_not_skipped(tmp_path):
    """VIC ASCII/binary output columns carry no variable names (their order is declared in the
    VIC global parameter file), so only the VIC NetCDF output is read; no file is skipped."""
    from relic.coupling.external.hydrological_models import vic_adapter

    for file_format in ("ascii", "binary"):
        with pytest.raises(ValueError, match="NetCDF"):
            vic_adapter.VICConfig(root_path=tmp_path, file_format=file_format)
    assert not hasattr(vic_adapter.VICAdapter, "_open_ascii_files")


def test_forcing_is_read_at_the_simulation_time_without_a_cursor(tmp_path):
    """Records follow t (origin 2000-01-01 00:00, noleap): t0 and t0 + 1 day read different records.

    The read is a pure function of t: no position is kept between calls, a later call at an
    earlier time returns the earlier record, and the restart binding stays (source, model_id).
    """
    import torch

    def f32(values):
        return torch.tensor(values, dtype=torch.float32)

    vic = _vic_by_time(tmp_path, rule="lower")
    day = 86400.0

    def swe(time):
        return vic.forward({}, time=time, dt=3600.0, **_NOLEAP)["surface_snow_amount"].value

    torch.testing.assert_close(swe(0.0), f32([150.0, 0.0]))
    torch.testing.assert_close(swe(day), f32([140.0, 10.0]))
    # lower: t_k <= t < t_(k+1) holds record k until the next record time.
    torch.testing.assert_close(swe(0.5 * day), f32([150.0, 0.0]))
    torch.testing.assert_close(swe(2.0 * day), f32([130.0, 20.0]))
    # No cursor: an earlier time after a later one returns the earlier record again.
    torch.testing.assert_close(swe(0.0), f32([150.0, 0.0]))
    assert set(vic.get_extra_state()) == {"format", "version", "source", "model_id"}

    linear = _vic_by_time(tmp_path, rule="linear")
    half = linear.forward({}, time=0.5 * day, dt=3600.0, **_NOLEAP)
    torch.testing.assert_close(half["surface_snow_amount"].value, f32([145.0, 5.0]))
    torch.testing.assert_close(half["soil_temperature"].value, f32([281.15, 282.15]))
    quarter = linear.forward({}, time=1.25 * day, dt=3600.0, **_NOLEAP)
    torch.testing.assert_close(quarter["surface_snow_amount"].value, f32([137.5, 12.5]))
    # An origin later than the first record shifts the records: t = 0 is 2000-01-02.
    shifted = vic.forward({}, time=0.0, dt=3600.0, calendar_type="noleap", time_origin="2000-01-02T00:00:00")
    torch.testing.assert_close(shifted["surface_snow_amount"].value, f32([140.0, 10.0]))


def test_time_stepped_reads_need_an_origin_a_matching_calendar_and_coverage(tmp_path):
    import numpy as np
    import xarray as xr

    vic = _vic_by_time(tmp_path)
    day = 86400.0
    with pytest.raises(ValueError, match="time_origin"):
        vic.forward({}, time=0.0, dt=3600.0, calendar_type="noleap", time_origin=None)
    with pytest.raises(ValueError, match="time_origin"):
        vic.forward({}, time=0.0, dt=3600.0, calendar_type="noleap")
    with pytest.raises(ValueError, match="calendar_type"):
        vic.forward({}, time=0.0, dt=3600.0, time_origin="2000-01-01T00:00:00")
    # An origin that is not a date of the calendar raises (no leap day in noleap).
    with pytest.raises(ValueError, match="time_origin"):
        vic.forward({}, time=0.0, dt=3600.0, calendar_type="noleap", time_origin="2001-02-29T00:00:00")
    # The dataset calendar must be the simulator calendar; nothing is converted.
    with pytest.raises(ValueError, match="calendar"):
        vic.forward({}, time=0.0, dt=3600.0, calendar_type="360_day", time_origin="2000-01-01T00:00:00")
    gregorian = _vic_by_time(tmp_path)
    gregorian.current_datasets["main"] = gregorian.current_datasets["main"].assign_coords(
        time=np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]")
    )
    with pytest.raises(ValueError, match="calendar"):
        gregorian.forward({}, time=0.0, dt=3600.0, **_NOLEAP)
    # Outside the records [2000-01-01, 2000-01-03] the read raises; it never extrapolates.
    for time in (-1.0, 2.0 * day + 1.0):
        with pytest.raises(ValueError, match="outside"):
            vic.forward({}, time=time, dt=3600.0, **_NOLEAP)
    # A time-dependent field without a decoded time coordinate cannot be placed in time.
    untimed = _vic_by_time(tmp_path)
    untimed.current_datasets["main"] = untimed.current_datasets["main"].drop_vars("time")
    with pytest.raises(ValueError, match="time coordinate"):
        untimed.forward({}, time=0.0, dt=3600.0, **_NOLEAP)
    # The configuration must declare the rule; only lower and linear exist.
    undeclared = _adapter(
        "relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", tmp_path
    )
    undeclared.current_datasets = _vic_by_time(tmp_path).current_datasets
    with pytest.raises(ValueError, match="time_interpolation"):
        undeclared.forward({}, time=0.0, dt=3600.0, **_NOLEAP)
    import importlib

    module = importlib.import_module("relic.coupling.external.hydrological_models.vic_adapter")
    with pytest.raises(ValueError, match="time_interpolation"):
        module.VICConfig(root_path=tmp_path, time_interpolation="nearest")
    # A field without a time dimension (a static field) is returned whole at any time.
    static = _vic_by_time(tmp_path)
    static.current_datasets["main"] = xr.Dataset(
        {"OUT_SWE": (("cell",), np.array([5.0, 6.0]), {"units": "mm"})}
    )
    values = static.forward(
        {}, time=10.0 * day, dt=3600.0, variables=["surface_snow_amount"], **_NOLEAP
    )["surface_snow_amount"].value
    assert values.tolist() == [5.0, 6.0]


def test_record_weights_follow_the_datm_lower_and_linear_rules():
    """forcing_record_weights: CDEPS/E3SM shr_tInterp_getFactors 'lower' and 'linear'."""
    from relic.coupling.external.common.base_adapter import forcing_record_weights

    seconds = [0.0, 100.0, 300.0]
    assert forcing_record_weights(seconds, 0.0, "lower") == [(0, 1.0)]
    assert forcing_record_weights(seconds, 99.0, "lower") == [(0, 1.0)]
    assert forcing_record_weights(seconds, 100.0, "lower") == [(1, 1.0)]
    assert forcing_record_weights(seconds, 300.0, "lower") == [(2, 1.0)]
    assert forcing_record_weights(seconds, 150.0, "linear") == [(1, 0.75), (2, 0.25)]
    assert forcing_record_weights(seconds, 100.0, "linear") == [(1, 1.0)]
    for bad in (-0.5, 300.5, float("nan")):
        with pytest.raises(ValueError):
            forcing_record_weights(seconds, bad, "lower")
    with pytest.raises(ValueError, match="increasing"):
        forcing_record_weights([0.0, 0.0], 0.0, "lower")
    with pytest.raises(ValueError, match="time_interpolation"):
        forcing_record_weights(seconds, 0.0, "nearest")


def test_the_restart_contract_text_describes_the_read_at_the_simulation_time():
    import inspect

    from relic.coupling.external.common import base_adapter

    text = inspect.getsource(base_adapter)
    for stale in (
        "[t, t+dt]",
        "consumed forcing position",
        "current time window",
        "Temporal resampling/aggregation",
        "based on time/space windows",
        "temporal aggregation)",
        "is not implemented, so a restored run returns record 0 again",
        "y_n = X_n[record 0]",
    ):
        assert stale not in text, stale
    assert "time_origin" in text and "forcing_record_weights" in text


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter"),
        ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter"),
        ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter"),
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter"),
        ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter"),
    ],
)
def test_transform_policy_is_removed_with_its_unapplied_constructor_argument(module_name, class_name):
    """TransformPolicy was stored and never applied; its comments claimed gap filling and interpolation."""
    import importlib
    import inspect

    import relic.coupling.external as external
    import relic.coupling.external.common as common
    from relic.coupling.external.common import base_adapter

    assert not hasattr(base_adapter, "TransformPolicy")
    assert "TransformPolicy" not in common.__all__
    assert "TransformPolicy" not in external.__all__
    assert "policy" not in inspect.signature(base_adapter.ExternalAdapter.transform_to_internal).parameters
    cls = getattr(importlib.import_module(module_name), class_name)
    assert "transform_policy" not in inspect.signature(cls.__init__).parameters


def test_base_adapter_string_annotations_name_module_level_symbols():
    """A string annotation names a module-level (or TYPE_CHECKING) import, never an unbound name (F821)."""
    import ast
    import builtins
    from pathlib import Path

    from relic.coupling.external.common import base_adapter

    tree = ast.parse(Path(base_adapter.__file__).read_text(encoding="utf-8-sig"))
    module_names = set(dir(builtins))

    def collect(statements):
        for node in statements:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                module_names.add(node.name)
            elif isinstance(node, ast.Assign):
                module_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                module_names.add(node.target.id)
            elif isinstance(node, ast.If):
                collect(node.body)

    collect(tree.body)
    unresolved = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for annotation in [node.returns, *(argument.annotation for argument in arguments)]:
            if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                for name in ast.walk(ast.parse(annotation.value, mode="eval")):
                    if isinstance(name, ast.Name) and name.id not in module_names:
                        unresolved.append(f"{node.name}: {name.id}")
    assert unresolved == []


def test_adapter_fields_are_written_by_the_state_manager_without_alias_rejection(tmp_path):
    """CESM PS, TREFHT and U10 at t = 0 go through StateManager.apply_updates unchanged in name."""
    import numpy as np
    import torch
    import xarray as xr

    from relic.core.state import StateManager

    cesm = _time_config(
        "relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", tmp_path, "lower",
        provided=["atmospheric_pressure", "air_temperature", "wind_speed"],
    )
    cesm.current_datasets = {
        "h0": xr.Dataset(
            {
                "PS": (("time", "cell"), np.array([[101325.0, 98000.0]]), {"units": "Pa"}),
                "TREFHT": (("time", "cell"), np.array([[280.0, 281.0]]), {"units": "K"}),
                "U10": (("time", "cell"), np.array([[3.5, 0.5]]), {"units": "m/s"}),
            },
            coords={"time": _noleap_days(1)},
        )
    }
    updates = cesm.forward({}, time=0.0, dt=3600.0, **_NOLEAP)
    assert set(updates) == {"atmospheric_pressure", "air_temperature", "wind_speed"}
    manager = StateManager()
    initial = {"atmospheric_pressure": 100000.0, "air_temperature": 290.0, "wind_speed": 1.0}
    manager.initialize_state({name: torch.full((2,), value) for name, value in initial.items()})
    manager.apply_updates(updates)
    state = manager.get_state()
    torch.testing.assert_close(
        state["atmospheric_pressure"].cpu(), torch.tensor([101325.0, 98000.0]), check_dtype=False
    )
    torch.testing.assert_close(state["wind_speed"].cpu(), torch.tensor([3.5, 0.5]), check_dtype=False)


def test_adapter_roots_come_from_the_configuration_or_the_declared_parameter(monkeypatch):
    """No adapter falls back to the current directory; a failed parameter lookup propagates."""
    import importlib
    from pathlib import Path as _Path

    vic_module = importlib.import_module("relic.coupling.external.hydrological_models.vic_adapter")
    wrf_module = importlib.import_module("relic.coupling.external.atmospheric_models.wrf_adapter")
    cesm_module = importlib.import_module("relic.coupling.external.atmospheric_models.cesm_adapter")
    cmip6_module = importlib.import_module("relic.coupling.external.climate_models.cmip6_adapter")

    # Without a configuration, VIC and WRF use their declared parameters coupling.external.<model>.
    # default_root_path (./data/vic, ./data/wrf).
    assert _Path(vic_module.VICAdapter().config.root_path).as_posix().endswith("data/vic")
    assert _Path(wrf_module.WRFAdapter().config.root_path).as_posix().endswith("data/wrf")

    def failing_lookup(self, name, **kwargs):
        raise KeyError(f"lookup of {name} failed")

    for module, cls in ((vic_module, vic_module.VICAdapter), (wrf_module, wrf_module.WRFAdapter)):
        with monkeypatch.context() as patched:
            patched.setattr(cls, "get_parameter_value", failing_lookup)
            with pytest.raises(KeyError, match="default_root_path"):
                cls()
    # CESM and CMIP6 declare no default root: a configuration is required.
    with pytest.raises(ValueError, match="CESMConfig"):
        cesm_module.CESMAdapter()
    with pytest.raises(ValueError, match="CMIP6Config"):
        cmip6_module.CMIP6Adapter()


def test_wrf_time_format_is_the_configured_or_the_declared_parameter(tmp_path, monkeypatch):
    """An explicit WRFConfig.time_format is kept; without one the parameter is used; a failed
    lookup propagates instead of silently keeping a code default."""
    import importlib

    wrf_module = importlib.import_module("relic.coupling.external.atmospheric_models.wrf_adapter")
    assert wrf_module.WRFConfig(root_path=tmp_path).time_format is None
    declared = wrf_module.WRFAdapter(wrf_module.WRFConfig(root_path=tmp_path))
    assert declared.config.time_format == "%Y-%m-%d_%H:%M:%S"
    explicit = wrf_module.WRFAdapter(wrf_module.WRFConfig(root_path=tmp_path, time_format="%Y/%m/%d_%H%M"))
    assert explicit.config.time_format == "%Y/%m/%d_%H%M"

    original = wrf_module.WRFAdapter.get_parameter_value

    def failing_time_format(self, name, **kwargs):
        if name == "time_format":
            raise KeyError("time_format lookup failed")
        return original(self, name, **kwargs)

    monkeypatch.setattr(wrf_module.WRFAdapter, "get_parameter_value", failing_time_format)
    with pytest.raises(KeyError, match="time_format"):
        wrf_module.WRFAdapter(wrf_module.WRFConfig(root_path=tmp_path))


def _write_daily(path, days, value, calendar="noleap", with_time=True):
    import numpy as np
    import xarray as xr

    coords = {}
    if with_time:
        coords["time"] = ("time", np.asarray(days, dtype=float), {"units": "days since 2000-01-01", "calendar": calendar})
    xr.Dataset(
        {
            "OUT_SWE": (("time", "cell"), np.full((len(days), 2), value), {"units": "mm"}),
            "TREFHT": (("time", "cell"), np.full((len(days), 2), 270.0 + value), {"units": "K"}),
        },
        coords=coords,
    ).to_netcdf(path, engine="netcdf4")


@pytest.mark.parametrize(
    ("module_name", "class_name", "config_name", "names", "key"),
    [
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig",
         ("b_vic.nc", "a_vic.nc"), "main"),
        ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig",
         ("run.cam.h0.b.nc", "run.cam.h0.a.nc"), "h0"),
    ],
    ids=["vic", "cesm"],
)
def test_files_are_concatenated_by_their_decoded_record_times(tmp_path, module_name, class_name, config_name, names, key):
    """File names and modification times never order the records; overlapping or undated files raise."""
    import importlib

    import numpy as np

    module = importlib.import_module(module_name)
    first, second = (tmp_path / name for name in names)
    # The file whose name sorts last holds the earlier records.
    _write_daily(first, [0.0, 1.0], 1.0)
    _write_daily(second, [2.0, 3.0], 2.0)
    adapter = getattr(module, class_name)(getattr(module, config_name)(root_path=tmp_path))
    adapter.discover()
    adapter.open()
    days = [
        (value - adapter.current_datasets[key]["time"].values[0]).days
        for value in adapter.current_datasets[key]["time"].values
    ]
    assert days == [0, 1, 2, 3]
    np.testing.assert_array_equal(adapter.current_datasets[key]["OUT_SWE"].values[:, 0], [1.0, 1.0, 2.0, 2.0])
    adapter.close()
    for cls_attr in ("_sort_vic_files", "_sort_cesm_files"):
        assert not hasattr(getattr(module, class_name), cls_attr)

    # Overlapping records (day 1 in both files) raise instead of being concatenated.
    _write_daily(second, [1.0, 2.0], 2.0)
    overlapping = getattr(module, class_name)(getattr(module, config_name)(root_path=tmp_path))
    overlapping.discover()
    with pytest.raises((ValueError, IOError), match="increasing"):
        overlapping.open()
    # A file without a decoded time coordinate cannot be placed in time.
    _write_daily(second, [2.0, 3.0], 2.0, with_time=False)
    undated = getattr(module, class_name)(getattr(module, config_name)(root_path=tmp_path))
    undated.discover()
    with pytest.raises((ValueError, IOError), match="time coordinate"):
        undated.open()


# --- Round 27 (167): broad handlers of the external adapters expose their errors ---------------------

class _FailingDataset:
    def __init__(self, message):
        self.message = message

    def close(self):
        raise OSError(self.message)


@pytest.mark.parametrize(
    ("module_name", "class_name", "config_name", "extra_state"),
    [
        ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", "coordinate_info"),
        ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter", "WRFConfig", "domain_info"),
        ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter", "CMIP6Config", None),
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", "grid_info"),
        ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter", "SWATConfig", "watershed_info"),
    ],
    ids=["cesm", "wrf", "cmip6", "vic", "swat"],
)
def test_close_releases_every_dataset_and_raises_the_collected_errors(
    tmp_path, module_name, class_name, config_name, extra_state
):
    adapter = _adapter(module_name, class_name, config_name, tmp_path)
    adapter.current_datasets = {"first": _FailingDataset("disk gone"), "second": _FailingDataset("handle lost")}
    with pytest.raises(RuntimeError, match="disk gone") as caught:
        adapter.close()
    assert "handle lost" in str(caught.value)
    # Every resource was attempted and the adapter state is cleared either way.
    assert adapter.current_datasets == {}


def test_discovery_analysis_errors_propagate(tmp_path):
    import importlib

    cesm_module = importlib.import_module("relic.coupling.external.atmospheric_models.cesm_adapter")
    vic_module = importlib.import_module("relic.coupling.external.hydrological_models.vic_adapter")
    (tmp_path / "cesm").mkdir()
    (tmp_path / "cesm" / "run.cam.h0.0001.nc").write_bytes(b"not a netcdf file")
    cesm = cesm_module.CESMAdapter(cesm_module.CESMConfig(root_path=tmp_path / "cesm"))
    with pytest.raises((OSError, ValueError)):
        cesm.discover()
    (tmp_path / "vic").mkdir()
    (tmp_path / "vic" / "fluxes.nc").write_bytes(b"not a netcdf file")
    vic = vic_module.VICAdapter(vic_module.VICConfig(root_path=tmp_path / "vic"))
    with pytest.raises((OSError, ValueError)):
        vic.discover()
    # VIC reads NetCDF output only: text files matched by the pattern are rejected, not guessed.
    (tmp_path / "vic_text").mkdir()
    (tmp_path / "vic_text" / "fluxes.txt").write_text("1 2 3 4 5\n", encoding="ascii")
    text = vic_module.VICAdapter(vic_module.VICConfig(root_path=tmp_path / "vic_text", file_pattern="*.txt"))
    with pytest.raises(ValueError, match="NetCDF"):
        text.discover()


def test_cmip6_discovery_rejects_unparsable_and_unreadable_files(tmp_path):
    import importlib

    module = importlib.import_module("relic.coupling.external.climate_models.cmip6_adapter")
    leaf = tmp_path / "ScenarioMIP" / "NCAR" / "CESM2" / "historical" / "r1i1p1f1" / "Amon" / "tas" / "gn" / "v1"
    leaf.mkdir(parents=True)
    (leaf / "notdrs.nc").write_bytes(b"x")
    adapter = module.CMIP6Adapter(module.CMIP6Config(root_path=tmp_path))
    with pytest.raises(ValueError, match="CMIP6 filename"):
        adapter.discover()
    (leaf / "notdrs.nc").unlink()
    (leaf / "tas_Amon_CESM2_historical_r1i1p1f1_gn_200001-200012.nc").write_bytes(b"not a netcdf file")
    unreadable = module.CMIP6Adapter(module.CMIP6Config(root_path=tmp_path))
    with pytest.raises((OSError, ValueError)):
        unreadable.discover()


def test_update_mode_must_be_a_declared_mode():
    with pytest.raises(ValueError, match="update_mode"):
        _DummyAdapter(source=DataSource.WRF, provided_variables=["air_temperature"], name="dummy_external_adapter", update_mode="overwrite")


def test_io_handler_probes_and_decoding_expose_their_errors(tmp_path, monkeypatch):
    import xarray as xr

    from relic.coupling.external.common import io as io_module

    handler = io_module.IOHandler()
    blob = tmp_path / "unknown.bin"
    blob.write_bytes(b"neither netcdf nor hdf5")
    # A file of neither format is reported as such ...
    with pytest.raises(ValueError, match="Could not determine format"):
        handler._detect_format(blob)

    # ... but an unexpected error of the probe propagates instead of being read as "not NetCDF".
    def exploding_open(*args, **kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(io_module.xr, "open_dataset", exploding_open)
    with pytest.raises(RuntimeError, match="probe exploded"):
        handler._detect_format(blob)
    monkeypatch.undo()

    def failing_decode(*args, **kwargs):
        raise ValueError("unable to decode time units")

    monkeypatch.setattr(io_module.xr, "decode_cf", failing_decode)
    with pytest.raises(ValueError, match="decode time units"):
        handler._postprocess_dataset(xr.Dataset({"a": ("x", [1.0])}))
    monkeypatch.undo()

    merged = []

    def failing_concat(*args, **kwargs):
        raise ValueError("cannot concatenate along time")

    monkeypatch.setattr(io_module.xr, "concat", failing_concat)
    monkeypatch.setattr(io_module.xr, "merge", lambda *args, **kwargs: merged.append(True))
    first, second = tmp_path / "a.nc", tmp_path / "b.nc"
    for path in (first, second):
        xr.Dataset({"a": ("time", [1.0])}).to_netcdf(path, engine="netcdf4")
    with pytest.raises((ValueError, IOError), match="concatenate"):
        handler._read_multiple_files([first, second])
    assert merged == []


def test_temporal_spacing_is_measured_for_cftime_records_and_a_failed_check_fails_the_report():
    import cftime
    import numpy as np
    import xarray as xr

    from relic.coupling.external.common.validators import ValidationConfig, Validator

    times = [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 1, 2), cftime.DatetimeNoLeap(2000, 1, 9)]
    dataset = xr.Dataset({"tas": ("time", np.array([280.0, 281.0, 282.0]), {"units": "K"})}, coords={"time": times})
    validator = Validator(ValidationConfig(max_time_gap_hours=48.0))
    issues = validator._validate_temporal_quality(dataset, DataSource.UNKNOWN, {})
    gaps = [issue for issue in issues if "temporal gap" in issue.message]
    assert gaps and "168.0 hours" in gaps[0].message

    def broken_check(ds):
        raise TypeError("custom check bug")

    report = Validator(ValidationConfig(custom_validators=[broken_check])).validate(dataset, DataSource.UNKNOWN)
    assert any("custom check bug" in message for message in report.errors)


def test_wrf_two_metre_mixing_ratio_becomes_specific_humidity(tmp_path):
    """WRF Q2 is the diagnostic 2 m water vapour mixing ratio w (kg kg-1; WRF module_sf_sfclay:
    "Q2 diagnostic 2m mixing ratio"); the declared source transform gives the specific humidity
    q = w / (1 + w) (w = m_v / m_d, q = m_v / (m_v + m_d)) before the catalog conversion."""
    import numpy as np
    import torch
    import xarray as xr

    from relic.coupling.external.common.datasource import DataSource
    from relic.coupling.external.common.schema import InternalSchema

    schema = InternalSchema()
    assert schema.get_source_mapping(DataSource.WRF)["Q2"] == "specific_humidity"
    assert schema.get_source_transform(DataSource.WRF, "Q2") == {
        "kind": "mixing_ratio",
        "mixing_ratio_units": "kg kg-1",
    }
    wrf = _time_config(
        "relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter", "WRFConfig", tmp_path, "lower"
    )
    w = np.array([[0.0, 0.01, 0.02]])
    wrf.current_datasets = {"d01": xr.Dataset({"Q2": (("Time", "cell"), w, {"units": "kg kg-1"})})}
    q = wrf.read(variables=["specific_humidity"])["specific_humidity"]
    torch.testing.assert_close(q, torch.tensor(w / (1.0 + w), dtype=torch.float32))
    assert float(q[0, 2]) < 0.02
    # The file must declare the mixing ratio in the declared units; nothing is assumed.
    for units in (None, "g kg-1"):
        attrs = {} if units is None else {"units": units}
        wrf.current_datasets = {"d01": xr.Dataset({"Q2": (("Time", "cell"), w, attrs)})}
        with pytest.raises(ValueError, match="mixing ratio"):
            wrf.read(variables=["specific_humidity"])


_CONFIGURED = [
    ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter", "WRFConfig", "wrf"),
    ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", "cesm"),
    ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter", "CMIP6Config", "cmip6"),
    ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", "vic"),
    ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter", "SWATConfig", "swat"),
]


def _configured(module_name, class_name, config_name, root, **config):
    import importlib

    module = importlib.import_module(module_name)
    if config_name == "SWATConfig":
        config.setdefault("output_level", "hru")
    return getattr(module, class_name)(getattr(module, config_name)(root_path=root, **config))


@pytest.mark.parametrize(("module_name", "class_name", "config_name", "source"), _CONFIGURED)
def test_adapter_metadata_declares_the_configured_catalog_outputs(
    tmp_path, module_name, class_name, config_name, source
):
    """The adapter's outputs are the catalog variables its configuration provides (every mapping
    target of the source when provided_variables is None); it reads no state and is the exclusive
    writer of its outputs (a second writer of one of them is refused at registration)."""
    from relic.core.functor import ConflictResolutionPolicy, VariableKind
    from relic.coupling.external.common.datasource import DataSource
    from relic.coupling.external.common.schema import InternalSchema

    mapping = InternalSchema().get_source_mapping(DataSource(source))
    every = sorted(set(mapping.values()))
    default = _configured(module_name, class_name, config_name, tmp_path)
    assert default.metadata.input_variables == []
    assert default.metadata.output_variables == every
    assert default.metadata.variable_kind == VariableKind.FORCING
    assert default.metadata.conflict_resolution_policy == ConflictResolutionPolicy.EXCLUSIVE
    chosen = every[:2]
    configured = _configured(module_name, class_name, config_name, tmp_path, provided_variables=chosen)
    assert configured.metadata.output_variables == chosen
    for unknown in (["wind_u"], ["not_a_catalog_variable"]):
        with pytest.raises(KeyError):
            _configured(module_name, class_name, config_name, tmp_path, provided_variables=unknown)
    with pytest.raises(ValueError, match="duplicate"):
        _configured(module_name, class_name, config_name, tmp_path, provided_variables=[chosen[0], chosen[0]])


def test_forward_returns_exactly_the_provided_variables_in_catalog_units(tmp_path):
    """forward writes the provided variables (UpdateItems in the catalog units, which StateManager
    checks against the contract units); a provided variable the data lack raises, and a request
    outside the provided variables raises."""
    import numpy as np
    import xarray as xr

    from relic.coupling.external.common.schema import InternalSchema

    vic = _configured(
        "relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", tmp_path,
        time_interpolation="lower", provided_variables=["surface_snow_amount", "soil_temperature"],
    )
    vic.current_datasets = _vic_by_time(tmp_path).current_datasets
    updates = vic.forward({}, time=0.0, dt=3600.0, **_NOLEAP)
    assert set(updates) == {"surface_snow_amount", "soil_temperature"}
    schema = InternalSchema()
    for name, item in updates.items():
        assert item.unit == schema.get_variable(name).units
    only = vic.forward({}, time=0.0, dt=3600.0, variables=["soil_temperature"], **_NOLEAP)
    assert set(only) == {"soil_temperature"}
    with pytest.raises(ValueError, match="provided"):
        vic.forward({}, time=0.0, dt=3600.0, variables=["wind_speed"], **_NOLEAP)
    vic.current_datasets["main"] = xr.Dataset(
        {"OUT_SWE": (("time", "cell"), np.array([[1.0, 2.0]]), {"units": "mm"})},
        coords={"time": _noleap_days(1)},
    )
    with pytest.raises(ValueError, match="soil_temperature"):
        vic.forward({}, time=0.0, dt=3600.0, **_NOLEAP)


def _simulation_adapters(root):
    import numpy as np
    import xarray as xr

    vic = _configured(
        "relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter", "VICConfig", root,
        time_interpolation="lower", record_interval=1.0, record_interval_units="day",
        provided_variables=["surface_snow_amount", "soil_temperature", "evapotranspiration_flux"],
    )
    vic.current_datasets = {
        "main": xr.Dataset(
            {
                "OUT_SWE": (("time", "cell"), np.array([[150.0, 0.0], [140.0, 10.0]]), {"units": "mm"}),
                "OUT_SOIL_TEMP": (("time", "cell"), np.array([[7.0, 8.0], [9.0, 10.0]]), {"units": "degC"}),
                "OUT_EVAP": (("time", "cell"), np.array([[8.64, 0.864], [4.32, 0.0]]), {"units": "mm"}),
            },
            coords={"time": _noleap_days(2)},
        )
    }
    cesm = _configured(
        "relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter", "CESMConfig", root,
        time_interpolation="lower",
        provided_variables=["atmospheric_pressure", "shortwave_down", "latent_heat_flux"],
    )
    cesm.current_datasets = {
        "h0": xr.Dataset(
            {
                "PS": (("time", "cell"), np.array([[101325.0, 98000.0], [100000.0, 97000.0]]), {"units": "Pa"}),
                "FSDS": (("time", "cell"), np.array([[400.0, 300.0], [200.0, 100.0]]), {"units": "W m-2"}),
                "EFLX_LH_TOT": (("time", "cell"), np.array([[120.0, -5.0], [60.0, 0.0]]), {"units": "W m-2"}),
            },
            coords={"time": _noleap_days(2)},
        )
    }
    return vic, cesm


@pytest.mark.parametrize("conservation_checks", [True, False])
def test_registered_adapters_write_their_forcing_into_the_simulator_state(tmp_path, conservation_checks):
    """Two adapters registered in a RELICSimulator: every step writes each provided variable
    through StateManager (contract writers, contract units), at the simulation time."""
    import torch

    from relic.core.simulator import RELICSimulator

    sim = RELICSimulator({
        "device": "cpu", "calendar_type": "noleap", "time_origin": "2000-01-01T00:00:00",
        "conservation_checks": conservation_checks, "strict_dependencies": True,
        "adaptive_dt_enabled": False,
    })
    vic, cesm = _simulation_adapters(tmp_path)
    # Initial values inside the contract bounds; the first step replaces every one of them.
    initial = {
        "surface_snow_amount": 0.0, "soil_temperature": 273.15, "evapotranspiration_flux": 0.0,
        "atmospheric_pressure": 101325.0, "shortwave_down": 0.0, "latent_heat_flux": 0.0,
    }
    assert set(initial) == set(vic.provided_variables) | set(cesm.provided_variables)
    sim.initialize_state({name: torch.full((2,), value) for name, value in initial.items()})
    sim.register_functor(vic, execution_group="fast")
    sim.register_functor(cesm, execution_group="fast")
    day = 86400.0

    def state(name):
        return sim.get_state()[name].detach().to(torch.float64).tolist()

    sim.step(dt=day)
    assert state("surface_snow_amount") == pytest.approx([150.0, 0.0])
    assert state("soil_temperature") == pytest.approx([280.15, 281.15])
    assert state("evapotranspiration_flux") == pytest.approx([1.0e-4, 1.0e-5])
    assert state("atmospheric_pressure") == pytest.approx([101325.0, 98000.0])
    assert state("shortwave_down") == pytest.approx([400.0, 300.0])
    assert state("latent_heat_flux") == pytest.approx([120.0, -5.0])
    sim.step(dt=day)
    assert state("surface_snow_amount") == pytest.approx([140.0, 10.0])
    assert state("atmospheric_pressure") == pytest.approx([100000.0, 97000.0])
    assert sim.get_variable_provenance("surface_snow_amount")[-1]["functor"] == "VICAdapter"
    assert sim.get_variable_provenance("shortwave_down")[-1]["functor"] == "CESMAdapter"


def test_an_internal_process_and_an_adapter_never_write_one_variable(tmp_path):
    """Registration refuses a second writer of an adapter output: an internal process
    (Penman-Monteith ET, radiation) and an adapter, or two adapters, in either order."""
    from relic.core.simulator import RELICSimulator
    from relic.scientific.atmosphere.radiation import RadiationFunctor
    from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET

    def simulator():
        return RELICSimulator({"device": "cpu", "strict_dependencies": False})

    vic, cesm = _simulation_adapters(tmp_path)
    sim = simulator()
    sim.register_functor(PenmanMonteithET(config={"canopy_resistance_source": "state"}))
    with pytest.raises(ValueError, match="evapotranspiration_flux"):
        sim.register_functor(vic)
    assert "VICAdapter" not in sim.functors

    sim = simulator()
    sim.register_functor(cesm)
    with pytest.raises(ValueError, match="shortwave_down"):
        sim.register_functor(RadiationFunctor())

    sim = simulator()
    sim.register_functor(
        _configured("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter", "WRFConfig", tmp_path,
                    provided_variables=["air_temperature"])
    )
    with pytest.raises(ValueError, match="air_temperature"):
        sim.register_functor(
            _configured("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter", "CMIP6Config",
                        tmp_path, provided_variables=["air_temperature"])
        )


def test_wrf_ten_metre_wind_is_rotated_to_earth_relative_components(tmp_path):
    """WRF U10/V10 are grid-relative (UCAR MMM WRF forum: "U10 & V10 wind components are grid
    relative"); eastward_wind/northward_wind are earth-relative, so the declared source transform
    rotates them with the file's COSALPHA/SINALPHA: u_e = u cos(a) - v sin(a), v_e = v cos(a) + u sin(a)."""
    import numpy as np
    import torch
    import xarray as xr

    from relic.coupling.external.common.datasource import DataSource
    from relic.coupling.external.common.schema import InternalSchema

    schema = InternalSchema()
    for field, component in (("U10", "eastward"), ("V10", "northward")):
        assert schema.get_source_transform(DataSource.WRF, field) == {
            "kind": "earth_relative_wind",
            "component": component,
            "u_field": "U10",
            "v_field": "V10",
            "cos_field": "COSALPHA",
            "sin_field": "SINALPHA",
        }
    wrf = _time_config(
        "relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter", "WRFConfig", tmp_path, "lower"
    )
    alpha = np.deg2rad(np.array([[30.0, -10.0]]))
    u = np.array([[1.0, 2.0]])
    v = np.array([[0.0, -1.0]])
    fields = {
        "U10": (("Time", "cell"), u, {"units": "m s-1"}),
        "V10": (("Time", "cell"), v, {"units": "m s-1"}),
        "COSALPHA": (("Time", "cell"), np.cos(alpha), {"units": "1"}),
        "SINALPHA": (("Time", "cell"), np.sin(alpha), {"units": "1"}),
    }
    wrf.current_datasets = {"d01": xr.Dataset(fields)}
    winds = wrf.read(variables=["eastward_wind", "northward_wind"])
    expected_u = u * np.cos(alpha) - v * np.sin(alpha)
    expected_v = v * np.cos(alpha) + u * np.sin(alpha)
    torch.testing.assert_close(winds["eastward_wind"], torch.tensor(expected_u, dtype=torch.float32))
    torch.testing.assert_close(winds["northward_wind"], torch.tensor(expected_v, dtype=torch.float32))
    # Without the rotation fields the components cannot be made earth-relative: nothing is assumed.
    for missing in ("COSALPHA", "SINALPHA", "V10"):
        wrf.current_datasets = {"d01": xr.Dataset({k: val for k, val in fields.items() if k != missing})}
        with pytest.raises(ValueError, match=missing):
            wrf.read(variables=["eastward_wind"])


@pytest.mark.parametrize(("module_name", "class_name", "config_name", "source"), _CONFIGURED)
def test_adapter_metadata_describes_only_the_actual_mapping_and_transforms(
    tmp_path, module_name, class_name, config_name, source
):
    """The expression an adapter declares is its read path, generated from InternalSchema: the
    schema mapping and one formula per declared source transform, and no unmapped diagnostic
    (WRF: no potential temperature, eta pressure or wind speed). FunctorMetadata stores it in the
    canonical IPHS form, whose outputs are the provided catalog variables."""
    import importlib
    import inspect

    from relic.coupling.external.common import base_adapter
    from relic.coupling.external.common.base_adapter import TRANSFORM_EXPRESSIONS
    from relic.coupling.external.common.datasource import DataSource
    from relic.coupling.external.common.schema import InternalSchema

    data_source = DataSource(source)
    expression = ExternalAdapter.mapping_expression(data_source)
    for external, internal in InternalSchema().get_source_mapping(data_source).items():
        assert f"{external}->{internal}" in expression
    kinds = {declaration["kind"] for declaration in InternalSchema.source_transforms().get(data_source, {}).values()}
    for kind in kinds:
        assert TRANSFORM_EXPRESSIONS[kind] in expression
    for kind in set(TRANSFORM_EXPRESSIONS) - kinds:
        assert TRANSFORM_EXPRESSIONS[kind] not in expression
    all_kinds = {
        declaration["kind"]
        for table in InternalSchema.source_transforms().values()
        for declaration in table.values()
    }
    assert all_kinds <= set(TRANSFORM_EXPRESSIONS)
    # The adapters declare this generated expression, never a hand-written formula.
    declared = inspect.getsource(importlib.import_module(module_name)) + inspect.getsource(base_adapter.ExternalAdapter.__init__)
    assert "mapping_expression(" in declared
    for stale in ("theta=T*", "P(eta)=", "wind_speed=sqrt", "x_internal = T("):
        assert stale not in declared, stale
    adapter = _configured(module_name, class_name, config_name, tmp_path)
    outputs_line = next(
        line for line in adapter.metadata.mathematical_expression.splitlines() if line.strip().startswith("outputs:")
    )
    # The canonical form lists the first 12 outputs and counts the rest (iphs_metadata._format_sequence).
    provided = list(adapter.provided_variables)
    expected = ", ".join(provided) if len(provided) <= 12 else f"{', '.join(provided[:12])}, and {len(provided) - 12} more"
    assert outputs_line.split(":", 1)[1].strip() == expected
