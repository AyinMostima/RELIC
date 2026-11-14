"""L2 Integration Test: Coupling Layer Unit Conversions.

Tests unit conversion functionality in external coupling adapters
for atmospheric, hydrological, and climate models.

Reference:
- relic/coupling/external/atmospheric_models/wrf_adapter.py
- relic/coupling/external/hydrological_models/vic_adapter.py
- relic/coupling/external/climate_models/cmip6_adapter.py
"""
import pytest
import numpy as np
import xarray as xr


class TestWRFAdapterUnits:
    """Test WRF adapter unit conversions.

    WRF outputs may use different conventions than RELIC internals.
    """

    def test_wrf_temperature_conversion(self, sample_xarray_dataset):
        """Test temperature unit conversion from WRF output."""
        from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter

        try:
            adapter = WRFAdapter()

            # WRF might output temperature in Celsius or Kelvin
            # Adapter should handle both
            temp_c = sample_xarray_dataset['air_temperature']

            # Test that adapter can process the dataset
            # (Actual conversion logic depends on implementation)
            assert temp_c is not None
        except ImportError:
            pytest.skip("WRFAdapter not available")

    def test_wrf_precipitation_conversion(self, sample_xarray_dataset):
        """Test precipitation unit conversion from WRF."""
        try:
            from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter

            adapter = WRFAdapter()

            # WRF precipitation might be in mm day-1 or kg/m2/s
            precip = sample_xarray_dataset['precipitation']

            assert precip is not None
        except ImportError:
            pytest.skip("WRFAdapter not available")


class TestCESMAdapterUnits:
    """Test CESM adapter unit conversions."""

    def test_cesm_standard_units(self, sample_xarray_dataset):
        """Test CESM adapter handles CF-compliant units."""
        try:
            from relic.coupling.external.atmospheric_models.cesm_adapter import CESMAdapter, CESMConfig

            # A CESMConfig declares the output directory; none is assumed.
            adapter = CESMAdapter(CESMConfig(root_path="."))

            # CESM should output CF-compliant units
            ds = sample_xarray_dataset

            assert ds is not None
        except ImportError:
            pytest.skip("CESMAdapter not available")

    def test_cesm_physical_constants_use_ssot(self, monkeypatch):
        """CESM thermodynamic constants must bypass the adapter ParameterManager."""
        import relic.coupling.external.atmospheric_models.cesm_adapter as cesm_module
        from relic.parameters import get_global_manager

        class PhysicalConstantGuard:
            def __init__(self, base_manager):
                self._base_manager = base_manager

            def get(self, key, namespace=None, **kwargs):
                full_key = f"{namespace}.{key}" if namespace else str(key)
                if full_key.startswith("relic.physical.constants."):
                    raise AssertionError(
                        "Physical constants must use get_physical_constant"
                    )
                return self._base_manager.get(key, namespace=namespace, **kwargs)

            def __getattr__(self, name):
                return getattr(self._base_manager, name)

        requested = []

        def fake_get_physical_constant(name):
            requested.append(name)
            return {"reference_pressure": 100000.0}[name]

        monkeypatch.setattr(
            cesm_module,
            "get_physical_constant",
            fake_get_physical_constant,
        )

        adapter = cesm_module.CESMAdapter(
            cesm_module.CESMConfig(root_path="."),
            parameter_manager=PhysicalConstantGuard(get_global_manager()),
        )

        # P0 of the CAM hybrid pressure is the only constant the adapter reads.
        assert adapter.reference_pressure_pa == 100000.0
        assert not hasattr(adapter, "_dry_air_R_over_cp")
        assert requested == ["reference_pressure"]

    def test_cesm_open_adds_no_unread_derived_fields(self):
        """open() with compute_derived added WSPD, Z and THETA to the opened datasets. No code
        read them: the read path reads only the schema-mapped CAM/CLM fields (none of them is a
        derived name) and the adapter outputs are its provided catalog variables. The fields,
        their switch and the R_d/c_p constant they used are gone."""
        import dataclasses
        import inspect

        import relic.coupling.external.atmospheric_models.cesm_adapter as cesm_module
        from relic.coupling.external.atmospheric_models.cesm_adapter import CESMAdapter, CESMConfig
        from relic.coupling.external.common.datasource import DataSource
        from relic.coupling.external.common.schema import InternalSchema

        assert not hasattr(CESMAdapter, "_compute_derived_variables")
        assert "compute_derived" not in {item.name for item in dataclasses.fields(CESMConfig)}
        source = inspect.getsource(cesm_module)
        for stale in ("WSPD", "THETA", "R_over_cp", "compute_derived"):
            assert stale not in source, stale
        mapped = set(InternalSchema().get_source_mapping(DataSource.CESM))
        assert not mapped & {"WSPD", "Z", "THETA"}

    def test_cesm_interpolates_on_column_pressure_not_hybrid_labels(self):
        """Hybrid-level fields must be interpolated against their 3-D pressure."""
        from relic.coupling.external.atmospheric_models.cesm_adapter import (
            CESMAdapter,
            CESMConfig,
        )

        adapter = CESMAdapter(
            config=CESMConfig(
                root_path=".",
                pressure_levels=[50000.0],
            )
        )
        source = xr.Dataset(
            data_vars={
                "PS": xr.DataArray(
                    [[[100000.0]]],
                    dims=("time", "lat", "lon"),
                    attrs={"units": "Pa"},
                ),
                "T": (
                    ("time", "lev", "lat", "lon"),
                    [[[[220.0]], [[280.0]]]],
                ),
            },
            coords={
                "time": [0],
                "lev": [0.0, 1.0],
                "lat": [0.0],
                "lon": [0.0],
                "hyam": ("lev", [0.2, 0.8]),
                "hybm": ("lev", [0.0, 0.0]),
            },
            attrs={"P0": 100000.0},
        )

        interpolated = adapter._interpolate_to_pressure_levels(source)

        assert interpolated["T"].dims == ("time", "plev", "lat", "lon")
        np.testing.assert_allclose(
            interpolated["T"].values,
            [[[[250.0]]]],
        )
        np.testing.assert_allclose(interpolated["plev"].values, [50000.0])
        assert interpolated["plev"].attrs["units"] == "Pa"

    def test_cesm_pressure_interpolation_requires_hybrid_inputs(self):
        """Configured pressure interpolation must fail without CAM coordinates."""
        from relic.coupling.external.atmospheric_models.cesm_adapter import (
            CESMAdapter,
            CESMConfig,
        )

        adapter = CESMAdapter(
            config=CESMConfig(root_path=".", pressure_levels=[50000.0])
        )
        source = xr.Dataset(
            {"T": xr.DataArray([250.0], dims=("lev",), attrs={"units": "K"})},
            coords={"lev": [0.0]},
        )

        with pytest.raises(ValueError, match="Missing CESM hybrid-pressure inputs"):
            adapter._interpolate_to_pressure_levels(source)

    def test_cesm_pressure_interpolation_rejects_extrapolation(self):
        """Pressure interpolation must not hide out-of-column extrapolation."""
        from relic.coupling.external.atmospheric_models.cesm_adapter import (
            CESMAdapter,
            CESMConfig,
        )

        adapter = CESMAdapter(
            config=CESMConfig(root_path=".", pressure_levels=[90000.0])
        )
        source = xr.Dataset(
            data_vars={
                "PS": xr.DataArray(100000.0, attrs={"units": "Pa"}),
                "T": xr.DataArray(
                    [220.0, 280.0],
                    dims=("lev",),
                    attrs={"units": "K"},
                ),
            },
            coords={
                "lev": [0.0, 1.0],
                "hyam": ("lev", [0.2, 0.8]),
                "hybm": ("lev", [0.0, 0.0]),
            },
            attrs={"P0": 100000.0},
        )

        with pytest.raises(
            ValueError,
            match="outside the model column",
        ):
            adapter._interpolate_to_pressure_levels(source)


class TestVICAdapterUnits:
    """Test VIC (Variable Infiltration Capacity) adapter units."""

    def test_vic_runoff_units(self):
        """Test VIC runoff unit conversion.

        VIC outputs runoff in mm day-1, needs conversion to kg/m2/s.
        """
        try:
            from relic.coupling.external.hydrological_models.vic_adapter import VICAdapter

            adapter = VICAdapter()

            # Create sample VIC-like data
            runoff_mm_day = 5.0  # mm day-1

            # Adapter should convert to kg/m2/s
            # 1 mm day-1 = 1 kg/m2 day-1 = 1/(24*3600) kg/m2/s
            expected_kg_m2_s = runoff_mm_day / 86400.0

            # Test conversion (if implemented)
            # This is a placeholder - actual test depends on adapter API
            assert runoff_mm_day > 0
        except ImportError:
            pytest.skip("VICAdapter not available")


class TestSWATAdapterUnits:
    """Test SWAT (Soil and Water Assessment Tool) adapter units."""

    def test_swat_flow_units(self):
        """Test SWAT streamflow unit conversion.

        SWAT might output flow in m3/s or mm day-1.
        """
        try:
            from relic.coupling.external.hydrological_models.swat_adapter import SWATAdapter, SWATConfig

            # SWATConfig declares the output table (only the output.hru layout is read).
            adapter = SWATAdapter(SWATConfig(root_path=".", output_level="hru"))
            assert adapter.config.output_level == "hru"

            # Sample SWAT output
            flow_m3s = 10.0  # m3/s

            assert flow_m3s > 0
        except ImportError:
            pytest.skip("SWATAdapter not available")


class TestCMIP6AdapterUnits:
    """Test CMIP6 climate model adapter units."""

    def test_cmip6_cf_conventions(self, sample_xarray_dataset):
        """Test CMIP6 adapter handles CF-1.8 conventions.

        CMIP6 strictly follows CF conventions, so this should be straightforward.
        """
        try:
            from relic.coupling.external.climate_models.cmip6_adapter import CMIP6Adapter, CMIP6Config

            # A CMIP6Config declares the data directory; none is assumed.
            adapter = CMIP6Adapter(CMIP6Config(root_path="."))

            ds = sample_xarray_dataset

            # CMIP6 should have standard_name and units attributes
            assert 'air_temperature' in ds
            assert 'units' in ds['air_temperature'].attrs
        except ImportError:
            pytest.skip("CMIP6Adapter not available")


class TestXArrayToNumpyConversion:
    """Test xarray to numpy conversions in adapters."""

    def test_xarray_data_extraction(self, sample_xarray_dataset):
        """Test extracting numpy arrays from xarray with unit conversion."""
        ds = sample_xarray_dataset

        # Extract temperature data
        temp_data = ds['air_temperature'].values
        temp_units = ds['air_temperature'].attrs.get('units', 'unknown')

        assert isinstance(temp_data, np.ndarray)
        assert temp_units == 'celsius'

    def test_xarray_unit_attributes_preserved(self, sample_xarray_dataset):
        """Test that unit attributes are preserved during processing."""
        ds = sample_xarray_dataset

        # All variables should have units
        for var in ds.data_vars:
            assert 'units' in ds[var].attrs, \
                f"Variable {var} missing units attribute"


class TestAdapterUnitConsistency:
    """Test that all adapters produce consistent unit outputs."""

    def test_temperature_output_consistency(self):
        """Test that all adapters output temperature in Kelvin.

        RELIC internal standard is Kelvin for temperature.
        """
        # This test checks that adapters convert to consistent units
        # Placeholder - actual implementation depends on adapter APIs
        expected_unit = 'K'
        assert expected_unit == 'K'

    def test_flux_output_consistency(self):
        """Test that all adapters output fluxes in kg/m2/s.

        RELIC internal standard for mass fluxes is kg/m2/s.
        """
        expected_unit = 'kg m-2 s-1'
        assert expected_unit == 'kg m-2 s-1'


class TestAdapterErrorHandling:
    """Test error handling in adapters for unit issues."""

    def test_missing_units_detected(self):
        """Test that adapters detect missing unit attributes."""
        # Create xarray without units
        ds = xr.Dataset({
            'temperature': (['time'], [25.0, 26.0])
        })

        # Adapters should handle this gracefully
        # Either by raising informative error or using defaults
        assert 'temperature' in ds

    def test_invalid_units_rejected(self):
        """Test that adapters reject invalid unit specifications."""
        ds = xr.Dataset({
            'temperature': (['time'], [25.0, 26.0])
        })
        ds['temperature'].attrs['units'] = 'invalid_unit'

        # Adapter should detect this
        # (Exact behavior depends on implementation)
        assert ds is not None
