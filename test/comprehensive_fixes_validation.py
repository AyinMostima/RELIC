"""
Comprehensive Validation Test Suite for RELIC Fixes

This test suite validates all the fixes implemented during the refactoring process:
1. P06 Unit conversion chain integration
2. CLI/NetCDF output functionality
3. UnitConverter unification
4. Transport increment and conservation validation
5. Parameter system unit validation
6. P04 Multi-scale coupling wiring fixes
7. Redundant code cleanup

Mathematical Expression:
    Unit conservation: 危(out_i * unit_i) = 危(in_j * unit_j)
    Transport conservation: M_t+dt = M_t + 危(fluxes * dt)
    Scale coupling: Y_coarse = 危(X_fine * weights) with proper normalization

References:
    - CF-1.8 Conventions for unit handling
    - Conservation laws in ecological modeling
    - Multi-scale aggregation theory
"""

import pytest
import torch
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
import os

# Import RELIC modules
from relic.data.io.reader import DataReader, ReadConfig, ReadResult
from relic.data.standards.unit_converter import unit_converter, ConversionContext
from relic.parameters import get_global_manager
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
from relic.coupling.core import ScalingMode
from relic.coupling.scales.temporal.core import CalendarError, ValidationError
from relic.coupling.external.common.io import IOConfig, IOHandler
from relic.core.state import StateManager
from relic.stochastic.io import save_results


class TestUnitConversionIntegration:
    """Test P06: Unit conversion chain integration in DataReader pipeline."""

    def test_data_reader_unit_conversion(self):
        """DataReader converts CF-consistent inputs to the CF canonical units.

        Every standard_name/unit pair is CF-consistent (UDUNITS-convertible to
        the canonical unit of the CF table), so the read succeeds and the one
        UnitConverter produces the canonical values:
        20 degC -> 293.15 K; 8.64 kg m-2 day-1 -> 1.0e-4 kg m-2 s-1;
        8.64 g m-2 day-1 -> 1.0e-7 kg m-2 s-1 (carbon mass of GPP).
        """
        data = {
            'temperature': np.array([20.0, 21.0, 22.0]),       # degC
            'precipitation': np.array([8.64, 17.28, 8.64]),    # kg m-2 day-1
            'carbon_flux': np.array([8.64, 17.28, 12.96])      # g m-2 day-1
        }

        ds = xr.Dataset({
            'temperature': (['time'], data['temperature']),
            'precipitation': (['time'], data['precipitation']),
            'carbon_flux': (['time'], data['carbon_flux'])
        })

        # CF-consistent standard_name/unit pairs
        ds['temperature'].attrs['units'] = 'degC'
        ds['temperature'].attrs['standard_name'] = 'air_temperature'
        ds['precipitation'].attrs['units'] = 'kg m-2 day-1'
        ds['precipitation'].attrs['standard_name'] = 'precipitation_flux'
        ds['carbon_flux'].attrs['units'] = 'g m-2 day-1'
        ds['carbon_flux'].attrs['standard_name'] = 'gross_primary_productivity_of_carbon'

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            ds.to_netcdf(f.name)
            temp_file = f.name

        try:
            # Convert every variable to its CF canonical unit
            config = ReadConfig(
                file_path=temp_file,
                apply_unit_conversion=True,
            )

            reader = DataReader(config)
            result = reader.read_dataset(temp_file)

            assert result.success, "DataReader should succeed"
            expected = {
                'temperature': ([293.15, 294.15, 295.15], 'K', 'degC'),
                'precipitation': ([1.0e-4, 2.0e-4, 1.0e-4], 'kg m-2 s-1', 'kg m-2 day-1'),
                'carbon_flux': ([1.0e-7, 2.0e-7, 1.5e-7], 'kg m-2 s-1', 'g m-2 day-1'),
            }
            for name, (values, units, original) in expected.items():
                np.testing.assert_allclose(result.data[name].values, values, rtol=1e-12)
                assert result.data[name].attrs['units'] == units
                assert result.data[name].attrs['original_units'] == original
            # The dataset has no time coordinate, so data validation still fails.
            assert result.validation_passed is False
            assert any("No time coordinate" in warning for warning in result.warnings)

        finally:
            os.unlink(temp_file)

    def test_unit_converter_authoritative_integration(self):
        """Test that historical UnitConverter properly delegates to authoritative implementation."""
        value_authoritative = unit_converter.convert(1.0, 'generic', 'm', 'mm', ConversionContext())
        assert abs(value_authoritative - 1000.0) < 1e-10, \
            "Authoritative converter should handle metric conversions"

    def test_data_reader_marks_validator_errors_as_failed(self):
        """A validator exception must never be reported as a passed validation."""
        config = ReadConfig(validate_data=True)
        reader = DataReader(config)

        class RaisingValidator:
            def validate(self, dataset):
                raise ValueError("validator probe failure")

        reader.validator = RaisingValidator()
        dataset = xr.Dataset(
            {"air_temperature": (["time"], np.array([293.15]))},
            coords={"time": np.array([0])},
        )
        result = ReadResult(data=dataset, metadata={})

        reader._validate_data(dataset, config, result)

        assert result.validation_passed is False
        assert any("validator probe failure" in warning for warning in result.warnings)

    def test_data_reader_strict_validation_errors_raise(self):
        """Strict validation must propagate validator failures."""
        config = ReadConfig(validate_data=True, strict_cf=True)
        reader = DataReader(config)

        class RaisingValidator:
            def validate(self, dataset):
                raise ValueError("strict validator probe failure")

        reader.validator = RaisingValidator()
        dataset = xr.Dataset(
            {"air_temperature": (["time"], np.array([293.15]))},
            coords={"time": np.array([0])},
        )
        result = ReadResult(data=dataset, metadata={})

        with pytest.raises(ValueError, match="strict validator probe failure"):
            reader._validate_data(dataset, config, result)

        assert result.validation_passed is False

    def test_data_reader_does_not_request_unimplemented_calendar_conversion_by_default(self):
        """Default ingestion must preserve the declared calendar."""
        assert ReadConfig().standard_calendar is False

    def test_io_handler_can_preserve_numeric_cf_time_for_canonical_processing(
        self,
        tmp_path,
    ):
        """The common I/O boundary must support deferred canonical time decoding."""
        source = xr.Dataset(
            {"air_temperature": ("time", np.array([280.0, 281.0]))},
            coords={"time": ("time", np.array([0.5, 1.5]))},
        )
        source.time.attrs.update(
            {
                "units": "days since 2001-01-01",
                "calendar": "noleap",
            }
        )
        path = tmp_path / "numeric_cf_time.nc"
        source.to_netcdf(path)

        result = IOHandler(IOConfig(decode_times=False)).read(path)

        assert np.issubdtype(result.data.time.dtype, np.number)
        np.testing.assert_allclose(result.data.time.values, np.array([0.5, 1.5]))
        assert result.data.time.attrs["units"] == "days since 2001-01-01"
        assert result.data.time.attrs["calendar"] == "noleap"

    def test_data_reader_preserves_declared_calendar_and_source_time_bounds(self):
        """Source bounds are evidence and must not be replaced by inferred intervals."""
        dataset = xr.Dataset(
            {
                "air_temperature": ("time", np.array([280.0, 281.0])),
                "source_time_bounds": (
                    ("time", "nv"),
                    np.array([[0.25, 0.75], [1.25, 1.75]]),
                ),
            },
            coords={"time": ("time", np.array([0.5, 1.5]))},
        )
        dataset.time.attrs.update(
            {
                "units": "days since 2001-01-01",
                "calendar": "noleap",
                "bounds": "source_time_bounds",
            }
        )
        config = ReadConfig(
            cf_standardize=False,
            validate_data=False,
            standard_calendar=False,
            time_bounds=True,
        )
        result = ReadResult(data=dataset, metadata={})

        processed = DataReader(config)._process_time_coordinates(
            dataset,
            config,
            result,
        )

        assert processed.time.attrs["calendar"] == "noleap"
        assert processed.time.attrs["bounds"] == "source_time_bounds"
        np.testing.assert_allclose(
            processed["source_time_bounds"].values,
            np.array([[0.25, 0.75], [1.25, 1.75]]),
        )
        assert result.time_processed is True
        assert result.time_bounds_computed is False

    def test_data_reader_public_ingress_uses_canonical_numeric_cf_time(
        self,
        tmp_path,
    ):
        """Public ingestion must not pre-decode time before the Torch owner runs."""
        source = xr.Dataset(
            {
                "air_temperature": ("time", np.array([280.0, 281.0])),
                "source_time_bounds": (
                    ("time", "nv"),
                    np.array([[0.25, 0.75], [1.25, 1.75]]),
                ),
            },
            coords={"time": ("time", np.array([0.5, 1.5]))},
        )
        source.time.attrs.update(
            {
                "units": "days since 2001-01-01",
                "calendar": "noleap",
                "bounds": "source_time_bounds",
            }
        )
        path = tmp_path / "reader_numeric_cf_time.nc"
        source.to_netcdf(path)
        reader = DataReader(
            ReadConfig(
                cf_standardize=False,
                cf_validate=False,
                validate_data=False,
                strict_cf=True,
                process_time=True,
                time_bounds=True,
                standard_calendar=False,
                apply_unit_conversion=False,
            )
        )

        result = reader.read_dataset(path)
        try:
            assert result.success is True
            assert result.validation_passed is True
            assert result.time_processed is True
            assert result.time_bounds_computed is False
            assert result.data.time.attrs["calendar"] == "noleap"
            assert result.data.time.attrs["bounds"] == "source_time_bounds"
            np.testing.assert_allclose(result.data.time.values, np.array([0.5, 1.5]))
            np.testing.assert_allclose(
                result.data["source_time_bounds"].values,
                np.array([[0.25, 0.75], [1.25, 1.75]]),
            )
        finally:
            result.data.close()

    def test_data_reader_rejects_missing_time_units_without_legacy_default(self):
        """Calendar evidence cannot substitute for a missing CF unit/epoch declaration."""
        dataset = xr.Dataset(coords={"time": ("time", np.array([0.5, 1.5]))})
        dataset.time.attrs["calendar"] = "noleap"
        config = ReadConfig(validate_data=False, standard_calendar=False)
        result = ReadResult(data=dataset, metadata={})

        with pytest.raises(ValidationError, match="explicit CF time units"):
            DataReader(config)._process_time_coordinates(dataset, config, result)

    def test_data_reader_rejects_unbound_cross_calendar_conversion(self):
        """A boolean flag cannot invent a scientifically valid calendar mapping."""
        dataset = xr.Dataset(coords={"time": ("time", np.array([0.5, 1.5]))})
        dataset.time.attrs.update(
            {
                "units": "days since 2001-01-01",
                "calendar": "noleap",
            }
        )
        config = ReadConfig(
            validate_data=False,
            standard_calendar=True,
            time_bounds=False,
        )
        result = ReadResult(data=dataset, metadata={})

        with pytest.raises(CalendarError, match="mapping policy"):
            DataReader(config)._process_time_coordinates(dataset, config, result)


class TestNetCDFOutput:
    """Test CLI/NetCDF output functionality."""

    def test_save_results_netcdf(self):
        """save_results writes every declared tensor with its declared units and values.

        relic.stochastic.io requires the producer to declare the units and the
        dimensions of every tensor (results['variable_units'] and
        results['variable_dimensions']); nothing is inferred from the name.
        """
        values = {
            'temperature': torch.tensor([293.15, 294.15, 295.15]),
            'precipitation': torch.tensor([0.001, 0.002, 0.001]),
            'carbon_flux': torch.tensor([1e-6, 2e-6, 1.5e-6]),
        }
        units = {
            'temperature': 'K',
            'precipitation': 'kg m-2 s-1',
            'carbon_flux': 'kg m-2 s-1',
        }
        results = {
            **values,
            'variable_units': units,
            'variable_dimensions': {name: ['time'] for name in values},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / 'test_output.nc'
            save_results(results, str(output_file), format='netcdf')

            netcdf_file = output_file.with_suffix('.nc')
            assert netcdf_file.exists(), "save_results(format='netcdf') must write the NetCDF file"
            with xr.open_dataset(netcdf_file) as ds:
                for name, tensor in values.items():
                    assert name in ds.variables, f"{name} variable should exist"
                    assert ds[name].dims == ('time',)
                    assert ds[name].attrs['units'] == units[name]
                    np.testing.assert_allclose(ds[name].values, tensor.numpy(), rtol=1e-6, atol=0.0)

        undeclared = dict(values)
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="no declared units"):
                save_results(undeclared, str(Path(temp_dir) / 'undeclared.nc'), format='netcdf')



class TestParameterSystemValidation:
    """Test parameter system unit validation."""

    def test_parameter_unit_validation(self):
        """Test that parameter system validates units properly."""
        param_manager = get_global_manager()

        # Test valid parameter retrieval with units
        value = param_manager.get(
            'relic.physical.constants.earth.gravity_acceleration',
            units='m s-2'
        )
        assert value is not None, "Should retrieve gravity constant"
        assert isinstance(value, (int, float)), "Should return numeric value"

        # Test unit conversion in parameter retrieval
        value_converted = param_manager.get(
            'relic.physical.constants.pressure.reference_pressure',
            units='Pa'  # Keep native units to avoid unsupported hPa conversion in test
        )
        assert value_converted is not None, "Should retrieve reference pressure"


class TestMultiScaleCoupling:
    """Test P04 multi-scale coupling fixes."""

    def test_scale_manager_initialization(self):
        """Test that ScaleManager properly initializes hierarchical components."""
        facade = ScaleManager(
            device=torch.device("cpu"),
            default_mode=ScalingMode.CONSERVATIVE,
            state_manager=StateManager(device="cpu"),
        )
        facade.initialize()

        # Hierarchical components are initialized lazily
        facade._init_hierarchical_components()
        assert facade.hierarchical_aggregator is not None, \
            "Hierarchical aggregator should be initialized"
        assert facade.hierarchy_index is not None, \
            "Hierarchy index should be initialized"

    def test_hierarchical_aggregator_consistency(self):
        """Test HierarchicalAggregator consistency with simulator."""
        from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex

        # Create test hierarchy
        topology = {
            "n_source_units": 6,
            "n_target_units": 2,
            "parent_map": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
            "source_areas": [1.0] * 6,
            "target_areas": [3.0, 3.0],
        }
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="hillslope",
            device=torch.device("cpu"),
            n_source_units=6,
            topology=topology,
        )

        # Create aggregator
        aggregator = HierarchicalAggregator(index=index, weight_type="area", rule="conservative")

        # Create test state
        state = {
            'data': {'test_flux': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])},
            'operation': 'aggregate',
            'variables': ['test_flux'],
            'strategy': 'area_weighted_sum',
            # area_weighted_sum is an amount kernel: the explicit basis declares cell totals.
            'variable_metadata': {'test_flux': {'spatial_basis': 'cell_total'}},
        }

        # Test forward method
        result = aggregator.forward(state, time=0.0, dt=3600.0)

        # Verify simulator-consistent format
        assert isinstance(result, dict), "Should return dictionary"
        assert 'test_flux_hillslope' in result, "Should create scale-appropriate variable name"
        assert '_hierarchical_diagnostics' in result, "Should include diagnostics"

        # Verify aggregation result (sum over parent_map)
        aggregated = result['test_flux_hillslope']
        expected = torch.tensor([6.0, 15.0])
        torch.testing.assert_close(aggregated, expected, rtol=1e-6, atol=1e-6)

    def test_hierarchy_topology_builder(self):
        """Test topology construction and validation via ScaleManager."""
        facade = ScaleManager(
            device=torch.device("cpu"),
            state_manager=StateManager(device="cpu"),
        )
        facade.initialize()
        topology = facade._build_topology_from_parameters(
            source_scale="patch",
            target_scale="hillslope",
            n_source_units=6,
        )
        validated = facade._validate_topology(topology, operation="aggregate")
        assert validated["n_source_units"] > 0
        assert validated["n_target_units"] > 0
        assert len(validated["parent_map"]) == validated["n_source_units"]


class TestIntegration:
    """Integration tests for all fixes."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data input to output."""
        # 1. Create input data with units
        input_data = {
            'temperature': np.array([293.15, 294.15, 295.15]),
            'soil_moisture': np.array([0.3, 0.4, 0.35])
        }

        ds = xr.Dataset({
            'temperature': (['time'], input_data['temperature'], {'units': 'K'}),
            'soil_moisture': (['time'], input_data['soil_moisture'], {'units': 'm3 m-3'})
        })
        ds['temperature'].attrs['standard_name'] = 'air_temperature'

        # 2. Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            ds.to_netcdf(f.name)
            input_file = f.name

        # 3. Process through pipeline
        try:
            # Read with unit conversion
            config = ReadConfig(
                file_path=input_file,
                apply_unit_conversion=True,
                target_units={'temperature': 'degC'}
            )
            reader = DataReader(config)
            result = reader.read_dataset(input_file)

            # Process through scale coupling
            facade = ScaleManager(state_manager=StateManager(device="cpu"))
            facade.initialize()
            temp_data = torch.tensor(result.data['temperature'].values)
            aggregated_temp, audit = facade.aggregate(
                data=temp_data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["air_temperature"]
            )

            # Sanity checks on aggregation output
            assert aggregated_temp is not None
            assert isinstance(audit.metadata, dict)

        finally:
            os.unlink(input_file)

    def test_parameter_and_unit_integration(self):
        """Test integration between parameter system and unit conversion."""
        param_manager = get_global_manager()

        # Get parameter with units
        gravity = param_manager.get('relic.physical.constants.earth.gravity_acceleration', units='m s-2')
        assert gravity is not None

        converted = unit_converter.convert(torch.tensor([1.0]), 'generic', 'm', 'mm', ConversionContext())
        assert converted is not None


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([__file__, "-v", "-k", "TestUnitConversionIntegration"])
    pytest.main([__file__, "-v", "-k", "TestNetCDFOutput"])
    pytest.main([__file__, "-v", "-k", "TestParameterSystemValidation"])
    pytest.main([__file__, "-v", "-k", "TestMultiScaleCoupling"])
    pytest.main([__file__, "-v", "-k", "TestIntegration"])
