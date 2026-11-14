#!/usr/bin/env python3
"""
L2 Integration Tests for Unit Conversion System

This module tests the integration of unit conversion across the RELIC framework,
including:
- End-to-end unit conversion workflows
- Cross-module unit consistency
- Data pipeline unit conversion
- Conservation law validation with unit conversions

References:
- CF-1.8 Climate and Forecast Metadata Conventions
- RELIC Architecture Documentation
- Unit Conversion Integration Design

Author: Yuchong Long
Date: 2025-10-11
Version: 1.0.0
"""

import pytest
import torch
import numpy as np
import xarray as xr
import tempfile
import os

# Import RELIC modules
from relic.data.standards.unit_converter import UnitConverter, ConversionContext
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorMetadata,
    TransportMechanismFunctor,
)
from relic.data.io.writer import DataWriter
try:
    from relic.data.io.reader import DataReader, ReadConfig
except ImportError:
    DataReader = None
    ReadConfig = None
from relic.utils.physical_constants import get_physical_constant
try:
    from relic.parameters import get_global_manager
except ImportError:
    get_global_manager = None


class TestTransportMechanism(TransportMechanismFunctor):
    """Test concrete implementation of TransportMechanismFunctor."""

    TRANSPORT_TYPE = "diffusion"
    __test__ = False

    def __init__(self, name):
        metadata = FunctorMetadata(
            name=name,
            description="Unit conversion integration transport stub.",
            references=["CF-1.8 unit conversion integration test."],
            input_variables=["test_flux_input"],
            output_variables=["test_flux"],
        )
        super().__init__(name=name, metadata=metadata)
    
    def forward(self, state, time, dt, **kwargs):
        """Minimal implementation for testing."""
        return {'test_flux': torch.tensor([1.0])}


class TestBiologicalProcess(BiologicalProcessFunctor):
    """Test concrete implementation of BiologicalProcessFunctor."""
    __test__ = False

    def __init__(self, name):
        metadata = FunctorMetadata(
            name=name,
            description="Unit conversion integration biological stub.",
            references=["CF-1.8 unit conversion integration test."],
            input_variables=["test_gpp_input"],
            output_variables=["test_gpp"],
        )
        super().__init__(name=name, metadata=metadata)
    
    def forward(self, state, time, dt, **kwargs):
        """Minimal implementation for testing."""
        return {'test_gpp': torch.tensor([1.0])}


class TestDataPipelineUnitConversion:
    """Test unit conversion through complete data pipelines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.unit_converter = UnitConverter()
        self.data_writer = DataWriter()
        # self.data_reader = DataReader()  # Not available

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_dataset_preserves_explicit_units_and_values(self):
        """
        Test writing a dataset without implicit unit conversion.

        Scenario:
        1. Create data with mixed units
        2. Write through the explicit-metadata DataWriter path
        3. Verify units and values remain caller-owned
        """
        # Create test data with mixed units
        test_data = {
            'time': np.arange(10) * 3600.0,  # seconds since start
            'temperature': 273.15 + np.random.randn(10) * 5,  # K
            'pressure': 1013.25 + np.random.randn(10) * 50,  # hPa
            'carbon_flux': np.random.randn(10) * 1e-6,  # kg C m-2 s-1 (already CF)
            'water_flux': np.random.rand(10) * 10,  # mm day-1
            'heat_flux': np.random.randn(10) * 200,  # cal cm-2 day-1
        }

        # Create xarray dataset
        ds = xr.Dataset(
            {
                'temperature': (['time'], test_data['temperature']),
                'pressure': (['time'], test_data['pressure']),
                'carbon_flux': (['time'], test_data['carbon_flux']),
                'water_flux': (['time'], test_data['water_flux']),
                'heat_flux': (['time'], test_data['heat_flux']),
            },
            coords={'time': test_data['time']}
        )

        # Set initial units (some non-CF)
        ds['temperature'].attrs['units'] = 'K'
        ds['pressure'].attrs['units'] = 'hPa'
        ds['carbon_flux'].attrs['units'] = 'kg C m-2 s-1'
        ds['water_flux'].attrs['units'] = 'mm day-1'
        ds['heat_flux'].attrs['units'] = 'cal cm-2 day-1'
        ds['time'].attrs.update(
            {'units': 'seconds since 2000-01-01', 'calendar': 'noleap'}
        )

        # Writing must not infer or apply a unit conversion.
        output_file = os.path.join(self.temp_dir, 'test_output.nc')
        result = self.data_writer.write_dataset(
            ds,
            output_file,
            format='netcdf4',
            compression='none',
            parallel=False,
            cf_sync=False,
            add_bounds=False,
            auto_chunk=False,
            validate_output=False,
        )
        assert result.validation_passed
        assert os.path.exists(output_file)

        with xr.open_dataset(output_file, decode_times=False) as written_data:
            for name in ('temperature', 'pressure', 'carbon_flux', 'water_flux', 'heat_flux'):
                assert written_data[name].attrs['units'] == ds[name].attrs['units']
                assert np.allclose(
                    written_data[name].values,
                    test_data[name],
                    rtol=1e-6,
                    atol=1e-12,
                )

    def test_multi_variable_explicit_unit_consistency(self):
        """
        Test consistency across multiple related variables.

        Scenario: Water balance components with different units
        """
        # Create water balance components
        time_steps = 24
        test_data = {
            'time': np.arange(time_steps) * 3600.0,
            'precipitation': np.random.rand(time_steps) * 20,  # mm day-1
            'evaporation': np.random.rand(time_steps) * 8,   # mm day-1
            'runoff': np.random.rand(time_steps) * 5,        # mm day-1
            'soil_moisture': np.random.rand(time_steps) * 0.4,  # m3 m-3
        }

        ds = xr.Dataset(
            {
                'precipitation': (['time'], test_data['precipitation']),
                'evaporation': (['time'], test_data['evaporation']),
                'runoff': (['time'], test_data['runoff']),
                'soil_moisture': (['time'], test_data['soil_moisture']),
            },
            coords={'time': test_data['time']}
        )

        # Set units
        ds['precipitation'].attrs['units'] = 'mm day-1'
        ds['evaporation'].attrs['units'] = 'mm day-1'
        ds['runoff'].attrs['units'] = 'mm day-1'
        ds['soil_moisture'].attrs['units'] = 'm3 m-3'
        ds['time'].attrs.update(
            {'units': 'seconds since 2000-01-01', 'calendar': 'noleap'}
        )

        # Write without inventing standard units or changing numeric values.
        output_file = os.path.join(self.temp_dir, 'water_balance.nc')
        result = self.data_writer.write_dataset(
            ds,
            output_file,
            format='netcdf4',
            compression='none',
            parallel=False,
            cf_sync=False,
            add_bounds=False,
            auto_chunk=False,
            validate_output=False,
        )
        assert result.validation_passed
        assert os.path.exists(output_file)

        with xr.open_dataset(output_file, decode_times=False) as written_data:
            for name in ('precipitation', 'evaporation', 'runoff', 'soil_moisture'):
                assert written_data[name].attrs['units'] == ds[name].attrs['units']
                assert np.allclose(
                    written_data[name].values,
                    test_data[name],
                    rtol=1e-6,
                    atol=1e-12,
                )
                assert written_data[name].shape == (time_steps,)


class TestTransportMechanismIntegration:
    """Test transport mechanism unit conversion integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transport = TestTransportMechanism(name="test_transport")

    def test_carbon_transport_with_conversion(self):
        """
        Test carbon transport with unit conversion.

        Scenario: GPP and respiration in g C m-2 day-1 ->transport into pools
        whose variable contracts declare different units (leaf_carbon kg C m-2,
        slow_som_c g C m-2) ->each flux reaches its pool in that pool's rate.
        """
        state = {
            'leaf_carbon': torch.tensor([0.10, 0.12, 0.08]),  # kg C m-2
            'slow_som_c': torch.tensor([1000.0, 1100.0, 900.0]),  # g C m-2
            'units': {
                'gross_photosynthesis_flux': 'g C m-2 day-1',
                'auto_resp_flux': 'g C m-2 day-1',
                'het_resp_flux': 'g C m-2 day-1',
            },
        }

        # Signed fluxes into their target pools (uptake > 0, loss < 0).
        transport_outputs = {
            'gross_photosynthesis_flux': torch.tensor([20.0, 25.0, 15.0]),
            'auto_resp_flux': torch.tensor([-5.0, -6.0, -4.0]),
            'het_resp_flux': torch.tensor([-2.0, -2.5, -1.5]),
        }

        dt = 3600.0  # 1 hour in seconds
        transport_mapping = {
            'gross_photosynthesis_flux': 'leaf_carbon',
            'auto_resp_flux': 'leaf_carbon',
            'het_resp_flux': 'slow_som_c',
        }
        updates = self.transport.finalize_transport_updates(
            transport_outputs, dt, state, mapping=transport_mapping
        )

        unit_converter = UnitConverter()
        context = ConversionContext()
        factor_kg = float(
            unit_converter.convert(
                torch.tensor(1.0), 'carbon_flux', 'g C m-2 day-1', 'kg C m-2 s-1', context
            )
        )
        factor_g = float(
            unit_converter.convert(
                torch.tensor(1.0), 'carbon_flux', 'g C m-2 day-1', 'g C m-2 s-1', context
            )
        )

        torch.testing.assert_close(
            updates['leaf_carbon'],
            (
                transport_outputs['gross_photosynthesis_flux']
                + transport_outputs['auto_resp_flux']
            ) * factor_kg * dt,
            rtol=1e-6,
            atol=1e-12,
        )
        torch.testing.assert_close(
            updates['slow_som_c'],
            transport_outputs['het_resp_flux'] * factor_g * dt,
            rtol=1e-6,
            atol=1e-12,
        )

        # One hour of these fluxes changes each pool by well under 1 %.
        for pool in ('leaf_carbon', 'slow_som_c'):
            relative_change = torch.abs(updates[pool]) / state[pool]
            assert torch.all(relative_change < 0.01)

    def test_water_transport_with_conversion(self):
        """
        Test water transport with unit conversion.

        Scenario: precipitation in mm day-1 (a depth rate converted with the
        water density) and evaporation and runoff declared in kg m-2 s-1 ->
        transport into soil_water_storage (contract unit kg m-2).
        """
        water_density = get_physical_constant(
            "relic.physical.constants.water.density_liquid"
        )

        state = {
            'soil_water_storage': torch.tensor([300.0, 350.0, 250.0]),  # kg m-2
            'water_density': torch.tensor(water_density),
            'units': {
                'precipitation_flux': 'mm day-1',
                'evaporation_flux': 'kg m-2 s-1',
                'runoff_flux': 'kg m-2 s-1',
            },
        }

        precipitation_mm_day = torch.tensor([10.0, 8.0, 12.0])  # mm day-1
        evaporation_kg = torch.tensor([5.0, 4.0, 6.0]) / 86400.0  # kg m-2 s-1
        runoff_kg = torch.tensor([2.0, 1.5, 2.5]) / 86400.0  # kg m-2 s-1

        # Signed fluxes into the storage (input > 0, loss < 0).
        transport_outputs = {
            'precipitation_flux': precipitation_mm_day,
            'evaporation_flux': -evaporation_kg,
            'runoff_flux': -runoff_kg,
        }

        dt = 86400.0  # 1 day in seconds
        updates = self.transport.finalize_transport_updates(
            transport_outputs,
            dt,
            state,
            mapping={key: 'soil_water_storage' for key in transport_outputs},
        )

        unit_converter = UnitConverter()
        context = ConversionContext(water_density=water_density)
        factor = unit_converter.convert(
            torch.tensor(1.0),
            'precipitation_flux',
            'mm day-1',
            'kg m-2 s-1',
            context,
        )
        factor = float(factor.detach().cpu().item())

        # Storage change = (P * f - E - R) * dt with f the depth-rate factor.
        expected_net = (
            precipitation_mm_day * factor - evaporation_kg - runoff_kg
        ) * dt

        torch.testing.assert_close(
            updates['soil_water_storage'],
            expected_net,
            rtol=1e-6,
            atol=1e-10,
        )


class TestDataUnitConversions:
    """Test data unit conversion workflows without DataReader dependency."""

    def test_temperature_conversion_workflow(self):
        """Test temperature unit conversion workflow."""
        # Import unit converter directly
        from relic.data.standards.unit_converter import unit_converter, ConversionContext

        # Create test data in Celsius
        temp_celsius = torch.tensor([15.0, 20.0, 25.0, 30.0])

        # Convert to Kelvin using unit converter
        context = ConversionContext()
        temp_kelvin = unit_converter.convert(
            temp_celsius, 'air_temperature', 'celsius', 'K', context
        )

        # Verify conversion: 15degC + 273.15 = 288.15K
        expected = torch.tensor([288.15, 293.15, 298.15, 303.15])
        assert torch.allclose(temp_kelvin, expected, rtol=1e-3)

    def test_pressure_conversion_workflow(self):
        """Test pressure unit conversion workflow."""
        from relic.data.standards.unit_converter import unit_converter, ConversionContext

        # Create test data in hPa
        pressure_hpa = torch.tensor([1000.0, 1013.25, 1020.0])

        # Convert to Pa using unit converter
        context = ConversionContext()
        pressure_pa = unit_converter.convert(
            pressure_hpa, 'air_pressure', 'hPa', 'Pa', context
        )

        # Verify conversion: 1000 hPa * 100 = 100000 Pa
        expected = torch.tensor([100000.0, 101325.0, 102000.0])
        assert torch.allclose(pressure_pa, expected, rtol=1e-3)

    def test_soil_moisture_conversion_workflow(self):
        """Test soil moisture unit conversion workflow."""
        from relic.data.standards.unit_converter import unit_converter, ConversionContext

        # Create test data: volumetric soil moisture
        moisture_vol = torch.tensor([0.2, 0.3, 0.4, 0.5])  # m3/m3

        # Convert to mass-based with soil depth context
        context = ConversionContext(soil_depth=1.0, water_density=1000.0)
        moisture_mass = unit_converter.convert(
            moisture_vol, 'soil_moisture_content', 'm3 m-3', 'kg m-2', context
        )

        # Verify conversion: 0.2 * 1.0 * 1000 = 200 kg m-2
        expected = torch.tensor([200.0, 300.0, 400.0, 500.0])
        assert torch.allclose(moisture_mass, expected, rtol=1e-3)


class TestBiologicalProcessIntegration:
    """Test biological process unit conversion integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bio_process = TestBiologicalProcess(name="test_bio_process")

    def test_photosynthesis_coupling_units(self):
        """
        Test photosynthesis coupling with consistent units.

        Scenario: Environmental inputs ->photosynthesis ->carbon output ->transport
        """
        # Molar carbon fluxes (umol m-2 s-1) reach leaf_carbon (contract unit
        # kg C m-2) through the UnitConverter carbon_flux molar-to-mass rule,
        # selected by the flux variables carbon_flux and carbon_respiration_flux.
        state = {
            'leaf_carbon': torch.tensor([0.10, 0.12, 0.08]),  # kg C m-2
            'units': {
                'carbon_flux': 'umol m-2 s-1',
                'carbon_respiration_flux': 'umol m-2 s-1',
            },
        }

        # Signed photosynthesis outputs (uptake > 0, respiration loss < 0).
        photosynthesis_outputs = {
            'carbon_flux': torch.tensor([15.0, 20.0, 10.0]),
            'carbon_respiration_flux': torch.tensor([-3.0, -4.0, -2.0]),
        }

        dt = 3600.0
        transport = TestTransportMechanism(name="coupling_transport")
        updates = transport.finalize_transport_updates(
            photosynthesis_outputs,
            dt,
            state,
            mapping={key: 'leaf_carbon' for key in photosynthesis_outputs},
        )

        unit_converter = UnitConverter()
        context = ConversionContext()
        factor = unit_converter.convert(
            torch.tensor(1.0),
            'carbon_flux',
            'umol m-2 s-1',
            'kg C m-2 s-1',
            context,
        )
        factor = float(factor.detach().cpu().item())

        torch.testing.assert_close(
            updates['leaf_carbon'],
            (
                photosynthesis_outputs['carbon_flux']
                + photosynthesis_outputs['carbon_respiration_flux']
            ) * factor * dt,
            rtol=1e-6,
            atol=1e-12,
        )


class TestConservationLawValidation:
    """Test conservation law validation with unit conversions."""

    def test_carbon_mass_conservation(self):
        """
        Test carbon mass conservation across unit conversions.

        Scenario: Multiple carbon fluxes with different units ->conservation check
        """
        # Initial carbon pools
        initial_carbon = {
            'leaf': torch.tensor([100.0, 120.0, 80.0]),    # kg C m-2
            'stem': torch.tensor([200.0, 180.0, 150.0]),   # kg C m-2
            'root': torch.tensor([50.0, 60.0, 40.0]),     # kg C m-2
            'soil': torch.tensor([1000.0, 1100.0, 900.0]) # kg C m-2
        }

        # Carbon fluxes in different units
        fluxes = {
            'gpp': torch.tensor([20.0, 25.0, 15.0]),        # g C m-2 day-1
            'auto_resp': torch.tensor([5.0, 6.0, 4.0]),     # g C m-2 day-1
            'het_resp': torch.tensor([2.0, 2.5, 1.5]),      # g C m-2 day-1
            'litterfall': torch.tensor([1.0, 1.2, 0.8])     # g C m-2 day-1
        }

        # Convert all fluxes to kg C m-2 s-1
        conversion_factor = 1.0 / (1000.0 * 86400.0)
        converted_fluxes = {k: v * conversion_factor for k, v in fluxes.items()}

        # Calculate net change
        net_change = (
            converted_fluxes['gpp'] -
            converted_fluxes['auto_resp'] -
            converted_fluxes['het_resp']
        )

        # Update carbon pools
        dt = 86400.0  # 1 day
        updated_carbon = {}
        for pool, initial_value in initial_carbon.items():
            if pool == 'soil':
                # Soil gets litterfall, loses heterotrophic respiration
                soil_change = converted_fluxes['litterfall'] - converted_fluxes['het_resp']
                updated_carbon[pool] = initial_value + soil_change * dt
            elif pool == 'leaf':
                # Leaf gets GPP, loses auto respiration and litterfall
                leaf_change = (
                    converted_fluxes['gpp']
                    - converted_fluxes['auto_resp']
                    - converted_fluxes['litterfall']
                )
                updated_carbon[pool] = initial_value + leaf_change * dt
            else:
                updated_carbon[pool] = initial_value  # No change for other pools in this simple test

        # Check mass conservation
        total_initial = sum(initial_carbon.values())
        total_final = sum(updated_carbon.values())

        # The difference should equal the net carbon gain over time
        expected_difference = net_change * dt
        actual_difference = total_final - total_initial

        torch.testing.assert_close(actual_difference, expected_difference, rtol=1e-6, atol=1e-10)

    def test_water_mass_conservation(self):
        """
        Test water mass conservation across unit conversions.

        Scenario: Precipitation, evaporation, runoff in different units ->conservation check
        """
        # Initial water states
        initial_water = {
            'canopy': torch.tensor([2.0, 1.5, 2.5]),       # kg m-2
            'snow': torch.tensor([10.0, 8.0, 12.0]),       # kg m-2
            'soil_surface': torch.tensor([20.0, 25.0, 15.0]), # kg m-2
            'soil_root': torch.tensor([80.0, 85.0, 75.0])     # kg m-2
        }

        # Water fluxes in mm day-1
        fluxes = {
            'precipitation': torch.tensor([10.0, 8.0, 12.0]),  # mm day-1
            'snowfall': torch.tensor([2.0, 1.5, 2.5]),        # mm day-1
            'evaporation': torch.tensor([4.0, 3.0, 5.0]),      # mm day-1
            'transpiration': torch.tensor([3.0, 2.5, 3.5]),     # mm day-1
            'runoff': torch.tensor([1.0, 0.8, 1.2])           # mm day-1
        }

        # Convert mm day-1 to kg m-2 s-1
        conversion_factor = 1.0 / 86400.0
        converted_fluxes = {k: v * conversion_factor for k, v in fluxes.items()}

        # Update water pools
        dt = 86400.0  # 1 day
        updated_water = {}

        # Canopy loses evaporation (no precipitation interception in this simple test)
        canopy_change = -converted_fluxes['evaporation']
        updated_water['canopy'] = initial_water['canopy'] + canopy_change * dt

        # Snow gets snowfall
        updated_water['snow'] = initial_water['snow'] + converted_fluxes['snowfall'] * dt

        # Soil surface gets precipitation, loses runoff
        surface_change = converted_fluxes['precipitation'] - converted_fluxes['runoff']
        updated_water['soil_surface'] = initial_water['soil_surface'] + surface_change * dt

        # Soil root gets transpiration loss
        root_change = -converted_fluxes['transpiration']
        updated_water['soil_root'] = initial_water['soil_root'] + root_change * dt

        # Check mass conservation
        total_initial = sum(initial_water.values())
        total_final = sum(updated_water.values())

        # Net water input should equal total change
        net_input = converted_fluxes['precipitation'] + converted_fluxes['snowfall']
        net_output = converted_fluxes['evaporation'] + converted_fluxes['transpiration'] + converted_fluxes['runoff']
        net_change = (net_input - net_output) * dt

        actual_change = total_final - total_initial
        torch.testing.assert_close(actual_change, net_change, rtol=1e-6, atol=1e-10)


class TestParameterManagerIntegration:
    """Test parameter manager integration with unit conversions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.param_manager = get_global_manager()
        self.converter = UnitConverter()

    def test_parameter_unit_validation(self):
        """
        Test parameter unit validation and conversion.

        Scenario: Load parameters with various units ->validate consistency
        """
        gravity = get_physical_constant(
            "relic.physical.constants.earth.gravity_acceleration"
        )
        gas_constant = get_physical_constant(
            "relic.physical.constants.universal.gas_constant"
        )
        ref_temp = get_physical_constant(
            "relic.physical.constants.temperature.reference_temperature"
        )

        assert abs(gravity - 9.80665) < 0.01
        assert abs(gas_constant - 8.314) < 0.01
        assert 298.0 < ref_temp < 299.0

    def test_parameter_unit_conversion_request(self):
        """
        Test parameter unit conversion requests.

        Scenario: Request parameter in different units ->automatic conversion
        """
        ref_temp_k = get_physical_constant(
            "relic.physical.constants.temperature.reference_temperature"
        )
        ref_temp_c = self.converter.convert(
            ref_temp_k,
            "air_temperature",
            "K",
            "degC",
            ConversionContext(),
        )
        restored_k = self.converter.convert(
            ref_temp_c,
            "air_temperature",
            "degC",
            "K",
            ConversionContext(),
        )

        assert float(ref_temp_c) == pytest.approx(25.0, abs=1e-5)
        assert float(restored_k) == pytest.approx(ref_temp_k, abs=1e-5)


class TestErrorHandlingIntegration:
    """Test error handling in integrated unit conversion scenarios."""

    def test_mixed_unit_data_validation(self):
        """
        Test validation of data with mixed/inconsistent units.

        Scenario: Dataset with conflicting units ->graceful handling
        """
        # Create dataset with conflicting units
        time = np.arange(10)
        temp1 = 273.15 + np.random.randn(10) * 5
        temp2 = 20.0 + np.random.randn(10) * 2  # Celsius

        ds = xr.Dataset({
            'temp1': (['time'], temp1),
            'temp2': (['time'], temp2),
        }, coords={'time': time})

        # Set conflicting units
        ds['temp1'].attrs['units'] = 'K'
        ds['temp2'].attrs['units'] = 'C'

        # Should handle gracefully (would need actual writer to test fully)
        # For now, just verify the conflict exists
        assert ds['temp1'].attrs['units'] != ds['temp2'].attrs['units']

    def test_unknown_declared_flux_unit_raises(self):
        """
        Scenario: a flux declares a unit label the dimensionality owner cannot
        parse ->the transport conversion raises instead of passing the flux on.
        """
        transport = TestTransportMechanism(name="test_transport")

        with pytest.raises(ValueError, match="Unknown unit"):
            transport._apply_flux_unit_conversion(
                torch.tensor([1.0, 2.0, 3.0]),
                "tracer_flux",
                "fast_som_c",
                state={"units": {"tracer_flux": "furlong fortnight-1"}},
            )


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])
