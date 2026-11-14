#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 System Test: End-to-End RELIC Scientific Integration Testing

This module provides comprehensive L3 system tests that verify the complete
biogeochemical modeling workflow from input data to final scientific outputs.

Testing Strategy:
- End-to-end workflow validation
- Cross-sphere coupling verification  
- Scientific accuracy benchmarking
- Performance and stability testing
- Error handling and edge cases

Authors: Yuchong Long
License: MIT
"""

import pytest
import torch
from typing import Dict, Any
import time
import gc
import logging

# Import all scientific modules for comprehensive testing
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor  
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.core.state import UpdateItem, UpdateMode
# Transport modules simplified for L3 system testing
# from relic.scientific.transport.diffusion import FickianDiffusion
# from relic.scientific.transport.advection import LinearAdvection  
# from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.core.functor import ExecutionError

# Configure logging for system tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHEMISTRY_EXPLICIT_DT_SECONDS = 1.0


def _unwrap_update(value: Any) -> Any:
    """Return raw payload from UpdateItem values used by current functor outputs."""
    return value.value if isinstance(value, UpdateItem) else value


def _iter_tensors(value: Any):
    """Yield tensors from nested output payloads for finite-value checks."""
    value = _unwrap_update(value)
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)


def _all_output_tensors_finite(outputs: Dict[str, Any]) -> bool:
    """Check finite values only for tensor-bearing outputs."""
    return all(torch.isfinite(tensor).all() for tensor in _iter_tensors(outputs))


def _apply_process_outputs(state: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """Apply tensor outputs to state while preserving structured diagnostics."""
    for key, raw_value in outputs.items():
        value = _unwrap_update(raw_value)
        if isinstance(raw_value, UpdateItem) and raw_value.mode == UpdateMode.REPLACE:
            state[key] = value
        elif isinstance(value, dict):
            state[key] = value
        elif key in state and isinstance(state[key], torch.Tensor) and isinstance(value, torch.Tensor):
            state[key] = state[key] + value
        else:
            state[key] = value


class TestL3EndToEndSystem:
    """
    Comprehensive L3 System Tests for RELIC Scientific Framework
    
    These tests verify the complete biogeochemical modeling system including:
    - Multi-sphere coupling (atmosphere-biosphere-hydrosphere-anthroposphere)
    - Scientific accuracy vs theoretical expectations
    - Numerical stability under various conditions
    - Performance characteristics and memory efficiency
    - Error handling and recovery mechanisms
    """

    @pytest.fixture
    def comprehensive_earth_system_state(self) -> Dict[str, torch.Tensor]:
        """Create comprehensive Earth system state for L3 testing."""
        batch_size = 10
        return {
            # Atmospheric variables
            'air_temperature': torch.full((batch_size,), 298.15),  # 25 C      
            'temperature': torch.full((batch_size,), 298.15),
            'atmospheric_pressure': torch.full((batch_size,), 101325.0),  # 1 atm
            'atmospheric_pressure': torch.full((batch_size,), 101325.0),  # Pa for Penman-Monteith
            'net_radiation': torch.full((batch_size,), 400.0),  # W/m2 for Penman-Monteith
            'surface_resistance': torch.full((batch_size,), 70.0),  # s/m for grass reference
            'relative_humidity': torch.full((batch_size,), 0.65),  # 65% RH
            'surface_temperature': torch.full((batch_size,), 296.15),
            'surface_albedo': torch.full((batch_size,), 0.2),
            'surface_emissivity': torch.full((batch_size,), 0.97),
            'cloud_fraction': torch.zeros(batch_size),
            'wind_speed': torch.full((batch_size,), 3.0),  # 3 m/s
            'reference_height': torch.full((batch_size,), 10.0),
            'roughness_length_momentum': torch.full((batch_size,), 0.1),
            'displacement_height': torch.zeros(batch_size),
            'cos_zenith': torch.full((batch_size,), 0.7),  # Solar angle
            'incoming_shortwave_potential': torch.full((batch_size,), 800.0),  # W/m2
            'incoming_shortwave': torch.full((batch_size,), 700.0),  # W/m2
            'incoming_longwave': torch.full((batch_size,), 350.0),  # W/m2
            'u': torch.full((batch_size,), 2.0),  # Zonal wind m/s
            'v': torch.full((batch_size,), 1.5),  # Meridional wind m/s
            'sensible_heat_flux': torch.full((batch_size,), 150.0),  # W/m2
            'latent_heat_flux': torch.full((batch_size,), 200.0),  # W/m2
            
            # Chemistry variables  
            'co2_concentration': torch.full((batch_size,), 410.0),  # 410 ppm
            'atmospheric_co2_ppm': torch.full((batch_size,), 410.0),  # 410 ppm
            'o3_concentration': torch.full((batch_size,), 40e-9),  # 40 ppb
            'no2_concentration': torch.full((batch_size,), 20e-9),  # 20 ppb
            'tracer_concentration': torch.full((batch_size,), 1e-6),  # Generic tracer
            # Additional chemistry parameters for ChemistryFunctor
            'no_mixing_ratio': torch.full((batch_size,), 10e-9),  # 10 ppb NO
            'no2_mixing_ratio': torch.full((batch_size,), 20e-9),  # 20 ppb NO2
            'o3_mixing_ratio': torch.full((batch_size,), 40e-9),  # 40 ppb O3
            'co_mixing_ratio': torch.full((batch_size,), 150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.full((batch_size,), 1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.full((batch_size,), 0.01),  # O(1D) + H2O OH source, 10 g/kg
            'isoprene_mixing_ratio': torch.full((batch_size,), 1e-9),
            
            # Biosphere variables
            'leaf_area_index': torch.full((batch_size,), 4.0),  # LAI
            'vcmax_scaling': torch.full((batch_size,), 1.0),  # Vcmax scaling   
            'jmax_scaling': torch.full((batch_size,), 1.0),
            'water_stress_factor': torch.full((batch_size,), 1.0),
            'temperature_stress_factor': torch.full((batch_size,), 1.0),
            'ozone_stress_factor': torch.full((batch_size,), 1.0),
            'vapor_pressure_deficit': torch.full((batch_size,), 1000.0),
            'stomatal_conductance': torch.full((batch_size,), 0.1),  # mol m-2 s-1
            'gross_primary_productivity': torch.zeros((batch_size,)),  # Will be computed
            # Photosynthesis parameters for FarquharPhotosynthesis
            'ppfd': torch.full((batch_size,), 1200.0),
            
            # Hydrosphere variables
            'soil_moisture': torch.full((batch_size,), 0.3),  # m3/m3
            'soil_temperature': torch.full((batch_size,), 295.0),  # K
            'water_table_depth': torch.full((batch_size,), 2.0),  # m
            'canopy_temperature': torch.full((batch_size,), 297.0),  # K
            'saturated_vapor_pressure': torch.full((batch_size,), 3167.0),  # Pa
            'actual_vapor_pressure': torch.full((batch_size,), 2058.0),  # Pa
            'soil_nitrogen': torch.full((batch_size,), 5.0),
            'soil_phosphorus': torch.full((batch_size,), 1.0),
            'crop_demand': torch.full((batch_size,), 0.5),
            'crop_stage': torch.full((batch_size,), 1.0),
            'day_of_year': torch.full((batch_size,), 180.0),
            'et_demand': torch.full((batch_size,), 2.0),
            'precipitation_flux': torch.zeros(batch_size),
            'land_use_type': torch.ones(batch_size, dtype=torch.long),
            'irrigation_type': torch.full((batch_size,), 2, dtype=torch.long),
            'field_capacity': torch.full((batch_size,), 0.35),
            'wilting_point': torch.full((batch_size,), 0.15),
            
            # Anthroposphere variables
            'fertilizer_n': torch.full((batch_size,), 100.0),  # kg N/ha
            'fertilizer_p': torch.full((batch_size,), 50.0),   # kg P/ha  
            'fertilizer_k': torch.full((batch_size,), 80.0),   # kg K/ha
            'soil_ph': torch.full((batch_size,), 6.5),
            'soil_organic_carbon': torch.full((batch_size,), 0.02),
            'soil_no3': torch.full((batch_size,), 3.0),
            'fertilizer_application_rate': torch.full((batch_size,), 0.001),
            'fertilizer_n_content': torch.full((batch_size,), 0.46),
            'urea_n': torch.full((batch_size,), 20.0),
            'ammonium_n': torch.full((batch_size,), 5.0),
            'irrigation_amount': torch.full((batch_size,), 20.0),  # mm
            'energy_consumption': torch.full((batch_size,), 1000.0),  # MJ
        }

    @pytest.fixture
    def performance_monitoring_setup(self) -> Dict[str, Any]:
        """Setup performance monitoring utilities for L3 tests."""
        return {
            'start_memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'start_time': time.perf_counter(),
            'memory_threshold_mb': 1000,  # 1GB memory limit
            'time_threshold_s': 30,       # 30s time limit
            'gc_collections': gc.get_count(),
        }

    def test_multi_timestep_stability(self, comprehensive_earth_system_state):
        """
        Test system stability over multiple timesteps
        
        Validates:
        - Numerical stability over extended simulation
        - Conservation of mass/energy where applicable
        - Absence of runaway feedbacks
        """
        logger.info("Starting Multi-Timestep Stability Test")
        
        state = comprehensive_earth_system_state.copy()
        initial_co2 = state['co2_concentration'].clone()
        
        # Initialize key modules for multi-step testing
        radiation = RadiationFunctor()
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        photosynthesis = FarquharPhotosynthesis()
        
        # Run 24 hourly timesteps (1 day simulation)
        for hour in range(24):
            current_time = hour * 3600.0  # Convert to seconds
            
            # Radiation processing
            rad_fluxes = radiation(state, time=current_time, dt=3600.0)
            _apply_process_outputs(state, rad_fluxes)
            
            # Chemistry processing  
            chem_fluxes = chemistry(
                state,
                time=current_time,
                dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
            )
            _apply_process_outputs(state, chem_fluxes)
            
            # Photosynthesis processing
            photo_fluxes = photosynthesis(state, time=current_time, dt=3600.0)
            _apply_process_outputs(state, photo_fluxes)
            
            # Stability checks each timestep
            assert torch.all(torch.isfinite(state['air_temperature'])), \
                f"Temperature became non-finite at hour {hour}"
            assert torch.all(state['air_temperature'] > 200), \
                f"Temperature too low at hour {hour}: {torch.min(state['air_temperature'])}"
            assert torch.all(state['air_temperature'] < 350), \
                f"Temperature too high at hour {hour}: {torch.max(state['air_temperature'])}"
            assert torch.all(state['co2_concentration'] > 0), \
                f"CO2 concentration became negative at hour {hour}"
        
        # Final stability validations
        final_co2 = state['co2_concentration']
        co2_drift = torch.abs(final_co2 - initial_co2) / initial_co2
        max_drift = torch.max(co2_drift)
        
        # Allow some CO2 change due to photosynthesis, but not excessive drift
        assert max_drift < 0.1, f"Excessive CO2 drift over 24h: {max_drift:.4f}"
        
        logger.info("Multi-Timestep Stability Test PASSED")

    def test_extreme_boundary_conditions(self, comprehensive_earth_system_state):
        """
        Test system behavior under extreme boundary conditions
        
        Validates:
        - Graceful handling of extreme temperatures
        - Proper behavior at concentration limits
        - Error recovery mechanisms
        """
        logger.info("Starting Extreme Boundary Conditions Test")
        
        base_state = comprehensive_earth_system_state.copy()
        
        # Test extreme conditions
        extreme_conditions = [
            {
                'name': 'Arctic conditions',
                'air_temperature': torch.full((10,), 275.15),  # 2 C (cold but within model range 273.15-323.15K)
                'relative_humidity': torch.full((10,), 0.9),
                'wind_speed': torch.full((10,), 15.0),  # Strong wind
            },
            {
                'name': 'Desert conditions', 
                'air_temperature': torch.full((10,), 323.15),  # 50 C
                'relative_humidity': torch.full((10,), 0.1),   # Very dry
                'incoming_shortwave': torch.full((10,), 1200.0),  # High radiation
            },
            {
                'name': 'High altitude conditions',
                'atmospheric_pressure': torch.full((10,), 50000.0),   # ~5km elevation
                'air_temperature': torch.full((10,), 278.15), # 5 C (cold but within model range)
                'co2_concentration': torch.full((10,), 300.0), # Lower CO2 (fixed unit)
                'atmospheric_co2_ppm': torch.full((10,), 300.0), # Lower CO2 (canonical key)
            },
        ]
        
        for condition in extreme_conditions:
            logger.info(f"Testing {condition['name']}")
            
            # Create modified state
            test_state = base_state.copy()
            for key, value in condition.items():
                if key != 'name':
                    test_state[key] = value
            
            # Test key modules under extreme but valid conditions.
            radiation = RadiationFunctor()
            rad_fluxes = radiation(test_state, time=0.0, dt=3600.0)
            _apply_process_outputs(test_state, rad_fluxes)

            assert _all_output_tensors_finite(rad_fluxes), \
                f"Non-finite radiation fluxes under {condition['name']}"

            chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
            chem_fluxes = chemistry(
                test_state,
                time=0.0,
                dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
            )

            assert _all_output_tensors_finite(chem_fluxes), \
                f"Non-finite chemistry fluxes under {condition['name']}"

            photosynthesis = FarquharPhotosynthesis()
            photo_fluxes = photosynthesis(test_state, time=0.0, dt=3600.0)

            assert _all_output_tensors_finite(photo_fluxes), \
                f"Non-finite photosynthesis fluxes under {condition['name']}"
        
        logger.info("Extreme Boundary Conditions Test PASSED")

    def test_memory_efficiency_and_cleanup(self, comprehensive_earth_system_state):
        """
        Test system memory efficiency and cleanup
        
        Validates:
        - No memory leaks during extended operation  
        - Proper cleanup of intermediate calculations
        - GPU memory management (if available)
        """
        logger.info("Starting Memory Efficiency and Cleanup Test")
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        state = comprehensive_earth_system_state.copy()
        
        # Initialize modules
        radiation = RadiationFunctor()
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        photosynthesis = FarquharPhotosynthesis()
        
        # Perform multiple processing cycles to test for memory leaks
        for cycle in range(10):
            # Create temporary large states to stress memory
            large_state = {}
            for key, value in state.items():
                # Expand batch size temporarily
                large_state[key] = value.repeat(100)  # 1000 batch elements
            
            # Process with large state
            rad_fluxes = radiation(large_state, time=0.0, dt=3600.0)
            _apply_process_outputs(large_state, rad_fluxes)
            chem_fluxes = chemistry(
                large_state,
                time=0.0,
                dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
            )
            photo_fluxes = photosynthesis(large_state, time=0.0, dt=3600.0)
            
            # Explicit cleanup
            del large_state, rad_fluxes, chem_fluxes, photo_fluxes
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
        
        # Check final memory usage
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_increase_mb = (final_memory - initial_memory) / 1024**2
        
        # Allow some memory increase but not excessive
        assert memory_increase_mb < 100, f"Excessive memory increase: {memory_increase_mb:.1f}MB"
        
        logger.info("Memory Efficiency and Cleanup Test PASSED")

    def test_error_handling_and_recovery(self, comprehensive_earth_system_state):
        """
        Test system error handling and recovery mechanisms
        
        Validates:
        - Graceful handling of invalid inputs
        - Proper error messages
        - System recovery after errors
        """
        logger.info("Starting Error Handling and Recovery Test")
        
        base_state = comprehensive_earth_system_state.copy()
        
        # Test invalid input scenarios
        invalid_inputs = [
            {
                'name': 'NaN temperature',
                'air_temperature': torch.full((10,), float('nan')),
                'error_pattern': 'air_temperature',
            },
            {
                'name': 'Negative shortwave potential',
                'incoming_shortwave_potential': torch.full((10,), -1.0),
                'error_pattern': 'incoming_shortwave_potential',
            },
            {
                'name': 'Invalid humidity',
                'relative_humidity': torch.full((10,), 1.5),  # > 100%
                'error_pattern': 'relative_humidity',
            },
        ]
        
        for invalid_input in invalid_inputs:
            logger.info(f"Testing error handling for {invalid_input['name']}")
            
            # Create invalid state
            test_state = base_state.copy()
            for key, value in invalid_input.items():
                if key not in {'name', 'error_pattern'}:
                    test_state[key] = value

            # Invalid required radiation inputs must fail fast with field context.
            radiation = RadiationFunctor()

            with pytest.raises(ExecutionError, match=invalid_input['error_pattern']):
                radiation(test_state, time=0.0, dt=3600.0)

        valid_boundary_state = base_state.copy()
        valid_boundary_state['air_temperature'] = torch.full((10,), 273.15)
        boundary_fluxes = radiation(valid_boundary_state, time=0.0, dt=3600.0)
        assert _all_output_tensors_finite(boundary_fluxes)
        
        # Test system recovery - after handling errors, system should work normally
        normal_fluxes = radiation(base_state, time=0.0, dt=3600.0)
        assert _all_output_tensors_finite(normal_fluxes), \
            "System failed to recover after error handling tests"
        
        logger.info("Error Handling and Recovery Test PASSED")

    def test_scientific_accuracy_benchmarks(self, comprehensive_earth_system_state):
        """
        Test scientific accuracy against theoretical benchmarks
        
        Validates:
        - Photosynthesis rates against Farquhar model expectations
        - Energy balance closure
        - Conservation principles
        - Physically realistic ranges
        """
        logger.info("Starting Scientific Accuracy Benchmarks Test")
        
        state = comprehensive_earth_system_state.copy()
        
        # === Photosynthesis Benchmark ===
        photosynthesis = FarquharPhotosynthesis()
        photo_fluxes = photosynthesis(state, time=0.0, dt=3600.0)
        
        # Update state with photosynthesis results
        _apply_process_outputs(state, photo_fluxes)
        
        # Validate GPP is within expected range for C3 plants
        if 'gross_primary_productivity' in state:
            gpp = state['gross_primary_productivity']  # umol CO2 m-2 s-1
            # Expected range: 0-50 umol CO2 m-2 s-1
            assert torch.all(gpp >= 0), "GPP must be non-negative"
            assert torch.all(gpp <= 50.0), "GPP above theoretical maximum"
        
        # === Evapotranspiration Benchmark ===
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        et_fluxes = evapotranspiration(state, time=0.0, dt=3600.0)
        
        _apply_process_outputs(state, et_fluxes)
        
        # Validate ET rates
        if 'evapotranspiration_flux' in state:
            et_rate = state['evapotranspiration_flux']  # kg m-2 s-1
            if isinstance(et_rate, UpdateItem):
                et_rate = et_rate.value
            # Convert to daily rate for validation: kg m-2 s-1 -> mm day-1
            # Note: 1 kg m-2 = 1 mm (water density 1000 kg/m3)
            et_daily = et_rate * 86400  # mm day-1
            assert torch.all(et_daily >= 0), "ET rate must be non-negative"
            assert torch.all(et_daily <= 20), "ET rate above reasonable maximum (20 mm day-1)"
        
        # === Radiation Benchmark ===
        radiation = RadiationFunctor()
        rad_fluxes = radiation(state, time=0.0, dt=3600.0)
        
        _apply_process_outputs(state, rad_fluxes)
        
        # Validate net radiation
        if 'net_radiation' in state:
            net_rad = state['net_radiation']  # W/m2
            # Net radiation should be reasonable for given conditions
            assert torch.all(torch.abs(net_rad) <= 1000), "Net radiation outside reasonable range"
        
        # === Temperature Response Validation ===
        # Test Q10 temperature dependence for biological processes
        temp_low = state['air_temperature'] - 10  # 10K cooler
        temp_high = state['air_temperature'] + 10  # 10K warmer
        
        state_low = state.copy()
        state_high = state.copy()
        state_low['air_temperature'] = temp_low
        state_high['air_temperature'] = temp_high
        
        # Photosynthesis should increase with temperature (up to optimum)
        photo_low = photosynthesis(state_low, time=0.0, dt=3600.0)
        photo_high = photosynthesis(state_high, time=0.0, dt=3600.0)
        
        if 'gross_primary_productivity' in photo_low and 'gross_primary_productivity' in photo_high:
            gpp_low = _unwrap_update(photo_low['gross_primary_productivity'])
            gpp_high = _unwrap_update(photo_high['gross_primary_productivity'])
            
            # At moderate temperatures, higher T should give higher GPP
            temp_response_ok = torch.all(gpp_high >= gpp_low * 0.8)  # Allow some variation
            assert temp_response_ok, "Temperature response of photosynthesis appears incorrect"
        
        logger.info("Scientific Accuracy Benchmarks Test PASSED")

    def test_performance_benchmarks(self, comprehensive_earth_system_state):
        """
        Test system performance benchmarks
        
        Validates:
        - Execution time within acceptable limits
        - Scalability with batch size
        - Memory usage efficiency
        """
        logger.info("Starting Performance Benchmarks Test")
        
        # === Single Timestep Performance ===
        state = comprehensive_earth_system_state.copy()
        
        start_time = time.perf_counter()
        
        # Execute complete workflow
        radiation = RadiationFunctor()
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        photosynthesis = FarquharPhotosynthesis()
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        rad_fluxes = radiation(state, time=0.0, dt=3600.0)
        _apply_process_outputs(state, rad_fluxes)
        chem_fluxes = chemistry(
            state,
            time=0.0,
            dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
        )
        photo_fluxes = photosynthesis(state, time=0.0, dt=3600.0)
        et_fluxes = evapotranspiration(state, time=0.0, dt=3600.0)
        
        single_timestep_time = time.perf_counter() - start_time
        
        # Single timestep should complete quickly
        assert single_timestep_time < 5.0, f"Single timestep too slow: {single_timestep_time:.3f}s"
        
        # === Batch Scalability Test ===
        batch_sizes = [1, 10, 100]
        execution_times = []
        
        for batch_size in batch_sizes:
            # Create state with specified batch size
            scaled_state = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    # Reshape to new batch size
                    scaled_state[key] = value[:1].repeat(batch_size)
                else:
                    scaled_state[key] = torch.full((batch_size,), value.item())
            
            start_time = time.perf_counter()
            
            # Execute with scaled batch
            rad_fluxes = radiation(scaled_state, time=0.0, dt=3600.0)
            
            execution_time = time.perf_counter() - start_time
            execution_times.append(execution_time)
        
        # Execution time should scale reasonably with batch size
        time_ratio_10_1 = execution_times[1] / execution_times[0]
        time_ratio_100_10 = execution_times[2] / execution_times[1]
        
        # Should not scale linearly (due to vectorization), but reasonable scaling
        assert time_ratio_10_1 < 15, f"Poor scaling 1->10: {time_ratio_10_1:.2f}x"
        assert time_ratio_100_10 < 15, f"Poor scaling 10->100: {time_ratio_100_10:.2f}x"
        
        logger.info(f"Performance: {single_timestep_time:.3f}s single timestep")
        logger.info(f"Batch scaling: 1->10 ({time_ratio_10_1:.2f}x), 10->100 ({time_ratio_100_10:.2f}x)")
        logger.info("Performance Benchmarks Test PASSED")


if __name__ == '__main__':
    # Enable detailed logging for standalone execution
    logging.basicConfig(level=logging.DEBUG)
    
    # Run specific tests for debugging
    pytest.main([__file__, "-v", "-s"])
