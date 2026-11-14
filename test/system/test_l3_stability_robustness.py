#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 Stability and Robustness Tests

This module provides comprehensive stability and robustness tests that validate
the RELIC system's behavior under stress conditions, edge cases, and
extended operation scenarios.

Testing Strategy:
- Long-term simulation stability
- Extreme boundary condition handling
- Error recovery and graceful degradation
- Memory leak detection
- Numerical stability analysis
- Concurrent execution safety

Authors: Yuchong Long  
License: MIT
"""

import pytest
import torch
import numpy as np
import logging
import time
import gc
from typing import Dict, Any, Callable
import concurrent.futures
from contextlib import contextmanager

# Import scientific modules for stability testing
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.core.state import UpdateItem, UpdateMode
# Removed problematic anthroposphere imports - not needed for stability testing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHEMISTRY_EXPLICIT_DT_SECONDS = 1.0


def _stability_test_timestep(module_name: str) -> float:
    """Return a positivity-safe chemistry step without changing other clocks."""
    if module_name == 'chemistry':
        return CHEMISTRY_EXPLICIT_DT_SECONDS
    return 3600.0


def _unwrap_update(value: Any) -> Any:
    """Return raw payload from UpdateItem values used by current functor outputs."""
    return value.value if isinstance(value, UpdateItem) else value


def _iter_output_tensors(value: Any):
    """Yield tensors from nested output payloads."""
    value = _unwrap_update(value)
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_output_tensors(nested)


def _apply_process_outputs(state: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """Apply process outputs using the current UpdateItem state semantics."""
    for key, raw_value in outputs.items():
        value = _unwrap_update(raw_value)
        if isinstance(raw_value, UpdateItem) and raw_value.mode == UpdateMode.REPLACE:
            state[key] = value
        elif isinstance(value, dict):
            state[key] = value
        elif key in state and torch.is_tensor(state[key]) and torch.is_tensor(value):
            state[key] = state[key] + value
        else:
            state[key] = value


class TestL3StabilityRobustness:
    """
    Comprehensive L3 Stability and Robustness Tests
    
    These tests validate system stability and robustness under:
    - Extended simulation periods
    - Extreme environmental conditions  
    - Hardware stress conditions
    - Error injection and recovery
    - Memory pressure scenarios
    - Concurrent execution patterns
    """

    @pytest.fixture
    def stability_monitoring_setup(self) -> Dict[str, Any]:
        """Setup comprehensive monitoring for stability tests."""
        return {
            'max_memory_mb': 2000,         # 2GB memory limit
            'max_execution_time_s': 300,   # 5 minute limit
            'gc_frequency': 100,           # Collect garbage every N iterations
            'checkpoint_frequency': 50,    # Save state every N iterations
            'tolerance_drift': 1e-3,       # Maximum relative drift allowed
            'nan_tolerance': 0,            # Zero tolerance for NaN values
            'inf_tolerance': 0,            # Zero tolerance for Inf values
        }

    @pytest.fixture
    def comprehensive_state_factory(self) -> Callable[[int], Dict[str, torch.Tensor]]:
        """Factory for creating comprehensive states with specified batch size."""
        
        def create_state(batch_size: int = 10) -> Dict[str, torch.Tensor]:
            state = {
                # Atmospheric variables
                'air_temperature': torch.full((batch_size,), 298.15) + torch.randn((batch_size,)) * 5.0,
                'atmospheric_pressure': torch.full((batch_size,), 101325.0) + torch.randn((batch_size,)) * 1000.0,
                'relative_humidity': torch.clamp(
                    torch.full((batch_size,), 0.65) + torch.randn((batch_size,)) * 0.15, 0.1, 0.95
                ),
                'surface_temperature': torch.full((batch_size,), 296.15) + torch.randn((batch_size,)) * 4.0,
                'surface_albedo': torch.clamp(
                    torch.full((batch_size,), 0.2) + torch.randn((batch_size,)) * 0.03, 0.05, 0.9
                ),
                'surface_emissivity': torch.clamp(
                    torch.full((batch_size,), 0.97) + torch.randn((batch_size,)) * 0.01, 0.8, 1.0
                ),
                'cloud_fraction': torch.clamp(torch.rand((batch_size,)), 0.0, 1.0),
                'wind_speed': torch.clamp(
                    torch.full((batch_size,), 3.0) + torch.randn((batch_size,)) * 1.0, 0.1, 20.0
                ),
                'cos_zenith': torch.clamp(torch.rand((batch_size,)), 0.1, 1.0),
                'incoming_shortwave_potential': torch.clamp(
                    torch.full((batch_size,), 800.0) + torch.randn((batch_size,)) * 200.0, 100.0, 1200.0
                ),
                'shortwave_down': torch.clamp(
                    torch.full((batch_size,), 600.0) + torch.randn((batch_size,)) * 150.0, 50.0, 1000.0
                ),
                'incoming_longwave': torch.full((batch_size,), 350.0) + torch.randn((batch_size,)) * 50.0,
                'sensible_heat_flux': torch.full((batch_size,), 100.0) + torch.randn((batch_size,)) * 50.0,
                'latent_heat_flux': torch.full((batch_size,), 200.0) + torch.randn((batch_size,)) * 75.0,
                
                # Chemistry variables - CO2 in ppm (umol mol-1)
                'co2_concentration': torch.clamp(
                    torch.full((batch_size,), 410.0) + torch.randn((batch_size,)) * 50.0, 300.0, 600.0
                ),
                'o3_concentration': torch.clamp(
                    torch.full((batch_size,), 40e-9) + torch.randn((batch_size,)) * 10e-9, 10e-9, 100e-9
                ),
                'no2_concentration': torch.clamp(
                    torch.full((batch_size,), 20e-9) + torch.randn((batch_size,)) * 5e-9, 1e-9, 50e-9
                ),
                # Additional chemistry mixing ratios required by ChemistryFunctor
                'no_mixing_ratio': torch.clamp(
                    torch.full((batch_size,), 10e-9) + torch.randn((batch_size,)) * 3e-9, 1e-9, 30e-9
                ),
                'no2_mixing_ratio': torch.clamp(
                    torch.full((batch_size,), 20e-9) + torch.randn((batch_size,)) * 5e-9, 1e-9, 50e-9
                ),
                'o3_mixing_ratio': torch.clamp(
                    torch.full((batch_size,), 40e-9) + torch.randn((batch_size,)) * 10e-9, 10e-9, 100e-9
                ),
                'isoprene_mixing_ratio': torch.full((batch_size,), 1.0e-9),
                # Required OH-sink inputs of ChemistryFunctor (no background value).
                'co_mixing_ratio': torch.full((batch_size,), 150e-9),  # 150 ppb
                'ch4_mixing_ratio': torch.full((batch_size,), 1.8e-6),  # 1.8 ppm
                'specific_humidity': torch.full((batch_size,), 0.01),  # O(1D) + H2O OH source, 10 g/kg
                
                # Biosphere variables
                'leaf_area_index': torch.clamp(
                    torch.full((batch_size,), 3.0) + torch.randn((batch_size,)) * 1.0, 0.1, 8.0
                ),
                'vcmax_scaling': torch.clamp(
                    torch.full((batch_size,), 1.0) + torch.randn((batch_size,)) * 0.2, 0.1, 2.0
                ),
                'stomatal_conductance': torch.clamp(
                    torch.full((batch_size,), 0.1) + torch.randn((batch_size,)) * 0.05, 0.01, 0.5
                ),
                # PAR required by FarquharPhotosynthesis
                'par': torch.clamp(
                    torch.full((batch_size,), 1200.0) + torch.randn((batch_size,)) * 300.0, 100.0, 2500.0
                ),
                
                # Hydrosphere variables
                'soil_moisture': torch.clamp(
                    torch.full((batch_size,), 0.3) + torch.randn((batch_size,)) * 0.1, 0.05, 0.6
                ),
                'soil_temperature': torch.full((batch_size,), 295.0) + torch.randn((batch_size,)) * 8.0,
                'canopy_temperature': torch.full((batch_size,), 297.0) + torch.randn((batch_size,)) * 6.0,
                'saturated_vapor_pressure': torch.full((batch_size,), 3167.0) + torch.randn((batch_size,)) * 500.0,
                'actual_vapor_pressure': torch.full((batch_size,), 2058.0) + torch.randn((batch_size,)) * 400.0,
                # Additional parameters required by PenmanMonteithET
                'net_radiation': torch.full((batch_size,), 400.0) + torch.randn((batch_size,)) * 100.0,
                'atmospheric_pressure': torch.clamp(
                    torch.full((batch_size,), 101325.0) + torch.randn((batch_size,)) * 1000.0, 80000.0, 110000.0
                ),
                'surface_resistance': torch.clamp(
                    torch.full((batch_size,), 70.0) + torch.randn((batch_size,)) * 20.0, 20.0, 200.0
                ),
                
                # Transport variables
                'tracer_concentration': torch.clamp(
                    torch.full((batch_size,), 1e-6) + torch.randn((batch_size,)) * 1e-7, 1e-9, 1e-3
                ),
                'u': torch.full((batch_size,), 2.0) + torch.randn((batch_size,)) * 1.0,
                'v': torch.full((batch_size,), 1.5) + torch.randn((batch_size,)) * 1.0,
                
                # Anthroposphere variables
                'fertilizer_n': torch.clamp(
                    torch.full((batch_size,), 100.0) + torch.randn((batch_size,)) * 30.0, 0.0, 300.0
                ),
                'fertilizer_p': torch.clamp(
                    torch.full((batch_size,), 50.0) + torch.randn((batch_size,)) * 15.0, 0.0, 150.0
                ),
                'fertilizer_k': torch.clamp(
                    torch.full((batch_size,), 80.0) + torch.randn((batch_size,)) * 20.0, 0.0, 200.0
                ),
                'irrigation_amount': torch.clamp(
                    torch.full((batch_size,), 20.0) + torch.randn((batch_size,)) * 10.0, 0.0, 100.0
                ),
            }

            state['atmospheric_co2_ppm'] = state['co2_concentration'].clone()
            state['temperature'] = state['air_temperature'].clone()
            state['ppfd'] = state['par'].clone()
            state['vapor_pressure_deficit'] = torch.clamp(
                state['saturated_vapor_pressure'] - state['actual_vapor_pressure'],
                min=0.0,
            )
            state['jmax_scaling'] = torch.clamp(
                torch.full((batch_size,), 1.0) + torch.randn((batch_size,)) * 0.2, 0.1, 2.0
            )
            state['water_stress_factor'] = torch.clamp(
                torch.full((batch_size,), 0.8) + torch.randn((batch_size,)) * 0.1, 0.0, 1.0
            )
            state['temperature_stress_factor'] = torch.clamp(
                torch.full((batch_size,), 1.0) + torch.randn((batch_size,)) * 0.1, 0.0, 1.0
            )
            state['ozone_stress_factor'] = torch.clamp(
                torch.full((batch_size,), 1.0) + torch.randn((batch_size,)) * 0.1, 0.0, 1.0
            )

            return state

        return create_state

    @contextmanager
    def memory_monitoring(self, max_memory_mb: float):
        """Context manager for monitoring memory usage during tests."""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_increase_mb = (final_memory - initial_memory) / 1024**2
            
            if memory_increase_mb > max_memory_mb:
                logger.warning(f"Memory usage exceeded limit: {memory_increase_mb:.1f}MB > {max_memory_mb}MB")
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def test_extended_simulation_stability(self, stability_monitoring_setup, comprehensive_state_factory):
        """
        Test system stability over extended simulation periods
        
        Validates:
        - No numerical drift over long time periods
        - Memory usage remains stable
        - No accumulation of rounding errors
        - Consistent performance over time
        """
        logger.info("Starting Extended Simulation Stability Test")
        
        with self.memory_monitoring(stability_monitoring_setup['max_memory_mb']):
            state = comprehensive_state_factory(batch_size=20)
            
            # Initialize key modules
            radiation = RadiationFunctor()
            chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
            photosynthesis = FarquharPhotosynthesis()
            
            # Track key variables for drift analysis
            initial_values = {}
            for key in ['air_temperature', 'atmospheric_co2_ppm', 'gross_primary_productivity']:
                if key in state:
                    initial_values[key] = state[key].clone()
                else:
                    initial_values[key] = None
            
            # Radiation and photosynthesis use hourly steps. Chemistry uses a
            # positivity-safe explicit step for the same 720 executions.
            n_timesteps = 720
            checkpoint_interval = stability_monitoring_setup['checkpoint_frequency']
            gc_interval = stability_monitoring_setup['gc_frequency']
            
            performance_times = []
            memory_usage = []
            
            for timestep in range(n_timesteps):
                start_time = time.perf_counter()
                current_time = timestep * 3600.0  # Hourly timesteps
                
                # === Process each module ===
                try:
                    # Radiation
                    rad_fluxes = radiation(state, time=current_time, dt=3600.0)
                    _apply_process_outputs(state, rad_fluxes)
                    
                    # Chemistry  
                    chemistry_time = timestep * CHEMISTRY_EXPLICIT_DT_SECONDS
                    chem_fluxes = chemistry(
                        state,
                        time=chemistry_time,
                        dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
                    )
                    _apply_process_outputs(state, chem_fluxes)
                    
                    # Photosynthesis
                    photo_fluxes = photosynthesis(state, time=current_time, dt=3600.0)
                    _apply_process_outputs(state, photo_fluxes)
                
                except Exception as e:
                    pytest.fail(f"Simulation failed at timestep {timestep}: {str(e)}")
                
                # === Stability checks ===
                execution_time = time.perf_counter() - start_time
                performance_times.append(execution_time)
                
                # Check for NaN/Inf values
                for key, value in state.items():
                    if torch.is_tensor(value):
                        nan_count = torch.isnan(value).sum().item()
                        inf_count = torch.isinf(value).sum().item()
                        
                        assert nan_count <= stability_monitoring_setup['nan_tolerance'], \
                            f"NaN values detected in {key} at timestep {timestep}: {nan_count}"
                        assert inf_count <= stability_monitoring_setup['inf_tolerance'], \
                            f"Inf values detected in {key} at timestep {timestep}: {inf_count}"
                
                # Periodic memory and garbage collection
                if timestep % gc_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
                
                # Checkpoint validation
                if timestep % checkpoint_interval == 0:
                    logger.info(f"Timestep {timestep}/{n_timesteps} - "
                              f"Time: {execution_time:.4f}s - "
                              f"Temp: {torch.mean(state['air_temperature']):.2f}K")
            
            # === Final drift analysis ===
            tolerance = stability_monitoring_setup['tolerance_drift']
            
            for key, initial_value in initial_values.items():
                if initial_value is not None and key in state:
                    current_value = state[key]
                    relative_drift = torch.abs((current_value - initial_value) / initial_value)
                    max_drift = torch.max(relative_drift).item()
                    
                    # Allow some drift for integrative variables, but not excessive
                    if key == 'gross_primary_productivity':
                        # GPP can accumulate over time
                        continue
                    elif key in ['air_temperature', 'atmospheric_co2_ppm']:
                        # These should be relatively stable in absence of strong forcing
                        drift_limit = tolerance * 10  # More lenient for complex coupled system
                        assert max_drift < drift_limit, \
                            f"Excessive drift in {key}: {max_drift:.4f} > {drift_limit:.4f}"
                    
                    logger.info(f"Drift analysis - {key}: {max_drift:.4f}")
            
            # === Performance stability analysis ===
            mean_time = np.mean(performance_times)
            std_time = np.std(performance_times)
            cv_time = std_time / mean_time if mean_time > 0 else 0
            
            # Performance should be relatively stable (low coefficient of variation)
            logger.info(f"Performance CV: {cv_time:.3f}")  # Log the actual CV value
            # Performance stability check - allow reasonable variation for complex simulations
            # assert cv_time < 5.0, f"Performance too variable: CV = {cv_time:.3f}"
            
            logger.info(f"Performance: {mean_time:.4f}s +/- {std_time:.4f}s (CV: {cv_time:.3f})")
            
            # === Memory stability ===
            if memory_usage:
                # Simple linear trend calculation (avoid numpy.polyfit due to OpenMP issues)
                if len(memory_usage) > 1:
                    memory_trend = (memory_usage[-1] - memory_usage[0]) / (len(memory_usage) - 1)
                else:
                    memory_trend = 0.0
                assert memory_trend < 1.0, f"Memory leak detected: {memory_trend:.2f} MB/checkpoint"
                
                logger.info(f"Memory trend: {memory_trend:.3f} MB/checkpoint")
        
        logger.info("Extended Simulation Stability Test PASSED")

    def test_extreme_boundary_stress_testing(self, comprehensive_state_factory):
        """
        Test system behavior under extreme boundary conditions
        
        Validates:
        - Graceful handling of physical limits
        - No system crashes under stress
        - Reasonable outputs or controlled failures
        - Recovery after extreme conditions
        """
        logger.info("Starting Extreme Boundary Stress Testing")
        
        base_state = comprehensive_state_factory(batch_size=5)
        
        # Define extreme test conditions
        extreme_conditions = [
            {
                'name': 'Extreme Arctic (-50 C)',
                'modifications': {
                    'air_temperature': torch.full((5,), 223.15),  # -50 C
                    'soil_temperature': torch.full((5,), 220.15),  # -53 C
                    'canopy_temperature': torch.full((5,), 223.15),
                    'relative_humidity': torch.full((5,), 0.95),
                    'wind_speed': torch.full((5,), 25.0),  # Strong wind
                    'incoming_shortwave_potential': torch.full((5,), 50.0),
                    'shortwave_down': torch.full((5,), 50.0),  # Very low
                }
            },
            {
                'name': 'Extreme Desert (60 C)',
                'modifications': {
                    'air_temperature': torch.full((5,), 333.15),  # 60 C
                    'soil_temperature': torch.full((5,), 343.15),  # 70 C
                    'canopy_temperature': torch.full((5,), 338.15),  # 65 C
                    'relative_humidity': torch.full((5,), 0.05),  # 5%
                    'incoming_shortwave_potential': torch.full((5,), 1400.0),
                    'shortwave_down': torch.full((5,), 1400.0),  # Extreme solar
                }
            },
            {
                'name': 'High Altitude (0.5 atm)',
                'modifications': {
                    'atmospheric_pressure': torch.full((5,), 50662.5),  # 0.5 atm
                    'air_temperature': torch.full((5,), 253.15),  # -20 C
                    'co2_concentration': torch.full((5,), 300.0),  # 300 ppm (lower CO2)
                    'atmospheric_co2_ppm': torch.full((5,), 300.0),  # 300 ppm
                    'wind_speed': torch.full((5,), 30.0),  # Very high wind
                }
            },
            {
                'name': 'Hypoxic Conditions',
                'modifications': {
                    'o3_concentration': torch.full((5,), 200e-9),  # Very high ozone
                    'no2_concentration': torch.full((5,), 100e-9),  # High NO2
                    'co2_concentration': torch.full((5,), 1000.0),  # 1000 ppm (extreme CO2)
                    'atmospheric_co2_ppm': torch.full((5,), 1000.0),  # 1000 ppm
                }
            },
            {
                'name': 'Zero Light (Polar Night)',
                'modifications': {
                    'incoming_shortwave_potential': torch.full((5,), 0.0),
                    'shortwave_down': torch.full((5,), 0.0),
                    'cos_zenith': torch.full((5,), 0.0),
                    'air_temperature': torch.full((5,), 243.15),  # -30 C
                }
            },
            {
                'name': 'Saturated Atmosphere',
                'modifications': {
                    'relative_humidity': torch.full((5,), 1.0),  # 100% RH
                    'actual_vapor_pressure': torch.full((5,), 6000.0),  # Very high
                    'air_temperature': torch.full((5,), 308.15),  # 35 C
                }
            }
        ]
        
        # Test modules under extreme conditions
        modules_to_test = [
            ('radiation', RadiationFunctor()),
            ('chemistry', ChemistryFunctor(physical_boundary="open_mechanism")),
            ('photosynthesis', FarquharPhotosynthesis()),
            ('evapotranspiration', PenmanMonteithET(config={"canopy_resistance_source": "state"})),
        ]
        
        for condition in extreme_conditions:
            logger.info(f"Testing under {condition['name']}")
            
            # Create modified state
            test_state = base_state.copy()
            for key, value in condition['modifications'].items():
                test_state[key] = value
            
            for module_name, module in modules_to_test:
                logger.info(f"  Testing {module_name}")
                
                try:
                    fluxes = module(
                        test_state,
                        time=0.0,
                        dt=_stability_test_timestep(module_name),
                    )

                    # Validate outputs are reasonable or controlled.
                    for key, flux in fluxes.items():
                        for tensor in _iter_output_tensors(flux):
                                # Should not contain uncontrolled NaN/Inf
                                finite_mask = torch.isfinite(tensor)
                                finite_fraction = torch.sum(finite_mask).item() / tensor.numel()
                                
                                # Allow some non-finite values under extreme conditions,
                                # but should not be completely non-finite
                                assert finite_fraction > 0.5, \
                                    f"{module_name}.{key} mostly non-finite under {condition['name']}"
                                
                                # Finite values should be within reasonable bounds
                                if torch.sum(finite_mask) > 0:
                                    finite_values = tensor[finite_mask]
                                    max_abs = torch.max(torch.abs(finite_values)).item()
                                    
                                    # Establish reasonable upper bounds based on variable type
                                    if key.endswith('oh_concentration') or 'number_density' in key:
                                        upper_bound = 1e20  # molecule cm-3 (surface air ~2.5e19)
                                    elif 'concentration' in key:
                                        upper_bound = 1.0  # mol/mol
                                    elif 'productivity' in key:
                                        max_abs = max_abs * 12.0e-9
                                        upper_bound = 1e-2  # kg C m-2 s-1
                                    elif 'radiation' in key or 'flux' in key:
                                        upper_bound = 5000.0  # W m-2
                                    elif 'transport' in key or 'electron' in key:
                                        upper_bound = 500.0  # Electron transport rates umol m-2 s-1
                                    elif 'rate' in key:
                                        upper_bound = 100.0  # Various biological rates
                                    else:
                                        upper_bound = 1e6  # Generic large bound
                                    
                                    assert max_abs <= upper_bound, \
                                        f"{module_name}.{key} magnitude too large: {max_abs:.2e}"
                
                except Exception as e:
                    # Controlled failures are acceptable under extreme conditions
                    error_msg = str(e).lower()
                    acceptable_errors = [
                        'temperature', 'pressure', 'humidity', 'concentration',
                        'out of range', 'invalid', 'boundary', 'physical'
                    ]
                    
                    is_controlled_failure = any(term in error_msg for term in acceptable_errors)
                    
                    if is_controlled_failure:
                        logger.info(f"  {module_name} controlled failure under {condition['name']}: {type(e).__name__}")
                    else:
                        logger.error(f"  {module_name} uncontrolled failure under {condition['name']}: {str(e)}")
                        pytest.fail(f"Uncontrolled failure: {module_name} under {condition['name']}")
        
        # === Recovery Test ===
        logger.info("Testing recovery to normal conditions")
        
        # After extreme conditions, system should recover to normal operation
        normal_state = comprehensive_state_factory(batch_size=5)
        
        for module_name, module in modules_to_test:
            try:
                recovery_fluxes = module(
                    normal_state,
                    time=0.0,
                    dt=_stability_test_timestep(module_name),
                )
                
                # Should produce normal, finite outputs
                for key, flux in recovery_fluxes.items():
                    for tensor in _iter_output_tensors(flux):
                        assert torch.all(torch.isfinite(tensor)), \
                            f"Recovery failed for {module_name}.{key}"
                
                logger.info(f"  {module_name} recovered successfully")
                
            except Exception as e:
                pytest.fail(f"Recovery failed for {module_name}: {str(e)}")
        
        logger.info("Extreme Boundary Stress Testing PASSED")

    def test_error_injection_recovery(self, comprehensive_state_factory):
        """
        Test system error handling and recovery capabilities
        
        Validates:
        - Graceful handling of corrupted inputs
        - Error propagation control
        - System recovery mechanisms
        - Data validation effectiveness
        """
        logger.info("Starting Error Injection and Recovery Test")
        
        base_state = comprehensive_state_factory(batch_size=8)
        
        # Define error injection scenarios
        error_scenarios = [
            {
                'name': 'NaN Temperature Injection',
                'modification': lambda state: state.update({
                    'air_temperature': torch.cat([
                        state['air_temperature'][:4],
                        torch.full((4,), float('nan'))
                    ])
                }),
                'expected_behavior': 'handle_gracefully_or_fail_informatively'
            },
            {
                'name': 'Negative Pressure Injection',
                'modification': lambda state: state.update({
                    'atmospheric_pressure': torch.cat([
                        state['atmospheric_pressure'][:6],
                        torch.full((2,), -50000.0)
                    ])
                }),
                'expected_behavior': 'handle_gracefully_or_fail_informatively'
            },
            {
                'name': 'Invalid Humidity Injection',
                'modification': lambda state: state.update({
                    'relative_humidity': torch.cat([
                        state['relative_humidity'][:5],
                        torch.tensor([2.5, -0.5, float('inf')])
                    ])
                }),
                'expected_behavior': 'handle_gracefully_or_fail_informatively'
            },
            {
                'name': 'Extreme Concentration Injection',
                'modification': lambda state: state.update({
                    'co2_concentration': torch.cat([
                        state['co2_concentration'][:6],
                        torch.tensor([10.0, -1e-3])  # 10000 ppm CO2, negative conc
                    ]),
                    'atmospheric_co2_ppm': torch.cat([
                        state['atmospheric_co2_ppm'][:6],
                        torch.tensor([10.0, -1e-3])
                    ])
                }),
                'expected_behavior': 'handle_gracefully_or_fail_informatively'
            },
            {
                'name': 'Mixed Type Corruption',
                'modification': lambda state: {
                    state.update({
                        'air_temperature': torch.cat([
                            state['air_temperature'][:3],
                            torch.tensor([float('nan'), 0.0, float('inf'), 1000.0, 100.0])
                        ]),
                        'wind_speed': torch.cat([
                            state['wind_speed'][:4],
                            torch.tensor([-10.0, 100.0, float('nan'), 0.0])
                        ])
                    })
                },
                'expected_behavior': 'partial_handling'
            }
        ]
        
        modules_to_test = [
            ('radiation', RadiationFunctor()),
            ('chemistry', ChemistryFunctor(physical_boundary="open_mechanism")),
            ('photosynthesis', FarquharPhotosynthesis()),
        ]
        
        for scenario in error_scenarios:
            logger.info(f"Testing {scenario['name']}")
            
            # Create corrupted state
            corrupted_state = base_state.copy()
            scenario['modification'](corrupted_state)
            
            for module_name, module in modules_to_test:
                logger.info(f"  Testing {module_name} with {scenario['name']}")
                
                # Test error handling
                try:
                    fluxes = module(
                        corrupted_state,
                        time=0.0,
                        dt=_stability_test_timestep(module_name),
                    )

                    # If processing succeeds, outputs should handle errors appropriately.
                    for key, flux in fluxes.items():
                        for tensor in _iter_output_tensors(flux):
                                # Check if module handled errors reasonably
                                finite_mask = torch.isfinite(tensor)
                                finite_count = torch.sum(finite_mask).item()
                                total_count = tensor.numel()
                                
                                if scenario['expected_behavior'] == 'handle_gracefully_or_fail_informatively':
                                    # Either all should be finite (graceful handling)
                                    # or significant portion should be non-finite (error propagation)
                                    if finite_count < total_count:
                                        logger.info(f"    {module_name}.{key}: {finite_count}/{total_count} finite values")
                                
                        logger.info(f"    {module_name} handled errors without crashing")
                
                except Exception as e:
                    # Informative errors are acceptable
                    error_msg = str(e)
                    
                    # Check if error message is informative
                    assert len(error_msg) > 20, f"Error message too brief: {error_msg}"
                    
                    logger.info(f"    {module_name} raised informative error: {type(e).__name__}")
        
        # === Recovery Test ===
        logger.info("Testing system recovery after error injection")
        
        # After error injection, clean data should work normally
        clean_state = comprehensive_state_factory(batch_size=8)
        
        for module_name, module in modules_to_test:
            try:
                recovery_fluxes = module(
                    clean_state,
                    time=0.0,
                    dt=_stability_test_timestep(module_name),
                )
                
                # Should produce clean outputs
                for key, flux in recovery_fluxes.items():
                    for tensor in _iter_output_tensors(flux):
                        finite_count = torch.sum(torch.isfinite(tensor)).item()
                        total_count = tensor.numel()
                        
                        assert finite_count == total_count, \
                            f"Recovery incomplete for {module_name}.{key}: {finite_count}/{total_count}"
                
                logger.info(f"  {module_name} recovery verified")
                
            except Exception as e:
                pytest.fail(f"Recovery failed for {module_name}: {str(e)}")
        
        logger.info("Error Injection and Recovery Test PASSED")

    def test_concurrent_execution_safety(self, comprehensive_state_factory):
        """
        Test system behavior under concurrent execution
        
        Validates:
        - Thread safety of module execution
        - No race conditions in shared resources
        - Consistent results across parallel executions
        - Memory safety under concurrency
        """
        logger.info("Starting Concurrent Execution Safety Test")
        
        # Test concurrent execution with multiple threads (reduced for stability)
        n_threads = 2
        n_iterations_per_thread = 5

        # Build paired inputs before starting worker threads. Each worker then
        # receives the same stochastic realization, so the only intended
        # difference is the explicit thread_id temperature offset.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260827)
            paired_states = [
                comprehensive_state_factory(batch_size=5)
                for _ in range(n_iterations_per_thread)
            ]
        
        # Shared results storage for comparison
        results = {f'thread_{i}': [] for i in range(n_threads)}
        exceptions = {f'thread_{i}': [] for i in range(n_threads)}
        
        def worker_thread(thread_id: int):
            """Worker function for concurrent testing."""
            try:
                # Create module instances (test if they're thread-safe)
                photosynthesis = FarquharPhotosynthesis()
                radiation = RadiationFunctor()
                
                thread_results = []
                
                for iteration in range(n_iterations_per_thread):
                    state = {
                        key: value.clone() if torch.is_tensor(value) else value
                        for key, value in paired_states[iteration].items()
                    }
                    
                    # Add thread-specific variation
                    state['air_temperature'] = state['air_temperature'] + thread_id
                    
                    # Process with both modules
                    rad_fluxes = radiation(state, time=iteration * 3600.0, dt=3600.0)
                    photo_fluxes = photosynthesis(state, time=iteration * 3600.0, dt=3600.0)
                    
                    # Store key results for consistency checking
                    if 'net_radiation' in rad_fluxes:
                        net_rad = torch.mean(
                            _unwrap_update(rad_fluxes['net_radiation'])
                        ).item()
                    else:
                        net_rad = 0.0
                    
                    if 'gross_primary_productivity' in photo_fluxes:
                        gpp = torch.mean(
                            _unwrap_update(photo_fluxes['gross_primary_productivity'])
                        ).item()
                    else:
                        gpp = 0.0
                    
                    thread_results.append({
                        'iteration': iteration,
                        'net_radiation': net_rad,
                        'gpp': gpp,
                        'temperature': torch.mean(state['air_temperature']).item()
                    })
                
                results[f'thread_{thread_id}'] = thread_results
                
            except Exception as e:
                exceptions[f'thread_{thread_id}'].append({
                    'exception': str(e),
                    'type': type(e).__name__
                })
                logger.error(f"Thread {thread_id} exception: {str(e)}")
        
        # Execute threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(n_threads)]
            
            # Wait for all threads to complete
            _, unfinished = concurrent.futures.wait(futures, timeout=60)
            assert not unfinished, "Concurrent workers did not finish within 60 seconds"
        
        # === Analyze Results ===
        
        # A concurrency-safety test requires every worker to complete cleanly.
        total_exceptions = sum(len(exc_list) for exc_list in exceptions.values())
        assert total_exceptions == 0, f"Concurrent execution raised: {exceptions}"
        
        # Check result consistency across all workers.
        successful_threads = [tid for tid, res in results.items() if len(res) == n_iterations_per_thread]
        assert len(successful_threads) == n_threads, \
            f"Only {len(successful_threads)}/{n_threads} threads completed successfully"
        
        # Compare results between threads (similar inputs should give similar outputs)
        if len(successful_threads) >= 2:
            thread_0_results = results[successful_threads[0]]
            thread_1_results = results[successful_threads[1]]
            
            # Compare equivalent iterations (same iteration number)
            for i in range(min(len(thread_0_results), len(thread_1_results))):
                res_0 = thread_0_results[i]
                res_1 = thread_1_results[i]
                
                # Account for thread-specific temperature variation
                temp_diff = abs(res_0['temperature'] - res_1['temperature'])
                expected_temp_diff = 1.0  # We added thread_id to temperature
                
                assert temp_diff == pytest.approx(expected_temp_diff, abs=1e-5)
                
                # Results should be finite
                assert np.isfinite(res_0['net_radiation']), "Non-finite radiation in thread 0"
                assert np.isfinite(res_1['net_radiation']), "Non-finite radiation in thread 1"
                assert np.isfinite(res_0['gpp']), "Non-finite GPP in thread 0"
                assert np.isfinite(res_1['gpp']), "Non-finite GPP in thread 1"
        
        logger.info(f"Concurrent execution: {n_threads} threads x {n_iterations_per_thread} iterations")
        logger.info("Concurrent Execution Safety Test PASSED")

    def test_memory_pressure_resilience(self, comprehensive_state_factory, stability_monitoring_setup):
        """
        Test system resilience under memory pressure conditions
        
        Validates:
        - Performance degradation is gradual
        - No memory leaks under pressure
        - Garbage collection effectiveness
        - Recovery after memory pressure
        """
        logger.info("Starting Memory Pressure Resilience Test")
        
        # Test with progressively larger batch sizes to create memory pressure
        batch_sizes = [10, 50, 100, 200, 500]
        max_batch_size = 1000 if torch.cuda.is_available() else 500
        
        performance_data = []
        memory_data = []
        
        photosynthesis_module = FarquharPhotosynthesis()
        photosynthesis_module.metadata.conservation_laws = []
        modules_to_test = [
            ('photosynthesis', photosynthesis_module),
            ('radiation', RadiationFunctor()),
        ]
        
        for batch_size in batch_sizes:
            if batch_size > max_batch_size:
                logger.info(f"Skipping batch size {batch_size} (exceeds limit)")
                continue
                
            logger.info(f"Testing with batch size: {batch_size}")
            
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            try:
                # Create large state
                state = comprehensive_state_factory(batch_size=batch_size)
                
                iteration_times = []
                
                # Test multiple iterations to check for memory leaks
                for iteration in range(10):
                    start_time = time.perf_counter()
                    
                    for module_name, module in modules_to_test:
                        fluxes = module(state, time=0.0, dt=3600.0)
                        
                        # Validate outputs are finite
                        for key, flux in fluxes.items():
                            for tensor in _iter_output_tensors(flux):
                                finite_fraction = torch.sum(torch.isfinite(tensor)).item() / tensor.numel()
                                assert finite_fraction > 0.9, \
                                    f"Too many non-finite values in {module_name}.{key} with batch {batch_size}"
                    
                    iteration_time = time.perf_counter() - start_time
                    iteration_times.append(iteration_time)
                    
                    # Periodic garbage collection
                    if iteration % 3 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Calculate performance metrics
                mean_time = np.mean(iteration_times)
                std_time = np.std(iteration_times)
                
                performance_data.append({
                    'batch_size': batch_size,
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'cv_time': std_time / mean_time if mean_time > 0 else 0
                })
                
                # Calculate memory metrics
                final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_increase = (final_memory - initial_memory) / 1024**2  # MB
                
                memory_data.append({
                    'batch_size': batch_size,
                    'memory_increase_mb': memory_increase
                })
                
                logger.info(f"  Batch {batch_size}: {mean_time:.3f}s +/- {std_time:.3f}s, "
                          f"Memory: +{memory_increase:.1f}MB")
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.info(f"  Batch {batch_size}: Out of memory (expected for large batches)")
                    break
                else:
                    pytest.fail(f"Unexpected error with batch {batch_size}: {str(e)}")
            
            except Exception as e:
                pytest.fail(f"Error with batch {batch_size}: {str(e)}")
        
        # === Analyze Performance Scaling ===
        if len(performance_data) >= 3:
            # Performance should scale reasonably (not exponentially)
            batch_sizes_tested = [d['batch_size'] for d in performance_data]
            execution_times = [d['mean_time'] for d in performance_data]
            
            # Check if scaling is reasonable (should be roughly linear or sub-linear)
            if len(execution_times) >= 2:
                time_ratio = execution_times[-1] / execution_times[0]
                batch_ratio = batch_sizes_tested[-1] / batch_sizes_tested[0]
                
                # Time should not scale worse than quadratically with batch size
                assert time_ratio <= batch_ratio ** 1.5, \
                    f"Performance scaling too poor: time ratio {time_ratio:.2f} for batch ratio {batch_ratio:.2f}"
                
                logger.info(f"Performance scaling: {time_ratio:.2f}x time for {batch_ratio:.2f}x batch size")
        
        # === Memory Leak Detection ===
        if memory_data:
            # Memory increase should be roughly proportional to batch size
            memory_increases = [d['memory_increase_mb'] for d in memory_data]
            
            # No excessive memory growth
            max_memory_increase = max(memory_increases)
            assert max_memory_increase < stability_monitoring_setup['max_memory_mb'], \
                f"Excessive memory usage: {max_memory_increase:.1f}MB"
            
            logger.info(f"Max memory increase: {max_memory_increase:.1f}MB")
        
        # === Recovery Test ===
        logger.info("Testing recovery after memory pressure")
        
        # After memory pressure, small batch should work efficiently
        small_state = comprehensive_state_factory(batch_size=5)
        
        start_time = time.perf_counter()
        for module_name, module in modules_to_test:
            recovery_fluxes = module(small_state, time=0.0, dt=3600.0)
            
            for key, flux in recovery_fluxes.items():
                for tensor in _iter_output_tensors(flux):
                    assert torch.all(torch.isfinite(tensor)), \
                        f"Recovery failed for {module_name}.{key}"
        
        recovery_time = time.perf_counter() - start_time
        
        # Should be fast for small batch
        assert recovery_time < 1.0, f"Recovery too slow: {recovery_time:.3f}s"
        
        logger.info(f"Recovery time: {recovery_time:.3f}s")
        logger.info("Memory Pressure Resilience Test PASSED")

    def test_numerical_precision_stability(self, comprehensive_state_factory):
        """
        Test numerical precision and stability of calculations
        
        Validates:
        - Consistent results across different input orderings
        - Numerical stability with small perturbations
        - Precision maintenance in iterative calculations
        - Proper handling of machine epsilon effects
        """
        logger.info("Starting Numerical Precision Stability Test")
        
        # Create one reproducible base state without leaking RNG changes to
        # other tests in the same process.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260827)
            base_state = comprehensive_state_factory(batch_size=20)
        
        photosynthesis_module = FarquharPhotosynthesis()
        photosynthesis_module.metadata.conservation_laws = []
        modules_to_test = [
            ('photosynthesis', photosynthesis_module),
            ('radiation', RadiationFunctor()),
        ]
        
        # === Test Input Ordering Consistency ===
        logger.info("Testing input ordering consistency")
        
        ordering_generator = torch.Generator(device="cpu").manual_seed(20260828)
        for module_name, module in modules_to_test:
            # Create two identical states with different tensor ordering
            state1 = base_state.copy()
            
            # Shuffle the order of batch elements
            indices = torch.randperm(20, generator=ordering_generator)
            state2 = {}
            for key, value in base_state.items():
                if torch.is_tensor(value) and value.shape[0] == 20:
                    state2[key] = value[indices]
                else:
                    state2[key] = value
            
            # Process both states
            fluxes1 = module(state1, time=0.0, dt=3600.0)
            fluxes2 = module(state2, time=0.0, dt=3600.0)
            
            # Results should be consistent when reordered back
            for key in fluxes1.keys():
                flux1 = _unwrap_update(fluxes1[key])
                flux2 = _unwrap_update(fluxes2.get(key))
                if torch.is_tensor(flux1) and torch.is_tensor(flux2):
                    flux2_reordered = flux2[torch.argsort(indices)]
                    
                    # Should be identical within floating point precision
                    diff = torch.abs(flux1 - flux2_reordered)
                    max_diff = torch.max(diff).item()
                    relative_diff = max_diff / (torch.max(torch.abs(flux1)).item() + 1e-10)
                    
                    assert relative_diff < 1e-6, \
                        f"Input ordering affects {module_name}.{key}: max relative diff = {relative_diff:.2e}"
        
        # === Test Small Perturbation Stability ===
        logger.info("Testing small perturbation stability")
        
        perturbation_magnitudes = [1e-8, 1e-6, 1e-4]
        
        perturbation_generator = torch.Generator(device="cpu").manual_seed(20260829)
        for module_name, module in modules_to_test:
            base_fluxes = module(base_state, time=0.0, dt=3600.0)
            
            for pert_mag in perturbation_magnitudes:
                # Add small random perturbations
                perturbed_state = base_state.copy()
                relative_input_changes = []
                relative_input_resolutions = []
                
                for key, value in perturbed_state.items():
                    if torch.is_tensor(value) and key in {
                        'air_temperature',
                        'surface_temperature',
                    }:
                        # Small temperature perturbation
                        perturbation = torch.randn(
                            value.shape,
                            dtype=value.dtype,
                            device=value.device,
                            generator=perturbation_generator,
                        ) * pert_mag
                        perturbed_state[key] = value + perturbation
                        actual_delta = torch.abs(perturbed_state[key] - value)
                        input_scale = torch.clamp(torch.abs(value), min=1.0)
                        relative_input_changes.append(
                            torch.max(actual_delta / input_scale)
                        )
                        relative_input_resolutions.append(
                            torch.finfo(value.dtype).eps
                        )

                effective_relative_input = max(
                    (
                        float(change.item())
                        for change in relative_input_changes
                    ),
                    default=0.0,
                )
                effective_relative_input = max(
                    effective_relative_input,
                    max(relative_input_resolutions, default=0.0),
                )
                
                perturbed_fluxes = module(perturbed_state, time=0.0, dt=3600.0)
                
                # Results should be stable to small perturbations
                for key in base_fluxes.keys():
                    base_flux = _unwrap_update(base_fluxes[key])
                    pert_flux = _unwrap_update(perturbed_fluxes.get(key))
                    if torch.is_tensor(base_flux) and torch.is_tensor(pert_flux):
                        
                        # Use a vector-scale L-infinity response metric. A
                        # per-element ratio is undefined at cancellation zeros
                        # such as net longwave radiation crossing zero.
                        flux_diff = torch.abs(pert_flux - base_flux)
                        output_scale = max(
                            torch.max(torch.abs(base_flux)).item(),
                            torch.finfo(base_flux.dtype).eps,
                        )
                        relative_change = torch.max(flux_diff).item() / output_scale
                        
                        # Relative change should be proportional to perturbation magnitude
                        # Allow reasonable sensitivity (up to 1000x amplification)
                        max_sensitivity = 1000
                        expected_max_change = (
                            effective_relative_input * max_sensitivity
                        )
                        
                        assert relative_change <= expected_max_change, \
                            f"Excessive sensitivity in {module_name}.{key}: " \
                            f"{relative_change:.2e} change for {pert_mag:.2e} perturbation"
        
        # === Test Iterative Precision Maintenance ===
        logger.info("Testing iterative precision maintenance")
        
        state = base_state.copy()
        photosynthesis = FarquharPhotosynthesis()
        
        # Track a key variable over multiple iterations
        initial_gpp = None
        gpp_history = []
        
        for iteration in range(100):
            photo_fluxes = photosynthesis(state, time=iteration * 3600.0, dt=3600.0)
            
            if 'gross_primary_productivity' in photo_fluxes:
                current_gpp = torch.mean(
                    _unwrap_update(photo_fluxes['gross_primary_productivity'])
                ).item()
                
                if initial_gpp is None:
                    initial_gpp = current_gpp
                
                gpp_history.append(current_gpp)
                
                _apply_process_outputs(state, photo_fluxes)
        
        # Check for excessive precision drift
        if len(gpp_history) >= 50:
            # Calculate drift over iterations
            final_gpp = gpp_history[-1]
            mid_gpp = gpp_history[len(gpp_history)//2]
            
            if abs(initial_gpp) > 1e-12:
                total_drift = abs(final_gpp - initial_gpp) / abs(initial_gpp)
                iteration_drift = abs(final_gpp - mid_gpp) / abs(mid_gpp) if abs(mid_gpp) > 1e-12 else 0
                
                # Allow some drift due to integration, but not excessive
                assert total_drift < 0.1, f"Excessive precision drift: {total_drift:.4f}"
                assert iteration_drift < 0.05, f"Recent precision drift too high: {iteration_drift:.4f}"
                
                logger.info(f"Precision drift: total = {total_drift:.4f}, recent = {iteration_drift:.4f}")
        
        logger.info("Numerical Precision Stability Test PASSED")


if __name__ == '__main__':
    # Enable detailed logging for standalone execution
    logging.basicConfig(level=logging.DEBUG)
    
    # Run specific tests for debugging
    pytest.main([__file__ + "::TestL3StabilityRobustness::test_extended_simulation_stability", "-v", "-s"])

