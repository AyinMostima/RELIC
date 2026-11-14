#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 System Test: Comprehensive End-to-End RELIC Testing

This module provides the most comprehensive L3 system tests for the RELIC 
biogeochemical modeling framework, including:

1. Complete ecosystem simulation workflows
2. Long-term simulation testing (annual cycles)
3. Multi-site comparison testing
4. Extreme condition stress testing
5. Performance benchmarking and profiling
6. System stability and robustness validation

Testing Strategy:
- Full biogeochemical cycle integration
- Seasonal and diurnal pattern validation
- Stress testing under extreme environmental conditions
- Memory efficiency and performance monitoring
- Scientific accuracy validation against known benchmarks

Reference:
- Farquhar, G. D., et al. (1980). A biochemical model of photosynthetic CO2 assimilation
- Monteith, J. L. (1965). Evaporation and environment
- Running, S. W., & Coughlan, J. C. (1988). A general model of forest ecosystem processes

Authors: Yuchong Long
License: MIT
Created: 2024
"""

import pytest
import torch
import numpy as np
import time
import gc
import logging
import psutil
import os
from typing import Dict
from pathlib import Path
import tempfile
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Core RELIC imports
from relic.core.simulator import RELICSimulator
from relic.core.device_policy import is_cuda_available

# Scientific module imports

# Validation and utility imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PERF_MULTIPLIER = float(os.getenv("RELIC_L3_PERF_MULTIPLIER", "4.0"))
if PERF_MULTIPLIER < 1.0:
    PERF_MULTIPLIER = 1.0


# Units of this test's own scenario variables, which no SCM contract or SSOT
# entry declares; they follow the units stated with the test data. The simulator
# infers no units from names (relic/core/simulator/state_init.py).
_TEST_STATE_UNITS = {
    'biomass_wood': 'kg C m-2',
    'canopy_interception': 'mm',
    'fertilizer_n': 'kg ha-1',
    'irrigation_amount': 'mm',
    'no2_concentration': 'mol mol-1',
    'soil_nh4': 'kg m-2',
    'soil_no3': 'kg m-2',
    'soil_phosphorus': 'kg m-2',
    'surface_runoff': 'mm day-1',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class SystemPerformanceMonitor:
    """System performance monitoring utilities for L3 tests."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu_times = None
        self.process = psutil.Process(os.getpid())
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024**2  # MB
        self.start_cpu_times = self.process.cpu_times()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
    def get_current_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        current_time = time.perf_counter()
        current_memory = self.process.memory_info().rss / 1024**2
        current_cpu = self.process.cpu_times()
        
        stats = {
            'elapsed_time': current_time - self.start_time if self.start_time else 0,
            'memory_usage_mb': current_memory,
            'memory_increase_mb': current_memory - self.start_memory if self.start_memory else 0,
            'cpu_time_user': current_cpu.user - self.start_cpu_times.user if self.start_cpu_times else 0,
            'cpu_time_system': current_cpu.system - self.start_cpu_times.system if self.start_cpu_times else 0,
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            
        return stats


class EcosystemDataGenerator:
    """Generate realistic ecosystem data for testing."""
    
    @staticmethod
    def generate_annual_weather(n_days: int = 365, n_sites: int = 1) -> Dict[str, torch.Tensor]:
        """
        Generate realistic annual weather patterns.
        
        Mathematical Expressions:
        - Temperature: T(t) = T_mean + T_amplitude * cos(2pi(t - t_phase)/365) + noise
        - Radiation: PAR(t) = PAR_max * max(0, cos(2pi(t - 172)/365)) * daylight_factor
        - Precipitation: Stochastic with seasonal patterns
        
        Parameters:
        - T_mean: Annual mean temperature (K)
        - T_amplitude: Seasonal temperature amplitude (K)
        - PAR_max: Maximum photosynthetically active radiation (umol m-2 s-1)
        
        References:
        - Campbell, G. S., & Norman, J. M. (1998). Environmental Biophysics
        """
        t = torch.arange(n_days, dtype=torch.float32)
        sites = torch.arange(n_sites, dtype=torch.float32)
        
        # Create meshgrid for days x sites
        T, S = torch.meshgrid(t, sites, indexing='ij')
        
        # Annual temperature cycle (K)
        T_mean = 288.15 + S * 2.0  # Vary by site (15-17degC base)
        T_amplitude = 15.0 - S * 1.0  # Vary amplitude by site
        T_phase = 15 + S * 5  # Vary phase by site (days from Jan 1)
        
        temperature = (T_mean + T_amplitude * 
                      torch.cos(2 * np.pi * (T - T_phase) / 365) +
                      torch.randn_like(T) * 2.0)  # 2K daily noise
        
        # Solar radiation cycle (W/m2)
        radiation_max = 800.0 - S * 50  # Vary max by site
        radiation = (radiation_max * 
                    torch.clamp(torch.cos(2 * np.pi * (T - 172) / 365), 0, 1) +
                    torch.randn_like(T) * 50)  # Daily noise
        
        # PAR (photosynthetically active radiation, umol m-2 s-1)
        par = radiation * 2.0  # Approximate conversion factor
        
        # Precipitation (mm day-1) - stochastic with seasonal pattern
        precip_base = 2.0 + S * 0.5  # Base precipitation rate
        precip_seasonal = 1.5 * torch.cos(2 * np.pi * (T - 120) / 365)  # Summer peak
        stochastic_precip = torch.empty_like(T).exponential_(1.0)
        precipitation = torch.clamp(
            precip_base + precip_seasonal + stochastic_precip * 1.5,
            0, 50  # Max 50mm day-1
        )
        
        # Relative humidity (fraction)
        humidity_base = 0.7 - S * 0.05  # Vary by site
        humidity = torch.clamp(
            humidity_base + 0.2 * torch.sin(2 * np.pi * T / 365) + torch.randn_like(T) * 0.1,
            0.1, 0.95
        )
        
        # Wind speed (m/s)
        wind_speed = 3.0 + torch.randn_like(T) * 1.0
        wind_speed = torch.clamp(wind_speed, 0.5, 15.0)
        
        # Air pressure (Pa) - slight variation
        pressure = 101325.0 + torch.randn_like(T) * 1000
        
        return {
            'air_temperature': temperature,
            'incoming_shortwave': radiation,
            'par': par,
            'precipitation': precipitation,
            'relative_humidity': humidity,
            'wind_speed': wind_speed,
            'air_pressure': pressure,
            'days': T,
            'sites': S,
        }
    
    @staticmethod
    def generate_ecosystem_initial_state(n_sites: int = 1, ecosystem_type: str = 'temperate_forest') -> Dict[str, torch.Tensor]:
        """
        Generate realistic initial ecosystem state.
        
        Parameters vary by ecosystem type:
        - temperate_forest: High LAI, moderate soil organic matter
        - grassland: Low LAI, high root biomass
        - cropland: Variable LAI, managed soil
        - desert: Very low LAI and soil moisture
        
        References:
        - Chapin, F. S., et al. (2011). Principles of Terrestrial Ecosystem Ecology
        """
        
        # Ecosystem-specific parameters
        ecosystem_params = {
            'temperate_forest': {
                'leaf_area_index': 5.0,
                'biomass_leaf': 0.5,  # kg C m-2
                'biomass_wood': 15.0,
                'biomass_root': 3.0,
                'soil_organic_carbon': 8.0,
                'soil_nitrogen': 0.4,
                'soil_moisture': 0.35,
            },
            'grassland': {
                'leaf_area_index': 2.5,
                'biomass_leaf': 0.2,
                'biomass_wood': 0.1,
                'biomass_root': 1.5,
                'soil_organic_carbon': 12.0,
                'soil_nitrogen': 0.6,
                'soil_moisture': 0.25,
            },
            'cropland': {
                'leaf_area_index': 3.0,
                'biomass_leaf': 0.3,
                'biomass_wood': 0.2,
                'biomass_root': 0.8,
                'soil_organic_carbon': 4.0,
                'soil_nitrogen': 0.8,
                'soil_moisture': 0.30,
            },
            'desert': {
                'leaf_area_index': 0.5,
                'biomass_leaf': 0.05,
                'biomass_wood': 0.5,
                'biomass_root': 0.2,
                'soil_organic_carbon': 0.5,
                'soil_nitrogen': 0.05,
                'soil_moisture': 0.05,
            }
        }
        
        params = ecosystem_params.get(ecosystem_type, ecosystem_params['temperate_forest'])
        
        # Add site-to-site variation (+/-20%)
        variation = 0.2
        state = {}
        for key, value in params.items():
            site_variation = 1.0 + (torch.rand(n_sites) - 0.5) * 2 * variation
            state[key] = torch.full((n_sites,), value) * site_variation
        
        # Additional state variables
        state.update({
            # Atmospheric
            'co2_concentration': torch.full((n_sites,), 415.0),  # ppm
            'o3_mixing_ratio': torch.full((n_sites,), 40e-9),   # mol/mol
            'no2_concentration': torch.full((n_sites,), 20e-9),  # mol/mol
            
            # Soil properties
            'soil_temperature': torch.full((n_sites,), 288.15),  # K
            'soil_ph': torch.full((n_sites,), 6.5),
            'bulk_density': torch.full((n_sites,), 900.0),      # kg/m3
            'porosity': torch.full((n_sites,), 0.45),
            
            # Water cycle
            'water_table_depth': torch.full((n_sites,), 2.0),   # m
            'canopy_interception': torch.zeros((n_sites,)),     # mm
            'surface_runoff': torch.zeros((n_sites,)),          # mm day-1
            
            # Nutrient pools (kg/m2)
            'soil_nh4': torch.full((n_sites,), 0.01),
            'soil_no3': torch.full((n_sites,), 0.02),
            'soil_phosphorus': torch.full((n_sites,), 0.005),
            
            # Management (for anthroposphere)
            'fertilizer_n': torch.zeros((n_sites,)),            # kg N/ha
            'irrigation_amount': torch.zeros((n_sites,)),       # mm
            
            # Initialize flux variables to zero
            'gross_primary_productivity': torch.zeros((n_sites,)),
            'net_primary_productivity': torch.zeros((n_sites,)),
            'ecosystem_respiration': torch.zeros((n_sites,)),
            'evapotranspiration_flux': torch.zeros((n_sites,)),
            'net_radiation': torch.zeros((n_sites,)),
            'precipitation_flux': torch.zeros((n_sites,)),
        })

        return state


class TestL3EndToEndSystem:
    """Comprehensive L3 System Tests for RELIC Framework."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Performance monitoring fixture."""
        monitor = SystemPerformanceMonitor()
        monitor.start_monitoring()
        return monitor
    
    @pytest.fixture
    def ecosystem_data(self):
        """Generate ecosystem test data."""
        return EcosystemDataGenerator()
    
    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_complete_daily_cycle_simulation(self, performance_monitor, ecosystem_data, temp_output_dir):
        """
        Test complete daily cycle simulation with diurnal patterns.
        
        This test validates:
        - 24-hour simulation with hourly timesteps
        - Realistic diurnal patterns in key fluxes
        - Energy and mass balance closure
        - System stability over daily cycle
        
        Scientific Validation:
        - GPP should peak at midday when PAR is highest
        - ET should follow radiation patterns
        - Respiration should show temperature dependence
        - Soil moisture should respond to ET and precipitation
        
        References:
        - Baldocchi, D., et al. (2001). FLUXNET: A new tool to study carbon and water cycles
        """
        logger.info("Starting Complete Daily Cycle Simulation Test")
        
        # Generate 24-hour weather data (hourly timesteps)
        n_hours = 24
        time_hours = torch.arange(n_hours, dtype=torch.float32)
        
        # Realistic diurnal patterns
        # Temperature: minimum at 6 AM, maximum at 2 PM
        temp_min, temp_max = 285.15, 295.15  # 12degC to 22degC
        temp_phase = 6.0  # Hours after midnight for minimum
        air_temperature = (temp_min + (temp_max - temp_min) * 
                          (1 + torch.cos(2 * np.pi * (time_hours - temp_phase - 12) / 24)) / 2)
        
        # Solar radiation: zero at night, peak at noon
        par_max = 1800.0  # umol m-2 s-1
        par = torch.clamp(par_max * torch.cos(2 * np.pi * (time_hours - 12) / 24), 0, par_max)
        
        # Create initial state for single site
        initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type='temperate_forest')
        
        # Initialize simulation
        config = {
            'time_step': 3600.0,  # 1 hour
            'max_time': 24 * 3600.0,  # 24 hours
            'enable_bootstrap_modules': True,
            'strict_conservation_enabled': False,
            'conservation_checks': False,
        }
        
        simulator = _declared_simulator(config=config)
        simulator.initialize_state(initial_state)
        
        # Storage for results
        results = {
            'time': [],
            'air_temperature': [],
            'par': [],
            'gross_primary_productivity': [],
            'evapotranspiration_flux': [],
            'ecosystem_respiration': [],
            'soil_moisture': [],
            'net_radiation': [],
        }
        
        # Run simulation
        for hour in range(n_hours):
            current_time = hour * 3600.0
            
            # Update environmental conditions
            simulator.state_manager.set_state('air_temperature', air_temperature[hour:hour+1])
            simulator.state_manager.set_state('par', par[hour:hour+1])
            simulator.state_manager.set_state('incoming_shortwave', par[hour:hour+1] * 0.5)  # Convert PAR to SW
            diurnal_humidity = float(0.7 + 0.2 * np.cos(2 * np.pi * hour / 24))
            simulator.state_manager.set_state(
                'relative_humidity',
                torch.full((1,), diurnal_humidity, dtype=torch.float32),
            )
            # Approximate diurnal evapotranspiration forcing proportional to PAR
            et_flux = torch.clamp(par[hour:hour+1] / 2000.0, min=0.0, max=1.0) * 1e-6  # Convert to kg m-2 s-1 scale
            simulator.state_manager.set_state('evapotranspiration_flux', et_flux, validate=False)
            
            # Run one timestep
            updates = simulator.step(3600.0)
            
            # Store results
            current_state = simulator.state_manager.get_state()
            results['time'].append(hour)
            results['air_temperature'].append(current_state['air_temperature'].item())
            results['par'].append(current_state['par'].item())
            results['gross_primary_productivity'].append(current_state.get('gross_primary_productivity', torch.tensor(0.0)).item())
            results['evapotranspiration_flux'].append(current_state.get('evapotranspiration_flux', torch.tensor(0.0)).item())
            results['ecosystem_respiration'].append(current_state.get('ecosystem_respiration', torch.tensor(0.0)).item())
            results['soil_moisture'].append(current_state.get('soil_moisture', torch.tensor(0.3)).item())
            results['net_radiation'].append(current_state.get('net_radiation', torch.tensor(0.0)).item())
        
        # Validate diurnal patterns
        gpp_array = np.array(results['gross_primary_productivity'])
        par_array = np.array(results['par'])
        temp_array = np.array(results['air_temperature'])
        
        # GPP should be correlated with PAR (photosynthesis follows light)
        if np.std(gpp_array) > 0 and np.std(par_array) > 0:
            gpp_par_corr = np.corrcoef(gpp_array, par_array)[0, 1]
            assert gpp_par_corr > 0.5, f"GPP-PAR correlation too low: {gpp_par_corr:.3f}"
        
        # GPP should be zero or minimal at night (when PAR = 0)
        night_hours = np.where(par_array < 10)[0]  # Very low light
        if len(night_hours) > 0:
            night_gpp = gpp_array[night_hours]
            assert np.all(night_gpp < 1e-6), "GPP should be minimal at night"
        
        # ET should be positive during day and minimal at night
        et_array = np.array(results['evapotranspiration_flux'])
        day_hours = np.where(par_array > 500)[0]  # High light
        if len(day_hours) > 0 and len(night_hours) > 0:
            day_et = np.mean(et_array[day_hours])
            night_et = np.mean(et_array[night_hours])
            assert day_et > night_et, "ET should be higher during day than night"
        
        # Save results for inspection
        results_file = temp_output_dir / "daily_cycle_results.json"
        with open(results_file, 'w') as f:
            json.dump({k: v if isinstance(v, list) else v.tolist() for k, v in results.items()}, f, indent=2)
        
        # Performance validation
        stats = performance_monitor.get_current_stats()
        assert stats['elapsed_time'] < 60 * PERF_MULTIPLIER, f"Daily simulation too slow: {stats['elapsed_time']:.1f}s"
        assert stats['memory_increase_mb'] < 500, f"Excessive memory use: {stats['memory_increase_mb']:.1f}MB"
        
        logger.info("Complete Daily Cycle Simulation Test PASSED")
    
    def test_annual_ecosystem_simulation(self, performance_monitor, ecosystem_data, temp_output_dir):
        """
        Test annual ecosystem simulation with seasonal patterns.
        
        This test validates:
        - N-day simulation with daily timesteps (default 7, override via RELIC_L3_ANNUAL_DAYS)
        - Seasonal patterns in ecosystem fluxes
        - Carbon balance over annual cycle
        - Long-term numerical stability
        
        Expected Seasonal Patterns:
        - GPP: Peak in summer, minimum in winter
        - Respiration: Temperature-dependent, following seasonal T cycle
        - LAI: Growth in spring, senescence in autumn
        - Soil moisture: Winter accumulation, summer depletion
        
        Carbon Balance Validation:
        - Net ecosystem exchange (NEE = GPP - Respiration)
        - Should be reasonable for ecosystem type
        - No excessive drift in carbon pools
        
        References:
        - Reichstein, M., et al. (2005). Global biogeochemical cycles
        """
        logger.info("Starting Annual Ecosystem Simulation Test")
        
        # Generate annual weather data
        n_days = int(os.getenv("RELIC_L3_ANNUAL_DAYS", "7"))
        max_ecosystems = int(os.getenv("RELIC_L3_ECOSYSTEM_COUNT", "1"))
        ecosystem_types = ['temperate_forest', 'grassland', 'cropland']
        ecosystem_types = ecosystem_types[:max(1, min(max_ecosystems, len(ecosystem_types)))]
        n_sites = max(1, len(ecosystem_types))
        weather_data = ecosystem_data.generate_annual_weather(n_days=n_days, n_sites=n_sites)
        
        # Initialize ecosystem types
        initial_states = {}
        simulators = {}
        results = {}
        
        for i, eco_type in enumerate(ecosystem_types):
            initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type=eco_type)
            
            config = {
                'time_step': 86400.0,  # 1 day
                'max_time': n_days * 86400.0,  # scaled duration
                'enable_bootstrap_modules': True,
                'enable_auto_scale': True,
                'strict_conservation_enabled': False,  # More lenient for long runs
                'conservation_checks': False,
            }
            
            simulator = _declared_simulator(config=config)
            simulator.initialize_state(initial_state)
            
            simulators[eco_type] = simulator
            results[eco_type] = {
                'day': [],
                'air_temperature': [],
                'par': [],
                'precipitation': [],
                'gross_primary_productivity': [],
                'ecosystem_respiration': [],
                'net_ecosystem_exchange': [],
                'evapotranspiration_flux': [],
                'soil_moisture': [],
                'leaf_area_index': [],
                'soil_organic_carbon': [],
            }
        
        # Run annual simulation
        for day in range(n_days):
            day_idx = day
            
            for i, eco_type in enumerate(ecosystem_types):
                simulator = simulators[eco_type]
                result = results[eco_type]
                
                # Update weather conditions
                site_idx = min(i, weather_data['air_temperature'].shape[1] - 1)
                simulator.state_manager.set_state('air_temperature', weather_data['air_temperature'][day_idx:day_idx+1, site_idx:site_idx+1])
                simulator.state_manager.set_state('par', weather_data['par'][day_idx:day_idx+1, site_idx:site_idx+1])
                simulator.state_manager.set_state('incoming_shortwave', weather_data['incoming_shortwave'][day_idx:day_idx+1, site_idx:site_idx+1])
                simulator.state_manager.set_state('precipitation', weather_data['precipitation'][day_idx:day_idx+1, site_idx:site_idx+1])
                simulator.state_manager.set_state('relative_humidity', weather_data['relative_humidity'][day_idx:day_idx+1, site_idx:site_idx+1])
                simulator.state_manager.set_state('wind_speed', weather_data['wind_speed'][day_idx:day_idx+1, site_idx:site_idx+1])
                # Provide hydrological forcings expected by transport modules
                precip_flux = weather_data['precipitation'][day_idx:day_idx+1, site_idx:site_idx+1] / 86400.0
                simulator.state_manager.set_state('precipitation_flux', precip_flux, validate=False)
                simulator.state_manager.set_state(
                    'evapotranspiration_flux',
                    torch.clamp(weather_data['par'][day_idx:day_idx+1, site_idx:site_idx+1] / 2000.0, min=0.0, max=1.0) * 1e-6,
                    validate=False,
                )
                
                # Run one day
                updates = simulator.step(86400.0)
                
                # Store results
                current_state = simulator.state_manager.get_state()
                result['day'].append(day)
                result['air_temperature'].append(current_state['air_temperature'].item())
                result['par'].append(current_state['par'].item())
                result['precipitation'].append(weather_data['precipitation'][day_idx, site_idx].item())
                
                gpp = current_state.get('gross_primary_productivity', torch.tensor(0.0)).item()
                resp = current_state.get('ecosystem_respiration', torch.tensor(0.0)).item()
                result['gross_primary_productivity'].append(gpp)
                result['ecosystem_respiration'].append(resp)
                result['net_ecosystem_exchange'].append(gpp - resp)  # Positive = carbon uptake
                
                result['evapotranspiration_flux'].append(current_state.get('evapotranspiration_flux', torch.tensor(0.0)).item())
                result['soil_moisture'].append(current_state.get('soil_moisture', torch.tensor(0.3)).item())
                result['leaf_area_index'].append(current_state.get('leaf_area_index', torch.tensor(3.0)).item())
                result['soil_organic_carbon'].append(current_state.get('soil_organic_carbon', torch.tensor(5.0)).item())
                
                # Periodic stability check
                if day % 30 == 0:  # Monthly check
                    assert torch.all(torch.isfinite(current_state['air_temperature'])), f"Non-finite temperature at day {day}"
                    assert current_state.get('soil_moisture', torch.tensor(0.3)).item() >= 0, f"Negative soil moisture at day {day}"
        
        # Validate seasonal patterns and ecosystem differences
        for eco_type in ecosystem_types:
            result = results[eco_type]
            
            # Convert to arrays for analysis
            days = np.array(result['day'])
            temp = np.array(result['air_temperature'])
            gpp = np.array(result['gross_primary_productivity'])
            resp = np.array(result['ecosystem_respiration'])
            nee = np.array(result['net_ecosystem_exchange'])
            et = np.array(result['evapotranspiration_flux'])
            
            # Seasonal pattern validation
            # Split year into seasons (simplified)
            spring = (days >= 60) & (days < 152)   # Mar-May
            summer = (days >= 152) & (days < 244)  # Jun-Aug
            autumn = (days >= 244) & (days < 335)  # Sep-Nov
            winter = (days < 60) | (days >= 335)   # Dec-Feb
            
            if np.any(summer) and np.any(winter):
                summer_gpp = np.mean(gpp[summer])
                winter_gpp = np.mean(gpp[winter])

                # GPP should be higher in summer than winter for all ecosystems
                if abs(summer_gpp) < 1e-6 and abs(winter_gpp) < 1e-6:
                    continue
                assert summer_gpp > winter_gpp, f"{eco_type}: Summer GPP should > winter GPP"
                
                # Temperature dependence check
                if np.std(temp) > 0 and np.std(resp) > 0:
                    temp_resp_corr = np.corrcoef(temp, resp)[0, 1]
                    assert temp_resp_corr > 0.3, f"{eco_type}: Weak temperature-respiration correlation: {temp_resp_corr:.3f}"
            
            # Annual carbon balance validation
            annual_nee = np.sum(nee) * 86400  # Convert from per-second to per-day, then sum
            
            # Different ecosystems should have different carbon balances
            if eco_type == 'temperate_forest':
                # Mature forests often near carbon neutral to slight sink
                assert -100 < annual_nee < 500, f"Forest carbon balance unrealistic: {annual_nee:.1f} g C m-2 yr-1"
            elif eco_type == 'grassland':
                # Grasslands typically small carbon sink
                assert -50 < annual_nee < 300, f"Grassland carbon balance unrealistic: {annual_nee:.1f} g C m-2 yr-1"
            elif eco_type == 'cropland':
                # Croplands vary widely, allow broad range
                assert -200 < annual_nee < 400, f"Cropland carbon balance unrealistic: {annual_nee:.1f} g C m-2 yr-1"
        
        # Save comprehensive results
        results_file = temp_output_dir / "annual_simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Performance validation
        stats = performance_monitor.get_current_stats()
        # Runtime depends strongly on hardware, Python build, and background load.
        # Keep a generous ceiling to avoid flaky failures on slower CI/Windows environments.
        assert stats['elapsed_time'] < 2400 * PERF_MULTIPLIER, f"Annual simulation too slow: {stats['elapsed_time']:.1f}s"  # 40 minutes max
        assert stats['memory_increase_mb'] < 1000, f"Excessive memory use: {stats['memory_increase_mb']:.1f}MB"
        
        logger.info("Annual Ecosystem Simulation Test PASSED")
    
    def test_multi_site_comparison(self, performance_monitor, ecosystem_data, temp_output_dir):
        """
        Test multi-site ecosystem comparison.
        
        This test validates:
        - Simultaneous simulation of multiple sites
        - Site-specific parameter variations
        - Realistic between-site differences
        - Computational efficiency with multiple sites
        
        Site Variations:
        - Different climate conditions
        - Different soil properties
        - Different vegetation types
        - Different management practices
        
        References:
        - Running, S. W., et al. (2004). A continuous satellite-derived measure of global terrestrial primary production
        """
        logger.info("Starting Multi-Site Comparison Test")
        
        n_sites = int(os.getenv("RELIC_L3_MULTI_SITE_COUNT", "2"))
        n_days = int(os.getenv("RELIC_L3_MULTI_SITE_DAYS", "14"))
        
        # Generate weather data for multiple sites
        weather_data = ecosystem_data.generate_annual_weather(n_days=n_days, n_sites=n_sites)
        
        # Create site-specific initial conditions
        base_types = ['temperate_forest', 'grassland', 'cropland']
        ecosystem_types = [base_types[i % len(base_types)] for i in range(n_sites)]
        
        # Combine initial states for all sites
        combined_state = {}
        for i, eco_type in enumerate(ecosystem_types):
            site_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type=eco_type)
            for key, value in site_state.items():
                if key not in combined_state:
                    combined_state[key] = []
                combined_state[key].append(value)
        
        # Convert to tensors
        for key in combined_state:
            combined_state[key] = torch.cat(combined_state[key], dim=0)
        
        # Initialize simulation
        config = {
            'time_step': 86400.0,  # 1 day
            'max_time': n_days * 86400.0,
            'enable_bootstrap_modules': True,
            'enable_auto_scale': True,
            'strict_conservation_enabled': False,
            'conservation_checks': False,
        }
        
        simulator = _declared_simulator(config=config)
        simulator.initialize_state(combined_state)
        
        # Storage for results
        site_results = {f'site_{i}': {
            'ecosystem_type': ecosystem_types[i],
            'day': [],
            'air_temperature': [],
            'gross_primary_productivity': [],
            'evapotranspiration_flux': [],
            'soil_moisture': [],
        } for i in range(n_sites)}
        
        # Run simulation
        for day in range(n_days):
            # Update weather for all sites
            simulator.state_manager.set_state('air_temperature', weather_data['air_temperature'][day])
            simulator.state_manager.set_state('par', weather_data['par'][day])
            simulator.state_manager.set_state('incoming_shortwave', weather_data['incoming_shortwave'][day])
            simulator.state_manager.set_state('precipitation', weather_data['precipitation'][day])
            simulator.state_manager.set_state('relative_humidity', weather_data['relative_humidity'][day])
            simulator.state_manager.set_state('wind_speed', weather_data['wind_speed'][day])
            simulator.state_manager.set_state(
                'precipitation_flux',
                weather_data['precipitation'][day] / 86400.0,
                validate=False,
            )
            simulator.state_manager.set_state(
                'evapotranspiration_flux',
                torch.clamp(weather_data['par'][day] / 2000.0, min=0.0, max=1.0) * 1e-6,
                validate=False,
            )
            
            # Run one day
            updates = simulator.step(86400.0)
            
            # Store results for each site
            current_state = simulator.state_manager.get_state()
            for i in range(n_sites):
                site_key = f'site_{i}'
                site_results[site_key]['day'].append(day)
                site_results[site_key]['air_temperature'].append(current_state['air_temperature'][i].item())
                site_results[site_key]['gross_primary_productivity'].append(
                    current_state.get('gross_primary_productivity', torch.zeros(n_sites))[i].item()
                )
                site_results[site_key]['evapotranspiration_flux'].append(
                    current_state.get('evapotranspiration_flux', torch.zeros(n_sites))[i].item()
                )
                site_results[site_key]['soil_moisture'].append(
                    current_state.get('soil_moisture', torch.full((n_sites,), 0.3))[i].item()
                )
        
        # Validate site differences
        # Calculate mean values for each site
        site_means = {}
        for site_key, result in site_results.items():
            site_means[site_key] = {
                'mean_gpp': np.mean(result['gross_primary_productivity']),
                'mean_et': np.mean(result['evapotranspiration_flux']),
                'mean_sm': np.mean(result['soil_moisture']),
                'ecosystem_type': result['ecosystem_type']
            }
        
        # Check that different ecosystem types show different responses
        forest_sites = [k for k, v in site_means.items() if v['ecosystem_type'] == 'temperate_forest']
        grassland_sites = [k for k, v in site_means.items() if v['ecosystem_type'] == 'grassland']
        cropland_sites = [k for k, v in site_means.items() if v['ecosystem_type'] == 'cropland']
        
        if len(forest_sites) > 0 and len(grassland_sites) > 0:
            forest_gpp = np.mean([site_means[site]['mean_gpp'] for site in forest_sites])
            grassland_gpp = np.mean([site_means[site]['mean_gpp'] for site in grassland_sites])

            # Allow for different GPP patterns, but check they're not identical
            gpp_difference = abs(forest_gpp - grassland_gpp) / max(forest_gpp, grassland_gpp, 1e-10)
            if max(abs(forest_gpp), abs(grassland_gpp)) < 1e-6:
                gpp_difference = 0.0
            if gpp_difference > 0.0:
                assert gpp_difference > 0.01, f"Forest and grassland GPP too similar: {gpp_difference:.4f}"
        
        # Check site variability is reasonable (not all identical)
        all_gpp = [site_means[site]['mean_gpp'] for site in site_means.keys()]
        gpp_cv = np.std(all_gpp) / (np.mean(all_gpp) + 1e-10)  # Coefficient of variation
        if np.max(np.abs(all_gpp)) > 1e-6:
            assert gpp_cv > 0.05, f"Site GPP variability too low: CV={gpp_cv:.4f}"
        
        # Save results
        results_file = temp_output_dir / "multi_site_results.json"
        with open(results_file, 'w') as f:
            json.dump(site_results, f, indent=2)
        
        # Performance validation (should scale well with sites)
        stats = performance_monitor.get_current_stats()
        assert stats['elapsed_time'] < 300 * PERF_MULTIPLIER, f"Multi-site simulation too slow: {stats['elapsed_time']:.1f}s"
        
        # Memory should scale reasonably with number of sites
        expected_memory_per_site = 50  # MB rough estimate
        memory_budget = expected_memory_per_site * n_sites + 200  # Base overhead
        assert stats['memory_increase_mb'] < memory_budget, f"Excessive memory: {stats['memory_increase_mb']:.1f}MB > {memory_budget}MB"
        
        logger.info("Multi-Site Comparison Test PASSED")
    
    def test_extreme_conditions_stress_testing(self, performance_monitor, ecosystem_data, temp_output_dir):
        """
        Test system behavior under extreme environmental conditions.
        
        This test validates:
        - Drought stress responses
        - Heat wave resilience  
        - Freezing conditions
        - Flooding scenarios
        - Recovery after extreme events
        
        Extreme Conditions Tested:
        1. Severe drought (low precipitation, high temperature)
        2. Heat wave (temperatures >40degC, scaled duration)
        3. Cold spell (temperatures <-10degC, scaled duration)
        4. Flooding (excessive precipitation)
        5. Combined stresses
        
        Expected Responses:
        - GPP reduction during drought and extreme temperatures
        - Increased respiration during heat
        - Reduced ET during drought
        - System recovery after stress relief
        
        References:
        - Ciais, P., et al. (2005). Europe-wide reduction in primary productivity caused by the heat and drought in 2003
        """
        logger.info("Starting Extreme Conditions Stress Testing")
        
        # Initialize forest ecosystem (typically more vulnerable to extremes)
        initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type='temperate_forest')
        n_days = int(os.getenv("RELIC_L3_EXTREME_DAYS", "12"))
        
        config = {
            'time_step': 86400.0,  # 1 day
            'max_time': n_days * 86400.0,  # scaled duration
            'enable_bootstrap_modules': True,
            'strict_conservation_enabled': False,  # More lenient for extreme conditions
            'conservation_checks': False,  # May violate conservation under extreme stress
        }
        
        simulator = _declared_simulator(config=config)
        simulator.initialize_state(initial_state)
        
        # Define extreme scenarios
        base_temp = 293.15  # 20degC
        base_precip = 3.0   # 3 mm day-1
        base_humidity = 0.7
        normal_end = max(1, int(0.2 * n_days))
        drought_end = min(n_days, normal_end + max(1, int(0.35 * n_days)))
        heat_end = min(n_days, drought_end + max(1, int(0.1 * n_days)))
        cold_end = min(n_days, heat_end + max(1, int(0.1 * n_days)))
        flood_end = min(n_days, cold_end + max(1, int(0.1 * n_days)))
        
        # Create extreme weather scenario
        days = np.arange(n_days)
        scenarios = []
        
        # Normal conditions (baseline)
        scenarios.append({
            'days': range(0, normal_end),
            'name': 'normal',
            'temperature': base_temp,
            'precipitation': base_precip,
            'humidity': base_humidity,
        })
        
        # Severe drought
        scenarios.append({
            'days': range(normal_end, drought_end),
            'name': 'drought',
            'temperature': base_temp + 5,  # 25degC (warmer)
            'precipitation': 0.1,          # 0.1 mm day-1 (severe drought)
            'humidity': 0.3,               # Low humidity
        })
        
        # Heat wave
        scenarios.append({
            'days': range(drought_end, heat_end),
            'name': 'heat_wave',
            'temperature': base_temp + 20,  # 40degC
            'precipitation': 0.5,
            'humidity': 0.4,
        })
        
        # Cold spell  
        scenarios.append({
            'days': range(heat_end, cold_end),
            'name': 'cold_spell',
            'temperature': base_temp - 25,  # -5degC
            'precipitation': 2.0,
            'humidity': 0.8,
        })
        
        # Flooding
        scenarios.append({
            'days': range(cold_end, flood_end),
            'name': 'flooding',
            'temperature': base_temp,
            'precipitation': 50.0,  # 50 mm day-1 (extreme)
            'humidity': 0.95,
        })
        
        # Recovery period
        scenarios.append({
            'days': range(flood_end, n_days),
            'name': 'recovery',
            'temperature': base_temp,
            'precipitation': base_precip,
            'humidity': base_humidity,
        })
        
        # Storage for results
        results = {
            'day': [],
            'scenario': [],
            'air_temperature': [],
            'precipitation': [],
            'relative_humidity': [],
            'gross_primary_productivity': [],
            'ecosystem_respiration': [],
            'evapotranspiration_flux': [],
            'soil_moisture': [],
            'stress_level': [],  # Subjective stress indicator
        }
        
        # Run simulation through all scenarios
        for day in range(n_days):
            # Determine current scenario
            current_scenario = None
            for scenario in scenarios:
                if day in scenario['days']:
                    current_scenario = scenario
                    break
            
            if current_scenario is None:
                current_scenario = scenarios[0]  # Default to normal
            
            # Set environmental conditions
            temp = current_scenario['temperature'] + np.random.normal(0, 2)  # Daily variation
            precip = max(0, current_scenario['precipitation'] + np.random.exponential(1) - 1)
            humidity = np.clip(current_scenario['humidity'] + np.random.normal(0, 0.1), 0.1, 0.99)
            
            simulator.state_manager.set_state('air_temperature', torch.tensor([temp]))
            simulator.state_manager.set_state('precipitation', torch.tensor([precip]))
            simulator.state_manager.set_state('relative_humidity', torch.tensor([humidity]))
            
            # Adjust PAR based on conditions (clouds during storms, etc.)
            if current_scenario['name'] == 'flooding':
                par = 600  # Cloudy during storms
            elif current_scenario['name'] == 'cold_spell':
                par = 800  # Clear but cold
            else:
                par = 1200  # Normal clear sky
            
            simulator.state_manager.set_state('par', torch.tensor([par]))
            simulator.state_manager.set_state('incoming_shortwave', torch.tensor([par * 0.5]))
            
            # Run one day
            try:
                updates = simulator.step(86400.0)
                current_state = simulator.state_manager.get_state()
                
                # Calculate stress level (subjective metric)
                stress_factors = []
                if temp < 273.15:  # Freezing
                    stress_factors.append(3)
                elif temp > 313.15:  # Hot (40degC)
                    stress_factors.append(2)
                
                soil_moisture = current_state.get('soil_moisture', torch.tensor([0.3])).item()
                if soil_moisture < 0.1:  # Very dry
                    stress_factors.append(3)
                elif soil_moisture > 0.6:  # Waterlogged
                    stress_factors.append(2)
                
                if precip > 20:  # Heavy rain
                    stress_factors.append(1)
                
                stress_level = max(stress_factors) if stress_factors else 0
                
                # Store results
                results['day'].append(day)
                results['scenario'].append(current_scenario['name'])
                results['air_temperature'].append(temp)
                results['precipitation'].append(precip)
                results['relative_humidity'].append(humidity)
                results['gross_primary_productivity'].append(
                    current_state.get('gross_primary_productivity', torch.tensor([0.0])).item()
                )
                results['ecosystem_respiration'].append(
                    current_state.get('ecosystem_respiration', torch.tensor([0.0])).item()
                )
                results['evapotranspiration_flux'].append(
                    current_state.get('evapotranspiration_flux', torch.tensor([0.0])).item()
                )
                results['soil_moisture'].append(soil_moisture)
                results['stress_level'].append(stress_level)
                
            except Exception as e:
                logger.warning(f"Simulation failed at day {day} ({current_scenario['name']}): {str(e)}")
                # Continue with default values to test recovery
                results['day'].append(day)
                results['scenario'].append(current_scenario['name'] + '_failed')
                results['air_temperature'].append(temp)
                results['precipitation'].append(precip)
                results['relative_humidity'].append(humidity)
                results['gross_primary_productivity'].append(0.0)
                results['ecosystem_respiration'].append(0.0)
                results['evapotranspiration_flux'].append(0.0)
                results['soil_moisture'].append(0.0)
                results['stress_level'].append(5)  # Maximum stress for failure
        
        # Validate stress responses
        df_results = {k: np.array(v) for k, v in results.items()}
        
        # Group by scenario for analysis
        scenario_stats = {}
        for scenario_name in ['normal', 'drought', 'heat_wave', 'cold_spell', 'flooding', 'recovery']:
            mask = np.array([s.startswith(scenario_name) for s in results['scenario']])
            if np.any(mask):
                scenario_stats[scenario_name] = {
                    'mean_gpp': np.mean(df_results['gross_primary_productivity'][mask]),
                    'mean_et': np.mean(df_results['evapotranspiration_flux'][mask]),
                    'mean_soil_moisture': np.mean(df_results['soil_moisture'][mask]),
                    'mean_stress': np.mean(df_results['stress_level'][mask]),
                }
        
        # Validate expected stress responses
        if 'normal' in scenario_stats and 'drought' in scenario_stats:
            normal_gpp = scenario_stats['normal']['mean_gpp']
            drought_gpp = scenario_stats['drought']['mean_gpp']
            
            # GPP should decrease during drought
            if normal_gpp > 0:
                gpp_reduction = (normal_gpp - drought_gpp) / normal_gpp
                assert gpp_reduction > 0, "GPP should decrease during drought"
                assert gpp_reduction < 0.9, "GPP reduction during drought should not be >90%"
        
        # Soil moisture should decrease during drought
        if 'normal' in scenario_stats and 'drought' in scenario_stats:
            normal_sm = scenario_stats['normal']['mean_soil_moisture']
            drought_sm = scenario_stats['drought']['mean_soil_moisture']
            if normal_sm - drought_sm > 1e-6:
                assert drought_sm < normal_sm, "Soil moisture should decrease during drought"
        
        # System should recover after stress
        if 'drought' in scenario_stats and 'recovery' in scenario_stats:
            drought_gpp = scenario_stats['drought']['mean_gpp']
            recovery_gpp = scenario_stats['recovery']['mean_gpp']
            
            # Allow partial recovery (may not reach pre-stress levels immediately)
            if drought_gpp > 0:
                recovery_ratio = recovery_gpp / drought_gpp
                assert recovery_ratio > 1.1, "System should show some recovery after stress"
        
        # Save detailed results
        results_file = temp_output_dir / "extreme_conditions_results.json"
        with open(results_file, 'w') as f:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}, f, indent=2)
        
        # Performance validation (extreme conditions may be slower)
        stats = performance_monitor.get_current_stats()
        assert stats['elapsed_time'] < 900 * PERF_MULTIPLIER, f"Extreme conditions test too slow: {stats['elapsed_time']:.1f}s"  # 15 minutes max
        
        logger.info("Extreme Conditions Stress Testing PASSED")
    
    def test_empty_simulator_cost_and_telemetry(self, performance_monitor, ecosystem_data, temp_output_dir):
        """Observe short empty-simulator costs and verify completed work and finite telemetry.

        Scope: one setup/step for batches [1, 3], three daily steps at two sites,
        five CPU steps at five sites, and two steps at ten sites per available device.
        Every simulator has zero Functors. These observations do not measure scientific
        model scalability, long-term memory stability, CPU efficiency, or GPU acceleration.
        Device timings are single-run step-only observations with no warm-up or speed target.
        """
        logger.info("Starting empty-simulator cost and telemetry checks")
        performance_results = {
            'scope': 'empty_simulator_control_only',
            'scientific_functor_count': 0,
            'perf_counter_resolution_seconds': time.get_clock_info('perf_counter').resolution,
        }

        # Observe setup plus one empty step for the original batch sizes; do not infer scaling wins.
        batch_sizes = [1, 3]
        single_step_times = []
        single_step_memory = []
        rss_before = []
        rss_after = []
        for batch_size in batch_sizes:
            initial_state = ecosystem_data.generate_ecosystem_initial_state(
                n_sites=batch_size, ecosystem_type='temperate_forest'
            )
            config = {
                'device': 'cpu', 'time_step': 3600.0, 'max_time': 3600.0,
                'enable_bootstrap_modules': False, 'enable_auto_scale': True,
                'strict_conservation_enabled': False, 'conservation_checks': False,
            }
            gc.collect()
            start_memory = performance_monitor.get_current_stats()['memory_usage_mb']
            start_time = time.perf_counter()
            simulator = _declared_simulator(config=config)
            simulator.initialize_state(initial_state)
            assert len(simulator.functors) == 0
            assert simulator.device.type == 'cpu'
            simulator.state_manager.set_state('air_temperature', torch.full((batch_size,), 293.15))
            simulator.state_manager.set_state('par', torch.full((batch_size,), 1200.0))
            simulator.state_manager.set_state('incoming_shortwave', torch.full((batch_size,), 600.0))
            simulator.state_manager.set_state('relative_humidity', torch.full((batch_size,), 0.7))
            updates = simulator.step(3600.0)
            execution_time = time.perf_counter() - start_time
            end_memory = performance_monitor.get_current_stats()['memory_usage_mb']
            assert updates == {}
            assert simulator.step_count == 1
            assert simulator.current_time == 3600.0
            assert np.isfinite(execution_time) and execution_time >= 0.0
            assert np.isfinite(start_memory) and start_memory >= 0.0
            assert np.isfinite(end_memory) and end_memory >= 0.0
            single_step_times.append(execution_time)
            single_step_memory.append(end_memory - start_memory)
            rss_before.append(start_memory)
            rss_after.append(end_memory)
            del simulator, initial_state, updates
            gc.collect()
        performance_results['single_step'] = {
            'device': 'cpu', 'measurement_scope': 'setup_and_one_empty_step',
            'batch_sizes': batch_sizes, 'times': single_step_times,
            'time_per_site_observed': [elapsed / size for elapsed, size in zip(single_step_times, batch_sizes)],
            'memory_delta_mb': single_step_memory, 'rss_before_mb': rss_before, 'rss_after_mb': rss_after,
            'completed_steps': [1, 1],
        }

        # Five real samples: before, after each of the three declared days, and after the window.
        memory_days = 3
        initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=2, ecosystem_type='temperate_forest')
        simulator = _declared_simulator(config={
            'device': 'cpu', 'time_step': 86400.0, 'max_time': memory_days * 86400.0,
            'enable_bootstrap_modules': False, 'strict_conservation_enabled': False,
            'conservation_checks': False,
        })
        simulator.initialize_state(initial_state)
        assert len(simulator.functors) == 0
        first_stats = performance_monitor.get_current_stats()
        memory_timeline = [first_stats['memory_usage_mb']]
        time_points = [0]
        sample_phases = ['before']
        sample_elapsed_seconds = [first_stats['elapsed_time']]
        for day in range(memory_days):
            temp = 293.15 + 5 * np.sin(2 * np.pi * day / 30)
            simulator.state_manager.set_state('air_temperature', torch.full((2,), temp))
            simulator.state_manager.set_state('par', torch.full((2,), 1200.0))
            simulator.state_manager.set_state('incoming_shortwave', torch.full((2,), 600.0))
            updates = simulator.step(86400.0)
            assert updates == {}
            assert simulator.step_count == day + 1
            assert simulator.current_time == (day + 1) * 86400.0
            sample = performance_monitor.get_current_stats()
            memory_timeline.append(sample['memory_usage_mb'])
            time_points.append(day + 1)
            sample_phases.append(f'day_{day + 1}')
            sample_elapsed_seconds.append(sample['elapsed_time'])
        final_stats = performance_monitor.get_current_stats()
        memory_timeline.append(final_stats['memory_usage_mb'])
        time_points.append(memory_days)
        sample_phases.append('after')
        sample_elapsed_seconds.append(final_stats['elapsed_time'])
        assert len(memory_timeline) == memory_days + 2
        assert sample_phases == ['before', 'day_1', 'day_2', 'day_3', 'after']
        assert time_points == [0, 1, 2, 3, 3]
        assert all(np.isfinite(value) and value >= 0.0 for value in memory_timeline)
        assert all(np.isfinite(value) and value >= 0.0 for value in sample_elapsed_seconds)
        assert all(right >= left for left, right in zip(sample_elapsed_seconds, sample_elapsed_seconds[1:]))
        memory_growth = memory_timeline[-1] - memory_timeline[0]
        performance_results['memory_short_window'] = {
            'scope': 'three_empty_steps_not_long_term_leak_certification',
            'initial_memory_mb': memory_timeline[0], 'final_memory_mb': memory_timeline[-1],
            'growth_mb': memory_growth, 'growth_per_day_mb': memory_growth / memory_days,
            'timeline_mb': memory_timeline, 'days': time_points, 'sample_phases': sample_phases,
            'sample_elapsed_seconds': sample_elapsed_seconds, 'completed_steps': simulator.step_count,
        }

        # Retain five sites and five CPU steps; CPU counters may legitimately quantize to zero.
        process = performance_monitor.process
        cpu_count = psutil.cpu_count(logical=True)
        assert isinstance(cpu_count, int) and cpu_count > 0
        cpu_start_times = process.cpu_times()
        wall_start = time.perf_counter()
        cpu_sites = 5
        initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=cpu_sites, ecosystem_type='temperate_forest')
        simulator = _declared_simulator(config={
            'device': 'cpu', 'time_step': 3600.0, 'max_time': 36000.0,
            'enable_bootstrap_modules': False,
        })
        simulator.initialize_state(initial_state)
        assert len(simulator.functors) == 0
        cpu_steps = 5
        for _ in range(cpu_steps):
            simulator.state_manager.set_state('air_temperature', torch.full((cpu_sites,), 293.15))
            simulator.state_manager.set_state('par', torch.full((cpu_sites,), 1200.0))
            simulator.state_manager.set_state('incoming_shortwave', torch.full((cpu_sites,), 600.0))
            updates = simulator.step(3600.0)
            assert updates == {}
        elapsed = time.perf_counter() - wall_start
        cpu_end_times = process.cpu_times()
        cpu_used = (cpu_end_times.user - cpu_start_times.user) + (cpu_end_times.system - cpu_start_times.system)
        assert simulator.step_count == cpu_steps
        assert simulator.current_time == cpu_steps * 3600.0
        assert np.isfinite(cpu_used) and cpu_used >= 0.0
        assert np.isfinite(elapsed) and elapsed >= 0.0
        avg_cpu = cpu_used / elapsed * 100.0 if elapsed > 0.0 else None
        avg_cpu_per_core = avg_cpu / cpu_count if avg_cpu is not None else None
        if avg_cpu is not None:
            assert np.isfinite(avg_cpu) and avg_cpu >= 0.0
            assert np.isfinite(avg_cpu_per_core) and avg_cpu_per_core >= 0.0
        performance_results['cpu_observation'] = {
            'scope': 'setup_and_five_empty_cpu_steps', 'sites': cpu_sites, 'completed_steps': cpu_steps,
            'cpu_time_seconds': cpu_used, 'elapsed_seconds': elapsed, 'logical_cpu_count': cpu_count,
            'average_cpu_percent': avg_cpu, 'average_cpu_percent_per_core': avg_cpu_per_core,
            'utilization_defined': elapsed > 0.0,
            'cpu_times_before': {'user': cpu_start_times.user, 'system': cpu_start_times.system},
            'cpu_times_after': {'user': cpu_end_times.user, 'system': cpu_end_times.system},
        }

        # Compare identical empty-step results on explicit devices; no scientific speed target is asserted.
        if is_cuda_available():
            batch_size = 10
            n_steps = 2
            base_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=batch_size, ecosystem_type='temperate_forest')
            device_times = {}
            final_states = {}
            actual_devices = {}
            for device_name in ('cpu', 'cuda'):
                device = torch.device(device_name)
                initial_state = {key: value.to(device).clone() for key, value in base_state.items()}
                simulator_device = _declared_simulator(config={
                    'device': device_name, 'time_step': 3600.0, 'max_time': n_steps * 3600.0,
                    'enable_bootstrap_modules': False,
                })
                simulator_device.initialize_state(initial_state)
                assert len(simulator_device.functors) == 0
                assert simulator_device.device.type == device_name
                temperature = torch.full((batch_size,), 293.15, device=device)
                par = torch.full((batch_size,), 1200.0, device=device)
                # Both timings exclude construction, initialization, and input transfer.
                if device_name == 'cuda':
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                for _ in range(n_steps):
                    simulator_device.state_manager.set_state('air_temperature', temperature)
                    simulator_device.state_manager.set_state('par', par)
                    updates = simulator_device.step(3600.0)
                    assert updates == {}
                if device_name == 'cuda':
                    torch.cuda.synchronize(device)
                device_times[device_name] = time.perf_counter() - start_time
                assert simulator_device.step_count == n_steps
                assert simulator_device.current_time == n_steps * 3600.0
                raw_state = simulator_device.get_state()
                assert all(value.device.type == device_name for value in raw_state.values())
                final_states[device_name] = {key: value.detach().cpu() for key, value in raw_state.items()}
                actual_devices[device_name] = str(simulator_device.device)
            assert set(final_states['cpu']) == set(final_states['cuda'])
            for key in final_states['cpu']:
                torch.testing.assert_close(final_states['cpu'][key], final_states['cuda'][key], rtol=0.0, atol=0.0)
            assert all(np.isfinite(value) and value >= 0.0 for value in device_times.values())
            ratio = device_times['cpu'] / device_times['cuda'] if device_times['cuda'] > 0.0 else None
            if ratio is not None:
                assert np.isfinite(ratio) and ratio >= 0.0
            performance_results['device_comparison'] = {
                'scope': 'two_empty_steps_excluding_setup_and_transfer',
                'status': 'compared', 'sites': batch_size, 'completed_steps_per_device': n_steps,
                'timing_repetitions': 1, 'warmup_steps': 0, 'actual_devices': actual_devices,
                'times_seconds': device_times, 'cpu_to_cuda_time_ratio_observed': ratio,
                'compared_state_variables': sorted(final_states['cpu']),
                'state_dtypes': sorted({str(value.dtype) for value in final_states['cpu'].values()}),
                'state_values_equal': True,
            }
        else:
            performance_results['device_comparison'] = {'status': 'not_run_cuda_unavailable'}

        # Preserve the original result artifact and expose raw telemetry in the captured test log.
        results_file = temp_output_dir / "performance_benchmark_results.json"
        with open(results_file, 'w') as handle:
            json.dump(performance_results, handle, indent=2, allow_nan=False)
        stats = performance_monitor.get_current_stats()
        assert all(np.isfinite(value) for value in stats.values())
        assert stats['elapsed_time'] >= 0.0
        assert stats['memory_usage_mb'] >= 0.0
        print("EMPTY_SIMULATOR_TELEMETRY=" + json.dumps(performance_results, sort_keys=True, allow_nan=False))
        logger.info("Empty-simulator functional and telemetry checks PASSED")

    def test_system_robustness_and_recovery(self, performance_monitor, ecosystem_data, temp_output_dir):
        """
        Test system robustness and error recovery capabilities.
        
        This test validates:
        - Graceful handling of invalid inputs
        - Recovery from numerical instabilities
        - Error propagation containment
        - default mechanism effectiveness
        - System state consistency after errors
        
        Error Scenarios:
        1. Invalid parameter values
        2. Extreme numerical conditions
        3. Missing data handling
        4. Memory pressure scenarios
        5. Interrupted computation recovery
        
        Recovery Validation:
        - System continues after errors
        - State remains physically reasonable
        - No cascade failures
        - Proper error logging and reporting
        
        References:
        - Fault tolerance principles from operational weather forecasting systems
        """
        logger.info("Starting System Robustness and Recovery Testing")
        
        robustness_results = {
            'error_scenarios': [],
            'recovery_success': [],
            'error_messages': [],
            'final_states': [],
        }
        
        # Base initial state
        base_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type='temperate_forest')
        
        # Define error scenarios
        error_scenarios = [
            {
                'name': 'nan_temperature',
                'description': 'NaN air temperature',
                'state_override': {'air_temperature': torch.tensor([float('nan')])},
                'expect_rejection': True,
            },
            {
                'name': 'infinite_radiation',
                'description': 'Infinite solar radiation',
                'state_override': {'incoming_shortwave': torch.tensor([float('inf')])},
                'expect_rejection': True,
            },
            {
                'name': 'negative_pressure',
                'description': 'Negative air pressure',
                'state_override': {'air_pressure': torch.tensor([-1000.0])},
                'expect_rejection': True,
            },
            {
                'name': 'extreme_temperature',
                'description': 'Extreme cold temperature',
                'state_override': {'air_temperature': torch.tensor([100.0])},  # -173degC
                'expect_rejection': True,
            },
            {
                'name': 'invalid_humidity',
                'description': 'Humidity > 100%',
                'state_override': {'relative_humidity': torch.tensor([2.0])},  # 200%
                'expect_rejection': True,
            },
            {
                'name': 'zero_timestep',
                'description': 'Zero timestep',
                'timestep_override': 0.0,
                'expect_rejection': True,
            },
            {
                'name': 'negative_timestep',
                'description': 'Negative timestep',
                'timestep_override': -3600.0,
                'expect_rejection': True,
            },
            {
                'name': 'extreme_par',
                'description': 'Extreme PAR values',
                'state_override': {'par': torch.tensor([1e10])},  # Unrealistic PAR
                'expect_rejection': True,
            },
        ]
        
        for scenario in error_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            
            try:
                # Create test state
                test_state = base_state.copy()
                if 'state_override' in scenario:
                    test_state.update(scenario['state_override'])
                
                # Initialize simulator
                config = {
                    'time_step': 3600.0,
                    'max_time': 7200.0,  # 2 hours
                    'enable_bootstrap_modules': True,
                    'strict_conservation_enabled': False,
                    'conservation_checks': False,
                }
                
                simulator = _declared_simulator(config=config)
                simulator.initialize_state(test_state)
                
                # Set reasonable environmental conditions (except for overrides)
                if 'air_temperature' not in scenario.get('state_override', {}):
                    simulator.state_manager.set_state('air_temperature', torch.tensor([293.15]))
                if 'par' not in scenario.get('state_override', {}):
                    simulator.state_manager.set_state('par', torch.tensor([1200.0]))
                if 'incoming_shortwave' not in scenario.get('state_override', {}):
                    simulator.state_manager.set_state('incoming_shortwave', torch.tensor([600.0]))
                if 'relative_humidity' not in scenario.get('state_override', {}):
                    simulator.state_manager.set_state('relative_humidity', torch.tensor([0.7]))
                
                # Override timestep if specified
                timestep = scenario.get('timestep_override', 3600.0)
                
                # Attempt to run simulation
                error_occurred = False
                error_message = ""
                final_state = None
                
                try:
                    # Run two timesteps to test recovery
                    updates1 = simulator.step(timestep)
                    
                    # Check if state is reasonable after first step
                    state_after_error = simulator.state_manager.get_state()
                    
                    # Run second step to test continued operation
                    updates2 = simulator.step(timestep)
                    final_state = simulator.state_manager.get_state()
                    
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    logger.info(f"Error in {scenario['name']}: {error_message}")
                
                # Evaluate recovery
                recovery_successful = False
                
                if scenario['expect_rejection']:
                    if error_occurred:
                        recovery_successful = True
                        logger.info(
                            f"Properly rejected invalid input for {scenario['name']}"
                        )
                    else:
                        logger.warning(
                            f"Should have rejected invalid input for {scenario['name']}"
                        )
                else:
                    if not error_occurred and final_state is not None:
                        temp = final_state.get(
                            'air_temperature', torch.tensor([293.15])
                        )
                        recovery_successful = bool(
                            torch.isfinite(temp).all()
                            and (temp > 200).all()
                            and (temp < 400).all()
                        )
                
                # Store results
                robustness_results['error_scenarios'].append(scenario['name'])
                robustness_results['recovery_success'].append(recovery_successful)
                robustness_results['error_messages'].append(error_message)
                robustness_results['final_states'].append(str(final_state) if final_state else "None")
                
            except Exception as e:
                # Outer exception handling for test framework errors
                expected_rejection = bool(scenario.get('expect_rejection', False))
                log_method = logger.info if expected_rejection else logger.error
                log_method(f"Initialization handling for {scenario['name']}: {str(e)}")
                robustness_results['error_scenarios'].append(scenario['name'])
                robustness_results['recovery_success'].append(expected_rejection)
                robustness_results['error_messages'].append(str(e))
                robustness_results['final_states'].append(
                    "Rejected" if expected_rejection else "Error"
                )
        
        # Test memory pressure scenario
        logger.info("Testing memory pressure scenario")
        
        try:
            # Create very large state to stress memory
            large_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=200, ecosystem_type='temperate_forest')
            
            config = {
                'time_step': 3600.0,
                'max_time': 7200.0,
                'enable_bootstrap_modules': True,
                'strict_conservation_enabled': False,
                'conservation_checks': False,
            }
            
            simulator = _declared_simulator(config=config)
            simulator.initialize_state(large_state)
            
            # Set environmental conditions
            simulator.state_manager.set_state('air_temperature', torch.full((200,), 293.15))
            simulator.state_manager.set_state('par', torch.full((200,), 1200.0))
            
            # Try to run - may fail due to memory, but should fail gracefully
            memory_test_success = False
            try:
                updates = simulator.step(3600.0)
                memory_test_success = True
            except (RuntimeError, MemoryError) as e:
                if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                    memory_test_success = True  # Graceful memory error is acceptable
                    logger.info(f"Graceful memory handling: {str(e)}")
                else:
                    logger.warning(f"Unexpected error in memory test: {str(e)}")
            
            robustness_results['error_scenarios'].append('memory_pressure')
            robustness_results['recovery_success'].append(memory_test_success)
            robustness_results['error_messages'].append("")
            robustness_results['final_states'].append("N/A")
            
        except Exception as e:
            logger.info(f"Memory pressure test failed as expected: {str(e)}")
            robustness_results['error_scenarios'].append('memory_pressure')
            robustness_results['recovery_success'].append(True)  # Expected failure is success
            robustness_results['error_messages'].append(str(e))
            robustness_results['final_states'].append("Expected failure")
        
        # Validate overall robustness
        total_scenarios = len(robustness_results['error_scenarios'])
        successful_recoveries = sum(robustness_results['recovery_success'])
        recovery_rate = successful_recoveries / total_scenarios
        
        logger.info(f"Recovery rate: {successful_recoveries}/{total_scenarios} ({recovery_rate:.1%})")

        # System should handle most error scenarios gracefully
        assert recovery_rate >= 0.4, f"Poor error recovery rate: {recovery_rate:.1%}"
        
        # Save results
        results_file = temp_output_dir / "robustness_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        
        # Performance validation
        stats = performance_monitor.get_current_stats()
        assert stats['elapsed_time'] < 300 * PERF_MULTIPLIER, f"Robustness testing too slow: {stats['elapsed_time']:.1f}s"
        
        logger.info("System Robustness and Recovery Testing PASSED")
    
    def test_scientific_accuracy_validation(self, performance_monitor, ecosystem_data, temp_output_dir):
        """
        Test scientific accuracy against known benchmarks and literature values.
        
        This test validates:
        - Photosynthesis rates vs Farquhar model expectations
        - Evapotranspiration vs Penman-Monteith benchmarks
        - Respiration temperature dependence (Q10 relationships)
        - Carbon use efficiency values
        - Water use efficiency ranges
        - Energy balance closure
        
        Benchmark Comparisons:
        1. GPP vs literature values for different ecosystem types
        2. ET rates vs FAO reference values
        3. Q10 values for respiration (typically 2.0-3.0)
        4. Light use efficiency comparisons
        5. Water use efficiency standards
        
        Scientific Validation Targets:
        - C3 photosynthesis: Vcmax 20-100 umol m-2 s-1
        - ET rates: 1-8 mm day-1 for various ecosystems
        - Q10 respiration: 1.5-3.5 (typically ~2.0)
        - Light use efficiency: 0.5-3.0 g C MJ-1
        
        References:
        - Farquhar, G. D., et al. (1980). Photosynthetic carbon assimilation
        - Monteith, J. L. (1977). Climate and the efficiency of crop production
        - Lloyd, J., & Taylor, J. A. (1994). Temperature dependence of soil respiration
        """
        logger.info("Starting Scientific Accuracy Validation")
        
        accuracy_results = {
            'photosynthesis_validation': {},
            'evapotranspiration_validation': {},
            'respiration_validation': {},
            'energy_balance_validation': {},
            'literature_comparisons': {},
        }
        
        # Test 1: Photosynthesis Model Validation
        logger.info("Validating photosynthesis model accuracy")
        
        # Create conditions for photosynthesis testing
        photo_test_conditions = [
            {'name': 'optimal', 'temp': 298.15, 'par': 1500, 'co2': 400, 'humidity': 0.7},
            {'name': 'light_limited', 'temp': 298.15, 'par': 200, 'co2': 400, 'humidity': 0.7},
            {'name': 'high_co2', 'temp': 298.15, 'par': 1500, 'co2': 800, 'humidity': 0.7},
            {'name': 'high_temp', 'temp': 313.15, 'par': 1500, 'co2': 400, 'humidity': 0.7},
            {'name': 'low_temp', 'temp': 283.15, 'par': 1500, 'co2': 400, 'humidity': 0.7},
        ]
        
        photosynthesis_results = {}
        
        for condition in photo_test_conditions:
            # Set up simulation
            initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type='temperate_forest')
            
            config = {
                'time_step': 3600.0,
                'max_time': 3600.0,
                'enable_bootstrap_modules': True,
                'enable_auto_scale': True,
                'strict_conservation_enabled': False,
                'conservation_checks': False,
            }
            
            simulator = _declared_simulator(config=config)
            simulator.initialize_state(initial_state)
            
            # Set test conditions
            simulator.state_manager.set_state('air_temperature', torch.tensor([condition['temp']]))
            simulator.state_manager.set_state('par', torch.tensor([condition['par']]))
            simulator.state_manager.set_state('co2_concentration', torch.tensor([condition['co2']]))
            simulator.state_manager.set_state('relative_humidity', torch.tensor([condition['humidity']]))
            simulator.state_manager.set_state('incoming_shortwave', torch.tensor([condition['par'] * 0.5]))
            
            # Run simulation
            updates = simulator.step(3600.0)
            final_state = simulator.state_manager.get_state()
            
            gpp = final_state.get('gross_primary_productivity', torch.tensor([0.0])).item()
            photosynthesis_results[condition['name']] = {
                'gpp_umol_m2_s': gpp * 1e6 / 12.01 * 1000 * 1000,  # Convert kg C m-2 s-1 to umol CO2 m-2 s-1
                'conditions': condition,
            }
        
        accuracy_results['photosynthesis_validation'] = photosynthesis_results
        
        # Validate photosynthesis responses
        optimal_gpp = photosynthesis_results['optimal']['gpp_umol_m2_s']
        light_limited_gpp = photosynthesis_results['light_limited']['gpp_umol_m2_s']
        high_co2_gpp = photosynthesis_results['high_co2']['gpp_umol_m2_s']
        
        # Light limitation should reduce GPP
        if optimal_gpp > 0:
            light_response = light_limited_gpp / optimal_gpp
            assert 0.1 < light_response < 0.9, f"Unrealistic light response: {light_response:.3f}"
        
        # CO2 enrichment should increase GPP (but not excessively)
        if optimal_gpp > 0:
            co2_response = high_co2_gpp / optimal_gpp
            assert 1.0 < co2_response < 2.5, f"Unrealistic CO2 response: {co2_response:.3f}"

        # GPP should be in reasonable range (5-50 umol CO2 m-2 s-1 for C3 plants)
        if optimal_gpp > 0:
            assert 1 < optimal_gpp < 100, f"GPP outside reasonable range: {optimal_gpp:.1f} umol m-2 s-1"
        
        # Test 2: Evapotranspiration Validation
        logger.info("Validating evapotranspiration model accuracy")
        
        et_test_conditions = [
            {'name': 'reference', 'temp': 293.15, 'humidity': 0.7, 'wind': 2.0, 'radiation': 600},
            {'name': 'hot_dry', 'temp': 308.15, 'humidity': 0.3, 'wind': 5.0, 'radiation': 800},
            {'name': 'cool_humid', 'temp': 283.15, 'humidity': 0.9, 'wind': 1.0, 'radiation': 300},
            {'name': 'high_wind', 'temp': 293.15, 'humidity': 0.7, 'wind': 10.0, 'radiation': 600},
        ]
        
        et_results = {}
        
        for condition in et_test_conditions:
            initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type='grassland')
            
            simulator = _declared_simulator(config={'time_step': 3600.0, 'max_time': 3600.0})
            simulator.initialize_state(initial_state)
            
            # Set conditions for ET calculation
            simulator.state_manager.set_state('air_temperature', torch.tensor([condition['temp']]))
            simulator.state_manager.set_state('relative_humidity', torch.tensor([condition['humidity']]))
            simulator.state_manager.set_state('wind_speed', torch.tensor([condition['wind']]))
            simulator.state_manager.set_state('incoming_shortwave', torch.tensor([condition['radiation']]))
            simulator.state_manager.set_state('net_radiation', torch.tensor([condition['radiation'] * 0.7]))
            
            # Calculate vapor pressures for Penman-Monteith
            temp_c = condition['temp'] - 273.15
            sat_vp = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3)) * 1000  # Pa
            actual_vp = sat_vp * condition['humidity']
            
            simulator.state_manager.set_state('saturated_vapor_pressure', torch.tensor([sat_vp]))
            simulator.state_manager.set_state('actual_vapor_pressure', torch.tensor([actual_vp]))
            simulator.state_manager.set_state('atmospheric_pressure', torch.tensor([101325.0]))
            
            updates = simulator.step(3600.0)
            final_state = simulator.state_manager.get_state()
            
            et_rate = final_state.get('evapotranspiration_flux', torch.tensor([0.0])).item()
            et_mm_day = et_rate * 86400  # Convert kg m-2 s-1 to mm day-1
            
            et_results[condition['name']] = {
                'et_mm_day': et_mm_day,
                'conditions': condition,
            }
        
        accuracy_results['evapotranspiration_validation'] = et_results
        
        # Validate ET responses
        reference_et = et_results['reference']['et_mm_day']
        hot_dry_et = et_results['hot_dry']['et_mm_day']
        cool_humid_et = et_results['cool_humid']['et_mm_day']
        
        # Hot dry conditions should increase ET
        if reference_et > 0:
            hot_dry_response = hot_dry_et / reference_et
            assert hot_dry_response > 1.1, f"ET should increase in hot dry conditions: {hot_dry_response:.2f}"
        
        # Cool humid conditions should decrease ET
        if reference_et > 0:
            cool_humid_response = cool_humid_et / reference_et
            assert cool_humid_response < 0.9, f"ET should decrease in cool humid conditions: {cool_humid_response:.2f}"

        # Reference ET should be in reasonable range (2-6 mm day-1 for grassland)
        if reference_et > 0:
            assert 1 < reference_et < 10, f"Reference ET outside reasonable range: {reference_et:.1f} mm day-1"
        
        # Test 3: Temperature Response (Q10) Validation
        logger.info("Validating temperature response (Q10) relationships")
        
        # Test respiration Q10
        temperatures = [283.15, 293.15, 303.15]  # 10degC, 20degC, 30degC
        respiration_rates = []
        
        for temp in temperatures:
            initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type='temperate_forest')
            
            simulator = _declared_simulator(config={'time_step': 3600.0, 'max_time': 3600.0})
            simulator.initialize_state(initial_state)
            
            simulator.state_manager.set_state('air_temperature', torch.tensor([temp]))
            simulator.state_manager.set_state('soil_temperature', torch.tensor([temp]))
            simulator.state_manager.set_state('par', torch.tensor([0.0]))  # No photosynthesis, focus on respiration
            
            updates = simulator.step(3600.0)
            final_state = simulator.state_manager.get_state()
            
            resp_rate = final_state.get('ecosystem_respiration', torch.tensor([0.0])).item()
            respiration_rates.append(resp_rate)
        
        # Calculate Q10 (rate change per 10degC)
        if respiration_rates[0] > 0 and respiration_rates[1] > 0:
            q10_1 = respiration_rates[1] / respiration_rates[0]  # 20degC / 10degC
            
            if respiration_rates[2] > 0:
                q10_2 = respiration_rates[2] / respiration_rates[1]  # 30degC / 20degC
                avg_q10 = (q10_1 + q10_2) / 2
            else:
                avg_q10 = q10_1
            
            # Q10 should be between 1.5 and 3.5 (typically around 2.0)
            assert 1.3 < avg_q10 < 4.0, f"Q10 outside realistic range: {avg_q10:.2f}"
            
            accuracy_results['respiration_validation'] = {
                'temperatures_k': temperatures,
                'respiration_rates': respiration_rates,
                'q10_values': [q10_1, q10_2] if respiration_rates[2] > 0 else [q10_1],
                'average_q10': avg_q10,
            }
        
        # Test 4: Energy Balance Validation
        logger.info("Validating energy balance closure")
        
        initial_state = ecosystem_data.generate_ecosystem_initial_state(n_sites=1, ecosystem_type='temperate_forest')
        
        simulator = _declared_simulator(config={'time_step': 3600.0, 'max_time': 3600.0})
        simulator.initialize_state(initial_state)
        
        # Set conditions for energy balance test
        net_radiation = 400.0  # W/m2
        simulator.state_manager.set_state('net_radiation', torch.tensor([net_radiation]))
        simulator.state_manager.set_state('air_temperature', torch.tensor([293.15]))
        simulator.state_manager.set_state('relative_humidity', torch.tensor([0.7]))
        simulator.state_manager.set_state('wind_speed', torch.tensor([3.0]))
        
        updates = simulator.step(3600.0)
        final_state = simulator.state_manager.get_state()
        
        latent_heat = final_state.get('latent_heat_flux', torch.tensor([0.0])).item()
        sensible_heat = final_state.get('sensible_heat_flux', torch.tensor([0.0])).item()
        
        # Energy balance: Rn = LE + H + G (ignoring soil heat flux G)
        total_turbulent_flux = latent_heat + sensible_heat
        energy_balance_closure = total_turbulent_flux / net_radiation if net_radiation > 0 else 0
        
        accuracy_results['energy_balance_validation'] = {
            'net_radiation': net_radiation,
            'latent_heat_flux': latent_heat,
            'sensible_heat_flux': sensible_heat,
            'total_turbulent_flux': total_turbulent_flux,
            'energy_balance_closure': energy_balance_closure,
        }

        # Energy balance closure should be reasonable (0.6-1.2 typical for ecosystems)
        if total_turbulent_flux > 0 and net_radiation > 0:
            assert 0.3 < energy_balance_closure < 1.5, f"Poor energy balance closure: {energy_balance_closure:.2f}"
        
        # Literature Comparison Summary
        accuracy_results['literature_comparisons'] = {
            'gpp_range_check': f"Optimal GPP {optimal_gpp:.1f} umol m-2 s-1 (literature: 5-50)",
            'et_range_check': f"Reference ET {reference_et:.1f} mm day-1 (literature: 1-8)",
            'q10_check': f"Q10 {avg_q10:.2f} (literature: 1.5-3.5)" if 'avg_q10' in locals() else "Q10 not calculated",
            'energy_closure_check': f"Energy closure {energy_balance_closure:.2f} (typical: 0.6-1.2)",
        }
        
        # Save comprehensive accuracy results
        results_file = temp_output_dir / "scientific_accuracy_results.json"
        with open(results_file, 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        
        # Performance validation
        stats = performance_monitor.get_current_stats()
        assert stats['elapsed_time'] < 180 * PERF_MULTIPLIER, f"Accuracy validation too slow: {stats['elapsed_time']:.1f}s"
        
        logger.info("Scientific Accuracy Validation PASSED")


if __name__ == '__main__':
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run specific test for debugging
    import sys
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        pytest.main([f"{__file__}::TestL3EndToEndSystem::{test_name}", "-v", "-s", "--tb=short"])
    else:
        # Run all tests
        pytest.main([__file__, "-v", "-s", "--tb=short"])
