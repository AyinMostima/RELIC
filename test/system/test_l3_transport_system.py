#!/usr/bin/env python3
"""
L3 System Tests for Transport Mechanisms

This module provides comprehensive L3 (system-level) testing for transport
mechanisms across atmosphere and hydrosphere domains. Tests focus on 
long-term stability, multi-process coupling, and realistic scenarios.

Test Coverage:
    - Long-term transport stability (24+ hours)
    - Multi-domain coupling under realistic conditions
    - Parameter sensitivity and uncertainty propagation
    - Performance benchmarking and scalability
    - Extreme condition handling and robustness
    - Scientific validation against known solutions

Scientific Validation:
    - Comparison with analytical solutions where available
    - Verification against field measurement patterns
    - Cross-validation with established models
    - Physical constraint verification (mass, energy, momentum)

Test Structure:
    Level 3: Full system validation
    - Multi-day simulation stability
    - Realistic environmental forcing
    - Multi-scale coupling verification
    - Performance and scalability testing
    - Robustness under extreme conditions
    
References:
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology
    - Wesely, M.L. (1989). Parameterization of surface resistances
    - van Genuchten, M.T. (1980). Closed-form equation for hydraulic conductivity
"""

import sys
import os
import pytest
import numpy as np
import torch
import time
from typing import Dict, Any, Tuple, Optional

# Add relic to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import transport modules
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
    EddyDiffusionBLMFunctor, EddyDiffusionConfig
)
from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionAggregator,
    DryDepositionConfig,
)
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport
)
from relic.core.state import UpdateItem


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestL3TransportSystem:
    """System-level tests for transport mechanisms."""
    
    @pytest.fixture
    def parameter_override(self):
        """Helper to override ParameterManager values and restore after tests."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        original_specs: Dict[str, Tuple[Any, Optional[str]]] = {}

        def _apply(overrides: Dict[str, Tuple[Any, Optional[str], Optional[bool]]]) -> None:
            for key, payload in overrides.items():
                if len(payload) == 3:
                    value, units, validate = payload
                else:
                    value, units = payload
                    validate = True
                spec = manager.get_parameter(key)
                if spec is None:
                    raise KeyError(f"Missing parameter: {key}")
                if key not in original_specs:
                    original_specs[key] = (spec.value, spec.units)
                manager.set_value(key, value, units=units, validate=bool(validate), notify=False)

        yield _apply

        for key, (value, units) in original_specs.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)

    @pytest.fixture
    def percolation_overrides(self, parameter_override):
        """Apply percolation overrides for realistic system tests."""
        overrides = {
            "hydrosphere.transport.vertical_percolation.enable_temperature_effects": (True, "dimensionless", True),
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (5e-6, "m s-1", True),
            "hydrosphere.soil_moisture.porosity": (0.41, "m3 m-3", True),
            "hydrosphere.soil_moisture.residual_water_content": (0.065, "m3 m-3", True),
            "hydrosphere.soil_moisture.field_capacity": (0.20, "m3 m-3", True),
            "hydrosphere.soil_moisture.wilting_point": (0.095, "m3 m-3", True),
        }
        parameter_override(overrides)
        return overrides

    @pytest.fixture
    def realistic_transport_config(self, percolation_overrides):
        """Realistic transport configuration for system testing."""
        eddy_config = EddyDiffusionConfig(
            enabled=True,
        )
        
        deposition_config = DryDepositionConfig(enabled=True)
        
        return eddy_config, deposition_config
    
    @pytest.fixture
    def diurnal_environmental_forcing(self):
        """Create realistic diurnal environmental forcing data."""
        # 48-hour simulation with hourly time steps
        n_hours = 48
        hours = np.arange(n_hours)
        
        # Diurnal temperature cycle in K.
        temp_cycle = 293.15 + 10 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
        
        # Solar radiation (W m-2) peaks at noon.
        solar_cycle = np.maximum(0, 800 * np.sin(np.pi * (hours % 24) / 24))
        
        # Wind speed (m/s) - higher during day
        wind_cycle = 3.0 + 2.0 * np.sin(2 * np.pi * hours / 24)
        
        # Relative humidity (%) - higher at night
        rh_cycle = 0.6 + 0.2 * np.sin(2 * np.pi * hours / 24 + np.pi)
        
        # PBL height - shallow at night, deep during day
        pbl_cycle = 200 + 800 * np.maximum(0, np.sin(np.pi * (hours % 24) / 24))
        
        # Soil temperature lags air temperature
        soil_temp_cycle = temp_cycle - 2.0 * np.sin(2 * np.pi * hours / 24 - np.pi)
        
        # Create batch dimension for spatial variability
        batch_size = 4
        forcing_data = []
        rng = np.random.default_rng(0)
        
        for i, hour in enumerate(hours):
            # Add spatial variability
            spatial_noise = rng.normal(0, 0.1, batch_size)
            
            hourly_state = {
                'air_temperature': torch.tensor(temp_cycle[i] + spatial_noise),
                'shortwave_down': torch.tensor(solar_cycle[i] * (1 + spatial_noise * 0.1)),
                'wind_speed': torch.tensor(np.maximum(0.5, wind_cycle[i] + spatial_noise)),
                'relative_humidity': torch.clamp(
                    torch.tensor(rh_cycle[i] + spatial_noise * 0.1), 0.3, 0.95
                ),
                'pbl_height': torch.tensor(pbl_cycle[i] + spatial_noise * 50),
                'soil_temperature': torch.tensor(soil_temp_cycle[i] + spatial_noise * 0.5),
                'air_pressure': torch.tensor(101325.0 + spatial_noise * 100),
            }
            
            # Environmental diagnostics
            hourly_environment = {
                'friction_velocity': hourly_state['wind_speed'] * 0.1,
                'monin_obukhov_length': torch.tensor(
                    -200.0 if solar_cycle[i] > 100 else 100.0  # Unstable/stable
                ),
                'aerodynamic_resistance': 50.0 / (hourly_state['wind_speed'] + 1.0),
                'sensible_heat_flux': torch.tensor(
                    solar_cycle[i] * 0.3 if solar_cycle[i] > 0 else -50.0
                )
            }
            
            forcing_data.append((hourly_state, hourly_environment))
        
        return forcing_data
    
    @pytest.fixture
    def initial_realistic_state(self):
        """Initial state with realistic atmospheric and soil conditions."""
        batch_size = 4
        
        # Realistic atmospheric concentrations (rural background)
        return {
            # Atmospheric species [mol/mol]
            'no_mixing_ratio': torch.tensor([0.5e-9, 0.3e-9, 0.8e-9, 0.2e-9]),  # Rural NO
            'no2_mixing_ratio': torch.tensor([2.0e-9, 1.5e-9, 3.0e-9, 1.0e-9]), # Rural NO2
            'o3_mixing_ratio': torch.tensor([40e-9, 45e-9, 35e-9, 50e-9]),      # Background O3
            'isoprene_mixing_ratio': torch.tensor([1.0e-9, 1.5e-9, 0.8e-9, 2.0e-9]), # Forest isoprene
            'so2_mixing_ratio': torch.tensor([0.5e-9, 0.3e-9, 0.7e-9, 0.2e-9]), # Background SO2
            
            # Soil water state
            'soil_moisture': torch.tensor([0.25, 0.22, 0.28, 0.20]),
            'soil_layer_thickness': torch.tensor([1.0, 1.5, 0.8, 1.2]),
            
            # Physical state  
            'air_temperature': torch.tensor([293.15, 295.15, 291.15, 296.15]),
            'soil_temperature': torch.tensor([291.15, 293.15, 289.15, 294.15]),
            'air_pressure': torch.tensor([101325., 101300., 101350., 101275.]),
            'relative_humidity': torch.tensor([0.70, 0.65, 0.75, 0.60]),
            'pbl_height': torch.tensor([800., 1000., 600., 1200.]),
            'aerodynamic_resistance': torch.tensor([30., 25., 35., 20.]),
            'friction_velocity': torch.tensor([0.3, 0.35, 0.25, 0.4]),
            'monin_obukhov_length': torch.tensor([-100., -120., -80., -150.]),
        }
    
    def test_long_term_transport_stability(self, 
                                         realistic_transport_config,
                                         diurnal_environmental_forcing,
                                         initial_realistic_state):
        """Test transport system stability over 48-hour period."""
        
        print("--- L3-1: Long-term Transport Stability (48h) ---")
        
        eddy_config, deposition_config = realistic_transport_config
        
        # Initialize transport mechanisms
        eddy_transport = EddyDiffusionBLMFunctor(
            name="system_eddy", transport_config=eddy_config
        )
        dry_deposition = DryDepositionAggregator(
            name="system_deposition", deposition_config=deposition_config
        )
        soil_transport = VerticalPercolationTransport(
            name="system_percolation"
        )
        
        state = initial_realistic_state.copy()
        dt = 3600.0  # 1 hour time step
        
        # Track key variables over time
        time_series = {
            'time': [],
            'no2_conc': [],
            'o3_conc': [],
            'soil_moisture': [],
            'drainage_flux': [],
            'deposition_increment_no2': [],
        }
        
        print(f"Initial state: NO2={state['no2_mixing_ratio'].mean():.2e}, "
              f"O3={state['o3_mixing_ratio'].mean():.2e}, "
              f"Soil H2O={state['soil_moisture'].mean():.3f}")
        
        # Run 48-hour simulation
        for i, (hourly_state, hourly_environment) in enumerate(diurnal_environmental_forcing):
            current_time = i * dt
            
            # Update environmental conditions
            for key, value in hourly_state.items():
                if key in state:
                    state[key] = value
            state['friction_velocity'] = hourly_environment['friction_velocity']
            state['monin_obukhov_length'] = hourly_environment[
                'monin_obukhov_length'
            ]
            state['aerodynamic_resistance'] = hourly_environment[
                'aerodynamic_resistance'
            ]

            # Cover the full hour with CFL-stable 240 s Eddy steps.
            eddy_dt = 240.0
            for substep in range(int(dt / eddy_dt)):
                eddy_result = eddy_transport.forward(
                    state=state,
                    time=current_time + substep * eddy_dt,
                    dt=eddy_dt,
                )
                for species in ['no', 'no2', 'o3', 'isoprene']:
                    species_key = f'{species}_mixing_ratio'
                    state[species_key] = state[species_key] + _unwrap_update(
                        eddy_result[species_key]
                    )

            # Dry deposition uses the Eddy-updated state.
            depo_result = dry_deposition.forward(
                state=state,
                time=current_time,
                dt=dt,
            )
            for species in ['no', 'no2', 'o3', 'so2']:
                species_key = f'{species}_mixing_ratio'
                state[species_key] = state[species_key] + _unwrap_update(
                    depo_result[species_key]
                )

            soil_result = soil_transport.forward(
                state=state, time=current_time, dt=dt
            )
            drainage_value = _unwrap_update(soil_result['drainage_flux'])
            state['soil_moisture'] = state['soil_moisture'] - (
                drainage_value * dt / (1000.0 * state['soil_layer_thickness'])
            )
            
            # Record time series data
            time_series['time'].append(current_time)
            time_series['no2_conc'].append(state['no2_mixing_ratio'].mean().item())
            time_series['o3_conc'].append(state['o3_mixing_ratio'].mean().item())
            time_series['soil_moisture'].append(state['soil_moisture'].mean().item())
            time_series['drainage_flux'].append(drainage_value.mean().item())
            time_series['deposition_increment_no2'].append(
                _unwrap_update(depo_result['no2_mixing_ratio']).mean().item()
            )
            
            # Check for numerical stability every 6 hours
            if i % 6 == 0:
                self._verify_state_stability(state, i)
                
        print(f"Final state: NO2={state['no2_mixing_ratio'].mean():.2e}, "
              f"O3={state['o3_mixing_ratio'].mean():.2e}, "
              f"Soil H2O={state['soil_moisture'].mean():.3f}")
        
        # Verify long-term stability
        self._analyze_time_series_stability(time_series)
        
        print("[OK] 48-hour transport simulation completed successfully")
    
    def test_multi_domain_coupling_realism(self,
                                         realistic_transport_config,
                                         initial_realistic_state):
        """Test realistic multi-domain coupling scenarios."""
        
        print("--- L3-2: Multi-domain Coupling Realism ---")
        
        eddy_config, deposition_config = realistic_transport_config
        
        # Initialize all transport mechanisms
        transport_mechanisms = {
            'eddy': EddyDiffusionBLMFunctor("coupled_eddy", transport_config=eddy_config),
            'deposition': DryDepositionAggregator("coupled_deposition", deposition_config=deposition_config),
            'percolation': VerticalPercolationTransport("coupled_percolation")
        }
        
        # Test different realistic scenarios
        scenarios = [
            {
                'name': 'sunny_day',
                'description': 'Clear sunny day with high PBL and active transport',
                'environment': {
                    'shortwave_down': torch.tensor([800., 750., 820., 780.]),
                    'wind_speed': torch.tensor([5.0, 4.5, 5.5, 4.8]),
                    'friction_velocity': torch.tensor([0.5, 0.45, 0.55, 0.48]),
                    'monin_obukhov_length': torch.tensor([-150., -180., -120., -160.]),
                    'aerodynamic_resistance': torch.tensor([10., 12., 8., 11.]),
                    'pbl_height': torch.tensor([1500., 1400., 1600., 1450.])
                }
            },
            {
                'name': 'stable_night',
                'description': 'Stable nocturnal conditions with shallow BL',
                'environment': {
                    'shortwave_down': torch.tensor([0., 0., 0., 0.]),
                    'wind_speed': torch.tensor([2.0, 1.8, 2.2, 1.9]),
                    'friction_velocity': torch.tensor([0.15, 0.12, 0.18, 0.14]),
                    'monin_obukhov_length': torch.tensor([80., 90., 70., 85.]),
                    'aerodynamic_resistance': torch.tensor([100., 120., 80., 110.]),
                    'pbl_height': torch.tensor([150., 120., 180., 140.])
                }
            },
            {
                'name': 'rainy_day',
                'description': 'Low aerodynamic resistance and soil saturation',
                'environment': {
                    'shortwave_down': torch.tensor([200., 180., 220., 190.]),
                    'wind_speed': torch.tensor([8.0, 7.5, 8.5, 7.8]),
                    'friction_velocity': torch.tensor([0.8, 0.75, 0.85, 0.78]),
                    'monin_obukhov_length': torch.tensor([-80., -90., -70., -85.]),
                    'aerodynamic_resistance': torch.tensor([6., 7., 5., 6.5]),
                    'pbl_height': torch.tensor([800., 750., 850., 780.]),
                    'relative_humidity': torch.tensor([0.95, 0.93, 0.97, 0.94])
                }
            }
        ]
        
        coupling_results = {}
        
        for scenario in scenarios:
            print(f"  Testing scenario: {scenario['name']} - {scenario['description']}")
            
            state = initial_realistic_state.copy()
            scenario_env = scenario['environment']
            
            # Update state with scenario conditions
            for key, value in scenario_env.items():
                if key in [
                    'pbl_height',
                    'relative_humidity',
                    'friction_velocity',
                    'monin_obukhov_length',
                    'aerodynamic_resistance',
                ]:
                    state[key] = value
            
            dt = 1800.0  # 30 minutes
            results = {}
            
            eddy_result = transport_mechanisms['eddy'].forward(
                state=state,
                time=0.0,
                dt=dt,
            )
            results['eddy'] = eddy_result
            for species in ['no', 'no2', 'o3', 'isoprene']:
                species_key = f'{species}_mixing_ratio'
                state[species_key] = state[species_key] + _unwrap_update(
                    eddy_result[species_key]
                )

            deposition_result = transport_mechanisms['deposition'].forward(
                state=state,
                time=0.0,
                dt=dt,
            )
            results['deposition'] = deposition_result
            for species in ['no', 'no2', 'o3', 'so2']:
                species_key = f'{species}_mixing_ratio'
                state[species_key] = state[species_key] + _unwrap_update(
                    deposition_result[species_key]
                )

            results['percolation'] = transport_mechanisms['percolation'].forward(
                state=state,
                time=0.0,
                dt=dt,
            )
            self._verify_coupling_consistency(results, scenario['name'])
            coupling_results[scenario['name']] = results
            print(f"    [OK] Scenario {scenario['name']} completed successfully")
        
        # Cross-scenario analysis
        self._analyze_scenario_differences(coupling_results)
        
        print("[OK] Multi-domain coupling realism verified")
    
    def test_parameter_sensitivity_system(self,
                                        realistic_transport_config,
                                        initial_realistic_state,
                                        parameter_override):
        """Test system-level parameter sensitivity and uncertainty propagation."""
        
        print("--- L3-3: Parameter Sensitivity System Test ---")
        
        _, _ = realistic_transport_config
        
        # Define parameter variations for sensitivity analysis
        parameter_variations = {
            'z_over_L_limit': [0.5, 2.0, 5.0],
            'k_max': [1.0, 10.0, 50.0],
            'no2_effective_resistance': [100.0, 200.0, 400.0],
            'saturated_conductivity': [2e-6, 5e-6, 1e-5],
        }

        parameter_keys = {
            'z_over_L_limit': (
                'atmosphere.transport.z_over_L_limit',
                'dimensionless',
            ),
            'k_max': ('atmosphere.transport.k_max', 'm2 s-1'),
            'no2_effective_resistance': (
                'atmosphere.dry_deposition.species.no2.'
                'effective_non_aerodynamic_resistance',
                's m-1',
            ),
            'saturated_conductivity': (
                'hydrosphere.soil_moisture.saturated_hydraulic_conductivity',
                'm s-1',
            ),
        }

        from relic.parameters import get_global_manager

        manager = get_global_manager()
        baseline_overrides = {
            key: (
                manager.get_parameter(key).value,
                units,
                True,
            )
            for key, units in parameter_keys.values()
        }
        
        base_state = initial_realistic_state.copy()
        base_environment = {
            'shortwave_down': torch.tensor([400., 450., 350., 500.]),
            'wind_speed': torch.tensor([3.0, 3.5, 2.5, 4.0]),
            'friction_velocity': torch.tensor([0.3, 0.35, 0.25, 0.4]),
            'monin_obukhov_length': torch.tensor([-100., -120., -80., -150.]),
            'aerodynamic_resistance': torch.tensor([30., 25., 35., 20.]),
        }
        
        sensitivity_results = {}
        
        for param_name, param_values in parameter_variations.items():
            print(f"  Analyzing sensitivity to {param_name}")
            param_results = []
            
            for param_value in param_values:
                parameter_override(baseline_overrides)
                parameter_key, units = parameter_keys[param_name]
                parameter_override({
                    parameter_key: (param_value, units, True),
                })

                state = base_state.copy()
                state['friction_velocity'] = base_environment[
                    'friction_velocity'
                ]
                state['monin_obukhov_length'] = base_environment[
                    'monin_obukhov_length'
                ]
                state['aerodynamic_resistance'] = base_environment[
                    'aerodynamic_resistance'
                ]
                if param_name == 'z_over_L_limit':
                    state['monin_obukhov_length'] = torch.abs(
                        state['monin_obukhov_length']
                    )

                if param_name in ['z_over_L_limit', 'k_max']:
                    mechanism = EddyDiffusionBLMFunctor(
                        "sensitivity_eddy",
                        transport_config=EddyDiffusionConfig(
                            enabled=True,
                        ),
                    )
                    result = mechanism.forward(
                        state=state,
                        time=0.0,
                        dt=600.0,
                    )
                    no2_change = _unwrap_update(
                        result['no2_mixing_ratio']
                    ).abs().mean().item()
                    drainage = 0.0
                elif param_name == 'no2_effective_resistance':
                    mechanism = DryDepositionAggregator(
                        "sensitivity_deposition",
                        deposition_config=DryDepositionConfig(enabled=True),
                    )
                    result = mechanism.forward(
                        state=state,
                        time=0.0,
                        dt=600.0,
                    )
                    no2_change = _unwrap_update(
                        result['no2_mixing_ratio']
                    ).abs().mean().item()
                    drainage = 0.0
                else:
                    mechanism = VerticalPercolationTransport(
                        "sensitivity_percolation"
                    )
                    result = mechanism.forward(
                        state=state,
                        time=0.0,
                        dt=3600.0,
                    )
                    no2_change = 0.0
                    drainage = _unwrap_update(
                        result['drainage_flux']
                    ).mean().item()

                param_results.append({
                    'parameter_value': param_value,
                    'no2_change': no2_change,
                    'drainage_flux': drainage,
                    'response': (
                        drainage
                        if param_name == 'saturated_conductivity'
                        else no2_change
                    ),
                })
            
            sensitivity_results[param_name] = param_results
        
        # Analyze sensitivity patterns
        self._analyze_parameter_sensitivity(sensitivity_results)
        
        print("[OK] Parameter sensitivity analysis completed")
    
    def test_extreme_conditions_robustness(self,
                                         realistic_transport_config,
                                         initial_realistic_state):
        """Test system robustness under extreme environmental conditions."""
        
        print("--- L3-4: Extreme Conditions Robustness ---")
        
        eddy_config, deposition_config = realistic_transport_config
        
        # Initialize transport mechanisms
        eddy_transport = EddyDiffusionBLMFunctor(
            "robust_eddy", transport_config=eddy_config
        )
        dry_deposition = DryDepositionAggregator(
            "robust_deposition", deposition_config=deposition_config
        )
        soil_transport = VerticalPercolationTransport(
            "robust_percolation"
        )
        
        # Define extreme condition scenarios
        extreme_scenarios = [
            {
                'name': 'arctic_winter',
                'state_mods': {
                    'air_temperature': torch.tensor(
                        [233.15, 235.15, 231.15, 237.15]
                    ),
                    'soil_temperature': torch.tensor(
                        [273.15, 275.15, 274.15, 276.15]
                    ),
                },
                'environment': {
                    'wind_speed': torch.tensor([15.0, 16.0, 14.0, 17.0]),
                    'shortwave_down': torch.tensor([50., 40., 60., 45.]),
                    'friction_velocity': torch.tensor([1.2, 1.3, 1.1, 1.4]),
                    'monin_obukhov_length': torch.tensor(
                        [50., 45., 55., 40.]
                    ),
                    'aerodynamic_resistance': torch.tensor(
                        [5., 4.5, 5.5, 4.0]
                    ),
                    'pbl_height': torch.tensor([100., 80., 120., 90.]),
                }
            },
            {
                'name': 'desert_summer',
                'state_mods': {
                    'air_temperature': torch.tensor(
                        [323.15, 325.15, 321.15, 327.15]
                    ),
                    'soil_temperature': torch.tensor(
                        [321.15, 322.15, 320.15, 323.15]
                    ),
                    'relative_humidity': torch.tensor(
                        [0.05, 0.04, 0.06, 0.03]
                    ),
                    'soil_moisture': torch.tensor(
                        [0.07, 0.08, 0.09, 0.065]
                    ),
                },
                'environment': {
                    'shortwave_down': torch.tensor(
                        [1200., 1150., 1250., 1180.]
                    ),
                    'wind_speed': torch.tensor([1.0, 0.8, 1.2, 0.9]),
                    'friction_velocity': torch.tensor(
                        [0.1, 0.08, 0.12, 0.09]
                    ),
                    'monin_obukhov_length': torch.tensor(
                        [-200., -180., -220., -190.]
                    ),
                    'aerodynamic_resistance': torch.tensor(
                        [80., 100., 70., 90.]
                    ),
                    'pbl_height': torch.tensor(
                        [2900., 2800., 3000., 2850.]
                    ),
                }
            },
            {
                'name': 'tropical_storm',
                'state_mods': {
                    'relative_humidity': torch.tensor([0.98, 0.97, 0.99, 0.96]),
                    'soil_moisture': torch.tensor(
                        [0.409, 0.405, 0.408, 0.400]
                    ),
                },
                'environment': {
                    'wind_speed': torch.tensor([25.0, 27.0, 23.0, 29.0]),
                    'shortwave_down': torch.tensor([100., 80., 120., 90.]),
                    'friction_velocity': torch.tensor([1.8, 2.0, 1.7, 1.9]),
                    'monin_obukhov_length': torch.tensor(
                        [-50., -45., -55., -40.]
                    ),
                    'aerodynamic_resistance': torch.tensor(
                        [2., 1.8, 2.2, 1.6]
                    ),
                    'pbl_height': torch.tensor([500., 450., 550., 400.]),
                }
            }
        ]
        
        robustness_results = {}
        
        for scenario in extreme_scenarios:
            print(f"  Testing extreme scenario: {scenario['name']}")
            
            state = initial_realistic_state.copy()
            
            # Apply extreme modifications
            for key, value in scenario['state_mods'].items():
                state[key] = value
            
            environment = scenario['environment']
            for key in [
                'pbl_height',
                'friction_velocity',
                'monin_obukhov_length',
                'aerodynamic_resistance',
            ]:
                state[key] = environment[key]

            eddy_dt = 30.0
            process_dt = 1800.0
            
            scenario_results = {}

            n_eddy_substeps = int(process_dt / eddy_dt)
            for substep in range(n_eddy_substeps):
                eddy_result = eddy_transport.forward(
                    state=state,
                    time=substep * eddy_dt,
                    dt=eddy_dt,
                )
                for species in ['no', 'no2', 'o3', 'isoprene']:
                    key = f'{species}_mixing_ratio'
                    increment = _unwrap_update(eddy_result[key])
                    assert torch.isfinite(increment).all(), (
                        f"Non-finite values in {key} for {scenario['name']}"
                    )
                    assert torch.all(state[key] + increment >= 0.0), (
                        f"Negative Eddy-updated state in {key} for "
                        f"{scenario['name']}"
                    )
                    state[key] = state[key] + increment
            scenario_results['eddy'] = eddy_result

            depo_result = dry_deposition.forward(
                state=state,
                time=0.0,
                dt=process_dt,
            )
            scenario_results['deposition'] = depo_result
            for species in ['no', 'no2', 'o3', 'so2']:
                key = f'{species}_mixing_ratio'
                increment = _unwrap_update(depo_result[key])
                assert torch.isfinite(increment).all(), (
                    f"Non-finite values in {key} for {scenario['name']}"
                )
                assert torch.all(increment <= 0.0), (
                    f"Positive deposition increment in {key} for "
                    f"{scenario['name']}"
                )
                assert torch.all(state[key] + increment >= 0.0), (
                    f"Negative deposition-updated state in {key} for "
                    f"{scenario['name']}"
                )
                state[key] = state[key] + increment

            soil_result = soil_transport.forward(
                state=state,
                time=0.0,
                dt=process_dt,
            )
            scenario_results['soil'] = soil_result
            drainage = _unwrap_update(
                soil_result['drainage_flux']
            )
            assert torch.isfinite(drainage).all(), (
                f"Non-finite drainage for {scenario['name']}"
            )
            assert drainage.min() >= 0, (
                f"Negative drainage for {scenario['name']}"
            )

            robustness_results[scenario['name']] = scenario_results
            print(f"    [OK] Survived extreme scenario: {scenario['name']}")
        
        # Analyze robustness patterns
        self._analyze_robustness_results(robustness_results)
        
        print("[OK] Extreme conditions robustness testing completed")
    
    def test_performance_benchmarking(self,
                                    realistic_transport_config,
                                    initial_realistic_state):
        """Test performance and scalability of transport system."""
        
        print("--- L3-5: Performance Benchmarking ---")
        
        eddy_config, deposition_config = realistic_transport_config
        
        # Test different batch sizes for scalability
        batch_sizes = [1, 4, 8, 16, 32]
        performance_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create state with specified batch size
            batch_state = {}
            for key, value in initial_realistic_state.items():
                if isinstance(value, torch.Tensor):
                    batch_state[key] = value[:1].expand(batch_size).clone()
                else:
                    batch_state[key] = value
            
            # Environment for batch
            spatial_offset = torch.linspace(-1.0, 1.0, batch_size)
            batch_environment = {
                'wind_speed': 3.0 + 0.5 * spatial_offset,
                'friction_velocity': 0.3 + 0.05 * spatial_offset,
                'monin_obukhov_length': -100.0 + 25.0 * spatial_offset,
                'aerodynamic_resistance': 30.0 + 10.0 * spatial_offset,
                'shortwave_down': 400.0 + 100.0 * spatial_offset,
            }
            batch_state['friction_velocity'] = batch_environment[
                'friction_velocity'
            ]
            batch_state['monin_obukhov_length'] = batch_environment[
                'monin_obukhov_length'
            ]
            batch_state['aerodynamic_resistance'] = batch_environment[
                'aerodynamic_resistance'
            ]
            
            # Initialize mechanisms
            eddy_transport = EddyDiffusionBLMFunctor(
                f"perf_eddy_{batch_size}", transport_config=eddy_config
            )
            dry_deposition = DryDepositionAggregator(
                f"perf_deposition_{batch_size}", deposition_config=deposition_config
            )
            
            # Performance timing
            times = {'eddy': [], 'deposition': [], 'total': []}
            n_runs = 5  # Multiple runs for averaging
            
            for run in range(n_runs):
                start_total = time.perf_counter()
                
                # Time eddy diffusion
                start_eddy = time.perf_counter()
                eddy_result = eddy_transport.forward(
                    state=batch_state,
                    time=0.0,
                    dt=600.0,
                )
                end_eddy = time.perf_counter()
                times['eddy'].append(end_eddy - start_eddy)
                
                # Time dry deposition
                start_depo = time.perf_counter()
                depo_result = dry_deposition.forward(
                    state=batch_state,
                    time=0.0,
                    dt=600.0,
                )
                end_depo = time.perf_counter()
                times['deposition'].append(end_depo - start_depo)
                
                end_total = time.perf_counter()
                times['total'].append(end_total - start_total)
                
                # Verify results are valid
                assert all(
                    torch.isfinite(_unwrap_update(value)).all()
                    for value in eddy_result.values()
                    if isinstance(_unwrap_update(value), torch.Tensor)
                )
                assert all(
                    torch.isfinite(_unwrap_update(value)).all()
                    for value in depo_result.values()
                    if isinstance(_unwrap_update(value), torch.Tensor)
                )
            
            # Calculate performance metrics
            perf_metrics = {}
            for process, time_list in times.items():
                mean_time = float(np.mean(time_list))
                std_time = float(np.std(time_list))
                safe_mean_time = mean_time if mean_time > 0 else 1e-12
                perf_metrics[process] = {
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'time_per_element': mean_time / batch_size,
                    'throughput': batch_size / safe_mean_time
                }
            
            performance_results[batch_size] = perf_metrics
            
            print(f"    [OK] Batch size {batch_size}: "
                  f"Eddy {perf_metrics['eddy']['mean_time']:.3f}s, "
                  f"Deposition {perf_metrics['deposition']['mean_time']:.3f}s")
        
        # Analyze performance scaling
        self._analyze_performance_scaling(performance_results)
        
        print("[OK] Performance benchmarking completed")
    
    # Helper methods for analysis
    
    def _verify_state_stability(self, state: Dict, hour: int):
        """Verify state remains physically reasonable."""
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"Non-finite values in {key} at hour {hour}"
                
                # Species concentrations should be positive
                if 'mixing_ratio' in key:
                    assert value.min() >= 0, f"Negative concentration in {key} at hour {hour}"
                    assert value.max() < 1e-3, f"Unrealistic concentration in {key} at hour {hour}"
                
                # Soil moisture should be reasonable
                if key == 'soil_moisture':
                    assert value.min() >= 0.01, f"Too dry soil in {key} at hour {hour}"
                    assert value.max() <= 0.6, f"Oversaturated soil in {key} at hour {hour}"
    
    def _analyze_time_series_stability(self, time_series: Dict):
        """Analyze time series for stability patterns."""
        concentration_sinks = {'no2_conc', 'o3_conc'}
        deposition_increments = {'deposition_increment_no2'}
        # Allow slightly higher trend in drainage flux during equilibration.
        trend_thresholds = {
            'drainage_flux': 0.04,
        }
        default_trend_threshold = 0.01
        for var_name, values in time_series.items():
            if var_name == 'time' or len(values) == 0:
                continue
            
            values_array = np.array(values)
            assert np.isfinite(values_array).all(), (
                f"Non-finite time-series values in {var_name}"
            )

            # An open deposition sink must deplete a source-free column rather
            # than satisfy a stationary-trend criterion.
            if var_name in concentration_sinks:
                assert np.all(values_array >= 0.0), (
                    f"Negative concentration in {var_name}"
                )
                assert values_array[-1] <= values_array[0], (
                    f"Source-free deposition increased {var_name}"
                )
                continue

            if var_name in deposition_increments:
                assert np.all(values_array <= 0.0), (
                    f"Positive deposition increment in {var_name}"
                )
                continue

            # Check for trends or instabilities
            if len(values_array) > 24:  # More than 24 hours
                # Linear trend test
                hours = np.arange(len(values_array))
                # Simple linear trend calculation (avoid numpy.polyfit due to OpenMP issues)
                if len(values_array) > 1:
                    trend_slope = (values_array[-1] - values_array[0]) / (len(values_array) - 1)
                else:
                    trend_slope = 0.0
                
                # Relative trend per hour
                mean_value = np.mean(values_array)
                if mean_value > 0:
                    relative_trend = abs(trend_slope) / mean_value
                    threshold = trend_thresholds.get(var_name, default_trend_threshold)
                    assert relative_trend < threshold, (
                        f"Strong trend in {var_name}: {relative_trend:.3f}/hour "
                        f"(limit {threshold:.3f}/hour)"
                    )
            
            # Check for oscillation stability
            std_dev = np.std(values_array)
            mean_value = np.mean(values_array)
            if mean_value > 0:
                coefficient_of_variation = std_dev / mean_value
                assert coefficient_of_variation < 2.0, f"High variability in {var_name}: CV={coefficient_of_variation:.2f}"
    
    def _verify_coupling_consistency(self, results: Dict, scenario_name: str):
        """Verify that coupling between mechanisms is physically consistent."""
        def _assert_finite(value: Any, path: str) -> None:
            candidate = _unwrap_update(value)
            if isinstance(candidate, torch.Tensor):
                assert torch.isfinite(candidate).all(), (
                    f"Non-finite result in {path} for {scenario_name}"
                )
            elif isinstance(candidate, dict):
                for nested_key, nested_value in candidate.items():
                    _assert_finite(nested_value, f"{path}.{nested_key}")

        # Check that all mechanisms produced finite results
        for mech_name, result in results.items():
            _assert_finite(result, mech_name)
    
    def _analyze_scenario_differences(self, coupling_results: Dict):
        """Analyze differences between scenarios."""
        scenario_names = list(coupling_results.keys())
        relative_differences = []
        
        for i, scenario1 in enumerate(scenario_names):
            for j, scenario2 in enumerate(scenario_names[i+1:], i+1):
                # Compare key metrics between scenarios
                if ('eddy' in coupling_results[scenario1] and 
                    'eddy' in coupling_results[scenario2]):
                    
                    # Example: Compare NO2 transport rates
                    if ('no2_mixing_ratio' in coupling_results[scenario1]['eddy'] and
                        'no2_mixing_ratio' in coupling_results[scenario2]['eddy']):
                        
                        rate1 = _unwrap_update(
                            coupling_results[scenario1]['eddy'][
                                'no2_mixing_ratio'
                            ]
                        ).abs().mean()
                        rate2 = _unwrap_update(
                            coupling_results[scenario2]['eddy'][
                                'no2_mixing_ratio'
                            ]
                        ).abs().mean()
                        
                        # Different scenarios should show different transport rates
                        relative_diff = abs(rate1 - rate2) / (rate1 + rate2 + 1e-12)
                        relative_differences.append(float(relative_diff))
                        print(f"    Scenario difference {scenario1} vs {scenario2}: {relative_diff:.3f}")

        assert relative_differences
        assert max(relative_differences) > 0.0, (
            "Distinct meteorological scenarios produced identical Eddy transport"
        )
    
    def _analyze_parameter_sensitivity(self, sensitivity_results: Dict):
        """Analyze parameter sensitivity patterns."""
        for param_name, param_results in sensitivity_results.items():
            if len(param_results) < 2:
                continue
            
            # Extract parameter values and responses
            param_values = [r['parameter_value'] for r in param_results]
            responses = [r['response'] for r in param_results]
            
            if len(param_values) >= 2:
                # Calculate sensitivity coefficient
                param_range = max(param_values) - min(param_values)
                response_range = max(responses) - min(responses)
                assert np.isfinite(responses).all(), (
                    f"Non-finite response for {param_name}"
                )
                assert response_range > 0.0, (
                    f"No measurable response for {param_name}"
                )
                
                if param_range > 0:
                    parameter_scale = max(
                        abs(float(np.mean(param_values))),
                        np.finfo(float).tiny,
                    )
                    response_scale = max(
                        abs(float(np.mean(responses))),
                        np.finfo(float).tiny,
                    )
                    normalized_sensitivity = (
                        response_range / response_scale
                    ) / (param_range / parameter_scale)
                    assert np.isfinite(normalized_sensitivity)
                    assert normalized_sensitivity >= 0.0
                    print(
                        f"    Normalized sensitivity to {param_name}: "
                        f"{normalized_sensitivity:.2e}"
                    )
    
    def _analyze_robustness_results(self, robustness_results: Dict):
        """Analyze robustness test results."""
        total_scenarios = len(robustness_results)
        assert total_scenarios > 0

        for scenario_name, results in robustness_results.items():
            assert set(results) == {'eddy', 'deposition', 'soil'}
            print(f"    [OK] {scenario_name}: Survived extreme conditions")

        print(f"  Robustness scenarios passed: {total_scenarios}/{total_scenarios}")
    
    def _analyze_performance_scaling(self, performance_results: Dict):
        """Analyze performance scaling characteristics."""
        batch_sizes = sorted(performance_results.keys())
        
        for process in ['eddy', 'deposition', 'total']:
            throughputs = [performance_results[bs][process]['throughput'] for bs in batch_sizes]
            
            # Check that throughput generally increases with batch size (better efficiency)
            if len(throughputs) >= 3:
                # Calculate scaling efficiency
                base_throughput = throughputs[0]
                max_throughput = max(throughputs)
                scaling_efficiency = max_throughput / base_throughput
                
                print(f"    {process.capitalize()} scaling efficiency: {scaling_efficiency:.2f}x")
                
                # Expect at least some improvement with larger batches
                assert scaling_efficiency > 1.0, f"Poor scaling for {process}"


def run_l3_system_tests():
    """Run all L3 system tests for transport mechanisms."""
    test_files = [__file__]
    
    print("Running L3 System Tests for Transport Mechanisms...")
    print("=" * 60)
    
    # Configure pytest for comprehensive system testing
    pytest_args = [
        "-v",  # Verbose
        "-s",  # Don't capture output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "-x",  # Stop on first failure for system tests
    ] + test_files
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("ALL L3 TRANSPORT SYSTEM TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME L3 SYSTEM TESTS FAILED - Check output above")
        print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    # Run tests when called directly
    exit_code = run_l3_system_tests()
    sys.exit(exit_code)

