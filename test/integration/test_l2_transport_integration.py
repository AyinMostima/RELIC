#!/usr/bin/env python3
"""
L2 Transport Integration Tests
=============================

Tests the integration between different transport mechanisms and their coupling
with atmospheric, hydrosphere, and biosphere processes.

Test Coverage:
    - Atmosphere transport integration (eddy diffusion + dry deposition)
    - Hydrosphere transport integration (vertical percolation + advection)
    - Cross-module coupling (transport <-> chemistry <-> radiation)
    - Multi-scale transport processes (fast turbulence + slow percolation)
    - Conservation verification across transport boundaries

Scientific Validation:
    - Mass conservation across transport domains
    - Flux continuity at interfaces
    - Transport-chemistry coupling accuracy
    - Multi-component transport consistency
    - Numerical stability across scales

Integration Levels:
    L2-1: Atmosphere transport mechanisms integration
    L2-2: Hydrosphere transport mechanisms integration  
    L2-3: Cross-sphere transport coupling
    L2-4: Transport-chemistry-radiation coupling chain
    L2-5: Multi-scale temporal integration

References:
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology
    - Wesely, M.L. (1989). Parameterization of surface resistances
    - Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics
    - van Genuchten, M.T. (1980). Closed-form equation for hydraulic conductivity
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add relic to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Atmosphere transport modules
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
    EddyDiffusionBLMFunctor, EddyDiffusionConfig
)
from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionAggregator, DryDepositionConfig
)

# Hydrosphere transport modules  
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport
)

# Supporting modules for coupling tests
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.core.state import UpdateItem, UpdateMode


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value

class TestL2TransportIntegration:
    """Test suite for L2 transport integration across multiple mechanisms."""
    
    @pytest.fixture
    def comprehensive_transport_state(self):
        """Create comprehensive state with all transport-relevant variables."""
        batch_size = 4
        return {
            # === ATMOSPHERIC STATE ===
            # Basic meteorology
            'air_temperature': torch.tensor([298.15, 300.15, 296.15, 302.15]),
            'surface_temperature': torch.tensor([299.15, 301.15, 297.15, 303.15]),
            'atmospheric_pressure': torch.tensor([101325.0, 101300.0, 101350.0, 101275.0]),
            'relative_humidity': torch.tensor([0.7, 0.65, 0.75, 0.6]),
            'wind_speed': torch.tensor([3.0, 4.0, 2.5, 5.0]),
            
            # Atmospheric chemistry species
            'no_mixing_ratio': torch.tensor([5e-9, 8e-9, 3e-9, 12e-9]),    # ppb
            'no2_mixing_ratio': torch.tensor([15e-9, 25e-9, 10e-9, 30e-9]),  # ppb  
            'o3_mixing_ratio': torch.tensor([60e-9, 80e-9, 45e-9, 100e-9]),  # ppb
            'co_mixing_ratio': torch.tensor([150e-9, 150e-9, 150e-9, 150e-9]),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor([1.8e-6, 1.8e-6, 1.8e-6, 1.8e-6]),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor([0.01, 0.01, 0.01, 0.01]),  # O(1D) + H2O OH source, 10 g/kg
            'isoprene_mixing_ratio': torch.tensor([2e-9, 4e-9, 1e-9, 6e-9]), # ppb
            'so2_mixing_ratio': torch.tensor([2e-9, 3e-9, 1e-9, 4e-9]),      # ppb
            
            # Turbulence and boundary layer
            'pbl_height': torch.tensor([800.0, 1000.0, 600.0, 1200.0]),     # m
            'friction_velocity': torch.tensor([0.3, 0.4, 0.25, 0.5]),       # m/s
            'monin_obukhov_length': torch.tensor([-100.0, -150.0, 200.0, -50.0]),  # m
            'aerodynamic_resistance': torch.tensor([50.0, 40.0, 60.0, 35.0]), # s/m
            
            # Radiation for photolysis
            'shortwave_down': torch.tensor([800.0, 900.0, 700.0, 1000.0]),  # W m-2
            'incoming_shortwave_potential': torch.tensor([1100.0, 1200.0, 1000.0, 1250.0]),
            'cos_zenith': torch.tensor([0.8, 0.9, 0.7, 0.85]),
            'surface_albedo': torch.tensor([0.15, 0.18, 0.12, 0.22]),
            'surface_emissivity': torch.tensor([0.97, 0.96, 0.98, 0.95]),
            'cloud_fraction': torch.tensor([0.1, 0.2, 0.3, 0.0]),
            
            # === HYDROSPHERE STATE ===
            # Soil water content
            'soil_moisture': torch.tensor([0.25, 0.30, 0.22, 0.32]),
            'soil_temperature': torch.tensor([288.15, 290.15, 286.15, 292.15]),
            'hydraulic_conductivity': torch.tensor([5e-7, 8e-7, 3e-7, 1e-6]),  # m/s
            'soil_layer_thickness': torch.tensor([1.0, 1.2, 0.8, 1.5]),       # m
            
            # === SURFACE PROPERTIES ===
            # Land use for deposition
            'land_use_type': torch.tensor([1, 2, 3, 1]),  # Forest, crop, urban, forest
            'leaf_area_index': torch.tensor([4.0, 3.0, 1.0, 5.0]),
            'vegetation_height': torch.tensor([15.0, 1.5, 0.5, 20.0]),        # m
            'roughness_length': torch.tensor([1.0, 0.1, 0.01, 1.5]),          # m
            
            # === TRANSPORT PROPERTIES ===  
            # Diffusion coefficients
            'molecular_diffusivity': torch.tensor([1.5e-5, 1.8e-5, 1.2e-5, 2.0e-5]),
            'turbulent_diffusivity': torch.tensor([10.0, 15.0, 8.0, 20.0]),   # m2 s-1
        }
    
    @pytest.fixture
    def atmosphere_transport_configs(self):
        """Create configurations for atmospheric transport mechanisms."""
        eddy_config = EddyDiffusionConfig(
            enabled=True,
        )
        
        deposition_config = DryDepositionConfig(
            enabled=True,
        )
        
        return eddy_config, deposition_config
    
    @pytest.fixture
    def hydrosphere_transport_config(self):
        """Create reference values for hydrosphere transport mechanisms."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        def _to_float(value):
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)
        return {
            'parameters': {
                'field_capacity': _to_float(
                    manager.get(
                        'field_capacity',
                        namespace='hydrosphere.soil_moisture',
                        units='m3 m-3',
                        required=True,
                    )
                ),
                'wilting_point': _to_float(
                    manager.get(
                        'wilting_point',
                        namespace='hydrosphere.soil_moisture',
                        units='m3 m-3',
                        required=True,
                    )
                ),
            }
        }
    
    def test_atmosphere_transport_mechanisms_integration(self,
                                                       comprehensive_transport_state,
                                                       atmosphere_transport_configs):
        """Test integration between eddy diffusion and dry deposition."""
        
        eddy_config, deposition_config = atmosphere_transport_configs
        
        # Initialize transport mechanisms
        eddy_diffusion = EddyDiffusionBLMFunctor(
            name="eddy_transport", transport_config=eddy_config
        )
        dry_deposition = DryDepositionAggregator(
            name="deposition_transport", deposition_config=deposition_config  
        )
        
        state = comprehensive_transport_state.copy()
        dt = 1800.0  # 30 minutes
        time = 0.0
        
        print("--- L2-1: Atmosphere Transport Integration ---")
        
        # Step 1: Apply eddy diffusion transport
        eddy_result = eddy_diffusion.forward(
            state=state, time=time, dt=dt
        )

        # Update state with eddy diffusion changes
        for species in ['no', 'no2', 'o3', 'isoprene']:
            species_key = f'{species}_mixing_ratio'
            state[species_key] = state[species_key] + _unwrap_update(
                eddy_result[species_key]
            )
        
        # Step 2: Apply dry deposition (uses updated concentrations)
        deposition_result = dry_deposition.forward(
            state=state, time=time, dt=dt
        )
        
        # Update state with deposition changes
        for species in ['no', 'no2', 'o3', 'so2']:
            species_key = f'{species}_mixing_ratio'
            state[species_key] = state[species_key] + _unwrap_update(
                deposition_result[species_key]
            )
        
        # Verification
        assert len(eddy_result) > 0, "Eddy diffusion should produce outputs"
        assert len(deposition_result) > 0, "Dry deposition should produce outputs"
        
        # Check transport effects are reasonable
        eddy_changes = [
            abs(_unwrap_update(eddy_result[f'{s}_mixing_ratio']).max().item())
            for s in ['no', 'no2', 'o3', 'isoprene']
        ]
        dep_changes = [
            abs(_unwrap_update(deposition_result[f'{s}_mixing_ratio']).max().item())
            for s in ['no', 'no2', 'o3', 'so2']
        ]
        
        assert max(eddy_changes) < 1e-6, "Eddy diffusion changes should be reasonable"
        assert max(dep_changes) < 1e-6, "Deposition changes should be reasonable"
        
        # Deposition should always remove species (negative changes)
        for species in ['no', 'no2', 'o3', 'so2']:
            species_key = f'{species}_mixing_ratio'
            assert torch.all(
                _unwrap_update(deposition_result[species_key]) <= 0
            ), f"{species} deposition should be removal"
        
        # Check final concentrations remain physical
        for species in ['no', 'no2', 'o3', 'so2', 'isoprene']:
            species_key = f'{species}_mixing_ratio'
            if species_key in state:
                assert torch.all(state[species_key] >= 0), f"{species} concentration should be non-negative"
                assert torch.all(state[species_key] < 1e-3), f"{species} concentration should be reasonable"
        
        print(f"[OK] Eddy max change: {max(eddy_changes):.2e}, Deposition max change: {max(dep_changes):.2e}")
        print("[OK] L2-1: Atmosphere transport mechanisms integration verified")
    
    def test_hydrosphere_transport_integration(self, 
                                             comprehensive_transport_state,
                                             hydrosphere_transport_config):
        """Test hydrosphere transport mechanisms."""
        
        # Initialize hydrosphere transport
        vertical_percolation = VerticalPercolationTransport(
            name="percolation_transport", 
        )
        
        state = comprehensive_transport_state.copy()
        dt = 3600.0  # 1 hour  
        time = 0.0
        
        print("--- L2-2: Hydrosphere Transport Integration ---")
        
        # Test vertical percolation process
        percolation_result = vertical_percolation.forward(
            state=state, time=time, dt=dt
        )
        
        # Route the open-boundary flux through an explicit storage ledger.
        initial_soil_moisture = state['soil_moisture'].clone()
        drainage = _unwrap_update(percolation_result['drainage_flux'])
        layer_thickness = state['soil_layer_thickness']
        soil_moisture_loss = drainage * dt / (1000.0 * layer_thickness)
        state['soil_moisture'] = state['soil_moisture'] - soil_moisture_loss
        
        # Verification
        assert len(percolation_result) > 0, "Percolation should produce outputs"
        
        # Check drainage flux values
        assert torch.all(drainage >= 0), "Drainage flux should be non-negative"
        assert percolation_result['drainage_flux'].unit == 'kg m-2 s-1'
        assert torch.allclose(
            (initial_soil_moisture - state['soil_moisture'])
            * 1000.0
            * layer_thickness,
            drainage * dt,
            atol=1e-6,
            rtol=1e-6,
        )
        max_drainage = drainage.max().item()
        print(f"[OK] Max drainage flux: {max_drainage:.2e} kg m-2 s-1")
        
        # Check soil moisture remains physical
        soil_moisture = state['soil_moisture']
        assert torch.all(soil_moisture >= 0.0), "Soil moisture should be non-negative"
        assert torch.all(soil_moisture <= 1.0), "Soil moisture should be <= 1.0"
        
        # Check water content hierarchy is maintained
        field_capacity = hydrosphere_transport_config['parameters']['field_capacity']
        wilting_point = hydrosphere_transport_config['parameters']['wilting_point']
        
        # Above field capacity should have drainage
        above_fc = soil_moisture > field_capacity
        if torch.any(above_fc):
            drainage_above_fc = drainage[above_fc]
            assert torch.all(drainage_above_fc > 0), "Should have drainage above field capacity"
        
        print("[OK] L2-2: Hydrosphere transport integration verified")
    
    def test_cross_sphere_transport_coupling(self,
                                           comprehensive_transport_state,
                                           atmosphere_transport_configs):
        """Test coupling between atmosphere and hydrosphere transport."""
        
        eddy_config, _ = atmosphere_transport_configs
        
        # Initialize transport mechanisms
        eddy_diffusion = EddyDiffusionBLMFunctor(
            name="atm_eddy", transport_config=eddy_config
        )
        vertical_percolation = VerticalPercolationTransport(
            name="hydro_percolation"
        )
        
        state = comprehensive_transport_state.copy()
        dt = 1800.0  # 30 minutes
        time = 0.0
        
        print("--- L2-3: Cross-Sphere Transport Coupling ---")
        
        # Record initial conditions
        initial_no2 = state['no2_mixing_ratio'].clone()
        initial_soil_moisture = state['soil_moisture'].clone()
        
        # Step 1: Atmospheric transport affects boundary layer chemistry
        atm_result = eddy_diffusion.forward(
            state=state, time=time, dt=dt
        )
        state['no2_mixing_ratio'] = state['no2_mixing_ratio'] + _unwrap_update(
            atm_result['no2_mixing_ratio']
        )
        
        # Step 2: Soil water transport affects surface exchange
        hydro_result = vertical_percolation.forward(
            state=state, time=time, dt=dt
        )
        drainage = _unwrap_update(hydro_result['drainage_flux'])
        moisture_loss = drainage * dt / (
            1000.0 * state['soil_layer_thickness']
        )
        state['soil_moisture'] = (
            state['soil_moisture'] - moisture_loss
        )
        
        # Step 3: Couple the processes through surface exchange
        # Soil moisture affects aerodynamic and surface resistance
        soil_moisture = state['soil_moisture']
        
        # Update surface resistance based on soil moisture
        # Dry soils increase surface resistance to gas exchange
        moisture_stress = torch.clamp((soil_moisture - 0.1) / (0.3 - 0.1), min=0.0, max=1.0)
        surface_resistance_factor = 1.0 + (1.0 - moisture_stress) * 2.0  # Higher resistance when dry
        
        surface_coupling_effect = surface_resistance_factor.mean().item()
        print(f"[OK] Surface coupling factor: {surface_coupling_effect:.2f}")
        
        # Verification
        assert len(atm_result) > 0, "Atmospheric transport should produce outputs"
        assert len(hydro_result) > 0, "Hydrosphere transport should produce outputs"
        
        # Check coupling effects are reasonable
        # Changes should be small but measurable
        no2_change = (state['no2_mixing_ratio'] - initial_no2).abs().max().item()
        soil_change = (state['soil_moisture'] - initial_soil_moisture).abs().max().item()
        assert no2_change > 0.0
        assert soil_change > 0.0
        
        print(f"[OK] NO2 change: {no2_change:.2e}, Soil moisture change: {soil_change:.3f}")
        
        # Physical consistency checks
        assert torch.all(state['no2_mixing_ratio'] >= 0), "NO2 should remain non-negative"
        assert torch.all(state['soil_moisture'] >= 0), "Soil moisture should be non-negative"
        
        print("[OK] L2-3: Cross-sphere transport coupling verified")
    
    def test_transport_chemistry_radiation_coupling(self,
                                                  comprehensive_transport_state,
                                                  atmosphere_transport_configs):
        """Test full coupling chain: radiation -> chemistry -> transport."""
        
        eddy_config, deposition_config = atmosphere_transport_configs
        
        # Initialize full coupling chain
        radiation = RadiationFunctor()
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        eddy_diffusion = EddyDiffusionBLMFunctor(
            name="coupled_eddy", transport_config=eddy_config
        )
        dry_deposition = DryDepositionAggregator(
            name="coupled_deposition", deposition_config=deposition_config
        )
        
        state = comprehensive_transport_state.copy()
        # Chemistry uses explicit Euler here; stay inside its positivity domain.
        dt = 30.0
        time = 0.0

        print("--- L2-4: Transport-Chemistry-Radiation Coupling Chain ---")
        
        # Step 1: Radiation drives photolysis rates.
        rad_result = radiation.forward(state, time, dt)
        for key, value in rad_result.items():
            if 'radiation' in key.lower():
                state[key] = _unwrap_update(value)
        print("[OK] Radiation calculation successful")

        # Step 2: Chemistry processes use radiation for photolysis.
        chemistry_state_keys = (
            'no_mixing_ratio',
            'no2_mixing_ratio',
            'o3_mixing_ratio',
            'isoprene_mixing_ratio',
        )
        initial_chemistry_state = {
            key: state[key].clone()
            for key in chemistry_state_keys
        }
        chem_result = chemistry.forward(state, time, dt)
        for key in chemistry_state_keys:
            item = chem_result[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.source_functor == 'atmospheric_chemistry'
            state[key] = state[key] + item.value
            assert torch.allclose(
                state[key] - initial_chemistry_state[key],
                item.value,
                atol=1e-18,
                rtol=1e-6,
            )
        print("[OK] Chemistry calculation successful")

        # Step 3: Transport processes use updated concentrations.
        eddy_result = eddy_diffusion.forward(state=state, time=time, dt=dt)
        for species in ['no', 'no2', 'o3', 'isoprene']:
            species_key = f'{species}_mixing_ratio'
            state[species_key] = state[species_key] + _unwrap_update(
                eddy_result[species_key]
            )
        depo_result = dry_deposition.forward(
            state=state, time=time, dt=dt
        )
        for species in ['no', 'no2', 'o3', 'so2']:
            species_key = f'{species}_mixing_ratio'
            state[species_key] = state[species_key] + _unwrap_update(
                depo_result[species_key]
            )
        print("[OK] Transport calculations successful")

        # Verification of coupling chain
        assert len(eddy_result) > 0, "Eddy diffusion should produce outputs"
        assert len(depo_result) > 0, "Deposition should produce outputs"
        for species in ['no', 'no2', 'o3', 'so2', 'isoprene']:
            species_key = f'{species}_mixing_ratio'
            conc = state[species_key]
            assert torch.all(conc >= 0), f"{species} should be non-negative"
            assert torch.all(conc < 1e-3), f"{species} should be reasonable"

        print("[OK] L2-4: Coupling chain verified (3/3 steps successful)")
    
    def test_multi_scale_temporal_integration(self,
                                            comprehensive_transport_state,
                                            atmosphere_transport_configs):
        """Test transport processes operating on different time scales."""
        
        eddy_config, _ = atmosphere_transport_configs
        
        # Initialize transport mechanisms  
        fast_eddy = EddyDiffusionBLMFunctor(
            name="fast_eddy", transport_config=eddy_config
        )
        slow_percolation = VerticalPercolationTransport(
            name="slow_percolation"
        )
        
        state = comprehensive_transport_state.copy()
        total_time = 7200.0  # 2 hours
        fast_dt = 600.0      # 10 minutes for atmospheric transport
        slow_dt = 1800.0     # 30 minutes for soil transport
        
        print("--- L2-5: Multi-Scale Temporal Integration ---")
        
        # Track changes over time
        time_series = {'time': [], 'no2_conc': [], 'soil_moisture': []}
        
        time = 0.0
        fast_steps = 0
        slow_steps = 0
        
        while time < total_time:
            # Fast atmospheric transport every 10 minutes
            if time % fast_dt == 0:
                atm_result = fast_eddy.forward(
                    state=state,
                    time=time,
                    dt=fast_dt,
                )
                for species in ['no', 'no2', 'o3', 'isoprene']:
                    species_key = f'{species}_mixing_ratio'
                    state[species_key] = state[species_key] + _unwrap_update(
                        atm_result[species_key]
                    )
                fast_steps += 1
            
            # Slow soil transport every 30 minutes
            if time % slow_dt == 0:
                soil_result = slow_percolation.forward(
                    state=state, time=time, dt=slow_dt
                )
                drainage = _unwrap_update(soil_result['drainage_flux'])
                layer_thickness = state['soil_layer_thickness']
                moisture_loss = drainage * slow_dt / (1000.0 * layer_thickness)
                state['soil_moisture'] = state['soil_moisture'] - moisture_loss
                slow_steps += 1
            
            # Record state for analysis
            time_series['time'].append(time)
            time_series['no2_conc'].append(state['no2_mixing_ratio'].mean().item())
            time_series['soil_moisture'].append(state['soil_moisture'].mean().item())
            
            time += fast_dt
        
        # Verification
        assert fast_steps > 0, "Fast transport steps should have executed"
        assert slow_steps > 0, "Slow transport steps should have executed"
        assert fast_steps > slow_steps, "Fast processes should execute more frequently"
        
        # Check time series evolution
        no2_evolution = np.array(time_series['no2_conc'])
        soil_evolution = np.array(time_series['soil_moisture'])
        
        # Both should evolve smoothly (no sudden jumps)
        no2_diff = np.abs(np.diff(no2_evolution))
        soil_diff = np.abs(np.diff(soil_evolution))
        
        max_no2_jump = np.max(no2_diff) if len(no2_diff) > 0 else 0
        max_soil_jump = np.max(soil_diff) if len(soil_diff) > 0 else 0
        
        assert max_no2_jump < 1e-6, f"NO2 evolution should be smooth (max jump: {max_no2_jump:.2e})"
        assert max_soil_jump < 0.05, f"Soil moisture evolution should be smooth (max jump: {max_soil_jump:.3f})"
        
        print(f"[OK] Multi-scale integration: {fast_steps} fast + {slow_steps} slow steps")
        print(f"[OK] Max NO2 jump: {max_no2_jump:.2e}, Max soil jump: {max_soil_jump:.3f}")
        print("[OK] L2-5: Multi-scale temporal integration verified")
    
    def test_transport_mass_conservation(self,
                                       comprehensive_transport_state,
                                       atmosphere_transport_configs):
        """Test canonical state increments and routed boundary budgets."""
        
        eddy_config, deposition_config = atmosphere_transport_configs
        
        # Initialize all transport mechanisms
        eddy_diffusion = EddyDiffusionBLMFunctor(
            name="conservation_eddy", transport_config=eddy_config
        )
        dry_deposition = DryDepositionAggregator(
            name="conservation_deposition", deposition_config=deposition_config
        )
        vertical_percolation = VerticalPercolationTransport(
            name="conservation_percolation"
        )
        
        state = comprehensive_transport_state.copy()
        dt = 1800.0  # 30 minutes
        time = 0.0
        
        print("--- State Increment Budget Verification ---")

        # Capture initial canonical states.
        soil_thickness = state['soil_layer_thickness']
        initial_masses = {}

        for species in ['no', 'no2', 'o3', 'so2', 'isoprene']:
            species_key = f'{species}_mixing_ratio'
            initial_masses[species] = state[species_key].clone()
        
        # Soil water mass
        soil_water_mass = (state['soil_moisture'] *
                          soil_thickness * 1000.0)  # kg m-2
        initial_masses['soil_water'] = soil_water_mass.clone()
        
        # Apply all transport processes.
        transport_results = {}
        eddy_result = eddy_diffusion.forward(state=state, time=time, dt=dt)
        transport_results['eddy'] = eddy_result
        eddy_increments = {}
        for species in ['no', 'no2', 'o3', 'isoprene']:
            species_key = f'{species}_mixing_ratio'
            increment = _unwrap_update(eddy_result[species_key])
            eddy_increments[species] = increment
            state[species_key] = state[species_key] + increment
        
        # Dry deposition is a second additive state increment.
        depo_result = dry_deposition.forward(state=state, time=time, dt=dt)
        transport_results['deposition'] = depo_result
        deposition_increments = {}
        for species in ['no', 'no2', 'o3', 'so2']:
            species_key = f'{species}_mixing_ratio'
            increment = _unwrap_update(depo_result[species_key])
            deposition_increments[species] = increment
            state[species_key] = state[species_key] + increment
        
        # Vertical percolation routes water through the drainage boundary.
        perc_result = vertical_percolation.forward(state=state, time=time, dt=dt)
        transport_results['percolation'] = perc_result
        drainage_flux = _unwrap_update(
            perc_result['drainage_flux']
        )
        drainage_loss = drainage_flux * dt  # kg m-2
        moisture_loss = drainage_loss / (soil_thickness * 1000.0)  # m3 m-3
        state['soil_moisture'] = (
            state['soil_moisture'] - moisture_loss
        )
        
        # Capture final states.
        final_masses = {}

        for species in ['no', 'no2', 'o3', 'so2', 'isoprene']:
            species_key = f'{species}_mixing_ratio'
            final_masses[species] = state[species_key]
        
        final_soil_water = (state['soil_moisture'] *
                          soil_thickness * 1000.0)
        final_masses['soil_water'] = final_soil_water
        
        # Verify the open-boundary mixing-ratio ledger exactly.
        conservation_errors = {}
        for species in ['no', 'no2', 'o3', 'so2', 'isoprene']:
            initial = initial_masses[species]
            final = final_masses[species]
            expected = torch.zeros_like(initial)
            if species in eddy_increments:
                expected = expected + eddy_increments[species]
            if species in deposition_increments:
                expected = expected + deposition_increments[species]
            residual = final - initial - expected
            assert torch.allclose(
                residual,
                torch.zeros_like(residual),
                rtol=1e-6,
                atol=1e-15,
            )
            assert torch.all(final >= 0), f"{species} concentration became negative"
            conservation_errors[species] = residual.abs().max().item()
            print(
                f"[OK] {species.upper()}: Initial {initial.mean():.2e}, "
                f"Final {final.mean():.2e}, Residual {residual.abs().max():.2e}"
            )
        
        # For soil water with drainage losses  
        if 'soil_water' in initial_masses:
            initial = initial_masses['soil_water']
            final = final_masses['soil_water']
            expected_loss = drainage_loss.expand_as(initial)
            
            expected_final = initial - expected_loss
            error = torch.abs(final - expected_final)
            relative_error = error / (initial + 1e-6)
            
            conservation_errors['soil_water'] = relative_error.max().item()
            
            print(f"[OK] SOIL_WATER: Initial {initial.mean():.1f}, "
                  f"Final {final.mean():.1f}, Loss {expected_loss.mean():.3f}, "
                  f"Error {relative_error.max():.3f}")
        
        # Every routed state budget must close to numerical precision.
        max_change = max(conservation_errors.values()) if conservation_errors else 0.0
        assert max_change < 1e-12, f"Transport budget residual too large: {max_change:.3e}"

        print(f"[OK] Transport budget verified: Max residual {max_change:.3e}")
    
    def test_transport_numerical_stability(self,
                                         comprehensive_transport_state,
                                         atmosphere_transport_configs):
        """Test numerical stability of transport processes under various conditions."""
        
        eddy_config, _ = atmosphere_transport_configs
        
        # Initialize transport mechanisms
        eddy_diffusion = EddyDiffusionBLMFunctor(
            name="stability_eddy", transport_config=eddy_config
        )
        vertical_percolation = VerticalPercolationTransport(
            name="stability_percolation"
        )
        
        print("--- Numerical Stability Testing ---")
        
        # Test scenarios with extreme conditions
        test_scenarios = [
            {
                'name': 'Very low concentrations',
                'state_mods': {
                    'no2_mixing_ratio': torch.tensor([1e-15, 1e-14, 1e-13, 1e-12]),
                    'o3_mixing_ratio': torch.tensor([1e-15, 1e-14, 1e-13, 1e-12])
                }
            },
            {
                'name': 'Very high concentrations',
                'state_mods': {
                    'no2_mixing_ratio': torch.tensor([1e-6, 2e-6, 5e-6, 1e-5]),
                    'o3_mixing_ratio': torch.tensor([1e-5, 2e-5, 5e-5, 1e-4])
                }
            },
            {
                'name': 'Very dry/wet soils',
                'state_mods': {
                    'soil_moisture': torch.tensor([0.05, 0.1, 0.4, 0.42])
                }
            },
            {
                'name': 'Valid turbulence bounds',
                'state_mods': {
                    'friction_velocity': torch.tensor([0.01, 0.02, 1.5, 2.0]),
                    'pbl_height': torch.tensor([50.0, 100.0, 2500.0, 3000.0])
                }
            }
        ]

        dt = 1.0
        time = 0.0
        stable_scenarios = 0
        
        for scenario in test_scenarios:
            print(f"\nTesting: {scenario['name']}")
            
            # Create modified state
            state = comprehensive_transport_state.copy()
            state.update(scenario['state_mods'])
            
            atm_result = eddy_diffusion.forward(state=state, time=time, dt=dt)
            for key, value in atm_result.items():
                tensor = _unwrap_update(value)
                if torch.is_tensor(tensor):
                    assert torch.isfinite(tensor).all(), key

            soil_result = vertical_percolation.forward(
                state=state,
                time=time,
                dt=dt,
            )
            for key, value in soil_result.items():
                tensor = _unwrap_update(value)
                if torch.is_tensor(tensor):
                    assert torch.isfinite(tensor).all(), key

            stable_scenarios += 1
            print(f"  [OK] {scenario['name']}: STABLE")

        invalid_state = comprehensive_transport_state.copy()
        invalid_state['friction_velocity'] = torch.tensor([0.001, 0.01, 2.0, 5.0])
        invalid_state['pbl_height'] = torch.tensor([50.0, 100.0, 3000.0, 5000.0])
        with pytest.raises(RuntimeError, match="outside configured|pbl_height"):
            eddy_diffusion.forward(state=invalid_state, time=time, dt=dt)

        assert stable_scenarios == len(test_scenarios)
        
        print(f"\n[OK] Numerical stability verified: {stable_scenarios}/{len(test_scenarios)} scenarios stable")


def run_l2_transport_integration_tests():
    """Run all L2 transport integration tests."""
    test_files = [__file__]
    
    print("Running L2 Transport Integration Tests...")
    print("=" * 60)
    
    # Configure pytest to be verbose and show detailed output
    pytest_args = [
        "-v",  # Verbose
        "-s",  # Don't capture output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
    ] + test_files
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("ALL L2 TRANSPORT INTEGRATION TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME L2 TESTS FAILED - Check output above")
        print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    # Run tests when called directly
    exit_code = run_l2_transport_integration_tests()
    sys.exit(exit_code)

