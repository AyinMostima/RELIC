"""
L2 Cross-Module Integration Tests
================================

Tests the integration between different scientific sphere modules.
Focuses on:
1. Cross-sphere data flow and coupling (atmosphere-biosphere-hydrosphere)
2. Module interdependencies and scientific workflows
3. State consistency across different scientific domains
4. End-to-end scientific process chains

References:
- Earth System Model Coupling Protocols
- RELIC Multi-Scale Integration Framework
- Scientific Process Interaction Validation
"""

import pytest
import torch

# Atmosphere modules
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor

# Biosphere modules
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

# Hydrosphere modules
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET

# Transport modules
from relic.scientific.transport.diffusion import FickianDiffusion

from relic.core.functor import ExecutionError
from relic.core.state import UpdateItem, UpdateMode


CHEMISTRY_EXPLICIT_DT_SECONDS = 1.0


def _unwrap_update(value):
    """Return the tensor payload carried by an UpdateItem."""
    return value.value if isinstance(value, UpdateItem) else value


def _module_timestep(module, coupling_dt):
    """Keep explicit chemistry inside its positivity domain."""
    if isinstance(module, ChemistryFunctor):
        return CHEMISTRY_EXPLICIT_DT_SECONDS
    return coupling_dt


def _apply_chemistry_increments(state, outputs):
    """Apply canonical pre-integrated chemistry increments exactly once."""
    for key in (
        'no_mixing_ratio',
        'no2_mixing_ratio',
        'o3_mixing_ratio',
        'isoprene_mixing_ratio',
    ):
        if key in outputs:
            state[key] = state[key] + outputs[key].value


def _apply_update_items(state, outputs):
    """Apply explicit UpdateItem semantics without name-based inference."""
    for key, raw_value in outputs.items():
        if not isinstance(raw_value, UpdateItem):
            continue
        if raw_value.mode == UpdateMode.REPLACE:
            state[key] = raw_value.value
        elif raw_value.mode == UpdateMode.ACCUMULATE:
            if key not in state:
                state[key] = raw_value.value
            else:
                state[key] = state[key] + raw_value.value


def _all_output_tensors_finite(outputs):
    """Return True when every tensor payload is finite."""
    tensors = []
    for raw_value in outputs.values():
        value = _unwrap_update(raw_value)
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            tensors.extend(item for item in value.values() if isinstance(item, torch.Tensor))
    return all(bool(torch.isfinite(value).all()) for value in tensors)


class TestL2CrossModuleIntegration:
    """Test integration across different scientific sphere modules."""

    @pytest.fixture
    def comprehensive_earth_system_state(self):
        """Create a comprehensive Earth system state with all required variables."""
        batch_size = 4
        
        # Spatial grid configuration
        grid_shape = (batch_size, 10, 10)  # 4 time steps, 10x10 spatial grid
        spatial_shape = (batch_size,)
        
        return {
            # === ATMOSPHERIC STATE ===
            # Basic meteorological variables
            'air_temperature': torch.tensor([298.15, 300.15, 296.15, 302.15]),
            'surface_temperature': torch.tensor([297.15, 299.15, 295.15, 301.15]),
            'wind_speed': torch.tensor([3.0, 4.0, 2.5, 5.0]),
            'friction_velocity': torch.tensor([0.3, 0.4, 0.25, 0.5]),
            'relative_humidity': torch.tensor([0.7, 0.65, 0.75, 0.6]),
            'atmospheric_pressure': torch.tensor([101325.0, 101300.0, 101350.0, 101275.0]),
            'precipitation': torch.tensor([0.0, 2.0, 1.0, 0.5]),  # mm/hr
            
            # Radiation components
            'solar_radiation': torch.tensor([800.0, 850.0, 750.0, 900.0]),  # W m-2
            'downwelling_shortwave': torch.tensor([1000.0, 1050.0, 950.0, 1100.0]),
            'incoming_shortwave_potential': torch.tensor([1200.0, 1250.0, 1150.0, 1300.0]),  # W m-2 - potential solar input
            'net_radiation': torch.tensor([400.0, 450.0, 350.0, 500.0]),  # W m-2 - net radiation for ET
            'cos_zenith_angle': torch.tensor([0.8, 0.9, 0.7, 0.85]),
            'cos_zenith': torch.tensor([0.8, 0.9, 0.7, 0.85]),
            'cloud_fraction': torch.tensor([0.2, 0.1, 0.4, 0.0]),
            'albedo': torch.tensor([0.15, 0.18, 0.12, 0.22]),
            'surface_albedo': torch.tensor([0.15, 0.18, 0.12, 0.22]),
            'surface_emissivity': torch.tensor([0.97, 0.96, 0.98, 0.95]),
            
            # Atmospheric chemistry
            'no_mixing_ratio': torch.tensor([2e-9, 3e-9, 1.5e-9, 4e-9]),
            'no2_mixing_ratio': torch.tensor([5e-9, 7e-9, 3e-9, 8e-9]),
            'o3_mixing_ratio': torch.tensor([50e-9, 60e-9, 45e-9, 70e-9]),
            'co_mixing_ratio': torch.tensor([150e-9, 150e-9, 150e-9, 150e-9]),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor([1.8e-6, 1.8e-6, 1.8e-6, 1.8e-6]),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor([0.01, 0.01, 0.01, 0.01]),  # O(1D) + H2O OH source, 10 g/kg
            'isoprene_mixing_ratio': torch.tensor([1e-9, 2e-9, 0.5e-9, 3e-9]),
            'oh_concentration': torch.tensor([1e6, 1.5e6, 0.8e6, 2e6]),  # molecules cm-3
            
            # Atmospheric turbulence
            'boundary_layer_height': torch.tensor([800.0, 1000.0, 600.0, 1200.0]),
            'roughness_length': torch.tensor([0.1, 0.15, 0.08, 0.2]),
            'reference_height': torch.tensor([10.0, 10.0, 10.0, 10.0]),
            'roughness_length_momentum': torch.tensor([0.1, 0.15, 0.08, 0.2]),
            'displacement_height': torch.tensor([0.0, 0.0, 0.0, 0.0]),
            'monin_obukhov_length': torch.tensor([-50.0, -100.0, 200.0, -30.0]),
            'sensible_heat_flux': torch.tensor([150.0, 200.0, 100.0, 250.0]),
            'latent_heat_flux': torch.tensor([300.0, 350.0, 250.0, 400.0]),
            
            # === BIOSPHERE STATE ===
            # Photosynthesis and plant physiology
            'leaf_area_index': torch.tensor([4.0, 4.5, 3.5, 5.0]),
            'stomatal_conductance': torch.tensor([0.1, 0.12, 0.08, 0.15]),  # mol m-2 s-1
            'intercellular_co2': torch.tensor([280.0, 290.0, 270.0, 300.0]),  # umol mol-1
            'atmospheric_co2_ppm': torch.tensor([410.0, 410.0, 410.0, 410.0]),
            'ppfd': torch.tensor([1500.0, 1600.0, 1400.0, 1700.0]),  # umol photons m-2 s-1
            'vcmax_scaling': torch.ones(batch_size),
            'jmax_scaling': torch.ones(batch_size),
            'water_stress_factor': torch.ones(batch_size),
            'temperature_stress_factor': torch.ones(batch_size),
            'ozone_stress_factor': torch.ones(batch_size),
            'water_potential': torch.tensor([-0.5, -0.3, -0.8, -0.2]),  # MPa
            
            # Vegetation structure
            'vegetation_height': torch.tensor([15.0, 18.0, 12.0, 25.0]),
            'biomass': torch.tensor([1000.0, 1200.0, 800.0, 1500.0]),  # g m-2
            'root_depth': torch.tensor([0.5, 0.6, 0.4, 0.7]),
            
            # BVOC emissions
            'isoprene_emission_potential': torch.tensor([50.0, 60.0, 40.0, 80.0]),  # ug g-1 hr-1
            'monoterpene_emission_potential': torch.tensor([10.0, 12.0, 8.0, 15.0]),
            
            # === HYDROSPHERE STATE ===
            # Soil water
            'soil_moisture': torch.tensor([0.25, 0.30, 0.22, 0.32]),  # m3 m-3
            'field_capacity': torch.tensor([0.35, 0.40, 0.30, 0.42]),
            'wilting_point': torch.tensor([0.15, 0.18, 0.12, 0.20]),
            'soil_water_potential': torch.tensor([-0.1, -0.05, -0.2, -0.03]),  # MPa
            
            # Evapotranspiration components
            'vapor_pressure_deficit': torch.tensor([1000.0, 1200.0, 800.0, 1500.0]),  # Pa
            'saturated_vapor_pressure': torch.tensor([3.0, 3.5, 2.5, 4.0]),
            'actual_vapor_pressure': torch.tensor([2.0, 2.3, 1.7, 2.5]),
            'aerodynamic_resistance': torch.tensor([50.0, 45.0, 60.0, 40.0]),  # s/m
            'surface_resistance': torch.tensor([100.0, 120.0, 80.0, 150.0]),
            
            # === PEDOSPHERE STATE ===
            # Soil physical properties
            'soil_temperature': torch.tensor([288.15, 290.15, 286.15, 292.15]),
            'soil_bulk_density': torch.tensor([1300.0, 1250.0, 1400.0, 1200.0]),  # kg m-3
            'soil_porosity': torch.tensor([0.45, 0.50, 0.40, 0.52]),
            'soil_ph': torch.tensor([6.5, 6.8, 6.2, 7.0]),
            
            # Soil chemistry and nutrients
            'soil_carbon': torch.tensor([50.0, 60.0, 45.0, 65.0]),  # kg m-2
            'soil_nitrogen': torch.tensor([5.0, 6.0, 4.5, 6.5]),
            'available_nitrogen': torch.tensor([10.0, 12.0, 8.0, 15.0]),  # g m-2
            'soil_phosphorus': torch.tensor([2.0, 2.5, 1.8, 2.8]),
            
            # === ANTHROPOSPHERE STATE ===
            # Land use
            'forest_fraction': torch.tensor([0.6, 0.5, 0.7, 0.4]),
            'crop_fraction': torch.tensor([0.3, 0.4, 0.2, 0.5]),
            'urban_fraction': torch.tensor([0.1, 0.1, 0.1, 0.1]),
            
            # Anthropogenic emissions
            'anthropogenic_nox_flux': torch.tensor([0.1, 0.15, 0.08, 0.2]),  # kg m-2 s-1
            'anthropogenic_co_flux': torch.tensor([0.5, 0.6, 0.4, 0.8]),
            'anthropogenic_voc_flux': torch.tensor([0.3, 0.35, 0.25, 0.45]),
            'anthropogenic_nh3_flux': torch.tensor([0.05, 0.08, 0.03, 0.1]),
            'anthropogenic_so2_flux': torch.tensor([0.02, 0.03, 0.015, 0.04]),  # kg m-2 s-1
            'anthropogenic_pm25_flux': torch.tensor([0.01, 0.012, 0.008, 0.015]),  # kg m-2 s-1
            'pbl_height': torch.full((batch_size,), 1000.0),
            
            # === TRANSPORT STATE ===
            # Concentration fields for transport
            'tracer_concentration': torch.ones(grid_shape) * 0.1,
            'co2_concentration': torch.ones(spatial_shape) * 410.0,  # umol mol-1
            'h2o_concentration': torch.ones(spatial_shape) * 0.02,    # Water vapor
            
            # Transport properties
            'diffusivity': torch.tensor([1e-5, 1.2e-5, 0.8e-5, 1.5e-5]),  # m2 s-1
            'advection_velocity': torch.tensor([0.1, 0.15, 0.08, 0.2]),   # m/s
        }

    @pytest.fixture
    def basic_environment_vars(self):
        """Create basic environment variables for all tests."""
        return {
            'time_of_day': 12.0,  # noon
            'day_of_year': 180,   # summer solstice
            'latitude': 45.0,     # mid-latitude
            'longitude': 0.0,
        }

    def test_atmosphere_biosphere_coupling(self, comprehensive_earth_system_state):
        """Test coupling between atmospheric processes and biosphere."""
        
        # Initialize modules
        radiation = RadiationFunctor()
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        photosynthesis = FarquharPhotosynthesis()
        
        state = comprehensive_earth_system_state.copy()
        dt = 3600.0  # 1 hour
        time = 0.0
        
        # Step 1: Calculate radiation (drives photosynthesis)
        rad_result = radiation(state, time=time, dt=dt)
        _apply_update_items(state, rad_result)
        
        # Step 2: Calculate photosynthesis (uses radiation, affects CO2)
        photo_result = photosynthesis(state, time=time, dt=dt)

        # Step 3: Calculate atmospheric chemistry on its explicit safe step.
        chem_result = chemistry(
            state,
            time=time,
            dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
        )
        
        # Verification
        assert len(rad_result) > 0, "Radiation module should produce outputs"
        assert len(photo_result) > 0, "Photosynthesis module should produce outputs"
        assert len(chem_result) > 0, "Chemistry module should produce outputs"
        
        # Check coupling consistency
        net_rad = _unwrap_update(rad_result['net_radiation'])
        assert torch.all(net_rad > -200), "Net radiation should be reasonable"
        assert torch.all(net_rad < 1200), "Net radiation should not exceed solar input"

        assert 'ppfd' in rad_result
        gross = _unwrap_update(photo_result['gross_photosynthesis'])
        gpp = _unwrap_update(photo_result['gross_primary_productivity'])
        assert torch.all(gross >= 0)
        assert torch.all(gpp >= 0)
        positive = gross > 0
        assert torch.any(positive)
        assert torch.allclose(
            gpp[positive] / gross[positive],
            torch.full_like(gpp[positive], 12.011e-9),
            rtol=1.0e-6,
            atol=1.0e-15,
        )
        assert _all_output_tensors_finite(chem_result)
        
        print("[OK] Atmosphere-biosphere coupling verified")

    def test_hydrosphere_biosphere_coupling(self, comprehensive_earth_system_state):
        """Test coupling between water cycle and vegetation processes."""
        
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        photosynthesis = FarquharPhotosynthesis()
        
        state = comprehensive_earth_system_state.copy()
        dt = 3600.0  # 1 hour
        time = 0.0
        
        # Step 1: Pass the hydrological stress control into photosynthesis.
        control_result = photosynthesis(state, time=time, dt=dt)
        stressed_state = state.copy()
        stressed_state['water_stress_factor'] = torch.full_like(
            state['water_stress_factor'],
            0.5,
        )
        stressed_result = photosynthesis(stressed_state, time=time, dt=dt)

        # Step 2: Diagnose the open surface-water flux independently.
        et_result = evapotranspiration(state, time=time, dt=dt)
        
        # Verification
        assert len(control_result) > 0, "Photosynthesis should produce outputs"
        assert len(et_result) > 0, "Evapotranspiration should produce outputs"

        control_gpp = _unwrap_update(control_result['gross_primary_productivity'])
        stressed_gpp = _unwrap_update(stressed_result['gross_primary_productivity'])
        assert torch.allclose(stressed_gpp, 0.5 * control_gpp, rtol=1.0e-5, atol=1.0e-15)

        et_flux = _unwrap_update(et_result['evapotranspiration_flux'])
        assert torch.all(torch.isfinite(et_flux))
        assert torch.all(et_flux >= 0)
        
        print("[OK] Hydrosphere-biosphere coupling verified")

    def test_atmospheric_transport_chemistry_coupling(self, comprehensive_earth_system_state):
        """Test coupling between atmospheric transport and chemistry."""
        
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        turbulence = TurbulenceFunctor()
        radiation = RadiationFunctor()
        diffusion = FickianDiffusion()
        
        state = comprehensive_earth_system_state.copy()
        dt = 1800.0  # 30 minutes
        time = 0.0
        
        # Step 1: Calculate turbulent mixing (affects transport)
        turb_result = turbulence(state, time=time, dt=dt)
        
        # Update turbulent transport properties
        friction_item = turb_result['friction_velocity']
        resistance_item = turb_result['aerodynamic_resistance']
        assert isinstance(friction_item, UpdateItem)
        assert isinstance(resistance_item, UpdateItem)
        assert friction_item.mode == UpdateMode.REPLACE
        assert resistance_item.mode == UpdateMode.REPLACE
        state['friction_velocity'] = friction_item.value
        state['aerodynamic_resistance'] = resistance_item.value
        
        rad_result = radiation(state, time=time, dt=dt)
        _apply_update_items(state, rad_result)

        # Step 2: Calculate atmospheric chemistry (produces/consumes species)
        chem_result = chemistry(
            state,
            time=time,
            dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
        )
        _apply_chemistry_increments(state, chem_result)
        
        # Step 3: Transport chemical species (uses turbulent properties)
        # Create a simple transport scenario
        transport_state = {
            'tracer_concentration': state['o3_mixing_ratio'].unsqueeze(-1).expand(-1, 10),
            'source_term': torch.zeros((4, 10), dtype=state['o3_mixing_ratio'].dtype),
        }
        diff_result = diffusion(transport_state, time=time, dt=dt)
        
        # Verification
        assert len(turb_result) > 0, "Turbulence should produce outputs"
        assert len(chem_result) > 0, "Chemistry should produce outputs"
        assert isinstance(diff_result['tracer_concentration'], UpdateItem)
        assert diff_result['tracer_concentration'].mode == UpdateMode.ACCUMULATE
        assert torch.isfinite(diff_result['tracer_concentration'].value).all()
        
        # Check turbulence outputs
        u_star = turb_result['friction_velocity'].value
        assert torch.all(u_star > 0), "Friction velocity should be positive"
        assert torch.all(u_star < 2.0), "Friction velocity should be reasonable"
        
        # Check chemical concentrations remain physical
        assert torch.all(state['o3_mixing_ratio'] >= 0), "O3 concentration should be non-negative"
        assert torch.all(state['no_mixing_ratio'] >= 0), "NO concentration should be non-negative"
        
        print("[OK] Atmospheric transport-chemistry coupling verified")

    def test_multi_scale_temporal_coupling(self, comprehensive_earth_system_state):
        """Test coupling processes that operate on different time scales."""
        
        # Fast processes (minutes-hours)
        radiation = RadiationFunctor()
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Medium processes (hours-days)
        photosynthesis = FarquharPhotosynthesis()
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        state = comprehensive_earth_system_state.copy()
        initial_state = {k: v.clone() for k, v in state.items()}
        
        # Exercise two scheduler cadences without violating the explicit
        # atmospheric chemistry positivity-domain time step.
        total_time = 4.0
        fast_dt = CHEMISTRY_EXPLICIT_DT_SECONDS
        medium_dt = 2.0
        
        time = 0.0
        fast_steps = 0
        medium_steps = 0
        
        while time < total_time:
            # Fast processes every 30 minutes
            if time % fast_dt == 0:
                rad_result = radiation(state, time=time, dt=fast_dt)
                _apply_update_items(state, rad_result)
                chem_result = chemistry(
                    state,
                    time=time,
                    dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
                )
                _apply_chemistry_increments(state, chem_result)
                
                fast_steps += 1
            
            # Medium processes every hour
            if time % medium_dt == 0:
                photo_result = photosynthesis(state, time=time, dt=medium_dt)
                et_result = evapotranspiration(state, time=time, dt=medium_dt)
                _apply_update_items(state, photo_result)
                _apply_update_items(state, et_result)
                
                medium_steps += 1
            
            time += fast_dt
        
        # Verification
        assert fast_steps > 0, "Fast processes should have executed"
        assert medium_steps > 0, "Medium processes should have executed"
        assert fast_steps > medium_steps, "Fast processes should execute more frequently"
        
        # Check that state evolved reasonably
        for key, initial_value in initial_state.items():
            if key in state:
                current_value = state[key]
                # State should not have changed dramatically (order of magnitude)
                if torch.is_tensor(current_value) and torch.is_tensor(initial_value):
                    # Handle division by zero in relative change calculation
                    non_zero_mask = torch.abs(initial_value) > 1e-12
                    if torch.any(non_zero_mask):
                        relative_change = torch.abs((current_value - initial_value) / torch.where(non_zero_mask, initial_value, torch.ones_like(initial_value)))
                        # Only check non-zero initial values for relative change
                        valid_changes = relative_change[non_zero_mask]
                        if len(valid_changes) > 0:
                            max_change = torch.max(valid_changes).item()
                            # Skip certain variables that are allowed to change dramatically
                            if key not in ['precipitation', 'cloud_fraction', 'wind_speed']:
                                assert max_change < 10.0, f"{key} changed too dramatically: {max_change}"
        
        print(f"[OK] Multi-scale temporal coupling verified (fast: {fast_steps}, medium: {medium_steps} steps)")

    def test_conservation_across_modules(self, comprehensive_earth_system_state):
        """Test that mass and energy are conserved across module interactions."""
        
        photosynthesis = FarquharPhotosynthesis()
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        radiation = RadiationFunctor()
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        state = comprehensive_earth_system_state.copy()
        dt = 3600.0  # 1 hour
        time = 0.0
        
        # Run the three open-system process ledgers.
        rad_result = radiation(state, time=time, dt=dt)
        _apply_update_items(state, rad_result)
        photo_result = photosynthesis(state, time=time, dt=dt)
        et_result = evapotranspiration(state, time=time, dt=dt)
        chem_result = chemistry(
            state,
            time=time,
            dt=CHEMISTRY_EXPLICIT_DT_SECONDS,
        )

        gross = _unwrap_update(photo_result['gross_photosynthesis'])
        gpp = _unwrap_update(photo_result['gross_primary_productivity'])
        positive = gross > 0
        assert torch.any(positive)
        assert torch.allclose(
            gpp[positive] / gross[positive],
            torch.full_like(gpp[positive], 12.011e-9),
            rtol=1.0e-6,
            atol=1.0e-15,
        )

        no_delta = chem_result['no_mixing_ratio'].value
        no2_delta = chem_result['no2_mixing_ratio'].value
        assert torch.allclose(no_delta + no2_delta, torch.zeros_like(no_delta), atol=1.0e-18)

        et_flux = _unwrap_update(et_result['evapotranspiration_flux'])
        latent_heat_flux = _unwrap_update(et_result['latent_heat_flux'])
        assert torch.all(et_flux >= 0)
        assert torch.allclose(
            latent_heat_flux,
            et_flux * evapotranspiration.latent_heat_vaporization,
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        
        print("[OK] Conservation across modules verified")

    def test_error_propagation_across_modules(self, comprehensive_earth_system_state):
        """Test how errors propagate across module boundaries."""
        
        radiation = RadiationFunctor()
        photosynthesis = FarquharPhotosynthesis()
        
        state = comprehensive_earth_system_state.copy()
        dt = 3600.0
        time = 0.0
        
        # Test 1: Invalid input to upstream module
        state['cos_zenith'] = torch.tensor([-0.5, 1.5, 0.5, 0.8])

        with pytest.raises(ExecutionError, match='cos_zenith'):
            radiation(state, time=time, dt=dt)
        
        # Test 2: Missing required inputs
        state_missing = state.copy()
        del state_missing['ppfd']

        with pytest.raises(ExecutionError, match='ppfd'):
            photosynthesis(state_missing, time=time, dt=dt)
        
        # Test 3: NaN propagation
        state['air_temperature'] = torch.tensor([298.15, float('nan'), 296.15, 302.15])
        
        with pytest.raises(ExecutionError, match='air_temperature'):
            photosynthesis(state, time=time, dt=dt)

        recovered = comprehensive_earth_system_state.copy()
        recovered_result = photosynthesis(recovered, time=time, dt=dt)
        assert _all_output_tensors_finite(recovered_result)

    def test_state_consistency_validation(self, comprehensive_earth_system_state):
        """Test that state remains physically consistent across module calls."""
        
        modules = [
            ('radiation', RadiationFunctor()),
            ('chemistry', ChemistryFunctor(physical_boundary="open_mechanism")),
            ('photosynthesis', FarquharPhotosynthesis()),
            ('evapotranspiration', PenmanMonteithET(config={"canopy_resistance_source": "state"})),
        ]
        
        state = comprehensive_earth_system_state.copy()
        dt = 3600.0
        time = 0.0
        
        # Run all modules in sequence
        for name, module in modules:
            result = module(
                state,
                time=time,
                dt=_module_timestep(module, dt),
            )
            assert result, f"{name} must produce outputs"
            assert _all_output_tensors_finite(result)

            if name == 'chemistry':
                _apply_chemistry_increments(state, result)
            else:
                _apply_update_items(state, result)
        
        # Validate final state consistency
        consistency_checks = [
            ('air_temperature', lambda x: (x > 200) & (x < 350)),
            ('soil_moisture', lambda x: (x >= 0) & (x <= 1)),
            ('atmospheric_co2_ppm', lambda x: (x > 300.0) & (x < 500.0)),
            ('relative_humidity', lambda x: (x >= 0) & (x <= 1)),
            ('soil_ph', lambda x: (x > 0) & (x < 14)),
        ]
        
        for var_name, check_func in consistency_checks:
            if var_name in state:
                value = state[var_name]
                if torch.is_tensor(value):
                    is_valid = check_func(value)
                    assert torch.all(is_valid), f"{var_name} contains out-of-range values"

    def test_performance_and_memory_efficiency(self, comprehensive_earth_system_state):
        """Test computational performance and memory usage across modules."""
        
        import time as time_module
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        modules = [
            ('radiation', RadiationFunctor()),
            ('chemistry', ChemistryFunctor(physical_boundary="open_mechanism")),
            ('turbulence', TurbulenceFunctor()),
            ('photosynthesis', FarquharPhotosynthesis()),
        ]
        
        state = comprehensive_earth_system_state.copy()
        dt = 3600.0
        time = 0.0
        
        performance_metrics = {}
        
        for name, module in modules:
            # Measure execution time
            start_time = time_module.time()
            result = module(
                state,
                time=time,
                dt=_module_timestep(module, dt),
            )
            
            end_time = time_module.time()
            execution_time = end_time - start_time
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            
            performance_metrics[name] = {
                'execution_time': execution_time,
                'memory_current': current / 1024 / 1024,  # MB
                'memory_peak': peak / 1024 / 1024,        # MB
                'output_count': len(result),
            }
            
            # Performance assertions
            assert execution_time < 10.0, f"{name} should complete within 10 seconds"
            assert current < 100 * 1024 * 1024, f"{name} should use < 100MB memory"
            assert result, f"{name} must produce outputs"
            assert _all_output_tensors_finite(result)
            _apply_update_items(state, result)
            
        tracemalloc.stop()
        
        # Overall performance check
        total_time = sum(m['execution_time'] for m in performance_metrics.values())
        assert total_time < 30.0, "Total integration test should complete within 30 seconds"
        
        print(f"[OK] Performance verified: Total time {total_time:.3f}s")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

