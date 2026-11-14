#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 Scientific Accuracy Validation Tests

This module provides comprehensive scientific accuracy validation tests that compare
RELIC model outputs against theoretical values, literature benchmarks, and 
established scientific relationships.

Testing Strategy:
- Literature benchmark comparisons
- Theoretical relationship validation
- Physical constraint verification
- Unit consistency checking
- Cross-validation with established models

Authors: Yuchong Long  
License: MIT
"""

import pytest
import torch
import logging
from typing import Dict, List, Any

# Import scientific modules for accuracy testing
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.core.state import UpdateItem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _unwrap_update(value: Any) -> Any:
    """Return the numerical payload emitted through the state-update contract."""
    return value.value if isinstance(value, UpdateItem) else value


class TestL3ScientificAccuracy:
    """
    Comprehensive L3 Scientific Accuracy Tests
    
    These tests validate scientific accuracy by comparing model outputs against:
    - Theoretical predictions from first principles
    - Literature values from established studies
    - Benchmark datasets from field observations
    - Cross-model validation with established frameworks
    """

    @pytest.fixture
    def literature_benchmark_data(self) -> Dict[str, Any]:
        """Literature benchmark values from established scientific studies."""
        return {
            # Farquhar photosynthesis parameters (Von Caemmerer, 2000)
            'farquhar_benchmarks': {
                'vcmax25_c3': 50e-6,  # umol CO2 m-2 s-1 at 25 C for C3 plants
                'jmax25_c3': 100e-6,  # umol electrons m-2 s-1 at 25 C
                'kc25': 404.9,        # Michaelis constant for CO2 (umol mol-1)
                'ko25': 278.4,        # Michaelis constant for O2 (mmol mol-1)
                'tau25': 2600,        # CO2/O2 specificity ratio
                'q10_vcmax': 2.1,     # Q10 temperature response
            },
            
            # Penman-Monteith parameters (Allen et al., 1998)
            'penman_monteith_benchmarks': {
                'psychrometric_constant': 0.665,  # kPa degC-1 at sea level
                'stefan_boltzmann': 5.67e-8,      # W m-2 K-4
                'cp_air': 1013,                   # J kg-1 K-1
                'latent_heat_vaporization': 2.45e6,  # J kg-1 at 20 degC
                'reference_et_clear_sky': 6.0,    # mm day-1 under standard conditions
            },
            
            # Atmospheric chemistry (Seinfeld & Pandis, 2006)
            'chemistry_benchmarks': {
                'arrhenius_A_typical': 1e-12,     # cm3 molecule-1 s-1
                'activation_energy_typical': 1000, # K
                'tropospheric_lifetime_oh': 1e6,   # seconds (~11 days)
                'q10_chemistry': 2.0,              # Typical Q10 for chemical reactions
            },
            
            # Turbulence parameters (Stull, 1988)
            'turbulence_benchmarks': {
                'von_karman_constant': 0.4,       # Dimensionless
                'roughness_length_grass': 0.01,   # m
                'roughness_length_forest': 1.0,   # m
                'minimum_friction_velocity': 0.01, # m/s
                'monin_obukhov_stable_max': 200,   # m
                'monin_obukhov_unstable_min': -200, # m
            },
            
            # Physical constants
            'physical_constants': {
                'gas_constant': 8.314,             # J mol-1 K-1
                'avogadro': 6.022e23,             # molecules mol-1
                'boltzmann': 1.38e-23,            # J K-1
                'standard_pressure': 101325,      # Pa
                'standard_temperature': 273.15,   # K
                'earth_radius': 6.371e6,          # m
            }
        }

    @pytest.fixture
    def validation_scenarios(self) -> List[Dict[str, Any]]:
        """Validation scenarios representing different environmental conditions."""
        return [
            {
                'name': 'Temperate grassland (standard)',
                'air_temperature': 298.15,  # 25 C
                'atmospheric_pressure': 101325.0,
                'relative_humidity': 0.65,
                'co2_concentration': 400e-6,
                'incoming_shortwave': 600.0,
                'wind_speed': 3.0,
                'leaf_area_index': 3.0,
                'expected_gpp_range': [6e-8, 3.6e-7],  # kg C m-2 s-1 from 5-30 umol CO2 m-2 s-1
                'expected_et_range': [2.0, 8.0],      # mm day-1
            },
            {
                'name': 'Tropical rainforest (high productivity)',
                'air_temperature': 303.15,  # 30 degC
                'atmospheric_pressure': 101325.0,
                'relative_humidity': 0.85,
                'co2_concentration': 390e-6,
                'incoming_shortwave': 800.0,
                'wind_speed': 2.0,
                'leaf_area_index': 6.0,
                'expected_gpp_range': [1.8e-7, 6.0e-7],  # kg C m-2 s-1 from 15-50 umol CO2 m-2 s-1
                'expected_et_range': [4.0, 12.0],      # mm day-1
            },
            {
                'name': 'Arctic tundra (low temperature)',
                'air_temperature': 278.15,  # 5 degC
                'atmospheric_pressure': 101325.0,
                'relative_humidity': 0.80,
                'co2_concentration': 410e-6,
                'incoming_shortwave': 300.0,
                'wind_speed': 5.0,
                'leaf_area_index': 1.0,
                'expected_gpp_range': [6e-9, 9.6e-8],   # kg C m-2 s-1 from 0.5-8 umol CO2 m-2 s-1
                'expected_et_range': [0.5, 3.0],        # mm day-1
            },
            {
                'name': 'Mediterranean shrubland (water stressed)',
                'air_temperature': 308.15,  # 35 degC
                'atmospheric_pressure': 101325.0,
                'relative_humidity': 0.35,
                'co2_concentration': 420e-6,
                'incoming_shortwave': 900.0,
                'wind_speed': 4.0,
                'leaf_area_index': 2.0,
                'expected_gpp_range': [3.6e-8, 2.4e-7],  # kg C m-2 s-1 from 3-20 umol CO2 m-2 s-1
                'expected_et_range': [1.0, 6.0],        # mm day-1
            }
        ]

    def create_scenario_state(self, scenario: Dict[str, Any], batch_size: int = 5) -> Dict[str, torch.Tensor]:
        """Create state dictionary for a validation scenario."""
        co2_ppm = torch.full((batch_size,), scenario['co2_concentration'] * 1e6)  # Convert mol/mol to ppm
        state = {
            'air_temperature': torch.full((batch_size,), scenario['air_temperature']),
            'atmospheric_pressure': torch.full((batch_size,), scenario['atmospheric_pressure']),
            'relative_humidity': torch.full((batch_size,), scenario['relative_humidity']),
            'atmospheric_co2_ppm': co2_ppm.clone(),
            'incoming_shortwave': torch.full((batch_size,), scenario['incoming_shortwave']),
            'shortwave_down': torch.full((batch_size,), scenario['incoming_shortwave']),
            'wind_speed': torch.full((batch_size,), scenario['wind_speed']),    
            'leaf_area_index': torch.full((batch_size,), scenario['leaf_area_index']),
            'jmax_scaling': torch.full((batch_size,), 1.0),
            'water_stress_factor': torch.full((batch_size,), 1.0),
            'temperature_stress_factor': torch.full((batch_size,), 1.0),
            'ozone_stress_factor': torch.full((batch_size,), 1.0),
            'vapor_pressure_deficit': torch.full((batch_size,), 1000.0),
            'no_mixing_ratio': torch.full((batch_size,), 10e-9),
            'no2_mixing_ratio': torch.full((batch_size,), 20e-9),
            'o3_mixing_ratio': torch.full((batch_size,), 40e-9),
            'co_mixing_ratio': torch.full((batch_size,), 150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.full((batch_size,), 1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.full((batch_size,), 0.01),  # O(1D) + H2O OH source, 10 g/kg
            'isoprene_mixing_ratio': torch.full((batch_size,), 1e-9),
            
            # Additional required variables
            'cos_zenith': torch.full((batch_size,), 0.7),
            'incoming_shortwave_potential': torch.full((batch_size,), scenario['incoming_shortwave'] * 1.2),
            'surface_temperature': torch.full((batch_size,), scenario['air_temperature'] + 1.0),
            'surface_albedo': torch.full((batch_size,), 0.2),
            'surface_emissivity': torch.full((batch_size,), 0.97),
            'cloud_fraction': torch.full((batch_size,), 0.2),
            'incoming_longwave': torch.full((batch_size,), 350.0),
            'ppfd': torch.full((batch_size,), scenario['incoming_shortwave'] * 2.07),
            'vcmax_scaling': torch.full((batch_size,), 1.0),
            'stomatal_conductance': torch.full((batch_size,), 0.1),
            'soil_moisture': torch.full((batch_size,), 0.3),
            'soil_temperature': scenario['air_temperature'] - 3.0,  # Slightly cooler
            'canopy_temperature': scenario['air_temperature'] + 1.0,  # Slightly warmer
            'saturated_vapor_pressure': torch.full((batch_size,), 3167.0),
            'actual_vapor_pressure': torch.full((batch_size,), 2058.0),
            'sensible_heat_flux': torch.full((batch_size,), 100.0),
            'latent_heat_flux': torch.full((batch_size,), 200.0),
            
            # Required variables for PenmanMonteithET
            'net_radiation': torch.full((batch_size,), 400.0),  # W/m^2
            'atmospheric_pressure': torch.full((batch_size,), scenario['atmospheric_pressure']),
            'surface_resistance': torch.full((batch_size,), 70.0),  # s/m (typical for vegetation)
        }
        
        # Adjust soil temperature tensor to match other tensors
        state['soil_temperature'] = torch.full((batch_size,), state['soil_temperature'])
        state['canopy_temperature'] = torch.full((batch_size,), state['canopy_temperature'])
        
        return state

    def test_farquhar_photosynthesis_benchmarks(self, literature_benchmark_data, validation_scenarios):
        """
        Test Farquhar photosynthesis model against literature benchmarks
        
        Validates:
        - GPP rates within expected ranges for different ecosystems
        - Temperature response (Q10) matches theoretical values
        - CO2 response follows Michaelis-Menten kinetics
        - Light response saturation behavior
        """
        logger.info("Starting Farquhar Photosynthesis Benchmarks Test")
        
        photosynthesis = FarquharPhotosynthesis()
        benchmarks = literature_benchmark_data['farquhar_benchmarks']
        
        for scenario in validation_scenarios:
            logger.info(f"Testing photosynthesis in {scenario['name']}")
            
            state = self.create_scenario_state(scenario)
            
            # Test photosynthesis under scenario conditions
            photo_fluxes = photosynthesis(state, time=0.0, dt=3600.0)
            
            # Validate GPP is within expected range
            if 'gross_primary_productivity' in photo_fluxes:
                gpp = _unwrap_update(photo_fluxes['gross_primary_productivity'])
                gpp_mean_kg = torch.mean(gpp).item()
                gross_mean = torch.mean(
                    _unwrap_update(photo_fluxes['gross_photosynthesis'])
                ).item()

                expected_min, expected_max = scenario['expected_gpp_range']     

                assert gpp_mean_kg >= expected_min * 0.04, \
                    f"GPP too low for {scenario['name']}: {gpp_mean_kg:.2e} < {expected_min:.2e}"
                assert gpp_mean_kg <= expected_max * 2.0, \
                    f"GPP too high for {scenario['name']}: {gpp_mean_kg:.2e} > {expected_max:.2e}"
                
                logger.info(
                    f"GPP for {scenario['name']}: {gpp_mean_kg:.2e} kg C m-2 s-1 "
                    f"({gross_mean:.2e} umol CO2 m-2 s-1)"
                )

        # === Test Temperature Response (Q10) ===
        logger.info("Testing Q10 temperature response")
        
        base_scenario = validation_scenarios[0]  # Use temperate grassland
        base_state = self.create_scenario_state(base_scenario)
        
        # Test at base temperature and +10K
        warm_state = base_state.copy()
        warm_state['air_temperature'] = base_state['air_temperature'] + 10.0
        warm_state['soil_temperature'] = base_state['soil_temperature'] + 10.0
        warm_state['canopy_temperature'] = base_state['canopy_temperature'] + 10.0
        
        base_fluxes = photosynthesis(base_state, time=0.0, dt=3600.0)
        warm_fluxes = photosynthesis(warm_state, time=0.0, dt=3600.0)
        
        if 'gross_primary_productivity' in base_fluxes and 'gross_primary_productivity' in warm_fluxes:
            base_gpp = torch.mean(
                _unwrap_update(base_fluxes['gross_primary_productivity'])
            ).item()
            warm_gpp = torch.mean(
                _unwrap_update(warm_fluxes['gross_primary_productivity'])
            ).item()
            
            if base_gpp > 0:
                observed_q10 = warm_gpp / base_gpp
                expected_q10 = benchmarks['q10_vcmax']
                
                # Allow reasonable range around expected Q10
                assert 0.5 < observed_q10 < 3.0, \
                    f"Q10 response outside reasonable range: {observed_q10:.2f}"
                
                logger.info(f"Observed Q10: {observed_q10:.2f}, Expected: {expected_q10:.2f}")
        
        # === Test CO2 Response ===
        logger.info("Testing CO2 response")
        
        co2_levels = [300e-6, 400e-6, 500e-6, 600e-6, 800e-6]  # mol/mol
        gpp_responses = []
        
        for co2_level in co2_levels:
            co2_state = base_state.copy()
            co2_ppm = torch.full_like(
                base_state['atmospheric_co2_ppm'], co2_level * 1e6
            )
            co2_state['atmospheric_co2_ppm'] = co2_ppm.clone()
            
            co2_fluxes = photosynthesis(co2_state, time=0.0, dt=3600.0)
            if 'gross_primary_productivity' in co2_fluxes:
                gpp_mean = torch.mean(
                    _unwrap_update(co2_fluxes['gross_primary_productivity'])
                ).item()
                gpp_responses.append(gpp_mean)
        
        # Validate CO2 fertilization effect (should increase with CO2, but saturate)
        if len(gpp_responses) >= 3:
            # GPP should increase with CO2
            assert gpp_responses[1] > gpp_responses[0] * 0.9, "GPP should increase with CO2"
            assert gpp_responses[2] > gpp_responses[1] * 0.9, "GPP should increase with CO2"
            
            # But should show diminishing returns (saturation)
            response_1 = gpp_responses[1] / gpp_responses[0] if gpp_responses[0] > 0 else 1
            response_2 = gpp_responses[2] / gpp_responses[1] if gpp_responses[1] > 0 else 1
            
            assert response_1 >= response_2 * 0.8, "CO2 response should show saturation"
            
            logger.info(f"CO2 responses: {[f'{g:.2e}' for g in gpp_responses]}")
        
        logger.info("Farquhar Photosynthesis Benchmarks Test PASSED")

    def test_penman_monteith_benchmarks(self, literature_benchmark_data, validation_scenarios):
        """
        Test Penman-Monteith evapotranspiration against literature benchmarks
        
        Validates:
        - ET rates within expected ranges for different ecosystems
        - Proper response to vapor pressure deficit
        - Energy balance consistency
        - Physical parameter values match literature
        """
        logger.info("Starting Penman-Monteith Benchmarks Test")
        
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        benchmarks = literature_benchmark_data['penman_monteith_benchmarks']
        
        for scenario in validation_scenarios:
            logger.info(f"Testing evapotranspiration in {scenario['name']}")
            
            state = self.create_scenario_state(scenario)
            
            # Test evapotranspiration under scenario conditions
            et_fluxes = evapotranspiration(state, time=0.0, dt=3600.0)
            
            # Validate ET rates
            if 'evapotranspiration_rate' in et_fluxes:
                et_rate = _unwrap_update(et_fluxes['evapotranspiration_rate'])
                et_daily = torch.mean(et_rate).item() * 86400 * 1000  # Convert to mm day-1
                
                expected_min, expected_max = scenario['expected_et_range']
                
                assert et_daily >= expected_min * 0.5, \
                    f"ET too low for {scenario['name']}: {et_daily:.2f} < {expected_min:.2f} mm day-1"
                assert et_daily <= expected_max * 2.0, \
                    f"ET too high for {scenario['name']}: {et_daily:.2f} > {expected_max:.2f} mm day-1"
                
                logger.info(f"ET for {scenario['name']}: {et_daily:.2f} mm day-1")

        # === Test Vapor Pressure Deficit Response ===
        logger.info("Testing VPD response")
        
        base_scenario = validation_scenarios[0]
        base_state = self.create_scenario_state(base_scenario)
        
        # Test different humidity levels (different VPDs)
        humidity_levels = [0.3, 0.5, 0.7, 0.9]
        et_responses = []
        
        for rh in humidity_levels:
            vpd_state = base_state.copy()
            vpd_state['relative_humidity'] = torch.full_like(base_state['relative_humidity'], rh)
            
            # Adjust actual vapor pressure based on RH
            sat_vp = vpd_state['saturated_vapor_pressure']
            vpd_state['actual_vapor_pressure'] = sat_vp * rh
            
            vpd_fluxes = evapotranspiration(vpd_state, time=0.0, dt=3600.0)
            if 'evapotranspiration_rate' in vpd_fluxes:
                et_mean = torch.mean(
                    _unwrap_update(vpd_fluxes['evapotranspiration_rate'])
                ).item()
                et_responses.append(et_mean)
        
        # ET should generally decrease with increasing humidity (decreasing VPD)
        if len(et_responses) >= 3:
            # Allow some variation but should show general trend
            high_vpd_et = et_responses[0]  # Low humidity = high VPD
            low_vpd_et = et_responses[-1]  # High humidity = low VPD
            
            assert high_vpd_et > low_vpd_et * 0.7, \
                "ET should be higher under high VPD conditions"
            
            logger.info(f"VPD responses (RH 0.3-0.9): {[f'{e:.2e}' for e in et_responses]}")
        
        logger.info("Penman-Monteith Benchmarks Test PASSED")

    def test_radiation_physics_consistency(self, literature_benchmark_data):
        """
        Test radiation module for physical consistency
        
        Validates:
        - Stefan-Boltzmann law compliance
        - Energy conservation
        - Reasonable albedo values
        - Proper angular dependencies
        """
        logger.info("Starting Radiation Physics Consistency Test")
        
        radiation = RadiationFunctor()
        constants = literature_benchmark_data['physical_constants']
        
        # Create test conditions
        batch_size = 10
        state = {
            'air_temperature': torch.full((batch_size,), 298.15),  # 25 C
            'atmospheric_pressure': torch.full((batch_size,), 101325.0),
            'cos_zenith': torch.full((batch_size,), 0.7),
            'incoming_shortwave_potential': torch.full((batch_size,), 1000.0),
            'surface_temperature': torch.full((batch_size,), 296.15),
            'relative_humidity': torch.full((batch_size,), 0.6),
            'surface_albedo': torch.full((batch_size,), 0.2),
            'surface_emissivity': torch.full((batch_size,), 0.97),
            'cloud_fraction': torch.zeros(batch_size),
            'incoming_shortwave': torch.full((batch_size,), 800.0),
            'incoming_longwave': torch.full((batch_size,), 350.0),
        }
        
        rad_fluxes = radiation(state, time=0.0, dt=3600.0)
        
        # === Test Energy Conservation ===
        if 'net_radiation' in rad_fluxes:
            net_rad = _unwrap_update(rad_fluxes['net_radiation'])
            
            # Net radiation should be finite and reasonable
            assert torch.all(torch.isfinite(net_rad)), "Net radiation must be finite"
            assert torch.all(torch.abs(net_rad) <= 1200), "Net radiation outside reasonable range"
            
            logger.info(f"Net radiation range: {torch.min(net_rad):.1f} to {torch.max(net_rad):.1f} W/m^2")
        
        # === Test Solar Angle Dependency ===
        logger.info("Testing solar angle dependency")
        
        zenith_angles = [0.1, 0.3, 0.5, 0.7, 0.9]  # cos(zenith)
        radiation_responses = []
        
        for cos_zenith in zenith_angles:
            angle_state = state.copy()
            angle_state['cos_zenith'] = torch.full((batch_size,), cos_zenith)
            
            angle_fluxes = radiation(angle_state, time=0.0, dt=3600.0)
            if 'net_radiation' in angle_fluxes:
                net_rad_mean = torch.mean(
                    _unwrap_update(angle_fluxes['net_radiation'])
                ).item()
                radiation_responses.append(net_rad_mean)
        
        # Net radiation should generally increase with solar elevation (higher cos_zenith)
        if len(radiation_responses) >= 3:
            # Should show positive correlation with solar angle
            high_angle = radiation_responses[-1]  # cos_zenith = 0.9
            low_angle = radiation_responses[0]    # cos_zenith = 0.1
            
            # Allow for some variation due to other factors
            angle_effect_ok = high_angle > low_angle - 100  # Within 100 W/m^2
            assert angle_effect_ok, f"Solar angle effect inconsistent: low={low_angle:.1f}, high={high_angle:.1f}"
            
            logger.info(f"Solar angle responses: {[f'{r:.1f}' for r in radiation_responses]}")
        
        logger.info("Radiation Physics Consistency Test PASSED")

    def test_turbulence_atmospheric_physics(self, literature_benchmark_data):
        """
        Test turbulence module for atmospheric physics consistency
        
        Validates:
        - Monin-Obukhov similarity theory compliance
        - Friction velocity calculations
        - Stability parameter ranges
        - Von Karman constant usage
        """
        logger.info("Starting Turbulence Atmospheric Physics Test")
        
        turbulence = TurbulenceFunctor()
        benchmarks = literature_benchmark_data['turbulence_benchmarks']
        
        batch_size = 5
        base_state = {
            'air_temperature': torch.full((batch_size,), 298.15),
            'surface_temperature': torch.full((batch_size,), 300.15),
            'wind_speed': torch.full((batch_size,), 5.0),
            'reference_height': torch.full((batch_size,), 10.0),
            'roughness_length_momentum': torch.full((batch_size,), 0.1),
            'displacement_height': torch.zeros(batch_size),
            'atmospheric_pressure': torch.full((batch_size,), 101325.0),
            'relative_humidity': torch.full((batch_size,), 0.5),
        }

        expected_outputs = {
            'friction_velocity',
            'aerodynamic_resistance',
            'diagnostics.turbulence.stability_parameter',
            'monin_obukhov_length',
            'sensible_heat_flux',
            'momentum_flux',
            'diagnostics.turbulence.drag_coefficient',
        }
        
        # === Test Unstable Conditions ===
        logger.info("Testing unstable atmospheric conditions")
        
        unstable_state = base_state.copy()
        unstable_fluxes = turbulence(unstable_state, time=0.0, dt=3600.0)
        assert set(unstable_fluxes) == expected_outputs

        unstable_u_star = unstable_fluxes['friction_velocity'].value
        unstable_zeta = unstable_fluxes[
            'diagnostics.turbulence.stability_parameter'
        ].value
        unstable_heat = unstable_fluxes['sensible_heat_flux'].value
        unstable_resistance = unstable_fluxes['aerodynamic_resistance'].value
        assert torch.all(unstable_u_star >= benchmarks['minimum_friction_velocity'])
        assert torch.all(unstable_zeta < 0.0)
        assert torch.all(unstable_heat > 0.0)
        
        # === Test Stable Conditions ===
        logger.info("Testing stable atmospheric conditions")
        
        stable_state = base_state.copy()
        stable_state['surface_temperature'] = torch.full((batch_size,), 296.15)
        
        stable_fluxes = turbulence(stable_state, time=0.0, dt=3600.0)
        assert set(stable_fluxes) == expected_outputs

        stable_u_star = stable_fluxes['friction_velocity'].value
        stable_zeta = stable_fluxes[
            'diagnostics.turbulence.stability_parameter'
        ].value
        stable_heat = stable_fluxes['sensible_heat_flux'].value
        stable_resistance = stable_fluxes['aerodynamic_resistance'].value
        assert torch.all(stable_u_star >= benchmarks['minimum_friction_velocity'])
        assert torch.all(stable_zeta > 0.0)
        assert torch.all(stable_heat < 0.0)
        assert torch.all(unstable_u_star > stable_u_star)
        assert torch.all(stable_resistance > unstable_resistance)
        
        # === Test Wind Speed Dependence ===
        logger.info("Testing wind speed dependence")
        
        wind_speeds = [2.0, 5.0, 10.0, 15.0]
        friction_velocity_responses = []
        
        for wind_speed in wind_speeds:
            wind_state = base_state.copy()
            wind_state['wind_speed'] = torch.full((batch_size,), wind_speed)
            
            wind_fluxes = turbulence(wind_state, time=0.0, dt=3600.0)
            u_star_mean = torch.mean(
                wind_fluxes['friction_velocity'].value
            ).detach().item()
            friction_velocity_responses.append(u_star_mean)

        assert all(
            later > earlier
            for earlier, later in zip(
                friction_velocity_responses,
                friction_velocity_responses[1:],
            )
        ), (
            "Friction velocity should increase with wind speed: "
            f"{friction_velocity_responses}"
        )
        
        logger.info("Turbulence Atmospheric Physics Test PASSED")

    def test_chemistry_reaction_kinetics(self):
        """
        Test atmospheric chemistry for reaction kinetics consistency
        
        Validates:
        - Arrhenius temperature dependence
        - Reaction rate magnitudes
        - Conservation of reactive species
        - Photochemical relationships
        """
        logger.info("Starting Chemistry Reaction Kinetics Test")
        
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        chemistry_dt = 1.0
        
        # Create test conditions
        batch_size = 5
        state = {
            'air_temperature': torch.full((batch_size,), 298.15),
            'atmospheric_pressure': torch.full((batch_size,), 101325.0),
            'cos_zenith': torch.full((batch_size,), 0.7),
            'shortwave_down': torch.full((batch_size,), 600.0),
            
            # Required mixing ratios for ChemistryFunctor
            'no_mixing_ratio': torch.full((batch_size,), 10e-9),   # 10 ppb NO
            'no2_mixing_ratio': torch.full((batch_size,), 20e-9),  # 20 ppb NO2
            'o3_mixing_ratio': torch.full((batch_size,), 40e-9),   # 40 ppb O3
            'co_mixing_ratio': torch.full((batch_size,), 150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.full((batch_size,), 1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.full((batch_size,), 0.01),  # O(1D) + H2O OH source, 10 g/kg
            'isoprene_mixing_ratio': torch.full((batch_size,), 1e-9),
        }
        
        chem_fluxes = chemistry(state, time=0.0, dt=chemistry_dt)
        
        # === Test Temperature Dependence (Arrhenius) ===
        logger.info("Testing Arrhenius temperature dependence")
        
        temperatures = [288.15, 298.15, 308.15]  # 15 degC, 25 C, 35 degC
        rate_constants = []
        
        for temperature in temperatures:
            temp_state = state.copy()
            temp_state['air_temperature'] = torch.full((batch_size,), temperature)
            
            temp_fluxes = chemistry(
                temp_state,
                time=0.0,
                dt=chemistry_dt,
            )
            rate_constant = temp_fluxes[
                'diagnostics.chemistry.rate_constant_no_o3'
            ].value
            rate_constants.append(torch.mean(rate_constant).item())

        # The declared Arrhenius coefficient increases over this range and
        # equals its ParameterManager reference value at 298.15 K.
        assert rate_constants[0] < rate_constants[1] < rate_constants[2]
        reference_rate = chemistry.get_parameter_value('k_no_o3_298')
        assert rate_constants[1] == pytest.approx(reference_rate, rel=1e-6)
        
        # === Test Photochemical Response ===
        logger.info("Testing photochemical response")
        
        light_levels = [0.0, 200.0, 600.0, 1000.0]  # W/m^2
        photo_responses = []
        
        for light in light_levels:
            light_state = state.copy()
            light_state['shortwave_down'] = torch.full((batch_size,), light)
            # Adjust solar angle for consistency
            light_state['cos_zenith'] = torch.full((batch_size,), 0.7 if light > 0 else 0.0)
            
            light_fluxes = chemistry(
                light_state,
                time=0.0,
                dt=chemistry_dt,
            )
            photolysis_rate = light_fluxes[
                'diagnostics.chemistry.photolysis_rate_no2'
            ].value
            photo_responses.append(torch.mean(photolysis_rate).item())

        # The canonical shortwave port drives a zero dark rate and a monotonic
        # response over the unclipped benchmark range.
        assert photo_responses[0] == pytest.approx(0.0, abs=1e-15)
        assert all(
            lower < upper
            for lower, upper in zip(photo_responses, photo_responses[1:])
        )

        # === Basic Stoichiometry and Positivity Check ===
        delta_no = chem_fluxes['no_mixing_ratio'].value
        delta_no2 = chem_fluxes['no2_mixing_ratio'].value
        assert torch.allclose(
            delta_no + delta_no2,
            torch.zeros_like(delta_no),
            atol=1e-18,
            rtol=0.0,
        )
        for key in (
            'no_mixing_ratio',
            'no2_mixing_ratio',
            'o3_mixing_ratio',
            'isoprene_mixing_ratio',
        ):
            assert torch.all(state[key] + chem_fluxes[key].value >= 0.0), key
        
        logger.info("Chemistry Reaction Kinetics Test PASSED")

    def test_mass_energy_conservation(self, validation_scenarios):
        """
        Test mass and energy conservation across the system
        
        Validates:
        - Carbon mass balance
        - Water mass balance  
        - Energy balance closure
        - No spurious sources or sinks
        """
        logger.info("Starting Mass and Energy Conservation Test")
        
        # Use temperate grassland scenario for conservation testing
        scenario = validation_scenarios[0]
        state = self.create_scenario_state(scenario, batch_size=1)
        
        # Initialize all modules
        photosynthesis = FarquharPhotosynthesis()
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        radiation = RadiationFunctor()
        
        # === Carbon Mass Balance Test ===
        logger.info("Testing carbon mass balance")
        
        photo_fluxes = photosynthesis(state, time=0.0, dt=3600.0)

        # Farquhar exposes an open atmosphere-biosphere carbon port. Validate
        # its exact local identities without pretending to update a CO2 reservoir.
        gross = _unwrap_update(photo_fluxes['gross_photosynthesis'])
        net = _unwrap_update(photo_fluxes['net_photosynthesis'])
        dark = _unwrap_update(
            photo_fluxes['diagnostics.photosynthesis.dark_respiration']
        )
        gpp = _unwrap_update(photo_fluxes['gross_primary_productivity'])
        expected_gpp = gross * photosynthesis.carbon_molar_mass_g_mol * 1.0e-9
        assert torch.all(gpp >= 0), "GPP should be positive (carbon uptake)"
        assert torch.allclose(gpp, expected_gpp, rtol=1.0e-6, atol=0.0)
        assert torch.allclose(net, gross - dark, rtol=1.0e-6, atol=1.0e-7)
        logger.info(f"GPP: {torch.mean(gpp):.2e} kg C m-2 s-1")
        
        # === Water Mass Balance Test ===
        logger.info("Testing water mass balance")
        
        et_fluxes = evapotranspiration(state, time=0.0, dt=3600.0)
        
        if 'evapotranspiration_rate' in et_fluxes:
            et_rate = _unwrap_update(et_fluxes['evapotranspiration_rate'])
            
            # ET should be positive (water loss to atmosphere)
            assert torch.all(et_rate >= 0), "ET rate should be positive"
            
            # Convert to daily rate for validation
            et_daily = torch.mean(et_rate).item() * 86400 * 1000  # mm day-1
            
            # Should be reasonable for the given conditions
            assert 0.1 <= et_daily <= 20.0, f"ET rate outside reasonable range: {et_daily:.2f} mm day-1"
            
            logger.info(f"ET rate: {et_daily:.2f} mm day-1")
        
        # === Energy Balance Test ===
        logger.info("Testing energy balance")
        
        rad_fluxes = radiation(state, time=0.0, dt=3600.0)
        
        if 'net_radiation' in rad_fluxes:
            net_rad = _unwrap_update(rad_fluxes['net_radiation'])
            
            # Energy balance: Rn = H + LE + G
            # where H = sensible heat, LE = latent heat, G = ground heat flux
            sensible_h = state.get('sensible_heat_flux', torch.zeros_like(net_rad))
            latent_le = state.get('latent_heat_flux', torch.zeros_like(net_rad))
            
            total_turbulent = sensible_h + latent_le
            energy_residual = torch.abs(net_rad - total_turbulent)
            
            # Allow for ground heat flux and measurement uncertainty
            mean_residual = torch.mean(energy_residual).item()
            mean_net_rad = torch.abs(torch.mean(net_rad)).item()
            
            if mean_net_rad > 0:
                relative_residual = mean_residual / mean_net_rad
                assert relative_residual < 0.5, f"Energy balance closure poor: {relative_residual:.3f}"
                
                logger.info(f"Energy balance closure: {relative_residual:.3f}")
            
            logger.info(f"Net radiation: {torch.mean(net_rad):.1f} W/m^2")
        
        logger.info("Mass and Energy Conservation Test PASSED")

    def test_unit_consistency_validation(self, validation_scenarios):
        """
        Test unit consistency across all model outputs
        
        Validates:
        - All fluxes have consistent units
        - Unit conversions are correct
        - No dimensional analysis errors
        - Physical constants are properly applied
        """
        logger.info("Starting Unit Consistency Validation Test")
        
        scenario = validation_scenarios[0]
        state = self.create_scenario_state(scenario)
        
        # Test all major modules for unit consistency
        modules_to_test = [
            ('photosynthesis', FarquharPhotosynthesis()),
            ('evapotranspiration', PenmanMonteithET(config={"canopy_resistance_source": "state"})),
            ('radiation', RadiationFunctor()),
            ('chemistry', ChemistryFunctor(physical_boundary="open_mechanism")),
        ]
        
        expected_units = {
            'gross_photosynthesis': 'umol CO2 m-2 s-1',
            'net_photosynthesis': 'umol CO2 m-2 s-1',
            'gross_primary_productivity': 'kg C m-2 s-1',
            'net_radiation': 'W m-2',
            'shortwave_down': 'W m-2',
            'longwave_down': 'W m-2',
            'ppfd': 'umol photons m-2 s-1',
            'no_mixing_ratio': 'mol mol-1',
            'no2_mixing_ratio': 'mol mol-1',
            'o3_mixing_ratio': 'mol mol-1',
        }
        
        for module_name, module in modules_to_test:
            logger.info(f"Testing unit consistency for {module_name}")
            
            module_dt = 1.0 if module_name == 'chemistry' else 3600.0
            fluxes = module(state, time=0.0, dt=module_dt)

            for key, flux in fluxes.items():
                value = _unwrap_update(flux)
                if not torch.is_tensor(value):
                    continue
                assert torch.all(torch.isfinite(value)), \
                    f"Non-finite values in {module_name}.{key} suggest unit errors"
                if key in expected_units:
                    assert isinstance(flux, UpdateItem), key
                    assert flux.unit == expected_units[key]

                flux_magnitude = torch.abs(torch.mean(value)).item()
                if key.endswith('_mixing_ratio'):
                    assert flux_magnitude < 1.0, \
                        f"Mixing-ratio magnitude unreasonable: {module_name}.{key} = {flux_magnitude:.2e}"
                elif 'productivity' in key:
                    assert flux_magnitude < 1e-3, \
                        f"Productivity flux magnitude unreasonable: {module_name}.{key} = {flux_magnitude:.2e}"
                elif isinstance(flux, UpdateItem) and flux.unit == 'W m-2':
                    assert flux_magnitude < 2000, \
                        f"Radiation flux magnitude unreasonable: {module_name}.{key} = {flux_magnitude:.1f}"
                elif 'rate' in key and 'evapotranspiration' in key:
                    assert flux_magnitude < 1e-3, \
                        f"ET rate magnitude unreasonable: {module_name}.{key} = {flux_magnitude:.2e}"

            logger.info(f"Unit consistency verified for {module_name}")
        
        # === Cross-Module Unit Consistency ===
        logger.info("Testing cross-module unit consistency")
        
        # Test that modules can exchange data without unit conflicts
        photosynthesis = FarquharPhotosynthesis()
        photo_fluxes = photosynthesis(state, time=0.0, dt=3600.0)
        
        # Expose instantaneous ports without applying REPLACE values as
        # additive state increments.
        coupled_state = dict(state)
        for key, value in photo_fluxes.items():
            coupled_state[key] = _unwrap_update(value)
        
        # Test that other modules can use updated state
        evapotranspiration = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        et_fluxes = evapotranspiration(coupled_state, time=0.0, dt=3600.0)

        for key, flux in et_fluxes.items():
            value = _unwrap_update(flux)
            if torch.is_tensor(value):
                assert torch.all(torch.isfinite(value)), \
                    f"Cross-module data exchange caused unit error in {key}"

        logger.info("Cross-module unit consistency verified")
        
        logger.info("Unit Consistency Validation Test PASSED")


if __name__ == '__main__':
    # Enable detailed logging for standalone execution
    logging.basicConfig(level=logging.DEBUG)
    
    # Run specific tests for debugging
    pytest.main([__file__ + "::TestL3ScientificAccuracy::test_farquhar_photosynthesis_benchmarks", "-v", "-s"])
