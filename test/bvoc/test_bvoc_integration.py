"""
L2 Integration Tests for BVOC Module

This test suite provides integration-level testing for the BVOC emission module,
focusing on interactions between components, data flow, coupling with other modules,
and system-level behaviors. Tests use realistic data and validate conservation
principles.

Test Coverage
-------------
- MEGAN model integration with real data flow
- Photosynthesis coupling and feedback mechanisms
- Temperature and light response integration
- Stress response propagation
- Parameter system integration and consistency
- Conservation principles (mass, energy, carbon)
- Multi-component interactions and dependencies

Scientific Validation
---------------------
Test scenarios based on field measurements and benchmark datasets:
- FluxNet site data for realistic environmental conditions
- MEGAN global emission inventories for validation targets
- Controlled experiment data for response function validation
- Multi-site comparisons for model robustness

References
----------
- Guenther, A., et al. (2012). MEGAN2.1: Extended framework for modeling 
  biogenic emissions. Geoscientific Model Development 5, 1471-1492
- Baldocchi, D., et al. (2001). FLUXNET: A new tool to study the temporal 
  and spatial variability of ecosystem-scale carbon dioxide, water vapor, 
  and energy flux densities. Bulletin of the American Meteorological Society 82, 2415-2434
- Monson, R.K., et al. (2007). Isoprene emission from terrestrial ecosystems
  in response to global change: minding the gap between models and observations.
  Philosophical Transactions of the Royal Society A 365, 1677-1695
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from test.bvoc_parameter_manager_helper import build_realistic_parameter_manager

# Import BVOC modules for integration testing
from relic.scientific.biosphere.bvoc.megan_model import (
    MEGANModel, CouplingStrategy, EmissionPathway
)
from relic.scientific.biosphere.bvoc.emission_factors import (
    EmissionFactorDatabase,
)
from relic.scientific.biosphere.bvoc.responses import (
    BVOCResponseIntegrator,
)
from relic.scientific.biosphere.bvoc.synthesis.isoprene import IsopreneSynthesis
from relic.utils.physical_constants import get_physical_constant
from relic.data.standards.unit_converter import unit_converter


def _output_value(result, key):
    """Return the tensor carried by a canonical MEGAN UpdateItem."""
    item = result[key]
    assert hasattr(item, "value")
    return item.value

# The real ParameterManager entry point; the former try/except fell back to a
# Mock manager because relic.core.state exports no StateVariable.
from relic.parameters import get_global_manager


class TestMEGANModelIntegration:
    """Test MEGAN model integration with realistic data flows."""
    
    def setup_method(self):
        """Setup realistic test environment with parameter manager."""
        # Create comprehensive mock parameter manager
        self.mock_param_manager = self._create_realistic_parameter_manager()
        
        with patch('relic.parameters.manager.get_global_manager', 
                   return_value=self.mock_param_manager):
            
            # Create MEGAN model with different coupling strategies
            self.megan_standard = MEGANModel(
                name="megan_standard",
                coupling_strategy=CouplingStrategy.INDEPENDENT,
                emission_pathway=EmissionPathway.STANDARD
            )
            
            self.megan_coupled = MEGANModel(
                name="megan_coupled", 
                coupling_strategy=CouplingStrategy.SEQUENTIAL,
                emission_pathway=EmissionPathway.HYBRID
            )
            
            # Create emission factor database
            self.emission_db = EmissionFactorDatabase()
    
    def _create_realistic_parameter_manager(self):
        """retired helper retained for backward consistency."""
        return build_realistic_parameter_manager()
    
    def test_full_emission_pipeline_integration(self):
        """Test complete BVOC emission pipeline with realistic inputs.
        
        This test simulates a complete diurnal cycle with varying environmental
        conditions to test the integration of all BVOC components.
        
        References:
        - Guenther et al. (2012): Complete MEGAN algorithm integration
        - FluxNet data: Realistic environmental forcing patterns
        """
        # Simulate diurnal cycle (24 hours, hourly data)
        hours = np.linspace(0, 23, 24)
        
        # Realistic diurnal patterns based on FluxNet observations
        temperature_cycle = 288.15 + 15.0 * np.sin(np.pi * (hours - 6) / 12)  # 15degC amplitude
        par_cycle = np.maximum(0, 1800 * np.sin(np.pi * hours / 12))  # Solar radiation pattern
        
        # Model state for temperate deciduous forest
        base_state = {
            'leaf_area_index': torch.tensor(1.5),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        daily_emissions = []
        daily_factors = []
        
        for hour in range(24):
            # Update environmental conditions
            state = base_state.copy()
            state['air_temperature'] = torch.tensor(temperature_cycle[hour])
            state['ppfd'] = torch.tensor(par_cycle[hour])
            
            # Run MEGAN model
            result = self.megan_standard.forward(state, time=hour*3600, dt=3600)
            
            daily_emissions.append(_output_value(result, 'total_bvoc_emission').item())
            daily_factors.append({
                'temp_factor': _output_value(
                    result, 'diagnostics.bvoc.megan.isoprene.gamma_temperature'
                ).item(),
                'light_factor': _output_value(
                    result, 'diagnostics.bvoc.megan.isoprene.gamma_light'
                ).item(),
            })
        
        # Validate diurnal patterns
        daily_emissions = np.array(daily_emissions)
        
        # Should have clear diurnal pattern with peak during day
        daytime_avg = np.mean(daily_emissions[8:18])  # 8 AM to 6 PM
        nighttime_avg = np.mean(daily_emissions[np.concatenate([np.arange(0, 6), np.arange(20, 24)])])
        
        assert daytime_avg > nighttime_avg * 2  # Daytime should be much higher
        assert np.max(daily_emissions) > 0.0  # Should have positive emissions
        
        # Peak emissions should occur during warm, bright hours
        peak_hour = np.argmax(daily_emissions)
        assert 10 <= peak_hour <= 16  # Peak between 10 AM and 4 PM
        
        # The public total is a molar flux; its daily integral must be finite.
        daily_total = np.sum(daily_emissions) * 3600
        assert np.isfinite(daily_total)
        assert daily_total > 0.0
    
    def test_photosynthesis_coupling_integration(self):
        """Test integration between BVOC emissions and photosynthesis.
        
        Mathematical Expression: E_coupled = f_photo * A_n * gamma_env * MW / (n_C * 12.01)
        References: 
        - Niinemets et al. (2010): Photosynthetic carbon allocation to BVOC
        - Morfopoulos et al. (2014): Process-based isoprene model
        Parameters: f_photo [dimensionless], A_n [umol CO2 m-2 s-1], MW [g mol-1], n_C [dimensionless]
        """
        # Test scenarios with different photosynthetic rates
        photosynthesis_rates = [5.0, 15.0, 25.0, 35.0]  # umol CO2 m-2 s-1
        
        base_state = {
            'air_temperature': torch.tensor(303.15),
            'ppfd': torch.tensor(1200.0),
            'leaf_area_index': torch.tensor(1.25),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        standard_emissions = []
        coupled_emissions = []
        
        for ps_rate in photosynthesis_rates:
            # Test standard model (no coupling)
            state_standard = base_state.copy()
            result_standard = self.megan_standard.forward(state_standard, time=0, dt=3600)
            standard_emissions.append(
                _output_value(result_standard, 'total_bvoc_emission').item()
            )
            
            # Test coupled model
            state_coupled = base_state.copy()
            state_coupled.update({
                'net_photosynthesis': torch.tensor(ps_rate)
            })
            
            result_coupled = self.megan_coupled.forward(state_coupled, time=0, dt=3600)
            coupled_emissions.append(
                _output_value(result_coupled, 'total_bvoc_emission').item()
            )
        
        # Validate coupling behavior
        standard_emissions = np.array(standard_emissions)
        coupled_emissions = np.array(coupled_emissions)
        
        # Coupled emissions should vary with photosynthesis
        ps_array = np.asarray(photosynthesis_rates, dtype=np.float64)
        emission_array = np.asarray(coupled_emissions, dtype=np.float64)
        ps_anom = ps_array - ps_array.mean()
        emission_anom = emission_array - emission_array.mean()
        denom = np.linalg.norm(ps_anom) * np.linalg.norm(emission_anom)
        ps_correlation = float(np.dot(ps_anom, emission_anom) / denom) if denom > 0 else 0.0
        assert ps_correlation > 0.5  # Positive correlation with photosynthesis
        
        # Standard emissions should be relatively constant
        standard_cv = np.std(standard_emissions) / np.mean(standard_emissions)
        assert standard_cv < 0.1  # Low coefficient of variation
        
        # A removed coupling-factor alias must not reappear as a public output.
        assert 'coupling_factor' not in result_coupled
    
    def test_environmental_stress_propagation(self):
        """Test propagation of environmental stress through BVOC system.
        
        References:
        - Penuelas & Staudt (2010): Biogenic emissions under stress
        - Niinemets (2010): Stress effects on isoprene emission
        """
        # Test different stress scenarios
        stress_scenarios = [
            {'water_stress': 1.0, 'temperature_stress': 1.0, 'name': 'no_stress'},
            {'water_stress': 0.7, 'temperature_stress': 1.0, 'name': 'water_stress'},
            {'water_stress': 1.0, 'temperature_stress': 0.8, 'name': 'heat_stress'},
            {'water_stress': 0.6, 'temperature_stress': 0.7, 'name': 'combined_stress'}
        ]
        
        with patch(
            'relic.parameters.manager.get_global_manager',
            return_value=self.mock_param_manager,
        ):
            stressed_model = MEGANModel(
                name="megan_stress_enabled",
                emission_pathway=EmissionPathway.STANDARD,
                config={"use_stress_functors": True},
            )

        base_state = {
            'air_temperature': torch.tensor(308.15),
            'ppfd': torch.tensor(1500.0),
            'leaf_area_index': torch.tensor(1.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ozone_stress_factor': torch.tensor(1.0),
        }
        
        stress_results = {}
        
        for scenario in stress_scenarios:
            state = base_state.copy()
            state['water_stress_factor'] = torch.tensor(scenario['water_stress'])
            state['temperature_stress_factor'] = torch.tensor(
                scenario['temperature_stress']
            )

            result = stressed_model.forward(state, time=0, dt=3600)

            stress_results[scenario['name']] = {
                'total_emission': _output_value(
                    result, 'total_bvoc_emission'
                ).item(),
                'isoprene_emission': _output_value(
                    result, 'isoprene_emission'
                ).item(),
                'stress_factor': _output_value(
                    result,
                    'diagnostics.bvoc.megan.environmental_stress_factor',
                ).item(),
            }
        
        # Validate stress effects
        no_stress = stress_results['no_stress']
        
        # Water stress should reduce emissions
        water_stress = stress_results['water_stress']
        assert water_stress['total_emission'] < no_stress['total_emission']
        assert water_stress['stress_factor'] < no_stress['stress_factor']
        
        # Heat stress should reduce emissions
        heat_stress = stress_results['heat_stress']
        assert heat_stress['total_emission'] < no_stress['total_emission']
        
        # Combined stress should have largest reduction
        combined_stress = stress_results['combined_stress']
        assert combined_stress['total_emission'] < water_stress['total_emission']
        assert combined_stress['total_emission'] < heat_stress['total_emission']
        
        # Stress factors should multiply correctly
        expected_combined = 0.6 * 0.7  # water * temperature stress
        assert abs(combined_stress['stress_factor'] - expected_combined) < 0.01
    
    def test_parameter_system_integration(self):
        """Test integration with unified parameter management system.
        
        References: RELIC parameter management architecture
        """
        # Test parameter registration and retrieval
        model_params = {}
        
        # Collect all parameters from model
        for name, param in self.megan_standard.named_parameters():
            if hasattr(param, 'data'):
                model_params[name] = param.data.item()
        
        for name, buffer in self.megan_standard.named_buffers():
            if hasattr(buffer, 'data'):
                model_params[name] = buffer.data.item()
        
        # Verify key parameters are present and reasonable
        assert 'isoprene_ef' in model_params or any('isoprene' in k for k in model_params.keys())
        assert 'alpha_light' in model_params or any('alpha' in k for k in model_params.keys())
        
        # Test parameter consistency across multiple model instances
        with patch('relic.parameters.manager.get_global_manager', 
                   return_value=self.mock_param_manager):
            
            model2 = MEGANModel(name="megan_test2")
            
            # Key parameters should be identical
            assert torch.allclose(self.megan_standard.isoprene_ef, model2.isoprene_ef)
            assert torch.allclose(self.megan_standard.alpha_light, model2.alpha_light)
    
    def test_pft_parameter_ssot_integration(self):
        """Test compile-time MEGAN binding to the canonical PFT parameter SSOT."""
        # Test different PFT scenarios
        pft_scenarios = [
            ('broadleaf_deciduous_temperate', 'deciduous_forest'),
            ('needleleaf_evergreen_boreal', 'boreal_conifer_forest'),
            ('c3_grass_temperate', 'grassland')
        ]
        
        base_state = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(800.0),
            'leaf_area_index': torch.tensor(0.75),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        pft_emissions = {}

        for pft, name in pft_scenarios:
            pft_model = MEGANModel(
                name=f"megan_{name}",
                emission_pathway=EmissionPathway.STANDARD,
                config={"default_pft": pft},
            )
            result = pft_model.forward(base_state, time=0, dt=3600)

            pft_emissions[name] = {
                'total': _output_value(result, 'total_bvoc_emission').item(),
                'isoprene': _output_value(result, 'isoprene_emission').item(),
                'monoterpenes': _output_value(
                    result, 'monoterpenes_emission'
                ).item(),
            }
        
        # Validate PFT-specific patterns
        # Deciduous forests should have high isoprene
        assert pft_emissions['deciduous_forest']['isoprene'] > pft_emissions['boreal_conifer_forest']['isoprene']
        
        # Conifer forests should have high monoterpenes
        assert pft_emissions['boreal_conifer_forest']['monoterpenes'] > pft_emissions['deciduous_forest']['monoterpenes']
        
        # Grasslands should have moderate emissions
        grass_total = pft_emissions['grassland']['total']
        deciduous_total = pft_emissions['deciduous_forest']['total']
        assert 0.0 < grass_total < deciduous_total


class TestConservationPrinciples:
    """Test conservation principles in BVOC emission calculations."""
    
    def setup_method(self):
        """Setup for conservation testing."""
        self.mock_param_manager = build_realistic_parameter_manager()
        
        with patch('relic.parameters.manager.get_global_manager', 
                   return_value=self.mock_param_manager):
            self.model = MEGANModel(
                coupling_strategy=CouplingStrategy.SEQUENTIAL,
                emission_pathway=EmissionPathway.HYBRID
            )
    
    def test_mass_conservation_in_carbon_allocation(self):
        """Test mass conservation in photosynthetic carbon allocation to BVOC.
        
        Mathematical Expression: Sum(BVOC_C_flux) <= f_allocation * GPP_C_flux
        References: Carbon balance in photosynthetic allocation
        Parameters: f_allocation [dimensionless], GPP_C_flux [umol C m-2 s-1]
        """
        # Test scenario with known photosynthetic rates
        state = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'gross_photosynthesis': torch.tensor(30.0),
            'net_photosynthesis': torch.tensor(25.0),
        }
        
        result = self.model.forward(state, time=0, dt=3600)
        
        carbon_allocation_flux = _output_value(
            result, 'bvoc_carbon_allocation_flux'
        ).item()
        
        # Total BVOC carbon should be small fraction of GPP
        carbon_molar_mass = float(
            get_physical_constant(
                'relic.physical.constants.molecular_weights.carbon'
            )
        )
        micromoles_per_mole = float(
            get_physical_constant(
                'relic.physical.constants.conversions.micromoles_per_mole'
            )
        )
        gpp_carbon_g = (
            state['gross_photosynthesis'].item()
            * carbon_molar_mass
            / micromoles_per_mole
        )
        gpp_carbon_kg = unit_converter.convert(
            torch.tensor(gpp_carbon_g),
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        ).item()
        allocation_fraction = carbon_allocation_flux / gpp_carbon_kg
        
        # Should be less than 5% of GPP (typical allocation to secondary metabolites)
        assert allocation_fraction < 0.05
        assert allocation_fraction > 0.0
    
    def test_energy_balance_in_synthesis_pathways(self):
        """Test energy balance in BVOC synthesis pathways.
        
        Mathematical Expression: ATP_demand + NADPH_demand <= Available_energy
        References: Energetics of secondary metabolite synthesis
        Parameters: ATP_demand [umol m-2 s-1], NADPH_demand [umol m-2 s-1]
        """
        # Test with mechanistic isoprene synthesis model
        with patch('relic.parameters.manager.get_global_manager', 
                   return_value=self.mock_param_manager):
            
            synthesis_model = IsopreneSynthesis()

            state = {
                'air_temperature': torch.tensor(303.15),
                'gross_photosynthesis': torch.tensor(20.0),
                'electron_transport_rate': torch.tensor(0.01),
                'atmospheric_co2_ppm': torch.tensor(280.0),
                'leaf_area_index': torch.tensor(1.0),
            }

            result = synthesis_model.forward(state, time=0, dt=3600)

            atp_demand = _output_value(
                result,
                "diagnostics.bvoc.isoprene.atp_demand",
            )
            nadph_demand = _output_value(
                result,
                "diagnostics.bvoc.isoprene.nadph_demand",
            )
            dmapp_flux = _output_value(
                result,
                "diagnostics.bvoc.isoprene.mep_pathway_flux",
            )
            torch.testing.assert_close(
                atp_demand,
                (
                    dmapp_flux
                    * synthesis_model.mep_pathway.atp_per_dmapp
                ).to(dtype=atp_demand.dtype, device=atp_demand.device),
            )
            torch.testing.assert_close(
                nadph_demand,
                (
                    dmapp_flux
                    * synthesis_model.mep_pathway.nadph_per_dmapp
                ).to(dtype=nadph_demand.dtype, device=nadph_demand.device),
            )

            available_atp = (
                state["electron_transport_rate"]
                * synthesis_model.atp_per_electron_transport
            ).to(dtype=atp_demand.dtype, device=atp_demand.device)
            available_nadph = (
                state["electron_transport_rate"]
                * synthesis_model.nadph_per_electron_transport
            ).to(dtype=nadph_demand.dtype, device=nadph_demand.device)
            assert torch.all(atp_demand <= available_atp)
            assert torch.all(nadph_demand <= available_nadph)
    
    def test_stoichiometric_consistency(self):
        """Test stoichiometric consistency in emission calculations.
        
        References: Chemical stoichiometry in biosynthetic pathways
        """
        # Test multiple emission scenarios
        scenarios = [
            {'temp': 288.15, 'par': 500.0, 'name': 'cool_low_light'},
            {'temp': 303.15, 'par': 1000.0, 'name': 'standard'},
            {'temp': 318.15, 'par': 1800.0, 'name': 'hot_high_light'}
        ]
        
        base_state = {
            'leaf_area_index': torch.tensor(1.25),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'net_photosynthesis': torch.tensor(20.0),
        }
        
        for scenario in scenarios:
            state = base_state.copy()
            state['air_temperature'] = torch.tensor(scenario['temp'])
            state['ppfd'] = torch.tensor(scenario['par'])
            
            result = self.model.forward(state, time=0, dt=3600)
            
            # Sum of individual compounds should equal total
            individual_sum = 0.0
            for compound in ['isoprene', 'monoterpenes', 'sesquiterpenes', 'ovocs']:
                if f'{compound}_emission' in result:
                    individual_sum += _output_value(
                        result, f'{compound}_emission'
                    ).item()

            total_emission = _output_value(
                result, 'total_bvoc_emission'
            ).item()
            
            # Should match within numerical precision
            assert abs(individual_sum - total_emission) < 1e-6
            
            # All emissions should be non-negative
            assert total_emission >= 0.0
            for compound in ['isoprene', 'monoterpenes', 'sesquiterpenes', 'ovocs']:
                if f'{compound}_emission' in result:
                    assert _output_value(
                        result, f'{compound}_emission'
                    ).item() >= 0.0


class TestMultiComponentInteractions:
    """Test interactions between multiple BVOC components."""
    
    def setup_method(self):
        """Setup multi-component test environment."""
        self.mock_param_manager = build_realistic_parameter_manager()
        
        with patch('relic.parameters.manager.get_global_manager', 
                   return_value=self.mock_param_manager):
            
            # Create multiple model configurations
            self.models = {
                'standard': MEGANModel(
                    name="standard_megan",
                    coupling_strategy=CouplingStrategy.INDEPENDENT,
                    emission_pathway=EmissionPathway.STANDARD
                ),
                'coupled': MEGANModel(
                    name="coupled_megan",
                    coupling_strategy=CouplingStrategy.SEQUENTIAL,
                    emission_pathway=EmissionPathway.PHOTOSYNTHESIS
                ),
                'hybrid': MEGANModel(
                    name="hybrid_megan",
                    coupling_strategy=CouplingStrategy.SEQUENTIAL,
                    emission_pathway=EmissionPathway.HYBRID
                )
            }

    def test_response_function_integration(self):
        """Test integration of multiple response functions.
        
        Mathematical Expression: gamma_total = gamma_T * gamma_L * gamma_stress
        References: Multiplicative response function integration in MEGAN
        Parameters: gamma_T [dimensionless], gamma_L [dimensionless], gamma_stress [dimensionless]
        """
        # Test environmental conditions
        environment = {
            'temperature': torch.tensor(308.15),  # 35degC
            'par': torch.tensor(1200.0),
            'soil_moisture': torch.tensor(0.4),  # Dry conditions
            'vapor_pressure_deficit': torch.tensor(2.5),  # kPa
            'co2_concentration': torch.tensor(420.0)  # ppm
        }
        
        # Calculate individual response factors
        response_integrator = BVOCResponseIntegrator(
            parameter_manager=get_global_manager()
        )
        responses = response_integrator.calculate_integrated_response(
            environment, compound='isoprene'
        )
        
        # Verify individual responses are reasonable
        assert 'temperature_response' in responses
        assert 'light_response' in responses
        assert 'stress_response' in responses
        assert 'integrated_response' in responses
        
        # Check response factor ranges
        for factor_name, factor_value in responses.items():
            assert factor_value > 0.0
            assert factor_value < 10.0  # Should not be extremely large
            assert torch.isfinite(factor_value)
        
        # Integrated response should be product of individual responses
        expected_integrated = (responses['temperature_response'] * 
                             responses['light_response'] * 
                             responses['stress_response'])
        
        assert torch.allclose(responses['integrated_response'], expected_integrated, rtol=0.01)
    
    def test_model_intercomparison(self):
        """Test consistency between different model configurations.
        
        References: Model intercomparison protocols for BVOC emissions
        """
        # Standard test conditions
        test_state = {
            'air_temperature': torch.tensor(303.15),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        # Add photosynthesis data for coupled models
        coupled_state = test_state.copy()
        coupled_state.update({
            'net_photosynthesis': torch.tensor(18.0)
        })
        
        results = {}
        
        # Run standard model
        results['standard'] = self.models['standard'].forward(test_state, time=0, dt=3600)
        
        # Run coupled models
        results['coupled'] = self.models['coupled'].forward(coupled_state, time=0, dt=3600)
        results['hybrid'] = self.models['hybrid'].forward(coupled_state, time=0, dt=3600)
        
        # Compare results
        standard_total = _output_value(
            results['standard'], 'total_bvoc_emission'
        ).item()
        coupled_total = _output_value(
            results['coupled'], 'total_bvoc_emission'
        ).item()
        hybrid_total = _output_value(
            results['hybrid'], 'total_bvoc_emission'
        ).item()
        
        # All should produce positive emissions
        assert standard_total > 0.0
        assert coupled_total > 0.0
        assert hybrid_total > 0.0
        
        # Hybrid should be between standard and coupled
        # (This depends on the specific weighting, but generally true)
        emission_range = [standard_total, coupled_total]
        min_emission = min(emission_range)
        max_emission = max(emission_range)
        
        # Hybrid should be within reasonable range
        assert min_emission * 0.5 <= hybrid_total <= max_emission * 1.5
        
        # Models should have similar temperature and light responses
        for model_name in results:
            temp_factor = _output_value(
                results[model_name],
                'diagnostics.bvoc.megan.isoprene.gamma_temperature',
            ).item()
            assert 0.5 < temp_factor < 3.0
    
    def test_dependency_resolution(self):
        """Test proper dependency resolution between BVOC components.
        
        References: Dependency management in modular scientific software
        """
        # Test dependency chains
        for model_name, model in self.models.items():
            dependencies = model.get_dependencies()
            
            # PPFD is produced by the canonical atmospheric-radiation owner.
            assert 'atmospheric_radiation' in dependencies
            
            # Coupled models should depend on photosynthesis
            if model.emission_pathway in (
                EmissionPathway.PHOTOSYNTHESIS,
                EmissionPathway.HYBRID,
            ):
                assert 'farquhar_photosynthesis' in dependencies
            
            # Dependencies should be strings (functor names)
            for dep in dependencies:
                assert isinstance(dep, str)
                assert len(dep) > 0
        
        # Test provided outputs
        for model_name, model in self.models.items():
            provides = model.get_provides()
            
            # All models should provide total emission
            assert 'total_bvoc_emission' in provides
            
            # Should provide compound-specific outputs
            compound_outputs = [p for p in provides if 'emission' in p and 'total' not in p]
            assert len(compound_outputs) > 0
    
    def test_parameter_sensitivity_propagation(self):
        """Test parameter sensitivity propagation through model components.
        
        Mathematical Expression: dE_total/dtheta = sum(dE_i/dtheta)
        References: Sensitivity analysis in coupled models
        Parameters: theta [parameter], E_total [total emission], E_i [component emission]
        """
        # Test state for sensitivity analysis
        test_state = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(800.0),
            'leaf_area_index': torch.tensor(0.75),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        # Test sensitivity to key parameters
        parameters_to_test = ['alpha_light']
        
        for param_name in parameters_to_test:
            if hasattr(self.models['standard'], param_name):
                # Calculate sensitivity for standard model
                sensitivity = self.models['standard'].parameter_sensitivity(
                    param_name, test_state, time=0.0, dt=1.0,
                    target_output='total_bvoc_emission', relative_eps=0.01,
                )
                
                # Sensitivity should be finite and reasonable
                assert torch.isfinite(sensitivity)
                assert sensitivity != 0.0  # Should have some effect
                
                # Alpha_light sensitivity should be positive under lighted conditions
                assert sensitivity > 0.0


class TestRealisticScenarios:
    """Test BVOC models with realistic environmental scenarios."""
    
    def setup_method(self):
        """Setup realistic scenario testing."""
        self.mock_param_manager = build_realistic_parameter_manager()
        
        with patch('relic.parameters.manager.get_global_manager', 
                   return_value=self.mock_param_manager):
            self.model = MEGANModel(
                coupling_strategy=CouplingStrategy.INDEPENDENT,
                emission_pathway=EmissionPathway.STANDARD,
            )
    
    def test_seasonal_variation_scenario(self):
        """Test BVOC emissions across seasonal variations.
        
        References: Seasonal patterns in biogenic emissions (Guenther et al. 2006)
        """
        # Simulate seasonal conditions (monthly averages)
        seasonal_conditions = [
            {'month': 'Jan', 'temp': 278.15, 'par': 400, 'foliar': 50},   # Winter
            {'month': 'Apr', 'temp': 288.15, 'par': 800, 'foliar': 150},  # Spring
            {'month': 'Jul', 'temp': 303.15, 'par': 1200, 'foliar': 300}, # Summer
            {'month': 'Oct', 'temp': 293.15, 'par': 600, 'foliar': 200}   # Fall
        ]
        
        seasonal_emissions = {}
        
        for conditions in seasonal_conditions:
            state = {
                'air_temperature': torch.tensor(conditions['temp']),
                'ppfd': torch.tensor(conditions['par']),
                'leaf_area_index': torch.tensor(conditions['foliar'] / 200.0),
                'atmospheric_co2_ppm': torch.tensor(400.0),
            }
            
            result = self.model.forward(state, time=0, dt=3600)
            seasonal_emissions[conditions['month']] = _output_value(
                result, 'total_bvoc_emission'
            ).item()
        
        # Validate seasonal patterns
        # Summer should have highest emissions
        assert seasonal_emissions['Jul'] == max(seasonal_emissions.values())
        
        # Winter should have lowest emissions
        assert seasonal_emissions['Jan'] == min(seasonal_emissions.values())
        
        # Spring and fall should be intermediate
        assert seasonal_emissions['Jan'] < seasonal_emissions['Apr'] < seasonal_emissions['Jul']
        assert seasonal_emissions['Jan'] < seasonal_emissions['Oct'] < seasonal_emissions['Jul']
    
    def test_climate_change_scenario(self):
        """Test BVOC response to climate change conditions.
        
        References: Climate change impacts on biogenic emissions
        """
        # Current and future climate scenarios
        scenarios = {
            'current': {
                'temperature': 298.15,  # 25degC
                'co2': 410.0,          # ppm
                'par': 1000.0
            },
            'future_2050': {
                'temperature': 301.15,  # 28degC (+3degC warming)
                'co2': 500.0,          # ppm (+90 ppm)
                'par': 1000.0          # Assume similar radiation
            },
            'future_2100': {
                'temperature': 304.15,  # 31degC (+6degC warming)
                'co2': 600.0,          # ppm (+190 ppm)
                'par': 1000.0
            }
        }
        
        climate_results = {}
        
        for scenario_name, climate in scenarios.items():
            state = {
                'air_temperature': torch.tensor(climate['temperature']),
                'ppfd': torch.tensor(climate['par']),
                'leaf_area_index': torch.tensor(1.25),
                'atmospheric_co2_ppm': torch.tensor(climate['co2']),
            }
            
            result = self.model.forward(state, time=0, dt=3600)
            
            climate_results[scenario_name] = {
                'total_emission': _output_value(
                    result, 'total_bvoc_emission'
                ).item(),
                'isoprene_emission': _output_value(
                    result, 'isoprene_emission'
                ).item(),
                'co2_inhibition': _output_value(
                    result, 'diagnostics.bvoc.megan.co2_inhibition_factor'
                ).item(),
            }
        
        # Validate climate change responses
        current = climate_results['current']
        future_2050 = climate_results['future_2050']
        future_2100 = climate_results['future_2100']
        
        # Temperature increase should generally increase emissions
        # But CO2 increase should decrease isoprene (inhibition effect)
        
        # Total emissions may increase due to temperature
        assert future_2050['total_emission'] > current['total_emission'] * 0.8  # Allow for CO2 inhibition
        
        # But isoprene specifically should show CO2 inhibition
        assert future_2100['co2_inhibition'] < current['co2_inhibition']
    
    def test_drought_stress_scenario(self):
        """Test BVOC response to progressive drought stress.
        
        Mathematical Expression: E_stressed = E_potential * f_water * f_temperature
        References: Drought effects on biogenic emissions (Penuelas & Staudt 2010)
        Parameters: f_water [dimensionless], f_temperature [dimensionless]
        """
        # Progressive drought scenario
        drought_levels = [1.0, 0.8, 0.6, 0.4, 0.2]  # Water stress factors
        
        with patch(
            'relic.parameters.manager.get_global_manager',
            return_value=self.mock_param_manager,
        ):
            stressed_model = MEGANModel(
                name="megan_drought_stress",
                coupling_strategy=CouplingStrategy.INDEPENDENT,
                emission_pathway=EmissionPathway.STANDARD,
                config={"use_stress_functors": True},
            )

        base_conditions = {
            'air_temperature': torch.tensor(308.15),
            'ppfd': torch.tensor(1400.0),
            'leaf_area_index': torch.tensor(1.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'temperature_stress_factor': torch.tensor(0.9),
            'ozone_stress_factor': torch.tensor(1.0),
        }
        
        drought_emissions = []
        stress_factors = []
        
        for water_stress in drought_levels:
            state = base_conditions.copy()
            state['water_stress_factor'] = torch.tensor(water_stress)

            result = stressed_model.forward(state, time=0, dt=3600)

            drought_emissions.append(
                _output_value(result, 'total_bvoc_emission').item()
            )
            stress_factors.append(
                _output_value(
                    result,
                    'diagnostics.bvoc.megan.environmental_stress_factor',
                ).item()
            )
        
        # Validate drought response
        drought_emissions = np.array(drought_emissions)
        stress_factors = np.array(stress_factors)
        
        # Emissions should decrease with increasing drought
        assert np.all(np.diff(drought_emissions) <= 0)  # Monotonic decrease
        
        # Stress factors should match water stress levels
        expected_stress = np.array(drought_levels) * 0.9  # * temperature stress
        assert np.allclose(stress_factors, expected_stress, rtol=0.01)
        
        # Severe drought should reduce emissions by >50%
        severe_reduction = drought_emissions[-1] / drought_emissions[0]
        assert severe_reduction < 0.5


# Test fixtures for integration testing
@pytest.fixture
def realistic_forest_state():
    """Realistic forest ecosystem state for testing."""
    return {
        'air_temperature': torch.tensor(298.15),
        'ppfd': torch.tensor(800.0),
        'leaf_area_index': torch.tensor(1.4),
        'atmospheric_co2_ppm': torch.tensor(400.0),
        'net_photosynthesis': torch.tensor(20.0),
    }


@pytest.fixture
def comprehensive_parameter_database():
    """Comprehensive parameter database for testing."""
    return {
        # Basic emission factors
        "biosphere.bvoc.default.isoprene": 24.0,
        "biosphere.bvoc.default.monoterpenes": 2.3,
        "biosphere.bvoc.default.sesquiterpenes": 0.45,
        "biosphere.bvoc.default.ovocs": 1.8,
        
        # Response parameters
        "biosphere.bvoc.megan.alpha_light": 0.0027,
        "biosphere.bvoc.megan.eopt_isoprene": 1.75,
        "biosphere.bvoc.megan.topt": 313.0,
        "biosphere.bvoc.megan.beta_monoterpenes": 0.09,
        
        # PFT-specific factors
        "biosphere.bvoc.broadleaf_deciduous_temperate.isoprene": 70.0,
        "biosphere.bvoc.needleleaf_evergreen_temperate.monoterpenes": 5.2,
        
        # Physical constants
        "relic.physical.constants.universal.gas_constant": 8.314
    }


if __name__ == "__main__":
    # Run integration tests with detailed output
    pytest.main([__file__, "-v", "--tb=long", "-s"])


