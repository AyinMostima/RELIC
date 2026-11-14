"""
Integration tests for atmospheric chemistry module.

Tests the integration of ChemistryFunctor with other RELIC components:
- Integration with radiation module for photolysis
- Coupling with BVOC emission modules
- State management and conservation
- Parameter database integration
- Simulator orchestration
"""

import copy
import math

import pytest
import torch

from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.core.state import UpdateItem
from relic.core.simulator import RELICSimulator
from relic.core.dependency_graph import DependencyGraphResolver
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
    EddyDiffusionConfig,
    EddyDiffusionBLMFunctor,
)
from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionConfig,
    DryDepositionAggregator,
)


def _output_value(fluxes, key):
    """Return the value of one canonical chemistry output."""

    item = fluxes[key]
    assert isinstance(item, UpdateItem)
    return item.value


class TestRadiationCoupling:
    """Test integration with radiation module for photolysis."""
    
    def test_photolysis_from_radiation(self):
        """Test that chemistry receives photolysis rates from radiation."""
        # Create chemistry module
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Create state with radiation data
        # Use simple dict for state since StateManager API may have changed
        state = {}
        
        # Add required variables
        state['air_temperature'] = torch.tensor(298.15)
        state['atmospheric_pressure'] = torch.tensor(101325.0)
        state['no_mixing_ratio'] = torch.tensor(5e-9)
        state['no2_mixing_ratio'] = torch.tensor(10e-9)
        state['o3_mixing_ratio'] = torch.tensor(40e-9)
        state['co_mixing_ratio'] = torch.tensor(150e-9)  # OH sink, 150 ppb
        state['ch4_mixing_ratio'] = torch.tensor(1.8e-6)  # OH sink, 1.8 ppm
        state['specific_humidity'] = torch.tensor(0.01)  # O(1D) + H2O OH source, 10 g/kg
        
        # Radiation module would provide these
        state['shortwave_down'] = torch.tensor(800.0)
        state['cos_zenith'] = torch.tensor(0.866)  # 30 degrees
        
        # Run chemistry
        fluxes = chem.forward(state, time=0.0, dt=60.0)
        
        # Check photolysis is calculated
        key = 'diagnostics.chemistry.photolysis_rate_no2'
        assert key in fluxes
        assert _output_value(fluxes, key) > 0
        
    def test_diurnal_photochemistry(self):
        """Test diurnal cycle of photochemistry."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Base state
        state_base = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Daytime conditions
        state_day = state_base.copy()
        state_day['shortwave_down'] = torch.tensor(800.0)
        state_day['cos_zenith'] = torch.tensor(0.866)
        
        # Nighttime conditions
        state_night = state_base.copy()
        state_night['shortwave_down'] = torch.tensor(0.0)
        state_night['cos_zenith'] = torch.tensor(-0.5)  # Below horizon
        
        # Use one admissible explicit step for both regimes; the dark-state
        # positivity limit is 56.4 s for this concentration vector.
        fluxes_day = chem.forward(state_day, time=43200.0, dt=30.0)  # Noon
        fluxes_night = chem.forward(state_night, time=0.0, dt=30.0)  # Midnight
        
        # Day should have photolysis, night should not
        diagnostic_key = 'diagnostics.chemistry.photolysis_rate_no2'
        assert _output_value(fluxes_day, diagnostic_key) > 0
        assert _output_value(fluxes_night, diagnostic_key) == 0
        
        # O3 production during day, destruction at night
        assert _output_value(
            fluxes_day,
            'o3_mixing_ratio',
        ) > _output_value(fluxes_night, 'o3_mixing_ratio')


class TestBVOCCoupling:
    """Test integration with BVOC emission modules."""
    
    def test_isoprene_from_emissions(self):
        """Test chemistry response to BVOC emissions."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # State without isoprene
        state_no_bvoc = {
            'air_temperature': torch.tensor(303.15),  # Warm for emissions
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(2e-9),
            'no2_mixing_ratio': torch.tensor(5e-9),
            'o3_mixing_ratio': torch.tensor(50e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # State with isoprene from emissions
        state_with_bvoc = state_no_bvoc.copy()
        state_with_bvoc['isoprene_mixing_ratio'] = torch.tensor(3e-9)  # From BVOC module
        
        fluxes_no_bvoc = chem.forward(state_no_bvoc, time=0.0, dt=60.0)
        fluxes_with_bvoc = chem.forward(state_with_bvoc, time=0.0, dt=60.0)
        
        # The optional species has no hidden zero writer when absent.
        assert 'isoprene_mixing_ratio' not in fluxes_no_bvoc
        assert 'isoprene_mixing_ratio' in fluxes_with_bvoc
        assert _output_value(
            fluxes_with_bvoc,
            'isoprene_mixing_ratio',
        ) < 0
        
        # BVOC should affect O3 (consumption via ozonolysis)
        assert _output_value(
            fluxes_with_bvoc,
            'o3_mixing_ratio',
        ) < _output_value(fluxes_no_bvoc, 'o3_mixing_ratio')
        
    def test_temperature_driven_chemistry(self):
        """Test temperature effects on BVOC chemistry."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Base state with isoprene
        state_base = {
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(2e-9),
            'no2_mixing_ratio': torch.tensor(5e-9),
            'o3_mixing_ratio': torch.tensor(50e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(3e-9),
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Cold conditions
        state_cold = state_base.copy()
        state_cold['air_temperature'] = torch.tensor(283.15)  # 10 C
        
        # Warm conditions
        state_warm = state_base.copy()
        state_warm['air_temperature'] = torch.tensor(308.15)  # 35 C
        
        fluxes_cold = chem.forward(state_cold, time=0.0, dt=60.0)
        fluxes_warm = chem.forward(state_warm, time=0.0, dt=60.0)

        # The temperature enters the isoprene loss through the IUPAC rate coefficients:
        # OH addition slows with warming (II.A2.10, E/R = -390 K) and ozonolysis speeds
        # up (II.A1.157, E/R = 1995 K). The loss rate is (k_OH [OH] + k_O3 [O3]) chi.
        coefficients = {}
        for label, fluxes, temperature in (
            ('cold', fluxes_cold, 283.15),
            ('warm', fluxes_warm, 308.15),
        ):
            rates = chem._calculate_temperature_dependent_rates(
                torch.tensor(temperature),
                reactions=('k_isoprene_oh', 'k_isoprene_o3'),
            )
            oh = _output_value(fluxes, 'diagnostics.chemistry.oh_concentration')
            air = _output_value(fluxes, 'diagnostics.chemistry.air_number_density')
            expected = (
                rates['k_isoprene_oh'] * oh + rates['k_isoprene_o3'] * 50e-9 * air
            ) * 3e-9
            assert float(
                _output_value(fluxes, 'diagnostics.chemistry.isoprene_oxidation_rate')
            ) == pytest.approx(float(expected), rel=1e-5)
            coefficients[label] = rates
        assert coefficients['warm']['k_isoprene_oh'] < coefficients['cold']['k_isoprene_oh']
        assert coefficients['warm']['k_isoprene_o3'] > coefficients['cold']['k_isoprene_o3']


class TestStateManagement:
    """Test integration with StateManager."""

    def test_canonical_species_multiwriter_step(self):
        """Chemistry and transport aggregate once with complete provenance."""

        dt = 60.0
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'shortwave_down': torch.tensor(800.0),
            'no_mixing_ratio': torch.tensor(5.0e-9),
            'no2_mixing_ratio': torch.tensor(10.0e-9),
            'o3_mixing_ratio': torch.tensor(40.0e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(2.0e-9),
            'so2_mixing_ratio': torch.tensor(2.0e-9),
            'pbl_height': torch.tensor(1000.0),
            'friction_velocity': torch.tensor(0.3),
            'monin_obukhov_length': torch.tensor(200.0),
            'aerodynamic_resistance': torch.tensor(30.0),
        }
        simulator = RELICSimulator({
            'max_time': dt,
            'time_step': dt,
            'conservation_checks': False,
            'strict_scale_detection': False,
        })
        functors = (
            ChemistryFunctor(config={'dependencies': []}, physical_boundary="open_mechanism"),
            EddyDiffusionBLMFunctor(
                config={'dependencies': []},
                transport_config=EddyDiffusionConfig(enabled=True),
            ),
            DryDepositionAggregator(
                config={'dependencies': []},
                deposition_config=DryDepositionConfig(enabled=True),
            ),
        )
        expected_increments = {
            key: torch.zeros_like(state[key])
            for key in (
                'no_mixing_ratio',
                'no2_mixing_ratio',
                'o3_mixing_ratio',
                'isoprene_mixing_ratio',
            )
        }
        for functor in functors:
            output = functor.forward(state, time=0.0, dt=dt)
            for key in expected_increments:
                if key in output:
                    expected_increments[key] += _output_value(output, key)

        simulator.initialize_state(state)
        for functor in functors:
            simulator.register_functor(functor, execution_group='fast')

        simulator.step(dt=dt)

        expected_writers = {
            'no_mixing_ratio': {
                'atmospheric_chemistry',
                'eddy_diffusion_blm',
                'dry_deposition_aggregator',
            },
            'no2_mixing_ratio': {
                'atmospheric_chemistry',
                'eddy_diffusion_blm',
                'dry_deposition_aggregator',
            },
            'o3_mixing_ratio': {
                'atmospheric_chemistry',
                'eddy_diffusion_blm',
                'dry_deposition_aggregator',
            },
            'isoprene_mixing_ratio': {
                'atmospheric_chemistry',
                'eddy_diffusion_blm',
            },
        }
        for key, expected_increment in expected_increments.items():
            actual_increment = simulator.state_manager.get_state(key) - state[key]
            torch.testing.assert_close(
                actual_increment,
                expected_increment.to(actual_increment),
            )
            records = simulator.get_variable_provenance(key)
            sources = {
                record['functor']
                for record in records
                if record.get('role') == 'increment_source'
            }
            assert sources == expected_writers[key], (
                key,
                records,
                simulator._step_functor_increments.get(key),
            )
    
    def test_state_updates(self):
        """Test that chemistry properly updates state."""
        # Create simple state dict
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Run chemistry
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        fluxes = chem.forward(state, time=0.0, dt=60.0)
        
        # Canonical payloads are already integrated over dt and apply once.
        new_no = state['no_mixing_ratio'] + _output_value(
            fluxes,
            'no_mixing_ratio',
        )
        new_no2 = state['no2_mixing_ratio'] + _output_value(
            fluxes,
            'no2_mixing_ratio',
        )
        new_o3 = state['o3_mixing_ratio'] + _output_value(
            fluxes,
            'o3_mixing_ratio',
        )
        
        # Check updates are reasonable
        assert new_no != state['no_mixing_ratio']  # Should have changed
        assert new_no > 0  # Should remain positive
        assert new_no2 > 0
        assert new_o3 > 0
        
    def test_conservation_checks(self):
        """Test conservation validation."""
        # Create simple state
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(0.0),  # No photolysis
        }
        
        initial_nox = state['no_mixing_ratio'] + state['no2_mixing_ratio']
        
        # Run chemistry
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        fluxes = chem.forward(state, time=0.0, dt=30.0)
        
        # Check the closed gas-phase NO plus NO2 reaction family exactly.
        no_increment = _output_value(fluxes, 'no_mixing_ratio')
        no2_increment = _output_value(fluxes, 'no2_mixing_ratio')
        final_no = state['no_mixing_ratio'] + no_increment
        final_no2 = state['no2_mixing_ratio'] + no2_increment
        final_nox = final_no + final_no2

        nox_tendency_residual = no_increment + no2_increment
        torch.testing.assert_close(
            nox_tendency_residual,
            torch.zeros_like(nox_tendency_residual),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            final_nox,
            initial_nox,
            rtol=torch.finfo(initial_nox.dtype).eps,
            atol=0.0,
        )


class TestDependencyResolution:
    """Test dependency graph integration."""
    
    def test_chemistry_dependencies(self):
        """Test that chemistry dependencies are properly resolved."""
        resolver = DependencyGraphResolver()
        
        # Add chemistry functor
        chem = ChemistryFunctor(name="atmospheric_chemistry", physical_boundary="open_mechanism")
        resolver.add_functor(chem)
        
        # Chemistry depends on radiation as an optional provider
        from relic.core.dependency_graph import normalize_dependency_declaration

        deps = normalize_dependency_declaration(chem.get_dependencies(), owner=chem.name)
        assert 'atmospheric_radiation' in deps.optional
        
        # Check topological sort places radiation before chemistry
        # (would need radiation module to fully test)
        
    def test_parallel_execution_groups(self):
        """Test identification of parallel execution groups."""
        # This test requires full DependencyGraphResolver implementation
        # For now, just test that chemistry functors can be created independently
        chem1 = ChemistryFunctor(name="chemistry_layer1", physical_boundary="open_mechanism")
        chem2 = ChemistryFunctor(name="chemistry_layer2", physical_boundary="open_mechanism")
        
        # Check they have the same dependencies
        assert chem1.get_dependencies() == chem2.get_dependencies()
        
        # They should be parallelizable (same dependencies means can run in parallel)
        # Full test would require DependencyGraphResolver implementation


class TestParameterDatabase:
    """Test parameter database integration."""
    
    def test_parameter_loading(self):
        """Test loading parameters from database."""
        # Test with custom parameter values
        config = {
            'parameters': {
                'uv_a_surface_albedo': 0.1,  # Custom value
                'k_no_o3_298': 2.0e-14    # Custom value
            }
        }
        
        chem = ChemistryFunctor(config=config, physical_boundary="open_mechanism")
        
        # Check custom parameters are used
        assert chem.get_parameter_value('uv_a_surface_albedo') == 0.1
        assert chem.get_parameter_value('k_no_o3_298') == 2.0e-14
        
        # Check defaults are still loaded for unspecified parameters
        assert chem.get_parameter_value('k_isoprene_oh_298') is not None
        
    def test_parameter_bounds_validation(self):
        """Out-of-bounds overrides are rejected against the canonical YAML bounds."""
        # uv_a_surface_albedo bounds are (0, 0.2) in atmosphere/chemistry.yaml (Trebs et al. 2009).
        config = {
            'parameters': {
                'uv_a_surface_albedo': 0.5,  # Above the 0.2 validity limit (snow)
                'minimum_concentration': 0.5e-3  # Use valid positive value
            }
        }
        with pytest.raises(ValueError, match="bounds"):
            ChemistryFunctor(config=config, physical_boundary="open_mechanism")

        # A valid override inside the canonical bounds (1e-15, 1e-6) ppm is loaded and stays positive.
        chem = ChemistryFunctor(config={'parameters': {'minimum_concentration': 1e-7}}, physical_boundary="open_mechanism")
        min_conc = chem.get_parameter_value('minimum_concentration')
        assert min_conc > 0  # Should be positive
        assert float(chem.get_parameter_value('uv_a_surface_albedo')) == pytest.approx(0.03)


class TestValidationFramework:
    """Test integration with validation framework."""
    
    def test_validation_output(self):
        """Test that chemistry produces validation-consistent outputs."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        fluxes = chem.forward(state, time=0.0, dt=60.0)
        
        expected = {
            'diagnostics.chemistry.reaction_no_o3_rate',
            'diagnostics.chemistry.photolysis_no2_rate',
            'diagnostics.chemistry.oh_concentration',
        }
        assert expected <= set(fluxes)
        assert not any('deposition_loss' in key for key in fluxes)
        assert 'diagnostics.chemistry.pbl_height_used' not in fluxes
        
    def test_photostationary_validation(self):
        """Test validation of photostationary state relationship."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # State that should be near photostationary equilibrium
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(1e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        fluxes = chem.forward(state, time=0.0, dt=60.0)
        
        # Check photolysis is active
        j_no2 = _output_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
        )
        assert j_no2 > 0
        
        assert 'diagnostics.chemistry.reaction_no_o3_rate' in fluxes
        assert 'diagnostics.chemistry.photolysis_no2_rate' in fluxes


class TestMultiScaleIntegration:
    """Test multi-scale coupling capabilities."""
    
    def test_time_scale_separation(self):
        """Test handling of different chemical time scales."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Fast chemistry (NOx-O3 cycle)
        state_fast = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(50e-9),  # High NOx
            'no2_mixing_ratio': torch.tensor(50e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Slow chemistry (low NOx)
        state_slow = state_fast.copy()
        state_slow['no_mixing_ratio'] = torch.tensor(0.1e-9)  # Very low NOx
        state_slow['no2_mixing_ratio'] = torch.tensor(0.1e-9)
        
        # Run with different time steps
        dt_fast = 1.0    # 1 second for fast chemistry
        dt_slow = 60.0   # 1 minute for slow chemistry
        
        fluxes_fast = chem.forward(state_fast, time=0.0, dt=dt_fast)
        fluxes_slow = chem.forward(state_slow, time=0.0, dt=dt_slow)
        
        # Compare rates rather than increments generated with different dt.
        reaction_key = 'diagnostics.chemistry.reaction_no_o3_rate'
        assert abs(_output_value(fluxes_fast, reaction_key)) > abs(
            _output_value(fluxes_slow, reaction_key)
        )
        
    def test_spatial_coupling_readiness(self):
        """Test readiness for spatial coupling (multi-box model)."""
        # Create multiple chemistry boxes for different spatial locations
        chem_urban = ChemistryFunctor(name="chemistry_urban", physical_boundary="open_mechanism")
        chem_rural = ChemistryFunctor(name="chemistry_rural", physical_boundary="open_mechanism")
        
        # Urban conditions (high NOx)
        state_urban = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(20e-9),
            'no2_mixing_ratio': torch.tensor(30e-9),
            'o3_mixing_ratio': torch.tensor(30e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Rural conditions (low NOx, high O3)
        state_rural = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(1e-9),
            'no2_mixing_ratio': torch.tensor(2e-9),
            'o3_mixing_ratio': torch.tensor(60e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Run both boxes
        fluxes_urban = chem_urban.forward(state_urban, time=0.0, dt=60.0)
        fluxes_rural = chem_rural.forward(state_rural, time=0.0, dt=60.0)
        
        # Different chemistry regimes
        # Urban: NOx-saturated, O3 destruction
        # Rural: NOx-limited, O3 production
        assert _output_value(
            fluxes_urban,
            'o3_mixing_ratio',
        ) != _output_value(fluxes_rural, 'o3_mixing_ratio')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




class TestClosedConstantVolumeBoxIntegration:
    """GraphIR, StateManager, budget, restart, and rollback of the closed NO + O3 box."""

    NODE_TYPE = (
        "Functor:relic.scientific.atmosphere.chemistry.ChemistryFunctor"
        "#closed_constant_volume"
    )
    SPECIES = ("box_no_amount", "box_no2_amount", "box_o3_amount", "box_o2_amount")
    STATE = SPECIES + ("box_internal_energy",)
    ATOMS = {"nitrogen": (1.0, 1.0, 0.0, 0.0), "oxygen": (1.0, 2.0, 3.0, 2.0)}

    @staticmethod
    def _box():
        return ChemistryFunctor(
            name="atmospheric_chemistry",
            physical_boundary="closed_constant_volume",
        )

    @classmethod
    def _initial_state(cls, chem, *, temperature=298.15):
        # References: closed adiabatic constant-volume ideal-gas box, U = sum n_i u_i(T).
        # Units: amounts mol, energy J, volume m3, temperature K.
        n = torch.tensor([1.0e-8, 1.0e-8, 1.0e-8, 8.0e-3], dtype=torch.float64)
        t = torch.tensor(temperature, dtype=torch.float64)
        u = (n * chem.standard_molar_properties(t)["u"]).sum().detach()
        state = {name: n[i].clone() for i, name in enumerate(cls.SPECIES)}
        state["box_internal_energy"] = u.clone()
        state["box_volume"] = torch.tensor(1.0e-3, dtype=torch.float64)
        return state

    @classmethod
    def _reference_trajectory(cls, chem, state, *, n_steps, dt):
        """Explicit Euler reference from direct kernel calls (not governance evidence)."""
        current = {k: v.clone() for k, v in state.items()}
        entropies = []
        for _ in range(n_steps):
            out = chem.forward(dict(current), 0.0, dt)
            entropies.append(out["diagnostics.chemistry.box_entropy"].value.detach())
            for name in cls.STATE:
                current[name] = (current[name] + out[name].value).detach()
        return current, entropies

    @classmethod
    def _atom_totals(cls, values):
        n = torch.stack([values[name].reshape(()) for name in cls.SPECIES]).detach()
        return {
            element: float(torch.tensor(row, dtype=n.dtype) @ n)
            for element, row in cls.ATOMS.items()
        }

    @classmethod
    def _assert_atoms_conserved(cls, values, initial):
        # The per-step increments satisfy C nu = 0 exactly; the summed inventories
        # only agree to float64 roundoff because 8e-3 mol O2 absorbs 1e-9 mol steps.
        totals, reference = cls._atom_totals(values), cls._atom_totals(initial)
        for element in cls.ATOMS:
            assert totals[element] == pytest.approx(reference[element], rel=1e-13, abs=0.0), element

    def _registry_and_graph(self, chem):
        from relic.orchestration.graph_ir import GraphIR, NodeSpec
        from relic.orchestration.node_registry import NodeRegistry
        from relic.orchestration.ssot_catalog import SSOTCatalog

        registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
        definition = registry.register_functor_instance(chem, node_type_id=self.NODE_TYPE)
        definition.metadata["config_contract"] = {"physical_boundary": "closed_constant_volume"}
        graph = GraphIR(
            schema_version="1.0.0",
            nodes={
                "atmospheric_chemistry": NodeSpec(
                    class_type=self.NODE_TYPE,
                    inputs={name: {"state": name} for name in chem.metadata.input_variables},
                    outputs=list(chem.metadata.output_variables),
                    config={"physical_boundary": "closed_constant_volume"},
                )
            },
            links=[],
            metadata={},
        )
        return registry, graph

    def test_graphir_compiles_and_executes_the_closed_box(self, tmp_path):
        from relic.orchestration.compiler import compile_graph
        from relic.orchestration.executor import GraphExecutor

        chem = self._box()
        registry, graph = self._registry_and_graph(chem)
        plan = compile_graph(graph, registry, strict=True)
        assert plan.report.validation.errors == []

        initial = self._initial_state(chem)
        n_steps, dt = 3, 1.0
        reference, entropies = self._reference_trajectory(chem, initial, n_steps=n_steps, dt=dt)

        executor = GraphExecutor(plan, registry=registry)
        final = executor.execute({k: v.clone() for k, v in initial.items()}, n_steps=n_steps, dt=dt)

        # The executor runs on the simulator device policy (GPU first); compare values only.
        final = {key: value.detach().cpu() for key, value in final.items()}
        for name in self.STATE:
            torch.testing.assert_close(final[name].reshape(()), reference[name], rtol=1e-9, atol=0.0)
        assert float(final["box_internal_energy"]) == float(initial["box_internal_energy"])
        assert float(final["box_volume"]) == float(initial["box_volume"])
        self._assert_atoms_conserved(final, initial)
        assert float(final["box_no2_amount"]) > float(initial["box_no2_amount"])
        assert all(later > earlier for earlier, later in zip(entropies, entropies[1:]))
        provenance = executor._simulator.get_variable_provenance("box_no2_amount")
        assert provenance[-1]["functor"] == "atmospheric_chemistry"
        assert provenance[-1]["role"] == "final_writer"
        assert {record["functor"] for record in provenance} == {"atmospheric_chemistry"}
        assert executor._build_functor(plan.steps[0]).physical_boundary == "closed_constant_volume"

    def test_simulator_commits_one_governed_increment_with_closed_budgets(self):
        chem = self._box()
        initial = self._initial_state(chem)
        dt = 1.0
        kernel = chem.forward(dict(initial), 0.0, dt)
        expected_increment = {name: kernel[name].value.detach() for name in self.STATE}
        simulator = RELICSimulator({
            "max_time": dt,
            "time_step": dt,
            "device": "cpu",
            "conservation_checks": True,
            "strict_dependencies": True,
        })
        simulator.initialize_state({k: v.clone() for k, v in initial.items()})
        simulator.register_functor(chem, execution_group="fast")
        updates = simulator.step(dt=dt)
        for name in self.STATE:
            # The governed step reports the committed state difference, which
            # equals the accepted increment up to the ulp of the stored inventory.
            ulp = 4.0 * torch.finfo(torch.float64).eps * float(abs(initial[name]))
            torch.testing.assert_close(updates[name].reshape(()), expected_increment[name], rtol=0.0, atol=ulp)
            torch.testing.assert_close(
                simulator.state_manager.get_state(name).reshape(()),
                initial[name] + expected_increment[name],
                rtol=0.0,
                atol=0.0,
            )
        assert float(expected_increment["box_no2_amount"]) > 0.0 > float(expected_increment["box_no_amount"])
        assert simulator.step_count == 1
        self._assert_atoms_conserved(simulator.state_manager.get_state(), initial)
        assert float(simulator.state_manager.get_state("box_internal_energy")) == float(initial["box_internal_energy"])
        assert float(simulator.state_manager.get_state("box_volume")) == float(initial["box_volume"])

    def test_oversized_step_is_rejected_and_state_is_untouched(self):
        chem = self._box()
        initial = self._initial_state(chem)
        simulator = RELICSimulator({
            "max_time": 1.0e6,
            "time_step": 1.0,
            "device": "cpu",
            "conservation_checks": True,
            "strict_dependencies": True,
        })
        simulator.initialize_state({k: v.clone() for k, v in initial.items()})
        simulator.register_functor(chem, execution_group="fast")
        with pytest.raises(Exception, match="positivity"):
            simulator.step(dt=600.0)
        assert simulator.step_count == 0
        for name, value in initial.items():
            torch.testing.assert_close(simulator.state_manager.get_state(name).reshape(()), value, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("boundary", ["exact_limit", "previous_float", "next_float"])
    @pytest.mark.parametrize("accepted_before_rejection", [False, True])
    def test_boundary_rejection_rolls_back_and_next_valid_step_commits_once(
        self, boundary, accepted_before_rejection,
    ):
        # References: existing closed-box thermodynamic domain and StateManager transaction contract.
        # Mathematical expression: candidate=x+h*f must stay in domain; rejected Delta_x=0.
        # Parameter meanings: h inventory-only bound; f one-step drift; C_N/C_O elemental ledgers.
        # Units: inventory mol, energy J, volume m3, time s, entropy J K-1.
        # Domain/constraints: exact/adjacent boundary, both before and after an accepted public step.
        # Sign convention: NO/O3 losses, NO2/O2 gains; valid closed-box energy is unchanged.
        chem = self._box()
        initial = self._initial_state(chem)
        simulator = RELICSimulator({
            "max_time": 100.0, "time_step": 1.0, "device": "cpu",
            "conservation_checks": True, "strict_dependencies": True,
        })
        simulator.initialize_state({key: value.clone() for key, value in initial.items()})
        simulator.register_functor(chem, execution_group="fast")
        if accepted_before_rejection:
            simulator.step(dt=1.0)
        snapshot = {key: value.detach().clone() for key, value in simulator.get_state().items()}
        time_before = simulator.current_time
        steps_before = simulator.step_count
        provenance_before = copy.deepcopy(simulator.get_variable_provenance())

        reference = self._box()
        valid_increment = reference(dict(snapshot), time=time_before, dt=1.0)
        limit = float(snapshot["box_no_amount"].reshape(()) / (-valid_increment["box_no_amount"].value.detach()))
        rejected_dt = {
            "exact_limit": limit,
            "previous_float": math.nextafter(limit, 0.0),
            "next_float": math.nextafter(limit, math.inf),
        }[boundary]
        with pytest.raises(Exception, match="strictly positive|affinity|positivity"):
            simulator.step(dt=rejected_dt)
        assert simulator.current_time == time_before
        assert simulator.step_count == steps_before
        assert chem.last_strict_iphs_audit() is None
        restored = simulator.get_state()
        assert set(restored) == set(snapshot)
        for key, value in snapshot.items():
            torch.testing.assert_close(restored[key], value, rtol=0.0, atol=0.0)
        assert simulator.get_variable_provenance() == provenance_before

        # The next public accepted step must commit the one-second increment exactly once.
        updates = simulator.step(dt=1.0)
        final = simulator.get_state()
        assert simulator.current_time == time_before + 1.0
        assert simulator.step_count == steps_before + 1
        for name in self.STATE:
            expected = snapshot[name].reshape(()) + valid_increment[name].value.detach()
            torch.testing.assert_close(final[name].reshape(()), expected, rtol=0.0, atol=0.0)
            ulp = 4.0 * torch.finfo(torch.float64).eps * float(abs(snapshot[name].reshape(())))
            torch.testing.assert_close(updates[name].reshape(()), valid_increment[name].value.detach(), rtol=0.0, atol=ulp)
        self._assert_atoms_conserved(final, initial)
        assert float(final["box_internal_energy"]) == float(initial["box_internal_energy"])
        assert float(final["box_volume"]) == float(initial["box_volume"])
        assert all(float(final[name]) > 0.0 for name in self.SPECIES)
        audit = chem.last_strict_iphs_audit()
        assert audit["runtime_verified_level"] == 5
        assert audit["reaction_energy_rate"] == 0.0
        assert audit["entropy_production"] > 0.0
        next_point = reference._closed_box_thermodynamic_point(final)
        assert float(next_point["affinity"].detach()) > 0.0
        old_entropy = valid_increment["diagnostics.chemistry.box_entropy"].value.detach()
        assert float(next_point["mixture"]["entropy"].detach()) > float(old_entropy)
        final_provenance = simulator.get_variable_provenance("box_no2_amount")
        assert final_provenance[-1]["functor"] == "atmospheric_chemistry"
        assert final_provenance[-1]["role"] == "final_writer"

    @pytest.mark.parametrize("missing", ["box_internal_energy", "box_volume", "box_o2_amount"])
    def test_missing_closed_box_inputs_fail_the_governed_step(self, missing):
        chem = self._box()
        initial = self._initial_state(chem)
        initial.pop(missing)
        simulator = RELICSimulator({
            "max_time": 1.0,
            "time_step": 1.0,
            "device": "cpu",
            "conservation_checks": True,
            "strict_dependencies": True,
        })
        simulator.initialize_state({k: v.clone() for k, v in initial.items()})
        simulator.register_functor(chem, execution_group="fast")
        with pytest.raises(Exception, match=missing):
            simulator.step(dt=1.0)
        assert simulator.step_count == 0

    def test_checkpoint_restart_reproduces_the_uninterrupted_trajectory(self, tmp_path):
        chem = self._box()
        initial = self._initial_state(chem)
        dt = 1.0

        def make_simulator():
            simulator = RELICSimulator({
                "max_time": 10.0 * dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": True,
                "strict_dependencies": True,
            })
            simulator.initialize_state({k: v.clone() for k, v in initial.items()})
            simulator.register_functor(self._box(), execution_group="fast")
            return simulator

        uninterrupted = make_simulator()
        for _ in range(4):
            uninterrupted.step(dt=dt)

        source = make_simulator()
        for _ in range(2):
            source.step(dt=dt)
        checkpoint = tmp_path / "closed_box.pt"
        source.create_checkpoint(checkpoint)
        for _ in range(2):
            source.step(dt=dt)

        restored = make_simulator()
        restored.restore_checkpoint(checkpoint)
        assert restored.step_count == 2
        for _ in range(2):
            restored.step(dt=dt)

        for name in self.STATE:
            expected = uninterrupted.state_manager.get_state(name).reshape(())
            torch.testing.assert_close(source.state_manager.get_state(name).reshape(()), expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(restored.state_manager.get_state(name).reshape(()), expected, rtol=0.0, atol=0.0)
        assert restored.step_count == uninterrupted.step_count == 4
        self._assert_atoms_conserved(restored.state_manager.get_state(), initial)
