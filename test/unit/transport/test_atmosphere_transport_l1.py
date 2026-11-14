#!/usr/bin/env python3
"""
L1 Unit Tests for Atmosphere Transport Mechanisms

This module provides comprehensive L1 (unit-level) testing for atmospheric transport
functors including eddy diffusion and dry deposition mechanisms.

Test Coverage:
    - EddyDiffusionBLMFunctor: Boundary layer eddy diffusion transport
    - DryDepositionAggregator: Dry deposition flux aggregation
    - Parameter validation and boundary conditions
    - Numerical stability and conservation checks
    - Canonical execution and fail-fast input validation

Scientific Validation:
    - MOST stability functions verification
    - CFL stability condition checking
    - Physical parameter bounds validation
    - Units consistency verification
    - Mass conservation verification

Test Structure:
    Level 1: Individual component validation
    - Parameter initialization and bounds
    - Core calculation methods
    - Error handling and edge cases
    - Diagnostic output validation

References:
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology
    - Wesely, M.L. (1989). Parameterization of surface resistances to gaseous dry deposition
    - Businger, J.A. et al. (1971). Flux-profile relationships in atmospheric surface layer
"""

import sys
import os
import pytest
import torch

# Add relic to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the modules to test
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
    EddyDiffusionBLMFunctor, EddyDiffusionConfig
)
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.atmosphere.most.similarity_functions import phi_h
from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionAggregator, DryDepositionConfig
)
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.utils.physical_constants import get_von_karman


class TestEddyDiffusionBLMFunctor:
    """Test suite for atmospheric boundary layer eddy diffusion transport."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for eddy diffusion testing."""
        return EddyDiffusionConfig(
            enabled=True,
        )
    
    @pytest.fixture 
    def eddy_functor(self, basic_config):
        """Create eddy diffusion functor instance."""
        return EddyDiffusionBLMFunctor(
            name="test_eddy_diffusion",
            transport_config=basic_config
        )
    
    @pytest.fixture
    def basic_state(self):
        """Basic atmospheric state for testing."""
        return {
            'no_mixing_ratio': torch.tensor(10e-9),   # 10 ppb
            'no2_mixing_ratio': torch.tensor(20e-9),  # 20 ppb  
            'o3_mixing_ratio': torch.tensor(50e-9),   # 50 ppb
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(2e-9),
            'pbl_height': torch.tensor(800.0),
            'friction_velocity': torch.tensor(0.3),
            'monin_obukhov_length': torch.tensor(-200.0),
        }
    
    def test_initialization_default(self):
        """Test default initialization of eddy diffusion functor."""
        functor = EddyDiffusionBLMFunctor()
        expected_kappa = get_von_karman()
        
        # Check basic properties
        assert functor.name == "eddy_diffusion_blm"
        assert not functor.config_transport.enabled  # Default disabled
        metadata = functor.get_transport_metadata()
        snapshot = metadata['parameter_snapshot']
        assert snapshot['von_karman_constant'] == pytest.approx(expected_kappa)
        assert metadata['static_diffusivity'] is None
        assert metadata['diffusivity_model'] == 'dynamic_most_k_theory'
        
        # Check parameter bounds
        assert 0.3 <= snapshot['von_karman_constant'] <= 0.5
        assert snapshot['k_bounds'][0] > 0
        assert snapshot['k_bounds'][1] > snapshot['k_bounds'][0]
        
    def test_initialization_custom_config(self, basic_config):
        """Test initialization with custom configuration."""
        functor = EddyDiffusionBLMFunctor(transport_config=basic_config)
        
        assert functor.config_transport.enabled
        assert functor.target_species == ("no", "no2", "o3", "isoprene")
        
    def test_parameter_validation_bounds(self):
        """Test that numeric and species overrides cannot bypass SSOT."""
        forbidden_overrides = [
            {'von_karman_constant': 0.1},
            {'k_min': 10.0, 'k_max': 5.0},
            {'target_species': ['no', 'no2', 'o3']},
            {'cfl_factor': 0.5},
        ]
        for override in forbidden_overrides:
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                EddyDiffusionConfig(**override)

    def test_constant_diffusivity_paths_fail_fast(self, eddy_functor):
        """Generic constant-D formulas must not become a second Eddy runtime."""
        field = torch.ones(2)
        with pytest.raises(RuntimeError, match="state-dependent"):
            eddy_functor._resolve_diffusivity_tensor(field)
        with pytest.raises(RuntimeError, match="requires PBL"):
            eddy_functor.compute_diffusivity_tensor(field, field)
        with pytest.raises(RuntimeError, match="canonical forward"):
            eddy_functor.boundary_conditions(
                "z_max",
                field,
                torch.ones(2, 1),
            )
    
    def test_dependencies(self, eddy_functor):
        """Test dependency declaration."""
        deps = eddy_functor.get_dependencies()
        
        assert deps == ['atmospheric_turbulence']
        
    def test_forward_disabled(self, eddy_functor, basic_state):
        """Test forward pass when functor is disabled."""
        eddy_functor.config_transport.enabled = False
        
        output = eddy_functor.forward(
            state=basic_state,
            time=0.0,
            dt=60.0,
        )
        assert set(output) == {'diagnostics.eddy_diffusion.enabled'}
        assert output['diagnostics.eddy_diffusion.enabled'].item() == 0.0

    def test_forward_requires_canonical_inputs(self, eddy_functor, basic_state):
        """Enabled transport fails fast when a declared input port is absent."""
        incomplete_state = dict(basic_state)
        incomplete_state.pop('friction_velocity')
        with pytest.raises(RuntimeError, match="Missing required Eddy diffusion"):
            eddy_functor.forward(incomplete_state, time=0.0, dt=60.0)

    def test_forward_k_theory_formula(self, eddy_functor, basic_state):
        """The state increment equals dt*K/h^2*(C_background-C)."""
        output = eddy_functor.forward(
            state=basic_state,
            time=0.0,
            dt=60.0,
        )
        k_value = output['diagnostics.eddy_diffusion.eddy_diffusivity']
        h_value = basic_state['pbl_height']
        backgrounds = {
            'no_mixing_ratio': eddy_functor._transport_parameters['background_no_mixing_ratio'],
            'no2_mixing_ratio': eddy_functor._transport_parameters['background_no2_mixing_ratio'],
            'o3_mixing_ratio': eddy_functor._transport_parameters['background_o3_mixing_ratio'],
            'isoprene_mixing_ratio': eddy_functor._transport_parameters[
                'background_isoprene_mixing_ratio'
            ],
        }
        for state_key, background in backgrounds.items():
            item = output[state_key]
            assert isinstance(item, UpdateItem)
            expected = 60.0 * k_value / h_value**2 * (
                basic_state[state_key].new_tensor(background)
                - basic_state[state_key]
            )
            assert torch.allclose(item.value, expected, rtol=1e-6, atol=1e-15)

    def test_forward_update_contract(self, eddy_functor, basic_state):
        """Canonical state writes carry explicit increment time and unit semantics."""
        output = eddy_functor.forward(basic_state, time=0.0, dt=60.0)
        state_keys = {
            'no_mixing_ratio',
            'no2_mixing_ratio',
            'o3_mixing_ratio',
            'isoprene_mixing_ratio',
        }
        assert state_keys.issubset(output)
        for key in state_keys:
            item = output[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == 'mol mol-1'
            assert item.source_functor == eddy_functor.name
            assert torch.isfinite(item.value).all()
    
    def test_most_stability_functions(self, eddy_functor):
        """Test MOST stability function calculations."""
        # Test stable conditions (z/L > 0)
        z_over_L_stable = torch.tensor([0.1, 0.5, 1.0])
        phi_h_stable = phi_h(z_over_L_stable)
        
        # Stable conditions: phi_h = 1 + beta_h*z/L, so phi_h > 1.
        assert torch.all(phi_h_stable > 1.0)
        assert torch.all(phi_h_stable < 10.0)  # Reasonable upper bound
        
        # Test unstable conditions (z/L < 0)
        z_over_L_unstable = torch.tensor([-0.1, -0.5, -1.0])
        phi_h_unstable = phi_h(z_over_L_unstable)
        
        # Unstable conditions typically have phi_h < 1.
        assert torch.all(phi_h_unstable >= 0.1)  # Numerical stability bound
        assert torch.all(phi_h_unstable <= 10.0)
        
        # Test neutral conditions (z/L = 0).
        z_over_L_neutral = torch.tensor([0.0])
        phi_h_neutral = phi_h(z_over_L_neutral)
        assert torch.allclose(phi_h_neutral, torch.tensor(1.0), atol=0.1)
    
    def test_eddy_diffusivity_calculation(self, eddy_functor):
        """Test eddy diffusivity calculation."""
        pbl_height = torch.tensor(800.0)
        
        # Mock turbulence data 
        turbulence_data = {
            'friction_velocity': torch.tensor(0.3),
            'monin_obukhov_length': torch.tensor(-200.0)
        }
        
        K, z_over_l, clamped, clipping_residual = (
            eddy_functor._calculate_eddy_diffusivity(
                pbl_height,
                turbulence_data,
            )
        )
        
        # Check K is within reasonable bounds
        k_bounds = eddy_functor.get_transport_metadata()[
            'parameter_snapshot'
        ]['k_bounds']
        assert K >= k_bounds[0]
        assert K <= k_bounds[1]
        assert torch.isfinite(K)
        assert torch.isfinite(z_over_l)
        assert clamped.item() in {0.0, 1.0}
        assert torch.allclose(clipping_residual, torch.zeros_like(z_over_l))
        
        # K is in m2 s-1 and remains inside configured physical bounds.
        assert 0.01 <= K.item() <= 100.0
    
    def test_cfl_stability_check(self, eddy_functor):
        """Test CFL stability condition checking."""
        K = torch.tensor(10.0)  # m2 s-1
        pbl_height = torch.tensor(1000.0)  # m
        dt = 3600.0  # 1 hour

        stability = eddy_functor._check_cfl_stability(K, pbl_height, dt)
        assert torch.allclose(stability, torch.tensor(0.036))

        # A shallow box and long step must fail instead of partially advancing.
        with pytest.raises(ValueError, match="use dt <="):
            eddy_functor._check_cfl_stability(
                torch.tensor(50.0),
                torch.tensor(50.0),
                60.0,
            )

        # A small time step remains stable.
        dt_small = 10.0  # 10 seconds
        stability_small = eddy_functor._check_cfl_stability(
            K,
            pbl_height,
            dt_small,
        )
        assert torch.all(stability_small < stability)
    
    def test_species_transport_calculation(self, eddy_functor, basic_state):
        """Test species-specific transport rate calculations."""
        K = torch.tensor(5.0)  # m2 s-1
        pbl_height = torch.tensor(800.0)  # m
        dt = 300.0  # 5 minutes
        
        updates = eddy_functor._calculate_species_transport(
            basic_state,
            K,
            pbl_height,
            dt,
        )
        
        # Should have updates for all target species
        expected_keys = {
            'no_mixing_ratio',
            'no2_mixing_ratio',
            'o3_mixing_ratio',
            'isoprene_mixing_ratio',
        }
        assert expected_keys == set(updates)
        
        # All updates should be tensors
        for key in expected_keys:
            assert isinstance(updates[key], torch.Tensor)
            assert torch.isfinite(updates[key])
    
    def test_diagnostics_output(self, eddy_functor, basic_state):
        """Test diagnostic information output."""
        output = eddy_functor.forward(
            state=basic_state,
            time=0.0,
            dt=60.0,
        )

        diagnostic_keys = {
            key for key in eddy_functor.metadata.output_variables
            if key.startswith('diagnostics.')
        }
        assert diagnostic_keys.issubset(output)
        for key in diagnostic_keys:
            value = output[key]
            assert isinstance(value, torch.Tensor)
            assert torch.isfinite(value).all()
        assert output['diagnostics.eddy_diffusion.enabled'].item() == 1.0
        assert output['diagnostics.eddy_diffusion.pbl_height'].item() > 0.0
        assert output['diagnostics.eddy_diffusion.friction_velocity'].item() > 0.0
    
    def test_boundary_conditions_extreme_inputs(self, eddy_functor):
        """Test handling of extreme input conditions."""
        # Test the accepted lower PBL and concentration bounds.
        cold_state = {
            'no_mixing_ratio': torch.tensor(1e-12),  # Very low
            'no2_mixing_ratio': torch.tensor(1e-12),
            'o3_mixing_ratio': torch.tensor(1e-12),
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(1e-12),
            'pbl_height': torch.tensor(50.0),
            'friction_velocity': torch.tensor(0.01),
            'monin_obukhov_length': torch.tensor(-50.0),
        }
        
        output = eddy_functor.forward(
            state=cold_state,
            time=0.0,
            dt=10.0,
        )
        
        # Output should be finite
        for key, value in output.items():
            if isinstance(value, UpdateItem):
                value = value.value
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
        
        # Test with very high values
        hot_state = {
            'no_mixing_ratio': torch.tensor(1e-6),   # Very high
            'no2_mixing_ratio': torch.tensor(1e-6),
            'o3_mixing_ratio': torch.tensor(1e-6),
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(1e-6),
            'pbl_height': torch.tensor(3000.0),
            'friction_velocity': torch.tensor(2.0),
            'monin_obukhov_length': torch.tensor(2000.0),
        }
        
        # Should not crash
        output_hot = eddy_functor.forward(
            state=hot_state,
            time=0.0,
            dt=60.0,
        )
        
        # Output should be finite
        for key, value in output_hot.items():
            if isinstance(value, UpdateItem):
                value = value.value
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
    
    def test_missing_turbulence_data_handling(self, eddy_functor, basic_state):
        """Missing turbulence ports fail fast without an empirical fallback."""
        state = dict(basic_state)
        state.pop('friction_velocity')
        state.pop('monin_obukhov_length')
        with pytest.raises(RuntimeError, match="friction_velocity"):
            eddy_functor.forward(state, time=0.0, dt=60.0)


class TestDryDepositionAggregator:
    """Test the canonical dry-deposition transport interface."""

    @pytest.fixture
    def deposition_functor(self):
        """Create an enabled canonical dry-deposition Functor."""
        return DryDepositionAggregator(
            name="test_dry_deposition",
            deposition_config=DryDepositionConfig(enabled=True),
        )

    @pytest.fixture
    def deposition_state(self):
        """Build a complete scalar dry-deposition state."""
        return {
            "no_mixing_ratio": torch.tensor(10.0e-9),
            "no2_mixing_ratio": torch.tensor(25.0e-9),
            "o3_mixing_ratio": torch.tensor(60.0e-9),
            "atmospheric_pressure": torch.tensor(101325.0),  # Pa
            "co_mixing_ratio": torch.tensor(150e-9),  # OH sink, 150 ppb
            "ch4_mixing_ratio": torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            "specific_humidity": torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            "cos_zenith": torch.tensor(1.0),  # overhead sun
            "so2_mixing_ratio": torch.tensor(2.0e-9),
            "pbl_height": torch.tensor(1200.0),
            "aerodynamic_resistance": torch.tensor(30.0),
        }

    def test_initialization_uses_canonical_parameter_contract(self):
        """Initialization exposes one immutable closure and species order."""
        functor = DryDepositionAggregator()

        assert functor.name == "dry_deposition_aggregator"
        assert functor.config_deposition.enabled
        assert functor.target_species == ("no", "no2", "o3", "so2")
        assert functor.get_transport_metadata()["closure"] == "resistance_law"
        assert not hasattr(functor, "config_transport")

    def test_disabled_forward_has_no_source_or_sink_writer(self):
        """Disabled execution has no identity or zero state output."""
        functor = DryDepositionAggregator(
            deposition_config=DryDepositionConfig(enabled=False),
        )

        output = functor.forward({}, time=0.0, dt=60.0)

        assert set(output) == {
            "diagnostics.dry_deposition.enabled",
            "diagnostics.dry_deposition.normalized_mass_balance_residual",
        }
        assert output["diagnostics.dry_deposition.enabled"].item() == 0.0

    def test_canonical_closure_writes_all_declared_species(
        self,
        deposition_functor,
        deposition_state,
    ):
        """All declared species receive finite non-positive increments."""
        output = deposition_functor.forward(
            state=deposition_state,
            time=0.0,
            dt=60.0,
        )

        for species in deposition_functor.target_species:
            key = deposition_functor.species_map[species]
            item = output[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep
            assert torch.isfinite(item.value)
            assert item.value < 0.0

    def test_species_surface_resistance_changes_response(
        self,
        deposition_functor,
        deposition_state,
    ):
        """Species-specific YAML resistance remains visible in the response."""
        resistances = {
            species: params.effective_resistance
            for species, params in deposition_functor.species_parameters.items()
        }
        assert all(value > 0.0 for value in resistances.values())
        assert len(set(resistances.values())) > 1

        state = {
            **deposition_state,
            "no_mixing_ratio": torch.tensor(10.0e-9),
            "no2_mixing_ratio": torch.tensor(10.0e-9),
            "o3_mixing_ratio": torch.tensor(10.0e-9),
            "atmospheric_pressure": torch.tensor(101325.0),  # Pa
            "co_mixing_ratio": torch.tensor(150e-9),  # OH sink, 150 ppb
            "ch4_mixing_ratio": torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            "specific_humidity": torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            "cos_zenith": torch.tensor(1.0),  # overhead sun
            "so2_mixing_ratio": torch.tensor(10.0e-9),
        }
        output = deposition_functor.forward(state, time=0.0, dt=60.0)
        ordered = sorted(resistances, key=resistances.get)
        low_resistance = output[
            deposition_functor.species_map[ordered[0]]
        ].value.abs()
        high_resistance = output[
            deposition_functor.species_map[ordered[-1]]
        ].value.abs()
        assert low_resistance > high_resistance

    def test_lower_upstream_resistance_increases_deposition(
        self,
        deposition_functor,
        deposition_state,
    ):
        """The sole upstream turbulence input has the expected monotonic effect."""
        high_resistance_state = {
            **deposition_state,
            "aerodynamic_resistance": torch.tensor(100.0),
        }
        low_resistance_state = {
            **deposition_state,
            "aerodynamic_resistance": torch.tensor(10.0),
        }

        high_resistance = deposition_functor.forward(
            high_resistance_state,
            time=0.0,
            dt=60.0,
        )
        low_resistance = deposition_functor.forward(
            low_resistance_state,
            time=0.0,
            dt=60.0,
        )

        for species in deposition_functor.target_species:
            key = deposition_functor.species_map[species]
            assert low_resistance[key].value.abs() > high_resistance[key].value.abs()

    def test_batch_and_extreme_concentrations_remain_finite(
        self,
        deposition_functor,
    ):
        """Broadcasted grid cells preserve sign, finiteness, and inventory bounds."""
        state = {
            "no_mixing_ratio": torch.tensor([1.0e-12, 1.0e-6]),
            "no2_mixing_ratio": torch.tensor([2.0e-12, 2.0e-6]),
            "o3_mixing_ratio": torch.tensor([5.0e-12, 5.0e-6]),
            "atmospheric_pressure": torch.tensor([101325.0, 101325.0]),  # Pa
            "co_mixing_ratio": torch.tensor([150e-9, 150e-9]),  # OH sink, 150 ppb
            "ch4_mixing_ratio": torch.tensor([1.8e-6, 1.8e-6]),  # OH sink, 1.8 ppm
            "specific_humidity": torch.tensor([0.01, 0.01]),  # O(1D) + H2O OH source, 10 g/kg
            "cos_zenith": torch.tensor([1.0, 1.0]),  # overhead sun
            "so2_mixing_ratio": torch.tensor([1.0e-13, 1.0e-7]),
            "pbl_height": torch.tensor([100.0, 2000.0]),
            "aerodynamic_resistance": torch.tensor([10.0, 100.0]),
        }

        output = deposition_functor.forward(state, time=0.0, dt=3600.0)

        for species in deposition_functor.target_species:
            key = deposition_functor.species_map[species]
            increment = output[key].value
            assert increment.shape == state[key].shape
            assert torch.isfinite(increment).all()
            assert torch.all(increment <= 0.0)
            assert torch.all(state[key] + increment >= 0.0)


class TestChemistryCanonicalRuntime:
    """Test the single canonical atmospheric chemistry runtime."""
    
    @pytest.fixture
    def chemistry_functor(self):
        """Create chemistry functor for testing."""
        return ChemistryFunctor(physical_boundary="open_mechanism")
    
    @pytest.fixture
    def chemistry_state(self):
        """Basic chemistry state."""
        return {
            'air_temperature': torch.tensor(298.15),
            'no_mixing_ratio': torch.tensor(10e-9),
            'no2_mixing_ratio': torch.tensor(20e-9),
            'o3_mixing_ratio': torch.tensor(50e-9),
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(500.0)
        }
    
    def test_canonical_runtime_has_no_debug_mode(
        self,
        chemistry_functor,
        chemistry_state,
    ):
        """Canonical chemistry must not expose a dormant debug formula mode."""

        assert 'debug_sensitivity' not in chemistry_functor.metadata.parameters
        assert not any(
            key.endswith('.debug_sensitivity')
            for key in chemistry_functor.required_parameters()
        )

        result = chemistry_functor.forward(
            chemistry_state,
            time=0.0,
            dt=1.0,
        )
        no_increment = result['no_mixing_ratio']
        assert isinstance(no_increment, UpdateItem)
        assert torch.isfinite(no_increment.value).all()

    def test_debug_mode_override_is_rejected(self):
        """Removed debug-mode configuration must fail instead of being ignored."""

        with pytest.raises(ValueError, match='Unknown chemistry parameter override'):
            ChemistryFunctor(
                config={'parameters': {'debug_sensitivity': True}}, physical_boundary="open_mechanism",
            )

    def test_bvoc_uses_same_canonical_runtime(
        self,
        chemistry_functor,
        chemistry_state,
    ):
        """BVOC chemistry adds declared species updates without a mode switch."""

        bvoc_state = chemistry_state.copy()
        bvoc_state['isoprene_mixing_ratio'] = torch.tensor(2e-9)

        result = chemistry_functor.forward(
            bvoc_state,
            time=0.0,
            dt=1.0,
        )
        isoprene_increment = result['isoprene_mixing_ratio']
        assert isinstance(isoprene_increment, UpdateItem)
        assert isoprene_increment.mode == UpdateMode.ACCUMULATE
        assert isoprene_increment.value < 0.0


def run_l1_tests():
    """Run all L1 unit tests for atmosphere transport."""
    test_files = [__file__]
    
    print("Running L1 Unit Tests for Atmosphere Transport...")
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
        print("ALL L1 ATMOSPHERE TRANSPORT TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME L1 TESTS FAILED - Check output above")
        print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    # Run tests when called directly
    exit_code = run_l1_tests()
    sys.exit(exit_code)

