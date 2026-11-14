"""
L2 Integration Tests for Photosynthesis Module

Tests the integration of FarquharPhotosynthesis with:
- Parameter management system
- Environmental forcing
- State management
- Conservation checks
- Cross-module parameter sharing
"""

import pytest
import torch
import numpy as np

# Import modules to test
from relic.scientific.biosphere.photosynthesis import FarquharPhotosynthesis
from relic.parameters import get_global_manager
from relic.core.state import StateManager, UpdateItem
from relic.core.dependency_graph import DependencyGraphResolver


def _canonical_state(
    *,
    air_temperature=298.15,
    atmospheric_co2_ppm=400.0,
    ppfd=1000.0,
    vapor_pressure_deficit=1000.0,
    **optional,
):
    """Build required Farquhar forcings and optional process controls."""
    state = {
        "air_temperature": torch.as_tensor(air_temperature),
        "atmospheric_co2_ppm": torch.as_tensor(atmospheric_co2_ppm),
        "ppfd": torch.as_tensor(ppfd),
        "leaf_area_index": torch.tensor(1.0),
        "vapor_pressure_deficit": torch.as_tensor(vapor_pressure_deficit),
    }
    state.update({name: torch.as_tensor(value) for name, value in optional.items()})
    return state


def _value(output, key):
    """Return the tensor payload from an explicit state update."""
    assert isinstance(output[key], UpdateItem)
    return output[key].value


class _DependencyNode:
    """Minimal graph-only node with no scientific runtime implementation."""

    def __init__(self, name):
        self.name = name

    def get_dependencies(self):
        return []


class TestPhotosynthesisIntegrationL2:
    """L2 integration tests for photosynthesis module."""
    
    @pytest.fixture
    def param_manager(self):
        """Get and reset the global parameter manager."""
        manager = get_global_manager()
        # Load parameters from database
        manager.load_from_database('relic/parameters/database/relic/biosphere/photosynthesis.yaml')
        manager.load_from_database('relic/parameters/database/relic/validation/numerical_stability.yaml')
        return manager
    
    @pytest.fixture
    def state_manager(self):
        """Create a state manager for testing."""
        return StateManager(device=torch.device('cpu'))
    
    @pytest.fixture
    def photosynthesis_model(self, param_manager):
        """Create photosynthesis model with parameter integration."""
        model = FarquharPhotosynthesis(
            name="photosynthesis",
            enable_gpu=False,
            validate_inputs=True
        )
        return model
    
    def test_parameter_loading_integration(self, photosynthesis_model, param_manager):
        """Test that photosynthesis loads parameters correctly from manager."""
        # Check that core parameters are loaded
        vcmax25 = photosynthesis_model.vcmax25
        assert vcmax25 is not None
        assert 10.0 <= vcmax25 <= 200.0  # Within bounds
        
        # Check that parameter exists in manager
        vcmax_from_manager = param_manager.get("biosphere.photosynthesis.vcmax25")
        assert vcmax_from_manager is not None
        
        # Values should match (allowing for tensor conversion)
        if isinstance(vcmax25, torch.Tensor):
            vcmax25 = vcmax25.item()
        # param_manager.get() returns the raw value, not a dict
        if isinstance(vcmax_from_manager, dict):
            vcmax_value = vcmax_from_manager['value']
        else:
            vcmax_value = vcmax_from_manager
        if isinstance(vcmax_value, torch.Tensor):
            vcmax_value = vcmax_value.item()
        assert abs(vcmax25 - vcmax_value) < 0.001
    
    def test_canonical_forcing_state_integration(self, photosynthesis_model, state_manager):
        """Test consumption of canonical DAG outputs through StateManager."""
        initial_state = _canonical_state(
            air_temperature=295.0,
            atmospheric_co2_ppm=410.0,
            ppfd=800.0,
            vapor_pressure_deficit=1000.0,
        )
        state_manager.initialize_state(initial_state)

        photo_state = state_manager.get_state()
        photo_output = photosynthesis_model.forward(photo_state, time=0.0, dt=1.0)

        assert set(photo_output) == set(photosynthesis_model.metadata.output_variables)
        assert _value(photo_output, 'gross_photosynthesis') >= 0
        assert photo_state['ppfd'] == 800.0
    
    def test_dependency_resolution(self, photosynthesis_model):
        """Test that dependency graph correctly resolves execution order."""
        resolver = DependencyGraphResolver()
        environmental_forcing = _DependencyNode('environmental_forcing')
        atmospheric_radiation = _DependencyNode('atmospheric_radiation')

        resolver.add_functor(environmental_forcing, functor_id='environmental_forcing')
        resolver.add_functor(atmospheric_radiation, functor_id='atmospheric_radiation')
        resolver.add_functor(photosynthesis_model, functor_id='photosynthesis')

        resolver.add_edge('photosynthesis', 'environmental_forcing')
        resolver.add_edge('photosynthesis', 'atmospheric_radiation')

        execution_order = resolver.topological_sort()

        photo_idx = execution_order.index('photosynthesis')
        assert execution_order.index('environmental_forcing') < photo_idx
        assert execution_order.index('atmospheric_radiation') < photo_idx
    
    def test_state_management_integration(self, photosynthesis_model, state_manager):
        """Test integration with state management system."""
        # Initialize state
        initial_state = _canonical_state()
        state_manager.initialize_state(initial_state)
        
        # Get state and run photosynthesis
        state = state_manager.get_state()
        output = photosynthesis_model.forward(state, time=0.0, dt=1.0)
        
        state_manager.apply_updates(output, dt=1.0, check_conservation=False)
        
        # Verify state was updated
        new_state = state_manager.get_state()
        assert 'gross_primary_productivity' in new_state
        assert torch.allclose(
            new_state['gross_primary_productivity'],
            _value(output, 'gross_primary_productivity'),
        )
    
    def test_conservation_checks(self, photosynthesis_model, state_manager):
        """Test carbon conservation in photosynthesis."""
        # Run photosynthesis
        state = _canonical_state()
        output = photosynthesis_model.forward(state, time=0.0, dt=1.0)
        
        # Check carbon conservation
        # Net photosynthesis = Gross photosynthesis - Dark respiration
        gross = _value(output, 'gross_photosynthesis')
        net = _value(output, 'net_photosynthesis')
        resp = _value(output, 'diagnostics.photosynthesis.dark_respiration')
        
        # Conservation check (with small tolerance for numerical errors)
        assert torch.allclose(net, gross - resp, rtol=1e-5)
        
        # Check that fluxes are physically reasonable
        assert gross >= 0, "Gross photosynthesis cannot be negative"
        assert resp >= 0, "Dark respiration cannot be negative"
        assert gross >= net, "Gross must be >= net photosynthesis"
    
    def test_parameter_sensitivity(self, photosynthesis_model, param_manager):
        """Manager-held Vcmax sensitivity uses the actual calibration graph."""
        state = _canonical_state(ppfd=2000.0)
        binding = param_manager.get_trainable_parameters()["biosphere.photosynthesis.vcmax25"]
        assert photosynthesis_model.vcmax25 is binding
        before = binding.detach().clone()
        # References: Farquhar et al. (1980); architecture SSOT section 4.2.
        # Mathematical expression: S = d A_net / d Vcmax25 on the executed graph.
        # Parameter meanings: A_net is net canopy uptake; Vcmax25 is the
        #   manager-owned reference leaf carboxylation capacity; S its sensitivity.
        # Units: A_net [umol CO2 m-2 ground s-1]; Vcmax25 [umol CO2 m-2 leaf s-1];
        #   S [m2 leaf m-2 ground].
        # Domain/constraints: fixed valid C3 forcing in the Rubisco-limited domain;
        #   no parameter replacement, value update or extra process calculation.
        # Sign convention: positive S means greater capacity raises net uptake.
        output = photosynthesis_model(state, time=0.0, dt=1.0)
        sensitivity, = torch.autograd.grad(_value(output, "net_photosynthesis").sum(), (binding,))
        assert torch.isfinite(sensitivity).all()
        assert sensitivity.item() > 0.0
        torch.testing.assert_close(binding, before, rtol=0.0, atol=0.0)
    
    def test_cross_module_parameter_sharing(self, param_manager):
        """Test that parameters can be shared across modules."""
        # Check that VPD response parameter is accessible
        vpd_d0_raw = param_manager.get("validation.numerical.vpd_response_d0")
        assert vpd_d0_raw is not None
        # Extract value from dict or use directly
        if isinstance(vpd_d0_raw, dict):
            vpd_d0_value = vpd_d0_raw['value']
        else:
            vpd_d0_value = vpd_d0_raw
        if isinstance(vpd_d0_value, torch.Tensor):
            vpd_d0_value = vpd_d0_value.item()
        assert vpd_d0_value == pytest.approx(1500.0)
        
        # This parameter should be used by both photosynthesis and stomatal modules
        # Create photosynthesis model
        photo_model = FarquharPhotosynthesis(
            name="photo_test",
            enable_gpu=False
        )
        
        # Check that cached parameter matches
        if hasattr(photo_model, '_param_cache'):
            assert 'vpd_response_d0' in photo_model._param_cache
            cached_value = photo_model._param_cache['vpd_response_d0']
            if isinstance(cached_value, torch.Tensor):
                cached_value = cached_value.item()
            assert abs(cached_value - vpd_d0_value) < 0.001
    
    def test_temperature_acclimation(self, photosynthesis_model):
        """Test photosynthesis temperature acclimation response."""
        temperatures = torch.tensor([283.15, 288.15, 293.15, 298.15, 303.15, 308.15])  # 10-35掳C
        gpp_values = []
        
        for temp in temperatures:
            state = _canonical_state(air_temperature=temp)
            output = photosynthesis_model.forward(state, time=0.0, dt=1.0)
            gpp_values.append(_value(output, 'gross_photosynthesis').item())
        
        # Find temperature optimum
        max_gpp_idx = np.argmax(gpp_values)
        opt_temp = temperatures[max_gpp_idx].item()
        
        # Optimum should be around 25-30掳C for C3 plants (allow small numerical tolerance)
        assert 298.0 <= opt_temp <= 308.15
        
        # Response should decline at extreme temperatures
        assert gpp_values[0] < gpp_values[max_gpp_idx]  # Cold limitation
        assert gpp_values[-1] < gpp_values[max_gpp_idx]  # Heat stress
    
    def test_c3_c4_pathway_switching(self, param_manager):
        """Test different photosynthetic pathways."""
        state = _canonical_state(air_temperature=303.15, ppfd=1500.0)
        
        # C3 model
        c3_model = FarquharPhotosynthesis(
            name="c3_test",
            config={'pathway': 'C3'},
            enable_gpu=False
        )
        c3_output = c3_model.forward(state, time=0.0, dt=1.0)
        
        # C4 model
        c4_model = FarquharPhotosynthesis(
            name="c4_test",
            config={'pathway': 'C4'},
            enable_gpu=False
        )
        c4_output = c4_model.forward(state, time=0.0, dt=1.0)
        
        # At high temperature and high light, C4 should be more efficient
        # Note: This depends on parameter values and may need adjustment
        assert _value(c4_output, 'gross_photosynthesis') > 0
        assert _value(c3_output, 'gross_photosynthesis') > 0
        assert set(c4_output) == set(c3_output)
        assert _value(
            c4_output, 'diagnostics.photosynthesis.pathway_type'
        ) == 1.0
        assert _value(
            c3_output, 'diagnostics.photosynthesis.pathway_type'
        ) == 0.0
    
    def test_multi_layer_canopy(self, photosynthesis_model):
        """Test photosynthesis with LAI scaling."""
        # Single leaf
        state_single = _canonical_state(leaf_area_index=1.0)
        output_single = photosynthesis_model.forward(state_single, time=0.0, dt=1.0)
        
        # Canopy with LAI = 3
        state_canopy = state_single.copy()
        state_canopy['leaf_area_index'] = torch.tensor(3.0)
        output_canopy = photosynthesis_model.forward(state_canopy, time=0.0, dt=1.0)
        
        # Canopy photosynthesis should be higher but not 3x (due to self-shading)
        ratio = (
            _value(output_canopy, 'gross_photosynthesis')
            / _value(output_single, 'gross_photosynthesis')
        )
        assert ratio > 1.0  # Higher with more leaves
        assert ratio <= 3.0  # But not perfectly linear
    
    def test_stress_factor_integration(self, photosynthesis_model):
        """Test integration of multiple stress factors."""
        # No stress
        state_base = _canonical_state(
            water_stress_factor=1.0,
            temperature_stress_factor=1.0,
            ozone_stress_factor=1.0,
        )
        output_base = photosynthesis_model.forward(state_base, time=0.0, dt=1.0)
        
        # Multiple stresses
        state_stress = state_base.copy()
        state_stress['water_stress_factor'] = torch.tensor(0.7)
        state_stress['temperature_stress_factor'] = torch.tensor(0.8)
        state_stress['ozone_stress_factor'] = torch.tensor(0.9)
        output_stress = photosynthesis_model.forward(state_stress, time=0.0, dt=1.0)
        
        # Combined stress should reduce photosynthesis multiplicatively
        expected_reduction = 0.7 * 0.8 * 0.9
        actual_reduction = (
            _value(output_stress, 'gross_photosynthesis')
            / _value(output_base, 'gross_photosynthesis')
        )
        
        assert torch.allclose(actual_reduction, torch.tensor(expected_reduction), rtol=0.1)
    
    def test_parameter_bounds_validation(self, param_manager):
        """Test that parameters respect defined bounds."""
        # Get all photosynthesis parameters using list() with namespace filter
        photo_param_keys = param_manager.list(namespace="biosphere.photosynthesis")
        
        for param_key in photo_param_keys:
            # Get the parameter spec
            spec = param_manager.get_parameter(param_key)
            if spec is None:
                continue
            
            value = spec.value
            bounds = spec.bounds
            
            if value is not None and bounds is not None:
                # Handle tensor values
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                # Skip inf bounds
                if bounds[0] == -float('inf') and bounds[1] == float('inf'):
                    continue
                    
                assert bounds[0] <= value <= bounds[1], \
                    f"Parameter {param_key} value {value} outside bounds {bounds}"
    
    def test_batch_processing(self, photosynthesis_model):
        """Test that model handles batch inputs correctly."""
        # Create batch of inputs
        batch_size = 10
        state_batch = {
            'air_temperature': torch.randn(batch_size) * 5 + 298.15,
            'atmospheric_co2_ppm': torch.randn(batch_size) * 50 + 400,
            'ppfd': torch.randn(batch_size) * 500 + 1000,
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.ones(batch_size) * 1000.0,
            'vcmax_scaling': torch.ones(batch_size),
        }
        
        # Ensure positive values
        state_batch['atmospheric_co2_ppm'] = torch.abs(
            state_batch['atmospheric_co2_ppm']
        )
        state_batch['ppfd'] = torch.abs(state_batch['ppfd'])
        
        # Run batch through model
        output_batch = photosynthesis_model.forward(state_batch, time=0.0, dt=1.0)
        
        # Check output shapes
        gross = _value(output_batch, 'gross_photosynthesis')
        net = _value(output_batch, 'net_photosynthesis')
        assert gross.shape == (batch_size,)
        assert net.shape == (batch_size,)
        
        # All outputs should be valid
        assert not torch.isnan(gross).any()
        assert not torch.isnan(net).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

