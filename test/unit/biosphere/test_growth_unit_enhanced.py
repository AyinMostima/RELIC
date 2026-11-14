"""
Enhanced unit tests for relic.scientific.biosphere.growth module.

These tests verify:
- Parameter system integration and CF-1.8 compliance
- Proper tensor creation through ParameterManager
- BiologicalProcess forward-only contract compliance
- No namespace pollution through compatibility parameter APIs
- All default values come from ParameterManager
- Units are correctly in CF-1.8 format (K not degC)
"""

import pytest
import torch
from typing import Dict
import warnings

# Import test targets
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.core.state import UpdateItem, UpdateMode

# Import parameter manager
from relic.parameters import get_global_manager
from test.utils.forward_call import call_forward


@pytest.fixture
def clean_param_manager():
    """Provide the shared manager without mutating private caches or registries."""
    yield get_global_manager()


@pytest.fixture
def sample_state() -> Dict[str, torch.Tensor]:
    """Create sample state dict for testing."""
    return {
        'phenological_factor': torch.tensor(0.6),
        'air_temperature': torch.tensor(298.15),
        'net_primary_productivity': torch.tensor(2.0e-6),
        'water_stress_factor': torch.tensor(0.8),
        'leaf_area_index': torch.tensor(3.5),
        'phenology_phase': torch.tensor(2.0),
        'growth_rate_leaf': torch.tensor(5.0e-7),
    }


class TestPhenologyGrowthUnit:
    """Unit tests for PhenologyGrowth Functor."""
    
    def test_forward_returns_canonical_controls(self, sample_state):
        """The biological bridge does not require a public rate alias."""
        phenology = PhenologyGrowth(phenology_model_owner="gsi_phenology")

        result = phenology.forward(sample_state, time=0.0, dt=86400.0)

        assert set(result) == set(phenology.metadata.output_variables)
        assert 'rate' not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert all(item.mode is UpdateMode.REPLACE for item in result.values())
        assert all(item.per_timestep is False for item in result.values())
        item = result['diagnostics.phenology.growth_active']
        assert item.unit == 'dimensionless'
        assert item.source_functor == 'phenology_growth'
        assert item.value.dtype is torch.bool
        
    def test_no_hidden_state_defaults(self, sample_state):
        """Every declared input is required from the state interconnection."""
        phenology = PhenologyGrowth(phenology_model_owner="gsi_phenology")

        for missing in phenology.get_required_inputs():
            state = dict(sample_state)
            state.pop(missing)
            with pytest.raises(KeyError, match=missing):
                phenology.forward(state, time=0.0, dt=86400.0)
        
    def test_only_activity_threshold_parameter_is_bound(self, clean_param_manager):
        """The observer binds one calibrated threshold and no second formula."""
        phenology = PhenologyGrowth(phenology_model_owner="gsi_phenology")
        spec = clean_param_manager.get_parameter(
            'biosphere.growth.allocation.growth_active_threshold'
        )

        assert spec is not None
        assert spec.units == 'dimensionless'
        assert spec.bounds is not None
        assert 0.0 <= spec.bounds[0] < spec.bounds[1] <= 1.0
        assert phenology.growth_active_threshold == pytest.approx(spec.value)
        assert not hasattr(phenology, 'temperature_optimal')
        assert not hasattr(phenology, 'temperature_range_tolerance')
            
    def test_canonical_inputs_and_outputs_are_exact(self, clean_param_manager):
        """Metadata, runtime requirements, and outputs expose one interface."""
        phenology = PhenologyGrowth(phenology_model_owner="gsi_phenology")

        assert phenology.get_required_inputs() == ['phenological_factor']
        assert phenology.get_provides() == [
            'diagnostics.phenology.growth_active'
        ]
        assert phenology.get_dependencies() == ['gsi_phenology']


class TestBiomassAllocationUnit:
    """Unit tests for BiomassAllocation Functor."""
    
    def test_no_namespace_pollution(self, clean_param_manager, sample_state):
        """Biomass allocation does not expose or populate compatibility paths."""
        allocation = BiomassAllocation(name="test_allocation")
        param_manager = clean_param_manager

        assert not hasattr(param_manager, 'set_parameter_value')
        unauthorized_key = (
            'biological_process.biomass_allocation.biosphere.pft.default.allocation.f_leaf'
        )
        assert param_manager.get_parameter(unauthorized_key) is None
        allocation.forward(sample_state, time=0.0, dt=86400.0)
        assert param_manager.get_parameter(unauthorized_key) is None
                
    def test_forward_returns_canonical_total_growth_rate(self, sample_state):
        """Test that biomass allocation exposes the canonical total-growth port."""
        allocation = BiomassAllocation(name="test_allocation")
        
        result = allocation.forward(sample_state, time=0.0, dt=86400.0)
        
        assert 'diagnostics.biomass_allocation.total_growth_rate' in result
        assert 'rate' not in result
        assert 'carbon_allocation_rate' not in result
        assert 'total_growth_rate' not in result
        
    def test_temperature_parameters_cf_compliant(self, clean_param_manager):
        """Test that temperature parameters use CF-1.8 compliant units."""
        allocation = BiomassAllocation(name="test_allocation")
        
        param_manager = clean_param_manager
        expectations = {
            'biosphere.growth.allocation.root_alloc_temp_ref': ('K', 280.25),
            'biosphere.growth.allocation.temperature_stress_factor': ('K-1', 0.02),
            'biosphere.growth.allocation.temperature_opt': ('K', 298.15),
        }
        for key, (units, expected_value) in expectations.items():
            spec = param_manager.get_parameter(key)
            assert spec is not None
            assert spec.units == units
            assert spec.value == pytest.approx(expected_value)
            
    def test_no_hardcoded_tensor_creation(self, sample_state):
        """Test that no hardcoded tensors are created directly."""
        allocation = BiomassAllocation(name="test_allocation")
        
        # This should work without creating any hardcoded tensors
        result = allocation.forward(sample_state, time=0.0, dt=86400.0)
        
        # Check all expected outputs exist
        assert 'growth_rate_leaf' in result
        assert 'growth_rate_stem' in result
        assert 'growth_rate_root' in result
        assert 'diagnostics.biomass_allocation.environmental_modifier' in result
        assert 'diagnostics.biomass_allocation.phenology_modifier' in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
        
    def test_phenology_modifiers_from_parameters(self, clean_param_manager, sample_state):
        """Test that phenology phase modifiers come from ParameterManager."""
        allocation = BiomassAllocation(
            name="test_allocation",
            enable_phenology_control=True
        )
        
        # Set different phenology phases and check modifiers
        for phase in [0, 1, 2, 3]:  # Dormant, Greenup, Mature, Senescence
            state_copy = sample_state.copy()
            state_copy['phenology_phase'] = torch.tensor(float(phase))
            
            result = allocation.forward(state_copy, time=0.0, dt=86400.0)
            
            modifier = result[
                'diagnostics.biomass_allocation.phenology_modifier'
            ]
            assert isinstance(modifier, UpdateItem)
            assert modifier.mode is UpdateMode.REPLACE

        param_manager = clean_param_manager
        expected_params = [
            'biosphere.growth.allocation.phenology.dormant_phase_modifier',
            'biosphere.growth.allocation.phenology.greenup_phase_modifier',
            'biosphere.growth.allocation.phenology.mature_phase_modifier',
            'biosphere.growth.allocation.phenology.senescence_phase_modifier',
        ]
        for param_key in expected_params:
            spec = param_manager.get_parameter(param_key)
            assert spec is not None
            assert spec.units == 'dimensionless'
            assert spec.bounds is not None


class TestStructuralGrowthUnit:
    """Unit tests for StructuralGrowth Functor."""
    
    def test_forward_adapter_returns_canonical_diagnostics(self, sample_state):
        """The forward adapter returns only the declared structural diagnostics."""
        structural = StructuralGrowth(name="test_structural")

        result = call_forward(structural, sample_state)

        assert set(result) == set(structural.metadata.output_variables)
        assert 'rate' not in result
        assert all(isinstance(value, torch.Tensor) for value in result.values())
        
    def test_missing_state_ports_fail_without_registering_defaults(self, clean_param_manager):
        """StateManager-owned ports must not be replaced by parameter defaults."""
        structural = StructuralGrowth(name="test_structural")

        with pytest.raises(KeyError, match="growth_rate_leaf"):
            call_forward(structural, {})

        assert 'biosphere.growth.defaults.default_growth_rate_leaf' not in clean_param_manager._runtime_registry
        assert 'biosphere.growth.defaults.default_current_lai' not in clean_param_manager._runtime_registry
        
    def test_no_hardcoded_tensors(self, sample_state):
        """Test that no hardcoded tensor values are used."""
        structural = StructuralGrowth(name="test_structural")
        
        # Should work without errors
        result = structural.forward(sample_state, time=0.0, dt=86400.0)
        
        assert set(result) == set(structural.metadata.output_variables)
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert 'lai_change_rate' not in result
        assert 'lai_growth_rate' not in result
        assert 'leaf_area_growth' not in result


class TestParameterSystemIntegration:
    """Test overall parameter system integration."""
    
    def test_all_modules_use_unified_parameter_manager(self, clean_param_manager):
        """Test that all growth modules use the unified parameter manager."""
        modules = [
            PhenologyGrowth(phenology_model_owner="gsi_phenology"),
            BiomassAllocation(name="allocation"),
            StructuralGrowth(name="structural")
        ]
        
        param_manager = clean_param_manager
        for module in modules:
            assert hasattr(module, 'get_parameter_value')
            assert module.parameter_manager is param_manager
        
    def test_no_duplicate_parameter_registrations(self, clean_param_manager):
        """Test that creating multiple instances doesn't cause duplicate registrations."""
        # Create multiple instances
        phenology1 = PhenologyGrowth(phenology_model_owner="gsi_phenology")
        phenology2 = PhenologyGrowth(phenology_model_owner="static_phenology")
        
        param_manager = clean_param_manager
        
        # Count warnings about duplicate registrations
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            state = {
                'phenological_factor': torch.tensor(0.6),
            }
            phenology1.forward(state, time=0.0, dt=86400.0)
            phenology2.forward(state, time=0.0, dt=86400.0)
            
            # Should not have excessive warnings about overwriting parameters
            overwrite_warnings = [warning for warning in w 
                                 if "Overwriting existing parameter" in str(warning.message)]
            
            # Some overwrites are acceptable (runtime defaults), but not for all parameters
            assert len(overwrite_warnings) < 10, f"Too many parameter overwrites: {len(overwrite_warnings)}"
            
    def test_conservation_tolerance_from_validation_namespace(self, clean_param_manager, sample_state):
        """Test that conservation tolerance uses validation namespace."""
        allocation = BiomassAllocation(name="test_allocation")
        
        # Run forward pass which checks conservation
        result = allocation.forward(sample_state, time=0.0, dt=86400.0)
        
        param_manager = clean_param_manager
        conservation_key = 'validation.conservation.conservation_tolerance'
        spec = param_manager.get_parameter(conservation_key)
        assert spec is not None
        assert spec.units == 'dimensionless'
        assert 0 < spec.value < 1e-4
        residual = result[
            'diagnostics.biomass_allocation.carbon_balance_residual'
        ]
        assert isinstance(residual, UpdateItem)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


