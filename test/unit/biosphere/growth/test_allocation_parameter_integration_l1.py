"""
L1 Unit Tests for Growth Allocation Parameter Integration (P0.3-P0.5 Verification)

This test module verifies that all P0.3, P0.4, and P0.5 parameter fixes are correctly
integrated with the biomass allocation, structural growth, and phenology growth modules.

Test Coverage:
- P0.3: Growth allocation default naming (default_environmental_modifier, etc.)
- Structural growth diagnostic-only state-port contract
- P0.5: Phenology control thresholds (phenology_clamp_min/max, growth_active_threshold, etc.)

References:
- Gao et al. (2025). pnetr: An R package for the PnET model family
- Zhang et al. (2011). CASA-CNP model documentation
- Richardson et al. (2013). Climate change, phenology, and plant productivity
"""

import pytest
import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from relic.parameters import get_global_manager
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from test.utils.forward_call import call_forward


@pytest.fixture(scope="module")
def param_manager():
    """Get global parameter manager instance."""
    return get_global_manager()


# =============================================================================
# P0.3 Tests: Growth Allocation Default Naming
# =============================================================================

class TestP03AllocationDefaults:
    """Test P0.3 fixes: default_environmental_modifier, default_phenological_modifier, etc."""
    
    def test_default_environmental_modifier_exists(self, param_manager):
        """Verify default_environmental_modifier parameter exists."""
        value = param_manager.get(
            "biosphere.growth.allocation.defaults.default_environmental_modifier"
        )
        assert value is not None, "default_environmental_modifier parameter not found"
        assert value == 1.0, f"Expected 1.0, got {value}"
    
    def test_default_phenological_modifier_exists(self, param_manager):
        """Verify default_phenological_modifier parameter exists."""
        value = param_manager.get(
            "biosphere.growth.allocation.defaults.default_phenological_modifier"
        )
        assert value is not None, "default_phenological_modifier parameter not found"
        assert value == 1.0, f"Expected 1.0, got {value}"
    
    def test_second_construction_cost_is_absent(self, param_manager):
        """BiomassAllocation must not own a second construction-respiration law."""
        assert param_manager.get_parameter(
            "biosphere.growth.allocation.defaults.default_construction_cost"
        ) is None
    
    def test_conservation_tolerance_exists(self, param_manager):
        """Verify validation.conservation.conservation_tolerance exists."""
        value = param_manager.get("validation.conservation.conservation_tolerance")
        assert value is not None, "conservation_tolerance parameter not found"
        # Accept both float and string representations of 1e-6
        if isinstance(value, str):
            assert value == '1e-6' or float(value) == 1e-6, f"Expected 1e-6, got {value}"
        else:
            assert float(value) == 1e-6, f"Expected 1e-6, got {value}"
    
    def test_strict_conservation_exists(self, param_manager):
        """Verify validation.conservation.strict_conservation exists."""
        value = param_manager.get("validation.conservation.strict_conservation")
        assert value is not None, "strict_conservation parameter not found"
        assert value == False, f"Expected False, got {value}"
    
    def test_all_defaults_have_correct_units(self, param_manager):
        """Verify all P0.3 parameters have correct units."""
        params_units = {
            "biosphere.growth.allocation.defaults.default_environmental_modifier": "dimensionless",
            "biosphere.growth.allocation.defaults.default_phenological_modifier": "dimensionless",
            "validation.conservation.conservation_tolerance": "dimensionless",
            "validation.conservation.strict_conservation": "dimensionless",
        }
        
        for param_key, expected_units in params_units.items():
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Parameter spec not found for {param_key}"
            actual_units = spec.units
            # Note: dimensionless for bool is acceptable
            if spec.dtype == "bool":
                continue
            assert actual_units == expected_units, \
                f"{param_key}: expected units '{expected_units}', got '{actual_units}'"


# =============================================================================
# Structural Growth Port Contract
# =============================================================================

class TestP04StructuralGrowthDefaults:
    """Test explicit structural-growth interconnection ports."""

    def test_removed_state_defaults_are_absent(self, param_manager):
        """State variables must not be duplicated in the parameter database."""
        for key in (
            "biosphere.growth.allocation.defaults.default_growth_rate_leaf",
            "biosphere.growth.allocation.defaults.default_current_lai",
        ):
            assert param_manager.get_parameter(key) is None

    def test_structural_growth_requires_explicit_state(self):
        """Missing leaf growth and the canonical LAI state fail fast."""
        functor = StructuralGrowth(name="test_structural_growth")
        with pytest.raises(KeyError, match="growth_rate_leaf"):
            call_forward(functor, {})

    def test_structural_growth_consumes_declared_ports(self):
        """Valid explicit ports produce namespaced diagnostic tendencies only."""
        functor = StructuralGrowth(name="test_structural_growth")
        result = functor.forward(
            {
                "growth_rate_leaf": torch.tensor(1.0e-7),
                "leaf_area_index": torch.tensor(1.0),
            },
            time=0.0,
            dt=3600.0,
        )
        assert "rate" not in result
        assert "leaf_area_index" not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())


# =============================================================================
# P0.5 Tests: Phenology Control Thresholds
# =============================================================================

class TestP05PhenologyControlThresholds:
    """Test P0.5 fixes: phenology_clamp_min/max, growth_active_threshold, etc."""
    
    def test_phenology_clamp_min_exists(self, param_manager):
        """Verify phenology_clamp_min parameter exists."""
        value = param_manager.get("biosphere.growth.allocation.phenology_clamp_min")
        assert value is not None, "phenology_clamp_min parameter not found"
        assert value == 0.0, f"Expected 0.0, got {value}"
    
    def test_phenology_clamp_max_exists(self, param_manager):
        """Verify phenology_clamp_max parameter exists."""
        value = param_manager.get("biosphere.growth.allocation.phenology_clamp_max")
        assert value is not None, "phenology_clamp_max parameter not found"
        assert value == 1.0, f"Expected 1.0, got {value}"
    
    def test_growth_active_threshold_exists(self, param_manager):
        """Verify growth_active_threshold parameter exists."""
        value = param_manager.get("biosphere.growth.allocation.growth_active_threshold")
        assert value is not None, "growth_active_threshold parameter not found"
        assert value == 0.1, f"Expected 0.1, got {value}"
    
    def test_temperature_range_tolerance_exists(self, param_manager):
        """Verify temperature_range_tolerance parameter exists."""
        value = param_manager.get("biosphere.growth.defaults.temperature_range_tolerance")
        assert value is not None, "temperature_range_tolerance parameter not found"
        assert value == 10.0, f"Expected 10.0, got {value}"
    
    def test_phenology_thresholds_have_correct_units(self, param_manager):
        """Verify P0.5 parameters have correct units."""
        params_units = {
            "biosphere.growth.allocation.phenology_clamp_min": "dimensionless",
            "biosphere.growth.allocation.phenology_clamp_max": "dimensionless",
            "biosphere.growth.allocation.growth_active_threshold": "dimensionless",
            "biosphere.growth.defaults.temperature_range_tolerance": "K",
        }
        
        for param_key, expected_units in params_units.items():
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Parameter spec not found for {param_key}"
            actual_units = spec.units
            assert actual_units == expected_units, \
                f"{param_key}: expected units '{expected_units}', got '{actual_units}'"


# =============================================================================
# Integration Tests: BiomassAllocation with P0.3-P0.5 Parameters
# =============================================================================

class TestBiomassAllocationParameterIntegration:
    """Test that BiomassAllocation correctly uses all P0.3-P0.5 parameters."""
    
    def test_biomass_allocation_initialization(self):
        """Verify BiomassAllocation initializes without errors."""
        functor = BiomassAllocation(
            name="test_biomass_allocation",
            enable_environmental_control=False,
            enable_phenology_control=False,
            enable_allometry_control=False
        )
        assert functor is not None
        assert functor.name == "test_biomass_allocation"
    
    def test_forward_uses_default_environmental_modifier(self):
        """Verify forward() uses default_environmental_modifier when controls disabled."""
        functor = BiomassAllocation(
            name="test_allocation",
            enable_environmental_control=False,
            enable_phenology_control=False
        )
        
        state = {
            'net_primary_productivity': torch.tensor(5.0e-7)
        }
        
        output = functor.forward(state, time=0.0, dt=86400.0)
        
        key = 'diagnostics.biomass_allocation.environmental_modifier'
        assert key in output
        env_mod = output[key]
        assert isinstance(env_mod, UpdateItem)
        
        # Should be default value (1.0) when environmental control is disabled
        expected_default = 1.0
        env_mod_value = env_mod.value.item()
        
        assert abs(env_mod_value - expected_default) < 1e-6, \
            f"Expected default environmental modifier {expected_default}, got {env_mod_value}"
    
    def test_forward_uses_default_phenological_modifier(self):
        """Verify forward() uses default_phenological_modifier when controls disabled."""
        functor = BiomassAllocation(
            name="test_allocation",
            enable_environmental_control=False,
            enable_phenology_control=False
        )
        
        state = {
            'net_primary_productivity': torch.tensor(5.0e-7)
        }
        
        output = functor.forward(state, time=0.0, dt=86400.0)
        
        key = 'diagnostics.biomass_allocation.phenology_modifier'
        assert key in output
        pheno_mod = output[key]
        assert isinstance(pheno_mod, UpdateItem)
        
        # Should be default value (1.0) when phenology control is disabled
        expected_default = 1.0
        pheno_mod_value = pheno_mod.value.item()
        
        assert abs(pheno_mod_value - expected_default) < 1e-6, \
            f"Expected default phenological modifier {expected_default}, got {pheno_mod_value}"
    
    def test_conservation_check_uses_validation_parameters(self):
        """Verify conservation check uses validation.conservation parameters."""
        functor = BiomassAllocation(
            name="test_allocation",
            enable_environmental_control=False,
            enable_phenology_control=False
        )
        
        state = {
            'net_primary_productivity': torch.tensor(1.0e-6)
        }
        
        # Should not raise error with valid inputs
        output = functor.forward(state, time=0.0, dt=86400.0)
        
        # Should complete successfully
        total_key = 'diagnostics.biomass_allocation.total_growth_rate'
        sum_key = 'diagnostics.biomass_allocation.allocation_sum'
        residual_key = 'diagnostics.biomass_allocation.carbon_balance_residual'
        assert total_key in output
        assert sum_key in output
        assert residual_key in output
        
        # Sum should be close to 1.0
        alloc_sum_value = output[sum_key].value.item()
        
        assert abs(alloc_sum_value - 1.0) < 0.01, \
            f"Allocation fractions should sum to 1.0, got {alloc_sum_value}"
        assert output[residual_key].value.item() == pytest.approx(0.0, abs=1e-12)
        for state_key in ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'):
            assert output[state_key].mode == UpdateMode.ACCUMULATE
            assert output[state_key].per_timestep is True


# =============================================================================
# Parameter Consistency Tests
# =============================================================================

class TestParameterConsistency:
    """Test that all P0.3-P0.5 parameters are internally consistent."""
    
    def test_no_duplicate_env_modifier_params(self, param_manager):
        """Verify redundant env_modifier parameter was removed."""
        # Should NOT exist (ParameterManager.get returns None for missing params)
        value = param_manager.get("biosphere.growth.allocation.defaults.env_modifier")
        assert value is None, "Redundant parameter 'env_modifier' should not exist"
        
        # Correct parameter should exist
        correct_value = param_manager.get("biosphere.growth.allocation.defaults.default_environmental_modifier")
        assert correct_value is not None, "default_environmental_modifier should exist"
    
    def test_no_duplicate_pheno_modifier_params(self, param_manager):
        """Verify redundant pheno_modifier parameter was removed."""
        # Should NOT exist (ParameterManager.get returns None for missing params)
        value = param_manager.get("biosphere.growth.allocation.defaults.pheno_modifier")
        assert value is None, "Redundant parameter 'pheno_modifier' should not exist"
        
        # Correct parameter should exist
        correct_value = param_manager.get("biosphere.growth.allocation.defaults.default_phenological_modifier")
        assert correct_value is not None, "default_phenological_modifier should exist"
    
    def test_conservation_tolerance_units_consistency(self, param_manager):
        """Verify conservation_tolerance parameters have appropriate units."""
        # biosphere.growth.allocation.conservation_tolerance should be dimensionless
        alloc_spec = param_manager.get_parameter("biosphere.growth.allocation.conservation_tolerance")
        assert alloc_spec is not None
        assert alloc_spec.units == "dimensionless", \
            f"Expected dimensionless, got {alloc_spec.units}"
        
        # validation.conservation.conservation_tolerance should be dimensionless
        valid_spec = param_manager.get_parameter("validation.conservation.conservation_tolerance")
        assert valid_spec is not None
        assert valid_spec.units == "dimensionless", \
            f"Expected 'dimensionless', got {valid_spec.units}"
    
    def test_default_lai_is_phenology_only(self, param_manager):
        """The remaining default_lai is not a StructuralGrowth state fallback."""
        default_lai = param_manager.get("biosphere.growth.allocation.defaults.default_lai")
        assert default_lai == 3.0, "default_lai should be 3.0"
        assert StructuralGrowth().get_required_inputs() == [
            "growth_rate_leaf",
            "leaf_area_index",
        ]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

