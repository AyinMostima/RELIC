"""
Test Suite for Growth Module Parameter Integration

This test file verifies that the growth module correctly uses the ParameterManager
and that all parameters are properly loaded from the database.
"""

import pytest
import torch
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.parameters import get_global_manager
from relic.core.state import UpdateMode


class TestBiomassAllocationParameterIntegration:
    """Test BiomassAllocation parameter integration."""
    
    @pytest.fixture
    def allocation_functor(self):
        """Create a BiomassAllocation functor for testing."""
        return BiomassAllocation(
            name="test_biomass_allocation",
            enable_environmental_control=True,
            enable_phenology_control=True
        )
    
    def test_parameter_loading_from_manager(self, allocation_functor):
        """Test that BiomassAllocation Functor can be instantiated with ParameterManager."""
        assert allocation_functor is not None
        assert allocation_functor.name == "test_biomass_allocation"

        manager = get_global_manager()
        spec = manager.get_parameter('biosphere.growth.allocation.f_leaf_base')
        assert spec is not None
        value = allocation_functor.get_parameter_value(
            'f_leaf_base',
            namespace='biosphere.growth.allocation',
            required=True,
        )
        scalar = value.detach().item() if isinstance(value, torch.Tensor) else value
        assert scalar == pytest.approx(spec.value)
    
    def test_no_hardcoded_parameters(self, allocation_functor):
        """Ensure BiomassAllocation uses ParameterManager (not hardcoded values)."""
        assert hasattr(allocation_functor, 'get_parameter_value')
        manager = get_global_manager()
        for key in (
            'biosphere.growth.allocation.normalization_epsilon',
            'biosphere.growth.allocation.allometry_biomass_floor',
            'biosphere.growth.allocation.phenology.mature_phase_modifier',
            'validation.conservation.conservation_tolerance',
        ):
            spec = manager.get_parameter(key)
            assert spec is not None
            assert spec.units
            assert spec.bounds is not None
            assert spec.references
            assert spec.provenance
    
    def test_carbon_conservation(self, allocation_functor):
        """Test carbon conservation in allocation."""
        # Create test state
        npp = torch.tensor(1.0e-6)
        state = {
            'net_primary_productivity': npp,
            'air_temperature': torch.tensor(298.15),
            'leaf_area_index': torch.tensor(4.0),
            'phenology_phase': torch.tensor(2),  # Mature phase
            'water_stress_factor': torch.tensor(1.0),
        }

        result = allocation_functor.forward(state, time=0.0, dt=3600.0)

        total_net_growth = (
            result['growth_rate_leaf'].value
            + result['growth_rate_stem'].value
            + result['growth_rate_root'].value
        )
        expected = npp.to(dtype=total_net_growth.dtype, device=total_net_growth.device)
        residual = result[
            'diagnostics.biomass_allocation.carbon_balance_residual'
        ].value
        assert torch.allclose(total_net_growth, expected, atol=1.0e-12)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-12)
        assert 'construction_respiration_leaf' not in result
        for key in ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'):
            assert result[key].mode is UpdateMode.ACCUMULATE
            assert result[key].per_timestep is True
    
    def test_allocation_fractions_sum_to_one(self, allocation_functor):
        """Test that allocation fractions always sum to 1."""
        state = {
            'net_primary_productivity': torch.tensor(1.0e-6),
            'air_temperature': torch.tensor(298.15),
            'leaf_area_index': torch.tensor(4.0),
            'phenology_phase': torch.tensor(2),
            'water_stress_factor': torch.tensor(1.0),
        }

        result = allocation_functor.forward(state, time=0.0, dt=3600.0)

        total_fractions = (
            result['diagnostics.biomass_allocation.allocation_fraction_leaf'].value
            + result['diagnostics.biomass_allocation.allocation_fraction_stem'].value
            + result['diagnostics.biomass_allocation.allocation_fraction_root'].value
        )

        assert torch.allclose(
            total_fractions,
            torch.ones_like(total_fractions),
            atol=1.0e-7,
        )


class TestPhenologyGrowthParameterIntegration:
    """Test PhenologyGrowth parameter integration."""
    
    @pytest.fixture
    def phenology_functor(self):
        """Create a PhenologyGrowth functor for testing."""
        return PhenologyGrowth(phenology_model_owner="gsi_phenology")
    
    def test_phenology_parameters_loaded(self, phenology_functor):
        """Test that PhenologyGrowth Functor can be instantiated."""
        # Verify functor is properly initialized
        assert phenology_functor is not None
        assert phenology_functor.name == "phenology_growth"
        
        assert hasattr(phenology_functor, 'get_parameter_value')
        manager = get_global_manager()
        threshold = manager.get_parameter(
            'biosphere.growth.allocation.growth_active_threshold'
        )
        assert threshold is not None
        assert threshold.units == 'dimensionless'
        assert threshold.bounds is not None
        assert phenology_functor.growth_active_threshold == pytest.approx(
            threshold.value
        )
        assert phenology_functor.metadata.parameters[
            'growth_active_threshold'
        ]['namespace'] == 'biosphere.growth.allocation'
        assert phenology_functor.get_dependencies() == ['gsi_phenology']


class TestStructuralGrowthParameterIntegration:
    """Test StructuralGrowth parameter integration."""
    
    @pytest.fixture
    def structural_functor(self):
        """Create a StructuralGrowth functor for testing."""
        return StructuralGrowth(name="test_structural_growth")
    
    def test_structural_parameters_loaded(self, structural_functor):
        """Test that StructuralGrowth Functor can be instantiated."""
        # Verify functor is properly initialized
        assert structural_functor is not None
        assert structural_functor.name == "test_structural_growth"
        
        assert hasattr(structural_functor, 'get_parameter_value')
        manager = get_global_manager()
        sla = manager.get_parameter('biosphere.phenology.morphology.sla')
        turnover = manager.get_parameter(
            'biosphere.phenology.turnover.leaf_turnover_rate'
        )
        assert sla is not None and sla.units == 'm2 kg-1 C'
        assert turnover is not None and turnover.units == 's-1'
        assert structural_functor.metadata.parameters['sla']['namespace'] == (
            'biosphere.phenology.morphology'
        )


class TestParameterNamespaceConsistency:
    """Test that parameters use consistent namespaces."""
    
    def test_namespace_consistency(self):
        """Test that ParameterManager uses consistent namespace pattern."""
        param_manager = get_global_manager()
        
        assert param_manager is not None
        assert param_manager.get_parameter(
            'biosphere.growth.allocation.f_leaf_base'
        ) is not None
        assert param_manager.get_parameter(
            'biosphere.phenology.morphology.sla'
        ) is not None
        assert param_manager.get_parameter(
            'biosphere.growth.test_parameter'
        ) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

