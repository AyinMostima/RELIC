"""
Unit tests for relic.scientific.biosphere.growth module.

Tests for BiomassAllocation, PhenologyGrowth, and StructuralGrowth Functors.
Tests verify parameter system integration, forward pass correctness, and
architectural compliance with FunctorBase and BiologicalProcessFunctor.
"""

import pytest
import torch
# Import test targets
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.core.state import UpdateItem, UpdateMode

# Import parameter manager
from relic.parameters import get_global_manager
from test.utils.forward_call import call_forward


class TestBiomassAllocationUnit:
    """Unit tests for BiomassAllocation Functor."""
    
    def test_initialization(self):
        """Test BiomassAllocation can be initialized without errors."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=torch.device("cpu")
        )
        
        # Verify basic attributes
        assert allocation.name == "test_biomass_allocation"
        assert allocation.device.type == "cpu"
        assert hasattr(allocation, 'f_leaf_base')
        assert hasattr(allocation, 'f_stem_base')
        assert hasattr(allocation, 'f_root_base')
    
    def test_parameter_namespace_allocation(self):
        """Test allocation parameters use correct namespaces."""
        BiomassAllocation()
        param_manager = get_global_manager()
        allocation_params = [
            'biosphere.growth.allocation.f_leaf_base',
            'biosphere.growth.allocation.f_stem_base',
            'biosphere.growth.allocation.f_root_base',
            'biosphere.growth.allocation.root_alloc_temp_coeff',
        ]
        for param_name in allocation_params:
            spec = param_manager.get_parameter(param_name)
            assert spec is not None
            assert spec.units
            assert spec.bounds is not None
    
    def test_forward_pass_basic(self):
        """Test forward pass with basic inputs."""
        allocation = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False
        )
        
        state = {
            'net_primary_productivity': torch.tensor(1.0e-6),
        }

        dt = 3600.0
        result = allocation.forward(state, time=0, dt=dt)

        rates = [
            result[name].value
            for name in ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')
        ]
        total = result['diagnostics.biomass_allocation.total_growth_rate'].value
        fraction_sum = result['diagnostics.biomass_allocation.allocation_sum'].value
        npp = state['net_primary_productivity'].to(
            device=total.device,
            dtype=total.dtype,
        )
        assert torch.allclose(total, sum(rates), atol=1.0e-12)
        assert torch.allclose(total, npp, atol=1.0e-12)
        assert torch.allclose(fraction_sum, fraction_sum.new_tensor(1.0), atol=1.0e-7)

        for state_name, rate in zip(
            ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'), rates
        ):
            update = result[state_name]
            assert update.mode is UpdateMode.ACCUMULATE
            assert update.per_timestep is True
            assert torch.allclose(update.value, rate * dt, atol=1.0e-9)
    
    def test_forward_with_environmental_control(self):
        """Test forward pass with environmental controls enabled."""
        allocation = BiomassAllocation(
            enable_environmental_control=True,
            enable_phenology_control=False
        )
        
        state = {
            'net_primary_productivity': torch.tensor(1.0e-6),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(0.5),
        }

        result = allocation.forward(state, time=0, dt=86400)

        env_mod = result[
            'diagnostics.biomass_allocation.environmental_modifier'
        ].value
        assert torch.isfinite(env_mod)
        assert 0 < env_mod <= 2.0
    
    def test_forward_with_phenology_control(self):
        """Test forward pass with phenological controls enabled."""
        allocation = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=True
        )
        
        # Test greenup phase (high leaf allocation)
        state_greenup = {
            'net_primary_productivity': torch.tensor(1.0e-6),
            'leaf_area_index': torch.tensor(2.0),
            'phenology_phase': torch.tensor(1.0)  # GREENUP_PHASE
        }
        
        result_greenup = allocation.forward(state_greenup, time=0, dt=86400)
        
        # During greenup, leaf allocation should be high
        f_leaf = result_greenup[
            'diagnostics.biomass_allocation.allocation_fraction_leaf'
        ].value
        assert f_leaf > 0.5  # Should be around 0.8 during greenup
        
        # Test dormant phase (no leaf allocation)
        state_dormant = {
            'net_primary_productivity': torch.tensor(1.0e-6),
            'leaf_area_index': torch.tensor(2.0),
            'phenology_phase': torch.tensor(0.0)  # DORMANT_PHASE
        }
        
        result_dormant = allocation.forward(state_dormant, time=0, dt=86400)
        
        # During dormancy, leaf allocation should be zero
        f_leaf_dormant = result_dormant[
            'diagnostics.biomass_allocation.allocation_fraction_leaf'
        ].value
        assert f_leaf_dormant < 0.01

    def test_get_dependencies_returns_canonical_owners(self):
        """NPP, water stress, and phenology use their canonical owners."""
        allocation = BiomassAllocation()
        deps = allocation.get_dependencies()
        assert deps == ['autotrophic_respiration', 'water_stress', 'gsi_phenology']
    
    def test_get_required_inputs(self):
        """Test get_required_inputs returns correct state variables."""
        allocation = BiomassAllocation(
            enable_environmental_control=True,
            enable_phenology_control=True
        )
        
        inputs = allocation.get_required_inputs()
        assert inputs == [
            'net_primary_productivity', 'air_temperature',
            'water_stress_factor', 'leaf_area_index', 'phenology_phase',
        ]
    
    def test_carbon_conservation(self):
        """Test carbon conservation is maintained."""
        allocation = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
        )

        npp_value = 1.0e-6
        state = {
            'net_primary_productivity': torch.tensor(npp_value),
        }

        result = allocation.forward(state, time=0, dt=86400)

        total_allocated = result[
            'diagnostics.biomass_allocation.total_growth_rate'
        ].value
        residual = result[
            'diagnostics.biomass_allocation.carbon_balance_residual'
        ].value
        expected = torch.as_tensor(
            npp_value, dtype=total_allocated.dtype, device=total_allocated.device
        )
        assert torch.isclose(total_allocated, expected, atol=1e-12)
        assert torch.isclose(residual, torch.zeros_like(residual), atol=1e-12)
        assert 'total_construction_respiration' not in result


class TestPhenologyGrowthUnit:
    """Unit tests for PhenologyGrowth Functor."""

    @staticmethod
    def _functor(**kwargs):
        kwargs.setdefault('phenology_model_owner', 'gsi_phenology')
        return PhenologyGrowth(**kwargs)
    
    def test_initialization(self):
        """Test the canonical selected-owner observer initializes."""
        pheno = self._functor(
            device=torch.device("cpu")
        )
        
        assert pheno.name == "phenology_growth"
        assert pheno.device.type == "cpu"
        assert pheno.get_dependencies() == ['gsi_phenology']
        assert pheno.metadata.input_variables == ['phenological_factor']
        assert pheno.metadata.output_variables == [
            'diagnostics.phenology.growth_active'
        ]
    
    def test_forward_pass(self):
        """Test forward pass returns correct structure."""
        pheno = self._functor()
        state = {'phenological_factor': torch.tensor(0.2)}
        
        result = pheno.forward(state, time=0, dt=86400)
        
        diagnostic = result['diagnostics.phenology.growth_active']
        assert isinstance(diagnostic, UpdateItem)
        assert diagnostic.mode is UpdateMode.REPLACE
        assert diagnostic.per_timestep is False
        assert diagnostic.value.item() is True
    
    def test_forward_returns_only_declared_control_ports(self):
        """The production forward path must not expose a rate alias."""
        pheno = self._functor()
        state = {'phenological_factor': torch.tensor(0.05)}
        result = pheno.forward(state, time=0.0, dt=86400.0)

        assert set(result) == set(pheno.metadata.output_variables)
        assert 'phenological_factor' not in result
        assert 'growth_active' not in result
        assert 'rate' not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert all(item.mode is UpdateMode.REPLACE for item in result.values())
    
    def test_forward_uses_only_the_parameter_backed_activity_threshold(self):
        """The observer diagnoses activity without rebuilding phenology."""
        pheno = self._functor()
        threshold = pheno.growth_active_threshold

        inactive = pheno.forward(
            {'phenological_factor': torch.tensor(threshold)},
            time=0,
            dt=86400,
        )['diagnostics.phenology.growth_active'].value
        active = pheno.forward(
            {'phenological_factor': torch.tensor(threshold + 0.01)},
            time=0,
            dt=86400,
        )['diagnostics.phenology.growth_active'].value

        assert inactive.item() is False
        assert active.item() is True
    
    def test_parameter_namespace(self):
        """Test the observer loads only parameters used by its response law."""
        pheno = self._functor()

        process_parameters = {
            name
            for name, entry in pheno.metadata.parameters.items()
            if entry['namespace'].startswith('biosphere.')
        }
        assert process_parameters == {'growth_active_threshold'}


class TestStructuralGrowthUnit:
    """Unit tests for StructuralGrowth Functor."""
    
    def test_initialization(self):
        """Test StructuralGrowth can be initialized without errors."""
        struct = StructuralGrowth(
            name="test_structural",
            device=torch.device("cpu")
        )
        
        assert struct.name == "test_structural"
        assert struct.device.type == "cpu"
        assert set(struct.get_required_inputs()) == {
            'growth_rate_leaf', 'leaf_area_index'
        }
        assert 'sla' in struct.metadata.parameters
        assert 'leaf_turnover_rate' in struct.metadata.parameters
        assert not hasattr(struct, 'senescence_rate')
    
    def test_forward_pass(self):
        """Test forward pass returns correct structure."""
        struct = StructuralGrowth()
        
        state = {
            'growth_rate_leaf': torch.tensor(2.0e-7),
            'leaf_area_index': torch.tensor(3.0)
        }
        
        result = struct.forward(state, time=0, dt=86400)
        
        assert set(result) == set(struct.metadata.output_variables)
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert all(item.mode is UpdateMode.REPLACE for item in result.values())
        assert result[
            'diagnostics.structural_growth.step_leaf_area_delta'
        ].per_timestep is True
    
    def test_forward_adapter_returns_canonical_diagnostics(self):
        """The test adapter unwraps the four canonical diagnostics."""
        struct = StructuralGrowth()
        
        state = {
            'growth_rate_leaf': torch.tensor(2.0e-7),
            'leaf_area_index': torch.tensor(3.0)
        }
        
        rate_dict = call_forward(struct, state)
        
        assert isinstance(rate_dict, dict)
        assert set(rate_dict) == set(struct.metadata.output_variables)
        assert 'rate' not in rate_dict
        assert all(isinstance(value, torch.Tensor) for value in rate_dict.values())
    
    def test_lai_dynamics(self):
        """Test LAI growth and senescence dynamics."""
        struct = StructuralGrowth()
        
        # Test with growth
        state = {
            'growth_rate_leaf': torch.tensor(2.0e-7),
            'leaf_area_index': torch.tensor(3.0)
        }
        
        dt = 86400.0  # 1 day in seconds
        result = struct.forward(state, time=0, dt=dt)
        
        gross = result[
            'diagnostics.structural_growth.gross_leaf_area_growth_tendency'
        ].value
        loss = result[
            'diagnostics.structural_growth.leaf_area_senescence_tendency'
        ].value
        net = result[
            'diagnostics.structural_growth.net_leaf_area_tendency'
        ].value
        step = result[
            'diagnostics.structural_growth.step_leaf_area_delta'
        ].value

        assert torch.allclose(net, gross - loss, atol=1.0e-12)
        assert torch.allclose(step, net * dt, atol=1.0e-8)
        assert net > 0
    
    def test_parameter_namespace(self):
        """Test parameters use correct namespace."""
        struct = StructuralGrowth()
        
        manager = get_global_manager()
        sla = manager.get_parameter('biosphere.phenology.morphology.sla')
        turnover = manager.get_parameter(
            'biosphere.phenology.turnover.leaf_turnover_rate'
        )
        assert sla is not None and sla.units == 'm2 kg-1 C'
        assert turnover is not None and turnover.units == 's-1'


class TestModuleIntegration:
    """Integration tests for growth module components."""
    
    def test_all_functors_implement_required_methods(self):
        """Test all Functors satisfy the forward-only public API."""
        functors = [
            BiomassAllocation(),
            PhenologyGrowth(phenology_model_owner='gsi_phenology'),
            StructuralGrowth()
        ]
        
        for functor in functors:
            # Check required methods exist
            assert hasattr(functor, 'forward')
            assert hasattr(functor, 'get_dependencies')
            assert hasattr(functor, 'get_provides')
            assert hasattr(functor, 'forward')
            
            # Check they're callable
            assert callable(functor.forward)
            assert callable(functor.get_dependencies)
            assert callable(functor.get_provides)
            assert not hasattr(functor, "compute_rate")
    
    def test_no_state_mutation(self):
        """Test Functors don't mutate input state."""
        allocation = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
        )

        original_npp = torch.tensor(1.0e-6)
        state = {
            'net_primary_productivity': original_npp
        }
        
        # Store original value
        original_value = original_npp.clone()
        
        # Run forward pass
        result = allocation.forward(state, time=0, dt=86400)
        
        # Check state wasn't mutated
        assert torch.equal(state['net_primary_productivity'], original_value)
    
    def test_device_consistency(self):
        """Test outputs are on same device as inputs."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        
        allocation = BiomassAllocation(
            device=device,
            enable_environmental_control=False,
            enable_phenology_control=False,
        )

        state = {
            'net_primary_productivity': torch.tensor(1.0e-6, device=device)
        }
        
        result = allocation.forward(state, time=0, dt=86400)
        
        # All outputs should be on same device
        for item in result.values():
            assert isinstance(item, UpdateItem)
            assert item.value.device.type == device.type


@pytest.fixture
def sample_state():
    """Provide sample state for testing."""
    return {
        'net_primary_productivity': torch.tensor(1.5e-6),
        'air_temperature': torch.tensor(298.15),
        'water_stress_factor': torch.tensor(0.8),
        'leaf_area_index': torch.tensor(4.5),
        'phenology_phase': torch.tensor(2),  # Mature phase
        'phenological_factor': torch.tensor(0.8),
        'growth_rate_leaf': torch.tensor(3.0e-7),
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

