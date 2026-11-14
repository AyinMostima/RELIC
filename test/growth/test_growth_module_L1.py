"""
L1 Unit Tests for Growth Module

This module provides comprehensive L1 unit testing for the growth module components:
- BiomassAllocation
- PhenologyGrowth 
- StructuralGrowth

All tests verify:
1. ParameterManager integration
2. No hardcoded values
3. Correct tensor operations
4. Metadata compliance
5. Conservation laws where applicable
"""

import pytest
import torch
import ast
import inspect

from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.time_constants import SECONDS_PER_DAY
from test.utils.forward_call import call_forward


class TestBiomassAllocationL1:
    """L1 Unit tests for BiomassAllocation functor."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.device = torch.device('cpu')
        
    def test_initialization(self):
        """Test proper initialization with ParameterManager."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device
        )
        
        # Verify the allocation object was created successfully
        assert allocation is not None
        assert allocation.name == "test_biomass_allocation"
        
        # Test that parameters are fetched from ParameterManager
        from relic.parameters import get_global_manager
        param_manager = get_global_manager()
        
        # Get allocation fractions from ParameterManager
        f_leaf = param_manager.get("f_leaf_base", namespace="biosphere.growth.allocation")
        f_stem = param_manager.get("f_stem_base", namespace="biosphere.growth.allocation")
        f_root = param_manager.get("f_root_base", namespace="biosphere.growth.allocation")
        
        # Verify allocation fractions sum to 1
        total = f_leaf + f_stem + f_root
        assert abs(total - 1.0) < 0.01
        
    def test_no_hardcoded_tensors(self):
        """Required state ports fail fast instead of using hidden defaults."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device
        )

        state = {
            'net_primary_productivity': torch.tensor(1.0e-6, device=self.device)
        }

        with pytest.raises(KeyError, match="air_temperature"):
            allocation.forward(state, time=0.0, dt=1.0)

        state['air_temperature'] = torch.tensor(298.15, device=self.device)
        state['water_stress_factor'] = torch.tensor(1.0, device=self.device)
        state['leaf_area_index'] = torch.tensor(3.0, device=self.device)
        state['phenology_phase'] = torch.tensor(2.0, device=self.device)
        result = allocation.forward(state, time=0.0, dt=1.0)
        assert set(result) == set(allocation.metadata.output_variables)
        
    def test_parameter_manager_integration(self):
        """Test that allocation fractions come from immutable ParameterSpecs."""
        param_manager = get_global_manager()

        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device
        )

        values = []
        for name in ('f_leaf_base', 'f_stem_base', 'f_root_base'):
            spec = param_manager.get_parameter(
                f'biosphere.growth.allocation.{name}'
            )
            assert spec is not None
            assert spec.units == 'dimensionless'
            assert spec.bounds is not None
            values.append(spec.value)
        assert sum(values) == pytest.approx(1.0)
        assert allocation.parameter_manager is param_manager

    def test_canonical_required_outputs(self):
        """Ensure the forward path exposes only the required growth ports."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device,
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        state = {
            "net_primary_productivity": torch.tensor(1.2e-6, device=self.device)
        }
        result = allocation.forward(state, time=0.0, dt=1.0)
        assert set(result) == set(allocation.metadata.output_variables)
        for key in ("growth_rate_leaf", "growth_rate_stem", "growth_rate_root"):
            assert isinstance(result[key], UpdateItem)
        assert "rate" not in result
        assert "carbon_allocation_rate" not in result
        assert "total_growth_rate" not in result

    def test_missing_npp_fails_fast(self):
        """A missing carbon input must not produce synthetic diagnostics."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device,
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        with pytest.raises(KeyError, match="net_primary_productivity"):
            call_forward(allocation, {})
        
    def test_conservation(self):
        """Test carbon conservation in allocation."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device,
            enable_environmental_control=False,
            enable_phenology_control=False,
        )

        npp = 1.0e-6
        state = {
            'net_primary_productivity': torch.tensor(npp, device=self.device),
        }

        result = allocation.forward(state, time=0.0, dt=3600.0)

        total_growth = (
            result['growth_rate_leaf'].value
            + result['growth_rate_stem'].value
            + result['growth_rate_root'].value
        )
        expected_total = torch.as_tensor(npp, dtype=total_growth.dtype, device=self.device)
        residual = result[
            'diagnostics.biomass_allocation.carbon_balance_residual'
        ].value
        assert torch.allclose(total_growth, expected_total, rtol=1e-6, atol=1e-12)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12)
        assert 'total_construction_respiration' not in result
        for key in ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'):
            assert result[key].mode is UpdateMode.ACCUMULATE
            assert result[key].per_timestep is True

    def test_noncanonical_npp_key_is_rejected(self):
        """Only the declared net_primary_productivity state port is valid."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device,
            enable_environmental_control=False,
            enable_phenology_control=False,
            enable_allometry_control=False,
        )
        state = {'net_primary_production': torch.tensor(5.0, device=self.device)}
        with pytest.raises(KeyError, match="net_primary_productivity"):
            call_forward(allocation, state)

    def test_nonfinite_npp_fails_fast(self):
        """A non-finite carbon budget must not be converted to a zero flux."""
        allocation = BiomassAllocation(
            name="test_biomass_allocation",
            device=self.device,
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        with pytest.raises(ValueError, match="finite"):
            allocation.forward(
                {"net_primary_productivity": torch.tensor(float("nan"))},
                time=0.0,
                dt=SECONDS_PER_DAY,
            )


class TestPhenologyGrowthL1:
    """L1 Unit tests for PhenologyGrowth functor."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.device = torch.device('cpu')
        
    def test_initialization(self):
        """Test proper initialization with ParameterManager."""
        phenology = PhenologyGrowth(
            device=self.device,
            phenology_model_owner="gsi_phenology",
        )
        
        # Verify the phenology object was created successfully
        assert phenology is not None
        assert phenology.name == "phenology_growth"
        
        assert 0.0 <= phenology.growth_active_threshold <= 1.0
        assert phenology.metadata.input_variables == ['phenological_factor']
        assert phenology.metadata.output_variables == [
            'diagnostics.phenology.growth_active'
        ]
        assert phenology.get_dependencies() == ['gsi_phenology']
        
    def test_activity_threshold(self):
        """Test the calibrated observation threshold without a second model."""
        phenology = PhenologyGrowth(
            device=self.device,
            phenology_model_owner="gsi_phenology",
        )
        
        test_cases = [
            (0.0, False),
            (phenology.growth_active_threshold, False),
            (1.0, True),
        ]
        
        for factor_value, expected in test_cases:
            state = {
                'phenological_factor': torch.tensor(
                    factor_value, device=self.device
                ),
            }
            result = phenology.forward(state, time=0.0, dt=86400.0)

            active = result['diagnostics.phenology.growth_active']
            assert isinstance(active, UpdateItem)
            assert active.mode is UpdateMode.REPLACE
            assert active.per_timestep is False
            assert active.unit == 'dimensionless'
            assert bool(active.value.item()) is expected
            
    def test_required_inputs_fail_fast(self):
        """Required state ports cannot be replaced by parameter defaults."""
        phenology = PhenologyGrowth(
            device=self.device,
            phenology_model_owner="gsi_phenology",
        )
        
        canonical = {
            'phenological_factor': torch.tensor(0.6),
        }
        for missing in phenology.get_required_inputs():
            state = dict(canonical)
            state.pop(missing)
            with pytest.raises(KeyError, match=missing):
                phenology.forward(state, time=0.0, dt=86400.0)

        result = phenology.forward(canonical, time=0.0, dt=86400.0)
        assert set(result) == set(phenology.metadata.output_variables)
        assert 'rate' not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert all(item.mode is UpdateMode.REPLACE for item in result.values())


class TestStructuralGrowthL1:
    """L1 Unit tests for StructuralGrowth functor."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.device = torch.device('cpu')
        
    def test_initialization(self):
        """Test proper initialization with ParameterManager."""
        structural = StructuralGrowth(
            name="test_structural_growth",
            device=self.device
        )
        
        # Verify the structural object was created successfully
        assert structural is not None
        assert structural.name == "test_structural_growth"
        
        param_manager = get_global_manager()
        
        sla = param_manager.get_parameter('biosphere.phenology.morphology.sla')
        turnover = param_manager.get_parameter(
            'biosphere.phenology.turnover.leaf_turnover_rate'
        )
        assert sla is not None and sla.units == 'm2 kg-1 C'
        assert turnover is not None and turnover.units == 's-1'
        assert 0.01 <= sla.value <= 50.0
        assert 0 < turnover.value < 0.01
        
    def test_lai_calculation(self):
        """Test LAI change rate calculation."""
        structural = StructuralGrowth(
            name="test_structural_growth",
            device=self.device
        )
        
        leaf_growth = 2.0e-7  # kg C m-2 s-1
        current_lai = 3.0
        
        state = {
            'growth_rate_leaf': torch.tensor(leaf_growth, device=self.device),
            'leaf_area_index': torch.tensor(current_lai, device=self.device)
        }
        environment = {}
        
        result = call_forward(structural, {**state, **environment})
        
        gross = result[
            'diagnostics.structural_growth.gross_leaf_area_growth_tendency'
        ]
        loss = result[
            'diagnostics.structural_growth.leaf_area_senescence_tendency'
        ]
        net = result[
            'diagnostics.structural_growth.net_leaf_area_tendency'
        ]
        assert net.item() > 0
        assert torch.allclose(net, gross - loss, atol=1.0e-12)
            
    def test_structural_requires_canonical_ports(self):
        structural = StructuralGrowth(
            name="test_structural_growth",
            device=self.device
        )
        alias_state = {
            'leaf_growth_rate': torch.tensor(0.5, device=self.device),
            'leaf_area_index': torch.tensor(2.0, device=self.device)
        }
        with pytest.raises(KeyError, match="growth_rate_leaf"):
            call_forward(structural, alias_state)

        with pytest.raises(KeyError, match="growth_rate_leaf"):
            call_forward(structural, {})
        
    def test_parameter_ranges(self):
        """Test that all parameters are within expected ranges."""
        StructuralGrowth(
            name="test_structural_growth",
            device=self.device
        )
        
        param_manager = get_global_manager()
        sla = param_manager.get_parameter('biosphere.phenology.morphology.sla')
        turnover = param_manager.get_parameter(
            'biosphere.phenology.turnover.leaf_turnover_rate'
        )
        assert sla is not None and 0.01 <= sla.value <= 50.0
        assert turnover is not None and 0 < turnover.value < 0.01

    def test_rate_core_promotes_over_the_floating_input_dtypes(self):
        """Mixed float64 growth and float32 LAI must yield float64 tendencies, not float32."""
        structural = StructuralGrowth(name="test_structural_dtype", device=self.device)
        growth = torch.tensor(1.0e-6, dtype=torch.float64)
        lai = torch.tensor(3.0, dtype=torch.float32)
        mixed = structural.forward({"growth_rate_leaf": growth, "leaf_area_index": lai}, time=0.0, dt=300.0)
        gross = mixed["diagnostics.structural_growth.gross_leaf_area_growth_tendency"].value
        net = mixed["diagnostics.structural_growth.net_leaf_area_tendency"].value
        assert gross.dtype == torch.float64
        assert net.dtype == torch.float64
        single = structural.forward(
            {"growth_rate_leaf": growth.to(torch.float32), "leaf_area_index": lai},
            time=0.0,
            dt=300.0,
        )
        assert single["diagnostics.structural_growth.net_leaf_area_tendency"].value.dtype == torch.float32
        # References: Wright et al. (2004); PyTorch type promotion.
        # Mathematical expression: r_growth = SLA * growth_rate_leaf is precision-invariant to float32 rounding.
        # Parameter meanings: SLA from ParameterManager; growth_rate_leaf the input flux.
        # Units: r_growth [m2 m-2 s-1]; growth_rate_leaf [kg C m-2 s-1]; SLA [m2 kg-1 C].
        # Domain/constraints: identical inputs in both precisions.
        # Sign convention: growth positive.
        torch.testing.assert_close(
            single["diagnostics.structural_growth.gross_leaf_area_growth_tendency"].value.double(),
            gross,
            rtol=1.0e-6,
            atol=0.0,
        )


def test_parameter_consistency():
    """Test that parameters are consistent across all modules."""
    param_manager = get_global_manager()
    modules = (
        BiomassAllocation(name="allocation"),
        PhenologyGrowth(phenology_model_owner="gsi_phenology"),
        StructuralGrowth(name="structural"),
    )

    assert all(module.parameter_manager is param_manager for module in modules)
    assert modules[0].get_dependencies() == [
        'autotrophic_respiration', 'water_stress', 'gsi_phenology'
    ]
    assert modules[1].get_dependencies() == ['gsi_phenology']
    assert modules[2].get_dependencies() == ['biomass_allocation']

def test_no_torch_tensor_calls():
    """Reject numeric literal arguments to torch.tensor in executable class code."""
    modules = [BiomassAllocation, PhenologyGrowth, StructuralGrowth]

    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        hardcoded_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == 'torch'
                and function.attr == 'tensor'
            ):
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(
                argument.value, (int, float)
            ):
                hardcoded_calls.append((node.lineno, argument.value))
        assert not hardcoded_calls, (
            f"{module.__name__} contains hardcoded torch.tensor calls: "
            f"{hardcoded_calls}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


