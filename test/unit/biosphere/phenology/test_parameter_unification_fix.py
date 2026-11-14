"""
Test suite to verify parameter unification fixes for phenology module.

This test specifically validates the fixes applied for:
- P0: Strategy classes parameter access interface
- P1: Parameter name matching and hardcoded constants
- Parameter system integration

Author: RELIC Parameter Unification Team  
Date: 2024-12-23
"""

import pytest
import torch
import sys
import os
import inspect

import relic.scientific.biosphere.phenology.common as phenology_common
import relic.scientific.biosphere.phenology.common.strategies as phenology_strategies

# Add project root to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

from relic.scientific.biosphere.phenology.common.strategies import (
    PhenologyStrategy, StaticGrowthStrategy, GDDGrowthStrategy, 
    CompositeStrategy, create_temperate_deciduous_strategy,
    create_crop_strategy,
)
from relic.parameters import get_global_manager


class TestP0FixStrategyParameterAccess:
    """Test P0 fix: Strategy classes parameter access interface."""

    def test_strategy_requires_explicit_parameter_manager(self):
        """Reject construction when the non-Functor manager owner is omitted."""
        with pytest.raises(TypeError):
            StaticGrowthStrategy()

    def test_strategy_family_requires_parameter_manager_in_every_signature(self):
        """Keep explicit injection mandatory across every strategy constructor."""
        strategy_classes = (
            phenology_strategies.PhenologyStrategy,
            phenology_strategies.GrowthFactorStrategy,
            phenology_strategies.StaticGrowthStrategy,
            phenology_strategies.GDDGrowthStrategy,
            phenology_strategies.SenescenceStrategy,
            phenology_strategies.TimeBasedSenescenceStrategy,
            phenology_strategies.DroughtSenescenceStrategy,
            phenology_strategies.TransitionStrategy,
            phenology_strategies.SmoothTransitionStrategy,
            phenology_strategies.CompositeStrategy,
        )

        for strategy_class in strategy_classes:
            parameter = inspect.signature(strategy_class).parameters[
                "parameter_manager"
            ]
            assert parameter.default is inspect.Parameter.empty

        for factory in (
            create_temperate_deciduous_strategy,
            create_crop_strategy,
        ):
            parameter = inspect.signature(factory).parameters[
                "parameter_manager"
            ]
            assert parameter.default is inspect.Parameter.empty

    def test_strategy_source_has_no_global_manager_or_local_key_resolver(self):
        """Forbid both the global fallback and a parallel namespace resolver."""
        source = inspect.getsource(phenology_strategies)
        assert "get_global_manager" not in source
        assert "_resolve_parameter_key" not in source

    def test_strategy_parameter_access_is_thin_injected_manager_delegation(self):
        """Forward the untouched key and namespace to the injected owner."""

        class RecordingManager:
            def __init__(self):
                self.calls = []

            def get(self, key, **kwargs):
                self.calls.append((key, kwargs))
                return 17.0

        manager = RecordingManager()
        strategy = StaticGrowthStrategy(parameter_manager=manager)
        value = strategy.get_parameter_value(
            "leaf_onset_doy",
            namespace="biosphere.phenology.static",
            required=True,
            units="day",
            expected_type=float,
        )

        assert value == 17.0
        assert manager.calls == [
            (
                "leaf_onset_doy",
                {
                    "namespace": "biosphere.phenology.static",
                    "units": "day",
                    "expected_type": float,
                    "required": True,
                },
            )
        ]

    def test_strategy_factories_propagate_one_explicit_manager(self):
        """Inject the same canonical owner into factory products and children."""
        manager = get_global_manager()

        temperate = create_temperate_deciduous_strategy(
            parameter_manager=manager
        )
        crop = create_crop_strategy(
            "wheat",
            parameter_manager=manager,
        )

        for composite in (temperate, crop):
            assert composite.parameter_manager is manager
            assert all(
                strategy.parameter_manager is manager
                for strategy in composite.strategies
            )

    def test_common_package_does_not_export_secondary_gsi_strategy(self):
        """Keep the canonical GSI runtime as the only public GSI formula."""
        assert not hasattr(phenology_common, "GSIGrowthStrategy")
        assert not hasattr(phenology_common, "gsi_response_component")
    
    def test_phenology_strategy_exposes_parameter_entrypoint(self):
        """Test that PhenologyStrategy exposes canonical get_parameter_value()."""
        assert hasattr(PhenologyStrategy, "get_parameter_value")
        
    def test_strategy_parameter_namespace(self):
        """Test that strategy has correct parameter namespace."""
        strategy = StaticGrowthStrategy(
            parameter_manager=get_global_manager()
        )
        assert hasattr(strategy, 'parameter_namespace')
        assert strategy.parameter_namespace == "biosphere.phenology"
        
    def test_strategy_get_parameter_value_method(self):
        """Test that strategy can call get_parameter_value method."""
        strategy = StaticGrowthStrategy(
            parameter_manager=get_global_manager()
        )
        
        # This should not raise AttributeError anymore
        try:
            value = strategy.get_parameter_value('clamp_min',
                                               namespace='biosphere.phenology.computational')
            assert value is not None  # Should get value or default
        except AttributeError:
            pytest.fail("Strategy class should have get_parameter_value method")
            
    def test_gdd_strategy_parameter_access(self):
        """Test GDD strategy parameter access works."""
        strategy = GDDGrowthStrategy(
            parameter_manager=get_global_manager()
        )
        
        # Test accessing computational parameters that were added in fix
        value = strategy.get_parameter_value('clamp_min',
                                           namespace='biosphere.phenology.computational')
        assert isinstance(value, (int, float))
        
    def test_direct_strategy_module_has_no_secondary_gsi_runtime(self):
        """Reject the duplicate four-factor GSI strategy and its factory."""
        assert not hasattr(phenology_strategies, "GSIGrowthStrategy")
        assert not hasattr(phenology_strategies, "create_mediterranean_strategy")
        
    def test_composite_strategy_parameter_access(self):
        """Test composite strategy parameter access works."""
        manager = get_global_manager()
        sub_strategy = StaticGrowthStrategy(parameter_manager=manager)
        strategy = CompositeStrategy(
            strategies=[sub_strategy],
            parameter_manager=manager,
        )
        
        # Test accessing validation parameters
        clamp_min = strategy.get_parameter_value('clamp_min',
                                                namespace='biosphere.phenology.validation')
        clamp_max = strategy.get_parameter_value('clamp_max',
                                                namespace='biosphere.phenology.validation')
        assert isinstance(clamp_min, (int, float))
        assert isinstance(clamp_max, (int, float))
        assert clamp_min <= clamp_max


class TestP1FixParameterNameMatching:
    """Test P1 fix: Parameter name matching and hardcoded constants."""
    
    def test_parameter_manager_has_added_parameters(self):
        """Test that all P1 parameters were added to parameter manager."""
        param_manager = get_global_manager()
        
        # Test parameters that were added to fix P1 issues
        required_params = [
            'biosphere.phenology.validation.clamp_min',
            'biosphere.phenology.validation.clamp_max',
            'biosphere.phenology.turnover.n_retrans_efficiency'
        ]
        
        for param in required_params:
            try:
                value = param_manager.get(param)
                assert value is not None, f"Parameter {param} should have a value"
            except Exception as e:
                pytest.fail(f"Parameter {param} not found: {e}")
                
    def test_parameter_values_are_reasonable(self):
        """Test that parameter values are within reasonable ranges."""
        param_manager = get_global_manager()
        
        # Test specific parameter values
        n_retrans = param_manager.get('biosphere.phenology.turnover.n_retrans_efficiency')
        assert 0.0 <= n_retrans <= 1.0, f"N retranslocation efficiency {n_retrans} out of range"
        
        clamp_min = param_manager.get('biosphere.phenology.validation.clamp_min')
        clamp_max = param_manager.get('biosphere.phenology.validation.clamp_max')
        assert 0.0 <= clamp_min <= clamp_max <= 1.0, "Clamping values out of range"
        

class TestIntegratedParameterSystem:
    """Test integrated parameter system functionality."""
    
    def test_strategy_environment_integration(self):
        """Test strategy can work with mock environment data."""
        strategy = StaticGrowthStrategy(
            parameter_manager=get_global_manager()
        )
        
        # Create mock environment data
        mock_env = {
            'day_of_year': torch.tensor(150.0),  # Day 150 (May 30)
        }
        
        # This should work without errors
        try:
            result = strategy.compute(mock_env)
            assert isinstance(result, torch.Tensor)
            assert 0.0 <= result.item() <= 1.0, "Growth factor should be in [0,1] range"
        except Exception as e:
            pytest.fail(f"Strategy compute failed with environment data: {e}")
            
    def test_gdd_strategy_with_temperature(self):
        """Test GDD strategy with temperature data."""
        strategy = GDDGrowthStrategy(
            parameter_manager=get_global_manager()
        )
        
        # Create mock environment with temperature
        mock_env = {
            'air_temperature': torch.tensor(293.15),  # 20掳C
        }
        
        try:
            result = strategy.compute(mock_env)
            assert isinstance(result, torch.Tensor)
            assert 0.0 <= result.item() <= 1.0
        except Exception as e:
            pytest.fail(f"GDD strategy failed with temperature data: {e}")
            
    def test_composite_strategy_integration(self):
        """Test composite strategy with multiple sub-strategies."""
        manager = get_global_manager()
        sub1 = StaticGrowthStrategy(parameter_manager=manager)
        sub2 = StaticGrowthStrategy(
            onset_doy=100,
            offset_doy=300,
            parameter_manager=manager,
        )
        
        composite = CompositeStrategy(
            strategies=[sub1, sub2],
            operation='multiply',
            parameter_manager=manager,
        )
        
        mock_env = {'day_of_year': torch.tensor(200.0)}
        
        try:
            result = composite.compute(mock_env)
            assert isinstance(result, torch.Tensor)
            assert 0.0 <= result.item() <= 1.0
        except Exception as e:
            pytest.fail(f"Composite strategy failed: {e}")


class TestBackwardconsistency:
    """Test that fixes maintain backward consistency."""
    
    def test_strategy_old_parameter_interface_still_works(self):
        """Test that strategies' old get_parameter method still works."""
        strategy = StaticGrowthStrategy(
            parameter_manager=get_global_manager()
        )
        
        # Old interface should still work
        onset_doy = strategy.get_parameter('onset_doy', 120)
        assert isinstance(onset_doy, (int, float))
        
    def test_parameter_manager_global_access(self):
        """Test that global parameter manager access still works."""
        from relic.parameters import get_global_manager
        
        param_manager = get_global_manager()
        assert param_manager is not None
        
        # Test getting a known parameter
        try:
            value = param_manager.get('biosphere.phenology.turnover.n_retrans_efficiency')
            assert value is not None
        except Exception as e:
            pytest.fail(f"Global parameter manager access failed: {e}")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])


