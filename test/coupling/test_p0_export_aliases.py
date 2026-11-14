"""
Verify the remaining spatial adapter aliases and functional APIs.
"""

import pytest
from relic.coupling.scales.adapters import (
    SpatialAggregation,
    ConservativeRemap,
    SpatialScaleCoupling,
    GridRemapping,
    ConservativeGridRemapping,
    aggregate_spatial,
    conservative_remap,
)


class TestSpatialExportAliases:
    """Verify the remaining spatial exports are present and functional."""

    def test_spatial_scale_coupling_alias(self):
        """
        Verify SpatialScaleCoupling is an alias for SpatialAggregation.
        
        P0-2 Addition: SpatialScaleCoupling = SpatialAggregation
        """
        assert SpatialScaleCoupling is SpatialAggregation, \
            "SpatialScaleCoupling should be an alias for SpatialAggregation"
        
        config = {'name': 'test_spatial', 'aggregation_strategy': 'area_weighted'}
        functor = SpatialScaleCoupling(name='test_spatial', config=config, source_scale="fine_grid", target_scale="coarse_grid")
        
        assert isinstance(functor, SpatialAggregation), \
            "Instance created with alias should be of type SpatialAggregation"

    def test_grid_remapping_alias(self):
        """
        Verify GridRemapping is an alias for ConservativeRemap.
        
        P0-2 Addition: GridRemapping = ConservativeRemap
        """
        assert GridRemapping is ConservativeRemap, \
            "GridRemapping should be an alias for ConservativeRemap"
        
        config = {'name': 'test_remap', 'remapping_method': 'conservative'}
        functor = GridRemapping(name='test_remap', config=config)
        
        assert isinstance(functor, ConservativeRemap), \
            "Instance created with alias should be of type ConservativeRemap"

    def test_conservative_grid_remapping_alias(self):
        """
        Verify ConservativeGridRemapping is an alias for ConservativeRemap.
        
        P0-2 Addition: ConservativeGridRemapping = ConservativeRemap
        """
        assert ConservativeGridRemapping is ConservativeRemap, \
            "ConservativeGridRemapping should be an alias for ConservativeRemap"
        
        config = {'name': 'test_conservative', 'remapping_method': 'conservative'}
        functor = ConservativeGridRemapping(name='test_conservative', config=config)
        
        assert isinstance(functor, ConservativeRemap), \
            "Instance created with alias should be of type ConservativeRemap"

    def test_all_class_aliases_exist(self):
        """
        Verify all remaining class aliases point to their canonical classes.
        
        This is a comprehensive check for P0-2 class alias additions.
        """
        expected_aliases = {
            'SpatialScaleCoupling': SpatialAggregation,
            'GridRemapping': ConservativeRemap,
            'ConservativeGridRemapping': ConservativeRemap,
        }
        
        for alias_name, expected_class in expected_aliases.items():
            # Get the alias from module globals
            alias = globals()[alias_name]
            
            # Verify it points to the expected class
            assert alias is expected_class, \
                f"{alias_name} should be an alias for {expected_class.__name__}"

    def test_aggregate_spatial_function(self):
        """
        Verify aggregate_spatial functional API exists and is callable.
        
        P0-2 Addition: aggregate_spatial() function
        """
        assert callable(aggregate_spatial), \
            "aggregate_spatial should be a callable function"
        
        import inspect
        sig = inspect.signature(aggregate_spatial)
        
        assert 'data' in sig.parameters, \
            "aggregate_spatial should have 'data' parameter"
        assert 'strategy' in sig.parameters, \
            "aggregate_spatial should have 'strategy' parameter"
        
        assert sig.parameters['strategy'].default == 'area_weighted', \
            "aggregate_spatial strategy default should be 'area_weighted'"

    def test_conservative_remap_function(self):
        """
        Verify conservative_remap functional API exists and is callable.
        
        P0-2 Addition: conservative_remap() function
        """
        assert callable(conservative_remap), \
            "conservative_remap should be a callable function"
        
        import inspect
        sig = inspect.signature(conservative_remap)
        
        assert 'data' in sig.parameters, \
            "conservative_remap should have 'data' parameter"
        assert 'source_grid' in sig.parameters, \
            "conservative_remap should have 'source_grid' parameter"
        assert 'target_grid' in sig.parameters, \
            "conservative_remap should have 'target_grid' parameter"

    def test_all_functional_apis_exist(self):
        """
        Verify the remaining functional APIs are present and callable.
        
        This is a comprehensive check for P0-2 functional API additions.
        """
        functional_apis = [
            aggregate_spatial,
            conservative_remap,
        ]
        
        for func in functional_apis:
            assert callable(func), \
                f"{func.__name__} should be callable"
            
            # Verify it has a docstring
            assert func.__doc__ is not None, \
                f"{func.__name__} should have a docstring"
            
            # Verify docstring mentions "Functional API"
            assert "Functional API" in func.__doc__, \
                f"{func.__name__} docstring should mention 'Functional API'"

    def test_backward_consistency_complete(self):
        """
        Verify the remaining three class aliases and two functions.
        
        This test verifies the complete P0-2 requirement is satisfied.
        """
        # Class aliases (3)
        class_aliases = [
            SpatialScaleCoupling,
            GridRemapping,
            ConservativeGridRemapping,
        ]
        
        # Functional API aliases (2)
        function_aliases = [
            aggregate_spatial,
            conservative_remap,
        ]
        
        # Verify all are imported successfully
        assert len(class_aliases) == 3, "Should have 3 class aliases"
        assert len(function_aliases) == 2, "Should have 2 functional API aliases"
        
        # Verify all class aliases are classes
        for alias in class_aliases:
            assert isinstance(alias, type), \
                f"{alias} should be a class"
        
        # Verify all function aliases are callable
        for func in function_aliases:
            assert callable(func), \
                f"{func} should be callable"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


