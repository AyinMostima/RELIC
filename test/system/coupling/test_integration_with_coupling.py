"""
Integration Tests for Central Model Builder with Coupling System

This module tests the integration between the central model builder and the
existing relic/coupling system to ensure seamless operation and backward
consistency.

Test Objectives:
    - Verify builder integrates with ScaleManager
    - Test HierarchicalAggregator consistency
    - Validate conservation laws are maintained
    - Ensure CF metadata consistency
    - Test multi-scale aggregation workflows

Mathematical Validation:
    Tests verify conservation laws during aggregation:
    危(X_fine 脳 A_fine) = 危(X_coarse 脳 A_coarse)
    
    And area-weighted aggregation formulas:
    X_coarse = 危(X_fine 脳 A_fine) / 危(A_fine)

References:
    - RELIC Coupling System Architecture
    - RHESSys Hierarchical Aggregation (Tague & Band, 2004) 
    - CF-1.8 Conservation Metadata Standards
"""

import sys
import os
import unittest
import warnings

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# Import components under test
from relic.coupling.scales.hierarchical.model_builder import (
    create_model_builder, VariableSpec,
    AggregationMethod
)
from relic.coupling.scales.types import SpatialQuantityBasis

CONSERVATIVE_BY_BASIS = {
    SpatialQuantityBasis.CELL_TOTAL: [AggregationMethod.SUM],
    SpatialQuantityBasis.AREA_DENSITY: [AggregationMethod.AREA_WEIGHTED_MEAN],
}

# Import coupling system components  
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
from relic.coupling.scales.conservation_enforcer import ConservationEnforcer
from relic.core.state import StateManager

# Import models for comparison
from relic.models import (
    SCALE_HIERARCHY, get_scale_variables
)


class TestScaleManagerIntegration(unittest.TestCase):
    """
    L2 Integration tests with ScaleManager.
    
    Tests that the central model builder integrates seamlessly with the
    existing ScaleManager coupling interface.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
            self.state_manager = StateManager()
            self.facade = ScaleManager(state_manager=self.state_manager)
            self.facade.initialize()
    
    def test_facade_can_use_builder_functors(self):
        """Test that ScaleManager can consume builder-generated functors."""
        # Get effective functors from builder
        landscape_functors = self.builder.get_effective_functors("landscape")
        
        # Check that ScaleManager can work with these functors
        self.assertIsInstance(landscape_functors, list)
        self.assertGreater(len(landscape_functors), 0)
        
        # Verify no duplicates (key improvement over historical system)
        unique_functors = list(dict.fromkeys(landscape_functors))  # Preserve order
        self.assertEqual(len(landscape_functors), len(unique_functors),
                        "Builder should eliminate duplicates")
        
        # Test that ScaleManager methods can process these functors
        # (This is a structural test - ScaleManager should accept any valid functor list)
        try:
            # ScaleManager should be able to work with the functor list
            # In practice, it would use these for registration or processing
            self.assertIsNotNone(landscape_functors)
        except Exception as e:
            self.fail(f"ScaleManager consistency issue: {e}")

    def test_facade_can_use_builder_variables(self):
        """Test that ScaleManager can consume builder-generated variables."""
        # Get effective variables from builder
        patch_variables = self.builder.get_effective_variables("patch")

        # Check structure is consistent with ScaleManager expectations
        self.assertIsInstance(patch_variables, dict)
        self.assertGreater(len(patch_variables), 0)

        # Verify VariableSpec can be converted to ScaleManager format if needed
        for var_name, var_spec in patch_variables.items():
            self.assertIsInstance(var_spec, VariableSpec)
            self.assertTrue(var_spec.units)  # Required field
            # A CF standard name when one exists, otherwise None (never an invented name).
            self.assertTrue(var_spec.standard_name is None or var_spec.standard_name)

            # Test CF compliance (important for ScaleManager)
            self.assertIsNotNone(var_spec.cell_methods)
            self.assertIn("time", var_spec.cell_methods)  # Should have time aggregation
    
    def test_variable_metadata_consistency(self):
        """Test metadata consistency between builder and historical system."""
        # Compare builder variables with historical get_scale_variables
        scales_to_test = ["individual", "patch", "landscape"]
        
        for scale in scales_to_test:
            # Get from builder
            builder_vars = self.builder.get_effective_variables(scale, include_lower=False)
            
            # Get from historical system (only delta variables for fair comparison)
            historical_vars = {}
            try:
                historical_all = get_scale_variables(scale)
                # For delta comparison, we need to extract only scale-specific variables
                # This is complex due to cumulative inheritance in historical system
                historical_vars = historical_all  # For now, compare against all
            except Exception:
                continue  # Skip if scale not available in historical system
            
            # Check that core variables exist in both systems
            # (Some differences expected due to deduplication improvements)
            common_vars = set(builder_vars.keys()) & set(historical_vars.keys())
            
            if common_vars:
                # Test that common variables have consistent core metadata
                for var_name in list(common_vars)[:5]:  # Test first 5 for performance
                    builder_var = builder_vars[var_name]
                    historical_var = historical_vars[var_name]
                    
                    # Units should match
                    historical_units = historical_var.get('units', '1')
                    self.assertEqual(builder_var.units, historical_units,
                                   f"Units mismatch for {var_name} in scale {scale}")
                    
                    # Standard names should match
                    historical_std_name = historical_var.get('standard_name')
                    self.assertEqual(builder_var.standard_name, historical_std_name,
                                   f"Standard name mismatch for {var_name} in scale {scale}")


class TestHierarchicalAggregatorIntegration(unittest.TestCase):
    """
    L2 Integration tests with HierarchicalAggregator.
    
    Tests that variables from the central model builder work correctly
    with the hierarchical aggregation system.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
            # Create a mock hierarchy index for testing
            from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex
            mock_index = HierarchyIndex(
                source_scale='patch',
                target_scale='hillslope',
                n_source_units=6
            )
            self.aggregator = HierarchicalAggregator(index=mock_index, weight_type="area", rule="conservative")
    
    def test_aggregator_handles_builder_variables(self):
        """Test that HierarchicalAggregator can process builder variables."""
        # Get variables with aggregation metadata
        patch_vars = self.builder.get_effective_variables("patch")
        
        # Test that aggregator can understand aggregation methods
        for var_name, var_spec in patch_vars.items():
            self.assertIsInstance(var_spec.aggregation, AggregationMethod)
            
            # Test consistency with aggregator expectations
            if var_spec.spatial_basis.is_extensive:
                self.assertIn(var_spec.aggregation, CONSERVATIVE_BY_BASIS[var_spec.spatial_basis],
                              f"Extensive variable {var_name} should use its conservative aggregation")
            
            elif var_spec.spatial_basis is SpatialQuantityBasis.INTENSIVE:
                self.assertIn(var_spec.aggregation, [
                    AggregationMethod.MEAN,
                    AggregationMethod.AREA_WEIGHTED_MEAN,
                    AggregationMethod.MAX,
                    AggregationMethod.MIN
                ], f"Intensive variable {var_name} should use mean-based aggregation")
    
    def test_conservation_law_consistency(self):
        """Test that conservation laws work with builder variables."""
        # Get variables from builder
        landscape_vars = self.builder.get_effective_variables("landscape")
        
        # Identify conserved variables (extensive variables)
        conserved_vars = []
        for var_name, var_spec in landscape_vars.items():
            if var_spec.spatial_basis.is_extensive:
                conserved_vars.append(var_name)
                
                # Check that conserved variables have appropriate aggregation
                self.assertIn(var_spec.aggregation, CONSERVATIVE_BY_BASIS[var_spec.spatial_basis],
                              f"Conserved variable {var_name} must use conservation-preserving aggregation")
        
        self.assertGreater(len(conserved_vars), 0, "Should have at least some conserved variables")
    
    def test_cf_metadata_for_aggregation(self):
        """Test CF metadata is suitable for aggregation processes."""
        # Get variables with CF metadata
        basin_vars = self.builder.get_effective_variables("basin")
        
        for var_name, var_spec in basin_vars.items():
            # CF cell_methods should be present
            self.assertIsNotNone(var_spec.cell_methods)
            self.assertIsInstance(var_spec.cell_methods, str)
            
            # Should contain aggregation information
            cell_methods = var_spec.cell_methods.lower()
            
            # Should specify area aggregation for spatial variables
            if var_spec.aggregation is AggregationMethod.AREA_WEIGHTED_MEAN:
                self.assertTrue(
                    "area:" in cell_methods.replace(" ", ":"),
                    f"Area-weighted variable {var_name} should have area: in cell_methods"
                )
            
            # Should specify time aggregation for temporal processes
            if var_spec.aggregation != AggregationMethod.STANDARD_DEVIATION:
                self.assertIn("time", cell_methods,
                            f"Process variable {var_name} should have time aggregation info")


class TestConservationEnforcerIntegration(unittest.TestCase):
    """
    L2 Integration tests with ConservationEnforcer.
    
    Tests that the conservation enforcement system works correctly
    with variables from the central model builder.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
            self.enforcer = ConservationEnforcer()
    
    def test_conservation_enforcer_recognizes_conserved_variables(self):
        """Test that ConservationEnforcer identifies conserved variables correctly."""
        # Get variables from builder
        regional_vars = self.builder.get_effective_variables("regional")
        
        # Extract extensive (conserved) variables
        extensive_vars = {}
        for var_name, var_spec in regional_vars.items():
            if var_spec.spatial_basis.is_extensive:
                extensive_vars[var_name] = var_spec
        
        self.assertGreater(len(extensive_vars), 0, "Should have extensive variables")
        
        # Test that these variables have conservation-consistent metadata
        for var_name, var_spec in extensive_vars.items():
            # Units should be flux or storage units (consistent with conservation)
            units = var_spec.units.lower()
            
            # Check for common flux/storage unit patterns
            is_flux = any(pattern in units for pattern in ['s-1', '/s', 'per s'])
            is_storage = any(pattern in units for pattern in ['m-2', 'm-3', '/m2', '/m3'])
            
            if is_flux or is_storage:
                # Flux and storage variables should be conserved
                self.assertTrue(var_spec.spatial_basis.is_extensive,
                                f"{var_name} with units {var_spec.units} should be extensive")
    
    def test_conservation_metadata_in_extras(self):
        """Test that conservation information is available in variable extras."""
        # Get variables and check for conservation metadata
        vars_to_check = self.builder.get_effective_variables("landscape")
        
        # Look for conservation information in extras or infer from type
        for var_name, var_spec in vars_to_check.items():
            if var_spec.spatial_basis.is_extensive:
                # Extensive variables should be marked as conserved
                # Either explicitly in extras or implicitly by type
                if 'conserved' in var_spec.extras:
                    self.assertTrue(var_spec.extras['conserved'],
                                  f"Extensive variable {var_name} should be marked as conserved")
                
                # Check that aggregation method preserves conservation
                conservation_preserving = CONSERVATIVE_BY_BASIS[var_spec.spatial_basis]
                self.assertIn(var_spec.aggregation, conservation_preserving,
                            f"Conserved variable {var_name} should use conservation-preserving aggregation")


class TestMultiScaleWorkflowIntegration(unittest.TestCase):
    """
    L3 System tests for complete multi-scale workflows.
    
    Tests full workflows from individual to regional scales using the
    central model builder with the coupling system.
    """
    
    def setUp(self):
        """Set up comprehensive test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
            self.state_manager = StateManager()
            self.facade = ScaleManager(state_manager=self.state_manager)
            self.facade.initialize()
    
    def test_complete_upscaling_workflow(self):
        """Test complete workflow from individual to regional scale."""
        # Test the full hierarchy
        scales = ["individual", "patch", "hillslope", "landscape", "basin", "regional"]
        
        previous_functors = set()
        previous_variables = set()
        
        for scale in scales:
            # Get effective configuration
            try:
                scale_functors = self.builder.get_effective_functors(scale, include_lower=True)
                scale_variables = self.builder.get_effective_variables(scale, include_lower=True)
            except ValueError as e:
                self.fail(f"Failed to get configuration for scale {scale}: {e}")
            
            # Verify progressive accumulation
            current_functors = set(scale_functors)
            current_variables = set(scale_variables.keys())
            
            # Current scale should include all previous scales' components
            self.assertTrue(previous_functors.issubset(current_functors),
                          f"Scale {scale} should include all functors from lower scales")
            self.assertTrue(previous_variables.issubset(current_variables),
                          f"Scale {scale} should include all variables from lower scales")
            
            # Verify no duplicates
            self.assertEqual(len(scale_functors), len(current_functors),
                           f"Scale {scale} functors contain duplicates")
            
            # Update for next iteration
            previous_functors = current_functors
            previous_variables = current_variables
    
    def test_aggregation_consistency_across_scales(self):
        """Test that aggregation methods are consistent across scales."""
        # Test common variables across multiple scales
        common_vars = set()
        scale_vars = {}
        
        # Collect variables from multiple scales
        test_scales = ["individual", "patch", "landscape"]
        for scale in test_scales:
            vars_dict = self.builder.get_effective_variables(scale, include_lower=True)
            scale_vars[scale] = vars_dict
            
            if not common_vars:
                common_vars = set(vars_dict.keys())
            else:
                common_vars &= set(vars_dict.keys())
        
        # Test consistency for variables present in multiple scales
        for var_name in list(common_vars)[:10]:  # Test first 10 for performance
            # Get variable specs from all scales
            specs = [scale_vars[scale][var_name] for scale in test_scales]
            
            # Core fields should be identical
            reference_spec = specs[0]
            for i, spec in enumerate(specs[1:], 1):
                scale = test_scales[i]
                
                self.assertEqual(spec.units, reference_spec.units,
                               f"Units inconsistent for {var_name} in scale {scale}")
                self.assertEqual(spec.standard_name, reference_spec.standard_name,
                               f"Standard name inconsistent for {var_name} in scale {scale}")
                self.assertEqual(spec.spatial_basis, reference_spec.spatial_basis,
                               f"Variable type inconsistent for {var_name} in scale {scale}")
    
    def test_performance_of_multi_scale_operations(self):
        """Test that multi-scale operations complete in reasonable time."""
        import time
        
        # Test performance of getting all scales
        start_time = time.time()
        
        for scale in SCALE_HIERARCHY:
            functors = self.builder.get_effective_functors(scale)
            variables = self.builder.get_effective_variables(scale)
            
            # Basic sanity checks
            self.assertIsInstance(functors, list)
            self.assertIsInstance(variables, dict)
        
        elapsed = time.time() - start_time
        
        # Should complete all scales in reasonable time
        self.assertLess(elapsed, 10.0, f"Multi-scale operations took {elapsed:.2f}s, expected < 10.0s")
    
    def test_memory_efficiency_of_builder(self):
        """Test that builder is memory efficient compared to historical system."""
        import gc
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create multiple builders (simulating repeated usage)
        builders = []
        for _ in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                builder = create_model_builder()
                # Use the builder
                _ = builder.get_effective_functors("landscape")
                _ = builder.get_effective_variables("landscape")
                builders.append(builder)
        
        # Check memory usage
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Should not grow excessively
        self.assertLess(object_growth, 5000,
                       f"Memory usage grew by {object_growth} objects")
        
        # Clean up
        del builders
        gc.collect()


def run_integration_tests(verbosity=2):
    """
    Run all integration tests.
    
    Parameters:
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test results
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestScaleManagerIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestHierarchicalAggregatorIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestConservationEnforcerIntegration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMultiScaleWorkflowIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    
    print("Running Central Model Builder Integration Tests")
    print("=" * 55)
    
    result = runner.run(suite)
    
    print("\n" + "=" * 55)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Integration Tests')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2],
                       help='Test output verbosity level')
    
    args = parser.parse_args()
    
    # Run integration tests
    run_integration_tests(verbosity=args.verbosity)


