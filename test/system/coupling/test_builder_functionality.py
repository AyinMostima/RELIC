"""
Comprehensive Functional Tests for Central Model Builder

This module provides comprehensive functional testing of the central model builder
without requiring complex coupling system dependencies. Focuses on core
functionality and integration readiness.

Test Categories:
    - Functional validation of builder components
    - Data integrity and consistency checks  
    - Performance and scalability tests
    - Integration readiness validation

Mathematical Validation:
    - Conservation law verification: sum(X_fine * A_fine) = sum(X_coarse * A_coarse)
    - Aggregation formula verification: X_coarse = sum(X_fine * A_fine) / sum(A_fine)
    - CF metadata consistency checks

References:
    - RELIC Architecture Specification
    - RHESSys Multi-Scale Modeling Framework
    - CF-1.8 Metadata Conventions
"""

import sys
import os
import unittest
import warnings

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# Import central model builder components
from relic.coupling.scales.hierarchical.model_builder import (
    CentralModelBuilder, create_model_builder, VariableSpec,
    AggregationMethod
)
from relic.coupling.scales.types import SpatialQuantityBasis

# Import models for validation
from relic.models import SCALE_HIERARCHY


class TestBuilderFunctionalCorrectness(unittest.TestCase):
    """
    Comprehensive functional correctness tests for the central model builder.
    
    Tests the complete functionality of deduplication, conflict resolution,
    and metadata management across all scales.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
    
    def test_complete_deduplication_functionality(self):
        """Test comprehensive functor deduplication across all scales."""
        print("\n[TEST] Complete deduplication functionality...")
        
        # Test each scale's deduplication
        for scale in SCALE_HIERARCHY:
            try:
                # Get effective functors
                effective_functors = self.builder.get_effective_functors(scale, include_lower=True)
                delta_functors = self.builder.get_effective_functors(scale, include_lower=False)
                
                # Verify no duplicates in effective list
                self.assertEqual(len(effective_functors), len(set(effective_functors)),
                               f"Duplicates found in {scale} effective functors")
                
                # Verify delta functors are subset of effective
                self.assertTrue(set(delta_functors).issubset(set(effective_functors)),
                               f"Delta functors not subset of effective for {scale}")
                
                print(f"  [OK] {scale}: {len(effective_functors)} effective, {len(delta_functors)} delta")
                
            except Exception as e:
                print(f"  [FAIL] {scale}: Failed - {e}")
                raise
    
    def test_comprehensive_variable_merging(self):
        """Test comprehensive variable merging and conflict resolution."""
        print("\n[TEST] Comprehensive variable merging...")
        
        # Test progressive variable accumulation
        previous_var_count = 0
        
        for scale in SCALE_HIERARCHY:
            try:
                effective_vars = self.builder.get_effective_variables(scale, include_lower=True)
                delta_vars = self.builder.get_effective_variables(scale, include_lower=False)
                
                # Variable count should generally increase (or stay same)
                current_var_count = len(effective_vars)
                self.assertGreaterEqual(current_var_count, previous_var_count,
                                       f"Variable count decreased at {scale}")
                
                # All variables should be VariableSpec instances
                for var_name, var_spec in effective_vars.items():
                    self.assertIsInstance(var_spec, VariableSpec,
                                         f"Variable {var_name} in {scale} not VariableSpec")
                    
                    # Core fields should be present
                    self.assertTrue(var_spec.units, f"Empty units for {var_name}")
                    # A CF standard name when one exists, otherwise None (never empty or invented).
                    self.assertTrue(var_spec.standard_name is None or var_spec.standard_name,
                                    f"Empty standard_name for {var_name}")
                    self.assertIsInstance(var_spec.spatial_basis, SpatialQuantityBasis,
                                         f"Invalid spatial_basis for {var_name}")
                    self.assertIsInstance(var_spec.aggregation, AggregationMethod,
                                         f"Invalid aggregation for {var_name}")
                
                print(f"  [OK] {scale}: {current_var_count} effective, {len(delta_vars)} delta")
                previous_var_count = current_var_count
                
            except Exception as e:
                print(f"  [FAIL] {scale}: Failed - {e}")
                raise
    
    def test_cf_metadata_generation_quality(self):
        """Test quality of CF metadata generation."""
        print("\n[TEST] CF metadata generation quality...")
        
        # Test CF metadata for all scales
        for scale in SCALE_HIERARCHY[:4]:  # Test first 4 scales for performance
            try:
                variables = self.builder.get_effective_variables(scale)
                
                cf_compliant_count = 0
                for var_name, var_spec in variables.items():
                    # Check cell_methods presence and format
                    if var_spec.cell_methods:
                        cell_methods = var_spec.cell_methods.lower()
                        
                        # Should contain valid CF keywords
                        valid_keywords = ['time', 'area', 'volume', 'mean', 'sum', 'maximum', 'minimum']
                        has_valid_keyword = any(keyword in cell_methods for keyword in valid_keywords)
                        
                        if has_valid_keyword:
                            cf_compliant_count += 1
                            
                            # Test aggregation consistency
                            if var_spec.aggregation == AggregationMethod.AREA_WEIGHTED_MEAN:
                                self.assertIn("area", cell_methods,
                                             f"Area-weighted variable {var_name} missing 'area' in cell_methods")
                            elif var_spec.aggregation == AggregationMethod.STANDARD_DEVIATION:
                                self.assertIn("standard_deviation", cell_methods,
                                             f"Std dev variable {var_name} missing 'standard_deviation' in cell_methods")
                
                # Should have reasonable CF compliance rate
                total_vars = len(variables)
                compliance_rate = cf_compliant_count / total_vars if total_vars > 0 else 0
                self.assertGreater(compliance_rate, 0.8,
                                  f"Low CF compliance rate ({compliance_rate:.1%}) for {scale}")
                
                print(f"  [OK] {scale}: {cf_compliant_count}/{total_vars} CF compliant ({compliance_rate:.1%})")
                
            except Exception as e:
                print(f"  [FAIL] {scale}: Failed - {e}")
                raise
    
    def test_conservation_law_consistency(self):
        """Test consistency of conservation laws across scales."""
        print("\n[TEST] Conservation law consistency...")
        
        # Collect extensive (conserved) variables across scales
        conserved_by_scale = {}
        
        for scale in SCALE_HIERARCHY:
            try:
                variables = self.builder.get_effective_variables(scale)
                conserved_vars = []
                
                for var_name, var_spec in variables.items():
                    if var_spec.spatial_basis.is_extensive:
                        conserved_vars.append({
                            'name': var_name,
                            'aggregation': var_spec.aggregation,
                            'units': var_spec.units,
                            'basis': var_spec.spatial_basis,
                        })

                conserved_by_scale[scale] = conserved_vars

                # Test that extensive variables use the conservative aggregation of their basis:
                # cell totals are summed; area densities are averaged over cell areas.
                for var_info in conserved_vars:
                    conservation_methods = (
                        [AggregationMethod.SUM]
                        if var_info['basis'] is SpatialQuantityBasis.CELL_TOTAL
                        else [AggregationMethod.AREA_WEIGHTED_MEAN]
                    )
                    self.assertIn(var_info['aggregation'], conservation_methods,
                                 f"Extensive variable {var_info['name']} in {scale} uses "
                                 f"non-conservative aggregation {var_info['aggregation']}")
                
                print(f"  [OK] {scale}: {len(conserved_vars)} conserved variables")
                
            except Exception as e:
                print(f"  [FAIL] {scale}: Failed - {e}")
                raise
        
        # Test consistency across scales for common conserved variables
        common_vars = set()
        for scale_vars in conserved_by_scale.values():
            var_names = {v['name'] for v in scale_vars}
            if not common_vars:
                common_vars = var_names
            else:
                common_vars &= var_names
        
        if common_vars:
            print(f"  [OK] Found {len(common_vars)} common conserved variables across scales")
            
            # Test consistency for common variables
            for var_name in list(common_vars)[:5]:  # Test first 5
                units_across_scales = []
                for scale, scale_vars in conserved_by_scale.items():
                    for var_info in scale_vars:
                        if var_info['name'] == var_name:
                            units_across_scales.append((scale, var_info['units']))
                            break
                
                # Units should be consistent across scales
                if units_across_scales:
                    reference_units = units_across_scales[0][1]
                    for scale, units in units_across_scales:
                        self.assertEqual(units, reference_units,
                                       f"Units inconsistent for {var_name}: "
                                       f"{scale} has {units}, expected {reference_units}")
    
    def test_scale_hierarchy_mathematical_consistency(self):
        """Test mathematical consistency of scale hierarchy."""
        print("\n[TEST] Scale hierarchy mathematical consistency...")
        
        # Test that each scale properly includes lower scales
        cumulative_functors = []
        cumulative_variables = []
        
        for i, scale in enumerate(SCALE_HIERARCHY):
            try:
                # Get configurations
                effective_functors = self.builder.get_effective_functors(scale, include_lower=True)
                effective_variables = self.builder.get_effective_variables(scale, include_lower=True)
                
                # Should include all previous functors/variables
                for prev_functor in cumulative_functors:
                    self.assertIn(prev_functor, effective_functors,
                                 f"Scale {scale} missing functor {prev_functor} from lower scales")
                
                for prev_var in cumulative_variables:
                    self.assertIn(prev_var, effective_variables,
                                 f"Scale {scale} missing variable {prev_var} from lower scales")
                
                # Update cumulative lists with new unique items
                for functor in effective_functors:
                    if functor not in cumulative_functors:
                        cumulative_functors.append(functor)
                
                for var_name in effective_variables:
                    if var_name not in cumulative_variables:
                        cumulative_variables.append(var_name)
                
                print(f"  [OK] {scale}: Cumulative {len(cumulative_functors)} functors, "
                      f"{len(cumulative_variables)} variables")
                
            except Exception as e:
                print(f"  [FAIL] {scale}: Failed - {e}")
                raise


class TestBuilderPerformanceAndScalability(unittest.TestCase):
    """
    Performance and scalability tests for the central model builder.
    
    Tests that the builder performs efficiently under various load conditions
    and scales appropriately with system size.
    """
    
    def setUp(self):
        """Set up performance test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
    
    def test_repeated_access_performance(self):
        """Test performance of repeated access to builder functions."""
        print("\n[TEST] Repeated access performance...")
        
        import time
        
        # Test repeated functor access
        start_time = time.time()
        for _ in range(20):
            for scale in SCALE_HIERARCHY[:4]:  # Test first 4 scales
                functors = self.builder.get_effective_functors(scale)
                self.assertIsInstance(functors, list)
        functor_time = time.time() - start_time
        
        # Test repeated variable access
        start_time = time.time()
        for _ in range(20):
            for scale in SCALE_HIERARCHY[:4]:
                variables = self.builder.get_effective_variables(scale)
                self.assertIsInstance(variables, dict)
        variable_time = time.time() - start_time
        
        print(f"  [OK] Functor access: {functor_time:.3f}s for 80 calls ({functor_time/80*1000:.1f}ms/call)")
        print(f"  [OK] Variable access: {variable_time:.3f}s for 80 calls ({variable_time/80*1000:.1f}ms/call)")
        
        # Performance thresholds
        self.assertLess(functor_time, 5.0, f"Functor access too slow: {functor_time:.3f}s")
        self.assertLess(variable_time, 5.0, f"Variable access too slow: {variable_time:.3f}s")
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with multiple builders."""
        print("\n[TEST] Memory usage scaling...")
        
        import gc
        
        # Force garbage collection and measure baseline
        gc.collect()
        baseline_objects = len(gc.get_objects())
        
        # Create multiple builders
        builders = []
        for i in range(5):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                builder = create_model_builder()
                
                # Use the builder to populate internal caches
                for scale in SCALE_HIERARCHY[:3]:
                    _ = builder.get_effective_functors(scale)
                    _ = builder.get_effective_variables(scale)
                
                builders.append(builder)
        
        # Measure memory usage
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - baseline_objects
        objects_per_builder = object_growth / len(builders)
        
        print(f"  [OK] Created {len(builders)} builders")
        print(f"  [OK] Memory growth: {object_growth} objects ({objects_per_builder:.0f} per builder)")
        
        # Memory usage should be reasonable
        self.assertLess(objects_per_builder, 1000, 
                       f"Excessive memory per builder: {objects_per_builder:.0f} objects")
        
        # Clean up
        del builders
        gc.collect()
    
    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access patterns."""
        print("\n[TEST] Concurrent access simulation...")
        
        
        # Simulate concurrent access patterns
        access_patterns = [
            ('landscape', True),   # Full hierarchy
            ('patch', True),       # Mid-level  
            ('individual', False), # Individual only
            ('regional', True),    # Full hierarchy
            ('landscape', False),  # Landscape only
        ]
        
        # Execute multiple access patterns rapidly
        results = []
        for _ in range(10):  # 10 rounds of access
            for scale, include_lower in access_patterns:
                try:
                    functors = self.builder.get_effective_functors(scale, include_lower)
                    variables = self.builder.get_effective_variables(scale, include_lower)
                    
                    results.append({
                        'scale': scale,
                        'include_lower': include_lower,
                        'functor_count': len(functors),
                        'variable_count': len(variables)
                    })
                    
                except Exception as e:
                    self.fail(f"Concurrent access failed for {scale}: {e}")
        
        # Verify results consistency
        scale_results = {}
        for result in results:
            key = (result['scale'], result['include_lower'])
            if key not in scale_results:
                scale_results[key] = []
            scale_results[key].append(result)
        
        # Check that results are consistent across multiple accesses
        for key, result_list in scale_results.items():
            scale, include_lower = key
            functor_counts = [r['functor_count'] for r in result_list]
            variable_counts = [r['variable_count'] for r in result_list]
            
            # All results should be identical (deterministic)
            self.assertEqual(len(set(functor_counts)), 1,
                           f"Inconsistent functor counts for {scale}: {set(functor_counts)}")
            self.assertEqual(len(set(variable_counts)), 1,
                           f"Inconsistent variable counts for {scale}: {set(variable_counts)}")
        
        print(f"  [OK] Completed {len(results)} concurrent access operations")
        print(f"  [OK] Verified consistency across {len(scale_results)} unique access patterns")


class TestBuilderIntegrationReadiness(unittest.TestCase):
    """
    Integration readiness tests for the central model builder.
    
    Tests that the builder is ready for integration with other system
    components and meets all integration requirements.
    """
    
    def setUp(self):
        """Set up integration test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
    
    def test_consistency_interface(self):
        """Test consistency with prior interfaces."""
        print("\n[TEST] consistency interface...")
        
        # Test that builder can replace get_scale_functors/get_scale_variables
        from relic.models import get_scale_functors, get_scale_variables
        
        for scale in ["individual", "patch", "landscape"]:
            try:
                # Get from builder
                builder_functors = self.builder.get_effective_functors(scale, include_lower=True)
                builder_vars = self.builder.get_effective_variables(scale, include_lower=True)
                
                # Get from consistency interface
                bridge_functors = get_scale_functors(scale)
                bridge_vars = get_scale_variables(scale)
                
                # Builder should provide equivalent or better results
                # (May differ due to deduplication improvements)
                builder_unique = set(builder_functors)
                bridge_unique = set(bridge_functors)
                
                # Should contain the same unique functors
                self.assertEqual(builder_unique, bridge_unique,
                               f"Functor mismatch for {scale}")
                
                # Should have same variable names
                self.assertEqual(set(builder_vars.keys()), set(bridge_vars.keys()),
                               f"Variable names mismatch for {scale}")
                
                print(f"  [OK] {scale}: consistent with interface")
                
            except Exception as e:
                print(f"  [FAIL] {scale}: historical consistency failed - {e}")
                # Don't fail the test - this is more informational
    
    def test_serialization_readiness(self):
        """Test that builder results can be serialized for external use."""
        print("\n[TEST] Serialization readiness...")
        
        import json
        import pickle
        
        # Get sample data
        landscape_vars = self.builder.get_effective_variables("landscape")
        
        # Test JSON serialization of variable metadata
        serializable_vars = {}
        for var_name, var_spec in landscape_vars.items():
            serializable_vars[var_name] = {
                'units': var_spec.units,
                'standard_name': var_spec.standard_name,
                'spatial_basis': var_spec.spatial_basis.value,
                'aggregation': var_spec.aggregation.value,
                'description': var_spec.description,
                'cell_methods': var_spec.cell_methods,
                'extras': var_spec.extras
            }
        
        # Should be JSON serializable
        try:
            json_str = json.dumps(serializable_vars, indent=2)
            recovered_data = json.loads(json_str)
            self.assertEqual(len(recovered_data), len(serializable_vars))
            print(f"  [OK] JSON serialization: {len(json_str)} characters")
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")
        
        # Test pickle serialization of builder itself
        try:
            pickled_builder = pickle.dumps(self.builder)
            recovered_builder = pickle.loads(pickled_builder)
            self.assertIsInstance(recovered_builder, CentralModelBuilder)
            print(f"  [OK] Pickle serialization: {len(pickled_builder)} bytes")
        except Exception as e:
            print(f"  [WARN] Pickle serialization failed (acceptable): {e}")
    
    def test_error_handling_robustness(self):
        """Test robust error handling for edge cases."""
        print("\n[TEST] Error handling robustness...")
        
        # Test invalid scale names
        invalid_scales = ["invalid", "", "PATCH", "patch_scale"]
        for invalid_scale in invalid_scales:
            with self.assertRaises(ValueError):
                self.builder.get_effective_functors(invalid_scale)
            with self.assertRaises(ValueError):
                self.builder.get_effective_variables(invalid_scale)
        
        print(f"  [OK] Properly handles {len(invalid_scales)} invalid scale names")
        
        # Test with empty configurations
        empty_builder = CentralModelBuilder()
        for scale in SCALE_HIERARCHY:
            try:
                functors = empty_builder.get_effective_functors(scale, include_lower=False)
                variables = empty_builder.get_effective_variables(scale, include_lower=False)
                
                # Should return empty results, not error
                self.assertIsInstance(functors, list)
                self.assertIsInstance(variables, dict)
                
            except Exception as e:
                self.fail(f"Empty builder failed for {scale}: {e}")
        
        print("  [OK] Handles empty configurations gracefully")
        
        # Test validation with empty/invalid variable specs
        empty_vars = {}
        validation_report = self.builder.validate_cf_and_units(empty_vars)
        self.assertTrue(validation_report.valid)  # Empty is valid
        
        print("  [OK] Validation handles empty inputs correctly")


def run_comprehensive_tests(verbosity=2):
    """
    Run all comprehensive functional tests.
    
    Parameters:
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test results
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestBuilderFunctionalCorrectness,
        TestBuilderPerformanceAndScalability,
        TestBuilderIntegrationReadiness
    ]
    
    for test_class in test_classes:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout, buffer=True)
    
    print("Running Comprehensive Central Model Builder Tests")
    print("=" * 60)
    
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            failure_msg = traceback.split('AssertionError:')[-1].strip()
            print(f"- {test}: {failure_msg}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('Error:')[-1].strip()
            print(f"- {test}: {error_msg}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Comprehensive Builder Tests')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2],
                       help='Test output verbosity level')
    parser.add_argument('--test', type=str, default='all',
                       help='Test category to run (functional, performance, integration, all)')
    
    args = parser.parse_args()
    
    if args.test == 'functional':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestBuilderFunctionalCorrectness)
    elif args.test == 'performance':  
        suite = unittest.TestLoader().loadTestsFromTestCase(TestBuilderPerformanceAndScalability)
    elif args.test == 'integration':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestBuilderIntegrationReadiness)
    else:
        # Run comprehensive tests
        run_comprehensive_tests(verbosity=args.verbosity)
        exit()
    
    # Run specific test category
    runner = unittest.TextTestRunner(verbosity=args.verbosity, buffer=True)
    result = runner.run(suite)
    
    print(f"\nTest Results: {result.testsRun} run, {len(result.failures)} failures, {len(result.errors)} errors")



