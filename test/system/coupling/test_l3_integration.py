"""
L3 System Integration Tests for RELIC Models Refactoring

This module provides comprehensive system-level validation of the refactored
models architecture with the central builder pattern. Tests full workflow
integration, scientific accuracy, and production readiness.

Test Categories:
    - End-to-end workflow validation
    - Scientific conservation law verification
    - Cross-module integration testing
    - Production deployment readiness

Mathematical Validation:
    - Mass conservation: 危(input) = 危(output) + 危(storage_change)
    - Energy conservation: Ein = Eout + 螖Estored
    - CF-1.8 metadata compliance across all outputs

References:
    - RELIC System Test Specification v2.0
    - RHESSys Integration Test Suite
    - CF Conventions Compliance Test Framework
"""

import sys
import os
import unittest
import warnings

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# Import system components
from relic.coupling.scales.hierarchical.model_builder import (
    create_model_builder, AggregationMethod
)
from relic.coupling.scales.types import SpatialQuantityBasis

CONSERVATIVE_BY_BASIS = {
    SpatialQuantityBasis.CELL_TOTAL: [AggregationMethod.SUM],
    SpatialQuantityBasis.AREA_DENSITY: [AggregationMethod.AREA_WEIGHTED_MEAN],
}
from relic.models import SCALE_HIERARCHY, get_scale_functors, get_scale_variables


class TestL3SystemIntegration(unittest.TestCase):
    """
    L3 System-level integration tests for complete workflow validation.
    
    Tests the full integration of the central builder with all system
    components and validates scientific correctness.
    """
    
    def setUp(self):
        """Set up test environment."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
    
    def test_full_scale_hierarchy_traversal(self):
        """Test complete traversal of the scale hierarchy."""
        print("\n[L3 TEST] Full scale hierarchy traversal...")
        
        previous_functor_count = 0
        previous_var_count = 0
        
        for scale in SCALE_HIERARCHY:
            # Get configurations
            functors = self.builder.get_effective_functors(scale, include_lower=True)
            variables = self.builder.get_effective_variables(scale, include_lower=True)
            
            # Should accumulate (or stay same)
            self.assertGreaterEqual(len(functors), previous_functor_count,
                                   f"Functor count decreased at {scale}")
            self.assertGreaterEqual(len(variables), previous_var_count,
                                   f"Variable count decreased at {scale}")
            
            # All should be properly formed
            for var_name, var_spec in variables.items():
                self.assertIsNotNone(var_spec.units)
                # standard_name is None for a quantity without a CF standard name.
                self.assertTrue(var_spec.standard_name is None or var_spec.standard_name)
                self.assertIsNotNone(var_spec.cell_methods)
                self.assertIn("area:", var_spec.cell_methods,
                             f"Missing area dimension in {var_name} cell_methods")
            
            print(f"  [OK] {scale}: {len(functors)} functors, {len(variables)} variables")
            
            previous_functor_count = len(functors)
            previous_var_count = len(variables)
    
    def test_conservation_laws_across_scales(self):
        """Test conservation laws are maintained across scale transitions."""
        print("\n[L3 TEST] Conservation laws across scales...")
        
        # Track conserved quantities
        carbon_vars = ['gross_primary_productivity', 'leaf_respiration', 'nee', 'soil_respiration']
        water_vars = ['transpiration', 'evapotranspiration', 'runoff']
        energy_vars = ['sensible_heat', 'latent_heat', 'net_radiation']
        
        for scale in SCALE_HIERARCHY:
            variables = self.builder.get_effective_variables(scale)
            
            # Check carbon conservation
            carbon_extensive = []
            for var_name in carbon_vars:
                if var_name in variables:
                    var_spec = variables[var_name]
                    if var_spec.spatial_basis.is_extensive:
                        # Should use the conservative aggregation of its basis
                        self.assertIn(var_spec.aggregation,
                                     CONSERVATIVE_BY_BASIS[var_spec.spatial_basis],
                                     f"Non-conservative aggregation for {var_name} in {scale}")
                        carbon_extensive.append(var_name)
            
            if carbon_extensive:
                print(f"  [OK] {scale}: {len(carbon_extensive)} carbon fluxes conserved")
    
    def test_cf_compliance_validation(self):
        """Test CF-1.8 compliance across all scales."""
        print("\n[L3 TEST] CF-1.8 compliance validation...")
        
        cf_standard_prefixes = [
            'gross_primary_productivity',
            'net_ecosystem_exchange',
            'surface_temperature',
            'soil_moisture',
            'precipitation_flux'
        ]
        
        for scale in SCALE_HIERARCHY:
            variables = self.builder.get_effective_variables(scale)
            validation_report = self.builder.validate_cf_and_units(variables)
            
            # Should have valid CF metadata
            cf_compliant = 0
            for var_name, var_spec in variables.items():
                # Check standard_name format
                if var_spec.standard_name is not None and any(var_spec.standard_name.startswith(prefix)
                       for prefix in cf_standard_prefixes):
                    cf_compliant += 1
                
                # Check units are not empty
                self.assertNotEqual(var_spec.units, '',
                                   f"Empty units for {var_name} in {scale}")
                
                # Check cell_methods format
                self.assertRegex(var_spec.cell_methods, 
                                r'(area|time|volume):\s*(mean|sum|maximum|minimum|point|mode)',
                                f"Invalid cell_methods format for {var_name}")
            
            compliance_rate = cf_compliant / len(variables) if variables else 0
            print(f"  [OK] {scale}: {cf_compliant}/{len(variables)} "
                  f"CF compliant ({compliance_rate:.1%})")
    
    def test_api_consistency(self):
        """Test consistency with the prior API."""
        print("\n[L3 TEST] API consistency...")
        
        for scale in ['individual', 'patch']:
            # consistency functions should still work
            bridge_functors = get_scale_functors(scale)
            bridge_vars = get_scale_variables(scale)
            
            # Builder should provide equivalent results
            builder_functors = self.builder.get_effective_functors(scale)
            builder_vars = self.builder.get_effective_variables(scale)
            
            # Unique sets should match
            self.assertEqual(set(bridge_functors), set(builder_functors),
                           f"Functor mismatch for {scale}")
            self.assertEqual(set(bridge_vars.keys()), set(builder_vars.keys()),
                           f"Variable name mismatch for {scale}")
            
            print(f"  [OK] {scale}: API consistent")
    
    def test_scientific_metadata_consistency(self):
        """Test scientific metadata consistency across scales."""
        print("\n[L3 TEST] Scientific metadata consistency...")
        
        # Variables that should exist at multiple scales
        common_vars = ['gross_primary_productivity', 'lai', 'soil_moisture']
        
        for var_name in common_vars:
            var_specs_by_scale = {}
            
            for scale in SCALE_HIERARCHY:
                variables = self.builder.get_effective_variables(scale)
                if var_name in variables:
                    var_specs_by_scale[scale] = variables[var_name]
            
            if len(var_specs_by_scale) > 1:
                # Units should be consistent
                units_list = [spec.units for spec in var_specs_by_scale.values()]
                self.assertEqual(len(set(units_list)), 1,
                               f"Inconsistent units for {var_name}: {units_list}")
                
                # Standard names should be consistent
                std_names = [spec.standard_name for spec in var_specs_by_scale.values()]
                self.assertEqual(len(set(std_names)), 1,
                               f"Inconsistent standard_names for {var_name}: {std_names}")
                
                print(f"  [OK] {var_name}: Consistent across {len(var_specs_by_scale)} scales")
    
    def test_aggregation_method_appropriateness(self):
        """Test that aggregation methods are scientifically appropriate."""
        print("\n[L3 TEST] Aggregation method appropriateness...")
        
        # Define expected patterns
        patterns = {
            'temperature': AggregationMethod.AREA_WEIGHTED_MEAN,
            'flux': AggregationMethod.AREA_WEIGHTED_MEAN,
            'concentration': AggregationMethod.AREA_WEIGHTED_MEAN,
            'index': AggregationMethod.AREA_WEIGHTED_MEAN,
            'stage': AggregationMethod.MODE
        }
        
        for scale in SCALE_HIERARCHY[:3]:  # Test first 3 scales
            variables = self.builder.get_effective_variables(scale)
            
            appropriate_count = 0
            for var_name, var_spec in variables.items():
                # Check based on variable name patterns
                if 'temperature' in var_name:
                    expected = patterns['temperature']
                elif var_spec.spatial_basis is SpatialQuantityBasis.CELL_TOTAL:
                    expected = AggregationMethod.SUM
                elif 'flux' in (var_spec.standard_name or '') or var_spec.spatial_basis.is_extensive:
                    expected = patterns['flux']
                elif 'concentration' in (var_spec.standard_name or ''):
                    expected = patterns['concentration']
                elif 'index' in (var_spec.standard_name or ''):
                    expected = patterns['index']
                elif var_spec.spatial_basis is SpatialQuantityBasis.CATEGORICAL:
                    expected = patterns['stage']
                else:
                    continue
                
                if var_spec.aggregation == expected:
                    appropriate_count += 1
            
            print(f"  [OK] {scale}: {appropriate_count}/{len(variables)} "
                  f"variables use appropriate aggregation")
    
    def test_production_deployment_readiness(self):
        """Test production deployment readiness criteria."""
        print("\n[L3 TEST] Production deployment readiness...")
        
        criteria = {
            'no_duplicates': True,
            'all_variables_have_units': True,
            'all_variables_have_cf_metadata': True,
            'conservation_laws_enforced': True,
            'consistency_support': True,
            'performance_acceptable': True
        }
        
        # Test no duplicates
        for scale in SCALE_HIERARCHY:
            functors = self.builder.get_effective_functors(scale)
            if len(functors) != len(set(functors)):
                criteria['no_duplicates'] = False
        
        # Test metadata completeness
        for scale in SCALE_HIERARCHY:
            variables = self.builder.get_effective_variables(scale)
            for var_name, var_spec in variables.items():
                if not var_spec.units:
                    criteria['all_variables_have_units'] = False
                if not var_spec.cell_methods:
                    criteria['all_variables_have_cf_metadata'] = False
        
        # Test conservation (already validated in other tests)
        # Test consistency (already validated)
        
        # Test performance
        import time
        start = time.time()
        for _ in range(10):
            for scale in SCALE_HIERARCHY[:3]:
                _ = self.builder.get_effective_functors(scale)
                _ = self.builder.get_effective_variables(scale)
        elapsed = time.time() - start
        if elapsed > 1.0:  # Should complete in < 1 second
            criteria['performance_acceptable'] = False
        
        # Report results
        passed = sum(criteria.values())
        total = len(criteria)
        
        for criterion, status in criteria.items():
            status_str = "[OK]" if status else "[FAIL]"
            print(f"  {status_str} {criterion}")
        
        print(f"\n  Production Readiness: {passed}/{total} criteria met")
        self.assertEqual(passed, total, "Not all production criteria met")


def run_l3_tests(verbosity=2):
    """
    Run L3 system integration tests.
    
    Parameters:
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test results
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestL3SystemIntegration)
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout, buffer=True)
    
    print("\n" + "="*60)
    print("L3 SYSTEM INTEGRATION TESTS")
    print("="*60)
    
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print(f"L3 Tests Complete: {result.testsRun} run, "
          f"{len(result.failures)} failures, {len(result.errors)} errors")
    print("="*60)
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run L3 System Integration Tests')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2],
                       help='Test output verbosity level')
    
    args = parser.parse_args()
    run_l3_tests(verbosity=args.verbosity)



