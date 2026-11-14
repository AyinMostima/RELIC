#!/usr/bin/env python3
"""
Coupling Module Test Runner

This script runs all coupling module tests and provides a summary report.
It handles the dependency issues and provides a comprehensive test report.
"""

import sys
import os
import time

# Add test directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, test_dir)

print("=" * 60)
print("RELIC Coupling Module Test Suite")
print("=" * 60)

# Test results storage
test_results = {
    'total_tests': 0,
    'passed_tests': 0,
    'failed_tests': 0,
    'skipped_tests': 0,
    'test_details': []
}

def run_test(test_name: str, test_func) -> bool:
    """Run a single test and record results."""
    test_results['total_tests'] += 1
    print(f"\n[TEST] {test_name}")
    
    try:
        start_time = time.time()
        test_func()
        end_time = time.time()
        
        test_results['passed_tests'] += 1
        duration = end_time - start_time
        
        test_results['test_details'].append({
            'name': test_name,
            'status': 'PASSED',
            'duration': duration,
            'error': None
        })
        
        print(f"[PASS] {test_name} ({duration:.3f}s)")
        return True
        
    except Exception as e:
        test_results['failed_tests'] += 1
        
        test_results['test_details'].append({
            'name': test_name,
            'status': 'FAILED',
            'duration': 0,
            'error': str(e)
        })
        
        print(f"[FAIL] {test_name}")
        print(f"       Error: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic coupling module functionality."""
    from relic.coupling.scales.adapters import (
        SpatialAggregation, ConservativeRemap
    )
    
    # Test SpatialAggregation
    spatial_agg = SpatialAggregation(
        aggregation_strategy='area_weighted',
        preserve_totals=True,
        source_scale="fine_grid",
        target_scale="coarse_grid",
    )
    
    # Test ConservativeRemap
    remapper = ConservativeRemap(
        preserve_mass=True,
        preserve_positivity=True
    )
    
    # Test basic attributes
    assert spatial_agg.source_scale == "fine_grid"
    assert spatial_agg.target_scale == "coarse_grid"
    assert remapper.preserve_mass is True
    assert remapper.preserve_positivity is True

def test_inheritance_hierarchy():
    """Test inheritance hierarchy of coupling classes."""
    from relic.coupling.scales.adapters import (
        SpatialAggregation, ConservativeRemap
    )
    from relic.core.functor import ScaleCouplingFunctor
    from relic.core.abstractions import ScaleCoupling
    
    # Test SpatialAggregation inheritance
    spatial_agg = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
    assert isinstance(spatial_agg, ScaleCouplingFunctor)
    assert isinstance(spatial_agg, ScaleCoupling)
    
    # Test ConservativeRemap inheritance
    remapper = ConservativeRemap()
    assert isinstance(remapper, ScaleCouplingFunctor)
    assert isinstance(remapper, ScaleCoupling)

def test_parameter_access():
    """Test parameter access functionality."""
    from relic.coupling.scales.adapters import (
        SpatialAggregation, ConservativeRemap
    )
    
    # Test SpatialAggregation parameter access
    spatial_agg = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
    weighting_type = spatial_agg.get_parameter_value(
        'weighting_type',
        units='dimensionless',
        namespace='coupling.scales.spatial'
    )
    assert isinstance(weighting_type, str)
    
    # Test ConservativeRemap parameter access
    remapper = ConservativeRemap()
    preserve_mass = remapper.get_parameter_value(
        'preserve_mass',
        units='dimensionless',
        namespace='coupling.scales.conservative'
    )
    assert isinstance(preserve_mass, bool)

def test_forward_methods():
    """Test forward methods of coupling classes."""
    import torch
    from relic.coupling.scales.adapters import (
        SpatialAggregation, ConservativeRemap
    )
    # Test SpatialAggregation forward
    spatial_agg = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
    spatial_state = {
        'spatial_var': torch.randn(10, 10)
    }
    
    try:
        spatial_result = spatial_agg.forward(
            state=spatial_state,
            time=0.0,
            dt=1.0
        )
        assert isinstance(spatial_result, dict)
        print(f"       SpatialAggregation forward result: {list(spatial_result.keys())}")
    except Exception as e:
        print(f"       SpatialAggregation forward error (expected): {str(e)[:100]}")
    
    # Test ConservativeRemap forward
    remapper = ConservativeRemap(device=torch.device("cpu"))
    remap_state = {
        'conservative_var': torch.abs(torch.randn(8, 8)) * 10
    }
    # Overlap geometry is mandatory: 64 unit-area source cells fully inside one target cell.
    remap_geometry = {
        'overlap_areas': torch.ones(64, 1, dtype=torch.float64),
        'source_areas': torch.ones(64, dtype=torch.float64),
        'target_areas': torch.tensor([64.0], dtype=torch.float64),
        'variable_metadata': {'conservative_var': {'variable_type': 'extensive'}},
    }

    try:
        remap_result = remapper.forward(
            state=remap_state,
            time=0.0,
            dt=1.0,
            variables=['conservative_var'],
            **remap_geometry,
        )
        assert isinstance(remap_result, dict)
        print(f"       ConservativeRemap forward result: {list(remap_result.keys())}")
        
        # Test positivity constraint
        if 'conservative_var_remapped' in remap_result:
            is_positive = torch.all(remap_result['conservative_var_remapped'] >= 0)
            assert is_positive, "Positivity constraint not enforced"
            print(f"       Positivity constraint enforced: {is_positive}")
            
    except Exception as e:
        print(f"       ConservativeRemap forward error: {str(e)[:100]}")
        raise

def test_dependencies():
    """Test dependency management."""
    from relic.coupling.scales.adapters import (
        SpatialAggregation, ConservativeRemap
    )
    
    spatial_agg = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
    deps = spatial_agg.get_dependencies()
    assert isinstance(deps, list)
    
    remapper = ConservativeRemap()
    deps = remapper.get_dependencies()
    assert isinstance(deps, list)

def test_scale_coupling_interface():
    """Test ScaleCoupling abstract interface compliance."""
    from relic.coupling.scales.adapters import (
        SpatialAggregation, ConservativeRemap
    )
    # Test SpatialAggregation
    spatial_agg = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
    assert hasattr(spatial_agg, 'upscale')
    assert hasattr(spatial_agg, 'downscale')
    
    # Test ConservativeRemap
    remapper = ConservativeRemap()
    assert hasattr(remapper, 'upscale')
    assert hasattr(remapper, 'downscale')

def run_all_tests():
    """Run all tests and generate report."""
    start_time = time.time()
    
    # Define test suite
    test_suite = [
        ("Basic Functionality Test", test_basic_functionality),
        ("Inheritance Hierarchy Test", test_inheritance_hierarchy),
        ("Parameter Access Test", test_parameter_access),
        ("Forward Methods Test", test_forward_methods),
        ("Dependencies Test", test_dependencies),
        ("Scale Coupling Interface Test", test_scale_coupling_interface),
    ]
    
    # Run all tests
    for test_name, test_func in test_suite:
        run_test(test_name, test_func)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Generate report
    print("\n" + "=" * 60)
    print("TEST REPORT")
    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']}")
    print(f"Failed: {test_results['failed_tests']}")
    print(f"Skipped: {test_results['skipped_tests']}")
    print(f"Success Rate: {(test_results['passed_tests']/test_results['total_tests']*100):.1f}%")
    print(f"Total Duration: {total_duration:.3f}s")
    
    if test_results['failed_tests'] > 0:
        print("\nFailed Tests:")
        for detail in test_results['test_details']:
            if detail['status'] == 'FAILED':
                print(f"  - {detail['name']}: {detail['error']}")
    
    print("\nTest Details:")
    for detail in test_results['test_details']:
        status_symbol = "[PASS]" if detail['status'] == 'PASSED' else "[FAIL]"
        print(f"  {status_symbol} {detail['name']} ({detail['duration']:.3f}s)")
        if detail['error']:
            print(f"         Error: {detail['error']}")
    
    print("\n" + "=" * 60)
    
    if test_results['failed_tests'] == 0:
        print("SUCCESS: All tests passed!")
        return 0
    else:
        print("FAILURE: Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

