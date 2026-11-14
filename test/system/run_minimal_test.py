#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Minimal L3 Test with Environment Setup

This script properly sets up the environment and runs the minimal L3 test.
"""

import os
import sys
from pathlib import Path

# Fix OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import and run the test (package-aware)
from .test_minimal_l3 import TestMinimalL3System
import tempfile

def main():
    print("Running Minimal L3 System Tests")
    print("=" * 50)
    
    # Create a temporary directory for this run
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Create test instance
        test_instance = TestMinimalL3System()
        
        try:
            print("\n1. Daily Cycle Simulation")
            test_instance.test_complete_daily_cycle_simulation(temp_path)
            
            print("\n2. Multi-Site Ecosystem Comparison")
            test_instance.test_multi_site_ecosystem_comparison(temp_path)
            
            print("\n3. Extreme Conditions Stress Testing")
            test_instance.test_extreme_conditions_stress_testing(temp_path)
            
            print("\n4. Performance and Scalability")
            test_instance.test_performance_and_scalability(temp_path)
            
            print("\n5. Scientific Accuracy Validation")
            test_instance.test_scientific_accuracy_validation(temp_path)
            
            print("\n" + "=" * 50)
            print("[SUCCESS] All Minimal L3 System Tests PASSED!")
            print("\nThis demonstrates that the L3 testing framework concepts are working correctly.")
            print("The full L3 test suite with actual RELIC modules is available in test_end_to_end.py")
            print("\nKey L3 Testing Concepts Validated:")
            print("- Complete ecosystem simulation workflows")
            print("- Multi-site comparison and scalability")
            print("- Extreme condition stress testing")
            print("- Performance benchmarking and profiling")
            print("- Scientific accuracy validation")
            print("- Automated result analysis and reporting")
            
            return 0
            
        except Exception as e:
            print(f"\n[ERROR] Minimal L3 test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

