#!/usr/bin/env python3
"""
Simple instantiation test for anthroposphere modules

This test checks if all anthroposphere classes can be instantiated
after the inheritance refactoring, focusing on the core functionality.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from test.utils.functor_minimal_construction import construct_minimal

def test_instantiation():
    """Test basic instantiation of all anthroposphere classes."""
    
    # Test data
    test_classes = [
        # PhysicalField classes
        ("LandUseParameterMapping", "relic.scientific.anthroposphere.land_use.mapping", "PhysicalField"),
        
        # BiologicalProcess classes  
        ("GrazingManagement", "relic.scientific.anthroposphere.management.grazing", "BiologicalProcess"),
        ("HarvestManagement", "relic.scientific.anthroposphere.management.harvest", "BiologicalProcess"),
        ("FireManagement", "relic.scientific.anthroposphere.management.fire_management", "BiologicalProcess"),
        
        # TransportMechanism classes
        ("FertilizationManagement", "relic.scientific.anthroposphere.management.fertilization", "TransportMechanism"),
        ("IrrigationManagement", "relic.scientific.anthroposphere.management.irrigation", "TransportMechanism"),
        ("TillageManagement", "relic.scientific.anthroposphere.management.tillage", "TransportMechanism"),
        
        # ScaleCoupling classes
        ("SectorAllocation", "relic.scientific.anthroposphere.emissions.sector_allocation", "ScaleCoupling"),
        ("LandUseTransitions", "relic.scientific.anthroposphere.land_use.transitions", "ScaleCoupling"),
    ]
    
    results = {"success": [], "failed": []}
    
    print("Testing anthroposphere class instantiation...")
    print("=" * 60)
    
    for class_name, module_name, base_type in test_classes:
        try:
            # Import the module
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            # Try to instantiate
            instance = construct_minimal(f"{module_name}:{class_name}")
            
            # Basic checks
            assert instance is not None
            assert hasattr(instance, '__class__')
            
            # Check inheritance
            from relic.core.abstractions import PhysicalField, BiologicalProcess, TransportMechanism, ScaleCoupling
            
            base_class_map = {
                "PhysicalField": PhysicalField,
                "BiologicalProcess": BiologicalProcess, 
                "TransportMechanism": TransportMechanism,
                "ScaleCoupling": ScaleCoupling
            }
            
            expected_base = base_class_map[base_type]
            assert isinstance(instance, expected_base), f"{class_name} should inherit from {base_type}"
            
            # Test basic method existence
            if base_type == "PhysicalField":
                assert hasattr(instance, 'get_value_at')
                assert hasattr(instance, 'compute_gradient')
            elif base_type == "BiologicalProcess":
                assert hasattr(instance, 'forward')
            elif base_type == "TransportMechanism":
                assert hasattr(instance, 'forward')
            elif base_type == "ScaleCoupling":
                assert hasattr(instance, 'upscale')
                assert hasattr(instance, 'downscale')
            
            # Test forward method (backward consistency)
            assert hasattr(instance, 'forward'), f"{class_name} missing forward method"
            
            print(f"[OK] {class_name}: Instantiation and method check successful")
            results["success"].append(class_name)
            
        except Exception as e:
            print(f"[FAIL] {class_name}: {e}")
            results["failed"].append((class_name, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("INSTANTIATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total classes tested: {len(test_classes)}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['success']:
        print(f"\nSuccessful classes: {', '.join(results['success'])}")
    
    if results['failed']:
        print("\nFailed classes:")
        for class_name, error in results['failed']:
            print(f"  - {class_name}: {error}")
    success = len(results["failed"]) == 0
    assert success, "Anthroposphere instantiation failures detected"

if __name__ == "__main__":
    try:
        test_instantiation()
        exit(0)
    except AssertionError:
        exit(1)


