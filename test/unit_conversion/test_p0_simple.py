"""Simple verification test for core unit conversion fixes."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from relic.data.standards.unit_converter import UnitConverter, ConversionContext

def test_all_types():
    """Test all input types work correctly."""
    converter = UnitConverter()
    context = ConversionContext()
    
    print("\n[TEST] Testing Tensor input...")
    temp_tensor = torch.tensor([25.0])
    result = converter.convert(temp_tensor, 'air_temperature', 'celsius', 'K', context)
    print(f"  Input: {temp_tensor} (type: {type(temp_tensor)})")
    print(f"  Output: {result} (type: {type(result)})")
    assert isinstance(result, torch.Tensor), "FAIL: Tensor type not preserved"
    print("  [OK] Tensor type preserved")
    
    print("\n[TEST] Testing float input...")
    result = converter.convert(25.0, 'air_temperature', 'celsius', 'K', context)
    print(f"  Input: 25.0 (type: float)")
    print(f"  Output: {result} (type: {type(result)})")
    assert isinstance(result, float), "FAIL: float type not preserved"
    assert abs(result - 298.15) < 1e-5, f"FAIL: Incorrect value {result}"
    print("  [OK] float type preserved and value correct")
    
    print("\n[TEST] Testing numpy input...")
    arr = np.array([25.0])
    result = converter.convert(arr, 'air_temperature', 'celsius', 'K', context)
    print(f"  Input: {arr} (type: {type(arr)})")
    print(f"  Output: {result} (type: {type(result)})")
    assert isinstance(result, np.ndarray), "FAIL: numpy type not preserved"
    print("  [OK] numpy type preserved")
    
    print("\n[TEST] Testing direct conversion...")
    result = converter.convert(100.0, 'generic', 'm', 'km', context)
    print(f"  convert(100.0, 'm', 'km') = {result}")
    assert abs(result - 0.1) < 1e-5, f"FAIL: Incorrect value {result}"
    print("  [OK] Legacy wrapper works")
    
    print("\n[SUCCESS] All core verification tests passed!")

if __name__ == '__main__':
    test_all_types()

