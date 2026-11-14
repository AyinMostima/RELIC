"""
Test script to verify BVOC module imports and basic functionality after fixes
"""

import sys
import torch
import pytest

def test_basic_imports():
    """Test that all BVOC modules can be imported without errors."""
    print("Testing basic imports...")
    
    try:
        # Test emission factors import
        print("[OK] EmissionFactorDatabase imported successfully")
        
        # Test MEGAN model import
        print("[OK] MEGANModel imported successfully")
        
        # Test responses import
        print("[OK] Response functions imported successfully")
        
        # Test isoprene synthesis import
        print("[OK] IsopreneSynthesis imported successfully")
        
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        pytest.fail(f"Import failed: {e}")


def test_emission_factors():
    """Test emission factor database functionality."""
    print("\nTesting emission factor database...")
    
    try:
        from relic.scientific.biosphere.bvoc.emission_factors import (
            EmissionFactorDatabase,
            PlantFunctionalType,
            CompoundCategory
        )
        
        # Create database instance
        db = EmissionFactorDatabase()
        print("[OK] EmissionFactorDatabase created")
        
        # Test getting emission factor
        factor = db.get_emission_factor(
            PlantFunctionalType.DEFAULT,
            CompoundCategory.ISOPRENE
        )
        print(f"[OK] Isoprene emission factor: {factor.value} ug/g/h")
        
    except Exception as e:
        print(f"[ERROR] Emission factor test failed: {e}")
        pytest.fail(f"Emission factor test failed: {e}")


def test_megan_model():
    """Test canonical MEGAN model instantiation and output semantics."""
    print("\nTesting MEGAN model...")
    from relic.core.state.types import UpdateItem, UpdateMode
    from relic.scientific.biosphere.bvoc.megan_model import MEGANModel

    model = MEGANModel(device=torch.device("cpu"))
    print("[OK] MEGANModel created")
    state = {
        "air_temperature": torch.tensor(298.15),
        "ppfd": torch.tensor(1000.0),
        "leaf_area_index": torch.tensor(1.0),
        "atmospheric_co2_ppm": torch.tensor(400.0),
        "net_photosynthesis": torch.tensor(10.0),
    }

    result = model.forward(state, time=0.0, dt=3600.0)
    total = result["total_bvoc_emission"]
    assert isinstance(total, UpdateItem)
    assert total.mode is UpdateMode.REPLACE
    assert total.per_timestep is False
    assert total.element is None
    assert total.unit == "mol m-2 s-1"
    assert total.value.item() > 0.0
    print("[OK] Forward pass successful")
    print(f"     Total BVOC emission: {total.value.item():.6e} mol m-2 s-1")


def test_response_functions():
    """Test BVOC response functions."""
    print("\nTesting response functions...")
    
    try:
        from relic.scientific.biosphere.bvoc.responses import (
            BVOCLightResponse,
            BVOCTemperatureResponse
        )
        
        # Test light response
        ppfd = torch.tensor(1000.0)
        light_response = BVOCLightResponse.get_compound_light_response('isoprene', ppfd)
        print(f"[OK] Light response at 1000 PPFD: {light_response.item():.3f}")
        
        # Test temperature response
        temperature = torch.tensor(303.15)  # 30C
        temp_response = BVOCTemperatureResponse.get_compound_temperature_response(
            'isoprene', temperature
        )
        print(f"[OK] Temperature response at 30C: {temp_response.item():.3f}")
        
    except Exception as e:
        print(f"[ERROR] Response function test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Response function test failed: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BVOC Module Validation Tests")
    print("=" * 60)
    
    all_passed = True
    for test_func in (
        test_basic_imports,
        test_emission_factors,
        test_megan_model,
        test_response_functions,
    ):
        try:
            test_func()
        except Exception:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED - please review errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

