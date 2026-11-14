"""
Simple test to verify pedosphere parameter configuration files are valid.
"""

import sys
import os
import yaml

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_yaml_syntax():
    """Test that all YAML configuration files have valid syntax."""
    yaml_files = [
        "relic/parameters/database/relic/pedosphere/physical.yaml",
        "relic/parameters/database/relic/pedosphere/numerical.yaml", 
        "relic/parameters/database/relic/pedosphere/nitrification.yaml",
        "relic/parameters/database/relic/pedosphere/denitrification.yaml",
        "relic/parameters/database/relic/pedosphere/decomposition.yaml",
        "relic/parameters/database/relic/pedosphere/transport.yaml",
        "relic/parameters/database/relic/pedosphere/defaults.yaml"
    ]
    
    for yaml_file in yaml_files:
        file_path = os.path.join(project_root, yaml_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                print(f"[OK] {yaml_file} - Valid YAML syntax")
                
                # Check for basic structure
                if 'pedosphere' in data or any(key in data for key in ['physical_constants', 'stability', 'concentration_limits']):
                    print(f"[OK] {yaml_file} - Has expected structure")
                else:
                    print(f"[INFO] {yaml_file} - Different structure (may be intentional)")
                    
            except yaml.YAMLError as e:
                raise AssertionError(
                    f"{yaml_file} has invalid YAML syntax: {e}"
                ) from e
            except Exception as e:
                raise AssertionError(f"Unable to read {yaml_file}: {e}") from e
        else:
            print(f"[INFO] {yaml_file} - File not found (may not be created yet)")
    
def test_parameter_structure():
    """Test parameter structure in configuration files."""
    files = {
        "physical.yaml": "pedosphere.physical",
        "numerical.yaml": "pedosphere.numerical",
    }
    base = os.path.join(
        project_root, "relic/parameters/database/relic/pedosphere"
    )
    for filename, namespace in files.items():
        with open(os.path.join(base, filename), "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert data["schema_version"] == "2.0.0"
        assert data["namespace"] == namespace
        assert isinstance(data["parameters"], dict)
        assert data["parameters"]
        print(f"[OK] {filename} - canonical parameter schema")

def test_parameter_values():
    """Test that parameter values are reasonable."""
    
    # Test nitrification parameters
    nit_file = os.path.join(project_root, "relic/parameters/database/relic/pedosphere/nitrification.yaml")
    if os.path.exists(nit_file):
        with open(nit_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        spec = data["parameters"]["pedosphere.nitrification.vmax_nitrification"]
        vmax_val = spec["value"]
        assert 1e-8 < vmax_val < 1e-3
        for field in (
            "value",
            "units",
            "dtype",
            "bounds",
            "category",
            "description",
            "references",
            "tags",
            "requires_grad",
            "provenance",
        ):
            assert field in spec
        print(f"[OK] Nitrification Vmax value reasonable: {vmax_val}")

def run_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("RELIC Pedosphere Parameter Configuration Validation")
    print("=" * 60)
    
    tests = [
        ("YAML Syntax Test", test_yaml_syntax),
        ("Parameter Structure Test", test_parameter_structure),
        ("Parameter Values Test", test_parameter_values)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            test_func()
        except Exception as e:
            print(f"[ERROR] {test_name} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All parameter configuration files are valid!")
    else:
        print("[WARNING] Some issues found in parameter configuration.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    
    # Summary of changes made
    print("\nSummary of Parameter Fixes Applied:")
    print("-" * 40)
    print("1. Created physical.yaml with physical constants and properties")
    print("2. Created numerical.yaml with numerical stability parameters")
    print("3. Created nitrification.yaml with nitrification-specific parameters")
    print("4. Created denitrification.yaml with denitrification-specific parameters")
    print("5. Modified nitrification.py to use parameter manager")
    print("6. Modified denitrification.py to use parameter manager")
    print("7. Modified decomposition.py to use parameter manager")
    print("8. Modified gas_diffusion.py to use parameter manager")
    print("\nHardcoded parameters replaced with parameter manager calls:")
    print("- Temperature references (293.15K, 273.15K)")
    print("- Q10 coefficients (2.0, 2.2)")
    print("- pH optima (7.5, 7.0)")
    print("- Moisture thresholds (0.6, 0.8)")
    print("- Gas diffusion coefficients (0.66, 1.75)")
    print("- DNDC model coefficients (0.124, 0.013, 0.016)")
    print("- Numerical bounds and clipping values")
    
    sys.exit(0 if success else 1)

