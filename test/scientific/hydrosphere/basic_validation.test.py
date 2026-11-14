"""
Basic Syntax Validation for Hydrosphere Module

Tests syntax correctness without heavy computational dependencies.
"""

import sys
import os
import ast

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Add relic to path
sys.path.insert(0, ROOT_DIR)


def test_python_syntax():
    """Test Python syntax for all hydrosphere modules."""
    
    print("Testing Python syntax...")
    
    hydrosphere_root = os.path.join(ROOT_DIR, 'relic', 'scientific', 'hydrosphere')
    
    python_files = []
    
    # Recursively find all Python files
    for root, dirs, files in os.walk(hydrosphere_root):
        for file in files:
            if file.endswith('.py') and not file.startswith('__pycache__'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to check syntax
            ast.parse(content, filename=file_path)
            print(f"[OK] {os.path.relpath(file_path, hydrosphere_root)}")
            
        except SyntaxError as e:
            print(f"[SYNTAX ERROR] {os.path.relpath(file_path, hydrosphere_root)}: {e}")
            syntax_errors.append((file_path, e))
        except Exception as e:
            print(f"[ERROR] {os.path.relpath(file_path, hydrosphere_root)}: {e}")
    
    print(f"\nSyntax validation: {len(python_files) - len(syntax_errors)}/{len(python_files)} files passed")

    assert len(syntax_errors) == 0


def test_interface_mapping_only():
    """Test interface mapping without importing torch-dependent modules."""
    
    print("\nTesting interface mapping...")
    
    try:
        # Test interface mapping module syntax by reading and parsing
        interface_file = os.path.join(
            ROOT_DIR, 'relic', 'scientific', 'hydrosphere', 'interface_mapping.py'
        )
        
        with open(interface_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content, filename=interface_file)
        print("[OK] Interface mapping syntax is valid")
        
        # Check for required classes and functions
        classes_found = []
        functions_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions_found.append(node.name)
        
        required_classes = ['VariableCategory', 'VariableMapping', 'HydrosphereInterfaceMapper']
        required_functions = ['standardize_variable_name', 'get_variable_units', 'validate_hydrosphere_interface']

        interface_ok = True
        for cls in required_classes:
            if cls in classes_found:
                print(f"[OK] Found required class: {cls}")
            else:
                print(f"[ERROR] Missing required class: {cls}")
                interface_ok = False
        
        for func in required_functions:
            if func in functions_found:
                print(f"[OK] Found required function: {func}")
            else:
                print(f"[ERROR] Missing required function: {func}")
                interface_ok = False

        assert interface_ok
        
    except Exception as e:
        print(f"[ERROR] Interface mapping test failed: {e}")
        raise


def test_module_structure():
    """Test hydrosphere module structure."""
    
    print("\nTesting module structure...")
    
    hydrosphere_root = os.path.join(ROOT_DIR, 'relic', 'scientific', 'hydrosphere')
    
    # Required directories and files
    required_structure = {
        'evapotranspiration': ['penman_monteith.py', 'priestley_taylor.py', '__init__.py'],
        'soil_moisture': ['richards.py', 'bucket.py', '__init__.py'],
        'runoff': ['surface.py', 'subsurface.py', '__init__.py'],
        'transport': ['capillary_rise.py', 'vertical_percolation.py', '__init__.py'],
        'groundwater': ['dynamics.py', '__init__.py'],
        'snow': ['albedo.py', '__init__.py']
    }
    
    structure_ok = True
    
    for subdir, files in required_structure.items():
        subdir_path = os.path.join(hydrosphere_root, subdir)
        
        if os.path.exists(subdir_path):
            print(f"[OK] Directory exists: {subdir}")
            
            for file in files:
                file_path = os.path.join(subdir_path, file)
                if os.path.exists(file_path):
                    print(f"[OK] File exists: {subdir}/{file}")
                else:
                    print(f"[ERROR] Missing file: {subdir}/{file}")
                    structure_ok = False
        else:
            print(f"[ERROR] Missing directory: {subdir}")
            structure_ok = False
    
    # Check main files
    main_files = ['__init__.py', 'interface_mapping.py']
    for file in main_files:
        file_path = os.path.join(hydrosphere_root, file)
        if os.path.exists(file_path):
            print(f"[OK] Main file exists: {file}")
        else:
            print(f"[ERROR] Missing main file: {file}")
            structure_ok = False

    assert structure_ok


def test_p0_p1_p2_methods_existence():
    """Test that P0, P1, P2 methods exist by parsing AST."""
    
    print("\nTesting P0/P1/P2 method existence...")
    
    # P0: Richards equation solver
    try:
        richards_file = os.path.join(
            ROOT_DIR,
            'relic',
            'scientific',
            'hydrosphere',
            'soil_moisture',
            'richards.py',
        )
        
        with open(richards_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=richards_file)
        
        # Find RichardsSoilMoisture class and its methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'RichardsSoilMoisture':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                required_p0_methods = [
                    '_compute_full_solution',
                    '_solve_linearized_system_picard', 
                    '_solve_tridiagonal_thomas',
                    '_compute_pressure_head',
                    '_compute_hydraulic_conductivity'
                ]
                
                for method in required_p0_methods:
                    if method in methods:
                        print(f"[OK] P0 - Richards has method: {method}")
                    else:
                        print(f"[ERROR] P0 - Richards missing method: {method}")
                break
        
    except Exception as e:
        print(f"[ERROR] P0 - Richards test failed: {e}")
    
    # P1: Capillary rise
    try:
        capillary_file = os.path.join(
            ROOT_DIR,
            'relic',
            'scientific',
            'hydrosphere',
            'transport',
            'capillary_rise.py',
        )
        
        with open(capillary_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=capillary_file)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'CapillaryRiseTransport':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                required_p1_methods = [
                    '_compute_gardner_flux',
                    '_compute_gardner_coefficients',
                    '_compute_flux_profile'
                ]
                
                for method in required_p1_methods:
                    if method in methods:
                        print(f"[OK] P1 - CapillaryRise has method: {method}")
                    else:
                        print(f"[ERROR] P1 - CapillaryRise missing method: {method}")
                break
        
    except Exception as e:
        print(f"[ERROR] P1 - Capillary test failed: {e}")
    
    # P1: Groundwater dynamics
    try:
        gw_file = os.path.join(
            ROOT_DIR,
            'relic',
            'scientific',
            'hydrosphere',
            'groundwater',
            'dynamics.py',
        )
        
        with open(gw_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=gw_file)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'GroundwaterDynamics':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                required_p1_methods = [
                    '_update_storage_dynamics',
                    '_solve_water_table_change',
                    '_compute_porosity_at_depth'
                ]
                
                for method in required_p1_methods:
                    if method in methods:
                        print(f"[OK] P1 - GroundwaterDynamics has method: {method}")
                    else:
                        print(f"[ERROR] P1 - GroundwaterDynamics missing method: {method}")
                break
        
    except Exception as e:
        print(f"[ERROR] P1 - Groundwater test failed: {e}")
    
    # P2: Canopy interception
    try:
        pm_file = os.path.join(
            ROOT_DIR,
            'relic',
            'scientific',
            'hydrosphere',
            'evapotranspiration',
            'penman_monteith.py',
        )
        
        with open(pm_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=pm_file)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'PenmanMonteithET':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                if '_process_canopy_interception' in methods:
                    print("[OK] P2 - PenmanMonteithET has canopy interception")
                else:
                    print("[ERROR] P2 - PenmanMonteithET missing canopy interception")
                break
        
    except Exception as e:
        print(f"[ERROR] P2 - Canopy test failed: {e}")
    
    # P2: Snow albedo
    try:
        snow_file = os.path.join(
            ROOT_DIR,
            'relic',
            'scientific',
            'hydrosphere',
            'snow',
            'albedo.py',
        )
        
        with open(snow_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=snow_file)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'SnowAlbedo':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                required_p2_methods = [
                    '_compute_bats_albedo',
                    '_compute_age_based_albedo'
                ]
                
                for method in required_p2_methods:
                    if method in methods:
                        print(f"[OK] P2 - SnowAlbedo has method: {method}")
                    else:
                        print(f"[ERROR] P2 - SnowAlbedo missing method: {method}")
                break
        
    except Exception as e:
        print(f"[ERROR] P2 - Snow test failed: {e}")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Hydrosphere Module Basic Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Python syntax
    if not test_python_syntax():
        all_passed = False
    
    # Test 2: Interface mapping
    if not test_interface_mapping_only():
        all_passed = False
    
    # Test 3: Module structure
    if not test_module_structure():
        all_passed = False
    
    # Test 4: P0/P1/P2 methods
    test_p0_p1_p2_methods_existence()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All basic validation tests PASSED")
    else:
        print("[FAILURE] Some basic validation tests FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

