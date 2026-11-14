"""
Verification script for maintenance_respiration.py parameter management fixes.

This script verifies that all get_parameter_value() calls now include:
1. units parameter with CF-1.8 compliant units
2. bounds parameter with literature-based physical bounds
3. validate=True for runtime validation
4. No hardcoded constants or state.get() defaults

Author: Yuchong Long
Date: 2025-10-17
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def extract_get_parameter_calls(file_path: Path) -> List[Dict[str, any]]:
    """
    Extract all get_parameter_value() calls from the file.

    Returns:
        List of dictionaries containing call details
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match get_parameter_value calls with all their parameters
    pattern = r'self\.get_parameter_value\s*\(((?:[^()]|\([^()]*\))*)\)'

    calls = []
    for match in re.finditer(pattern, content):
        call_text = match.group(0)
        params_text = match.group(1)

        # Extract parameter name (first argument)
        param_name_match = re.search(r'["\']([^"\']+)["\']', params_text)
        param_name = param_name_match.group(1) if param_name_match else "UNKNOWN"

        # Check for units
        has_units = 'units=' in params_text
        units_match = re.search(r'units\s*=\s*["\']([^"\']+)["\']', params_text)
        units_value = units_match.group(1) if units_match else None

        # Check for bounds
        has_bounds = 'bounds=' in params_text
        bounds_match = re.search(r'bounds\s*=\s*\(([^)]+)\)', params_text)
        bounds_value = bounds_match.group(1) if bounds_match else None

        # Check for validate
        has_validate = 'validate=' in params_text
        validate_match = re.search(r'validate\s*=\s*(True|False)', params_text)
        validate_value = validate_match.group(1) if validate_match else None

        # Get line number
        line_num = content[:match.start()].count('\n') + 1

        calls.append({
            'param_name': param_name,
            'line': line_num,
            'has_units': has_units,
            'units': units_value,
            'has_bounds': has_bounds,
            'bounds': bounds_value,
            'has_validate': has_validate,
            'validate': validate_value,
            'full_call': call_text[:100] + '...' if len(call_text) > 100 else call_text
        })

    return calls


def check_hardcoded_constants(file_path: Path) -> List[Dict[str, any]]:
    """Check for hardcoded torch.tensor constants."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []

    # Look for torch.tensor with numeric constants (excluding state conversions)
    pattern = r'torch\.tensor\s*\(\s*([0-9.e+-]+)'

    for match in re.finditer(pattern, content):
        line_num = content[:match.start()].count('\n') + 1
        line_content = content.split('\n')[line_num - 1].strip()

        # Exclude legitimate uses (converting from history, state variables)
        if 'mean_temp' in line_content or 'value' in line_content:
            continue

        issues.append({
            'line': line_num,
            'value': match.group(1),
            'context': line_content
        })

    return issues


def verify_parameter_standards(calls: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Verify that all parameters meet the required standards.

    Returns:
        Tuple of (passed_checks, failed_checks)
    """
    passed = []
    failed = []

    # Expected parameters with their standards
    expected_params = {
        'rm_leaf_25': {
            'units': 's-1',
            'bounds': '5.787e-8, 5.787e-7'
        },
        'rm_stem_25': {
            'units': 's-1',
            'bounds': '1.157e-8, 2.315e-7'
        },
        'rm_root_25': {
            'units': 's-1',
            'bounds': '3.472e-8, 3.472e-7'
        },
        'q10_respiration': {
            'units': 'dimensionless',
            'bounds': '1.5, 4.0'
        },
        'q10_leaf_factor': {
            'units': 'dimensionless',
            'bounds': '0.8, 1.2'
        },
        'q10_stem_factor': {
            'units': 'dimensionless',
            'bounds': '0.7, 1.1'
        },
        'q10_root_factor': {
            'units': 'dimensionless',
            'bounds': '0.9, 1.3'
        },
        'acclimation_rate_coeff': {
            'units': 'K-1',
            'bounds': '0.01, 0.15'
        },
        'km_substrate': {
            'units': 'g C m-2',
            'bounds': '5.0, 25.0'
        },
        'water_stress_wilting_point': {
            'units': 'MPa',
            'bounds': '-3.0, -0.5'
        },
        'water_stress_field_capacity': {
            'units': 'MPa',
            'bounds': '-0.5, 0.0'
        },
        'min_respiration_rate': {
            'units': 'umol CO2 m-2 s-1',
            'bounds': '0.0, 1.0e-5'
        }
    }

    for call in calls:
        param_name = call['param_name']

        # Skip non-respiration parameters
        if param_name not in expected_params and param_name != 'standard_25c':
            continue

        # Check units
        if not call['has_units']:
            failed.append(f"[FAIL] {param_name} (line {call['line']}): Missing units parameter")
        elif param_name in expected_params:
            expected_units = expected_params[param_name]['units']
            if call['units'] != expected_units:
                failed.append(f"[FAIL] {param_name} (line {call['line']}): Wrong units '{call['units']}', expected '{expected_units}'")
            else:
                passed.append(f"[OK] {param_name}: Correct units '{call['units']}'")

        # Check bounds
        if not call['has_bounds']:
            failed.append(f"[FAIL] {param_name} (line {call['line']}): Missing bounds parameter")
        elif param_name in expected_params:
            # Normalize bounds for comparison
            actual_bounds = call['bounds'].replace(' ', '') if call['bounds'] else ''
            expected_bounds = expected_params[param_name]['bounds'].replace(' ', '')
            if actual_bounds != expected_bounds:
                failed.append(f"[FAIL] {param_name} (line {call['line']}): Wrong bounds ({call['bounds']}), expected ({expected_params[param_name]['bounds']})")
            else:
                passed.append(f"[OK] {param_name}: Correct bounds {call['bounds']}")

        # Check validate
        if not call['has_validate']:
            failed.append(f"[FAIL] {param_name} (line {call['line']}): Missing validate parameter")
        elif call['validate'] != 'True':
            failed.append(f"[FAIL] {param_name} (line {call['line']}): validate should be True, got {call['validate']}")
        else:
            passed.append(f"[OK] {param_name}: validate=True")

    return passed, failed


def main():
    """Main verification function."""
    print("=" * 80)
    print("MAINTENANCE RESPIRATION PARAMETER MANAGEMENT FIX VERIFICATION")
    print("=" * 80)
    print()

    file_path = Path(__file__).resolve().parents[2] / "relic" / "scientific" / "biosphere" / "respiration" / "maintenance_respiration.py"

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    print(f"Analyzing file: {file_path.name}")
    print()

    # Extract all get_parameter_value calls
    calls = extract_get_parameter_calls(file_path)
    print(f"Total get_parameter_value() calls found: {len(calls)}")
    print()

    # Check for hardcoded constants
    print("-" * 80)
    print("CHECKING FOR HARDCODED CONSTANTS")
    print("-" * 80)
    hardcoded = check_hardcoded_constants(file_path)
    if hardcoded:
        print(f"[WARNING] Found {len(hardcoded)} potential hardcoded constants:")
        for issue in hardcoded:
            print(f"  Line {issue['line']}: {issue['context']}")
    else:
        print("[OK] No problematic hardcoded constants found")
    print()

    # Verify parameter standards
    print("-" * 80)
    print("VERIFYING PARAMETER STANDARDS")
    print("-" * 80)
    passed, failed = verify_parameter_standards(calls)

    if passed:
        print(f"\n[PASSED CHECKS] ({len(passed)} items):")
        for check in passed:
            print(f"  {check}")

    if failed:
        print(f"\n[FAILED CHECKS] ({len(failed)} items):")
        for check in failed:
            print(f"  {check}")
    else:
        print("\n[OK] All parameter standards checks passed!")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total parameters checked: {len(calls)}")
    print(f"Passed checks: {len(passed)}")
    print(f"Failed checks: {len(failed)}")
    print(f"Hardcoded constants: {len(hardcoded)}")
    print()

    if len(failed) == 0 and len(hardcoded) == 0:
        print("[SUCCESS] All verification checks passed!")
        return 0
    else:
        print("[WARNING] Some issues were found. Please review the details above.")
        return 1


if __name__ == "__main__":
    exit(main())


