"""
Find actual parameter keys in database for warmup optimization.
"""

import sys
from pathlib import Path

# Add relic to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.parameters import get_global_manager
import pytest


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


def find_common_params():
    """Find parameters that actually exist for warmup."""
    manager = get_global_manager()

    print("Searching for common parameter patterns...")
    print("=" * 70)

    patterns = [
        'vcmax', 'jmax', 'rd', 'kc', 'ko', 'gamma',  # Photosynthesis
        'q10', 'respir',  # Respiration
        'time', 'step', 'dt',  # Simulation
        'R_over_Cp', 'gas_constant', 'stefan',  # Constants
        'field_capacity', 'wilting', 'bulk_density',  # Soil
        'g0', 'g1',  # Stomata
        'gdd', 'photoperiod',  # Phenology
    ]

    matches = {}
    for pattern in patterns:
        matching_keys = [k for k in manager.database.cache.keys() if pattern.lower() in k.lower()]
        if matching_keys:
            matches[pattern] = matching_keys[:5]  # First 5 matches

    print("\nPattern matches:")
    for pattern, keys in matches.items():
        print(f"\n{pattern}:")
        for key in keys:
            value = manager.get(key)
            print(f"  - {key}: {value}")

    # Also search for common prefixes
    print("\n\nSearching by namespace prefix...")
    prefixes = ['biosphere', 'pedosphere', 'atmosphere', 'hydrosphere', 'core', 'physical']

    prefix_counts = {}
    for prefix in prefixes:
        count = sum(1 for k in manager.database.cache.keys() if k.startswith(prefix))
        prefix_counts[prefix] = count
        print(f"{prefix}: {count} parameters")

    # Show some examples from each major namespace
    print("\n\nExample parameters by namespace:")
    for prefix in prefixes:
        matching = [k for k in manager.database.cache.keys() if k.startswith(prefix)]
        if matching:
            print(f"\n{prefix} (showing first 10):")
            for key in matching[:10]:
                value = manager.get(key)
                print(f"  {key}: {value}")


if __name__ == '__main__':
    find_common_params()

