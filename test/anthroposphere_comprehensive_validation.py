"""
Comprehensive validation for anthroposphere modules.

This suite ensures all refactored anthroposphere modules can be imported,
instantiated, and comply with the four abstract base classes while using the
unified parameter system.
"""

import sys
import traceback
from pathlib import Path
import pytest

# Add project path for direct imports during tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "relic"))


def test_module_imports():
    """Ensure all anthroposphere modules import successfully."""
    print("=" * 80)
    print("1. Import anthroposphere modules")
    print("=" * 80)

    import_tests = [
        # Management modules
        ("relic.scientific.anthroposphere.management.coordinator", "ManagementCoordinator"),
        ("relic.scientific.anthroposphere.management.fertilization", "FertilizationManagement"),
        ("relic.scientific.anthroposphere.management.irrigation", "IrrigationManagement"),
        ("relic.scientific.anthroposphere.management.harvest", "HarvestManagement"),
        ("relic.scientific.anthroposphere.management.grazing", "GrazingManagement"),
        ("relic.scientific.anthroposphere.management.tillage", "TillageManagement"),
        ("relic.scientific.anthroposphere.management.fire_management", "FireManagement"),
        # Emissions modules
        ("relic.scientific.anthroposphere.emissions.species_mapping", "SpeciesMapper"),
            ("relic.scientific.anthroposphere.emissions.sector_allocation", "SectorAllocation"),
            ("relic.scientific.anthroposphere.emissions.coupling_to_atmosphere", "AtmosphereCoupling"),
            # Land use modules
        ("relic.scientific.anthroposphere.land_use.mapping", "LandUseParameterMapping"),
        ("relic.scientific.anthroposphere.land_use.transitions", "LandUseTransitions"),
        # Package init
        ("relic.scientific.anthroposphere", "__init__"),
    ]

    success_count = 0
    total_count = len(import_tests)

    for module_name, class_name in import_tests:
        try:
            if class_name == "__init__":
                __import__(module_name, fromlist=[class_name])
                print(f"[OK] Imported module {module_name}")
            else:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                print(f"[OK] Imported {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to import {module_name}.{class_name}: {e}")
            traceback.print_exc()

    print(f"\nImport results: {success_count}/{total_count} succeeded")
    assert success_count == total_count


def test_class_instantiation():
    """Instantiate key anthroposphere classes."""
    print("\n" + "=" * 80)
    print("2. Instantiate anthroposphere classes")
    print("=" * 80)

    instantiation_tests = [
        # Management modules
        ("relic.scientific.anthroposphere.management.fertilization", "FertilizationManagement", {}),
        ("relic.scientific.anthroposphere.management.irrigation", "IrrigationManagement", {}),
        ("relic.scientific.anthroposphere.management.harvest", "HarvestManagement", {}),
        ("relic.scientific.anthroposphere.management.grazing", "GrazingManagement", {}),
        ("relic.scientific.anthroposphere.management.tillage", "TillageManagement", {}),
        ("relic.scientific.anthroposphere.management.fire_management", "FireManagement", {"config": {"seed": 0}}),
        # Emissions modules
            ("relic.scientific.anthroposphere.emissions.species_mapping", "SpeciesMapper", {}),
            ("relic.scientific.anthroposphere.emissions.sector_allocation", "SectorAllocation", {}),
            # Land use modules
        ("relic.scientific.anthroposphere.land_use.mapping", "LandUseParameterMapping", {}),
        ("relic.scientific.anthroposphere.land_use.transitions", "LandUseTransitions", {}),
    ]

    success_count = 0
    total_count = len(instantiation_tests)

    for module_name, class_name, kwargs in instantiation_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            _instance = cls(**kwargs)
            print(f"[OK] Instantiated {class_name}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to instantiate {class_name}: {e}")
            traceback.print_exc()

    print(f"\nInstantiation results: {success_count}/{total_count} succeeded")
    assert success_count == total_count


def test_abstraction_inheritance():
    """Verify anthroposphere classes follow abstract base contracts."""
    print("\n" + "=" * 80)
    print("3. Validate inheritance of core abstractions")
    print("=" * 80)

    from relic.core.abstractions import (
        BiologicalProcess,
        TransportMechanism,
        PhysicalField,
        ScaleCoupling,
    )

    inheritance_tests = [
        # BiologicalProcess
        ("relic.scientific.anthroposphere.management.harvest", "HarvestManagement", BiologicalProcess),
        ("relic.scientific.anthroposphere.management.grazing", "GrazingManagement", BiologicalProcess),
        ("relic.scientific.anthroposphere.management.fire_management", "FireManagement", BiologicalProcess),
        # TransportMechanism
        ("relic.scientific.anthroposphere.management.irrigation", "IrrigationManagement", TransportMechanism),
        ("relic.scientific.anthroposphere.management.fertilization", "FertilizationManagement", TransportMechanism),
        ("relic.scientific.anthroposphere.management.tillage", "TillageManagement", TransportMechanism),
        ("relic.scientific.anthroposphere.emissions.coupling_to_atmosphere", "AtmosphereCoupling", TransportMechanism),
        # PhysicalField
        ("relic.scientific.anthroposphere.land_use.mapping", "LandUseParameterMapping", PhysicalField),
        # ScaleCoupling
        ("relic.scientific.anthroposphere.emissions.sector_allocation", "SectorAllocation", ScaleCoupling),
        ("relic.scientific.anthroposphere.emissions.species_mapping", "SpeciesMapper", ScaleCoupling),
        ("relic.scientific.anthroposphere.land_use.transitions", "LandUseTransitions", ScaleCoupling),
    ]

    success_count = 0
    total_count = len(inheritance_tests)

    for module_name, class_name, expected_base in inheritance_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            if issubclass(cls, expected_base):
                print(f"[OK] {class_name} inherits {expected_base.__name__}")
                success_count += 1
            else:
                bases = [base.__name__ for base in cls.__bases__]
                print(f"[ERROR] {class_name} does not inherit {expected_base.__name__}; bases: {bases}")
        except Exception as e:
            print(f"[ERROR] Failed inheritance check for {class_name}: {e}")
            traceback.print_exc()

    print(f"\nInheritance results: {success_count}/{total_count} succeeded")
    assert success_count == total_count


def test_parameter_system():
    """Ensure parameter manager can be imported and accessed."""
    print("\n" + "=" * 80)
    print("4. Validate parameter system access")
    print("=" * 80)

    try:
        from relic.parameters import get_global_manager
    except ImportError as e:
        print(f"[ERROR] Failed to import ParameterManager: {e}")
        raise

    manager = get_global_manager()
    sample_keys = [
        "anthroposphere.management.irrigation.application_depth",
        "anthroposphere.emissions.sector_allocation.spatial_proxy",
    ]
    for key in sample_keys:
        try:
            _ = manager.get(key)
            print(f"[OK] Retrieved parameter {key}")
        except Exception as e:
            print(f"[ERROR] Failed to retrieve parameter {key}: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
