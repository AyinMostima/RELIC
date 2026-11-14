#!/usr/bin/env python3
"""
Basic validation test for phenology module after P0-P3 fixes.
Tests core functionality and parameter integration.
"""

import pytest
import torch
import sys
import os

from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_phenology_imports():
    """Test that phenology modules can be imported successfully."""
    print("Testing phenology module imports...")

    from relic.scientific.biosphere.phenology import PhenologyFunctor
    from relic.scientific.biosphere.phenology.models import (
        StaticPhenology,
        GDDPhenology,
        GSIPhenology,
        CropPhenology,
    )

    assert PhenologyFunctor is not None
    assert StaticPhenology is not None
    assert GDDPhenology is not None
    assert GSIPhenology is not None
    assert CropPhenology is not None
    print("  OK: All phenology imports successful")

def test_parameter_system():
    """Test that phenology parameters can be loaded from the database."""
    print("Testing parameter system...")

    from relic.parameters import get_global_manager

    param_manager = get_global_manager()
    key_params = [
        'biosphere.phenology.lai.baseline',
        'biosphere.phenology.turnover.n_retrans_efficiency',
        'biosphere.phenology.static.leaf_onset_doy',
        'biosphere.phenology.static.dormant_lai',
        'biosphere.phenology.lai.response_exponent',
    ]

    for param in key_params:
        spec = param_manager.get_parameter(param)
        assert spec is not None, f"Parameter spec for {param} should not be None"
        assert spec.value is not None, f"Parameter value for {param} should not be None"
        assert spec.units is not None, f"Parameter units for {param} should not be None"
        print(f"  OK: {param} = {spec.value} {spec.units}")

def test_phenology_functor_creation():
    """Test compile-time selection of canonical model owners."""
    print("Testing PhenologyFunctor creation...")

    from relic.scientific.biosphere.phenology import PhenologyFunctor

    for owner in ('static_phenology', 'gdd_phenology'):
        functor = PhenologyFunctor(
            config={"canonical_model_owner": owner}
        )
        assert hasattr(functor, 'forward')
        assert functor.canonical_model_owner == owner

def test_static_phenology_calculation():
    """Test static phenology model calculation."""
    print("Testing static phenology calculation...")

    from relic.scientific.biosphere.phenology.models import StaticPhenology
    from relic.scientific.biosphere.phenology.base import PhenologyPhase

    model = StaticPhenology()
    day_of_year = torch.tensor([150.0])
    temperature = torch.tensor([288.15])  # 15 degC in Kelvin
    environment = {
        'solar_radiation': torch.tensor([20.0]),
        'soil_water_potential': torch.tensor([-0.5]),
    }
    result = model.compute_phenological_state(
        {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
        dt=get_physical_constant("seconds_per_day"),
    )

    assert isinstance(result.leaf_area_index, torch.Tensor)
    assert isinstance(result.growth_factor, torch.Tensor)
    assert isinstance(result.phenology_phase, (torch.Tensor, PhenologyPhase))

    print("  OK: Static phenology calculation successful")
    print(f"    Growth factor: {result.growth_factor.item():.3f}")
    print(f"    LAI: {result.leaf_area_index.item():.3f}")
    if isinstance(result.phenology_phase, torch.Tensor):
        phase_value = f"{result.phenology_phase.item():.3f}"
    else:
        phase_value = str(result.phenology_phase.value)
    print(f"    Phase: {phase_value}")

def test_phenology_functor_forward():
    """Test dependency-injected observation aggregation."""
    print("Testing PhenologyFunctor forward method...")

    from relic.scientific.biosphere.phenology import PhenologyFunctor

    functor = PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"}
    )
    owner_output = {
        'leaf_area_index': torch.tensor([2.0]),
        'phenological_factor': torch.tensor([0.5]),
        'phenology_senescence_factor': torch.tensor([0.1]),
        'phenology_phase': torch.tensor([2.0]),
    }
    result = functor.forward(
        {},
        150.0,
        86400.0,
        subprocess_outputs={"static_phenology": owner_output},
    )

    assert isinstance(result, dict)
    expected = {
        'diagnostics.phenology_state': ('phenological_factor', 'dimensionless'),
        'diagnostics.growth_stage': ('phenology_phase', 'dimensionless'),
        'diagnostics.leaf_area_index': ('leaf_area_index', 'm2 m-2'),
        'diagnostics.phenology.senescence_factor': (
            'phenology_senescence_factor',
            'dimensionless',
        ),
    }
    for diagnostic, (source, unit) in expected.items():
        item = result[diagnostic]
        assert isinstance(item, UpdateItem)
        assert item.value is owner_output[source]
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is True
        assert item.unit == unit
        assert item.source_functor == functor.name

def test_dependency_system():
    """Test dependency system integration."""
    print("Testing dependency system...")

    from relic.scientific.biosphere.phenology import PhenologyFunctor

    functor = PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"}
    )
    assert functor.get_dependencies() == {
        "required": ["static_phenology"],
        "optional": [],
    }

def test_error_handling():
    """Test error handling in phenology models."""
    print("Testing error handling...")

    from relic.scientific.biosphere.phenology import PhenologyFunctor

    with pytest.raises(ValueError, match="Unsupported canonical"):
        PhenologyFunctor(
            config={"canonical_model_owner": "invalid_type"}
        )

    functor = PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"}
    )
    with pytest.raises(KeyError, match="static_phenology"):
        functor.forward({}, 150.0, 86400.0, subprocess_outputs={})

def main():
    """Run all tests manually if executed directly."""
    print("=" * 60)
    print("PHENOLOGY MODULE VALIDATION TESTS")
    print("=" * 60)
    print("Testing phenology module after P0-P3 fixes")
    print()

    test_phenology_imports()
    test_parameter_system()
    test_phenology_functor_creation()
    test_static_phenology_calculation()
    test_phenology_functor_forward()
    test_dependency_system()
    test_error_handling()

    print()
    print("=" * 60)
    print("ALL VALIDATION TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

