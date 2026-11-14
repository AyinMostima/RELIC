#!/usr/bin/env python3
"""
Comprehensive test for phenology module functionality.
Tests all phenology models and their integration with the RELIC framework.
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
    print("OK: All phenology imports successful")

def test_phenology_functor_creation():
    """Test compile-time binding to canonical phenology owners."""
    from relic.scientific.biosphere.phenology import PhenologyFunctor

    for owner in ('static_phenology', 'gdd_phenology', 'gsi_phenology'):
        functor = PhenologyFunctor(
            config={"canonical_model_owner": owner}
        )
        assert functor.canonical_model_owner == owner

def test_phenology_model_creation():
    """Test that individual phenology models can be created."""
    from relic.scientific.biosphere.phenology.models import (
        StaticPhenology,
        GDDPhenology,
        GSIPhenology,
        CropPhenology
    )

    models_to_test = [
        (StaticPhenology, 'static_phenology'),
        (GDDPhenology, 'gdd_phenology'),
        (GSIPhenology, 'gsi_phenology'),
        (CropPhenology, 'crop_phenology')
    ]

    for model_class, _model_name in models_to_test:
        model = model_class()
        assert hasattr(model, 'compute_phenological_state')
        print(f"OK: Created {model_class.__name__} instance")

def test_parameter_system():
    """Test that phenology parameters can be loaded from the database."""
    from relic.parameters import get_global_manager

    param_manager = get_global_manager()
    key_params = [
        'biosphere.phenology.lai.baseline',
        'biosphere.phenology.turnover.n_retrans_efficiency',
        'biosphere.phenology.static.leaf_onset_doy',
        'biosphere.phenology.gdd.base_temp',
        'biosphere.phenology.gsi.tmin_threshold',
    ]

    for param in key_params:
        spec = param_manager.get_parameter(param)
        assert spec is not None
        print(f"OK: Found parameter spec for '{param}'")

def test_static_phenology_calculation():
    """Test static phenology model calculation."""
    from relic.scientific.biosphere.phenology.models import StaticPhenology

    model = StaticPhenology()
    day_of_year = torch.tensor([150.0])
    temperature = torch.tensor([288.15])  # 15 degC in Kelvin
    environment = {}
    result = model.compute_phenological_state(
        {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
        dt=get_physical_constant("seconds_per_day"),
    )

    assert isinstance(result.leaf_area_index, torch.Tensor)
    assert isinstance(result.growth_factor, torch.Tensor)
    print("OK: Static phenology calculation successful")
    print(f"  Growth factor: {result.growth_factor.item():.3f}")
    print(f"  LAI: {result.leaf_area_index.item():.3f}")

def test_gdd_phenology_calculation():
    """Test GDD phenology model calculation."""
    from relic.scientific.biosphere.phenology.models import GDDPhenology

    model = GDDPhenology()
    day_of_year = torch.tensor([150.0])
    temperature = torch.tensor([288.15])  # 15 degC in Kelvin
    environment = {
        'air_temperature_max': torch.tensor([293.15]),
        'air_temperature_min': torch.tensor([283.15]),
        'phenology_gdd_accumulated': torch.tensor(0.0),
        'phenology_gdd_previous_day_of_year': torch.tensor(149.0),
    }
    result = model.compute_phenological_state(
        {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
        dt=torch.tensor(
            get_physical_constant("seconds_per_day")
        ),
    )

    assert hasattr(result, 'leaf_area_index')
    assert hasattr(result, 'growth_factor')
    assert hasattr(result, 'phenology_phase')
    assert result.additional_outputs[
        'phenology_gdd_accumulated'
    ].item() == pytest.approx(10.0)
    assert result.additional_outputs[
        'phenology_gdd_previous_day_of_year'
    ].item() == pytest.approx(150.0)
    print("OK: GDD phenology calculation successful")
    print(f"  Growth factor: {result.growth_factor.item():.3f}")
    print(f"  LAI: {result.leaf_area_index.item():.3f}")

def test_phenology_functor_forward():
    """Test strict dependency-output aggregation signature."""
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

    expected = {
        'diagnostics.phenology_state': ('phenological_factor', 'dimensionless'),
        'diagnostics.growth_stage': ('phenology_phase', 'dimensionless'),
        'diagnostics.leaf_area_index': ('leaf_area_index', 'm2 m-2'),
        'diagnostics.phenology.senescence_factor': (
            'phenology_senescence_factor',
            'dimensionless',
        ),
    }
    assert set(result) == set(expected)
    for diagnostic, (source, unit) in expected.items():
        item = result[diagnostic]
        assert isinstance(item, UpdateItem)
        assert item.value is owner_output[source]
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is True
        assert item.unit == unit
        assert item.source_functor == functor.name

def test_parameter_integration():
    """The observation aggregator owns no phenology parameters."""
    from relic.scientific.biosphere.phenology import PhenologyFunctor

    functor = PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"}
    )
    assert "biosphere.phenology" not in str(functor.metadata.parameters)

def test_dependency_system():
    """Test dependency system integration."""
    from relic.scientific.biosphere.phenology import PhenologyFunctor

    functor = PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"}
    )
    assert functor.get_dependencies() == {
        "required": ["static_phenology"],
        "optional": [],
    }

def test_conservation_validation():
    """Diagnostic aggregation is state-neutral by construction."""
    from relic.scientific.biosphere.phenology import PhenologyFunctor

    functor = PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"}
    )
    assert not functor.metadata.conservation_laws
    assert all(key.startswith("diagnostics.") for key in functor.metadata.output_variables)

def test_error_handling():
    """Test error handling in phenology models."""
    from relic.scientific.biosphere.phenology import PhenologyFunctor

    with pytest.raises(ValueError, match="Unsupported canonical"):
        PhenologyFunctor(
            config={"canonical_model_owner": "invalid_type"}
        )

    functor = PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"}
    )
    with pytest.raises(KeyError, match="required observation"):
        functor.forward(
            {},
            150.0,
            86400.0,
            subprocess_outputs={"static_phenology": {}},
        )

def main():
    """Run all tests manually if executed directly."""
    print("Running comprehensive phenology tests...")

    test_phenology_imports()
    test_phenology_functor_creation()
    test_phenology_model_creation()
    test_parameter_system()
    test_static_phenology_calculation()
    test_gdd_phenology_calculation()
    test_phenology_functor_forward()
    test_parameter_integration()
    test_dependency_system()
    test_conservation_validation()
    test_error_handling()

    print("\nAll comprehensive tests completed!")

if __name__ == "__main__":
    main()
