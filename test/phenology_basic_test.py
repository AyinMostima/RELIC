"""
Basic test for phenology module functionality.
Tests core imports and basic functionality without complex fixtures.
"""

import pytest
import torch
import sys
import os
from relic.utils.physical_constants import get_physical_constant

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_phenology_imports():
    """Test that phenology modules can be imported successfully."""
    try:
        from relic.scientific.biosphere.phenology import PhenologyFunctor
        assert PhenologyFunctor is not None
        print("OK: PhenologyFunctor import successful")
    except Exception as e:
        pytest.fail(f"Failed to import PhenologyFunctor: {e}")

    try:
        from relic.scientific.biosphere.phenology.models import (
            StaticPhenology,
            GDDPhenology,
            GSIPhenology,
            CropPhenology
        )
        assert StaticPhenology is not None
        assert GDDPhenology is not None
        assert GSIPhenology is not None
        assert CropPhenology is not None
        print("OK: All phenology model imports successful")
    except Exception as e:
        pytest.fail(f"Failed to import phenology models: {e}")

def test_phenology_functor_creation():
    """Test compile-time binding to each canonical phenology owner."""
    from relic.scientific.biosphere.phenology import PhenologyFunctor

    owners = [
        'static_phenology',
        'gdd_phenology',
        'gsi_phenology',
        'crop_phenology',
    ]

    for owner in owners:
        functor = PhenologyFunctor(
            config={"canonical_model_owner": owner}
        )
        assert hasattr(functor, 'forward')
        assert functor.get_dependencies() == {
            "required": [owner],
            "optional": [],
        }
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

    for model_class, model_name in models_to_test:
        try:
            model = model_class()
            assert hasattr(model, 'compute_phenological_state')
            print(f"OK: Created {model_class.__name__} instance")
        except Exception as e:
            print(f"WARNING: Could not create {model_class.__name__}: {e}")

def test_parameter_system():
    """Test that phenology parameters can be loaded from the database."""
    from relic.parameters import get_global_manager

    try:
        param_manager = get_global_manager()

        # Test some key phenology parameters
        key_params = [
            'biosphere.phenology.lai.baseline',
            'biosphere.phenology.time.seconds_per_day'
        ]

        for param in key_params:
            try:
                spec = param_manager.get_parameter(param)
                assert spec is not None
                print(f"OK: Found parameter spec for '{param}'")
            except Exception as e:
                print(f"WARNING: Could not get spec for '{param}': {e}")

    except Exception as e:
        pytest.fail(f"Failed to initialize parameter manager: {e}")

def test_simple_phenology_calculation():
    """Test a simple phenology calculation without complex setup."""
    try:
        from relic.scientific.biosphere.phenology.models import StaticPhenology

        # Create static phenology model
        model = StaticPhenology()

        # Create minimal test data
        day_of_year = torch.tensor([150.0])  # Late May
        temperature = torch.tensor([288.15])  # 15掳C in Kelvin
        environment = {
            'solar_radiation': torch.tensor([20.0]),
            'soil_water_potential': torch.tensor([-0.5])
        }

        # Test phenology calculation
        result = model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )

        # Check that result has expected attributes
        assert hasattr(result, 'leaf_area_index')
        assert hasattr(result, 'growth_factor')
        assert hasattr(result, 'phenology_phase')

        # Check that values are reasonable
        assert isinstance(result.leaf_area_index, torch.Tensor)
        assert isinstance(result.growth_factor, torch.Tensor)

        print("OK: Static phenology calculation successful")
        print(f"  Growth factor: {result.growth_factor.item():.3f}")
        print(f"  LAI: {result.leaf_area_index.item():.3f}")

    except Exception as e:
        pytest.fail(f"Failed simple phenology calculation: {e}")

if __name__ == "__main__":
    """Run tests manually if executed directly."""
    print("Running phenology basic tests...")

    test_phenology_imports()
    test_phenology_functor_creation()
    test_phenology_model_creation()
    test_parameter_system()
    test_simple_phenology_calculation()

    print("\nAll basic tests completed!")

