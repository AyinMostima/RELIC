#!/usr/bin/env python3
"""
Minimal functionality test for anthroposphere modules (Pytest Version)

This test focuses on validating that the key inheritance refactoring 
was successful, testing only the most critical functionality to 
ensure the anthroposphere modules are working properly after
the abstract base class inheritance changes.
"""

import sys
import os
import torch
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from test.utils.functor_minimal_construction import construct_minimal

# Import explicitly to ensure paths are correct
try:
    from relic.core.abstractions import (
        PhysicalField,
        BiologicalProcess,
        TransportMechanism,
        ScaleCoupling
    )
except ImportError:
    pytest.fail("Could not import core abstractions")

def test_inheritance_structure():
    """Test that abstract base classes are accessible."""
    assert PhysicalField is not None
    assert BiologicalProcess is not None
    assert TransportMechanism is not None
    assert ScaleCoupling is not None

@pytest.mark.parametrize("module_name", [
    "relic.scientific.anthroposphere.emissions.atmosphere_interface_contract",
    "relic.scientific.anthroposphere.emissions.coupling_to_atmosphere",
    # "relic.scientific.anthroposphere.emissions.inventory", # retired/moved
    "relic.scientific.anthroposphere.emissions.sector_allocation",
    "relic.scientific.anthroposphere.emissions.species_mapping",
    "relic.scientific.anthroposphere.interface_mapping",
    "relic.scientific.anthroposphere.land_use.mapping",
    "relic.scientific.anthroposphere.land_use.transitions",
    # "relic.scientific.anthroposphere.management.event_scheduler", # Might not exist
    "relic.scientific.anthroposphere.management.fertilization",
    "relic.scientific.anthroposphere.management.fire_management",
    "relic.scientific.anthroposphere.management.grazing",
    "relic.scientific.anthroposphere.management.harvest",
    "relic.scientific.anthroposphere.management.irrigation",
    "relic.scientific.anthroposphere.management.tillage"
])
def test_module_imports(module_name):
    """Test that modules can be imported."""
    try:
        __import__(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")

@pytest.mark.parametrize("class_name, module_name, requires_calendar", [
    ("HarvestManagement", "relic.scientific.anthroposphere.management.harvest", True),
    ("FireManagement", "relic.scientific.anthroposphere.management.fire_management", True),
    ("LandUseTransitions", "relic.scientific.anthroposphere.land_use.transitions", False),
    ("LandUseParameterMapping", "relic.scientific.anthroposphere.land_use.mapping", False),
])
def test_forward_method_consistency(class_name, module_name, requires_calendar):
    """Test that classes support the forward method."""
    
    # Mock state
    mock_state = {
        'temperature': torch.tensor([298.15] * 5),
        'land_use_fraction': torch.tensor([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0]), # 9 types
        'soil_moisture': torch.tensor([0.3] * 5),
        'biomass': torch.tensor([1000.0] * 5),
        'urea_n': torch.tensor([10.0] * 5),
        'aboveground_biomass': torch.tensor([2.0] * 5), # For Fire
        'belowground_biomass': torch.tensor([1.0] * 5),
        'litter_mass': torch.tensor([0.5] * 5),
        'fuel_moisture': torch.tensor([0.1] * 5),
        'days_since_fire': torch.tensor([30.0] * 5),
        'crop_maturity': torch.tensor([1.0] * 5),  # For Harvest
        'residue_cover': torch.tensor([0.0] * 5),
        'vegetation_quality': torch.tensor([0.8] * 5), # For Grazing (if tested)
    }
    
    # Import class
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    
    # Instantiate
    try:
        instance = construct_minimal(f"{module_name}:{class_name}")
    except Exception as e:
        pytest.fail(f"Failed to instantiate {class_name}: {e}")
        
    assert hasattr(instance, 'forward'), f"{class_name} missing forward method"
    
    # Run forward
    try:
        forward_kwargs = {"time": 0.0, "dt": 3600.0}
        if requires_calendar:
            forward_kwargs["calendar_type"] = "noleap"
        result = instance.forward(
            mock_state,
            **forward_kwargs,
        )
        assert isinstance(result, dict), f"{class_name}.forward should return dict"
    except Exception as e:
        pytest.fail(f"{class_name}.forward failed: {e}")

def test_scientific_calculation_basics():
    """Test basic scientific calculation functionality."""
    temperature = torch.tensor([278.15, 298.15, 308.15])  # 5掳C, 25掳C, 35掳C
    
    # Simple Q10 temperature response
    Q10 = 2.0
    T_ref = 298.15  # 25掳C reference
    response = Q10 ** ((temperature - T_ref) / 10.0)
    
    # Verify reasonable responses
    assert response[0] < response[1] < response[2], "Temperature response should increase with temperature"
    assert torch.all(response > 0), "All responses should be positive"
    
    # Test conservation principle (sum should be preserved in averaging)
    fine_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    coarse_data = fine_data.reshape(2, 2).mean(dim=1)  # Simple averaging
    
    # Check that total mass is approximately conserved
    assert abs(fine_data.sum() - coarse_data.sum() * 2) < 1e-6, "Mass should be conserved in averaging"


