"""
Test script to verify ManagementCoordinator Phase 1 refactoring

This test verifies that ManagementCoordinator no longer directly calls other Functors
and properly uses the RELICSimulator dependency chain.
"""

import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator
from relic.core.state import UpdateItem, UpdateMode


def test_coordinator_no_direct_functor_calls():
    """Test that ManagementCoordinator doesn't directly instantiate other Functors"""
    
    # Create coordinator
    coordinator = ManagementCoordinator("test_coordinator")
    
    # Verify no management_functors attribute exists
    assert not hasattr(coordinator, 'management_functors'), "ManagementCoordinator should not have management_functors attribute"
    
    # Verify proper dependencies are declared
    dependencies = coordinator.get_dependencies()
    assert dependencies == {
        "required": [],
        "optional": [],
        "lagged": [
            "irrigation_management",
            "harvest_management",
            "grazing_management",
            "fire_management",
        ],
    }


def test_coordinator_event_handlers():
    """Test that event handlers work without direct functor calls"""
    
    coordinator = ManagementCoordinator("test_coordinator")
    
    # Test state
    state = {
        "soil_moisture": torch.tensor(0.2),
        "aboveground_biomass": torch.tensor(2.5),
        "soil_temperature": torch.tensor(288.15)
    }
    
    # Test irrigation event handler
    irrigation_params = {"water_amount": 25.0, "dt": 3600.0}
    result = coordinator._handle_irrigation_event(state, irrigation_params)
    
    assert result['status'] == 'completed', "Irrigation event should complete successfully"
    assert result['requested_water_depth'] == 25.0
    assert result['requested_duration'] == 3600.0
    assert 'irrigation_flux' not in result
    
    # Test tillage event handler
    tillage_params = {"intensity": "moderate", "dt": 3600.0}
    result = coordinator._handle_tillage_event(state, tillage_params)
    
    assert result['status'] == 'completed', "Tillage event should complete successfully"
    assert result['tillage_intensity_class'] == 'moderate'
    assert 'decomposition_modifier' not in result
    
    # Test harvest event handler
    harvest_params = {"expected_yield": 5.0, "dt": 3600.0}
    result = coordinator._handle_harvest_event(state, harvest_params)
    
    assert result['status'] == 'completed', "Harvest event should complete successfully"
    assert result['expected_yield'] == 5.0
    assert 'harvest_flux' not in result
    
    # Test grazing event handler
    grazing_params = {"stocking_rate": 1.0, "dt": 86400.0}
    result = coordinator._handle_grazing_event(state, grazing_params)
    
    assert result['status'] == 'completed', "Grazing event should complete successfully"
    assert result['requested_stocking_rate'] == 1.0
    assert 'grazing_flux' not in result
    
    # Test fire event handler
    fire_params = {"fire_type": "prescribed", "dt": 3600.0}
    result = coordinator._handle_fire_event(state, fire_params)
    
    assert result['status'] == 'completed', "Fire event should complete successfully"
    assert result['requested_fire_type'] == 'prescribed'
    assert 'co2_emission' not in result


def test_coordinator_forward_method():
    """Test that coordinator forward method works without direct functor calls"""
    
    coordinator = ManagementCoordinator("test_coordinator")
    
    # Test state
    state = {
        "day_of_year": torch.tensor(180.0),
        "soil_moisture": torch.tensor(0.2),
        "soil_nitrogen": torch.tensor(0.01),
        "soil_organic_carbon": torch.tensor(5.0),
        "crop_maturity": torch.tensor(0.7),
        "aboveground_biomass": torch.tensor(2.5),
        "air_temperature": torch.tensor(298.15),
        "irrigation_flux": torch.tensor(0.001),
        "harvest_flux": torch.tensor(0.0),
        "grazing_flux": torch.tensor(0.0),
        "co2_emission": torch.tensor(0.0),
        "field_capacity": torch.tensor(0.35),
        "wilting_point": torch.tensor(0.15),
        "evapotranspiration_flux": torch.tensor(3.0e-6),
        "precipitation_flux": torch.tensor(0.0)
    }
    
    # Run forward method
    time = 86400.0 * 180  # Day 180
    dt = 3600.0
    
    # The day of year is the declared state input above; forward takes only the
    # calendar keyword besides (state, time, dt).
    outputs = coordinator.forward(
        state,
        time,
        dt,
        calendar_type="noleap",
    )
    
    # Verify outputs
    assert "diagnostics.management_coordinator.active" in outputs
    assert "diagnostics.management_coordinator.events_executed" in outputs
    assert "coordination_status" in outputs, "Should have coordination_status output"
    
    # Verify tensor types
    for key, value in outputs.items():
        assert isinstance(value, UpdateItem), f"Output {key} should be explicit"
        assert value.mode is UpdateMode.REPLACE
        assert value.per_timestep is False


def test_coordinator_aggregate_effects():
    """Test that effect aggregation works without direct functor calls"""
    
    coordinator = ManagementCoordinator("test_coordinator")
    
    # Create mock executed events
    from relic.scientific.anthroposphere.management.event_scheduler import ManagementEvent, EventType
    
    events = [
        ManagementEvent(
            event_id="test_irrigation",
            event_type=EventType.IRRIGATION,
            scheduled_time=0.0,
            priority=1,
            duration=3600.0,
            parameters={"water_amount": 25.0}
        )
    ]
    
    # A historical process-flux payload must not enter the canonical aggregate.
    events[0].execution_result = {
        "status": "completed",
        "irrigation_flux": torch.tensor(0.001)
    }
    
    # Test state
    state = {
        "soil_moisture": torch.tensor(0.2),
        "irrigation_flux": torch.tensor(0.0005),
        "harvest_flux": torch.tensor(0.0),
        "grazing_flux": torch.tensor(0.0),
        "co2_emission": torch.tensor(0.0),
        "field_capacity": torch.tensor(0.35),
        "wilting_point": torch.tensor(0.15),
        "evapotranspiration_flux": torch.tensor(3.0e-6),
        "precipitation_flux": torch.tensor(0.0)
    }
    
    # Aggregate effects
    effects = coordinator._aggregate_management_effects(events, state, 3600.0)
    
    # Verify aggregation
    assert "total_irrigation_flux" in effects, "Should have total_irrigation_flux"
    assert torch.allclose(
        effects["total_irrigation_flux"],
        torch.tensor(0.0005),
    ), "Only the lagged canonical dependency flux may be aggregated"
    assert "irrigation_events_count" in effects, "Should have event counts"
    

if __name__ == "__main__":
    print("Testing ManagementCoordinator Phase 1 refactoring...")
    
    test_coordinator_no_direct_functor_calls()
    print("[OK] No direct functor calls test passed")
    
    test_coordinator_event_handlers()
    print("[OK] Event handlers test passed")
    
    test_coordinator_forward_method()
    print("[OK] Forward method test passed")
    
    test_coordinator_aggregate_effects()
    print("[OK] Effect aggregation test passed")
    
    print("\nAll Phase 1 tests passed! ManagementCoordinator successfully refactored.")
    print("The coordinator no longer directly calls other Functors and properly uses")
    print("the RELICSimulator dependency chain.")

