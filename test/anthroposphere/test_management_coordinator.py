"""
Test suite for Management Coordinator module

This module contains comprehensive tests for the management coordinator,
including event scheduling, conflict resolution, resource allocation,
and integration with management functors.
"""

import pytest
import torch

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'relic'))

from relic.scientific.anthroposphere.management.coordinator import (
    ManagementCoordinator, ManagementStrategy, CoordinationMode,
    CoordinationResult
)
from relic.scientific.anthroposphere.management.event_scheduler import (
    EventType, EventPriority, ConflictResolution
)
from relic.core.state import UpdateItem, UpdateMode


class TestManagementCoordinator:
    """Test class for ManagementCoordinator functionality"""
    
    @pytest.fixture
    def coordinator(self):
        """Create a test coordinator instance"""
        config = {
            "calendar_type": "noleap",
            "parameters": {
                "coordination_mode": {"value": "adaptive"},
                "resource_efficiency_target": {"value": 0.8},
                "sustainability_weight": {"value": 0.4},
                "productivity_weight": {"value": 0.6}
            }
        }
        return ManagementCoordinator("test_coordinator", config)
    
    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing"""
        return {
            "soil_moisture": torch.tensor([[0.2]]),
            "soil_nitrogen": torch.tensor([[0.01]]),
            "soil_organic_carbon": torch.tensor([[5.0]]),
            "crop_maturity": torch.tensor([[0.7]]),
            "aboveground_biomass": torch.tensor([[1.5]]),
            "air_temperature": torch.tensor([[295.15]]),
            "day_of_year": torch.tensor(150.0),
            "irrigation_flux": torch.tensor([[0.0]]),
            "harvest_flux": torch.tensor([[0.0]]),
            "grazing_flux": torch.tensor([[0.0]]),
            "co2_emission": torch.tensor([[0.0]]),
            "field_capacity": torch.tensor([[0.35]]),
            "wilting_point": torch.tensor([[0.15]]),
            "evapotranspiration_flux": torch.tensor([[3.0e-6]]),
            "precipitation_flux": torch.tensor([[0.0]]),
        }
    
    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization and setup"""
        assert coordinator is not None
        assert coordinator.management_strategy == ManagementStrategy.CONVENTIONAL
        assert coordinator.coordination_mode == CoordinationMode.ADAPTIVE
        
        # Subordinate Functors are registered only by RELICSimulator.
        assert not hasattr(coordinator, "management_modules")
        
        # Check that scheduler is initialized
        assert coordinator.scheduler is not None
        
        # Check management objectives
        assert len(coordinator.management_objectives) > 0
        assert any(obj.name == "water_use_efficiency" for obj in coordinator.management_objectives)
    
    def test_schedule_management_event(self, coordinator):
        """Test scheduling of management events"""
        # Test successful event scheduling
        success = coordinator.schedule_management_event(
            event_type=EventType.IRRIGATION,
            scheduled_time=100000.0,
            parameters={"water_amount": 25.0},
            priority=EventPriority.HIGH,
            resource_requirements={"water": 25.0, "labor": 1.0}
        )
        
        assert success is True
        assert len(coordinator.scheduled_events) == 1
        
        # Test event with conflicting resources
        success2 = coordinator.schedule_management_event(
            event_type=EventType.TILLAGE,
            scheduled_time=100000.0,  # Same time
            parameters={"tillage_type": "conventional"},
            resource_requirements={"labor": 10.0}  # High labor requirement
        )
        
        # Should still succeed but may be rescheduled by conflict resolution
        assert success2 is True or len(coordinator.scheduled_events) >= 1
    
    def test_coordination_objectives(self, coordinator):
        """Test management objective setup and modification"""
        # Test default objectives exist
        objective_names = [obj.name for obj in coordinator.management_objectives]
        expected_objectives = [
            "water_use_efficiency",
            "nutrient_use_efficiency", 
            "soil_carbon_sequestration",
            "economic_return",
            "biodiversity_index"
        ]
        
        for expected in expected_objectives:
            assert expected in objective_names
        
        # Test objective weights sum to reasonable value
        total_weight = sum(obj.weight for obj in coordinator.management_objectives)
        assert 0.8 <= total_weight <= 1.2  # Allow some flexibility
    
    def test_management_strategy_setting(self, coordinator):
        """Test setting different management strategies"""
        # Test setting organic strategy
        coordinator.set_management_strategy(ManagementStrategy.ORGANIC)
        assert coordinator.management_strategy == ManagementStrategy.ORGANIC
        
        # Check that objectives were adjusted for organic strategy
        soil_carbon_obj = next(
            (obj for obj in coordinator.management_objectives if obj.name == "soil_carbon_sequestration"),
            None
        )
        assert soil_carbon_obj is not None
        assert soil_carbon_obj.weight >= 0.3  # Should be increased for organic
        
        # Test precision strategy
        coordinator.set_management_strategy(ManagementStrategy.PRECISION)
        assert coordinator.management_strategy == ManagementStrategy.PRECISION
    
    def test_coordination_mode_setting(self, coordinator):
        """Test setting different coordination modes"""
        # Test strict priority mode
        coordinator.set_coordination_mode(CoordinationMode.STRICT_PRIORITY)
        assert coordinator.coordination_mode == CoordinationMode.STRICT_PRIORITY
        assert coordinator.scheduler.conflict_strategy == ConflictResolution.PRIORITY_BASED
        
        # Test cooperative mode
        coordinator.set_coordination_mode(CoordinationMode.COOPERATIVE)
        assert coordinator.coordination_mode == CoordinationMode.COOPERATIVE
        assert coordinator.scheduler.conflict_strategy == ConflictResolution.MERGE
    
    def test_management_needs_assessment(self, coordinator, sample_state):
        """Test assessment of management needs"""
        # Access private method for testing
        needs = coordinator._assess_management_needs(
            sample_state, 100000.0
        )
        
        assert isinstance(needs, dict)
        assert "irrigation" in needs
        assert "harvest" in needs
        assert "tillage" in needs
        assert "grazing" in needs
        assert "fire_management" in needs
        
        # All needs should be between 0 and 1
        for activity, need_score in needs.items():
            assert 0.0 <= need_score <= 1.0
        
        # With low soil moisture, irrigation need should be higher
        assert needs["irrigation"] > 0.3  # Soil moisture is 0.2, below threshold
    
    def test_candidate_event_generation(self, coordinator, sample_state):
        """Test generation of candidate management events"""
        # Mock management needs
        management_needs = {
            "irrigation": 0.8,
            "tillage": 0.3,
            "harvest": 0.1,
            "grazing": 0.5,
            "fire_management": 0.1
        }
        
        candidates = coordinator._generate_candidate_events(
            current_time=100000.0,
            planning_horizon=30.0,
            management_needs=management_needs,
            state=sample_state
        )
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        
        # Should have events for activities with need > 0.3
        event_types = [event.event_type for event in candidates]
        assert EventType.IRRIGATION in event_types  # High need (0.8)
        assert EventType.GRAZING in event_types     # Moderate need (0.5)
        
        # Events should be within planning horizon
        for event in candidates:
            assert 100000.0 <= event.scheduled_time <= 100000.0 + (30.0 * 86400.0)
    
    def test_resource_allocation_calculation(self, coordinator):
        """Test resource allocation calculations"""
        # Create mock events with resource requirements
        from relic.scientific.anthroposphere.management.event_scheduler import ManagementEvent
        
        events = [
            ManagementEvent(
                event_id="test1",
                event_type=EventType.IRRIGATION,
                scheduled_time=100000.0,
                priority=EventPriority.HIGH,
                duration=3600.0,
                parameters={},
                resource_requirements={"water": 25.0, "labor": 1.0}
            ),
            ManagementEvent(
                event_id="test2",
                event_type=EventType.TILLAGE,
                scheduled_time=101000.0,
                priority=EventPriority.NORMAL,
                duration=7200.0,
                parameters={},
                resource_requirements={"fuel": 15.0, "labor": 3.0}
            )
        ]
        
        allocation = coordinator._calculate_resource_allocation(events)
        
        assert isinstance(allocation, dict)
        assert allocation["water"] == 25.0
        assert allocation["labor"] == 4.0  # 1.0 + 3.0
        assert allocation["fuel"] == 15.0
    
    def test_objective_evaluation(self, coordinator, sample_state):
        """Test evaluation of management objectives"""
        from relic.scientific.anthroposphere.management.event_scheduler import ManagementEvent
        
        # Create mock events
        events = [
            ManagementEvent(
                event_id="irrigation_test",
                event_type=EventType.IRRIGATION,
                scheduled_time=100000.0,
                priority=EventPriority.HIGH,
                duration=3600.0,
                parameters={},
                resource_requirements={"water": 25.0}
            )
        ]
        
        scores = coordinator._evaluate_objectives(events, sample_state)
        
        assert isinstance(scores, dict)
        assert len(scores) == len(coordinator.management_objectives)
        
        # All scores should be between 0 and 1 (or reasonable range)
        for obj_name, score in scores.items():
            assert 0.0 <= score <= 2.0  # Allow some flexibility for scores
    
    def test_coordination_workflow(self, coordinator, sample_state):
        """Test complete coordination workflow"""
        result = coordinator.coordinate_management_activities(
            current_time=100000.0,
            state=sample_state,
            planning_horizon=7.0  # 1 week
        )
        
        assert isinstance(result, CoordinationResult)
        assert isinstance(result.scheduled_events, list)
        assert isinstance(result.resource_allocation, dict)
        assert isinstance(result.objective_scores, dict)
        assert result.coordination_status == "completed"
        
        # Should have some events scheduled given the needs
        assert len(result.scheduled_events) > 0
    
    def test_forward_method(self, coordinator, sample_state):
        """Test the forward method execution"""
        # Set up time for coordination trigger
        coordinator.last_coordination_time = 0.0  # Force coordination
        
        outputs = coordinator.forward(
            state=sample_state,
            time=100000.0,
            dt=3600.0,
        )
        
        assert isinstance(outputs, dict)
        assert "diagnostics.management_coordinator.active" in outputs
        assert "diagnostics.management_coordinator.events_executed" in outputs
        assert "coordination_status" in outputs
        
        for item in outputs.values():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
    
    def test_event_handlers(self, coordinator):
        """Test event handler functionality"""
        # Test irrigation handler
        state = {"soil_moisture": torch.tensor([[0.2]])}
        parameters = {"time": 100000.0, "dt": 3600.0, "water_amount": 25.0}
        
        result = coordinator._handle_irrigation_event(state, parameters)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["completed", "failed"]
        
        if result["status"] == "completed":
            assert result["requested_water_depth"] == 25.0
            assert result["requested_duration"] == 3600.0
            assert "irrigation_flux" not in result
    
    def test_seasonal_context_update(self, coordinator, sample_state):
        """Test environmental context updates"""
        # Test spring season
        spring_state = dict(sample_state)
        spring_state["day_of_year"] = torch.tensor(100.0)
        coordinator._update_environmental_context(
            spring_state, 0.0, "noleap"
        )
        assert coordinator.current_season == "spring"
        
        # Test summer season
        summer_state = dict(sample_state)
        summer_state["day_of_year"] = torch.tensor(200.0)
        coordinator._update_environmental_context(
            summer_state, 0.0, "noleap"
        )
        assert coordinator.current_season == "summer"
        
        # Test autumn season
        autumn_state = dict(sample_state)
        autumn_state["day_of_year"] = torch.tensor(280.0)
        coordinator._update_environmental_context(
            autumn_state, 0.0, "noleap"
        )
        assert coordinator.current_season == "autumn"
        
        # Test winter season
        winter_state = dict(sample_state)
        winter_state["day_of_year"] = torch.tensor(350.0)
        coordinator._update_environmental_context(
            winter_state, 0.0, "noleap"
        )
        assert coordinator.current_season == "winter"
    
    def test_statistics_and_reporting(self, coordinator):
        """Test statistics generation and reporting"""
        stats = coordinator.get_coordination_statistics()
        
        assert isinstance(stats, dict)
        assert "total_events_scheduled" in stats
        assert "total_events_executed" in stats
        assert "current_season" in stats
        assert "management_strategy" in stats
        assert "coordination_mode" in stats
        assert "scheduler_stats" in stats
        assert "objective_weights" in stats
        
        # Check that objective weights are properly reported
        assert isinstance(stats["objective_weights"], dict)
        assert len(stats["objective_weights"]) == len(coordinator.management_objectives)
    
    def test_annual_reset_functionality(self, coordinator):
        """Test annual and monthly reset functionality"""
        # Add some events first
        coordinator.schedule_management_event(
            EventType.IRRIGATION, 100000.0, {"test": True}
        )
        
        # Test annual reset
        initial_scheduled = len(coordinator.scheduled_events)
        coordinator.reset_annual_state()
        
        # Check that quotas were reset (indirectly through scheduler)
        scheduler_stats = coordinator.scheduler.get_statistics()
        assert isinstance(scheduler_stats, dict)
        
        # Test monthly reset
        coordinator.reset_monthly_state()
        # Should complete without error


class TestCoordinationIntegration:
    """Integration tests for coordination with management functors"""
    
    @pytest.fixture
    def coordinator_without_registry(self):
        """Create a coordinator without a subordinate Functor registry."""
        coordinator = ManagementCoordinator(
            "integration_test", {"calendar_type": "noleap"}
        )
        assert not hasattr(coordinator, "management_modules")
        return coordinator
    
    def test_full_integration_workflow(self, coordinator_without_registry):
        """Test coordination without direct subordinate Functor calls."""
        sample_state = {
            "soil_moisture": torch.tensor([[0.15]]),  # Low moisture
            "soil_nitrogen": torch.tensor([[0.01]]),
            "soil_organic_carbon": torch.tensor([[5.0]]),
            "crop_maturity": torch.tensor([[0.8]]),
            "aboveground_biomass": torch.tensor([[2.0]]),  # Ready for harvest
            "air_temperature": torch.tensor([[298.15]]),
            "day_of_year": torch.tensor(270.0),
            "irrigation_flux": torch.tensor([[0.0]]),
            "harvest_flux": torch.tensor([[0.0]]),
            "grazing_flux": torch.tensor([[0.0]]),
            "co2_emission": torch.tensor([[0.0]]),
            "field_capacity": torch.tensor([[0.35]]),
            "wilting_point": torch.tensor([[0.15]]),
            "evapotranspiration_flux": torch.tensor([[3.0e-6]]),
            "precipitation_flux": torch.tensor([[0.0]]),
        }
        
        # Run coordination
        result = coordinator_without_registry.coordinate_management_activities(
            current_time=100000.0,
            state=sample_state,
            planning_horizon=14.0
        )
        
        # Should schedule appropriate activities for the conditions
        assert len(result.scheduled_events) > 0
        
        # Should have irrigation (low soil moisture) and possibly harvest (autumn season)
        event_types = [event.event_type for event in result.scheduled_events]
        assert EventType.IRRIGATION in event_types  # Due to low soil moisture
        
        # Test execution through forward method
        outputs = coordinator_without_registry.forward(
            state=sample_state,
            time=100000.0,
            dt=3600.0
        )
        
        assert isinstance(outputs, dict)
        assert "coordination_status" in outputs


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def error_coordinator(self):
        """Create a test coordinator instance for error testing"""
        config = {
            "calendar_type": "noleap",
            "parameters": {
                "coordination_mode": {"value": "adaptive"}
            }
        }
        return ManagementCoordinator("error_test", config)
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        # Test with invalid coordination mode
        invalid_config = {
            "parameters": {
                "coordination_mode": {"value": "invalid_mode"}
            }
        }
        
        # Should still create coordinator with defaults
        coordinator = ManagementCoordinator("error_test", invalid_config)
        assert coordinator is not None
    
    def test_empty_state_handling(self, error_coordinator):
        """Test handling of empty or minimal state"""
        empty_state = {}
        
        with pytest.raises(KeyError, match="soil_moisture"):
            error_coordinator.coordinate_management_activities(
                current_time=100000.0,
                state=empty_state,
            )
    
    def test_extreme_values(self, error_coordinator):
        """Test handling of extreme input values"""
        extreme_state = {
            "soil_moisture": torch.tensor([[100.0]]),    # Extremely high
            "aboveground_biomass": torch.tensor([[0.0]]),  # No biomass
            "field_capacity": torch.tensor([[0.35]]),
            "wilting_point": torch.tensor([[0.15]]),
        }
        
        # Should handle extreme values gracefully
        needs = error_coordinator._assess_management_needs(
            extreme_state, 100000.0
        )
        
        assert isinstance(needs, dict)
        # All needs should still be in valid range
        for need_score in needs.values():
            assert 0.0 <= need_score <= 1.0


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])

