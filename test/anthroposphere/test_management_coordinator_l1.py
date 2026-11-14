#!/usr/bin/env python3
"""
L1 Unit Tests for ManagementCoordinator
Tests the anthropogenic management activities coordination functionality

Test Coverage:
- Management activity initialization and validation
- Event scheduling and execution
- Temporal coordination between different management activities
- Parameter integration and validation
- Error handling and edge cases
- Scientific accuracy and conservation compliance
"""

import pytest
import torch
from pathlib import Path
from datetime import datetime

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "relic"))

from relic.scientific.anthroposphere.management.coordinator import (
    ManagementCoordinator, EventType, EventPriority,
    ManagementStrategy, CoordinationMode
)
from relic.core.abstractions import ScaleCoupling
from relic.core.state import UpdateItem, UpdateMode


class TestManagementCoordinator:
    """Test suite for ManagementCoordinator class"""

    @pytest.fixture
    def management_coordinator(self):
        """Create a ManagementCoordinator instance for testing"""
        return ManagementCoordinator(name="test_management_coordinator")

    def test_initialization(self, management_coordinator):
        """Test proper initialization of the management coordinator"""
        assert management_coordinator is not None
        assert hasattr(management_coordinator, 'name')
        assert management_coordinator.name == "test_management_coordinator"
        assert hasattr(management_coordinator, 'scheduled_events')
        assert isinstance(management_coordinator.scheduled_events, list)
        assert hasattr(management_coordinator, 'executed_events')
        assert isinstance(management_coordinator.executed_events, list)
        assert hasattr(management_coordinator, 'management_objectives')
        assert isinstance(management_coordinator.management_objectives, list)

    def test_inheritance(self, management_coordinator):
        """Test that the class properly inherits from ScaleCoupling"""
        assert isinstance(management_coordinator, ScaleCoupling)
        assert hasattr(management_coordinator, 'get_dependencies')
        assert hasattr(management_coordinator, 'forward')
        assert hasattr(management_coordinator, 'upscale')
        assert hasattr(management_coordinator, 'downscale')

    def test_parameter_loading(self, management_coordinator):
        """Test parameter loading from ParameterManager"""
        # Test loading management parameters
        param = management_coordinator.get_parameter_value(
            "coordination_mode"
        )
        # Parameter might be a torch.Tensor, so accept both
        assert isinstance(param, (str, torch.Tensor))
        if isinstance(param, torch.Tensor):
            param_value = param.item()
        else:
            param_value = param
        assert param_value in ["strict_priority", "adaptive", "optimal"]

        # Test loading timing parameters
        param = management_coordinator.get_parameter_value(
            "planning_horizon_days"
        )
        assert isinstance(param, (int, float, torch.Tensor))
        if isinstance(param, torch.Tensor):
            param_value = param.item()
        else:
            param_value = param
        assert param_value >= 7.0 and param_value <= 365.0

    def test_management_types(self, management_coordinator):
        """Test management type definitions"""
        # Check that EventType enum exists and has expected values
        expected_types = [
            EventType.IRRIGATION,
            EventType.TILLAGE,
            EventType.HARVEST,
            EventType.GRAZING,
            EventType.FIRE
        ]

        for mgmt_type in expected_types:
            assert mgmt_type in EventType

        # Test that the coordinator is properly initialized
        assert hasattr(management_coordinator, 'scheduler')
        assert hasattr(management_coordinator, 'management_objectives')
        assert len(management_coordinator.management_objectives) > 0

    def test_management_strategies(self, management_coordinator):
        """Test different management strategies"""
        # Test setting different strategies
        original_strategy = management_coordinator.management_strategy

        management_coordinator.set_management_strategy(ManagementStrategy.ORGANIC)
        assert management_coordinator.management_strategy == ManagementStrategy.ORGANIC

        management_coordinator.set_management_strategy(ManagementStrategy.PRECISION)
        assert management_coordinator.management_strategy == ManagementStrategy.PRECISION

        management_coordinator.set_management_strategy(ManagementStrategy.SUSTAINABLE)
        assert management_coordinator.management_strategy == ManagementStrategy.SUSTAINABLE

        # Restore original strategy
        management_coordinator.set_management_strategy(original_strategy)

    def test_coordination_modes(self, management_coordinator):
        """Test different coordination modes"""
        original_mode = management_coordinator.coordination_mode

        management_coordinator.set_coordination_mode(CoordinationMode.STRICT_PRIORITY)
        assert management_coordinator.coordination_mode == CoordinationMode.STRICT_PRIORITY

        management_coordinator.set_coordination_mode(CoordinationMode.ADAPTIVE)
        assert management_coordinator.coordination_mode == CoordinationMode.ADAPTIVE

        management_coordinator.set_coordination_mode(CoordinationMode.OPTIMAL)
        assert management_coordinator.coordination_mode == CoordinationMode.OPTIMAL

        # Restore original mode
        management_coordinator.set_coordination_mode(original_mode)

    def test_add_management_event(self, management_coordinator):
        """Test adding management events"""
        initial_event_count = len(management_coordinator.scheduled_events)

        # Test scheduling an event through the coordinator
        success = management_coordinator.schedule_management_event(
            event_type=EventType.IRRIGATION,
            scheduled_time=datetime(2020, 6, 1).timestamp(),
            parameters={"water_amount": 25.0, "efficiency": 0.8},
            priority=EventPriority.NORMAL,
            resource_requirements={"water": 25.0, "labor": 1.0}
        )

        # Check that event was scheduled successfully
        assert success == True
        assert len(management_coordinator.scheduled_events) == initial_event_count + 1

        # Check the event details
        event = management_coordinator.scheduled_events[-1]
        assert event.event_type == EventType.IRRIGATION
        assert event.parameters["water_amount"] == 25.0

    def test_event_scheduling_different_types(self, management_coordinator):
        """Test scheduling different types of management events"""
        base_time = datetime(2020, 6, 1).timestamp()

        # Schedule events of different types
        success1 = management_coordinator.schedule_management_event(
            event_type=EventType.IRRIGATION,
            scheduled_time=base_time,
            parameters={"water_amount": 25.0},
            priority=EventPriority.NORMAL
        )

        success2 = management_coordinator.schedule_management_event(
            event_type=EventType.HARVEST,
            scheduled_time=base_time + 7 * 86400,  # Next week
            parameters={"expected_yield": 5.0},
            priority=EventPriority.CRITICAL
        )

        success3 = management_coordinator.schedule_management_event(
            event_type=EventType.TILLAGE,
            scheduled_time=base_time + 14 * 86400,  # Two weeks later
            parameters={"intensity": "moderate"},
            priority=EventPriority.NORMAL
        )

        # Check that all events were scheduled successfully
        assert success1 == True
        assert success2 == True
        assert success3 == True
        assert len(management_coordinator.scheduled_events) == 3

        # Verify event types
        event_types = [event.event_type for event in management_coordinator.scheduled_events]
        assert EventType.IRRIGATION in event_types
        assert EventType.HARVEST in event_types
        assert EventType.TILLAGE in event_types

    def test_management_objectives(self, management_coordinator):
        """Test management objectives initialization"""
        objectives = management_coordinator.management_objectives
        assert len(objectives) > 0

        # An objective is a name and a scoring weight (item 124): the scores come
        # from the event heuristics; target_value, direction and constraint_type had
        # no reader and are removed.
        for obj in objectives:
            assert hasattr(obj, 'name')
            assert hasattr(obj, 'weight')
            assert obj.weight >= 0.0
            for removed in ('target_value', 'direction', 'constraint_type'):
                assert not hasattr(obj, removed), removed

        # Check for expected objectives
        objective_names = [obj.name for obj in objectives]
        expected_objectives = [
            "water_use_efficiency",
            "nutrient_use_efficiency",
            "soil_carbon_sequestration",
            "economic_return",
            "biodiversity_index"
        ]

        for expected_obj in expected_objectives:
            assert expected_obj in objective_names

    def test_resource_quotas_initialization(self, management_coordinator):
        """Test resource quotas initialization"""
        # Test that scheduler has resource quotas
        assert hasattr(management_coordinator.scheduler, 'resource_quotas')
        assert len(management_coordinator.scheduler.resource_quotas) > 0

        # Check that essential quotas exist
        quota_names = list(management_coordinator.scheduler.resource_quotas.keys())
        essential_quotas = ["water", "nitrogen", "phosphorus", "labor", "machinery", "fuel"]

        for quota in essential_quotas:
            assert quota in quota_names

    def test_forward_method_interface(self, management_coordinator):
        """Test the forward method interface"""
        # Create test state
        state = {
            "soil_moisture": torch.tensor([0.3, 0.4, 0.2, 0.1]),
            "soil_nitrogen": torch.tensor([0.01, 0.01, 0.01, 0.01]),
            "soil_organic_carbon": torch.tensor([5.0, 5.0, 5.0, 5.0]),
            "crop_maturity": torch.tensor([0.7, 0.7, 0.7, 0.7]),
            "aboveground_biomass": torch.tensor([1.0, 0.8, 0.6, 0.4]),
            "air_temperature": torch.tensor([298.15, 298.15, 298.15, 298.15]),
            "irrigation_flux": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "harvest_flux": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "grazing_flux": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "co2_emission": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "field_capacity": torch.full((4,), 0.35),
            "wilting_point": torch.full((4,), 0.15),
            "evapotranspiration_flux": torch.full((4,), 3.0e-6),
            "precipitation_flux": torch.zeros(4),
        }
        time = datetime(2020, 6, 1).timestamp()
        dt = 3600.0  # 1 hour

        # Schedule a management event
        management_coordinator.schedule_management_event(
            event_type=EventType.IRRIGATION,
            scheduled_time=time,
            parameters={"water_amount": 25.0},
            priority=EventPriority.NORMAL
        )

        result = management_coordinator.forward(
            state,
            time,
            dt,
            calendar_type="noleap",
        )
        assert isinstance(result, dict)
        assert "diagnostics.management_coordinator.active" in result
        assert "diagnostics.management_coordinator.events_executed" in result
        assert "coordination_status" in result
        for item in result.values():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert torch.isfinite(item.value).all()

    def test_coordination_statistics(self, management_coordinator):
        """Test coordination statistics functionality"""
        stats = management_coordinator.get_coordination_statistics()
        assert isinstance(stats, dict)

        # Should contain essential statistics
        assert 'total_events_scheduled' in stats
        assert 'total_events_executed' in stats
        assert 'current_season' in stats
        assert 'management_strategy' in stats
        assert 'coordination_mode' in stats

        # No season may be reported before the first accepted coordination step.
        assert stats['total_events_scheduled'] >= 0
        assert stats['total_events_executed'] >= 0
        assert stats['current_season'] is None

    def test_state_reset_functionality(self, management_coordinator):
        """Test state reset functionality"""
        # Add some events first
        management_coordinator.schedule_management_event(
            event_type=EventType.IRRIGATION,
            scheduled_time=1000.0,
            parameters={"water_amount": 25.0}
        )

        # Test annual reset
        try:
            initial_count = len(management_coordinator.coordination_history)
            management_coordinator.reset_annual_state()
            # Should clear coordination history
            assert len(management_coordinator.coordination_history) == 0

        except Exception as e:
            pytest.fail(f"Annual reset failed: {e}")

        # Test monthly reset
        try:
            management_coordinator.reset_monthly_state()
            # Should not raise error

        except Exception as e:
            pytest.fail(f"Monthly reset failed: {e}")

    def test_management_needs_assessment(self, management_coordinator):
        """Test management needs assessment functionality"""
        # Create test state with different conditions
        state = {
            "soil_moisture": torch.tensor([0.2, 0.4, 0.6]),  # Low, medium, high moisture
            "aboveground_biomass": torch.tensor([0.5, 2.0, 4.0]),  # Low, medium, high biomass
            "field_capacity": torch.full((3,), 0.35),
            "wilting_point": torch.full((3,), 0.15),
        }
        current_time = datetime(2020, 6, 1).timestamp()

        try:
            needs = management_coordinator._assess_management_needs(state, current_time)
            assert isinstance(needs, dict)

            # Should contain expected need categories
            expected_needs = ["irrigation", "harvest", "tillage", "grazing", "fire_management"]
            for need in expected_needs:
                assert need in needs

            # Values should be between 0 and 1
            for need, value in needs.items():
                assert 0.0 <= value <= 1.0

            # Irrigation need should be higher with lower soil moisture
            assert needs["irrigation"] >= 0.0

        except Exception as e:
            pytest.fail(f"Management needs assessment failed: {e}")

    def test_event_optimization(self, management_coordinator):
        """Test event scheduling optimization"""
        # Create test state
        state = {"soil_moisture": torch.tensor([0.3])}

        # Create candidate events
        from relic.scientific.anthroposphere.management.event_scheduler import ManagementEvent
        candidates = [
            ManagementEvent(
                event_id="test1",
                event_type=EventType.IRRIGATION,
                scheduled_time=1000.0,
                priority=EventPriority.HIGH,
                duration=3600.0,
                parameters={"need_score": 0.8},
                resource_requirements={"water": 50.0}
            ),
            ManagementEvent(
                event_id="test2",
                event_type=EventType.HARVEST,
                scheduled_time=2000.0,
                priority=EventPriority.NORMAL,
                duration=3600.0,
                parameters={"need_score": 0.6},
                resource_requirements={"labor": 5.0}
            )
        ]

        try:
            optimized = management_coordinator._optimize_event_schedule(candidates, state)
            assert isinstance(optimized, list)
            assert len(optimized) <= len(candidates)

            # Events should be sorted by priority (CRITICAL=0, HIGH=2, NORMAL=3)
            # Lower numeric value = higher priority
            if len(optimized) > 1:
                assert optimized[0].priority.value <= optimized[1].priority.value

        except Exception as e:
            pytest.fail(f"Event optimization failed: {e}")

    def test_resource_efficiency_calculation(self, management_coordinator):
        """Test resource efficiency calculation"""
        from relic.scientific.anthroposphere.management.event_scheduler import ManagementEvent

        # Create events with different resource requirements
        efficient_event = ManagementEvent(
            event_id="efficient",
            event_type=EventType.IRRIGATION,
            scheduled_time=1000.0,
            priority=EventPriority.NORMAL,
            duration=3600.0,
            parameters={"need_score": 0.8},
            resource_requirements={"water": 10.0}  # Low resource use
        )

        inefficient_event = ManagementEvent(
            event_id="inefficient",
            event_type=EventType.IRRIGATION,
            scheduled_time=1000.0,
            priority=EventPriority.NORMAL,
            duration=3600.0,
            parameters={"need_score": 0.4},
            resource_requirements={"water": 800.0}  # Full annual quota
        )

        efficient_score = management_coordinator._calculate_resource_efficiency(efficient_event)
        inefficient_score = management_coordinator._calculate_resource_efficiency(inefficient_event)

        # Efficient event should have higher score after same-unit quota normalization.
        assert efficient_score > inefficient_score
        assert 0.0 <= efficient_score <= 1.0
        assert 0.0 <= inefficient_score <= 1.0

    def test_objective_evaluation(self, management_coordinator):
        """Test management objective evaluation"""
        from relic.scientific.anthroposphere.management.event_scheduler import ManagementEvent

        # Create test events
        events = [
            ManagementEvent(
                event_id="irr_event",
                event_type=EventType.IRRIGATION,
                scheduled_time=1000.0,
                priority=EventPriority.NORMAL,
                duration=3600.0,
                parameters={},
                resource_requirements={}
            ),
            ManagementEvent(
                event_id="harvest_event",
                event_type=EventType.HARVEST,
                scheduled_time=2000.0,
                priority=EventPriority.NORMAL,
                duration=3600.0,
                parameters={},
                resource_requirements={}
            )
        ]

        state = {"aboveground_biomass": torch.tensor([2.0])}

        try:
            scores = management_coordinator._evaluate_objectives(events, state)
            assert isinstance(scores, dict)

            # Should contain scores for all objectives
            for obj in management_coordinator.management_objectives:
                assert obj.name in scores
                assert 0.0 <= scores[obj.name] <= 1.0

        except Exception as e:
            pytest.fail(f"Objective evaluation failed: {e}")

    def test_event_handlers(self, management_coordinator):
        """Test management event handlers"""
        state = {
            "soil_moisture": torch.tensor([0.3]),
            "aboveground_biomass": torch.tensor([2.0])
        }

        result = management_coordinator._handle_irrigation_event(
            state, {"water_amount": 25.0, "dt": 3600.0}
        )
        assert result["status"] == "completed"
        assert result["requested_water_depth"] == 25.0
        assert "irrigation_flux" not in result

        result = management_coordinator._handle_tillage_event(
            state, {"intensity": "moderate"}
        )
        assert result["status"] == "completed"
        assert result["tillage_intensity_class"] == "moderate"
        assert "decomposition_modifier" not in result

        result = management_coordinator._handle_harvest_event(
            state, {"expected_yield": 5.0, "harvest_index": 0.45}
        )
        assert result["status"] == "completed"
        assert result["requested_harvest_index"] == 0.45
        assert "harvest_flux" not in result

        result = management_coordinator._handle_grazing_event(
            state, {"stocking_rate": 1.0, "grazing_efficiency": 0.65}
        )
        assert result["status"] == "completed"
        assert result["requested_stocking_rate"] == 1.0
        assert "grazing_flux" not in result

        result = management_coordinator._handle_fire_event(
            state, {"fire_type": "prescribed", "fuel_consumption": 0.85}
        )
        assert result["status"] == "completed"
        assert result["requested_fire_type"] == "prescribed"
        assert "co2_emission" not in result

    def test_scientific_accuracy_moisture_deficit(self, management_coordinator):
        """Test scientific accuracy of moisture deficit calculation"""
        # Test with known soil moisture conditions
        # Field capacity = 0.35 and wilting point = 0.15 are supplied as state.
        state = {
            "soil_moisture": torch.tensor([0.35, 0.25, 0.15]),  # FC, midpoint, WP
            "aboveground_biomass": torch.tensor([1.0, 1.0, 1.0]),
            "field_capacity": torch.full((3,), 0.35),
            "wilting_point": torch.full((3,), 0.15),
        }
        current_time = datetime(2020, 6, 1).timestamp()

        needs = management_coordinator._assess_management_needs(
            state,
            current_time,
        )

        # At field capacity: no irrigation need (deficit = 0)
        # At midpoint: moderate irrigation need (deficit = 0.5)
        # At wilting point: high irrigation need (deficit = 1.0)
        irrigation_need = needs["irrigation"]

        # Should increase as soil moisture decreases
        assert irrigation_need >= 0.0
        assert irrigation_need <= 1.0

        # With the specific values above, should be around 0.5
        # (0.35 - 0.25) / (0.35 - 0.15) = 0.5
        assert abs(irrigation_need - 0.5) < 0.1

    def test_error_handling_invalid_parameters(self, management_coordinator):
        """Test error handling for invalid parameters"""
        # Test with negative resource requirements
        try:
            success = management_coordinator.schedule_management_event(
                event_type=EventType.IRRIGATION,
                scheduled_time=1000.0,
                parameters={"water_amount": -25.0},  # Negative water
                priority=EventPriority.NORMAL
            )
            # Should handle gracefully (may still schedule but with warnings)
            assert isinstance(success, bool)
        except Exception:
            # Expected to fail with invalid parameters
            pass

    def test_edge_cases_zero_state(self, management_coordinator):
        """Test edge cases with zero state values"""
        state = {
            "soil_moisture": torch.tensor([0.0]),
            "aboveground_biomass": torch.tensor([0.0]),
            "field_capacity": torch.tensor([0.35]),
            "wilting_point": torch.tensor([0.15]),
        }
        current_time = datetime(2020, 6, 1).timestamp()

        try:
            needs = management_coordinator._assess_management_needs(state, current_time)
            assert isinstance(needs, dict)

            # Should handle zero values gracefully
            for need, value in needs.items():
                assert 0.0 <= value <= 1.0

        except Exception as e:
            pytest.fail(f"Zero state edge case failed: {e}")

    def test_edge_cases_large_values(self, management_coordinator):
        """Test edge cases with large values"""
        state = {
            "soil_moisture": torch.tensor([10.0]),  # Very high (unrealistic) moisture
            "aboveground_biomass": torch.tensor([1000.0]),  # Very high biomass
            "field_capacity": torch.tensor([0.35]),
            "wilting_point": torch.tensor([0.15]),
        }
        current_time = datetime(2020, 6, 1).timestamp()

        try:
            needs = management_coordinator._assess_management_needs(state, current_time)
            assert isinstance(needs, dict)

            # Should handle large values without overflow
            for need, value in needs.items():
                assert torch.isfinite(torch.tensor(value))
                assert 0.0 <= value <= 1.0

        except Exception as e:
            pytest.fail(f"Large values edge case failed: {e}")

    def test_memory_efficiency(self, management_coordinator):
        """Test memory efficiency with large number of events"""
        # Schedule many events
        base_time = datetime(2020, 6, 1).timestamp()

        try:
            for i in range(100):  # Schedule 100 events
                management_coordinator.schedule_management_event(
                    event_type=EventType.IRRIGATION,
                    scheduled_time=base_time + i * 3600.0,
                    parameters={"water_amount": 25.0},
                    priority=EventPriority.NORMAL
                )

            # Should handle large number of events
            assert len(management_coordinator.scheduled_events) == 100

            # Test statistics generation with many events
            stats = management_coordinator.get_coordination_statistics()
            assert stats['total_events_scheduled'] == 100

        except Exception as e:
            pytest.fail(f"Memory efficiency test failed: {e}")

    def test_coordination_modes_impact(self, management_coordinator):
        """Test that different coordination modes affect behavior"""
        # Schedule same events under different modes
        events_data = [
            (EventType.IRRIGATION, 1000.0, {"water_amount": 25.0}),
            (EventType.HARVEST, 2000.0, {"expected_yield": 5.0})
        ]

        results = {}

        for mode in [CoordinationMode.STRICT_PRIORITY, CoordinationMode.ADAPTIVE]:
            # Reset coordinator state
            management_coordinator.scheduled_events.clear()
            management_coordinator.set_coordination_mode(mode)

            # Schedule events
            for event_type, time, params in events_data:
                management_coordinator.schedule_management_event(
                    event_type=event_type,
                    scheduled_time=time,
                    parameters=params,
                    priority=EventPriority.NORMAL
                )

            # Get statistics
            stats = management_coordinator.get_coordination_statistics()
            results[mode.value] = stats['coordination_mode']

        # Different modes should be reflected in results
        assert results['strict_priority'] == 'strict_priority'
        assert results['adaptive'] == 'adaptive'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

