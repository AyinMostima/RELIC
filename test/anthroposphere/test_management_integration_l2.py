#!/usr/bin/env python3
"""
L2 Integration Test for Anthroposphere Management
Tests the interaction between ManagementCoordinator and specific management modules.
"""

import pytest
import torch
import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator, EventType
from relic.scientific.anthroposphere.management.event_scheduler import EventPriority
from relic.core.state import UpdateItem
from relic.core.state import StateManager
from relic.core.simulator import RELICSimulator


def _value(item):
    assert isinstance(item, UpdateItem)
    return item.value

class TestManagementIntegration:
    
    @pytest.fixture
    def coordinator(self):
        return ManagementCoordinator(
            name="test_coordinator",
            config={"calendar_type": "noleap"},
        )
        
    @pytest.fixture
    def mock_state(self):
        batch_size = 10
        return {
            "day_of_year": torch.tensor(120.0), # Global simulator calendar coordinate
            "soil_moisture": torch.tensor([0.3] * batch_size),
            "soil_nitrogen": torch.tensor([0.01] * batch_size),
            "soil_organic_carbon": torch.tensor([5.0] * batch_size),
            "crop_maturity": torch.tensor([0.6] * batch_size),
            "aboveground_biomass": torch.tensor([2.0] * batch_size), # kg/m2
            "air_temperature": torch.tensor([290.0] * batch_size),
            "belowground_biomass": torch.tensor([1.0] * batch_size),
            "litter_mass": torch.tensor([0.5] * batch_size),
            "vegetation_quality": torch.tensor([0.8] * batch_size),
            "fuel_moisture": torch.tensor([0.15] * batch_size),
            "days_since_fire": torch.tensor([30.0] * batch_size),
            # Placeholder fluxes for aggregation
            "irrigation_flux": torch.zeros(batch_size),
            "harvest_flux": torch.zeros(batch_size),
            "grazing_flux": torch.zeros(batch_size),
            "co2_emission": torch.zeros(batch_size),
            "field_capacity": torch.full((batch_size,), 0.35),
            "wilting_point": torch.full((batch_size,), 0.15),
            "evapotranspiration_flux": torch.full((batch_size,), 3.0e-6),
            "precipitation_flux": torch.zeros(batch_size),
        }

    def test_coordinator_assessment(self, coordinator, mock_state):
        """Test that coordinator correctly assesses management needs."""
        # Set conditions that should trigger needs
        mock_state["soil_moisture"] = torch.tensor([0.15] * 10) # Low moisture -> Irrigation need
        mock_state["aboveground_biomass"] = torch.tensor([3.0] * 10) # High biomass -> Harvest/Grazing need
        coordinator._update_environmental_context(
            mock_state,
            current_time=0.0,
            calendar_type="noleap",
        )
        needs = coordinator._assess_management_needs(mock_state, 0.0)
        
        assert "irrigation" in needs
        assert needs["irrigation"] > 0.0
        assert "grazing" in needs
        assert coordinator.current_season == "spring"
        assert needs["grazing"] > 0.0

    def test_grazing_execution_via_coordinator(self, coordinator, mock_state):
        """Test scheduling and executing a grazing event."""
        # Schedule a grazing event
        success = coordinator.schedule_management_event(
            EventType.GRAZING,
            scheduled_time=0.0,
            parameters={"stocking_rate": 2.0},
            priority=EventPriority.HIGH
        )
        assert success
        
        result = coordinator.forward(mock_state, time=0.0, dt=86400.0)
        assert _value(result["diagnostics.management_coordinator.active"]) > 0.0
        assert _value(result["diagnostics.management_coordinator.events_executed"]) == 1.0
        assert _value(
            result["diagnostics.management_coordinator.grazing_events_count"]
        ) == 1.0
        assert _value(result["total_grazing_flux"]).sum() == 0.0

        # The next accepted state carries the only canonical process flux.
        next_state = dict(mock_state)
        next_state["grazing_flux"] = torch.full_like(
            mock_state["grazing_flux"],
            2.5e-8,
        )
        next_result = coordinator.forward(next_state, time=1.0, dt=86400.0)
        assert torch.equal(
            _value(next_result["total_grazing_flux"]),
            next_state["grazing_flux"],
        )

    def test_fire_scheduling(self, coordinator, mock_state):
        """Test fire event scheduling."""
        # Force a fire event
        success = coordinator.schedule_management_event(
            EventType.FIRE,
            scheduled_time=0.0,
            parameters={"fire_type": "prescribed", "fuel_consumption": 0.5}
        )
        assert success
        
        result = coordinator.forward(mock_state, time=0.0, dt=3600.0)
        assert _value(
            result["diagnostics.management_coordinator.fire_events_count"]
        ) == 1.0
        assert _value(result["total_fire_co2_flux"]).sum() == 0.0

        next_state = dict(mock_state)
        next_state["co2_emission"] = torch.full_like(
            mock_state["co2_emission"],
            1.0e-8,
        )
        next_result = coordinator.forward(next_state, time=1.0, dt=3600.0)
        assert torch.equal(
            _value(next_result["total_fire_co2_flux"]),
            next_state["co2_emission"],
        )

    def test_real_simulator_scm_commit_and_provenance(self, mock_state):
        """Canonical outputs must retain value, unit, writer, and device evidence."""
        direct_model = ManagementCoordinator(config={"calendar_type": "noleap"})
        direct = direct_model.forward(mock_state, time=0.0, dt=3600.0)

        state_manager = StateManager(device="cpu")
        state_manager.set_state(mock_state, validate=False)
        simulator = RELICSimulator(
            {
                "max_time": 3600.0,
                "time_step": 3600.0,
                "conservation_checks": False,
                "strict_dependencies": False,
                "device": "cpu",
            }
        )
        simulator.set_state_manager(state_manager)
        runtime_model = ManagementCoordinator()
        simulator.register_functor(runtime_model, execution_group="fast")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runtime_outputs = simulator.step(dt=3600.0)

        assert set(runtime_outputs) == set(runtime_model.OUTPUT_UNITS)
        committed_names = {
            name for name in runtime_model.OUTPUT_UNITS
            if not name.startswith("diagnostics.")
        }
        diagnostic_names = set(runtime_model.OUTPUT_UNITS) - committed_names
        assert len(committed_names) == 8
        assert len(diagnostic_names) == 12

        final_state = state_manager.get_state()
        for name in committed_names:
            expected = direct[name].value.to(
                device=final_state[name].device,
                dtype=final_state[name].dtype,
            )
            torch.testing.assert_close(final_state[name], expected)
            spec = state_manager.get_variable_spec(name)
            assert spec is not None
            assert spec.units == runtime_model.OUTPUT_UNITS[name]
            provenance = simulator.get_variable_provenance(name)
            assert {entry["functor"] for entry in provenance} == {runtime_model.name}
            assert {entry["role"] for entry in provenance} == {"writer", "final_writer"}

        for name in diagnostic_names:
            assert name not in final_state
            assert simulator.get_variable_provenance(name) == []

        assert simulator.get_effective_device(runtime_model.name).type == "cpu"
        assert caught == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

