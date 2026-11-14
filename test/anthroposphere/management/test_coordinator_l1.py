#!/usr/bin/env python3
"""
L1 Unit Tests for Management Coordinator Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy (coordination logic)
- L1.4: Conservation laws validation
- L1.5: Management strategy coordination
- L1.6: Abstract base class compliance

Scientific References:
- Lal, R. (2004). Soil carbon sequestration to mitigate climate change. Geoderma 123, 1-22.
- Robertson, G.P., & Swinton, S.M. (2005). Reconciling agricultural productivity and environmental integrity. CRC Press.
- Matson, P.A., et al. (1997). Agricultural intensification and ecosystem properties. Science 277, 504-509.

Test Framework: pytest
"""

import copy
import heapq

import pytest
import torch
import sys
import os
import logging
import warnings

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
import relic.scientific.anthroposphere.management.coordinator as coordinator_module
from relic.scientific.anthroposphere.management.coordinator import (
    CoordinationResult,
    ManagementCoordinator,
    ManagementStrategy,
)
from relic.scientific.anthroposphere.management.event_scheduler import (
    EventScheduler,
    EventPriority,
    EventType,
    ManagementEvent,
    ResourceQuota,
    resolve_day_of_year,
)
from relic.scientific.anthroposphere.management.fertilization import FertilizationManagement
from relic.scientific.anthroposphere.management.irrigation import IrrigationManagement
from relic.scientific.anthroposphere.management.tillage import TillageManagement
from relic.core.abstractions import ScaleCoupling
from relic.core.functor import ScaleCouplingFunctor
from relic.core.state import UpdateItem, UpdateMode

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3

CANONICAL_OUTPUT_UNITS = {
    "coordination_status": "1",
    "total_irrigation_flux": "kg m-2 s-1",
    "total_harvest_flux": "kg DM m-2 s-1",
    "total_grazing_flux": "kg DM m-2 s-1",
    "total_fire_co2_flux": "kg CO2 m-2 s-1",
    "irrigation_demand_flux": "kg m-2 s-1",
    "fertilizer_product_flux": "kg product m-2 s-1",
    "tillage_intensity": "1",
    "diagnostics.management_coordinator.active": "1",
    "diagnostics.management_coordinator.events_executed": "1",
    "diagnostics.management_coordinator.irrigation_events_count": "1",
    "diagnostics.management_coordinator.tillage_events_count": "1",
    "diagnostics.management_coordinator.harvest_events_count": "1",
    "diagnostics.management_coordinator.grazing_events_count": "1",
    "diagnostics.management_coordinator.fire_events_count": "1",
    "diagnostics.management_coordinator.priority_weights_sum": "1",
    "diagnostics.management_coordinator.interaction_effects_count": "1",
    "diagnostics.management_coordinator.conflict_resolution_applied": "1",
    "diagnostics.coordinator_mass_balance_error": "kg DM m-2 s-1",
    "diagnostics.management_coordinator.canonical_iphs_runtime": "1",
}

CANONICAL_REQUIRED_INPUTS = {
    "soil_moisture",
    "soil_nitrogen",
    "soil_organic_carbon",
    "crop_maturity",
    "aboveground_biomass",
    "air_temperature",
    "irrigation_flux",
    "harvest_flux",
    "grazing_flux",
    "co2_emission",
    "field_capacity",
    "wilting_point",
    "evapotranspiration_flux",
    "precipitation_flux",
}

CANONICAL_OPTIONAL_INPUTS = {
    "day_of_year",
}


def _value(item):
    """Return the tensor payload of an explicit state update."""

    return item.value if isinstance(item, UpdateItem) else item


def _canonical_scalar_state(**overrides):
    """Build one complete canonical coordinator state in SI-compatible units."""
    state = {
        "soil_moisture": torch.tensor([0.3], device=DEVICE),
        "soil_nitrogen": torch.tensor([0.01], device=DEVICE),
        "soil_organic_carbon": torch.tensor([5.0], device=DEVICE),
        "crop_maturity": torch.tensor([0.7], device=DEVICE),
        "aboveground_biomass": torch.tensor([1.0], device=DEVICE),
        "air_temperature": torch.tensor([298.15], device=DEVICE),
        "irrigation_flux": torch.tensor([0.0], device=DEVICE),
        "harvest_flux": torch.tensor([0.0], device=DEVICE),
        "grazing_flux": torch.tensor([0.0], device=DEVICE),
        "co2_emission": torch.tensor([0.0], device=DEVICE),
        "field_capacity": torch.tensor([0.35], device=DEVICE),
        "wilting_point": torch.tensor([0.15], device=DEVICE),
        "evapotranspiration_flux": torch.tensor([3.0e-6], device=DEVICE),
        "precipitation_flux": torch.tensor([0.0], device=DEVICE),
    }
    state.update(overrides)
    return state


def _scheduled_custom_event(event_id: str) -> ManagementEvent:
    return ManagementEvent(
        event_id=event_id,
        event_type=EventType.CUSTOM,
        scheduled_time=3600.0,
        priority=EventPriority.NORMAL,
        duration=60.0,
        parameters={"label": event_id},
    )


def test_event_scheduler_requeues_and_reraises_handler_failure():
    scheduler = EventScheduler()
    scheduler.register_resource_quota(
        ResourceQuota(
            resource_type="water",
            total_available=10.0,
            unit="kg",
        )
    )

    def fail_handler(_state, _parameters):
        raise RuntimeError("injected management handler failure")

    scheduler.register_event_handler(EventType.CUSTOM, fail_handler)
    event = _scheduled_custom_event("failing_event")
    event.resource_requirements = {"water": 2.0}
    assert scheduler.schedule_event(event) is True

    with pytest.raises(RuntimeError, match="injected management handler failure"):
        scheduler.execute_until(3600.0, {})

    assert scheduler.events == [event]
    assert event.executed is False
    assert event.execution_result is None
    assert scheduler.execution_log == []
    assert scheduler.resource_quotas["water"].current_used == pytest.approx(0.0)


def test_event_scheduler_tie_order_is_independent_of_insertion_order():
    def execute_order(event_ids):
        scheduler = EventScheduler()
        executed_ids = []

        def record_handler(_state, parameters):
            executed_ids.append(parameters["label"])
            return {"status": "completed"}

        scheduler.register_event_handler(EventType.CUSTOM, record_handler)
        for event_id in event_ids:
            assert scheduler.schedule_event(_scheduled_custom_event(event_id)) is True
        scheduler.execute_until(3600.0, {})
        return executed_ids

    expected = ["a", "b", "c", "d", "e"]
    assert execute_order(expected) == expected
    assert execute_order(reversed(expected)) == expected


def test_event_scheduler_rejects_duplicate_event_id():
    scheduler = EventScheduler()

    assert scheduler.schedule_event(_scheduled_custom_event("duplicate")) is True
    with pytest.warns(UserWarning, match="Duplicate event_id"):
        assert scheduler.schedule_event(_scheduled_custom_event("duplicate")) is False


def test_management_day_is_derived_from_simulation_time():
    """Management events must not default to a configured event date."""
    from relic.utils.time_constants import SECONDS_PER_DAY

    assert resolve_day_of_year(0.0, calendar_type="noleap") == 1
    assert resolve_day_of_year(
        119.0 * SECONDS_PER_DAY,
        calendar_type="noleap",
    ) == 120
    assert resolve_day_of_year(
        365.0 * SECONDS_PER_DAY,
        calendar_type="noleap",
    ) == 1
    assert resolve_day_of_year(
        360.0 * SECONDS_PER_DAY,
        calendar_type="360_day",
    ) == 1
    assert resolve_day_of_year(
        0.0,
        explicit_day=torch.tensor(270),
        calendar_type="noleap",
    ) == 270
    with pytest.raises(ValueError, match="day_of_year"):
        resolve_day_of_year(
            0.0,
            explicit_day=361,
            calendar_type="360_day",
        )
    with pytest.raises(TypeError, match="calendar_type"):
        resolve_day_of_year(0.0)


@pytest.mark.parametrize(
    "explicit_day",
    [
        True,
        1.5,
        float("nan"),
        torch.tensor([1, 2]),
    ],
)
def test_management_day_rejects_non_integer_or_non_scalar_explicit_day(
    explicit_day,
):
    with pytest.raises(ValueError, match="integer numeric scalar"):
        resolve_day_of_year(
            0.0,
            explicit_day=explicit_day,
            calendar_type="noleap",
        )


class TestManagementCoordinatorL1:
    """L1 Unit Tests for ManagementCoordinator"""
    
    @pytest.fixture
    def coordinator(self):
        """Create management coordinator instance for testing"""
        return ManagementCoordinator(config={"calendar_type": "noleap"})
    
    @pytest.fixture
    def management_modules(self):
        """Create individual management modules for testing"""
        return {
            'fertilization': FertilizationManagement(),
            'irrigation': IrrigationManagement(),
            'tillage': TillageManagement()
        }
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            # Soil conditions
            'soil_moisture': torch.tensor([0.22, 0.28, 0.35, 0.18, 0.25], device=DEVICE).reshape(batch_shape),
            'soil_temperature': torch.tensor([288.15, 293.15, 298.15, 303.15, 308.15], device=DEVICE).reshape(batch_shape),
            'soil_nitrogen': torch.tensor([0.008, 0.010, 0.006, 0.014, 0.009], device=DEVICE).reshape(batch_shape),
            'soil_phosphorus': torch.tensor([8.0, 10.0, 7.0, 12.0, 9.0], device=DEVICE).reshape(batch_shape),
            'soil_organic_carbon': torch.tensor([4.0, 5.0, 6.0, 3.5, 4.5], device=DEVICE).reshape(batch_shape),
            'soil_ph': torch.tensor([6.2, 6.8, 7.1, 6.5, 6.7], device=DEVICE).reshape(batch_shape),
            
            # Crop conditions
            'aboveground_biomass': torch.tensor([0.8, 1.0, 0.7, 1.2, 0.9], device=DEVICE).reshape(batch_shape),
            'belowground_biomass': torch.tensor([0.3, 0.4, 0.25, 0.5, 0.35], device=DEVICE).reshape(batch_shape),
            'leaf_area_index': torch.tensor([2.5, 3.2, 2.8, 4.1, 3.0], device=DEVICE).reshape(batch_shape),
            'crop_maturity': torch.tensor([0.6, 0.7, 0.8, 0.9, 0.5], device=DEVICE).reshape(batch_shape),
            'crop_height': torch.tensor([1.0, 1.2, 1.4, 1.6, 0.9], device=DEVICE).reshape(batch_shape),
            
            # Management inputs
            'urea_n': torch.tensor([8.0, 12.0, 5.0, 15.0, 7.0], device=DEVICE).reshape(batch_shape),
            'ammonium_n': torch.tensor([4.0, 6.0, 3.0, 8.0, 3.5], device=DEVICE).reshape(batch_shape),
            'fertilizer_application_rate': torch.tensor([0.008, 0.012, 0.005, 0.015, 0.007], device=DEVICE).reshape(batch_shape),
            
            # Environmental conditions
            'evapotranspiration_flux': torch.tensor([2.5e-6, 3.2e-6, 4.1e-6, 2.1e-6, 3.0e-6], device=DEVICE).reshape(batch_shape),
            'precipitation_flux': torch.tensor([0.8e-6, 0.5e-6, 1.2e-6, 0.3e-6, 0.7e-6], device=DEVICE).reshape(batch_shape),
            'field_capacity': torch.full(batch_shape, 0.35, device=DEVICE),
            'wilting_point': torch.full(batch_shape, 0.15, device=DEVICE),
            'air_temperature': torch.tensor([295.15, 298.15, 302.15, 293.15, 297.15], device=DEVICE).reshape(batch_shape),
            'relative_humidity': torch.tensor([0.65, 0.70, 0.60, 0.75, 0.68], device=DEVICE).reshape(batch_shape),
            'irrigation_flux': torch.zeros(batch_shape, device=DEVICE),
            'harvest_flux': torch.zeros(batch_shape, device=DEVICE),
            'grazing_flux': torch.zeros(batch_shape, device=DEVICE),
            'co2_emission': torch.zeros(batch_shape, device=DEVICE),
        }
    
    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, coordinator):
        """Test that ManagementCoordinator can be instantiated"""
        assert coordinator is not None
        assert isinstance(coordinator, ManagementCoordinator)
        assert isinstance(coordinator, ScaleCouplingFunctor)

    def test_time_constants_use_physical_constant_ssot(self, monkeypatch):
        """Coordinator time conversions must use the physical-constant API."""

        calls = []
        values = {
            "seconds_per_day": 12345.0,
            "seconds_per_hour": 678.0,
        }

        def tracked_get_physical_constant(name):
            calls.append(name)
            return values[name]

        monkeypatch.setattr(
            coordinator_module,
            "get_physical_constant",
            tracked_get_physical_constant,
        )
        coordinator = ManagementCoordinator()

        assert coordinator.seconds_per_day == pytest.approx(12345.0)
        assert coordinator.seconds_per_hour == pytest.approx(678.0)
        assert calls == ["seconds_per_day", "seconds_per_hour"]
        assert (
            "relic.physical.constants.time.seconds_per_day"
            not in coordinator.required_parameters()
        )

    def test_canonical_outputs_have_explicit_replace_semantics(
        self,
        coordinator,
        sample_state,
    ):
        """Every retained output has one explicit runtime and unit contract."""

        outputs = coordinator.forward(
            sample_state,
            time=0.0,
            dt=3600.0,
        )

        assert set(coordinator.metadata.output_variables) == set(CANONICAL_OUTPUT_UNITS)
        assert set(outputs) == set(CANONICAL_OUTPUT_UNITS)
        for name, unit in CANONICAL_OUTPUT_UNITS.items():
            item = outputs[name]
            assert isinstance(item, UpdateItem), name
            assert item.mode is UpdateMode.REPLACE, name
            assert item.per_timestep is False, name
            assert item.unit == unit, name
            assert item.source_functor == coordinator.name, name
            assert torch.all(torch.isfinite(item.value)), name
        assert torch.all(_value(outputs["tillage_intensity"]) >= 0.0)
        assert torch.all(_value(outputs["tillage_intensity"]) <= 1.0)
        forbidden = {
            "management_active",
            "events_executed",
            "total_events_count",
            "energy_consumption",
            "total_fire_emissions",
            "irrigation_rate",
            "fertilizer_rate",
        }
        assert forbidden.isdisjoint(outputs)

    def test_canonical_input_contract_is_state_only_and_fail_fast(
        self,
        coordinator,
        sample_state,
    ):
        """Dynamic state must never be supplied by aliases or parameter defaults."""

        assert set(coordinator.metadata.input_variables) == (
            CANONICAL_REQUIRED_INPUTS | CANONICAL_OPTIONAL_INPUTS
        )
        assert coordinator.optional_input_variables == CANONICAL_OPTIONAL_INPUTS

        for missing_name in sorted(CANONICAL_REQUIRED_INPUTS):
            incomplete = dict(sample_state)
            incomplete.pop(missing_name)
            with pytest.raises(KeyError, match=missing_name):
                coordinator.forward(
                    incomplete,
                    time=0.0,
                    dt=3600.0,
                )

        # air_temperature is read from the state only: an alias does not stand in
        # for it (forward takes (state, time, dt, calendar_type) only).
        aliased_temperature = dict(sample_state)
        aliased_temperature.pop("air_temperature")
        aliased_temperature["temperature"] = sample_state["air_temperature"]
        with pytest.raises(KeyError, match="air_temperature"):
            coordinator.forward(
                aliased_temperature,
                time=0.0,
                dt=3600.0,
            )

        non_tensor = dict(sample_state)
        non_tensor["soil_moisture"] = 0.3
        with pytest.raises(TypeError, match="soil_moisture"):
            coordinator.forward(non_tensor, time=0.0, dt=3600.0)

        non_finite = dict(sample_state)
        non_finite["soil_nitrogen"] = torch.full_like(
            sample_state["soil_nitrogen"], float("nan")
        )
        with pytest.raises(ValueError, match="soil_nitrogen"):
            coordinator.forward(non_finite, time=0.0, dt=3600.0)

    def test_event_handlers_emit_control_intents_not_process_fluxes(
        self,
        coordinator,
        sample_state,
    ):
        """The coordinator must not duplicate subordinate process equations."""

        cases = (
            (
                coordinator._handle_irrigation_event,
                {"water_amount": 25.0, "dt": 3600.0},
                {"irrigation_flux"},
            ),
            (
                coordinator._handle_harvest_event,
                {"harvest_index": 0.45},
                {"harvest_flux", "biomass_harvested", "residue_remaining"},
            ),
            (
                coordinator._handle_grazing_event,
                {"stocking_rate": 1.5},
                {"grazing_flux", "biomass_removed", "manure_deposited"},
            ),
            (
                coordinator._handle_fire_event,
                {"fuel_consumption": 0.85},
                {"co2_emission", "burned_biomass"},
            ),
        )

        for handler, parameters, process_flux_keys in cases:
            result = handler(sample_state, parameters)
            assert result["status"] == "completed"
            assert process_flux_keys.isdisjoint(result)

    def test_lagged_process_fluxes_are_aggregated_without_event_recalculation(
        self,
        coordinator,
    ):
        """Only canonical dependency ports contribute to aggregate outputs."""

        state = {
            "soil_moisture": torch.tensor([0.3]),
            "soil_nitrogen": torch.tensor([0.01]),
            "soil_organic_carbon": torch.tensor([5.0]),
            "crop_maturity": torch.tensor([0.75]),
            "aboveground_biomass": torch.tensor([3.0]),
            "air_temperature": torch.tensor([298.15]),
            "irrigation_flux": torch.tensor([2.0e-6]),
            "harvest_flux": torch.tensor([3.0e-7]),
            "grazing_flux": torch.tensor([4.0e-8]),
            "co2_emission": torch.tensor([5.0e-9]),
            "field_capacity": torch.full_like(torch.tensor([5.0e-9]), 0.35),
            "wilting_point": torch.full_like(torch.tensor([5.0e-9]), 0.15),
            "evapotranspiration_flux": torch.full_like(torch.tensor([5.0e-9]), 3.0e-6),
            "precipitation_flux": torch.zeros_like(torch.tensor([5.0e-9])),
        }
        outputs = coordinator.forward(state, time=0.0, dt=3600.0)

        assert torch.equal(_value(outputs["total_irrigation_flux"]), state["irrigation_flux"])
        assert torch.equal(_value(outputs["total_harvest_flux"]), state["harvest_flux"])
        assert torch.equal(_value(outputs["total_grazing_flux"]), state["grazing_flux"])
        assert torch.equal(_value(outputs["total_fire_co2_flux"]), state["co2_emission"])
        assert isinstance(coordinator, ScaleCoupling)
        assert hasattr(coordinator, 'name')
        assert coordinator.name == "management_coordinator"
    
    def test_metadata_completeness(self, coordinator):
        """Test that metadata is properly set"""
        metadata = coordinator.get_metadata()
        assert metadata is not None
        assert metadata.name == "management_coordinator"
        assert len(metadata.references) > 0
        assert "coordination" in metadata.mathematical_expression or "optimization" in metadata.mathematical_expression
        assert "mass" in metadata.conservation_laws
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration"""
        custom_config = {
            "parameters": {
                "coordination_efficiency": {"value": 0.9, "units": "1"},
                "priority_weights.nutrients": {"value": 0.4, "units": "1"}
            }
        }
        
        manager = ManagementCoordinator(config=custom_config)
        assert manager is not None
        
        # Parameters should be accessible through get_parameter_value
        efficiency = manager.get_parameter_value("coordination_efficiency")
        assert abs(efficiency - 0.9) < 1e-3
    
    def test_subordinate_modules_are_simulator_owned(self, coordinator):
        """The coordinator must not maintain a second module registry."""
        assert not hasattr(coordinator, "management_modules")
    
    # L1.2: Parameter management system validation
    def test_parameter_access_through_manager(self, coordinator):
        """Test that parameters are accessible through ParameterManager"""
        # Test getting parameters
        coordination_efficiency = coordinator.get_parameter_value("coordination_efficiency")
        assert isinstance(coordination_efficiency, (float, torch.Tensor))
        assert 0.7 <= coordination_efficiency <= 0.95
        
        priority_weight_nutrients = coordinator.get_parameter_value(
            "priority_weights.nutrients",
            namespace="anthroposphere.management.coordinator",
        )
        assert isinstance(priority_weight_nutrients, (float, torch.Tensor))
        assert 0.1 <= priority_weight_nutrients <= 0.6
        
        # Test parameter bounds validation
        interaction_factor = coordinator.get_parameter_value("interaction_factor")
        assert 0.5 <= interaction_factor <= 1.5  # Reasonable range for interactions
    
    def test_parameter_units_validation(self, coordinator):
        """Test that parameter units are properly validated"""
        # Test that parameters have proper units
        decision_threshold = coordinator.get_parameter_value(
            "decision_threshold",
            units="dimensionless",
            validate=True
        )
        assert isinstance(decision_threshold, (float, torch.Tensor))
    
    def test_parameter_bounds_checking(self, coordinator):
        """Test that parameter bounds are enforced"""
        # These should work within bounds
        valid_params = {
            "coordination_efficiency": 0.85,
            "priority_weights.nutrients": 0.4,
            "interaction_factor": 1.1
        }
        
        # An in-bounds instance value is accepted and read back from the config
        # source (there is no caller default to fall back on).
        for param_name, value in valid_params.items():
            coordinator.set_parameter_value(param_name, value)
            result = coordinator.get_parameter_value(param_name, source_order=("config",), required=True)
            assert float(result) == pytest.approx(value)
    
    # L1.3: Scientific calculation accuracy (coordination logic)
    def test_coordination_decision_calculation(self, coordinator, sample_state):
        """Test coordination decision calculation accuracy"""
        decisions = coordinator.compute_coordination_decisions(sample_state)
        
        # Check that coordination decisions are calculated
        assert "management_decisions" in decisions
        assert "priority_weights" in decisions
        assert "interaction_effects" in decisions
        
        mgmt_decisions = decisions["management_decisions"]
        priority_weights = decisions["priority_weights"]
        
        # Should be properly structured
        assert isinstance(mgmt_decisions, dict)
        assert isinstance(priority_weights, dict)
        assert len(mgmt_decisions) > 0
        assert len(priority_weights) > 0
        
        # All decision values should be tensors
        for decision_name, decision_value in mgmt_decisions.items():
            assert isinstance(decision_value, torch.Tensor)
    
    def test_priority_weight_calculation(self, coordinator, sample_state):
        """Test priority weight calculation for different management goals"""
        priorities = coordinator.calculate_priority_weights(sample_state)
        
        # Check expected priority categories
        expected_priorities = ['nutrients', 'water', 'soil_health', 'productivity', 'environmental']
        for priority in expected_priorities:
            assert priority in priorities, f"Missing priority: {priority}"
            assert isinstance(priorities[priority], torch.Tensor)
        
        # Priority weights should sum to 1 (or close to it)
        total_weight = sum(priorities.values())
        assert torch.allclose(total_weight, torch.tensor(1.0), rtol=1e-3), \
            f"Priority weights should sum to 1, got {total_weight}"
        
        # Should respond to stress conditions
        # Create stressed conditions (low nitrogen, low moisture)
        stressed_state = sample_state.copy()
        stressed_state['soil_nitrogen'] *= 0.3  # Low nitrogen
        stressed_state['soil_moisture'] *= 0.4  # Low moisture
        
        stressed_priorities = coordinator.calculate_priority_weights(stressed_state)
        
        # Nutrient and water priorities should increase under stress
        normal_nutrient_priority = priorities['nutrients'].mean()
        stressed_nutrient_priority = stressed_priorities['nutrients'].mean()
        assert stressed_nutrient_priority > normal_nutrient_priority, \
            "Nutrient priority should increase under nutrient stress"
        
        normal_water_priority = priorities['water'].mean()
        stressed_water_priority = stressed_priorities['water'].mean()
        assert stressed_water_priority > normal_water_priority, \
            "Water priority should increase under water stress"
    
    def test_interaction_effects_calculation(self, coordinator, sample_state):
        """Test calculation of interaction effects between management practices"""
        interactions = coordinator.calculate_interaction_effects(sample_state)
        
        # Check that interaction effects are calculated
        assert isinstance(interactions, dict)
        assert len(interactions) > 0
        
        # Should include common management interactions
        expected_interactions = [
            'fertilizer_irrigation',  # Fertilizer response enhanced by irrigation
            'tillage_fertilizer',    # Tillage affects fertilizer efficiency
            'irrigation_tillage',     # Irrigation affects tillage effectiveness
            'fertilizer_tillage'     # Combined effects on soil
        ]
        
        for interaction in expected_interactions:
            if interaction in interactions:
                interaction_value = interactions[interaction]
                assert isinstance(interaction_value, torch.Tensor)
                assert torch.all(interaction_value >= 0.5) and torch.all(interaction_value <= 2.0), \
                    f"Interaction factor should be in reasonable range: {interaction}"
    
    def test_management_optimization(self, coordinator, sample_state):
        """Test management practice optimization"""
        optimization = coordinator.optimize_management_practices(sample_state)
        
        # Check optimization outputs
        assert "optimized_decisions" in optimization
        assert "constraint_satisfaction" in optimization
        assert "objective_function_value" not in optimization
        
        optimized_decisions = optimization["optimized_decisions"]
        
        # Should be properly structured
        assert isinstance(optimized_decisions, dict)
        
        # All optimized values should be reasonable
        for decision_name, decision_value in optimized_decisions.items():
            assert isinstance(decision_value, torch.Tensor)
            assert torch.all(torch.isfinite(decision_value)), \
                f"Optimized decision {decision_name} should be finite"
            assert torch.all(decision_value >= 0), \
                f"Optimized decision {decision_name} should be non-negative"
    
    # L1.4: Conservation laws validation
    def test_mass_balance_coordination(self, coordinator, management_modules, sample_state):
        """Test mass balance across coordinated management practices"""
        # Get individual module outputs
        module_outputs = {}
        
        for module_name, module in management_modules.items():
            try:
                if module_name == 'fertilization':
                    output = module.forward(sample_state, time=0, dt=3600, day_of_year=150)
                else:
                    output = module.forward(sample_state, time=0, dt=3600)
                module_outputs[module_name] = output
            except Exception as e:
                print(f"Warning: {module_name} module failed: {e}")
        
        # Get coordinated outputs
        coordinated_output = coordinator.forward(sample_state, time=0, dt=3600)
        
        # Check that coordinated outputs don't violate mass balance
        if 'fertilization' in module_outputs and 'fertilizer_n_flux' in coordinated_output:
            individual_fert = module_outputs['fertilization'].get('fertilizer_n_flux', torch.zeros(1))
            coordinated_fert = coordinated_output['fertilizer_n_flux']
            
            # Coordinated flux should be reasonable compared to individual
            if torch.any(individual_fert > 0):
                ratio = coordinated_fert / torch.clamp(individual_fert, min=1e-10)
                assert torch.all(ratio >= 0.5) and torch.all(ratio <= 2.0), \
                    "Coordinated fertilizer flux should be within reasonable range of individual flux"

    def test_open_management_port_conservation_ledger(self, coordinator, sample_state):
        """Open management ports and resource quotas use separate physical ledgers."""
        output = coordinator.forward(
            sample_state,
            time=0.0,
            dt=3600.0,
        )

        ledger = coordinator.check_conservation_laws(sample_state, output)
        assert ledger == {"mass": 0.0, "resource_quotas": 0.0}

        invalid_output = dict(output)
        invalid_output["total_harvest_flux"] = -torch.ones_like(
            _value(output["total_harvest_flux"])
        )
        invalid_mass_ledger = coordinator.check_conservation_laws(
            sample_state,
            invalid_output,
        )
        assert invalid_mass_ledger["mass"] > 0.0
        assert invalid_mass_ledger["resource_quotas"] == 0.0

        quota = coordinator.scheduler.resource_quotas["nitrogen"]
        quota.current_used = quota.total_available + 1.0
        invalid_quota_ledger = coordinator.check_conservation_laws(
            sample_state,
            output,
        )
        assert invalid_quota_ledger["resource_quotas"] == pytest.approx(1.0)

    def test_runtime_has_no_false_conservation_or_optional_input_warning(
        self,
        coordinator,
        sample_state,
        caplog,
    ):
        """A dependency-output observer must not report closed-ledger violations."""
        caplog.set_level(logging.WARNING)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = coordinator(
                sample_state,
                time=0.0,
                dt=3600.0,
            )

        assert _value(output["diagnostics.coordinator_mass_balance_error"]).max() == 0.0
        warning_messages = [str(item.message) for item in caught]
        assert not any("Conservation law" in message for message in warning_messages)
        assert "Conservation violation" not in caplog.text
        assert "missing state inputs" not in caplog.text
    
    def test_resource_balance_coordination(self, coordinator, sample_state):
        """Test resource balance across coordinated practices"""
        # Test that coordination doesn't cause resource conflicts
        coordinated_output = coordinator.forward(sample_state, time=0, dt=3600)
        
        # Check water balance
        if 'irrigation_flux' in coordinated_output and 'drainage_flux' in coordinated_output:
            irrigation = coordinated_output['irrigation_flux']
            drainage = coordinated_output['drainage_flux']
            
            # Should not create impossible water balance situations
            net_water_change = irrigation - drainage
            assert torch.all(torch.abs(net_water_change) < 1e-4), \
                "Net water change should be balanced"
        
        # Check nutrient balance
        if 'fertilizer_n_flux' in coordinated_output and 'volatilization_flux' in coordinated_output:
            fertilizer_n = coordinated_output['fertilizer_n_flux']
            volatilization = coordinated_output['volatilization_flux']
            
            # Volatilization should not exceed fertilizer application
            if torch.any(fertilizer_n > 0):
                volatilization_ratio = volatilization / torch.clamp(fertilizer_n, min=1e-10)
                assert torch.all(volatilization_ratio <= 0.3), \
                    "Volatilization should not exceed reasonable fraction of fertilizer"
    
    # L1.5: Management strategy coordination
    def test_strategy_implementation(self, coordinator):
        """Test implementation of different management strategies"""
        strategies = [
            ManagementStrategy.PRODUCTIVITY_FOCUSED,
            ManagementStrategy.SUSTAINABILITY_FOCUSED,
            ManagementStrategy.BALANCED,
            ManagementStrategy.CONSERVATION_FOCUSED
        ]
        
        strategy_outputs = []
        
        for strategy in strategies:
            # Set strategy
            coordinator.set_management_strategy(strategy)
            
            # Create test state
            state = _canonical_scalar_state()
            
            output = coordinator.forward(state, time=0, dt=3600)
            strategy_outputs.append((strategy, output))
        
        # Different strategies should produce different outcomes
        productivity_output = None
        sustainability_output = None
        
        for strategy, output in strategy_outputs:
            if strategy == ManagementStrategy.PRODUCTIVITY_FOCUSED:
                productivity_output = output
            elif strategy == ManagementStrategy.SUSTAINABILITY_FOCUSED:
                sustainability_output = output
        
        if productivity_output and sustainability_output:
            # Productivity-focused should have higher input rates
            if 'fertilizer_n_flux' in productivity_output and 'fertilizer_n_flux' in sustainability_output:
                prod_fert = productivity_output['fertilizer_n_flux']
                sust_fert = sustainability_output['fertilizer_n_flux']
                
                if torch.any(prod_fert > 0) and torch.any(sust_fert > 0):
                    assert prod_fert.mean() > sust_fert.mean(), \
                        "Productivity-focused strategy should have higher fertilizer rates"

    def test_strategy_application_is_idempotent_and_parameter_backed(self, coordinator):
        """Changing strategy must rebuild weights from the ParameterManager baseline."""
        baseline = {objective.name: objective.weight for objective in coordinator.management_objectives}

        coordinator.set_management_strategy(ManagementStrategy.PRECISION)
        first = {objective.name: objective.weight for objective in coordinator.management_objectives}
        first_multiplier = coordinator.strategy_nutrient_multiplier
        coordinator.set_management_strategy(ManagementStrategy.PRECISION)
        second = {objective.name: objective.weight for objective in coordinator.management_objectives}

        assert second == pytest.approx(first)
        assert coordinator.strategy_nutrient_multiplier == pytest.approx(first_multiplier)
        assert first_multiplier == pytest.approx(
            coordinator.require_scalar(
                "strategies.precision.nutrient_multiplier",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
        )

        coordinator.set_management_strategy(ManagementStrategy.CONVENTIONAL)
        restored = {objective.name: objective.weight for objective in coordinator.management_objectives}
        assert restored == pytest.approx(baseline)
        assert coordinator.strategy_nutrient_multiplier == pytest.approx(
            coordinator.require_scalar(
                "strategies.baseline.nutrient_multiplier",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
        )

    def test_event_duration_is_read_from_parameter_manager(self, coordinator, monkeypatch):
        """Event duration is a configured scheduling policy, not a Python literal."""
        original_require = coordinator.require_scalar

        def configured_duration(name, *args, **kwargs):
            if name == "event_duration_hours.irrigation":
                return 3.0
            return original_require(name, *args, **kwargs)

        monkeypatch.setattr(coordinator, "require_scalar", configured_duration)
        assert coordinator.schedule_management_event(
            EventType.IRRIGATION,
            scheduled_time=1000.0,
            parameters={"water_amount": 25.0},
        )
        assert coordinator.scheduled_events[-1].duration == pytest.approx(
            3.0 * coordinator.seconds_per_hour
        )

    def test_season_boundaries_are_parameter_backed(self, coordinator, sample_state, monkeypatch):
        """Season classification must use the configured northern-hemisphere boundaries."""
        original_require = coordinator.require_scalar
        configured = {
            "season_boundaries.spring_start_day": 1.0,
            "season_boundaries.summer_start_day": 10.0,
            "season_boundaries.autumn_start_day": 20.0,
            "season_boundaries.winter_start_day": 30.0,
        }

        def configured_boundary(name, *args, **kwargs):
            if name in configured:
                return configured[name]
            return original_require(name, *args, **kwargs)

        monkeypatch.setattr(coordinator, "require_scalar", configured_boundary)
        state = dict(sample_state)
        state["day_of_year"] = torch.tensor(15.0)
        coordinator._update_environmental_context(
            state,
            current_time=0.0,
            calendar_type="noleap",
        )
        assert coordinator.current_season == "summer"

    def test_360_day_season_boundaries_reject_noleap_only_days(
        self,
        sample_state,
        monkeypatch,
    ):
        """A 360-day simulation must not accept a day-361 seasonal boundary."""
        coordinator = ManagementCoordinator(config={"calendar_type": "360_day"})
        original_require = coordinator.require_scalar
        configured = {
            "season_boundaries.spring_start_day": 90.0,
            "season_boundaries.summer_start_day": 180.0,
            "season_boundaries.autumn_start_day": 270.0,
            "season_boundaries.winter_start_day": 361.0,
        }

        def configured_boundary(name, *args, **kwargs):
            if name in configured:
                return configured[name]
            return original_require(name, *args, **kwargs)

        monkeypatch.setattr(coordinator, "require_scalar", configured_boundary)
        state = dict(sample_state)
        state["day_of_year"] = torch.tensor(360.0)
        with pytest.raises(ValueError, match=r"\[1, 360\]"):
            coordinator._update_environmental_context(
                state,
                current_time=0.0,
                calendar_type="360_day",
            )

    def test_validate_state_uses_the_configured_calendar_domain(self, sample_state):
        """State validation must reject day 361 for the configured 360-day calendar."""
        coordinator = ManagementCoordinator(config={"calendar_type": "360_day"})
        state = dict(sample_state)
        state["day_of_year"] = torch.tensor(361.0)

        issues = coordinator.validate_state(state)

        assert len(issues) == 1
        assert "[1, 360]" in issues[0]

    def test_initial_step_runs_one_parameterized_coordination_cycle(self, coordinator, sample_state):
        """The first accepted step must not inherit an uncomputed season or policy state."""
        assert coordinator.coordination_history == []
        coordinator.forward(sample_state, time=0.0, dt=3600.0)
        assert len(coordinator.coordination_history) == 1

        coordinator.forward(sample_state, time=0.5 * coordinator.coordination_interval, dt=3600.0)
        assert len(coordinator.coordination_history) == 1

        coordinator.forward(sample_state, time=coordinator.coordination_interval, dt=3600.0)
        assert len(coordinator.coordination_history) == 2

    def test_runtime_state_roundtrip_restores_scheduler_and_coordination_state(
        self,
        coordinator,
        sample_state,
    ):
        """Accepted-step rollback must restore every replay-critical coordinator value."""
        coordinator.forward(sample_state, time=0.0, dt=3600.0)
        assert coordinator.schedule_management_event(
            EventType.IRRIGATION,
            scheduled_time=coordinator.seconds_per_day,
            parameters={"water_amount": 4.0},
        )
        scheduled_event = coordinator.scheduled_events[-1]
        executed_event = ManagementEvent(
            event_id="completed_irrigation",
            event_type=EventType.IRRIGATION,
            scheduled_time=0.0,
            priority=EventPriority.NORMAL,
            duration=coordinator.seconds_per_hour,
            parameters={"water_amount": 2.0},
            executed=True,
            execution_result={"water_applied": 2.0},
        )
        coordinator.executed_events.append(executed_event)
        coordinator.scheduler.execution_log.append(executed_event)
        coordinator.scheduler.resource_quotas["water"].current_used = 2.0
        for event_id in ("tie_a", "tie_b", "tie_c", "tie_d", "tie_e"):
            heapq.heappush(
                coordinator.scheduler.events,
                ManagementEvent(
                    event_id=event_id,
                    event_type=EventType.CUSTOM,
                    scheduled_time=2.0 * coordinator.seconds_per_day,
                    priority=EventPriority.LOW,
                    duration=coordinator.seconds_per_hour,
                    parameters={"label": event_id},
                ),
            )
        coordinator.coordination_history.append(
            CoordinationResult(
                scheduled_events=[scheduled_event],
                conflicts_resolved=0,
                resource_allocation={"water": 2.0},
                objective_scores={"productivity": 0.75},
                coordination_status="completed",
                warnings=["test snapshot"],
            )
        )
        expected_season = coordinator.current_season
        expected_time = coordinator.last_coordination_time
        expected_residual = coordinator._last_management_mass_residual.clone()
        expected_scheduler_event_ids = [
            event.event_id for event in coordinator.scheduler.events
        ]
        snapshot = copy.deepcopy(coordinator.state_dict())

        coordinator.current_season = "winter"
        coordinator.last_coordination_time = 999999.0
        coordinator._last_management_mass_residual = torch.tensor(123.0)
        coordinator.scheduled_events.clear()
        coordinator.executed_events.clear()
        coordinator.coordination_history.clear()
        coordinator.scheduler.events.clear()
        coordinator.scheduler.execution_log.clear()
        coordinator.scheduler.resource_quotas["water"].current_used = 99.0

        coordinator.load_state_dict(snapshot)

        assert coordinator.current_season == expected_season
        assert coordinator.last_coordination_time == pytest.approx(expected_time)
        assert torch.equal(coordinator._last_management_mass_residual, expected_residual)
        assert [event.event_id for event in coordinator.scheduled_events] == [
            scheduled_event.event_id
        ]
        assert [event.event_id for event in coordinator.scheduler.events] == (
            expected_scheduler_event_ids
        )
        restored_scheduled = {
            event.event_id: event for event in coordinator.scheduler.events
        }
        assert restored_scheduled[scheduled_event.event_id] is coordinator.scheduled_events[0]
        assert [event.event_id for event in coordinator.executed_events] == [
            executed_event.event_id
        ]
        assert coordinator.scheduler.execution_log[0] is coordinator.executed_events[0]
        assert coordinator.scheduler.resource_quotas["water"].current_used == pytest.approx(2.0)
        assert len(coordinator.coordination_history) == 2
        restored_result = coordinator.coordination_history[-1]
        assert restored_result.scheduled_events[0] is coordinator.scheduled_events[0]
        assert restored_result.objective_scores == {"productivity": 0.75}

    @pytest.mark.parametrize(
        "corruption",
        [
            "season",
            "scheduled_time",
            "duration",
            "resource_requirement",
            "quota_negative",
            "quota_overflow",
            "complex_residual",
            "empty_residual",
            "negative_conflicts",
        ],
    )
    def test_runtime_state_rejects_invalid_domains_atomically(
        self,
        coordinator,
        sample_state,
        corruption,
    ):
        """Malformed replay state must fail before changing the live coordinator."""
        coordinator.forward(sample_state, time=0.0, dt=3600.0)
        assert coordinator.schedule_management_event(
            EventType.IRRIGATION,
            scheduled_time=coordinator.seconds_per_day,
            parameters={"water_amount": 4.0},
            resource_requirements={"water": 4.0},
        )
        snapshot = copy.deepcopy(coordinator.state_dict())
        corrupted = copy.deepcopy(snapshot)
        extra = corrupted["_extra_state"]
        event_id = next(iter(extra["event_records"]))
        if corruption == "season":
            extra["current_season"] = "monsoon"
        elif corruption == "scheduled_time":
            extra["event_records"][event_id]["scheduled_time"] = -1.0
        elif corruption == "duration":
            extra["event_records"][event_id]["duration"] = 0.0
        elif corruption == "resource_requirement":
            extra["event_records"][event_id]["resource_requirements"] = {
                "water": -1.0
            }
        elif corruption == "quota_negative":
            extra["resource_quota_usage"]["water"] = -1.0
        elif corruption == "quota_overflow":
            water_quota = coordinator.scheduler.resource_quotas["water"]
            extra["resource_quota_usage"]["water"] = water_quota.total_available + 1.0
        elif corruption == "complex_residual":
            extra["last_management_mass_residual"] = torch.tensor(1.0 + 1.0j)
        elif corruption == "empty_residual":
            extra["last_management_mass_residual"] = torch.empty(0)
        elif corruption == "negative_conflicts":
            extra["coordination_history"][0]["conflicts_resolved"] = -1

        before_season = coordinator.current_season
        before_time = coordinator.last_coordination_time
        before_event_ids = [event.event_id for event in coordinator.scheduler.events]
        before_water_use = coordinator.scheduler.resource_quotas["water"].current_used
        with pytest.raises((TypeError, ValueError)):
            coordinator.load_state_dict(corrupted)

        assert coordinator.current_season == before_season
        assert coordinator.last_coordination_time == before_time
        assert [event.event_id for event in coordinator.scheduler.events] == before_event_ids
        assert coordinator.scheduler.resource_quotas["water"].current_used == (
            before_water_use
        )

    def test_management_need_and_candidate_timing_are_parameter_backed(
        self,
        coordinator,
        sample_state,
        monkeypatch,
    ):
        """Need factors, thresholds, and delays must be configurable policy values."""
        original_require = coordinator.require_scalar
        configured = {
            "need_scores.harvest.autumn_factor": 0.42,
            "candidate_need_threshold": 0.5,
            "high_priority_threshold": 0.35,
            "event_delay_days.irrigation": 2.0,
        }

        def configured_policy(name, *args, **kwargs):
            if name in configured:
                return configured[name]
            return original_require(name, *args, **kwargs)

        monkeypatch.setattr(coordinator, "require_scalar", configured_policy)
        coordinator.current_season = "autumn"
        needs = coordinator._assess_management_needs(sample_state, current_time=0.0)
        biomass_threshold = coordinator.require_scalar(
            "coordinator.harvest_biomass_threshold",
            source_order=("manager",),
        )
        readiness = torch.clamp(
            sample_state["aboveground_biomass"] / biomass_threshold,
            min=0.0,
            max=1.0,
        ).mean().item()
        assert needs["harvest"] == pytest.approx(readiness * 0.42)

        no_candidates = coordinator._generate_candidate_events(
            current_time=100.0,
            planning_horizon=10.0,
            management_needs={"irrigation": 0.4},
            state=sample_state,
        )
        assert no_candidates == []

        configured["candidate_need_threshold"] = 0.3
        candidates = coordinator._generate_candidate_events(
            current_time=100.0,
            planning_horizon=10.0,
            management_needs={"irrigation": 0.4},
            state=sample_state,
        )
        assert len(candidates) == 1
        assert candidates[0].priority is EventPriority.HIGH
        assert candidates[0].scheduled_time == pytest.approx(
            100.0 + 2.0 * coordinator.seconds_per_day
        )

    def test_resource_requirement_units_match_registered_quotas(self, coordinator):
        """Every administrative resource is normalized only against the same unit."""
        manager = coordinator.get_parameter_manager()
        requirements = coordinator._estimate_resource_requirements("irrigation", 0.5)
        for resource in requirements:
            key = (
                "anthroposphere.management.coordinator.resource_requirements."
                f"irrigation.{resource}"
            )
            spec = manager.get_parameter(key)
            assert spec is not None, key
            assert spec.units == coordinator.scheduler.resource_quotas[resource].unit

    def test_objective_and_status_proxies_are_parameter_backed(self, coordinator, monkeypatch):
        """Phenomenological objective and status coefficients must come from parameters."""
        original_require = coordinator.require_scalar
        configured = {
            "objective_scoring.water_use_efficiency.event_penalty": 0.05,
            "status.neutral_score": 0.25,
            "status.success_weight": 0.2,
            "status.objective_weight": 0.8,
        }

        def configured_proxy(name, *args, **kwargs):
            if name in configured:
                return configured[name]
            return original_require(name, *args, **kwargs)

        monkeypatch.setattr(coordinator, "require_scalar", configured_proxy)
        event = ManagementEvent(
            event_id="irrigation_objective",
            event_type=EventType.IRRIGATION,
            scheduled_time=0.0,
            priority=EventPriority.NORMAL,
            duration=coordinator.seconds_per_hour,
            parameters={"need_score": 0.5},
        )
        objective = next(
            item for item in coordinator.management_objectives
            if item.name == "water_use_efficiency"
        )
        assert coordinator._heuristic_objective_score(objective, [event], {}) == pytest.approx(0.95)
        assert coordinator._get_coordination_status() == pytest.approx(0.25)

        coordinator.coordination_history.append(
            CoordinationResult(
                scheduled_events=[event],
                conflicts_resolved=0,
                resource_allocation={},
                objective_scores={"test": 0.5},
                coordination_status="completed",
            )
        )
        assert coordinator._get_coordination_status() == pytest.approx(0.6)
    
    def test_conflict_resolution(self, coordinator):
        """Test resolution of conflicts between management practices"""
        # Create conflicting scenario (high fertilizer but low irrigation)
        conflicting_state = _canonical_scalar_state(
            soil_moisture=torch.tensor([0.15], device=DEVICE),
            soil_nitrogen=torch.tensor([0.004], device=DEVICE),
            aboveground_biomass=torch.tensor([0.8], device=DEVICE),
            crop_maturity=torch.tensor([0.6], device=DEVICE),
            air_temperature=torch.tensor([303.15], device=DEVICE),
        )
        conflicting_state.update({
            'fertilizer_application_rate': torch.tensor([0.02], device=DEVICE),
            'et_demand': torch.tensor([5e-6], device=DEVICE),
        })
        
        # Get coordinated decision
        coordinated_output = coordinator.forward(conflicting_state, time=0, dt=3600)
        
        # Should resolve conflict by adjusting rates
        if 'fertilizer_n_flux' in coordinated_output and 'irrigation_flux' in coordinated_output:
            fert_rate = coordinated_output['fertilizer_n_flux']
            irr_rate = coordinated_output['irrigation_flux']
            
            # Should not recommend high fertilizer with low irrigation
            # Either reduce fertilizer or increase irrigation
            assert torch.all(fert_rate >= 0) and torch.all(irr_rate >= 0), \
                "Both rates should be non-negative"
            
            # Should show some adjustment from conflict resolution
            conflict_resolution = coordinated_output.get('conflict_resolution_applied', torch.tensor([0.0]))
            assert conflict_resolution > 0, "Conflict resolution should be applied"
    
    def test_temporal_coordination(self, coordinator):
        """Test temporal coordination of management practices"""
        # Test coordination across different growth stages
        growth_stages = [0.3, 0.5, 0.7, 0.9]  # Early to late maturity
        temporal_outputs = []
        
        for stage in growth_stages:
            state = _canonical_scalar_state(
                crop_maturity=torch.tensor([stage], device=DEVICE)
            )
            
            output = coordinator.forward(state, time=0, dt=3600)
            temporal_outputs.append((stage, output))
        
        # Should show temporal variation in management decisions
        early_fert = None
        late_fert = None
        
        for stage, output in temporal_outputs:
            if 'fertilizer_n_flux' in output:
                fert_rate = output['fertilizer_n_flux'].item()
                if stage <= 0.4:  # Early stage
                    early_fert = fert_rate
                elif stage >= 0.8:  # Late stage
                    late_fert = fert_rate
        
        if early_fert is not None and late_fert is not None:
            # Should show different rates at different stages
            assert abs(early_fert - late_fert) > 1e-6, \
                "Fertilizer rates should vary with growth stage"
    
    # L1.6: Abstract base class compliance
    def test_scale_coupling_interface(self, coordinator, sample_state):
        """Test compliance with ScaleCoupling interface"""
        # Should implement upscale and downscale methods
        assert hasattr(coordinator, 'upscale')
        assert hasattr(coordinator, 'downscale')
        assert callable(coordinator.upscale)
        assert callable(coordinator.downscale)

        # Test upscale method
        fine_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        upscaled = coordinator.upscale(fine_data)
        assert isinstance(upscaled, torch.Tensor)

        # Test downscale method
        coarse_data = torch.tensor([1.0, 2.0])
        downscaled = coordinator.downscale(coarse_data)
        assert isinstance(downscaled, torch.Tensor)
    
    def test_forward_method_consistency(self, coordinator):
        """Test backward consistency with forward method"""
        state = _canonical_scalar_state()
        
        # Should be able to call forward method
        outputs = coordinator.forward(state, time=0, dt=3600)
        
        # Should return dictionary with expected outputs
        assert isinstance(outputs, dict)
        assert len(outputs) > 0
    
    def test_dependency_declaration(self, coordinator):
        """Test dependency declaration functionality"""
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
    
    # Additional validation tests
    def test_coordinator_parameter_values(self, coordinator):
        """Test that coordinator parameters have reasonable values"""
        # Test key coordinator parameters
        coordination_efficiency = coordinator.get_parameter_value("coordination_efficiency")
        assert 0.7 <= coordination_efficiency <= 0.95  # Reasonable range for efficiency
        
        priority_weight_nutrients = coordinator.get_parameter_value(
            "priority_weights.nutrients",
            namespace="anthroposphere.management.coordinator",
        )
        assert 0.1 <= priority_weight_nutrients <= 0.6  # Reasonable range for priority weight

        manager = coordinator.get_parameter_manager()
        stale_keys = (
            "anthroposphere.management.coordinator.priority_weight_nutrients",
            "anthroposphere.management.coordinator.priority_weights.resource_availability",
            "anthroposphere.management.coordinator.priority_weights.timing_constraints",
            "anthroposphere.management.coordinator.priority_weights.conflict_severity",
            "anthroposphere.management.coordinator.spatial_resolution.lat",
        )
        for key in stale_keys:
            assert manager.get_parameter(key) is None
        
        interaction_factor = coordinator.get_parameter_value("interaction_factor")
        assert 0.5 <= interaction_factor <= 1.5  # Reasonable range for interactions

    def test_policy_limits_are_parameter_backed_and_complete(self, coordinator):
        """Calibratable policy limits must have complete ParameterSpec metadata."""
        manager = coordinator.get_parameter_manager()
        keys = (
            "anthroposphere.management.coordinator.management_strategy",
            "anthroposphere.management.coordinator.coordination_interval_days",
            "anthroposphere.management.coordinator.score_limits.benefit_max",
            "anthroposphere.management.coordinator.interaction_multiplier_min",
            "anthroposphere.management.coordinator.interaction_multiplier_max",
            "anthroposphere.management.coordinator.maturity_factor_min",
            "anthroposphere.management.coordinator.maturity_factor_max",
            "anthroposphere.management.coordinator.strategies.baseline.nutrient_multiplier",
            "anthroposphere.management.coordinator.strategies.organic.soil_carbon_weight",
            "anthroposphere.management.coordinator.strategies.organic.biodiversity_weight",
            "anthroposphere.management.coordinator.strategies.organic.nutrient_multiplier",
            "anthroposphere.management.coordinator.strategies.precision.efficiency_weight_multiplier",
            "anthroposphere.management.coordinator.strategies.precision.nutrient_multiplier",
            "anthroposphere.management.coordinator.strategies.productivity.objective_weight_multiplier",
            "anthroposphere.management.coordinator.strategies.productivity.nutrient_multiplier",
            "anthroposphere.management.coordinator.strategies.conservation.objective_weight_multiplier",
            "anthroposphere.management.coordinator.strategies.conservation.nutrient_multiplier",
            "anthroposphere.management.coordinator.event_duration_hours.irrigation",
            "anthroposphere.management.coordinator.event_duration_hours.tillage",
            "anthroposphere.management.coordinator.event_duration_hours.harvest",
            "anthroposphere.management.coordinator.event_duration_hours.grazing",
            "anthroposphere.management.coordinator.event_duration_hours.fire_management",
            "anthroposphere.management.coordinator.season_boundaries.spring_start_day",
            "anthroposphere.management.coordinator.season_boundaries.summer_start_day",
            "anthroposphere.management.coordinator.season_boundaries.autumn_start_day",
            "anthroposphere.management.coordinator.season_boundaries.winter_start_day",
            "anthroposphere.management.coordinator.need_scores.harvest.autumn_factor",
            "anthroposphere.management.coordinator.need_scores.harvest.other_factor",
            "anthroposphere.management.coordinator.need_scores.tillage.conventional_peak",
            "anthroposphere.management.coordinator.need_scores.tillage.alternative_peak",
            "anthroposphere.management.coordinator.need_scores.tillage.off_season",
            "anthroposphere.management.coordinator.need_scores.grazing.biomass_reference",
            "anthroposphere.management.coordinator.need_scores.grazing.off_season",
            "anthroposphere.management.coordinator.need_scores.fire.peak_factor",
            "anthroposphere.management.coordinator.need_scores.fire.off_season_factor",
            "anthroposphere.management.coordinator.need_scores.fire.base_need",
            "anthroposphere.management.coordinator.candidate_need_threshold",
            "anthroposphere.management.coordinator.high_priority_threshold",
            "anthroposphere.management.coordinator.event_delay_days.irrigation",
            "anthroposphere.management.coordinator.event_delay_days.harvest",
            "anthroposphere.management.coordinator.event_delay_days.tillage",
            "anthroposphere.management.coordinator.event_delay_days.grazing",
            "anthroposphere.management.coordinator.event_delay_days.fire_management",
            "anthroposphere.management.coordinator.resource_quotas.water",
            "anthroposphere.management.coordinator.resource_requirements.irrigation.water",
            "anthroposphere.management.coordinator.resource_requirements.irrigation.labor",
            "anthroposphere.management.coordinator.resource_requirements.irrigation.fuel",
            "anthroposphere.management.coordinator.resource_requirements.tillage.fuel",
            "anthroposphere.management.coordinator.resource_requirements.tillage.labor",
            "anthroposphere.management.coordinator.resource_requirements.tillage.machinery",
            "anthroposphere.management.coordinator.resource_requirements.harvest.fuel",
            "anthroposphere.management.coordinator.resource_requirements.harvest.labor",
            "anthroposphere.management.coordinator.resource_requirements.harvest.machinery",
            "anthroposphere.management.coordinator.resource_requirements.grazing.labor",
            "anthroposphere.management.coordinator.resource_requirements.fire_management.labor",
            "anthroposphere.management.coordinator.resource_requirements.fire_management.water",
            "anthroposphere.management.coordinator.objective_scoring.water_use_efficiency.event_penalty",
            "anthroposphere.management.coordinator.objective_scoring.nutrient_use_efficiency.event_ratio_penalty",
            "anthroposphere.management.coordinator.objective_scoring.soil_carbon_sequestration.tillage_penalty",
            "anthroposphere.management.coordinator.objective_scoring.economic_return.harvest_reward",
            "anthroposphere.management.coordinator.objective_scoring.biodiversity_index.event_penalty",
            "anthroposphere.management.coordinator.objective_scoring.neutral_score",
            "anthroposphere.management.coordinator.status.neutral_score",
            "anthroposphere.management.coordinator.status.success_weight",
            "anthroposphere.management.coordinator.status.objective_weight",
            "anthroposphere.management.coordinator.event_defaults.tillage_intensity_class",
            "anthroposphere.management.coordinator.statistics_history_limit",
        )
        for key in keys:
            spec = manager.get_parameter(key)
            assert spec is not None, key
            assert spec.units
            if key.endswith(("management_strategy", "tillage_intensity_class")):
                assert spec.allowed_values
            else:
                assert spec.bounds is not None
            assert spec.category
            assert spec.description
            assert spec.references
            assert spec.tags
            assert spec.provenance

    def test_positive_reference_parameters_fail_fast(self, coordinator, sample_state):
        """Physical reference denominators must not use a cross-unit epsilon floor."""
        original_get = coordinator.get_parameter_value

        def zero_nitrogen_reference(name, *args, **kwargs):
            if name == "reference.soil_nitrogen_optimal":
                return torch.tensor(0.0)
            return original_get(name, *args, **kwargs)

        coordinator.get_parameter_value = zero_nitrogen_reference
        with pytest.raises(ValueError, match="soil_nitrogen_optimal"):
            coordinator.calculate_priority_weights(sample_state)

    def test_hodge_weights_do_not_use_identity_fallback(self, coordinator):
        """A missing declared priority weight must block structural assembly."""
        coordinator.priority_weights.pop("environmental")
        with pytest.raises(ValueError, match="priority weights"):
            coordinator.build_hodge_contract(strict=True)

    def test_resource_efficiency_normalizes_each_resource_by_its_quota(self, coordinator):
        """Different resource dimensions may only be combined after quota normalization."""
        water_event = ManagementEvent(
            event_id="water_fraction",
            event_type=EventType.IRRIGATION,
            scheduled_time=0.0,
            priority=EventPriority.NORMAL,
            duration=coordinator.seconds_per_hour,
            parameters={"need_score": 0.5},
            resource_requirements={"water": 80.0},
        )
        labor_event = ManagementEvent(
            event_id="labor_fraction",
            event_type=EventType.HARVEST,
            scheduled_time=0.0,
            priority=EventPriority.NORMAL,
            duration=coordinator.seconds_per_hour,
            parameters={"need_score": 0.5},
            resource_requirements={"labor": 10.0},
        )
        assert coordinator.scheduler.resource_quotas["water"].total_available == 800.0
        assert coordinator.scheduler.resource_quotas["labor"].total_available == 100.0
        assert coordinator._calculate_resource_efficiency(water_event) == pytest.approx(
            coordinator._calculate_resource_efficiency(labor_event)
        )

        missing_need = ManagementEvent(
            event_id="missing_need",
            event_type=EventType.CUSTOM,
            scheduled_time=0.0,
            priority=EventPriority.NORMAL,
            duration=coordinator.seconds_per_hour,
            parameters={},
            resource_requirements={"water": 1.0},
        )
        with pytest.raises(KeyError, match="need_score"):
            coordinator._calculate_resource_efficiency(missing_need)

        unknown_resource = ManagementEvent(
            event_id="unknown_resource",
            event_type=EventType.CUSTOM,
            scheduled_time=0.0,
            priority=EventPriority.NORMAL,
            duration=coordinator.seconds_per_hour,
            parameters={"need_score": 0.5},
            resource_requirements={"unknown": 1.0},
        )
        with pytest.raises(KeyError, match="unknown"):
            coordinator._calculate_resource_efficiency(unknown_resource)
    
    def test_edge_cases_and_error_handling(self, coordinator):
        """Test edge cases and error handling"""
        # Missing dynamic decision states must fail before any calculation.
        minimal_state = {
            'soil_moisture': torch.tensor([0.3], device=DEVICE),
            'aboveground_biomass': torch.tensor([1000.0], device=DEVICE)
        }
        with pytest.raises(KeyError):
            coordinator.forward(minimal_state, time=0, dt=3600)
        
        # Test with extreme values
        extreme_state = {
            'soil_moisture': torch.tensor([0.01], device=DEVICE),  # Very dry
            'soil_nitrogen': torch.tensor([200.0], device=DEVICE),  # Very high N
            'soil_organic_carbon': torch.tensor([30.0], device=DEVICE),
            'crop_maturity': torch.tensor([0.9], device=DEVICE),
            'aboveground_biomass': torch.tensor([10.0], device=DEVICE),
            'air_temperature': torch.tensor([318.15], device=DEVICE),
            'irrigation_flux': torch.tensor([0.0], device=DEVICE),
            'harvest_flux': torch.tensor([0.0], device=DEVICE),
            'grazing_flux': torch.tensor([0.0], device=DEVICE),
            'co2_emission': torch.tensor([0.0], device=DEVICE),
            'field_capacity': torch.tensor([0.35], device=DEVICE),
            'wilting_point': torch.tensor([0.15], device=DEVICE),
            'evapotranspiration_flux': torch.tensor([3.0e-6], device=DEVICE),
            'precipitation_flux': torch.tensor([0.0], device=DEVICE),
        }

        # Should handle extreme values without crashing
        extreme_outputs = coordinator.forward(extreme_state, time=0, dt=3600)
        assert isinstance(extreme_outputs, dict)
        
        # All outputs should be finite
        for output_name, output_value in extreme_outputs.items():
            assert torch.all(torch.isfinite(_value(output_value))), \
                f"Output {output_name} should be finite with extreme inputs"


# ---------------------------------------------------------------------------
# Soil water limits, the irrigation requirement, and event requests (MA-21 round 12)
# ---------------------------------------------------------------------------


def _coordinator():
    return ManagementCoordinator(config={"calendar_type": "noleap"})


def test_soil_water_limits_and_water_fluxes_are_required_state_inputs():
    """The needs assessment reads the site's field capacity and wilting point, and the
    irrigation requirement reads the evapotranspiration and precipitation fluxes; no
    nominal ParameterManager value stands in for them."""
    import inspect

    coordinator = _coordinator()
    required = set(coordinator.metadata.input_variables) - set(
        coordinator.optional_input_variables
    )
    assert {
        "field_capacity",
        "wilting_point",
        "evapotranspiration_flux",
        "precipitation_flux",
    } <= required
    for key in (
        "anthroposphere.management.irrigation.et_demand_default",
        "anthroposphere.management.irrigation.field_capacity_default",
        "anthroposphere.management.irrigation.wilting_point_default",
        "anthroposphere.management.harvest.default_expected_yield",
    ):
        assert key not in coordinator.metadata.parameters
    source = inspect.getsource(coordinator_module)
    for stale in (
        "et_demand_default",
        "field_capacity_default",
        "wilting_point_default",
        "default_expected_yield",
    ):
        assert stale not in source


def test_irrigation_need_uses_the_state_soil_water_limits():
    """D = clip((theta_fc - theta) / (theta_fc - theta_wp), 0, 1) with the state limits
    (Allen et al. 1998, FAO-56 root-zone depletion between field capacity and wilting
    point); a missing limit or theta_fc <= theta_wp raises."""
    coordinator = _coordinator()
    state = _canonical_scalar_state(
        soil_moisture=torch.tensor([0.30], device=DEVICE),
        field_capacity=torch.tensor([0.40], device=DEVICE),
        wilting_point=torch.tensor([0.10], device=DEVICE),
    )
    needs = coordinator._assess_management_needs(state, current_time=0.0)
    assert needs["irrigation"] == pytest.approx((0.40 - 0.30) / (0.40 - 0.10))

    missing = dict(state)
    missing.pop("field_capacity")
    with pytest.raises(KeyError, match="field_capacity"):
        coordinator._assess_management_needs(missing, current_time=0.0)

    inverted = dict(state, wilting_point=torch.tensor([0.45], device=DEVICE))
    with pytest.raises(ValueError, match="field_capacity"):
        coordinator._assess_management_needs(inverted, current_time=0.0)


def test_irrigation_demand_is_the_net_irrigation_requirement():
    """F_net = max(E - P, 0): the FAO-24 net irrigation requirement rate (Doorenbos and
    Pruitt 1977) is the evapotranspiration not met by precipitation, the depletion rate
    of the FAO-56 root-zone water balance (Allen et al. 1998, Eq. 85). The gross
    application F_net / eta belongs to IrrigationManagement, the owner of the
    irrigation-system application efficiencies; no coordination score scales it."""
    coordinator = _coordinator()
    cells = 3
    state = {
        name: value.expand(cells).clone()
        for name, value in _canonical_scalar_state().items()
    }
    state["evapotranspiration_flux"] = torch.tensor([4.0e-6, 1.0e-6, -2.0e-7], device=DEVICE)
    state["precipitation_flux"] = torch.tensor([1.0e-6, 3.0e-6, 0.0], device=DEVICE)
    state["field_capacity"] = torch.full((cells,), 0.35, device=DEVICE)
    state["wilting_point"] = torch.full((cells,), 0.15, device=DEVICE)
    demand = coordinator.optimize_management_practices(state)["optimized_decisions"][
        "irrigation_demand_flux"
    ]
    expected = torch.tensor([3.0e-6, 0.0, 0.0], dtype=demand.dtype, device=DEVICE)
    torch.testing.assert_close(demand, expected, rtol=1e-6, atol=1e-15)

    outputs = coordinator.forward(
        {name: value[:1].clone() for name, value in state.items()}, time=0.0, dt=3600.0
    )
    torch.testing.assert_close(
        _value(outputs["irrigation_demand_flux"]),
        torch.tensor([3.0e-6], dtype=_value(outputs["irrigation_demand_flux"]).dtype),
        rtol=1e-6,
        atol=1e-15,
    )

    missing = dict(state)
    missing.pop("evapotranspiration_flux")
    with pytest.raises(KeyError, match="evapotranspiration_flux"):
        coordinator.optimize_management_practices(missing)


def test_harvest_request_reports_expected_yield_only_when_the_event_supplies_it():
    """The coordinator echoes an event-supplied expected yield; it does not invent one."""
    coordinator = _coordinator()
    request = coordinator._handle_harvest_event(
        _canonical_scalar_state(), {"harvest_index": 0.45}
    )
    assert request["status"] == "completed"
    assert "expected_yield" not in request
    supplied = coordinator._handle_harvest_event(
        _canonical_scalar_state(), {"harvest_index": 0.45, "expected_yield": 6.5}
    )
    assert supplied["expected_yield"] == 6.5


@pytest.mark.parametrize(
    "handler_name, parameters, message",
    [
        ("_handle_irrigation_event", {"water_amount": -1.0}, "Irrigation request depth"),
        ("_handle_tillage_event", {"intensity": "deep_ripping"}, "Unsupported tillage intensity"),
        ("_handle_harvest_event", {"harvest_index": 1.5}, "Harvest index"),
        ("_handle_grazing_event", {"stocking_rate": -1.0}, "requested_stocking_rate"),
        ("_handle_fire_event", {"fuel_consumption": 1.5}, "fuel_consumption"),
    ],
)
def test_invalid_event_requests_raise_instead_of_reporting_a_failed_status(
    handler_name, parameters, message
):
    """An invalid control request is an invalid state. The handler raises, so
    EventScheduler.execute_until releases the resources, requeues the event, and
    re-raises into the simulator transaction; a {"status": "failed"} result would mark
    the event executed and let the step continue."""
    coordinator = _coordinator()
    handler = getattr(coordinator, handler_name)
    with pytest.raises(ValueError, match=message):
        handler(_canonical_scalar_state(), parameters)


def test_invalid_scheduled_event_fails_the_step_and_stays_queued():
    """A scheduled invalid request fails the coordinator step instead of being counted
    as an executed event."""
    coordinator = _coordinator()
    assert coordinator.schedule_management_event(
        EventType.IRRIGATION, 0.0, {"water_amount": -1.0}
    )
    with pytest.raises(ValueError, match="Irrigation request depth"):
        coordinator.forward(_canonical_scalar_state(), time=0.0, dt=3600.0)
    assert coordinator.executed_events == []
    assert any(
        event.parameters == {"water_amount": -1.0} for event in coordinator.scheduler.events
    )


def test_environmental_priority_reads_the_state_air_temperature():
    """s_env = clip(|T_air - T_ref| / T_ref, 0, 1) reads the declared air_temperature state
    (item 113: the coordinator takes no environment mapping). Between a state at T_ref and
    one at T_ref + 30 K only s_env changes, so the ratio of the environmental to the
    nutrient priority grows by exactly 1 + 30 / T_ref (the normalisation cancels)."""
    from relic.parameters import get_global_manager

    coordinator = _coordinator()
    t_ref = float(
        get_global_manager().get(
            "anthroposphere.management.coordinator.reference.environment_temperature_optimal"
        )
    )
    reference = coordinator.calculate_priority_weights(
        _canonical_scalar_state(air_temperature=torch.tensor([t_ref], device=DEVICE))
    )
    heated = coordinator.calculate_priority_weights(
        _canonical_scalar_state(air_temperature=torch.tensor([t_ref + 30.0], device=DEVICE))
    )
    ratio = (heated["environmental"] / heated["nutrients"]) / (
        reference["environmental"] / reference["nutrients"]
    )
    assert float(ratio) == pytest.approx(1.0 + 30.0 / t_ref, rel=1e-5)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


