#!/usr/bin/env python3
"""
L1 Unit Tests for Fire Management Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy (emissions, burned area)
- L1.4: Conservation laws validation
- L1.5: Fire intensity and effects
- L1.6: Abstract base class compliance

Scientific References:
- Bond, W.J. & Keeley, J.E. (2005). Fire as a global 'herbivore'. Trends in Ecology & Evolution.
- Seiler, W. & Crutzen, P.J. (1980). Estimates of gross and net fluxes of carbon between the biosphere and the atmosphere.

Test Framework: pytest
"""

import copy
import json
import os
import sys
import warnings
from pathlib import Path

import pytest
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from relic.core.abstractions import BiologicalProcess
from relic.core.functor import BiologicalProcessFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.management import fire_management as fire_module
from relic.scientific.anthroposphere.management.fire_management import (
    FireEvent,
    FireIntensity,
    FireManagement,
    FireType,
    VegetationType,
)
from test.utils.forward_call import call_forward

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3
REPO_ROOT = Path(__file__).resolve().parents[3]


class TestFireManagementL1:
    """L1 Unit Tests for FireManagement"""
    
    @pytest.fixture
    def fire_manager(self):
        """Create fire manager instance for testing"""
        return FireManagement(config={"calendar_type": "noleap", "seed": 0})
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'aboveground_biomass': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE).reshape(batch_shape), # kg m-2
            'belowground_biomass': torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], device=DEVICE).reshape(batch_shape), # kg m-2
            'litter_mass': torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], device=DEVICE).reshape(batch_shape), # kg m-2
            'fuel_moisture': torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25], device=DEVICE).reshape(batch_shape), # fraction
            'days_since_fire': torch.full(batch_shape, 30.0, device=DEVICE), # day (owned state)
            'temperature': torch.tensor([298.15, 303.15, 308.15, 313.15, 318.15], device=DEVICE).reshape(batch_shape) # K
        }
    
    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, fire_manager):
        """Test that FireManagement can be instantiated"""
        assert fire_manager is not None
        assert isinstance(fire_manager, FireManagement)
        assert isinstance(fire_manager, BiologicalProcessFunctor)
        assert isinstance(fire_manager, BiologicalProcess)
        assert hasattr(fire_manager, 'name')
        assert fire_manager.name == "fire_management"
    
    def test_metadata_completeness(self, fire_manager):
        """Test that metadata is properly set"""
        metadata = fire_manager.get_metadata()
        assert metadata is not None
        assert metadata.name == "fire_management"
        assert len(metadata.references) > 0
        # Check mathematical expression
        assert "C_combusted =" in metadata.mathematical_expression
        assert "mass" in metadata.conservation_laws
    
    def test_fire_schedule_initialization(self, fire_manager):
        """Test that default fire schedule is properly initialized"""
        assert len(fire_manager.fire_events) > 0
        assert any(event.fire_type == FireType.PRESCRIBED for event in fire_manager.fire_events)
        assert any(event.fire_type == FireType.WILDFIRE for event in fire_manager.fire_events)

    def test_state_dict_restores_schedule_and_substream(self):
        """The module state is the fire schedule and the substream position; the
        Functor holds no tensor buffer (item 118: days_since_fire is StateManager
        state, the cumulative accounting buffers were removed)."""
        manager = FireManagement(config={"seed": 90210})
        expected_event = FireEvent(
            fire_type=FireType.WILDFIRE,
            intensity=FireIntensity.EXTREME,
            vegetation_type=VegetationType.FOREST_BOREAL,
            date=217,
            burned_fraction=0.35,
            fuel_moisture=0.08,
            wind_speed=14.0,
            temperature=311.0,
            suppression_effectiveness=0.15,
        )
        manager.fire_events = [expected_event]

        saved_state = copy.deepcopy(manager.state_dict())
        assert set(saved_state) == {"_extra_state"}
        expected_random_draws = torch.rand(4, generator=manager.generator)

        manager.fire_events = []
        manager.generator.manual_seed(1)

        manager.load_state_dict(saved_state)
        restored_random_draws = torch.rand(4, generator=manager.generator)

        assert manager.fire_events == [expected_event]
        assert torch.equal(restored_random_draws, expected_random_draws)

    def test_day_length_uses_physical_constant_ssot(self, monkeypatch):
        """Fire time conversion must use the physical-constant API."""
        constant_calls = []

        def tracked_get_physical_constant(name):
            constant_calls.append(name)
            return 12345.0

        monkeypatch.setattr(
            fire_module,
            "get_physical_constant",
            tracked_get_physical_constant,
            raising=False,
        )
        manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})

        assert manager._seconds_per_day_value() == pytest.approx(12345.0)
        assert constant_calls == ["relic.physical.constants.time.seconds_per_day"]
        assert (
            "relic.physical.constants.time.seconds_per_day"
            not in manager.required_parameters()
        )
    
    # L1.3: Scientific calculation accuracy
    def test_emissions_calculation(self, fire_manager, sample_state):
        """Test emissions calculation accuracy"""
        # Use a day when fire is scheduled (e.g. prescribed burn day 100 or wildfire day 240)
        day_of_year = 100 # Prescribed burn
        dt = 86400.0
        
        outputs = call_forward(
            fire_manager,
            sample_state,
            time=0,
            dt=dt,
            day_of_year=day_of_year,
        )
        
        assert "co2_emission" in outputs
        assert "co_emission" in outputs
        assert "fire_ch4_emission" in outputs
        assert "ch4_emission" not in outputs
        
        # Emissions should be non-negative
        assert torch.all(outputs["co2_emission"] >= 0)

        # CO2 should be dominant C emission
        assert outputs["co2_emission"].mean() > outputs["co_emission"].mean()
        assert outputs["co2_emission"].mean() > outputs["fire_ch4_emission"].mean()

    def test_360_day_calendar_repeats_day_one_fire_at_year_boundary(
        self,
        sample_state,
    ):
        manager = FireManagement(config={"calendar_type": "360_day", "seed": 0})
        manager.fire_events = [
            FireEvent(
                fire_type=FireType.PRESCRIBED,
                intensity=FireIntensity.MODERATE,
                vegetation_type=VegetationType.GRASSLAND,
                date=1,
                burned_fraction=0.25,
                fuel_moisture=0.1,
                wind_speed=3.0,
                temperature=298.15,
                suppression_effectiveness=0.0,
            )
        ]

        outputs = call_forward(
            manager,
            sample_state,
            time=360.0 * 86400.0,
            dt=3600.0,
            calendar_type="360_day",
        )

        assert outputs["diagnostics.fire.events_today"].item() == pytest.approx(1.0)

    def test_forward_requires_explicit_calendar_contract(self, sample_state):
        manager = FireManagement(config={"seed": 0})

        with pytest.raises(ValueError, match="requires explicit calendar_type"):
            manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                day_of_year=100,
            )

    @pytest.mark.parametrize("calendar_type", ["", "gregorian"])
    def test_invalid_configured_calendar_fails_at_construction(
        self,
        calendar_type,
    ):
        with pytest.raises(ValueError, match="calendar_type"):
            FireManagement(config={"calendar_type": calendar_type, "seed": 0})

    @pytest.mark.parametrize(
        "event_date",
        [
            pytest.param(True, id="bool"),
            pytest.param(1.5, id="fractional"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(torch.tensor([1.0, 2.0]), id="non_scalar"),
        ],
    )
    def test_schedule_rejects_non_integral_event_dates(
        self,
        fire_manager,
        sample_state,
        event_date,
    ):
        fire_manager.fire_events[0].date = event_date

        with pytest.raises(ValueError, match="Fire event date"):
            fire_manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                day_of_year=100,
            )

    def test_state_dict_rejects_event_date_coercion(self, fire_manager):
        fire_manager.fire_events[0].date = 100.5

        with pytest.raises(ValueError, match="Fire event date"):
            fire_manager.state_dict()

    def test_extra_state_restore_validates_configured_calendar(self):
        source = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        source.fire_events[0].date = 365
        extra_state = copy.deepcopy(source.get_extra_state())
        target = FireManagement(config={"calendar_type": "360_day", "seed": 0})

        with pytest.raises(ValueError, match="Fire event date"):
            target.set_extra_state(extra_state)

    def test_forward_rejects_conflicting_day_sources(
        self,
        fire_manager,
        sample_state,
    ):
        state = dict(sample_state)
        state["day_of_year"] = torch.tensor(100.0)

        with pytest.raises(ValueError, match="Conflicting day_of_year"):
            fire_manager.forward(
                state,
                time=0.0,
                dt=3600.0,
                day_of_year=101,
            )

    def test_default_schedule_rejects_fractional_parameter_date(
        self,
        monkeypatch,
    ):
        original_safe_fire_param = FireManagement._safe_fire_param

        def fractional_schedule_date(manager, key, namespace):
            if key.endswith(".date"):
                return 100.5
            return original_safe_fire_param(manager, key, namespace)

        monkeypatch.setattr(
            FireManagement,
            "_safe_fire_param",
            fractional_schedule_date,
        )

        with pytest.raises(ValueError, match="Fire event date"):
            FireManagement(config={"calendar_type": "noleap", "seed": 0})

    def test_add_fire_event_rejects_fractional_date(self, fire_manager):
        event = copy.deepcopy(fire_manager.fire_events[0])
        event.date = 100.5

        with pytest.raises(ValueError, match="Fire event date"):
            fire_manager.add_fire_event(event)

    def test_forward_rejects_conflicting_runtime_calendar(
        self,
        fire_manager,
        sample_state,
    ):
        with pytest.raises(ValueError, match="Runtime calendar_type conflicts"):
            fire_manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                calendar_type="360_day",
                day_of_year=100,
            )

    def test_burned_fraction_calculation(self, fire_manager, sample_state):
        """Test burned fraction calculation"""
        day_of_year = 100 # Prescribed burn
        dt = 86400.0
        
        outputs = call_forward(
            fire_manager,
            sample_state,
            time=0,
            dt=dt,
            day_of_year=day_of_year,
        )
        
        assert "burned_fraction" in outputs
        burned = outputs["burned_fraction"]
        
        assert torch.all(burned >= 0)
        assert torch.all(burned <= 1.0)
        
        # Ensure non-zero burn for prescribed fire event
        assert torch.any(burned > 0)

    def test_fuel_moisture_effects(self, fire_manager, sample_state):
        """Test fuel moisture effects on emissions"""
        day_of_year = 240 # Wildfire (sensitive to state moisture)
        dt = 86400.0
        
        # Create states with low and high moisture
        state_dry = {k: v.clone() for k, v in sample_state.items()}
        state_wet = {k: v.clone() for k, v in sample_state.items()}
        
        state_dry['fuel_moisture'][:] = 0.05
        state_wet['fuel_moisture'][:] = 0.30
        
        # Also need to set temperature constant to isolate moisture effect
        state_dry['temperature'][:] = 298.15
        state_wet['temperature'][:] = 298.15
        
        dry_manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        wet_manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        out_dry = call_forward(
            dry_manager,
            state_dry,
            time=0,
            dt=dt,
            day_of_year=day_of_year,
            force_fire=True,
        )
        out_wet = call_forward(
            wet_manager,
            state_wet,
            time=0,
            dt=dt,
            day_of_year=day_of_year,
            force_fire=True,
        )
        
        # Drier fuel -> more emissions (higher combustion completeness)
        assert out_dry["co2_emission"].mean() > out_wet["co2_emission"].mean()

    # L1.4: Conservation laws validation
    def test_carbon_mass_conservation(self, fire_manager, sample_state):
        """Test carbon mass conservation"""
        day_of_year = 100
        dt = 86400.0
        
        outputs = call_forward(
            fire_manager,
            sample_state,
            time=0,
            dt=dt,
            day_of_year=day_of_year,
        )
        
        # Calculate total C emitted
        # CO2 is kg CO2. C is 12/44.
        c_emitted = outputs["co2_emission"] * (12.0/44.0) + \
                    outputs["co_emission"] * (12.0/28.0) + \
                    outputs["fire_ch4_emission"] * (12.0/16.0)
        
        # Ash is biomass. C content ~45%? Or ash is mineral? 
        # Code says: ash_deposition += fuel_load * CC * fraction * 0.05.
        # Code says: carbon_combusted = fuel_load * CC * fraction * 0.45.
        # Emissions are calculated from carbon_combusted.
        
        assert torch.all(c_emitted > 0)
        torch.testing.assert_close(
            outputs["fire_fuel_carbon_input"],
            outputs["carbon_loss"],
        )
        assert torch.all(
            outputs["diagnostics.fire.carbon_availability_residual"]
            <= TOLERANCE
        )

    # L1.6: Abstract base class compliance
    def test_forward_method_compliance(self, fire_manager, sample_state):
        """Test compliance with FunctorBase forward interface"""
        outputs = fire_manager.forward(
            sample_state,
            time=0,
            dt=3600,
            day_of_year=100,
        )
        assert isinstance(outputs, dict)
        expected_contracts = {
            "burned_fraction": (UpdateMode.REPLACE, "1"),
            "co2_emission": (UpdateMode.REPLACE, "kg m-2 s-1"),
            "co_emission": (UpdateMode.REPLACE, "kg m-2 s-1"),
            "fire_ch4_emission": (UpdateMode.REPLACE, "kg m-2 s-1"),
            "n2o_emission": (UpdateMode.REPLACE, "kg m-2 s-1"),
            "ash_deposition": (UpdateMode.REPLACE, "kg m-2 s-1"),
            "days_since_fire": (UpdateMode.REPLACE, "day"),
            "fire_fuel_carbon_input": (
                UpdateMode.REPLACE,
                "kg C m-2 s-1",
            ),
            "carbon_loss": (UpdateMode.REPLACE, "kg C m-2 s-1"),
            "combustion_flux": (UpdateMode.REPLACE, "kg m-2 s-1"),
            "diagnostics.fire.events_today": (UpdateMode.REPLACE, "1"),
            "diagnostics.fire.event_support_seconds": (UpdateMode.REPLACE, "s"),
            "diagnostics.fire.carbon_availability_residual": (
                UpdateMode.REPLACE,
                "kg C m-2 s-1",
            ),
            "diagnostics.fire_management.canonical_iphs_runtime": (
                UpdateMode.REPLACE,
                "1",
            ),
        }

        assert set(outputs) == set(expected_contracts)
        assert "co2_emissions" not in outputs
        assert set(outputs).issubset(set(fire_manager.metadata.output_variables))
        for name, (mode, unit) in expected_contracts.items():
            item = outputs[name]
            assert isinstance(item, UpdateItem), name
            assert item.mode is mode, name
            assert item.unit == unit, name
            assert item.source_functor == fire_manager.name, name

    def test_scheduled_event_executes_once_per_calendar_occurrence(
        self,
        sample_state,
    ):
        manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        first = call_forward(
            manager,
            sample_state,
            time=0.0,
            dt=3600.0,
            day_of_year=100,
        )
        second = call_forward(
            manager,
            sample_state,
            time=3600.0,
            dt=3600.0,
            day_of_year=100,
        )

        assert first["diagnostics.fire.events_today"].item() == pytest.approx(1.0)
        assert second["diagnostics.fire.events_today"].item() == pytest.approx(0.0)
        assert torch.count_nonzero(second["co2_emission"]) == 0

    def test_executed_schedule_occurrences_roundtrip_through_state_dict(
        self,
        sample_state,
    ):
        source = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        call_forward(
            source,
            sample_state,
            time=0.0,
            dt=3600.0,
            day_of_year=100,
        )
        saved_state = copy.deepcopy(source.state_dict())

        restored = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        restored.load_state_dict(saved_state)
        replay = call_forward(
            restored,
            sample_state,
            time=3600.0,
            dt=3600.0,
            day_of_year=100,
        )

        assert replay["diagnostics.fire.events_today"].item() == pytest.approx(0.0)
        assert torch.count_nonzero(replay["co2_emission"]) == 0

    def test_real_simulator_commits_fire_ports_without_duplicate_event(self):
        simulator = RELICSimulator(
            {
                "max_time": 7200.0,
                "time_step": 3600.0,
                "calendar_type": "noleap",
                "conservation_checks": True,
                "device": "cpu",
            }
        )
        simulator.initialize_state(
            {
                "aboveground_biomass": torch.tensor([2.0]),
                "belowground_biomass": torch.tensor([0.5]),
                "litter_mass": torch.tensor([0.4]),
                "fuel_moisture": torch.tensor([0.1]),
                "days_since_fire": torch.tensor([30.0]),
                "temperature": torch.tensor([298.15]),
                "day_of_year": torch.tensor(100.0),
            }
        )
        simulator.register_functor(
            FireManagement(config={"calendar_type": "noleap", "seed": 0}),
            execution_group="fast",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            first = simulator.step(dt=3600.0)
            second = simulator.step(dt=3600.0)

        assert torch.any(first["co2_emission"] > 0.0)
        assert torch.count_nonzero(second["co2_emission"]) == 0
        assert torch.all(second["burned_fraction"] == 0.0)

    def test_event_mass_is_step_invariant_for_hourly_daily_and_two_day_steps(self, sample_state):
        """One executed occurrence releases the same event mass for any dt (no under- or double-integration)."""
        species = (
            "co2_emission",
            "co_emission",
            "fire_ch4_emission",
            "n2o_emission",
            "ash_deposition",
            "carbon_loss",
            "combustion_flux",
        )

        def _run(dt, n_steps):
            manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
            integrated = {name: None for name in species}
            executing_steps = 0
            supports = []
            for index in range(n_steps):
                outputs = call_forward(
                    manager,
                    sample_state,
                    time=index * dt,
                    dt=dt,
                    day_of_year=100,
                )
                supports.append(float(outputs["diagnostics.fire.event_support_seconds"]))
                if outputs["diagnostics.fire.events_today"].item() > 0:
                    executing_steps += 1
                for name in species:
                    # References: Seiler and Crutzen (1980).
                    # Mathematical expression: M_s = sum_k F_s,k * dt_k over the steps of the event day.
                    # Parameter meanings: F_s,k is the species rate returned in step k.
                    # Units: F_s [kg m-2 s-1]; dt [s]; M_s [kg m-2].
                    # Domain/constraints: exactly one step executes the calendar occurrence.
                    # Sign convention: emissions and fuel losses are positive.
                    amount = outputs[name] * dt
                    integrated[name] = amount if integrated[name] is None else integrated[name] + amount
            return manager, integrated, executing_steps, supports

        daily_manager, daily, daily_events, daily_support = _run(86400.0, 1)
        hourly_manager, hourly, hourly_events, hourly_support = _run(3600.0, 24)
        two_day_manager, two_day, two_day_events, two_day_support = _run(172800.0, 1)

        assert (daily_events, hourly_events, two_day_events) == (1, 1, 1)
        assert daily_support == [86400.0]
        assert hourly_support[0] == 3600.0
        assert all(support == 0.0 for support in hourly_support[1:])
        assert two_day_support == [172800.0]
        assert torch.all(daily["co2_emission"] > 0.0)
        assert torch.all(daily["carbon_loss"] > 0.0)
        for name in species:
            torch.testing.assert_close(hourly[name], daily[name], rtol=1.0e-5, atol=0.0)
            torch.testing.assert_close(two_day[name], daily[name], rtol=1.0e-5, atol=0.0)

    def test_event_support_diagnostic_matches_the_executing_step(self, fire_manager, sample_state):
        """The released rate times the reported support reproduces the combusted carbon mass."""
        outputs = call_forward(
            fire_manager,
            sample_state,
            time=0.0,
            dt=3600.0,
            day_of_year=100,
        )
        support = outputs["diagnostics.fire.event_support_seconds"]
        assert float(support) == 3600.0
        released_carbon = outputs["carbon_loss"] * support
        assert torch.all(released_carbon > 0.0)
        quiet = call_forward(
            fire_manager,
            sample_state,
            time=3600.0,
            dt=3600.0,
            day_of_year=100,
        )
        assert float(quiet["diagnostics.fire.event_support_seconds"]) == 0.0

    def test_fire_methane_port_has_unique_scm_contract(self, fire_manager):
        """Keep fire methane separate from the soil methane budget writer."""
        contract_path = (
            REPO_ROOT
            / "relic"
            / "contracts"
            / "variables"
            / "anthroposphere"
            / "fire_ch4_emission.yaml"
        )
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        policies = json.loads(
            (
                REPO_ROOT
                / "relic"
                / "core"
                / "simulator"
                / "output_overlap_whitelist.json"
            ).read_text(encoding="utf-8")
        )

        assert contract["variable"] == "fire_ch4_emission"
        assert contract["dimension"]["udunits"] == "kg m-2 s-1"
        assert contract["owner"] == fire_manager.name
        assert contract["writers"]["increments"] == []
        assert contract["writers"]["tendencies"] == [
            {
                "name": fire_manager.name,
                "module": (
                    "relic.scientific.anthroposphere.management.fire_management"
                ),
                "provides": ["fire_ch4_emission"],
                "unit": "kg m-2 s-1",
            }
        ]
        assert contract["provenance"]["runtime_writer"] == fire_manager.name
        assert contract["provenance"]["runtime_role"] == (
            "managed_fire_atmosphere_ch4_boundary_flux"
        )
        assert "ch4_emission" not in policies

    @pytest.mark.parametrize(
        ("variable", "kind", "unit", "allowed_in", "allowed_out"),
        [
            ("burned_fraction", "state", "1", [], []),
            ("co2_emission", "flux", "kg m-2 s-1", [], ["fire_management"]),
            ("co_emission", "flux", "kg m-2 s-1", [], ["fire_management"]),
            (
                "fire_ch4_emission",
                "flux",
                "kg m-2 s-1",
                [],
                ["fire_management"],
            ),
            ("n2o_emission", "flux", "kg m-2 s-1", [], ["fire_management"]),
            ("ash_deposition", "flux", "kg m-2 s-1", ["fire_management"], []),
            ("days_since_fire", "state", "day", [], []),
            (
                "fire_fuel_carbon_input",
                "flux",
                "kg C m-2 s-1",
                ["fire_management"],
                [],
            ),
            (
                "carbon_loss",
                "flux",
                "kg C m-2 s-1",
                [],
                ["fire_management"],
            ),
            (
                "combustion_flux",
                "flux",
                "kg m-2 s-1",
                [],
                ["fire_management"],
            ),
        ],
    )
    def test_fire_public_ports_have_one_scm_writer(
        self,
        variable,
        kind,
        unit,
        allowed_in,
        allowed_out,
    ):
        contract_path = (
            REPO_ROOT
            / "relic"
            / "contracts"
            / "variables"
            / "anthroposphere"
            / f"{variable}.yaml"
        )
        text = contract_path.read_text(encoding="utf-8")
        contract = yaml.safe_load(text)

        assert "REVIEW_NEEDED" not in text
        assert contract["variable"] == variable
        assert contract["kind"] == kind
        assert contract["dimension"]["udunits"] == unit
        assert contract["owner"] == "fire_management"
        assert contract["incrementable"] is False
        assert contract["writers"]["increments"] == []
        assert contract["writers"]["tendencies"] == [
            {
                "name": "fire_management",
                "module": (
                    "relic.scientific.anthroposphere.management.fire_management"
                ),
                "provides": [variable],
                "unit": unit,
            }
        ]
        assert contract["flux_contract"]["allowed_in"] == allowed_in
        assert contract["flux_contract"]["allowed_out"] == allowed_out
        assert contract["provenance"]["runtime_writer"] == "fire_management"

    def test_dependency_declaration(self, fire_manager):
        """Test dependency declaration functionality"""
        dependencies = fire_manager.get_dependencies()
        assert isinstance(dependencies, list)
        assert dependencies == []

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

