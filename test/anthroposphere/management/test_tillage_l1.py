#!/usr/bin/env python3
"""
L1 Unit Tests for Tillage Management Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy (soil disturbance)
- L1.4: Conservation laws validation
- L1.5: Tillage intensity effects on soil properties
- L1.6: Abstract base class compliance

Scientific References:
- Lal, R. (2004). Soil carbon sequestration to mitigate climate change. Geoderma 123, 1-22.
- Paustian, K., et al. (2000). Management options for enhancing carbon storage. CRC Press.
- Six, J., et al. (2002). Soil structure and organic matter. Soil Science Society of America Journal.

Test Framework: pytest
"""

import copy
import warnings

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
from relic.scientific.anthroposphere.management.tillage import (
    TillageDepth,
    TillageIntensity,
    TillageManagement,
    TillageEvent
)
import relic.scientific.anthroposphere.management.tillage as tillage_module
from relic.core.abstractions import TransportMechanism
from relic.core.functor import TransportMechanismFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant
from test.utils.forward_call import call_forward

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3


# Units of this test's own scenario variables, which no SCM contract or SSOT
# entry declares; they follow the units stated with the test data. The simulator
# infers no units from names (relic/core/simulator/state_init.py).
_TEST_STATE_UNITS = {
    'crop_residue': '1',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class TestTillageManagementL1:
    """L1 Unit Tests for TillageManagement"""
    
    @pytest.fixture
    def tillage_manager(self):
        """Create tillage manager instance for testing"""
        return TillageManagement(config={"calendar_type": "noleap"})
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'soil_moisture': torch.tensor([0.25, 0.30, 0.35, 0.40, 0.45], device=DEVICE).reshape(batch_shape),
            'soil_temperature': torch.tensor([288.15, 293.15, 298.15, 303.15, 308.15], device=DEVICE).reshape(batch_shape),
            'soil_bulk_density': torch.tensor([1.2, 1.3, 1.4, 1.5, 1.6], device=DEVICE).reshape(batch_shape),
            'soil_organic_carbon': torch.tensor([0.01, 0.015, 0.02, 0.025, 0.03], device=DEVICE).reshape(batch_shape),
            'soil_aggregates': torch.tensor([0.7, 0.65, 0.6, 0.55, 0.5], device=DEVICE).reshape(batch_shape),
            'residue_cover': torch.tensor([0.8, 0.6, 0.4, 0.2, 0.1], device=DEVICE).reshape(batch_shape),
            'days_since_tillage': torch.full(batch_shape, 365.0, device=DEVICE),
        }
    
    @pytest.fixture
    def sample_environment(self):
        """Sample environment dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'air_temperature': torch.tensor([288.15, 293.15, 298.15, 303.15, 308.15], device=DEVICE).reshape(batch_shape),
            'soil_temperature': torch.tensor([288.15, 293.15, 298.15, 303.15, 308.15], device=DEVICE).reshape(batch_shape),
            'precipitation': torch.tensor([0.0, 0.001, 0.002, 0.0005, 0.0], device=DEVICE).reshape(batch_shape)
        }

    def test_soil_moisture_and_soil_carbon_are_declared_inputs_without_defaults(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        """soil_moisture [m3 m-3] and the soil organic carbon stock [kg C m-2] have no
        ParameterManager fallback; the carbon stock is the contract variable
        soil_organic_carbon that the management coordinator also reads."""
        from relic.contracts import load_contract_registry

        inputs = tillage_manager.metadata.input_variables
        assert "soil_organic_carbon" in inputs and "soil_carbon" not in inputs
        assert load_contract_registry().get_contract("soil_organic_carbon").udunits == "kg C m-2"
        for key in ("soil_moisture", "soil_organic_carbon"):
            state = dict(sample_state)
            state.pop(key)
            with pytest.raises(KeyError, match=key):
                tillage_manager.forward(
                    state, time=0.0, dt=3600.0
                )

    def test_inputs_are_checked_against_their_physical_domain(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        """Wet organic soil (0.7 m3 m-3), a frozen tilled layer (245 K), a peat
        carbon stock (60 kg C m-2) and a field untilled for 27 years (10000 days)
        are read as given; a value outside the physical domain raises."""
        state = dict(sample_state)
        state.update({
            "soil_moisture": torch.full_like(sample_state["soil_moisture"], 0.7),
            "soil_temperature": torch.full_like(sample_state["soil_temperature"], 245.0),
            "soil_organic_carbon": torch.full_like(sample_state["soil_moisture"], 60.0),
            "days_since_tillage": torch.full_like(sample_state["soil_moisture"], 10000.0),
        })
        outputs = tillage_manager.forward(
            state, time=0.0, dt=3600.0
        )
        assert "carbon_loss_rate" in outputs
        for key, value in (
            ("soil_moisture", 1.2),
            ("soil_temperature", -1.0),
            ("soil_organic_carbon", -1.0),
            ("days_since_tillage", -1.0),
        ):
            invalid = dict(state)
            invalid[key] = torch.full_like(sample_state["soil_moisture"], value)
            with pytest.raises(ValueError, match=key):
                tillage_manager.forward(
                    invalid, time=0.0, dt=3600.0
                )

    def test_carbon_flush_q10_is_the_shared_decomposition_parameter(
        self, restore_global_parameter_manager
    ):
        """f_T = Q10^((T - T_standard) / 10 K) with Q10 read from the shared soil
        organic matter decomposition coefficient pedosphere.physical.q10_decomposition
        (Lloyd and Taylor 1994; Kirschbaum 1995; Davidson and Janssens 2006):
        the pulse ratio over 10 K equals the parameter value, here set to 2.5."""
        from relic.utils.physical_constants import get_physical_constant

        manager = restore_global_parameter_manager
        manager.set_value("pedosphere.physical.q10_decomposition", 2.5)
        tillage = TillageManagement(config={"calendar_type": "noleap"})
        reference = float(get_physical_constant("standard_temperature"))
        carbon = torch.tensor([5.0], dtype=torch.float64)
        cold = tillage._calculate_carbon_flush(
            1.0, 1.0, carbon, torch.tensor([reference], dtype=torch.float64)
        )
        warm = tillage._calculate_carbon_flush(
            1.0, 1.0, carbon, torch.tensor([reference + 10.0], dtype=torch.float64)
        )
        torch.testing.assert_close(warm / cold, torch.tensor([2.5], dtype=torch.float64))

    def test_days_since_tillage_has_no_parameter_default(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        """days_since_tillage [day] is a declared input and the tillage state itself
        (tillage_management is its only writer): the governed call requires it in the
        state, and a direct forward call without it now raises as well."""
        assert "days_since_tillage" in tillage_manager.get_required_inputs()
        assert (
            "anthroposphere.management.tillage.default_days_since_tillage"
            not in tillage_manager.metadata.parameters
        )
        state = dict(sample_state)
        state.pop("days_since_tillage", None)
        with pytest.raises(KeyError, match="days_since_tillage"):
            tillage_manager.forward(
                state, time=0.0, dt=3600.0
            )

    @staticmethod
    def _rate_core_state(soil_moisture):
        return {
            "soil_moisture": torch.tensor([soil_moisture], dtype=torch.float64),
            "residue_cover": torch.tensor([0.5], dtype=torch.float64),
        }

    @staticmethod
    def _operation(event_intensity=None, carbon_flux=0.0):
        return {
            "event_intensity": event_intensity,
            "carbon_flux": torch.tensor([carbon_flux], dtype=torch.float64),
        }

    def test_soil_co2_flux_uses_the_ssot_molar_mass_ratio(self, tillage_manager):
        """F_CO2 = F_C * M_CO2 / M_C with the physical-constant SSOT molar masses
        (44.009 and 12.011 g mol-1), not the rounded ratio 44/12."""
        rates = tillage_manager._compute_rate_core(
            self._rate_core_state(0.3), **self._operation(carbon_flux=1.0e-7))
        assert torch.count_nonzero(rates["carbon_loss_rate"]) == 1
        m_co2 = get_physical_constant("relic.physical.constants.molecular_weights.co2")
        m_c = get_physical_constant("relic.physical.constants.molecular_weights.carbon")
        torch.testing.assert_close(
            rates["soil_co2_flux"],
            rates["carbon_loss_rate"] * (m_co2 / m_c),
            rtol=1.0e-12,
            atol=0.0,
        )

    def test_soil_disturbance_moisture_response_is_not_clipped(self, tillage_manager):
        """D = k_D * I * theta / theta_den: the moisture response is not clipped to
        [0.5, 1.5], a bound without a literature basis (theta itself is checked
        against its domain [0, 1])."""
        k_d = float(tillage_manager.get_parameter_value(
            "disturbance_factor", namespace="anthroposphere.management.tillage", required=True))
        theta_den = float(tillage_manager.get_parameter_value(
            "moisture_effect_denominator", namespace="anthroposphere.management.tillage",
            required=True))
        for theta in (0.1, 0.7):
            assert not 0.5 <= theta / theta_den <= 1.5
            rates = tillage_manager._compute_rate_core(
                self._rate_core_state(theta), **self._operation(event_intensity=0.6))
            torch.testing.assert_close(
                rates["soil_disturbance_rate"],
                torch.tensor([k_d * 0.6 * theta / theta_den], dtype=torch.float64),
                rtol=1.0e-6,
                atol=0.0,
            )

    def test_rate_core_reads_no_tillage_type(self, tillage_manager):
        """The removed tillage_type branch computed nothing: the rates do not depend
        on a tillage_type entry, and the rate core no longer reads one."""
        import inspect

        operation = self._operation(event_intensity=0.6, carbon_flux=1.0e-7)
        baseline = tillage_manager._compute_rate_core(self._rate_core_state(0.3), **operation)
        for tillage_type in ("no_till", "not_a_tillage_type", torch.tensor([2.0])):
            state = self._rate_core_state(0.3)
            state["tillage_type"] = tillage_type
            rates = tillage_manager._compute_rate_core(state, **operation)
            for key, value in baseline.items():
                torch.testing.assert_close(rates[key], value, rtol=0.0, atol=0.0)
        assert "tillage_type" not in inspect.getsource(TillageManagement._compute_rate_core)

    @staticmethod
    def _event_state(*, day, days_since, residue_cover=0.5, control=None):
        """Direct-call state; day 90 is the default primary pass (HIGH, DEEP,
        moisture limit 0.22) and triggers when more than 14 days have elapsed."""
        state = {
            "soil_moisture": torch.tensor([0.20], dtype=torch.float64),
            "soil_temperature": torch.tensor([293.15], dtype=torch.float64),
            "soil_organic_carbon": torch.tensor([5.0], dtype=torch.float64),
            "days_since_tillage": torch.tensor([days_since], dtype=torch.float64),
            "day_of_year": torch.tensor(float(day)),
        }
        if residue_cover is not None:
            state["residue_cover"] = torch.tensor([residue_cover], dtype=torch.float64)
        if control is not None:
            state["tillage_intensity"] = torch.tensor([control], dtype=torch.float64)
        return state

    @staticmethod
    def _output(outputs, key):
        item = outputs[key]
        return (item.value if hasattr(item, "value") else item).to(torch.float64)

    @staticmethod
    def _tillage_parameter(manager, name, units=None):
        return float(manager.get_parameter_value(
            name, namespace="anthroposphere.management.tillage", units=units, required=True))

    def test_never_tilled_soil_has_no_tillage_fluxes(self):
        """No operation has executed: the explicit no-tillage case gives zero
        disturbance, residue incorporation, carbon loss and CO2 flux, whatever the
        coordinator control tillage_intensity says."""
        manager = TillageManagement(config={"calendar_type": "noleap"})
        outputs = manager.forward(
            self._event_state(day=200, days_since=365.0, control=0.9),
            time=199 * 86400.0, dt=86400.0)
        for key in ("soil_disturbance_rate", "residue_incorporation_rate",
                    "carbon_loss_rate", "soil_co2_flux", "delta_residue_cover"):
            assert torch.count_nonzero(self._output(outputs, key)) == 0, key

    def test_response_intensity_is_the_operation_intensity(self):
        """At the default HIGH primary pass the disturbance uses the HIGH intensity
        class factor, D = k_D I_high theta / theta_den, independent of the
        coordinator control tillage_intensity."""
        results = []
        for control in (0.2, 0.9):
            manager = TillageManagement(config={"calendar_type": "noleap"})
            outputs = manager.forward(
                self._event_state(day=90, days_since=100.0, control=control),
                time=89 * 86400.0, dt=86400.0)
            results.append(self._output(outputs, "soil_disturbance_rate"))
        k_d = self._tillage_parameter(manager, "disturbance_factor")
        theta_den = self._tillage_parameter(manager, "moisture_effect_denominator")
        i_high = self._tillage_parameter(manager, "high_intensity_factor")
        expected = torch.tensor([k_d * i_high * 0.20 / theta_den], dtype=torch.float64)
        for result in results:
            torch.testing.assert_close(result, expected, rtol=1.0e-6, atol=0.0)

    def test_residue_cover_is_an_optional_input_read_at_an_operation(self):
        """residue_cover [1] is declared and optional: a step without an operation
        does not read it; an operation without it raises instead of assuming bare soil."""
        manager = TillageManagement(config={"calendar_type": "noleap"})
        assert "residue_cover" in manager.metadata.input_variables
        assert "residue_cover" in manager.optional_input_variables
        manager.forward(
            self._event_state(day=200, days_since=365.0, residue_cover=None),
            time=199 * 86400.0, dt=86400.0)
        with pytest.raises(ValueError, match="residue_cover"):
            manager.forward(
                self._event_state(day=90, days_since=100.0, residue_cover=None),
                time=89 * 86400.0, dt=86400.0)

    def test_rate_core_has_no_default_intensity_or_unpublished_response(self):
        """default_intensity (the substitute of a missing control) and the
        unpublished mixing-efficiency response with its unsourced 0.5 offset are gone."""
        import inspect

        manager = TillageManagement(config={"calendar_type": "noleap"})
        assert "anthroposphere.management.tillage.default_intensity" not in manager.metadata.parameters
        source = inspect.getsource(TillageManagement._compute_rate_core)
        assert "default_intensity" not in source
        assert "mixing_eff" not in source
        assert '"tillage_intensity"' not in source

    def test_missing_soil_temperature_raises_instead_of_a_state_default(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        """soil_temperature has no ParameterManager fallback: a missing value raises."""
        state = dict(sample_state)
        state.pop("soil_temperature")
        with pytest.raises(KeyError, match="soil_temperature"):
            tillage_manager.forward(
                state, time=0.0, dt=3600.0
            )

    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, tillage_manager):
        """Test that TillageManagement can be instantiated"""
        assert tillage_manager is not None
        assert isinstance(tillage_manager, TillageManagement)
        assert isinstance(tillage_manager, TransportMechanismFunctor)
        assert isinstance(tillage_manager, TransportMechanism)
        assert hasattr(tillage_manager, 'name')
        assert tillage_manager.name == "tillage_management"
    
    def test_metadata_completeness(self, tillage_manager):
        """Test that metadata is properly set"""
        metadata = tillage_manager.get_metadata()
        assert metadata is not None
        assert metadata.name == "tillage_management"
        assert len(metadata.references) > 0
        assert "carbon" in metadata.mathematical_expression or "soil" in metadata.mathematical_expression
        assert "mass" in metadata.conservation_laws

    def test_undeclared_crop_residue_does_not_change_the_outputs(
        self, tillage_manager, sample_state
    ):
        """R_inc = clamp(k_residue I (a_theta + f_moisture), 0, 1) needs no residue
        mass (theory page, section 2): crop_residue is neither declared nor read."""
        assert "crop_residue" not in tillage_manager.metadata.input_variables

        state = dict(sample_state, day_of_year=torch.tensor(1.0))
        without = TillageManagement(config={"calendar_type": "noleap"}).forward(
            dict(state), 0.0, 86400.0
        )
        with_residue = TillageManagement(config={"calendar_type": "noleap"}).forward(
            dict(state, crop_residue=torch.full_like(state["soil_moisture"], 5.0)),
            0.0,
            86400.0,
        )
        assert set(without) == set(with_residue)
        for key, item in without.items():
            left = getattr(item, "value", item)
            right = getattr(with_residue[key], "value", with_residue[key])
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right)

    def test_forward_outputs_are_declared_in_metadata(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        """Require the runtime tillage ports to match the Functor contract."""
        outputs = tillage_manager.forward(
            sample_state,
            time=0.0,
            dt=3600.0,
        )

        declared = set(tillage_manager.metadata.output_variables)
        undeclared = sorted(set(outputs) - declared)
        assert not undeclared, f"Undeclared tillage outputs: {undeclared}"
        assert "soil_disturbance_flux" not in outputs
        assert "carbon_loss_flux" not in outputs

    def test_forward_declares_update_mode_unit_and_writer(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        outputs = tillage_manager.forward(
            sample_state,
            time=0.0,
            dt=3600.0,
        )
        expected_contracts = {
            "porosity_modifier": (UpdateMode.REPLACE, "1"),
            "roughness_modifier": (UpdateMode.REPLACE, "1"),
            "days_since_tillage": (UpdateMode.REPLACE, "day"),
            "tillage_intensity_factor": (UpdateMode.REPLACE, "1"),
            "time_decay_factor": (UpdateMode.REPLACE, "1"),
            "cumulative_operations": (UpdateMode.REPLACE, "1"),
            "soil_disturbance_rate": (UpdateMode.REPLACE, "1"),
            "residue_incorporation_rate": (UpdateMode.REPLACE, "1"),
            "carbon_loss_rate": (UpdateMode.REPLACE, "kg C m-2 s-1"),
            "soil_co2_flux": (UpdateMode.REPLACE, "kg CO2 m-2 s-1"),
            "delta_residue_cover": (UpdateMode.ACCUMULATE, "s-1"),
            "diagnostics.tillage_mass_balance_error": (
                UpdateMode.REPLACE,
                "kg C m-2",
            ),
        }

        assert set(outputs) == set(expected_contracts)
        for name, (mode, unit) in expected_contracts.items():
            item = outputs[name]
            assert isinstance(item, UpdateItem), name
            assert item.mode is mode, name
            assert item.unit == unit, name
            assert item.source_functor == tillage_manager.name, name
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration"""
        custom_config = {
            "parameters": {
                "disturbance_factor": {"value": 0.8, "units": "1"},
            }
        }

        manager = TillageManagement(config=custom_config)
        assert manager is not None

        # The config value is reachable through FunctorBase.get_parameter_value when
        # the config source is requested first; the default order reads the YAML
        # database first, as for every Functor (no private config-first accessor).
        disturbance = manager.get_parameter_value(
            "disturbance_factor",
            namespace="anthroposphere.management.tillage",
            source_order=("config", "manager"),
        )
        assert abs(disturbance - 0.8) < 1e-3
    
    def test_tillage_schedule_initialization(self, tillage_manager):
        """Test that default tillage schedule is properly initialized"""
        assert len(tillage_manager.tillage_events) > 0
        assert any(event.intensity == TillageIntensity.HIGH for event in tillage_manager.tillage_events)
        assert any(event.intensity == TillageIntensity.LOW for event in tillage_manager.tillage_events)

    def test_runtime_modifiers_roundtrip_through_state_dict(self, tillage_manager):
        tillage_manager.current_intensity = TillageIntensity.HIGH
        tillage_manager.current_depth = TillageDepth.DEEP
        tillage_manager.cumulative_tillage_operations = 4
        tillage_manager.porosity_modifier = torch.tensor(1.2)
        tillage_manager.roughness_modifier = torch.tensor(1.3)
        saved_state = copy.deepcopy(tillage_manager.state_dict())

        tillage_manager.current_intensity = TillageIntensity.NONE
        tillage_manager.current_depth = TillageDepth.SHALLOW
        tillage_manager.cumulative_tillage_operations = 0
        tillage_manager.porosity_modifier = torch.tensor(9.0)
        tillage_manager.roughness_modifier = torch.tensor(9.0)
        tillage_manager.load_state_dict(saved_state)

        assert tillage_manager.current_intensity is TillageIntensity.HIGH
        assert tillage_manager.current_depth is TillageDepth.DEEP
        assert tillage_manager.cumulative_tillage_operations == 4
        torch.testing.assert_close(tillage_manager.porosity_modifier, torch.tensor(1.2))
        torch.testing.assert_close(tillage_manager.roughness_modifier, torch.tensor(1.3))

    def test_360_day_schedule_window_wraps_at_year_boundary(self):
        manager = TillageManagement(config={"calendar_type": "360_day"})
        event = manager.tillage_events[0]
        event.date = 359
        manager.tillage_events = [event]

        active = manager._check_tillage_schedule(
            1,
            torch.tensor([0.1]),
            torch.tensor([30.0]),
            calendar_type="360_day",
        )

        assert active is event

    def test_forward_requires_explicit_calendar_contract(self, sample_state):
        manager = TillageManagement()

        with pytest.raises(ValueError, match="requires explicit calendar_type"):
            manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                day_of_year=90,
            )

    @pytest.mark.parametrize(
        "invalid_date",
        (
            True,
            90.5,
            float("nan"),
            torch.tensor([90.0, 91.0]),
        ),
    )
    def test_schedule_rejects_non_integral_event_dates(
        self,
        tillage_manager,
        invalid_date,
    ):
        event = tillage_manager.tillage_events[0]
        event.date = invalid_date
        tillage_manager.tillage_events = [event]

        with pytest.raises(ValueError, match="Tillage event date"):
            tillage_manager._check_tillage_schedule(
                90,
                torch.tensor([0.1]),
                torch.tensor([30.0]),
                calendar_type="noleap",
            )

    @pytest.mark.parametrize(
        "invalid_day",
        (True, 90.5, 0, 366, float("nan")),
    )
    def test_schedule_rejects_invalid_current_day(
        self,
        tillage_manager,
        invalid_day,
    ):
        with pytest.raises(ValueError, match="day_of_year"):
            tillage_manager._check_tillage_schedule(
                invalid_day,
                torch.tensor([0.1]),
                torch.tensor([30.0]),
                calendar_type="noleap",
            )

    def test_invalid_configured_calendar_fails_during_initialization(self):
        with pytest.raises(ValueError, match="calendar_type"):
            TillageManagement(config={"calendar_type": ""})

    def test_forward_rejects_conflicting_day_sources(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        state = dict(sample_state)
        state["day_of_year"] = torch.tensor(90.0)

        with pytest.raises(ValueError, match="Conflicting day_of_year inputs"):
            tillage_manager.forward(
                state,
                time=0.0,
                dt=3600.0,
                day_of_year=91,
            )

    def test_add_event_validates_date_against_configured_calendar(self):
        manager = TillageManagement(config={"calendar_type": "360_day"})
        event = copy.deepcopy(manager.tillage_events[0])
        event.date = 361

        with pytest.raises(ValueError, match="Tillage event date"):
            manager.add_tillage_event(event)
    
    # L1.2: Parameter management system validation
    def test_parameter_access_through_manager(self, tillage_manager):
        """Test that parameters are accessible through ParameterManager"""
        # Test getting parameters
        disturbance_factor = tillage_manager.get_parameter_value(
            "disturbance_factor", namespace="anthroposphere.management.tillage", required=True
        )
        assert isinstance(disturbance_factor, (float, torch.Tensor))
        assert 0.1 <= disturbance_factor <= 1.0

        # The carbon-flush base fraction lies within its YAML bounds [0, 0.01].
        base_fraction = tillage_manager.get_parameter_value(
            "co2_pulse_base_fraction",
            namespace="anthroposphere.management.tillage",
            required=True,
        )
        assert 0.0 <= base_fraction <= 0.01
    
    def test_parameter_units_validation(self, tillage_manager):
        """Test that parameter units are properly validated"""
        # Test that parameters have proper units
        residue_incorporation = tillage_manager.get_parameter_value(
            "residue_incorporation_rate",
            namespace="anthroposphere.management.tillage",
            units="dimensionless",
            validate=True
        )
        assert isinstance(residue_incorporation, (float, torch.Tensor))
    
    def test_parameter_bounds_checking(self, tillage_manager):
        """Test that parameter bounds are enforced"""
        # Every value read lies within the bounds of its YAML entry.
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        for param_name in ("disturbance_factor", "co2_pulse_factor", "co2_pulse_base_fraction"):
            key = f"anthroposphere.management.tillage.{param_name}"
            lower, upper = manager.get_parameter(key).bounds
            result = float(tillage_manager.get_parameter_value(
                param_name, namespace="anthroposphere.management.tillage", required=True
            ))
            assert lower <= result <= upper, key

    def test_carbon_flush_uses_physical_constant_ssot(
        self,
        tillage_manager,
        monkeypatch,
    ):
        """Require the temperature reference to use the physical-constant owner."""
        constant_calls = []
        original_parameter_lookup = tillage_manager.get_parameter_value

        def tracked_physical_constant(name):
            constant_calls.append(name)
            assert name == "standard_temperature"
            return 300.0

        def reject_physical_parameter_lookup(name, **kwargs):
            namespace = str(kwargs.get("namespace", ""))
            if namespace.startswith("relic.physical.constants"):
                raise AssertionError(
                    "Physical constants must not use Functor parameter lookup."
                )
            return original_parameter_lookup(name, **kwargs)

        monkeypatch.setattr(
            tillage_module,
            "get_physical_constant",
            tracked_physical_constant,
        )
        monkeypatch.setattr(
            tillage_manager,
            "get_parameter_value",
            reject_physical_parameter_lookup,
        )

        soil_carbon = torch.tensor([2.0], dtype=torch.float64)
        temperature = torch.tensor([310.0], dtype=torch.float64)
        pulse = tillage_manager._calculate_carbon_flush(
            intensity_factor=1.0,
            depth_factor=1.0,
            soil_carbon=soil_carbon,
            temperature=temperature,
        )

        assert torch.isfinite(pulse).all()
        assert constant_calls == ["standard_temperature"]
        pulse_factor = original_parameter_lookup(
            "co2_pulse_factor",
            namespace="anthroposphere.management.tillage",
            required=True,
        )
        base_fraction = original_parameter_lookup(
            "co2_pulse_base_fraction",
            namespace="anthroposphere.management.tillage",
            required=True,
        )
        expected = (
            soil_carbon
            * float(pulse_factor)
            * float(base_fraction)
            * 2.0
            / float(get_physical_constant("seconds_per_day"))
        )
        assert torch.allclose(pulse, expected, rtol=1.0e-7, atol=0.0)
    
    # L1.3: Scientific calculation accuracy (soil disturbance)
    def test_soil_disturbance_calculation(self, tillage_manager, sample_state, sample_environment):
        """Test soil disturbance rate calculation accuracy"""
        rates = call_forward(tillage_manager, {**sample_state, **sample_environment})
        
        # Check that disturbance rate is calculated
        assert "soil_disturbance_rate" in rates
        disturbance_rate = rates["soil_disturbance_rate"]
        
        # Should be positive for tillage operations
        assert torch.all(disturbance_rate >= 0)
        
        # Should increase with the intensity class of the executing operation
        rates_by_intensity = []
        for intensity in (
            TillageIntensity.LOW,
            TillageIntensity.MODERATE,
            TillageIntensity.HIGH,
            TillageIntensity.VERY_HIGH,
        ):
            manager = TillageManagement(config={"calendar_type": "noleap"})
            manager.tillage_events = [
                TillageEvent(
                    depth=TillageDepth.MEDIUM,
                    intensity=intensity,
                    date=200,
                    soil_moisture_limit=0.5,
                )
            ]
            test_state = dict(sample_state)
            test_state["day_of_year"] = torch.tensor(200.0)
            intensity_rates = call_forward(
                manager, {**test_state, **sample_environment}, time=199 * 86400.0, dt=86400.0
            )
            rates_by_intensity.append(intensity_rates["soil_disturbance_rate"].mean().item())

        # Rates should increase with intensity
        assert rates_by_intensity == sorted(rates_by_intensity)
        assert rates_by_intensity[3] > rates_by_intensity[0]  # High > Low intensity
    
    def test_carbon_loss_calculation(self, tillage_manager, sample_state, sample_environment):
        """Test carbon loss rate calculation"""
        rates = call_forward(tillage_manager, {**sample_state, **sample_environment})
        
        # Check carbon loss outputs
        assert "carbon_loss_rate" in rates
        assert "soil_co2_flux" in rates
        
        carbon_loss_rate = rates["carbon_loss_rate"]
        co2_flux = rates["soil_co2_flux"]
        
        # All should be non-negative
        assert torch.all(carbon_loss_rate >= 0)
        assert torch.all(co2_flux >= 0)
        
        # CO2 flux should be proportional to carbon loss
        if torch.any(carbon_loss_rate > 0):
            co2_to_c_ratio = co2_flux / torch.clamp(carbon_loss_rate, min=1e-10)
            # CO2 is 44 g/mol, C is 12 g/mol, so ratio should be 44/12 鈮?3.67
            assert torch.allclose(co2_to_c_ratio, torch.tensor(44/12), rtol=0.1)
    
    def test_residue_incorporation_calculation(self, tillage_manager, sample_state, sample_environment):
        """Test residue incorporation calculation"""
        rates = call_forward(tillage_manager, {**sample_state, **sample_environment})
        
        # Check residue incorporation outputs
        assert "residue_incorporation_rate" in rates
        assert "delta_residue_cover" in rates
        
        incorporation_rate = rates["residue_incorporation_rate"]
        cover_change = rates["delta_residue_cover"]
        
        # Should be related to residue cover
        initial_cover = sample_state['residue_cover']
        
        # Higher residue cover should lead to more incorporation
        if torch.any(initial_cover > 0):
            assert torch.all(incorporation_rate >= 0)
            assert torch.all(cover_change <= 0)  # Cover should decrease
    
    # L1.4: Conservation laws validation
    def test_soil_mass_conservation(self, tillage_manager):
        """Test soil mass conservation in tillage operations"""
        # Create a controlled test scenario
        initial_bulk_density = torch.tensor([1.4], device=DEVICE)
        initial_organic_carbon = torch.tensor([0.02], device=DEVICE)
        
        state = {
            'soil_bulk_density': initial_bulk_density,
            'soil_organic_carbon': initial_organic_carbon,
            'soil_moisture': torch.tensor([0.3], device=DEVICE),
            'soil_temperature': torch.tensor([298.15], device=DEVICE),
            'days_since_tillage': torch.tensor([365.0], device=DEVICE),
        }
        
        environment = {
            'soil_temperature': torch.tensor([298.15], device=DEVICE)
        }
        
        rates = call_forward(tillage_manager, {**state, **environment})
        
        # Check mass balance
        carbon_loss_rate = rates["carbon_loss_rate"]
        co2_flux = rates["soil_co2_flux"]
        
        # Carbon loss should be reflected in CO2 flux
        if torch.any(carbon_loss_rate > 0):
            # Mass balance: C lost = CO2 emitted (converted from CO2 to C)
            carbon_from_co2 = co2_flux * (12/44)
            mass_balance_error = torch.abs(carbon_loss_rate - carbon_from_co2)
            assert mass_balance_error < TOLERANCE
    
    def test_aggregate_mass_conservation(self, tillage_manager):
        """Test soil aggregate mass conservation"""
        # Test aggregate stability changes
        initial_aggregates = torch.tensor([0.6], device=DEVICE)
        
        state = {
            'soil_aggregates': initial_aggregates,
            'soil_moisture': torch.tensor([0.3], device=DEVICE),
            'soil_temperature': torch.tensor([298.15], device=DEVICE),
            'soil_organic_carbon': torch.tensor([5.0], device=DEVICE),  # kg C m-2
            'tillage_intensity': torch.tensor([0.7], device=DEVICE),
            'days_since_tillage': torch.tensor([365.0], device=DEVICE),
        }
        
        environment = {
            'soil_temperature': torch.tensor([298.15], device=DEVICE)
        }
        
        rates = call_forward(tillage_manager, {**state, **environment})
        
        # Check aggregate destruction rate
        if "aggregate_destruction_rate" in rates:
            destruction_rate = rates["aggregate_destruction_rate"]
            assert torch.all(destruction_rate >= 0)
            assert torch.all(destruction_rate <= 0.1)  # Should not destroy more than 10% per operation
    
    # L1.5: Tillage intensity effects on soil properties
    def test_tillage_intensity_effects(self, tillage_manager):
        """Test tillage intensity effects on soil properties"""
        # Test different tillage intensities
        intensities = [0.1, 0.3, 0.5, 0.7, 0.9]  # Very low to very high
        disturbance_rates = []
        carbon_losses = []
        
        for intensity in intensities:
            state = {
                'soil_moisture': torch.tensor([0.3], device=DEVICE),
                'soil_temperature': torch.tensor([298.15], device=DEVICE),
                'soil_organic_carbon': torch.tensor([0.02], device=DEVICE),
                'tillage_intensity': torch.tensor([intensity], device=DEVICE),
                'days_since_tillage': torch.tensor([365.0], device=DEVICE),
            }
            
            environment = {
                'soil_temperature': torch.tensor([298.15], device=DEVICE)
            }
            
            rates = call_forward(tillage_manager, {**state, **environment})
            disturbance_rates.append(rates["soil_disturbance_rate"].item())
            carbon_losses.append(rates["carbon_loss_rate"].item())
        
        # Both should increase with intensity
        for i in range(1, len(intensities)):
            assert disturbance_rates[i] >= disturbance_rates[i-1], \
                f"Disturbance should increase with intensity at intensity {intensities[i]}"
            assert carbon_losses[i] >= carbon_losses[i-1], \
                f"Carbon loss should increase with intensity at intensity {intensities[i]}"
    
    def test_moisture_effects_on_tillage(self, tillage_manager):
        """Test soil moisture effects on tillage efficiency"""
        # Test different moisture levels
        moisture_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Dry to wet
        mixing_efficiencies = []
        
        for moisture in moisture_levels:
            state = {
                'soil_moisture': torch.tensor([moisture], device=DEVICE),
                'soil_temperature': torch.tensor([298.15], device=DEVICE),
                'soil_organic_carbon': torch.tensor([5.0], device=DEVICE),  # kg C m-2
                'soil_bulk_density': torch.tensor([1.4], device=DEVICE),
                'tillage_intensity': torch.tensor([0.6], device=DEVICE),
                'days_since_tillage': torch.tensor([365.0], device=DEVICE),
            }
            
            environment = {
                'soil_temperature': torch.tensor([298.15], device=DEVICE)
            }
            
            rates = call_forward(tillage_manager, {**state, **environment})
            if "mixing_efficiency" in rates:
                mixing_efficiencies.append(rates["mixing_efficiency"].item())
        
        if len(mixing_efficiencies) > 1:
            # Optimum moisture should give highest efficiency
            # Usually around field capacity (0.3-0.4)
            max_efficiency_idx = np.argmax(mixing_efficiencies)
            optimum_moisture = moisture_levels[max_efficiency_idx]
            assert 0.2 <= optimum_moisture <= 0.5, \
                f"Optimum moisture should be in reasonable range: {optimum_moisture}"
    
    # L1.6: Abstract base class compliance
    def test_transport_mechanism_interface(self, tillage_manager, sample_state, sample_environment):
        """Test compliance with TransportMechanism-oriented interface"""
        # Forward-only entrypoint is required by architecture.
        assert hasattr(tillage_manager, 'forward')
        assert callable(tillage_manager.forward)
        assert not hasattr(tillage_manager, "compute_rate")
        
        # forward should return dictionary with rates
        rates = call_forward(tillage_manager, {**sample_state, **sample_environment})
        assert isinstance(rates, dict)
        assert len(rates) > 0
        
        # All rate values should be tensors
        for rate_name, rate_value in rates.items():
            assert isinstance(rate_value, torch.Tensor)
    
    def test_forward_method_consistency(self, tillage_manager):
        """Test backward consistency with forward method"""
        state = {
            'soil_moisture': torch.tensor([0.3], device=DEVICE),
            'soil_temperature': torch.tensor([298.15], device=DEVICE),
            'soil_organic_carbon': torch.tensor([5.0], device=DEVICE),  # kg C m-2
            'tillage_intensity': torch.tensor([0.6], device=DEVICE),
            'days_since_tillage': torch.tensor([365.0], device=DEVICE),
        }
        
        # Should be able to call forward method
        outputs = tillage_manager.forward(state, time=0, dt=3600)
        
        # Should return dictionary with expected outputs
        assert isinstance(outputs, dict)
        assert "soil_disturbance_rate" in outputs
        assert "carbon_loss_rate" in outputs
    
    def test_dependency_declaration(self, tillage_manager):
        """Test dependency declaration functionality"""
        dependencies = tillage_manager.get_dependencies()
        assert dependencies == []

    def test_carbon_loss_rate_is_converted_to_canonical_si_mass_flux(
        self,
        tillage_manager,
        sample_state,
        sample_environment,
    ):
        # A HIGH-intensity, MEDIUM-depth operation executes on day 200: the carbon
        # flush C f_0 f_p I D f_T / S_day of that step in canonical kg C m-2 s-1.
        tillage_manager.tillage_events = [
            TillageEvent(
                depth=TillageDepth.MEDIUM,
                intensity=TillageIntensity.HIGH,
                date=200,
                soil_moisture_limit=0.5,
            )
        ]
        operation_state = dict(sample_state)
        operation_state["day_of_year"] = torch.tensor(200.0)
        outputs = call_forward(
            tillage_manager,
            {**operation_state, **sample_environment},
            time=199 * 86400.0,
            dt=86400.0,
        )
        expected = tillage_manager._calculate_carbon_flush(
            tillage_manager._get_tillage_intensity_factor(TillageIntensity.HIGH),
            tillage_manager._get_tillage_depth_factor(TillageDepth.MEDIUM),
            sample_state["soil_organic_carbon"],
            sample_state["soil_temperature"],
        )

        assert torch.count_nonzero(outputs["carbon_loss_rate"]) == expected.numel()
        torch.testing.assert_close(outputs["carbon_loss_rate"], expected.detach())

    def test_real_simulator_replaces_tillage_state_without_compounding(self):
        simulator = _declared_simulator(
            {
                "max_time": 7200.0,
                "time_step": 3600.0,
                "calendar_type": "noleap",
                "conservation_checks": False,
                "device": "cpu",
            }
        )
        simulator.initialize_state(
            {
                "soil_moisture": torch.tensor([0.25]),
                "soil_temperature": torch.tensor([293.15]),
                "soil_organic_carbon": torch.tensor([5.0]),
                "day_of_year": torch.tensor(50.0),
                "days_since_tillage": torch.tensor([100.0]),
                "crop_residue": torch.tensor([0.5]),
                "residue_cover": torch.tensor([0.5]),
                "tillage_intensity": torch.tensor([0.0]),
            }
        )
        simulator.register_functor(
            TillageManagement(config={"calendar_type": "noleap"}),
            execution_group="fast",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            first = simulator.step(dt=3600.0)
            second = simulator.step(dt=3600.0)

        torch.testing.assert_close(
            first["days_since_tillage"],
            torch.tensor([100.0 + 1.0 / 24.0]),
        )
        torch.testing.assert_close(
            second["days_since_tillage"],
            torch.tensor([100.0 + 2.0 / 24.0]),
        )
        torch.testing.assert_close(
            first["porosity_modifier"],
            torch.ones_like(first["porosity_modifier"]),
        )
        torch.testing.assert_close(
            second["porosity_modifier"],
            torch.ones_like(second["porosity_modifier"]),
        )

    def test_real_tillage_event_keeps_residue_cover_within_physical_bounds(self):
        simulator = _declared_simulator(
            {
                "max_time": 3600.0,
                "time_step": 3600.0,
                "calendar_type": "noleap",
                "conservation_checks": False,
                "device": "cpu",
            }
        )
        initial_cover = torch.tensor([0.5])
        simulator.initialize_state(
            {
                "soil_moisture": torch.tensor([0.20]),
                "soil_temperature": torch.tensor([293.15]),
                "soil_organic_carbon": torch.tensor([5.0]),
                "day_of_year": torch.tensor(100.0),
                "days_since_tillage": torch.tensor([100.0]),
                "crop_residue": initial_cover.clone(),
                "residue_cover": initial_cover.clone(),
                "tillage_intensity": torch.tensor([0.5]),
            }
        )
        simulator.register_functor(
            TillageManagement(config={"calendar_type": "noleap"}),
            execution_group="fast",
        )

        simulator.step(dt=3600.0)
        residue_cover = simulator.state_manager.get_state()["residue_cover"]

        assert torch.all(residue_cover >= 0.0)
        assert torch.all(residue_cover <= initial_cover)

    def test_tillage_carbon_pulse_follows_the_soil_temperature(self):
        """The Q10 factor of the tillage carbon flush reads soil_temperature, not temperature."""
        pulses = {}
        for soil_temperature in (283.15, 303.15):
            simulator = _declared_simulator(
                {
                    "max_time": 3600.0,
                    "time_step": 3600.0,
                    "calendar_type": "noleap",
                    "conservation_checks": False,
                    "device": "cpu",
                }
            )
            simulator.initialize_state(
                {
                    "soil_moisture": torch.tensor([0.20]),
                    "soil_temperature": torch.tensor([soil_temperature]),
                    # A different air temperature, which the pulse must not read.
                    "temperature": torch.tensor([260.0]),
                    "soil_organic_carbon": torch.tensor([5.0]),
                    # Day 90 is the default primary pass (moisture limit 0.22).
                    "day_of_year": torch.tensor(90.0),
                    "days_since_tillage": torch.tensor([100.0]),
                    "crop_residue": torch.tensor([0.5]),
                    "residue_cover": torch.tensor([0.5]),
                    "tillage_intensity": torch.tensor([0.5]),
                }
            )
            simulator.register_functor(
                TillageManagement(config={"calendar_type": "noleap"}),
                execution_group="fast",
            )
            pulses[soil_temperature] = simulator.step(dt=3600.0)["carbon_loss_rate"]

        assert torch.all(pulses[283.15] > 0.0)
        # References: Reicosky (1997) tillage CO2 flush; Davidson and Janssens
        #   (2006) Q10 temperature response of soil organic matter oxidation.
        # Mathematical expression: F(T1) / F(T2) = 2 ** ((T1 - T2) / 10 K).
        # Parameter meanings: T1, T2 the soil temperatures of the two runs.
        # Units: T [K]; F [kg C m-2 s-1]; the ratio [1].
        # Domain/constraints: identical states apart from soil_temperature.
        # Sign convention: positive F is carbon emitted from the soil pool.
        torch.testing.assert_close(
            pulses[303.15] / pulses[283.15],
            torch.full_like(pulses[283.15], 4.0),
            rtol=1.0e-5,
            atol=0.0,
        )
        tillage = TillageManagement(config={"calendar_type": "noleap"})
        assert "soil_temperature" in tillage.metadata.input_variables
        assert "temperature" not in tillage.metadata.input_variables
    
    # Additional validation tests
    def test_tillage_parameter_values(self, tillage_manager):
        """Test that tillage model parameters have reasonable values"""
        # Test key tillage parameters
        disturbance_factor = tillage_manager.get_parameter_value(
            "disturbance_factor", namespace="anthroposphere.management.tillage", required=True
        )
        assert 0.1 <= disturbance_factor <= 1.0

        pulse_factor = tillage_manager.get_parameter_value(
            "co2_pulse_factor", namespace="anthroposphere.management.tillage", required=True
        )
        assert 1.0 <= pulse_factor <= 5.0

    def test_edge_cases_and_error_handling(self, tillage_manager):
        """Test edge cases and error handling"""
        # Test with zero tillage intensity
        zero_intensity_state = {
            'soil_moisture': torch.tensor([0.3], device=DEVICE),
            'soil_temperature': torch.tensor([298.15], device=DEVICE),
            'soil_organic_carbon': torch.tensor([5.0], device=DEVICE),  # kg C m-2
            'tillage_intensity': torch.tensor([0.0], device=DEVICE),
            'days_since_tillage': torch.tensor([365.0], device=DEVICE),
        }
        
        zero_intensity_env = {
            'soil_temperature': torch.tensor([298.15], device=DEVICE)
        }
        
        rates = call_forward(tillage_manager, {**zero_intensity_state, **zero_intensity_env})
        
        # Should not crash and should return minimal fluxes
        for rate_name, rate_value in rates.items():
            if "rate" in rate_name:
                assert torch.all(rate_value >= 0)  # Should be non-negative
        
        # Test with maximum tillage intensity
        max_intensity_state = zero_intensity_state.copy()
        max_intensity_state['tillage_intensity'] = torch.tensor([1.0], device=DEVICE)
        
        max_rates = call_forward(tillage_manager, {**max_intensity_state, **zero_intensity_env})
        
        # Should handle high intensity without crashing
        assert all(torch.isfinite(rate).all() for rate in max_rates.values())


# ---------------------------------------------------------------------------
# One parameter accessor, explicit class lookups, parameter-backed schedule and one
# tillage carbon flux (MA-21 round 14)
# ---------------------------------------------------------------------------


def _operation_state(**overrides):
    state = {
        "soil_moisture": torch.tensor([0.2]),
        "soil_temperature": torch.tensor([288.15]),
        "soil_organic_carbon": torch.tensor([5.0]),
        "day_of_year": torch.tensor(90.0),
        "days_since_tillage": torch.tensor([365.0]),
        "residue_cover": torch.tensor([0.4]),
    }
    state.update(overrides)
    return state


def test_tillage_has_no_private_parameter_accessor():
    """Parameters are read through FunctorBase.get_parameter_value only; there is no
    override that falls back to a per-type property table."""
    assert "get_parameter_value" not in vars(TillageManagement)
    assert not hasattr(TillageManagement, "_get_tillage_properties")


def test_intensity_and_depth_lookups_reject_an_unknown_class():
    """Every declared class has a factor; anything else raises instead of 0.0 or 0.5."""
    manager = TillageManagement(config={"calendar_type": "noleap"})
    with pytest.raises(TypeError, match="TillageIntensity"):
        manager._get_tillage_intensity_factor("high")
    with pytest.raises(TypeError, match="TillageDepth"):
        manager._get_tillage_depth_factor("deep")


def test_schedule_window_and_minimum_interval_are_parameters(monkeypatch):
    """d_C <= W and t_since > T_min with W and T_min read from the parameter database
    (scheduling conventions, units day) instead of the literals 3 and 14."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    for key in ("schedule_window_days", "minimum_operation_interval_days"):
        spec = manager.get_parameter(f"anthroposphere.management.tillage.{key}")
        assert spec is not None, key
        assert spec.units == "day", key

    tillage = TillageManagement(config={"calendar_type": "noleap"})
    dry = torch.tensor([0.1])
    assert tillage._check_tillage_schedule(
        92, dry, torch.tensor([30.0]), calendar_type="noleap"
    ) is not None
    assert tillage._check_tillage_schedule(
        90, dry, torch.tensor([10.0]), calendar_type="noleap"
    ) is None

    original = tillage.require_scalar
    configured = {"schedule_window_days": 1.0, "minimum_operation_interval_days": 7.0}

    def require_scalar(name, *args, **kwargs):
        if name in configured:
            return configured[name]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(tillage, "require_scalar", require_scalar)
    assert tillage._check_tillage_schedule(
        92, dry, torch.tensor([30.0]), calendar_type="noleap"
    ) is None
    assert tillage._check_tillage_schedule(
        90, dry, torch.tensor([10.0]), calendar_type="noleap"
    ) is not None


def test_tillage_emits_one_soil_carbon_flux_per_operation():
    """The tillage-induced flush (Reicosky 1997) is one carbon flux: the operation-step
    pulse F = C f_0 f_p I D Q10^((T - T_std)/10) / S_day. soil_co2_flux is its CO2-mass
    form, the mass-balance check counts it once, and no co2_pulse port or contract
    remains."""
    from relic.contracts.contract_loader import PACKAGED_CONTRACTS_DIR, load_contract_registry
    from relic.parameters import get_global_manager

    tillage = TillageManagement(config={"calendar_type": "noleap"})
    outputs = tillage.forward(_operation_state(), 0.0, 86400.0)
    assert "co2_pulse" not in tillage.metadata.output_variables
    assert "co2_pulse" not in outputs
    assert load_contract_registry(PACKAGED_CONTRACTS_DIR).get_contract("co2_pulse") is None

    manager = get_global_manager()

    def parameter(name):
        return float(manager.get(f"anthroposphere.management.tillage.{name}"))

    q10 = float(manager.get("pedosphere.physical.q10_decomposition"))
    t_standard = float(get_physical_constant("standard_temperature"))
    seconds_per_day = float(get_physical_constant("seconds_per_day"))
    expected = (
        5.0
        * parameter("co2_pulse_base_fraction")
        * parameter("co2_pulse_factor")
        * parameter("high_intensity_factor")
        * parameter("deep_depth_factor")
        * q10 ** ((288.15 - t_standard) / 10.0)
        / seconds_per_day
    )
    carbon = outputs["carbon_loss_rate"].value.detach()
    torch.testing.assert_close(carbon, torch.full_like(carbon, expected), rtol=1e-5, atol=0.0)
    m_co2 = get_physical_constant("relic.physical.constants.molecular_weights.co2")
    m_c = get_physical_constant("relic.physical.constants.molecular_weights.carbon")
    torch.testing.assert_close(
        outputs["soil_co2_flux"].value.detach(), carbon * (m_co2 / m_c), rtol=1e-6, atol=0.0
    )

    # Available carbon 1.5 F dt closes the balance only when F is counted once.
    flux = torch.tensor([expected])
    residual = tillage._check_conservation(
        soil_carbon=flux * 86400.0 * 1.5, carbon_flux=flux, dt=86400.0
    )
    assert torch.count_nonzero(residual) == 0


def test_tillage_carbon_flux_is_zero_outside_operation_steps():
    """Without an operation in the step the tillage emits no soil carbon; post-tillage
    decomposition is the decomposition Functors' process."""
    tillage = TillageManagement(config={"calendar_type": "noleap"})
    tillage.forward(_operation_state(), 0.0, 86400.0)
    later = tillage.forward(
        _operation_state(
            day_of_year=torch.tensor(100.0), days_since_tillage=torch.tensor([10.0])
        ),
        10 * 86400.0,
        86400.0,
    )
    assert torch.count_nonzero(later["carbon_loss_rate"].value) == 0
    assert torch.count_nonzero(later["soil_co2_flux"].value) == 0


def test_default_tillage_schedule_comes_from_the_parameter_database():
    """The four default operations take their day of year and soil-moisture upper limit
    from anthroposphere.management.tillage.default_schedule.* (item 111)."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    tillage = TillageManagement(config={"calendar_type": "noleap"})
    names = ("primary", "secondary", "conservation", "autumn_chisel")
    assert len(tillage.tillage_events) == len(names)
    for name, event in zip(names, tillage.tillage_events):
        prefix = f"anthroposphere.management.tillage.default_schedule.{name}"
        assert event.date == int(manager.get(f"{prefix}.date"))
        assert event.soil_moisture_limit == pytest.approx(
            float(manager.get(f"{prefix}.soil_moisture_limit"))
        )


def test_tillage_event_has_only_the_fields_the_model_reads():
    """TillageEvent carries depth, intensity, date and soil-moisture limit, the fields
    the schedule and the responses read; the unread type, residue-incorporation and
    duration fields and the uncalled set_tillage_system are gone (item 111)."""
    import dataclasses

    assert {f.name for f in dataclasses.fields(TillageEvent)} == {
        "depth",
        "intensity",
        "date",
        "soil_moisture_limit",
    }
    assert not hasattr(TillageManagement, "set_tillage_system")


def test_tillage_reads_no_environment_channel():
    """Inputs come from the state only: forward reads no environment keyword and the
    rate core takes no environment mapping (item 112)."""
    import inspect

    assert "environment" not in inspect.signature(TillageManagement._compute_rate_core).parameters
    assert "environment" not in inspect.getsource(TillageManagement.forward)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


