#!/usr/bin/env python3
"""
L1 Unit Tests for Grazing Management Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy (grazing flux)
- L1.4: Conservation laws validation
- L1.5: Grazing intensity and pasture management
- L1.6: Abstract base class compliance

Scientific References:
- Hodgson, J. (1990). Grazing management: Science into practice. Longman Scientific & Technical.
- Briske, D.D., et al. (2008). Conservation benefits of rangeland practices. Ecological Applications 18, 1743-1757.
- Milchunas, D.G., et al. (1988). A generalized model of grazing effects. Ecological Applications 5, 748-758.

Test Framework: pytest
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
from relic.scientific.anthroposphere.management.grazing import (
    GrazingManagement,
    GrazingSystem,
    LivestockType,
)
from relic.scientific.anthroposphere.management import grazing as grazing_module
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3


def _value(item: UpdateItem | torch.Tensor) -> torch.Tensor:
    return item.value if isinstance(item, UpdateItem) else item


class TestGrazingManagementL1:
    """L1 Unit Tests for GrazingManagement"""
    
    @pytest.fixture
    def grazing_manager(self):
        """Create grazing manager instance for testing"""
        return GrazingManagement(config={"calendar_type": "noleap"})
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'aboveground_biomass': torch.tensor([2.0, 2.5, 3.0, 3.5, 4.0], device=DEVICE).reshape(batch_shape), # kg m-2
            'belowground_biomass': torch.tensor([0.6, 0.75, 0.9, 1.05, 1.2], device=DEVICE).reshape(batch_shape), # kg m-2
            'vegetation_quality': torch.tensor([0.7, 0.75, 0.8, 0.85, 0.9], device=DEVICE).reshape(batch_shape), # dimensionless
            'soil_moisture': torch.tensor([0.2, 0.25, 0.3, 0.35, 0.4], device=DEVICE).reshape(batch_shape), # m3 m-3
            'temperature': torch.tensor([288.15, 293.15, 298.15, 303.15, 308.15], device=DEVICE).reshape(batch_shape) # K
        }
    
    @pytest.fixture
    def sample_environment(self):
        """Sample environment dictionary for testing"""
        # Grazing mostly depends on state variables (biomass), environment passed as state or context
        return {}

    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, grazing_manager):
        """Test that GrazingManagement can be instantiated"""
        assert grazing_manager is not None
        assert isinstance(grazing_manager, GrazingManagement)
        assert isinstance(grazing_manager, BiologicalProcessFunctor)
        assert hasattr(grazing_manager, 'name')
        assert grazing_manager.name == "grazing_management"
    
    def test_metadata_completeness(self, grazing_manager):
        """Test that metadata is properly set"""
        metadata = grazing_manager.get_metadata()
        assert metadata is not None
        assert metadata.name == "grazing_management"
        assert len(metadata.references) > 0
        # Check for key terms in mathematical expression
        assert "I =" in metadata.mathematical_expression
        assert set(metadata.conservation_laws) == {"grazing_matter_boundary"}
        assert "vegetation_quality" in metadata.input_variables
        assert "forage_quality" not in metadata.input_variables
        assert "temperature" not in metadata.input_variables
        assert {
            "aboveground_biomass",
            "livestock_carbon_intake_flux",
            "livestock_nitrogen_intake_flux",
            "livestock_carbon_export_flux",
            "livestock_nitrogen_export_flux",
        }.issubset(metadata.output_variables)
        assert {
            "diagnostics.grazing_biomass_removal",
            "diagnostics.grazing_vegetation_change",
            "diagnostics.grazing_mass_balance_error",
        }.isdisjoint(metadata.output_variables)

    @pytest.mark.parametrize("livestock_type", tuple(LivestockType))
    def test_every_declared_livestock_type_has_parameter_support(
        self,
        grazing_manager,
        livestock_type,
    ):
        """Every public livestock enum member must resolve a complete bundle."""
        characteristics = grazing_manager._get_livestock_characteristics(
            livestock_type
        )

        assert set(characteristics) == {
            "intake_rate",
            "digestibility",
            "n_content",
            "p_content",
            "metabolism_factor",
            "trampling_impact",
        }

    @pytest.mark.parametrize("livestock_type", tuple(LivestockType))
    @pytest.mark.parametrize(
        "field",
        ("digestibility", "n_content", "p_content"),
    )
    def test_livestock_mass_fraction_bounds_use_unit_interval(
        self,
        grazing_manager,
        livestock_type,
        field,
    ):
        """Mass-fraction bounds in the parameter SSOT cannot exceed one."""
        key = (
            "anthroposphere.management.grazing.livestock."
            f"{livestock_type.value}.{field}"
        )
        specification = grazing_manager.parameter_manager.get_parameter(key)

        assert specification.bounds is not None
        lower, upper = (float(value) for value in specification.bounds)
        assert 0.0 <= lower <= float(specification.value) <= upper <= 1.0
    
    def test_grazing_schedule_initialization(self, grazing_manager):
        """Test that default grazing schedule is properly initialized"""
        assert len(grazing_manager.grazing_events) > 0
        # Check for updated attribute names (system instead of grazing_type)
        assert any(event.system == GrazingSystem.CONTINUOUS for event in grazing_manager.grazing_events)
        assert any(event.system == GrazingSystem.ROTATIONAL for event in grazing_manager.grazing_events)
    
    # L1.2: Parameter management system validation
    def test_parameter_access_through_manager(self, grazing_manager):
        """Test that parameters are accessible through ParameterManager"""
        # Test getting parameters using get_parameter_value
        # "n_retention" is used in _calculate_excretion.
        # It should return a float (or tensor) from the parameter database.
        val = grazing_manager.get_parameter_value(
            "n_retention",
            namespace="anthroposphere.management.grazing",
            validate=True,
            required=True,
            source_order=("manager",),
        )
        assert val is not None
        # It should be a tensor or convertable to float
        assert isinstance(val, (float, torch.Tensor))

    def test_day_length_uses_physical_constant_ssot(self, monkeypatch):
        """Grazing time conversion must use the physical-constant API."""
        constant_calls = []

        def tracked_get_physical_constant(name):
            constant_calls.append(name)
            return 12345.0

        monkeypatch.setattr(
            grazing_module,
            "get_physical_constant",
            tracked_get_physical_constant,
            raising=False,
        )
        manager = GrazingManagement(config={"calendar_type": "noleap"})

        assert manager._seconds_per_day_value() == pytest.approx(12345.0)
        assert constant_calls == ["relic.physical.constants.time.seconds_per_day"]
        assert (
            "relic.physical.constants.time.seconds_per_day"
            not in manager.required_parameters()
        )
    
    # L1.3: Scientific calculation accuracy (grazing flux)
    def test_grazing_flux_calculation(self, grazing_manager, sample_state):
        """Test grazing flux calculation accuracy"""
        # Mock a specific day to ensure grazing is active (Spring event starts day 120)
        day_of_year = 150
        dt = 86400.0 # 1 day
        
        outputs = grazing_manager.forward(sample_state, time=0, dt=dt, day_of_year=day_of_year)
        
        assert "grazing_flux" in outputs
        grazing_flux = _value(outputs["grazing_flux"])
        
        # Should be positive
        assert torch.all(grazing_flux >= 0)
        
        # Should increase with biomass (functional response)
        # sample_state biomass increases from index 0 to 4
        # flux should also increase
        flux_mean = grazing_flux.mean(dim=1) # if shape is (batch, 1)
        assert flux_mean[4] > flux_mean[0]

    def test_360_day_cross_year_schedule_uses_calendar_length(
        self,
        sample_state,
    ):
        manager = GrazingManagement(config={"calendar_type": "360_day"})
        event = manager.grazing_events[0]
        event.start_date = 350
        event.duration = 20.0
        manager.grazing_events = [event]

        active_outputs = manager.forward(
            sample_state,
            time=0.0,
            dt=3600.0,
            day_of_year=9,
            calendar_type="360_day",
        )
        endpoint_outputs = manager.forward(
            sample_state,
            time=0.0,
            dt=3600.0,
            day_of_year=10,
            calendar_type="360_day",
        )

        assert torch.all(_value(active_outputs["grazing_active"]) == 1.0)
        assert torch.count_nonzero(
            _value(endpoint_outputs["grazing_active"])
        ) == 0

    def test_zero_duration_grazing_event_has_empty_half_open_window(
        self,
        sample_state,
    ):
        manager = GrazingManagement(config={"calendar_type": "noleap"})
        event = manager.grazing_events[0]
        event.start_date = 120
        event.duration = 0.0
        manager.grazing_events = [event]

        outputs = manager.forward(
            sample_state,
            time=0.0,
            dt=3600.0,
            day_of_year=120,
        )

        assert torch.count_nonzero(_value(outputs["grazing_active"])) == 0

    def test_grazing_duration_cannot_exceed_calendar_year(
        self,
        sample_state,
    ):
        manager = GrazingManagement(config={"calendar_type": "360_day"})
        event = manager.grazing_events[0]
        event.duration = 361.0
        manager.grazing_events = [event]

        with pytest.raises(ValueError, match="no greater than the calendar year"):
            manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                day_of_year=120,
                calendar_type="360_day",
            )

    @pytest.mark.parametrize("invalid_start_date", [True, 120.5])
    def test_grazing_event_start_date_rejects_boolean_and_fractional_values(
        self,
        sample_state,
        invalid_start_date,
    ):
        manager = GrazingManagement(config={"calendar_type": "noleap"})
        event = manager.grazing_events[0]
        event.start_date = invalid_start_date
        manager.grazing_events = [event]

        with pytest.raises(
            ValueError,
            match="day_of_year must be a finite in-domain integer numeric scalar",
        ):
            manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                day_of_year=150,
            )

    @pytest.mark.parametrize(
        "invalid_duration",
        [True, float("nan"), float("inf"), -1.0],
    )
    def test_grazing_event_duration_requires_finite_nonnegative_real_scalar(
        self,
        sample_state,
        invalid_duration,
    ):
        manager = GrazingManagement(config={"calendar_type": "noleap"})
        event = manager.grazing_events[0]
        event.duration = invalid_duration
        manager.grazing_events = [event]

        with pytest.raises(
            ValueError,
            match="Grazing event duration must be finite and non-negative",
        ):
            manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                day_of_year=150,
            )

    def test_forward_requires_explicit_calendar_contract(self, sample_state):
        manager = GrazingManagement()

        with pytest.raises(ValueError, match="requires explicit calendar_type"):
            manager.forward(
                sample_state,
                time=0.0,
                dt=3600.0,
                day_of_year=150,
            )
        
    def test_vegetation_quality_effects(self, grazing_manager, sample_state):
        """Test vegetation quality effects on grazing rates"""
        # Mock day
        day_of_year = 150
        dt = 86400.0
        
        # Clone state and modify quality
        state_low_qual = {k: v.clone() for k, v in sample_state.items()}
        state_high_qual = {k: v.clone() for k, v in sample_state.items()}
        
        state_low_qual['vegetation_quality'][:] = 0.5
        state_high_qual['vegetation_quality'][:] = 0.9
        
        out_low = grazing_manager.forward(state_low_qual, time=0, dt=dt, day_of_year=day_of_year)
        out_high = grazing_manager.forward(state_high_qual, time=0, dt=dt, day_of_year=day_of_year)
        
        # Higher quality -> higher intake (selectivity)
        assert _value(out_high["grazing_flux"]).mean() > _value(
            out_low["grazing_flux"]
        ).mean()

    def test_trampling_damage_calculation(self, grazing_manager, sample_state):
        """Test trampling damage calculation"""
        day_of_year = 150
        dt = 86400.0
        
        outputs = grazing_manager.forward(sample_state, time=0, dt=dt, day_of_year=day_of_year)
        
        assert "trampling_damage" in outputs
        trampling = _value(outputs["trampling_damage"])
        
        assert torch.all(trampling >= 0)
        
        # Trampling increases with soil moisture (sample_state moisture increases)
        # Check relationship
        tramp_mean = trampling.mean(dim=1)
        assert tramp_mean[4] > tramp_mean[0]

    def test_manure_deposition_calculation(self, grazing_manager, sample_state):
        """Test manure deposition outputs"""
        day_of_year = 150
        dt = 86400.0
        
        outputs = grazing_manager.forward(sample_state, time=0, dt=dt, day_of_year=day_of_year)
        
        assert "dung_c_flux" in outputs
        assert "dung_n_flux" in outputs
        assert "urine_n_flux" in outputs
        
        # Check non-negativity
        assert torch.all(_value(outputs["dung_c_flux"]) >= 0)
        
        # Check N partitioning (Urine usually > Dung N)
        # Based on code: urine_n = n_excreted * 0.6, dung_n = n_excreted * 0.4
        # So urine > dung
        assert _value(outputs["urine_n_flux"]).mean() > _value(
            outputs["dung_n_flux"]
        ).mean()
        assert torch.allclose(
            _value(outputs["livestock_carbon_intake_flux"]),
            _value(outputs["dung_c_flux"])
            + _value(outputs["livestock_carbon_export_flux"]),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        assert torch.allclose(
            _value(outputs["livestock_nitrogen_intake_flux"]),
            _value(outputs["dung_n_flux"])
            + _value(outputs["urine_n_flux"])
            + _value(outputs["livestock_nitrogen_export_flux"]),
            rtol=1.0e-6,
            atol=1.0e-12,
        )

        cold_state = {key: value.clone() for key, value in sample_state.items()}
        warm_state = {key: value.clone() for key, value in sample_state.items()}
        cold_state["temperature"].fill_(278.15)
        warm_state["temperature"].fill_(308.15)
        cold = grazing_manager.forward(
            cold_state, time=0, dt=dt, day_of_year=day_of_year
        )
        warm = grazing_manager.forward(
            warm_state, time=0, dt=dt, day_of_year=day_of_year
        )
        for key in ("dung_c_flux", "dung_n_flux", "urine_n_flux"):
            assert torch.allclose(_value(cold[key]), _value(warm[key]))

    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        (
            ("digestibility", -0.01),
            ("digestibility", 1.01),
            ("digestibility", float("nan")),
            ("n_content", -0.01),
            ("n_content", 1.01),
            ("n_content", float("inf")),
        ),
    )
    def test_excreta_partition_rejects_nonphysical_mass_fractions(
        self,
        grazing_manager,
        field,
        invalid_value,
    ):
        """Mass fractions outside the finite unit interval are invalid."""
        livestock_chars = {
            "digestibility": 0.65,
            "n_content": 0.025,
        }
        livestock_chars[field] = invalid_value

        with pytest.raises(ValueError, match=field):
            grazing_manager._calculate_excreta_return(
                torch.tensor([1.0], dtype=torch.float64),
                livestock_chars,
            )

    @pytest.mark.parametrize(
        ("urine_partition", "dung_partition"),
        (
            (0.0, 0.0),
            (-0.1, 1.0),
            (1.0, -0.1),
            (float("nan"), 0.4),
        ),
    )
    def test_excreta_partition_requires_nonnegative_positive_support(
        self,
        grazing_manager,
        monkeypatch,
        urine_partition,
        dung_partition,
    ):
        """Partition weights cannot rely on an epsilon fallback."""
        original_require_scalar = grazing_manager.require_scalar

        def require_scalar(name, *args, **kwargs):
            if name == "urine_partition":
                return urine_partition
            if name == "dung_partition":
                return dung_partition
            return original_require_scalar(name, *args, **kwargs)

        monkeypatch.setattr(grazing_manager, "require_scalar", require_scalar)

        with pytest.raises(ValueError, match="partition"):
            grazing_manager._calculate_excreta_return(
                torch.tensor([1.0], dtype=torch.float64),
                {"digestibility": 0.65, "n_content": 0.025},
            )

    # L1.4: Conservation laws validation
    def test_state_update_and_open_matter_partitions(
        self, grazing_manager, sample_state
    ):
        """Dry matter, carbon, and nitrogen routes must close explicitly."""
        day_of_year = 150
        dt = 86400.0
        outputs = grazing_manager.forward(
            sample_state, time=0, dt=dt, day_of_year=day_of_year
        )

        grazing_flux = _value(outputs["grazing_flux"])
        step_intake = grazing_flux * dt
        biomass_update = outputs["aboveground_biomass"]
        cumulative_intake = outputs["cumulative_intake"]
        days_grazed = outputs["days_grazed"]

        assert biomass_update.mode is UpdateMode.ACCUMULATE
        assert biomass_update.per_timestep is True
        assert biomass_update.unit == "kg m-2"
        assert biomass_update.source_functor == grazing_manager.name
        assert torch.allclose(_value(biomass_update), -step_intake)
        assert torch.all(-_value(biomass_update) <= sample_state["aboveground_biomass"])

        assert cumulative_intake.mode is UpdateMode.ACCUMULATE
        assert cumulative_intake.per_timestep is True
        assert torch.allclose(_value(cumulative_intake), step_intake)
        assert days_grazed.mode is UpdateMode.ACCUMULATE
        assert days_grazed.per_timestep is True
        assert torch.allclose(
            _value(days_grazed),
            torch.ones_like(sample_state["aboveground_biomass"]),
        )

        repeated = grazing_manager.forward(
            sample_state, time=0, dt=dt, day_of_year=day_of_year
        )
        for key in ("aboveground_biomass", "cumulative_intake", "days_grazed"):
            assert torch.equal(_value(outputs[key]), _value(repeated[key]))

    def test_declared_open_boundary_conservation_checker(
        self, grazing_manager, sample_state
    ):
        """The declared open matter boundary closes DM, carbon, and nitrogen."""
        outputs = grazing_manager.forward(
            sample_state, time=0, dt=86400.0, day_of_year=150
        )

        residuals = grazing_manager.check_conservation_laws(sample_state, outputs)

        assert residuals == {
            "grazing_matter_boundary": pytest.approx(0.0, abs=1.0e-12)
        }

        tampered = dict(outputs)
        item = tampered["livestock_nitrogen_export_flux"]
        tampered["livestock_nitrogen_export_flux"] = UpdateItem(
            value=item.value * item.value.new_tensor(0.9),
            mode=item.mode,
            unit=item.unit,
            per_timestep=item.per_timestep,
            source_functor=item.source_functor,
        )
        assert grazing_manager.check_conservation_laws(
            sample_state, tampered
        )["grazing_matter_boundary"] > 0.0

    def test_conservation_checker_fails_closed_on_malformed_ports(
        self, grazing_manager, sample_state
    ):
        """Missing, non-finite, and incompatible boundary ports are invalid."""
        outputs = grazing_manager.forward(
            sample_state, time=0, dt=86400.0, day_of_year=150
        )
        required = (
            "aboveground_biomass",
            "cumulative_intake",
            "livestock_carbon_intake_flux",
            "dung_c_flux",
            "livestock_carbon_export_flux",
            "livestock_nitrogen_intake_flux",
            "dung_n_flux",
            "urine_n_flux",
            "livestock_nitrogen_export_flux",
        )
        for key in required:
            missing = dict(outputs)
            missing.pop(key)
            with pytest.raises(KeyError, match=key):
                grazing_manager.check_conservation_laws(sample_state, missing)

        malformed = dict(outputs)
        item = malformed["dung_n_flux"]
        malformed["dung_n_flux"] = UpdateItem(
            value=torch.full_like(item.value, float("nan")),
            mode=item.mode,
            unit=item.unit,
            per_timestep=item.per_timestep,
            source_functor=item.source_functor,
        )
        with pytest.raises(ValueError, match="finite"):
            grazing_manager.check_conservation_laws(sample_state, malformed)

        malformed = dict(outputs)
        item = malformed["urine_n_flux"]
        malformed["urine_n_flux"] = UpdateItem(
            value=torch.ones(2, dtype=item.value.dtype),
            mode=item.mode,
            unit=item.unit,
            per_timestep=item.per_timestep,
            source_functor=item.source_functor,
        )
        with pytest.raises(ValueError, match="shape"):
            grazing_manager.check_conservation_laws(sample_state, malformed)

    # L1.6: Abstract base class compliance
    def test_forward_method_compliance(self, grazing_manager, sample_state):
        """Test compliance with FunctorBase forward interface"""
        state = {key: value.clone() for key, value in sample_state.items()}
        outputs = grazing_manager.forward(state, time=0, dt=3600, day_of_year=150)
        assert isinstance(outputs, dict)
        assert set(outputs) == set(grazing_manager.metadata.output_variables)
        assert all(
            isinstance(value, (torch.Tensor, UpdateItem))
            for value in outputs.values()
        )
        for name in (
            "grazing_flux",
            "dung_c_flux",
            "dung_n_flux",
            "urine_n_flux",
            "livestock_carbon_intake_flux",
            "livestock_nitrogen_intake_flux",
            "livestock_carbon_export_flux",
            "livestock_nitrogen_export_flux",
            "trampling_damage",
            "grazing_active",
            "current_stocking_rate",
            "utilization_rate",
        ):
            item = outputs[name]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
        for name in ("aboveground_biomass", "cumulative_intake", "days_grazed"):
            item = outputs[name]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True

        # D is the stocking rate of the active (spring) grazing event (item 116).
        expected_pressure = (
            grazing_manager.grazing_events[0].stocking_rate * state["vegetation_quality"]
        )
        assert torch.allclose(
            outputs["diagnostics.grazing_pressure"],
            expected_pressure,
            rtol=0.0,
            atol=0.0,
        )
        assert outputs["diagnostics.grazing_pressure"].dtype == state["aboveground_biomass"].dtype
        assert outputs["diagnostics.grazing_pressure"].device == state["aboveground_biomass"].device

        inactive_outputs = grazing_manager.forward(
            state,
            time=0,
            dt=3600,
            day_of_year=50,
        )
        assert torch.count_nonzero(inactive_outputs["diagnostics.grazing_pressure"]) == 0

    def test_nonfinite_event_stocking_rate_fails_fast(self, grazing_manager, sample_state):
        """Reject a non-finite event stocking density before scientific calculations."""
        state = {key: value.clone() for key, value in sample_state.items()}
        grazing_manager.grazing_events[0].stocking_rate = float("nan")

        with pytest.raises(ValueError, match="stocking_rate must contain only finite values"):
            grazing_manager.forward(state, time=0, dt=3600, day_of_year=150)

    @pytest.mark.parametrize("quality", [float("nan"), float("inf")])
    def test_nonfinite_vegetation_quality_fails_fast(
        self,
        grazing_manager,
        sample_state,
        quality,
    ):
        """Reject non-finite canonical forage quality before calculations."""
        state = {key: value.clone() for key, value in sample_state.items()}
        state["vegetation_quality"] = torch.full_like(
            state["aboveground_biomass"],
            quality,
        )

        with pytest.raises(ValueError, match="vegetation_quality must contain only finite values"):
            grazing_manager.forward(state, time=0, dt=3600, day_of_year=150)

    def test_old_forage_quality_key_does_not_bypass_required_input(
        self,
        grazing_manager,
        sample_state,
    ):
        """Require the canonical vegetation_quality state port."""
        state = {key: value.clone() for key, value in sample_state.items()}
        state["forage_quality"] = state.pop("vegetation_quality")

        with pytest.raises(KeyError, match="vegetation_quality"):
            grazing_manager.forward(state, time=0, dt=3600, day_of_year=150)

    @pytest.mark.parametrize(
        "name",
        ("aboveground_biomass", "vegetation_quality", "soil_moisture"),
    )
    def test_required_state_ports_fail_fast(
        self,
        grazing_manager,
        sample_state,
        name,
    ):
        """Required scientific state ports cannot use hidden defaults."""
        state = {key: value.clone() for key, value in sample_state.items()}
        state.pop(name)

        with pytest.raises(KeyError, match=name):
            grazing_manager.forward(state, time=0, dt=3600, day_of_year=150)

    @pytest.mark.parametrize(
        ("name", "value", "error", "message"),
        (
            ("aboveground_biomass", 2.0, TypeError, "Tensor"),
            (
                "aboveground_biomass",
                torch.empty(0),
                ValueError,
                "must not be empty",
            ),
            (
                "aboveground_biomass",
                torch.ones(5, 1, dtype=torch.int64),
                TypeError,
                "floating point",
            ),
            (
                "soil_moisture",
                torch.ones(2, dtype=torch.float32),
                ValueError,
                "shape",
            ),
        ),
    )
    def test_malformed_state_ports_fail_fast(
        self,
        grazing_manager,
        sample_state,
        name,
        value,
        error,
        message,
    ):
        """Malformed required state ports fail before grazing calculations."""
        state = {key: tensor.clone() for key, tensor in sample_state.items()}
        state[name] = value

        with pytest.raises(error, match=message):
            grazing_manager.forward(state, time=0, dt=3600, day_of_year=150)

    def test_long_timestep_cannot_overdraw_biomass(
        self,
        grazing_manager,
        sample_state,
    ):
        """The integrated grazing update is limited by available storage."""
        state = {key: value.clone() for key, value in sample_state.items()}
        initial = state["aboveground_biomass"].clone()
        outputs = grazing_manager.forward(
            state,
            time=0,
            dt=10.0 * 86400.0,
            day_of_year=150,
        )

        assert torch.all(-outputs["aboveground_biomass"].value <= initial)
        assert torch.all(initial + outputs["aboveground_biomass"].value >= 0)
        assert torch.equal(state["aboveground_biomass"], initial)

    def test_dependency_declaration(self, grazing_manager):
        """Test dependency declaration functionality"""
        dependencies = grazing_manager.get_dependencies()
        assert isinstance(dependencies, list)
        assert dependencies == []

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

