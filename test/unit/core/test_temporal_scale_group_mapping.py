"""Unit tests for TemporalScale-to-execution-group mapping."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import pytest
import torch

from relic.core.functor import (
    ExecutionPhase,
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateMode
from relic.coupling.scales.temporal.core import PeriodIndexer
from relic.parameters import get_global_manager
from relic.utils.time_constants import (
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
)


class _TemporalScaleFunctor(FunctorBase):
    """Minimal functor used for execution-group resolution tests."""

    def __init__(
        self,
        name: str,
        temporal_scale: TemporalScale,
        execution_group: Optional[str] = None,
        characteristic_time_days: Optional[float] = None,
    ) -> None:
        self._output_name = f"diagnostics.test.temporal_scale.{name}"
        metadata = FunctorMetadata(
            name=name,
            description="Temporal scale mapping test functor",
            references=["RELIC simulator execution-group policy"],
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            temporal_scale=temporal_scale,
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.DIAGNOSTIC,
            output_variables=[self._output_name],
        )
        super().__init__(name=name, config={}, metadata=metadata)
        if execution_group is not None:
            self.execution_group = execution_group
        if characteristic_time_days is not None:
            self.characteristic_time_days = characteristic_time_days

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, Any]:
        return {
            self._output_name: self.wrap_update(
                torch.tensor(1.0, device=self.device),
                mode=UpdateMode.REPLACE,
            )
        }


def test_temporal_scale_default_mapping_to_execution_group() -> None:
    """TemporalScale should map to fast/medium/slow groups via characteristic time."""
    simulator = RELICSimulator({"time_step": 60.0})

    second_functor = _TemporalScaleFunctor("ts_second", TemporalScale.SECOND)
    day_functor = _TemporalScaleFunctor("ts_day", TemporalScale.DAY)
    month_functor = _TemporalScaleFunctor("ts_month", TemporalScale.MONTH)
    static_functor = _TemporalScaleFunctor("ts_static", TemporalScale.STATIC)

    assert simulator._determine_execution_group(second_functor) == "fast"
    assert simulator._determine_execution_group(day_functor) == "fast"
    assert simulator._determine_execution_group(month_functor) == "slow"
    assert simulator._determine_execution_group(static_functor) == "fast"


def test_execution_group_attribute_overrides_temporal_scale() -> None:
    """Explicit execution_group must override metadata-derived temporal scale."""
    simulator = RELICSimulator({"time_step": 60.0})
    functor = _TemporalScaleFunctor(
        name="ts_override_group",
        temporal_scale=TemporalScale.YEAR,
        execution_group="medium",
    )
    assert simulator._determine_execution_group(functor) == "medium"


def test_characteristic_time_overrides_temporal_scale() -> None:
    """Explicit characteristic_time_days must override metadata temporal scale."""
    simulator = RELICSimulator({"time_step": 60.0})
    functor = _TemporalScaleFunctor(
        name="ts_override_char_time",
        temporal_scale=TemporalScale.SECOND,
        characteristic_time_days=20.0,
    )
    assert simulator._determine_execution_group(functor) == "slow"


def test_fixed_temporal_scales_use_time_constant_ssot() -> None:
    assert TemporalScale.SECOND.to_characteristic_time_days() == pytest.approx(
        1.0 / float(SECONDS_PER_DAY)
    )
    assert TemporalScale.MINUTE.to_characteristic_time_days() == pytest.approx(
        float(SECONDS_PER_MINUTE) / float(SECONDS_PER_DAY)
    )
    assert TemporalScale.HOUR.to_characteristic_time_days() == pytest.approx(
        float(SECONDS_PER_HOUR) / float(SECONDS_PER_DAY)
    )
    assert TemporalScale.DAY.to_characteristic_time_days() == 1.0
    assert TemporalScale.STATIC.to_characteristic_time_days() == 0.0
    assert TemporalScale.STEADY_STATE.to_characteristic_time_days() == 0.0


@pytest.mark.parametrize(
    "scale",
    [
        TemporalScale.SUB_DAILY,
        TemporalScale.WEEK,
        TemporalScale.MONTH,
        TemporalScale.SEASON,
        TemporalScale.YEAR,
    ],
)
def test_context_dependent_temporal_scales_fail_without_period_days(scale) -> None:
    with pytest.raises(ValueError, match="period_days.*required"):
        scale.to_characteristic_time_days()


@pytest.mark.parametrize(
    ("scale", "expected_group"),
    [
        (TemporalScale.SUB_DAILY, "fast"),
        (TemporalScale.WEEK, "medium"),
        (TemporalScale.MONTH, "slow"),
        (TemporalScale.SEASON, "slow"),
        (TemporalScale.YEAR, "slow"),
    ],
)
def test_scheduler_preserves_context_dependent_scale_order(
    scale,
    expected_group,
) -> None:
    simulator = RELICSimulator({"time_step": 60.0})
    functor = _TemporalScaleFunctor(f"ts_{scale.value}", scale)

    assert simulator._determine_execution_group(functor) == expected_group


def test_unknown_temporal_scale_has_no_characteristic_default() -> None:
    with pytest.raises(ValueError, match="UNKNOWN"):
        TemporalScale.UNKNOWN.to_characteristic_time_days()


@pytest.mark.parametrize(
    ("frequency", "start", "end", "scale", "noleap_days", "day360_days"),
    [
        ("M", "2001-02-01", "2001-03-01", TemporalScale.MONTH, 28.0, 30.0),
        (
            "SEASON-MAM",
            "2001-03-01",
            "2001-06-01",
            TemporalScale.SEASON,
            92.0,
            90.0,
        ),
        ("Y", "2001-01-01", "2002-01-01", TemporalScale.YEAR, 365.0, 360.0),
    ],
)
def test_calendar_owner_supplies_dynamic_characteristic_period_days(
    frequency,
    start,
    end,
    scale,
    noleap_days,
    day360_days,
) -> None:
    indexer = PeriodIndexer(
        device=torch.device("cpu"),
        parameter_manager=get_global_manager(),
    )

    resolved = []
    for calendar in ("noleap", "360_day"):
        coordinate = indexer.create_period_index(
            start,
            end,
            frequency,
            calendar=calendar,
            units="days since 2001-01-01",
        )
        period_days = float((coordinate.bounds[-1, 1] - coordinate.bounds[0, 0]).item())
        resolved.append(scale.to_characteristic_time_days(period_days=period_days))

    assert resolved == [noleap_days, day360_days]
    assert resolved[0] != resolved[1]


def test_characteristic_time_source_has_no_literal_mapping_or_fallback() -> None:
    source = inspect.getsource(TemporalScale.to_characteristic_time_days)

    assert "mapping =" not in source
    for stale in ("0.001", "0.01", "0.05", "0.5", "30.0", "90.0", "365.0"):
        assert stale not in source
    assert ".get(" not in source


def test_execution_group_thresholds_come_from_the_parameter_database() -> None:
    """The fast/medium characteristic-time thresholds are YAML parameters, not Python literals (D78)."""
    import relic.core.simulator.dependency_ops as dependency_ops

    manager = get_global_manager()
    fast_max = float(
        manager.get("computation.scheduling.fast_group_max_characteristic_days", units="day")
    )
    medium_max = float(
        manager.get("computation.scheduling.medium_group_max_characteristic_days", units="day")
    )
    assert 0.0 < fast_max < medium_max

    simulator = RELICSimulator({"time_step": 60.0})
    cases = [
        (fast_max, "fast"),
        (fast_max * (1.0 + 1.0e-6), "medium"),
        (medium_max, "medium"),
        (medium_max * (1.0 + 1.0e-6), "slow"),
    ]
    for index, (characteristic_days, expected) in enumerate(cases):
        functor = _TemporalScaleFunctor(
            name=f"ts_threshold_{expected}_{index}",
            temporal_scale=TemporalScale.SECOND,
            characteristic_time_days=characteristic_days,
        )
        assert simulator._determine_execution_group(functor) == expected, characteristic_days

    source = inspect.getsource(dependency_ops)
    assert "<= 1.0" not in source
    assert "<= 10.0" not in source
    # The namespace and both key names must be referenced by the resolver
    # (the key is assembled as f"{namespace}.{name}").
    assert 'computation.scheduling' in source
    assert "fast_group_max_characteristic_days" in source
    assert "medium_group_max_characteristic_days" in source
    assert dependency_ops.resolve_scheduling_thresholds() == (fast_max, medium_max)


def test_invalid_execution_group_attribute_raises() -> None:
    simulator = RELICSimulator({"time_step": 60.0})
    functor = _TemporalScaleFunctor("attr_invalid", TemporalScale.UNKNOWN, execution_group="fastest")

    with pytest.raises(ValueError, match="fastest"):
        simulator._determine_execution_group(functor)


@pytest.mark.parametrize("value", ["soon", float("nan"), float("inf"), -1.0, True])
def test_characteristic_time_that_is_not_a_finite_nonnegative_number_raises(value) -> None:
    simulator = RELICSimulator({"time_step": 60.0})
    functor = _TemporalScaleFunctor("characteristic_invalid", TemporalScale.UNKNOWN)
    functor.characteristic_time_days = value

    with pytest.raises(ValueError, match="characteristic_time_days"):
        simulator._determine_execution_group(functor)


def test_unparseable_temporal_scale_raises() -> None:
    simulator = RELICSimulator({"time_step": 60.0})
    functor = _TemporalScaleFunctor("scale_invalid", TemporalScale.UNKNOWN)
    functor.metadata.temporal_scale = "fortnightly"

    with pytest.raises(ValueError, match="fortnightly"):
        simulator._determine_execution_group(functor)


@pytest.mark.parametrize("process_groups", [{"entry_invalid": "daily"}, ["entry_invalid"]])
def test_invalid_process_groups_configuration_raises(process_groups) -> None:
    simulator = RELICSimulator({"time_step": 60.0, "process_groups": process_groups})
    functor = _TemporalScaleFunctor("entry_invalid", TemporalScale.UNKNOWN)

    with pytest.raises(ValueError, match="process_groups"):
        simulator._determine_execution_group(functor)


def test_process_groups_entry_conflicting_with_an_earlier_rule_raises() -> None:
    simulator = RELICSimulator({"time_step": 60.0, "process_groups": {"entry_conflict": "slow"}})
    functor = _TemporalScaleFunctor("entry_conflict", TemporalScale.SECOND)

    with pytest.raises(ValueError, match="temporal_scale") as error:
        simulator._determine_execution_group(functor)

    message = str(error.value)
    assert "'fast'" in message and "'slow'" in message
    assert "remove" in message and "register_functor" in message


def test_process_groups_entry_agreeing_with_an_earlier_rule_is_accepted() -> None:
    simulator = RELICSimulator({"time_step": 60.0, "process_groups": {"entry_agree": "fast"}})
    functor = _TemporalScaleFunctor("entry_agree", TemporalScale.SECOND)

    assert simulator._determine_execution_group(functor) == "fast"


def test_process_groups_fills_in_only_for_functors_without_a_time_declaration() -> None:
    simulator = RELICSimulator({"time_step": 60.0, "process_groups": {"entry_fill": "medium"}})

    assert simulator._determine_execution_group(
        _TemporalScaleFunctor("entry_fill", TemporalScale.UNKNOWN)
    ) == "medium"
    assert simulator._determine_execution_group(
        _TemporalScaleFunctor("undeclared_default", TemporalScale.UNKNOWN)
    ) == "fast"


@pytest.mark.parametrize("declared", ["fortnightly", "hourly", "secnd", "", "   "])
def test_from_value_rejects_a_string_that_names_no_scale(declared) -> None:
    """A misspelled or blank declaration is an error, not a silent UNKNOWN."""
    with pytest.raises(ValueError, match="temporal scale") as error:
        TemporalScale.from_value(declared)

    message = str(error.value)
    assert repr(declared) in message
    for member in TemporalScale:
        assert repr(member.value) in message


@pytest.mark.parametrize("declared", [5, 1.0, object()])
def test_from_value_rejects_a_value_that_is_not_a_string(declared) -> None:
    with pytest.raises(TypeError, match="TemporalScale"):
        TemporalScale.from_value(declared)


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (None, TemporalScale.UNKNOWN),
        (TemporalScale.UNKNOWN, TemporalScale.UNKNOWN),
        ("unknown", TemporalScale.UNKNOWN),
        (" Unknown ", TemporalScale.UNKNOWN),
        ("second", TemporalScale.SECOND),
        ("DAY", TemporalScale.DAY),
        (" sub_daily ", TemporalScale.SUB_DAILY),
        (TemporalScale.MONTH, TemporalScale.MONTH),
    ],
)
def test_from_value_accepts_members_their_values_and_the_undeclared_forms(declared, expected) -> None:
    assert TemporalScale.from_value(declared) is expected


@pytest.mark.parametrize("declared", ["daily", "fast", "none", " Daily "])
def test_from_value_rejects_the_removed_aliases(declared) -> None:
    """Only member values are spellings: "fast" is an execution group, not a scale."""
    with pytest.raises(ValueError, match="temporal scale") as error:
        TemporalScale.from_value(declared)

    message = str(error.value)
    assert "alias" not in message
    assert [member.value for member in TemporalScale].__repr__() in message


def test_functor_metadata_rejects_a_misspelled_temporal_scale() -> None:
    with pytest.raises(ValueError, match="hourly"):
        FunctorMetadata(
            name="misspelled_scale",
            description="Temporal scale declaration test",
            references=["RELIC TemporalScale contract"],
            temporal_scale="hourly",
        )
