"""
Unit test ensuring CalendarHandler delegates to TimeWeightGenerator for weight/coverage.
"""

import inspect

import pytest
import torch

import relic.coupling.scales.temporal.calendar as calendar_module
import relic.coupling.scales.temporal.core as temporal_core
from relic.core import device_policy
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.coupling.scales.temporal.core import (
    TimeCoordinate,
    TimeWeightGenerator,
    ValidationError,
)
from relic.coupling.scales.temporal.types import TemporalSemantics


def test_calendar_handler_uses_time_weight_generator():
    device = torch.device("cpu")
    handler = CalendarHandler(device=device)
    generator = TimeWeightGenerator(device=device)

    # Simple daily to weekly mapping: 7 daily intervals to 1 weekly interval
    days = torch.arange(7, dtype=torch.float64)
    daily_bounds = torch.stack([days, days + 1.0], dim=1)
    weekly_bounds = torch.tensor([[0.0, 7.0]], dtype=torch.float64)
    daily_time = days + 0.5
    weekly_time = torch.tensor([3.5], dtype=torch.float64)

    # Expected weights via the canonical overlap plan
    source_coord = TimeCoordinate(
        values=daily_time,
        bounds=daily_bounds,
        calendar="gregorian",
        units="days since 2000-01-01",
    )
    target_coord = TimeCoordinate(
        values=weekly_time,
        bounds=weekly_bounds,
        calendar="gregorian",
        units="days since 2000-01-01",
    )
    plan = generator.build_overlap_plan(source_coord, target_coord)

    weights, coverage = handler.compute_time_weights(
        daily_time, daily_bounds, weekly_time, weekly_bounds
    )

    torch.testing.assert_close(weights, plan.target_fractions, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        coverage,
        plan.geometric_target_coverage,
        rtol=0.0,
        atol=0.0,
    )


def test_calendar_weight_helpers_reject_an_unconsumed_coverage_threshold():
    """Weights and coverage do not depend on a threshold, so none is accepted."""
    handler = CalendarHandler(device=torch.device("cpu"))
    days = torch.arange(2, dtype=torch.float64)
    bounds = torch.stack([days, days + 1.0], dim=1)
    target_bounds = torch.tensor([[0.0, 2.0]], dtype=torch.float64)

    with pytest.raises(TypeError, match="coverage_threshold"):
        handler.compute_time_weights(  # type: ignore[call-arg]
            days + 0.5,
            bounds,
            torch.tensor([1.0], dtype=torch.float64),
            target_bounds,
            coverage_threshold=0.5,
        )
    with pytest.raises(TypeError, match="coverage_threshold"):
        calendar_module.compute_time_weights(  # type: ignore[call-arg]
            days + 0.5,
            bounds,
            torch.tensor([1.0], dtype=torch.float64),
            target_bounds,
            coverage_threshold=0.5,
        )


def test_time_weight_generator_rejects_mismatched_calendars():
    """Weights must not combine coordinate bounds from different calendars."""
    device = torch.device("cpu")
    generator = TimeWeightGenerator(device=device)
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    values = torch.tensor([0.5], dtype=torch.float64)
    source = TimeCoordinate(
        values=values,
        bounds=bounds,
        calendar="gregorian",
        units="days since 2000-01-01",
    )
    target = TimeCoordinate(
        values=values,
        bounds=bounds,
        calendar="noleap",
        units="days since 2000-01-01",
    )

    with pytest.raises(ValidationError, match="Different calendars"):
        generator.build_overlap_plan(source, target)


def test_time_weight_generator_has_no_string_role_weight_selector():
    """Weight views come from the plan, not from an intensive/extensive string."""
    assert not hasattr(TimeWeightGenerator, "compute_weights")


def test_time_weight_generator_has_no_uncalled_threshold_validator():
    """Threshold acceptance is owned by TemporalAggregator, not the weight owner."""
    assert not hasattr(TimeWeightGenerator, "validate_coverage")
    assert "warnings.warn" not in inspect.getsource(TimeWeightGenerator)


def _coordinate(
    bounds: torch.Tensor,
    *,
    units: str = "days since 2000-01-01",
    calendar: str = "noleap",
) -> TimeCoordinate:
    return TimeCoordinate(
        values=torch.mean(bounds, dim=1),
        bounds=bounds,
        units=units,
        calendar=calendar,
    )


def _unit_contract(
    source_units: str,
    target_units: str,
    measure_units: str,
    integration_time_unit: str | None = None,
):
    return temporal_core.TemporalUnitContract(
        source_units=source_units,
        target_units=target_units,
        measure_units=measure_units,
        integration_time_unit=integration_time_unit,
    )


def _nonuniform_plan():
    source_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 3.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    target_bounds = torch.tensor(
        [[0.0, 2.0], [2.0, 4.0]],
        dtype=torch.float64,
    )
    generator = TimeWeightGenerator(device=torch.device("cpu"))
    return generator.build_overlap_plan(
        _coordinate(source_bounds),
        _coordinate(target_bounds),
    )


def test_overlap_plan_exposes_hand_computed_nonuniform_geometry():
    plan = _nonuniform_plan()

    torch.testing.assert_close(
        plan.overlap,
        torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        plan.source_fractions,
        torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        plan.target_fractions,
        torch.tensor(
            [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        plan.geometric_target_coverage,
        torch.ones(2, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        plan.geometric_source_coverage,
        torch.ones(3, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_overlap_plan_rejects_overlapping_source_intervals_instead_of_clamping():
    generator = TimeWeightGenerator(device=torch.device("cpu"))
    source = _coordinate(
        torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float64)
    )
    target = _coordinate(torch.tensor([[0.0, 3.0]], dtype=torch.float64))

    with pytest.raises(ValidationError, match="source.*overlap"):
        generator.build_overlap_plan(source, target)


def test_overlap_plan_rejects_different_time_references_before_mapping():
    generator = TimeWeightGenerator(device=torch.device("cpu"))
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    source = _coordinate(bounds, units="days since 2000-01-01")
    target = _coordinate(bounds, units="days since 2000-01-02")

    with pytest.raises(ValidationError, match="reference"):
        generator.build_overlap_plan(source, target)


def test_overlap_plan_converts_compatible_cf_time_units_through_unit_owner():
    generator = TimeWeightGenerator(device=torch.device("cpu"))
    source = _coordinate(
        torch.tensor([[0.0, 3600.0], [3600.0, 7200.0]], dtype=torch.float64),
        units="seconds since 2000-01-01",
    )
    target = _coordinate(
        torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        units="hours since 2000-01-01",
    )

    plan = generator.build_overlap_plan(source, target)

    torch.testing.assert_close(
        plan.target_bounds,
        torch.tensor([[0.0, 7200.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        plan.overlap,
        torch.tensor([[3600.0], [3600.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert plan.coordinate_unit == "s"
    assert plan.relation is temporal_core.TemporalBoundsRelation.COARSENING


def test_overlap_plan_rejects_unknown_cf_time_unit_even_when_labels_match():
    generator = TimeWeightGenerator(device=torch.device("cpu"))
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    source = _coordinate(bounds, units="bananas")
    target = _coordinate(bounds, units="bananas")

    with pytest.raises(ValidationError, match="CF time units"):
        generator.build_overlap_plan(source, target)


@pytest.mark.parametrize(
    ("source_bounds", "target_bounds", "expected_relation"),
    [
        (
            [[0.0, 1.0], [1.0, 2.0]],
            [[0.0, 1.0], [1.0, 2.0]],
            "identity",
        ),
        (
            [[0.0, 1.0], [1.0, 2.0]],
            [[0.0, 2.0]],
            "coarsening",
        ),
        (
            [[0.0, 2.0]],
            [[0.0, 1.0], [1.0, 2.0]],
            "refinement",
        ),
        (
            [[0.0, 1.5], [1.5, 2.0]],
            [[0.0, 1.0], [1.0, 2.0]],
            "non_nested",
        ),
    ],
)
def test_overlap_plan_classifies_relation_from_actual_bounds(
    source_bounds,
    target_bounds,
    expected_relation,
):
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(torch.tensor(source_bounds, dtype=torch.float64)),
        _coordinate(torch.tensor(target_bounds, dtype=torch.float64)),
    )

    assert plan.relation.value == expected_relation


def test_temporal_kernel_matches_piecewise_constant_mean_and_total_oracles():
    plan = _nonuniform_plan()

    mean_result = temporal_core.apply_temporal_kernel(
        plan,
        torch.tensor([2.0, 5.0, 8.0], dtype=torch.float64),
        semantics=TemporalSemantics.INTERVAL_MEAN,
        unit_contract=_unit_contract("K", "K", "K day", "day"),
    )
    total_result = temporal_core.apply_temporal_kernel(
        plan,
        torch.tensor([5.0, 10.0, 5.0], dtype=torch.float64),
        semantics=TemporalSemantics.INTERVAL_ACCUMULATION,
        unit_contract=_unit_contract("kg", "kg", "kg"),
    )

    torch.testing.assert_close(
        mean_result.values,
        torch.tensor([3.5, 6.5], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        total_result.values,
        torch.tensor([10.0, 10.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(mean_result.complete_coverage, torch.tensor([True, True]))


def test_rate_kernel_keeps_per_element_valid_coverage_separate():
    source_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    target_bounds = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(source_bounds),
        _coordinate(target_bounds),
    )
    rates = torch.tensor([[2.0, 2.0], [4.0, 4.0]], dtype=torch.float64)
    valid_mask = torch.tensor([[True, True], [True, False]])

    result = temporal_core.apply_temporal_kernel(
        plan,
        rates,
        semantics=TemporalSemantics.RATE,
        valid_mask=valid_mask,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
    )

    torch.testing.assert_close(
        result.values,
        torch.tensor([[3.0, 2.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.observed_measure,
        torch.tensor([[6.0, 2.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.observed_duration,
        torch.tensor([[2.0, 1.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.valid_coverage,
        torch.tensor([[1.0, 0.5]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        result.complete_coverage,
        torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(
        result.gap_diagnostic.geometric_gap_fraction,
        torch.zeros((1, 2), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.gap_diagnostic.missing_data_gap_fraction,
        torch.tensor([[0.0, 0.5]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.gap_diagnostic.total_gap_fraction,
        torch.tensor([[0.0, 0.5]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert result.gap_diagnostic.kind == "gap"
    assert result.gap_diagnostic.imputation == "none"
    assert result.gap_diagnostic.units == "day"


def test_rate_kernel_uses_unit_owner_for_hour_rate_on_second_bounds():
    bounds = torch.tensor([[0.0, 7200.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds, units="seconds since 2000-01-01"),
        _coordinate(bounds, units="seconds since 2000-01-01"),
    )

    result = temporal_core.apply_temporal_kernel(
        plan,
        torch.tensor([1.0], dtype=torch.float64),
        semantics=TemporalSemantics.RATE,
        output_semantics=TemporalSemantics.INTERVAL_ACCUMULATION,
        unit_contract=_unit_contract("kg h-1", "kg", "kg", "hour"),
    )

    torch.testing.assert_close(
        result.values,
        torch.tensor([2.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_mapping_residual_detects_known_injected_amount():
    plan = _nonuniform_plan()
    source_rate = torch.full((3,), 5.0, dtype=torch.float64)
    target_rate = torch.tensor([5.0, 5.0005], dtype=torch.float64)

    residual = temporal_core.compute_temporal_mapping_residual(
        plan,
        source_rate,
        target_rate,
        semantics=TemporalSemantics.RATE,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
        law_id="temporal.mass.mapping",
        q_ref=1.0,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )
    roundoff_bound = torch.finfo(torch.float64).eps * 20.0 * 8.0

    torch.testing.assert_close(
        residual.residual,
        torch.tensor(0.001, dtype=torch.float64),
        rtol=0.0,
        atol=roundoff_bound,
    )
    torch.testing.assert_close(
        residual.source_activity,
        torch.tensor(20.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        residual.relative_residual,
        torch.tensor(5.0e-5, dtype=torch.float64),
        rtol=0.0,
        atol=roundoff_bound / 20.0,
    )


def test_exact_state_boundary_mapping_rejects_unbound_interpolation():
    source = TimeCoordinate(
        values=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        units="days since 2000-01-01",
        calendar="noleap",
    )
    exact_target = TimeCoordinate(
        values=torch.tensor([1.0], dtype=torch.float64),
        units=source.units,
        calendar=source.calendar,
    )
    unbound_target = TimeCoordinate(
        values=torch.tensor([0.5], dtype=torch.float64),
        units=source.units,
        calendar=source.calendar,
    )
    state = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)

    exact = temporal_core.map_state_boundaries(source, exact_target, state)
    torch.testing.assert_close(
        exact,
        torch.tensor([20.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValidationError, match="interpolation"):
        temporal_core.map_state_boundaries(source, unbound_target, state)


def test_temporal_kernel_preserves_non_time_axes_and_autograd():
    source_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    target_bounds = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(source_bounds),
        _coordinate(target_bounds),
    )
    rates = torch.tensor(
        [[2.0, 4.0], [6.0, 8.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = temporal_core.apply_temporal_kernel(
        plan,
        rates,
        semantics=TemporalSemantics.RATE,
        time_dim=1,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
    )

    assert result.values.shape == (2, 1)
    torch.testing.assert_close(
        result.values,
        torch.tensor([[3.0], [7.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    result.values.sum().backward()
    torch.testing.assert_close(
        rates.grad,
        torch.full_like(rates, 0.5),
        rtol=0.0,
        atol=0.0,
    )


def test_mapping_residual_uses_absolute_activity_under_signed_cancellation():
    source_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    target_bounds = source_bounds.clone()
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(source_bounds),
        _coordinate(target_bounds),
    )

    result = temporal_core.compute_temporal_mapping_residual(
        plan,
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        semantics=TemporalSemantics.RATE,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
        law_id="temporal.mass.mapping",
        q_ref=0.1,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )

    assert result.residual.item() == pytest.approx(0.0, abs=0.0)
    assert result.source_activity.item() == pytest.approx(2.0, abs=0.0)
    assert result.relative_residual.item() == pytest.approx(0.0, abs=0.0)


def test_mapping_residual_compares_only_accepted_target_support():
    plan = _nonuniform_plan()
    result = temporal_core.compute_temporal_mapping_residual(
        plan,
        torch.full((3,), 5.0, dtype=torch.float64),
        torch.tensor([5.0, 999.0], dtype=torch.float64),
        semantics=TemporalSemantics.RATE,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
        law_id="temporal.mass.mapping",
        accepted_target_mask=torch.tensor([True, False]),
        q_ref=1.0,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )

    assert result.residual.item() == pytest.approx(0.0, abs=0.0)
    assert result.source_activity.item() == pytest.approx(10.0, abs=0.0)


@pytest.mark.parametrize(
    "bad_bounds",
    [
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, float("nan")]], dtype=torch.float64),
    ],
)
def test_overlap_plan_rejects_nonpositive_or_nonfinite_intervals(bad_bounds):
    generator = TimeWeightGenerator(device=torch.device("cpu"))
    target = _coordinate(torch.tensor([[0.0, 1.0]], dtype=torch.float64))

    with pytest.raises(ValidationError):
        generator.build_overlap_plan(
            TimeCoordinate(
                values=torch.tensor([0.0], dtype=torch.float64),
                bounds=bad_bounds,
                units=target.units,
                calendar=target.calendar,
            ),
            target,
        )


def test_overlap_plan_snapshots_bounds_against_caller_mutation():
    source_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    target_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(source_bounds),
        _coordinate(target_bounds),
    )

    source_bounds[0, 0] = -100.0
    target_bounds[0, 1] = 100.0

    torch.testing.assert_close(
        plan.source_bounds,
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        plan.target_bounds,
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_overlap_plan_rejects_one_ulp_overlap_at_large_coordinates():
    end = torch.tensor(1.0e12 + 1.0, dtype=torch.float64)
    second_start = torch.nextafter(
        end,
        torch.tensor(float("-inf"), dtype=torch.float64),
    ).item()
    source_bounds = torch.tensor(
        [[1.0e12, end.item()], [second_start, 1.0e12 + 2.0]],
        dtype=torch.float64,
    )
    target_bounds = torch.tensor(
        [[1.0e12, 1.0e12 + 2.0]],
        dtype=torch.float64,
    )

    with pytest.raises(ValidationError, match="source.*overlap"):
        TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
            _coordinate(source_bounds),
            _coordinate(target_bounds),
        )


def test_mapping_residual_ignores_nonfinite_rejected_target_values():
    plan = _nonuniform_plan()
    result = temporal_core.compute_temporal_mapping_residual(
        plan,
        torch.full((3,), 5.0, dtype=torch.float64),
        torch.tensor([5.0, float("nan")], dtype=torch.float64),
        semantics=TemporalSemantics.RATE,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
        law_id="temporal.mass.mapping",
        accepted_target_mask=torch.tensor([True, False]),
        q_ref=1.0,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )

    assert result.residual.item() == pytest.approx(0.0, abs=0.0)
    assert result.source_activity.item() == pytest.approx(10.0, abs=0.0)


def test_rate_to_accumulation_kernel_and_residual_form_closed_unit_chain():
    bounds = torch.tensor([[0.0, 7200.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds, units="seconds since 2000-01-01"),
        _coordinate(bounds, units="seconds since 2000-01-01"),
    )
    contract = _unit_contract("kg h-1", "kg", "kg", "hour")
    source_rate = torch.tensor([1.0], dtype=torch.float64)
    mapped = temporal_core.apply_temporal_kernel(
        plan,
        source_rate,
        semantics=TemporalSemantics.RATE,
        output_semantics=TemporalSemantics.INTERVAL_ACCUMULATION,
        unit_contract=contract,
    )

    residual = temporal_core.compute_temporal_mapping_residual(
        plan,
        source_rate,
        mapped.values,
        semantics=TemporalSemantics.RATE,
        output_semantics=TemporalSemantics.INTERVAL_ACCUMULATION,
        unit_contract=contract,
        law_id="temporal.mass.mapping",
        q_ref=torch.tensor(0.5, dtype=torch.float64),
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )

    torch.testing.assert_close(
        mapped.values,
        torch.tensor([2.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        residual.residual,
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert bool(residual.checked)
    assert residual.kind == "mapping"
    assert residual.law_id == "temporal.mass.mapping"
    assert residual.units == "kg"
    assert residual.support == "accepted_target_intervals_x_non_time_support"
    assert residual.reduction_axes == (0, 1)
    assert residual.imputation == "none"
    assert residual.q_ref_provenance == "relic.parameters.test.temporal_q_ref"


@pytest.mark.parametrize("scenario", ["all_false", "zero_overlap", "missing_source"])
def test_mapping_residual_rejects_vacuous_accepted_support(scenario):
    source_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    target_bounds = (
        torch.tensor([[2.0, 3.0]], dtype=torch.float64)
        if scenario == "zero_overlap"
        else torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    )
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(source_bounds),
        _coordinate(target_bounds),
    )
    accepted = (
        torch.tensor([False]) if scenario == "all_false" else None
    )
    source_valid = (
        torch.tensor([False]) if scenario == "missing_source" else None
    )

    with pytest.raises(ValidationError, match="accepted support"):
        temporal_core.compute_temporal_mapping_residual(
            plan,
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            semantics=TemporalSemantics.RATE,
            unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
            law_id="temporal.mass.mapping",
            source_valid_mask=source_valid,
            accepted_target_mask=accepted,
            q_ref=1.0,
            q_ref_provenance="relic.parameters.test.temporal_q_ref",
        )


def test_mapping_residual_skips_explicitly_excluded_fully_masked_outputs():
    """A declared fully masked output element is reported unchecked, not raised."""
    plan = _nonuniform_plan()
    source = torch.tensor(
        [[5.0, 7.0], [5.0, 7.0], [5.0, 7.0]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[5.0, float("nan")], [5.0, float("nan")]],
        dtype=torch.float64,
    )
    accepted = torch.tensor([[True, False], [True, False]])

    result = temporal_core.compute_temporal_mapping_residual(
        plan,
        source,
        target,
        semantics=TemporalSemantics.RATE,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
        law_id="temporal.mass.mapping",
        accepted_target_mask=accepted,
        excluded_output_mask=torch.tensor([False, True]),
        q_ref=1.0,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )

    assert result.checked.tolist() == [True, False]
    assert result.residual[0].item() == pytest.approx(0.0, abs=0.0)
    assert torch.isnan(result.residual[1])
    assert result.relative_residual[0].item() == pytest.approx(0.0, abs=0.0)
    assert torch.isnan(result.relative_residual[1])
    # Source activity on the accepted support: 5 kg day-1 over 4 days = 20 kg;
    # the excluded element has an empty accepted support.
    assert result.source_activity.tolist() == [20.0, 0.0]
    assert result.accepted_target_mask.tolist() == accepted.tolist()


def test_mapping_residual_rejects_exclusion_of_outputs_with_accepted_support():
    """Exclusion cannot hide a residual on an output that has accepted targets."""
    plan = _nonuniform_plan()

    with pytest.raises(ValidationError, match="excluded_output_mask"):
        temporal_core.compute_temporal_mapping_residual(
            plan,
            torch.full((3, 2), 5.0, dtype=torch.float64),
            torch.full((2, 2), 5.0, dtype=torch.float64),
            semantics=TemporalSemantics.RATE,
            unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
            law_id="temporal.mass.mapping",
            excluded_output_mask=torch.tensor([True, False]),
            q_ref=1.0,
            q_ref_provenance="relic.parameters.test.temporal_q_ref",
        )

    with pytest.raises(ValidationError, match="excluded_output_mask"):
        temporal_core.compute_temporal_mapping_residual(
            plan,
            torch.full((3, 2), 5.0, dtype=torch.float64),
            torch.full((2, 2), 5.0, dtype=torch.float64),
            semantics=TemporalSemantics.RATE,
            unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
            law_id="temporal.mass.mapping",
            excluded_output_mask=torch.tensor([0, 1]),
            q_ref=1.0,
            q_ref_provenance="relic.parameters.test.temporal_q_ref",
        )


def test_zero_support_mean_has_nan_value_but_finite_zero_gradient():
    source_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    target_bounds = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(source_bounds),
        _coordinate(target_bounds),
    )
    data = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

    result = temporal_core.apply_temporal_kernel(
        plan,
        data,
        semantics=TemporalSemantics.INTERVAL_MEAN,
        unit_contract=_unit_contract("K", "K", "K day", "day"),
    )
    assert torch.isnan(result.values).all()
    torch.nan_to_num(result.values).sum().backward()
    torch.testing.assert_close(
        data.grad,
        torch.zeros_like(data),
        rtol=0.0,
        atol=0.0,
    )


def test_state_boundary_mapping_uses_exact_values_and_ordered_targets():
    source = TimeCoordinate(
        values=torch.tensor([1.0e15, 1.0e15 + 2.0], dtype=torch.float64),
        units="seconds since 2000-01-01",
        calendar="noleap",
    )
    data = torch.tensor([10.0, 20.0], dtype=torch.float64)
    near_not_exact = TimeCoordinate(
        values=torch.tensor([1.0e15 + 1.0], dtype=torch.float64),
        units=source.units,
        calendar=source.calendar,
    )
    reversed_target = TimeCoordinate(
        values=torch.tensor([1.0e15 + 2.0, 1.0e15], dtype=torch.float64),
        units=source.units,
        calendar=source.calendar,
    )

    with pytest.raises(ValidationError, match="interpolation"):
        temporal_core.map_state_boundaries(source, near_not_exact, data)
    with pytest.raises(ValidationError, match="target.*increasing"):
        temporal_core.map_state_boundaries(source, reversed_target, data)


def test_mapping_residual_broadcasts_q_ref_over_restored_support_shape():
    bounds = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds),
        _coordinate(bounds),
    )
    source = torch.ones((2, 2, 3), dtype=torch.float64)
    q_ref = torch.full((2, 3), 0.25, dtype=torch.float64)

    result = temporal_core.compute_temporal_mapping_residual(
        plan,
        source,
        source.clone(),
        semantics=TemporalSemantics.RATE,
        unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
        law_id="temporal.mass.mapping",
        q_ref=q_ref,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )

    assert result.relative_residual is not None
    assert result.relative_residual.shape == (2, 3)
    torch.testing.assert_close(
        result.relative_residual,
        torch.zeros((2, 3), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_kernel_rejects_removed_free_numeric_duration_scale():
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds),
        _coordinate(bounds),
    )

    with pytest.raises(TypeError, match="duration_scale"):
        temporal_core.apply_temporal_kernel(
            plan,
            torch.tensor([1.0], dtype=torch.float64),
            semantics=TemporalSemantics.RATE,
            unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
            duration_scale=123.0,
        )


@pytest.mark.parametrize(
    "contract",
    [
        ("bananas", "bananas", "bananas", "day"),
        ("kg day-1", "K", "kg", "day"),
        ("kg", "kg", "K", None),
    ],
)
def test_kernel_rejects_unknown_or_dimensionally_inconsistent_unit_contract(contract):
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds),
        _coordinate(bounds),
    )
    source_units, target_units, measure_units, integration_unit = contract
    semantics = (
        TemporalSemantics.INTERVAL_ACCUMULATION
        if integration_unit is None
        else TemporalSemantics.RATE
    )

    with pytest.raises(ValidationError):
        temporal_core.apply_temporal_kernel(
            plan,
            torch.tensor([1.0], dtype=torch.float64),
            semantics=semantics,
            unit_contract=_unit_contract(
                source_units,
                target_units,
                measure_units,
                integration_unit,
            ),
        )


def test_mapping_residual_requires_parameter_provenance_for_q_ref():
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds),
        _coordinate(bounds),
    )

    with pytest.raises(ValidationError, match="ParameterManager provenance"):
        temporal_core.compute_temporal_mapping_residual(
            plan,
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            semantics=TemporalSemantics.RATE,
            unit_contract=_unit_contract("kg day-1", "kg day-1", "kg", "day"),
            law_id="temporal.mass.mapping",
            q_ref=1.0,
        )


@pytest.mark.skipif(
    not device_policy.is_cuda_available(),
    reason="CUDA is unavailable",
)
def test_unspecified_cuda_generator_materializes_exact_tensor_device_and_matches_cpu():
    cpu_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    cuda_bounds = cpu_bounds.to(torch.device("cuda"))
    cuda_generator = TimeWeightGenerator(device=torch.device("cuda"))
    cuda_plan = cuda_generator.build_overlap_plan(
        _coordinate(cuda_bounds),
        _coordinate(cuda_bounds),
    )
    cpu_plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(cpu_bounds),
        _coordinate(cpu_bounds),
    )

    assert cuda_generator.device == cuda_bounds.device
    torch.testing.assert_close(
        cuda_plan.overlap.cpu(),
        cpu_plan.overlap,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("calendar", "units"),
    [
        ("noleap", "days since 2001-02-31"),
        ("noleap", "days since 2001-04-31"),
        ("noleap", "days since 0000-01-01"),
        ("360_day", "days since 2001-01-31"),
    ],
)
def test_overlap_plan_reuses_calendar_owner_to_reject_invalid_reference_date(
    calendar,
    units,
):
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    source = _coordinate(bounds, units=units, calendar=calendar)
    target = _coordinate(bounds, units=units, calendar=calendar)

    with pytest.raises(ValidationError, match="Year must|Invalid day"):
        TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
            source,
            target,
        )


def test_rate_unit_denominator_cannot_disagree_with_integration_time_unit():
    bounds = torch.tensor([[0.0, 7200.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds, units="seconds since 2000-01-01"),
        _coordinate(bounds, units="seconds since 2000-01-01"),
    )

    with pytest.raises(ValidationError, match="rate.*time unit"):
        temporal_core.apply_temporal_kernel(
            plan,
            torch.tensor([1.0], dtype=torch.float64),
            semantics=TemporalSemantics.RATE,
            output_semantics=TemporalSemantics.INTERVAL_ACCUMULATION,
            unit_contract=_unit_contract("kg s-1", "kg", "kg", "hour"),
        )


def test_interval_accumulation_converts_quantity_scale_through_unit_owner():
    bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds),
        _coordinate(bounds),
    )

    result = temporal_core.apply_temporal_kernel(
        plan,
        torch.tensor([1.0], dtype=torch.float64),
        semantics=TemporalSemantics.INTERVAL_ACCUMULATION,
        unit_contract=_unit_contract("kg", "g", "g"),
    )

    torch.testing.assert_close(
        result.values,
        torch.tensor([1000.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_rate_output_rejects_unexecutable_source_target_scale_change():
    bounds = torch.tensor([[0.0, 3600.0]], dtype=torch.float64)
    plan = TimeWeightGenerator(device=torch.device("cpu")).build_overlap_plan(
        _coordinate(bounds, units="seconds since 2000-01-01"),
        _coordinate(bounds, units="seconds since 2000-01-01"),
    )

    with pytest.raises(ValidationError, match="rate.*unit"):
        temporal_core.apply_temporal_kernel(
            plan,
            torch.tensor([1.0], dtype=torch.float64),
            semantics=TemporalSemantics.RATE,
            unit_contract=_unit_contract("kg s-1", "kg h-1", "kg", "second"),
        )


def test_linear_rate_mms_separates_continuous_and_piecewise_constant_oracles():
    plan = _nonuniform_plan()
    source_interval_means = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float64)
    contract = _unit_contract("kg day-1", "kg day-1", "kg", "day")

    mapped = temporal_core.apply_temporal_kernel(
        plan,
        source_interval_means,
        semantics=TemporalSemantics.RATE,
        unit_contract=contract,
    )
    continuous_target_means = torch.tensor([3.0, 7.0], dtype=torch.float64)
    analytic_integral = torch.tensor(20.0, dtype=torch.float64)

    torch.testing.assert_close(
        mapped.values,
        torch.tensor([3.5, 6.5], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(mapped.values, continuous_target_means)
    torch.testing.assert_close(
        mapped.observed_measure.sum(),
        analytic_integral,
        rtol=0.0,
        atol=0.0,
    )
    residual = temporal_core.compute_temporal_mapping_residual(
        plan,
        source_interval_means,
        mapped.values,
        semantics=TemporalSemantics.RATE,
        unit_contract=contract,
        law_id="temporal.mass.mapping.linear_mms",
        q_ref=1.0,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )
    torch.testing.assert_close(
        residual.residual,
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 64.0,
    )


def test_overlap_plan_public_tensors_are_defensive_snapshots():
    plan = _nonuniform_plan()
    expected_overlap = plan.overlap.clone()
    exposed_overlap = plan.overlap
    exposed_overlap.fill_(99.0)

    torch.testing.assert_close(
        plan.overlap,
        expected_overlap,
        rtol=0.0,
        atol=0.0,
    )


def test_kernel_and_residual_public_tensors_are_defensive_snapshots():
    plan = _nonuniform_plan()
    contract = _unit_contract("kg day-1", "kg day-1", "kg", "day")
    source = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float64)
    mapped = temporal_core.apply_temporal_kernel(
        plan,
        source,
        semantics=TemporalSemantics.RATE,
        unit_contract=contract,
    )
    expected_values = mapped.values.clone()
    expected_gap = mapped.gap_diagnostic.total_gap_fraction.clone()
    mapped.values.fill_(88.0)
    mapped.gap_diagnostic.total_gap_fraction.fill_(77.0)

    torch.testing.assert_close(mapped.values, expected_values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        mapped.gap_diagnostic.total_gap_fraction,
        expected_gap,
        rtol=0.0,
        atol=0.0,
    )

    residual = temporal_core.compute_temporal_mapping_residual(
        plan,
        source,
        mapped.values,
        semantics=TemporalSemantics.RATE,
        unit_contract=contract,
        law_id="temporal.mass.mapping.snapshot",
        q_ref=1.0,
        q_ref_provenance="relic.parameters.test.temporal_q_ref",
    )
    expected_residual = residual.residual.clone()
    residual.residual.fill_(66.0)
    torch.testing.assert_close(
        residual.residual,
        expected_residual,
        rtol=0.0,
        atol=0.0,
    )

