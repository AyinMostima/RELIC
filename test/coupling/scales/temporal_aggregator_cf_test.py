import inspect
from dataclasses import replace
from typing import cast

import pytest
import torch

import relic.coupling.scales.temporal.aggregator as aggregator_module
import relic.coupling.scales.temporal.core as temporal_core
import relic.coupling.scales.temporal.types as temporal_types
from relic.core import device_policy
from relic.core.simulator import RELICSimulator
from relic.coupling.scales.temporal import (
    ConservationLevel,
    ScalingDirection,
    TemporalConfig,
    get_module_status,
)
from relic.coupling.scales.temporal.aggregator import (
    AggregationConfig,
    AggregationMethod,
    TemporalAggregator,
)
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.parameters import get_global_manager


def _regular_bounds(
    count: int,
    width: float = 1.0,
    offset: float = 0.0,
) -> torch.Tensor:
    starts = torch.arange(count, dtype=torch.float64) * width + offset
    return torch.stack((starts, starts + width), dim=1)


def _centers(bounds: torch.Tensor) -> torch.Tensor:
    return torch.mean(bounds, dim=1)


def _temporal_config(
    *,
    source_frequency: str = "hourly",
    target_frequency: str = "daily",
    spatial_basis: SpatialQuantityBasis = SpatialQuantityBasis.INTENSIVE,
    conservation_level: ConservationLevel = ConservationLevel.NONE,
) -> TemporalConfig:
    return TemporalConfig(
        source_frequency=source_frequency,
        target_frequency=target_frequency,
        direction=ScalingDirection.UPSCALE,
        spatial_basis=spatial_basis,
        conservation_level=conservation_level,
        device=torch.device("cpu"),
    )


def _aggregate_with_bounds(
    aggregator: TemporalAggregator,
    data: torch.Tensor,
    source_bounds: torch.Tensor,
    target_bounds: torch.Tensor,
    *,
    temporal_semantics: str = "interval_mean",
    units: str = "K",
    calendar: str = "noleap",
    time_units: str = "hours since 2000-01-01",
    target_units: str | None = None,
    measure_units: str | None = None,
    integration_time_unit: str | None = None,
    q_ref_parameter: str | None = None,
    variance_units: str | None = None,
):
    semantics = temporal_types.TemporalSemantics(temporal_semantics)
    if target_units is None:
        target_units = units
    if semantics in {
        temporal_types.TemporalSemantics.INTERVAL_MEAN,
        temporal_types.TemporalSemantics.RATE,
    }:
        if integration_time_unit is None:
            if semantics is temporal_types.TemporalSemantics.RATE:
                for suffix, denominator_unit in (
                    (" s-1", "s"),
                    (" h-1", "hour"),
                    (" hour-1", "hour"),
                    (" day-1", "day"),
                ):
                    if units.endswith(suffix):
                        integration_time_unit = denominator_unit
                        measure_units = units[: -len(suffix)]
                        break
            else:
                unit_name = time_units.split(" since ", maxsplit=1)[0].strip().lower()
                integration_time_unit = {
                    "seconds": "s",
                    "second": "s",
                    "hours": "hour",
                    "hour": "hour",
                    "days": "day",
                    "day": "day",
                }[unit_name]
        if measure_units is None:
            if semantics is temporal_types.TemporalSemantics.RATE:
                raise AssertionError("rate test helper requires explicit measure units")
            else:
                measure_units = f"{target_units} {integration_time_unit}"
    elif measure_units is None:
        measure_units = target_units
    return aggregator.aggregate(
        data=data,
        source_time=_centers(source_bounds),
        target_time=_centers(target_bounds),
        units=units,
        calendar=calendar,
        time_units=time_units,
        source_bounds=source_bounds,
        target_bounds=target_bounds,
        temporal_semantics=temporal_semantics,
        target_units=target_units,
        measure_units=measure_units,
        integration_time_unit=integration_time_unit,
        q_ref_parameter=q_ref_parameter,
        variance_units=variance_units,
    )


def _install_q_ref_lookup(
    aggregator: TemporalAggregator,
    monkeypatch,
    value: float = 1.0,
) -> str:
    parameter_name = "tests.temporal.mapping_q_ref"
    original_lookup = aggregator.get_parameter_value

    def lookup(name, **kwargs):
        if name == parameter_name:
            assert kwargs["required"] is True
            return value
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(aggregator, "get_parameter_value", lookup)
    return parameter_name


def test_temporal_aggregator_metadata_does_not_claim_dataset_cf_conformance():
    aggregator = TemporalAggregator(_temporal_config())
    temporal_metadata = aggregator.config["temporal_config"]

    assert "cf_compliant" not in temporal_metadata
    assert temporal_metadata["dataset_conformance_checked"] is False
    assert temporal_metadata["cf_validation_scope"] == (
        "time_coordinate_and_cell_method_subset"
    )


def test_temporal_module_status_does_not_claim_dataset_cf_conformance():
    status = get_module_status()

    assert "cf_compliant" not in status
    assert status["dataset_cf_conformance_status"] == "not_implemented"
    assert status["cf_validation_scope"] == (
        "time_coordinate_and_cell_method_subset"
    )


def test_temporal_config_has_no_dataset_cf_conformance_switch():
    assert "cf_compliant" not in TemporalConfig.__dataclass_fields__


def test_temporal_config_rejects_coverage_threshold_owned_by_parameter_database() -> None:
    """The threshold has one owner; a config value would be silently ignored."""
    assert "coverage_threshold" not in TemporalConfig.__dataclass_fields__
    with pytest.raises(TypeError, match="coverage_threshold"):
        TemporalConfig(  # type: ignore[call-arg]
            source_frequency="hourly",
            target_frequency="daily",
            conservation_level=ConservationLevel.NONE,
            coverage_threshold=0.5,
        )
    with pytest.raises(ValueError, match="Unknown temporal setting keys: coverage_threshold"):
        TemporalConfig.from_resolved_settings(
            source_frequency="hourly",
            target_frequency="daily",
            settings={"coverage_threshold": 0.5},
        )


def test_aggregator_rejects_a_configuration_declared_for_downscaling() -> None:
    """TemporalConfig.direction is checked against the operator, not ignored."""
    downscale_config = TemporalConfig(
        source_frequency="daily",
        target_frequency="hourly",
        direction=ScalingDirection.DOWNSCALE,
        conservation_level=ConservationLevel.NONE,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="direction.*upscale"):
        TemporalAggregator(downscale_config)

    aggregator = TemporalAggregator(_temporal_config())
    assert aggregator.config["temporal_config"]["direction"] == "upscale"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("label", "right"), ("closed", "right"), ("percentile_q", 0.9)],
)
def test_aggregation_config_rejects_fields_without_a_consumer(
    field_name: str,
    value: object,
) -> None:
    """Interval label/closure come from explicit bounds; percentile is rejected."""
    assert set(AggregationConfig.__dataclass_fields__) == {"method", "time_dim"}
    with pytest.raises(TypeError, match=field_name):
        AggregationConfig(**{field_name: value})  # type: ignore[arg-type]


def test_temporal_semantics_are_explicit_and_orthogonal_to_extent() -> None:
    semantics = temporal_types.TemporalSemantics

    assert {member.value for member in semantics} == {
        "interval_mean",
        "rate",
        "interval_accumulation",
        "state_at_boundary",
        "point_sample",
        "categorical",
    }

    config = TemporalConfig(
        source_frequency="hourly",
        target_frequency="daily",
        spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
        temporal_semantics=semantics.INTERVAL_MEAN,
        conservation_level=ConservationLevel.NONE,
    )

    assert config.spatial_basis is SpatialQuantityBasis.AREA_DENSITY
    assert config.temporal_semantics is semantics.INTERVAL_MEAN


def test_temporal_config_uses_the_single_spatial_basis_owner() -> None:
    """The quantity classification is SpatialQuantityBasis; no temporal copy exists."""
    import relic.coupling as coupling_package
    import relic.coupling.scales as scales_package
    import relic.coupling.scales.temporal as temporal_package
    assert "variable_type" not in TemporalConfig.__dataclass_fields__
    assert not hasattr(temporal_types, "VariableType")
    for package in (temporal_package, scales_package, coupling_package):
        assert not hasattr(package, "VariableType")
        assert "VariableType" not in package.__all__
    assert coupling_package.SpatialQuantityBasis is SpatialQuantityBasis
    assert scales_package.SpatialQuantityBasis is SpatialQuantityBasis

    default = TemporalConfig(source_frequency="hourly", target_frequency="daily")
    assert default.spatial_basis is SpatialQuantityBasis.INTENSIVE
    parsed = TemporalConfig(
        source_frequency="hourly",
        target_frequency="daily",
        spatial_basis="area_density",
    )
    assert parsed.spatial_basis is SpatialQuantityBasis.AREA_DENSITY
    with pytest.raises(ValueError, match="ambiguous"):
        TemporalConfig(
            source_frequency="hourly",
            target_frequency="daily",
            spatial_basis="extensive",
        )
    resolved = TemporalConfig.from_resolved_settings(
        source_frequency="hourly",
        target_frequency="daily",
        settings={"conservation_level": "none"},
        spatial_basis="cell_total",
    )
    assert resolved.spatial_basis is SpatialQuantityBasis.CELL_TOTAL


def test_categorical_spatial_basis_rejects_temporal_aggregation() -> None:
    aggregator = TemporalAggregator(
        TemporalConfig(
            source_frequency="hourly",
            target_frequency="daily",
            spatial_basis=SpatialQuantityBasis.CATEGORICAL,
            conservation_level=ConservationLevel.NONE,
            device=torch.device("cpu"),
        ),
        AggregationConfig(),
    )
    assert aggregator.config["temporal_config"]["spatial_basis"] == "categorical"

    with pytest.raises(ValueError, match="categorical"):
        _aggregate_with_bounds(
            aggregator,
            torch.ones(2, dtype=torch.float64),
            _regular_bounds(2),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        )


@pytest.mark.parametrize(
    ("level", "compute_residual", "compare_tolerance", "block_on_violation"),
    [
        (ConservationLevel.NONE, False, False, False),
        (ConservationLevel.RELAXED, True, False, False),
        (ConservationLevel.MODERATE, True, True, False),
        (ConservationLevel.STRICT, True, True, True),
    ],
)
def test_conservation_level_exposes_machine_policy(
    level: ConservationLevel,
    compute_residual: bool,
    compare_tolerance: bool,
    block_on_violation: bool,
) -> None:
    assert level.compute_residual is compute_residual
    assert level.compare_tolerance is compare_tolerance
    assert level.block_on_violation is block_on_violation


def test_temporal_config_resolves_exact_functor_override_immutably() -> None:
    overrides = {
        "temporal_aggregator": "strict",
        "temporal_disaggregator": ConservationLevel.RELAXED,
    }
    config = TemporalConfig(
        source_frequency="hourly",
        target_frequency="daily",
        conservation_level=cast(ConservationLevel, "moderate"),
        conservation_level_overrides=overrides,
    )

    overrides["temporal_aggregator"] = "none"

    assert config.resolve_conservation_level("temporal_aggregator") is ConservationLevel.STRICT
    assert config.resolve_conservation_level("temporal_disaggregator") is ConservationLevel.RELAXED
    assert config.resolve_conservation_level("other_functor") is ConservationLevel.MODERATE
    immutable_overrides = cast(
        dict[str, ConservationLevel | str],
        config.conservation_level_overrides,
    )
    with pytest.raises(TypeError):
        immutable_overrides["temporal_scaling"] = ConservationLevel.NONE


@pytest.mark.parametrize(
    "overrides",
    [
        {"": "strict"},
        {" temporal_aggregator": "strict"},
        {"temporal_aggregator ": "strict"},
        {"temporal_aggregator": "unknown"},
    ],
)
def test_temporal_config_rejects_invalid_override_contract(overrides) -> None:
    with pytest.raises((TypeError, ValueError)):
        TemporalConfig(
            source_frequency="hourly",
            target_frequency="daily",
            conservation_level_overrides=overrides,
        )


def test_temporal_config_builds_from_resolved_settings_without_owning_managers() -> None:
    config = TemporalConfig.from_resolved_settings(
        source_frequency="hourly",
        target_frequency="daily",
        settings={
            "conservation_level": "relaxed",
            "conservation_level_overrides": {"temporal_aggregator": "strict"},
            "temporal_semantics": "rate",
        },
    )

    assert config.temporal_semantics is temporal_types.TemporalSemantics.RATE
    assert config.resolve_conservation_level("temporal_aggregator") is ConservationLevel.STRICT
    assert config.resolve_conservation_level("other") is ConservationLevel.RELAXED

    source = inspect.getsource(temporal_types)
    assert "ParameterManager" not in source
    assert "CalendarHandler" not in source


def test_temporal_config_does_not_claim_frequency_direction_without_bounds() -> None:
    config = TemporalConfig(
        source_frequency="48hourly",
        target_frequency="daily",
        direction=ScalingDirection.UPSCALE,
        conservation_level=ConservationLevel.NONE,
    )

    assert config.source_frequency == "48hourly"
    assert config.target_frequency == "daily"
    assert "frequency_hierarchy" not in inspect.getsource(TemporalConfig.__post_init__)


def test_temporal_config_rejects_unknown_frequency_token() -> None:
    with pytest.raises(ValueError, match="frequency"):
        TemporalConfig(
            source_frequency="fortnightly",
            target_frequency="monthly",
            conservation_level=ConservationLevel.NONE,
        )


def test_temporal_aggregator_uses_parameter_mapping_and_cf_metadata():
    config = _temporal_config()
    aggregator = TemporalAggregator(config)

    data = torch.ones(24, dtype=torch.float32)
    source_bounds = _regular_bounds(24)
    target_bounds = torch.tensor([[0.0, 24.0]], dtype=torch.float64)
    source_time = _centers(source_bounds)
    target_time = _centers(target_bounds)

    result = aggregator.aggregate(
        data=data,
        source_time=source_time,
        target_time=target_time,
        variable_name="gross_primary_productivity",
        standard_name="gross_primary_productivity_of_carbon",
        units="kg m-2 s-1",
        calendar="noleap",
        time_units="hours since 2000-01-01",
        source_bounds=source_bounds,
        target_bounds=target_bounds,
        temporal_semantics="rate",
        target_units="kg m-2 s-1",
        measure_units="kg m-2",
        integration_time_unit="s",
    )

    # The ParameterManager value (0.8) is the sole threshold owner.
    assert aggregator.coverage_threshold == pytest.approx(0.8, rel=1e-6)
    assert result.diagnostics["aggregation_method"] == "time_weighted_mean"
    assert result.data.item() == pytest.approx(1.0)
    assert result.metadata.get("calendar") == "noleap"
    assert "time_units" in result.metadata
    assert result.metadata["cell_methods"] == "time: mean"
    assert "interval:" not in result.metadata["cell_methods"]
    assert result.metadata["history"].count("aggregation using time_weighted_mean") == 1
    assert int(result.quality_flags.insufficient_coverage_mask.sum().item()) == 0


def test_coverage_threshold_parameter_failure_is_not_masked(monkeypatch) -> None:
    """A canonical parameter lookup failure must propagate without config fallback."""
    aggregator = TemporalAggregator(_temporal_config())

    def fail_lookup(*_args, **_kwargs):
        raise RuntimeError("injected coverage lookup failure")

    monkeypatch.setattr(aggregator, "get_parameter_value", fail_lookup)

    with pytest.raises(RuntimeError, match="injected coverage lookup failure"):
        aggregator._resolve_coverage_threshold()


def test_coverage_threshold_lookup_is_required_and_has_no_default(monkeypatch) -> None:
    """The YAML ParameterSpec is the sole numeric threshold owner."""
    aggregator = TemporalAggregator(_temporal_config())
    captured = {}

    def capture_lookup(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return 0.75

    monkeypatch.setattr(aggregator, "get_parameter_value", capture_lookup)

    threshold = aggregator._resolve_coverage_threshold()

    assert threshold == pytest.approx(0.75)
    assert captured["name"] == "coverage_threshold"
    assert captured["kwargs"]["required"] is True
    assert "default" not in captured["kwargs"]


def test_coverage_threshold_cannot_be_reassigned_after_construction() -> None:
    """An attribute assignment cannot bypass coupling.scales.temporal.coverage_threshold."""
    aggregator = TemporalAggregator(_temporal_config())
    resolved = aggregator.coverage_threshold

    with pytest.raises(AttributeError):
        aggregator.coverage_threshold = 0.5

    assert aggregator.coverage_threshold == resolved
    assert resolved == pytest.approx(aggregator._resolve_coverage_threshold())


def test_per_instance_coverage_threshold_is_set_through_the_parameter_owner() -> None:
    """One aggregator's threshold comes from its validated parameter at construction."""
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value(
            "coupling.scales.temporal.coverage_threshold", 0.5, units="dimensionless",
        )
        aggregator = TemporalAggregator(_temporal_config())
    finally:
        manager.restore(snapshot)

    assert aggregator.coverage_threshold == pytest.approx(0.5)
    assert TemporalAggregator(_temporal_config()).coverage_threshold == pytest.approx(0.8)


def test_standard_name_does_not_supply_temporal_semantics() -> None:
    """CF identity metadata cannot select a temporal numerical law."""
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    source_bounds = _regular_bounds(2)
    target_bounds = torch.tensor([[0.0, 2.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="temporal_semantics is required"):
        aggregator.aggregate(
            data=torch.tensor([1.0, 2.0], dtype=torch.float64),
            source_time=_centers(source_bounds),
            target_time=_centers(target_bounds),
            standard_name="air_temperature",
            units="K",
            calendar="noleap",
            time_units="hours since 2000-01-01",
            source_bounds=source_bounds,
            target_bounds=target_bounds,
        )


def test_descriptive_weights_have_one_view_without_a_role_selector() -> None:
    """TemporalSemantics selects the law; no intensive/extensive re-map remains."""
    assert not hasattr(TemporalAggregator, "_variable_role_for_temporal_semantics")
    assert "temporal_semantics" not in inspect.signature(
        TemporalAggregator._compute_weights
    ).parameters
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    source_bounds = _regular_bounds(2)
    target_bounds = torch.tensor([[0.0, 2.0]], dtype=torch.float64)

    weights, _coverage, _target_durations, plan = aggregator._compute_weights(
        _centers(source_bounds),
        _centers(target_bounds),
        source_bounds,
        target_bounds,
        time_units="hours since 2000-01-01",
        calendar="noleap",
    )

    torch.testing.assert_close(weights, plan.target_fractions, rtol=0.0, atol=0.0)


def test_temporal_aggregator_forward_emits_state_updates() -> None:
    pm = get_global_manager()
    config = _temporal_config()
    aggregator = TemporalAggregator(config, parameter_manager=pm)

    data = torch.arange(24, dtype=torch.float32)
    source_bounds = _regular_bounds(24)
    target_bounds = torch.tensor([[0.0, 24.0]], dtype=torch.float64)
    source_time = _centers(source_bounds)
    target_time = _centers(target_bounds)

    outputs = aggregator.forward(
        {},
        time=0.0,
        dt=0.0,
        data=data,
        source_time=source_time,
        target_time=target_time,
        variable_name="daily_mean_test",
        standard_name="air_temperature",
        units="K",
        calendar="noleap",
        time_units="hours since 2000-01-01",
        source_bounds=source_bounds,
        target_bounds=target_bounds,
        temporal_semantics="interval_mean",
        target_units="K",
        measure_units="K hour",
        integration_time_unit="hour",
    )

    assert "daily_mean_test" in outputs
    assert outputs["daily_mean_test"].shape[0] == target_time.shape[0]
    assert outputs["daily_mean_test"].item() == pytest.approx(11.5)
    assert "_temporal_metadata" in outputs
    assert outputs["_temporal_metadata"].get("units") == "K"
    assert outputs["_temporal_metadata"]["cell_methods"] == "time: mean"
    assert "time_weighted_mean" not in outputs["_temporal_metadata"]["cell_methods"]
    assert "_temporal_quality_flags" in outputs
    assert "_temporal_diagnostics" in outputs


def test_temporal_aggregator_injects_parameter_manager_into_time_components() -> None:
    pm = get_global_manager()
    config = _temporal_config()
    aggregator = TemporalAggregator(config, parameter_manager=pm)

    assert aggregator.parameter_manager is pm
    assert aggregator.time_weight_generator.parameter_manager is pm
    assert aggregator.cf_time_processor.parameter_manager is pm


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mean", "time: mean area: mean"),
        ("time_weighted_mean", "time: mean area: mean"),
        ("min", "time: minimum area: mean"),
        ("max", "time: maximum area: mean"),
        ("median", "time: median area: mean"),
        ("sum", "time: sum area: mean"),
        ("variance", "time: variance area: mean"),
        ("std", "time: standard_deviation area: mean"),
        ("range", "time: range area: mean"),
    ],
)
def test_temporal_metadata_uses_controlled_cf_cell_method_tokens(
    method: str,
    expected: str,
) -> None:
    config = _temporal_config()
    aggregator = TemporalAggregator(config)

    result = aggregator.update_metadata(
        {
            "cell_methods": "time: point area: mean",
            "history": "source",
            "units": "kg m-2 s-1",
        },
        method=method,
    )

    assert result["cell_methods"] == expected
    assert result["history"] == f"source; aggregation using {method}"
    assert "interval:" not in result["cell_methods"]


@pytest.mark.parametrize("method", ["percentile", "unknown"])
def test_temporal_metadata_rejects_unverifiable_methods_before_mutation(method: str) -> None:
    config = _temporal_config()
    aggregator = TemporalAggregator(config)
    metadata = {
        "cell_methods": "area: mean",
        "history": "source",
        "units": "K",
    }

    with pytest.raises(ValueError, match=method):
        aggregator.update_metadata(metadata, method=method)

    assert metadata == {
        "cell_methods": "area: mean",
        "history": "source",
        "units": "K",
    }


@pytest.mark.parametrize(
    "method",
    [AggregationMethod.PERCENTILE],
)
def test_temporal_aggregate_rejects_unverifiable_metadata_before_weight_cache(
    method: AggregationMethod,
) -> None:
    config = _temporal_config()
    aggregator = TemporalAggregator(config, AggregationConfig(method=method))
    data = torch.arange(4, dtype=torch.float32)
    source_time = torch.arange(4, dtype=torch.float64)
    target_time = torch.tensor([1.5], dtype=torch.float64)

    with pytest.raises(ValueError, match=method.value):
        aggregator.aggregate(data, source_time, target_time, units="K")

    assert not hasattr(aggregator, "_weight_cache")


def test_temporal_aggregator_has_no_legacy_mean_kernel() -> None:
    assert not hasattr(TemporalAggregator, "_aggregate_time_weighted_mean")


@pytest.mark.parametrize("method", ["std", "range"])
def test_temporal_temperature_dispersion_rejects_on_scale_metadata(method: str) -> None:
    config = _temporal_config()
    aggregator = TemporalAggregator(config)
    metadata = {
        "units": "K",
        "units_metadata": "temperature: on_scale",
        "history": "source",
    }

    with pytest.raises(ValueError, match="temperature: difference"):
        aggregator.update_metadata(metadata, method=method)

    assert metadata["history"] == "source"
    assert "cell_methods" not in metadata


def test_temporal_aggregator_has_no_second_cf_metadata_owner() -> None:
    config = _temporal_config()
    aggregator = TemporalAggregator(config)

    assert not hasattr(aggregator, "cf_metadata_manager")


def test_temporal_aggregator_recomputes_same_shape_different_coordinates() -> None:
    config = _temporal_config(target_frequency="12hourly")
    aggregator = TemporalAggregator(config, AggregationConfig())
    source_bounds = _regular_bounds(24)
    data = torch.arange(24, dtype=torch.float64)

    first = _aggregate_with_bounds(
        aggregator,
        data,
        source_bounds,
        torch.tensor([[0.0, 12.0]], dtype=torch.float64),
    )
    second = _aggregate_with_bounds(
        aggregator,
        data,
        source_bounds,
        torch.tensor([[12.0, 24.0]], dtype=torch.float64),
    )

    assert first.data.item() == pytest.approx(5.5)
    assert second.data.item() == pytest.approx(17.5)
    assert not hasattr(aggregator, "_weight_cache")
    assert not hasattr(aggregator, "clear_cache")


def test_temporal_aggregator_recomputes_autograd_graph_for_each_call() -> None:
    config = _temporal_config(target_frequency="12hourly")
    aggregator = TemporalAggregator(config, AggregationConfig())
    source_bounds = _regular_bounds(24).requires_grad_()
    target_bounds = torch.tensor([[0.0, 12.0]], dtype=torch.float64, requires_grad=True)
    data = torch.arange(24, dtype=torch.float64, requires_grad=True)

    first = _aggregate_with_bounds(aggregator, data, source_bounds, target_bounds)
    first.data.sum().backward()
    assert data.grad is not None

    data.grad = None
    source_bounds.grad = None
    target_bounds.grad = None
    second = _aggregate_with_bounds(aggregator, data, source_bounds, target_bounds)
    second.data.sum().backward()

    assert data.grad is not None
    assert all(
        not isinstance(value, torch.Tensor) or value.grad_fn is None
        for value in vars(aggregator).values()
    )


def test_temporal_rate_mean_is_invariant_to_coordinate_time_unit() -> None:
    config = _temporal_config()
    hours = TemporalAggregator(config, AggregationConfig())
    seconds = TemporalAggregator(config, AggregationConfig())
    data = torch.ones(24, dtype=torch.float64)

    result_hours = _aggregate_with_bounds(
        hours,
        data,
        _regular_bounds(24),
        torch.tensor([[0.0, 24.0]], dtype=torch.float64),
        temporal_semantics="rate",
        units="kg m-2 s-1",
    )
    result_seconds = _aggregate_with_bounds(
        seconds,
        data,
        _regular_bounds(24, width=3600.0),
        torch.tensor([[0.0, 86400.0]], dtype=torch.float64),
        temporal_semantics="rate",
        units="kg m-2 s-1",
        time_units="seconds since 2000-01-01",
    )

    assert result_hours.data.item() == pytest.approx(1.0)
    assert result_seconds.data.item() == pytest.approx(1.0)
    assert result_hours.metadata["units"] == "kg m-2 s-1"
    assert result_seconds.metadata["units"] == "kg m-2 s-1"
    assert result_hours.metadata["cell_methods"] == "time: mean"
    assert result_seconds.metadata["cell_methods"] == "time: mean"


def test_temporal_interval_total_semantics_fail_before_bounds() -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())

    with pytest.raises(ValueError, match="interval_total"):
        aggregator.aggregate(
            torch.ones(24, dtype=torch.float64),
            torch.arange(24, dtype=torch.float64),
            torch.tensor([12.0], dtype=torch.float64),
            units="kg m-2",
            calendar="noleap",
            time_units="hours since 2000-01-01",
            temporal_semantics="interval_total",
        )


def test_temporal_missing_values_are_renormalized_per_output() -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    data = torch.ones((24, 2), dtype=torch.float64)
    data[0, 1] = float("nan")

    result = _aggregate_with_bounds(
        aggregator,
        data,
        _regular_bounds(24),
        torch.tensor([[0.0, 24.0]], dtype=torch.float64),
    )

    torch.testing.assert_close(result.data, torch.ones((1, 2), dtype=torch.float64))
    torch.testing.assert_close(
        result.quality_flags.valid_fraction,
        torch.tensor([[1.0, 23.0 / 24.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.quality_flags.missing_fraction,
        torch.tensor([[0.0, 1.0 / 24.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.quality_flags.num_samples,
        torch.tensor([[24, 23]]),
    )
    assert not torch.any(result.quality_flags.insufficient_coverage_mask)


def test_temporal_insufficient_coverage_is_independent_per_output() -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    data = torch.ones((24, 2), dtype=torch.float64)
    data[10:, 1] = float("nan")

    result = _aggregate_with_bounds(
        aggregator,
        data,
        _regular_bounds(24),
        torch.tensor([[0.0, 24.0]], dtype=torch.float64),
    )

    assert result.data[0, 0].item() == pytest.approx(1.0)
    assert torch.isnan(result.data[0, 1])
    assert result.quality_flags.insufficient_coverage_mask.tolist() == [[False, True]]
    assert result.quality_flags.num_samples.tolist() == [[24, 10]]


def test_temporal_time_dim_minus_one_preserves_per_output_missing_mask() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(time_dim=-1),
    )
    data = torch.ones((2, 24), dtype=torch.float64)
    data[1, 10:] = float("nan")

    result = _aggregate_with_bounds(
        aggregator,
        data,
        _regular_bounds(24),
        torch.tensor([[0.0, 24.0]], dtype=torch.float64),
    )

    assert result.data.shape == (2, 1)
    assert result.data[0, 0].item() == pytest.approx(1.0)
    assert torch.isnan(result.data[1, 0])
    assert result.quality_flags.insufficient_coverage_mask.tolist() == [[False], [True]]


@pytest.mark.parametrize(
    "data",
    [
        torch.ones(24, dtype=torch.int64),
        torch.ones(24, dtype=torch.bool),
        torch.ones(24, dtype=torch.float16),
        torch.ones(24, dtype=torch.bfloat16),
        torch.ones(24, dtype=torch.complex64),
    ],
)
def test_temporal_rejects_unsupported_scientific_dtype_before_bounds(
    data: torch.Tensor,
) -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())

    with pytest.raises(ValueError, match="float32 or float64"):
        aggregator.aggregate(
            data,
            torch.arange(24, dtype=torch.float64),
            torch.tensor([12.0], dtype=torch.float64),
            units="K",
            calendar="noleap",
            time_units="hours since 2000-01-01",
            temporal_semantics="interval_mean",
        )


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
def test_temporal_rejects_infinite_data_before_bounds(bad_value: float) -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    data = torch.ones(24, dtype=torch.float64)
    data[0] = bad_value

    with pytest.raises(ValueError, match="infinite"):
        aggregator.aggregate(
            data,
            torch.arange(24, dtype=torch.float64),
            torch.tensor([12.0], dtype=torch.float64),
            units="K",
            calendar="noleap",
            time_units="hours since 2000-01-01",
            temporal_semantics="interval_mean",
        )


@pytest.mark.parametrize(
    ("source_time", "target_time", "message"),
    [
        (torch.ones((2, 2), dtype=torch.float64), torch.ones(1), "one-dimensional"),
        (torch.tensor([0.0, float("inf")]), torch.ones(1), "finite"),
        (torch.tensor([0.0, 0.0]), torch.ones(1), "strictly increasing"),
    ],
)
def test_temporal_rejects_invalid_time_coordinates_before_bounds(
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    message: str,
) -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())

    with pytest.raises(ValueError, match=message):
        aggregator.aggregate(
            torch.ones(source_time.numel(), dtype=torch.float64),
            source_time,
            target_time.to(torch.float64),
            units="K",
            calendar="noleap",
            time_units="hours since 2000-01-01",
            temporal_semantics="interval_mean",
        )


@pytest.mark.parametrize("missing", ["calendar", "time_units", "source_bounds", "target_bounds"])
def test_temporal_requires_explicit_time_contract(missing: str) -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    source_bounds = _regular_bounds(24)
    target_bounds = torch.tensor([[0.0, 24.0]], dtype=torch.float64)
    kwargs = {
        "units": "K",
        "calendar": "noleap",
        "time_units": "hours since 2000-01-01",
        "source_bounds": source_bounds,
        "target_bounds": target_bounds,
        "temporal_semantics": "interval_mean",
    }
    kwargs.pop(missing)

    with pytest.raises(ValueError, match=missing):
        aggregator.aggregate(
            torch.ones(24, dtype=torch.float64),
            _centers(source_bounds),
            _centers(target_bounds),
            **kwargs,
        )


@pytest.mark.parametrize("calendar", ["gregorian", "proleptic_gregorian"])
@pytest.mark.parametrize("epoch", ["2000-02-29", "2020-02-29"])
def test_temporal_gregorian_signed_heat_flux_keeps_duration_and_calendar(
    calendar: str, epoch: str,
) -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    values = torch.tensor([-20.0, 100.0, -10.0, 50.0], dtype=torch.float64)
    results = []
    for coordinate_unit, scale in (("hours", 1.0), ("seconds", 3600.0)):
        result = _aggregate_with_bounds(
            aggregator, values, _regular_bounds(4, 12.0 * scale),
            torch.tensor([[0.0, 24.0], [24.0, 48.0]], dtype=torch.float64) * scale,
            calendar=calendar, time_units=f"{coordinate_unit} since {epoch}",
            units="W m-2", target_units="W m-2", measure_units="W m-2 s",
            integration_time_unit="s", temporal_semantics="interval_mean",
        )
        torch.testing.assert_close(result.data, torch.tensor([40.0, 20.0], dtype=torch.float64))
        assert result.metadata["calendar"] == calendar
        assert result.metadata["time_units"] == f"{coordinate_unit} since {epoch}"
        assert result.metadata["units"] == "W m-2"
        torch.testing.assert_close(result.data.sum() * 86400.0, values.sum() * 43200.0)
        results.append(result.data)
    torch.testing.assert_close(results[0], results[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("epoch", ["1900-02-29", "1582-10-10"])
def test_temporal_gregorian_invalid_epoch_rejected_by_calendar_owner(epoch: str) -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    with pytest.raises(temporal_core.ValidationError):
        _aggregate_with_bounds(
            aggregator, torch.ones(24, dtype=torch.float64), _regular_bounds(24),
            torch.tensor([[0.0, 24.0]], dtype=torch.float64), calendar="gregorian",
            time_units=f"hours since {epoch}",
        )


def test_temporal_rejects_unknown_runtime_calendar() -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    with pytest.raises(ValueError, match="calendar"):
        _aggregate_with_bounds(
            aggregator, torch.ones(24, dtype=torch.float64), _regular_bounds(24),
            torch.tensor([[0.0, 24.0]], dtype=torch.float64), calendar="unknown",
        )


@pytest.mark.parametrize("target_frequency", ["monthly", "seasonal", "annual"])
def test_temporal_calendar_dependent_frequency_fails_before_bounds(
    target_frequency: str,
) -> None:
    aggregator = TemporalAggregator(
        _temporal_config(source_frequency="daily", target_frequency=target_frequency),
        AggregationConfig(),
    )

    with pytest.raises(ValueError, match=target_frequency):
        aggregator.aggregate(
            torch.ones(30, dtype=torch.float64),
            torch.arange(30, dtype=torch.float64),
            torch.tensor([15.0], dtype=torch.float64),
            units="K",
            calendar="noleap",
            time_units="days since 2000-01-01",
            temporal_semantics="interval_mean",
        )


@pytest.mark.parametrize(
    "conservation_level",
    [ConservationLevel.STRICT, ConservationLevel.MODERATE, ConservationLevel.RELAXED],
)
def test_temporal_conservation_requires_parameter_backed_q_ref(
    conservation_level: ConservationLevel,
) -> None:
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=conservation_level),
        AggregationConfig(),
    )
    source_bounds = _regular_bounds(2)
    target_bounds = torch.tensor([[0.0, 2.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="q_ref_parameter"):
        _aggregate_with_bounds(
            aggregator,
            torch.ones(2, dtype=torch.float64),
            source_bounds,
            target_bounds,
            temporal_semantics="rate",
            units="kg day-1",
            time_units="days since 2000-01-01",
            target_units="kg day-1",
            measure_units="kg",
            integration_time_unit="day",
        )


def test_temporal_categorical_semantics_fail_before_bounds() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(spatial_basis=SpatialQuantityBasis.CATEGORICAL),
        AggregationConfig(),
    )

    with pytest.raises(ValueError, match="categorical"):
        aggregator.aggregate(
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            units="1",
            calendar="noleap",
            time_units="hours since 2000-01-01",
            temporal_semantics="categorical",
        )


def test_temporal_unknown_semantics_fail_before_bounds() -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())

    with pytest.raises(ValueError, match="temporal_semantics"):
        aggregator.aggregate(
            torch.ones(24, dtype=torch.float64),
            torch.arange(24, dtype=torch.float64),
            torch.tensor([12.0], dtype=torch.float64),
            units="K",
            calendar="noleap",
            time_units="hours since 2000-01-01",
        )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (AggregationMethod.MEAN, 10.0),
        (AggregationMethod.TIME_WEIGHTED_MEAN, 10.0),
        (AggregationMethod.MEDIAN, 10.0),
        (AggregationMethod.STANDARD_DEVIATION, 40.0 ** 0.5),
    ],
)
def test_temporal_statistics_use_overlap_duration_weights(
    method: AggregationMethod,
    expected: float,
) -> None:
    aggregator = TemporalAggregator(
        _temporal_config(source_frequency="hourly", target_frequency="12hourly"),
        AggregationConfig(method=method),
    )
    source_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 4.0], [4.0, 5.0]],
        dtype=torch.float64,
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64),
        source_bounds,
        torch.tensor([[0.0, 5.0]], dtype=torch.float64),
    )

    assert result.data.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (AggregationMethod.MINIMUM, 2.0),
        (AggregationMethod.MAXIMUM, 5.0),
        (AggregationMethod.RANGE, 3.0),
    ],
)
def test_temporal_extrema_ignore_nan_with_positive_coverage(
    method: AggregationMethod,
    expected: float,
) -> None:
    aggregator = TemporalAggregator(
        _temporal_config(target_frequency="12hourly"),
        AggregationConfig(method=method),
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([float("nan"), 5.0, 2.0], dtype=torch.float64),
        _regular_bounds(3),
        torch.tensor([[0.0, 3.0]], dtype=torch.float64),
    )

    assert result.data.item() == pytest.approx(expected)
    assert result.quality_flags.valid_fraction.item() == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(("calendar", "year_end"), [("noleap", 365.0), ("360_day", 360.0)])
def test_temporal_fixed_interval_mean_crosses_supported_year_boundary(
    calendar: str,
    year_end: float,
) -> None:
    aggregator = TemporalAggregator(
        _temporal_config(source_frequency="daily", target_frequency="2daily"),
        AggregationConfig(),
    )
    source_bounds = torch.tensor(
        [[year_end - 1.0, year_end], [year_end, year_end + 1.0]],
        dtype=torch.float64,
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([1.0, 3.0], dtype=torch.float64),
        source_bounds,
        torch.tensor([[year_end - 1.0, year_end + 1.0]], dtype=torch.float64),
        calendar=calendar,
        time_units="days since 2000-01-01",
    )

    assert result.data.item() == pytest.approx(2.0)


def test_temporal_rejects_overlapping_source_bounds() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(target_frequency="12hourly"),
        AggregationConfig(),
    )

    with pytest.raises(ValueError, match="overlap"):
        _aggregate_with_bounds(
            aggregator,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float64),
            torch.tensor([[0.0, 3.0]], dtype=torch.float64),
        )


def test_temporal_public_bypasses_are_absent_or_fail_closed() -> None:
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())

    with pytest.raises(ValueError, match="source_time.*target_time"):
        aggregator.upscale(torch.ones(24, dtype=torch.float64))
    with pytest.raises(ValueError, match="TemporalDisaggregator"):
        aggregator.downscale(torch.ones(1, dtype=torch.float64))

    for name in (
        "to_daily",
        "to_monthly",
        "resample",
        "clear_cache",
        "compute_time_weights",
        "check_conservation",
    ):
        assert not hasattr(TemporalAggregator, name)


def test_canonical_interval_mean_uses_m2_kernel_not_legacy_mean(
    monkeypatch,
) -> None:
    """The scientific mean path must delegate to the M2 canonical apply kernel."""
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    calls = {"count": 0}

    def capture_kernel(*args, **kwargs):
        calls["count"] += 1
        return temporal_core.apply_temporal_kernel(*args, **kwargs)

    monkeypatch.setattr(
        aggregator_module,
        "apply_temporal_kernel",
        capture_kernel,
        raising=False,
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([1.0, 3.0], dtype=torch.float64),
        _regular_bounds(2),
        torch.tensor([[0.0, 2.0]], dtype=torch.float64),
    )

    assert calls["count"] == 1
    torch.testing.assert_close(
        result.data,
        torch.tensor([2.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert result.diagnostics["gap_kind"] == "gap"


def test_rate_to_interval_accumulation_uses_real_time_unit_conversion() -> None:
    """One kg per hour over 7200 seconds must produce two kg."""
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.SUM),
    )
    bounds = torch.tensor([[0.0, 7200.0]], dtype=torch.float64)

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([1.0], dtype=torch.float64),
        bounds,
        bounds,
        temporal_semantics="rate",
        units="kg h-1",
        time_units="seconds since 2000-01-01",
        target_units="kg",
        measure_units="kg",
        integration_time_unit="hour",
    )

    torch.testing.assert_close(
        result.data,
        torch.tensor([2.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert result.metadata["units"] == "kg"


def test_interval_accumulation_partitions_source_totals() -> None:
    """Source interval totals must use overlap over source duration."""
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.SUM),
    )
    source_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 3.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    target_bounds = torch.tensor(
        [[0.0, 2.0], [2.0, 4.0]],
        dtype=torch.float64,
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([5.0, 10.0, 5.0], dtype=torch.float64),
        source_bounds,
        target_bounds,
        temporal_semantics="interval_accumulation",
        units="kg",
        target_units="kg",
        measure_units="kg",
        integration_time_unit=None,
    )

    torch.testing.assert_close(
        result.data,
        torch.tensor([10.0, 10.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_state_boundary_uses_exact_m2_mapping_without_interval_bounds() -> None:
    """Boundary state selection is exact and does not enter overlap aggregation."""
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())
    source_time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

    result = aggregator.aggregate(
        torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
        source_time,
        torch.tensor([1.0], dtype=torch.float64),
        units="K",
        target_units="K",
        calendar="noleap",
        time_units="days since 2000-01-01",
        temporal_semantics="state_at_boundary",
    )
    torch.testing.assert_close(
        result.data,
        torch.tensor([20.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(temporal_core.ValidationError, match="interpolation"):
        aggregator.aggregate(
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
            source_time,
            torch.tensor([0.5], dtype=torch.float64),
            units="K",
            target_units="K",
            calendar="noleap",
            time_units="days since 2000-01-01",
            temporal_semantics="state_at_boundary",
        )


@pytest.mark.parametrize(
    "level",
    [
        ConservationLevel.RELAXED,
        ConservationLevel.MODERATE,
        ConservationLevel.STRICT,
    ],
)
def test_state_boundary_rejects_unimplemented_conservation_levels(
    level: ConservationLevel,
) -> None:
    """State mapping must not claim a residual that it did not evaluate."""
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=level),
        AggregationConfig(),
    )

    with pytest.raises(ValueError, match="state_at_boundary.*conservation"):
        aggregator.aggregate(
            torch.tensor([10.0, 20.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            units="K",
            target_units="K",
            calendar="noleap",
            time_units="days since 2000-01-01",
            temporal_semantics="state_at_boundary",
        )


@pytest.mark.parametrize(
    ("level", "expected_checked", "expected_passed"),
    [
        (ConservationLevel.NONE, False, None),
        (ConservationLevel.RELAXED, False, None),
        (ConservationLevel.MODERATE, True, True),
        (ConservationLevel.STRICT, True, True),
    ],
)
def test_conservation_level_runtime_matrix_on_zero_residual(
    monkeypatch,
    level,
    expected_checked,
    expected_passed,
) -> None:
    """Level policy controls compute, compare, and blocking independently."""
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=level),
        AggregationConfig(),
    )
    q_ref_parameter = (
        None
        if level is ConservationLevel.NONE
        else _install_q_ref_lookup(aggregator, monkeypatch)
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([5.0, 5.0], dtype=torch.float64),
        _regular_bounds(2),
        torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        temporal_semantics="rate",
        units="kg day-1",
        time_units="days since 2000-01-01",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    assert result.diagnostics["conservation_checked"] is expected_checked
    assert result.diagnostics["conservation_passed"] is expected_passed
    assert result.diagnostics["conservation_level"] == level.value
    if level.compute_residual:
        torch.testing.assert_close(
            result.diagnostics["mapping_residual"],
            torch.tensor(0.0, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("level", "should_raise"),
    [
        (ConservationLevel.MODERATE, False),
        (ConservationLevel.STRICT, True),
    ],
)
def test_conservation_violation_flags_or_blocks_without_repair(
    monkeypatch,
    level,
    should_raise,
) -> None:
    """A known output perturbation must be reported and never corrected."""
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=level),
        AggregationConfig(),
    )
    q_ref_parameter = _install_q_ref_lookup(aggregator, monkeypatch)
    original_apply = temporal_core.apply_temporal_kernel

    def perturbed_apply(*args, **kwargs):
        base = original_apply(*args, **kwargs)
        return replace(base, values=base.values + 1.0e-3)

    monkeypatch.setattr(
        aggregator_module,
        "apply_temporal_kernel",
        perturbed_apply,
        raising=False,
    )

    def call():
        return _aggregate_with_bounds(
            aggregator,
            torch.tensor([5.0, 5.0], dtype=torch.float64),
            _regular_bounds(2),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
            temporal_semantics="rate",
            units="kg day-1",
            time_units="days since 2000-01-01",
            target_units="kg day-1",
            measure_units="kg",
            integration_time_unit="day",
            q_ref_parameter=q_ref_parameter,
        )

    if should_raise:
        with pytest.raises(ValueError, match="temporal conservation"):
            call()
    else:
        result = call()
        assert result.diagnostics["conservation_checked"] is True
        assert result.diagnostics["conservation_passed"] is False
        assert result.diagnostics["conservation_violation"] is True
        torch.testing.assert_close(
            result.data,
            torch.tensor([5.001], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-15,
        )


def test_half_coverage_keeps_mapping_residual_separate_from_gap(monkeypatch) -> None:
    """Accepted 50 percent support can map exactly while remaining incomplete.

    The 0.5 threshold of this aggregator is set on its parameter owner before
    construction (the threshold is read-only afterwards).
    """
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value(
            "coupling.scales.temporal.coverage_threshold", 0.5, units="dimensionless",
        )
        aggregator = TemporalAggregator(
            _temporal_config(conservation_level=ConservationLevel.RELAXED),
            AggregationConfig(),
        )
    finally:
        manager.restore(snapshot)
    assert aggregator.coverage_threshold == pytest.approx(0.5)
    q_ref_parameter = _install_q_ref_lookup(aggregator, monkeypatch)

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([5.0], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        temporal_semantics="rate",
        units="kg day-1",
        time_units="days since 2000-01-01",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    torch.testing.assert_close(
        result.diagnostics["mapping_residual"],
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert result.diagnostics["complete_coverage"].item() is False
    assert result.diagnostics["gap_total_fraction"].item() == pytest.approx(0.5)


def test_fully_masked_output_is_nan_and_explicitly_excluded_from_residual(
    monkeypatch,
) -> None:
    """Partial and full coverage masking coexist with a checked mapping residual."""
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=ConservationLevel.MODERATE),
        AggregationConfig(),
    )
    q_ref_parameter = _install_q_ref_lookup(aggregator, monkeypatch)
    data = torch.full((48, 2), 3.0, dtype=torch.float64)
    # Column 0: day 2 keeps 6 of 24 hours (0.25 < 0.8) -> partial masking.
    data[30:, 0] = float("nan")
    # Column 1: day 1 keeps 10 of 24 hours, day 2 none -> fully masked.
    data[10:, 1] = float("nan")

    result = _aggregate_with_bounds(
        aggregator,
        data,
        _regular_bounds(48),
        _regular_bounds(2, width=24.0),
        q_ref_parameter=q_ref_parameter,
    )

    assert result.data[0, 0].item() == pytest.approx(3.0)
    assert torch.isnan(result.data[1, 0])
    assert torch.isnan(result.data[:, 1]).all()
    torch.testing.assert_close(
        result.quality_flags.valid_fraction,
        torch.tensor([[1.0, 10.0 / 24.0], [0.25, 0.0]], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert result.quality_flags.insufficient_coverage_mask.tolist() == [
        [False, True],
        [True, True],
    ]
    diagnostics = result.diagnostics
    assert diagnostics["mapping_accepted_target_mask"].tolist() == [
        [True, False],
        [False, False],
    ]
    assert diagnostics["mapping_excluded_target_count"] == 3
    assert diagnostics["mapping_excluded_output_mask"].tolist() == [False, True]
    assert diagnostics["mapping_excluded_output_count"] == 1
    assert diagnostics["mapping_residual"][0].item() == pytest.approx(0.0, abs=0.0)
    assert torch.isnan(diagnostics["mapping_residual"][1])
    assert diagnostics["conservation_computed"] is True
    assert diagnostics["conservation_checked"] is True
    assert diagnostics["conservation_passed"] is True
    assert diagnostics["maximum_relative_residual"].item() == pytest.approx(
        0.0,
        abs=0.0,
    )
    assert "conservation_skip_reason" not in diagnostics


def test_all_outputs_fully_masked_return_nan_without_residual_failure(
    monkeypatch,
) -> None:
    """An all-NaN coverage result is returned; the residual is skipped explicitly."""
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=ConservationLevel.STRICT),
        AggregationConfig(),
    )
    q_ref_parameter = _install_q_ref_lookup(aggregator, monkeypatch)
    data = torch.full((24,), 3.0, dtype=torch.float64)
    data[5:] = float("nan")

    result = _aggregate_with_bounds(
        aggregator,
        data,
        _regular_bounds(24),
        torch.tensor([[0.0, 24.0]], dtype=torch.float64),
        q_ref_parameter=q_ref_parameter,
    )

    assert torch.isnan(result.data).all()
    assert result.quality_flags.valid_fraction.item() == pytest.approx(5.0 / 24.0)
    assert result.quality_flags.insufficient_coverage_mask.tolist() == [True]
    diagnostics = result.diagnostics
    assert diagnostics["mapping_excluded_output_mask"].item() is True
    assert diagnostics["mapping_excluded_output_count"] == 1
    assert diagnostics["mapping_excluded_target_count"] == 1
    assert torch.isnan(diagnostics["mapping_residual"])
    assert diagnostics["conservation_computed"] is True
    assert diagnostics["conservation_checked"] is False
    assert diagnostics["conservation_passed"] is None
    assert diagnostics["conservation_violation"] is False
    assert diagnostics["conservation_skip_reason"] == "no_accepted_target_support"
    assert "maximum_relative_residual" not in diagnostics


def test_moderate_conservation_matches_aligned_linear_analytic_oracle(
    monkeypatch,
) -> None:
    """The checked mapping must reproduce an exactly integrable linear rate."""
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=ConservationLevel.MODERATE),
        AggregationConfig(),
    )
    q_ref_parameter = _install_q_ref_lookup(aggregator, monkeypatch)
    source_bounds = torch.tensor(
        [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]],
        dtype=torch.float64,
    )
    target_bounds = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )

    # q(t) = 2t + 1 has exact source-cell means q((a+b)/2).
    source_means = torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=torch.float64)
    result = _aggregate_with_bounds(
        aggregator,
        source_means,
        source_bounds,
        target_bounds,
        temporal_semantics="rate",
        units="kg day-1",
        time_units="days since 2000-01-01",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    torch.testing.assert_close(
        result.data,
        torch.tensor([2.0, 4.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.diagnostics["observed_measure"].sum(),
        torch.tensor(6.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.diagnostics["mapping_residual"],
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 64.0,
    )
    assert result.diagnostics["conservation_checked"] is True
    assert result.diagnostics["conservation_passed"] is True


def test_moderate_conservation_separates_nonaligned_sine_oracles(
    monkeypatch,
) -> None:
    """A nonlinear MMS distinguishes exact integrals from discrete remapping."""
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=ConservationLevel.MODERATE),
        AggregationConfig(),
    )
    q_ref_parameter = _install_q_ref_lookup(aggregator, monkeypatch)
    source_bounds = torch.tensor(
        [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]],
        dtype=torch.float64,
    )
    target_bounds = torch.tensor(
        [[0.0, 0.75], [0.75, 2.0]],
        dtype=torch.float64,
    )
    pi = float(torch.pi)

    def analytic_integral(left: float, right: float) -> float:
        left_cos = torch.cos(torch.tensor(pi * left, dtype=torch.float64)).item()
        right_cos = torch.cos(torch.tensor(pi * right, dtype=torch.float64)).item()
        return 2.0 * (right - left) + (left_cos - right_cos) / pi

    source_means = torch.tensor(
        [
            analytic_integral(float(left), float(right)) / float(right - left)
            for left, right in source_bounds.tolist()
        ],
        dtype=torch.float64,
    )
    discrete_oracle = torch.stack(
        (
            (0.5 * source_means[0] + 0.25 * source_means[1]) / 0.75,
            (
                0.25 * source_means[1]
                + 0.5 * source_means[2]
                + 0.5 * source_means[3]
            )
            / 1.25,
        )
    )
    continuous_oracle = torch.tensor(
        [
            analytic_integral(0.0, 0.75) / 0.75,
            analytic_integral(0.75, 2.0) / 1.25,
        ],
        dtype=torch.float64,
    )

    result = _aggregate_with_bounds(
        aggregator,
        source_means,
        source_bounds,
        target_bounds,
        temporal_semantics="rate",
        units="kg day-1",
        time_units="days since 2000-01-01",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    torch.testing.assert_close(
        result.data,
        discrete_oracle,
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 64.0,
    )
    assert not torch.allclose(result.data, continuous_oracle, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(
        result.diagnostics["observed_measure"].sum(),
        torch.tensor(analytic_integral(0.0, 2.0), dtype=torch.float64),
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 64.0,
    )
    torch.testing.assert_close(
        result.diagnostics["mapping_residual"],
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 64.0,
    )
    assert result.diagnostics["conservation_checked"] is True
    assert result.diagnostics["conservation_passed"] is True


def test_variance_has_squared_units_and_population_oracle() -> None:
    """Variance is a descriptive statistic with squared output units."""
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.VARIANCE),
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        _regular_bounds(2),
        torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        units="K",
        variance_units="K2",
    )

    torch.testing.assert_close(
        result.data,
        torch.tensor([1.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert result.metadata["units"] == "K2"
    assert result.metadata["cell_methods"] == "time: variance"
    assert result.diagnostics["conservation_checked"] is False


def test_variance_requires_explicit_squared_units_for_compound_units() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.VARIANCE),
    )

    with pytest.raises(ValueError, match="variance_units"):
        _aggregate_with_bounds(
            aggregator,
            torch.tensor([0.0, 2.0], dtype=torch.float64),
            _regular_bounds(2),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
            units="m s-1",
        )


def test_variance_rejects_units_without_squared_dimensionality() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.VARIANCE),
    )

    with pytest.raises(ValueError, match="squared dimensionality"):
        _aggregate_with_bounds(
            aggregator,
            torch.tensor([0.0, 2.0], dtype=torch.float64),
            _regular_bounds(2),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
            units="m s-1",
            variance_units="m s-1",
        )


def test_variance_accepts_explicit_compound_squared_units() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.VARIANCE),
    )

    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        _regular_bounds(2),
        torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        units="m s-1",
        variance_units="m2 s-2",
    )

    assert result.metadata["units"] == "m2 s-2"


def test_descriptive_statistic_rejects_unapplied_target_unit_conversion() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.MINIMUM),
    )

    with pytest.raises(ValueError, match="identical source and target units"):
        _aggregate_with_bounds(
            aggregator,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            _regular_bounds(2),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
            units="kg",
            target_units="g",
            measure_units="g hour",
            integration_time_unit="hour",
        )


def test_variance_rejects_squared_units_with_different_scale() -> None:
    aggregator = TemporalAggregator(
        _temporal_config(),
        AggregationConfig(method=AggregationMethod.VARIANCE),
    )

    with pytest.raises(ValueError, match="same scale"):
        _aggregate_with_bounds(
            aggregator,
            torch.tensor([0.0, 2.0], dtype=torch.float64),
            _regular_bounds(2),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
            units="m",
            variance_units="cm2",
        )


@pytest.mark.skipif(
    not device_policy.is_cuda_available(),
    reason="CUDA is required for the registration migration probe",
)
def test_registration_migrates_temporal_helpers_and_executes_cuda_kernel() -> None:
    simulator = RELICSimulator(
        {
            "device": "cuda:0",
            "time_step": 3600.0,
            "strict_dependencies": False,
        }
    )
    aggregator = TemporalAggregator(_temporal_config(), AggregationConfig())

    simulator.register_functor(aggregator, execution_group="fast")

    expected_device = torch.device("cuda:0")
    assert aggregator.device == expected_device
    assert aggregator.cf_time_processor.device == expected_device
    assert aggregator.time_weight_generator.device == expected_device
    assert aggregator.time_weight_generator.cf_processor.device == expected_device

    source_bounds = _regular_bounds(2).to(expected_device)
    target_bounds = torch.tensor(
        [[0.0, 2.0]],
        dtype=torch.float64,
        device=expected_device,
    )
    result = _aggregate_with_bounds(
        aggregator,
        torch.tensor([1.0, 3.0], dtype=torch.float64, device=expected_device),
        source_bounds,
        target_bounds,
    )

    assert result.data.device == expected_device
    torch.testing.assert_close(
        result.data,
        torch.tensor([2.0], dtype=torch.float64, device=expected_device),
        rtol=0.0,
        atol=0.0,
    )


_LEDGER_TOLERANCE_KEY = "validation.conservation.conservation_tolerance"


def _record_parameter_reads(functor, monkeypatch) -> list:
    """Record every parameter key the functor requests through get_parameter_value."""
    reads = []
    lookup = functor.get_parameter_value

    def recording(name, **kwargs):
        reads.append(name)
        return lookup(name, **kwargs)

    monkeypatch.setattr(functor, "get_parameter_value", recording)
    return reads


def _with_ledger_tolerance(value, run):
    """Run with the declared ledger tolerance key set to value, then restore it."""
    manager = get_global_manager()
    original = float(manager.get(_LEDGER_TOLERANCE_KEY))
    manager.set_value(_LEDGER_TOLERANCE_KEY, value)
    try:
        return run()
    finally:
        manager.set_value(_LEDGER_TOLERANCE_KEY, original)


def test_aggregation_tolerance_is_the_declared_key_and_decides_the_verdict(monkeypatch):
    """The ledger reads the one tolerance key it declares; changing it flips the verdict.

    Two 5 kg day-1 days perturbed by +1e-3 kg day-1 give R = 2e-3 kg over an
    activity of 10 kg, a relative residual of 2e-4: a violation at the packaged
    1e-6 and a pass once the declared key is set to 1e-3.
    """
    aggregator = TemporalAggregator(
        _temporal_config(conservation_level=ConservationLevel.MODERATE),
        AggregationConfig(),
    )
    assert _LEDGER_TOLERANCE_KEY in aggregator.metadata.parameters
    q_ref_parameter = _install_q_ref_lookup(aggregator, monkeypatch)
    reads = _record_parameter_reads(aggregator, monkeypatch)
    original_apply = temporal_core.apply_temporal_kernel

    def perturbed_apply(*args, **kwargs):
        base = original_apply(*args, **kwargs)
        return replace(base, values=base.values + 1.0e-3)

    monkeypatch.setattr(aggregator_module, "apply_temporal_kernel", perturbed_apply, raising=False)

    def run():
        return _aggregate_with_bounds(
            aggregator,
            torch.tensor([5.0, 5.0], dtype=torch.float64),
            _regular_bounds(2),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
            temporal_semantics="rate",
            units="kg day-1",
            time_units="days since 2000-01-01",
            target_units="kg day-1",
            measure_units="kg",
            integration_time_unit="day",
            q_ref_parameter=q_ref_parameter,
        )

    packaged = run()
    assert [name for name in reads if "tolerance" in name] == [_LEDGER_TOLERANCE_KEY]
    assert packaged.diagnostics["conservation_passed"] is False
    relaxed = _with_ledger_tolerance(1.0e-3, run)
    assert relaxed.diagnostics["conservation_tolerance"] == pytest.approx(1.0e-3)
    assert relaxed.diagnostics["conservation_passed"] is True


def _forward_arguments(**overrides):
    source_bounds = _regular_bounds(24)
    target_bounds = torch.tensor([[0.0, 24.0]], dtype=torch.float64)
    arguments = dict(
        data=torch.arange(24, dtype=torch.float32),
        source_time=_centers(source_bounds),
        target_time=_centers(target_bounds),
        variable_name="daily_mean_test",
        standard_name="air_temperature",
        units="K",
        calendar="noleap",
        time_units="hours since 2000-01-01",
        source_bounds=source_bounds,
        target_bounds=target_bounds,
        temporal_semantics="interval_mean",
        target_units="K",
        measure_units="K hour",
        integration_time_unit="hour",
    )
    arguments.update(overrides)
    return arguments


def test_forward_keys_its_update_by_the_declared_variable_name() -> None:
    """forward requires variable_name and units; standard_name never stands in for the key."""
    aggregator = TemporalAggregator(_temporal_config())
    for missing in ("variable_name", "units"):
        arguments = _forward_arguments()
        arguments.pop(missing)
        with pytest.raises(ValueError, match=f"requires a non-empty {missing}"):
            aggregator.forward({}, time=0.0, dt=0.0, **arguments)


def test_absent_identity_metadata_is_omitted_not_published_empty() -> None:
    """aggregate: variable_name, standard_name and cell_methods are optional; absent means
    an unnamed field without a CF identity or an earlier cell method, and none of them is
    published as an empty string. units is required."""
    aggregator = TemporalAggregator(_temporal_config())
    arguments = _forward_arguments()
    for name in ("variable_name", "standard_name"):
        arguments.pop(name)
    result = aggregator.aggregate(**arguments)
    assert "variable_name" not in result.metadata
    assert "standard_name" not in result.metadata
    assert result.metadata["cell_methods"] == "time: mean"

    for name in ("variable_name", "standard_name", "cell_methods"):
        with pytest.raises(ValueError, match=name):
            aggregator.aggregate(**_forward_arguments(**{name: ""}))
    no_units = _forward_arguments()
    no_units.pop("units")
    with pytest.raises((TypeError, ValueError), match="units"):
        aggregator.aggregate(**no_units)
