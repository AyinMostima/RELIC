import inspect
import math
from dataclasses import replace

import pytest
import torch

import relic.coupling.scales.temporal.core as temporal_core
import relic.coupling.scales.temporal.disaggregator as disaggregator_module
from relic.core import device_policy
from relic.core.simulator import RELICSimulator
from relic.coupling.scales.temporal import (
    ConservationLevel,
    DisaggregationConfig,
    DisaggregationMethod,
    ScalingDirection,
    TemporalConfig,
    TemporalDisaggregator,
)
from relic.coupling.scales.temporal.types import TemporalSemantics
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.parameters import get_global_manager


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _config(
    *,
    level: ConservationLevel = ConservationLevel.NONE,
    source_frequency: str = "daily",
    target_frequency: str = "hourly",
    direction: ScalingDirection = ScalingDirection.DOWNSCALE,
    spatial_basis: SpatialQuantityBasis = SpatialQuantityBasis.INTENSIVE,
) -> TemporalConfig:
    return TemporalConfig(
        source_frequency=source_frequency,
        target_frequency=target_frequency,
        direction=direction,
        spatial_basis=spatial_basis,
        conservation_level=level,
        device=torch.device("cpu"),
    )


def _disaggregator(
    *,
    level: ConservationLevel = ConservationLevel.NONE,
    method: DisaggregationMethod = DisaggregationMethod.UNIFORM,
    allow_negative: bool = True,
    use_physical_constraints: bool = False,
    direction: ScalingDirection = ScalingDirection.DOWNSCALE,
    spatial_basis: SpatialQuantityBasis = SpatialQuantityBasis.INTENSIVE,
) -> TemporalDisaggregator:
    return TemporalDisaggregator(
        _config(level=level, direction=direction, spatial_basis=spatial_basis),
        DisaggregationConfig(
            method=method,
            time_dim=0,
            allow_negative=allow_negative,
            use_physical_constraints=use_physical_constraints,
        ),
    )


def _upscaling_disaggregator() -> TemporalDisaggregator:
    return _disaggregator(direction=ScalingDirection.UPSCALE)


def _centers(bounds: torch.Tensor) -> torch.Tensor:
    return torch.mean(bounds, dim=1)


def _map_with_bounds(
    disaggregator: TemporalDisaggregator,
    data: torch.Tensor,
    coarse_bounds: torch.Tensor,
    fine_bounds: torch.Tensor,
    *,
    semantics: str = "interval_mean",
    units: str = "K",
    target_units: str | None = None,
    measure_units: str | None = None,
    integration_time_unit: str | None = None,
    q_ref_parameter: str | None = None,
    valid_mask: torch.Tensor | None = None,
):
    temporal_semantics = TemporalSemantics(semantics)
    target_units = units if target_units is None else target_units
    if temporal_semantics is TemporalSemantics.INTERVAL_MEAN:
        measure_units = measure_units or f"{target_units} day"
        integration_time_unit = integration_time_unit or "day"
    elif temporal_semantics is TemporalSemantics.RATE:
        if measure_units is None or integration_time_unit is None:
            raise AssertionError("rate helper requires measure and time units")
    elif temporal_semantics is TemporalSemantics.INTERVAL_ACCUMULATION:
        measure_units = measure_units or target_units
        integration_time_unit = None
    return disaggregator.disaggregate(
        data=data,
        coarse_time=_centers(coarse_bounds),
        fine_time=_centers(fine_bounds),
        variable_name="test_quantity",
        standard_name="identity_only",
        units=units,
        calendar="noleap",
        time_units="days since 2000-01-01",
        coarse_bounds=coarse_bounds,
        fine_bounds=fine_bounds,
        temporal_semantics=semantics,
        target_units=target_units,
        measure_units=measure_units,
        integration_time_unit=integration_time_unit,
        q_ref_parameter=q_ref_parameter,
        valid_mask=valid_mask,
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


def _install_q_ref_lookup(
    disaggregator: TemporalDisaggregator,
    monkeypatch,
    value: float = 1.0,
) -> str:
    parameter_name = "tests.temporal.disaggregation_q_ref"
    original_lookup = disaggregator.get_parameter_value

    def lookup(name, **kwargs):
        if name == parameter_name:
            assert kwargs["required"] is True
            return value
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(disaggregator, "get_parameter_value", lookup)
    return parameter_name


def test_mapping_tolerance_is_the_declared_key_and_decides_the_verdict(monkeypatch):
    """The ledger reads the one tolerance key it declares; changing it flips the verdict.

    A +1e-3 kg day-1 perturbation of both half-day rates gives R = 1e-3 kg over an
    activity of 3 kg, a relative residual of 3.3e-4: a violation at the packaged
    1e-6 and a pass once the declared key is set to 1e-3.
    """
    disaggregator = _disaggregator(level=ConservationLevel.MODERATE)
    assert not hasattr(disaggregator, "relative_error_default")
    assert _LEDGER_TOLERANCE_KEY in disaggregator.metadata.parameters
    assert "coupling.scales.defaults.conservation_tolerance" not in disaggregator.metadata.parameters
    q_ref_parameter = _install_q_ref_lookup(disaggregator, monkeypatch)
    reads = _record_parameter_reads(disaggregator, monkeypatch)
    original_apply = temporal_core.apply_temporal_kernel

    def perturbed_apply(*args, **kwargs):
        base = original_apply(*args, **kwargs)
        return replace(base, values=base.values + 1.0e-3)

    monkeypatch.setattr(disaggregator_module, "apply_temporal_kernel", perturbed_apply, raising=False)

    def run():
        return _map_with_bounds(
            disaggregator,
            torch.tensor([3.0], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            semantics="rate",
            units="kg day-1",
            target_units="kg day-1",
            measure_units="kg",
            integration_time_unit="day",
            q_ref_parameter=q_ref_parameter,
        )

    packaged = run()
    assert [name for name in reads if "tolerance" in name] == [_LEDGER_TOLERANCE_KEY]
    assert packaged.diagnostics["conservation_tolerance"] == pytest.approx(
        float(get_global_manager().get(_LEDGER_TOLERANCE_KEY))
    )
    assert packaged.diagnostics["conservation_passed"] is False
    relaxed = _with_ledger_tolerance(1.0e-3, run)
    assert relaxed.diagnostics["conservation_tolerance"] == pytest.approx(1.0e-3)
    assert relaxed.diagnostics["conservation_passed"] is True


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("shape_function", "constant"),
        ("smooth_transitions", True),
        ("external_weights", torch.ones(2)),
        ("stochastic_noise_level", 0.1),
        ("conservation_tolerance", 1.0e-6),
    ],
)
def test_disaggregation_config_rejects_fields_without_a_consumer(field_name, value):
    assert set(DisaggregationConfig.__dataclass_fields__) == {
        "method",
        "allow_negative",
        "use_physical_constraints",
        "time_dim",
    }
    with pytest.raises(TypeError, match=field_name):
        DisaggregationConfig(**{field_name: value})


def test_disaggregator_does_not_keep_an_unused_profile_helper():
    disaggregator = _disaggregator()
    assert not hasattr(disaggregator, "shape_function")
    # No reconstruction path consumes profile shapes, so the module keeps no
    # standalone profile utility or selector enum.
    for name in ("ShapeFunction", "ShapeFunctionType"):
        assert not hasattr(disaggregator_module, name)
        assert name not in disaggregator_module.__all__


def test_configured_direction_must_match_the_entry_point() -> None:
    coarse_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    fine_bounds = torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="direction.*'upscale'.*disaggregate.*'downscale'"):
        _upscaling_disaggregator().disaggregate(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            coarse_bounds=coarse_bounds,
            fine_bounds=fine_bounds,
            **_interval_mean_contract(),
        )
    with pytest.raises(ValueError, match="direction.*'downscale'.*upscale.*'upscale'"):
        _disaggregator().upscale(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            fine_time=torch.tensor([0.25, 0.75], dtype=torch.float64),
            coarse_time=torch.tensor([0.5], dtype=torch.float64),
            fine_bounds=fine_bounds,
            coarse_bounds=coarse_bounds,
            **_interval_mean_contract(),
        )


def test_categorical_spatial_basis_is_rejected_like_the_aggregator() -> None:
    disaggregator = _disaggregator(spatial_basis=SpatialQuantityBasis.CATEGORICAL)

    with pytest.raises(ValueError, match="categorical"):
        _map_with_bounds(
            disaggregator,
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        )


def test_missing_parameter_raises_clear_error(monkeypatch):
    class DummyManager:
        def get(self, key, *args, **kwargs):
            if key.startswith("coupling.scales.conservative"):
                return None
            return get_global_manager().get(key)

        def get_parameter(self, key):
            return None

        def list(self, *args, **kwargs):
            return []

    dummy_manager = DummyManager()

    def fake_get_parameter_manager(self):
        return dummy_manager

    monkeypatch.setattr(TemporalDisaggregator, "get_parameter_manager", fake_get_parameter_manager)

    config = TemporalConfig(
        source_frequency="monthly",
        target_frequency="daily",
        direction=ScalingDirection.DOWNSCALE,
        spatial_basis=SpatialQuantityBasis.CELL_TOTAL,
        conservation_level=ConservationLevel.MODERATE,
    )

    with pytest.raises(ValueError) as excinfo:
        TemporalDisaggregator(config)

    assert "coupling.scales.conservative.clamp_min" in str(excinfo.value)


def test_disaggregate_requires_explicit_semantics_despite_standard_name() -> None:
    disaggregator = _disaggregator()

    with pytest.raises(ValueError, match="temporal_semantics is required"):
        disaggregator.disaggregate(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            standard_name="precipitation_flux",
            units="kg day-1",
            calendar="noleap",
            time_units="days since 2000-01-01",
            coarse_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            fine_bounds=torch.tensor(
                [[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64
            ),
            target_units="kg day-1",
            measure_units="kg",
            integration_time_unit="day",
        )


def test_disaggregate_requires_explicit_bounds_and_has_no_bounds_guesser() -> None:
    disaggregator = _disaggregator()

    with pytest.raises(ValueError, match="coarse_bounds.*fine_bounds"):
        disaggregator.disaggregate(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            units="K",
            calendar="noleap",
            time_units="days since 2000-01-01",
            temporal_semantics="interval_mean",
            target_units="K",
            measure_units="K day",
            integration_time_unit="day",
        )
    assert not hasattr(TemporalDisaggregator, "_create_time_bounds")


def _interval_mean_contract() -> dict:
    return {
        "units": "K",
        "calendar": "noleap",
        "time_units": "days since 2000-01-01",
        "temporal_semantics": "interval_mean",
        "target_units": "K",
        "measure_units": "K day",
        "integration_time_unit": "day",
    }


def test_missing_bounds_messages_name_the_keys_of_each_entry_point() -> None:
    """Bound errors name the keys each public entry actually reads."""
    disaggregator = _disaggregator()
    coarse_time = torch.tensor([0.5], dtype=torch.float64)
    fine_time = torch.tensor([0.25, 0.75], dtype=torch.float64)
    coarse_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    fine_bounds = torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    with pytest.raises(
        ValueError,
        match=r"missing fine_bounds \(target_bounds\)$",
    ):
        disaggregator.disaggregate(
            torch.tensor([2.0], dtype=torch.float64),
            coarse_time,
            fine_time,
            coarse_bounds=coarse_bounds,
            **_interval_mean_contract(),
        )
    with pytest.raises(
        ValueError,
        match=r"missing coarse_bounds \(source_bounds\)$",
    ):
        disaggregator.disaggregate(
            torch.tensor([2.0], dtype=torch.float64),
            coarse_time,
            fine_time,
            target_bounds=fine_bounds,
            **_interval_mean_contract(),
        )
    with pytest.raises(
        ValueError,
        match=(
            r"missing fine_bounds \(source_bounds\), "
            r"coarse_bounds \(target_bounds\)$"
        ),
    ):
        _upscaling_disaggregator().upscale(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            fine_time=fine_time,
            coarse_time=coarse_time,
            **_interval_mean_contract(),
        )


def test_duplicate_bound_aliases_are_rejected_instead_of_overwritten() -> None:
    disaggregator = _disaggregator()
    coarse_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    fine_bounds = torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="both coarse_bounds and source_bounds"):
        disaggregator.disaggregate(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            coarse_bounds=coarse_bounds,
            source_bounds=coarse_bounds,
            fine_bounds=fine_bounds,
            **_interval_mean_contract(),
        )
    with pytest.raises(ValueError, match="both coarse_bounds and target_bounds"):
        _upscaling_disaggregator().upscale(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            fine_time=torch.tensor([0.25, 0.75], dtype=torch.float64),
            coarse_time=torch.tensor([0.5], dtype=torch.float64),
            fine_bounds=fine_bounds,
            coarse_bounds=coarse_bounds,
            target_bounds=coarse_bounds,
            **_interval_mean_contract(),
        )


def test_coordinate_and_bound_errors_name_public_arguments() -> None:
    disaggregator = _disaggregator()
    coarse_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    fine_bounds = torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="data time axis must match coarse_time"):
        disaggregator.disaggregate(
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            coarse_bounds=coarse_bounds,
            fine_bounds=fine_bounds,
            **_interval_mean_contract(),
        )
    with pytest.raises(ValueError, match="^fine_time must be strictly increasing"):
        disaggregator.disaggregate(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.75, 0.25], dtype=torch.float64),
            coarse_bounds=coarse_bounds,
            fine_bounds=fine_bounds,
            **_interval_mean_contract(),
        )
    with pytest.raises(ValueError, match=r"^fine_bounds must have shape \(2, 2\)"):
        _upscaling_disaggregator().upscale(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            fine_time=torch.tensor([0.25, 0.75], dtype=torch.float64),
            coarse_time=torch.tensor([0.5], dtype=torch.float64),
            fine_bounds=coarse_bounds,
            coarse_bounds=coarse_bounds,
            **_interval_mean_contract(),
        )
    with pytest.raises(
        ValueError,
        match=r"explicit coordinates and bounds; missing: coarse_bounds, fine_bounds$",
    ):
        disaggregator.to_hourly(
            torch.tensor([2.0], dtype=torch.float64),
            coarse_time=torch.tensor([0.5], dtype=torch.float64),
            fine_time=torch.tensor([0.25, 0.75], dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "level",
    [ConservationLevel.MODERATE, ConservationLevel.STRICT],
)
def test_fully_missing_coarse_series_is_nan_and_excluded_from_residual(
    monkeypatch,
    level,
) -> None:
    """A fully missing output series is reported, not a vacuous-support error."""
    disaggregator = _disaggregator(level=level)
    q_ref_parameter = _install_q_ref_lookup(disaggregator, monkeypatch)

    partial = _map_with_bounds(
        disaggregator,
        torch.tensor([[3.0, float("nan")]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        semantics="rate",
        units="kg day-1",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    assert partial.data[:, 0].tolist() == [3.0, 3.0]
    assert torch.isnan(partial.data[:, 1]).all()
    assert partial.diagnostics["mapping_excluded_output_mask"].tolist() == [
        False,
        True,
    ]
    assert partial.diagnostics["mapping_excluded_output_count"] == 1
    assert partial.diagnostics["mapping_excluded_target_count"] == 2
    assert partial.diagnostics["mapping_residual"][0].item() == pytest.approx(
        0.0,
        abs=torch.finfo(torch.float64).eps * 64.0,
    )
    assert torch.isnan(partial.diagnostics["mapping_residual"][1])
    assert partial.diagnostics["conservation_checked"] is True
    assert partial.diagnostics["conservation_passed"] is True
    assert partial.conservation_error == pytest.approx(
        0.0,
        abs=torch.finfo(torch.float64).eps * 64.0,
    )

    empty = _map_with_bounds(
        disaggregator,
        torch.tensor([float("nan")], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        semantics="rate",
        units="kg day-1",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    assert torch.isnan(empty.data).all()
    assert empty.diagnostics["mapping_excluded_output_count"] == 1
    assert empty.diagnostics["conservation_computed"] is True
    assert empty.diagnostics["conservation_checked"] is False
    assert empty.diagnostics["conservation_passed"] is None
    assert empty.diagnostics["conservation_skip_reason"] == (
        "no_accepted_target_support"
    )
    assert math.isnan(empty.conservation_error)


def test_uniform_interval_accumulation_uses_fractional_overlap_oracle() -> None:
    disaggregator = _disaggregator()
    result = _map_with_bounds(
        disaggregator,
        torch.tensor([7200.0], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.25], [0.25, 1.0]], dtype=torch.float64),
        semantics="interval_accumulation",
        units="kg",
    )

    torch.testing.assert_close(
        result.data,
        torch.tensor([1800.0, 5400.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.diagnostics["observed_measure"],
        result.data,
        rtol=0.0,
        atol=0.0,
    )
    assert result.metadata["units"] == "kg"


def test_uniform_rate_target_crosses_source_bins_with_hand_oracle() -> None:
    disaggregator = _disaggregator()
    result = _map_with_bounds(
        disaggregator,
        torch.tensor([2.0, 5.0], dtype=torch.float64),
        torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
        torch.tensor([[0.5, 2.0]], dtype=torch.float64),
        semantics="rate",
        units="kg day-1",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
    )

    torch.testing.assert_close(
        result.data,
        torch.tensor([4.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.diagnostics["observed_measure"],
        torch.tensor([6.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_nonfinite_source_is_not_silently_replaced() -> None:
    disaggregator = _disaggregator()
    coarse_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    fine_bounds = torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="infinite|Inf"):
        _map_with_bounds(
            disaggregator,
            torch.tensor([float("inf")], dtype=torch.float64),
            coarse_bounds,
            fine_bounds,
        )

    result = _map_with_bounds(
        disaggregator,
        torch.tensor([float("nan")], dtype=torch.float64),
        coarse_bounds,
        fine_bounds,
    )
    assert torch.isnan(result.data).all()
    assert torch.all(result.diagnostics["gap_total_fraction"] == 1.0)


@pytest.mark.parametrize(
    ("level", "expected_computed", "expected_checked", "expected_passed"),
    [
        (ConservationLevel.NONE, False, False, None),
        (ConservationLevel.RELAXED, True, False, None),
        (ConservationLevel.MODERATE, True, True, True),
        (ConservationLevel.STRICT, True, True, True),
    ],
)
def test_conservation_level_matrix_uses_canonical_mapping_residual(
    monkeypatch,
    level,
    expected_computed,
    expected_checked,
    expected_passed,
) -> None:
    disaggregator = _disaggregator(level=level)
    q_ref_parameter = (
        None
        if level is ConservationLevel.NONE
        else _install_q_ref_lookup(disaggregator, monkeypatch)
    )
    result = _map_with_bounds(
        disaggregator,
        torch.tensor([3.0], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        semantics="rate",
        units="kg day-1",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    assert result.diagnostics["conservation_computed"] is expected_computed
    assert result.diagnostics["conservation_checked"] is expected_checked
    assert result.diagnostics["conservation_passed"] is expected_passed
    if expected_computed:
        torch.testing.assert_close(
            result.diagnostics["mapping_residual"],
            torch.tensor(0.0, dtype=torch.float64),
            rtol=0.0,
            atol=torch.finfo(torch.float64).eps * 64.0,
        )


@pytest.mark.parametrize(
    ("level", "raises"),
    [
        (ConservationLevel.MODERATE, False),
        (ConservationLevel.STRICT, True),
    ],
)
def test_known_residual_is_flagged_or_blocked_without_repair(
    monkeypatch,
    level,
    raises,
) -> None:
    disaggregator = _disaggregator(level=level)
    q_ref_parameter = _install_q_ref_lookup(disaggregator, monkeypatch)
    original_apply = temporal_core.apply_temporal_kernel

    def perturbed_apply(*args, **kwargs):
        base = original_apply(*args, **kwargs)
        return replace(base, values=base.values + 1.0e-3)

    monkeypatch.setattr(
        disaggregator_module,
        "apply_temporal_kernel",
        perturbed_apply,
        raising=False,
    )

    def call():
        return _map_with_bounds(
            disaggregator,
            torch.tensor([3.0], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            semantics="rate",
            units="kg day-1",
            target_units="kg day-1",
            measure_units="kg",
            integration_time_unit="day",
            q_ref_parameter=q_ref_parameter,
        )

    if raises:
        with pytest.raises(ValueError, match="temporal conservation"):
            call()
    else:
        result = call()
        assert result.diagnostics["conservation_passed"] is False
        assert result.diagnostics["conservation_violation"] is True
        torch.testing.assert_close(
            result.data,
            torch.tensor([3.001, 3.001], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-15,
        )


@pytest.mark.parametrize(
    "method",
    [
        DisaggregationMethod.LINEAR,
        DisaggregationMethod.CLIMATOLOGY_DIURNAL,
        DisaggregationMethod.SHAPE_PRESERVING_SPLINE,
        DisaggregationMethod.CUBIC_SPLINE,
        DisaggregationMethod.EXTERNAL_DRIVER,
        DisaggregationMethod.STOCHASTIC,
    ],
)
def test_unbound_reconstruction_methods_fail_closed(method) -> None:
    disaggregator = _disaggregator(method=method)

    with pytest.raises(ValueError, match="canonical.*(reconstruction|profile|RNG)"):
        _map_with_bounds(
            disaggregator,
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        )


def test_stochastic_method_never_consumes_global_rng(monkeypatch) -> None:
    disaggregator = _disaggregator(method=DisaggregationMethod.STOCHASTIC)

    def reject_rng(*_args, **_kwargs):
        raise AssertionError("global RNG must not be consumed")

    monkeypatch.setattr(disaggregator_module.torch, "randn_like", reject_rng)
    with pytest.raises(ValueError, match="RNG"):
        _map_with_bounds(
            disaggregator,
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        )


def test_downscale_and_ratio_only_api_fail_without_explicit_contract() -> None:
    disaggregator = _disaggregator()
    data = torch.tensor([2.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="coarse_time.*fine_time"):
        disaggregator.downscale(data)
    with pytest.raises(ValueError, match="ratio-only.*unbound"):
        disaggregator.downscale_simple(data)


def test_upscale_uses_canonical_core_kernel_instead_of_identity() -> None:
    disaggregator = _upscaling_disaggregator()
    result = disaggregator.upscale(
        torch.tensor([1.0, 3.0], dtype=torch.float64),
        fine_time=torch.tensor([0.5, 1.5], dtype=torch.float64),
        coarse_time=torch.tensor([1.0], dtype=torch.float64),
        fine_bounds=torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
        coarse_bounds=torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        calendar="noleap",
        time_units="days since 2000-01-01",
        temporal_semantics="interval_mean",
        units="K",
        target_units="K",
        measure_units="K day",
        integration_time_unit="day",
    )

    torch.testing.assert_close(
        result,
        torch.tensor([2.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_metadata_uses_cf_cell_method_not_reconstruction_name() -> None:
    disaggregator = _disaggregator()
    metadata = disaggregator.update_metadata(
        {
            "cell_methods": "area: mean",
            "temporal_semantics": "interval_mean",
        },
        method="uniform",
    )

    assert metadata["cell_methods"] == "time: mean area: mean"
    assert "uniform" not in metadata["cell_methods"]


def test_descending_coordinates_fail_instead_of_warning() -> None:
    disaggregator = _disaggregator()
    with pytest.raises(ValueError, match="strictly increasing"):
        disaggregator.disaggregate(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([1.5, 0.5], dtype=torch.float64),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            units="K",
            calendar="noleap",
            time_units="days since 2000-01-01",
            coarse_bounds=torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64),
            fine_bounds=torch.tensor(
                [[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64
            ),
            temporal_semantics="interval_mean",
            target_units="K",
            measure_units="K day",
            integration_time_unit="day",
        )


def test_state_boundary_policies_are_explicit() -> None:
    disaggregator = _disaggregator()
    source_time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    source = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)

    for policy, target_time, expected in (
        ("exact", torch.tensor([0.0, 2.0], dtype=torch.float64), [10.0, 30.0]),
        (
            "left",
            torch.tensor([0.25, 1.0, 1.75], dtype=torch.float64),
            [10.0, 20.0, 20.0],
        ),
        (
            "right",
            torch.tensor([0.25, 1.0, 1.75], dtype=torch.float64),
            [20.0, 20.0, 30.0],
        ),
    ):
        result = disaggregator.disaggregate(
            source,
            source_time,
            target_time,
            units="K",
            calendar="noleap",
            time_units="days since 2000-01-01",
            temporal_semantics="state_at_boundary",
            state_boundary_policy=policy,
            target_units="K",
        )
        torch.testing.assert_close(
            result.data,
            torch.tensor(expected, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )


def test_state_boundary_conservation_levels_fail_closed() -> None:
    disaggregator = _disaggregator(level=ConservationLevel.RELAXED)

    with pytest.raises(ValueError, match="state_at_boundary.*conservation"):
        disaggregator.disaggregate(
            torch.tensor([10.0, 20.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            units="K",
            calendar="noleap",
            time_units="days since 2000-01-01",
            temporal_semantics="state_at_boundary",
            state_boundary_policy="exact",
            target_units="K",
        )


def test_physical_constraints_reject_instead_of_clamp() -> None:
    disaggregator = _disaggregator(
        allow_negative=False,
        use_physical_constraints=True,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        _map_with_bounds(
            disaggregator,
            torch.tensor([-1.0], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            semantics="interval_accumulation",
            units="kg",
        )


def test_convenience_wrappers_do_not_invent_24_or_30_step_axes() -> None:
    disaggregator = _disaggregator()
    data = torch.tensor([2.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="explicit.*coordinates.*bounds"):
        disaggregator.to_hourly(data)
    with pytest.raises(ValueError, match="explicit.*coordinates.*bounds"):
        disaggregator.to_daily(data)


def test_legacy_local_formula_and_correction_paths_are_absent() -> None:
    for name in (
        "_determine_variable_role",
        "_perform_disaggregation",
        "_disaggregate_uniform",
        "_disaggregate_linear",
        "_disaggregate_spline",
        "_disaggregate_stochastic",
        "_create_time_bounds",
        "_correct_conservation",
        "check_conservation",
    ):
        assert not hasattr(TemporalDisaggregator, name)

    source = inspect.getsource(disaggregator_module)
    assert "torch.randn_like" not in source
    assert "1e-12" not in source
    assert "warnings.warn" not in source


def test_linear_rate_mms_checks_analytic_integral_and_mapping_residual(
    monkeypatch,
) -> None:
    """q(t)=2t+1 has exact source means [2,4] and integral six."""
    disaggregator = _disaggregator(level=ConservationLevel.MODERATE)
    q_ref_parameter = _install_q_ref_lookup(disaggregator, monkeypatch)
    result = _map_with_bounds(
        disaggregator,
        torch.tensor([2.0, 4.0], dtype=torch.float64),
        torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]],
            dtype=torch.float64,
        ),
        semantics="rate",
        units="kg day-1",
        target_units="kg day-1",
        measure_units="kg",
        integration_time_unit="day",
        q_ref_parameter=q_ref_parameter,
    )

    torch.testing.assert_close(
        result.data,
        torch.tensor([2.0, 2.0, 4.0, 4.0], dtype=torch.float64),
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
    assert result.diagnostics["mapping_law_id"] == (
        "temporal_rate_integral_consistency"
    )
    assert result.diagnostics["q_ref_provenance"] == q_ref_parameter


def test_disaggregation_preserves_per_support_missing_masks() -> None:
    disaggregator = _disaggregator()
    result = _map_with_bounds(
        disaggregator,
        torch.tensor([[10.0, 20.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        valid_mask=torch.tensor([[True, False]]),
    )

    torch.testing.assert_close(
        result.data[:, 0],
        torch.tensor([10.0, 10.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isnan(result.data[:, 1]).all()
    torch.testing.assert_close(
        result.diagnostics["valid_coverage"],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_conservation_levels_require_q_ref_parameter() -> None:
    disaggregator = _disaggregator(level=ConservationLevel.RELAXED)

    with pytest.raises(ValueError, match="q_ref_parameter is required"):
        _map_with_bounds(
            disaggregator,
            torch.tensor([3.0], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            semantics="rate",
            units="kg day-1",
            target_units="kg day-1",
            measure_units="kg",
            integration_time_unit="day",
        )


@pytest.mark.parametrize(
    ("wrapper_name", "calendar", "target_count"),
    [
        ("to_hourly", "noleap", 24),
        ("to_daily", "360_day", 30),
    ],
)
def test_convenience_wrappers_execute_only_with_explicit_calendar_bounds(
    wrapper_name,
    calendar,
    target_count,
) -> None:
    disaggregator = _disaggregator()
    coarse_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    edges = torch.linspace(0.0, 1.0, target_count + 1, dtype=torch.float64)
    fine_bounds = torch.stack((edges[:-1], edges[1:]), dim=1)
    wrapper = getattr(disaggregator, wrapper_name)

    result = wrapper(
        torch.tensor([float(target_count)], dtype=torch.float64),
        coarse_time=torch.tensor([0.5], dtype=torch.float64),
        fine_time=torch.mean(fine_bounds, dim=1),
        coarse_bounds=coarse_bounds,
        fine_bounds=fine_bounds,
        units="kg",
        calendar=calendar,
        time_units="days since 2000-01-01",
        temporal_semantics="interval_accumulation",
        target_units="kg",
        measure_units="kg",
    )

    torch.testing.assert_close(
        result.data,
        torch.ones(target_count, dtype=torch.float64),
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 128.0,
    )


@pytest.mark.skipif(
    not device_policy.is_cuda_available(),
    reason="CUDA is required for the registration migration probe",
)
def test_registration_migrates_disaggregation_helpers_to_cuda() -> None:
    simulator = RELICSimulator(
        {
            "device": "cuda:0",
            "time_step": 3600.0,
            "strict_dependencies": False,
        }
    )
    disaggregator = _disaggregator()

    simulator.register_functor(disaggregator, execution_group="fast")

    expected = torch.device("cuda:0")
    assert disaggregator.device == expected
    assert disaggregator.cf_time_processor.device == expected
    assert disaggregator.time_weight_generator.device == expected
    assert disaggregator.time_weight_generator.cf_processor.device == expected

