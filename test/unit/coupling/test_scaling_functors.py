import inspect
from dataclasses import replace

import pytest
import torch

import relic.coupling.scales.temporal.core as temporal_core
import relic.coupling.scales.temporal.temporal_scaling as temporal_scaling_module
from relic.core import device_policy
from relic.core.simulator import RELICSimulator
from relic.coupling.external.common.regrid import RegridMethod, RegridConfig, RegridOperator
from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling
from relic.coupling.scales.temporal.aggregator import (
    AggregationConfig,
    TemporalAggregator,
)
from relic.coupling.scales.temporal.disaggregator import TemporalDisaggregator
from relic.coupling.scales.temporal.temporal_scaling import TemporalScaling
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.parameters import get_global_manager
from relic.coupling.scales.temporal.types import (
    ConservationLevel,
    ScalingDirection,
    TemporalConfig,
)


def test_temporal_scaling_forward_identity():
    scaler = TemporalScaling(source_resolution=60.0, target_resolution=60.0, method="sum", source_scale="source", target_scale="target")
    state = {"time_series": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    result = scaler.forward(state, time=0.0, dt=60.0)
    torch.testing.assert_close(result["time_series_scaled"], state["time_series"])


def test_temporal_scaling_forward_aggregate_sum():
    scaler = TemporalScaling(source_resolution=60.0, target_resolution=120.0, method="sum", source_scale="source", target_scale="target")
    series = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    result = scaler.forward({"time_series": series}, time=0.0, dt=120.0)
    expected = torch.tensor([3.0, 7.0], dtype=torch.float32)
    torch.testing.assert_close(result["time_series_scaled"], expected)


def test_temporal_scaling_public_signatures_and_output_key_are_stable():
    assert list(inspect.signature(TemporalScaling.resample).parameters) == [
        "self",
        "series",
        "source_dt",
        "target_dt",
        "method",
    ]
    assert list(inspect.signature(TemporalScaling.forward).parameters) == [
        "self",
        "state",
        "time",
        "dt",
        "kwargs",
    ]
    scaler = TemporalScaling(source_resolution=60.0, target_resolution=120.0, method="mean", source_scale="source", target_scale="target")
    assert scaler.metadata.input_variables == ["time_series"]
    assert scaler.metadata.output_variables == ["time_series_scaled"]


def test_temporal_scaling_uses_core_without_child_functors(monkeypatch):
    def reject_child(*_args, **_kwargs):
        raise AssertionError("child temporal Functor must not execute")

    monkeypatch.setattr(TemporalAggregator, "aggregate", reject_child)
    monkeypatch.setattr(TemporalDisaggregator, "disaggregate", reject_child)
    calls = {"count": 0}
    original_apply = temporal_core.apply_temporal_kernel

    def capture_core(*args, **kwargs):
        calls["count"] += 1
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        temporal_scaling_module,
        "apply_temporal_kernel",
        capture_core,
        raising=False,
    )
    scaler = TemporalScaling(source_resolution=1.0, target_resolution=2.0, method="mean", source_scale="source", target_scale="target")
    result = scaler.resample(
        torch.tensor([1.0, 3.0], dtype=torch.float64),
        source_dt=1.0,
        target_dt=2.0,
        method="mean",
    )

    assert calls["count"] == 1
    torch.testing.assert_close(
        result,
        torch.tensor([2.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("method", "source_dt", "target_dt", "data", "expected"),
    [
        ("mean", 1.0, 2.0, [1.0, 3.0, 5.0, 7.0], [2.0, 6.0]),
        ("sum", 1.0, 2.0, [1.0, 3.0, 5.0, 7.0], [4.0, 12.0]),
        ("mean", 2.0, 1.0, [4.0], [4.0, 4.0]),
        ("sum", 2.0, 1.0, [4.0], [2.0, 2.0]),
    ],
)
def test_temporal_scaling_fixed_si_ratio_oracles(
    method,
    source_dt,
    target_dt,
    data,
    expected,
):
    scaler = TemporalScaling(
        source_resolution=source_dt,
        target_resolution=target_dt,
        method=method,
        source_scale="source",
        target_scale="target",
    )

    result = scaler.resample(
        torch.tensor(data, dtype=torch.float64),
        source_dt=source_dt,
        target_dt=target_dt,
        method=method,
    )

    torch.testing.assert_close(
        result,
        torch.tensor(expected, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert scaler.last_temporal_diagnostics["active_law"]


def test_temporal_scaling_rejects_incomplete_ratio_domain():
    scaler = TemporalScaling(source_resolution=60.0, target_resolution=120.0, method="mean", source_scale="source", target_scale="target")

    with pytest.raises(ValueError, match="complete target intervals"):
        scaler.resample(
            torch.tensor([1.0, 2.0, 3.0]),
            source_dt=60.0,
            target_dt=120.0,
            method=scaler.method,
        )


def test_temporal_scaling_active_law_tracks_semantics():
    scaler = TemporalScaling(source_resolution=1.0, target_resolution=2.0, method="mean", source_scale="source", target_scale="target")
    assert set(scaler.metadata.conservation_laws) == {
        "temporal_observed_integral_consistency"
    }

    scaler.resample(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        source_dt=1.0,
        target_dt=2.0,
        method="sum",
    )

    assert set(scaler.metadata.conservation_laws) == {
        "temporal_interval_accumulation_partition"
    }
    assert scaler.last_temporal_diagnostics["active_law"] == (
        "temporal_interval_accumulation_partition"
    )


def test_temporal_scaling_canonical_call_runs_conservation_hook(monkeypatch):
    scaler = TemporalScaling(source_resolution=60.0, target_resolution=120.0, method="sum", source_scale="source", target_scale="target")
    calls = {"count": 0}
    original_check = scaler.check_conservation_laws

    def tracked_check(input_state, output_updates):
        calls["count"] += 1
        return original_check(input_state, output_updates)

    monkeypatch.setattr(scaler, "check_conservation_laws", tracked_check)
    output = scaler(
        {"time_series": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        time=0.0,
        dt=120.0,
    )

    assert calls["count"] == 1
    torch.testing.assert_close(
        output["time_series_scaled"],
        torch.tensor([3.0, 7.0]),
    )


def test_temporal_scaling_known_residual_is_not_repaired(monkeypatch):
    scaler = TemporalScaling(
        source_resolution=1.0,
        target_resolution=2.0,
        method="sum",
        conservation_level=ConservationLevel.MODERATE,
        q_ref_parameter="tests.temporal.facade_q_ref",
        source_scale="source",
        target_scale="target",
    )
    original_lookup = scaler.get_parameter_value

    def lookup(name, **kwargs):
        if name == "tests.temporal.facade_q_ref":
            return 1.0
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(scaler, "get_parameter_value", lookup)
    original_apply = temporal_core.apply_temporal_kernel

    def perturbed_apply(*args, **kwargs):
        base = original_apply(*args, **kwargs)
        return replace(base, values=base.values + 1.0e-3)

    monkeypatch.setattr(
        temporal_scaling_module,
        "apply_temporal_kernel",
        perturbed_apply,
        raising=False,
    )

    result = scaler.resample(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        source_dt=1.0,
        target_dt=2.0,
        method="sum",
    )

    torch.testing.assert_close(
        result,
        torch.tensor([3.001], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert scaler.last_temporal_diagnostics["conservation_passed"] is False
    assert scaler.last_temporal_diagnostics["conservation_violation"] is True


def test_temporal_scaling_linear_mms_checks_analytic_integral(monkeypatch):
    q_ref_parameter = "tests.temporal.facade_linear_q_ref"
    scaler = TemporalScaling(
        source_resolution=1.0,
        target_resolution=0.5,
        method="rate",
        conservation_level=ConservationLevel.MODERATE,
        q_ref_parameter=q_ref_parameter,
        source_scale="source",
        target_scale="target",
    )
    original_lookup = scaler.get_parameter_value

    def lookup(name, **kwargs):
        if name == q_ref_parameter:
            return 1.0
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(scaler, "get_parameter_value", lookup)
    result = scaler.resample(
        torch.tensor([2.0, 4.0], dtype=torch.float64),
        source_dt=1.0,
        target_dt=0.5,
        method="rate",
    )

    torch.testing.assert_close(
        result,
        torch.tensor([2.0, 2.0, 4.0, 4.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    diagnostics = scaler.last_temporal_diagnostics
    torch.testing.assert_close(
        diagnostics["observed_measure"].sum(),
        torch.tensor(6.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["mapping_residual"],
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 64.0,
    )
    assert diagnostics["conservation_checked"] is True
    assert diagnostics["conservation_passed"] is True
    assert diagnostics["q_ref_provenance"] == q_ref_parameter


@pytest.mark.skipif(
    not device_policy.is_cuda_available(),
    reason="CUDA is required for the registration migration probe",
)
def test_temporal_scaling_registration_migrates_core_helper_to_cuda():
    simulator = RELICSimulator(
        {"device": "cuda:0", "time_step": 1.0, "strict_dependencies": False}
    )
    scaler = TemporalScaling(device=torch.device("cpu"), method="mean", source_scale="source", target_scale="target")

    simulator.register_functor(scaler, execution_group="fast")

    expected = torch.device("cuda:0")
    assert scaler.device == expected
    assert scaler.time_weight_generator.device == expected
    assert scaler.time_weight_generator.cf_processor.device == expected


@pytest.mark.parametrize("scale", ["monthly", "quarterly", "seasonal", "yearly"])
def test_temporal_scaling_ratio_api_rejects_calendar_dependent_scales(scale):
    scaler = TemporalScaling(
        source_scale=scale,
        target_scale="daily",
        source_resolution=1.0,
        target_resolution=1.0,
        method="mean",
    )

    with pytest.raises(ValueError, match="calendar-dependent.*explicit bounds"):
        scaler.resample(torch.tensor([1.0]), source_dt=1.0, target_dt=1.0, method=scaler.method)


def test_temporal_scaling_pattern_type_names_the_executed_reconstruction():
    """Only piecewise-constant reconstruction runs, so only 'constant' is accepted."""
    scaler = TemporalScaling(
        source_resolution=1.0,
        target_resolution=0.5,
        method="mean",
        pattern_type="constant",
        source_scale="source",
        target_scale="target",
    )
    torch.testing.assert_close(
        scaler.resample(torch.tensor([2.0], dtype=torch.float64), method=scaler.method),
        torch.tensor([2.0, 2.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    for pattern in ("sinusoidal", "diurnal", ""):
        with pytest.raises(ValueError, match="pattern_type"):
            TemporalScaling(
                source_resolution=1.0,
                target_resolution=0.5,
                pattern_type=pattern,
                method="mean",
                source_scale="source",
                target_scale="target",
            )


@pytest.mark.parametrize("owner_kind", ["aggregator", "disaggregator", "facade"])
def test_temporal_functors_report_declared_dependencies(owner_kind):
    """add_dependency must reach get_dependencies (the resolver input)."""
    if owner_kind == "aggregator":
        functor = TemporalAggregator(
            TemporalConfig(
                source_frequency="hourly",
                target_frequency="daily",
                conservation_level=ConservationLevel.NONE,
            )
        )
    elif owner_kind == "disaggregator":
        functor = TemporalDisaggregator(
            TemporalConfig(
                source_frequency="daily",
                target_frequency="hourly",
                direction=ScalingDirection.DOWNSCALE,
                conservation_level=ConservationLevel.NONE,
            )
        )
    else:
        functor = TemporalScaling(source_resolution=1.0, target_resolution=2.0, method="mean", source_scale="source", target_scale="target")

    assert functor.get_dependencies() == []
    functor.add_dependency("advection")
    assert functor.get_dependencies() == ["advection"]


def test_temporal_scaling_rejects_unconsumed_keyword_arguments():
    with pytest.raises(TypeError, match="unexpected_option"):
        TemporalScaling(
            source_resolution=1.0,
            target_resolution=2.0,
            unexpected_option=True,
            method="mean",
            source_scale="source",
            target_scale="target",
        )
    for bad in ("", "  ", 3):
        with pytest.raises((TypeError, ValueError), match="q_ref_parameter"):
            TemporalScaling(
                source_resolution=1.0,
                target_resolution=2.0,
                q_ref_parameter=bad,
                method="mean",
                source_scale="source",
                target_scale="target",
            )


@pytest.mark.parametrize("owner_kind", ["aggregator", "disaggregator"])
def test_temporal_scaling_reuses_writer_exclusivity_gate(owner_kind):
    simulator = RELICSimulator(
        {"device": "cpu", "time_step": 1.0, "strict_dependencies": False}
    )
    if owner_kind == "aggregator":
        owner = TemporalAggregator(
            TemporalConfig(
                source_frequency="hourly",
                target_frequency="daily",
                direction=ScalingDirection.UPSCALE,
                spatial_basis=SpatialQuantityBasis.INTENSIVE,
                conservation_level=ConservationLevel.NONE,
                device=torch.device("cpu"),
            ),
            AggregationConfig(),
        )
        shared_output = "temporal_aggregated"
    else:
        owner = TemporalDisaggregator(
            TemporalConfig(
                source_frequency="daily",
                target_frequency="hourly",
                direction=ScalingDirection.DOWNSCALE,
                spatial_basis=SpatialQuantityBasis.INTENSIVE,
                conservation_level=ConservationLevel.NONE,
                device=torch.device("cpu"),
            )
        )
        shared_output = "fine_time_series"
    facade = TemporalScaling(
        name=f"facade_against_{owner_kind}",
        output_variables=[shared_output],
        device=torch.device("cpu"),
        method="mean",
        source_scale="source",
        target_scale="target",
    )
    simulator.register_functor(owner, execution_group="fast")

    with pytest.raises(ValueError, match="Output variable overlap"):
        simulator.register_functor(facade, execution_group="fast")


def test_temporal_scaling_source_has_no_local_formula_or_child_owner():
    source = inspect.getsource(temporal_scaling_module)
    for forbidden in (
        "TemporalAggregator",
        "TemporalDisaggregator",
        "repeat_interleave",
        "flat[-1]",
        ".mean(",
        ".sum(",
    ):
        assert forbidden not in source


def test_temporal_scaling_reads_no_process_division_guard(monkeypatch):
    """The facade reads no validation.numerical key and keeps no unity threshold.

    Its only former use of validation.numerical.epsilon_divide was the unity shortcut
    |target_dt / source_dt - 1| < eps of the removed ratio code. The overlap kernel treats a
    unity ratio exactly (O_ij = delta_ij on coincident fixed-SI bounds, whose partition is
    checked in exact Fraction arithmetic), so no numerical threshold is left to read; a
    scale-operator threshold would be the relative coupling.scales.defaults.epsilon, never the
    additive process-kernel division guard.
    """
    reads = []
    original = TemporalScaling.get_parameter_value

    def recording(self, name, **kwargs):
        reads.append((name, kwargs.get("namespace")))
        return original(self, name, **kwargs)

    monkeypatch.setattr(TemporalScaling, "get_parameter_value", recording)
    scaler = TemporalScaling(source_resolution=1.0, target_resolution=1.0, method="mean", source_scale="source", target_scale="target")
    series = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    scaled = scaler.resample(series, source_dt=1.0, target_dt=1.0, method="mean")

    assert torch.equal(scaled, series)
    assert not hasattr(scaler, "unity_tolerance")
    assert "validation.numerical.epsilon_divide" not in scaler.metadata.parameters
    assert [
        (name, namespace)
        for name, namespace in reads
        if "epsilon" in name or str(namespace or "").startswith("validation.numerical")
    ] == []


def test_spatial_scaling_forward_uses_named_variable():
    scaler = SpatialScaling(
        source_scale="coarse",
        target_scale="fine",
        source_resolution=(1.0, 1.0),
        target_resolution=(0.5, 0.5),
        method="conservative",
        input_variables=["test_field"],
        output_variables=["test_field_scaled"],
    )
    coarse = torch.ones(1, 2, 2)
    output = scaler.forward({"test_field": coarse}, time=0.0, dt=0.0)
    assert "test_field_scaled" in output
    assert output["test_field_scaled"].shape == (1, 4, 4)


def test_regrid_operator_parameter_defaults():
    operator = RegridOperator()
    config = RegridConfig(spatial_basis="intensive", method=RegridMethod.CONSERVATIVE)
    operator._apply_parameter_defaults(config)
    # The one regridder default tolerance (the value every regridder built without one reads).
    expected_tol = operator._get_parameter_manager().get(
        "conservation_tolerance",
        namespace="coupling.scales.defaults",
        required=True,
    )
    assert pytest.approx(config.conservation_tolerance, rel=0.0, abs=1e-12) == expected_tol


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


def test_temporal_scaling_tolerance_is_the_declared_key_and_decides_the_verdict(monkeypatch):
    """The ledger reads the one tolerance key it declares; changing it flips the verdict.

    Summing [1, 2] into 3.001 gives R = 1e-3 over an activity of 3, a relative
    residual of 3.3e-4: a violation at the packaged 1e-6 and a pass at 1e-3.
    """
    q_ref_parameter = "tests.temporal.facade_ledger_q_ref"
    scaler = TemporalScaling(
        source_resolution=1.0,
        target_resolution=2.0,
        method="sum",
        conservation_level=ConservationLevel.MODERATE,
        q_ref_parameter=q_ref_parameter,
        source_scale="source",
        target_scale="target",
    )
    assert _LEDGER_TOLERANCE_KEY in scaler.metadata.parameters
    original_lookup = scaler.get_parameter_value

    def lookup(name, **kwargs):
        if name == q_ref_parameter:
            return 1.0
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(scaler, "get_parameter_value", lookup)
    reads = _record_parameter_reads(scaler, monkeypatch)
    original_apply = temporal_core.apply_temporal_kernel

    def perturbed_apply(*args, **kwargs):
        base = original_apply(*args, **kwargs)
        return replace(base, values=base.values + 1.0e-3)

    monkeypatch.setattr(temporal_scaling_module, "apply_temporal_kernel", perturbed_apply, raising=False)

    def run():
        scaler.resample(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            source_dt=1.0,
            target_dt=2.0,
            method="sum",
        )
        return dict(scaler.last_temporal_diagnostics)

    packaged = run()
    assert [name for name in reads if "tolerance" in name] == [_LEDGER_TOLERANCE_KEY]
    assert packaged["conservation_passed"] is False
    relaxed = _with_ledger_tolerance(1.0e-3, run)
    assert relaxed["conservation_tolerance"] == pytest.approx(1.0e-3)
    assert relaxed["conservation_passed"] is True


def test_scaling_method_parameters_are_required():
    """MA-21 item 203: the method of the spatial and temporal scaling constructors, of every
    regrid and resample override of the coupling package and of RegridConfig has no default;
    every caller states the method it applies."""
    import dataclasses

    from relic.coupling.scales.adapters import ConservativeRemap
    from relic.coupling.scales.spatial.regrid_bilinear import BilinearRegridder
    from relic.coupling.scales.spatial.regrid_conservative import ConservativeRegridder
    from relic.coupling.scales.spatial.regrid_nearest import NearestNeighborRegridder

    callables = {
        "SpatialScaling.__init__": SpatialScaling.__init__,
        "SpatialScaling.regrid": SpatialScaling.regrid,
        "TemporalScaling.__init__": TemporalScaling.__init__,
        "TemporalScaling.resample": TemporalScaling.resample,
        "ConservativeRemap.regrid": ConservativeRemap.regrid,
        "BilinearRegridder.regrid": BilinearRegridder.regrid,
        "ConservativeRegridder.regrid": ConservativeRegridder.regrid,
        "NearestNeighborRegridder.regrid": NearestNeighborRegridder.regrid,
    }
    defaulted = [
        name
        for name, function in callables.items()
        if inspect.signature(function).parameters["method"].default is not inspect.Parameter.empty
    ]
    assert defaulted == []
    method_field = {field.name: field for field in dataclasses.fields(RegridConfig)}["method"]
    assert method_field.default is dataclasses.MISSING
    assert method_field.default_factory is dataclasses.MISSING
    with pytest.raises(TypeError):
        TemporalScaling(source_resolution=60.0, target_resolution=60.0, source_scale="source", target_scale="target")


def test_temporal_scaling_scales_are_required():
    """MA-21 item 205: TemporalScaling states its source and target scale (their names drive the
    calendar-dependent-scale check), so neither has a default."""
    parameters = inspect.signature(TemporalScaling.__init__).parameters
    for name in ("source_scale", "target_scale"):
        assert parameters[name].default is inspect.Parameter.empty, name
    with pytest.raises(TypeError, match="source_scale"):
        TemporalScaling(target_scale="target", method="mean")
    scaler = TemporalScaling(source_scale="source", target_scale="target", method="mean")
    assert (scaler.source_scale, scaler.target_scale) == ("source", "target")

