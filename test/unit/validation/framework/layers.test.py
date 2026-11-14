from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from relic.validation.framework.layers import (
    BENCHMARK_LAYERS,
    CORE_LAYERS,
    DEFAULT_LAYER_ORDER,
    L1_UNIT,
    L2_INTEGRATION,
    L3_BENCHMARK,
    L4_SCIENTIFIC,
    LAYER_METADATA,
    SCIENTIFIC_LAYERS,
    LayerExecutionStrategy,
    get_all_layers,
    get_execution_strategy,
    get_layer_metadata,
    get_layers_for_domains,
    validate_layer_dependencies,
)
from relic.validation.framework.models import (
    ValidationCode,
    ValidationContext,
    ValidationLevel,
    ValidationResultModel,
)


def _result(
    level: ValidationLevel = ValidationLevel.INFO,
    *,
    layer: str = L2_INTEGRATION,
    scope: str = "test.rule",
) -> ValidationResultModel:
    code = (
        ValidationCode.GENERIC_VALIDATION_OK
        if level == ValidationLevel.INFO
        else ValidationCode.GENERIC_VALIDATION_FAILED
    )
    return ValidationResultModel(
        level=level,
        code=code,
        message=f"{level.value} result",
        layer=layer,
        scope=scope,
    )


def test_layer_metadata_and_dependency_order_are_canonical() -> None:
    assert get_all_layers() == [
        L1_UNIT,
        L2_INTEGRATION,
        L3_BENCHMARK,
        L4_SCIENTIFIC,
    ]
    assert DEFAULT_LAYER_ORDER == get_all_layers()
    assert CORE_LAYERS == [L1_UNIT, L2_INTEGRATION]
    assert BENCHMARK_LAYERS == [L3_BENCHMARK]
    assert SCIENTIFIC_LAYERS == [L4_SCIENTIFIC]
    assert get_layer_metadata(L2_INTEGRATION).failure_handling == (
        "fail_fast_on_critical"
    )
    assert validate_layer_dependencies(L4_SCIENTIFIC) == [
        L1_UNIT,
        L2_INTEGRATION,
        L3_BENCHMARK,
    ]


def test_unknown_layer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown validation layer"):
        get_layer_metadata("L9")


def test_execution_strategy_dispatches_all_declared_layers() -> None:
    assert get_execution_strategy(L1_UNIT) is LayerExecutionStrategy.parallel
    assert (
        get_execution_strategy(L2_INTEGRATION)
        is LayerExecutionStrategy.dependency_ordered
    )
    assert (
        get_execution_strategy(L3_BENCHMARK)
        is LayerExecutionStrategy.sequential_with_data_dependency
    )
    assert (
        get_execution_strategy(L4_SCIENTIFIC)
        is LayerExecutionStrategy.data_driven_sequential
    )


def test_unknown_execution_strategy_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        LAYER_METADATA[L1_UNIT],
        "execution_strategy",
        "unsupported",
    )

    with pytest.raises(ValueError, match="Unknown execution strategy"):
        get_execution_strategy(L1_UNIT)


def test_dependency_lists_are_defensive_copies() -> None:
    dependencies = validate_layer_dependencies(L2_INTEGRATION)
    dependencies.append("caller_mutation")

    assert validate_layer_dependencies(L2_INTEGRATION) == [L1_UNIT]


def test_domain_filter_deduplicates_inputs_and_ignores_unknown_domains() -> None:
    selected = get_layers_for_domains(
        ["parameters", "parameters", "conservation", "unknown"]
    )

    assert selected == {
        L1_UNIT: ["parameters"],
        L2_INTEGRATION: ["conservation"],
    }


def test_l1_strategy_flattens_results_and_normalizes_exceptions() -> None:
    context = ValidationContext(execution_layer=L1_UNIT)

    def list_rule(_context: ValidationContext):
        return [_result(layer=L1_UNIT, scope="list.first")]

    def scalar_rule(_context: ValidationContext):
        return _result(layer=L1_UNIT, scope="scalar")

    def failing_rule(_context: ValidationContext):
        raise ValueError("invalid parameter")

    results = LayerExecutionStrategy.parallel(
        [list_rule, scalar_rule, failing_rule],
        context,
    )

    assert [result.scope for result in results] == [
        "list.first",
        "scalar",
        "failing_rule",
    ]
    assert results[-1].level == ValidationLevel.ERROR
    assert results[-1].tags == ["execution_error"]


def test_l2_strategy_stops_after_critical_result() -> None:
    calls: list[str] = []
    context = ValidationContext(execution_layer=L2_INTEGRATION)

    def critical_rule(_context: ValidationContext):
        calls.append("critical")
        return _result(ValidationLevel.CRITICAL, scope="critical")

    def must_not_run(_context: ValidationContext):
        calls.append("after")
        return _result(scope="after")

    results = LayerExecutionStrategy.dependency_ordered(
        [critical_rule, must_not_run],
        context,
    )

    assert calls == ["critical"]
    assert [result.scope for result in results] == ["critical"]


def test_l2_strategy_stops_after_rule_exception() -> None:
    calls: list[str] = []
    context = ValidationContext(execution_layer=L2_INTEGRATION)

    def failing_rule(_context: ValidationContext):
        calls.append("failing")
        raise RuntimeError("broken interface")

    def must_not_run(_context: ValidationContext):
        calls.append("after")
        return _result(scope="after")

    results = LayerExecutionStrategy.dependency_ordered(
        [failing_rule, must_not_run],
        context,
    )

    assert calls == ["failing"]
    assert len(results) == 1
    assert results[0].level == ValidationLevel.CRITICAL
    assert results[0].scope == "failing_rule"


def test_l3_strategy_skips_missing_data_and_executes_when_available() -> None:
    calls: list[str] = []

    def benchmark_rule(_context: ValidationContext):
        calls.append("benchmark")
        return _result(layer=L3_BENCHMARK, scope="benchmark")

    benchmark_rule._required_data = ["reference_dataset"]

    missing_results = (
        LayerExecutionStrategy.sequential_with_data_dependency(
            [benchmark_rule],
            ValidationContext(execution_layer=L3_BENCHMARK, metadata={}),
        )
    )
    available_results = (
        LayerExecutionStrategy.sequential_with_data_dependency(
            [benchmark_rule],
            ValidationContext(
                execution_layer=L3_BENCHMARK,
                metadata={"reference_dataset": object()},
            ),
        )
    )

    assert calls == ["benchmark"]
    assert missing_results[0].code == ValidationCode.MISSING_REQUIRED_DATA
    assert missing_results[0].level == ValidationLevel.WARNING
    assert "not applicable (domain absent)" in missing_results[0].message
    assert missing_results[0].tags == ["data_dependency", "not_applicable"]
    assert available_results[0].scope == "benchmark"


@pytest.mark.parametrize(
    ("strategy", "expected_level", "expected_tag"),
    [
        (
            LayerExecutionStrategy.sequential_with_data_dependency,
            ValidationLevel.ERROR,
            "benchmark",
        ),
        (
            LayerExecutionStrategy.data_driven_sequential,
            ValidationLevel.ERROR,
            "scientific_review",
        ),
    ],
)
def test_sequential_strategies_normalize_rule_exceptions(
    strategy: Callable[[list[Callable[..., Any]], Any], list[Any]],
    expected_level: ValidationLevel,
    expected_tag: str,
) -> None:
    def failing_rule(_context: ValidationContext):
        raise OSError("data unavailable")

    results = strategy(
        [failing_rule],
        ValidationContext(execution_layer=L3_BENCHMARK),
    )

    assert len(results) == 1
    assert results[0].level == expected_level
    assert results[0].passed is False
    assert expected_tag in results[0].tags


def test_l3_strategy_partial_required_data_is_a_failed_result() -> None:
    """Some but not all required data present: the rule is not evaluable (ERROR)."""
    calls: list[str] = []

    def benchmark_rule(_context: ValidationContext):
        calls.append("benchmark")
        return _result(layer=L3_BENCHMARK, scope="benchmark")

    benchmark_rule._required_data = ["reference_dataset", "reference_units"]

    results = LayerExecutionStrategy.sequential_with_data_dependency(
        [benchmark_rule],
        ValidationContext(
            execution_layer=L3_BENCHMARK,
            metadata={"reference_dataset": object()},
        ),
    )

    assert calls == []
    assert results[0].code == ValidationCode.MISSING_REQUIRED_DATA
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "reference_units" in results[0].message
