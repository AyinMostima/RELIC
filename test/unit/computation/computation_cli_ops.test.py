"""Tests for computation CLI runtime capability gates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import relic.computation as computation
from relic.computation import cli_ops


class _ProgressReporter:
    def __init__(self, *args, **kwargs) -> None:
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def update(self, *args) -> None:
        self.updates.append(args)


def _io_recorder():
    messages = {
        "success": [],
        "error": [],
        "warning": [],
        "info": [],
        "header": [],
        "saved": [],
    }

    def _append(name):
        return lambda message, *args, **kwargs: messages[name].append(str(message))

    return messages, {
        "print_success": _append("success"),
        "print_error": _append("error"),
        "print_warning": _append("warning"),
        "print_info": _append("info"),
        "print_header": _append("header"),
        "save_output": lambda *args, **kwargs: messages["saved"].append(args),
        "ProgressReporter": _ProgressReporter,
        "check_experimental_feature": lambda *_args, **_kwargs: True,
    }


def _failed_compile_health() -> dict:
    return {
        "overall": "degraded",
        "dependencies": {
            "torch": True,
            "torch_compile": True,
            "torch_jit": True,
            "cuda": False,
            "distributed": False,
            "amp": False,
            "parallel": True,
        },
        "capabilities": {
            "torch_compile": {
                "runtime_ready": False,
                "probe_status": "failed",
                "blocker": {
                    "phase": "first_call",
                    "type": "InvalidCxxCompiler",
                    "message": "cl.exe is unavailable",
                },
            }
        },
        "warnings": [],
        "recommendations": [],
    }


def test_benchmark_jit_request_requires_runtime_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_healthcheck(**kwargs):
        calls.append(kwargs)
        return _failed_compile_health()

    monkeypatch.setattr(computation, "healthcheck", fake_healthcheck)
    messages, io = _io_recorder()
    args = SimpleNamespace(
        target="photosynthesis",
        device="cpu",
        batch_size=2,
        n_points=4,
        jit=True,
        output=None,
    )

    assert cli_ops.run_benchmark(args, {"computation": {}}, io) is False
    assert calls == [
        {
            "runtime_probe": True,
            "compile_backend": "inductor",
            "compile_device": "cpu",
        }
    ]
    assert any("InvalidCxxCompiler" in message for message in messages["error"])


def test_verbose_health_reports_compile_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_healthcheck(**kwargs):
        calls.append(kwargs)
        return _failed_compile_health()

    monkeypatch.setattr(computation, "healthcheck", fake_healthcheck)
    messages, io = _io_recorder()
    args = SimpleNamespace(verbose=True, output=None, format="json")

    assert cli_ops.run_health_check(args, {}, io) is True
    assert calls == [{"verbose": True, "runtime_probe": True}]
    assert any("JIT Compilation" in message for message in messages["error"])
    assert not any("JIT Compilation" in message for message in messages["success"])


def test_verbose_health_fails_when_gpu_memory_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed GPU memory query makes the health command fail, not just print."""
    health = _failed_compile_health()
    health["overall"] = "healthy"
    health["dependencies"]["cuda"] = True
    health["dependencies"]["cuda_devices"] = [{"name": "probe", "memory": 1}]
    monkeypatch.setattr(computation, "healthcheck", lambda **_kwargs: health)

    import relic.computation.gpu as gpu_package

    def failing_memory(*_args, **_kwargs):
        raise RuntimeError("memory query failed")

    monkeypatch.setattr(gpu_package, "get_memory_usage", failing_memory)
    messages, io = _io_recorder()
    args = SimpleNamespace(verbose=True, output=None, format="json")

    assert cli_ops.run_health_check(args, {}, io) is False
    assert any("memory query failed" in message for message in messages["error"])


# ---------------------------------------------------------------------------
# Round 5: the interface layer (relic.interfaces.cli.compute) resolves every
# diagnostic setting (command line or parameter database); cli_ops reads the
# namespace as given and keeps no second default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["series_length", "dt", "frequency"])
def test_dynamical_benchmark_has_no_second_default(missing: str) -> None:
    values = {"series_length": 64, "dt": 60.0, "frequency": 1.0e-3, "output": None}
    values.pop(missing)
    _, io = _io_recorder()
    with pytest.raises(AttributeError, match=missing):
        cli_ops.run_dynamical_benchmark(SimpleNamespace(**values), {}, io)


def test_splitting_sensitivity_has_no_second_default() -> None:
    simulator = SimpleNamespace(
        get_committed_provider_view=lambda: SimpleNamespace(order_fast=("a", "b")),
        get_state=lambda: {},
    )
    _, io = _io_recorder()
    args = SimpleNamespace(group_order="", dt=60.0, output=None)
    with pytest.raises(AttributeError, match="n_permutations"):
        cli_ops.run_splitting_sensitivity(args, {"simulator": simulator}, io)


# ---------------------------------------------------------------------------
# Round 6: the structure-function lag range is read from the parameter
# database; the splitting scan reads the fast order from the committed
# provider view (public API) and the state through get_state().
# ---------------------------------------------------------------------------


def test_dynamical_benchmark_structure_lags_come_from_the_parameter_database(monkeypatch) -> None:
    import relic.computation.benchmarks as benchmarks
    from relic.parameters import get_global_manager

    seen = {}
    original_function = benchmarks.compute_structure_function

    def recording(field, max_lag, order):
        seen["max_lag"] = max_lag
        return original_function(field, max_lag=max_lag, order=order)

    monkeypatch.setattr(benchmarks, "compute_structure_function", recording)
    _, io = _io_recorder()
    args = SimpleNamespace(series_length=64, dt=60.0, frequency=1.0e-3, output=None)
    assert cli_ops.run_dynamical_benchmark(args, {}, io) is True
    assert seen["max_lag"] == 16  # min(50, 64 // 4)
    manager = get_global_manager()
    key = "computation.diagnostics.dynamical_benchmark.structure_function_max_lag"
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, 10)
        assert cli_ops.run_dynamical_benchmark(args, {}, io) is True
        assert seen["max_lag"] == 10  # min(10, 64 // 4)
    finally:
        manager.set_value(key, original)


def test_splitting_sensitivity_reads_the_committed_provider_view(monkeypatch) -> None:
    import relic.core.simulator.execution_loop as execution_loop

    seen = {}

    def fake_scan(simulator, state, dt, group_order, n_permutations):
        seen.update(group_order=group_order, state=state)
        return {"permutations_tested": 1, "commutativity_violation": 0.0}

    monkeypatch.setattr(execution_loop, "compute_splitting_sensitivity", fake_scan)
    simulator = SimpleNamespace(
        get_committed_provider_view=lambda: SimpleNamespace(order_fast=("a", "b")),
        get_state=lambda: {"x": 1},
    )
    _, io = _io_recorder()
    args = SimpleNamespace(group_order="", dt=60.0, n_permutations=2, output=None)
    assert cli_ops.run_splitting_sensitivity(args, {"simulator": simulator}, io) is True
    assert seen == {"group_order": ["a", "b"], "state": {"x": 1}}


def test_splitting_sensitivity_without_a_committed_view_is_an_error() -> None:
    simulator = SimpleNamespace(get_committed_provider_view=lambda: None, get_state=lambda: {})
    messages, io = _io_recorder()
    args = SimpleNamespace(group_order="", dt=60.0, n_permutations=2, output=None)
    assert cli_ops.run_splitting_sensitivity(args, {"simulator": simulator}, io) is False
    assert any("provider view" in message for message in messages["error"])
