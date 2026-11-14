"""Behavioral tests for computation capability and runtime health probes."""

from __future__ import annotations

import pytest
import torch

import relic.computation as computation


def test_dependency_check_does_not_execute_compile_or_torchscript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch,
        "compile",
        lambda *_args, **_kwargs: pytest.fail("torch.compile must not run at import health"),
    )
    monkeypatch.setattr(
        torch.jit,
        "script",
        lambda *_args, **_kwargs: pytest.fail("torch.jit.script must not run at import health"),
    )

    status = computation._check_dependencies()

    assert status["torch"] is True
    assert status["torch_compile"] is True
    assert status["torch_jit"] is True


def test_unprobed_compile_capability_is_not_runtime_ready() -> None:
    status = computation.healthcheck(runtime_probe=False)

    compile_status = status["capabilities"]["torch_compile"]
    assert compile_status["probe_status"] == "not_run"
    assert compile_status["runtime_ready"] is None
    assert compile_status["first_call_attempted"] is False
    assert compile_status["probe_scope"] == (
        "non_scientific_single_function_current_environment"
    )
    assert compile_status["scientific_kernel_ready"] is False
    assert compile_status["scientific_equivalence"] is False


def test_is_available_does_not_promote_unprobed_compile_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(computation._DEPENDENCIES, "torch_compile", True)

    assert computation.is_available("torch_compile") is False
    assert computation.is_available("compile") is False


def test_compile_probe_executes_returned_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"constructed": 0, "executed": 0}

    def fake_maybe_compile(func, **kwargs):
        calls["constructed"] += 1

        def compiled(value):
            calls["executed"] += 1
            return func(value)

        return compiled

    monkeypatch.setattr(computation, "maybe_compile", fake_maybe_compile)
    monkeypatch.setitem(computation._DEPENDENCIES, "torch_compile", True)

    status = computation.healthcheck(
        runtime_probe=True,
        compile_backend="inductor",
        compile_device="cpu",
    )

    compile_status = status["capabilities"]["torch_compile"]
    assert calls == {"constructed": 1, "executed": 1}
    assert compile_status["probe_status"] == "passed"
    assert compile_status["runtime_ready"] is True
    assert compile_status["first_call_completed"] is True
    assert compile_status["fallback_used"] is False
    assert compile_status["probe_scope"] == (
        "non_scientific_single_function_current_environment"
    )
    assert compile_status["scientific_kernel_ready"] is False
    assert compile_status["scientific_equivalence"] is False


def test_compile_probe_records_first_call_blocker_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_maybe_compile(_func, **_kwargs):
        def compiled(_value):
            raise RuntimeError("first call failed")

        return compiled

    monkeypatch.setattr(computation, "maybe_compile", fake_maybe_compile)
    monkeypatch.setitem(computation._DEPENDENCIES, "torch_compile", True)

    status = computation.healthcheck(runtime_probe=True)

    compile_status = status["capabilities"]["torch_compile"]
    assert status["overall"] == "degraded"
    assert compile_status["probe_status"] == "failed"
    assert compile_status["runtime_ready"] is False
    assert compile_status["fallback_used"] is False
    assert compile_status["blocker"]["phase"] == "first_call"
    assert compile_status["blocker"]["type"] == "RuntimeError"
    assert "first call failed" in compile_status["blocker"]["message"]


def test_compile_probe_records_construction_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_construction(*_args, **_kwargs):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(computation, "maybe_compile", fail_construction)
    monkeypatch.setitem(computation._DEPENDENCIES, "torch_compile", True)

    status = computation.healthcheck(runtime_probe=True)

    compile_status = status["capabilities"]["torch_compile"]
    assert compile_status["probe_status"] == "failed"
    assert compile_status["first_call_attempted"] is False
    assert compile_status["blocker"]["phase"] == "construction"


def test_dependency_probe_failure_propagates_instead_of_disabling_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed version probe is not reported as 'PyTorch unavailable'."""
    real_import = computation.importlib.import_module

    def failing_import(name, *args, **kwargs):
        if name == "packaging.version":
            raise ImportError("packaging is broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(computation.importlib, "import_module", failing_import)

    with pytest.raises(ImportError, match="packaging is broken"):
        computation._check_dependencies()
