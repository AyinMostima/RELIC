"""L1 tests for the GPU diagnostic helpers.

Modules under test: relic/computation/gpu/accelerator.py (get_device_info,
GPUProfiler.get_memory_usage, NumericalConsistencyChecker) and the
gpu_healthcheck of relic/computation/gpu/__init__.py.

A failing device or memory query is an error with its cause; it is never
reported as an unknown device or as zero memory, and a consistency check that
cannot compare two results reports why.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

import relic.computation.gpu as gpu_package
import relic.computation.gpu.accelerator as accelerator


def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


def test_invalid_device_string_is_an_error_not_an_unknown_device() -> None:
    with pytest.raises(ValueError, match="cdua:0"):
        accelerator.get_device_info("cdua:0")


def test_failed_device_property_query_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)

    def failing_properties(_device):
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "get_device_properties", failing_properties)

    with pytest.raises(RuntimeError, match="driver query failed"):
        accelerator.get_device_info("cuda:0")


def test_failed_memory_query_is_not_reported_as_zero_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())

    def failing_allocated(*_args, **_kwargs):
        raise RuntimeError("memory query failed")

    monkeypatch.setattr(torch.cuda, "memory_allocated", failing_allocated)

    with pytest.raises(RuntimeError, match="memory query failed"):
        accelerator.GPUProfiler().get_memory_usage(torch.device("cuda", 0))


def test_consistency_check_reports_why_a_comparison_failed() -> None:
    result = accelerator.NumericalConsistencyChecker().check_consistency(
        torch.zeros(2), torch.zeros(3)
    )

    assert result["consistent"] is False
    assert result["failed_keys"] == ["tensor_error"]
    assert result["errors"] and "size" in result["errors"][0]


def test_consistency_check_passes_matching_results_without_errors() -> None:
    result = accelerator.NumericalConsistencyChecker().check_consistency(
        torch.ones(3), torch.ones(3)
    )

    assert result["consistent"] is True
    assert result["errors"] == []


def test_gpu_healthcheck_propagates_a_failed_device_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_package, "is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def failing_info(_device):
        raise RuntimeError("device enumeration failed")

    monkeypatch.setattr(gpu_package, "get_device_info", failing_info)

    with pytest.raises(RuntimeError, match="device enumeration failed"):
        gpu_package.gpu_healthcheck()


def test_consistency_tolerances_are_the_parameter_database_values() -> None:
    """Round 5: computation.gpu.consistency.* is the single source."""
    from relic.parameters import get_global_manager

    checker = accelerator.NumericalConsistencyChecker()
    assert (checker.rtol, checker.atol) == (pytest.approx(1.0e-5), pytest.approx(1.0e-8))
    assert (checker.amp_rtol, checker.amp_atol) == (pytest.approx(1.0e-3), pytest.approx(1.0e-5))
    manager = get_global_manager()
    key = "computation.gpu.consistency.rtol"
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, 2.0e-5)
        assert accelerator.NumericalConsistencyChecker().rtol == pytest.approx(2.0e-5)
        assert accelerator.NumericalConsistencyChecker(rtol=3.0e-5).rtol == pytest.approx(3.0e-5)
    finally:
        manager.set_value(key, original)
