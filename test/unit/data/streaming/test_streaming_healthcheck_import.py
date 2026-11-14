"""Regression tests for streaming integration healthcheck import paths."""

from __future__ import annotations

import importlib

from relic.data.streaming.healthcheck import HealthStatus, StreamingHealthChecker


def test_temporal_aggregator_module_path_is_singular() -> None:
    """Ensure the canonical temporal aggregator module is importable."""
    module = importlib.import_module("relic.coupling.scales.temporal.aggregator")
    assert hasattr(module, "TemporalAggregator")


def test_streaming_integration_no_plural_aggregators_import_error() -> None:
    """Regression guard for historical typo: temporal.aggregators."""
    checker = StreamingHealthChecker()
    result = checker.check_streaming_integration()

    if result.status == HealthStatus.ERROR:
        assert "temporal.aggregators" not in result.message
        assert "temporal.aggregators" not in str(result.details.get("error", ""))


def test_streaming_benchmark_errors_fail_the_throughput_check(monkeypatch):
    """A benchmark that raises makes the check an error instead of a low-throughput measurement."""
    import asyncio

    from relic.data.streaming import healthcheck as streaming_healthcheck

    if not streaming_healthcheck.STREAMING_ENABLED:
        import pytest

        pytest.skip("streaming is disabled in this environment")
    checker = streaming_healthcheck.StreamingHealthChecker()

    async def failing_throughput():
        raise OSError("benchmark directory not writable")

    monkeypatch.setattr(checker, "_benchmark_streaming_throughput", failing_throughput)
    result = asyncio.run(checker.check_streaming_throughput())
    assert result.status == streaming_healthcheck.HealthStatus.ERROR
    assert "not writable" in result.message
