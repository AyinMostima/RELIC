"""Tests for bounded data I/O health reporting."""

from relic.data.io.healthcheck import HealthStatus, IOHealthChecker


def test_cf_health_reports_only_single_variable_metadata_subset():
    checker = IOHealthChecker()

    assert not hasattr(checker, "check_cf_compliance")
    result = checker.check_cf_metadata_subset()

    assert result.status is HealthStatus.HEALTHY
    assert result.name == "cf_metadata_subset"
    assert "single-variable" in result.message
    assert "fully supported" not in result.message
    assert result.details["validation_scope"] == (
        "single_variable_standard_name_and_units"
    )
    assert result.details["dataset_conformance_checked"] is False
    assert result.details["coordinates_checked"] is False
    assert result.details["cell_methods_checked"] is False


def test_performance_status_is_an_error_when_a_benchmark_fails(monkeypatch):
    """A failed read or write benchmark makes the performance check fail, whatever the other throughput."""
    from relic.data.io.healthcheck import HealthStatus, IOHealthChecker

    checker = IOHealthChecker()

    def failing_read():
        raise OSError("read benchmark failed")

    monkeypatch.setattr(checker, "_benchmark_read_performance", failing_read)
    monkeypatch.setattr(checker, "_benchmark_write_performance", lambda: {"throughput_mbps": 1000.0})
    monkeypatch.setattr(checker, "_benchmark_memory_usage", lambda: {"efficiency_score": 1.0})
    result = checker.check_performance()
    assert result.status == HealthStatus.ERROR
    assert "read benchmark failed" in result.details["read_performance"]["error"]
