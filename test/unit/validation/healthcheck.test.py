"""Unit tests for HealthChecker benchmark helpers."""

from types import SimpleNamespace

import pytest


def test_run_system_benchmark_includes_memory(monkeypatch):
    from relic.validation.healthcheck import HealthChecker

    checker = HealthChecker({})
    monkeypatch.setattr(checker, "_run_basic_benchmark", lambda: {"status": "completed"})

    dummy_memory = SimpleNamespace(total=1024**3, available=512 * 1024**2, percent=50.0)
    dummy_psutil = SimpleNamespace(virtual_memory=lambda: dummy_memory)
    monkeypatch.setitem(__import__("sys").modules, "psutil", dummy_psutil)

    payload = checker.run_system_benchmark()
    assert payload["benchmark_results"]["status"] == "completed"
    assert payload["memory_info"]["usage_percent"] == 50.0
    assert "total_time" in payload
    assert "timestamp" in payload


def test_stochastic_health_does_not_report_healthy_for_numeric_only_sampler(
    monkeypatch,
):
    from relic.validation.healthcheck import HealthChecker, STATUS_DEGRADED

    checker = HealthChecker({})
    monkeypatch.setattr(
        checker,
        "_import_symbols",
        lambda *_args: (
            lambda: {
                "ParameterDistribution": {
                    "available": True,
                    "error": None,
                },
            },
            lambda: {
                "WeatherGenerator": {
                    "available": True,
                    "error": None,
                    "runtime_ready": True,
                    "status": "[I]",
                },
                "GillespieSSA": {
                    "available": True,
                    "error": None,
                    "contract": "numeric_only",
                    "runtime_ready": False,
                    "status": "[N]",
                },
            },
        ),
    )

    result = checker._check_stochastic_health(detailed=True)

    assert result["status"] == STATUS_DEGRADED
    assert result["components"]["GillespieSSA"] is True
    assert result["runtime"] == {
        "status": "[N]",
        "ready": False,
        "ready_samplers": ["WeatherGenerator"],
        "blocked_samplers": ["GillespieSSA"],
        "unclassified_samplers": [],
    }
    assert any(
        "not production-runtime-ready" in warning
        for warning in checker.warnings
    )


def test_optimization_health_separates_imports_from_runtime_readiness(
    monkeypatch,
):
    from relic.validation.healthcheck import HealthChecker, STATUS_DEGRADED

    checker = HealthChecker({})
    monkeypatch.setattr(
        checker,
        "_import_symbols",
        lambda *_args: (
            lambda: {
                "calibration": {
                    "component_available": True,
                    "error": None,
                    "runtime_ready": False,
                    "status": "[N]",
                    "contract": "algorithm_component_only",
                },
                "sensitivity": {
                    "component_available": True,
                    "error": None,
                    "runtime_ready": "false",
                    "status": "[N]",
                },
                "assimilation": {
                    "component_available": True,
                    "error": None,
                    "runtime_ready": True,
                    "status": "[N]",
                    "contract": "algorithm_component_only",
                },
                "uncertainty": {
                    "component_available": False,
                    "error": "missing component",
                    "runtime_ready": False,
                    "status": "[N]",
                    "contract": "algorithm_component_only",
                },
            },
        ),
    )

    result = checker._check_optimize_health(detailed=True)

    assert result["status"] == STATUS_DEGRADED
    assert result["subsystems"] == {
        "calibration": True,
        "sensitivity": True,
        "assimilation": True,
        "uncertainty": False,
    }
    assert result["available_optimizers"] == [
        "calibration",
        "sensitivity",
        "assimilation",
    ]
    assert result["missing"] == {"uncertainty": "missing component"}
    assert result["runtime"] == {
        "status": "[N]",
        "ready": False,
        "ready_optimizers": [],
        "blocked_optimizers": ["calibration"],
        "unclassified_optimizers": ["sensitivity", "assimilation"],
    }
    assert any(
        "unavailable" in warning
        for warning in checker.warnings
    )
    assert any(
        "not production-runtime-ready" in warning
        for warning in checker.warnings
    )


def test_live_optimization_metadata_is_explicitly_runtime_unbound():
    import relic.optimization as optimization
    from relic.optimization import get_available_optimizers

    expected_items = {
        "calibration": ["ParameterCalibrator"],
        "assimilation": ["EnsembleKalmanFilter", "ThreeDVar"],
        "sensitivity": ["MorrisAnalyzer", "SobolAnalyzer"],
        "uncertainty": ["EnsembleUQ", "BootstrapUQ"],
    }
    expected_requirements = {
        "calibration": [
            "ParameterManager parameter, bounds, transform, and unit binding",
            "sealed GraphIR compiled by compile_graph into an ExecutionPlan",
            "GraphExecutor and Simulator model-evaluation binding",
            "declared objective, observation projection, shape, and unit contract",
            "per-evaluation StateManager, ParameterManager, forcing-cursor, and RNG rollback-replay",
        ],
        "assimilation": [
            "StateManager and Simulator forecast-state binding",
            "sealed GraphIR, ExecutionPlan, and GraphExecutor forecast binding",
            "declared observation operator, variable, shape, unit, domain, and time contract",
            "symmetric positive-semidefinite covariance contract with solver-compatible regularity evidence",
            "atomic analysis-state validation, commit, and rollback",
            "restartable ensemble RNG substream and durable checkpoint binding when stochastic",
        ],
        "sensitivity": [
            "ParameterManager parameter-range, transform, bounds, and unit binding",
            "sealed GraphIR, ExecutionPlan, and GraphExecutor model-evaluation binding",
            "declared output statistic, shape, unit, and aggregation contract",
            "per-sample StateManager, ParameterManager, forcing-cursor, and RNG isolation",
            "method-specific sample-design and estimator-validity evidence",
        ],
        "uncertainty": [
            "ParameterManager-backed marginal distribution, bounds, unit, and correlation binding",
            "sealed GraphIR, ExecutionPlan, and GraphExecutor model-evaluation binding",
            "declared output quantity, shape, unit, and aggregation contract",
            "per-member StateManager, ParameterManager, forcing-cursor, and RNG isolation",
            "restartable ensemble substream and durable checkpoint binding",
            "method-specific sampling and interval-estimator validity evidence",
            "declared observation and residual contract when observation-conditioned",
        ],
    }
    optimizers = get_available_optimizers()
    assert set(optimizers) == set(expected_items)

    for family, info in optimizers.items():
        assert type(info["component_available"]) is bool
        assert info["items"] == expected_items[family]
        assert info["component_available"] is all(
            getattr(optimization, item, None) is not None
            for item in info["items"]
        )
        if info["component_available"]:
            assert info["error"] is None
        else:
            assert isinstance(info["error"], str)
            assert info["error"]
        assert info["runtime_ready"] is False
        assert info["status"] == "[N]"
        assert info["contract"] == "algorithm_component_only"
        assert info["runtime_blocker"] == "runtime_unbound"
        assert info["requirements"] == expected_requirements[family]
        assert "available" not in info
        assert "claim_status" not in info


def test_params_health_is_unhealthy_when_parameter_validation_crashes(monkeypatch):
    from relic.validation.healthcheck import STATUS_UNHEALTHY, HealthChecker

    class _Manager:
        def list(self):
            return ["relic.physical.constants.time.seconds_per_day"]

        def validate_parameters(self):
            raise RuntimeError("validator crashed on malformed bounds")

    checker = HealthChecker({})
    monkeypatch.setattr(
        checker,
        "_import_symbols",
        lambda module_path, *symbols: (lambda preload_all=True: _Manager(),),
    )

    result = checker._check_params_health(detailed=True)

    assert result["status"] == STATUS_UNHEALTHY
    assert result["validation_status"] == "error"
    assert any("validator crashed" in error for error in checker.errors)


def test_own_module_import_failure_points_to_a_code_defect():
    from relic.validation.healthcheck import STATUS_UNHEALTHY, HealthChecker

    checker = HealthChecker({})
    result = checker._handle_check_error(
        ImportError("Module not found: relic.data.io.healthcheck", name="relic.data.io.healthcheck"),
        "Data layer not available",
        "Data health check failed",
    )
    checker.results = {"data": result}
    checker._aggregate_results()

    assert result["status"] == STATUS_UNHEALTHY
    assert "code defect" in result["error"]
    assert not any("Install missing dependencies" in item for item in checker.recommendations)


def test_third_party_import_failure_keeps_the_dependency_advice():
    from relic.validation.healthcheck import STATUS_UNAVAILABLE, HealthChecker

    checker = HealthChecker({})
    result = checker._handle_check_error(
        ImportError("No module named 'netCDF4'", name="netCDF4"),
        "Data layer not available",
        "Data health check failed",
    )
    checker.results = {"data": result}
    checker._aggregate_results()

    assert result["status"] == STATUS_UNAVAILABLE
    assert any("Install missing dependencies for data" in item for item in checker.recommendations)


def test_benchmark_requires_an_explicit_gpu_benchmark_switch():
    import pytest

    from relic.validation.healthcheck import HealthChecker

    with pytest.raises(KeyError, match="enable_gpu_benchmark"):
        HealthChecker({})._run_basic_benchmark()
    with pytest.raises(TypeError, match="enable_gpu_benchmark"):
        HealthChecker({"enable_gpu_benchmark": "yes"})._run_basic_benchmark()


def test_benchmark_skips_the_gpu_part_only_on_an_explicit_false():
    from relic.validation.healthcheck import HealthChecker

    result = HealthChecker({"enable_gpu_benchmark": False})._run_basic_benchmark()

    assert result["gpu_benchmark"]["status"] == "skipped"
    assert result["cpu_benchmark"]["operations_per_second"] > 0


def test_a_subsystem_result_without_status_is_not_counted_as_healthy():
    """HealthChecker builds every subsystem result with a status; a missing one is a defect."""
    import pytest

    from relic.validation.healthcheck import HealthChecker

    checker = HealthChecker({})
    checker.results = {"compute": {"details": {}}}

    with pytest.raises(KeyError, match="status"):
        checker._aggregate_results()


def test_compute_probe_payload_without_overall_status_is_a_failed_check(monkeypatch):
    import relic.computation as computation

    from relic.validation.healthcheck import STATUS_UNHEALTHY, HealthChecker

    monkeypatch.setattr(computation, "healthcheck", lambda verbose=False: {"dependencies": {}})
    result = HealthChecker({})._check_compute_health(detailed=False)

    assert result["status"] == STATUS_UNHEALTHY
    assert "overall" in result["error"]


def test_data_probe_payload_without_overall_status_is_a_defect():
    import pytest

    from relic.validation.healthcheck import HealthChecker

    with pytest.raises(KeyError, match="overall"):
        HealthChecker({})._compose_data_health_result(
            io_summary={},
            streaming_status={"overall": "disabled"},
            streaming_enabled=False,
            detailed=False,
        )


@pytest.mark.parametrize("raw", ["unknown", "bogus"])
def test_unrecognized_raw_status_is_rejected(raw):
    from relic.validation.healthcheck import HealthChecker

    with pytest.raises(ValueError, match=raw):
        HealthChecker({})._normalize_status(raw)
