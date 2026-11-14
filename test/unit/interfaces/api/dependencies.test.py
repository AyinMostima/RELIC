"""Unit tests for API dependency helpers."""

import asyncio
import importlib.util
from pathlib import Path


def _run(coro):
    return asyncio.run(coro)


def test_get_parameter_manager_delegates(monkeypatch):
    from relic.interfaces.api import dependencies as deps

    sentinel = object()
    monkeypatch.setattr(deps, "SERVICE_REGISTRY_AVAILABLE", True)
    monkeypatch.setattr(deps, "_get_parameter_manager", lambda: sentinel)

    assert _run(deps.get_parameter_manager()) is sentinel


def test_get_simulator_uses_request_config_for_isolated_session(monkeypatch):
    from relic.interfaces.api import dependencies as deps

    request = object()
    config = {"time_step": 1.0, "calendar_type": "noleap"}
    simulator = object()
    observed_requests = []
    observed_configs = []

    def _get_config(candidate=None):
        observed_requests.append(candidate)
        return config

    def _get_simulator(config=None):
        observed_configs.append(config)
        return simulator

    monkeypatch.setattr(deps, "SERVICE_REGISTRY_AVAILABLE", True)
    monkeypatch.setattr(deps, "_get_simulator", _get_simulator)
    monkeypatch.setattr(deps, "get_config", _get_config)

    assert _run(deps.get_simulator(request)) is simulator
    assert observed_requests == [request]
    assert observed_configs == [config]


def test_simulator_dependency_uses_fastapi_request_cache():
    from relic.interfaces.api import dependencies as deps

    assert deps.SimulatorDependency.dependency is deps.get_simulator
    assert deps.SimulatorDependency.use_cache is True


def test_get_data_io_manager_delegates(monkeypatch):
    from relic.interfaces.api import dependencies as deps

    sentinel = object()
    monkeypatch.setattr(deps, "SERVICE_REGISTRY_AVAILABLE", True)
    monkeypatch.setattr(deps, "_get_data_io_manager", lambda: sentinel)

    assert _run(deps.get_data_io_manager()) is sentinel


def test_api_dependencies_do_not_expose_unbound_optimization_runtime():
    from relic.interfaces.api import dependencies as deps

    forbidden_names = (
        "_get_optimization_manager",
        "_clear_optimization_cache",
        "get_optimization_manager",
        "OptimizationDependency",
        "cleanup_components",
    )
    for name in forbidden_names:
        assert not hasattr(deps, name)


def test_core_api_dependency_helpers_have_no_dead_cleanup_layer():
    from relic.core.services import api_dependencies as core_deps

    assert not hasattr(core_deps, "cleanup_components")


def test_api_dependencies_do_not_expose_stochastic_manager_shell():
    from relic.interfaces.api import dependencies as deps

    for name in (
        "_get_stochastic_manager",
        "get_stochastic_manager",
        "StochasticDependency",
    ):
        assert not hasattr(deps, name)


def test_stochastic_manager_module_is_absent_and_algorithms_stay_unbound():
    from relic.stochastic import get_available_samplers

    project_root = Path(__file__).resolve().parents[4]
    manager_path = project_root / "relic" / "stochastic" / "manager.py"
    assert not manager_path.exists()
    assert importlib.util.find_spec("relic.stochastic.manager") is None

    samplers = get_available_samplers()
    assert samplers["WeatherGenerator"]["status"] == "[N]"
    assert samplers["WeatherGenerator"]["runtime_ready"] is False
    assert samplers["WeatherGenerator"]["contract"] == "numeric_only"
    # The numeric classes stay numeric-only; their runtime readiness comes
    # from the bound scientific Functor/GraphIR paths, not from a manager shell.
    assert samplers["EulerMaruyamaSolver"]["contract"] == "numeric_only"
    assert samplers["EulerMaruyamaSolver"]["runtime_ready"] is True
    assert samplers["EulerMaruyamaSolver"]["bound_processes"] == [
        "stochastic_air_temperature"
    ]
    assert samplers["GillespieSSA"]["contract"] == "numeric_only"
    assert samplers["GillespieSSA"]["runtime_ready"] is True
    assert samplers["GillespieSSA"]["bound_processes"] == ["stochastic_rainfall"]
