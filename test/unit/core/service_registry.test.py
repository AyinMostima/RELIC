"""Unit tests for core service registry helpers."""

def test_get_parameter_manager_delegates_to_global_singleton(monkeypatch):
    from relic.core import service_registry
    import relic.parameters

    sentinel = object()
    call_counter = {"count": 0}

    def _mock_get_global_manager():
        call_counter["count"] += 1
        return sentinel

    monkeypatch.setattr(relic.parameters, "get_global_manager", _mock_get_global_manager)

    service_registry.clear_service_cache()
    assert service_registry.get_parameter_manager() is sentinel
    assert service_registry.get_parameter_manager() is sentinel
    assert call_counter["count"] == 2


def test_get_simulator_delegates_to_canonical_factory(monkeypatch):
    from relic.core import service_registry
    import relic.core.simulator as simulator_module
    import relic.core.state as state_manager_module

    class DummySimulator:
        def __init__(self, config=None):
            self.config = config
            self.state_manager = None

    calls = []

    def _create_simulator(config):
        calls.append(config)
        return DummySimulator(config)

    def _forbid_state_manager_construction(*args, **kwargs):
        raise AssertionError("service_registry must not construct StateManager")

    monkeypatch.setattr(simulator_module, "create_simulator", _create_simulator)
    monkeypatch.setattr(
        state_manager_module,
        "StateManager",
        _forbid_state_manager_construction,
    )

    service_registry.clear_service_cache()
    config = {"time_step": 1.0}
    simulator = service_registry.get_simulator(config)

    assert isinstance(simulator, DummySimulator)
    assert simulator.state_manager is None
    assert calls == [config]
    assert "simulator" not in service_registry._COMPONENT_CACHE


def test_get_simulator_returns_isolated_sessions_for_same_config(monkeypatch):
    from relic.core import service_registry
    import relic.core.simulator as simulator_module

    class DummySimulator:
        def __init__(self, config):
            self.config = config
            self.runtime_state = {}

    monkeypatch.setattr(
        simulator_module,
        "create_simulator",
        lambda config: DummySimulator(config),
    )

    service_registry.clear_service_cache()
    config = {"time_step": 60.0, "device": "cpu"}
    first = service_registry.get_simulator(config)
    first.runtime_state["audit_marker"] = "first-session"
    second = service_registry.get_simulator(config)

    assert second is not first
    assert second.runtime_state == {}


def test_get_simulator_never_reuses_none_or_different_configs(monkeypatch):
    from relic.core import service_registry
    import relic.core.simulator as simulator_module

    created = []

    def _create_simulator(config):
        simulator = object()
        created.append((config, simulator))
        return simulator

    monkeypatch.setattr(simulator_module, "create_simulator", _create_simulator)

    service_registry.clear_service_cache()
    default_first = service_registry.get_simulator()
    default_second = service_registry.get_simulator()
    configured = service_registry.get_simulator({"time_step": 120.0})

    assert default_first is not default_second
    assert configured is not default_first
    assert configured is not default_second
    assert [config for config, _ in created] == [None, None, {"time_step": 120.0}]


def test_get_data_io_manager_cached(monkeypatch):
    from relic.core import service_registry
    import relic.data.io as data_io_module

    class DummyDataIO:
        pass

    monkeypatch.setattr(data_io_module, "DataIO", DummyDataIO)

    service_registry.clear_service_cache()
    manager = service_registry.get_data_io_manager()
    assert isinstance(manager, DummyDataIO)
    assert service_registry.get_data_io_manager() is manager
