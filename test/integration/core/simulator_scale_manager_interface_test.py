from relic.core.simulator import RELICSimulator, DependencyGraph
from relic.core.state import StateManager


def test_simulator_uses_scale_manager_setter(monkeypatch):
    # RELICSimulator() is the empty configuration (enable_auto_scale False); the
    # scale manager is requested explicitly.
    simulator = RELICSimulator({"enable_auto_scale": True})
    state_manager = StateManager()
    dependency_graph = DependencyGraph()

    scale_manager = simulator.scale_manager
    assert scale_manager is not None

    calls = {}
    original_setter = scale_manager.set_state_manager

    def tracked_setter(state_manager_arg):
        calls["state_manager"] = state_manager_arg
        return original_setter(state_manager_arg)

    monkeypatch.setattr(scale_manager, "set_state_manager", tracked_setter)

    simulator.initialize(state_manager, dependency_graph)

    assert calls.get("state_manager") is state_manager
    assert simulator.scale_manager.state_manager is state_manager
