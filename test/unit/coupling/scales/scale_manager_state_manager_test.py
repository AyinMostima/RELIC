import pytest

from relic.core.managers.base import ManagerState
from relic.core.managers.exceptions import ManagerInitializationError
from relic.core.state import StateManager
import relic.coupling.scales as scales_package
import relic.coupling.scales.scale_manager as scale_manager_module
from relic.coupling.scales.physical_constraint_manager import PhysicalConstraintManager
from relic.coupling.scales.scale_manager import ScaleManager
from relic.parameters import get_global_manager


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


def test_set_state_manager_is_idempotent():
    state_manager = StateManager()
    manager = ScaleManager(parameter_manager=get_global_manager())

    manager.set_state_manager(state_manager)
    assert manager.state_manager is state_manager
    assert manager._init_state_manager is state_manager

    manager.initialize()
    manager.configure({})
    manager.validate()

    # Idempotent when the same instance is provided after initialization; the audit collector
    # keeps its records itself and holds no StateManager (no audit state synchronization).
    manager.set_state_manager(state_manager)
    assert manager.state_manager is state_manager
    assert not hasattr(manager.audit_collector, "state_manager")

    with pytest.raises(RuntimeError):
        manager.set_state_manager(StateManager())


def test_initialize_requires_explicit_state_manager_owner():
    manager = ScaleManager(parameter_manager=get_global_manager())

    with pytest.raises(ManagerInitializationError, match="explicit StateManager"):
        manager.initialize()

    assert manager.state_manager is None


def test_scale_package_exposes_no_global_owner_bypass():
    assert not hasattr(scales_package, "get_global_scale_manager")
    assert not hasattr(scales_package, "set_global_scale_manager")


def test_scale_manager_module_exposes_no_global_owner_bypass():
    assert not hasattr(scale_manager_module, "_global_scale_manager")
    assert not hasattr(scale_manager_module, "get_global_scale_manager")
    assert not hasattr(scale_manager_module, "set_global_scale_manager")
    assert not hasattr(ScaleManager, "get_instance")
    assert not hasattr(ScaleManager, "set_global_manager")


def test_initialize_reuses_state_manager_constraint_owner_without_lifecycle_calls(
    monkeypatch,
):
    parameter_manager = get_global_manager()
    constraint_manager = PhysicalConstraintManager(parameter_manager=parameter_manager)
    constraint_manager.initialize()
    constraint_manager.validate()
    state_manager = StateManager(physical_constraint_manager=constraint_manager)
    manager = ScaleManager(
        parameter_manager=parameter_manager,
        state_manager=state_manager,
    )

    def forbidden_lifecycle_call(*args, **kwargs):
        raise AssertionError("ScaleManager must not own the constraint-manager lifecycle")

    monkeypatch.setattr(constraint_manager, "initialize", forbidden_lifecycle_call)
    monkeypatch.setattr(constraint_manager, "configure", forbidden_lifecycle_call)
    monkeypatch.setattr(constraint_manager, "validate", forbidden_lifecycle_call)

    manager.initialize()

    assert manager.constraint_manager is constraint_manager
    assert manager.constraint_manager is state_manager.physical_constraint_manager


def test_initialize_fails_closed_when_state_constraint_owner_is_missing(monkeypatch):
    state_manager = StateManager()
    monkeypatch.setattr(
        state_manager,
        "_get_physical_constraint_manager",
        lambda: None,
    )
    manager = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=state_manager,
    )

    with pytest.raises(ManagerInitializationError, match="constraint owner"):
        manager.initialize()

    assert manager.constraint_manager is None
    assert manager.state == ManagerState.ERROR


def test_initialize_fails_closed_when_state_constraint_owner_is_not_ready():
    parameter_manager = get_global_manager()
    constraint_manager = PhysicalConstraintManager(parameter_manager=parameter_manager)
    state_manager = StateManager(physical_constraint_manager=constraint_manager)
    manager = ScaleManager(
        parameter_manager=parameter_manager,
        state_manager=state_manager,
    )

    with pytest.raises(ManagerInitializationError, match="READY"):
        manager.initialize()

    assert constraint_manager.state == ManagerState.UNINITIALIZED
    assert manager.state == ManagerState.ERROR


def test_state_manager_binding_is_typed_idempotent_and_frozen_after_start():
    parameter_manager = get_global_manager()
    state_manager = StateManager()
    manager = ScaleManager(parameter_manager=parameter_manager)

    with pytest.raises(TypeError, match="StateManager"):
        manager.set_state_manager(object())

    manager.set_state_manager(state_manager)
    manager.set_state_manager(state_manager)
    manager.initialize()

    with pytest.raises(RuntimeError, match="cannot replace"):
        manager.set_state_manager(StateManager())

    assert manager.state == ManagerState.INITIALIZING
    assert manager.state_manager is state_manager
