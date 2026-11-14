"""L1 tests for simulator StateManager/default guardrails."""

import pytest

from relic.core.simulator import RELICSimulator, SimulationState
from relic.core.state import StateManager


def test_state_manager_has_no_runtime_alias_sync_registry() -> None:
    """State aliases are resolved by ContractRegistry, not callback side channels."""
    assert not hasattr(StateManager, "register_alias_sync_handler")
    assert not hasattr(StateManager, "_alias_sync_handlers")


def test_get_current_state_raises_without_state_manager_when_test_default_disabled():
    """Simulator must fail fast when state manager default is disabled."""
    simulator = RELICSimulator(
        config={
            "time_step": 60.0,
            "allow_state_manager_test_default": False,
        }
    )
    simulator.state = SimulationState.INITIALIZED

    with pytest.raises(RuntimeError, match="StateManager is required"):
        simulator._get_current_state()


def test_initialize_state_blocks_default_module_registration_without_opt_in(monkeypatch):
    """default module registration should be blocked without explicit opt-in."""
    simulator = RELICSimulator(
        config={
            "time_step": 60.0,
            "enable_bootstrap_modules": True,
        }
    )

    monkeypatch.setattr(simulator, "_allow_bootstrap_module_registration", lambda: False)

    with pytest.raises(RuntimeError, match="enable_bootstrap_modules is test-only"):
        simulator.initialize_state({"temperature": 295.0})




_GUARD_STATE_UNITS = {"guard_x": "dimensionless"}


def _initialized_with(**settings):
    simulator = RELICSimulator(
        config={"time_step": 60.0, "state_units": _GUARD_STATE_UNITS, **settings}
    )
    simulator.initialize_state({"guard_x": 0.0})
    return simulator


@pytest.mark.parametrize(
    ("setting", "value", "named"),
    [
        ("conservation_tolerance", "abc", "'abc'"),
        ("conservation_tolerance", float("nan"), "nan"),
        ("conservation_tolerance", -1.0e-6, "-1e-06"),
        ("conservation_tolerance", 0.0, "0.0"),
        ("conservation_tolerance", 1.0, "1.0"),
        ("conservation_tolerance", True, "True"),
        ("conservation_raise_on_violation", "false", "'false'"),
        ("conservation_rollback_on_failure", "false", "'false'"),
        ("conservation_raise_on_violation", 0, "0"),
        ("conservation_rollback_on_failure", 1, "1"),
        ("conservation_checks", "false", "'false'"),
        ("conservation_checks", 0, "0"),
        ("conservation_checks", {"enabled": True, "tolerence": 1.0e-6}, "tolerence"),
        ("conservation_checks", {"enabled": "yes"}, "'yes'"),
        ("conservation_checks", {"tolerance": float("inf")}, "inf"),
    ],
)
def test_invalid_conservation_setting_is_rejected_naming_the_key_and_value(setting, value, named):
    """No silent fallback and no bool() coercion: "false" is not True, "abc" is not ignored."""
    with pytest.raises(ValueError, match=setting) as error:
        _initialized_with(**{setting: value})

    assert named in str(error.value)


def test_flat_conservation_setting_contradicting_the_mapping_is_rejected():
    with pytest.raises(ValueError, match="conservation_raise_on_violation") as error:
        _initialized_with(
            conservation_checks={"raise_on_violation": False},
            conservation_raise_on_violation=True,
        )

    assert "conservation_checks.raise_on_violation" in str(error.value)


def test_valid_conservation_settings_reach_the_state_manager():
    simulator = _initialized_with(
        conservation_checks={"enabled": True, "raise_on_violation": False},
        conservation_rollback_on_failure=True,
        conservation_tolerance=1.0e-5,
    )

    state_manager = simulator.state_manager
    assert state_manager._conservation_enabled is True
    assert state_manager._conservation_raise_on_violation is False
    assert state_manager._conservation_rollback_default is True
    assert state_manager._abs_conservation_tol == 1.0e-5
