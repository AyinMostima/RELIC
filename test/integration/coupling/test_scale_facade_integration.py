import pytest

from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.core import ScalingMode
from relic.coupling.core.policy import get_precision_policy
from relic.core.state import StateManager


def test_scale_manager_uses_per_mode_tolerances():
    manager = ScaleManager(state_manager=StateManager())
    manager.initialize()

    # Every scaling mode has its own ledger tolerance, read from its precision policy
    for mode in ScalingMode:
        assert manager.ledger_tolerance(mode) == get_precision_policy(mode).tolerance

    # Ensure strict/fast tolerances are not collapsed to a single value
    tol_strict = manager.ledger_tolerance(ScalingMode.STRICT_CONSERVATIVE)
    tol_balanced = manager.ledger_tolerance(ScalingMode.BALANCED)
    tol_fast = manager.ledger_tolerance(ScalingMode.FAST_APPROX)
    assert tol_strict < tol_balanced < tol_fast


def test_scale_manager_propagates_conservation_tolerance_from_policy():
    manager = ScaleManager(state_manager=StateManager())
    manager.initialize()
    default_policy = get_precision_policy(manager.default_mode)

    # The manager reports the default policy tolerance; the enforcer keeps none of its own
    assert not hasattr(manager.conservation_enforcer, "tolerance")
    assert manager.conservation_tolerance == pytest.approx(default_policy.tolerance)

