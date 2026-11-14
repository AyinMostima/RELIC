"""ScaleManager lifecycle contract of RELICSimulator.

The ScaleManager needs the simulator's StateManager owner (and its READY
PhysicalConstraintManager). Its lifecycle therefore runs only once that owner
exists: ``initialize(state_manager, ...)`` with a StateManager,
``initialize_state(...)``, or lazily before the first regridding. Without an
owner the simulator raises its own error before calling the ScaleManager, and
a lifecycle failure propagates instead of silently switching auto-scale off.
"""

import pytest
import torch

from relic.core.managers.base import ManagerState
from relic.core.simulator import DependencyGraph, RELICSimulator
from relic.core.state import StateManager


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'tracer': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class FailingScaleManager:
    def __init__(self):
        self.is_ready = False
        self.enforce_conservation = True
        self.conservation_tolerance = 1e-6
        self.cf_strict_mode = False
        self.state_manager = None

    def set_state_manager(self, state_manager):
        self.state_manager = state_manager

    def reset(self):
        return None

    def initialize(self):
        raise RuntimeError("init failure")

    def configure(self, *_args, **_kwargs):
        return None

    def validate(self):
        return None


def _auto_scale_simulator(**config) -> RELICSimulator:
    return _declared_simulator(
        {"enable_auto_scale": True, "device": "cpu", "time_step": 60.0, **config}
    )


def test_scale_manager_lifecycle_failure_propagates_without_disabling_autoscale():
    simulator = _declared_simulator()
    failing = FailingScaleManager()
    simulator.scale_manager = failing
    simulator.enable_auto_scale = True

    with pytest.raises(RuntimeError, match="init failure"):
        simulator.initialize(StateManager(), DependencyGraph())

    assert simulator.enable_auto_scale is True
    assert simulator.scale_manager is failing


def test_ensure_without_state_manager_raises_the_simulator_error_first():
    simulator = _auto_scale_simulator()
    scale_manager = simulator.scale_manager
    assert simulator.state_manager is None

    with pytest.raises(RuntimeError, match="requires the simulator's StateManager owner"):
        simulator._ensure_scale_manager_ready()

    assert scale_manager.state == ManagerState.UNINITIALIZED
    assert simulator.enable_auto_scale is True
    assert simulator.scale_manager is scale_manager


def test_initialize_without_state_manager_defers_the_lifecycle():
    simulator = _auto_scale_simulator()
    scale_manager = simulator.scale_manager

    simulator.initialize(None, DependencyGraph())
    assert scale_manager.state == ManagerState.UNINITIALIZED

    simulator.initialize_state({"tracer": torch.ones(4)})
    assert scale_manager.is_ready
    assert scale_manager.state_manager is simulator.state_manager


def test_initialize_state_first_binds_and_readies_the_scale_manager():
    simulator = _auto_scale_simulator()

    simulator.initialize_state({"tracer": torch.ones(4)})

    assert simulator.enable_auto_scale is True
    assert simulator.scale_manager.is_ready
    assert simulator.scale_manager.state_manager is simulator.state_manager


def test_step_first_binds_and_readies_the_scale_manager():
    simulator = _auto_scale_simulator(allow_uninitialized_step=True)

    simulator.step()

    assert simulator.state_manager is not None
    assert simulator.scale_manager.is_ready
    assert simulator.scale_manager.state_manager is simulator.state_manager


def test_step_first_without_bootstrap_is_refused_before_the_scale_manager():
    simulator = _auto_scale_simulator()

    with pytest.raises(RuntimeError, match="Cannot step simulation"):
        simulator.step()

    assert simulator.scale_manager.state == ManagerState.UNINITIALIZED


class _ExplodingScaleManager:
    def __init__(self, **_kwargs):
        raise RuntimeError("scale manager construction failed")


def test_scale_manager_construction_failure_raises_instead_of_disabling(monkeypatch):
    import relic.core.simulator.main as simulator_main

    monkeypatch.setattr(simulator_main, "ScaleManager", _ExplodingScaleManager)

    with pytest.raises(RuntimeError, match="ScaleManager.*construction failed") as error:
        _auto_scale_simulator()
    assert "scale manager construction failed" in str(error.value.__cause__)


def test_unavailable_scale_manager_with_auto_scale_requested_raises(monkeypatch):
    import relic.core.simulator as simulator_package

    monkeypatch.setattr(simulator_package, "HAS_SCALE_MANAGER", False)

    with pytest.raises(RuntimeError, match="enable_auto_scale=True.*unavailable"):
        _auto_scale_simulator()


def test_invalid_auto_scale_mode_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="not_a_mode"):
        _auto_scale_simulator(auto_scale={"mode": "not_a_mode"})


def test_auto_scale_off_builds_no_scale_manager(monkeypatch):
    import relic.core.simulator.main as simulator_main

    monkeypatch.setattr(simulator_main, "ScaleManager", _ExplodingScaleManager)
    simulator = _declared_simulator({"enable_auto_scale": False, "device": "cpu"})

    assert simulator.enable_auto_scale is False
    assert simulator.scale_manager is None


class _RecordingScaleManager:
    """Lifecycle stub that records every configure payload the simulator sends."""

    def __init__(self):
        self.is_ready = False
        self.enforce_conservation = True
        self.conservation_tolerance = 1e-9
        self.cf_strict_mode = False
        self.state_manager = None
        self.configured = []

    def set_state_manager(self, state_manager):
        self.state_manager = state_manager

    def reset(self):
        self.is_ready = False

    def initialize(self):
        return None

    def configure(self, payload):
        self.configured.append(dict(payload))

    def validate(self):
        self.is_ready = True


def test_simulator_configures_no_ledger_tolerance_on_the_scale_manager():
    """The facade ledger tolerance has one source, the precision policy; the
    simulator's configuration carries only enforce_conservation and cf_strict_mode."""
    simulator = _declared_simulator()
    recording = _RecordingScaleManager()
    simulator.scale_manager = recording
    simulator.enable_auto_scale = True

    simulator.initialize(StateManager(), DependencyGraph())

    assert recording.configured == [{"enforce_conservation": True, "cf_strict_mode": False}]


def test_ledger_tolerance_is_the_precision_policy_value_after_a_configure_round_trip(monkeypatch):
    from relic.parameters import get_global_manager

    simulator = _auto_scale_simulator()
    simulator.initialize_state({"tracer": torch.ones(4)})
    scale_manager = simulator.scale_manager
    key = f"coupling.scales.defaults.precision.{scale_manager.default_mode.value}.tolerance"
    expected = float(get_global_manager().get(key, required=True))
    assert scale_manager.conservation_tolerance == expected

    payloads = []
    original_configure = scale_manager.configure

    def recording_configure(config):
        payloads.append(dict(config))
        return original_configure(config)

    monkeypatch.setattr(scale_manager, "configure", recording_configure)
    scale_manager.reset()
    simulator._ensure_scale_manager_ready()

    assert scale_manager.is_ready
    assert scale_manager.conservation_tolerance == expected
    assert payloads and all("conservation_tolerance" not in payload for payload in payloads)
