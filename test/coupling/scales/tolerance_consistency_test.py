import inspect
import re

import pytest
import torch
from unittest.mock import Mock

from relic.core.managers.base import ManagerState
from relic.core.managers.exceptions import ManagerInitializationError
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.core.policy import get_precision_policy, ScalingMode
from relic.core.state import StateManager
from relic.parameters import get_global_manager

# The one source of the ScaleManager facade ledger tolerance.
_PRECISION_KEY = "coupling.scales.defaults.precision.{mode}.tolerance"


def _mock_parameter_manager(overrides=None):
    pm = Mock()
    prefix = "coupling.scales.defaults.precision"
    mapping = {
        "coupling.defaults.device": "cpu",
        "coupling.scales.spatial.coord_names": {"x": "lon", "y": "lat"},
        "coupling.scales.defaults.physical_constraint_method": "clip",
        "coupling.scales.defaults.preserve_gradients": True,
        "coupling.scales.defaults.constraint_device": "cpu",
        f"{prefix}.conservative.tolerance": 1e-9,
        f"{prefix}.conservative.relative_tolerance": 1e-6,
        f"{prefix}.conservative.absolute_tolerance": 1e-12,
        f"{prefix}.conservative.max_iterations": 50,
        f"{prefix}.conservative.dtype": "float32",
        f"{prefix}.conservative.enforce_conservation": True,
        f"{prefix}.strict_conservative.tolerance": 1e-12,
        f"{prefix}.strict_conservative.relative_tolerance": 1e-8,
        f"{prefix}.strict_conservative.absolute_tolerance": 1e-14,
        f"{prefix}.strict_conservative.max_iterations": 100,
        f"{prefix}.strict_conservative.dtype": "float64",
        f"{prefix}.strict_conservative.enforce_conservation": True,
        f"{prefix}.balanced.tolerance": 1e-6,
        f"{prefix}.balanced.relative_tolerance": 1e-6,
        f"{prefix}.balanced.absolute_tolerance": 1e-8,
        f"{prefix}.balanced.max_iterations": 100,
        f"{prefix}.balanced.dtype": "float32",
        f"{prefix}.balanced.enforce_conservation": False,
        f"{prefix}.fast_approx.tolerance": 1e-3,
        f"{prefix}.fast_approx.relative_tolerance": 1e-3,
        f"{prefix}.fast_approx.absolute_tolerance": 1e-5,
        f"{prefix}.fast_approx.max_iterations": 50,
        f"{prefix}.fast_approx.dtype": "float32",
        f"{prefix}.fast_approx.enforce_conservation": False,
    }
    mapping.update(overrides or {})

    def _get(key, default=None, namespace=None, required=False, **kwargs):
        full_key = f"{namespace}.{key}" if namespace and not str(key).startswith(str(namespace)) else str(key)
        if full_key in mapping:
            return mapping[full_key]
        if required:
            raise KeyError(full_key)
        return default

    pm.get.side_effect = _get
    return pm


def test_manager_tolerance_alignment():
    """ledger_tolerance(mode) is the precision tolerance of that mode; set_mode moves the default."""
    pm = _mock_parameter_manager()
    manager = ScaleManager(
        default_mode=ScalingMode.CONSERVATIVE,
        parameter_manager=pm,
        state_manager=StateManager(),
    )
    manager.initialize()

    for mode in ScalingMode:
        assert manager.ledger_tolerance(mode) == get_precision_policy(mode, parameter_manager=pm).tolerance
    assert manager.ledger_tolerance() == manager.conservation_tolerance == pytest.approx(1e-9)

    manager.set_mode(ScalingMode.STRICT_CONSERVATIVE)
    assert manager.ledger_tolerance() == manager.conservation_tolerance == pytest.approx(1e-12)


# Two hillslopes of 4 m2, each covering two patches (cell totals: the ledger is exact).
_TOPOLOGY = {
    "n_source_units": 4,
    "n_target_units": 2,
    "parent_map": {0: 0, 1: 0, 2: 1, 3: 1},
    "source_areas": [1.0, 3.0, 2.0, 2.0],
    "target_areas": [4.0, 4.0],
}


def _governed_manager() -> ScaleManager:
    manager = ScaleManager(
        default_mode=ScalingMode.CONSERVATIVE,
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    manager.initialize()
    return manager


class TestOneFacadeToleranceSource:
    """The facade ledger tolerance has one source: coupling.scales.defaults.precision.<mode>.tolerance."""

    def test_scale_manager_keeps_no_second_tolerance(self):
        assert "conservation_tolerance" not in inspect.signature(ScaleManager.__init__).parameters
        static = inspect.getattr_static(ScaleManager, "conservation_tolerance")
        assert isinstance(static, property) and static.fset is None
        assert not hasattr(ScaleManager, "_get_conservation_config")
        manager = ScaleManager(parameter_manager=get_global_manager(), state_manager=StateManager())
        # Undefined before initialize(): the simulator reads it with a None default.
        assert getattr(manager, "conservation_tolerance", None) is None
        manager.initialize()
        for name in ("tolerance_map", "_default_conservation_tolerance", "_init_conservation_tolerance"):
            assert not hasattr(manager, name), name
        assert not hasattr(manager.conservation_enforcer, "tolerance")
        with pytest.raises(AttributeError):
            manager.conservation_tolerance = 1.0e-4

    def test_configure_rejects_every_conservation_tolerance(self):
        """The ledger tolerance is not a configuration setting: every value, the one in force
        included, is rejected with the key that sets it; the ledger keeps using that key."""
        key = _PRECISION_KEY.format(mode="conservative")
        manager = _governed_manager()
        in_force = manager.conservation_tolerance
        assert in_force == get_precision_policy(ScalingMode.CONSERVATIVE).tolerance
        for rejected in (in_force, 1.0e-4, float("nan"), None):
            other = _governed_manager()
            with pytest.raises(ValueError, match=re.escape(key)):
                other.configure({"conservation_tolerance": rejected})
            # An input rejection, not a lifecycle failure: the manager stays configurable.
            assert other.state == ManagerState.INITIALIZING and other.error_info is None
            assert other.conservation_tolerance == in_force
        # The ledger of a configured manager uses the precision tolerance of its mode.
        manager.configure({"enforce_conservation": True})
        manager.validate()
        _, audit = manager.aggregate(
            torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
            "patch",
            "hillslope",
            variables=["litter_mass"],
            metadata={"litter_mass": {"spatial_basis": "cell_total", "units": "kg"}},
            topology=dict(_TOPOLOGY),
        )
        assert audit.metadata["conservation_audit"]["litter_mass"].tolerance == in_force

    def test_nan_precision_tolerance_raises_instead_of_falling_back(self):
        key = _PRECISION_KEY.format(mode="conservative")
        pm = _mock_parameter_manager(
            {
                key: float("nan"),
                # The former fallback: it must not replace an invalid policy tolerance.
                "coupling.scales.defaults.conservation_tolerance": 1.0e-8,
            }
        )
        manager = ScaleManager(
            default_mode=ScalingMode.CONSERVATIVE,
            parameter_manager=pm,
            state_manager=StateManager(),
        )
        with pytest.raises(ManagerInitializationError, match=re.escape(key)):
            manager.initialize()
