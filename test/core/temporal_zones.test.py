import inspect
from typing import List

import pytest
import torch

from relic.core.simulator.dependency_ops import TEMPORAL_ZONE_LABELS
from relic.core.simulator.execution_loop import _check_cross_zone_error_transfer
from relic.parameters import get_global_manager


class _DummySimulator:
    def __init__(self) -> None:
        self._groups = {
            "fast": ["fast_functor"],
            "medium": [],
            "slow": ["slow_functor"],
        }
        self._step_functor_increments = {
            "soil_carbon": {
                "fast_functor": torch.tensor([1.0], dtype=torch.float32),
                "slow_functor": torch.tensor([3.0], dtype=torch.float32),
            }
        }
        self._step_update_sources = {
            "soil_carbon": ("slow_functor", "replace"),
        }


def test_zone_labels_mapping() -> None:
    assert TEMPORAL_ZONE_LABELS["fast"] == "core"
    assert TEMPORAL_ZONE_LABELS["medium"] == "physics"
    assert TEMPORAL_ZONE_LABELS["slow"] == "coupling"


def test_cross_zone_error_monitor_disabled_by_default() -> None:
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value("computation.temporal_zones.monitoring_enabled", False)
        result = _check_cross_zone_error_transfer(
            _DummySimulator(),
            {"soil_carbon": torch.tensor([1.0])},
            dt=60.0,
        )
        assert result["enabled"] is False
    finally:
        manager.restore(snapshot)


def test_cross_zone_error_detection() -> None:
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value("computation.temporal_zones.monitoring_enabled", True)
        manager.set_value("computation.temporal_zones.cross_transfer_threshold", 0.6)
        result = _check_cross_zone_error_transfer(
            _DummySimulator(),
            {"soil_carbon": torch.tensor([1.0], dtype=torch.float32)},
            dt=60.0,
        )
        assert result["enabled"] is True
        assert len(result["violations"]) == 1
        assert result["violations"][0]["cross_transfer_ratio"] > 0.6
    finally:
        manager.restore(snapshot)


def test_cross_zone_monitoring_reads_required_yaml_parameters(monkeypatch) -> None:
    """monitoring_enabled and cross_transfer_threshold have no Python fallback defaults."""
    import relic.core.simulator.execution_loop as execution_loop

    source = inspect.getsource(execution_loop._check_cross_zone_error_transfer)
    assert "0.8" not in source
    assert list(inspect.signature(execution_loop._get_temporal_zone_parameter).parameters) == ["key"]

    requested: List[tuple] = []

    class _Manager:
        def __init__(self, values):
            self.values = values

        def get(self, key, *args, **kwargs):
            requested.append((key, kwargs.get("required")))
            if key not in self.values:
                raise KeyError(key)
            return self.values[key]

    monkeypatch.setattr(
        execution_loop, "get_global_manager",
        lambda: _Manager({"computation.temporal_zones.monitoring_enabled": False}),
    )
    simulator = _DummySimulator()
    result = execution_loop._check_cross_zone_error_transfer(simulator, {"test_signal": torch.tensor(1.0)}, 60.0)
    assert result == {"enabled": False, "checked_variables": 0, "violations": []}
    assert requested == [("computation.temporal_zones.monitoring_enabled", True)]

    requested.clear()
    monkeypatch.setattr(
        execution_loop, "get_global_manager",
        lambda: _Manager({"computation.temporal_zones.monitoring_enabled": True}),
    )
    with pytest.raises(KeyError, match="cross_transfer_threshold"):
        execution_loop._check_cross_zone_error_transfer(simulator, {"test_signal": torch.tensor(1.0)}, 60.0)
    assert requested[-1] == ("computation.temporal_zones.cross_transfer_threshold", True)

    monkeypatch.setattr(
        execution_loop, "get_global_manager",
        lambda: _Manager(
            {
                "computation.temporal_zones.monitoring_enabled": True,
                "computation.temporal_zones.cross_transfer_threshold": 1.5,
            }
        ),
    )
    with pytest.raises(ValueError, match="cross_transfer_threshold"):
        execution_loop._check_cross_zone_error_transfer(simulator, {"test_signal": torch.tensor(1.0)}, 60.0)


def test_cross_zone_monitoring_uses_the_yaml_values() -> None:
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    assert manager.get("computation.temporal_zones.monitoring_enabled", required=True) is False
    assert float(
        manager.get("computation.temporal_zones.cross_transfer_threshold", required=True)
    ) == pytest.approx(0.8)
