"""L1 tests for StateManager checkpoint and restore APIs."""

from __future__ import annotations

import pytest
import torch

from relic.core.state import StateManager


def test_checkpoint_restore_roundtrip() -> None:
    manager = StateManager()
    manager.initialize_state({"time": torch.tensor(1.0), "soil_moisture": torch.tensor([0.3])})

    snapshot = manager.checkpoint()
    manager.set_state("time", torch.tensor(9.0), validate=False)
    manager.set_state("soil_moisture", torch.tensor([0.1]), validate=False)

    manager.restore(snapshot)
    restored = manager.get_current_state()
    manager_device = torch.device(manager.device)
    assert restored["time"].device.type == manager_device.type
    assert restored["soil_moisture"].device.type == manager_device.type
    if manager_device.index is not None:
        assert restored["time"].device.index == manager_device.index
        assert restored["soil_moisture"].device.index == manager_device.index
    assert torch.allclose(restored["time"], torch.tensor(1.0, device=restored["time"].device))
    assert torch.allclose(
        restored["soil_moisture"],
        torch.tensor([0.3], device=restored["soil_moisture"].device),
    )


def test_checkpoint_snapshot_isolation() -> None:
    manager = StateManager()
    manager.initialize_state({"time": torch.tensor(2.0)})

    snapshot = manager.checkpoint()
    manager.set_state("time", torch.tensor(5.0), validate=False)

    snap_state = snapshot["state"]
    assert isinstance(snap_state, dict)
    assert torch.allclose(snap_state["time"], torch.tensor(2.0, device=snap_state["time"].device))


def test_checkpoint_restores_transaction_diagnostics_and_histories() -> None:
    manager = StateManager()
    manager.initialize_state({"time": torch.tensor(2.0)})
    manager._history = [{"time": torch.tensor(1.0)}]
    manager._current_diagnostics = {"module.metric": 3.0}
    manager._clip_loss_diagnostics = {"carbon": 0.25}
    manager.generate_conservation_report({}, current_time=1.0)
    manager._current_diagnostics = {"module.metric": 3.0}
    manager.set_auto_report_enabled(False)

    snapshot = manager.checkpoint()
    manager._history[0]["time"].fill_(9.0)
    manager._conservation_history.clear()
    manager._last_summary = None
    manager._current_diagnostics.clear()
    manager._clip_loss_diagnostics.clear()
    manager.set_auto_report_enabled(True)

    manager.restore(snapshot)

    assert torch.equal(manager._history[0]["time"], torch.tensor(1.0))
    assert len(manager.conservation_history) == 1
    assert manager._last_summary is not None
    assert manager._current_diagnostics == {"module.metric": 3.0}
    assert manager._clip_loss_diagnostics == {"carbon": 0.25}
    assert manager._enable_auto_report is False


@pytest.mark.parametrize("payload", [None, {}, {"state": "invalid"}])
def test_restore_rejects_invalid_payload(payload) -> None:
    manager = StateManager()
    manager.initialize_state({"time": torch.tensor(0.0)})

    with pytest.raises((TypeError, ValueError)):
        manager.restore(payload)  # type: ignore[arg-type]
