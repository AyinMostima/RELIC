"""Invariant tests for update mode semantics.

References:
- ESMF 8.9.1 NUOPC Reference Documentation (field exchange semantics)
- LeVeque (2002), finite-volume time integration

Mathematical expressions:
- FLUX with rate basis: delta = value * dt (dt is mandatory; no silent amount fallback)
- FLUX with per_timestep payload: delta = value (already integrated, dt not needed)
- ACCUMULATE: x_new = x_old + delta
- REPLACE: x_new = value

Parameter meanings:
- value: update payload
- dt: model timestep in seconds
- x_old/x_new: state variable before/after update
"""

from __future__ import annotations

import pytest
import torch

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def _build_state_manager(initial: float = 0.0) -> StateManager:
    schema = {
        "water_storage": StateVariableSpec(
            name="water_storage",
            units="kg m-2",
            conserved=True,
            required=False,
        )
    }
    manager = StateManager(schema=schema)
    manager.set_state({"water_storage": torch.tensor(float(initial), dtype=torch.float32)})
    return manager


def test_flux_with_dt_scaling() -> None:
    manager = _build_state_manager(initial=0.0)
    updates = {
        "water_storage": UpdateItem(
            value=torch.tensor(2.0, dtype=torch.float32),
            mode=UpdateMode.FLUX,
            per_timestep=False,
        )
    }
    manager.apply_updates(updates, dt=60.0, check_conservation=False)
    assert pytest.approx(manager.get_state("water_storage").item(), rel=1e-6) == 120.0


def test_flux_pre_integrated() -> None:
    manager = _build_state_manager(initial=0.0)
    updates = {
        "water_storage": UpdateItem(
            value=torch.tensor(2.0, dtype=torch.float32),
            mode=UpdateMode.FLUX,
            per_timestep=True,
        )
    }
    manager.apply_updates(updates, dt=60.0, check_conservation=False)
    assert pytest.approx(manager.get_state("water_storage").item(), rel=1e-6) == 2.0


def test_flux_without_dt_is_refused() -> None:
    """An unintegrated rate cannot be applied without dt: fail fast, keep the state."""
    manager = _build_state_manager(initial=0.0)
    updates = {
        "water_storage": UpdateItem(
            value=torch.tensor(2.0, dtype=torch.float32),
            mode=UpdateMode.FLUX,
            per_timestep=False,
        )
    }
    with pytest.raises(ValueError, match="dt is required for an unintegrated state rate"):
        manager.apply_updates(updates, dt=None, check_conservation=False)
    assert pytest.approx(manager.get_state("water_storage").item(), rel=1e-6) == 0.0


def test_pre_integrated_flux_without_dt() -> None:
    """A per_timestep payload is an integrated amount and is applied without dt."""
    manager = _build_state_manager(initial=0.0)
    updates = {
        "water_storage": UpdateItem(
            value=torch.tensor(2.0, dtype=torch.float32),
            mode=UpdateMode.FLUX,
            per_timestep=True,
        )
    }
    manager.apply_updates(updates, dt=None, check_conservation=False)
    assert pytest.approx(manager.get_state("water_storage").item(), rel=1e-6) == 2.0


def test_accumulate_unchanged() -> None:
    manager = _build_state_manager(initial=1.0)
    manager.apply_updates(
        {"water_storage": UpdateItem(value=3.0, mode=UpdateMode.ACCUMULATE)},
        dt=60.0,
        check_conservation=False,
    )
    assert pytest.approx(manager.get_state("water_storage").item(), rel=1e-6) == 4.0


def test_replace_unchanged() -> None:
    manager = _build_state_manager(initial=5.0)
    manager.apply_updates(
        {"water_storage": UpdateItem(value=3.0, mode=UpdateMode.REPLACE)},
        dt=60.0,
        check_conservation=False,
    )
    assert pytest.approx(manager.get_state("water_storage").item(), rel=1e-6) == 3.0


def test_conservation_with_flux(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _build_state_manager(initial=10.0)
    captured = {}

    def _capture_updates(self, updates, dt=None, scale_factor=None):  # type: ignore[no-untyped-def]
        captured.update(updates)
        return {}

    monkeypatch.setattr(StateManager, "_check_conservation", _capture_updates, raising=True)
    manager.apply_updates(
        {
            "water_storage": UpdateItem(
                value=torch.tensor(2.0, dtype=torch.float32),
                mode=UpdateMode.FLUX,
                per_timestep=False,
            )
        },
        dt=30.0,
        check_conservation=True,
    )
    assert "water_storage" in captured
    assert pytest.approx(captured["water_storage"].item(), rel=1e-6) == 60.0


def test_backward_compat_basis() -> None:
    manager = _build_state_manager(initial=0.0)
    manager.apply_updates(
        {
            "water_storage": {
                "value": torch.tensor(2.0, dtype=torch.float32),
                "mode": "flux",
                "basis": "per_step",
            }
        },
        dt=60.0,
        check_conservation=False,
    )
    assert pytest.approx(manager.get_state("water_storage").item(), rel=1e-6) == 2.0
