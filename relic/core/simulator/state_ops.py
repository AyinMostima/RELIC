"""State interaction helpers for RELICSimulator."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def set_state_manager(simulator: Any, state_manager: Any) -> None:
    """Set or replace the simulator StateManager."""
    simulator.state_manager = state_manager
    if state_manager is not None:
        target_device = getattr(simulator, "device", None)
        if target_device is not None and hasattr(state_manager, "to_device"):
            state_manager.to_device(target_device)
        state_manager.initialize()
        simulator.state = simulator.state.__class__.INITIALIZED
        simulator._refresh_conservation_ledger()


def get_current_state(simulator: Any) -> dict[str, Any]:
    """
    Get current state from StateManager as single source of truth.

    Adds ``time`` and ``step`` metadata tensors for functor consumption.
    """
    if simulator.state_manager is not None:
        state = cast(dict[str, Any], simulator.state_manager.get_state(clone=False))
        state["time"] = torch.as_tensor(
            simulator.current_time,
            dtype=torch.float32,
            device=simulator.device,
        )
        state["step"] = torch.as_tensor(
            simulator.step_count,
            dtype=torch.int64,
            device=simulator.device,
        )
        if "state_var" not in state:
            placeholder = getattr(simulator, "_state_var_placeholder", None)
            if placeholder is None or not isinstance(placeholder, Tensor):
                placeholder = torch.zeros((), dtype=torch.float32, device=simulator.device)
                simulator._state_var_placeholder = placeholder
            state["state_var"] = placeholder
        return state

    if not simulator._allow_stateless_mode():
        raise RuntimeError("StateManager is required for simulator execution.")

    logger.warning("Getting state without StateManager in explicit explicit stateless mode")
    return {
        "time": torch.tensor(simulator.current_time, device=simulator.device),
        "step": torch.tensor(simulator.step_count, device=simulator.device),
    }


def apply_updates(
    simulator: Any,
    state: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply updates via StateManager transactional path when available.

    Falls back to explicit tensor addition only in approved stateless mode.
    """
    if simulator.state_manager is not None:
        try:
            simulator.state_manager.apply_updates(
                updates,
                dt=simulator.time_step,
                check_conservation=True,
                rollback_on_failure=True,
            )
            return get_current_state(simulator)
        except ValueError as exc:
            logger.error("State update failed: %s", exc)
            raise RuntimeError(f"State update violated constraints: {exc}") from exc

    if not simulator._allow_stateless_mode():
        raise RuntimeError("Cannot apply updates without StateManager.")

    logger.warning("Applying updates without StateManager in explicit explicit stateless mode")
    updated_state = state.copy()
    for var_name, update in updates.items():
        if var_name in updated_state:
            current_state_var = updated_state[var_name]
            if isinstance(current_state_var, Tensor) and isinstance(update, Tensor):
                update = update.to(
                    device=current_state_var.device,
                    dtype=current_state_var.dtype,
                )
            updated_state[var_name] = current_state_var + update
        else:
            updated_state[var_name] = update
    return updated_state


def store_outputs(simulator: Any, outputs: dict[str, Any]) -> None:
    """Store outputs for time-series analysis."""
    stage_timings = getattr(simulator, "_active_step_stage_timings", None)
    for var_name, value in outputs.items():
        if var_name not in simulator.time_series:
            simulator.time_series[var_name] = []
        if isinstance(value, Tensor):
            transfer_started = (
                time.perf_counter() if isinstance(stage_timings, dict) else None
            )
            stored_value = value.detach().cpu()
            if transfer_started is not None:
                stage_timings["cpu_gpu_transfer"] += max(
                    0.0,
                    time.perf_counter() - transfer_started,
                )
        else:
            stored_value = value
        simulator.time_series[var_name].append(stored_value)
