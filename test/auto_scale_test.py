"""
Auto scale coupling smoke tests.

These tests validate that RELICSimulator can operate with auto-scale
functionality enabled and handles edge conditions without raising errors.
"""

import torch

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager


def _build_simulator(enable_auto_scale: bool = True) -> RELICSimulator:
    """Construct a simulator instance with optional auto-scale enabled."""
    config = {
        "time_step": 60.0,
        "max_time": 3600.0,
        "enable_auto_scale": enable_auto_scale,
        "enable_bootstrap_modules": True,
    }
    simulator = RELICSimulator(config)
    simulator.set_state_manager(StateManager())
    return simulator


def test_auto_scale_enabled_functionality() -> None:
    """Ensure auto-scale path initializes and returns consistency metadata."""
    simulator = _build_simulator(True)

    assert simulator.enable_auto_scale is True
    assert simulator.scale_manager is not None

    mixed_scale_state = {
        "soil_moisture": torch.tensor([0.3, 0.4, 0.2, 0.5, 0.1, 0.3]),
        "basin_runoff": torch.tensor([1.5, 2.0]),
    }
    scale_check = simulator._check_scale_consistency(mixed_scale_state)

    assert isinstance(scale_check, dict)
    assert "needs_regrid" in scale_check


def test_scale_check_edge_cases() -> None:
    """Cover empty state, tensor-only state, and mixed-type inputs."""
    simulator = _build_simulator(True)

    empty_state = {}
    check_empty = simulator._check_scale_consistency(empty_state)
    assert "needs_regrid" in check_empty

    tensor_only_state = {
        "var1": torch.tensor([1.0, 2.0, 3.0]),
        "var2": torch.tensor(4.0),
    }
    check_tensor = simulator._check_scale_consistency(tensor_only_state)
    assert isinstance(check_tensor.get("detected_scales", {}), dict)

    mixed_type_state = {
        "tensor_var": torch.tensor([1.0, 2.0]),
        "float_var": 3.14,
        "int_var": 42,
        "string_var": "test",
    }
    check_mixed = simulator._check_scale_consistency(mixed_type_state)
    assert check_mixed["needs_regrid"] in {True, False}


def test_regridding_edge_cases() -> None:
    """Validate regridding handler behaves safely for invalid requests."""
    simulator = _build_simulator(True)
    state = {"var": torch.tensor([1.0, 2.0, 3.0])}

    no_regrid_check = {"needs_regrid": False, "reason": "single scale"}
    assert simulator._perform_regridding(state, no_regrid_check) is state

    invalid_regrid_check = {
        "needs_regrid": True,
        "target_scale": None,
        "detected_scales": {"var": "patch"},
    }
    assert simulator._perform_regridding(state, invalid_regrid_check) is state


