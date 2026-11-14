"""Unit tests for the canonical pedosphere stability helper surface."""

import pytest
import torch

import relic.scientific.pedosphere.mass_conservation as mass_conservation
from relic.scientific.pedosphere.mass_conservation import (
    NumericalStabilityController,
)


def test_parallel_mass_conservation_checker_is_not_exposed() -> None:
    """StateManager and ConservationGuardian must remain the only budget owners."""
    removed_names = {
        "MassConservationChecker",
        "mass_conservation_checker",
        "get_mass_conservation_checker",
        "check_and_correct_mass_balance",
        "check_comprehensive_mass_balance",
        "log_conservation_summary",
        "get_stability_controller",
        "_LazyInstance",
        "stability_controller",
    }

    for name in removed_names:
        assert not hasattr(mass_conservation, name)
        assert name not in mass_conservation.__all__


def test_limit_fluxes_tensor_profile() -> None:
    """Flux limiting should clamp per-layer tensors without errors."""
    controller = NumericalStabilityController()
    dt = 10.0
    pools = {"nh4_nitrogen": torch.tensor([10.0, 5.0])}
    max_flux = pools["nh4_nitrogen"] * controller.max_flux_fraction / dt
    fluxes = {"nitrification_flux": max_flux * torch.tensor([2.0, -3.0])}

    limited = controller.limit_fluxes(
        fluxes,
        pools,
        dt,
        source_pool_by_flux={"nitrification_flux": "nh4_nitrogen"},
    )
    expected = torch.clamp(
        fluxes["nitrification_flux"],
        min=-max_flux,
        max=max_flux,
    )

    assert torch.allclose(limited["nitrification_flux"], expected)
    assert controller.flux_limiting_events == 1


def test_limit_fluxes_requires_exact_explicit_source_mapping() -> None:
    """Unknown or incomplete donor mappings must fail instead of bypassing."""
    controller = NumericalStabilityController()
    fluxes = {"nitrogen_leaching_flux": torch.tensor(1.0)}
    pools = {"no3_nitrogen": torch.tensor(1.0)}

    with pytest.raises(ValueError, match="exactly match"):
        controller.limit_fluxes(
            fluxes,
            pools,
            dt=10.0,
            source_pool_by_flux={},
        )

    with pytest.raises(KeyError, match="missing_pool"):
        controller.limit_fluxes(
            fluxes,
            pools,
            dt=10.0,
            source_pool_by_flux={"nitrogen_leaching_flux": "missing_pool"},
        )

    assert not hasattr(controller, "_identify_source_pool")


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_limit_fluxes_rejects_invalid_dt(dt: float) -> None:
    """Flux limiting requires one positive finite integration interval."""
    controller = NumericalStabilityController()

    with pytest.raises(ValueError, match="positive and finite"):
        controller.limit_fluxes(
            {"nitrogen_leaching_flux": torch.tensor(1.0)},
            {"no3_nitrogen": torch.tensor(1.0)},
            dt=dt,
            source_pool_by_flux={"nitrogen_leaching_flux": "no3_nitrogen"},
        )


def test_enforce_pool_bounds_tensor() -> None:
    """Pool bounds should clamp negative tensor values."""
    controller = NumericalStabilityController()
    pools = {"NH4": torch.tensor([-1.0, 0.5])}

    bounded = controller.enforce_pool_bounds(pools)

    assert torch.all(bounded["NH4"] >= controller.min_pool_size)


def test_calculate_stable_timestep_uses_global_maximum_across_grid() -> None:
    """Stable timestep calculation must reduce across all tensor elements, not just index 0."""
    controller = NumericalStabilityController(
        cfl_number=1.0,
        min_time_step=1e-6,
        max_time_step=10.0,
    )
    dt_max = controller.calculate_stable_timestep(
        {"fast_grid": torch.tensor([1.0, 1000.0])},
        {"pool": torch.tensor([1.0, 1.0])},
    )
    assert dt_max == pytest.approx(0.001)

