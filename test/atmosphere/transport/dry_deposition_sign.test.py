"""Regression tests for dry deposition flux sign handling."""

import torch

from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionAggregator,
)


def _state() -> dict[str, torch.Tensor]:
    """Build the complete canonical dry-deposition state."""
    return {
        "no_mixing_ratio": torch.tensor([1.0e-8]),
        "no2_mixing_ratio": torch.tensor([2.0e-8]),
        "o3_mixing_ratio": torch.tensor([5.0e-8]),
        "so2_mixing_ratio": torch.tensor([1.0e-9]),
        "pbl_height": torch.tensor([1000.0]),
        "aerodynamic_resistance": torch.tensor([50.0]),
    }


def test_deposition_increment_keeps_negative_sign():
    """Deposition increment must remain non-positive for removal processes."""
    functor = DryDepositionAggregator(name="dry_dep_sign_test")
    output = functor.forward(_state(), time=0.0, dt=60.0)

    for species in functor.target_species:
        increment = output[functor.species_map[species]].value
        assert torch.all(increment <= 0.0)
        assert torch.any(increment < 0.0)
    limiter_residual = output[
        "diagnostics.dry_deposition.total_limiter_residual"
    ]
    assert torch.equal(limiter_residual, torch.zeros_like(limiter_residual))


def test_deposition_step_reduces_mass_consistently():
    """Single-step update should reduce concentration by the integrated sink flux."""
    functor = DryDepositionAggregator(name="dry_dep_mass_test")
    state = _state()
    output = functor.forward(state, time=0.0, dt=600.0)

    total_removed = torch.zeros_like(state["no_mixing_ratio"])
    for species in functor.target_species:
        key = functor.species_map[species]
        delta = output[key].value
        mixing_ratio_new = state[key] + delta
        assert torch.all(mixing_ratio_new < state[key])
        assert torch.all(mixing_ratio_new >= 0.0)
        removed_mass = state[key] - mixing_ratio_new
        assert torch.allclose(
            removed_mass,
            -delta,
            rtol=1.0e-6,
            atol=1.0e-18,
        )
        total_removed = total_removed + removed_mass
    diagnostic_increment = output[
        "diagnostics.dry_deposition.total_mixing_ratio_increment"
    ]
    assert torch.allclose(total_removed, -diagnostic_increment)
    limiter_residual = output[
        "diagnostics.dry_deposition.total_limiter_residual"
    ]
    assert torch.isfinite(limiter_residual).all()
