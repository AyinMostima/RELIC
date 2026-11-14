"""
Gradient flow contract test for NitrogenLeachingProcess.

This test guards against float-casting requires_grad parameters, which would
break differentiable calibration and sensitivity workflows.
"""

from __future__ import annotations

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import (
    NitrogenLeachingProcess,
)


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _as_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def test_nitrate_distribution_coefficient_grad_flow() -> None:
    """
    References:
    - van Genuchten and Wierenga (1986), doi:10.2136/sssaj1986.03615995005000050013x

    Mathematical expressions:
    - R = 1 + rho_b * K_d / theta
    - F_N = Q_drain * C_NO3 / R

    Parameter meanings:
    - K_d: nitrate distribution coefficient [L kg-1], learnable
    - rho_b: soil bulk density [kg m-3]
    - theta: volumetric water content [m3 m-3]
    """
    manager = get_global_manager()
    key = "pedosphere.transport.nitrate_distribution_coefficient"
    spec = manager.get_parameter(key)
    assert spec is not None
    assert spec.requires_grad is True

    original_value = manager.get(key, required=True)
    original_scalar = _as_scalar(original_value)

    try:
        manager.set_value(key, 0.1, units=spec.units, validate=True)
        trainable = manager.get_trainable_parameters()
        assert key in trainable
        trainable[key].grad = None
        kd_parameter = trainable[key]

        functor = NitrogenLeachingProcess("nitrogen_leaching_grad_flow")
        kd_value = functor.nitrate_distribution_coefficient
        assert isinstance(kd_value, torch.Tensor)
        assert kd_value.requires_grad

        state = {
            "no3_nitrogen": torch.tensor([0.05], dtype=torch.float32),
            "soil_moisture": torch.tensor([0.3], dtype=torch.float32),
            "drainage_flux": torch.tensor([1.0e-5], dtype=torch.float32),
            "soil_depth": torch.tensor([1.0], dtype=torch.float32),
            "bulk_density": torch.tensor([1300.0], dtype=torch.float32),
        }
        out = functor.forward(state=state, time=0.0, dt=3600.0)
        flux = out["nitrogen_leaching_flux"]
        assert isinstance(flux, UpdateItem)
        assert flux.mode is UpdateMode.REPLACE
        assert flux.per_timestep is False
        loss = flux.value.sum()
        loss.backward()

        assert kd_parameter.grad is not None
        assert torch.isfinite(kd_parameter.grad).all()
    finally:
        manager.set_value(key, original_scalar, units=spec.units, validate=True)


_KD_KEY = "pedosphere.transport.nitrate_distribution_coefficient"


def _leaching_state() -> dict:
    """A draining, nitrate-bearing soil column below both export limits."""
    return {
        "no3_nitrogen": torch.tensor([0.05], dtype=torch.float32),
        "soil_moisture": torch.tensor([0.3], dtype=torch.float32),
        "drainage_flux": torch.tensor([1.0e-5], dtype=torch.float32),
        "soil_depth": torch.tensor([1.0], dtype=torch.float32),
        "bulk_density": torch.tensor([1300.0], dtype=torch.float32),
    }


def _leaching_tensors(result) -> dict:
    """Every tensor of a leaching update, keyed by output name."""
    tensors = {}
    for name, update in result.items():
        value = update.value if isinstance(update, UpdateItem) else update
        members = value.items() if isinstance(value, dict) else [(None, value)]
        for member, item in members:
            if isinstance(item, torch.Tensor):
                tensors[name if member is None else f"{name}:{member}"] = item
    return tensors


def test_nitrate_distribution_coefficient_follows_a_changed_binding() -> None:
    """K_d is held by reference and read at use, so an instance built before
    the binding changed computes like one built afterwards."""
    # References: van Genuchten and Wierenga (1986), solute retardation by
    #   linear equilibrium sorption; Jury and Horton (2004), Soil Physics.
    # Mathematical expression: R = 1 + rho_b (K_d c_L) / theta;
    #   f_mobile = 1 / R; F_N = q C_NO3 f_mobile.
    # Parameter meanings: K_d nitrate distribution coefficient, c_L the
    #   liter-to-cubic-meter conversion, rho_b bulk density, theta volumetric
    #   water content, R retardation factor, f_mobile mobile fraction, q
    #   drainage water flux, C_NO3 soil-solution nitrate concentration.
    # Units: K_d [L kg-1]; c_L [m3 L-1]; rho_b [kg m-3]; theta [m3 m-3];
    #   R, f_mobile [1]; q [m s-1]; C_NO3 [g N m-3]; F_N [g N m-2 s-1].
    # Domain/constraints: K_d >= 0; both K_d values lie inside the
    #   ParameterManager bounds, so neither is clamped; the state keeps the
    #   flux below the time-constant and pool-fraction limits.
    # Sign convention: F_N is a positive export of nitrate from the soil.
    manager = get_global_manager()
    spec = manager.get_parameter(_KD_KEY)
    manager.set_value(_KD_KEY, 0.1, units=spec.units, validate=True)
    functor = NitrogenLeachingProcess("nitrogen_leaching_built_before_the_change")
    manager.set_value(_KD_KEY, 0.11, units=spec.units, validate=True)
    fresh = NitrogenLeachingProcess("nitrogen_leaching_built_after_the_change")

    expected = _leaching_tensors(fresh(_leaching_state(), time=0.0, dt=3600.0))
    actual = _leaching_tensors(functor(_leaching_state(), time=0.0, dt=3600.0))
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )

    binding = manager.get_trainable_parameters()[_KD_KEY]
    assert float(binding.detach()) == pytest.approx(0.11)
    held = functor.held_manager_parameters()
    assert set(held) == {_KD_KEY}
    assert held[_KD_KEY] is binding
    assert functor.nitrate_distribution_coefficient is binding

    state = _leaching_state()
    kd_m3_kg = binding.detach().cpu() * functor.liter_to_cubic_meter
    expected_mobile_fraction = 1.0 / (
        1.0 + state["bulk_density"] * kd_m3_kg / state["soil_moisture"]
    )
    torch.testing.assert_close(
        actual["diagnostics.nitrogen_leaching.mobile_fraction"].detach().cpu(),
        expected_mobile_fraction,
        rtol=1.0e-6,
        atol=0.0,
    )


def test_nitrogen_leaching_supports_repeated_backward_through_one_instance() -> None:
    """No autograd graph is stored on the instance: it holds the K_d binding
    itself, a second backward through it works and it gives the gradient of
    the first, which lands on the ParameterManager binding."""
    functor = NitrogenLeachingProcess("nitrogen_leaching_repeated_backward")
    stored_graphs = sorted(
        name
        for name, attribute in vars(functor).items()
        if isinstance(attribute, torch.Tensor) and attribute.grad_fn is not None
    )
    assert stored_graphs == []

    binding = get_global_manager().get_trainable_parameters()[_KD_KEY]
    assert functor.held_manager_parameters()[_KD_KEY] is binding
    gradients = []
    try:
        for _ in range(2):
            binding.grad = None
            update = functor(_leaching_state(), time=0.0, dt=3600.0)
            update["nitrogen_leaching_flux"].value.sum().backward()
            gradients.append(binding.grad.detach().cpu().clone())
    finally:
        binding.grad = None

    assert torch.isfinite(gradients[0]).all()
    assert bool((gradients[0] != 0).all())
    torch.testing.assert_close(gradients[1], gradients[0], rtol=0.0, atol=0.0)
