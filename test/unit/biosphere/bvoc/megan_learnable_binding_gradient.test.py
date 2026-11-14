"""MEGANModel holds its learnable ParameterManager bindings by reference.

The parameter database marks the MEGAN light and temperature coefficients and the
dry-mass emission factors as learnable (requires_grad: true). The manager owns one
binding per key (architecture SSOT Sections 4.2 and 4.4.2); a Functor that copies
the value at construction cuts every gradient to it. These tests pin that MEGANModel
holds the bindings themselves and that gradients of the isoprene emission reach them
with the values of central finite differences.
"""

from __future__ import annotations

from copy import copy
from typing import Dict

import torch

from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway, MEGANModel

HELD_MEGAN_BINDING_KEYS = [
    "biosphere.bvoc.emission_factors.isoprene",
    "biosphere.bvoc.emission_factors.monoterpenes",
    "biosphere.bvoc.emission_factors.ovocs",
    "biosphere.bvoc.emission_factors.sesquiterpenes",
    "biosphere.bvoc.megan.alpha_light",
    "biosphere.bvoc.megan.cl_light",
    "biosphere.bvoc.megan.ct1_isoprene",
    "biosphere.bvoc.megan.ct2",
    "biosphere.bvoc.megan.eopt_isoprene",
    "biosphere.bvoc.megan.topt",
]


def _model() -> MEGANModel:
    return MEGANModel(
        emission_pathway=EmissionPathway.STANDARD,
        config={"enabled_compounds": ["isoprene"]},
    )


def _inputs(dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """One leaf-scale cell on a warm, bright hour (canonical SI ports)."""
    return {
        "air_temperature": torch.tensor([303.15], dtype=dtype),  # K
        "ppfd": torch.tensor([1200.0], dtype=dtype),  # umol m-2 s-1
        "leaf_area_index": torch.tensor([3.0], dtype=dtype),  # m2 m-2
        "atmospheric_co2_ppm": torch.tensor([400.0], dtype=dtype),  # umol mol-1
    }


def _emission(model: MEGANModel, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return model.forward(_inputs(dtype), 0.0, 3600.0)["isoprene_emission"].value.sum()


def test_megan_holds_the_learnable_database_coefficients_by_reference():
    """Every learnable coefficient MEGAN reads at construction is the manager's own
    binding, not a module buffer copy, so calibration and checkpoints see it."""
    model = _model()
    manager = model.get_parameter_manager()
    held = model.held_manager_parameters()
    assert sorted(held) == HELD_MEGAN_BINDING_KEYS
    for key, binding in held.items():
        assert binding is manager.nn_parameters[key], key
    assert not set(model.state_dict()) & {
        "alpha_light", "cl_light", "topt", "eopt_isoprene", "ct1_iso", "ct2", "isoprene_ef",
    }


def test_isoprene_emission_gradients_reach_the_held_bindings():
    """Backward through one call leaves finite gradients on the bindings the isoprene
    law uses; the emission factor gradient equals E / epsilon (linear law)."""
    # References: Guenther et al. (2012), Geosci. Model Dev. 5, 1471-1492, Eqs. (2)-(8).
    # Mathematical expression: E = epsilon gamma_T gamma_P rho_f / 3600, so
    #   dE/d epsilon = E / epsilon; gamma_P depends on alpha and C_L, gamma_T on
    #   T_opt, C_T1, C_T2 and E_opt.
    # Parameter meanings: epsilon isoprene dry-mass emission factor; alpha, C_L light
    #   coefficients; T_opt, C_T1, C_T2, E_opt temperature coefficients.
    # Units: E [mol m-2 s-1]; epsilon [ug g-1 DW h-1]; alpha [m2 s umol-1]; C_L [1];
    #   T_opt [K]; C_T1, C_T2 [J mol-1]; E_opt [1].
    # Domain/constraints: T > T_s and PPFD > 0 so every gradient is nonzero.
    # Sign convention: E is a nonnegative emission magnitude.
    model = _model()
    held = model.held_manager_parameters()
    used = [
        "biosphere.bvoc.emission_factors.isoprene",
        "biosphere.bvoc.megan.alpha_light",
        "biosphere.bvoc.megan.cl_light",
        "biosphere.bvoc.megan.ct1_isoprene",
        "biosphere.bvoc.megan.ct2",
        "biosphere.bvoc.megan.eopt_isoprene",
        "biosphere.bvoc.megan.topt",
    ]
    try:
        for binding in held.values():
            binding.grad = None
        emission = _emission(model)
        emission.backward()
        for key in used:
            grad = held[key].grad
            assert grad is not None, key
            assert torch.isfinite(grad).all() and bool((grad != 0).all()), key
        epsilon = held["biosphere.bvoc.emission_factors.isoprene"]
        torch.testing.assert_close(
            epsilon.grad, (emission / epsilon).detach(), rtol=1.0e-5, atol=0.0, check_dtype=False
        )
    finally:
        for binding in held.values():
            binding.grad = None


def test_nonlinear_coefficient_gradients_match_central_differences():
    """In float64 bindings the autograd gradient of the nonlinear light and temperature
    coefficients equals the central difference (truncation error O(h^2))."""
    # Mathematical expression: g_fd = (E(theta + h) - E(theta - h)) / (2 h),
    #   h = 1e-4 |theta|; |g_ad - g_fd| <= 1e-4 |g_fd|.
    # Units: as in the previous test; theta in the units of its key.
    keys = [
        "biosphere.bvoc.megan.alpha_light",
        "biosphere.bvoc.megan.ct1_isoprene",
        "biosphere.bvoc.megan.topt",
    ]
    model_probe = _model()
    manager = model_probe.get_parameter_manager()
    originals = {key: copy(manager.get_parameter(key)) for key in keys}
    try:
        for key in keys:
            spec = copy(originals[key])
            spec.dtype = "float64"
            spec.requires_grad = True
            manager.register_parameter(spec)
        model = _model()
        held = model.held_manager_parameters()
        for key in keys:
            binding = held[key]
            assert binding.dtype == torch.float64, key
            emission = _emission(model, torch.float64)
            (g_ad,) = torch.autograd.grad(emission, binding)
            x0 = float(binding.detach())
            h = 1.0e-4 * abs(x0)
            values = []
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    binding.fill_(x0 + sign * h)
                values.append(float(_emission(model, torch.float64).detach()))
            with torch.no_grad():
                binding.fill_(x0)
            g_fd = (values[0] - values[1]) / (2.0 * h)
            assert abs(float(g_ad) - g_fd) <= 1.0e-4 * abs(g_fd), (key, float(g_ad), g_fd)
    finally:
        for key, spec in originals.items():
            manager.register_parameter(spec)
