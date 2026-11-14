"""
Audit tests for Forward-Only API contracts in pedosphere nutrient modules.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import (
    RootNutrientUptake,
)
from relic.scientific.pedosphere.nutrient_cycling.stoichiometry import (
    StoichiometryConstraint,
)


def test_root_nutrient_uptake_forward_signature_is_forward_only() -> None:
    """
    References:
    - docs/source/architecture_overview.md Section 18 (Forward-Only Execution Contract).

    Mathematical expression:
    - API contract check (no numerical equation).

    Parameter meanings:
    - forward positional parameters must be (state, time, dt) only.
    """
    signature = inspect.signature(RootNutrientUptake.forward)
    names = list(signature.parameters.keys())
    assert names[:4] == ["self", "state", "time", "dt"]
    assert "environment" not in signature.parameters


def test_root_nutrient_uptake_rejects_legacy_positional_order() -> None:
    """
    References:
    - docs/source/architecture_overview.md Section 18.

    Mathematical expression:
    - API contract check (no numerical equation).

    Parameter meanings:
    - Legacy positional environment argument must not be accepted.
    """
    functor = RootNutrientUptake()
    with pytest.raises(TypeError):
        functor.forward({}, {}, 0.0, 1.0)


def test_stoichiometry_forward_signature_is_forward_only() -> None:
    """
    References:
    - docs/source/architecture_overview.md Section 18.

    Mathematical expression:
    - API contract check (no numerical equation).

    Parameter meanings:
    - Stoichiometry forward positional parameters must be (state, time, dt) only.
    """
    signature = inspect.signature(StoichiometryConstraint.forward)
    names = list(signature.parameters.keys())
    assert names[:4] == ["self", "state", "time", "dt"]
    assert "carbon_fluxes" not in signature.parameters
    assert "nitrogen_fluxes" not in signature.parameters


def test_stoichiometry_missing_required_fluxes_fails_fast() -> None:
    """
    References:
    - docs/source/architecture_overview.md Section 19 (parameter/state fail-fast discipline).

    Mathematical expression:
    - C/N flux diagnostics are required for stoichiometric correction evaluation.

    Parameter meanings:
    - diagnostics.carbon_fluxes and diagnostics.nitrogen_fluxes are required inputs.
    """
    functor = StoichiometryConstraint()
    with pytest.raises(KeyError, match="diagnostics.carbon_fluxes"):
        functor.forward({}, time=0.0, dt=1.0)


def test_stoichiometry_kwargs_override_equivalence() -> None:
    """
    References:
    - Manzoni and Porporato (2009); RELIC forward-only contract.

    Mathematical expression:
    - N_target = C_flux / CN_target with diagnostic conservation checks.

    Parameter meanings:
    - kwargs overrides should be semantically equivalent to state diagnostics inputs.
    """
    functor_state = StoichiometryConstraint()
    functor_kwargs = StoichiometryConstraint()
    carbon_fluxes = {"delta_metabolic_litter_c": torch.tensor(0.2)}
    nitrogen_fluxes = {"delta_metabolic_litter_n": torch.tensor(0.01)}
    carbon_inputs = torch.tensor(0.3)
    nitrogen_inputs = torch.tensor(0.02)
    carbon_outputs = torch.tensor(0.1)
    nitrogen_outputs = torch.tensor(0.005)

    state = {
        "diagnostics.carbon_fluxes": carbon_fluxes,
        "diagnostics.nitrogen_fluxes": nitrogen_fluxes,
        "diagnostics.carbon_inputs": carbon_inputs,
        "diagnostics.nitrogen_inputs": nitrogen_inputs,
        "diagnostics.carbon_outputs": carbon_outputs,
        "diagnostics.nitrogen_outputs": nitrogen_outputs,
    }

    result_from_state = functor_state.forward(state, time=0.0, dt=1.0)
    result_from_kwargs = functor_kwargs.forward(
        {},
        time=0.0,
        dt=1.0,
        carbon_fluxes=carbon_fluxes,
        nitrogen_fluxes=nitrogen_fluxes,
        carbon_inputs=carbon_inputs,
        nitrogen_inputs=nitrogen_inputs,
        carbon_outputs=carbon_outputs,
        nitrogen_outputs=nitrogen_outputs,
    )

    compare_keys = [
        "diagnostics.stoichiometry_mass_balance_error_carbon",
        "diagnostics.stoichiometry_mass_balance_error_nitrogen",
        "diagnostics.stoichiometry_constraint_violations_count",
    ]
    for key in compare_keys:
        assert key in result_from_state
        assert key in result_from_kwargs
        assert torch.allclose(
            torch.as_tensor(result_from_state[key], dtype=torch.float64),
            torch.as_tensor(result_from_kwargs[key], dtype=torch.float64),
        )
