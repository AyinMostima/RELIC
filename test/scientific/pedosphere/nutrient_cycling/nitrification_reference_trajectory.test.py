"""L4 scientific benchmarks for the canonical nitrification runtime.

References:
- Li et al. (1992), J. Geophys. Res. 97:9759-9776.
- Parton et al. (1996), Global Biogeochemical Cycles 10:401-412.
- Prosser (1989), Adv. Microb. Physiol. 30:125-181.

Mathematical expressions:
- R_nit = Vmax S / (Km + S) f_T f_O2 f_pH.
- f_pH = exp(-(pH - pH_opt)^2 / (2 sigma_pH^2)).
- dS/dt = -Veff S / (Km + S).
- Delta NH4 + Delta NO3 + Delta N2O = 0.

Parameter meanings:
- S is the ammonium nitrogen pool [g N m-2].
- Vmax and Veff are nitrification rates [g N m-2 s-1].
- Km is the ammonium half-saturation pool [g N m-2].
- sigma_pH is the dimensionless pH response width.
"""

from __future__ import annotations

import math

import pytest
import torch

from relic.core.state import UpdateItem
from relic.scientific.pedosphere.nutrient_cycling.nitrification import (
    NitrificationProcess,
)

pytestmark = pytest.mark.scientific


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def _environment(functor: NitrificationProcess) -> dict[str, torch.Tensor]:
    """Return constant, non-limiting environmental drivers."""
    return {
        "soil_temperature": torch.tensor(
            _scalar(functor.temperature_reference),
            dtype=torch.float64,
        ),
        "water_filled_pore_space": torch.tensor(
            min(0.5, _scalar(functor.wfps_optimum) * 0.5),
            dtype=torch.float64,
        ),
        "soil_ph": torch.tensor(
            _scalar(functor.ph_optimum),
            dtype=torch.float64,
        ),
    }


def _state(
    *,
    ammonium: float = 1.0,
    nitrate: float = 0.25,
) -> dict[str, torch.Tensor]:
    return {
        "nh4_nitrogen": torch.tensor(ammonium, dtype=torch.float64),
        "no3_nitrogen": torch.tensor(nitrate, dtype=torch.float64),
    }


def _advance(
    functor: NitrificationProcess,
    state: dict[str, torch.Tensor],
    environment: dict[str, torch.Tensor],
    *,
    dt: float,
    steps: int,
    start_time: float = 0.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Advance the reaction ledger and integrate its open N2O export.

    References:
    - Li et al. (1992); Parton et al. (1996).
    Mathematical expressions:
    - NH4_(n+1) = NH4_n + Delta NH4_n.
    - NO3_(n+1) = NO3_n + Delta NO3_n.
    - N2O_export = sum_n(R_N2O,n dt).
    Parameter meanings:
    - dt is the integration interval [s].
    - R_N2O is the nitrous oxide nitrogen flux [g N m-2 s-1].
    """
    current = {key: value.clone() for key, value in state.items()}
    n2o_export = torch.zeros(
        (),
        dtype=current["nh4_nitrogen"].dtype,
        device=current["nh4_nitrogen"].device,
    )
    for step in range(steps):
        # The environment drivers are state inputs of the rate core.
        updates = functor.forward(
            {**current, **environment},
            time=start_time + step * dt,
            dt=dt,
        )
        for key in ("nh4_nitrogen", "no3_nitrogen"):
            update = updates[key]
            assert isinstance(update, UpdateItem)
            current[key] = current[key] + update.value
        n2o_update = updates["n2o_flux"]
        assert isinstance(n2o_update, UpdateItem)
        n2o_export = n2o_export + n2o_update.value * dt
    return current, n2o_export


def _implicit_michaelis_menten_solution(
    initial_substrate: float,
    half_saturation: float,
    effective_vmax: float,
    horizon: float,
) -> float:
    """Solve S + Km ln(S) = S0 + Km ln(S0) - Veff t by bisection."""
    target = (
        initial_substrate
        + half_saturation * math.log(initial_substrate)
        - effective_vmax * horizon
    )

    def residual(substrate: float) -> float:
        return substrate + half_saturation * math.log(substrate) - target

    lower = max(initial_substrate * 1.0e-15, 1.0e-300)
    upper = initial_substrate
    assert residual(lower) < 0.0 < residual(upper)
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


# References:
#   - Prosser (1989); Parton et al. (1996).
# Mathematical expression:
#   f_pH(pH_opt + sigma_pH) = exp(-1/2).
# Parameter meanings:
#   pH_opt is the optimum and sigma_pH is the Gaussian width [-].
def test_ph_response_matches_documented_gaussian_width() -> None:
    functor = NitrificationProcess(name="nitrification_l4_ph")
    environment = _environment(functor)
    environment["soil_ph"] = torch.tensor(
        _scalar(functor.ph_optimum) + _scalar(functor.ph_response_width),
        dtype=torch.float64,
    )

    rates = functor._compute_rate_core({**_state(), **environment})

    assert _scalar(rates["ph_factor"]) == pytest.approx(
        math.exp(-0.5),
        rel=2.0e-7,
    )


# References:
#   - Feinberg (2019), Foundations of Chemical Reaction Network Theory.
# Mathematical expression:
#   Delta x = N_h r_h dt with dtype(Delta x) = dtype(x).
# Parameter meanings:
#   x is the nitrogen state vector and N_h is the stoichiometric matrix.
def test_reaction_updates_preserve_state_dtype() -> None:
    functor = NitrificationProcess(name="nitrification_l4_dtype")
    state = _state()
    updates = functor.forward(
        {**state, **_environment(functor)},
        time=0.0,
        dt=3600.0,
    )

    assert updates["nh4_nitrogen"].value.dtype == state["nh4_nitrogen"].dtype
    assert updates["no3_nitrogen"].value.dtype == state["no3_nitrogen"].dtype


# References:
#   - Michaelis and Menten (1913); Prosser (1989).
# Mathematical expressions:
#   dS/dt = -Veff S / (Km + S).
#   S + Km ln(S) = S0 + Km ln(S0) - Veff t.
# Parameter meanings:
#   S is NH4 [g N m-2], Km is half-saturation [g N m-2],
#   and Veff is the environment-adjusted maximum rate [g N m-2 s-1].
def test_explicit_runtime_converges_to_implicit_substrate_solution() -> None:
    probe = NitrificationProcess(name="nitrification_l4_probe")
    initial = _state(ammonium=1.0)
    environment = _environment(probe)
    initial_rate = probe._compute_rate_core({**initial, **environment})[
        "nitrification_rate"
    ]
    substrate_0 = _scalar(initial["nh4_nitrogen"])
    half_saturation = _scalar(probe.km_nh4)
    effective_vmax = (
        _scalar(initial_rate)
        * (half_saturation + substrate_0)
        / substrate_0
    )
    horizon = 2.0 * 86400.0
    exact = _implicit_michaelis_menten_solution(
        substrate_0,
        half_saturation,
        effective_vmax,
        horizon,
    )

    coarse_state, _ = _advance(
        NitrificationProcess(name="nitrification_l4_coarse"),
        initial,
        environment,
        dt=21600.0,
        steps=8,
    )
    fine_state, _ = _advance(
        NitrificationProcess(name="nitrification_l4_fine"),
        initial,
        environment,
        dt=600.0,
        steps=288,
    )
    coarse_error = abs(_scalar(coarse_state["nh4_nitrogen"]) - exact)
    fine_error = abs(_scalar(fine_state["nh4_nitrogen"]) - exact)

    assert fine_error < coarse_error * 0.1
    assert _scalar(fine_state["nh4_nitrogen"]) == pytest.approx(
        exact,
        rel=2.0e-5,
        abs=1.0e-8,
    )


# References:
#   - Li et al. (1992); Parton et al. (1996).
# Mathematical expression:
#   NH4_0 + NO3_0 = NH4_t + NO3_t + integral(R_N2O dt).
# Parameter meanings:
#   NH4 and NO3 are soil nitrogen pools [g N m-2].
#   R_N2O is the gaseous nitrogen export [g N m-2 s-1].
def test_multistep_restart_preserves_trajectory_and_nitrogen_ledger() -> None:
    initial = _state(ammonium=1.0, nitrate=0.25)
    reference_functor = NitrificationProcess(name="nitrification_l4_reference")
    environment = _environment(reference_functor)
    dt = 3600.0
    steps = 24
    reference_state, reference_n2o = _advance(
        reference_functor,
        initial,
        environment,
        dt=dt,
        steps=steps,
    )

    first_functor = NitrificationProcess(name="nitrification_l4_first")
    midpoint_state, first_n2o = _advance(
        first_functor,
        initial,
        environment,
        dt=dt,
        steps=steps // 2,
    )
    checkpoint = {
        key: value.detach().clone()
        for key, value in midpoint_state.items()
    }
    restarted_functor = NitrificationProcess(name="nitrification_l4_restarted")
    restarted_state, second_n2o = _advance(
        restarted_functor,
        checkpoint,
        environment,
        dt=dt,
        steps=steps // 2,
        start_time=(steps // 2) * dt,
    )
    restarted_n2o = first_n2o + second_n2o

    assert torch.allclose(
        restarted_state["nh4_nitrogen"],
        reference_state["nh4_nitrogen"],
        rtol=0.0,
        atol=2.0e-12,
    )
    assert torch.allclose(
        restarted_state["no3_nitrogen"],
        reference_state["no3_nitrogen"],
        rtol=0.0,
        atol=2.0e-12,
    )
    assert torch.allclose(restarted_n2o, reference_n2o, rtol=0.0, atol=2.0e-12)
    assert torch.allclose(
        reference_state["nh4_nitrogen"]
        + reference_state["no3_nitrogen"]
        + reference_n2o,
        initial["nh4_nitrogen"] + initial["no3_nitrogen"],
        rtol=0.0,
        atol=2.0e-11,
    )
