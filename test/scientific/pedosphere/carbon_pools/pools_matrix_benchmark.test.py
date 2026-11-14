"""L4 analytical benchmarks for the canonical five-pool carbon closure.

References:
- Sierra et al. (2012), SoilR compartment-model framework.
- Sierra and Mueller (2015), linear soil organic matter systems.
- Coleman and Jenkinson (1996), RothC compartment transfers.

Mathematical expressions:
- dC/dt = A C, where A = (T - I) diag(k f_env).
- C(t) = exp(A t) C(0) for constant coefficients and zero boundary input.
- sum(C(t)) + integral(R dt) = sum(C(0)).

Parameter meanings:
- C is the five-pool carbon stock vector [g C m-2].
- T is the source-column transfer matrix [dimensionless].
- k is the ParameterManager turnover-rate vector [s-1].
- f_env is the external environmental modifier [dimensionless].
- R is heterotrophic respiration [g C m-2 s-1].
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from relic.core.state import UpdateItem
from relic.scientific.pedosphere.carbon_pools import PoolsMatrix
from relic.utils.physical_constants import get_physical_constant

pytestmark = pytest.mark.scientific

POOL_KEYS = (
    "metabolic_litter_c",
    "structural_litter_c",
    "fast_som_c",
    "slow_som_c",
    "passive_som_c",
)
BOUNDARY_KEYS = (
    "metabolic_litter_input",
    "structural_litter_input",
    "fast_som_input",
    "slow_som_input",
    "passive_som_input",
)
SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))


def _initial_pools(*, requires_grad: bool = False) -> torch.Tensor:
    """Return a positive five-pool benchmark state in canonical order."""
    return torch.tensor(
        [0.1, 0.3, 0.8, 2.0, 4.0],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def _state_from_pools(
    pools: torch.Tensor,
    *,
    environmental_modifier: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Build the exact 11-port runtime state with a closed input boundary."""
    zero = torch.zeros((), dtype=pools.dtype, device=pools.device)
    state = {
        key: pools[index]
        for index, key in enumerate(POOL_KEYS)
    }
    state["environmental_modifier"] = torch.as_tensor(
        environmental_modifier,
        dtype=pools.dtype,
        device=pools.device,
    )
    state.update({key: zero.clone() for key in BOUNDARY_KEYS})
    return state


def _pool_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack the canonical pool state without detaching autograd."""
    return torch.stack([state[key] for key in POOL_KEYS])


def _system_matrix(
    functor: PoolsMatrix,
    *,
    dtype: torch.dtype,
    device: torch.device,
    environmental_modifier: float,
) -> torch.Tensor:
    """Build the constant analytical compartment operator.

    References:
    - Sierra et al. (2012); Sierra and Mueller (2015).
    Mathematical expression:
    - A = (T - I) diag(k f_env).
    Parameter meanings:
    - T is the transfer matrix, k is turnover [s-1], and f_env is dimensionless.
    """
    rates = torch.stack(
        [
            torch.as_tensor(
                functor._base_decomposition_rates[pool_name],
                dtype=dtype,
                device=device,
            )
            for pool_name in functor.pool_names
        ]
    )
    transfer = functor.transfer_matrix.to(dtype=dtype, device=device)
    identity = torch.eye(functor.n_pools, dtype=dtype, device=device)
    return (transfer - identity) @ torch.diag(
        rates * environmental_modifier
    )


def _advance(
    functor: PoolsMatrix,
    state: dict[str, torch.Tensor],
    *,
    dt: float,
    steps: int,
    start_time: float = 0.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Advance the explicit runtime and integrate its open carbon export.

    References:
    - Sierra et al. (2012); RELIC decomposition theory contract.
    Mathematical expressions:
    - C_(n+1) = C_n + dt A C_n.
    - R_total = sum_n R_n dt.
    Parameter meanings:
    - dt is the integration interval [s], and R_n is respiration [g C m-2 s-1].
    """
    current = dict(state)
    respiration = torch.zeros_like(current[POOL_KEYS[0]])
    for step in range(steps):
        updates = functor.forward(
            current,
            time=start_time + step * dt,
            dt=dt,
        )
        for key in POOL_KEYS:
            item = updates[key]
            assert isinstance(item, UpdateItem)
            current[key] = current[key] + item.value
        respiration_item = updates["heterotrophic_respiration"]
        assert isinstance(respiration_item, UpdateItem)
        respiration = respiration + respiration_item.value * dt
    return current, respiration


def test_euler_runtime_converges_to_matrix_exponential_and_closes_carbon() -> None:
    functor = PoolsMatrix(name="pools_matrix_l4_convergence")
    initial = _initial_pools()
    modifier = 1.0
    horizon = 20.0 * SECONDS_PER_DAY
    operator = _system_matrix(
        functor,
        dtype=initial.dtype,
        device=initial.device,
        environmental_modifier=modifier,
    )
    exact = torch.matrix_exp(operator * horizon) @ initial

    coarse_state, _ = _advance(
        functor,
        _state_from_pools(initial, environmental_modifier=modifier),
        dt=5.0 * SECONDS_PER_DAY,
        steps=4,
    )
    fine_state, fine_respiration = _advance(
        functor,
        _state_from_pools(initial, environmental_modifier=modifier),
        dt=0.5 * SECONDS_PER_DAY,
        steps=40,
    )
    coarse = _pool_vector(coarse_state)
    fine = _pool_vector(fine_state)
    coarse_error = torch.linalg.vector_norm(coarse - exact)
    fine_error = torch.linalg.vector_norm(fine - exact)

    assert fine_error < coarse_error * 0.2
    assert torch.allclose(fine, exact, rtol=5.0e-3, atol=1.0e-10)
    assert torch.allclose(
        fine.sum() + fine_respiration,
        initial.sum(),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_runtime_autograd_matches_analytical_linear_sensitivity() -> None:
    functor = PoolsMatrix(name="pools_matrix_l4_gradient")
    initial = _initial_pools(requires_grad=True)
    modifier = 0.8
    horizon = 5.0 * SECONDS_PER_DAY
    steps = 40
    final_state, _ = _advance(
        functor,
        _state_from_pools(initial, environmental_modifier=modifier),
        dt=horizon / steps,
        steps=steps,
    )
    final_total = _pool_vector(final_state).sum()
    (runtime_gradient,) = torch.autograd.grad(final_total, initial)

    operator = _system_matrix(
        functor,
        dtype=initial.dtype,
        device=initial.device,
        environmental_modifier=modifier,
    )
    exact_gradient = torch.matrix_exp(operator * horizon).T @ torch.ones_like(
        initial
    )

    assert torch.isfinite(runtime_gradient).all()
    assert torch.allclose(
        runtime_gradient,
        exact_gradient,
        rtol=2.0e-3,
        atol=1.0e-10,
    )


def test_checkpoint_restart_matches_continuous_trajectory() -> None:
    initial = _initial_pools()
    dt = 0.5 * SECONDS_PER_DAY
    steps = 12

    continuous_functor = PoolsMatrix(name="pools_matrix_l4_continuous")
    continuous_state, continuous_respiration = _advance(
        continuous_functor,
        _state_from_pools(initial),
        dt=dt,
        steps=steps,
    )

    first_functor = PoolsMatrix(name="pools_matrix_l4_first")
    midpoint_state, first_respiration = _advance(
        first_functor,
        _state_from_pools(initial),
        dt=dt,
        steps=steps // 2,
    )
    checkpoint: dict[str, Any] = {
        "functor_state": {
            key: value.detach().clone()
            for key, value in first_functor.state_dict().items()
        },
        "runtime_state": {
            key: value.detach().clone()
            for key, value in midpoint_state.items()
        },
    }
    restarted_functor = PoolsMatrix(name="pools_matrix_l4_restarted")
    restarted_functor.load_state_dict(checkpoint["functor_state"])
    restarted_state, second_respiration = _advance(
        restarted_functor,
        checkpoint["runtime_state"],
        dt=dt,
        steps=steps // 2,
        start_time=(steps // 2) * dt,
    )

    assert torch.equal(
        _pool_vector(restarted_state),
        _pool_vector(continuous_state),
    )
    assert torch.allclose(
        first_respiration + second_respiration,
        continuous_respiration,
        rtol=0.0,
        atol=torch.finfo(continuous_respiration.dtype).eps,
    )
