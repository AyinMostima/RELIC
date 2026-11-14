"""
L1 tests for the pedosphere carbon pools functor output contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.core.functor import VariableKind
from relic.scientific.pedosphere.carbon_pools.carbon_pools_functor import CarbonPoolsFunctor


def test_carbon_pools_functor_is_state_neutral_observer() -> None:
    """Runtime pool observation must not compete with state writers."""
    functor = CarbonPoolsFunctor()
    state = {
        key: torch.tensor(float(index + 1), dtype=torch.float32)
        for index, key in enumerate(functor._pool_keys)
    }
    outputs = functor.forward(state, time=0.0, dt=1.0)

    assert functor._pool_keys == [
        "metabolic_litter_c",
        "structural_litter_c",
        "fast_som_c",
        "slow_som_c",
        "passive_som_c",
    ]
    assert functor.metadata.variable_kind == VariableKind.DIAGNOSTIC
    assert functor.metadata.parameters == {}
    assert set(functor.metadata.output_variables) == {
        "diagnostics.carbon_pools_snapshot",
        "diagnostics.total_carbon_stock",
    }
    assert set(outputs) == set(functor.metadata.output_variables)
    assert not set(functor._pool_keys).intersection(outputs)
    assert all(isinstance(value, torch.Tensor) for value in outputs.values())

    pool_vector = outputs["diagnostics.carbon_pools_snapshot"]
    assert pool_vector.ndim == 1
    assert pool_vector.shape[0] == len(functor._pool_keys)
    assert torch.allclose(outputs["diagnostics.total_carbon_stock"], pool_vector.sum())


def test_carbon_pools_observer_preserves_supplied_state_without_writing_it() -> None:
    """The observer reads canonical stocks but emits diagnostics only."""
    functor = CarbonPoolsFunctor()
    state = {
        key: torch.tensor(float(index + 1), dtype=torch.float32)
        for index, key in enumerate(functor._pool_keys)
    }

    outputs = functor.forward(state, time=0.0, dt=3600.0)

    expected = torch.stack([state[key] for key in functor._pool_keys])
    assert torch.equal(outputs["diagnostics.carbon_pools_snapshot"], expected)
    assert torch.equal(outputs["diagnostics.total_carbon_stock"], expected.sum())
    assert all(key not in outputs for key in state)


def test_carbon_pools_observer_totals_each_grid_cell_over_the_pool_axis() -> None:
    """C_total(c) = sum_i C_i(c) is an areal stock [g C m-2] of one grid cell c.

    The sum runs over the pool axis only. Adding areal densities of different cells
    has no physical meaning without cell areas, so no cross-cell total is formed.
    """
    functor = CarbonPoolsFunctor()
    cell_stocks = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0],
            [100.0, 200.0, 300.0],
            [1000.0, 2000.0, 3000.0],
            [5.0, 6.0, 7.0],
        ],
        dtype=torch.float64,
    )
    state = {
        key: cell_stocks[index].clone() for index, key in enumerate(functor._pool_keys)
    }

    outputs = functor.forward(state, time=0.0, dt=1.0)

    assert torch.equal(outputs["diagnostics.carbon_pools_snapshot"], cell_stocks)
    total = outputs["diagnostics.total_carbon_stock"]
    assert total.shape == (3,)
    assert torch.equal(total, torch.tensor([1116.0, 2228.0, 3340.0], dtype=torch.float64))

    # Changing one cell leaves the stock of every other cell unchanged.
    perturbed = {key: value.clone() for key, value in state.items()}
    perturbed["passive_som_c"][2] = 0.0
    perturbed_total = functor.forward(perturbed, time=0.0, dt=1.0)[
        "diagnostics.total_carbon_stock"
    ]
    assert torch.equal(perturbed_total[:2], total[:2])
    assert perturbed_total[2] == total[2] - 7.0


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"metabolic_litter_c": torch.tensor(1.0)},
    ],
)
def test_carbon_pools_observer_rejects_incomplete_state_basis(state) -> None:
    """A diagnostic snapshot requires the complete StateManager pool basis."""
    functor = CarbonPoolsFunctor()

    with pytest.raises(ValueError, match="complete canonical pool basis"):
        functor.forward(state, time=0.0, dt=1.0)


def test_carbon_pool_contracts_bind_only_selected_decomposition_candidates() -> None:
    """SCM contracts must authorize alternatives and reject the observer."""
    registry = ContractRegistry(strict=True)
    contract_dir = Path("relic/contracts/variables/pedosphere")
    pool_names = (
        "metabolic_litter_c",
        "structural_litter_c",
        "fast_som_c",
        "slow_som_c",
        "passive_som_c",
    )
    for pool_name in pool_names:
        registry.load_contract_file(contract_dir / f"{pool_name}.yaml")

    five_pool_candidates = {"pools_matrix", "first_order_decomposition"}
    for pool_name in ("fast_som_c", "slow_som_c", "passive_som_c"):
        contract = registry.get_contract(pool_name)
        assert contract is not None
        assert contract.owner is None
        assert set(contract.increment_writers) == five_pool_candidates
        assert contract.raw["dimension"]["udunits"] == "g C m-2"
        assert contract.raw["provenance"]["runtime_role"] == (
            "mutually_exclusive_decomposition_state"
        )

    litter_candidates = five_pool_candidates | {"microbial_enzyme_decomposition"}
    for pool_name in ("metabolic_litter_c", "structural_litter_c"):
        contract = registry.get_contract(pool_name)
        assert contract is not None
        assert contract.owner is None
        assert set(contract.increment_writers) == litter_candidates
        assert contract.raw["dimension"]["udunits"] == "g C m-2"

    for pool_name in pool_names:
        assert not registry.can_increment(pool_name, "carbon_pools_observer")


def test_forward_kernel_takes_time_and_dt_without_defaults():
    """Round 47: the forward-only kernel signature is (state, time, dt)."""
    import inspect

    parameters = inspect.signature(CarbonPoolsFunctor.forward).parameters
    assert parameters["time"].default is inspect.Parameter.empty
    assert parameters["dt"].default is inspect.Parameter.empty
