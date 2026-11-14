"""L1 tests for SCM enforcement in StateManager."""

import pytest
import torch

from relic.core.state import StateManager, StateVariableSpec
from relic.contracts.contract_loader import ContractRegistry, VariableContract


def _make_state_manager() -> StateManager:
    schema = {
        "biosphere.leaf_carbon": StateVariableSpec(
            name="biosphere.leaf_carbon",
            units="kg m-2",
            bounds=(0.0, 10.0),
        )
    }
    sm = StateManager(schema=schema)
    sm.set_state({"biosphere.leaf_carbon": torch.tensor(1.0)}, validate=False)
    return sm


def test_alias_write_blocked() -> None:
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="biosphere.leaf_carbon",
            kind="state",
            udunits="kg m-2",
            aliases=["C_leaf"],
        )
    )

    sm = _make_state_manager()
    sm._contract_registry = registry

    with pytest.raises(PermissionError):
        sm.apply_updates({"C_leaf": torch.tensor(0.1)})


def test_writer_whitelist_enforced() -> None:
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="biosphere.leaf_carbon",
            kind="state",
            udunits="kg m-2",
            owner="Allocation",
            writers=["Photosynthesis"],
        )
    )

    sm = _make_state_manager()
    sm._contract_registry = registry

    with pytest.raises(PermissionError):
        sm.apply_updates(
            {"biosphere.leaf_carbon": torch.tensor(0.1)},
            update_sources={"biosphere.leaf_carbon": "OtherFunctor"},
            check_conservation=False,
        )

    sm.apply_updates(
        {"biosphere.leaf_carbon": torch.tensor(0.1)},
        update_sources={"biosphere.leaf_carbon": "Photosynthesis"},
        check_conservation=False,
    )
