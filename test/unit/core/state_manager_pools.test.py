"""L1 tests for StateManager element pool initialization."""

import pytest
import torch

from relic.core.state import (
    StateManager,
    StateVariableSpec,
    StateVariableType,
    UpdateItem,
)
from relic.contracts.contract_loader import ContractRegistry, VariableContract


def _make_state_manager() -> StateManager:
    schema = {
        "biosphere.leaf_carbon": StateVariableSpec(
            name="biosphere.leaf_carbon",
            units="kg m-2",
            conserved=True,
            spatial_dims=0,
            required=False,
            variable_type=StateVariableType.SCALAR,
        )
    }
    sm = StateManager(schema=schema)
    sm.set_state({"biosphere.leaf_carbon": torch.tensor(1.0)}, validate=False)
    return sm


def test_element_pools_from_contract_registry() -> None:
    sm = _make_state_manager()
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="biosphere.leaf_carbon",
            kind="state",
            udunits="kg m-2",
            accumulates="carbon.total",
        )
    )
    sm._contract_registry = registry
    sm._conservation_ledger = {}
    sm._element_pools.clear()
    sm._initialize_element_pools()

    pools = sm.element_pools
    assert "carbon" in pools
    assert "biosphere.leaf_carbon" in pools["carbon"]


def test_element_pools_from_parameter_ledger() -> None:
    sm = _make_state_manager()
    sm._contract_registry = None
    sm._conservation_ledger = {"carbon": ["leaf_carbon"]}
    sm._element_pools.clear()
    sm._initialize_element_pools()

    pools = sm.element_pools
    assert "carbon" in pools
    assert "leaf_carbon" in pools["carbon"]


def test_no_declared_element_domain_gives_no_element_pools() -> None:
    """Without a conservation ledger or contract domains no element is declared,
    so no element pool is invented from a default element list and variable names."""
    sm = StateManager(schema=_make_state_manager()._schema, autoload_contract_registry=False)
    sm._conservation_ledger = {}
    sm._element_pools.clear()
    sm._initialize_element_pools()

    assert sm.element_pools == {}


def test_explicit_non_accumulating_contract_blocks_name_inference() -> None:
    schema = {
        "soil_water_potential": StateVariableSpec(
            name="soil_water_potential",
            units="Pa",
            conserved=False,
            required=False,
            variable_type=StateVariableType.SCALAR,
        )
    }
    sm = StateManager(schema=schema)
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="soil_water_potential",
            kind="diagnostic",
            udunits="Pa",
            accumulates=None,
        )
    )
    sm._contract_registry = registry
    sm._conservation_ledger = {}
    sm._element_pools.clear()

    sm._initialize_element_pools()

    assert "soil_water_potential" not in sm.element_pools.get("water", [])


def test_non_accumulating_contract_is_excluded_from_runtime_balance() -> None:
    schema = {
        "soil_water_potential": StateVariableSpec(
            name="soil_water_potential",
            units="Pa",
            conserved=False,
            required=False,
            variable_type=StateVariableType.SCALAR,
        )
    }
    sm = StateManager(schema=schema)
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="soil_water_potential",
            kind="diagnostic",
            udunits="Pa",
            accumulates=None,
        )
    )
    sm._contract_registry = registry
    sm._conservation_ledger = {}
    sm._element_pools.clear()

    violations = sm._check_conservation(
        {"soil_water_potential": torch.tensor(5.0)},
        dt=None,
        scale_factor=None,
    )

    assert violations == {}


def test_update_item_element_classifies_uncontracted_conserved_update() -> None:
    schema = {
        "unclassified_pool": StateVariableSpec(
            name="unclassified_pool",
            units="g C m-2",
            conserved=True,
            required=False,
            variable_type=StateVariableType.SCALAR,
        )
    }
    sm = StateManager(schema=schema, autoload_contract_registry=False)
    sm._conservation_ledger = {}
    sm._element_pools.clear()

    violations = sm._check_conservation(
        {
            "unclassified_pool": UpdateItem(
                value=torch.tensor(2.0),
                unit="g C m-2",
                element="C",
            )
        },
        dt=None,
        scale_factor=None,
    )

    assert violations == {"carbon": pytest.approx(2.0)}


def test_update_item_element_conflict_with_contract_fails_fast() -> None:
    schema = {
        "leaf_carbon": StateVariableSpec(
            name="leaf_carbon",
            units="kg C m-2",
            conserved=True,
            required=False,
            variable_type=StateVariableType.SCALAR,
        )
    }
    sm = StateManager(schema=schema)
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="leaf_carbon",
            kind="state",
            udunits="kg C m-2",
            accumulates="carbon.total",
        )
    )
    sm._contract_registry = registry
    sm._conservation_ledger = {"carbon": ["leaf_carbon"]}
    sm._element_pools.clear()

    with pytest.raises(ValueError, match="element.*conflicts.*carbon"):
        sm._check_conservation(
            {
                "leaf_carbon": UpdateItem(
                    value=torch.tensor(1.0),
                    element="N",
                )
            },
            dt=None,
            scale_factor=None,
        )


def test_update_item_element_conflict_with_parameter_ledger_fails_fast() -> None:
    schema = {
        "legacy_pool": StateVariableSpec(
            name="legacy_pool",
            units="kg C m-2",
            conserved=True,
            required=False,
            variable_type=StateVariableType.SCALAR,
        )
    }
    sm = StateManager(schema=schema, autoload_contract_registry=False)
    sm._conservation_ledger = {"carbon": ["legacy_pool"]}
    sm._element_pools.clear()

    with pytest.raises(ValueError, match="element.*conflicts.*carbon"):
        sm._check_conservation(
            {
                "legacy_pool": UpdateItem(
                    value=torch.tensor(1.0),
                    element="N",
                )
            },
            dt=None,
            scale_factor=None,
        )
