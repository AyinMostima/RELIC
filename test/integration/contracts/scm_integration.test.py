"""L2 integration tests for SCM enforcement in simulator."""

import pytest
import torch

from relic.core.simulator import RELICSimulator, SimulationState
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
)
from relic.core.state import StateManager, StateVariableSpec
from relic.core.dependency_graph import DependencyGraphResolver
from relic.contracts.contract_loader import ContractRegistry, VariableContract


class AllowedFunctor(BiologicalProcessFunctor):
    def __init__(self):
        metadata = FunctorMetadata(
            name="AllowedFunctor",
            version="1.0.0",
            description="Allowed writer",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            input_variables=[],
            output_variables=["biosphere.leaf_carbon"],
            parameters={},
            references=["10.1007/BF00386231"],
        )
        super().__init__(name="AllowedFunctor", config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"biosphere.leaf_carbon": torch.tensor(0.1)}


class BlockedFunctor(BiologicalProcessFunctor):
    def __init__(self):
        metadata = FunctorMetadata(
            name="BlockedFunctor",
            version="1.0.0",
            description="Blocked writer",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            input_variables=[],
            output_variables=["biosphere.leaf_carbon"],
            parameters={},
            references=["10.1007/BF00386231"],
        )
        super().__init__(name="BlockedFunctor", config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"biosphere.leaf_carbon": torch.tensor(0.1)}


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


def _attach_contracts(sm: StateManager) -> None:
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="biosphere.leaf_carbon",
            kind="state",
            udunits="kg m-2",
            owner="AllowedFunctor",
            writers=["AllowedFunctor"],
        )
    )
    sm._contract_registry = registry


def _make_simulator(functor: FunctorBase, sm: StateManager) -> RELICSimulator:
    sim = RELICSimulator({"time_step": 1.0})
    dep = DependencyGraphResolver()
    dep.add_functor(functor)
    sim.register_functor(functor)
    sim.initialize(sm, dep)
    sim.enable_conservation_checks = False
    sim.state = SimulationState.INITIALIZED
    return sim


def test_allowed_functor_step() -> None:
    sm = _make_state_manager()
    _attach_contracts(sm)
    sim = _make_simulator(AllowedFunctor(), sm)

    updates = sim.step(dt=1.0)
    assert "biosphere.leaf_carbon" in updates


def test_blocked_functor_step() -> None:
    sm = _make_state_manager()
    _attach_contracts(sm)
    sim = _make_simulator(BlockedFunctor(), sm)

    with pytest.raises(PermissionError):
        sim.step(dt=1.0)
