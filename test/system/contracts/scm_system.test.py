"""L3 system test for SCM provenance integration."""

import json
from pathlib import Path

import torch

from relic.core.simulator import RELICSimulator, SimulationState
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
)
from relic.core.state import StateManager, StateVariableSpec
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.provenance.recorder import ProvenanceRecorder


class ProvenanceFunctor(BiologicalProcessFunctor):
    def __init__(self):
        metadata = FunctorMetadata(
            name="biosphere_phenology_functor",
            version="1.0.0",
            description="Contract-owned functor for provenance system test",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            input_variables=[],
            output_variables=["biosphere.leaf_carbon"],
            parameters={},
            references=["RELIC SCM contract registry single-writer policy"],
        )
        super().__init__(name="biosphere_phenology_functor", config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"biosphere.leaf_carbon": torch.tensor(0.1)}


def test_system_provenance_pipeline(tmp_path: Path) -> None:
    schema = {
        "biosphere.leaf_carbon": StateVariableSpec(
            name="biosphere.leaf_carbon",
            units="kg m-2",
            bounds=(0.0, 10.0),
        )
    }
    sm = StateManager(schema=schema)
    sm.set_state({"biosphere.leaf_carbon": torch.tensor(1.0)}, validate=False)

    recorder = ProvenanceRecorder(output_dir=tmp_path)
    sm.set_provenance_recorder(recorder)

    functor = ProvenanceFunctor()
    dep = DependencyGraphResolver()
    dep.add_functor(functor)

    sim = RELICSimulator({"time_step": 1.0})
    sim.register_functor(functor)
    sim.initialize(sm, dep)
    sim.enable_conservation_checks = False
    sim.state = SimulationState.INITIALIZED

    sim.step(dt=1.0)

    prov_path = tmp_path / "prov.jsonld"
    payload = json.loads(prov_path.read_text(encoding="utf-8"))
    assert f"state:{recorder.run_id}:biosphere.leaf_carbon@step1" in payload.get("entity", {})
