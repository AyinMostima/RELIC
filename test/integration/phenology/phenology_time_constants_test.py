"""L2 integration checks for phenology dependency-output injection."""

from __future__ import annotations

import torch

from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
)
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.phenology.phenology_functor import PhenologyFunctor


class _GSIPhenologyOwner(BiologicalProcessFunctor):
    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="gsi_phenology",
            version="1.0.0",
            description="Canonical phenology owner integration fixture",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["Jolly et al. (2005)"],
            input_variables=[],
            output_variables=[
                "leaf_area_index",
                "phenological_factor",
                "phenology_senescence_factor",
                "phenology_phase",
            ],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        del state, time, dt
        return {
            "leaf_area_index": UpdateItem(
                value=torch.tensor(3.2),
                mode=UpdateMode.REPLACE,
                unit="m2 m-2",
                per_timestep=True,
                source_functor=self.name,
            ),
            "phenological_factor": UpdateItem(
                value=torch.tensor(0.8),
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                per_timestep=True,
                source_functor=self.name,
            ),
            "phenology_senescence_factor": UpdateItem(
                value=torch.tensor(0.05),
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                per_timestep=True,
                source_functor=self.name,
            ),
            "phenology_phase": UpdateItem(
                value=torch.tensor(2.0),
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                per_timestep=True,
                source_functor=self.name,
            ),
        }


def test_simulator_injects_selected_canonical_owner_output() -> None:
    simulator = RELICSimulator(
        {
            "conservation_checks": False,
            "strict_dependencies": True,
            "time_step": 86400.0,
            # Test-only placeholder state; the test declares its unit.
            "state_units": {"integration_fixture": "1"},
        }
    )
    simulator.initialize_state({"integration_fixture": torch.tensor(0.0)})
    simulator.register_functor(_GSIPhenologyOwner())
    simulator.register_functor(
        PhenologyFunctor(
            config={"canonical_model_owner": "gsi_phenology"}
        )
    )

    updates = simulator.step(dt=86400.0)

    assert torch.allclose(
        updates["diagnostics.phenology_state"], torch.tensor(0.8)
    )
    assert torch.allclose(
        updates["diagnostics.leaf_area_index"], torch.tensor(3.2)
    )
    assert torch.allclose(
        updates["diagnostics.phenology.senescence_factor"],
        torch.tensor(0.05),
    )


def test_selected_owner_remains_the_only_prognostic_writer() -> None:
    owner = _GSIPhenologyOwner()
    aggregator = PhenologyFunctor(
        config={"canonical_model_owner": "gsi_phenology"}
    )

    assert not set(owner.metadata.output_variables).intersection(
        aggregator.metadata.output_variables
    )
    assert all(
        name.startswith("diagnostics.")
        for name in aggregator.metadata.output_variables
    )
