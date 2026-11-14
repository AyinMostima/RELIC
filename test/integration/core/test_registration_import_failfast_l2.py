import sys
import types

import pytest
import torch

from relic.core.functor import FunctorBase, FunctorMetadata, ProcessCategory
from relic.core.simulator import RELICSimulator


class _MinimalFunctor(FunctorBase):
    def __init__(self, name: str = "minimal_import_failfast") -> None:
        metadata = FunctorMetadata(
            name=name,
            category=ProcessCategory.PHYSICAL_FIELD,
            description="L2 test functor for registration fail-fast behavior.",
            references=["RELIC registration boundary test fixture"],
            output_variables=["test.delta"],
        )
        super().__init__(name=name, metadata=metadata, config={})

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"test.delta": torch.tensor(0.0)}


def test_register_functor_fails_fast_on_bridge_import_error_l2(monkeypatch) -> None:
    simulator = RELICSimulator()
    functor = _MinimalFunctor()
    stub = types.ModuleType("relic.core.functor.functor_bridges")
    monkeypatch.setitem(sys.modules, "relic.core.functor.functor_bridges", stub)

    with pytest.raises(
        RuntimeError,
        match="Failed to import bridge functor classes required for registration",
    ):
        simulator.register_functor(functor)
