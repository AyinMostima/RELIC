import torch

from relic.coupling.external.common import ExternalAdapter, DataSource
from relic.core.functor import (
    ConflictResolutionPolicy,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.state import StateManager, UpdateItem, UpdateMode


class DummyExternalAdapter(ExternalAdapter):
    """Minimal adapter to verify update_mode semantics."""

    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="dummy_adapter",
            category=ProcessCategory.SCALE_COUPLING,
            description="Test adapter for replace-mode verification",
            references=["test-dummy"],
            mathematical_expression="T_air = T_external",
            input_variables=[],
            output_variables=["air_temperature"],
            execution_phase=ExecutionPhase.COUPLING,
            temporal_scale=TemporalScale.SECOND,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FORCING,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        super().__init__(
            source=DataSource.WRF,
            name="dummy_adapter",
            parameter_manager=None,
            metadata=metadata,
            provided_variables=["air_temperature"],
        )
        self._connected = True

    def discover(self, *args, **kwargs):
        return {}

    def open(self, *args, **kwargs):
        self._connected = True
        return {}

    def read(self, *args, **kwargs):
        return {}

    def transform_to_internal(self, data, mapping=None, policy=None):
        return {"air_temperature": torch.tensor(280.0, dtype=torch.float32)}

    def close(self) -> None:
        self._connected = False

    def forward(self, state, time, dt, **kwargs):
        return self._wrap_updates({"air_temperature": torch.tensor(280.0, dtype=torch.float32)})


def test_external_adapter_returns_replace_mode():
    adapter = DummyExternalAdapter()
    updates = adapter.forward({}, time=0.0, dt=1.0)

    assert isinstance(updates["air_temperature"], UpdateItem)
    assert updates["air_temperature"].mode == UpdateMode.REPLACE


def test_replace_mode_applies_absolute_state():
    adapter = DummyExternalAdapter()
    updates = adapter.forward({}, time=0.0, dt=1.0)

    manager = StateManager()
    manager.set_state("air_temperature", torch.tensor(300.0, dtype=torch.float32))

    manager.apply_updates(updates)
    assert torch.isclose(manager.state_vars["air_temperature"], torch.tensor(280.0, dtype=torch.float32))
