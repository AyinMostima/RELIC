import pytest
import torch

from relic.core.state import StateManager, UpdateMode
from relic.coupling.external.common import (
    ExternalAdapter,
    DataSource,
    ValidationReport,
)


class DummyAdapter(ExternalAdapter):
    """Minimal external adapter to verify update_mode semantics."""

    def __init__(self) -> None:
        super().__init__(source=DataSource.WRF, name="DummyAdapter", parameter_manager=None)
        self._connected = True

    def open(self, *args, **kwargs):
        self._connected = True

    def close(self):
        self._connected = False

    def discover(self, variables=None, **kwargs):
        return {}

    def read(self, time_window=None, spatial_window=None, variables=None):
        return {"air_temperature": torch.tensor(3.0)}

    def get_variable_mapping(self, variables):
        vars_list = variables or ["air_temperature"]
        return {v: v for v in vars_list}

    def transform_to_internal(self, external_data, mapping):
        return {mapping.get(k, k): v for k, v in external_data.items()}

    def validate_data(self, data):
        return ValidationReport(source_path="dummy", validation_time="")

    def forward(self, state, time, dt, **kwargs):
        variables = kwargs.get("variables") or ["air_temperature"]
        external = self.read(variables=variables)
        mapping = self.get_variable_mapping(variables)
        internal = self.transform_to_internal(external, mapping)
        return self._wrap_updates(internal)


@pytest.mark.filterwarnings("ignore:No scientific references provided for functor DummyAdapter")
def test_external_adapter_uses_replace_mode():
    adapter = DummyAdapter()
    updates = adapter.forward(state={}, time=0.0, dt=0.0)
    item = updates["air_temperature"]
    assert item.mode == UpdateMode.REPLACE
    state_manager = StateManager()
    state_manager.set_state("air_temperature", torch.tensor(1.0))
    state_manager.apply_updates(updates)
    stored = state_manager.get_state()["air_temperature"]
    assert torch.equal(stored, torch.tensor(3.0, device=stored.device))
