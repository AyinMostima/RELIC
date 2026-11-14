"""External adapter restart binding (F02).

The base ``ExternalAdapter`` serializes its source/model binding through the
standard ``state_dict`` extra state, so a checkpoint produced by an adapter
bound to one external source is refused by an adapter bound to another one,
while a same-binding round trip is accepted. Connection handles and cached
datasets are resources and are not part of the restart contract.
"""

from __future__ import annotations

import pytest
import torch

from relic.coupling.external.common import (
    DataSource,
    ExternalAdapter,
    ValidationReport,
)


class _DummyAdapter(ExternalAdapter):
    def __init__(self, source=DataSource.WRF, model_id=None) -> None:
        super().__init__(
            source=source, name="DummyAdapter", model_id=model_id, parameter_manager=None,
        )
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
        return {name: name for name in (variables or ["air_temperature"])}

    def transform_to_internal(self, external_data, mapping):
        return {mapping.get(k, k): v for k, v in external_data.items()}

    def validate_data(self, data):
        return ValidationReport(source_path="dummy", validation_time="")

    def forward(self, state, time, dt, **kwargs):
        external = self.read(variables=["air_temperature"])
        return self._wrap_updates(self.transform_to_internal(
            external, self.get_variable_mapping(None),
        ))


pytestmark = pytest.mark.filterwarnings(
    "ignore:No scientific references provided for functor DummyAdapter"
)


def test_state_dict_carries_the_source_binding_and_round_trips():
    adapter = _DummyAdapter(model_id="wrf-v4")
    state = adapter.state_dict()
    assert state["_extra_state"] == {
        "format": ExternalAdapter.EXTRA_STATE_FORMAT,
        "version": 1,
        "source": DataSource.WRF.value,
        "model_id": "wrf-v4",
    }
    other = _DummyAdapter(model_id="wrf-v4")
    other.load_state_dict(state, strict=True)
    assert other.state_dict()["_extra_state"] == state["_extra_state"]
    # Connection state is a resource, not restart state.
    other.close()
    other.load_state_dict(state, strict=True)
    assert other.connected is False


@pytest.mark.parametrize(
    "target",
    [
        _DummyAdapter(source=DataSource.WRF, model_id="wrf-v3"),
        _DummyAdapter(source=DataSource.CESM, model_id="wrf-v4"),
    ],
)
def test_restore_into_another_source_or_model_is_refused(target):
    state = _DummyAdapter(model_id="wrf-v4").state_dict()
    with pytest.raises(ValueError, match="bound to source/model"):
        target.load_state_dict(state, strict=True)
    assert target.state_dict()["_extra_state"]["model_id"] == target.model_id


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.pop("model_id"),
        lambda s: s.__setitem__("extra", 1),
        lambda s: s.__setitem__("format", "other.format"),
        lambda s: s.__setitem__("version", 2),
    ],
)
def test_malformed_binding_is_rejected(mutation):
    adapter = _DummyAdapter(model_id="wrf-v4")
    state = adapter.state_dict()
    mutation(state["_extra_state"])
    with pytest.raises(ValueError):
        _DummyAdapter(model_id="wrf-v4").load_state_dict(state, strict=True)
