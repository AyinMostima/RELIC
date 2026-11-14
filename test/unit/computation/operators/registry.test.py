"""L1 tests for OperatorRegistry registration (relic/computation/operators/registry.py).

Registration builds one instance from the YAML defaults to record the
operator metadata. A failure of that construction is a registration error;
it is never logged and replaced by an entry without metadata.
"""

from __future__ import annotations

import pytest

from relic.computation.operators.base import OperatorBase
from relic.computation.operators.registry import OperatorRegistry

PROBE_NAME = "probe.missing_parameter_operator"


class _MissingParameterOperator(OperatorBase):
    """Operator whose required YAML key does not exist."""

    operator_type = "probe"

    def __init__(self, name: str = "probe", config=None, **kwargs) -> None:
        super().__init__(
            name=name,
            config=config,
            namespace="computation.operators.no_such_probe",
        )
        self.dx = self.get_parameter("dx")

    def apply(self, field):
        return field


@pytest.fixture
def _registry_cleanup():
    yield
    OperatorRegistry._registry.pop(PROBE_NAME, None)


def test_metadata_extraction_failure_is_a_registration_error(_registry_cleanup) -> None:
    with pytest.raises(KeyError, match="computation.operators.no_such_probe.dx"):
        OperatorRegistry.register(PROBE_NAME, _MissingParameterOperator)

    assert not OperatorRegistry.is_registered(PROBE_NAME)


def test_every_registered_operator_entry_carries_metadata() -> None:
    import relic.computation.operators  # noqa: F401 - registers the operators

    names = OperatorRegistry.list_operators()
    assert names
    for name in names:
        assert OperatorRegistry.get_entry(name).metadata is not None, name
