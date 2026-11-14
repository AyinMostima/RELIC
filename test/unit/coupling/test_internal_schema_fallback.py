import pytest
import pytest

from relic.coupling.external.common.schema import InternalSchema


def test_internal_schema_rejects_unknown_argument() -> None:
    with pytest.raises(TypeError):
        InternalSchema(parameter_manager=None, parameter_manager_resolver=None, unexpected_arg=True)
