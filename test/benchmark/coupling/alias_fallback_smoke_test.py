import pytest

from relic.coupling.external.common.io import IOConfig, IOHandler
import relic.coupling.external.common.io as io_module
from relic.core.functor import ProcessCategory


def test_alias_and_io_default_coexist(monkeypatch):
    monkeypatch.setattr(io_module, "get_global_manager", lambda: None)
    category = ProcessCategory("scale_coupling")
    with pytest.raises(RuntimeError, match="IOHandler requires ParameterManager"):
        IOHandler(config=IOConfig(), parameter_manager=None, parameter_manager_resolver=None)
    assert category is ProcessCategory.SCALE_COUPLING

