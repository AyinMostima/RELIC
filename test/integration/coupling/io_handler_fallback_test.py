import pytest

from relic.coupling.external.common.io import IOConfig, IOHandler
import relic.coupling.external.common.io as io_module


class _FailingManager:
    def get(self, *args, **kwargs):
        raise KeyError("missing")


def test_iohandler_requires_parameter_manager(monkeypatch):
    monkeypatch.setattr(io_module, "get_global_manager", lambda: None)
    with pytest.raises(RuntimeError, match="IOHandler requires ParameterManager"):
        IOHandler(config=IOConfig(), parameter_manager=None, parameter_manager_resolver=None)


def test_iohandler_raises_on_parameter_lookup_failure():
    with pytest.raises(KeyError, match="Missing required parameter"):
        IOHandler(config=IOConfig(), parameter_manager=_FailingManager())
