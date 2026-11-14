import pytest

from relic.coupling.external.common.io import IOConfig, IOHandler
import relic.coupling.external.common.io as io_module


def test_iohandler_requires_parameter_manager(monkeypatch) -> None:
    monkeypatch.setattr(io_module, "get_global_manager", lambda: None)
    config = IOConfig(
        parallel=False,
        enable_cache=False,
        validate_checksums=False,
    )
    with pytest.raises(RuntimeError, match="IOHandler requires ParameterManager"):
        IOHandler(
            config=config,
            parameter_manager=None,
            parameter_manager_resolver=lambda: None,
        )
