"""Strict behavior tests for coupling components without default defaults."""

import pytest

from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.external.common.io import IOHandler, IOConfig
import relic.coupling.scales.scale_manager as scale_manager_module
from relic.core.managers.exceptions import ManagerInitializationError


def test_scale_manager_requires_parameter_manager(monkeypatch):
    """ScaleManager should error when ParameterManager is missing."""
    monkeypatch.setattr(
        scale_manager_module,
        "get_global_manager",
        lambda: (_ for _ in ()).throw(RuntimeError("ParameterManager is required")),
    )
    mgr = ScaleManager(parameter_manager=None)
    with pytest.raises(ManagerInitializationError, match="ParameterManager is required"):
        mgr.initialize()


def test_iohandler_requires_parameter_manager(monkeypatch):
    """IOHandler should error when ParameterManager is missing."""
    monkeypatch.setattr(
        IOHandler,
        "_get_parameter_manager",
        lambda self: None,
    )
    with pytest.raises(RuntimeError, match="IOHandler requires ParameterManager"):
        IOHandler(config=IOConfig(), parameter_manager=None)

