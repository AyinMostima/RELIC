"""Quickstart and cache behavior tests for ScaleManager."""

import torch

from relic.coupling.scales.scale_manager import (
    ScaleManager,
    build_quickstart_scale_config,
)
from relic.core.state import StateManager
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace


def test_quickstart_config_reads_defaults_namespace():
    pm = get_global_manager()
    quickstart = build_quickstart_scale_config(parameter_manager=pm)

    defaults_ns = ParameterNamespace.COUPLING_DEFAULTS.value
    expected_mode = pm.get(
        "auto_scale_mode",
        namespace=defaults_ns,
        expected_type=str,
        required=True,
    )
    assert quickstart["mode"] == expected_mode
    assert isinstance(quickstart["enforce_conservation"], bool)
    assert "cf_strict_mode" in quickstart
    # The ScaleManager ledger tolerance is coupling.scales.defaults.precision.<mode>.tolerance of
    # the quickstart mode; coupling.scales.defaults.conservation_tolerance is not a ScaleManager setting.
    assert "conservation_tolerance" not in quickstart
    optional_keys = ("relative_tolerance", "absolute_tolerance", "coverage_epsilon")
    for key in optional_keys:
        value = pm.get(
            key,
            namespace=defaults_ns,
            units="dimensionless",
            expected_type=float,
            required=False,
        )
        if value is not None:
            assert key in quickstart
            assert quickstart[key] == float(value)


def test_get_param_uses_cache(monkeypatch):
    pm = get_global_manager()
    manager = ScaleManager(
        parameter_manager=pm,
        state_manager=StateManager(),
    )
    manager.initialize()
    manager.configure({})
    manager.validate()
    manager._param_cache.clear()

    call_counter = {"count": 0}
    original_get = manager.parameter_manager.get

    def counting_get(*args, **kwargs):
        call_counter["count"] += 1
        return original_get(*args, **kwargs)

    monkeypatch.setattr(manager.parameter_manager, "get", counting_get)

    ns = ParameterNamespace.COUPLING_SCALES_TEMPORAL.value
    first = manager._get_param(
        "coverage_threshold",
        namespace=ns,
        units="dimensionless",
        expected_type=float,
    )
    second = manager._get_param(
        "coverage_threshold",
        namespace=ns,
        units="dimensionless",
        expected_type=float,
    )

    assert torch.is_tensor(torch.as_tensor(first)) or isinstance(first, float)
    assert first == second
    assert call_counter["count"] == 1
