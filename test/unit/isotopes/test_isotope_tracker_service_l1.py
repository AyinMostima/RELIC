"""Fail-closed tests for the runtime-unbound isotope service boundary."""

import inspect

import pytest

from relic.scientific.isotopes.config import get_default_isotope_config
from relic.scientific.isotopes.config.models import Nitrogen15Config
import relic.scientific.isotopes.tracker_service as service_module


def _assert_constructor_rejects_without_effects(
    config,
    monkeypatch,
) -> None:
    effects = []

    class _ParameterManager:
        def get(self, key, **_kwargs):
            effects.append(("parameter", key))
            return 1.0

    class _StateManager:
        def get_state(self):
            effects.append(("state", None))
            return {}

    class _Mapper:
        def __init__(self):
            effects.append(("mapper", None))

    monkeypatch.setattr(
        service_module,
        "get_global_manager",
        lambda: _ParameterManager(),
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "VariableNameMapper",
        _Mapper,
        raising=False,
    )

    with pytest.raises(
        service_module.IsotopeTrackingError,
        match=r"\[N\].*runtime_unbound",
    ):
        service_module.IsotopeTrackerService(config, _StateManager())

    assert effects == []


def test_root_config_rejects_before_constructor_side_effects(monkeypatch) -> None:
    config = get_default_isotope_config()
    _assert_constructor_rejects_without_effects(config, monkeypatch)


def test_internal_nitrogen_config_is_not_a_second_runtime_entry(monkeypatch) -> None:
    config = Nitrogen15Config()
    _assert_constructor_rejects_without_effects(config, monkeypatch)


def test_private_isotope_state_types_are_absent() -> None:
    for name in ("IsotopePool", "IsotopeEvent"):
        assert not hasattr(service_module, name)
        assert name not in service_module.__all__


def test_service_surface_contains_no_scientific_runtime_methods() -> None:
    forbidden = {
        "process_updates",
        "process_state_update",
        "initialize_pools",
        "get_current_pools",
        "get_current_ratios",
        "check_conservation",
        "update_isotope_tracking",
        "connect_runtime",
    }
    assert forbidden.isdisjoint(service_module.IsotopeTrackerService.__dict__)
    source = inspect.getsource(service_module.IsotopeTrackerService)
    assert "isotope_pools" not in source
