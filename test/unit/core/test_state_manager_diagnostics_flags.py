import pytest
import torch

from relic.core.state import StateManager
from relic.core.state import conservation as state_conservation


def test_module_diagnostics_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELIC_ENABLE_STATE_DIAGNOSTICS", "0")
    sm = StateManager()
    sm.ingest_module_diagnostics("mod", {"x": torch.tensor(1.0)})
    assert sm._current_diagnostics == {}
    summary = sm.generate_conservation_report({}, current_time=0.0, store_history=False)
    assert summary.custom_checks == {}


def test_conservation_diagnostics_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELIC_ENABLE_CONSERVATION_DIAGNOSTICS", "0")
    sm = StateManager()
    summary = sm.generate_conservation_report({"carbon_pool": torch.tensor(1.0)}, current_time=0.0)
    assert summary.balance_error == {}
    assert summary.total_violations == 0
    assert summary.tolerance_value == sm.conservation_tolerance


def test_state_manager_exposes_only_the_canonical_conservation_report_api() -> None:
    sm = StateManager()

    assert callable(sm.generate_conservation_report)
    assert isinstance(sm.conservation_history, list)
    assert not hasattr(sm, "check_conservation")
    assert not hasattr(sm, "get_all_states")
    assert not hasattr(sm, "get_conservation_history")
    assert not hasattr(state_conservation, "check_conservation_transition")
    assert not hasattr(state_conservation, "get_conservation_history")
