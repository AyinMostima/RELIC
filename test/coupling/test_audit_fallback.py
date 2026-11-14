from pathlib import Path

import pytest

from relic.coupling.core import audit
from relic.parameters import get_global_manager, reset_global_manager


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def test_audit_record_requires_parameter_manager(monkeypatch):
    """AuditRecord should raise when ParameterManager is unavailable."""
    monkeypatch.setattr(
        audit,
        "_get_parameter_manager_required",
        lambda parameter_manager=None: (_ for _ in ()).throw(RuntimeError("missing manager")),
    )

    with pytest.raises(RuntimeError):
        audit.AuditRecord(operation="test")


def test_audit_record_never_defaults_or_accepts_dataset_cf_success():
    record = audit.AuditRecord(operation="scope_check")
    assert record.cf_compliant is False
    assert record.metadata_updates["dataset_conformance_checked"] is False
    assert record.metadata_updates["cf_validation_scope"] == "not_evaluated"

    with pytest.raises(ValueError, match="dataset-level CF conformance"):
        audit.AuditRecord(operation="forbidden_claim", cf_compliant=True)


def _configure_jsonl(pm, *, enabled: bool, path: Path) -> None:
    pm.set_value(
        "coupling.scales.audit.jsonl_enabled",
        enabled,
        units="dimensionless",
        validate=True,
    )
    pm.set_value(
        "coupling.scales.audit.jsonl_path",
        str(path),
        units="dimensionless",
        validate=False,
    )


def test_audit_jsonl_disabled_does_not_write(tmp_path: Path):
    reset_global_manager()
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    jsonl_path = tmp_path / "audit.jsonl"
    _configure_jsonl(pm, enabled=False, path=jsonl_path)

    collector = audit.AuditCollector()
    record = audit.AuditRecord(operation="jsonl_disabled")
    collector.add_record(record)

    assert not jsonl_path.exists()


def test_audit_jsonl_enabled_writes_record(tmp_path: Path):
    reset_global_manager()
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    jsonl_path = tmp_path / "audit.jsonl"
    _configure_jsonl(pm, enabled=True, path=jsonl_path)

    collector = audit.AuditCollector()
    record = audit.AuditRecord(operation="jsonl_enabled")
    collector.add_record(record)

    assert jsonl_path.exists()
    content = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    assert "\"operation\": \"jsonl_enabled\"" in content[0]


def test_audit_sinks_raise_instead_of_dropping_records(tmp_path):
    """A failing JSONL sink raises; an audit is never dropped."""
    import pytest

    from relic.coupling.core.audit import AuditCollector, AuditRecord

    collector = AuditCollector()
    record = AuditRecord(operation="regrid", source_scale="a", target_scale="b", variables=["v"])
    collector._jsonl_enabled = True
    blocked = tmp_path / "file_not_dir"
    blocked.write_text("x", encoding="ascii")
    collector._jsonl_path = blocked / "audit.jsonl"
    with pytest.raises(OSError):
        collector._emit_jsonl(record)


def test_audit_records_stay_in_the_collector_and_never_reach_state_diagnostics(tmp_path):
    """Audit records are read through the AuditCollector API (and its JSONL sink): no path sends
    the audit summary mapping into the StateManager diagnostics channel, whose stored diagnostics
    are real scalars; ScaleManager creates its collector without a StateManager."""
    import inspect

    import torch

    from relic.coupling.core.audit import AuditCollector, AuditRecord
    from relic.coupling.scales.scale_manager import ScaleManager
    from relic.core.state import StateManager

    parameters = inspect.signature(AuditCollector.__init__).parameters
    assert "enable_state_sync" not in parameters and "state_manager" not in parameters
    assert not hasattr(AuditCollector, "_emit_to_state_manager")
    state_manager = StateManager()
    manager = ScaleManager(device=torch.device("cpu"), state_manager=state_manager)
    manager.initialize()
    assert not hasattr(manager.audit_collector, "state_manager")
    manager.audit_collector.add_record(
        AuditRecord(operation="aggregate", source_scale="a", target_scale="b", variables=["v"])
    )
    assert [record.operation for record in manager.audit_collector.get_by_operation("aggregate")] == ["aggregate"]
    current = state_manager.get_diagnostics()["current"]
    assert not any(key.endswith(".audit") for key in current)
