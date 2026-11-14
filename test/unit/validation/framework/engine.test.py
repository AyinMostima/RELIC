"""ValidationEngine reports metrics that were actually counted, and no inert settings."""

import dataclasses
from datetime import datetime

import pytest

from relic.validation.framework.engine import ExecutionMetrics, ValidationEngine
from relic.validation.framework.models import ValidationLevel

pytestmark = pytest.mark.unit


def test_l1_report_metrics_match_the_reported_results():
    """The report is generated after counting: passed + failed equals the result count."""
    report = ValidationEngine().run(layers=["L1"], include_metrics=True)
    meta = report.execution_metadata
    results = report.results

    passed = sum(1 for r in results if r.level in (ValidationLevel.INFO, ValidationLevel.WARNING))
    assert len(results) > 0
    assert meta["rules_passed"] == passed
    assert meta["rules_failed"] == len(results) - passed
    assert meta["rules_passed"] + meta["rules_failed"] == len(results)
    assert meta["total_rules_executed"] == sum(1 for r in results if r.scope)
    assert meta["end_time"] is not None
    assert meta["execution_time_seconds"] is not None
    assert meta["execution_time_seconds"] >= 0.0
    assert meta["execution_time_seconds"] == pytest.approx(
        (
            datetime.fromisoformat(meta["end_time"]) - datetime.fromisoformat(meta["start_time"])
        ).total_seconds()
    )


def test_execution_metrics_fields_have_no_defaults():
    fields = {f.name: f for f in dataclasses.fields(ExecutionMetrics)}

    assert "rules_skipped" not in fields
    for name in (
        "start_time",
        "end_time",
        "total_rules_executed",
        "rules_passed",
        "rules_failed",
        "execution_time_seconds",
        "layer_execution_times",
    ):
        assert fields[name].default is dataclasses.MISSING, name
        assert fields[name].default_factory is dataclasses.MISSING, name


@pytest.mark.parametrize("setting", ["max_workers", "execution_timeout"])
def test_engine_has_no_inert_worker_or_timeout_setting(setting):
    with pytest.raises(TypeError, match=setting):
        ValidationEngine(**{setting: 1})

    engine = ValidationEngine()
    assert not hasattr(engine, setting)
    assert "engine_configuration" not in engine.get_registry_info()
