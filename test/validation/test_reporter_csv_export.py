# -*- coding: utf-8 -*-
"""
Tests for ValidationReporter CSV export.

ASCII-only comments and prints.
"""
from relic.validation.framework.models import (
    ValidationReportModel,
    ValidationResultModel,
    ValidationLevel,
    ValidationCode,
)
from relic.validation.framework.reporters import ValidationReporter
import pytest


def make_sample_report():
    r1 = ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.MASS_BALANCE_OK_WATER,
        message="Water conservation satisfied",
        layer="L2",
        scope="water_mass_conservation",
        tags=["conservation", "water"],
        metrics={"water_balance_residual_max": 0.0},
    )
    r2 = ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.CF_UNIT_MISMATCH,
        message="Unit mismatch detected",
        layer="L1",
        scope="cf_unit_check",
        tags=["cf"],
        metrics={"mismatch_count": 2},
    )
    report = ValidationReportModel(results=[r1, r2])
    # Force summary update
    report._update_summary()
    return report


def test_reporter_to_csv_string():
    reporter = ValidationReporter()
    report = make_sample_report()
    csv_str = reporter.to_csv(report)
    # Header present
    assert "level,code,layer,scope,passed,message,tags,metrics,details_count" in csv_str
    # Both result codes present
    assert "MASS_BALANCE_OK_WATER" in csv_str
    assert "CF_UNIT_MISMATCH" in csv_str
    # Passed flag should be true for *_OK item
    assert "true" in csv_str


def test_reporter_save_csv_local_path():
    reporter = ValidationReporter()
    report = make_sample_report()
    from pathlib import Path
    out_dir = Path("build") / "test_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "validation_results.csv"
    reporter.save_report(report, out_file, format="csv")
    content = out_file.read_text(encoding="utf-8")
    assert "validation_results" not in content  # not JSON
    assert "MASS_BALANCE_OK_WATER" in content
    assert "CF_UNIT_MISMATCH" in content
    # Clean up artifact
    try:
        out_file.unlink(missing_ok=True)
    except Exception:
        pass



def test_reporter_csv_does_not_replace_unserializable_metrics():
    reporter = ValidationReporter()
    report = make_sample_report()
    # Mutation bypasses the model validation of metric values.
    report.results[0].metrics["tensor_metric"] = object()

    with pytest.raises(TypeError, match="not JSON serializable"):
        reporter.to_csv(report)
