"""L1 tests for CalibrationResult persistence.

Module under test: relic/parameters/calibration/results.py. A result that
cannot be written or read raises; it is not reported as False or None.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from relic.parameters.calibration.results import (
    CalibrationMetrics,
    CalibrationReport,
    CalibrationResult,
    CalibrationStatus,
    SensitivityAnalysis,
)


def _result(**overrides) -> CalibrationResult:
    values = dict(
        method="differential_evolution",
        target_parameters=["test_param"],
        initial_parameters={"test_param": 0.5},
        optimized_parameters={"test_param": 0.25},
        parameter_bounds={"test_param": [0.0, 1.0]},
        objective_value=0.125,
        convergence_history=[0.5, 0.125],
        iteration_count=2,
        elapsed_time=0.5,
        success=True,
        status=CalibrationStatus.SUCCESS,
        termination_reason="Converged",
    )
    values.update(overrides)
    return CalibrationResult(**values)


def test_result_record_fields_have_no_fabricated_defaults() -> None:
    fields = CalibrationResult.__dataclass_fields__
    for name in ("iteration_count", "elapsed_time", "status", "termination_reason"):
        assert fields[name].default is dataclasses.MISSING, name
    assert SensitivityAnalysis.__dataclass_fields__["method"].default is dataclasses.MISSING
    with pytest.raises(TypeError, match="method"):
        SensitivityAnalysis(parameter_sensitivities={"a": 0.5})
    assert SensitivityAnalysis(parameter_sensitivities={"a": 0.5}, method="morris").method == "morris"


def test_result_success_and_status_must_agree() -> None:
    with pytest.raises(ValueError, match="status"):
        _result(success=True, status=CalibrationStatus.CONVERGENCE_FAILURE)
    with pytest.raises(ValueError, match="status"):
        _result(success=False, status=CalibrationStatus.SUCCESS)
    failed = _result(success=False, status=CalibrationStatus.CONVERGENCE_FAILURE,
                     termination_reason="Max iterations reached")
    assert failed.status is CalibrationStatus.CONVERGENCE_FAILURE
    loaded = CalibrationResult.from_dict(failed.to_dict())
    assert loaded.status is CalibrationStatus.CONVERGENCE_FAILURE
    assert loaded.termination_reason == "Max iterations reached"


def test_result_round_trips_through_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "calibration_results.json"

    assert _result().save_to_file(path) is None
    loaded = CalibrationResult.load_from_file(path)

    assert loaded.optimized_parameters == {"test_param": 0.25}
    assert loaded.objective_value == pytest.approx(0.125)
    assert loaded.convergence_history == [0.5, 0.125]


def test_unwritable_result_path_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")

    with pytest.raises(OSError):
        _result().save_to_file(blocker / "calibration_results.json")


def test_missing_result_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CalibrationResult.load_from_file(tmp_path / "absent.json")


def test_corrupt_result_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "calibration_results.json"
    path.write_text('{"method": "differential_evolution",', encoding="utf-8")

    with pytest.raises(ValueError):
        CalibrationResult.load_from_file(path)


# --- Metrics record and performance rating (Moriasi et al. 2007, Table 4) -------------------


def _metrics(nse: float, pbias: float) -> CalibrationMetrics:
    return CalibrationMetrics(
        mse=0.01, rmse=0.1, mae=0.08, r_squared=0.9, nash_sutcliffe=nse,
        bias=0.0, percent_bias=pbias, correlation=0.95,
    )


def test_metrics_record_holds_no_never_computed_statistic() -> None:
    fields = CalibrationMetrics.__dataclass_fields__
    for name in ("kling_gupta", "index_of_agreement", "volumetric_efficiency"):
        assert name not in fields, name


@pytest.mark.parametrize(
    ("category", "nse", "pbias", "grade"),
    [
        ("streamflow", 0.80, 5.0, "Very good"),
        ("streamflow", 0.70, 5.0, "Good"),       # NSE rating limits the grade
        ("streamflow", 0.80, -20.0, "Satisfactory"),
        ("sediment", 0.80, -20.0, "Good"),        # 15 <= |PBIAS| < 30 for sediment
        ("nitrogen", 0.80, 20.0, "Very good"),    # |PBIAS| < 25 for N and P
        ("phosphorus", 0.80, 45.0, "Satisfactory"),
        ("streamflow", 0.50, 0.0, "Unsatisfactory"),  # NSE <= 0.50
        ("streamflow", 0.90, 25.0, "Unsatisfactory"),  # |PBIAS| >= 25 for streamflow
    ],
)
def test_performance_grade_is_the_lowest_moriasi_table_4_rating(category, nse, pbias, grade) -> None:
    assert _metrics(nse, pbias).get_performance_grade(variable_category=category) == grade


def test_performance_grade_requires_a_declared_variable_category() -> None:
    with pytest.raises(TypeError, match="variable_category"):
        _metrics(0.8, 5.0).get_performance_grade()
    with pytest.raises(ValueError, match="streamflow"):
        _metrics(0.8, 5.0).get_performance_grade(variable_category="carbon_flux")


def test_report_summary_grades_with_the_declared_category() -> None:
    result = _result(metrics=_metrics(0.80, -20.0))
    with pytest.raises(TypeError, match="variable_category"):
        CalibrationReport(result=result)
    assert CalibrationReport(result=result, variable_category="sediment").generate_summary()[
        "performance_grade"] == "Good"
