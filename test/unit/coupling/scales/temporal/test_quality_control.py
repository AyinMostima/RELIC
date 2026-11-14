import dataclasses

import pytest
import torch

from relic.coupling.scales.temporal.quality_control import (
    CoverageEvaluator,
    GapFillingMethod,
    OutlierDetector,
    OutlierMethod,
    QualityConfig,
    QualityController,
)
from relic.coupling.scales.temporal.core import TimeCoordinate
from relic.parameters import get_global_manager

_QC = "coupling.scales.temporal.quality_control"
_EXPECTED_KEYS = {
    "coverage_threshold": "coupling.scales.temporal.coverage_threshold",
    "max_gap_length": f"{_QC}.max_gap_length",
    "gap_filling_method": f"{_QC}.gap_filling_method",
    "outlier_method": f"{_QC}.outlier_method",
    "outlier_threshold": f"{_QC}.outlier_threshold",
    "outlier_min_valid_samples": f"{_QC}.outlier_min_valid_samples",
    "iqr_fence_factor": f"{_QC}.iqr_fence_factor",
    "grade_excellent_min_fraction": f"{_QC}.grade_excellent_min_fraction",
    "grade_excellent_max_gap_length": f"{_QC}.grade_excellent_max_gap_length",
    "grade_good_min_fraction": f"{_QC}.grade_good_min_fraction",
    "grade_acceptable_min_fraction": f"{_QC}.grade_acceptable_min_fraction",
    "grade_poor_min_fraction": f"{_QC}.grade_poor_min_fraction",
}


class _CapturingManager:
    """Record every lookup and serve the canonical ParameterManager values."""

    def __init__(self, overrides=None):
        self.calls = []
        self._overrides = dict(overrides or {})
        self._canonical = get_global_manager()

    def get(self, key, **kwargs):
        self.calls.append((key, kwargs))
        if key in self._overrides:
            return self._overrides[key]
        return self._canonical.get(key, **kwargs)


def test_coverage_evaluator_reports_missing_fraction():
    """Ensure CoverageEvaluator returns a finite missing_fraction instead of raising NameError."""
    data = torch.tensor([1.0, torch.nan, 2.0, torch.nan], dtype=torch.float32)
    time_coord = TimeCoordinate(
        values=torch.arange(4, dtype=torch.float32),
        units="days since 2000-01-01",
        calendar="gregorian",
    )
    evaluator = CoverageEvaluator(QualityConfig.from_parameter_manager())

    coverage, metrics = evaluator.evaluate_coverage(data, time_coord)

    expected_coverage = torch.tensor(0.5, dtype=coverage.dtype)
    assert torch.isclose(coverage, expected_coverage), "Coverage fraction should be 0.5"
    assert metrics.coverage_fraction == 0.5
    assert metrics.missing_fraction == 0.5


def test_quality_config_has_no_python_threshold_defaults():
    """Every threshold is required; no dataclass default can act as a fallback."""
    fields = dataclasses.fields(QualityConfig)

    assert {item.name for item in fields} == set(_EXPECTED_KEYS)
    for item in fields:
        assert item.default is dataclasses.MISSING, item.name
        assert item.default_factory is dataclasses.MISSING, item.name
    with pytest.raises(TypeError):
        QualityConfig()  # type: ignore[call-arg]


def test_quality_config_reads_every_threshold_as_required_parameter():
    manager = _CapturingManager()

    config = QualityConfig.from_parameter_manager(manager)

    requested = {key for key, _kwargs in manager.calls}
    assert requested == set(_EXPECTED_KEYS.values())
    for _key, kwargs in manager.calls:
        assert kwargs["required"] is True
        assert "default" not in kwargs
    # Values held by the YAML parameter database (scales_parameters.yaml).
    assert config.coverage_threshold == pytest.approx(0.8)
    assert config.max_gap_length == 7
    assert config.gap_filling_method is GapFillingMethod.LINEAR
    assert config.outlier_method is OutlierMethod.ZSCORE
    assert config.outlier_threshold == pytest.approx(3.0)
    assert config.outlier_min_valid_samples == 10
    assert config.iqr_fence_factor == pytest.approx(1.5)
    assert config.grade_excellent_min_fraction == pytest.approx(0.95)
    assert config.grade_excellent_max_gap_length == 1
    assert config.grade_good_min_fraction == pytest.approx(0.8)
    assert config.grade_acceptable_min_fraction == pytest.approx(0.6)
    assert config.grade_poor_min_fraction == pytest.approx(0.4)


def test_quality_config_propagates_missing_parameter_without_fallback():
    class _MissingManager(_CapturingManager):
        def get(self, key, **kwargs):
            if key == _EXPECTED_KEYS["outlier_threshold"]:
                raise KeyError(key)
            return super().get(key, **kwargs)

    with pytest.raises(KeyError, match="outlier_threshold"):
        QualityConfig.from_parameter_manager(_MissingManager())


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("coverage_threshold", 1.5, "coverage_threshold"),
        ("coverage_threshold", 0.0, "coverage_threshold"),
        ("max_gap_length", -1, "max_gap_length"),
        ("max_gap_length", 2.5, "max_gap_length"),
        ("gap_filling_method", "spline", "gap_filling_method"),
        ("gap_filling_method", "seasonal", "gap_filling_method"),
        ("outlier_method", "isolation_forest", "outlier_method"),
        ("outlier_threshold", 0.0, "outlier_threshold"),
        ("outlier_min_valid_samples", 1, "outlier_min_valid_samples"),
        ("iqr_fence_factor", 0.0, "iqr_fence_factor"),
        ("grade_good_min_fraction", 0.99, "grade"),
        ("grade_poor_min_fraction", 0.0, "grade"),
        ("grade_excellent_max_gap_length", 8, "grade_excellent_max_gap_length"),
    ],
)
def test_quality_config_rejects_invalid_or_unimplemented_values(
    field_name,
    value,
    message,
):
    manager = _CapturingManager(overrides={_EXPECTED_KEYS[field_name]: value})

    with pytest.raises((TypeError, ValueError), match=message):
        QualityConfig.from_parameter_manager(manager)


def test_quality_controller_default_config_comes_from_parameter_manager():
    controller = QualityController()

    assert controller.config == QualityConfig.from_parameter_manager()


def _time_coordinate(count: int) -> TimeCoordinate:
    return TimeCoordinate(
        values=torch.arange(count, dtype=torch.float64),
        units="days since 2000-01-01",
        calendar="noleap",
    )


def test_quality_grades_follow_the_parameter_cutoffs():
    """Seven of ten samples valid, longest gap one sample."""
    data = torch.tensor(
        [1.0, float("nan"), 2.0, 3.0, float("nan"), 4.0, 5.0, float("nan"), 6.0, 7.0],
        dtype=torch.float64,
    )
    default = CoverageEvaluator(QualityConfig.from_parameter_manager())
    _coverage, metrics = default.evaluate_coverage(data, _time_coordinate(10))
    assert metrics.quality_level.value == "acceptable"  # 0.6 <= 0.7 < 0.8

    relaxed = CoverageEvaluator(
        QualityConfig.from_parameter_manager(
            _CapturingManager(overrides={_EXPECTED_KEYS["grade_good_min_fraction"]: 0.65})
        )
    )
    _coverage, metrics = relaxed.evaluate_coverage(data, _time_coordinate(10))
    assert metrics.quality_level.value == "good"


def test_outlier_sample_minimum_and_iqr_fence_come_from_parameters():
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], dtype=torch.float64)
    iqr_key = _EXPECTED_KEYS["outlier_method"]
    min_key = _EXPECTED_KEYS["outlier_min_valid_samples"]
    fence_key = _EXPECTED_KEYS["iqr_fence_factor"]

    def detector(overrides):
        manager = _CapturingManager(overrides={iqr_key: "iqr", **overrides})
        return OutlierDetector(QualityConfig.from_parameter_manager(manager))

    time_coord = _time_coordinate(6)
    # Six samples are below the configured minimum of ten: no detection.
    assert not detector({}).detect_outliers(data, time_coord).any()
    # q1 = 2.25, q3 = 4.75, IQR = 2.5, upper fence 4.75 + 1.5 * 2.5 = 8.5.
    assert detector({min_key: 5}).detect_outliers(data, time_coord).tolist() == [
        False, False, False, False, False, True,
    ]
    # Fence factor 40: upper fence 104.75 keeps 100 inside.
    assert not detector({min_key: 5, fence_key: 40.0}).detect_outliers(
        data, time_coord
    ).any()


@pytest.mark.parametrize("scale", [1.0, 1.0e-14])
def test_outlier_detection_is_invariant_to_data_scale(scale):
    """A fixed absolute spread guard would call small-magnitude data constant."""
    data = torch.full((12,), scale, dtype=torch.float64)
    data[-1] = 100.0 * scale
    detector = OutlierDetector(QualityConfig.from_parameter_manager())

    mask = detector.detect_outliers(data, _time_coordinate(12))

    # z = (x_12 - mean) / std = 3.175 > 3.0 for this sample at any scale.
    assert mask.tolist() == [False] * 11 + [True]


def test_assess_quality_handles_one_series_below_the_coverage_threshold():
    """A 1-D series yields a 0-d coverage; the insufficient branch must not crash."""
    nan = float("nan")
    data = torch.tensor(
        [280.0] + [nan] * 6 + [281.0, 282.0, 283.0],
        dtype=torch.float64,
    )

    with pytest.warns(UserWarning, match="insufficient coverage"):
        _processed, flags, metrics = QualityController().assess_quality(
            data,
            _time_coordinate(10),
        )

    assert metrics.coverage_fraction == pytest.approx(0.4)
    assert metrics.insufficient_coverage_count == 1
    # Exactly the six missing samples carry the missing bit; nothing else is
    # flagged, in particular no synthetic "questionable" tail.
    missing = torch.isnan(data)
    assert flags.has_flag(torch.where(missing)[0], 1).all()
    assert torch.all(flags.flags[~missing] == 0)
    assert not flags.has_flag(torch.arange(10), 4).any()
    assert metrics.quality_level.value == "poor"  # 4 of 10 samples unflagged


def test_insufficient_series_does_not_flag_samples_of_a_complete_series():
    """Coverage is a per-series verdict; it never marks unrelated samples."""
    complete = torch.linspace(280.0, 281.0, 10, dtype=torch.float64)
    sparse = complete.clone()
    sparse[2:9] = float("nan")
    data = torch.stack([complete, sparse])
    controller = QualityController()

    with pytest.warns(UserWarning, match="insufficient coverage"):
        _processed, flags, metrics = controller.assess_quality(
            data,
            _time_coordinate(10),
        )

    assert metrics.insufficient_coverage_count == 1
    # Flags describe the representative first series, which is complete.
    assert torch.all(flags.flags == 0)
    with pytest.warns(UserWarning, match="insufficient coverage"):
        assert controller.validate_for_operation(
            data,
            _time_coordinate(10),
            "aggregation",
        ) is False


@pytest.mark.parametrize(
    ("n_missing", "expected_grade", "expected_valid"),
    [
        (3, "acceptable", False),  # coverage 0.7 < threshold 0.8
        (1, "good", True),         # coverage 0.9 >= threshold 0.8
    ],
)
def test_validation_requires_the_coverage_threshold_and_the_grade(
    n_missing,
    expected_grade,
    expected_valid,
):
    data = torch.linspace(280.0, 281.0, 10, dtype=torch.float64)
    data[[2, 5, 8][:n_missing]] = float("nan")
    controller = QualityController()

    _processed, _flags, metrics = controller.assess_quality(data, _time_coordinate(10))

    assert metrics.quality_level.value == expected_grade
    assert controller.validate_for_operation(
        data,
        _time_coordinate(10),
        "aggregation",
    ) is expected_valid


def test_validation_rejects_an_unknown_operation():
    data = torch.linspace(280.0, 281.0, 10, dtype=torch.float64)

    with pytest.raises(ValueError, match="operation"):
        QualityController().validate_for_operation(
            data,
            _time_coordinate(10),
            "regridding",
        )


def test_constant_series_has_no_outliers_at_rounding_level():
    data = torch.full((12,), 0.1, dtype=torch.float32)
    data[-1] = torch.nextafter(data[-1], torch.tensor(1.0, dtype=torch.float32))
    detector = OutlierDetector(QualityConfig.from_parameter_manager())

    assert not detector.detect_outliers(data, _time_coordinate(12)).any()
