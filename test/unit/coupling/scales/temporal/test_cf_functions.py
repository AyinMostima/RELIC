"""Tests for relic.coupling.scales.temporal.cf_functions (temporal CF metadata)."""

import inspect

import pytest
import torch

import relic.coupling.scales.temporal.cf_functions as cf_functions
from relic.coupling.scales.temporal.core import TimeCoordinate
from relic.parameters import get_global_manager

_THRESHOLD_KEY = "coupling.scales.temporal.coverage_threshold"


class _RecordingManager:
    """Serve canonical values, optionally override one key, record lookups."""

    def __init__(self, overrides=None):
        self._canonical = get_global_manager()
        self._overrides = dict(overrides or {})
        self.calls = []

    def get(self, key, **kwargs):
        self.calls.append((key, kwargs))
        if key in self._overrides:
            return self._overrides[key]
        return self._canonical.get(key, **kwargs)


def _time_coordinate() -> TimeCoordinate:
    return TimeCoordinate(
        values=torch.tensor([0.5], dtype=torch.float64),
        units="days since 2000-01-01",
        calendar="noleap",
        bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    )


def test_quality_metadata_reads_the_coverage_parameter_without_default():
    manager = _RecordingManager()

    metadata = cf_functions.create_quality_metadata(
        {"valid_points": 7, "total_points": 10},
        parameter_manager=manager,
    )

    assert metadata == {
        "coverage_threshold": pytest.approx(0.8),
        "coverage_fraction": pytest.approx(0.7),
        "quality_state": "insufficient_coverage",
    }
    assert [key for key, _ in manager.calls] == [_THRESHOLD_KEY]
    assert manager.calls[0][1]["required"] is True
    assert "default" not in manager.calls[0][1]
    assert "coverage_threshold" not in inspect.signature(
        cf_functions.create_quality_metadata
    ).parameters


def test_quality_metadata_follows_the_parameter_value():
    manager = _RecordingManager(overrides={_THRESHOLD_KEY: 0.6})

    metadata = cf_functions.create_quality_metadata(
        {"coverage_fraction": 0.7},
        parameter_manager=manager,
    )

    assert metadata["quality_state"] == "good_data"
    assert metadata["coverage_threshold"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "coverage_stats",
    [{}, {"valid_points": 3}, {"total_points": 10}],
)
def test_quality_metadata_requires_explicit_coverage(coverage_stats):
    with pytest.raises(ValueError, match="coverage"):
        cf_functions.create_quality_metadata(coverage_stats)


@pytest.mark.parametrize(
    ("coverage_stats", "error"),
    [
        ({"valid_points": 3, "total_points": 0}, ValueError),
        ({"valid_points": "three", "total_points": 10}, ValueError),
        ({"coverage_fraction": 1.5}, ValueError),
        ({"valid_points": 11, "total_points": 10}, ValueError),
    ],
)
def test_time_coverage_rejects_invalid_counts_instead_of_dropping_them(
    coverage_stats,
    error,
):
    with pytest.raises(error):
        cf_functions.compute_time_coverage(coverage_stats=coverage_stats)


def test_temporal_rescaling_metadata_describes_only_the_time_axis_and_coverage():
    signature = inspect.signature(cf_functions.create_temporal_rescaling_metadata)
    assert signature.parameters["coverage_stats"].default is inspect.Parameter.empty

    metadata = cf_functions.create_temporal_rescaling_metadata(
        _time_coordinate(),
        coverage_stats={"valid_points": 24, "total_points": 24},
    )

    assert set(metadata) == {
        "time_coordinate",
        "valid_points",
        "total_points",
        "coverage_fraction",
        "coverage_threshold",
        "quality_state",
    }
    assert metadata["time_coordinate"]["units"] == "days since 2000-01-01"
    assert metadata["time_coordinate"]["bounds"] == "time_bounds"
    assert metadata["quality_state"] == "good_data"
    assert metadata["coverage_threshold"] == pytest.approx(0.8)


def test_variable_level_temporal_sync_copies_are_absent():
    for removed in (
        "synchronize_cf_temporal_metadata",
        "create_time_bounds_metadata",
        "create_history_entry",
        "update_history",
    ):
        assert not hasattr(cf_functions, removed)


def test_time_attributes_declare_bounds_only_when_the_coordinate_has_them():
    unbounded = TimeCoordinate(
        values=torch.tensor([0.5], dtype=torch.float64),
        units="days since 2000-01-01",
        calendar="noleap",
    )

    assert "bounds" not in cf_functions.create_cf_time_attributes(unbounded)
    assert cf_functions.create_cf_time_attributes(_time_coordinate())["bounds"] == "time_bounds"


def test_temporal_compliance_checks_the_time_coordinate_attributes():
    attributes = cf_functions.create_cf_time_attributes(_time_coordinate())

    assert cf_functions.validate_cf_temporal_compliance(attributes) == []
    assert cf_functions.validate_cf_temporal_compliance(
        dict(attributes, standard_name="precipitation_flux")
    ) == ["Time coordinate standard_name must be 'time'"]
    assert cf_functions.validate_cf_temporal_compliance(
        {key: value for key, value in attributes.items() if key != "calendar"}
    ) == ["Missing or invalid CF time coordinate attribute: calendar"]


def test_module_has_no_numeric_threshold_fallback():
    source = inspect.getsource(cf_functions)
    assert "0.8" not in source
    assert '"coverage_fraction", 1.0' not in source
