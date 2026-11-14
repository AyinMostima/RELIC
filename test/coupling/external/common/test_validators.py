import inspect

import numpy as np
import pytest
import xarray as xr

import relic.coupling.external.common.validators as validators_module
from relic.coupling.external.atmospheric_models.cesm_adapter import CESMAdapter
from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter
from relic.coupling.external.climate_models.cmip6_adapter import CMIP6Adapter
from relic.coupling.external.common.base_adapter import ExternalAdapter, ValidationReport
from relic.coupling.external.common.datasource import DataSource
from relic.coupling.external.common.validators import ValidationConfig, Validator
from relic.coupling.external.hydrological_models.swat_adapter import SWATAdapter, SWATConfig
from relic.coupling.external.hydrological_models.vic_adapter import VICAdapter
from relic.utils.time_constants import SECONDS_PER_HOUR


def test_validator_returns_report_instance():
    """Ensure Validator.validate returns a ValidationReport without errors."""
    dataset = xr.Dataset(
        {"air_temperature": (("time",), [280.0])},
        coords={"time": [0]},
        attrs={"Conventions": "CF-1.8"},
    )

    validator = Validator(ValidationConfig())
    report = validator.validate(dataset, source=DataSource.UNKNOWN)

    assert isinstance(report, ValidationReport)
    assert isinstance(report.errors, list)
    assert isinstance(report.warnings, list)
    assert report.cf_compliant is False
    assert report.dataset_conformance_checked is False
    assert report.cf_validation_scope == "external_adapter_checks_not_dataset_cf"


def test_external_validation_report_rejects_positive_dataset_cf_claim():
    with pytest.raises(ValueError, match="dataset-level CF conformance"):
        ValidationReport(
            source_path="dummy.nc",
            validation_time="now",
            cf_compliant=True,
        )


def test_temporal_gap_hours_use_seconds_per_hour_ssot(monkeypatch):
    config = ValidationConfig(max_time_gap_hours=30.0)
    validator = Validator(config)
    original_get_constant = validators_module.get_physical_constant

    def get_constant(name):
        if name == "seconds_per_hour":
            return float(SECONDS_PER_HOUR) / 2.0
        return original_get_constant(name)

    monkeypatch.setattr(validators_module, "get_physical_constant", get_constant)
    dataset = xr.Dataset(
        coords={
            "time": np.array(
                ["2001-01-01T00", "2001-01-02T00"],
                dtype="datetime64[h]",
            )
        }
    )

    issues = validator._validate_temporal_quality(
        dataset,
        DataSource.UNKNOWN,
        None,
    )

    assert len(issues) == 1
    assert "48.0 hours" in issues[0].message


def test_temporal_gap_conversion_has_no_local_hour_literal():
    source = inspect.getsource(Validator._validate_temporal_quality)

    assert "/ 3600" not in source
    assert "get_physical_constant" in source


@pytest.mark.parametrize(
    "adapter_cls",
    [CESMAdapter, WRFAdapter, CMIP6Adapter, SWATAdapter, VICAdapter],
    ids=["cesm", "wrf", "cmip6", "swat", "vic"],
)
def test_external_adapters_validate_only_through_validate_data(adapter_cls):
    """ExternalAdapter.validate_data (the existing Validator) is the one validation entry.

    The adapters' own validate() methods called a Validator.validate_dataset that does not
    exist (CESM, WRF, VIC, SWAT) or repeated a subset of the Validator checks (CMIP6), and no
    code called them; they and their unused Validator instances are gone.
    """
    assert not hasattr(adapter_cls, "validate")
    assert adapter_cls.validate_data is ExternalAdapter.validate_data
    assert "self.validator" not in inspect.getsource(adapter_cls.__init__)


def test_validate_data_never_claims_dataset_cf_conformance(tmp_path):
    adapter = SWATAdapter(SWATConfig(root_path=tmp_path, output_level="hru"))
    dataset = xr.Dataset(
        {"PRECIP": (("time", "hru"), np.array([[1.0, 2.0]]), {"units": "mm"})},
        coords={"time": [0.0], "hru": [1, 2]},
    )
    report = adapter.validate_data(dataset)
    assert isinstance(report, ValidationReport)
    assert report.cf_compliant is False
    assert report.dataset_conformance_checked is False


def test_validator_keeps_no_unsourced_reference_statistics():
    """The per-variable reference statistics (mean and standard deviation of air temperature,
    precipitation, shortwave radiation and pressure) had no literature source and no reader: the
    outlier check derives its statistics from the data. They are removed, not moved to YAML."""
    validator = Validator()
    assert not hasattr(validator, "_variable_statistics")
    assert not hasattr(Validator, "_load_reference_statistics")
    source = inspect.getsource(validators_module)
    for stale in ("'mean': 285.0", "'std': 150.0", "stats_db", "reference database"):
        assert stale not in source, stale


def test_validator_thresholds_are_declared_parameters():
    """Every threshold of the external data validator is a declared parameter with references:
    unset ValidationConfig fields take the values of their keys, explicit values are kept and
    the caller's configuration is not modified. The missing-data thresholds are the external IO
    warning threshold and the adapter quality-control limit, not second copies. The unread fields
    (level, allow_missing_coordinates, min/max_spatial_resolution) and their level enum are gone."""
    import dataclasses

    from relic.parameters import get_global_manager

    manager = get_global_manager()
    fields = {item.name: item for item in dataclasses.fields(ValidationConfig)}
    for stale in ("level", "allow_missing_coordinates", "min_spatial_resolution", "max_spatial_resolution"):
        assert stale not in fields, stale
    assert not hasattr(validators_module, "ExternalValidationLevel")
    assert not hasattr(validators_module, "ValidationLevel")
    keys = validators_module.VALIDATION_PARAMETER_KEYS
    assert keys["max_missing_fraction"] == "coupling.external.io.missing_data_warning_threshold"
    assert keys["missing_fraction_error"] == "coupling.external.regrid.missing_threshold"
    assert set(keys) == {
        "max_missing_fraction",
        "missing_fraction_error",
        "outlier_threshold",
        "outlier_fraction_warning",
        "outlier_fraction_error",
        "require_monotonic_time",
        "max_time_gap_hours",
        "require_global_coverage",
        "global_longitude_span_min",
        "global_latitude_span_min",
        "require_cf_conventions",
        "cf_version_min",
        "require_standard_names",
    }
    resolved = Validator().config
    for name, key in keys.items():
        assert fields[name].default is None, name
        assert getattr(resolved, name) == manager.get(key, required=True), name
        assert manager.get_parameter(key).references, key
    explicit = ValidationConfig(max_time_gap_hours=30.0)
    assert Validator(explicit).config.max_time_gap_hours == 30.0
    assert explicit.outlier_threshold is None


def test_validator_checks_keep_no_threshold_literals():
    """The completeness, outlier and coverage checks read their thresholds from the configuration;
    no fraction, sample-size or span literal remains in them."""
    source = "".join(
        inspect.getsource(method)
        for method in (
            Validator._validate_data_completeness,
            Validator._validate_statistical_outliers,
            Validator._validate_spatial_quality,
        )
    )
    for literal in ("0.05", "0.1", "< 10", "0.5", "350", "170", "5%"):
        assert literal not in source, literal


def test_outlier_check_needs_no_minimum_sample_parameter():
    """With the population standard deviation the largest attainable |z| of n finite values is
    sqrt(n - 1) (Shiffler 1988), so a small sample simply cannot exceed the threshold and no
    minimum sample size is assumed: 8 values evaluated at threshold 2 flag one outlier (12.5 %,
    above the error fraction), 4 values cannot (sqrt(3) is below 2)."""
    from relic.coupling.external.common.validators import ValidationCategory, ValidationSeverity

    def outlier_issues(n):
        values = np.zeros(n)
        values[-1] = 1.0
        dataset = xr.Dataset({"x": (("cell",), values)})
        validator = Validator(ValidationConfig(outlier_threshold=2.0))
        return validator._validate_statistical_outliers(dataset, DataSource.UNKNOWN, None)

    issues = outlier_issues(8)
    assert len(issues) == 1
    assert issues[0].category is ValidationCategory.STATISTICAL_OUTLIERS
    assert issues[0].severity is ValidationSeverity.ERROR
    assert outlier_issues(4) == []


def test_missing_data_severity_follows_the_declared_thresholds():
    """A missing fraction above the external IO warning threshold is a WARNING and above the
    adapter quality-control limit (coupling.external.regrid.missing_threshold) an ERROR."""
    from relic.coupling.external.common.validators import ValidationSeverity
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    warning = manager.get("coupling.external.io.missing_data_warning_threshold", required=True)
    error = manager.get("coupling.external.regrid.missing_threshold", required=True)

    def severity(missing):
        values = np.ones(100)
        values[:missing] = np.nan
        dataset = xr.Dataset({"x": (("cell",), values)})
        issues = Validator()._validate_data_completeness(dataset, DataSource.UNKNOWN, None)
        return [issue.severity for issue in issues]

    assert severity(int(warning * 100)) == []
    assert severity(int(warning * 100) + 1) == [ValidationSeverity.WARNING]
    assert severity(int(error * 100) + 1) == [ValidationSeverity.ERROR]
