"""Tests for scope-honest external I/O metadata reporting."""

from types import SimpleNamespace

import pytest
import xarray as xr

from relic.coupling.external.common.io import CF_DECLARED_SCOPE, DataFormat, IOHandler, IOResult


def test_io_result_never_defaults_to_dataset_cf_success():
    result = IOResult(
        data=xr.Dataset(),
        format_detected=DataFormat.NETCDF4,
        file_size=0,
        load_time=0.0,
    )
    assert result.cf_compliant is False
    assert result.dataset_conformance_checked is False
    assert result.cf_validation_scope == "not_evaluated"
    assert result.cf_conformance_issues == ()

    # Conformance can only be claimed by the declared-scope check itself.
    with pytest.raises(ValueError, match="dataset-level CF conformance"):
        IOResult(
            data=xr.Dataset(),
            format_detected=DataFormat.NETCDF4,
            file_size=0,
            load_time=0.0,
            cf_compliant=True,
        )


def test_io_quality_metrics_run_the_declared_scope_check_and_report_its_issues(tmp_path):
    handler = object.__new__(IOHandler)
    handler.config = SimpleNamespace(strict_cf=False, validate_checksums=False)
    handler.missing_data_warning_threshold = 1.0
    handler._verify_checksums = lambda _dataset, _path: True
    dataset = xr.Dataset(
        {"signal": ("time", [1.0])},
        coords={"time": [0]},
        attrs={"Conventions": "CF-1.13"},
    )

    metrics = handler._compute_quality_metrics(dataset, tmp_path / "source.nc")

    assert metrics["cf_conventions_declared"] is True
    assert metrics["dataset_conformance_checked"] is True
    assert metrics["cf_validation_scope"] == CF_DECLARED_SCOPE
    # The fixture declares CF but its variables carry no units: the check
    # ran and reports the defect instead of defaulting to success.
    assert metrics["cf_compliant"] is False
    assert any("has no units attribute" in issue for issue in metrics["cf_conformance_issues"])
    assert any(warning.startswith("CF conformance:") for warning in metrics["warnings"])
