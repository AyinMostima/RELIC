"""Tests for RELIC-IPHS-DEC structural parameter audit."""

import pytest
import torch

from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.parameters import ParameterCategory, ParameterSpec
from relic.parameters.audit import StructuralParameterAudit, run_structural_parameter_audit


def _structural_spec(**overrides):
    payload = {
        "key": "hydrosphere.soil.transport.hodge.k",
        "value": 1.0,
        "units": "m s-1",
        "dtype": "float32",
        "bounds": (0.0, float("inf")),
        "category": ParameterCategory.PHYSICAL,
        "description": "Saturated conductivity for Hodge assembly.",
        "references": ["Hirani (2003)", "Darcy (1856)"],
        "tags": {"hodge", "structural", "transport"},
        "dimension": "hydraulic_conductivity",
        "provenance": {"source": "unit_test"},
        "requires_grad": False,
    }
    payload.update(overrides)
    return ParameterSpec(**payload)


def _bad_hodge():
    domain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="primal",
        n_dofs=2,
        units="m",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="head_gradient",
        orientation_convention="outward_positive",
    )
    codomain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="dual",
        n_dofs=2,
        units="kg s-1",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="water_flux",
        orientation_convention="outward_positive",
    )
    return HodgeStarSpec(
        name="bad_hodge",
        domain=domain,
        codomain=codomain,
        matrix=torch.diag(torch.tensor([1.0, -0.1])),
        parameter_keys=("hydrosphere.soil.transport.hodge.k",),
        metric_keys=("hydrosphere.soil.transport.metric.area",),
        material_law="darcy",
    )


def test_structural_parameter_audit_accepts_complete_structural_spec():
    report = run_structural_parameter_audit([_structural_spec()], strict=True)

    assert report.observed_total == 1
    assert report.has_failures() is False


def test_structural_parameter_audit_flags_missing_references_and_provenance():
    bad_payload = _structural_spec().to_dict()
    bad_payload["references"] = []
    bad_payload["provenance"] = {}

    report = run_structural_parameter_audit(
        [bad_payload],
        strict=True,
    )

    assert report.has_failures() is True
    assert {issue.context["field"] for issue in report.issues} >= {"references", "provenance"}

    summary = StructuralParameterAudit.from_report(report)

    assert summary.references_ok is False
    assert summary.provenance_ok is False
    assert summary.errors


@pytest.mark.parametrize("bounds", [None, (0.0, None), [None, 1.0]])
def test_structural_audit_flags_undeclared_numeric_bounds(bounds):
    """A numeric structural parameter declares an interval; -inf/inf marks an open side."""
    payload = _structural_spec().to_dict()
    payload["bounds"] = bounds

    report = run_structural_parameter_audit([payload], strict=True)

    assert "bounds" in {issue.context["field"] for issue in report.issues}
    assert StructuralParameterAudit.from_report(report).bounds_ok is False


def test_structural_audit_accepts_none_bounds_for_a_non_numeric_dtype():
    """None states that no interval applies, which holds for a text selector only."""
    selector = _structural_spec(
        key="hydrosphere.soil.transport.hodge.material_law",
        value="darcy",
        units="1",
        dtype="string",
        bounds=None,
        dimension="dimensionless",
    )
    payload = _structural_spec().to_dict()
    payload.pop("bounds")

    accepted = run_structural_parameter_audit([selector], strict=True)
    missing = run_structural_parameter_audit([payload], strict=True)

    assert "bounds" not in {issue.context["field"] for issue in accepted.issues}
    assert "bounds" in {issue.context["field"] for issue in missing.issues}


def test_structural_parameter_audit_validates_generated_hodge_operator():
    report = run_structural_parameter_audit(
        [_structural_spec()],
        hodge_specs=[_bad_hodge()],
        strict=True,
    )

    assert report.has_failures() is True
    assert any("positive semidefinite" in issue.message for issue in report.issues)

    summary = StructuralParameterAudit.from_report(report)

    assert summary.hodge_psd_ok is False
    assert summary.to_dict()["hodge_psd_ok"] is False


class _StructureStub:
    """Generated-structure stand-in whose validate() raises the given error."""

    def __init__(self, name, error):
        self.name = name
        self._error = error

    def validate(self, **_kwargs):
        raise self._error


def test_structural_audit_records_a_rejected_iphs_structure_as_critical():
    report = run_structural_parameter_audit(
        [_structural_spec()],
        iphs_specs=[_StructureStub("bad_iphs", ValueError("J is not skew-symmetric"))],
    )

    critical = [issue for issue in report.issues if issue.key == "bad_iphs"]
    assert report.has_failures() is True
    assert [issue.severity.value for issue in critical] == ["critical"]
    assert critical[0].message == "J is not skew-symmetric"


@pytest.mark.parametrize(
    ("keyword", "error"),
    [
        ("hodge_specs", RuntimeError("internal R initialization failed")),
        ("iphs_specs", AttributeError("stub has no matrix")),
    ],
)
def test_structural_audit_propagates_errors_that_are_not_structural_rejections(keyword, error):
    """Only a structural rejection (ValueError) becomes an audit issue."""
    with pytest.raises(type(error), match=str(error)):
        run_structural_parameter_audit(
            [_structural_spec()],
            **{keyword: [_StructureStub("broken_structure", error)]},
        )
