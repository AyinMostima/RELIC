"""Isotope parameter audit: the audit CLI exit codes and the auditor rule handling.

The CLI commands are a boundary: a failed audit rule makes the command fail (non-zero exit
code), and only data or file errors are turned into an exit code; a defect propagates. The
auditor records a rule that fails on the audited data as an ERROR result and lets a defect
in a rule propagate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import relic.scientific.isotopes.audit_cli as audit_cli
import relic.scientific.isotopes.parameter_auditor as parameter_auditor_module
from relic.scientific.isotopes.parameter_auditor import AuditResult, IsotopeParameterAuditor


def _args(tmp_path, **extra):
    return SimpleNamespace(
        output_dir=str(tmp_path), format="md", quality_format="md", verbose=False, **extra
    )


def _result(status: str, key: str = "isotopes.temperature.reference") -> AuditResult:
    return AuditResult(parameter_key=key, rule_name="rule", status=status, message=status)


class TestAuditCommandExitCodes:
    """A failed audit rule is a failed command."""

    def test_audit_command_fails_when_a_rule_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            audit_cli, "run_comprehensive_audit",
            lambda: {"isotopes.temperature.reference": [_result("ERROR")]},
        )
        assert audit_cli.run_audit_command(_args(tmp_path)) != 0

    def test_full_command_fails_when_a_rule_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            audit_cli, "run_comprehensive_audit",
            lambda: {"isotopes.temperature.reference": [_result("ERROR")]},
        )
        assert audit_cli.run_full_command(_args(tmp_path)) != 0

    def test_check_command_fails_when_a_rule_fails(self, tmp_path, monkeypatch):
        class Auditor:
            # The check reads the declared metadata of the entry (MA-21 15.1).
            parameter_metadata = staticmethod(IsotopeParameterAuditor.parameter_metadata)

            def audit_parameter(self, key, metadata):
                return [_result("ERROR", key)]

        monkeypatch.setattr(audit_cli, "create_parameter_auditor", lambda: Auditor())
        args = _args(tmp_path, parameter_key="isotopes.temperature.reference")
        assert audit_cli.run_check_command(args) != 0

    def test_a_defect_is_not_turned_into_an_exit_code(self, tmp_path, monkeypatch):
        def defect(*args, **kwargs):
            raise AttributeError("defect in the audit")

        monkeypatch.setattr(audit_cli, "run_comprehensive_audit", defect)
        with pytest.raises(AttributeError, match="defect"):
            audit_cli.run_audit_command(_args(tmp_path))
        monkeypatch.setattr(audit_cli, "create_quality_assessor", defect)
        with pytest.raises(AttributeError, match="defect"):
            audit_cli.run_quality_command(_args(tmp_path))


class TestAuditorRuleHandling:
    """Rules that fail on the audited data are ERROR results; defects propagate."""

    def test_rule_defect_propagates_and_data_failure_is_an_error(self):
        auditor = IsotopeParameterAuditor()

        def defect(key, metadata):
            raise AttributeError("defect in a rule")

        auditor.audit_rules = {"broken": defect}
        with pytest.raises(AttributeError, match="defect"):
            auditor.audit_parameter("isotopes.temperature.reference", {"value": 1.0})

        def bad_value(key, metadata):
            raise ValueError("value is not a number")

        auditor.audit_rules = {"bad": bad_value}
        [result] = auditor.audit_parameter("isotopes.temperature.reference", {"value": "a"})
        assert result.status == "ERROR"

    def test_dependency_check_does_not_report_a_defect_as_a_missing_dependency(
        self, monkeypatch
    ):
        auditor = IsotopeParameterAuditor()

        def defect(key, *args, **kwargs):
            raise TypeError("defect in the constant lookup")

        monkeypatch.setattr(parameter_auditor_module, "get_physical_constant", defect)
        with pytest.raises(TypeError, match="defect"):
            auditor._check_dependency_completeness(
                "isotopes.temperature.reference",
                {"dependencies": ["relic.physical.constants.universal.gas_constant"]},
            )

    def test_report_status_icon_follows_the_worst_result(self):
        auditor = IsotopeParameterAuditor()
        report = auditor.generate_audit_report({
            "isotopes.a": [_result("ERROR", "isotopes.a"), _result("WARNING", "isotopes.a")],
            "isotopes.b": [_result("WARNING", "isotopes.b"), _result("PASS", "isotopes.b")],
            "isotopes.c": [_result("PASS", "isotopes.c")],
        })
        assert "### [ERROR] isotopes.a" in report
        assert "### [WARN] isotopes.b" in report
        assert "### [OK] isotopes.c" in report


class TestAuditorDeclaredBoundsOnly:
    """MA-21 14.1: no constraint is guessed from a key, a dimension or a magnitude; the
    bounds declared in the YAML entry are the only physical constraint."""

    def test_no_dimension_keyed_constraint_validator_remains(self):
        auditor = IsotopeParameterAuditor()
        assert not hasattr(auditor, "constraint_validators")
        assert "physical_constraints" not in auditor.audit_rules
        for name in (
            "_validate_temperature_constraint",
            "_validate_concentration_constraint",
            "_validate_ratio_constraint",
            "_validate_flux_constraint",
            "_validate_dimensionless_constraint",
            "_check_physical_constraints",
        ):
            assert not hasattr(IsotopeParameterAuditor, name), name

    def test_a_signed_temperature_difference_within_its_bounds_passes(self):
        """isotopes.validation.temperature_diff_lower is a temperature difference of
        -50 K with declared bounds [-100, -10] K; a rule guessing 'temperature > 0 K'
        from its dimension would reject it."""
        auditor = IsotopeParameterAuditor()
        results = auditor.audit_parameter(
            "isotopes.validation.temperature_diff_lower",
            {
                "value": -50.0,
                "units": "K",
                "dtype": "float32",
                "description": "Lower limit of the temperature difference from the reference",
                "dimension": "temperature",
                "bounds": (-100.0, -10.0),
                "references": ["doi:10.1016/0016-7037(94)90016-7"],
            },
        )
        assert not [result for result in results if result.status == "ERROR"]

    def test_a_value_outside_its_declared_bounds_is_an_error(self):
        auditor = IsotopeParameterAuditor()
        for value in (5.0, float("nan")):
            result = auditor._check_bounds_consistency(
                "isotopes.pool.max_ratio", {"value": value, "bounds": (0.0, 1.0)}
            )
            assert result.status == "ERROR", value

    def test_bounds_rule_follows_the_canonical_bounds_declaration_rule(self):
        """The bounds declaration is judged by relic.parameters.spec.parameter_bounds_error
        (None only for a non-numeric dtype; a degenerate [c, c] interval fixes a
        constant); only a numeric value is compared with the interval. Every entry of
        the isotope database passes."""
        import logging

        auditor = IsotopeParameterAuditor()
        rule = auditor._check_bounds_consistency
        assert rule("k", {"value": "C3", "dtype": "string", "bounds": None}).status == "PASS"
        assert rule("k", {"value": "C3", "dtype": "string", "bounds": (-1e10, 1e10)}).status == "PASS"
        assert rule("k", {"value": 0.0, "dtype": "float32", "bounds": (0.0, 0.0)}).status == "PASS"
        assert rule("k", {"value": 1.0, "dtype": "float32", "bounds": None}).status == "ERROR"
        assert rule("k", {"value": 2.0, "dtype": "float32", "bounds": (1.0, 0.0)}).status == "ERROR"
        logging.disable(logging.CRITICAL)
        try:
            results = auditor.audit_all_parameters()
        finally:
            logging.disable(logging.NOTSET)
        failed = sorted(
            key for key, rows in results.items()
            for row in rows if row.rule_name == "bounds_consistency" and row.status == "ERROR"
        )
        assert failed == []

    def test_non_numeric_isotope_entries_declare_no_interval(self):
        """MA-21 16.1: the bounds key is present in every isotope entry; a text, list,
        mapping or boolean parameter declares bounds: null and a numeric one an interval
        (the canonical rule of relic.parameters.spec)."""
        import yaml
        from pathlib import Path

        from relic.parameters.spec import _NON_NUMERIC_DTYPES, _dtype_root

        root = Path(__file__).resolve().parents[4]
        database = root / "relic" / "parameters" / "database" / "relic" / "isotopes"
        wrong = []
        for path in sorted(database.rglob("*.yaml")):
            entries = yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"]
            for key, entry in entries.items():
                numeric = _dtype_root(entry.get("dtype")) not in _NON_NUMERIC_DTYPES
                if "bounds" not in entry or (entry["bounds"] is None) == numeric:
                    wrong.append((key, entry.get("dtype"), entry.get("bounds")))
        assert wrong == []

    def test_the_isotope_temperature_domain_minimum_has_physical_bounds(self):
        """isotopes.temperature.min is an absolute temperature: its declared bounds keep
        it above absolute zero and at or below the lowest admissible
        isotopes.temperature.max, so that min <= max."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        minimum = manager.get_parameter("isotopes.temperature.min")
        maximum = manager.get_parameter("isotopes.temperature.max")
        assert minimum.bounds[0] > 0.0
        assert minimum.bounds[1] <= maximum.bounds[0]
        assert "Vostok" in minimum.description


class TestAuditorUnitsAndRepresentation:
    """MA-21 15.1 and 15.2: units are checked by the SI dimensionality owner
    (relic.data.standards.dimensionality) against the declared dimension, and the
    numerical check follows torch.finfo of the declared dtype."""

    @staticmethod
    def _rule(auditor, rule, **metadata):
        return auditor.audit_rules[rule]("isotopes.probe", metadata)

    def test_units_are_checked_by_the_dimensionality_owner(self):
        auditor = IsotopeParameterAuditor()
        assert not hasattr(auditor, "valid_units")
        assert not hasattr(IsotopeParameterAuditor, "_build_valid_units_registry")
        ok = self._rule(
            auditor, "units_validity", units="umol photons m-2 s-1", dimension="photon_flux_density"
        )
        assert ok.status == "PASS"
        mismatch = self._rule(auditor, "units_validity", units="K", dimension="time")
        assert mismatch.status == "ERROR"
        unknown = self._rule(auditor, "units_validity", units="furlongs", dimension="length")
        assert unknown.status == "ERROR"
        undeclared = self._rule(auditor, "units_validity", units="K")
        assert undeclared.status == "WARNING"

    def test_audited_metadata_carries_dimension_and_dtype_of_every_isotope_entry(self):
        import yaml
        from pathlib import Path

        auditor = IsotopeParameterAuditor()
        metadata = auditor.parameter_metadata("isotopes.temperature.min")
        assert metadata["dimension"] == "temperature"
        assert metadata["dtype"] == "float32"
        root = Path(__file__).resolve().parents[4]
        database = root / "relic" / "parameters" / "database" / "relic" / "isotopes"
        entries = set()
        for path in database.rglob("*.yaml"):
            entries.update(yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"])
        assert set(auditor._get_isotope_parameters_from_manager()) == entries

    def test_numerical_check_follows_the_declared_dtype(self):
        auditor = IsotopeParameterAuditor()

        def status(value, dtype):
            return self._rule(auditor, "numerical_stability", value=value, dtype=dtype).status

        # float32: tiny = 1.18e-38 (smallest normal), max = 3.40e38.
        assert status(1.0e-20, "float32") == "PASS"
        assert status(1.0e20, "float32") == "PASS"
        assert status(1.0e-40, "float32") == "WARNING"  # subnormal
        assert status(1.0e-40, "float64") == "PASS"
        assert status(1.0e39, "float32") == "ERROR"  # overflows to inf
        assert status(1.0e39, "float64") == "PASS"
        assert status(0.0, "float32") == "PASS"
        assert status("C3", "string") == "PASS"  # no floating-point representation


class TestQualityAssessorFieldScores:
    """MA-21 16.2: the quality assessor scores the unit through the SI dimensionality
    owner, the value through the representation limits of its declared dtype and the
    bounds through the canonical declaration rule, with the checks of the auditor."""

    @staticmethod
    def _assessor():
        from relic.scientific.isotopes.parameter_quality_assessor import (
            IsotopeParameterQualityAssessor,
        )

        return IsotopeParameterQualityAssessor()

    def _score(self, field, **metadata):
        return self._assessor()._assess_field_quality(field, metadata)

    def test_unit_score_follows_the_dimensionality_owner(self):
        assert self._score("units", units="umol photons m-2 s-1", dimension="photon_flux_density") == 1.0
        assert self._score("units", units="K", dimension="time") == 0.0
        assert self._score("units", units="furlongs", dimension="length") == 0.0

    def test_value_score_follows_the_declared_dtype(self):
        assert self._score("value", value=1.0e-25, dtype="float32") == 1.0
        assert self._score("value", value=1.0e-40, dtype="float32") == 0.5  # subnormal
        assert self._score("value", value=1.0e39, dtype="float32") == 0.0  # overflow
        assert self._score("value", value=1.0e39, dtype="float64") == 1.0
        assert self._score("value", value="C3", dtype="string") == 1.0

    def test_bounds_score_follows_the_canonical_declaration_rule(self):
        assert self._score("bounds", bounds=None, dtype="string") == 1.0
        assert self._score("bounds", bounds=(0.0, 0.0), dtype="float32") == 1.0
        assert self._score("bounds", bounds=None, dtype="float32") == 0.0
        assert self._score("bounds", bounds=(2.0, 1.0), dtype="float32") == 0.0
        text = {
            "value": "C3", "units": "dimensionless", "dtype": "string",
            "description": "Photosynthetic pathway of the configured parameter set.",
        }
        assessor = self._assessor()
        assert assessor._assess_completeness({**text, "bounds": None}) > (
            assessor._assess_completeness(text)
        )
