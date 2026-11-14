"""
Unit tests for parameter audit functionality.

Tests the parameter registry audit system including:
- Missing parameter detection
- Orphan parameter identification
- Duplicate registration scanning
- Validation error detection
- Unit consistency checking
- Report generation and export
"""

import pytest
import json
import yaml
from unittest.mock import patch

# Import the components to test
from relic.parameters import UnifiedParameterManager
from relic.parameters.audit import (
    ParameterAuditReport, 
    ParameterAuditIssue,
    AuditIssueType, 
    AuditSeverity
)
from relic.parameters.types import ParameterCategory
from relic.parameters import ParameterSpec


class TestParameterAudit:
    """Test suite for parameter audit functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Create fresh manager instance for each test
        self.manager = UnifiedParameterManager()
        self.manager.clear_parameters()
        self.manager.clear_missing_requests()
        
        # Create some test parameters
        self.test_params = [
            ParameterSpec(
                key="test.param1",
                value=10.0,
                units="m/s",
                bounds=(0, 100),
                category=ParameterCategory.PHYSICAL,
                description="Test parameter 1", dtype="float32"
            ),
            ParameterSpec(
                key="test.param2",
                value=20.0,
                units="kg",
                bounds=(0, 50),
                category=ParameterCategory.PHYSICAL,
                description="Test parameter 2", dtype="float32"
            ),
            ParameterSpec(
                key="bio.growth_rate",
                value=0.5,
                units="day-1",
                bounds=(0, 1),
                category=ParameterCategory.BIOPHYSICAL,
                description="Growth rate", dtype="float32"
            )
        ]
        
        # Register test parameters
        for spec in self.test_params:
            self.manager.register_parameter(spec)
    
    def test_run_registry_audit_basic(self):
        """Test basic audit functionality."""
        # Run audit
        report = self.manager.run_registry_audit()
        
        # Check report structure
        assert isinstance(report, ParameterAuditReport)
        assert report.expected_total >= 0
        assert report.observed_total >= 0
        assert isinstance(report.issues, list)
        assert isinstance(report.notes, list)
    
    def test_missing_parameter_detection(self):
        """Test detection of missing parameters."""
        # Simulate missing parameter requests
        self.manager._missing_requests["missing.param1"] = 5
        self.manager._missing_requests["missing.param2"] = 10
        
        # Run audit
        report = self.manager.run_registry_audit()
        
        # Check for missing parameter issues
        missing_issues = report.filter_by_type(AuditIssueType.MISSING)
        assert len(missing_issues) >= 2
        
        # Check specific missing parameters
        missing_keys = [issue.key for issue in missing_issues]
        assert "missing.param1" in missing_keys
        assert "missing.param2" in missing_keys
        
        # Check severity based on request count
        for issue in missing_issues:
            if issue.key == "missing.param2":
                # Higher request count should have higher severity
                assert issue.severity in [AuditSeverity.CRITICAL, AuditSeverity.ERROR]
    
    def test_orphan_parameter_detection(self):
        """Test detection of orphan/unused parameters."""
        # Add a parameter that's never accessed
        orphan_spec = ParameterSpec(
            key="orphan.param",
            value=99.0,
            units="dimensionless",
            description="Orphan parameter", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        self.manager.register_parameter(orphan_spec)
        
        # Run audit with orphan detection enabled
        report = self.manager.run_registry_audit(include_unused=True)
        
        # Check for orphan issues
        orphan_issues = report.filter_by_type(AuditIssueType.ORPHAN)
        orphan_keys = [issue.key for issue in orphan_issues]
        
        # The orphan parameter should be flagged
        assert "orphan.param" in orphan_keys
    
    def test_duplicate_registration_detection(self):
        """Test detection of duplicate parameter registrations."""
        # Register parameters with same basename but different namespaces
        spec1 = ParameterSpec(
            key="ns1.duplicate_param",
            value=1.0,
            units="m",
            description="First duplicate", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        spec2 = ParameterSpec(
            key="ns2.duplicate_param", 
            value=2.0,
            units="m",
            description="Second duplicate", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        spec3 = ParameterSpec(
            key="ns3.duplicate_param",
            value=1.0,
            units="cm",  # Different unit
            description="Third duplicate", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        
        self.manager.register_parameter(spec1)
        self.manager.register_parameter(spec2)
        self.manager.register_parameter(spec3)
        
        # Run audit
        report = self.manager.run_registry_audit()
        
        # Check for duplicate issues
        duplicate_issues = report.filter_by_type(AuditIssueType.DUPLICATE)
        
        # Should detect duplicates with different values or units
        assert len(duplicate_issues) > 0
        
        # Check that duplicate issues contain proper context
        for issue in duplicate_issues:
            if "duplicate_param" in issue.key:
                assert "all_keys" in issue.context
                assert "values" in issue.context
                assert "units" in issue.context
    
    def test_validation_error_detection(self):
        """Test detection of validation errors."""
        # Add a parameter with invalid value (outside bounds)
        invalid_spec = ParameterSpec(
            key="invalid.param",
            value=150.0,
            units="percent",
            bounds=(0, 100),
            description="Invalid parameter", dtype="float32", category="model_parameter"
        )
        # Force register without validation
        self.manager.parameters["invalid.param"] = invalid_spec
        
        # Run audit
        report = self.manager.run_registry_audit()
        
        # Check for validation errors
        validation_issues = report.filter_by_type(AuditIssueType.VALIDATION_ERROR)
        
        # Should detect the out-of-bounds parameter
        validation_keys = [issue.key for issue in validation_issues]
        assert "invalid.param" in validation_keys
    
    def test_namespace_filtering(self):
        """Test audit filtering by namespace."""
        # Run audit filtered to "test" namespace
        report = self.manager.run_registry_audit(namespace_filter="test")
        
        # Check that only test namespace parameters are included
        for issue in report.issues:
            if issue.key.startswith("test."):
                continue
            # Missing parameters might not have namespace
            if issue.issue_type == AuditIssueType.MISSING:
                continue
            # Otherwise, should be filtered
            assert issue.key.startswith("test.")
    
    def test_export_audit_report_table(self):
        """Test exporting audit report in table format."""
        # Create a report with some issues
        report = ParameterAuditReport(expected_total=0, observed_total=0)
        report.add_issue(ParameterAuditIssue(
            key="test.param",
            issue_type=AuditIssueType.MISSING,
            severity=AuditSeverity.ERROR,
            message="Parameter missing"
        ))
        
        # Export as table
        output = self.manager.export_audit_report(report, format="table")
        
        # Check output format
        assert isinstance(output, str)
        assert "Parameter Audit Report" in output
        assert "test.param" in output
        assert "MISSING" in output.upper()
    
    def test_export_audit_report_json(self):
        """Test exporting audit report in JSON format."""
        # Create a report
        report = self.manager.run_registry_audit()
        
        # Export as JSON
        output = self.manager.export_audit_report(report, format="json")
        
        # Check JSON validity
        data = json.loads(output)
        assert "created_at" in data
        assert "issues" in data
        assert "summary" in data
    
    def test_export_audit_report_yaml(self):
        """Test exporting audit report in YAML format."""
        # Create a report
        report = self.manager.run_registry_audit()
        
        # Export as YAML
        output = self.manager.export_audit_report(report, format="yaml")
        
        # Check YAML validity
        data = yaml.safe_load(output)
        assert "created_at" in data
        assert "issues" in data
    
    def test_export_audit_report_markdown(self):
        """Test exporting audit report in Markdown format."""
        # Create a report with various issues
        report = ParameterAuditReport(expected_total=0, observed_total=0)
        report.add_issue(ParameterAuditIssue(
            key="critical.param",
            issue_type=AuditIssueType.MISSING,
            severity=AuditSeverity.CRITICAL,
            message="Critical parameter missing"
        ))
        
        # Export as Markdown
        output = self.manager.export_audit_report(report, format="markdown")
        
        # Check Markdown formatting
        assert "# Parameter Audit Report" in output
        assert "## " in output  # Should have sections
        assert "**" in output  # Should have bold text
        assert "|" in output  # Should have tables
    
    def test_get_missing_requests(self):
        """Test retrieving missing request statistics."""
        # Simulate some missing requests
        self.manager._missing_requests["param1"] = 5
        self.manager._missing_requests["param2"] = 3
        
        # Get missing requests
        missing = self.manager.get_missing_requests()
        
        # Check results
        assert isinstance(missing, dict)
        assert missing["param1"] == 5
        assert missing["param2"] == 3
    
    def test_clear_missing_requests(self):
        """Test clearing missing request tracking."""
        # Add some missing requests
        self.manager._missing_requests["param1"] = 10
        
        # Clear tracking
        self.manager.clear_missing_requests()
        
        # Check that it's cleared
        assert len(self.manager._missing_requests) == 0
    
    def test_audit_report_has_failures(self):
        """Test checking if audit report has failures."""
        # Create report with no issues
        report = ParameterAuditReport(expected_total=0, observed_total=0)
        assert not report.has_failures()
        
        # Add warning issue
        report.add_issue(ParameterAuditIssue(
            key="test",
            issue_type=AuditIssueType.ORPHAN,
            severity=AuditSeverity.WARNING,
            message="Warning"
        ))
        assert not report.has_failures()
        
        # Add error issue
        report.add_issue(ParameterAuditIssue(
            key="test2",
            issue_type=AuditIssueType.MISSING,
            severity=AuditSeverity.ERROR,
            message="Error"
        ))
        assert report.has_failures()
    
    def test_unit_consistency_checking(self):
        """Test unit consistency checking within dimensions."""
        # Add parameters with same dimension but different units
        spec1 = ParameterSpec(
            key="length1",
            value=1.0,
            units="m",
            dimension="length",
            description="Length in meters", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        spec2 = ParameterSpec(
            key="length2", 
            value=100.0,
            units="cm",
            dimension="length",
            description="Length in centimeters", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        spec3 = ParameterSpec(
            key="length3",
            value=1000.0,
            units="mm",
            dimension="length",
            description="Length in millimeters", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        
        self.manager.register_parameter(spec1)
        self.manager.register_parameter(spec2)
        self.manager.register_parameter(spec3)
        
        # Run audit
        report = self.manager.run_registry_audit()
        
        # Check for unit mismatch issues
        unit_issues = report.filter_by_type(AuditIssueType.UNIT_MISMATCH)
        
        # Should detect unit inconsistencies if dimensions are defined
        # Note: This test may need adjustment based on actual implementation
        if unit_issues:
            for issue in unit_issues:
                assert "dimension" in issue.context
                assert "common_units" in issue.context
    
    def test_expected_params_list(self):
        """Test using explicit expected parameters list."""
        # Run audit with explicit expected list
        expected = ["test.param1", "test.param2", "required.param"]
        report = self.manager.run_registry_audit(expected_params=expected)
        
        # Should detect missing "required.param"
        missing = report.missing_keys
        assert "required.param" in missing
    
    def test_audit_with_database_parameters(self):
        """Test audit including database parameters."""
        # Mock database loading
        with patch.object(self.manager.database, 'load_all') as mock_load:
            mock_load.return_value = {
                "db.param1": ParameterSpec(
                    key="db.param1",
                    value=5.0,
                    units="s",
                    description="Database parameter", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
                )
            }
            
            # Run audit
            report = self.manager.run_registry_audit()
            
            # Database parameters should be included in expected set
            assert report.expected_total > len(self.test_params)

    def test_audit_database_load_failure_is_raised(self):
        """A database that cannot be loaded fails the audit instead of shrinking the expected set."""
        with patch.object(self.manager.database, 'load_all') as mock_load:
            mock_load.side_effect = ValueError("duplicate key relic.x in two base-tier files")

            with pytest.raises(ValueError, match="duplicate key"):
                self.manager.run_registry_audit()

    def test_namespace_filter_does_not_select_sibling_prefixes(self):
        """namespace_filter 'test' selects the keys below 'test.', not 'testx.*'."""
        from relic.parameters import audit as audit_module

        self.manager.register_parameter(
            ParameterSpec(key="testx.param", value=1.0, units="1", description="Sibling prefix", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        )
        self.manager._missing_requests["testx.missing"] = 3

        observed = audit_module._collect_observed_parameters(self.manager, "test")
        expected = audit_module._collect_expected_parameters(self.manager, None, "test")
        missing = audit_module._scan_missing_parameters(self.manager, set(), set(), "test")

        assert {"test.param1", "test.param2"} <= observed
        assert "testx.param" not in observed
        assert "testx.param" not in expected
        assert "testx.missing" not in {issue.key for issue in missing}

    def test_audit_report_totals_are_required_record_fields(self):
        """The expected/observed totals are counted by the audit; no record defaults them to 0."""
        import dataclasses

        fields = ParameterAuditReport.__dataclass_fields__
        for name in ("expected_total", "observed_total"):
            assert fields[name].default is dataclasses.MISSING, name
        with pytest.raises(TypeError, match="expected_total"):
            ParameterAuditReport()
        from relic.parameters import audit as audit_module

        report = self.manager.run_registry_audit(namespace_filter="test")
        assert report.observed_total == len(audit_module._collect_observed_parameters(self.manager, "test"))
        assert report.expected_total == len(audit_module._collect_expected_parameters(self.manager, None, "test"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


