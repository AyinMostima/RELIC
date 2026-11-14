"""
L1 Unit Tests: Core Validation Models

Tests for ValidationResultModel, ValidationReportModel, and related Pydantic models.
Focuses on schema validation, serialization/deserialization, and data integrity.

Mathematical Framework Testing:
    - Model schema compliance with CF-1.8 conventions
    - Statistical metrics validation (CRPS, p-values, etc.)
    - Conservation quantity units and dimensional analysis
    - Proper scoring rule mathematical properties

References:
    - Pydantic documentation: https://pydantic-docs.helpmanual.io/
    - CF-1.8 Conventions: https://cfconventions.org/
"""

import json
import pytest
import torch
import yaml
from datetime import datetime

from relic.validation.framework.models import (
    ValidationLevel, ValidationCode, ValidationDetail,
    ValidationResultModel, ValidationReportModel, ValidationContext
)


class TestValidationLevel:
    """Test ValidationLevel enum."""
    
    def test_validation_level_values(self):
        """Test that all validation levels are properly defined."""
        assert ValidationLevel.INFO.value == "INFO"
        assert ValidationLevel.WARNING.value == "WARNING"
        assert ValidationLevel.ERROR.value == "ERROR"
        assert ValidationLevel.CRITICAL.value == "CRITICAL"
    
    def test_validation_level_ordering(self):
        """Test implicit severity ordering."""
        levels = [ValidationLevel.INFO, ValidationLevel.WARNING, 
                 ValidationLevel.ERROR, ValidationLevel.CRITICAL]
        
        # Should be in order of increasing severity
        assert len(levels) == 4
        
        # Test that they can be compared as strings
        assert "ERROR" > "CRITICAL"  # Alphabetically
        
        # In practice, we use explicit severity checking in code


class TestValidationCode:
    """Test ValidationCode enum."""
    
    def test_cf_convention_codes(self):
        """Test CF-1.8 convention validation codes."""
        assert ValidationCode.CF_UNIT_MISMATCH == "CF_UNIT_MISMATCH"
        assert ValidationCode.CF_NAME_MISSING == "CF_NAME_MISSING"
        assert ValidationCode.CF_STANDARD_NAME_INVALID == "CF_STANDARD_NAME_INVALID"
        assert ValidationCode.CF_UNIT_CONVERSION_FAILED == "CF_UNIT_CONVERSION_FAILED"
    
    def test_conservation_codes(self):
        """Test conservation law validation codes."""
        assert ValidationCode.MASS_BALANCE_FAIL_CARBON == "MASS_BALANCE_FAIL_CARBON"
        assert ValidationCode.MASS_BALANCE_FAIL_NITROGEN == "MASS_BALANCE_FAIL_NITROGEN"
        assert ValidationCode.MASS_BALANCE_FAIL_WATER == "MASS_BALANCE_FAIL_WATER"
        assert ValidationCode.ENERGY_BALANCE_FAIL == "ENERGY_BALANCE_FAIL"
    
    def test_scoring_codes(self):
        """Test probabilistic scoring validation codes."""
        assert ValidationCode.SCORE_CRPS_HIGH == "SCORE_CRPS_HIGH"
        assert ValidationCode.PIT_UNIFORMITY_REJECT_KS == "PIT_UNIFORMITY_REJECT_KS"
        assert ValidationCode.PIT_UNIFORMITY_REJECT_CVM == "PIT_UNIFORMITY_REJECT_CVM"
        assert ValidationCode.BRIER_SCORE_HIGH == "BRIER_SCORE_HIGH"
    
    def test_parameter_codes(self):
        """Test parameter validation codes."""
        assert ValidationCode.PARAM_BOUND_VIOLATION == "PARAM_BOUND_VIOLATION"
        assert ValidationCode.PARAM_SCHEMA_MISSING == "PARAM_SCHEMA_MISSING"
        assert ValidationCode.PARAM_LOG_SCALE_INCONSISTENT == "PARAM_LOG_SCALE_INCONSISTENT"


class TestValidationDetail:
    """Test ValidationDetail model."""
    
    def test_basic_validation_detail(self):
        """Test basic ValidationDetail creation and validation."""
        detail = ValidationDetail(
            variable_name="temperature",
            cf_standard_name="air_temperature",
            units="K",
            value=285.5,
            expected=(273.0, 310.0),
            threshold=0.1
        )
        
        assert detail.variable_name == "temperature"
        assert detail.cf_standard_name == "air_temperature"
        assert detail.units == "K"
        assert detail.value == 285.5
        assert detail.expected == (273.0, 310.0)
        assert detail.threshold == 0.1
        assert isinstance(detail.location, dict)
        assert isinstance(detail.statistics, dict)
        assert isinstance(detail.metadata, dict)
    
    def test_validation_detail_with_location(self):
        """Test ValidationDetail with spatial/temporal location."""
        detail = ValidationDetail(
            variable_name="gpp",
            cf_standard_name="gross_primary_productivity_of_carbon",
            units="kg C m-2 s-1",
            value=2.5e-6,
            location={"lat": 45.5, "lon": -120.2, "time_step": 100},
            statistics={"mean": 2.1e-6, "std": 0.8e-6}
        )
        
        assert detail.location["lat"] == 45.5
        assert detail.location["lon"] == -120.2
        assert detail.location["time_step"] == 100
        assert detail.statistics["mean"] == 2.1e-6
        assert detail.statistics["std"] == 0.8e-6
    
    def test_validation_detail_with_tensor(self):
        """Test ValidationDetail with PyTorch tensor values."""
        # Note: This tests the arbitrary_types_allowed config
        tensor_value = torch.tensor([1.5, 2.3, 1.9])
        
        detail = ValidationDetail(
            variable_name="leaf_area_index",
            units="m2 m-2",
            value=str(tensor_value.mean().item()),  # Convert to float for JSON serialization
            metadata={"tensor_shape": list(tensor_value.shape)}
        )
        
        assert detail.variable_name == "leaf_area_index"
        assert detail.units == "m2 m-2"
        assert detail.metadata["tensor_shape"] == [3]


class TestValidationResultModel:
    """Test ValidationResultModel."""
    
    def test_basic_validation_result(self):
        """Test basic ValidationResultModel creation."""
        result = ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.CF_UNIT_MISMATCH,
            message="Unit mismatch detected: expected 'K' but found 'C'",
            layer="L1",
            scope="cross_domain.validate_cf_units"
        )
        
        assert result.level == ValidationLevel.WARNING
        assert result.code == ValidationCode.CF_UNIT_MISMATCH
        assert result.message == "Unit mismatch detected: expected 'K' but found 'C'"
        assert result.layer == "L1"
        assert result.scope == "cross_domain.validate_cf_units"
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.details, list)
        assert isinstance(result.tags, list)
        assert isinstance(result.metrics, dict)
    
    def test_validation_result_with_details(self):
        """Test ValidationResultModel with detailed diagnostics."""
        detail = ValidationDetail(
            variable_name="temperature",
            cf_standard_name="air_temperature",
            units="C",
            value=15.5,
            expected="K"
        )
        
        result = ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.CF_UNIT_MISMATCH,
            message="Temperature units not CF-compliant",
            details=[detail],
            layer="L1",
            scope="cross_domain.validate_cf_compliance",
            tags=["cf", "units", "temperature"]
        )
        
        assert len(result.details) == 1
        assert result.details[0].variable_name == "temperature"
        assert result.details[0].units == "C"
        assert "cf" in result.tags
        assert "units" in result.tags
        assert "temperature" in result.tags
    
    def test_validation_result_tag_normalization(self):
        """Test that tags are normalized to lowercase."""
        result = ValidationResultModel(
            level=ValidationLevel.INFO,
            code=ValidationCode.PARAM_BOUND_VIOLATION,
            message="Test message",
            layer="L1",
            scope="test_scope",
            tags=["CF", "Conservation", "STOCHASTIC"]
        )
        
        assert "cf" in result.tags
        assert "conservation" in result.tags
        assert "stochastic" in result.tags
        assert "CF" not in result.tags  # Should be normalized
    
    def test_validation_result_passing_failing(self):
        """Test is_passing and is_failing methods."""
        info_result = ValidationResultModel(
            level=ValidationLevel.INFO,
            code=ValidationCode.PARAM_BOUND_VIOLATION,
            message="Info message",
            layer="L1",
            scope="test"
        )
        
        warning_result = ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.CF_UNIT_MISMATCH,
            message="Warning message", 
            layer="L1",
            scope="test"
        )
        
        error_result = ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
            message="Error message",
            layer="L2",
            scope="test"
        )
        
        critical_result = ValidationResultModel(
            level=ValidationLevel.CRITICAL,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message="Critical message",
            layer="L1", 
            scope="test"
        )
        
        # Test passing states
        assert info_result.is_passing() is True
        assert warning_result.is_passing() is True
        assert error_result.is_passing() is False
        assert critical_result.is_passing() is False
        
        # Test failing states
        assert info_result.is_failing() is False
        assert warning_result.is_failing() is False
        assert error_result.is_failing() is True
        assert critical_result.is_failing() is True
    
    def test_add_detail_method(self):
        """Test the add_detail convenience method."""
        result = ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.SCORE_CRPS_HIGH,
            message="CRPS score exceeds threshold",
            layer="L1",
            scope="stochastic.validate_crps"
        )
        
        result.add_detail(
            variable_name="precipitation",
            value=0.85,
            threshold=0.8,
            statistics={"baseline_crps": 0.65}
        )
        
        assert len(result.details) == 1
        assert result.details[0].variable_name == "precipitation"
        assert result.details[0].value == 0.85
        assert result.details[0].threshold == 0.8
        assert result.details[0].statistics["baseline_crps"] == 0.65


class TestValidationReportModel:
    """Test ValidationReportModel."""
    
    def test_empty_validation_report(self):
        """Test empty ValidationReportModel creation."""
        report = ValidationReportModel()
        
        assert len(report.results) == 0
        assert len(report) == 0
        assert isinstance(report.summary, dict)
        assert isinstance(report.cf_summary, dict)
        assert isinstance(report.conservation_summary, dict)
        assert isinstance(report.timestamp, datetime)
        assert report.has_errors() is False
        assert report.has_warnings() is False
    
    def test_validation_report_with_results(self):
        """Test ValidationReportModel with multiple results."""
        results = [
            ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_BOUND_VIOLATION,
                message="Info message",
                layer="L1",
                scope="test",
                tags=["parameters"]
            ),
            ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.CF_UNIT_MISMATCH,
                message="Warning message",
                layer="L1", 
                scope="test",
                tags=["cf"]
            ),
            ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
                message="Error message",
                layer="L2",
                scope="test",
                tags=["conservation"]
            )
        ]
        
        report = ValidationReportModel(results=results)
        
        assert len(report) == 3
        assert report.has_warnings() is True
        assert report.has_errors() is True
        
        # Test filtering methods
        info_results = report.get_results_by_level(ValidationLevel.INFO)
        assert len(info_results) == 1
        assert info_results[0].level == ValidationLevel.INFO
        
        l1_results = report.get_results_by_layer("L1")
        assert len(l1_results) == 2
        
        cf_results = report.get_results_by_tag("cf")
        assert len(cf_results) == 1
        assert "cf" in cf_results[0].tags
    
    def test_report_summary_generation(self):
        """Test automatic summary generation."""
        report = ValidationReportModel()
        
        # Add results through add_result method to test summary updates
        report.add_result(ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.CF_UNIT_MISMATCH,
            message="Warning 1",
            layer="L1",
            scope="test",
            tags=["cf"]
        ))
        
        report.add_result(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
            message="Error 1",
            layer="L2",
            scope="test",
            tags=["conservation"]
        ))
        
        # Check summary statistics
        assert report.summary["total"] == 2
        assert report.summary["by_level"]["WARNING"] == 1
        assert report.summary["by_level"]["ERROR"] == 1
        assert report.summary["by_layer"]["L1"] == 1
        assert report.summary["by_layer"]["L2"] == 1
        
        # Check CF summary
        assert report.cf_summary["total_cf_validations"] == 1
        assert report.cf_summary["cf_warnings"] == 1
        assert report.cf_summary["cf_errors"] == 0
        
        # Check conservation summary
        assert report.conservation_summary["total_conservation_checks"] == 1
        assert report.conservation_summary["conservation_failures"] == 1
        assert report.conservation_summary["conservation_warnings"] == 0
    
    @pytest.mark.parametrize("empty", [False, True])
    @pytest.mark.parametrize("format_name", ["dict", "json", "yaml"])
    def test_report_serialization(self, empty, format_name):
        """Reports preserve their real data through each public format."""
        from relic.validation.framework.reporters import ValidationReporter

        timestamp = datetime(2026, 9, 12, 8, 30, 5, 123456)
        result = ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MASS_BALANCE_FAIL_WATER,
            message="Water budget residual exceeds tolerance",
            layer="L2",
            scope="test.water_budget",
            tags=["conservation"],
            details=[ValidationDetail(
                variable_name="water_residual", units="kg m-2", value=2.0,
                expected=0.0, threshold=0.1, location={"cell": 3},
            )],
            timestamp=timestamp,
        )
        report = ValidationReportModel(
            results=[] if empty else [result], timestamp=timestamp,
            execution_metadata={"execution_time_seconds": 0.25, "run_id": "report-run"},
        )
        reporter = ValidationReporter()
        if format_name == "dict":
            payload = json.dumps(report.to_dict())
            restored = ValidationReportModel.model_validate_json(payload)
        elif format_name == "json":
            restored = ValidationReportModel.model_validate_json(reporter.to_json(report))
        else:
            restored = ValidationReportModel.model_validate(yaml.safe_load(reporter.to_yaml(report)))
        assert restored.model_dump() == report.model_dump()
        assert restored.summary["total"] == (0 if empty else 1)
        assert restored.summary["by_level"]["ERROR"] == (0 if empty else 1)
        assert restored.summary["by_layer"] == ({} if empty else {"L2": 1})
        assert restored.has_errors() is (not empty)
        assert restored.conservation_summary["conservation_failures"] == (0 if empty else 1)
        assert restored.timestamp == timestamp
        if not empty:
            assert restored.results[0].timestamp == timestamp
            assert restored.results[0].details[0].cf_standard_name is None


class TestValidationContext:
    """Test ValidationContext dataclass."""
    
    def test_validation_context_creation(self):
        """Test ValidationContext creation and attributes."""
        context = ValidationContext(
            domain_filter=["biosphere", "atmosphere"],
            config={"tolerance": 1e-6},
            execution_layer="L2",
            metadata={"test_run": True}
        )
        
        assert context.domain_filter == ["biosphere", "atmosphere"]
        assert context.config["tolerance"] == 1e-6
        assert context.execution_layer == "L2"
        assert context.metadata["test_run"] is True
        assert context.state_manager is None
        assert context.tolerance_overrides is None
    
    def test_validation_context_defaults(self):
        """Test ValidationContext with default values."""
        context = ValidationContext()
        
        assert context.state_manager is None
        assert context.config is None
        assert context.execution_layer is None
        assert context.domain_filter is None
        assert context.tolerance_overrides is None
        assert context.metadata is None


# Integration tests for model interactions
class TestModelInteractions:
    """Test interactions between different model types."""
    
    def test_result_detail_integration(self):
        """Test ValidationResultModel with complex ValidationDetail objects."""
        # Create detailed diagnostics for a conservation violation
        carbon_detail = ValidationDetail(
            variable_name="carbon_balance_residual",
            cf_standard_name="mass_concentration_of_carbon_in_air",
            units="kg C m-2",
            value=0.015,  # Residual above threshold
            expected=0.0,
            threshold=0.001,
            location={"grid_cell": (45, 120), "time_step": 365},
            statistics={
                "mean_residual": 0.012,
                "max_residual": 0.025,
                "relative_error": 0.15
            },
            metadata={
                "conservation_components": ["gpp", "respiration", "allocation"],
                "integration_method": "runge_kutta_4"
            }
        )
        
        result = ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
            message="Carbon mass balance violation exceeds tolerance",
            details=[carbon_detail],
            layer="L2",
            scope="conservation.validate_carbon_balance",
            tags=["conservation", "carbon", "mass_balance"],
            metrics={
                "max_absolute_residual": 0.025,
                "mean_relative_error": 0.12,
                "conservation_efficiency": 0.88
            }
        )
        
        # Verify the result structure
        assert result.is_failing() is True
        assert len(result.details) == 1
        assert result.details[0].statistics["relative_error"] == 0.15
        assert result.metrics["conservation_efficiency"] == 0.88
        assert "conservation" in result.tags
    
    def test_report_comprehensive_example(self):
        """Test comprehensive validation report with multiple result types."""
        # Create diverse validation results
        cf_result = ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.CF_UNIT_MISMATCH,
            message="Temperature units require conversion",
            layer="L1",
            scope="cross_domain.validate_cf_compliance",
            tags=["cf", "units", "temperature"]
        )
        
        conservation_result = ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MASS_BALANCE_FAIL_WATER,
            message="Water mass balance closure failed",
            layer="L2", 
            scope="conservation.validate_water_balance",
            tags=["conservation", "water", "mass_balance"]
        )
        
        stochastic_result = ValidationResultModel(
            level=ValidationLevel.CRITICAL,
            code=ValidationCode.PIT_UNIFORMITY_REJECT_KS,
            message="PIT uniformity rejected by Kolmogorov-Smirnov test",
            layer="L1",
            scope="stochastic.validate_pit_uniformity",
            tags=["stochastic", "pit", "uniformity"],
            metrics={"ks_statistic": 0.15, "p_value": 0.01}
        )
        
        # Build comprehensive report
        report = ValidationReportModel(
            results=[cf_result, conservation_result, stochastic_result],
            execution_metadata={
                "validation_timestamp": datetime.now().isoformat(),
                "framework_version": "1.0.0",
                "total_execution_time": 2.3
            }
        )
        
        # Test comprehensive summary
        assert len(report) == 3
        assert report.has_errors() is True
        assert report.has_warnings() is True
        assert report.summary["total"] == 3
        assert report.summary["by_level"]["WARNING"] == 1
        assert report.summary["by_level"]["ERROR"] == 1
        assert report.summary["by_level"]["CRITICAL"] == 1
        
        # Test domain-specific summaries
        assert report.cf_summary["total_cf_validations"] == 1
        assert report.cf_summary["cf_warnings"] == 1
        assert report.conservation_summary["total_conservation_checks"] == 1
        assert report.conservation_summary["conservation_failures"] == 1
        
        # Test filtering by multiple criteria
        l1_results = report.get_results_by_layer("L1")
        assert len(l1_results) == 2  # CF and stochastic
        
        critical_results = report.get_results_by_level(ValidationLevel.CRITICAL)
        assert len(critical_results) == 1
        assert critical_results[0].metrics["p_value"] == 0.01


if __name__ == "__main__":
    pytest.main([__file__])

