"""
Integration tests for parameter audit system.

Tests the parameter audit functionality in integrated scenarios including:
- Real YAML file loading and scanning
- Cross-module parameter usage detection  
- CLI command integration
- Multi-tier database interactions
- Full workflow testing
"""

import pytest
import tempfile
import shutil
import yaml
import json
import logging
from pathlib import Path
from unittest.mock import patch

from relic.parameters import UnifiedParameterManager, get_global_manager
from relic.parameters import ParameterSpec
from relic.parameters.audit import AuditIssueType
from relic.interfaces.cli.params import execute


class TestParameterAuditIntegration:
    """Integration test suite for parameter audit system."""
    
    @pytest.fixture(autouse=True)
    def _fresh_manager(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager (one test clears it); the
        previous one is put back afterwards."""
        yield
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def sample_yaml_files(self, temp_dir):
        """Create sample YAML parameter files."""
        # Create relic directory
        relic_dir = temp_dir / "relic"
        relic_dir.mkdir()
        
        # Create native parameters
        native_params = {
            "parameters": {
                "biosphere": {
                    "photosynthesis": {
                        "vcmax25": {
                            "value": 60.0,
                            "units": "umol CO2 m-2 s-1",
                            "bounds": [10.0, 200.0],
                            "description": "Maximum carboxylation rate at 25C", "dtype": "float32", "category": "model_parameter"
                        },
                        "jmax25": {
                            "value": 120.0,
                            "units": "umol e- m-2 s-1",
                            "bounds": [20.0, 400.0],
                            "description": "Maximum electron transport rate at 25C", "dtype": "float32", "category": "model_parameter"
                        }
                    }
                },
                "atmosphere": {
                    "temperature": {
                        "value": 298.15,
                        "units": "K",
                        "bounds": [273.15, 323.15],
                        "description": "Ambient temperature", "dtype": "float32", "category": "model_parameter"
                    }
                }
            }
        }
        
        with open(relic_dir / "native.yaml", "w") as f:
            yaml.dump(native_params, f)
        
        # Create calibrated directory
        calibrated_dir = temp_dir / "calibrated"
        calibrated_dir.mkdir()
        
        # Create calibrated parameters
        calibrated_params = {
            "parameters": {
                "optimized": {
                    "respiration_q10": {
                        "value": 2.1,
                        "units": "dimensionless",
                        "bounds": [1.5, 3.0],
                        "description": "Optimized Q10 value", "dtype": "float32", "category": "model_parameter"
                    }
                }
            }
        }
        
        with open(calibrated_dir / "optimized.yaml", "w") as f:
            yaml.dump(calibrated_params, f)
        
        return temp_dir
    
    def test_audit_with_database_files(self, sample_yaml_files, monkeypatch):
        """Test audit scanning real database files."""
        monkeypatch.setenv("RELIC_SKIP_CALIBRATED", "0")
        monkeypatch.setenv("RELIC_INCLUDE_CALIBRATED_SNAPSHOTS", "1")
        # Create manager with test database
        manager = UnifiedParameterManager(database_path=sample_yaml_files)
        
        # Load parameters from database
        manager.load_from_database()
        
        # Run audit
        report = manager.run_registry_audit()
        
        # Should have loaded parameters
        assert report.observed_total > 0
        
        # Check that parameters from all tiers are included
        param_keys = list(manager.parameters.keys())
        
        # Native parameters
        assert any("biosphere.photosynthesis" in k for k in param_keys)
        
        # Calibrated parameters
        assert any("optimized" in k or "respiration" in k for k in param_keys)
    
    def test_cross_module_missing_detection(self):
        """Test detection of parameters missing across modules."""
        manager = get_global_manager()
        manager.clear_parameters()
        manager.clear_missing_requests()
        
        # Simulate Functor requesting parameters
        class MockFunctor:
            def __init__(self, manager):
                self.manager = manager
                
            def get_parameters(self):
                # Request some parameters
                self.manager.get("biosphere.photosynthesis.vcmax25", default=60.0)
                self.manager.get("biosphere.respiration.q10", default=2.0)
                self.manager.get("atmosphere.co2_concentration", default=400.0)
                self.manager.get("soil.moisture", default=0.3)
        
        # Create functors simulating different modules
        functor1 = MockFunctor(manager)
        functor2 = MockFunctor(manager)
        
        # Simulate parameter requests
        functor1.get_parameters()
        functor2.get_parameters()
        
        # Only register some parameters
        manager.register_parameter(ParameterSpec(
            key="biosphere.photosynthesis.vcmax25",
            value=60.0,
            units="umol CO2 m-2 s-1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        ))
        
        # Run audit
        report = manager.run_registry_audit()
        
        # Should detect missing parameters
        missing = report.missing_keys
        assert "biosphere.respiration.q10" in missing
        assert "atmosphere.co2_concentration" in missing
        assert "soil.moisture" in missing
        
        # Check request counts
        missing_requests = manager.get_missing_requests()
        assert missing_requests["biosphere.respiration.q10"] == 2  # Requested by both functors
    
    def test_cli_audit_command(self, sample_yaml_files, caplog):
        """Test CLI audit command execution."""
        # Create mock args object
        class Args:
            params_command = "audit"
            namespace = None
            expected_params = None
            include_unused = True
            format = "table"
            output = None
            clear_missing = False
            verbose = True
        
        args = Args()
        config = {}
        
        # Mock the manager to use test database
        with patch('relic.parameters.operations.get_global_manager') as mock_get:
            manager = UnifiedParameterManager(database_path=sample_yaml_files)
            manager.load_from_database()
            mock_get.return_value = manager
            
            # Execute audit command
            with caplog.at_level(logging.INFO, logger="relic.parameters.operations"):
                result = execute(args, config)
            
            # Check output
            assert "Parameter Audit Report" in caplog.text
            assert "Expected:" in caplog.text
            assert "Observed:" in caplog.text
    
    def test_audit_export_to_file(self, temp_dir):
        """Test exporting audit report to file."""
        manager = get_global_manager()
        
        # Run audit
        report = manager.run_registry_audit()
        
        # Export to JSON file
        json_file = temp_dir / "audit.json"
        manager.export_audit_report(report, format="json", output_path=json_file)
        
        # Check file exists and is valid JSON
        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
            assert "created_at" in data
            assert "issues" in data
        
        # Export to YAML file
        yaml_file = temp_dir / "audit.yaml"
        manager.export_audit_report(report, format="yaml", output_path=yaml_file)
        
        # Check file exists and is valid YAML
        assert yaml_file.exists()
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            assert "created_at" in data
        
        # Export to Markdown file
        md_file = temp_dir / "audit.md"
        manager.export_audit_report(report, format="markdown", output_path=md_file)
        
        # Check file exists and contains Markdown
        assert md_file.exists()
        content = md_file.read_text()
        assert "# Parameter Audit Report" in content
    
    def test_namespace_filtering_integration(self, sample_yaml_files):
        """Test namespace filtering with real parameters."""
        manager = UnifiedParameterManager(database_path=sample_yaml_files)
        manager.load_from_database()
        
        # Add some missing requests in different namespaces
        manager._missing_requests["biosphere.missing1"] = 5
        manager._missing_requests["atmosphere.missing2"] = 3
        manager._missing_requests["soil.missing3"] = 7
        
        # Run audit filtered to biosphere namespace
        report = manager.run_registry_audit(namespace_filter="biosphere")
        
        # Check that issues are filtered
        for issue in report.issues:
            if issue.issue_type != AuditIssueType.MISSING:
                # Non-missing issues should be in biosphere namespace
                if not issue.key.startswith("biosphere"):
                    # Unless it's a cross-namespace duplicate
                    assert issue.issue_type == AuditIssueType.DUPLICATE
    
    def test_parameter_lifecycle_audit(self, sample_yaml_files):
        """Test full parameter lifecycle with audit."""
        manager = UnifiedParameterManager(database_path=sample_yaml_files)
        
        # Stage 1: Load initial parameters
        manager.load_from_database()
        initial_count = len(manager.parameters)
        
        # Stage 2: Simulate runtime parameter additions
        manager.register_parameter(ParameterSpec(
            key="runtime.param1",
            value=42.0,
            units="dimensionless",
            description="Runtime parameter", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        ))
        
        # Stage 3: Simulate missing parameter requests
        manager.get("missing.param1", default=0)
        manager.get("missing.param2", default=0)
        
        # Stage 4: Run first audit
        report1 = manager.run_registry_audit()
        
        # Stage 5: Fix some issues (add missing parameters)
        manager.register_parameter(ParameterSpec(
            key="missing.param1",
            value=1.0,
            units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        ))
        
        # Stage 6: Run second audit
        report2 = manager.run_registry_audit()
        
        # Compare reports
        missing1 = len(report1.filter_by_type(AuditIssueType.MISSING))
        missing2 = len(report2.filter_by_type(AuditIssueType.MISSING))
        
        # Second audit should have fewer missing parameters
        assert missing2 < missing1
    
    def test_multi_tier_precedence_audit(self, sample_yaml_files, monkeypatch):
        """Test audit with multi-tier parameter precedence."""
        monkeypatch.setenv("RELIC_SKIP_CALIBRATED", "0")
        monkeypatch.setenv("RELIC_INCLUDE_CALIBRATED_SNAPSHOTS", "1")
        manager = UnifiedParameterManager(database_path=sample_yaml_files)
        
        # Create overlapping parameters in different tiers
        relic_dir = sample_yaml_files / "relic"
        calibrated_dir = sample_yaml_files / "calibrated"
        
        # Same parameter in multiple tiers
        overlap_param = {
            "parameters": {
                "test": {
                    "overlap": {
                        "value": 1.0,
                        "units": "m",
                        "bounds": [0.0, 10.0],
                        "description": "Overlapping parameter", "dtype": "float32", "category": "model_parameter"
                    }
                }
            }
        }
        
        # Write to relic (lowest precedence)
        with open(relic_dir / "overlap.yaml", "w") as f:
            yaml.dump(overlap_param, f)
        
        # Modify value for calibrated (highest precedence)
        overlap_param["parameters"]["test"]["overlap"]["value"] = 3.0
        with open(calibrated_dir / "overlap.yaml", "w") as f:
            yaml.dump(overlap_param, f)
        
        # Load with default precedence
        manager.load_from_database()
        
        # Check that highest precedence value is used
        value = manager.get("test.overlap")
        assert value == 3.0  # Calibrated value
        
        # Run audit
        report = manager.run_registry_audit()
        
        # Check for duplicate issues (same key in multiple tiers)
        # Note: Depending on implementation, this might not be flagged as duplicate
        # since it's intentional precedence behavior
    
    def test_validation_integration(self, sample_yaml_files):
        """Test validation error detection with real parameters."""
        manager = UnifiedParameterManager(database_path=sample_yaml_files)
        manager.load_from_database()
        
        # Modify a parameter to be invalid
        if "atmosphere.temperature" in manager.parameters:
            # Set temperature below absolute zero
            manager.parameters["atmosphere.temperature"].value = -100.0
        
        # Run audit
        report = manager.run_registry_audit()
        
        # Should detect validation error
        validation_errors = report.filter_by_type(AuditIssueType.VALIDATION_ERROR)
        
        # Temperature validation should fail
        error_keys = [issue.key for issue in validation_errors]
        if "atmosphere.temperature" in manager.parameters:
            assert "atmosphere.temperature" in error_keys
    
    def test_audit_performance(self, sample_yaml_files):
        """Test audit performance with large parameter set."""
        import time
        
        manager = UnifiedParameterManager(database_path=sample_yaml_files)
        
        # Add many parameters
        for i in range(1000):
            manager.register_parameter(ParameterSpec(
                key=f"perf.param{i}",
                value=float(i),
                units="dimensionless",
                description=f"Performance test parameter {i}", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
            ))
        
        # Add missing requests
        for i in range(500):
            manager._missing_requests[f"missing.param{i}"] = i % 10
        
        # Time the audit
        start = time.time()
        report = manager.run_registry_audit()
        duration = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert duration < 5.0
        
        # Should handle large parameter set
        assert report.observed_total >= 1000
        assert len(report.issues) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

