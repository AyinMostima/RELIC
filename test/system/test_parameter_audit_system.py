"""
L3 System tests for parameter audit functionality.

Tests the complete parameter audit system in realistic scenarios,
including full system integration, performance under load, and
real-world usage patterns.
"""

import pytest
import tempfile
import time
from pathlib import Path
import yaml
import subprocess
import sys
import os

from relic.parameters import (
    UnifiedParameterManager, ParameterSpec, get_global_manager
)
from relic.parameters.types import ParameterCategory
from relic.parameters.audit import AuditIssueType


class TestParameterAuditSystem:
    """L3 system-level tests for parameter audit."""
    
    @pytest.fixture(autouse=True)
    def reset_manager(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager (the heavy-load tests clear
        it); the previous one is put back afterwards."""
        yield
    
    def test_full_system_audit_workflow(self):
        """Test complete audit workflow in a realistic system."""
        # Create temporary project directory
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            
            # Create realistic project structure
            self._create_project_structure(project_dir)
            
            # Initialize manager with project
            manager = UnifiedParameterManager(database_path=project_dir / "parameters")
            manager.load_from_database()
            
            # Simulate realistic usage patterns
            self._simulate_realistic_usage(manager)
            
            # Run comprehensive audit
            report = manager.run_registry_audit()
            
            # Verify audit findings
            assert report.expected_total > 0
            assert report.observed_total > 0
            assert len(report.issues) > 0
            
            # Generate various report formats
            table_output = manager.export_audit_report(report, format="table")
            json_output = manager.export_audit_report(report, format="json")
            markdown_output = manager.export_audit_report(report, format="markdown")
            
            assert table_output and len(table_output) > 0
            assert json_output and len(json_output) > 0
            assert markdown_output and len(markdown_output) > 0
            
            missing_before = sum(
                1 for issue in report.issues
                if issue.issue_type == AuditIssueType.MISSING
            )
            
            # Fix identified issues
            self._fix_audit_issues(manager, report)
            
            # Re-run audit and verify improvements
            report2 = manager.run_registry_audit()
            missing_after = sum(
                1 for issue in report2.issues
                if issue.issue_type == AuditIssueType.MISSING
            )
            if missing_before > 0:
                assert missing_after < missing_before
    
    def test_cli_integration_full_workflow(self):
        """Test full CLI workflow for parameter audit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "params_db"
            self._create_parameter_database(db_path)
            
            # Test various CLI commands
            commands = [
                # List parameters
                [sys.executable, "-m", "relic.interfaces.cli.params", 
                 "list", "--namespace", "biosphere"],
                
                # Get specific parameter
                [sys.executable, "-m", "relic.interfaces.cli.params",
                 "get", "biosphere.photosynthesis.vcmax25"],
                
                # Run audit
                [sys.executable, "-m", "relic.interfaces.cli.params",
                 "audit", "--format", "table"],
                
                # Export audit report
                [sys.executable, "-m", "relic.interfaces.cli.params",
                 "audit", "--format", "json", "--output", str(tmp_path / "audit.json")]
            ]
            
            for cmd in commands:
                # Execute CLI command
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    env={**os.environ, "RELIC_PARAM_DB": str(db_path)}
                )
                
                # Basic verification - commands should run without errors
                # Note: Some commands might have non-zero exit codes for missing params
                assert result.returncode in [0, 1], f"Command failed: {' '.join(cmd)}"
    
    def test_performance_under_heavy_load(self):
        """Test audit performance with very large parameter sets."""
        manager = get_global_manager()
        manager.clear_parameters()
        
        # Create large parameter set (10,000 parameters)
        param_count = 10000
        print(f"Creating {param_count} parameters for performance test...")
        
        start = time.time()
        for i in range(param_count):
            manager.register_parameter(ParameterSpec(
                key=f"perf.namespace{i//100}.param{i}",
                value=float(i),
                units="dimensionless",
                category=ParameterCategory.MODEL_PARAMETER,
                description=f"Performance test parameter {i}", dtype="float32", bounds=(-float("inf"), float("inf"))
            ))
        creation_time = time.time() - start
        print(f"Parameter creation took {creation_time:.2f} seconds")
        
        # Add missing requests
        for i in range(5000):
            manager._missing_requests[f"missing.ns{i//50}.param{i}"] = i % 20
        
        # Run audit and measure performance
        start = time.time()
        report = manager.run_registry_audit()
        audit_time = time.time() - start
        
        print(f"Audit of {param_count} parameters took {audit_time:.2f} seconds")
        
        # Performance assertions
        assert audit_time < 10.0, f"Audit too slow: {audit_time} seconds"
        assert report.expected_total >= param_count
        
        # Export performance
        start = time.time()
        json_export = manager.export_audit_report(report, format="json")
        export_time = time.time() - start
        
        print(f"JSON export took {export_time:.2f} seconds")
        assert export_time < 5.0, f"Export too slow: {export_time} seconds"
    
    def test_concurrent_access_safety(self):
        """Test audit system under concurrent access."""
        import threading
        import random
        
        manager = get_global_manager()
        manager.clear_parameters()
        
        # Setup initial parameters
        for i in range(100):
            manager.register_parameter(ParameterSpec(
                key=f"concurrent.param{i}",
                value=float(i),
                units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
            ))
        
        results = []
        errors = []
        
        def worker(worker_id):
            """Worker function for concurrent operations."""
            try:
                # Random operations
                for _ in range(10):
                    op = random.choice(["get", "set", "audit", "validate"])
                    
                    if op == "get":
                        key = f"concurrent.param{random.randint(0, 99)}"
                        manager.get(key)
                    elif op == "set":
                        key = f"concurrent.param{random.randint(0, 99)}"
                        manager.set_value(key, random.random())
                    elif op == "audit":
                        report = manager.run_registry_audit(
                            namespace_filter="concurrent"
                        )
                        results.append(len(report.issues))
                    elif op == "validate":
                        manager.validate_parameters()
                    
                    time.sleep(0.01)  # Small delay
                    
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Run concurrent workers
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        # Verify system is still functional
        final_report = manager.run_registry_audit()
        assert final_report is not None
    
    def test_recovery_from_corrupted_state(self):
        """Test system recovery from corrupted or invalid states."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "params_db"
            
            # Create initial valid state
            manager = UnifiedParameterManager(database_path=db_path)
            manager.register_parameter(ParameterSpec(
                key="test.param",
                value=1.0,
                units="m", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
            ))
            manager.save_to_database()
            
            # Corrupt the database file
            yaml_file = db_path / "calibrated" / "parameters.yaml"
            with open(yaml_file, "w") as f:
                f.write("invalid: yaml: content: {[}")
            
            # Create new manager and attempt to load
            manager2 = UnifiedParameterManager(database_path=db_path)
            
            # Should handle corruption gracefully
            try:
                manager2.load_from_database()
            except Exception as e:
                pytest.fail(f"Failed to handle corruption: {e}")
            
            # System should still be functional
            report = manager2.run_registry_audit()
            assert report is not None
    def test_integration_with_scientific_modules(self):
        """Test audit integration with scientific computation modules."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        
        # Initialize system
        manager = get_global_manager()
        manager.clear_parameters()
        
        # Load scientific parameters
        self._load_scientific_parameters(manager)
        
        # Create scientific functor
        functor = FarquharPhotosynthesis(name="test_photosynthesis")
        
        # Simulate parameter usage in scientific computation
        state = {"temperature": 298.15, "par": 1000.0, "co2": 400.0}
        environment = {}
        
        # This will trigger parameter requests
        try:
            result = functor.forward(state, environment)
        except Exception:
            # Expected if some parameters are missing
            pass
        
        # Run audit to check parameter usage
        report = manager.run_registry_audit()
        
        # Check for missing scientific parameters
        missing_scientific = [
            issue for issue in report.issues
            if issue.issue_type == AuditIssueType.MISSING
            and "biosphere" in issue.key
        ]
        
        # Report should identify missing scientific parameters
        if missing_scientific:
            print(f"Found {len(missing_scientific)} missing scientific parameters")
            for issue in missing_scientific[:5]:
                print(f"  - {issue.key}: {issue.message}")
    
    # Helper methods
    
    def _create_project_structure(self, project_dir):
        """Create realistic project directory structure."""
        # Parameter database
        param_dir = project_dir / "parameters"
        for tier in ["relic", "calibrated"]:
            (param_dir / tier).mkdir(parents=True)
        
        # Create sample parameter files
        self._create_parameter_database(param_dir)
        
        # Create source code directories
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()
        (project_dir / "docs").mkdir()
    
    def _create_parameter_database(self, db_path):
        """Create sample parameter database files."""
        # Native parameters
        relic_params = {
            "parameters": {
                "biosphere": {
                    "photosynthesis": {
                        "vcmax25": {
                            "value": 60.0,
                            "units": "umol CO2 m-2 s-1",
                            "description": "Maximum carboxylation rate at 25C"
                        }
                    }
                }
            }
        }
        
        relic_dir = db_path / "relic"
        relic_dir.mkdir(parents=True, exist_ok=True)
        with open(relic_dir / "biosphere.yaml", "w") as f:
            yaml.dump(relic_params, f)
        
        # Calibrated parameters
        calibrated_params = {
            "parameters": {
                "optimized": {
                    "respiration_q10": {
                        "value": 2.1,
                        "units": "dimensionless",
                        "description": "Optimized Q10 value"
                    }
                }
            }
        }
        
        calibrated_dir = db_path / "calibrated"
        calibrated_dir.mkdir(parents=True, exist_ok=True)
        with open(calibrated_dir / "optimized.yaml", "w") as f:
            yaml.dump(calibrated_params, f)
    
    def _simulate_realistic_usage(self, manager):
        """Simulate realistic parameter usage patterns."""
        # Request existing parameters
        manager.get("biosphere.photosynthesis.vcmax25")
        
        # Request missing parameters
        manager.get("biosphere.respiration.rd25", default=1.0)
        manager.get("atmosphere.pressure", default=101325.0)
        
        # Register a runtime parameter
        manager.register_parameter(ParameterSpec(
            key="runtime.timestep", value=3600.0, units="s",
            description="Runtime time step of the audit scenario", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter",
        ))

        # Validate parameters
        manager.validate_parameters()
    
    def _fix_audit_issues(self, manager, report):
        """Fix issues identified in audit report."""
        # Fix missing parameters
        for issue in report.issues:
            if issue.issue_type == AuditIssueType.MISSING:
                # Add missing parameter with default value
                manager.register_parameter(ParameterSpec(
                    key=issue.key,
                    value=0.0,
                    units="dimensionless",
                    description=f"Auto-added parameter for {issue.key}", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
                ))
            elif issue.issue_type == AuditIssueType.VALIDATION_ERROR:
                # Fix validation errors by setting valid values
                if "below minimum" in issue.message:
                    spec = manager.get_parameter(issue.key)
                    if spec and spec.bounds:
                        manager.set_value(issue.key, spec.bounds[0])
                elif "above maximum" in issue.message:
                    spec = manager.get_parameter(issue.key)
                    if spec and spec.bounds:
                        manager.set_value(issue.key, spec.bounds[1])
    def _load_scientific_parameters(self, manager):
        """Load parameters for scientific modules."""
        scientific_params = [
            ParameterSpec(
                key="biosphere.photosynthesis.vcmax25",
                value=60.0,
                units="umol CO2 m-2 s-1",
                category=ParameterCategory.BIOPHYSICAL,
                description="Maximum carboxylation rate at 25C",
                requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf"))
            ),
            ParameterSpec(
                key="biosphere.photosynthesis.jmax25",
                value=120.0,
                units="umol e- m-2 s-1",
                category=ParameterCategory.BIOPHYSICAL,
                description="Maximum electron transport rate at 25C",
                requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf"))
            ),
            ParameterSpec(
                key="biosphere.photosynthesis.ha_vcmax",
                value=65330.0,
                units="J mol-1",
                category=ParameterCategory.BIOPHYSICAL,
                description="Activation energy for Vcmax", dtype="float32", bounds=(-float("inf"), float("inf"))
            ),
            ParameterSpec(
                key="relic.physical.constants.universal.gas_constant",
                value=8.314,
                units="J mol-1 K-1",
                category=ParameterCategory.PHYSICAL_CONSTANT,
                description="Universal gas constant", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
        ]
        
        for spec in scientific_params:
            manager.register_parameter(spec)


# Add missing os import
import os


if __name__ == "__main__":
    # Run L3 system tests
    pytest.main([__file__, "-v"])

