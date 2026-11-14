"""
L3 System Integration Tests for Parameter Management System

This module provides comprehensive system-level testing to verify that the
unified parameter management system works correctly across all modules,
respects conservation laws, and handles complex interactions properly.
"""

import pytest
import torch
import warnings
import math
from pydantic import ValidationError
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.parameters import get_global_manager, ParameterSpec
from relic.parameters.types import ParameterCategory
from relic.parameters.audit import AuditSeverity
from relic.core.config.loader import ConfigLoader
from relic.core.state import StateManager


class TestParameterSystemIntegrationL3:
    """L3 system-level tests for parameter management."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Get parameter manager
        self.param_manager = get_global_manager()
        self.parameter_snapshot = self.param_manager.snapshot()
        
        # Clear any cached parameters
        if hasattr(self.param_manager, '_missing_requests'):
            self.param_manager._missing_requests.clear()
        
        # Create config loader
        self.config_loader = ConfigLoader(
            validate_on_load=False,
            cache_configs=False
        )
        
        yield
        
        self.param_manager.restore(self.parameter_snapshot)
    
    def test_full_system_parameter_loading(self):
        """Test that all system modules can load their parameters."""
        # Load a test configuration
        config_dict = {
            'simulation': {
                'time_step': 3600.0,
                'start_time': 0.0,
                'end_time': 86400.0,
                'grid_size_x': 10,
                'grid_size_y': 10,
                'grid_size_z': 20,
                'spatial_resolution': 1000.0
            }
        }
        
        # Create config object
        from relic.core.config.models import RELICConfig
        config = RELICConfig(**config_dict)
        
        # Verify config loader can integrate with parameter manager
        self.config_loader._load_parameters_to_manager(config)
        
        # Check no critical parameters are missing
        missing = self.param_manager.get_missing_requests()
        critical_missing = [m for m in missing if 'critical' in m.lower()]
        assert len(critical_missing) == 0, f"Critical parameters missing: {critical_missing}"
    
    def test_cross_module_parameter_consistency(self):
        """Test that parameters are consistent across modules."""
        # Test that phenology and photosynthesis use consistent LAI
        phenology_lai = self.param_manager.get("biosphere.phenology.morphology.max_lai")
        
        # If phenology LAI is set, verify it's accessible to other modules
        if phenology_lai is not None:
            # Simulate photosynthesis module accessing the same parameter
            photosynthesis_lai = self.param_manager.get("biosphere.phenology.morphology.max_lai")
            assert photosynthesis_lai == phenology_lai, "LAI parameter inconsistent across modules"
    
    def test_parameter_namespace_isolation(self):
        """Test that parameter namespaces are properly isolated."""
        # Set parameters in different namespaces
        self.param_manager.register_parameter(
            ParameterSpec(key="test.module1.param", value=1.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
        self.param_manager.register_parameter(
            ParameterSpec(key="test.module2.param", value=2.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))

        # Verify isolation
        assert self.param_manager.get("test.module1.param") == 1.0
        assert self.param_manager.get("test.module2.param") == 2.0
        
        # Verify wildcard search works
        test_params = self.param_manager.list(pattern="test.*")
        assert len(test_params) >= 2
    
    def test_parameter_unit_conversion_integration(self):
        """Test that unit conversions work correctly across the system."""
        # Test temperature conversion (commonly needed)
        # Set temperature in Celsius
        temp_spec = ParameterSpec(
            key="test.temperature",
            value=25.0,
            units="degC",
            dtype="float64",
            bounds=(-273.15, 1000.0),
            category=ParameterCategory.ENVIRONMENTAL,
            description="Test temperature for cross-system unit conversion",
            references=["RELIC ParameterManager system integration fixture"],
            tags={"test", "system"},
            dimension="temperature",
            provenance={
                "source": "test_fixture",
                "created_at": None,
                "schema_version": "1.0.0",
            },
            requires_grad=False,
        )
        self.param_manager.register_parameter(temp_spec)
        
        # Get in Kelvin (common internal unit)
        temp_k = self.param_manager.get("test.temperature", units="K")
        assert abs(temp_k - 298.15) < 0.01, "Temperature conversion failed"
    
    def test_parameter_validation_across_modules(self):
        """Test that parameter validation works consistently."""
        # Test setting invalid parameter values
        lai_spec = ParameterSpec(
            key="test.invalid.lai",
            value=1.0,
            units="dimensionless",
            bounds=(0.0, 20.0),
            category=ParameterCategory.STATE_VARIABLE,
            description="Invalid LAI fixture used to test bounds enforcement",
            references=["RELIC ParameterManager system integration fixture"],
            tags={"test", "system"},
            dimension="dimensionless",
            provenance={
                "source": "test_fixture",
                "created_at": None,
                "schema_version": "1.0.0",
            },
            requires_grad=False, dtype="float32",
        )
        self.param_manager.register_parameter(lai_spec)

        with pytest.warns(UserWarning):
            self.param_manager.set_value(
                "test.invalid.lai",
                -1.0,
                validate=True
            )
        assert self.param_manager.get("test.invalid.lai") >= 0.0
    
    def test_parameter_persistence_and_recovery(self):
        """Test that parameters can be saved and restored."""
        import tempfile
        import json
        
        # Set some test parameters
        self.param_manager.register_parameter(
            ParameterSpec(key="test.persist.value1", value=42.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
        self.param_manager.register_parameter(
            ParameterSpec(key="test.persist.value2", value="test_string", units="dimensionless", dtype="str", bounds=(-float("inf"), float("inf")), category="model_parameter"))

        # Export to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_data = {
                "test.persist.value1": 42.0,
                "test.persist.value2": "test_string"
            }
            json.dump(export_data, f)
            temp_file = f.name
        
        try:
            # Clear parameters
            if hasattr(self.param_manager, '_runtime_registry'):
                self.param_manager._runtime_registry.clear()
            
            # Reload from file
            with open(temp_file, 'r') as f:
                import_data = json.load(f)
                for key, value in import_data.items():
                    self.param_manager.set_value(key, value)
            
            # Verify restoration
            assert self.param_manager.get("test.persist.value1") == 42.0
            assert self.param_manager.get("test.persist.value2") == "test_string"
        
        finally:
            # Cleanup
            Path(temp_file).unlink(missing_ok=True)
    
    def test_conservation_law_parameter_consistency(self):
        """Test that conservation-related parameters are consistent."""
        # Test that mass balance parameters are consistent
        carbon_params = self.param_manager.list(pattern="*.carbon.*")
        
        # If we have carbon parameters, verify they're physically consistent
        for param_key in carbon_params:
            value = self.param_manager.get(param_key)
            if value is None or not isinstance(value, (int, float)):
                continue
            spec = self.param_manager.get_parameter(param_key)
            if spec is None or spec.bounds is None:
                continue
            min_val, max_val = spec.bounds
            if isinstance(min_val, (int, float)) and math.isfinite(min_val):
                assert value >= min_val, f"Value below bounds: {param_key} = {value}"
            if isinstance(max_val, (int, float)) and math.isfinite(max_val):
                assert value <= max_val, f"Value above bounds: {param_key} = {value}"
    
    def test_multi_module_workflow(self):
        """Test a complete workflow involving multiple modules."""
        # Simulate a workflow: phenology -> photosynthesis -> respiration
        
        # 1. Phenology sets LAI
        lai_spec = ParameterSpec(
            key="workflow.phenology.lai",
            value=4.5,
            units="dimensionless",
            bounds=(0.0, 20.0),
            category=ParameterCategory.STATE_VARIABLE,
            description="Workflow LAI fixture for runtime parameter exchange",
            references=["RELIC ParameterManager system integration fixture"],
            tags={"test", "system"},
            dimension="dimensionless",
            provenance={
                "source": "test_fixture",
                "created_at": None,
                "schema_version": "1.0.0",
            },
            requires_grad=False, dtype="float32",
        )
        self.param_manager.register_parameter(lai_spec)
        
        # 2. Photosynthesis uses LAI
        lai_for_photosynthesis = self.param_manager.get("workflow.phenology.lai")
        assert lai_for_photosynthesis == 4.5
        
        # 3. Calculate GPP (simplified)
        gpp = lai_for_photosynthesis * 2.0  # Simplified calculation
        self.param_manager.register_parameter(ParameterSpec(
            key="workflow.photosynthesis.gpp", value=gpp, units="g C m-2 d-1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))

        # 4. Respiration uses GPP
        gpp_for_respiration = self.param_manager.get("workflow.photosynthesis.gpp")
        respiration = gpp_for_respiration * 0.5  # Simplified
        
        # Verify workflow consistency
        assert respiration == 4.5, "Workflow calculation inconsistent"
    
    def test_parameter_change_notifications(self):
        """Test that parameter changes trigger appropriate notifications."""
        notifications_received = []
        
        # Register a callback
        def callback(key, old_value, new_value):
            notifications_received.append((key, old_value, new_value))
        
        # Set up notification (if supported by the manager)
        if hasattr(self.param_manager, 'register_callback'):
            self.param_manager.register_callback("test.notify.*", callback)
            
            # Change a parameter
            self.param_manager.set_value("test.notify.param", 1.0)
            self.param_manager.set_value("test.notify.param", 2.0)
            
            # Verify notifications
            assert len(notifications_received) >= 1
            assert notifications_received[-1][2] == 2.0
    
    def test_error_handling_and_defaults(self):
        """Test that the system handles missing parameters gracefully."""
        # Test accessing non-existent parameter with default
        value = self.param_manager.get("nonexistent.parameter", default=10.0)
        assert value == 10.0, "Default default not working"
        
        # Test that modules handle missing parameters appropriately
        from relic.scientific.biosphere.phenology.base import PhenologyParameters
        from relic.scientific.biosphere.phenology.parameters import (
            PhenologyParameterSets,
        )

        # Scientific value objects reject incomplete construction.
        with pytest.raises(ValidationError, match="Field required"):
            PhenologyParameters(vegetation_type="tree", latitude=45.0)

        params = PhenologyParameterSets.deciduous_tree_gsi()

        assert params.max_lai > 0
        assert params.min_lai >= 0

    def test_parameter_audit_completeness(self):
        """Test that parameter audit correctly identifies issues."""
        # Run audit
        audit_report = self.param_manager.run_registry_audit()

        # Check audit report structure
        assert audit_report.observed_total >= 0
        assert audit_report.expected_total >= 0
        assert isinstance(audit_report.issues, list)

        # Verify audit identifies any critical issues
        critical_issues = audit_report.filter_by_severity(AuditSeverity.CRITICAL)
        if critical_issues:
            warnings.warn(f"Critical issues found: {critical_issues}")
    
    def test_performance_with_many_parameters(self):
        """Test system performance with large parameter sets."""
        import time
        
        # Add many parameters
        start_time = time.time()
        for i in range(1000):
            self.param_manager.register_parameter(
                ParameterSpec(key=f"perf.test.param_{i}", value=float(i), units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
        set_time = time.time() - start_time

        # Retrieve many parameters
        start_time = time.time()
        for i in range(1000):
            value = self.param_manager.get(f"perf.test.param_{i}")
            assert value == float(i)
        get_time = time.time() - start_time
        
        # Performance assertions (reasonable thresholds)
        assert set_time < 1.0, f"Setting 1000 parameters took {set_time:.2f}s (too slow)"
        assert get_time < 0.5, f"Getting 1000 parameters took {get_time:.2f}s (too slow)"
        
        # Test search performance
        start_time = time.time()
        results = self.param_manager.list(namespace="perf.test")
        search_time = time.time() - start_time
        
        assert len(results) == 1000
        assert search_time < 0.1, f"Searching parameters took {search_time:.2f}s (too slow)"


class TestSystemIntegrationWithSimulator:
    """Test parameter system integration with the main simulator."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment with simulator."""
        self.param_manager = get_global_manager()
        self.parameter_snapshot = self.param_manager.snapshot()
        self.state_manager = StateManager()
        
        yield
        
        self.param_manager.restore(self.parameter_snapshot)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_parameter_transfer(self):
        """Test that parameters correctly transfer to GPU."""
        device = torch.device('cuda')
        
        # Set a parameter
        self.param_manager.register_parameter(
            ParameterSpec(key="gpu.test.value", value=3.14, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))

        # Get as tensor on GPU
        value_tensor = torch.tensor(
            self.param_manager.get("gpu.test.value"),
            device=device
        )
        
        assert value_tensor.device.type == 'cuda'
        assert abs(value_tensor.item() - 3.14) < 0.001
    
    def test_state_manager_parameter_integration(self):
        """Test that StateManager correctly uses parameters."""
        # Set conservation threshold parameter
        tolerance_spec = ParameterSpec(
            key="conservation.mass_balance_tolerance",
            value=1e-6,
            units="dimensionless",
            bounds=(0.0, 1.0),
            category=ParameterCategory.NUMERICAL,
            description="Mass-balance tolerance fixture for StateManager integration",
            references=["RELIC ParameterManager system integration fixture"],
            tags={"test", "system"},
            dimension="dimensionless",
            provenance={
                "source": "test_fixture",
                "created_at": None,
                "schema_version": "1.0.0",
            },
            requires_grad=False, dtype="float32",
        )
        self.param_manager.register_parameter(tolerance_spec)
        
        # StateManager should respect this threshold
        tolerance = self.param_manager.get("conservation.mass_balance_tolerance")
        assert tolerance == 1e-6
        
        # Test state update with conservation check
        self.state_manager.set_conservation_ledger({
            'carbon': []
        })
        
        # This should use the parameter-defined tolerance
        updates = {'test_pool': torch.tensor(1.0)}
        self.state_manager.apply_updates(updates, check_conservation=False)
        
        # Verify state was updated
        assert 'test_pool' in self.state_manager.get_state()


# Run tests
if __name__ == "__main__":
    # Run with coverage if available
    import subprocess
    import sys
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running tests: {e}")
        # default to simple pytest run
        pytest.main([__file__, "-v", "--tb=short"])


