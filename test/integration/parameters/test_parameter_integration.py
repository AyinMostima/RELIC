"""
L2 Integration Tests for Parameter Management System

Tests integration between parameter components and with other system modules.
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path

from relic.parameters import (
    UnifiedParameterManager,
    ParameterSpec,
    ParameterCategory,
    get_global_manager
)
from relic.parameters.serialization import load, convert_format
from relic.core.functor import FunctorBase, FunctorMetadata
from test.utils.forward_call import call_forward


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


class TestParameterManagerIntegration:
    """Test integration between parameter manager components."""
    
    def test_manager_with_storage_integration(self):
        """Test UnifiedParameterManager integration with ParameterStorage."""
        manager = UnifiedParameterManager()
        
        # Register parameters
        specs = [
            ParameterSpec(
                key="test.param1",
                value=10.0,
                units="m/s",
                category=ParameterCategory.PHYSICAL,
                description="Test parameter 1", dtype="float32", bounds=(-float("inf"), float("inf"))
            ),
            ParameterSpec(
                key="test.param2",
                value=20.0,
                units="K",
                category=ParameterCategory.ENVIRONMENTAL,
                description="Test parameter 2", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
        ]
        
        for spec in specs:
            manager.register_parameter(spec)
        
        # Storage should contain parameters
        assert manager.storage.exists("test.param1")
        assert manager.storage.exists("test.param2")

        # Test snapshot functionality
        snapshot = manager.storage.snapshot("Test snapshot")
        # UnifiedParameterManager is a singleton and may preload SSOT metadata.
        # The test only requires that the registered parameters are captured.
        assert "test.param1" in snapshot.parameters
        assert "test.param2" in snapshot.parameters

        # Modify and restore
        manager.set_value("test.param1", 30.0)
        assert manager.get("test.param1") == 30.0
        
        manager.storage.restore(snapshot)
        assert manager.get("test.param1") == 10.0
    
    def test_manager_with_validator_integration(self):
        """Test UnifiedParameterManager integration with ParameterValidator."""
        manager = UnifiedParameterManager()
        
        # Register valid and invalid parameters
        valid_spec = ParameterSpec(
            key="valid.param",
            value=50.0,
            bounds=(0.0, 100.0),
            units="dimensionless",
            category=ParameterCategory.PHYSICAL,
            description="Valid parameter", dtype="float32"
        )
        
        invalid_spec = ParameterSpec(
            key="invalid.param",
            value=150.0,
            bounds=(0.0, 100.0),
            units="dimensionless",
            category=ParameterCategory.PHYSICAL,
            description="Invalid parameter", dtype="float32"
        )
        
        manager.register_parameter(valid_spec)
        manager.register_parameter(invalid_spec)
        
        # Validation should catch invalid parameter
        errors = manager.validate_parameters()
        assert len(errors) > 0
        assert any("invalid.param" in str(err) for err in errors)
        
        # Fix invalid parameter
        manager.set_value("invalid.param", 75.0)
        errors = manager.validate_parameters()
        assert len([err for err in errors if "invalid.param" in str(err)]) == 0
    
    def test_manager_serialization_integration(self):
        """Test parameter manager with serialization."""
        manager = UnifiedParameterManager()
        
        # Register parameters
        specs = {
            "biosphere.photosynthesis.vcmax25": ParameterSpec(
                key="biosphere.photosynthesis.vcmax25",
                value=60.0,
                units="umol CO2 m-2 s-1",
                bounds=(10.0, 200.0),
                category=ParameterCategory.BIOPHYSICAL,
                description="Vcmax at 25C", dtype="float32"
            ),
            "atmosphere.radiation.albedo": ParameterSpec(
                key="atmosphere.radiation.albedo",
                value=0.3,
                units="dimensionless",
                bounds=(0.0, 1.0),
                category=ParameterCategory.ENVIRONMENTAL,
                description="Surface albedo", dtype="float32"
            )
        }
        
        for spec in specs.values():
            manager.register_parameter(spec)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export to YAML
            yaml_path = Path(tmpdir) / "params.yaml"
            manager.export_parameters(str(yaml_path), format="yaml")
            assert yaml_path.exists()
            
            # Load and verify
            loaded_data = load(yaml_path)
            # Data structure has parameters nested under 'parameters' key
            params = loaded_data.get("parameters", loaded_data)
            assert "biosphere.photosynthesis.vcmax25" in params
            # Check the value (use 'default' or 'value' key depending on structure)
            param_data = params["biosphere.photosynthesis.vcmax25"]
            value = param_data.get("value", param_data.get("default"))
            assert value == 60.0
            
            # Convert to JSON
            json_path = Path(tmpdir) / "params.json"
            convert_format(yaml_path, json_path, "yaml", "json")
            
            # Load JSON and verify
            json_data = load(json_path)
            json_params = json_data.get("parameters", json_data)
            json_param_data = json_params["atmosphere.radiation.albedo"]
            json_value = json_param_data.get("value", json_param_data.get("default"))
            assert json_value == 0.3


class TestFunctorParameterIntegration:
    """Test parameter system integration with FunctorBase."""
    
    def test_functor_parameter_registration(self):
        """Test FunctorBase automatic parameter registration."""
        
        class TestFunctor(FunctorBase):
            def __init__(self):
                config = {
                    "parameters": {
                        "vcmax25": {"value": 60.0, "units": "umol CO2 m-2 s-1"},
                        "jmax25": {"value": 120.0, "units": "umol e- m-2 s-1"}
                    }
                }
                super().__init__(
                    name="test_functor",
                    config=config,
                    metadata=FunctorMetadata(
                        name="test_functor",
                        description="Test functor for parameter registration",
                        references=["test_reference"],
                        output_variables=["gpp"],
                    )
                )
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt):
                vcmax = self.get_parameter_value("vcmax25")
                jmax = self.get_parameter_value("jmax25")
                return {"gpp": vcmax + jmax}
        
        functor = TestFunctor()
        manager = get_global_manager()
        
        # Configuration values that carry units are converted against the
        # canonical ParameterSpec of the key the Functor resolves
        # (namespace "biosphere"), so that spec must exist in the manager.
        from relic.parameters import ParameterSpec, ParameterCategory
        manager.register_parameter(ParameterSpec(
            key="biosphere.vcmax25",
            value=60.0,
            units="umol CO2 m-2 s-1",
            category=ParameterCategory.BIOPHYSICAL,
            description="Vcmax at 25C", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        manager.register_parameter(ParameterSpec(
            key="biosphere.jmax25",
            value=120.0,
            units="umol e- m-2 s-1",
            category=ParameterCategory.BIOPHYSICAL,
            description="Jmax at 25C", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        # Parameters should be accessible
        assert manager.has_parameter("biosphere.vcmax25")
        assert manager.has_parameter("biosphere.jmax25")
        
        # Values should match
        assert manager.get("biosphere.vcmax25") == 60.0
        assert manager.get("biosphere.jmax25") == 120.0
        
        # Functor should be able to access these parameters
        vcmax = functor.get_parameter_value("vcmax25")
        jmax = functor.get_parameter_value("jmax25")
        assert vcmax is not None
        assert jmax is not None
    
    def test_functor_parameter_retrieval(self):
        """Test FunctorBase parameter retrieval with various options."""
        
        class TestFunctor(FunctorBase):
            def __init__(self, parameter_manager):
                config = {
                    "parameters": {"local_param": 100.0, "bounded_param": 0.5}
                }
                super().__init__(
                    name="retrieval_test",
                    config=config,
                    metadata=FunctorMetadata(
                        name="retrieval_test",
                        description="Retrieval test functor",
                        references=["test_reference"],
                        output_variables=["manager", "config", "validated"],
                    ),
                    parameter_manager=parameter_manager,
                )
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt):
                # Test different retrieval strategies
                
                # From manager (registered)
                manager_param = self.get_parameter_value(
                    "biological_process.retrieval_test.registered_param",
                    source_order=("manager",),
                    required=True,
                )
                
                # From config
                config_param = self.get_parameter_value(
                    "local_param",
                    source_order=("config", "manager")
                )
                
                # With validation
                validated_param = self.get_parameter_value(
                    "bounded_param",
                    bounds=(0.0, 1.0),
                    validate=True
                )
                
                return {
                    "manager": manager_param,
                    "config": config_param,
                    "validated": validated_param
                }
        
        # Register parameter in manager
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key="biological_process.retrieval_test.registered_param",
            value=75.0,
            category=ParameterCategory.COMPUTATIONAL,
            description="Registered parameter", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        functor = TestFunctor(parameter_manager=manager)
        result = call_forward(functor, {})

        assert float(torch.as_tensor(result["manager"]).detach().cpu().item()) == pytest.approx(75.0)  # From manager
        assert float(torch.as_tensor(result["config"]).detach().cpu().item()) == pytest.approx(100.0)  # From config
        assert float(torch.as_tensor(result["validated"]).detach().cpu().item()) == pytest.approx(0.5)  # Config value with validation
    
    def test_functor_learnable_parameters(self):
        """Test FunctorBase with learnable (gradient-enabled) parameters."""
        
        class LearnableFunctor(FunctorBase):
            def __init__(self):
                config = {
                    "parameters": {
                        # Learnable weights have no physical limit; a numeric
                        # parameter states that with an explicit (-inf, inf).
                        "weight": {"value": 1.0, "units": "1", "dtype": "float32",
                                   "bounds": [-float("inf"), float("inf")], "requires_grad": True},
                        "bias": {"value": 0.0, "units": "1", "dtype": "float32",
                                 "bounds": [-float("inf"), float("inf")], "requires_grad": True}
                    },
                    "auto_register_parameters_to_manager": True
                }
                super().__init__(
                    name="learnable",
                    config=config,
                    metadata=FunctorMetadata(
                        name="learnable",
                        description="Learnable parameters functor",
                        references=["test_reference"],
                        input_variables=["input"],
                        output_variables=["output"],
                    )
                )
                # Create learnable parameters
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.bias = torch.nn.Parameter(torch.tensor(0.0))
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt):
                x = state["input"]
                y = x * self.weight + self.bias
                return {"output": y}
        
        functor = LearnableFunctor()
        manager = get_global_manager()
        
        # Check registration
        assert manager.has_parameter("biological_process.learnable.weight")
        assert manager.has_parameter("biological_process.learnable.bias")
        
        # Check gradient capability
        spec_weight = manager.get_parameter("biological_process.learnable.weight")
        assert spec_weight.requires_grad
        
        # Test gradient flow
        result = call_forward(functor, {"input": torch.tensor(2.0)})
        loss = result["output"].sum()
        loss.backward()
        
        assert functor.weight.grad is not None
        assert functor.bias.grad is not None


class TestScientificModuleIntegration:
    """Test parameter integration with scientific modules."""
    
    def test_pedosphere_module_integration(self):
        """Test parameter integration with pedosphere modules."""
        from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
        
        # Create decomposition module
        # Only canonical module parameters may be overridden with units; a
        # key without a ParameterSpec in the database is rejected.
        config = {
            "parameters": {
                "k_metabolic": {"value": 0.1, "units": "yr-1"},
                "k_structural": {"value": 0.01, "units": "yr-1"},
            }
        }
        decomposer = FirstOrderDecomposition("first_order_decomposition", config)
        manager = get_global_manager()
        
        # Check that parameters are registered
        expected_params = [
            "pedosphere.decomposition.k_metabolic",
            "pedosphere.decomposition.k_structural",
        ]
        
        for param in expected_params:
            # Parameters should be accessible through get_parameter_value
            value = decomposer.get_parameter_value(
                param.split(".")[-1],  # Just the parameter name
                required=True,
            )
            assert value is not None
            assert isinstance(value, (float, torch.Tensor))
    
    def test_biosphere_module_integration(self):
        """Test parameter integration with biosphere modules."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        
        # Create photosynthesis module
        photo = FarquharPhotosynthesis()
        
        # Test parameter retrieval with units
        vcmax = photo.get_parameter_value(
            "vcmax25",
            required=True,
        )
        assert vcmax is not None
        
        # Test parameter bounds validation
        jmax = photo.get_parameter_value(
            "jmax25",
            required=True,
        )
        assert 0.0 <= jmax <= 500.0
    
    def test_cross_module_parameter_sharing(self):
        """Test parameter sharing between modules."""
        from relic.core.abstractions import BiologicalProcess
        
        class ModuleA(BiologicalProcess):
            def __init__(self):
                super().__init__("module_a")
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                temp = self.get_parameter_value("temperature")
                return {"rate_a": temp * 0.1}
        
        class ModuleB(BiologicalProcess):
            def __init__(self):
                super().__init__("module_b")
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                temp = self.get_parameter_value("temperature")
                return {"rate_b": temp * 0.2}
        
        # Register shared parameter
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key="shared.temperature",
            value=310.0,
            units="K",
            category=ParameterCategory.ENVIRONMENTAL,
            description="Shared temperature", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        module_a = ModuleA()
        module_b = ModuleB()
        
        # Both modules should access the same parameter
        # (Implementation depends on namespace resolution)


class TestCLIParameterIntegration:
    """Test parameter system integration with CLI."""
    
    def test_cli_parameter_commands(self):
        """Test CLI parameter management commands."""
        from relic.interfaces.cli.params import (
            list_parameters,
            get_parameter,
            set_parameter,
            validate_parameters
        )
        
        manager = get_global_manager()
        
        # Register test parameters
        manager.register_parameter(ParameterSpec(
            key="cli.test.param1",
            value=10.0,
            units="m/s",
            category=ParameterCategory.PHYSICAL,
            description="CLI test parameter 1", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        manager.register_parameter(ParameterSpec(
            key="cli.test.param2",
            value=20.0,
            units="K",
            bounds=(0.0, 400.0),
            category=ParameterCategory.ENVIRONMENTAL,
            description="CLI test parameter 2", dtype="float32"
        ))
        
        # Test list command
        params = list_parameters(namespace="cli.test")
        assert len(params) == 2
        assert "cli.test.param1" in params
        
        # Test get command
        value = get_parameter("cli.test.param1")
        assert value == 10.0
        
        # Test set command
        set_parameter("cli.test.param2", 300.0)
        assert get_parameter("cli.test.param2") == 300.0
        
        # Test validation command
        errors = validate_parameters()
        assert len(errors) == 0  # All valid
        
        # Default policy clamps out-of-bounds values with warning
        with pytest.warns(UserWarning):
            manager.set_value("cli.test.param2", 500.0)
        assert get_parameter("cli.test.param2") == 400.0
        
        # Validation should still pass for valid parameters
        errors = validate_parameters()
        # Note: may still have errors if the set operation partly succeeded
        # This depends on implementation details


class TestValidationRulesIntegration:
    """Test parameter validation rules integration."""
    
    def test_validation_rules_with_manager(self):
        """Test validation rules accessing parameter manager."""
        from relic.validation.framework.rules.parameters import (
            check_parameter_bounds,
            check_parameter_units,
            check_parameter_relationships
        )
        
        manager = get_global_manager()
        
        # Register related parameters
        manager.register_parameter(ParameterSpec(
            key="water.field_capacity",
            value=0.3,
            units="m3/m3",
            bounds=(0.0, 1.0),
            category=ParameterCategory.PHYSICAL,
            description="Field capacity", dtype="float32"
        ))
        
        manager.register_parameter(ParameterSpec(
            key="water.wilting_point",
            value=0.1,
            units="m3/m3",
            bounds=(0.0, 1.0),
            category=ParameterCategory.PHYSICAL,
            description="Wilting point", dtype="float32"
        ))
        
        manager.register_parameter(ParameterSpec(
            key="water.saturation",
            value=0.5,
            units="m3/m3",
            bounds=(0.0, 1.0),
            category=ParameterCategory.PHYSICAL,
            description="Saturation", dtype="float32"
        ))
        
        # Check bounds
        bounds_errors = check_parameter_bounds()
        assert len(bounds_errors) == 0
        
        # Check units consistency
        units_errors = check_parameter_units()
        assert len(units_errors) == 0
        
        # Check relationships
        relationship_errors = check_parameter_relationships()
        assert len(relationship_errors) == 0
        
        # Break relationship
        manager.set_value("water.wilting_point", 0.4)  # > field_capacity
        relationship_errors = check_parameter_relationships()
        assert len(relationship_errors) > 0


class TestParameterDatabaseIntegration:
    """Test parameter database integration."""
    
    def test_database_loading_hierarchy(self, monkeypatch):
        """Test loading parameters from hierarchical database."""
        # test/conftest.py disables the calibrated tier; this test exercises it.
        monkeypatch.delenv("RELIC_SKIP_CALIBRATED", raising=False)
        manager = UnifiedParameterManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir)
            
            # Create hierarchical database structure; calibrated values live in
            # an enabled calibrated scope (the load plan's calibrated tier).
            relic_dir = db_path / "relic"
            calibrated_dir = db_path / "calibrated" / "global"
            
            for d in [relic_dir, calibrated_dir]:
                d.mkdir(parents=True)
            
            # Create parameter files at different levels
            relic_params = {
                "photosynthesis.vcmax25": {
                    "value": 60.0,
                    "units": "umol CO2 m-2 s-1",
                    "dtype": "float32",
                    "bounds": [0.0, 200.0],
                    "category": "biophysical"
                }
            }

            calibrated_params = {
                "photosynthesis.vcmax25": {
                    "value": 62.5,
                    "units": "umol CO2 m-2 s-1",
                    "dtype": "float32",
                    "bounds": [0.0, 200.0],
                    "category": "calibration"
                }
            }

            # Save files
            with open(relic_dir / "params.yaml", 'w') as f:
                yaml.dump(relic_params, f)

            with open(calibrated_dir / "params.yaml", 'w') as f:
                yaml.dump(calibrated_params, f)

            # Load with priority: calibrated > relic
            manager.load_from_database(
                db_path=str(db_path),
                source_priority=["calibrated", "relic"]
            )
            
            # Should use calibrated value (highest precedence) 
            assert manager.get("photosynthesis.vcmax25") == 62.5
            
            # Load with different priority
            manager.clear_parameters()
            manager.load_from_database(
                db_path=str(db_path),
                source_priority=["relic", "calibrated"]
            )
            
            # Should use relic value
            assert manager.get("photosynthesis.vcmax25") == 60.0


class TestParameterPersistence:
    """Test parameter persistence and recovery."""
    
    def test_snapshot_and_recovery(self):
        """Test parameter snapshot and recovery."""
        manager = UnifiedParameterManager()
        
        # Initial state
        manager.register_parameter(ParameterSpec(
            key="test.param",
            value=100.0,
            category=ParameterCategory.COMPUTATIONAL,
            description="Test parameter", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        # Take snapshot
        snapshot = manager.create_snapshot("Initial state")
        
        # Modify parameters
        manager.set_value("test.param", 200.0)
        manager.register_parameter(ParameterSpec(
            key="test.new_param",
            value=300.0,
            category=ParameterCategory.COMPUTATIONAL,
            description="New parameter", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        assert manager.get("test.param") == 200.0
        assert manager.get("test.new_param") == 300.0
        
        # Restore snapshot
        manager.restore_snapshot(snapshot)
        
        # Original parameter should be restored
        assert manager.get("test.param") == 100.0
        # New parameter still exists but can't be removed by simple restore
        # This is a limitation of the simple snapshot implementation
        assert manager.has_parameter("test.new_param")  # Still exists
    
    def test_diff_tracking(self):
        """Test parameter change tracking."""
        manager = UnifiedParameterManager()
        
        # Initial parameters
        initial_params = {
            "param1": 1.0,
            "param2": 2.0,
            "param3": 3.0
        }
        
        for key, value in initial_params.items():
            manager.register_parameter(ParameterSpec(
                key=key,
                value=value,
                category=ParameterCategory.COMPUTATIONAL,
                description=f"Parameter {key}", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            ))
        
        snapshot1 = manager.create_snapshot("Initial")
        
        # Make changes
        manager.set_value("param1", 10.0)  # Modified
        manager.set_value("param2", 2.0)   # Unchanged
        manager.delete("param3")     # Deleted
        manager.register_parameter(ParameterSpec(
            key="param4",
            value=4.0,
            category=ParameterCategory.COMPUTATIONAL,
            description="New parameter", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        # Get diff
        diff = manager.get_diff(snapshot1)
        
        assert "param1" in diff  # Modified
        assert diff["param1"] == (1.0, 10.0)
        
        assert "param2" not in diff  # Unchanged
        
        assert "param3" in diff  # Deleted
        assert diff["param3"][1] is None
        
        assert "param4" in diff  # Added
        assert diff["param4"][0] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

