"""
L3 System Tests for Parameter Management

End-to-end tests for the complete parameter management system in realistic scenarios.
"""

import pytest
import torch
import tempfile
import yaml
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from relic.parameters import (
    ParameterSpec,
    ParameterCategory,
    ValidationLevel,
    BoundsPolicy,
    get_global_manager
)
from relic.parameters.types import ValidationResult, ValidationContext
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.scientific.atmosphere.environmental_forcing import (
    EnvironmentalForcingFunctor,
)
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET


class TestCompleteEcosystemSimulation:
    """Test parameter management in complete ecosystem simulation."""
    
    @pytest.fixture(autouse=True)
    def reset_system(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        yield
    
    @pytest.fixture
    def ecosystem_setup(self):
        """Set up a complete ecosystem simulation."""
        # Initialize components with config
        config = {
            "device": "cpu",
            "time_step": 3600.0,  # 1 hour in seconds
            "max_time": 86400.0,  # 24 hours in seconds
            "output_frequency": 1,
            "checkpoint_frequency": 3600.0,
            # This test suite validates parameter plumbing rather than strict
            # conservation closure (the mock ecosystem is an open system).
            "conservation_checks": False,
            "enable_auto_scale": False,
            "strict_scale_detection": False,
        }
        simulator = RELICSimulator(config)
        state_manager = StateManager(conservation_checks={"enabled": False})
        state_manager.initialize()  # Ensure proper initialization
        param_manager = get_global_manager()
        
        # Initialize state
        initial_state = {
            "atmospheric_co2_ppm": torch.tensor(400.0),
            "vcmax_scaling": torch.tensor(1.0),
            "vpmax_scaling": torch.tensor(1.0),
            "jmax_scaling": torch.tensor(1.0),
            "water_stress_factor": torch.tensor(1.0),
            "temperature_stress_factor": torch.tensor(1.0),
            "ozone_stress_factor": torch.tensor(1.0),
            "leaf_area_index": torch.tensor(3.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.6),
            "incoming_shortwave_potential": torch.tensor(700.0),
            "cos_zenith": torch.tensor(0.8),
            "surface_temperature": torch.tensor(296.15),
            "surface_albedo": torch.tensor(0.2),
            "surface_emissivity": torch.tensor(0.96),
            "cloud_fraction": torch.tensor(0.2),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(100.0),
            "metabolic_litter_c": torch.tensor(1.0),
            "structural_litter_c": torch.tensor(1.0),
            "fast_som_c": torch.tensor(1.0),
            "slow_som_c": torch.tensor(1.0),
            "passive_som_c": torch.tensor(1.0),
            "soil_temperature": torch.tensor(296.15),
            "soil_moisture": torch.tensor(0.25),
            "metabolic_litter_input": torch.tensor(0.0),
            "structural_litter_input": torch.tensor(0.0),
            "lignin_n_ratio": torch.tensor(0.3),
            "clay_content": torch.tensor(0.2),
        }
        
        # Set initial state using set_state method
        state_manager.set_state(initial_state, validate=False)
        
        # Register ecosystem parameters
        ecosystem_params = [
            # Photosynthesis parameters
            ParameterSpec(
                key="biosphere.photosynthesis.vcmax25",
                value=60.0,
                units="umol CO2 m-2 s-1",
                bounds=(10.0, 200.0),
                category=ParameterCategory.BIOPHYSICAL,
                description="Maximum carboxylation rate at 25C", dtype="float32"
            ),
            ParameterSpec(
                key="biosphere.photosynthesis.jmax25",
                value=120.0,
                units="umol e- m-2 s-1",
                bounds=(20.0, 400.0),
                category=ParameterCategory.BIOPHYSICAL,
                description="Maximum electron transport rate at 25C", dtype="float32"
            ),
            # Evapotranspiration parameters
            ParameterSpec(
                key="hydrosphere.et.stomatal_resistance",
                value=100.0,
                units="s m-1",
                bounds=(50.0, 500.0),
                category=ParameterCategory.BIOPHYSICAL,
                description="Minimum stomatal resistance", dtype="float32"
            ),
        ]
        
        for spec in ecosystem_params:
            param_manager.register_parameter(spec)
        
        # Create canonical functors and their dependencies.
        env_forcing = EnvironmentalForcingFunctor()
        radiation_dependency = RadiationFunctor()
        
        # Main functors
        photosynthesis = FarquharPhotosynthesis()
        decomposition = FirstOrderDecomposition(
            name="first_order_decomposition",
            config={},
        )
        evapotranspiration = PenmanMonteithET(
            name="penman_monteith_et",
            config={"canopy_resistance_source": "state"},
        )

        # Initialize simulator with required components
        from relic.core.dependency_graph import DependencyGraphResolver
        dependency_graph = DependencyGraphResolver()
        simulator.initialize(state_manager, dependency_graph)
        
        # Register canonical dependencies first.
        simulator.register_functor(env_forcing)
        simulator.register_functor(radiation_dependency)
        
        # Register main functors
        simulator.register_functor(photosynthesis)
        simulator.register_functor(decomposition)
        simulator.register_functor(evapotranspiration)
        
        # Start simulation (set to RUNNING state)
        from relic.core.simulator import SimulationState
        simulator.current_time = 0.0
        simulator.state = SimulationState.RUNNING
        
        return {
            "simulator": simulator,
            "state_manager": state_manager,
            "param_manager": param_manager,
            "functors": {
                "photosynthesis": photosynthesis,
                "decomposition": decomposition,
                "evapotranspiration": evapotranspiration
            }
        }
    
    def test_ecosystem_simulation_with_parameters(self, ecosystem_setup):
        """Test complete ecosystem simulation with parameter management."""
        simulator = ecosystem_setup["simulator"]
        state_manager = ecosystem_setup["state_manager"]
        param_manager = ecosystem_setup["param_manager"]
        
        # Run 10 hourly steps: PenmanMonteithET rejects steps longer than
        # hydrosphere.physical.soil_heat_flux_hourly_max_step (FAO-56 hourly G, 3600 s).
        dt = 3600.0
        results = []

        for day in range(10):
            # Update environment - set environmental variables directly
            current_state = state_manager.get_state()
            
            # Set environmental variables (not accumulated)
            current_state["air_temperature"] = torch.tensor(
                298.15 + 2 * np.sin(day * np.pi / 5)
            )
            current_state["atmospheric_co2_ppm"] = torch.tensor(400.0)
            current_state["vcmax_scaling"] = torch.tensor(1.0)
            current_state["relative_humidity"] = torch.tensor(0.6)
            current_state["wind_speed"] = torch.tensor(2.0)
            current_state["atmospheric_pressure"] = torch.tensor(101325.0)
            current_state["surface_resistance"] = torch.tensor(100.0)
            
            # Set soil conditions
            current_state["soil_temperature"] = torch.clamp(
                current_state["air_temperature"] - 2.0,
                min=235.0,
                max=330.0,
            )
            
            state_manager.set_state(current_state, validate=False)
            
            # Step simulation
            fluxes = simulator.step(dt=dt)
            
            # Record results
            gpp_tensor = fluxes["gross_primary_productivity"]
            respiration_tensor = fluxes["heterotrophic_respiration"]
            et_tensor = fluxes["evapotranspiration_flux"]
            assert isinstance(gpp_tensor, torch.Tensor)
            assert isinstance(respiration_tensor, torch.Tensor)
            assert isinstance(et_tensor, torch.Tensor)
            assert torch.isfinite(gpp_tensor).all()
            assert torch.isfinite(respiration_tensor).all()
            assert torch.isfinite(et_tensor).all()
            gpp_val = float(gpp_tensor.detach().cpu().item())
            respiration_val = float(respiration_tensor.detach().cpu().item())
            et_val = float(et_tensor.detach().cpu().item())
            results.append({
                "day": day,
                "gpp": gpp_val,
                "respiration": respiration_val,
                "et": et_val,
            })
        
        # Verify simulation ran
        assert len(results) == 10
        
        # Confirm simulation completed without failure state
        from relic.core.simulator import SimulationState
        assert simulator.state != SimulationState.FAILED
        
        # Validate parameter influence
        vcmax = param_manager.get("biosphere.photosynthesis.vcmax25")
        assert vcmax == 60.0
        
        evaluation_state = state_manager.get_state()
        baseline_model = ecosystem_setup["functors"]["photosynthesis"]
        baseline_item = baseline_model.forward(
            evaluation_state,
            time=simulator.current_time,
            dt=dt,
        )["gross_primary_productivity"]
        assert isinstance(baseline_item, UpdateItem)

        # Manager updates take effect when the configured model is rebuilt.
        param_manager.set_value("biosphere.photosynthesis.vcmax25", 80.0)
        param_manager.set_value("biosphere.photosynthesis.jmax25", 160.0)
        updated_model = FarquharPhotosynthesis()
        updated_item = updated_model.forward(
            evaluation_state,
            time=simulator.current_time,
            dt=dt,
        )["gross_primary_productivity"]
        assert isinstance(updated_item, UpdateItem)
        assert not torch.allclose(
            updated_item.value,
            baseline_item.value,
            rtol=1.0e-8,
            atol=1.0e-15,
        )
    
    def test_parameter_optimization_workflow(self, ecosystem_setup):
        """Test parameter optimization workflow."""
        param_manager = ecosystem_setup["param_manager"]
        simulator = ecosystem_setup["simulator"]
        state_manager = ecosystem_setup["state_manager"]
        vcmax_key = "biosphere.photosynthesis.vcmax25"
        jmax_key = "biosphere.photosynthesis.jmax25"
        baseline = (60.0, 120.0)
        target_candidate = (80.0, 160.0)
        simulator.step(dt=3600.0)
        evaluation_state = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in state_manager.get_state().items()
        }

        def evaluate(candidate):
            """Build one model from manager values and return canonical GPP."""
            param_manager.set_value(vcmax_key, candidate[0])
            param_manager.set_value(jmax_key, candidate[1])
            model = FarquharPhotosynthesis()
            outputs = model.forward(evaluation_state, time=0.0, dt=3600.0)
            item = outputs["gross_primary_productivity"]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == "kg C m-2 s-1"
            assert torch.isfinite(item.value).all()
            return item.value.detach().clone()

        baseline_gpp = evaluate(baseline)
        target_gpp = evaluate(target_candidate)
        assert not torch.allclose(
            target_gpp,
            baseline_gpp,
            rtol=1.0e-8,
            atol=1.0e-15,
        )

        candidates = (
            baseline,
            (80.0, 120.0),
            (60.0, 160.0),
            target_candidate,
            (50.0, 100.0),
        )
        evaluated = []
        for candidate in candidates:
            gpp = evaluate(candidate)
            loss = float(torch.square(gpp - target_gpp).detach().cpu().item())
            evaluated.append((loss, candidate, gpp))

        baseline_loss = next(loss for loss, candidate, _ in evaluated if candidate == baseline)
        best_loss, best_candidate, best_gpp = min(evaluated, key=lambda item: item[0])
        assert best_loss < baseline_loss
        assert best_candidate != baseline

        param_manager.set_value(vcmax_key, best_candidate[0])
        param_manager.set_value(jmax_key, best_candidate[1])
        assert param_manager.get(vcmax_key) == best_candidate[0]
        assert param_manager.get(jmax_key) == best_candidate[1]
        rebuilt_gpp = evaluate(best_candidate)
        torch.testing.assert_close(rebuilt_gpp, best_gpp, rtol=0.0, atol=0.0)
    
    def test_parameter_validation_in_simulation(self, ecosystem_setup):
        """Test parameter validation during simulation."""
        param_manager = ecosystem_setup["param_manager"]

        # Set strict validation
        param_manager.validation_level = ValidationLevel.STRICT
        param_manager.set_bounds_policy(BoundsPolicy.ERROR)
        
        # Validate all parameters
        errors = param_manager.validate_parameters()
        target_keys = {
            "biosphere.photosynthesis.vcmax25",
            "biosphere.photosynthesis.jmax25",
        }
        def _contains_target(err):
            if isinstance(err, dict):
                return err.get("parameter") in target_keys
            return any(k in str(err) for k in target_keys)
        assert not any(_contains_target(err) for err in errors)
        
        # Introduce invalid parameter (should raise under strict bounds policy)
        with pytest.raises(ValueError):
            param_manager.set_value("biosphere.photosynthesis.vcmax25", -10.0)  # Negative rate

        # Validation should remain clean because invalid set was rejected
        errors = param_manager.validate_parameters()
        assert not any(_contains_target(err) for err in errors)
        
        # Test physical consistency
        param_manager.set_value("biosphere.photosynthesis.vcmax25", 60.0)  # Fix
        param_manager.set_value("biosphere.photosynthesis.jmax25", 50.0)  # jmax < vcmax (unusual)
        
        # Add custom validator for photosynthesis consistency
        def photosynthesis_validator(key, value, spec):
            if key != "biosphere.photosynthesis.jmax25":
                return ValidationResult(is_valid=True, errors=[])
            vcmax = param_manager.get("biosphere.photosynthesis.vcmax25")
            if value is None or vcmax is None:
                return ValidationResult(is_valid=True, errors=[])
            try:
                value_float = float(value)
                vcmax_float = float(vcmax)
            except (TypeError, ValueError):
                return ValidationResult(is_valid=True, errors=[])
            if value_float < vcmax_float:
                return ValidationResult(
                    is_valid=True,
                    errors=[],
                    warnings=["Warning: Jmax is typically > Vcmax"],
                )
            return ValidationResult(is_valid=True, errors=[])
        
        param_manager.validator.add_custom_validator(
            "photosynthesis_consistency",
            photosynthesis_validator
        )
        
        context = ValidationContext(
            all_parameters=param_manager.storage.get_all(),
            strict_mode=(param_manager.validation_level == ValidationLevel.STRICT),
        )
        jmax_spec = param_manager.get_parameter("biosphere.photosynthesis.jmax25")
        result = param_manager.validator.validate_value(
            "biosphere.photosynthesis.jmax25",
            jmax_spec.value,
            jmax_spec,
            param_manager.validation_level,
            context,
        )
        assert any("Jmax" in w for w in result.warnings)


class TestParameterPersistenceAndRecovery:
    """Test parameter persistence and recovery scenarios."""
    
    @pytest.fixture(autouse=True)
    def reset_system(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        yield
    
    def test_save_and_load_simulation_state(self):
        """Test saving and loading complete simulation state."""
        param_manager = get_global_manager()
        
        # Set up parameters
        params = {
            "model.param1": 1.0,
            "model.param2": 2.0,
            "model.param3": 3.0
        }
        
        for key, value in params.items():
            param_manager.register_parameter(ParameterSpec(
                key=key,
                value=value,
                category=ParameterCategory.COMPUTATIONAL,
                description=f"Parameter {key}", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            ))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "simulation_state.yaml"
            
            # Save state
            param_manager.save_to_database(str(save_path))
            
            # Clear and reload
            param_manager.clear_parameters()
            assert len(param_manager.get_all_parameters()) == 0
            
            param_manager.load_from_database(str(save_path))
            
            # Verify restoration
            for key, value in params.items():
                assert param_manager.get(key) == value
    
    def test_multi_format_persistence(self):
        """Test persistence in multiple formats."""
        param_manager = get_global_manager()
        
        # Register diverse parameters
        param_manager.register_parameter(ParameterSpec(
            key="test.float",
            value=3.14159,
            category=ParameterCategory.COMPUTATIONAL,
            description="Float parameter", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        param_manager.register_parameter(ParameterSpec(
            key="test.tensor",
            value=torch.tensor([1.0, 2.0, 3.0]),
            category=ParameterCategory.COMPUTATIONAL,
            description="Tensor parameter", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        param_manager.register_parameter(ParameterSpec(
            key="test.boolean",
            value=True,
            category=ParameterCategory.COMPUTATIONAL,
            description="Boolean parameter", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        ))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test each format
            for format_name in ["yaml", "json"]:
                file_path = Path(tmpdir) / f"params.{format_name}"
                
                # Export
                param_manager.export_parameters(str(file_path), format=format_name)
                assert file_path.exists()
                
                # Clear and reload
                param_manager.clear_parameters()
                
                # Load based on format
                if format_name == "yaml":
                    with open(file_path) as f:
                        data = yaml.safe_load(f)
                else:
                    with open(file_path) as f:
                        data = json.load(f)
                
                # Re-register from loaded data
                # Handle the new format with version and timestamp
                parameters_data = data.get("parameters", data)
                
                for key, param_data in parameters_data.items():
                    # Handle both string (direct value) and dict (ParameterSpec) formats
                    if isinstance(param_data, dict):
                        # ParameterSpec format
                        if "value" in param_data:
                            value_data = param_data["value"]
                            if isinstance(value_data, dict) and "data" in value_data:
                                # Tensor was serialized
                                value = torch.tensor(value_data["data"])
                            else:
                                value = value_data
                        else:
                            # Direct dict is the value
                            value = param_data
                        
                        param_manager.register_parameter(ParameterSpec(
                            key=key,
                            value=value,
                            category=ParameterCategory.COMPUTATIONAL,
                            description=param_data.get("description", ""), units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
                        ))
                    else:
                        # Direct value
                        param_manager.register_parameter(ParameterSpec(
                            key=key,
                            value=param_data,
                            category=ParameterCategory.COMPUTATIONAL,
                            description="", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
                        ))
                
                # Verify
                assert abs(param_manager.get("test.float") - 3.14159) < 1e-5
                assert param_manager.get("test.boolean") == True


class TestConcurrentParameterAccess:
    """Test concurrent parameter access and modification."""
    
    @pytest.fixture(autouse=True)
    def reset_system(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        yield
    
    def test_thread_safe_parameter_access(self):
        """Test thread-safe concurrent parameter access."""
        param_manager = get_global_manager()
        
        # Register initial parameters
        n_params = 100
        for i in range(n_params):
            param_manager.register_parameter(ParameterSpec(
                key=f"concurrent.param{i}",
                value=float(i),
                category=ParameterCategory.COMPUTATIONAL,
                description=f"Concurrent parameter {i}", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            ))
        
        errors = []
        results = {}
        
        def reader_thread(param_id):
            """Read parameters repeatedly."""
            try:
                local_results = []
                for _ in range(50):
                    value = param_manager.get(f"concurrent.param{param_id}")
                    local_results.append(value)
                    time.sleep(0.001)  # Small delay
                results[f"reader_{param_id}"] = local_results
            except Exception as e:
                errors.append(e)
        
        def writer_thread(param_id):
            """Write parameters repeatedly."""
            try:
                for iteration in range(50):
                    new_value = param_id * 1000 + iteration
                    param_manager.set_value(f"concurrent.param{param_id}", new_value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Launch concurrent threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            
            # Start readers
            for i in range(10):
                futures.append(executor.submit(reader_thread, i))
            
            # Start writers
            for i in range(10):
                futures.append(executor.submit(writer_thread, i))
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        # Check no errors occurred
        assert len(errors) == 0
        
        # Verify results are consistent
        for key, values in results.items():
            # All read values should be valid numbers
            assert all(isinstance(v, (int, float)) for v in values)
    
    def test_snapshot_under_load(self):
        """Test snapshot creation under concurrent load."""
        param_manager = get_global_manager()
        baseline_count = len(param_manager.parameters)

        # Register parameters
        for i in range(50):
            param_manager.register_parameter(ParameterSpec(
                key=f"load.param{i}",
                value=float(i),
                category=ParameterCategory.COMPUTATIONAL,
                description=f"Load parameter {i}", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            ))
        
        snapshots = []
        errors = []
        
        def snapshot_thread(thread_id):
            """Create snapshots repeatedly."""
            try:
                for i in range(10):
                    snapshot = param_manager.create_snapshot(f"Thread {thread_id} snapshot {i}")
                    snapshots.append(snapshot)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        def modify_thread(thread_id):
            """Modify parameters while snapshots are being taken."""
            try:
                for i in range(50):
                    key = f"load.param{i % 50}"
                    param_manager.set_value(key, thread_id * 100 + i)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent snapshot and modification
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for i in range(3):
                futures.append(executor.submit(snapshot_thread, i))
            
            for i in range(3):
                futures.append(executor.submit(modify_thread, i))
            
            for future in as_completed(futures):
                future.result()
        
        # Verify no errors
        assert len(errors) == 0
        
        # Verify snapshots are valid
        assert len(snapshots) > 0
        expected_min = baseline_count + 50
        for snapshot in snapshots:
            assert snapshot is not None
            assert len(snapshot.parameters) >= expected_min


class TestParameterMigration:
    """Test parameter migration and consistency."""
    
    @pytest.fixture(autouse=True)
    def reset_system(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        yield
    
    def test_parameter_migration(self):
        """Test migration from source parameter formats."""
        param_manager = get_global_manager()
        
        # Simulate source parameter format
        source_params = {
            "old_vcmax": 55.0,
            "old_jmax": 110.0,
            "decomp_rate_1": 0.5,
            "decomp_rate_2": 0.05
        }
        
        # Define migration mapping
        migration_map = {
            "old_vcmax": ("biosphere.photosynthesis.vcmax25", 1.0),
            "old_jmax": ("biosphere.photosynthesis.jmax25", 1.0),
            "decomp_rate_1": ("pedosphere.decomposition.k_fast_som", 1.0),
            "decomp_rate_2": ("pedosphere.decomposition.k_slow_som", 1.0)
        }
        
        # Migrate parameters
        for old_key, old_value in source_params.items():
            if old_key in migration_map:
                new_key, conversion_factor = migration_map[old_key]
                new_value = old_value * conversion_factor
                
                param_manager.register_parameter(ParameterSpec(
                    key=new_key,
                    value=new_value,
                    category=ParameterCategory.MODEL_PARAMETER,
                    description=f"Migrated from {old_key}",
                    provenance={
                        "source": "migration",
                        "original_key": old_key,
                        "conversion_factor": conversion_factor
                    }, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
                ))
        
        # Verify migration
        assert param_manager.get("biosphere.photosynthesis.vcmax25") == 55.0
        assert param_manager.get("biosphere.photosynthesis.jmax25") == 110.0
        assert param_manager.get("pedosphere.decomposition.k_fast_som") == 0.5
        assert param_manager.get("pedosphere.decomposition.k_slow_som") == 0.05
        
        # Check provenance
        vcmax_spec = param_manager.get_parameter("biosphere.photosynthesis.vcmax25")
        assert vcmax_spec.provenance["original_key"] == "old_vcmax"
    
    def test_version_consistency(self):
        """Test parameter consistency across versions."""
        param_manager = get_global_manager()
        
        # Version 1 parameters
        v1_params = {
            "model.param_a": 1.0,
            "model.param_b": 2.0
        }
        
        # Version 2 parameters (renamed and restructured)
        v2_mapping = {
            "model.param_a": "model.component.param_alpha",
            "model.param_b": "model.component.param_beta"
        }
        
        # Register v1 parameters
        for key, value in v1_params.items():
            param_manager.register_parameter(ParameterSpec(
                key=key,
                value=value,
                category=ParameterCategory.COMPUTATIONAL,
                description=f"Version 1: {key}", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            ))
        
        # Create consistency layer
        class consistencyManager:
            def __init__(self, param_manager, mapping):
                self.param_manager = param_manager
                self.mapping = mapping
            
            def get_consistent(self, key):
                """Get parameter with version consistency."""
                # Try direct access
                if self.param_manager.has_parameter(key):
                    return self.param_manager.get(key)
                
                # Try mapped access
                if key in self.mapping:
                    new_key = self.mapping[key]
                    if self.param_manager.has_parameter(new_key):
                        return self.param_manager.get(new_key)
                
                # Try reverse mapping
                for old_key, new_key in self.mapping.items():
                    if new_key == key and self.param_manager.has_parameter(old_key):
                        return self.param_manager.get(old_key)
                
                return None
        
        bridge_mgr = consistencyManager(param_manager, v2_mapping)
        
        # Access v1 parameters with v1 names
        assert bridge_mgr.get_consistent("model.param_a") == 1.0
        
        # Access v1 parameters with v2 names (through mapping)
        assert bridge_mgr.get_consistent("model.component.param_alpha") == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])



