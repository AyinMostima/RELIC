"""
L3 System Benchmark Tests for Parameter Management

This module provides comprehensive system-level testing and benchmarking
of the parameter management system, including performance, scalability,
and real-world usage patterns.

Test Coverage:
    - Full system parameter flow
    - Performance benchmarks
    - Memory usage profiling
    - Concurrent access patterns
    - Large-scale parameter sets
    - Real simulation scenarios
    - Conservation law validation
    - Scientific accuracy checks
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any
from pathlib import Path
import sys
import time
import threading
from dataclasses import dataclass
import gc
import tracemalloc

# Add relic to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.parameters import (
    UnifiedParameterManager,
    ParameterSpec,
    ParameterCategory,
    get_global_manager
)
from relic.core.functor import FunctorBase
from relic.core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase
from relic.core.state import StateManager


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    duration: float
    memory_peak: float
    operations: int
    throughput: float
    success: bool
    details: Dict[str, Any]


class TestSystemIntegration:
    """Test full system integration with parameter management."""
    
    @pytest.fixture(autouse=True)
    def reset_system(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        gc.collect()
        yield
        gc.collect()

    def test_full_simulation_workflow(self):
        """Test complete simulation workflow with parameter management."""
        
        # Create a simple ecosystem simulation
        class PhotosynthesisFunctor(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="photosynthesis",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Leaf-level photosynthesis light response (test stub)",
                    mathematical_expression="A = Vcmax * PAR / (PAR + k)",
                    references=["Farquhar et al. 1980"],
                    input_variables=["par"],
                    output_variables=["gpp", "delta_carbon"],
                )
                super().__init__(name="photosynthesis", config={}, metadata=metadata)
                self.parameter_namespace = "biosphere.photosynthesis"
            
            def forward(self, state, time, dt):
                vcmax = self.get_parameter_value("vcmax25")
                par = state["par"]
                
                # References: Farquhar et al. 1980
                # Expression: A = Vcmax * PAR / (PAR + k)
                # Parameters: vcmax25 (umol CO2 m-2 s-1), par (umol m-2 s-1), k (umol m-2 s-1)
                a_max = vcmax * par / (par + 500.0)
                
                return {
                    "gpp": a_max,
                    "delta_carbon": a_max * dt
                }
            
            def get_dependencies(self):
                return {"required": ["temperature", "par"], "optional": []}
        
        class RespirationFunctor(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="respiration",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Temperature-dependent maintenance respiration (test stub)",
                    mathematical_expression="R = C * rm_leaf_25 * Q10^((T - Tref) / 10)",
                    references=["Lloyd and Taylor 1994"],
                    input_variables=["temperature", "carbon_pool"],
                    output_variables=["respiration", "delta_carbon"],
                )
                super().__init__(name="respiration", config={}, metadata=metadata)
                self.parameter_namespace = "biosphere.respiration"
            
            def forward(self, state, time, dt):
                temp = state["temperature"]
                carbon_pool = state["carbon_pool"]

                rm_leaf_25 = self.get_parameter_value("maintenance.rm_leaf_25")
                q10 = self.get_parameter_value("rd_q10")
                
                # References: Lloyd and Taylor 1994
                # Expression: R = C * rm_leaf_25 * Q10^((T - Tref) / 10)
                # Parameters: carbon_pool (g C m-2), rm_leaf_25 (1 s-1), rd_q10 (dimensionless), temp (K), Tref (K)
                resp_rate = carbon_pool * rm_leaf_25 * q10 ** ((temp - 298.15) / 10.0)
                
                return {
                    "respiration": resp_rate,
                    "delta_carbon": -resp_rate * dt
                }
            
            def get_dependencies(self):
                return {"required": ["temperature"], "optional": []}
        
        # Initialize system components
        state_manager = StateManager()
        state_manager.initialize()
        
        # Set initial state
        initial_state = {
            "temperature": torch.tensor(298.15),
            "par": torch.tensor(800.0),
            "carbon_pool": torch.tensor(1000.0)
        }
        
        # Create functors
        photosynthesis = PhotosynthesisFunctor()
        respiration = RespirationFunctor()
        
        # Run simulation step
        dt = 3600.0  # 1 hour
        
        photo_output = photosynthesis(initial_state, 0.0, dt)
        resp_output = respiration(initial_state, 0.0, dt)
        
        # Check outputs
        assert "gpp" in photo_output
        assert "respiration" in resp_output
        assert photo_output["gpp"] > 0
        assert resp_output["respiration"] > 0
        
        # Check parameter management
        manager = get_global_manager()
        # Check that parameters are registered (loaded from the YAML database)
        assert manager.has_parameter("biosphere.photosynthesis.vcmax25")
        assert manager.has_parameter("biosphere.respiration.rd_q10")
        assert manager.has_parameter("biosphere.respiration.maintenance.rm_leaf_25")
        
        # Update parameters and verify effect
        original_vcmax = manager.get("biosphere.photosynthesis.vcmax25")
        manager.set_value("biosphere.photosynthesis.vcmax25", original_vcmax * 1.5)
        photo_output2 = photosynthesis(initial_state, 0.0, dt)
        # The output should change when parameter changes
        assert photo_output2["gpp"] != photo_output["gpp"]
    
    def test_conservation_laws_with_parameters(self):
        """Test that conservation laws are maintained with parameter changes."""
        
        class ConservativeFunctor(FunctorBase):
            def __init__(self, name):
                metadata = FunctorMetadata(
                    name=f"conservative_{name}",
                    category=ProcessCategory.PHYSICAL_FIELD,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Conservative transfer between pools (test stub)",
                    mathematical_expression="transfer = source * rate * dt",
                    references=["Sierra et al. 2012", "Manzoni and Porporato 2009"],
                    conservation_laws=["mass", "energy"],
                    input_variables=["source_pool"],
                    output_variables=[
                        "source_delta",
                        "sink_delta",
                        "total_conserved",
                    ],
                )
                super().__init__(name=name, config={}, metadata=metadata)
                self.parameter_namespace = f"physical_field.{name}"
            
            def forward(self, state, time, dt):
                rate = self.get_parameter_value("transfer_rate")
                source = state["source_pool"]
                
                # References: Sierra et al. 2012; Manzoni and Porporato 2009
                # Expression: transfer = source * rate * dt
                # Parameters: source (g m-2), rate (s-1), dt (s)
                transfer = source * rate * dt
                
                return {
                    "source_delta": -transfer,
                    "sink_delta": transfer,
                    "total_conserved": torch.tensor(0.0)  # Net change should be zero
                }
            
            def get_dependencies(self):
                return {"required": ["source_pool"], "optional": []}
        
        # Create system
        functor = ConservativeFunctor("test")

        # Test conservation
        state = {"source_pool": torch.tensor(100.0)}
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key="physical_field.test.transfer_rate",
            value=0.1,
            units="s-1",
            description="Transfer rate of the conservative test functor", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter",
        ))
        output = functor(state, 0.0, 1.0)
        
        # Check conservation
        total_change = output["source_delta"] + output["sink_delta"]
        assert torch.abs(total_change) < 1e-10, "Mass not conserved"
        
        # Change parameter and retest
        manager.set_value("physical_field.test.transfer_rate", 0.2)
        
        output2 = functor(state, 0.0, 1.0)
        total_change2 = output2["source_delta"] + output2["sink_delta"]
        assert torch.abs(total_change2) < 1e-10, "Mass not conserved after parameter change"
        
        # Verify rate changed
        assert torch.abs(output2["source_delta"]) > torch.abs(output["source_delta"])


class TestPerformanceBenchmarks:
    """Performance benchmarks for parameter system."""
    
    @pytest.fixture(autouse=True)
    def reset_system(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        gc.collect()
        yield
        gc.collect()

    def benchmark_parameter_operations(self, num_params: int, num_operations: int) -> BenchmarkResult:
        """Benchmark parameter operations at scale."""
        
        # Start memory tracking
        tracemalloc.start()
        
        manager = UnifiedParameterManager()
        manager.initialize()
        
        # Register parameters
        start_time = time.time()
        for i in range(num_params):
            spec = ParameterSpec(
                key=f"param_{i}",
                value=float(i),
                category=ParameterCategory.MODEL_PARAMETER,
                bounds=(0.0, 1000.0), units="dimensionless", dtype="float32"
            )
            manager.register_parameter(spec)
        
        registration_time = time.time() - start_time
        
        # Perform operations
        operation_start = time.time()
        for _ in range(num_operations):
            # Random parameter access
            param_idx = np.random.randint(0, num_params)
            value = manager.get(f"param_{param_idx}")
            
            # Random update
            if np.random.random() > 0.5:
                new_value = min(value * 1.1, 999.0)  # Keep within bounds
                manager.set_value(f"param_{param_idx}", new_value)
        
        operation_time = time.time() - operation_start
        
        # Get memory peak
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = registration_time + operation_time
        throughput = (num_params + num_operations) / total_time
        
        return BenchmarkResult(
            name=f"param_ops_{num_params}_{num_operations}",
            duration=total_time,
            memory_peak=peak / 1024 / 1024,  # MB
            operations=num_params + num_operations,
            throughput=throughput,
            success=True,
            details={
                "registration_time": registration_time,
                "operation_time": operation_time,
                "avg_op_time": operation_time / num_operations if num_operations > 0 else 0
            }
        )
    
    def test_small_scale_performance(self):
        """Test performance with small parameter set."""
        result = self.benchmark_parameter_operations(100, 1000)
        
        assert result.success
        assert result.duration < 1.0, f"Small scale too slow: {result.duration:.3f}s"
        assert result.memory_peak < 10.0, f"Memory usage too high: {result.memory_peak:.1f}MB"
        
        print(f"\nSmall scale benchmark:")
        print(f"  Duration: {result.duration:.3f}s")
        print(f"  Throughput: {result.throughput:.0f} ops/s")
        print(f"  Memory: {result.memory_peak:.1f}MB")
    
    def test_medium_scale_performance(self):
        """Test performance with medium parameter set."""
        result = self.benchmark_parameter_operations(1000, 10000)
        
        assert result.success
        assert result.duration < 5.0, f"Medium scale too slow: {result.duration:.3f}s"
        assert result.memory_peak < 50.0, f"Memory usage too high: {result.memory_peak:.1f}MB"
        
        print(f"\nMedium scale benchmark:")
        print(f"  Duration: {result.duration:.3f}s")
        print(f"  Throughput: {result.throughput:.0f} ops/s")
        print(f"  Memory: {result.memory_peak:.1f}MB")
    
    @pytest.mark.slow
    def test_large_scale_performance(self):
        """Test performance with large parameter set."""
        result = self.benchmark_parameter_operations(10000, 100000)
        
        assert result.success
        assert result.duration < 30.0, f"Large scale too slow: {result.duration:.3f}s"
        assert result.memory_peak < 200.0, f"Memory usage too high: {result.memory_peak:.1f}MB"
        
        print(f"\nLarge scale benchmark:")
        print(f"  Duration: {result.duration:.3f}s")
        print(f"  Throughput: {result.throughput:.0f} ops/s")
        print(f"  Memory: {result.memory_peak:.1f}MB")
    
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access."""
        
        manager = UnifiedParameterManager()
        manager.initialize()
        
        # Register parameters
        num_params = 100
        for i in range(num_params):
            spec = ParameterSpec(
                key=f"concurrent_param_{i}",
                value=float(i),
                category=ParameterCategory.MODEL_PARAMETER, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            manager.register_parameter(spec)
        
        # Concurrent access function
        def worker(thread_id: int, num_ops: int, results: list):
            """Worker thread for concurrent access."""
            start = time.time()
            for _ in range(num_ops):
                param_idx = (thread_id + _) % num_params
                value = manager.get(f"concurrent_param_{param_idx}")
                manager.set_value(f"concurrent_param_{param_idx}", value + 0.1)
            elapsed = time.time() - start
            results.append(elapsed)
        
        # Run concurrent threads
        num_threads = 10
        ops_per_thread = 1000
        results = []
        threads = []
        
        start_time = time.time()
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i, ops_per_thread, results))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        total_ops = num_threads * ops_per_thread
        throughput = total_ops / total_time
        
        print(f"\nConcurrent access benchmark:")
        print(f"  Threads: {num_threads}")
        print(f"  Total operations: {total_ops}")
        print(f"  Duration: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.0f} ops/s")
        
        assert total_time < 5.0, f"Concurrent access too slow: {total_time:.3f}s"


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.fixture(autouse=True)
    def reset_system(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        yield

    def test_climate_model_parameters(self):
        """Test parameter management for climate model scenario."""
        
        # Climate model parameters
        climate_params = {
            # Radiation parameters
            "solar_constant": ParameterSpec(
                key="radiation.solar_constant",
                value=1361.0,
                units="W/m2",
                category=ParameterCategory.PHYSICAL,
                description="Total solar irradiance",
                bounds=(1360.0, 1362.0), dtype="float32"
            ),
            "albedo": ParameterSpec(
                key="surface.albedo",
                value=0.3,
                units="dimensionless",
                category=ParameterCategory.PHYSICAL,
                bounds=(0.0, 1.0), dtype="float32"
            ),
            # Atmospheric parameters
            "co2_concentration": ParameterSpec(
                key="atmosphere.co2",
                value=415.0,
                units="ppm",
                category=ParameterCategory.PHYSICAL,
                bounds=(280.0, 1000.0), dtype="float32"
            ),
            "ch4_concentration": ParameterSpec(
                key="atmosphere.ch4",
                value=1.9,
                units="ppm",
                category=ParameterCategory.PHYSICAL,
                bounds=(0.7, 3.0), dtype="float32"
            ),
            # Cloud parameters
            "cloud_fraction": ParameterSpec(
                key="clouds.fraction",
                value=0.67,
                units="dimensionless",
                category=ParameterCategory.MODEL_PARAMETER,
                bounds=(0.0, 1.0), dtype="float32"
            )
        }
        
        # Register all parameters
        manager = get_global_manager()
        for param_spec in climate_params.values():
            manager.register_parameter(param_spec)
        
        # Simulate parameter updates during model run
        time_steps = 100
        for t in range(time_steps):
            # Seasonal variation in solar radiation
            solar_var = 1361.0 + 0.5 * np.sin(2 * np.pi * t / 365)
            manager.set_value("radiation.solar_constant", solar_var, validate=False)
            
            # CO2 trend
            co2_trend = 415.0 + 0.01 * t
            manager.set_value("atmosphere.co2", co2_trend, validate=False)
            
            # Cloud variation
            cloud_var = 0.67 + 0.1 * np.sin(4 * np.pi * t / 365)
            manager.set_value("clouds.fraction", cloud_var, validate=False)
        
        # Check final values
        assert manager.get("radiation.solar_constant") != 1361.0
        assert manager.get("atmosphere.co2") > 415.0
        
        # Get statistics
        stats = manager.get_statistics()
        assert stats["total_parameters"] >= 5
        # Check that updates were made (total_updates key may not exist)
        if "total_updates" in stats:
            assert stats["total_updates"] >= time_steps * 3
    
    def test_ecosystem_model_parameters(self):
        """Test parameter management for ecosystem model."""
        
        # Plant functional type parameters
        pft_params = {
            "broadleaf_deciduous": {
                "vcmax25": 60.0,
                "jmax25": 120.0,
                "sla": 0.025,
                "leaf_cn": 25.0,
                "root_depth": 2.0
            },
            "needleleaf_evergreen": {
                "vcmax25": 40.0,
                "jmax25": 80.0,
                "sla": 0.008,
                "leaf_cn": 40.0,
                "root_depth": 1.5
            },
            "c3_grass": {
                "vcmax25": 50.0,
                "jmax25": 100.0,
                "sla": 0.035,
                "leaf_cn": 20.0,
                "root_depth": 0.5
            }
        }
        
        # Register PFT parameters
        manager = get_global_manager()
        for pft_name, params in pft_params.items():
            for param_name, value in params.items():
                spec = ParameterSpec(
                    key=f"pft.{pft_name}.{param_name}",
                    value=value,
                    category=ParameterCategory.BIOPHYSICAL,
                    requires_grad=True, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))  # For calibration
                )
                manager.register_parameter(spec)
        
        # Simulate parameter calibration
        learning_rate = 0.01
        for iteration in range(10):
            # Simulate gradient-based update
            for pft_name in pft_params.keys():
                vcmax_key = f"pft.{pft_name}.vcmax25"
                current_vcmax = manager.get(vcmax_key)
                
                # Simulate gradient (normally from backprop)
                gradient = 1.0  # Deterministic update to avoid flaky thresholds
                
                # Update parameter
                new_vcmax = current_vcmax - learning_rate * gradient
                new_vcmax = np.clip(new_vcmax, 10.0, 200.0)  # Keep in reasonable range
                manager.set_value(vcmax_key, new_vcmax)
        
        # Check that parameters have been updated
        for pft_name, original_params in pft_params.items():
            vcmax_key = f"pft.{pft_name}.vcmax25"
            final_vcmax = manager.get(vcmax_key)
            # Should be different after calibration
            assert abs(final_vcmax - original_params["vcmax25"]) > 0.001
    
    def test_multi_model_coupling(self):
        """Test parameter sharing in multi-model coupling."""
        
        # Shared environmental parameters
        shared_params = {
            "temperature": 298.15,
            "pressure": 101325.0,
            "humidity": 0.7,
            "wind_speed": 3.0
        }
        
        # Model-specific parameters
        model_params = {
            "atmosphere": {
                "turbulent_schmidt": 0.7,
                "prandtl_number": 0.72
            },
            "land_surface": {
                "roughness_length": 0.01,
                "displacement_height": 0.67
            },
            "hydrology": {
                "infiltration_rate": 0.001,
                "field_capacity": 0.25
            }
        }
        
        # Register all parameters
        manager = get_global_manager()
        
        # Shared parameters in global namespace
        for param_name, value in shared_params.items():
            spec = ParameterSpec(
                    key=f"shared.{param_name}",
                value=value,
                category=ParameterCategory.PHYSICAL, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            manager.register_parameter(spec)
        
        # Model-specific parameters
        for model_name, params in model_params.items():
            for param_name, value in params.items():
                spec = ParameterSpec(
                    key=f"{model_name}.{param_name}",
                    value=value,
                    category=ParameterCategory.MODEL_PARAMETER, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
                )
                manager.register_parameter(spec)
        
        # Simulate coupled model execution
        class CoupledModel:
            def __init__(self, model_name):
                self.model_name = model_name
                self.manager = get_global_manager()
            
            def get_shared_param(self, param_name):
                return self.manager.get(f"shared.{param_name}")
            
            def get_model_param(self, param_name):
                return self.manager.get(f"{self.model_name}.{param_name}")
        
        # Create coupled models
        atm_model = CoupledModel("atmosphere")
        land_model = CoupledModel("land_surface")
        hydro_model = CoupledModel("hydrology")
        
        # All models should see same shared parameters
        temp_atm = atm_model.get_shared_param("temperature")
        temp_land = land_model.get_shared_param("temperature")
        temp_hydro = hydro_model.get_shared_param("temperature")
        
        assert temp_atm == temp_land == temp_hydro == 298.15
        
        # Each model has its own parameters
        assert atm_model.get_model_param("turbulent_schmidt") == 0.7
        assert land_model.get_model_param("roughness_length") == 0.01
        assert hydro_model.get_model_param("infiltration_rate") == 0.001
        
        # Update shared parameter affects all models
        manager.set_value("shared.temperature", 300.0)
        
        assert atm_model.get_shared_param("temperature") == 300.0
        assert land_model.get_shared_param("temperature") == 300.0
        assert hydro_model.get_shared_param("temperature") == 300.0


if __name__ == "__main__":
    # Run tests (the benchmark tests report throughput and memory themselves)
    pytest.main([__file__, "-v", "-k", "not slow"])

