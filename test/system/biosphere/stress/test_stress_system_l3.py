"""
L3 System Tests for Stress Module

Tests complete system integration including:
- Full simulation pipeline
- Real-world parameter values
- Conservation laws
- Performance benchmarks
"""

import pytest
import torch
import numpy as np
import time
import warnings
from typing import Dict, Any

from test.utils.forward_call import call_forward

# Import stress functors
from relic.scientific.biosphere.stress import (
    WaterStress,
    ThermalStress,
    NutrientStress,
    OzoneStress
)

# Import system components
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.dependency_graph import DependencyGraphResolver
from relic.parameters import get_global_manager
from relic.core.abstractions import FluxDict, StateDict

# Real functors used to satisfy dependency resolution in system tests
from relic.scientific.atmosphere.environmental_forcing import EnvironmentalForcingFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition

# StateDict and FluxDict are typing aliases; use dict for runtime construction in tests.
StateDict = dict
FluxDict = dict


class _NoOpDependency:
    """Minimal dependency stub used to close graph dependencies in system tests."""

    def __init__(self, name: str) -> None:
        self.name = name

    def forward(self, state: Dict[str, Any], time: float, dt: float) -> Dict[str, Any]:
        return {}


def _build_dependency_functors() -> Dict[str, Any]:
    """Build real functors with minimal config to satisfy dependency resolution."""
    return {
        "environmental_forcing": EnvironmentalForcingFunctor(name="environmental_forcing"),
        "bucket_soil_moisture": BucketSoilMoisture(name="bucket_soil_moisture"),
        "leaf_temperature_calculator": _NoOpDependency("leaf_temperature_calculator"),
        "atmospheric_radiation": RadiationFunctor(name="atmospheric_radiation"),
        "atmospheric_chemistry": ChemistryFunctor(name="atmospheric_chemistry", physical_boundary="open_mechanism"),
        "farquhar_photosynthesis": FarquharPhotosynthesis(
            name="farquhar_photosynthesis",
            enable_gpu=False,
        ),
        "first_order_decomposition": FirstOrderDecomposition(
            name="first_order_decomposition",
            config={},
        ),
        "mineralization": MineralizationProcess(name="mineralization", config={}),
        "nitrification": NitrificationProcess(name="nitrification", config={}),
    }


@pytest.fixture
def simulator():
    """Create full simulator with stress modules."""
    sim = RELICSimulator()

    for functor in _build_dependency_functors().values():
        if not isinstance(functor, _NoOpDependency):
            sim.register_functor(functor, execution_group="fast")

    # Add stress functors
    sim.register_functor(WaterStress())
    sim.register_functor(ThermalStress())
    sim.register_functor(NutrientStress())
    sim.register_functor(OzoneStress())

    return sim


@pytest.fixture
def realistic_climate_data():
    """Generate realistic climate forcing data."""
    # One year of hourly data
    hours_per_year = 365 * 24
    time = torch.arange(hours_per_year, dtype=torch.float32)
    
    # Temperature: seasonal variation + daily cycle
    temp_mean = 288.15  # 15C mean
    temp_seasonal = 10 * torch.sin(2 * np.pi * time / (365 * 24))
    temp_daily = 5 * torch.sin(2 * np.pi * time / 24)
    temperature = temp_mean + temp_seasonal + temp_daily
    
    # Soil water potential: seasonal drought pattern
    water_mean = -0.5  # MPa
    water_seasonal = 0.3 * torch.sin(2 * np.pi * time / (365 * 24) + np.pi)
    water_potential = water_mean + water_seasonal
    
    # Nutrients: gradual depletion with occasional replenishment
    nitrogen = 0.02 * torch.exp(-time / (365 * 24 * 2)) + 0.01
    phosphorus = 0.001 * torch.exp(-time / (365 * 24 * 3)) + 0.0005
    
    # Ozone: daily pattern with seasonal variation
    ozone_mean = 40.0  # ppb
    ozone_daily = 20 * torch.sin(2 * np.pi * time / 24 - np.pi/2)
    ozone_seasonal = 10 * torch.sin(2 * np.pi * time / (365 * 24))
    ozone = torch.clamp(ozone_mean + ozone_daily + ozone_seasonal, min=0)
    
    return {
        'air_temperature': temperature,
        'soil_water_potential': water_potential,
        'nitrogen_concentration': nitrogen,
        'phosphorus_concentration': phosphorus,
        'o3_mixing_ratio': ozone,
        'stomatal_conductance': torch.ones_like(time) * 0.2  # mol m-2 s-1
    }


class TestStressSystemIntegration:
    """Test complete system integration."""
    
    def test_full_simulation_pipeline(self, simulator, realistic_climate_data):
        """Test full simulation with all stress factors."""
        # Initialize state manager
        state_manager = StateManager()
        
        # Run simulation for one day (24 hours)
        n_hours = 24
        results = {
            'water_stress': [],
            'thermal_stress': [],
            'nutrient_stress': [],
            'ozone_stress': [],
            'combined_stress': []
        }
        
        for hour in range(n_hours):
            # Update state with climate data
            state = StateDict({
                'nitrogen_concentration': realistic_climate_data['nitrogen_concentration'][hour],
                'phosphorus_concentration': realistic_climate_data['phosphorus_concentration'][hour],
            })
            environment = StateDict({
                'soil_water_potential': realistic_climate_data['soil_water_potential'][hour],
                'air_temperature': realistic_climate_data['air_temperature'][hour],
                'o3_mixing_ratio': realistic_climate_data['o3_mixing_ratio'][hour],
                'stomatal_conductance': realistic_climate_data['stomatal_conductance'][hour],
            })
            
            # Compute all stress factors
            water_result = call_forward(simulator.functors['water_stress'], {**state, **environment})
            thermal_result = call_forward(simulator.functors['thermal_stress'], {**state, **environment})
            nutrient_result = call_forward(simulator.functors['nutrient_stress'], {**state, **environment})
            ozone_result = call_forward(simulator.functors['ozone_stress'], {**state, **environment})
            
            # Store results
            results['water_stress'].append(water_result['water_stress_factor'].item())
            results['thermal_stress'].append(thermal_result['temperature_stress_factor'].item())
            nutrient_factor = nutrient_result['nutrient_stress_factor']
            nitrogen_factor = nutrient_result[
                'diagnostics.nutrient_stress.nitrogen_factor'
            ]
            phosphorus_factor = nutrient_result[
                'diagnostics.nutrient_stress.phosphorus_factor'
            ]
            torch.testing.assert_close(
                nutrient_factor,
                torch.minimum(nitrogen_factor, phosphorus_factor),
            )
            results['nutrient_stress'].append(nutrient_factor.item())
            results['ozone_stress'].append(ozone_result['ozone_stress_factor'].item())
            
            # Combined stress (multiplicative)
            combined = (
                water_result['water_stress_factor'] *
                thermal_result['temperature_stress_factor'] *
                nutrient_factor *
                ozone_result['ozone_stress_factor']
            ).item()
            results['combined_stress'].append(combined)
        
        # Verify results are reasonable
        for stress_type, values in results.items():
            values_array = np.array(values)
            
            # All values should be in [0, 1]
            assert np.all(values_array >= 0.0)
            assert np.all(values_array <= 1.0)
            
            # Should have some variation
            if stress_type == "combined_stress":
                coefficient_of_variation = values_array.std() / values_array.mean()
                assert coefficient_of_variation > 0.1
            elif stress_type not in ("nutrient_stress", "ozone_stress"):
                min_std = 0.001 if stress_type == "water_stress" else 0.01
                assert values_array.std() > min_std, f"{stress_type} has no variation"
            
            # Mean should be reasonable (not all minimum or maximum)
            mean_value = values_array.mean()
            if stress_type == "nutrient_stress":
                assert 0.0 < mean_value < 1.0, f"{stress_type} mean is extreme"
            elif stress_type == "ozone_stress":
                assert mean_value >= 0.099, f"{stress_type} mean is too low"
            elif stress_type == "combined_stress":
                assert mean_value > 0.0, f"{stress_type} mean is too low"
            else:
                assert 0.1 < mean_value < 0.9, f"{stress_type} mean is extreme"
    
    def test_dependency_resolution(self):
        """Test that stress functors integrate with dependency resolver."""
        resolver = DependencyGraphResolver()

        for name, functor in _build_dependency_functors().items():
            resolver.add_functor(functor, name)

        # Create stress functors
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        nutrient_stress = NutrientStress()
        ozone_stress = OzoneStress()
        
        # Add to resolver
        resolver.add_functor(water_stress, 'water_stress')
        resolver.add_functor(thermal_stress, 'thermal_stress')
        resolver.add_functor(nutrient_stress, 'nutrient_stress')
        resolver.add_functor(ozone_stress, 'ozone_stress')
        
        # Get execution order
        execution_order = resolver.topological_sort()
        
        # All stress functors should be in execution order
        functor_names = execution_order
        assert 'water_stress' in functor_names
        assert 'thermal_stress' in functor_names
        assert 'nutrient_stress' in functor_names
        assert 'ozone_stress' in functor_names
        assert nutrient_stress.get_dependencies() == []
        assert 'nitrogen_availability' not in functor_names
        assert 'phosphorus_availability' not in functor_names

    def test_nutrient_stress_simulator_provenance(self):
        """Commit the canonical control through the real simulator runtime."""
        simulator = RELICSimulator({
            "max_time": 3600.0,
            "time_step": 3600.0,
            "conservation_checks": False,
            "device": "cpu",
        })
        simulator.initialize_state({
            "nitrogen_concentration": torch.tensor(0.02),
            "phosphorus_concentration": torch.tensor(0.002),
        })
        functor = NutrientStress()
        simulator.register_functor(functor, execution_group="fast")
        initial_n = simulator.state_manager.get_state(
            "nitrogen_concentration"
        ).clone()
        initial_p = simulator.state_manager.get_state(
            "phosphorus_concentration"
        ).clone()

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            updates = simulator.step(dt=3600.0)

        runtime_output = updates["nutrient_stress_factor"]
        target = {
            "dtype": runtime_output.dtype,
            "device": runtime_output.device,
        }
        runtime_n = initial_n.to(**target)
        runtime_p = initial_p.to(**target)
        n_min = torch.as_tensor(functor.get_parameter_value(
            "n_minimum_concentration",
            namespace=functor.namespace,
            required=True,
        ), **target)
        n_crit = torch.as_tensor(functor.get_parameter_value(
            "n_critical_concentration",
            namespace=functor.namespace,
            required=True,
        ), **target)
        p_min = torch.as_tensor(functor.get_parameter_value(
            "p_minimum_concentration",
            namespace=functor.namespace,
            required=True,
        ), **target)
        p_crit = torch.as_tensor(functor.get_parameter_value(
            "p_critical_concentration",
            namespace=functor.namespace,
            required=True,
        ), **target)
        exponent = torch.as_tensor(functor.get_parameter_value(
            "nutrient_stress_exponent",
            namespace=functor.namespace,
            required=True,
        ), **target)
        expected_n = ((runtime_n - n_min) / (n_crit - n_min)).clamp_min(0).pow(
            exponent
        )
        expected_p = ((runtime_p - p_min) / (p_crit - p_min)).clamp_min(0).pow(
            exponent
        )
        expected = torch.minimum(expected_n, expected_p)

        torch.testing.assert_close(runtime_output, expected)
        torch.testing.assert_close(
            simulator.state_manager.get_state("nitrogen_concentration"),
            initial_n,
        )
        torch.testing.assert_close(
            simulator.state_manager.get_state("phosphorus_concentration"),
            initial_p,
        )
        provenance = simulator.get_variable_provenance("nutrient_stress_factor")
        assert {record["role"] for record in provenance} == {
            "writer",
            "final_writer",
        }
        assert {record["mode"] for record in provenance} == {
            "UpdateMode.REPLACE"
        }
        assert {record["functor"] for record in provenance} == {
            "nutrient_stress"
        }
        assert str(simulator.get_effective_device("nutrient_stress")) == "cpu"
    
    def test_conservation_with_stress(self, simulator):
        """Test that stress factors maintain conservation laws."""
        # Stress factors only modulate rates, don't create/destroy mass/energy
        
        # Create a mock photosynthesis rate
        base_gpp = torch.tensor(10.0)  # umol CO2 m-2 s-1

        state = StateDict({
            'nitrogen_concentration': torch.tensor(0.02),
            'phosphorus_concentration': torch.tensor(0.002)
        })
        environment = StateDict({
            'soil_water_potential': torch.tensor(-0.5),
            'air_temperature': torch.tensor(298.0),
            'o3_mixing_ratio': torch.tensor(40.0e-9),  # mol mol-1 (40 ppb)
            'stomatal_conductance': torch.tensor(0.2)
        })
        
        # Get all stress factors
        water_stress = call_forward(simulator.functors['water_stress'], {**state, **environment})
        thermal_stress = call_forward(simulator.functors['thermal_stress'], {**state, **environment})
        nutrient_stress = call_forward(simulator.functors['nutrient_stress'], {**state, **environment})
        ozone_stress = call_forward(simulator.functors['ozone_stress'], {**state, **environment})
        
        # Apply stress to GPP
        stressed_gpp = (
            base_gpp *
            water_stress['water_stress_factor'] *
            thermal_stress['temperature_stress_factor'] *
            nutrient_stress['nutrient_stress_factor'] *
            ozone_stress['ozone_stress_factor']
        )
        
        # Stressed GPP should be less than or equal to base
        assert stressed_gpp <= base_gpp
        assert stressed_gpp >= 0.0
        
        # Conservation: stress doesn't create carbon
        carbon_in = base_gpp
        carbon_out = stressed_gpp + (base_gpp - stressed_gpp)  # Lost potential
        assert torch.isclose(carbon_in, carbon_out, rtol=1e-6)
    
    def test_stress_response_curves(self):
        """Test stress response curves match expected patterns."""
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        
        # Test water stress response curve
        water_potentials = torch.linspace(-3.0, 0.0, 100)
        water_stress_values = []

        for wp in water_potentials:
            environment = StateDict({'soil_water_potential': wp})
            result = call_forward(water_stress, {**StateDict({}), **environment})
            water_stress_values.append(result['water_stress_factor'].item())
        
        water_stress_values = np.array(water_stress_values)
        
        # Should be monotonically increasing (less negative = less stress)
        assert all(water_stress_values[i] <= water_stress_values[i+1] 
                  for i in range(len(water_stress_values)-1))
        
        # Test thermal stress response curve
        temperatures = torch.linspace(273, 323, 100)  # 0-50C
        thermal_stress_values = []

        for temp in temperatures:
            environment = StateDict({'air_temperature': temp})
            result = call_forward(thermal_stress, {**StateDict({}), **environment})
            thermal_stress_values.append(result['temperature_stress_factor'].item())
        
        thermal_stress_values = np.array(thermal_stress_values)
        
        # Should have a peak (optimal temperature)
        max_idx = np.argmax(thermal_stress_values)
        assert 20 < max_idx < 80  # Peak not at extremes
        
        # Should decrease on both sides of optimum
        assert thermal_stress_values[0] < thermal_stress_values[max_idx]
        assert thermal_stress_values[-1] < thermal_stress_values[max_idx]
    
    def test_performance_benchmarks(self, simulator, realistic_climate_data):
        """Test performance of stress calculations."""
        # Prepare batch data
        batch_size = 1000
        state = StateDict({
            'nitrogen_concentration': realistic_climate_data['nitrogen_concentration'][:batch_size],
            'phosphorus_concentration': realistic_climate_data['phosphorus_concentration'][:batch_size],
        })
        environment = StateDict({
            'soil_water_potential': realistic_climate_data['soil_water_potential'][:batch_size],
            'air_temperature': realistic_climate_data['air_temperature'][:batch_size],
            'o3_mixing_ratio': realistic_climate_data['o3_mixing_ratio'][:batch_size],
            'stomatal_conductance': realistic_climate_data['stomatal_conductance'][:batch_size],
        })
        
        # Benchmark each stress functor
        benchmarks = {}
        stress_functors = {
            name: simulator.functors[name]
            for name in ["water_stress", "thermal_stress", "nutrient_stress", "ozone_stress"]
        }

        for name, functor in stress_functors.items():
            start_time = time.time()
            
            # Run computation multiple times
            n_runs = 100
            for _ in range(n_runs):
                result = call_forward(functor, {**state, **environment})
            
            elapsed = time.time() - start_time
            benchmarks[name] = elapsed / n_runs
        
        # All stress calculations should be fast
        for name, time_per_run in benchmarks.items():
            # Should process 1000 samples in less than 10ms
            assert time_per_run < 0.01, f"{name} is too slow: {time_per_run:.4f}s"
        
        print("\nPerformance Benchmarks (1000 samples):")
        for name, time_per_run in benchmarks.items():
            print(f"  {name}: {time_per_run*1000:.2f} ms")
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter changes."""
        param_manager = get_global_manager()
        
        # Test water stress sensitivity to wilting point
        water_stress = WaterStress()
        
        # Original calculation
        environment = StateDict({'soil_water_potential': torch.tensor(-1.0)})
        original_result = call_forward(water_stress, {**StateDict({}), **environment})
        original_stress = original_result['water_stress_factor'].item()

        # Modify wilting point parameter
        original_wilting = param_manager.get('biosphere.stress.water.wilting_point_potential')
        param_manager.set_value('biosphere.stress.water.wilting_point_potential', -2.0)
        
        # Recalculate with new parameter
        water_stress_new = WaterStress()
        new_result = call_forward(water_stress_new, {**StateDict({}), **environment})
        new_stress = new_result['water_stress_factor'].item()
        
        # Should be different
        assert abs(new_stress - original_stress) > 0.1
        
        # Restore original parameter
        param_manager.set_value('biosphere.stress.water.wilting_point_potential', original_wilting)
    
    def test_multi_scale_integration(self, simulator):
        """Test stress factors at different temporal scales."""
        # Hourly data
        hourly_environment = StateDict({
            'air_temperature': torch.tensor(298.0)
        })
        
        # Daily average (24 hours)
        daily_temps = torch.linspace(295, 305, 24)
        daily_environment = StateDict({
            'air_temperature': daily_temps.mean()
        })
        
        # Compute stress at both scales
        thermal_stress = simulator.functors['thermal_stress']
        
        hourly_result = call_forward(thermal_stress, {**StateDict({}), **hourly_environment})
        daily_result = call_forward(thermal_stress, {**StateDict({}), **daily_environment})
        
        # Results should be similar but not identical
        hourly_val = hourly_result['temperature_stress_factor'].item()
        daily_val = daily_result['temperature_stress_factor'].item()
        
        assert abs(hourly_val - daily_val) < 0.5  # Not too different
        assert hourly_val != daily_val  # But not identical


class TestStressRealWorldScenarios:
    """Test with real-world scenarios and edge cases."""
    
    def test_drought_scenario(self):
        """Test stress response during drought conditions."""
        water_stress = WaterStress()
        
        # Simulate progressive drought
        days = 30
        water_potentials = torch.linspace(-0.5, -2.5, days)
        
        stress_values = []
        
        for wp in water_potentials:
            environment = StateDict({'soil_water_potential': wp})
            result = call_forward(water_stress, {**StateDict({}), **environment})
            stress_values.append(result['water_stress_factor'].item())
        
        stress_values = np.array(stress_values)
        
        # Stress should increase (factor decreases) with drought
        assert stress_values[0] > stress_values[-1]
        
        # Should show gradual decline
        differences = np.diff(stress_values)
        assert all(d <= 0 for d in differences)
    
    def test_heatwave_scenario(self):
        """Test stress response during heatwave."""
        thermal_stress = ThermalStress()
        
        # Normal -> Heatwave -> Recovery
        temps = []
        temps.extend([298] * 5)  # Normal (25C)
        temps.extend([308, 313, 318, 313, 308])  # Heatwave (35-45C)
        temps.extend([298] * 5)  # Recovery
        
        temperatures = torch.tensor(temps, dtype=torch.float32)
        
        stress_values = []
        
        for temp in temperatures:
            environment = StateDict({'air_temperature': temp})
            result = call_forward(thermal_stress, {**StateDict({}), **environment})
            stress_values.append(result['temperature_stress_factor'].item())
        
        # Stress should decrease during heatwave
        normal_stress = np.mean(stress_values[:5])
        heatwave_stress = np.mean(stress_values[5:10])
        recovery_stress = np.mean(stress_values[10:])
        
        assert heatwave_stress < normal_stress
        assert abs(recovery_stress - normal_stress) < 0.1  # Should recover
    
    def test_nutrient_deficiency(self):
        """Test progressive nutrient deficiency."""
        nutrient_stress = NutrientStress()
        
        # Simulate nutrient depletion over growing season
        days = 100
        nitrogen = 0.03 * np.exp(-np.linspace(0, 2, days)) + 0.005
        phosphorus = 0.002 * np.exp(-np.linspace(0, 3, days)) + 0.0002
        
        stress_values = []
        
        for n, p in zip(nitrogen, phosphorus):
            state = StateDict({
                'nitrogen_concentration': torch.tensor(n),
                'phosphorus_concentration': torch.tensor(p)
            })
            result = call_forward(nutrient_stress, {**state, **StateDict({})})
            stress_values.append(result['nutrient_stress_factor'].item())
        
        stress_values = np.array(stress_values)
        
        # Stress should increase with depletion
        early_season = np.mean(stress_values[:20])
        late_season = np.mean(stress_values[-20:])
        
        assert early_season > late_season
    
    def test_ozone_pollution_event(self):
        """Test ozone stress during pollution event."""
        ozone_stress = OzoneStress()
        
        # Normal -> Pollution event -> Recovery
        ozone_conc = []
        ozone_conc.extend([40] * 10)  # Normal
        ozone_conc.extend([80, 120, 150, 120, 80])  # Pollution event
        ozone_conc.extend([40] * 10)  # Recovery
        
        ozone_concentrations = torch.tensor(ozone_conc, dtype=torch.float32)
        
        stress_values = []
        
        for ozone in ozone_concentrations:
            environment = StateDict({
                'o3_mixing_ratio': ozone,
                'stomatal_conductance': torch.tensor(0.2)
            })
            result = call_forward(ozone_stress, {**StateDict({}), **environment})
            stress_values.append(result['ozone_stress_factor'].item())
        
        # Stress should increase during pollution
        normal_stress = np.mean(stress_values[:10])
        pollution_stress = np.mean(stress_values[10:15])
        
        assert pollution_stress <= normal_stress


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

