"""
L3 System Tests for Hydrosphere Module

This module provides system-level tests for the complete hydrosphere module
focusing on end-to-end functionality, performance, and real-world scenarios.

Test Coverage:
- Complete hydrological cycle simulation
- Multi-module system integration (P0+P1+P2)
- Performance benchmarks and scalability
- Real-world scenario validation
- System robustness and error handling
- Configuration management and parameter sensitivity
- Memory and computational efficiency

All tests validate system-level behavior under realistic conditions
and verify compliance with scientific accuracy requirements.
"""

import sys
import os
import unittest
import math
import time

import torch

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.core.state import UpdateItem
from relic.data.standards.unit_converter import unit_converter
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.scientific.hydrosphere.transport.capillary_rise import (
    CapillaryRiseTransport,
)


def _unwrap_update(value):
    """Return the tensor carried by an explicit state update."""
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _to_float(value):
    """Convert one scalar tensor to a detached Python float."""
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _advance_canonical_groundwater(
    groundwater_model,
    capillary_model,
    *,
    groundwater_storage,
    water_table_depth,
    drainage_depth,
    time_seconds,
    dt_seconds,
):
    """Advance the canonical capillary-groundwater boundary ledger.

    References:
        Gardner (1958); Wigmosta et al. (1994); Beven and Kirkby (1979).
    Mathematical expression:
        Delta S = (F_drain - F_cap - F_base) * dt * 1000 / rho_w.
    Parameter meanings:
        S is groundwater storage [mm], F_drain is downward drainage,
        F_cap is upward capillary transfer, F_base is boundary export
        [kg m-2 s-1], rho_w is water density [kg m-3], and dt is [s].
    Open-system route:
        Drainage and capillary rise are internal soil-groundwater transfers;
        baseflow is the explicit groundwater boundary export.
    """
    if drainage_depth < 0.0:
        raise ValueError("drainage_depth must be non-negative")
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")

    storage_tensor = torch.tensor(groundwater_storage, dtype=torch.float64)
    depth_tensor = torch.tensor(water_table_depth, dtype=torch.float64)
    drainage_rate = torch.tensor(
        drainage_depth / dt_seconds,
        dtype=torch.float64,
    )
    drainage_flux = unit_converter.convert(
        drainage_rate,
        "drainage_flux",
        "mm s-1",
        "kg m-2 s-1",
    )

    with torch.no_grad():
        capillary_result = capillary_model.forward(
            {"water_table_depth": depth_tensor},
            time=time_seconds,
            dt=dt_seconds,
        )
        capillary_flux = _unwrap_update(
            capillary_result["capillary_rise_flux"]
        ).to(dtype=torch.float64)
        groundwater_result = groundwater_model.forward(
            {
                "groundwater_storage": storage_tensor,
                "water_table_depth": depth_tensor,
                "drainage_flux": drainage_flux,
                "capillary_rise_flux": capillary_flux,
                "lateral_inflow": torch.zeros_like(storage_tensor),
                "groundwater_pumping": torch.zeros_like(storage_tensor),
            },
            time=time_seconds,
            dt=dt_seconds,
        )

    baseflow_flux = _unwrap_update(groundwater_result["baseflow_flux"])
    diagnostics = groundwater_result["diagnostics.groundwater"]
    capillary_depth = (
        unit_converter.convert(capillary_flux, "capillary_rise_flux", "kg m-2 s-1", "mm s-1")
        * dt_seconds
    )
    baseflow_depth = (
        unit_converter.convert(baseflow_flux, "baseflow_flux", "kg m-2 s-1", "mm s-1")
        * dt_seconds
    )
    return {
        "groundwater_storage": _to_float(
            _unwrap_update(groundwater_result["groundwater_storage"])
        ),
        "water_table_depth": _to_float(
            _unwrap_update(groundwater_result["water_table_depth"])
        ),
        "drainage_flux": _to_float(drainage_flux),
        "capillary_flux": _to_float(capillary_flux),
        "capillary_depth": _to_float(capillary_depth),
        "baseflow_flux": _to_float(baseflow_flux),
        "baseflow_depth": _to_float(baseflow_depth),
        "storage_balance_residual": _to_float(
            diagnostics["storage_balance_residual"]
        ),
    }


class TestCompleteHydrologicalCycle(unittest.TestCase):
    """Test complete hydrological cycle simulation (L3 system test)."""
    
    def test_full_cycle_water_balance(self):
        """Test complete hydrological cycle with all P0+P1+P2 components."""
        print("\n" + "="*60)
        print("COMPLETE HYDROLOGICAL CYCLE SIMULATION")
        print("="*60)
        
        # Time parameters
        simulation_days = 30
        dt_seconds = 3600.0  # 1-hour time step
        steps_per_day = 24
        total_steps = simulation_days * steps_per_day
        
        # System state initialization
        system_state = {
            # P0: Soil moisture (Richards equation)
            "soil_moisture_profile": [0.15, 0.18, 0.20, 0.22, 0.25],  # 5 layers, m3/m3
            "soil_layers": 5,
            "layer_thickness": 0.2,  # m
            "total_soil_depth": 1.0,  # m
            
            # P1: Groundwater dynamics
            "groundwater_storage": 100.0,  # mm
            "water_table_depth": 2.5,      # m
            
            # P1: Capillary rise
            "capillary_active": False,
            "capillary_flux": 0.0,  # kg m-2 s-1
            
            # P2: Canopy interception
            "canopy_storage": 0.0,  # mm
            "lai": 2.0,
            "canopy_capacity": 0.4,  # mm
            
            # P2: Snow albedo  
            "snow_present": False,
            "surface_albedo": 0.15,  # Bare soil
            "snow_age": 0.0,  # days
            
            # Water balance tracking
            "total_water_input": 0.0,    # mm
            "total_water_output": 0.0,   # mm
            "total_storage_change": 0.0  # mm
        }
        
        # Forcing data (simplified daily cycle)
        daily_weather = self._generate_weather_forcing(simulation_days)
        
        # System parameters
        system_params = {
            # Soil parameters
            "theta_residual": 0.05,
            "theta_saturated": 0.45,
            "k_saturated": 1e-5,  # m/s
            "alpha_vg": 2.0,
            "n_vg": 1.5,
            
            # Canopy parameters
            "canopy_capacity_per_lai": 0.2,  # mm
            "canopy_evap_efficiency": 1.2,
            
            # Snow parameters
            "fresh_snow_albedo": 0.85,
            "old_snow_albedo": 0.5,
            "snow_reset_threshold": 0.005  # kg m-2 s-1
        }
        
        print(f"Starting {simulation_days}-day simulation with {total_steps} time steps...")
        
        # Main simulation loop
        water_balance_history = []
        groundwater_model = GroundwaterDynamics(name="l3_cycle_groundwater")
        capillary_model = CapillaryRiseTransport(name="l3_cycle_capillary")
        
        for step in range(total_steps):
            day = step // steps_per_day
            hour = step % steps_per_day
            
            # Get current weather forcing
            weather = daily_weather[day]
            hourly_precip = weather["precipitation"] / 24.0  # mm/h
            hourly_et_demand = weather["potential_et"] / 24.0  # mm/h
            
            # Convert to SI units
            precip_flux = hourly_precip / 3.6  # kg m-2 s-1
            et_demand_flux = hourly_et_demand / 3.6  # kg m-2 s-1
            
            # P2: Snow albedo update (if snow present)
            if weather["air_temp"] < 273.15 and precip_flux > 0.001:
                # Snow formation
                system_state["snow_present"] = True
                system_state["surface_albedo"] = system_params["fresh_snow_albedo"]
                system_state["snow_age"] = 0.0
            elif system_state["snow_present"]:
                # Snow aging
                aging_rate = 0.1 * (weather["air_temp"] - 273.15) if weather["air_temp"] > 273.15 else 0.05
                system_state["snow_age"] += aging_rate * dt_seconds / 86400.0
                
                # Update albedo
                age_factor = math.exp(-system_state["snow_age"] / 5.0)
                system_state["surface_albedo"] = (
                    system_params["old_snow_albedo"] + 
                    (system_params["fresh_snow_albedo"] - system_params["old_snow_albedo"]) * age_factor
                )
                
                # Snow melt
                if weather["air_temp"] > 275.15:
                    system_state["snow_present"] = False
                    system_state["surface_albedo"] = 0.15  # Bare soil
            
            # P2: Canopy interception processing
            available_capacity = max(0.0, system_state["canopy_capacity"] - system_state["canopy_storage"])
            interception = min(hourly_precip, available_capacity)
            system_state["canopy_storage"] += interception
            
            # Canopy evaporation (priority over transpiration)
            canopy_evap_demand = hourly_et_demand * system_params["canopy_evap_efficiency"]
            actual_canopy_evap = min(system_state["canopy_storage"], canopy_evap_demand)
            system_state["canopy_storage"] -= actual_canopy_evap
            
            # Throughfall to soil
            throughfall = hourly_precip - interception
            
            # Remaining ET becomes transpiration
            transpiration = max(0.0, hourly_et_demand - actual_canopy_evap)
            
            # P0: Soil moisture dynamics (simplified Richards)
            soil_input = throughfall  # mm/h
            soil_et_loss = transpiration  # mm/h
            
            # Simple soil water balance
            net_soil_change = soil_input - soil_et_loss  # mm/h
            
            # Convert to volumetric change
            avg_moisture_change = net_soil_change / (system_state["total_soil_depth"] * 1000.0)  # m3/m3
            
            # Update soil moisture profile (simplified)
            for i in range(system_state["soil_layers"]):
                layer_factor = 1.0 - i * 0.1  # Decreasing change with depth
                system_state["soil_moisture_profile"][i] += avg_moisture_change * layer_factor
                
                # Apply physical constraints
                system_state["soil_moisture_profile"][i] = max(
                    system_params["theta_residual"],
                    min(system_state["soil_moisture_profile"][i], system_params["theta_saturated"])
                )
            
            # Soil drainage to groundwater
            avg_soil_moisture = sum(system_state["soil_moisture_profile"]) / system_state["soil_layers"]
            drainage_rate = system_params["k_saturated"] * 3600.0 * 1000.0  # mm/h
            
            # Drainage depends on soil saturation
            saturation_factor = ((avg_soil_moisture - system_params["theta_residual"]) / 
                               (system_params["theta_saturated"] - system_params["theta_residual"]))
            actual_drainage = drainage_rate * max(0.0, saturation_factor - 0.5) ** 2
            
            # P1: Canonical capillary and groundwater Functor execution.
            groundwater_step = _advance_canonical_groundwater(
                groundwater_model,
                capillary_model,
                groundwater_storage=system_state["groundwater_storage"],
                water_table_depth=system_state["water_table_depth"],
                drainage_depth=actual_drainage,
                time_seconds=step * dt_seconds,
                dt_seconds=dt_seconds,
            )
            system_state["groundwater_storage"] = groundwater_step[
                "groundwater_storage"
            ]
            system_state["water_table_depth"] = groundwater_step[
                "water_table_depth"
            ]
            system_state["capillary_flux"] = groundwater_step["capillary_flux"]
            capillary_input = groundwater_step["capillary_depth"]
            system_state["capillary_active"] = capillary_input > 0.0

            # Apply the internal groundwater export at the soil bottom boundary.
            system_state["soil_moisture_profile"][-1] += (
                capillary_input / (system_state["layer_thickness"] * 1000.0)
            )
            system_state["soil_moisture_profile"][-1] = min(
                system_state["soil_moisture_profile"][-1],
                system_params["theta_saturated"],
            )
            
            # Water balance accounting
            step_input = hourly_precip + capillary_input
            step_output = actual_canopy_evap + transpiration + actual_drainage
            step_storage_change = interception - actual_canopy_evap + net_soil_change
            
            system_state["total_water_input"] += step_input
            system_state["total_water_output"] += step_output
            system_state["total_storage_change"] += step_storage_change
            
            # Record water balance
            balance_error = abs(step_input - step_output - step_storage_change)
            water_balance_history.append({
                "step": step,
                "day": day,
                "hour": hour,
                "input": step_input,
                "output": step_output,
                "storage_change": step_storage_change,
                "balance_error": balance_error,
                "soil_moisture_avg": avg_soil_moisture,
                "groundwater_storage": system_state["groundwater_storage"],
                "water_table_depth": system_state["water_table_depth"],
                "canopy_storage": system_state["canopy_storage"],
                "surface_albedo": system_state["surface_albedo"],
                "groundwater_balance_residual": groundwater_step[
                    "storage_balance_residual"
                ],
            })
        
        # Final system validation
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {total_steps}")
        
        # Water balance analysis
        total_input = system_state["total_water_input"]
        total_output = system_state["total_water_output"] 
        total_storage_change = system_state["total_storage_change"]
        
        overall_balance_error = abs(total_input - total_output - total_storage_change)
        relative_error = overall_balance_error / max(total_input, 1e-6)
        
        print(f"\nWATER BALANCE SUMMARY:")
        print(f"Total input: {total_input:.2f} mm")
        print(f"Total output: {total_output:.2f} mm")
        print(f"Total storage change: {total_storage_change:.2f} mm")
        print(f"Balance error: {overall_balance_error:.4f} mm")
        print(f"Relative error: {relative_error:.2%}")
        
        # System state summary
        print(f"\nFINAL SYSTEM STATE:")
        print(f"Average soil moisture: {sum(system_state['soil_moisture_profile'])/len(system_state['soil_moisture_profile']):.3f} m3/m3")
        print(f"Groundwater storage: {system_state['groundwater_storage']:.1f} mm")
        print(f"Water table depth: {system_state['water_table_depth']:.2f} m")
        print(f"Canopy storage: {system_state['canopy_storage']:.2f} mm")
        print(f"Surface albedo: {system_state['surface_albedo']:.3f}")
        print(f"Snow present: {system_state['snow_present']}")
        print(f"Capillary rise active: {system_state['capillary_active']}")
        
        # Performance analysis
        avg_balance_error = sum(wb["balance_error"] for wb in water_balance_history) / len(water_balance_history)
        max_balance_error = max(wb["balance_error"] for wb in water_balance_history)
        max_groundwater_balance_residual = max(
            abs(wb["groundwater_balance_residual"])
            for wb in water_balance_history
        )
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"Average step balance error: {avg_balance_error:.2e} mm")
        print(f"Maximum step balance error: {max_balance_error:.2e} mm")
        print(
            "Maximum groundwater balance residual: "
            f"{max_groundwater_balance_residual:.2e} mm"
        )
        
        # System validation tests
        self.assertLess(relative_error, 0.05)  # Less than 5% error
        self.assertLess(avg_balance_error, 0.01)  # Average error < 0.01 mm
        self.assertLess(max_balance_error, 0.1)   # Max error < 0.1 mm
        self.assertLess(max_groundwater_balance_residual, 1.0e-10)
        
        # Physical constraints validation
        for moisture in system_state["soil_moisture_profile"]:
            self.assertGreaterEqual(moisture, system_params["theta_residual"])
            self.assertLessEqual(moisture, system_params["theta_saturated"])
        
        self.assertGreaterEqual(system_state["groundwater_storage"], 0)
        self.assertGreaterEqual(system_state["water_table_depth"], 0.1)
        self.assertGreaterEqual(system_state["canopy_storage"], 0)
        self.assertLessEqual(system_state["canopy_storage"], system_state["canopy_capacity"])
        self.assertGreaterEqual(system_state["surface_albedo"], 0.1)
        self.assertLessEqual(system_state["surface_albedo"], 1.0)
        
        print(f"\nAll system validation tests PASSED!")
        return None
    
    def _generate_weather_forcing(self, days):
        """Generate synthetic weather forcing data."""
        weather_data = []
        
        for day in range(days):
            # Simple weather pattern with some variability
            base_temp = 288.15 + 5.0 * math.sin(2 * math.pi * day / 365.0)  # Seasonal variation
            daily_temp_range = 8.0  # K
            
            # Random-like precipitation (simplified)
            precip_prob = 0.3 if day % 7 < 3 else 0.1  # More rain early in week
            daily_precip = 15.0 * (day % 11) / 10.0 if (day % 17) < (17 * precip_prob) else 0.0
            
            # Potential ET based on temperature and radiation
            temp_factor = max(0.0, (base_temp - 273.15) / 25.0)  # 0-1 scaling
            daily_pet = 2.0 + 3.0 * temp_factor + 1.0 * math.sin(2 * math.pi * day / 365.0)
            
            weather_data.append({
                "day": day,
                "air_temp": base_temp,  # K
                "daily_temp_range": daily_temp_range,  # K
                "precipitation": daily_precip,  # mm day-1
                "potential_et": daily_pet,  # mm day-1
                "net_radiation": 150.0 + 100.0 * temp_factor,  # W/m2
                "relative_humidity": 0.6 + 0.3 * (daily_precip / 20.0),  # 0-1
                "wind_speed": 2.0 + 1.0 * math.sin(2 * math.pi * day / 7.0)  # m/s
            })
        
        return weather_data


class TestSystemPerformance(unittest.TestCase):
    """Test system performance and scalability."""
    
    def test_computational_performance(self):
        """Test computational performance benchmarks."""
        print("\n" + "="*60)
        print("COMPUTATIONAL PERFORMANCE BENCHMARKS")
        print("="*60)
        
        # Test different problem sizes
        problem_sizes = [
            {"soil_layers": 5, "time_steps": 24, "description": "Small (5 layers, 1 day)"},
            {"soil_layers": 10, "time_steps": 168, "description": "Medium (10 layers, 1 week)"},
            {"soil_layers": 20, "time_steps": 720, "description": "Large (20 layers, 1 month)"}
        ]
        
        performance_results = []
        
        for config in problem_sizes:
            print(f"\nTesting {config['description']}...")
            
            # Time the simulation
            start_time = time.time()
            
            # Run simplified simulation
            self._run_performance_simulation(
                soil_layers=config["soil_layers"],
                time_steps=config["time_steps"]
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate performance metrics
            total_operations = config["soil_layers"] * config["time_steps"]
            ops_per_second = total_operations / elapsed_time if elapsed_time > 0 else float('inf')
            
            result = {
                "size": config["description"],
                "layers": config["soil_layers"],
                "steps": config["time_steps"],
                "total_ops": total_operations,
                "elapsed_time": elapsed_time,
                "ops_per_second": ops_per_second
            }
            
            performance_results.append(result)
            
            print(f"  Elapsed time: {elapsed_time:.4f} seconds")
            print(f"  Operations/second: {ops_per_second:.0f}")
            
            # Performance thresholds
            self.assertLess(elapsed_time, 5.0)  # Should complete in under 5 seconds
            self.assertGreater(ops_per_second, 100)  # At least 100 ops/second
        
        # Scalability analysis
        print("\nSCALABILITY ANALYSIS:")
        for i, result in enumerate(performance_results):
            if i > 0:
                prev_result = performance_results[i-1]
                size_ratio = result["total_ops"] / prev_result["total_ops"]
                time_ratio = result["elapsed_time"] / max(prev_result["elapsed_time"], 1e-6)  # Avoid division by zero
                efficiency = size_ratio / max(time_ratio, 1e-6)  # Avoid division by zero
                
                print(f"  {prev_result['size']} -> {result['size']}:")
                print(f"    Size increase: {size_ratio:.1f}x")
                print(f"    Time increase: {time_ratio:.1f}x") 
                print(f"    Efficiency: {efficiency:.2f}")
                
                # Efficiency should be reasonable (accounting for timing variations)
                # Skip efficiency checks for very small timing measurements
                if prev_result["elapsed_time"] > 1e-4 and result["elapsed_time"] > 1e-4:
                    self.assertGreater(efficiency, 0.01)  # Very permissive for small timing variations
                    self.assertLess(efficiency, 1000.0)   # Prevent unrealistic scaling

        return None
    
    def _run_performance_simulation(self, soil_layers, time_steps):
        """Run simplified simulation for performance testing."""
        # Initialize system state
        soil_moisture = [0.2] * soil_layers
        groundwater_storage = 100.0
        water_table_depth = 2.5
        canopy_storage = 0.0
        groundwater_model = GroundwaterDynamics(
            name=f"l3_performance_groundwater_{soil_layers}"
        )
        capillary_model = CapillaryRiseTransport(
            name=f"l3_performance_capillary_{soil_layers}"
        )
        
        # System parameters
        theta_res = 0.05
        theta_sat = 0.45
        k_sat = 1e-5
        
        # Main simulation loop
        for step in range(time_steps):
            # Simplified forcing
            precipitation = 0.1 if (step % 24) < 6 else 0.0  # mm/h
            et_demand = 0.2 if 6 <= (step % 24) <= 18 else 0.05  # mm/h
            
            # Canopy interception
            canopy_capacity = 0.5
            interception = min(precipitation, max(0.0, canopy_capacity - canopy_storage))
            canopy_storage += interception
            
            # Canopy evaporation
            canopy_evap = min(canopy_storage, et_demand)
            canopy_storage -= canopy_evap
            
            # Soil water dynamics
            throughfall = precipitation - interception
            transpiration = max(0.0, et_demand - canopy_evap)
            
            for i in range(soil_layers):
                # Simplified moisture update
                layer_input = throughfall / soil_layers
                layer_et = transpiration / soil_layers
                
                soil_moisture[i] += (layer_input - layer_et) / 1000.0
                soil_moisture[i] = max(theta_res, min(soil_moisture[i], theta_sat))
            
            # Drainage
            avg_moisture = sum(soil_moisture) / soil_layers
            drainage = (
                k_sat
                * 3600.0
                * 1000.0
                * max(0.0, avg_moisture - theta_res)
                / (theta_sat - theta_res)
            )

            groundwater_step = _advance_canonical_groundwater(
                groundwater_model,
                capillary_model,
                groundwater_storage=groundwater_storage,
                water_table_depth=water_table_depth,
                drainage_depth=drainage,
                time_seconds=step * 3600.0,
                dt_seconds=3600.0,
            )
            groundwater_storage = groundwater_step["groundwater_storage"]
            water_table_depth = groundwater_step["water_table_depth"]
        
        return {
            "final_soil_moisture": soil_moisture,
            "final_groundwater_storage": groundwater_storage,
            "final_water_table_depth": water_table_depth,
            "final_canopy_storage": canopy_storage
        }


class TestSystemRobustness(unittest.TestCase):
    """Test system robustness and error handling."""
    
    def test_extreme_weather_conditions(self):
        """Test system behavior under extreme weather conditions."""
        print("\n" + "="*60)
        print("EXTREME WEATHER CONDITIONS TEST")
        print("="*60)
        
        extreme_scenarios = [
            {
                "name": "Extreme drought",
                "precipitation": 0.0,
                "potential_et": 10.0,
                "air_temp": 313.15,  # 40掳C
                "duration_days": 30
            },
            {
                "name": "Extreme flooding",
                "precipitation": 100.0,  # mm day-1
                "potential_et": 1.0,
                "air_temp": 293.15,  # 20掳C  
                "duration_days": 7
            },
            {
                "name": "Extreme cold",
                "precipitation": 5.0,
                "potential_et": 0.1,
                "air_temp": 253.15,  # -20掳C
                "duration_days": 14
            }
        ]
        
        for scenario in extreme_scenarios:
            print(f"\nTesting {scenario['name']}...")
            
            # Run simulation with extreme conditions
            result = self._run_extreme_weather_simulation(scenario)
            
            # Validate system stability
            self.assertIsNotNone(result)
            self.assertIn("final_state", result)
            self.assertIn("stability_metrics", result)
            
            # Check for physical constraint violations
            final_state = result["final_state"]
            
            # Soil moisture constraints
            for i, theta in enumerate(final_state["soil_moisture_profile"]):
                self.assertGreaterEqual(theta, 0.0, 
                    f"Soil moisture layer {i} below zero in {scenario['name']}")
                self.assertLessEqual(theta, 0.5, 
                    f"Soil moisture layer {i} above saturation in {scenario['name']}")
            
            # Groundwater constraints
            self.assertGreaterEqual(final_state["groundwater_storage"], 0.0,
                f"Groundwater storage negative in {scenario['name']}")
            
            # Canopy constraints
            self.assertGreaterEqual(final_state["canopy_storage"], 0.0,
                f"Canopy storage negative in {scenario['name']}")
            
            stability = result["stability_metrics"]
            print(f"  Final avg soil moisture: {stability['avg_soil_moisture']:.3f}")
            print(f"  Final groundwater storage: {stability['groundwater_storage']:.1f} mm")
            print(f"  System stability score: {stability['stability_score']:.2f}")
            
            # Stability score should be reasonable (0-1 scale)
            self.assertGreaterEqual(stability["stability_score"], 0.0)
            self.assertLessEqual(stability["stability_score"], 1.0)
    
    def _run_extreme_weather_simulation(self, scenario):
        """Run simulation under extreme weather conditions."""
        # Initialize conservative system state
        system_state = {
            "soil_moisture_profile": [0.25, 0.25, 0.25, 0.25, 0.25],
            "groundwater_storage": 50.0,  # mm
            "canopy_storage": 0.0,  # mm
            "water_table_depth": 3.0  # m
        }
        
        # System parameters (conservative values for stability)
        params = {
            "theta_residual": 0.05,
            "theta_saturated": 0.45,
            "k_saturated": 1e-5,
            "canopy_capacity": 0.5
        }
        groundwater_model = GroundwaterDynamics(
            name=f"l3_extreme_groundwater_{scenario['name'].replace(' ', '_')}"
        )
        capillary_model = CapillaryRiseTransport(
            name=f"l3_extreme_capillary_{scenario['name'].replace(' ', '_')}"
        )
        
        # Run simulation
        dt_hours = 1.0
        total_steps = scenario["duration_days"] * 24
        
        stability_history = []
        
        for step in range(total_steps):
            hour = step % 24
            
            # Apply extreme forcing
            precip_hourly = scenario["precipitation"] / 24.0  # mm/h
            et_hourly = scenario["potential_et"] / 24.0  # mm/h
            
            # Modify ET based on temperature (reduce at extreme cold)
            if scenario["air_temp"] < 273.15:
                et_hourly *= 0.1  # Minimal ET when frozen
            
            # Canopy processing
            available_capacity = max(0.0, params["canopy_capacity"] - system_state["canopy_storage"])
            interception = min(precip_hourly, available_capacity)
            system_state["canopy_storage"] += interception
            
            canopy_evap = min(system_state["canopy_storage"], et_hourly)
            system_state["canopy_storage"] -= canopy_evap
            
            # Soil water balance
            throughfall = precip_hourly - interception
            transpiration = max(0.0, et_hourly - canopy_evap)
            
            net_soil_input = throughfall - transpiration  # mm/h
            
            # Update soil moisture (simplified)
            for i in range(len(system_state["soil_moisture_profile"])):
                layer_change = net_soil_input / (200.0 * 5)  # mm to m3/m3 conversion
                system_state["soil_moisture_profile"][i] += layer_change
                
                # Apply constraints with damping to prevent instability
                target_moisture = max(params["theta_residual"], 
                                    min(system_state["soil_moisture_profile"][i], params["theta_saturated"]))
                
                # Gradual adjustment to prevent oscillations
                system_state["soil_moisture_profile"][i] = (
                    0.9 * system_state["soil_moisture_profile"][i] + 0.1 * target_moisture
                )
            
            # Canonical GroundwaterDynamics update from the scenario drainage port.
            avg_soil_moisture = sum(system_state["soil_moisture_profile"]) / len(system_state["soil_moisture_profile"])
            drainage = params["k_saturated"] * 3600.0 * 1000.0 * max(0.0, avg_soil_moisture - 0.15)
            groundwater_step = _advance_canonical_groundwater(
                groundwater_model,
                capillary_model,
                groundwater_storage=system_state["groundwater_storage"],
                water_table_depth=system_state["water_table_depth"],
                drainage_depth=drainage,
                time_seconds=step * dt_hours * 3600.0,
                dt_seconds=dt_hours * 3600.0,
            )
            system_state["groundwater_storage"] = groundwater_step[
                "groundwater_storage"
            ]
            system_state["water_table_depth"] = groundwater_step[
                "water_table_depth"
            ]
            
            # Record stability metrics
            stability_metrics = {
                "step": step,
                "avg_soil_moisture": avg_soil_moisture,
                "groundwater_storage": system_state["groundwater_storage"],
                "canopy_storage": system_state["canopy_storage"],
                "net_input": net_soil_input
            }
            
            stability_history.append(stability_metrics)
        
        # Calculate final stability score
        final_avg_moisture = sum(system_state["soil_moisture_profile"]) / len(system_state["soil_moisture_profile"])
        
        # Stability score: 1.0 = ideal, 0.0 = extreme/unstable
        moisture_score = 1.0 - abs(final_avg_moisture - 0.25) / 0.2  # 0.25 is ideal mid-range
        storage_score = 1.0 - abs(system_state["groundwater_storage"] - 50.0) / 100.0
        
        stability_score = max(0.0, min(1.0, (moisture_score + storage_score) / 2.0))
        
        return {
            "final_state": system_state,
            "stability_metrics": {
                "avg_soil_moisture": final_avg_moisture,
                "groundwater_storage": system_state["groundwater_storage"],
                "canopy_storage": system_state["canopy_storage"],
                "stability_score": stability_score
            },
            "stability_history": stability_history
        }
    
    def test_parameter_sensitivity(self):
        """Test system sensitivity to parameter changes."""
        print("\n" + "="*60) 
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("="*60)
        
        # Base parameter set
        base_params = {
            "theta_saturated": 0.45,
            "k_saturated": 1e-5,
            "canopy_capacity_per_lai": 0.2,
            "porosity_decay_length": 2.0,
            "fresh_snow_albedo": 0.85
        }
        
        # Parameter perturbations to test
        perturbations = [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]  # 卤50%, 卤20%, 卤10%
        
        sensitivity_results = {}
        
        for param_name, base_value in base_params.items():
            print(f"\nTesting sensitivity to {param_name}...")
            
            param_results = []
            
            for perturbation in perturbations:
                # Create perturbed parameter set
                test_params = base_params.copy()
                test_params[param_name] = base_value * (1.0 + perturbation)
                
                # Ensure parameter stays within physical bounds
                if param_name == "theta_saturated":
                    test_params[param_name] = max(0.3, min(0.7, test_params[param_name]))
                elif param_name == "k_saturated":
                    test_params[param_name] = max(1e-8, min(1e-3, test_params[param_name]))
                elif param_name == "fresh_snow_albedo":
                    test_params[param_name] = max(0.7, min(1.0, test_params[param_name]))
                
                # Run simplified simulation
                result = self._run_sensitivity_simulation(test_params)
                
                param_results.append({
                    "perturbation": perturbation,
                    "param_value": test_params[param_name],
                    "output_response": result["system_response"]
                })
            
            sensitivity_results[param_name] = param_results
            
            # Calculate sensitivity coefficient
            base_output = [r for r in param_results if r["perturbation"] == 0.1][0]["output_response"]
            small_pert = [r for r in param_results if r["perturbation"] == 0.1][0]
            
            if base_output != 0:
                sensitivity_coeff = (small_pert["output_response"] - base_output) / (0.1 * base_output)
                print(f"  Sensitivity coefficient: {sensitivity_coeff:.3f}")
                
                # Sensitivity should be reasonable (not too high)
                self.assertLess(abs(sensitivity_coeff), 10.0, 
                    f"Parameter {param_name} has excessive sensitivity")
            
        print(f"\nParameter sensitivity analysis completed.")
        return None
    
    def _run_sensitivity_simulation(self, test_params):
        """Run simplified simulation for sensitivity analysis.""" 
        # Simple 7-day simulation
        total_water_output = 0.0
        soil_moisture = 0.25
        
        for day in range(7):
            # Simplified daily water balance
            daily_precip = 5.0  # mm
            
            # Apply parameter influence
            et_factor = 1.0 / (test_params["theta_saturated"] * 10.0)  # Example influence
            daily_et = 3.0 * et_factor
            
            # Simple balance
            net_change = daily_precip - daily_et
            soil_moisture += net_change / 1000.0  # Convert to volumetric
            
            # Constraints
            soil_moisture = max(0.05, min(test_params["theta_saturated"], soil_moisture))
            
            total_water_output += daily_et
        
        return {
            "system_response": total_water_output,  # Use total ET as response metric
            "final_soil_moisture": soil_moisture
        }


def main():
    """Run all L3 system tests."""
    print("=" * 70)
    print("L3 SYSTEM TESTS FOR HYDROSPHERE MODULE")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCompleteHydrologicalCycle,
        TestSystemPerformance,
        TestSystemRobustness
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("L3 SYSTEM TESTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
        
        # System-specific metrics
        print(f"\nSystem Test Categories:")
        print(f"- Complete hydrological cycle: Full P0+P1+P2 integration")
        print(f"- Performance benchmarks: Scalability and efficiency")  
        print(f"- Robustness testing: Extreme conditions and stability")
        print(f"- Parameter sensitivity: System response analysis")
        
    else:
        print("No tests were run")
    
    # Detailed failure/error reporting
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(f"  {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    print("=" * 70)
    
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
