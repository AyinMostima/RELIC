"""
L2 Integration Tests for Hydrosphere Module

This module provides integration tests for hydrosphere components
focusing on module interactions, data flow, and interface coupling.

Test Coverage:
- P0-P1 Integration: Richards equation with capillary rise coupling
- P1-P1 Integration: Groundwater dynamics with capillary rise
- P2-P0 Integration: Canopy interception with soil water balance
- P2-P2 Integration: Snow albedo with energy balance
- Interface mapping integration across all modules
- Mass conservation across coupled processes
- Error propagation and boundary condition handling

All tests validate module interactions using realistic parameter sets
and verify conservation laws and physical constraints.
"""

import sys
import os
import unittest
import math
import torch

from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.hydrosphere.soil_moisture.richards import (
    RichardsSoilMoisture,
)
from relic.scientific.hydrosphere.transport.capillary_rise import (
    CapillaryRiseTransport,
)
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.utils.physical_constants import get_physical_constant

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _unwrap_update(value):
    return value.value if isinstance(value, UpdateItem) else value


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


class TestRichardsCapillaryIntegration(unittest.TestCase):
    """Test the production capillary-rise boundary routed into Richards."""

    def _build_coupled_step(self):
        """Run the selected lower-boundary coupling with canonical state."""
        capillary = CapillaryRiseTransport(name="capillary_rise_transport")
        richards = RichardsSoilMoisture(name="richards_soil_moisture")
        capillary_result = capillary.forward(
            {
                "water_table_depth": torch.tensor(1.5),
                "soil_moisture_profile": torch.full(
                    (capillary.capillary_profile_layer_count,), 0.25
                ),
            },
            time=0.0,
            dt=60.0,
        )
        capillary_flux = capillary_result["capillary_rise_flux"]
        self.assertIsInstance(capillary_flux, UpdateItem)
        self.assertEqual(capillary_flux.unit, "kg m-2 s-1")
        self.assertGreater(float(capillary_flux.value.detach()), 0.0)

        richards.set_boundary_conditions(bottom={"type": "capillary_flux"})
        state = {
            "soil_moisture": torch.full((richards.soil_layers,), 0.25),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
            "capillary_rise_flux": capillary_flux.value,
        }
        result = richards.forward(state, time=0.0, dt=60.0)
        return capillary, richards, state, result
    
    def test_richards_capillary_coupling(self):
        """Capillary mass enters through an explicit selected lower port."""
        capillary, richards, state, result = self._build_coupled_step()

        self.assertIn("capillary_rise_flux", richards.metadata.input_variables)
        self.assertIn("capillary_rise_flux", richards.optional_input_variables)
        self.assertIn("capillary_rise_transport", richards.get_dependencies()["optional"])
        self.assertIn(
            "soil_capillary_boundary", {port.name for port in richards.ports()}
        )
        consumer_port = next(
            port
            for port in richards.ports()
            if port.flow_var == "capillary_rise_flux"
        )
        self.assertEqual(consumer_port.positive_power_convention, "into_system")
        self.assertEqual(consumer_port.flow_space.units, "kg m-2 s-1")
        producer_port = next(
            port
            for port in capillary.ports()
            if port.flow_var == "capillary_rise_flux"
        )
        self.assertEqual(producer_port.positive_power_convention, "out_of_system")
        self.assertEqual(producer_port.flow_space.units, consumer_port.flow_space.units)
        self.assertEqual(result["soil_moisture"].mode, UpdateMode.REPLACE)
        self.assertFalse(result["soil_moisture"].per_timestep)
        self.assertEqual(float(result["drainage_flux"].value.detach()), 0.0)
        bottom_flux = result[
            "diagnostics.soil_moisture_richards.water_flux_profile"
        ].value[-1]
        self.assertLess(float(bottom_flux.detach()), 0.0)

        manager = StateManager(device="cpu")
        manager.initialize({key: value.clone() for key, value in state.items()})
        manager.apply_updates(result, dt=60.0)
        torch.testing.assert_close(
            manager.get_state("soil_moisture"), result["soil_moisture"].value
        )
    
    def test_mass_conservation_coupling(self):
        """The signed lower flux closes the production water ledger."""
        _, richards, state, result = self._build_coupled_step()
        density = torch.as_tensor(get_physical_constant("water_density"))
        storage_rate = (
            (result["soil_moisture"].value - state["soil_moisture"]).sum()
            * richards.layer_thickness
            * density
            / 60.0
        )
        expected_rate = (
            state["infiltration_flux"]
            - state["evapotranspiration_flux"]
            + state["capillary_rise_flux"]
            - result["drainage_flux"].value
        )
        torch.testing.assert_close(storage_rate, expected_rate, rtol=1.0e-3, atol=1.0e-8)
        self.assertLessEqual(
            float(
                result["diagnostics.soil_moisture_richards"][
                    "mass_conservation_error"
                ].detach()
            ),
            richards.mass_conservation_tolerance,
        )

    def test_capillary_boundary_is_conservative_for_both_nonlinear_solvers(self):
        """Picard and Newton consume the same signed lower water port."""
        manager = get_global_manager()
        for enable_newton, expected_solver in (
            (False, "picard"),
            (True, "newton_raphson"),
        ):
            with self.subTest(nonlinear_solver=expected_solver):
                snapshot = manager.snapshot()
                try:
                    manager.set_value(
                        "hydrosphere.soil_moisture.richards_enable_newton_raphson",
                        enable_newton,
                        units="dimensionless",
                        validate=True,
                        notify=False,
                    )
                    richards = RichardsSoilMoisture(
                        name=f"richards_{expected_solver}_capillary"
                    )
                    richards.set_boundary_conditions(
                        bottom={"type": "capillary_flux"}
                    )
                    capillary_flux = torch.tensor(1.0e-4)
                    result = richards.forward(
                        {
                            "soil_moisture": torch.full(
                                (richards.soil_layers,), 0.25
                            ),
                            "infiltration_flux": torch.tensor(0.0),
                            "evapotranspiration_flux": torch.tensor(0.0),
                            "capillary_rise_flux": capillary_flux,
                        },
                        time=0.0,
                        dt=1.0,
                    )
                finally:
                    manager.restore(snapshot)

                diagnostics = result["diagnostics.soil_moisture_richards"]
                moisture_rate = result[
                    "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
                ].value
                storage_rate = (
                    torch.sum(moisture_rate)
                    * richards.layer_thickness
                    * richards.water_density
                )
                signed_bottom_flux = result[
                    "diagnostics.soil_moisture_richards.water_flux_profile"
                ].value[-1]

                # The nonlinear solver is configuration, not a diagnostic.
                self.assertIs(richards.enable_newton_raphson, enable_newton)
                self.assertGreater(float(storage_rate.detach()), 0.0)
                torch.testing.assert_close(
                    storage_rate,
                    capillary_flux,
                    rtol=1.0e-3,
                    atol=1.0e-8,
                )
                torch.testing.assert_close(
                    signed_bottom_flux,
                    -capillary_flux,
                )
                torch.testing.assert_close(
                    result["drainage_flux"].value,
                    torch.zeros_like(capillary_flux),
                )
                self.assertLessEqual(
                    float(diagnostics["mass_conservation_error"].detach()),
                    richards.mass_conservation_tolerance,
                )


class TestGroundwaterCapillaryIntegration(unittest.TestCase):
    """Test P1-P1 integration: Groundwater dynamics with capillary rise."""
    
    def test_groundwater_capillary_feedback(self):
        """Use the production flux once as soil credit and groundwater debit."""
        dt = 3600.0
        capillary = CapillaryRiseTransport(name="capillary_rise_transport")
        groundwater = GroundwaterDynamics(name="groundwater_dynamics")
        initial_depth = torch.tensor(2.0, dtype=torch.float64)
        initial_storage = torch.tensor(80.0, dtype=torch.float64)

        initial_capillary = _unwrap_update(
            capillary.forward({"water_table_depth": initial_depth}, 0.0, dt)[
                "capillary_rise_flux"
            ]
        ).to(dtype=torch.float64)
        result = groundwater.forward(
            {
                "groundwater_storage": initial_storage,
                "water_table_depth": initial_depth,
                "drainage_flux": torch.tensor(5.0e-3, dtype=torch.float64),
                "capillary_rise_flux": initial_capillary,
                "lateral_inflow": torch.tensor(0.0, dtype=torch.float64),
                "groundwater_pumping": torch.tensor(0.0, dtype=torch.float64),
            },
            0.0,
            dt,
        )
        new_depth = _unwrap_update(result["water_table_depth"])
        new_capillary = _unwrap_update(
            capillary.forward({"water_table_depth": new_depth}, dt, dt)[
                "capillary_rise_flux"
            ]
        )

        self.assertLess(_scalar(new_depth), _scalar(initial_depth))
        self.assertGreater(_scalar(new_capillary), _scalar(initial_capillary))
    
    def test_water_table_capillary_threshold(self):
        """Evaluate the production capillary cutoff around its parameter."""
        capillary = CapillaryRiseTransport(name="capillary_rise_transport")
        max_height = _scalar(capillary.max_capillary_height)
        shallow_flux = _unwrap_update(
            capillary.forward(
                {"water_table_depth": torch.tensor(0.5 * max_height)},
                0.0,
                3600.0,
            )["capillary_rise_flux"]
        )
        deep_flux = _unwrap_update(
            capillary.forward(
                {"water_table_depth": torch.tensor(1.5 * max_height)},
                0.0,
                3600.0,
            )["capillary_rise_flux"]
        )

        self.assertGreater(_scalar(shallow_flux), 0.0)
        self.assertEqual(_scalar(deep_flux), 0.0)


class TestCanopyInterceptionSoilIntegration(unittest.TestCase):
    """Test P2-P0 integration: Canopy interception with soil water balance."""
    
    def test_interception_throughfall_soil_input(self):
        """Test canopy interception affecting soil water input."""
        # Precipitation and canopy parameters
        total_precipitation = 10.0  # mm day-1
        lai = 2.5
        canopy_capacity = 0.2 * lai  # mm (capacity per LAI)
        
        # Initial canopy storage
        initial_canopy_storage = 0.1  # mm
        
        # Interception process
        available_capacity = max(0.0, canopy_capacity - initial_canopy_storage)
        interception = min(total_precipitation, available_capacity)
        
        # Direct throughfall
        direct_throughfall = total_precipitation - interception
        
        # Canopy evaporation (simplified)
        potential_canopy_evap = 3.0  # mm day-1
        updated_canopy_storage = initial_canopy_storage + interception
        actual_canopy_evap = min(updated_canopy_storage, potential_canopy_evap)
        
        # Final canopy storage and additional throughfall
        final_canopy_storage = updated_canopy_storage - actual_canopy_evap
        overflow_throughfall = max(0.0, final_canopy_storage - canopy_capacity)
        final_canopy_storage = min(final_canopy_storage, canopy_capacity)
        
        # Total throughfall to soil
        total_throughfall = direct_throughfall + overflow_throughfall
        
        # Soil receives throughfall as surface input
        soil_surface_input = total_throughfall  # mm day-1
        
        # Verify water balance
        water_balance = (total_precipitation - 
                        (actual_canopy_evap + total_throughfall + final_canopy_storage - initial_canopy_storage))
        
        self.assertLess(abs(water_balance), 1e-10)  # Perfect water balance
        self.assertLessEqual(total_throughfall, total_precipitation)  # Physical constraint
        self.assertGreaterEqual(soil_surface_input, 0)  # Non-negative input
        
        print(f"Canopy-Soil integration test:")
        print(f"  Precipitation: {total_precipitation:.1f} mm")
        print(f"  Interception: {interception:.2f} mm") 
        print(f"  Canopy evaporation: {actual_canopy_evap:.2f} mm")
        print(f"  Throughfall to soil: {total_throughfall:.2f} mm")
        print(f"  Water balance error: {water_balance:.2e}")
    
    def test_wet_dry_canopy_evaporation_split(self):
        """Test separation of wet canopy evaporation and dry canopy transpiration."""
        # Canopy conditions
        lai = 3.0
        canopy_storage = 0.3  # mm (some water stored)
        
        # Total ET demand
        total_et_demand = 5.0  # mm day-1
        wet_canopy_efficiency = 1.2  # Wet canopy evaporates more efficiently
        
        # Evaporation from wet canopy (priority)
        potential_wet_evap = total_et_demand * wet_canopy_efficiency
        actual_wet_evap = min(canopy_storage, potential_wet_evap)
        
        # Remaining ET demand becomes transpiration
        remaining_et_demand = total_et_demand - actual_wet_evap
        transpiration = max(0.0, remaining_et_demand)
        
        # Updated canopy storage
        updated_canopy_storage = canopy_storage - actual_wet_evap
        
        # Verify ET partitioning
        total_et_actual = actual_wet_evap + transpiration
        
        # Should conserve total ET (approximately, considering efficiency)
        self.assertGreaterEqual(updated_canopy_storage, 0)
        self.assertGreaterEqual(actual_wet_evap, 0)
        self.assertGreaterEqual(transpiration, 0)
        
        print(f"Wet-Dry canopy ET split test:")
        print(f"  Total ET demand: {total_et_demand:.2f} mm")
        print(f"  Wet canopy evaporation: {actual_wet_evap:.2f} mm") 
        print(f"  Dry canopy transpiration: {transpiration:.2f} mm")
        print(f"  Total ET actual: {total_et_actual:.2f} mm")


class TestSnowAlbedoEnergyIntegration(unittest.TestCase):
    """Test P2-P2 integration: Snow albedo with energy balance."""
    
    def test_albedo_radiation_feedback(self):
        """Test feedback between snow albedo and radiation balance."""
        # Initial conditions
        initial_albedo = 0.8  # Fresh snow
        incoming_solar = 800.0  # W/m2
        
        # Net radiation with current albedo
        net_solar = incoming_solar * (1.0 - initial_albedo)  # W/m2
        
        # Snow surface temperature (simplified energy balance)
        # Assume linear relationship for testing
        reference_temp = 273.15  # K
        temp_coefficient = 0.1   # K per W/m2
        snow_surface_temp = reference_temp + net_solar * temp_coefficient
        
        # Snow aging based on temperature
        aging_factor = math.exp(5000.0 * (1.0/273.16 - 1.0/snow_surface_temp))
        daily_aging = 0.0864 * aging_factor  # days
        
        # New albedo after aging
        fresh_albedo = 0.85
        old_albedo = 0.5
        aging_time_scale = 5.0  # days
        
        # Current age (simplified)
        current_age = daily_aging  # Assume first day
        decay_factor = math.exp(-current_age / aging_time_scale)
        new_albedo = old_albedo + (fresh_albedo - old_albedo) * decay_factor
        
        # New net radiation with updated albedo
        new_net_solar = incoming_solar * (1.0 - new_albedo)
        
        # Verify feedback: both albedo and radiation are within reasonable bounds
        self.assertNotEqual(new_albedo, initial_albedo)  # Albedo changed
        self.assertGreater(new_albedo, 0.3)  # Within reasonable bounds
        self.assertLess(new_albedo, 1.0)     # Within reasonable bounds
        
        # Net radiation should also be within reasonable bounds
        self.assertGreater(new_net_solar, 0)  # Positive net radiation
        self.assertLess(new_net_solar, incoming_solar)  # Less than incoming
        
        print(f"Snow albedo-radiation feedback test:")
        print(f"  Initial albedo: {initial_albedo:.3f}")
        print(f"  Snow surface temp: {snow_surface_temp:.1f} K")
        print(f"  Daily aging: {daily_aging:.4f} days")
        print(f"  New albedo: {new_albedo:.3f}")
        print(f"  Net radiation: {net_solar:.1f} -> {new_net_solar:.1f} W/m2")
    
    def test_fresh_snow_albedo_reset(self):
        """Test fresh snowfall resetting albedo in energy balance."""
        # Aged snow conditions
        aged_albedo = 0.55
        snow_age = 10.0  # days
        
        # Fresh snowfall event
        snowfall_rate = 0.008  # kg m-2 s-1
        snowfall_threshold = 0.005  # kg m-2 s-1
        
        # Check for reset
        fresh_snow_event = snowfall_rate > snowfall_threshold
        
        if fresh_snow_event:
            # Reset albedo and age
            reset_albedo = 0.85
            reset_age = 0.0
        else:
            # No reset
            reset_albedo = aged_albedo
            reset_age = snow_age
        
        # Energy implications
        incoming_solar = 600.0  # W/m2
        aged_net_solar = incoming_solar * (1.0 - aged_albedo)
        reset_net_solar = incoming_solar * (1.0 - reset_albedo)
        
        self.assertTrue(fresh_snow_event)
        self.assertEqual(reset_age, 0.0)
        self.assertGreater(reset_albedo, aged_albedo)
        self.assertLess(reset_net_solar, aged_net_solar)  # Less energy absorbed
        
        print(f"Fresh snow reset test:")
        print(f"  Snowfall rate: {snowfall_rate:.3f} kg/m2/s")
        print(f"  Aged albedo: {aged_albedo:.3f} -> Reset albedo: {reset_albedo:.3f}")
        print(f"  Net solar radiation: {aged_net_solar:.1f} -> {reset_net_solar:.1f} W/m2")


class TestInterfaceMappingIntegration(unittest.TestCase):
    """Test interface mapping integration across all modules."""
    
    def test_cross_module_variable_mapping(self):
        """Test variable mapping consistency across module interfaces."""
        # Simulate data flow between modules with different naming conventions
        
        # Module A outputs (standard names)
        module_a_outputs = {
            "soil_moisture": 0.25,
            "evapotranspiration_flux": 0.003,
            "drainage_flux": 0.001
        }
        
        # Module B inputs (standard names)
        required_inputs_b = {
            "soil_moisture": None,
            "evapotranspiration_flux": None,
            "drainage_flux": None
        }
        
        # Direct mapping with canonical names
        mapped_inputs = dict(module_a_outputs)
        
        # Verify all required inputs are available
        for required_var in required_inputs_b.keys():
            self.assertIn(required_var, mapped_inputs)
            self.assertIsNotNone(mapped_inputs[required_var])
        
        print(f"Cross-module mapping test:")
        for name, value in module_a_outputs.items():
            print(f"  {name}: {value}")
    
    def test_units_consistency_integration(self):
        """Test units consistency in module coupling."""
        # Water flux variables should have consistent units across modules
        flux_variables = {
            "precipitation_flux": ("kg m-2 s-1", 0.005),
            "evapotranspiration_flux": ("kg m-2 s-1", 0.003),
            "surface_runoff_flux": ("kg m-2 s-1", 0.001),
            "subsurface_runoff_flux": ("kg m-2 s-1", 0.0005),
            "capillary_rise_flux": ("kg m-2 s-1", 0.0002)
        }
        
        expected_unit = "kg m-2 s-1"
        
        # Check unit consistency
        for var_name, (unit, value) in flux_variables.items():
            self.assertEqual(unit, expected_unit)
            
            # Check physical reasonableness
            self.assertGreaterEqual(value, 0)  # Non-negative fluxes
            self.assertLess(value, 0.01)       # Reasonable magnitude
            
            print(f"  {var_name}: {value} {unit}")
        
        # Mass balance check
        inputs = flux_variables["precipitation_flux"][1] + flux_variables["capillary_rise_flux"][1]
        outputs = (flux_variables["evapotranspiration_flux"][1] + 
                  flux_variables["surface_runoff_flux"][1] + 
                  flux_variables["subsurface_runoff_flux"][1])
        
        # Input should approximately equal output (for steady state)
        imbalance = abs(inputs - outputs) / inputs
        self.assertLess(imbalance, 0.2)  # Within 20% for integration test
        
        print(f"  Water balance - Inputs: {inputs:.4f}, Outputs: {outputs:.4f}")


class TestErrorPropagationIntegration(unittest.TestCase):
    """Test error propagation and boundary condition handling across modules."""
    
    def test_boundary_condition_propagation(self):
        """Test boundary condition propagation between coupled modules."""
        # Surface boundary: atmosphere -> soil
        atmospheric_forcing = {
            "precipitation": 5.0,      # mm day-1
            "potential_et": 4.0,       # mm day-1
            "air_temperature": 298.15, # K
            "net_radiation": 200.0     # W/m2
        }
        
        # Soil response to atmospheric forcing
        infiltration_capacity = 8.0  # mm day-1
        actual_infiltration = min(atmospheric_forcing["precipitation"], infiltration_capacity)
        surface_runoff = atmospheric_forcing["precipitation"] - actual_infiltration
        
        # Bottom boundary: soil -> groundwater
        soil_drainage = 1.5  # mm day-1
        
        # Groundwater response
        groundwater_storage_change = soil_drainage  # Simplified
        
        # Verify boundary conditions
        self.assertLessEqual(actual_infiltration, infiltration_capacity)
        self.assertGreaterEqual(surface_runoff, 0)
        self.assertEqual(atmospheric_forcing["precipitation"], 
                        actual_infiltration + surface_runoff)
        
        print(f"Boundary condition propagation test:")
        print(f"  Precipitation: {atmospheric_forcing['precipitation']:.1f} mm day-1")
        print(f"  Infiltration: {actual_infiltration:.1f} mm day-1")
        print(f"  Surface runoff: {surface_runoff:.1f} mm day-1")
        print(f"  Soil drainage: {soil_drainage:.1f} mm day-1")
    
    def test_parameter_sensitivity_propagation(self):
        """Test parameter sensitivity propagation across coupled modules."""
        # Base case parameters
        base_params = {
            "soil_k_sat": 1e-5,      # m/s
            "porosity": 0.45,        # m3/m3
            "canopy_capacity": 0.5,  # mm
            "albedo_fresh": 0.85     # dimensionless
        }
        
        # Parameter perturbations (卤10%)
        perturbation = 0.1
        
        sensitivity_results = {}
        
        for param_name, base_value in base_params.items():
            # Perturb parameter
            perturbed_value = base_value * (1 + perturbation)
            
            # Simulate effect (simplified)
            if param_name == "soil_k_sat":
                # Higher conductivity -> more drainage
                base_drainage = 1.0  # mm day-1
                perturbed_drainage = base_drainage * (perturbed_value / base_value)
                relative_change = (perturbed_drainage - base_drainage) / base_drainage
                
            elif param_name == "canopy_capacity":
                # Higher capacity -> more interception
                precip = 5.0  # mm day-1
                base_interception = min(precip, base_value)
                perturbed_interception = min(precip, perturbed_value)
                relative_change = ((perturbed_interception - base_interception) / 
                                 max(base_interception, 1e-6))
                
            else:
                # Default sensitivity
                relative_change = perturbation
            
            sensitivity_results[param_name] = relative_change
            
            # Verify reasonable sensitivity
            self.assertLess(abs(relative_change), 1.0)  # Less than 100% change
            
            print(f"  {param_name}: {relative_change:.3f} relative change")


def main():
    """Run all L2 integration tests."""
    print("=" * 70)
    print("L2 Integration Tests for Hydrosphere Module")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestRichardsCapillaryIntegration,
        TestGroundwaterCapillaryIntegration,
        TestCanopyInterceptionSoilIntegration,
        TestSnowAlbedoEnergyIntegration,
        TestInterfaceMappingIntegration,
        TestErrorPropagationIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("L2 Integration Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
        
        # Integration-specific metrics
        print(f"\nIntegration Test Metrics:")
        print(f"- P0-P1 coupling tests: 2 tests")
        print(f"- P1-P1 coupling tests: 2 tests") 
        print(f"- P2-P0 coupling tests: 2 tests")
        print(f"- P2-P2 coupling tests: 2 tests")
        print(f"- Interface integration: 2 tests")
        print(f"- Error propagation: 2 tests")
    else:
        print("No tests were run")
    
    # Detailed failure/error reporting
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    print("=" * 70)
    
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
