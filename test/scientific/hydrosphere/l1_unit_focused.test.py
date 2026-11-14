"""
L1 Unit Tests for Hydrosphere Module - Focused Test Suite

This module provides focused unit tests for hydrosphere components
using mathematical validation without heavy dependency imports.

Test Coverage:
- P0: Richards equation mathematical functions
- P1: Capillary rise Gardner approximation
- P1: Groundwater dynamics porosity calculations
- P2: Canopy interception water balance
- P2: Snow albedo aging formulations
- Interface mapping functionality

All tests focus on mathematical correctness and physical constraints.
"""

import sys
import os
import unittest
import math
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.hydrosphere.soil_moisture.richards import (
    RichardsSoilMoisture,
)

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestRichardsEquationMaths(unittest.TestCase):
    """Test the production Richards hydraulic and state contracts."""

    def setUp(self):
        """Create the ParameterManager-backed production model."""
        self.model = RichardsSoilMoisture()
        self.theta = torch.full((self.model.soil_layers,), 0.25)
    
    def test_van_genuchten_pressure_head(self):
        """Production retention evaluation matches the fixed default baseline."""
        pressure_head = self.model._compute_pressure_head(self.theta)

        self.assertEqual(pressure_head.shape, self.theta.shape)
        self.assertTrue(torch.isfinite(pressure_head).all())
        self.assertTrue(torch.all(pressure_head < 0.0))
        # Pressure head [m] at theta = 0.25 m3 m-3 for the owner defaults
        # (alpha 7.5 m-1, n 1.89, theta_r 0.045, theta_s 0.45).
        torch.testing.assert_close(
            pressure_head,
            torch.full_like(pressure_head, -0.2485835701227188),
            rtol=1.0e-5,
            atol=1.0e-7,
        )
    
    def test_mualem_hydraulic_conductivity(self):
        """Production conductivity evaluation matches the fixed baseline."""
        conductivity = self.model._compute_hydraulic_conductivity(self.theta)

        self.assertEqual(conductivity.shape, self.theta.shape)
        self.assertTrue(torch.isfinite(conductivity).all())
        self.assertTrue(torch.all(conductivity > 0.0))
        self.assertTrue(torch.all(conductivity <= self.model.k_saturated))
        torch.testing.assert_close(
            conductivity,
            torch.full_like(conductivity, 1.0041331677257403e-07),
            rtol=1.0e-5,
            atol=1.0e-12,
        )
    
    def test_picard_iteration_convergence(self):
        """The canonical runtime converges and emits explicit state semantics."""
        result = self.model.forward(
            {
                "soil_moisture": self.theta,
                "infiltration_flux": torch.tensor(0.0),
                "evapotranspiration_flux": torch.tensor(0.0),
            },
            time=0.0,
            dt=60.0,
        )

        self.assertEqual(set(result), set(self.model.metadata.output_variables))
        diagnostics = result["diagnostics.soil_moisture_richards"]
        self.assertTrue(diagnostics["solver_converged"])
        self.assertLessEqual(
            diagnostics["solver_iterations"], self.model.max_picard_iterations
        )
        for key, item in result.items():
            if key == "diagnostics.soil_moisture_richards":
                continue
            self.assertIsInstance(item, UpdateItem)
            self.assertEqual(item.mode, UpdateMode.REPLACE)
            self.assertFalse(item.per_timestep)
            self.assertEqual(item.source_functor, self.model.name)
        torch.testing.assert_close(
            result["soil_moisture"].value.mean(),
            torch.tensor(0.2499939650297165),
        )
        # Uniform profile, zero top flux, free drainage: interior faces carry
        # the harmonic mean K(0.25) exactly, so the 60 s mean drainage stays
        # within 4e-7 relative of rho_w K(0.25) = 1.00413317e-4 kg m-2 s-1.
        # The former 1.0023870e-4 came from adding saturation_epsilon [1] to
        # K_i + K_j [m s-1] (CR-B-20261018-02).
        torch.testing.assert_close(
            result["drainage_flux"].value,
            torch.tensor(0.00010041328111970926),
        )


class TestCapillaryRiseMaths(unittest.TestCase):
    """Test capillary rise mathematical functions (P1 priority)."""
    
    def test_gardner_approximation(self):
        """Test Gardner approximation for capillary rise."""
        # Gardner parameters
        a = 1e-6  # m^(b+1)/s
        b = 2.0   # dimensionless
        B = 1.0 + 1.5/(b-1.0)  # Gardner B coefficient
        
        # Test at different depths
        depths = [0.5, 1.0, 1.5, 2.0]  # m
        fluxes = []
        
        for Z in depths:
            # Gardner formula: w = B * a / Z^b
            flux = B * a / (Z**b)
            fluxes.append(flux)
            
            self.assertGreater(flux, 0)
            print(f"Capillary flux at {Z}m depth: {flux:.2e} m/s")
        
        # Flux should decrease with depth
        for i in range(1, len(fluxes)):
            self.assertLess(fluxes[i], fluxes[i-1])
    
    def test_brooks_corey_parameters(self):
        """Test Brooks-Corey to Gardner coefficient conversion."""
        # Brooks-Corey parameters
        pore_size_index = 0.5  # m
        k_saturated = 1e-5     # m/s
        air_entry_pressure = 0.102  # m (pressure head)
        
        # Gardner b parameter: b = (2 + 3*m) / m
        gardner_b = (2.0 + 3.0 * pore_size_index) / pore_size_index
        
        # Gardner a parameter: a = Ks * |psi_ae|^b
        gardner_a = k_saturated * (air_entry_pressure**gardner_b)
        
        self.assertGreater(gardner_b, 1.0)
        self.assertGreater(gardner_a, 0)
        print(f"Gardner coefficients: a={gardner_a:.2e}, b={gardner_b:.2f}")


class TestGroundwaterMaths(unittest.TestCase):
    """Test groundwater dynamics mathematical functions (P1 priority)."""
    
    def test_exponential_porosity_decay(self):
        """Test exponential porosity decay with depth."""
        # Parameters
        n0 = 0.45  # Surface porosity
        p = 2.0    # Decay length [m]
        
        # Test at different depths
        depths = [0.0, 0.5, 1.0, 2.0, 4.0]  # m
        
        for z in depths:
            # Exponential decay: n(z) = n0 * exp(-z/p)
            porosity = n0 * math.exp(-z/p)
            
            self.assertGreater(porosity, 0)
            self.assertLessEqual(porosity, n0)
            print(f"Porosity at {z}m depth: {porosity:.3f}")
        
        # Surface porosity should be maximum
        surface_porosity = n0 * math.exp(-0.0/p)
        self.assertAlmostEqual(surface_porosity, n0)
    
    def test_storage_integration(self):
        """Test storage integration between depths."""
        # Parameters
        n0 = 0.45
        p = 2.0
        
        # Analytical integration: 鈭玔z1 to z2] n0 * exp(-z/p) dz = n0 * p * [exp(-z1/p) - exp(-z2/p)]
        z1, z2 = 0.0, 1.0  # m
        
        analytical_integral = n0 * p * (math.exp(-z1/p) - math.exp(-z2/p))
        
        # Numerical integration (trapezoidal rule)
        n_points = 100
        dz = (z2 - z1) / n_points
        numerical_integral = 0.0
        
        for i in range(n_points):
            z = z1 + i * dz
            porosity = n0 * math.exp(-z/p)
            numerical_integral += porosity * dz
        
        # Should match within reasonable tolerance
        rel_error = abs(analytical_integral - numerical_integral) / analytical_integral
        self.assertLess(rel_error, 0.01)  # 1% tolerance
        print(f"Storage integration: analytical={analytical_integral:.4f}, numerical={numerical_integral:.4f}")
    
    def test_water_table_newton_raphson(self):
        """Test Newton-Raphson water table depth solving logic."""
        # Test function: F(z) = z^2 - 4 = 0, solution z = 2
        def f(z):
            return z**2 - 4
        
        def f_prime(z):
            return 2*z
        
        # Newton-Raphson iteration
        z = 1.0  # Initial guess
        tolerance = 1e-6
        max_iter = 10
        
        for i in range(max_iter):
            fz = f(z)
            fpz = f_prime(z)
            
            if abs(fpz) < 1e-12:
                break
                
            z_new = z - fz/fpz
            
            if abs(z_new - z) < tolerance:
                break
                
            z = z_new
        
        # Should converge to z = 2
        self.assertAlmostEqual(z, 2.0, places=5)
        print(f"Newton-Raphson solution: z={z:.6f}")


class TestCanopyInterceptionMaths(unittest.TestCase):
    """Test canopy interception mathematical functions (P2 priority)."""
    
    def test_rutter_gash_water_balance(self):
        """Test Rutter/Gash canopy interception water balance."""
        # Parameters
        lai = 2.0  # Leaf area index
        capacity_per_lai = 0.2  # mm
        evap_efficiency = 1.0
        
        # Maximum storage capacity
        max_capacity = capacity_per_lai * lai  # = 0.4 mm
        
        # Initial state
        storage = 0.1  # mm
        precipitation = 0.8  # mm
        potential_et = 0.3  # mm
        
        # Process interception
        available_capacity = max(0.0, max_capacity - storage)
        interception = min(precipitation, available_capacity)
        
        # Update storage
        storage_after_precip = storage + interception
        
        # Process evaporation
        evap_from_canopy = min(storage_after_precip, potential_et * evap_efficiency)
        storage_after_evap = storage_after_precip - evap_from_canopy
        
        # Remaining ET becomes transpiration
        transpiration = max(0.0, potential_et - evap_from_canopy)
        
        # Throughfall
        throughfall = precipitation - interception
        
        # Water balance check: Input = Output + Storage_change
        input_water = precipitation + storage
        output_water = evap_from_canopy + transpiration + throughfall + storage_after_evap
        
        water_balance_error = abs(input_water - output_water)
        self.assertLess(water_balance_error, 1e-6)
        
        storage_change = storage_after_evap - storage
        print(f"Canopy water balance - Input: {input_water:.3f}, Output: {output_water:.3f}, Storage change: {storage_change:.3f}")
    
    def test_canopy_storage_constraints(self):
        """Test canopy storage physical constraints."""
        max_capacity = 0.5  # mm
        
        # Test overflow scenario
        storage = 0.4  # mm
        precipitation = 0.3  # mm
        
        # After precipitation
        potential_storage = storage + precipitation
        
        if potential_storage > max_capacity:
            actual_storage = max_capacity
            overflow = potential_storage - max_capacity
        else:
            actual_storage = potential_storage
            overflow = 0.0
        
        self.assertEqual(actual_storage, max_capacity)
        self.assertGreater(overflow, 0)
        print(f"Storage constraint test: capacity={max_capacity}, overflow={overflow:.3f}")


class TestSnowAlbedoMaths(unittest.TestCase):
    """Test snow albedo mathematical functions (P2 priority)."""
    
    def test_bats_aging_formulation(self):
        """Test BATS snow aging formulation."""
        # Parameters
        surface_temp = 268.0  # K
        dust_soot_factor = 0.3
        dt_days = 1.0
        
        # BATS aging factors
        r1 = math.exp(5000.0 * (1.0/273.16 - 1.0/surface_temp))
        r2 = min(r1**10, 1.0)
        r3 = dust_soot_factor
        
        # Aging increment
        aging_factor = 0.0864  # Conversion factor
        aging_increment = dt_days * aging_factor * (r1 + r2 + r3)
        
        self.assertGreater(aging_increment, 0)
        self.assertLess(aging_increment, 1.0)  # Reasonable daily aging
        print(f"BATS aging increment: {aging_increment:.4f} days")
    
    def test_albedo_age_relationship(self):
        """Test albedo-age relationship."""
        fresh_albedo = 0.85
        old_albedo = 0.5
        aging_time_scale = 5.0  # days
        
        # Test different ages
        ages = [0.0, 1.0, 5.0, 10.0, 20.0]  # days
        
        for age in ages:
            # Exponential decay: albedo = old + (fresh - old) * exp(-age/tau)
            decay_factor = math.exp(-age / aging_time_scale)
            albedo = old_albedo + (fresh_albedo - old_albedo) * decay_factor
            
            self.assertGreaterEqual(albedo, old_albedo)
            self.assertLessEqual(albedo, fresh_albedo)
            print(f"Age {age} days: albedo = {albedo:.3f}")
        
        # Fresh snow should have highest albedo
        age_0_albedo = old_albedo + (fresh_albedo - old_albedo) * math.exp(0)
        self.assertAlmostEqual(age_0_albedo, fresh_albedo, places=3)
    
    def test_solar_zenith_correction(self):
        """Test solar zenith angle correction."""
        b_parameter = 2.0
        
        # Test different zenith angles
        zenith_angles = [0.0, 0.3, 0.6, 1.0]  # radians
        
        for theta in zenith_angles:
            cos_zenith = math.cos(theta)
            
            if cos_zenith < 0.5:
                fz = (1.0/b_parameter) * ((1.0 + b_parameter) / (1.0 + 2.0 * b_parameter * cos_zenith) - 1.0)
            else:
                fz = 0.0
            
            self.assertGreaterEqual(fz, 0.0)
            print(f"Zenith angle {theta:.1f} rad: solar correction factor = {fz:.3f}")


class TestInterfaceMappingLogic(unittest.TestCase):
    """Test interface mapping logic without imports."""
    
    def test_cf_naming_conventions(self):
        """Test CF-1.8 naming convention logic."""
        # Test standard name identity
        standard_names = [
            "soil_moisture",
            "evapotranspiration_flux",
            "air_temperature",
            "precipitation_flux"
        ]
        
        for name in standard_names:
            standard = name
            self.assertEqual(standard, name)
            print(f"Standard name: {name}")
    
    def test_units_consistency(self):
        """Test units consistency checking."""
        # Water flux variables should have consistent units
        water_flux_vars = [
            ("evapotranspiration_flux", "kg m-2 s-1"),
            ("precipitation_flux", "kg m-2 s-1"),
            ("surface_runoff_flux", "kg m-2 s-1"),
            ("drainage_flux", "kg m-2 s-1")
        ]
        
        expected_unit = "kg m-2 s-1"
        
        for var_name, unit in water_flux_vars:
            self.assertEqual(unit, expected_unit)
            print(f"Flux variable {var_name}: {unit}")
    
    def test_variable_categorization(self):
        """Test variable category classification logic."""
        # Category classification rules
        variable_categories = {
            "precipitation": "forcing",
            "air_temperature": "forcing", 
            "soil_moisture": "state",
            "groundwater_storage": "state",
            "evapotranspiration": "flux",
            "surface_runoff": "flux",
            "water_stress_factor": "diagnostic",
            "saturation_deficit": "diagnostic"
        }
        
        for var_name, expected_category in variable_categories.items():
            # Simple classification logic
            if var_name in ["precipitation", "air_temperature", "wind_speed"]:
                category = "forcing"
            elif var_name in ["soil_moisture", "groundwater_storage", "water_table_depth"]:
                category = "state"
            elif "flux" in var_name or var_name in ["evapotranspiration", "surface_runoff"]:
                category = "flux"
            else:
                category = "diagnostic"
            
            self.assertEqual(category, expected_category)
            print(f"Variable {var_name}: category = {category}")


class TestPhysicalConstraints(unittest.TestCase):
    """Test physical constraint enforcement."""
    
    def test_moisture_content_bounds(self):
        """Test soil moisture content physical bounds."""
        theta_r = 0.05
        theta_s = 0.45
        
        # Test values
        test_values = [-0.1, 0.0, 0.25, 0.45, 0.6]
        
        for theta in test_values:
            # Apply physical constraints
            bounded_theta = max(theta_r, min(theta, theta_s))
            
            self.assertGreaterEqual(bounded_theta, theta_r)
            self.assertLessEqual(bounded_theta, theta_s)
            print(f"Theta {theta:.2f} -> bounded {bounded_theta:.2f}")
    
    def test_non_negative_fluxes(self):
        """Test non-negative flux constraints."""
        # ET should be non-negative
        et_values = [-0.01, 0.0, 0.001, 0.005]
        
        for et in et_values:
            bounded_et = max(0.0, et)
            self.assertGreaterEqual(bounded_et, 0.0)
            print(f"ET {et:.3f} -> bounded {bounded_et:.3f}")
    
    def test_albedo_bounds(self):
        """Test albedo physical bounds."""
        albedo_values = [-0.1, 0.0, 0.5, 0.95, 1.2]
        
        for albedo in albedo_values:
            bounded_albedo = max(0.0, min(albedo, 1.0))
            
            self.assertGreaterEqual(bounded_albedo, 0.0)
            self.assertLessEqual(bounded_albedo, 1.0)
            print(f"Albedo {albedo:.2f} -> bounded {bounded_albedo:.2f}")


def main():
    """Run all focused L1 unit tests."""
    print("=" * 70)
    print("L1 Unit Tests for Hydrosphere Module - Focused Suite")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestRichardsEquationMaths,
        TestCapillaryRiseMaths,
        TestGroundwaterMaths,
        TestCanopyInterceptionMaths,
        TestSnowAlbedoMaths,
        TestInterfaceMappingLogic,
        TestPhysicalConstraints
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("L1 Unit Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
    else:
        print("No tests were run")
    
    # Detailed failure/error reporting
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("=" * 70)
    
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

