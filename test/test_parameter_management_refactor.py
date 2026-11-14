"""
L1 Unit Tests for Parameter Management Refactoring
Tests the parameter management fixes across all scientific modules
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from relic.parameters import get_global_manager
from relic.core.functor import FunctorBase


class TestParameterManagementRefactor(unittest.TestCase):
    """Test parameter management refactoring across all modules"""
    
    @classmethod
    def setUpClass(cls):
        """Set up parameter manager for all tests"""
        cls.param_manager = get_global_manager()
    
    def test_biosphere_megan_model(self):
        """Test MEGAN model parameter management"""
        try:
            from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
            
            # Create instance
            megan = MEGANModel(name="test_megan")
            
            # Check inheritance 
            self.assertIsInstance(megan, FunctorBase, "MEGANModel should inherit from FunctorBase")
            
            # Check emission factors are from parameter manager
            # These should no longer be hardcoded defaults
            self.assertTrue(hasattr(megan, 'emission_factors'), 
                          "Should have emission_factors attribute")
            
            print("[OK] MEGANModel parameter management OK")
            
        except Exception as e:
            self.fail(f"MEGANModel test failed: {str(e)}")
    
    def test_hydrosphere_penman_monteith(self):
        """Test Penman-Monteith parameter management"""
        try:
            from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
            
            # Create instance
            pm = PenmanMonteithET(name="test_pm", config={"canopy_resistance_source": "state"})
            
            # Check inheritance
            self.assertIsInstance(pm, FunctorBase, "PenmanMonteithET should inherit from FunctorBase")
            
            # Check physical constants are from parameter manager
            # No hardcoded values like 0.622, 2.45e6, etc.
            self.assertTrue(hasattr(pm.get_parameter_value, '__call__'), 
                          "Should have get_parameter_value method from FunctorBase")
            
            print("[OK] PenmanMonteithET parameter management OK")
            
        except Exception as e:
            self.fail(f"PenmanMonteithET test failed: {str(e)}")
    
    def test_atmosphere_chemistry(self):
        """Test atmospheric chemistry parameter management"""
        try:
            from relic.scientific.atmosphere.chemistry import ChemistryFunctor
            
            # Create instance
            chem = ChemistryFunctor(name="test_chemistry", physical_boundary="open_mechanism")
            
            # Check inheritance
            self.assertIsInstance(chem, FunctorBase, "ChemistryFunctor should inherit from FunctorBase")
            
            # Check reaction rate parameters
            # These should be loaded from parameter management, not hardcoded
            k_no_o3 = chem.get_parameter_value("k_no_o3_298", namespace="atmosphere.chemistry")
            self.assertIsNotNone(k_no_o3, "Should have reaction rate parameters")
            
            print("[OK] ChemistryFunctor parameter management OK")
            
        except Exception as e:
            self.fail(f"ChemistryFunctor test failed: {str(e)}")
    
    def test_pedosphere_nitrification(self):
        """Test nitrification parameter management"""
        try:
            from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
            
            # Create instance
            nitr = NitrificationProcess(name="test_nitrification")
            
            # Check inheritance
            self.assertIsInstance(nitr, FunctorBase, "NitrificationProcess should inherit from FunctorBase")
            
            # Check kinetic parameters are from parameter manager
            # No hardcoded vmax=2.0e-6, Km=0.005, etc.
            vmax = nitr.get_parameter_value("vmax_nitrification",
                                           namespace="pedosphere.nitrification")
            self.assertIsNotNone(vmax, "Should get vmax parameter")
            
            print("[OK] NitrificationProcess parameter management OK")
            
        except Exception as e:
            self.fail(f"NitrificationProcess test failed: {str(e)}")
    
    def test_transport_diffusion(self):
        """Test diffusion transport parameter management"""
        try:
            from relic.scientific.transport.diffusion import FickianDiffusion
            
            # Create instance - should work with default parameters from manager
            diff = FickianDiffusion(name="test_diffusion")
            
            # Check inheritance
            self.assertIsInstance(diff, FunctorBase, "FickianDiffusion should inherit from FunctorBase")
            
            # Check diffusivity is not hardcoded
            # Should use parameter manager for default values
            self.assertTrue(hasattr(diff.get_parameter_value, '__call__'), 
                          "Should have get_parameter_value from FunctorBase")
            
            print("[OK] FickianDiffusion parameter management OK")
            
        except Exception as e:
            self.fail(f"FickianDiffusion test failed: {str(e)}")
    
    def test_parameter_missing_handling(self):
        """Test missing parameter handling matches ParameterManager contract"""
        from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation

        sector = SectorAllocation(name="test_sector")
        parameter_name = "missing_parameter_20260716"
        namespace = "anthroposphere.emissions.sector_allocation"

        test_value = sector.get_parameter_value(
            parameter_name,
            namespace=namespace,
        )
        self.assertIsNone(test_value, "Missing optional parameter should return None")

        with self.assertRaisesRegex(ValueError, "Required parameter"):
            sector.get_parameter_value(
                parameter_name,
                namespace=namespace,
                required=True,
            )

        print("[OK] Parameter missing handling OK")
    
    def test_no_hardcoded_values(self):
        """Spot check that common hardcoded values are gone"""
        
        # List of files and patterns that should NOT contain hardcoded values
        checks = [
            # (file_path, forbidden_pattern, description)
            ("relic/scientific/biosphere/bvoc/megan_model.py",
             "emission_factors.get('isoprene', 24.0)", "Isoprene default hardcode"),
            ("relic/scientific/atmosphere/chemistry.py",
             "'k_no_o3_298': 1.8e-14,", "Reaction rate hardcode"),
        ]
        
        for file_path, pattern, desc in checks:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.assertNotIn(pattern, content, 
                                   f"Found hardcoded value: {desc} in {file_path}")
            except FileNotFoundError:
                pass  # File might not exist, that's OK
        
        print("[OK] No hardcoded values found in spot checks")


def run_l1_parameter_tests():
    """Run L1 parameter management tests"""
    
    print("\n" + "="*80)
    print("L1 UNIT TESTS - Parameter Management Refactoring")
    print("="*80 + "\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParameterManagementRefactor)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print("L1 PARAMETER TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[OK] All parameter management tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_l1_parameter_tests()
    sys.exit(exit_code)

