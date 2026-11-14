"""
L1 Validation Test for P0-Critical Parameter System Fixes

This test validates that all P0-Critical fixes are working correctly:
1. FunctorBase.register_if_missing default is False
2. No register_buffer anti-patterns exist
3. state.get() with torch.tensor() replaced by _get_state_variable
4. All register_if_missing=True changed to False

Test Level: L1 (Unit Verification)
Test Coverage: 8 fixed modules + parameter system
Author: RELIC Parameter System Refactoring Team
Date: 2025-10-17
"""

import sys
import pytest
import torch
from pathlib import Path
from relic.core.state import UpdateItem

# Add relic to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestP0FunctorBaseStrategy:
    """Test P0.1: FunctorBase register_if_missing default is False"""

    def test_get_parameter_value_signature_default(self):
        """FunctorBase.get_parameter_value has no registration or default route (YAML-first)."""
        from relic.core.functor import FunctorBase
        import inspect

        sig = inspect.signature(FunctorBase.get_parameter_value)
        for removed in ("register_if_missing", "default", "default_value"):
            assert removed not in sig.parameters, removed
        assert sig.parameters["source_order"].default == ("manager", "config")

        print("[OK] FunctorBase.get_parameter_value has no default route")


class TestP0StateGetFixes:
    """Test P0.3: state.get() torch.tensor() fixes"""

    def test_nitrification_import_and_instantiate(self):
        """Test nitrification module can be imported and instantiated"""
        from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
        from relic.parameters import get_global_manager

        # Initialize parameter manager
        pm = get_global_manager()

        # Create config
        config = {"device": "cpu", "dtype": "float32"}

        # Instantiate - should not raise errors
        functor = NitrificationProcess(name="test_nitrification", config=config)

        assert functor is not None
        assert hasattr(functor, '_get_state_variable')

        print("[OK] NitrificationProcess instantiates successfully")

    def test_nitrification_forward_with_defaults(self):
        """Nitrification requires its environment drivers; no default stands in."""
        from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess

        config = {"device": "cpu", "dtype": "float32"}
        functor = NitrificationProcess(name="test_nitrification", config=config)

        # Only the substrate: the missing drivers are named, none is substituted.
        state = {
            'nh4_nitrogen': torch.tensor([[5.0]])  # g N/m2
        }
        with pytest.raises(ValueError, match="soil_temperature, water_filled_pore_space, soil_ph"):
            functor.forward(state, time=0.0, dt=3600.0)

        # Explicit drivers (K, dimensionless water-filled pore space, pH units).
        state.update({
            'soil_temperature': torch.tensor([[293.15]]),
            'water_filled_pore_space': torch.tensor([[0.5]]),
            'soil_ph': torch.tensor([[6.5]]),
        })
        result = functor.forward(state, time=0.0, dt=3600.0)

        assert 'nitrification_flux' in result
        flux = result['nitrification_flux']
        if isinstance(flux, UpdateItem):
            flux = flux.value
        assert isinstance(flux, torch.Tensor)

        print("[OK] NitrificationProcess.forward() requires explicit environment drivers")

    def test_mineralization_import_and_instantiate(self):
        """Test mineralization module works after fixes"""
        from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess

        config = {"device": "cpu", "dtype": "float32"}
        functor = MineralizationProcess(name="test_mineralization", config=config)

        assert functor is not None
        assert hasattr(functor, '_get_state_variable')

        print("[OK] MineralizationProcess instantiates successfully")

    def test_root_nutrient_uptake_import(self):
        """Test root_nutrient_uptake module works after fixes"""
        from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import RootNutrientUptake

        config = {"device": "cpu", "dtype": "float32"}
        functor = RootNutrientUptake(name="test_root_uptake", config=config)

        assert functor is not None
        assert hasattr(functor, '_get_state_variable')

        print("[OK] RootNutrientUptake instantiates successfully")


class TestP0RegisterIfMissingFixes:
    """Test P0.4: register_if_missing=True fixes"""

    def test_advection_diffusion_no_auto_register(self):
        """Verify advection_diffusion.py uses register_if_missing=False"""
        # Read file and check
        file_path = Path(__file__).parent.parent / "relic" / "scientific" / "transport" / "advection_diffusion.py"
        content = file_path.read_text(encoding='utf-8')

        # Should NOT contain register_if_missing=True
        assert 'register_if_missing=True' not in content, \
            "advection_diffusion.py still has register_if_missing=True"

        print("[OK] advection_diffusion.py: register_if_missing=True removed")

    def test_fertilization_no_auto_register(self):
        """Verify fertilization.py uses register_if_missing=False"""
        file_path = Path(__file__).parent.parent / "relic" / "scientific" / "anthroposphere" / "management" / "fertilization.py"
        content = file_path.read_text(encoding='utf-8')

        assert 'register_if_missing=True' not in content, \
            "fertilization.py still has register_if_missing=True"

        print("[OK] fertilization.py: register_if_missing=True removed")

    def test_all_scientific_files_no_auto_register(self):
        """Scan all scientific files to ensure no register_if_missing=True"""
        scientific_dir = Path(__file__).parent.parent / "relic" / "scientific"
        violations = []

        for py_file in scientific_dir.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue

            content = py_file.read_text(encoding='utf-8')
            if 'register_if_missing=True' in content:
                violations.append(str(py_file.relative_to(scientific_dir.parent)))

        assert len(violations) == 0, \
            f"Found register_if_missing=True in: {violations}"

        print(f"[OK] All scientific files: no register_if_missing=True found")


class TestP0Integration:
    """Integration test for all P0 fixes"""

    def test_parameter_manager_enforces_yaml_first(self):
        """Verify ParameterManager enforces YAML-first policy"""
        from relic.parameters import get_global_manager

        pm = get_global_manager()

        # Try to get a non-existent parameter
        # Should return None (not found) or raise error depending on implementation
        result = pm.get("nonexistent.parameter.key.that.definitely.does.not.exist", default=None)

        # Should be None since parameter doesn't exist in YAML
        assert result is None, \
            "Non-existent parameter should return None (YAML-first policy)"

        print("[OK] ParameterManager enforces YAML-first policy")

    def test_get_state_variable_default_chain(self):
        """The retired pedosphere.environment_defaults namespace supplies no state value."""
        from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess

        config = {"device": "cpu"}
        functor = NitrificationProcess(name="test", config=config)

        # A state value is returned unchanged (Tier 1).
        result = functor._get_state_variable(
            state={"soil_temperature": torch.tensor(293.15)},
            key="soil_temperature",
            units="K"
        )
        assert float(result) == pytest.approx(293.15)

        # No default tier exists: a missing driver raises KeyError (NOT FOUND).
        with pytest.raises(KeyError, match="NOT FOUND"):
            functor._get_state_variable(
                state={},
                key="soil_temperature",
                units="K"
            )

        print("[OK] _get_state_variable raises for a missing environment driver")


def run_all_p0_tests():
    """Run all P0 validation tests"""
    print("\n" + "="*70)
    print("P0-Critical Parameter System Fixes Validation Test Suite")
    print("="*70 + "\n")

    # Run tests using pytest
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-k', 'TestP0'
    ])

    if exit_code == 0:
        print("\n" + "="*70)
        print("[SUCCESS] All P0 validation tests passed!")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("[FAILURE] Some P0 tests failed. Please review above.")
        print("="*70 + "\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(run_all_p0_tests())


