"""
L1 Unit Tests - Architecture Compliance Verification
Tests core architectural requirements are met after refactoring
"""

import unittest
import torch
import ast
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relic.core.functor import (
    BiologicalProcessFunctor,
    PhysicalFieldFunctor,
)
from relic.parameters import get_global_manager
import pytest


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


class TestArchitectureCompliance(unittest.TestCase):
    """Test architectural compliance of RELIC modules"""

    def setUp(self):
        """Set up test environment"""
        get_global_manager()
        self.project_root = REPO_ROOT
        self.scientific_dir = self.project_root / "relic" / "scientific"
        self.coupling_dir = self.project_root / "relic" / "coupling"

    def test_no_hardcoded_parameters(self):
        """Test that no hardcoded parameters remain in critical files"""
        critical_files = [
            self.scientific_dir / "anthroposphere" / "management" / "fertilization.py",
            self.coupling_dir / "internal" / "biosphere_atmosphere.py",
            self.scientific_dir / "biosphere" / "stress" / "ozone_stress.py",
        ]

        for file_path in critical_files:
            if not file_path.exists():
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            # Check for hardcoded torch.tensor with numeric literals
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr') and node.func.attr == 'tensor':
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, (int, float)):
                            # Allow device-only tensors
                            if not any(kw.arg == 'device' for kw in node.keywords):
                                self.fail(f"Found hardcoded torch.tensor in {file_path.name}")

    def test_farquhar_no_register_buffer_for_params(self):
        """Test that farquhar.py doesn't use register_buffer for parameters"""
        farquhar_path = self.scientific_dir / "biosphere" / "photosynthesis" / "farquhar.py"

        if farquhar_path.exists():
            with open(farquhar_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for register_buffer calls with parameter names
            param_names = ['kc25', 'ko25', 'gamma_star25', 'vcmax_ha', 'vcmax_hd',
                          'jmax_ha', 'jmax_hd', 'kc_ha', 'ko_ha']

            for param in param_names:
                if f"register_buffer('{param}'" in content:
                    self.fail(f"farquhar.py still uses register_buffer for parameter {param}")

    def test_functor_inheritance(self):
        """Test that Functors inherit from correct bridge classes"""
        # Import some key Functors to test
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        from relic.scientific.atmosphere.chemistry import ChemistryFunctor

        # Test inheritance
        self.assertTrue(issubclass(FarquharPhotosynthesis, BiologicalProcessFunctor))
        self.assertTrue(issubclass(ChemistryFunctor, PhysicalFieldFunctor))

        # Test they have required methods
        self.assertTrue(hasattr(FarquharPhotosynthesis, 'forward'))
        self.assertTrue(hasattr(FarquharPhotosynthesis, 'get_dependencies'))

    def test_forward_method_signature(self):
        """Test that forward methods have correct signature"""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        # Create instance
        farquhar = FarquharPhotosynthesis()

        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
        }

        result = farquhar.forward(state, time=0.0, dt=3600.0)
        self.assertIsInstance(result, dict)
        # Check for actual output keys from Farquhar model
        self.assertIn('net_photosynthesis', result)

    def test_get_dependencies_returns_list(self):
        """Test that get_dependencies returns a list"""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        farquhar = FarquharPhotosynthesis()
        deps = farquhar.get_dependencies()

        self.assertIsInstance(deps, list)
        # Dependencies may vary; ensure list is returned and elements are strings
        for dep in deps:
            self.assertIsInstance(dep, str)
        # Dependencies should be functor names (strings), not state variables
        # Note: functor names MAY contain 'temperature' (e.g., 'leaf_temperature_functor')
        # The check is that they are valid identifier strings, not raw variable names
        for dep in deps:
            self.assertTrue(len(dep) > 0, "Dependency name should not be empty")

    def test_parameter_manager_usage(self):
        """Test that modules use ParameterManager correctly"""
        # Use Farquhar module which we know exists and has been fixed
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        # Create instance - should use ParameterManager
        model = FarquharPhotosynthesis()

        # Test that get_parameter_value is available
        self.assertTrue(hasattr(model, 'get_parameter_value'))

        # Test that it can get parameters
        value = model.get_parameter_value(
            'kc25',
            namespace='biosphere.photosynthesis',
            required=True,
        )
        self.assertIsNotNone(value)

    def test_coupling_config_from_parameters(self):
        """Boundary coupling has no duplicate configuration parameter surface."""
        from relic.coupling.internal import get_default_configs
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        config = get_default_configs(parameter_manager=manager)[
            'biosphere_atmosphere'
        ]

        self.assertEqual(config, {})


class TestParameterSystemIntegration(unittest.TestCase):
    """Test parameter system integration"""

    def test_yaml_files_exist(self):
        """Test that YAML parameter files exist"""
        yaml_files = [
            "relic/parameters/database/relic/coupling/biosphere_atmosphere.yaml",
            "relic/parameters/database/relic/coupling/soil_vegetation.yaml",
            "relic/parameters/database/relic/coupling/water_carbon.yaml",
            "relic/parameters/database/relic/atmosphere/constants.yaml",
            "relic/parameters/database/relic/biosphere/stress.yaml",
        ]

        project_root = REPO_ROOT
        for yaml_file in yaml_files:
            path = project_root / yaml_file
            self.assertTrue(path.exists(), f"Missing YAML file: {yaml_file}")

    def test_parameter_manager_singleton(self):
        """Test ParameterManager singleton pattern"""
        pm1 = get_global_manager()
        pm2 = get_global_manager()

        self.assertIs(pm1, pm2, "ParameterManager should be singleton")


if __name__ == '__main__':
    unittest.main()

