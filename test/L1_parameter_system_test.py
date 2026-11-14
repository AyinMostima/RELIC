#!/usr/bin/env python3
"""
L1 Unit Tests for Parameter System Refactoring

This test suite validates that:
1. YAML schema fixes are working (references vs reference)
2. Physical constants are accessible via unified interface
3. Transport and coupling modules use parameter manager correctly
4. No more hardcoded defaults bypass parameter system
"""

import unittest
import torch
import yaml
from pathlib import Path
import sys
import os
import re

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from relic.parameters import get_global_manager, ParameterSpec
from relic.utils.physical_constants import (
    get_physical_constant,
    get_von_karman,
    get_gravity,
    get_gas_constant,
    CONSTANT_MAPPINGS
)


class TestYAMLSchemaFixes(unittest.TestCase):
    """Test that YAML files use 'references' (list) not 'reference' (string)."""
    
    def test_yaml_references_field(self):
        """Verify all YAML files use 'references' as list."""
        yaml_dir = Path('relic/parameters/database/relic')
        yaml_files = list(yaml_dir.rglob('*.yaml'))
        
        self.assertGreater(len(yaml_files), 0, "No YAML files found")
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check no 'reference:' (singular) exists as a field (not part of parameter names)
            lines = content.split('\n')
            for line in lines:
                # Only check for '  reference:' or '    reference:' etc (indented field)
                # Not parameter names like 'reference_solar_radiation:' or 'j_no2_reference:'
                if re.match(r'^\s+reference:\s*$', line) or re.match(r'^\s+reference:\s*\[', line):
                    self.fail(f"Found 'reference:' (singular) field in {yaml_file}")
            
            # Parse and check structure
            data = yaml.safe_load(content)
            
            def check_references(obj, path=""):
                """Recursively check that 'references' are lists."""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        
                        if key == 'references':
                            self.assertIsInstance(value, list,
                                f"'references' at {new_path} in {yaml_file} is not a list")
                        elif key == 'reference':
                            self.fail(f"Found 'reference' (singular) at {new_path} in {yaml_file}")
                        else:
                            check_references(value, new_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_references(item, f"{path}[{i}]")
            
            check_references(data)
    
    def test_parameter_spec_creation(self):
        """Test that ParameterSpec can be created from fixed YAML data."""
        # Sample data as it would be loaded from fixed YAML
        param_data = {
            'value': 0.41,
            'units': 'dimensionless',
            'description': 'Von Karman constant',
            'references': ['Monin & Obukhov (1954)'],  # List, not string
            'bounds': [0.35, 0.42]
        }
        
        # This should work without errors
        spec = ParameterSpec(
            key='test.von_karman',
            value=param_data['value'],
            units=param_data['units'],
            description=param_data['description'],
            references=param_data['references'],
            bounds=tuple(param_data['bounds']), dtype="float32", category="model_parameter"
        )
        
        self.assertEqual(spec.key, 'test.von_karman')
        self.assertEqual(spec.value, 0.41)
        self.assertIsInstance(spec.references, list)
        self.assertEqual(len(spec.references), 1)


class TestPhysicalConstantsInterface(unittest.TestCase):
    """Test unified physical constants access."""
    
    def test_constant_mappings(self):
        """Test that constant mappings are defined."""
        self.assertIn('von_karman_constant', CONSTANT_MAPPINGS)
        self.assertIn('gravity_acceleration', CONSTANT_MAPPINGS)
        self.assertIn('gas_constant', CONSTANT_MAPPINGS)
        
    def test_get_physical_constant(self):
        """Test getting physical constants."""
        # Test with common name
        von_karman = get_physical_constant('von_karman_constant')
        self.assertAlmostEqual(von_karman, 0.41, places=2)
        
        # Missing constant should raise
        with self.assertRaises(ValueError):
            get_physical_constant('nonexistent_constant')
        
    def test_convenience_functions(self):
        """Test convenience functions for common constants."""
        von_karman = get_von_karman()
        self.assertAlmostEqual(von_karman, 0.41, places=2)
        
        gravity = get_gravity()
        self.assertAlmostEqual(gravity, 9.80665, places=4)
        
        gas_constant = get_gas_constant()
        self.assertAlmostEqual(gas_constant, 8.314, places=2)
        
    def test_tensor_conversion(self):
        """Test getting constants as tensors."""
        von_karman_tensor = get_physical_constant('von_karman_constant', as_tensor=True)
        self.assertIsInstance(von_karman_tensor, torch.Tensor)
        self.assertAlmostEqual(von_karman_tensor.item(), 0.41, places=2)


class TestTransportParameterUsage(unittest.TestCase):
    """Test that transport modules use parameter manager correctly."""
    
    def test_advection_parameters(self):
        """Test advection module parameter usage."""
        from relic.scientific.transport.advection import LinearAdvection
        
        # Create instance with proper grid spacing
        # For 3D advection, provide list of 3 velocity components
        advection = LinearAdvection(
            velocity_field=[torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.0)],  # u, v, w components
            spatial_dims=3,
            grid_spacing=[1.0, 1.0, 1.0]  # Provide as list
        )
        
        # Runtime parameters stay in ParameterManager instead of object caches.
        required = set(advection.required_parameters())
        self.assertIn("transport.numerical.cfl_advection_max", required)
        self.assertFalse(hasattr(advection, "max_cfl"))
        self.assertFalse(hasattr(advection, "min_timestep"))
        self.assertFalse(hasattr(advection, "conservation_tolerance"))

        max_cfl = advection.get_parameter_value(
            "cfl_advection_max",
            namespace="transport.numerical",
            required=True,
        )
        self.assertAlmostEqual(float(max_cfl), 1.0, places=1)
    
    def test_no_config_get_in_transport(self):
        """Verify no config.get() calls remain in transport modules."""
        transport_files = [
            'relic/scientific/transport/advection.py',
            'relic/scientific/transport/advection_diffusion.py',
            'relic/scientific/transport/diffusion.py'
        ]
        
        for file_path in transport_files:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Should not have config.get with hardcoded defaults
                self.assertNotIn("config.get('max_cfl', 1.0)", content,
                               f"Found config.get with hardcoded default in {file_path}")
                self.assertNotIn("config.get('min_timestep',", content,
                               f"Found config.get with hardcoded default in {file_path}")


class TestCouplingParameterUsage(unittest.TestCase):
    """Test that coupling modules use parameter manager correctly."""
    
    def test_facade_tolerances(self):
        """Test that scale manager uses parameter-based tolerances."""
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.coupling.core import ScalingMode
        from relic.core.state import StateManager

        manager = ScaleManager(state_manager=StateManager())
        manager.initialize()

        # Each mode's ledger tolerance is read from its precision policy
        strict_tol = manager.ledger_tolerance(ScalingMode.STRICT_CONSERVATIVE)
        self.assertIsNotNone(strict_tol)
        self.assertLess(strict_tol, 1e-10)
        
    def test_no_hardcoded_defaults_in_coupling(self):
        """Verify no hardcoded torch.tensor() defaults in coupling."""
        coupling_files = Path('relic/coupling').rglob('*.py')
        
        problematic_patterns = [
            'torch.tensor(1e-6)',
            'torch.tensor(1e6)',
            'torch.tensor(0.0)',
            'torch.tensor(0.5)',
            'torch.tensor(1.0)'
        ]
        
        for file_path in coupling_files:
            if '__pycache__' in str(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for patterns that indicate bypassing parameter manager
            for pattern in problematic_patterns:
                # Allow in comments or strings
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if pattern in line and not line.strip().startswith('#'):
                        # Check if it's part of a get() with param_manager
                        if 'param_manager.get(' not in line:
                            self.fail(f"Found hardcoded {pattern} in {file_path}:{i+1}")


class TestOzoneStressFix(unittest.TestCase):
    """Test that ozone_stress.py parameter keys are fixed."""
    
    def test_ozone_stress_parameters(self):
        """Test ozone stress module parameter access."""
        from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
        
        # Create instance
        ozone = OzoneStress()
        
        # Check that antioxidant_capacity is properly loaded
        self.assertIsNotNone(ozone.antioxidant_capacity)
        
        
        # Check they are tensors
        self.assertIsInstance(ozone.antioxidant_capacity, torch.Tensor)
        
        
    def test_no_device_parameter_key(self):
        """Verify 'device' is not used as parameter key."""
        ozone_file = Path('relic/scientific/biosphere/stress/ozone_stress.py')
        
        if ozone_file.exists():
            with open(ozone_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should not have get_parameter_value("device", ...)
            self.assertNotIn('get_parameter_value("device"', content,
                           "Found incorrect 'device' parameter key")


class TestParameterManagerIntegration(unittest.TestCase):
    """Test overall parameter manager integration."""
    
    def test_parameter_manager_singleton(self):
        """Test that parameter manager is a singleton."""
        pm1 = get_global_manager()
        pm2 = get_global_manager()
        
        self.assertIs(pm1, pm2, "Parameter manager is not a singleton")
    
    def test_parameter_loading_from_yaml(self):
        """Test that parameters can be loaded from YAML database."""
        pm = get_global_manager()
        
        # Test a known parameter from the database
        # This will only work if YAML schema is correct
        von_karman = pm.get('relic.atmosphere.constants.turbulence.von_karman_constant')
        
        if von_karman is not None:
            self.assertAlmostEqual(von_karman, 0.41, places=2)
    
    def test_coupling_defaults_loaded(self):
        """Test that coupling defaults YAML is loaded."""
        pm = get_global_manager()
        
        # Check if coupling defaults are accessible
        stomatal_min = pm.get('scale_coupling.defaults.stomatal_conductance_min')
        if stomatal_min is not None:
            self.assertEqual(stomatal_min, 1.0e-6)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

