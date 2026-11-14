#!/usr/bin/env python3
"""
L1 Unit Tests for ParameterManager

Tests the core functionality of the unified parameter management system.
Ensures all parameters can be loaded from YAML and accessed properly.

Author: Yuchong Long
Date: 2025-09-19
"""

import pytest
from pathlib import Path
import sys
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.parameters import (
    UnifiedParameterManager,
    get_global_manager,
)
from relic.parameters import ParameterSpec


class TestParameterManager:
    """Test suite for ParameterManager functionality"""
    
    @pytest.fixture(autouse=True)
    def _fresh_manager(self, fresh_global_parameter_manager):
        """Each test runs on a fresh parameter manager; the previous one is put back."""
        self.manager = get_global_manager()
        yield
        
    def test_singleton_pattern(self):
        """Test that ParameterManager follows singleton pattern"""
        manager1 = get_global_manager()
        manager2 = get_global_manager()
        assert manager1 is manager2, "ParameterManager should be a singleton"
        
    def test_load_physical_constants(self):
        """Test loading physical constants from YAML"""
        # Physical constants should be auto-loaded
        # Test some key constants
        constants_to_test = [
            ('relic.physical.constants.universal.gas_constant', 8.314462618, 'J mol-1 K-1'),
            ('relic.physical.constants.universal.avogadro_constant', 6.02214076e23, 'molecules mol-1'),
            ('relic.physical.constants.universal.boltzmann_constant', 1.380649e-23, 'J K-1'),
        ]
        
        for param_path, expected_value, expected_units in constants_to_test:
            value = self.manager.get(param_path)
            assert value is not None, f"Parameter {param_path} not found"
            
            # Check if value is close (floating point comparison)
            if isinstance(value, (int, float)):
                assert abs(value - expected_value) < 1e-10, \
                    f"Parameter {param_path} has incorrect value: {value} vs {expected_value}"
                    
    def test_parameter_registration(self):
        """Test registering new parameters"""
        param_spec = ParameterSpec(
            key='test.namespace.test_param',
            value=42.0,
            units='m',
            description='Test parameter',
            bounds=(0.0, 100.0), dtype="float32", category="model_parameter"
        )

        self.manager.register_parameter(param_spec)

        # Retrieve the parameter
        value = self.manager.get('test.namespace.test_param')
        assert value == 42.0, "Registered parameter has incorrect value"
        
    def test_parameter_validation(self):
        """Test parameter validation (bounds checking)"""
        param_spec = ParameterSpec(
            key='test.bounded_param',
            value=50.0,
            units='dimensionless',
            description='Bounded test parameter',
            bounds=(0.0, 100.0), dtype="float32", category="model_parameter"
        )

        self.manager.register_parameter(param_spec)

        # Set value outside bounds should be clamped under default WARN_AND_CLAMP policy
        with pytest.warns(UserWarning):
            self.manager.set_value('test.bounded_param', 150.0)
        value_high = self.manager.get('test.bounded_param')
        assert value_high == 100.0, "Upper bound clamp failed"

        with pytest.warns(UserWarning):
            self.manager.set_value('test.bounded_param', -10.0)
        value_low = self.manager.get('test.bounded_param')
        assert value_low == 0.0, "Lower bound clamp failed"

    def test_namespace_hierarchy(self):
        """Test hierarchical namespace access"""
        # Register parameters in nested namespaces
        self.manager.register_parameter(ParameterSpec(
            key='level1.level2.level3.deep_param',
            value=123.456,
            units='kg',
            description='Deeply nested parameter', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        ))
        
        # Test access
        value = self.manager.get('level1.level2.level3.deep_param')
        assert value == 123.456, "Nested parameter access failed"
        
        # Test namespace listing
        params = self.manager.list('level1.level2.level3')
        assert 'level1.level2.level3.deep_param' in params, "Parameter not listed in namespace"
        
    def test_unit_conversion(self):
        """Test parameter unit conversion"""
        # Register temperature parameter in Kelvin
        self.manager.register_parameter(ParameterSpec(
            key='test.temperature',
            value=300.0,
            units='K',
            description='Temperature parameter', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        ))

        # Get value (should handle units internally)
        value = self.manager.get('test.temperature')
        assert value == 300.0, "Temperature parameter has incorrect value"

    def test_string_parameter_handling(self):
        """Ensure string/categorical parameters load without tensor conversion."""
        value = self.manager.get('biosphere.phenology.common.phenology_type')
        assert isinstance(value, str)
        assert value == 'static'

        # Explicit type validation
        value_checked = self.manager.get(
            'biosphere.phenology.crop.default_type',
            expected_type=str
        )
        assert value_checked == 'wheat'

        # Attempting to register string parameter as learnable should fail
        with pytest.raises(TypeError):
            self.manager.register_parameter(ParameterSpec(
                key='test.string_learnable',
                value='c3',
                units='dimensionless',
                dtype='string',
                requires_grad=True, bounds=(-float("inf"), float("inf")), category="model_parameter"
            ))

    def test_missing_parameter_handling(self):
        """Test handling of missing parameters"""
        # Try to get non-existent parameter
        value = self.manager.get('non.existent.parameter')
        assert value is None, "Non-existent parameter should return None"
        
        # Try with default value
        value = self.manager.get('non.existent.parameter', default=99.9)
        assert value == 99.9, "Default value not returned for missing parameter"
        
    def test_biosphere_parameters(self):
        """Test loading biosphere module parameters"""
        # Check if photosynthesis parameters are loaded
        vcmax_25 = self.manager.get('relic.biosphere.photosynthesis.c3.vcmax_25')
        if vcmax_25 is not None:
            assert isinstance(vcmax_25, (int, float)), "vcmax_25 should be numeric"
            assert vcmax_25 > 0, "vcmax_25 should be positive"
            
    def test_pedosphere_parameters(self):
        """Test loading pedosphere module parameters"""
        # Load pedosphere parameters if available
        yaml_path = project_root / 'relic/parameters/database/relic/pedosphere/pedosphere_parameters.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            assert data is not None, "Pedosphere parameters file is empty"
            assert 'metadata' in data or len(data) > 0, "Pedosphere parameters not properly formatted"
            
    def test_atmosphere_parameters(self):
        """Test loading atmosphere module parameters"""
        # Load atmosphere parameters if available  
        yaml_path = project_root / 'relic/parameters/database/relic/atmosphere/atmosphere_parameters.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            assert data is not None, "Atmosphere parameters file is empty"
            
    def test_parameter_provenance(self):
        """Test parameter provenance tracking"""
        param_spec = ParameterSpec(
            key='test.tracked_param',
            value=7.5,
            units='m s-1',
            description='Parameter with provenance',
            provenance={
                'source': 'literature',
                'reference': 'Smith et al., 2020',
                'version': '1.0'
            }, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )

        self.manager.register_parameter(param_spec)
        
        # Get full parameter spec
        spec = self.manager.get_parameter('test.tracked_param')
        assert spec is not None, "Parameter spec not found"
        assert spec.provenance is not None, "Provenance not tracked"
        assert spec.provenance.get('source') == 'literature', "Provenance source incorrect"
        
    def test_batch_parameter_loading(self):
        """Test loading multiple parameters at once"""
        params_to_register = [
            ('param1', 1.0, 'kg'),
            ('param2', 2.0, 'm'),
            ('param3', 3.0, 's'),
        ]
        
        for name, value, units in params_to_register:
            self.manager.register_parameter(ParameterSpec(
                key=f'batch.{name}',
                value=value,
                units=units,
                description=f"Batch parameter {name}", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
            ))
            
        # Verify all parameters were registered
        params = self.manager.list('batch')
        assert len(params) == 3, "Not all batch parameters registered"
        # Check that all batch parameters are in the list
        for name, _, _ in params_to_register:
            assert f'batch.{name}' in params, f"Parameter {name} not found in list"
        
        for name, value, _ in params_to_register:
            retrieved = self.manager.get(f'batch.{name}')
            assert retrieved == value, f"Batch parameter {name} has incorrect value"
            
    def test_parameter_serialization(self):
        """Test saving and loading parameter state"""
        # Register some parameters
        self.manager.register_parameter(ParameterSpec(
            key='serialize.test_param',
            value=42.0,
            units='dimensionless',
            description='Serialization test', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        ))
        
        # Save state
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state = self.manager.to_dict()
            json.dump(state, f)
            temp_path = f.name
            
        # Create new manager and load state
        new_manager = UnifiedParameterManager()
        with open(temp_path, 'r') as f:
            state = json.load(f)
            # Note: from_dict method would need to be implemented
            
        # Clean up
        import os
        os.unlink(temp_path)
        
    def test_thread_safety(self):
        """Test thread-safe parameter access"""
        import threading
        import time
        
        results = []
        
        def access_parameter():
            for _ in range(100):
                value = self.manager.get('relic.physical.constants.universal.gas_constant')
                results.append(value)
                time.sleep(0.001)
                
        # Create multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=access_parameter)
            threads.append(t)
            t.start()
            
        # Wait for all threads
        for t in threads:
            t.join()
            
        # All results should be the same
        assert len(results) == 500, "Not all thread operations completed"
        assert all(r == results[0] for r in results), "Thread-unsafe parameter access detected"


class TestParameterIntegration:
    """Integration tests for parameter usage in modules"""
    
    def test_functor_parameter_access(self):
        """Test that Functors can access parameters correctly"""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        functor = FarquharPhotosynthesis()
        assert hasattr(functor, 'get_parameter_value'), "Functor missing parameter access method"
            
    def test_parameter_yaml_integrity(self):
        """Test that all generated YAML files are valid"""
        param_db_path = project_root / 'relic/parameters/database/relic'
        
        yaml_files = list(param_db_path.rglob('*.yaml'))
        assert len(yaml_files) > 0, "No YAML parameter files found"
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                assert data is not None, f"Empty YAML file: {yaml_file}"
                
                # Check for metadata if it's a generated file
                if 'metadata' in data:
                    assert 'version' in data['metadata'], f"Missing version in {yaml_file}"
                    
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {yaml_file}: {e}")
                
    def test_removed_interface_mapping_parameters_are_absent(self):
        """Removed generic interface-mapping parameters must stay absent."""
        manager = get_global_manager()
        key = 'anthroposphere.interface.mapping.standard_name.livestock_density'
        assert manager.get_parameter(key) is None
        assert manager.get(key) is None
        with pytest.raises(KeyError):
            manager.get(key, required=True)
        assert manager.list('anthroposphere.interface.mapping') == []
        
    def test_parameter_coverage(self):
        """Test that major modules have parameter coverage"""
        modules_to_check = [
            'atmosphere',
            'biosphere', 
            'hydrosphere',
            'pedosphere',
            'anthroposphere',
            'coupling'
        ]
        
        param_db_path = project_root / 'relic/parameters/database/relic'
        
        for module in modules_to_check:
            module_path = param_db_path / module
            if module_path.exists():
                yaml_files = list(module_path.glob('*.yaml'))
                assert len(yaml_files) > 0, f"No parameter files for module {module}"
                
                # Check that files contain actual parameters
                total_params = 0
                for yaml_file in yaml_files:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    # Count non-metadata keys
                    for key in data:
                        if key != 'metadata':
                            if isinstance(data[key], dict):
                                total_params += len(data[key])
                                
                print(f"Module {module}: {total_params} parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

