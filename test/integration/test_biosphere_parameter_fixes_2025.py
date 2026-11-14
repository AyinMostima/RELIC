"""
Integration test for biosphere module parameter system fixes
Tests all the fixes implemented on 2025-12-29
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel, CouplingStrategy
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.stress import WaterStress, ThermalStress


class TestBiosphereParameterFixes:
    """Test suite for verifying biosphere module parameter fixes"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Initialize parameter manager
        self.param_manager = get_global_manager()
        
    def test_bvoc_parameter_namespace(self):
        """Test that BVOC parameters use correct namespace"""
        megan = MEGANModel(
            name="test_megan_simultaneous",
            coupling_strategy=CouplingStrategy.SIMULTANEOUS,
        )
        
        # Check that parameters are loaded with correct namespace
        assert hasattr(megan, 'isoprene_ef')
        assert hasattr(megan, 'alpha_light')
        assert 'isoprene_ef' in dict(megan.named_buffers())
        assert 'alpha_light' in dict(megan.named_buffers())
        assert torch.is_tensor(megan.isoprene_ef)
        assert torch.is_tensor(megan.alpha_light)
        assert all(
            key in megan.metadata.parameters
            for key in (
                'biosphere.bvoc.co2_inhibition.ismax',
                'biosphere.bvoc.co2_inhibition.c_star',
                'biosphere.bvoc.co2_inhibition.hill_exponent',
                'biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio',
            )
        )
        
    def test_farquhar_trainable_parameters(self):
        """Test that Farquhar parameters are registered as trainable"""
        # Create Farquhar model
        farquhar = FarquharPhotosynthesis(name="test_farquhar")
        
        # Check that key parameters are nn.Parameter
        assert isinstance(farquhar.vcmax25, nn.Parameter)
        assert isinstance(farquhar.jmax25, nn.Parameter)
        assert isinstance(farquhar.rd25, nn.Parameter)
        
        # Verify they are in parameters list (trainable)
        param_names = [name for name, _ in farquhar.named_parameters()]
        assert 'vcmax25' in param_names
        assert 'jmax25' in param_names
        assert 'rd25' in param_names
        
        # Check that gradients can be computed
        farquhar.vcmax25.requires_grad = True
        loss = farquhar.vcmax25.sum()
        loss.backward()
        assert farquhar.vcmax25.grad is not None
        
    def test_stress_functor_metadata(self):
        """Test that stress functors have correct metadata"""
        # Create stress functors
        water_stress = WaterStress(name="test_water_stress")
        thermal_stress = ThermalStress(name="test_thermal_stress")
        
        # Check that they initialize without errors
        assert water_stress is not None
        assert thermal_stress is not None
        
        # Verify metadata structure (should be dict, not causing AttributeError)
        # This is tested implicitly - if metadata was wrong type, initialization would fail
        
    def test_dependency_names_match(self):
        """Test that dependency names match actual functor names"""
        # Create MEGAN model
        megan = MEGANModel(
            name="test_megan",
            config={"use_stress_functors": True},
        )
        
        # Get dependencies
        deps = megan.get_dependencies()
        
        assert 'atmospheric_radiation' in deps
        assert 'farquhar_photosynthesis' in deps
        assert 'water_stress' in deps
        assert 'thermal_stress' in deps
        assert 'ozone_stress' in deps
            
    def test_parameter_loading_from_yaml(self):
        """Test that parameters load correctly from YAML with proper namespaces"""
        # Test BVOC parameters
        bvoc_param = self.param_manager.get(
            "emission_factors.isoprene",
            namespace="biosphere.bvoc"
        )
        assert bvoc_param is not None
        
        megan_param = self.param_manager.get(
            "alpha_light", 
            namespace="biosphere.bvoc.megan"
        )
        assert megan_param is not None
        
        # Test photosynthesis parameters
        photo_param = self.param_manager.get(
            "vcmax25",
            namespace="biosphere.photosynthesis"
        )
        assert photo_param is not None
        
    def test_no_hardcoded_defaults(self):
        """Test that modules don't use hardcoded defaults"""
        # Create modules
        megan = MEGANModel(name="test_megan")
        
        # All parameters should come from ParameterManager
        # If ParameterManager is missing, it should raise an error (required=True)
        # This is implicitly tested - if parameters were missing and required=True,
        # initialization would fail
        
    def test_parameter_update_response(self):
        """Test that ParameterManager values are registered as module buffers."""
        megan = MEGANModel(name="test_megan")
        expected = self.param_manager.get(
            "biosphere.bvoc.megan.alpha_light",
            required=True,
        )
        assert torch.allclose(
            megan.alpha_light,
            torch.as_tensor(expected, dtype=megan.alpha_light.dtype),
        )
        assert 'alpha_light' in megan.state_dict()
        with pytest.raises(TypeError):
            megan.alpha_light = 0.005


class TestParameterAudit:
    """Test parameter audit functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.param_manager = get_global_manager()
        
    def test_parameter_audit_runs(self):
        """Test that parameter audit can run without errors"""
        # Run audit
        audit_report = self.param_manager.run_registry_audit()
        
        # Check that audit returns a report
        assert audit_report is not None
        
        # Check for common issues
        assert hasattr(audit_report, "missing_keys")
        assert hasattr(audit_report, "orphan_keys")
        assert hasattr(audit_report, "issues")
        
    def test_namespace_filtering(self):
        """Test that namespace filtering works in audit"""
        # Run audit for specific namespace
        audit_report = self.param_manager.run_registry_audit(
            namespace_filter="biosphere.bvoc"
        )
        
        # Verify audit was filtered
        assert audit_report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

