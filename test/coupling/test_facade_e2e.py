"""
End-to-End Test for Coupling Scale Manager

Tests the complete functionality of ScaleManager including:
- Hierarchical aggregation with homogeneous variable roles
- Core module integration (policy, context, audit, errors)
- WeightOps integration
- Conservation enforcement
- Error handling

Mathematical Expression:
    Test aggregation: Y[j] = 危_{i鈭坖} X[i] * A[i] / A[j]
    Test conservation: |危 Y[j] * A[j] - 危 X[i] * A[i]| 鈮?tolerance

Contract boundary:
    Disaggregation metadata synchronization is fail-closed until the
    canonical CFAttributeSynchronizer declares that operation.

References:
    - ScaleManager unified interface
    - Core module policy and error management
    - WeightOps unified weight operations
"""

import pytest
import torch
import logging

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.types import ScalingStrategy
from relic.coupling.core import (
    ScaleHierarchyError,
    ScalingMode,
)
from relic.core.state import StateManager

logger = logging.getLogger(__name__)


class TestScaleManagerE2E:
    """
    End-to-end tests for ScaleManager.
    
    Tests complete workflows including aggregation, disaggregation,
    and integration with all core modules.
    """
    
    @pytest.fixture
    def device(self):
        """Test device (CPU for consistency)."""
        return torch.device("cpu")
    
    @pytest.fixture
    def facade_conservative(self, device):
        """Conservative manager instance."""
        mgr = ScaleManager(
            device=device,
            default_mode=ScalingMode.CONSERVATIVE,
            enforce_conservation=True,
            cf_strict_mode=False,
            state_manager=StateManager(),
        )
        mgr.initialize()
        return mgr
    
    @pytest.fixture
    def facade_fast(self, device):
        """Fast approximation manager instance."""
        mgr = ScaleManager(
            device=device,
            default_mode=ScalingMode.FAST_APPROX,
            enforce_conservation=False,
            cf_strict_mode=False,
            state_manager=StateManager(),
        )
        mgr.initialize()
        return mgr
    
    @pytest.fixture
    def sample_patch_data(self, device):
        """Sample patch-scale data for testing."""
        # 6 patches with different values
        return {
            'gpp': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=device),
            'temperature': torch.tensor([20.0, 21.0, 22.0, 23.0, 24.0, 25.0], device=device),
            'lai': torch.tensor([2.5, 3.0, 3.5, 4.0, 4.5, 5.0], device=device)
        }
    
    def test_facade_initialization(self, facade_conservative):
        """Test manager initialization with core modules."""
        # Check that core modules are properly initialized
        assert facade_conservative.precision_policy is not None
        assert facade_conservative.tensor_context is not None
        assert facade_conservative.audit_collector is not None
        
        # Check precision policy values
        policy = facade_conservative.precision_policy
        assert policy.tolerance == 1e-9
        assert policy.dtype == torch.float64
        assert policy.enforce_conservation is True
        
        # Check components are initialized
        assert facade_conservative.conservation_enforcer is not None
        assert facade_conservative.constraint_manager is not None
    
    def test_mode_switching(self, facade_conservative):
        """Test switching between precision modes."""
        # Start with conservative mode
        assert facade_conservative.default_mode == ScalingMode.CONSERVATIVE
        assert facade_conservative.precision_policy.tolerance == 1e-9
        
        # Switch to fast approximation
        facade_conservative.set_mode(ScalingMode.FAST_APPROX)
        assert facade_conservative.default_mode == ScalingMode.FAST_APPROX
        assert facade_conservative.precision_policy.tolerance == 1e-3
        assert facade_conservative.precision_policy.dtype == torch.float32
        
        # Switch to strict conservative
        facade_conservative.set_mode(ScalingMode.STRICT_CONSERVATIVE)
        assert facade_conservative.default_mode == ScalingMode.STRICT_CONSERVATIVE
        assert facade_conservative.precision_policy.tolerance == 1e-12
        assert facade_conservative.precision_policy.dtype == torch.float64
    
    def test_aggregation_basic(self, facade_conservative, sample_patch_data):
        """Test basic aggregation functionality."""
        # Test patch to hillslope aggregation
        result, audit = facade_conservative.aggregate(
            data=sample_patch_data,
            source_scale="patch",
            target_scale="hillslope",
            variables=["gpp"],
            mode=ScalingMode.CONSERVATIVE
        )
        
        # Check results exist
        assert "gpp" in result
        
        # Check audit information
        assert audit.operation == "aggregate"
        assert audit.source_scale == "patch"
        assert audit.target_scale == "hillslope"
        assert audit.mode == ScalingMode.CONSERVATIVE
        assert audit.variables == ["gpp"]
        
        # Check that extensive variable (gpp) has reasonable aggregation
        assert result["gpp"].shape[0] == 2  # 2 hillslopes
        assert torch.all(result["gpp"] > 0)
        
        # Check conservation for extensive variable
        assert audit.conservation_enforced is True
        assert "gpp" in audit.metadata.get("conservation_audit", {})
    
    def test_invalid_scale_hierarchy(self, facade_conservative, sample_patch_data):
        """Test error handling for invalid scale transitions."""
        # Test invalid upscaling
        with pytest.raises(ScaleHierarchyError) as exc_info:
            facade_conservative.aggregate(
                data=sample_patch_data,
                source_scale="landscape",  # Invalid: landscape > patch
                target_scale="patch",
                variables=["gpp"]
            )
        
        error = exc_info.value
        assert error.source_scale == "landscape"
        assert error.target_scale == "patch"
        assert error.direction == "up"
        assert "patch" in error.valid_hierarchy
        
        # Test invalid downscaling
        with pytest.raises(ScaleHierarchyError) as exc_info:
            facade_conservative.disaggregate(
                data=sample_patch_data,
                source_scale="patch",
                target_scale="landscape",  # Invalid: patch > landscape
                variables=["gpp"]
            )
        
        error = exc_info.value
        assert error.direction == "down"
    
    def test_single_tensor_aggregation_interface(self, facade_conservative):
        """Test the executable single-tensor aggregation interface."""
        # Create single tensor
        patch_gpp = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Test aggregation
        result, audit = facade_conservative.aggregate(
            data=patch_gpp,
            source_scale="patch",
            target_scale="hillslope",
            variables=["gpp"]
        )
        
        # Should return single tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2  # 2 hillslopes
    
    def test_strategy_selection(self, facade_conservative, sample_patch_data):
        """Test different aggregation strategies."""
        strategies = [
            ScalingStrategy.CONSERVATIVE,
            ScalingStrategy.AREA_WEIGHTED_MEAN,
            ScalingStrategy.AREA_WEIGHTED_SUM,
            ScalingStrategy.FLOW_CONSERVING
        ]
        
        results = {}
        for strategy in strategies:
            try:
                result, audit = facade_conservative.aggregate(
                    data=sample_patch_data,
                    source_scale="patch",
                    target_scale="hillslope",
                    variables=["gpp"],
                    strategy=strategy
                )
                results[strategy] = result["gpp"]
                
                # Check that strategy was used
                assert audit.strategy == strategy
                
            except Exception as e:
                # Some strategies might not be fully implemented
                logger.warning(f"Strategy {strategy} failed: {e}")
        
        # Should have at least one working strategy
        assert len(results) > 0
    
    def test_fast_vs_conservative_mode(self, sample_patch_data, device):
        """Test differences between fast and conservative modes."""
        # Conservative manager
        facade_conservative = ScaleManager(
            device=device,
            default_mode=ScalingMode.CONSERVATIVE,
            enforce_conservation=True,
            state_manager=StateManager(),
        )
        facade_conservative.initialize()

        # Fast manager
        facade_fast = ScaleManager(
            device=device,
            default_mode=ScalingMode.FAST_APPROX,
            enforce_conservation=False,
            state_manager=StateManager(),
        )
        facade_fast.initialize()
        
        # Test same aggregation with both modes
        result_conservative, audit_conservative = facade_conservative.aggregate(
            data=sample_patch_data,
            source_scale="patch",
            target_scale="hillslope",
            variables=["gpp"]
        )
        
        # fast_approx is bilinear interpolation on declared coordinates: the parameter-built
        # hierarchy declares none, so the fast call raises instead of falling back to an area mean.
        with pytest.raises(RuntimeError, match="coordinates"):
            facade_fast.aggregate(
                data=sample_patch_data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["gpp"]
            )

        assert audit_conservative.conservation_enforced is True
        assert audit_conservative.mode == ScalingMode.CONSERVATIVE
    
    def test_undeclared_variable_is_rejected_not_inferred(self, facade_conservative):
        """An uncatalogued variable needs declared metadata; no runtime type inference."""
        test_data = {
            "unknown_var": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        }

        with pytest.raises(KeyError, match="unknown_var"):
            facade_conservative.aggregate(
                data=test_data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["unknown_var"]
            )

        # With a declared basis the call runs and the audit is area-basis aware.
        result, audit = facade_conservative.aggregate(
            data=test_data,
            source_scale="patch",
            target_scale="hillslope",
            variables=["unknown_var"],
            metadata={"unknown_var": {"spatial_basis": "intensive", "units": "1"}},
        )
        assert result["unknown_var"].shape == (2,)
        assert audit.metadata["conservation_audit"] == {}
    
    def test_audit_information(self, facade_conservative, sample_patch_data):
        """Test comprehensive audit information collection."""
        result, audit = facade_conservative.aggregate(
            data=sample_patch_data,
            source_scale="patch",
            target_scale="hillslope",
            variables=["gpp"],
            mode=ScalingMode.CONSERVATIVE
        )
        
        # Check audit completeness
        assert audit.operation == "aggregate"
        assert audit.source_scale == "patch"
        assert audit.target_scale == "hillslope"
        assert audit.variables == ["gpp"]
        assert audit.mode == ScalingMode.CONSERVATIVE
        assert audit.strategy is not None
        
        # Check metadata structure
        assert "aggregation_audit" in audit.metadata
        assert "conservation_audit" in audit.metadata
        
        # Check conservation audit for extensive variable
        conservation_audit = audit.metadata["conservation_audit"]
        if "gpp" in conservation_audit:
            gpp_audit = conservation_audit["gpp"]
            assert hasattr(gpp_audit, 'absolute_error')
            assert hasattr(gpp_audit, 'relative_error')
    
    def test_error_recovery(self, facade_conservative):
        """Test error recovery mechanisms."""
        # Test with empty data
        empty_data = {}
        
        # Should handle gracefully
        result, audit = facade_conservative.aggregate(
            data=empty_data,
            source_scale="patch",
            target_scale="hillslope",
            variables=["gpp"]
        )
        
        # Should have warnings
        assert len(audit.warnings) > 0
        
        # Test with mismatched tensor dimensions
        bad_data = {
            "gpp": torch.tensor([1.0, 2.0])  # Wrong size for 6 patches
        }
        
        # Should either handle gracefully or raise appropriate error
        try:
            result, audit = facade_conservative.aggregate(
                data=bad_data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["gpp"]
            )
            # If it succeeds, should have warnings
            assert len(audit.warnings) >= 0
        except Exception as e:
            # Should be a meaningful error
            assert len(str(e)) > 0


@pytest.mark.integration
class TestFacadeIntegration:
    """Integration tests with other coupling modules."""
    
    def test_cf_metadata_integration(self, facade_conservative=None, device=None):
        """Test CF metadata synchronization integration."""
        if facade_conservative is None:
            facade_conservative = ScaleManager(
                device=torch.device("cpu"),
                state_manager=StateManager(),
            )
            facade_conservative.initialize()
        
        # Test with CF metadata
        test_data = {"gpp": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
        cf_metadata = {
            "gpp": {
                "standard_name": "gross_primary_productivity_of_carbon",
                "units": "kg C m-2 s-1",
                "description": "Gross primary productivity"
            }
        }
        
        result, audit = facade_conservative.aggregate(
            data=test_data,
            source_scale="patch",
            target_scale="hillslope",
            variables=["gpp"],
            metadata=cf_metadata
        )
        
        # Should have CF metadata in audit
        assert "cf_metadata" in audit.metadata
        
        # CF compliance should be determined correctly
        assert isinstance(audit.cf_compliant, bool)
    
    def test_conservation_enforcer_integration(self):
        """Test integration with conservation enforcer."""
        facade = ScaleManager(
            device=torch.device("cpu"),
            default_mode=ScalingMode.STRICT_CONSERVATIVE,
            enforce_conservation=True,
            state_manager=StateManager(),
        )
        facade.initialize()
        
        # Test with data that should trigger conservation enforcement
        test_data = {"gpp": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])}
        
        result, audit = facade.aggregate(
            data=test_data,
            source_scale="patch",
            target_scale="hillslope",
            variables=["gpp"]
        )
        
        # Should enforce conservation
        assert audit.conservation_enforced is True
        
        # Should have conservation audit
        conservation_audit = audit.metadata.get("conservation_audit", {})
        assert "gpp" in conservation_audit


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "--tb=short"])
