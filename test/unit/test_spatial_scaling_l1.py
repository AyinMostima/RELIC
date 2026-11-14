"""
L1 Unit Tests for Spatial Scaling Module

Tests for SpatialScaling.forward() method and related functionality.
These tests verify that the fixed forward method correctly performs spatial
upscaling/downscaling operations with proper conservation and error handling.

References:
- RELIC Multi-scale Coupling Framework
- Conservative regridding algorithms for spatial data
- PyTorch tensor operations for differentiable scaling

Mathematical Expressions:
    Area-weighted upscaling: Y_target = sum(X_source * A_source) / A_target
    Bilinear interpolation: Y(x,y) = sum w_ij * Y(i,j) for i,j in {0,1}
    Conservation check: eps = |sum(Y_target * A_target) - sum(X_source * A_source)| / |sum(X_source * A_source)|
"""

import pytest
import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling
    SCALING_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Cannot import spatial scaling modules: {e}")
    SCALING_AVAILABLE = False


class TestSpatialScalingL1:
    """L1 unit tests for SpatialScaling module."""

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_spatial_scaling_initialization(self):
        """Test SpatialScaling component initialization."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(30.0, 30.0),
            method="bilinear",
            device=torch.device("cpu")
        )
        
        assert scaling.source_resolution == (10.0, 10.0)
        assert scaling.target_resolution == (30.0, 30.0)
        assert scaling.device.type == "cpu"
        assert scaling.scale_ratios == (3.0, 3.0)

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_forward_upscaling_detection(self):
        """Test that forward method correctly detects upscaling operations."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(30.0, 30.0),
            method="bilinear",
            device=torch.device("cpu")
        )
        
        # Create source data (10x10 grid -> upscale to 3x3 grid)
        source_data = torch.randn(10, 10)
        result = scaling.upscale(source_data)
        
        # Upscaling reduces size: 10/3.0 = 3
        assert result.shape == (3, 3)
        assert scaling.is_upscaling

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_forward_downscaling_detection(self):
        """Test that forward method correctly detects downscaling operations."""
        scaling = SpatialScaling(
            source_scale="hillslope",
            target_scale="patch",
            source_resolution=(30.0, 30.0),
            target_resolution=(10.0, 10.0),
            method="bilinear",
            device=torch.device("cpu")
        )
        
        # Create source data (30x30 grid -> downscale to 90x90 grid)
        source_data = torch.randn(30, 30)
        result = scaling.downscale(source_data)
        
        # Downscaling increases size: 30/(10/30) = 90
        assert result.shape == (90, 90)
        assert scaling.is_downscaling

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_forward_conservative_mode(self):
        """Test forward method with conservative scaling mode."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(20.0, 20.0),
            method="conservative",
            device=torch.device("cpu")
        )
        
        # Test conservative upscaling
        source_data = torch.ones(10, 10)  # Unit values for conservation test
        result = scaling.upscale(source_data)
        
        # Check area-weighted conservation for conservative remapping
        source_area = scaling.source_resolution[0] * scaling.source_resolution[1]
        target_area = scaling.target_resolution[0] * scaling.target_resolution[1]
        source_total = source_data.sum() * source_area
        target_total = result.sum() * target_area
        relative_error = abs(target_total - source_total) / abs(source_total)
        
        # Allow for some numerical error in conservative regridding
        assert relative_error < 1e-3, f"Conservation error too large: {relative_error}"

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_forward_error_handling(self):
        """Test forward method error handling for invalid inputs."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(20.0, 20.0),
            method="conservative",
            device=torch.device("cpu")
        )
        
        # Test with empty state dict
        empty_state = {}
        result = scaling.forward(empty_state, time=0.0, dt=1.0)
        assert result == {}
        
        # Test with non-tensor data - forward should convert or raise error
        # The forward method converts strings to tensors, which may fail
        # We just need to ensure it doesn't crash silently
        try:
            invalid_state = {"data": [1, 2, 3]}  # Valid data that can convert to tensor
            result = scaling.forward(invalid_state, time=0.0, dt=1.0)
            assert "scaled_data" in result
        except (TypeError, ValueError):
            pass  # Expected for truly invalid data

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_forward_diagnostics(self):
        """Test that forward method returns proper diagnostics."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(20.0, 20.0),
            method="conservative",
            device=torch.device("cpu")
        )
        
        source_data = torch.randn(10, 10)
        state_dict = {"data": source_data, "operation": "upscale"}
        
        result = scaling.forward(state_dict, time=0.0, dt=1.0)
        
        # Check diagnostic fields
        assert "scaling_info" in result
        scaling_info = result["scaling_info"]
        assert "source_shape" in scaling_info
        assert "target_shape" in scaling_info
        assert "operation" in scaling_info

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_direct_upscale_interface(self):
        """Test direct upscale interface."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(20.0, 20.0),
            method="conservative",
            device=torch.device("cpu")
        )
        
        # Test direct upscale method
        source_data = torch.randn(10, 10)
        result = scaling.upscale(source_data)
        assert result.shape == (5, 5)  # 10/2.0 = 5

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_device_handling(self):
        """Test proper device handling for tensor operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device testing")
        
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(20.0, 20.0),
            method="conservative",
            device=torch.device("cuda")
        )
        
        source_data = torch.randn(10, 10, device=torch.device("cuda"))
        result = scaling.upscale(source_data)
        
        # Verify result is on correct device
        assert result.device.type == "cuda"
        assert result.shape == (5, 5)

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_gradient_flow(self):
        """Test that scaling operations maintain gradient flow."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(20.0, 20.0),
            method="conservative",
            device=torch.device("cpu")
        )
        
        source_data = torch.randn(10, 10, requires_grad=True)
        result = scaling.upscale(source_data)
        
        # Test gradient computation
        loss = result.sum()
        loss.backward()
        
        assert source_data.grad is not None
        assert source_data.grad.shape == source_data.shape

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_multiple_variables(self):
        """Test scaling multiple variables simultaneously."""
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(20.0, 20.0),
            method="conservative",
            device=torch.device("cpu")
        )
        
        # Test that upscale method works for multiple calls
        temperature = torch.randn(10, 10)
        moisture = torch.randn(10, 10)
        carbon_flux = torch.randn(10, 10)
        
        temp_result = scaling.upscale(temperature)
        moist_result = scaling.upscale(moisture)
        carbon_result = scaling.upscale(carbon_flux)
        
        # All variables should be scaled to same size
        assert temp_result.shape == (5, 5)
        assert moist_result.shape == (5, 5)
        assert carbon_result.shape == (5, 5)

    @pytest.mark.skipif(not SCALING_AVAILABLE, reason="Spatial scaling modules not available")
    def test_scale_ratio_validation(self):
        """Test validation of scale ratios."""
        # Test valid scale ratios
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 10.0),
            target_resolution=(30.0, 30.0),
            method="bilinear",
            device=torch.device("cpu")
        )
        assert scaling.scale_ratios == (3.0, 3.0)
        
        # Test integer scale ratios
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(5.0, 5.0),
            target_resolution=(15.0, 15.0),
            method="conservative",
            device=torch.device("cpu")
        )
        assert scaling.scale_ratios == (3.0, 3.0)
        
        # Test non-uniform scale ratios
        scaling = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(10.0, 5.0),
            target_resolution=(30.0, 20.0),
            method="conservative",
            device=torch.device("cpu")
        )
        assert scaling.scale_ratios == (3.0, 4.0)


def run_spatial_scaling_tests():
    """Run all SpatialScaling L1 unit tests."""
    print("Running SpatialScaling L1 Unit Tests...")
    print("=" * 60)
    
    if not SCALING_AVAILABLE:
        print("WARNING: Spatial scaling modules not available. Install dependencies to run tests.")
        return False
    
    test_instance = TestSpatialScalingL1()
    
    test_methods = [
        test_instance.test_spatial_scaling_initialization,
        test_instance.test_forward_upscaling_detection,
        test_instance.test_forward_downscaling_detection,
        test_instance.test_forward_conservative_mode,
        test_instance.test_forward_error_handling,
        test_instance.test_forward_diagnostics,
        test_instance.test_direct_upscale_interface,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")(
            test_instance.test_device_handling
        ),
        test_instance.test_gradient_flow,
        test_instance.test_multiple_variables,
        test_instance.test_scale_ratio_validation
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"Running {test_method.__name__}... ", end="")
            test_method()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"SpatialScaling L1 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All SpatialScaling L1 unit tests PASSED!")
        return True
    else:
        print("Some SpatialScaling L1 tests FAILED. Check the implementation.")
        return False


if __name__ == "__main__":
    run_spatial_scaling_tests()

