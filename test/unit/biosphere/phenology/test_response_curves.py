"""
Unit Tests for Response Curve Functions

Tests mathematical response functions used in phenological models with focus on
mathematical accuracy, edge cases, and PyTorch tensor consistency.

Scientific Validation Focus:
- Mathematical function accuracy against analytical solutions
- Edge case behavior (infinity, zero, negative values)
- Gradient computation for differentiability
- Parameter sensitivity and numerical stability
- Absence of a duplicate GSI component formula surface

Test Organization:
- Individual response function tests
- Cross-function consistency
- Tensor operation validation
- Gradient computation tests
- Performance and numerical stability
"""

import pytest
import torch

import relic.scientific.biosphere.phenology.common.response_curves as response_curves

from relic.scientific.biosphere.phenology.common.response_curves import (
    sigmoid_response,
    tanh_response,
    linear_threshold_response,
    exponential_response,
    step_response,
    double_sigmoid_response,
    growth_onset_response,
    senescence_response,
    temperature_stress_response
)


class TestSigmoidResponse:
    """Test sigmoid response function."""
    
    def test_basic_sigmoid(self):
        """Test basic sigmoid functionality."""
        x = torch.linspace(-10, 10, 21)
        response = sigmoid_response(x, x_center=0.0, steepness=1.0)
        
        # Check output shape
        assert response.shape == x.shape
        
        # Check center point
        center_idx = len(x) // 2
        assert torch.isclose(response[center_idx], torch.tensor(0.5), atol=1e-6)
        
        # Check bounds
        assert torch.all(response >= 0.0)
        assert torch.all(response <= 1.0)
        
        # Check monotonicity
        assert torch.all(response[1:] >= response[:-1])
    
    def test_sigmoid_parameters(self):
        """Test sigmoid parameter effects."""
        x = torch.linspace(-5, 15, 101)
        
        # Test center shift
        resp1 = sigmoid_response(x, x_center=0.0)
        resp2 = sigmoid_response(x, x_center=5.0)
        
        # Center shift should align overlapping segments without wraparound artifacts
        shift_steps = 25
        assert torch.allclose(resp1[:-shift_steps], resp2[shift_steps:], atol=1e-3)
        
        # Test steepness
        steep_low = sigmoid_response(x, x_center=5.0, steepness=0.5)
        steep_high = sigmoid_response(x, x_center=5.0, steepness=2.0)
        
        # Higher steepness should have sharper transition
        # Check variance as proxy for sharpness
        assert torch.var(steep_high) > torch.var(steep_low)
    
    def test_sigmoid_bounds(self):
        """Test sigmoid with custom bounds."""
        x = torch.linspace(-5, 5, 51)
        response = sigmoid_response(x, x_center=0.0, y_min=0.2, y_max=0.8)
        
        # Check custom bounds
        assert torch.all(response >= 0.2 - 1e-6)
        assert torch.all(response <= 0.8 + 1e-6)
        
        # Check center value
        center_idx = len(x) // 2
        expected_center = (0.2 + 0.8) / 2
        assert torch.isclose(response[center_idx], torch.tensor(expected_center), atol=1e-5)
    
    def test_sigmoid_edge_cases(self):
        """Test sigmoid edge cases."""
        # Single value
        x_scalar = torch.tensor(2.5)
        resp_scalar = sigmoid_response(x_scalar, x_center=2.5)
        assert torch.isclose(resp_scalar, torch.tensor(0.5), atol=1e-6)
        
        # Zero steepness should give constant 0.5
        x = torch.linspace(-5, 5, 11)
        resp_zero = sigmoid_response(x, x_center=0.0, steepness=0.0)
        assert torch.allclose(resp_zero, torch.tensor(0.5), atol=1e-6)
        
        # Very high steepness should approximate step function
        resp_steep = sigmoid_response(x, x_center=0.0, steepness=100.0)
        expected_step = torch.where(
            x > 0.0,
            torch.ones_like(resp_steep),
            torch.where(
                x < 0.0,
                torch.zeros_like(resp_steep),
                torch.full_like(resp_steep, 0.5),
            ),
        )
        assert torch.allclose(resp_steep, expected_step, atol=1e-2)


class TestTanhResponse:
    """Test hyperbolic tangent response function."""
    
    def test_basic_tanh(self):
        """Test basic tanh functionality."""
        x = torch.linspace(-5, 5, 51)
        response = tanh_response(x, x_center=0.0, steepness=1.0)
        
        # Check output shape and bounds
        assert response.shape == x.shape
        assert torch.all(response >= 0.0)
        assert torch.all(response <= 1.0)
        
        # Check center point
        center_idx = len(x) // 2
        assert torch.isclose(response[center_idx], torch.tensor(0.5), atol=1e-6)
        
        # Check monotonicity
        assert torch.all(response[1:] >= response[:-1])
    
    def test_tanh_vs_sigmoid(self):
        """Compare tanh and sigmoid responses."""
        x = torch.linspace(-3, 3, 61)
        sigmoid_resp = sigmoid_response(x, x_center=0.0, steepness=1.0)
        tanh_resp = tanh_response(x, x_center=0.0, steepness=1.0)
        
        # Both should be similar but tanh should be slightly different
        # They should have same center value
        center_idx = len(x) // 2
        assert torch.isclose(sigmoid_resp[center_idx], tanh_resp[center_idx], atol=1e-6)
        
        # Both should be monotonic
        assert torch.all(sigmoid_resp[1:] >= sigmoid_resp[:-1])
        assert torch.all(tanh_resp[1:] >= tanh_resp[:-1])
    
    def test_tanh_custom_bounds(self):
        """Test tanh with custom bounds."""
        x = torch.linspace(-2, 8, 51)
        response = tanh_response(x, x_center=3.0, y_min=0.1, y_max=0.9)
        
        assert torch.all(response >= 0.1 - 1e-6)
        assert torch.all(response <= 0.9 + 1e-6)


class TestLinearThresholdResponse:
    """Test piecewise linear threshold response."""
    
    def test_basic_linear_threshold(self):
        """Test basic linear threshold functionality."""
        x = torch.linspace(-5, 15, 21)
        response = linear_threshold_response(x, x_min=0.0, x_max=10.0)
        
        # Check segments
        low_mask = x <= 0.0
        high_mask = x >= 10.0
        mid_mask = (x > 0.0) & (x < 10.0)
        
        assert torch.all(response[low_mask] == 0.0)
        assert torch.all(response[high_mask] == 1.0)
        assert torch.all((response[mid_mask] > 0.0) & (response[mid_mask] < 1.0))
        
        # Check linearity in middle segment
        x_mid = torch.linspace(2, 8, 7)
        resp_mid = linear_threshold_response(x_mid, x_min=0.0, x_max=10.0)
        
        # Linear segment should have constant differences
        diff = resp_mid[1:] - resp_mid[:-1]
        assert torch.allclose(diff[0], diff, atol=1e-6)
    
    def test_linear_threshold_bounds(self):
        """Test linear threshold with custom bounds."""
        x = torch.linspace(-2, 12, 15)
        response = linear_threshold_response(x, x_min=2.0, x_max=8.0, y_min=0.3, y_max=0.7)
        
        # Check custom bounds
        low_mask = x <= 2.0
        high_mask = x >= 8.0
        
        assert torch.allclose(response[low_mask], torch.tensor(0.3), atol=1e-6)
        assert torch.allclose(response[high_mask], torch.tensor(0.7), atol=1e-6)
        
        # Check exact values at boundaries
        x_boundary = torch.tensor([2.0, 5.0, 8.0])
        resp_boundary = linear_threshold_response(x_boundary, x_min=2.0, x_max=8.0, y_min=0.3, y_max=0.7)
        
        expected = torch.tensor([0.3, 0.5, 0.7])
        assert torch.allclose(resp_boundary, expected, atol=1e-6)
    
    def test_linear_threshold_edge_cases(self):
        """Test linear threshold edge cases."""
        # Invalid range should raise error
        with pytest.raises(ValueError, match="x_max must be greater than x_min"):
            x = torch.tensor([1.0, 2.0, 3.0])
            linear_threshold_response(x, x_min=5.0, x_max=2.0)
        
        # Equal x_min and x_max should raise error
        with pytest.raises(ValueError):
            x = torch.tensor([1.0, 2.0, 3.0])
            linear_threshold_response(x, x_min=3.0, x_max=3.0)


class TestExponentialResponse:
    """Test exponential response function."""
    
    def test_basic_exponential(self):
        """Test basic exponential functionality."""
        x = torch.linspace(0, 5, 11)
        response = exponential_response(x, rate=1.0)
        
        # Check exponential growth
        assert torch.all(response[1:] >= response[:-1])
        
        # Check specific values
        assert torch.isclose(response[0], torch.tensor(1.0), atol=1e-6)  # exp(0) = 1
        
        # Check rate effect
        resp_slow = exponential_response(x, rate=0.5)
        resp_fast = exponential_response(x, rate=2.0)
        
        # Faster rate should give higher values for x > 0
        assert torch.all(resp_fast[1:] >= resp_slow[1:])
    
    def test_saturating_exponential(self):
        """Test saturating exponential."""
        x = torch.linspace(0, 5, 11)
        response = exponential_response(x, rate=1.0, saturating=True)
        
        # Check bounds for saturating exponential
        assert torch.all(response >= 0.0)
        assert torch.all(response <= 1.0)
        
        # Check asymptotic behavior
        assert torch.isclose(response[0], torch.tensor(0.0), atol=1e-6)
        assert response[-1] > 0.99  # Should approach 1.0
    
    def test_exponential_offsets(self):
        """Test exponential with offsets."""
        x = torch.linspace(1, 6, 11)
        response = exponential_response(x, rate=0.5, x_offset=1.0, y_offset=2.0)
        
        # Check x_offset effect
        assert torch.isclose(response[0], torch.tensor(3.0), atol=1e-6)  # 2 + exp(0)
        
        # Check y_offset effect
        assert torch.all(response >= 2.0)


class TestStepResponse:
    """Test step function response."""
    
    def test_basic_step(self):
        """Test basic step function."""
        x = torch.linspace(-2, 8, 21)
        response = step_response(x, threshold=3.0)
        
        # Check step behavior
        low_mask = x < 3.0
        high_mask = x >= 3.0
        
        assert torch.all(response[low_mask] == 0.0)
        assert torch.all(response[high_mask] == 1.0)
    
    def test_step_custom_values(self):
        """Test step function with custom values."""
        x = torch.linspace(0, 10, 11)
        response = step_response(x, threshold=5.0, y_low=0.2, y_high=0.8)
        
        low_mask = x < 5.0
        high_mask = x >= 5.0
        
        assert torch.all(response[low_mask] == 0.2)
        assert torch.all(response[high_mask] == 0.8)


class TestDoubleSigmoidResponse:
    """Test double sigmoid response for seasonal patterns."""
    
    def test_basic_double_sigmoid(self):
        """Test basic double sigmoid functionality."""
        x = torch.linspace(0, 365, 365)  # Day of year
        response = double_sigmoid_response(
            x, x_onset=120, x_offset=280, 
            steepness_onset=0.1, steepness_offset=0.1
        )
        
        # Should have peak somewhere between onset and offset
        peak_idx = torch.argmax(response)
        peak_day = x[peak_idx]
        assert 120 < peak_day < 280
        
        # Should start low, increase, then decrease
        early_response = response[:60].mean()  # Before onset
        late_response = response[300:].mean()  # After offset
        peak_response = response[peak_idx]
        
        assert peak_response > early_response
        assert peak_response > late_response
    
    def test_double_sigmoid_parameters(self):
        """Test double sigmoid parameter effects."""
        x = torch.linspace(50, 350, 301)
        
        # Test onset/offset timing
        early_response = double_sigmoid_response(x, x_onset=100, x_offset=250)
        late_response = double_sigmoid_response(x, x_onset=150, x_offset=300)
        
        early_peak_idx = torch.argmax(early_response)
        late_peak_idx = torch.argmax(late_response)
        
        assert early_peak_idx < late_peak_idx
        
        # Test steepness effects
        gradual_response = double_sigmoid_response(
            x, x_onset=120, x_offset=280, 
            steepness_onset=0.05, steepness_offset=0.05
        )
        sharp_response = double_sigmoid_response(
            x, x_onset=120, x_offset=280,
            steepness_onset=0.2, steepness_offset=0.2
        )
        
        # Sharp response should have higher peak
        assert torch.max(sharp_response) > torch.max(gradual_response)
    
    def test_double_sigmoid_bounds(self):
        """Test double sigmoid with custom bounds."""
        x = torch.linspace(0, 365, 365)
        response = double_sigmoid_response(
            x, x_onset=120, x_offset=280,
            y_base=0.1, y_peak=0.9
        )
        
        # Check bounds
        assert torch.all(response >= 0.1 - 1e-6)
        assert torch.all(response <= 0.9 + 1e-6)
        
        # Check that peak approaches y_peak
        assert torch.max(response) > 0.8  # Should get close to 0.9


class TestGSIResponseComponents:
    """Keep GSI response equations in the canonical phenology utility only."""

    def test_duplicate_gsi_response_component_is_absent(self):
        """Reject the obsolete four-component helper surface."""
        assert not hasattr(response_curves, "gsi_response_component")


class TestConvenienceFunctions:
    """Test convenience functions for common responses."""
    
    def test_growth_onset_response(self):
        """Test growth onset response function."""
        doy = torch.linspace(1, 365, 365)
        response = growth_onset_response(doy, onset_doy=120.0, steepness=0.2)
        
        # Should be sigmoid with appropriate onset
        assert response.shape == doy.shape
        assert torch.all(response >= 0.0)
        assert torch.all(response <= 1.0)
        
        # Check onset timing
        onset_response = growth_onset_response(torch.tensor(120.0), onset_doy=120.0)
        assert torch.isclose(onset_response, torch.tensor(0.5), atol=1e-6)
    
    def test_senescence_response(self):
        """Test senescence response function."""
        doy = torch.linspace(200, 365, 166)
        response = senescence_response(doy, offset_doy=280.0, steepness=0.2)
        
        # Should be inverted sigmoid - high to low
        assert response.shape == doy.shape
        assert torch.all(response >= 0.0)
        assert torch.all(response <= 1.0)
        
        # Should decrease with time
        assert response[0] > response[-1]
        
        # Check offset timing
        offset_response = senescence_response(torch.tensor(280.0), offset_doy=280.0)
        assert torch.isclose(offset_response, torch.tensor(0.5), atol=1e-6)
    
    def test_temperature_stress_response(self):
        """Test temperature stress response function."""
        temp = torch.linspace(268, 318, 51)  # -5掳C to 45掳C
        response = temperature_stress_response(temp, optimal_temp=298.15, tolerance=10.0)
        
        # Should be maximum at optimal temperature
        optimal_idx = torch.argmin(torch.abs(temp - 298.15))
        assert response[optimal_idx] == torch.max(response)
        
        # Should decrease away from optimum
        assert torch.all(response >= 0.0)
        assert torch.all(response <= 1.0)


class TestTensorconsistency:
    """Test tensor operations and device consistency."""
    
    def test_tensor_input_handling(self):
        """Test that functions handle various tensor inputs."""
        # Test with different tensor types
        x_float = torch.linspace(0, 10, 11, dtype=torch.float32)
        x_double = torch.linspace(0, 10, 11, dtype=torch.float64)
        x_list = [0, 1, 2, 3, 4, 5]
        x_scalar = 5.0
        
        for x in [x_float, x_double, x_list, x_scalar]:
            response = sigmoid_response(x, x_center=5.0)
            assert isinstance(response, torch.Tensor)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_consistency(self):
        """Test functions work on CUDA tensors."""
        x_cpu = torch.linspace(0, 10, 11)
        x_cuda = x_cpu.cuda()
        
        resp_cpu = sigmoid_response(x_cpu, x_center=5.0)
        resp_cuda = sigmoid_response(x_cuda, x_center=5.0)
        
        # Results should be equivalent
        assert torch.allclose(resp_cpu, resp_cuda.cpu(), atol=1e-6)
    
    def test_gradient_computation(self):
        """Test that functions support gradient computation."""
        x = torch.linspace(-2, 8, 11, requires_grad=True)
        
        # Test sigmoid gradients
        response = sigmoid_response(x, x_center=3.0, steepness=1.0)
        loss = torch.sum(response)
        loss.backward()
        
        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))
        assert not torch.any(torch.isinf(x.grad))
    
    def test_batch_processing(self):
        """Test functions handle batched inputs."""
        # Test with batch dimension
        x_batch = torch.randn(5, 20)  # 5 batches, 20 points each
        
        response = sigmoid_response(x_batch, x_center=0.0)
        assert response.shape == x_batch.shape
        assert torch.all((response >= 0.0) & (response <= 1.0))


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_extreme_values(self):
        """Test functions with extreme input values."""
        # Test with very large values
        x_large = torch.tensor([1e6, 1e8, 1e10])
        resp_large = sigmoid_response(x_large, x_center=0.0)
        assert torch.allclose(resp_large, torch.tensor(1.0), atol=1e-6)
        
        # Test with very small values
        x_small = torch.tensor([-1e6, -1e8, -1e10])
        resp_small = sigmoid_response(x_small, x_center=0.0)
        assert torch.allclose(resp_small, torch.tensor(0.0), atol=1e-6)
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and infinity values."""
        # Test with NaN
        x_nan = torch.tensor([1.0, float('nan'), 3.0])
        response_nan = sigmoid_response(x_nan, x_center=2.0)
        
        # NaN should propagate
        assert torch.isnan(response_nan[1])
        assert not torch.isnan(response_nan[0])
        assert not torch.isnan(response_nan[2])
        
        # Test with infinity
        x_inf = torch.tensor([1.0, float('inf'), -float('inf')])
        response_inf = sigmoid_response(x_inf, x_center=2.0)
        
        assert torch.isclose(response_inf[1], torch.tensor(1.0), atol=1e-6)  # +inf -> 1
        assert torch.isclose(response_inf[2], torch.tensor(0.0), atol=1e-6)  # -inf -> 0
    
    def test_precision_consistency(self):
        """Test that functions maintain precision across ranges."""
        # Test with high precision requirements
        x = torch.linspace(0, 1, 1001, dtype=torch.float64)
        
        # Linear threshold should be exactly linear in middle
        response = linear_threshold_response(x, x_min=0.2, x_max=0.8, y_min=0.0, y_max=1.0)
        
        # Extract linear portion
        linear_mask = (x >= 0.2) & (x <= 0.8)
        x_linear = x[linear_mask]
        resp_linear = response[linear_mask]
        
        # Check linearity with high precision
        if len(x_linear) > 2:
            diffs = resp_linear[1:] - resp_linear[:-1]
            assert torch.allclose(diffs, diffs[0], atol=1e-10)


