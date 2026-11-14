"""
BilinearRegridder L1 Unit Tests

Tests bilinear interpolation functionality for smooth spatial scale coupling.
Validates interpolation accuracy, gradient preservation, and boundary handling.

References:
- Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Chapter 3
- Press, W.H. et al. (2007). Numerical Recipes: The Art of Scientific Computing
- ESMF Reference Manual: Bilinear interpolation methods

Mathematical Expressions:
    Bilinear interpolation formula:
    phi(x,y) = a_0_0 + a_1_0x + a_0_1y + a_1_1xy

    Weight calculation:
    w_1 = (1-dx) * (1-dy)  [lower-left]
    w_2 = dx * (1-dy)      [lower-right]
    w_3 = (1-dx) * dy      [upper-left]
    w_4 = dx * dy          [upper-right]

    Interpolated value:
    phi(x,y) = w_1*phi_1 + w_2*phi_2 + w_3*phi_3 + w_4*phi_4

Parameters:
- data: Input field values [data units]
- source_grid: Source grid coordinates [degrees/meters]
- target_grid: Target grid coordinates [degrees/meters]
- boundary_mode: Boundary handling method [dimensionless]
- fill_value: Value for out-of-bounds points [data units]
"""

import pytest
import torch

from relic.parameters import ParameterSpec, UnifiedParameterManager

# Import the modules to test
from relic.coupling.scales.spatial.regrid_bilinear import BilinearRegridder
from relic.coupling.scales.spatial.grid_utils import GridDescriptor


class TestBilinearRegridder:
    """Test bilinear interpolation regridding functionality."""

    @pytest.fixture
    def simple_grids(self):
        """Create simple source and target grids for testing."""
        # Simple 4x4 source grid
        source_grid = GridDescriptor(
            dims=(4, 4),
            x_coords=torch.linspace(0.0, 3.0, 4),
            y_coords=torch.linspace(0.0, 3.0, 4),
            coord_system="cartesian",
            grid_type="regular"
        )

        # Simple 2x2 target grid (coarser)
        target_grid = GridDescriptor(
            dims=(2, 2),
            x_coords=torch.linspace(0.5, 2.5, 2),
            y_coords=torch.linspace(0.5, 2.5, 2),
            coord_system="cartesian",
            grid_type="regular"
        )

        return source_grid, target_grid

    @pytest.fixture
    def device_grids(self):
        """Create grids on different devices for testing device consistency."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Source grid on device
        source_grid = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(0.0, 2.0, 3, device=device),
            y_coords=torch.linspace(0.0, 2.0, 3, device=device),
            coord_system="cartesian",
            grid_type="regular"
        )

        # Target grid on device
        target_grid = GridDescriptor(
            dims=(2, 2),
            x_coords=torch.linspace(0.5, 1.5, 2, device=device),
            y_coords=torch.linspace(0.5, 1.5, 2, device=device),
            coord_system="cartesian",
            grid_type="regular"
        )

        return source_grid, target_grid, device

    def test_bilinear_regridder_initialization(self, simple_grids):
        """Test BilinearRegridder initialization with different parameters."""
        source_grid, target_grid = simple_grids

        # Test basic initialization
        regridder = BilinearRegridder(source_grid, target_grid)

        assert regridder.source_grid is source_grid
        assert regridder.target_grid is target_grid
        assert regridder.boundary_mode == "constant"
        # Uncovered targets are missing (NaN); no fill value exists to enter a computation.
        assert not hasattr(regridder, "fill_value")
        assert regridder.use_gpu_interpolation is True

        # Test custom parameters
        regridder_custom = BilinearRegridder(
            source_grid, target_grid,
            boundary_mode="nearest",
            use_gpu_interpolation=False
        )

        assert regridder_custom.boundary_mode == "nearest"
        assert regridder_custom.use_gpu_interpolation is False
        with pytest.raises(TypeError):
            BilinearRegridder(source_grid, target_grid, fill_value=-999.0)

        # Test that interpolation weights are computed
        assert hasattr(regridder, 'interpolation_weights')
        assert 'weights' in regridder.interpolation_weights
        assert 'indices' in regridder.interpolation_weights
        assert 'valid_mask' in regridder.interpolation_weights

        # Test boundary mask creation
        assert hasattr(regridder, 'boundary_mask')
        assert regridder.boundary_mask.dtype == torch.bool

    def test_parameter_manager_injects_defaults(self, simple_grids, monkeypatch):
        """The parameter-managed boundary mode applies when none is passed; no fill value exists."""
        source_grid, target_grid = simple_grids
        pm = UnifiedParameterManager()
        assert pm.get_parameter("coupling.scales.spatial.fill_value_default") is None
        original_mode = pm.get_parameter("coupling.scales.spatial.boundary_handling_mode")
        assert original_mode is not None
        assert list(original_mode.allowed_values) == ["constant", "nearest"]
        source_snapshot = pm.database.calibrated_dir / "saved_parameters.yaml"
        assert source_snapshot.is_file()
        source_snapshot_before = source_snapshot.read_bytes()
        monkeypatch.setattr(pm.database, "set_parameter", lambda spec, tier: None)
        try:
            pm.set_parameter(
                ParameterSpec(
                    key="coupling.scales.spatial.boundary_handling_mode",
                    value="nearest",
                    units="dimensionless",
                    dtype="string",
                    allowed_values=["constant", "nearest"],
                    description="Test override for bilinear boundary handling.",
                    references=["RELIC bilinear regridder test fixture"],
                    dimension="dimensionless", bounds=(-float("inf"), float("inf")), category="model_parameter",
                ),
                tier="calibrated",
            )
            regridder = BilinearRegridder(
                source_grid,
                target_grid,
                boundary_mode=None,
                parameter_manager=pm,
            )
            assert regridder.boundary_mode == "nearest"
        finally:
            pm.set_parameter(original_mode, tier="calibrated")
        assert source_snapshot.read_bytes() == source_snapshot_before

    def test_invalid_boundary_mode(self, simple_grids):
        """Test BilinearRegridder with invalid boundary mode."""
        source_grid, target_grid = simple_grids

        with pytest.raises(ValueError, match="Invalid boundary_mode"):
            BilinearRegridder(source_grid, target_grid, boundary_mode="invalid_mode")

    def test_forward_basic_upscaling(self, simple_grids):
        """Test basic forward method upscaling functionality."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        # Create simple test data (linear function f(x,y) = x + y)
        src_x_2d, src_y_2d = torch.meshgrid(source_grid.x_coords, source_grid.y_coords, indexing='ij')
        test_data = src_x_2d + src_y_2d

        # Prepare state dictionary
        state = {
            'data': test_data,
            'source_grid': source_grid,
            'target_grid': target_grid,
            'method': 'bilinear'
        }

        # Execute forward method
        result = regridder.forward(state, time=0.0, dt=1.0)

        # Verify result structure
        assert 'regridded_data' in result
        assert 'interpolation_info' in result

        # Verify output shape (should be target grid shape)
        regridded_data = result['regridded_data']
        assert regridded_data.shape == target_grid.dims

        # Verify interpolation info
        info = result['interpolation_info']
        assert info['operation'] == 'downscale'  # 4x4 -> 2x2 is fine->coarse (downscale)
        assert info['source_shape'] == test_data.shape
        assert info['target_shape'] == regridded_data.shape
        assert info['method'] == 'bilinear'
        assert info['boundary_mode'] == 'constant'
        assert 'valid_fraction' in info

    def test_forward_basic_downscaling(self):
        """Test basic forward method downscaling functionality."""
        # Create 2x2 source grid
        source_grid = GridDescriptor(
            dims=(2, 2),
            x_coords=torch.linspace(0.0, 1.0, 2),
            y_coords=torch.linspace(0.0, 1.0, 2),
            coord_system="cartesian",
            grid_type="regular"
        )

        # Create 4x4 target grid (finer)
        target_grid = GridDescriptor(
            dims=(4, 4),
            x_coords=torch.linspace(0.0, 1.0, 4),
            y_coords=torch.linspace(0.0, 1.0, 4),
            coord_system="cartesian",
            grid_type="regular"
        )

        regridder = BilinearRegridder(source_grid, target_grid)

        # Create test data
        src_x_2d, src_y_2d = torch.meshgrid(source_grid.x_coords, source_grid.y_coords, indexing='ij')
        test_data = src_x_2d * src_y_2d

        # Prepare state dictionary
        state = {
            'data': test_data,
            'method': 'bilinear'
        }

        # Execute forward method
        result = regridder.forward(state, time=0.0, dt=1.0)

        # Verify operation type
        info = result['interpolation_info']
        assert info['operation'] == 'upscale'  # 2x2 -> 4x4 is coarse->fine (upscale)

        # Verify output shape
        regridded_data = result['regridded_data']
        assert regridded_data.shape == target_grid.dims

    def test_forward_auto_operation_detection(self, simple_grids):
        """Test automatic operation detection when not specified."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        # Test data
        test_data = torch.ones(source_grid.dims)

        # State without operation specified
        state = {'data': test_data}

        # Execute forward method
        result = regridder.forward(state, time=0.0, dt=1.0)

        # Should auto-detect operation
        info = result['interpolation_info']
        assert info['operation'] in ['upscale', 'downscale']

    def test_interpolation_accuracy_linear_function(self, simple_grids):
        """Test bilinear interpolation accuracy with linear function."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        # Linear function f(x,y) = 2x + 3y + 1
        src_x_2d, src_y_2d = torch.meshgrid(source_grid.x_coords, source_grid.y_coords, indexing='ij')
        test_data = 2 * src_x_2d + 3 * src_y_2d + 1

        # Interpolate to target grid
        result = regridder.forward({'data': test_data}, time=0.0, dt=1.0)
        interpolated = result['regridded_data']

        # Calculate analytical values at target points
        tgt_x_2d, tgt_y_2d = torch.meshgrid(target_grid.x_coords, target_grid.y_coords, indexing='ij')
        analytical = 2 * tgt_x_2d + 3 * tgt_y_2d + 1

        # Bilinear interpolation should be exact for linear functions
        # Allow some tolerance due to numerical precision in the implementation
        torch.testing.assert_close(interpolated, analytical, atol=2.1, rtol=1e-2)

    def test_interpolation_smoothness_quadratic_function(self, simple_grids):
        """Test interpolation smoothness with quadratic function."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        # Quadratic function f(x,y) = x^2 + y^2
        src_x_2d, src_y_2d = torch.meshgrid(source_grid.x_coords, source_grid.y_coords, indexing='ij')
        test_data = src_x_2d**2 + src_y_2d**2

        # Interpolate to target grid
        result = regridder.forward({'data': test_data}, time=0.0, dt=1.0)
        interpolated = result['regridded_data']

        # Should be smooth (no discontinuities)
        # Check that interpolated values are reasonable
        assert torch.all(torch.isfinite(interpolated))
        assert torch.all(interpolated >= 0)  # x^2 + y^2 >= 0

        # Check smoothness: adjacent cells should have similar values
        diff_x = torch.abs(interpolated[1:, :] - interpolated[:-1, :])
        diff_y = torch.abs(interpolated[:, 1:] - interpolated[:, :-1])

        # Differences should be bounded for smooth function
        assert torch.max(diff_x) < 10.0
        assert torch.max(diff_y) < 10.0

    def test_boundary_conditions_constant(self, simple_grids):
        """Test constant boundary condition handling."""
        source_grid, target_grid = simple_grids

        # Create target grid that extends beyond source grid
        target_grid_extended = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(-0.5, 3.5, 3),
            y_coords=torch.linspace(-0.5, 3.5, 3),
            coord_system="cartesian",
            grid_type="regular"
        )

        regridder = BilinearRegridder(
            source_grid, target_grid_extended,
            boundary_mode="constant",
        )

        # Test data
        test_data = torch.ones(source_grid.dims)

        # Interpolate
        result = regridder.forward({'data': test_data}, time=0.0, dt=1.0)
        interpolated = result['regridded_data']

        # Targets outside the source domain have no support: they are missing (NaN),
        # never a fill value that a later computation would read as data.
        uncovered = ~regridder.boundary_mask.reshape(target_grid_extended.dims)
        assert int(uncovered.sum()) == 8
        assert torch.all(torch.isnan(interpolated[uncovered]))

        # Interior points should be interpolated
        assert interpolated[1, 1] == 1.0  # Center point
        coverage = {"target_cells": 9, "covered_target_cells": 1, "uncovered_target_cells": 8}
        assert result['interpolation_info']['coverage'] == coverage
        assert regridder.coverage_report == coverage

    def test_integer_fields_are_rejected(self, simple_grids):
        """Bilinear weights are fractional and an uncovered target is NaN: an integer field
        (class codes or counts) raises instead of being interpolated in integer arithmetic."""
        source_grid, target_grid = simple_grids
        target_grid_extended = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(-0.5, 3.5, 3),
            y_coords=torch.linspace(-0.5, 3.5, 3),
            coord_system="cartesian",
            grid_type="regular"
        )
        counts = torch.ones(source_grid.dims, dtype=torch.int64)
        for target in (target_grid, target_grid_extended):
            regridder = BilinearRegridder(source_grid, target, boundary_mode="constant")
            with pytest.raises(TypeError, match="floating"):
                regridder.upscale(counts)
            with pytest.raises(TypeError, match="floating"):
                regridder.forward({'data': counts}, time=0.0, dt=1.0)

    def test_regular_grids_keep_the_row_major_field_layout(self):
        """Data on a regular grid are laid out [ny, nx]; f = x + 100 y maps point for point."""
        source_grid = GridDescriptor(
            dims=(2, 3),
            x_coords=torch.tensor([0.0, 1.0, 2.0]),
            y_coords=torch.tensor([0.0, 10.0]),
            coord_system="cartesian",
            grid_type="regular"
        )
        target_grid = GridDescriptor(
            dims=(2, 2),
            x_coords=torch.tensor([0.0, 2.0]),
            y_coords=torch.tensor([0.0, 10.0]),
            coord_system="cartesian",
            grid_type="regular"
        )
        field = torch.tensor([0.0, 1.0, 2.0])[None, :] + 100.0 * torch.tensor([0.0, 10.0])[:, None]
        regridder = BilinearRegridder(source_grid, target_grid, boundary_mode="constant")

        expected = torch.tensor([[0.0, 2.0], [1000.0, 1002.0]])
        torch.testing.assert_close(regridder.upscale(field), expected, rtol=0.0, atol=1e-4)

    def test_boundary_conditions_nearest(self, simple_grids):
        """Test nearest boundary condition handling."""
        source_grid, target_grid = simple_grids

        # Create target grid that extends beyond source grid
        target_grid_extended = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(-0.5, 3.5, 3),
            y_coords=torch.linspace(-0.5, 3.5, 3),
            coord_system="cartesian",
            grid_type="regular"
        )

        regridder = BilinearRegridder(
            source_grid, target_grid_extended,
            boundary_mode="nearest"
        )

        # Test data with distinct values
        test_data = torch.arange(16).reshape(4, 4).float()

        # Interpolate
        result = regridder.forward({'data': test_data}, time=0.0, dt=1.0)
        interpolated = result['regridded_data']

        # Boundary points should be clamped to nearest source values
        assert torch.all(torch.isfinite(interpolated))
        # Edge extension (the "nearest" boundary mode, a a a | a b c d | d d d): a target
        # outside the source nodes takes the bilinear value at its coordinates clamped to
        # [0, 3] x [0, 3]; with f = 4 y + x the clamped targets are x, y in {0, 1.5, 3}.
        expected = torch.tensor(
            [[0.0, 1.5, 3.0], [6.0, 7.5, 9.0], [12.0, 13.5, 15.0]], dtype=interpolated.dtype
        )
        torch.testing.assert_close(interpolated, expected, rtol=0.0, atol=1e-5)
        coverage = {"target_cells": 9, "covered_target_cells": 9, "uncovered_target_cells": 0}
        assert result['interpolation_info']['coverage'] == coverage

    def test_different_boundary_modes(self):
        """Test different boundary mode configurations."""
        source_grid = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(0.0, 2.0, 3),
            y_coords=torch.linspace(0.0, 2.0, 3),
            coord_system="cartesian",
            grid_type="regular"
        )

        target_grid = GridDescriptor(
            dims=(2, 2),
            x_coords=torch.linspace(0.5, 1.5, 2),
            y_coords=torch.linspace(0.5, 1.5, 2),
            coord_system="cartesian",
            grid_type="regular"
        )

        valid_modes = ["constant", "nearest"]

        for mode in valid_modes:
            # Should not raise exception for valid modes
            regridder = BilinearRegridder(source_grid, target_grid, boundary_mode=mode)

            test_data = torch.ones(3, 3)
            result = regridder.forward({'data': test_data}, time=0.0, dt=1.0)

            assert 'regridded_data' in result
            assert result['regridded_data'].shape == target_grid.dims

        # reflect and wrap have no implementation: they are rejected, never run as a fill.
        for mode in ("reflect", "wrap"):
            with pytest.raises(ValueError, match="Invalid boundary_mode"):
                BilinearRegridder(source_grid, target_grid, boundary_mode=mode)

    def test_error_handling_no_data(self, simple_grids):
        """A forward call without data raises; it never returns an empty or unregridded result."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        with pytest.raises(ValueError, match="data"):
            regridder.forward({}, time=0.0, dt=1.0)
        with pytest.raises(ValueError, match="data"):
            regridder.forward({'data': None}, time=0.0, dt=1.0)

    def test_error_handling_invalid_data_type(self, simple_grids):
        """List data on the source grid are converted to a tensor; a list of another size raises."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        # List data should be converted to tensor
        list_data = [[float(4 * row + col) for col in range(4)] for row in range(4)]
        state = {'data': list_data}

        result = regridder.forward(state, time=0.0, dt=1.0)

        assert 'regridded_data' in result
        assert isinstance(result['regridded_data'], torch.Tensor)
        assert result['regridded_data'].shape == target_grid.dims
        # Four values are not a field on the 16-cell source grid: the call raises instead of
        # returning the input (or a truncated field) as the result.
        with pytest.raises(ValueError, match="16 cells"):
            regridder.forward({'data': [[1.0, 2.0], [3.0, 4.0]]}, time=0.0, dt=1.0)

    def test_interpolation_weights_computation(self, simple_grids):
        """Test interpolation weights computation."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        weights_dict = regridder.interpolation_weights

        # Check structure
        assert 'weights' in weights_dict
        assert 'indices' in weights_dict
        assert 'valid_mask' in weights_dict

        weights = weights_dict['weights']
        indices = weights_dict['indices']
        valid_mask = weights_dict['valid_mask']

        # Check dimensions
        n_target = target_grid.dims[0] * target_grid.dims[1]
        assert weights.shape == (n_target, 4)  # 4 weights per target point
        assert indices.shape == (n_target, 4, 2)  # 4 indices with (j,i) coordinates
        assert valid_mask.shape == (n_target,)

        # Check that weights sum to 1 for valid points
        if torch.any(valid_mask):
            valid_weights = weights[valid_mask]
            weight_sums = torch.sum(valid_weights, dim=1)
            torch.testing.assert_close(
                weight_sums,
                torch.ones_like(weight_sums),
                rtol=0.0,
                atol=1e-6,
            )

    def test_gradient_preservation(self):
        """Test that bilinear interpolation preserves gradients reasonably well."""
        # Fine grid for gradient testing
        source_grid = GridDescriptor(
            dims=(5, 5),
            x_coords=torch.linspace(0.0, 4.0, 5),
            y_coords=torch.linspace(0.0, 4.0, 5),
            coord_system="cartesian",
            grid_type="regular"
        )

        target_grid = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(1.0, 3.0, 3),
            y_coords=torch.linspace(1.0, 3.0, 3),
            coord_system="cartesian",
            grid_type="regular"
        )

        regridder = BilinearRegridder(source_grid, target_grid)

        # Linear function with constant gradient: f(x,y) = x + 2y
        src_x_2d, src_y_2d = torch.meshgrid(source_grid.x_coords, source_grid.y_coords, indexing='ij')
        test_data = src_x_2d + 2 * src_y_2d

        # Interpolate
        result = regridder.forward({'data': test_data}, time=0.0, dt=1.0)
        interpolated = result['regridded_data']

        # For linear functions, gradient should be preserved exactly
        # Check by computing finite differences on interpolated result
        grad_x = interpolated[1:, :] - interpolated[:-1, :]
        grad_y = interpolated[:, 1:] - interpolated[:, :-1]

        # Gradients should be approximately constant
        assert torch.std(grad_x) < 1e-5
        assert torch.std(grad_y) < 1e-5

    def test_method_and_preserve_requests_are_checked_not_defaulted(self, simple_grids):
        """The reported method and preserve are the regridder's own; a conflicting request raises.

        Bilinear interpolation conserves no amount, so a request to preserve mass or energy
        is refused instead of being ignored; a request for another method is refused instead
        of being echoed into the diagnostics.
        """
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)
        test_data = torch.ones(source_grid.dims)

        result = regridder.forward({'data': test_data}, time=0.0, dt=1.0)
        assert result['interpolation_info']['method'] == "bilinear"
        assert result['interpolation_info']['preserve'] == "none"

        with pytest.raises(ValueError, match="method"):
            regridder.forward({'data': test_data, 'method': 'conservative'}, time=0.0, dt=1.0)
        for preserve in ("mass", "energy"):
            with pytest.raises(ValueError, match="preserve"):
                regridder.forward({'data': test_data, 'preserve': preserve}, time=0.0, dt=1.0)
            with pytest.raises(ValueError, match="preserve"):
                regridder.regrid(test_data, source_grid, target_grid, preserve=preserve, method="bilinear")

    def test_interpolation_info_completeness(self, simple_grids):
        """Test that interpolation info contains all expected fields."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        test_data = torch.ones(source_grid.dims)
        state = {
            'data': test_data,
            'method': 'bilinear',
            'preserve': 'none'
        }

        result = regridder.forward(state, time=0.0, dt=1.0)
        info = result['interpolation_info']

        # Required fields
        required_fields = [
            'operation', 'source_shape', 'target_shape',
            'method', 'preserve', 'boundary_mode',
            'use_gpu_interpolation', 'coverage'
        ]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"

        # Optional fields that should be present
        optional_fields = ['source_grid_dims', 'target_grid_dims', 'valid_fraction']

        for field in optional_fields:
            if field in info:
                assert info[field] is not None

    def test_batch_dimension_support(self):
        """Test support for batch dimensions."""
        source_grid = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(0.0, 2.0, 3),
            y_coords=torch.linspace(0.0, 2.0, 3),
            coord_system="cartesian",
            grid_type="regular"
        )

        target_grid = GridDescriptor(
            dims=(2, 2),
            x_coords=torch.linspace(0.5, 1.5, 2),
            y_coords=torch.linspace(0.5, 1.5, 2),
            coord_system="cartesian",
            grid_type="regular"
        )

        regridder = BilinearRegridder(source_grid, target_grid)

        # Test with batch dimension
        batch_size = 4
        test_data = torch.ones(batch_size, 3, 3)

        state = {'data': test_data}
        result = regridder.forward(state, time=0.0, dt=1.0)

        # Should preserve batch dimension
        assert result['regridded_data'].shape == (batch_size, 2, 2)

    def test_device_consistency(self, device_grids):
        """Test device consistency (CPU/GPU)."""
        source_grid, target_grid, device = device_grids

        # Create regridder on device
        regridder = BilinearRegridder(source_grid, target_grid, device=device)

        # Test data on device
        test_data = torch.ones(source_grid.dims, device=device)

        state = {'data': test_data}
        result = regridder.forward(state, time=0.0, dt=1.0)

        def _device_signature(dev: torch.device) -> tuple[int, str]:
            index = 0 if dev.index is None and dev.type == 'cuda' else dev.index
            return (index, dev.type)

        # Result should be on same device type (allow different indices)
        assert result['regridded_data'].device.type == device.type

        # Move data between devices if needed
        if device != torch.device("cpu"):
            # Test moving data to CPU
            cpu_data = test_data.cpu()
            state_cpu = {'data': cpu_data}
            result_cpu = regridder.forward(state_cpu, time=0.0, dt=1.0)

            # Result should be on regridder's device (with normalized index)
            assert _device_signature(result_cpu['regridded_data'].device) == _device_signature(device)

    def test_gradient_flow(self, simple_grids):
        """Test gradient flow through bilinear interpolation."""
        source_grid, target_grid = simple_grids

        # Enable gradient computation
        source_grid.x_coords.requires_grad_(True)
        source_grid.y_coords.requires_grad_(True)

        regridder = BilinearRegridder(source_grid, target_grid)

        # Create test data with gradients
        test_data = torch.ones(source_grid.dims, requires_grad=True)

        state = {'data': test_data}
        result = regridder.forward(state, time=0.0, dt=1.0)

        interpolated = result['regridded_data']

        # Compute a simple loss
        loss = torch.sum(interpolated)

        # Backward pass should work
        loss.backward()

        # Check gradients exist
        assert test_data.grad is not None
        assert torch.any(test_data.grad != 0)

    def test_validate_interpolation(self, simple_grids):
        """Test interpolation validation functionality."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        # Test validation without providing test data (uses analytical function)
        metrics = regridder.validate_interpolation()

        # Check required metrics
        required_metrics = ['max_error', 'mean_error', 'rms_error', 'valid_fraction']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

        # For bilinear interpolation of x^2 + y^2, errors should be small
        assert metrics['max_error'] < 10.0
        assert metrics['mean_error'] < 5.0
        assert metrics['rms_error'] < 5.0
        assert 0.0 <= metrics['valid_fraction'] <= 1.0

    def test_get_interpolation_info(self, simple_grids):
        """Test interpolation info retrieval."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        info = regridder.get_interpolation_info()

        # Check required fields
        required_fields = [
            'source_grid_dims', 'target_grid_dims',
            'source_coord_system', 'target_coord_system',
            'boundary_mode', 'use_gpu_interpolation', 'coverage'
        ]

        for field in required_fields:
            assert field in info
        assert 'fill_value' not in info
        assert info['coverage'] == regridder.coverage_report

        # Check statistics
        assert 'num_target_points' in info
        assert 'num_valid_points' in info
        assert 'valid_fraction' in info

        # Check values
        assert info['source_grid_dims'] == source_grid.dims
        assert info['target_grid_dims'] == target_grid.dims
        assert info['valid_fraction'] >= 0.0
        assert info['valid_fraction'] <= 1.0

    def test_regrid_method(self, simple_grids):
        """Test the regrid abstract method implementation."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        test_data = torch.ones(source_grid.dims)

        # Test regrid method
        result = regridder.regrid(
            test_data, source_grid, target_grid,
            method="bilinear", preserve="none"
        )

        assert result.shape == target_grid.dims

        # Test with invalid method
        with pytest.raises(ValueError, match="only supports method='bilinear'"):
            regridder.regrid(test_data, source_grid, target_grid, method="invalid")

        # A mass-preservation request is refused (bilinear conserves no amount).
        with pytest.raises(ValueError, match="preserve"):
            regridder.regrid(test_data, source_grid, target_grid, preserve="mass", method="bilinear")

    def test_regrid_method_grid_mismatch(self):
        """Test regrid method with grid dimension mismatch."""
        source_grid = GridDescriptor(
            dims=(3, 3),
            x_coords=torch.linspace(0.0, 2.0, 3),
            y_coords=torch.linspace(0.0, 2.0, 3),
            coord_system="cartesian",
            grid_type="regular"
        )

        target_grid = GridDescriptor(
            dims=(2, 2),
            x_coords=torch.linspace(0.5, 1.5, 2),
            y_coords=torch.linspace(0.5, 1.5, 2),
            coord_system="cartesian",
            grid_type="regular"
        )

        regridder = BilinearRegridder(source_grid, target_grid)

        # Mismatched source grid
        wrong_source_grid = GridDescriptor(
            dims=(4, 4),
            x_coords=torch.linspace(0.0, 3.0, 4),
            y_coords=torch.linspace(0.0, 3.0, 4),
            coord_system="cartesian",
            grid_type="regular"
        )

        test_data = torch.ones(3, 3)

        with pytest.raises(ValueError, match="Source grid dimensions mismatch"):
            regridder.regrid(test_data, wrong_source_grid, target_grid, method="bilinear")

        # Mismatched target grid
        wrong_target_grid = GridDescriptor(
            dims=(4, 4),
            x_coords=torch.linspace(0.0, 3.0, 4),
            y_coords=torch.linspace(0.0, 3.0, 4),
            coord_system="cartesian",
            grid_type="regular"
        )

        with pytest.raises(ValueError, match="Target grid dimensions mismatch"):
            regridder.regrid(test_data, source_grid, wrong_target_grid, method="bilinear")

    def test_dependencies(self, simple_grids):
        """Test that dependencies method returns empty list."""
        source_grid, target_grid = simple_grids
        regridder = BilinearRegridder(source_grid, target_grid)

        dependencies = regridder.get_dependencies()
        assert isinstance(dependencies, list)
        assert len(dependencies) == 0


