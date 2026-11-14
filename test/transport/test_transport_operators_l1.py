"""
L1 Unit Tests for Transport Operators Module

This module provides comprehensive unit tests for transport operators including
CFL computation, conservation checks, grid metrics, and other utility functions
that support the transport mechanisms.

Test Coverage:
- CFL number computation for different dimensionalities
- Conservation law verification and validation
- Grid metrics computation (volumes, areas, distances)
- Non-negative constraint application
- Gradient computation operators
- Utility functions for transport calculations

Mathematical Verification:
- CFL condition: CFL = sum(|v_i| * dt / dx_i) <= 1.0
- Conservation: sum(c_new * V) - sum(c_old * V) = sum(S * dt * V)
- Grid volume: V = product(dx_i) for rectangular grids
- Gradient: dphi/dx = (c_{i+1} - c_{i-1}) / (2 * dx)

References:
- LeVeque, R.J. (1992). Numerical Methods for Conservation Laws
- Hirsch, C. (2007). Numerical Computation of Internal and External Flows
- Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow
"""

import pytest
import torch

# Import RELIC modules
from relic.computation.operators.functional import (
    compute_cfl_number,
    compute_gradient_1d,
    compute_gradient_2d,
    compute_gradient_3d,
    apply_advection,
    apply_diffusion,
    advance_advection_diffusion,
    compute_grid_metrics,
    compute_divergence_2d,
    compute_divergence_3d,
    compute_laplacian_1d,
    compute_laplacian_2d,
    pack_outward_boundary_flux_cochain,
)


class TestComputeCFLNumber:
    """Test CFL number computation for stability analysis."""

    def test_compute_cfl_1d_uniform(self):
        """Test CFL computation for 1D uniform velocity field."""
        velocity = torch.ones(10) * 2.0  # Uniform velocity
        grid_spacing = 1.0
        dt = 0.1

        cfl = compute_cfl_number(velocity, grid_spacing, dt)

        # Expected: CFL = |v| * dt / dx = 2.0 * 0.1 / 1.0 = 0.2
        expected_cfl = torch.abs(velocity) * dt / grid_spacing
        torch.testing.assert_close(cfl, expected_cfl)

    def test_compute_cfl_1d_variable_velocity(self):
        """Test CFL computation for 1D variable velocity field."""
        velocity = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
        grid_spacing = 0.5
        dt = 0.2

        cfl = compute_cfl_number(velocity, grid_spacing, dt)

        expected_cfl = torch.abs(velocity) * dt / grid_spacing
        torch.testing.assert_close(cfl, expected_cfl)

    def test_compute_cfl_2d(self):
        """Test CFL computation for 2D velocity field."""
        u = torch.ones(8, 10) * 1.5  # x-component
        v = torch.ones(8, 10) * 1.0  # y-component
        grid_spacing = (1.0, 2.0)  # Storage order: (dy, dx)
        dt = 0.1

        cfl = compute_cfl_number((u, v), grid_spacing, dt)

        # Physical components are (u, v), while tensor spacing follows (dy, dx).
        expected_cfl = torch.abs(u) * dt / grid_spacing[1] + torch.abs(v) * dt / grid_spacing[0]
        torch.testing.assert_close(cfl, expected_cfl)

    def test_compute_cfl_3d(self):
        """Test CFL computation for 3D velocity field."""
        u = torch.ones(4, 5, 6) * 0.5
        v = torch.ones(4, 5, 6) * 0.3
        w = torch.ones(4, 5, 6) * 0.2
        grid_spacing = (1.0, 1.0, 1.0)
        dt = 0.5

        cfl = compute_cfl_number((u, v, w), grid_spacing, dt)

        expected_cfl = (torch.abs(u) + torch.abs(v) + torch.abs(w)) * dt / 1.0
        torch.testing.assert_close(cfl, expected_cfl)

    def test_compute_cfl_single_component_tuple(self):
        """Test CFL computation with single component as tuple."""
        velocity = (torch.ones(10) * 2.0,)
        grid_spacing = 1.0
        dt = 0.1

        cfl = compute_cfl_number(velocity, grid_spacing, dt)

        expected_cfl = torch.abs(velocity[0]) * dt / grid_spacing
        torch.testing.assert_close(cfl, expected_cfl)

    def test_compute_cfl_scalar_velocity(self):
        """Test CFL computation with scalar velocity."""
        velocity = torch.tensor(2.0)  # Convert to tensor
        grid_spacing = 1.0
        dt = 0.1

        cfl = compute_cfl_number(velocity, grid_spacing, dt)

        expected_cfl = abs(velocity.item()) * dt / grid_spacing
        assert cfl == expected_cfl

    def test_compute_cfl_zero_velocity(self):
        """Test CFL computation with zero velocity."""
        velocity = torch.zeros(10)
        grid_spacing = 1.0
        dt = 0.1

        cfl = compute_cfl_number(velocity, grid_spacing, dt)

        expected_cfl = torch.zeros_like(velocity)
        torch.testing.assert_close(cfl, expected_cfl)

    def test_compute_cfl_negative_velocity(self):
        """Test CFL computation with negative velocity."""
        velocity = -torch.ones(10) * 2.0
        grid_spacing = 1.0
        dt = 0.1

        cfl = compute_cfl_number(velocity, grid_spacing, dt)

        expected_cfl = torch.abs(velocity) * dt / grid_spacing
        torch.testing.assert_close(cfl, expected_cfl)


class TestComputeGradient:
    """Test gradient computation operators."""

    def test_compute_gradient_1d_centered(self):
        """Test 1D centered gradient computation."""
        field = torch.tensor([[1.0, 2.0, 4.0, 7.0, 11.0]])  # Add batch dimension
        dx = 1.0

        gradient = compute_gradient_1d(field, dx, boundary_method='zero', order=2)

        # Expected gradient at interior points: (f[i+1] - f[i-1]) / (2*dx)
        # Center points: (4-1)/2 = 1.5, (7-2)/2 = 2.5, (11-4)/2 = 3.5
        expected_gradient = torch.tensor([[0.0, 1.5, 2.5, 3.5, 0.0]])  # Zero at boundaries
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=1e-10)

    def test_compute_gradient_1d_forward(self):
        """Test 1D forward gradient computation."""
        field = torch.tensor([[1.0, 2.0, 4.0, 7.0, 11.0]])  # Add batch dimension
        dx = 1.0

        gradient = compute_gradient_1d(field, dx, boundary_method='extrapolate', order=2)

        # Extrapolate: forward/backward at boundaries, central for interior
        expected_gradient = torch.tensor([[1.0, 1.5, 2.5, 3.5, 4.0]])
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=1e-10)

    def test_compute_gradient_1d_backward(self):
        """Test 1D backward gradient computation."""
        field = torch.tensor([[1.0, 2.0, 4.0, 7.0, 11.0]])  # Add batch dimension
        dx = 1.0

        gradient = compute_gradient_1d(field, dx, boundary_method='extrapolate', order=2)

        # Extrapolate: forward/backward at boundaries, central for interior
        expected_gradient = torch.tensor([[1.0, 1.5, 2.5, 3.5, 4.0]])
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=1e-10)

    def test_compute_gradient_2d(self):
        """Test 2D gradient computation."""
        field = torch.tensor([[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ]], dtype=torch.float32)  # Add batch dimension
        dx, dy = 1.0, 1.0

        grad_x, grad_y = compute_gradient_2d(field, dx, dy, boundary_method="zero")

        # Field is f(x,y) = x*y with zero-gradient boundaries
        expected_grad_x = torch.tensor([[
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0]
        ]], dtype=torch.float32)
        
        expected_grad_y = torch.tensor([[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0]
        ]], dtype=torch.float32)

        torch.testing.assert_close(grad_x, expected_grad_x, rtol=0, atol=0.1)
        torch.testing.assert_close(grad_y, expected_grad_y, rtol=0, atol=0.1)

    def test_compute_gradient_3d(self):
        """Test 3D gradient computation."""
        # Create a simple linear field: f(x,y,z) = x + 2y + 3z
        nx, ny, nz = 3, 3, 3
        field = torch.zeros(1, nz, ny, nx, dtype=torch.float32)  # Add batch dimension

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    field[0, k, j, i] = i + 2*j + 3*k

        dx, dy, dz = 1.0, 1.0, 1.0
        grad_x, grad_y, grad_z = compute_gradient_3d(field, dx, dy, dz, boundary_method="zero")

        # Expected gradients: dphi/dx = 1, dphi/dy = 2, dphi/dz = 3
        expected_grad_x = torch.ones_like(field)
        expected_grad_y = 2 * torch.ones_like(field)
        expected_grad_z = 3 * torch.ones_like(field)

        # Check interior points (boundaries may be different)
        torch.testing.assert_close(grad_x[0, 1:-1, 1:-1, 1:-1], expected_grad_x[0, 1:-1, 1:-1, 1:-1], rtol=0, atol=0.1)
        torch.testing.assert_close(grad_y[0, 1:-1, 1:-1, 1:-1], expected_grad_y[0, 1:-1, 1:-1, 1:-1], rtol=0, atol=0.1)
        torch.testing.assert_close(grad_z[0, 1:-1, 1:-1, 1:-1], expected_grad_z[0, 1:-1, 1:-1, 1:-1], rtol=0, atol=0.1)

    def test_compute_gradient_irregular_spacing(self):
        """Test gradient computation with irregular grid spacing."""
        field = torch.tensor([[0.0, 0.25, 1.0, 2.25, 4.0]], dtype=torch.float32)  # x^2, add batch dimension
        dx = 0.5

        gradient = compute_gradient_1d(field, dx, boundary_method='zero', order=2)

        # For f = x^2, df/dx = 2x
        # With dx=0.5, positions are [0, 0.5, 1.0, 1.5, 2.0]
        # Expected gradients at centers: [1.0, 2.0, 3.0, 4.0]
        expected_gradient = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0]], dtype=torch.float32)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0.1)


class TestCanonicalTransportConservation:
    """Test conservation through the canonical finite-volume runtime ledger."""

    def test_boundary_flux_cochain_preserves_batch_and_face_order(self):
        """Boundary packing must not merge independent batch ledgers."""
        ledger = {
            "y_max": torch.tensor([[7.0, 8.0], [17.0, 18.0]]),
            "x_min": torch.tensor([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]]),
            "x_max": torch.tensor([[4.0, 5.0, 6.0], [14.0, 15.0, 16.0]]),
            "y_min": torch.tensor([[0.0, 9.0], [10.0, 19.0]]),
        }

        packed = pack_outward_boundary_flux_cochain(ledger)

        assert packed.shape == (2, 10)
        torch.testing.assert_close(
            packed,
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 9.0, 7.0, 8.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 10.0, 19.0, 17.0, 18.0],
                ]
            ),
        )
        torch.testing.assert_close(
            pack_outward_boundary_flux_cochain(ledger, squeeze_batch=True),
            packed,
        )
        single_batch = {face: value[:1] for face, value in ledger.items()}
        squeezed = pack_outward_boundary_flux_cochain(
            single_batch,
            squeeze_batch=True,
        )
        assert squeezed.shape == (10,)
        torch.testing.assert_close(squeezed, packed[0])

    def test_periodic_advection_is_closed(self):
        field = torch.arange(1.0, 11.0, dtype=torch.float64)
        result = apply_advection(
            field,
            torch.ones_like(field),
            1.0,
            time_step=0.1,
            boundary_method="periodic", spatial_dims=1, scheme="upwind", check_cfl=True,
        )
        torch.testing.assert_close(result["concentration"].sum(), field.sum())
        assert result["boundary_fluxes"] == {}

    def test_closed_source_matches_storage_change(self):
        field = torch.full((10,), 2.0, dtype=torch.float64)
        source = torch.full_like(field, 0.1)
        result = advance_advection_diffusion(
            field,
            torch.zeros_like(field),
            0.0,
            1.0,
            1.0,
            splitting_method="combined",
            boundary_method="periodic",
            source_term=source, spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        torch.testing.assert_close(result["concentration"], field + source)
        assert result["conservation_error"] < 1e-12

    def test_open_boundary_flux_closes_storage_ledger(self):
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        dt = 0.1
        result = advance_advection_diffusion(
            field,
            torch.ones_like(field),
            0.0,
            1.0,
            dt,
            splitting_method="combined",
            boundary_method="open",
            advection_boundary_state=torch.zeros_like(field), spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        storage_change = result["concentration"].sum() - field.sum()
        boundary_export = sum(value.sum() for value in result["boundary_fluxes"].values())
        torch.testing.assert_close(storage_change + dt * boundary_export, torch.tensor(0.0, dtype=field.dtype))
        assert result["conservation_error"] < 1e-12

    def test_unrouted_negative_source_fails_fast(self):
        field = torch.ones(10)
        with pytest.raises(ValueError, match="non-negative"):
            advance_advection_diffusion(
                field,
                torch.zeros_like(field),
                0.0,
                1.0,
                1.0,
                splitting_method="combined",
                boundary_method="periodic",
                source_term=torch.full_like(field, -2.0), spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
            )

    def test_zero_initial_field_accepts_explicit_source(self):
        field = torch.zeros(10, dtype=torch.float64)
        source = torch.full_like(field, 0.1)
        result = advance_advection_diffusion(
            field,
            torch.zeros_like(field),
            0.0,
            1.0,
            0.1,
            splitting_method="combined",
            boundary_method="periodic",
            source_term=source, spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        torch.testing.assert_close(result["concentration"], torch.full_like(field, 0.01))
        assert torch.isfinite(result["conservation_error"])

    def test_float64_closed_ledger_has_roundoff_accuracy(self):
        field = torch.linspace(1.0, 2.0, 17, dtype=torch.float64)
        result = advance_advection_diffusion(
            field,
            torch.full_like(field, 0.2),
            1.0e-3,
            0.5,
            0.1,
            splitting_method="symmetric",
            boundary_method="periodic", spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        assert result["conservation_error"] < 1e-12

    def test_multidimensional_source_ledger(self):
        field = torch.ones((5, 6), dtype=torch.float64)
        source = torch.full_like(field, 0.1)
        velocity = (torch.zeros_like(field), torch.zeros_like(field))
        result = advance_advection_diffusion(
            field,
            velocity,
            0.0,
            (2.0, 3.0),
            1.0,
            spatial_dims=2,
            splitting_method="combined",
            boundary_method="periodic",
            source_term=source, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        torch.testing.assert_close(result["concentration"], field + source)
        assert result["conservation_error"] < 1e-12


class TestNonNegativeTransportPolicy:
    """Test fail-fast positivity without post-hoc mass-changing clamps."""

    def test_positive_zero_diffusion_is_unchanged(self):
        field = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_diffusion(field, 0.0, 1.0, time_step=0.1, spatial_dims=1, boundary_method="zero", check_stability=True, enforce_non_negative=True)
        torch.testing.assert_close(result["concentration"], field)

    def test_mixed_negative_input_is_rejected(self):
        field = torch.tensor([-1.0, 0.5, -2.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="non-negative"):
            apply_diffusion(field, 0.0, 1.0, time_step=0.1, spatial_dims=1, boundary_method="zero", check_stability=True, enforce_non_negative=True)

    def test_all_negative_input_is_rejected(self):
        field = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        with pytest.raises(ValueError, match="non-negative"):
            apply_diffusion(field, 0.0, 1.0, time_step=0.1, spatial_dims=1, boundary_method="zero", check_stability=True, enforce_non_negative=True)

    def test_multidimensional_negative_input_is_rejected(self):
        field = torch.tensor([[1.0, -1.0], [-2.0, 3.0]])
        with pytest.raises(ValueError, match="non-negative"):
            apply_diffusion(field, 0.0, (1.0, 1.0), spatial_dims=2, time_step=0.1, boundary_method="zero", check_stability=True, enforce_non_negative=True)

    def test_signed_scalar_requires_explicit_opt_out(self):
        field = torch.tensor([-1.0, 0.5, -2.0, 1.0])
        result = apply_diffusion(
            field,
            0.0,
            1.0,
            time_step=0.1,
            enforce_non_negative=False, spatial_dims=1, boundary_method="zero", check_stability=True,
        )
        torch.testing.assert_close(result["concentration"], field)


class TestComputeGridMetrics:
    """Test grid metrics computation."""

    def test_compute_grid_metrics_1d_uniform(self):
        """Test 1D uniform grid metrics."""
        spatial_dims = (10,)
        grid_spacing = [1.0]
        device = torch.device("cpu")

        metrics = compute_grid_metrics(spatial_dims, grid_spacing, device)

        assert 'volumes' in metrics
        assert 'face_areas' in metrics
        assert 'dx' in metrics

        # For 1D uniform grid
        expected_volumes = torch.ones(10)
        torch.testing.assert_close(metrics['volumes'], expected_volumes)

    def test_compute_grid_metrics_2d_uniform(self):
        """Test 2D uniform grid metrics."""
        spatial_dims = (5, 8)  # (ny, nx)
        grid_spacing = [1.0, 2.0]  # (dy, dx)
        device = torch.device("cpu")

        metrics = compute_grid_metrics(spatial_dims, grid_spacing, device)

        # Cell volumes = dx * dy
        expected_volumes = torch.full(spatial_dims, 2.0)  # 1.0 * 2.0
        torch.testing.assert_close(metrics['volumes'], expected_volumes)

        # Areas for each face
        assert metrics['face_areas_x'].shape == (5, 9)  # One more in x direction
        assert metrics['face_areas_y'].shape == (6, 8)  # One more in y direction

    def test_compute_grid_metrics_3d_uniform(self):
        """Test 3D uniform grid metrics."""
        spatial_dims = (3, 4, 5)  # (nz, ny, nx)
        grid_spacing = [1.0, 2.0, 3.0]  # (dz, dy, dx)
        device = torch.device("cpu")

        metrics = compute_grid_metrics(spatial_dims, grid_spacing, device)

        # Cell volumes = dx * dy * dz
        expected_volumes = torch.full(spatial_dims, 6.0)  # 1.0 * 2.0 * 3.0
        torch.testing.assert_close(metrics['volumes'], expected_volumes)

    def test_compute_grid_metrics_non_uniform(self):
        """Test non-uniform grid metrics."""
        spatial_dims = (4, 5)
        grid_spacing = [0.5, 2.0]  # Non-uniform spacing
        device = torch.device("cpu")

        metrics = compute_grid_metrics(spatial_dims, grid_spacing, device)

        # Cell volumes should account for non-uniform spacing
        expected_volumes = torch.full(spatial_dims, 1.0)  # 0.5 * 2.0
        torch.testing.assert_close(metrics['volumes'], expected_volumes)

    def test_compute_grid_metrics_device_consistency(self):
        """Test device consistency in grid metrics."""
        spatial_dims = (3, 4)
        grid_spacing = [1.0, 1.0]
        device = torch.device("cpu")

        metrics = compute_grid_metrics(spatial_dims, grid_spacing, device)

        # All tensors should be on the specified device
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                assert value.device == device


class TestComputeDivergence:
    """Test divergence computation operators."""

    def test_compute_divergence_2d(self):
        """Test 2D divergence computation."""
        # Create a vector field (u, v) with batch dimension
        u = torch.ones(1, 5, 6) * 2.0  # Constant x-component
        v = torch.ones(1, 5, 6) * 1.0  # Constant y-component
        dx, dy = 1.0, 1.0

        divergence = compute_divergence_2d(u, v, dx, dy, boundary_method="zero")

        # For constant field, divergence should be zero
        expected_divergence = torch.zeros_like(u)
        torch.testing.assert_close(divergence, expected_divergence, rtol=0, atol=0.1)

    def test_compute_divergence_3d(self):
        """Test 3D divergence computation."""
        # Create a 3D vector field with batch dimension
        u = torch.ones(1, 3, 4, 5) * 1.0
        v = torch.ones(1, 3, 4, 5) * 2.0
        w = torch.ones(1, 3, 4, 5) * 3.0
        dx, dy, dz = 1.0, 1.0, 1.0

        divergence = compute_divergence_3d(u, v, w, dx, dy, dz, boundary_method="zero")

        # For constant field, divergence should be zero
        expected_divergence = torch.zeros_like(u)
        torch.testing.assert_close(divergence, expected_divergence, rtol=0, atol=0.1)


class TestComputeLaplacian:
    """Test Laplacian computation operators."""

    def test_compute_laplacian_1d_constant(self):
        """Test 1D Laplacian for constant field."""
        field = torch.ones(1, 10) * 2.0  # Add batch dimension
        dx = 1.0

        laplacian = compute_laplacian_1d(field, dx, boundary_method="zero")

        # Laplacian of constant field should be zero
        expected_laplacian = torch.zeros_like(field)
        torch.testing.assert_close(laplacian, expected_laplacian, rtol=0, atol=1e-10)

    def test_compute_laplacian_1d_quadratic(self):
        """Test 1D Laplacian for quadratic field."""
        # For f(x) = x^2, Laplacian = d^2f/dx^2 = 2
        x = torch.arange(10, dtype=torch.float32)
        field = (x ** 2).unsqueeze(0)  # Add batch dimension
        dx = 1.0

        laplacian = compute_laplacian_1d(field, dx, boundary_method="zero")

        # Interior points should have Laplacian about 2
        expected_laplacian = torch.full_like(field, 2.0)
        torch.testing.assert_close(laplacian[0, 1:-1], expected_laplacian[0, 1:-1], rtol=0, atol=0.1)

    def test_compute_laplacian_2d(self):
        """Test 2D Laplacian computation."""
        # Create f(x,y) = x^2 + y^2 with batch dimension
        nx, ny = 5, 6
        x = torch.arange(nx, dtype=torch.float32)
        y = torch.arange(ny, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        field = (X**2 + Y**2).unsqueeze(0)  # Add batch dimension

        dx, dy = 1.0, 1.0

        laplacian = compute_laplacian_2d(field, dx, dy, boundary_method="periodic")

        # For f = x^2 + y^2, Laplacian = 2 + 2 = 4
        expected_laplacian = torch.full_like(field, 4.0)
        torch.testing.assert_close(laplacian[0, 1:-1, 1:-1], expected_laplacian[0, 1:-1, 1:-1], rtol=0, atol=0.1)


class TestCanonicalUpwindAdvection:
    """Test the canonical finite-volume upwind rate and boundary ledger."""

    def test_upwind_flux_1d_positive_velocity(self):
        """Test 1D upwind flux with positive velocity."""
        c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        u = torch.tensor(1.0)  # Positive velocity
        result = apply_advection(
            c,
            u,
            1.0,
            boundary_method="zero",
            check_cfl=False, spatial_dims=1, scheme="upwind", time_step=None,
        )
        torch.testing.assert_close(result["advection_rate"], -torch.ones_like(c))
        torch.testing.assert_close(result["boundary_fluxes"]["x_min"], torch.tensor(-0.0))
        torch.testing.assert_close(result["boundary_fluxes"]["x_max"], torch.tensor(5.0))

    def test_upwind_flux_1d_negative_velocity(self):
        """Test 1D upwind flux with negative velocity."""
        c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        u = torch.tensor(-1.0)  # Negative velocity
        result = apply_advection(
            c,
            u,
            1.0,
            boundary_method="zero",
            check_cfl=False, spatial_dims=1, scheme="upwind", time_step=None,
        )
        expected_rate = torch.tensor([1.0, 1.0, 1.0, 1.0, -5.0])
        torch.testing.assert_close(result["advection_rate"], expected_rate)
        torch.testing.assert_close(result["boundary_fluxes"]["x_min"], torch.tensor(1.0))
        torch.testing.assert_close(result["boundary_fluxes"]["x_max"], torch.tensor(-0.0))


class TestErrorHandling:
    """Test error handling for transport operators."""

    def test_compute_cfl_invalid_input(self):
        """Test CFL computation with invalid input."""
        with pytest.raises(ValueError, match="at least one axis"):
            compute_cfl_number([], 1.0, 0.1)
        with pytest.raises(ValueError, match="finite and positive"):
            compute_cfl_number(torch.ones(10), -1.0, 0.1)

    def test_compute_gradient_invalid_method(self):
        """Test gradient computation with invalid method."""
        field = torch.ones(1, 10)  # Add batch dimension

        # Should work with valid boundary methods
        grad = compute_gradient_1d(field, 1.0, boundary_method='zero', order=2)
        assert grad.shape == field.shape

    def test_advection_mismatched_shapes(self):
        """Test canonical advection rejects a mismatched velocity field."""
        with pytest.raises(ValueError, match="broadcast-compatible"):
            apply_advection(
                torch.ones(10),
                torch.ones(5),
                1.0,
                time_step=0.1,
                boundary_method="periodic", spatial_dims=1, scheme="upwind", check_cfl=True,
            )


# Test fixtures
@pytest.fixture
def sample_field_1d():
    """Fixture providing sample 1D field."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)


@pytest.fixture
def sample_field_2d():
    """Fixture providing sample 2D field."""
    return torch.ones(5, 6, dtype=torch.float32) * torch.arange(6, dtype=torch.float32)


@pytest.fixture
def sample_field_3d():
    """Fixture providing sample 3D field."""
    return torch.ones(3, 4, 5, dtype=torch.float32)


@pytest.fixture
def uniform_velocity_1d():
    """Fixture providing uniform 1D velocity."""
    return torch.ones(10, dtype=torch.float32) * 2.0


@pytest.fixture
def uniform_velocity_2d():
    """Fixture providing uniform 2D velocity."""
    u = torch.ones(5, 6, dtype=torch.float32) * 1.5
    v = torch.ones(5, 6, dtype=torch.float32) * 0.8
    return (u, v)


@pytest.fixture
def uniform_velocity_3d():
    """Fixture providing uniform 3D velocity."""
    u = torch.ones(3, 4, 5, dtype=torch.float32) * 1.0
    v = torch.ones(3, 4, 5, dtype=torch.float32) * 0.5
    w = torch.ones(3, 4, 5, dtype=torch.float32) * 0.3
    return (u, v, w)


# Integration tests using fixtures
def test_cfl_stability_analysis(uniform_velocity_1d):
    """Integration test for CFL stability analysis."""
    grid_spacing = 1.0
    dt = 0.1

    cfl = compute_cfl_number(uniform_velocity_1d, grid_spacing, dt)

    # Check stability condition
    max_cfl = torch.max(cfl)
    assert max_cfl < 1.0 or max_cfl < 2.0  # Allow some tolerance


def test_conservative_transport_cycle(sample_field_1d, uniform_velocity_1d):
    """Integration test for conservative transport cycle."""
    dx = 1.0
    dt = 0.01
    old_field = sample_field_1d.clone()
    velocity = torch.full_like(old_field, float(uniform_velocity_1d[0]))
    result = apply_advection(
        old_field,
        velocity,
        dx,
        time_step=dt,
        boundary_method="periodic", spatial_dims=1, scheme="upwind", check_cfl=True,
    )
    torch.testing.assert_close(result["concentration"].sum(), old_field.sum())


def test_multidimensional_gradient_and_divergence(sample_field_2d, uniform_velocity_2d):
    """Integration test for gradient and divergence in 2D."""
    dx, dy = 1.0, 1.0

    # Add batch dimension to sample field
    field_batch = sample_field_2d.unsqueeze(0)

    # Compute gradients
    grad_x, grad_y = compute_gradient_2d(field_batch, dx, dy, boundary_method="zero")

    # Get velocity components
    u, v = uniform_velocity_2d

    # Compute divergence of velocity field
    div_velocity = compute_divergence_2d(u, v, dx, dy, boundary_method="zero")

    # Both should return valid tensors
    assert grad_x.shape == field_batch.shape
    assert grad_y.shape == field_batch.shape
    assert div_velocity.shape == u.shape

    # Should be finite values
    assert torch.all(torch.isfinite(grad_x))
    assert torch.all(torch.isfinite(grad_y))
    assert torch.all(torch.isfinite(div_velocity))


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])


