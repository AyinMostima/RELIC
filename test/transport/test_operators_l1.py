"""
L1 Unit Tests for Unified Transport Operators

This module provides comprehensive unit tests for the unified transport operators
in relic.computation.operators.functional. Tests cover:

- Basic functionality of diffusion operators
- Basic functionality of advection operators  
- Combined advection-diffusion operators
- Boundary condition handling
- Numerical stability and conservation
- Error handling and edge cases

References:
    - Crank, J. (1975). The Mathematics of Diffusion. Oxford University Press
    - LeVeque, R.J. (1992). Numerical Methods for Conservation Laws
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations
"""

import dataclasses

import pytest
import torch

from relic.parameters import get_global_manager

# Import the operators to test
from relic.computation.operators.functional import (
    # Differential operators
    compute_gradient_1d, compute_gradient_2d, compute_divergence_2d, compute_laplacian_1d, compute_second_derivative_1d,
    
    # High-level transport operators
    apply_diffusion, apply_advection, advance_advection_diffusion,
    
    # Utility functions
    compute_grid_metrics, compute_cfl_number, compute_diffusion_stability,
)
from relic.computation.operators.functional import (
    _integrate_rate_system,
    _resolve_transport_time_integration,
)

_TIME_INTEGRATION_KEY = "transport.numerical.time_integration_method"


def _serve_time_integration_allowed_values(monkeypatch, allowed_values):
    """Serve the YAML ParameterSpec of the time integration key with other allowed_values.

    The spec is the single source of the accepted methods; replacing its
    allowed_values through the manager's get_parameter must change what the
    transport kernels accept.
    """
    manager = get_global_manager()
    original = manager.get_parameter

    def _patched(key, namespace=None):
        spec = original(key, namespace)
        if key == _TIME_INTEGRATION_KEY:
            return dataclasses.replace(spec, allowed_values=list(allowed_values))
        return spec

    monkeypatch.setattr(manager, "get_parameter", _patched)

# Test fixtures
@pytest.fixture
def device():
    """Default device for tests."""
    return torch.device("cpu")

@pytest.fixture
def grid_1d_small():
    """Small 1D grid for basic tests."""
    return torch.linspace(0, 1, 11)  # 11 points, dx = 0.1

@pytest.fixture
def grid_1d_medium():
    """Medium 1D grid for stability tests."""
    return torch.linspace(0, 1, 101)  # 101 points, dx = 0.01

@pytest.fixture
def grid_2d_small():
    """Small 2D grid for basic tests."""
    x = torch.linspace(0, 1, 11)
    y = torch.linspace(0, 1, 11) 
    return x, y

@pytest.fixture
def concentration_field_1d(grid_1d_small):
    """Sample concentration field for 1D tests."""
    # Gaussian pulse
    center = 0.5
    width = 0.1
    return torch.exp(-((grid_1d_small - center) / width) ** 2)

@pytest.fixture
def concentration_field_2d(grid_2d_small):
    """Sample concentration field for 2D tests."""
    x, y = grid_2d_small
    X, Y = torch.meshgrid(x, y, indexing='ij')
    # 2D Gaussian pulse
    center_x, center_y = 0.5, 0.5
    width_x, width_y = 0.1, 0.1
    return torch.exp(-((X - center_x) / width_x) ** 2 - ((Y - center_y) / width_y) ** 2)

@pytest.fixture
def velocity_field_1d_constant(grid_1d_small):
    """Constant 1D velocity field."""
    return torch.full_like(grid_1d_small, 1.0)  # u = 1.0 m/s

@pytest.fixture
def velocity_field_2d_constant(grid_2d_small):
    """Constant 2D velocity field."""
    x, y = grid_2d_small
    X, Y = torch.meshgrid(x, y, indexing='ij')
    u = torch.ones_like(X)  # u = 1.0 m/s
    v = torch.zeros_like(Y)  # v = 0.0 m/s
    return u, v


class TestDifferentialOperators:
    """Test basic differential operators."""
    
    def test_compute_gradient_1d_constant(self, device):
        """Test gradient of constant field (should be zero)."""
        x = torch.linspace(0, 1, 11, device=device)
        field = torch.ones_like(x).unsqueeze(0)  # Add batch dimension
        grad = compute_gradient_1d(field, 0.1, boundary_method="zero", order=2)
        
        # Gradient of constant should be approximately zero
        assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-10)
        
    def test_compute_gradient_1d_linear(self, device):
        """Test gradient of linear field."""
        x = torch.linspace(0, 1, 11, device=device)
        field = (2 * x + 1).unsqueeze(0)  # Add batch dimension, df/dx = 2
        grad = compute_gradient_1d(field, 0.1, boundary_method="zero", order=2)
        
        # Gradient should be approximately 2 (except at boundaries)
        assert torch.allclose(grad[0, 1:-1], torch.full_like(grad[0, 1:-1], 2.0), atol=0.1)
        
    def test_compute_laplacian_1d_constant(self, device):
        """Test Laplacian of constant field (should be zero)."""
        x = torch.linspace(0, 1, 11, device=device)
        field = torch.ones_like(x)
        lap = compute_laplacian_1d(field, 0.1, boundary_method="zero")
        
        # Laplacian of constant should be approximately zero
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-10)
        
    def test_compute_laplacian_1d_quadratic(self, device):
        """Test Laplacian of quadratic field."""
        x = torch.linspace(0, 1, 11, device=device)
        field = (x ** 2).unsqueeze(0)  # Add batch dimension, d虏f/dx虏 = 2
        lap = compute_laplacian_1d(field, 0.1, boundary_method="zero")
        
        # Laplacian should be approximately 2 (except at boundaries)
        assert torch.allclose(lap[0, 1:-1], torch.full_like(lap[0, 1:-1], 2.0), atol=0.5)

    def test_compute_gradient_1d_rejects_invalid_order(self, device):
        """Unknown finite-difference orders must fail before evaluation."""
        field = torch.ones((1, 7), device=device)

        with pytest.raises(ValueError, match="order"):
            compute_gradient_1d(field, 1.0, order=3, boundary_method="zero")

    def test_compute_gradient_1d_rejects_unknown_boundary(self, device):
        """Gradient boundaries are explicit closed policies, not aliases."""
        field = torch.ones((1, 7), device=device)

        with pytest.raises(ValueError, match="boundary"):
            compute_gradient_1d(field, 1.0, boundary_method="reflect", order=2)

    def test_compute_laplacian_1d_rejects_unknown_boundary(self, device):
        """Laplacian boundaries must fail fast when undeclared."""
        field = torch.ones((1, 7), device=device)

        with pytest.raises(ValueError, match="boundary"):
            compute_laplacian_1d(field, 1.0, boundary_method="reflect")

    def test_second_derivative_rejects_unknown_boundary(self, device):
        """Axis-specific derivatives must not silently zero unknown boundaries."""
        field = torch.ones((1, 7), device=device)

        with pytest.raises(ValueError, match="boundary"):
            compute_second_derivative_1d(field, 1.0, boundary_method="reflect")

    def test_fourth_order_periodic_gradient_closes_all_cells(self, device):
        """Fourth-order periodic differences evaluate the wrapped end cells."""
        point_count = 32
        dx = 2.0 * torch.pi / point_count
        x = torch.arange(point_count, device=device) * dx
        field = torch.sin(x).unsqueeze(0)

        gradient = compute_gradient_1d(
            field,
            float(dx),
            boundary_method="periodic",
            order=4,
        )

        torch.testing.assert_close(
            gradient,
            torch.cos(x).unsqueeze(0),
            rtol=2.0e-4,
            atol=2.0e-4,
        )
        
    def test_compute_gradient_2d_constant(self, device):
        """Test 2D gradient of constant field."""
        x = torch.linspace(0, 1, 11, device=device)
        y = torch.linspace(0, 1, 11, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        field = torch.ones_like(X).unsqueeze(0)  # Add batch dimension
        
        grad_x, grad_y = compute_gradient_2d(field, 0.1, 0.1, boundary_method="zero")
        
        # Gradients of constant should be approximately zero
        assert torch.allclose(grad_x, torch.zeros_like(grad_x), atol=1e-10)
        assert torch.allclose(grad_y, torch.zeros_like(grad_y), atol=1e-10)
        
    def test_compute_divergence_2d_constant(self, device):
        """Test 2D divergence of constant velocity field."""
        x = torch.linspace(0, 1, 11, device=device)
        y = torch.linspace(0, 1, 11, device=device)
        u = torch.ones((1, 11, 11), device=device)  # u = 1, with batch dimension
        v = torch.zeros((1, 11, 11), device=device)  # v = 0, with batch dimension
        
        div = compute_divergence_2d(u, v, 0.1, 0.1, boundary_method="zero")  # Fix function signature
        
        # Divergence of constant field should be approximately zero
        assert torch.allclose(div, torch.zeros_like(div), atol=1e-10)


class TestDiffusionOperator:
    """Test the apply_diffusion high-level operator."""
    
    def test_apply_diffusion_gaussian_field(self, concentration_field_1d, device):
        """Test diffusion of Gaussian field (should spread out)."""
        # Add batch dimension
        conc = concentration_field_1d.unsqueeze(0).to(device)
        
        result = apply_diffusion(
            concentration=conc,
            diffusivity=1e-5,
            grid_spacing=0.1,
            spatial_dims=1,
            time_step=0.01,
            check_stability=True,
            enforce_non_negative=True, boundary_method="zero"
        )
        
        # Diffusion rate should exist and be finite
        delta_conc = result['diffusion_rate']  # Fix key name
        assert torch.all(torch.isfinite(delta_conc))
        
    def test_apply_diffusion_gaussian_pulse(self, concentration_field_1d, device):
        """Test diffusion of Gaussian pulse (should spread out)."""
        conc = concentration_field_1d.unsqueeze(0).to(device)
        
        result = apply_diffusion(
            concentration=conc,
            diffusivity=1e-3,  # Higher diffusivity for visible effect
            grid_spacing=0.1,
            spatial_dims=1,
            time_step=0.01,
            check_stability=True,
            enforce_non_negative=True, boundary_method="zero"
        )
        
        delta_conc = result['diffusion_rate']  # Fix key name
        
        # Peak should decrease (negative change at center)
        center_idx = len(concentration_field_1d) // 2
        assert delta_conc[0, center_idx] < 0
        
        # Should spread outward (positive change near edges of pulse)
        # Verify mass conservation
        total_mass_before = torch.sum(conc)
        total_mass_after = torch.sum(result['concentration'])  # Use returned concentration
        
        # Mass should be approximately conserved
        relative_change = abs(total_mass_after - total_mass_before) / total_mass_before
        assert relative_change < 0.01  # Less than 1% mass change
        
    def test_apply_diffusion_stability_check(self, concentration_field_1d, device):
        """Test that an unstable explicit diffusion step fails fast."""
        conc = concentration_field_1d.unsqueeze(0).to(device)

        with pytest.raises(ValueError, match="Diffusion stability violation"):
            apply_diffusion(
                concentration=conc,
                diffusivity=1e-2,
                grid_spacing=0.1,
                spatial_dims=1,
                time_step=10.0,
                check_stability=True,
                enforce_non_negative=True, boundary_method="zero",
            )

    def test_apply_diffusion_multidimensional_stability_uses_axis_sum(self, device):
        """Test the multidimensional explicit stability sum."""
        concentration = torch.ones((1, 4, 4), device=device)

        with pytest.raises(ValueError, match="Diffusion stability violation"):
            apply_diffusion(
                concentration=concentration,
                diffusivity=1e-2,
                grid_spacing=(0.1, 0.1),
                spatial_dims=2,
                time_step=0.3,
                check_stability=True,
                enforce_non_negative=True, boundary_method="zero",
            )
        
    def test_apply_diffusion_non_negative_constraint(self, device):
        """Test that negative concentration input is rejected without clipping."""
        conc = torch.full((1, 11), 0.1, device=device)
        conc[0, 5] = -0.01

        with pytest.raises(ValueError, match="non-negative"):
            apply_diffusion(
                concentration=conc,
                diffusivity=1e-5,
                grid_spacing=0.1,
                spatial_dims=1,
                time_step=0.01,
                check_stability=True,
                enforce_non_negative=True, boundary_method="zero",
            )

    def test_apply_diffusion_zero_field_has_no_hidden_mass_source(self, device):
        """Test that positivity enforcement does not inject a hidden floor."""
        concentration = torch.zeros((1, 11), device=device)

        result = apply_diffusion(
            concentration=concentration,
            diffusivity=1e-5,
            grid_spacing=0.1,
            spatial_dims=1,
            boundary_method="periodic",
            time_step=0.01,
            check_stability=True,
            enforce_non_negative=True,
        )

        assert torch.equal(result["concentration"], concentration)
        assert result["conservation_error"] == 0.0
        
    def test_apply_diffusion_2d(self, concentration_field_2d, device):
        """Test 2D diffusion operator."""
        conc = concentration_field_2d.unsqueeze(0).to(device)
        
        result = apply_diffusion(
            concentration=conc,
            diffusivity=1e-3,
            grid_spacing=(0.1, 0.1),
            spatial_dims=2,
            time_step=0.01,
            check_stability=True,
            enforce_non_negative=True, boundary_method="zero"
        )
        
        # Check that result has correct shape
        assert result['diffusion_rate'].shape == conc.shape  # Fix key name
        
        # Check mass conservation
        total_mass_before = torch.sum(conc)
        total_mass_after = torch.sum(result['concentration'])  # Use returned concentration
        relative_change = abs(total_mass_after - total_mass_before) / total_mass_before
        assert relative_change < 0.01

    def test_apply_diffusion_rate_only_rejects_numerical_overflow(self):
        concentration = torch.tensor([0.0, 3.0e38, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="diffusion_rate.*finite"):
            apply_diffusion(
                concentration,
                3.0e38,
                1.0,
                check_stability=False, spatial_dims=1, boundary_method="zero", time_step=None, enforce_non_negative=True,
            )


class TestAdvectionOperator:
    """Test the apply_advection high-level operator."""
    
    def test_apply_advection_gaussian_field(self, concentration_field_1d, 
                                          velocity_field_1d_constant, device):
        """Test advection of Gaussian field (should advect)."""
        conc = concentration_field_1d.unsqueeze(0).to(device)
        vel = velocity_field_1d_constant.unsqueeze(0).to(device)
        
        result = apply_advection(
            concentration=conc,
            velocity=vel,
            grid_spacing=0.1,
            spatial_dims=1,
            time_step=0.01,
            check_cfl=True, scheme="upwind", boundary_method="extrapolate"
        )
        
        # Should have advection rate and it should be finite
        assert 'advection_rate' in result
        assert torch.all(torch.isfinite(result['advection_rate']))
        
    def test_apply_advection_gaussian_pulse(self, concentration_field_1d,
                                          velocity_field_1d_constant, device):
        """Test advection of Gaussian pulse (should translate)."""
        conc = concentration_field_1d.unsqueeze(0).to(device)
        vel = velocity_field_1d_constant.unsqueeze(0).to(device)
        
        result = apply_advection(
            concentration=conc,
            velocity=vel,
            grid_spacing=0.1,
            spatial_dims=1,
            time_step=0.01,
            check_cfl=True, scheme="upwind", boundary_method="extrapolate"  # Fix parameter name
        )
        
        delta_conc = result['advection_rate']  # Fix key name
        new_conc = result['concentration']  # Use returned concentration
        
        # Peak should have moved in direction of velocity
        peak_idx_before = torch.argmax(conc[0]).item()
        peak_idx_after = torch.argmax(new_conc[0]).item()
        
        # With positive velocity, peak should move right
        assert peak_idx_after >= peak_idx_before
        
    def test_apply_advection_cfl_check(self, concentration_field_1d, device):
        """Test that unstable requested advection steps fail fast."""
        conc = concentration_field_1d.unsqueeze(0).to(device)
        # High velocity that should violate CFL
        vel = torch.full_like(conc, 10.0)  # u = 10 m/s

        with pytest.raises(ValueError, match="CFL stability violation"):
            apply_advection(
                concentration=conc,
                velocity=vel,
                grid_spacing=0.1,
                spatial_dims=1,
                time_step=1.0,
                check_cfl=True, scheme="upwind", boundary_method="extrapolate",
            )

    def test_apply_advection_rejects_repeated_velocity_shape(self, device):
        concentration = torch.ones((1, 6), device=device)
        velocity = torch.tensor([[1.0, 2.0]], device=device)

        with pytest.raises(ValueError, match="broadcast-compatible"):
            apply_advection(
                concentration=concentration,
                velocity=velocity,
                grid_spacing=1.0,
                spatial_dims=1,
                boundary_method="periodic",
                time_step=0.1, scheme="upwind", check_cfl=True,
            )

    @pytest.mark.parametrize(
        ("concentration", "velocity", "message"),
        [
            (torch.tensor([[1.0, float("nan")]]), torch.ones((1, 2)), "concentration"),
            (torch.ones((1, 2)), torch.tensor([[1.0, float("nan")]]), "velocity"),
        ],
    )
    def test_apply_advection_rejects_nonfinite_fields(
        self,
        concentration,
        velocity,
        message,
    ):
        with pytest.raises(ValueError, match=message):
            apply_advection(
                concentration=concentration,
                velocity=velocity,
                grid_spacing=1.0,
                spatial_dims=1,
                boundary_method="periodic",
                time_step=0.1, scheme="upwind", check_cfl=True,
            )

    def test_apply_advection_rate_only_rejects_numerical_overflow(self):
        concentration = torch.tensor([3.0e38, 1.0], dtype=torch.float32)
        velocity = torch.full_like(concentration, 3.0e38)
        with pytest.raises(ValueError, match="advection_rate.*finite"):
            apply_advection(
                concentration,
                velocity,
                1.0,
                boundary_method="periodic",
                check_cfl=False, spatial_dims=1, scheme="upwind", time_step=None,
            )

    def test_apply_advection_rejects_complex_velocity(self):
        concentration = torch.ones(4)
        velocity = torch.ones(4, dtype=torch.complex64)
        with pytest.raises(TypeError, match="real floating-point"):
            apply_advection(
                concentration,
                velocity,
                1.0,
                boundary_method="periodic",
                check_cfl=False, spatial_dims=1, scheme="upwind", time_step=None,
            )

    def test_central_advection_rejects_forward_euler_ftcs(self, device):
        concentration = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        with pytest.raises(ValueError, match="central.*implicit"):
            apply_advection(
                concentration=concentration,
                velocity=torch.ones_like(concentration),
                grid_spacing=1.0,
                spatial_dims=1,
                scheme="central",
                boundary_method="periodic",
                time_step=0.1,
                time_integration_method="euler", check_cfl=True,
            )

    def test_implicit_advection_reports_but_does_not_enforce_explicit_cfl(self, device):
        concentration = torch.ones((1, 4), dtype=torch.float64, device=device)
        result = apply_advection(
            concentration,
            torch.full_like(concentration, 2.0),
            1.0,
            scheme="central",
            boundary_method="periodic",
            time_step=1.0,
            check_cfl=True,
            time_integration_method="implicit_midpoint", spatial_dims=1,
        )
        torch.testing.assert_close(result["concentration"], concentration)
        torch.testing.assert_close(result["cfl_number"], torch.full_like(concentration, 2.0))

    @pytest.mark.parametrize("scheme", ["lax_wendroff", "superbee", "van_leer"])
    def test_fully_discrete_advection_schemes_reject_implicit_reintegration(
        self,
        device,
        scheme,
    ):
        concentration = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        with pytest.raises(ValueError, match="fully discrete.*euler"):
            apply_advection(
                concentration=concentration,
                velocity=torch.ones_like(concentration),
                grid_spacing=1.0,
                spatial_dims=1,
                scheme=scheme,
                boundary_method="periodic",
                time_step=0.1,
                time_integration_method="implicit_midpoint", check_cfl=True,
            )
        
    def test_apply_advection_2d(self, concentration_field_2d,
                               velocity_field_2d_constant, device):
        """Test 2D advection operator."""
        conc = concentration_field_2d.unsqueeze(0).to(device)
        u, v = velocity_field_2d_constant
        u = u.unsqueeze(0).to(device)
        v = v.unsqueeze(0).to(device)
        
        result = apply_advection(
            concentration=conc,
            velocity=(u, v),
            grid_spacing=(0.1, 0.1),
            spatial_dims=2,
            time_step=0.01,
            check_cfl=True, scheme="upwind", boundary_method="extrapolate"  # Fix parameter name
        )
        
        # Check that result has correct shape
        assert result['advection_rate'].shape == conc.shape  # Fix key name
        
        # Check mass conservation
        total_mass_before = torch.sum(conc)
        total_mass_after = torch.sum(result['concentration'])  # Use returned concentration
        relative_change = abs(total_mass_after - total_mass_before) / total_mass_before
        assert relative_change < 0.01


class TestTimeIntegrationMethodChoice:
    """Accepted methods are the allowed_values of the YAML ParameterSpec, not a code copy."""

    def test_resolver_rejects_a_method_the_yaml_spec_does_not_allow(self, monkeypatch):
        _serve_time_integration_allowed_values(monkeypatch, ["euler", "crank_nicolson"])

        with pytest.raises(
            ValueError,
            match=r"Unsupported time integration method 'implicit_midpoint'.*euler, crank_nicolson",
        ):
            _resolve_transport_time_integration("implicit_midpoint", 4, 1.0e-10)

    def test_resolver_accepts_every_method_the_yaml_spec_allows(self, monkeypatch):
        _serve_time_integration_allowed_values(
            monkeypatch, ["euler", "implicit_midpoint", "crank_nicolson", "rk4"]
        )

        assert _resolve_transport_time_integration("rk4", 4, 1.0e-10)[0] == "rk4"

    def test_rate_integrator_rejects_a_method_the_yaml_spec_does_not_allow(self, monkeypatch):
        _serve_time_integration_allowed_values(monkeypatch, ["euler", "crank_nicolson"])
        state = torch.ones(3, dtype=torch.float64)

        with pytest.raises(ValueError, match=r"'implicit_midpoint'.*euler, crank_nicolson"):
            _integrate_rate_system(state, 0.1, lambda y: -y, "implicit_midpoint", 4, 1.0e-10)

    @pytest.mark.parametrize("kernel", ["diffusion", "advection"])
    def test_boundary_flux_ledger_has_no_fallback_quadrature(self, monkeypatch, kernel):
        """A YAML-allowed method without a boundary-flux quadrature fails fast.

        The step ledger pairs each integrator with its quadrature (euler: rate
        at t_n; implicit_midpoint: midpoint; crank_nicolson: trapezoid); an
        accepted method outside those must not fall into the trapezoid branch.
        """
        _serve_time_integration_allowed_values(
            monkeypatch, ["euler", "implicit_midpoint", "crank_nicolson", "rk4"]
        )
        concentration = torch.tensor([[1.0, 2.0, 3.0, 2.0]], dtype=torch.float64)

        with pytest.raises(ValueError, match="boundary-flux quadrature.*'rk4'"):
            if kernel == "diffusion":
                apply_diffusion(
                    concentration,
                    torch.full_like(concentration, 0.1),
                    1.0,
                    1,
                    "zero",
                    0.1,
                    False,
                    False,
                    time_integration_method="rk4",
                    diffusivity_layout="scalar_field",
                )
            else:
                apply_advection(
                    concentration,
                    torch.full_like(concentration, 0.5),
                    1.0,
                    1,
                    "upwind",
                    "zero",
                    0.1,
                    False,
                    time_integration_method="rk4",
                )


def _periodic_step_profile() -> torch.Tensor:
    """Step profile: excites every Fourier mode, the worst case of the Picard iteration."""
    cells = torch.arange(64, dtype=torch.float64)
    return ((cells < 32).to(torch.float64) + 0.1).reshape(1, -1)


class TestImplicitFixedPointIterationLimit:
    """transport.numerical.implicit_midpoint_max_iter bounds the Picard iteration.

    For y = y_n + dt f((y_n + y) / 2) (and theta = 1/2 Crank-Nicolson) the
    fixed-point map contracts with q = (dt / 2) L; central advection has
    L = |u| / dx, so q = C / 2 <= 1/2 over the documented range
    C <= transport.numerical.cfl_advection_max = 1.
    """

    @pytest.mark.parametrize("method", ["implicit_midpoint", "crank_nicolson"])
    def test_database_limit_covers_central_advection_at_the_documented_cfl_limit(self, method):
        courant = float(
            get_global_manager().get("transport.numerical.cfl_advection_max", required=True)
        )
        concentration = _periodic_step_profile()

        result = apply_advection(
            concentration, torch.ones_like(concentration), 1.0, 1, "central", "periodic",
            courant, False, time_integration_method=method,
        )

        torch.testing.assert_close(
            torch.sum(result["concentration"]), torch.sum(concentration), rtol=0.0, atol=1.0e-9
        )

    @pytest.mark.parametrize("method", ["implicit_midpoint", "crank_nicolson"])
    def test_iteration_limit_exhaustion_raises_instead_of_returning_a_state(self, method):
        concentration = _periodic_step_profile()

        with pytest.raises(RuntimeError, match=rf"{method} did not converge after 2 iterations"):
            apply_advection(
                concentration, torch.ones_like(concentration), 1.0, 1, "central", "periodic",
                0.5, False, time_integration_method=method, implicit_midpoint_max_iter=2,
            )

    @pytest.mark.parametrize("method", ["implicit_midpoint", "crank_nicolson"])
    def test_non_contractive_step_raises_with_the_database_limit(self, method):
        """Upwind at C = 1 gives q = C = 1: the Picard map does not contract."""
        concentration = _periodic_step_profile()

        with pytest.raises(RuntimeError, match=rf"{method} did not converge"):
            apply_advection(
                concentration, torch.ones_like(concentration), 1.0, 1, "upwind", "periodic",
                1.0, False, time_integration_method=method,
            )

    @pytest.mark.parametrize("method", ["implicit_midpoint", "crank_nicolson"])
    def test_raising_the_limit_does_not_change_a_converged_step(self, method):
        """A step converging within 4 iterations returns the same state at any larger limit."""
        cells = torch.arange(64, dtype=torch.float64)
        concentration = torch.exp(-(((cells - 32.0) / 4.0) ** 2)).reshape(1, -1)

        results = [
            apply_advection(
                concentration, torch.ones_like(concentration), 1.0, 1, "central", "periodic",
                0.01, False, time_integration_method=method, implicit_midpoint_max_iter=limit,
            )["concentration"]
            for limit in (4, None)
        ]

        assert torch.equal(results[0], results[1])


class TestCombinedOperator:
    """Test the advance_advection_diffusion combined operator."""
    
    def test_combined_operator_constant_field(self, concentration_field_1d,
                                            velocity_field_1d_constant, device):
        """Test combined operator on constant field."""
        conc = concentration_field_1d.unsqueeze(0).to(device)
        vel = velocity_field_1d_constant.unsqueeze(0).to(device)
        
        result = advance_advection_diffusion(
            concentration=conc,
            velocity=vel,
            diffusivity=1e-5,
            grid_spacing=0.1,
            spatial_dims=1,
            time_step=0.01,
            splitting_method="sequential",
            check_stability=True,
            enforce_non_negative=True, advection_scheme="upwind", boundary_method="zero"
        )
        
        # Should return updated concentration
        assert 'concentration' in result
        assert result['concentration'].shape == conc.shape
        
    def test_combined_operator_splitting_methods(self, concentration_field_1d,
                                               velocity_field_1d_constant, device):
        """Test different operator splitting methods."""
        conc = concentration_field_1d.unsqueeze(0).to(device)
        vel = velocity_field_1d_constant.unsqueeze(0).to(device)
        
        methods = ["sequential", "symmetric", "combined"]
        results = {}
        
        for method in methods:
            results[method] = advance_advection_diffusion(
                concentration=conc,
                velocity=vel,
                diffusivity=1e-4,
                grid_spacing=0.1,
                spatial_dims=1,
                time_step=0.01,
                splitting_method=method,
                check_stability=True,
                enforce_non_negative=True, advection_scheme="upwind", boundary_method="zero"
            )
        
        # All methods should produce results with correct shape
        for method, result in results.items():
            assert result['concentration'].shape == conc.shape  # Fix key name
            
        # Symmetric splitting should be more accurate than sequential
        # (This is a qualitative test - in practice, the difference may be small)
        sequential_result = results["sequential"]
        symmetric_result = results["symmetric"]
        
        # Results should be similar but not identical (increase tolerance for numerical differences)
        # Note: Very similar results are expected for small time steps
        assert torch.allclose(
            sequential_result['concentration'],
            symmetric_result['concentration'],
            rtol=0.01, atol=1e-3  # Allow 1% relative difference
        )


class TestUtilityFunctions:
    """Test utility functions for transport operators."""
    
    def test_compute_grid_metrics_1d(self, device):
        """Test grid metrics computation for 1D."""
        spatial_dims = (100,)  # Fix parameter name
        grid_spacing = 0.1
        
        metrics = compute_grid_metrics(spatial_dims, grid_spacing, device)
        
        assert 'volumes' in metrics
        # For 1D, volumes are cell lengths
        expected_volume = grid_spacing
        assert torch.allclose(metrics['volumes'], torch.full_like(metrics['volumes'], expected_volume))
        
    def test_compute_grid_metrics_2d(self, device):
        """Test grid metrics computation for 2D."""
        spatial_dims = (50, 100)  # Fix parameter name
        grid_spacing = (0.1, 0.2)
        
        metrics = compute_grid_metrics(spatial_dims, grid_spacing, device)
        
        assert 'volumes' in metrics
        # For 2D, volumes are cell areas
        expected_area = grid_spacing[0] * grid_spacing[1]
        assert torch.allclose(metrics['volumes'], torch.full_like(metrics['volumes'], expected_area))

    @pytest.mark.parametrize("spacing", [0.0, -1.0, float("inf"), float("nan")])
    def test_compute_grid_metrics_rejects_invalid_spacing(self, device, spacing):
        """Grid geometry requires finite positive spacing."""
        with pytest.raises(ValueError, match="spacing"):
            compute_grid_metrics((4,), spacing, device)

    @pytest.mark.parametrize("spatial_dims", [(), (0,), (2, 0), (2, 3, 4, 5)])
    def test_compute_grid_metrics_rejects_invalid_shape(self, device, spatial_dims):
        """Grid geometry supports one to three strictly positive axes."""
        with pytest.raises(ValueError, match="spatial dimensions"):
            compute_grid_metrics(spatial_dims, 1.0, device)

    def test_compute_grid_metrics_requires_explicit_device(self):
        """Low-level geometry must not choose a hidden CPU device."""
        with pytest.raises(ValueError, match="device"):
            compute_grid_metrics((4,), 1.0, None)
        
    def test_compute_cfl_number(self, device):
        """Test CFL number computation."""
        velocity = torch.tensor([1.0, 2.0, 0.5], device=device)
        grid_spacing = 0.1
        time_step = 0.05
        
        cfl = compute_cfl_number(velocity, grid_spacing, time_step)
        
        # Expected CFL = |v| * dt / dx
        expected_cfl = torch.abs(velocity) * time_step / grid_spacing
        assert torch.allclose(cfl, expected_cfl)

    def test_compute_cfl_number_uses_storage_order_spacing(self, device):
        """Velocity (u, v) pairs with storage-order spacing (dy, dx)."""
        velocity_x = torch.ones((1, 2, 3), device=device)
        velocity_y = torch.zeros_like(velocity_x)

        cfl = compute_cfl_number(
            (velocity_x, velocity_y),
            (10.0, 1.0),
            1.0,
        )

        torch.testing.assert_close(cfl, torch.ones_like(cfl))

    def test_compute_cfl_number_rejects_axis_count_mismatch(self, device):
        velocity = (
            torch.ones((1, 2, 3), device=device),
            torch.zeros((1, 2, 3), device=device),
            torch.zeros((1, 2, 3), device=device),
        )

        with pytest.raises(ValueError, match="same number of axes"):
            compute_cfl_number(velocity, (1.0, 1.0), 0.1)
        
    def test_compute_diffusion_stability(self, device):
        """Test diffusion stability criterion."""
        diffusivity = torch.tensor(1e-5, device=device)
        grid_spacing = 0.1
        time_step = 0.05
        
        stability_number = compute_diffusion_stability(diffusivity, grid_spacing, time_step)
        
        # Stability number: sigma = D * dt / dx^2
        # Stability requires: sigma <= 0.5
        expected_stability = diffusivity * time_step / (grid_spacing ** 2)
        assert torch.allclose(stability_number, expected_stability)

    @pytest.mark.parametrize(
        ("diffusivity", "grid_spacing", "time_step", "message"),
        [
            (torch.tensor(-1.0), 1.0, 0.1, "diffusivity"),
            (torch.tensor(float("nan")), 1.0, 0.1, "finite"),
            (torch.tensor(1.0), 0.0, 0.1, "grid spacing"),
            (torch.tensor(1.0), 1.0, -0.1, "time_step"),
        ],
    )
    def test_compute_diffusion_stability_rejects_invalid_domain(
        self,
        diffusivity,
        grid_spacing,
        time_step,
        message,
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            compute_diffusion_stability(diffusivity, grid_spacing, time_step)

    def test_canonical_conservation_error_is_returned(self, device):
        concentration = torch.ones((1, 10), dtype=torch.float64, device=device)
        source = torch.full_like(concentration, 0.1)
        result = advance_advection_diffusion(
            concentration,
            torch.zeros_like(concentration),
            0.0,
            1.0,
            0.1,
            splitting_method="combined",
            boundary_method="periodic",
            source_term=source, spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        assert result["conservation_error"] < 1e-12

    def test_negative_source_is_rejected_without_clipping(self, device):
        concentration = torch.full((1, 5), 0.1, device=device)
        with pytest.raises(ValueError, match="non-negative"):
            advance_advection_diffusion(
                concentration,
                torch.zeros_like(concentration),
                0.0,
                1.0,
                1.0,
                splitting_method="combined",
                boundary_method="periodic",
                source_term=torch.full_like(concentration, -0.2), spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
            )


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_input_shapes(self, device):
        """Test handling of invalid input shapes."""
        # Mismatched concentration and velocity shapes
        conc = torch.ones((1, 10), device=device)
        vel = torch.ones((1, 15), device=device)  # Different length
        
        with pytest.raises((ValueError, RuntimeError)):
            apply_advection(
                concentration=conc,
                velocity=vel,
                grid_spacing=0.1,
                spatial_dims=1,
                time_step=0.01, scheme="upwind", boundary_method="extrapolate", check_cfl=True
            )
            
    def test_negative_parameters(self, device):
        """Test rejection of negative diffusivity."""
        conc = torch.ones((1, 10), device=device)

        with pytest.raises(ValueError, match="diffusivity"):
            apply_diffusion(
                concentration=conc,
                diffusivity=-1e-5,
                grid_spacing=0.1,
                spatial_dims=1,
                time_step=0.01, boundary_method="zero", check_stability=True, enforce_non_negative=True,
            )

    def test_zero_grid_spacing(self, device):
        """Test rejection of zero grid spacing."""
        conc = torch.ones((1, 10), device=device)

        with pytest.raises(ValueError, match="grid spacing"):
            apply_diffusion(
                concentration=conc,
                diffusivity=1e-5,
                grid_spacing=0.0,
                spatial_dims=1,
                time_step=0.01, boundary_method="zero", check_stability=True, enforce_non_negative=True,
            )

    def test_negative_advection_boundary_state_is_rejected(self, device):
        concentration = torch.ones((1, 4), device=device)
        with pytest.raises(ValueError, match="advection_boundary_state.*non-negative"):
            advance_advection_diffusion(
                concentration,
                torch.ones_like(concentration),
                0.0,
                1.0,
                0.1,
                splitting_method="combined",
                boundary_method="open",
                advection_boundary_state=-torch.ones_like(concentration), spatial_dims=1, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
            )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

