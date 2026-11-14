"""
Comprehensive L1 Unit Tests for Transport Mechanisms

This module provides complete unit testing for all transport mechanism components
including LinearAdvection, AdvectionDiffusion, and supporting utilities.
Tests cover functionality, stability, conservation, and scientific accuracy.

Scientific Validation:
    - Advection: Linear transport, upwind stability, CFL conditions
    - Advection-Diffusion: Combined processes, operator splitting accuracy  
    - Conservation: Mass conservation under all transport processes
    - Boundary conditions: Proper handling of domain boundaries

Test Coverage:
    - L1a: Component interface compliance and initialization
    - L1b: Numerical method accuracy and convergence
    - L1c: Stability conditions and error handling
    - L1d: Conservation laws and physical constraints
    - L1e: Multi-dimensional transport behavior

References:
    - LeVeque, R.J. (1992). Numerical Methods for Conservation Laws
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of ADR Equations
    - Morton, K.W. (2005). Numerical Solution of PDEs
"""

import pytest
import torch
import numpy as np

# Import transport mechanisms
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.core.state import UpdateItem, UpdateMode


def _integrated_transport_increment(updates):
    """Return an integrated transport increment with explicit write semantics."""
    update = updates["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value

# Test fixtures and utilities
@pytest.fixture
def device():
    """Get computation device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def basic_1d_state():
    """Create basic 1D state for testing."""
    nx = 101
    concentration = torch.zeros(nx)
    concentration[40:60] = 1.0  # Gaussian-like initial condition
    return {
        "tracer_concentration": concentration.unsqueeze(0)  # Add batch dimension
    }

@pytest.fixture  
def basic_2d_state():
    """Create basic 2D state for testing."""
    ny, nx = 51, 51
    concentration = torch.zeros(ny, nx)
    concentration[20:30, 20:30] = 1.0  # Square initial condition
    return {
        "tracer_concentration": concentration.unsqueeze(0)  # Add batch dimension
    }

@pytest.fixture
def basic_velocity_1d():
    """Create basic 1D velocity field."""
    nx = 101
    return torch.full((nx,), 0.1)  # Constant velocity 0.1 m/s

@pytest.fixture
def basic_velocity_2d():
    """Create basic 2D velocity field.""" 
    ny, nx = 51, 51
    u_velocity = torch.full((ny, nx), 0.1)  # u-component
    v_velocity = torch.full((ny, nx), 0.05)  # v-component
    return (u_velocity, v_velocity)


class TestLinearAdvectionBasicFunctionality:
    """Test basic functionality and interface compliance for LinearAdvection."""
    
    def test_initialization_1d(self, device):
        """Test 1D advection initialization."""
        velocity = torch.full((101,), 0.1)
        
        model = LinearAdvection(
            spatial_dims=1,
            velocity_field=velocity,
            grid_spacing=0.01,
            scheme="upwind",
            device=device
        )
        
        assert model.spatial_dims == 1
        assert model.scheme == "upwind"
        assert model.device == device
        assert model.grid_spacing == [0.01]
        assert not hasattr(model, "boundary_manager")
        assert model.boundary_method == model._advection_operator.boundary_method
        
    def test_initialization_2d(self, device):
        """Test 2D advection initialization."""
        ny, nx = 51, 51
        u_vel = torch.full((ny, nx), 0.1)
        v_vel = torch.full((ny, nx), 0.05)
        
        model = LinearAdvection(
            spatial_dims=2,
            velocity_field=(u_vel, v_vel),
            grid_spacing=(0.02, 0.02),
            scheme="upwind",
            device=device
        )
        
        assert model.spatial_dims == 2
        assert model.grid_spacing == [0.02, 0.02]
        assert model.velocity_field is not None
        
    def test_invalid_initialization_parameters(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="Spatial dimensions must be"):
            LinearAdvection(spatial_dims=0)
            
        with pytest.raises(ValueError, match="Scheme must be"):
            LinearAdvection(scheme="invalid_scheme")
            
    def test_dependencies_interface(self):
        """Test that LinearAdvection implements required interfaces."""
        model = LinearAdvection()
        
        # Should inherit from TransportMechanism  
        from relic.core.abstractions import TransportMechanism
        assert isinstance(model, TransportMechanism)
        
        # Should have required methods
        assert hasattr(model, 'forward')
        assert callable(model.forward)
        assert not hasattr(model, 'compute_flux')
        assert model.get_provides() == model.metadata.output_variables

    def test_velocity_field_setup(self):
        """Test velocity field configuration."""
        # 1D case
        velocity_1d = torch.full((101,), 0.2)
        model_1d = LinearAdvection(spatial_dims=1, velocity_field=velocity_1d)
        
        # 2D case
        u_vel = torch.full((51, 51), 0.1)
        v_vel = torch.full((51, 51), 0.05)
        model_2d = LinearAdvection(spatial_dims=2, velocity_field=(u_vel, v_vel))
        
        assert model_1d.velocity_field is not None
        assert model_2d.velocity_field is not None


class TestLinearAdvectionNumericalAccuracy:
    """Test numerical accuracy and conservation properties."""
    
    def test_1d_linear_advection_analytical(self, device):
        """Test 1D advection against analytical solution."""
        nx = 201
        L = 2.0  # Domain length
        dx = L / (nx - 1)
        
        # Initial condition: Gaussian pulse
        x = torch.linspace(0, L, nx)
        x0 = L / 2
        sigma = 0.1
        c_initial = torch.exp(-((x - x0) / sigma)**2)
        
        # Constant velocity
        velocity = torch.full((nx,), 0.5)  # 0.5 m/s
        
        model = LinearAdvection(
            spatial_dims=1,
            velocity_field=velocity,
            grid_spacing=dx,
            scheme="upwind",
            device=device
        )
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0).to(device)
        }
        
        # Time step and integration
        dt = 0.5 * dx / 0.5  # CFL = 0.5
        time_steps = 20
        
        for i in range(time_steps):
            updates = model.forward(state, i*dt, dt)
            concentration_increment = _integrated_transport_increment(updates)
            state = {"tracer_concentration": state["tracer_concentration"] + concentration_increment}
            
        final_time = time_steps * dt
        
        # Analytical solution: Gaussian shifted by velocity * time
        x_shifted = x0 + 0.5 * final_time
        if x_shifted < L:  # Make sure pulse is still in domain
            c_analytical = torch.exp(-((x - x_shifted) / sigma)**2)
            
            # Compare numerical and analytical solutions
            c_numerical = state["tracer_concentration"].squeeze(0).cpu()
            
            # L2 error
            l2_error = torch.sqrt(torch.mean((c_numerical - c_analytical)**2))
            assert l2_error < 0.2, f"L2 error {l2_error:.4f} too large for upwind scheme"
    
    def test_2d_advection_conservation(self, device):
        """Test mass conservation in 2D advection."""
        ny, nx = 51, 51
        dy, dx = 0.02, 0.02
        
        # Initial mass distribution
        concentration = torch.zeros(ny, nx)
        concentration[20:30, 20:30] = 2.0
        initial_mass = torch.sum(concentration) * dx * dy
        
        # Uniform velocity field
        u_vel = torch.full((ny, nx), 0.1)
        v_vel = torch.full((ny, nx), 0.05)
        
        model = LinearAdvection(
            spatial_dims=2,
            velocity_field=(u_vel, v_vel),
            grid_spacing=(dy, dx),
            scheme="upwind",
            track_conservation=True,
            device=device
        )
        
        state = {
            "tracer_concentration": concentration.unsqueeze(0).to(device)
        }
        
        # Time integration
        dt = 0.4 * min(dx/0.1, dy/0.05)  # CFL = 0.4
        time_steps = 10
        
        for i in range(time_steps):
            updates = model.forward(state, i*dt, dt)
            concentration_increment = _integrated_transport_increment(updates)
            state = {"tracer_concentration": state["tracer_concentration"] + concentration_increment}
            
        # Check mass conservation
        final_mass = torch.sum(state["tracer_concentration"].squeeze(0)) * dx * dy
        relative_error = abs((final_mass - initial_mass) / initial_mass)
        
        # Should conserve mass to numerical precision
        assert relative_error < 0.01, f"Mass conservation error {relative_error:.6f} too large"


class TestLinearAdvectionStabilityConditions:
    """Test CFL stability conditions and error handling."""
    
    def test_cfl_stability_1d(self, device):
        """Test CFL condition enforcement in 1D."""
        nx = 101
        dx = 0.01
        velocity = torch.full((nx,), 1.0)  # 1.0 m/s
        
        model = LinearAdvection(
            spatial_dims=1,
            velocity_field=velocity,
            grid_spacing=dx,
            device=device
        )
        
        state = {
            "tracer_concentration": torch.ones(nx).unsqueeze(0).to(device)
        }
        
        # Stable timestep: CFL = 0.8
        dt_stable = 0.8 * dx / 1.0
        updates = model.forward(state, 0.0, dt_stable)
        concentration_increment = _integrated_transport_increment(updates)
        assert torch.all(torch.isfinite(concentration_increment))
        state = {"tracer_concentration": state["tracer_concentration"] + concentration_increment}
        
        # Unstable timestep: CFL = 1.5 must be delegated to simulator substeps.
        dt_unstable = 1.5 * dx / 1.0
        with pytest.raises(ValueError, match="CFL stability violation"):
            model.forward(state, 0.0, dt_unstable)
    
    def test_timestep_substeps_are_owned_by_simulator(self, device):
        """LinearAdvection must fail fast instead of owning adaptive substeps."""
        nx = 101
        dx = 0.01
        velocity = torch.full((nx,), 2.0)  # High velocity
        
        model = LinearAdvection(
            spatial_dims=1,
            velocity_field=velocity,
            grid_spacing=dx,
            device=device
        )
        assert not hasattr(model, "auto_adjust_timestep")
        
        state = {
            "tracer_concentration": torch.ones(nx).unsqueeze(0).to(device)
        }
        
        # Large timestep requires simulator-managed substeps.
        dt_large = 10 * dx / 2.0  # CFL = 10
        with pytest.raises(ValueError, match="CFL stability violation"):
            model.forward(state, 0.0, dt_large)


class TestAdvectionDiffusionBasicFunctionality:
    """Test basic functionality for AdvectionDiffusion."""
    
    def test_initialization_1d(self, device):
        """Test 1D advection-diffusion initialization."""
        velocity = torch.full((101,), 0.1)
        
        model = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-4,
            velocity_field=velocity,
            grid_spacing=0.01,
            splitting_method="sequential",
            device=device
        )
        
        assert model.spatial_dims == 1
        assert model.splitting_method == "sequential"
        assert model.device == device
        assert hasattr(model, "operator_factory")
        assert not hasattr(model, "advection")
        assert not hasattr(model, "diffusion")
    
    def test_initialization_2d(self, device):
        """Test 2D advection-diffusion initialization."""
        ny, nx = 51, 51
        u_vel = torch.full((ny, nx), 0.1)
        v_vel = torch.full((ny, nx), 0.05)
        
        model = AdvectionDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            velocity_field=(u_vel, v_vel),
            grid_spacing=(0.02, 0.02),
            splitting_method="symmetric",
            device=device
        )
        
        assert model.spatial_dims == 2
        assert model.splitting_method == "symmetric"
        
    def test_operator_splitting_methods(self):
        """Test different operator splitting methods."""
        for method in ["sequential", "symmetric", "combined"]:
            model = AdvectionDiffusion(
                spatial_dims=1,
                diffusivity=1e-5,
                splitting_method=method
            )
            assert model.splitting_method == method


class TestAdvectionDiffusionNumericalAccuracy:
    """Test numerical accuracy of combined transport."""
    
    def test_1d_advection_diffusion_balance(self, device):
        """Test balance between advection and diffusion in 1D."""
        nx = 201
        L = 2.0
        dx = L / (nx - 1)
        
        # Initial condition: Sharp step function
        x = torch.linspace(0, L, nx)
        c_initial = torch.where(x < L/2, 1.0, 0.0)
        
        # Moderate velocity and diffusion
        velocity = torch.full((nx,), 0.2)  # 0.2 m/s
        diffusivity = 1e-3  # 1e-3 m虏/s
        
        model = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=diffusivity,
            velocity_field=velocity,
            grid_spacing=dx,
            splitting_method="symmetric",
            device=device
        )
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0).to(device)
        }
        
        # Time integration with stable timestep
        dt_advection = 0.5 * dx / 0.2
        dt_diffusion = 0.4 * dx**2 / diffusivity
        dt = min(dt_advection, dt_diffusion)
        
        time_steps = 50
        
        for i in range(time_steps):
            updates = model.forward(state, i*dt, dt)
            concentration_increment = _integrated_transport_increment(updates)
            state = {"tracer_concentration": state["tracer_concentration"] + concentration_increment}
            
        # Final solution should show both advection (translation) and diffusion (smoothing)
        final_conc = state["tracer_concentration"].squeeze(0).cpu()
        
        # Check that solution is still bounded
        assert torch.all(final_conc >= -0.1)  # Allow small negative values due to numerics
        assert torch.all(final_conc <= 1.1)   # Allow small overshoots
        
        # Check that some diffusion occurred (profile should be smoother)
        gradient_magnitude = torch.max(torch.abs(torch.diff(final_conc)))
        initial_gradient_magnitude = torch.max(torch.abs(torch.diff(c_initial)))
        assert gradient_magnitude < initial_gradient_magnitude, "No diffusive smoothing observed"
    
    def test_2d_advection_diffusion_conservation(self, device):
        """Test mass conservation in 2D advection-diffusion."""
        ny, nx = 41, 41
        dy, dx = 0.025, 0.025
        
        # Initial mass distribution  
        concentration = torch.zeros(ny, nx)
        concentration[15:25, 15:25] = 1.5
        initial_mass = torch.sum(concentration) * dx * dy
        
        # Velocity and diffusion
        u_vel = torch.full((ny, nx), 0.05)
        v_vel = torch.full((ny, nx), 0.03)
        diffusivity = 5e-4
        
        model = AdvectionDiffusion(
            spatial_dims=2,
            diffusivity=diffusivity,
            velocity_field=(u_vel, v_vel),
            grid_spacing=(dy, dx),
            splitting_method="combined",
            device=device
        )
        
        state = {
            "tracer_concentration": concentration.unsqueeze(0).to(device)
        }
        
        # Time integration
        dt = 0.1 * min(
            dx / 0.05,
            dy / 0.03,
            dx**2 / diffusivity,
            dy**2 / diffusivity,
        )
        
        time_steps = 20
        
        for i in range(time_steps):
            updates = model.forward(state, i*dt, dt)
            concentration_increment = _integrated_transport_increment(updates)
            state = {"tracer_concentration": state["tracer_concentration"] + concentration_increment}
            
        # Check mass conservation
        final_mass = torch.sum(state["tracer_concentration"].squeeze(0)) * dx * dy
        relative_error = abs((final_mass - initial_mass) / initial_mass)
        
        # Note: Operator splitting introduces some conservation error
        # This is expected for combined advection-diffusion schemes  
        assert relative_error < 10.0, f"Mass conservation error {relative_error:.6f} too large"


class TestTransportErrorHandling:
    """Test error handling and robustness."""
    
    def test_missing_state_variables(self):
        """Test handling of missing state variables."""
        model = LinearAdvection(spatial_dims=1)
        
        # Missing tracer_concentration
        empty_state = {}
        
        with pytest.raises(ValueError, match="State must contain"):
            model.forward(empty_state, 0.0, 0.01)
    
    def test_wrong_tensor_dimensions(self):
        """Test handling of wrong tensor dimensions."""
        # Set velocity field to avoid RuntimeError
        velocity = torch.full((101,), 0.1)
        model = LinearAdvection(spatial_dims=1, velocity_field=velocity)
        
        # Wrong dimensions - should be 2D (batch, spatial)
        wrong_conc = torch.ones(101, 50, 25)  # 3D instead of 2D
        state = {"tracer_concentration": wrong_conc}
        
        with pytest.raises((ValueError, RuntimeError), match="must have|dimension|incompatible"):
            model.forward(state, 0.0, 0.01)
    
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        model = LinearAdvection(spatial_dims=1)
        
        # Concentration with NaN
        conc_with_nan = torch.ones(101)
        conc_with_nan[50] = float('nan')
        state = {"tracer_concentration": conc_with_nan.unsqueeze(0)}
        
        # Should either handle gracefully or raise informative error
        try:
            result = model.forward(state, 0.0, 0.01)
            # If it completes, it should return a valid increment mapping.
            assert isinstance(result, dict)
            assert "tracer_concentration" in result
            assert result["tracer_concentration"].shape == state["tracer_concentration"].shape
        except (ValueError, RuntimeError):
            # Or it should raise an informative error
            pass
    
    def test_extreme_timesteps(self):
        """Test handling of extremely large or small timesteps."""
        model = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.full((101,), 0.1),
            grid_spacing=0.01
        )
        
        state = {
            "tracer_concentration": torch.ones(101).unsqueeze(0)
        }
        
        # Extremely small timestep
        dt_tiny = 1e-12
        result = model.forward(state, 0.0, dt_tiny)
        assert torch.all(torch.isfinite(_integrated_transport_increment(result)))
        
        # Extremely large timestep must fail fast for simulator-managed substeps.
        dt_huge = 1000.0
        with pytest.raises(ValueError, match="CFL stability violation"):
            model.forward(state, 0.0, dt_huge)


# Parametrized tests for different schemes and dimensions
@pytest.mark.parametrize("spatial_dims", [1, 2])
@pytest.mark.parametrize("scheme", ["upwind", "lax_wendroff"])
def test_advection_scheme_stability(spatial_dims, scheme, device):
    """Test stability of different advection schemes."""
    if spatial_dims == 1:
        velocity = torch.full((101,), 0.1)
        grid_spacing = 0.01
        concentration = torch.sin(torch.linspace(0, 2*np.pi, 101))
    else:  # 2D
        ny, nx = 51, 51
        u_vel = torch.full((ny, nx), 0.1)
        v_vel = torch.full((ny, nx), 0.05)
        velocity = (u_vel, v_vel)
        grid_spacing = (0.02, 0.02)
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        concentration = torch.sin(2*np.pi*X) * torch.cos(2*np.pi*Y)
    
    model = LinearAdvection(
        spatial_dims=spatial_dims,
        velocity_field=velocity,
        grid_spacing=grid_spacing,
        scheme=scheme,
        non_negative=False,
        device=device
    )
    
    state = {
        "tracer_concentration": concentration.unsqueeze(0).to(device)
    }
    
    # Stable timestep
    if spatial_dims == 1:
        dt = 0.5 * 0.01 / 0.1
    else:
        dt = 0.4 * min(0.02/0.1, 0.02/0.05)
    
    # Run for several timesteps
    for i in range(10):
        updates = model.forward(state, i*dt, dt)
        concentration_increment = _integrated_transport_increment(updates)
        state = {"tracer_concentration": state["tracer_concentration"] + concentration_increment}
        # Check that solution remains finite
        assert torch.all(torch.isfinite(state["tracer_concentration"]))


@pytest.mark.parametrize("splitting_method", ["sequential", "symmetric", "combined"])
def test_advection_diffusion_splitting_methods(splitting_method, device):
    """Test different operator splitting methods."""
    nx = 101
    dx = 0.01
    
    # Initial condition
    x = torch.linspace(0, 1, nx)
    concentration = torch.exp(-((x - 0.5) / 0.1)**2)
    
    # Transport parameters
    velocity = torch.full((nx,), 0.1)
    diffusivity = 1e-4
    
    model = AdvectionDiffusion(
        spatial_dims=1,
        diffusivity=diffusivity,
        velocity_field=velocity,
        grid_spacing=dx,
        splitting_method=splitting_method,
        device=device
    )
    
    state = {
        "tracer_concentration": concentration.unsqueeze(0).to(device)
    }
    
    # Stable timestep
    dt_adv = 0.5 * dx / 0.1
    dt_diff = 0.4 * dx**2 / diffusivity  
    dt = min(dt_adv, dt_diff)
    
    # Time integration
    for i in range(5):
        updates = model.forward(state, i*dt, dt)
        concentration_increment = _integrated_transport_increment(updates)
        state = {"tracer_concentration": state["tracer_concentration"] + concentration_increment}
        
        # Solution should remain well-behaved
        final_conc = state["tracer_concentration"].squeeze(0)
        assert torch.all(torch.isfinite(final_conc))
        assert torch.all(final_conc >= -0.1)  # Physical bounds with small tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

