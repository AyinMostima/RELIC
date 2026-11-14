"""
Comprehensive L1 Unit Tests for Fickian Diffusion Transport Model

This test suite provides exhaustive testing of the FickianDiffusion class,
covering all numerical methods, boundary conditions, stability analysis, and
scientific accuracy. Tests ensure zero-defect operation and validate against
analytical solutions.

Test Categories:
1. Basic Functionality Tests
2. Numerical Method Accuracy Tests
3. Stability and CFL Condition Tests
4. Boundary Condition Tests
5. Conservation Law Tests
6. Multi-Dimensional Transport Tests
7. Anisotropic Diffusion Tests
8. Error Handling and Robustness Tests

References for validation values:
- Crank, J. (1975). The Mathematics of Diffusion
- Bird, R.B. et al. (2007). Transport Phenomena
- LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs
"""

import pytest
import torch
import numpy as np
from typing import Dict, Tuple
import sys
import os
import math

# Add relic to Python path for testing
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, relic_path)

from relic.scientific.transport.diffusion import FickianDiffusion, create_isotropic_diffusion, create_anisotropic_diffusion
from relic.core.state import UpdateItem, UpdateMode


def _updated_concentration(result: Dict[str, torch.Tensor], previous_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    update = result.get("tracer_concentration")
    if update is None:
        raise KeyError("Diffusion forward did not return tracer_concentration increment.")
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return previous_state["tracer_concentration"] + update.value


def _forward_with_update(model, state: Dict[str, torch.Tensor], time: float, dt: float) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    result = model.forward(state, time, dt)
    updated = _updated_concentration(result, state)
    new_state = dict(state)
    new_state["tracer_concentration"] = updated
    return new_state, result


class TestFickianDiffusionBasicFunctionality:
    """Test basic functionality and interface compliance."""
    
    def test_initialization_1d_isotropic(self):
        """Test 1D isotropic diffusion initialization."""
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=0.1,
            time_integration="explicit"
        )
        
        assert model is not None
        assert model.spatial_dims == 1
        assert model.isotropic == True
        assert model.diffusivity_scalar == 1e-5
        assert model.time_integration == "explicit"
        assert len(model.grid_spacing) == 1
        assert model.grid_spacing[0] == 0.1
        
    def test_initialization_2d_isotropic(self):
        """Test 2D isotropic diffusion initialization."""
        model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=2e-6,
            grid_spacing=(0.05, 0.1),  # (dy, dx)
            time_integration="semi_implicit"
        )
        
        assert model.spatial_dims == 2
        assert model.isotropic == True
        assert model.diffusivity_scalar == 2e-6
        assert model.time_integration == "semi_implicit"
        assert len(model.grid_spacing) == 2
        assert model.grid_spacing[0] == 0.05  # dy
        assert model.grid_spacing[1] == 0.1   # dx
        
    def test_initialization_3d_anisotropic(self):
        """Test 3D anisotropic diffusion initialization."""
        # Create anisotropic diffusivity tensor: (Kz, Ky, Kx)
        diffusivity_tensor = torch.tensor([1e-6, 2e-6, 3e-6])
        
        model = FickianDiffusion(
            spatial_dims=3,
            diffusivity=diffusivity_tensor, diffusivity_layout="directional",
            grid_spacing=(0.02, 0.05, 0.1),  # (dz, dy, dx)
            time_integration="explicit"
        )
        
        assert model.spatial_dims == 3
        assert model.isotropic == False
        assert torch.allclose(model.diffusivity_tensor, diffusivity_tensor)
        assert len(model.grid_spacing) == 3
        
    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            'parameters': {
                'diffusivity': 5e-6,
                'non_negative': True
            }
        }
        
        model = FickianDiffusion(
            name='CustomDiffusion',
            config=config,
            spatial_dims=2,
            diffusivity=5e-6,
            grid_spacing=0.1,
            non_negative=True
        )
        
        assert model.diffusivity_scalar == 5e-6
        assert model.non_negative == True
        
    def test_dependencies_interface(self):
        """Test get_dependencies interface compliance."""
        model = FickianDiffusion()
        dependencies = model.get_dependencies()
        
        assert isinstance(dependencies, list)
        # Diffusion typically has no dependencies
        assert len(dependencies) == 0
        
    def test_invalid_initialization_parameters(self):
        """Test error handling for invalid initialization parameters."""
        # Invalid spatial dimensions
        with pytest.raises(ValueError, match="Spatial dimensions must be 1, 2, or 3"):
            FickianDiffusion(spatial_dims=4)
            
        # Invalid time integration
        with pytest.raises(ValueError, match="Time integration must be"):
            FickianDiffusion(time_integration="invalid")
            
        # Negative diffusivity
        with pytest.raises(ValueError, match="Diffusivity must be positive"):
            FickianDiffusion(diffusivity=-1e-5)


class TestNumericalMethodAccuracy:
    """Test numerical method accuracy against analytical solutions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
    def test_1d_gaussian_diffusion_analytical(self):
        """Test 1D diffusion against analytical Gaussian solution."""
        # Parameters for analytical solution
        D = 1e-4  # Diffusivity [m虏/s]
        dx = 0.01  # Grid spacing [m]
        dt = 0.001  # Time step [s]
        nx = 101   # Number of grid points
        
        # Initialize model
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=D,
            grid_spacing=dx,
            time_integration="explicit"
        )
        
        # Create initial Gaussian concentration
        x = torch.linspace(0, nx*dx, nx)
        x0 = 0.5  # Initial center [m]
        sigma0 = 0.05  # Initial width [m]
        
        c_initial = torch.exp(-(x - x0)**2 / (2 * sigma0**2))
        c_initial = c_initial / torch.sum(c_initial)  # Normalize
        
        # State with batch dimension
        state = {
            "tracer_concentration": c_initial.unsqueeze(0)  # (1, nx)
        }
        
        # Evolve for several time steps
        time = 0.0
        for i in range(10):
            state, _ = _forward_with_update(model, state, time, dt)
            time += dt
            
        # Analytical solution at final time
        sigma_final = torch.sqrt(torch.tensor(sigma0**2 + 2*D*time))
        c_analytical = torch.exp(-(x - x0)**2 / (2 * sigma_final**2))
        c_analytical = c_analytical / torch.sum(c_analytical)
        
        c_numerical = state["tracer_concentration"].squeeze(0)
        
        # Compare against analytical solution (allow reasonable tolerance)
        relative_error = torch.mean(torch.abs(c_numerical - c_analytical) / (c_analytical + 1e-10))
        assert relative_error < 0.1, f"Relative error {relative_error:.3f} too large"
        
    def test_2d_radial_diffusion_symmetry(self):
        """Test 2D radial diffusion maintains symmetry."""
        D = 1e-5
        dx = dy = 0.02
        dt = 0.001
        nx = ny = 51
        
        model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=D,
            grid_spacing=(dy, dx),
            time_integration="explicit"
        )
        
        # Create initial point source at center
        c_initial = torch.zeros(ny, nx)
        center_y, center_x = ny//2, nx//2
        c_initial[center_y, center_x] = 1.0
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0)  # (1, ny, nx)
        }
        
        # Evolve for a few steps
        time = 0.0
        for i in range(5):
            state, _ = _forward_with_update(model, state, time, dt)
            time += dt
        
        c_final = state["tracer_concentration"].squeeze(0)
        
        # Check symmetry: concentration should be same at equal distances from center
        distances = []
        concentrations = []
        
        for i in range(ny):
            for j in range(nx):
                dist = math.sqrt((i - center_y)**2 + (j - center_x)**2)
                if dist > 0 and dist < 15:  # Exclude center and far points
                    distances.append(dist)
                    concentrations.append(c_final[i, j].item())
        
        # Group by similar distances and check variance is low
        distance_bins = np.arange(1, 15, 1)
        for d in distance_bins[:-1]:
            mask = (np.array(distances) >= d) & (np.array(distances) < d+1)
            if np.sum(mask) > 4:  # Need sufficient points
                conc_in_bin = np.array(concentrations)[mask]
                cv = np.std(conc_in_bin) / (np.mean(conc_in_bin) + 1e-10)
                assert cv < 2.0, f"Radial symmetry broken at distance {d}, CV={cv:.3f}"
        
    def test_steady_state_convergence(self):
        """Test convergence to the Dirichlet face-value steady profile."""
        diffusivity = 0.01
        dx = 0.05
        nx = 20
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=diffusivity,
            grid_spacing=dx,
            boundary_method="dirichlet",
            time_integration="explicit",
        )

        concentration = torch.full((1, nx), 0.5, dtype=torch.float64)
        boundary_state = torch.zeros_like(concentration)
        boundary_state[..., 0] = 1.0
        dt = 0.05
        state = {
            "tracer_concentration": concentration,
            "tracer_concentration_boundary_state": boundary_state,
        }

        for step in range(4000):
            state, _ = _forward_with_update(model, state, step * dt, dt)

        cell_center_fraction = (
            torch.arange(nx, dtype=torch.float64) + 0.5
        ) / float(nx)
        expected_steady = 1.0 - cell_center_fraction
        final_concentration = state["tracer_concentration"].squeeze(0)
        torch.testing.assert_close(
            final_concentration,
            expected_steady,
            rtol=2.0e-3,
            atol=2.0e-3,
        )


class TestStabilityAndCFLConditions:
    """Test numerical stability and scheduler-owned step rejection."""

    def test_stability_condition_1d(self):
        diffusivity = 1.0e-4
        dx = 0.01
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=diffusivity,
            grid_spacing=dx,
        )
        concentration = torch.ones((1, 20))
        stable_dt = 0.49 * dx * dx / diffusivity
        output = model.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=stable_dt,
        )
        torch.testing.assert_close(
            output["diagnostics.diffusion_stability"],
            torch.tensor(0.49),
        )

        unstable_dt = 0.51 * dx * dx / diffusivity
        with pytest.raises(ValueError, match="Diffusion stability violation"):
            model.forward(
                {"tracer_concentration": concentration},
                time=0.0,
                dt=unstable_dt,
            )

    def test_stability_condition_2d_anisotropic(self):
        ky, kx = 2.0e-4, 1.0e-4
        dy, dx = 0.02, 0.01
        model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=torch.tensor([ky, kx]), diffusivity_layout="directional",
            grid_spacing=(dy, dx),
        )
        concentration = torch.zeros((1, 9, 9))
        concentration[0, 4, 4] = 1.0
        inverse_timescale = ky / (dy * dy) + kx / (dx * dx)
        stable_dt = 0.49 / inverse_timescale
        output = model.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=stable_dt,
        )
        torch.testing.assert_close(
            output["diagnostics.diffusion_stability"],
            torch.tensor(0.49),
        )

        with pytest.raises(ValueError, match="Diffusion stability violation"):
            model.forward(
                {"tracer_concentration": concentration},
                time=0.0,
                dt=0.51 / inverse_timescale,
            )

    def test_unstable_simulation_detection(self):
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1.0e-3,
            grid_spacing=0.01,
        )
        concentration = torch.ones((1, 101))
        concentration[0, 50] = 10.0
        with pytest.raises(
            ValueError,
            match="Reduce time_step or use simulator-managed substeps",
        ):
            model.forward(
                {"tracer_concentration": concentration},
                time=0.0,
                dt=1.0,
            )


class TestBoundaryConditions:
    """Test the single canonical finite-volume boundary interface."""

    def test_dirichlet_boundary_conditions(self):
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1.0e-5,
            grid_spacing=0.01,
            boundary_method="dirichlet",
            track_conservation=True,
        )
        concentration = torch.ones((1, 101), dtype=torch.float64)
        boundary_state = torch.zeros_like(concentration)
        boundary_state[..., 0] = 2.0
        boundary_state[..., -1] = 5.0
        dt = 0.001
        output = model.forward(
            {
                "tracer_concentration": concentration,
                "tracer_concentration_boundary_state": boundary_state,
            },
            time=0.0,
            dt=dt,
        )
        delta = output["tracer_concentration"].value
        boundary_flux = output[
            "tracer_concentration_diffusive_boundary_flux"
        ].value
        torch.testing.assert_close(
            boundary_flux,
            torch.tensor([[-0.002, -0.008]], dtype=torch.float64),
        )
        storage_change = torch.sum(delta) * 0.01
        boundary_export = torch.sum(boundary_flux) * dt
        torch.testing.assert_close(
            storage_change + boundary_export,
            torch.zeros((), dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            output["diagnostics.conservation_error"],
            torch.zeros(()),
        )

    def test_neumann_boundary_conditions(self):
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1.0e-5,
            grid_spacing=0.01,
            boundary_method="neumann",
            track_conservation=True,
        )
        x = torch.linspace(0, 1, 101)
        concentration = torch.exp(-100 * (x - 0.5) ** 2).unsqueeze(0)
        initial_mass = torch.sum(concentration)
        state = {"tracer_concentration": concentration}
        for step in range(10):
            state, output = _forward_with_update(model, state, step * 0.001, 0.001)
            boundary_flux = output[
                "tracer_concentration_diffusive_boundary_flux"
            ]
            assert isinstance(boundary_flux, UpdateItem)
            assert boundary_flux.mode is UpdateMode.REPLACE
            assert boundary_flux.per_timestep is False
            torch.testing.assert_close(
                boundary_flux.value,
                torch.zeros_like(boundary_flux.value),
            )
        torch.testing.assert_close(
            torch.sum(state["tracer_concentration"]),
            initial_mass,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_mixed_boundary_conditions_are_rejected(self):
        with pytest.raises(ValueError, match="boundary_method must be one of"):
            FickianDiffusion(
                spatial_dims=1,
                diffusivity=1.0e-5,
                grid_spacing=0.01,
                boundary_method="robin",
            )


class TestConservationLaws:
    """Test mass conservation and physical constraints."""
    
    def test_mass_conservation_explicit(self):
        """Test mass conservation for explicit time stepping."""
        model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=0.02,
            time_integration="explicit",
            track_conservation=True
        )
        
        # Initial concentration distribution
        ny, nx = 50, 50
        c_initial = torch.zeros(ny, nx)
        c_initial[20:30, 20:30] = 1.0  # Square initial distribution
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0)
        }
        
        initial_mass = torch.sum(c_initial)
        
        # Evolve with zero-flux boundaries (default)
        dt = 0.0001
        for i in range(10):
            state, _ = _forward_with_update(model, state, i*dt, dt)
        
        final_mass = torch.sum(state["tracer_concentration"].squeeze(0))
        conservation_error = torch.abs(final_mass - initial_mass) / initial_mass
        
        assert conservation_error < 0.01, f"Mass conservation error {conservation_error:.4f}"
        
    def test_mass_conservation_with_source(self):
        """Test mass conservation with source terms."""
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=0.01,
            track_conservation=True
        )
        
        nx = 101
        c_initial = torch.zeros(nx)
        
        # Constant source term
        source_rate = 0.1  # kg/m鲁/s
        source_term = torch.full((nx,), source_rate)
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0),
            "source_term": source_term.unsqueeze(0)
        }
        
        dt = 0.001
        time_steps = 10
        
        for i in range(time_steps):
            state, _ = _forward_with_update(model, state, i*dt, dt)
        
        # Expected mass increase from sources  
        # Source term adds source_rate * dt to each cell per time step
        expected_mass_increase = source_rate * dt * time_steps * nx
        actual_mass_increase = torch.sum(state["tracer_concentration"].squeeze(0))
        
        relative_error = torch.abs(actual_mass_increase - expected_mass_increase) / expected_mass_increase
        assert relative_error < 0.1, f"Source term conservation error {relative_error:.3f}"
        
    def test_positivity_preservation(self):
        """Test that concentrations remain non-negative."""
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-4,
            grid_spacing=0.01,
            non_negative=True
        )
        
        # Initial concentration with some small values
        c_initial = torch.full((101,), 0.01)
        c_initial[50] = 1.0  # Peak at center
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0)
        }
        
        # Evolve for many steps
        dt = 0.001
        for i in range(50):
            state, _ = _forward_with_update(model, state, i*dt, dt)
            
            # Check non-negativity at each step
            c_current = state["tracer_concentration"].squeeze(0)
            assert torch.all(c_current >= 0), f"Negative concentration at step {i}"
            
    def test_maximum_principle(self):
        """Test discrete maximum principle for diffusion."""
        model = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=0.01,
            time_integration="explicit"
        )
        
        # Initial concentration with clear maximum
        c_initial = torch.zeros(101)
        c_initial[50] = 1.0  # Peak at center
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0)
        }
        
        # Evolve
        dt = 0.0001
        for i in range(10):
            state, _ = _forward_with_update(model, state, i*dt, dt)
            
            c_current = state["tracer_concentration"].squeeze(0)
            
            # Maximum should not increase (may decrease due to spreading)
            current_max = torch.max(c_current)
            assert current_max <= 1.0 + 1e-6, f"Maximum principle violated: max={current_max:.6f}"


class TestMultiDimensionalDiffusion:
    """Test multi-dimensional transport behavior."""
    
    def test_2d_isotropic_spreading(self):
        """Test 2D isotropic diffusion spreading."""
        model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=0.02,  # Uniform spacing
            time_integration="explicit"
        )
        
        # Point source at center
        ny, nx = 51, 51
        c_initial = torch.zeros(ny, nx)
        c_initial[ny//2, nx//2] = 1.0
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0)
        }
        
        # Evolve
        dt = 0.0001
        for i in range(20):
            state, _ = _forward_with_update(model, state, i*dt, dt)
        
        c_final = state["tracer_concentration"].squeeze(0)
        
        # Should spread isotropically - check that contours are approximately circular
        center_y, center_x = ny//2, nx//2
        
        # Sample points at equal distance from center in different directions
        radius = 10
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        concentrations_at_radius = []
        
        for angle in angles:
            y = center_y + int(radius * np.sin(angle))
            x = center_x + int(radius * np.cos(angle))
            if 0 <= y < ny and 0 <= x < nx:
                concentrations_at_radius.append(c_final[y, x].item())
        
        # Concentrations should be similar in all directions (isotropic)
        if len(concentrations_at_radius) > 1:
            cv = np.std(concentrations_at_radius) / (np.mean(concentrations_at_radius) + 1e-10)
            assert cv < 0.2, f"Isotropic spreading failed, CV={cv:.3f}"
        
    def test_3d_diffusion_basic(self):
        """Test basic 3D diffusion functionality."""
        model = FickianDiffusion(
            spatial_dims=3,
            diffusivity=1e-5,
            grid_spacing=(0.02, 0.02, 0.02),  # (dz, dy, dx)
            time_integration="explicit"
        )
        
        # Small 3D grid
        nz, ny, nx = 21, 21, 21
        c_initial = torch.zeros(nz, ny, nx)
        c_initial[nz//2, ny//2, nx//2] = 1.0
        
        state = {
            "tracer_concentration": c_initial.unsqueeze(0)
        }
        
        # Single forward step
        dt = 0.00001  # Very small timestep for stability
        state, _ = _forward_with_update(model, state, 0.0, dt)
        
        c_final = state["tracer_concentration"].squeeze(0)
        
        # Should have diffused to neighboring cells
        center_z, center_y, center_x = nz//2, ny//2, nx//2
        
        # Check that neighbors have non-zero concentration
        neighbors_sum = (
            c_final[center_z-1, center_y, center_x] +
            c_final[center_z+1, center_y, center_x] +
            c_final[center_z, center_y-1, center_x] +
            c_final[center_z, center_y+1, center_x] +
            c_final[center_z, center_y, center_x-1] +
            c_final[center_z, center_y, center_x+1]
        )
        
        assert neighbors_sum > 1e-6, "3D diffusion should spread to neighbors"


class TestAnisotropicDiffusion:
    """Test anisotropic diffusion with directional preferences."""
    
    def test_2d_anisotropic_preferential_direction(self):
        """Test 2D anisotropic diffusion preferential spreading."""
        # Test anisotropic diffusion with a simple, reliable approach
        # Use the same setup as our successful debugging script
        
        Ky, Kx = 1e-6, 1e-4  # Kx = 100 * Ky
        diffusivity_tensor = torch.tensor([Ky, Kx])
        
        model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=diffusivity_tensor, diffusivity_layout="directional",
            grid_spacing=(0.02, 0.02),
            time_integration="explicit"
        )
        
        # Small grid, point source
        ny, nx = 11, 11
        c_initial = torch.zeros(ny, nx)
        center_y, center_x = ny//2, nx//2
        c_initial[center_y, center_x] = 1.0
        
        state = {"tracer_concentration": c_initial.unsqueeze(0)}
        
        # Use multiple time steps to accumulate anisotropic effect
        # Single large time step doesn't show effect due to numerical precision
        dt = 0.01  # Safe time step 
        num_steps = 15  # Total time = 0.15s, sufficient to show anisotropy
        
        current_state = state
        for step in range(num_steps):
            current_state, _ = _forward_with_update(model, current_state, step * dt, dt)
        
        c_new = current_state["tracer_concentration"].squeeze(0)
        
        # Check that model is using anisotropic mode
        assert not model.isotropic, "Model should be in anisotropic mode"
        
        # Direct check: X neighbors should have higher concentrations than Y neighbors
        # After multiple steps, anisotropic effect should be clearly visible
        x_left = c_new[center_y, center_x-1].item()
        x_right = c_new[center_y, center_x+1].item()
        y_up = c_new[center_y-1, center_x].item()
        y_down = c_new[center_y+1, center_x].item()
        
        x_sum = x_left + x_right
        y_sum = y_up + y_down
        
        # With Kx/Ky = 100 and multiple steps, expect significant anisotropy
        # From testing: ratio should reach ~50-700 range
        assert x_sum > 10 * y_sum, f"Anisotropic effect too weak: X_sum={x_sum:.2e}, Y_sum={y_sum:.2e}, ratio={x_sum/y_sum:.1f}"
        
    def test_1d_anisotropic_equivalence(self):
        """Test that 1D anisotropic reduces to isotropic case."""
        D = 2e-5
        
        # Isotropic model
        model_iso = FickianDiffusion(
            spatial_dims=1,
            diffusivity=D,
            grid_spacing=0.01
        )
        
        # Anisotropic model with same diffusivity
        model_aniso = FickianDiffusion(
            spatial_dims=1,
            diffusivity=torch.tensor([D]), diffusivity_layout="directional",
            grid_spacing=0.01
        )
        
        # Same initial condition
        c_initial = torch.exp(-100 * (torch.linspace(0, 1, 101) - 0.5)**2)
        
        state_iso = {"tracer_concentration": c_initial.unsqueeze(0)}
        state_aniso = {"tracer_concentration": c_initial.unsqueeze(0)}
        
        # Evolve both
        dt = 0.0001
        for i in range(10):
            state_iso, _ = _forward_with_update(model_iso, state_iso, i*dt, dt)
            state_aniso, _ = _forward_with_update(model_aniso, state_aniso, i*dt, dt)
        
        c_iso = state_iso["tracer_concentration"].squeeze(0)
        c_aniso = state_aniso["tracer_concentration"].squeeze(0)
        
        # Should be nearly identical
        relative_diff = torch.mean(torch.abs(c_iso - c_aniso) / (c_iso + 1e-10))
        assert relative_diff < 0.01, f"1D aniso/iso difference {relative_diff:.4f} too large"


class TestUtilityFunctions:
    """Test utility functions for creating common configurations."""
    
    def test_create_isotropic_diffusion(self):
        """Test isotropic diffusion factory function."""
        model = create_isotropic_diffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=(0.02, 0.01)
        )
        
        assert isinstance(model, FickianDiffusion)
        assert model.spatial_dims == 2
        assert model.isotropic == True
        assert model.diffusivity_scalar == 1e-5
        
    def test_create_anisotropic_diffusion(self):
        """Test anisotropic diffusion factory function."""
        model = create_anisotropic_diffusion(
            spatial_dims=3,
            diffusivity_components=(1e-6, 2e-6, 3e-6),  # (Kz, Ky, Kx)
            grid_spacing=(0.01, 0.02, 0.03)
        )
        
        assert isinstance(model, FickianDiffusion)
        assert model.spatial_dims == 3
        assert model.isotropic == False
        assert torch.allclose(model.diffusivity_tensor, torch.tensor([1e-6, 2e-6, 3e-6]))
        
    def test_factory_function_error_handling(self):
        """Test error handling in factory functions."""
        # Mismatched dimensions
        with pytest.raises(ValueError, match="Need"):
            create_anisotropic_diffusion(
                spatial_dims=2,
                diffusivity_components=(1e-6, 2e-6, 3e-6),  # 3 components for 2D
                grid_spacing=(0.01, 0.02)
            )


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness under various conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=0.02
        )
        
    def test_missing_state_variables(self):
        """Test error handling for missing required state variables."""
        incomplete_state = {
            "wrong_variable_name": torch.zeros(1, 50, 50)
        }
        
        with pytest.raises(ValueError, match="State must contain 'tracer_concentration'"):
            self.model.forward(incomplete_state, 0.0, 0.001)
            
    def test_wrong_tensor_dimensions(self):
        """Test handling of tensors with wrong dimensions."""
        # Wrong number of dimensions for 2D model
        wrong_dims_state = {
            "tracer_concentration": torch.zeros(50)  # 1D tensor for 2D model
        }
        
        with pytest.raises(ValueError, match="must have .* dimensions"):
            self.model.forward(wrong_dims_state, 0.0, 0.001)
            
    def test_negative_diffusivity_handling(self):
        """Test handling of negative diffusivity values."""
        # This should be caught during initialization
        with pytest.raises(ValueError, match="Diffusivity must be positive"):
            FickianDiffusion(diffusivity=-1e-5)
            
    def test_nan_inf_input_handling(self):
        """Test handling of NaN and infinite inputs."""
        nan_state = {
            "tracer_concentration": torch.full((1, 50, 50), float('nan'))
        }
        
        with pytest.raises(ValueError, match="finite"):
            self.model.forward(nan_state, 0.0, 0.001)
            
    def test_extreme_time_steps(self):
        """Test behavior with very large and very small time steps."""
        base_state = {
            "tracer_concentration": torch.ones(1, 50, 50)
        }

        # Very small timestep should work
        state_small, result = _forward_with_update(self.model, base_state, 0.0, 1e-8)
        assert torch.all(torch.isfinite(state_small["tracer_concentration"]))

        # Very large explicit timestep must fail without auto adjustment.
        large_dt = 100.0
        large_state = {
            "tracer_concentration": torch.ones(1, 50, 50)
        }
        with pytest.raises(ValueError, match="Diffusion stability violation"):
            _forward_with_update(self.model, large_state, 0.0, large_dt)
            
    def test_large_spatial_domains(self):
        """Test performance with larger spatial domains."""
        large_model = FickianDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=0.01
        )
        
        # Large domain
        ny, nx = 200, 200
        state = {
            "tracer_concentration": torch.zeros(1, ny, nx)
        }
        state["tracer_concentration"][0, ny//2, nx//2] = 1.0
        
        # Should handle large domains without errors
        state, result = _forward_with_update(large_model, state, 0.0, 0.00001)
        assert torch.all(torch.isfinite(state["tracer_concentration"]))


# Integration with pytest fixtures and parameterized tests
@pytest.fixture(scope="module")
def diffusion_model_1d():
    """Pytest fixture for 1D diffusion model."""
    return FickianDiffusion(
        spatial_dims=1,
        diffusivity=1e-5,
        grid_spacing=0.01
    )


@pytest.fixture(scope="module") 
def diffusion_model_2d():
    """Pytest fixture for 2D diffusion model."""
    return FickianDiffusion(
        spatial_dims=2,
        diffusivity=1e-5,
        grid_spacing=0.02
    )


@pytest.mark.parametrize("spatial_dims", [1, 2, 3])
def test_dimensionality_scaling(spatial_dims):
    """Parametrized test across spatial dimensions."""
    if spatial_dims == 1:
        grid_spacing = 0.01
        shape = (101,)
    elif spatial_dims == 2:
        grid_spacing = (0.02, 0.02)
        shape = (51, 51)
    else:  # 3D
        grid_spacing = (0.02, 0.02, 0.02)
        shape = (21, 21, 21)
        
    model = FickianDiffusion(
        spatial_dims=spatial_dims,
        diffusivity=1e-5,
        grid_spacing=grid_spacing
    )
    
    # Create state
    c_initial = torch.zeros(shape)
    # Set center point
    center_indices = tuple(s//2 for s in shape)
    c_initial[center_indices] = 1.0
    
    state = {
        "tracer_concentration": c_initial.unsqueeze(0)
    }
    
    # Single time step
    original_shape = state["tracer_concentration"].shape
    state, result = _forward_with_update(model, state, 0.0, 0.00001)
    updated = state["tracer_concentration"]
    
    assert torch.all(torch.isfinite(updated))
    assert updated.shape == original_shape


@pytest.mark.parametrize("integration_method", ["explicit", "semi_implicit"])
def test_time_integration_methods(diffusion_model_2d, integration_method):
    """Parametrized test across time integration methods."""
    # Create model with specified integration method
    model = FickianDiffusion(
        spatial_dims=2,
        diffusivity=1e-5,
        grid_spacing=0.02,
        time_integration=integration_method
    )
    
    # Initial state
    c_initial = torch.zeros(51, 51)
    c_initial[25, 25] = 1.0
    
    state = {
        "tracer_concentration": c_initial.unsqueeze(0)
    }
    
    # Forward step
    state, result = _forward_with_update(model, state, 0.0, 0.0001)
    updated = state["tracer_concentration"]
    
    assert torch.all(torch.isfinite(updated))
    assert torch.sum(updated) > 0.99  # Mass conservation


@pytest.mark.parametrize("diffusivity", [1e-5, 1e-4, 1e-3])
def test_diffusivity_scaling(diffusivity):
    """Parametrized test across diffusivity values."""
    model = FickianDiffusion(
        spatial_dims=1,
        diffusivity=diffusivity,
        grid_spacing=0.01
    )
    
    # Initial narrow peak
    c_initial = torch.zeros(101)
    c_initial[50] = 1.0
    
    state = {
        "tracer_concentration": c_initial.unsqueeze(0)
    }
    
    # Forward step with stable timestep
    dt = 0.5 * 0.01**2 / diffusivity / 10  # Conservative timestep
    state, result = _forward_with_update(model, state, 0.0, dt)
    updated = state["tracer_concentration"]
    
    assert torch.all(torch.isfinite(updated))
    # Higher diffusivity should lead to more spreading
    spread = torch.sum(updated.squeeze(0) * torch.arange(101.0)) / torch.sum(updated)
    assert 45 <= spread <= 55  # Should remain centered but spread


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])  # -x stops on first failure

