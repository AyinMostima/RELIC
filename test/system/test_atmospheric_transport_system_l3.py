"""
L3 System Tests for Atmospheric Transport

End-to-end system tests verifying the complete atmospheric transport system
including MOST theory, transport operators, and their integration with
the RELIC framework.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.scientific.atmosphere.most import (
    friction_velocity,
    eddy_diffusivity,
    iterative_most_solution,
    psi_h,
    zeta,
)
from relic.scientific.hydrosphere.utils.psychrometrics import compute_moist_air_density

from relic.computation.operators.functional import (
    compute_gradient_1d,
    compute_divergence_2d,
    compute_laplacian_1d,
    compute_laplacian_2d,
    compute_diffusion_stability,
)

# Import atmosphere transport modules
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
    EddyDiffusionBLMFunctor,
    EddyDiffusionConfig,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager


class TestAtmosphericTransportSystem:
    """Complete system tests for atmospheric transport."""
    
    def test_complete_boundary_layer_simulation(self):
        """Test complete boundary layer transport simulation."""
        # Domain setup
        nz = 40  # Surface-layer vertical levels
        nx = 50   # Horizontal points
        z = torch.linspace(0.5, 20.0, nz)
        x = torch.linspace(0, 5000, nx)    # 5km horizontal domain
        
        # Surface parameters
        z0m = torch.tensor(0.1)   # Momentum roughness
        z0h = torch.tensor(0.01)  # Heat roughness
        d = torch.tensor(0.0)     # Displacement height
        
        # Environmental conditions
        U_geostrophic = 10.0  # Geostrophic wind [m/s]
        T_surface = 295.0     # Surface temperature [K]
        T_air_top = 290.0     # Temperature at top [K]
        
        # Initialize profiles
        U = torch.zeros(nz, nx)
        V = torch.zeros(nz, nx)
        T = torch.zeros(nz, nx)
        
        # Set initial conditions inside the MOST constant-flux surface layer.
        for i in range(nz):
            zi = z[i].item()
            U[i, :] = U_geostrophic * (zi / 20.0)
            T[i, :] = T_surface - (T_surface - T_air_top) * (zi / 1000.0)
        
        # Time stepping parameters
        # Explicit diffusion uses K * dt / dz^2 < 0.5.
        dt = 0.01  # Time step [s]
        nt = 10   # Number of time steps
        reference_index = int(torch.argmin(torch.abs(z - 10.0)).item())
        
        # Arrays to store results
        u_star_history = []
        L_history = []
        H_history = []
        
        for t in range(nt):
            # Calculate surface layer properties at each x location
            for j in range(nx):
                # Get surface values
                U10 = U[reference_index, j]
                T10 = T[reference_index, j]
                T_sfc = T10.new_tensor(T_surface)
                
                # Solve MOST equations iteratively
                if abs(T10 - T_sfc) > 0.01:
                    # Moist-air density of the 10 m air at p = 101325 Pa and
                    # e = 1000 Pa (Holton and Hakim 2012, Eqs. 2.63-2.64).
                    u_star, L, H = iterative_most_solution(
                        U10, T10, T_sfc,
                        z[reference_index], z0m, z0h, d,
                        max_iterations=10,
                        tolerance=0.01,
                        rho=compute_moist_air_density(
                            T10, T10.new_tensor(101325.0), T10.new_tensor(1000.0)
                        ),
                    )
                else:
                    # Neutral conditions
                    u_star = friction_velocity(
                        U10, z[reference_index], z0m, d,
                        torch.tensor(0.0)
                    )
                    L = torch.tensor(1e6)  # Very large L
                    H = torch.tensor(0.0)
                
                u_star_history.append(u_star.item() if torch.is_tensor(u_star) else u_star)
                L_history.append(L.item() if torch.is_tensor(L) else L)
                H_history.append(H.item() if torch.is_tensor(H) else H)
                
                # Calculate eddy diffusivities for this column
                for i in range(1, nz-1):
                    zi = z[i]
                    zeta_i = zeta(zi - d, L)
                    
                    # Eddy diffusivity for heat/scalars
                    K_h = eddy_diffusivity(u_star, zi - d, zeta_i)
                    
                    # Apply diffusion to temperature
                    dz = (z[i+1] - z[i-1]) / 2
                    local_stability = compute_diffusion_stability(
                        K_h, dz.item(), dt
                    )
                    assert local_stability < 0.5
                    d2T_dz2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / (dz**2)
                    
                    # Update temperature
                    T[i, j] += dt * K_h.item() * d2T_dz2
                    
                    # Apply mixing to momentum
                    K_m = K_h * 1.35  # Turbulent Prandtl number
                    d2U_dz2 = (U[i+1, j] - 2*U[i, j] + U[i-1, j]) / (dz**2)
                    U[i, j] += dt * K_m.item() * d2U_dz2
        
        # Verify results
        # Check that u* values are reasonable
        u_star_mean = sum(u_star_history) / len(u_star_history)
        assert 0.1 < u_star_mean < 1.0  # Typical range
        
        # Check that L values are finite
        finite_L = [L for L in L_history if abs(L) < 1e5]
        if finite_L:
            manager = get_global_manager()
            l_min = float(
                manager.get(
                    "atmosphere.most.monin_obukhov_length_clamp_min",
                    units="m",
                    required=True,
                )
            )
            l_max = float(
                manager.get(
                    "atmosphere.most.monin_obukhov_length_clamp_max",
                    units="m",
                    required=True,
                )
            )
            assert all(l_min <= length <= l_max for length in finite_L)
        
        # Check temperature profile is stable
        assert torch.all(torch.isfinite(T))
        assert torch.all(T > 250)  # Reasonable temperature range
        assert torch.all(T < 350)
        
        # Check momentum profile
        assert torch.all(torch.isfinite(U))
        assert torch.all(U >= 0)  # No negative winds
        
    def test_scalar_transport_with_most(self):
        """Test scalar transport using MOST-derived diffusivities."""
        # Setup vertical profile
        nz = 50
        z = torch.linspace(0.1, 100, nz)
        dz = (z[1] - z[0]).item()
        
        # Environmental conditions
        u_star = torch.tensor(0.3)
        L = torch.tensor(-50.0)  # Unstable
        
        # Initialize scalar concentration (e.g., CO2)
        C = 400 * torch.ones(nz)  # Background 400 ppm
        C[:5] = 420  # Enhanced near surface
        C = C.unsqueeze(0)  # Add batch dimension
        
        # Calculate eddy diffusivities
        K_c = torch.zeros(nz)
        for i in range(nz):
            zeta_i = zeta(z[i], L)
            K_c[i] = eddy_diffusivity(u_star, z[i], zeta_i) * 0.74  # Schmidt number correction
        
        K_c = K_c.unsqueeze(0)  # Add batch dimension
        
        # Time stepping
        # Select dt from the explicit one-dimensional diffusion CFL limit.
        K_max = K_c.max().item()
        dt = 0.45 * dz**2 / K_max
        nt = 100
        
        # Check stability
        stability = compute_diffusion_stability(torch.tensor(K_max), dz, dt)
        assert stability < 0.5  # Must be stable
        
        # Evolve concentration
        for t in range(nt):
            # Calculate Laplacian
            d2C_dz2 = compute_laplacian_1d(C, dz, boundary_method="zero")
            
            # Update concentration
            dC_dt = K_c * d2C_dz2
            C = C + dt * dC_dt
            
            # Apply boundary conditions
            C[0, 0] = 420  # Fixed surface concentration
            C[0, -1] = 400  # Fixed top concentration
            
            # Check conservation (approximately, due to boundary fluxes)
            total_C = torch.sum(C)
            assert torch.isfinite(total_C)
        
        # After mixing, profile should be smoother
        final_gradient = torch.abs(compute_gradient_1d(C, dz, boundary_method="zero", order=2))
        initial_C = 400 * torch.ones(1, nz)
        initial_C[0, :5] = 420
        initial_gradient = torch.abs(compute_gradient_1d(initial_C, dz, boundary_method="zero", order=2))
        
        # Gradient should be reduced
        assert torch.mean(final_gradient) < torch.mean(initial_gradient)
        
    def test_coupled_heat_momentum_transport(self):
        """Test coupled heat and momentum transport in boundary layer."""
        # Create turbulent transport instance
        turbulent = TurbulenceFunctor(device=torch.device("cpu"))
        
        # Treat the batch as independent surface grid cells at one valid
        # surface-layer reference height. A 300 m vertical profile is not a
        # valid single-surface MOST solve.
        nz = 30
        state = {
            'wind_speed': torch.linspace(1.0, 10.0, nz).unsqueeze(0),
            'air_temperature': torch.full((1, nz), 298.15),
            'reference_height': torch.full((1, nz), 10.0),
            'surface_temperature': torch.full((1, nz), 298.65),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        # Calculate fluxes
        fluxes = turbulent.forward(state, time=0.0, dt=1.0)
        
        # Verify flux structure
        assert 'momentum_flux' in fluxes
        assert 'sensible_heat_flux' in fluxes
        
        # All fluxes should be finite
        for key, value in fluxes.items():
            assert isinstance(value, UpdateItem), key
            assert value.mode == UpdateMode.REPLACE
            assert torch.all(torch.isfinite(value.value))
    
    def test_eddy_diffusion_blm_functor(self):
        """Run the enabled Eddy functor over a real 3-D spatial field."""
        eddy_blm = EddyDiffusionBLMFunctor(
            device=torch.device("cpu"),
            transport_config=EddyDiffusionConfig(
                enabled=True,
            ),
        )
        
        # Setup state
        nz = 40
        nx = 20
        shape = (1, nz, nx)
        state = {
            'no_mixing_ratio': torch.full(shape, 10.0e-9),
            'no2_mixing_ratio': torch.full(shape, 20.0e-9),
            'o3_mixing_ratio': torch.full(shape, 50.0e-9),
            'isoprene_mixing_ratio': torch.full(shape, 2.0e-9),
            'pbl_height': torch.full(shape, 500.0),
            'friction_velocity': torch.full(shape, 0.3),
            'monin_obukhov_length': torch.full(shape, -200.0),
        }
        
        # Forward pass
        outputs = eddy_blm.forward(state, time=0.0, dt=1.0)
        assert set(outputs) == set(eddy_blm.metadata.output_variables)
        diffusivity = outputs['diagnostics.eddy_diffusion.eddy_diffusivity']
        for species in ('no', 'no2', 'o3', 'isoprene'):
            state_key = f'{species}_mixing_ratio'
            item = outputs[state_key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == 'mol mol-1'
            assert item.value.shape == shape
            background = eddy_blm._transport_parameters[
                f'background_{species}_mixing_ratio'
            ]
            expected = diffusivity / state['pbl_height']**2 * (
                state[state_key].new_tensor(background) - state[state_key]
            )
            assert torch.allclose(item.value, expected, rtol=1e-6, atol=1e-15)
            assert torch.isfinite(item.value).all()
            assert torch.all(state[state_key] + item.value >= 0.0)
        
    def test_conservation_in_transport(self):
        """Test conservation properties in transport calculations."""
        # Setup domain
        nx = 40
        ny = 40
        dx = 100.0  # 100m spacing
        dy = 100.0
        
        # Create concentration field
        X, Y = torch.meshgrid(
            torch.linspace(0, nx*dx, nx),
            torch.linspace(0, ny*dy, ny),
            indexing='ij'
        )
        
        # Gaussian blob
        C = torch.exp(-((X - nx*dx/2)**2 + (Y - ny*dy/2)**2) / (1000**2))
        C = C.unsqueeze(0)  # Add batch dimension
        
        # Wind field (divergence-free)
        U = -torch.sin(2 * torch.pi * Y / (ny * dy)).unsqueeze(0)
        V = torch.sin(2 * torch.pi * X / (nx * dx)).unsqueeze(0)
        
        # Check divergence is near zero
        div = compute_divergence_2d(U, V, dx, dy, boundary_method="zero")
        assert torch.mean(torch.abs(div)) < 0.01
        
        # Calculate advective fluxes
        Fx = U * C
        Fy = V * C
        
        # Calculate divergence of flux
        div_flux = compute_divergence_2d(Fx, Fy, dx, dy, boundary_method="zero")
        
        # Total change should be near zero (conservation)
        total_change = torch.sum(div_flux)
        total_mass = torch.sum(C)
        
        # Relative change should be small
        assert abs(total_change / total_mass) < 0.01
        
        # Test with diffusion
        K = 100.0  # Diffusivity [m^2/s]
        laplacian = compute_laplacian_2d(C, dx, dy, boundary_method="periodic")
        diffusion = K * laplacian
        
        # Check stability
        dt = 0.1
        stability = compute_diffusion_stability(torch.tensor(K), min(dx, dy), dt)
        assert stability < 0.5
        
        # Total diffusion should sum to approximately zero (no sources/sinks)
        total_diffusion = torch.sum(diffusion) * dx * dy
        assert abs(total_diffusion) < 0.001 * total_mass


class TestSystemPerformance:
    """Test system performance and scaling."""
    
    def test_large_domain_performance(self):
        """Test performance with large atmospheric domain."""
        # Large domain
        nz = 200
        nx = 100
        ny = 100
        
        z = torch.linspace(0.1, 2000, nz)
        x = torch.linspace(0, 10000, nx)
        y = torch.linspace(0, 10000, ny)
        
        # Create 3D fields
        U = torch.randn(1, nz, ny, nx) * 10 + 5
        V = torch.randn(1, nz, ny, nx) * 5
        W = torch.randn(1, nz, ny, nx) * 0.1
        T = 290 + torch.randn(1, nz, ny, nx) * 2
        
        # Test that operations complete in reasonable time
        import time
        
        # Gradient calculations
        start = time.time()
        dU_dx = compute_gradient_1d(U, 100.0, boundary_method="zero", order=2)
        grad_time = time.time() - start
        assert grad_time < 1.0  # Should complete in under 1 second
        
        # MOST calculations on surface layer
        start = time.time()
        for i in range(10):  # Sample locations
            u_star = friction_velocity(
                torch.tensor(5.0),
                torch.tensor(10.0),
                torch.tensor(0.1), torch.tensor(0.0),
                torch.tensor(0.0)
            )
        most_time = time.time() - start
        assert most_time < 0.1  # Should be very fast
        
    def test_numerical_stability(self):
        """Test numerical stability under extreme conditions."""
        # Test with very small/large values
        small_u_star = torch.tensor(1e-6)
        large_u_star = torch.tensor(100.0)
        
        z = torch.tensor(10.0)
        
        # Should handle small u*
        K_small = eddy_diffusivity(small_u_star, z, torch.tensor(0.0))
        assert torch.isfinite(K_small)
        assert K_small >= 0
        
        # Should handle large u*
        K_large = eddy_diffusivity(large_u_star, z, torch.tensor(0.0))
        assert torch.isfinite(K_large)
        assert K_large >= 0
        
        # The implemented Businger-Dyer closure has an explicit configured
        # application domain. More extreme states require another closure.
        very_stable = torch.tensor(10.0)  # Very stable
        very_unstable = torch.tensor(-10.0)  # Very unstable

        with pytest.raises(ValueError, match="Businger-Dyer validity domain"):
            psi_h(very_stable)
        with pytest.raises(ValueError, match="Businger-Dyer validity domain"):
            psi_h(very_unstable)

        assert torch.isfinite(psi_h(torch.tensor(2.0)))
        assert torch.isfinite(psi_h(torch.tensor(-2.0)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

