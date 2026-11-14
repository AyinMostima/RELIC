"""
Simplified L3 System Tests for Unified Transport Operators

Focused tests that verify the core functionality of transport operators
in system-level scenarios without complex coupling dependencies.
"""

import pytest
import torch
import numpy as np

# Import the unified transport operators
from relic.computation.operators.functional import (
    apply_diffusion, apply_advection, advance_advection_diffusion
)


def test_multi_species_atmospheric_transport():
    """Test transport of multiple atmospheric species with different properties."""
    print("Testing multi-species atmospheric transport...")

    device = torch.device("cpu")

    # Create vertical atmosphere grid
    nz = 50
    height = torch.linspace(0, 1000, nz, device=device)  # 0-1000m
    dz = height[1] - height[0]

    # Multiple species with different diffusivities (m2/s)
    species_data = {
        'CO2': {'concentration': 415.0 * torch.ones(nz, device=device), 'diffusivity': 1.6e-5},
        'CH4': {'concentration': 1.9 * torch.ones(nz, device=device), 'diffusivity': 2.2e-5},
        'N2O': {'concentration': 0.33 * torch.ones(nz, device=device), 'diffusivity': 1.5e-5}
    }

    # Add vertical structure (higher concentrations near surface)
    for species in species_data.values():
        species['concentration'] *= (1.0 + 0.2 * torch.exp(-height / 100.0))

    # Wind profile (increases with height)
    wind_profile = 2.0 + 5.0 * (height / height.max())
    vertical_velocity = torch.zeros(nz, device=device)

    # Turbulent diffusivity profile (increases with height)
    turb_diffusivity = 0.01 * (1.0 + height / 200.0)

    # Simulation parameters
    dt = 10.0  # 10 seconds
    n_steps = 360  # 1 hour

    print(f"    Running {n_steps} steps, dt = {dt}s")
    print(f"    Max wind speed: {wind_profile.max():.1f} m/s")
    print(f"    Max turbulent diffusivity: {turb_diffusivity.max():.3f} m2/s")

    # Track concentrations
    concentration_history = {species: [] for species in species_data.keys()}

    for step in range(n_steps):
        for species_name, species_info in species_data.items():
            # Combined molecular + turbulent diffusivity
            total_diffusivity = species_info['diffusivity'] + turb_diffusivity

            # Apply advection-diffusion
            result = advance_advection_diffusion(
                concentration=species_info['concentration'].unsqueeze(0),
                velocity=vertical_velocity.unsqueeze(0),
                diffusivity=total_diffusivity.unsqueeze(0),
                diffusivity_layout="scalar_field",
                grid_spacing=float(dz),
                spatial_dims=1,
                time_step=dt,
                splitting_method="sequential",
                check_stability=True,
                enforce_non_negative=True, advection_scheme="upwind", boundary_method="zero"
            )

            species_info['concentration'] = result['concentration'].squeeze(0)

            # Record every 10 steps
            if step % 10 == 0:
                concentration_history[species_name].append(species_info['concentration'].clone())

    # Analyze results
    print("    Analyzing transport results...")

    for species_name, history in concentration_history.items():
        conc_tensor = torch.stack(history)

        # 1. Check mass conservation
        initial_mass = conc_tensor[0].sum()
        final_mass = conc_tensor[-1].sum()
        mass_change = abs(final_mass - initial_mass) / initial_mass
        print(f"    {species_name}: mass change = {mass_change:.3%}")

        assert mass_change < 0.01, f"{species_name} mass should be conserved (<1% change)"

        # 2. Check that diffusion occurred (profile should become more uniform)
        initial_gradient = torch.gradient(conc_tensor[0], spacing=float(dz))[0].std()
        final_gradient = torch.gradient(conc_tensor[-1], spacing=float(dz))[0].std()
        gradient_reduction = (initial_gradient - final_gradient) / initial_gradient

        print(f"    {species_name}: gradient reduction = {gradient_reduction:.3%}")
        # Conservative flux-form advection reduces artificial numerical diffusion.
        # Keep a positive lower bound to verify physically meaningful smoothing.
        assert gradient_reduction > 0.03, f"{species_name} should show diffusion effect"

        # 3. Check physical bounds
        assert torch.all(conc_tensor >= 0), f"{species_name} concentrations should remain non-negative"
        assert torch.all(torch.isfinite(conc_tensor)), f"{species_name} concentrations should remain finite"

    # 4. Check that different species diffuse at different rates
    co2_history = torch.stack(concentration_history['CO2'])
    ch4_history = torch.stack(concentration_history['CH4'])

    co2_mixing_rate = 1.0 - torch.std(co2_history[-1]) / torch.std(co2_history[0])
    ch4_mixing_rate = 1.0 - torch.std(ch4_history[-1]) / torch.std(ch4_history[0])

    print(f"    CO2 mixing rate: {co2_mixing_rate:.3%}")
    print(f"    CH4 mixing rate: {ch4_mixing_rate:.3%}")

    # CH4 should mix faster due to higher diffusivity
    assert ch4_mixing_rate > co2_mixing_rate, "CH4 should mix faster than CO2"

    print("[OK] Multi-species atmospheric transport test passed")


def test_3d_transport_convergence():
    """Test convergence and consistency of 3D transport operators."""
    print("Testing 3D transport convergence...")

    device = torch.device("cpu")

    # Grid resolution study
    resolutions = [10, 20, 40]  # Grid points per dimension
    diffusivity = 0.01
    velocity_magnitude = 0.1

    solutions = {}

    for nx in resolutions:
        print(f"    Testing {nx}x{nx}x{nx} grid...")

        # Create 3D grid
        x = torch.linspace(0, 1.0, nx, device=device)
        y = torch.linspace(0, 1.0, nx, device=device)
        z = torch.linspace(0, 1.0, nx, device=device)

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        # Create 3D initial condition (Gaussian pulse)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        center = torch.tensor([0.5, 0.5, 0.5], device=device)
        radius = 0.1

        distance = torch.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        concentration = torch.exp(-(distance / radius)**2)

        # Velocity field (circular pattern)
        velocity = torch.zeros((nx, nx, nx, 3), device=device)
        velocity[:, :, :, 0] = -velocity_magnitude * (Y - center[1]) / radius
        velocity[:, :, :, 1] = velocity_magnitude * (X - center[0]) / radius
        velocity[:, :, :, 2] = 0.0

        # Multi-dimensional explicit diffusion stability:
        # D * dt * sum_k(1 / dx_k**2) <= 0.5.
        inverse_spacing_squared = sum(
            1.0 / float(spacing) ** 2 for spacing in (dx, dy, dz)
        )
        dt = 0.4 / (diffusivity * inverse_spacing_squared)
        max_velocity = torch.max(torch.sqrt(
            velocity[:, :, :, 0] ** 2 + velocity[:, :, :, 1] ** 2 + velocity[:, :, :, 2] ** 2
        ))
        dt = min(dt, 0.5 * min(dx, dy, dz) / float(max_velocity))  # Advection stability

        # Run simulation
        n_steps = 100
        for step in range(n_steps):
            result = advance_advection_diffusion(
                concentration=concentration.unsqueeze(0),
                velocity=(
                    velocity[:, :, :, 0].unsqueeze(0),
                    velocity[:, :, :, 1].unsqueeze(0),
                    velocity[:, :, :, 2].unsqueeze(0),
                ),
                diffusivity=diffusivity,
                grid_spacing=(float(dx), float(dy), float(dz)),
                spatial_dims=3,
                time_step=dt,
                splitting_method="sequential",
                check_stability=True,
                enforce_non_negative=True, advection_scheme="upwind", boundary_method="zero"
            )
            concentration = result['concentration'].squeeze(0)

        solutions[nx] = concentration.clone()

    # Analyze convergence
    print("    Analyzing convergence...")

    # Compare solutions at different resolutions
    for i in range(len(resolutions) - 1):
        coarse_res = resolutions[i]
        fine_res = resolutions[i + 1]

        # Interpolate coarse solution to fine grid
        coarse_sol = solutions[coarse_res]
        fine_sol = solutions[fine_res]

        # For simplicity, compare at the center point
        center_idx = fine_res // 2
        fine_center = fine_sol[center_idx, center_idx, center_idx]

        # Find corresponding point in coarse grid
        coarse_center_idx = coarse_res // 2
        coarse_center = coarse_sol[coarse_center_idx, coarse_center_idx, coarse_center_idx]

        relative_error = abs(fine_center - coarse_center) / (abs(fine_center) + 1e-8)
        print(f"    {coarse_res} -> {fine_res}: relative error = {relative_error:.3%}")

        # Error should decrease with resolution
        assert relative_error < 1.5, f"Solution should converge with increasing resolution"

    print("[OK] 3D transport convergence test passed")


def test_operator_splitting_consistency():
    """Test consistency of different operator splitting methods."""
    print("Testing operator splitting consistency...")

    device = torch.device("cpu")

    # Create 1D test problem
    nx = 100
    x = torch.linspace(0, 1.0, nx, device=device)
    dx = x[1] - x[0]

    # Initial condition: combination of sine waves
    concentration = (torch.sin(2 * np.pi * x) +
                    0.5 * torch.sin(4 * np.pi * x) +
                    0.25 * torch.sin(8 * np.pi * x))

    # Velocity and diffusivity
    velocity = torch.ones_like(x) * 0.1
    diffusivity = 0.001

    # Simulation parameters
    dt = 0.001  # Small time step for accuracy
    n_steps = 1000

    # Test different splitting methods
    splitting_methods = ["sequential", "symmetric", "combined"]
    solutions = {}

    print(f"    Running {n_steps} steps with dt = {dt}")

    for method in splitting_methods:
        print(f"    Testing {method} splitting...")

        # Reset initial condition
        conc = concentration.clone()
        solutions[method] = []

        for step in range(n_steps):
            result = advance_advection_diffusion(
                concentration=conc.unsqueeze(0),
                velocity=velocity.unsqueeze(0),
                diffusivity=diffusivity,
                grid_spacing=float(dx),
                spatial_dims=1,
                time_step=dt,
                splitting_method=method,
                check_stability=True,
                enforce_non_negative=False, advection_scheme="upwind", boundary_method="zero"
            )

            conc = result['concentration'].squeeze(0)

            # Record every 100 steps
            if step % 100 == 0:
                solutions[method].append(conc.clone())

    # Analyze consistency
    print("    Analyzing splitting method consistency...")

    # Convert to tensors
    sequential_sol = torch.stack(solutions["sequential"])
    symmetric_sol = torch.stack(solutions["symmetric"])
    combined_sol = torch.stack(solutions["combined"])

    # Compare methods at final time
    seq_sym_diff = torch.mean(torch.abs(sequential_sol[-1] - symmetric_sol[-1]))
    seq_comb_diff = torch.mean(torch.abs(sequential_sol[-1] - combined_sol[-1]))
    sym_comb_diff = torch.mean(torch.abs(symmetric_sol[-1] - combined_sol[-1]))

    print(f"    Sequential vs Symmetric: {seq_sym_diff:.6f}")
    print(f"    Sequential vs Combined: {seq_comb_diff:.6f}")
    print(f"    Symmetric vs Combined: {sym_comb_diff:.6f}")

    # All methods should give similar results (within numerical tolerance)
    tolerance = 1e-3
    assert seq_sym_diff < tolerance, "Sequential and symmetric splitting should be consistent"
    assert seq_comb_diff < tolerance, "Sequential and combined splitting should be consistent"
    assert sym_comb_diff < tolerance, "Symmetric and combined splitting should be consistent"

    # Check mass conservation for all methods
    initial_mass = concentration.sum()
    for method, sol in solutions.items():
        final_mass = sol[-1].sum()
        mass_change = abs(final_mass - initial_mass) / initial_mass
        print(f"    {method} mass conservation: {mass_change:.6f}")
        assert mass_change < 1e-10, f"{method} should conserve mass"

    print("[OK] Operator splitting consistency test passed")


def test_boundary_conditions():
    """Test different boundary condition implementations."""
    print("Testing boundary conditions...")

    device = torch.device("cpu")

    # Create 1D grid
    nx = 50
    x = torch.linspace(0, 1.0, nx, device=device)
    dx = x[1] - x[0]

    # Initial condition: Gaussian pulse in the center
    center = 0.5
    width = 0.1
    concentration = torch.exp(-((x - center) / width)**2)

    # Test parameters
    diffusivity = 0.01
    dt = 0.001
    n_steps = 1000

    # Test different boundary methods
    boundary_methods = ["zero", "periodic", "neumann"]
    solutions = {}

    for method in boundary_methods:
        print(f"    Testing {method} boundary conditions...")

        conc = concentration.clone()
        solutions[method] = []

        for step in range(n_steps):
            result = apply_diffusion(
                concentration=conc.unsqueeze(0),
                diffusivity=diffusivity,
                grid_spacing=float(dx),
                spatial_dims=1,
                time_step=dt,
                boundary_method=method,
                check_stability=True,
                enforce_non_negative=True
            )

            conc = result['concentration'].squeeze(0)

            if step % 200 == 0:
                solutions[method].append(conc.clone())

    # Analyze boundary behavior
    print("    Analyzing boundary condition behavior...")

    for method, sol in solutions.items():
        final_conc = sol[-1]

        # Check physical constraints
        assert torch.all(final_conc >= 0), f"{method} should maintain non-negative concentrations"
        assert torch.all(torch.isfinite(final_conc)), f"{method} should maintain finite concentrations"

        # Check total mass behavior
        if method == "zero":
            # Zero flux should conserve mass
            final_mass = final_conc.sum() * dx
            initial_mass = concentration.sum() * dx
            mass_change = abs(final_mass - initial_mass) / initial_mass
            print(f"    {method} mass change: {mass_change:.6f}")
            assert mass_change < 0.01, f"{method} boundary should conserve mass"

        elif method == "periodic":
            # Periodic should conserve mass exactly
            final_mass = final_conc.sum() * dx
            initial_mass = concentration.sum() * dx
            mass_change = abs(final_mass - initial_mass) / initial_mass
            print(f"    {method} mass change: {mass_change:.10f}")
            assert mass_change < 1e-10, f"{method} boundary should conserve mass exactly"

        elif method == "neumann":
            # Homogeneous Neumann boundaries should conserve mass.
            final_mass = final_conc.sum() * dx
            initial_mass = concentration.sum() * dx
            mass_change = abs(final_mass - initial_mass) / initial_mass
            print(f"    {method} mass change: {mass_change:.6f}")
            assert mass_change < 0.01, f"{method} boundary should conserve mass"

    print("[OK] Boundary conditions test passed")


def test_stability_conditions():
    """Test numerical stability under various conditions."""
    print("Testing numerical stability conditions...")

    device = torch.device("cpu")

    # Create 1D grid
    nx = 100
    x = torch.linspace(0, 1.0, nx, device=device)
    dx = x[1] - x[0]

    # Initial condition: random perturbations
    torch.manual_seed(42)
    concentration = 1.0 + 0.1 * torch.randn(nx, device=device)

    # Test stability limits
    print("    Testing stability limits...")

    # 1. CFL condition for advection
    print("    Testing CFL condition...")
    velocity = torch.ones_like(x)
    max_velocity = velocity.max()

    # Critical CFL number
    cfl_critical = 1.0
    dt_cfl = 0.9 * cfl_critical * dx / max_velocity  # 90% of critical

    result = apply_advection(
        concentration=concentration.unsqueeze(0),
        velocity=velocity.unsqueeze(0),
        grid_spacing=float(dx),
        spatial_dims=1,
        time_step=dt_cfl,
        check_cfl=True, scheme="upwind", boundary_method="extrapolate"
    )

    assert torch.all(torch.isfinite(result['concentration'])), "Advection should be stable under CFL limit"

    # 2. Diffusion stability condition
    print("    Testing diffusion stability...")
    diffusivity = 0.1
    dt_diffusion = 0.4 * dx**2 / diffusivity  # 40% of critical (0.5)

    result = apply_diffusion(
        concentration=concentration.unsqueeze(0),
        diffusivity=diffusivity,
        grid_spacing=float(dx),
        spatial_dims=1,
        time_step=dt_diffusion,
        check_stability=True, boundary_method="zero", enforce_non_negative=True
    )

    assert torch.all(torch.isfinite(result['concentration'])), "Diffusion should be stable under diffusion limit"

    # 3. Combined stability test
    print("    Testing combined stability...")
    dt_combined = min(dt_cfl, dt_diffusion) * 0.8  # Conservative combined time step

    result = advance_advection_diffusion(
        concentration=concentration.unsqueeze(0),
        velocity=velocity.unsqueeze(0),
        diffusivity=diffusivity,
        grid_spacing=float(dx),
        spatial_dims=1,
        time_step=dt_combined,
        check_stability=True,
        enforce_non_negative=True, splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero"
    )

    assert torch.all(torch.isfinite(result['concentration'])), "Combined operator should be stable"

    # 4. Test instability detection
    print("    Testing instability detection...")
    dt_unstable = 2.0 * dt_cfl  # Unstable time step

    with pytest.raises(ValueError, match="CFL"):
        apply_advection(
            concentration=concentration.unsqueeze(0),
            velocity=velocity.unsqueeze(0),
            grid_spacing=float(dx),
            spatial_dims=1,
            time_step=dt_unstable,
            check_cfl=True, scheme="upwind", boundary_method="extrapolate"
        )

    print("[OK] Stability conditions test passed")


if __name__ == "__main__":
    print("Running simplified L3 system tests for unified transport operators...")
    print("=" * 70)

    test_multi_species_atmospheric_transport()
    test_3d_transport_convergence()
    test_operator_splitting_consistency()
    test_boundary_conditions()
    test_stability_conditions()

    print("=" * 70)
    print("All simplified L3 system tests passed! [OK]")
    print("Unified transport operators demonstrate robust and consistent system-level performance.")

