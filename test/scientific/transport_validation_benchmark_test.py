"""
L3 Scientific Validation Tests for Transport Processes

This test suite provides scientific validation of transport processes against
analytical solutions, published results, and physical constraints. It validates
that the refactored transport modules maintain scientific accuracy.

Test Coverage:
- Advection: Comparison with analytical solutions
- Diffusion: Verification against analytical heat equation solutions
- Advection-Diffusion: Benchmark against published test cases
- Conservation laws validation
- Numerical stability and convergence analysis

Mathematical Framework:
    Advection Equation: 鈭侰/鈭倀 + v路鈭嘋 = 0
    Analytical Solution: C(x,t) = C鈧€(x - vt) for constant velocity

    Diffusion Equation: 鈭侰/鈭倀 = D鈭嚶睠
    Analytical Solution: C(x,t) = C鈧€/鈭?4蟺Dt) * exp(-x虏/4Dt) for point source

    Advection-Diffusion: 鈭侰/鈭倀 + v路鈭嘋 = D鈭嚶睠
    Peclet Number: Pe = vL/D (advection vs diffusion dominance)

References:
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of
      Time-Dependent Advection-Diffusion-Reaction Equations. Springer.
    - LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems.
      Cambridge University Press.
    - R. B. Fisher, "The solution of diffusion problems in which the
      diffusion coefficient varies with concentration", Proc. R. Soc. Lond. A
"""

import pytest
import torch
import numpy as np
import math

# Import transport modules
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.core.state import UpdateItem, UpdateMode

def _extract_rate_tensor(rate_output):
    if isinstance(rate_output, dict):
        preferred_keys = [
            "flux",
            "rate",
            "advection_rate",
            "diffusion_rate",
            "concentration_tendency",
            "tracer_concentration",
        ]
        for key in preferred_keys:
            value = rate_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if hasattr(value, "value") and isinstance(value.value, torch.Tensor):
                return value.value
        tensor_values = [
            value for value in rate_output.values() if isinstance(value, torch.Tensor)
        ]
        if tensor_values:
            return tensor_values[0]
        raise TypeError("Transport rate output did not contain a tensor tendency.")
    return rate_output


def _transport_rate(module, state, **kwargs):
    if isinstance(module, LinearAdvection):
        dt = float(kwargs["dt"])
        velocity = kwargs["velocity"]
        velocity_field = tuple(velocity[: module.spatial_dims])
        output = module.forward(
            {
                "tracer_concentration": state,
                "velocity_field": velocity_field,
            },
            time=0.0,
            dt=dt,
        )
        update = output["tracer_concentration"]
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.ACCUMULATE
        assert update.per_timestep is True
        return update.value / dt

    if isinstance(module, FickianDiffusion):
        dt = float(kwargs["dt"])
        output = module.forward(
            {"tracer_concentration": state},
            time=0.0,
            dt=dt,
        )
        update = output["tracer_concentration"]
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.ACCUMULATE
        assert update.per_timestep is True
        return update.value / dt

    raise TypeError(f"Unsupported transport benchmark module: {type(module).__name__}")


def _stable_dt_diffusion_1d(dx: float, diffusivity: float, safety: float = 0.45) -> float:
    diffusivity_value = float(diffusivity)
    if diffusivity_value <= 0.0:
        return float("inf")
    dx_value = float(dx)
    if dx_value <= 0.0:
        raise ValueError("dx must be positive")
    return safety * dx_value * dx_value / diffusivity_value


def _stable_dt_advection_1d(dx: float, velocity: float, safety: float = 0.9) -> float:
    velocity_value = abs(float(velocity))
    if velocity_value <= 0.0:
        return float("inf")
    dx_value = float(dx)
    if dx_value <= 0.0:
        raise ValueError("dx must be positive")
    return safety * dx_value / velocity_value


class TestTransportScientificValidation:
    """Scientific validation tests for transport processes."""

    @pytest.fixture
    def transport_modules(self):
        """Create transport module instances."""
        return {
            'advection': LinearAdvection(
                boundary_method="periodic",
                non_negative=False,
            ),
            'diffusion': FickianDiffusion(),
            'advection_diffusion': AdvectionDiffusion()
        }

    @pytest.fixture
    def analytical_parameters(self):
        """Parameters for analytical solution tests."""
        return {
            'domain_length': 100.0,  # meters
            'grid_points': 200,
            'velocity': 1.0,  # m/s
            'diffusivity': 10.0,  # m虏/s
            'simulation_time': 10.0,  # seconds
            'dt': 0.1  # time step
        }

    def test_advection_analytical_solution(self, transport_modules, analytical_parameters):
        """
        Test advection against analytical solution.

        Mathematical Expression:
            C(x,t) = C鈧€(x - vt) for constant velocity v
            Solution should translate without change in shape
        """
        L = analytical_parameters['domain_length']
        nx = analytical_parameters['grid_points']
        v = analytical_parameters['velocity']
        T = analytical_parameters['simulation_time']
        dt = analytical_parameters['dt']
        dx = L / (nx - 1)
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=dx,
            boundary_method="periodic",
            non_negative=False,
        )

        # Create initial condition: Gaussian pulse
        x = torch.linspace(0, L, nx, dtype=torch.float32)
        x0 = L / 4  # Initial position
        sigma = L / 20  # Pulse width

        C0 = torch.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

        # Initialize state
        state = {
            'concentration': C0.clone(),
            'velocity_x': torch.full_like(C0, v),
            'velocity_y': torch.zeros_like(C0)
        }

        # Time integration
        steps = int(T / dt)
        for step in range(steps):
            # Compute advection rate
            rate = _transport_rate(advection, 
                state['concentration'],
                velocity=(state['velocity_x'], state['velocity_y']),
                dx=dx, dy=1.0, dt=dt
            )

            # Update concentration
            state['concentration'] = state['concentration'] + rate * dt

            # Apply boundary conditions (periodic)
            state['concentration'][0] = state['concentration'][-2]
            state['concentration'][-1] = state['concentration'][1]

        # Analytical solution after time T
        x_final = x0 + v * T
        # Apply periodic boundary to analytical solution
        x_final = x_final % L

        C_analytical = torch.exp(-((x - x_final) ** 2) / (2 * sigma ** 2))

        # Compare numerical and analytical solutions
        numerical_solution = state['concentration']

        # Calculate error metrics
        l2_error = torch.norm(numerical_solution - C_analytical) / torch.norm(C_analytical)
        max_error = torch.max(torch.abs(numerical_solution - C_analytical))

        # Assert solution accuracy (allowing for numerical diffusion)
        assert l2_error < 0.1, f"L2 error too large: {l2_error:.4f}"
        assert max_error < 0.2, f"Max error too large: {max_error:.4f}"

        # Check mass conservation
        initial_mass = torch.sum(C0) * dx
        final_mass = torch.sum(numerical_solution) * dx
        mass_error = torch.abs(final_mass - initial_mass) / initial_mass

        assert mass_error < 0.01, f"Mass not conserved: {mass_error:.4f}"

    def test_diffusion_analytical_solution(self, transport_modules, analytical_parameters):
        """
        Test diffusion against analytical solution for Gaussian pulse.

        Mathematical Expression:
            sigma_t^2 = sigma0^2 + 2*D*t
            C(x,t) = 1/(sqrt(2*pi)*sigma_t) * exp(-(x-x0)^2/(2*sigma_t^2))
        """
        L = analytical_parameters['domain_length']
        nx = analytical_parameters['grid_points']
        D = analytical_parameters['diffusivity']
        T = analytical_parameters['simulation_time']
        dt = analytical_parameters['dt']
        dx = L / (nx - 1)
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=D,
            grid_spacing=dx,
            boundary_method="neumann",
            track_conservation=False,
        )

        # Create initial condition: Delta-like pulse approximated by narrow Gaussian
        x = torch.linspace(0, L, nx, dtype=torch.float32)
        x0 = L / 2  # Center position
        sigma0 = L / 100  # Very narrow initial pulse

        C0 = torch.exp(-((x - x0) ** 2) / (2 * sigma0 ** 2))
        C0 = C0 / (torch.sum(C0) * dx)  # Normalize to unit mass

        # Initialize state
        state = {
            'concentration': C0.clone(),
            'diffusivity': torch.full_like(C0, D)
        }

        dt_stable = _stable_dt_diffusion_1d(dx=dx, diffusivity=float(state["diffusivity"].max().item()))
        dt_eff = min(float(dt), dt_stable)
        steps = max(1, int(math.ceil(T / dt_eff)))
        dt_eff = T / steps

        # Time integration
        for step in range(steps):
            # Compute diffusion rate
            rate = _transport_rate(
                diffusion,
                state['concentration'],
                dt=dt_eff,
            )

            # Update concentration
            state['concentration'] = state['concentration'] + rate * dt_eff

        # Analytical solution after time T
        sigma_t = math.sqrt(2.0 * D * T + sigma0 ** 2)
        C_analytical = torch.exp(-((x - x0) ** 2) / (2 * sigma_t ** 2))
        C_analytical = C_analytical / (math.sqrt(2.0 * math.pi) * sigma_t)

        # Compare numerical and analytical solutions
        numerical_solution = state['concentration']

        # Calculate error metrics
        l2_error = torch.norm(numerical_solution - C_analytical) / torch.norm(C_analytical)
        max_error = torch.max(torch.abs(numerical_solution - C_analytical))

        # Assert solution accuracy
        assert l2_error < 0.05, f"L2 error too large: {l2_error:.4f}"
        assert max_error < 0.1, f"Max error too large: {max_error:.4f}"

        # Check mass conservation
        initial_mass = torch.sum(C0) * dx
        final_mass = torch.sum(numerical_solution) * dx
        mass_error = torch.abs(final_mass - initial_mass) / initial_mass

        assert mass_error < 0.01, f"Mass not conserved: {mass_error:.4f}"

    def test_peclet_number_regimes(self, transport_modules, analytical_parameters):
        """
        Test advection-diffusion across different Peclet number regimes.

        Mathematical Expression:
            Pe = vL/D
            Pe << 1: Diffusion-dominated
            Pe >> 1: Advection-dominated
            Pe 鈮?1: Balanced advection and diffusion
        """
        L = analytical_parameters['domain_length']
        nx = analytical_parameters['grid_points']
        dt = analytical_parameters['dt']
        T = 20.0  # Shorter time for this test
        dx = L / (nx - 1)

        # Test different Peclet numbers
        peclet_numbers = [0.1, 1.0, 10.0, 100.0]

        for Pe in peclet_numbers:
            # Keep diffusivity inside the canonical operator domain and vary
            # velocity to preserve Pe = v L / D exactly.
            D = 0.01
            v = Pe * D / L
            assert math.isclose(v * L / D, Pe, rel_tol=1.0e-12)

            # Create initial condition: Gaussian pulse away from boundaries
            x = torch.linspace(0, L, nx, dtype=torch.float32)
            x0 = L / 2
            sigma = L / 20
            C0 = torch.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
            C0 = C0 / (torch.sum(C0) * dx)

            advection_diffusion = AdvectionDiffusion(
                spatial_dims=1,
                diffusivity=D,
                velocity_field=torch.full_like(C0, v),
                grid_spacing=dx,
                boundary_method="periodic",
            )

            # Initialize state
            state = {
                'concentration': C0.clone(),
                'velocity_x': torch.full_like(C0, v),
                'velocity_y': torch.zeros_like(C0),
                'diffusivity': torch.full_like(C0, D)
            }

            # Time integration
            dt_adv = _stable_dt_advection_1d(dx=dx, velocity=v)
            dt_diff = _stable_dt_diffusion_1d(dx=dx, diffusivity=float(D))
            dt_eff = min(float(dt), dt_adv, dt_diff)
            max_steps = 5000
            steps = max(1, min(int(math.ceil(T / dt_eff)), max_steps))
            for step in range(steps):
                updates = advection_diffusion.forward(
                    {"tracer_concentration": state["concentration"]},
                    time=step * dt_eff,
                    dt=dt_eff,
                )
                state['concentration'] = (
                    state['concentration'] + _extract_rate_tensor(updates)
                )

            # Verify solution properties based on Peclet number
            final_concentration = state['concentration']

            # Check that solution remains bounded and physically meaningful
            assert torch.all(final_concentration >= 0), f"Negative concentrations for Pe={Pe}"
            assert torch.all(final_concentration <= 1.1), f"Concentrations > 1.1 for Pe={Pe}"

            # Check mass conservation
            initial_mass = torch.sum(C0) * dx
            final_mass = torch.sum(final_concentration) * dx
            mass_error = torch.abs(final_mass - initial_mass) / initial_mass

            assert mass_error < 0.05, f"Mass not conserved for Pe={Pe}: {mass_error:.4f}"

    def test_numerical_convergence(self, transport_modules):
        """
        Test numerical convergence with grid refinement.

        Mathematical Expression:
            Error ~ O(dx^p) where p is the order of accuracy
        """
        # Grid sizes for convergence test
        grid_sizes = [50, 100, 200, 400]
        errors = []

        # Fixed parameters
        L = 100.0
        D = 10.0
        T = 1.0
        dt = 0.001

        for nx in grid_sizes:
            dx = L / (nx - 1)
            diffusion = FickianDiffusion(
                spatial_dims=1,
                diffusivity=D,
                grid_spacing=dx,
                boundary_method="neumann",
                track_conservation=False,
            )
            dt_stable = _stable_dt_diffusion_1d(dx=dx, diffusivity=D)
            dt_eff = min(float(dt), dt_stable)
            steps = max(1, int(math.ceil(T / dt_eff)))
            dt_eff = T / steps

            # Create initial condition
            x = torch.linspace(0, L, nx, dtype=torch.float32)
            x0 = L / 2
            sigma = L / 20

            C0 = torch.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
            C0 = C0 / (torch.sum(C0) * dx)

            state = {
                'concentration': C0.clone(),
                'diffusivity': torch.full_like(C0, D)
            }

            # Time integration
            for step in range(steps):
                rate = _transport_rate(
                    diffusion,
                    state['concentration'],
                    dt=dt_eff,
                )
                state['concentration'] = state['concentration'] + rate * dt_eff

            # Use analytical solution for error calculation
            sigma_t = math.sqrt(2.0 * D * T + sigma ** 2)
            C_analytical = torch.exp(-((x - x0) ** 2) / (2 * sigma_t ** 2))
            C_analytical = C_analytical / (math.sqrt(2.0 * math.pi) * sigma_t)

            # Calculate error
            error = torch.norm(state['concentration'] - C_analytical) / torch.norm(C_analytical)
            errors.append(error.item())

        # Check convergence (error should decrease with grid refinement)
        for i in range(1, len(errors)):
            assert errors[i] < errors[i-1], f"No convergence: error[{grid_sizes[i]}] >= error[{grid_sizes[i-1]}]"

        # Estimate order of convergence
        if len(errors) >= 2:
            convergence_rate = np.log(errors[-1] / errors[-2]) / np.log(grid_sizes[-2] / grid_sizes[-1])
            assert convergence_rate > 0.5, f"Convergence rate too low: {convergence_rate:.2f}"

    def test_stability_conditions(self, transport_modules):
        """
        Test numerical stability conditions.

        Mathematical Expression:
            CFL condition: v*螖t/螖x < 1 for advection
            Diffusion stability: D*螖t/螖x虏 < 0.5 for explicit schemes
        """
        advection = transport_modules['advection']
        diffusion = transport_modules['diffusion']

        # Test parameters that should be stable
        stable_params = [
            {'dx': 1.0, 'dt': 0.5, 'v': 1.0, 'D': 0.1},  # CFL = 0.5, diffusion number = 0.05
            {'dx': 1.0, 'dt': 0.8, 'v': 1.0, 'D': 0.1},  # CFL = 0.8, diffusion number = 0.08
        ]

        # Test parameters that might be unstable
        unstable_params = [
            {'dx': 1.0, 'dt': 2.0, 'v': 1.0, 'D': 0.1},  # CFL = 2.0 (violates CFL)
            {'dx': 1.0, 'dt': 1.0, 'v': 1.0, 'D': 1.0},  # diffusion number = 1.0 (unstable)
        ]

        nx = 50
        x = torch.linspace(0, 50.0, nx, dtype=torch.float32)

        for params in stable_params:
            # Test advection stability
            C0 = 1.0 + 0.1 * torch.sin(2 * np.pi * x / 50.0)
            state = {
                'concentration': C0.clone(),
                'velocity_x': torch.full_like(C0, params['v']),
                'velocity_y': torch.zeros_like(C0)
            }

            try:
                # Multiple time steps to test stability
                for _ in range(100):
                    rate = _transport_rate(advection, 
                        state['concentration'],
                        velocity=(state['velocity_x'], state['velocity_y']),
                        dx=params['dx'], dy=1.0, dt=params['dt']
                    )
                    state['concentration'] = state['concentration'] + rate * params['dt']

                    # Check for numerical instabilities
                    assert torch.all(torch.isfinite(state['concentration'])), "Non-finite values detected"
                    assert torch.max(torch.abs(state['concentration'])) < 1e6, "Numerical blow-up detected"

            except Exception as e:
                pytest.fail(f"Stable parameters caused instability: {params}, error: {e}")

    def test_conservation_laws(self, transport_modules):
        """
        Test conservation laws for transport processes.

        Mathematical Expression:
            d/dt integral(C dV) + integral(F·n dS) = integral(S dV)
            For conservative transport: integral(C dV) is constant if S=0 and boundary flux is closed
        """
        # Test parameters
        L = 100.0
        nx = 100
        dx = L / (nx - 1)
        diffusivity_value = 0.01
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=diffusivity_value,
            grid_spacing=dx,
            boundary_method="periodic",
            track_conservation=False,
        )
        T = 50.0
        dt = 0.1
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=dx,
            boundary_method="periodic",
            non_negative=False,
        )

        # Initial condition
        x = torch.linspace(0, L, nx, dtype=torch.float32)
        C0 = torch.exp(-((x - L/2) ** 2) / (2 * (L/10) ** 2))
        initial_mass = torch.sum(C0) * dx
        advection_diffusion = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=diffusivity_value,
            velocity_field=torch.ones_like(C0),
            grid_spacing=dx,
            boundary_method="periodic",
        )

        # Test each transport module
        modules = [
            ('advection', advection, {'velocity_x': torch.ones_like(C0), 'velocity_y': torch.zeros_like(C0)}),
            ('diffusion', diffusion, {'diffusivity': torch.ones_like(C0) * diffusivity_value}),
            ('advection_diffusion', advection_diffusion, {
                'velocity_x': torch.ones_like(C0),
                'velocity_y': torch.zeros_like(C0),
                'diffusivity': torch.ones_like(C0) * diffusivity_value
            })
        ]

        for name, module, extra_state in modules:
            state = {
                'concentration': C0.clone(),
                **extra_state
            }

            # Time integration
            dt_eff = float(dt)
            if "diffusivity" in state:
                dt_eff = min(
                    dt_eff,
                    _stable_dt_diffusion_1d(dx=dx, diffusivity=float(state["diffusivity"].max().item())),
                )
            if "velocity_x" in state:
                dt_eff = min(
                    dt_eff,
                    _stable_dt_advection_1d(dx=dx, velocity=float(state["velocity_x"].abs().max().item())),
                )
            steps = max(1, int(math.ceil(T / dt_eff)))
            dt_eff = T / steps
            for step in range(steps):
                if name == 'advection':
                    rate = _transport_rate(
                        module,
                        state['concentration'],
                        velocity=(state['velocity_x'], state['velocity_y']),
                        dx=dx, dy=1.0, dt=dt_eff
                    )
                elif name == 'diffusion':
                    rate = _transport_rate(
                        module,
                        state['concentration'],
                        dt=dt_eff,
                    )
                else:  # advection_diffusion
                    updates = module.forward(
                        {"tracer_concentration": state["concentration"]},
                        time=step * dt_eff,
                        dt=dt_eff,
                    )
                    state['concentration'] = (
                        state['concentration'] + _extract_rate_tensor(updates)
                    )
                    continue
                state['concentration'] = state['concentration'] + rate * dt_eff

            # Check mass conservation
            final_mass = torch.sum(state['concentration']) * dx
            mass_error = torch.abs(final_mass - initial_mass) / initial_mass

            assert mass_error < 0.01, f"Mass not conserved for {name}: {mass_error:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

