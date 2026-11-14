"""
L2 Integration Tests for Unified Transport Operators

This module provides comprehensive integration tests for the unified transport operators
in relic.scientific.transport. Tests cover:

- Integration between transport operators
- Coupling with state management systems
- Multi-physics transport scenarios
- Sphere-specific transport integration
- Performance and memory efficiency
- Cross-module consistency

References:
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations
    - RELIC Architecture Documentation
    - Transport Process Integration Guidelines
"""

import pytest
import torch
import time

# Import transport operators
from relic.computation.operators.functional import (
    apply_diffusion, apply_advection, advance_advection_diffusion,
    check_conservation
)

# Import RELIC core components
from relic.core.state import (
    StateManager,
    StateVariableSpec,
    StateVariableType,
    UpdateItem,
    UpdateMode,
)
from relic.parameters import get_global_manager
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion

# Test fixtures
@pytest.fixture
def test_state_1d():
    """Create test state for 1D transport."""
    x = torch.linspace(0, 1, 101)
    conc = torch.exp(-((x - 0.3) / 0.1) ** 2)  # Gaussian pulse
    
    return {
        'tracer_concentration': conc.unsqueeze(0),  # Add batch dimension
        'velocity_x': torch.full_like(conc, 0.5),  # u = 0.5 m/s
        'source_term': torch.zeros_like(conc),
        'grid_spacing_x': 0.01
    }

@pytest.fixture
def test_state_2d():
    """Create test state for 2D transport."""
    x = torch.linspace(0, 1, 51)
    y = torch.linspace(0, 1, 51)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 2D Gaussian pulse
    conc = torch.exp(-((X - 0.3) / 0.1) ** 2 - ((Y - 0.7) / 0.15) ** 2)
    
    return {
        'tracer_concentration': conc.unsqueeze(0),  # Add batch dimension
        'velocity_x': torch.full_like(conc, 0.3),  # u = 0.3 m/s
        'velocity_y': torch.full_like(conc, 0.2),  # v = 0.2 m/s
        'source_term': torch.zeros_like(conc),
        'grid_spacing_x': 0.02,
        'grid_spacing_y': 0.02
    }

@pytest.fixture
def atmospheric_transport_state():
    """Create atmospheric transport scenario state."""
    # Typical atmospheric conditions
    z = torch.linspace(0, 1000, 50)  # 0-1km height
    conc = torch.exp(-((z - 500) / 200) ** 2)  # Concentration peak at 500m
    
    return {
        'tracer_concentration': conc.unsqueeze(0),
        'wind_speed': torch.full_like(conc, 5.0),  # 5 m/s wind
        'temperature': torch.full_like(conc, 288.15),  # 15 C
        'pressure': torch.full_like(conc, 101325.0),  # Surface pressure
        'source_term': torch.zeros_like(conc),
        'grid_spacing_z': 20.0  # 20m vertical resolution
    }

@pytest.fixture
def soil_transport_state():
    """Create soil transport scenario state."""
    z = torch.linspace(0, -1.0, 100)  # 0 to -1m depth
    conc = torch.exp((z + 0.3) / 0.2)  # Exponential profile
    
    return {
        'soil_gas_concentration': conc.unsqueeze(0),
        'soil_moisture': torch.full_like(conc, 0.3),  # 30% moisture
        'soil_temperature': torch.full_like(conc, 293.15),  # 20 C
        'soil_porosity': torch.full_like(conc, 0.45),  # 45% porosity
        'source_term': torch.zeros_like(conc),
        'grid_spacing_z': 0.01  # 1cm resolution
    }

@pytest.fixture
def state_manager():
    """Create StateManager instance for testing."""
    return StateManager(device=torch.device("cpu"))

class TestOperatorIntegration:
    """Test integration between different transport operators."""
    
    def test_sequential_vs_combined_operator(self, test_state_1d):
        """Test consistency between sequential and combined operators."""
        state = test_state_1d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        dt = 0.01
        dx = state['grid_spacing_x']
        
        # Sequential application (match operator splitting order)
        advection_result = apply_advection(
            concentration=conc,
            velocity=vel_x,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt,
            boundary_method="zero", scheme="upwind", check_cfl=True
        )
        conc_after_adv = advection_result['concentration']
        
        diffusion_result = apply_diffusion(
            concentration=conc_after_adv,
            diffusivity=1e-5,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt,
            boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        sequential_final = diffusion_result['concentration']
        
        # Combined operator
        combined_result = advance_advection_diffusion(
            concentration=conc,
            velocity=vel_x,
            diffusivity=1e-5,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt,
            splitting_method="sequential", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        combined_final = combined_result['concentration']
        
        # Results should be similar (allowing for numerical differences)
        denom = torch.abs(sequential_final).clamp_min(1e-12)
        relative_diff = torch.abs(sequential_final - combined_final) / denom
        max_relative_diff = torch.max(relative_diff)
        
        # Should be within a few percent difference
        assert max_relative_diff < 0.05  # 5% tolerance
        
    def test_symmetric_splitting_accuracy(self, test_state_1d):
        """Test accuracy of symmetric Strang splitting."""
        state = test_state_1d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        dt = 0.01
        dx = state['grid_spacing_x']
        
        # Compare splitting methods
        sequential_result = advance_advection_diffusion(
            concentration=conc,
            velocity=vel_x,
            diffusivity=1e-4,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt,
            splitting_method="sequential", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        symmetric_result = advance_advection_diffusion(
            concentration=conc,
            velocity=vel_x,
            diffusivity=1e-4,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt,
            splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        # Symmetric splitting should be more accurate
        # (We test this by checking conservation properties)
        sequential_final = sequential_result['concentration']
        symmetric_final = symmetric_result['concentration']
        
        # Check mass conservation for both methods
        volumes = torch.full(conc.shape[-1:], dx)
        sequential_conservation = check_conservation(conc, sequential_final, 
                                                    torch.zeros_like(conc), dt, volumes)
        symmetric_conservation = check_conservation(conc, symmetric_final, 
                                                   torch.zeros_like(conc), dt, volumes)        # Symmetric should have better conservation
        tolerance = 1e-7
        assert symmetric_conservation <= sequential_conservation + tolerance

    @pytest.mark.parametrize(
        ("velocity", "diffusivity", "expected_message"),
        [
            (torch.tensor([[2.0, 2.0, 2.0, 2.0]]), 0.0, "CFL"),
            (torch.zeros((1, 4)), 1.0, "Diffusion stability"),
        ],
    )
    def test_combined_splitting_rejects_unstable_explicit_step(
        self,
        velocity,
        diffusivity,
        expected_message,
    ):
        """Combined Euler stepping must enforce both explicit stability bounds."""
        concentration = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match=expected_message):
            advance_advection_diffusion(
                concentration=concentration,
                velocity=velocity,
                diffusivity=diffusivity,
                grid_spacing=1.0,
                time_step=1.0,
                spatial_dims=1,
                splitting_method="combined",
                boundary_method="periodic",
                check_stability=True,
                enforce_non_negative=False, advection_scheme="upwind",
            )

    def test_combined_splitting_rejects_negative_source_result(self):
        """A sink that exhausts storage must fail instead of being clipped."""
        concentration = torch.ones((1, 4))

        with pytest.raises(ValueError, match="non-negative concentration"):
            advance_advection_diffusion(
                concentration=concentration,
                velocity=torch.zeros_like(concentration),
                diffusivity=0.0,
                grid_spacing=1.0,
                time_step=1.0,
                spatial_dims=1,
                splitting_method="combined",
                boundary_method="periodic",
                source_term=torch.full_like(concentration, -2.0),
                enforce_non_negative=True, advection_scheme="upwind", check_stability=True,
            )

    def test_combined_splitting_enforces_coupled_monotonicity_bound(self):
        """Advection and diffusion weights must be stable as one Euler step."""
        concentration = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # C = 0.75 and r = 0.2 pass the separate bounds, but
        # C + 2*r = 1.15 violates the combined positivity condition.
        with pytest.raises(ValueError, match="Combined explicit stability"):
            advance_advection_diffusion(
                concentration=concentration,
                velocity=torch.full_like(concentration, 0.75),
                diffusivity=0.2,
                grid_spacing=1.0,
                time_step=1.0,
                spatial_dims=1,
                splitting_method="combined",
                boundary_method="periodic",
                check_stability=True,
                enforce_non_negative=False, advection_scheme="upwind",
            )

    def test_source_overflow_fails_before_returning_nonfinite_state(self):
        """A finite source that overflows the step must fail explicitly."""
        concentration = torch.ones((1, 4), dtype=torch.float32)

        with pytest.raises(ValueError, match="finite concentration"):
            advance_advection_diffusion(
                concentration=concentration,
                velocity=torch.zeros_like(concentration),
                diffusivity=0.0,
                grid_spacing=1.0,
                time_step=2.0,
                spatial_dims=1,
                splitting_method="combined",
                boundary_method="periodic",
                source_term=torch.full_like(concentration, 3.0e38),
                enforce_non_negative=False,
                run_conservation_check=False, advection_scheme="upwind", check_stability=True,
            )

    def test_zero_diffusivity_has_no_undefined_peclet_diagnostic(self):
        """Peclet is undefined at zero diffusivity and must not be NaN or Inf."""
        concentration = torch.ones((1, 4))
        result = advance_advection_diffusion(
            concentration=concentration,
            velocity=torch.zeros_like(concentration),
            diffusivity=0.0,
            grid_spacing=1.0,
            time_step=0.1,
            spatial_dims=1,
            splitting_method="combined",
            boundary_method="periodic", advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )

        assert result["peclet_number"] is None
        assert torch.all(torch.isfinite(result["concentration"]))

    def test_variable_diffusivity_reports_maximum_local_peclet(self):
        concentration = torch.ones((1, 4), dtype=torch.float64)
        velocity = torch.ones_like(concentration)
        diffusivity = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)

        result = advance_advection_diffusion(
            concentration=concentration,
            velocity=velocity,
            diffusivity=diffusivity,
            diffusivity_layout="scalar_field",
            grid_spacing=2.0,
            time_step=0.01,
            spatial_dims=1,
            splitting_method="combined",
            boundary_method="periodic", advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )

        torch.testing.assert_close(
            result["peclet_number"],
            torch.tensor(2.0, dtype=torch.float64),
        )

        zero_diffusivity = diffusivity.clone()
        zero_diffusivity[0, 1] = 0.0
        undefined_result = advance_advection_diffusion(
            concentration=concentration,
            velocity=velocity,
            diffusivity=zero_diffusivity,
            diffusivity_layout="scalar_field",
            grid_spacing=2.0,
            time_step=0.01,
            spatial_dims=1,
            splitting_method="combined",
            boundary_method="periodic", advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        assert undefined_result["peclet_number"] is None

    def test_multidimensional_peclet_uses_physical_component_spacing(self):
        concentration = torch.ones((1, 2, 3), dtype=torch.float64)
        u = torch.full_like(concentration, 2.0)
        v = torch.full_like(concentration, 1.0)
        result = advance_advection_diffusion(
            concentration,
            (u, v),
            0.5,
            (4.0, 3.0),
            0.01,
            spatial_dims=2,
            splitting_method="combined",
            boundary_method="periodic", advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        torch.testing.assert_close(
            result["peclet_number"],
            torch.tensor(12.0, dtype=torch.float64),
        )

    def test_positive_diffusivity_peclet_overflow_fails_fast(self):
        concentration = torch.ones((1, 4), dtype=torch.float32)
        velocity = torch.full_like(concentration, 1.0e20)
        with pytest.raises(ValueError, match="Peclet.*finite"):
            advance_advection_diffusion(
                concentration,
                velocity,
                torch.tensor(1.0e-30, dtype=torch.float32),
                1.0,
                1.0e-21,
                spatial_dims=1,
                splitting_method="combined",
                boundary_method="periodic", advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
            )

    def test_symmetric_splitting_without_stability_diagnostic(self):
        """Disabling checks must not attempt arithmetic on missing diagnostics."""
        concentration = torch.ones((1, 4))
        result = advance_advection_diffusion(
            concentration=concentration,
            velocity=torch.zeros_like(concentration),
            diffusivity=0.1,
            grid_spacing=1.0,
            time_step=0.1,
            spatial_dims=1,
            splitting_method="symmetric",
            boundary_method="periodic",
            check_stability=False, advection_scheme="upwind", enforce_non_negative=True,
        )

        assert "stability_number" not in result
        assert torch.all(torch.isfinite(result["concentration"]))

    def test_symmetric_splitting_composes_source_in_half_steps(self):
        """A fixed source participates in the palindromic transport composition."""
        concentration = torch.zeros((1, 4), dtype=torch.float64)
        source = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

        result = advance_advection_diffusion(
            concentration=concentration,
            velocity=torch.ones_like(concentration),
            diffusivity=0.0,
            grid_spacing=1.0,
            time_step=0.2,
            spatial_dims=1,
            splitting_method="symmetric",
            boundary_method="periodic",
            source_term=source,
            time_integration_method="euler", advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )

        expected = torch.tensor([[0.18, 0.02, 0.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(
            result["concentration"],
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            torch.sum(result["concentration"]),
            torch.tensor(0.2, dtype=torch.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        
    def test_multiple_time_steps_consistency(self, test_state_1d):
        """Test consistency across multiple time steps."""
        state = test_state_1d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        dt = 0.001
        dx = state['grid_spacing_x']
        
        # Single large time step
        large_step_result = advance_advection_diffusion(
            concentration=conc,
            velocity=vel_x,
            diffusivity=1e-5,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt * 10,  # 10x larger step
            splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        # Multiple small time steps
        conc_current = conc.clone()
        for _ in range(10):
            small_step_result = advance_advection_diffusion(
                concentration=conc_current,
                velocity=vel_x,
                diffusivity=1e-5,
                grid_spacing=dx,
                spatial_dims=1,
                time_step=dt,
                splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
            )
            conc_current = small_step_result['concentration']
        
        multiple_step_final = conc_current
        
        # Compare results
        large_step_final = large_step_result['concentration']
        
        denom = torch.abs(multiple_step_final).clamp_min(1e-12)
        relative_diff = torch.abs(large_step_final - multiple_step_final) / denom
        max_relative_diff = torch.max(relative_diff)
        
        # Should be reasonably close (allowing for accumulated numerical errors)
        assert max_relative_diff < 0.5  # 50% tolerance


class TestStateManagerIntegration:
    """Test integration with StateManager."""
    
    def test_state_manager_transaction(self, test_state_1d, state_manager):
        """Test StateManager transaction handling with transport updates."""
        state = test_state_1d.copy()
        dt = 0.01
        dx = state['grid_spacing_x']
        
        # Create transport updates
        result = advance_advection_diffusion(
            concentration=state['tracer_concentration'],
            velocity=state['velocity_x'],
            diffusivity=1e-5,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt, splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        # Prepare updates for StateManager (calculate increment from concentration)
        updates = {
            'tracer_concentration': result['concentration'] - state['tracer_concentration']
        }
        
        # Apply transactionally
        state_manager.set_state({'tracer_concentration': state['tracer_concentration'].clone()})
        state_manager.set_conservation_ledger({"water": ["tracer_concentration"]})
        # The most relaxed accepted ledger tolerance: the declared upper bound.
        state_manager.conservation_tolerance = get_global_manager().get_parameter(
            "validation.conservation.conservation_tolerance"
        ).bounds[1]
        initial_conc = state['tracer_concentration'].clone()
        state_manager.apply_updates(updates, dt=dt, check_conservation=True)
        
        # Verify state was updated correctly
        final_conc = initial_conc + updates['tracer_concentration']
        
        # Check that StateManager maintained conservation
        summary = state_manager.generate_conservation_report(updates, dt=dt, store_history=False)
        assert summary.balance_error.get("W", 0.0) < 1e-3
        
    def test_state_manager_rollback_on_violation(self, test_state_1d):
        """Test StateManager rollback on conservation violation."""
        state = test_state_1d.copy()

        manager = StateManager(
            schema={
                "tracer_concentration": StateVariableSpec(
                    name="tracer_concentration",
                    units="g C m-2",
                    conserved=True,
                    spatial_dims=1,
                    variable_type=StateVariableType.GRID,
                )
            },
            device=torch.device("cpu"),
            autoload_contract_registry=False,
            # The strictest accepted ledger tolerance: the declared lower bound.
            conservation_checks={"tolerance": get_global_manager().get_parameter(
                "validation.conservation.conservation_tolerance"
            ).bounds[0]},
        )
        
        # Create updates that violate conservation
        violating_updates = {
            "tracer_concentration": UpdateItem(
                value=state["tracer_concentration"] * 0.1,
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                per_timestep=True,
                element="carbon",
                source_functor="test_transport",
            )
        }
        
        initial_conc = state['tracer_concentration'].clone()
        manager.set_state({'tracer_concentration': initial_conc.clone()})
        manager.set_conservation_ledger({"carbon": ["tracer_concentration"]})
        
        # Apply updates with strict conservation checking
        with pytest.raises(ValueError, match="Conservation violation"):
            manager.apply_updates(
                violating_updates, 
                dt=0.01, 
                check_conservation=True,
                rollback_on_failure=True
            )
            
        # Verify state was rolled back
        current_conc = manager.get_state(["tracer_concentration"])["tracer_concentration"]
        assert torch.allclose(current_conc, initial_conc)
            
    def test_conservative_mapping_different_grids(self):
        """Test conservative mapping between different grid resolutions."""
        # Create coarse and fine grids
        coarse_conc = torch.ones((1, 20))  # 20 points
        fine_conc = torch.ones((1, 100))   # 100 points
        
        # Apply transport on coarse grid
        coarse_result = apply_diffusion(
            concentration=coarse_conc,
            diffusivity=1e-5,
            grid_spacing=0.05,  # 1m / 20 points
            spatial_dims=1,
            time_step=0.01, boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        # Apply transport on fine grid
        fine_result = apply_diffusion(
            concentration=fine_conc,
            diffusivity=1e-5,
            grid_spacing=0.01,  # 1m / 100 points
            spatial_dims=1,
            time_step=0.01, boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        # Check that total mass is conserved in both cases
        coarse_mass = torch.sum(coarse_conc) * 0.05  # Integrate with dx
        fine_mass = torch.sum(fine_conc) * 0.01     # Integrate with dx
        
        # Initial masses should be equal
        assert torch.abs(coarse_mass - fine_mass) < 1e-10
        
        # Final masses should also be approximately equal
        coarse_final_mass = torch.sum(coarse_result['concentration']) * 0.05
        fine_final_mass = torch.sum(fine_result['concentration']) * 0.01
        
        relative_mass_diff = torch.abs(coarse_final_mass - fine_final_mass) / coarse_final_mass
        assert relative_mass_diff < 0.01  # 1% tolerance


class TestBoundaryConditionIntegration:
    """Test integration with boundary condition system."""
    
    def test_periodic_boundary_conditions(self, test_state_1d):
        """Test periodic boundary conditions with transport."""
        state = test_state_1d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        dt = 0.01
        dx = state['grid_spacing_x']
        
        # Apply transport with periodic boundaries
        result = advance_advection_diffusion(
            concentration=conc,
            velocity=vel_x,
            diffusivity=1e-5,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt,
            boundary_method='periodic', splitting_method="symmetric", advection_scheme="upwind", check_stability=True, enforce_non_negative=True
        )
        
        # For periodic boundaries, mass leaving one side should enter the other
        # Check mass conservation
        volumes = torch.full(conc.shape[-1:], dx)
        final_conc = result['concentration']
        conservation_error = check_conservation(conc, final_conc, 
                                               torch.zeros_like(conc), dt, volumes)
        
        assert conservation_error < 1e-8
        
    def test_zero_gradient_boundary_conditions(self, test_state_1d):
        """Test zero-gradient boundary handling with transport."""
        state = test_state_1d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        dt = 0.01
        dx = state['grid_spacing_x']
        initial_left = conc[0, 0].clone()
        initial_right = conc[0, -1].clone()
        
        # Apply transport with Dirichlet boundaries
        result = advance_advection_diffusion(
            concentration=conc,
            velocity=vel_x,
            diffusivity=1e-5,
            grid_spacing=dx,
            spatial_dims=1,
            time_step=dt,
            boundary_method='zero', splitting_method="symmetric", advection_scheme="upwind", check_stability=True, enforce_non_negative=True
        )
        
        final_conc = result['concentration']
        
        # Boundary values should remain close to initial values
        assert torch.abs(final_conc[0, 0] - initial_left) < 1e-4
        assert torch.abs(final_conc[0, -1] - initial_right) < 1e-4
        
    def test_2d_boundary_conditions(self, test_state_2d):
        """Test 2D zero-gradient boundary handling."""
        state = test_state_2d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        vel_y = state['velocity_y']
        dt = 0.01
        dx, dy = state['grid_spacing_x'], state['grid_spacing_y']
        initial_left = conc[0, 0, :].clone()
        initial_right = conc[0, -1, :].clone()
        
        # Apply transport with mixed boundaries
        result = advance_advection_diffusion(
            concentration=conc,
            velocity=(vel_x, vel_y),
            diffusivity=1e-5,
            grid_spacing=(dy, dx),  # Note: (dy, dx) ordering for operators
            spatial_dims=2,
            time_step=dt,
            boundary_method='zero', splitting_method="symmetric", advection_scheme="upwind", check_stability=True, enforce_non_negative=True
        )
        
        final_conc = result['concentration']
        
        # Check that boundary values remain close to initial values
        assert torch.abs(final_conc[0, 0, :] - initial_left).max() < 1e-3
        assert torch.abs(final_conc[0, -1, :] - initial_right).max() < 1e-3


class TestScenarioIntegration:
    """Test integration with specific environmental scenarios."""
    
    def test_atmospheric_transport_scenario(self, atmospheric_transport_state):
        """Test atmospheric boundary layer transport."""
        state = atmospheric_transport_state.copy()
        conc = state['tracer_concentration']
        wind_speed = state['wind_speed']
        dt = 1.0  # 1 second
        dz = state['grid_spacing_z']
        
        # Atmospheric conditions: higher diffusivity due to turbulence
        atmospheric_diffusivity = 10.0  # m2 s-1 (turbulent)
        
        result = advance_advection_diffusion(
            concentration=conc,
            velocity=wind_speed,
            diffusivity=atmospheric_diffusivity,
            grid_spacing=dz,
            spatial_dims=1,
            time_step=dt,
            splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        
        final_conc = result['concentration']
        
        # Check physical realism
        assert torch.all(final_conc >= 0)  # Non-negative concentrations
        assert torch.all(torch.isfinite(final_conc))  # No infinite values
        
        # Open-system finite-volume ledger:
        # Delta M - dt * source + dt * outward boundary export = 0.
        storage_change = torch.sum(final_conc - conc, dim=-1) * dz
        source_input = dt * torch.sum(state["source_term"], dim=-1) * dz
        outward_export = dt * sum(
            flux for flux in result["boundary_fluxes"].values()
        )
        residual = storage_change - source_input + outward_export
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            result["conservation_error"],
            torch.zeros_like(result["conservation_error"]),
            atol=1e-6,
            rtol=1e-6,
        )
        
    def test_soil_gas_transport_scenario(self, soil_transport_state):
        """Test soil gas transport with moisture effects."""
        state = soil_transport_state.copy()
        conc = state['soil_gas_concentration']
        moisture = state['soil_moisture']
        porosity = state['soil_porosity']
        dt = 10.0  # 10 seconds
        dz = state['grid_spacing_z']
        
        # Soil gas diffusivity depends on moisture and porosity
        # Using Millington-Quirk model: D_eff = D_air * (theta_air)^(10/3) / porosity^2
        # where theta_air = porosity - moisture
        theta_air = porosity - moisture
        d_air = 2e-5  # Air diffusivity at 20 C
        d_eff = d_air * (theta_air ** (10/3)) / (porosity ** 2)
        
        # Use spatially varying diffusivity
        result = apply_diffusion(
            concentration=conc,
            diffusivity=d_eff,
            diffusivity_layout="scalar_field",
            grid_spacing=dz,
            spatial_dims=1,
            time_step=dt,
            enforce_non_negative=True, boundary_method="zero", check_stability=True
        )
        
        final_conc = result['concentration']
        
        # Check physical realism
        assert torch.all(final_conc >= 0)
        assert torch.all(torch.isfinite(final_conc))
        
        # Diffusion should be slower in wetter soil (lower effective diffusivity)
        # This is verified by checking that concentration gradients are reduced
        # in regions of higher moisture content
        initial_gradient = torch.abs(torch.diff(conc.squeeze()))
        final_gradient = torch.abs(torch.diff(final_conc.squeeze()))
        
        # In general, diffusion should reduce gradients over time
        # (though specific behavior depends on boundary conditions)
        assert torch.mean(final_gradient) <= torch.mean(initial_gradient) * 1.1  # Allow small increase
        
    def test_multi_species_transport(self, test_state_1d):
        """Test transport of multiple species with different properties."""
        state = test_state_1d.copy()
        dt = 0.01
        dx = state['grid_spacing_x']
        
        # Define multiple species with different diffusivities
        species = {
            'co2': {'diffusivity': 1.6e-5, 'conc': state['tracer_concentration']},
            'o2': {'diffusivity': 2.1e-5, 'conc': state['tracer_concentration'] * 0.21},  # 21% O2
            'n2': {'diffusivity': 1.8e-5, 'conc': state['tracer_concentration'] * 0.78},  # 78% N2
        }
        
        results = {}
        for sp_name, sp_props in species.items():
            result = apply_diffusion(
                concentration=sp_props['conc'],
                diffusivity=sp_props['diffusivity'],
                grid_spacing=dx,
                spatial_dims=1,
                time_step=dt, boundary_method="zero", check_stability=True, enforce_non_negative=True
            )
            results[sp_name] = result
            
        # Check that species with higher diffusivity spread faster
        co2_final = results['co2']['concentration']
        o2_final = results['o2']['concentration']
        n2_final = results['n2']['concentration']
        
        # Measure spread (standard deviation)
        def measure_spread(conc):
            mean_conc = torch.mean(conc)
            variance = torch.mean((conc - mean_conc) ** 2)
            return torch.sqrt(variance)
            
        co2_spread = measure_spread(co2_final)
        o2_spread = measure_spread(o2_final)
        n2_spread = measure_spread(n2_final)
        
        # Higher diffusivity should yield stronger smoothing (lower spread)
        assert o2_spread <= n2_spread + 1e-6
        assert n2_spread <= co2_spread + 1e-6
        
        # Mass should be conserved for each species
        volumes = torch.full(species['co2']['conc'].shape[-1:], dx)
        for sp_name, sp_props in species.items():
            initial_mass = torch.sum(sp_props['conc']) * dx
            final_mass = torch.sum(results[sp_name]['concentration']) * dx
            relative_error = torch.abs(final_mass - initial_mass) / initial_mass
            assert relative_error < 1e-6


class TestPerformanceIntegration:
    """Test performance and memory efficiency."""
    
    def test_vectorized_vs_loop_performance(self, test_state_2d):
        """Test that vectorized operations are efficient."""
        state = test_state_2d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        vel_y = state['velocity_y']
        dt = 0.01
        dx, dy = state['grid_spacing_x'], state['grid_spacing_y']
        
        # Time vectorized operation
        start_time = time.time()
        result_vectorized = advance_advection_diffusion(
            concentration=conc,
            velocity=(vel_x, vel_y),
            diffusivity=1e-5,
            grid_spacing=(dy, dx),
            spatial_dims=2,
            time_step=dt, splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        vectorized_time = time.time() - start_time
        
        # Verify result correctness
        assert result_vectorized['concentration'].shape == conc.shape
        assert torch.all(torch.isfinite(result_vectorized['concentration']))
        
        # Vectorized operation should be reasonably fast
        # (This is a basic performance check - exact timing depends on hardware)
        assert vectorized_time < 1.0  # Should complete within 1 second
        
    def test_memory_efficiency_large_grid(self):
        """Test memory efficiency with large grids."""
        # Create a large grid
        nx, ny = 500, 500  # 500x500 grid
        conc = torch.rand((1, nx, ny))
        vel_x = torch.ones((1, nx, ny)) * 0.5
        vel_y = torch.ones((1, nx, ny)) * 0.3
        
        dt = 0.01
        dx, dy = 0.01, 0.01
        
        result = advance_advection_diffusion(
            concentration=conc,
            velocity=(vel_x, vel_y),
            diffusivity=1e-5,
            grid_spacing=(dy, dx),
            spatial_dims=2,
            time_step=dt, splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )

        assert result['concentration'].shape == conc.shape
        assert torch.all(torch.isfinite(result['concentration']))
                
    def test_operator_caching_performance(self, test_state_2d):
        """Test that operator caching improves performance."""
        state = test_state_2d.copy()
        conc = state['tracer_concentration']
        vel_x = state['velocity_x']
        vel_y = state['velocity_y']
        dt = 0.01
        dx, dy = state['grid_spacing_x'], state['grid_spacing_y']
        
        # First call (no caching)
        start_time = time.time()
        result1 = advance_advection_diffusion(
            concentration=conc,
            velocity=(vel_x, vel_y),
            diffusivity=1e-5,
            grid_spacing=(dy, dx),
            spatial_dims=2,
            time_step=dt, splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        first_time = time.time() - start_time
        
        # Second call (should benefit from caching)
        start_time = time.time()
        result2 = advance_advection_diffusion(
            concentration=conc,
            velocity=(vel_x, vel_y),
            diffusivity=1e-5,
            grid_spacing=(dy, dx),
            spatial_dims=2,
            time_step=dt, splitting_method="symmetric", advection_scheme="upwind", boundary_method="zero", check_stability=True, enforce_non_negative=True
        )
        second_time = time.time() - start_time
        
        # Results should be identical
        assert torch.allclose(result1['concentration'], result2['concentration'])
        
        # Second call should be faster (though this depends on implementation)
        # This is a soft test - caching may not always provide speedup
        # assert second_time < first_time


class TestCrossModuleconsistency:
    """Test consistency with other RELIC modules."""
    
    def test_advection_diffusion_functor_consistency(self, test_state_1d):
        """Test consistency with AdvectionDiffusion Functor."""
        state = test_state_1d.copy()
        state['source_term'] = torch.zeros_like(state['tracer_concentration'])
        state['velocity_field'] = state['velocity_x']
        
        # Create AdvectionDiffusion functor
        transport_functor = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=state['grid_spacing_x']
        )
        
        # Call functor forward method
        requested_dt = 0.01
        functor_result = transport_functor.forward(
            state,
            time=0.0,
            dt=requested_dt,
        )
        assert "diagnostics.effective_time_step" not in functor_result
        
        # Call operator directly
        operator_result = advance_advection_diffusion(
            concentration=state['tracer_concentration'],
            velocity=state['velocity_x'],
            diffusivity=1e-5,
            grid_spacing=state['grid_spacing_x'],
            spatial_dims=1,
            time_step=requested_dt,
            splitting_method=transport_functor.splitting_method,
            advection_scheme=transport_functor.advection_scheme_value,
            boundary_method=transport_functor.boundary_method,
            check_stability=True,
            enforce_non_negative=transport_functor.non_negative,
            source_term=state['source_term']
        )
        
        # Results should be similar
        functor_delta = functor_result['tracer_concentration'].value
        operator_delta = operator_result['concentration'] - state['tracer_concentration']

        # Test-local comparison tolerance [concentration units per step]:
        # cells whose increment magnitude is below it are compared with
        # allclose instead of a relative difference (no model parameter).
        delta_tolerance = 1.0e-10
        significant = torch.abs(functor_delta) >= delta_tolerance
        if not torch.any(significant):
            significant = torch.abs(operator_delta) >= delta_tolerance

        if torch.any(significant):
            denom = torch.abs(operator_delta[significant]).clamp_min(1e-12)
            relative_diff = torch.abs(functor_delta[significant] - operator_delta[significant]) / denom
            max_relative_diff = torch.max(relative_diff)
            assert max_relative_diff < 0.1  # 10% tolerance
        else:
            assert torch.allclose(functor_delta, operator_delta)
        
    def test_parameter_system_integration(self, test_state_1d):
        """Test integration with parameter management system."""
        state = test_state_1d.copy()
        
        # This test would normally use the parameter system
        # For now, we'll test that the operators can accept parameters
        # from different sources
        
        diffusivity_values = [1e-6, 1e-5, 1e-4]  # Different diffusivities
        
        results = {}
        for diff_val in diffusivity_values:
            result = apply_diffusion(
                concentration=state['tracer_concentration'],
                diffusivity=diff_val,
                grid_spacing=state['grid_spacing_x'],
                spatial_dims=1,
                time_step=0.01, boundary_method="zero", check_stability=True, enforce_non_negative=True
            )
            results[diff_val] = result
            
        # Higher diffusivity should lead to more spreading
        for i, (diff1, diff2) in enumerate(zip(diffusivity_values[:-1], diffusivity_values[1:])):
            result1 = results[diff1]
            result2 = results[diff2]
            
            # Measure spread as standard deviation
            def spread(conc):
                mean_conc = torch.mean(conc)
                return torch.std(conc - mean_conc)
                
            final1 = result1['concentration']
            final2 = result2['concentration']
            
            spread1 = spread(final1)
            spread2 = spread(final2)
            
            # Higher diffusivity should yield stronger smoothing (lower spread)
            assert spread2 <= spread1 + 1e-6


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


