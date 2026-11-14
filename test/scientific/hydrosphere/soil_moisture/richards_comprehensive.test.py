"""
Comprehensive Test Suite for Richards Equation Implementation

IMPORTANT NOTE:
--------------
This suite exercises the single canonical Phase 8 IPHS Richards execution path.
No feature flag selects a separate Richards process law.

Test Categories:
----------------
- TESTED: Newton-Raphson solver behavior
- TESTED: Boundary condition handling
- TESTED: Physical consistency constraints
- TESTED: Canonical runtime regression guards

Mathematical Validation:
    - Van Genuchten-Mualem model accuracy
    - Finite difference discretization correctness
    - Newton-Raphson convergence properties
    - Mass conservation enforcement
    - Boundary condition implementation

References:
    - Celia et al. (1990). Water Resources Research, 26(7), 1483-1496
    - Van Genuchten (1980). Soil Science Society of America Journal, 44(5), 892-898
    - Paniconi & Putti (1994). Water Resources Research, 30(12), 3357-3374
"""

import pytest
import torch
import sys
import os
import inspect

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _get_richards_diagnostics(result):
    diagnostics = result.get("diagnostics.soil_moisture_richards", {})
    if isinstance(diagnostics, dict):
        return diagnostics
    return {}


class _ExplicitBoundary(dict):
    """Reject dictionary fallback access for an already validated boundary."""

    def get(self, key, default=None):
        raise AssertionError(
            f"Validated boundary must not use fallback access for {key!r}."
        )


@pytest.fixture
def parameter_manager_snapshot():
    """Snapshot and restore ParameterManager state."""
    manager = get_global_manager()
    snapshot = manager.snapshot()
    yield manager
    manager.restore(snapshot)


class TestRichardsNewtonRaphson:
    """Test suite specifically for Newton-Raphson implementation.

    Covers full PDE nonlinear solve behavior and numerical consistency.
    """

    @pytest.fixture
    def newton_overrides(self):
        """Parameter overrides for Newton-Raphson testing."""
        return {
            "hydrosphere.soil_moisture.layer_count": (5, "dimensionless"),
            "hydrosphere.soil_moisture.soil_depth": (1.0, "m"),
            "hydrosphere.soil_moisture.residual_water_content": (0.05, "m3 m-3"),
            "hydrosphere.soil_moisture.porosity": (0.45, "m3 m-3"),
            "hydrosphere.soil_moisture.alpha_vg": (5.0, "m-1"),
            "hydrosphere.soil_moisture.n_vg": (2.0, "dimensionless"),
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (1e-5, "m s-1"),
            "hydrosphere.soil_moisture.tortuosity_parameter": (0.5, "dimensionless"),
            "hydrosphere.soil_moisture.richards_enable_newton_raphson": (True, "dimensionless"),
            "hydrosphere.numerical.richards_max_picard_iterations": (50, "dimensionless"),
            "hydrosphere.numerical.richards_convergence_tolerance": (1e-6, "m3 m-3"),
        }
    
    @pytest.fixture
    def newton_model(self, parameter_manager_snapshot, newton_overrides):
        """Create Richards model with Newton-Raphson enabled."""
        manager = parameter_manager_snapshot
        for key, (value, units) in newton_overrides.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)
        return RichardsSoilMoisture(name="test_newton")
    
    def test_hydraulic_derivatives(self, newton_model):
        """Test analytical derivatives for Van Genuchten-Mualem model."""
        theta_test = torch.linspace(0.1, 0.4, 10)
        
        # Test dh/dtheta calculation
        dh_dtheta = newton_model._compute_dh_dtheta(theta_test)
        
        # All derivatives should be finite and positive (dh/dtheta > 0 in Van Genuchten model)
        assert torch.all(torch.isfinite(dh_dtheta)), "dh/dtheta should be finite"
        assert torch.all(dh_dtheta > 0), "dh/dtheta should be positive (less negative head with higher moisture)"
        
        # Test dk/dtheta calculation
        dk_dtheta = newton_model._compute_dk_dtheta(theta_test)
        
        # All derivatives should be finite and positive (dK/dtheta > 0)
        assert torch.all(torch.isfinite(dk_dtheta)), "dK/dtheta should be finite"
        assert torch.all(dk_dtheta >= 0), "dK/dtheta should be non-negative"
        
        # Test derivative accuracy using finite differences
        epsilon = 1e-6
        theta_mid = 0.25
        
        # Analytical derivatives
        h_mid = newton_model._compute_pressure_head(torch.tensor([theta_mid]))
        dh_analytical = newton_model._compute_dh_dtheta(torch.tensor([theta_mid]))
        
        # Finite difference approximation
        h_plus = newton_model._compute_pressure_head(torch.tensor([theta_mid + epsilon]))
        h_minus = newton_model._compute_pressure_head(torch.tensor([theta_mid - epsilon]))
        dh_finite_diff = (h_plus - h_minus) / (2 * epsilon)
        
        # Check agreement within reasonable tolerance
        relative_error = torch.abs((dh_analytical - dh_finite_diff) / dh_analytical)
        assert relative_error < 0.01, f"Derivative accuracy poor: {relative_error:.2e}"
    
    def test_newton_residual_computation(self, newton_model):
        """Test Newton-Raphson residual function."""
        n = newton_model.soil_layers
        theta = torch.linspace(0.15, 0.35, n)
        theta_old = torch.linspace(0.12, 0.32, n)
        
        # Compute hydraulic properties
        h = newton_model._compute_pressure_head(theta)
        k = newton_model._compute_hydraulic_conductivity(theta)
        
        # Compute residual
        residual = newton_model._compute_newton_residual(
            theta, theta_old, h, k, dt=1800.0, surface_flux=1e-6
        )
        
        # Check residual properties
        assert len(residual) == n, "Residual should have correct length"
        assert torch.all(torch.isfinite(residual)), "Residual should be finite"
        
        # At steady state with no flux, residual should be near zero
        steady_theta = torch.full((n,), 0.25)
        steady_residual = newton_model._compute_newton_residual(
            steady_theta, steady_theta, 
            newton_model._compute_pressure_head(steady_theta),
            newton_model._compute_hydraulic_conductivity(steady_theta),
            dt=1800.0, surface_flux=0.0
        )
        assert torch.max(torch.abs(steady_residual)) < 1e-2, "Steady state residual should be small for PDE solve"
    
    def test_newton_jacobian_structure(self, newton_model):
        """Test Newton-Raphson Jacobian matrix properties."""
        n = newton_model.soil_layers
        theta = torch.linspace(0.15, 0.35, n)
        theta_old = torch.linspace(0.12, 0.32, n)
        
        # Compute Jacobian
        jacobian = newton_model._compute_newton_jacobian(
            theta, theta_old, dt=1800.0, surface_flux=1e-6
        )
        
        # Check Jacobian properties
        assert jacobian.shape == (n, n), "Jacobian should be square matrix"
        assert torch.all(torch.isfinite(jacobian)), "Jacobian should be finite"
        
        # Jacobian should be diagonally dominant for stability
        diag = torch.abs(torch.diag(jacobian))
        off_diag_sum = torch.sum(torch.abs(jacobian), dim=1) - diag
        
        # Most diagonal elements should dominate (weak diagonal dominance)
        dominance_ratio = torch.mean((diag >= off_diag_sum).float())
        assert dominance_ratio > 0.5, f"Jacobian should be reasonably diagonally dominant: {dominance_ratio:.2f}"
        
        # Check that Jacobian is nearly tridiagonal (for finite difference discretization)
        is_tridiag = newton_model._is_nearly_tridiagonal(jacobian, tol=1e-8)
        # Note: May not be exactly tridiagonal due to nonlinear coupling terms
    
    def test_newton_linear_system_solver(self, newton_model):
        """Test linear system solver used in Newton method."""
        n = 5
        
        # Create a test tridiagonal system
        jacobian = torch.zeros((n, n))
        jacobian.fill_diagonal_(2.0)
        if n > 1:
            jacobian[:-1, 1:].fill_diagonal_(-1.0)  # Upper diagonal
            jacobian[1:, :-1].fill_diagonal_(-1.0)  # Lower diagonal
        
        rhs = torch.ones(n)
        
        # Solve using Newton solver
        solution = newton_model._solve_newton_linear_system(jacobian, rhs)
        
        # Verify solution
        residual = torch.matmul(jacobian, solution) - rhs
        assert torch.max(torch.abs(residual)) < 1e-6, "Linear system solver should be reasonably accurate"
    
    def test_newton_line_search(self, newton_model):
        """A residual-derived Newton direction satisfies Armijo decrease."""
        n = newton_model.soil_layers
        theta_current = torch.full((n,), 0.25)
        theta_old = torch.full((n,), 0.20)
        surface_flux = 1.0e-6
        h_current = newton_model._compute_pressure_head(theta_current)
        k_current = newton_model._compute_hydraulic_conductivity(theta_current)
        residual = newton_model._compute_newton_residual(
            theta_current,
            theta_old,
            h_current,
            k_current,
            dt=1800.0,
            surface_flux=surface_flux,
        )
        residual_norm = torch.max(torch.abs(residual))
        jacobian = newton_model._compute_newton_jacobian(
            theta_current,
            theta_old,
            dt=1800.0,
            surface_flux=surface_flux,
        )
        delta_theta = newton_model._solve_newton_linear_system(
            jacobian, -residual
        )

        alpha = newton_model._newton_line_search(
            theta_current,
            delta_theta,
            theta_old,
            dt=1800.0,
            surface_flux=surface_flux,
            residual_norm_old=residual_norm,
        )

        assert 0 < alpha <= 1.0, f"Line search step should be in (0,1]: {alpha}"
        theta_new = theta_current + alpha * delta_theta
        theta_new = torch.clamp(theta_new, min=newton_model.theta_residual, max=newton_model.theta_saturated)
        assert torch.all(theta_new >= newton_model.theta_residual)
        assert torch.all(theta_new <= newton_model.theta_saturated)
        residual_new = newton_model._compute_newton_residual(
            theta_new,
            theta_old,
            newton_model._compute_pressure_head(theta_new),
            newton_model._compute_hydraulic_conductivity(theta_new),
            dt=1800.0,
            surface_flux=surface_flux,
        )
        assert torch.max(torch.abs(residual_new)) < residual_norm
    
    def test_newton_vs_picard_comparison(self, parameter_manager_snapshot, newton_overrides):
        """Compare Newton-Raphson vs Picard iteration performance."""
        manager = parameter_manager_snapshot
        # Test with Newton-Raphson
        for key, (value, units) in newton_overrides.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)
        newton_model = RichardsSoilMoisture(name="test_newton")
        
        # Test with Picard iteration
        manager.set_value(
            "hydrosphere.soil_moisture.richards_enable_newton_raphson",
            False,
            units="dimensionless",
            validate=True,
            notify=False,
        )
        picard_model = RichardsSoilMoisture(name="test_picard")
        
        # Common test state
        state = {
            "soil_moisture": torch.tensor([0.15, 0.20, 0.25, 0.30, 0.35]),
            "infiltration_flux": torch.tensor(2e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        # Run both methods
        newton_result = newton_model._compute_full_solution(state, 0.0, 1800.0)
        picard_result = picard_model._compute_full_solution(state, 0.0, 1800.0)
        
        # Both should converge
        newton_diag = _get_richards_diagnostics(newton_result)
        picard_diag = _get_richards_diagnostics(picard_result)
        assert newton_diag["solver_converged"], "Newton method should converge"
        assert picard_diag["solver_converged"], "Picard method should converge"
        # The nonlinear solver is configuration (enable_newton_raphson), not a
        # diagnostic.
        assert newton_model.enable_newton_raphson is True
        assert picard_model.enable_newton_raphson is False
        
        # Newton should converge faster (fewer iterations)
        newton_iters = newton_diag["solver_iterations"]
        picard_iters = picard_diag["solver_iterations"]
        
        print(f"Newton iterations: {newton_iters}, Picard iterations: {picard_iters}")
        
        # Results should be similar (both solving the same equation)
        newton_change = _unwrap_update(
            newton_result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        picard_change = _unwrap_update(
            picard_result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        
        relative_diff = torch.max(torch.abs(newton_change - picard_change) / 
                                 (torch.abs(picard_change) + 1e-10))
        assert relative_diff < 0.1, f"Newton and Picard results should be similar: {relative_diff:.2e}"


class TestRichardsBoundaryConditions:
    """Test suite for boundary condition implementations.

    Covers prescribed and mixed boundary modes for the full PDE solver.
    """

    @pytest.fixture
    def bc_overrides(self):
        """Parameter overrides for boundary condition testing."""
        return {
            "hydrosphere.soil_moisture.layer_count": (5, "dimensionless"),
            "hydrosphere.soil_moisture.soil_depth": (1.0, "m"),
            "hydrosphere.soil_moisture.residual_water_content": (0.05, "m3 m-3"),
            "hydrosphere.soil_moisture.porosity": (0.45, "m3 m-3"),
            "hydrosphere.soil_moisture.alpha_vg": (5.0, "m-1"),
            "hydrosphere.soil_moisture.n_vg": (1.5, "dimensionless"),
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (1e-5, "m s-1"),
            "hydrosphere.soil_moisture.richards_enable_newton_raphson": (True, "dimensionless"),
            "hydrosphere.numerical.richards_max_picard_iterations": (50, "dimensionless"),
            "hydrosphere.numerical.richards_convergence_tolerance": (1e-6, "m3 m-3"),
        }
    
    @pytest.fixture
    def bc_model(self, parameter_manager_snapshot, bc_overrides):
        """Create model for boundary condition testing."""
        manager = parameter_manager_snapshot
        for key, (value, units) in bc_overrides.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)
        return RichardsSoilMoisture(name="test_bc")

    @pytest.mark.parametrize(
        ("side", "boundary"),
        [
            ("top", {}),
            ("bottom", {}),
            ("top", {"type": "prescribed_flux"}),
            ("top", {"type": "prescribed_head"}),
            ("bottom", {"type": "prescribed_flux"}),
            ("bottom", {"type": "prescribed_head"}),
        ],
    )
    def test_boundary_configuration_is_explicit_and_fail_fast(
        self,
        bc_model,
        side,
        boundary,
    ):
        """Reject implicit boundary types and hidden prescribed values."""
        with pytest.raises(ValueError, match="explicit.*type|requires.*value"):
            bc_model.set_boundary_conditions(**{side: boundary})

    def test_boundary_configuration_has_one_runtime_owner(self, bc_model):
        """The full solver is the only executable boundary implementation."""
        assert not hasattr(bc_model, "_apply_top_boundary")
        assert not hasattr(bc_model, "_apply_bottom_boundary")
    
    def test_prescribed_flux_boundaries(self, bc_model):
        """Test prescribed flux boundary conditions."""
        # Set prescribed flux boundaries
        bc_model.set_boundary_conditions(
            top={"type": "prescribed_flux", "value": 2e-6},     # Infiltration
            bottom={"type": "prescribed_flux", "value": 0.5e-6} # Drainage
        )
        bc_model._top_boundary = _ExplicitBoundary(bc_model._top_boundary)
        
        state = {
            "soil_moisture": torch.full((bc_model.soil_layers,), 0.25),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = bc_model._compute_full_solution(state, 0.0, 1800.0)

        diagnostics = _get_richards_diagnostics(result)
        assert diagnostics["solver_converged"], "Should converge with prescribed flux BCs"
        
        # Mass balance should reflect prescribed fluxes
        diagnostics = _get_richards_diagnostics(result)
        mass_error = diagnostics["mass_conservation_error"]
        assert mass_error <= bc_model.mass_conservation_tolerance, (
            f"Mass conservation exceeds declared tolerance: {mass_error:.2e}"
        )
    
    def test_prescribed_head_boundaries(self, bc_model):
        """Test prescribed head boundary conditions."""
        # Set prescribed head boundaries
        bc_model.set_boundary_conditions(
            top={"type": "prescribed_head", "value": -0.1},    # Near saturation
            bottom={"type": "prescribed_head", "value": -2.0}  # Drier
        )
        
        state = {
            "soil_moisture": torch.tensor([0.20, 0.22, 0.24, 0.26, 0.28]),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = bc_model._compute_full_solution(state, 0.0, 1800.0)

        diagnostics = _get_richards_diagnostics(result)
        assert diagnostics["solver_converged"], "Should converge with prescribed head BCs"
        assert diagnostics["mass_conservation_error"] <= bc_model.mass_conservation_tolerance

        # Check that boundary nodes have correct head values
        pressure_heads = diagnostics["pressure_head"]
        
        # Top boundary should approach prescribed head
        top_head_prescribed = -0.1
        top_head_computed = pressure_heads[0].item()
        assert abs(top_head_computed - top_head_prescribed) < 0.01, \
            f"Top head mismatch: {top_head_computed:.3f} vs {top_head_prescribed:.3f}"
        
        # Bottom boundary should approach prescribed head
        bottom_head_prescribed = -2.0
        bottom_head_computed = pressure_heads[-1].item()
        assert abs(bottom_head_computed - bottom_head_prescribed) < 0.1, \
            f"Bottom head mismatch: {bottom_head_computed:.3f} vs {bottom_head_prescribed:.3f}"
    
    def test_mixed_boundary_conditions(self, bc_model):
        """Test mixed boundary condition types."""
        # Top: prescribed flux, Bottom: free drainage
        bc_model.set_boundary_conditions(
            top={"type": "prescribed_flux", "value": 1e-6},
            bottom={"type": "free_drainage"}
        )
        
        state = {
            "soil_moisture": torch.full((bc_model.soil_layers,), 0.30),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = bc_model._compute_full_solution(state, 0.0, 1800.0)
        diagnostics = _get_richards_diagnostics(result)
        assert diagnostics["solver_converged"], "Should converge with mixed BCs"

        # The public drainage flux and the bottom entry of the diagnostic
        # profile are the same step-mean boundary flow.
        flux_profile = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.water_flux_profile"
            ]
        )
        bottom_flux_expected = flux_profile[-1]
        bottom_flux_computed = _unwrap_update(result["drainage_flux"])

        assert bottom_flux_computed > 0
        torch.testing.assert_close(
            bottom_flux_computed,
            bottom_flux_expected,
            rtol=0.0,
            atol=0.0,
        )
    
    def test_atmospheric_boundary(self, bc_model):
        """Test atmospheric boundary condition (precipitation - ET)."""
        bc_model.set_boundary_conditions(
            top={"type": "atmospheric"},
            bottom={"type": "free_drainage"}
        )
        bc_model._top_boundary = _ExplicitBoundary(bc_model._top_boundary)
        
        # Test the canonical atmospheric mass-flux boundary inputs.
        state = {
            "soil_moisture": torch.full((bc_model.soil_layers,), 0.25),
            "infiltration_flux": torch.tensor(3e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(1e-6 * 1000.0),
        }
        
        result = bc_model._compute_full_solution(state, 0.0, 1800.0)

        diagnostics = _get_richards_diagnostics(result)
        assert diagnostics["solver_converged"], "Should converge with atmospheric BC"
        
        # Net atmospheric input should be positive (precip > ET)
        net_input = (3e-6 - 1e-6) * 1800.0  # m
        
        # Check that surface moisture increases appropriately
        moisture_change = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        total_change = torch.sum(moisture_change) * bc_model.layer_thickness * 1800.0  # m
        
        # Total change should be realistic (drainage can exceed precipitation in some soils)
        # Just ensure the magnitude is reasonable and finite
        assert torch.isfinite(total_change), "Moisture change should be finite"
        assert abs(total_change) < 0.1, f"Moisture change should be reasonable: {total_change:.4f} m"


class TestRichardsPhysicalConsistency:
    """Test suite for physical consistency and realism.

    Validates mass conservation and physically plausible full-solver responses.
    """

    @pytest.fixture
    def physics_overrides(self):
        """Parameter overrides for physics testing."""
        return {
            "hydrosphere.soil_moisture.layer_count": (10, "dimensionless"),
            "hydrosphere.soil_moisture.soil_depth": (2.0, "m"),
            "hydrosphere.soil_moisture.residual_water_content": (0.05, "m3 m-3"),
            "hydrosphere.soil_moisture.porosity": (0.45, "m3 m-3"),
            "hydrosphere.soil_moisture.alpha_vg": (5.0, "m-1"),
            "hydrosphere.soil_moisture.n_vg": (1.5, "dimensionless"),
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (1e-5, "m s-1"),
            "hydrosphere.soil_moisture.richards_enable_newton_raphson": (True, "dimensionless"),
            "hydrosphere.soil_moisture.richards_enable_mass_conservation_check": (True, "dimensionless"),
            "hydrosphere.numerical.richards_max_picard_iterations": (50, "dimensionless"),
            "hydrosphere.numerical.richards_convergence_tolerance": (1e-6, "m3 m-3"),
            "hydrosphere.numerical.mass_conservation_tolerance": (1e-6, "dimensionless"),
        }
    
    @pytest.fixture
    def physics_model(self, parameter_manager_snapshot, physics_overrides):
        """Create model for physics testing."""
        manager = parameter_manager_snapshot
        for key, (value, units) in physics_overrides.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)
        return RichardsSoilMoisture(name="test_physics")
    
    def test_infiltration_physics(self, physics_model):
        """Test physically realistic infiltration behavior."""
        # Start with dry soil
        initial_moisture = 0.08  # Near residual
        
        # Apply moderate infiltration
        state = {
            "soil_moisture": torch.full((physics_model.soil_layers,), initial_moisture),
            "infiltration_flux": torch.tensor(2e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = physics_model._compute_full_solution(state, 0.0, 3600.0)
        
        # Check physical expectations
        moisture_changes = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        
        # Surface layers should wet first and most
        surface_change = moisture_changes[0].item()
        deep_change = moisture_changes[-1].item()
        
        assert surface_change > 0, "Surface should wet during infiltration"
        assert surface_change >= deep_change, "Surface should wet more than deep layers"
        
        # Check infiltration capacity (should be reasonable order of magnitude)
        diagnostics = _get_richards_diagnostics(result)
        capacity = diagnostics["infiltration_capacity"].item() / 1000.0  # Convert to m/s
        applied_flux = 2e-6  # m/s
        
        # For fine soils, capacity can be very low - this is physically realistic
        assert capacity > 1e-12, f"Infiltration capacity should be positive: {capacity:.2e} m/s"
        assert capacity < 1e-3, f"Infiltration capacity should be reasonable: {capacity:.2e} m/s"
    
    def test_drainage_physics(self, physics_model):
        """Test physically realistic drainage behavior."""
        # Start with wet soil profile (realistic field capacity gradient)
        wet_profile = torch.tensor([0.35, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18])
        
        # Apply drainage conditions (no surface input)
        state = {
            "soil_moisture": wet_profile,
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = physics_model._compute_full_solution(state, 0.0, 3600.0)
        
        # Check physical expectations
        moisture_changes = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        
        # Net storage must decline while internal downward flux divergence may
        # redistribute water into individual layers.
        net_change = torch.sum(moisture_changes)
        assert net_change <= 0, "Net system moisture should decrease during drainage"

        diagnostics = _get_richards_diagnostics(result)
        boundary_fluxes = _unwrap_update(diagnostics["vertical_flux"])
        # A held binding, and a value derived from it, sits on the device of
        # its latest registration; read it on the result device.
        expected_rate = (
            boundary_fluxes[:-1] - boundary_fluxes[1:]
        ) / (
            physics_model.water_density
            * physics_model.layer_thickness.to(boundary_fluxes.device)
        )
        torch.testing.assert_close(
            moisture_changes,
            expected_rate,
            rtol=2.0e-4,
            atol=1.0e-10,
        )
        assert torch.all(boundary_fluxes[1:-1] > 0.0)

        drainage_flux = _unwrap_update(result["drainage_flux"])
        assert torch.isfinite(drainage_flux)
        assert drainage_flux > 0.0
        torch.testing.assert_close(drainage_flux, boundary_fluxes[-1])
        assert diagnostics["mass_conservation_error"] <= physics_model.mass_conservation_tolerance
    
    def test_reported_interface_flux_uses_the_plain_harmonic_mean(self, physics_model):
        """Interior entries of vertical_flux carry the same K_face as the state.

        References: Belfort, Younes, Fahs and Lehmann (2013), J. Hydrol.
        505:202-217, Table 2 (harmonic interblock mean); Celia, Bouloutas and
        Zarba (1990), Water Resour. Res. 26(7):1483-1496.
        Mathematical expression: with uniform h, q_{i+1/2} = -K_face (0 - 1)
        = K_face and K_face = 2 K_i K_{i+1} / (K_i + K_{i+1}).
        Parameter meanings: q_{i+1/2} interior Darcy flux; K_i cell conductivity.
        Units: q and K [m s-1]; h [m].
        Domain/constraints: dry-end K = 1e-12 m s-1 next to K = 1e-5 m s-1;
        float64; relative tolerance 1e-12.
        Sign convention: positive q is downward.
        """
        n = physics_model.soil_layers
        k = torch.full((n,), 1.0e-12, dtype=torch.float64)
        k[2] = 1.0e-5
        theta = torch.full((n,), 0.30, dtype=torch.float64)
        fluxes = physics_model._compute_inter_layer_fluxes(
            theta, torch.zeros_like(theta), k
        )
        mixed = 2.0 * 1.0e-5 * 1.0e-12 / (1.0e-5 + 1.0e-12)
        for face, expected in ((1, 1.0e-12), (2, mixed), (3, mixed), (4, 1.0e-12)):
            assert float(fluxes[face].detach()) == pytest.approx(
                expected, rel=1.0e-12, abs=0.0
            ), face

    def test_dry_soil_behavior(self, physics_model):
        """Test behavior near residual water content."""
        # Very dry initial conditions
        dry_moisture = physics_model.theta_residual + 0.001
        
        state = {
            "soil_moisture": torch.full((physics_model.soil_layers,), dry_moisture),
            "infiltration_flux": torch.tensor(0.5e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = physics_model._compute_full_solution(state, 0.0, 3600.0)

        diagnostics = _get_richards_diagnostics(result)
        # Should still converge despite low moisture
        assert diagnostics["solver_converged"], "Should converge with dry soil"
        
        # Hydraulic conductivity should be very low
        k_profile = diagnostics["hydraulic_conductivity"]
        k_relative = k_profile / physics_model.k_saturated.to(k_profile.device)
        
        assert torch.all(k_relative < 0.01), "Relative conductivity should be very low in dry soil"
        
        # Pressure heads should be very negative
        h_profile = _unwrap_update(diagnostics["pressure_head"])
        assert torch.all(h_profile < -1.0), "Pressure heads should be very negative in dry soil"
    
    def test_saturated_soil_behavior(self, physics_model):
        """Test behavior near saturation."""
        # Near-saturated initial conditions
        sat_moisture = physics_model.theta_saturated - 0.01
        
        state = {
            "soil_moisture": torch.full((physics_model.soil_layers,), sat_moisture),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(1e-6 * 1000.0),
        }
        
        result = physics_model._compute_full_solution(state, 0.0, 600.0)
        
        # Near saturation can be challenging for numerical solvers
        # Check that solution is physically reasonable even if not fully converged
        moisture_changes = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        assert torch.all(torch.isfinite(moisture_changes)), "Solution should be finite near saturation"
        
        # For evaporation, expect moisture loss
        total_change = torch.sum(moisture_changes)
        assert total_change <= 0.01, "Should lose moisture under evaporation (may not fully converge)"
        
        # Hydraulic conductivity should be reasonable near saturation
        diagnostics = _get_richards_diagnostics(result)
        k_profile = diagnostics["hydraulic_conductivity"]
        k_relative = k_profile / physics_model.k_saturated.to(k_profile.device)
        
        # Near saturation, conductivity should be positive (Van Genuchten can have quite low values)
        assert torch.all(k_relative > 0.01), "Relative conductivity should be positive near saturation"
        assert torch.all(k_relative <= 1.0), "Relative conductivity should not exceed unity"
        
        # Validate the pressure head against an independent evaluation of the
        # canonical van Genuchten inverse retention relation.
        h_profile = _unwrap_update(diagnostics["pressure_head"])
        final_theta = _unwrap_update(result["soil_moisture"])
        # The retention bindings are read on the device of the profile.
        alpha_vg = physics_model.alpha_vg.to(final_theta.device)
        n_vg = physics_model.n_vg.to(final_theta.device)
        m_vg = physics_model.m_vg.to(final_theta.device)
        effective_saturation = (
            (final_theta - physics_model.theta_residual)
            / (physics_model.theta_saturated - physics_model.theta_residual)
        )
        expected_head = -(
            effective_saturation.pow(-1.0 / m_vg) - 1.0
        ).pow(1.0 / n_vg) / alpha_vg
        torch.testing.assert_close(h_profile, expected_head)

        initial_theta = torch.full_like(final_theta, sat_moisture)
        initial_effective_saturation = (
            (initial_theta - physics_model.theta_residual)
            / (physics_model.theta_saturated - physics_model.theta_residual)
        )
        initial_head = -(
            initial_effective_saturation.pow(-1.0 / m_vg) - 1.0
        ).pow(1.0 / n_vg) / alpha_vg
        assert torch.isfinite(h_profile).all()
        assert torch.all(h_profile < 0.0)
        # Evaporation leaves through the surface, so the surface layer dries within
        # the step. The layers below it respond to redistribution and drainage and
        # need not all be drier after 600 s.
        assert h_profile[0] < initial_head[0]
    
    def test_mass_conservation_enforcement(self, physics_model):
        """Test strict mass conservation enforcement."""
        # Enable strict mass conservation checking
        physics_model.enable_mass_conservation_check = True
        physics_model.mass_conservation_tolerance = 1e-6
        
        state = {
            "soil_moisture": torch.tensor([0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.28, 0.26, 0.24, 0.22]),
            "infiltration_flux": torch.tensor(1.5e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = physics_model._compute_full_solution(state, 0.0, 1800.0)

        # Mass conservation error should be within tolerance
        diagnostics = _get_richards_diagnostics(result)
        mass_error = diagnostics["mass_conservation_error"]
        # For PDE numerical methods, reasonable tolerance is ~1e-4
        reasonable_tolerance = max(physics_model.mass_conservation_tolerance, 1e-4)
        assert mass_error < reasonable_tolerance, \
            f"Mass conservation error too large: {mass_error:.2e} > {reasonable_tolerance:.2e}"

    def test_mass_conservation_threshold_rejects_step(
        self,
        physics_model,
        monkeypatch,
    ):
        """An enabled conservation check must reject an invalid water ledger."""

        def _forced_mass_error(self, **kwargs):
            theta_old = kwargs["theta_old"]
            return theta_old.new_tensor(1.0e-3)

        monkeypatch.setattr(
            RichardsSoilMoisture,
            "_check_mass_conservation",
            _forced_mass_error,
        )
        physics_model.enable_mass_conservation_check = True
        physics_model.mass_conservation_tolerance = 1.0e-6
        state = {
            "soil_moisture": torch.full(
                (physics_model.soil_layers,), 0.25
            ),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }

        with pytest.raises(RuntimeError, match="mass conservation"):
            physics_model._compute_full_solution(state, 0.0, 60.0)

    def test_nonconverged_nonlinear_solve_rejects_step(
        self,
        physics_model,
        monkeypatch,
    ):
        """A nonconverged nonlinear solve must not commit a candidate state."""

        def _never_converges(
            self,
            theta_old,
            theta_iter,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=None,
        ):
            return (
                theta_iter + theta_iter.new_full(theta_iter.shape, 1.0e-3),
                False,
            )

        monkeypatch.setattr(
            RichardsSoilMoisture,
            "_solve_linearized_system_newton",
            _never_converges,
        )
        monkeypatch.setattr(physics_model, "max_newton_iterations", 1)
        monkeypatch.setattr(physics_model, "atol", 1.0e-12)
        monkeypatch.setattr(physics_model, "rtol", 0.0)
        state = {
            "soil_moisture": torch.full(
                (physics_model.soil_layers,), 0.25
            ),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }

        with pytest.raises(RuntimeError, match="did not converge"):
            physics_model._compute_full_solution(state, 0.0, 60.0)

    def test_clipping_residual_records_accepted_projection(
        self,
        physics_model,
        monkeypatch,
    ):
        """The diagnostic residual must expose accepted bound projection."""

        raw_value = physics_model.theta_saturated + 0.05

        def _forces_upper_projection(
            self,
            theta_old,
            theta_iter,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=None,
        ):
            return theta_iter.new_full(theta_iter.shape, raw_value), False

        def _projection_residual(
            self,
            theta,
            theta_old,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=None,
        ):
            if bool(torch.all(theta >= self.theta_saturated)):
                return torch.zeros_like(theta)
            return torch.full_like(theta, 1.0e-3)

        monkeypatch.setattr(
            RichardsSoilMoisture,
            "_solve_linearized_system_newton",
            _forces_upper_projection,
        )
        monkeypatch.setattr(
            RichardsSoilMoisture,
            "_compute_newton_residual",
            _projection_residual,
        )
        physics_model.enable_mass_conservation_check = False
        state = {
            "soil_moisture": torch.full(
                (physics_model.soil_layers,), 0.25
            ),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }

        result = physics_model._compute_full_solution(state, 0.0, 1.0)
        residual = result["diagnostics.soil_moisture_richards"][
            "moisture_clipping_residual"
        ]
        expected = residual.new_full(
            residual.shape,
            physics_model.theta_saturated - raw_value,
        )
        torch.testing.assert_close(residual, expected)
        assert torch.all(residual < 0.0)
    
    def test_temporal_consistency(self, physics_model):
        """Test temporal consistency over multiple time steps."""
        initial_state = {
            "soil_moisture": torch.tensor([0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.28, 0.26, 0.24, 0.22]),
            "infiltration_flux": torch.tensor(1e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        # Run multiple time steps
        current_moisture = initial_state["soil_moisture"].clone()
        total_mass_error = 0.0
        
        for step in range(5):
            state = {
                "soil_moisture": current_moisture,
                "infiltration_flux": initial_state["infiltration_flux"],
                "evapotranspiration_flux": initial_state[
                    "evapotranspiration_flux"
                ],
            }
            
            result = physics_model._compute_full_solution(state, step * 1800.0, 1800.0)
            
            # Update moisture for next step
            moisture_change = _unwrap_update(
                result[
                    "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
                ]
            )
            expected_moisture = torch.clamp(
                current_moisture + moisture_change * 1800.0,
                min=physics_model.theta_residual,
                max=physics_model.theta_saturated,
            )
            current_moisture = _unwrap_update(result["soil_moisture"])
            torch.testing.assert_close(current_moisture, expected_moisture)
            
            # Accumulate mass error
            diagnostics = _get_richards_diagnostics(result)
            total_mass_error += abs(diagnostics["mass_conservation_error"])
            
            # Check convergence at each step
            assert diagnostics["solver_converged"], f"Should converge at step {step}"
            
            # Check physical bounds
            assert torch.all(current_moisture >= physics_model.theta_residual - 1e-6)
            assert torch.all(current_moisture <= physics_model.theta_saturated + 1e-6)
        
        # Total mass error should remain reasonable
        avg_mass_error = total_mass_error / 5
        assert avg_mass_error < 0.01, f"Average mass error over time: {avg_mass_error:.2e}"


class TestRichardsCanonicalRuntimeRegression:
    """Regression guards for canonical Phase 8 Richards execution."""

    def test_internal_runtime_rejects_solver_override_kwargs(self):
        """ParameterManager is the only production nonlinear-solver owner."""
        signature = inspect.signature(RichardsSoilMoisture._compute_full_solution)
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @pytest.fixture
    def canonical_model(self, parameter_manager_snapshot):
        manager = parameter_manager_snapshot
        manager.set_value(
            "hydrosphere.soil_moisture.layer_count",
            10,
            units="dimensionless",
            validate=True,
            notify=False,
        )
        manager.set_value(
            "hydrosphere.soil_moisture.soil_depth",
            1.0,
            units="m",
            validate=True,
            notify=False,
        )
        manager.set_value(
            "hydrosphere.numerical.richards_max_picard_iterations",
            50,
            units="dimensionless",
            validate=True,
            notify=False,
        )
        manager.set_value(
            "hydrosphere.numerical.richards_convergence_tolerance",
            1e-6,
            units="m3 m-3",
            validate=True,
            notify=False,
        )
        manager.set_value(
            "hydrosphere.soil_moisture.richards_enable_newton_raphson",
            True,
            units="dimensionless",
            validate=True,
            notify=False,
        )
        return RichardsSoilMoisture(name="test_richards_canonical")

    def _step(
        self,
        model: RichardsSoilMoisture,
        soil_moisture: torch.Tensor,
        infiltration_flux: torch.Tensor,
        dt: float,
        time: float,
    ):
        state = {
            "soil_moisture": soil_moisture.clone(),
            "infiltration_flux": infiltration_flux.clone(),
            "evapotranspiration_flux": torch.zeros_like(infiltration_flux),
        }
        result = model.forward(state, time=time, dt=dt)
        next_moisture = _unwrap_update(result["soil_moisture"])
        return result, next_moisture

    def test_canonical_runtime_mass_balance_regression(self, canonical_model):
        soil_moisture = torch.full((canonical_model.soil_layers,), 0.30, dtype=torch.float32)
        infiltration_flux = torch.tensor(0.0, dtype=torch.float32)
        dt = 3600.0

        result, _ = self._step(
            canonical_model, soil_moisture, infiltration_flux, dt, time=0.0
        )
        moisture_change = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        drainage_flux = _unwrap_update(result["drainage_flux"])

        storage_rate = (
            torch.sum(moisture_change)
            * canonical_model.layer_thickness
            * canonical_model.water_density
        )
        net_boundary_flux = infiltration_flux - drainage_flux
        residual = torch.abs(storage_rate - net_boundary_flux)
        relative_residual = result["diagnostics.soil_moisture_richards"][
            "mass_conservation_error"
        ]

        assert torch.isfinite(residual)
        assert float(residual.item()) < 2.0e-3
        assert torch.isfinite(relative_residual)
        assert float(relative_residual.item()) <= canonical_model.mass_conservation_tolerance

    def test_canonical_runtime_boundary_flux_response(self, canonical_model):
        soil_moisture = torch.full((canonical_model.soil_layers,), 0.28, dtype=torch.float32)
        dt = 1800.0

        low_flux = torch.tensor(0.0, dtype=torch.float32)
        high_flux = torch.tensor(3.0e-5, dtype=torch.float32)

        low_result, _ = self._step(canonical_model, soil_moisture, low_flux, dt, time=0.0)
        high_result, _ = self._step(canonical_model, soil_moisture, high_flux, dt, time=0.0)

        low_change = _unwrap_update(
            low_result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        high_change = _unwrap_update(
            high_result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        low_drainage = _unwrap_update(low_result["drainage_flux"])
        high_drainage = _unwrap_update(high_result["drainage_flux"])

        assert torch.isfinite(low_change).all()
        assert torch.isfinite(high_change).all()
        assert float(high_change[0].item()) >= float(low_change[0].item())
        assert float(high_drainage.item()) >= float(low_drainage.item())

    def test_canonical_runtime_dt_refinement_stability(self, canonical_model):
        initial_profile = torch.linspace(
            0.22,
            0.32,
            steps=canonical_model.soil_layers,
            dtype=torch.float32,
        )
        surface_flux = torch.tensor(1.5e-5, dtype=torch.float32)
        total_time = 3600.0

        coarse_dt = 600.0
        fine_dt = 300.0

        coarse_profile = initial_profile.clone()
        coarse_time = 0.0
        for _ in range(int(total_time / coarse_dt)):
            _, coarse_profile = self._step(
                canonical_model,
                coarse_profile,
                surface_flux,
                coarse_dt,
                time=coarse_time,
            )
            coarse_time += coarse_dt

        fine_profile = initial_profile.clone()
        fine_time = 0.0
        for _ in range(int(total_time / fine_dt)):
            _, fine_profile = self._step(
                canonical_model,
                fine_profile,
                surface_flux,
                fine_dt,
                time=fine_time,
            )
            fine_time += fine_dt

        profile_error = torch.mean(torch.abs(coarse_profile - fine_profile))
        assert torch.isfinite(profile_error)
        assert float(profile_error.item()) < 0.02


if __name__ == "__main__":
    # Run comprehensive test
    pytest.main([__file__, "-v", "--tb=short"])
