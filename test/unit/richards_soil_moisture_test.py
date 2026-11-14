"""
L1 Unit Tests for Richards Soil Moisture Module

This test suite validates that the RichardsSoilMoisture module correctly uses
the unified solver system instead of hardcoded implementations.

Test Coverage:
- Unified solver system integration
- Thomas algorithm solver usage
- Parameter system integration
- Mathematical correctness of Richards equation solution

Mathematical Expression:
    dtheta/dt = d/dz[K(theta)(dh/dz - 1)] - S
    where:
    - theta: volumetric water content [m3 m-3]
    - h: pressure head [m]
    - K(theta): hydraulic conductivity [m s-1]
    - S: sink term [s-1]
    - z: depth [m]

References:
    - Richards, L.A. (1931). Capillary conduction of liquids through porous
      mediums. Physics, 1(5), 318-333.
    - van Genuchten, M.Th. (1980). A closed-form equation for predicting
      the hydraulic conductivity of unsaturated soils. Soil Science Society
      of America Journal, 44(5), 892-898.
"""

import pytest
import torch

from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.computation.solvers import SolverFactory
from relic.computation.solvers.linear import ThomasAlgorithmSolver
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestRichardsSoilMoistureIntegration:
    """Test suite for Richards soil moisture module integration."""

    @pytest.fixture
    def sample_state(self):
        """Create sample state for Richards equation testing."""
        nz = 10
        return {
            "soil_moisture": torch.full((nz,), 0.3, dtype=torch.float32),
            "infiltration_flux": torch.tensor(0.0, dtype=torch.float32),
            "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
        }

    @pytest.fixture
    def richards_module(self):
        """Create Richards soil moisture module instance."""
        return RichardsSoilMoisture()

    def test_module_initialization(self, richards_module):
        """Test that Richards module initializes correctly."""
        assert richards_module is not None
        assert hasattr(richards_module, "get_parameter_value")
        assert hasattr(richards_module, "forward")

    def test_unified_solver_integration(self, richards_module):
        """Test that module uses unified solver system."""
        assert not hasattr(richards_module, "_solve_tridiagonal_thomas"), \
            "Module should not have hardcoded Thomas algorithm"
        assert isinstance(richards_module.thomas_solver, ThomasAlgorithmSolver)
        assert hasattr(richards_module.thomas_solver, "solve")

    def test_forward_method(self, richards_module, sample_state):
        """Test the forward method execution."""
        dt = 3600.0  # 1 hour time step [s]
        time = 0.0

        result = richards_module.forward(state=sample_state, time=time, dt=dt)

        assert isinstance(result, dict)
        assert "soil_moisture" in result
        assert (
            "diagnostics.soil_moisture_richards.water_flux_profile" in result
        )
        assert "soil_matric_potential" in result
        assert "diagnostics.soil_moisture_richards.relative_moisture" in result
        full_state = _unwrap_update(result["soil_moisture"])
        assert full_state.shape == sample_state["soil_moisture"].shape
        assert torch.isfinite(full_state).all()
        flux_profile = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.water_flux_profile"
            ]
        )
        assert flux_profile.shape == sample_state["soil_moisture"].shape
        assert torch.all(torch.isfinite(flux_profile))

        # References: van Genuchten (1980) and Celia et al. (1990).
        # Mathematical expression: psi_m = rho_w * g * h(theta).
        # Parameter meanings: h is pressure head [m], rho_w is water density
        # [kg m-3], g is gravitational acceleration [m s-2], and psi_m is
        # matric pressure [Pa]. This derived profile has no water source/sink.
        matric_potential = result["soil_matric_potential"]
        pressure_head = result[
            "diagnostics.soil_moisture_richards"
        ]["pressure_head"]
        expected_matric_potential = (
            pressure_head
            * get_physical_constant("gravity_acceleration")
            * get_physical_constant("water_density")
        )
        assert isinstance(matric_potential, UpdateItem)
        assert matric_potential.mode is UpdateMode.REPLACE
        assert matric_potential.per_timestep is False
        assert matric_potential.unit == "Pa"
        assert matric_potential.source_functor == richards_module.name
        torch.testing.assert_close(
            matric_potential.value,
            expected_matric_potential,
        )
        assert matric_potential.value.shape == full_state.shape
        assert "matric_potential" not in result[
            "diagnostics.soil_moisture_richards"
        ]

    def test_parameter_system_usage(self, richards_module):
        """Test that module uses parameter system correctly."""
        gravity = get_physical_constant("gravity_acceleration")
        water_density = get_physical_constant("water_density")
        assert gravity > 0
        assert water_density > 0
        assert 0 < richards_module.field_capacity_fraction <= 1
        assert richards_module.wilting_point_fraction > 0
        field_capacity = (
            richards_module.field_capacity_fraction * richards_module.theta_saturated
        )
        wilting_point = (
            richards_module.wilting_point_fraction * richards_module.theta_residual
        )
        assert richards_module.theta_residual < wilting_point < field_capacity
        assert richards_module.theta_residual < richards_module.theta_saturated

    def test_mass_conservation(self, richards_module, sample_state):
        """Test that Richards equation solution conserves mass."""
        result = richards_module.forward(state=sample_state, time=0.0, dt=3600.0)
        diagnostics = result["diagnostics.soil_moisture_richards"]
        mass_error = diagnostics["mass_conservation_error"]
        assert torch.isfinite(mass_error)
        assert mass_error >= 0
        assert mass_error < 0.1

    def test_boundary_conditions(self, richards_module, sample_state):
        """Test boundary condition handling."""
        # Test with different boundary conditions
        dry_state = dict(sample_state)
        infiltrated_state = dict(sample_state)
        infiltrated_state["infiltration_flux"] = torch.tensor(1e-4)

        dry_result = richards_module.forward(state=dry_state, time=0.0, dt=3600.0)
        infiltrated_result = richards_module.forward(
            state=infiltrated_state,
            time=0.0,
            dt=3600.0,
        )
        infiltrated_profile = _unwrap_update(infiltrated_result["soil_moisture"])
        dry_profile = _unwrap_update(dry_result["soil_moisture"])
        assert infiltrated_profile[0] > dry_profile[0]

    def test_stability_conditions(self, richards_module, sample_state):
        """Test numerical stability of the solution."""
        result_small = richards_module.forward(
            state=sample_state,
            time=0.0,
            dt=300.0,
        )
        result_large = richards_module.forward(
            state=sample_state,
            time=0.0,
            dt=7200.0,
        )

        for result in (result_small, result_large):
            moisture = _unwrap_update(result["soil_moisture"])
            assert torch.all(torch.isfinite(moisture))
            assert torch.all(moisture >= richards_module.theta_residual)
            assert torch.all(moisture <= richards_module.theta_saturated)

    def test_solver_availability(self):
        """Test that unified solver system provides required solvers."""
        thomas_solver = SolverFactory.create(
            solver_name="thomas_algorithm",
            name="richards_test_thomas",
        )
        assert isinstance(thomas_solver, ThomasAlgorithmSolver)
        assert hasattr(thomas_solver, "solve")

    def test_error_handling(self, richards_module):
        """Test error handling for invalid inputs."""
        # Test with invalid state
        invalid_state = {
            "soil_moisture": torch.full(
                (richards_module.soil_layers,), -0.1, dtype=torch.float32
            ),
            "infiltration_flux": torch.tensor(0.0, dtype=torch.float32),
            "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
        }
        with pytest.raises(ValueError, match=r"range \[0, 1\]"):
            richards_module.forward(state=invalid_state, time=0.0, dt=3600.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

