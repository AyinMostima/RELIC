"""
L1 Unit Tests for Linear Advection Transport Module

This module provides comprehensive unit tests for the LinearAdvection class
and related transport mechanisms, ensuring proper functionality, parameter
integration, CFL stability, and conservation properties.

Test Coverage:
- LinearAdvection class initialization and configuration
- Parameter system integration and validation
- Velocity field setup and management
- CFL condition computation and stability checking
- Upwind numerical scheme implementation
- Conservation law verification
- Boundary condition handling
- Forward method execution and error handling
- Multi-dimensional transport (1D, 2D, 3D)
- Flux computation methods

Mathematical Verification:
- CFL condition: CFL = |v| * dt / dx <= 1.0
- Conservation: dc/dt + d(vc)/dx = S
- Upwind scheme: d(uc)/dx = u[i] * (c[i] - c_upwind) / dx

References:
- LeVeque, R.J. (1992). Numerical Methods for Conservation Laws
- Hirsch, C. (2007). Numerical Computation of Internal and External Flows
- Godunov, S.K. (1959). A difference method for numerical calculation
"""

import pytest
import torch
from unittest.mock import patch

# Import RELIC modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import relic.scientific.transport.advection as advection_module
from relic.scientific.transport.advection import (
    LinearAdvection,
    advective_flux_units,
    create_uniform_advection,
)
from relic.core.functor import (
    ExecutionPhase,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode


_ADVECTION_NUMERICAL_PARAMS = {
    "grid_spacing": 1.0,
    "track_conservation": True,
    "enforce_non_negative": True,
    "boundary_method": None,
    "default_boundary_type": "neumann",
}


def _mock_advection_parameter_value(name, *args, **kwargs):
    """Return test values for LinearAdvection numerical parameter lookups."""
    return _ADVECTION_NUMERICAL_PARAMS[name]


def _linear_increment(result):
    """Return the canonical integrated tracer increment."""
    update = result["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


class TestLinearAdvectionInit:
    """Test LinearAdvection class initialization and configuration."""

    def test_public_api_excludes_temporal_aggregation_alias(self):
        """Temporal aggregation remains owned by its canonical coupling module."""
        assert not hasattr(advection_module, "AggregationMethod")
        assert "AggregationMethod" not in advection_module.__all__

    def test_init_1d_default(self):
        """Test 1D LinearAdvection initialization with default parameters."""
        advection = LinearAdvection(spatial_dims=1)

        assert advection.spatial_dims == 1
        assert advection.scheme == "upwind"
        assert advection.device == torch.device("cpu")
        assert len(advection.grid_spacing) == 1
        assert advection.non_negative is True  # Default from parameter system
        assert advection.boundary_method == "neumann"
        assert not hasattr(advection, "boundary_manager")
        assert not hasattr(advection, "set_boundary_conditions")
        assert not hasattr(advection, "_compute_rate_core")

    def test_init_2d_custom(self):
        """Test 2D LinearAdvection initialization with custom parameters."""
        device = torch.device("cpu")
        velocity_field = [
            torch.ones(10, 20) * 2.0,  # u component
            torch.ones(10, 20) * 1.0   # v component
        ]
        grid_spacing = [1.0, 2.0]

        advection = LinearAdvection(
            spatial_dims=2,
            velocity_field=velocity_field,
            grid_spacing=grid_spacing,
            scheme="lax_wendroff",
            non_negative=False,
            device=device
        )

        assert advection.spatial_dims == 2
        assert advection.scheme == "lax_wendroff"
        assert advection.device == device
        assert advection.grid_spacing == [1.0, 2.0]
        assert advection.non_negative is False
        assert len(advection.velocity_field) == 2

    def test_init_3d_with_tensor_velocity(self):
        """Test 3D initialization with tensor velocity field."""
        velocity_tensor = torch.randn(5, 10, 15, 3)  # [z, y, x, components]

        advection = LinearAdvection(
            spatial_dims=3,
            velocity_field=velocity_tensor,
            grid_spacing=[0.5, 0.5, 0.5]
        )

        assert advection.spatial_dims == 3
        assert len(advection.velocity_field) == 3
        # Verify velocity components are extracted correctly
        for i in range(3):
            assert advection.velocity_field[i].shape == velocity_tensor.shape[:-1]

    def test_init_invalid_spatial_dims(self):
        """Test initialization with invalid spatial dimensions."""
        with pytest.raises(ValueError, match="Spatial dimensions must be 1, 2, or 3"):
            LinearAdvection(spatial_dims=4)

    def test_init_invalid_scheme(self):
        """Test initialization with invalid numerical scheme."""
        with pytest.raises(ValueError, match="Scheme must be"):
            LinearAdvection(spatial_dims=1, scheme="invalid_scheme")

    def test_init_invalid_grid_spacing(self):
        """Test initialization with invalid grid spacing."""
        with pytest.raises(ValueError, match="Grid spacing must have"):
            LinearAdvection(spatial_dims=2, grid_spacing=[1.0])  # Missing second dimension

    def test_init_with_parameter_system(self):
        """Test initialization with parameter system integration."""
        with patch.object(
            LinearAdvection,
            "get_parameter_value",
            side_effect=_mock_advection_parameter_value,
        ) as mock_get_parameter_value:
            advection = LinearAdvection(spatial_dims=1)

            # Verify only directly consumed controls are loaded by the functor.
            assert advection.track_conservation is True
            assert advection.non_negative is True
            requested = {call.args[0] for call in mock_get_parameter_value.call_args_list}
            assert requested == set(_ADVECTION_NUMERICAL_PARAMS)

    def test_metadata_creation(self):
        """Test proper metadata creation during initialization."""
        advection = LinearAdvection(
            spatial_dims=1,
            name="test_advection"
        )

        metadata = advection.metadata
        assert metadata.name == "test_advection"
        assert metadata.category == ProcessCategory.TRANSPORT_MECHANISM
        assert metadata.execution_phase == ExecutionPhase.MAIN_PROCESSES
        assert "mass" in metadata.conservation_laws
        assert "Canonical IPHS:" in metadata.mathematical_expression
        assert "passive transport uses" in metadata.mathematical_expression
        assert metadata.spatial_scale == SpatialScale.GRID_CELL
        assert metadata.variable_kind == VariableKind.STATE
        assert set(metadata.input_variables) == {
            "tracer_concentration",
            "velocity_field",
            "source_term",
            "tracer_concentration_boundary_state",
        }


class TestVelocityFieldManagement:
    """Test velocity field setup and management."""

    def test_set_velocity_field_1d_tensor(self):
        """Test setting 1D velocity field from tensor."""
        advection = LinearAdvection(spatial_dims=1)
        velocity = torch.ones(20) * 2.5

        advection.set_velocity_field(velocity)

        assert len(advection.velocity_field) == 1
        torch.testing.assert_close(advection.velocity_field[0], velocity)

    def test_set_velocity_field_2d_tuple(self):
        """Test setting 2D velocity field from tuple."""
        advection = LinearAdvection(spatial_dims=2)
        u = torch.ones(10, 15) * 1.5
        v = torch.ones(10, 15) * 0.8

        advection.set_velocity_field((u, v))

        assert len(advection.velocity_field) == 2
        torch.testing.assert_close(advection.velocity_field[0], u)
        torch.testing.assert_close(advection.velocity_field[1], v)

    def test_set_velocity_field_3d_list(self):
        """Test setting 3D velocity field from list."""
        advection = LinearAdvection(spatial_dims=3)
        u = torch.randn(5, 8, 12)
        v = torch.randn(5, 8, 12)
        w = torch.randn(5, 8, 12)

        advection.set_velocity_field([u, v, w])

        assert len(advection.velocity_field) == 3
        torch.testing.assert_close(advection.velocity_field[0], u)
        torch.testing.assert_close(advection.velocity_field[1], v)
        torch.testing.assert_close(advection.velocity_field[2], w)

    def test_set_velocity_field_wrong_dimensions(self):
        """Test error handling for incorrect velocity field dimensions."""
        advection = LinearAdvection(spatial_dims=2)
        u = torch.ones(10, 15)

        # Single component for 2D should fail
        with pytest.raises(ValueError, match="Need 2 velocity components"):
            advection.set_velocity_field(u)

        # Three components for 2D should fail
        with pytest.raises(ValueError, match="Need 2 velocity components"):
            advection.set_velocity_field([u, u, u])

    def test_set_velocity_field_device_transfer(self):
        """Test velocity field is transferred to correct device."""
        device = torch.device("cpu")
        advection = LinearAdvection(spatial_dims=1, device=device)
        velocity = torch.ones(20) * 2.0  # On default device

        advection.set_velocity_field(velocity)

        assert advection.velocity_field[0].device == device


class TestCFLStability:
    """Test the single operator-owned CFL diagnostic and guard."""

    def test_operator_owned_cfl_number_1d(self):
        """The canonical finite-volume operator reports the 1D CFL value."""
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            boundary_method="periodic",
        )
        velocity = torch.ones(10) * 2.0
        advection.set_velocity_field(velocity)

        outputs = advection.forward(
            {"tracer_concentration": torch.ones(10)},
            time=0.0,
            dt=0.1,
        )

        torch.testing.assert_close(
            outputs["diagnostics.cfl_number"],
            outputs["diagnostics.cfl_number"].new_tensor(0.2),
        )

    def test_operator_owned_cfl_number_2d_uses_storage_axis_order(self):
        """The operator maps storage spacing (dy, dx) to velocity (u, v)."""
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=[1.0, 2.0],
            boundary_method="periodic",
        )
        u = torch.ones(8, 10) * 1.5
        v = torch.ones(8, 10) * 1.0
        advection.set_velocity_field([u, v])

        outputs = advection.forward(
            {"tracer_concentration": torch.ones(8, 10)},
            time=0.0,
            dt=0.2,
        )

        torch.testing.assert_close(
            outputs["diagnostics.cfl_number"],
            outputs["diagnostics.cfl_number"].new_tensor(0.35),
        )
        assert advection._resolve_operator_spacing(u.shape) == (1.0, 2.0)

    def test_operator_owned_cfl_number_3d(self):
        """The canonical operator reports the multidimensional outflow CFL."""
        advection = LinearAdvection(
            spatial_dims=3,
            grid_spacing=[1.0, 1.0, 1.0],
            boundary_method="periodic",
        )
        u = torch.ones(5, 6, 7) * 0.5
        v = torch.ones(5, 6, 7) * 0.3
        w = torch.ones(5, 6, 7) * 0.2
        advection.set_velocity_field([u, v, w])

        outputs = advection.forward(
            {"tracer_concentration": torch.ones(5, 6, 7)},
            time=0.0,
            dt=0.5,
        )

        torch.testing.assert_close(
            outputs["diagnostics.cfl_number"],
            outputs["diagnostics.cfl_number"].new_tensor(0.5),
        )

    def test_unsafe_direct_step_fails_without_local_timestep_adjustment(self):
        """An unsafe direct step is rejected by the canonical operator."""
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            boundary_method="periodic",
        )
        velocity = torch.ones(10) * 10.0
        advection.set_velocity_field(velocity)

        with pytest.raises(ValueError, match="CFL stability violation"):
            advection.forward(
                {"tracer_concentration": torch.ones(10)},
                time=0.0,
                dt=0.2,
            )

    def test_local_cfl_and_auto_adjust_compatibility_paths_are_absent(self):
        """CFL ownership remains in the unified finite-volume operator."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)

        assert not hasattr(advection, "_compute_cfl_numbers")
        assert not hasattr(advection, "_apply_cfl_control")
        assert not hasattr(advection, "auto_adjust_timestep")
        assert not hasattr(advection, "min_timestep")
        assert not hasattr(advection, "limiter_epsilon")
        with pytest.raises(TypeError, match="auto_adjust_timestep"):
            LinearAdvection(auto_adjust_timestep=True)


class TestNumericalSchemes:
    """Test numerical schemes through the single canonical forward path."""

    def test_functor_has_no_duplicate_scheme_formulas(self):
        """Scheme formulas remain owned by the computation operator."""
        advection = LinearAdvection(spatial_dims=1)

        for method_name in (
            "_upwind_flux_1d",
            "_upwind_flux_interface",
            "_lax_wendroff_flux_1d",
            "_flux_limited_1d",
            "_limited_gradient_ratio",
        ):
            assert not hasattr(advection, method_name)

    def test_upwind_flux_1d_positive_velocity(self):
        """Positive upwind flow follows the periodic conservative stencil."""
        concentration = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dtype=torch.float32,
        )
        velocity = torch.full_like(concentration, 2.0)
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=velocity,
            grid_spacing=1.0,
            scheme="upwind",
            boundary_method="periodic",
            non_negative=False,
        )

        outputs = advection.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )

        expected = torch.tensor(
            [0.8, -0.2, -0.2, -0.2, -0.2],
            dtype=torch.float32,
        )
        torch.testing.assert_close(_linear_increment(outputs), expected)
        assert float(outputs["diagnostics.conservation_error"]) <= torch.finfo(
            torch.float32
        ).eps

    def test_upwind_flux_1d_negative_velocity(self):
        """Negative upwind flow selects the opposite periodic donor cell."""
        concentration = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dtype=torch.float32,
        )
        velocity = torch.full_like(concentration, -2.0)
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=velocity,
            grid_spacing=1.0,
            scheme="upwind",
            boundary_method="periodic",
            non_negative=False,
        )

        outputs = advection.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )

        expected = torch.tensor(
            [0.2, 0.2, 0.2, 0.2, -0.8],
            dtype=torch.float32,
        )
        torch.testing.assert_close(_linear_increment(outputs), expected)
        assert float(outputs["diagnostics.conservation_error"]) <= torch.finfo(
            torch.float32
        ).eps

    def test_lax_wendroff_flux_1d(self):
        """Lax-Wendroff executes through the canonical operator."""
        concentration = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.ones_like(concentration),
            grid_spacing=1.0,
            scheme="lax_wendroff",
            boundary_method="periodic",
            non_negative=False,
        )

        outputs = advection.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )
        increment = _linear_increment(outputs)

        assert increment.shape == concentration.shape
        assert torch.all(torch.isfinite(increment))
        assert torch.sum(increment) == pytest.approx(0.0, abs=1.0e-6)

    def test_flux_limited_superbee(self):
        """Superbee remains finite, conservative, and monotone."""
        concentration = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.ones_like(concentration),
            grid_spacing=1.0,
            scheme="superbee",
            boundary_method="periodic",
            non_negative=False,
        )

        outputs = advection.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )
        increment = _linear_increment(outputs)
        updated = concentration + increment

        assert torch.all(torch.isfinite(increment))
        assert torch.sum(increment) == pytest.approx(0.0, abs=1.0e-6)
        assert updated.min() >= concentration.min()
        assert updated.max() <= concentration.max()

    def test_flux_limited_van_leer(self):
        """Van Leer remains finite, conservative, and monotone."""
        concentration = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0], dtype=torch.float32)
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.ones_like(concentration),
            grid_spacing=1.0,
            scheme="van_leer",
            boundary_method="periodic",
            non_negative=False,
        )

        outputs = advection.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )
        increment = _linear_increment(outputs)
        updated = concentration + increment

        assert torch.all(torch.isfinite(increment))
        assert torch.sum(increment) == pytest.approx(0.0, abs=1.0e-6)
        assert updated.min() >= concentration.min()
        assert updated.max() <= concentration.max()


class TestConservationProperties:
    """Test conservation law verification."""

    def test_mass_conservation_1d(self):
        """Test mass conservation for 1D advection."""
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            track_conservation=True,
            boundary_method="periodic",
        )

        # Create initial concentration field
        concentration = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], dtype=torch.float32)
        velocity = torch.ones(5) * 0.5
        advection.set_velocity_field(velocity)

        # Small time step for accuracy
        dt = 0.01

        # Create state dictionary
        state = {
            "tracer_concentration": concentration.unsqueeze(0),  # Add batch dimension
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        # Compute forward step
        updates = advection.forward(state, time=0.0, dt=dt)

        # Check that conservation diagnostics are included
        assert "diagnostics.conservation_error" in updates
        assert "diagnostics.cfl_number" in updates

        # Conservation error should be small for advection
        conservation_error = updates["diagnostics.conservation_error"]
        assert torch.isfinite(conservation_error)

    def test_conservation_with_source(self):
        """Test conservation with source term."""
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            track_conservation=True
        )

        concentration = torch.ones(10, dtype=torch.float32)
        velocity = torch.ones(10) * 0.1
        source = torch.ones(10, dtype=torch.float32) * 0.01  # Constant source
        advection.set_velocity_field(velocity)

        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": source.unsqueeze(0)
        }

        dt = 0.1
        updates = advection.forward(state, time=0.0, dt=dt)

        # Total mass should increase due to source
        delta_concentration = _linear_increment(updates)
        expected_increase = source.unsqueeze(0) * dt

        torch.testing.assert_close(delta_concentration, expected_increase, atol=1e-6, rtol=1e-3)

    def test_non_negative_constraint(self):
        """Negative inventories fail instead of creating tracer mass."""
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            non_negative=True
        )

        # Concentration with negative values
        concentration = torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
        velocity = torch.ones(4) * 0.1
        advection.set_velocity_field(velocity)

        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        with pytest.raises(ValueError, match="non-negative concentration"):
            advection.forward(state, time=0.0, dt=0.1)


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_boundary_method_precedence(self):
        """Constructor override wins over config, then ParameterManager default."""
        config_selected = LinearAdvection(
            spatial_dims=1,
            config={"boundary_method": "open"},
        )
        constructor_selected = LinearAdvection(
            spatial_dims=1,
            boundary_method="periodic",
            config={"boundary_method": "open"},
        )

        assert config_selected.boundary_method == "open"
        assert constructor_selected.boundary_method == "periodic"
        assert constructor_selected._advection_operator.boundary_method == "periodic"

    def test_invalid_boundary_method_fails_fast(self):
        """Unknown boundary methods cannot silently select an operator branch."""
        with pytest.raises(ValueError, match="boundary_method must be one of"):
            LinearAdvection(spatial_dims=1, boundary_method="unsupported")

    def test_nonperiodic_boundary_requires_explicit_boundary_state(self):
        """Open boundaries cannot synthesize an undeclared inflow state."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.ones(8) * 0.1,
            grid_spacing=1.0,
            boundary_method="open",
            track_conservation=True,
        )
        state = {"tracer_concentration": torch.linspace(1.0, 2.0, 8)}

        with pytest.raises(
            ValueError,
            match="requires explicit boundary_state",
        ):
            advection.forward(state, time=0.0, dt=0.1)

    def test_nonperiodic_boundary_emits_signed_advective_port(self):
        """Open transport retains its outward-positive boundary cochain."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.ones(8) * 0.1,
            grid_spacing=1.0,
            boundary_method="open",
            track_conservation=True,
        )
        state = {
            "tracer_concentration": torch.linspace(1.0, 2.0, 8),
            "tracer_concentration_boundary_state": torch.zeros(8),
        }

        updates = advection.forward(state, time=0.0, dt=0.1)

        boundary_update = updates[
            "tracer_concentration_advective_boundary_flux"
        ]
        assert isinstance(boundary_update, UpdateItem)
        assert boundary_update.mode == UpdateMode.REPLACE
        assert boundary_update.per_timestep is False
        assert boundary_update.unit == "kg m-2 s-1"
        assert boundary_update.source_functor == "linear_advection"
        torch.testing.assert_close(
            boundary_update.value,
            torch.tensor(
                [-0.0, 0.2],
                dtype=boundary_update.value.dtype,
            ),
        )
        assert float(updates["diagnostics.conservation_error"]) < 1.0e-6

    def test_boundary_port_preserves_independent_batch_ledgers(self):
        """Two simulations must retain separate oriented boundary cochains."""
        concentration = torch.stack(
            (
                torch.linspace(1.0, 2.0, 8),
                torch.linspace(2.0, 4.0, 8),
            )
        )
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.full_like(concentration, 0.1),
            grid_spacing=1.0,
            boundary_method="open",
            track_conservation=True,
        )

        updates = advection.forward(
            {
                "tracer_concentration": concentration,
                "tracer_concentration_boundary_state": torch.zeros_like(
                    concentration
                ),
            },
            time=0.0,
            dt=0.1,
        )

        boundary_cochain = updates[
            "tracer_concentration_advective_boundary_flux"
        ].value
        assert boundary_cochain.shape == (2, 2)
        torch.testing.assert_close(
            boundary_cochain,
            torch.tensor(
                [[-0.0, 0.2], [-0.0, 0.4]],
                dtype=boundary_cochain.dtype,
            ),
        )

    def test_periodic_boundary_rejects_unused_boundary_state(self):
        """Closed transport rejects a boundary effort with no active port."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.ones(8) * 0.1,
            grid_spacing=1.0,
            boundary_method="periodic",
        )
        state = {
            "tracer_concentration": torch.ones(8),
            "tracer_concentration_boundary_state": torch.zeros(8),
        }

        with pytest.raises(
            ValueError,
            match="boundary_state is only supported",
        ):
            advection.forward(state, time=0.0, dt=0.1)

    def test_boundary_flux_computation(self):
        """Test boundary flux computation."""
        advection = LinearAdvection(spatial_dims=1)
        velocity = torch.ones(3) * 2.0
        advection.set_velocity_field(velocity)

        field_values = torch.tensor([1.0])
        normal_vector = torch.tensor([-1.0])  # Outward normal at x_min

        boundary_flux = advection.boundary_conditions(
            "x_min", field_values, normal_vector
        )

        # Flux should be (v dot n) * c.
        expected_flux = velocity[0] * normal_vector[0] * field_values
        torch.testing.assert_close(boundary_flux, expected_flux)

    def test_boundary_flux_uses_multidimensional_normal_velocity(self):
        """Boundary flux contracts every velocity component with the normal."""
        advection = LinearAdvection(spatial_dims=2)
        velocity_x = torch.full((2, 3), 2.0)
        velocity_y = torch.full((2, 3), 0.5)
        advection.set_velocity_field((velocity_x, velocity_y))

        field_values = torch.tensor([1.0, 2.0])
        normal_vectors = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])

        boundary_flux = advection.boundary_conditions(
            "x_min",
            field_values,
            normal_vectors,
        )

        torch.testing.assert_close(
            boundary_flux,
            torch.tensor([-2.0, -4.0]),
        )


class TestForwardMethod:
    """Test forward method execution and error handling."""

    def test_forward_basic_1d(self):
        """Test basic forward method execution for 1D advection."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)
        velocity = torch.ones(10) * 0.5
        advection.set_velocity_field(velocity)

        concentration = torch.ones(10, dtype=torch.float32)
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 0.1
        updates = advection.forward(state, time=0.0, dt=dt)

        # Check required output keys
        assert "tracer_concentration" in updates
        assert "diagnostics.cfl_number" in updates

        # Check output shapes
        assert _linear_increment(updates).shape == concentration.unsqueeze(0).shape
        tracer_update = updates["tracer_concentration"]
        assert tracer_update.unit == "kg m-3"
        assert tracer_update.source_functor == "linear_advection"

        # The canonical operator reports |u| dt / dx for constant 1D flow.
        cfl = updates["diagnostics.cfl_number"]
        assert torch.isfinite(cfl)
        torch.testing.assert_close(cfl, cfl.new_tensor(0.05))

    def test_forward_with_velocity_in_state(self):
        """Test forward method with velocity field provided in state."""
        advection = LinearAdvection(spatial_dims=2, grid_spacing=[1.0, 1.0])

        concentration = torch.ones(8, 10, dtype=torch.float32)
        u = torch.ones(8, 10) * 1.0
        v = torch.ones(8, 10) * 0.5

        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "velocity_field": torch.stack((u, v), dim=-1),
        }

        dt = 0.1
        advection.forward(state, time=0.0, dt=dt)

        # Runtime forcing must not mutate or cache functor configuration.
        assert advection.velocity_field is None

        with pytest.raises(RuntimeError, match="Velocity field not set"):
            advection.forward(
                {"tracer_concentration": concentration.unsqueeze(0)},
                time=dt,
                dt=dt,
            )

    def test_forward_missing_concentration(self):
        """Test error handling for missing concentration field."""
        advection = LinearAdvection(spatial_dims=1)

        state = {}  # Missing tracer_concentration

        with pytest.raises(ValueError, match="State must contain"):
            advection.forward(state, time=0.0, dt=0.1)

    def test_forward_missing_velocity(self):
        """Required velocity cannot be synthesized from parameter defaults."""
        advection = LinearAdvection(spatial_dims=1)
        state = {"tracer_concentration": torch.ones(1, 8)}

        with pytest.raises(RuntimeError, match="Velocity field not set"):
            advection.forward(state, time=0.0, dt=0.1)

    @pytest.mark.parametrize("alias", ["concentration", "concentration_field"])
    def test_forward_rejects_concentration_aliases(self, alias):
        """Only the declared tracer state key is accepted."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.zeros(8),
        )

        with pytest.raises(ValueError, match="tracer_concentration"):
            advection.forward({alias: torch.ones(8)}, time=0.0, dt=0.1)

    @pytest.mark.parametrize("alias", ["u", "x", "velocity_x"])
    def test_forward_rejects_velocity_aliases(self, alias):
        """Only velocity_field may provide runtime velocity."""
        advection = LinearAdvection(spatial_dims=1)
        state = {
            "tracer_concentration": torch.ones(1, 8),
            alias: torch.zeros(8),
        }

        with pytest.raises(RuntimeError, match="Velocity field not set"):
            advection.forward(state, time=0.0, dt=0.1)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("tracer_concentration", float("nan"), "finite"),
            ("velocity_field", float("inf"), "finite"),
            ("source_term", float("nan"), "finite"),
        ],
    )
    def test_forward_rejects_non_finite_inputs(self, field, value, match):
        """Non-finite state, forcing, and source values fail before transport."""
        state = {
            "tracer_concentration": torch.ones(1, 8),
            "velocity_field": torch.zeros(8),
            "source_term": torch.zeros(1, 8),
        }
        state[field] = torch.full_like(state[field], value)
        advection = LinearAdvection(spatial_dims=1)

        with pytest.raises(ValueError, match=match):
            advection.forward(state, time=0.0, dt=0.1)

    def test_forward_dimension_mismatch(self):
        """Test error handling for tensor dimension mismatch."""
        advection = LinearAdvection(spatial_dims=2)

        # 1D concentration for 2D advection should fail
        concentration = torch.ones(10, dtype=torch.float32)
        state = {"tracer_concentration": concentration}

        with pytest.raises(ValueError, match="must have"):
            advection.forward(state, time=0.0, dt=0.1)

    def test_forward_velocity_field_from_state_tensor(self):
        """Test velocity field setup from tensor in state."""
        advection = LinearAdvection(spatial_dims=2, grid_spacing=[1.0, 1.0])

        concentration = torch.ones(8, 10, dtype=torch.float32)
        velocity_tensor = torch.randn(8, 10, 2)  # [y, x, components]

        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "velocity_field": velocity_tensor
        }

        dt = 0.1
        updates = advection.forward(state, time=0.0, dt=dt)

        # Per-step velocity is consumed without becoming persistent state.
        assert advection.velocity_field is None
        assert _linear_increment(updates).shape == concentration.unsqueeze(0).shape


class TestMultiDimensionalTransport:
    """Test multi-dimensional transport capabilities."""

    def test_2d_advection_forward(self):
        """Test 2D advection forward computation."""
        advection = LinearAdvection(spatial_dims=2, grid_spacing=[1.0, 1.0])

        concentration = torch.ones(6, 8, dtype=torch.float32)
        u = torch.ones(6, 8) * 1.0
        v = torch.ones(6, 8) * 0.5

        advection.set_velocity_field([u, v])

        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 0.1
        updates = advection.forward(state, time=0.0, dt=dt)

        # Check output
        assert "tracer_concentration" in updates
        assert _linear_increment(updates).shape == concentration.unsqueeze(0).shape

    def test_3d_advection_forward(self):
        """Test 3D advection forward computation."""
        advection = LinearAdvection(spatial_dims=3, grid_spacing=[1.0, 1.0, 1.0])

        concentration = torch.ones(4, 5, 6, dtype=torch.float32)
        u = torch.ones(4, 5, 6) * 0.5
        v = torch.ones(4, 5, 6) * 0.3
        w = torch.ones(4, 5, 6) * 0.2

        advection.set_velocity_field([u, v, w])

        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 0.1
        updates = advection.forward(state, time=0.0, dt=dt)

        # Check output
        assert "tracer_concentration" in updates
        assert _linear_increment(updates).shape == concentration.unsqueeze(0).shape

    def test_cfl_computation_multidimensional(self):
        """Forward propagates the operator result without a second CFL formula."""
        advection = LinearAdvection(spatial_dims=2, grid_spacing=[1.0, 2.0])
        concentration = torch.ones(4, 5)
        velocity = (torch.ones(4, 5), torch.ones(4, 5))
        advection.set_velocity_field(velocity)
        operator_result = {
            "concentration": concentration.clone(),
            "advection_rate": torch.zeros_like(concentration),
            "cfl_number": torch.full_like(concentration, 0.375),
            "boundary_fluxes": {},
            "conservation_error": torch.zeros(()),
        }

        with patch.object(
            advection._advection_operator,
            "apply",
            return_value=operator_result,
        ) as apply:
            outputs = advection.forward(
                {"tracer_concentration": concentration},
                time=0.0,
                dt=0.1,
            )

        apply.assert_called_once()
        torch.testing.assert_close(
            outputs["diagnostics.cfl_number"],
            outputs["diagnostics.cfl_number"].new_tensor(0.375),
        )


class TestComputeFluxMethods:
    """Test that the public API exposes integrated state updates only."""

    def test_compute_flux_1d(self):
        """Test flux computation for 1D case."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)
        velocity = torch.ones(20) * 2.0
        advection.set_velocity_field(velocity)

        concentration = torch.ones(20, dtype=torch.float32)

        outputs = advection.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )

        assert "advective_flux_x" not in outputs
        assert _linear_increment(outputs).shape == concentration.shape

    def test_compute_flux_2d(self):
        """Test flux computation for 2D case."""
        advection = LinearAdvection(spatial_dims=2, grid_spacing=[1.0, 1.0])
        u = torch.ones(8, 10) * 1.0
        v = torch.ones(8, 10) * 0.5
        advection.set_velocity_field([u, v])

        concentration = torch.ones(8, 10, dtype=torch.float32)

        outputs = advection.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )

        assert "advective_flux_x" not in outputs
        assert _linear_increment(outputs).shape == concentration.shape

    def test_compute_flux_no_velocity(self):
        """Missing velocity must fail instead of using a hidden default."""
        advection = LinearAdvection(spatial_dims=1)
        concentration = torch.ones(10, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="Velocity field not set"):
            advection.forward(
                {"tracer_concentration": concentration},
                time=0.0,
                dt=0.1,
            )


class TestComputeRateMethod:
    """Test canonical forward state-update semantics."""

    def test_compute_rate_basic(self):
        """Test basic compute_rate functionality."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)
        velocity = torch.ones(10) * 0.5
        advection.set_velocity_field(velocity)

        concentration = torch.ones(10, dtype=torch.float32)
        state = {"tracer_concentration": concentration.unsqueeze(0)}
        outputs = advection.forward(state, time=0.0, dt=0.1)

        assert "tracer_concentration" in outputs
        assert "diagnostics.cfl_number" in outputs
        assert _linear_increment(outputs).shape == concentration.unsqueeze(0).shape

    def test_compute_rate_with_environment_velocity(self):
        """Test compute_rate with velocity from environment."""
        advection = LinearAdvection(spatial_dims=2, grid_spacing=[1.0, 1.0])

        concentration = torch.ones(8, 10, dtype=torch.float32)
        env_velocity_x = torch.ones(8, 10) * 1.0
        env_velocity_y = torch.ones(8, 10) * 0.5
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "velocity_field": torch.stack(
                (env_velocity_x, env_velocity_y),
                dim=-1,
            )
        }

        outputs = advection.forward(state, time=0.0, dt=0.1)

        # Per-step velocity must not be cached as functor configuration.
        assert advection.velocity_field is None
        assert "tracer_concentration" in outputs
        assert isinstance(outputs["tracer_concentration"], UpdateItem)

    def test_compute_rate_missing_concentration(self):
        """Test error handling for missing concentration in compute_rate."""
        advection = LinearAdvection(spatial_dims=1)

        state = {}  # Missing tracer_concentration
        with pytest.raises(ValueError, match="State must contain"):
            advection.forward(state, time=0.0, dt=0.1)


class TestUtilityFunctions:
    """Test utility functions and helper methods."""

    def test_create_uniform_advection_1d(self):
        """Test create_uniform_advection for 1D case."""
        advection = create_uniform_advection(
            spatial_dims=1,
            velocity_magnitude=2.0,
            grid_spacing=1.0
        )

        assert advection.spatial_dims == 1
        assert advection.scheme == "upwind"
        assert len(advection.velocity_field) == 1
        assert torch.allclose(advection.velocity_field[0], torch.full_like(advection.velocity_field[0], 2.0))

    def test_create_uniform_advection_2d_with_direction(self):
        """Test create_uniform_advection for 2D case with direction."""
        velocity_magnitude = 3.0
        velocity_direction = (0.6, 0.8)  # Unit vector

        advection = create_uniform_advection(
            spatial_dims=2,
            velocity_magnitude=velocity_magnitude,
            velocity_direction=velocity_direction,
            grid_spacing=[1.0, 1.0]
        )

        assert advection.spatial_dims == 2
        assert len(advection.velocity_field) == 2

        # Check velocity magnitude and direction
        u_mag = torch.mean(torch.abs(advection.velocity_field[0]))
        v_mag = torch.mean(torch.abs(advection.velocity_field[1]))

        assert abs(u_mag - velocity_magnitude * velocity_direction[0]) < 0.1
        assert abs(v_mag - velocity_magnitude * velocity_direction[1]) < 0.1

    def test_create_uniform_advection_with_custom_grid_shape(self):
        """Test create_uniform_advection respects explicit grid_shape input."""
        advection = create_uniform_advection(
            spatial_dims=2,
            velocity_magnitude=1.0,
            velocity_direction=(1.0, 0.0),
            grid_spacing=[1.0, 1.0],
            grid_shape=(7, 9),
        )

        assert advection.velocity_field is not None
        assert advection.velocity_field[0].shape == torch.Size([7, 9])
        assert advection.velocity_field[1].shape == torch.Size([7, 9])

    def test_create_uniform_advection_invalid_direction(self):
        """Test error handling for invalid velocity direction."""
        with pytest.raises(ValueError, match="Direction must have"):
            create_uniform_advection(
                spatial_dims=2,
                velocity_magnitude=1.0,
                velocity_direction=(1.0,)  # Missing y-component
            )

    def test_grid_coordinates_have_no_functor_local_builder(self):
        """Grid geometry remains owned outside the scientific Functor."""
        advection = LinearAdvection(spatial_dims=2, grid_spacing=[1.0, 1.0])

        assert not hasattr(advection, "_generate_grid_coordinates")

    def test_compute_velocity_magnitude_1d(self):
        """Test velocity magnitude computation for 1D."""
        advection = LinearAdvection(spatial_dims=1)
        velocity = torch.ones(10) * 2.5
        advection.set_velocity_field(velocity)

        magnitude = advection._compute_velocity_magnitude()

        expected_magnitude = torch.abs(velocity)
        torch.testing.assert_close(magnitude, expected_magnitude)

    def test_compute_velocity_magnitude_2d(self):
        """Test velocity magnitude computation for 2D."""
        advection = LinearAdvection(spatial_dims=2)
        u = torch.ones(8, 10) * 3.0
        v = torch.ones(8, 10) * 4.0
        advection.set_velocity_field([u, v])

        magnitude = advection._compute_velocity_magnitude()

        expected_magnitude = torch.sqrt(u**2 + v**2)
        torch.testing.assert_close(magnitude, expected_magnitude)

    def test_hodge_uses_euclidean_velocity_magnitude(self):
        """The structural Hodge speed uses the full vector norm."""
        advection = LinearAdvection(
            spatial_dims=2,
            velocity_field=(
                torch.full((2, 3), 3.0),
                torch.full((2, 3), 4.0),
            ),
        )

        hodge = advection.build_hodge_contract(n_dofs=2)

        torch.testing.assert_close(
            torch.diag(hodge.matrix),
            torch.full((2,), 5.0, dtype=torch.float32),
        )

    def test_compute_velocity_magnitude_3d(self):
        """Test velocity magnitude computation for 3D."""
        advection = LinearAdvection(spatial_dims=3)
        u = torch.ones(4, 5, 6) * 1.0
        v = torch.ones(4, 5, 6) * 2.0
        w = torch.ones(4, 5, 6) * 2.0
        advection.set_velocity_field([u, v, w])

        magnitude = advection._compute_velocity_magnitude()

        expected_magnitude = torch.sqrt(u**2 + v**2 + w**2)
        torch.testing.assert_close(magnitude, expected_magnitude)


class TestParameterSystemIntegration:
    """Test integration with parameter system."""

    def test_parameter_loading(self):
        """Test parameter loading from parameter manager."""
        with patch.object(
            LinearAdvection,
            "get_parameter_value",
            side_effect=_mock_advection_parameter_value,
        ) as mock_get_parameter_value:
            advection = LinearAdvection(spatial_dims=1)

        # Verify only parameters used directly by this functor were loaded.
        assert advection.track_conservation is True
        assert advection.non_negative is True
        assert not hasattr(advection, "max_cfl")
        assert not hasattr(advection, "auto_adjust_timestep")
        assert not hasattr(advection, "min_timestep")
        assert not hasattr(advection, "conservation_tolerance")
        assert not hasattr(advection, "limiter_epsilon")
        assert not hasattr(advection, "artificial_viscosity")
        requested = {call.args[0] for call in mock_get_parameter_value.call_args_list}
        assert requested == {
            "grid_spacing",
            "track_conservation",
            "enforce_non_negative",
            "boundary_method",
            "default_boundary_type",
        }

    def test_parameter_system_fail_fast(self):
        """Test fail-fast behavior when parameter system is unavailable."""
        with patch.object(
            LinearAdvection,
            "get_parameter_value",
            side_effect=RuntimeError("ParameterManager unavailable"),
        ):
            with pytest.raises(RuntimeError, match="ParameterManager"):
                LinearAdvection(spatial_dims=1)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_zero_velocity_field(self):
        """Test behavior with zero velocity field."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)
        zero_velocity = torch.zeros(10)
        advection.set_velocity_field(zero_velocity)

        concentration = torch.ones(10, dtype=torch.float32)
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 0.1
        updates = advection.forward(state, time=0.0, dt=dt)

        # With zero velocity, should have minimal changes
        delta_concentration = _linear_increment(updates)
        assert torch.allclose(delta_concentration, torch.zeros_like(delta_concentration), atol=1e-10)

    def test_very_small_timestep(self):
        """Test behavior with very small time steps."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)
        velocity = torch.ones(10) * 1.0
        advection.set_velocity_field(velocity)

        concentration = torch.ones(10, dtype=torch.float32)
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 1e-10  # Very small time step
        updates = advection.forward(state, time=0.0, dt=dt)

        # Should handle small time steps gracefully
        assert "tracer_concentration" in updates
        assert torch.isfinite(_linear_increment(updates)).all()

    def test_large_concentration_values(self):
        """Test behavior with large concentration values."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)
        velocity = torch.ones(5) * 0.1
        advection.set_velocity_field(velocity)

        # Very large concentration values
        concentration = torch.tensor([1e6, 1e7, 1e8, 1e9, 1e10], dtype=torch.float32)
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 0.01
        updates = advection.forward(state, time=0.0, dt=dt)

        # Should handle large values without overflow
        assert torch.isfinite(_linear_increment(updates)).all()

    def test_single_point_domain(self):
        """Test behavior with single-point domain."""
        advection = LinearAdvection(spatial_dims=1, grid_spacing=1.0)
        velocity = torch.ones(1) * 1.0
        advection.set_velocity_field(velocity)

        concentration = torch.tensor([1.0], dtype=torch.float32)
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 0.1
        updates = advection.forward(state, time=0.0, dt=dt)

        # Should handle single point gracefully
        assert "tracer_concentration" in updates
        assert _linear_increment(updates).shape == concentration.unsqueeze(0).shape


# Test fixtures for common setup
@pytest.fixture
def basic_1d_advection():
    """Fixture providing basic 1D advection setup."""
    return LinearAdvection(spatial_dims=1, grid_spacing=1.0)


@pytest.fixture
def basic_2d_advection():
    """Fixture providing basic 2D advection setup."""
    velocity = [torch.ones(8, 10) * 1.0, torch.ones(8, 10) * 0.5]
    return LinearAdvection(spatial_dims=2, velocity_field=velocity, grid_spacing=[1.0, 1.0])


@pytest.fixture
def basic_3d_advection():
    """Fixture providing basic 3D advection setup."""
    velocity = [
        torch.ones(4, 5, 6) * 0.5,
        torch.ones(4, 5, 6) * 0.3,
        torch.ones(4, 5, 6) * 0.2
    ]
    return LinearAdvection(spatial_dims=3, velocity_field=velocity, grid_spacing=[1.0, 1.0, 1.0])


@pytest.fixture
def sample_concentration_1d():
    """Fixture providing sample 1D concentration field."""
    return torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], dtype=torch.float32)


@pytest.fixture
def sample_concentration_2d():
    """Fixture providing sample 2D concentration field."""
    return torch.ones(8, 10, dtype=torch.float32)


@pytest.fixture
def sample_concentration_3d():
    """Fixture providing sample 3D concentration field."""
    return torch.ones(4, 5, 6, dtype=torch.float32)


# Integration test examples using fixtures
def test_fixture_based_1d_advection(basic_1d_advection, sample_concentration_1d):
    """Example test using fixtures for 1D advection."""
    velocity = torch.ones(5) * 0.5
    basic_1d_advection.set_velocity_field(velocity)

    state = {
        "tracer_concentration": sample_concentration_1d.unsqueeze(0),
        "source_term": torch.zeros_like(sample_concentration_1d.unsqueeze(0))
    }

    updates = basic_1d_advection.forward(state, time=0.0, dt=0.1)

    assert "tracer_concentration" in updates
    assert "diagnostics.cfl_number" in updates


def test_fixture_based_2d_advection(basic_2d_advection, sample_concentration_2d):
    """Example test using fixtures for 2D advection."""
    state = {
        "tracer_concentration": sample_concentration_2d.unsqueeze(0),
        "source_term": torch.zeros_like(sample_concentration_2d.unsqueeze(0))
    }

    updates = basic_2d_advection.forward(state, time=0.0, dt=0.1)

    assert "tracer_concentration" in updates
    assert _linear_increment(updates).shape == sample_concentration_2d.unsqueeze(0).shape

class TestTracerUnitsContract:
    """The tracer unit is declared per instance and drives the returned UpdateItem units."""

    def _run(self, functor):
        velocity = torch.ones(1, 10) * 0.5
        functor.set_velocity_field([velocity])
        concentration = torch.linspace(1.0, 2.0, 10).reshape(1, 10)
        return functor.forward({functor.tracer_key: concentration}, 0.0, 0.3)

    def test_default_tracer_is_a_mass_density(self):
        functor = LinearAdvection(spatial_dims=1, grid_spacing=[1.0], name="default_units")
        assert functor.tracer_units == "kg m-3"
        assert functor.boundary_flux_units == "kg m-2 s-1"
        result = self._run(functor)
        assert result[functor.tracer_key].unit == "kg m-3"
        if functor.boundary_flux_key in result:
            assert result[functor.boundary_flux_key].unit == "kg m-2 s-1"

    def test_declared_tracer_units_propagate_to_the_update_items(self):
        functor = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[1.0],
            tracer_key="temp_state",
            tracer_units="K",
            name="temperature_units",
        )
        assert functor.boundary_flux_units == "K m s-1"
        result = self._run(functor)
        assert result["temp_state"].unit == "K"
        if functor.boundary_flux_key in result:
            assert result[functor.boundary_flux_key].unit == "K m s-1"

    def test_unit_one_tracer_flux_is_a_velocity(self):
        assert advective_flux_units("dimensionless") == "m s-1"
        assert advective_flux_units("1") == "m s-1"
        assert advective_flux_units("g C m-3") == "g C m-2 s-1"
        with pytest.raises(ValueError, match="tracer_units"):
            LinearAdvection(spatial_dims=1, grid_spacing=[1.0], tracer_units="   ")



class TestLinearAdvectionRequiredInputs:
    """A velocity field bound to the instance is an optional state input."""

    def test_bound_velocity_field_is_optional_and_unbound_is_required(self):
        from relic.orchestration.node_registry import NodeRegistry
        from relic.orchestration.ssot_catalog import SSOTCatalog

        bound = LinearAdvection(spatial_dims=1, grid_spacing=[1.0], velocity_field=torch.ones(5))
        unbound = LinearAdvection(spatial_dims=1, grid_spacing=[1.0])

        assert "velocity_field" not in bound.get_required_inputs()
        assert "velocity_field" in unbound.get_required_inputs()
        definition = NodeRegistry(SSOTCatalog()).register_functor_instance(bound)
        assert definition.inputs["velocity_field"].required is False
        assert bound.validate_state({"tracer_concentration": torch.ones(5)}) == []

        unbound.set_velocity_field(torch.ones(5))
        assert "velocity_field" not in unbound.get_required_inputs()


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])


