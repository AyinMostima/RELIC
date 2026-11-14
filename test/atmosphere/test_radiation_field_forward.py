"""
Test suite for RadiationField.forward method implementation.

This module validates the forward computation interface for RadiationField,
ensuring proper field initialization, state management, evolution, and output.

References:
    - Campbell & Norman (1998). Environmental Biophysics
    - RELIC Architecture Guidelines
"""

import torch
import pytest
import relic.scientific.atmosphere.radiation as radiation_module
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.atmosphere.radiation import RadiationField


def _value(result, key):
    """Return the tensor carried by one explicit runtime update."""
    update = result[key]
    assert isinstance(update, UpdateItem)
    assert isinstance(update.value, torch.Tensor)
    return update.value


class TestRadiationFieldForward:
    """Test cases for RadiationField.forward method."""

    @pytest.fixture
    def basic_field(self):
        """Create a basic radiation field for testing."""
        field = RadiationField(
            device=torch.device("cpu")
        )
        return field

    @pytest.fixture
    def initialized_field(self):
        """Create and initialize a radiation field."""
        field = RadiationField()

        # Initialize with grid
        grid_coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], dtype=torch.float32)

        initial_values = torch.tensor([100.0, 150.0, 120.0, 180.0], dtype=torch.float32)
        field.initialize_field(grid_coords, initial_values)

        return field

    @pytest.fixture
    def canonical_state(self, initialized_field):
        """Return the exact state owned by StateManager for this field."""
        return {
            'grid_coordinates': initialized_field._grid_coordinates.detach().clone(),
            'radiation_field': initialized_field._field_values.detach().clone(),
        }

    def test_forward_missing_required_state_fails_fast(self, basic_field):
        """Missing grid and field state must fail instead of returning no update."""
        state = {}  # No grid_coordinates
        time = 0.0
        dt = 3600.0

        with pytest.raises(KeyError, match="grid_coordinates"):
            basic_field.forward(state, time, dt)

    def test_forward_auto_initialization_from_state(self, basic_field):
        """Test automatic field initialization from state grid_coordinates."""
        grid_coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=torch.float32)

        state = {
            'grid_coordinates': grid_coords,
            'radiation_field': torch.tensor([100.0, 100.0, 100.0]),
        }
        time = 0.0
        dt = 3600.0

        result = basic_field.forward(state, time, dt)

        # Should return field values
        assert 'radiation_field' in result
        assert _value(result, 'radiation_field').shape == (3,)
        assert torch.all(torch.isfinite(_value(result, 'radiation_field')))

    def test_forward_returns_current_field(self, initialized_field, canonical_state):
        """Test that forward returns current field values."""
        state = canonical_state
        time = 0.0
        dt = 3600.0

        result = initialized_field.forward(state, time, dt)

        assert 'radiation_field' in result
        assert _value(result, 'radiation_field').shape == (4,)

        # Values should be within physical bounds
        assert torch.all(_value(result, 'radiation_field') >= initialized_field.min_radiation)
        assert torch.all(_value(result, 'radiation_field') <= initialized_field.max_radiation)

    def test_forward_uses_state_field_values(self, initialized_field):
        """Test that forward prioritizes radiation_field from state."""
        new_field_values = torch.tensor([300.0, 350.0, 320.0, 380.0], dtype=torch.float32)

        state = {
            'grid_coordinates': initialized_field._grid_coordinates.detach().clone(),
            'radiation_field': new_field_values,
        }
        time = 0.0
        dt = 3600.0

        result = initialized_field.forward(state, time, dt)

        assert 'radiation_field' in result
        # Should use values from state (clamped to physical bounds)
        assert torch.allclose(
            _value(result, 'radiation_field'),
            torch.clamp(new_field_values,
                       min=initialized_field.min_radiation,
                       max=initialized_field.max_radiation),
            rtol=1e-5
        )

    def test_forward_rejects_source_term_state_advancement(self, initialized_field, canonical_state):
        """The production forward path must not expose a second Euler advance."""
        source_terms = {
            'shortwave_input': torch.tensor([50.0, 50.0, 50.0, 50.0]),
            'longwave_emission': torch.tensor([20.0, 20.0, 20.0, 20.0]),
            'absorption_rate': torch.tensor([10.0, 10.0, 10.0, 10.0])
        }

        state = canonical_state
        time = 0.0
        dt = 1800.0  # 30 minutes

        with pytest.raises(TypeError, match="source_terms|update_field"):
            initialized_field.forward(
                state,
                time,
                dt,
                update_field=True,
                source_terms=source_terms,
            )

        with pytest.raises(RuntimeError, match="state-driven"):
            initialized_field.update_field(dt, source_terms)

    def test_forward_gradient_computation(self, initialized_field, canonical_state):
        """Test gradient computation when return_gradients=True."""
        state = canonical_state
        time = 0.0
        dt = 3600.0

        result = initialized_field.forward(
            state, time, dt,
            return_gradients=True
        )

        assert 'radiation_field' in result
        gradient_key = 'diagnostics.radiation_field.gradient'
        assert gradient_key in result

        # Gradients should have shape [n_points, spatial_dims]
        assert _value(result, gradient_key).ndim == 2
        assert _value(result, gradient_key).shape[1] == RadiationField.SPATIAL_DIMS == 2
        assert torch.all(torch.isfinite(_value(result, gradient_key)))

    def test_forward_gradient_with_custom_coordinates(self, initialized_field):
        """Test gradient computation with custom source_coordinates."""
        query_coords = torch.tensor([
            [0.5, 0.5],
            [0.25, 0.75]
        ], dtype=torch.float32)

        state = {
            'grid_coordinates': initialized_field._grid_coordinates.detach().clone(),
            'radiation_field': initialized_field._field_values.detach().clone(),
            'source_coordinates': query_coords,
        }
        time = 0.0
        dt = 3600.0

        result = initialized_field.forward(
            state, time, dt,
            return_gradients=True
        )

        gradient_key = 'diagnostics.radiation_field.gradient'
        assert gradient_key in result
        # Gradients computed at query coordinates
        assert _value(result, gradient_key).shape[0] == 2
        assert _value(result, gradient_key).shape[1] == 2

    def test_inverse_distance_interpolation_is_exact_at_source_points(
        self,
        initialized_field,
    ):
        """The declared IDW projection must preserve coincident source values."""
        assert initialized_field.interpolation_method == "inverse_distance"
        values = initialized_field.get_value_at(
            initialized_field._grid_coordinates.detach().clone()
        )
        torch.testing.assert_close(
            values,
            initialized_field._field_values.detach(),
            rtol=0.0,
            atol=0.0,
        )

    def test_forward_applies_physical_constraints(self, initialized_field):
        """Test that output is clamped to physical bounds."""
        # Provide extreme values in state
        extreme_values = torch.tensor([5000.0, -5000.0, 2000.0, -1000.0], dtype=torch.float32)

        state = {
            'grid_coordinates': initialized_field._grid_coordinates.detach().clone(),
            'radiation_field': extreme_values,
        }
        time = 0.0
        dt = 3600.0

        result = initialized_field.forward(state, time, dt)

        # All values should be within physical bounds
        assert torch.all(_value(result, 'radiation_field') >= initialized_field.min_radiation)
        assert torch.all(_value(result, 'radiation_field') <= initialized_field.max_radiation)

    def test_forward_rejects_nan_inf_values(self, initialized_field):
        """Non-finite field state must fail before clipping."""
        bad_values = torch.tensor([100.0, float('nan'), float('inf'), float('-inf')], dtype=torch.float32)

        state = {
            'grid_coordinates': initialized_field._grid_coordinates.detach().clone(),
            'radiation_field': bad_values,
        }
        time = 0.0
        dt = 3600.0

        with pytest.raises(ValueError, match="finite"):
            initialized_field.forward(state, time, dt)

    @pytest.mark.parametrize('invalid_dt', [0.0, -100.0, float('nan'), float('inf')])
    def test_forward_rejects_invalid_dt(
        self,
        initialized_field,
        canonical_state,
        invalid_dt,
    ):
        """Non-positive or non-finite dt must fail instead of retaining state."""
        state = canonical_state
        time = 0.0

        with pytest.raises(ValueError, match="dt must be positive and finite"):
            initialized_field.forward(state, time, invalid_dt)

    def test_forward_requires_boolean_gradient_option(self, initialized_field, canonical_state):
        """The diagnostics option must not act as a truthy mode selector."""
        with pytest.raises(TypeError, match='return_gradients must be a bool'):
            initialized_field.forward(
                canonical_state,
                time=0.0,
                dt=3600.0,
                return_gradients='yes',
            )

    def test_forward_returns_detached_copy(self, initialized_field, canonical_state):
        """Test that forward returns detached copies, not direct references."""
        state = canonical_state
        time = 0.0
        dt = 3600.0

        result = initialized_field.forward(state, time, dt)

        # Returned field should be detached
        assert not _value(result, 'radiation_field').requires_grad

        # Modifying result should not affect internal field
        _value(result, 'radiation_field')[0] = 999.0
        assert initialized_field._field_values[0] != 999.0

    def test_forward_integration_with_state_manager(self, initialized_field, canonical_state):
        """Test that forward output is consistent with StateManager interface."""
        state = canonical_state
        time = 0.0
        dt = 3600.0

        result = initialized_field.forward(state, time, dt)

        # Output should be a dict with Tensor values
        assert isinstance(result, dict)
        assert 'radiation_field' in result
        assert isinstance(result['radiation_field'], UpdateItem)

        # Should be valid for StateManager consumption
        # (no NaN/Inf, within bounds, proper shape)
        assert torch.all(torch.isfinite(_value(result, 'radiation_field')))
        assert _value(result, 'radiation_field').ndim == 1  # 1D for grid points

    def test_canonical_identity_update_mode_and_diagnostic_contract(self):
        """The field has one canonical writer identity and explicit state semantics."""
        field = RadiationField()
        assert field.name == "radiation_field"
        # Units are the radiation_field state contract's, not a field attribute.
        assert not hasattr(field, "units")

        state = {
            "grid_coordinates": torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ),
            "radiation_field": torch.tensor(
                [100.0, 5000.0, 200.0],
                dtype=torch.float32,
            ),
        }
        result = field.forward(
            state,
            time=0.0,
            dt=3600.0,
            return_gradients=True,
        )

        assert set(result) <= set(field.metadata.output_variables)
        assert "diagnostics.radiation_field.gradient" in result
        assert "diagnostics.radiation_field.clipping_residual" in result
        update = result["radiation_field"]
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.REPLACE
        assert update.unit == "W m-2"
        assert update.per_timestep is True
        assert update.source_functor == "radiation_field"

        with pytest.raises(TypeError, match="canonical"):
            RadiationField(field_name="alternate_radiation")
        with pytest.raises(TypeError, match="canonical"):
            RadiationField(units="W m-3")
        # Dimensionality is the class attribute SPATIAL_DIMS, not an argument.
        for keyword, value in (("spatial_dims", 2), ("temporal_dependence", True)):
            with pytest.raises(TypeError, match=keyword):
                RadiationField(**{keyword: value})
        for removed in ("field_name", "units", "default_value", "spatial_dims", "temporal_dependence"):
            assert not hasattr(field, removed), removed
        assert not hasattr(radiation_module, "create_radiation_field")

    def test_functor_lifecycle_uses_domain_energy_audit(
        self,
        initialized_field,
        canonical_state,
    ):
        """The bridge lifecycle accepts optional coordinates and audits projection."""
        result = initialized_field(canonical_state, time=0.0, dt=3600.0)

        assert torch.all(torch.isfinite(_value(result, 'radiation_field')))
        assert initialized_field.check_conservation_laws(
            canonical_state,
            result,
        ) == {'energy': 0.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


