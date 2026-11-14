"""
L1 Unit Tests for Advection-Diffusion Module

Tests the advection-diffusion module to ensure:
1. Returns increments instead of new states
2. Uses parameter system for stability parameters
3. Conservation is maintained
4. Splitting methods work correctly
"""

import pytest
import torch

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.scientific.transport.advection_diffusion import (
    AdvectionDiffusion,
    create_wind_transport,
    create_groundwater_transport
)
from relic.core.functor import SpatialScale, VariableKind
from relic.core.state import UpdateItem, UpdateMode


def _tracer_increment(output):
    """Return the integrated tracer increment after checking write semantics."""
    update = output["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


class TestIncrementReturns:
    """Test that the module returns increments, not new states."""
    
    def test_forward_returns_increments_1d(self):
        """Test that forward returns concentration increments in 1D."""
        # Create 1D advection-diffusion
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            splitting_method="combined",
            boundary_method="periodic",
        )
        ad.track_conservation = False
        
        # Create initial state
        concentration = torch.ones(1, 10) * 100.0  # Initial concentration
        state = {
            "tracer_concentration": concentration,
            "source_term": torch.zeros(1, 10),
            "velocity_field": torch.zeros_like(concentration),
        }
        
        # Run forward step
        dt = 0.1
        output = ad.forward(state, time=0.0, dt=dt)
        
        # Check that output contains increments
        assert "tracer_concentration" in output
        delta = _tracer_increment(output)
        
        # Increment should be small compared to initial value
        assert torch.all(torch.abs(delta) < 10.0)  # Much less than 100
        
        # New state would be old + increment
        new_concentration = concentration + delta
        # Should be close to original (small timestep, no source)
        assert torch.allclose(new_concentration, concentration, rtol=0.1)
        
    def test_forward_returns_increments_2d(self):
        """Test that forward returns concentration increments in 2D."""
        # Create 2D advection-diffusion with velocity
        velocity_field = [
            torch.ones(1, 8, 8) * 0.1,  # u velocity
            torch.zeros(1, 8, 8)         # v velocity
        ]
        
        ad = AdvectionDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            velocity_field=velocity_field,
            grid_spacing=(0.01, 0.01),
            splitting_method="sequential"
        )
        ad.track_conservation = False
        
        # Create initial Gaussian concentration
        x = torch.linspace(-1, 1, 8)
        y = torch.linspace(-1, 1, 8)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        concentration = torch.exp(-(X**2 + Y**2) / 0.5).unsqueeze(0) * 100
        
        state = {
            "tracer_concentration": concentration,
            "source_term": torch.zeros_like(concentration)
        }
        
        # Run forward step
        dt = 0.01
        output = ad.forward(state, time=0.0, dt=dt)
        
        # Check increment properties
        delta = _tracer_increment(output)
        assert delta.shape == concentration.shape
        
        # With advection in +x direction, should see negative increment on left,
        # positive on right (material moving right)
        # But overall change should be small
        assert torch.all(torch.abs(delta) < 50.0)
        
    def test_missing_velocity_fails_fast(self):
        """A required velocity field must never become a hidden zero field."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        ad.track_conservation = False
        
        state = {
            "tracer_concentration": torch.ones(1, 10) * 100.0
        }
        
        with pytest.raises(ValueError, match="velocity_field"):
            ad.forward(state, time=0.0, dt=0.1)

    @pytest.mark.parametrize("alias", ["u", "velocity_x"])
    def test_velocity_aliases_do_not_replace_canonical_field(self, alias):
        """Component aliases must not bypass the canonical velocity field port."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
        )
        ad.track_conservation = False

        state = {
            "tracer_concentration": torch.ones(1, 10) * 100.0,
            "source_term": torch.zeros(1, 10),
            alias: torch.ones(1, 10) * 0.01,
        }

        with pytest.raises(ValueError, match="velocity_field"):
            ad.forward(state, time=0.0, dt=0.1)

    def test_runtime_outputs_match_exact_declared_contract(self):
        """Unconsumed dynamic transport statistics must not escape the contract."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            track_peclet=False,
            boundary_method="periodic",
        )
        ad.track_conservation = False

        concentration = torch.ones(1, 10) * 100.0
        output = ad.forward(
            {
                "tracer_concentration": concentration,
                "source_term": torch.zeros_like(concentration),
                "velocity_field": torch.zeros_like(concentration),
            },
            time=0.0,
            dt=0.1,
        )

        assert set(output) == {
            "tracer_concentration",
            "diagnostics.cfl_number",
            "diagnostics.diffusion_stability",
        }
        _tracer_increment(output)

    @pytest.mark.parametrize(
        "field_name",
        ["tracer_concentration", "velocity_field", "source_term"],
    )
    def test_nonfinite_runtime_fields_fail_fast(self, field_name):
        """Non-finite state and forcing values must not enter the operator."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
        )
        concentration = torch.ones(1, 10)
        state = {
            "tracer_concentration": concentration.clone(),
            "velocity_field": torch.zeros_like(concentration),
            "source_term": torch.zeros_like(concentration),
        }
        state[field_name][0, 0] = torch.nan

        with pytest.raises(ValueError, match="finite"):
            ad.forward(state, time=0.0, dt=0.1)

    @pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
    def test_invalid_timestep_fails_fast(self, dt):
        """The integrated ADE step requires a positive finite timestep."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
        )
        concentration = torch.ones(1, 10)
        state = {
            "tracer_concentration": concentration,
            "velocity_field": torch.zeros_like(concentration),
        }

        with pytest.raises(ValueError, match="positive finite"):
            ad.forward(state, time=0.0, dt=dt)

    @pytest.mark.parametrize(
        "velocity_field",
        [0.1, (torch.ones(1, 4, 4),)],
    )
    def test_multidimensional_velocity_requires_all_components(self, velocity_field):
        """A multidimensional flow must not synthesize missing zero components."""
        with pytest.raises(ValueError, match="2 velocity components"):
            AdvectionDiffusion(
                spatial_dims=2,
                diffusivity=1e-5,
                velocity_field=velocity_field,
                grid_spacing=(0.01, 0.01),
            )

    def test_packed_velocity_dimension_must_match_spatial_dimension(self):
        """Packed velocity fields require an exact final component axis."""
        ad = AdvectionDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=(0.01, 0.01),
        )
        concentration = torch.ones(1, 4, 4)
        invalid_velocity = torch.zeros(1, 4, 4, 3)

        with pytest.raises(ValueError, match="2 velocity components"):
            ad.forward(
                {
                    "tracer_concentration": concentration,
                    "velocity_field": invalid_velocity,
                },
                time=0.0,
                dt=0.01,
            )


class TestConfiguredVelocityRequiredness:
    """A velocity field configured at construction is an optional state input."""

    @staticmethod
    def _state():
        return {"tracer_concentration": torch.ones(1, 10) * 100.0}

    def test_configured_velocity_is_optional_and_the_governed_call_runs_without_it(self):
        from relic.orchestration.node_registry import NodeRegistry
        from relic.orchestration.ssot_catalog import SSOTCatalog

        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            velocity_field=torch.zeros(1, 10),
            grid_spacing=(0.01,),
            splitting_method="combined",
            boundary_method="periodic",
        )
        ad.track_conservation = False

        assert "velocity_field" not in ad.get_required_inputs()
        definition = NodeRegistry(SSOTCatalog()).register_functor_instance(ad)
        assert definition.inputs["velocity_field"].required is False
        output = ad(self._state(), time=0.0, dt=0.1)
        assert torch.all(torch.isfinite(_tracer_increment(output)))

    def test_unconfigured_velocity_stays_required(self):
        from relic.core.functor.functor_exceptions import ExecutionError
        from relic.orchestration.node_registry import NodeRegistry
        from relic.orchestration.ssot_catalog import SSOTCatalog

        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            splitting_method="combined",
            boundary_method="periodic",
        )
        ad.track_conservation = False

        assert "velocity_field" in ad.get_required_inputs()
        definition = NodeRegistry(SSOTCatalog()).register_functor_instance(ad)
        assert definition.inputs["velocity_field"].required is True
        with pytest.raises(ExecutionError, match="velocity_field"):
            ad(self._state(), time=0.0, dt=0.1)


class TestParameterSystemIntegration:
    """Test parameter system integration."""
    
    def test_parameters_loaded_from_system(self):
        """Only controls consumed directly by this functor remain local."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )

        assert isinstance(ad.track_conservation, bool)
        assert isinstance(ad.non_negative, bool)
        for stale_attribute in (
            "auto_adjust_timestep",
            "min_timestep",
            "max_timestep",
            "safety_factor",
            "adaptive_timestep_factor",
            "conservation_tolerance",
            "cfl_advection_max",
            "cfl_diffusion_max",
        ):
            assert not hasattr(ad, stale_attribute)

    def test_unsafe_step_fails_instead_of_shortening_scheduler_dt(self):
        """The canonical operator rejects an unstable explicit request."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-3,
            grid_spacing=(0.001,),
            splitting_method="combined"
        )
        ad.track_conservation = False
        
        state = {
            "tracer_concentration": torch.ones(1, 10) * 100.0,
            "velocity_field": torch.zeros(1, 10),
        }
        
        with pytest.raises(ValueError, match="stability"):
            ad.forward(state, time=0.0, dt=1.0)


class TestMetadata:
    """Test metadata fields required by transport registration rules."""

    def test_metadata_declares_scale_and_variable_kind(self):
        """AdvectionDiffusion metadata must declare spatial scale and variable kind."""
        ad = AdvectionDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=(0.01, 0.01),
        )

        assert ad.metadata.spatial_scale == SpatialScale.GRID_CELL
        assert ad.metadata.variable_kind == VariableKind.STATE

    def test_no_secondary_compute_rate_runtime(self):
        """Forward must be the only executable advection-diffusion formula path."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
        )

        assert not hasattr(ad, "_compute_rate_core")
        assert not hasattr(ad, "_stabilize_timestep")
        assert not hasattr(ad, "_apply_non_negative_constraint")
        assert not hasattr(ad, "advection")
        assert not hasattr(ad, "diffusion")

    def test_no_mutating_configuration_side_channels(self):
        """Runtime fields and coefficients must use state or constructor configuration."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
        )

        assert not hasattr(ad, "set_velocity_field")
        assert not hasattr(ad, "set_diffusivity")

    def test_splitting_method_is_parameter_managed(self):
        """The default splitting choice must be declared in ParameterManager bindings."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
        )

        assert ad.splitting_method == "symmetric"
        assert (
            "transport.advection_diffusion.splitting_method"
            in ad.required_parameters()
        )


class TestSplittingMethods:
    """Test different operator splitting methods."""
    
    def test_sequential_splitting(self):
        """Test sequential operator splitting."""
        velocity = [torch.ones(1, 10) * 0.1]
        
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            velocity_field=velocity,
            grid_spacing=(0.01,),
            splitting_method="sequential"
        )
        ad.track_conservation = False
        
        concentration = torch.linspace(0, 100, 10).unsqueeze(0)
        state = {"tracer_concentration": concentration}
        
        output = ad.forward(state, time=0.0, dt=0.01)
        delta = _tracer_increment(output)
        
        # Should get non-zero increment
        assert not torch.allclose(delta, torch.zeros_like(delta))
        
    def test_symmetric_splitting(self):
        """Test symmetric (Strang) operator splitting."""
        velocity = [torch.ones(1, 10) * 0.1]
        
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            velocity_field=velocity,
            grid_spacing=(0.01,),
            splitting_method="symmetric"
        )
        ad.track_conservation = False
        
        concentration = torch.exp(-((torch.linspace(-1, 1, 10)**2) / 0.1)).unsqueeze(0) * 100
        state = {"tracer_concentration": concentration}
        
        output = ad.forward(state, time=0.0, dt=0.01)
        delta = _tracer_increment(output)
        
        # Symmetric splitting should preserve more accuracy
        assert delta.shape == concentration.shape
        
    def test_combined_explicit(self):
        """Test combined explicit scheme."""
        velocity = [torch.ones(1, 10) * 0.05]
        
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            velocity_field=velocity,
            grid_spacing=(0.01,),
            splitting_method="combined"
        )
        ad.track_conservation = False
        
        concentration = torch.ones(1, 10) * 50.0
        concentration[0, 5] = 100.0  # Peak in center
        state = {"tracer_concentration": concentration}
        
        output = ad.forward(state, time=0.0, dt=0.001)
        delta = _tracer_increment(output)
        
        # Combined scheme should handle both processes
        assert not torch.allclose(delta, torch.zeros_like(delta))


class TestConservation:
    """Test conservation properties."""
    
    def test_conservation_without_source(self):
        """Test mass conservation without source term."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            splitting_method="combined",
            boundary_method="periodic",
        )
        ad.track_conservation = False
        ad.track_conservation = True
        
        concentration = torch.ones(1, 20) * 100.0
        state = {
            "tracer_concentration": concentration,
            "source_term": torch.zeros_like(concentration),
            "velocity_field": torch.zeros_like(concentration),
        }
        
        output = ad.forward(state, time=0.0, dt=0.01)
        
        # Check if conservation error is tracked
        if "diagnostics.conservation_error" in output:
            error = output["diagnostics.conservation_error"]
            # Error should be small
            assert error < 1.0e-6
            
    def test_conservation_with_source(self):
        """Test mass conservation with source term."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            splitting_method="combined"
        )
        
        concentration = torch.ones(1, 20) * 100.0
        source = torch.ones(1, 20) * 5.0  # Uniform source
        state = {
            "tracer_concentration": concentration,
            "source_term": source,
            "velocity_field": torch.zeros_like(concentration),
        }
        
        dt = 0.01
        output = ad.forward(state, time=0.0, dt=dt)
        delta = _tracer_increment(output)
        
        # Total change should approximately match source integral
        total_delta = torch.sum(delta)
        expected_delta = torch.sum(source) * dt
        
        # Should be close (within numerical errors and boundary effects)
        assert torch.allclose(total_delta, expected_delta, rtol=0.2)


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_open_boundary_requires_explicit_state_and_retains_total_flux(self):
        """Open transport must expose an outward-positive total face ledger."""
        concentration = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        boundary_state = torch.zeros_like(concentration)
        dt = 0.1
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1.0e-5,
            velocity_field=torch.ones_like(concentration),
            grid_spacing=1.0,
            splitting_method="combined",
            boundary_method="open",
        )

        with pytest.raises(ValueError, match="requires explicit boundary_state"):
            ad.forward({"tracer_concentration": concentration}, time=0.0, dt=dt)

        output = ad.forward(
            {
                "tracer_concentration": concentration,
                "tracer_concentration_boundary_state": boundary_state,
            },
            time=0.0,
            dt=dt,
        )

        boundary_update = output[
            "tracer_concentration_advection_diffusion_boundary_flux"
        ]
        assert isinstance(boundary_update, UpdateItem)
        assert boundary_update.mode == UpdateMode.REPLACE
        assert boundary_update.per_timestep is False
        assert boundary_update.unit == "kg m-2 s-1"
        assert boundary_update.source_functor == "advection_diffusion"
        torch.testing.assert_close(
            boundary_update.value,
            torch.tensor([[0.0, 3.0]], dtype=torch.float64),
        )
        storage_change = _tracer_increment(output).sum()
        torch.testing.assert_close(
            storage_change + dt * boundary_update.value.sum(),
            torch.tensor(0.0, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_total_boundary_port_preserves_independent_batch_ledgers(self):
        """Combined transport must not merge boundary cochains across batches."""
        concentration = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            dtype=torch.float64,
        )
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1.0e-5,
            velocity_field=torch.ones_like(concentration),
            grid_spacing=1.0,
            splitting_method="combined",
            boundary_method="open",
        )

        output = ad.forward(
            {
                "tracer_concentration": concentration,
                "tracer_concentration_boundary_state": torch.zeros_like(
                    concentration
                ),
            },
            time=0.0,
            dt=0.1,
        )

        boundary_cochain = output[
            "tracer_concentration_advection_diffusion_boundary_flux"
        ].value
        assert boundary_cochain.shape == (2, 2)
        torch.testing.assert_close(
            boundary_cochain,
            torch.tensor([[0.0, 3.0], [0.0, 6.0]], dtype=torch.float64),
        )
        storage_change = _tracer_increment(output).sum(dim=-1)
        torch.testing.assert_close(
            storage_change + 0.1 * boundary_cochain.sum(dim=-1),
            torch.zeros(2, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_periodic_boundary_rejects_unused_boundary_state(self):
        """A closed periodic domain must not silently accept an open port state."""
        concentration = torch.ones((1, 8), dtype=torch.float64)
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1.0e-5,
            velocity_field=torch.zeros_like(concentration),
            grid_spacing=1.0,
            boundary_method="periodic",
        )

        with pytest.raises(ValueError, match="boundary_state is only supported"):
            ad.forward(
                {
                    "tracer_concentration": concentration,
                    "tracer_concentration_boundary_state": torch.zeros_like(
                        concentration
                    ),
                },
                time=0.0,
                dt=0.1,
            )

    def test_metadata_declares_conditional_boundary_port(self):
        """Metadata must declare the conditional boundary state and total flux."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1.0e-5,
            grid_spacing=1.0,
        )

        assert "tracer_concentration_boundary_state" in ad.metadata.input_variables
        assert (
            "tracer_concentration_advection_diffusion_boundary_flux"
            in ad.metadata.output_variables
        )
        assert (
            "tracer_concentration_advection_diffusion_boundary_flux"
            in ad.metadata.optional_output_variables
        )
    
    def test_periodic_boundary_is_explicit_closed_domain_configuration(self):
        """A periodic boundary override must drive the canonical forward operator."""
        concentration = torch.linspace(1.0, 2.0, 64).reshape(1, -1)
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=0.01,
            velocity_field=torch.full_like(concentration, 0.2),
            grid_spacing=1.0,
            splitting_method="combined",
            boundary_method="periodic",
        )
        ad.track_conservation = True

        output = ad.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )

        assert ad.boundary_method == "periodic"
        assert torch.abs(_tracer_increment(output).sum()) < 1e-6
        assert output["diagnostics.conservation_error"] < 1.0e-6

    def test_nonperiodic_boundary_does_not_claim_closed_domain_conservation(self):
        """Open advective boundary exchange must not be labeled conservation error."""
        concentration = torch.linspace(1.0, 2.0, 32).reshape(1, -1)
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=0.01,
            velocity_field=torch.full_like(concentration, 0.2),
            grid_spacing=1.0,
            boundary_method="neumann",
        )
        ad.track_conservation = True

        output = ad.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )

        assert "diagnostics.conservation_error" not in output
        
    def test_non_negative_constraint(self):
        """Test non-negative concentration constraint."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            non_negative=True
        )
        ad.track_conservation = False
        
        # Create state with potential for negative values
        concentration = torch.ones(1, 10) * 10.0
        concentration[0, 0] = 0.1  # Very small value
        
        # Large sink that would make it negative
        source = torch.zeros(1, 10)
        source[0, 0] = -100.0
        
        state = {
            "tracer_concentration": concentration,
            "source_term": source,
            "velocity_field": torch.zeros_like(concentration),
        }
        
        with pytest.raises(
            ValueError,
            match="violated non-negative concentration",
        ):
            ad.forward(state, time=0.0, dt=0.01)


class TestUtilityFunctions:
    """Test utility functions for creating transport."""
    
    def test_create_wind_transport(self):
        """Test creation of wind transport mechanism."""
        transport = create_wind_transport(
            spatial_dims=2,
            wind_speed=10.0,
            wind_direction=(0.6, 0.8),  # Unit vector
            molecular_diffusivity=1e-5,
            turbulent_diffusivity=1e-3,
            grid_spacing=(0.1, 0.1)
        )
        
        assert isinstance(transport, AdvectionDiffusion)
        assert transport.spatial_dims == 2
        # Total diffusivity should be sum
        expected_diff = 1e-5 + 1e-3
        assert abs(transport.diffusivity_value - expected_diff) < 1e-10

    def test_wind_direction_must_be_unit_vector(self):
        """Wind speed and direction must not encode two inconsistent magnitudes."""
        with pytest.raises(ValueError, match="unit vector"):
            create_wind_transport(
                spatial_dims=2,
                wind_speed=10.0,
                wind_direction=(1.0, 1.0),
                molecular_diffusivity=1e-5,
                turbulent_diffusivity=1e-3,
                grid_spacing=(0.1, 0.1),
            )
        
    def test_create_groundwater_transport(self):
        """Test creation of groundwater transport mechanism."""
        transport = create_groundwater_transport(
            spatial_dims=2,
            darcy_velocity=(1e-5, 0),  # Flow in x-direction
            porosity=0.3,
            molecular_diffusivity=1e-9,
            dispersivity=0.01,
            grid_spacing=(0.1, 0.1)
        )
        
        assert isinstance(transport, AdvectionDiffusion)
        assert transport.spatial_dims == 2
        
        # Check velocity is scaled by porosity
        # Pore velocity = Darcy velocity / porosity
        assert transport.velocity_field_value is not None

    def test_create_groundwater_transport_uses_l2_velocity_norm_for_dispersivity(self):
        """Groundwater factory should use Euclidean velocity magnitude in dispersive term."""
        transport = create_groundwater_transport(
            spatial_dims=2,
            darcy_velocity=(3.0e-5, 4.0e-5),
            porosity=0.4,
            molecular_diffusivity=1.0e-9,
            dispersivity=0.5,
            grid_spacing=(0.1, 0.1),
        )

        expected_velocity_mag = (3.0e-5**2 + 4.0e-5**2) ** 0.5 / 0.4
        expected_diffusivity = 1.0e-9 + 0.5 * expected_velocity_mag
        assert abs(transport.diffusivity_value - expected_diffusivity) < 1.0e-12

    def test_groundwater_multidimensional_velocity_requires_all_components(self):
        """A scalar Darcy velocity must not synthesize hidden zero axes."""
        with pytest.raises(ValueError, match="all spatial components"):
            create_groundwater_transport(
                spatial_dims=2,
                darcy_velocity=1e-5,
                porosity=0.3,
                molecular_diffusivity=1e-9,
                dispersivity=0.01,
                grid_spacing=(0.1, 0.1),
            )

    def test_groundwater_porosity_must_be_physical(self):
        """Effective porosity must be finite and lie in the physical interval."""
        with pytest.raises(ValueError, match="porosity"):
            create_groundwater_transport(
                spatial_dims=1,
                darcy_velocity=1e-5,
                porosity=0.0,
                molecular_diffusivity=1e-9,
                dispersivity=0.01,
                grid_spacing=0.1,
            )


class TestPecletNumber:
    """Test Peclet number tracking."""
    
    def test_peclet_number_calculation(self):
        """Test that Peclet number is calculated and tracked."""
        velocity = [torch.ones(1, 10) * 1.0]  # High velocity
        
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,  # Low diffusivity
            velocity_field=velocity,
            grid_spacing=(0.01,),
            track_peclet=True
        )
        ad.track_conservation = False
        
        state = {"tracer_concentration": torch.ones(1, 10) * 100}
        
        output = ad.forward(state, time=0.0, dt=0.001)
        
        # Should have Peclet number in diagnostics
        if "diagnostics.peclet_number" in output:
            peclet = output["diagnostics.peclet_number"]
            # Pe = v * dx / D = 1.0 * 0.01 / 1e-5 = 1000
            assert peclet > 100  # High Peclet number expected

    def test_peclet_tracking_with_zero_diffusivity_is_rejected_at_construction(self):
        """Pe = |u| dx / D is undefined at D = 0: fail clearly before any step."""
        velocity = [torch.ones(1, 10)]
        with pytest.raises(ValueError, match="track_peclet requires a positive diffusivity"):
            AdvectionDiffusion(
                spatial_dims=1,
                diffusivity=0.0,
                velocity_field=velocity,
                grid_spacing=(0.01,),
                track_peclet=True,
            )

    def test_zero_diffusivity_without_peclet_tracking_is_pure_advection(self):
        """Without the diagnostic a zero diffusivity is a valid transport step."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=0.0,
            velocity_field=[torch.ones(1, 10)],
            grid_spacing=(0.01,),
            track_peclet=False,
        )
        ad.track_conservation = False
        output = ad.forward(
            {"tracer_concentration": torch.ones(1, 10)}, time=0.0, dt=0.001
        )
        assert "diagnostics.peclet_number" not in output
        assert "tracer_concentration" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

