"""
Test Suite for Transport Mechanisms

This module tests the transport mechanisms (advection, diffusion, advection-diffusion)
to verify:

1. Proper inheritance from TransportMechanismFunctor
2. Increments-only output (no direct state modification)
3. CFL condition checking for advection
4. Conservation properties
5. Forward method returns FluxDict format

References:
    - LeVeque (1992) Numerical Methods for Conservation Laws
    - Hirsch (2007) Numerical Computation of Internal and External Flows
"""

import pytest
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.core.functor import TransportMechanismFunctor
from relic.core.functor import FunctorBase
from relic.core.state import UpdateItem, UpdateMode


def _accumulate_increment(result, key="tracer_concentration"):
    """Return an integrated transport increment with canonical write semantics."""
    update = result[key]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    assert isinstance(update.value, torch.Tensor)
    return update.value


class TestLinearAdvection:
    """Test LinearAdvection transport mechanism."""

    def test_inheritance(self):
        """Verify LinearAdvection inherits from TransportMechanismFunctor."""
        assert issubclass(LinearAdvection, TransportMechanismFunctor)
        assert issubclass(LinearAdvection, FunctorBase)

    def test_initialization_1d(self):
        """Test 1D advection initialization."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.tensor([1.0]),
            grid_spacing=(0.1,),
            name="test_advection_1d"
        )

        assert advection.name == "test_advection_1d"
        assert advection.spatial_dims == 1
        assert hasattr(advection, 'metadata')

    def test_forward_returns_increments(self):
        """Test that forward returns increments, not new state values."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.tensor([1.0]),
            grid_spacing=(0.1,),
            scheme="upwind"
        )

        # Create test state with concentration field
        state = {
            "tracer_concentration": torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]),
        }

        dt = 0.01  # Small timestep for CFL
        result = advection.forward(state, time=0.0, dt=dt)

        # Check that result contains increments
        assert isinstance(result, dict)
        assert "tracer_concentration" in result or any("tracer" in k for k in result.keys())

        # The result should be an increment (change), not the new value
        # With upwind advection, we expect negative values where concentration decreases
        # and positive where it increases
        increment = _accumulate_increment(result)

        # Increment should not be equal to the new state
        assert not torch.allclose(increment, state["tracer_concentration"])

    def test_cfl_condition_check(self):
        """Unsafe direct steps fail instead of changing scheduler time."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.tensor([10.0]),  # High velocity
            grid_spacing=(0.1,),  # Small spacing
        )

        state = {
            "tracer_concentration": torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]),
        }

        # Large timestep violating CFL
        dt = 0.2  # CFL = 10.0 * 0.2 / 0.1 = 20 >> 1

        with pytest.raises(
            ValueError,
            match="CFL stability violation.*simulator-managed FunctorMetadata.substep_count",
        ):
            advection.forward(state, time=0.0, dt=dt)

    def test_get_dependencies(self):
        """Test that get_dependencies method exists and works."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.tensor([1.0]),
            grid_spacing=(0.1,)
        )

        deps = advection.get_dependencies()
        assert isinstance(deps, list)

    def test_conservation_tracking(self):
        """Test conservation tracking when enabled."""
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.tensor([0.0]),  # No flow for perfect conservation
            grid_spacing=(0.1,),
            track_conservation=True
        )

        state = {
            "tracer_concentration": torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]),
            "source_term": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        }

        result = advection.forward(state, time=0.0, dt=0.01)

        # With zero velocity and no sources, increment should be near zero
        if "tracer_concentration" in result:
            increment = _accumulate_increment(result)
            assert torch.allclose(increment, torch.zeros_like(increment), atol=1e-6)


class TestFickianDiffusion:
    """Test FickianDiffusion transport mechanism."""

    def test_inheritance(self):
        """Verify FickianDiffusion inherits from TransportMechanismFunctor."""
        assert issubclass(FickianDiffusion, TransportMechanismFunctor)
        assert issubclass(FickianDiffusion, FunctorBase)

    def test_initialization(self):
        """Test diffusion initialization."""
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.1,),
            name="test_diffusion"
        )

        assert diffusion.name == "test_diffusion"
        assert diffusion.spatial_dims == 1
        # FickianDiffusion owns its coefficient (scalar or field) through
        # set_diffusivity/get_diffusivity; a scalar D is also the base scalar.
        assert float(diffusion.get_diffusivity()) == pytest.approx(1e-5)
        assert diffusion.diffusivity == pytest.approx(1e-5)

    def test_forward_returns_increments(self):
        """Test that forward returns increments for diffusion."""
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-3,
            grid_spacing=(0.1,)
        )

        # Create test state with concentration gradient
        state = {
            "tracer_concentration": torch.tensor([0.0, 1.0, 4.0, 1.0, 0.0])
        }

        result = diffusion.forward(state, time=0.0, dt=0.01)

        # Check for increments
        assert isinstance(result, dict)

        # Use the declared prognostic state key, not the separate boundary port.
        conc_key = diffusion.tracer_key
        assert conc_key in result
        increment = _accumulate_increment(result, key=conc_key)

        # FickianDiffusion should smooth gradients - peak should decrease, valleys increase
        # Middle point (high concentration) should decrease
        assert increment[2] < 0
        # Adjacent points should increase
        assert increment[1] > 0
        assert increment[3] > 0

    def test_stability_check(self):
        """Test numerical stability checking for diffusion."""
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1.0,  # High diffusivity
            grid_spacing=(0.01,),  # Small spacing
        )

        state = {
            "tracer_concentration": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        }

        # Large timestep that may violate stability
        dt = 1.0  # D*dt/dx^2 = 1.0*1.0/0.01^2 = 10000 >> 0.5

        # An unstable explicit step must fail before changing state.
        with pytest.raises(ValueError, match="Diffusion stability violation"):
            diffusion.forward(state, time=0.0, dt=dt)


class TestAdvectionDiffusion:
    """Test combined AdvectionDiffusion transport."""

    def test_inheritance(self):
        """Verify AdvectionDiffusion inherits from TransportMechanismFunctor."""
        assert issubclass(AdvectionDiffusion, TransportMechanismFunctor)
        assert issubclass(AdvectionDiffusion, FunctorBase)

    def test_initialization(self):
        """Test advection-diffusion initialization."""
        adv_diff = AdvectionDiffusion(
            spatial_dims=1,
            velocity_field=torch.tensor([1.0]),
            diffusivity=1e-5,
            grid_spacing=(0.1,),
            name="test_adv_diff"
        )

        assert adv_diff.name == "test_adv_diff"
        assert hasattr(adv_diff, 'metadata')

    def test_forward_combines_processes(self):
        """Test that forward combines advection and diffusion."""
        adv_diff = AdvectionDiffusion(
            spatial_dims=1,
            velocity_field=torch.tensor([0.5]),
            diffusivity=1e-3,
            grid_spacing=(0.1,)
        )

        # Create Gaussian-like concentration
        x = torch.linspace(-2, 2, 20)
        state = {
            "tracer_concentration": torch.exp(-x**2)
        }

        result = adv_diff.forward(state, time=0.0, dt=0.01)

        # Should have increment output
        assert isinstance(result, dict)

        # Use the declared prognostic state key, not a substring match.
        conc_key = "tracer_concentration"
        assert conc_key in result
        increment = _accumulate_increment(result, key=conc_key)

        # Should show both advection (shift) and diffusion (smoothing)
        assert not torch.allclose(increment, torch.zeros_like(increment))

    def test_peclet_number_calculation(self):
        """Test Peclet number diagnostic calculation."""
        adv_diff = AdvectionDiffusion(
            spatial_dims=1,
            velocity_field=torch.tensor([1.0]),
            diffusivity=0.01,
            grid_spacing=(0.1,)
        )

        state = {
            "tracer_concentration": torch.ones(10)
        }

        result = adv_diff.forward(state, time=0.0, dt=0.01)

        # Check for Peclet number in diagnostics
        # Pe = v*L/D = 1.0*0.1/0.01 = 10
        if "diagnostics.peclet_number" in result:
            pe = result["diagnostics.peclet_number"]
            assert pe > 0
            assert abs(pe - 10.0) < 1.0  # Should be close to 10


class TestTransportIntegration:
    """Integration tests for all transport mechanisms."""

    def test_all_return_flux_dict_format(self):
        """Test that all transport mechanisms return proper FluxDict format."""

        # Create instances of each transport type
        advection = LinearAdvection(
            spatial_dims=1,
            velocity_field=torch.tensor([1.0]),
            grid_spacing=(0.1,)
        )

        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.1,)
        )

        adv_diff = AdvectionDiffusion(
            spatial_dims=1,
            velocity_field=torch.tensor([1.0]),
            diffusivity=1e-5,
            grid_spacing=(0.1,)
        )

        # Test state
        state = {
            "tracer_concentration": torch.ones(10)
        }

        # Test each mechanism
        for transport in [advection, diffusion, adv_diff]:
            result = transport.forward(state, time=0.0, dt=0.01)

            # Should return a dictionary
            assert isinstance(result, dict)

            # Should have at least one non-diagnostic key
            non_diagnostic_keys = [k for k in result.keys() if not k.startswith("diagnostics")]
            assert len(non_diagnostic_keys) > 0

            # State and boundary outputs carry explicit StateManager semantics.
            for key, value in result.items():
                if key == "tracer_concentration":
                    _accumulate_increment(result, key=key)
                elif not key.startswith("diagnostics"):
                    assert isinstance(value, UpdateItem)
                    assert value.mode == UpdateMode.REPLACE
                    assert value.per_timestep is False
                else:
                    assert isinstance(value, torch.Tensor)


def test_imports():
    """Test that all required transport classes can be imported."""
    from relic.scientific.transport.advection import LinearAdvection
    from relic.scientific.transport.diffusion import FickianDiffusion
    from relic.scientific.transport.advection_diffusion import AdvectionDiffusion

    assert LinearAdvection is not None
    assert FickianDiffusion is not None
    assert AdvectionDiffusion is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

