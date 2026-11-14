"""
Transport module refactoring validation tests.

Focus:
- Integration of advection/diffusion operators with Functor bridges.
- Basic mass conservation/finite checks on updates.
- Presence of OperatorFactory wiring.
"""

import torch
import pytest

from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.computation.operators import OperatorFactory
from relic.computation.operators.transport import (
    DiffusionOperator,
    AdvectionOperator,
)
from relic.core.state import UpdateItem, UpdateMode


@pytest.fixture
def sample_state():
    return {
        "tracer_concentration": torch.ones((10, 10, 5), dtype=torch.float32),
    }


def _assert_accumulate_update(updates, key, shape):
    update = updates[key]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    assert update.value.shape == shape
    assert torch.isfinite(update.value).all()


class TestTransportRefactoring:
    def test_linear_advection_operator_factory_integration(self, sample_state):
        advection = LinearAdvection(
            spatial_dims=2,
            name="test_advection",
        )
        linear_state = dict(sample_state)
        linear_state["velocity_field"] = (
            torch.full_like(sample_state["tracer_concentration"], 0.1),
            torch.zeros_like(sample_state["tracer_concentration"]),
        )
        updates = advection.forward(
            state=linear_state,

            time=0.0,
            dt=1.0,
        )
        _assert_accumulate_update(
            updates,
            "tracer_concentration",
            sample_state["tracer_concentration"].shape,
        )
        assert isinstance(advection.operator_factory, OperatorFactory)
        assert isinstance(advection._advection_operator, AdvectionOperator)

    def test_fickian_diffusion_operator_factory_integration(self, sample_state):
        diffusion = FickianDiffusion(
            spatial_dims=2,
            name="test_diffusion",
            diffusivity=1e-4,
            grid_spacing=1.0,
        )
        updates = diffusion.forward(
            state=sample_state,
            time=0.0,
            dt=1.0,
        )
        _assert_accumulate_update(
            updates,
            "tracer_concentration",
            sample_state["tracer_concentration"].shape,
        )
        assert isinstance(diffusion.operator_factory, OperatorFactory)
        assert isinstance(diffusion._diffusion_operator, DiffusionOperator)

    def test_advection_diffusion_operator_factory_integration(self, sample_state):
        advdiff = AdvectionDiffusion(
            spatial_dims=2,
            name="test_advdiff",
            diffusivity=1e-4,
        )
        advdiff_state = dict(sample_state)
        advdiff_state["velocity_field"] = torch.stack(
            (
                torch.full_like(sample_state["tracer_concentration"], 0.1),
                torch.zeros_like(sample_state["tracer_concentration"]),
            ),
            dim=-1,
        )
        updates = advdiff.forward(
            state=advdiff_state,

            time=0.0,
            dt=1.0,
        )
        _assert_accumulate_update(
            updates,
            "tracer_concentration",
            sample_state["tracer_concentration"].shape,
        )
        assert isinstance(advdiff.operator_factory, OperatorFactory)

    def test_mass_conservation_properties(self, sample_state):
        advdiff = AdvectionDiffusion(
            spatial_dims=2,
            name="test_advdiff_mass",
            diffusivity=1e-4,
            boundary_method="periodic",
        )
        advdiff_state = dict(sample_state)
        advdiff_state["velocity_field"] = torch.stack(
            (
                torch.zeros_like(sample_state["tracer_concentration"]),
                torch.zeros_like(sample_state["tracer_concentration"]),
            ),
            dim=-1,
        )
        before = sample_state["tracer_concentration"].sum()
        updates = advdiff.forward(
            state=advdiff_state,

            time=0.0,
            dt=1.0,
        )
        update = updates["tracer_concentration"]
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.ACCUMULATE
        assert update.per_timestep is True
        after = (sample_state["tracer_concentration"] + update.value).sum()
        assert torch.isfinite(after)
        assert torch.allclose(before, after, atol=1e-5)

    def test_parameter_system_integration(self, sample_state):
        # Ensure YAML parameter lookup and canonical state velocity integrate.
        adv = LinearAdvection(
            spatial_dims=2,
            name="test_param_integration",
        )
        linear_state = dict(sample_state)
        linear_state["velocity_field"] = (
            torch.full_like(sample_state["tracer_concentration"], 0.05),
            torch.zeros_like(sample_state["tracer_concentration"]),
        )
        updates = adv.forward(
            state=linear_state,

            time=0.0,
            dt=1.0,
        )
        _assert_accumulate_update(
            updates,
            "tracer_concentration",
            sample_state["tracer_concentration"].shape,
        )
