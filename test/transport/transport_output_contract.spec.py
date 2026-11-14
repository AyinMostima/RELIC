"""Output contract checks for transport functor metadata vs runtime outputs."""

from __future__ import annotations

from typing import Iterable

import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.transport.diffusion import FickianDiffusion


def _matches_declared(output_key: str, declared: Iterable[str]) -> bool:
    for item in declared:
        if item == output_key:
            return True
        if item.endswith("*") and output_key.startswith(item[:-1]):
            return True
    return False


def _assert_output_contract(runtime_outputs: dict[str, torch.Tensor], declared: Iterable[str]) -> None:
    unexpected = [key for key in runtime_outputs if not _matches_declared(key, declared)]
    assert not unexpected, f"Undeclared output keys: {unexpected}"


def test_linear_advection_output_keys_covered_by_metadata() -> None:
    functor = LinearAdvection(
        spatial_dims=1,
        grid_spacing=[0.1],
        boundary_method="periodic",
    )
    functor.set_velocity_field([torch.zeros(1, 16, dtype=torch.float32)])
    functor.track_conservation = True
    state = {"tracer_concentration": torch.linspace(0.0, 1.0, 16).unsqueeze(0)}

    outputs = functor.forward(state=state, time=0.0, dt=0.01)
    expected_metadata = {
        "tracer_concentration",
        "tracer_concentration_advective_boundary_flux",
        "diagnostics.cfl_number",
        "diagnostics.conservation_error",
    }
    expected_runtime = {
        "tracer_concentration",
        "diagnostics.cfl_number",
        "diagnostics.conservation_error",
    }
    assert set(functor.metadata.output_variables) == expected_metadata
    assert set(functor.metadata.optional_output_variables) == {
        "tracer_concentration_advective_boundary_flux",
        "diagnostics.conservation_error",
    }
    assert set(outputs) == expected_runtime

    tracer_update = outputs["tracer_concentration"]
    assert isinstance(tracer_update, UpdateItem)
    assert tracer_update.mode == UpdateMode.ACCUMULATE
    assert tracer_update.per_timestep is True


def test_fickian_diffusion_output_keys_covered_by_metadata() -> None:
    functor = FickianDiffusion(spatial_dims=1, diffusivity=1.0e-5, grid_spacing=[0.1])
    functor.track_conservation = True
    state = {
        "tracer_concentration": torch.linspace(0.0, 1.0, 16).unsqueeze(0),
        "source_term": torch.zeros(1, 16, dtype=torch.float32),
    }

    outputs = functor.forward(state=state, time=0.0, dt=0.01)
    assert set(functor.metadata.output_variables) == {
        "tracer_concentration",
        "tracer_concentration_diffusive_boundary_flux",
        "diagnostics.diffusion_stability",
        "diagnostics.conservation_error",
    }
    assert set(functor.metadata.optional_output_variables) == {
        "tracer_concentration_diffusive_boundary_flux",
        "diagnostics.conservation_error",
    }
    assert set(outputs) == {
        "tracer_concentration",
        "tracer_concentration_diffusive_boundary_flux",
        "diagnostics.diffusion_stability",
        "diagnostics.conservation_error",
    }
    _assert_output_contract(outputs, functor.metadata.output_variables)

    tracer_update = outputs["tracer_concentration"]
    assert isinstance(tracer_update, UpdateItem)
    assert tracer_update.mode == UpdateMode.ACCUMULATE
    assert tracer_update.per_timestep is True

    boundary_update = outputs["tracer_concentration_diffusive_boundary_flux"]
    assert isinstance(boundary_update, UpdateItem)
    assert boundary_update.mode == UpdateMode.REPLACE
    assert boundary_update.per_timestep is False
    assert torch.count_nonzero(boundary_update.value).item() == 0


def test_advection_diffusion_output_keys_covered_by_metadata() -> None:
    velocity = [torch.zeros(1, 12, dtype=torch.float32)]
    functor = AdvectionDiffusion(
        spatial_dims=1,
        diffusivity=1.0e-5,
        velocity_field=velocity,
        grid_spacing=(0.1,),
        track_peclet=True,
        boundary_method="periodic",
    )
    functor.track_conservation = True
    state = {
        "tracer_concentration": torch.linspace(0.0, 1.0, 12).unsqueeze(0),
        "source_term": torch.zeros(1, 12, dtype=torch.float32),
    }

    outputs = functor.forward(state=state, time=0.0, dt=0.01)
    assert set(functor.metadata.output_variables) == {
        "tracer_concentration",
        "tracer_concentration_advection_diffusion_boundary_flux",
        "diagnostics.cfl_number",
        "diagnostics.diffusion_stability",
        "diagnostics.peclet_number",
        "diagnostics.conservation_error",
    }
    assert set(functor.metadata.optional_output_variables) == {
        "tracer_concentration_advection_diffusion_boundary_flux",
        "diagnostics.peclet_number",
        "diagnostics.conservation_error",
    }
    assert set(outputs) == {
        "tracer_concentration",
        "diagnostics.cfl_number",
        "diagnostics.diffusion_stability",
        "diagnostics.peclet_number",
        "diagnostics.conservation_error",
    }
    _assert_output_contract(outputs, functor.metadata.output_variables)
