"""Consistency checks between transport YAML functor_metadata and runtime metadata."""

from __future__ import annotations

from pathlib import Path

import yaml

from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.transport.diffusion import FickianDiffusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSPORT_PARAM_DIR = PROJECT_ROOT / "relic" / "parameters" / "database" / "relic" / "transport"


def _load_yaml(name: str) -> dict:
    path = TRANSPORT_PARAM_DIR / name
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def _assert_functor_metadata_alignment(yaml_name: str, runtime_metadata) -> None:
    payload = _load_yaml(yaml_name)
    block = payload.get("functor_metadata")
    assert isinstance(block, dict), f"{yaml_name} missing top-level functor_metadata block"

    assert block.get("spatial_scale") == runtime_metadata.spatial_scale.value
    assert block.get("variable_kind") == runtime_metadata.variable_kind.value
    assert set(block.get("input_variables", [])) == set(runtime_metadata.input_variables)
    assert set(block.get("output_variables", [])) == set(runtime_metadata.output_variables)
    assert set(block.get("conservation_laws", [])) == set(runtime_metadata.conservation_laws)


def test_advection_yaml_functor_metadata_matches_runtime() -> None:
    functor = LinearAdvection(spatial_dims=1, grid_spacing=[0.1])
    _assert_functor_metadata_alignment("advection.yaml", functor.metadata)


def test_diffusion_yaml_functor_metadata_matches_runtime() -> None:
    functor = FickianDiffusion(spatial_dims=1, diffusivity=1.0e-5, grid_spacing=[0.1])
    _assert_functor_metadata_alignment("diffusion.yaml", functor.metadata)


def test_advection_diffusion_yaml_functor_metadata_matches_runtime() -> None:
    functor = AdvectionDiffusion(spatial_dims=1, diffusivity=1.0e-5, grid_spacing=(0.1,))
    _assert_functor_metadata_alignment("advection_diffusion.yaml", functor.metadata)


def test_numerical_provenance_subfields_are_nested_for_conservation_error_threshold() -> None:
    payload = _load_yaml("numerical.yaml")
    params = payload.get("parameters", {})
    row = params.get("transport.numerical.conservation_error_threshold", {})
    provenance = row.get("provenance")
    assert isinstance(provenance, dict)
    assert "created_by" in provenance
    assert "last_modified" in provenance
    assert "created_by" not in row
    assert "last_modified" not in row


def test_shared_default_boundary_type_is_supported_by_every_transport_consumer() -> None:
    payload = _load_yaml("numerical.yaml")
    params = payload.get("parameters", {})
    row = params.get("transport.numerical.default_boundary_type", {})

    assert set(row.get("allowed_values", [])) == {
        "dirichlet",
        "neumann",
        "open",
        "periodic",
    }
