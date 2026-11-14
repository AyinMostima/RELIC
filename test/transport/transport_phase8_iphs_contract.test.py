"""Phase 8 IPHS contracts for generic transport functors."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor, VariableKind
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.transport.diffusion import FickianDiffusion


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


def test_fickian_diffusion_satisfies_phase8_iphs_protocol() -> None:
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=1.0e-5,
        grid_spacing=[0.1],
    )

    forms = functor.declared_forms()
    hodge_terms = functor.build_hodge_terms()
    dissipative = functor.dissipative_structure({})
    ports = functor.ports()

    assert isinstance(functor, IPHSDECFunctor)
    assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert "transport.diffusion.diffusivity" in functor.required_parameters()
    assert {form.semantic_name for form in forms} == {
        "tracer_concentration",
        "tracer_gradient",
        "diffusive_tracer_flux",
    }
    assert len(hodge_terms) == 1
    assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) > 0.0)
    assert {port.name for port in ports} == {"diffusive_boundary_flux"}
    boundary_port = ports[0]
    assert boundary_port.effort_var == "tracer_concentration_boundary_state"
    assert boundary_port.flow_var == "tracer_concentration_diffusive_boundary_flux"
    assert boundary_port.positive_power_convention == "out_of_system"


def test_fickian_diffusion_exposes_structural_audits_without_fake_runtime_ports() -> None:
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=1.0e-5,
        grid_spacing=[0.1],
        non_negative=False,
        track_conservation=False,
    )
    state = {
        "tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        "source_term": torch.zeros(3, dtype=torch.float32),
    }

    energy_terms = functor.energy_terms(state)
    entropy_terms = functor.entropy_terms(state)
    outputs = functor.forward(state, time=0.0, dt=1.0)

    assert torch.is_tensor(energy_terms["storage_functional"])
    assert torch.is_tensor(energy_terms["effort"])
    assert torch.is_tensor(entropy_terms["dissipation_rate"])
    assert entropy_terms["dissipation_rate"] >= 0.0
    assert set(outputs) == {
        "tracer_concentration",
        "tracer_concentration_diffusive_boundary_flux",
        "diagnostics.diffusion_stability",
    }
    assert not {
        "diagnostics.transport.energy_storage",
        "diagnostics.transport.dissipation_rate",
        "diagnostics.transport.dH_dt_audit",
        "diagnostics.transport.port_power_residual",
    } & set(outputs)


def test_fickian_structural_dissipation_uses_harmonic_face_diffusivity() -> None:
    diffusivity = torch.tensor([[1.0, 4.0, 1.0]], dtype=torch.float64)
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=diffusivity, diffusivity_layout="scalar_field",
        grid_spacing=[1.0],
        track_conservation=False,
    )
    state = {
        "tracer_concentration": torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64
        )
    }

    dissipation = functor.entropy_terms(state)["dissipation_rate"]

    torch.testing.assert_close(
        dissipation,
        torch.tensor(3.2, dtype=torch.float64),
    )


@pytest.mark.parametrize("boundary,boundary_value,expected", [
    ("periodic", None, 0.5), ("dirichlet", 0.0, 1.0),
    ("dirichlet", 0.5, 0.25), ("dirichlet", 2.0, 1.0),
])
def test_fickian_norm_audit_matches_fv_boundary_balance(boundary, boundary_value, expected):
    functor = FickianDiffusion(
        spatial_dims=1, diffusivity=1.0, grid_spacing=[2.0],
        boundary_method=boundary, non_negative=False, track_conservation=False,
    )
    concentration = torch.tensor([1.0, 0.0, 0.0] if boundary == "periodic" else [1.0, 1.0, 1.0], dtype=torch.float64)
    state = {functor.tracer_key: concentration}
    if boundary_value is not None:
        state[functor.boundary_state_key] = torch.full_like(concentration, boundary_value)
    sigma = functor.entropy_terms(state)["dissipation_rate"]
    torch.testing.assert_close(sigma, torch.tensor(expected, dtype=torch.float64))
    updates = functor.forward(state, time=0.0, dt=0.125)
    rate = updates[functor.tracer_key].value / 0.125
    outward_power = 0.0
    if boundary_value is not None:
        outward_power = boundary_value * updates[functor.boundary_flux_key].value.sum() / 2.0
    torch.testing.assert_close((concentration * rate).sum() + sigma + outward_power, torch.tensor(0.0, dtype=torch.float64))


@pytest.mark.parametrize("batched", [False, True])
def test_fickian_cell_diffusivity_audit_matches_rate_and_parameter_gradient(batched):
    coefficients = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    functor = FickianDiffusion(
        spatial_dims=1, diffusivity=coefficients, diffusivity_layout="scalar_field", grid_spacing=[2.0],
        boundary_method="zero", non_negative=False, track_conservation=False,
    )
    concentration = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    if batched:
        concentration = torch.stack([concentration, 2.0 * concentration])
    state = {functor.tracer_key: concentration}
    sigma = functor.entropy_terms(state)["dissipation_rate"]
    factor = 5.0 if batched else 1.0
    torch.testing.assert_close(sigma, torch.tensor(factor * (4.0 / 3.0 + 48.0 / 5.0) / 4.0, dtype=torch.float64))
    updates = functor.forward(state, time=0.0, dt=0.125)
    torch.testing.assert_close((concentration * updates[functor.tracer_key].value / 0.125).sum(), -sigma)
    sigma.backward()
    torch.testing.assert_close(coefficients.grad, factor * torch.tensor([2.0 / 9.0, 349.0 / 450.0, 8.0 / 25.0], dtype=torch.float64))


def test_fickian_extrapolated_norm_audit_includes_boundary_half_cells():
    functor = FickianDiffusion(
        spatial_dims=1, diffusivity=1.0, grid_spacing=[2.0],
        boundary_method="extrapolate", non_negative=False, track_conservation=False,
    )
    concentration = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    state = {functor.tracer_key: concentration}
    sigma = functor.entropy_terms(state)["dissipation_rate"]
    torch.testing.assert_close(sigma, torch.tensor(0.75, dtype=torch.float64))
    updates = functor.forward(state, time=0.0, dt=0.125)
    outward_power = (torch.tensor([1.5, 4.5], dtype=torch.float64) * updates[functor.boundary_flux_key].value).sum() / 2.0
    torch.testing.assert_close((concentration * updates[functor.tracer_key].value / 0.125).sum() + sigma + outward_power, torch.tensor(0.0, dtype=torch.float64))


def test_fickian_closed_boundary_query_accepts_abstract_contract_keywords():
    functor = FickianDiffusion(spatial_dims=1, diffusivity=1.0, grid_spacing=[2.0], boundary_method="neumann")
    field = torch.tensor([1.0, 2.0])
    flux = functor.boundary_conditions(
        boundary_name="x_min", field_values=field, normal_vector=torch.tensor([-1.0]),
        boundary_type="neumann", boundary_value=0.0, use_bc_manager=False,
    )
    torch.testing.assert_close(flux, torch.zeros_like(field))


def test_fickian_closed_boundary_query_rejects_unrepresented_exchange():
    functor = FickianDiffusion(spatial_dims=1, diffusivity=1.0, grid_spacing=[2.0], boundary_method="neumann")
    with pytest.raises((ValueError, RuntimeError), match="forward|boundary"):
        functor.boundary_conditions(
            boundary_name="x_min", field_values=torch.tensor([1.0, 2.0]), normal_vector=torch.tensor([-1.0]),
            boundary_type="dirichlet", boundary_value=1.0, use_bc_manager=False,
        )


@pytest.mark.parametrize("n_dofs", [True, 1.5, "2"])
def test_fickian_hodge_rejects_non_integer_dof_declarations(n_dofs):
    functor = FickianDiffusion(spatial_dims=1, diffusivity=1.0, grid_spacing=[2.0])
    with pytest.raises(TypeError, match="integer"):
        functor.build_hodge_contract(n_dofs=n_dofs)


def test_fickian_hodge_rejects_unmapped_directional_coefficients():
    functor = FickianDiffusion(
        spatial_dims=2, diffusivity=torch.tensor([1.0, 2.0], dtype=torch.float64), grid_spacing=[1.0, 1.0],
        diffusivity_layout="directional",
    )
    with pytest.raises(ValueError, match="coefficients|DOF|dofs"):
        functor.build_hodge_contract(n_dofs=3)


@pytest.mark.parametrize("values,n_dofs", [([1.0e-50], 3), ([1.0, 2.0], 2)])
def test_fickian_hodge_preserves_supplied_coefficients_precision_and_gradient(values, n_dofs):
    coefficients = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    functor = FickianDiffusion(
        spatial_dims=1, diffusivity=coefficients, diffusivity_layout="scalar_field", grid_spacing=[1.0]
    )
    matrix = functor.build_hodge_contract(n_dofs=n_dofs).matrix
    assert matrix.dtype == torch.float64
    expected = coefficients.expand(n_dofs) if len(values) == 1 else coefficients
    torch.testing.assert_close(matrix.diag(), expected, rtol=0.0, atol=0.0)
    matrix.sum().backward()
    torch.testing.assert_close(coefficients.grad, torch.full_like(coefficients, n_dofs if len(values) == 1 else 1))


def test_fickian_directional_structural_dissipation_preserves_gradient() -> None:
    directional_diffusivity = torch.tensor(
        [0.1, 0.4],
        dtype=torch.float64,
        requires_grad=True,
    )
    functor = FickianDiffusion(
        spatial_dims=2,
        diffusivity=directional_diffusivity, diffusivity_layout="directional",
        grid_spacing=[1.0, 1.0],
        track_conservation=False,
    )
    state = {
        "tracer_concentration": torch.tensor(
            [[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]],
            dtype=torch.float64,
        )
    }

    dissipation = functor.entropy_terms(state)["dissipation_rate"]
    dissipation.backward()

    assert directional_diffusivity.grad is not None
    assert torch.isfinite(directional_diffusivity.grad).all()
    assert torch.all(directional_diffusivity.grad >= 0.0)


def test_fickian_diffusion_phase8_certificate_is_valid() -> None:
    certificate = load_iphs_equivalence_certificate(
        _certificate_path("transport_diffusion_fickian_diffusion.yaml")
    )
    audit = certificate.validate()
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=1.0e-5,
        grid_spacing=[0.1],
    )

    assert audit["passed"] is True
    assert certificate.process_id == "relic.scientific.transport.diffusion.FickianDiffusion"
    assert certificate.canonical_role == "transport"
    assert certificate.iphs_level == 3
    assert certificate.iphs_level_evidence["level_3_limit"]
    assert audit["runtime_evidence_status"] == "descriptive_only"
    assert set(certificate.input_variables) == set(functor.metadata.input_variables)
    assert set(certificate.output_variables) == set(functor.metadata.output_variables)
    assert set(certificate.parameter_keys) == set(functor.required_parameters())
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False


def test_linear_advection_satisfies_phase8_iphs_protocol() -> None:
    functor = LinearAdvection(
        spatial_dims=1,
        velocity_field=[torch.ones(1, 8, dtype=torch.float32)],
        grid_spacing=[0.1],
    )

    forms = functor.declared_forms()
    hodge_terms = functor.build_hodge_terms()
    dissipative = functor.dissipative_structure({})
    ports = functor.ports()

    assert isinstance(functor, IPHSDECFunctor)
    assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert functor.metadata.variable_kind is VariableKind.STATE
    expected_parameter_keys = {
        "transport.advection.grid_spacing",
        "transport.advection.default_velocity_magnitude",
        "transport.advection.inflow_boundary_type",
        "transport.advection.outflow_boundary_type",
        "transport.numerical.cfl_advection_max",
        "transport.numerical.track_conservation",
        "transport.numerical.enforce_non_negative",
        "transport.numerical.default_boundary_type",
    }
    assert set(functor.required_parameters()) == expected_parameter_keys
    assert set(functor.metadata.parameters) == expected_parameter_keys
    assert {form.semantic_name for form in forms} == {
        "tracer_concentration",
        "velocity_field",
        "advective_tracer_flux",
    }
    assert len(hodge_terms) == 1
    assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
    assert {port.name for port in ports} == {"advective_boundary_flux"}


def test_linear_advection_phase8_certificate_is_valid() -> None:
    certificate = load_iphs_equivalence_certificate(
        _certificate_path("transport_advection_linear_advection.yaml")
    )
    audit = certificate.validate()

    assert audit["passed"] is True
    assert certificate.process_id == "relic.scientific.transport.advection.LinearAdvection"
    assert certificate.canonical_role == "transport"


def test_advection_diffusion_satisfies_phase8_iphs_protocol() -> None:
    functor = AdvectionDiffusion(
        spatial_dims=1,
        diffusivity=1.0e-5,
        velocity_field=[torch.ones(1, 8, dtype=torch.float32)],
        grid_spacing=(0.1,),
    )

    forms = functor.declared_forms()
    hodge_terms = functor.build_hodge_terms()
    dissipative = functor.dissipative_structure({})
    ports = functor.ports()

    assert isinstance(functor, IPHSDECFunctor)
    assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    expected_parameter_keys = {
        "transport.advection.default_velocity_magnitude",
        "transport.advection_diffusion.grid_spacing",
        "transport.advection_diffusion.splitting_method",
        "transport.diffusion.diffusivity",
        "transport.numerical.cfl_advection_max",
        "transport.numerical.cfl_diffusion_max",
        "transport.numerical.default_boundary_type",
        "transport.numerical.enforce_non_negative",
        "transport.numerical.track_conservation",
    }
    assert set(functor.required_parameters()) == expected_parameter_keys
    assert {form.semantic_name for form in forms} == {
        "tracer_concentration",
        "velocity_field",
        "tracer_gradient",
        "total_transport_flux",
    }
    assert len(hodge_terms) == 2
    assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
    assert {port.name for port in ports} == {
        "advection_diffusion_boundary_flux",
    }
    boundary_port = ports[0]
    assert boundary_port.effort_var == "tracer_concentration_boundary_state"
    assert (
        boundary_port.flow_var
        == "tracer_concentration_advection_diffusion_boundary_flux"
    )
    assert boundary_port.positive_power_convention == "out_of_system"


def test_advection_diffusion_hodge_spaces_match_complex_dofs() -> None:
    functor = AdvectionDiffusion(
        spatial_dims=1,
        diffusivity=1.0e-5,
        velocity_field=[torch.ones(1, 3, dtype=torch.float32)],
        grid_spacing=(0.1,),
    )

    hodge_terms = functor.build_hodge_terms(
        complex_bundle=SimpleNamespace(n_dofs=3)
    )

    for hodge in hodge_terms:
        assert hodge.domain.n_dofs == 3
        assert hodge.codomain.n_dofs == 3
        assert hodge.matrix.shape == (3, 3)


def test_advection_diffusion_phase8_certificate_is_valid() -> None:
    certificate = load_iphs_equivalence_certificate(
        _certificate_path("transport_advection_diffusion_advection_diffusion.yaml")
    )
    audit = certificate.validate()

    assert audit["passed"] is True
    assert (
        certificate.process_id
        == "relic.scientific.transport.advection_diffusion.AdvectionDiffusion"
    )
    assert certificate.canonical_role == "transport"
