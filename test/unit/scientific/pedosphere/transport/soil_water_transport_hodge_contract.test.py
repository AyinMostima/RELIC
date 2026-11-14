"""Tests for SoilWaterTransportFunctor DEC/Hodge structural contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.scientific.pedosphere.transport.SoilWaterTransportFunctor import (
    SoilWaterTransportFunctor,
)


def _functor() -> SoilWaterTransportFunctor:
    return SoilWaterTransportFunctor(device=torch.device("cpu"))


def test_soil_water_transport_declares_dec_form_metadata():
    functor = _functor()

    metadata = functor.get_dec_form_metadata(n_dofs=2)

    assert metadata["soil_moisture"]["form_degree"] == 3
    assert metadata["soil_moisture"]["extensivity"] == "intensive"
    assert metadata["soil_moisture"]["semantic_name"] == "soil_moisture_profile"
    assert metadata["water_flux"]["form_degree"] == 2
    assert metadata["water_flux"]["physical_domain"] == "pedosphere"


def test_soil_water_transport_builds_positive_hodge_contract():
    functor = _functor()

    hodge = functor.build_hodge_contract(
        n_dofs=2,
        strict=True,
        conductivity=torch.tensor([1.0, 2.0]),
    )

    assert hodge.name == "soil_water_transport_hodge"
    assert torch.allclose(
        torch.diag(hodge.matrix),
        torch.tensor([1.0, 2.0], dtype=hodge.matrix.dtype, device=hodge.matrix.device),
    )


def test_soil_water_transport_negative_conductivity_fails_strict_hodge_contract():
    functor = _functor()

    with pytest.raises(ValueError, match="positive semidefinite"):
        functor.build_hodge_contract(
            n_dofs=2,
            strict=True,
            conductivity=torch.tensor([1.0, -0.1]),
        )


def test_resolved_hodge_parameters_do_not_mutate_functor_state():
    functor = _functor()
    original = torch.as_tensor(functor.saturated_conductivity).clone()
    original_builder = functor.build_hodge_contract

    def _guarded_builder(*args, **kwargs):
        assert torch.equal(torch.as_tensor(functor.saturated_conductivity), original)
        return original_builder(*args, **kwargs)

    functor.build_hodge_contract = _guarded_builder

    (hodge,) = functor.build_hodge_terms(
        resolved_parameters={
            "pedosphere.hydraulic.saturated_conductivity": torch.tensor(2.0),
        }
    )

    diagonal = torch.diag(hodge.matrix)
    assert torch.allclose(diagonal, torch.full_like(diagonal, 2.0))
    assert torch.equal(torch.as_tensor(functor.saturated_conductivity), original)


def test_soil_water_transport_satisfies_phase8_iphs_protocol():
    functor = _functor()

    forms = functor.declared_forms()
    hodge_terms = functor.build_hodge_terms()
    dissipative = functor.dissipative_structure({})
    ports = functor.ports()

    assert isinstance(functor, IPHSDECFunctor)
    assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert "pedosphere.hydraulic.saturated_conductivity" in functor.required_parameters()
    assert {form.semantic_name for form in forms} == {
        "soil_moisture_profile",
        "hydraulic_head_gradient",
        "darcy_water_flux",
    }
    assert len(hodge_terms) == 1
    assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
    assert {port.name for port in ports} == {
        "surface_water_flux_boundary",
        "bottom_drainage_boundary",
    }


def test_soil_water_transport_phase8_certificate_is_valid():
    functor = _functor()
    certificate_path = (
        Path(__file__).resolve().parents[5]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "pedosphere_transport_soil_water_transport_functor_soil_water_transport_functor.yaml"
    )

    certificate = load_iphs_equivalence_certificate(certificate_path)
    audit = certificate.validate()

    assert audit["passed"] is True
    assert certificate.process_id.endswith("SoilWaterTransportFunctor.SoilWaterTransportFunctor")
    assert certificate.canonical_role == "transport"
    assert certificate.input_variables == tuple(functor.metadata.input_variables)
    assert certificate.output_variables == tuple(functor.metadata.output_variables)
    assert certificate.parameter_keys == functor.required_parameters()
    assert "REPLACE" in certificate.interface_map["state_write_mode"]
    assert "per_timestep=false" in certificate.interface_map["state_write_mode"]
    assert "full state" in certificate.state_map["soil_moisture"]


def test_structural_layout_has_no_literal_fallback_and_one_parameter_order():
    """Round 47: n_dofs is explicit, the conductivity support is the YAML-loaded value
    (no literal second source), and the certificate order is the metadata order."""
    import inspect

    parameter = inspect.signature(SoilWaterTransportFunctor.get_dec_form_metadata).parameters["n_dofs"]
    assert parameter.default is inspect.Parameter.empty
    source = inspect.getsource(SoilWaterTransportFunctor._structural_n_dofs)
    assert 'getattr(self, "saturated_conductivity"' not in source
    functor = _functor()
    assert functor.required_parameters() == tuple(functor.metadata.parameters)
