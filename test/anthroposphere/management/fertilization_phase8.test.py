"""Phase 8 contract tests for the fertilization management boundary."""

from pathlib import Path

import torch

from relic.computation.operators.iphs_certificate import (
    load_iphs_equivalence_certificate,
)
from relic.core.state import UpdateItem
from relic.scientific.anthroposphere.management.fertilization import (
    FertilizationManagement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CERTIFICATE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "anthroposphere_management_fertilization_fertilization_management.yaml"
)


def _value(value):
    return value.value if isinstance(value, UpdateItem) else value


def test_fertilization_declares_canonical_iphs_boundary_ports():
    manager = FertilizationManagement(config={"calendar_type": "noleap"})

    assert manager.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert manager.runtime_path == "canonical_iphs_fertilization_boundary_port"
    assert {form.semantic_name for form in manager.declared_forms()} == {
        "fertilization_nitrogen_boundary_flow",
        "fertilization_phosphorus_boundary_flow",
    }
    assert manager.build_hodge_terms() == ()
    assert {port.name for port in manager.ports()} == {
        "fertilization_nitrogen_input_port",
        "fertilization_phosphorus_input_port",
    }
    assert manager.reversible_structure({})["J"].numel() == 0
    assert manager.dissipative_structure({})["M"].numel() == 0


def test_fertilization_forward_reports_canonical_runtime():
    manager = FertilizationManagement(config={"calendar_type": "noleap"})
    outputs = manager.forward(
        {"day_of_year": torch.tensor(120.0)},
        time=0.0,
        dt=3600.0,
    )

    assert set(outputs) == set(manager.metadata.output_variables)
    assert outputs[
        "diagnostics.fertilization.canonical_iphs_runtime"
    ].item() == 1.0
    assert all(torch.isfinite(_value(value)).all() for value in outputs.values())


def test_fertilization_compute_uses_forward_contract():
    manager = FertilizationManagement(config={"calendar_type": "noleap"})
    state = {"day_of_year": torch.tensor(120.0)}

    outputs = manager.compute(state, time=0.0, dt=3600.0)

    assert set(outputs) == set(manager.metadata.output_variables)
    assert isinstance(outputs["fertilizer_n_flux"], UpdateItem)
    assert isinstance(outputs["fertilizer_p_flux"], UpdateItem)
    assert isinstance(outputs["available_phosphorus"], UpdateItem)


def test_fertilization_source_has_no_parallel_reaction_runtime():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "management"
        / "fertilization.py"
    ).read_text(encoding="utf-8")

    forbidden_terms = (
        "advance_reaction_network(",
        "_compute_rate_core",
        "_calculate_dndc_denitrification",
        "_calculate_nutrient_release",
        "legacy_mode",
        "use_legacy",
        "zero_flux_fallback",
    )
    for term in forbidden_terms:
        assert term not in source


def test_fertilization_phase8_certificate_is_valid_and_non_strict():
    manager = FertilizationManagement(config={"calendar_type": "noleap"})
    certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.canonical_role == "port"
    assert certificate.iphs_level == 2
    assert certificate.input_variables == tuple(manager.metadata.input_variables)
    assert certificate.output_variables == tuple(manager.metadata.output_variables)
    assert certificate.interface_map["bridge_class"] == "TransportMechanismFunctor"
    assert certificate.interface_map["abstraction_contract"] == "TransportMechanism"
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert "G_h u_h" in certificate.state_advancement_map["state_advancer"]
    assert certificate.dependency_interconnection_map["direct_functor_calls"] == (
        "none; forward reads day_of_year and schedule state, calls only local "
        "event-scheduler helpers, and routes elemental N and P through declared "
        "StateManager ports"
    )
