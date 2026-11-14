from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _state() -> dict[str, torch.Tensor]:
    return {
        "day_of_year": torch.tensor(150.0, dtype=torch.float32),
        "soil_moisture": torch.tensor([[0.24]], dtype=torch.float32),
        "air_temperature": torch.tensor([[295.15]], dtype=torch.float32),
        "aboveground_biomass": torch.tensor([[1.5]], dtype=torch.float32),
        "soil_nitrogen": torch.tensor([[0.01]], dtype=torch.float32),
        "soil_organic_carbon": torch.tensor([[5.0]], dtype=torch.float32),
        "crop_maturity": torch.tensor([[0.6]], dtype=torch.float32),
        "irrigation_flux": torch.tensor([[0.0]], dtype=torch.float32),
        "harvest_flux": torch.tensor([[0.0]], dtype=torch.float32),
        "grazing_flux": torch.tensor([[0.0]], dtype=torch.float32),
        "co2_emission": torch.tensor([[0.0]], dtype=torch.float32),
        "field_capacity": torch.tensor([[0.35]], dtype=torch.float32),
        "wilting_point": torch.tensor([[0.15]], dtype=torch.float32),
        "evapotranspiration_flux": torch.tensor([[3.0e-6]], dtype=torch.float32),
        "precipitation_flux": torch.tensor([[0.0]], dtype=torch.float32),
    }


def test_management_coordinator_declares_canonical_iphs_port_map():
    coordinator = ManagementCoordinator(config={"calendar_type": "noleap"})

    assert coordinator.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert coordinator.runtime_path == "canonical_iphs_management_port_coordinator"

    forms = coordinator.declared_forms()
    assert {form.semantic_name for form in forms} >= {
        "management_priority_state",
        "management_control_projection",
        "management_resource_allocation",
    }

    hodge = coordinator.build_hodge_contract(strict=True)
    audit = hodge.validate(passive=True)
    assert audit["psd"] is True
    assert torch.all(torch.diag(hodge.matrix) >= 0.0)

    ports = coordinator.ports()
    assert len(ports) == 1
    assert ports[0].name == "management_coordination_control_port"
    assert ports[0].port_kind == "management"


def test_management_coordinator_leaves_subordinate_functors_to_simulator():
    coordinator = ManagementCoordinator(config={"calendar_type": "noleap"})

    assert not hasattr(coordinator, "management_modules")
    assert coordinator.get_dependencies() == {
        "required": [],
        "optional": [],
        "lagged": [
            "irrigation_management",
            "harvest_management",
            "grazing_management",
            "fire_management",
        ],
    }


def test_management_coordinator_forward_reports_canonical_runtime():
    coordinator = ManagementCoordinator(config={"calendar_type": "noleap"})

    outputs = coordinator.forward(_state(), time=0.0, dt=86400.0)

    assert set(outputs) == set(coordinator.metadata.output_variables)
    assert {
        "irrigation_flux",
        "harvest_flux",
        "grazing_flux",
        "co2_emission",
        "total_fire_emissions",
        "irrigation_rate",
        "fertilizer_rate",
        "energy_consumption",
    }.isdisjoint(outputs)
    for item in outputs.values():
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.source_functor == coordinator.name
    assert (
        outputs[
            "diagnostics.management_coordinator.canonical_iphs_runtime"
        ].value.item()
        == 1.0
    )
    assert torch.isfinite(
        outputs["diagnostics.coordinator_mass_balance_error"].value
    ).all()
    assert torch.all(outputs["total_irrigation_flux"].value == 0.0)


def test_management_coordinator_fails_fast_when_a_canonical_effect_is_missing(
    monkeypatch,
):
    coordinator = ManagementCoordinator(config={"calendar_type": "noleap"})
    original = coordinator.compute_coordination_decisions

    def _without_tillage_intensity(state):
        decisions = original(state)
        decisions["management_decisions"].pop("tillage_intensity")
        return decisions

    monkeypatch.setattr(
        coordinator,
        "compute_coordination_decisions",
        _without_tillage_intensity,
    )

    with pytest.raises(KeyError, match="tillage_intensity"):
        coordinator.forward(_state(), time=0.0, dt=86400.0)


def test_management_coordinator_source_has_no_parallel_runtime_terms():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "management"
        / "coordinator.py"
    ).read_text(encoding="utf-8")

    forbidden_terms = (
        "fallback",
        "legacy_mode",
        "use_legacy",
        "enable_legacy",
        "compute_mode",
        "zero_flux_fallback",
    )
    for term in forbidden_terms:
        assert term not in source


def test_management_coordinator_phase8_certificate_is_valid():
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_management_coordinator_management_coordinator.yaml"
    )

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert (
        certificate.dependency_interconnection_map["direct_functor_calls"]
        == "none_no_direct_functor_calls"
    )


def test_management_coordinator_internal_methods_read_the_state_only():
    """Item 113: forward passed its own state a second time as an ``environment``
    mapping that no method read (only _infer_device iterated it, after the state that
    already holds tensors). Scientific inputs come from the declared state and the
    calendar from the forward keyword, so no method takes an environment mapping."""
    import inspect

    for name in (
        "calculate_priority_weights",
        "calculate_interaction_effects",
        "optimize_management_practices",
        "compute_coordination_decisions",
        "coordinate_management_activities",
        "_update_environmental_context",
        "_assess_management_needs",
        "_coordinate_if_due",
        "_infer_device",
    ):
        parameters = inspect.signature(getattr(ManagementCoordinator, name)).parameters
        assert "environment" not in parameters, name
        assert "context" not in parameters, name
    assert list(inspect.signature(ManagementCoordinator._infer_device).parameters) == [
        "self",
        "state",
    ]
    assert list(inspect.signature(ManagementCoordinator.forward).parameters) == [
        "self",
        "state",
        "time",
        "dt",
        "calendar_type",
    ]
    coordinator = ManagementCoordinator(config={"calendar_type": "noleap"})
    with pytest.raises(ValueError, match="tensor"):
        coordinator._infer_device({})


def test_management_coordinator_certificate_records_the_forward_signature():
    import yaml

    certificate = yaml.safe_load(
        (
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "anthroposphere_management_coordinator_management_coordinator.yaml"
        ).read_text(encoding="utf-8")
    )
    assert certificate["interface_map"]["forward_signature"] == (
        "forward(self, state, time, dt, calendar_type=None)"
    )
