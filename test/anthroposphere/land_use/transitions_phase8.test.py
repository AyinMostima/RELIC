from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions, LandUseType


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _state() -> dict[str, torch.Tensor]:
    fractions = torch.zeros((len(LandUseType),), dtype=torch.float32)
    fractions[0] = 0.25
    fractions[1] = 0.15
    fractions[2] = 0.20
    fractions[3] = 0.05
    fractions[4] = 0.15
    fractions[5] = 0.10
    fractions[6] = 0.05
    fractions[7] = 0.03
    fractions[8] = 0.02
    return {
        "land_use_fraction": fractions,
        "time_index": torch.tensor(0, dtype=torch.int64),
        "gdp_per_capita": torch.tensor(25000.0, dtype=torch.float32),
        "population_density": torch.tensor(150.0, dtype=torch.float32),
        "policy_intensity": torch.tensor(0.2, dtype=torch.float32),
        "market_accessibility": torch.tensor(0.4, dtype=torch.float32),
    }


def _unwrap(value):
    return value.value if hasattr(value, "value") else value


def test_land_use_transitions_declares_canonical_iphs_ctmc_chain_map():
    transitions = LandUseTransitions()

    assert transitions.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert transitions.runtime_path == "canonical_iphs_land_use_ctmc_chain_map"

    forms = transitions.declared_forms()
    assert {form.semantic_name for form in forms} >= {
        "land_use_fraction_state",
        "land_use_transition_flux",
        "land_use_transition_port_flow",
    }

    hodge = transitions.build_hodge_contract(strict=True)
    audit = hodge.validate(passive=True)
    assert audit["psd"] is True
    assert torch.all(torch.diag(hodge.matrix) >= 0.0)

    ports = transitions.ports()
    assert len(ports) == 1
    assert ports[0].name == "land_use_transition_area_fraction_port"
    assert ports[0].port_kind == "management"


def test_land_use_transitions_hodge_rejects_invalid_generator_rates():
    transitions = LandUseTransitions()
    transitions.transition_matrix[0, 1] = -0.1

    with pytest.raises(ValueError, match="non-negative"):
        transitions.build_hodge_contract(strict=True)


def test_land_use_transitions_forward_reports_canonical_runtime():
    transitions = LandUseTransitions()

    output = transitions.forward(_state(), time=0.0, dt=86400.0)

    diagnostic = _unwrap(
        output["diagnostics.land_use_transitions.canonical_iphs_runtime"]
    )
    assert diagnostic.item() == 1.0
    fractions = _unwrap(output["land_use_fraction"])
    assert torch.isclose(fractions.sum(), torch.tensor(1.0, dtype=fractions.dtype), atol=1e-6)
    delta = _unwrap(output["diagnostics.land_use_transitions.delta_fraction"])
    assert torch.isfinite(delta).all()
    assert torch.isclose(delta.sum(), torch.tensor(0.0, dtype=delta.dtype), atol=1e-6)


def test_land_use_transitions_rejects_missing_fraction_state():
    transitions = LandUseTransitions()

    with pytest.raises(KeyError, match="land_use_fraction"):
        transitions.forward({}, time=0.0, dt=86400.0)


def test_land_use_transitions_source_has_no_parallel_runtime_terms():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "land_use"
        / "transitions.py"
    ).read_text(encoding="utf-8")

    forbidden_terms = (
        "fallback",
        "legacy_mode",
        "use_legacy",
        "enable_legacy",
        "compute_mode",
        "enable_simplified_mode",
    )
    for term in forbidden_terms:
        assert term not in source


def test_land_use_transitions_scale_changes_require_explicit_chain_maps():
    transitions = LandUseTransitions()
    data = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
        transitions.upscale(data)
    with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
        transitions.downscale(data)
    with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
        transitions.resolution_converter(data, 1000.0, 2000.0)
    with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
        transitions.resolution_converter(data, 2000.0, 1000.0)


def test_land_use_transitions_uses_bound_scale_chain_maps():
    calls: list[str] = []

    def upscale_bridge(data):
        calls.append("upscale")
        return data.reshape(-1, 2).mean(dim=1)

    def downscale_bridge(data):
        calls.append("downscale")
        return data.repeat_interleave(2)

    transitions = LandUseTransitions(
        upscale_bridge=upscale_bridge,
        downscale_bridge=downscale_bridge,
    )
    fine = torch.tensor([0.1, 0.3, 0.2, 0.4], dtype=torch.float32)
    coarse = torch.tensor([0.2, 0.3], dtype=torch.float32)

    upscaled = transitions.upscale(fine)
    downscaled = transitions.downscale(coarse)
    converted_up = transitions.resolution_converter(
        fine, 1000.0, 2000.0, method="conservative"
    )
    converted_down = transitions.resolution_converter(
        coarse, 2000.0, 1000.0, method="fraction_weighted"
    )

    assert calls == ["upscale", "downscale", "upscale", "downscale"]
    torch.testing.assert_close(
        upscaled, torch.tensor([0.2, 0.3], dtype=fine.dtype)
    )
    torch.testing.assert_close(
        downscaled,
        torch.tensor([0.2, 0.2, 0.3, 0.3], dtype=coarse.dtype),
    )
    torch.testing.assert_close(converted_up, upscaled)
    torch.testing.assert_close(converted_down, downscaled)
    assert torch.any(upscaled != 0.0)
    assert torch.any(downscaled != 0.0)


def test_land_use_transitions_fraction_weighted_refinement_has_no_synthetic_grid():
    calls: list[torch.Tensor] = []

    def downscale_bridge(data):
        calls.append(data)
        return data.repeat_interleave(2)

    transitions = LandUseTransitions(downscale_bridge=downscale_bridge)
    fractions = _state()["land_use_fraction"]

    refined = transitions.resolution_converter(
        fractions,
        2000.0,
        1000.0,
        method="fraction_weighted",
    )

    assert calls == [fractions]
    assert refined.shape == (2 * len(LandUseType),)
    assert refined.dim() == 1


def test_land_use_transitions_rejects_undeclared_resolution_method():
    transitions = LandUseTransitions(upscale_bridge=lambda data: data)

    with pytest.raises(ValueError, match="Unsupported resolution conversion method"):
        transitions.resolution_converter(
            torch.tensor([1.0]),
            1000.0,
            2000.0,
            method="invented",
        )


def test_land_use_transitions_phase8_certificate_is_valid():
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_land_use_transitions_land_use_transitions.yaml"
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
