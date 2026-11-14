"""Tests for IPHS equivalence certificate loading and validation."""

import json
import tempfile
from pathlib import Path

import pytest

from relic.computation.operators import iphs_certificate as certificate_module
from relic.computation.operators.iphs_certificate import (
    IPHSEquivalenceCertificate,
    load_iphs_equivalence_certificate,
)


def _certificate_payload():
    return {
        "process_id": "relic.scientific.pedosphere.transport.SoilWaterTransportFunctor",
        "module_name": "soil_water_transport",
        "canonical_role": "transport",
        "iphs_level": 3,
        "iphs_level_evidence": {
            "classification": "IPHS-structured transport certificate",
            "compatible_basis": "DEC cochain transport with conservative ports",
            "runtime_replacement": "forward path evaluates canonical IPHS transport operator",
            "canonical_operator": "theta_dot=-D_h q_h + G_h u_h",
            "structural_contribution": "Hodge, ports, and DEC flux divergence are declared",
            "assembly_participation": "certificate contributes to Phase 8 structural audit",
        },
        "theory_document": "docs\\source\\theory\\soil_hydraulics.md",
        "theory_document_status": "already_current",
        "legacy_formula": "q = -K grad(h)",
        "literature_formula": "Darcy flux q = -K grad(h + z)",
        "iphs_formula": "x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h",
        "legacy_expression": "q = -K grad(h)",
        "iphs_expression": "x_dot = J_h e_E + M_h e_S + G_h u_h",
        "input_variables": ["soil_water_storage", "hydraulic_head"],
        "output_variables": ["soil_water_flux"],
        "parameter_keys": ["pedosphere.hydraulic_conductivity"],
        "composition_ports": ["water_flux_boundary"],
        "references": ["van der Schaft and Maschke (1995)"],
        "validation_tests": [
            "test\\unit\\computation\\operators\\iphs_certificate.test.py"
        ],
        "runtime_replacement_map": {
            "runtime_uses_iphs": False,
            "legacy_removed": False,
        },
        "state_map": {"soil_water_storage": "x_h.storage"},
        "parameter_map": {
            "K": "pedosphere.hydraulic_conductivity",
        },
        "parameter_management_map": {
            "pedosphere.hydraulic_conductivity": {"source": "ParameterManager"}
        },
        "operator_map": {"darcy_gradient": "D_h", "conductivity": "M_h"},
        "state_advancement_map": {
            "potential_basis": "G_w(x)=0.5*x_h.T*H_w*x_h",
            "effort_variable": "mu_w=dG_w/dx_h=H_w*x_h",
            "state_advancer": "x_dot=M_h*mu_w+G_h*u_h",
            "runtime_usage": "forward evaluates the IPHS water flux operator directly",
        },
        "unit_map": {"soil_water_flux": "m s-1"},
        "input_output_map": {
            "soil_water_storage": "x_h.storage",
            "soil_water_flux": "port.water_flux",
        },
        "composition_management_map": {
            "functor_metadata": "existing FunctorMetadata fields",
            "port_spec": "existing PortSpec or PortSpecIPHS assembly view",
            "state_contract": "existing ContractRegistry and StateManager",
            "validation_rule": "existing ValidationRegistry rule",
        },
        "dependency_interconnection_map": {
            "dependency_type": "causal_state",
            "declared_dependency": "not_applicable",
        },
        "overflow_routing_map": {
            "residual_name": "water_balance_residual",
            "destination": "diagnostic_residual",
        },
        "conservation_map": {"water": "storage plus boundary ledger"},
        "entropy_audit": {"sigma_expression": "X_h.T L_h X_h >= 0"},
        "equivalence_type": "port_equivalent",
        "baseline_tests": ["soil_water_transport_iphs_equivalence.test.py"],
        "interface_map": {
            "forward_signature": "unchanged",
            "bridge_class": "unchanged",
            "dependency_semantics": "unchanged",
            "state_write_mode": "existing ContractRegistry and StateManager semantics",
            "legacy_inputs": ["soil_water_storage", "hydraulic_head"],
            "iphs_inputs": ["soil_water_storage", "hydraulic_head"],
            "legacy_outputs": ["soil_water_flux"],
            "iphs_outputs": ["soil_water_flux"],
        },
    }


def test_iphs_equivalence_certificate_validates_complete_payload():
    certificate = IPHSEquivalenceCertificate.from_mapping(_certificate_payload())

    audit = certificate.validate()

    assert audit["passed"] is True
    assert audit["parameter_count"] == 1
    assert audit["iphs_level"] == 3
    assert audit["iphs_level_name"] == "iphs_structured"
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_evidence_status"] == "descriptive_only"


def test_iphs_certificate_source_paths_are_cwd_independent(
    monkeypatch,
    tmp_path,
):
    """Repository-relative theory and test evidence must not depend on cwd."""
    certificate = IPHSEquivalenceCertificate.from_mapping(_certificate_payload())
    monkeypatch.chdir(tmp_path)

    audit = certificate.validate()

    assert audit["passed"] is True
    assert not any("current source tree" in issue for issue in audit["issues"])


@pytest.mark.parametrize("path_kind", ["absolute", "parent_escape"])
def test_iphs_certificate_rejects_evidence_paths_outside_source_tree(
    monkeypatch,
    tmp_path,
    path_kind,
):
    """Certificate evidence must remain inside the canonical source tree."""
    from relic.computation.operators import iphs_certificate as certificate_module

    source_root = tmp_path / "source_root"
    outside_root = tmp_path / "outside"
    source_root.mkdir()
    outside_root.mkdir()
    theory_path = outside_root / "theory.md"
    test_path = outside_root / "evidence.test.py"
    theory_path.write_text("theory", encoding="utf-8")
    test_path.write_text("test", encoding="utf-8")
    monkeypatch.setattr(certificate_module, "_SOURCE_TREE_ROOT", source_root)

    if path_kind == "absolute":
        theory_evidence = str(theory_path)
        test_evidence = str(test_path)
    else:
        theory_evidence = r"..\outside\theory.md"
        test_evidence = r"..\outside\evidence.test.py"

    payload = _certificate_payload()
    payload["theory_document"] = theory_evidence
    payload["validation_tests"] = [test_evidence]
    payload["baseline_tests"] = [test_evidence]

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert any("theory_document" in issue for issue in audit["issues"])
    assert any("validation_tests" in issue for issue in audit["issues"])


@pytest.mark.parametrize("field", ["theory_document", "validation_tests"])
def test_iphs_certificate_rejects_malformed_evidence_path_without_raising(field):
    """Malformed filesystem paths must become validation issues, not exceptions."""
    payload = _certificate_payload()
    malformed = "docs/source/\x00invalid-evidence"
    if field == "theory_document":
        payload[field] = malformed
    else:
        payload[field] = [malformed]
        payload["baseline_tests"] = [malformed]

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert any(field in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_accepts_explicit_parameter_free_morphism():
    """A pure interconnection morphism must not invent ParameterManager keys."""
    payload = _certificate_payload()
    payload["parameter_keys"] = []
    payload["parameter_free"] = True
    payload["parameter_map"] = {}
    payload["parameter_management_map"] = {}

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is True
    assert audit["parameter_count"] == 0
    assert audit["parameter_free"] is True


def test_iphs_equivalence_certificate_rejects_ambiguous_parameter_free_claim():
    """A certificate cannot be both parameter-free and parameter-bound."""
    payload = _certificate_payload()
    payload["parameter_free"] = True

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert any("cannot declare parameter_keys" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_non_strict_runtime_self_attestation():
    payload = _certificate_payload()
    payload["runtime_replacement_map"] = {
        "runtime_uses_iphs": True,
        "legacy_removed": True,
    }

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert any("cannot self-attest strict runtime replacement" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_reports_missing_runtime_replacement():
    payload = _certificate_payload()
    payload["runtime_replacement_map"] = {}

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("runtime_replacement_map" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_requires_declared_iphs_level():
    payload = _certificate_payload()
    payload.pop("iphs_level")

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("iphs_level is required" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_level_without_evidence():
    payload = _certificate_payload()
    payload["iphs_level"] = 3
    payload["iphs_level_evidence"] = {
        "classification": "IPHS-structured transport certificate",
    }

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("iphs_level_evidence" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_strict_iphs_overclaim():
    payload = _certificate_payload()
    payload["iphs_level"] = 4

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("storage_functional" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_prose_only_strict_iphs_evidence():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    payload["iphs_level_evidence"].update(
        {
            "storage_functional": "H_h(theta) is the water storage/free-energy functional",
            "effort_definition": "e_h=dH_h/dtheta_h",
            "dissipation_audit": "e_h.T M_h e_h >= 0",
            "port_power_audit": "dH/dt=-D_h+<y,u>",
        }
    )

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert audit["runtime_evidence_status"] == "invalid"
    assert any("structured runtime binding" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_generic_without_degeneracy():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    payload["iphs_level_evidence"].update(
        {
            "storage_functional": "H_h(theta)",
            "effort_definition": "e_h=dH_h/dtheta_h",
            "dissipation_audit": "e_h.T M_h e_h >= 0",
            "port_power_audit": "dH/dt=-D_h+<y,u>",
        }
    )

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("generic_degeneracy" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_partial_runtime_replacement():
    payload = _certificate_payload()
    payload["runtime_replacement_map"] = {"runtime_uses_iphs": False}

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("must define boolean runtime_uses_iphs and legacy_removed" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_reports_missing_state_advancement_map():
    payload = _certificate_payload()
    payload["state_advancement_map"] = {}

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("state_advancement_map" in issue for issue in audit["issues"])


def test_reaction_certificate_requires_explicit_nh_rh_runtime_operator():
    payload = _certificate_payload()
    payload["canonical_role"] = "reaction"
    payload["operator_map"] = {"legacy_decay": "k*x"}
    payload["state_advancement_map"]["state_advancer"] = "x_dot=-k*x"

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("explicit N_h" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_reports_missing_theory_status():
    payload = _certificate_payload()
    payload["theory_document_status"] = "missing"

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("theory_document_status" in issue for issue in audit["issues"])


def test_low_level_certificate_rejects_unbound_artifacts_and_parameter_source():
    payload = _certificate_payload()
    payload["iphs_level"] = 2
    payload["theory_document"] = "docs\\source\\theory\\missing.md"
    payload["validation_tests"] = ["test\\missing_certificate.test.py"]
    payload["baseline_tests"] = ["test\\missing_certificate.test.py"]
    payload["parameter_management_map"] = {
        "pedosphere.hydraulic_conductivity": {"source": ""}
    }

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert any("theory_document" in issue and "exist" in issue for issue in audit["issues"])
    assert any("validation_tests" in issue and "exist" in issue for issue in audit["issues"])
    assert any("ParameterManager" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_accepts_stochastic_sdae_maps():
    payload = _certificate_payload()
    payload["stochastic_sdae_map"] = {
        "runtime_owner": "relic.stochastic",
        "declares_stochastic_terms": True,
        "requires_noise_covariance": True,
        "requires_conservative_noise_flux": True,
    }
    payload["noise_covariance_map"] = {
        "relation": "fdt",
        "scale_parameter_key": "thermal.noise_scale",
    }
    payload["conservative_noise_flux_map"] = {
        "topology_operator": "D_flux",
        "route": "storage",
    }
    payload["stochastic_reproducibility_map"] = {
        "random_seed": 1,
        "noise_type": "white",
        "solver_method": "euler_maruyama",
    }

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is True


def test_iphs_equivalence_certificate_rejects_missing_required_stochastic_map():
    payload = _certificate_payload()
    payload["stochastic_sdae_map"] = {
        "runtime_owner": "relic.stochastic",
        "declares_stochastic_terms": True,
        "requires_noise_covariance": True,
    }

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    audit = certificate.validate()

    assert audit["passed"] is False
    assert any("noise_covariance_map" in issue for issue in audit["issues"])


def test_load_iphs_equivalence_certificate_reads_json():
    with tempfile.TemporaryDirectory(prefix="relic_iphs_certificate_") as tmp_dir:
        path = Path(tmp_dir) / "certificate.json"
        path.write_text(json.dumps(_certificate_payload()), encoding="utf-8")
        certificate = load_iphs_equivalence_certificate(path)

    assert certificate.module_name == "soil_water_transport"
    assert certificate.validate()["passed"] is True


def _complete_parameter_spec(key):
    return {
        "key": key,
        "category": "model_parameter",
        "units": "1",
        "dtype": "float32",
        "bounds": (0.0, 1.0),
        "references": ("unit-test",),
        "provenance": {"source": "unit_test"},
    }


def _strict_level4_bindings(payload):
    test_path = "test\\unit\\computation\\operators\\iphs_certificate.test.py"
    runtime_owner = (
        "relic.computation.operators.iphs.IPHSStructureSpec.canonical_drift"
    )
    validate_owner = "relic.computation.operators.iphs.IPHSStructureSpec.validate"
    payload["process_id"] = "relic.computation.operators.iphs.IPHSStructureSpec"
    payload["validation_tests"] = [test_path]
    payload["baseline_tests"] = [test_path]
    payload["formulation"] = "dissipative_ph"
    payload["iphs_expression"] = "x_dot=(J_h-R_h)e_H+G_h*u_h"
    payload["operator_map"] = {
        "reversible_operator": "J_h",
        "dissipative_operator": "R_h",
        "port_operator": "G_h",
    }
    payload["state_advancement_map"].update(
        {
            "effort_variable": "e_H",
            "state_advancer": "x_dot=(J_h-R_h)e_H+G_h*u_h",
        }
    )
    payload["runtime_replacement_map"] = {
        "runtime_uses_iphs": True,
        "legacy_removed": True,
        "runtime_owner": runtime_owner,
        "removed_legacy_symbols": ["legacy_flux_branch"],
        "validation_test": test_path,
    }
    payload["iphs_level_evidence"].update(
        {
            "storage_functional": {
                "runtime_owner": runtime_owner,
                "value_key": "energy",
                "state_keys": ["soil_water_storage"],
                "units": "J m-2",
                "theory_anchor": "soil_hydraulics:storage",
                "validation_test": test_path,
            },
            "effort_definition": {
                "runtime_owner": runtime_owner,
                "value_key": "energy_effort",
                "derivative_of": "storage_functional",
                "method": "analytic",
                "validation_test": test_path,
            },
            "dissipation_audit": {
                "runtime_owner": validate_owner,
                "operator_owner": validate_owner,
                "operator_kind": "resistance_R",
                "value_key": "energy_dissipation",
                "balance_key": "closed_port_energy_rate",
                "inequality": "dH_dt<=port_power",
                "tolerance_parameter_key": "test.iphs.tolerance",
                "validation_test": test_path,
            },
            "port_power_audit": {
                "runtime_owner": runtime_owner,
                "balance_key": "port_power_residual",
                "sign_convention": "into_system",
                "tolerance_parameter_key": "test.iphs.tolerance",
                "validation_test": test_path,
            },
        }
    )
    return payload


def _strict_level5_bindings(
    payload,
    *,
    state_dependence="constant",
    jacobi_basis="constant_operator",
):
    _strict_level4_bindings(payload)
    payload["formulation"] = "generic"
    payload["iphs_expression"] = "x_dot=L_h*e_E+M_h*e_S+G_h*u_h"
    payload["operator_map"] = {
        "reversible_operator": "L_h",
        "dissipative_operator": "M_h",
        "port_operator": "G_h",
    }
    payload["state_advancement_map"].update(
        {
            "effort_variable": "e_E,e_S",
            "state_advancer": "x_dot=L_h*e_E+M_h*e_S+G_h*u_h",
        }
    )
    test_path = payload["validation_tests"][0]
    validate_owner = "relic.computation.operators.iphs.IPHSStructureSpec.validate"
    payload["iphs_level_evidence"]["dissipation_audit"].update(
        {
            "operator_kind": "mobility_M",
            "value_key": "entropy_production",
            "balance_key": "entropy_production",
            "inequality": "dS_dt>=port_entropy_flow",
        }
    )
    payload["iphs_level_evidence"].update(
        {
            "energy_entropy_pair": {
                "energy_owner": validate_owner,
                "entropy_owner": validate_owner,
                "energy_key": "energy",
                "entropy_key": "entropy",
                "validation_test": test_path,
            },
            "generic_degeneracy": {
                "structure_owner": validate_owner,
                "j_entropy_residual_key": "J_entropy_gradient_residual",
                "m_energy_residual_key": "M_energy_gradient_residual",
                "reversible_operator_state_dependence": state_dependence,
                "jacobi_identity_basis": jacobi_basis,
                "tolerance_parameter_key": "test.iphs.tolerance",
                "validation_test": test_path,
            },
            "entropy_production": {
                "runtime_owner": validate_owner,
                "value_key": "entropy_production",
                "lower_bound": 0.0,
                "tolerance_parameter_key": "test.iphs.tolerance",
                "validation_test": test_path,
            },
        }
    )
    return payload


def test_iphs_equivalence_certificate_rejects_mapping_only_level4_bindings():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    certificate = IPHSEquivalenceCertificate.from_mapping(payload)

    audit = certificate.validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_evidence_status"] == "invalid"
    assert any(
        "process_bound_runtime_unavailable" in issue and "[N]" in issue
        for issue in audit["issues"]
    )
    assert audit["formulation"] == "dissipative_ph"
    assert certificate.strict_tolerance_parameter_keys() == (
        "test.iphs.tolerance",
    )


def test_level4_rejects_real_functor_forward_without_execution_witness():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    process_id = (
        "relic.scientific.pedosphere.transport.SoilWaterTransportFunctor."
        "SoilWaterTransportFunctor"
    )
    payload["process_id"] = process_id
    payload["runtime_replacement_map"]["runtime_owner"] = process_id + ".forward"

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert any("process_bound_runtime_unavailable" in issue for issue in audit["issues"])


def test_level4_rejects_explicit_generic_mobility_structure():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload["iphs_expression"] = "x_dot=J_h*e_E+M_h*e_S+G_h*u_h"
    payload["operator_map"] = {
        "reversible_operator": "J_h",
        "dissipative_operator": "M_h",
    }
    payload["state_advancement_map"].update(
        {
            "effort_variable": "e_S",
            "state_advancer": "x_dot=J_h*e_E+M_h*e_S+G_h*u_h",
        }
    )

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any("Level 4 formulation structure" in issue for issue in audit["issues"])


def test_level5_rejects_explicit_dissipative_ph_structure():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(payload)
    payload["iphs_expression"] = "x_dot=(J_h-R_h)e_H+G_h*u_h"
    payload["operator_map"] = {
        "reversible_operator": "J_h",
        "dissipative_operator": "R_h",
    }
    payload["state_advancement_map"].update(
        {
            "effort_variable": "e_H",
            "state_advancer": "x_dot=(J_h-R_h)e_H+G_h*u_h",
        }
    )

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any("Level 5 formulation structure" in issue for issue in audit["issues"])


def test_level3_rejects_truthy_but_materially_empty_required_map():
    payload = _certificate_payload()
    payload["state_map"] = {"soil_water_storage": ""}

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert any("state_map contains empty evidence" in issue for issue in audit["issues"])


def test_level3_rejects_empty_composition_payload():
    payload = _certificate_payload()
    payload["composition_ports"] = []
    payload["composition_management_map"] = {"ports": []}

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert any("material composition" in issue for issue in audit["issues"])


def test_level3_rejects_nested_empty_level_evidence():
    payload = _certificate_payload()
    payload["iphs_level_evidence"]["structural_contribution"] = {"owner": ""}

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

    assert audit["passed"] is False
    assert any("iphs_level_evidence" in issue for issue in audit["issues"])


@pytest.mark.parametrize(
    "removed_symbols",
    [
        [""],
        ["legacy_branch", "legacy_branch"],
    ],
)
def test_level4_rejects_invalid_removed_legacy_symbol_identity(removed_symbols):
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload["runtime_replacement_map"]["removed_legacy_symbols"] = removed_symbols

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any("removed_legacy_symbols" in issue for issue in audit["issues"])


def test_strict_level_evidence_requires_one_shared_tolerance_parameter_key():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload["iphs_level_evidence"]["port_power_audit"][
        "tolerance_parameter_key"
    ] = "test.iphs.port_power_tolerance"

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any(
        "single tolerance_parameter_key" in issue for issue in audit["issues"]
    )


def test_iphs_equivalence_certificate_rejects_empty_parameter_metadata_fields():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)

    def incomplete_resolver(key):
        spec = _complete_parameter_spec(key)
        spec["references"] = []
        return spec

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=incomplete_resolver
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert any(
        "complete ParameterManager metadata" in issue
        for issue in audit["issues"]
    )


def test_parameter_resolver_failure_propagates_instead_of_an_incomplete_metadata_issue():
    """A missing key resolves to None (an issue); a failing resolver is an error."""
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)

    def failing_resolver(key):
        raise RuntimeError(f"parameter backend unavailable for {key}")

    with pytest.raises(RuntimeError, match="parameter backend unavailable"):
        IPHSEquivalenceCertificate.from_mapping(payload).validate(
            parameter_resolver=failing_resolver
        )


def test_unresolved_parameter_key_is_reported_as_an_issue():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: None
    )

    assert audit["passed"] is False
    assert any(
        "complete ParameterManager metadata" in issue for issue in audit["issues"]
    )


def test_strict_certificate_validates_every_declared_parameter_key():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload["parameter_keys"] = [
        "test.iphs.incomplete_parameter",
        "test.iphs.complete_parameter",
    ]

    def first_parameter_incomplete(key):
        spec = _complete_parameter_spec(key)
        if key == "test.iphs.incomplete_parameter":
            spec["references"] = ()
        return spec

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=first_parameter_incomplete
    )

    assert audit["passed"] is False
    assert any(
        "test.iphs.incomplete_parameter" in issue for issue in audit["issues"]
    )


def test_strict_level_claim_cannot_disable_runtime_replacement_obligations():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload["runtime_replacement_map"]["runtime_uses_iphs"] = False
    payload["runtime_replacement_map"]["legacy_removed"] = False

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        strict=False,
        parameter_resolver=lambda key: _complete_parameter_spec(key),
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_evidence_status"] == "invalid"
    assert any(
        "runtime_uses_iphs and legacy_removed" in issue
        for issue in audit["issues"]
    )


def test_iphs_equivalence_certificate_rejects_non_integral_level_values():
    for invalid_level in (True, 4.5):
        payload = _certificate_payload()
        payload["iphs_level"] = invalid_level

        audit = IPHSEquivalenceCertificate.from_mapping(payload).validate()

        assert audit["passed"] is False
        assert any("iphs_level" in issue for issue in audit["issues"])


def test_low_level_claim_cannot_self_attest_during_compatibility_validation():
    payload = _certificate_payload()
    payload["runtime_replacement_map"] = {
        "runtime_uses_iphs": True,
        "legacy_removed": True,
    }

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(strict=False)

    assert audit["passed"] is False
    assert any(
        "cannot self-attest strict runtime replacement" in issue
        for issue in audit["issues"]
    )


def test_required_runtime_level_rejects_descriptive_certificate():
    audit = IPHSEquivalenceCertificate.from_mapping(_certificate_payload()).validate(
        strict=False,
        required_level=4,
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert any("required runtime-verified IPHS level 4" in issue for issue in audit["issues"])


@pytest.mark.parametrize("required_level", [1, 2, 3])
def test_strict_content_validation_preserves_requested_runtime_floor(required_level):
    audit = IPHSEquivalenceCertificate.from_mapping(_certificate_payload()).validate(
        strict=True,
        required_level=required_level,
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert audit["issues"] == [
        f"required runtime-verified IPHS level {required_level} was not met; verified level is 0"
    ]


@pytest.mark.parametrize("missing_map", [None, "state_map", "parameter_management_map", "interface_map"])
def test_zero_runtime_floor_does_not_disable_strict_certificate_content(missing_map):
    payload = _certificate_payload()
    if missing_map is not None:
        payload[missing_map] = {}

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        strict=True, required_level=0,
    )

    assert audit["runtime_verified_level"] == 0
    assert audit["passed"] is (missing_map is None)
    assert not any("required_level >= 4" in issue for issue in audit["issues"])
    if missing_map is not None:
        assert any(missing_map in issue for issue in audit["issues"])


@pytest.mark.parametrize("required_level", [0, 1, 3])
def test_low_requested_floor_does_not_weaken_declared_level_four(required_level):
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        strict=True, required_level=required_level,
        parameter_resolver=_complete_parameter_spec,
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert any("process_bound_runtime_unavailable" in issue for issue in audit["issues"])
    assert not any("strict validation requires required_level >= 4" in issue for issue in audit["issues"])


@pytest.mark.parametrize(
    ("declarations", "strict_iphs", "role", "expected"),
    [
        ({"context": {"required_level": 0}}, False, "", None),
        ({"context": {"iphs_level": 0}}, False, "", None),
        ({"context": {"iphs_level": 1}}, False, "", None),
        ({"context": {"iphs_level": 2}}, False, "", None),
        ({"context": {"iphs_level": 3}}, False, "", None),
        ({"context": {"required_level": 1}}, False, "", 1),
        ({"context": {"required_level": 3}}, False, "", 3),
        ({"context": {"required_level": 0}}, True, "", 4),
        ({"context": {"required_level": 4, "iphs_level": 5}}, False, "", 5),
        ({"outer": {"required_level": 5}, "inner": {"required_level": 0}}, True, "", 5),
        ({"context": {}}, True, "observation", None),
        ({"context": {"required_level": 5}}, True, "observation", 5),
    ],
)
def test_certificate_owner_resolves_one_runtime_requirement(declarations, strict_iphs, role, expected):
    resolver = getattr(certificate_module, "resolve_iphs_level_requirement", None)
    assert callable(resolver)

    level, invalid = resolver(declarations, strict_iphs=strict_iphs, canonical_role=role)

    assert level == expected
    assert invalid == []


@pytest.mark.parametrize("field", ["required_level", "iphs_level"])
@pytest.mark.parametrize("value", [None, True, "3", 3.0, -1, 6])
def test_runtime_requirement_parser_does_not_coerce_metadata_levels(field, value):
    resolver = getattr(certificate_module, "resolve_iphs_level_requirement", None)
    assert callable(resolver)

    level, invalid = resolver({"context": {field: value}})

    assert level is None
    assert invalid == [(f"context.{field}", value)]


def test_iphs_equivalence_certificate_rejects_unrelated_callable_owners():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload["runtime_replacement_map"]["runtime_owner"] = "math.sqrt"
    for evidence in payload["iphs_level_evidence"].values():
        if not isinstance(evidence, dict):
            continue
        for owner_key in (
            "runtime_owner",
            "operator_owner",
            "energy_owner",
            "entropy_owner",
            "structure_owner",
        ):
            if owner_key in evidence:
                evidence[owner_key] = "math.sqrt"

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert any("bound" in issue and "process" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_level4_without_formulation():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload.pop("formulation")

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any("formulation='dissipative_ph'" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_generic_level4_claim():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    payload["formulation"] = "generic"

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any("Level 4" in issue and "dissipative_ph" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_assigned_zero_level5_evidence():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(payload)
    payload["iphs_level_evidence"]["generic_degeneracy"].update(
        {
            "j_entropy_residual_key": "assigned_zero",
            "m_energy_residual_key": "assigned_zero",
        }
    )

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any("assigned_zero" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_level5_without_jacobi_evidence():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(
        payload,
        state_dependence="state_dependent",
        jacobi_basis="",
    )

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert any("jacobi_identity_basis" in issue for issue in audit["issues"])


def test_iphs_equivalence_certificate_rejects_text_only_state_dependent_jacobi_claim():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(
        payload,
        state_dependence="state_dependent",
        jacobi_basis="trust me",
    )

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    # A text-only basis is refused; only the executable sampled-domain residual basis
    # produced by IPHSStructureSpec.validate is accepted for a state-dependent operator.
    assert any(
        "sampled_domain_jacobi_residual" in issue and "not Level 5 evidence" in issue
        for issue in audit["issues"]
    )


def test_iphs_equivalence_certificate_rejects_text_only_constant_operator_level5():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(payload)

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] < 5
    assert audit["formulation"] == "generic"
    # The text label constant_operator is not evidence; constant_field_exact from the
    # executable constant_field policy is required.
    assert any(
        "constant_field_exact" in issue and "not Level 5 evidence" in issue
        for issue in audit["issues"]
    )


def _fail_closed_nonimplementation_payload():
    payload = _certificate_payload()
    payload.update(
        {
            "iphs_level": 0,
            "iphs_expression": "[N] runtime_unbound",
            "input_variables": [],
            "output_variables": [],
            "composition_ports": [],
            "interface_map": {"forward_signature": "fail_closed"},
            "runtime_replacement_map": {
                "operational_claim": False,
                "runtime_uses_iphs": False,
                "legacy_removed": False,
                "replacement_entrypoint": "none_runtime_unbound",
            },
        }
    )
    return payload


def test_certificate_owns_explicit_fail_closed_nonimplementation_predicate():
    certificate = IPHSEquivalenceCertificate.from_mapping(
        _fail_closed_nonimplementation_payload()
    )

    assert certificate.declares_fail_closed_nonimplementation() is True
    assert certificate.validate(strict=True)["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "level",
        "operational_claim",
        "runtime_uses_iphs",
        "legacy_removed",
        "replacement_entrypoint",
        "inputs",
        "outputs",
        "ports",
        "expression",
    ],
)
def test_fail_closed_nonimplementation_predicate_requires_complete_exact_claim(
    mutation: str,
):
    payload = _fail_closed_nonimplementation_payload()
    if mutation == "level":
        payload["iphs_level"] = 1
    elif mutation == "operational_claim":
        payload["runtime_replacement_map"]["operational_claim"] = True
    elif mutation == "runtime_uses_iphs":
        payload["runtime_replacement_map"]["runtime_uses_iphs"] = True
    elif mutation == "legacy_removed":
        payload["runtime_replacement_map"]["legacy_removed"] = True
    elif mutation == "replacement_entrypoint":
        payload["runtime_replacement_map"]["replacement_entrypoint"] = "forward"
    elif mutation == "inputs":
        payload["input_variables"] = ["state_input"]
    elif mutation == "outputs":
        payload["output_variables"] = ["state_output"]
    elif mutation == "ports":
        payload["composition_ports"] = ["runtime_port"]
    elif mutation == "expression":
        payload["iphs_expression"] = "x_dot = N_h r_h"

    certificate = IPHSEquivalenceCertificate.from_mapping(payload)

    assert certificate.declares_fail_closed_nonimplementation() is False


def _process_bound_runtime_point_audit(payload, **overrides):
    """A Level 5 runtime-point audit as returned by IPHSStructureSpec.validate."""
    committed = [
        name for name in payload["output_variables"] if not str(name).startswith("diagnostics.")
    ]
    audit = {
        "name": "unit_runtime_point",
        "formulation": "generic",
        "poisson_operator": "zero_field",
        "reaction_representation": "absorbed",
        "required_level": 5,
        "runtime_verified_level": 5,
        "process_level_status": "runtime_point_verified",
        "evidence_scope": "runtime_point",
        "jacobi_identity_verified": True,
        "jacobi_identity_basis": "canonical_zero_field",
        "J_entropy_gradient_residual": 0.0,
        "M_energy_gradient_residual": 0.0,
        "entropy_production": 0.6,
        "reaction_energy_rate": 0.0,
        "reaction_entropy_rate": 0.6,
        "governed_execution": {
            "entry": "FunctorBase.__call__",
            "functor": payload["module_name"],
            "committed_variables": committed,
            "step": 1,
        },
    }
    audit.update(overrides)
    return audit


def _level5_zero_field_payload():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(
        payload,
        state_dependence="zero_field",
        jacobi_basis="canonical_zero_field",
    )
    return payload


def test_level5_certificate_verifies_with_process_bound_runtime_point_audit():
    payload = _level5_zero_field_payload()
    certificate = IPHSEquivalenceCertificate.from_mapping(payload)

    audit = certificate.validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )

    assert audit["issues"] == []
    assert audit["passed"] is True
    assert audit["runtime_verified_level"] == 5
    assert audit["runtime_evidence_status"] == "verified"
    assert audit["runtime_binding"] == "process_bound"
    assert audit["formulation"] == "generic"


def test_level5_certificate_without_runtime_point_audit_stays_process_unbound():
    payload = _level5_zero_field_payload()

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_binding"] == "descriptive"
    assert any("process_bound_runtime_unavailable" in issue for issue in audit["issues"])
    assert not any("constant reversible operator" in issue for issue in audit["issues"])


@pytest.mark.parametrize("overrides,fragment", [
    ({"runtime_verified_level": 3}, "runtime_verified_level"),
    ({"process_level_status": "[N] runtime_unbound"}, "process_level_status"),
    ({"formulation": "dissipative_ph"}, "formulation"),
    ({"jacobi_identity_basis": "unimplemented"}, "jacobi_identity_basis"),
    ({"entropy_production": -1.0e-3}, "entropy_production"),
    ({"reaction_entropy_rate": -1.0e-3}, "reaction_entropy_rate"),
    ({"governed_execution": None}, "governed_execution"),
    ({"governed_execution": {"entry": "forward", "functor": "x", "committed_variables": ["soil_water_flux"], "step": 1}}, "governed_execution.entry"),
    ({"governed_execution": {"entry": "FunctorBase.__call__", "functor": "x", "committed_variables": [], "step": 1}}, "committed_variables"),
    ({"governed_execution": {"entry": "FunctorBase.__call__", "functor": "x", "committed_variables": ["soil_water_flux"], "step": 0}}, "step"),
])
def test_level5_certificate_rejects_inconsistent_runtime_point_audit(overrides, fragment):
    payload = _level5_zero_field_payload()

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload, **overrides),
    )

    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert any("runtime_point_audit" in issue and fragment in issue for issue in audit["issues"]), audit["issues"]


def test_level5_zero_field_degeneracy_requires_the_canonical_basis():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(payload, state_dependence="zero_field", jacobi_basis="trust me")

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )

    assert audit["passed"] is False
    assert any(
        "zero_field" in issue and "jacobi_identity_basis" in issue
        for issue in audit["issues"]
    )


def test_strict_runtime_replacement_accepts_a_process_that_never_had_legacy_code():
    payload = _level5_zero_field_payload()
    payload["runtime_replacement_map"]["removed_legacy_symbols"] = []
    without_flag = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )
    assert without_flag["passed"] is False
    assert any("removed_legacy_symbols" in issue for issue in without_flag["issues"])

    payload["runtime_replacement_map"]["legacy_never_existed"] = True
    with_flag = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )
    assert with_flag["issues"] == []
    assert with_flag["passed"] is True

    payload["runtime_replacement_map"]["legacy_never_existed"] = "yes"
    bad_flag = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )
    assert bad_flag["passed"] is False
    assert any("legacy_never_existed" in issue for issue in bad_flag["issues"])


def test_runtime_point_audit_is_only_accepted_for_strict_levels():
    payload = _certificate_payload()
    certificate = IPHSEquivalenceCertificate.from_mapping(payload)

    audit = certificate.validate(
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )

    assert audit["passed"] is False
    assert any(
        "runtime_point_audit" in issue and "Level 4 or 5" in issue
        for issue in audit["issues"]
    )


def test_level4_certificate_verifies_with_process_bound_runtime_point_audit():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    audit = _process_bound_runtime_point_audit(
        payload,
        formulation="dissipative_ph",
        required_level=4,
        runtime_verified_level=4,
        closed_port=True,
        dissipation_power=50.0,
        closed_port_energy_rate=-50.0,
        power_balance_residual=0.0,
    )
    for key in ("jacobi_identity_verified", "jacobi_identity_basis", "entropy_production",
                "reaction_energy_rate", "reaction_entropy_rate",
                "J_entropy_gradient_residual", "M_energy_gradient_residual"):
        audit.pop(key)

    result = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=audit,
    )

    assert result["issues"] == []
    assert result["passed"] is True
    assert result["runtime_verified_level"] == 4
    assert result["runtime_binding"] == "process_bound"
    assert result["formulation"] == "dissipative_ph"


def test_level4_certificate_rejects_generic_runtime_point_audit():
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    audit = _process_bound_runtime_point_audit(payload, required_level=4, runtime_verified_level=4)

    result = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=audit,
    )

    assert result["passed"] is False
    assert any("runtime_point_audit.formulation" in issue for issue in result["issues"])


@pytest.mark.parametrize("overrides,fragment,accepted", [
    ({"storage_kind": "free_energy", "reaction_energy_rate": -0.8}, "", True),
    ({"storage_kind": "total_energy", "reaction_energy_rate": 0.0}, "", True),
    ({"storage_kind": "total_energy", "reaction_energy_rate": 0.1}, "must vanish", False),
    ({"storage_kind": "free_energy", "reaction_energy_rate": 0.1}, "non-positive", False),
    ({"reaction_energy_rate": -0.8}, "storage_kind is required", False),
    ({"storage_kind": "enthalpy"}, "storage_kind must be", False),
])
def test_level4_certificate_checks_reaction_pairing_keys_of_the_runtime_audit(overrides, fragment, accepted):
    payload = _certificate_payload()
    payload["iphs_level"] = 4
    _strict_level4_bindings(payload)
    audit = _process_bound_runtime_point_audit(
        payload, formulation="dissipative_ph", required_level=4, runtime_verified_level=4,
        closed_port=True, dissipation_power=50.0, closed_port_energy_rate=-50.0,
        power_balance_residual=0.0,
    )
    for key in ("jacobi_identity_verified", "jacobi_identity_basis", "entropy_production",
                "reaction_energy_rate", "reaction_entropy_rate",
                "J_entropy_gradient_residual", "M_energy_gradient_residual"):
        audit.pop(key)
    audit.update(overrides)

    result = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=audit,
    )

    assert result["passed"] is accepted, result["issues"]
    if fragment:
        assert any(fragment in issue for issue in result["issues"]), result["issues"]


def test_iphs_equivalence_certificate_accepts_constant_field_exact_jacobi_basis():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(payload, state_dependence="constant", jacobi_basis="constant_field_exact")

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )

    assert not any("jacobi" in issue.lower() for issue in audit["issues"])


def test_iphs_equivalence_certificate_requires_numeric_sample_evidence_for_state_field():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(
        payload, state_dependence="state_dependent", jacobi_basis="sampled_domain_jacobi_residual"
    )

    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )
    assert any("jacobi_sample_count" in issue for issue in audit["issues"])
    assert any("jacobi_tolerance" in issue for issue in audit["issues"])

    payload["iphs_level_evidence"]["generic_degeneracy"].update(
        {"jacobi_sample_count": 8, "jacobi_tolerance": 1.0e-8}
    )
    audit = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key)
    )
    assert not any("jacobi" in issue.lower() for issue in audit["issues"])


# MA-21: a state_field runtime point is accepted only with the Jacobi residual
# evaluated at the executed runtime state (finite samples plus that point).
def _level5_state_field_payload():
    payload = _certificate_payload()
    payload["iphs_level"] = 5
    _strict_level5_bindings(
        payload, state_dependence="state_dependent", jacobi_basis="sampled_domain_jacobi_residual"
    )
    payload["iphs_level_evidence"]["generic_degeneracy"].update(
        {"jacobi_sample_count": 3, "jacobi_tolerance": 1.0e-8}
    )
    return payload


def _state_field_runtime_point_audit(payload, **overrides):
    audit = _process_bound_runtime_point_audit(
        payload,
        poisson_operator="state_field",
        jacobi_identity_basis="sampled_domain_jacobi_residual",
        jacobi_sample_count=3,
        jacobi_max_residual=0.0,
        jacobi_tolerance=1.0e-8,
        runtime_point_skew_residual=0.0,
        runtime_point_jacobi_residual=2.0e-12,
        runtime_point_jacobi_point="bound_state",
    )
    audit.update(overrides)
    return audit


@pytest.mark.parametrize("overrides,fragment", [
    ({}, ""),
    ({"runtime_point_jacobi_residual": None}, "runtime_point_jacobi_residual"),
    ({"runtime_point_jacobi_residual": 1.0e-6}, "runtime_point_jacobi_residual"),
    ({"runtime_point_jacobi_residual": float("nan")}, "runtime_point_jacobi_residual"),
    ({"runtime_point_skew_residual": None}, "runtime_point_skew_residual"),
])
def test_level5_state_field_runtime_audit_requires_the_runtime_point_jacobi_residual(overrides, fragment):
    payload = _level5_state_field_payload()
    audit = _state_field_runtime_point_audit(payload, **overrides)
    for key, value in overrides.items():
        if value is None:
            audit.pop(key)
    result = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=audit,
    )
    if not fragment:
        assert result["issues"] == []
        assert result["runtime_verified_level"] == 5
        return
    assert result["passed"] is False
    assert any("runtime_point_audit" in issue and fragment in issue for issue in result["issues"]), result["issues"]


# MA-21: constructor_selection is a certificate field that marks a constructor-scoped
# runtime-point certificate (constructor keyword -> JSON scalar).
def test_constructor_selection_is_a_validated_certificate_field():
    payload = _level5_zero_field_payload()
    assert IPHSEquivalenceCertificate.from_mapping(payload).constructor_selection == {}
    payload["constructor_selection"] = {"formulation": "generic", "poisson_operator": "zero_field"}
    certificate = IPHSEquivalenceCertificate.from_mapping(payload)
    assert certificate.constructor_selection == {"formulation": "generic", "poisson_operator": "zero_field"}
    result = certificate.validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )
    assert result["issues"] == []
    assert result["passed"] is True
    assert result["constructor_selection"] == {"formulation": "generic", "poisson_operator": "zero_field"}


@pytest.mark.parametrize("selection", ["physical_boundary=closed_constant_volume", ["formulation"], 3])
def test_constructor_selection_must_load_as_a_mapping(selection):
    payload = _level5_zero_field_payload()
    payload["constructor_selection"] = selection
    with pytest.raises(ValueError, match="constructor_selection must be a mapping"):
        IPHSEquivalenceCertificate.from_mapping(payload)


@pytest.mark.parametrize("selection,level,fragment", [
    ({"formulation": ["generic"]}, 5, "JSON scalar"),
    ({"formulation": {"kind": "generic"}}, 5, "JSON scalar"),
    ({"skew_tol": float("inf")}, 5, "JSON scalar"),
    ({"not an identifier": True}, 5, "keyword"),
    ({"no_such_constructor_keyword": True}, 5, "not a constructor keyword"),
    ({}, 5, ""),
    ({"formulation": "generic"}, 3, "iphs_level 4 or 5"),
])
def test_constructor_selection_contents_are_checked_against_the_process_constructor(selection, level, fragment):
    payload = _level5_zero_field_payload() if level == 5 else _certificate_payload()
    if level == 5:
        runtime_audit = _process_bound_runtime_point_audit(payload)
    else:
        payload["process_id"] = "relic.computation.operators.iphs.IPHSStructureSpec"
        runtime_audit = None
    payload["constructor_selection"] = selection
    result = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=runtime_audit,
    )
    if not fragment:
        assert result["issues"] == []
        return
    assert result["passed"] is False
    assert any("constructor_selection" in issue and fragment in issue for issue in result["issues"]), result["issues"]


def test_constructor_selection_requires_a_resolvable_process_class():
    payload = _level5_zero_field_payload()
    payload["constructor_selection"] = {"formulation": "generic"}
    payload["process_id"] = "relic.scientific.no_such_module.NoSuchFunctor"
    result = IPHSEquivalenceCertificate.from_mapping(payload).validate(
        parameter_resolver=lambda key: _complete_parameter_spec(key),
        runtime_point_audit=_process_bound_runtime_point_audit(payload),
    )
    assert any(
        "constructor_selection" in issue and "process class" in issue for issue in result["issues"]
    ), result["issues"]


@pytest.mark.parametrize("certificate_file,selection", [
    ("atmosphere_chemistry_chemistry_functor_closed_constant_volume.yaml",
     {"physical_boundary": "closed_constant_volume"}),
    ("transport_diffusion_fickian_diffusion_strict_runtime_point.yaml",
     {"strict_runtime_point": True, "spatial_dims": 1}),
])
def test_repository_runtime_point_certificates_declare_their_constructor_selection(certificate_file, selection):
    root = Path(__file__).resolve().parents[4] / "docs" / "source" / "theory" / "iphs_certificates"
    certificate = load_iphs_equivalence_certificate(root / certificate_file)
    assert certificate.constructor_selection == selection
    main_file = certificate_file.replace("_closed_constant_volume", "").replace("_strict_runtime_point", "")
    assert load_iphs_equivalence_certificate(root / main_file).constructor_selection == {}
