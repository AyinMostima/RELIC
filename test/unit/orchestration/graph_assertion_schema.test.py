"""Unit tests for Graph IR assertion schema and lightweight assertion validation."""

import pytest
from jsonschema import ValidationError

from relic.orchestration.graph_io import validate_graph_schema
from relic.orchestration.graph_ir import GraphIR
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.orchestration.validator import build_audit_summary, validate_graph


def _minimal_graph_payload():
    return {
        "schema_version": "1.0.0",
        "nodes": {
            "n0": {
                "class_type": "State:ExternalForcing",
                "outputs": ["precipitation_flux"],
            }
        },
        "links": [],
        "metadata": {},
    }


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("nodes", "n0"),
        ("links", 0),
        ("links", 0, "from"),
        ("links", 0, "to"),
    ],
)
def test_graph_schema_rejects_unknown_structural_fields(path):
    payload = _minimal_graph_payload()
    payload["nodes"]["n1"] = {
        "class_type": "State:ExternalForcing",
        "inputs": {"precipitation_flux": {"link": "n0.precipitation_flux"}},
    }
    payload["links"] = [
        {
            "from": {"node": "n0", "port": "precipitation_flux"},
            "to": {"node": "n1", "port": "precipitation_flux"},
        }
    ]
    target = payload
    for key in path:
        target = target[key]
    target["unexpected_field"] = True

    with pytest.raises(ValidationError):
        validate_graph_schema(payload)


def test_graph_schema_accepts_optional_assertion():
    payload = _minimal_graph_payload()
    payload["assertion"] = {
        "phi": "soil_moisture >= 0",
        "op": "n0",
        "scope": {"scale": "patch", "context": "unit_test"},
        "context": {"dataset": "synthetic"},
    }
    validate_graph_schema(payload)


def test_validate_graph_reports_assertion_error_when_op_missing():
    payload = _minimal_graph_payload()
    payload["assertion"] = {
        "phi": "soil_moisture >= 0",
        "scope": "patch",
    }
    graph = GraphIR.from_dict(payload)
    registry = NodeRegistry(SSOTCatalog())
    report = validate_graph(graph, registry, strict=True, ssot_catalog=SSOTCatalog())
    assert any(issue.code == "assertion_missing_op" for issue in report.errors)
    assert report.assertion_audit["present"] is True
    assert report.assertion_audit["valid"] is False


def test_validate_graph_resolves_assertion_op_with_node_id():
    payload = _minimal_graph_payload()
    payload["assertion"] = {
        "phi": "precipitation_flux >= 0",
        "op": "n0",
        "scope": {"scale": "grid_cell"},
    }
    graph = GraphIR.from_dict(payload)
    registry = NodeRegistry(SSOTCatalog())
    report = validate_graph(graph, registry, strict=True, ssot_catalog=SSOTCatalog())
    assert not any(issue.code.startswith("assertion_") for issue in report.errors)
    assert report.assertion_audit["valid"] is True
    assert report.assertion_audit["op_resolved"] is True


def test_build_audit_summary_contains_conservation_section():
    payload = _minimal_graph_payload()
    graph = GraphIR.from_dict(payload)
    registry = NodeRegistry(SSOTCatalog())
    report = validate_graph(graph, registry, strict=True, ssot_catalog=SSOTCatalog())
    audit = build_audit_summary(
        report,
        metadata={
            "topology_signature": {
                "rank": 0,
                "nullity": 1,
                "betti_number": 0,
                "conservation_vectors": [[1.0]],
            }
        },
    )
    assert audit["conservation_audit"]["enabled"] is True
    assert audit["conservation_audit"]["conservation_vector_count"] == 1


def test_graph_schema_accepts_assertion_batch():
    payload = _minimal_graph_payload()
    payload["assertion"] = {
        "assertions": [
            {
                "phi": "topology",
                "op": "cochain_complex_check",
                "scope": "grid",
                "context": {},
                "audit_key": "dec_topology_audit",
            },
            {
                "phi": "state_domain",
                "op": "state_domain_check",
                "scope": "state",
                "context": {},
                "audit_key": "state_domain_audit",
            },
        ]
    }

    validate_graph_schema(payload)


def test_graph_schema_rejects_non_mapping_assertion_batch_item():
    import pytest
    from jsonschema import ValidationError

    payload = _minimal_graph_payload()
    payload["assertion"] = {"assertions": ["not-a-mapping"]}

    with pytest.raises(ValidationError):
        validate_graph_schema(payload)


def test_validate_graph_rejects_structural_assertion_audit_key_mismatch():
    payload = _minimal_graph_payload()
    payload["assertion"] = {
        "phi": "topology",
        "op": "cochain_complex_check",
        "scope": "grid",
        "context": {},
        "audit_key": "hodge_audit",
    }
    graph = GraphIR.from_dict(payload)

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        strict=True,
        ssot_catalog=SSOTCatalog(),
    )

    assert any(issue.code == "assertion_audit_key_mismatch" for issue in report.errors)


def test_validate_graph_rejects_duplicate_structural_assertion_ops():
    payload = _minimal_graph_payload()
    assertion = {
        "phi": "state_domain",
        "op": "state_domain_check",
        "scope": "state",
        "context": {"state": {}, "state_domain_constraints": {}},
        "audit_key": "state_domain_audit",
    }
    payload["assertion"] = {"assertions": [assertion, dict(assertion)]}
    graph = GraphIR.from_dict(payload)

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        strict=True,
        ssot_catalog=SSOTCatalog(),
    )

    assert any(issue.code == "duplicate_structural_assertion" for issue in report.errors)
