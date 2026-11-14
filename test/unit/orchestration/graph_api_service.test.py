"""Unit tests for orchestration API service helpers."""

from types import SimpleNamespace


def test_validate_graph_payload(monkeypatch):
    import relic.orchestration.api_service as svc
    import relic.orchestration.topology_algebra as topo_mod
    import relic.orchestration.validator as validator_mod

    err = SimpleNamespace(code="E1")
    warn = SimpleNamespace(code="W1")
    report = SimpleNamespace(errors=[err], warnings=[warn], assertion_audit={"present": False, "valid": True})

    monkeypatch.setattr(svc, "_load_graph_payload", lambda payload: "graph")
    monkeypatch.setattr(svc, "_build_registry", lambda: "registry")
    monkeypatch.setattr(validator_mod, "validate_graph", lambda graph, registry, strict=True: report)
    monkeypatch.setattr(topo_mod, "get_topology_signature", lambda graph: {"rank": 1, "nullity": 0, "conservation_vectors": []})

    result = svc.validate_graph_payload({"graph": {}}, strict=True)
    assert result["errors"] == [err.__dict__]
    assert result["warnings"] == [warn.__dict__]
    assert result["assertion_audit"]["valid"] is True
    assert "conservation_audit" in result["audit"]


def test_submit_graph_payload_records_history(monkeypatch):
    import relic.orchestration.api_service as svc
    import relic.orchestration.compiler as compiler_mod

    svc._RUN_HISTORY.clear()

    monkeypatch.setattr(svc, "_load_graph_payload", lambda payload: "graph")
    monkeypatch.setattr(svc, "_build_registry", lambda: "registry")
    monkeypatch.setattr(
        compiler_mod,
        "compile_graph",
        lambda graph, registry, auto_adapt=False, strict=True, include_topology=True: "plan",
    )
    monkeypatch.setattr(compiler_mod, "plan_to_dict", lambda plan: {"plan": plan, "report": {"audit": {"syntax_ok": True}}})

    run_id, report = svc.submit_graph_payload({"graph": {}}, strict=True)
    assert report["plan"] == "plan"
    assert report["report"]["audit"]["syntax_ok"] is True

    record = svc.get_run_history(run_id)
    assert record is not None
    assert record["status"] == "completed"
    status = svc.get_run_status(run_id)
    assert status["run_id"] == run_id
