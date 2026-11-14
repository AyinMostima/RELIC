"""L1 tests for ProvenanceRecorder state updates and environment records."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import relic.core.provenance.recorder as recorder_module
from relic.core.provenance.recorder import ProvenanceRecorder


def test_record_state_update(tmp_path: Path) -> None:
    recorder = ProvenanceRecorder(output_dir=tmp_path)

    artifacts = {"compile_report": tmp_path / "compile_report.json"}
    artifacts["compile_report"].write_text("{}", encoding="utf-8")

    recorder.record_run_bundle(artifacts=artifacts, input_files=[])

    recorder.record_state_update(
        step=0,
        variable="biosphere.leaf_carbon",
        pre_value=torch.tensor(1.0),
        post_value=torch.tensor(1.1),
        functor_increments={"Photosynthesis": torch.tensor(0.1)},
        unit="kg m-2",
    )

    prov_path = tmp_path / "prov.jsonld"
    payload = json.loads(prov_path.read_text(encoding="utf-8"))

    assert "entity" in payload
    assert "activity" in payload
    assert f"state:{recorder.run_id}:biosphere.leaf_carbon@step1" in payload["entity"]
    assert f"functor:{recorder.run_id}:Photosynthesis@step0" in payload["activity"]


def _event(variable="pool", unit="kg m-2"):
    return dict(step=0, variable=variable, pre_value=torch.tensor([1.0, 2.0], dtype=torch.float32),
                post_value=torch.tensor([2.0, 3.0], dtype=torch.float32),
                functor_increments={"process": torch.tensor([1.0, 1.0], dtype=torch.float32)}, unit=unit)


def _files(path):
    return {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


def test_same_notification_replays_without_writes_or_duplicate_relations(tmp_path, monkeypatch):
    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    recorder.record_state_update(**_event())
    before = _files(tmp_path)
    monkeypatch.setattr(recorder_module, "_write_json", lambda *a, **k: pytest.fail("Replay must not rewrite PROV"))
    recorder.record_state_update(**_event())
    assert _files(tmp_path) == before
    payload = json.loads((tmp_path / "prov.jsonld").read_text())
    assert len(payload["wasGeneratedBy"]) == len(payload["wasDerivedFrom"]) == 1


@pytest.mark.parametrize("change", ["values", "dtype", "shape", "unit", "increments"])
def test_same_identity_conflicts_on_complete_content_not_only_sum(tmp_path, change):
    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    recorder.record_state_update(**_event())
    before = _files(tmp_path)
    changed = _event()
    if change == "values":
        changed["post_value"] = torch.tensor([3.0, 2.0], dtype=torch.float32)
    elif change == "dtype":
        changed["post_value"] = changed["post_value"].double()
    elif change == "shape":
        changed["post_value"] = changed["post_value"].reshape(1, 2)
    elif change == "unit":
        changed["unit"] = "g m-2"
    else:
        changed["functor_increments"] = {"process": torch.tensor([0.0, 2.0], dtype=torch.float32)}
    with pytest.raises(ValueError, match="conflict"):
        recorder.record_state_update(**changed)
    assert _files(tmp_path) == before


def test_distinct_runs_do_not_overwrite_state_or_activity_identity(tmp_path):
    for run_id in ("run-a", "run-b"):
        ProvenanceRecorder(tmp_path, run_id=run_id).record_state_update(**_event())
    payload = json.loads((tmp_path / "prov.jsonld").read_text())
    assert set(payload["entity"]) == {"state:run-a:pool@step1", "state:run-b:pool@step1"}
    assert len(payload["activity"]) == 2
    rows = [json.loads(line) for line in (tmp_path / "runtime.log.jsonl").read_text().splitlines()]
    assert [row["run_id"] for row in rows] == ["run-a", "run-b"]


def test_shared_functor_activity_keeps_each_variable_contribution(tmp_path):
    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    recorder.record_state_update(**_event("carbon", "g m-2"))
    recorder.record_state_update(**_event("water", "kg m-2"))
    payload = json.loads((tmp_path / "prov.jsonld").read_text())
    assert len(payload["activity"]) == 1
    assert "relic:delta_unit" not in next(iter(payload["activity"].values()))
    assert {row["relic:delta_unit"] for row in payload["wasGeneratedBy"]} == {"g m-2", "kg m-2"}


@pytest.mark.parametrize("after_append", [False, True])
def test_partial_notification_retry_reconciles_actual_files(tmp_path, monkeypatch, after_append):
    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    append = recorder_module._append_runtime_event
    def interrupt(path, event):
        if after_append:
            append(path, event)
        raise KeyboardInterrupt("interrupted notification")
    with monkeypatch.context() as local:
        local.setattr(recorder_module, "_append_runtime_event", interrupt)
        with pytest.raises(KeyboardInterrupt):
            recorder.record_state_update(**_event())
    prov_before = (tmp_path / "prov.jsonld").read_bytes()
    recorder.record_state_update(**_event())
    assert (tmp_path / "prov.jsonld").read_bytes() == prov_before
    assert len((tmp_path / "runtime.log.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("name,content", [
    ("prov.jsonld", "broken"), ("prov.jsonld", '{"entity":{},"entity":{}}'),
    ("runtime.log.jsonl", '{"event":'), ("runtime.log.jsonl", '[]\n'),
])
def test_bad_existing_record_is_preserved_and_not_replaced(tmp_path, name, content):
    (tmp_path / name).write_text(content)
    before = _files(tmp_path)
    with pytest.raises(ValueError):
        ProvenanceRecorder(tmp_path, run_id="run-a").record_state_update(**_event())
    assert _files(tmp_path) == before


def test_bundle_sealing_preserves_state_lineage_and_repeated_relations(tmp_path):
    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    recorder.record_state_update(**_event())
    state_doc = json.loads((tmp_path / "prov.jsonld").read_text())
    report = tmp_path / "report.json"
    report.write_text("{}")
    recorder.record_run_bundle({"report": report})
    recorder.record_run_bundle({"report": report})
    sealed = json.loads((tmp_path / "prov.jsonld").read_text())
    for identity, value in state_doc["entity"].items():
        assert sealed["entity"][identity] == value
    for relation in state_doc["wasGeneratedBy"]:
        assert sealed["wasGeneratedBy"].count(relation) == 1
    assert recorder.verify_run_bundle()["valid"]


def test_frozen_notification_context_survives_recorder_rebinding(tmp_path):
    recorder = ProvenanceRecorder(tmp_path / "old", run_id="old-run")
    context = recorder.get_notification_context()
    recorder.run_id = "new-run"
    recorder.output_dir = tmp_path / "new"
    recorder.record_state_update(**_event(), notification_context=context)
    assert not (tmp_path / "new").exists()
    payload = json.loads((tmp_path / "old" / "prov.jsonld").read_text())
    assert "state:old-run:pool@step1" in payload["entity"]


@pytest.mark.parametrize("after_replace", [False, True])
def test_atomic_record_cancellation_retains_complete_files_for_retry(tmp_path, monkeypatch, after_replace):
    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    recorder.record_state_update(**_event())
    original = _files(tmp_path)
    replace = recorder_module.os.replace
    def cancel_replace(source, target):
        if Path(target).name == "prov.jsonld":
            if after_replace:
                replace(source, target)
            raise KeyboardInterrupt("interrupted publication")
        return replace(source, target)
    with monkeypatch.context() as local:
        local.setattr(recorder_module.os, "replace", cancel_replace)
        with pytest.raises(KeyboardInterrupt):
            recorder.record_state_update(**_event("second"))
    assert not list(tmp_path.glob("*.tmp"))
    if not after_replace:
        assert _files(tmp_path) == original
    recorder.record_state_update(**_event("second"))
    payload = json.loads((tmp_path / "prov.jsonld").read_text())
    assert len(payload["entity"]) == 2
    assert len((tmp_path / "runtime.log.jsonl").read_text().splitlines()) == 2


def test_sealing_invalid_prov_does_not_write_other_bundle_files(tmp_path):
    (tmp_path / "prov.jsonld").write_text('{"activity":[]}')
    report = tmp_path / "report.json"
    report.write_text("{}")
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        ProvenanceRecorder(tmp_path, run_id="run-a").record_run_bundle({"report": report})
    assert _files(tmp_path) == before


@pytest.mark.parametrize("run_id", ["", " ", False])
def test_explicit_invalid_run_identity_is_not_replaced_by_a_default(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        ProvenanceRecorder(tmp_path, run_id=run_id)


def test_archive_content_conflict_leaves_the_verified_bundle_unchanged(tmp_path):
    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    report = tmp_path / "report.json"
    report.write_text("{}")
    recorder.record_run_bundle({"report": report}, metadata={"mode": "a"})
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="conflict"):
        recorder.record_run_bundle({"report": report}, metadata={"mode": "b"})
    assert _files(tmp_path) == before
    assert recorder.verify_run_bundle()["valid"]


@pytest.mark.parametrize("dtype,value", [(torch.float32, 3e38), (torch.float64, 1e308)])
def test_finite_tensor_notification_survives_summary_overflow(tmp_path, dtype, value):
    event = _event()
    event["pre_value"] = event["post_value"] = torch.tensor([value, value], dtype=dtype)
    event["functor_increments"] = {"process": torch.zeros(2, dtype=dtype)}
    ProvenanceRecorder(tmp_path, run_id="run-a").record_state_update(**event)
    payload = json.loads((tmp_path / "prov.jsonld").read_text())
    entity = payload["entity"]["state:run-a:pool@step1"]
    assert len(entity["relic:content_sha256"]) == 64
    if dtype == torch.float32:
        assert entity["relic:post_value_sum"] == pytest.approx(6e38)
    else:
        assert entity["relic:post_value_sum"] is None


def test_notification_summary_uses_the_same_frozen_values_as_its_digest(tmp_path, monkeypatch):
    from relic.core.simulator import checkpoint
    event = _event()
    encode = checkpoint._encode_checkpoint_value
    def freeze_then_mutate(value, *, path):
        frozen = encode(value, path=path)
        if path == "provenance_notification":
            event["post_value"].zero_()
        return frozen
    monkeypatch.setattr(checkpoint, "_encode_checkpoint_value", freeze_then_mutate)
    ProvenanceRecorder(tmp_path, run_id="run-a").record_state_update(**event)
    payload = json.loads((tmp_path / "prov.jsonld").read_text())
    assert payload["entity"]["state:run-a:pool@step1"]["relic:post_value_sum"] == 5.0


@pytest.mark.parametrize(
    "broken_metadata",
    [{}, {"Name": None}, {"Name": ""}],
    ids=["missing-header", "none-header", "empty-header"],
)
def test_env_snapshot_does_not_leave_out_a_nameless_distribution(monkeypatch, broken_metadata):
    """A distribution that cannot be named raises instead of vanishing from the snapshot."""
    distributions = [
        SimpleNamespace(metadata={"Name": "relic", "Version": "1.0"}, version="1.0"),
        SimpleNamespace(metadata=broken_metadata, version="0.1", _path="site-packages/broken.dist-info"),
    ]
    monkeypatch.setattr(recorder_module.metadata, "distributions", lambda: iter(distributions))

    with pytest.raises(RuntimeError, match="broken.dist-info"):
        recorder_module._list_packages()


def test_env_snapshot_lists_named_distributions(monkeypatch):
    distributions = [
        SimpleNamespace(metadata={"Name": "torch", "Version": "2.13.0"}, version="2.13.0"),
        SimpleNamespace(metadata={"Name": "numpy", "Version": "2.0.0"}, version="2.0.0"),
    ]
    monkeypatch.setattr(recorder_module.metadata, "distributions", lambda: iter(distributions))

    assert recorder_module._list_packages() == ["numpy==2.0.0", "torch==2.13.0"]


def test_git_status_run_failure_is_raised_not_recorded_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "git")

    def _blocked(*_args, **_kwargs):
        raise PermissionError("git blocked")

    monkeypatch.setattr(recorder_module.subprocess, "run", _blocked)

    with pytest.raises(PermissionError, match="git blocked"):
        recorder_module._get_git_info(tmp_path)


def test_missing_git_executable_is_recorded_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(
        recorder_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("git must not run without an executable"),
    )

    info = recorder_module._get_git_info(tmp_path)

    assert info["available"] is False
    assert info["head"] is None
    assert info["repo_root"] == str(tmp_path)
