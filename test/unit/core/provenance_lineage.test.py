"""OpenLineage event emission tests."""

import json

from relic.core.provenance.recorder import ProvenanceRecorder


def test_emit_lineage_event(tmp_path) -> None:
    recorder = ProvenanceRecorder(output_dir=tmp_path, run_id="run-1")
    recorder.emit_lineage_event(
        node_id="node_a",
        event_type="START",
        cache_key="cache_key_1",
        metrics={"hits": 1},
    )

    path = tmp_path / "lineage.jsonl"
    assert path.exists()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(lines[-1])
    assert event["event_type"] == "START"
    assert event["node_id"] == "node_a"
    assert event["run_id"] == "run-1"
    assert event["cache_key"] == "cache_key_1"
    assert event["metrics"]["hits"] == 1