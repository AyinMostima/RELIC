"""Behavioral tests for ParameterManager event helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from relic.parameters import events

pytestmark = pytest.mark.unit


def test_record_clamp_event_captures_auditable_utc_fields() -> None:
    manager = SimpleNamespace(validation_level="strict")
    spec = SimpleNamespace(key="soil.decay_rate", bounds=(0.0, 1.0))

    events.record_clamp_event(
        manager,
        spec,
        attempted_value=1.5,
        clamped_value=1.0,
        message="above upper bound",
    )

    assert len(manager._clamp_events) == 1
    event = manager._clamp_events[0]
    assert event["key"] == "soil.decay_rate"
    assert event["attempted_value"] == 1.5
    assert event["clamped_value"] == 1.0
    assert event["bounds"] == (0.0, 1.0)
    assert event["validation_level"] == "strict"
    assert event["message"] == "above upper bound"
    timestamp = datetime.fromisoformat(event["timestamp"])
    assert timestamp.utcoffset() == timedelta(0)


def test_get_clamp_events_returns_snapshot_and_supports_clear() -> None:
    manager = SimpleNamespace(
        _clamp_events=[{"key": "soil.decay_rate", "clamped_value": 1.0}]
    )

    snapshot = events.get_clamp_events(manager)
    snapshot[0]["key"] = "changed"

    assert manager._clamp_events[0]["key"] == "soil.decay_rate"
    cleared_snapshot = events.get_clamp_events(manager, clear=True)
    assert cleared_snapshot[0]["key"] == "soil.decay_rate"
    assert manager._clamp_events == []


def test_change_callback_registration_is_unique_and_removal_is_idempotent() -> None:
    manager = SimpleNamespace()

    def callback(key, old_value, new_value):
        return None

    events.add_change_callback(manager, callback)
    events.add_change_callback(manager, callback)

    assert manager.change_callbacks == [callback]
    events.remove_change_callback(manager, callback)
    events.remove_change_callback(manager, callback)
    assert manager.change_callbacks == []


def test_remove_change_callback_accepts_manager_without_registry() -> None:
    manager = SimpleNamespace()

    events.remove_change_callback(manager, lambda *_: None)

    assert not hasattr(manager, "change_callbacks")


def test_notify_change_records_timestamp_without_callbacks() -> None:
    manager = SimpleNamespace()

    events.notify_change(manager, "soil.decay_rate", 0.2, 0.3)

    assert len(manager.change_history) == 1
    record = manager.change_history[0]
    assert record["key"] == "soil.decay_rate"
    assert record["old_value"] == 0.2
    assert record["new_value"] == 0.3
    timestamp = datetime.fromisoformat(record["timestamp"])
    assert timestamp.utcoffset() == timedelta(0)


def test_notify_change_uses_registration_order_snapshot() -> None:
    manager = SimpleNamespace(change_callbacks=[])
    calls = []

    def first(key, old_value, new_value):
        calls.append("first")
        events.remove_change_callback(manager, first)

    def second(key, old_value, new_value):
        calls.append("second")

    events.add_change_callback(manager, first)
    events.add_change_callback(manager, second)

    events.notify_change(manager, "soil.decay_rate", 0.2, 0.3)
    events.notify_change(manager, "soil.decay_rate", 0.3, 0.4)

    assert calls == ["first", "second", "second"]


def test_notify_change_notifies_every_callback_then_raises_the_failures() -> None:
    """A failing callback neither stops the other callbacks nor disappears.

    The committed change stays recorded; the failures are raised together
    after every callback was notified.
    """
    calls = []

    def failing_callback(key, old_value, new_value):
        raise KeyError("callback failure")

    def succeeding_callback(key, old_value, new_value):
        calls.append((key, old_value, new_value))

    def second_failing_callback(key, old_value, new_value):
        raise ValueError("cache refresh failed")

    manager = SimpleNamespace(
        change_callbacks=[failing_callback, succeeding_callback, second_failing_callback]
    )

    with pytest.raises(ExceptionGroup, match="soil.decay_rate") as exc_info:
        events.notify_change(manager, "soil.decay_rate", 0.2, 0.3)

    assert calls == [("soil.decay_rate", 0.2, 0.3)]
    assert [type(error) for error in exc_info.value.exceptions] == [KeyError, ValueError]
    assert [record["new_value"] for record in manager.change_history] == [0.3]


def test_history_and_callbacks_receive_their_own_copies_of_container_values() -> None:
    """A change notification hands out copies of mapping and list values.

    The new value is the object the manager stores. A callback that changes
    what it receives changes neither that stored object, the recorded change
    history nor what the next callback receives. A tensor keeps its identity
    (architecture overview Sections 4.2 and 6.1).
    """
    import torch

    stored = {"k": 1, "inner": [1]}
    received = []

    def changing_callback(key, old_value, new_value):
        new_value["k"] = 99
        new_value["inner"].append(2)
        old_value.append(0)

    def observing_callback(key, old_value, new_value):
        received.append((old_value, new_value))

    manager = SimpleNamespace(change_callbacks=[changing_callback, observing_callback])
    events.notify_change(manager, "soil.options", [5], stored)

    assert stored == {"k": 1, "inner": [1]}
    assert received == [([5], {"k": 1, "inner": [1]})]
    assert manager.change_history[0]["old_value"] == [5]
    assert manager.change_history[0]["new_value"] == {"k": 1, "inner": [1]}
    assert manager.change_history[0]["new_value"] is not stored

    tensor = torch.tensor([1.0])
    seen = []
    manager = SimpleNamespace(change_callbacks=[lambda key, old_value, new_value: seen.append(new_value)])
    events.notify_change(manager, "soil.theta", None, tensor)
    assert seen[0] is tensor
    assert manager.change_history[0]["new_value"] is tensor
