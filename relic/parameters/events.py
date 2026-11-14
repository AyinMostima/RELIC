"""Parameter change notification and clamp event helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .ops.type_utils import caller_owned_value
from .spec import ParameterSpec
from .types import ChangeCallback, ParameterValue

logger = logging.getLogger(__name__)


def get_clamp_events(manager, clear: bool = False) -> list[dict[str, Any]]:
    """Return recorded clamp events with optional clearing."""
    events = [
        dict(event)
        for event in getattr(manager, "_clamp_events", [])
    ]
    if clear and hasattr(manager, "_clamp_events"):
        manager._clamp_events.clear()
    return events


def record_clamp_event(
    manager,
    spec: ParameterSpec,
    attempted_value: Any,
    clamped_value: float,
    message: str | None = None,
) -> None:
    """Record a clamp operation for auditing."""
    event = {
        "key": getattr(spec, "key", None),
        "attempted_value": attempted_value,
        "clamped_value": clamped_value,
        "bounds": getattr(spec, "bounds", None),
        "validation_level": str(getattr(manager, "validation_level", None)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if message:
        event["message"] = message
    if not hasattr(manager, "_clamp_events"):
        manager._clamp_events = []
    manager._clamp_events.append(event)


def add_change_callback(manager, callback: ChangeCallback) -> None:
    """Add parameter change notification callback."""
    if not hasattr(manager, "change_callbacks"):
        manager.change_callbacks = []
    if callback not in manager.change_callbacks:
        manager.change_callbacks.append(callback)


def remove_change_callback(manager, callback: ChangeCallback) -> None:
    """Remove parameter change notification callback."""
    if not hasattr(manager, "change_callbacks"):
        return
    if callback in manager.change_callbacks:
        manager.change_callbacks.remove(callback)


def notify_change(manager, key: str, old_value: ParameterValue, new_value: ParameterValue) -> None:
    """Notify callbacks of parameter change and record history.

    The change is already committed when this runs and stays committed. Every
    callback is notified even when an earlier one fails; the failures are then
    raised together as one ExceptionGroup naming the key, so a callback that
    could not react (for example a cache refresh) is never skipped silently.

    new_value is the object the manager stores. The history record and every
    callback receive their own copies of a container value (caller_owned_value,
    architecture overview Section 6.1), so no receiver changes the stored value
    or what another receiver sees; a tensor keeps its identity (Section 4.2).
    """
    change_record = {
        "key": key,
        "old_value": caller_owned_value(old_value),
        "new_value": caller_owned_value(new_value),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not hasattr(manager, "change_history"):
        manager.change_history = []
    manager.change_history.append(change_record)

    if not hasattr(manager, "change_callbacks"):
        return
    failures: list[Exception] = []
    for callback in tuple(manager.change_callbacks):
        try:
            callback(key, caller_owned_value(old_value), caller_owned_value(new_value))
        except Exception as exc:  # noqa: BLE001 - collected, raised below as one ExceptionGroup
            failures.append(exc)
    if failures:
        raise ExceptionGroup(
            f"{len(failures)} parameter change callback(s) failed after '{key}' changed "
            f"from {old_value!r} to {new_value!r} (the change stays committed)",
            failures,
        )
