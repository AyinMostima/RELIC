"""L1 tests for the BaseManager cleanup error contract.

Module under test: relic/core/managers/base.py. A failing or interrupted
cleanup must raise and leave the manager in ERROR, so a later cleanup() or
reset() retries it; it must not log the failure and report a clean manager.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import pytest

import relic.core.managers as managers_package
import relic.core.managers.base as base_module
from relic.core.managers.base import (
    BaseManager,
    DeviceAwareManagerMixin,
    ManagerState,
)
from relic.core.managers.exceptions import ManagerError


class _ProbeManager(BaseManager):
    """Minimal manager whose cleanup can be made to fail."""

    def __init__(self, cleanup_error: BaseException | None = None) -> None:
        super().__init__()
        self.cleanup_error = cleanup_error
        self.cleanup_calls = 0

    def _do_initialize(self) -> None:
        return None

    def _do_configure(self, config: Dict[str, Any]) -> None:
        return None

    def _do_validate(self) -> List[str]:
        return []

    def _do_cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _no_warning:
    """Context manager asserting that no warning is emitted inside it."""

    def __enter__(self) -> "_no_warning":
        self._catcher = warnings.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._catcher.__exit__(*exc_info)
        assert [str(item.message) for item in self._records] == []


def _ready(manager: BaseManager) -> BaseManager:
    manager.initialize()
    manager.configure({})
    manager.validate()
    assert manager.state == ManagerState.READY
    return manager


class TestBaseManagerCleanup:
    def test_successful_cleanup_returns_to_uninitialized(self) -> None:
        manager = _ready(_ProbeManager())

        manager.cleanup()

        assert manager.state == ManagerState.UNINITIALIZED
        assert manager.error_info is None
        assert manager.cleanup_calls == 1

    def test_do_cleanup_failure_is_raised_not_forced_to_uninitialized(self) -> None:
        manager = _ready(_ProbeManager(cleanup_error=OSError("handle stuck")))

        with _no_warning():
            with pytest.raises(ManagerError, match="handle stuck") as exc_info:
                manager.cleanup()

        assert isinstance(exc_info.value.__cause__, OSError)
        assert manager.state == ManagerState.ERROR
        assert manager.error_info is exc_info.value

    def test_reset_retries_a_failed_cleanup_and_recovers(self) -> None:
        manager = _ready(_ProbeManager(cleanup_error=OSError("handle stuck")))
        with pytest.raises(ManagerError):
            manager.cleanup()

        with pytest.raises(ManagerError, match="handle stuck"):
            manager.reset()
        assert manager.state == ManagerState.ERROR

        manager.cleanup_error = None
        manager.reset()

        assert manager.state == ManagerState.UNINITIALIZED
        assert manager.error_info is None
        assert manager.cleanup_calls == 3


class TestInterruptedCleanup:
    def test_interrupted_cleanup_leaves_error_and_the_next_cleanup_retries(self) -> None:
        manager = _ready(_ProbeManager(cleanup_error=KeyboardInterrupt()))

        with pytest.raises(KeyboardInterrupt):
            manager.cleanup()
        assert manager.state == ManagerState.ERROR

        manager.cleanup_error = None
        manager.cleanup()

        assert manager.cleanup_calls == 2
        assert manager.state == ManagerState.UNINITIALIZED
        assert manager.error_info is None


def test_removed_manager_helpers_are_absent() -> None:
    """AsyncBaseManager, SingletonManagerMixin, the lifecycle callbacks and the
    unused device-memory and tensor-allocation helpers had no user in relic, test,
    tutorial, examples, scripts or docs and were removed."""
    for name in ("AsyncBaseManager", "SingletonManagerMixin", "LifecycleCallback"):
        assert not hasattr(base_module, name)
        assert not hasattr(managers_package, name)
    assert not hasattr(BaseManager, "add_cleanup_callback")
    assert not hasattr(BaseManager, "add_initialization_callback")
    assert not hasattr(DeviceAwareManagerMixin, "get_memory_stats")
    assert not hasattr(DeviceAwareManagerMixin, "clear_memory")
    assert not hasattr(DeviceAwareManagerMixin, "allocate_tensor")
    manager = _ProbeManager()
    assert not hasattr(manager, "_initialization_callbacks")


class _DeviceProbe(DeviceAwareManagerMixin, _ProbeManager):
    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.device_changes = []

    def _on_device_changed(self, old_device, new_device) -> None:
        self.device_changes.append((old_device, new_device))


def test_device_migration_switches_the_device_and_calls_the_hook() -> None:
    probe = _DeviceProbe()
    assert not hasattr(probe, "_allocated_tensors")

    probe.to_device("meta")

    assert probe.device.type == "meta"
    assert [(old.type, new.type) for old, new in probe.device_changes] == [("cpu", "meta")]
