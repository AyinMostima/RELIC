"""L1 tests for the parameter operation helpers and the params CLI command executor.

Module under test: relic/parameters/operations.py. A failing manager call
propagates; it is not turned into an empty list, a default value, False or a
synthetic validation message. The process-level CLI boundary that turns an
error into exit code 1 is relic/interfaces/cli/main.py::main.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import relic.parameters.operations as operations
from relic.parameters.types import ParameterCategory


class _StubManager:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.list_calls = []
        self.parameters = {"soil.decay_rate": SimpleNamespace(value=0.2)}

    def _maybe_fail(self) -> None:
        if self.error is not None:
            raise self.error

    def list(self, namespace=None, pattern=None, category=None):
        self._maybe_fail()
        self.list_calls.append(category)
        return ["soil.decay_rate"]

    def get(self, key, default=None, namespace=None, required=False):
        self._maybe_fail()
        if key != "soil.decay_rate":
            if required:
                raise KeyError(key)
            return default
        return 0.2

    def get_parameter(self, key):
        return SimpleNamespace(value=0.2, units="day-1", description="decay rate")

    def set_value(self, key, value, units=None, validate=True):
        self._maybe_fail()

    def validate_parameters(self, keys):
        self._maybe_fail()
        return []


@pytest.fixture
def use_manager(monkeypatch):
    def _install(manager, cli: bool = False):
        if cli:
            monkeypatch.setattr(
                operations,
                "get_global_manager",
                lambda preload_all=False, warmup_cache=False: manager,
            )
        else:
            monkeypatch.setattr(operations, "get_global_manager", lambda: manager)
        return manager

    return _install


class TestOperationHelpers:
    def test_list_failure_is_raised_not_returned_as_an_empty_list(self, use_manager) -> None:
        use_manager(_StubManager(RuntimeError("storage offline")))

        with pytest.raises(RuntimeError, match="storage offline"):
            operations.list_parameters(namespace="soil")

    def test_get_failure_is_raised_not_replaced_by_the_default(self, use_manager) -> None:
        use_manager(_StubManager(ValueError("unit conversion failed")))

        with pytest.raises(ValueError, match="unit conversion failed"):
            operations.get_parameter("soil.decay_rate", default=1.0)

    def test_set_failure_is_raised_not_returned_as_false(self, use_manager) -> None:
        use_manager(_StubManager(ValueError("value above bounds")))

        with pytest.raises(ValueError, match="value above bounds"):
            operations.set_parameter("soil.decay_rate", 5.0)

    def test_successful_set_returns_none(self, use_manager) -> None:
        use_manager(_StubManager())

        assert operations.set_parameter("soil.decay_rate", 0.3) is None

    def test_validation_failure_is_raised_not_reported_as_a_finding(self, use_manager) -> None:
        use_manager(_StubManager(RuntimeError("validator backend failed")))

        with pytest.raises(RuntimeError, match="validator backend failed"):
            operations.validate_parameters()

    def test_category_filter_uses_the_canonical_category_normalizer(self, use_manager) -> None:
        manager = use_manager(_StubManager())

        operations.list_parameters(category=" Physical ")

        assert manager.list_calls == [ParameterCategory.PHYSICAL]

    def test_unknown_category_is_rejected_instead_of_listing_everything(self, use_manager) -> None:
        use_manager(_StubManager())

        with pytest.raises(ValueError, match="no_such_category"):
            operations.list_parameters(category="no_such_category")


class TestParamsCommandExecutor:
    @staticmethod
    def _args(command: str, **values):
        return SimpleNamespace(params_command=command, verbose=False, **values)

    def test_manager_construction_failure_is_raised(self, monkeypatch) -> None:
        def _broken(preload_all=False, warmup_cache=False):
            raise RuntimeError("parameter database unreadable")

        monkeypatch.setattr(operations, "get_global_manager", _broken)

        with pytest.raises(RuntimeError, match="parameter database unreadable"):
            operations.execute_cli(self._args("list", namespace=None, pattern=None, category=None))

    def test_command_failure_is_raised(self, use_manager) -> None:
        use_manager(_StubManager(OSError("storage offline")), cli=True)

        with pytest.raises(OSError, match="storage offline"):
            operations.execute_cli(self._args("list", namespace=None, pattern=None, category=None))

    def test_list_command_applies_the_category_filter(self, use_manager) -> None:
        manager = use_manager(_StubManager(), cli=True)

        assert operations.execute_cli(
            self._args("list", namespace=None, pattern=None, category="physical")
        ) is True
        assert manager.list_calls == [ParameterCategory.PHYSICAL]

    def test_missing_required_key_is_a_failed_command(self, use_manager) -> None:
        use_manager(_StubManager(), cli=True)

        assert operations.execute_cli(self._args("get", key="soil.unknown", namespace=None)) is False

    def test_missing_subcommand_is_a_failed_command(self) -> None:
        assert operations.execute_cli(SimpleNamespace(params_command=None)) is False
