"""Unit tests for API application state initialization (relic.core.services.api_runtime)."""

from types import SimpleNamespace

import pytest

from relic.core.services.api_runtime import initialize_api_state


def _dependencies():
    return {"optional_dependencies": {}}


def test_configuration_load_failure_stops_api_startup():
    app = SimpleNamespace(state=SimpleNamespace())

    def _unreadable_config():
        raise OSError("RELIC_CONFIG_PATH file is unreadable")

    with pytest.raises(OSError, match="unreadable"):
        initialize_api_state(app, _unreadable_config, _dependencies)

    assert not hasattr(app.state, "config")


def test_configuration_is_stored_on_the_app_state():
    app = SimpleNamespace(state=SimpleNamespace())
    validation = SimpleNamespace(is_valid=True)

    initialize_api_state(app, lambda: ({"simulation": {}}, validation, None), _dependencies)

    assert app.state.config == {"simulation": {}}
