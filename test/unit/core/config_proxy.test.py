"""Behavioral tests for the canonical Functor configuration proxy."""

from typing import Any

import pytest

from relic.core.config_proxy import ConfigProxy, ParametersProxy, StrictConfigProxy


class _FunctorStub:
    def __init__(
        self,
        config: dict[str, Any],
        parameter_values: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = "stub_functor"
        self._original_config = config
        self.parameter_values = parameter_values or {}
        self.error = error
        self.calls: list[str] = []

    def get_parameter_value(self, key: str) -> Any:
        """Mirror FunctorBase.get_parameter_value: no caller default; absent is None."""
        self.calls.append(key)
        if self.error is not None:
            raise self.error
        return self.parameter_values.get(key)


def test_get_prefers_parameter_manager_value() -> None:
    functor = _FunctorStub(
        {"rate": 1.0},
        parameter_values={"rate": 2.0},
    )

    assert ConfigProxy(functor).get("rate", 3.0) == 2.0
    assert functor.calls == ["rate"]


def test_get_falls_back_to_original_config_before_default() -> None:
    functor = _FunctorStub({"rate": 1.0})

    assert ConfigProxy(functor).get("rate", 3.0) == 1.0
    assert functor.calls == ["rate"]


def test_get_uses_default_only_when_parameter_and_config_are_missing() -> None:
    functor = _FunctorStub({})

    assert ConfigProxy(functor).get("missing", 3.0) == 3.0


@pytest.mark.parametrize(
    "access",
    [
        lambda functor: ConfigProxy(functor)["rate"],
        lambda functor: ConfigProxy(functor).get("rate", 3.0),
        lambda functor: ParametersProxy(functor)["rate"],
        lambda functor: ParametersProxy(functor).get("rate", 3.0),
        lambda functor: ParametersProxy(functor).get("rate", {"value": 3.0}),
    ],
)
def test_parameter_lookup_errors_propagate_instead_of_reading_raw_config(access) -> None:
    """A failed managed lookup (validation, units, bounds) is not absence: no raw value, no default."""
    functor = _FunctorStub(
        {"rate": 1.0, "parameters": {"rate": 1.0}},
        error=ValueError("rate outside bounds"),
    )

    with pytest.raises(ValueError, match="rate outside bounds"):
        access(functor)


def test_parameters_proxy_prefers_manager_then_nested_original() -> None:
    functor = _FunctorStub(
        {"parameters": {"fallback_rate": 1.0}},
        parameter_values={"managed_rate": 2.0},
    )
    parameters = ConfigProxy(functor)["parameters"]

    assert isinstance(parameters, ParametersProxy)
    assert parameters["managed_rate"] == 2.0
    assert parameters.get("fallback_rate", 3.0) == 1.0
    assert parameters.get("missing", 3.0) == 3.0


def test_parameters_proxy_preserves_legacy_value_mapping_shape() -> None:
    functor = _FunctorStub(
        {"parameters": {"rate": {"value": 1.0}}},
    )
    parameters = ParametersProxy(functor)

    assert parameters.get("rate", {"value": 3.0}) == {"value": 1.0}


def test_proxy_tracks_access_and_supports_mapping_comparison() -> None:
    functor = _FunctorStub({"rate": 1.0})
    proxy = ConfigProxy(functor)

    assert "rate" in proxy
    assert proxy == {"rate": 1.0}
    assert proxy == ConfigProxy(_FunctorStub({"rate": 1.0}))
    assert proxy != object()
    assert proxy["rate"] == 1.0
    assert proxy.get("rate") == 1.0
    assert "accesses=2" in repr(proxy)


@pytest.mark.parametrize("access_type", ["get", "getitem"])
def test_strict_proxy_rejects_direct_config_access(access_type: str) -> None:
    proxy = StrictConfigProxy(_FunctorStub({"rate": 1.0}))

    with pytest.raises(RuntimeError, match="Direct config access not allowed"):
        if access_type == "get":
            proxy.get("rate")
        else:
            proxy["rate"]
