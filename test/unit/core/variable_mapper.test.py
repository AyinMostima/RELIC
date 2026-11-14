from __future__ import annotations

from typing import Any

from relic.core.variable_mapper import VariableNameMapper


class StubParameterManager:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, key: str, **kwargs: Any) -> Any:
        self.calls.append((key, kwargs))
        if key not in self.payloads and kwargs.get("required"):
            raise KeyError(key)
        return self.payloads.get(key)


def _manager(
    *,
    external: dict[str, Any] | None = None,
    scale: dict[str, Any] | None = None,
    aliases: dict[str, Any] | None = None,
) -> StubParameterManager:
    return StubParameterManager(
        {
            "coupling.external.variables.variable_definitions": external or {},
            "coupling.scales.defaults.variables": scale or {},
            "coupling.external.variables.aliases": aliases,
        }
    )


def test_identity_mapping_without_aliases() -> None:
    mapper = VariableNameMapper()

    assert mapper.to_standard("soil_carbon") == "soil_carbon"
    assert mapper.to_module("soil_carbon") == "soil_carbon"
    assert mapper.map_name("soil_carbon", target="unsupported") == "soil_carbon"
    assert mapper.conflicts == []


def test_refresh_loads_catalog_and_explicit_aliases() -> None:
    mapper = VariableNameMapper()
    manager = _manager(
        external={"temp": {"standard_name": "air_temperature"}},
        scale={"sm": {"standard_name": "soil_moisture"}},
        aliases={" lai ": " leaf_area_index "},
    )

    mapper.refresh_from_parameter_manager(manager)

    assert mapper.to_standard("temp") == "air_temperature"
    assert mapper.to_standard("sm") == "soil_moisture"
    assert mapper.to_standard("lai") == "leaf_area_index"
    assert mapper.to_module("air_temperature") == "temp"
    assert mapper.to_module("leaf_area_index") == "lai"
    assert [key for key, _ in manager.calls] == [
        "coupling.external.variables.variable_definitions",
        "coupling.scales.defaults.variables",
        "coupling.external.variables.aliases",
    ]


def test_contradicting_catalog_declarations_raise_and_explicit_overrides_are_recorded() -> None:
    import pytest

    mapper = VariableNameMapper()
    with pytest.raises(ValueError, match="temp"):
        mapper.refresh_from_parameter_manager(
            _manager(
                external={"temp": {"standard_name": "air_temperature"}},
                scale={"temp": {"standard_name": "soil_temperature"}},
            )
        )

    mapper.refresh_from_parameter_manager(
        _manager(
            external={"temp": {"standard_name": "air_temperature"}},
            aliases={"temp": "canopy_temperature"},
        )
    )
    assert mapper.to_standard("temp") == "canopy_temperature"
    assert mapper.conflicts == ["alias_conflict:temp:air_temperature:canopy_temperature"]


def test_conflicts_property_returns_a_copy() -> None:
    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(
        _manager(
            external={"temp": {"standard_name": "air_temperature"}},
            aliases={"temp": "canopy_temperature"},
        )
    )

    reported = mapper.conflicts
    reported.append("caller mutation")

    assert "caller mutation" not in mapper.conflicts


def test_refresh_ignores_catalog_entries_without_a_standard_name() -> None:
    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(
        _manager(
            external={
                "__metadata__": {"standard_name": "ignored"},
                "invalid_payload": "not-a-mapping",
                "": {"standard_name": ""},
                "valid": {"standard_name": "standard_valid"},
            },
            aliases={"valid_explicit": "standard_explicit"},
        )
    )

    assert mapper.to_standard("valid") == "standard_valid"
    assert mapper.to_standard("valid_explicit") == "standard_explicit"
    assert mapper.to_standard("") == ""


def test_invalid_explicit_aliases_raise_instead_of_being_skipped() -> None:
    """coupling.external.variables.aliases is configuration: a malformed entry is an error."""
    import pytest

    for aliases, error in (
        ({7: "numeric_alias"}, TypeError),
        ({"": "air_temperature"}, ValueError),
        ({"blank_target": " "}, ValueError),
        (["not", "a", "mapping"], TypeError),
    ):
        mapper = VariableNameMapper()
        with pytest.raises(error):
            mapper.refresh_from_parameter_manager(
                _manager(external={"valid": {"standard_name": "standard_valid"}}, aliases=aliases)
            )


def test_required_catalog_failure_raises_and_keeps_the_last_valid_mapping() -> None:
    import pytest

    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(
        _manager(external={"temp": {"standard_name": "air_temperature"}})
    )
    failing_manager = StubParameterManager({})

    with pytest.raises(KeyError, match="coupling.external.variables.variable_definitions"):
        mapper.refresh_from_parameter_manager(failing_manager)

    assert mapper.to_standard("temp") == "air_temperature"


def test_alias_catalog_load_errors_propagate() -> None:
    import pytest

    class _Failing(StubParameterManager):
        def get(self, key, **kwargs):
            if key == "coupling.external.variables.aliases":
                raise ValueError("aliases failed validation")
            return super().get(key, **kwargs)

    mapper = VariableNameMapper()
    with pytest.raises(ValueError, match="aliases failed validation"):
        mapper.refresh_from_parameter_manager(
            _Failing(
                {
                    "coupling.external.variables.variable_definitions": {},
                    "coupling.scales.defaults.variables": {},
                }
            )
        )


def test_several_names_for_one_standard_identity_raise_instead_of_picking_the_first() -> None:
    import pytest

    mapper = VariableNameMapper()
    with pytest.raises(ValueError, match="air_temperature"):
        mapper.refresh_from_parameter_manager(
            _manager(
                external={
                    "temp": {"standard_name": "air_temperature"},
                    "tair": {"standard_name": "air_temperature"},
                }
            )
        )


def test_qualified_entries_keep_their_name_and_never_map_to_the_mean() -> None:
    """CF-1.8 7.3: cell_methods qualify the identity; daily extrema are not air_temperature."""
    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(
        _manager(
            external={
                "air_temperature": {"standard_name": "air_temperature", "units": "K"},
                "daily_maximum_air_temperature": {
                    "standard_name": "air_temperature",
                    "cell_methods": "time: maximum",
                    "units": "K",
                },
                "daily_minimum_air_temperature": {
                    "standard_name": "air_temperature",
                    "cell_methods": "time: minimum",
                    "units": "K",
                },
            }
        )
    )
    assert mapper.to_standard("daily_maximum_air_temperature") == "daily_maximum_air_temperature"
    assert mapper.to_standard("daily_minimum_air_temperature") == "daily_minimum_air_temperature"
    assert mapper.to_module("air_temperature") == "air_temperature"


def test_parameter_catalog_keeps_the_daily_extrema_apart() -> None:
    from relic.parameters import get_global_manager

    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(get_global_manager())
    assert mapper.to_standard("daily_maximum_air_temperature") == "daily_maximum_air_temperature"
    assert mapper.to_standard("daily_minimum_air_temperature") == "daily_minimum_air_temperature"


def test_standard_mapped_view_is_non_mutating_and_preserves_existing_key() -> None:
    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(
        _manager(external={"temp": {"standard_name": "air_temperature"}})
    )
    state = {"temp": 280.0, "air_temperature": 285.0}

    mapped = mapper.create_mapped_view(state, target="standard")

    assert mapped == {"temp": 280.0, "air_temperature": 285.0}
    assert mapped is not state
    assert state == {"temp": 280.0, "air_temperature": 285.0}


def test_module_mapped_view_adds_alias_without_mutating_input() -> None:
    mapper = VariableNameMapper()
    mapper.refresh_from_parameter_manager(
        _manager(external={"temp": {"standard_name": "air_temperature"}})
    )
    state = {"air_temperature": 285.0}

    mapped = mapper.create_mapped_view(state, target="module")

    assert mapped == {"air_temperature": 285.0, "temp": 285.0}
    assert state == {"air_temperature": 285.0}


def test_invalid_mapped_view_target_returns_independent_copy() -> None:
    mapper = VariableNameMapper()
    state = {"soil_carbon": 1.0}

    mapped = mapper.create_mapped_view(state, target="invalid")

    assert mapped == state
    assert mapped is not state
