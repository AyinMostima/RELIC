"""L1 tests for UnitSystem contract-based unit mappings and unit-module error contracts."""

from pathlib import Path

import pytest

import relic.core.units as core_units
import relic.core.units.definitions as unit_definitions
import relic.core.units.system as unit_system_module
import relic.utils.physical_constants as physical_constants_module

from relic.contracts import ContractRegistry
from relic.core.units import OutputUnitConfig, UnitSystem


def test_contract_unit_mapping_resolves_core_contract() -> None:
    # The SCM contract of the volumetric liquid soil water content theta (read by the
    # soil water models and equilibrium_sorption) declares m3 m-3.
    assert UnitSystem.get_standard_unit("soil_moisture") == "m3 m-3"


def test_contract_unit_lookup_excludes_generic_type_fallbacks() -> None:
    assert UnitSystem.get_contract_unit("air_temperature") == "K"
    assert UnitSystem.get_contract_unit("pressure") is None
    assert UnitSystem.get_standard_unit("pressure") == "Pa"
    assert UnitSystem.get_contract_unit("unknown_surrogate_feature") is None


def test_radiation_type_resolves_to_the_energy_flux_unit() -> None:
    """No state variable is named radiation; the type maps to W m-2."""
    assert UnitSystem.get_contract_unit("radiation") is None
    assert UnitSystem.get_standard_unit("radiation") == "W m-2"


def test_canonical_productivity_units_come_from_scm() -> None:
    assert (
        UnitSystem.get_standard_unit("gross_primary_productivity")
        == "kg C m-2 s-1"
    )
    assert (
        UnitSystem.get_standard_unit("net_primary_productivity")
        == "kg C m-2 s-1"
    )


def test_removed_productivity_aliases_have_no_heuristic_unit(monkeypatch) -> None:
    """Without a contract for gpp or npp no generic unit is guessed."""
    import relic.contracts.contract_loader as contract_loader

    monkeypatch.setattr(
        contract_loader, "load_contract_registry", lambda: ContractRegistry(strict=True)
    )
    monkeypatch.setattr(UnitSystem, "_contract_unit_map", None)
    monkeypatch.setattr(UnitSystem, "_contract_unit_map_lower", None)

    assert UnitSystem.get_standard_unit("gpp") == "unknown"
    assert UnitSystem.get_standard_unit("npp") == "unknown"


def test_output_preference_uses_scm_unit_category() -> None:
    config = OutputUnitConfig()

    assert (
        config.get_output_unit("gross_primary_productivity")
        == "g C m-2 day-1"
    )


def test_molecular_weight_api_is_not_exposed_by_core_units() -> None:
    assert not hasattr(core_units, "MOLECULAR_WEIGHTS")
    assert not hasattr(core_units, "get_molecular_weight")


def test_duplicate_molecular_weight_table_is_absent() -> None:
    assert not hasattr(unit_definitions, "MOLECULAR_WEIGHTS")
    assert not hasattr(unit_definitions, "get_molecular_weight")


def test_unit_system_keeps_no_second_variable_unit_table() -> None:
    """A variable's unit comes only from its SCM contract (get_contract_unit).

    The removed standardize_units converted co2_concentration, o3_concentration
    and ch4_concentration to a hardcoded mol mol-1: co2_concentration is ppm in
    the SSOT catalog, o3_concentration is retired (o3_mixing_ratio) and
    ch4_concentration has no declaration. The removed validate_state_units
    expected heterotrophic_respiration in kg C m-2 s-1 against its contract
    g C m-2 s-1. Neither had a caller in relic, the tests or the scripts.
    """
    assert not hasattr(UnitSystem, "standardize_units")
    assert not hasattr(UnitSystem, "validate_state_units")
    assert UnitSystem.get_standard_unit("o3_mixing_ratio") == "mol mol-1"
    assert UnitSystem.get_standard_unit("ozone_concentration") == "mol mol-1"
    assert UnitSystem.get_standard_unit("heterotrophic_respiration") == "g C m-2 s-1"


def test_standard_conditions_come_from_the_constants_ssot() -> None:
    assert unit_definitions.STANDARD_TEMPERATURE == physical_constants_module.get_physical_constant(
        "standard_temperature"
    )
    assert unit_definitions.STANDARD_PRESSURE == physical_constants_module.get_physical_constant(
        "standard_pressure"
    )


def test_unavailable_standard_conditions_fail_the_import(monkeypatch) -> None:
    """The definitions module must not swallow a constants lookup error at import time.

    The module source runs in a separate namespace, so the imported module and
    its enum classes stay untouched.
    """

    def _missing(name, *args, **kwargs):
        raise ValueError(f"Physical constant '{name}' not found in parameter database.")

    monkeypatch.setattr(physical_constants_module, "get_physical_constant", _missing)
    source_path = Path(unit_definitions.__file__)
    code = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")

    with pytest.raises(ValueError, match="standard_temperature"):
        exec(code, {"__name__": "relic.core.units._definitions_import_probe"})


def test_unit_convertibility_check_failure_is_raised_not_reported_invalid(monkeypatch) -> None:
    def _broken(*_args, **_kwargs):
        raise ValueError("unit parser broke")

    monkeypatch.setattr(unit_system_module.unit_converter, "can_convert", _broken)

    with pytest.raises(ValueError, match="unit parser broke"):
        UnitSystem.validate_unit("temperature", "degC")


def test_generic_conversion_error_points_to_the_variable_aware_owner() -> None:
    """UnitSystem.convert is the variable-independent entry of the unit converter.

    A depth rate to areal mass flux needs the liquid-water-equivalent rule of a
    declared liquid-water variable, so the error names the owner entry.
    """
    with pytest.raises(ValueError) as exc_info:
        UnitSystem.convert(5.0, "mm day-1", "kg m-2 s-1")

    message = str(exc_info.value)
    assert "unit_converter.convert" in message
    assert "variable_name" in message
    assert "Unsupported conversion" in str(exc_info.value.__cause__)


def test_variable_aware_owner_performs_the_liquid_water_equivalent_conversion() -> None:
    from relic.data.standards.unit_converter import ConversionContext, unit_converter

    converted = unit_converter.convert(
        5.0, "precipitation_flux", "mm day-1", "kg m-2 s-1", ConversionContext()
    )

    assert converted == pytest.approx(5.0 / 86400.0)


def test_generic_conversions_run_through_the_owner() -> None:
    assert UnitSystem.convert(25.0, "degC", "K") == pytest.approx(298.15)
    assert UnitSystem.convert(400.0, "ppm", "mol mol-1") == pytest.approx(4.0e-4)


def test_unit_system_has_no_converter_availability_switch() -> None:
    """The unit converter is a hard dependency; the always-true flag is removed."""
    assert not hasattr(unit_system_module, "UNIT_CONVERTER_AVAILABLE")
    assert "UNIT_CONVERTER_AVAILABLE" not in unit_system_module.__all__


def test_ignored_molecular_weight_argument_is_removed() -> None:
    with pytest.raises(TypeError):
        UnitSystem.convert(1.0, "g", "kg", molecular_weight=44.0)


def test_unit_validation_uses_the_converter_answer() -> None:
    assert UnitSystem.validate_unit("temperature", "degC") is True
    assert UnitSystem.validate_unit("temperature", "g m-2") is False
