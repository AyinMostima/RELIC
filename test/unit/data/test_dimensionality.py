"""Unit tests for SI dimensionality governance."""

from __future__ import annotations

from typing import Any, Dict

import pytest

import relic.data.standards.dimensionality as dimensionality_module
from relic.data.standards.dimensionality import (
    Dimensionality,
    dimension_category_from_unit,
    dimensionality_from_unit,
    is_declared_dimension_compatible,
)
from relic.orchestration.ssot_catalog import SSOTCatalog


def _require_dim(unit: str) -> Dimensionality:
    dim = dimensionality_from_unit(unit)
    assert dim is not None, f"Expected dimensionality for unit '{unit}'"
    return dim


def _assert_addition_compatible(lhs: Dimensionality, rhs: Dimensionality) -> None:
    if not lhs.is_compatible(rhs):
        raise ValueError("Cannot add variables with incompatible dimensions.")


def test_kelvin_is_temperature() -> None:
    assert _require_dim("K") == Dimensionality(Theta=1)


def test_watt_per_m2_exponents() -> None:
    dim = _require_dim("W m-2")
    assert dim.M == 1
    assert dim.L == 0
    assert dim.T == -3


def test_dimensionless_ppm() -> None:
    dim = _require_dim("ppm")
    assert dim.is_dimensionless()


def test_compatible_units() -> None:
    lhs = _require_dim("kg m-2 s-1")
    rhs = _require_dim("g m-2 s-1")
    assert lhs.is_compatible(rhs)


def test_incompatible_addition() -> None:
    temperature = _require_dim("K")
    radiative_flux = _require_dim("W m-2")
    with pytest.raises(ValueError):
        _assert_addition_compatible(temperature, radiative_flux)


def test_has_time_inverse() -> None:
    dim = _require_dim("kg m-2 s-1")
    assert dim.has_time_inverse()


@pytest.mark.parametrize("unit", ["s day-1", "h s-1", "min s-1", "s yr-1"])
def test_same_dimension_time_ratios_are_dimensionless(unit: str) -> None:
    assert _require_dim(unit).is_dimensionless()


@pytest.mark.parametrize(
    "unit",
    [
        "g C m-2",
        "g N m-2",
        "g P m-2",
        "kg C m-2",
        "kg N m-2",
        "kg P m-2",
    ],
)
def test_element_annotations_preserve_mass_per_area_dimension(unit: str) -> None:
    assert _require_dim(unit) == Dimensionality(M=1, L=-2)


@pytest.mark.parametrize(
    "unit",
    ["umol photons m-2 s-1", "umol CO2 m-2 s-1", "mol m-2 s-1"],
)
def test_amount_flux_dimension_and_category(unit: str) -> None:
    assert _require_dim(unit) == Dimensionality(N=1, L=-2, T=-1)
    assert dimension_category_from_unit(unit) == "amount_flux"


def test_specific_area_dimension() -> None:
    assert _require_dim("m2 g-1") == Dimensionality(M=-1, L=2)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("deg-days", Dimensionality(T=1, Theta=1)),
        ("degC_day", Dimensionality(T=1, Theta=1)),
        ("K day-0.5", Dimensionality(T=-0.5, Theta=1)),
    ],
)
def test_thermal_time_dimensions(unit: str, expected: Dimensionality) -> None:
    assert _require_dim(unit) == expected


@pytest.mark.parametrize("unit", ["ppm", "ppb", "ppbv", "mol mol-1"])
def test_mixing_ratios_are_si_dimensionless(unit: str) -> None:
    assert _require_dim(unit).is_dimensionless()
    assert dimension_category_from_unit(unit) == "dimensionless"


def test_standalone_c_is_electric_charge_not_carbon_annotation() -> None:
    assert _require_dim("C") == Dimensionality(T=1, Q=1)
    assert _require_dim("c") == Dimensionality(Theta=1)


def test_dimensionality_owner_does_not_depend_on_converter_instance() -> None:
    assert not hasattr(dimensionality_module, "unit_" + "converter")


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("g glucose g-1 biomass", Dimensionality()),
        ("hours", Dimensionality(T=1)),
        ("mol e- mol-1 photon", Dimensionality()),
        ("cm3 molecule-1 s-1", Dimensionality(L=3, T=-1, N=-1)),
        ("kg product m-2", Dimensionality(M=1, L=-2)),
        ("degrees_north", Dimensionality()),
        ("kg DM kg-1 body weight day-1", Dimensionality(T=-1)),
        ("MJ m-2 day-1", Dimensionality(M=1, T=-3)),
        ("nmol O3 m-2 s-1", Dimensionality(N=1, L=-2, T=-1)),
        ("mol O3 m-2 s-1", Dimensionality(N=1, L=-2, T=-1)),
        ("mol O3 m-2", Dimensionality(N=1, L=-2)),
        ("dS m-1", Dimensionality(M=-1, L=-3, T=3, Q=2)),
        ("cal J-1", Dimensionality()),
        ("kJ J-1", Dimensionality()),
        ("MJ J-1", Dimensionality()),
    ],
)
def test_active_parameter_unit_vocabulary_has_si_dimensions(
    unit: str,
    expected: Dimensionality,
) -> None:
    assert _require_dim(unit) == expected


@pytest.mark.parametrize(
    ("unit", "expected", "category"),
    [
        ("C mol-1", Dimensionality(T=1, Q=1, N=-1), "charge_per_mole"),
        ("H m-1", Dimensionality(M=1, L=1, T=-2, Q=-2), "permeability"),
        (
            "M atm-1",
            Dimensionality(M=-1, L=-2, T=2, N=1),
            "amount_energy_inverse",
        ),
        ("J mol-1", Dimensionality(M=1, L=2, T=-2, N=-1), "energy_per_amount"),
        (
            "J mol-1 K-1",
            Dimensionality(M=1, L=2, T=-2, Theta=-1, N=-1),
            "entropy_per_amount",
        ),
        ("K-1", Dimensionality(Theta=-1), "inverse_temperature"),
        ("kg yr-1", Dimensionality(M=1, T=-1), "mass_per_time"),
        (
            "J kg-1 K-1",
            Dimensionality(L=2, T=-2, Theta=-1),
            "specific_heat_capacity",
        ),
        ("m-1", Dimensionality(L=-1), "inverse_length"),
        (
            "mV",
            Dimensionality(M=1, L=2, T=-3, Q=-1),
            "electric_potential",
        ),
        ("animals ha-1", Dimensionality(L=-2), "stocking_density"),
        (
            "kg m-3 s-1",
            Dimensionality(M=1, L=-3, T=-1),
            "mass_volumetric_rate",
        ),
        (
            "mol m-3 s-1",
            Dimensionality(N=1, L=-3, T=-1),
            "amount_volumetric_rate",
        ),
        ("J s", Dimensionality(M=1, L=2, T=-1), "action"),
        ("L kg-1", Dimensionality(M=-1, L=3), "distribution_coefficient"),
        ("Pa-1", Dimensionality(M=-1, L=1, T=2), "inverse_pressure"),
        (
            "cm3 molecule-1 s-1",
            Dimensionality(L=3, T=-1, N=-1),
            "second_order_rate_constant",
        ),
        (
            "m s-0.5",
            Dimensionality(L=1, T=-0.5),
            "boundary_layer_transfer_coefficient",
        ),
        (
            "dS m-1",
            Dimensionality(M=-1, L=-3, T=3, Q=2),
            "electrical_conductivity",
        ),
        ("K day-0.5", Dimensionality(T=-0.5, Theta=1), "temperature_volatility"),
    ],
)
def test_case_sensitive_and_derived_dimension_categories(
    unit: str,
    expected: Dimensionality,
    category: str,
) -> None:
    assert _require_dim(unit) == expected
    assert dimension_category_from_unit(unit) == category


@pytest.mark.parametrize(
    ("unit", "declared_dimension"),
    [
        ("g C m-2", "mass_per_area"),
        ("umol m-2 s-1", "molar_flux"),
        ("mol m-2", "molar_areal_density"),
        ("umol photons m-2 s-1", "photon_flux_density"),
        # Photon count per area, time and wavelength interval (photons is a count label).
        ("photons cm-2 s-1 um-1", "spectral_photon_flux_density"),
        # Photolysis frequency per irradiance and per squared irradiance (Trebs et al. 2009).
        ("W-1 m2 s-1", "irradiance_response_coefficient"),
        ("W-2 m4 s-1", "irradiance_quadratic_response_coefficient"),
        ("kg m-3", "density"),
        ("m3 m-3", "volume_fraction"),
        ("g kg-1", "mass_fraction"),
        ("m2 m-2", "area_ratio"),
        ("rad", "angular"),
        ("mol mol-1", "concentration_ratio"),
        ("mol e- mol-1 photon", "quantum_yield"),
        ("g C g-1 C", "specific_cost"),
        ("ug g-1 h-1", "emission_factor"),
        ("g C g-1 C day-1", "specific_rate"),
        ("kg DM kg-1 body weight day-1", "intake_rate"),
        ("kPa^0.5", "pressure^0.5"),
        ("Pa**0.5", "pressure^0.5"),
        ("kg0.5 m-0.5 s-1", "pressure^0.5"),
    ],
)
def test_declared_semantic_aliases_match_by_si_vector(
    unit: str,
    declared_dimension: str,
) -> None:
    assert is_declared_dimension_compatible(unit, declared_dimension)


@pytest.mark.parametrize(
    ("unit", "declared_dimension"),
    [
        ("K", "configuration"),
        ("J mol-1", "energy"),
        ("ppm", "concentration"),
        ("mm hr-1", "mass_flux"),
        ("K day-0.5", "thermal_time"),
        ("dimensionless", "temperature"),
        ("Pa kPa-1", "unit_conversion"),
        ("kPa^0.5", "dimensionless"),
    ],
)
def test_ambiguous_or_incorrect_declared_dimensions_are_not_accepted(
    unit: str,
    declared_dimension: str,
) -> None:
    assert not is_declared_dimension_compatible(unit, declared_dimension)


class _FakeManager:
    def __init__(self) -> None:
        self._payload: Dict[str, Any] = {
            "coupling.external.variables.variable_definitions": {
                "test_mass_flux": {
                    "standard_name": "test_mass_flux",
                    "units": "kg m-2 s-1",
                    "semantics": "state",
                }
            },
            "coupling.scales.defaults.variables": {},
            "coupling.external.variables.aliases": {},
        }

    def get(self, key: str, **kwargs: Any) -> Any:
        if key not in self._payload:
            if kwargs.get("required", False):
                raise KeyError(key)
            return None
        return self._payload[key]


def test_ssot_entry_auto_dimensionality() -> None:
    catalog = SSOTCatalog.from_parameter_manager(manager=_FakeManager())
    entry = catalog.get("test_mass_flux")
    assert entry is not None
    assert entry.dimensionality == Dimensionality(M=1, L=-2, T=-1)


def test_unknown_unit_token_fails_closed() -> None:
    assert dimensionality_from_unit("banana") is None


@pytest.mark.parametrize("unit", ["kg/(m2 s)", "kg /"])
def test_unsupported_or_incomplete_division_syntax_fails_closed(unit: str) -> None:
    assert dimensionality_from_unit(unit) is None


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("mol P m-2", Dimensionality(N=1, L=-2)),
        ("mol P m-2 s-1", Dimensionality(N=1, L=-2, T=-1)),
        ("Ly day-1", Dimensionality(M=1, T=-3)),
    ],
)
def test_domain_unit_vocabulary_has_physical_dimensions(
    unit: str,
    expected: Dimensionality,
) -> None:
    assert _require_dim(unit) == expected


def test_reaction_affinity_rate_product_has_power_dimension() -> None:
    product = _require_dim("mol s-1 J mol-1")
    power = _require_dim("J s-1")

    assert product.is_compatible(power)


# ---------------------------------------------------------------------------
# Element labels: the labels the SI derivation skips (C, N, P after a mass or
# amount token) are reported, so a caller can compare them with a CF name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("kg C m-2 s-1", ("C",)),
        ("g N m-2", ("N",)),
        ("kg P s-1", ("P",)),
        ("mol C m-2 s-1", ("C",)),
        ("kg m-2 s-1", ()),
        ("umol CO2 m-2 s-1", ()),
        ("C", ()),
        ("N m-2", ()),
    ],
)
def test_element_annotations_are_the_labels_the_derivation_skips(unit: str, expected) -> None:
    from relic.data.standards.dimensionality import element_annotations_from_unit

    assert element_annotations_from_unit(unit) == expected


def test_element_annotations_of_an_underivable_unit_are_none() -> None:
    from relic.data.standards.dimensionality import element_annotations_from_unit

    assert element_annotations_from_unit("kg (C) m-2") is None
    assert element_annotations_from_unit(None) is None


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("kg C m-2 s-1", "kg m-2 s-1"),
        ("g N m-2", "g m-2"),
        ("mol C m-2 s-1", "mol m-2 s-1"),
        ("kg m-2 s-1", "kg m-2 s-1"),
        ("N m-2", "N m-2"),
    ],
)
def test_unit_without_element_annotations_drops_exactly_the_skipped_labels(unit: str, expected: str) -> None:
    from relic.data.standards.dimensionality import (
        dimensionality_from_unit,
        unit_without_element_annotations,
    )

    stripped = unit_without_element_annotations(unit)
    assert stripped == expected
    assert dimensionality_from_unit(stripped) == dimensionality_from_unit(unit)


@pytest.mark.parametrize(
    ("unit", "labels", "vector"),
    [
        ("Pg C yr-1", ("C",), {"M": 1.0, "T": -1.0}),
        ("Tg N yr-1", ("N",), {"M": 1.0, "T": -1.0}),
        ("Gg C", ("C",), {"M": 1.0}),
        ("t C ha-1", ("C",), {"M": 1.0, "L": -2.0}),
    ],
)
def test_large_mass_units_carry_element_labels(unit: str, labels, vector) -> None:
    # Global budgets use Pg C and Tg N; land carbon stocks use t C ha-1. The element
    # is a label after these mass units, never a coulomb or a newton.
    from relic.data.standards.dimensionality import (
        Dimensionality,
        dimensionality_from_unit,
        element_annotations_from_unit,
    )

    assert element_annotations_from_unit(unit) == labels
    assert dimensionality_from_unit(unit) == Dimensionality(**vector)


def test_unit_without_element_annotations_of_an_underivable_unit_is_none() -> None:
    from relic.data.standards.dimensionality import unit_without_element_annotations

    assert unit_without_element_annotations("kg (C) m-2") is None
    assert unit_without_element_annotations(None) is None
