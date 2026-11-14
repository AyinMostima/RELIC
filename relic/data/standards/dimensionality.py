"""SI dimensionality helpers for coupling/unit governance."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Dict, Optional


@dataclass(frozen=True)
class Dimensionality:
    """
    SI base dimension exponents (M, L, T, Theta, Q, N, J).

    References:
    - BIPM SI Brochure, 9th edition (2019), Table 2.

    Mathematical expression:
    - [X] = M^a * L^b * T^c * Theta^d * Q^e * N^f * J^g

    Parameter meanings:
    - M: mass exponent (kg)
    - L: length exponent (m)
    - T: time exponent (s)
    - Theta: thermodynamic temperature exponent (K)
    - Q: electric current exponent (A)
    - N: amount of substance exponent (mol)
    - J: luminous intensity exponent (cd)
    """

    M: float = 0.0
    L: float = 0.0
    T: float = 0.0
    Theta: float = 0.0
    Q: float = 0.0
    N: float = 0.0
    J: float = 0.0

    def is_dimensionless(self) -> bool:
        return (
            self.M == 0
            and self.L == 0
            and self.T == 0
            and self.Theta == 0
            and self.Q == 0
            and self.N == 0
            and self.J == 0
        )

    def is_compatible(self, other: "Dimensionality") -> bool:
        return (
            self.M == other.M
            and self.L == other.L
            and self.T == other.T
            and self.Theta == other.Theta
            and self.Q == other.Q
            and self.N == other.N
            and self.J == other.J
        )

    def has_time_inverse(self) -> bool:
        return self.T < 0


_ZERO = Dimensionality()

_CASE_SENSITIVE_UNIT_DIMENSIONS: Dict[str, Dimensionality] = {
    "C": Dimensionality(T=1, Q=1),
    "N": Dimensionality(M=1, L=1, T=-2),
    "H": Dimensionality(M=1, L=2, T=-2, Q=-2),
    "M": Dimensionality(N=1, L=-3),
    # Reference: NIST Technical Note 1187, solar-radiation unit definition.
    # Mathematical expression: 1 Ly = 1 cal cm-2, so [Ly] = M T-2.
    # Parameter meaning: Ly is radiant exposure (energy per unit area).
    "Ly": Dimensionality(M=1, T=-2),
    # Reference: BIPM SI Brochure, 9th edition (2019), Table 8 (tonne, 1 t = 1000 kg)
    # and Section 3 (SI prefixes giga, tera, peta on the gram).
    # Mathematical expression: [t] = [Gg] = [Tg] = [Pg] = M.
    # Parameter meaning: large mass units of land stocks (t C ha-1) and global
    # budgets (Tg N yr-1, Pg C yr-1); case-sensitive so that T (tesla) is not a tonne.
    "t": Dimensionality(M=1),
    "Gg": Dimensionality(M=1),
    "Tg": Dimensionality(M=1),
    "Pg": Dimensionality(M=1),
}

_BASE_UNIT_DIMENSIONS: Dict[str, Dimensionality] = {
    "1": _ZERO,
    "%": _ZERO,
    "count": _ZERO,
    "fraction": _ZERO,
    "dimensionless": _ZERO,
    "unitless": _ZERO,
    "ppm": _ZERO,
    "ppb": _ZERO,
    "ppbv": _ZERO,
    "ppt": _ZERO,
    "rad": _ZERO,
    "deg": _ZERO,
    "degree": _ZERO,
    "degrees": _ZERO,
    "kg": Dimensionality(M=1),
    "g": Dimensionality(M=1),
    "mg": Dimensionality(M=1),
    "ug": Dimensionality(M=1),
    "microgram": Dimensionality(M=1),
    "m": Dimensionality(L=1),
    "cm": Dimensionality(L=1),
    "mm": Dimensionality(L=1),
    "km": Dimensionality(L=1),
    "um": Dimensionality(L=1),
    "ha": Dimensionality(L=2),
    "l": Dimensionality(L=3),
    "s": Dimensionality(T=1),
    "sec": Dimensionality(T=1),
    "min": Dimensionality(T=1),
    "h": Dimensionality(T=1),
    "hr": Dimensionality(T=1),
    "hour": Dimensionality(T=1),
    "hours": Dimensionality(T=1),
    "d": Dimensionality(T=1),
    "day": Dimensionality(T=1),
    "days": Dimensionality(T=1),
    "yr": Dimensionality(T=1),
    "year": Dimensionality(T=1),
    "k": Dimensionality(Theta=1),
    "c": Dimensionality(Theta=1),
    "degc": Dimensionality(Theta=1),
    "celsius": Dimensionality(Theta=1),
    "mol": Dimensionality(N=1),
    "mmol": Dimensionality(N=1),
    "umol": Dimensionality(N=1),
    "mumol": Dimensionality(N=1),
    "nmol": Dimensionality(N=1),
    "molecule": Dimensionality(N=1),
    "molecules": Dimensionality(N=1),
    "pa": Dimensionality(M=1, L=-1, T=-2),
    "kpa": Dimensionality(M=1, L=-1, T=-2),
    "hpa": Dimensionality(M=1, L=-1, T=-2),
    "mpa": Dimensionality(M=1, L=-1, T=-2),
    "bar": Dimensionality(M=1, L=-1, T=-2),
    "atm": Dimensionality(M=1, L=-1, T=-2),
    "j": Dimensionality(M=1, L=2, T=-2),
    "kj": Dimensionality(M=1, L=2, T=-2),
    "mj": Dimensionality(M=1, L=2, T=-2),
    "cal": Dimensionality(M=1, L=2, T=-2),
    "w": Dimensionality(M=1, L=2, T=-3),
    "hz": Dimensionality(T=-1),
    "a": Dimensionality(Q=1),
    "v": Dimensionality(M=1, L=2, T=-3, Q=-1),
    "mv": Dimensionality(M=1, L=2, T=-3, Q=-1),
    "f": Dimensionality(M=-1, L=-2, T=4, Q=2),
    "hry": Dimensionality(M=1, L=2, T=-2, Q=-2),
    "siemens": Dimensionality(M=-1, L=-2, T=3, Q=2),
    "ds": Dimensionality(M=-1, L=-2, T=3, Q=2),
    "molar": Dimensionality(N=1, L=-3),
}

_ANNOTATION_TOKENS = {
    "co2",
    "ch4",
    "n2o",
    "h2o",
    "o2",
    "nh3",
    "no3",
    "nh4",
    "photons",
    "photon",
    "element",
    "dm",
    "dw",
    "dry",
    "matter",
    "biomass",
    "body",
    "weight",
    "soil",
    "animals",
    "animal",
    "glucose",
    "product",
    "north",
    "east",
    "e-",
    "o3",
}

_ANNOTATION_BASIS_TOKENS = {
    "pg",
    "tg",
    "gg",
    "t",
    "kg",
    "g",
    "mg",
    "ug",
    "mol",
    "mmol",
    "umol",
    "mumol",
    "nmol",
}

_CATEGORY_BY_VECTOR = {
    _ZERO: "dimensionless",
    Dimensionality(Theta=1): "temperature",
    Dimensionality(M=1, L=-1, T=-2): "pressure",
    Dimensionality(L=1): "length",
    Dimensionality(L=2): "area",
    Dimensionality(L=3): "volume",
    Dimensionality(M=1): "mass",
    Dimensionality(T=1): "time",
    Dimensionality(N=1): "amount",
    Dimensionality(M=1, L=-3): "mass_concentration",
    Dimensionality(N=1, L=-3): "amount_concentration",
    Dimensionality(L=1, T=-1): "velocity",
    Dimensionality(L=2, T=-1): "diffusivity",
    Dimensionality(T=-1): "frequency",
    Dimensionality(L=-1, T=1): "resistance",
    Dimensionality(M=1, T=-3): "energy_flux",
    Dimensionality(M=1, L=2, T=-2): "energy",
    Dimensionality(L=2, T=-2): "specific_energy",
    Dimensionality(M=1, L=-2): "mass_areal_density",
    Dimensionality(N=1, L=-2): "amount_areal_density",
    Dimensionality(M=1, L=-2, T=-1): "mass_flux",
    Dimensionality(N=1, L=-2, T=-1): "amount_flux",
    Dimensionality(M=-1, L=2): "specific_area",
    Dimensionality(Theta=1, T=1): "thermal_time",
    Dimensionality(M=1, N=-1): "molar_mass",
    Dimensionality(T=1, Q=1): "electric_charge",
    Dimensionality(T=1, Q=1, N=-1): "charge_per_mole",
    Dimensionality(M=1, L=2, T=-2, N=-1): "energy_per_amount",
    Dimensionality(M=1, L=2, T=-2, Theta=-1, N=-1): "entropy_per_amount",
    Dimensionality(Theta=-1): "inverse_temperature",
    Dimensionality(M=1, T=-1): "mass_per_time",
    Dimensionality(L=2, T=-2, Theta=-1): "specific_heat_capacity",
    Dimensionality(L=-1): "inverse_length",
    Dimensionality(M=1, L=2, T=-3, Q=-1): "electric_potential",
    Dimensionality(L=-2): "stocking_density",
    Dimensionality(M=1, L=-3, T=-1): "mass_volumetric_rate",
    Dimensionality(N=1, L=-3, T=-1): "amount_volumetric_rate",
    Dimensionality(M=1, L=2, T=-1): "action",
    Dimensionality(M=-1, L=3): "distribution_coefficient",
    Dimensionality(M=-1, L=1, T=2): "inverse_pressure",
    Dimensionality(L=3, T=-1, N=-1): "second_order_rate_constant",
    Dimensionality(Theta=1, T=-0.5): "temperature_volatility",
    Dimensionality(M=1, L=1, T=-2, Q=-2): "permeability",
    Dimensionality(M=-1, L=-3, T=4, Q=2): "permittivity",
    Dimensionality(M=1, L=2, T=-2, Theta=-1): "heat_capacity",
    Dimensionality(Theta=1, L=-1): "temperature_gradient",
    Dimensionality(Theta=2): "temperature_squared",
    Dimensionality(Theta=3): "temperature_cubed",
    Dimensionality(Theta=-2): "inverse_temperature_squared",
    Dimensionality(Theta=-3): "inverse_temperature_cubed",
    Dimensionality(Theta=-4): "inverse_temperature_quartic",
    Dimensionality(M=1, T=-3, Theta=-4): "radiative_constant",
    Dimensionality(L=1, Theta=1): "length_temperature",
    Dimensionality(L=1, T=-1, Theta=-1): "transfer_coefficient",
    Dimensionality(L=1, T=-2): "acceleration",
    Dimensionality(L=2, T=1, N=-1): "light_response_parameter",
    Dimensionality(M=-1, L=3, T=-2): "gravitational_constant",
    Dimensionality(N=-1): "inverse_amount",
    Dimensionality(M=-1, L=-2, T=2, N=1): "amount_energy_inverse",
    Dimensionality(L=1, T=-0.5): "boundary_layer_transfer_coefficient",
    Dimensionality(M=-1, L=-3, T=3, Q=2): "electrical_conductivity",
    Dimensionality(M=1, L=-1, T=-2, Theta=-1): "pressure_temperature_coefficient",
}

_DECLARED_DIMENSION_VECTORS = {
    category: vector for vector, category in _CATEGORY_BY_VECTOR.items()
}
_DECLARED_DIMENSION_VECTORS.update(
    {
        "mass_per_area": Dimensionality(M=1, L=-2),
        "molar_flux": Dimensionality(N=1, L=-2, T=-1),
        "molar_areal_density": Dimensionality(N=1, L=-2),
        "photon_flux_density": Dimensionality(N=1, L=-2, T=-1),
        # Photon count per area, time and wavelength interval (photons cm-2 s-1 um-1;
        # the photon label is a count annotation, so [F_lambda] = L-3 T-1).
        "spectral_photon_flux_density": Dimensionality(L=-3, T=-1),
        # Photolysis frequency per irradiance (s-1 per W m-2 = W-1 m2 s-1) and per
        # squared irradiance (W-2 m4 s-1), e.g. the J(NO2)-irradiance polynomial.
        "irradiance_response_coefficient": Dimensionality(M=-1, T=2),
        "irradiance_quadratic_response_coefficient": Dimensionality(M=-2, T=5),
        # Time per areal amount (s cm2 molecule-1 = s m2 mol-1 up to N_A and 1e4), e.g. the
        # NO2-column slope of the HCHO column-emission sensitivity (Marais et al., 2012, Fig. 9).
        "time_per_areal_amount": Dimensionality(L=2, T=1, N=-1),
        "density": Dimensionality(M=1, L=-3),
        "volume_fraction": _ZERO,
        "mass_fraction": _ZERO,
        "area_ratio": _ZERO,
        "angular": _ZERO,
        "concentration_ratio": _ZERO,
        "quantum_yield": _ZERO,
        "specific_cost": _ZERO,
        "emission_factor": Dimensionality(T=-1),
        "specific_rate": Dimensionality(T=-1),
        "mass_flux_per_biomass": Dimensionality(T=-1),
        "intake_rate": Dimensionality(T=-1),
        "exposure": Dimensionality(T=1),
        "transport_coefficient": Dimensionality(L=1, T=-1),
        "degree_day_coefficient": Dimensionality(L=1, T=-1, Theta=-1),
        "amount_per_energy": Dimensionality(M=-1, L=-2, T=2, N=1),
        "henry_constant": Dimensionality(M=-1, L=-2, T=2, N=1),
        # References: BIPM SI Brochure; the Medlyn slope multiplies D^(-1/2).
        # Mathematical expression: [Pa^(1/2)] = M^(1/2) L^(-1/2) T^(-1).
        # Parameter meanings: M mass, L length, T time; the half power is exact.
        # Units: pressure^0.5, including Pa^0.5 and kPa^0.5.
        # Domain/constraints: semantic dimension only; prefix scale belongs to UnitConverter.
        # Sign convention: dimension exponents, not a signed physical flux.
        "pressure^0.5": Dimensionality(M=0.5, L=-0.5, T=-1),
    }
)

_TOKEN_PATTERN = re.compile(
    r"^(?P<symbol>[A-Za-z%]+?)(?P<power>[+-]?\d+(?:\.\d+)?)?$"
)


def _add_scaled(
    total: Dimensionality,
    base: Dimensionality,
    power: float,
) -> Dimensionality:
    """Add one powered base dimension to an accumulated SI vector."""
    values = {
        field: getattr(total, field) + power * getattr(base, field)
        for field in ("M", "L", "T", "Theta", "Q", "N", "J")
    }
    return Dimensionality(
        **{
            key: 0.0 if abs(value) < 1.0e-12 else value
            for key, value in values.items()
        }
    )


def _normalize_unit_expression(unit: str) -> str:
    """Normalize supported UDUNITS spellings without changing dimensions."""
    normalized = unit.strip().replace("\u00b5", "u").replace("\u03bc", "u")
    normalized = normalized.replace("**", "").replace("^", "")
    thermal_aliases = {
        "deg-days": "K day",
        "degree-days": "K day",
        "degree days": "K day",
        "degC_day": "K day",
        "degC-day": "K day",
    }
    normalized = thermal_aliases.get(normalized, normalized)
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("*", " ").replace("/", " / ")
    normalized = normalized.replace("(", " ").replace(")", " ")
    return " ".join(normalized.split())


def _is_element_annotation(tokens: list[str], index: int, symbol: str) -> bool:
    """Distinguish C/N/P element labels from charge or force unit symbols."""
    if symbol not in {"C", "N", "P", "S", "H", "K"} or index == 0:
        return False
    previous = tokens[index - 1]
    match = _TOKEN_PATTERN.fullmatch(previous)
    return bool(
        match and match.group("symbol").lower() in _ANNOTATION_BASIS_TOKENS
    )


def _parse_dimensionality(unit: str) -> Optional[Dimensionality]:
    expression = _normalize_unit_expression(unit)
    if not expression:
        return None
    tokens = expression.split()
    total = _ZERO
    divide_next = False

    for index, token in enumerate(tokens):
        if token == "/":
            if index == 0 or index == len(tokens) - 1 or divide_next:
                return None
            divide_next = True
            continue
        if token.lower() in _ANNOTATION_TOKENS:
            divide_next = False
            continue
        if _is_element_annotation(tokens, index, token):
            divide_next = False
            continue
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", token, re.I):
            divide_next = False
            continue

        match = _TOKEN_PATTERN.fullmatch(token)
        if match is None:
            return None
        symbol = match.group("symbol")
        power_text = match.group("power")
        power = float(power_text) if power_text is not None else 1.0
        if divide_next:
            power *= -1.0
        divide_next = False

        base = _CASE_SENSITIVE_UNIT_DIMENSIONS.get(symbol)
        if base is None:
            base = _BASE_UNIT_DIMENSIONS.get(symbol.lower())
        if base is None:
            return None
        total = _add_scaled(total, base, power)

    return total


@lru_cache(maxsize=1024)
def dimensionality_from_unit(unit_str: Optional[str]) -> Optional[Dimensionality]:
    """Convert unit string to SI dimensionality exponents."""
    if unit_str is None:
        return None
    unit_clean = str(unit_str).strip()
    if not unit_clean:
        return None
    if "(" in unit_clean or ")" in unit_clean:
        return None

    return _parse_dimensionality(unit_clean)


def _element_annotation_split(unit_str: Optional[str]) -> Optional[tuple[tuple[str, ...], tuple[bool, ...]]]:
    """Normalized tokens of a unit and, per token, whether it is an element label.

    The labels are exactly the tokens that dimensionality_from_unit skips as
    element annotations (C, N, P, S, H, K directly after a mass or amount
    token), found with the same normalization and annotation rule, so no
    second unit parser exists. None when the unit has no SI derivation.
    """
    if dimensionality_from_unit(unit_str) is None:
        return None
    tokens = tuple(_normalize_unit_expression(str(unit_str).strip()).split())
    flags = tuple(
        token.lower() not in _ANNOTATION_TOKENS and _is_element_annotation(list(tokens), index, token)
        for index, token in enumerate(tokens)
    )
    return tokens, flags


@lru_cache(maxsize=1024)
def element_annotations_from_unit(unit_str: Optional[str]) -> Optional[tuple[str, ...]]:
    """Element labels of a unit, in order (e.g. ('C',) for 'kg C m-2 s-1').

    None when the unit has no SI derivation (see _element_annotation_split).
    """
    split = _element_annotation_split(unit_str)
    if split is None:
        return None
    tokens, flags = split
    return tuple(token for token, is_label in zip(tokens, flags) if is_label)


@lru_cache(maxsize=1024)
def unit_without_element_annotations(unit_str: Optional[str]) -> Optional[str]:
    """The unit with its element labels removed (e.g. 'kg m-2 s-1' for 'kg C m-2 s-1').

    The removed tokens are exactly those element_annotations_from_unit reports,
    so the SI vector is unchanged; the other tokens keep their normalized
    spelling. None when the unit has no SI derivation.
    """
    split = _element_annotation_split(unit_str)
    if split is None:
        return None
    tokens, flags = split
    return " ".join(token for token, is_label in zip(tokens, flags) if not is_label)


@lru_cache(maxsize=1024)
def dimension_category_from_unit(unit_str: Optional[str]) -> Optional[str]:
    """Return the canonical physical category for one unit expression."""
    dimensionality = dimensionality_from_unit(unit_str)
    if dimensionality is None:
        return None
    return _CATEGORY_BY_VECTOR.get(dimensionality, "derived")


@lru_cache(maxsize=2048)
def is_declared_dimension_compatible(
    unit_str: Optional[str],
    declared_dimension: Optional[str],
) -> bool:
    """Compare a declared semantic dimension with the unit SI vector."""
    actual = dimensionality_from_unit(unit_str)
    declared = str(declared_dimension or "").strip().lower()
    expected = _DECLARED_DIMENSION_VECTORS.get(declared)
    return actual is not None and expected is not None and actual.is_compatible(expected)


__all__ = [
    "Dimensionality",
    "dimension_category_from_unit",
    "dimensionality_from_unit",
    "element_annotations_from_unit",
    "is_declared_dimension_compatible",
    "unit_without_element_annotations",
]
