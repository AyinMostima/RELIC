"""G01 unit consistency gate: certificate unit maps agree with the SCM contracts.

Every ``unit_map`` entry of a process certificate must be a parseable unit
(never prose) and must have the same dimensionality as the SCM contract of the
same variable; where the SSOT catalog also lists the variable its units must
agree too. Spelling may differ (``m^-2`` versus ``m-2``, a species qualifier
such as ``kg C``) but the dimension may not, and a placeholder contract unit
(``dimensionless`` for a physical quantity) is a defect of the contract, not of
the certificate.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml

from relic.contracts.contract_loader import ContractRegistry
from relic.data.standards.dimensionality import dimensionality_from_unit
from relic.orchestration.ssot_catalog import SSOTCatalog

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFICATES = sorted(glob.glob(str(REPO_ROOT / "docs" / "source" / "theory" / "iphs_certificates" / "*.yaml")))
PROSE_MARKERS = (" per ", "declared in", "computed as", "from the canonical", "state_unit", "boundary rate", "export", "profile", "increment")


# "non-quantity" marks an output that carries no physical quantity (a status token or
# a report dictionary); it is accepted only where no contract exists.
NON_QUANTITY = "non-quantity"


def _normalize(units: str | None) -> str:
    text = (units or "").strip()
    return "dimensionless" if text in {"", "1", "dimensionless"} else text


@pytest.fixture(scope="module")
def registry() -> ContractRegistry:
    reg = ContractRegistry()
    reg.load_contracts_dir(REPO_ROOT / "relic" / "contracts" / "variables")
    return reg


@pytest.fixture(scope="module")
def catalog() -> SSOTCatalog:
    return SSOTCatalog.from_parameter_manager()


def _unit_maps():
    for path in CERTIFICATES:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        yield Path(path).name, data.get("module_name"), dict(data.get("unit_map") or {})


def test_certificate_units_are_units_not_prose():
    offenders = []
    for name, module, unit_map in _unit_maps():
        for variable, units in unit_map.items():
            text = str(units)
            if text == NON_QUANTITY:
                continue
            if any(marker in text for marker in PROSE_MARKERS) or len(text) > 32:
                offenders.append((name, variable, text))
                continue
            try:
                parsed = dimensionality_from_unit(_normalize(text))
            except Exception as exc:  # noqa: BLE001 - the offender list is the assertion
                offenders.append((name, variable, f"{text}: {exc}"))
                continue
            if parsed is None:
                offenders.append((name, variable, f"{text}: not a parseable unit"))
    assert offenders == []


def test_certificate_units_match_contract_and_catalog_dimensions(registry, catalog):
    disagreements = []
    for name, module, unit_map in _unit_maps():
        for variable, units in unit_map.items():
            contract = registry.get_contract(variable)
            if contract is None:
                continue
            assert str(units) != NON_QUANTITY, (name, variable, "a contracted variable is a quantity")
            certificate_units = _normalize(str(units))
            contract_units = _normalize(contract.udunits)
            try:
                same = dimensionality_from_unit(certificate_units) == dimensionality_from_unit(contract_units)
            except Exception:  # noqa: BLE001 - reported below
                same = False
            if not same:
                disagreements.append((name, variable, certificate_units, contract_units, "contract"))
            entry = catalog.get(variable)
            if entry is not None:
                catalog_units = _normalize(entry.units)
                try:
                    same_catalog = dimensionality_from_unit(certificate_units) == dimensionality_from_unit(catalog_units)
                except Exception:  # noqa: BLE001 - reported below
                    same_catalog = False
                if not same_catalog:
                    disagreements.append((name, variable, certificate_units, catalog_units, "catalog"))
    assert disagreements == []


@pytest.mark.parametrize(
    "variable, expected",
    [
        ("snow_water_equivalent", "mm"),
        ("snow_depth", "m"),
        ("snow_surface_age", "day"),
        ("snowfall", "kg m-2 s-1"),
        ("rain_flux", "kg m-2 s-1"),
        ("snow_melt_flux", "kg m-2 s-1"),
        # Velocity-dimension hydrology stub (the hydraulic_conductivity
        # placeholder had no state reader or writer and was removed).
        ("canopy_conductance", "m s-1"),
        ("diagnostics.eddy_diffusivity", "m2 s-1"),
    ],
)
def test_formerly_stubbed_hydrology_and_transport_contracts_carry_physical_units(registry, variable, expected):
    contract = registry.get_contract(variable)
    assert contract is not None
    assert dimensionality_from_unit(_normalize(contract.udunits)) == dimensionality_from_unit(expected)
    assert _normalize(contract.udunits) != "dimensionless"
