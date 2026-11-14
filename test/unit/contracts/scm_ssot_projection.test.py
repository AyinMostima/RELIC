"""One prognostic variable, one projection (B03).

The SCM contract (ContractRegistry), the orchestration port metadata
(NodeRegistry via SSOTCatalog), and the StateManager default schema must all
project the same units and kind for a variable; none of them is a second
schema registry. Units are compared after canonical normalization ("1" and
"dimensionless" are the same dimensionless unit).
"""

from __future__ import annotations

import pytest

from relic.contracts.contract_loader import ContractRegistry, load_contract_registry
from relic.core.state.manager import StateManager
from relic.data.standards.dimensionality import dimensionality_from_unit
from relic.orchestration.ssot_catalog import SSOTCatalog


def _normalize(units: str | None) -> str:
    text = (units or "").strip()
    return "dimensionless" if text in {"", "1", "dimensionless"} else text


@pytest.fixture(scope="module")
def registry() -> ContractRegistry:
    """The packaged contracts through the single contract reader."""
    return load_contract_registry()


@pytest.fixture(scope="module")
def catalog() -> SSOTCatalog:
    return SSOTCatalog.from_parameter_manager()


def test_every_contract_variable_shares_units_with_the_ssot_catalog(registry, catalog):
    names = list(registry.list_variables())
    assert names, "contract registry loaded no variables"
    disagreements = []
    for name in names:
        entry = catalog.get(name)
        if entry is None:
            continue
        contract_units = _normalize(registry.to_state_variable_spec(name).units)
        catalog_units = _normalize(entry.units)
        if contract_units != catalog_units or dimensionality_from_unit(contract_units) != dimensionality_from_unit(catalog_units):
            disagreements.append((name, contract_units, catalog_units))
    assert disagreements == []


def test_state_manager_default_schema_is_the_contract_projection(registry):
    manager = StateManager()
    schema = manager.schema
    assert set(schema) == set(registry.list_variables())
    for name, spec in schema.items():
        projected = registry.to_state_variable_spec(name)
        assert _normalize(spec.units) == _normalize(projected.units), name
        assert spec.variable_type == projected.variable_type, name
        assert spec.owner == projected.owner, name


@pytest.mark.parametrize(
    "name, units, kind",
    [
        # Externally supplied drivers that no relic Functor writes: forcing.
        ("temperature", "K", "forcing"),
        ("wind_speed", "m s-1", "forcing"),
        ("sensible_heat_flux", "W m-2", "flux"),
        ("par", "umol photons m-2 s-1", "forcing"),
        # Read from the state by the cross-domain carbon balance check; the
        # SSOT scale catalog declares it a molar flux.
        ("gpp", "umol m-2 s-1", "flux"),
        ("canopy_conductance", "m s-1", "state"),
        # Carbon mass flux that replaced the dimensionless total_growth_rate
        # placeholder (the bare carbon_flux placeholder had no reader or
        # writer and was removed).
        ("diagnostics.biomass_allocation.total_growth_rate", "kg C m-2 s-1", "diagnostic"),
        ("water_stress_factor", "dimensionless", "diagnostic"),
    ],
)
def test_previously_stubbed_contracts_project_physical_units(registry, catalog, name, units, kind):
    contract = registry.get_contract(name)
    assert contract is not None and contract.kind == kind
    assert _normalize(contract.udunits) == units
    assert _normalize(catalog.get(name).units) == units
    assert _normalize(StateManager().schema[name].units) == units
