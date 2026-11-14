"""Unit tests for conservation registry utilities."""

import relic.core.conservation_registry as conservation_registry
from relic.core.conservation_registry import (
    build_conservation_ledger,
    load_parameter_ledger,
    merge_ledgers,
)


def test_parameter_ledger_contains_standard_entries():
    ledger = load_parameter_ledger(force_reload=True)
    assert "carbon" in ledger
    assert "biosphere.leaf_carbon" in ledger["carbon"]


def test_canonical_ledger_merges_contract_domains_and_parameter_members(monkeypatch):
    """The canonical ledger must be rebuilt only from the two explicit SSOTs."""

    class _ContractRegistry:
        _domains = {
            "carbon.total": ["contract_carbon"],
            "nitrogen.total": ["contract_nitrogen"],
        }

        def list_domains(self):
            return list(self._domains)

        def get_domain_members(self, domain):
            return list(self._domains[domain])

    monkeypatch.setattr(
        "relic.core.conservation_registry.load_parameter_ledger",
        lambda force_reload=False: {
            "carbon": ["parameter_carbon", "contract_carbon"],
            "water": ["parameter_water"],
        },
    )

    ledger = build_conservation_ledger(_ContractRegistry(), force_reload=True)

    assert ledger == {
        "carbon": ["contract_carbon", "parameter_carbon"],
        "nitrogen": ["contract_nitrogen"],
        "water": ["parameter_water"],
    }


def test_metadata_claims_are_not_an_element_membership_api():
    assert not hasattr(conservation_registry, "derive_metadata_ledger")


def test_merge_ledgers_deduplicates_explicit_members():
    combined = merge_ledgers(
        {"carbon": ["existing_pool"]},
        {"carbon": ["existing_pool", "second_pool"]},
    )
    assert combined == {"carbon": ["existing_pool", "second_pool"]}

