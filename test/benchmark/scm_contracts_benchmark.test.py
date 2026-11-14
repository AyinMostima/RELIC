"""L3 benchmark tests for SCM contracts and domain ledger."""

import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.validation.conservation_guardian import ConservationGuardian


def test_scm_domain_ledger_from_parameters() -> None:
    """Ensure parameter ledger loads domain members for SCM."""
    registry = ContractRegistry()
    members = registry.get_domain_members("carbon.total")
    assert any(name in members for name in ("biosphere.leaf_carbon", "leaf_carbon"))


def test_scm_conservation_guardian_with_ledger() -> None:
    """Run a lightweight domain budget check using ledger members."""
    registry = ContractRegistry()
    guardian = ConservationGuardian(registry)

    result = guardian.check_domain(
        domain="carbon.total",
        flux_in={"biosphere.leaf_carbon_in": torch.tensor(0.1)},
        flux_out={"biosphere.leaf_carbon_out": torch.tensor(0.05)},
        storage_before={"biosphere.leaf_carbon": torch.tensor(1.0)},
        storage_after={"biosphere.leaf_carbon": torch.tensor(1.05)},
        dt=1.0,
    )

    assert result.passed
