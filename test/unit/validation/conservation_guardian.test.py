"""L1 tests for ConservationGuardian."""

import torch

from relic.validation.conservation_guardian import ConservationGuardian


class DummyRegistry:
    def __init__(self, domains):
        self._domains = domains

    def get_domain_members(self, domain):
        return list(self._domains.get(domain, []))


def test_conservation_guardian_pass() -> None:
    registry = DummyRegistry({"carbon.total": ["pool_a", "pool_b"]})
    guardian = ConservationGuardian(registry)

    result = guardian.check_domain(
        domain="carbon.total",
        flux_in={"pool_a_in": torch.tensor(0.2)},
        flux_out={"pool_b_out": torch.tensor(0.1)},
        storage_before={"pool_a": torch.tensor(1.0), "pool_b": torch.tensor(2.0)},
        storage_after={"pool_a": torch.tensor(1.05), "pool_b": torch.tensor(2.05)},
        dt=1.0,
    )

    assert result.passed


def test_conservation_guardian_fail() -> None:
    registry = DummyRegistry({"carbon.total": ["pool_a"]})
    guardian = ConservationGuardian(registry)

    result = guardian.check_domain(
        domain="carbon.total",
        flux_in={"pool_a_in": torch.tensor(0.1)},
        flux_out={"pool_a_out": torch.tensor(0.05)},
        storage_before={"pool_a": torch.tensor(1.0)},
        storage_after={"pool_a": torch.tensor(1.5)},
        dt=1.0,
    )

    assert not result.passed
