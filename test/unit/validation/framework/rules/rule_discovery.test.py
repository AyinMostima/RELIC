from __future__ import annotations

import pytest

from relic.validation.framework import registry as registry_module
from relic.validation.framework.registry import RuleMetadata, ValidationRegistry
from relic.validation.framework.rules import (
    get_registered_rule_count,
    list_available_rules_by_category,
)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> ValidationRegistry:
    isolated = ValidationRegistry()
    monkeypatch.setattr(registry_module, "_global_registry", isolated)
    return isolated


def _register(
    registry: ValidationRegistry,
    rule_id: str,
    *,
    domains: list[str] | None = None,
    tags: list[str] | None = None,
) -> None:
    def rule(_context):
        return []

    registry.register_rule(
        rule,
        RuleMetadata(
            rule_id=rule_id,
            name=f"Rule {rule_id}",
            description=f"Discovery test for {rule_id}",
            layer="L2",
            domains=domains or [],
            tags=tags or [],
        ),
    )


def test_registered_rule_count_uses_canonical_registry(
    registry: ValidationRegistry,
) -> None:
    _register(registry, "first")
    _register(registry, "second")

    assert get_registered_rule_count() == 2


def test_rule_discovery_categorizes_from_declared_tags_and_domains(
    registry: ValidationRegistry,
) -> None:
    _register(registry, "cross", tags=["cross_domain"])
    _register(registry, "stochastic", domains=["stochastic"])
    _register(registry, "parameters", tags=["parameters"])
    _register(registry, "conservation", tags=["conservation", "coupling"])
    _register(registry, "coupling", domains=["coupling"])
    _register(registry, "other")

    categories = list_available_rules_by_category()

    assert set(categories) == {
        "cross_domain",
        "stochastic_scores",
        "parameters",
        "conservation",
        "coupling_contracts",
        "other",
    }
    assert categories["cross_domain"][0]["rule_id"] == "cross"
    assert categories["stochastic_scores"][0]["rule_id"] == "stochastic"
    assert categories["parameters"][0]["rule_id"] == "parameters"
    assert categories["conservation"][0]["rule_id"] == "conservation"
    assert categories["coupling_contracts"][0]["rule_id"] == "coupling"
    assert categories["other"][0] == {
        "rule_id": "other",
        "name": "Rule other",
        "layer": "L2",
        "domains": [],
    }
