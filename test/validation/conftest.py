"""Shared fixtures for the validation framework tests.

These tests reset or extend the process-wide rule registry, and a
ValidationEngine built on a reset registry reloads the rule modules to register
the core rules again, which rebinds every function and class of those modules.
The autouse fixture below puts that state back after every test: the registry
object with its rules, the global Functor-metadata integrator, and the
namespaces of the rule modules. A later test in the same session therefore sees
the registry and the rule objects it imported.
"""
import sys

import pytest

import relic.validation.framework.functor_integration as functor_integration
import relic.validation.framework.registry as registry_module

RULE_PACKAGE = "relic.validation.framework.rules"


@pytest.fixture(autouse=True)
def _restore_global_validation_registry():
    """Restore the process-wide validation registry state after each test."""
    registry = registry_module._global_registry
    integrator = functor_integration._global_integrator
    contents = None
    if registry is not None:
        contents = (
            dict(registry._rules),
            dict(registry._metadata),
            {group: set(rule_ids) for group, rule_ids in registry._rule_groups.items()},
            registry._dependency_cache,
        )
    namespaces = [
        (module, dict(module.__dict__))
        for name, module in list(sys.modules.items())
        if name == RULE_PACKAGE or name.startswith(RULE_PACKAGE + ".")
    ]
    yield
    for module, namespace in namespaces:
        module.__dict__.clear()
        module.__dict__.update(namespace)
    if contents is not None:
        rules, metadata, groups, dependency_cache = contents
        registry._rules.clear()
        registry._rules.update(rules)
        registry._metadata.clear()
        registry._metadata.update(metadata)
        registry._rule_groups.clear()
        registry._rule_groups.update(groups)
        registry._dependency_cache = dependency_cache
    registry_module._global_registry = registry
    functor_integration._global_integrator = integrator
