"""
OperatorFactory parameter enrichment contract.

The factory reads an explicit parameter namespace through the ParameterManager
and lets every lookup error propagate. Without an explicit namespace it leaves
the configuration unchanged: each operator resolves its parameters from its own
namespace (OperatorBase.get_parameter).
"""

from __future__ import annotations

import pytest

import relic.computation.operators.factory as factory_module
from relic.computation.operators import OperatorFactory
from relic.parameters import get_global_manager


class TestOperatorFactoryConfigValidation:
    """create_from_config never continues with an unvalidated raw dict."""

    def test_invalid_mapping_raises_the_validation_error(self):
        with pytest.raises(ValueError, match="name"):
            OperatorFactory.create_from_config(
                {
                    "operator_name": "gradient.central_difference_1d",
                    "name": 123,
                    "config": {"dx": 1.0},
                }
            )

    def test_non_mapping_config_is_a_type_error(self):
        with pytest.raises(TypeError, match="OperatorConfig or a mapping"):
            OperatorFactory.create_from_config(42)

    def test_valid_mapping_creates_the_operator(self):
        operator = OperatorFactory.create_from_config(
            {
                "operator_name": "gradient.central_difference_1d",
                "name": "validated_gradient",
                "config": {"dx": 1.0},
            }
        )
        assert operator.name == "validated_gradient"


class TestOperatorFactoryEnrichment:
    """Explicit namespaces are read strictly; nothing is swallowed."""

    def test_without_namespace_the_config_is_returned_unchanged(self, monkeypatch):
        def unavailable_manager(*args, **kwargs):
            raise AssertionError("the factory must not consult the manager without a namespace")

        monkeypatch.setattr(factory_module, "get_global_manager", unavailable_manager)
        config = {"dx": 2.0, "order": 2}
        enriched = OperatorFactory._validate_and_enrich_config(
            "gradient.central_difference_1d", config, None
        )
        assert enriched == config
        assert enriched is not config

    def test_explicit_namespace_values_are_merged_under_the_config(self):
        manager = get_global_manager()
        namespace = "computation.operators.gradient"
        expected = manager.get_namespace_parameters(
            namespace=namespace, include_subnamespaces=True, return_specs=False
        )
        assert expected
        enriched = OperatorFactory._validate_and_enrich_config(
            "gradient.central_difference_1d", {"order": 4}, namespace
        )
        assert enriched["order"] == 4
        for full_key, value in expected.items():
            short_key = full_key[len(namespace) + 1:]
            if "." in short_key or short_key == "order":
                continue
            assert enriched[short_key] == value

    def test_explicit_namespace_propagates_manager_failure(self, monkeypatch):
        def unavailable_manager(*args, **kwargs):
            raise RuntimeError("parameter manager unavailable")

        monkeypatch.setattr(factory_module, "get_global_manager", unavailable_manager)
        with pytest.raises(RuntimeError, match="parameter manager unavailable"):
            OperatorFactory._validate_and_enrich_config(
                "gradient.central_difference_1d", {}, "computation.operators.gradient"
            )

    def test_explicit_namespace_propagates_lookup_errors(self, monkeypatch):
        real_manager = get_global_manager()

        class _FailingNamespaceManager:
            def get_namespace_dict(self, namespace):
                return real_manager.get_namespace_dict(namespace)

            def get_namespace_parameters(self, **kwargs):
                raise ValueError("namespace parameter failed validation")

        monkeypatch.setattr(
            factory_module, "get_global_manager", lambda *args, **kwargs: _FailingNamespaceManager()
        )
        with pytest.raises(ValueError, match="namespace parameter failed validation"):
            OperatorFactory._validate_and_enrich_config(
                "gradient.central_difference_1d", {}, "computation.operators.gradient"
            )


# Operator ids only in operator-id contexts (not parameter keys such as "gradient.dx").
_OPERATOR_ID_PATTERN = (
    r"(?:operator_name\"?\s*[=:]\s*|register_operator\(|OperatorRegistry\.get\(|"
    r"OperatorFactory\.create\(|get_operator_info\(|operator name \(e\.g\., )"
    r"\"([a-z_]+\.[a-z0-9_]+)\""
)


def _unregistered_operator_ids(text):
    import re

    from relic.computation.operators.registry import OperatorRegistry

    return sorted(
        {name for name in re.findall(_OPERATOR_ID_PATTERN, text) if not OperatorRegistry.is_registered(name)}
    )


def test_code_examples_only_name_registered_operator_ids():
    """Docstring examples in the operator/solver/config packages use registered operator ids."""
    from pathlib import Path

    import relic.computation as computation_package

    root = Path(computation_package.__file__).parent
    offenders = {}
    for folder in ("operators", "solvers", "config"):
        for path in sorted((root / folder).glob("*.py")):
            unregistered = _unregistered_operator_ids(path.read_text(encoding="utf-8"))
            if unregistered:
                offenders[path.name] = unregistered
    assert offenders == {}


def test_get_operator_rejects_unknown_keywords():
    """A misspelled keyword (here the config) is an error, not silently dropped."""
    with pytest.raises(TypeError, match="confg"):
        OperatorFactory().get_operator(
            "transport.diffusion", confg={"spatial_dims": 1, "boundary_method": "zero"}
        )
