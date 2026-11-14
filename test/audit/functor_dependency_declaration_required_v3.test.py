"""
Audit gate for explicit get_dependencies declarations on key functors.
"""

from __future__ import annotations

from relic.coupling.scales.adapters import ConservativeRemap
from relic.scientific.atmosphere.radiation import RadiationField
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import (
    SesquiterpeneStressModel,
)


def _assert_explicit_dependency_method(functor_cls) -> None:
    assert "get_dependencies" in functor_cls.__dict__, (
        f"{functor_cls.__name__} must explicitly declare get_dependencies()"
    )
    instance = functor_cls()
    deps = instance.get_dependencies()
    assert isinstance(deps, list)


def test_key_functors_explicitly_declare_dependencies() -> None:
    """
    References:
    - docs/source/architecture_overview.md dependency graph contract.

    Mathematical expression:
    - G = (V, E), E requires explicit dependency edges for deterministic order.

    Parameter meanings:
    - V: functor nodes.
    - E: dependency edges returned by get_dependencies().
    """
    _assert_explicit_dependency_method(RadiationField)
    _assert_explicit_dependency_method(SesquiterpeneStressModel)
    _assert_explicit_dependency_method(ConservativeRemap)

