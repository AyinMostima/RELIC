"""Structural protocol for RELIC-IPHS-DEC Functor declarations."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class IPHSDECFunctor(Protocol):
    """Optional structural declaration protocol for regular RELIC Functors.

    This protocol does not define a second lifecycle. Implementations remain
    ordinary FunctorBase subclasses and keep using forward(...) as the runtime
    calculation entrypoint.
    """

    name: str
    metadata: Any

    def declared_forms(self) -> tuple[object, ...]:
        """Declare input and output cochain spaces."""
        ...

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys needed by structural builders."""
        ...

    def build_hodge_terms(
        self,
        complex_bundle: Any,
        resolved_parameters: Mapping[str, Any],
    ) -> tuple[object, ...]:
        """Build material or metric Hodge structures."""
        ...

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return named contributions to the discrete energy functional."""
        ...

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return named contributions to the discrete entropy functional."""
        ...

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any,
    ) -> Any:
        """Return a skew contribution to J_h, or None."""
        ...

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any,
    ) -> Any:
        """Return a PSD contribution to M_h or an Onsager operator, or None."""
        ...

    def ports(self) -> tuple[object, ...]:
        """Declare boundary, management, or external IPHS ports."""
        ...

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        """Pure calculation helper; must not mutate global state."""
        ...


__all__ = ["IPHSDECFunctor"]
