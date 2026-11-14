"""Executable chain-map contracts between declared finite cochain spaces (G12-MAP).

A ``LinearChainMap`` is a matrix map between two declared ``CochainSpaceSpec``
objects. The contract offers identity, composition, ledger preservation
(``C_target M = C_source``), and an operator-norm error bound. Equivalence of
maps is exact; an ``epsilon``-closeness relation is provided as a distance
bound and is deliberately not an equivalence (it is not transitive), which
the negative test demonstrates. Units are carried on the spaces and must be
declared for the map; a map between spaces whose units or degrees do not
match its declaration is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .forms import CochainSpaceSpec, TypedCochain


@dataclass(frozen=True)
class LinearChainMap:
    """M: source -> target on declared cochain spaces.

    References: Hatcher (2002) Algebraic Topology ch. 2 (chain maps);
    Desbrun, Hirani, Leok and Marsden (2005) Discrete Exterior Calculus;
    RELIC manuscript appendix on G_phys ledgers.
    Mathematical expression: y = M x with M of shape (n_target, n_source);
    ledger preservation C_t M = C_s; composition (M2 o M1) = M2 M1.
    Parameter meanings: M linear map, C_s and C_t ledger row matrices of
    the source and target spaces, x source cochain values.
    Units: unit_factor (required) converts source coefficient units to
    target units (a pure number); the spaces declare the units themselves.
    Domain/constraints: finite-dimensional declared spaces only; degrees
    must match the declaration; the map is exact (no tolerance in
    equality); closeness is a distance, not equivalence.
    Sign convention: as the underlying cochains (orientation convention of
    the spaces).
    """

    name: str
    source: CochainSpaceSpec
    target: CochainSpaceSpec
    matrix: Tensor
    unit_factor: float

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("chain map matrix must be two-dimensional")
        if tuple(self.matrix.shape) != (self.target.n_dofs, self.source.n_dofs):
            raise ValueError(
                f"chain map '{self.name}' shape {tuple(self.matrix.shape)} does not match "
                f"(target n_dofs {self.target.n_dofs}, source n_dofs {self.source.n_dofs})"
            )
        if not torch.isfinite(self.matrix).all():
            raise ValueError("chain map matrix must be finite")
        if not (self.unit_factor > 0.0):
            raise ValueError("unit_factor must be positive")

    @classmethod
    def identity(cls, space: CochainSpaceSpec, *, name: str = "identity") -> "LinearChainMap":
        # Same source and target space: the unit factor is exactly one.
        return cls(
            name=name, source=space, target=space,
            matrix=torch.eye(space.n_dofs, dtype=torch.float64), unit_factor=1.0,
        )

    def apply(self, cochain: TypedCochain) -> TypedCochain:
        if cochain.space != self.source:
            raise ValueError(f"chain map '{self.name}' expects the declared source space")
        values = (self.matrix.to(cochain.values.dtype) @ cochain.values.reshape(-1)) * self.unit_factor
        return TypedCochain(name=f"{self.name}({cochain.name})", values=values, space=self.target, time=cochain.time, provenance=cochain.provenance)

    def compose(self, first: "LinearChainMap", *, name: Optional[str] = None) -> "LinearChainMap":
        """self o first: first maps into self.source."""
        if first.target != self.source:
            raise ValueError("composition requires first.target == self.source")
        return LinearChainMap(
            name=name or f"{self.name}o{first.name}",
            source=first.source,
            target=self.target,
            matrix=self.matrix.to(torch.float64) @ first.matrix.to(torch.float64),
            unit_factor=self.unit_factor * first.unit_factor,
        )

    def preserves_ledger(
        self,
        source_ledger: Tensor,
        target_ledger: Tensor,
        *,
        tolerance: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """C_t (unit_factor * M) = C_s within tolerance.

        The check is max|C_t (u M) - C_s| <= tolerance on the ledger
        coefficients; None reads ParameterManager key
        validation.iphs.chain_map_ledger_tolerance.
        """
        if tolerance is None:
            from relic.parameters import get_global_manager

            tolerance = get_global_manager().get(
                "validation.iphs.chain_map_ledger_tolerance", required=True
            )
        lhs = target_ledger.to(torch.float64) @ (self.matrix.to(torch.float64) * self.unit_factor)
        residual = float(torch.max(torch.abs(lhs - source_ledger.to(torch.float64))))
        return residual <= tolerance, residual

    def equals(self, other: "LinearChainMap") -> bool:
        """Exact equality of declared maps (an equivalence relation)."""
        return (
            self.source == other.source
            and self.target == other.target
            and self.unit_factor == other.unit_factor
            and bool(torch.equal(self.matrix.to(torch.float64), other.matrix.to(torch.float64)))
        )

    def distance(self, other: "LinearChainMap") -> float:
        """Operator 2-norm distance of the unit-scaled matrices."""
        if self.source != other.source or self.target != other.target:
            raise ValueError("distance requires the same declared spaces")
        return float(torch.linalg.matrix_norm(self.matrix.to(torch.float64) * self.unit_factor - other.matrix.to(torch.float64) * other.unit_factor, ord=2))

    def error_bound(self, perturbation: Tensor) -> float:
        """||M|| * ||delta x|| bounds the propagated cochain error."""
        norm = float(torch.linalg.matrix_norm(self.matrix.to(torch.float64) * self.unit_factor, ord=2))
        return norm * float(torch.linalg.norm(perturbation.to(torch.float64).reshape(-1)))


def close_within(first: LinearChainMap, second: LinearChainMap, epsilon: float) -> bool:
    """Distance bound d(M1, M2) <= epsilon: reflexive and symmetric, not transitive."""
    return first.distance(second) <= epsilon


__all__ = ["LinearChainMap", "close_within"]
