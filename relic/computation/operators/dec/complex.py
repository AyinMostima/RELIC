"""Discrete exterior calculus complex bundle."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor

from .forms import TypedChain, TypedCochain


def _require_materialized_dense_tensor(value: Any, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor.")
    if (
        value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
        or bool(getattr(value, "is_nested", False))
    ):
        raise TypeError(f"{label} must be a materialized dense strided tensor.")
    return value


@dataclass(frozen=True)
class DECComplexBundle:
    """Discrete chain/cochain complex and metric support data.

    References:
      - Desbrun et al. (2005), Discrete Exterior Calculus.
      - Arnold, Falk, and Winther (2006), Finite Element Exterior Calculus.
    Mathematical expression:
      D_k = partial_{k+1}^T, D_{k+1} D_k = 0, and D_0 1 = 0.
    Parameter meanings:
      D: already-oriented canonical coboundary matrices D_k. Orientation
        receipts are never multiplied into D a second time.
      n_cells: number of cells for each topological degree.
      tol: norm tolerance for accepting D_{k+1} @ D_k as zero.
      boundary_masks: declared proper boundary-subcomplex membership.
      orientation: int8 basis-orientation receipts; degree 0 is canonical +1.
      cell_measures/dual_measures: positive geometric declarations.
    Units, domain, constraints, and sign convention:
      D is dimensionless integer-valued incidence stored as float32/float64.
      Measures carry geometry but their units are outside this local topology
      class. D signs already encode the current oriented cell basis.
    """

    complex_id: str
    dimension: int
    n_cells: Mapping[int, int]
    D: Mapping[int, Tensor]
    boundary_masks: Mapping[int, Tensor]
    orientation: Mapping[int, Tensor]
    cell_measures: Mapping[int, Tensor]
    dual_measures: Mapping[int, Tensor]
    orientation_convention: str

    def __post_init__(self) -> None:
        if not isinstance(self.complex_id, str) or not self.complex_id.strip():
            raise ValueError("complex_id must be a non-empty string.")
        if type(self.dimension) is not int or self.dimension < 0 or self.dimension > 3:
            raise ValueError("dimension must be an integer in 0..3.")
        if not isinstance(self.orientation_convention, str) or not self.orientation_convention.strip():
            raise ValueError("orientation_convention must be a non-empty string.")
        for map_name, values in (
            ("n_cells", self.n_cells),
            ("D", self.D),
            ("boundary_masks", self.boundary_masks),
            ("orientation", self.orientation),
            ("cell_measures", self.cell_measures),
            ("dual_measures", self.dual_measures),
        ):
            if not isinstance(values, Mapping):
                raise TypeError(f"{map_name} must be a mapping.")
            if any(type(degree) is not int for degree in values):
                raise TypeError(f"{map_name} keys must be integer degrees.")
        for degree in range(self.dimension + 1):
            count = self.n_cells.get(degree)
            if type(count) is not int or count <= 0:
                raise ValueError(f"n_cells[{degree}] must be a positive integer.")
        for degree, matrix in self.D.items():
            if not isinstance(matrix, Tensor):
                raise TypeError("D matrices must be torch.Tensor instances.")
            if matrix.ndim != 2:
                raise ValueError("D matrices must be rank-2 tensors.")
            expected_shape = (
                self.n_cells.get(degree + 1),
                self.n_cells.get(degree),
            )
            if None not in expected_shape and tuple(matrix.shape) != expected_shape:
                raise ValueError(
                    f"D[{degree}] shape {tuple(matrix.shape)} does not match "
                    f"expected shape {expected_shape}."
                )

    def coboundary_matrix(self, degree: int) -> Tensor:
        """Return canonical D_k without applying orientation a second time."""

        if type(degree) is not int:
            raise TypeError("coboundary degree must be an integer.")
        if degree < 0 or degree >= self.dimension:
            raise ValueError(
                f"coboundary degree must be in 0..{self.dimension - 1}."
            )
        matrix = self.D.get(degree)
        if matrix is None:
            raise ValueError(f"D[{degree}] is not declared.")
        return matrix

    def boundary_matrix(self, degree: int) -> Tensor:
        """Return partial_k as the transpose view of the unique D owner."""

        if type(degree) is not int:
            raise TypeError("boundary degree must be an integer.")
        if degree <= 0 or degree > self.dimension:
            raise ValueError(f"boundary degree must be in 1..{self.dimension}.")
        return self.coboundary_matrix(degree - 1).transpose(0, 1)

    def _trace_matrix_unchecked(self, degree: int) -> Tensor:
        mask = self.boundary_masks[degree]
        measure = self.cell_measures[degree]
        identity = torch.eye(
            self.n_cells[degree],
            dtype=measure.dtype,
            device=measure.device,
        )
        return identity[mask]

    def trace_matrix(self, degree: int) -> Tensor:
        """Return the selection map for the declared boundary subcomplex."""

        if type(degree) is not int:
            raise TypeError("trace degree must be an integer.")
        if degree < 0 or degree > self.dimension:
            raise ValueError(f"trace degree must be in 0..{self.dimension}.")
        self.validate_complex(strict=True)
        return self._trace_matrix_unchecked(degree)

    def typed_stokes_audit(
        self,
        alpha: TypedCochain,
        chain: TypedChain,
    ) -> dict[str, Tensor]:
        """Evaluate one typed discrete Stokes pairing without a Hodge star.

        References:
          - Hirani (2003), Discrete Exterior Calculus, Equation 3.6.1.
          - Desbrun, Kanso, and Tong (2006), discrete Stokes theorem.
        Mathematical expression:
          left = <D_k alpha, c>, right = <alpha, D_k^T c>,
          residual = left - right.
        Parameter meanings:
          alpha: a typed k-cochain on this complex.
          chain: a typed (k+1)-chain on this complex.
        Units, domain, constraints, and sign convention:
          Chain coefficients are dimensionless, so both pairings have alpha's
          units. D already contains oriented incidence signs. This point audit
          does not prove mesh provenance or boundary-mask completeness.
        """

        if type(alpha) is not TypedCochain:
            raise TypeError("alpha must be exactly TypedCochain.")
        if type(chain) is not TypedChain:
            raise TypeError("chain must be exactly TypedChain.")
        self.validate_complex(strict=True)
        if alpha.space.complex_id != self.complex_id or chain.space.complex_id != self.complex_id:
            raise ValueError("Typed Stokes complex_id must match the DEC bundle.")
        if chain.space.degree != alpha.space.degree + 1:
            raise ValueError(
                "Typed Stokes degree requires alpha in C^k and chain in C_{k+1}."
            )
        degree = alpha.space.degree
        if degree < 0 or degree >= self.dimension:
            raise ValueError("Typed Stokes degree is outside the DEC bundle.")
        if alpha.space.primality != chain.space.primality:
            raise ValueError("Typed Stokes primality must match.")
        if alpha.space.physical_domain != chain.space.physical_domain:
            raise ValueError("Typed Stokes physical_domain must match.")
        if (
            alpha.space.orientation_convention != self.orientation_convention
            or chain.space.orientation_convention != self.orientation_convention
            or alpha.space.orientation_convention
            != chain.space.orientation_convention
        ):
            raise ValueError("Typed Stokes orientation_convention must match.")
        if alpha.space.n_dofs != self.n_cells[degree]:
            raise ValueError("Typed Stokes alpha n_dofs do not match the complex.")
        if chain.space.n_dofs != self.n_cells[degree + 1]:
            raise ValueError("Typed Stokes chain n_dofs do not match the complex.")
        matrix = self.coboundary_matrix(degree)
        if (
            alpha.values.dtype != matrix.dtype
            or chain.coefficients.dtype != matrix.dtype
            or alpha.values.device != matrix.device
            or chain.coefficients.device != matrix.device
        ):
            raise ValueError(
                "Typed Stokes matrix, cochain, and chain must share dtype and device."
            )

        # References:
        #   - Hirani (2003), Discrete Exterior Calculus, Eq. 3.6.1.
        # Mathematical expression:
        #   <D_k alpha, c> - <alpha, partial_{k+1} c> = 0.
        # Parameter meanings:
        #   matrix is D_k; alpha.values and chain.coefficients are typed inputs.
        left = torch.dot(matrix @ alpha.values, chain.coefficients)
        right = torch.dot(
            alpha.values,
            self.boundary_matrix(degree + 1) @ chain.coefficients,
        )
        residual = left - right
        if not bool(
            torch.stack((left, right, residual)).isfinite().all()
        ):
            raise ValueError("Typed Stokes audit produced a non-finite value.")
        return {"lhs": left, "rhs": right, "residual": residual}

    def validate_complex(
        self,
        tol: float = 0.0,
        *,
        strict: bool = False,
    ) -> dict[tuple[int, int], float]:
        """Validate the discrete complex, optionally requiring full topology data.

        Strict validation is reserved for a declared DEC/BIHSS structural
        claim. Compatibility callers may keep partial bundles and validate only
        the consecutive operators they provide. ``tol`` bounds the norm of
        D[k+1] @ D[k]; the default 0.0 means an exact check (integer incidence
        composes to an exact zero).
        """

        if strict:
            if isinstance(tol, bool) or not isinstance(tol, Real):
                raise ValueError(
                    "tol must be a finite non-negative non-boolean real scalar."
                )
            tol_value = float(tol)
            if not math.isfinite(tol_value) or tol_value < 0.0 or tol_value >= 0.5:
                raise ValueError("strict tol must satisfy 0 <= tol < 0.5.")
        else:
            try:
                tol_value = float(tol)
            except (TypeError, ValueError) as exc:
                raise ValueError("tol must be a finite non-negative number.") from exc
            if not math.isfinite(tol_value) or tol_value < 0.0:
                raise ValueError("tol must be a finite non-negative number.")

        reference_dtype: torch.dtype | None = None
        reference_device: torch.device | None = None
        for degree in sorted(self.D):
            matrix = _require_materialized_dense_tensor(self.D[degree], f"D[{degree}]")
            if matrix.dtype not in {torch.float32, torch.float64}:
                raise TypeError(f"D[{degree}] must use float32 or float64.")
            if matrix.requires_grad or matrix.grad_fn is not None:
                raise ValueError(
                    f"D[{degree}] is a topological incidence operator and "
                    "must not require gradients."
                )
            if not bool(torch.isfinite(matrix).all()):
                raise ValueError(f"D[{degree}] must be finite.")
            if reference_dtype is None:
                reference_dtype = matrix.dtype
                reference_device = matrix.device
            elif matrix.dtype != reference_dtype or matrix.device != reference_device:
                raise ValueError("All DEC D operators must share one dtype and device.")

        if strict:
            required_derivatives = set(range(self.dimension))
            missing_derivatives = sorted(required_derivatives.difference(self.D))
            if missing_derivatives:
                raise ValueError(
                    "Strict DEC validation requires the complete derivative sequence; "
                    f"missing D degrees: {missing_derivatives}."
                )
            unexpected_derivatives = set(self.D).difference(required_derivatives)
            if unexpected_derivatives:
                raise ValueError(
                    "Strict DEC validation found unexpected D degrees: "
                    f"{sorted(unexpected_derivatives, key=str)}."
                )
            if set(self.n_cells) != set(range(self.dimension + 1)):
                raise ValueError(
                    "Strict DEC validation requires exact n_cells degrees 0..dimension."
                )

            for degree in sorted(required_derivatives):
                matrix = self.D[degree]
                if not bool(torch.equal(matrix, torch.round(matrix))):
                    raise ValueError(f"D[{degree}] incidence must be integer-valued.")
                exact_integer_limit = (
                    float(2**24) if matrix.dtype == torch.float32 else float(2**53)
                )
                if bool(torch.any(torch.abs(matrix) > exact_integer_limit)):
                    raise ValueError(
                        f"D[{degree}] incidence exceeds the exact integer range "
                        f"of {matrix.dtype}."
                    )

            if self.dimension > 0:
                d0 = self.D[0]
                constant = torch.ones(
                    self.n_cells[0],
                    dtype=d0.dtype,
                    device=d0.device,
                )
                constant_residual = d0 @ constant
                if not bool(torch.isfinite(constant_residual).all()):
                    raise ValueError("D[0] constant residual must be finite.")
                if bool(torch.any(torch.abs(constant_residual) > tol_value)):
                    raise ValueError("D[0] must annihilate the constant 0-cochain.")

            required_support_degrees = set(range(self.dimension + 1))
            support_maps = {
                "boundary_masks": self.boundary_masks,
                "orientation": self.orientation,
                "cell_measures": self.cell_measures,
                "dual_measures": self.dual_measures,
            }
            for map_name, values in support_maps.items():
                missing_degrees = sorted(required_support_degrees.difference(values))
                if missing_degrees:
                    raise ValueError(
                        f"Strict DEC validation requires {map_name} for every degree; "
                        f"missing degrees: {missing_degrees}."
                    )
                unexpected_degrees = set(values).difference(required_support_degrees)
                if unexpected_degrees:
                    raise ValueError(
                        f"Strict DEC validation found unexpected degrees in {map_name}: "
                        f"{sorted(unexpected_degrees, key=str)}."
                    )

            for degree in sorted(required_support_degrees):
                mask = _require_materialized_dense_tensor(
                    self.boundary_masks[degree],
                    f"boundary_masks[{degree}]",
                )
                if mask.dtype is not torch.bool:
                    raise TypeError(f"boundary_masks[{degree}] must use torch.bool.")
                if mask.ndim != 1 or tuple(mask.shape) != (self.n_cells[degree],):
                    raise ValueError(
                        f"boundary_masks[{degree}] must have shape "
                        f"({self.n_cells[degree]},)."
                    )
                if mask.requires_grad or mask.grad_fn is not None:
                    raise ValueError("boundary_masks must not require gradients.")

                orientation = _require_materialized_dense_tensor(
                    self.orientation[degree],
                    f"orientation[{degree}]",
                )
                if orientation.dtype is not torch.int8:
                    raise TypeError(f"orientation[{degree}] must use torch.int8.")
                if orientation.ndim != 1 or tuple(orientation.shape) != (
                    self.n_cells[degree],
                ):
                    raise ValueError(
                        f"orientation[{degree}] must have shape "
                        f"({self.n_cells[degree]},)."
                    )
                if orientation.requires_grad or orientation.grad_fn is not None:
                    raise ValueError("orientation receipts must not require gradients.")
                if not bool(torch.all(torch.abs(orientation) == 1)):
                    raise ValueError(
                        f"orientation[{degree}] entries must be -1 or 1."
                    )
                if degree == 0 and not bool(torch.all(orientation == 1)):
                    raise ValueError("degree-0 orientation receipts must all be +1.")

                for map_name, values in (
                    ("cell_measures", self.cell_measures),
                    ("dual_measures", self.dual_measures),
                ):
                    measure = _require_materialized_dense_tensor(
                        values[degree],
                        f"{map_name}[{degree}] measure",
                    )
                    if measure.dtype not in {torch.float32, torch.float64}:
                        raise TypeError(
                            f"{map_name}[{degree}] measure must use float32 or float64."
                        )
                    if measure.ndim != 1:
                        raise ValueError(f"{map_name}[{degree}] must be rank-1.")
                    if tuple(measure.shape) != (self.n_cells[degree],):
                        raise ValueError(
                            f"{map_name}[{degree}] must have shape "
                            f"({self.n_cells[degree]},)."
                        )
                    if not bool(torch.isfinite(measure).all()):
                        raise ValueError(f"{map_name}[{degree}] must be finite.")
                    if bool(torch.any(measure <= 0)):
                        raise ValueError(f"{map_name}[{degree}] must be positive.")
                    if reference_dtype is None:
                        reference_dtype = measure.dtype
                        reference_device = measure.device
                    elif (
                        measure.dtype != reference_dtype
                        or measure.device != reference_device
                    ):
                        raise ValueError(
                            "All DEC measure tensors must share the D dtype and device."
                        )

                if reference_device is not None and (
                    mask.device != reference_device
                    or orientation.device != reference_device
                ):
                    raise ValueError(
                        "All DEC support tensors must share the D device."
                    )

            if bool(torch.any(self.boundary_masks[self.dimension])):
                raise ValueError(
                    "A proper boundary subcomplex cannot mark top-degree cells."
                )
            for degree in range(1, self.dimension + 1):
                higher_mask = self.boundary_masks[degree]
                if not bool(torch.any(higher_mask)):
                    continue
                incidence = self.boundary_matrix(degree)
                required_lower = torch.any(incidence[:, higher_mask] != 0, dim=1)
                missing_lower = torch.logical_and(
                    required_lower,
                    torch.logical_not(self.boundary_masks[degree - 1]),
                )
                if bool(torch.any(missing_lower)):
                    raise ValueError(
                        "Declared boundary subcomplex is not closed under the "
                        f"degree-{degree} boundary operator."
                    )

        residuals: dict[tuple[int, int], float] = {}
        for degree in range(self.dimension - 1):
            current = self.D.get(degree)
            next_matrix = self.D.get(degree + 1)
            if current is None or next_matrix is None:
                continue
            if next_matrix.shape[1] != current.shape[0]:
                raise ValueError(
                    f"D[{degree + 1}] and D[{degree}] cannot compose: "
                    f"{tuple(next_matrix.shape)} @ {tuple(current.shape)}."
                )
            # References:
            #   - Arnold, Falk, and Winther (2006), Acta Numerica 15:1-155.
            # Mathematical expression:
            #   D_{k+1,h} D_{k,h} = 0.
            # Parameter meanings:
            #   current is D_k; next_matrix is D_{k+1}; tol is the residual norm limit.
            residual = next_matrix @ current
            if not bool(torch.isfinite(residual).all()):
                raise ValueError(
                    f"D[{degree + 1}] @ D[{degree}] residual must be finite."
                )
            norm_value = float(torch.linalg.norm(residual).detach().cpu().item())
            residuals[(degree + 1, degree)] = norm_value
            if norm_value > tol_value:
                raise ValueError(
                    f"D[{degree + 1}] @ D[{degree}] != 0; residual norm = {norm_value}."
                )
        return residuals


__all__ = ["DECComplexBundle"]
