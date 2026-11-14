"""Discrete Hodge star specifications and passivity checks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .forms import CochainSpaceSpec

_HODGE_ROLES = frozenset({"unspecified", "metric_hodge", "material_hodge"})
_DEFINITENESS_CLASSES = frozenset(
    {"unspecified", "positive_definite", "positive_semidefinite"}
)
_RELATIVE_TOLERANCE_PARAMETER_KEYS = {
    "symmetry_tol": "validation.iphs.hodge_symmetry_rtol",
    "psd_tol": "validation.iphs.hodge_psd_rtol",
    "kernel_rtol": "validation.iphs.hodge_kernel_rtol",
}


def _default_relative_tolerance(field_name: str) -> Any:
    """Read a default relative Hodge tolerance from the parameter database.

    HodgeStarSpec is infrastructure, not a Functor, so it uses the canonical
    ``get_global_manager().get(key, required=True)`` entry; a missing key raises
    instead of falling back. The value is validated like an explicit tolerance.
    """
    from relic.parameters import get_global_manager

    return get_global_manager().get(_RELATIVE_TOLERANCE_PARAMETER_KEYS[field_name], required=True)


def _validate_relative_tolerance(value: Any, field_name: str) -> None:
    """Validate a dimensionless tolerance that cannot disable a sign gate."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} relative tolerance must be a finite number in [0, 0.5)."
        )
    if not math.isfinite(float(value)) or not 0.0 <= float(value) < 0.5:
        raise ValueError(
            f"{field_name} relative tolerance must be a finite number in [0, 0.5)."
        )


def _validate_dense_real_matrix(value: Any, field_name: str) -> Tensor:
    """Validate a materialized real float matrix without copying or detaching it."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor matrix.")
    if value.device.type == "meta":
        raise TypeError(f"{field_name} matrix must be materialized, not meta.")
    if (
        value.layout != torch.strided
        or value.is_quantized
        or bool(getattr(value, "is_nested", False))
    ):
        raise TypeError(f"{field_name} matrix must use a dense strided Tensor layout.")
    if value.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{field_name} matrix must use float32 or float64 dtype.")
    if value.ndim != 2:
        raise ValueError(f"{field_name} matrix must be rank-2.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field_name} matrix must be finite.")
    return value


@dataclass(frozen=True)
class HodgeStarSpec:
    """Declared metric or material Hodge star operator.

    References:
      - Hirani (2003), Discrete Exterior Calculus.
      - Desbrun et al. (2005), Discrete Exterior Calculus.
    Mathematical expression:
      star_h: C^k(K) -> C^{n-k}(dual K).
      A metric Hodge is symmetric positive definite. A singular material
      Hodge is symmetric positive semidefinite with an explicit full kernel K:
      H K = 0 and dim(K) = numerical_nullity(H).
    Parameter meanings:
      matrix: discrete Hodge matrix.
      parameter_keys: ParameterManager keys used to assemble material terms.
      metric_keys: ParameterManager keys used to assemble metric terms.
      role: unspecified, metric_hodge, or material_hodge.
      definiteness: unspecified, positive_definite, or positive_semidefinite.
      kernel_basis: explicit columns spanning a singular material kernel.
      symmetry_tol: dimensionless relative tolerance for H - H^T.
      psd_tol: dimensionless relative spectral tolerance.
      kernel_rtol: dimensionless relative kernel residual/rank tolerance.
      A tolerance of None reads ParameterManager key
        validation.iphs.hodge_symmetry_rtol, hodge_psd_rtol, or
        hodge_kernel_rtol at construction; an explicit value overrides.
    """

    name: str
    domain: CochainSpaceSpec
    codomain: CochainSpaceSpec
    matrix: Tensor
    parameter_keys: tuple[str, ...]
    metric_keys: tuple[str, ...]
    material_law: str
    symmetry_tol: float | None = None
    psd_tol: float | None = None
    complex_dimension: int | None = None
    operator_units: str | None = None
    role: str = "unspecified"
    definiteness: str = "unspecified"
    kernel_basis: Tensor | None = None
    kernel_rtol: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        if type(self.domain) is not CochainSpaceSpec:
            raise TypeError("domain must be exactly CochainSpaceSpec.")
        if type(self.codomain) is not CochainSpaceSpec:
            raise TypeError("codomain must be exactly CochainSpaceSpec.")
        if type(self.role) is not str or self.role not in _HODGE_ROLES:
            raise ValueError(
                "role must be one of unspecified, metric_hodge, or material_hodge."
            )
        if (
            type(self.definiteness) is not str
            or self.definiteness not in _DEFINITENESS_CLASSES
        ):
            raise ValueError(
                "definiteness must be unspecified, positive_definite, "
                "or positive_semidefinite."
            )
        _validate_dense_real_matrix(self.matrix, "Hodge")
        expected_shape = (self.codomain.n_dofs, self.domain.n_dofs)
        if tuple(self.matrix.shape) != expected_shape:
            raise ValueError(
                f"Hodge matrix shape {tuple(self.matrix.shape)} does not match "
                f"expected shape {expected_shape}."
            )
        if not isinstance(self.material_law, str) or not self.material_law.strip():
            raise ValueError("material_law must be a non-empty string.")
        if not self.parameter_keys:
            raise ValueError("parameter_keys must not be empty.")
        if not self.metric_keys:
            raise ValueError("metric_keys must not be empty.")
        for key in (*self.parameter_keys, *self.metric_keys):
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Hodge parameter and metric keys must be non-empty strings.")
        for field_name in _RELATIVE_TOLERANCE_PARAMETER_KEYS:
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, _default_relative_tolerance(field_name))
            _validate_relative_tolerance(getattr(self, field_name), field_name)
        if self.complex_dimension is not None and (
            isinstance(self.complex_dimension, bool)
            or not isinstance(self.complex_dimension, int)
            or self.complex_dimension < 0
            or self.complex_dimension > 3
        ):
            raise ValueError("complex_dimension must be an integer in 0..3 when provided.")
        if self.operator_units is not None and (
            not isinstance(self.operator_units, str) or not self.operator_units.strip()
        ):
            raise ValueError("operator_units must be a non-empty string when provided.")
        if self.kernel_basis is not None:
            kernel = _validate_dense_real_matrix(self.kernel_basis, "kernel_basis")
            if tuple(kernel.shape[:1]) != (self.domain.n_dofs,):
                raise ValueError(
                    "kernel_basis shape must be (domain.n_dofs, kernel_dimension)."
                )
            if kernel.dtype != self.matrix.dtype:
                raise TypeError("kernel_basis dtype must match the Hodge matrix dtype.")
            if kernel.device != self.matrix.device:
                raise TypeError("kernel_basis device must match the Hodge matrix device.")
            if kernel.requires_grad:
                raise ValueError("kernel_basis requires_grad must be False.")

    @property
    def is_square(self) -> bool:
        """Return whether the Hodge matrix is square."""

        return self.matrix.shape[0] == self.matrix.shape[1]

    def _validate_strict_structure(
        self,
        *,
        parameter_resolver: Callable[[str], Any] | None,
    ) -> None:
        if self.domain.complex_id != self.codomain.complex_id:
            raise ValueError("A strict Hodge star must map within the same complex.")
        if self.domain.primality == self.codomain.primality:
            raise ValueError("A strict Hodge star requires opposite primality.")
        if self.domain.physical_domain != self.codomain.physical_domain:
            raise ValueError("A strict Hodge star must preserve the physical domain.")
        if self.domain.orientation_convention != self.codomain.orientation_convention:
            raise ValueError("A strict Hodge star requires a consistent orientation convention.")
        if self.domain.boundary_role != self.codomain.boundary_role:
            raise ValueError("A strict Hodge star must preserve the boundary role.")
        if self.complex_dimension is None:
            raise ValueError("A strict Hodge star requires complex_dimension.")
        if self.domain.degree + self.codomain.degree != self.complex_dimension:
            raise ValueError(
                "A strict Hodge star requires complementary form degrees "
                "k and complex_dimension - k."
            )

        if not callable(parameter_resolver):
            raise ValueError("Strict Hodge validation requires a ParameterManager key resolver.")
        for key in (*self.parameter_keys, *self.metric_keys):
            if "*" in key:
                raise ValueError(f"Strict Hodge ParameterManager key '{key}' cannot be a wildcard.")
            try:
                resolved = parameter_resolver(key)
            except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
                raise ValueError(
                    f"Strict Hodge ParameterManager key '{key}' could not be resolved."
                ) from exc
            if resolved is None:
                raise ValueError(
                    f"Strict Hodge ParameterManager key '{key}' could not be resolved."
                )

        if not self.operator_units:
            raise ValueError("Strict Hodge validation requires operator_units.")
        try:
            from cf_units import Unit

            mapped_units = Unit(self.operator_units) * Unit(self.domain.units)
            unit_relation_ok = mapped_units.is_convertible(Unit(self.codomain.units))
        except (ImportError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Strict Hodge unit relation could not be parsed.") from exc
        if not unit_relation_ok:
            raise ValueError(
                "Strict Hodge unit relation is inconsistent: "
                "operator_units * domain.units must be convertible to codomain.units."
            )

    def validate(
        self,
        *,
        passive: bool = True,
        strict: bool = False,
        parameter_resolver: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        """Validate the current tensor value and declared Hodge structure."""

        if type(passive) is not bool or type(strict) is not bool:
            raise TypeError("passive and strict must be exact bool values.")

        audit: dict[str, Any] = {
            "name": self.name,
            "shape_ok": True,
            "role": self.role,
            "declared_definiteness": self.definiteness,
            "passivity_evaluated": False,
            "symmetric": False,
            "psd": False,
            "definiteness_verified": False,
            "min_eigenvalue": None,
            "kernel_dimension": None,
            "kernel_residual": None,
            "active_rank": None,
            "zero_operator": False,
            "pointwise_current_value_only": True,
            "structural_duality": False,
            "parameter_binding": False,
            "unit_relation": False,
        }

        if strict:
            if self.role == "unspecified":
                raise ValueError("Strict Hodge validation requires an explicit role.")
            if self.definiteness == "unspecified":
                raise ValueError(
                    "Strict Hodge validation requires an explicit definiteness."
                )
            if not passive:
                raise ValueError("Strict Hodge validation requires passive=True.")
            if self.role == "metric_hodge" and self.definiteness != "positive_definite":
                raise ValueError(
                    "A strict metric_hodge must declare positive_definite."
                )

            precision_limit = math.sqrt(torch.finfo(self.matrix.dtype).eps)
            if any(
                float(getattr(self, field_name)) > precision_limit
                for field_name in ("symmetry_tol", "psd_tol", "kernel_rtol")
            ):
                raise ValueError(
                    "A strict tolerance cannot exceed sqrt(machine epsilon) "
                    "for the Hodge matrix dtype."
                )
            self._validate_strict_structure(parameter_resolver=parameter_resolver)
            audit["structural_duality"] = True
            audit["parameter_binding"] = True
            audit["unit_relation"] = True

        if not passive:
            return audit
        if not self.is_square:
            raise ValueError("A passive Hodge operator must be square.")

        matrix = self.matrix
        # References:
        #   - Hirani (2003), Discrete Exterior Calculus, Chapter 6.
        #   - Desbrun, Kanso, and Tong (2008), Discrete Differential Forms.
        # Mathematical expression:
        #   rho_sym = ||H - H^T||_F / max(||H||_F, tiny).
        #   tau_psd = psd_tol * max_i |lambda_i|.
        #   PD: lambda_min > tau_psd; PSD: lambda_min >= -tau_psd.
        #   Singular material PSD: rank(K) = dim ker(H) and H K = 0.
        # Parameter meanings:
        #   matrix is the current discrete Hodge tensor value.
        #   symmetry_tol and psd_tol are dimensionless relative tolerances.
        #   kernel_rtol controls normalized kernel residual and rank checks.
        dtype_info = torch.finfo(matrix.dtype)
        dimension_factor = max(matrix.shape)
        roundoff_floor = float(dtype_info.eps) * float(dimension_factor)
        effective_symmetry_rtol = max(float(self.symmetry_tol), roundoff_floor)
        effective_psd_rtol = max(float(self.psd_tol), roundoff_floor)
        effective_kernel_rtol = max(float(self.kernel_rtol), roundoff_floor)
        audit["configured_symmetry_rtol"] = float(self.symmetry_tol)
        audit["configured_psd_rtol"] = float(self.psd_tol)
        audit["configured_kernel_rtol"] = float(self.kernel_rtol)
        audit["effective_symmetry_rtol"] = effective_symmetry_rtol
        audit["effective_psd_rtol"] = effective_psd_rtol
        audit["effective_kernel_rtol"] = effective_kernel_rtol

        matrix_norm = torch.linalg.matrix_norm(matrix, ord="fro")
        sym_residual = torch.linalg.matrix_norm(
            matrix - matrix.transpose(-1, -2), ord="fro"
        )
        matrix_norm_value = float(matrix_norm.detach().cpu().item())
        sym_residual_value = float(sym_residual.detach().cpu().item())
        if not math.isfinite(matrix_norm_value) or not math.isfinite(sym_residual_value):
            raise ValueError("Derived Hodge symmetry diagnostics must be finite.")
        if matrix_norm_value == 0.0:
            relative_symmetry_residual = 0.0 if sym_residual_value == 0.0 else math.inf
        else:
            relative_symmetry_residual = sym_residual_value / matrix_norm_value
        if not math.isfinite(relative_symmetry_residual):
            raise ValueError("Derived Hodge symmetry residual must be finite.")
        audit["symmetry_residual"] = relative_symmetry_residual
        audit["symmetric"] = relative_symmetry_residual <= effective_symmetry_rtol
        if not audit["symmetric"]:
            raise ValueError(
                f"Hodge matrix '{self.name}' is not symmetric; relative residual = "
                f"{relative_symmetry_residual}."
            )

        symmetric_part = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(symmetric_part)
        if not bool(torch.isfinite(eigenvalues).all()):
            raise ValueError("Derived Hodge eigenvalues must be finite.")
        min_eig = float(torch.min(eigenvalues).detach().cpu().item())
        spectral_scale = float(torch.max(torch.abs(eigenvalues)).detach().cpu().item())
        psd_threshold = effective_psd_rtol * spectral_scale
        if not all(math.isfinite(value) for value in (min_eig, spectral_scale, psd_threshold)):
            raise ValueError("Derived Hodge spectral diagnostics must be finite.")
        audit["passivity_evaluated"] = True
        audit["min_eigenvalue"] = min_eig
        audit["spectral_scale"] = spectral_scale
        audit["psd_threshold"] = psd_threshold
        audit["zero_operator"] = spectral_scale == 0.0
        audit["psd"] = min_eig >= -psd_threshold
        if not audit["psd"]:
            raise ValueError(
                f"Hodge matrix '{self.name}' is not positive semidefinite; "
                f"min eigenvalue = {min_eig}."
            )

        if not strict:
            return audit

        if self.definiteness == "positive_definite":
            if min_eig <= psd_threshold:
                raise ValueError(
                    f"Hodge matrix '{self.name}' is not positive definite; "
                    f"min eigenvalue = {min_eig}, threshold = {psd_threshold}."
                )
            if self.kernel_basis is not None:
                raise ValueError(
                    "A positive definite Hodge must use kernel_basis=None."
                )
            audit["active_rank"] = int(matrix.shape[0])
            audit["definiteness_verified"] = True
            return audit

        if self.role != "material_hodge":
            raise ValueError(
                "Only a strict material_hodge may declare positive_semidefinite."
            )
        kernel = self.kernel_basis
        if kernel is None or kernel.shape[1] == 0:
            raise ValueError(
                "A positive_semidefinite material_hodge requires a nonempty kernel_basis."
            )

        nullity = int(torch.count_nonzero(torch.abs(eigenvalues) <= psd_threshold).item())
        kernel_dimension = int(kernel.shape[1])
        singular_values = torch.linalg.svdvals(kernel)
        if not bool(torch.isfinite(singular_values).all()):
            raise ValueError("Derived kernel singular values must be finite.")
        largest_singular = float(torch.max(singular_values).detach().cpu().item())
        rank_threshold = effective_kernel_rtol * largest_singular
        kernel_rank = int(torch.count_nonzero(singular_values > rank_threshold).item())
        if kernel_rank != kernel_dimension:
            raise ValueError("kernel_basis columns must be linearly independent.")
        if kernel_dimension != nullity:
            raise ValueError(
                "kernel_basis dimension must equal the numerical Hodge nullity."
            )

        kernel_image = matrix @ kernel
        kernel_image_norm = torch.linalg.matrix_norm(kernel_image, ord="fro")
        operator_norm = torch.linalg.matrix_norm(matrix, ord=2)
        kernel_norm = torch.linalg.matrix_norm(kernel, ord="fro")
        image_norm_value = float(kernel_image_norm.detach().cpu().item())
        denominator = float((operator_norm * kernel_norm).detach().cpu().item())
        if not math.isfinite(image_norm_value) or not math.isfinite(denominator):
            raise ValueError("Derived kernel residual terms must be finite.")
        if denominator == 0.0:
            kernel_residual = 0.0 if image_norm_value == 0.0 else math.inf
        else:
            kernel_residual = image_norm_value / denominator
        if not math.isfinite(kernel_residual):
            raise ValueError("Derived kernel residual must be finite.")
        if kernel_residual > effective_kernel_rtol:
            raise ValueError(
                f"kernel_basis residual {kernel_residual} exceeds the relative tolerance."
            )

        audit["kernel_dimension"] = kernel_dimension
        audit["kernel_residual"] = kernel_residual
        audit["active_rank"] = int(matrix.shape[0]) - kernel_dimension
        audit["definiteness_verified"] = True
        return audit


__all__ = ["HodgeStarSpec"]
