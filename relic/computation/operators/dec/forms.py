"""Typed discrete-form views for RELIC-IPHS-DEC structural audits.

These classes are structural views over existing RELIC state and SSOT
metadata. They do not replace StateManager, SSOTCatalog, or ContractRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Optional

import torch
from torch import Tensor


FormDegree = Literal[0, 1, 2, 3]
Primality = Literal["primal", "dual"]
Extensivity = Literal["intensive", "extensive", "density", "integral"]
BoundaryRole = Literal["interior", "boundary", "port", "mixed"]


_VALID_DEGREES = {0, 1, 2, 3}
_VALID_PRIMALITY = {"primal", "dual"}
_VALID_EXTENSIVITY = {"intensive", "extensive", "density", "integral"}
_VALID_BOUNDARY_ROLES = {"interior", "boundary", "port", "mixed"}


def _require_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _validate_discrete_space_identity(
    *,
    degree: Any,
    primality: Any,
    n_dofs: Any,
    extensivity: Any = None,
    boundary_role: Any,
    text_fields: Mapping[str, Any],
) -> None:
    if type(degree) is not int or degree not in _VALID_DEGREES:
        raise ValueError("degree must be one of 0, 1, 2, or 3.")
    if primality not in _VALID_PRIMALITY:
        raise ValueError("primality must be 'primal' or 'dual'.")
    if type(n_dofs) is not int or n_dofs <= 0:
        raise ValueError("n_dofs must be a positive integer.")
    if extensivity is not None and extensivity not in _VALID_EXTENSIVITY:
        raise ValueError("extensivity is not recognized.")
    if boundary_role not in _VALID_BOUNDARY_ROLES:
        raise ValueError("boundary_role is not recognized.")
    for field_name, value in text_fields.items():
        _require_non_empty_text(value, field_name)


@dataclass(frozen=True)
class ChainSpaceSpec:
    """Typed support space for a real discrete chain.

    References:
      - Hirani (2003), Discrete Exterior Calculus, Sections 3.2 and 3.6.
      - Desbrun, Kanso, and Tong (2006), discrete differential forms.
    Mathematical expression:
      C_k(K; R) contains c = sum_i c_i sigma_i. The boundary operator is
      intentionally owned by DECComplexBundle, not this value-space class.
    Parameter meanings:
      complex_id: identifier of the oriented discrete complex K.
      degree: k in C_k(K; R), restricted to 0..3 in RELIC.
      primality: whether cells belong to the primal or dual complex.
      n_dofs: number of oriented k-cell basis elements.
      coefficient_units: units of c_i; canonical topological chains use "1".
      physical_domain: declared scientific domain of the integration chain.
      semantic_name: declared chain role without resolving a mesh object.
      orientation_convention: sign convention for oriented basis cells.
      boundary_role: declared interior, boundary, port, or mixed support.
    Units, domain, constraints, and sign convention:
      Coefficients are dimensionless real scalars. Cell measures, dual
      measures, Jacobians, and Hodge weights are excluded from this class.
      Orientation metadata is declarative until bound by DECComplexBundle.
    """

    complex_id: str
    degree: FormDegree
    primality: Primality
    n_dofs: int
    coefficient_units: str
    physical_domain: str
    semantic_name: str
    orientation_convention: str
    boundary_role: BoundaryRole = "interior"

    def __post_init__(self) -> None:
        _validate_discrete_space_identity(
            degree=self.degree,
            primality=self.primality,
            n_dofs=self.n_dofs,
            boundary_role=self.boundary_role,
            text_fields={
                "complex_id": self.complex_id,
                "physical_domain": self.physical_domain,
                "semantic_name": self.semantic_name,
                "orientation_convention": self.orientation_convention,
            },
        )
        if self.coefficient_units != "1":
            raise ValueError(
                "coefficient_units must be '1' for C_k(K; R); geometric "
                "measure belongs to DECComplexBundle."
            )


@dataclass(frozen=True)
class CochainSpaceSpec:
    """Typed support space for a discrete cochain.

    References:
      - Desbrun et al. (2005), Discrete Exterior Calculus.
      - Hirani (2003), Discrete Exterior Calculus.
    Mathematical expression:
      C^k(K) is the vector space of k-cochains on complex K.
    Parameter meanings:
      complex_id: identifier of the discrete complex K.
      degree: k in C^k(K), restricted to 0..3 for RELIC physical domains.
      n_dofs: number of cochain degrees of freedom.
      units: physical unit string inherited from SSOT metadata.
    """

    complex_id: str
    degree: FormDegree
    primality: Primality
    n_dofs: int
    units: str
    extensivity: Extensivity
    physical_domain: str
    semantic_name: str
    orientation_convention: str
    boundary_role: BoundaryRole = "interior"

    def __post_init__(self) -> None:
        _validate_discrete_space_identity(
            degree=self.degree,
            primality=self.primality,
            n_dofs=self.n_dofs,
            extensivity=self.extensivity,
            boundary_role=self.boundary_role,
            text_fields={
                "complex_id": self.complex_id,
                "units": self.units,
                "physical_domain": self.physical_domain,
                "semantic_name": self.semantic_name,
                "orientation_convention": self.orientation_convention,
            },
        )


def _validate_typed_vector(
    *,
    kind: str,
    name: Any,
    payload: Any,
    payload_name: str,
    space: Any,
    expected_space_type: type[Any],
    time: Any,
    provenance: Any,
) -> None:
    if type(space) is not expected_space_type:
        raise TypeError(
            f"{kind} space must be exactly {expected_space_type.__name__}."
        )
    _require_non_empty_text(name, "name")
    if not isinstance(payload, Tensor):
        raise TypeError(f"{kind} {payload_name} must be a torch.Tensor.")
    if payload.device.type == "meta":
        raise TypeError(f"{kind} {payload_name} must be materialized, not meta.")
    if (
        payload.layout != torch.strided
        or payload.is_quantized
        or bool(getattr(payload, "is_nested", False))
    ):
        raise TypeError(
            f"{kind} {payload_name} must use a dense strided Tensor layout."
        )
    if payload.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{kind} {payload_name} must use float32 or float64.")
    if payload.ndim != 1:
        raise ValueError(f"{kind} {payload_name} must be rank-1.")
    if tuple(payload.shape) != (space.n_dofs,):
        raise ValueError(
            f"{kind} {payload_name} do not match n_dofs: "
            f"{tuple(payload.shape)} != ({space.n_dofs},)."
        )
    if not bool(torch.isfinite(payload).all()):
        raise ValueError(f"{kind} {payload_name} must be finite.")
    if time is not None:
        if type(time) not in {int, float}:
            raise ValueError(f"{kind} time must be a finite number when provided.")
        try:
            finite_time = math.isfinite(time)
        except (OverflowError, TypeError, ValueError):
            finite_time = False
        if not finite_time:
            raise ValueError(f"{kind} time must be a finite number when provided.")
    if provenance is not None:
        if not isinstance(provenance, Mapping) or not all(
            type(key) is str
            and bool(key.strip())
            and type(value) is str
            and bool(value.strip())
            for key, value in provenance.items()
        ):
            raise ValueError(
                f"{kind} provenance must map non-empty strings to non-empty strings."
            )


@dataclass(frozen=True)
class TypedCochain:
    """Tensor values with a DEC cochain space contract."""

    name: str
    values: Tensor
    space: CochainSpaceSpec
    time: Optional[float] = None
    provenance: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        _validate_typed_vector(
            kind="TypedCochain",
            name=self.name,
            payload=self.values,
            payload_name="values",
            space=self.space,
            expected_space_type=CochainSpaceSpec,
            time=self.time,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class TypedChain:
    """Real coefficients with an exact DEC chain-space contract.

    References:
      - Hirani (2003), Discrete Exterior Calculus, discrete chains/forms.
      - Desbrun, Kanso, and Tong (2006), discrete Stokes foundations.
    Mathematical expression:
      c = sum_i c_i sigma_i belongs to C_k(K; R). This class stores c_i;
      it does not evaluate boundary, coboundary, pairing, or Stokes operators.
    Parameter meanings:
      name: exact semantic identity of the chain value.
      coefficients: real dimensionless coefficients c_i in the oriented basis.
      space: exact ChainSpaceSpec containing degree, domain, and sign metadata.
      time: optional finite coordinate; it does not imply ALE execution.
      provenance: descriptive non-empty string key/value metadata.
    Units, domain, constraints, and sign convention:
      Coefficients are float32/float64 with coefficient_units="1". They are
      rank-1, finite, dense, materialized, and aligned to space.n_dofs.
      Orientation is declared by space and is not resolved in this class.
    """

    name: str
    coefficients: Tensor
    space: ChainSpaceSpec
    time: Optional[float] = None
    provenance: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        _validate_typed_vector(
            kind="TypedChain",
            name=self.name,
            payload=self.coefficients,
            payload_name="coefficients",
            space=self.space,
            expected_space_type=ChainSpaceSpec,
            time=self.time,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class ObservationOperatorSpec:
    """Typed observation map from model state space to data space."""

    name: str
    input_state_vars: tuple[str, ...]
    output_name: str
    output_units: str
    support: str
    averaging_window: str
    footprint_description: str
    error_model: str
    conservative_projection_required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "output_name",
            "output_units",
            "support",
            "averaging_window",
            "footprint_description",
            "error_model",
        ):
            _require_non_empty_text(getattr(self, field_name), field_name)
        if not self.input_state_vars:
            raise ValueError("input_state_vars must not be empty.")


@dataclass(frozen=True)
class DiagnosticSpec:
    """Typed read-only diagnostic view; feedback ports do not authorize writes."""

    name: str
    input_state_vars: tuple[str, ...]
    output_units: str
    diagnostic_kind: str
    writes_state: bool = False
    feedback_port: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("name", "output_units", "diagnostic_kind"):
            _require_non_empty_text(getattr(self, field_name), field_name)
        if not self.input_state_vars:
            raise ValueError("input_state_vars must not be empty.")
        if not isinstance(self.writes_state, bool):
            raise TypeError("writes_state must be bool.")
        if self.feedback_port is not None:
            _require_non_empty_text(self.feedback_port, "feedback_port")
        if self.writes_state:
            raise ValueError(
                "DiagnosticSpec is read-only; feedback_port cannot authorize state writes."
            )


def cochain_space_from_ssot_entry(
    entry: Any,
    *,
    complex_id: str,
    n_dofs: int,
    degree: Optional[int] = None,
    primality: Optional[str] = None,
    extensivity: Optional[str] = None,
    physical_domain: Optional[str] = None,
    orientation_convention: Optional[str] = None,
    boundary_role: Optional[str] = None,
) -> CochainSpaceSpec:
    """Build a CochainSpaceSpec from an existing SSOTEntry.

    The adapter reads optional DEC metadata from ``entry.raw`` so variable
    identity remains owned by SSOTCatalog.
    """

    raw = getattr(entry, "raw", {}) if entry is not None else {}
    if not isinstance(raw, Mapping):
        raw = {}
    units = getattr(entry, "units", None)
    if not units:
        raise ValueError("SSOT entry must define units before DEC binding.")
    semantic_name = getattr(entry, "standard_name", None) or getattr(entry, "name", None)
    if not semantic_name:
        raise ValueError("SSOT entry must define a semantic name.")

    resolved_degree = degree if degree is not None else raw.get("form_degree")
    resolved_primality = primality if primality is not None else raw.get("primality")
    resolved_extensivity = extensivity if extensivity is not None else raw.get("extensivity")
    resolved_domain = physical_domain if physical_domain is not None else raw.get("physical_domain")
    resolved_orientation = (
        orientation_convention
        if orientation_convention is not None
        else raw.get("orientation_convention")
    )
    # boundary_role enters the strict Hodge boundary-role check, so it is part
    # of the declared structural identity like the fields below; an undeclared
    # role is not read as "interior".
    resolved_boundary = boundary_role if boundary_role is not None else raw.get("boundary_role")

    if resolved_degree is None:
        raise ValueError("form_degree is required for DEC cochain binding.")
    if resolved_primality is None:
        raise ValueError("primality is required for DEC cochain binding.")
    if resolved_extensivity is None:
        raise ValueError("extensivity is required for DEC cochain binding.")
    if resolved_domain is None:
        raise ValueError("physical_domain is required for DEC cochain binding.")
    if resolved_orientation is None:
        raise ValueError("orientation_convention is required for DEC cochain binding.")
    if resolved_boundary is None:
        raise ValueError("boundary_role is required for DEC cochain binding.")

    return CochainSpaceSpec(
        complex_id=complex_id,
        degree=resolved_degree,
        primality=str(resolved_primality),
        n_dofs=n_dofs,
        units=str(units),
        extensivity=str(resolved_extensivity),
        physical_domain=str(resolved_domain),
        semantic_name=str(semantic_name),
        orientation_convention=str(resolved_orientation),
        boundary_role=str(resolved_boundary),
    )


__all__ = [
    "BoundaryRole",
    "ChainSpaceSpec",
    "CochainSpaceSpec",
    "DiagnosticSpec",
    "Extensivity",
    "FormDegree",
    "ObservationOperatorSpec",
    "Primality",
    "TypedChain",
    "TypedCochain",
    "cochain_space_from_ssot_entry",
]
