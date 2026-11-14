"""Structural specifications for stochastic IPHS/SDAE declarations.

These classes describe stochastic, descriptor, jump, stabilized-advection,
memory-kernel, ALE, and hybrid-topology terms for validation and compiler
audits. They do not implement a stochastic runtime. Continuous SDE stepping
remains owned by ``relic.stochastic.sde_solver``; jump simulation remains owned
by ``relic.stochastic.gillespie``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Literal, Mapping, Optional, cast

import torch
from torch import Tensor

from .dec.forms import CochainSpaceSpec


NoiseClassification = Literal[
    "fdt_bound",
    "external_forcing",
    "observation_error",
    "parameter_uncertainty",
    "jump_reset",
    "hybrid_topology",
]
CovarianceRelation = Literal["fdt", "coarse_grained", "external", "not_applicable"]


def _parameter_default(key: str) -> float:
    """Read a default tolerance through the canonical infrastructure parameter entry."""
    from relic.parameters import get_global_manager

    return get_global_manager().get(key, required=True)


def _declared_tolerance(payload: Mapping[str, Any]) -> Optional[float]:
    """Return a declared tolerance as float, or None when the key is absent.

    A declared null or non-number is not replaced by the parameter default;
    float() rejects it.
    """
    return float(payload["tolerance"]) if "tolerance" in payload else None


def _reject_unconsumed_tolerance(spec_name: str, payload: Mapping[str, Any]) -> None:
    """Refuse a tolerance for a declaration whose audit consumes none.

    MemoryKernelSpec and HybridTopologyTransitionSpec checks are structural
    (declared operators, routes, keys); a declared tolerance would be silently
    ignored, so it is rejected instead.
    """
    if "tolerance" in payload:
        raise ValueError(
            f"{spec_name} has no tolerance field; no audit of this declaration "
            "consumes a tolerance, so a declared tolerance is not accepted."
        )


def _require_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _declared_bool(
    payload: Mapping[str, Any],
    key: str,
    *,
    required: bool,
    default: bool = False,
) -> bool:
    """Read a boolean declaration without truthiness coercion.

    A required flag that the audit evaluates must be declared by the producer
    (it is never filled with the value that passes the audit). An optional flag
    keeps its fail-safe default when absent. A present value must be a bool:
    bool("false") would be True.
    """
    if key not in payload:
        if required:
            raise ValueError(f"{key} must be declared as a boolean.")
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean, got {value!r}.")
    return value


def _tuple_text(values: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values.strip(),) if values.strip() else ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float64) -> Optional[Tensor]:
    if value is None:
        return None
    return torch.as_tensor(value, dtype=dtype)


def _declared_relation(payload: Mapping[str, Any]) -> str:
    """The covariance relation must be declared; 'not_applicable' is explicit.

    An undeclared relation is not read as 'not_applicable', because that value
    skips every covariance check of validate().
    """
    relation = payload.get("relation", payload.get("covariance_relation"))
    if relation is None:
        raise ValueError(
            "relation (or covariance_relation) must be declared; state "
            "'not_applicable' explicitly when no covariance relation applies."
        )
    return str(relation)


@dataclass(frozen=True)
class NoiseCovarianceSpec:
    """FDT or coarse-grained covariance evidence for stochastic IPHS terms.

    References:
      - Grmela and Ottinger (1997), GENERIC formalism.
      - Gneiting and Raftery (2007), probabilistic validation context.
    Mathematical expression:
      Sigma_h Sigma_h^T = scale * M_noise.
      M_noise = M_noise^T >= 0 and scale >= 0.
    Parameter meanings:
      sigma: volatility matrix Sigma_h.
      dissipative_operator: dissipative matrix M_noise.
      covariance_scale: ParameterManager-resolved scale such as 2 k_B T_ref
        or a coarse-graining factor alpha_noise.
      tolerance: absolute tolerance for the symmetry, PSD, and covariance
        relation checks; None reads ParameterManager key
        validation.iphs.noise_covariance_tolerance.
    """

    name: str
    relation: CovarianceRelation
    sigma: Optional[Tensor] = None
    dissipative_operator: Optional[Tensor] = None
    covariance_scale: Optional[float] = None
    scale_parameter_key: str = ""
    temperature_parameter_key: str = ""
    physical_constant_keys: tuple[str, ...] = field(default_factory=tuple)
    tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        if self.relation not in {"fdt", "coarse_grained", "external", "not_applicable"}:
            raise ValueError("relation is not recognized.")
        if self.tolerance is None:
            object.__setattr__(
                self, "tolerance", _parameter_default("validation.iphs.noise_covariance_tolerance")
            )
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, Real)
            or not math.isfinite(float(self.tolerance))
            or float(self.tolerance) < 0.0
        ):
            raise ValueError("tolerance must be finite and nonnegative.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NoiseCovarianceSpec":
        return cls(
            name=_require_text(str(payload.get("name", "noise_covariance")), "name"),
            relation=cast(CovarianceRelation, _declared_relation(payload)),
            sigma=_as_tensor(payload.get("sigma", payload.get("Sigma"))),
            dissipative_operator=_as_tensor(
                payload.get("dissipative_operator", payload.get("M_noise"))
            ),
            covariance_scale=(
                float(payload["covariance_scale"])
                if payload.get("covariance_scale") is not None
                else None
            ),
            scale_parameter_key=str(payload.get("scale_parameter_key", "")),
            temperature_parameter_key=str(payload.get("temperature_parameter_key", "")),
            physical_constant_keys=_tuple_text(payload.get("physical_constant_keys")),
            tolerance=_declared_tolerance(payload),
        )

    def validate(self) -> dict[str, Any]:
        """Validate covariance evidence without owning stochastic stepping."""

        if self.relation in {"external", "not_applicable"}:
            return {
                "passed": True,
                "name": self.name,
                "relation": self.relation,
                "residual": 0.0,
            }
        issues: list[str] = []
        residual = 0.0
        if self.sigma is None:
            issues.append("sigma is required")
        elif (
            not isinstance(self.sigma, Tensor)
            or self.sigma.ndim != 2
            or self.sigma.numel() == 0
        ):
            issues.append("sigma must be a non-empty rank-2 tensor")
        elif torch.is_complex(self.sigma):
            issues.append("sigma must be real")
        elif not bool(torch.isfinite(self.sigma).all().item()):
            issues.append("sigma must be finite")
        if self.dissipative_operator is None:
            issues.append("dissipative_operator is required")
        elif (
            not isinstance(self.dissipative_operator, Tensor)
            or self.dissipative_operator.ndim != 2
            or self.dissipative_operator.numel() == 0
        ):
            issues.append("dissipative_operator must be a non-empty rank-2 tensor")
        elif self.dissipative_operator.shape[0] != self.dissipative_operator.shape[1]:
            issues.append("dissipative_operator must be square")
        elif torch.is_complex(self.dissipative_operator):
            issues.append("dissipative_operator must be real")
        elif not bool(torch.isfinite(self.dissipative_operator).all().item()):
            issues.append("dissipative_operator must be finite")
        else:
            symmetry_ok = torch.allclose(
                self.dissipative_operator,
                self.dissipative_operator.transpose(-1, -2),
                atol=float(self.tolerance),
                rtol=0.0,
            )
            if not symmetry_ok:
                issues.append("dissipative_operator must be symmetric")
            else:
                minimum_eigenvalue = float(
                    torch.linalg.eigvalsh(self.dissipative_operator)
                    .amin()
                    .detach()
                    .cpu()
                    .item()
                )
                if not math.isfinite(minimum_eigenvalue):
                    issues.append("dissipative_operator eigenvalues must be finite")
                elif minimum_eigenvalue < -float(self.tolerance):
                    issues.append("dissipative_operator must be positive semidefinite")
        if self.covariance_scale is None:
            issues.append("covariance_scale is required")
        elif (
            isinstance(self.covariance_scale, bool)
            or not isinstance(self.covariance_scale, Real)
            or not math.isfinite(float(self.covariance_scale))
            or float(self.covariance_scale) < 0.0
        ):
            issues.append("covariance_scale must be finite nonnegative")
        if not (self.scale_parameter_key or self.temperature_parameter_key):
            issues.append("scale_parameter_key or temperature_parameter_key is required")
        if self.relation == "fdt" and not self.physical_constant_keys:
            issues.append("physical_constant_keys are required for FDT-bound noise")

        sigma = self.sigma
        dissipative_operator = self.dissipative_operator
        covariance_scale = self.covariance_scale
        structurally_valid = (
            isinstance(sigma, Tensor)
            and sigma.ndim == 2
            and sigma.numel() > 0
            and not torch.is_complex(sigma)
            and bool(torch.isfinite(sigma).all().item())
            and isinstance(dissipative_operator, Tensor)
            and dissipative_operator.ndim == 2
            and dissipative_operator.numel() > 0
            and dissipative_operator.shape[0] == dissipative_operator.shape[1]
            and not torch.is_complex(dissipative_operator)
            and bool(torch.isfinite(dissipative_operator).all().item())
            and isinstance(covariance_scale, Real)
            and not isinstance(covariance_scale, bool)
            and math.isfinite(float(covariance_scale))
            and float(covariance_scale) >= 0.0
        )
        if structurally_valid:
            assert isinstance(sigma, Tensor)
            assert isinstance(dissipative_operator, Tensor)
            assert isinstance(covariance_scale, Real)
            covariance = sigma @ sigma.transpose(-1, -2)
            expected = float(covariance_scale) * dissipative_operator
            if tuple(covariance.shape) != tuple(expected.shape):
                issues.append("Sigma Sigma^T shape must match M_noise")
            elif not bool(torch.isfinite(covariance).all().item()) or not bool(
                torch.isfinite(expected).all().item()
            ):
                residual = float("inf")
                issues.append("covariance relation must remain finite")
            else:
                residual = float(torch.linalg.norm(covariance - expected).detach().cpu().item())
                if not math.isfinite(residual):
                    issues.append("covariance residual must be finite")
                elif residual > float(self.tolerance):
                    issues.append("covariance residual exceeds tolerance")

        return {
            "passed": not issues,
            "name": self.name,
            "relation": self.relation,
            "residual": residual,
            "issues": issues,
        }


@dataclass(frozen=True)
class ConservativeNoiseFluxSpec:
    """Conservative flux-space stochastic noise routing declaration.

    Mathematical expression:
      eta_h = Sigma_flux dW_flux; dx_noise = -D_flux eta_h.
    Parameter meanings:
      flux_space: cochain space where stochastic flux is sampled.
      storage_space: cochain space receiving the divergence of flux noise.
      topology_operator_name: declared DEC incidence or exterior derivative.
    """

    name: str
    flux_space: CochainSpaceSpec | Mapping[str, Any] | str
    storage_space: CochainSpaceSpec | Mapping[str, Any] | str
    topology_operator_name: str
    routes_to_storage: bool
    direct_storage_noise: bool

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.topology_operator_name, "topology_operator_name")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConservativeNoiseFluxSpec":
        return cls(
            name=str(payload.get("name", "conservative_noise_flux")),
            flux_space=payload.get("flux_space", payload.get("flow_space", "")),
            storage_space=payload.get("storage_space", payload.get("state_space", "")),
            topology_operator_name=str(
                payload.get(
                    "topology_operator_name",
                    payload.get("incidence_operator", payload.get("exterior_derivative", "")),
                )
            ),
            routes_to_storage=_declared_bool(payload, "routes_to_storage", required=True),
            direct_storage_noise=_declared_bool(payload, "direct_storage_noise", required=True),
        )


@dataclass(frozen=True)
class DescriptorConstraintSpec:
    """Descriptor or DAE constraint declaration for IPHS/SDAE.

    Mathematical expression:
      A_h x = 0 and A_h dx = 0.
    Parameter meanings:
      constraint_operator: A_h.
      constrained_space: cochain space constrained by A_h.
      projection_method: projection or multiplier strategy.
      residual_route: approved route for projection residual.
    """

    name: str
    constraint_operator: Optional[Tensor]
    constrained_space: CochainSpaceSpec | Mapping[str, Any] | str
    units: str
    projection_method: str
    residual_route: str
    enforces_state_constraint: bool
    enforces_tangent_constraint: bool
    constraint_force_neutral: bool
    port_accounting: str = ""

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.units, "units")
        _require_text(self.projection_method, "projection_method")
        _require_text(self.residual_route, "residual_route")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DescriptorConstraintSpec":
        return cls(
            name=str(payload.get("name", "descriptor_constraint")),
            constraint_operator=_as_tensor(payload.get("constraint_operator", payload.get("A_h"))),
            constrained_space=payload.get("constrained_space", payload.get("cochain_space", "")),
            units=str(payload.get("units", "")),
            projection_method=str(
                payload.get("projection_method", payload.get("multiplier_strategy", ""))
            ),
            residual_route=str(payload.get("residual_route", "")),
            enforces_state_constraint=_declared_bool(
                payload, "enforces_state_constraint", required=True
            ),
            enforces_tangent_constraint=_declared_bool(
                payload, "enforces_tangent_constraint", required=True
            ),
            constraint_force_neutral=_declared_bool(
                payload, "constraint_force_neutral", required=True
            ),
            port_accounting=str(payload.get("port_accounting", "")),
        )


@dataclass(frozen=True)
class JumpResetSpec:
    """Jump or reset map declaration for stochastic reaction events.

    Mathematical expression:
      dx_jump = R_h(x^-, eta) dN_t and C_h R_h = 0 for closed ledgers.
    Parameter meanings:
      reset_map: R_h or stoichiometric jump map.
      conservation_matrix: C_h for closed ledger checks.
      propensity_keys: ParameterManager keys for jump propensities.
      tolerance: absolute tolerance for max|C_h R_h| (closed) and the open
        ledger residual; None reads ParameterManager key
        validation.iphs.jump_ledger_tolerance.
    """

    name: str
    reset_map: Optional[Tensor] = None
    stoichiometric_N: Optional[Tensor] = None
    conservation_matrix: Optional[Tensor] = None
    ledger_behavior: str = ""
    propensity_keys: tuple[str, ...] = field(default_factory=tuple)
    tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        if self.tolerance is None:
            object.__setattr__(
                self, "tolerance", _parameter_default("validation.iphs.jump_ledger_tolerance")
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JumpResetSpec":
        return cls(
            name=str(payload.get("name", "jump_reset")),
            reset_map=_as_tensor(payload.get("reset_map", payload.get("R_h"))),
            stoichiometric_N=_as_tensor(payload.get("stoichiometric_N", payload.get("N_h"))),
            conservation_matrix=_as_tensor(payload.get("conservation_matrix", payload.get("C_h"))),
            ledger_behavior=str(payload.get("ledger_behavior", "")),
            propensity_keys=_tuple_text(payload.get("propensity_keys", payload.get("parameter_keys"))),
            tolerance=_declared_tolerance(payload),
        )


@dataclass(frozen=True)
class StabilizedAdvectionSpec:
    """Conservative stabilized advection declaration.

    Mathematical expression:
      x_dot_adv = -A_h(v) x.
    Parameter meanings:
      operator_name: A_h(v) operator identifier.
      scheme: upwind, TVD, limiter, or equivalent conservative scheme.
      cfl_metadata: Courant metadata or ParameterManager key.
    """

    name: str
    operator_name: str
    scheme: str
    conservative: bool
    cfl_metadata: Mapping[str, Any] = field(default_factory=dict)
    peclet_metadata: Mapping[str, Any] = field(default_factory=dict)
    limiter: str = ""
    boundary_flux_route: str = ""
    nonnegative_domain_protection: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StabilizedAdvectionSpec":
        return cls(
            name=str(payload.get("name", "stabilized_advection")),
            operator_name=str(payload.get("operator_name", payload.get("A_h", ""))),
            scheme=str(payload.get("scheme", payload.get("limiter_scheme", ""))),
            conservative=_declared_bool(payload, "conservative", required=False),
            cfl_metadata=dict(payload.get("cfl_metadata", {}))
            if isinstance(payload.get("cfl_metadata", {}), Mapping)
            else {},
            peclet_metadata=dict(payload.get("peclet_metadata", {}))
            if isinstance(payload.get("peclet_metadata", {}), Mapping)
            else {},
            limiter=str(payload.get("limiter", "")),
            boundary_flux_route=str(payload.get("boundary_flux_route", "")),
            nonnegative_domain_protection=_declared_bool(
                payload, "nonnegative_domain_protection", required=False
            ),
        )


@dataclass(frozen=True)
class MemoryKernelSpec:
    """Passive memory-kernel declaration for BIHSS/IPHS audits.

    References:
      - Grmela and Ottinger (1997), GENERIC formalism.
      - Hairer, Lubich, and Wanner (2006), structure-preserving integration.
    Mathematical expression:
      x_dot_memory = -integral_0^t K_h(t - tau) e_h(tau) d tau.
    Parameter meanings:
      kernel_operator: K_h or a finite-dimensional realization of the memory
        kernel.
      hidden_storage_space: cochain space for the auxiliary memory state.
      effort_variable: thermodynamic effort e_h driving the memory term.
      passivity_evidence: passive, positive-real, or storage-realization
        evidence for K_h.
      parameter_keys: ParameterManager keys for kernel coefficients.
      residual_route: declared route for approximation or truncation residuals.
    """

    name: str
    kernel_operator: Optional[Tensor] = None
    hidden_storage_space: CochainSpaceSpec | Mapping[str, Any] | str = ""
    effort_variable: str = ""
    passivity_evidence: str = ""
    parameter_keys: tuple[str, ...] = field(default_factory=tuple)
    residual_route: str = ""

    def __post_init__(self) -> None:
        _require_text(self.name, "name")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MemoryKernelSpec":
        _reject_unconsumed_tolerance(cls.__name__, payload)
        return cls(
            name=str(payload.get("name", "memory_kernel")),
            kernel_operator=_as_tensor(payload.get("kernel_operator", payload.get("K_h"))),
            hidden_storage_space=payload.get(
                "hidden_storage_space",
                payload.get("memory_space", payload.get("cochain_space", "")),
            ),
            effort_variable=str(payload.get("effort_variable", payload.get("effort", ""))),
            passivity_evidence=str(
                payload.get(
                    "passivity_evidence",
                    payload.get("positive_real_evidence", payload.get("storage_evidence", "")),
                )
            ),
            parameter_keys=_tuple_text(payload.get("parameter_keys")),
            residual_route=str(payload.get("residual_route", payload.get("overflow_route", ""))),
        )


@dataclass(frozen=True)
class ALEGeometryCorrectionSpec:
    """ALE moving-geometry Hodge-energy correction declaration.

    References:
      - Desbrun, Kanso, and Tong, discrete differential forms.
      - Arnold, Falk, and Winther (2006), finite element exterior calculus.
    Mathematical expression:
      e_E^T f_ALE = -0.5 x_h^T dot(star_h) x_h.
    Parameter meanings:
      hodge_time_derivative: dot(star_h) induced by moving geometry.
      state_space: cochain space whose metric/Hodge star changes.
      energy_effort: energy effort e_E used for the power audit.
      correction_route: declared route for f_ALE or the equivalent geometry
        port.
      port_accounting: diagnostic or port ledger for geometry power.
      parameter_keys: ParameterManager keys for geometry or mesh velocity.
    """

    name: str
    hodge_time_derivative: Optional[Tensor] = None
    state_space: CochainSpaceSpec | Mapping[str, Any] | str = ""
    energy_effort: str = ""
    correction_route: str = ""
    port_accounting: str = ""
    parameter_keys: tuple[str, ...] = field(default_factory=tuple)
    hodge_operator_name: str = ""

    def __post_init__(self) -> None:
        _require_text(self.name, "name")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ALEGeometryCorrectionSpec":
        return cls(
            name=str(payload.get("name", "ale_geometry_correction")),
            hodge_time_derivative=_as_tensor(
                payload.get(
                    "hodge_time_derivative",
                    payload.get("dot_star_h", payload.get("dstar_dt")),
                )
            ),
            state_space=payload.get("state_space", payload.get("cochain_space", "")),
            energy_effort=str(payload.get("energy_effort", payload.get("effort_variable", ""))),
            correction_route=str(payload.get("correction_route", payload.get("ale_route", ""))),
            port_accounting=str(payload.get("port_accounting", payload.get("geometry_port", ""))),
            parameter_keys=_tuple_text(payload.get("parameter_keys")),
            hodge_operator_name=str(payload.get("hodge_operator_name", payload.get("hodge", ""))),
        )


@dataclass(frozen=True)
class HybridTopologyTransitionSpec:
    """Hybrid topology event and dimension-change remap declaration.

    References:
      - Goebel, Sanfelice, and Teel (2012), hybrid dynamical systems.
      - Arnold, Falk, and Winther (2006), finite element exterior calculus.
    Mathematical expression:
      x_h^+ = P_h x_h^- + R_h eta_h.
      C_h^+ x_h^+ = C_h^- x_h^- + Pi_port.
    Parameter meanings:
      event_name: explicit event that triggers the topology transition.
      source_complex: source DEC complex or cochain space before the reset.
      target_complex: target DEC complex or cochain space after the reset.
      state_remap: P_h state remapping operator across complexes.
      ledger_map: declared ledger or invariant map used for conservation audit.
      ledger_change: declared open-port or managed ledger change Pi_port.
      reset_route: approved event/reset route in the simulator or scheduler.
      parameter_keys: ParameterManager keys controlling event thresholds.
      audit_route: diagnostic route for dimension and ledger remap residuals.
    """

    name: str
    event_name: str = ""
    source_complex: CochainSpaceSpec | Mapping[str, Any] | str = ""
    target_complex: CochainSpaceSpec | Mapping[str, Any] | str = ""
    state_remap: Optional[Tensor] = None
    ledger_map: Optional[Tensor] = None
    ledger_change: Optional[Tensor] = None
    dimension_change: bool = False
    reset_route: str = ""
    parameter_keys: tuple[str, ...] = field(default_factory=tuple)
    audit_route: str = ""

    def __post_init__(self) -> None:
        _require_text(self.name, "name")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HybridTopologyTransitionSpec":
        _reject_unconsumed_tolerance(cls.__name__, payload)
        return cls(
            name=str(payload.get("name", "hybrid_topology_transition")),
            event_name=str(payload.get("event_name", payload.get("event", ""))),
            source_complex=payload.get(
                "source_complex",
                payload.get("source_space", payload.get("source_cochain_space", "")),
            ),
            target_complex=payload.get(
                "target_complex",
                payload.get("target_space", payload.get("target_cochain_space", "")),
            ),
            state_remap=_as_tensor(
                payload.get(
                    "state_remap",
                    payload.get("remap_operator", payload.get("P_h")),
                )
            ),
            ledger_map=_as_tensor(
                payload.get(
                    "ledger_map",
                    payload.get("conservation_matrix", payload.get("C_h")),
                )
            ),
            ledger_change=_as_tensor(
                payload.get(
                    "ledger_change",
                    payload.get("port_ledger_change", payload.get("Pi_port")),
                )
            ),
            dimension_change=_declared_bool(payload, "dimension_change", required=False),
            reset_route=str(payload.get("reset_route", payload.get("event_route", ""))),
            parameter_keys=_tuple_text(payload.get("parameter_keys")),
            audit_route=str(payload.get("audit_route", payload.get("residual_route", ""))),
        )


@dataclass(frozen=True)
class StochasticTermSpec:
    """One stochastic IPHS/SDAE term declaration.

    Mathematical expression:
      dx_noise = Sigma_h dW_t or dx_jump = R_h dN_t.
    Parameter meanings:
      classification: physical source of uncertainty.
      runtime_owner: existing RELIC owner that executes the stochastic term.
      parameter_keys: ParameterManager keys for stochastic coefficients.
    """

    name: str
    classification: NoiseClassification
    runtime_owner: str = "relic.stochastic"
    state_vars: tuple[str, ...] = field(default_factory=tuple)
    parameter_keys: tuple[str, ...] = field(default_factory=tuple)
    noise_type: str = ""
    solver_name: str = ""

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        if self.classification not in {
            "fdt_bound",
            "external_forcing",
            "observation_error",
            "parameter_uncertainty",
            "jump_reset",
            "hybrid_topology",
        }:
            raise ValueError("classification is not recognized.")
        _require_text(self.runtime_owner, "runtime_owner")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StochasticTermSpec":
        return cls(
            name=str(payload.get("name", "stochastic_term")),
            classification=cast(
                NoiseClassification,
                str(
                    payload.get(
                        "classification",
                        payload.get("noise_classification", ""),
                    )
                ),
            ),
            # No package default: an undeclared owner fails _require_text.
            runtime_owner=str(payload.get("runtime_owner", payload.get("owner", ""))),
            state_vars=_tuple_text(payload.get("state_vars")),
            parameter_keys=_tuple_text(payload.get("parameter_keys")),
            noise_type=str(payload.get("noise_type", "")),
            solver_name=str(payload.get("solver_name", payload.get("solver_method", ""))),
        )


__all__ = [
    "ALEGeometryCorrectionSpec",
    "ConservativeNoiseFluxSpec",
    "CovarianceRelation",
    "DescriptorConstraintSpec",
    "HybridTopologyTransitionSpec",
    "JumpResetSpec",
    "MemoryKernelSpec",
    "NoiseClassification",
    "NoiseCovarianceSpec",
    "StabilizedAdvectionSpec",
    "StochasticTermSpec",
]
