"""Irreversible port-Hamiltonian structural specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Literal, Mapping, Optional

import torch
from torch import Tensor

from .dec.forms import CochainSpaceSpec


PortKind = Literal["energy", "entropy", "matter", "heat", "momentum", "radiation", "management"]
PowerConvention = Literal["into_system", "out_of_system"]
PortRole = Literal["unspecified", "internal", "external"]
IPHSFormulation = Literal["unspecified", "dissipative_ph", "generic"]
OperatorStateDependence = Literal["unspecified", "constant", "state_dependent"]
PoissonOperatorKind = Literal["matrix_snapshot", "zero_field", "constant_field", "state_field"]
ReactionRepresentation = Literal["independent", "absorbed"]
StorageKind = Literal["unspecified", "total_energy", "free_energy"]
OpenSystemRoute = Literal[
    "boundary_export",
    "managed_export",
    "diagnostic_residual",
    "port",
    "observation",
]

_VALID_OPEN_SYSTEM_ROUTES = {
    "boundary_export",
    "managed_export",
    "diagnostic_residual",
    "port",
    "observation",
}
_VALID_PORT_KINDS = {
    "energy",
    "entropy",
    "matter",
    "heat",
    "momentum",
    "radiation",
    "management",
}
_VALID_POWER_CONVENTIONS = {"into_system", "out_of_system"}
_VALID_PORT_ROLES = {"unspecified", "internal", "external"}
_SUPPORTED_IPHS_DTYPES = {torch.float32, torch.float64}
_REACTION_POINT_AUDIT_DTYPES = {torch.float32, torch.float64}
_AFFINITY_RATE_SIGN_CONVENTION = "affinity_times_rate_nonnegative"
_THERMODYNAMIC_OPEN_ROUTES = {"port", "boundary_export", "managed_export"}
_TOLERANCE_PARAMETER_KEYS = {
    "skew_tol": "validation.iphs.skew_tolerance",
    "psd_tol": "validation.iphs.psd_tolerance",
    "degeneracy_tol": "validation.iphs.degeneracy_tolerance",
    "jacobi_tol": "validation.iphs.jacobi_tolerance",
    "conservation_tolerance": "validation.iphs.reaction_conservation_tolerance",
    "thermodynamic_rtol": "validation.iphs.reaction_thermodynamic_rtol",
    "reaction_power_scale_floor": "validation.iphs.reaction_power_scale_floor",
}
_STRUCTURE_TOLERANCE_FIELDS = ("skew_tol", "psd_tol", "degeneracy_tol", "jacobi_tol")


def _resolve_iphs_tolerance(argument: str, value: Any) -> Any:
    """Return an explicit tolerance unchanged, or read its default from the parameter database.

    The IPHS owner is infrastructure, not a Functor, so a missing value is read
    through the canonical ``get_global_manager().get(key, required=True)``
    entry; a missing key raises instead of falling back. Explicit and resolved
    values pass through the same validation at their point of use.
    """
    if value is not None:
        return value
    from relic.parameters import get_global_manager

    return get_global_manager().get(_TOLERANCE_PARAMETER_KEYS[argument], required=True)


def poisson_jacobi_residual(field: Any, point: Tensor) -> Tensor:
    """Evaluate the Jacobi identity residual of a state-dependent Poisson candidate at one point.

    References:
      - Grmela and Ottinger (1997), GENERIC formalism, Poisson bracket requirements.
      - Marsden and Ratiu (1999), Introduction to Mechanics and Symmetry, Ch. 10 (Poisson
        manifolds; coordinate form of the Jacobi identity).
    Mathematical expression:
      R_ijk(x) = sum_l ( J_il d_l J_jk + J_jl d_l J_ki + J_kl d_l J_ij ), evaluated at x;
      J is a Poisson tensor iff J^T = -J and R(x) = 0 for every x in the domain.
    Parameter meanings:
      field maps a state vector x (n,) to the candidate matrix J(x) (n, n); dJ[i, j, l] is
      the autograd Jacobian d J_ij / d x_l at the point.
    Units:
      R carries the units of J times J per state unit; the check compares its absolute
      value with a dimensionless-scaled tolerance on the declared normalized basis.
    Domain/constraints:
      the point must be a finite rank-1 floating tensor; the field must be differentiable
      at the point (autograd); a nonzero residual disproves the Poisson property, a zero
      residual at sampled points is evidence only on the sampled domain.
    Sign convention:
      not applicable; the residual is an algebraic identity defect.
    """
    if not isinstance(point, Tensor) or point.ndim != 1 or not point.is_floating_point():
        raise ValueError("Jacobi residual requires a rank-1 floating point tensor.")
    if not bool(torch.isfinite(point).all().item()):
        raise ValueError("Jacobi residual requires a finite point.")
    probe = point.detach().clone()
    matrix = field(probe)
    if not isinstance(matrix, Tensor) or matrix.shape != (probe.shape[0], probe.shape[0]):
        raise ValueError("poisson_field must return an (n, n) tensor for an n-dimensional state.")
    d_matrix = torch.autograd.functional.jacobian(field, probe)
    if not isinstance(d_matrix, Tensor) or d_matrix.shape != (probe.shape[0], probe.shape[0], probe.shape[0]):
        raise ValueError("poisson_field Jacobian must have shape (n, n, n).")
    matrix = matrix.detach()
    d_matrix = d_matrix.detach()
    term_one = torch.einsum("il,jkl->ijk", matrix, d_matrix)
    term_two = torch.einsum("jl,kil->ijk", matrix, d_matrix)
    term_three = torch.einsum("kl,ijl->ijk", matrix, d_matrix)
    return term_one + term_two + term_three


@dataclass(frozen=True)
class PortSpecIPHS:
    """Internal IPHS assembly view over the existing orchestration port system."""

    name: str
    effort_var: str
    flow_var: str
    effort_space: CochainSpaceSpec
    flow_space: CochainSpaceSpec
    power_units: str
    positive_power_convention: PowerConvention
    port_kind: PortKind
    role: PortRole = "unspecified"

    def __post_init__(self) -> None:
        for field_name in ("name", "effort_var", "flow_var", "power_units"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"PortSpecIPHS {field_name} must be a non-empty string.")
        if not isinstance(self.effort_space, CochainSpaceSpec) or not isinstance(
            self.flow_space,
            CochainSpaceSpec,
        ):
            raise TypeError(
                "PortSpecIPHS effort_space and flow_space must be CochainSpaceSpec instances."
            )
        if self.positive_power_convention not in _VALID_POWER_CONVENTIONS:
            raise ValueError(
                "PortSpecIPHS positive_power_convention must be 'into_system' "
                "or 'out_of_system'."
            )
        if self.port_kind not in _VALID_PORT_KINDS:
            raise ValueError("PortSpecIPHS port_kind is not recognized.")
        if self.role not in _VALID_PORT_ROLES:
            raise ValueError(
                "PortSpecIPHS role must be 'unspecified', 'internal', or 'external'."
            )
        if self.effort_space.n_dofs != self.flow_space.n_dofs:
            raise ValueError(
                "PortSpecIPHS effort and flow spaces must have equal n_dofs."
            )
        if (
            self.effort_space.boundary_role == "interior"
            and self.flow_space.boundary_role == "interior"
        ):
            raise ValueError(
                "PortSpecIPHS requires boundary or port support for effort or flow."
            )

    @property
    def n_dofs(self) -> int:
        """Return the dimension of the dual effort-flow port block."""

        return int(self.effort_space.n_dofs)


def _port_dof_count(ports: Iterable[PortSpecIPHS]) -> int:
    """Return the total typed port dimension without creating a second owner."""

    total = 0
    for port in ports:
        if not isinstance(port, PortSpecIPHS):
            raise TypeError("ports must contain only PortSpecIPHS instances.")
        total += port.n_dofs
    return total


def _port_column_indices(
    ports: Iterable[PortSpecIPHS],
    *,
    role: Literal["internal", "external"],
) -> tuple[int, ...]:
    """Expand typed port blocks to their contiguous G-column indices."""

    indices: list[int] = []
    offset = 0
    for port in ports:
        if not isinstance(port, PortSpecIPHS):
            raise TypeError("ports must contain only PortSpecIPHS instances.")
        stop = offset + port.n_dofs
        if port.role == role:
            indices.extend(range(offset, stop))
        offset = stop
    return tuple(indices)


@dataclass(frozen=True)
class IPHSStructureSpec:
    """Formulation-specific port-Hamiltonian or GENERIC structure.

    References:
      - van der Schaft and Maschke (1995), Port-Hamiltonian systems.
      - Grmela and Ottinger (1997), GENERIC formalism.
      - Ramirez et al. (2022), Irreversible port-Hamiltonian systems.
    Mathematical expressions:
      dissipative_ph: x_dot = (J - R) e_H + G_ext u_ext + N r.
      generic: x_dot = J e_E + M e_S + G_ext u_ext + N r.
      These expressions use independent reactions. Absorbed reactions satisfy
      N r = -R e_H or M e_S and omit the final independent N r term.
    Parameter meanings:
      J: reversible skew-symmetric interconnection matrix.
      R: dissipative port-Hamiltonian resistance matrix.
      M: GENERIC entropy-gradient mobility matrix.
      G: optional combined port incidence matrix. Internal columns are eliminated
        through C/D feedback; external columns receive runtime port input.
      stoichiometric_N: optional reaction stoichiometric matrix; strict Level 4/5
        coupling is checked at a bound runtime point (Level 5 GENERIC pairing;
        Level 4 pairing selected by storage_kind); no production dPH process with
        a reaction is bound yet.
      reversible_operator_state_dependence: descriptive state-dependence claim only;
        a matrix sample alone cannot certify a global Jacobi identity.
      poisson_operator: actual reversible evaluation policy (required); zero_field
        generates J(x)=0 and checks the declared matrix for exact consistency on
        each use.
      reaction_representation: independent adds N r to the drift; absorbed
        requires N r to equal the full irreversible rate and does not add it again
        (required).
    """

    name: str
    state_vars: tuple[str, ...]
    energy_functional: str
    entropy_functional: str
    J: Tensor
    M: Tensor
    G: Optional[Tensor]
    stoichiometric_N: Optional[Tensor]
    ports: tuple[PortSpecIPHS, ...]
    R: Optional[Tensor] = None
    formulation: IPHSFormulation = "unspecified"
    reversible_operator_state_dependence: OperatorStateDependence = "unspecified"
    jacobi_identity_evidence: str = ""
    # Structural tolerances: None reads validation.iphs.skew_tolerance,
    # validation.iphs.psd_tolerance and validation.iphs.degeneracy_tolerance from
    # ParameterManager at construction; an explicit value overrides.
    skew_tol: Optional[float] = None
    psd_tol: Optional[float] = None
    degeneracy_tol: Optional[float] = None
    energy_gradient: Optional[Tensor] = None
    entropy_gradient: Optional[Tensor] = None
    contribution_count: int = 0
    contribution_sources: tuple[str, ...] = ()
    poisson_operator: PoissonOperatorKind = field(kw_only=True)
    reaction_representation: ReactionRepresentation = field(kw_only=True)
    # dPH storage classification used only by the Level 4 reaction pairing:
    # total_energy requires e_H^T N r = 0, free_energy requires e_H^T N r <= 0.
    storage_kind: StorageKind = "unspecified"
    # Executable Poisson-field declaration (constant_field / state_field): the callable
    # J(x) on the declared state basis, the (m, n) domain sample points, and the Jacobi
    # residual tolerance. matrix_snapshot and zero_field ignore them. Without an
    # explicit jacobi_tol the tolerance is read from ParameterManager key
    # validation.iphs.jacobi_tolerance at construction; there is no Python default.
    poisson_field: Optional[Callable[[Tensor], Tensor]] = field(default=None, compare=False, repr=False)
    poisson_domain: Optional[Tensor] = field(default=None, compare=False, repr=False)
    jacobi_tol: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate_declaration_fields()
        if self.R is None:
            object.__setattr__(self, "R", torch.zeros_like(self.J))
        self._validate_tensor_contracts()
        object.__setattr__(self, "_poisson_field_audit", None)
        for field_name in _STRUCTURE_TOLERANCE_FIELDS:
            object.__setattr__(
                self, field_name, _resolve_iphs_tolerance(field_name, getattr(self, field_name))
            )
        self._validate_tolerances()
        self._evaluate_poisson_matrix()
        if (
            isinstance(self.contribution_count, bool)
            or not isinstance(self.contribution_count, int)
            or self.contribution_count < 0
        ):
            raise ValueError("contribution_count must be a non-negative integer.")
        if any(not isinstance(source, str) or not source.strip() for source in self.contribution_sources):
            raise ValueError("contribution_sources must contain non-empty strings.")

    def _validate_declaration_fields(self) -> None:
        """Check the descriptive declaration: storage kind, name, state basis, functionals,
        formulation and reversible-operator state-dependence claim."""

        if self.storage_kind not in ("unspecified", "total_energy", "free_energy"):
            raise ValueError(
                "storage_kind must be 'unspecified', 'total_energy', or 'free_energy'."
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        if not self.state_vars:
            raise ValueError("state_vars must not be empty.")
        if len(set(self.state_vars)) != len(self.state_vars) or any(
            not isinstance(item, str) or not item.strip() for item in self.state_vars
        ):
            raise ValueError("state_vars must contain unique non-empty names.")
        for field_name in ("energy_functional", "entropy_functional"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.formulation not in {"unspecified", "dissipative_ph", "generic"}:
            raise ValueError(
                "formulation must be 'unspecified', 'dissipative_ph', or 'generic'."
            )
        if self.reversible_operator_state_dependence not in {
            "unspecified",
            "constant",
            "state_dependent",
        }:
            raise ValueError(
                "reversible_operator_state_dependence must be 'unspecified', "
                "'constant', or 'state_dependent'."
            )
        if not isinstance(self.jacobi_identity_evidence, str):
            raise ValueError("jacobi_identity_evidence must be a string.")

    def _validate_tolerances(self) -> None:
        """Check the resolved structural tolerances (finite, non-negative) and the Jacobi
        tolerance (finite, positive)."""

        for field_name in ("skew_tol", "psd_tol", "degeneracy_tol"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a finite non-negative number.")
            scalar = torch.as_tensor(float(value), dtype=torch.float64)
            if not bool(torch.isfinite(scalar).item()) or float(value) < 0.0:
                raise ValueError(f"{field_name} must be a finite non-negative number.")
        if (
            isinstance(self.jacobi_tol, bool)
            or not isinstance(self.jacobi_tol, (int, float))
            or not math.isfinite(float(self.jacobi_tol))
            or not float(self.jacobi_tol) > 0.0
        ):
            raise ValueError("jacobi_tol must be a finite positive number.")

    def _validate_tensor_contracts(self) -> None:
        """Revalidate mutable tensor contents before structural or runtime use."""

        if not isinstance(self.reaction_representation, str) or self.reaction_representation not in (
            "independent", "absorbed",
        ):
            raise ValueError("reaction_representation must be 'independent' or 'absorbed'.")
        if any(not isinstance(port, PortSpecIPHS) for port in self.ports):
            raise TypeError("ports must contain only PortSpecIPHS instances.")
        if not isinstance(self.J, Tensor) or not isinstance(self.R, Tensor) or not isinstance(
            self.M,
            Tensor,
        ):
            raise TypeError("J, R, and M must be torch.Tensor instances.")
        if self.J.dtype not in _SUPPORTED_IPHS_DTYPES:
            raise ValueError(
                "J, R, and M must use a real floating dtype: float32 or float64."
            )
        if self.R.dtype != self.J.dtype or self.M.dtype != self.J.dtype:
            raise ValueError("J, R, and M must use the same dtype and device.")
        if self.J.device != self.R.device or self.J.device != self.M.device:
            raise ValueError("J, R, and M must use the same dtype and device.")
        if self.J.ndim != 2 or self.R.ndim != 2 or self.M.ndim != 2:
            raise ValueError("J, R, and M must be rank-2 tensors.")
        expected = (len(self.state_vars), len(self.state_vars))
        for field_name in ("J", "R", "M"):
            value = getattr(self, field_name)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{field_name} shape {tuple(value.shape)} does not match {expected}."
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("J, R, and M must be finite.")
        for field_name in ("G", "stoichiometric_N"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Tensor) or value.ndim != 2:
                raise ValueError(f"{field_name} must be a rank-2 torch.Tensor when provided.")
            if int(value.shape[0]) != len(self.state_vars):
                raise ValueError(
                    f"{field_name} row count must match the state basis length."
                )
            if value.dtype != self.J.dtype or value.device != self.J.device:
                raise ValueError(f"{field_name} must use the J/R/M dtype and device.")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{field_name} must be finite.")
        if self.G is None:
            if self.ports:
                raise ValueError("Typed ports require a declared G matrix.")
        elif self.ports:
            if _port_dof_count(self.ports) != int(self.G.shape[1]):
                raise ValueError(
                    "Typed port degrees of freedom must match the declared G columns."
                )
            roles = tuple(port.role for port in self.ports)
            if any(role == "unspecified" for role in roles) and any(
                role != "unspecified" for role in roles
            ):
                raise ValueError(
                    "Port roles cannot mix explicit and unspecified declarations."
                )
        for field_name in ("energy_gradient", "entropy_gradient"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(f"{field_name} must be a rank-1 torch.Tensor when provided.")
            if int(value.shape[0]) != len(self.state_vars):
                raise ValueError(f"{field_name} length must match the state basis length.")
            if value.dtype != self.J.dtype or value.device != self.J.device:
                raise ValueError(f"{field_name} must use the J/R/M dtype and device.")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{field_name} must be finite.")
        if self.reaction_representation == "absorbed":
            if not self._has_nonzero(self.stoichiometric_N, 0.0):
                raise ValueError("absorbed reaction requires a nonzero stoichiometric_N declaration.")
            if self.formulation == "unspecified":
                raise ValueError("absorbed reaction requires an explicit formulation.")

    @property
    def internal_port_indices(self) -> tuple[int, ...]:
        """Return G columns eliminated through declared internal feedback."""

        return _port_column_indices(self.ports, role="internal")

    @property
    def external_port_indices(self) -> tuple[int, ...]:
        """Return G columns that accept external runtime inputs."""

        return _port_column_indices(self.ports, role="external")

    @property
    def port_partition_resolved(self) -> bool:
        """Return whether every nonempty G column has one explicit typed role."""

        if self.G is None or int(self.G.shape[1]) == 0:
            return True
        return (
            bool(self.ports)
            and _port_dof_count(self.ports) == int(self.G.shape[1])
            and all(port.role in {"internal", "external"} for port in self.ports)
        )

    @staticmethod
    def _has_nonzero(value: Optional[Tensor], tolerance: float) -> bool:
        if value is None or value.numel() == 0:
            return False
        return bool(torch.max(torch.abs(value)).detach().cpu().item() > tolerance)

    def _evaluate_poisson_matrix(self, state: Optional[Tensor] = None) -> Tensor:
        """Evaluate the selected reversible operator, not a descriptive claim.

        References: Grmela and Ottinger (1997), GENERIC formalism; Marsden and
          Ratiu (1999), Poisson manifolds.
        Mathematical expression: zero_field defines J(x)=0 for every x;
          constant_field defines J(x)=J for every x so dJ_ij/dx_k=0 and the Jacobi
          identity holds identically; state_field evaluates the declared callable
          J(x) and requires J(x)^T=-J(x) and sum_l(J_il d_l J_jk + cyclic)=0 at every
          declared domain sample (poisson_jacobi_residual) within jacobi_tol.
        Parameter meanings: J supplies the declared basis, dtype/device, the exact
          consistency constraint in zero_field mode, and the reference-point value
          in constant_field/state_field mode; poisson_field is the callable J(x);
          poisson_domain holds the (m, n) sample points; state is the runtime point.
        Units: the existing declared coordinate/effort basis is retained.
        Domain/constraints: finite declared state dimension; any nonzero sample
          contradicts zero_field; any sample differing from J contradicts
          constant_field; matrix_snapshot makes no field claim and stays a sample.
          state_field evaluates J(x) at the runtime state on the autograd path, so
          the drift derivative contains dJ/dx e_E and the sensitivity to any
          parameter the field closes over; the sample audit reads detached values.
          Without a runtime state the declared reference-point J is returned.
        Sign convention: reversible drift is the generated J(x) times e_E.
        """
        if self.poisson_operator == "matrix_snapshot":
            return self.J
        if self.poisson_operator == "zero_field":
            if self.reversible_operator_state_dependence == "state_dependent":
                raise ValueError("zero_field cannot declare a state-dependent reversible operator.")
            if bool(torch.any(self.J != 0).item()):
                raise ValueError("zero_field requires an exactly zero declared J matrix.")
            dimension = len(self.state_vars)
            return torch.zeros((dimension, dimension), dtype=self.J.dtype, device=self.J.device)
        if self.poisson_operator == "constant_field":
            if self.reversible_operator_state_dependence == "state_dependent":
                raise ValueError("constant_field cannot declare a state-dependent reversible operator.")
            self._verify_poisson_field_domain()
            if state is not None and self.poisson_field is not None:
                self._require_field_equals_declared(self._evaluate_field_at(state), "runtime state")
            return self.J
        if self.poisson_operator == "state_field":
            if self.poisson_field is None or self.poisson_domain is None:
                raise ValueError("state_field requires poisson_field and poisson_domain.")
            if self.reversible_operator_state_dependence != "state_dependent":
                raise ValueError(
                    "state_field requires reversible_operator_state_dependence='state_dependent'."
                )
            self._verify_poisson_field_domain()
            if state is None:
                return self.J
            return self._evaluate_field_at(state)
        raise ValueError(
            "poisson_operator must be 'matrix_snapshot', 'zero_field', 'constant_field', or 'state_field'."
        )

    def _evaluate_field_at(self, point: Tensor) -> Tensor:
        """Evaluate poisson_field at a point and enforce shape, dtype, finiteness and skewness.

        The returned matrix keeps the autograd graph of the executed map: a runtime
        state that requires grad, and any parameter tensor the callable closes over,
        propagate dJ/dx and dJ/dtheta into the drift. The contract checks read
        detached values only; audit callers pass detached domain sample points.
        """
        if self.poisson_field is None:
            raise ValueError("poisson_field is not declared.")
        dimension = len(self.state_vars)
        if not isinstance(point, Tensor) or point.ndim != 1 or int(point.shape[0]) != dimension:
            raise ValueError(f"poisson_field point must be a rank-1 tensor of length {dimension}.")
        matrix = self.poisson_field(point.to(dtype=self.J.dtype, device=self.J.device))
        if not isinstance(matrix, Tensor) or tuple(matrix.shape) != (dimension, dimension):
            raise ValueError("poisson_field must return an (n, n) tensor on the declared state basis.")
        if matrix.dtype != self.J.dtype or matrix.device != self.J.device:
            raise ValueError("poisson_field must return the structure dtype and device.")
        values = matrix.detach()
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("poisson_field returned a non-finite matrix.")
        skew_defect = float(torch.max(torch.abs(values + values.transpose(0, 1))).item())
        if skew_defect > float(self.skew_tol):
            raise ValueError(
                f"poisson_field is not skew-symmetric at the sampled point; defect = {skew_defect}."
            )
        return matrix

    def _require_field_equals_declared(self, matrix: Tensor, label: str) -> None:
        defect = float(torch.max(torch.abs(matrix.detach() - self.J.detach())).item())
        if defect > float(self.skew_tol):
            raise ValueError(
                f"constant_field requires poisson_field to equal the declared J at the {label}; "
                f"max difference = {defect}."
            )

    def _verify_poisson_field_domain(self) -> None:
        """Verify constancy (constant_field) or the Jacobi identity (state_field) on the domain."""
        cached = getattr(self, "_poisson_field_audit", None)
        if cached is not None:
            return
        audit: dict[str, float | int | str] = {}
        if self.poisson_operator == "constant_field":
            sample_count = 0
            if self.poisson_field is not None:
                if self.poisson_domain is None:
                    raise ValueError("constant_field with a poisson_field requires poisson_domain samples.")
                self._validate_domain_tensor()
                for index in range(int(self.poisson_domain.shape[0])):
                    self._require_field_equals_declared(
                        self._evaluate_field_at(self.poisson_domain[index].detach()),
                        f"domain sample {index}",
                    )
                    sample_count += 1
            audit.update(
                jacobi_identity_basis="constant_field_exact",
                jacobi_sample_count=sample_count,
                jacobi_max_residual=0.0,
                jacobi_tolerance=float(self.jacobi_tol),
            )
        elif self.poisson_operator == "state_field":
            self._validate_domain_tensor()
            self._require_field_equals_declared(
                self._evaluate_field_at(self.poisson_domain[0].detach()), "first domain sample"
            )
            max_residual = 0.0
            for index in range(int(self.poisson_domain.shape[0])):
                point = self.poisson_domain[index].detach().to(dtype=self.J.dtype, device=self.J.device)
                self._evaluate_field_at(point)
                residual = poisson_jacobi_residual(self.poisson_field, point)
                value = float(torch.max(torch.abs(residual)).item())
                if not torch.isfinite(residual).all():
                    raise ValueError(f"Jacobi residual is not finite at domain sample {index}.")
                if value > float(self.jacobi_tol):
                    raise ValueError(
                        f"poisson_field violates the Jacobi identity on the declared domain; "
                        f"max residual = {value} at sample {index} exceeds jacobi_tol = {self.jacobi_tol}."
                    )
                max_residual = max(max_residual, value)
            audit.update(
                jacobi_identity_basis="sampled_domain_jacobi_residual",
                jacobi_sample_count=int(self.poisson_domain.shape[0]),
                jacobi_max_residual=max_residual,
                jacobi_tolerance=float(self.jacobi_tol),
            )
        object.__setattr__(self, "_poisson_field_audit", audit)

    def _validate_domain_tensor(self) -> None:
        domain = self.poisson_domain
        dimension = len(self.state_vars)
        if not isinstance(domain, Tensor) or domain.ndim != 2 or int(domain.shape[1]) != dimension:
            raise ValueError(f"poisson_domain must be an (m, {dimension}) tensor of sample points.")
        if int(domain.shape[0]) < 1:
            raise ValueError("poisson_domain must contain at least one sample point.")
        if not domain.is_floating_point() or not bool(torch.isfinite(domain).all().item()):
            raise ValueError("poisson_domain must be finite floating point.")

    def _jacobi_identity_audit(self) -> dict[str, float | bool | int | str]:
        """Return the executable Jacobi basis of the selected Poisson operator kind."""
        if self.poisson_operator == "zero_field":
            return {"jacobi_identity_verified": True, "jacobi_identity_basis": "canonical_zero_field"}
        if self.poisson_operator in {"constant_field", "state_field"}:
            self._verify_poisson_field_domain()
            audit = dict(getattr(self, "_poisson_field_audit"))
            audit["jacobi_identity_verified"] = True
            return audit
        return {"jacobi_identity_verified": False, "jacobi_identity_basis": "unimplemented"}

    def _runtime_point_jacobi_audit(
        self,
        poisson_matrix: Tensor,
        state: Optional[Tensor],
    ) -> dict[str, float | str]:
        """Evaluate the state_field Poisson conditions at the point the drift executed.

        References:
          - Grmela and Ottinger (1997), GENERIC formalism, Poisson bracket requirements.
          - Marsden and Ratiu (1999), Introduction to Mechanics and Symmetry, Ch. 10.
        Mathematical expression:
          skew defect s(x*) = max_ij |J_ij(x*) + J_ji(x*)|;
          Jacobi residual r(x*) = max_ijk |sum_l (J_il d_l J_jk + J_jl d_l J_ki
            + J_kl d_l J_ij)(x*)| (poisson_jacobi_residual), with x* the bound
          runtime state, or the first declared sample when the point binds
          explicit efforts and the drift used the reference-point J.
        Parameter meanings:
          poisson_matrix is J(x*) as executed by the drift; skew_tol and jacobi_tol
          are the declared structure tolerances.
        Units:
          s carries the units of J; r carries the units of J times J per state unit;
          both are compared on the declared normalized coordinate basis.
        Domain/constraints:
          finite x*; s <= skew_tol and r <= jacobi_tol, otherwise the runtime point
          is rejected. Together with the declared samples this is finite-point
          evidence (samples plus the evaluated runtime point), not a continuum proof.
        Sign convention:
          not applicable; both quantities are algebraic identity defects.
        """

        if self.poisson_domain is None or self.poisson_field is None:
            raise ValueError("state_field requires poisson_field and poisson_domain.")
        if state is None:
            point = self.poisson_domain[0].detach().to(dtype=self.J.dtype, device=self.J.device)
            point_label = "declared_reference_sample"
        else:
            point = state.detach()
            point_label = "bound_state"
        values = poisson_matrix.detach()
        skew_defect = float(torch.max(torch.abs(values + values.transpose(0, 1))).item())
        if skew_defect > float(self.skew_tol):
            raise ValueError(
                f"IPHS structure '{self.name}' state_field is not skew-symmetric at the "
                f"runtime point; defect = {skew_defect}."
            )
        residual = poisson_jacobi_residual(self.poisson_field, point)
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError(
                f"IPHS structure '{self.name}' state_field Jacobi residual is not finite at "
                "the runtime point."
            )
        residual_value = float(torch.max(torch.abs(residual)).item())
        if residual_value > float(self.jacobi_tol):
            raise ValueError(
                f"IPHS structure '{self.name}' state_field violates the Jacobi identity at "
                f"the runtime point; max residual = {residual_value} exceeds jacobi_tol = "
                f"{self.jacobi_tol}."
            )
        return {
            "runtime_point_skew_residual": skew_defect,
            "runtime_point_jacobi_residual": residual_value,
            "runtime_point_jacobi_point": point_label,
        }

    def _validate_runtime_vector(
        self,
        value: Tensor,
        *,
        name: str,
        length: int,
    ) -> None:
        if not isinstance(value, Tensor) or value.ndim != 1 or int(value.shape[0]) != length:
            raise ValueError(f"{name} must be a rank-1 tensor of length {length}.")
        if value.dtype != self.J.dtype or value.device != self.J.device:
            raise ValueError(f"{name} must use the structure dtype and device.")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite.")

    @staticmethod
    def _require_finite_derived(name: str, value: Tensor) -> None:
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite after canonical evaluation.")

    def _external_port_signs(self) -> Tensor:
        signs: list[float] = []
        for port in self.ports:
            if port.role != "external":
                continue
            sign = 1.0 if port.positive_power_convention == "into_system" else -1.0
            signs.extend([sign] * port.n_dofs)
        return torch.tensor(signs, dtype=self.J.dtype, device=self.J.device)

    def _evaluate_canonical_terms(
        self,
        *,
        state: Optional[Tensor] = None,
        energy_value: Optional[Tensor] = None,
        entropy_value: Optional[Tensor] = None,
        energy_effort: Optional[Tensor] = None,
        entropy_effort: Optional[Tensor] = None,
        port_input: Optional[Tensor] = None,
        reaction_rates: Optional[Tensor] = None,
        structure_validated: bool = False,
    ) -> dict[str, Tensor]:
        """Evaluate all canonical terms once for drift and point audits."""

        if not structure_validated:
            self.validate()
        if self.formulation == "unspecified":
            raise ValueError(
                "canonical_drift requires an explicit formulation: "
                "'dissipative_ph' or 'generic'."
            )
        resistance = self.R
        if resistance is None:
            raise RuntimeError("IPHSStructureSpec internal R initialization failed.")
        e_E = energy_effort if energy_effort is not None else self.energy_gradient
        e_S = entropy_effort if entropy_effort is not None else self.entropy_gradient
        if state is not None or energy_value is not None or entropy_value is not None:
            if state is None or energy_value is None:
                raise ValueError("state and energy_value must be supplied together.")
            if any(value is not None for value in (
                energy_effort, entropy_effort, self.energy_gradient, self.entropy_gradient,
            )):
                raise ValueError("Potential values and precomputed efforts are mutually exclusive.")
            self._validate_runtime_vector(state, name="state", length=len(self.state_vars))
            if not state.requires_grad:
                raise ValueError("Potential binding requires state.requires_grad=True.")
            if self.formulation == "generic" and entropy_value is None:
                raise ValueError("GENERIC potential binding requires entropy_value.")
            if self.formulation == "dissipative_ph" and entropy_value is not None:
                raise ValueError("dissipative_ph potential binding does not accept entropy_value.")
            # References: van der Schaft (2022); Grmela and Ottinger (1997).
            # Mathematical expression: e_H=dH/dx (dPH); e_E=dE/dx, e_S=dS/dx
            #   (GENERIC), evaluated on the same runtime state x.
            # Parameter meanings: scalar potentials are supplied as autograd results;
            #   their explicit parameter dependencies remain in the derivative graph.
            # Units: each effort has its potential units divided by its state units.
            # Domain/constraints: finite real scalar values on a connected state graph;
            #   no graph is invented for detached values or unrelated state tensors.
            # Sign convention: canonical R/M terms below supply the physical sign.
            efforts: dict[str, Tensor] = {}
            for name, potential in (("energy_value", energy_value), ("entropy_value", entropy_value)):
                if potential is None:
                    continue
                if not isinstance(potential, Tensor) or potential.ndim != 0:
                    raise ValueError(f"{name} must be a scalar tensor.")
                if potential.dtype != state.dtype or potential.device != state.device:
                    raise ValueError(f"{name} must use the state dtype and device.")
                if not bool(torch.isfinite(potential).item()):
                    raise ValueError(f"{name} must be finite.")
                if not potential.requires_grad:
                    raise ValueError(f"{name} must retain its autograd connection to state.")
                efforts[name] = torch.autograd.grad(
                    potential, state, create_graph=torch.is_grad_enabled(),
                    retain_graph=True, allow_unused=False,
                )[0]
            e_E = efforts["energy_value"]
            e_S = efforts.get("entropy_value")
        if e_E is None:
            raise ValueError("canonical_drift requires energy_effort or energy_gradient.")
        self._validate_runtime_vector(
            e_E,
            name="energy_effort",
            length=len(self.state_vars),
        )
        if self.formulation == "dissipative_ph":
            if entropy_effort is not None:
                raise ValueError(
                    "dissipative_ph canonical_drift does not accept entropy_effort."
                )
            if bool(torch.any(self.M != 0).item()):
                raise ValueError(
                    "dissipative_ph canonical_drift cannot use GENERIC mobility M."
                )
        else:
            if bool(torch.any(resistance != 0).item()):
                raise ValueError("generic canonical_drift cannot use dPH resistance R.")
            if e_S is None:
                raise ValueError(
                    "canonical_drift requires entropy_effort or entropy_gradient."
                )
            self._validate_runtime_vector(
                e_S,
                name="entropy_effort",
                length=len(self.state_vars),
            )

        # References:
        #   - van der Schaft (2022), port-Hamiltonian systems.
        #   - Grmela and Oettinger (1997), GENERIC formalism.
        # Mathematical expressions:
        #   dPH: x_dot = J e_H - R e_H + G_ext u_into + N r.
        #   GENERIC: x_dot = J e_E + M e_S + G_ext u_into + N r.
        #   The final N r term is included only for independent reactions.
        # Parameter meanings, units, domain, and sign convention:
        #   e_E/e_S are real covectors on the declared state basis; G columns
        #   use into-system-normalized flow, so effort times flow has power units.
        poisson_matrix = self._evaluate_poisson_matrix(state=state)
        reversible_rate = poisson_matrix @ e_E
        self._require_finite_derived("reversible_rate", reversible_rate)
        zero_state_rate = torch.zeros_like(reversible_rate)
        dissipative_rate = zero_state_rate
        entropy_rate = zero_state_rate
        if self.formulation == "dissipative_ph":
            dissipative_rate = resistance @ e_E
            self._require_finite_derived("dissipative_rate", dissipative_rate)
            drift = reversible_rate - dissipative_rate
        else:
            if e_S is None:
                raise RuntimeError("GENERIC entropy effort initialization failed.")
            entropy_rate = self.M @ e_S
            self._require_finite_derived("entropy_rate", entropy_rate)
            drift = reversible_rate + entropy_rate

        external_indices = self.external_port_indices
        port_rate = zero_state_rate
        port_output = torch.empty(0, dtype=self.J.dtype, device=self.J.device)
        port_input_into = torch.empty(0, dtype=self.J.dtype, device=self.J.device)
        if self.G is not None and int(self.G.shape[1]) > 0 and not self.port_partition_resolved:
            raise ValueError(
                "canonical_drift requires an explicit port role partition for every G column."
            )
        if self.G is not None and external_indices:
            if port_input is None:
                raise ValueError("canonical_drift requires port_input for external G columns.")
            self._validate_runtime_vector(
                port_input,
                name="port_input",
                length=len(external_indices),
            )
            external_index = torch.tensor(
                external_indices,
                device=self.G.device,
                dtype=torch.long,
            )
            G_external = torch.index_select(self.G, 1, external_index)
            port_input_into = port_input * self._external_port_signs()
            self._require_finite_derived("port_input_into", port_input_into)
            port_rate = G_external @ port_input_into
            self._require_finite_derived("port_rate", port_rate)
            port_output = G_external.transpose(-1, -2) @ e_E
            self._require_finite_derived("port_output", port_output)
            drift = drift + port_rate
        elif port_input is not None:
            raise ValueError("port_input was provided but no external G column is declared.")

        reaction_rate = zero_state_rate
        if self.stoichiometric_N is not None:
            if reaction_rates is None:
                raise ValueError(
                    "canonical_drift requires reaction_rates when stoichiometric_N is declared."
                )
            self._validate_runtime_vector(
                reaction_rates,
                name="reaction_rates",
                length=int(self.stoichiometric_N.shape[1]),
            )
            reaction_rate = self.stoichiometric_N @ reaction_rates
            self._require_finite_derived("reaction_rate", reaction_rate)
            if self.reaction_representation == "independent":
                drift = drift + reaction_rate
            else:
                # References: Grmela and Ottinger (1997); van der Schaft (2022).
                # Mathematical expression: N r = M e_S (GENERIC) or -R e_H
                #   (dPH); an absorbed reaction is not added to the drift twice.
                # Parameter meanings: N maps extents to the declared state basis;
                #   the entire irreversible operator must represent these reactions.
                # Units: both sides share each coordinate's state/time basis.
                # Domain/constraints: pointwise equality with componentwise relative
                #   error scaled by absolute matrix products, without an absolute floor.
                # Sign convention: dPH resistance enters with a minus sign.
                if self.formulation == "generic":
                    if e_S is None:
                        raise RuntimeError("GENERIC entropy effort initialization failed.")
                    represented_rate = entropy_rate
                    represented_scale = torch.abs(self.M) @ torch.abs(e_S)
                else:
                    represented_rate = -dissipative_rate
                    represented_scale = torch.abs(resistance) @ torch.abs(e_E)
                reaction_scale = torch.abs(self.stoichiometric_N) @ torch.abs(reaction_rates)
                scale = torch.maximum(represented_scale, reaction_scale)
                self._require_finite_derived("absorbed_reaction_scale", scale)
                denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
                residual = torch.abs(represented_rate - reaction_rate) / denominator
                self._require_finite_derived("absorbed_reaction_residual", residual)
                tolerance = _reduction_relative_tolerance_floor(
                    dtype=self.J.dtype,
                    term_count=max(len(self.state_vars), int(self.stoichiometric_N.shape[1])),
                )
                if tolerance >= 1.0:
                    raise ValueError("absorbed reaction alignment tolerance must be below one.")
                if bool(torch.any(residual > tolerance).item()):
                    raise ValueError("absorbed irreversible rate must equal declared N r.")
        elif reaction_rates is not None:
            raise ValueError(
                "reaction_rates were provided but stoichiometric_N is not declared."
            )
        self._require_finite_derived("canonical_drift", drift)
        port_power = torch.zeros((), dtype=self.J.dtype, device=self.J.device)
        if port_output.numel() > 0:
            port_power = torch.dot(port_output, port_input_into)
            self._require_finite_derived("port_power", port_power)
        terms: dict[str, Tensor] = {
            "drift": drift,
            "poisson_matrix": poisson_matrix,
            "reversible_rate": reversible_rate,
            "dissipative_rate": dissipative_rate,
            "entropy_rate": entropy_rate,
            "port_rate": port_rate,
            "port_output": port_output,
            "port_input_into": port_input_into,
            "port_power": port_power,
            "reaction_rate": reaction_rate,
            "energy_effort": e_E,
        }
        if e_S is not None:
            terms["entropy_effort"] = e_S
        return terms

    def _audit_generic_runtime_point(
        self,
        *,
        state: Optional[Tensor],
        energy_value: Optional[Tensor],
        entropy_value: Optional[Tensor],
        energy_effort: Optional[Tensor],
        entropy_effort: Optional[Tensor],
        port_input: Optional[Tensor],
        reaction_rates: Optional[Tensor],
    ) -> dict[str, float | bool | str]:
        """Verify one GENERIC runtime point with the actual process potentials.

        References:
          - Grmela and Ottinger (1997), Phys. Rev. E 56, 6620: GENERIC
            degeneracy and entropy production.
          - Ottinger (2005), Beyond Equilibrium Thermodynamics, ch. 1-2.
          - Mielke (2011), Nonlinearity 24, 1329: reaction Onsager operators.
          - Kondepudi and Prigogine (1998), ch. 9: reaction affinity closure.
        Mathematical expression:
          e_E = dE/dx, e_S = dS/dx on the bound state x;
          L e_S = 0, M e_E = 0, sigma = e_S^T M e_S >= 0;
          closed reaction: e_E^T N r = 0 and e_S^T N r >= 0;
          absorbed representation: M e_S = N r (checked by the canonical terms).
        Parameter meanings:
          L (stored in J) reversible operator, M friction operator, N
          stoichiometry, r reaction rates, degeneracy_tol audit tolerance.
        Units:
          e_E energy per state unit, e_S entropy per state unit, N r state
          per time, sigma entropy per time, e_E^T N r energy per time.
        Domain/constraints:
          formulation generic, R = 0, finite bound potentials with autograd
          connection, poisson_operator zero_field, constant_field, or
          state_field for an executable Jacobi basis; state_field is checked for
          skew symmetry and the Jacobi residual at the declared samples and at the
          bound runtime state (finite-point evidence, not a continuum proof);
          matrix_snapshot and any port column remain [N] for the GENERIC point
          audit.
        Sign convention:
          positive sigma and positive reaction entropy rate are dissipation;
          a nonzero reaction energy rate is an energy leak of a closed reaction.
        """

        if self.formulation != "generic":
            raise ValueError(
                "Level 5 runtime-point verification requires formulation='generic'."
            )
        if self.G is not None and int(self.G.shape[1]) > 0:
            raise ValueError(
                "Level 5 GENERIC port power linkage remains [N]; the runtime point "
                "cannot include declared port columns."
            )
        if self._has_nonzero(self.stoichiometric_N, 0.0) and reaction_rates is None:
            raise ValueError(
                "Level 5 runtime point requires reaction_rates for the declared "
                "stoichiometric_N."
            )
        terms = self._evaluate_canonical_terms(
            state=state,
            energy_value=energy_value,
            entropy_value=entropy_value,
            energy_effort=energy_effort,
            entropy_effort=entropy_effort,
            port_input=port_input,
            reaction_rates=reaction_rates,
            structure_validated=True,
        )
        e_E = terms["energy_effort"]
        e_S = terms.get("entropy_effort")
        if e_S is None:
            raise RuntimeError("GENERIC runtime point did not bind an entropy effort.")
        poisson_matrix = terms["poisson_matrix"]
        tolerance = float(self.degeneracy_tol)
        audit: dict[str, float | bool | str] = {}

        residual_norm = float(torch.linalg.norm(poisson_matrix @ e_S).detach().cpu().item())
        audit["J_entropy_gradient_residual"] = residual_norm
        if residual_norm > tolerance:
            raise ValueError(
                f"IPHS structure '{self.name}' violates J e_S = 0 at the runtime point; "
                f"residual = {residual_norm}."
            )
        residual_norm = float(torch.linalg.norm(self.M @ e_E).detach().cpu().item())
        audit["M_energy_gradient_residual"] = residual_norm
        if residual_norm > tolerance:
            raise ValueError(
                f"IPHS structure '{self.name}' violates M e_E = 0 at the runtime point; "
                f"residual = {residual_norm}."
            )
        entropy_production = torch.dot(e_S, self.M @ e_S)
        self._require_finite_derived("entropy_production", entropy_production)
        entropy_production_value = float(entropy_production.detach().cpu().item())
        audit["entropy_production"] = entropy_production_value
        if entropy_production_value < -tolerance:
            raise ValueError(
                f"IPHS structure '{self.name}' has negative entropy production at the "
                f"runtime point; value = {entropy_production_value}."
            )

        if self._has_nonzero(self.stoichiometric_N, 0.0):
            reaction_rate = terms["reaction_rate"]
            energy_rate = torch.dot(e_E, reaction_rate)
            entropy_rate = torch.dot(e_S, reaction_rate)
            self._require_finite_derived("reaction_energy_rate", energy_rate)
            self._require_finite_derived("reaction_entropy_rate", entropy_rate)
            energy_scale = float(
                (torch.linalg.norm(e_E) * torch.linalg.norm(reaction_rate)).detach().cpu().item()
            )
            energy_rate_value = float(energy_rate.detach().cpu().item())
            entropy_rate_value = float(entropy_rate.detach().cpu().item())
            audit["reaction_energy_rate"] = energy_rate_value
            audit["reaction_entropy_rate"] = entropy_rate_value
            # References: Grmela and Ottinger (1997) (closed reaction energy
            #   pairing e_E^T N r = 0); Hairer, Norsett and Wanner (1993), Sect.
            #   II.4 (mixed absolute/relative error scale).
            # Mathematical expression: |e_E^T N r| <= tol * max(p_floor, ||e_E|| ||N r||).
            # Parameter meanings: tol = degeneracy_tol; p_floor =
            #   validation.iphs.reaction_power_scale_floor (RELIC convention 1.0).
            # Units: e_E^T N r, ||e_E|| ||N r|| and p_floor in power units of the
            #   declared normalized basis; tol dimensionless.
            # Domain/constraints: relative above the floor, absolute (tol * p_floor)
            #   below it; the floor is read from YAML, not a code literal.
            # Sign convention: only the magnitude of the pairing is tested.
            power_scale_floor = float(_resolve_iphs_tolerance("reaction_power_scale_floor", None))
            if abs(energy_rate_value) > tolerance * max(power_scale_floor, energy_scale):
                raise ValueError(
                    f"IPHS structure '{self.name}' violates the closed reaction energy "
                    f"closure e_E^T N r = 0; value = {energy_rate_value}."
                )
            if entropy_rate_value < -tolerance:
                raise ValueError(
                    f"IPHS structure '{self.name}' has a negative reaction entropy rate "
                    f"e_S^T N r; value = {entropy_rate_value}."
                )

        if self.poisson_operator == "matrix_snapshot":
            audit["jacobi_identity_verified"] = False
            audit["jacobi_identity_basis"] = "unimplemented"
            raise ValueError(
                "Level 5 executable Jacobi identity verification remains [N] for a "
                "matrix_snapshot; the tensor-only reversible-operator payload cannot prove "
                "a global Poisson identity. Declare zero_field, constant_field, or state_field."
            )
        audit.update(self._jacobi_identity_audit())
        if self.poisson_operator == "state_field":
            audit.update(self._runtime_point_jacobi_audit(poisson_matrix, state))
        audit.update(
            evidence_scope="runtime_point",
            process_level_status="runtime_point_verified",
            runtime_verified_level=5,
        )
        return audit

    def _audit_dph_runtime_point(
        self,
        *,
        energy_effort: Optional[Tensor],
        port_input: Optional[Tensor],
        state: Optional[Tensor] = None,
        energy_value: Optional[Tensor] = None,
        reaction_rates: Optional[Tensor] = None,
        runtime_bound: bool = False,
    ) -> dict[str, float | bool | str]:
        """Audit one dPH runtime point.

        With ``state`` and ``energy_value`` the effort e_H = dH/dx is derived on
        the actual process state graph and, when ``runtime_bound`` is set, the
        audit reports ``runtime_verified_level=4``; an explicit ``energy_effort``
        only yields the single-point pre-gate.  A declared reaction is admitted
        only on the bound runtime point with an explicit ``storage_kind``:

        References: van der Schaft (2022); Kondepudi and Prigogine (1998).
        Mathematical expression:
          independent: dH/dt = -e_H^T R e_H + y^T u + e_H^T N r;
          absorbed: N r = -R e_H, dH/dt = -e_H^T R e_H + y^T u;
          total_energy storage: e_H^T N r = 0; free_energy storage: e_H^T N r <= 0.
        Parameter meanings: e_H storage effort, N stoichiometry, r reaction rates.
        Units: every term carries the declared power unit of the storage.
        Domain/constraints: closed reaction without typed thermal or chemical
          ports; the pairing tolerance is degeneracy_tol times the power scale.
        Sign convention: positive dissipation, negative free-energy reaction rate.
        """

        if self.formulation != "dissipative_ph":
            raise ValueError(
                "single-runtime-point energy audit requires formulation='dissipative_ph'."
            )
        reaction_bound = runtime_bound and reaction_rates is not None
        if self.stoichiometric_N is not None and not reaction_bound:
            raise ValueError(
                "single-runtime-point dPH audit cannot include reaction N r until "
                "reaction energy closure is runtime-bound [N]."
            )
        if reaction_bound and self.storage_kind == "unspecified":
            raise ValueError(
                f"IPHS structure '{self.name}' requires storage_kind total_energy or "
                "free_energy before the dPH reaction closure linkage [N] can be verified."
            )
        if (state is None) != (energy_value is None):
            raise ValueError("state and energy_value must be supplied together.")
        if state is None and energy_effort is None:
            raise ValueError("dPH runtime point requires energy_effort or bound potentials.")
        terms = self._evaluate_canonical_terms(
            state=state,
            energy_value=energy_value,
            energy_effort=energy_effort,
            port_input=port_input,
            reaction_rates=reaction_rates,
            structure_validated=True,
        )
        energy_effort = terms["energy_effort"]
        resistance = self.R
        if resistance is None:
            raise RuntimeError("IPHSStructureSpec internal R initialization failed.")

        # References:
        #   - van der Schaft (2022), dissipative port-Hamiltonian passivity.
        # Mathematical expression:
        #   dH/dt = -e_H.T R e_H + y.T u_into and
        #   residual = dH/dt + e_H.T R e_H - y.T u_into.
        # Parameter meanings, units, domain, and sign convention:
        #   All terms share the declared power unit; degeneracy_tol is used only
        #   as a dimensionless relative multiplier of the computed power scale.
        reversible_power = torch.dot(energy_effort, terms["reversible_rate"])
        dissipation_power = torch.dot(energy_effort, terms["dissipative_rate"])
        energy_rate = torch.dot(energy_effort, terms["drift"])
        port_power = terms["port_power"]
        reaction_power = torch.dot(energy_effort, terms["reaction_rate"])
        # An absorbed reaction is already inside the dissipative rate; only an
        # independent reaction adds its own power term to the balance.
        independent_reaction_power = (
            reaction_power
            if (reaction_bound and self.reaction_representation == "independent")
            else torch.zeros_like(reaction_power)
        )
        balance_residual = (
            energy_rate + dissipation_power - port_power - independent_reaction_power
        )
        for name, value in (
            ("reversible_power", reversible_power),
            ("dissipation_power", dissipation_power),
            ("energy_rate", energy_rate),
            ("port_power", port_power),
            ("reaction_power", reaction_power),
            ("power_balance_residual", balance_residual),
        ):
            self._require_finite_derived(name, value)

        effort_norm = torch.linalg.vector_norm(energy_effort)
        rate_norm_sum = sum(
            torch.linalg.vector_norm(terms[name])
            for name in ("reversible_rate", "dissipative_rate", "port_rate")
        )
        balance_scale = (
            torch.abs(energy_rate)
            + torch.abs(dissipation_power)
            + torch.abs(port_power)
            + torch.abs(independent_reaction_power)
            + effort_norm * rate_norm_sum
        )
        self._require_finite_derived("balance_scale", balance_scale)
        n_state = len(self.state_vars)
        n_port = int(terms["port_input_into"].numel())
        operation_bound = max(1, 8 * n_state + 4 * n_port + 16)
        machine_epsilon = torch.finfo(self.J.dtype).eps
        epsilon_product = operation_bound * machine_epsilon
        if epsilon_product >= 1.0:
            raise ValueError("point-audit floating-point operation bound is invalid.")
        gamma = epsilon_product / (1.0 - epsilon_product)
        floating_tolerance = (
            gamma + float(self.degeneracy_tol)
        ) * balance_scale
        poisson_matrix = terms["poisson_matrix"]
        skew_operator = poisson_matrix + poisson_matrix.transpose(-1, -2)
        skew_tolerance = (
            0.5
            * torch.linalg.matrix_norm(skew_operator, ord=2)
            * effort_norm.square()
        )
        min_resistance_eigenvalue = torch.min(
            torch.linalg.eigvalsh(
                0.5 * (resistance + resistance.transpose(-1, -2))
            )
        )
        psd_tolerance = torch.clamp(
            -min_resistance_eigenvalue,
            min=0.0,
        ) * effort_norm.square()
        for name, value in (
            ("floating_tolerance", floating_tolerance),
            ("skew_power_tolerance", skew_tolerance),
            ("psd_power_tolerance", psd_tolerance),
        ):
            self._require_finite_derived(name, value)
        balance_tolerance = skew_tolerance + floating_tolerance
        passivity_tolerance = skew_tolerance + psd_tolerance + floating_tolerance
        if bool(torch.abs(balance_residual) > balance_tolerance):
            raise ValueError(
                f"IPHS structure '{self.name}' violates the point power-balance "
                f"identity; residual = {float(balance_residual.detach().cpu().item())}."
            )
        if bool(dissipation_power < -(psd_tolerance + floating_tolerance)):
            raise ValueError(
                f"IPHS structure '{self.name}' has negative point dissipation power."
            )
        if reaction_bound:
            reaction_scale = effort_norm * torch.linalg.vector_norm(terms["reaction_rate"])
            reaction_tolerance = (gamma + float(self.degeneracy_tol)) * torch.clamp(
                reaction_scale, min=1.0
            )
            self._require_finite_derived("reaction_tolerance", reaction_tolerance)
            if self.storage_kind == "total_energy":
                if bool(torch.abs(reaction_power) > reaction_tolerance):
                    raise ValueError(
                        f"IPHS structure '{self.name}' violates the closed reaction energy "
                        f"closure e_H^T N r = 0 for a total-energy storage; value = "
                        f"{float(reaction_power.detach().cpu().item())}."
                    )
            elif bool(reaction_power > reaction_tolerance):
                raise ValueError(
                    f"IPHS structure '{self.name}' violates the free-energy reaction pairing "
                    f"e_H^T N r <= 0; value = {float(reaction_power.detach().cpu().item())}."
                )
        if bool(energy_rate - port_power > passivity_tolerance):
            raise ValueError(
                f"IPHS structure '{self.name}' violates point passivity dH/dt <= port power."
            )
        closed_port = terms["port_input_into"].numel() == 0
        audit: dict[str, float | bool | str] = {
            "evidence_scope": "single_runtime_point",
            "closed_port": closed_port,
            "finite": True,
            "reversible_power": float(reversible_power.detach().cpu().item()),
            "dissipation_power": float(dissipation_power.detach().cpu().item()),
            "port_power_into_system": float(port_power.detach().cpu().item()),
            "power_balance_residual": float(balance_residual.detach().cpu().item()),
            "balance_scale": float(balance_scale.detach().cpu().item()),
            "balance_tolerance": float(balance_tolerance.detach().cpu().item()),
            "passivity_tolerance": float(passivity_tolerance.detach().cpu().item()),
            "process_level_status": "[N] runtime_unbound",
        }
        energy_rate_key = (
            "closed_port_energy_rate" if closed_port else "open_port_energy_rate"
        )
        audit[energy_rate_key] = float(energy_rate.detach().cpu().item())
        if reaction_bound:
            audit["reaction_energy_rate"] = float(reaction_power.detach().cpu().item())
            audit["storage_kind"] = self.storage_kind
        if runtime_bound:
            audit.update(
                evidence_scope="runtime_point",
                process_level_status="runtime_point_verified",
                runtime_verified_level=4,
            )
        return audit

    def validate(
        self,
        *,
        required_level: Optional[int] = None,
        energy_effort: Optional[Tensor] = None,
        port_input: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
        energy_value: Optional[Tensor] = None,
        entropy_value: Optional[Tensor] = None,
        entropy_effort: Optional[Tensor] = None,
        reaction_rates: Optional[Tensor] = None,
    ) -> dict[str, float | bool | str]:
        """Validate skew symmetry and dissipative positive semidefiniteness.

        A Level 5 request becomes a runtime-point verification when the actual
        process potentials are bound: either ``state`` with ``energy_value``
        and ``entropy_value`` (efforts by autograd on the same state graph) or
        explicit ``energy_effort`` and ``entropy_effort``; ``reaction_rates``
        must accompany a declared ``stoichiometric_N``.  Without that binding
        the strict levels keep their [N] rejections.
        """

        resistance = self.R
        if resistance is None:
            raise RuntimeError("IPHSStructureSpec internal R initialization failed.")
        self._validate_tensor_contracts()
        poisson_matrix = self._evaluate_poisson_matrix()
        if required_level is not None and (
            isinstance(required_level, bool)
            or not isinstance(required_level, int)
            or required_level < 1
            or required_level > 5
        ):
            raise ValueError("required_level must be an integer in 1..5 when provided.")
        if port_input is not None and energy_effort is None:
            raise ValueError(
                "point energy audit requires explicit runtime energy_effort when "
                "port_input is provided."
            )
        level5_runtime_bound = (
            required_level is not None
            and required_level >= 5
            and (
                state is not None
                or energy_value is not None
                or entropy_value is not None
                or (energy_effort is not None and entropy_effort is not None)
            )
        )
        level4_runtime_bound = (
            required_level == 4
            and state is not None
            and energy_value is not None
        )
        level4_reaction_bound = level4_runtime_bound and reaction_rates is not None
        if (
            not level5_runtime_bound
            and not level4_runtime_bound
            and any(value is not None for value in (state, energy_value, entropy_value, entropy_effort, reaction_rates))
        ):
            raise ValueError(
                "state, energy_value, entropy_value, entropy_effort, and reaction_rates "
                "are accepted only for a Level 4 or Level 5 runtime-point verification."
            )

        audit: dict[str, float | bool | str] = {
            "name": self.name,
            "formulation": self.formulation,
            "poisson_operator": self.poisson_operator,
            "reaction_representation": self.reaction_representation,
            "skew_symmetric": True,
            "dissipation_symmetric": True,
            "dissipation_psd": True,
            "min_dissipation_eigenvalue": float("nan"),
            "required_level": required_level if required_level is not None else 0,
            "runtime_contribution_count": self.contribution_count,
        }
        if self.poisson_operator == "zero_field":
            audit.update(
                jacobi_identity_verified=True,
                jacobi_identity_basis="canonical_zero_field",
                evidence_scope="zero_field_structural_pre_gate",
                process_level_status="[N] runtime_unbound",
            )
        elif self.poisson_operator in {"constant_field", "state_field"}:
            audit.update(self._jacobi_identity_audit())
            audit.update(
                evidence_scope="poisson_field_structural_pre_gate",
                process_level_status="[N] runtime_unbound",
            )
        # References:
        #   - van der Schaft (2022), dissipative port-Hamiltonian systems.
        #   - Grmela and Oettinger (1997), GENERIC formalism.
        # Mathematical expressions:
        #   dPH: x_dot = (J - R)e_H; GENERIC: x_dot = J e_E + M e_S.
        #   J + J^T = 0; R = R^T >= 0; M = M^T >= 0.
        # Parameter meanings:
        #   R acts on energy effort; M acts on entropy effort.
        skew_residual = torch.linalg.norm(poisson_matrix + poisson_matrix.transpose(-1, -2))
        self._require_finite_derived("skew_residual", skew_residual)
        skew_norm = float(skew_residual.detach().cpu().item())
        audit["skew_residual"] = skew_norm
        if skew_norm > float(self.skew_tol):
            audit["skew_symmetric"] = False
            raise ValueError(f"IPHS structure '{self.name}' has non-skew J; residual = {skew_norm}.")

        for operator_name, operator in (("R", resistance), ("M", self.M)):
            symmetry_residual = torch.linalg.norm(operator - operator.transpose(-1, -2))
            self._require_finite_derived(
                f"{operator_name}_symmetry_residual",
                symmetry_residual,
            )
            symmetry_norm = float(symmetry_residual.detach().cpu().item())
            if symmetry_norm > float(self.skew_tol):
                raise ValueError(
                    f"IPHS structure '{self.name}' has non-symmetric {operator_name}; "
                    f"residual = {symmetry_norm}."
                )
            symmetric_operator = 0.5 * (operator + operator.transpose(-1, -2))
            eigvals = torch.linalg.eigvalsh(symmetric_operator)
            self._require_finite_derived(f"{operator_name}_eigenvalues", eigvals)
            min_eig = float(torch.min(eigvals).detach().cpu().item())
            if min_eig < -float(self.psd_tol):
                raise ValueError(
                    f"IPHS structure '{self.name}' {operator_name} is not positive semidefinite; "
                    f"min eigenvalue = {min_eig}."
                )

        active_dissipator = resistance if self.formulation == "dissipative_ph" else self.M
        symmetry_residual = torch.linalg.norm(
            active_dissipator - active_dissipator.transpose(-1, -2)
        )
        self._require_finite_derived(
            "dissipation_symmetry_residual",
            symmetry_residual,
        )
        symmetry_norm = float(symmetry_residual.detach().cpu().item())
        audit["dissipation_symmetry_residual"] = symmetry_norm
        symmetric_dissipator = 0.5 * (
            active_dissipator + active_dissipator.transpose(-1, -2)
        )
        eigvals = torch.linalg.eigvalsh(symmetric_dissipator)
        self._require_finite_derived("dissipation_eigenvalues", eigvals)
        min_eig = float(torch.min(eigvals).detach().cpu().item())
        audit["min_dissipation_eigenvalue"] = min_eig

        if required_level is not None and required_level >= 4:
            if self.formulation == "unspecified":
                raise ValueError(
                    f"IPHS structure '{self.name}' requires an explicit strict formulation."
                )
            if required_level == 4 and self.formulation != "dissipative_ph":
                raise ValueError("Level 4 strict IPHS requires formulation='dissipative_ph'.")
            if required_level >= 5 and self.formulation != "generic":
                raise ValueError("Level 5 strict IPHS/GENERIC requires formulation='generic'.")
            if self.contribution_count < 1 or len(self.contribution_sources) != self.contribution_count:
                raise ValueError(
                    f"IPHS structure '{self.name}' requires runtime-linked contributions."
                )
            if (
                self._has_nonzero(self.stoichiometric_N, 0.0)
                and not level5_runtime_bound
                and not level4_reaction_bound
            ):
                raise ValueError(
                    f"IPHS structure '{self.name}' requires executable reaction closure "
                    "linkage [N] before nonzero N r can enter a Level 4/5 energy-entropy "
                    "audit; bind state, energy_value, entropy_value, and reaction_rates "
                    "for a Level 5 runtime point."
                )
            if not self._has_nonzero(poisson_matrix, self.skew_tol) and not self._has_nonzero(
                resistance, self.psd_tol
            ) and not self._has_nonzero(
                self.M, self.psd_tol
            ) and not self._has_nonzero(self.G, self.psd_tol) and not self._has_nonzero(
                self.stoichiometric_N, self.psd_tol
            ):
                raise ValueError(
                    f"IPHS structure '{self.name}' has no executable structural operator."
                )
            if required_level >= 5 and self.energy_gradient is None and not level5_runtime_bound:
                raise ValueError(
                    f"IPHS structure '{self.name}' requires energy_gradient for level 5."
                )
            has_port_columns = self.G is not None and int(self.G.shape[1]) > 0
            if has_port_columns and not self.ports:
                raise ValueError(
                    f"IPHS structure '{self.name}' requires typed ports for declared G columns."
                )
            if has_port_columns:
                if any(port.role == "unspecified" for port in self.ports):
                    raise ValueError(
                        f"IPHS structure '{self.name}' requires each port to declare an "
                        "explicit internal or external role."
                    )
                if not self.port_partition_resolved:
                    raise ValueError(
                        f"IPHS structure '{self.name}' requires a complete typed port "
                        "degree-of-freedom partition."
                    )

        if required_level == 4:
            if self._has_nonzero(self.M, self.psd_tol):
                raise ValueError(
                    f"IPHS structure '{self.name}' cannot use GENERIC mobility M in Level 4 "
                    "dissipative_ph formulation."
                )
            if level4_runtime_bound:
                if entropy_value is not None or entropy_effort is not None:
                    raise ValueError(
                        "dissipative_ph Level 4 runtime point does not accept entropy_value "
                        "or entropy_effort."
                    )
                if reaction_rates is not None and self.stoichiometric_N is None:
                    raise ValueError(
                        "reaction_rates were provided but stoichiometric_N is not declared."
                    )
                audit.update(
                    self._audit_dph_runtime_point(
                        energy_effort=None,
                        port_input=port_input,
                        state=state,
                        energy_value=energy_value,
                        reaction_rates=reaction_rates,
                        runtime_bound=True,
                    )
                )
                return audit
            if energy_effort is None:
                raise ValueError(
                    f"IPHS structure '{self.name}' requires explicit runtime energy_effort; "
                    "process runtime linkage remains [N]."
                )
            audit.update(
                self._audit_dph_runtime_point(
                    energy_effort=energy_effort,
                    port_input=port_input,
                )
            )
            raise ValueError(
                "Level 4 single-point mathematics may be audited, but process "
                "runtime linkage remains [N]."
            )

        if energy_effort is not None and not level5_runtime_bound:
            audit.update(
                self._audit_dph_runtime_point(
                    energy_effort=energy_effort,
                    port_input=port_input,
                )
            )

        if required_level is not None and required_level >= 5:
            if self._has_nonzero(resistance, 0.0):
                raise ValueError(
                    f"IPHS structure '{self.name}' cannot use dPH resistance R in Level 5 "
                    "generic formulation."
                )
            if level5_runtime_bound:
                audit.update(
                    self._audit_generic_runtime_point(
                        state=state,
                        energy_value=energy_value,
                        entropy_value=entropy_value,
                        energy_effort=energy_effort,
                        entropy_effort=entropy_effort,
                        port_input=port_input,
                        reaction_rates=reaction_rates,
                    )
                )
                return audit
            if self.entropy_gradient is None:
                raise ValueError(
                    f"IPHS structure '{self.name}' requires entropy_gradient for level 5."
                )
            energy_gradient = self.energy_gradient
            entropy_gradient = self.entropy_gradient
            if energy_gradient is None or entropy_gradient is None:
                raise RuntimeError("Level 5 effort initialization failed.")
            residual = poisson_matrix @ entropy_gradient
            residual_norm = float(torch.linalg.norm(residual).detach().cpu().item())
            audit["J_entropy_gradient_residual"] = residual_norm
            if residual_norm > float(self.degeneracy_tol):
                raise ValueError(
                    f"IPHS structure '{self.name}' violates J e_S = 0; "
                    f"residual = {residual_norm}."
                )
            residual = self.M @ energy_gradient
            residual_norm = float(torch.linalg.norm(residual).detach().cpu().item())
            audit["M_energy_gradient_residual"] = residual_norm
            if residual_norm > float(self.degeneracy_tol):
                raise ValueError(
                    f"IPHS structure '{self.name}' violates M e_E = 0; "
                    f"residual = {residual_norm}."
                )
            entropy_production = torch.dot(
                entropy_gradient,
                self.M @ entropy_gradient,
            )
            entropy_production_value = float(entropy_production.detach().cpu().item())
            audit["entropy_production"] = entropy_production_value
            if entropy_production_value < -float(self.degeneracy_tol):
                raise ValueError(
                    f"IPHS structure '{self.name}' has negative entropy production; "
                    f"value = {entropy_production_value}."
                )
            if self.poisson_operator == "zero_field":
                raise ValueError(
                    "Level 5 canonical zero-field Jacobi identity is verified, but "
                    "process runtime H/E/S effort linkage remains [N]."
                )
            elif self.poisson_operator in {"constant_field", "state_field"}:
                audit.update(self._jacobi_identity_audit())
                raise ValueError(
                    f"Level 5 Jacobi identity is verified on basis "
                    f"'{audit['jacobi_identity_basis']}', but process runtime H/E/S effort "
                    "linkage remains [N]."
                )
            else:
                audit["jacobi_identity_verified"] = False
                audit["jacobi_identity_basis"] = "unimplemented"
                raise ValueError(
                    "Level 5 executable Jacobi identity verification remains [N] for a "
                    "matrix_snapshot; the tensor-only reversible-operator payload cannot prove "
                    "a global Poisson identity."
                )
        return audit

    def canonical_drift(
        self,
        *,
        state: Optional[Tensor] = None,
        energy_value: Optional[Tensor] = None,
        entropy_value: Optional[Tensor] = None,
        energy_effort: Optional[Tensor] = None,
        entropy_effort: Optional[Tensor] = None,
        port_input: Optional[Tensor] = None,
        reaction_rates: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate canonical drift from potentials or precomputed point efforts.

        Potential values must already be connected to the supplied state through
        autograd. This binding is mutually exclusive with stored or explicit efforts;
        it does not infer physical units, a process law, or a StateManager commit.
        """

        terms = self._evaluate_canonical_terms(
            state=state,
            energy_value=energy_value,
            entropy_value=entropy_value,
            energy_effort=energy_effort,
            entropy_effort=entropy_effort,
            port_input=port_input,
            reaction_rates=reaction_rates,
        )
        return terms["drift"]


def _materialize_real_reaction_payload(value: Any, *, name: str) -> Any:
    """Reject complex data before dtype conversion and materialize iterators once."""

    if isinstance(value, Tensor):
        if value.device.type == "meta":
            raise ValueError(f"{name} cannot use the meta device")
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"{name} must use a dense, strided tensor")
        if torch.is_complex(value):
            raise TypeError(f"{name} must be real")
        if value.dtype == torch.bool:
            raise TypeError(f"{name} must be real numeric data, not bool")
        return value
    if isinstance(value, bool):
        raise TypeError(f"{name} must be real numeric data, not bool")
    if isinstance(value, complex):
        raise TypeError(f"{name} must be real")

    dtype_kind = getattr(getattr(value, "dtype", None), "kind", None)
    if dtype_kind == "c":
        raise TypeError(f"{name} must be real")
    if dtype_kind == "O":
        flat_values = getattr(value, "flat", ())
        for item in flat_values:
            _materialize_real_reaction_payload(item, name=name)
        return value
    if dtype_kind is not None:
        return value
    if isinstance(value, Mapping):
        return {
            key: _materialize_real_reaction_payload(item, name=name)
            for key, item in value.items()
        }
    if isinstance(value, (str, bytes)):
        return value
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return tuple(
        _materialize_real_reaction_payload(item, name=name) for item in iterator
    )


def _coerce_reaction_tensor(
    value: Any,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a real reaction tensor without allowing complex-to-real truncation."""

    payload = _materialize_real_reaction_payload(value, name=name)
    if isinstance(payload, Tensor):
        tensor = payload.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(payload, device=device, dtype=dtype)
    if torch.is_complex(tensor):
        raise TypeError(f"{name} must be real")
    if tensor.device.type == "meta" or tensor.layout != torch.strided or tensor.is_quantized:
        raise TypeError(f"{name} must use a materialized dense, strided tensor")
    return tensor


def _reduction_relative_tolerance_floor(
    *,
    dtype: torch.dtype,
    term_count: int,
) -> float:
    """Return an explicit forward-error floor for a floating-point reduction."""

    if dtype not in _REACTION_POINT_AUDIT_DTYPES:
        raise ValueError(
            "reaction thermodynamic point audits require float32 or float64 tensors"
        )
    unit_roundoff = float(torch.finfo(dtype).eps) / 2.0
    k = max(int(term_count), 1)
    if k * unit_roundoff >= 1.0:
        raise ValueError("reaction point-audit reduction is numerically unsupported")
    gamma_k = (k * unit_roundoff) / (1.0 - k * unit_roundoff)
    return 8.0 * gamma_k


@dataclass(frozen=True)
class ReactionNetworkAdvance:
    """Explicit IPHS/GENERIC reaction-network state advancement.

    References:
      - Feinberg (2019), Foundations of Chemical Reaction Network Theory.
      - Rao and Esposito (2016), thermodynamics of chemical reaction networks.
      - Grmela and Ottinger (1997), GENERIC formalism.
    Mathematical expression:
      x_dot = N_h r_h(x_h, theta, effort),
      P_reaction = sum_j A_j r_j >= 0, and
      C_h q_open = ledger_rate for a declared open exchange, and
      delta_x = x_dot * dt when a time step is supplied.
    Parameter meanings:
      stoichiometric_N: species-by-reaction stoichiometric matrix N_h.
      rates: reaction-rate tensor r_h with reaction index on axis 0.
      species: ordered state or output basis represented by N_h rows.
      reactions: ordered reaction-channel basis represented by N_h columns.
      affinities: signed reaction affinities under the explicit A_j r_j convention.
      open_exchange: separately routed state-space exchange q_open; it is audited
        here but is not added to x_dot a second time.
    Units:
      N_h maps reaction extents to species amounts; r_h has extent/time units.
      Physical A_j has energy/extent units; A_j r_j must share one power basis.
    Domain/constraints:
      No entropy rate is inferred without temperature. Executable unit and
      constitutive-domain
      binding remain [N] and this artifact is only a single runtime-point audit.
    Sign convention:
      Negative species derivatives remove donors; positive affinity-rate power
      is aggregate dissipation under the declared convention.
    """

    name: str
    species: tuple[str, ...]
    reactions: tuple[str, ...]
    stoichiometric_N: Tensor
    rates: Tensor
    state_derivative: Tensor
    state_delta: Optional[Tensor] = None
    conservation_audit: Optional[Mapping[str, object]] = field(
        default=None,
        init=False,
        compare=False,
    )
    thermodynamic_audit: Optional[Mapping[str, object]] = field(
        default=None,
        init=False,
        compare=False,
    )
    open_system_audit: Optional[Mapping[str, object]] = field(
        default=None,
        init=False,
        compare=False,
    )

    def species_rate(self, species_name: str) -> Tensor:
        """Return the computed x_dot row for a named species."""

        if species_name not in self.species:
            raise KeyError(f"Unknown species '{species_name}' in reaction network '{self.name}'.")
        return self.state_derivative[self.species.index(species_name)]

    def validate_conservation(
        self,
        conservation_matrix: Tensor | Any,
        *,
        tolerance: Optional[float] = None,
    ) -> dict[str, float | bool | str]:
        """Validate C_h N_h = 0 for conserved ledgers.

        ``tolerance`` bounds max|C_h N_h|; None reads ParameterManager key
        validation.iphs.reaction_conservation_tolerance.
        """

        tolerance = _resolve_iphs_tolerance("conservation_tolerance", tolerance)
        if isinstance(tolerance, bool):
            raise TypeError("conservation tolerance must be a real scalar")
        tolerance_value = float(tolerance)
        tolerance_tensor = torch.as_tensor(tolerance_value, dtype=torch.float64)
        if not bool(torch.isfinite(tolerance_tensor).item()) or tolerance_value < 0.0:
            raise ValueError("conservation tolerance must be finite and non-negative")
        C = _coerce_reaction_tensor(
            conservation_matrix,
            name="conservation_matrix",
            device=self.stoichiometric_N.device,
            dtype=self.stoichiometric_N.dtype,
        )
        if C.ndim == 1:
            C = C.reshape(1, -1)
        if C.ndim != 2:
            raise ValueError("conservation_matrix must be rank-1 or rank-2.")
        if int(C.shape[1]) != int(self.stoichiometric_N.shape[0]):
            raise ValueError("conservation_matrix columns must match species count.")
        if C.numel() == 0 or int(C.shape[0]) == 0:
            raise ValueError("conservation_matrix must be non-empty.")
        if not bool(torch.isfinite(C).all().item()):
            raise ValueError("conservation_matrix must be finite.")
        if bool(torch.any(torch.all(C == 0, dim=1)).item()):
            raise ValueError("Each conservation_matrix ledger row must be nonzero.")

        residual = C @ self.stoichiometric_N
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError("C_h N_h produced a non-finite conservation residual.")
        max_abs = float(torch.max(torch.abs(residual)).detach().cpu().item())
        return {
            "name": self.name,
            "passed": max_abs <= tolerance_value,
            "max_abs_residual": max_abs,
            "tolerance": tolerance_value,
        }


def _coerce_reaction_rates(
    rates: Tensor | Iterable[Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(rates, Tensor):
        return _coerce_reaction_tensor(
            rates,
            name="rates",
            device=device,
            dtype=dtype,
        )
    raw_values = _materialize_real_reaction_payload(rates, name="rates")
    if hasattr(raw_values, "shape") and hasattr(raw_values, "dtype"):
        return _coerce_reaction_tensor(
            raw_values,
            name="rates",
            device=device,
            dtype=dtype,
        )
    values = tuple(
        _coerce_reaction_tensor(
            item,
            name="rates",
            device=device,
            dtype=dtype,
        )
        for item in raw_values
    )
    if not values:
        raise ValueError("rates must contain at least one reaction channel.")
    return torch.stack(values, dim=0)


def _audit_reaction_thermodynamic_point(
    *,
    rates: Tensor,
    affinities: Any,
    sign_convention: str | None,
    relative_tolerance: float,
) -> dict[str, object]:
    """Audit aggregate affinity-rate power at one runtime point."""

    if sign_convention != _AFFINITY_RATE_SIGN_CONVENTION:
        raise ValueError(
            "reaction affinity sign convention must be "
            f"'{_AFFINITY_RATE_SIGN_CONVENTION}'"
        )
    A = _coerce_reaction_tensor(
        affinities,
        name="affinities",
        device=rates.device,
        dtype=rates.dtype,
    )
    if tuple(A.shape) != tuple(rates.shape):
        raise ValueError(
            "affinities shape must exactly match rates shape; broadcasting is forbidden"
        )
    if A.numel() == 0 or not bool(torch.isfinite(A).all().item()):
        raise ValueError("affinities must be non-empty and finite")

    # References:
    #   - Rao and Esposito (2016), PRX 6, 041064, Eqs. (55) and (58).
    # Mathematical expression:
    #   P = sum_j A_j r_j; tolerance = rtol * sum_j abs(A_j r_j).
    # Parameter meanings:
    #   A_j is physical affinity; r_j is signed reaction extent per time.
    # Units:
    #   A_j [energy/extent], r_j [extent/time], P [energy/time], rtol [1].
    # Domain/constraints:
    #   Finite tensors share a support and exact shape; axis 0 is reaction.
    #   No temperature or entropy balance is supplied to this point audit.
    # Sign convention:
    #   P >= -tolerance is aggregate dissipation, not a componentwise claim.
    affinity_rate_products = A * rates
    if not bool(torch.isfinite(affinity_rate_products).all().item()):
        raise ValueError("reaction affinity-rate product must remain finite")
    affinity_rate_power = torch.sum(affinity_rate_products, dim=0)
    affinity_rate_power_scale = torch.sum(
        torch.abs(affinity_rate_products),
        dim=0,
    )
    tolerance = relative_tolerance * affinity_rate_power_scale
    if not bool(torch.isfinite(affinity_rate_power).all().item()) or not bool(
        torch.isfinite(affinity_rate_power_scale).all().item()
    ) or not bool(torch.isfinite(tolerance).all().item()):
        raise ValueError("reaction affinity-rate power reduction must remain finite")
    violations = affinity_rate_power < -tolerance
    if bool(torch.any(violations).item()):
        minimum = float(torch.min(affinity_rate_power).detach().cpu().item())
        raise ValueError(
            "reaction affinity-rate power A.T r is negative beyond the declared "
            f"relative tolerance; minimum = {minimum}"
        )

    return {
        "validated": True,
        "passed": True,
        "evidence_scope": "single_runtime_point",
        "operational_claim": False,
        "affinity_sign_convention": sign_convention,
        "componentwise_required": False,
        "affinity_rate_power": affinity_rate_power,
        "affinity_rate_products": affinity_rate_products,
        "affinity_rate_power_min": float(
            torch.min(affinity_rate_power).detach().cpu().item()
        ),
        "affinity_rate_power_scale_max": float(
            torch.max(affinity_rate_power_scale).detach().cpu().item()
        ),
        "affinity_rate_power_tolerance_max": float(
            torch.max(tolerance).detach().cpu().item()
        ),
        "relative_tolerance": relative_tolerance,
        "zero_scale": bool(torch.all(affinity_rate_power_scale == 0).item()),
        "unit_binding": "[N]",
        "constitutive_domain_binding": "[N]",
    }


def _audit_open_reaction_ledger_point(
    *,
    state_derivative: Tensor,
    conservation_matrix: Tensor,
    open_exchange: Any,
    open_ledger_rate: Any,
    route: str,
    relative_tolerance: float,
) -> dict[str, object]:
    """Audit C q_open = ledger_rate without applying q_open to the state twice."""

    if route not in _THERMODYNAMIC_OPEN_ROUTES:
        raise ValueError(
            f"open_system_route '{route}' cannot satisfy a reaction thermodynamic "
            "point audit; use port, boundary_export, or managed_export"
        )
    q_open = _coerce_reaction_tensor(
        open_exchange,
        name="open_exchange",
        device=state_derivative.device,
        dtype=state_derivative.dtype,
    )
    if tuple(q_open.shape) != tuple(state_derivative.shape):
        raise ValueError(
            "open_exchange shape must exactly match state_derivative shape; "
            "broadcasting is forbidden"
        )
    if q_open.numel() == 0 or not bool(torch.isfinite(q_open).all().item()):
        raise ValueError("open_exchange must be non-empty and finite")

    expected_ledger_shape = (
        int(conservation_matrix.shape[0]),
    ) + tuple(state_derivative.shape[1:])
    ledger_rate = _coerce_reaction_tensor(
        open_ledger_rate,
        name="open_ledger_rate",
        device=state_derivative.device,
        dtype=state_derivative.dtype,
    )
    if tuple(ledger_rate.shape) != expected_ledger_shape:
        raise ValueError(
            "open_ledger_rate shape must exactly match C_h q_open shape; "
            "broadcasting is forbidden"
        )
    if ledger_rate.numel() == 0 or not bool(torch.isfinite(ledger_rate).all().item()):
        raise ValueError("open_ledger_rate must be non-empty and finite")

    # References: Rao and Esposito (2016), open reaction-network balance laws.
    # Mathematical expression: C_h q_open = ledger_rate.
    # Parameter meanings and units: q_open is a state-space exchange rate, C_h
    # maps state amounts to ledgers, and ledger_rate uses ledger-amount/time units.
    flat_exchange = q_open.reshape((int(q_open.shape[0]), -1))
    computed_ledger = (conservation_matrix @ flat_exchange).reshape(
        expected_ledger_shape
    )
    residual = computed_ledger - ledger_rate
    scale = (
        torch.abs(conservation_matrix) @ torch.abs(flat_exchange)
    ).reshape(expected_ledger_shape) + torch.abs(ledger_rate)
    tolerance = relative_tolerance * scale
    for name, value in (
        ("C_h q_open", computed_ledger),
        ("open ledger residual", residual),
        ("open ledger scale", scale),
        ("open ledger tolerance", tolerance),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must remain finite")
    violations = torch.abs(residual) > tolerance
    if bool(torch.any(violations).item()):
        maximum = float(torch.max(torch.abs(residual)).detach().cpu().item())
        raise ValueError(
            "open reaction ledger violates C_h q_open = ledger_rate; "
            f"max_abs_residual = {maximum}"
        )

    return {
        "validated": True,
        "passed": True,
        "evidence_scope": "single_runtime_point",
        "operational_claim": False,
        "route": route,
        "computed_ledger_rate": computed_ledger,
        "declared_ledger_rate": ledger_rate,
        "residual": residual,
        "max_abs_residual": float(
            torch.max(torch.abs(residual)).detach().cpu().item()
        ),
        "scale_max": float(torch.max(scale).detach().cpu().item()),
        "tolerance_max": float(torch.max(tolerance).detach().cpu().item()),
        "relative_tolerance": relative_tolerance,
        "zero_scale": bool(torch.all(scale == 0).item()),
        "exchange_applied_to_state_derivative": False,
        "unit_binding": "[N]",
        "route_application_binding": "[N]",
        "no_double_count_runtime_proof": "[N]",
    }


def advance_reaction_network(
    *,
    name: str,
    species: tuple[str, ...],
    reactions: tuple[str, ...],
    stoichiometric_N: Tensor | Any,
    rates: Tensor | Iterable[Any],
    dt: float | Tensor | None = None,
    conservation_matrix: Tensor | Any | None = None,
    conservation_tolerance: Optional[float] = None,
    strict_conservation: bool = True,
    require_conservation_matrix: bool = False,
    open_system_route: OpenSystemRoute | str | None = None,
    affinities: Tensor | Any | None = None,
    affinity_sign_convention: str | None = None,
    thermodynamic_rtol: Optional[float] = None,
    open_exchange: Tensor | Any | None = None,
    open_ledger_rate: Tensor | Any | None = None,
    require_thermodynamic_point_audit: bool = False,
) -> ReactionNetworkAdvance:
    """Advance a reaction network through the explicit N_h r_h operator.

    References:
      - Feinberg (2019), chemical reaction network stoichiometry.
      - Rao and Esposito (2016), entropy production in reaction networks.
      - Grmela and Ottinger (1997), GENERIC state evolution.
    Mathematical expression:
      x_dot = N_h r_h, delta_x = x_dot * dt,
      C_h N_h = 0, sum_j A_j r_j >= 0, and
      C_h q_open = ledger_rate for a complete open-system point audit.
    Parameter meanings:
      species: ordered output/state rows of N_h.
      reactions: ordered reaction-rate columns of N_h.
      stoichiometric_N: signed stoichiometric matrix [n_species, n_reactions].
      rates: reaction rates r_h [n_reactions, ...].
      dt: optional time increment for producing delta_x.
      conservation_matrix: optional ledger matrix C_h for validating C_h N_h = 0.
      conservation_tolerance: absolute bound on max|C_h N_h|; None reads
        validation.iphs.reaction_conservation_tolerance.
      strict_conservation: reject the reaction advance when declared C_h N_h != 0.
      require_conservation_matrix: reject the advance when no C_h is declared.
      open_system_route: explicit route for open-system or diagnostic output maps.
      affinities: A_h with exactly the same shape as rates; no broadcasting.
      affinity_sign_convention: must be affinity_times_rate_nonnegative.
      thermodynamic_rtol: relative reduction tolerance without an absolute floor;
        None reads validation.iphs.reaction_thermodynamic_rtol.
      open_exchange: q_open with exactly the state_derivative shape.
      open_ledger_rate: declared C_h q_open value with ledger-by-batch shape.
      require_thermodynamic_point_audit: require A, C, and complete open fields.
    Units:
      N_h maps reaction rates to state rates. A_j r_j must have one consistent
      power unit, with A_j measured as energy per reaction extent.
    Domain/constraints:
      Finite inputs obey explicit species, reaction, and batch shapes. A positive
      temperature and a valid entropy balance are needed for sigma = P / T;
      this helper does not infer or emit that entropy rate.
      Unit provenance, affinity construction, constitutive support, open-route
      application, and time-integrated no-double-count proof remain [N].
    Sign convention:
      Negative species rates remove donors; positive rates add recipients.
      The supplied physical affinity-rate pairing uses aggregate nonnegativity.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")
    if not species or any(not isinstance(item, str) or not item for item in species):
        raise ValueError("species must not be empty.")
    if len(set(species)) != len(species):
        raise ValueError("species must not contain duplicates.")
    if not reactions or any(not isinstance(item, str) or not item for item in reactions):
        raise ValueError("reactions must not be empty.")
    if len(set(reactions)) != len(reactions):
        raise ValueError("reactions must not contain duplicates.")
    for flag_name, flag_value in (
        ("strict_conservation", strict_conservation),
        ("require_conservation_matrix", require_conservation_matrix),
        ("require_thermodynamic_point_audit", require_thermodynamic_point_audit),
    ):
        if not isinstance(flag_value, bool):
            raise TypeError(f"{flag_name} must be a bool")
    conservation_tolerance = _resolve_iphs_tolerance("conservation_tolerance", conservation_tolerance)
    thermodynamic_rtol = _resolve_iphs_tolerance("thermodynamic_rtol", thermodynamic_rtol)
    if isinstance(conservation_tolerance, bool):
        raise TypeError("conservation_tolerance must be a real scalar")
    tolerance_value = float(conservation_tolerance)
    tolerance_tensor = torch.as_tensor(tolerance_value, dtype=torch.float64)
    if not bool(torch.isfinite(tolerance_tensor).item()) or tolerance_value < 0.0:
        raise ValueError(
            "conservation_tolerance must be finite and non-negative"
        )
    if isinstance(thermodynamic_rtol, bool):
        raise TypeError("thermodynamic_rtol must be a real scalar")
    relative_tolerance = float(thermodynamic_rtol)
    relative_tolerance_tensor = torch.as_tensor(
        relative_tolerance,
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(relative_tolerance_tensor).item()) or relative_tolerance < 0.0:
        raise ValueError("thermodynamic_rtol must be finite and non-negative")

    stoichiometric_payload = _materialize_real_reaction_payload(
        stoichiometric_N,
        name="stoichiometric_N",
    )
    prototype = stoichiometric_payload if isinstance(stoichiometric_payload, Tensor) else None
    if prototype is None:
        inferred = torch.as_tensor(stoichiometric_payload)
        device = torch.device("cpu")
        dtype = inferred.dtype if inferred.is_floating_point() else torch.float32
    else:
        device = prototype.device
        dtype = prototype.dtype if prototype.is_floating_point() else torch.float32
    N = _coerce_reaction_tensor(
        stoichiometric_payload,
        name="stoichiometric_N",
        device=device,
        dtype=dtype,
    )
    if N.ndim != 2:
        raise ValueError("stoichiometric_N must be a rank-2 matrix.")
    if int(N.shape[0]) != len(species):
        raise ValueError("stoichiometric_N row count must match species count.")
    if int(N.shape[1]) != len(reactions):
        raise ValueError("stoichiometric_N column count must match reaction count.")
    if not bool(torch.isfinite(N).all().item()):
        raise ValueError("stoichiometric_N must be finite.")

    r_h = _coerce_reaction_rates(rates, device=N.device, dtype=N.dtype)
    if r_h.ndim == 0:
        raise ValueError("rates must include a reaction axis.")
    if int(r_h.shape[0]) != len(reactions):
        raise ValueError("rates reaction count must match reaction count.")
    if r_h.numel() == 0 or any(int(size) == 0 for size in r_h.shape):
        raise ValueError("rates must be non-empty in every declared axis.")
    if not bool(torch.isfinite(r_h).all().item()):
        raise ValueError("rates must be finite.")

    rate_shape = tuple(r_h.shape[1:])
    flat_rates = r_h.reshape((len(reactions), -1))

    # References:
    #   - Feinberg (2019); Rao and Esposito (2016).
    # Mathematical expression:
    #   x_dot = N_h r_h.
    # Parameter meanings:
    #   N_h maps ordered reaction extents to species amounts; r_h stores rates.
    # Units:
    #   Each mapped output has the declared species-amount/time unit.
    # Domain/constraints:
    #   Finite rank-2 N_h and finite rates share the ordered reaction axis.
    # Sign convention:
    #   Negative species rates remove donors; positive rates add recipients.
    flat_derivative = N @ flat_rates
    if not bool(torch.isfinite(flat_derivative).all().item()):
        raise ValueError("Reaction operator N_h r_h produced a non-finite state derivative.")
    state_derivative = flat_derivative.reshape((len(species),) + rate_shape)
    state_delta: Optional[Tensor] = None
    if dt is not None:
        if isinstance(dt, bool):
            raise TypeError("dt must be a real scalar")
        dt_tensor = _coerce_reaction_tensor(
            dt,
            name="dt",
            device=N.device,
            dtype=N.dtype,
        )
        if dt_tensor.numel() != 1:
            raise TypeError("dt must be a real scalar")
        if not bool(torch.isfinite(dt_tensor).all().item()) or bool(
            torch.any(dt_tensor <= 0.0).item()
        ):
            raise ValueError("dt must be finite and positive")
        state_delta = state_derivative * dt_tensor
        if not bool(torch.isfinite(state_delta).all().item()):
            raise ValueError("Reaction time advance produced a non-finite state delta.")

    advance = ReactionNetworkAdvance(
        name=name,
        species=tuple(species),
        reactions=tuple(reactions),
        stoichiometric_N=N,
        rates=r_h,
        state_derivative=state_derivative,
        state_delta=state_delta,
    )
    if open_system_route is not None:
        route_value = str(open_system_route)
        if route_value not in _VALID_OPEN_SYSTEM_ROUTES:
            raise ValueError(
                "open_system_route must be one of "
                f"{sorted(_VALID_OPEN_SYSTEM_ROUTES)}; got '{route_value}'."
            )
    else:
        route_value = None

    point_audit_requested = bool(
        require_thermodynamic_point_audit
        or affinities is not None
        or affinity_sign_convention is not None
        or open_exchange is not None
        or open_ledger_rate is not None
    )
    if conservation_matrix is None:
        audit = {
            "validated": False,
            "passed": None,
            "reason": (
                "open_system_route_declared"
                if route_value is not None
                else "conservation_matrix_not_declared"
            ),
            "strict_conservation": bool(strict_conservation),
            "require_conservation_matrix": bool(require_conservation_matrix),
            "open_system_route": route_value,
        }
        object.__setattr__(advance, "conservation_audit", audit)
        if point_audit_requested:
            raise ValueError(
                "reaction thermodynamic point audit requires a non-empty "
                "conservation_matrix for C_h N_h = 0"
            )
        if require_conservation_matrix and route_value is None:
            raise ValueError(
                "Reaction network "
                f"'{name}' requires declared conservation_matrix for C_h N_h = 0 validation."
            )
        return advance

    conservation_tensor = _coerce_reaction_tensor(
        conservation_matrix,
        name="conservation_matrix",
        device=N.device,
        dtype=N.dtype,
    )
    audit = advance.validate_conservation(
        conservation_tensor,
        tolerance=tolerance_value,
    )
    object.__setattr__(advance, "conservation_audit", audit)
    if (strict_conservation or point_audit_requested) and not bool(
        audit.get("passed", False)
    ):
        raise ValueError(
            "Reaction network "
            f"'{name}' violates declared C_h N_h = 0; "
            f"max_abs_residual = {audit.get('max_abs_residual')}."
        )
    if not point_audit_requested:
        return advance

    if affinities is None:
        raise ValueError(
            "reaction thermodynamic point audit requires non-empty affinities"
        )
    reduction_terms = max(
        len(reactions),
        len(species) if route_value is not None else 1,
    )
    relative_tolerance_floor = _reduction_relative_tolerance_floor(
        dtype=N.dtype,
        term_count=reduction_terms,
    )
    if relative_tolerance < relative_tolerance_floor:
        raise ValueError(
            "thermodynamic_rtol is below the dtype reduction precision floor; "
            f"minimum = {relative_tolerance_floor}"
        )
    thermodynamic_audit = _audit_reaction_thermodynamic_point(
        rates=r_h,
        affinities=affinities,
        sign_convention=affinity_sign_convention,
        relative_tolerance=relative_tolerance,
    )
    thermodynamic_audit["relative_tolerance_floor"] = relative_tolerance_floor
    object.__setattr__(advance, "thermodynamic_audit", thermodynamic_audit)

    if route_value is None:
        if open_exchange is not None or open_ledger_rate is not None:
            raise ValueError(
                "open_exchange and open_ledger_rate require an open_system_route"
            )
        return advance
    if open_exchange is None:
        raise ValueError(
            "open reaction thermodynamic point audit requires open_exchange"
        )
    if open_ledger_rate is None:
        raise ValueError(
            "open reaction thermodynamic point audit requires open_ledger_rate"
        )
    if conservation_tensor.ndim == 1:
        conservation_tensor = conservation_tensor.reshape(1, -1)
    open_audit = _audit_open_reaction_ledger_point(
        state_derivative=state_derivative,
        conservation_matrix=conservation_tensor,
        open_exchange=open_exchange,
        open_ledger_rate=open_ledger_rate,
        route=route_value,
        relative_tolerance=relative_tolerance,
    )
    open_audit["relative_tolerance_floor"] = relative_tolerance_floor
    object.__setattr__(advance, "open_system_audit", open_audit)
    return advance


def _contribution_payload(contribution: Any) -> Mapping[str, Any]:
    if isinstance(contribution, Mapping):
        return contribution
    if hasattr(contribution, "contribute_iphs_structure"):
        payload = contribution.contribute_iphs_structure()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError(
        "IPHS contributions must be mappings or expose contribute_iphs_structure()."
    )


def _first_tensor_payload(payloads: Iterable[Mapping[str, Any]]) -> Optional[Tensor]:
    for payload in payloads:
        for key in (
            "J",
            "R",
            "M",
            "G",
            "stoichiometric_N",
            "energy_gradient",
            "entropy_gradient",
        ):
            value = payload.get(key)
            if isinstance(value, Tensor):
                return value
    return None


def _optional_tensor(
    value: Any, *, device: torch.device, dtype: torch.dtype,
    exact_zero_name: Optional[str] = None,
) -> Optional[Tensor]:
    if value is None:
        return None
    if exact_zero_name is not None:
        value = _materialize_real_reaction_payload(value, name=exact_zero_name)
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, Tensor):
                nonzero = bool(torch.any(item != 0).item())
            elif isinstance(item, (str, bytes, Mapping)):
                raise TypeError(f"{exact_zero_name} must contain real numeric data.")
            else:
                try:
                    iterator = iter(item)
                except TypeError:
                    nonzero = bool(item != 0)
                else:
                    pending.extend(iterator)
                    continue
            if nonzero:
                raise ValueError(
                    f"zero_field requires exactly zero {exact_zero_name} before dtype conversion."
                )
    return torch.as_tensor(value, device=device, dtype=dtype)


def _state_indices(
    *,
    local_state_vars: tuple[str, ...],
    global_state_vars: tuple[str, ...],
) -> tuple[int, ...]:
    if len(set(local_state_vars)) != len(local_state_vars):
        raise ValueError("IPHS contribution state_vars must not contain duplicates.")
    global_index = {name: index for index, name in enumerate(global_state_vars)}
    missing = [name for name in local_state_vars if name not in global_index]
    if missing:
        raise ValueError(
            "IPHS contribution state_vars must be a subset of assembly state_vars; "
            f"unknown state vars: {missing}."
        )
    return tuple(global_index[name] for name in local_state_vars)


def _embed_square_matrix(
    matrix: Tensor,
    *,
    indices: tuple[int, ...],
    n_state: int,
    key: str,
) -> Tensor:
    local_n = len(indices)
    if tuple(matrix.shape) != (local_n, local_n):
        raise ValueError(
            f"IPHS contribution {key} shape {tuple(matrix.shape)} does not match "
            f"local state size {(local_n, local_n)}."
        )
    embedded = torch.zeros((n_state, n_state), device=matrix.device, dtype=matrix.dtype)
    rows = torch.as_tensor(indices, device=matrix.device, dtype=torch.long)
    embedded[rows[:, None], rows[None, :]] = matrix
    return embedded


def _embed_row_matrix(
    matrix: Tensor,
    *,
    indices: tuple[int, ...],
    n_state: int,
    key: str,
) -> Tensor:
    local_n = len(indices)
    if matrix.ndim != 2:
        raise ValueError(f"IPHS contribution {key} must be a rank-2 matrix.")
    if int(matrix.shape[0]) != local_n:
        raise ValueError(
            f"IPHS contribution {key} row count {int(matrix.shape[0])} does not match "
            f"local state size {local_n}."
        )
    embedded = torch.zeros((n_state, int(matrix.shape[1])), device=matrix.device, dtype=matrix.dtype)
    rows = torch.as_tensor(indices, device=matrix.device, dtype=torch.long)
    embedded[rows, :] = matrix
    return embedded


def _embed_vector(
    vector: Tensor,
    *,
    indices: tuple[int, ...],
    n_state: int,
    key: str,
) -> Tensor:
    local_n = len(indices)
    if vector.ndim != 1:
        vector = vector.reshape(-1)
    if int(vector.shape[0]) != local_n:
        raise ValueError(
            f"IPHS contribution {key} length {int(vector.shape[0])} does not match "
            f"local state size {local_n}."
        )
    embedded = torch.zeros((n_state,), device=vector.device, dtype=vector.dtype)
    rows = torch.as_tensor(indices, device=vector.device, dtype=torch.long)
    embedded[rows] = vector
    return embedded


def _concat_optional_columns(
    current: Optional[Tensor],
    candidate: Optional[Tensor],
    *,
    key: str,
) -> Optional[Tensor]:
    if candidate is None:
        return current
    if current is None:
        return candidate.clone()
    if current.ndim != 2 or candidate.ndim != 2:
        raise ValueError(f"IPHS contribution '{key}' must be rank-2 for column assembly.")
    if int(current.shape[0]) != int(candidate.shape[0]):
        raise ValueError(f"IPHS contribution '{key}' row counts must match for assembly.")
    return torch.cat((current, candidate), dim=1)


def _validate_skew_matrix(matrix: Tensor, *, key: str, tolerance: float) -> None:
    if matrix.ndim != 2 or int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ValueError(f"{key} must be a square rank-2 matrix.")
    residual = torch.linalg.norm(matrix + matrix.transpose(-1, -2))
    residual_norm = float(residual.detach().cpu().item())
    if residual_norm > float(tolerance):
        raise ValueError(f"{key} must be skew-symmetric; residual = {residual_norm}.")


def _validate_psd_matrix(matrix: Tensor, *, key: str, tolerance: float) -> None:
    if matrix.ndim != 2 or int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ValueError(f"{key} must be a square rank-2 matrix.")
    residual = torch.linalg.norm(matrix - matrix.transpose(-1, -2))
    residual_norm = float(residual.detach().cpu().item())
    if residual_norm > float(tolerance):
        raise ValueError(f"{key} must be symmetric; residual = {residual_norm}.")
    eigvals = torch.linalg.eigvalsh(0.5 * (matrix + matrix.transpose(-1, -2)))
    min_eig = float(torch.min(eigvals).detach().cpu().item())
    if min_eig < -float(tolerance):
        raise ValueError(f"{key} must be positive semidefinite; min eigenvalue = {min_eig}.")


def assemble_iphs_structure(
    *,
    name: str,
    state_vars: tuple[str, ...],
    energy_functional: str,
    entropy_functional: str,
    contributions: Iterable[Any],
    formulation: IPHSFormulation = "unspecified",
    poisson_operator: PoissonOperatorKind,
    reaction_representation: ReactionRepresentation,
    reversible_operator_state_dependence: OperatorStateDependence = "unspecified",
    jacobi_identity_evidence: str = "",
    poisson_field: Optional[Callable[[Tensor], Tensor]] = None,
    poisson_domain: Optional[Tensor] = None,
    jacobi_tol: Optional[float] = None,
    port_interconnection_C: Any | None = None,
    port_dissipation_D: Any | None = None,
    validate: bool = False,
    strict: bool = False,
    required_level: Optional[int] = None,
    skew_tol: Optional[float] = None,
    psd_tol: Optional[float] = None,
    degeneracy_tol: Optional[float] = None,
) -> IPHSStructureSpec:
    """Assemble an IPHS structure from declared functor contributions.

    References:
      - van der Schaft and Maschke (1995), port-Hamiltonian interconnection.
      - Grmela and Oettinger (1997), GENERIC reversible/irreversible split.
    Mathematical expressions:
      J = sum_i i_i J_i i_i^T + G_int C G_int^T,
      R = sum_i i_i R_i i_i^T + G_int D G_int^T,
      M = sum_i i_i M_i i_i^T,
      N = [i_1 N_1, ..., i_m N_m],
      dissipative_ph: x_dot = (J - R)e_H + G_ext u_ext + N r,
      generic: x_dot = J e_E + M e_S + G_ext u_ext + N r.
      The final N r term is omitted for an absorbed reaction representation.
    Parameter meanings:
      state_vars: ordered global state basis for the assembled system.
      contributions: mappings or functors declaring local or global J, R, M, G, N, ports, gradients.
      port_interconnection_C: optional skew matrix over internal port columns.
      port_dissipation_D: optional PSD matrix over internal port columns.
      validate: run IPHSStructureSpec.validate after assembly.
      strict: require non-empty contributions with executable runtime owners.
      required_level: certificate-derived structural level to validate; not a second level owner.
      poisson_operator: zero_field requires each local J and C to be exactly zero
        before precision conversion; cancellation is not a field identity proof.
        constant_field and state_field use poisson_field (callable J(x)),
        poisson_domain ((m, n) sample points) and jacobi_tol, which this
        assembler forwards unchanged to IPHSStructureSpec; compile_graph
        resolves them from graph.metadata.iphs_assembly. A jacobi_tol of None
        selects ParameterManager key validation.iphs.jacobi_tolerance.
      skew_tol / psd_tol / degeneracy_tol: explicit tolerances, or None to read
        validation.iphs.skew_tolerance / psd_tolerance / degeneracy_tolerance;
        the resolved skew and PSD tolerances also check C and D here.
      reaction_representation: absorbed verifies the full irreversible rate against
        the retained N r at runtime, without adding an independent reaction term.
    Units: local matrices use the declared common state, effort, and port basis.
    Domain/constraints: declared finite basis; zero_field gives J(x)=0 globally
      in this basis but does not establish process-level H/E/S runtime linkage.
    Sign convention: C contributes reversible flow; D contributes positive resistance.
    """

    skew_tol = _resolve_iphs_tolerance("skew_tol", skew_tol)
    psd_tol = _resolve_iphs_tolerance("psd_tol", psd_tol)
    degeneracy_tol = _resolve_iphs_tolerance("degeneracy_tol", degeneracy_tol)
    global_state_vars = tuple(state_vars)
    if len(set(global_state_vars)) != len(global_state_vars):
        raise ValueError("state_vars must not contain duplicates for IPHS assembly.")

    raw_contributions = tuple(contributions)
    payloads = tuple(_contribution_payload(contribution) for contribution in raw_contributions)
    if strict and required_level is not None:
        if isinstance(required_level, bool) or not isinstance(required_level, int):
            raise ValueError("strict IPHS required_level must be integer 4 or 5.")
        if required_level < 4:
            raise ValueError("strict IPHS required_level cannot be lower than 4.")
        if required_level > 5:
            raise ValueError("strict IPHS required_level must be integer 4 or 5.")
    if formulation == "generic" and port_dissipation_D is not None:
        raise ValueError(
            "GENERIC assembly cannot declare dPH port_dissipation_D feedback."
        )
    if strict and not payloads:
        raise ValueError("Strict IPHS assembly requires at least one runtime-linked contribution.")
    contribution_sources: list[str] = []
    for index, (raw_contribution, payload) in enumerate(zip(raw_contributions, payloads)):
        source = str(
            payload.get("source_node_id", payload.get("runtime_owner", ""))
        ).strip()
        if not source and hasattr(raw_contribution, "contribute_iphs_structure"):
            source = (
                f"{raw_contribution.__class__.__module__}."
                f"{raw_contribution.__class__.__qualname__}"
            )
        if strict and not source:
            raise ValueError(
                f"Strict IPHS contribution {index} must declare a runtime owner or source_node_id."
            )
        if source:
            contribution_sources.append(source)
    prototype = _first_tensor_payload(payloads)
    device = prototype.device if prototype is not None else torch.device("cpu")
    dtype = prototype.dtype if prototype is not None else torch.float32
    n_state = len(global_state_vars)
    if n_state < 1:
        raise ValueError("state_vars must not be empty for IPHS assembly.")

    J = torch.zeros((n_state, n_state), device=device, dtype=dtype)
    R = torch.zeros((n_state, n_state), device=device, dtype=dtype)
    M = torch.zeros((n_state, n_state), device=device, dtype=dtype)
    G: Optional[Tensor] = None
    stoichiometric_N: Optional[Tensor] = None
    ports: list[PortSpecIPHS] = []
    has_typed_g_contribution = False
    has_untyped_g_contribution = False
    energy_gradient: Optional[Tensor] = None
    entropy_gradient: Optional[Tensor] = None

    for payload in payloads:
        if payload.get("reaction_representation", reaction_representation) != reaction_representation:
            raise ValueError(
                "Contribution reaction_representation must match the assembly; "
                "mixed independent and absorbed reactions require an explicit decomposition."
            )
        local_state_vars = tuple(payload.get("state_vars", global_state_vars))
        indices = _state_indices(
            local_state_vars=local_state_vars,
            global_state_vars=global_state_vars,
        )

        local_J = _optional_tensor(
            payload.get("J"), device=device, dtype=dtype,
            exact_zero_name="contribution J" if poisson_operator == "zero_field" else None,
        )
        local_R = _optional_tensor(payload.get("R"), device=device, dtype=dtype)
        local_M = _optional_tensor(payload.get("M"), device=device, dtype=dtype)
        if local_J is not None:
            J = J + _embed_square_matrix(local_J, indices=indices, n_state=n_state, key="J")
        if local_R is not None:
            R = R + _embed_square_matrix(local_R, indices=indices, n_state=n_state, key="R")
        if local_M is not None:
            M = M + _embed_square_matrix(local_M, indices=indices, n_state=n_state, key="M")

        local_G = _optional_tensor(payload.get("G"), device=device, dtype=dtype)
        raw_ports = tuple(payload.get("ports", ()) or ())
        if raw_ports and local_G is None:
            raise ValueError(
                "Typed ports and their G block must be declared in the same contribution."
            )
        if local_G is not None and raw_ports:
            if int(local_G.shape[1]) != _port_dof_count(raw_ports):
                raise ValueError(
                    "Each contribution G width must match its typed port degrees of freedom."
                )
            if has_untyped_g_contribution:
                raise ValueError(
                    "Typed and untyped G blocks cannot be mixed across contributions; "
                    "declare ports in the same contribution."
                )
            has_typed_g_contribution = True
        elif local_G is not None:
            if has_typed_g_contribution:
                raise ValueError(
                    "Typed and untyped G blocks cannot be mixed across contributions; "
                    "declare ports in the same contribution."
                )
            has_untyped_g_contribution = True
        G = _concat_optional_columns(
            G,
            None if local_G is None else _embed_row_matrix(local_G, indices=indices, n_state=n_state, key="G"),
            key="G",
        )
        local_N = _optional_tensor(payload.get("stoichiometric_N"), device=device, dtype=dtype)
        stoichiometric_N = _concat_optional_columns(
            stoichiometric_N,
            (
                None
                if local_N is None
                else _embed_row_matrix(local_N, indices=indices, n_state=n_state, key="stoichiometric_N")
            ),
            key="stoichiometric_N",
        )

        if raw_ports:
            ports.extend(raw_ports)

        local_energy_gradient = _optional_tensor(
            payload.get("energy_gradient"),
            device=device,
            dtype=dtype,
        )
        local_entropy_gradient = _optional_tensor(
            payload.get("entropy_gradient"),
            device=device,
            dtype=dtype,
        )
        if local_energy_gradient is not None:
            embedded_energy_gradient = _embed_vector(
                local_energy_gradient,
                indices=indices,
                n_state=n_state,
                key="energy_gradient",
            )
            energy_gradient = (
                embedded_energy_gradient
                if energy_gradient is None
                else energy_gradient + embedded_energy_gradient
            )
        if local_entropy_gradient is not None:
            embedded_entropy_gradient = _embed_vector(
                local_entropy_gradient,
                indices=indices,
                n_state=n_state,
                key="entropy_gradient",
            )
            entropy_gradient = (
                embedded_entropy_gradient
                if entropy_gradient is None
                else entropy_gradient + embedded_entropy_gradient
            )

    internal_port_indices: tuple[int, ...] = ()
    if G is not None:
        port_count = int(G.shape[1])
        if ports and _port_dof_count(ports) != port_count:
            raise ValueError(
                "Declared typed port degrees of freedom must match the assembled G width."
            )
        roles = tuple(port.role for port in ports)
        has_explicit_role = any(role != "unspecified" for role in roles)
        has_unspecified_role = any(role == "unspecified" for role in roles)
        if has_explicit_role and has_unspecified_role:
            raise ValueError(
                "Port roles cannot mix explicit and unspecified declarations."
            )
        if has_explicit_role:
            internal_port_indices = _port_column_indices(ports, role="internal")
        elif port_interconnection_C is not None or port_dissipation_D is not None:
            # C and D couple internal ports only. Ports without declared roles
            # are not assumed internal (the former legacy branch treated every
            # column of G as internal); the payload must declare the roles.
            raise ValueError(
                "port_interconnection_C and port_dissipation_D require typed ports with "
                "explicit internal or external roles for every column of G."
            )

    G_internal: Optional[Tensor] = None
    if G is not None and internal_port_indices:
        internal_index = torch.tensor(
            internal_port_indices,
            device=G.device,
            dtype=torch.long,
        )
        G_internal = torch.index_select(G, 1, internal_index)

    if port_interconnection_C is not None:
        if G is None:
            raise ValueError("port_interconnection_C requires an assembled G matrix.")
        if G_internal is None:
            raise ValueError(
                "port_interconnection_C requires at least one internal port role."
            )
        C = _optional_tensor(
            port_interconnection_C, device=G.device, dtype=G.dtype,
            exact_zero_name="port_interconnection_C" if poisson_operator == "zero_field" else None,
        )
        assert C is not None
        _validate_skew_matrix(C, key="port_interconnection_C", tolerance=skew_tol)
        if int(C.shape[0]) != int(G_internal.shape[1]):
            raise ValueError(
                "port_interconnection_C shape must match the internal port count."
            )
        J = J + G_internal @ C @ G_internal.transpose(-1, -2)

    if port_dissipation_D is not None:
        if G is None:
            raise ValueError("port_dissipation_D requires an assembled G matrix.")
        if G_internal is None:
            raise ValueError(
                "port_dissipation_D requires at least one internal port role."
            )
        D = torch.as_tensor(port_dissipation_D, device=G.device, dtype=G.dtype)
        _validate_psd_matrix(D, key="port_dissipation_D", tolerance=psd_tol)
        if int(D.shape[0]) != int(G_internal.shape[1]):
            raise ValueError(
                "port_dissipation_D shape must match the internal port count."
            )
        R = R + G_internal @ D @ G_internal.transpose(-1, -2)

    spec = IPHSStructureSpec(
        name=name,
        state_vars=global_state_vars,
        energy_functional=energy_functional,
        entropy_functional=entropy_functional,
        J=J,
        R=R,
        M=M,
        G=G,
        stoichiometric_N=stoichiometric_N,
        ports=tuple(ports),
        skew_tol=skew_tol,
        psd_tol=psd_tol,
        degeneracy_tol=degeneracy_tol,
        energy_gradient=energy_gradient,
        entropy_gradient=entropy_gradient,
        contribution_count=len(payloads),
        contribution_sources=tuple(contribution_sources),
        formulation=formulation,
        reversible_operator_state_dependence=reversible_operator_state_dependence,
        jacobi_identity_evidence=jacobi_identity_evidence,
        poisson_operator=poisson_operator,
        poisson_field=poisson_field,
        poisson_domain=poisson_domain,
        jacobi_tol=jacobi_tol,
        reaction_representation=reaction_representation,
    )
    if validate:
        level = required_level if required_level is not None else (4 if strict else None)
        spec.validate(required_level=level)
    return spec


__all__ = [
    "IPHSStructureSpec",
    "IPHSFormulation",
    "OperatorStateDependence",
    "PoissonOperatorKind",
    "ReactionRepresentation",
    "PortKind",
    "PortRole",
    "PortSpecIPHS",
    "PowerConvention",
    "ReactionNetworkAdvance",
    "assemble_iphs_structure",
    "advance_reaction_network",
]
