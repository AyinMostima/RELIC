"""Geometric time-integration compatibility helpers.

This module does not define a replacement integrator. It exposes the existing
ImplicitMidpointSolver through structure-aware compatibility and transaction
helpers used by RELIC-IPHS-DEC audits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import torch

from .base import SolverFamily, SolverResult, SolverStatus
from .registry import SolverRegistry
from .temporal import ImplicitMidpointSolver


@dataclass(frozen=True)
class IntegrationStageCompatibilitySpec:
    """Compatibility metadata for one integration stage."""

    solver_name: str
    uses_existing_solver: bool
    order: int
    implicit: bool
    compatible_structures: tuple[str, ...] = field(default_factory=tuple)
    transaction_controls: tuple[str, ...] = field(default_factory=tuple)
    references: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable compatibility record."""

        return {
            "solver_name": self.solver_name,
            "uses_existing_solver": self.uses_existing_solver,
            "order": self.order,
            "implicit": self.implicit,
            "compatible_structures": list(self.compatible_structures),
            "transaction_controls": list(self.transaction_controls),
            "references": list(self.references),
        }


CouplingMode = Literal[
    "direct",
    "lagged",
    "lie",
    "strang_half",
    "strang_full",
    "explicit_boundary",
]

RuntimeOwner = Literal["solver_registry", "relic.stochastic", "simulator"]

_STOCHASTIC_SOLVER_NAMES = {
    "euler_maruyama",
    "EulerMaruyamaSolver",
    "gillespie",
    "GillespieSSA",
}


@dataclass(frozen=True)
class IntegrationStage:
    """One structure-aware time-integration stage.

    References:
      - Hairer, Lubich, and Wanner (2006), Geometric Numerical Integration.
      - Hundsdorfer and Verwer (2003), operator splitting for ADR systems.
    Mathematical expression:
      x_{k+1} = Phi_i(alpha_i dt, x_k), where Phi_i is the solver-backed
      flow for one reversible, dissipative, transport, reaction, or boundary
      operator stage.
    Parameter meanings:
      solver_name: registered solver of the stage flow Phi_i (required).
      structure_kind: structure the stage preserves, e.g. iphs (required).
      dt_fraction: alpha_i in the stage time increment alpha_i dt (required;
        0 for an algebraic stage).
      coupling_mode: direct, strang_half or strang_full (required).
      depends_on: same-step dependencies that must precede this stage.
      lagged_dependencies: dependencies excluded from the same-step order
        (declaration only; the read time is set by the runtime binding).
    """

    stage_id: str
    operator_kind: str
    solver_name: str = field(kw_only=True)
    structure_kind: str = field(kw_only=True)
    dt_fraction: float = field(kw_only=True)
    advances_time: bool = True
    coupling_mode: CouplingMode = field(kw_only=True)
    runtime_owner: RuntimeOwner = "solver_registry"
    reads: tuple[str, ...] = field(default_factory=tuple)
    writes: tuple[str, ...] = field(default_factory=tuple)
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    lagged_dependencies: tuple[str, ...] = field(default_factory=tuple)
    validation_hooks: tuple[str, ...] = field(default_factory=tuple)
    stage_metadata: dict[str, Any] = field(default_factory=dict)
    references: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.stage_id.strip():
            raise ValueError("stage_id must be a non-empty string.")
        if not self.operator_kind.strip():
            raise ValueError("operator_kind must be a non-empty string.")
        if self.advances_time and self.dt_fraction <= 0.0:
            raise ValueError("time-advancing stages require positive dt_fraction.")
        if not self.advances_time and self.dt_fraction != 0.0:
            raise ValueError("algebraic stages require dt_fraction=0.")
        if self.runtime_owner not in {"solver_registry", "relic.stochastic", "simulator"}:
            raise ValueError("runtime_owner is not recognized.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable stage record."""

        return {
            "stage_id": self.stage_id,
            "operator_kind": self.operator_kind,
            "solver_name": self.solver_name,
            "structure_kind": self.structure_kind,
            "dt_fraction": self.dt_fraction,
            "advances_time": self.advances_time,
            "coupling_mode": self.coupling_mode,
            "runtime_owner": self.runtime_owner,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "depends_on": list(self.depends_on),
            "lagged_dependencies": list(self.lagged_dependencies),
            "validation_hooks": list(self.validation_hooks),
            "stage_metadata": dict(self.stage_metadata),
            "references": list(self.references),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntegrationStage":
        """Build one stage from a strict JSON-safe mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("IntegrationStage payload must be a mapping.")
        allowed = {
            "stage_id",
            "operator_kind",
            "solver_name",
            "structure_kind",
            "dt_fraction",
            "advances_time",
            "coupling_mode",
            "runtime_owner",
            "reads",
            "writes",
            "depends_on",
            "lagged_dependencies",
            "validation_hooks",
            "stage_metadata",
            "references",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                "IntegrationStage contains unknown fields: " + ",".join(unknown)
            )
        missing = sorted(
            {
                "stage_id",
                "operator_kind",
                "solver_name",
                "structure_kind",
                "dt_fraction",
                "coupling_mode",
            }
            - set(payload)
        )
        if missing:
            raise ValueError(
                "IntegrationStage is missing required fields: " + ",".join(missing)
            )
        values = dict(payload)
        for key in (
            "reads",
            "writes",
            "depends_on",
            "lagged_dependencies",
            "validation_hooks",
            "references",
        ):
            if key in values:
                sequence = values[key]
                if not isinstance(sequence, (list, tuple)):
                    raise ValueError(f"IntegrationStage field '{key}' must be a list")
                values[key] = tuple(sequence)
        if "stage_metadata" in values:
            metadata = values["stage_metadata"]
            if not isinstance(metadata, Mapping):
                raise ValueError("IntegrationStage stage_metadata must be a mapping")
            values["stage_metadata"] = dict(metadata)
        return cls(**values)


_NONLINEAR_STAGE_METADATA = {
    "runtime_node_id",
    "unknown_variables",
    "residual_id",
    "bracket_source",
    "requires_sign_change",
    "convergence_required",
    "failure_policy",
    "execution_binding",
    "solver_parameter_keys",
    "bracket_parameter_keys",
}


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_parameter_key_sequence(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(_is_nonempty_string(item) and "." in item for item in value)
    )


def _validate_nonlinear_algebraic_stage(
    stage: IntegrationStage,
    issues: list[str],
    runtime_node_ids: Optional[set[str]],
) -> None:
    """Validate one registry-backed, metadata-only scalar algebraic closure."""
    prefix = f"nonlinear stage '{stage.stage_id}'"
    if stage.runtime_owner != "solver_registry":
        issues.append(f"{prefix} must use runtime_owner='solver_registry'")
    if SolverRegistry.is_registered(stage.solver_name):
        entry = SolverRegistry.get_entry(stage.solver_name)
        if entry.solver_family is not SolverFamily.NONLINEAR:
            issues.append(f"{prefix} solver family must be NONLINEAR")
    if stage.advances_time:
        issues.append(f"{prefix} must not advance time")
    if stage.structure_kind != "algebraic_constraint":
        issues.append(f"{prefix} structure_kind must be 'algebraic_constraint'")
    if "state_domain_check" not in stage.validation_hooks:
        issues.append(f"{prefix} must declare state_domain_check")

    metadata = stage.stage_metadata
    missing = sorted(_NONLINEAR_STAGE_METADATA - set(metadata))
    if missing:
        issues.append(f"{prefix} missing metadata: " + ",".join(missing))
        return

    runtime_node_id = metadata["runtime_node_id"]
    if not _is_nonempty_string(runtime_node_id):
        issues.append(f"{prefix} runtime_node_id must be a non-empty string")
    elif runtime_node_ids is not None and runtime_node_id not in runtime_node_ids:
        issues.append(f"{prefix} runtime node '{runtime_node_id}' is not in GraphIR")

    unknown_variables = metadata["unknown_variables"]
    if not isinstance(unknown_variables, (list, tuple)) or len(unknown_variables) != 1:
        issues.append(f"{prefix} unknown_variables must contain exactly one variable")
    elif unknown_variables[0] not in stage.writes:
        issues.append(f"{prefix} unknown_variables must be included in writes")

    for key in ("residual_id",):
        if not _is_nonempty_string(metadata[key]):
            issues.append(f"{prefix} {key} must be a non-empty string")
    if metadata["bracket_source"] not in {
        "parameter_manager",
        "state_and_parameter_manager",
    }:
        issues.append(f"{prefix} bracket_source is not recognized")
    if metadata["requires_sign_change"] is not True:
        issues.append(f"{prefix} requires_sign_change must be true")
    if metadata["convergence_required"] is not True:
        issues.append(f"{prefix} convergence_required must be true")
    if metadata["failure_policy"] not in {"raise", "reject_step"}:
        issues.append(f"{prefix} failure_policy must be 'raise' or 'reject_step'")
    if metadata["execution_binding"] != "functor_forward":
        issues.append(f"{prefix} execution_binding must be 'functor_forward'")
    for key in ("solver_parameter_keys", "bracket_parameter_keys"):
        if not _is_parameter_key_sequence(metadata[key]):
            issues.append(f"{prefix} {key} must contain ParameterManager keys")

    direct_numeric_configuration = {
        "max_iterations",
        "rtol",
        "atol",
        "lower",
        "upper",
    }.intersection(metadata)
    if direct_numeric_configuration:
        issues.append(
            f"{prefix} contains direct numeric configuration: "
            + ",".join(sorted(direct_numeric_configuration))
        )


@dataclass(frozen=True)
class IntegrationPlan:
    """Validated execution plan for structure-aware stage splitting."""

    plan_id: str
    stages: tuple[IntegrationStage, ...]
    scheme: Literal["single", "lie", "strang", "custom"] = "custom"
    allow_lagged_cycles: bool = True
    references: tuple[str, ...] = field(default_factory=tuple)

    def validate(
        self,
        *,
        runtime_node_ids: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """Validate only the declared stage graph and splitting metadata.

        This method never executes a solver or proves a runtime binding.  A
        structurally valid plan therefore remains unverified and not runtime
        ready until the canonical Simulator/GraphIR execution path supplies
        executable evidence.
        """

        issues: list[str] = []
        if not self.plan_id.strip():
            issues.append("plan_id is required")
        if not self.stages:
            issues.append("at least one IntegrationStage is required")

        stage_ids = [stage.stage_id for stage in self.stages]
        duplicate_ids = sorted({stage_id for stage_id in stage_ids if stage_ids.count(stage_id) > 1})
        if duplicate_ids:
            issues.append("duplicate stage_id values: " + ",".join(duplicate_ids))

        stage_map = {stage.stage_id: stage for stage in self.stages}
        direct_edges: dict[str, set[str]] = {stage.stage_id: set() for stage in self.stages}
        lagged_edge_count = 0
        for stage in self.stages:
            for dependency in stage.depends_on:
                if dependency not in stage_map:
                    issues.append(f"stage '{stage.stage_id}' depends on unknown stage '{dependency}'")
                    continue
                is_lagged = dependency in stage.lagged_dependencies or stage.coupling_mode == "lagged"
                if is_lagged and self.allow_lagged_cycles:
                    lagged_edge_count += 1
                    continue
                direct_edges[dependency].add(stage.stage_id)

        order, cycle_detected = _toposort_stages(stage_ids, direct_edges)
        if cycle_detected:
            issues.append("direct stage dependency cycle detected")

        unregistered = [
            stage.solver_name
            for stage in self.stages
            if stage.runtime_owner == "solver_registry"
            and stage.solver_name
            and not SolverRegistry.is_registered(stage.solver_name)
        ]
        if unregistered:
            issues.append("unregistered solver_name values: " + ",".join(sorted(set(unregistered))))

        invalid_stochastic = [
            stage.solver_name
            for stage in self.stages
            if stage.runtime_owner == "relic.stochastic"
            and stage.solver_name
            and stage.solver_name not in _STOCHASTIC_SOLVER_NAMES
        ]
        if invalid_stochastic:
            issues.append(
                "invalid relic.stochastic solver_name values: "
                + ",".join(sorted(set(invalid_stochastic)))
            )

        stochastic_stage_kinds = {
            "stochastic_diffusion",
            "conservative_noise_flux",
            "jump_reaction",
        }
        for stage in self.stages:
            if stage.operator_kind in stochastic_stage_kinds and stage.runtime_owner != "relic.stochastic":
                issues.append(
                    f"stage '{stage.stage_id}' must use runtime_owner='relic.stochastic'"
                )
            if stage.operator_kind == "constraint_projection":
                required = {"constraint_id", "residual_route"}
                missing = sorted(required - set(stage.stage_metadata))
                if missing:
                    issues.append(
                        f"constraint_projection stage '{stage.stage_id}' missing metadata: "
                        + ",".join(missing)
                    )
            if stage.operator_kind == "stabilized_advection":
                required = {"scheme", "cfl_metadata", "boundary_flux_route"}
                missing = sorted(required - set(stage.stage_metadata))
                if missing:
                    issues.append(
                        f"stabilized_advection stage '{stage.stage_id}' missing metadata: "
                        + ",".join(missing)
                    )
            if stage.operator_kind == "nonlinear_algebraic_closure":
                _validate_nonlinear_algebraic_stage(
                    stage,
                    issues,
                    runtime_node_ids,
                )

        if self.scheme == "strang" and len(self.stages) >= 2:
            first = self.stages[0]
            last = self.stages[-1]
            if first.operator_kind != last.operator_kind:
                issues.append("Strang plan endpoints must use the same operator_kind")
            if abs(first.dt_fraction - last.dt_fraction) > 1.0e-12:
                issues.append("Strang endpoint dt_fraction values must match")
        elif self.scheme == "strang":
            issues.append("Strang plan requires at least two endpoint stages")

        structurally_validated = not issues
        return {
            "passed": structurally_validated,
            "structurally_validated": structurally_validated,
            "plan_id": self.plan_id,
            "scheme": self.scheme,
            "issue_count": len(issues),
            "issues": issues,
            "stage_count": len(self.stages),
            "lagged_edge_count": lagged_edge_count,
            "topological_order": order,
            "verified": False,
            "runtime_ready": False,
            "execution_semantics": "metadata_only",
            "solver_execution_count": 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plan record."""

        return {
            "plan_id": self.plan_id,
            "scheme": self.scheme,
            "allow_lagged_cycles": self.allow_lagged_cycles,
            "stages": [stage.to_dict() for stage in self.stages],
            "references": list(self.references),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntegrationPlan":
        """Build a plan from JSON-safe data and discard any self-attested audit."""
        if not isinstance(payload, Mapping):
            raise TypeError("IntegrationPlan payload must be a mapping.")
        allowed = {
            "plan_id",
            "scheme",
            "allow_lagged_cycles",
            "stages",
            "references",
            "audit",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                "IntegrationPlan contains unknown fields: " + ",".join(unknown)
            )
        missing = sorted({"plan_id", "stages"} - set(payload))
        if missing:
            raise ValueError(
                "IntegrationPlan is missing required fields: " + ",".join(missing)
            )
        raw_stages = payload["stages"]
        if not isinstance(raw_stages, (list, tuple)):
            raise ValueError("IntegrationPlan stages must be a list")
        raw_references = payload.get("references", ())
        if not isinstance(raw_references, (list, tuple)):
            raise ValueError("IntegrationPlan references must be a list")
        scheme = payload.get("scheme", "custom")
        if scheme not in {"single", "lie", "strang", "custom"}:
            raise ValueError("IntegrationPlan scheme is not recognized")
        return cls(
            plan_id=str(payload["plan_id"]),
            stages=tuple(IntegrationStage.from_dict(stage) for stage in raw_stages),
            scheme=scheme,
            allow_lagged_cycles=bool(payload.get("allow_lagged_cycles", True)),
            references=tuple(raw_references),
        )


def _toposort_stages(
    stage_ids: list[str],
    edges: dict[str, set[str]],
) -> tuple[list[str], bool]:
    indegree = {stage_id: 0 for stage_id in stage_ids}
    for source, targets in edges.items():
        if source not in indegree:
            continue
        for target in targets:
            if target in indegree:
                indegree[target] += 1

    queue = sorted(stage_id for stage_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for target in sorted(edges.get(current, ())):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    return order, len(order) != len(stage_ids)


def build_strang_iphs_plan(
    *,
    plan_id: str,
    reversible_operator: str,
    dissipative_operator: str,
    solver_name: str,
    validation_hooks: tuple[str, ...] = ("iphs_structure", "entropy_step"),
) -> IntegrationPlan:
    """Build a three-stage Strang plan for one IPHS split.

    References:
      - Strang (1968), construction and comparison of difference schemes.
      - Hairer, Lubich, and Wanner (2006), symmetric splitting methods.
    Mathematical expression:
      Phi(dt) = Phi_A(dt/2) Phi_B(dt) Phi_A(dt/2).
    Parameter meanings:
      reversible_operator: operator A, usually the skew IPHS J contribution.
      dissipative_operator: operator B, usually the PSD IPHS M contribution.
      solver_name: registered solver of every stage flow (required).
    """

    return IntegrationPlan(
        plan_id=plan_id,
        scheme="strang",
        stages=(
            IntegrationStage(
                stage_id="reversible_half_0",
                operator_kind=reversible_operator,
                solver_name=solver_name,
                structure_kind="iphs",
                dt_fraction=0.5,
                coupling_mode="strang_half",
                validation_hooks=validation_hooks,
            ),
            IntegrationStage(
                stage_id="dissipative_full",
                operator_kind=dissipative_operator,
                solver_name=solver_name,
                structure_kind="iphs",
                dt_fraction=1.0,
                coupling_mode="strang_full",
                depends_on=("reversible_half_0",),
                validation_hooks=validation_hooks,
            ),
            IntegrationStage(
                stage_id="reversible_half_1",
                operator_kind=reversible_operator,
                solver_name=solver_name,
                structure_kind="iphs",
                dt_fraction=0.5,
                coupling_mode="strang_half",
                depends_on=("dissipative_full",),
                validation_hooks=validation_hooks,
            ),
        ),
        references=(
            "Strang (1968), On the construction and comparison of difference schemes",
            "Hairer, Lubich, and Wanner (2006), Geometric Numerical Integration",
        ),
    )


def build_stochastic_sdae_plan(
    *,
    plan_id: str,
    deterministic_operator: str = "iphs_drift",
    include_conservative_noise_flux: bool = True,
    include_constraint_projection: bool = False,
    include_jump_reaction: bool = False,
    include_stabilized_advection: bool = False,
) -> IntegrationPlan:
    """Build metadata stages for a stochastic IPHS/SDAE split.

    References:
      - Gillespie (1976), stochastic simulation algorithm.
      - Hairer, Lubich, and Wanner (2006), geometric integration.
    Mathematical expression:
      dx = F_iphs dt + Sigma_h dW_t + R_h dN_t with optional A_h x = 0.
    Parameter meanings:
      deterministic_operator: existing deterministic IPHS drift stage.
      include_conservative_noise_flux: add flux-routed stochastic SDE stage.
      include_constraint_projection: add descriptor projection stage.
      include_jump_reaction: add Gillespie/SSA jump stage.
      include_stabilized_advection: add conservative advection metadata stage.
    """

    stages: list[IntegrationStage] = [
        IntegrationStage(
            stage_id="deterministic_iphs_drift",
            operator_kind=deterministic_operator,
            solver_name="implicit_midpoint",
            structure_kind="iphs",
            dt_fraction=1.0,
            coupling_mode="direct",
            validation_hooks=("iphs_structure", "entropy_step"),
        )
    ]
    previous = "deterministic_iphs_drift"
    if include_stabilized_advection:
        stages.append(
            IntegrationStage(
                stage_id="stabilized_advection",
                operator_kind="stabilized_advection",
                solver_name="implicit_midpoint",
                structure_kind="iphs",
                dt_fraction=1.0,
                coupling_mode="direct",
                depends_on=(previous,),
                validation_hooks=("state_domain",),
                stage_metadata={
                    "scheme": "declared_by_functor",
                    "cfl_metadata": "declared_by_functor",
                    "boundary_flux_route": "declared_by_functor",
                },
            )
        )
        previous = "stabilized_advection"
    if include_conservative_noise_flux:
        stages.append(
            IntegrationStage(
                stage_id="conservative_noise_flux",
                operator_kind="conservative_noise_flux",
                solver_name="euler_maruyama",
                structure_kind="iphs",
                dt_fraction=1.0,
                coupling_mode="direct",
                runtime_owner="relic.stochastic",
                depends_on=(previous,),
                validation_hooks=("stochastic_sdae_structure", "state_domain"),
                stage_metadata={"noise_route": "flux_space_dec_topology"},
            )
        )
        previous = "conservative_noise_flux"
    if include_jump_reaction:
        stages.append(
            IntegrationStage(
                stage_id="jump_reaction",
                operator_kind="jump_reaction",
                solver_name="gillespie",
                structure_kind="iphs",
                dt_fraction=1.0,
                coupling_mode="direct",
                runtime_owner="relic.stochastic",
                depends_on=(previous,),
                validation_hooks=("stochastic_sdae_structure", "reaction_network"),
                stage_metadata={"jump_map": "declared_by_functor"},
            )
        )
        previous = "jump_reaction"
    if include_constraint_projection:
        stages.append(
            IntegrationStage(
                stage_id="constraint_projection",
                operator_kind="constraint_projection",
                solver_name="implicit_midpoint",
                structure_kind="iphs",
                dt_fraction=1.0,
                coupling_mode="direct",
                depends_on=(previous,),
                validation_hooks=("stochastic_sdae_structure", "state_domain"),
                stage_metadata={
                    "constraint_id": "declared_by_functor",
                    "residual_route": "declared_by_functor",
                },
            )
        )

    return IntegrationPlan(
        plan_id=plan_id,
        scheme="custom",
        stages=tuple(stages),
        references=(
            "Gillespie (1976), A general method for numerically simulating stochastic reactions",
            "Hairer, Lubich, and Wanner (2006), Geometric Numerical Integration",
        ),
    )


def get_implicit_midpoint_compatibility_spec() -> IntegrationStageCompatibilitySpec:
    """Return the RELIC-IPHS-DEC compatibility record for implicit midpoint.

    References:
      - Hairer, Lubich, and Wanner (2006), Geometric Numerical Integration.
      - Leimkuhler and Reich (2004), Simulating Hamiltonian Dynamics.
    Mathematical expression:
      y_next = y + dt * f((y + y_next) / 2, t + dt / 2).
      This symmetric stage is compatible with Hamiltonian/IPHS audits because
      the midpoint effort can be checked before a transactional commit.
    Parameter meanings:
      solver_name: existing SolverRegistry key.
      transaction_controls: existing StateManager checkpoint and restore hooks.
    """

    return IntegrationStageCompatibilitySpec(
        solver_name="implicit_midpoint",
        uses_existing_solver=SolverRegistry.is_registered("implicit_midpoint"),
        order=2,
        implicit=True,
        compatible_structures=("hamiltonian", "iphs", "dec"),
        transaction_controls=("checkpoint_restore", "post_step_validation"),
        references=(
            "Hairer, Lubich, and Wanner (2006), Geometric Numerical Integration",
            "Leimkuhler and Reich (2004), Simulating Hamiltonian Dynamics",
        ),
    )


def transactional_implicit_midpoint_step(
    ode_func: Callable[..., torch.Tensor],
    y_current: torch.Tensor,
    t_current: float,
    dt: float,
    *,
    state_manager: Optional[Any] = None,
    solver: Optional[ImplicitMidpointSolver] = None,
    solver_config: Optional[dict[str, Any]] = None,
    commit_result: Optional[Callable[[SolverResult], None]] = None,
    validator: Optional[Callable[[SolverResult], bool]] = None,
    rollback_on_failure: bool = True,
    **kwargs: Any,
) -> SolverResult:
    """Run one implicit-midpoint step with optional StateManager rollback.

    References:
      - Hairer, Lubich, and Wanner (2006), Geometric Numerical Integration.
      - RELIC StateManager checkpoint/restore transaction contract.
    Mathematical expression:
      y_next = y + dt * f((y + y_next) / 2, t + dt / 2), followed by
      accept(y_next) only if status == SUCCESS and validator(y_next) is true.
    Parameter meanings:
      ode_func: right-hand side evaluated at the implicit midpoint.
      y_current: current tensor state.
      t_current: current time.
      dt: positive time step.
      state_manager: optional transaction owner supplying checkpoint/restore.
      commit_result: optional callback that writes the candidate result.
      validator: optional structural validation callback.
    """

    active_solver = solver or ImplicitMidpointSolver(config=solver_config or {})
    snapshot = state_manager.checkpoint() if state_manager is not None else None
    restored = False
    validation_passed = False

    try:
        result = active_solver.solve(ode_func, y_current, t_current, dt, **kwargs)
        validation_passed = result.status == SolverStatus.SUCCESS

        if validation_passed and commit_result is not None:
            commit_result(result)

        if validation_passed and validator is not None:
            validation_passed = bool(validator(result))

        if not validation_passed:
            if result.status == SolverStatus.SUCCESS:
                result.status = SolverStatus.INVALID_INPUT
            if snapshot is not None and rollback_on_failure:
                state_manager.restore(snapshot)
                restored = True

        result.metadata.update(
            {
                "compatibility": get_implicit_midpoint_compatibility_spec().to_dict(),
                "transaction_restored": restored,
                "validation_passed": validation_passed,
            }
        )
        return result
    except Exception:
        if snapshot is not None and rollback_on_failure:
            state_manager.restore(snapshot)
        raise


__all__ = [
    "CouplingMode",
    "IntegrationPlan",
    "IntegrationStage",
    "IntegrationStageCompatibilitySpec",
    "RuntimeOwner",
    "build_strang_iphs_plan",
    "build_stochastic_sdae_plan",
    "get_implicit_midpoint_compatibility_spec",
    "transactional_implicit_midpoint_step",
]
