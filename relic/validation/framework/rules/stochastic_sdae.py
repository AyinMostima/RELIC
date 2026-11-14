"""Validation rules for stochastic IPHS/SDAE structural declarations."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, get_args

import torch

from relic.computation.operators.stochastic_sdae import (
    ALEGeometryCorrectionSpec,
    ConservativeNoiseFluxSpec,
    DescriptorConstraintSpec,
    HybridTopologyTransitionSpec,
    JumpResetSpec,
    MemoryKernelSpec,
    NoiseCovarianceSpec,
    StabilizedAdvectionSpec,
    StochasticTermSpec,
)
from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.iphs import OpenSystemRoute
from relic.data.standards.dimensionality import dimensionality_from_unit

from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)

_ALLOWED_CLASSIFICATIONS = {
    "fdt_bound",
    "external_forcing",
    "observation_error",
    "parameter_uncertainty",
    "jump_reset",
    "hybrid_topology",
}

_ALLOWED_NOISE_TYPES = {
    "gaussian",
    "marked_poisson",
    "poisson",
    "white",
    "wiener",
}

_ALLOWED_SOLVER_METHODS = {
    "direct_ssa",
    "euler_maruyama",
    "gillespie",
    "next_reaction",
}

_CONTINUOUS_CLASSIFICATIONS = {
    "external_forcing",
    "fdt_bound",
    "observation_error",
    "parameter_uncertainty",
}

_CONTINUOUS_NOISE_TYPES = {"gaussian", "white", "wiener"}
_CONTINUOUS_SOLVERS = {"euler_maruyama"}
_MARKED_JUMP_NOISE_TYPES = {"marked_poisson", "poisson"}
_MARKED_JUMP_SOLVERS = {"direct_ssa", "gillespie", "next_reaction"}

_ALLOWED_PASSIVITY_DECLARATIONS = {
    "dissipative-realization",
    "positive-real-realization",
    "storage-function-certificate",
}

_ALLOWED_OPEN_SYSTEM_ROUTES = frozenset(get_args(OpenSystemRoute))

_UNBOUND_SEMANTIC_EXPECTATIONS = {
    "continuous_sde": "implemented GraphIR/Functor drift-diffusion binding [N]",
    "marked_jump": "implemented GraphIR/Functor marked-jump execution binding [N]",
    "memory": "implemented GraphIR/Functor passive-memory execution binding [N]",
    "ale": "implemented GraphIR/Functor ALE geometry-power binding [N]",
    "descriptor_dae": "implemented GraphIR/Functor DAE projection binding [N]",
    "topology_change": (
        "implemented GraphIR/Functor topology-transition binding on the canonical "
        "provider epoch owner [N]"
    ),
}

# The single provider-epoch owner (D01) and the accepted-step delivery queue
# that commits topology requests (G09-TOPO).
_GRAPH_EPOCH_OWNERS = frozenset({
    "relic.core.simulator.registration",
    "relic.core.simulator.registration._ProviderLifecycleState",
    "relic.core.simulator.registration.apply_topology_transition",
    "relic.core.simulator.main.RELICSimulator",
})
_TOPOLOGY_EVENT_QUEUE_OWNERS = frozenset({
    "relic.core.simulator.execution_loop.retry_external_deliveries",
    "relic.core.simulator.main.RELICSimulator._pending_external_deliveries",
})

_SEMANTIC_COUNT_KEYS = {
    "continuous_sde": "stochastic_term_count",
    "marked_jump": "jump_reset_count",
    "memory": "memory_kernel_count",
    "ale": "ale_geometry_correction_count",
    "descriptor_dae": "descriptor_constraint_count",
    "topology_change": "hybrid_topology_transition_count",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _lower_token(value: Any) -> str:
    return _clean_text(value).replace("/", "\\").lower()


def _iter_payloads(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _numeric_tensor_or_none(value: Any) -> torch.Tensor | None:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    if torch.is_complex(tensor) or not bool(torch.isfinite(tensor).all().item()):
        return None
    return tensor


def _detail(variable: str, expected: str, value: Any) -> ValidationDetail:
    return ValidationDetail(variable_name=variable, expected=expected, value=str(value))


def _term_from_payload(value: Any) -> StochasticTermSpec | None:
    if isinstance(value, StochasticTermSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return StochasticTermSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _covariance_from_payload(value: Any) -> NoiseCovarianceSpec | None:
    if isinstance(value, NoiseCovarianceSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return NoiseCovarianceSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _flux_from_payload(value: Any) -> ConservativeNoiseFluxSpec | None:
    if isinstance(value, ConservativeNoiseFluxSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return ConservativeNoiseFluxSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _constraint_from_payload(value: Any) -> DescriptorConstraintSpec | None:
    if isinstance(value, DescriptorConstraintSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return DescriptorConstraintSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _jump_from_payload(value: Any) -> JumpResetSpec | None:
    if isinstance(value, JumpResetSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return JumpResetSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _advection_from_payload(value: Any) -> StabilizedAdvectionSpec | None:
    if isinstance(value, StabilizedAdvectionSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return StabilizedAdvectionSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _memory_from_payload(value: Any) -> MemoryKernelSpec | None:
    if isinstance(value, MemoryKernelSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return MemoryKernelSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _ale_from_payload(value: Any) -> ALEGeometryCorrectionSpec | None:
    if isinstance(value, ALEGeometryCorrectionSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return ALEGeometryCorrectionSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _hybrid_from_payload(value: Any) -> HybridTopologyTransitionSpec | None:
    if isinstance(value, HybridTopologyTransitionSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return HybridTopologyTransitionSpec.from_mapping(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _owner_is_existing(owner: str) -> bool:
    parts = _clean_text(owner).replace("\\", ".").split(".")
    for split_index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            value: Any = importlib.import_module(module_name)
        except (ImportError, ValueError):
            continue
        for attribute in parts[split_index:]:
            if not hasattr(value, attribute):
                return False
            value = getattr(value, attribute)
        return isinstance(value, ModuleType) or callable(value)
    return False


def _stochastic_owner_is_existing(owner: str) -> bool:
    canonical_owner = _clean_text(owner).replace("\\", ".")
    if not (
        canonical_owner == "relic.stochastic"
        or canonical_owner.startswith("relic.stochastic.")
    ):
        return False
    return _owner_is_existing(canonical_owner)


def _resolve_owner_object(owner: str) -> Any:
    """Import a dotted owner path and return the resolved object, or None."""
    parts = _clean_text(owner).replace("\\", ".").split(".")
    if not parts or not parts[0]:
        return None
    for split_index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            value: Any = importlib.import_module(module_name)
        except (ImportError, ValueError):
            continue
        for attribute in parts[split_index:]:
            if not hasattr(value, attribute):
                return None
            value = getattr(value, attribute)
        return value
    return None


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_EXECUTION_BINDING_RNG_SCOPES = frozenset({"functor_extra_state"})


def _validation_test_exists(value: Any) -> bool:
    file_part = _clean_text(value).partition("::")[0].replace("\\", "/")
    if not file_part:
        return False
    try:
        candidate = (_PROJECT_ROOT / file_part).resolve()
        candidate.relative_to((_PROJECT_ROOT / "test").resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return candidate.is_file()


_EXECUTION_BINDING_CONTRACTS: dict[str, dict[str, Any]] = {
    # semantic -> Functor category attribute, admissible solver names (None
    # when the semantic has no relic.stochastic solver owner), the payload
    # section whose declared states must cover the bound state_vars, and the
    # ownership-scope field with its admissible values.
    "continuous_sde": {
        "category_attribute": "stochastic_category",
        "solver_names": frozenset({"euler_maruyama"}),
        "declared_state_source": "stochastic_terms",
        "scope_field": "rng_state_scope",
        "scopes": _EXECUTION_BINDING_RNG_SCOPES,
        "scope_expectation": "rng_state_scope=functor_extra_state (Functor-owned substream in module extra state)",
    },
    "marked_jump": {
        "category_attribute": "stochastic_category",
        "solver_names": _MARKED_JUMP_SOLVERS,
        "declared_state_source": "jump_resets",
        "scope_field": "rng_state_scope",
        "scopes": _EXECUTION_BINDING_RNG_SCOPES,
        "scope_expectation": "rng_state_scope=functor_extra_state (Functor-owned substream in module extra state)",
    },
    "memory": {
        "category_attribute": "memory_category",
        "solver_names": None,
        "declared_state_source": "memory_kernels",
        "scope_field": "history_state_scope",
        "scopes": frozenset({"state_manager", "functor_extra_state"}),
        "scope_expectation": (
            "history_state_scope in {state_manager, functor_extra_state} "
            "(hidden realization or history owned by canonical state or module extra state)"
        ),
    },
    "ale": {
        "category_attribute": "geometry_category",
        "solver_names": None,
        "declared_state_source": "ale_geometry_corrections",
        "scope_field": "geometry_state_scope",
        "scopes": frozenset({"state_manager"}),
        "scope_expectation": (
            "geometry_state_scope=state_manager (moving-geometry state owned by canonical state)"
        ),
    },
    "topology_change": {
        "category_attribute": "topology_category",
        "solver_names": None,
        "declared_state_source": "hybrid_topology_transitions",
        "scope_field": "topology_state_scope",
        "scopes": frozenset({"state_manager"}),
        "scope_expectation": (
            "topology_state_scope=state_manager (remapped states owned by canonical state; "
            "connectivity committed by the registration owner at the accepted-step boundary)"
        ),
    },
    "descriptor_dae": {
        "category_attribute": "constraint_category",
        "solver_names": None,
        "declared_state_source": "descriptor_constraints",
        "scope_field": "algebraic_state_scope",
        "scopes": frozenset({"state_manager"}),
        "scope_expectation": (
            "algebraic_state_scope=state_manager (algebraic states owned by canonical state and re-solved each step)"
        ),
    },
}


def _declared_binding_states(payload: Mapping[str, Any], source: str) -> set[str]:
    """Return the state names declared by the terms, jump resets, or memory kernels."""
    declared: set[str] = set()
    if source == "stochastic_terms":
        items = payload.get("stochastic_terms", payload.get("terms", ()))
    elif source == "memory_kernels":
        items = payload.get("memory_kernels", payload.get("memory_kernel", payload.get("memory_kernel_map", ())))
    elif source == "ale_geometry_corrections":
        items = payload.get(
            "ale_geometry_corrections",
            payload.get("ale_geometry_correction", payload.get("ale_geometry_map", ())),
        )
    elif source == "descriptor_constraints":
        items = payload.get(
            "descriptor_constraints",
            payload.get("descriptor_constraint", payload.get("descriptor_constraint_map", ())),
        )
    elif source == "hybrid_topology_transitions":
        items = payload.get(
            "hybrid_topology_transitions",
            payload.get("hybrid_topology_transition", payload.get("hybrid_topology_map", ())),
        )
    else:
        items = payload.get("jump_resets", payload.get("jump_reset", payload.get("jump_reset_map", ())))
    for raw_item in _iter_payloads(items):
        item = _as_mapping(raw_item)
        declared.update(
            _clean_text(name) for name in _iter_payloads(item.get("state_vars", ())) if _clean_text(name)
        )
    return declared


def _functor_class_states(functor_class: Any) -> tuple[str, ...]:
    """Return the state names a Functor class declares as its own."""
    if not isinstance(functor_class, type):
        return ()
    declared = getattr(functor_class, "STATE_VARIABLES", None)
    if isinstance(declared, (list, tuple)):
        return tuple(str(name) for name in declared)
    single = getattr(functor_class, "STATE_VARIABLE", None)
    return (str(single),) if isinstance(single, str) else ()


def _validate_execution_binding(
    semantic: str,
    payload: Mapping[str, Any],
    *,
    details: list[ValidationDetail],
) -> bool:
    """Verify the executable GraphIR/Functor binding of one stochastic semantic.

    References:
      - Kloeden and Platen (1992), ch. 10; Higham (2001): fixed-step
        Euler-Maruyama as the continuous-SDE solver contract.
      - Gillespie (1977): direct SSA as the marked-jump event clock.
      - Revised RELIC foundations: Functor-owned random substream in module
        extra state, accepted-step rollback, and checkpoint restart.
    Mathematical expression:
      dx = a(x,t) dt + b(x,t) dW_t (continuous_sde) or
      dx = h(x^-, z) N(dt, dz) (marked_jump) executed by the bound Functor's
      forward through the bound relic.stochastic owner on the declared states.
    Parameter meanings:
      functor_class: importable Functor whose stochastic_category equals the
      semantic and whose solver_name is admissible; forward_owner: that
      class's forward; solver_owner: existing relic.stochastic class with
      sample; state_vars: bound states declared by the terms or jump resets
      and covering the class state names; rng_state_scope: substream
      ownership; validation_test: existing test under test/.
    Units: none (structural binding).
    Domain/constraints: every field must resolve; a missing binding keeps the
      fail-closed [N] expectation; import resolution alone is not runtime proof,
      the named validation test carries the executed evidence.
    Sign convention: not applicable.
    """

    contract = _EXECUTION_BINDING_CONTRACTS[semantic]
    bindings = _as_mapping(payload.get("execution_bindings"))
    binding = _as_mapping(bindings.get(semantic)) if bindings else {}
    if not binding:
        return False
    variable = f"execution_bindings.{semantic}"
    solver_names: Optional[frozenset[str]] = contract["solver_names"]
    category_attribute: str = contract["category_attribute"]
    consistent = True

    functor_class_name = _clean_text(binding.get("functor_class"))
    functor_class = _resolve_owner_object(functor_class_name)
    solver_ok = solver_names is None or (
        getattr(functor_class, "solver_name", None) in solver_names
    )
    if not (
        isinstance(functor_class, type)
        and getattr(functor_class, category_attribute, None) == semantic
        and solver_ok
        and callable(getattr(functor_class, "forward", None))
        and callable(getattr(functor_class, "get_extra_state", None))
    ):
        solver_text = (
            f", solver_name in {sorted(solver_names)}" if solver_names is not None else ""
        )
        details.append(
            _detail(
                variable,
                f"functor_class is an importable Functor with {category_attribute} "
                f"{semantic}{solver_text}, forward, and get_extra_state",
                functor_class_name or "missing",
            )
        )
        consistent = False

    forward_owner = _clean_text(binding.get("forward_owner"))
    if (
        not functor_class_name
        or forward_owner != f"{functor_class_name}.forward"
        or not callable(_resolve_owner_object(forward_owner))
    ):
        details.append(
            _detail(
                variable,
                "forward_owner is the bound functor_class forward",
                forward_owner or "missing",
            )
        )
        consistent = False

    if solver_names is not None:
        solver_owner = _clean_text(binding.get("solver_owner"))
        solver = _resolve_owner_object(solver_owner)
        if not (
            solver_owner.startswith("relic.stochastic.")
            and isinstance(solver, type)
            and callable(getattr(solver, "sample", None))
        ):
            details.append(
                _detail(
                    variable,
                    "solver_owner is an existing relic.stochastic solver class with sample",
                    solver_owner or "missing",
                )
            )
            consistent = False

    bound_state_vars = tuple(
        _clean_text(item) for item in _iter_payloads(binding.get("state_vars", ()))
        if _clean_text(item)
    )
    declared_state_vars = _declared_binding_states(payload, contract["declared_state_source"])
    class_states = _functor_class_states(functor_class)
    if (
        not bound_state_vars
        or not set(bound_state_vars).issubset(declared_state_vars)
        or not set(class_states).issubset(bound_state_vars)
    ):
        details.append(
            _detail(
                variable,
                f"state_vars bound to declared {contract['declared_state_source']} "
                "states and covering the Functor state names",
                ",".join(bound_state_vars) or "missing",
            )
        )
        consistent = False

    scope_value = _clean_text(binding.get(contract["scope_field"]))
    if scope_value not in contract["scopes"]:
        details.append(
            _detail(
                variable,
                contract["scope_expectation"],
                scope_value or "missing",
            )
        )
        consistent = False

    if not _validation_test_exists(binding.get("validation_test")):
        details.append(
            _detail(
                variable,
                "validation_test names an existing test file under test/",
                _clean_text(binding.get("validation_test")) or "missing",
            )
        )
        consistent = False
    return consistent


def _normalize_required_semantics(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        candidates = (value,)
    elif isinstance(value, Iterable):
        candidates = tuple(value)
    else:
        candidates = (value,)
    return {_clean_text(item).lower() for item in candidates if _clean_text(item)}


def _validate_restart_map(
    payload: Mapping[str, Any],
    *,
    map_name: str,
    required_fields: tuple[str, ...],
    details: list[ValidationDetail],
) -> Mapping[str, Any]:
    restart = _as_mapping(payload.get(map_name))
    for field_name in required_fields:
        value = restart.get(field_name)
        if value is None or value == "" or value == () or value == []:
            details.append(
                _detail(map_name, f"{field_name} recorded for restart", "missing")
            )
    checkpoint_owner = _clean_text(restart.get("checkpoint_owner"))
    if checkpoint_owner and checkpoint_owner != "relic.core.simulator.checkpoint":
        details.append(
            _detail(
                map_name,
                "checkpoint_owner=relic.core.simulator.checkpoint",
                checkpoint_owner,
            )
        )
    elif checkpoint_owner and not _owner_is_existing(checkpoint_owner):
        details.append(
            _detail(map_name, "existing simulator checkpoint owner", checkpoint_owner)
        )
    return restart


def _validate_stochastic_terms(payload: Mapping[str, Any], details: list[ValidationDetail]) -> int:
    raw_terms = payload.get("stochastic_terms", payload.get("terms", ()))
    terms = tuple(_term_from_payload(item) for item in _iter_payloads(raw_terms))
    invalid_count = 0
    for index, term in enumerate(terms):
        if term is None:
            invalid_count += 1
            details.append(_detail(f"stochastic_terms[{index}]", "valid StochasticTermSpec", "invalid"))
            continue
        if term.classification not in _ALLOWED_CLASSIFICATIONS:
            details.append(
                _detail(term.name, "recognized stochastic noise classification", term.classification)
            )
        if not _stochastic_owner_is_existing(term.runtime_owner):
            details.append(
                _detail(term.name, "callable or module stochastic owner", term.runtime_owner)
            )
        if term.noise_type and _clean_text(term.noise_type).lower() not in _ALLOWED_NOISE_TYPES:
            details.append(
                _detail(term.name, "recognized stochastic noise_type", term.noise_type)
            )
        if term.solver_name and _clean_text(term.solver_name).lower() not in _ALLOWED_SOLVER_METHODS:
            details.append(
                _detail(term.name, "recognized stochastic solver_method", term.solver_name)
            )
    return len(terms)


def _validate_covariances(payload: Mapping[str, Any], details: list[ValidationDetail]) -> int:
    raw_covariances = payload.get(
        "noise_covariances",
        payload.get("noise_covariance", payload.get("noise_covariance_map", ())),
    )
    covariances = tuple(_covariance_from_payload(item) for item in _iter_payloads(raw_covariances))
    for index, covariance in enumerate(covariances):
        if covariance is None:
            details.append(_detail(f"noise_covariances[{index}]", "valid NoiseCovarianceSpec", "invalid"))
            continue
        audit = covariance.validate()
        if not bool(audit.get("passed", False)):
            details.append(
                _detail(
                    covariance.name,
                    "Sigma_h Sigma_h^T matches declared M_noise relation",
                    "; ".join(str(issue) for issue in audit.get("issues", ())),
                )
            )
    return len(covariances)


def _space_declared(value: Any) -> bool:
    if isinstance(value, CochainSpaceSpec):
        return True
    if isinstance(value, Mapping):
        complex_id = _clean_text(value.get("complex_id"))
        semantic_name = _clean_text(value.get("semantic_name"))
        degree = value.get("degree")
        degree_declared = (
            not isinstance(degree, bool)
            and isinstance(degree, Integral)
            and int(degree) in {0, 1, 2, 3}
        )
        return bool(complex_id and (semantic_name or degree_declared))
    return False


def _validate_fluxes(payload: Mapping[str, Any], details: list[ValidationDetail]) -> int:
    raw_fluxes = payload.get(
        "conservative_noise_fluxes",
        payload.get("conservative_noise_flux", payload.get("conservative_noise_flux_map", ())),
    )
    fluxes = tuple(_flux_from_payload(item) for item in _iter_payloads(raw_fluxes))
    for index, flux in enumerate(fluxes):
        if flux is None:
            details.append(
                _detail(f"conservative_noise_fluxes[{index}]", "valid ConservativeNoiseFluxSpec", "invalid")
            )
            continue
        if not _space_declared(flux.flux_space):
            details.append(_detail(flux.name, "declared flux cochain space", "missing"))
        if not _space_declared(flux.storage_space):
            details.append(_detail(flux.name, "declared storage cochain space", "missing"))
        if not flux.routes_to_storage:
            details.append(_detail(flux.name, "routes_to_storage=true", flux.routes_to_storage))
        if flux.direct_storage_noise:
            details.append(
                _detail(
                    flux.name,
                    "conserved stochastic noise routed through topology, not direct storage increments",
                    "direct_storage_noise=true",
                )
            )
    return len(fluxes)


def _validate_constraints(payload: Mapping[str, Any], details: list[ValidationDetail]) -> int:
    raw_constraints = payload.get(
        "descriptor_constraints",
        payload.get("descriptor_constraint", payload.get("descriptor_constraint_map", ())),
    )
    constraints = tuple(_constraint_from_payload(item) for item in _iter_payloads(raw_constraints))
    for index, constraint in enumerate(constraints):
        if constraint is None:
            details.append(
                _detail(f"descriptor_constraints[{index}]", "valid DescriptorConstraintSpec", "invalid")
            )
            continue
        if constraint.constraint_operator is None:
            details.append(_detail(constraint.name, "declared A_h constraint operator", "missing"))
        elif (
            constraint.constraint_operator.ndim != 2
            or constraint.constraint_operator.numel() == 0
            or torch.is_complex(constraint.constraint_operator)
            or not bool(torch.isfinite(constraint.constraint_operator).all().item())
        ):
            details.append(
                _detail(
                    constraint.name,
                    "non-empty finite rank-2 A_h constraint operator",
                    tuple(constraint.constraint_operator.shape),
                )
            )
        if not _space_declared(constraint.constrained_space):
            details.append(_detail(constraint.name, "declared constrained cochain space", "missing"))
        if dimensionality_from_unit(constraint.units) is None:
            details.append(
                _detail(
                    constraint.name,
                    "parseable descriptor constraint units",
                    constraint.units,
                )
            )
        if constraint.projection_method not in {
            "algebraic_elimination",
            "lagrange_multiplier",
            "null_space_projection",
            "orthogonal_projection",
        }:
            details.append(
                _detail(
                    constraint.name,
                    "recognized descriptor projection method",
                    constraint.projection_method,
                )
            )
        if not constraint.enforces_state_constraint:
            details.append(_detail(constraint.name, "A_h x = 0 compatibility", "not declared"))
        if not constraint.enforces_tangent_constraint:
            details.append(_detail(constraint.name, "A_h dx = 0 compatibility", "not declared"))
        if not constraint.constraint_force_neutral and not constraint.port_accounting:
            details.append(
                _detail(
                    constraint.name,
                    "neutral constraint force or explicit port accounting",
                    "missing port_accounting",
                )
            )
    return len(constraints)


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        return float("inf")
    return float(torch.max(torch.abs(value)).detach().cpu().item())


def _hodge_rate_is_symmetric(dot_hodge: torch.Tensor) -> bool:
    """Relative symmetry test of a declared Hodge rate dot(star_h).

    References:
      - Hirani (2003), Discrete Exterior Calculus; Desbrun, Kanso, and Tong
        (2008), Discrete Differential Forms (symmetric Hodge operators).
    Mathematical expression:
      rho_sym = ||A - A^T||_F / ||A||_F (0 when A = 0) for A = dot(star_h);
      symmetric iff rho_sym <= max(rtol, n eps), the HodgeStarSpec definition.
    Parameter meanings:
      rtol is ParameterManager key validation.iphs.hodge_symmetry_rtol; n is
      the matrix dimension; eps is the machine epsilon of the matrix dtype.
    Units:
      rho_sym, rtol, and n eps are dimensionless; A keeps the declared Hodge
      units per unit time.
    Domain/constraints:
      finite real square A (checked by the caller); the relative form makes the
      test independent of the scale of A.
    Sign convention:
      not applicable; the residual is a nonnegative defect norm.
    """
    from relic.parameters import get_global_manager

    rtol = float(get_global_manager().get("validation.iphs.hodge_symmetry_rtol", required=True))
    norm = float(torch.linalg.matrix_norm(dot_hodge, ord="fro").detach().cpu().item())
    defect = float(
        torch.linalg.matrix_norm(dot_hodge - dot_hodge.transpose(-1, -2), ord="fro").detach().cpu().item()
    )
    if norm == 0.0:
        return defect == 0.0
    effective = max(rtol, float(torch.finfo(dot_hodge.dtype).eps) * float(max(dot_hodge.shape)))
    return defect / norm <= effective


def _validate_jumps(payload: Mapping[str, Any], details: list[ValidationDetail]) -> int:
    raw_jumps = payload.get("jump_resets", payload.get("jump_reset", payload.get("jump_reset_map", ())))
    raw_items = _iter_payloads(raw_jumps)
    jumps = tuple(_jump_from_payload(item) for item in raw_items)
    for index, jump in enumerate(jumps):
        if jump is None:
            details.append(_detail(f"jump_resets[{index}]", "valid JumpResetSpec", "invalid"))
            continue
        raw_jump = _as_mapping(raw_items[index])
        map_tensor = jump.reset_map if jump.reset_map is not None else jump.stoichiometric_N
        map_valid = (
            isinstance(map_tensor, torch.Tensor)
            and map_tensor.ndim == 2
            and map_tensor.numel() > 0
            and not torch.is_complex(map_tensor)
            and bool(torch.isfinite(map_tensor).all().item())
        )
        if not map_valid:
            details.append(
                _detail(
                    jump.name,
                    "non-empty finite rank-2 reset_map or stoichiometric_N",
                    "missing or invalid",
                )
            )
        ledger_behavior = _clean_text(jump.ledger_behavior).lower()
        if ledger_behavior not in {"closed", "open"}:
            details.append(
                _detail(jump.name, "ledger_behavior is closed or open", ledger_behavior)
            )
        if not jump.propensity_keys:
            details.append(_detail(jump.name, "ParameterManager-backed propensity_keys", "missing"))
        tolerance_valid = (
            not isinstance(jump.tolerance, bool)
            and bool(
                torch.isfinite(
                    torch.as_tensor(jump.tolerance, dtype=torch.float64)
                ).item()
            )
            and float(jump.tolerance) >= 0.0
        )
        if not tolerance_valid:
            details.append(_detail(jump.name, "finite nonnegative jump tolerance", jump.tolerance))

        conservation_matrix = jump.conservation_matrix
        conservation_valid = (
            isinstance(conservation_matrix, torch.Tensor)
            and conservation_matrix.ndim == 2
            and conservation_matrix.numel() > 0
            and not torch.is_complex(conservation_matrix)
            and bool(torch.isfinite(conservation_matrix).all().item())
            and not bool(torch.any(torch.all(conservation_matrix == 0, dim=1)).item())
        )
        if ledger_behavior == "closed" and not conservation_valid:
            details.append(
                _detail(
                    jump.name,
                    "closed jump ledger requires non-empty conservation_matrix C_h",
                    "missing or invalid",
                )
            )
        if (
            isinstance(conservation_matrix, torch.Tensor)
            and conservation_matrix.ndim == 2
            and conservation_matrix.numel() > 0
            and bool(torch.isfinite(conservation_matrix).all().item())
            and bool(torch.any(torch.all(conservation_matrix == 0, dim=1)).item())
        ):
            details.append(
                _detail(jump.name, "nonzero conservation rows", "one or more zero rows")
            )

        ledger_projection = None
        if conservation_valid and map_valid:
            assert isinstance(conservation_matrix, torch.Tensor)
            assert isinstance(map_tensor, torch.Tensor)
            if conservation_matrix.shape[1] != map_tensor.shape[0]:
                details.append(
                    _detail(
                        jump.name,
                        "C_h columns match jump state rows",
                        f"{tuple(conservation_matrix.shape)} vs {tuple(map_tensor.shape)}",
                    )
                )
            else:
                ledger_projection = conservation_matrix @ map_tensor
                if not bool(torch.isfinite(ledger_projection).all().item()):
                    details.append(
                        _detail(jump.name, "finite jump ledger projection C_h R_h", "non-finite")
                    )

        if ledger_behavior == "closed" and ledger_projection is not None and tolerance_valid:
            residual = _max_abs(ledger_projection)
            if residual > float(jump.tolerance):
                details.append(
                    ValidationDetail(
                        variable_name=jump.name,
                        expected="C_h R_h = 0 or C_h N_h = 0",
                        value=residual,
                        threshold=jump.tolerance,
                    )
                )
        if ledger_behavior == "open":
            open_route = _clean_text(raw_jump.get("open_system_route"))
            ledger_change = _numeric_tensor_or_none(
                raw_jump.get("open_ledger_change", raw_jump.get("ledger_change", [])),
            )
            ledger_units = _clean_text(raw_jump.get("open_ledger_units"))
            open_contract_valid = (
                open_route in _ALLOWED_OPEN_SYSTEM_ROUTES
                and conservation_valid
                and ledger_projection is not None
                and ledger_change is not None
                and ledger_change.numel() > 0
                and tuple(ledger_change.shape) == tuple(ledger_projection.shape)
                and dimensionality_from_unit(ledger_units) is not None
            )
            if not open_contract_valid:
                details.append(
                    _detail(
                        jump.name,
                        "open jump ledger route, C_h, numerical ledger change, and units",
                        "missing or invalid",
                    )
                )
            elif (
                tolerance_valid
                and ledger_projection is not None
                and ledger_change is not None
            ):
                open_residual = _max_abs(ledger_projection - ledger_change)
                if open_residual > float(jump.tolerance):
                    details.append(
                        ValidationDetail(
                            variable_name=jump.name,
                            expected="C_h R_h equals declared open jump ledger change",
                            value=open_residual,
                            threshold=jump.tolerance,
                        )
                    )
    return len(jumps)


def _validate_advection(payload: Mapping[str, Any], details: list[ValidationDetail]) -> int:
    raw_advections = payload.get(
        "stabilized_advections",
        payload.get("stabilized_advection", payload.get("stabilized_advection_map", ())),
    )
    advections = tuple(_advection_from_payload(item) for item in _iter_payloads(raw_advections))
    for index, advection in enumerate(advections):
        if advection is None:
            details.append(
                _detail(f"stabilized_advections[{index}]", "valid StabilizedAdvectionSpec", "invalid")
            )
            continue
        if not advection.operator_name:
            details.append(_detail(advection.name, "declared A_h(v) operator", "missing"))
        if not advection.scheme:
            details.append(_detail(advection.name, "upwind/TVD/limiter scheme metadata", "missing"))
        if not advection.conservative:
            details.append(_detail(advection.name, "conservative=true", advection.conservative))
        if not advection.cfl_metadata:
            details.append(_detail(advection.name, "CFL metadata", "missing"))
        if not advection.boundary_flux_route:
            details.append(_detail(advection.name, "boundary flux route", "missing"))
    return len(advections)


def _has_passivity_evidence(value: str) -> bool:
    token = _lower_token(value).replace("_", "-")
    return token in _ALLOWED_PASSIVITY_DECLARATIONS


def _validate_memory_kernels(payload: Mapping[str, Any], details: list[ValidationDetail]) -> int:
    raw_kernels = payload.get(
        "memory_kernels",
        payload.get("memory_kernel", payload.get("memory_kernel_map", ())),
    )
    kernels = tuple(_memory_from_payload(item) for item in _iter_payloads(raw_kernels))
    for index, kernel in enumerate(kernels):
        if kernel is None:
            details.append(
                _detail(f"memory_kernels[{index}]", "valid MemoryKernelSpec", "invalid")
            )
            continue
        if kernel.kernel_operator is None:
            details.append(_detail(kernel.name, "declared K_h memory kernel operator", "missing"))
        elif (
            kernel.kernel_operator.ndim != 2
            or kernel.kernel_operator.numel() == 0
            or kernel.kernel_operator.shape[0] != kernel.kernel_operator.shape[1]
            or torch.is_complex(kernel.kernel_operator)
            or not bool(torch.isfinite(kernel.kernel_operator).all().item())
        ):
            details.append(
                _detail(
                    kernel.name,
                    "non-empty finite square K_h memory kernel",
                    tuple(kernel.kernel_operator.shape),
                )
            )
        if not _space_declared(kernel.hidden_storage_space):
            details.append(_detail(kernel.name, "declared hidden memory cochain space", "missing"))
        if not kernel.effort_variable:
            details.append(_detail(kernel.name, "declared effort variable e_h", "missing"))
        if not _has_passivity_evidence(kernel.passivity_evidence):
            details.append(
                _detail(
                    kernel.name,
                    "recognized structural passivity declaration",
                    kernel.passivity_evidence or "missing",
                )
            )
        if not kernel.parameter_keys:
            details.append(_detail(kernel.name, "ParameterManager-backed memory kernel keys", "missing"))
        if not kernel.residual_route:
            details.append(_detail(kernel.name, "declared memory residual route", "missing"))
    return len(kernels)


def _validate_ale_geometry(
    payload: Mapping[str, Any], details: list[ValidationDetail]
) -> int:
    raw_corrections = payload.get(
        "ale_geometry_corrections",
        payload.get("ale_geometry_correction", payload.get("ale_geometry_map", ())),
    )
    corrections = tuple(_ale_from_payload(item) for item in _iter_payloads(raw_corrections))
    for index, correction in enumerate(corrections):
        if correction is None:
            details.append(
                _detail(
                    f"ale_geometry_corrections[{index}]",
                    "valid ALEGeometryCorrectionSpec",
                    "invalid",
                )
            )
            continue
        dot_hodge = correction.hodge_time_derivative
        if dot_hodge is None:
            details.append(_detail(correction.name, "declared dot(star_h)", "missing"))
        elif (
            dot_hodge.ndim != 2
            or dot_hodge.numel() == 0
            or dot_hodge.shape[-1] != dot_hodge.shape[-2]
        ):
            details.append(
                _detail(
                    correction.name,
                    "non-empty square dot(star_h)",
                    tuple(dot_hodge.shape),
                )
            )
        elif torch.is_complex(dot_hodge) or not bool(torch.isfinite(dot_hodge).all().item()):
            details.append(_detail(correction.name, "finite dot(star_h)", "non-finite"))
        elif not _hodge_rate_is_symmetric(dot_hodge):
            details.append(_detail(correction.name, "symmetric dot(star_h)", "non-symmetric"))
        if not _space_declared(correction.state_space):
            details.append(_detail(correction.name, "declared moving-geometry state space", "missing"))
        if not correction.energy_effort:
            details.append(_detail(correction.name, "declared energy effort e_E", "missing"))
        if not correction.correction_route:
            details.append(
                _detail(correction.name, "Hodge-energy correction route", "missing")
            )
        if not correction.port_accounting:
            details.append(_detail(correction.name, "geometry port accounting", "missing"))
        if not correction.parameter_keys:
            details.append(_detail(correction.name, "ParameterManager-backed geometry keys", "missing"))
    return len(corrections)


def _validate_hybrid_topology(
    payload: Mapping[str, Any], details: list[ValidationDetail]
) -> int:
    raw_transitions = payload.get(
        "hybrid_topology_transitions",
        payload.get(
            "hybrid_topology_transition",
            payload.get("hybrid_topology_map", ()),
        ),
    )
    transitions = tuple(_hybrid_from_payload(item) for item in _iter_payloads(raw_transitions))
    for index, transition in enumerate(transitions):
        if transition is None:
            details.append(
                _detail(
                    f"hybrid_topology_transitions[{index}]",
                    "valid HybridTopologyTransitionSpec",
                    "invalid",
                )
            )
            continue
        if not transition.event_name:
            details.append(_detail(transition.name, "explicit hybrid event name", "missing"))
        if not _space_declared(transition.source_complex):
            details.append(_detail(transition.name, "declared source DEC complex", "missing"))
        if not _space_declared(transition.target_complex):
            details.append(_detail(transition.name, "declared target DEC complex", "missing"))
        if transition.state_remap is None:
            details.append(_detail(transition.name, "state remap P_h", "missing"))
        elif transition.state_remap.ndim != 2:
            details.append(
                _detail(transition.name, "matrix-valued state remap P_h", transition.state_remap.ndim)
            )
        if transition.ledger_map is None and transition.ledger_change is None:
            details.append(
                _detail(
                    transition.name,
                    "ledger-preserving remap or declared ledger change",
                    "missing",
                )
            )
        if transition.ledger_map is not None and transition.ledger_map.ndim != 2:
            details.append(
                _detail(transition.name, "matrix-valued ledger map", transition.ledger_map.ndim)
            )
        if transition.ledger_change is not None and transition.ledger_change.ndim not in {1, 2}:
            details.append(
                _detail(
                    transition.name,
                    "vector or matrix ledger change",
                    transition.ledger_change.ndim,
                )
            )
        if not transition.reset_route:
            details.append(_detail(transition.name, "declared hybrid reset route", "missing"))
        if not transition.audit_route:
            details.append(
                _detail(transition.name, "dimension-change audit route", "missing")
            )
        if not transition.parameter_keys:
            details.append(_detail(transition.name, "ParameterManager-backed event keys", "missing"))
    return len(transitions)


def _validate_reproducibility(
    payload: Mapping[str, Any],
    details: list[ValidationDetail],
    *,
    stochastic_count: int,
    jump_count: int,
    required_semantics: set[str],
) -> None:
    if (
        stochastic_count == 0
        and jump_count == 0
        and not required_semantics.intersection({"continuous_sde", "marked_jump"})
    ):
        return
    repro = _as_mapping(
        payload.get(
            "stochastic_reproducibility",
            payload.get("stochastic_reproducibility_map", payload.get("reproducibility", {})),
        )
    )
    seed = repro.get("random_seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or int(seed) < 0
        or int(seed) > 2**63 - 1
    ):
        details.append(
            _detail(
                "stochastic_reproducibility",
                "random_seed is a non-negative integer",
                seed if seed is not None else "missing",
            )
        )
    noise_type = _clean_text(repro.get("noise_type")).lower()
    if noise_type not in _ALLOWED_NOISE_TYPES:
        details.append(
            _detail(
                "stochastic_reproducibility",
                "recognized stochastic noise_type",
                noise_type or "missing",
            )
        )
    solver_method = _clean_text(repro.get("solver_method")).lower()
    if solver_method not in _ALLOWED_SOLVER_METHODS:
        details.append(
            _detail(
                "stochastic_reproducibility",
                "recognized stochastic solver_method",
                solver_method or "missing",
            )
        )
    uses_ensemble = repro.get("uses_ensemble", False)
    if not isinstance(uses_ensemble, bool):
        details.append(
            _detail("stochastic_reproducibility", "uses_ensemble is boolean", uses_ensemble)
        )
    if uses_ensemble:
        ensemble_size = repro.get("ensemble_size")
        if (
            isinstance(ensemble_size, bool)
            or not isinstance(ensemble_size, Integral)
            or int(ensemble_size) <= 0
        ):
            details.append(
                _detail(
                    "stochastic_reproducibility",
                    "ensemble_size is a positive integer",
                    ensemble_size if ensemble_size is not None else "missing",
                )
            )
    if jump_count > 0 and not (repro.get("jump_propensities") or repro.get("propensity_keys")):
        details.append(_detail("stochastic_reproducibility", "jump propensities recorded", "missing"))

    if required_semantics.intersection({"continuous_sde", "marked_jump"}):
        strict_required = (
            "checkpoint_owner",
            "rng_state_owner",
            "rng_substream",
            "restart_policy",
            "time_step_mode",
        )
        for key in strict_required:
            if key not in repro or _clean_text(repro.get(key)) == "":
                details.append(
                    _detail(
                        "stochastic_reproducibility",
                        f"{key} recorded for restartable RNG",
                        "missing",
                    )
                )
        for owner_key in ("checkpoint_owner", "rng_state_owner"):
            owner = _clean_text(repro.get(owner_key))
            if owner and owner != "relic.core.simulator.checkpoint":
                details.append(
                    _detail(
                        "stochastic_reproducibility",
                        f"{owner_key}=relic.core.simulator.checkpoint",
                        owner,
                    )
                )
            elif owner and not _owner_is_existing(owner):
                details.append(
                    _detail(
                        "stochastic_reproducibility",
                        f"existing {owner_key}",
                        owner,
                    )
                )
        if _clean_text(repro.get("restart_policy")) not in {"", "restore_and_replay"}:
            details.append(
                _detail(
                    "stochastic_reproducibility",
                    "restart_policy=restore_and_replay",
                    repro.get("restart_policy"),
                )
            )
        time_step_mode = _clean_text(repro.get("time_step_mode"))
        if time_step_mode not in {"", "fixed", "adaptive"}:
            details.append(
                _detail(
                    "stochastic_reproducibility",
                    "time_step_mode fixed or adaptive",
                    time_step_mode,
                )
            )
        if time_step_mode == "adaptive":
            bridge_owner = _clean_text(repro.get("brownian_bridge_owner"))
            details.append(
                _detail(
                    "stochastic_reproducibility",
                    "implemented Brownian bridge owner [N] for adaptive Wiener increments",
                    bridge_owner or "missing",
                )
            )


def _validate_semantic_category_pairings(
    payload: Mapping[str, Any],
    *,
    required_semantics: set[str],
    details: list[ValidationDetail],
) -> None:
    """Reject solver/noise labels that belong to a different primary semantic."""

    if "continuous_sde" in required_semantics:
        raw_terms = payload.get("stochastic_terms", payload.get("terms", ()))
        for index, raw_term in enumerate(_iter_payloads(raw_terms)):
            term = _term_from_payload(raw_term)
            if term is None:
                continue
            classification = _clean_text(term.classification).lower()
            noise_type = _clean_text(term.noise_type).lower()
            solver_name = _clean_text(term.solver_name).lower()
            if (
                classification not in _CONTINUOUS_CLASSIFICATIONS
                or noise_type not in _CONTINUOUS_NOISE_TYPES
                or solver_name not in _CONTINUOUS_SOLVERS
            ):
                details.append(
                    _detail(
                        f"stochastic_terms[{index}]",
                        (
                            "continuous SDE category uses non-jump classification, "
                            "white/wiener/gaussian noise, and euler_maruyama solver"
                        ),
                        f"{classification}/{noise_type}/{solver_name}",
                    )
                )

    if "marked_jump" in required_semantics:
        repro = _as_mapping(
            payload.get(
                "stochastic_reproducibility",
                payload.get(
                    "stochastic_reproducibility_map",
                    payload.get("reproducibility", {}),
                ),
            )
        )
        noise_type = _clean_text(repro.get("noise_type")).lower()
        solver_method = _clean_text(repro.get("solver_method")).lower()
        if noise_type not in _MARKED_JUMP_NOISE_TYPES:
            details.append(
                _detail(
                    "stochastic_reproducibility",
                    "marked jump noise_type is poisson or marked_poisson",
                    noise_type or "missing",
                )
            )
        if solver_method not in _MARKED_JUMP_SOLVERS:
            details.append(
                _detail(
                    "stochastic_reproducibility",
                    (
                        "marked jump solver_method is gillespie, direct_ssa, "
                        "or next_reaction"
                    ),
                    solver_method or "missing",
                )
            )


def _validate_semantic_contracts(
    payload: Mapping[str, Any],
    *,
    required_semantics: set[str],
    counts: Mapping[str, int],
    details: list[ValidationDetail],
) -> None:
    invalid_semantics = sorted(required_semantics.difference(_SEMANTIC_COUNT_KEYS))
    for semantic in invalid_semantics:
        details.append(
            _detail(
                "required_semantics",
                "recognized independent stochastic/hybrid semantic",
                semantic,
            )
        )
    for semantic in sorted(required_semantics.intersection(_SEMANTIC_COUNT_KEYS)):
        count_key = _SEMANTIC_COUNT_KEYS[semantic]
        if int(counts.get(count_key, 0)) < 1:
            details.append(
                _detail(semantic, f"{semantic} declaration count > 0", 0)
            )

    if int(counts.get("stochastic_term_count", 0)) > 0:
        state_units = _as_mapping(payload.get("state_units"))
        for raw_term in _iter_payloads(
            payload.get("stochastic_terms", payload.get("terms", ()))
        ):
            term = _as_mapping(raw_term)
            for state_var in _iter_payloads(term.get("state_vars")):
                state_name = _clean_text(state_var)
                units = _clean_text(state_units.get(state_name))
                if dimensionality_from_unit(units) is None:
                    details.append(
                        _detail(
                            f"state_units.{state_name or 'missing'}",
                            "unit label recognized by canonical dimensionality owner",
                            units or "missing",
                        )
                    )

    if "continuous_sde" in required_semantics:
        for index, raw_term in enumerate(
            _iter_payloads(payload.get("stochastic_terms", payload.get("terms", ())))
        ):
            term = _as_mapping(raw_term)
            for field_name in ("state_vars", "parameter_keys", "noise_type", "solver_name"):
                value = term.get(field_name)
                if value is None or value == "" or value == () or value == []:
                    details.append(
                        _detail(
                            f"stochastic_terms[{index}]",
                            f"non-empty {field_name} for continuous SDE",
                            "missing",
                        )
                    )
    if "marked_jump" in required_semantics:
        raw_jumps = payload.get(
            "jump_resets",
            payload.get("jump_reset", payload.get("jump_reset_map", ())),
        )
        for index, raw_jump in enumerate(_iter_payloads(raw_jumps)):
            jump = _as_mapping(raw_jump)
            for field_name in ("mark_space", "event_clock"):
                if not _clean_text(jump.get(field_name)):
                    details.append(
                        _detail(
                            f"jump_resets[{index}]",
                            f"non-empty {field_name} for marked jump",
                            "missing",
                        )
                    )

    if "memory" in required_semantics:
        _validate_restart_map(
            payload,
            map_name="memory_restart",
            required_fields=(
                "checkpoint_owner",
                "history_state_keys",
                "history_cursor",
            ),
            details=details,
        )
    if "ale" in required_semantics:
        _validate_restart_map(
            payload,
            map_name="ale_restart",
            required_fields=(
                "checkpoint_owner",
                "geometry_state_keys",
                "geometry_cursor",
            ),
            details=details,
        )
    if "descriptor_dae" in required_semantics:
        descriptor_restart = _validate_restart_map(
            payload,
            map_name="descriptor_restart",
            required_fields=(
                "checkpoint_owner",
                "algebraic_state_keys",
                "projection_on_restore",
            ),
            details=details,
        )
        if descriptor_restart and descriptor_restart.get("projection_on_restore") is not True:
            details.append(
                _detail(
                    "descriptor_restart",
                    "projection_on_restore=true",
                    descriptor_restart.get("projection_on_restore"),
                )
            )
    if "topology_change" in required_semantics:
        topology_restart = _validate_restart_map(
            payload,
            map_name="topology_restart",
            required_fields=(
                "checkpoint_owner",
                "graph_epoch_owner",
                "event_queue_state_key",
            ),
            details=details,
        )
        graph_epoch_owner = _clean_text(topology_restart.get("graph_epoch_owner"))
        epoch_owner_ok = bool(
            graph_epoch_owner
            and graph_epoch_owner.replace("\\", ".") in _GRAPH_EPOCH_OWNERS
            and _owner_is_existing(graph_epoch_owner)
        )
        if graph_epoch_owner and not epoch_owner_ok:
            details.append(
                _detail(
                    "topology_restart",
                    "graph_epoch_owner is the canonical provider epoch owner "
                    "(relic.core.simulator.registration)",
                    graph_epoch_owner,
                )
            )
        event_queue = _clean_text(topology_restart.get("event_queue_state_key"))
        queue_ok = event_queue.replace("\\", ".") in _TOPOLOGY_EVENT_QUEUE_OWNERS
        if event_queue and not queue_ok:
            details.append(
                _detail(
                    "topology_restart",
                    "event_queue_state_key names the simulator delivery queue owner "
                    "(relic.core.simulator.execution_loop.retry_external_deliveries)",
                    event_queue,
                )
            )
        binding_ok = (
            epoch_owner_ok
            and queue_ok
            and _validate_execution_binding("topology_change", payload, details=details)
        )
        if not binding_ok:
            details.append(
                _detail(
                    "topology_restart",
                    _UNBOUND_SEMANTIC_EXPECTATIONS["topology_change"],
                    graph_epoch_owner or "not implemented",
                )
            )

    for semantic in sorted(
        required_semantics.intersection(_UNBOUND_SEMANTIC_EXPECTATIONS)
    ):
        if semantic == "topology_change":
            continue
        expected = _UNBOUND_SEMANTIC_EXPECTATIONS[semantic]
        if semantic in _EXECUTION_BINDING_CONTRACTS and _validate_execution_binding(
            semantic, payload, details=details,
        ):
            continue
        details.append(_detail(semantic, expected, "not implemented"))


def validate_stochastic_sdae_payload(
    payload: Mapping[str, Any],
    *,
    required_semantics: Iterable[str] = (),
) -> ValidationResultModel:
    """Validate stochastic IPHS/SDAE structural declarations.

    References:
      - Grmela and Ottinger (1997), GENERIC formalism.
      - Gillespie (1976), stochastic simulation algorithm.
      - Gneiting and Raftery (2007), probabilistic validation.
    Mathematical expression:
      Continuous SDE: dx = a(x,t) dt + B(x,t) dW_t.
      Marked jump: x(tau+) = x(tau-) + K(x(tau-), mark).
      Descriptor DAE: E(x,t) dx = F(x,t) dt with g(x,t) = 0.
      Memory: dz/dt = A_m z + B_m e and f_m = C_m z + D_m e.
      ALE: d(H_h(t)x_h)/dt includes the declared geometry-power route.
      Topology change: x_new = T_event x_old with an audited ledger remap.
      FDT-bound noise satisfies Sigma_h Sigma_h^T = scale * M_noise.
    Parameter meanings:
      stochastic_terms: classified stochastic term declarations.
      noise_covariances: FDT or coarse-grained covariance evidence.
      descriptor_constraints: DAE constraints A_h x = 0.
      jump_resets: stochastic jump or reset maps.
      memory_kernels: passive or positive-real memory-kernel declarations.
      ale_geometry_corrections: moving-geometry Hodge-energy correction audits.
      hybrid_topology_transitions: event-triggered dimension-change remaps.
    """

    details: list[ValidationDetail] = []
    require_declared_value = payload.get("require_declared_structure", False)
    if "require_declared_structure" in payload and not isinstance(
        require_declared_value,
        bool,
    ):
        details.append(
            _detail(
                "require_declared_structure",
                "require_declared_structure is bool",
                type(require_declared_value).__name__,
            )
        )
    strict_structure_required = require_declared_value is True
    runtime_owner = _clean_text(payload.get("runtime_owner", payload.get("owner", "relic.stochastic")))
    # A payload that declares random terms or jump resets is executed by a
    # relic.stochastic owner; a purely deterministic hybrid semantic (memory,
    # ALE, descriptor DAE, topology change) names its executing RELIC owner.
    declares_random_structure = bool(
        _iter_payloads(payload.get("stochastic_terms", payload.get("terms", ())))
        or _iter_payloads(
            payload.get("jump_resets", payload.get("jump_reset", payload.get("jump_reset_map", ())))
        )
    )
    if runtime_owner and declares_random_structure and not _stochastic_owner_is_existing(runtime_owner):
        details.append(
            _detail("runtime_owner", "callable or module stochastic owner", runtime_owner)
        )
    elif runtime_owner and not declares_random_structure and not (
        runtime_owner.replace("\\", ".").startswith("relic.") and _owner_is_existing(runtime_owner)
    ):
        details.append(
            _detail("runtime_owner", "callable or module RELIC owner", runtime_owner)
        )

    stochastic_count = _validate_stochastic_terms(payload, details)
    covariance_count = _validate_covariances(payload, details)
    flux_count = _validate_fluxes(payload, details)
    constraint_count = _validate_constraints(payload, details)
    jump_count = _validate_jumps(payload, details)
    advection_count = _validate_advection(payload, details)
    memory_count = _validate_memory_kernels(payload, details)
    ale_count = _validate_ale_geometry(payload, details)
    hybrid_count = _validate_hybrid_topology(payload, details)
    requested_semantics = _normalize_required_semantics(required_semantics).union(
        _normalize_required_semantics(payload.get("required_semantics"))
    )
    counts = {
        "stochastic_term_count": stochastic_count,
        "noise_covariance_count": covariance_count,
        "conservative_noise_flux_count": flux_count,
        "descriptor_constraint_count": constraint_count,
        "jump_reset_count": jump_count,
        "stabilized_advection_count": advection_count,
        "memory_kernel_count": memory_count,
        "ale_geometry_correction_count": ale_count,
        "hybrid_topology_transition_count": hybrid_count,
    }
    inferred_semantics: set[str] = set()
    if hybrid_count > 0:
        inferred_semantics.add("topology_change")
    if strict_structure_required:
        if jump_count > 0:
            inferred_semantics.add("marked_jump")
        if memory_count > 0:
            inferred_semantics.add("memory")
        if ale_count > 0:
            inferred_semantics.add("ale")
        if constraint_count > 0:
            inferred_semantics.add("descriptor_dae")
        if stochastic_count > 0:
            raw_terms = payload.get("stochastic_terms", payload.get("terms", ()))
            for index, raw_term in enumerate(_iter_payloads(raw_terms)):
                if isinstance(raw_term, Mapping) and "semantic" in raw_term:
                    details.append(
                        _detail(
                            f"stochastic_terms[{index}]",
                            (
                                "canonical StochasticTermSpec fields without "
                                "validator-only semantic"
                            ),
                            raw_term.get("semantic"),
                        )
                    )
            details.append(
                _detail(
                    "continuous_sde",
                    (
                        "continuous SDE semantic is not canonically representable "
                        "by StochasticTermSpec [N]"
                    ),
                    "stochastic_terms declared without canonical semantic field",
                )
            )
    requested_semantics.update(inferred_semantics)
    primary_structure_count = sum(
        (
            constraint_count,
            jump_count,
            memory_count,
            ale_count,
            hybrid_count,
        )
    )
    auxiliary_structure_count = sum(
        (
            covariance_count,
            flux_count,
            advection_count,
        )
    )
    declared_structure_count = sum(
        (
            stochastic_count,
            covariance_count,
            flux_count,
            constraint_count,
            jump_count,
            advection_count,
            memory_count,
            ale_count,
            hybrid_count,
        )
    )
    if strict_structure_required and primary_structure_count == 0:
        details.append(
            _detail(
                "stochastic_sdae",
                "at least one canonically represented primary stochastic/hybrid semantic",
                "only auxiliary, unresolved continuous, or empty declarations",
            )
        )
    category_semantics = set(requested_semantics)
    if strict_structure_required and stochastic_count > 0:
        category_semantics.add("continuous_sde")
    _validate_semantic_category_pairings(
        payload,
        required_semantics=category_semantics,
        details=details,
    )
    _validate_reproducibility(
        payload,
        details,
        stochastic_count=stochastic_count,
        jump_count=jump_count,
        required_semantics=requested_semantics,
    )
    _validate_semantic_contracts(
        payload,
        required_semantics=requested_semantics,
        counts=counts,
        details=details,
    )

    passed = not details
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            "Stochastic IPHS/SDAE structure is valid."
            if passed
            else "Stochastic IPHS/SDAE structure is incomplete."
        ),
        passed=passed,
        layer="L1",
        scope="stochastic_sdae",
        tags=["iphs", "stochastic", "sdae", "structural"],
        details=details,
        metrics={
            **counts,
            "declared_structure_count": declared_structure_count,
            "primary_structure_count": primary_structure_count,
            "auxiliary_structure_count": auxiliary_structure_count,
            "inferred_semantic_count": len(inferred_semantics),
            "required_semantic_count": len(requested_semantics),
            "issue_count": len(details),
        },
    )


@validated_rule(
    "stochastic_sdae_structure",
    "L1",
    name="Stochastic IPHS/SDAE structure",
    description="Validate stochastic, descriptor, jump, stabilized-advection, memory-kernel, ALE, and hybrid-topology IPHS declarations.",
    domains=["computation", "stochastic", "orchestration"],
    tags=["iphs", "stochastic", "sdae", "structural"],
    references=[
        "Grmela and Ottinger (1997), GENERIC formalism",
        "Gillespie (1976), stochastic simulation algorithm",
        "Gneiting and Raftery (2007), strictly proper scoring rules",
    ],
)
def validate_stochastic_sdae_structure(context: ValidationContext) -> ValidationResults:
    """Validate stochastic IPHS/SDAE payloads passed through context metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    if "stochastic_sdae" in metadata:
        raw_payload = metadata.get("stochastic_sdae")
    elif "stochastic_sdae_structure" in metadata:
        raw_payload = metadata.get("stochastic_sdae_structure")
    else:
        raw_payload = metadata
    payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
    if metadata.get("strict") is True:
        payload["require_declared_structure"] = True
    required_semantics = _normalize_required_semantics(
        metadata.get("required_semantics")
    )
    return [
        validate_stochastic_sdae_payload(
            payload,
            required_semantics=required_semantics,
        )
    ]


__all__ = ["validate_stochastic_sdae_payload", "validate_stochastic_sdae_structure"]
