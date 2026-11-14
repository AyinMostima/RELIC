"""
Graph executor bridging orchestration execution plans to core simulator.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from torch import Tensor
from torch.nn.modules import module as torch_module

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.functor import FunctorBase
from relic.core.provenance import ProvenanceRecorder
from relic.core.simulator import RELICSimulator
from relic.core.simulator.checkpoint import _capture_rng_state
from relic.core.state import StateManager, UpdateItem, partition_outputs
from relic.orchestration.cache import NodeCache
from relic.orchestration.node_registry import NodeRegistry
from relic.scientific import get_functor_path

logger = logging.getLogger(__name__)

_EXECUTION_CONFIG_FIELDS = frozenset(
    {"cache_policy", "execution_group", "iphs_contribution"}
)
_DETERMINISTIC_PURE = "deterministic_pure"
_NODE_CACHE_CONTRACT_VERSION = "2.0.0"
_CACHE_RUNTIME_SHELL_FIELDS = frozenset(
    {
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_call_governor",
        "_conservation_targets",
        "_dependencies",
        "_dependents",
        "_error_count",
        "_execution_count",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_inside_canonical_call",
        "_is_full_backward_hook",
        "_last_error",
        "_last_execution_time",
        "_last_pipeline_timings_s",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_original_config",
        "_parameter_bounds",
        "_parameter_manager",
        "_parameter_requires_grad",
        "_parameter_units",
        "_parameters",
        "_performance_profiling_enabled",
        "_performance_stats",
        "_pipeline_timing_count",
        "_pipeline_timing_totals_s",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_total_execution_time",
        "_validation_engine",
        "boundary_conditions",
        "config",
        "default_value",
        "device",
        "field_name",
        "grid_resolution",
        "interpolation_method",
        "metadata",
        "name",
        "parameter_values",
        "spatial_dims",
        "state",
        "temporal_dependence",
        "training",
        "units",
    }
)
_PROTECTED_CONSTRUCTOR_CONFIG_FIELDS = frozenset(
    {
        "config",
        "dependencies",
        "device",
        "enable_gpu",
        "metadata",
        "name",
        "parameter_manager",
        "parameters",
        "registry",
        "require_gpu",
        "simulator",
        "state_manager",
    }
)


def _is_selection_annotation(annotation: Any) -> bool:
    if annotation in (str, bool):
        return True
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        values = [value for value in get_args(annotation) if value is not None]
        return bool(values) and all(type(value) in (str, bool) for value in values)
    if origin in (Union, UnionType):
        options = [
            option for option in get_args(annotation) if option is not type(None)
        ]
        return bool(options) and all(
            _is_selection_annotation(option) for option in options
        )
    return False


def _coerce_selection_value(
    *,
    field_name: str,
    annotation: Any,
    value: Any,
) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        options = list(get_args(annotation))
        if value is None and type(None) in options:
            return None
        matches = []
        for option in options:
            if option is type(None):
                continue
            try:
                matches.append(
                    _coerce_selection_value(
                        field_name=field_name,
                        annotation=option,
                        value=value,
                    )
                )
            except (TypeError, ValueError):
                continue
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"Constructor control '{field_name}' does not match one "
            "unambiguous allowed selection type."
        )
    if origin is Literal:
        allowed = get_args(annotation)
        for candidate in allowed:
            if type(value) is type(candidate) and value == candidate:
                return value
        raise ValueError(
            f"Constructor control '{field_name}' must be one of "
            f"{list(allowed)}."
        )
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            valid_values = [member.value for member in annotation]
            raise ValueError(
                f"Constructor control '{field_name}' must be one of "
                f"{valid_values}; got {value!r}."
            ) from exc
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(
                f"Constructor control '{field_name}' must be a string."
            )
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise TypeError(
                f"Constructor control '{field_name}' must be a bool."
            )
        return value
    raise ValueError(
        f"Constructor field '{field_name}' is not a safe selection control."
    )


# Runtime witness of a declared IntegrationPlan (CR-A-04 step 8; design
# 20261019_cr_a04_design_01, section 3, "Executor"). A graph.metadata.integration_plan
# whose stages declare runtime_owner simulator states the step composition the
# simulator executes; GraphExecutor compares its schedule with
# RELICSimulator.get_composition_plan() of the prepared (and of a restored) run,
# refuses a mismatch, records a match, and checks the commit contexts of the
# executed calls against the declared stages. Every executable plan is a strict
# compile (compile-run witness), so a mismatch is always refused.
_INTEGRATION_PLAN_WITNESS_SCHEMA_VERSION = "1.0.0"
_SIMULATOR_RUNTIME_OWNER = "simulator"
# Stage fields of the compared schedule; stage_id, solver_name, structure_kind,
# reads, writes, depends_on, validation_hooks and references are descriptive.
_SCHEDULE_STAGE_FIELDS = (
    "operator_kind",
    "dt_fraction",
    "advances_time",
    "coupling_mode",
    "runtime_owner",
    "stage_metadata",
)
_REPORTED_DIFFERENCES = 5
_REPORTED_VIOLATIONS = 20


def _canonical_schedule_value(value: Any) -> Any:
    """JSON value of one schedule field with every number as a float (pure).

    A declared plan may spell 1 for 1.0; booleans, strings and None are kept.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("integration_plan schedule numbers must be finite.")
        return number
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_schedule_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_schedule_value(item) for item in value]
    raise ValueError(
        "integration_plan schedule values must be JSON-safe, got "
        f"{type(value).__name__}."
    )


def composition_schedule(plan_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compared schedule of an IntegrationPlan payload (pure).

    The scheme and, per stage in execution order, the fields of
    _SCHEDULE_STAGE_FIELDS with canonical numbers: the identity GraphExecutor
    compares and digests (composition_schedule_sha256).
    """
    return {
        "scheme": str(plan_payload["scheme"]),
        "stages": [
            {
                name: _canonical_schedule_value(stage[name])
                for name in _SCHEDULE_STAGE_FIELDS
            }
            for stage in plan_payload["stages"]
        ],
    }


def composition_schedule_sha256(schedule: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON (sorted keys, no whitespace) of a schedule (pure)."""
    encoded = json.dumps(
        schedule,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _schedule_differences(
    declared: Mapping[str, Any],
    executed: Mapping[str, Any],
) -> list[str]:
    """Readable differences of two composition schedules (pure)."""
    differences: list[str] = []
    if declared["scheme"] != executed["scheme"]:
        differences.append(
            f"scheme declared {declared['scheme']!r}, executed {executed['scheme']!r}"
        )
    declared_stages = declared["stages"]
    executed_stages = executed["stages"]
    if len(declared_stages) != len(executed_stages):
        differences.append(
            f"{len(declared_stages)} declared stages, "
            f"{len(executed_stages)} executed stages"
        )
    for index, (left, right) in enumerate(zip(declared_stages, executed_stages)):
        for name in _SCHEDULE_STAGE_FIELDS:
            if left[name] == right[name]:
                continue
            if isinstance(left[name], dict) and isinstance(right[name], dict):
                for key in sorted(set(left[name]) | set(right[name])):
                    if left[name].get(key) != right[name].get(key):
                        differences.append(
                            f"stage {index} {name}.{key}: declared "
                            f"{left[name].get(key)!r}, executed "
                            f"{right[name].get(key)!r}"
                        )
                continue
            differences.append(
                f"stage {index} {name}: declared {left[name]!r}, "
                f"executed {right[name]!r}"
            )
    return differences


def _stage_matches(
    stage: Mapping[str, Any],
    functor: str,
    context: Mapping[str, Any],
) -> bool:
    """Whether one executed call (functor, commit context) is the given stage."""
    metadata = stage.get("stage_metadata")
    if not isinstance(metadata, Mapping) or stage.get("operator_kind") != functor:
        return False
    if (
        metadata.get("role") != context.get("role")
        or metadata.get("commit_stage") != context.get("stage")
    ):
        return False
    try:
        interval = float(context["composition_interval"])
        return (
            float(context["dt"]) == interval * float(stage["dt_fraction"])
            and float(context["interval_start"])
            == float(context["composition_start"])
            + interval * float(metadata["start_fraction"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def executed_stage_report(
    plan_payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    first_step: int,
) -> dict[str, Any]:
    """Check the commit context of every executed call against the declared stages (pure).

    References: W3C PROV-DM (2013) (generation records); design
      20261019_cr_a04_design_01, section 3 ("Executor") and T10.
    Mathematical expression: a call of Functor f with commit context (step s,
      stage k, role r, interval start a, interval h, composition start t0,
      composition interval H) matches declared stage j iff f = operator_kind_j,
      r = role_j, k = commit_stage_j, h = H alpha_j and a = t0 + H beta_j,
      evaluated in binary64 as the group owner computes them; every call with
      s >= first_step must match exactly one stage.
    Parameter meanings: alpha_j, beta_j the dt and start fractions of stage j;
      plan_payload the declared IntegrationPlan payload; provenance the
      mapping variable -> records of
      RELICSimulator.get_variable_provenance(include_commit_context=True);
      first_step the step count when the plan was last verified.
    Units: a, h, t0, H [s]; alpha_j, beta_j dimensionless; s, k counts.
    Domain/constraints: only writer records with a commit context are calls;
      a call without a numeric update leaves no record and the retained record
      window is bounded (max_provenance_records_per_variable), so stages
      without an observed call are reported, not refused; records of steps
      before first_step (a restored history) are not checked.
    Sign convention: not applicable.
    """
    stages = list(plan_payload["stages"])
    calls: dict[tuple[Any, ...], tuple[str, Mapping[str, Any]]] = {}
    for records in provenance.values():
        for record in records:
            if not isinstance(record, Mapping) or record.get("role") != "writer":
                continue
            context = record.get("commit_context")
            if not isinstance(context, Mapping):
                continue
            step = context.get("step")
            if type(step) is not int or step < first_step:
                continue
            functor = str(record.get("functor"))
            calls[(functor, tuple(sorted(context.items())))] = (functor, context)
    observed: set[str] = set()
    steps: set[int] = set()
    violations: list[dict[str, Any]] = []

    def _order(item: tuple[str, Mapping[str, Any]]) -> tuple[int, int, str]:
        stage_index = item[1].get("stage")
        return (
            int(item[1]["step"]),
            stage_index if type(stage_index) is int else -1,
            item[0],
        )

    for functor, context in sorted(calls.values(), key=_order):
        steps.add(int(context["step"]))
        matches = [
            str(stage["stage_id"])
            for stage in stages
            if _stage_matches(stage, functor, context)
        ]
        if len(matches) == 1:
            observed.add(matches[0])
            continue
        violations.append(
            {
                "functor": functor,
                "commit_context": dict(context),
                "matching_stages": matches,
            }
        )
    return {
        "passed": not violations,
        "source": (
            "RELICSimulator.get_variable_provenance(include_commit_context=True)"
        ),
        "first_step": int(first_step),
        "checked_calls": len(calls),
        "checked_steps": sorted(steps),
        "stages_observed": [
            str(stage["stage_id"]) for stage in stages if stage["stage_id"] in observed
        ],
        "stages_not_observed": [
            str(stage["stage_id"])
            for stage in stages
            if stage["stage_id"] not in observed
        ],
        "violation_count": len(violations),
        "violations": violations[:_REPORTED_VIOLATIONS],
    }


class ExecutionStrategy(Enum):
    """
    Graph execution strategy controlling node evaluation behavior.

    References
    ----------
    NVIDIA OmniGraph core concepts for push vs. lazy evaluation.

    Mathematical Expression
    -----------------------
    PUSH: execute all nodes on every tick.
    DIRTY: execute node only if inputs changed since last evaluation.
    """

    PUSH = auto()
    DIRTY = auto()


class GraphExecutor:
    """Execute compiled orchestration graphs with the core simulator."""

    def __init__(
        self,
        plan: Any,
        state_manager: StateManager | None = None,
        cache: NodeCache | None = None,
        output_dir: Path | None = None,
        registry: NodeRegistry | None = None,
        strategy: ExecutionStrategy = ExecutionStrategy.PUSH,
        dirty_include_time: bool = True,
        simulator_config: Mapping[str, Any] | None = None,
    ) -> None:
        if simulator_config is not None and not isinstance(
            simulator_config,
            Mapping,
        ):
            raise TypeError("simulator_config must be a mapping or None.")
        self.plan = plan
        self.state_manager = state_manager or StateManager()
        self.cache = cache
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.registry = registry
        self._simulator_config = (
            deepcopy(dict(simulator_config))
            if simulator_config is not None
            else None
        )

        self._simulator = RELICSimulator(
            deepcopy(self._simulator_config)
        )
        self._functor_instances: dict[str, Any] = {}
        self._steps_by_id: dict[str, Any] = {
            step.node_id: step for step in self.plan.steps
        }

        self._strategy = strategy
        self._dirty_include_time = bool(dirty_include_time)
        self._dirty_nodes: set[str] = set()
        self._last_input_hashes: dict[str, str] = {}
        self._last_outputs: dict[str, Any] = {}
        self._execution_count = 0
        self._execution_prepared = False
        # IntegrationPlan runtime witness of the prepared run and the
        # declared plan payload it verified (_verify_integration_plan).
        self._integration_plan_witness: dict[str, Any] | None = None
        self._witnessed_plan_payload: dict[str, Any] | None = None
        # Provider identity consumed from the simulator's committed view: the
        # last reconciled (simulator, epoch) and the per-node digest triple
        # (implementation, contract, binding) bound into cache keys.
        self._provider_view_key: tuple[int, int] | None = None
        self._provider_digests: dict[str, tuple[Any, Any, Any]] = {}

        self._recorder: ProvenanceRecorder | None = None
        if self.output_dir is not None:
            self._recorder = ProvenanceRecorder(
                output_dir=self.output_dir, run_id=self._simulator.run_id,
            )
            self.state_manager.set_provenance_recorder(
                self._recorder, run_id=self._simulator.run_id,
            )

    @property
    def simulator_config(self) -> dict[str, Any]:
        """Return a detached snapshot of the active simulator config."""

        return deepcopy(self._simulator.config)

    @staticmethod
    def _normalize_contract_names(
        value: Any,
        *,
        node_id: str,
        contract_name: str,
    ) -> list[str]:
        """Normalize a runtime metadata port declaration."""

        if value is None:
            return []
        if isinstance(value, Mapping):
            raw_names = list(value)
        elif isinstance(value, str):
            raw_names = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_names = list(value)
        else:
            raise TypeError(
                f"Graph node '{node_id}' runtime {contract_name} contract "
                "must be a sequence or mapping."
            )
        names = [str(name).strip() for name in raw_names]
        if any(not name for name in names):
            raise ValueError(
                f"Graph node '{node_id}' runtime {contract_name} contract "
                "contains an empty port name."
            )
        if len(names) != len(set(names)):
            raise ValueError(
                f"Graph node '{node_id}' runtime {contract_name} contract "
                "contains duplicate port names."
            )
        return sorted(names)

    @staticmethod
    def _compile_run_witness_payload(
        node_witnesses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the one canonical strict compile-run witness envelope."""

        return {
            "schema_version": "1.0.0",
            "document_kind": "compile_run_witness",
            "passed": True,
            "execution_performed": False,
            "node_count": len(node_witnesses),
            "checks": {
                "class_resolution": True,
                "constructor_validation": True,
                "functor_base": True,
                "port_validation": True,
                "writer_validation": True,
            },
            "nodes": node_witnesses,
        }

    def _runtime_node_witness(
        self,
        step: Any,
        functor: Any,
    ) -> dict[str, Any]:
        """Validate and describe one live Functor against its plan ports."""

        if not isinstance(functor, FunctorBase):
            raise TypeError(
                f"Graph node '{step.node_id}' runtime class must inherit "
                "from FunctorBase."
            )

        metadata = getattr(functor, "metadata", None)
        if metadata is None:
            raise ValueError(
                f"Graph node '{step.node_id}' runtime class has no "
                "FunctorMetadata."
            )
        runtime_inputs = self._normalize_contract_names(
            getattr(metadata, "input_variables", None),
            node_id=step.node_id,
            contract_name="input",
        )
        runtime_outputs = self._normalize_contract_names(
            getattr(metadata, "output_variables", None),
            node_id=step.node_id,
            contract_name="output",
        )
        plan_inputs = sorted(str(name) for name in step.inputs)
        plan_outputs = sorted(str(name) for name in step.outputs)

        missing_runtime_inputs = sorted(set(plan_inputs) - set(runtime_inputs))
        if missing_runtime_inputs:
            raise ValueError(
                f"Graph node '{step.node_id}' runtime input contract does "
                "not declare plan inputs "
                f"{missing_runtime_inputs}."
            )
        missing_runtime_outputs = sorted(
            set(plan_outputs) - set(runtime_outputs)
        )
        if missing_runtime_outputs:
            raise ValueError(
                f"Graph node '{step.node_id}' runtime output contract "
                "does not declare plan outputs "
                f"{missing_runtime_outputs}."
            )

        definition = (
            self.registry.get_definition(step.class_type)
            if self.registry is not None
            else None
        )
        if definition is None:
            raise ValueError(
                f"Graph node '{step.node_id}' has no registered "
                "NodeDefinition for runtime preflight."
            )
        required_inputs = sorted(
            name
            for name, port in definition.inputs.items()
            if bool(port.required)
        )
        missing_required_inputs = sorted(set(required_inputs) - set(plan_inputs))
        if missing_required_inputs:
            raise ValueError(
                f"Graph node '{step.node_id}' runtime input contract is "
                "missing required plan inputs "
                f"{missing_required_inputs}."
            )

        return {
            "node_id": step.node_id,
            "class_type": step.class_type,
            "runtime_class": (
                f"{functor.__class__.__module__}."
                f"{functor.__class__.__qualname__}"
            ),
            "plan_inputs": plan_inputs,
            "required_inputs": required_inputs,
            "runtime_inputs": runtime_inputs,
            "plan_outputs": plan_outputs,
            "runtime_outputs": runtime_outputs,
        }

    def _build_compile_run_witness(self) -> dict[str, Any]:
        """Construct the compiler witness without requiring a final plan seal."""

        preflight_simulator = RELICSimulator(deepcopy(self._simulator_config))
        preflight_simulator.strict_dependencies = True
        node_witnesses: list[dict[str, Any]] = []

        for step in self.plan.steps:
            functor = self._build_functor(step)
            node_witness = self._runtime_node_witness(step, functor)
            self._attach_dependencies(functor, step)
            execution_group = (
                step.config.get("execution_group")
                if isinstance(step.config, Mapping)
                else None
            )
            preflight_simulator.register_functor(
                functor,
                execution_group=execution_group,
                _prepare_device=False,
            )
            node_witnesses.append(node_witness)

        return self._compile_run_witness_payload(node_witnesses)

    def _require_executable_plan(self) -> dict[str, Any]:
        """Validate the compiler seal and return its strict runtime witness."""

        from .compiler import verify_execution_plan_integrity

        verify_execution_plan_integrity(self.plan, self.registry)
        report = getattr(self.plan, "report", None)
        metadata = getattr(report, "metadata", None)
        witness = (
            metadata.get("compile_run_witness")
            if isinstance(metadata, Mapping)
            else None
        )
        expected_checks = {
            "class_resolution": True,
            "constructor_validation": True,
            "functor_base": True,
            "port_validation": True,
            "writer_validation": True,
        }
        if (
            not isinstance(witness, Mapping)
            or witness.get("schema_version") != "1.0.0"
            or witness.get("document_kind") != "compile_run_witness"
            or witness.get("passed") is not True
            or witness.get("execution_performed") is not False
            or "skipped" in witness
            or type(witness.get("node_count")) is not int
            or witness.get("node_count") != len(self.plan.steps)
            or witness.get("checks") != expected_checks
            or type(witness.get("nodes")) is not list
            or len(witness["nodes"]) != len(self.plan.steps)
        ):
            raise ValueError(
                "Graph execution requires a strict compile-run witness."
            )
        return deepcopy(dict(witness))

    @staticmethod
    def _assert_runtime_witness(
        expected: Mapping[str, Any],
        actual: Mapping[str, Any],
    ) -> None:
        if actual != expected:
            raise ValueError(
                "Execution plan runtime witness does not match the sealed "
                "compile-run witness."
            )

    def preflight(self) -> dict[str, Any]:
        """Revalidate a compiler-issued plan without executing a step."""

        expected = self._require_executable_plan()
        actual = self._build_compile_run_witness()
        self._assert_runtime_witness(expected, actual)
        self._require_executable_plan()
        return actual

    @property
    def time_series(self) -> dict[str, list[Any]]:
        """Return detached execution diagnostics without exposing the simulator."""

        snapshot: dict[str, list[Any]] = {}
        for name, values in self._simulator.time_series.items():
            snapshot[name] = [
                value.detach().clone()
                if isinstance(value, Tensor)
                else deepcopy(value)
                for value in values
            ]
        return snapshot

    @property
    def execution_summary(self) -> dict[str, Any]:
        """Return immutable scalar execution evidence for the active run."""

        return {
            "current_time": float(self._simulator.current_time),
            "step_count": int(self._simulator.step_count),
            "execution_count": int(self._execution_count),
            "device": str(self._simulator.device),
            "execution_order": list(self._simulator.execution_order),
        }


    @property
    def integration_plan_witness(self) -> dict[str, Any] | None:
        """Return a detached copy of the IntegrationPlan runtime witness of the prepared run.

        None before a run is prepared and after a refused verification;
        otherwise status not_declared (no graph.metadata.integration_plan),
        not_simulator_owned (a declared plan without simulator stages: metadata
        only, not compared), matched (the declared schedule equals the
        simulator composition plan; executed_stage_check holds the commit
        context check of the executed steps) or superseded (a later provider
        epoch published another plan and the run was stopped).
        """

        return deepcopy(self._integration_plan_witness)

    def _declared_integration_plan(self) -> Any:
        """Return graph.metadata.integration_plan of the sealed plan, or None."""

        report = getattr(self.plan, "report", None)
        metadata = getattr(report, "metadata", None)
        graph_metadata = (
            metadata.get("graph_metadata") if isinstance(metadata, Mapping) else None
        )
        if not isinstance(graph_metadata, Mapping):
            return None
        return graph_metadata.get("integration_plan")

    def _verify_integration_plan(self, phase: str) -> None:
        """Compare a declared simulator-owned IntegrationPlan with the live composition.

        Called after the registrations of a prepared run (phase prepared) and
        after a restore (restored, migrated), before any step. A declared plan
        whose stages all declare runtime_owner simulator must equal
        RELICSimulator.get_composition_plan() in composition_schedule (scheme
        and, per stage, operator_kind, dt_fraction, advances_time,
        coupling_mode, runtime_owner and stage_metadata); a mismatch, a
        malformed plan and a plan that mixes simulator stages with other owners
        raise ValueError and leave no witness. A plan without simulator stages
        is metadata only (not compared). The witness records the phases, the
        provider epoch, the schedule digests and the first step it covers.
        """

        simulator = self._simulator
        verification = {
            "phase": phase,
            "provider_epoch": int(simulator.provider_epoch),
            "step_count": int(simulator.step_count),
        }
        previous = self._integration_plan_witness
        verifications = (
            list(previous["verifications"])
            if previous is not None and phase != "prepared"
            else []
        )
        verifications.append(verification)
        self._integration_plan_witness = None
        self._witnessed_plan_payload = None
        witness: dict[str, Any] = {
            "schema_version": _INTEGRATION_PLAN_WITNESS_SCHEMA_VERSION,
            "document_kind": "integration_plan_runtime_witness",
            "source": "graph.metadata.integration_plan",
            "verifications": verifications,
            "executed_stage_check": None,
        }
        declared = self._declared_integration_plan()
        if declared is None:
            witness["status"] = "not_declared"
            self._integration_plan_witness = witness
            return
        from relic.computation.solvers.geometric import IntegrationPlan

        try:
            declared_payload = IntegrationPlan.from_dict(declared).to_dict()
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                "graph.metadata.integration_plan is not a valid IntegrationPlan: "
                f"{exc}"
            ) from exc
        owners = sorted(
            {str(stage["runtime_owner"]) for stage in declared_payload["stages"]}
        )
        witness["declared_plan_id"] = declared_payload["plan_id"]
        witness["runtime_owners"] = owners
        if _SIMULATOR_RUNTIME_OWNER not in owners:
            witness["status"] = "not_simulator_owned"
            self._integration_plan_witness = witness
            return
        if owners != [_SIMULATOR_RUNTIME_OWNER]:
            raise ValueError(
                "graph.metadata.integration_plan mixes runtime_owner simulator with "
                f"{[owner for owner in owners if owner != _SIMULATOR_RUNTIME_OWNER]}; "
                "a declared simulator composition must own every stage to be "
                "compared with the simulator composition plan."
            )
        declared_schedule = composition_schedule(declared_payload)
        executed_schedule = composition_schedule(
            simulator.get_composition_plan().to_dict()
        )
        if declared_schedule != executed_schedule:
            differences = _schedule_differences(declared_schedule, executed_schedule)
            hidden = len(differences) - _REPORTED_DIFFERENCES
            raise ValueError(
                "GraphExecutor refuses the run: graph.metadata.integration_plan "
                "(runtime_owner simulator) does not match the simulator "
                "composition plan of provider epoch "
                f"{verification['provider_epoch']} ({phase}): "
                + "; ".join(differences[:_REPORTED_DIFFERENCES])
                + (f"; and {hidden} more" if hidden > 0 else "")
            )
        witness.update(
            {
                "status": "matched",
                "strict": True,
                "scheme": declared_schedule["scheme"],
                "stage_ids": [
                    str(stage["stage_id"]) for stage in declared_payload["stages"]
                ],
                "compared_fields": [
                    "scheme",
                    *(f"stages[].{name}" for name in _SCHEDULE_STAGE_FIELDS),
                ],
                "declared_schedule_sha256": composition_schedule_sha256(
                    declared_schedule
                ),
                "executed_schedule_sha256": composition_schedule_sha256(
                    executed_schedule
                ),
                "provider_epoch": verification["provider_epoch"],
                "first_step": verification["step_count"],
            }
        )
        self._integration_plan_witness = witness
        self._witnessed_plan_payload = declared_payload

    def _step_under_witness(self, dt_value: float) -> None:
        """Advance one step; stop when a new provider epoch changes the witnessed plan.

        A provider epoch published during the step (a topology transition, a
        registration) that keeps the composition schedule is recorded; one that
        changes it marks the witness superseded and raises RuntimeError before
        the next step, so every executed step ran under the declared plan.
        """

        self._simulator.step(dt=dt_value)
        witness = self._integration_plan_witness
        if witness is None or witness.get("status") != "matched":
            return
        epoch = int(self._simulator.provider_epoch)
        if epoch == witness["provider_epoch"]:
            return
        digest = composition_schedule_sha256(
            composition_schedule(self._simulator.get_composition_plan().to_dict())
        )
        step_count = int(self._simulator.step_count)
        witness["verifications"].append(
            {
                "phase": "provider_epoch_change",
                "provider_epoch": epoch,
                "step_count": step_count,
            }
        )
        if digest != witness["declared_schedule_sha256"]:
            witness["status"] = "superseded"
            witness["executed_schedule_sha256"] = digest
            raise RuntimeError(
                f"GraphExecutor stops after step {step_count}: provider epoch "
                f"{epoch} publishes a composition plan that differs from the "
                "declared graph.metadata.integration_plan (runtime_owner "
                "simulator); the steps before ran under the declared plan."
            )
        witness["provider_epoch"] = epoch

    def _check_executed_stages(self) -> None:
        """Check the commit contexts of the executed calls against the declared stages."""

        witness = self._integration_plan_witness
        if (
            witness is None
            or witness.get("status") != "matched"
            or self._witnessed_plan_payload is None
        ):
            return
        report = executed_stage_report(
            self._witnessed_plan_payload,
            self._simulator.get_variable_provenance(include_commit_context=True),
            first_step=int(witness["first_step"]),
        )
        witness["executed_stage_check"] = report
        if not report["passed"]:
            first = report["violations"][0] if report["violations"] else None
            raise RuntimeError(
                f"GraphExecutor: {report['violation_count']} executed call(s) do "
                "not match a stage of the declared graph.metadata.integration_plan "
                f"(first: {first!r})."
            )

    def _resolve_functor_class(self, class_type: str) -> type:
        if self.registry is not None:
            resolved = self.registry.resolve_class_type(class_type)
            if resolved is not None:
                return resolved

        if class_type.startswith("Functor:"):
            dotted = class_type.split(":", 1)[-1].strip()
            return self._import_class(dotted)

        if ":" in class_type:
            module_path, cls_name = class_type.split(":", 1)
            return self._import_class(f"{module_path}.{cls_name}")

        if "." in class_type:
            return self._import_class(class_type)

        # The functor registry is complete at import; an unknown name raises KeyError.
        path = get_functor_path(class_type)
        module_path, cls_name = path.split(":", 1)
        return self._import_class(f"{module_path}.{cls_name}")

    @staticmethod
    def _import_class(dotted: str) -> type:
        if "." not in dotted:
            raise ValueError(f"Invalid functor class path: {dotted}")
        module_path, cls_name = dotted.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    def _build_functor(self, step: Any) -> Any:
        cls = self._resolve_functor_class(step.class_type)
        raw_config = step.config
        if raw_config is None:
            config: dict[str, Any] = {}
        elif isinstance(raw_config, Mapping):
            config = dict(raw_config)
        else:
            raise TypeError(
                f"Graph node '{step.node_id}' config must be a mapping."
            )
        for field_name in _EXECUTION_CONFIG_FIELDS:
            config.pop(field_name, None)

        signature = inspect.signature(cls)
        explicit_parameters = {
            name: parameter
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        kwargs: dict[str, Any] = {}
        if "name" in explicit_parameters:
            kwargs["name"] = step.node_id

        type_hints: dict[str, Any] | None = None
        for field_name in sorted(list(config)):
            if field_name in _PROTECTED_CONSTRUCTOR_CONFIG_FIELDS:
                raise ValueError(
                    f"Graph node '{step.node_id}' config contains protected "
                    f"constructor field '{field_name}'."
                )
            parameter = explicit_parameters.get(field_name)
            if parameter is None:
                continue
            if type_hints is None:
                try:
                    type_hints = get_type_hints(cls.__init__)
                except (NameError, TypeError, ValueError):
                    type_hints = {}
            annotation = type_hints.get(field_name, parameter.annotation)
            if not _is_selection_annotation(annotation):
                raise ValueError(
                    f"Graph node '{step.node_id}' cannot override "
                    f"non-selection constructor field '{field_name}'."
                )
            kwargs[field_name] = _coerce_selection_value(
                field_name=field_name,
                annotation=annotation,
                value=config.pop(field_name),
            )

        if "config" in explicit_parameters:
            kwargs["config"] = config
        elif config:
            raise TypeError(
                f"Functor class '{step.class_type}' does not declare an "
                f"explicit config parameter for fields {sorted(config)}."
            )

        try:
            signature.bind(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Cannot bind constructor for graph node '{step.node_id}' "
                f"using class '{step.class_type}': {exc}"
            ) from exc
        try:
            instance = cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"Failed to construct graph node '{step.node_id}' with class "
                f"'{step.class_type}': {exc}"
            ) from exc
        bind_definition = getattr(instance, "bind_node_definition", None)
        if callable(bind_definition):
            if self.registry is None:
                raise RuntimeError(
                    f"Graph node '{step.node_id}' requires NodeRegistry "
                    "definition binding."
                )
            definition = self.registry.get_definition(step.class_type)
            if definition is None:
                raise RuntimeError(
                    f"Graph node '{step.node_id}' has no registered "
                    "NodeDefinition."
                )
            bind_definition(definition)
        return instance

    @staticmethod
    def _extract_dependencies(step: Any) -> list[str]:
        dependencies = []
        for payload in step.inputs.values():
            if not isinstance(payload, dict):
                continue
            link_payload = payload.get("link")
            if isinstance(link_payload, (list, tuple)) and len(link_payload) == 2:
                source_node = str(link_payload[0])
                if source_node:
                    dependencies.append(source_node)
        return sorted(set(dependencies))

    def _attach_dependencies(self, functor: Any, step: Any) -> None:
        if not isinstance(functor, FunctorBase):
            return
        dependencies = self._extract_dependencies(step)
        if not dependencies:
            return
        functor.set_dependencies(dependencies)

    def _wrap_functor(self, functor: Any, step: Any) -> None:
        """Bind execution policy without replacing the scientific kernel."""
        node_id = step.node_id
        declared_keys = tuple(
            dict.fromkeys(self._declared_state_keys(step).values())
        )

        def governed_call(
            module_call: Callable[..., Any],
            state: dict[str, Tensor],
            *,
            time: float,
            dt: Tensor,
            **kwargs: Any,
        ) -> Any:
            cache_key = None
            self._reconcile_provider_view()
            cache_allowed = self._is_cache_eligible(
                functor,
                step,
                state,
                dt,
                kwargs,
            )
            if (
                self._strategy == ExecutionStrategy.DIRTY
                and cache_allowed
                and not self._is_node_dirty(node_id, step, state, time, dt)
            ):
                cached_output = self._last_outputs.get(node_id)
                if cached_output is not None and self._is_output_cache_eligible(cached_output):
                    self._update_dirty_tracking(
                        node_id,
                        step,
                        state,
                        time,
                        dt,
                    )
                    return cached_output
                self._last_outputs.pop(node_id, None)
                self._last_input_hashes.pop(node_id, None)
                self._dirty_nodes.add(node_id)

            if self.cache is not None and cache_allowed:
                cache_key = self._compute_cache_key(
                    functor,
                    step,
                    state,
                    time,
                    dt,
                    kwargs,
                )
                if cache_key:
                    cached = self.cache.get(cache_key)
                    if cached is not None and not self._is_output_cache_eligible(cached):
                        self.cache.invalidate(node_id)
                        cached = None
                    if cached is not None:
                        if self._recorder is not None:
                            self._recorder.emit_lineage_event(
                                node_id=node_id,
                                event_type="CACHE_HIT",
                                cache_key=cache_key,
                            )
                        if self._strategy == ExecutionStrategy.DIRTY:
                            self._store_last_output(node_id, cached)
                            self._update_dirty_tracking(node_id, step, state, time, dt)
                        return cached

            if self._recorder is not None:
                self._recorder.emit_lineage_event(
                    node_id=node_id,
                    event_type="START",
                    cache_key=cache_key,
                )

            rng_before = (
                _capture_rng_state(self._simulator)
                if cache_allowed
                and (
                    self.cache is not None
                    or self._strategy == ExecutionStrategy.DIRTY
                )
                else None
            )
            # Declared input projection: the kernel receives the state keys
            # its plan step binds; any other key of the shared state is no
            # data dependency of this node (_declared_state_keys).
            result = module_call(
                {key: state[key] for key in declared_keys if key in state},
                time=time,
                dt=dt,
                **kwargs,
            )

            if rng_before is not None:
                rng_after = _capture_rng_state(self._simulator)
                if not self._rng_states_equal(rng_before, rng_after):
                    raise RuntimeError(
                        f"Graph node '{node_id}' declared deterministic_pure "
                        "but consumed runtime RNG state."
                    )

            result_cacheable = self._is_output_cache_eligible(result)
            if (
                self.cache is not None
                and cache_key
                and result_cacheable
            ):
                # The node opted into deterministic_pure caching; a result that
                # cannot be persisted (unpicklable output, cache I/O failure)
                # raises instead of silently running uncached.
                self.cache.put(cache_key, result)

            if self._recorder is not None:
                self._recorder.emit_lineage_event(
                    node_id=node_id,
                    event_type="COMPLETE",
                    cache_key=cache_key,
                )
            if (
                self._strategy == ExecutionStrategy.DIRTY
                and cache_allowed
                and result_cacheable
            ):
                self._store_last_output(node_id, result)
                self._update_dirty_tracking(node_id, step, state, time, dt)
            elif self._strategy == ExecutionStrategy.DIRTY:
                self._last_outputs.pop(node_id, None)
                self._last_input_hashes.pop(node_id, None)
                self._dirty_nodes.add(node_id)
            return result

        functor.set_call_governor(governed_call)

    def _compute_cache_key(
        self,
        functor: Any,
        step: Any,
        state: dict[str, Tensor],
        time: float,
        dt: Tensor,
        kwargs: dict[str, Any],
    ) -> str | None:
        if self.cache is None:
            return None
        node_id = str(step.node_id)
        kernel_identity = functor.get_kernel_identity()
        # The key binds every contract the architecture requires (inputs,
        # parameters, execution plan, provider binding). A contract that
        # cannot be hashed raises: the opted-in node is never run with its
        # caching silently disabled.
        declared_state = self._declared_state_inputs(step, state)
        inputs_payload = {
            "state": self._hash_value(declared_state),
            "time": float(time),
            "dt": self._hash_value(dt),
            "kwargs": self._hash_value(kwargs),
        }
        inputs_hash = self.cache.hash_object(inputs_payload)
        params_payload = {}
        if hasattr(functor, "state_dict") and callable(functor.state_dict):
            params_payload = self._hash_value(functor.state_dict())
        params_hash = self.cache.hash_object(params_payload)
        bound_code_hash = self.cache.hash_object(
            {
                "kernel_identity": kernel_identity,
                "execution_contract": self._hash_value(
                    self._cache_execution_contract(functor, step)
                ),
                "provider_binding": self._provider_binding_contract(node_id),
            }
        )
        return self.cache.compute_key(
            node_id,
            inputs_hash,
            params_hash,
            bound_code_hash,
        )

    def _is_cache_eligible(
        self,
        functor: FunctorBase,
        step: Any,
        state: Mapping[str, Any],
        dt: Any,
        kwargs: Mapping[str, Any],
    ) -> bool:
        config = step.config if isinstance(step.config, Mapping) else {}
        policy = config.get("cache_policy")
        if policy is None:
            return False
        if policy != _DETERMINISTIC_PURE:
            raise ValueError(
                f"Graph node '{step.node_id}' has unsupported cache_policy "
                f"'{policy}'."
            )
        classification = getattr(functor, "cache_classification", None)
        if classification != _DETERMINISTIC_PURE:
            raise ValueError(
                f"Graph node '{step.node_id}' requests deterministic_pure "
                "caching without the matching implementation declaration."
            )
        replay_critical_fields = sorted(
            name
            for name in vars(functor)
            if name not in _CACHE_RUNTIME_SHELL_FIELDS
        )
        if replay_critical_fields:
            raise ValueError(
                f"Graph node '{step.node_id}' declares deterministic_pure "
                "caching but owns replay-critical instance state: "
                + ", ".join(replay_critical_fields)
                + "."
            )
        # Hooks can change inputs, outputs, gradients, or external state.
        # Execute the standard PyTorch call whenever any local/global hook is
        # active; replaying a cached value would skip that invocation contract.
        if (
            functor._forward_pre_hooks
            or functor._forward_hooks
            or functor._backward_pre_hooks
            or functor._backward_hooks
            or torch_module._global_forward_pre_hooks
            or torch_module._global_forward_hooks
            or torch_module._global_backward_pre_hooks
            or torch_module._global_backward_hooks
        ):
            return False
        metadata = getattr(functor, "metadata", None)
        if getattr(metadata, "parameters", None):
            return False
        if getattr(functor, "_parameter_manager", None) is not None:
            return False
        if functor.state_dict():
            return False
        if any(parameter.requires_grad for parameter in functor.parameters()):
            return False
        declared_state = self._declared_state_inputs(step, state)
        return not self._has_autograd_dependency(
            {"state": declared_state, "dt": dt, "kwargs": kwargs}
        )

    def _cache_execution_contract(
        self,
        functor: Any,
        step: Any,
    ) -> dict[str, Any]:
        report = getattr(self.plan, "report", None)
        report_metadata = getattr(report, "metadata", {})
        definition = (
            self.registry.get_definition(step.class_type)
            if self.registry is not None
            else None
        )
        definition_payload = (
            definition.to_dict()
            if definition is not None and hasattr(definition, "to_dict")
            else None
        )
        metadata = getattr(functor, "metadata", None)
        metadata_to_dict = getattr(metadata, "to_dict", None)
        functor_metadata = (
            metadata_to_dict()
            if callable(metadata_to_dict)
            else None
        )
        plan_steps = [
            {
                "node_id": item.node_id,
                "class_type": item.class_type,
                "inputs": item.inputs,
                "outputs": item.outputs,
                "config": item.config,
            }
            for item in self.plan.steps
        ]
        return {
            "cache_contract_version": _NODE_CACHE_CONTRACT_VERSION,
            "classification": _DETERMINISTIC_PURE,
            "node": {
                "node_id": step.node_id,
                "class_type": step.class_type,
                "inputs": step.inputs,
                "outputs": step.outputs,
                "config": step.config,
            },
            "provider_contract": definition_payload,
            "functor_metadata": functor_metadata,
            "execution_plan": {
                "steps": plan_steps,
                "metadata": report_metadata,
            },
            "runtime_contract": {
                "simulator_config": self._simulator.config,
                "device": str(self._simulator.device),
            },
        }

    @classmethod
    def _is_output_cache_eligible(cls, result: Any) -> bool:
        """Replay only values without control requests or an autograd graph."""
        outputs = result if isinstance(result, Mapping) else {"result": result}
        try:
            _, _, intents = partition_outputs(outputs)
        except (TypeError, ValueError):
            # Invalid old entries are evicted by the cache owner. Fresh invalid
            # outputs still reach the canonical lifecycle validation and fail.
            return False
        return not intents and not cls._has_autograd_dependency(result)

    @classmethod
    def _has_autograd_dependency(cls, value: Any) -> bool:
        if isinstance(value, Tensor):
            return bool(value.requires_grad or value.grad_fn is not None)
        if isinstance(value, UpdateItem):
            return cls._has_autograd_dependency(value.value)
        if isinstance(value, Mapping):
            return any(cls._has_autograd_dependency(item) for item in value.values())
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return any(cls._has_autograd_dependency(item) for item in value)
        if isinstance(value, set):
            return any(cls._has_autograd_dependency(item) for item in value)
        return False

    @staticmethod
    def _rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        if left["python"] != right["python"]:
            return False
        if not bool((left["torch_cpu"] == right["torch_cpu"]).all()):
            return False
        left_cuda = left["torch_cuda"]
        right_cuda = right["torch_cuda"]
        if len(left_cuda) != len(right_cuda):
            return False
        if any(
            not bool((left_state == right_state).all())
            for left_state, right_state in zip(left_cuda, right_cuda)
        ):
            return False
        left_numpy = left["numpy"]
        right_numpy = right["numpy"]
        if (left_numpy is None) != (right_numpy is None):
            return False
        if left_numpy is None:
            return True
        return bool(
            left_numpy["algorithm"] == right_numpy["algorithm"]
            and (left_numpy["keys"] == right_numpy["keys"]).all()
            and left_numpy["position"] == right_numpy["position"]
            and left_numpy["has_gauss"] == right_numpy["has_gauss"]
            and left_numpy["cached_gaussian"]
            == right_numpy["cached_gaussian"]
        )

    def _hash_value(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return {
                "tensor": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "hash": self._hash_tensor(value),
            }
        if isinstance(value, dict):
            keys = list(value.keys())
            keys.sort(key=lambda item: str(item))
            return {str(k): self._hash_value(value[k]) for k in keys}
        if isinstance(value, (list, tuple)):
            return [self._hash_value(v) for v in value]
        return value

    @staticmethod
    def _declared_state_keys(step: Any) -> dict[str, Any]:
        """Return the state key of every input the plan step binds.

        References: GraphIR input bindings (inputs.<port>.link and
          inputs.<port>.state); docs/source/architecture_overview.md
          Section 3 (graph entry path).
        Mathematical expression: K_n = {k_p : p in inputs(n)}, where k_p is
          the configured key of a state binding and p itself otherwise; the
          kernel of node n receives {(k, s[k]) : k in K_n and k in dom(s)}.
        Parameter meanings: n plan step; p input port; s shared state.
        Units: not applicable (names only).
        Domain/constraints: strict validation requires k_p = p. A key the
          step does not bind is no data dependency of the node, whether it
          is an optional input of its Functor, an output of another node
          or a runtime bookkeeping key of the shared state.
        Sign convention: not applicable.
        """
        keys: dict[str, Any] = {}
        for input_name, input_spec in step.inputs.items():
            state_key = input_name
            if isinstance(input_spec, Mapping):
                configured_key = input_spec.get("state")
                if isinstance(configured_key, str) and configured_key:
                    state_key = configured_key
            keys[str(input_name)] = state_key
        return keys

    @staticmethod
    def _declared_state_inputs(
        step: Any,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project runtime state onto the node's declared input contract."""
        return {
            input_name: state.get(state_key)
            for input_name, state_key in GraphExecutor._declared_state_keys(
                step
            ).items()
        }

    @staticmethod
    def _hash_tensor(tensor: Tensor) -> str:
        payload = tensor.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def _compute_node_input_hash(
        self,
        step: Any,
        state: dict[str, Tensor],
        time: float,
        dt: Tensor,
    ) -> str:
        relevant_state = self._declared_state_inputs(step, state)
        link_map: dict[str, Any] = {}
        for input_name, input_spec in step.inputs.items():
            if isinstance(input_spec, dict) and "link" in input_spec:
                link = input_spec.get("link")
                if isinstance(link, (list, tuple)) and len(link) == 2:
                    link_map[input_name] = [str(link[0]), str(link[1])]
        payload: dict[str, Any] = {
            "kernel_identity": self._functor_instances[
                str(step.node_id)
            ].get_kernel_identity(),
            "inputs": self._hash_value(relevant_state),
            "links": link_map,
            "provider_binding": self._provider_binding_contract(str(step.node_id)),
        }
        if self._dirty_include_time:
            payload["time"] = float(time)
            payload["dt"] = self._hash_value(dt)
        return NodeCache.hash_object(payload)

    def _mark_all_dirty(self) -> None:
        self._dirty_nodes = set(self._steps_by_id.keys())

    # ------------------------------------------------------------------
    # Provider view consumption (D03): identity-bound keys and closure
    # invalidation. The simulator's committed view is the only identity
    # source; the executor reads it and never mints identity of its own.
    # ------------------------------------------------------------------
    def _canonical_state_key(self, name: str) -> str:
        registry = getattr(self.state_manager, "_contract_registry", None)
        resolver = getattr(registry, "resolve_alias", None)
        if callable(resolver):
            resolved = resolver(str(name))
            if isinstance(resolved, str) and resolved:
                return resolved
        return str(name)

    def _consumer_map(self) -> dict[str, set[str]]:
        """Direct consumers per node: plan links plus shared-state readers."""
        producers_by_key: dict[str, set[str]] = {}
        for step in self.plan.steps:
            node_id = str(step.node_id)
            declared_outputs: set[str] = {str(name) for name in (step.outputs or [])}
            functor = self._functor_instances.get(node_id)
            metadata = getattr(functor, "metadata", None)
            for name in getattr(metadata, "output_variables", None) or []:
                declared_outputs.add(str(name))
            for name in declared_outputs:
                producers_by_key.setdefault(self._canonical_state_key(name), set()).add(node_id)
        consumers: dict[str, set[str]] = {str(step.node_id): set() for step in self.plan.steps}
        for step in self.plan.steps:
            node_id = str(step.node_id)
            for input_name, input_spec in (step.inputs or {}).items():
                state_key = str(input_name)
                if isinstance(input_spec, Mapping):
                    link = input_spec.get("link")
                    if isinstance(link, (list, tuple)) and len(link) == 2:
                        source = str(link[0])
                        if source != node_id:
                            consumers.setdefault(source, set()).add(node_id)
                    configured = input_spec.get("state")
                    if isinstance(configured, str) and configured:
                        state_key = configured
                for producer in producers_by_key.get(self._canonical_state_key(state_key), ()):
                    if producer != node_id:
                        consumers.setdefault(producer, set()).add(node_id)
        return consumers

    @staticmethod
    def _transitive(start: str, edges: Mapping[str, set[str]]) -> set[str]:
        seen: set[str] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for nxt in edges.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    def _upstream_closure(self, node_id: str) -> set[str]:
        consumers = self._consumer_map()
        producers: dict[str, set[str]] = {}
        for source, targets in consumers.items():
            for target in targets:
                producers.setdefault(target, set()).add(source)
        return self._transitive(node_id, producers)

    def _provider_binding_contract(self, node_id: str) -> dict[str, Any]:
        """Identity triple of the node and of its transitive providers."""
        return {
            "provider_identity": self._provider_digests.get(node_id),
            "upstream_providers": {
                upstream: self._provider_digests.get(upstream)
                for upstream in sorted(self._upstream_closure(node_id))
            },
            "rng_contract": "deterministic_pure_no_runtime_rng",
        }

    def invalidate_node_closure(self, node_id: str) -> tuple[str, ...]:
        """Invalidate one node and every transitive consumer of its outputs.

        Consumers are reached through plan links and through shared-state
        reads of the node's declared outputs (aliases resolved through the
        StateManager contract registry). Disk entries, DIRTY reuse, and the
        last recorded input hashes are dropped for every member.
        """
        node_id = str(node_id)
        if node_id not in self._steps_by_id:
            raise KeyError(f"Unknown graph node '{node_id}'")
        closure = {node_id} | self._transitive(node_id, self._consumer_map())
        for member in closure:
            if self.cache is not None:
                self.cache.invalidate(member)
            self._last_outputs.pop(member, None)
            self._last_input_hashes.pop(member, None)
            self._dirty_nodes.add(member)
        return tuple(sorted(closure))

    def _reconcile_provider_view(self) -> tuple[str, ...]:
        """Consume the committed view; invalidate closures of changed providers.

        A new epoch with unchanged identities keeps every cache entry; a
        changed implementation, contract, or binding digest invalidates that
        node's closure so no old-epoch result can serve a new step.
        """
        simulator = self._simulator
        view = getattr(simulator, "_active_step_view", None)
        if view is None:
            view = simulator.get_committed_provider_view()
        if view is None:
            return ()
        key = (id(simulator), int(view.epoch))
        if key == self._provider_view_key:
            return ()
        digests: dict[str, tuple[Any, Any, Any]] = {}
        changed: list[str] = []
        for node_id, functor in self._functor_instances.items():
            functor_name = getattr(functor, "name", None)
            if not isinstance(functor_name, str):
                continue
            provider = view.providers.get(functor_name)
            if provider is None:
                continue
            digest = (
                provider.implementation_digest,
                provider.contract_digest,
                provider.binding_digest,
            )
            digests[node_id] = digest
            previous = self._provider_digests.get(node_id)
            if previous is not None and previous != digest:
                changed.append(node_id)
        self._provider_digests = digests
        self._provider_view_key = key
        invalidated: set[str] = set()
        for node_id in changed:
            invalidated.update(self.invalidate_node_closure(node_id))
        return tuple(sorted(invalidated))

    def _is_node_dirty(
        self,
        node_id: str,
        step: Any,
        state: dict[str, Tensor],
        time: float,
        dt: Tensor,
    ) -> bool:
        if self._strategy == ExecutionStrategy.PUSH:
            return True
        current_hash = self._compute_node_input_hash(step, state, time, dt)
        if node_id in self._dirty_nodes:
            return True
        last_hash = self._last_input_hashes.get(node_id)
        if last_hash is None:
            return True
        return current_hash != last_hash

    def _update_dirty_tracking(
        self,
        node_id: str,
        step: Any,
        state: dict[str, Tensor],
        time: float,
        dt: Tensor,
    ) -> None:
        current_hash = self._compute_node_input_hash(step, state, time, dt)
        self._last_input_hashes[node_id] = current_hash
        self._dirty_nodes.discard(node_id)

    def _store_last_output(self, node_id: str, result: Any) -> None:
        if isinstance(result, dict):
            self._last_outputs[node_id] = dict(result)

    def mark_dirty(self, node_id: str) -> None:
        """Explicitly mark a node as dirty, requiring re-evaluation."""
        self._dirty_nodes.add(node_id)

    def set_strategy(self, strategy: ExecutionStrategy) -> None:
        """Change execution strategy at runtime."""
        self._strategy = strategy
        if strategy == ExecutionStrategy.DIRTY:
            self._last_input_hashes.clear()
            self._last_outputs.clear()
            self._mark_all_dirty()
        else:
            self._dirty_nodes.clear()
            self._last_input_hashes.clear()
            self._last_outputs.clear()

    def _resolve_functor(self, step: Any) -> Any:
        if step.node_id in self._functor_instances:
            return self._functor_instances[step.node_id]
        functor = self._build_functor(step)
        self._attach_dependencies(functor, step)
        self._wrap_functor(functor, step)
        self._functor_instances[step.node_id] = functor
        return functor

    def _required_initial_state_keys(self) -> list[str]:
        """Return compiler-lowered external state keys with strict typing."""

        report = getattr(self.plan, "report", None)
        metadata = getattr(report, "metadata", {})
        lowering_metadata = (
            metadata.get("state_boundary_lowering", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        raw_required_keys = (
            lowering_metadata.get("required_initial_state_keys")
            if isinstance(lowering_metadata, Mapping)
            else None
        )
        if raw_required_keys is None:
            return []
        if not isinstance(raw_required_keys, (list, tuple, set)):
            raise ValueError(
                "State boundary required_initial_state_keys must be a sequence."
            )
        if any(
            not isinstance(key, str) or not key.strip()
            for key in raw_required_keys
        ):
            raise ValueError(
                "State boundary metadata contains an invalid initial state key."
            )
        return sorted(set(raw_required_keys))

    def _validate_required_initial_state(
        self,
        initial_state: Mapping[str, Any],
    ) -> list[str]:
        required_keys = self._required_initial_state_keys()
        missing_keys = [
            key for key in required_keys if key not in initial_state
        ]
        if missing_keys:
            raise ValueError(
                "Missing required initial state keys: "
                + ", ".join(missing_keys)
            )
        return required_keys

    def _prepare_execution(
        self,
        initial_state: dict[str, Tensor],
    ) -> None:
        """Initialize the one canonical simulator lifecycle for a graph run."""

        expected_witness = self._require_executable_plan()
        prepared_functors: list[tuple[Any, Any]] = []
        node_witnesses: list[dict[str, Any]] = []
        for step in self.plan.steps:
            functor = self._build_functor(step)
            node_witnesses.append(self._runtime_node_witness(step, functor))
            self._attach_dependencies(functor, step)
            prepared_functors.append((step, functor))
        actual_witness = self._compile_run_witness_payload(node_witnesses)
        self._assert_runtime_witness(expected_witness, actual_witness)
        self._require_executable_plan()

        self._validate_required_initial_state(initial_state)
        # Confirm accepted effects before discarding their owning simulator.
        # Failures retain the old run, queue, state, and prepared execution.
        self._simulator.retry_external_deliveries()
        self.state_manager.reset()
        self._execution_prepared = False
        if self._strategy == ExecutionStrategy.DIRTY:
            self._last_input_hashes.clear()
            self._last_outputs.clear()
            self._mark_all_dirty()

        self._functor_instances = {}
        # A witness belongs to the simulator it verified.
        self._integration_plan_witness = None
        self._witnessed_plan_payload = None
        self._simulator = RELICSimulator(
            deepcopy(self._simulator_config)
        )
        self.state_manager.to_device(self._simulator.device)

        if initial_state:
            self.state_manager.initialize(initial_state)
        else:
            self.state_manager.initialize()

        dependency_graph = DependencyGraphResolver()
        self._simulator.initialize(self.state_manager, dependency_graph)

        for step, functor in prepared_functors:
            self._wrap_functor(functor, step)
            self._functor_instances[step.node_id] = functor
            execution_group = None
            if isinstance(step.config, dict):
                execution_group = step.config.get("execution_group")
            self._simulator.register_functor(
                functor,
                execution_group=execution_group,
            )
        self._require_executable_plan()
        self._reconcile_provider_view()
        # Runtime witness of a declared IntegrationPlan before any step; a
        # mismatch raises and the run stays unprepared.
        self._verify_integration_plan("prepared")
        self._execution_prepared = True

    @staticmethod
    def _validate_fixed_step_request(
        *,
        n_steps: int,
        dt: float,
    ) -> tuple[int, float]:
        if type(n_steps) is not int or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer.")
        try:
            dt_value = float(dt)
        except (TypeError, ValueError) as exc:
            raise TypeError("dt must be a finite positive scalar.") from exc
        if not math.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("dt must be a finite positive scalar.")
        return n_steps, dt_value

    def execute(
        self,
        initial_state: dict[str, Tensor],
        n_steps: int,
        dt: float,
    ) -> dict[str, Tensor]:
        """Execute the graph for ``n_steps`` fixed steps of ``dt`` seconds.

        Both are stated by the caller: the time step has no general value.
        """
        n_steps, dt_value = self._validate_fixed_step_request(
            n_steps=n_steps,
            dt=dt,
        )
        self._prepare_execution(initial_state)

        for _ in range(n_steps):
            self._step_under_witness(dt_value)
        self._check_executed_stages()

        self._execution_count += 1

        return self.state_manager.get_state()

    def create_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        """Persist the prepared canonical simulator through its public API."""

        if not self._execution_prepared:
            raise RuntimeError(
                "GraphExecutor requires a prepared execution before checkpoint."
            )
        self._require_executable_plan()
        return dict(self._simulator.create_checkpoint(checkpoint_path))

    def execute_from_checkpoint(
        self,
        initial_state: dict[str, Tensor],
        checkpoint_path: str | Path,
        *,
        n_steps: int,
        dt: float,
        compatibility_certificate: Any = None,
    ) -> dict[str, Tensor]:
        """Rebuild the graph lifecycle, restore, and continue fixed steps.

        ``compatibility_certificate`` selects the compatible restart owned by
        the checkpoint module; without it the restore is exact.
        """

        n_steps, dt_value = self._validate_fixed_step_request(
            n_steps=n_steps,
            dt=dt,
        )
        self._prepare_execution(initial_state)
        try:
            self._simulator.restore_checkpoint(
                checkpoint_path,
                compatibility_certificate=compatibility_certificate,
            )
            self._require_executable_plan()
            self._verify_integration_plan("restored")
        except Exception:
            self._execution_prepared = False
            raise

        for _ in range(n_steps):
            self._step_under_witness(dt_value)
        self._check_executed_stages()

        self._execution_count += 1
        return self.state_manager.get_state()

    @staticmethod
    def _forcing_sequence_sha256(snapshots: Sequence[Mapping[str, Tensor]]) -> str:
        """Canonical digest of a normalized forcing sequence (order, keys, values)."""
        digest = hashlib.sha256()
        for index in range(len(snapshots)):
            snapshot = snapshots[index]
            digest.update(f"snapshot:{index}:{len(snapshot)};".encode("ascii"))
            for name in sorted(snapshot):
                tensor = snapshot[name].detach().cpu().contiguous()
                digest.update(
                    f"{name}:{tensor.dtype}:{list(tensor.shape)}:".encode("utf-8")
                )
                digest.update(tensor.numpy().tobytes(order="C"))
                digest.update(b";")
        return digest.hexdigest()

    def _normalize_forcing_sequence(
        self,
        initial_state: dict[str, Tensor],
        forcing_sequence: Sequence[Mapping[str, Tensor]],
        dt: float,
    ) -> tuple[Sequence[Mapping[str, Tensor]], float]:
        required_keys = self._validate_required_initial_state(initial_state)
        if isinstance(forcing_sequence, (str, bytes)) or not isinstance(
            forcing_sequence,
            Sequence,
        ):
            raise TypeError("forcing_sequence must be a non-empty sequence.")
        # References: RELIC architecture Section 5, forcing-sequence contract.
        # Mathematical expression: snapshot_i = source[i], i in [0, len(source));
        #   digest = SHA256(concat_i canonical(snapshot_i)).
        # Parameter meanings: source is a stable indexed forcing Sequence;
        #   i is its zero-based position; digest identifies its ordered content.
        # Units: i and len(source) are counts; tensor units remain port-declared.
        # Domain/constraints: finite nonempty Sequence, repeatable values and
        #   fixed length throughout validation, hashing and execution. Validate
        #   all samples before mutation without retaining a full materialization.
        # Sign convention: no flux changes; this only admits external forcing.
        snapshots = forcing_sequence
        if not snapshots:
            raise ValueError("forcing_sequence must be non-empty.")
        _, dt_value = self._validate_fixed_step_request(
            n_steps=len(snapshots),
            dt=dt,
        )

        allowed_keys = set(required_keys or initial_state)
        for index in range(len(snapshots)):
            snapshot = snapshots[index]
            if not isinstance(snapshot, Mapping):
                raise TypeError(
                    f"forcing_sequence[{index}] must be a mapping."
                )
            unknown = sorted(set(snapshot).difference(allowed_keys))
            if unknown:
                raise ValueError(
                    "Forcing sequence can only replace declared external "
                    f"forcing keys: {unknown}"
                )
            for name, value in snapshot.items():
                template = initial_state.get(name)
                if not isinstance(template, Tensor) or not isinstance(
                    value,
                    Tensor,
                ):
                    raise TypeError(
                        f"External forcing '{name}' must be a tensor."
                    )
                if value.shape != template.shape:
                    raise ValueError(
                        f"External forcing '{name}' shape does not match "
                        "the initial state."
                    )
                if not bool(value.is_floating_point()):
                    raise TypeError(
                        f"External forcing '{name}' must be floating point."
                    )
                if not bool(value.isfinite().all()):
                    raise ValueError(
                        f"External forcing '{name}' must be finite."
                    )
        return snapshots, dt_value

    def _run_forcing_sequence(
        self,
        snapshots: Sequence[Mapping[str, Tensor]],
        dt_value: float,
        *,
        start_index: int,
    ) -> dict[str, Tensor]:
        """Apply snapshots[start_index:] one per accepted step under the cursor."""
        for index in range(start_index, len(snapshots)):
            cursor = self._simulator.get_forcing_cursor()
            if cursor is None or cursor["consumed"] != index:
                raise RuntimeError(
                    "Forcing cursor drifted from the sequence position: "
                    f"expected {index}, cursor {cursor}"
                )
            snapshot = dict(snapshots[index])
            current_state = self.state_manager.get_state()
            device_aligned = {
                name: value.to(
                    device=current_state[name].device,
                    dtype=current_state[name].dtype,
                )
                for name, value in snapshot.items()
            }
            self.state_manager.set_state(
                device_aligned,
                validate=False,
            )
            self._step_under_witness(dt_value)
        self._check_executed_stages()
        self._execution_count += 1
        return self.state_manager.get_state()

    def migrate_from_checkpoint(
        self,
        initial_state: dict[str, Tensor],
        checkpoint_path: str | Path,
        migration_certificate: Any,
        *,
        n_steps: int,
        dt: float,
    ) -> dict[str, Tensor]:
        """Restore a checkpoint of another graph under an explicit migration.

        The target plan is prepared first; the checkpoint owner validates the
        migration certificate against the real saved and live topologies,
        transforms the payload into the target namespace, and commits it
        atomically. Fixed steps then continue on the target graph.
        """

        n_steps, dt_value = self._validate_fixed_step_request(
            n_steps=n_steps,
            dt=dt,
        )
        self._prepare_execution(initial_state)
        try:
            self._simulator.restore_checkpoint(
                checkpoint_path,
                migration_certificate=migration_certificate,
            )
            self._require_executable_plan()
            self._verify_integration_plan("migrated")
        except Exception:
            self._execution_prepared = False
            raise

        for _ in range(n_steps):
            self._step_under_witness(dt_value)
        self._check_executed_stages()

        self._execution_count += 1
        return self.state_manager.get_state()

    def execute_forcing_sequence(
        self,
        initial_state: dict[str, Tensor],
        forcing_sequence: Sequence[Mapping[str, Tensor]],
        *,
        dt: float,
    ) -> dict[str, Tensor]:
        """Execute fixed steps while replacing declared external forcing.

        The consumed position is owned by the simulator forcing cursor, which
        advances only inside accepted steps and is part of the checkpoint
        restart contract, so a checkpoint delivered mid-sequence resumes at
        the next unconsumed snapshot through
        :meth:`execute_forcing_sequence_from_checkpoint`.
        """

        self._require_executable_plan()
        snapshots, dt_value = self._normalize_forcing_sequence(
            initial_state, forcing_sequence, dt,
        )
        self._prepare_execution(initial_state)
        self._simulator.bind_forcing_cursor(
            sequence_sha256=self._forcing_sequence_sha256(snapshots),
            length=len(snapshots),
            consumed=0,
        )
        return self._run_forcing_sequence(snapshots, dt_value, start_index=0)

    def execute_forcing_sequence_from_checkpoint(
        self,
        initial_state: dict[str, Tensor],
        checkpoint_path: str | Path,
        forcing_sequence: Sequence[Mapping[str, Tensor]],
        *,
        dt: float,
    ) -> dict[str, Tensor]:
        """Restore a mid-sequence checkpoint and consume the remaining snapshots.

        The restored cursor must name this exact sequence (digest and length)
        and must not already be exhausted; any other sequence is refused
        before a step runs.
        """
        snapshots, dt_value = self._normalize_forcing_sequence(
            initial_state, forcing_sequence, dt,
        )
        self._prepare_execution(initial_state)
        try:
            self._simulator.restore_checkpoint(checkpoint_path)
            self._require_executable_plan()
            self._verify_integration_plan("restored")
        except Exception:
            self._execution_prepared = False
            raise
        cursor = self._simulator.get_forcing_cursor()
        expected_sha = self._forcing_sequence_sha256(snapshots)
        if cursor is None:
            raise RuntimeError(
                "Checkpoint carries no forcing cursor; it was not taken inside "
                "a forcing sequence run"
            )
        if cursor["sequence_sha256"] != expected_sha or cursor["length"] != len(snapshots):
            raise ValueError(
                "Checkpoint forcing cursor belongs to a different forcing "
                "sequence (digest or length mismatch)"
            )
        if cursor["consumed"] >= cursor["length"]:
            raise ValueError(
                "Checkpoint forcing cursor is already exhausted: "
                f"{cursor['consumed']}/{cursor['length']}"
            )
        return self._run_forcing_sequence(
            snapshots, dt_value, start_index=cursor["consumed"],
        )
