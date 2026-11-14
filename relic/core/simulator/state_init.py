"""
State initialization helpers for RELICSimulator.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

import torch

logger = logging.getLogger(__name__)

# Flat simulator settings and the StateManager conservation_checks key each one
# sets; the keys and their validation belong to StateManager
# (relic.core.state.manager.CONSERVATION_CHECK_KEYS, validate_conservation_*).
_CONSERVATION_FLAT_SETTINGS = (
    ("conservation_raise_on_violation", "raise_on_violation"),
    ("conservation_rollback_on_failure", "rollback_on_failure"),
    ("conservation_tolerance", "tolerance"),
)

def _declared_bounds_pair(raw: Any, name: str) -> Optional[Tuple[float, float]]:
    """Return a declared ``[lower, upper]`` pair; a null side is unbounded (+-inf)."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"Declared bounds of '{name}' must be a [lower, upper] pair, got {raw!r}.")
    lower, upper = raw
    if lower is None and upper is None:
        return None
    return (
        float("-inf") if lower is None else float(lower),
        float("inf") if upper is None else float(upper),
    )


def _caller_state_units(self: Any) -> Dict[str, str]:
    """Return the caller's unit declarations (simulator setting ``state_units``)."""
    get_setting = getattr(self, "_get_setting", None)
    raw = get_setting("state_units") if callable(get_setting) else None
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Simulator setting 'state_units' must map variable names to units, got {type(raw).__name__}.")
    declared: Dict[str, str] = {}
    for name, units in raw.items():
        if not isinstance(name, str) or not isinstance(units, str) or not units.strip() or units != units.strip():
            raise ValueError(f"Simulator setting 'state_units' entry {name!r}: {units!r} must map a name to one exact unit string.")
        declared[name] = units
    return declared


def _declared_state_schema(
    initial_conditions: Dict[str, Any],
    state_variable_spec_cls: Any,
    manager: Any,
    caller_units: Mapping[str, str],
) -> Dict[str, Any]:
    """Build initial-state schema entries from declarations only.

    A variable with an SCM contract gets its contract specification afterwards
    (_apply_contract_schema_overrides replaces the entry). For any other
    variable the declarations are:

    - its SSOT catalog entry (ParameterManager
      ``coupling.external.variables.variable_definitions`` and
      ``coupling.scales.defaults.variables``): units and bounds;
    - ``validation.numerical.state_variable_bounds[<variable>]``: units and
      the ParameterManager keys of process-specific lower and upper bounds,
      read in those units;
    - the caller's ``state_units`` setting: units only.

    Declarations that disagree on units, or two declarations of bounds, raise
    ValueError naming the variable. A variable without declared bounds is
    unbounded and no variable is treated as conserved here (a contract
    declares conservation). Units still undeclared after the contract
    overrides raise in _require_declared_units. Nothing is inferred from the
    variable name.
    """
    from ...orchestration.ssot_catalog import SSOTCatalog

    catalog = SSOTCatalog.from_parameter_manager(manager)
    process_bounds = manager.get(
        "validation.numerical.state_variable_bounds",
        expected_type=dict,
        required=True,
    )
    schema: Dict[str, Any] = {}
    for var_name in initial_conditions:
        units: Optional[str] = None
        bounds: Optional[Tuple[float, float]] = None
        entry = catalog.get(var_name)
        if entry is not None:
            units = entry.units
            bounds = _declared_bounds_pair(entry.raw.get("bounds"), var_name)
        declaration = process_bounds.get(var_name)
        if declaration is not None:
            declared_units = str(declaration["units"])
            if units is not None and units != declared_units:
                raise ValueError(
                    f"State variable '{var_name}' is declared in '{units}' by its SSOT catalog "
                    f"entry and in '{declared_units}' by validation.numerical.state_variable_bounds."
                )
            if bounds is not None:
                raise ValueError(
                    f"Bounds of state variable '{var_name}' are declared by its SSOT catalog entry "
                    "and by validation.numerical.state_variable_bounds; keep one declaration."
                )
            units = declared_units
            bounds = (
                float(manager.get(declaration["lower"], units=declared_units, expected_type=float, required=True)),
                float(manager.get(declaration["upper"], units=declared_units, expected_type=float, required=True)),
            )
        caller = caller_units.get(var_name)
        if caller is not None:
            if units is not None and units != caller:
                raise ValueError(
                    f"State variable '{var_name}' is declared in '{units}'; the simulator setting "
                    f"'state_units' declares '{caller}'."
                )
            units = caller
        schema[var_name] = state_variable_spec_cls(
            name=var_name,
            units=units,
            bounds=bounds,
            conserved=False,
            spatial_dims=0,
            required=True,
        )
    return schema


def _apply_contract_schema_overrides(
    state_manager: Any,
    initial_conditions: Dict[str, Any],
) -> None:
    """Replace the entries of contracted variables with their SCM contract specs.

    A declared contract alias takes the declaration of its canonical contract
    under its own name.
    """
    registry = getattr(state_manager, "_contract_registry", None)
    if registry is None:
        return

    for var_name in initial_conditions:
        contract_spec = registry.to_state_variable_spec(registry.resolve_alias(var_name))
        if contract_spec is None:
            continue
        contract_spec = dataclasses.replace(contract_spec, name=var_name, required=True, spatial_dims=0)
        state_manager.register_variable(contract_spec)


def _require_declared_units(
    state_manager: Any,
    initial_conditions: Dict[str, Any],
    caller_units: Mapping[str, str],
) -> None:
    """Every initial-state variable must have declared units after contract overrides."""
    undeclared = []
    for var_name in initial_conditions:
        spec = state_manager._schema.get(var_name)
        declared = None if spec is None else spec.units
        caller = caller_units.get(var_name)
        if caller is not None and declared is not None and declared != caller:
            raise ValueError(
                f"State variable '{var_name}' is declared in '{declared}' by its SCM contract; "
                f"the simulator setting 'state_units' declares '{caller}'."
            )
        if not declared:
            undeclared.append(var_name)
    if undeclared:
        raise ValueError(
            f"Initial-state variables without declared units: {sorted(undeclared)}. Declare them in an "
            "SCM contract (relic/contracts/variables), the SSOT variable catalog, "
            "validation.numerical.state_variable_bounds, or the simulator setting 'state_units'."
        )


def _resolve_conservation_checks(get_setting: Callable[..., Any]) -> Dict[str, Any]:
    """Merge the simulator settings into one StateManager ``conservation_checks`` mapping.

    ``conservation_checks`` (a bool or a mapping) and the flat settings
    ``conservation_raise_on_violation``, ``conservation_rollback_on_failure`` and
    ``conservation_tolerance`` are validated by StateManager's own rules
    (``validate_conservation_checks`` / ``validate_conservation_setting``), with the
    simulator key named in any error. The only rule here is the merge: a flat
    setting that contradicts the same key of the mapping raises instead of being
    ignored. Unset keys keep the StateManager defaults.
    """
    from ..state.manager import validate_conservation_checks, validate_conservation_setting

    conservation_cfg = validate_conservation_checks(
        get_setting("conservation_checks", True), source="conservation_checks"
    )
    for setting, key in _CONSERVATION_FLAT_SETTINGS:
        value = get_setting(setting, None)
        if value is None:
            continue
        value = validate_conservation_setting(key, value, source=setting)
        if key in conservation_cfg and conservation_cfg[key] != value:
            raise ValueError(
                f"Simulator setting '{setting}' = {value!r} contradicts "
                f"'conservation_checks.{key}' = {conservation_cfg[key]!r}; set one of them."
            )
        conservation_cfg[key] = value
    return conservation_cfg


def _ensure_state_manager_initialized(
    self: Any,
    initial_conditions: Dict[str, Any],
) -> None:
    if self.state_manager is not None:
        return

    from ..state import StateManager
    from ..state.types import StateVariableSpec
    from ...parameters import get_global_manager

    manager = get_global_manager()
    if manager is None:
        raise RuntimeError(
            "ParameterManager is unavailable during state initialization. "
            "Cannot resolve the declared initial-state schema."
        )

    caller_units = _caller_state_units(self)
    schema = _declared_state_schema(initial_conditions, StateVariableSpec, manager, caller_units)
    get_setting = getattr(self, "_get_setting", None)
    if callable(get_setting):
        conservation_cfg = _resolve_conservation_checks(get_setting)
    else:
        conservation_cfg = {"enabled": True}

    state_device = getattr(self, "device", None)
    state_manager = StateManager(
        schema,
        conservation_checks=conservation_cfg,
        device=state_device,
    )
    _apply_contract_schema_overrides(state_manager, initial_conditions)
    _require_declared_units(state_manager, initial_conditions, caller_units)
    self.state_manager = state_manager


def _ensure_dependency_graph_initialized(self: Any) -> None:
    if self.dependency_graph is not None:
        return
    from ..dependency_graph import DependencyGraphResolver

    self.dependency_graph = DependencyGraphResolver()


def _to_initial_state_tensor_dict(initial_conditions: Dict[str, Any]) -> Dict[str, Any]:
    initial_state: Dict[str, Any] = {}
    for var_name, value in initial_conditions.items():
        if isinstance(value, (int, float)):
            value = torch.tensor(float(value))
        initial_state[var_name] = value
    return initial_state


def _initialize_time_tracking(self: Any) -> None:
    compute_interval_index = getattr(self, "_compute_interval_index", None)
    if not callable(compute_interval_index):
        from .time_ops import compute_interval_index as _compute_interval_index

        compute_interval_index = _compute_interval_index

    if self.current_time == 0.0:
        self._previous_day_index = self._compute_day_index(self.current_time)
        self._previous_month_index = self._compute_month_index(self.current_time)

    self._previous_medium_index = compute_interval_index(
        self.current_time,
        self._medium_interval_seconds,
    )

    slow_interval_seconds = getattr(self, "_slow_interval_seconds", None)
    if slow_interval_seconds is not None and float(slow_interval_seconds) > 0.0:
        self._previous_slow_index = compute_interval_index(
            self.current_time,
            float(slow_interval_seconds),
        )
    else:
        self._previous_slow_index = -1

    self._last_medium_execution_time = self.current_time
    self._last_slow_execution_time = self.current_time


def _resolve_bootstrap_specs(self: Any) -> List[str]:
    """Return the requested bootstrap functors.

    An explicit ``bootstrap_functors`` setting (a string or a list, possibly
    empty) is used as given; only an absent setting requests the default
    name: the ``bootstrap_functor_name`` setting, else the declared parameter
    ``computation.simulator_bootstrap.default_functor_name``. Names are
    resolved by the caller, so a requested name that is not registered raises
    there with the name.
    """
    bootstrap_specs = self._get_setting("bootstrap_functors")
    requested: List[Any]
    if bootstrap_specs is None:
        default_name = self._get_setting("bootstrap_functor_name")
        if default_name is None:
            from ...parameters import get_global_manager

            default_name = get_global_manager().get(
                "computation.simulator_bootstrap.default_functor_name",
                expected_type=str,
                required=True,
            )
        requested = [default_name]
    elif isinstance(bootstrap_specs, str):
        requested = [bootstrap_specs]
    else:
        requested = list(bootstrap_specs)
    # Every request is checked before any is registered, so an invalid entry
    # never leaves the valid ones before it registered.
    names: List[str] = []
    for spec in requested:
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(f"Invalid bootstrap functor request: {spec!r}")
        names.append(spec)
    return names


def _register_bootstrap_functors(
    self: Any,
    get_functor_path: Callable[[str], str],
) -> None:
    for spec in _resolve_bootstrap_specs(self):
        # A registry name is resolved (KeyError names a missing functor) and is
        # kept as the registration name, which is the contract writer name; an
        # explicit 'module:Class' path registers under FUNCTOR_NAME or the class name.
        requested_name: Optional[str] = None
        path = spec
        if ":" not in spec:
            requested_name = spec
            path = get_functor_path(spec)

        module_path, separator, class_name = path.partition(":")
        if not separator or not module_path or not class_name or ":" in class_name:
            raise ValueError(f"Invalid bootstrap functor path '{path}' for request '{spec}'")

        functor_cls = getattr(importlib.import_module(module_path), class_name)
        functor_name = (
            requested_name
            or getattr(functor_cls, "FUNCTOR_NAME", None)
            or functor_cls.__name__
        )
        self.register_functor(functor_cls(functor_name), execution_group="fast")


def _maybe_register_bootstrap_functors(self: Any) -> None:
    if len(self.functors) != 0:
        return
    if not self._get_setting("enable_bootstrap_modules", False):
        return
    if not self._allow_bootstrap_module_registration():
        raise RuntimeError(
            "enable_bootstrap_modules is test-only. "
            "Set RELIC_ALLOW_BOOTSTRAP_MODULES=true to opt in explicitly."
        )

    scientific_module = importlib.import_module("relic.scientific")
    get_functor_path_attr = getattr(scientific_module, "get_functor_path", None)
    if not callable(get_functor_path_attr):
        raise RuntimeError("relic.scientific.get_functor_path is unavailable")
    get_functor_path = cast(Callable[[str], str], get_functor_path_attr)
    _register_bootstrap_functors(self, get_functor_path)


def initialize_state(
    self: Any,
    initial_conditions: Dict[str, Any],
    simulation_state_enum: Any,
) -> None:
    """
    Initialize simulator state and bootstrap optional test functors.
    """
    _ensure_state_manager_initialized(self, initial_conditions)
    _ensure_dependency_graph_initialized(self)

    initial_state = _to_initial_state_tensor_dict(initial_conditions)

    self.state_manager.set_state(initial_state, validate=True)
    self._initial_state = initial_state.copy()

    if self.state != simulation_state_enum.INITIALIZED:
        self.initialize(self.state_manager, self.dependency_graph)
    else:
        # initialize(None, ...) deferred the ScaleManager lifecycle; the
        # StateManager owner exists now (idempotent when already READY).
        self._ensure_scale_manager_ready()

    logger.info("Simulator state initialized with %s variables", len(initial_conditions))

    _initialize_time_tracking(self)
    _maybe_register_bootstrap_functors(self)

