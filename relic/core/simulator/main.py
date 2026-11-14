"""
RELIC Core Simulator - Minimalist Main Loop Engine

This module implements the core simulation engine with minimal responsibilities:
main time-stepping loop, functor orchestration, and high-level coordination.
All complex logic has been extracted to dedicated modules for better separation
of concerns and maintainability.

The simulator follows the single responsibility principle by focusing solely
on temporal coordination and functor execution ordering. State management,
dependency resolution, and category theory implementation are handled by
specialized modules.

Architecture:
    - Minimal main loop with time stepping
    - Functor registration and execution coordination
    - High-level simulation control (start, pause, stop, reset)
    - Integration with StateManager, DependencyGraph, and configuration system
    - Performance monitoring and checkpoint management
    - Multi-scale coupling integration with automatic regridding

Mathematical Expression:
    Time integration: y(t+dt) = y(t) + int[t,t+dt] F(y(tau), tau) dtau
    where F represents the composition of all functors: F = F_n o ... o F_1

    Operator splitting: F = F_transport + F_biological + F_coupling + F_physical
    Time stepping: y^(n+1) = S_dt[F](y^n) where S_dt is the time stepping operator

    Multi-scale coupling: y_scaled = R(y) where R is the regridding operator
    Conservation check: sum(y_scaled - y_original) = 0 for conserved quantities

References:
    - Hairer, E. et al. (1993). Solving Ordinary Differential Equations I. Springer
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of Time-Dependent Systems
    - Ascher, U.M. & Petzold, L.R. (1998). Computer Methods for Ordinary Differential Equations
    - Press, W.H. et al. (2007). Numerical Recipes - Integration of ODEs
    - Jones, P.W. (1999). Conservative remapping schemes for general grids
"""

import importlib
import importlib.util
import json
import logging
import math
import os
import re
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING, cast
from pathlib import Path
from enum import Enum
from importlib import import_module
from numbers import Integral, Real
from uuid import uuid4

import torch
from relic.core import device_policy
import torch.nn as nn
from torch import Tensor

# Import core components
from ..state import StateManager, UpdateMode
from ..dependency_graph import (
    DependencyGraphResolver as DependencyGraph,
    normalize_dependency_declaration,
)
from ..functor import FunctorBase
from ..abstractions import StateDict, FluxDict
from .checkpoint import (
    create_checkpoint as _create_checkpoint_impl,
    restore_checkpoint as _restore_checkpoint_impl,
)
from .diagnostics import (
    collect_functor_diagnostics as _collect_functor_diagnostics_impl,
    get_coupling_scales_statistics as _get_coupling_scales_statistics_impl,
    get_last_conservation_summary as _get_last_conservation_summary_impl,
    get_simulation_summary as _get_simulation_summary_impl,
    export_diagnostics as _export_diagnostics_impl,
)
from .dependency_ops import (
    determine_execution_group as _determine_execution_group_impl,
    find_missing_required_dependencies as _find_missing_required_dependencies_impl,
    refresh_conservation_ledger as _refresh_conservation_ledger_impl,
)
from .device_ops import get_device_info as _get_device_info_impl
from .execution_group import (
    GroupComposition,
    execute_group as _execute_group_impl,
    resolve_call_substep_count as _resolve_call_substep_count_impl,
)
from .execution_loop import (
    _apply_distributed_allreduce as _apply_distributed_allreduce_impl,
    plan_exchange_means as _plan_exchange_means_impl,
    register_exchange_means as _register_exchange_means_impl,
    retry_external_deliveries as _retry_external_deliveries_impl,
    step_simulation as _step_simulation_impl,
    run_simulation as _run_simulation_impl,
    splitting_admissibility_report as _splitting_admissibility_report_impl,
    splitting_order_report as _splitting_order_report_impl,
)
from .ensemble_ops import (
    run_ensemble_simulation as _run_ensemble_impl,
    sample_parameters as _sample_parameters_impl,
)
from .init_ops import (
    configure_calendar_settings as _configure_calendar_settings_impl,
    initialize_diagnostic_export as _initialize_diagnostic_export_impl,
    initialize_distributed_runtime as _initialize_distributed_runtime_impl,
    initialize_scale_manager as _initialize_scale_manager_impl,
)
from .registration import (
    CommittedProviderView,
    LifecycleReceipt,
    ProviderViewToken,
    TopologyTransition,
    _ProviderLifecycleState,
    _binding_payload as _binding_payload_impl,
    _contract_payload as _contract_payload_impl,
    _provider_digest as _provider_digest_impl,
    _publish_provider_view as _publish_provider_view_impl,
    _resolve_execution_group as _resolve_execution_group_impl,
    acquire_provider_view as _acquire_provider_view_impl,
    apply_topology_transition as _apply_topology_transition_impl,
    declare_topology_transition as _declare_topology_transition_impl,
    register_functor as _register_functor_impl,
    unregister_functor as _unregister_functor_impl,
)
from .regridding_ops import (
    perform_regridding as _perform_regridding_impl,
    validate_auto_scale_grid_descriptors as _validate_auto_scale_grid_descriptors_impl,
)
from .regrid_strategy_ops import determine_regrid_strategy as _determine_regrid_strategy_impl
from .scale_ops import (
    check_scale_consistency as _check_scale_consistency_impl,
    compute_group_execution_orders as _compute_group_execution_orders_impl,
)
from .state_init import (
    _initialize_time_tracking,
    initialize_state as _initialize_state_impl,
)
from .state_ops import (
    apply_updates as _apply_updates_impl,
    get_current_state as _get_current_state_impl,
    set_state_manager as _set_state_manager_impl,
    store_outputs as _store_outputs_impl,
)
from .time_ops import (
    compute_day_index as _compute_day_index_impl,
    compute_interval_index as _compute_interval_index_impl,
    check_event_triggers as _check_event_triggers_impl,
    should_execute_medium as _should_execute_medium_impl,
    should_execute_slow as _should_execute_slow_impl,
)
from .validation_ops import (
    validate_cross_domain_coupling as _validate_cross_domain_coupling_impl,
)
from ...parameters.ops.snapshots import (
    compute_parameter_diff as _compute_parameter_diff_impl,
    generate_experiment_diff_report as _generate_experiment_diff_report_impl,
)

if TYPE_CHECKING:
    from ...computation.solvers.geometric import IntegrationPlan
    from ...data.io.writer import WriteResult
    from ...stochastic.ensemble_runner import EnsembleResults
    from ..state.types import ConservationSummary
    from ..irf_specs import IrfSpecs, RunSpecs
from ..irf_specs import load_irf_specs, validate_irf_specs
from ...parameters import get_global_manager
from ...utils import time_constants

# Configuration default of the simulator time step [s] (one minute); the one
# source of the time_step default, listed in the simulator configuration table.
_DEFAULT_TIME_STEP_SECONDS = 60.0
# Exchange modes of computation.scheduling.multirate_exchanges (the entry's
# description documents them): hold is the zero-order hold of the committed fast
# value, interval_mean the exact interval mean of the piecewise-constant fast
# trajectory. Interpolation is not implemented and is refused by name.
_MULTIRATE_EXCHANGE_MODES = ("hold", "interval_mean")
_MULTIRATE_EXCHANGE_KEYS = ("id", "variable", "consumer_group", "mode")
# An exchange id names the StateManager variable of its interval mean
# (exchange_mean__<id>) and its owner identity (exchange:<id>), so it is an
# identifier: no dots, separators or leading digit.
_MULTIRATE_EXCHANGE_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
from ...parameters.namespaces import ParameterNamespace

def _load_optional_symbols(
    module_name: str,
    symbol_names: Tuple[str, ...],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Load optional symbols from a module with lightweight pre-check.

    Parameters
    ----------
    module_name : str
        Fully qualified module path.
    symbol_names : Tuple[str, ...]
        Required symbols to load from the module.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        Availability flag and symbol map.
    """
    empty_symbols = {name: None for name in symbol_names}
    # Absence of the module (no spec, or a missing parent package) is the only
    # optional outcome. A present module is imported and its errors propagate;
    # a present module lacking a declared symbol is a packaging error.
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return False, empty_symbols

    module = import_module(module_name)
    missing = [name for name in symbol_names if getattr(module, name, None) is None]
    if missing:
        raise ImportError(f"{module_name} does not define {missing}")
    return True, {name: getattr(module, name) for name in symbol_names}


_has_scale_manager_symbols, _scale_manager_symbols = _load_optional_symbols(
    "relic.coupling.scales.scale_manager",
    ("ScaleManager",),
)
_has_scaling_mode_symbols, _scaling_mode_symbols = _load_optional_symbols(
    "relic.coupling.core",
    ("ScalingMode",),
)
HAS_SCALE_MANAGER = _has_scale_manager_symbols and _has_scaling_mode_symbols
ScaleManager = _scale_manager_symbols["ScaleManager"] if HAS_SCALE_MANAGER else None
ScalingMode = _scaling_mode_symbols["ScalingMode"] if HAS_SCALE_MANAGER else None

CrossDomainValidator = None
HAS_CROSS_DOMAIN_VALIDATOR = False

HAS_DIAGNOSTIC_WRITER, _diagnostic_symbols = _load_optional_symbols(
    "relic.data.io.diagnostic_writer",
    ("DiagnosticWriter",),
)
DiagnosticWriter = _diagnostic_symbols["DiagnosticWriter"]

HAS_DISTRIBUTED, _distributed_symbols = _load_optional_symbols(
    "relic.computation.distributed.launcher",
    (
        "ProcessGroupManager",
        "get_process_group_manager",
        "is_distributed",
        "get_rank",
        "get_world_size",
    ),
)
ProcessGroupManager = _distributed_symbols["ProcessGroupManager"]
get_process_group_manager = _distributed_symbols["get_process_group_manager"]
is_distributed = _distributed_symbols["is_distributed"]
get_rank = _distributed_symbols["get_rank"]
get_world_size = _distributed_symbols["get_world_size"]

# Allow tests to override availability flags via relic.core.simulator module attributes.
def _get_package_flag(flag_name: str, default_value: bool) -> bool:
    """Return a flag from the simulator package if overridden by tests."""
    try:
        package_module = import_module(__package__ or "relic.core.simulator")
        if hasattr(package_module, flag_name):
            return bool(getattr(package_module, flag_name))
    except (ModuleNotFoundError, AttributeError, ValueError, TypeError):
        pass
    return bool(default_value)


def _has_scale_manager() -> bool:
    """Check ScaleManager availability, honoring package-level overrides."""
    return _get_package_flag("HAS_SCALE_MANAGER", HAS_SCALE_MANAGER)


def _has_distributed() -> bool:
    """Check distributed availability, honoring package-level overrides."""
    return _get_package_flag("HAS_DISTRIBUTED", HAS_DISTRIBUTED)

# Type aliases
SimulatorConfig = Dict[str, Any]
TimeSeriesData = Dict[str, List[Tensor]]
CheckpointData = Dict[str, Any]

logger = logging.getLogger(__name__)


def _resolve_runtime_device(config: Optional[SimulatorConfig]) -> torch.device:
    """
    Resolve simulator runtime device with CUDA-first policy and CPU fallback.

    Policy:
    - Default (device unset/auto): use CUDA when available, else CPU.
    - Explicit CUDA request: use CUDA if available; otherwise fallback to CPU unless require_gpu=True.
    - Explicit CPU request: use CPU; raise if require_gpu=True.
    """
    cfg = config or {}

    def _get_setting_value(key: str, default: Any) -> Any:
        if key in cfg:
            return cfg[key]
        for section_name in ("simulator", "runtime", "computation"):
            section = cfg.get(section_name)
            if isinstance(section, Mapping) and key in section:
                return section[key]
        return default

    requested_raw = _get_setting_value("device", "auto")
    requested = str(requested_raw).strip().lower()
    require_gpu = bool(_get_setting_value("require_gpu", False))

    def _auto_pick() -> torch.device:
        return device_policy.resolve_device(
            explicit_device="auto",
            module_name=__name__,
            distributed=True,
            require_gpu=require_gpu,
        )

    if requested in {"", "auto", "default"}:
        return _auto_pick()

    if requested.startswith("cuda"):
        if device_policy.is_cuda_available():
            return device_policy.resolve_device(
                explicit_device=requested,
                module_name=__name__,
                require_gpu=True,
            )
        if require_gpu:
            raise RuntimeError(
                f"Configured device '{requested}' requires CUDA, but CUDA is not available."
            )
        logger.warning(
            "Configured device '%s' unavailable; falling back to CPU.",
            requested,
        )
        return torch.device("cpu")

    if requested == "cpu":
        if require_gpu:
            raise RuntimeError(
                "require_gpu=True conflicts with configured device='cpu'."
            )
        return torch.device("cpu")

    logger.warning(
        "Unknown device setting '%s'; using CUDA-first auto selection.",
        requested_raw,
    )
    return _auto_pick()


def _resolve_single_device(device_str: str) -> torch.device:
    """
    Resolve a single device string with availability check.

    Used for per-functor device override parsing in SimulatorConfig.

    Parameters
    ----------
    device_str : str
        Device specification: "auto", "cpu", "cuda", "cuda:0", etc.

    Returns
    -------
    torch.device

    Raises
    ------
    RuntimeError
        If the requested device is not available.
    """
    device_str = device_str.strip().lower()
    if device_str in {"", "auto", "default"}:
        return device_policy.resolve_device(default="auto")
    if device_str.startswith("cuda"):
        if device_policy.is_cuda_available():
            return device_policy.resolve_device(
                explicit_device=device_str,
                module_name=__name__,
                require_gpu=True,
            )
        raise RuntimeError(
            f"Device '{device_str}' requires CUDA but CUDA is not available"
        )
    if device_str == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"Unknown device specification: '{device_str}'")


def _collect_config_override_map(
    config: Optional[SimulatorConfig],
    *,
    keys: Tuple[str, ...],
) -> Dict[str, Any]:
    """
    Collect override dictionaries from root and nested config sections.

    Supported sections:
    - root config
    - config["simulator"]
    - config["runtime"]
    - config["computation"]
    """
    cfg = config or {}
    merged: Dict[str, Any] = {}

    for key in keys:
        value = cfg.get(key)
        if isinstance(value, Mapping):
            merged.update(dict(value))

    for section_name in ("simulator", "runtime", "computation"):
        section = cfg.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for key in keys:
            value = section.get(key)
            if isinstance(value, Mapping):
                merged.update(dict(value))

    return merged


def _load_cross_domain_validator() -> Optional[type]:
    """Lazily import CrossDomainValidator to avoid circular imports at startup."""
    global CrossDomainValidator, HAS_CROSS_DOMAIN_VALIDATOR
    if CrossDomainValidator is not None:
        return CrossDomainValidator
    # RELIC's own validator module: import errors propagate (no silent disable).
    module = import_module("relic.validation.cross_domain_validator")
    CrossDomainValidator = cast(Optional[type], module.CrossDomainValidator)
    HAS_CROSS_DOMAIN_VALIDATOR = True
    return CrossDomainValidator


class SimulationState(Enum):
    """Enumeration of simulation execution states."""
    UNINITIALIZED = "uninitialized"    # Not configured
    INITIALIZED = "initialized"        # Ready to run
    RUNNING = "running"                # Currently executing
    PAUSED = "paused"                  # Temporarily stopped
    COMPLETED = "completed"            # Finished successfully
    FAILED = "failed"                  # Terminated with error


class RELICSimulator(nn.Module):  # type: ignore[misc,unused-ignore]
    """
    Minimal RELIC Core Simulator Engine with Multi-Scale Coupling

    Provides the essential time-stepping simulation framework with clean
    separation of concerns and integrated multi-scale coupling capabilities.

    Key Responsibilities:
        - Time integration loop with configurable time stepping
        - Functor registration and execution orchestration
        - Multi-rate execution (fast/medium/slow groups)
        - Automatic scale checking and regridding
        - Simulation state management (run, pause, stop, reset)
        - Performance monitoring and timing
        - Checkpoint creation and restoration
        - Integration with StateManager and DependencyGraph

    Multi-Scale Coupling Features:
        - Automatic detection of scale mismatches between functors
        - Conservative regridding using ScaleManager
        - Scale-aware execution group management
        - Conservation enforcement across scale transformations

    Mathematical Foundation:
        Main loop: for t = t_0 to t_final with step dt:
            1. state = state_manager.get_current_state()
            2. scale_check = detect_scale_mismatches(state)
            3. if scale_check.needs_regridding:
                   state = regrid(state, scale_check.target_scale)
            4. updates = execute_functors(state, t, dt)
            5. state_manager.apply_updates(updates)
            6. t = t + dt

        Conservation verification: sum updates_i = 0 for conserved quantities
        Scale coupling: y_new = R(y_old) where R preserves sum(y)
    """

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        """
        Initialize the core simulator with multi-scale coupling capabilities.

        Parameters:
            config : SimulatorConfig
                Core simulation parameters:
                - time_step: default time step size [s]
                - max_time: maximum simulation time [s]
                - output_frequency: how often to save outputs [steps]
                - checkpoint_frequency: checkpoint interval [s]
                - enable_checkpointing: opt in to automatic checkpoint writes
                - checkpoint_directory: explicit automatic checkpoint directory
                - enable_auto_scale: enable automatic multi-scale coupling [bool]

        Mathematical Expression:
            Initialization: Simulator(config) -> {state_manager, dependency_graph,
            functors, scale_manager}
            Time domain: T = [0, t_max] with discretization dt
        """
        super().__init__()

        # A missing configuration is the empty configuration: every setting
        # takes the one default of its _get_setting read below.
        self.config = {} if config is None else config.copy()
        self.state = SimulationState.UNINITIALIZED

        self.irf_specs_path = self._get_setting("irf_specs_path")
        self.irf_strict = bool(self._get_setting("irf_strict", False))
        self.strict_dependencies = bool(self._get_setting("strict_dependencies", False))
        self.irf_specs: Optional["IrfSpecs"] = None
        self.run_specs: Optional["RunSpecs"] = None

        # Core components (to be initialized)
        self.state_manager = None
        self.dependency_graph: Optional[DependencyGraph] = None

        # Functor management
        self.functors: nn.ModuleDict = nn.ModuleDict()
        self.execution_order: List[str] = []

        # Multi-rate execution groups
        self._groups: Dict[str, List[str]] = {
            "fast": [],
            "medium": [],
            "slow": []
        }
        self._bootstrap_groups: Set[str] = set()

        # Cached execution orders for each group
        self._order_fast: List[str] = []
        self._order_medium: List[str] = []
        self._order_slow: List[str] = []
        self._groups_dirty: bool = True  # Flag to recompute group orders

        # Provider lifecycle: the registration owner mints UIDs, publishes the
        # immutable committed view, and increments the single epoch counter;
        # the simulator only stores that state and admits step readers.
        self._provider_lifecycle: _ProviderLifecycleState = _ProviderLifecycleState()
        # Ledger of coordinated multi-rank registry transactions (E02).
        self._coordination_outcomes: List[Any] = []
        self._active_step_view: Optional[CommittedProviderView] = None
        # Forcing cursor: position inside an externally supplied forcing
        # sequence. It advances only inside an accepted step, rolls back with
        # the step transaction, and is part of the durable restart contract.
        self._forcing_cursor: Optional[Dict[str, Any]] = None
        # Step bookkeeping owners exist from construction so a rejected first
        # step restores them to their empty pre-step values instead of
        # deleting them.
        self._step_update_sources: Optional[Dict[str, Any]] = {}
        self._step_functor_increments: Dict[str, Any] = {}
        self._step_provenance_detail: Dict[str, Any] = {}
        self._active_composition: Optional[GroupComposition] = None

        # Simulation control
        self.run_id = uuid4().hex
        self._pending_external_deliveries: List[Dict[str, Any]] = []
        self._last_external_delivery_receipts: List[Dict[str, Any]] = []
        self.current_time: float = 0.0
        # Configuration defaults (documented in the simulator configuration
        # table): one minute per step, one day of run time.
        self.time_step: float = self._get_setting('time_step', _DEFAULT_TIME_STEP_SECONDS)
        self.max_time: float = self._get_setting('max_time', time_constants.SECONDS_PER_DAY)
        self.step_count: int = 0
        self.allow_uninitialized_step: bool = self._get_setting('allow_uninitialized_step', False)
        # E2: Adaptive time-stepping configuration. A configured value overrides
        # the parameter-database default computation.scheduling.adaptive.*; the
        # default of adaptive_dt_max is dt_max_factor * time_step.
        self._adaptive_dt_enabled: bool = bool(self._get_setting("adaptive_dt_enabled", False))
        # The controller values are read only when adaptive stepping is enabled
        # (otherwise they are unused and stay None); a missing database key then
        # raises.
        self._adaptive_atol: Optional[float] = None
        self._adaptive_rtol: Optional[float] = None
        self._adaptive_dt_min: Optional[float] = None
        self._adaptive_dt_max: Optional[float] = None
        if self._adaptive_dt_enabled:
            adaptive_database = get_global_manager()
            adaptive_values: Dict[str, float] = {}
            for setting, key in (
                ("adaptive_atol", "computation.scheduling.adaptive.atol"),
                ("adaptive_rtol", "computation.scheduling.adaptive.rtol"),
                ("adaptive_dt_min", "computation.scheduling.adaptive.dt_min_seconds"),
            ):
                configured = self._get_setting(setting)
                adaptive_values[setting] = float(
                    configured if configured is not None else adaptive_database.get(key, required=True)
                )
            configured_dt_max = self._get_setting("adaptive_dt_max")
            self._adaptive_atol = adaptive_values["adaptive_atol"]
            self._adaptive_rtol = adaptive_values["adaptive_rtol"]
            self._adaptive_dt_min = adaptive_values["adaptive_dt_min"]
            self._adaptive_dt_max = float(
                configured_dt_max
                if configured_dt_max is not None
                else float(adaptive_database.get("computation.scheduling.adaptive.dt_max_factor", required=True))
                * self.time_step
            )
        self._adaptive_monitor_vars: Optional[List[str]] = self._get_setting(
            "adaptive_monitor_vars",
            None,
        )
        # References: Hairer, Norsett and Wanner (1993), Solving ODEs I,
        #   Sect. II.4 (tolerances and step-size limits).
        # Mathematical expression: tol_k = atol + rtol |x_k| > 0 requires
        #   atol > 0 and rtol >= 0; the step limits satisfy 0 < dt_min <= dt_max.
        # Parameter meanings: atol, rtol absolute and relative tolerance of the
        #   update-magnitude metric; dt_min, dt_max bounds of the proposed step.
        # Units: atol in state units; rtol dimensionless; dt_min, dt_max [s].
        # Domain/constraints: finite values, checked when adaptive stepping is
        #   enabled (the settings are unused otherwise); violations raise
        #   ValueError here, where the settings are read, so the metric needs no
        #   guard term.
        # Sign convention: all values are non-negative magnitudes.
        adaptive_limits = (
            ("adaptive_atol", self._adaptive_atol, 0.0, False),
            ("adaptive_rtol", self._adaptive_rtol, 0.0, True),
            ("adaptive_dt_min", self._adaptive_dt_min, 0.0, False),
            ("adaptive_dt_max", self._adaptive_dt_max, self._adaptive_dt_min, True),
        )
        for name, value, lower, inclusive in adaptive_limits if self._adaptive_dt_enabled else ():
            valid_bound = value >= lower if inclusive else value > lower
            if not (math.isfinite(value) and valid_bound):
                relation = ">=" if inclusive else ">"
                raise ValueError(
                    f"{name} must be finite and {relation} {lower}, got {value}"
                )
        self._adaptive_next_dt: float = self.time_step
        # E3: Operator splitting controls of the fast-group composition. Each
        # takes its computation.scheduling.* value from the parameter database
        # unless the configuration sets it; a configured value is validated
        # against that entry (allowed_values or structure), so an unknown mode
        # raises ValueError instead of running lie, and a declared control the
        # scheduler does not execute yet raises NotImplementedError instead of
        # being ignored. Admissibility of the configured composition is
        # resolved at every provider-view publication (execution_loop).
        self._splitting_mode: str = self._resolve_integration_control(
            "splitting_mode", "computation.scheduling.splitting_mode"
        )
        self._splitting_policy: str = self._resolve_integration_control(
            "splitting_policy", "computation.scheduling.splitting_policy"
        )
        self._fast_order_mode: str = self._resolve_integration_control(
            "fast_order", "computation.scheduling.fast_order"
        )
        self._substep_overrides: Dict[str, Dict[str, Any]] = self._resolve_substep_overrides()
        self._multirate_exchanges: Tuple[Dict[str, str], ...] = (
            self._resolve_multirate_exchanges()
        )
        # E7: Transport time integration controls routed to Functors that declare
        # them. Unconfigured, each takes its transport.numerical.* value from the
        # parameter database; a configured value overrides and is validated
        # against that entry's allowed_values/bounds (no Python fallback).
        self._time_integration_method: str = self._resolve_integration_control(
            "time_integration_method", "transport.numerical.time_integration_method"
        )
        self._implicit_midpoint_max_iter: int = self._resolve_integration_control(
            "implicit_midpoint_max_iter", "transport.numerical.implicit_midpoint_max_iter"
        )
        self._implicit_midpoint_tolerance: float = self._resolve_integration_control(
            "implicit_midpoint_tolerance", "transport.numerical.implicit_midpoint_tolerance"
        )
        # E5: Auto-resolve stale cross-group dependencies
        self._resolve_cross_group_deps: bool = bool(
            self._get_setting("resolve_cross_group_deps", False)
        )
        # E4: Event-driven triggers {group: [callable(state, t) -> bool]}
        self._event_triggers: Dict[str, List[Callable[[Dict[str, Any], float], bool]]] = {}

        # Calendar settings for month boundary detection (CF-aware minimal)
        self._calendar_type: str
        _configure_calendar_settings_impl(self, logger)

        # Time boundary tracking for multi-rate execution
        self._previous_day_index: int = -1
        self._previous_month_index: int = -1
        self._previous_medium_index: int = -1
        self._previous_slow_index: int = -1
        # Last execution times for medium/slow groups (effective_dt integration windows)
        self._last_medium_execution_time: float = 0.0
        self._last_slow_execution_time: float = 0.0

        # Performance monitoring
        self.total_execution_time: float = 0.0
        self.enable_profiling: bool = bool(
            self._get_setting("enable_profiling", False)
        )
        self._performance_diagnostics: Optional[Dict[str, Any]] = None
        self.output_frequency: int = self._get_setting('output_frequency', 1)
        self.checkpoint_frequency: float = self._get_setting('checkpoint_frequency', time_constants.SECONDS_PER_HOUR)  # 1 hour
        self.enable_checkpointing: bool = bool(
            self._get_setting("enable_checkpointing", False)
        )
        raw_checkpoint_directory = self._get_setting(
            "checkpoint_directory",
            None,
        )
        if raw_checkpoint_directory is None:
            self.checkpoint_directory: Optional[str] = None
        elif isinstance(raw_checkpoint_directory, (str, os.PathLike)):
            self.checkpoint_directory = str(
                Path(raw_checkpoint_directory).expanduser().resolve()
            )
        else:
            raise TypeError(
                "checkpoint_directory must be a filesystem path or None."
            )
        if self.enable_checkpointing and self.checkpoint_directory is None:
            raise ValueError(
                "enable_checkpointing=True requires checkpoint_directory."
            )
        if self.enable_checkpointing and self.checkpoint_frequency <= 0.0:
            raise ValueError(
                "checkpoint_frequency must be positive when checkpointing "
                "is enabled."
            )

        # Output storage
        self.time_series: TimeSeriesData = {}
        self.last_checkpoint_time: float = 0.0
        self._last_state_reset_counter: int = 0
        self._baseline_snapshot_reference: Optional[Dict[str, Any]] = None
        self._baseline_config_reference: Optional[Dict[str, Any]] = None
        self._baseline_snapshot_path: Optional[str] = None

        # Multi-scale coupling integration
        self.enable_auto_scale: bool = self._get_setting('enable_auto_scale', False)
        self.strict_scale_detection: bool = self._get_setting('strict_scale_detection', True)
        self.scale_manager: Optional[Any] = None
        self._scale_check_history: List[Dict[str, Any]] = []

        # Conservation check settings
        self.enable_conservation_checks: bool = self._get_setting('conservation_checks', True)
        self.strict_conservation_enabled: bool = self._get_setting('strict_conservation_enabled', False)

        # Distributed computing configuration
        self.enable_distributed: bool = self._get_setting('enable_distributed', False)
        self.process_group_manager: Optional[Any] = None
        self.device: torch.device = _resolve_runtime_device(self.config)

        # Per-functor device override map: {"functor_name": torch.device}
        # Configured via SimulatorConfig["device_overrides"] dict
        self._device_overrides: Dict[str, torch.device] = {}
        runtime_override_map: Dict[str, torch.device] = {}

        raw_functor_overrides = _collect_config_override_map(
            self.config,
            keys=("device_overrides",),
        )
        # An override the device policy cannot resolve (unknown device, or CUDA
        # requested without CUDA under require_gpu) is a configuration error; it
        # is raised with its cause, never dropped.
        for functor_name, dev_str in raw_functor_overrides.items():
            try:
                resolved = _resolve_single_device(str(dev_str))
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Invalid device override for functor '{functor_name}': {exc}"
                ) from exc
            self._device_overrides[functor_name] = resolved
            # Keep this available to non-functor callers via device_policy.
            runtime_override_map[functor_name] = resolved
            logger.info(
                "Device override: functor '%s' -> %s",
                functor_name, resolved,
            )

        raw_module_overrides = _collect_config_override_map(
            self.config,
            keys=("module_device_overrides",),
        )
        for module_name, dev_str in raw_module_overrides.items():
            try:
                resolved = _resolve_single_device(str(dev_str))
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Invalid device override for module '{module_name}': {exc}"
                ) from exc
            runtime_override_map[module_name] = resolved
            logger.info(
                "Device override: module '%s' -> %s",
                module_name, resolved,
            )

        # Synchronize process-level device policy so non-functor modules can use
        # the same default/override rules without duplicating CUDA checks.
        device_policy.set_default_device(self.device)
        device_policy.set_device_overrides(
            runtime_override_map,
            clear_existing=True,
        )

        # Reuse the canonical CF calendar owner for simulator month scheduling.
        # Import locally to avoid a core/coupling package initialization cycle.
        from ...coupling.scales.temporal.core import PeriodIndexer

        self._calendar_period_indexer = PeriodIndexer(
            device=self.device,
            parameter_manager=get_global_manager(),
        )

        _initialize_distributed_runtime_impl(
            self,
            logger,
            _has_distributed,
            get_process_group_manager,
            get_rank,
            get_world_size,
            is_distributed,
        )

        _initialize_scale_manager_impl(
            self,
            logger,
            _has_scale_manager,
            get_global_manager,
            ParameterNamespace,
            ScalingMode,
            ScaleManager,
        )

        # Initialize cross_domain_validator as None (will be set during initialize())
        self.cross_domain_validator = None
        # Cross-domain findings: "raise" (default) fails the step on an ERROR
        # finding; "report" is the explicit opt-in that only records it. Steps
        # with findings are kept in the report returned by get_cross_domain_reports().
        self.cross_domain_validation_mode: str = str(
            self._get_setting("cross_domain_validation_mode", "raise")
        )
        if self.cross_domain_validation_mode not in ("raise", "report"):
            raise ValueError(
                "cross_domain_validation_mode must be 'raise' or 'report', got "
                f"{self.cross_domain_validation_mode!r}"
            )
        self._cross_domain_reports: List[Dict[str, Any]] = []
        # Framework validation findings of the governed Functor calls (warnings
        # and errors a strict_validation=False Functor downgraded), returned by
        # get_validation_findings().
        self._validation_findings: List[Dict[str, Any]] = []

        # The empty registry is a legitimate committed view (epoch 1); every
        # later registration, retirement, or restore republishes through the
        # registration owner.
        _publish_provider_view_impl(
            self,
            expected_epoch=0,
            operation="initialize",
            provider_uid="simulator",
            new_uids={},
        )

        # Initialize DiagnosticWriter for conservation report export
        self.diagnostic_writer: Optional[Any] = None
        self.enable_diagnostic_export: bool = self._get_setting(
            'enable_diagnostic_export',
            False,
        )
        # Export diagnostics every N steps.
        self.diagnostic_output_frequency: int = self._get_setting(
            'diagnostic_output_frequency',
            10,
        )
        self.diagnostic_output_path: Optional[Path] = None
        self.diagnostic_reference_date: str = self._get_setting(
            'diagnostic_reference_date',
            '1850-01-01',
        )

        _initialize_diagnostic_export_impl(
            self,
            logger,
            HAS_DIAGNOSTIC_WRITER,
            DiagnosticWriter,
        )

        logger.info(
            "RELIC Simulator initialized with dt=%ss, max_time=%ss",
            self.time_step,
            self.max_time,
        )
        logger.info("Multi-rate time integration enabled: fast/medium/slow execution groups")
        if self.enable_auto_scale:
            logger.info("Automatic multi-scale coupling enabled")

    @property
    def run_id(self) -> str:
        """Identity owned by this simulation, including restored runs."""
        return self._run_id

    @run_id.setter
    def run_id(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Simulator run_id must be a nonempty string")
        manager = getattr(self, "_state_manager", None)
        if (
            manager is not None and manager._provenance_transaction_active
            and value != getattr(self, "_run_id", None)
        ):
            raise RuntimeError("Cannot change run identity during an active transaction")
        self._run_id = value
        self._sync_provenance_identity()

    @property
    def state_manager(self) -> Optional[StateManager]:
        return self._state_manager

    @state_manager.setter
    def state_manager(self, value: Optional[StateManager]) -> None:
        previous = getattr(self, "_state_manager", None)
        if previous is not None and previous is not value:
            if previous._provenance_transaction_active:
                raise RuntimeError("Cannot replace StateManager during an active transaction")
            if previous._pending_provenance_events:
                previous.commit_provenance_transaction()
        # The interval means of declared multirate exchanges are StateManager
        # variables of their owner (execution_loop.plan_exchange_means): planned
        # before the binding, so undeclared variable units or a mean missing in
        # the middle of its consumer interval refuse the binding, and registered
        # through the schema owner after it.
        exchange_means = _plan_exchange_means_impl(self, value) if value is not None else []
        self._state_manager = value
        self._sync_provenance_identity()
        _register_exchange_means_impl(value, exchange_means)

    def _sync_provenance_identity(self) -> None:
        """Bind existing observer attachment to the one simulator identity."""
        manager = getattr(self, "_state_manager", None)
        run_id = getattr(self, "_run_id", None)
        if manager is not None and run_id is not None:
            manager.set_provenance_recorder(manager._provenance_recorder, run_id=run_id)

    def get_effective_device(self, functor_name: str) -> torch.device:
        """
        Return the device a functor should use (per-functor override > global).

        Parameters
        ----------
        functor_name : str
            Name of the functor as registered with the simulator.

        Returns
        -------
        torch.device
            The resolved device for this functor.
        """
        return self._device_overrides.get(functor_name, self.device)

    def _get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value with a default value."""
        if key in self.config:
            return self.config[key]
        for section_name in ("simulator", "runtime", "computation"):
            section = self.config.get(section_name)
            if isinstance(section, Mapping) and key in section:
                return section[key]
        return default

    def _resolve_integration_control(self, setting: str, key: str) -> Any:
        """Return one integration or scheduling control: the setting, else the database value.

        The value is checked by ``_validate_integration_control``.
        """
        manager = get_global_manager()
        if manager.get_parameter(key) is None:
            raise KeyError(f"Required parameter '{key}' not found")
        configured = self._get_setting(setting)
        raw = configured if configured is not None else manager.get(key, required=True)
        return self._validate_integration_control(setting, key, raw)

    def _validate_integration_control(self, setting: str, key: str, raw: Any) -> Any:
        """Return the checked value of one integration or scheduling control.

        The ParameterSpec of ``key`` fixes the value type (its dtype) and the
        admissible values (allowed_values or bounds); a value that violates
        them raises ValueError naming the setting. A text value is stripped and
        lower-cased before it is checked. The one check of a control value,
        used when the simulator is built and when a checkpoint restores the
        control (checkpoint owner).
        """
        spec = get_global_manager().get_parameter(key)
        if spec is None:
            raise KeyError(f"Required parameter '{key}' not found")
        dtype = str(spec.dtype).lower()
        if dtype in {"str", "string"}:
            value: Any = str(raw).strip().lower()
        elif dtype.startswith("int"):
            if isinstance(raw, bool) or float(raw) != int(float(raw)):
                raise ValueError(f"Simulator setting '{setting}' must be an integer, got {raw!r}.")
            value = int(float(raw))
        else:
            value = float(raw)
        errors = spec.validate_value(value)
        if errors:
            raise ValueError(f"Simulator setting '{setting}': " + "; ".join(errors))
        return value

    def _resolve_structured_control(self, setting: str, key: str) -> Any:
        """Return one structured scheduling control: the setting, else the database value.

        Used for the dict/list entries of computation.scheduling; their
        structure is checked by the caller, the ParameterSpec rejects a null.
        """
        manager = get_global_manager()
        spec = manager.get_parameter(key)
        if spec is None:
            raise KeyError(f"Required parameter '{key}' not found")
        configured = self._get_setting(setting)
        raw = configured if configured is not None else manager.get(key, required=True)
        errors = spec.validate_value(raw)
        if errors:
            raise ValueError(f"Simulator setting '{setting}': " + "; ".join(errors))
        return raw

    def _resolve_substep_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Return the per-run sub-step overrides (computation.scheduling.substep_overrides).

        The setting, else the database value, checked by ``_validate_substep_overrides``.
        """
        raw = self._resolve_structured_control(
            "substep_overrides", "computation.scheduling.substep_overrides"
        )
        return self._validate_substep_overrides(raw)

    def _validate_substep_overrides(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        """Validate per-run sub-step overrides and return their canonical form.

        A malformed value raises ValueError naming the setting. A valid entry
        replaces FunctorMetadata.substep_count of the named Functor: the
        registration owner records it in the provider binding (binding digest)
        and the group executor sub-steps each call with it
        (execution_group.resolve_call_substep_count); an entry that names no
        registered Functor is listed by get_splitting_order_report. The one
        check of the value, used when the simulator is built and when a
        checkpoint restores it (checkpoint owner).
        """
        # References: Gear and Wells (1984) BIT 24(4):484-502 (different step
        #   sizes for different components); RELIC architecture_overview.md
        #   4.4.1 (sub-stepping through FunctorMetadata.substep_count).
        # Mathematical expression: an entry {count: n} gives n sub-steps per
        #   group call; {max_seconds: s} gives n = ceil(dt_call / s) for a call
        #   over dt_call.
        # Parameter meanings: n the sub-step count of one call; s the largest
        #   admissible sub-step; dt_call the interval of one group call.
        # Units: n dimensionless; s and dt_call in seconds.
        # Domain/constraints: Functor names are non-empty strings; each entry is
        #   a mapping with exactly one key; n is an integer >= 1 (a boolean is no
        #   count); s is a finite real number > 0 (a boolean is no number).
        # Sign convention: none (positive magnitudes).
        setting = "substep_overrides"
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Simulator setting '{setting}' must map Functor names to {{count: n}} or "
                f"{{max_seconds: s}}, got {type(raw).__name__}"
            )
        overrides: Dict[str, Dict[str, Any]] = {}
        for name, entry in raw.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Simulator setting '{setting}': Functor names must be non-empty strings, got {name!r}"
                )
            if not isinstance(entry, Mapping) or len(entry) != 1:
                raise ValueError(
                    f"Simulator setting '{setting}': entry '{name}' must be a mapping with exactly "
                    f"one of the keys count and max_seconds, got {entry!r}"
                )
            ((field_name, value),) = entry.items()
            if field_name == "count":
                if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
                    raise ValueError(
                        f"Simulator setting '{setting}': count of '{name}' must be an integer >= 1, "
                        f"got {value!r}"
                    )
                overrides[name] = {"count": int(value)}
            elif field_name == "max_seconds":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"Simulator setting '{setting}': max_seconds of '{name}' must be a finite "
                        f"number of seconds > 0, got {value!r}"
                    )
                overrides[name] = {"max_seconds": float(value)}
            else:
                raise ValueError(
                    f"Simulator setting '{setting}': entry '{name}' has the unknown key "
                    f"{field_name!r}; use count or max_seconds"
                )
        return overrides

    def _resolve_multirate_exchanges(self) -> Tuple[Dict[str, str], ...]:
        """Return the declared multirate exchanges (computation.scheduling.multirate_exchanges).

        The setting, else the database value, checked by ``_validate_multirate_exchanges``.
        """
        raw = self._resolve_structured_control(
            "multirate_exchanges", "computation.scheduling.multirate_exchanges"
        )
        return self._validate_multirate_exchanges(raw)

    def _validate_multirate_exchanges(self, raw: Any) -> Tuple[Dict[str, str], ...]:
        """Validate declared multirate exchanges and return their canonical form (a tuple).

        A malformed value raises ValueError naming the setting. A hold exchange
        declares the zero-order hold its consumer group reads; an interval_mean
        exchange is executed by the execution loop (one owner commit of its
        interval mean per accepted fast step and an overlay of the consumer
        group call, execution_loop._commit_multirate_exchanges) and resolved at
        every provider-view publication (execution_loop.resolve_multirate_exchanges).
        The one check of the value, used when the simulator is built and when a
        checkpoint restores it (checkpoint owner).
        """
        # References: Gear and Wells (1984) BIT 24(4):484-502 (multirate
        #   methods); RELIC architecture_overview.md 4.4.1 (cross-group sampling
        #   lag and hold lag).
        # Mathematical expression: hold gives the consumer group v(t_c) = the
        #   last committed fast value; interval_mean keeps
        #   m_e <- (m_e tau + vbar h) / (tau + h), tau = t_k - t_last(consumer).
        # Parameter meanings: v the exchanged variable; vbar its accepted
        #   fast-step value; h the accepted fast step; t_k the step start;
        #   t_last the last execution time of the consumer group.
        # Units: tau and h in seconds; m_e and vbar in the units of v.
        # Domain/constraints: a list of mappings with exactly the keys id,
        #   variable, consumer_group and mode; id and variable non-empty strings,
        #   id an identifier (_MULTIRATE_EXCHANGE_ID; it names the mean variable
        #   and the owner); consumer_group a group slower than fast; mode one of
        #   _MULTIRATE_EXCHANGE_MODES (interpolate is refused by name); ids are
        #   unique and at most one exchange exists per (variable, consumer_group).
        # Sign convention: none (v keeps its own sign convention).
        setting = "multirate_exchanges"
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
            raise ValueError(
                f"Simulator setting '{setting}' must be a list of exchange mappings, "
                f"got {type(raw).__name__}"
            )
        consumer_groups = tuple(group for group in self._groups if group != "fast")
        exchanges: List[Dict[str, str]] = []
        seen_ids: Set[str] = set()
        seen_routes: Set[Tuple[str, str]] = set()
        for index, entry in enumerate(raw):
            if not isinstance(entry, Mapping) or set(entry) != set(_MULTIRATE_EXCHANGE_KEYS):
                raise ValueError(
                    f"Simulator setting '{setting}': exchange {index} must be a mapping with exactly "
                    f"the keys {list(_MULTIRATE_EXCHANGE_KEYS)}, got {entry!r}"
                )
            for field_name in ("id", "variable"):
                if not isinstance(entry[field_name], str) or not entry[field_name].strip():
                    raise ValueError(
                        f"Simulator setting '{setting}': {field_name} of exchange {index} must be a "
                        f"non-empty string, got {entry[field_name]!r}"
                    )
            if not _MULTIRATE_EXCHANGE_ID.fullmatch(entry["id"]):
                raise ValueError(
                    f"Simulator setting '{setting}': id of exchange {index} must be an identifier "
                    f"([A-Za-z_][A-Za-z0-9_]*; it names the StateManager mean variable), got {entry['id']!r}"
                )
            if entry["consumer_group"] not in consumer_groups:
                raise ValueError(
                    f"Simulator setting '{setting}': consumer_group of exchange '{entry['id']}' must "
                    f"be one of {list(consumer_groups)}, got {entry['consumer_group']!r}"
                )
            if entry["mode"] == "interpolate":
                raise ValueError(
                    f"Simulator setting '{setting}': exchange '{entry['id']}' requests interpolation, "
                    "which is not implemented between consumer-group calls; use hold or interval_mean"
                )
            if entry["mode"] not in _MULTIRATE_EXCHANGE_MODES:
                raise ValueError(
                    f"Simulator setting '{setting}': mode of exchange '{entry['id']}' must be one of "
                    f"{list(_MULTIRATE_EXCHANGE_MODES)}, got {entry['mode']!r}"
                )
            route = (entry["variable"], entry["consumer_group"])
            if entry["id"] in seen_ids or route in seen_routes:
                raise ValueError(
                    f"Simulator setting '{setting}': exchange '{entry['id']}' repeats an id or the "
                    f"route {route}; ids are unique and one exchange exists per (variable, consumer_group)"
                )
            seen_ids.add(entry["id"])
            seen_routes.add(route)
            exchanges.append({key: str(entry[key]) for key in _MULTIRATE_EXCHANGE_KEYS})
        return tuple(exchanges)

    def _allow_bootstrap_module_registration(self) -> bool:
        """
        Allow bootstrap-module registration only in explicit opt-in contexts.
        """
        env_flag = str(os.getenv("RELIC_ALLOW_BOOTSTRAP_MODULES", "")).strip().lower()
        if env_flag in {"1", "true", "yes", "on"}:
            return True
        if env_flag in {"0", "false", "no", "off"}:
            return False
        if "pytest" in sys.modules:
            return True
        return bool(self._get_setting("allow_bootstrap_modules", False))

    def _allow_stateless_mode(self) -> bool:
        """
        Stateless simulator execution is disabled.
        """
        return False

    def initialize(
        self,
        state_manager: StateManager,
        dependency_graph: DependencyGraph,
    ) -> None:
        """
        Initialize simulator with required components.

        Parameters:
            state_manager : StateManager
                Manages system state variables and updates
            dependency_graph : DependencyGraph
                Handles functor dependencies and execution ordering
        """
        self.state_manager = state_manager
        self.dependency_graph = dependency_graph

        # Initialize cross-domain validator with StateManager integration
        validator_cls = _load_cross_domain_validator() if state_manager is not None else None
        if validator_cls is not None and state_manager is not None:
            self.cross_domain_validator = validator_cls(state_manager)
            logger.info("CrossDomainValidator initialized with StateManager integration")
        else:
            self.cross_domain_validator = None

        self._load_irf_specs()
        if self.irf_specs is not None:
            report = validate_irf_specs(
                self.irf_specs,
                parameter_manager=get_global_manager(),
                state_manager=self.state_manager,
                functors=dict(self.functors.items()),
                strict=self.irf_strict,
            )
            if report.errors and not self.irf_strict:
                for msg in report.errors:
                    logger.warning("IRF validation error: %s", msg)
            for msg in report.warnings:
                logger.warning("IRF validation warning: %s", msg)
            if self.run_specs is not None and self.state_manager is not None:
                if hasattr(self.state_manager, "set_run_specs"):
                    self.state_manager.set_run_specs(self.run_specs)

        # The ScaleManager lifecycle needs the StateManager owner; without one it
        # is deferred to initialize_state(...) or the first regridding.
        if self.state_manager is not None:
            self._ensure_scale_manager_ready()
        else:
            logger.debug("ScaleManager lifecycle deferred until a StateManager is bound")

        # Direct owner injection (including GraphExecutor) must start from the
        # same interval baseline as initialize_state. Rebinding an initialized
        # or restored simulator must preserve its pending integration windows.
        if self.state == SimulationState.UNINITIALIZED:
            _initialize_time_tracking(self)
        self.state = SimulationState.INITIALIZED

        logger.info("Simulator initialized with state manager and dependency graph")
        self._refresh_conservation_ledger()

    def _ensure_scale_manager_ready(self) -> None:
        """Run the ScaleManager lifecycle against the simulator's StateManager.

        The ScaleManager owns no StateManager of its own: it is bound to
        ``self.state_manager`` and then initialized, configured, and validated.
        Without a StateManager this raises before the ScaleManager is touched.
        Lifecycle failures propagate; auto-scale is never switched off silently.
        Idempotent once the manager is READY for the same owner.
        """
        if not self.enable_auto_scale or self.scale_manager is None:
            return
        if self.state_manager is None:
            raise RuntimeError(
                "Automatic scale coupling requires the simulator's StateManager owner "
                "before the ScaleManager lifecycle can run; bind it through "
                "initialize(state_manager, dependency_graph), initialize_state(...), "
                "or set_state_manager(...) first."
            )

        self.scale_manager.set_state_manager(self.state_manager)
        if getattr(self.scale_manager, "is_ready", False):
            return

        # The facade ledger tolerance is not simulator configuration: its one
        # source is coupling.scales.defaults.precision.<mode>.tolerance, which
        # the ScaleManager reads itself when it initializes.
        config_payload = {}
        for key, value in (
            (
                "enforce_conservation",
                getattr(self.scale_manager, "enforce_conservation", None),
            ),
            (
                "cf_strict_mode",
                getattr(self.scale_manager, "cf_strict_mode", None),
            ),
        ):
            if value is not None:
                config_payload[key] = value

        # Start from a clean lifecycle state (recovers an ERROR state), then run
        # initialize -> configure -> validate against the bound owner.
        self.scale_manager.reset()
        self.scale_manager.initialize()
        self.scale_manager.configure(config_payload)
        self.scale_manager.validate()
        # The lifecycle attached the CF synchronizer: configured grid descriptors
        # are checked now, not at the first regridding.
        _validate_auto_scale_grid_descriptors_impl(self)

    def _load_irf_specs(self) -> None:
        """Load IRF specs if configured."""
        if not self.irf_specs_path:
            return
        irf_path = Path(self.irf_specs_path)
        self.irf_specs = load_irf_specs(irf_path)
        if self.irf_strict:
            self.irf_specs.strict = True
            self.irf_specs.run_specs.strict = True
        self.run_specs = self.irf_specs.run_specs

    def _resolve_update_mode(self, var_name: str, mode: UpdateMode) -> UpdateMode:
        """Resolve update mode using RunSpecs when available."""
        if self.run_specs is None:
            return mode
        mode_value = mode.value if isinstance(mode, UpdateMode) else str(mode)
        resolved = self.run_specs.resolve_update_mode(var_name, mode_value)
        return UpdateMode(resolved)

    def initialize_state(self, initial_conditions: StateDict) -> None:
        """
        Initialize simulator with initial state conditions.

        This method initializes simulator state and creates the state
        manager and dependency graph if not already present.

        Parameters:
            initial_conditions : StateDict
                Dictionary of initial state variable values
        """
        _initialize_state_impl(self, initial_conditions, SimulationState)

    # ------------------------------------------------------------------
    # Coordinated registry transactions (E02)
    # ------------------------------------------------------------------
    def _coordination_active(self) -> bool:
        """True when a real multi-rank process group must agree on registry changes."""
        if not self.enable_distributed or self.process_group_manager is None:
            return False
        from relic.computation.distributed.launcher import process_group_world

        initialized, _rank, world_size = process_group_world()
        return initialized and world_size > 1

    def get_coordination_outcomes(self) -> Tuple[Any, ...]:
        """Return the coordinated transaction ledger in commit order."""
        return tuple(self._coordination_outcomes)

    def _provider_identity_vote(self, functor: FunctorBase, execution_group: Optional[str]) -> Dict[str, Any]:
        """Identity every rank must agree on before a provider change commits.

        The contract and binding digests are the registration owner's own
        digests; the device is the resolved runtime device; the stochastic
        substream seed is the RNG identity (a rank drawing from a different
        stream would silently desynchronize the ensemble).
        """
        group = _resolve_execution_group_impl(self, functor, execution_group)
        return {
            "epoch": int(self._provider_lifecycle.epoch),
            "provider": functor.name,
            "contract": _provider_digest_impl(
                _contract_payload_impl(functor), path=f"providers.{functor.name}.contract"
            ),
            "binding": _provider_digest_impl(
                _binding_payload_impl(functor, str(group), self._substep_overrides),
                path=f"providers.{functor.name}.binding",
            ),
            "device": str(self.get_effective_device(functor.name)),
            "rng": getattr(functor, "seed", None),
        }

    def _reduce_updates_across_ranks(self, updates: Dict[str, Any]) -> None:
        """Replace tensor update payloads by their mean over the process group.

        Called by group execution before the StateManager write so every
        rank commits the same ensemble-mean update; a no-op outside an active
        distributed runtime.
        """
        _apply_distributed_allreduce_impl(
            self,
            updates,
            has_distributed_fn=_has_distributed,
            is_distributed_fn=is_distributed,
            get_world_size_fn=get_world_size,
        )

    def _run_coordinated(self, operation: str, **protocol: Any) -> Any:
        from relic.computation.distributed.launcher import coordinate_transaction

        outcome = coordinate_transaction(operation, **protocol)
        self._coordination_outcomes.append(outcome)
        if outcome.status != "committed":
            raise RuntimeError(
                f"Coordinated {operation} {outcome.transaction_id} ended '{outcome.status}' "
                f"on rank {outcome.rank}: {'; '.join(outcome.reasons) or outcome.error}"
            )
        return outcome

    def register_functor(
        self, functor: FunctorBase, execution_group: Optional[str] = None,
        *, _prepare_device: bool = True,
    ) -> None:
        """
        Register a scientific process functor for execution with multi-rate group assignment.

        Under an initialized multi-rank process group the registration is a
        coordinated transaction: every rank votes with the provider identity,
        the pure decision commits only on agreement, and a failed apply on any
        rank is compensated by retiring the provider where it was registered.

        Parameters:
            functor : FunctorBase
                Scientific process functor to register
            execution_group : Optional[str]
                Force assignment to specific execution group ('fast', 'medium', 'slow');
                it overrides every declaration of the functor (see
                ``_determine_execution_group``). An invalid value, or an invalid
                declaration of the functor, raises ValueError before any
                registration state changes.
            _prepare_device : bool
                Private compiler-witness control. False performs the same
                registration checks without moving the candidate or borrowed
                parameter bindings. Actual execution registration uses True.
        """
        if not self._coordination_active():
            _register_functor_impl(self, functor, execution_group, _prepare_device=_prepare_device)
            return
        self._run_coordinated(
            "register_functor",
            prepare=lambda: {
                **self._provider_identity_vote(functor, execution_group),
                "prepare_device": _prepare_device,
            },
            apply=lambda: _register_functor_impl(self, functor, execution_group, _prepare_device=_prepare_device),
            compensate=lambda: _unregister_functor_impl(self, functor.name, strict=False),
        )

    def unregister_functor(
        self,
        functor_name: str,
        *,
        strict: bool = True,
    ) -> bool:
        """
        Unregister a previously registered functor at runtime.

        Parameters
        ----------
        functor_name : str
            Registered functor instance name.
        strict : bool
            If True, missing functor raises KeyError. If False, returns False.

        Returns
        -------
        bool
            True when the functor was removed.
        """
        if not self._coordination_active():
            return _unregister_functor_impl(self, functor_name, strict=strict)
        if functor_name not in self.functors:
            if strict:
                raise KeyError(f"Functor '{functor_name}' is not registered")
            return False
        functor = cast(FunctorBase, self.functors[functor_name])
        group = next((name for name, members in self._groups.items() if functor_name in members), None)

        def _vote() -> Dict[str, Any]:
            vote = self._provider_identity_vote(functor, group)
            # The UID is minted per process and is informational, not identity.
            vote["local_provider_uid"] = self._provider_lifecycle.provider_uids.get(functor_name)
            return vote

        self._run_coordinated(
            "unregister_functor",
            prepare=_vote,
            apply=lambda: _unregister_functor_impl(self, functor_name, strict=True),
            compensate=lambda: _register_functor_impl(self, functor, group),
        )
        return True

    def _find_missing_required_dependencies(self) -> List[Tuple[str, str]]:
        """Return missing required dependencies across registered functors."""
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            List[Tuple[str, str]],
            _find_missing_required_dependencies_impl(self),
        )

    def _refresh_conservation_ledger(self) -> None:
        """Refresh StateManager conservation ledger using functor metadata contributions."""
        _refresh_conservation_ledger_impl(self)

    def _determine_execution_group(
        self,
        functor: FunctorBase,
        execution_group: Optional[str] = None,
    ) -> str:
        """
        Resolve the execution group of a functor by one priority chain.

        Priority order (the first declared rule decides):
        1. the ``execution_group`` argument of ``register_functor``
        2. functor.execution_group attribute
        3. functor.characteristic_time_days -> group by the
           computation.scheduling thresholds
        4. FunctorMetadata.temporal_scale -> ordered scale class
           (TemporalScale.UNKNOWN means no declaration)
        5. process_groups config entry: fills in only when rules 2-4 declare
           nothing; an entry that disagrees with the group of rules 2-4 raises
           ValueError (remove the entry or pass the argument of rule 1)
        6. "fast" when nothing is declared

        Every declared value is validated, also when a higher rule decides:
        an execution group outside fast/medium/slow, a characteristic time
        that is not a finite non-negative number of days, a temporal_scale
        that TemporalScale.from_value does not recognize, and a process_groups
        value that is not a mapping all raise (ValueError; TypeError for a
        temporal_scale that is not a string). There is no silent fallback.

        Parameters:
            functor : FunctorBase
                The functor to analyze
            execution_group : Optional[str]
                The register_functor argument (rule 1); None when not given

        Returns:
            str: Execution group name ("fast", "medium", or "slow")
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            str,
            _determine_execution_group_impl(self, functor, execution_group),
        )

    def _check_scale_consistency(self, state: StateDict, force: bool = False) -> Dict[str, Any]:
        """
        Check for scale consistency issues in the current state.

        This method analyzes the state variables to detect potential
        scale mismatches that would require regridding before functor execution.

        Parameters:
            state : StateDict
                Current system state
            force : bool
                When True, run detection even if auto-scale is disabled.

        Returns:
            Dict[str, Any]: Scale check result with any detected issues
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            Dict[str, Any],
            _check_scale_consistency_impl(self, state, force),
        )

    def _perform_regridding(self, state: StateDict, scale_check: Dict[str, Any]) -> StateDict:
        """
        Perform regridding of state variables to resolve scale mismatches.

        This method uses the ScaleManager and spatial scaling components to perform
        conservative regridding of state variables to a common scale.

        Parameters:
            state : StateDict
                Current system state
            scale_check : Dict[str, Any]
                Scale consistency check results

        Returns:
            StateDict: Regridded state variables
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            StateDict,
            _perform_regridding_impl(self, state, scale_check),
        )

    def _determine_regrid_strategy(
        self,
        var_name: str,
        metadata: Optional[Mapping[str, Any]],
    ) -> str:
        """Regridding method from the variable's declared spatial basis.

        ``metadata`` is the variable's catalog or caller entry; the basis is
        resolved by ``relic.coupling.scales.resolve_spatial_basis`` (raises when
        missing or inconsistent). Returns 'nearest' for categorical fields and
        'conservative' otherwise.
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            str,
            _determine_regrid_strategy_impl(var_name, metadata),
        )

    def _compute_group_execution_orders(self) -> None:
        """
        Compute topological execution order for each group independently.

        Uses DependencyGraph to determine execution order within each group
        while respecting functor dependencies. Groups are processed independently.
        """
        _compute_group_execution_orders_impl(self)

    def _build_dependency_graph(self) -> None:
        """Build the dependency graph for all functors."""
        if self.dependency_graph is None:
            return

        missing_required = self._find_missing_required_dependencies()
        if missing_required:
            missing_summary = ", ".join([f"{src} -> {dep}" for src, dep in missing_required])
            if self.strict_dependencies:
                raise RuntimeError(f"Missing required dependencies: {missing_summary}")
            logger.warning("Missing required dependencies: %s", missing_summary)

        # Re-build the global dependency graph
        for functor_name, functor in self.functors.items():
            # Registration admits only the canonical FunctorBase lifecycle.
            functor = cast(FunctorBase, functor)
            self.dependency_graph.add_functor(functor, functor_name)

            # Add edges for dependencies
            declared = normalize_dependency_declaration(
                functor.get_dependencies(), owner=functor_name
            )
            for dep in declared.required:
                if dep in self.functors:
                    # functor_name depends on dep
                    self.dependency_graph.add_edge(functor_name, dep, required=True)
            for dep in declared.optional:
                if dep in self.functors:
                    self.dependency_graph.add_edge(functor_name, dep, required=False)

        # Update global execution order
        self.execution_order = self.dependency_graph.topological_sort()

    def _get_execution_order(self) -> List[str]:
        """Get the topological execution order of functors."""
        if self.execution_order is None and self.dependency_graph is not None:
            missing_required = self._find_missing_required_dependencies()
            if missing_required and self.strict_dependencies:
                missing_summary = ", ".join([f"{src} -> {dep}" for src, dep in missing_required])
                raise RuntimeError(f"Missing required dependencies: {missing_summary}")
            self.execution_order = self.dependency_graph.topological_sort()
        return self.execution_order if self.execution_order is not None else []

    def _compute_day_index(self, seconds: float) -> int:
        """
        Compute day index from seconds since simulation start.

        Parameters:
            seconds : float
                Time in seconds since simulation start

        Returns:
            int: Day index (0, 1, 2, ...)
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            int,
            _compute_day_index_impl(seconds),
        )

    def _compute_month_index(self, seconds: float | Tensor) -> int:
        """Compute the zero-based month index through ``PeriodIndexer``.

        References:
            CF Conventions 1.8, Section 4.4, Time Coordinate.

        Mathematical Expression:
            m(t) = 12 * (y_C(o + t) - 1) + (q_C(o + t) - 1).

        Parameters:
            seconds : float or Tensor
                Finite elapsed simulation time ``t`` in seconds.

            ``C`` is ``self._calendar_type``; ``o`` is the configured
            ``time_origin`` (0001-01-01 when it is not configured); ``y_C``
            and ``q_C`` are the decoded calendar year and month. Calendar
            arithmetic remains owned by ``PeriodIndexer``.

        Returns:
            int: Monotone zero-based month index ``m(t)``.
        """
        if self._time_origin is None:
            return self._calendar_period_indexer.elapsed_month_index(
                seconds,
                self._calendar_type,
            )

        from ...coupling.external.common.base_adapter import parse_time_origin

        # References: CF Conventions 1.8, Section 4.4 (time coordinate, reference
        #   date with an optional time-zone offset); docs/source/theory/time_scaling.md
        #   Section 4.3.
        # Mathematical expression: o_utc = CF_decode_C(0; seconds since o);
        #   m_C(t; o) = 12 * (y_C(o_utc + t) - 1) + (q_C(o_utc + t) - 1).
        # Parameter meanings: o is the configured time_origin string, o_utc the
        #   same instant without a zone offset as decoded by the existing origin
        #   parser, t the elapsed simulation time, C the simulator calendar,
        #   y_C and q_C the decoded year and month, m_C the absolute month ordinal.
        # Units: t [s]; o and o_utc are calendar timestamps; y_C, q_C and m_C are
        #   dimensionless integers.
        # Domain/constraints: o was validated at construction; the zone offset is
        #   converted, never dropped; PeriodIndexer requires year >= 1 and a finite
        #   nonnegative scalar t and raises otherwise. The configured string is
        #   not modified.
        # Sign convention: positive t advances time and m_C is nondecreasing; the
        #   lookup transfers no quantity and advances no state.
        origin = parse_time_origin(self._time_origin, self._calendar_type)
        return self._calendar_period_indexer.elapsed_month_index(
            seconds,
            self._calendar_type,
            origin.isoformat(),
        )

    def _compute_interval_index(self, seconds: float, interval_seconds: float) -> int:
        """Compute interval index for configurable medium/slow scheduling."""
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            int,
            _compute_interval_index_impl(seconds, interval_seconds),
        )

    def _should_execute_medium(self, future_time: Optional[float] = None) -> bool:
        """
        Check if medium group should execute (day boundary crossed).

        Parameters:
            future_time : Optional[float]
                Time to check for boundary crossing (if None, uses current_time)

        Returns:
            bool: True if day boundary was crossed since last check
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            bool,
            _should_execute_medium_impl(self, future_time),
        )

    def _should_execute_slow(self, future_time: Optional[float] = None) -> bool:
        """
        Check if slow group should execute (month boundary crossed).

        Parameters:
            future_time : Optional[float]
                Time to check for boundary crossing (if None, uses current_time)

        Returns:
            bool: True if month boundary was crossed since last check
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            bool,
            _should_execute_slow_impl(self, future_time),
        )

    def _check_event_triggers(self, group: str, state: Dict[str, Any]) -> bool:
        """Check whether any event trigger forces execution for the given group."""
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            bool,
            _check_event_triggers_impl(self, group, state),
        )

    def register_event_trigger(
        self,
        group: str,
        trigger_fn: Callable[[Dict[str, Any], float], bool],
    ) -> None:
        """Register an event-driven trigger for medium/slow groups."""
        if group not in ("medium", "slow"):
            raise ValueError(
                f"Event triggers only apply to 'medium' or 'slow', got '{group}'"
            )
        self._event_triggers.setdefault(group, []).append(trigger_fn)

    def _execute_group(
        self,
        group_order: List[str],
        dt: float,
        *,
        start_time: float,
        composition: Optional[GroupComposition] = None,
    ) -> FluxDict:
        """Execute a group over ``[start_time, start_time + dt]`` (execution_group owner)."""
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            FluxDict,
            _execute_group_impl(
                self,
                group_order,
                dt,
                start_time=start_time,
                composition=composition,
            ),
        )

    def step(self, dt: Optional[float] = None) -> FluxDict:
        """Execute a single simulation time step."""
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            FluxDict,
            _step_simulation_impl(
            self,
            dt,
            SimulationState,
            _has_distributed,
            is_distributed,
            get_world_size,
            ),
        )

    def run(
            self,
            start_time: float = 0.0,
            end_time: Optional[float] = None,
            dt: Optional[float] = None) -> TimeSeriesData:
        """Run complete simulation from start to end time."""
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            TimeSeriesData,
            _run_simulation_impl(
            self,
            start_time,
            end_time,
            dt,
            SimulationState,
            ),
        )

    def run_ensemble(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        dt: Optional[float] = None,
        *,
        num_samples: int,
        outlier_detection: bool,
        parameter_distributions: Optional[Dict[str, Any]] = None,
        execution_mode: str = 'sequential',
        statistics: Optional[List[str]] = None,
        quantile_levels: Optional[List[float]] = None,
        store_samples: bool = False,
        seed: Optional[int] = None,
    ) -> 'EnsembleResults':
        """
        Run ensemble simulations with parameter uncertainty quantification.

        This method integrates the stochastic ensemble runner to perform Monte Carlo
        simulations with parameter sampling and statistical aggregation. The
        operational execution strategy is sequential because the Simulator,
        StateManager, and ParameterManager are shared mutable owners. Independent
        per-member runtimes required by vectorized or parallel execution are not yet
        implemented.

        Parameters
        ----------
        start_time : float, default=0.0
            Simulation start time [s]
        end_time : Optional[float]
            Simulation end time, uses max_time if None [s]
        dt : Optional[float]
            Time step size, uses default if None [s]
        num_samples : int
            Number of ensemble members to run (required; it sets the Monte
            Carlo precision). The running-mean convergence diagnostic is
            evaluated when num_samples reaches
            optimization.uncertainty.ensemble.convergence_min_members.
        outlier_detection : bool
            Whether members whose z-score exceeds the runner's outlier
            threshold are removed before the statistics (required: removing
            members changes the result, so the caller decides)
        parameter_distributions : Optional[Dict[str, Any]]
            Dictionary of parameter names to distribution specifications
            Format: {'param_name': {'type': 'normal', 'mean': X, 'std': Y}}
            Supported types: 'normal', 'lognormal', 'uniform', 'constant'
        execution_mode : str, default='sequential'
            Execution strategy. Only 'sequential' is operational. 'vectorized',
            'parallel_thread', and 'parallel_process' fail closed until each member
            owns an independent runtime.
        statistics : Optional[List[str]]
            Statistics to compute: 'mean', 'std', 'var', 'quantile', 'min', 'max', 'median', 'iqr'
        quantile_levels : Optional[List[float]]
            Quantile levels to compute (0-1), default=[0.05, 0.25, 0.5, 0.75, 0.95]
        store_samples : bool, default=False
            Whether to store all raw ensemble samples (memory intensive)
        seed : Optional[int]
            Random seed for reproducibility

        Returns
        -------
        EnsembleResults
            Ensemble statistics, metadata, and optionally raw samples

        Examples
        --------
        Monte Carlo simulation with parameter uncertainty:
            >>> distributions = {
            ...     'biosphere.photosynthesis.vcmax25': {
            ...         'type': 'normal', 'mean': 60.0, 'std': 10.0
            ...     },
            ...     'biosphere.respiration.q10': {
            ...         'type': 'lognormal', 'mean': 2.0, 'std': 0.3
            ...     }
            ... }
            >>> results = simulator.run_ensemble(
            ...     end_time=86400.0,
            ...     num_samples=1000,
            ...     outlier_detection=False,
            ...     parameter_distributions=distributions,
            ...     execution_mode='sequential'
            ... )
            >>> logger.info("GPP mean: %.2f", results.statistics['gpp_mean'])
            >>> logger.info("GPP std: %.2f", results.statistics['gpp_std'])

        References
        ----------
        - Robert, C.P. & Casella, G. (2004). Monte Carlo Statistical Methods. Springer
        - Soize, C. (2017). Uncertainty Quantification. Springer
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            "EnsembleResults",
            _run_ensemble_impl(
            self,
            start_time=start_time,
            end_time=end_time,
            dt=dt,
            num_samples=num_samples,
            outlier_detection=outlier_detection,
            parameter_distributions=parameter_distributions,
            execution_mode=execution_mode,
            statistics=statistics,
            quantile_levels=quantile_levels,
            store_samples=store_samples,
            seed=seed,
            ),
        )

    def _sample_parameters(
        self,
        distributions: Dict[str, Any],
        num_samples: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Sample parameters from specified distributions.

        Parameters
        ----------
        distributions : Dict[str, Any]
            Parameter distribution specifications
        num_samples : int
            Number of samples to generate
        seed : Optional[int]
            Random seed

        Returns
        -------
        Dict[str, Tensor]
            Sampled parameter values [num_samples, ...]
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            Dict[str, Tensor],
            _sample_parameters_impl(distributions, num_samples, seed),
        )

    def pause(self) -> None:
        """Pause the running simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            logger.info("Simulation paused")

    def resume(self) -> None:
        """Resume a paused simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            logger.info("Simulation resumed")

    def reset(self) -> None:
        """Reset simulation to initial state."""
        if (
            self.state_manager is not None
            and not self.state_manager._provenance_transaction_active
            and self.state_manager._pending_provenance_events
        ):
            # Confirm accepted notifications before changing any run fields.
            self.state_manager.commit_provenance_transaction()
        self.current_time = 0.0
        self.step_count = 0
        self.total_execution_time = 0.0
        self._performance_diagnostics = None
        self.time_series.clear()
        self.last_checkpoint_time = 0.0
        self._previous_day_index = -1
        self._previous_month_index = -1
        self._previous_medium_index = -1
        self._previous_slow_index = -1
        self._last_medium_execution_time = self.current_time
        self._last_slow_execution_time = self.current_time
        self._adaptive_next_dt = self.time_step
        self._scale_check_history.clear()

        if self.state_manager is not None:
            # Save current state if we don't have initial state saved
            if not hasattr(self, '_initial_state'):
                self._initial_state = self.state_manager.get_state().copy()

            # Reset and restore initial state
            self.state_manager.reset()
            if hasattr(self, '_initial_state'):
                self.state_manager.set_state(self._initial_state, validate=False)

        self.state = SimulationState.INITIALIZED
        # Accepted deliveries retain their original run identity and snapshot.
        # Reset starts new science without cancelling those outstanding effects.
        self.run_id = uuid4().hex
        logger.info("Simulation reset to initial state")

    def enable_distributed_execution(self, enabled: bool = True) -> Any:
        """
        Context manager for temporarily enabling/disabling distributed execution.

        This allows fine-grained control over distributed synchronization,
        useful for selective distributed execution of specific simulation sections.

        Parameters
        ----------
        enabled : bool, default=True
            Whether to enable (True) or disable (False) distributed execution

        Yields
        ------
        None
            Context where distributed execution state is modified

        Examples
        --------
        Temporarily enable distributed execution:
            >>> with simulator.enable_distributed_execution(True):
            ...     simulator.step()

        Temporarily disable distributed execution:
            >>> with simulator.enable_distributed_execution(False):
            ...     simulator.step()

        References
        ----------
        Context manager pattern for PyTorch distributed training
        """
        from contextlib import contextmanager

        @contextmanager
        def _context() -> Any:
            original_state = self.enable_distributed
            self.enable_distributed = enabled
            try:
                yield
            finally:
                self.enable_distributed = original_state

        return _context()

    def get_device_info(self) -> Dict[str, Any]:
        """
        Query current device configuration and distributed computing status.

        Returns device information, GPU memory usage, and distributed rank/world_size.
        Useful for monitoring and debugging distributed execution.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - device: Current torch.device
            - device_type: 'cpu' or 'cuda'
            - device_index: GPU index if CUDA, else None
            - distributed_enabled: Whether distributed mode is active
            - rank: Process rank in distributed mode (None if not distributed)
            - world_size: Total processes in distributed mode (None if not distributed)
            - cuda_available: Whether CUDA is available
            - gpu_memory_allocated: Allocated GPU memory in MB (None if CPU)
            - gpu_memory_cached: Cached GPU memory in MB (None if CPU)

        Examples
        --------
        Check device status:
            >>> info = simulator.get_device_info()
            >>> logger.info("Running on %s, rank %s/%s", info['device_type'], info['rank'], info['world_size'])

        Monitor GPU memory:
            >>> info = simulator.get_device_info()
            >>> if info['gpu_memory_allocated']:
            ...     logger.info("GPU memory: %.1f MB", info['gpu_memory_allocated'])
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            Dict[str, Any],
            _get_device_info_impl(
            self,
            _has_distributed,
            is_distributed,
            get_rank,
            get_world_size,
            logger,
            ),
        )

    def set_state_manager(self, state_manager: StateManager) -> None:
        """Set or replace the state manager."""
        _set_state_manager_impl(self, state_manager)

    def _get_current_state(self) -> StateDict:
        """
        Get current system state from StateManager.

        Returns the full state from StateManager if available,
        otherwise returns a minimal state with time and step.
        This is the single source of truth for current state.
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            StateDict,
            _get_current_state_impl(self),
        )

    def get_state(self) -> StateDict:
        return self._get_current_state()

    def _apply_updates(self, state: StateDict, updates: FluxDict) -> StateDict:
        """
        Apply updates through StateManager's transactional mechanism.

        This method is now a thin wrapper that delegates to StateManager
        for transactional updates with conservation checking and rollback.
        It's used within group execution for intermediate state updates.

        Parameters:
            state : StateDict
                Current state (not used; reserved for interface consistency)
            updates : FluxDict
                Updates to apply

        Returns:
            StateDict: Updated state from StateManager

        Note:
            This method does NOT directly modify state. All updates go through
            StateManager.apply_updates() which handles conservation checking,
            validation, and potential rollback on failures.
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            StateDict,
            _apply_updates_impl(self, state, updates),
        )

    def _store_outputs(self, outputs: FluxDict) -> None:
        """Store outputs for time series analysis."""
        _store_outputs_impl(self, outputs)

    def _resolve_checkpoint_path(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> Path:
        """Resolve one explicit target or the current run's automatic target."""
        resolved_path = checkpoint_path
        if resolved_path is None:
            if not self.enable_checkpointing or self.checkpoint_directory is None:
                raise RuntimeError(
                    "Automatic checkpointing requires enable_checkpointing "
                    "and checkpoint_directory."
                )
            resolved_path = (
                Path(self.checkpoint_directory)
                / self.run_id
                / f"checkpoint_step_{self.step_count:012d}.pt"
            )
        return Path(resolved_path).resolve()

    def _create_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        *, snapshot: Mapping[str, Any] | None = None,
    ) -> CheckpointData:
        """Delegate a live or frozen checkpoint to its canonical owner."""
        if snapshot is not None and checkpoint_path is None:
            raise ValueError("Frozen checkpoint delivery requires an explicit target")
        resolved_path = self._resolve_checkpoint_path(checkpoint_path)
        return cast(
            CheckpointData,
            _create_checkpoint_impl(self, resolved_path, snapshot=snapshot),
        )

    def create_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
        *, snapshot: Mapping[str, Any] | None = None,
    ) -> CheckpointData:
        """
        Public wrapper to create a simulation checkpoint.

        Provides a stable interface for tests and API callers, delegating
        to the internal `_create_checkpoint` implementation.
        """
        return self._create_checkpoint(checkpoint_path, snapshot=snapshot)

    def restore_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        compatibility_certificate: Any = None,
        migration_certificate: Any = None,
    ) -> CheckpointData:
        """
        Restore simulation state from checkpoint.

        Parameters:
            checkpoint_path : str
                Path to checkpoint file to restore
            compatibility_certificate : RestartCompatibilityCertificate, optional
                Explicit certificate for a compatible (non-exact) restart; the
                exact restart is the default and needs no certificate.
            migration_certificate : GraphMigrationCertificate, optional
                Explicit cross-graph migration binding; mutually exclusive
                with the compatibility certificate.
        """
        return cast(
            CheckpointData,
            _restore_checkpoint_impl(
                self,
                checkpoint_path,
                compatibility_certificate=compatibility_certificate,
                migration_certificate=migration_certificate,
            ),
        )

    def _collect_functor_diagnostics(
        self,
        functor: FunctorBase,
        state: StateDict,
        updates: FluxDict,
    ) -> None:
        _collect_functor_diagnostics_impl(self, functor, state, updates)

    def _validate_cross_domain_coupling(self, executed_groups: List[str]) -> None:
        """
        Perform cross-domain validation after step completion.

        This method integrates with CrossDomainValidator to check coupling
        consistency when multiple execution groups have been run. It validates
        that the coupling between different physical domains is consistent.

        Parameters:
            executed_groups : List[str]
                Names of execution groups that were executed in this step
        """
        _validate_cross_domain_coupling_impl(self, executed_groups)

    def get_cross_domain_reports(self) -> List[Dict[str, Any]]:
        """Return the cross-domain validation findings recorded so far.

        One entry per step with findings: ``time``, ``step``, ``mode``,
        ``couplings`` and the ``errors`` / ``warnings`` messages by coupling.
        """
        return [dict(report) for report in self._cross_domain_reports]

    def get_validation_findings(self) -> List[Dict[str, Any]]:
        """Return the framework validation findings of the governed Functor calls.

        One record per call with findings: ``functor``, ``time``, ``dt``,
        ``strict_validation``, ``errors`` and ``warnings`` (each a list of
        ``{"level", "code", "message", "scope"}``) and ``disposition``
        (``"raised"``, ``"downgraded"`` or ``"passed_with_warnings"``). The
        records are copies.
        """
        return [
            {
                **record,
                "errors": [dict(entry) for entry in record["errors"]],
                "warnings": [dict(entry) for entry in record["warnings"]],
            }
            for record in self._validation_findings
        ]

    def get_last_conservation_summary(self) -> Optional['ConservationSummary']:
        """
        Get the most recent conservation summary from StateManager.

        Provides access to the latest conservation report including element-level
        conservation status, custom module diagnostics, and any violations found.

        Returns:
            Optional[ConservationSummary]: Latest conservation report, None if unavailable
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            Optional["ConservationSummary"],
            _get_last_conservation_summary_impl(self),
        )

    def retry_external_deliveries(self) -> List[Dict[str, Any]]:
        """Retry accepted notifications and file deliveries without advancing state."""
        return _retry_external_deliveries_impl(self)

    def _export_diagnostics(
        self, *, snapshot: Optional[Dict[str, Any]] = None,
        delivery_key: Optional[str] = None,
    ) -> Optional["WriteResult"]:
        """
        Export accumulated diagnostics to NetCDF file.

        The diagnostics owner prepares live summaries when no snapshot is
        supplied. Frozen snapshots are delivered without repeating conversion.

        Called periodically based on diagnostic_output_frequency configuration.

        Mathematical Expression:
            Export operation: DS = DiagnosticWriter.convert(sum ConservationSummary_i)
            where DS preserves the declared variable and time contracts.
            Time coordinate: t_cf = (t_sim / S_d) days since reference_date;
            S_d is the physical-constants owner's seconds per day.

        Raises:
            RuntimeError: If diagnostic export fails critically
        """
        return _export_diagnostics_impl(self, snapshot=snapshot, delivery_key=delivery_key)

    def get_coupling_scales_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about multi-scale coupling operations.

        Returns:
            Dict[str, Any]: Statistics about scale checks and regridding operations
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            Dict[str, Any],
            _get_coupling_scales_statistics_impl(self),
        )

    @property
    def provider_epoch(self) -> int:
        """Epoch of the last published committed provider view (0 before any)."""
        return int(self._provider_lifecycle.epoch)

    def get_committed_provider_view(self) -> Optional[CommittedProviderView]:
        """Return the immutable committed provider view, or None if unpublished."""
        return self._provider_lifecycle.committed_view

    def acquire_provider_view(self) -> ProviderViewToken:
        """Admit one reader to the committed view; refused when it is stale."""
        return _acquire_provider_view_impl(self)

    def get_splitting_order_report(self) -> Dict[str, Any]:
        """Return a fresh copy of the effective fast-group splitting order.

        Derived from the committed provider view (fast order and the declared
        ``FunctorMetadata.sub_flow_accuracy`` of its Functors) by
        ``execution_loop.splitting_order_report``, together with the
        admissibility fields of ``execution_loop.splitting_admissibility_report``
        (splitting_policy, fast_order_mode, member_class, member_class_basis,
        refused_or_flagged_edges, contract_registry, time_step_seconds,
        effective_substep_seconds, exchanges (the records of
        execution_loop.resolve_multirate_exchanges), cross_group_composition,
        substep_overrides, unmatched_substep_overrides); nothing is stored.
        """
        view = self.get_committed_provider_view()
        if view is None:
            raise RuntimeError("No committed provider view has been published.")
        report = _splitting_order_report_impl(
            self._splitting_mode,
            view.order_fast,
            {name: view.providers[name].functor for name in view.order_fast},
            epoch=view.epoch,
        )
        report.update(_splitting_admissibility_report_impl(self, view))
        return report

    def get_composition_plan(self) -> "IntegrationPlan":
        """Return the IntegrationPlan of the step composition the committed provider view executes.

        A pure export (nothing is stored, nothing is published): the calls of
        one accepted step in time order, mapped onto
        ``relic.computation.solvers.geometric.IntegrationStage`` with
        runtime_owner simulator (design 20261019_cr_a04_design_01, section 3,
        "Executor"). The fast group comes first, in its published order
        (``CommittedProviderView.order_fast``, reversed for fast_order
        reverse): under lie, or with fewer than two members, one stage per
        member, all executed in the one group call (role lie, commit stage 0,
        dt fraction 1); under strang with n >= 2 members the 2n - 1 stages of
        the palindrome (roles first_half, middle and second_half; commit stage
        k the k-th group call; dt fractions 1/2, 1 and 1/2; the second
        half-steps start at half the step). The medium and then the slow group
        follow, one stage per member in group order, each over the elapsed
        interval of its group call (role group, commit stage 0). Stage-view
        restatements, composition-mean finalizations and multirate exchange
        commits are commits, not stages; an exchange is recorded on the
        consumer stages that read its variable. ``stage_metadata`` holds
        execution_group, role, commit_stage, start_fraction, interval_basis,
        group_trigger (the schedule boundary of the group; event triggers and
        bootstrap marks may also start a medium or slow group), substeps (the
        substep_override of the provider binding payload, else
        FunctorMetadata.substep_count), sub_flow_accuracy and exchanges, and
        for fast stages splitting_mode and fast_order_mode; they predict the
        commit context of every group commit (execution_group.
        build_commit_context). The plan is derived only from state that the
        exact-restart contract binds (topology with groups, execution order
        and simulator configuration; the splitting controls as runtime
        fields; sub-step overrides in the binding digests), so a checkpoint
        does not store it; GraphExecutor compares it with a declared
        graph.metadata.integration_plan.
        """
        # References: Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer,
        #   Lubich and Wanner (2006) Geometric Numerical Integration, 2nd ed.,
        #   Sect. II.5 (compositions of flows); Gear and Wells (1984) BIT
        #   24(4):484-502 (multirate groups).
        # Mathematical expression: lie Phi_h = Phi_{f_1}^h o ... o Phi_{f_n}^h;
        #   strang Phi_h = Phi_{f_1}^{h/2} o ... o Phi_{f_{n-1}}^{h/2} o Phi_{f_n}^h
        #   o Phi_{f_{n-1}}^{h/2} o ... o Phi_{f_1}^{h/2}; stage k with dt fraction
        #   a_k and start fraction b_k covers [t + b_k H, t + b_k H + a_k H] of
        #   its composition [t, t + H].
        # Parameter meanings: f_i the fast members in published order; h the
        #   fast step; H the composition interval (h for the fast group, the
        #   elapsed interval t_future - t_last of a medium or slow group call);
        #   a_k, b_k the dt and start fractions of stage k.
        # Units: a_k, b_k dimensionless; t, h, H and group interval seconds [s].
        # Domain/constraints: the committed provider view exists (epoch >= 1);
        #   the splitting controls were validated when the simulator was built
        #   or restored; n < 2 runs the plain group call under either mode.
        # Sign convention: sub-flows are applied left to right in time order.
        from ...computation.solvers.geometric import IntegrationPlan, IntegrationStage
        from ..functor.functor_types import SubFlowAccuracy

        view = self.get_committed_provider_view()
        if view is None:
            raise RuntimeError("No committed provider view has been published.")
        # The declared multirate exchanges resolved against the committed groups
        # (execution_loop.resolve_multirate_exchanges): an overlay of the
        # consumer stages that read the exchanged variable.
        overlays: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for record in self.get_splitting_order_report()["exchanges"]:
            for reader in record["readers"]:
                overlays.setdefault((record["consumer_group"], reader), []).append({
                    "id": record["id"],
                    "variable": record["variable"],
                    "mode": record["mode"],
                    "mean_variable": record["mean_variable"],
                })
        mode = self._splitting_mode
        fast = tuple(view.order_fast)
        palindrome = mode == "strang" and len(fast) >= 2
        # (functor, group, role, commit stage, dt fraction, start fraction,
        # coupling mode) of every call, in time order.
        schedule: List[Tuple[str, str, str, int, float, float, str]] = []
        if palindrome:
            n_fast = len(fast)
            schedule.extend(
                (name, "fast", "first_half", k, 0.5, 0.0, "strang_half")
                for k, name in enumerate(fast[:-1])
            )
            schedule.append((fast[-1], "fast", "middle", n_fast - 1, 1.0, 0.0, "strang_full"))
            schedule.extend(
                (name, "fast", "second_half", n_fast + j, 0.5, 0.5, "strang_half")
                for j, name in enumerate(reversed(fast[:-1]))
            )
        else:
            schedule.extend((name, "fast", "lie", 0, 1.0, 0.0, "lie") for name in fast)
        for group in ("medium", "slow"):
            schedule.extend(
                (name, group, "group", 0, 1.0, 0.0, "lie") for name in view.group_order(group)
            )
        # Schedule boundaries of the groups (time_ops.should_execute_medium/slow);
        # the intervals are set by init_ops.configure_calendar_settings.
        medium_interval = cast(float, self._medium_interval_seconds)
        slow_interval = cast(Optional[float], self._slow_interval_seconds)
        triggers: Dict[str, Dict[str, Any]] = {
            "fast": {"kind": "every_step"},
            "medium": {"kind": "interval", "seconds": float(medium_interval)},
            "slow": (
                {"kind": "interval", "seconds": float(slow_interval)}
                if slow_interval is not None and float(slow_interval) > 0.0
                else {"kind": "calendar_month", "calendar": str(self._calendar_type)}
            ),
        }
        stages: List[IntegrationStage] = []
        group_index: Dict[str, int] = {}
        previous: Optional[str] = None
        for name, group, role, commit_stage, dt_fraction, start_fraction, coupling in schedule:
            provider = view.providers[name]
            functor = provider.functor
            metadata = functor.metadata
            index = group_index.get(group, 0)
            group_index[group] = index + 1
            stage_id = f"{group}.{index}.{name}"
            override = _binding_payload_impl(
                functor, provider.execution_group, self._substep_overrides
            ).get("substep_override")
            substeps: Dict[str, Any] = (
                {
                    "source": "functor_metadata",
                    "count": _resolve_call_substep_count_impl(
                        name, functor, None, float(self.time_step)
                    ),
                }
                if override is None
                else {"source": "substep_override", **dict(override)}
            )
            stage_metadata: Dict[str, Any] = {
                "execution_group": group,
                "role": role,
                "commit_stage": commit_stage,
                "start_fraction": start_fraction,
                "interval_basis": "fast_step" if group == "fast" else "group_elapsed_interval",
                "group_trigger": dict(triggers[group]),
                "substeps": substeps,
                "sub_flow_accuracy": SubFlowAccuracy(metadata.sub_flow_accuracy).value,
                "exchanges": [dict(item) for item in overlays.get((group, name), ())],
            }
            if group == "fast":
                stage_metadata["splitting_mode"] = mode
                stage_metadata["fast_order_mode"] = self._fast_order_mode
            stages.append(
                IntegrationStage(
                    stage_id=stage_id,
                    operator_kind=name,
                    solver_name="governed_functor_call",
                    structure_kind=metadata.category.value,
                    dt_fraction=dt_fraction,
                    coupling_mode=cast(Any, coupling),
                    runtime_owner="simulator",
                    reads=tuple(str(item) for item in metadata.input_variables),
                    writes=tuple(str(item) for item in metadata.output_variables),
                    depends_on=() if previous is None else (previous,),
                    stage_metadata=stage_metadata,
                )
            )
            previous = stage_id
        slower = any(entry[1] != "fast" for entry in schedule)
        if not stages:
            scheme = "custom"
        elif len(stages) == 1:
            scheme = "single"
        elif palindrome:
            # A palindrome followed by medium or slow calls is a multirate
            # composition (strang within fast, lie across groups).
            scheme = "custom" if slower else "strang"
        else:
            scheme = "lie"
        references = ["Hairer, Lubich and Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.5"]
        if palindrome:
            references.append("Strang (1968) SIAM J. Numer. Anal. 5(3):506-517")
        if slower:
            references.append("Gear and Wells (1984) BIT 24(4):484-502")
        return IntegrationPlan(
            plan_id="relic_simulator_composition",
            stages=tuple(stages),
            scheme=cast(Any, scheme),
            references=tuple(references),
        )

    def get_lifecycle_receipts(self) -> Tuple[LifecycleReceipt, ...]:
        """Return the registration and retirement receipts in commit order."""
        return tuple(self._provider_lifecycle.receipts)

    def declare_topology_transition(self, transition: TopologyTransition) -> None:
        """Declare one hybrid topology transition the lifecycle owner may commit."""
        _declare_topology_transition_impl(self, transition)

    def get_topology_transitions(self) -> Dict[str, TopologyTransition]:
        """Return the declared topology transitions by name."""
        return dict(self._provider_lifecycle.topology_transitions)

    def get_topology_history(self) -> Tuple[Dict[str, Any], ...]:
        """Return the committed topology transitions in commit order."""
        return tuple(dict(item) for item in self._provider_lifecycle.topology_history)

    def apply_topology_transition(
        self,
        name: str,
        *,
        accepted_step: Optional[int] = None,
        accepted_time: Optional[float] = None,
    ) -> LifecycleReceipt:
        """Commit a declared transition at a quiescent boundary (never inside a step)."""
        return _apply_topology_transition_impl(
            self,
            name,
            accepted_step=self.step_count if accepted_step is None else accepted_step,
            accepted_time=self.current_time if accepted_time is None else accepted_time,
        )

    def bind_forcing_cursor(
        self,
        *,
        sequence_sha256: str,
        length: int,
        consumed: int = 0,
    ) -> Dict[str, Any]:
        """Bind the forcing sequence this simulator consumes one snapshot per step.

        The cursor records the canonical digest of the whole sequence, its
        length, and the number of snapshots already consumed by accepted
        steps. Re-binding a different sequence while one is partially consumed
        is refused; binding the same sequence keeps the consumed position.
        """
        if not isinstance(sequence_sha256, str) or len(sequence_sha256) != 64:
            raise ValueError("sequence_sha256 must be a 64-character hex digest")
        if type(length) is not int or length <= 0:
            raise ValueError("Forcing sequence length must be a positive integer")
        if type(consumed) is not int or consumed < 0 or consumed > length:
            raise ValueError("consumed must lie in [0, length]")
        current = self._forcing_cursor
        if current is not None and current["consumed"] not in (0, current["length"]):
            if current["sequence_sha256"] != sequence_sha256 or current["length"] != length:
                raise RuntimeError(
                    "A different forcing sequence is partially consumed "
                    f"({current['consumed']}/{current['length']}); finish or "
                    "clear it before binding another sequence"
                )
            consumed = current["consumed"]
        self._forcing_cursor = {
            "sequence_sha256": sequence_sha256,
            "length": length,
            "consumed": consumed,
        }
        return dict(self._forcing_cursor)

    def get_forcing_cursor(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the bound forcing cursor, or None."""
        return None if self._forcing_cursor is None else dict(self._forcing_cursor)

    def clear_forcing_cursor(self) -> None:
        """Unbind the forcing sequence once it is fully consumed or abandoned."""
        self._forcing_cursor = None

    def get_functor_summary(self) -> Dict[str, Any]:
        """
        Return metadata summary for all registered functors.

        Returns a dict keyed by functor name, each containing its
        metadata category, execution group, input/output variable counts,
        spatial scale, and variable kind.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Per-functor metadata summary.
        """
        summary: Dict[str, Any] = {}
        for fname, functor in self.functors.items():
            meta = cast(FunctorBase, functor).metadata
            group = "unknown"
            for grp_name, grp_members in self._groups.items():
                if fname in grp_members:
                    group = grp_name
                    break
            summary[fname] = {
                "category": meta.category.value,
                "execution_group": group,
                "input_variables": list(meta.input_variables),
                "output_variables": list(meta.output_variables),
                "spatial_scale": meta.spatial_scale.value
                    if hasattr(meta, 'spatial_scale') else "unknown",
                "variable_kind": meta.variable_kind.value
                    if hasattr(meta, 'variable_kind') else "unknown",
                "version": meta.version,
                "description": meta.description,
            }
        return summary

    def get_variable_provenance(
        self,
        var_name: Optional[str] = None,
        *,
        include_retention: bool = False,
        include_commit_context: bool = False,
    ) -> Any:
        """
        Return the retained provenance window for one variable or all variables.

        Reports the writer records kept in ``_step_provenance_detail``: each
        record names the Functor, the update mode, and the step index of the
        write. Records accumulate across accepted steps until the configured retention
        limit, max_provenance_records_per_variable (default 256). This is a
        rolling audit window, not a complete execution history. The detail map is
        not cleared at step start; a rejected step is rolled back, so the
        last record of a variable is its most recent writer. The per-step
        increment decomposition comes from ``_step_functor_increments``, which
        is reset at every step.

        Parameters
        ----------
        var_name : Optional[str]
            Name of a state variable to query. When ``None``, return all
            tracked variables as a mapping.
        include_retention : bool
            Return a mapping with records and evicted_records. The latter is
            a count for one variable or a mapping of counts for all variables.
            Counts survive checkpoint/restart and roll back with rejected steps.
            Default False preserves the list/mapping record query.
        include_commit_context : bool
            Add to every writer record the commit context of its group
            commit (``commit_context``: step, composition_start,
            composition_interval, stage, role, interval_start and dt, see
            execution_group.build_commit_context; None for a record written
            without one) and its StateManager commit index
            (``commit_index``), and keep apart records that differ only in
            them, so every executed call of a Functor is one record (the two
            Strang half-steps of a member and the calls of successive
            steps). Default False returns the records above unchanged.

        Returns
        -------
        Any
            - ``List[Dict[str, Any]]`` when ``var_name`` is provided.
            - ``Dict[str, List[Dict[str, Any]]]`` when ``var_name`` is ``None``.
        """
        def _records_for(target_name: str) -> List[Dict[str, Any]]:
            records: List[Dict[str, Any]] = []
            detail = getattr(self, "_step_provenance_detail", None)
            if isinstance(detail, dict) and target_name in detail:
                for entry in detail[target_name]:
                    role = entry[3] if len(entry) > 3 else "writer"
                    provider = entry[4] if len(entry) > 4 else None
                    record: Dict[str, Any] = {
                        "functor": entry[0],
                        "mode": str(entry[1]),
                        "step": entry[2] if len(entry) > 2 else None,
                        "role": role,
                        "provider_uid": (
                            None if provider is None else provider.get("provider_uid")
                        ),
                        "provider_epoch": (
                            None if provider is None else provider.get("provider_epoch")
                        ),
                    }
                    if include_commit_context:
                        # The commit context the group owner attached to the
                        # provider dict of this call (plain values; copied).
                        context = None if provider is None else provider.get("commit_context")
                        record["commit_context"] = None if context is None else dict(context)
                        record["commit_index"] = (
                            None if provider is None else provider.get("commit_index")
                        )
                    records.append(record)

            increment_map_all = getattr(self, "_step_functor_increments", None)
            if isinstance(increment_map_all, dict):
                increment_map = increment_map_all.get(target_name)
                if isinstance(increment_map, dict):
                    for increment_writer in sorted(increment_map.keys()):
                        if not isinstance(increment_writer, str):
                            continue
                        records.append(
                            {
                                "functor": increment_writer,
                                "mode": str(UpdateMode.ACCUMULATE),
                                "step": None,
                                "role": "increment_source",
                            }
                        )

            sources = getattr(self, "_step_update_sources", None)
            has_final_writer = False
            if isinstance(sources, dict) and target_name in sources:
                entry = sources[target_name]
                records.append({
                    "functor": entry[0],
                    "mode": str(entry[1]),
                    "step": None,
                    "role": "final_writer",
                })
                has_final_writer = True

            if not has_final_writer and isinstance(detail, dict) and target_name in detail:
                if detail[target_name]:
                    last_entry = detail[target_name][-1]
                    records.append(
                        {
                            "functor": last_entry[0],
                            "mode": str(last_entry[1]),
                            "step": last_entry[2] if len(last_entry) > 2 else None,
                            "role": "final_writer",
                        }
                    )

            deduped: List[Dict[str, Any]] = []
            seen: Set[Tuple[Any, ...]] = set()
            for record in records:
                key: Tuple[Any, ...] = (
                    record.get("functor"),
                    record.get("mode"),
                    record.get("step"),
                    record.get("role"),
                )
                if include_commit_context:
                    commit_context = record.get("commit_context")
                    key = key + (
                        None if commit_context is None else tuple(sorted(commit_context.items())),
                        record.get("commit_index"),
                    )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(record)
            return deduped

        if isinstance(var_name, str) and var_name:
            records = _records_for(var_name)
            if include_retention:
                return {
                    "records": records,
                    "evicted_records": getattr(self, "_provenance_evicted_counts", {}).get(var_name, 0),
                }
            return records

        detail = getattr(self, "_step_provenance_detail", None)
        sources = getattr(self, "_step_update_sources", None)
        tracked_names: Set[str] = set()
        if isinstance(detail, dict):
            tracked_names.update(name for name in detail.keys() if isinstance(name, str))
        if isinstance(sources, dict):
            tracked_names.update(name for name in sources.keys() if isinstance(name, str))

        records = {name: _records_for(name) for name in sorted(tracked_names)}
        if include_retention:
            counts = getattr(self, "_provenance_evicted_counts", {})
            return {
                "records": records,
                "evicted_records": {name: counts.get(name, 0) for name in records},
            }
        return records

    def set_baseline_reference(self, baseline_snapshot_path: str) -> None:
        """Register baseline snapshot file for experiment differential tracking."""
        path_obj = Path(baseline_snapshot_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Baseline snapshot not found: {baseline_snapshot_path}")

        with path_obj.open("r", encoding="utf-8-sig") as handle:
            loaded = json.load(handle)

        if isinstance(loaded, dict) and "snapshot" in loaded:
            snapshot_payload = loaded.get("snapshot", {})
            config_payload = loaded.get("config", {})
        else:
            snapshot_payload = loaded
            config_payload = {}

        if not isinstance(snapshot_payload, dict) or "parameters" not in snapshot_payload:
            raise ValueError(
                "Baseline snapshot must be a dict containing 'parameters' or "
                "a wrapper dict with key 'snapshot'."
            )

        self._baseline_snapshot_reference = snapshot_payload
        self._baseline_config_reference = config_payload if isinstance(config_payload, dict) else {}
        self._baseline_snapshot_path = str(path_obj)

    def get_experiment_diff(self) -> Optional[Dict[str, Any]]:
        """Get parameter/config diff relative to registered baseline snapshot."""
        if self._baseline_snapshot_reference is None:
            return None

        current_snapshot = get_global_manager().snapshot()
        baseline_snapshot = self._baseline_snapshot_reference
        baseline_config = self._baseline_config_reference or {}
        current_config = dict(self.config)

        parameter_diff = _compute_parameter_diff_impl(current_snapshot, baseline_snapshot)
        report = _generate_experiment_diff_report_impl(
            current_snapshot=current_snapshot,
            baseline_snapshot=baseline_snapshot,
            current_config=current_config,
            baseline_config=baseline_config,
        )
        return {
            "baseline_snapshot_path": self._baseline_snapshot_path,
            "parameter_diff": parameter_diff,
            "report": report,
            "current_timestamp": current_snapshot.get("timestamp"),
            "baseline_timestamp": baseline_snapshot.get("timestamp"),
        }

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive simulation status and statistics.

        Enhanced to include conservation status, recent diagnostics,
        and multi-scale coupling statistics from StateManager when available.

        Returns:
            Dict[str, Any]: Simulation summary information
        """
        return cast(  # type: ignore[redundant-cast,unused-ignore]
            Dict[str, Any],
            _get_simulation_summary_impl(self),
        )


# Factory function for creating simulator instances
def create_simulator(config: SimulatorConfig) -> RELICSimulator:
    """
    Factory function to create configured simulator instance.

    Parameters:
        config : SimulatorConfig
            Simulator configuration dictionary

    Returns:
        RELICSimulator: Configured simulator instance
    """
    return RELICSimulator(config)


# Export main classes
__all__ = [
    'RELICSimulator',
    'SimulationState',
    'create_simulator',
    'SimulatorConfig',
    'TimeSeriesData',
    'CheckpointData',
    'DependencyGraph',
]
