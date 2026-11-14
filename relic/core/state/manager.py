"""
RELIC State Manager

Centralized state variable management for the RELICSimulator. Provides a
single source of truth for state variables, schema-based validation,
transactional updates with optional mass-conservation checking, and
diagnostic aggregation.

Notes:
- ASCII-only docstrings and comments to avoid encoding issues.
- Minimal but complete functionality to satisfy simulator and tests.

Mathematical Expressions:
- State space: S = {s_i}, i = 1..N, s_i in R
- Transactional update: s_i(t+dt) = s_i(t) + u_i, apply atomically
- Mass balance (diagnostic): sum(u_j) approx 0 for a conserved element set

References:
- LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems.
- Press, W. H. et al. (2007). Numerical Recipes (update and conservation).
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, Set, cast, overload
import logging
import math
from numbers import Real
import os
import importlib
import inspect
import time

import torch
from relic.core import device_policy
import torch.nn.functional as F
from torch import Tensor

# Feature flags (tests patch relic.core.state.feature_manager)
from ..feature_flags import feature_manager as _feature_manager
from . import types as _state_types
_state_conservation = importlib.import_module("relic.core.state.conservation")

StateVariableType = _state_types.StateVariableType
UpdateMode = _state_types.UpdateMode
StateWriteMode = _state_types.StateWriteMode
UpdateItem = _state_types.UpdateItem
StateVariableSpec = _state_types.StateVariableSpec
ConservationSummary = _state_types.ConservationSummary

# Re-export for tests to patch
feature_manager = _feature_manager

logger = logging.getLogger(__name__)

_GLOBAL_MANAGER_RESOLVER = None


def _require_parameter_manager() -> Any:
    """Return the process parameter manager; resolution errors propagate."""
    global _GLOBAL_MANAGER_RESOLVER
    if _GLOBAL_MANAGER_RESOLVER is None:
        from ...parameters import get_global_manager as resolved
        _GLOBAL_MANAGER_RESOLVER = resolved
    return _GLOBAL_MANAGER_RESOLVER()


def _get_parameter_manager_optional() -> Optional[Any]:
    try:
        return _require_parameter_manager()
    except RuntimeError:
        return None


# The keys of the ``conservation_checks`` setting that StateManager reads; any
# other key is a configuration error.
CONSERVATION_CHECK_KEYS: Tuple[str, ...] = (
    "enabled",
    "raise_on_violation",
    "rollback_on_failure",
    "tolerance",
)
# The parameter whose value is the default commit-time ledger tolerance and whose
# declared bounds limit an explicit ``tolerance`` override.
CONSERVATION_TOLERANCE_PARAMETER = "validation.conservation.conservation_tolerance"
# The settings StateManager.configure accepts after construction:
#   conservation_tolerance: the commit-time ledger tolerance, validated like
#       conservation_checks["tolerance"] (the property of the same name).
#   max_history_length: optional non-negative integer bounding in-memory
#       history and conservation_history ledgers to O(1) peak storage.
# Any other key raises ValueError.
STATE_MANAGER_CONFIGURABLE_KEYS: Tuple[str, ...] = (
    "conservation_tolerance",
    "max_history_length",
)


def validate_max_history_length(value: Any, *, source: str = "max_history_length") -> Optional[int]:
    """Validate optional bounded history window capacity.

    1. References:
       - Gray, J., & Reuter, A. (1992). Transaction Processing: Concepts and Techniques.
       - Griewank, A., & Walther, A. (2008). Evaluating Derivatives: Principles and
         Techniques of Algorithmic Differentiation (bounded checkpointing / sliding window).
    2. Specific mathematical expressions for the calculation:
       - For ledger L_t at step t and bound M in Z_{>= 0}:
         |L_t| <= M = O(1), L_t = [e_{max(1, t-M+1)}, ..., e_t]
    3. Meanings of the parameters:
       - value (Optional[int]): Maximum retained entries M in _history and
         _conservation_history (entries, dimensionless count; None means unbounded).
       - source (str): Configuration field identifier for error reporting.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"'{source}' must be None or a non-negative int; got {value!r} "
            f"({type(value).__name__})."
        )
    return int(value)



def validate_conservation_setting(key: str, value: Any, *, source: str) -> Any:
    """Validate one ``conservation_checks`` entry; ``source`` names it in errors.

    The switches ``enabled``, ``raise_on_violation`` and ``rollback_on_failure``
    accept only a bool (a string such as "false" or an integer 0/1 raises).

    References: RELIC parameter validation.conservation.conservation_tolerance
        (default commit-time ledger tolerance) and its declared bounds.
    Mathematical expression: a ``tolerance`` tol is accepted iff tol is a finite
        real with b_l <= tol <= b_u.
    Parameter meanings: tol is the override; b_l and b_u are the lower and upper
        bounds declared for validation.conservation.conservation_tolerance.
    Units: tol, b_l and b_u are dimensionless relative tolerances.
    Domain/constraints: a bool, a string, NaN, an infinite value, or a value
        outside [b_l, b_u] raises ValueError; nothing is ignored or coerced. The
        bounds are read from the parameter database, which must be available.
    Sign convention: none; tolerances are positive magnitudes.
    """
    if key not in CONSERVATION_CHECK_KEYS:
        raise ValueError(
            f"'{source}' is not a conservation_checks key; StateManager reads only "
            f"{list(CONSERVATION_CHECK_KEYS)}."
        )
    if key != "tolerance":
        if not isinstance(value, bool):
            raise ValueError(
                f"'{source}' must be a bool (True or False); got {value!r} "
                f"({type(value).__name__})."
            )
        return value
    parameter_manager = _require_parameter_manager()
    lower, upper = (
        float(bound)
        for bound in parameter_manager.get_parameter(CONSERVATION_TOLERANCE_PARAMETER).bounds
    )
    tolerance = None
    if isinstance(value, Real) and not isinstance(value, bool):
        tolerance = float(value)
    if tolerance is None or not math.isfinite(tolerance) or not lower <= tolerance <= upper:
        raise ValueError(
            f"'{source}' must be a finite real number in [{lower!r}, {upper!r}] (bounds "
            f"declared for {CONSERVATION_TOLERANCE_PARAMETER}); got {value!r} "
            f"({type(value).__name__})."
        )
    return tolerance


def validate_conservation_checks(
    settings: Any,
    *,
    source: str = "conservation_checks",
) -> Dict[str, Any]:
    """Validate a ``conservation_checks`` setting and return its mapping form.

    ``None`` means no setting (the defaults apply), a bool is the ``enabled``
    switch, and a mapping may carry only ``CONSERVATION_CHECK_KEYS``, each value
    validated by ``validate_conservation_setting``. Anything else raises
    ValueError naming ``source`` and the value.
    """
    if settings is None:
        return {}
    if isinstance(settings, bool):
        return {"enabled": settings}
    if not isinstance(settings, Mapping):
        raise ValueError(
            f"'{source}' must be a bool or a mapping with keys {list(CONSERVATION_CHECK_KEYS)}; "
            f"got {settings!r} ({type(settings).__name__})."
        )
    unknown = sorted(str(key) for key in settings if key not in CONSERVATION_CHECK_KEYS)
    if unknown:
        raise ValueError(
            f"'{source}' has unknown keys {unknown}; StateManager reads only "
            f"{list(CONSERVATION_CHECK_KEYS)}."
        )
    return {
        key: validate_conservation_setting(key, value, source=f"{source}.{key}")
        for key, value in settings.items()
    }


# Keys StateManager adds to the copy of a commit context in the provenance events
# of the commit; a caller may not supply them:
#   conservation_checked: whether the commit ran its commit-time ledger check;
#   element_net_change: the signed net change of the element ledger of the commit;
#   element_net_change_error: why that ledger could not be evaluated.
COMMIT_CONTEXT_RESERVED_KEYS: Tuple[str, ...] = (
    "conservation_checked",
    "element_net_change",
    "element_net_change_error",
)


def validate_commit_context(commit_context: Any) -> Optional[Dict[str, Any]]:
    """Validate the optional provenance context of one commit and return a copy.

    References: W3C PROV-DM (2013), generation records with plain attribute
        values; design 20261019_cr_a04_design_01, D7 and section 2
        (commit_context of StateManager.apply_updates).
    Mathematical expression: none; the mapping is copied key by key.
    Parameter meanings: commit_context maps non-empty string keys to None,
        bool, int, float or str values, for example the step, composition
        start and interval, stage, role, interval start and interval of a
        simulator group commit.
    Units: those of the caller's values (seconds for the simulator's times).
    Domain/constraints: None means no context. A value that is not a mapping,
        a non-string key or a value of another type raises TypeError; an empty
        key, a non-finite float or a key of COMMIT_CONTEXT_RESERVED_KEYS raises
        ValueError; every message names commit_context; nothing is coerced
        (exact types, the checkpoint grammar of plain values).
    Sign convention: not applicable.
    """
    if commit_context is None:
        return None
    if not isinstance(commit_context, Mapping):
        raise TypeError(
            "commit_context must be None or a mapping of plain values; got "
            f"{type(commit_context).__name__}."
        )
    copied: Dict[str, Any] = {}
    for key, value in commit_context.items():
        if not isinstance(key, str):
            raise TypeError(f"commit_context keys must be strings; got {type(key).__name__}.")
        if not key:
            raise ValueError("commit_context keys must be non-empty strings.")
        if key in COMMIT_CONTEXT_RESERVED_KEYS:
            raise ValueError(
                f"commit_context key '{key}' is reserved; StateManager records it for the commit."
            )
        if value is not None and type(value) not in (bool, int, float, str):
            raise TypeError(
                f"commit_context['{key}'] must be None, bool, int, float or str; got "
                f"{type(value).__name__}."
            )
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"commit_context['{key}'] must be a finite float; got {value!r}.")
        copied[key] = value
    return copied


class StateManager:
    """Centralized state manager with transactional update support.

    This class maintains the system state, applies updates atomically,
    performs optional mass-conservation checks, and aggregates diagnostics.
    """
    def __init__(
        self,
        schema: Optional[Dict[str, StateVariableSpec]] = None,
        **kwargs: Any,
    ) -> None:
        # Extract optional config/schema
        config_obj = kwargs.pop('config', None)
        if schema is None and config_obj is not None:
            schema = config_obj
        # Extract runtime device with CUDA-first default and CPU fallback.
        requested_device = kwargs.pop(
            "device",
            device_policy.resolve_device(default="auto"),
        )
        try:
            resolved_device = torch.device(requested_device)
        except (TypeError, ValueError):
            resolved_device = device_policy.resolve_device(default="auto")
        if resolved_device.type == "cuda" and not device_policy.is_cuda_available():
            resolved_device = torch.device("cpu")
        self.device = resolved_device
        self._conservation_checks_config = validate_conservation_checks(
            kwargs.pop('conservation_checks', None)
        )
        self._physical_constraint_manager = kwargs.pop("physical_constraint_manager", None)
        # Diagnostics toggles (environment override):
        env_diag = os.getenv("RELIC_ENABLE_STATE_DIAGNOSTICS", "1").strip().lower()
        env_cons_diag = os.getenv("RELIC_ENABLE_CONSERVATION_DIAGNOSTICS", "1").strip().lower()
        self._diagnostics_enabled: bool = env_diag not in {"0", "false", "off"}
        self._conservation_diagnostics_enabled: bool = env_cons_diag not in {"0", "false", "off"}
        if schema is not None and not isinstance(schema, dict):
            if hasattr(schema, "to_schema"):
                schema = schema.to_schema()
            elif hasattr(schema, "schema"):
                schema = schema.schema
        self._contract_registry = None
        autoload_override = kwargs.pop("autoload_contract_registry", None)
        env_autoload_raw = os.getenv("RELIC_AUTOLOAD_CONTRACT_REGISTRY", "1")
        env_autoload = env_autoload_raw.strip().lower()
        if autoload_override is None:
            autoload_contract_registry = env_autoload not in {"0", "false", "no", "off"}
            autoload_switch = f"RELIC_AUTOLOAD_CONTRACT_REGISTRY={env_autoload_raw!r}"
        else:
            autoload_contract_registry = bool(autoload_override)
            autoload_switch = f"autoload_contract_registry={bool(autoload_override)}"
        if autoload_contract_registry:
            # The SCM registry governs ownership, aliases, writer permissions and
            # increments in apply_updates. The single contract loader reads the
            # packaged contracts strictly (once per process) and raises on a
            # missing directory or a malformed contract, naming it, instead of
            # switching governance off. Opting out is explicit:
            # autoload_contract_registry=False or RELIC_AUTOLOAD_CONTRACT_REGISTRY=0.
            from ...contracts.contract_loader import load_contract_registry

            self._contract_registry = load_contract_registry()
        else:
            logger.info(
                "SCM contract registry autoload disabled by %s: ownership, alias, "
                "writer-permission and increment checks in apply_updates are off.",
                autoload_switch,
            )
        self._schema: Dict[str, StateVariableSpec] = schema or self._create_default_schema()
        self._state_variables: Dict[str, Tensor] = {}
        self._history: List[Dict[str, Tensor]] = []
        self._conservation_history: List[ConservationSummary] = []
        self._max_history_length: Optional[int] = validate_max_history_length(
            kwargs.pop("max_history_length", None),
            source="max_history_length",
        )
        self._last_summary: Optional[ConservationSummary] = None
        self._current_diagnostics: Dict[str, Any] = {}
        self._clip_loss_diagnostics: Dict[str, Any] = {}
        self._provenance_recorder = None
        self._provenance_run_id: Optional[str] = None
        self._provenance_transaction_active: bool = False
        self._pending_provenance_events: List[Dict[str, Any]] = []
        self._provenance_delivery_active = False
        self._update_counter: int = 0
        self._state_reset_counter: int = 0
        self._run_specs = None
        # Conservation tolerance baseline (resolved from SSOT, no hardcoded literal).
        self._abs_conservation_tol: float = self._resolve_default_conservation_tolerance()
        # Validated values (validate_conservation_checks); an unset switch keeps
        # its documented default, an unset tolerance the parameter-database value.
        conservation_settings = self._conservation_checks_config
        if "enabled" in conservation_settings:
            self._conservation_enabled = conservation_settings["enabled"]
        else:
            self._conservation_enabled = True
        if "rollback_on_failure" in conservation_settings:
            self._conservation_rollback_default = conservation_settings["rollback_on_failure"]
        else:
            self._conservation_rollback_default = False
        if "raise_on_violation" in conservation_settings:
            self._conservation_raise_on_violation = conservation_settings["raise_on_violation"]
        else:
            self._conservation_raise_on_violation = True
        if "tolerance" in conservation_settings:
            self._abs_conservation_tol = conservation_settings["tolerance"]
        # Auto reporting flag for apply_updates; simulator may turn off per step
        self._enable_auto_report: bool = True
        # Optional explicit conservation ledger mapping element -> list of pool variable names
        # Example: {"carbon": ["c_pool_fast", "c_pool_slow"], "nitrogen": ["n_pool"]}
        self._conservation_ledger: Dict[str, List[str]] = {}
        # Optional scalar targets per element when ledger is provided as totals
        self._conservation_targets: Dict[str, float] = {}
        # Element pools for conservation tracking
        self._element_pools: Dict[str, List[str]] = {}
        # Build the canonical ledger from contract and parameter SSOTs once.
        self._initialize_default_conservation_ledger()

    def _ensure_internal_state_device(self) -> None:
        """Ensure in-memory state tensors are aligned to manager.device."""
        target_device = torch.device(self.device)
        for key, value in list(self._state_variables.items()):
            if isinstance(value, Tensor) and value.device != target_device:
                self._state_variables[key] = value.to(device=target_device)

    def to_device(self, device: Union[str, torch.device]) -> None:
        """Move managed state tensors to target device and update manager.device."""
        target_device = torch.device(device)
        if target_device.type == "cuda" and not device_policy.is_cuda_available():
            target_device = torch.device("cpu")
        self.device = target_device
        self._ensure_internal_state_device()

    def _resolve_default_conservation_tolerance(self) -> float:
        """The default commit-time ledger tolerance: one source, no fallback.

        It is the declared parameter ``validation.conservation.conservation_tolerance``
        (``CONSERVATION_TOLERANCE_PARAMETER``), read with unit and bounds validation; a
        missing parameter or an unavailable parameter database raises.
        """
        return float(
            _require_parameter_manager().get(
                CONSERVATION_TOLERANCE_PARAMETER,
                units="dimensionless",
                expected_type=float,
                required=True,
            )
        )

    @property
    def state_vars(self) -> Dict[str, Tensor]:
        """Return current state variables mapping."""
        return self._state_variables
    
    @property
    def schema(self) -> Dict[str, StateVariableSpec]:
        """Public accessor for variable schema."""
        return self._schema

    def set_auto_report_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic report generation for update application."""
        self._enable_auto_report = bool(enabled)

    def get_variable_spec(self, name: str) -> Optional[StateVariableSpec]:
        """Return variable specification by name."""
        return self._schema.get(name)

    @property
    def conservation_tolerance(self) -> float:
        """Get conservation tolerance value."""
        return self._abs_conservation_tol
    
    @conservation_tolerance.setter
    def conservation_tolerance(self, value: float) -> None:
        """Set the ledger tolerance; validated like ``conservation_checks["tolerance"]``."""
        self._abs_conservation_tol = validate_conservation_setting(
            "tolerance", value, source="StateManager.conservation_tolerance"
        )

    @property
    def max_history_length(self) -> Optional[int]:
        """Return optional O(1) upper bound on in-memory history and conservation ledgers."""
        return self._max_history_length

    @max_history_length.setter
    def max_history_length(self, value: Optional[int]) -> None:
        """Set optional O(1) history capacity and immediately trim retained ledgers in-place."""
        self._max_history_length = validate_max_history_length(
            value, source="StateManager.max_history_length"
        )
        self._trim_history_ledgers()

    def _trim_history_ledgers(self) -> None:
        """Enforce O(1) bounded length on _history and _conservation_history in-place.

        1. References:
           - Griewank, A., & Walther, A. (2008). Evaluating Derivatives (ring buffer /
             sliding window checkpointing).
        2. Specific mathematical expressions for the calculation:
           - L_t <- L_t[max(0, |L_t| - M) : |L_t|], ensuring |L_t| <= M = O(1).
        3. Meanings of the parameters:
           - M (self._max_history_length): Maximum retained entries (dimensionless count).
        """
        limit = self._max_history_length
        if limit is None:
            return
        if limit == 0:
            del self._history[:]
            del self._conservation_history[:]
            return
        if len(self._history) > limit:
            del self._history[:-limit]
        if len(self._conservation_history) > limit:
            del self._conservation_history[:-limit]

    def record_history_snapshot(self) -> Dict[str, Tensor]:
        """Record a cloned state snapshot into _history and enforce O(1) capacity bound.

        1. References:
           - Gray, J., & Reuter, A. (1992). Transaction Processing: Concepts and Techniques.
        2. Specific mathematical expressions for the calculation:
           - h_t = clone(S_t), H_t = trim(H_{t-1} + [h_t], M)
        3. Meanings of the parameters:
           - S_t: Current state dictionary mapping variable names to tensors (canonical units).
           - M: Optional maximum history length (entries, dimensionless).
        """
        snapshot = self.get_state(clone=True)
        self._history.append(snapshot)
        self._trim_history_ledgers()
        return snapshot

    def rollback(self, steps: int = 1) -> Dict[str, Tensor]:
        """Restore state from the bounded history ledger by popping ``steps`` snapshots.

        1. References:
           - Gray, J., & Reuter, A. (1992). Transaction Processing: Concepts and Techniques.
        2. Specific mathematical expressions for the calculation:
           - S_{t - k} = H_t[|H_t| - k], H_{t - k} = H_t[0 : |H_t| - k] for 1 <= k <= |H_t|.
        3. Meanings of the parameters:
           - steps (int): Number of historical snapshots k >= 1 to roll back (dimensionless).
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(
                f"'steps' must be a positive int; got {steps!r} ({type(steps).__name__})."
            )
        if len(self._history) < steps:
            raise ValueError(
                f"Cannot rollback {steps} step(s); only {len(self._history)} snapshot(s) "
                "available in history."
            )
        target_snapshot: Dict[str, Tensor] = {}
        for _ in range(steps):
            target_snapshot = self._history.pop()
        self._state_variables = {
            key: value.clone().to(device=self.device) if isinstance(value, Tensor) else value
            for key, value in target_snapshot.items()
        }
        return self.get_state()

    def set_run_specs(self, run_specs: Any) -> None:
        """Attach RunSpecs for update mode enforcement."""
        self._run_specs = run_specs

    def set_provenance_recorder(self, recorder: Any, *, run_id: Optional[str] = None) -> None:
        """Attach an observer and consume the simulator's run identity."""
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("Provenance run_id binding must be a nonempty string")
        if recorder is not None and not callable(getattr(recorder, "record_state_update", None)):
            raise TypeError("A provenance recorder must implement record_state_update")
        if recorder is not self._provenance_recorder:
            if self._provenance_transaction_active:
                raise RuntimeError("Cannot replace a recorder during a provenance transaction")
            if self._provenance_recorder is not None and self._pending_provenance_events:
                self.commit_provenance_transaction()
        self._provenance_recorder = recorder
        if run_id is not None:
            self._provenance_run_id = run_id
        if self._provenance_run_id is not None and recorder is not None:
            from ..provenance.recorder import ProvenanceRecorder
            if isinstance(recorder, ProvenanceRecorder):
                recorder.run_id = self._provenance_run_id

    @staticmethod
    def _clone_transaction_value(value: Any) -> Any:
        """Clone transaction payloads without retaining mutable tensor aliases."""
        if isinstance(value, Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: StateManager._clone_transaction_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [StateManager._clone_transaction_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(StateManager._clone_transaction_value(item) for item in value)
        return copy.deepcopy(value)

    def begin_provenance_transaction(self) -> None:
        """Buffer recorder notifications until the surrounding simulator step commits."""
        if self._provenance_transaction_active:
            raise RuntimeError("A provenance transaction is already active")
        if self._pending_provenance_events:
            self.commit_provenance_transaction()
        self._provenance_transaction_active = True

    def commit_provenance_transaction(self) -> None:
        """Publish buffered recorder notifications for an accepted simulator step."""
        if self._provenance_delivery_active:
            raise RuntimeError("Cannot reenter accepted provenance notification delivery")
        self._provenance_transaction_active = False
        self._provenance_delivery_active = True
        try:
            while self._pending_provenance_events:
                self._emit_provenance_event(self._pending_provenance_events[0])
                self._pending_provenance_events.pop(0)
        finally:
            self._provenance_delivery_active = False

    def rollback_provenance_transaction(self) -> None:
        """Discard recorder notifications produced by a rejected simulator step."""
        if self._provenance_transaction_active:
            self._pending_provenance_events.clear()
            self._provenance_transaction_active = False

    def _emit_provenance_event(self, event: Dict[str, Any]) -> None:
        """Send one accepted state-update event to the configured recorder."""
        recorder = self._provenance_recorder
        if recorder is None:
            raise RuntimeError("Accepted provenance notification has no recorder")
        try:
            delivery_options = (
                {"notification_context": event["notification_context"]}
                if "notification_context" in event else {}
            )
            # The commit context reaches a recorder whose record_state_update
            # declares the commit_context keyword or **kwargs; a six-argument
            # recorder keeps its protocol. The recorder receives a copy, so a
            # failed delivery retains the pending event unchanged for its retry.
            if "commit_context" in event and self._recorder_accepts_commit_context(recorder):
                delivery_options["commit_context"] = self._clone_transaction_value(
                    event["commit_context"]
                )
            recorder.record_state_update(
                event["step"],
                event["variable"],
                event["pre_value"],
                event["post_value"],
                event["functor_increments"],
                event["unit"],
                **delivery_options,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Provenance notification failed after state commit: {exc}"
            ) from exc

    @staticmethod
    def _recorder_accepts_commit_context(recorder: Any) -> bool:
        """Whether ``recorder.record_state_update`` declares a commit_context keyword or **kwargs.

        A method whose signature cannot be inspected is treated as the
        six-argument protocol (it receives no commit_context).
        """
        try:
            parameters = inspect.signature(recorder.record_state_update).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or (
                parameter.name == "commit_context"
                and parameter.kind in (
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            for parameter in parameters
        )

    def _initialize_default_conservation_ledger(self) -> None:
        """Finalize conservation ledger using parameter and contract sources."""
        _state_conservation.initialize_default_conservation_ledger(self)


    def _get_physical_constraint_manager(self) -> Any:
        """
        Return the injected constraint owner or construct this runtime's owner.

        Resolution order:
        1) Explicitly injected manager via constructor.
        2) One PhysicalConstraintManager created and cached by this StateManager.

        Resolution errors propagate so physical constraints cannot be silently disabled.
        """
        if self._physical_constraint_manager is not None:
            return self._physical_constraint_manager
        from ...coupling.scales.physical_constraint_manager import (
            PhysicalConstraintManager,
        )

        constraint_manager = PhysicalConstraintManager(
            parameter_manager=_get_parameter_manager_optional(),
        )
        constraint_manager.initialize()
        constraint_manager.validate()
        self._physical_constraint_manager = constraint_manager
        return self._physical_constraint_manager

    @property
    def physical_constraint_manager(self) -> Any:
        """Expose the unique physical-constraint owner through a read-only path."""
        return self._get_physical_constraint_manager()


    
    @staticmethod
    def _value_to_float(value: Any) -> Optional[float]:
        """Reduce a tensor, an UpdateItem or a scalar to a float budget.

        None only for an unsupported type; a complex tensor raises ValueError
        (the real-data constraint of ``_tensor_budget_sum``).
        """
        if isinstance(value, UpdateItem):
            return StateManager._value_to_float(value.value)
        if isinstance(value, Tensor):
            return float(StateManager._tensor_budget_sum(value.detach()).cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _tensor_budget_sum(value: Tensor) -> Tensor:
        """Reduce a real tensor budget in float64 without changing its stored state."""
        # References: Higham (2002), Accuracy and Stability of Numerical Algorithms.
        # Mathematical expression: budget = sum_i float64(delta_i).
        # Parameter meanings: delta_i are realized cell or pool changes.
        # Units: unchanged; the existing unit owner normalizes the scalar sum.
        # Domain/constraints: real data (a complex tensor raises ValueError);
        # promote before reduction so cancellation in a lower-precision
        # accumulator cannot erase a physical residual.
        # Sign convention: positive budget means a net gain of conserved storage.
        if value.is_complex():
            raise ValueError("Conservation budget values must be real")
        return value.to(dtype=torch.float64) if value.numel() == 1 else value.sum(dtype=torch.float64)

    @staticmethod
    def _value_to_sum_tensor(value: Any) -> Optional[Tensor]:
        """Reduce a budget value to a float64 tensor; None only for an unsupported type."""
        if isinstance(value, UpdateItem):
            return StateManager._value_to_sum_tensor(value.value)
        if isinstance(value, Tensor):
            return StateManager._tensor_budget_sum(value)
        if isinstance(value, (int, float)):
            return torch.tensor(float(value), dtype=torch.float64)
        return None

    def _check_conservation(
        self,
        updates: Dict[str, Any],
        dt: Optional[float] = None,
        scale_factor: Optional[float] = None,
    ) -> Dict[str, float]:
        """Check conservation violations for updates with optional dynamic tolerance."""
        return cast(
            Dict[str, float],
            _state_conservation.check_conservation(self, updates, dt=dt, scale_factor=scale_factor),
        )



    def _check_element_conservation(self, updates: Dict[str, Any], element: str) -> Dict[str, Any]:
        """Compute per-element conservation diagnostics with ledger awareness."""
        return cast(Dict[str, Any], _state_conservation.check_element_conservation(self, updates, element))
    def _get_element_pool_variables(self, element: str) -> List[str]:
        """Collect pool variable names for an element from contracts and ledger."""
        names: List[str] = []
        element_key = element.lower()

        if self._contract_registry is not None:
            domain = element_key if "." in element_key else f"{element_key}.total"
            members = self._contract_registry.get_domain_members(domain)
            if members:
                names.extend(members)

        ledger_members = self._conservation_ledger.get(element_key, [])
        if ledger_members:
            names.extend(ledger_members)

        return names

    def _initialize_element_pools(self) -> None:
        """Initialize element pool tracking for conservation."""
        pools: Dict[str, List[str]] = {}
        elements: Set[str] = set()

        elements.update(self._conservation_ledger.keys())

        if self._contract_registry is not None:
            for domain in self._contract_registry.list_domains():
                element = domain.split(".")[0].lower()
                if element:
                    elements.add(element)

        # Only declared elements (conservation ledger, contract domains) have
        # pools; with none declared there is no element pool to audit, and no
        # element list is assumed.
        for element in sorted(elements):
            names: List[str] = []
            names.extend(self._get_element_pool_variables(element))
            names.extend(self._infer_element_pools_from_schema(element))
            deduped: List[str] = []
            seen: Set[str] = set()
            for name in names:
                if not name:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(name)
            pools[element] = deduped
        self._element_pools = pools

    def _infer_element_pools_from_schema(self, element: str) -> List[str]:
        """Infer pool names for an element based on schema metadata."""
        inferred: List[str] = []
        for name, spec in self._schema.items():
            contract = (
                self._contract_registry.get_contract(name)
                if self._contract_registry is not None
                else None
            )
            if contract is not None:
                domain = getattr(contract, "accumulates", None)
                if domain and str(domain).split(".", 1)[0].lower() == element:
                    inferred.append(name)
                continue

            schema_domain = getattr(spec, "accumulates", None)
            if schema_domain is not None:
                if str(schema_domain).split(".", 1)[0].lower() == element:
                    inferred.append(name)
                continue

            lname = name.lower()
            if element != "water" and not spec.conserved:
                continue
            if self._matches_element_keyword(lname, element):
                inferred.append(name)
        return inferred

    @staticmethod
    def _matches_element_keyword(name_lower: str, element: str) -> bool:
        """Heuristic matching between variable names and conserved elements."""
        # Exclude common diagnostic/modifier variables from conservation-pool inference.
        # References:
        # - Clark et al. (2008), process factors are modifiers, not conserved storages.
        # Mathematical expression:
        # - pool_candidate(name) = keyword_match(name) AND NOT diagnostic_token(name)
        # Parameter meanings:
        # - diagnostic_token: lexical token indicating dimensionless modifiers (factor/index/ratio/stress).
        diagnostic_tokens = (
            "diagnostics.",
            "_factor",
            "factor_",
            "_index",
            "_ratio",
            "_coefficient",
            "_coef",
            "_fraction",
            "_weight",
            "stress",
            "limitation",
        )
        if any(token in name_lower for token in diagnostic_tokens):
            return False

        if element == "carbon":
            return (
                "carbon" in name_lower
                or name_lower.startswith("c_")
                or name_lower.endswith("_c")
                or "_carbon_" in name_lower
            )
        if element == "nitrogen":
            return (
                "nitrogen" in name_lower
                or name_lower.startswith("n_")
                or name_lower.endswith("_n")
                or "_nitrogen_" in name_lower
                or name_lower in {"nh3", "nh4", "no3", "no2", "n2o", "nox"}
            )
        if element == "phosphorus":
            return (
                "phosphorus" in name_lower
                or name_lower.startswith("p_")
                or name_lower.endswith("_p")
                or "_phosphorus_" in name_lower
            )
        if element == "water":
            return (
                "water" in name_lower
                or "h2o" in name_lower
                or name_lower.endswith("_swc")
            )
        return False

    def _compute_pool_reference_scale(self, element: str, pools: List[str]) -> float:
        """Compute reference magnitude for relative conservation errors."""
        reference = 0.0
        for pool in pools:
            if pool not in self._state_variables:
                continue
            value = self._value_to_float(self._state_variables[pool])
            if value is None:
                continue
            reference += abs(value)
        return max(reference, 1.0)

    @property
    def element_pools(self) -> Dict[str, List[str]]:
        """Expose a copy of the element pool mapping."""
        if not self._element_pools:
            self._initialize_element_pools()
        return {element: pools.copy() for element, pools in self._element_pools.items()}

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize internal state with optional initial values.

        Parameters
        ----------
        initial_state : Optional[Dict[str, Any]]
            Optional initial values. Non-tensors will be converted to float tensors.
        """
        entries: Dict[str, Any] = {}
        if initial_state:
            entries = initial_state
        else:
            for name, spec in self._schema.items():
                if not spec.required or name in self._state_variables:
                    continue
                if spec.default is not None:
                    entries[name] = spec.default
                elif spec.default_value is not None:
                    entries[name] = spec.default_value
                else:
                    entries[name] = 0.0

        prepared = self._prepare_state(entries)
        self._validate_bounds(prepared)
        self._state_variables.update(prepared)

    def configure(
        self,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Apply post-construction settings (``STATE_MANAGER_CONFIGURABLE_KEYS``).

        ``config`` and keyword arguments name the same settings; a key given both
        ways with different values, an undeclared key, or an invalid value raises
        ValueError before any setting changes.
        """
        if config is not None and not isinstance(config, Mapping):
            raise ValueError(
                f"StateManager.configure expects a mapping of settings; got {config!r} "
                f"({type(config).__name__})."
            )
        settings: Dict[str, Any] = dict(config or {})
        for key, value in kwargs.items():
            if key in settings and settings[key] != value:
                raise ValueError(
                    f"StateManager.configure received '{key}' twice with different values: "
                    f"{settings[key]!r} and {value!r}."
                )
            settings[key] = value
        unknown = sorted(str(key) for key in settings if key not in STATE_MANAGER_CONFIGURABLE_KEYS)
        if unknown:
            raise ValueError(
                f"StateManager.configure does not accept {unknown}; the configurable "
                f"settings are {list(STATE_MANAGER_CONFIGURABLE_KEYS)}."
            )
        validated_tolerance = None
        if "conservation_tolerance" in settings:
            validated_tolerance = validate_conservation_setting(
                "tolerance",
                settings["conservation_tolerance"],
                source="StateManager.conservation_tolerance",
            )
        has_max_history = "max_history_length" in settings
        validated_max_history: Optional[int] = None
        if has_max_history:
            validated_max_history = validate_max_history_length(
                settings["max_history_length"],
                source="StateManager.max_history_length",
            )
        if validated_tolerance is not None:
            self._abs_conservation_tol = validated_tolerance
        if has_max_history:
            self._max_history_length = validated_max_history
            self._trim_history_ledgers()

    def validate(self) -> List[str]:
        """Validate state and transition to READY. Returns validation errors (empty when clean)."""
        errors = self.get_validation_errors()

        if not errors:
            return []
        return errors
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors for current state."""
        errors = []
        
        # Check for negative values where not allowed
        for name, value in self._state_variables.items():
            spec = self._schema.get(name)
            # Bounds are enforced exactly where they are declared (spec.bounds);
            # a signed quantity declares signed bounds or none.
            if spec and spec.bounds:
                lo, hi = spec.bounds
                if isinstance(value, Tensor):
                    if torch.any(value < lo):
                        errors.append(f"Variable {name} has values below minimum {lo}")
                    if torch.any(value > hi):
                        errors.append(f"Variable {name} has values above maximum {hi}")
                else:
                    val = float(value)
                    if val < lo or val > hi:
                        errors.append(f"Variable {name} value {val} out of bounds [{lo}, {hi}]")
                        
        return errors
    


    def reset(self) -> None:
        """Reset runtime state and histories while preserving configuration."""
        if self._provenance_transaction_active:
            self.rollback_provenance_transaction()
        elif self._pending_provenance_events:
            self.commit_provenance_transaction()
        self._state_variables.clear()
        self._history.clear()
        self._conservation_history.clear()
        self._last_summary = None
        self._current_diagnostics.clear()
        self._clip_loss_diagnostics.clear()
        self._update_counter = 0

    # Test helper initializer used by integration tests
    def initialize_state(self, initial_state: Dict[str, Any]) -> None:
        """Initialize internal state with provided initial values.

        Parameters
        ----------
        initial_state : Dict[str, Any]
            Mapping of variable name to value.
        """
        prepared = self._prepare_state(initial_state)
        self._validate_bounds(prepared)
        self.reset()
        self._state_variables.update(prepared)
        self._state_reset_counter += 1

    def _create_default_schema(self) -> Dict[str, StateVariableSpec]:
        """Build schema dynamically from ContractRegistry.

        No hardcoded scientific variables - all definitions come from
        contract YAML files in relic/contracts/variables/.
        """
        schema: Dict[str, StateVariableSpec] = {}

        # Load from contract registry if available
        if hasattr(self, "_contract_registry") and self._contract_registry is not None:
            for var_name in self._contract_registry.list_variables():
                spec = self._contract_registry.to_state_variable_spec(var_name)
                if spec:
                    schema[var_name] = spec

        # Fallback: only 'time' as universal variable if no contracts loaded
        if not schema:
            schema["time"] = StateVariableSpec(
                name="time",
                units="s",
                bounds=(0.0, float("inf")),
                conserved=False,
                spatial_dims=0,
                required=False,
                default_value=0.0,
                variable_type=StateVariableType.SCALAR,
            )

        return schema

    def _prepare_state(self, entries: Dict[str, Any]) -> Dict[str, Tensor]:
        """Convert state entries to device tensors without mutating runtime state."""
        prepared: Dict[str, Tensor] = {}
        for key, raw_value in entries.items():
            if isinstance(raw_value, Tensor):
                prepared[key] = raw_value.clone().to(device=self.device)
            else:
                prepared[key] = torch.as_tensor(
                    raw_value,
                    dtype=torch.float32,
                    device=self.device,
                )
        return prepared

    def set_state(
        self,
        state: Union[Dict[str, Any], str],
        value: Optional[Any] = None,
        *,
        validate: bool = True,
    ) -> None:
        """Set the full state map.

        Parameters
        ----------
        state : Dict[str, Any]
            Mapping from variable to numeric or tensor value.
        validate : bool
            Whether to validate bounds against schema.
        """
        if isinstance(state, str):
            if value is None:
                raise ValueError("Setting a single state variable requires a value.")
            entries: Dict[str, Any] = {state: value}
        else:
            if value is not None:
                raise ValueError("Value argument can only be used when setting a single variable.")
            entries = state

        new_state = self._prepare_state(entries)
        if validate:
            self._validate_bounds(new_state)
        self._state_variables.update(new_state)

    def update_state(self, updates: Dict[str, Any], *, validate: bool = False) -> None:
        """
        Helper to merge values into the current state.

        Notes
        -----
        This is a convenience API used by some integration tests. The canonical
        state update path during simulation is 'apply_updates'.
        """
        merged = self.get_state()
        merged.update(updates)
        self.set_state(merged, validate=validate)

    @overload
    def get_state(self, variables: str, *, clone: bool = True) -> Optional[Tensor]: ...

    @overload
    def get_state(self, variables: Optional[List[str]] = None, *, clone: bool = True) -> Dict[str, Tensor]: ...

    def get_state(
        self, variables: Optional[Union[str, List[str]]] = None, *, clone: bool = True,
    ) -> Union[Dict[str, Tensor], Tensor, None]:
        """Return the current state dict or selected variables.

        Parameters
        ----------
        variables : str, list of str, or None
            A name returns its tensor or None; a list returns selected variables.
            None returns all variables.
        clone : bool
            When True (default), clone tensors to prevent accidental in-place mutation by callers.
            When False, return tensor references for performance-critical read-only access.
        """
        self._ensure_internal_state_device()
        if isinstance(variables, str):
            value = self._state_variables.get(variables)
            if isinstance(value, Tensor) and clone:
                return value.clone()
            return value
        if variables is None:
            if clone:
                return {k: v.clone() if isinstance(v, Tensor) else v for k, v in self._state_variables.items()}
            return dict(self._state_variables)
        else:
            # Return only requested variables
            result = {}
            for var_name in variables:
                if var_name in self._state_variables:
                    var_value = self._state_variables[var_name]
                    if isinstance(var_value, Tensor) and clone:
                        result[var_name] = var_value.clone()
                    else:
                        result[var_name] = var_value
            return result

    def get_current_state(
        self,
        variables: Optional[List[str]] = None,
        *,
        clone: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Retrieve the current state snapshot.

        Parameters
        ----------
        variables : Optional[List[str]]
            Optional subset of variable names.
        clone : bool
            Whether to clone tensors before returning.
        """
        return self.get_state(variables=variables, clone=clone)

    def create_transaction_savepoint(self) -> Dict[str, Any]:
        """Capture one lightweight append-only simulator-step savepoint."""
        savepoint: Dict[str, Any] = {
            "kind": "state_manager_transaction_savepoint_v1",
            "state": self.get_state(clone=True),
            "update_counter": int(self._update_counter),
            "state_reset_counter": int(self._state_reset_counter),
            "diagnostics": self._clone_transaction_value(
                self._current_diagnostics
            ),
            "history_owner": self._history,
            "history_length": len(self._history),
            "conservation_history_owner": self._conservation_history,
            "conservation_history_length": len(self._conservation_history),
            "max_history_length": self._max_history_length,
            "last_summary": self._clone_transaction_value(self._last_summary),
            "clip_loss_diagnostics": self._clone_transaction_value(
                self._clip_loss_diagnostics
            ),
            "auto_report_enabled": bool(self._enable_auto_report),
            "provenance_transaction_active": bool(
                self._provenance_transaction_active
            ),
            "pending_provenance_owner": self._pending_provenance_events,
            "pending_provenance_length": len(self._pending_provenance_events),
        }
        if self._max_history_length is not None:
            savepoint["bounded_history_window"] = list(self._history)
            savepoint["bounded_conservation_window"] = list(
                self._conservation_history
            )
        return savepoint

    def restore_transaction_savepoint(self, savepoint: Dict[str, Any]) -> None:
        """Restore a savepoint created by ``create_transaction_savepoint``."""
        if (
            not isinstance(savepoint, dict)
            or savepoint.get("kind")
            != "state_manager_transaction_savepoint_v1"
        ):
            raise ValueError("Invalid StateManager transaction savepoint")

        raw_state = savepoint.get("state")
        if not isinstance(raw_state, dict):
            raise ValueError("StateManager transaction savepoint has no state")

        restored_state: Dict[str, Tensor] = {}
        for key, value in raw_state.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                raise TypeError(
                    "StateManager transaction savepoint state must map names "
                    "to tensors"
                )
            restored_state[key] = value.clone().to(device=self.device)

        def restore_append_only(
            *,
            owner_key: str,
            length_key: str,
            attribute: str,
            bounded_window_key: Optional[str] = None,
        ) -> None:
            owner = savepoint.get(owner_key)
            length = savepoint.get(length_key)
            if not isinstance(owner, list) or type(length) is not int:
                raise ValueError(
                    "StateManager transaction savepoint append-only ledger "
                    f"is invalid: {attribute}"
                )
            bounded_window = (
                savepoint.get(bounded_window_key)
                if bounded_window_key is not None
                else None
            )
            if isinstance(bounded_window, list) and len(bounded_window) == length:
                owner[:] = bounded_window
                setattr(self, attribute, owner)
                return
            if length < 0 or len(owner) < length:
                raise RuntimeError(
                    "StateManager append-only ledger was destructively "
                    f"modified during the transaction: {attribute}"
                )
            del owner[length:]
            setattr(self, attribute, owner)

        restore_append_only(
            owner_key="history_owner",
            length_key="history_length",
            attribute="_history",
            bounded_window_key="bounded_history_window",
        )
        restore_append_only(
            owner_key="conservation_history_owner",
            length_key="conservation_history_length",
            attribute="_conservation_history",
            bounded_window_key="bounded_conservation_window",
        )
        restore_append_only(
            owner_key="pending_provenance_owner",
            length_key="pending_provenance_length",
            attribute="_pending_provenance_events",
        )

        if "max_history_length" in savepoint:
            self._max_history_length = validate_max_history_length(
                savepoint["max_history_length"],
                source="savepoint.max_history_length",
            )
        self._state_variables = restored_state
        self._update_counter = int(savepoint["update_counter"])
        self._state_reset_counter = int(savepoint["state_reset_counter"])
        self._current_diagnostics = self._clone_transaction_value(
            savepoint["diagnostics"]
        )
        self._last_summary = self._clone_transaction_value(
            savepoint["last_summary"]
        )
        self._clip_loss_diagnostics = self._clone_transaction_value(
            savepoint["clip_loss_diagnostics"]
        )
        self._enable_auto_report = bool(savepoint["auto_report_enabled"])
        self._provenance_transaction_active = bool(
            savepoint["provenance_transaction_active"]
        )
        self._trim_history_ledgers()

    def checkpoint(self) -> Dict[str, Any]:
        """
        Create a restorable snapshot of state manager runtime state.

        Returns
        -------
        Dict[str, Any]
            Snapshot payload for ``restore``.
        """
        return {
            "state": self.get_state(clone=True),
            "update_counter": int(self._update_counter),
            "state_reset_counter": int(self._state_reset_counter),
            "diagnostics": self._clone_transaction_value(self._current_diagnostics),
            "history": self._clone_transaction_value(self._history),
            "conservation_history": self._clone_transaction_value(
                self._conservation_history
            ),
            "last_summary": self._clone_transaction_value(self._last_summary),
            "clip_loss_diagnostics": self._clone_transaction_value(
                self._clip_loss_diagnostics
            ),
            "auto_report_enabled": bool(self._enable_auto_report),
            "provenance_transaction_active": bool(
                self._provenance_transaction_active
            ),
            "pending_provenance_events": self._clone_transaction_value(
                self._pending_provenance_events
            ),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore state manager runtime state from a checkpoint payload.

        Parameters
        ----------
        snapshot : Dict[str, Any]
            Payload produced by ``checkpoint``.
        """
        if not isinstance(snapshot, dict):
            raise TypeError("StateManager snapshot must be a dictionary.")
        raw_state = snapshot.get("state")
        if not isinstance(raw_state, dict):
            raise ValueError("StateManager snapshot is missing a valid 'state' mapping.")

        restored_state: Dict[str, Tensor] = {}
        for key, value in raw_state.items():
            if isinstance(value, Tensor):
                restored_state[key] = value.clone().to(device=self.device)
            else:
                restored_state[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        self._state_variables = restored_state
        self._update_counter = int(snapshot.get("update_counter", self._update_counter))
        self._state_reset_counter = int(
            snapshot.get("state_reset_counter", self._state_reset_counter)
        )
        raw_diag = snapshot.get("diagnostics", {})
        if isinstance(raw_diag, dict):
            self._current_diagnostics = self._clone_transaction_value(raw_diag)
        raw_history = snapshot.get("history", [])
        if isinstance(raw_history, list):
            self._history = self._clone_transaction_value(raw_history)
        raw_conservation_history = snapshot.get("conservation_history", [])
        if isinstance(raw_conservation_history, list):
            self._conservation_history = self._clone_transaction_value(
                raw_conservation_history
            )
        if "max_history_length" in snapshot:
            self._max_history_length = validate_max_history_length(
                snapshot["max_history_length"],
                source="snapshot.max_history_length",
            )
        self._last_summary = self._clone_transaction_value(
            snapshot.get("last_summary")
        )
        raw_clip_diagnostics = snapshot.get("clip_loss_diagnostics", {})
        if isinstance(raw_clip_diagnostics, dict):
            self._clip_loss_diagnostics = self._clone_transaction_value(
                raw_clip_diagnostics
            )
        self._enable_auto_report = bool(
            snapshot.get("auto_report_enabled", self._enable_auto_report)
        )
        self._provenance_transaction_active = bool(
            snapshot.get(
                "provenance_transaction_active",
                self._provenance_transaction_active,
            )
        )
        raw_pending_events = snapshot.get("pending_provenance_events", [])
        if isinstance(raw_pending_events, list):
            self._pending_provenance_events = self._clone_transaction_value(
                raw_pending_events
            )
        self._trim_history_ledgers()



    def get_spatial_variables(self) -> List[str]:
        """
        Get list of spatial state variables.

        Returns:
            List[str]: Names of variables with spatial_dims > 0
        """
        spatial_vars: List[str] = []
        for var_name, spec in self._schema.items():
            if spec.spatial_dims > 0:
                spatial_vars.append(var_name)
        return spatial_vars

    def get_variable_spatial_info(self, var_name: str) -> Dict[str, Any]:
        """
        Get spatial information for a specific variable.

        Parameters:
            var_name : str
                Name of the variable

        Returns:
            Dict[str, Any]: Spatial information including dimensions, shape, and scale
        """
        if var_name not in self._state_variables:
            return {}
        
        var_value = self._state_variables[var_name]
        var_spec = self._schema.get(var_name)
        
        if not isinstance(var_value, Tensor) or not var_spec:
            return {}
        
        info = {
            'name': var_name,
            'shape': list(var_value.shape),
            'size': var_value.numel(),
            'dimensions': var_value.dim(),
            'spatial_dims': var_spec.spatial_dims,
            'units': var_spec.units
        }
        
        # Determine variable type scale
        if var_spec.spatial_dims == 0:
            info['scale'] = 'scalar'
        elif var_spec.spatial_dims == 1:
            if var_value.numel() <= 10:
                info['scale'] = 'patch'
            elif var_value.numel() <= 100:
                info['scale'] = 'hillslope'
            else:
                info['scale'] = 'basin'
        elif var_spec.spatial_dims == 2:
            info['scale'] = 'grid_2d'
        elif var_spec.spatial_dims == 3:
            info['scale'] = 'grid_3d'
        
        return info

    def update_environment(self, env: Dict[str, Any]) -> None:
        """Update environment/forcing variables into the state.
        Non-tensor values will be converted to float tensors."""
        target_spatial = None
        target_batch = None
        for value in self._state_variables.values():
            if isinstance(value, Tensor) and value.dim() >= 2:
                target_spatial = tuple(int(v) for v in value.shape[-2:])
                target_batch = int(value.shape[0]) if value.dim() >= 3 else None
                break

        for k, v in env.items():
            if isinstance(v, Tensor):
                tensor = v.to(device=self.device)
            else:
                tensor = torch.tensor(float(v), device=self.device)
            if target_spatial is not None:
                if tensor.dim() == 0:
                    if target_batch is None:
                        tensor = tensor.view(1, 1).repeat(target_spatial[0], target_spatial[1])
                    else:
                        tensor = tensor.view(1, 1, 1).repeat(target_batch, target_spatial[0], target_spatial[1])
                elif tensor.dim() == 1:
                    if target_batch is not None and tensor.shape[0] == target_batch:
                        tensor = tensor.view(-1, 1, 1).repeat(1, target_spatial[0], target_spatial[1])
                elif tensor.dim() == 2:
                    if tensor.shape != target_spatial:
                        tensor = F.interpolate(
                            tensor.unsqueeze(0).unsqueeze(0),
                            size=target_spatial,
                            mode="nearest",
                        ).squeeze(0).squeeze(0)
                    if target_batch is not None:
                        tensor = tensor.unsqueeze(0).repeat(target_batch, 1, 1)
                elif tensor.dim() == 3:
                    if tensor.shape[-2:] != target_spatial:
                        tensor = F.interpolate(
                            tensor.unsqueeze(1),
                            size=target_spatial,
                            mode="nearest",
                        ).squeeze(1)
                    if target_batch is not None and tensor.shape[0] != target_batch:
                        if tensor.shape[0] == 1:
                            tensor = tensor.repeat(target_batch, 1, 1)
                        else:
                            tensor = tensor[:target_batch, ...]
                elif tensor.dim() >= 4:
                    if tensor.shape[-2:] != target_spatial:
                        tensor = F.interpolate(tensor, size=target_spatial, mode="nearest")
                    if target_batch is not None and tensor.shape[0] != target_batch:
                        if tensor.shape[0] == 1:
                            tensor = tensor.repeat(target_batch, *([1] * (tensor.dim() - 1)))
                        else:
                            tensor = tensor[:target_batch, ...]
            self._state_variables[k] = tensor

    def register_variable(self, spec: StateVariableSpec) -> None:
        """Register a new variable specification into the schema."""
        if isinstance(spec, StateVariableSpec):
            self._schema[spec.name] = spec
            if spec.default is not None:
                if isinstance(spec.default, Tensor):
                    self._state_variables.setdefault(spec.name, spec.default.clone().to(device=self.device))
                else:
                    self._state_variables.setdefault(spec.name, torch.tensor(float(spec.default), device=self.device))
            elif spec.default_value is not None:
                self._state_variables.setdefault(spec.name, torch.tensor(float(spec.default_value), device=self.device))
            else:
                self._state_variables.setdefault(spec.name, torch.tensor(0.0, device=self.device))
            # Refresh cached element pools so ledger aliases remain consistent
            self._element_pools.clear()
            return

        raise TypeError("register_variable expects a StateVariableSpec")





    def validate_state(self, state: Dict[str, Any]) -> List[str]:
        """Lightweight state validation against schema bounds.
        
        Returns a list of error strings (empty if valid).
        """
        errors: List[str] = []
        candidate: Dict[str, Tensor] = {}
        for k, v in state.items():
            if isinstance(v, Tensor):
                candidate[k] = v.to(device=self.device)
            else:
                candidate[k] = torch.tensor(float(v), device=self.device)
        for name, spec in self._schema.items():
            if spec.bounds is None or name not in candidate:
                continue
            val = candidate[name]
            lo, hi = spec.bounds
            if isinstance(val, Tensor):
                if torch.any(val < lo):
                    errors.append(f"{name} has values below minimum bound {lo}")
                elif torch.any(val > hi):
                    errors.append(f"{name} has values above maximum bound {hi}")
            else:
                x = float(val)
                if x < lo:
                    errors.append(f"{name} value {x} below minimum bound {lo}")
                elif x > hi:
                    errors.append(f"{name} value {x} above maximum bound {hi}")
        return errors

    def _apply_run_specs(self, name: str, item: UpdateItem) -> UpdateItem:
        run_specs = self._run_specs
        if run_specs is None or not hasattr(run_specs, "resolve_update_mode"):
            return item
        mode_value = item.mode.value if isinstance(item.mode, UpdateMode) else str(item.mode)
        resolved = run_specs.resolve_update_mode(name, mode_value)
        resolved_mode = UpdateMode(resolved)
        if resolved_mode != item.mode:
            return UpdateItem(
                value=item.value,
                mode=resolved_mode,
                write_mode=item.write_mode,
                unit=item.unit,
                basis=item.basis,
                per_timestep=item.per_timestep,
                element=item.element,
                source_functor=item.source_functor,
            )
        return item

    @staticmethod
    def _coerce_write_mode(value: Any) -> StateWriteMode:
        """Normalize a write mode declaration; an unknown mode raises."""
        if isinstance(value, StateWriteMode):
            return value
        if isinstance(value, str):
            return StateWriteMode(value.strip().lower())
        raise TypeError(f"Write mode must be a StateWriteMode or its name, got {type(value).__name__}")

    @staticmethod
    def _infer_per_timestep_flag(
        explicit_flag: Any,
        basis_value: Optional[str],
    ) -> bool:
        """
        Resolve per_timestep with backward compatibility for basis semantics.

        Explicit ``per_timestep`` has priority. If absent, infer from basis when
        provided explicitly:
        - per_step -> True
        - per_second -> False
        """
        if explicit_flag is not None:
            return bool(explicit_flag)
        if isinstance(basis_value, str):
            basis_norm = basis_value.strip().lower()
            if basis_norm == "per_step":
                return True
            if basis_norm == "per_second":
                return False
        return False

    def _resolve_write_mode(self, name: str, item: UpdateItem) -> StateWriteMode:
        """Resolve write semantics for a state variable update."""
        if item.write_mode is not None:
            return self._coerce_write_mode(item.write_mode)
        run_specs = self._run_specs
        if run_specs is not None and hasattr(run_specs, "resolve_semantic_mode"):
            resolved = run_specs.resolve_semantic_mode(name, None)
            return self._coerce_write_mode(resolved)
        spec = self._schema.get(name)
        if spec is not None and getattr(spec, "write_mode", None) is not None:
            return self._coerce_write_mode(spec.write_mode)
        return StateWriteMode.MUTABLE

    def _record_state_diagnostic(self, name: str, value: Any) -> None:
        """Record derived diagnostics without mutating state."""
        if not self._diagnostics_enabled:
            return
        numeric = self._value_to_float(value)
        if numeric is None:
            return
        self._current_diagnostics[name] = numeric

    def apply_increment(
        self,
        var_name: str,
        increment: UpdateItem,
        dt: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        """Delegate a single declared variable update to the batch commit owner."""
        if increment is None:
            raise ValueError("Increment is required.")
        item = self._coerce_update_item(increment)
        if item is None:
            raise ValueError("Increment is required.")
        spec = self._schema.get(var_name)
        if spec is None:
            raise KeyError(f"Unknown state variable: {var_name}")

        update_sources = {var_name: item.source_functor} if item.source_functor else None
        result = self.apply_updates(
            {var_name: item},
            dt=dt,
            check_conservation=True,
            rollback_on_failure=False,
            update_sources=update_sources,
        )
        if isinstance(result, dict) and var_name in result:
            return result[var_name]
        return self._state_variables.get(var_name, torch.tensor(0.0, device=self.device))

    @staticmethod
    def _update_target_name(name: str) -> str:
        """Return the existing commit target of an update name.

        The delta_ prefix denotes an increment of the suffix variable. SCM
        alias permissions and units remain governed by apply_updates; this
        naming operation does not make an alias writable or advance state.
        """
        return name[len("delta_"):] if name.startswith("delta_") else name

    def apply_updates(
        self,
        updates: Dict[str, Any],
        dt: Optional[Union[float, Tensor]] = None,
        check_conservation: bool = True,
        rollback_on_failure: bool = False,
        update_sources: Optional[Dict[str, str]] = None,
        update_increments: Optional[Dict[str, Dict[str, Any]]] = None,
        commit_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        """Normalize units and time basis once, then commit actual state changes.

        REPLACE carries an absolute value; ACCUMULATE and FLUX carry
        differences. Explicit SCM boundary rates retain their declared units
        and rate semantics. The conservation ledger observes the realized
        storage changes, including finite-precision and clipping effects.

        ``commit_context`` is the optional provenance context of this commit
        (validate_commit_context), for example the step, composition, stage,
        role, interval start and interval of a simulator group commit. It is
        validated before any state work. When the commit emits provenance
        events, each event stores a copy under ``commit_context`` together
        with ``conservation_checked`` (whether this commit ran its ledger
        check) and ``element_net_change`` (the signed net change of the
        element ledger of this commit, ``_commit_element_net_change``). None
        (the default) leaves every event unchanged.
        """
        frozen_commit_context = validate_commit_context(commit_context)
        # References: LeVeque (2002), finite-volume time integration.
        # Mathematical expression: dt > 0 when an interval is supplied.
        # Parameter meanings: dt is the interval used for rates and budgets.
        # Units: seconds.
        # Domain/constraints: finite real scalar; None supplies no interval.
        # Tensor intervals keep their graph for the integral; audit metadata
        # uses the same detached scalar value, never a rounded float32 copy.
        # Sign convention: positive dt advances the integration interval.
        integration_interval = dt
        if dt is not None:
            if isinstance(dt, Tensor):
                if dt.numel() != 1 or dt.dtype == torch.bool or dt.is_complex():
                    raise TypeError("State update time step must be one real scalar interval")
                integration_interval = dt.reshape(())
                dt = float(dt.detach().item())
            elif isinstance(dt, bool) or not isinstance(dt, Real):
                raise TypeError("State update time step must be a real interval")
            else:
                dt = float(dt)
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("State update time step must be finite and positive")
        if not self._provenance_transaction_active and self._pending_provenance_events:
            self.commit_provenance_transaction()
        _, diagnostics, intents = _state_types.partition_outputs(updates)
        if intents:
            raise TypeError("ExternalIntent cannot be submitted to StateManager")
        contract_registry = getattr(self, "_contract_registry", None)
        prepared_updates: Dict[str, UpdateItem] = {}
        canonical_event_units: Dict[str, Optional[str]] = {}
        for name, update in updates.items():
            item = self._coerce_update_item(update)
            if item is None:
                raise TypeError(f"Unsupported state payload for '{name}'")
            if name in diagnostics and item.write_mode != StateWriteMode.DERIVED:
                item = replace(item, write_mode=StateWriteMode.DERIVED)
            resolved_write_mode = self._resolve_write_mode(name, item)
            if (
                resolved_write_mode != StateWriteMode.DERIVED
                and not isinstance(item.value, (Tensor, float, int))
            ):
                raise TypeError(f"Unsupported state payload for '{name}': {type(item.value).__name__}")
            if resolved_write_mode == StateWriteMode.DERIVED:
                prepared_updates[name] = item
                continue
            target_name = self._update_target_name(name)
            is_delta = target_name != name
            original_mode = item.mode
            if is_delta or original_mode == UpdateMode.FLUX:
                item = replace(item, mode=UpdateMode.ACCUMULATE)
            item = self._apply_run_specs(target_name, item)
            spec = self._schema.get(target_name)
            contract = contract_registry.get_contract(target_name) if contract_registry is not None else None
            is_boundary_rate = (
                contract is not None and str(contract.kind).lower() == "flux"
                and not item.per_timestep and original_mode != UpdateMode.FLUX and not is_delta
            )
            basis = (item.basis or "per_step").strip().lower()
            if basis not in {"per_step", "per_second", "instantaneous"}:
                raise ValueError(f"Unsupported increment basis: {item.basis}")
            if basis == "instantaneous" and item.mode != UpdateMode.REPLACE and not is_boundary_rate:
                raise ValueError("instantaneous basis requires a replacement or declared boundary-rate aggregate")

            # References: LeVeque (2002), finite-volume time integration;
            # BIPM SI Brochure (9th ed.), affine temperature scales.
            # Mathematical expression: delta = q * dt or an integrated amount;
            # affine value T(v)=a*v+b gives difference delta'=a*delta.
            # Parameter meanings: q is a rate; a,b come from the unit owner.
            # Units: target schema units; dt is in seconds.
            # Domain/constraints: explicit integrated values are never scaled
            # again; SCM boundary rates keep their exact declared unit contract.
            # Sign convention: positive increments increase the target storage.
            integrate = not item.per_timestep and not is_boundary_rate and (
                original_mode == UpdateMode.FLUX
                or (item.mode == UpdateMode.ACCUMULATE and basis == "per_second")
                or (is_delta and dt is not None)
            )
            # The interval an unintegrated rate is integrated over; None when the
            # value is not integrated here.
            rate_interval: Optional[Union[float, Tensor]] = None
            if integrate:
                if integration_interval is None:
                    raise ValueError(f"dt is required for an unintegrated state rate: {target_name}")
                rate_interval = integration_interval
            value = item.value
            unit_in = item.unit
            unit_target = getattr(spec, "units", None) or getattr(contract, "udunits", None)
            if unit_in and unit_target and unit_in != unit_target:
                if contract is not None and str(contract.kind).lower() == "flux":
                    raise ValueError(
                        f"Boundary budget unit mismatch for '{target_name}': "
                        f"expected '{contract.udunits}', got '{unit_in}'"
                    )
                from ...data.standards.unit_converter import ConversionContext, unit_converter

                variable_name = getattr(spec, "cf_standard_name", None) or target_name
                conversion_target = unit_target
                if not unit_converter.can_convert(unit_in, conversion_target, variable_name):
                    # References: BIPM SI Brochure (9th ed.), Section 2.3.3, the unit one;
                    #   LeVeque (2002), rate-times-interval integration.
                    # Mathematical expression: [q] = [x] s-1; for [x] = 1 this is [q] = s-1.
                    # Parameter meanings: unit_target is the storage unit of the target state,
                    #   rate_target the per-second tendency label an ACCUMULATE rate may carry
                    #   (the group owner may already have integrated the value while keeping
                    #   the originating per-second label).
                    # Units: labels only; "dimensionless" and "1" both denote the unit one, so
                    #   their tendency label is "s-1", not "dimensionless s-1".
                    # Domain/constraints: only ACCUMULATE items integrated here or by their owner
                    #   may carry the rate label; any other label difference is a mismatch.
                    # Sign convention: none.
                    rate_target = (
                        "s-1" if unit_target in {"dimensionless", "1"} else f"{unit_target} s-1"
                    )
                    if item.mode == UpdateMode.ACCUMULATE and (integrate or item.per_timestep) and (
                        unit_in == rate_target
                        or unit_converter.can_convert(unit_in, rate_target, variable_name)
                    ):
                        conversion_target = rate_target
                    else:
                        raise ValueError(f"Unit mismatch for {target_name}: {unit_in} -> {unit_target}")
                if unit_in == conversion_target:
                    # Same label on both sides: the value already carries the resolved unit,
                    # so no numeric conversion is performed (an identity conversion would
                    # otherwise demand an explicit allow_identity context).
                    pass
                elif item.mode == UpdateMode.ACCUMULATE and unit_converter._is_temperature_affine_pair(
                    unit_in, conversion_target
                ):
                    offset = unit_converter.convert(0.0, variable_name, unit_in, conversion_target)
                    scale = unit_converter.convert(1.0, variable_name, unit_in, conversion_target) - offset
                    value = value * scale
                else:
                    # References: RELIC unit SSOT (data/standards/unit_converter.py alias table).
                    # Mathematical expression: v_target = f(v_in) for the registered pair; for a
                    #   registered alias pair (e.g. "1" <-> "dimensionless") f is the identity.
                    # Parameter meanings: unit_in is the writer's label, conversion_target the
                    #   resolved storage or tendency label; convertibility was established by
                    #   can_convert above, so an alias identity is accepted explicitly here.
                    # Units: labels only.
                    # Domain/constraints: unregistered pairs never reach this branch; the
                    #   explicit allow_identity applies only to pairs the table maps to identity.
                    # Sign convention: none.
                    value = unit_converter.convert(
                        value,
                        variable_name,
                        unit_in,
                        conversion_target,
                        context=ConversionContext(allow_identity=True),
                    )
                item = replace(item, unit=unit_target)
            if rate_interval is not None:
                value_interval = rate_interval
                if isinstance(value, Tensor):
                    integration_dtype = value.dtype
                    destination = self._state_variables.get(target_name)
                    if isinstance(destination, Tensor):
                        integration_dtype = torch.promote_types(integration_dtype, destination.dtype)
                    if isinstance(value_interval, Tensor):
                        integration_dtype = torch.promote_types(integration_dtype, value_interval.dtype)
                    elif not integration_dtype.is_floating_point and not integration_dtype.is_complex:
                        integration_dtype = torch.float64
                    value = value.to(dtype=integration_dtype)
                    if isinstance(value_interval, Tensor):
                        value_interval = value_interval.to(device=value.device, dtype=integration_dtype)
                value = value * value_interval
                item = replace(item, basis="per_step", per_timestep=True)
            prepared_updates[name] = replace(item, value=value)
            canonical_event_units[target_name] = unit_target
        check_conservation = bool(check_conservation and getattr(self, "_conservation_enabled", True))
        rollback_on_failure = bool(rollback_on_failure or getattr(self, "_conservation_rollback_default", False))
        snapshot = {k: v.clone() if isinstance(v, Tensor) else v for k, v in self._state_variables.items()}
        new_state = {k: v.clone() if isinstance(v, Tensor) else v for k, v in self._state_variables.items()}
        step_index = self._update_counter
        normalized_updates: Dict[str, Any] = {}
        clip_loss_record: Dict[str, float] = {}
        clip_loss_updates: Dict[str, Any] = {}
        element_hints: Dict[str, Optional[str]] = {}
        notification_sources: Dict[str, Optional[str]] = {}

        def _tag_conservation_delta(
            value: Tensor,
            element: Optional[str],
            source_functor: Optional[str],
            *,
            unit: Optional[str] = None,
            mode: UpdateMode = UpdateMode.ACCUMULATE,
            per_timestep: bool = True,
            force_item: bool = False,
        ) -> Any:
            if element is None and not force_item:
                return value
            return UpdateItem(
                value=value,
                mode=mode,
                unit=unit,
                basis="per_step" if per_timestep else "per_second",
                per_timestep=per_timestep,
                element=element,
                source_functor=source_functor,
            )

        physical_constraint_manager = self._get_physical_constraint_manager()
        source_map = update_sources or {}
        for name, upd in updates.items():
            # Units and time integration were resolved in the batch preflight.
            target_name = self._update_target_name(name)
            item = prepared_updates[name]
            if item.element is not None and isinstance(target_name, str):
                element_label = _state_conservation.normalize_element_label(
                    item.element
                )
                is_declared, declared_element = (
                    _state_conservation.resolve_declared_element_domain(
                        self,
                        target_name,
                    )
                )
                if is_declared and element_label != declared_element:
                    raise ValueError(
                        f"UpdateItem element '{element_label}' conflicts with "
                        f"declared element '{declared_element}' for '{target_name}'"
                    )
                element_hints[target_name] = element_label
            if contract_registry is not None and isinstance(target_name, str):
                if contract_registry.is_alias(target_name):
                    canonical = contract_registry.resolve_alias(target_name)
                    raise PermissionError(
                        f"Cannot write to alias '{target_name}'. Use '{canonical}'."
                    )
                source_name = source_map.get(target_name)
                increment_sources: List[str] = []
                if update_increments is not None:
                    raw_increment_map = update_increments.get(target_name)
                    if isinstance(raw_increment_map, dict):
                        increment_sources = [
                            name
                            for name in raw_increment_map.keys()
                            if isinstance(name, str) and name
                        ]
                if item.mode == UpdateMode.ACCUMULATE and increment_sources:
                    if hasattr(contract_registry, "is_incrementable") and not contract_registry.is_incrementable(target_name):
                        raise PermissionError(
                            f"Variable '{target_name}' does not allow ACCUMULATE updates "
                            "under ContractRegistry.incrementable policy."
                        )
                    unauthorized_increment_writers = [
                        name
                        for name in increment_sources
                        if hasattr(contract_registry, "can_increment")
                        and not contract_registry.can_increment(target_name, name)
                    ]
                    if unauthorized_increment_writers:
                        owner = contract_registry.get_owner(target_name)
                        raise PermissionError(
                            "Unauthorized increment writers for "
                            f"'{target_name}': {unauthorized_increment_writers}. Owner: {owner}"
                        )
                    if source_name and hasattr(contract_registry, "can_increment"):
                        if not contract_registry.can_increment(target_name, source_name):
                            owner = contract_registry.get_owner(target_name)
                            raise PermissionError(
                                f"Functor '{source_name}' is not authorized as final "
                                f"increment writer for '{target_name}'. Owner: {owner}"
                            )
                elif source_name and not contract_registry.can_write(target_name, source_name):
                    owner = contract_registry.get_owner(target_name)
                    message = (
                        f"Functor '{source_name}' is not authorized to write '{target_name}'. "
                        f"Owner: {owner}"
                    )
                    if bool(getattr(self._run_specs, "allow_unauthorized_contract_writes", False)):
                        logger.warning("%s; allowing update due to explicit override.", message)
                    else:
                        raise PermissionError(message)
            effective_source = source_map.get(target_name) or item.source_functor
            write_mode = self._resolve_write_mode(target_name, item)
            if write_mode == StateWriteMode.DERIVED:
                # Derived updates are diagnostics only.
                self._record_state_diagnostic(target_name, item.value)
                continue
            if write_mode == StateWriteMode.IMMUTABLE:
                message = f"Immutable state variable '{target_name}' cannot be updated"
                if getattr(self._run_specs, "strict", False):
                    raise ValueError(message)
                logger.warning(message)
                continue
            raw_value = item.value
            if not isinstance(raw_value, (Tensor, float, int)):
                raise TypeError(f"Unsupported state payload for '{target_name}': {type(raw_value).__name__}")
            base = new_state.get(target_name)
            delta = (
                raw_value.to(device=self.device)
                if isinstance(raw_value, Tensor)
                else torch.tensor(float(raw_value), device=self.device,
                                  dtype=base.dtype if isinstance(base, Tensor) else None)
            )
            base_tensor: Optional[Tensor] = None
            if isinstance(base, Tensor):
                base_tensor = base
            elif base is not None:
                base_tensor = torch.as_tensor(base, dtype=delta.dtype, device=delta.device)
            if isinstance(base_tensor, Tensor) and isinstance(delta, Tensor):
                delta = delta.to(device=base_tensor.device, dtype=base_tensor.dtype)

            contract = (
                contract_registry.get_contract(target_name)
                if contract_registry is not None
                else None
            )
            is_instantaneous_flux_aggregate = (
                item.mode == UpdateMode.ACCUMULATE
                and not bool(item.per_timestep)
                and contract is not None
                and str(contract.kind).lower() == "flux"
            )

            if item.mode == UpdateMode.REPLACE or is_instantaneous_flux_aggregate:
                new_state[target_name] = delta
                if contract is not None and str(contract.kind).lower() == "flux":
                    normalized_updates[target_name] = _tag_conservation_delta(
                        delta,
                        element_hints.get(target_name),
                        effective_source,
                        unit=item.unit,
                        mode=UpdateMode.REPLACE,
                        per_timestep=bool(item.per_timestep),
                        force_item=True,
                    )
                else:
                    if base_tensor is not None:
                        audit_dtype = torch.promote_types(delta.dtype, torch.float64)
                        conservation_delta = delta.to(dtype=audit_dtype) - base_tensor.to(dtype=audit_dtype)
                    else:
                        conservation_delta = delta
                    normalized_updates[target_name] = _tag_conservation_delta(
                        conservation_delta,
                        element_hints.get(target_name),
                        effective_source,
                        unit=item.unit,
                    )
            else:
                if base_tensor is not None:
                    if base_tensor.shape != delta.shape:
                        raise ValueError(
                            f"ACCUMULATE shape contract violation for '{target_name}': "
                            f"base shape {tuple(base_tensor.shape)} vs delta shape {tuple(delta.shape)}"
                        )
                    updated_value = base_tensor + delta
                else:
                    updated_value = delta
                new_state[target_name] = updated_value
                # References: LeVeque (2002), discrete finite-volume balance.
                # Mathematical expression: realized_delta = stored_new - stored_old.
                # Parameter meanings: stored values include arithmetic roundoff.
                # Units: target variable units; normalized by the existing ledger.
                # Domain/constraints: promote before subtraction, not only before
                # summation; requested transfers cannot hide lost updates.
                # Sign convention: positive realized delta increases storage.
                audit_dtype = torch.promote_types(updated_value.dtype, torch.float64)
                realized_delta = (
                    updated_value.to(dtype=audit_dtype) - base_tensor.to(dtype=audit_dtype)
                    if base_tensor is not None else updated_value
                )
                normalized_updates[target_name] = _tag_conservation_delta(
                    realized_delta,
                    element_hints.get(target_name),
                    effective_source,
                    unit=item.unit,
                )
            notification_sources[target_name] = effective_source

        # References: Beven 2012; Dingman 2015
        # Mathematical expression:
        #   If x < L: clip_loss = x - L, x = L
        #   If x > U: clip_loss = x - U, x = U
        # Parameter meanings: x current value, L lower bound, U upper bound.
        # Units: the target variable's canonical units.
        # Domain/constraints: declared bounds only; a one-sided declaration has
        # L = -inf or U = +inf, so only the declared side can clip. Audit
        # differences use promoted precision without changing the stored dtype.
        # Sign convention: positive clip_loss removes storage; negative values
        # disclose additions. A diagnostic residual does not close the budget.
        for var_name, value in list(new_state.items()):
            spec = self._schema.get(var_name)
            # Enforcement is decided only by the declared bounds of the spec;
            # variable names never exempt a declared bound.
            if spec is None or spec.bounds is None:
                continue
            if not isinstance(value, Tensor):
                continue
            lower, upper = spec.bounds
            if lower is None and upper is None:
                continue
            if physical_constraint_manager is None:
                raise RuntimeError(
                    "StateManager requires the canonical PhysicalConstraintManager "
                    f"to enforce bounds for '{var_name}'."
                )
            constrained = physical_constraint_manager.apply_constraints(
                value,
                bounds=(lower, upper),
                variable_name=var_name,
            )
            if torch.equal(constrained, value):
                continue
            audit_dtype = torch.promote_types(value.dtype, torch.float64)
            clip_delta = value.to(dtype=audit_dtype) - constrained.to(dtype=audit_dtype)
            clip_loss_value = float(clip_delta.detach().sum().cpu().item())
            clip_loss_record[var_name] = clip_loss_value
            new_state[var_name] = constrained

            base_value = snapshot.get(var_name)
            if isinstance(base_value, Tensor):
                base_tensor = base_value.to(device=constrained.device, dtype=constrained.dtype)
            elif base_value is None:
                base_tensor = torch.zeros_like(constrained)
            else:
                base_tensor = torch.as_tensor(
                    base_value, device=constrained.device, dtype=constrained.dtype
                )
            actual_delta = constrained.to(dtype=audit_dtype) - base_tensor.to(dtype=audit_dtype)
            normalized_updates[var_name] = _tag_conservation_delta(
                actual_delta,
                element_hints.get(var_name),
                None,
            )

            element_label = element_hints.get(var_name)
            if element_label is None:
                name_lower = var_name.lower()
                for candidate in ("carbon", "nitrogen", "phosphorus", "water"):
                    if self._matches_element_keyword(name_lower, candidate):
                        element_label = candidate
                        break
            if element_label:
                clip_key = f"clip_loss_{element_label}_flux"
            else:
                clip_key = f"clip_loss_{var_name}_flux"
            if clip_key in normalized_updates:
                clip_key = f"{clip_key}_{len(clip_loss_updates)}"
            clip_loss_updates[clip_key] = clip_delta
            if self._diagnostics_enabled:
                self._current_diagnostics[f"clip_loss.{var_name}"] = clip_loss_value

        if clip_loss_updates:
            normalized_updates.update(clip_loss_updates)
        if clip_loss_record:
            self._clip_loss_diagnostics = clip_loss_record
        else:
            self._clip_loss_diagnostics = {}

        check_updates = normalized_updates
        if clip_loss_updates:
            check_updates = {
                key: value
                for key, value in normalized_updates.items()
                if not key.startswith("clip_loss_")
            }

        strict_enabled = feature_manager.is_enabled('strict_conservation_enabled')

        # Check conservation BEFORE applying state changes
        if check_conservation:
            # Compute dynamic tolerance if dt is provided
            if dt is not None:
                # Dynamic tolerance: tol = dt * scale_factor
                # Default scale_factor is the base absolute tolerance
                scale_factor = self._abs_conservation_tol
                dynamic_tol = dt * scale_factor
            else:
                # Use fixed tolerance when dt not provided
                scale_factor = None
                dynamic_tol = self._abs_conservation_tol
            
            violations = self._check_conservation(check_updates, dt=dt, scale_factor=scale_factor)
            for element, total in violations.items():
                if abs(total) > dynamic_tol:
                    pool_set = {name.lower() for name in self.element_pools.get(element, [])}
                    contributors: List[Tuple[str, float]] = []
                    for var_name, raw_value in check_updates.items():
                        element_hint = _state_conservation.normalize_element_label(
                            raw_value.element
                            if isinstance(raw_value, UpdateItem)
                            else None
                        )
                        if not _state_conservation.matches_element_membership(
                            self,
                            var_name,
                            element,
                            pool_set,
                            element_hint,
                        ):
                            continue
                        value_float = self._value_to_float(raw_value)
                        if value_float is None:
                            continue
                        contributors.append((var_name, float(value_float)))
                    contributors.sort(key=lambda pair: abs(pair[1]), reverse=True)
                    top_contrib = contributors[:8]
                    contributors_text = ""
                    if top_contrib:
                        parts = [f"{name}={value:.6f}" for name, value in top_contrib]
                        contributors_text = " contributors=[" + ", ".join(parts) + "]"
                    violation_message = (
                        f"Conservation violation for {element}: net change = {total:.6f} "
                        f"(tolerance = {dynamic_tol:.6e}, dt = {dt}, "
                        f"{'dynamic' if dt else 'fixed'})"
                        f"{contributors_text}"
                    )
                    if not getattr(self, "_conservation_raise_on_violation", True):
                        logger.warning(violation_message)
                        continue
                    self._state_variables = snapshot
                    if rollback_on_failure:
                        # explicit semantic retention; state already restored
                        pass
                    raise ValueError(violation_message)
            if strict_enabled:
                strict_ok = self.check_strict_conservation(
                    check_updates, dt=dt, store_history=False
                )
                if not strict_ok:
                    self._state_variables = snapshot
                    raise ValueError("Strict conservation violation detected")

        errs = self.validate_state(new_state)
        if errs:
            self._state_variables = snapshot
            raise ValueError(f"State validation failed: {errs}")

        # Freeze the entire notification batch before publishing state. A
        # failed observer must not prevent later variable events from existing.
        pending_events: List[Dict[str, Any]] = []
        if self._provenance_recorder is not None:
            context_getter = getattr(self._provenance_recorder, "get_notification_context", None)
            delivery_context = (
                {"notification_context": context_getter()} if callable(context_getter) else {}
            )
            if frozen_commit_context is not None:
                # References: W3C PROV-DM (2013); design 20261019_cr_a04_design_01, D7.
                # Mathematical expression: the commit's element ledger R_E (see
                #   _commit_element_net_change), evaluated once per commit.
                # Parameter meanings: check_updates are the realized changes and
                #   boundary rates of this commit without clip losses, the input of
                #   the commit-time check; dt is the commit interval.
                # Units: g E m-2 (C, N, P), kg m-2 (water).
                # Domain/constraints: an observation only; each event receives its
                #   own copy through the clone below.
                # Sign convention: positive R_E is an unexplained storage gain.
                delivery_context["commit_context"] = {
                    **frozen_commit_context,
                    "conservation_checked": check_conservation,
                    **self._commit_element_net_change(check_updates, dt),
                }
            for var_name in normalized_updates:
                if var_name not in new_state:
                    continue
                pre_value = snapshot.get(var_name, torch.tensor(0.0, device=self.device))
                post_value = new_state.get(var_name, pre_value)
                increments = (update_increments or {}).get(var_name, {})
                source = notification_sources.get(var_name)
                if not increments and source:
                    # References: W3C PROV-DM (2013), derivation and generation.
                    # Mathematical expression: observed_delta = post - pre.
                    # Parameter meanings: pre/post are accepted state values.
                    # Units: the StateManager variable's canonical units.
                    # Domain/constraints: attribute only an explicit source;
                    # promote accepted values before subtracting so attribution
                    # retains the same realized changes as the budget audit.
                    # Sign convention: positive means an increase in the value.
                    audit_dtype = torch.promote_types(post_value.dtype, torch.float64)
                    increments = {source: post_value.to(dtype=audit_dtype) - pre_value.to(dtype=audit_dtype)}
                # References: W3C PROV-DM (2013); canonical unit preparation above.
                # Mathematical expression: unit_event(v) = resolved unit_target(v);
                #   an accepted target with no declared unit remains unknown.
                # Parameter meanings: v is this event's accepted canonical target,
                #   not the last variable from a different preparation iteration.
                # Units: the exact schema/SCM label already checked during conversion.
                # Domain/constraints: no caller-unit inference, new registry or
                #   numerical change; the local map belongs to this one commit.
                # Sign convention: event metadata does not change amount signs.
                unit = canonical_event_units.get(var_name) or "unknown"
                pending_events.append(self._clone_transaction_value(
                    {
                        "step": step_index,
                        "variable": var_name,
                        "pre_value": pre_value,
                        "post_value": post_value,
                        "functor_increments": increments,
                        "unit": unit,
                        **delivery_context,
                    }
                ))

        # Publish once, then finish internal reports before external observers.
        self._state_variables = new_state
        self._update_counter += 1
        self._pending_provenance_events.extend(pending_events)

        # Generate conservation report if enabled
        if check_conservation:
            try:
                if self._enable_auto_report:
                    self.generate_conservation_report(normalized_updates, time.time(), dt=dt)
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.warning("Conservation report generation failed: %s", exc)
        if not self._provenance_transaction_active and self._pending_provenance_events:
            self.commit_provenance_transaction()
        return self.get_state()

    def _commit_element_net_change(
        self,
        updates: Dict[str, Any],
        dt: Optional[float],
    ) -> Dict[str, Any]:
        """Signed element-ledger net change of one commit, for its provenance events.

        References: LeVeque (2002), discrete finite-volume balance; E3SM/ELM CNP
            budget diagnostics (conservation._normalize_domain_value); design
            20261019_cr_a04_design_01, D7 (per-commit ledger residuals recorded).
        Mathematical expression: R_E = sum_i Delta S_E,i + dt (sum F_E,out -
            sum F_E,in) over the ledger members of element E among ``updates``;
            this is the net change of check_element_conservation (the
            conservation report of this commit), kept signed instead of its
            absolute balance_error. Elements without a member are absent.
        Parameter meanings: updates are the realized storage changes and boundary
            rates of the commit without clip losses (the input of the commit-time
            check); dt is the commit interval (None without an interval).
        Units: g E m-2 for carbon, nitrogen and phosphorus; kg m-2 for water.
        Domain/constraints: an observation, never a gate. When the ledger cannot
            be evaluated (RuntimeError, ValueError or TypeError of its declared
            contract, unit or interval checks) the commit stands and the context
            records element_net_change None with element_net_change_error, as the
            conservation report of apply_updates is non-blocking.
        Sign convention: positive R_E is a gain of element storage that the
            committed boundary rates do not explain.
        """
        net_changes: Dict[str, float] = {}
        try:
            for element in _state_conservation._TRACKED_CONSERVATION_ELEMENTS:
                tracked, net_change = _state_conservation._collect_element_net_change(
                    self,
                    updates,
                    element=element,
                    pools=self.element_pools.get(element, []),
                    dt=dt,
                )
                if tracked and net_change is not None:
                    net_changes[element] = float(net_change.detach().cpu().item())
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Commit element ledger not evaluated for provenance: %s", exc)
            return {
                "element_net_change": None,
                "element_net_change_error": f"{type(exc).__name__}: {exc}",
            }
        return {"element_net_change": net_changes}

    def get_diagnostics(self) -> Dict[str, Any]:
        """Retrieve diagnostic information including clip_loss records.

        ``current`` carries the derived diagnostics recorded since the last
        conservation report (reserved ``diagnostics.*`` names and DERIVED
        outputs); they are never prognostic state.
        """
        return {
            "clip_loss": getattr(self, "_clip_loss_diagnostics", {}),
            "update_counter": self._update_counter,
            "conservation_enabled": getattr(self, "_conservation_enabled", True),
            "current": dict(self._current_diagnostics),
        }

    @staticmethod
    def _coerce_update_item(update: Any) -> Optional[UpdateItem]:
        """Normalize raw update payload into an UpdateItem."""
        if isinstance(update, UpdateItem):
            return update
        if isinstance(update, tuple) and len(update) >= 2:
            mode_raw = update[1]
            write_raw = update[2] if len(update) >= 3 else None
            unit_raw = update[3] if len(update) >= 4 else None
            basis_raw = update[4] if len(update) >= 5 else None
            element_raw = update[5] if len(update) >= 6 else None
            source_raw = update[6] if len(update) >= 7 else None
            per_timestep_raw = update[7] if len(update) >= 8 else None
            write_mode = None
            # Legacy boundary form: the second field is an UpdateMode, or else a
            # write mode with ACCUMULATE (ValueError = not an UpdateMode name);
            # a value that is neither raises from _coerce_write_mode.
            try:
                mode = mode_raw if isinstance(mode_raw, UpdateMode) else UpdateMode(str(mode_raw))
            except ValueError:
                mode = UpdateMode.ACCUMULATE
                write_mode = StateManager._coerce_write_mode(mode_raw)
            if write_raw is not None:
                write_mode = StateManager._coerce_write_mode(write_raw)
            per_timestep = StateManager._infer_per_timestep_flag(per_timestep_raw, basis_raw)
            return UpdateItem(
                value=update[0],
                mode=mode,
                write_mode=write_mode,
                unit=unit_raw,
                basis=basis_raw or "per_step",
                per_timestep=per_timestep,
                element=element_raw,
                source_functor=source_raw,
            )
        if isinstance(update, dict) and "value" in update:
            mode_raw = update.get("mode", UpdateMode.ACCUMULATE)
            write_raw = update.get("write_mode") or update.get("semantic_mode")
            unit_raw = update.get("unit") or update.get("units")
            basis_raw = update.get("basis") or update.get("rate_basis")
            element_raw = update.get("element") or update.get("element_tag")
            source_raw = update.get("source_functor") or update.get("source")
            per_timestep_raw = update.get("per_timestep")
            write_mode = None
            # Legacy boundary form: the second field is an UpdateMode, or else a
            # write mode with ACCUMULATE (ValueError = not an UpdateMode name);
            # a value that is neither raises from _coerce_write_mode.
            try:
                mode = mode_raw if isinstance(mode_raw, UpdateMode) else UpdateMode(str(mode_raw))
            except ValueError:
                mode = UpdateMode.ACCUMULATE
                write_mode = StateManager._coerce_write_mode(mode_raw)
            if write_raw is not None:
                write_mode = StateManager._coerce_write_mode(write_raw)
            per_timestep = StateManager._infer_per_timestep_flag(per_timestep_raw, basis_raw)
            return UpdateItem(
                value=update.get("value"),
                mode=mode,
                write_mode=write_mode,
                unit=unit_raw,
                basis=basis_raw or "per_step",
                per_timestep=per_timestep,
                element=element_raw,
                source_functor=source_raw,
            )
        return UpdateItem(
            value=update,
            mode=UpdateMode.ACCUMULATE,
            basis="per_step",
            per_timestep=False,
        )

    def ingest_module_diagnostics(self, module_name: str, diagnostics: Dict[str, Any]) -> None:
        """Collect diagnostics from a module to be reported per step.

        Stored diagnostics are real scalars: a real number, or a real tensor
        stored as its mean (the fraction of True for a boolean mask). Any other
        value (a string, None, a sequence, a mapping, a complex tensor) raises
        TypeError naming the key, and none of the call's diagnostics is stored.
        """
        if not diagnostics or not self._diagnostics_enabled:
            return
        namespaced: Dict[str, float] = {}
        for k, v in diagnostics.items():
            key = f"{module_name}.{k}"
            if isinstance(v, Tensor):
                if v.is_complex():
                    raise TypeError(f"Diagnostic {key} is complex; stored diagnostics are real scalars")
                namespaced[key] = float(v.detach().cpu().to(torch.float64).mean().item())
            elif isinstance(v, (int, float)):
                namespaced[key] = float(v)
            else:
                raise TypeError(
                    f"Diagnostic {key} is a {type(v).__name__}; stored diagnostics are real "
                    "numbers or real tensors"
                )
        self._current_diagnostics.update(namespaced)

    def generate_conservation_report(
        self,
        updates: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
        dt: Optional[float] = None,
        store_history: bool = True,
        **kwargs: Any,
    ) -> ConservationSummary:
        """Generate and optionally store a conservation summary for provided updates."""
        summary = cast(
            ConservationSummary,
            _state_conservation.generate_conservation_report(
                self,
                updates=updates,
                current_time=current_time,
                dt=dt,
                store_history=store_history,
                **kwargs,
            ),
        )
        if store_history:
            self._trim_history_ledgers()
        return summary



    def check_strict_conservation(
        self,
        updates: Dict[str, Any],
        dt: Optional[float] = None,
        store_history: bool = True,
    ) -> bool:
        """Evaluate strict conservation if the corresponding feature flag is enabled."""
        passed = cast(
            bool,
            _state_conservation.check_strict_conservation(
                self,
                updates,
                dt=dt,
                store_history=store_history,
            ),
        )
        if store_history:
            self._trim_history_ledgers()
        return passed



    def set_conservation_ledger(self, ledger: Any, pools: Optional[List[str]] = None) -> None:
        """Set explicit conservation ledger for element categories."""
        _state_conservation.set_conservation_ledger(self, ledger, pools=pools)



    def get_conservation_ledger(self) -> Dict[str, List[str]]:
        """Get current conservation ledger mapping (copy)."""
        return cast(Dict[str, List[str]], _state_conservation.get_conservation_ledger(self))



    def get_last_conservation_summary(self) -> Optional[ConservationSummary]:
        """Return the last stored conservation summary if any."""
        return cast(
            Optional[ConservationSummary],
            _state_conservation.get_last_conservation_summary(self),
        )



    @property
    def conservation_history(self) -> List[ConservationSummary]:
        """Read-only history of conservation summaries per application."""
        return list(self._conservation_history)

    def _validate_bounds(self, candidate: Dict[str, Tensor]) -> None:
        """Validate bounds defined in schema for candidate state."""
        for name, spec in self._schema.items():
            if spec.bounds is None or name not in candidate:
                continue
            v = candidate[name]
            lo, hi = spec.bounds
            if isinstance(v, Tensor):
                if torch.any(v < lo) or torch.any(v > hi):
                    raise ValueError(f"State variable {name} out of bounds for some elements")
            else:
                x = float(v)
                if x < lo or x > hi:
                    raise ValueError(f"State variable {name} out of bounds: {x} not in [{lo}, {hi}]")

    def _compute_conservation_groups(self, updates: Dict[str, Any]) -> Dict[str, float]:
        """Compute grouped update sums for conserved categories."""
        return cast(Dict[str, float], _state_conservation.compute_conservation_groups(self, updates))


__all__ = [
    "COMMIT_CONTEXT_RESERVED_KEYS",
    "CONSERVATION_CHECK_KEYS",
    "CONSERVATION_TOLERANCE_PARAMETER",
    "STATE_MANAGER_CONFIGURABLE_KEYS",
    "StateManager",
    "StateVariableSpec",
    "StateVariableType",
    "StateWriteMode",
    "UpdateMode",
    "UpdateItem",
    "ConservationSummary",
    "feature_manager",
    "VariableNameMapper",
    "validate_conservation_checks",
    "validate_commit_context",
    "validate_conservation_setting",
    "validate_max_history_length",
]

# Re-export canonical mapper symbols from the single source module.
from .. import variable_mapper as _variable_mapper

VariableNameMapper = _variable_mapper.VariableNameMapper
variable_name_mapper = _variable_mapper.variable_name_mapper

__all__.append("variable_name_mapper")
