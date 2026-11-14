"""
Unified Parameter Management System for RELIC

This module implements the authoritative parameter management system for RELIC,
providing hierarchical configuration loading, scientific validation, unit checking,
and seamless integration with PyTorch for learnable parameters.

The system serves as the single source of truth for all model parameters across
the entire RELIC ecosystem while maintaining a unified parameter interface.

Key Features:
    - Hierarchical parameter loading with precedence rules
    - Scientific constraint validation with physical bounds
    - Unit consistency checking using CF-1.8 conventions
    - PyTorch nn.Parameter integration for learnable parameters
    - Parameter database management (relic/calibrated)
    - Version tracking and provenance documentation
    - Device migration and precision control
    - Snapshot/diff capabilities for debugging

Architecture:
    - One process-wide manager, obtained only through
      relic.parameters.get_global_manager(); constructing
      UnifiedParameterManager(...) yields an independent instance that is
      injected explicitly where a separate owner is needed
    - Dot-notation parameter keys (e.g., 'biosphere.photosynthesis.vcmax25')
    - Two-tier database: relic/ (native), calibrated/
    - Integration with core/config for override capabilities
    - Event-driven parameter change notifications

Mathematical Foundation:
    Parameter space: Theta = {theta_1, theta_2, ..., theta} where theta in [a, b]
    Hierarchical merge: P_final = P_default  P_calibrated  P_config  P_runtime
    Constraint satisfaction: C(theta) =  (a <= theta <= b)  R(theta)
    where R(theta) represents inter-parameter relationship constraints

References:
    - Farquhar, G.D. et al. (1980). Biochemical model of photosynthetic CO2 assimilation
    - Guenther, A.B. et al. (2012). MEGAN2.1 model for biogenic volatile organic compounds
    - CF-1.8 Convention for Climate and Forecast Metadata
    - PyTorch Parameter and Module documentation
    - UDUNITS-2 unit specification standard
"""

import os
import warnings
import builtins
import threading
from copy import deepcopy
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from collections import defaultdict
import logging

from .lru_cache import ThreadSafeLRUCache

import torch
import torch.nn as nn
from torch import Tensor

from ..core.variable_mapper import variable_name_mapper

# Import shared types first
from .types import (
    ParameterCategory,
    ValidationLevel,
    BoundsPolicy,
    UnitValidationMode,
    ParameterValue,
    ParameterDict,
    MetadataDict,
    ValidationErrors,
    ChangeCallback,
)

from .audit import (
    ParameterAuditIssue,
    ParameterAuditReport,
)
from . import audit as _audit_ops

# Import new components
from .storage import ParameterStorage, ParameterSnapshot
from .validation import ParameterValidator
from . import events as _param_events
from .cache import ops as _cache_ops
from .ops import accessors as _accessors
from .ops.access_impl import get_impl as _access_impl
from .ops import type_utils as _type_utils
from .ops.exporter import (
    export_parameters as _export_parameters,
    export_namespace as _export_namespace,
)
from .ops.config_loader import (
    load_yaml_file as _load_yaml_file,
    load_omegaconf_config as _load_omegaconf_config,
    get_namespace_dict as _get_namespace_dict,
)
from .ops.key_normalization import normalize_parameter_key
from .ops.compare import values_differ as _values_differ
from .ops.namespaces import (
    list_parameters as _list_parameters,
    get_namespace_parameters as _get_namespace_parameters,
)
from .ops.spec_access import (
    ensure_parameter_spec as _ensure_parameter_spec,
    get_parameter as _get_parameter,
    exists as _exists,
)
from .ops.mutation import (
    set_parameter as _set_parameter,
    register_parameter as _register_parameter,
    register_parameters as _register_parameters,
    set_value as _set_value,
)
from .ops.nn_params import (
    register_nn_parameter as _register_nn_parameter,
    get_nn_parameters as _get_nn_parameters,
    get_trainable_parameters as _get_trainable_parameters,
    freeze as _freeze_parameters,
    unfreeze as _unfreeze_parameters,
    to_device as _to_device,
)
from .ops.unit_validation import (
    should_skip_unit_validation as _should_skip_unit_validation,
    validate_units_for_spec as _validate_units_for_spec,
    check_units_for_spec as _check_units_for_spec,
    convert_parameter_units as _convert_parameter_units,
    convert_parameter_units_impl as _convert_parameter_units_impl,
)
from .ops.validation_ops import validate_parameters as _validate_parameters
from .ops import bounds as _bounds_ops
from .spec import ParameterSpec
from .database_manager import ParameterDatabase
from .ops.database_ops import (
    load_from_database as _load_from_database,
    save_to_database as _save_to_database,
)
from .ops.snapshots import (
    snapshot as _snapshot_state,
    restore as _restore_state,
    on_storage_restore as _sync_storage_restore,
    diff_snapshots as _diff_snapshots,
    get_diff as _get_snapshot_diff,
    create_snapshot as _create_snapshot,
    restore_snapshot as _restore_snapshot,
    manager_to_dict as _manager_to_dict,
)
from .ops.bootstrap import (
    preload_parameters as _bootstrap_preload_parameters,
    build_runtime_registry as _build_runtime_registry,
    load_initial_parameters as _bootstrap_load_initial_parameters,
)
from .ops.lifecycle import (
    create_default_metadata as _create_default_metadata,
    do_initialize as _do_initialize,
    do_configure as _do_configure,
    do_validate as _do_validate,
    do_cleanup as _do_cleanup,
)
from .ops.metrics import (
    get_parameters_by_category as _get_parameters_by_category,
    get_statistics as _get_statistics,
    get_status as _get_status,
    debug_report_units as _debug_report_units,
)

# Import manager unification framework
from ..core.managers.base import BaseManager
from ..core.managers.metadata import (
    ManagerMetadata
)

logger = logging.getLogger(__name__)

def _format_expected_type(expected_type: Optional[Union[type, Tuple[type, ...]]]) -> str:
    """Format expected_type for cache keys and error messages."""
    return _type_utils.format_expected_type(expected_type)



class UnifiedParameterManager(BaseManager):
    """
    Unified Parameter Management System for RELIC
    
    The single, authoritative parameter management system that provides
    hierarchical configuration loading, scientific validation, unit checking,
    PyTorch integration, and comprehensive parameter lifecycle management.
    
    This class provides a consistent interface with name mapping for parameter access.
    
    Key Responsibilities:
        - Hierarchical parameter resolution with precedence rules
        - Scientific validation with physical constraint checking  
        - Unit consistency verification using CF-1.8 standards
        - PyTorch nn.Parameter registration for learnable parameters
        - Parameter database management across three tiers
        - Version tracking and provenance documentation
        - Event-driven change notifications
        - Device migration and precision control
        - Snapshot/restore capabilities for debugging
    
    Architecture Pattern:
        Facade + Observer + Strategy. Construction yields an independent
        instance; the only process-wide entry point is
        relic.parameters.get_global_manager() (explicit injection otherwise).
        
    Mathematical Foundation:
        Parameter space: Theta = {theta_1, theta_2, ..., theta}
        Hierarchical merge: P =  P where  is precedence-based merge
        Validation: V(theta) =  C(theta) where C are constraint functions
        Update notification: delta(theta) ->  in Observers: o.notify(theta, theta')
        
    References:
        - Gamma, E. et al. (1994). Design Patterns - Observer, Facade, Strategy
        - Fowler, M. (2002). Patterns of Enterprise Application Architecture
        - PyTorch Parameter and Module system documentation
        - CF-1.8 Convention for Climate and Forecast Metadata
    """
    
    def __init__(
        self,
        database_path: Optional[Path] = None,
        metadata: Optional[ManagerMetadata] = None,
        preload_all: bool = False,
        warmup_cache: bool = False,
        **kwargs,
    ):
        """Initialize one independent parameter manager.

        Construction always builds a new manager. The process-wide manager
        is obtained only through ``relic.parameters.get_global_manager()``;
        code that needs another owner constructs one and injects it
        explicitly.
        """

        # Use absolute path to database - resolve from relic package location
        if database_path is None:
            import relic
            relic_root = Path(relic.__file__).parent
            self.database_path = relic_root / "parameters" / "database"
        else:
            self.database_path = database_path
        
        # Core components (new modular design)
        self.storage = ParameterStorage()
        if hasattr(self.storage, "register_restore_callback"):
            self.storage.register_restore_callback(self._on_storage_restore)
        self.validator = ParameterValidator()
        self.database = ParameterDatabase(self.database_path, validator=self.validator)
        self.name_mapper = variable_name_mapper
        
        # Parameter storage (kept for continuity)
        self.parameters: Dict[str, ParameterSpec] = {}

        # External configuration cache (non-parameter YAML/OmegaConf)
        self._external_configs: Dict[str, Dict[str, Any]] = {}
        
        # PyTorch integration
        self.nn_parameters: Dict[str, nn.Parameter] = {}
        self.device = torch.device("cpu")

        # Change tracking and notifications
        self.change_callbacks: List[ChangeCallback] = []
        self.change_history: List[Dict[str, Any]] = []
        self._mutation_lock = threading.RLock()
        self._missing_requests: Dict[str, int] = defaultdict(int)
        self._access_history: Set[str] = builtins.set()  # Track accessed parameters
        self._clamp_events: List[Dict[str, Any]] = []
        self._bounds_policy: BoundsPolicy = BoundsPolicy.WARN_AND_CLAMP
        self._bounds_policy_explicit: bool = False
        env_enforce_bounds = os.getenv("RELIC_ENFORCE_BOUNDS_ERRORS", "0").lower()
        self.enforce_bounds_error: bool = env_enforce_bounds in {"1", "true", "yes", "on"}
        env_bounds_policy = os.getenv("RELIC_BOUNDS_POLICY")
        if env_bounds_policy:
            # An invalid policy is a configuration error, not a request for the default.
            try:
                self._bounds_policy = BoundsPolicy(env_bounds_policy.lower())
            except ValueError as err:
                raise ValueError(
                    f"Invalid RELIC_BOUNDS_POLICY '{env_bounds_policy}'; expected one of "
                    f"{[policy.value for policy in BoundsPolicy]}"
                ) from err
            self._bounds_policy_explicit = True
        
        # LRU cache for parameter queries and conversions
        cache_enabled = kwargs.get('cache_enabled', True)
        cache_max_size = kwargs.get('cache_max_size', 1000)
        cache_ttl = kwargs.get('cache_ttl', None)
        
        if cache_enabled:
            self._parameter_cache = ThreadSafeLRUCache(maxsize=cache_max_size, ttl=cache_ttl)
            self._unit_conversion_cache = ThreadSafeLRUCache(maxsize=500, ttl=cache_ttl)
        else:
            self._parameter_cache = None
            self._unit_conversion_cache = None
        
        self._cache_enabled = cache_enabled
        
        # Create metadata if not provided
        if metadata is None:
            metadata = self._create_default_metadata()
        
        # Initialize BaseManager
        super().__init__(metadata=metadata)
        
        # Validation settings  
        self.validation_level = ValidationLevel.STANDARD
        self.unit_validation_mode = UnitValidationMode.CHECK_ONLY
        self._unit_validation_issues: List[Dict[str, Any]] = []
        self._unit_validation_checked: Set[str] = builtins.set()
        env_unit_mode = os.getenv("RELIC_UNIT_VALIDATION_MODE")
        if env_unit_mode:
            normalized_mode = env_unit_mode.strip().lower().replace("_", "-")
            try:
                self.unit_validation_mode = UnitValidationMode(normalized_mode)
            except ValueError as err:
                raise ValueError(
                    f"Invalid RELIC_UNIT_VALIDATION_MODE '{env_unit_mode}'; expected one of "
                    f"{[mode.value for mode in UnitValidationMode]}"
                ) from err

        self._preloaded: bool = False
        self._preload_all_default = preload_all
        self._warmup_default = warmup_cache

        if preload_all:
            self._preload_parameters(warmup_cache=warmup_cache)

        # The process-wide variable name mapper is refreshed only by the owner of
        # the process-wide manager (ops.manager_registry.get_global_manager); an
        # independent manager, possibly over a partial database, never rewrites it.

        self._initialized = True
        logger.info("UnifiedParameterManager initialized successfully")

    def __len__(self) -> int:
        """Return the number of registered parameters."""
        if self.database is None:
            return 0
        # The database counts its own specifications; its failures propagate.
        return len(self.database)

    def _preload_parameters(self, warmup_cache: bool = True) -> None:
        """Load all parameters and optionally warm up caches."""
        return _bootstrap_preload_parameters(self, warmup_cache=warmup_cache)

    @property
    def _runtime_registry(self) -> Dict[str, Dict[str, Any]]:
        """Property to expose runtime registry for testing continuity."""
        return _build_runtime_registry(self)


    def _create_default_metadata(self) -> ManagerMetadata:
        """Create default metadata for UnifiedParameterManager."""
        return _create_default_metadata(self)

    def _do_initialize(self) -> None:
        """Initialize the parameter manager (BaseManager abstract method)."""
        return _do_initialize(self)

    def _do_configure(self, config: Dict[str, Any]) -> None:
        """Configure the parameter manager (BaseManager abstract method)."""
        return _do_configure(self, config)

    @property
    def bounds_policy(self) -> BoundsPolicy:
        """Return current handling strategy for out-of-bounds updates."""
        return self._bounds_policy

    def set_bounds_policy(self, policy: Union[str, BoundsPolicy]) -> None:
        """Set handling strategy for out-of-bounds updates."""
        if isinstance(policy, BoundsPolicy):
            self._bounds_policy = policy
        else:
            self._bounds_policy = BoundsPolicy(str(policy))
        self._bounds_policy_explicit = True

    def get_clamp_events(self, clear: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve recorded clamp events.

        Parameters
        ----------
        clear : bool
            Whether to clear stored events after retrieval.
        """
        return _param_events.get_clamp_events(self, clear=clear)



    def _do_validate(self) -> List[str]:
        """Validate parameter manager state (BaseManager abstract method)."""
        return _do_validate(self)

    def _do_cleanup(self) -> None:
        """Cleanup parameter manager resources (BaseManager abstract method)."""
        return _do_cleanup(self)

    def _load_initial_parameters(self) -> None:
        """Load initial parameters from database."""
        return _bootstrap_load_initial_parameters(self)

    def _validate_type(self, value: Any, expected_type: type, key: str) -> Any:
        """
        Validate that value matches expected type.
        
        Parameters:
            value: Value to validate
            expected_type: Expected Python type
            key: Parameter key for error messages
            
        Returns:
            The value if validation passes
            
        Raises:
            TypeError: If value doesn't match expected type
        """
        return _type_utils.validate_type(value, expected_type, key)

    @staticmethod
    def _unwrap_trainable_value(value: Any, expected_type: Optional[type]) -> Any:
        return _type_utils.unwrap_trainable_value(value, expected_type)

    @staticmethod
    def _format_expected_type(expected_type: Optional[Union[type, Tuple[type, ...]]]) -> str:
        return _type_utils.format_expected_type(expected_type)
    
    def get(self, key: str, default: Any = None,
            namespace: Optional[str] = None,
            units: Optional[str] = None,
            expected_type: Optional[type] = None,
            required: bool = False,
            bounds: Optional[Tuple[float, float]] = None) -> ParameterValue:
        """
        Get parameter value by key with optional namespace filtering and unit conversion.

        This method includes LRU caching to improve performance for frequently accessed parameters.

        Parameters:
            key : str
                Parameter key (dot notation)
            default : Any
                Default value if parameter not found
            namespace : Optional[str]
                Filter by namespace (e.g., 'biosphere.photosynthesis')
            units : Optional[str]
                Desired units for return value (will convert if different from stored units)
            expected_type : Optional[type]
                Expected Python type for the returned value (e.g., float, int, torch.Tensor)
                If provided, will validate that the returned value matches this type
            required : bool
                If True, raise KeyError when the parameter is not found instead of returning default
            bounds : Optional[Tuple[float, float]]
                If provided, clamp the returned numeric value to (lower, upper).
                Issues a UserWarning when clamping occurs.

        Returns:
            ParameterValue: Parameter value (converted to requested units if specified) or default

        Examples:
            >>> manager.get('biosphere.photosynthesis.vcmax25')
            60.0
            >>> manager.get('missing.parameter', default=42.0)
            42.0
            >>> manager.get('temperature', units='K')  # Convert from Celsius to Kelvin
            298.15
        """
        value = _accessors.get_value(
            self,
            key=key,
            default=default,
            namespace=namespace,
            units=units,
            expected_type=expected_type,
            required=required,
        )
        # Bounds clamping (merged from former get_parameter_value alias)
        if bounds is not None and value is not None:
            lower, upper = bounds
            scalar_value: Optional[float]
            if isinstance(value, Tensor):
                scalar_value = float(value.item()) if value.numel() == 1 else None
            elif isinstance(value, (int, float)):
                scalar_value = float(value)
            else:
                scalar_value = None

            if scalar_value is not None:
                clamped_value = scalar_value
                if lower is not None:
                    clamped_value = max(clamped_value, lower)
                if upper is not None:
                    clamped_value = min(clamped_value, upper)
                if clamped_value != scalar_value:
                    warnings.warn(
                        f"Parameter '{key}' value {scalar_value} outside bounds {bounds}; clamped to {clamped_value}",
                        UserWarning,
                        stacklevel=3,
                    )
                    if isinstance(value, Tensor):
                        value = torch.as_tensor(
                            clamped_value,
                            device=value.device,
                            dtype=value.dtype,
                        )
                    else:
                        value = clamped_value
        return value

    def get_parameter(self, key: str, namespace: Optional[str] = None) -> Optional[ParameterSpec]:
        """
        Return full ParameterSpec metadata instead of a raw scalar/tensor value.

        Use this method when callers need metadata fields (units, bounds,
        references, provenance) rather than a resolved runtime value.
        The specification is the one the manager holds, not a copy: its
        fields, the value field included (the stored value itself), are
        read and never changed in place. Values are read with get(), which
        hands out a caller-owned copy of a mapping or list value.
        """
        return _get_parameter(self, key, namespace)

    def _get_impl(self, full_key: str, default: Any = None,
                  units: Optional[str] = None,
                  expected_type: Optional[type] = None,
                  normalize: bool = True) -> ParameterValue:
        return _access_impl(
            self,
            full_key=full_key,
            default=default,
            units=units,
            expected_type=expected_type,
            normalize=normalize,
        )
    
    def set_parameter(self, spec: ParameterSpec, tier: str) -> None:
        """
        Persist a parameter specification and refresh local caches.

        Parameters
        ----------
        spec : ParameterSpec
            Fully defined parameter specification to store.
        tier : str
            Database tier to write to (default 'calibrated').
        """
        return _set_parameter(self, spec, tier=tier)

    def warmup_cache(self, hot_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Pre-load frequently accessed parameters into LRU cache.
        
        This method improves performance by pre-caching common parameters
        that are accessed repeatedly during simulation forward() calls.
        
        Parameters
        ----------
        hot_keys : Optional[List[str]]
            List of parameter keys to pre-load. If None, uses default hot parameters.
            
        Returns
        -------
        Dict[str, Any]
            Warmup statistics including number of parameters cached and cache hit rate
            
        Examples
        --------
        >>> manager = get_global_manager()
        >>> stats = manager.warmup_cache()
        >>> logger.info("Cached %s parameters", stats['cached_count'])
        
        Mathematical Expression
        -----------------------
        Cache warming improves access time:
            T_access_cold = O(log n) for database lookup
            T_access_warm = O(1) for cache hit
            Expected speedup = T_cold / T_warm approx 10-20x for hot paths
            
        References
        ----------
        - LRU Cache Performance Analysis: Belady (1966)
        - Hot-spot optimization techniques in scientific computing
        """
        return _cache_ops.warmup_cache(self, hot_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics including:
            - parameter_cache: Stats for parameter value cache
            - unit_conversion_cache: Stats for unit conversion cache
            - cache_enabled: Whether caching is enabled
            - most_accessed: Top 10 most accessed parameters
            
        Examples
        --------
        >>> manager = get_global_manager()
        >>> stats = manager.get_cache_stats()
        >>> logger.info("Parameter cache hit rate: %.2f%%", stats['parameter_cache']['hit_rate'] * 100.0)
        """
        return _cache_ops.get_cache_stats(self)
    
    def exists(self, key: str) -> bool:
        """
        Check if parameter exists.

        Parameters:
            key : str
                Parameter key

        Returns:
            bool: True if parameter exists
        """
        return _exists(self, key)
    
    def set_value(self, key: str, value: ParameterValue,
                  units: Optional[str] = None,
                  validate: bool = True,
                  notify: bool = True) -> None:
        """
        Set parameter value with optional validation and notification.

        Parameters:
            key : str
                Parameter key (dot notation)
            value : ParameterValue
                New parameter value
            units : Optional[str]
                Expected units for validation
            validate : bool
                Whether to validate the new value
            notify : bool
                Whether to notify change callbacks

        Raises:
            ValueError: If validation fails
            KeyError: If parameter not found and validation enabled

        Mathematical Expression:
            Parameter update: theta' = value where theta in Theta
            Validation: V(theta')  U(theta', expected_units) = True
            Notification:  in Observers: o.notify(key, theta, theta')
        """
        with self._mutation_lock:
            return _set_value(
                self,
                key,
                value,
                units=units,
                validate=validate,
                notify=notify,
            )

    def _record_clamp_event(
        self,
        spec: ParameterSpec,
        attempted_value: Any,
        clamped_value: float,
        message: Optional[str] = None,
    ) -> None:
        """Record a clamp operation for auditing."""
        _param_events.record_clamp_event(
            self,
            spec,
            attempted_value,
            clamped_value,
            message=message,
        )



    def _clamp_value_to_bounds(self, spec: ParameterSpec, attempted_value: Any) -> Tuple[bool, Optional[float]]:
        """Clamp numeric parameters to configured bounds when possible."""
        return _bounds_ops.clamp_value_to_bounds(self, spec, attempted_value)

    
    def list(self, namespace: Optional[str] = None,
             pattern: Optional[str] = None,
             category: Optional[ParameterCategory] = None) -> List[str]:
        """
        List parameter keys with optional filtering.

        Parameters:
            namespace : Optional[str]
                Filter by namespace prefix
            pattern : Optional[str]
                Filter by pattern (supports wildcards and regex)
            category : Optional[ParameterCategory]
                Filter by parameter category

        Returns:
            List[str]: Matching parameter keys
        """
        return _list_parameters(self, namespace=namespace, pattern=pattern, category=category)
    
    def _ensure_parameter_spec(self, key: str) -> Optional[ParameterSpec]:
        """
        Ensure a ParameterSpec object is available for the given key.

        Parameters
        ----------
        key : str
            Canonical parameter key.

        Returns
        -------
        Optional[ParameterSpec]
            Parameter specification if it exists in memory or the database.
        """
        return _ensure_parameter_spec(self, key)

    def get_namespace_parameters(
        self,
        namespace: str,
        include_subnamespaces: bool = True,
        return_specs: bool = False,
    ) -> Dict[str, Union[ParameterValue, ParameterSpec]]:
        """
        Retrieve all parameters within the specified namespace.

        Parameters
        ----------
        namespace : str
            Namespace (e.g., "relic.pedosphere.physical").
        include_subnamespaces : bool, optional
            Whether to include nested namespaces beneath the given prefix.
        return_specs : bool, optional
            If True, return ParameterSpec objects instead of raw values.

        Returns
        -------
        Dict[str, Union[ParameterValue, ParameterSpec]]
            Mapping of parameter keys to values or specifications.
        """
        return _get_namespace_parameters(
            self,
            namespace=namespace,
            include_subnamespaces=include_subnamespaces,
            return_specs=return_specs,
        )

    def load_yaml_file(
        self,
        file_path: Union[str, Path],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a YAML configuration file using safe parsing.

        This helper is for non-parameter configuration files. If a namespace is
        provided, the loaded mapping is cached under that key.
        """
        return _load_yaml_file(self, file_path, namespace)

    def load_omegaconf_config(
        self,
        config: Any,
        *,
        namespace: str,
    ) -> Dict[str, Any]:
        """
        Load an OmegaConf configuration into a plain mapping.

        The resolved config is cached under the namespace the caller names.
        """
        return _load_omegaconf_config(self, config, namespace=namespace)

    def get_namespace_dict(self, namespace: str) -> Dict[str, Any]:
        """Return a copy of the cached external configuration of a namespace.

        A namespace that was never ingested has no external configuration (an
        explicit empty mapping); callers merge the parameter database values.
        """
        return _get_namespace_dict(self, namespace)

    def register_parameter(self, spec: ParameterSpec) -> None:
        """
        Register a parameter specification with the manager.

        Parameters:
            spec : ParameterSpec
                Parameter specification to register

        Raises:
            ValueError: If parameter key conflicts or spec is invalid
        """
        return _register_parameter(self, spec)
    
    def _register_nn_parameter(self, spec: ParameterSpec) -> None:
        """Register parameter as PyTorch nn.Parameter."""
        return _register_nn_parameter(self, spec)
    
    def get_nn_parameters(self) -> Dict[str, nn.Parameter]:
        """Get all PyTorch parameters for optimization."""
        return _get_nn_parameters(self)
    
    def get_trainable_parameters(self) -> Dict[str, nn.Parameter]:
        """
        Get only trainable PyTorch parameters (requires_grad=True).
        """
        return _get_trainable_parameters(self)

    
    

    
    def freeze(self, keys: Optional[List[str]] = None) -> None:
        """
        Freeze parameters (disable gradients).
        """
        return _freeze_parameters(self, keys)
    
    def unfreeze(self, keys: Optional[List[str]] = None) -> None:
        """
        Unfreeze parameters (enable gradients).
        """
        return _unfreeze_parameters(self, keys)
    
    def to(self, device: Union[str, torch.device],
           dtype: Optional[torch.dtype] = None) -> 'UnifiedParameterManager':
        """
        Move all PyTorch parameters to specified device/dtype.
        """
        return _to_device(self, device, dtype)
    
    def validate_parameters(self, keys: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Validate parameters against their specifications."""
        return _validate_parameters(self, keys)

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of current parameter state.
        
        Returns:
            Dict[str, Any]: Serializable snapshot
        """
        return _snapshot_state(self)
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore parameter state from snapshot.
        
        Parameters:
            snapshot : Dict[str, Any]
                Snapshot data from previous snapshot() call
        """
        _restore_state(self, snapshot)

    def _on_storage_restore(self, snapshot) -> None:
        """
        Sync manager state when underlying storage is restored.

        Ensures ParameterSpec values and caches reflect the storage snapshot.
        """
        _sync_storage_restore(self, snapshot)
    
    def diff(self, other_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current state with another snapshot.
        
        Parameters:
            other_snapshot : Dict[str, Any]
                Snapshot to compare against
                
        Returns:
            Dict[str, Any]: Differences summary
        """
        return _diff_snapshots(self, other_snapshot)
    
    def export(self, namespace: Optional[str] = None,
               format: str = "yaml",
               output_path: Optional[Path] = None) -> Union[str, None]:
        """
        Export parameters to specified format.
        
        Parameters:
            namespace : Optional[str]
                Export only parameters under this namespace
            format : str
                Export format ('yaml', 'json', 'toml')
            output_path : Optional[Path]
                File path to save, return string if None
                
        Returns:
            Union[str, None]: Exported data as string or None if saved to file
        """
        return _export_namespace(
            self,
            namespace=namespace,
            format=format,
            output_path=output_path,
        )

    def set_unit_validation_mode(self,
                                 mode: Union[str, UnitValidationMode],
                                 recheck: bool = False) -> None:
        """Set unit validation mode for parameter specifications."""
        if isinstance(mode, UnitValidationMode):
            unit_mode = mode
        else:
            normalized_mode = str(mode).strip().lower().replace("_", "-")
            unit_mode = UnitValidationMode(normalized_mode)

        if unit_mode == self.unit_validation_mode and not recheck:
            return

        self.unit_validation_mode = unit_mode
        self._unit_validation_checked.clear()
        if recheck:
            self._unit_validation_issues.clear()
            if unit_mode != UnitValidationMode.OFF:
                for key, spec in self.parameters.items():
                    self._check_units_for_spec(key, spec, source="recheck")

    def get_unit_validation_issues(self, clear: bool = False) -> List[Dict[str, Any]]:
        """Return recorded unit validation issues."""
        issues = list(self._unit_validation_issues)
        if clear:
            self._unit_validation_issues.clear()
        return issues

    def scan_unit_parseability(self, strict: bool = False) -> List[Dict[str, Any]]:
        """Scan database parameters for unit parseability issues."""
        issues: List[Dict[str, Any]] = []
        if not hasattr(self, "database") or self.database is None:
            return issues
        specs = self.database.load_all()
        for key, spec in specs.items():
            issue = self._validate_units_for_spec(
                key,
                spec,
                strict=strict,
                source="scan",
            )
            if issue:
                issues.append(issue)
        return issues

    def _should_skip_unit_validation(self, spec: Optional[ParameterSpec]) -> bool:
        return _should_skip_unit_validation(self, spec)

    def _validate_units_for_spec(self,
                                 key: str,
                                 spec: Optional[ParameterSpec],
                                 strict: bool,
                                 source: str) -> Optional[Dict[str, Any]]:
        return _validate_units_for_spec(self, key, spec, strict=strict, source=source)

    def _check_units_for_spec(self, key: str, spec: Optional[ParameterSpec], source: str) -> None:
        return _check_units_for_spec(self, key, spec, source)
    
    def add_change_callback(self, callback: ChangeCallback) -> None:
        """Add parameter change notification callback."""
        _param_events.add_change_callback(self, callback)


    
    def remove_change_callback(self, callback: ChangeCallback) -> None:
        """Remove parameter change notification callback."""
        _param_events.remove_change_callback(self, callback)


    
    def _convert_parameter_units(self, value: ParameterValue, spec: ParameterSpec,
                                target_units: str) -> ParameterValue:
        """
        Convert parameter value to target units with caching.
        """
        return _convert_parameter_units(self, value, spec, target_units)

    def _convert_parameter_units_impl(self, value: ParameterValue, spec: ParameterSpec,
                                target_units: str) -> ParameterValue:
        """Convert parameter value to target units."""
        return _convert_parameter_units_impl(self, value, spec, target_units)

    def _notify_change(self, key: str, old_value: ParameterValue, new_value: ParameterValue) -> None:
        """Notify all callbacks of parameter change."""
        _param_events.notify_change(self, key, old_value, new_value)


    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive parameter manager status."""
        return _get_status(self)

    
    # Additional methods for test continuity
    
    def get_all_parameters(self) -> Dict[str, ParameterSpec]:
        """Get all registered parameters.
        
        Returns:
            Dict[str, ParameterSpec]: All parameter specifications
        """
        return self.parameters.copy()
    
    def has_parameter(self, key: str) -> bool:
        """Check if a parameter is registered.
        
        Parameters:
            key : str
                Parameter key to check
                
        Returns:
            bool: True if parameter exists
        """
        return self.get_parameter(key) is not None
    
    def register_parameters(self, specs: List[ParameterSpec]) -> None:
        """
        Register multiple parameter specifications.

        Parameters:
            specs : List[ParameterSpec]
                Parameter specifications to register
        """
        return _register_parameters(self, specs)
    
    def get_batch(self, keys: List[str]) -> List[Any]:
        """Get multiple parameter values at once.
        
        Parameters:
            keys : List[str]
                List of parameter keys
                
        Returns:
            List[Any]: List of parameter values
        """
        return [self.get(key) for key in keys]
    
    def set_batch(
        self,
        updates: Dict[str, Any],
        *,
        validate: bool = True,
        notify: bool = True,
    ) -> None:
        """Atomically set multiple parameter values through the canonical setter.
        
        Parameters:
            updates : Dict[str, Any]
                Dictionary of key-value pairs to update
            validate : bool
                Whether to validate every candidate value
            notify : bool
                Whether to emit change notifications after the full commit

        Notes:
            The batch is a single transaction over the existing parameter owners.
            Validation, unit handling, bounds policy, storage synchronization, and
            trainable tensor updates remain owned by ``set_value``. If any update
            fails, every affected owner is restored before the exception escapes.
        """
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary")
        if not updates:
            return

        canonical_updates: List[Tuple[str, Any]] = []
        seen_keys: Set[str] = builtins.set()
        for key, value in updates.items():
            canonical_key = normalize_parameter_key(key)
            if not isinstance(canonical_key, str) or not canonical_key:
                raise ValueError("Batch parameter keys must be non-empty strings")
            if canonical_key in seen_keys:
                raise ValueError(
                    f"Duplicate canonical parameter key in batch: {canonical_key}"
                )
            seen_keys.add(canonical_key)
            canonical_updates.append((canonical_key, value))

        with self._mutation_lock:
            snapshots: Dict[str, Dict[str, Any]] = {}
            for canonical_key, _value in canonical_updates:
                nn_parameter = self.nn_parameters.get(canonical_key)
                snapshots[canonical_key] = {
                    "parameter_present": canonical_key in self.parameters,
                    "parameter": self.parameters.get(canonical_key),
                    "storage_present": self.storage.exists(canonical_key),
                    "storage_value": self.storage.get(canonical_key),
                    "nn_present": nn_parameter is not None,
                    "nn_parameter": nn_parameter,
                    "nn_data": (
                        nn_parameter.detach().clone()
                        if nn_parameter is not None
                        else None
                    ),
                }

            clamp_events_before = deepcopy(self._clamp_events)
            pending_warnings = []
            try:
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    for canonical_key, value in canonical_updates:
                        _set_value(
                            self,
                            canonical_key,
                            value,
                            validate=validate,
                            notify=False,
                        )
                    pending_warnings = list(captured_warnings)
            except Exception:
                for canonical_key, _value in canonical_updates:
                    snapshot = snapshots[canonical_key]
                    if snapshot["parameter_present"]:
                        self.parameters[canonical_key] = snapshot["parameter"]
                    else:
                        self.parameters.pop(canonical_key, None)

                    if snapshot["storage_present"]:
                        self.storage.set(canonical_key, snapshot["storage_value"])
                    else:
                        self.storage.delete(canonical_key)

                    if snapshot["nn_present"]:
                        nn_parameter = snapshot["nn_parameter"]
                        self.nn_parameters[canonical_key] = nn_parameter
                        with torch.no_grad():
                            nn_parameter.copy_(snapshot["nn_data"])
                    else:
                        self.nn_parameters.pop(canonical_key, None)
                    self.invalidate_cache_key(canonical_key)
                self._clamp_events = clamp_events_before
                raise

            # Every committed key is notified, even when callbacks of an earlier
            # key failed; the callback failures (ExceptionGroup per key from
            # notify_change) are raised together after all notifications and the
            # replay of the batch warnings. The changes stay committed.
            notification_failures: List[Exception] = []
            if notify:
                for canonical_key, _value in canonical_updates:
                    old_value = snapshots[canonical_key]["storage_value"]
                    new_value = self.storage.get(canonical_key)
                    if _values_differ(old_value, new_value):
                        try:
                            self._notify_change(canonical_key, old_value, new_value)
                        except ExceptionGroup as failures:
                            notification_failures.append(failures)

            for warning_record in pending_warnings:
                warnings.warn(
                    str(warning_record.message),
                    warning_record.category,
                    stacklevel=2,
                )
            if notification_failures:
                raise ExceptionGroup(
                    f"Change callbacks failed for {len(notification_failures)} of "
                    f"{len(canonical_updates)} batch parameters (all changes stay committed)",
                    notification_failures,
                )
    
    def get_parameters_by_category(self, category: ParameterCategory) -> List[ParameterSpec]:
        """Get all parameters in a specific category.
        
        Parameters:
            category : ParameterCategory
                Parameter category to filter by
                
        Returns:
            List[ParameterSpec]: Parameters in the specified category
        """
        return _get_parameters_by_category(self, category)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parameter statistics.
        
        Returns:
            Dict[str, Any]: Statistics including counts by category
        """
        return _get_statistics(self)
    
    def clear_parameters(self) -> None:
        """Clear all parameters (for testing)."""
        self.parameters.clear()
        self.nn_parameters.clear()
        self.change_history.clear()
        # Also clear storage
        if hasattr(self, 'storage') and self.storage:
            self.storage.clear()

    def delete(self, key: str) -> None:
        """
        Delete a parameter from the manager.
        
        Parameters:
            key : str
                Parameter key to delete
                
        Raises:
            KeyError: If parameter not found
        """
        if key not in self.parameters:
            raise KeyError(f"Parameter {key} not found")
        
        # Remove from parameters dict
        del self.parameters[key]
        
        # Remove from storage if available
        if hasattr(self, 'storage') and self.storage:
            if hasattr(self.storage, 'delete'):
                self.storage.delete(key)
        
        # Remove from nn_parameters if exists
        if key in self.nn_parameters:
            del self.nn_parameters[key]

        # Cached query results for the deleted key must not survive it; the
        # cache key mixes namespace and unit context, so follow set_value and
        # clear the query and unit-conversion caches as a whole.
        if self._cache_enabled:
            if self._parameter_cache is not None:
                self._parameter_cache.clear()
            if self._unit_conversion_cache is not None:
                self._unit_conversion_cache.clear()

        logger.debug(f"Deleted parameter: {key}")
    
    def get_diff(self, snapshot: 'ParameterSnapshot') -> Dict[str, Tuple[Any, Any]]:
        """
        Get differences between current state and a snapshot.
        
        Parameters:
            snapshot : ParameterSnapshot
                Snapshot to compare against
                
        Returns:
            Dict[str, Tuple[Any, Any]]: Differences mapping parameter names to (old_value, new_value) tuples
        """
        return _get_snapshot_diff(self, snapshot)
    
    def load_from_database(self, db_path: str = None, source_priority: List[str] = None) -> int:
        """Load parameters from database files.
        
        Parameters:
            db_path : str
                Database path or file path
            source_priority : List[str]
                Priority order for sources (earlier = higher priority)
                
        Returns:
            int: Number of parameters loaded
        """
        return _load_from_database(self, db_path=db_path, source_priority=source_priority)
    
    def save_to_database(self, db_path: str = None, format: str = "yaml") -> int:
        """Save parameters to database files.
        
        Parameters:
            db_path : str
                Database path or file path
            format : str
                Output format
                
        Returns:
            int: Number of parameters saved
        """
        return _save_to_database(self, db_path=db_path, format=format)
    
    def export_parameters(self, output_path: str, format: str = "yaml") -> None:
        """
        Export parameters to file.
        
        Parameters:
            output_path : str
                Output file path
            format : str
                Output format (yaml, json, toml)
        """
        _export_parameters(self, output_path, format=format)
    
    def create_snapshot(self, description: str = "") -> 'ParameterSnapshot':
        """
        Create a snapshot of current parameter state.
        
        Parameters:
            description : str
                Description of the snapshot
                
        Returns:
            ParameterSnapshot: Snapshot object
        """
        return _create_snapshot(self, description)
    
    def restore_snapshot(self, snapshot: 'ParameterSnapshot') -> None:
        """
        Restore parameters from a snapshot.
        
        Parameters:
            snapshot : ParameterSnapshot
                Snapshot to restore from
        """
        _restore_snapshot(self, snapshot)
    
    # ============================================================================
    # Parameter Audit and Registry Scanning Methods
    # ============================================================================

    def load_category(self, category: str, subcategory: Optional[str] = None) -> List[str]:
        """
        Backport helper to preload a category/subcategory namespace.

        Notes
        -----
        The unified manager uses a lazy database loader. This helper preloads the
        requested namespace and returns the discovered keys.
        """
        return _audit_ops.load_category(self, category, subcategory)
    
    def run_registry_audit(self,
                          expected_params: Optional[List[str]] = None,
                          namespace_filter: Optional[str] = None,
                          include_unused: bool = True) -> ParameterAuditReport:
        """
        Execute comprehensive parameter registry audit.
        
        Scans the entire parameter ecosystem to identify:
        - Missing parameters (requested but not found)
        - Orphan parameters (defined but never used)
        - Duplicate registrations
        - Unit/bounds inconsistencies
        - Validation errors
        
        Parameters:
            expected_params : Optional[List[str]]
                Explicit list of expected parameters (if None, scans all YAML)
            namespace_filter : Optional[str]
                Filter audit to specific namespace
            include_unused : bool
                Whether to flag unused parameters as issues
                
        Returns:
            ParameterAuditReport: Comprehensive audit report
            
        Examples:
            >>> report = manager.run_registry_audit()
            >>> logger.info("Found %s missing parameters", len(report.missing_keys))
            >>> logger.info("Found %s orphan parameters", len(report.orphan_keys))
        """
        return _audit_ops.run_registry_audit(
            self,
            expected_params=expected_params,
            namespace_filter=namespace_filter,
            include_unused=include_unused,
            normalize_key=normalize_parameter_key,
        )
    
    def get_missing_requests(self) -> Dict[str, int]:
        """
        Get statistics on missing parameter requests.
        
        Returns:
            Dict[str, int]: Map of parameter keys to request counts
            
        Examples:
            >>> missing = manager.get_missing_requests()
            >>> for key, count in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:5]:
            ...     logger.info("%s: requested %s times", key, count)
        """
        return _audit_ops.get_missing_requests(self)
    
    def clear_missing_requests(self) -> None:
        """Clear the missing requests tracking."""
        _audit_ops.clear_missing_requests(self)
    
    def _collect_expected_parameters(self, 
                                    explicit_list: Optional[List[str]] = None,
                                    namespace_filter: Optional[str] = None) -> Set[str]:
        """
        Collect the set of expected parameters.
        
        Sources (in priority order):
        1. Explicit list provided by user
        2. Parameters from database YAML files
        3. Parameters from Functor metadata (future)
        """
        return _audit_ops._collect_expected_parameters(self, explicit_list, namespace_filter)
    
    def _collect_observed_parameters(self, namespace_filter: Optional[str] = None) -> Set[str]:
        """
        Collect the set of observed/available parameters.
        
        Sources:
        1. Currently registered parameters
        2. Storage cache
        3. Database entries
        """
        return _audit_ops._collect_observed_parameters(self, namespace_filter)
    
    def _scan_missing_parameters(self, expected: Set[str], observed: Set[str], namespace_filter: Optional[str] = None) -> List[ParameterAuditIssue]:
        """Identify parameters that were expected but not found."""
        return _audit_ops._scan_missing_parameters(self, expected, observed, namespace_filter)
    
    def _scan_orphan_parameters(self, expected: Set[str], observed: Set[str]) -> List[ParameterAuditIssue]:
        """Identify parameters that exist but are never accessed/used."""
        return _audit_ops._scan_orphan_parameters(self, expected, observed)
    
    def _scan_duplicate_registrations(self) -> List[ParameterAuditIssue]:
        """Identify parameters with duplicate or conflicting registrations."""
        return _audit_ops._scan_duplicate_registrations(
            self,
            normalize_key=normalize_parameter_key,
        )
    
    def _scan_validation_errors(self, namespace_filter: Optional[str] = None) -> List[ParameterAuditIssue]:
        """Scan for parameters that fail validation."""
        return _audit_ops._scan_validation_errors(self, namespace_filter)
    
    def _scan_unit_consistency(self) -> List[ParameterAuditIssue]:
        """Check for unit consistency across related parameters."""
        return _audit_ops._scan_unit_consistency(self)
    
    def export_audit_report(self, report: ParameterAuditReport, 
                           format: str = "table",
                           output_path: Optional[Path] = None) -> Optional[str]:
        """
        Export audit report in various formats.
        
        Parameters:
            report : ParameterAuditReport
                Report to export
            format : str
                Output format: 'table', 'json', 'yaml', 'markdown'
            output_path : Optional[Path]
                File to save, return string if None
                
        Returns:
            Optional[str]: Formatted report as string, or None if saved to file
        """
        return _audit_ops.export_audit_report(report, format, output_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter manager state to dictionary for serialization.

        Returns:
            Dictionary containing all parameters, metadata, and runtime state
        """
        return _manager_to_dict(self)
    
    # Cache management methods
    
    @property
    def cache_stats(self) -> Dict[str, Any]:
        """
        Get LRU cache statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing cache statistics including:
                - parameter_cache: Stats for parameter query cache
                - unit_conversion_cache: Stats for unit conversion cache
                - cache_enabled: Whether caching is enabled
        
        Examples:
            >>> manager = get_global_manager()
            >>> stats = manager.cache_stats
            >>> logger.info("Hit rate: %.2f%%", stats['parameter_cache']['hit_rate'] * 100.0)
        """
        return _cache_ops.cache_stats(self)
    
    def clear_cache(self) -> None:
        """
        Clear all parameter caches.
        
        This method clears both the parameter query cache and the unit conversion cache,
        resetting all cache statistics.
        
        Use this method when:
        - Parameters have been modified externally
        - You need to force fresh lookups
        - Testing or debugging cache behavior
        
        Examples:
            >>> manager.clear_cache()
            >>> stats = manager.cache_stats
            >>> assert stats['parameter_cache']['hits'] == 0
        """
        _cache_ops.clear_cache(self)

    def invalidate_cache_key(self, key: str) -> bool:
        """
        Invalidate all cache entries for a specific parameter key.
        
        This is useful when a parameter is updated and you want to ensure
        fresh values are retrieved on next access.
        
        Parameters:
            key : str
                Parameter key to invalidate
                
        Returns:
            bool: True if any cache entries were invalidated
        
        Examples:
            >>> manager.set_parameter(spec)
            >>> manager.invalidate_cache_key(spec.key)
        """
        return _cache_ops.invalidate_cache_key(self, key)
    
    def preheat_cache(self, parameter_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Pre-load common parameters into cache.
        
        This method loads frequently used parameters into the LRU cache to improve
        performance during simulation loops.
        
        Parameters:
            parameter_keys : Optional[List[str]]
                List of parameter keys to pre-load. If None, uses a default list
                of commonly accessed parameters.
                
        Returns:
            Dict[str, Any]: Summary of preheating operation including:
                - loaded: Number of parameters successfully loaded
                - failed: Number of parameters that failed to load
                - cache_size: Final cache size
        
        Examples:
            >>> manager.preheat_cache()
            >>> manager.preheat_cache(['biosphere.photosynthesis.vcmax25', 'time_step'])
        """
        return _cache_ops.preheat_cache(self, parameter_keys)

    def debug_report_units(self) -> Dict[str, Dict[str, Any]]:
        return _debug_report_units(self)


# Global manager helpers are defined in ops.manager_registry to keep this module lean.
def get_global_manager(
    preload_all: bool = False,
    warmup_cache: bool = False,
) -> UnifiedParameterManager:
    from .ops import manager_registry as _manager_registry
    return _manager_registry.get_global_manager(preload_all=preload_all, warmup_cache=warmup_cache)


def reset_global_manager() -> None:
    from .ops import manager_registry as _manager_registry
    return _manager_registry.reset_global_manager()


def get(
    key: str,
    default: Any = None,
    namespace: Optional[str] = None,
    units: Optional[str] = None,
    expected_type: Optional[type] = None,
    required: bool = False,
) -> ParameterValue:
    from .ops import manager_registry as _manager_registry
    return _manager_registry.get(
        key=key,
        default=default,
        namespace=namespace,
        units=units,
        expected_type=expected_type,
        required=required,
    )


def set_value(
    key: str,
    value: ParameterValue,
    units: Optional[str] = None,
    validate: bool = True,
    notify: bool = True,
) -> None:
    from .ops import manager_registry as _manager_registry
    return _manager_registry.set_value(key, value, units, validate, notify)

def list_parameters(
    namespace: Optional[str] = None,
    pattern: Optional[str] = None,
    category: Optional[ParameterCategory] = None,
) -> List[str]:
    from .ops import manager_registry as _manager_registry
    return _manager_registry.list_parameters(namespace, pattern, category)


def debug_report_units() -> Dict[str, Dict[str, Any]]:
    from .ops import manager_registry as _manager_registry
    return _manager_registry.debug_report_units()


def get_namespace_parameters(
    namespace: str,
    include_subnamespaces: bool = True,
    return_specs: bool = False,
) -> Dict[str, Union[ParameterValue, ParameterSpec]]:
    from .ops import manager_registry as _manager_registry
    return _manager_registry.get_namespace_parameters(
        namespace=namespace,
        include_subnamespaces=include_subnamespaces,
        return_specs=return_specs,
    )


__all__ = [
    'UnifiedParameterManager',
    'BoundsPolicy',
    'ParameterSpec',
    'ParameterDatabase',
    'ParameterCategory',
    'ValidationLevel',
    'UnitValidationMode',
    'get_global_manager',
    'reset_global_manager',
    'get',
    'set_value',
    'list_parameters',
    'debug_report_units',
    'get_namespace_parameters',
    'ParameterValue',
    'ParameterDict',
    'MetadataDict',
    'ValidationErrors',
    'ChangeCallback'
]


