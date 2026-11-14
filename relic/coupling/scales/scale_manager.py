"""
Unified manager for multi-scale spatial and temporal coupling operations.

This manager integrates spatial, temporal, and hierarchical scaling logic into a 
single entry point, replacing the previous ScaleFacade. It leverages specific 
strategy classes for different scaling domains while maintaining a coherent 
configuration and audit trail.

References:
    - Jones, P.W. (1999). Conservative remapping schemes for spherical grids.
    - CF Conventions 1.8: Cell methods and grid mapping.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Set

import torch
import yaml
from relic.core import device_policy
from torch import Tensor

from ...core.managers.base import BaseManager, ManagerState
from ...core.managers.metadata import (
    ManagerMetadata, ManagerCapability, ManagerResource, CapabilityType,
    ResourceType
)
from relic.parameters import get_global_manager
from ..core.policy import PRECISION_POLICY_NAMESPACE, ScalingMode
from ...core.state import StateManager

# New imports for integrated functionality
from .temporal_strategies import TemporalScalingStrategy
from .hierarchical_strategies import HierarchicalScalingStrategy
from .topology_builder import TopologyBuilder
from .conservation_enforcer import ConservationEnforcer
from .variable_catalog import find_catalog_entry
from .physical_constraint_manager import PhysicalConstraintManager
from ...data.standards.cf_conventions import CFAttributeSynchronizer
from .types import ScalingStrategy, ScalingAudit
from ..core import (
    get_tensor_context,
    AuditCollector,
    reset_precision_policy_cache
)
from ..core.contracts import validate_contracts_file
from ...core.constants import get_scale_hierarchy
from ...parameters.namespaces import ParameterNamespace
from .execution_plan import (
    load_scaling_specs,
    validate_topology as validate_plan_topology,
    build_execution_plan,
    build_manifest,
    write_manifest,
    LevelSpec,
)
from .quickstart_config import build_quickstart_scale_config_impl
from .scale_strategies.manager_hierarchy_ops import (
    validate_scale_transition_impl,
    ensure_hierarchy_config_impl,
    is_extensive_impl,
    select_strategy_impl,
    init_hierarchical_components_impl,
    register_hierarchical_components_impl,
    map_mode_to_rule_impl,
    compute_child_units_impl,
    build_topology_from_parameters_impl,
    validate_topology_impl,
    get_temporal_coverage_threshold_impl,
    aggregate_temporal_alias_impl,
    disaggregate_temporal_alias_impl,
    update_metadata_with_quality_impl,
    collect_facade_warnings_impl,
    sync_cf_metadata_impl,
    record_scaling_audit_impl,
)
from .scale_strategies.manager_alias_ops import (
    load_alias_configuration_impl,
    handle_alias_aggregation_impl,
    handle_alias_disaggregation_impl,
)
from .scale_strategies.manager_spatial_ops import (
    validate_spatial_inputs_impl,
    regrid_variable_impl,
    sync_aggregation_metadata_impl,
)

# Spatial imports
from .spatial.grid_utils import GridDescriptor
from .spatial.factory import SpatialRegridderFactory
from .spatial.grid_manager import GridManager

logger = logging.getLogger(__name__)

# Resolved namespaces -- SSOT is the ParameterNamespace enum.
# NOTE: Direct resolution from ParameterNamespace is intentional here to avoid
# circular imports (coupling.__init__ -> scales.__init__ -> scale_manager -> coupling.__init__).
# The authoritative definitions live in parameters.namespaces.ParameterNamespace.
DEFAULTS_NS = ParameterNamespace.COUPLING_DEFAULTS.value
SCALES_TEMPORAL_NS = ParameterNamespace.COUPLING_SCALES_TEMPORAL.value
SCALES_SPATIAL_NS = ParameterNamespace.COUPLING_SCALES_SPATIAL.value
SCALES_DEFAULTS_NS = ParameterNamespace.COUPLING_SCALES_DEFAULTS.value
SCALES_HIERARCHY_NS = ParameterNamespace.COUPLING_SCALES_HIERARCHY.value
ALIAS_NAMESPACE = "coupling.scales.aliases"
_DEFAULT_PRIORITY_ORDER = ["temporal", "spatial", "conservative"]

# The settings ScaleManager.configure accepts. Anything else (the facade ledger tolerance
# included) is rejected by _validate_config before the call stores or applies anything.
SCALE_MANAGER_CONFIGURABLE_KEYS: Tuple[str, ...] = (
    "enforce_conservation",
    "cf_strict_mode",
    "contracts_path",
    "contracts_strict",
    "execution_plan_path",
    "scaling_manifest_enabled",
    "scaling_manifest_path",
)
_CONFIGURABLE_FLAGS: Tuple[str, ...] = (
    "enforce_conservation",
    "cf_strict_mode",
    "contracts_strict",
    "scaling_manifest_enabled",
)

class ScaleManager(BaseManager):
    """
    Unified manager for multi-scale spatial, temporal, and hierarchical coupling operations.
    
    Acts as the Single Source of Truth for all scaling operations in RELIC.
    Replaces the previous ScaleFacade.
    """
    
    # Alias transition sets
    _DEFAULT_ALIAS_TRANSITIONS = set()
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        enforce_conservation: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
        parameter_manager: Optional[Any] = None,
        state_manager: Optional[StateManager] = None,
        default_mode: Optional[ScalingMode] = None,
        cf_strict_mode: Optional[bool] = None,
        metadata: Optional[ManagerMetadata] = None,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize scale manager with configuration.

        The facade ledger tolerance is not a constructor argument: it is
        ``coupling.scales.defaults.precision.<mode>.tolerance`` of each call's mode
        (``ledger_tolerance``).
        """
        dependency_state_manager = None
        if dependencies is not None:
            dependency_state_manager = dependencies.get("state_manager")
        if (
            state_manager is not None
            and dependency_state_manager is not None
            and state_manager is not dependency_state_manager
        ):
            raise ValueError(
                "ScaleManager received conflicting explicit StateManager owners."
            )
        resolved_state_manager = (
            state_manager
            if state_manager is not None
            else dependency_state_manager
        )
        if (
            resolved_state_manager is not None
            and not isinstance(resolved_state_manager, StateManager)
        ):
            raise TypeError(
                "ScaleManager dependency 'state_manager' must be a StateManager."
            )

        # Store initialization parameters
        self._init_device = device
        self._init_enforce_conservation = enforce_conservation
        self._init_config = config or {}
        self._init_parameter_manager = parameter_manager
        self._init_state_manager = resolved_state_manager
        self.state_manager: Optional[StateManager] = None
        self._default_enforce_conservation: Optional[bool] = None
        self._default_contracts_strict: Optional[bool] = None
        
        self.default_mode = default_mode
        self.cf_strict_mode = cf_strict_mode
        self._init_cf_strict_mode = cf_strict_mode
        self.hierarchy_config: Optional[Dict[str, Any]] = None

        self._execution_plan_path: Optional[Path] = None
        self._manifest_enabled: Optional[bool] = None
        self._manifest_path: Optional[Path] = None
        
        # Alias sets
        self._alias_transitions: Set[Tuple[str, str, str]] = set()
        self._temporal_alias_up: Set[Tuple[str, str]] = set()
        self._temporal_alias_down: Set[Tuple[str, str]] = set()
        self._facade_warnings: List[str] = []
        self._warnings: List[str] = []  # Alias loading warnings
        self._param_cache: Dict[Tuple[str, str, str, Optional[type]], Any] = {}
        self._defaults_ns_cache: Dict[str, Any] = {}
        
        # Components to be initialized
        self.grid_manager = None
        self.regridder_factory = None
        self.cf_synchronizer = None
        self.temporal_strategy = None
        self.hierarchical_strategy = None
        self.topology_builder = None
        self.conservation_enforcer = None
        self.constraint_manager = None
        self.audit_collector = None
        self.tensor_context = None
        self.precision_policy = None
        self.contracts: Optional[Dict[str, Any]] = None
        
        # Initialize base class with metadata
        super().__init__(metadata=metadata or self._create_metadata())
        
    def _get_parameter_manager(self):
        """Resolve ParameterManager for scale operations (required)."""
        if self._init_parameter_manager is not None:
            return self._init_parameter_manager
        return get_global_manager()
    
    def _resolve_state_manager(self):
        """Return the explicitly injected canonical StateManager owner."""
        if self._init_state_manager is not None:
            return self._init_state_manager
        raise RuntimeError(
            "ScaleManager requires an explicit StateManager owner; "
            "global fallback would create a split scientific-state ledger."
        )

    def set_state_manager(self, state_manager: StateManager) -> None:
        """
        Bind a StateManager for lifecycle execution and auditing.

        Parameters
        ----------
        state_manager : StateManager
            The StateManager instance used for topology building and audit collection.

        Raises
        ------
        ValueError
            If a null state_manager is provided.
        RuntimeError
            If attempting to replace an already configured manager after initialization.
        """
        if state_manager is None:
            raise ValueError("state_manager cannot be None for ScaleManager configuration.")
        if not isinstance(state_manager, StateManager):
            raise TypeError("state_manager must be a StateManager instance.")
        with self._state_lock:
            current_owner = self.state_manager or self._init_state_manager
            if current_owner is state_manager:
                return
            if self.state != ManagerState.UNINITIALIZED:
                raise RuntimeError(
                    "StateManager already configured; cannot replace after initialization."
                )
            self._init_state_manager = state_manager
            self.state_manager = state_manager

    @staticmethod
    def _create_metadata() -> ManagerMetadata:
        """Create metadata for ScaleManager."""
        capabilities = [
            ManagerCapability(
                type=CapabilityType.SPATIAL_OPERATIONS,
                level="advanced",
                description="Multi-scale spatial coupling operations"
            ),
            ManagerCapability(
                type=CapabilityType.TEMPORAL_PROCESSING,
                level="advanced",
                description="Multi-scale temporal coupling operations"
            ),
            ManagerCapability(
                type=CapabilityType.CONSTRAINT_ENFORCEMENT,
                level="advanced",
                description="Conservation law enforcement for scaling"
            ),
            ManagerCapability(
                type=CapabilityType.CF_CONVENTIONS,
                level="standard",
                description="CF-1.8 metadata compliance"
            ),
        ]
        
        resources = [
            ManagerResource(
                type=ResourceType.SYSTEM_MEMORY,
                quantity=None,
                unit="bytes"
            ),
        ]
        
        dependencies = []
        
        return ManagerMetadata(
            name="ScaleManager",
            version="2.0.0",
            description="Unified manager for multi-scale spatial and temporal coupling operations",
            capabilities=capabilities,
            resources=resources,
            dependencies=dependencies,
            tags={"thread_safe", "gpu_enabled"},
            custom_attributes={
                "thread_safe": True,
                "supports_gpu": True,
                "supports_distributed": False
            }
        )
    
    def _do_initialize(self) -> None:
        """Perform ScaleManager-specific initialization."""
        # Set core attributes
        self.parameter_manager = self._get_parameter_manager()
        self.state_manager = self._resolve_state_manager()
        self.constraint_manager = self.state_manager.physical_constraint_manager
        if self.constraint_manager is None:
            raise RuntimeError(
                "ScaleManager requires the StateManager physical constraint owner."
            )
        if not isinstance(self.constraint_manager, PhysicalConstraintManager):
            raise TypeError(
                "StateManager physical constraint owner must be a PhysicalConstraintManager."
            )
        if self.constraint_manager.state != ManagerState.READY:
            raise RuntimeError(
                "StateManager physical constraint owner must already be READY; "
                "ScaleManager does not own its lifecycle."
            )
        self._param_cache.clear()
        self._defaults_ns_cache.clear()
        self._warmup_parameter_cache()
        if self.parameter_manager is not None:
            self._defaults_ns_cache = self._load_default_namespace_values()

        if self._init_device is None:
            device_value = self._get_param(
                "device",
                namespace=DEFAULTS_NS,
                units="dimensionless",
                expected_type=str,
                required=False,
                validate=True,
            )
            self.device = device_policy.resolve_device(
                explicit_device=(str(device_value) if device_value else None),
                module_name=__name__,
                default="auto",
            )
        else:
            self.device = device_policy.resolve_device(
                explicit_device=self._init_device,
                module_name=__name__,
                default="auto",
            )

        if self.default_mode is None or isinstance(self.default_mode, str):
            mode_value = (
                self.default_mode
                if isinstance(self.default_mode, str)
                else self._get_param(
                    "auto_scale_mode",
                    namespace=DEFAULTS_NS,
                    units="dimensionless",
                    expected_type=str,
                    required=True,
                    validate=True,
                )
            )
            # The declared coupling.scales.defaults.auto_scale_mode is required: an undeclared
            # mode raises instead of becoming the conservative mode.
            try:
                self.default_mode = ScalingMode(str(mode_value))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid scaling mode '{mode_value}' from coupling defaults."
                ) from exc

        # Every setting that configure() can change starts here from its constructor value or
        # its parameter default, so reset() followed by initialize() discards configured values.
        if self._init_cf_strict_mode is None:
            cf_value = self._get_param(
                "auto_scale_cf_strict_mode",
                namespace=DEFAULTS_NS,
                units="dimensionless",
                expected_type=bool,
                required=False,
                validate=True,
            )
            self.cf_strict_mode = bool(cf_value) if cf_value is not None else False
        else:
            self.cf_strict_mode = self._init_cf_strict_mode

        contracts_value = self._get_param(
            "contracts_strict",
            namespace=DEFAULTS_NS,
            units="dimensionless",
            expected_type=bool,
            required=False,
            validate=True,
        )
        self._default_contracts_strict = bool(contracts_value) if contracts_value is not None else False
        self.contracts = None

        plan_value = self._get_param(
            "execution_plan_path",
            namespace=DEFAULTS_NS,
            units="dimensionless",
            expected_type=str,
            required=False,
            validate=True,
        )
        self._execution_plan_path = self._normalize_optional_path(
            plan_value, f"{DEFAULTS_NS}.execution_plan_path", existing_file=True
        )

        manifest_enabled = self._get_param(
            "scaling_manifest_enabled",
            namespace=DEFAULTS_NS,
            units="dimensionless",
            expected_type=bool,
            required=False,
            validate=True,
        )
        self._manifest_enabled = bool(manifest_enabled) if manifest_enabled is not None else False

        manifest_path_value = self._get_param(
            "scaling_manifest_path",
            namespace=DEFAULTS_NS,
            units="dimensionless",
            expected_type=str,
            required=False,
            validate=True,
        )
        self._manifest_path = self._normalize_optional_path(
            manifest_path_value, f"{DEFAULTS_NS}.scaling_manifest_path"
        )
        
        # Initialize precision policy. Its tolerance is the one source of the facade ledger
        # tolerance (ledger_tolerance / conservation_tolerance); a non-finite or out-of-range
        # value raises in the policy loader, naming its key.
        reset_precision_policy_cache()
        self.precision_policy = self._get_precision_policy(self.default_mode)
        self.tensor_context = get_tensor_context(self.device, self.precision_policy.dtype)

        self._default_enforce_conservation = bool(self.precision_policy.enforce_conservation)
        if self._defaults_ns_cache:
            enforce_default = self._defaults_ns_cache.get("auto_scale_enforce_conservation")
            if enforce_default is not None:
                self._default_enforce_conservation = bool(enforce_default)

        self.enforce_conservation = (
            self._init_enforce_conservation
            if self._init_enforce_conservation is not None
            else self._default_enforce_conservation
        )

        # Initialize Components
        
        # 1. Spatial Managers
        # A GridManager that cannot be initialized raises: spatial grid features are never
        # switched off silently.
        self.grid_manager = GridManager(
            device=self.device,
            parameter_manager=self.parameter_manager,
        )
        self.grid_manager.initialize()
        self.grid_manager.validate()
        # The facade factory keeps no tolerance: regrid_variable passes the ledger tolerance of
        # each call's mode to the conservative regridder it builds (ledger_tolerance).
        self.regridder_factory = SpatialRegridderFactory(
            enforce_conservation=self.enforce_conservation,
            device=self.device
        )
        
        # 2. CF Synchronizer
        self.cf_synchronizer = CFAttributeSynchronizer(
            strict_mode=self.cf_strict_mode
        )
        
        # 3. Topology Builder
        if self.parameter_manager is not None:
            self.topology_builder = TopologyBuilder(
                parameter_manager=self.parameter_manager,
                get_param=self._get_param,
            )
        else:
            self.topology_builder = None
        
        # 4. Alias Handling
        self._load_alias_configuration()
        
        # 5. Temporal Strategy
        self.temporal_strategy = TemporalScalingStrategy(self)
        
        # 6. Hierarchical Strategy
        self.hierarchical_strategy = HierarchicalScalingStrategy(self)
        # Note: Hierarchical components (index/aggregator) are initialized lazily by the strategy or on first use
        self.hierarchical_index = None
        self.hierarchical_aggregator = None
        
        # 7. Enforcers and Constraints (the facade passes ledger_tolerance(mode) per call)
        self.conservation_enforcer = ConservationEnforcer(device=self.device)
        # 8. Audit (records are read through the collector API; none reach StateManager diagnostics)
        self.audit_collector = AuditCollector()
        
        logger.info(
            f"ScaleManager initialized: device={self.device}, "
            f"enforce_conservation={self.enforce_conservation}, "
            f"tolerance={self.conservation_tolerance:.2e}, "
            f"mode={self.default_mode}"
        )
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check one ``configure`` call and return the settings ``_do_configure`` applies.

        BaseManager.configure runs this before it stores or applies any setting, so a
        rejected call raises its ValueError and changes nothing: the lifecycle state, the
        stored configuration and every setting stay as they were, and the next valid call
        proceeds. Rejected: ``conservation_tolerance`` (the facade ledger tolerance is not a
        setting), any key outside ``SCALE_MANAGER_CONFIGURABLE_KEYS``, a flag that is not
        True or False, ``contracts_strict`` without ``contracts_path``, a ``contracts_path``
        that is not an existing file or does not load as a contracts file, an
        ``execution_plan_path`` that is not an existing file (None or an empty string
        unsets it), and an enabled scaling manifest without an output path (in the call or
        already configured).
        """
        if "conservation_tolerance" in config:
            # The facade ledger tolerance is not a configuration setting: its one source is
            # coupling.scales.defaults.precision.<mode>.tolerance, so every value (the one in
            # force included) is rejected.
            key = f"{PRECISION_POLICY_NAMESPACE}.{self.default_mode.value}.tolerance"
            raise ValueError(
                f"ScaleManager.configure does not take conservation_tolerance (got "
                f"{config['conservation_tolerance']!r}): the facade ledger tolerance is {key} = "
                f"{self.conservation_tolerance!r} (a call in another mode uses that mode's key); "
                f"change {key} in the parameter database, or select the mode with set_mode or the "
                "mode argument of a call."
            )
        unknown = sorted(str(name) for name in config if name not in SCALE_MANAGER_CONFIGURABLE_KEYS)
        if unknown:
            raise ValueError(
                f"ScaleManager.configure does not accept {unknown}; the configurable settings "
                f"are {list(SCALE_MANAGER_CONFIGURABLE_KEYS)}."
            )

        settings: Dict[str, Any] = {}
        for name in _CONFIGURABLE_FLAGS:
            if name in config:
                value = config[name]
                if not isinstance(value, bool):
                    raise ValueError(
                        f"ScaleManager.configure: {name} must be True or False, got {value!r}."
                    )
                settings[name] = value

        if "contracts_strict" in settings and "contracts_path" not in config:
            raise ValueError(
                "ScaleManager.configure: contracts_strict applies to the contracts file of the "
                "same call; pass contracts_path with it."
            )
        if "contracts_path" in config:
            contracts_path = self._normalize_optional_path(
                config["contracts_path"], "contracts_path", existing_file=True
            )
            if contracts_path is None:
                raise ValueError("ScaleManager.configure: contracts_path must name a contracts file.")
            strict = settings.pop("contracts_strict", self._default_contracts_strict)
            try:
                settings["contracts"] = validate_contracts_file(contracts_path, strict=strict)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"ScaleManager.configure: contracts_path {str(contracts_path)!r} is not valid YAML: {exc}"
                ) from exc

        if "execution_plan_path" in config:
            settings["execution_plan_path"] = self._normalize_optional_path(
                config["execution_plan_path"], "execution_plan_path", existing_file=True
            )
        if "scaling_manifest_path" in config:
            settings["scaling_manifest_path"] = self._normalize_optional_path(
                config["scaling_manifest_path"], "scaling_manifest_path"
            )
        manifest_enabled = settings.get("scaling_manifest_enabled", self._manifest_enabled)
        manifest_path = settings.get("scaling_manifest_path", self._manifest_path)
        if manifest_enabled and manifest_path is None:
            raise ValueError(
                "ScaleManager.configure: scaling_manifest_enabled requires scaling_manifest_path "
                "(in the same call or already configured); the manifest has no default location."
            )
        return settings

    def _do_configure(self, settings: Dict[str, Any]) -> None:
        """Apply the settings that ``_validate_config`` returned for this call.

        An error raised here (a component that cannot be built, a requested scaling
        manifest that cannot be written) is a lifecycle failure: BaseManager.configure
        moves the manager to ERROR and raises ManagerInitializationError for the
        configuration phase; ``reset()`` recovers.
        """
        if "enforce_conservation" in settings:
            self.enforce_conservation = settings["enforce_conservation"]
            self.regridder_factory = SpatialRegridderFactory(
                enforce_conservation=self.enforce_conservation,
                device=self.device
            )
        if "cf_strict_mode" in settings:
            self.cf_strict_mode = settings["cf_strict_mode"]
            self.cf_synchronizer = CFAttributeSynchronizer(
                strict_mode=self.cf_strict_mode
            )
        if "contracts" in settings:
            self.contracts = settings["contracts"]
        if "execution_plan_path" in settings:
            self._execution_plan_path = settings["execution_plan_path"]
        if "scaling_manifest_enabled" in settings:
            self._manifest_enabled = settings["scaling_manifest_enabled"]
        if "scaling_manifest_path" in settings:
            self._manifest_path = settings["scaling_manifest_path"]
        if self._manifest_enabled:
            # _validate_config guarantees an output path whenever the manifest is enabled.
            write_manifest(self._manifest_path, self.execution_plan())

        logger.debug(f"ScaleManager configured with: {settings}")
    
    def _do_validate(self) -> List[str]:
        """Validate ScaleManager configuration and state."""
        errors = []
        if self.device.type == 'cuda' and not device_policy.is_cuda_available():
            errors.append(f"CUDA device {self.device} requested but not available")
        if self.regridder_factory is None:
            errors.append("SpatialRegridderFactory not initialized")
        if self.state_manager is None:
            errors.append("StateManager owner not initialized")
        elif self.constraint_manager is not self.state_manager.physical_constraint_manager:
            errors.append("PhysicalConstraintManager owner diverged from StateManager")
        elif self.constraint_manager.state != ManagerState.READY:
            errors.append("PhysicalConstraintManager owner is not READY")
        return errors
    
    def _do_cleanup(self) -> None:
        """Clean up ScaleManager resources."""
        self.clear_caches()
        if hasattr(self.grid_manager, 'cleanup'):
            self.grid_manager.cleanup()
        logger.info("ScaleManager cleanup completed")

    def ensure_ready(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Ensure the manager is fully initialized/configured/validated.

        This defends against accidental use before lifecycle completion.
        """
        if self.is_ready:
            return

        lifecycle_config = config or self._init_config or {}
        current_state = self.state

        # Preserve existing initialization (and any downstream patching)
        # when callers have already invoked initialize() and tweaked factories.
        if current_state == ManagerState.UNINITIALIZED:
            self.initialize()
        elif current_state == ManagerState.ERROR:
            self.reset()
            self.initialize()
        # If we are already INITIALIZING, keep current components intact.

        self.configure(lifecycle_config)
        self.validate()

    def execution_plan(self, specs_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate a scaling execution plan manifest without changing execution."""
        plan_path = specs_path or self._execution_plan_path
        if plan_path is not None:
            levels, regrids = load_scaling_specs(plan_path)
        else:
            hierarchy = get_scale_hierarchy(self.parameter_manager)
            levels = [LevelSpec(name=level) for level in hierarchy]
            regrids = []
        validate_plan_topology(levels, regrids)
        steps = build_execution_plan(levels, regrids)
        return build_manifest(levels, regrids, steps)

    def clear_caches(self):
        """Clear all internal caches to free memory."""
        self._param_cache.clear()
        self._defaults_ns_cache.clear()
        if self.grid_manager: self.grid_manager.clear_cache()
        if self.regridder_factory: self.regridder_factory.clear_cache()
        if self.conservation_enforcer: self.conservation_enforcer.reset_history()
        reset_precision_policy_cache()
        logger.info("Cleared ScaleManager caches")

    # =========================================================================
    # Helper Methods (Moved from ScaleFacade/AliasFacade)
    # =========================================================================

    @staticmethod
    def _normalize_optional_path(
        value: Optional[Any], setting: str, *, existing_file: bool = False
    ) -> Optional[Path]:
        """Normalize an optional path setting named ``setting``.

        None or an empty string is unset (None); a str or os.PathLike is a path, which must
        name an existing file when ``existing_file`` is set; any other value, or a missing
        required file, raises ValueError naming the setting.
        """
        if value is None:
            return None
        if isinstance(value, str):
            if not value.strip():
                return None
            path = Path(value)
        elif isinstance(value, os.PathLike):
            path = Path(value)
        else:
            raise ValueError(f"{setting} must be a path (str or os.PathLike) or empty, got {value!r}.")
        if existing_file and not path.is_file():
            raise ValueError(f"{setting} {str(path)!r} is not an existing file.")
        return path

    def _get_precision_policy(self, mode: ScalingMode) -> Any:
        """Resolve precision policy using ParameterManager configuration."""
        from ..core import policy as policy_module
        try:
            return policy_module.get_precision_policy(mode, parameter_manager=self.parameter_manager)
        except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"Precision policy parameters unavailable for mode {mode.value}: {exc}"
            ) from exc

    def _require_topology_builder(self):
        """Ensure topology builder is available when ParameterManager is missing."""
        if self.topology_builder is None:
            raise RuntimeError(
                "TopologyBuilder is unavailable because ParameterManager could not be resolved. "
                "Initialize the ParameterManager or supply explicit topology information."
            )
        return self.topology_builder

    def _get_param(
        self,
        key: str,
        *,
        namespace: Optional[str] = None,
        units: Optional[str] = None,
        expected_type: Optional[type] = None,
        required: bool = True,
        validate: bool = True,
    ) -> Any:
        """Fetch parameter with strict namespace."""
        ns_label = str(namespace or "global")
        cache_key = (ns_label, key, str(units or ""), expected_type)
        manager = self.parameter_manager
        if manager is None:
            raise RuntimeError("ParameterManager is required for scale manager operations.")

        if validate and cache_key in self._param_cache:
            return self._param_cache[cache_key]

        value: Any = None
        try:
            value = manager.get(
                key,
                namespace=namespace,
                units=units,
                expected_type=expected_type,
                required=required,
            )
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            # An optional parameter that is absent reads as None (manager.get with required=False);
            # an invalid value raises for optional and required parameters alike.
            if not required:
                raise
            raise RuntimeError(
                f"Required parameter '{key}' in namespace '{ns_label}' is missing or invalid: {exc}"
            ) from exc
        if required and value is None:
            raise KeyError(
                f"Missing required parameter '{key}' in namespace '{ns_label}'."
            )
        if validate:
            self._param_cache[cache_key] = value
        return value

    def _warmup_parameter_cache(self) -> None:
        """Pre-load frequently used parameters to reduce validation overhead."""
        if self.parameter_manager is None:
            raise RuntimeError("ParameterManager is required for ScaleManager operations.")

        warmup_entries: List[Tuple[str, str, str, Optional[type]]] = [
            ("auto_scale_mode", DEFAULTS_NS, "dimensionless", str),
            ("auto_scale_enforce_conservation", DEFAULTS_NS, "dimensionless", bool),
            ("auto_scale_cf_strict_mode", DEFAULTS_NS, "dimensionless", bool),
            ("relative_tolerance", DEFAULTS_NS, "dimensionless", float),
            ("absolute_tolerance", DEFAULTS_NS, "dimensionless", float),
            ("coverage_epsilon", DEFAULTS_NS, "dimensionless", float),
            ("coverage_threshold", SCALES_TEMPORAL_NS, "dimensionless", float),
        ]
        hot_keys = [f"{ns}.{key}" for key, ns, *_ in warmup_entries]
        self.parameter_manager.warmup_cache(hot_keys=hot_keys)

        for key, ns, units, expected in warmup_entries:
            value = self.parameter_manager.get(
                key,
                namespace=ns,
                units=units,
                expected_type=expected,
                required=False,
            )
            if value is not None:
                cache_key = (str(ns), key, str(units or ""), expected)
                self._param_cache[cache_key] = value

    def _load_default_namespace_values(self) -> Dict[str, Any]:
        """Validate and capture defaults from the coupling defaults namespace."""
        if self.parameter_manager is None:
            raise RuntimeError("ParameterManager is required for ScaleManager defaults.")
        defaults: Dict[str, Any] = {}
        keys: Tuple[Tuple[str, type, str], ...] = (
            ("auto_scale_mode", str, "dimensionless"),
            ("auto_scale_enforce_conservation", bool, "dimensionless"),
            ("auto_scale_cf_strict_mode", bool, "dimensionless"),
            ("relative_tolerance", float, "dimensionless"),
            ("absolute_tolerance", float, "dimensionless"),
            ("coverage_epsilon", float, "dimensionless"),
        )
        for key, expected, units in keys:
            value = self.parameter_manager.get(
                key,
                namespace=DEFAULTS_NS,
                units=units,
                expected_type=expected,
                required=False,
            )
            if value is not None:
                defaults[key] = value
        return defaults

    def _load_alias_configuration(self) -> None:
        load_alias_configuration_impl(
            self,
            alias_namespace=ALIAS_NAMESPACE,
            default_priority_order=list(_DEFAULT_PRIORITY_ORDER),
        )

    def _handle_alias_aggregation(
        self,
        data_dict: Dict[str, Tensor],
        source_scale: str,
        target_scale: str,
        variables: List[str],
        strategy: ScalingStrategy,
        mode: ScalingMode,
        metadata: Optional[Dict[str, Any]],
        var_metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        """Raise for a temporal alias pair; spatial and conservative pairs aggregate hierarchically."""
        handle_alias_aggregation_impl(
            self,
            data_dict=data_dict,
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            strategy=strategy,
            mode=mode,
            metadata=metadata,
            var_metadata=var_metadata,
        )

    def _handle_alias_disaggregation(
        self,
        data_dict: Dict[str, Tensor],
        source_scale: str,
        target_scale: str,
        variables: List[str],
        strategy: ScalingStrategy,
        mode: ScalingMode,
        metadata: Optional[Dict[str, Any]],
        var_metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        """Raise for a temporal alias pair; spatial and conservative pairs disaggregate hierarchically."""
        handle_alias_disaggregation_impl(
            self,
            data_dict=data_dict,
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            strategy=strategy,
            mode=mode,
            metadata=metadata,
            var_metadata=var_metadata,
        )

    # ... [Helper methods copied from facade.py] ...
    def _validate_scale_transition(self, source: str, target: str, direction: str) -> bool:
        return validate_scale_transition_impl(self, source=source, target=target, direction=direction)

    def _ensure_hierarchy_config(self) -> Dict[str, Any]:
        return ensure_hierarchy_config_impl(self)

    def get_variable_metadata(self, name: str, *, scale: str) -> Dict[str, Any]:
        """
        Catalog metadata of one variable at one scale (the single catalog lookup).

        Resolution order: ``coupling.scales.<scale>.variables``, then
        ``coupling.scales.defaults.variables`` (exact name, case-insensitive name, then the
        documented short aliases). A variable found in neither raises ``KeyError`` (one error
        type for every name; the name is never inspected). The returned mapping is a copy.

        Parameters:
            name: Variable name
            scale: Scale label whose scale-specific catalog is consulted first

        Returns:
            Dict[str, Any]: Metadata entry of the variable
        """
        if self.parameter_manager is None:
            raise RuntimeError(
                "Variable metadata resolution requires ParameterManager; initialize it or provide explicit metadata."
            )
        default_vars = self.parameter_manager.get(
            "variables",
            namespace=SCALES_DEFAULTS_NS,
            expected_type=dict,
            required=True,
        )
        if not isinstance(default_vars, dict):
            raise TypeError("coupling.scales.defaults.variables must be a mapping.")
        scoped_namespace = f"coupling.scales.{scale}"

        def _read_catalog(namespace: str) -> Optional[Dict[str, Any]]:
            if namespace == SCALES_DEFAULTS_NS:
                return default_vars
            try:
                return self.parameter_manager.get(
                    "variables",
                    namespace=namespace,
                    expected_type=dict,
                    required=False,
                )
            except (KeyError, ValueError, TypeError, RuntimeError, AttributeError) as exc:
                raise RuntimeError(
                    f"Failed to load variable metadata for scale {scale}: {exc}"
                ) from exc

        found = find_catalog_entry(name, (scoped_namespace, SCALES_DEFAULTS_NS), _read_catalog)
        if found is None:
            raise KeyError(
                f"Variable metadata for '{name}' not found in {scoped_namespace}.variables "
                f"or coupling.scales.defaults.variables; explicit metadata is required."
            )
        return found[0]

    def _get_variable_metadata(self, scale: str, variables: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch form kept for callers not yet switched to get_variable_metadata (delegates)."""
        return {var: self.get_variable_metadata(var, scale=scale) for var in variables}

    def _is_extensive(self, var_metadata: Dict[str, Any]) -> bool:
        return is_extensive_impl(self, var_metadata=var_metadata)

    def _select_strategy(self, var_metadata: Dict[str, Dict[str, Any]], mode: ScalingMode) -> ScalingStrategy:
        return select_strategy_impl(self, var_metadata=var_metadata, mode=mode)

    def _init_hierarchical_components(self):
        return init_hierarchical_components_impl(self)

    def register_hierarchical_components(self, index: Any, aggregator: Any) -> None:
        return register_hierarchical_components_impl(self, index=index, aggregator=aggregator)

    def _map_mode_to_rule(self, mode: ScalingMode) -> str:
        return map_mode_to_rule_impl(self, mode=mode)

    def _compute_child_units(self, coarse_scale: str, fine_scale: str, parent_units: int) -> Optional[int]:
        return compute_child_units_impl(
            self,
            coarse_scale=coarse_scale,
            fine_scale=fine_scale,
            parent_units=parent_units,
        )

    def _build_topology_from_parameters(
        self,
        source_scale: str,
        target_scale: str,
        n_source_units: int,
        n_target_units: Optional[int] = None,
    ):
        return build_topology_from_parameters_impl(
            self,
            source_scale=source_scale,
            target_scale=target_scale,
            n_source_units=n_source_units,
            n_target_units=n_target_units,
        )

    def _validate_topology(self, topology: Dict[str, Any], operation: str) -> Dict[str, Any]:
        return validate_topology_impl(self, topology=topology, operation=operation)

    def _get_temporal_coverage_threshold(self) -> float:
        return get_temporal_coverage_threshold_impl(self)

    def _aggregate_temporal_alias(self, **kwargs):
        return aggregate_temporal_alias_impl(self, **kwargs)

    def _disaggregate_temporal_alias(self, **kwargs):
        return disaggregate_temporal_alias_impl(self, **kwargs)

    def _update_metadata_with_quality(
        self,
        metadata: Dict[str, Any],
        quality: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        return update_metadata_with_quality_impl(
            self,
            metadata=metadata,
            quality=quality,
            diagnostics=diagnostics,
        )

    def _collect_facade_warnings(self) -> List[str]:
        return collect_facade_warnings_impl(self)

    def _sync_cf_metadata(
        self,
        operation: str,
        source_scale: str,
        target_scale: str,
        metadata: Any,
        var_metadata: Any,
        data: Any,
    ) -> Dict[str, Any]:
        return sync_cf_metadata_impl(
            self,
            operation=operation,
            source_scale=source_scale,
            target_scale=target_scale,
            metadata=metadata,
            var_metadata=var_metadata,
            data=data,
        )

    def _record_scaling_audit(
        self,
        scaling_audit: ScalingAudit,
        conservation_details: Optional[Dict[str, Any]] = None,
        cf_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        return record_scaling_audit_impl(
            self,
            scaling_audit=scaling_audit,
            conservation_details=conservation_details,
            cf_metadata=cf_metadata,
        )

    # =========================================================================
    # Public API (Consolidated from ScaleFacade)
    # =========================================================================

    def set_mode(self, mode: ScalingMode) -> None:
        """Make ``mode`` the default mode: its precision policy, tolerance and dtype.

        The facade ledger of a call without a mode, and the conservative regridder built for
        that call, then use ``coupling.scales.defaults.precision.<mode>.tolerance``.
        """
        reset_precision_policy_cache()
        policy = self._get_precision_policy(mode)
        self.precision_policy = policy
        self.default_mode = mode
        self.tensor_context = get_tensor_context(self.device, policy.dtype)
        self.enforce_conservation = bool(policy.enforce_conservation)

    def ledger_tolerance(self, mode: Optional[ScalingMode] = None) -> float:
        """Relative tolerance of the facade ledger of a call in ``mode`` (default mode if None).

        One source: ``coupling.scales.defaults.precision.<mode>.tolerance`` (dimensionless,
        finite, in (0, 1); the precision policy loader rejects any other value by name). The
        hierarchical aggregate/disaggregate audit and the regrid_variable covered-support audit
        of a call compare with the tolerance of that call's mode, and the conservative regridder
        that regrid_variable builds for the call carries the same tolerance.
        """
        if self.precision_policy is None:
            raise RuntimeError(
                "ScaleManager.ledger_tolerance needs initialize(): the tolerance is "
                f"{PRECISION_POLICY_NAMESPACE}.<mode>.tolerance of the call's mode."
            )
        resolved = self.default_mode if mode is None else mode
        if not isinstance(resolved, ScalingMode):
            raise TypeError(f"ledger_tolerance expects a ScalingMode, got {resolved!r}.")
        return float(self._get_precision_policy(resolved).tolerance)

    @property
    def conservation_tolerance(self) -> float:
        """Read-only facade ledger tolerance of the default mode (``ledger_tolerance()``).

        Undefined (AttributeError) before initialize(); it cannot be assigned, and
        ``configure`` rejects a conservation_tolerance setting (naming the key to change).
        """
        if self.precision_policy is None:
            raise AttributeError(
                "ScaleManager.conservation_tolerance is defined after initialize(): it is "
                f"{PRECISION_POLICY_NAMESPACE}.<default mode>.tolerance."
            )
        return self.ledger_tolerance()

    def verify_conservation(self, updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return diagnostics from the canonical ConservationEnforcer owner."""
        if self.conservation_enforcer is None:
            return {}
        return self.conservation_enforcer.get_conservation_report()

    def aggregate(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        source_scale: str,
        target_scale: str,
        variables: Optional[List[str]] = None,
        mode: Optional[ScalingMode] = None,
        strategy: Optional[ScalingStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        topology: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], ScalingAudit]:
        """
        Aggregate data from fine to coarse scale (Hierarchical/Unified).
        Delegates to HierarchicalScalingStrategy.
        """
        return self.hierarchical_strategy.aggregate(
            data, source_scale, target_scale, variables, mode, strategy, metadata, topology
        )

    def disaggregate(
        self,
        data: Union[Tensor, Dict[str, Tensor]],
        source_scale: str,
        target_scale: str,
        variables: Optional[List[str]] = None,
        mode: Optional[ScalingMode] = None,
        strategy: Optional[ScalingStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        topology: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], ScalingAudit]:
        """
        Disaggregate data from coarse to fine scale (Hierarchical/Unified).
        Delegates to HierarchicalScalingStrategy.
        """
        return self.hierarchical_strategy.disaggregate(
            data, source_scale, target_scale, variables, mode, strategy, metadata, topology
        )

    def _validate_spatial_inputs(self, var: Tensor, src_grid: GridDescriptor, tgt_grid: GridDescriptor, op: str) -> None:
        return validate_spatial_inputs_impl(self, var=var, src_grid=src_grid, tgt_grid=tgt_grid, op=op)

    def regrid_variable(
        self,
        var_name: str,
        var_value: Tensor,
        source_grid: Union[GridDescriptor, Dict[str, Any]],
        target_grid: Union[GridDescriptor, Dict[str, Any]],
        strategy: str,
        metadata: Optional[Dict[str, Any]] = None,
        mode: Optional[ScalingMode] = None,
        source_scale: str = "unknown",
        target_scale: str = "unknown",
    ) -> Tuple[Tensor, Dict[str, Any]]:
        return regrid_variable_impl(
            self,
            var_name=var_name,
            var_value=var_value,
            source_grid=source_grid,
            target_grid=target_grid,
            strategy=strategy,
            metadata=metadata,
            mode=mode,
            source_scale=source_scale,
            target_scale=target_scale,
        )

    def _sync_aggregation_metadata(self, var, metadata, method, rule, src, tgt, op, bounds=None):
        return sync_aggregation_metadata_impl(
            self,
            var=var,
            metadata=metadata,
            method=method,
            rule=rule,
            src=src,
            tgt=tgt,
            op=op,
            bounds=bounds,
        )

def build_quickstart_scale_config(parameter_manager: Optional[Any] = None) -> Dict[str, Any]:
    """
    Construct a minimal auto-scale configuration from the defaults namespace.

    This helper ensures that consumers use the centralized parameter database
    instead of hard-coding scale manager defaults.
    """
    return build_quickstart_scale_config_impl(
        defaults_namespace=DEFAULTS_NS,
        parameter_manager=parameter_manager,
    )
