"""
Functor Base Class - Unified Authority for RELIC Scientific Processes

This module provides the single, authoritative FunctorBase class that serves
as the foundation for all scientific process modules in the RELIC ecosystem.
It unifies functor implementations and
establishes a consistent interface for dynamic model assembly.

The design follows category theory principles where functors represent
morphisms (processes) that transform system states while preserving
structure and composability. This enables flexible, modular scientific
modeling with automatic differentiation support through PyTorch integration.

Architecture:
    - Inherits from torch.nn.Module for autodiff and GPU support
    - Dependency declaration system for topological execution ordering
    - Parameter management with validation and serialization
    - Comprehensive metadata system for scientific documentation
    - Performance monitoring and conservation law checking
    - Exception handling with informative error messages

References:
    - Mac Lane, S. (1998). Categories for the Working Mathematician
    - Goodfellow, I. et al. (2016). Deep Learning - Neural network architectures
    - Gamma, E. et al. (1994). Design Patterns - Abstract factory pattern
    - PyTorch Team (2019). PyTorch: An Imperative Style Deep Learning Library
    - Spivak, D.I. (2014). Category Theory for the Sciences
"""

import abc
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import struct
import sys
import time as time_module
import warnings
from collections.abc import Mapping
from enum import Enum
from types import CodeType, FunctionType, MethodType
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, TYPE_CHECKING, Union, cast

import torch
import torch.nn as nn
from torch import Tensor

# Import type aliases from abstractions
from ..abstractions import Device, FluxDict, StateDict
from .functor_types import (
    ConfigDict,
    ParameterDict,
    MetadataDict,
    ValidationErrors,
    DependencyList,
    DependencyDeclaration,
    ProcessCategory,
    ExecutionPhase,
    FunctorState,
    FunctorMetadata,
)
from .functor_exceptions import (
    FunctorException,
    ConfigurationError,
    DependencyError,
    ExecutionError,
    ConservationError,
)

if TYPE_CHECKING:
    from ..state.types import StateWriteMode, UpdateItem, UpdateMode

# Use lazy import to avoid circular dependency
PARAMETER_MANAGER_AVAILABLE = True  # Assume available, handle ImportError dynamically

logger = logging.getLogger(__name__)

_PIPELINE_TIMING_KEYS = (
    "state_validation",
    "forward",
    "local_conservation",
    "framework_validation",
    "total",
)


def _empty_pipeline_timings() -> Dict[str, float]:
    """Return the fixed Functor pipeline timing schema in seconds."""
    return {stage_name: 0.0 for stage_name in _PIPELINE_TIMING_KEYS}

def _get_parameter_manager() -> Optional[Any]:
    """
    Lazy import of parameter manager to avoid circular dependency.

    Returns:
        Parameter manager instance or None if not available
    """
    candidate_modules = (
        "relic.parameters.manager",
        "relic.parameters",
    )
    for module_name in candidate_modules:
        if importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        getter = getattr(module, "get_global_manager", None)
        if getter is not None:
            return getter()
    return None


def _load_runtime_symbol(module_name: str, symbol_name: str) -> Any:
    """Load runtime-only integration symbols without static import coupling."""
    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name, None)
    if symbol is None:
        raise ImportError(f"{symbol_name} not found in {module_name}")
    return symbol


class FunctorBase(nn.Module, abc.ABC):  # type: ignore[misc,unused-ignore]
    """
    Unified Abstract Base Class for Scientific Process Functors

    This is the single, authoritative base class for all scientific process
    modules in the RELIC ecosystem. It provides a complete interface for
    dynamic model assembly, automatic differentiation, parameter management,
    and scientific validation.

    The class inherits from torch.nn.Module to enable automatic differentiation,
    GPU acceleration, and integration with PyTorch's ecosystem. It follows
    category theory principles where functors preserve structure through
    well-defined morphisms between system states.

    Key Features:
        - Unified interface replacing prior functor base classes
        - Automatic differentiation support through PyTorch integration
        - Dependency declaration and resolution system
        - Comprehensive parameter management with validation
        - Scientific metadata system with literature references
        - Performance monitoring and execution timing
        - Conservation law enforcement and violation checking
        - Error handling with informative exception messages
        - Serialization support for checkpointing and deployment

    Architecture Pattern:
        Abstract Factory + Dependency Injection + Observer Pattern

    Mathematical Foundation:
        Functor F: StateSpace -> StateSpace
        F(s, t, dt, theta) = s + dt x process_rate(s, t, theta)
        where s in StateSpace, t in Time, theta in ParameterSpace

        Composition law: (F compose G)(s) = F(G(s))
        Identity law: id_functor(s) = s
        Associativity: (F compose G) compose H = F compose (G compose H)

    References:
        - Mac Lane, S. (1998). Categories for the Working Mathematician
        - Goodfellow, I. et al. (2016). Deep Learning - Chapter 6, Neural networks
        - Gamma, E. et al. (1994). Design Patterns - Abstract factory pattern
        - PyTorch Team (2019). PyTorch autograd system documentation
        - Spivak, D.I. (2014). Category Theory for the Sciences
    """

    def __init__(
        self,
        name: str,
        config: ConfigDict,
        metadata: Optional[FunctorMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize functor with configuration and metadata.

        Sets up the functor with all necessary parameters, dependencies,
        and metadata for scientific validation and documentation.
        Registers parameters with PyTorch for automatic differentiation.

        Parameters:
            name : str
                Unique identifier for this functor instance
                Must follow naming convention: [a-zA-Z][a-zA-Z0-9_-]*
            config : ConfigDict
                Configuration dictionary containing parameters and settings
                Structure: {"parameters": {...}, "dependencies": [...], ...}
            metadata : Optional[FunctorMetadata]
                Scientific documentation and process metadata
                Auto-generated if not provided

        Raises:
            ValueError: If name or config is invalid
            ConfigurationError: If configuration validation fails

        Mathematical Expression:
            Initialization mapping: F_init : Config x Metadata -> FunctorInstance
            Parameter resolution: theta is read lazily from ParameterManager
            Dependency graph: G = (V, E) where V are functors, E are dependencies
        """
        injected_parameter_manager = kwargs.pop("parameter_manager", None)
        runtime_device = kwargs.pop("device", None)
        super().__init__(**kwargs)

        # Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Functor name must be a non-empty string")
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid functor name format: {name}")
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        # Core attributes
        self.name = name
        # Store original config for ConfigProxy access
        self._original_config = config.copy() if config else {}
        # Import ConfigProxy here to avoid circular import
        from ..config_proxy import ConfigProxy
        self.config = ConfigProxy(self)
        self.state = FunctorState.UNINITIALIZED
        # Pre-registration placeholder: CPU is safe default.
        # The actual device is assigned by RELICSimulator at registration
        # via _maybe_migrate_functor_device(), which honors SimulatorConfig["device"]
        # and per-functor overrides in SimulatorConfig["device_overrides"].
        self.device = torch.device(runtime_device) if runtime_device is not None else torch.device("cpu")

        # Initialize metadata with defaults if not provided
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                description=f"Scientific process functor: {name}"
            )
        self.metadata = metadata

        # Dependency management
        self._dependencies: DependencyList = []
        self._dependents: DependencyList = []  # Functors that depend on this one

        # Performance tracking
        self._execution_count: int = 0
        self._total_execution_time: float = 0.0
        self._last_execution_time: float = 0.0
        self._performance_stats: Dict[str, float] = {
            'min_execution_time': float('inf'),
            'max_execution_time': 0.0,
            'mean_execution_time': 0.0
        }
        self._performance_profiling_enabled: bool = False
        self._last_pipeline_timings_s: Dict[str, float] = _empty_pipeline_timings()
        self._pipeline_timing_totals_s: Dict[str, float] = _empty_pipeline_timings()
        self._pipeline_timing_count: int = 0
        # Guard flag: True only while executing through FunctorBase.__call__.
        # This prevents duplicate local conservation checks when forward()
        # delegates to a shared helper.
        self._inside_canonical_call: bool = False

        # Error tracking
        self._last_error: Optional[Exception] = None
        self._error_count: int = 0

        # Parameter management
        self.parameter_values: ParameterDict = {}
        self._parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self._parameter_units: Dict[str, str] = {}
        self._parameter_requires_grad: Dict[str, bool] = {}
        # Cached parameter manager instance for dependency injection
        self._parameter_manager: Optional[Any] = injected_parameter_manager

        # Process configuration
        self._load_configuration()

        # Parameter database is the source of truth. Auto-registration is opt-in to avoid
        # creating redundant dynamic keys that bypass the YAML parameter system.
        if bool(self._original_config.get("auto_register_parameters_to_manager", False)):
            self.register_parameters_to_manager()

        # Framework validation strictness is a config setting, checked here so
        # an invalid value fails at construction (see _strict_validation_enabled).
        self._strict_validation_enabled()

        # Prepare validation engine and register metadata-driven rules
        self._validation_engine: Any = None
        # Metadata-derived validation rules are part of the governed call; a
        # registration failure propagates instead of leaving the rules missing.
        register_functor_validations = cast(
            Callable[..., Any],
            _load_runtime_symbol(
                "relic.validation.framework.functor_integration",
                "register_functor_validations",
            ),
        )
        register_functor_validations(self.name, self.metadata, overwrite=True)

        # Update state
        self.state = FunctorState.INITIALIZED

    def _load_configuration(self) -> None:
        """Load and validate configuration parameters."""
        try:
            # Load parameters from original config (not proxy)
            params = self._original_config.get('parameters', {})
            for param_name, param_value in params.items():
                if isinstance(param_value, dict):
                    # Detailed parameter specification. The value is declared.
                    # Units and bounds are recorded only when declared: by the
                    # entry, or else by the parameter's database specification.
                    # A config value without units is read in the canonical units
                    # (get_parameter_value); units are required where they are
                    # consumed (register_parameters_to_manager), never invented.
                    if 'value' not in param_value:
                        raise ConfigurationError(
                            f"{self.name}: config parameter '{param_name}' declares no 'value'"
                        )
                    spec = self._database_spec(param_name)
                    units = param_value.get('units')
                    if units is None and spec is not None:
                        units = getattr(spec, "units", None) or None
                    self.parameter_values[param_name] = param_value['value']
                    if units is not None:
                        self._parameter_units[param_name] = units
                    bounds = param_value.get('bounds')
                    if bounds is None and spec is not None:
                        bounds = getattr(spec, "bounds", None)
                    if bounds is not None:
                        self._parameter_bounds[param_name] = (float(bounds[0]), float(bounds[1]))
                    self._parameter_requires_grad[param_name] = bool(param_value.get('requires_grad', False))
                else:
                    # Simple parameter specification
                    self.parameter_values[param_name] = param_value

            # Also check for parameters at root level of config (for simple test cases)
            for key, value in self._original_config.items():
                if key not in ['parameters', 'dependencies'] and not key.startswith('_'):
                    # Only add if not already in parameter_values
                    if key not in self.parameter_values:
                        self.parameter_values[key] = value

            # Load dependencies from original config (not proxy)
            dependencies = self._original_config.get('dependencies', [])
            if isinstance(dependencies, list):
                self._dependencies = dependencies.copy()
            elif isinstance(dependencies, str):
                self._dependencies = [dependencies]
            else:
                raise ConfigurationError(f"Invalid dependencies format: {type(dependencies)}")

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            raise ConfigurationError(f"Configuration loading failed for {self.name}: {str(e)}") from e

    def _database_spec(self, param_name: str) -> Any:
        """Database specification of a config parameter of this Functor, or None.

        None when no manager is configured or the parameter has no
        specification; retrieval failures propagate.
        """
        manager = self.get_parameter_manager()
        if manager is None:
            return None
        full_key, _, _ = self._resolve_parameter_lookup(param_name, None)
        return manager.get_parameter(full_key)

    def get_parameter_manager(self) -> Optional[Any]:
        """
        Lazily retrieve and cache the unified parameter manager instance.

        Returns:
            Optional[Any]: Parameter manager instance if available, otherwise None.
        """
        if not PARAMETER_MANAGER_AVAILABLE:
            return None

        if self._parameter_manager is None:
            self._parameter_manager = _get_parameter_manager()
        return self._parameter_manager

    @property
    def parameter_manager(self) -> Optional[Any]:
        """Property access for the cached parameter manager."""
        return self.get_parameter_manager()

    @parameter_manager.setter
    def parameter_manager(self, manager: Optional[Any]) -> None:
        """Allow dependency injection of a parameter manager instance."""
        self._parameter_manager = manager

    def wrap_update(
        self,
        value: Any,
        mode: "Optional[UpdateMode]" = None,
        write_mode: "Optional[StateWriteMode]" = None,
        per_timestep: bool = False,
    ) -> "UpdateItem":
        """
        Create an UpdateItem with an explicit application mode.

        Parameters
        ----------
        value : Any
            State update payload (tensor or scalar).
        mode : UpdateMode, optional
            Application mode (accumulate or replace), by default ACCUMULATE.
        write_mode : StateWriteMode, optional
            State write semantics (mutable, immutable, derived).
        per_timestep : bool, optional
            True when ``value`` is already integrated over the current dt.

        Returns
        -------
        UpdateItem
            Wrapped update with explicit mode for StateManager.apply_updates.
        """
        from ..state.types import UpdateItem, UpdateMode, StateWriteMode

        resolved_mode = mode or UpdateMode.ACCUMULATE
        resolved_write_mode = write_mode if isinstance(write_mode, StateWriteMode) else write_mode
        return UpdateItem(
            value=value,
            mode=resolved_mode,
            write_mode=resolved_write_mode,
            per_timestep=bool(per_timestep),
        )

    @abc.abstractmethod
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Declare functor dependencies for execution ordering.

        Returns the list of functor names that must be executed before
        this functor during each time step. Used by the simulator for
        topological sorting and cycle detection.

        The four bridge classes implement this method as the instance
        declaration (``config["dependencies"]``, ``set_dependencies`` and
        ``add_dependency``). A concrete class that adds a static declaration
        must return ``self._merge_declared_dependencies(static)`` so the
        instance declaration is merged, never replaced or dropped.

        Returns:
            DependencyDeclaration
                List of functor names this functor depends on
                Empty list if no dependencies exist
                Order matters for execution priority
                Dict format: {"required": [...], "optional": [...], "lagged": [...]} supported by DependencyGraphResolver

        Example:
            >>> class PhotosynthesisFunctor(BiologicalProcessFunctor):
            ...     def get_dependencies(self) -> DependencyDeclaration:
            ...         return self._merge_declared_dependencies(
            ...             ["light_model", "temperature_model", "co2_model"]
            ...         )
            >>> class OptionalETFunctor(TransportMechanismFunctor):
            ...     def get_dependencies(self) -> DependencyDeclaration:
            ...         return self._merge_declared_dependencies(
            ...             {"required": ["soil_moisture"], "optional": ["penman_monteith_et"]}
            ...         )
            >>> class LaggedBoundaryFunctor(PhysicalFieldFunctor):
            ...     def get_dependencies(self) -> DependencyDeclaration:
            ...         return self._merge_declared_dependencies(
            ...             {"lagged": ["evapotranspiration_flux"]}
            ...         )

        Mathematical Expression:
            Dependency relation: F_i -> F_j if F_j depends on F_i
            Execution order: pi = topsort(G) where G = (V, E) is dependency graph
            Constraint: G must be acyclic (DAG property)
            Complexity: O(V + E) using Kahn's algorithm

        References:
            - Kahn, A.B. (1962). Topological sorting of large networks. CACM
            - Tarjan, R. (1972). Depth-first search and linear graph algorithms
            - Cormen, T.H. et al. (2009). Introduction to Algorithms - Graph algorithms
        """

    @abc.abstractmethod
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """
        Forward pass for functor execution - core scientific computation.

        Computes process rates, state updates, and flux calculations based
        on current system state and environmental conditions. This method
        implements the scientific process represented by this functor and
        must preserve conservation laws and physical constraints.

        Parameters:
            state : StateDict
                Current system state variables with gradient tracking
                Keys: state variable names (e.g., "carbon_pool", "temperature")
                Values: PyTorch tensors with appropriate units
            time : float
                Current simulation time since start
                Units: [s]
            dt : float
                Time step size for numerical integration
                Units: [s], must be positive
            **kwargs
                Integration controls of the governed call (for example
                calendar_type, time_origin, time_integration_method) and
                operation arguments such as "boundary_conditions"; scientific
                inputs come only from the state (an "environment" keyword is
                rejected by the governed call)

        Returns:
            FluxDict
                Process rates and state variable updates
                Keys: flux variable names (e.g., "carbon_flux", "heat_flux")
                Values: update rates [state_units s^-^1]
                Sign convention: positive = increase, negative = decrease

        Raises:
            ValueError: If required state variables are missing
            ExecutionError: If computation fails or becomes unstable
            ConservationError: If conservation laws are violated

        Mathematical Expression:
            Process operator: F: S x T x R^+ -> DeltaS
            State update: s(t+dt) = s(t) + dt x F(s(t), t, dt, theta)

            Conservation constraint: Sigma F_i(s,t,dt) = 0 for conserved quantities
            Stability constraint: ||F(s,t,dt)|| < M for bounded operator F

            Numerical integration methods:
            - Explicit Euler: s^(n+1) = s^n + dt x F(s^n, t^n, dt)
            - Implicit Euler: s^(n+1) = s^n + dt x F(s^(n+1), t^(n+1), dt)
            - Runge-Kutta 4: Higher-order accuracy with multiple evaluations

            Time scale separation: ensure dt << 1/lambda_max where lambda_max is fastest eigenvalue

        References:
            - Hairer, E. et al. (1993). Solving Ordinary Differential Equations I. Springer
            - Press, W.H. et al. (2007). Numerical Recipes - Integration of ODEs
            - Farquhar, G.D. et al. (1980). A biochemical model of photosynthetic CO2 assimilation
        """

    def set_dependencies(self, dependencies: DependencyList) -> None:
        """
        Set the dependency list for this functor.

        Parameters:
            dependencies : DependencyList
                List of functor names this functor depends on

        Raises:
            TypeError: If dependencies is not a list
            ValueError: If any dependency name is invalid
        """
        if not isinstance(dependencies, list):
            raise TypeError("Dependencies must be a list of strings")

        for dep in dependencies:
            if not isinstance(dep, str) or not dep.strip():
                raise ValueError("All dependencies must be non-empty strings")

        self._dependencies = dependencies.copy()

    def add_dependency(self, functor_name: str) -> None:
        """
        Add a single dependency to this functor.

        Parameters:
            functor_name : str
                Name of functor to depend on
        """
        if functor_name not in self._dependencies:
            self._dependencies.append(functor_name)

    def remove_dependency(self, functor_name: str) -> None:
        """
        Remove a dependency from this functor.

        Parameters:
            functor_name : str
                Name of dependency to remove
        """
        if functor_name in self._dependencies:
            self._dependencies.remove(functor_name)

    _DEPENDENCY_CATEGORIES = ("required", "optional", "lagged")

    def _merge_declared_dependencies(
        self,
        static: Optional[DependencyDeclaration] = None,
    ) -> DependencyDeclaration:
        """
        Merge a class-level static declaration with the instance declarations.

        This is the single owner of the dependency merge used by the four
        bridge defaults and by every concrete ``get_dependencies`` override:
        an override returns ``self._merge_declared_dependencies(static)`` so
        that names declared on the instance through ``config["dependencies"]``,
        ``set_dependencies(...)`` or ``add_dependency(...)`` are never dropped.

        Merge rule (DependencyGraphResolver structure):
            D = D_static U D_instance. An instance name carries no category and
            joins ``required`` unless ``D_static`` already lists it; then the
            static category (``required``, ``optional`` or ``lagged``) is kept,
            so an instance declaration cannot turn a lagged read into a
            same-step edge.

        Parameters:
            static : Optional[DependencyDeclaration]
                None (bridge default), a list of required Functor names, or a
                mapping whose keys are a subset of ``required``, ``optional``
                and ``lagged`` with list values.

        Returns:
            DependencyDeclaration
                A new list when ``static`` is None or a list; otherwise a new
                mapping with the keys of ``static`` in their order, plus
                ``required`` when an instance name has to be added. Static
                names come first, then instance names in declaration order;
                duplicates are removed.

        Raises:
            TypeError: If ``static`` or a category value has the wrong type, or
                a declared name is not a string.
            ValueError: If ``static`` has an unknown category key or a declared
                name is empty.
        """

        def _names(values: Any, source: str) -> list:
            if not isinstance(values, (list, tuple)):
                raise TypeError(
                    f"{self.name}: {source} must be a list of Functor names, "
                    f"got {type(values).__name__}"
                )
            names: list = []
            for value in values:
                if not isinstance(value, str):
                    raise TypeError(
                        f"{self.name}: {source} contains a non-string dependency "
                        f"{value!r}"
                    )
                if not value.strip():
                    raise ValueError(f"{self.name}: {source} contains an empty dependency name")
                if value not in names:
                    names.append(value)
            return names

        instance_names = _names(self._dependencies, "instance-declared dependencies")

        if static is None or isinstance(static, (list, tuple)):
            merged = _names(static if static is not None else [], "static dependencies")
            merged.extend(name for name in instance_names if name not in merged)
            return merged

        if not isinstance(static, dict):
            raise TypeError(
                f"{self.name}: static dependency declaration must be a list or a "
                f"mapping, got {type(static).__name__}"
            )
        unknown = [key for key in static if key not in self._DEPENDENCY_CATEGORIES]
        if unknown:
            raise ValueError(
                f"{self.name}: unknown dependency categories {unknown}; expected a "
                f"subset of {list(self._DEPENDENCY_CATEGORIES)}"
            )
        merged_mapping: Dict[str, list] = {
            key: _names(values, f"static '{key}' dependencies")
            for key, values in static.items()
        }
        declared = {name for names in merged_mapping.values() for name in names}
        additions = [name for name in instance_names if name not in declared]
        if additions:
            merged_mapping.setdefault("required", []).extend(additions)
        return merged_mapping

    @staticmethod
    def _infer_namespace_from_module(module: str) -> Optional[str]:
        """
        Infer a stable parameter namespace from the functor's Python module path.

        This avoids dynamic namespaces like '<category>.<instance_name>' which break
        strict lazy loading and parameter override determinism.
        """
        if not module or not isinstance(module, str):
            return None
        parts = module.split(".")
        if len(parts) >= 4 and parts[0] == "relic" and parts[1] == "scientific":
            sphere = parts[2]
            subsystem = parts[3]
            if sphere and subsystem and subsystem != "__init__":
                return f"{sphere}.{subsystem}"
            if sphere:
                return sphere
        if len(parts) >= 3 and parts[0] == "relic":
            domain = parts[1]
            sub = parts[2]
            if domain in {"coupling", "transport", "validation", "stochastic", "optimization"}:
                if sub and sub != "__init__":
                    return f"{domain}.{sub}"
                return domain
        return None

    def _default_parameter_namespace(self) -> str:
        """
        Resolve the default namespace used for parameter lookups when none is provided.
        """
        metadata = getattr(self, "metadata", None)
        for attr in ("parameter_namespace", "namespace"):
            if metadata is not None:
                value = getattr(metadata, attr, None)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            value = getattr(self, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        inferred = self._infer_namespace_from_module(getattr(self.__class__, "__module__", ""))
        if inferred:
            return inferred

        return str(self.metadata.category.to_namespace_prefix())

    def _resolve_parameter_lookup(
        self,
        name: Any,
        namespace: Optional[str],
    ) -> Tuple[str, str, Optional[str]]:
        """Normalize name/namespace inputs into a fully qualified parameter key."""
        query_name = name
        if isinstance(query_name, Enum):
            query_name = query_name.value
        elif hasattr(query_name, "value") and isinstance(query_name.value, str):
            query_name = query_name.value

        if not isinstance(query_name, str):
            raise TypeError(f"Parameter name must be a string or Enum, got {type(name)!r}")

        query_namespace = namespace
        if isinstance(query_namespace, Enum):
            query_namespace = query_namespace.value
        if isinstance(query_namespace, str) and not query_namespace.strip():
            query_namespace = None

        if query_namespace is None and "." in query_name:
            # If the incoming dotted name already exists as a fully qualified key in the
            # ParameterManager, prefer interpreting it as such. This keeps strict support
            # for hierarchical keys inside a functor namespace (e.g., 'maintenance.rm_leaf_25')
            # while allowing cross-module access to explicitly registered keys such as
            # 'test.integration.parameter' in integration tests.
            # None means no manager is configured; retrieval failures propagate.
            pm = self.get_parameter_manager()

            if pm is not None:
                specs = getattr(pm, "parameters", None)
                if isinstance(specs, dict) and query_name in specs:
                    ns_candidate, key_candidate = query_name.rsplit(".", 1)
                    if ns_candidate and key_candidate:
                        query_namespace = ns_candidate
                        query_name = key_candidate

            if query_namespace is None:
                # Delegate root validation to the parameter database namespace owner.
                # Relative hierarchical names such as 'maintenance.rm_leaf_25' remain
                # scoped to the Functor because their namespace is not database-registered.
                from ...parameters.namespaces import is_declared_namespace

                ns_candidate, key_candidate = query_name.rsplit(".", 1)
                if (
                    ns_candidate
                    and key_candidate
                    and is_declared_namespace(ns_candidate, pm)
                ):
                    query_namespace = ns_candidate
                    query_name = key_candidate

        if query_namespace is None:
            query_namespace = self._default_parameter_namespace()

        full_key = f"{query_namespace}.{query_name}" if query_namespace else query_name
        return full_key, query_name, query_namespace

    def _prepare_spec_units_bounds(
        self,
        full_key: str,
        units: Optional[str],
        bounds: Optional[Tuple[float, float]],
    ) -> Tuple[Any, Optional[str], Optional[Tuple[float, float]]]:
        """Load spec and derive default units/bounds using existing behavior."""
        spec = None
        param_manager = self.get_parameter_manager()
        if param_manager is not None:
            spec = param_manager.get_parameter(full_key)

        if spec is None:
            return spec, units, bounds

        if units is None and getattr(spec, "units", None):
            units = spec.units

        if bounds is None and getattr(spec, "bounds", None) is not None:
            bounds_value = spec.bounds
            if isinstance(bounds_value, list):
                bounds_value = tuple(bounds_value)
            bounds = bounds_value

            if (
                bounds is not None
                and units
                and getattr(spec, "units", None)
                and spec.units != units
            ):
                from ...data.standards.unit_converter import (
                    unit_converter as authoritative_unit_converter,
                    ConversionContext,
                )

                if isinstance(bounds, tuple) and len(bounds) == 2:
                    if all(isinstance(v, (int, float)) for v in bounds):
                        try:
                            converted_bounds = cast(
                                Tensor,
                                authoritative_unit_converter.convert(
                                    torch.tensor(bounds, dtype=torch.float64),
                                    "generic",
                                    spec.units,
                                    units,
                                    ConversionContext(allow_identity=True),
                                ),
                            )
                            bounds = (
                                float(converted_bounds[0].item()),
                                float(converted_bounds[1].item()),
                            )
                        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                            raise ValueError(
                                f"Cannot convert bounds for {full_key} from {spec.units} "
                                f"to required units {units}: {exc}"
                            ) from exc

        return spec, units, bounds

    def _resolve_value_from_source_order(
        self,
        *,
        source_order: Tuple[str, ...],
        full_key: str,
        query_name: str,
        original_name: str,
        units: Optional[str],
        expected_type: Optional[type],
        spec: Any,
    ) -> Tuple[Any, Optional[str], Any]:
        """Resolve parameter value by ordered source lookup."""
        value = None
        found_source = None
        resolved_spec = spec

        for source in source_order:
            if source == "manager" and PARAMETER_MANAGER_AVAILABLE:
                param_manager = self.get_parameter_manager()
                if param_manager is None:
                    continue
                resolved_spec = param_manager.get_parameter(full_key)
                if resolved_spec is not None and getattr(resolved_spec, "requires_grad", False):
                    value = param_manager.get(
                        full_key,
                        units=units,
                        expected_type=torch.Tensor,
                        required=True,
                    )
                    found_source = "manager"
                    break
                # The manager represents an absent optional value with None.
                # Validation, conversion, and backend failures are not absence.
                value = param_manager.get(
                    full_key,
                    units=units,
                    expected_type=expected_type,
                )
                if value is not None:
                    found_source = "manager"
                    break

            elif source == "config":
                # Check in original config - first try parameters section, then root level
                if self._original_config:
                    config_value = None
                    # First try parameters section
                    if "parameters" in self._original_config:
                        params = self._original_config["parameters"]
                        for candidate in (query_name, original_name):
                            if candidate in params:
                                config_value = params[candidate]
                                break
                    # If not found, try root level
                    if config_value is None:
                        for candidate in (query_name, original_name):
                            if candidate in self._original_config:
                                config_value = self._original_config.get(candidate)
                                if config_value is not None:
                                    break

                    if config_value is not None:
                        # If value is a dict with 'value' key, extract it
                        if isinstance(config_value, dict) and "value" in config_value:
                            value = config_value["value"]
                            if "units" in config_value:
                                from copy import copy
                                from ...parameters.ops.unit_validation import (
                                    _is_dimensionless_equivalent,
                                    _normalize_unit_alias,
                                    convert_parameter_units_impl,
                                )

                                source_units = config_value["units"]
                                if not isinstance(source_units, str) or not source_units.strip():
                                    raise ValueError(f"Configuration units for {full_key} must be a nonempty string")
                                if resolved_spec is None or not resolved_spec.units:
                                    raise ValueError(f"Configuration units for {full_key} require canonical parameter metadata")
                                normalized_source = _normalize_unit_alias(source_units)
                                normalized_target = _normalize_unit_alias(resolved_spec.units)
                                source_spec = copy(resolved_spec)
                                source_spec.units = (
                                    resolved_spec.units
                                    if _is_dimensionless_equivalent(normalized_source) and _is_dimensionless_equivalent(normalized_target)
                                    else source_units
                                )
                                # Source-to-canonical conversion belongs to each read,
                                # so mutable inputs and repeated backward calls use a
                                # fresh graph. Stored specs and caller data stay intact.
                                value = convert_parameter_units_impl(
                                    self.get_parameter_manager(), value, source_spec,
                                    resolved_spec.units,
                                )
                        else:
                            value = config_value
                        found_source = "config"
                        break

        return value, found_source, resolved_spec

    def _is_parameter_trainable(self, full_key: str, spec: Any) -> bool:
        """Determine whether parameter should preserve tensor form for training."""
        if not PARAMETER_MANAGER_AVAILABLE:
            return False

        param_manager = self.get_parameter_manager()
        if param_manager is None:
            return False

        resolved_spec = spec
        if resolved_spec is None:
            resolved_spec = param_manager.get_parameter(full_key)
        return bool(resolved_spec is not None and getattr(resolved_spec, "requires_grad", False))

    @staticmethod
    def _expected_type_allows_tensor(expected_type: Optional[type]) -> bool:
        """Check whether expected_type explicitly allows torch.Tensor."""
        if expected_type is None:
            return False
        if isinstance(expected_type, tuple):
            return any(
                isinstance(candidate, type) and issubclass(candidate, Tensor)
                for candidate in expected_type
                if isinstance(candidate, type)
            )
        return isinstance(expected_type, type) and issubclass(expected_type, Tensor)

    @staticmethod
    def _maybe_convert_scalar_value(val: Any, convert_to_scalar: bool) -> Tuple[Any, bool]:
        """Convert single-element tensor to scalar when scalar expectation is enforced."""
        if convert_to_scalar and isinstance(val, torch.Tensor) and val.numel() == 1:
            return val.detach().cpu().item(), True
        return val, False

    def _validate_parameter_value(
        self,
        *,
        value: Any,
        found_source: Optional[str],
        full_key: str,
        name: str,
        units: Optional[str],
        bounds: Optional[Tuple[float, float]],
        validate: bool,
        expected_type: Optional[type],
        convert_to_scalar: bool,
        converted_to_scalar: bool,
    ) -> Tuple[Any, bool]:
        """Run unit/type/bounds validation without replacing parameter ownership."""
        if value is None:
            return value, converted_to_scalar

        # Check units
        if units and PARAMETER_MANAGER_AVAILABLE:
            param_manager = self.get_parameter_manager()
            if param_manager is None:
                raise RuntimeError("Parameter manager unavailable for unit validation")
            spec = param_manager.get_parameter(full_key)
            if spec and spec.units and spec.units != units and found_source != "manager":
                from ...data.standards.unit_converter import (
                    unit_converter as authoritative_unit_converter,
                    ConversionContext,
                )
                try:
                    value = authoritative_unit_converter.convert(
                        value,
                        "generic",
                        spec.units,
                        units,
                        ConversionContext(allow_identity=True),
                    )
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                    logger.error(
                        "Unit conversion failed for parameter %s from %s to %s",
                        full_key,
                        spec.units,
                        units,
                    )
                    raise ValueError(
                        f"Failed to convert parameter {full_key} from {spec.units} to {units}"
                    ) from exc

        if not validate:
            return value, converted_to_scalar

        value, scalarized_now = self._maybe_convert_scalar_value(value, convert_to_scalar)
        converted_to_scalar = converted_to_scalar or scalarized_now

        if expected_type is not None and value is not None and not isinstance(value, expected_type):
            type_name = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else "/".join(t.__name__ for t in expected_type)
            )
            raise ValueError(
                f"Parameter {full_key} must be of type {type_name}, got {type(value).__name__}"
            )

        # Check bounds
        # A declaration of one (minimum, maximum) pair per component (a
        # list-valued parameter) is judged by the single componentwise rule of
        # the parameter system, whatever the type of the value; a flat
        # (lower, upper) interval keeps the scalar and Tensor check below.
        from ...parameters.spec import componentwise_bounds, componentwise_bounds_errors

        component_bounds = componentwise_bounds(bounds)
        if component_bounds is not None:
            reasons = componentwise_bounds_errors(value, component_bounds)
            if reasons:
                raise ValueError(
                    f"Parameter {name} value {value!r} violates its componentwise bounds "
                    f"{bounds}: " + "; ".join(reasons)
                )
        elif bounds and isinstance(value, (int, float, Tensor)):
            if isinstance(value, Tensor):
                in_bounds = bool(torch.all((value >= bounds[0]) & (value <= bounds[1])).item())
            else:
                in_bounds = bounds[0] <= float(value) <= bounds[1]
            if not in_bounds:
                raise ValueError(
                    f"Parameter {name} value {value} outside bounds {bounds}"
                )

        return value, converted_to_scalar

    @staticmethod
    def _resolve_spec_dtype(spec_value: Any) -> Optional[torch.dtype]:
        """Map parameter spec dtype metadata to torch dtype."""
        if spec_value is None:
            return None
        dtype_str = str(spec_value).lower()
        if dtype_str.startswith("<class '") and dtype_str.endswith("'>"):
            dtype_str = dtype_str[len("<class '"):-2]
        if dtype_str in {"bool", "boolean"}:
            return torch.bool
        if dtype_str in {"int32", "int"}:
            return torch.int32
        if dtype_str in {"int64", "long"}:
            return torch.int64
        if dtype_str in {"float64", "double"}:
            return torch.float64
        if dtype_str in {"float32", "float"}:
            return torch.float32
        return None

    def _tensorize_parameter_value(self, value: Any, spec: Any) -> Any:
        """Keep owned bindings; tensorize other values using the declared dtype."""
        # Architecture SSOT sections 4.2 and 4.4.2: the manager owns each
        # learnable binding, including its runtime dtype and device. Copying it
        # here would hide its identity from registration and checkpoint owners.
        # Unit-conversion views and private config tensors are not owned bindings.
        if self.manager_binding_key(value) is not None:
            return value
        tensor_dtype = self._resolve_spec_dtype(getattr(spec, "dtype", None))

        # Convert to tensor if numeric
        if isinstance(value, (int, float)):
            device = self.device
            resolved_dtype = tensor_dtype or torch.get_default_dtype()
            value = torch.tensor(float(value), device=device, dtype=resolved_dtype)
        elif isinstance(value, Tensor):
            device = self.device
            if value.device != device:
                value = value.to(device)
            if tensor_dtype is not None and value.dtype != tensor_dtype:
                value = value.to(dtype=tensor_dtype)

        return value

    def get_parameter_value(
        self,
        name: str,
        *,
        units: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
        validate: bool = True,
        namespace: Optional[str] = None,
        source_order: Tuple[str, ...] = ("manager", "config"),
        expected_type: Optional[type] = None,
        required: bool = False,
    ) -> Any:
        """
        Enhanced parameter value getter with validation and registration.

        This method provides a unified interface for parameter access with
        support for units, bounds, validation, and automatic registration.

        Parameters:
            name : str
                Parameter name
            units : Optional[str]
                Expected units for validation
            bounds : Optional[Tuple[float, float]]
                Expected bounds for validation
            validate : bool
                Whether to validate the parameter
            namespace : Optional[str]
                Override namespace (default uses functor category)
            source_order : Tuple[str, ...]
                Priority order of the value sources, a non-empty tuple of
                "manager" (the YAML parameter database) and "config" (this
                Functor's config); any other name raises ValueError. There is
                no caller-supplied default: the YAML database owns values.
            expected_type : Optional[type]
                Optional strict type check; raises TypeError if value is not of this type.
            required : bool
                Whether parameter is required (default: False)
                If True and parameter not found, raises ValueError; otherwise
                an absent parameter is None

        Returns:
            Any: Parameter value (as Tensor if numeric), or None when absent
                and not required

        Examples:
            >>> value = self.get_parameter_value(
            ...     "vcmax25",
            ...     units="umol CO2 m-2 s-1",
            ...     bounds=(10, 200),
            ...     required=True,
            ... )
        """
        if (
            not isinstance(source_order, tuple)
            or not source_order
            or any(source not in ("manager", "config") for source in source_order)
            or len(set(source_order)) != len(source_order)
        ):
            raise ValueError(
                f"{self.name}: source_order must be a non-empty tuple of distinct "
                f"'manager' / 'config' entries, got {source_order!r}"
            )
        # Accept both string and ParameterNamespace enum inputs
        from ...parameters.namespaces import resolve_namespace

        namespace = resolve_namespace(namespace)

        full_key, query_name, query_namespace = self._resolve_parameter_lookup(name, namespace)

        spec, units, bounds = self._prepare_spec_units_bounds(full_key, units, bounds)
        value, found_source, spec = self._resolve_value_from_source_order(
            source_order=source_order,
            full_key=full_key,
            query_name=query_name,
            original_name=name,
            units=units,
            expected_type=expected_type,
            spec=spec,
        )

        if value is None and required:
            raise ValueError(f"Required parameter '{full_key}' not found in any source")

        is_trainable = self._is_parameter_trainable(full_key, spec)
        allows_tensor = self._expected_type_allows_tensor(expected_type)
        convert_to_scalar = expected_type is not None and not allows_tensor and not is_trainable

        value, converted_to_scalar = self._maybe_convert_scalar_value(value, convert_to_scalar)
        value, converted_to_scalar = self._validate_parameter_value(
            value=value,
            found_source=found_source,
            full_key=full_key,
            name=name,
            units=units,
            bounds=bounds,
            validate=validate,
            expected_type=expected_type,
            convert_to_scalar=convert_to_scalar,
            converted_to_scalar=converted_to_scalar,
        )
        # Preserve boolean outputs when explicitly requested
        if isinstance(value, bool) or expected_type is bool:
            return bool(value)

        if converted_to_scalar or (convert_to_scalar and isinstance(value, (int, float))):
            return value

        return self._tensorize_parameter_value(value, spec)

    def _get_state_variable(self,
                           state: StateDict,
                           key: str,
                           units: Optional[str] = None,
                           validate_range: Optional[Tuple[float, float]] = None,
                           dtype: Optional[torch.dtype] = None) -> Tensor:
        """
        Get a state variable.

        Parameters
        ----------
        state : StateDict
            Current state dictionary.
        key : str
            State variable key to retrieve.
        units : Optional[str]
            Expected units of the variable (CF-1.8); documents the reader's
            contract, the value is returned unchanged.
        validate_range : Optional[Tuple[float, float]]
            Declared domain of the value (min, max), inclusive; a state value with
            any element outside it (NaN included) raises and is never clipped.
        dtype : Optional[torch.dtype]
            Target data type (default: float32).

        Returns
        -------
        Tensor
            State variable value as a properly formatted tensor.

        Raises
        ------
        KeyError
            If the variable is not in the state (NOT FOUND); there is no default.
        ValueError
            If any element of the state value is outside validate_range.

        Scientific Mathematical Expressions
        ----------------------------------
        v = state[key] if key in state else raise KeyError

        References
        ----------
        RELIC parameter system policy (YAML SSOT, 2025).
        """
        if key not in state:
            raise KeyError(
                f"State variable '{key}' not found in state dict: NOT FOUND. "
                f"Supply '{key}' as state or forcing; a reader has no default value."
            )
        value = self._ensure_tensor(state[key], dtype=dtype)

        # References: RELIC fail-fast input policy: a reader never modifies
        #   the state value it reads.
        # Mathematical expression: raise unless lo <= v_i <= hi for every
        #   element v_i; otherwise return v unchanged.
        # Parameter meanings: v the state value; (lo, hi) = validate_range,
        #   the declared domain of the variable.
        # Units: v, lo and hi share the units of the variable.
        # Domain/constraints: NaN is outside every range; no clipping.
        # Sign convention: none (a domain check, not a flux).
        if validate_range is not None:
            lower, upper = validate_range
            outside = ~((value >= lower) & (value <= upper))
            if bool(outside.any()):
                observed = value.detach()
                raise ValueError(
                    f"State variable '{key}' has values outside its declared range "
                    f"[{lower}, {upper}] (observed min {float(observed.min())}, "
                    f"max {float(observed.max())}); the value is not clipped."
                )

        return value

    def _ensure_tensor(self, value: Any, dtype: Optional[torch.dtype] = None) -> Tensor:
        """
        Ensure value is a tensor with proper device and dtype.

        Parameters:
            value : Any
                Value to convert
            dtype : Optional[torch.dtype]
                Target dtype (default: float32)

        Returns:
            Tensor: Properly formatted tensor
        """
        if isinstance(value, nn.Parameter):
            # A learnable value is the ParameterManager-owned binding (SSOT).
            # Never rewrite its dtype or device in place: that would change
            # the global binding for every consumer and break snapshot
            # restore. Return the Parameter itself when it already matches,
            # otherwise a differentiable converted view whose autograd graph
            # still leads back to the owned Parameter.
            tensor = value
            device = self.device
            if (dtype is None or tensor.dtype == dtype) and tensor.device == device:
                return tensor
            target_dtype = dtype if dtype is not None else tensor.dtype
            return tensor.to(device=device, dtype=target_dtype)

        if isinstance(value, Tensor):
            tensor = value
        else:
            target_dtype = dtype or torch.float32
            tensor = torch.as_tensor(value, dtype=target_dtype)

        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)

        device = self.device
        if tensor.device != device:
            tensor = tensor.to(device)

        return tensor

    def _map_state_variables(
        self,
        state: Dict[str, Any],
        aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Map state variable names using the shared VariableNameMapper.

        Parameters
        ----------
        state : Dict[str, Any]
            Input state dictionary.
        aliases : Optional[Dict[str, str]]
            Optional alias map of source_name -> target_name for module-specific needs.

        Returns
        -------
        Dict[str, Any]
            Shallow copy of state with mapped keys added.
        """
        from ..variable_mapper import variable_name_mapper

        mapped_state = variable_name_mapper.create_mapped_view(state, target="module")
        if aliases:
            for source_name, target_name in aliases.items():
                if source_name in mapped_state and target_name not in mapped_state:
                    mapped_state[target_name] = mapped_state[source_name]
        return cast(Dict[str, Any], mapped_state)  # type: ignore[redundant-cast,unused-ignore]

    def set_parameter_value(self, name: str, value: Union[float, int, Tensor]) -> None:
        """
        Set an instance-level parameter value and update its PyTorch parameter.

        The value stays on this instance: it is written into this Functor's
        config (its ``parameters`` section unless the name is a root-level
        config entry), the "config" source of ``get_parameter_value``, so no
        other Functor and no checkpoint contract sees it. Changing a global
        parameter is an explicit infrastructure call
        (``get_global_manager().set_value(...)``).

        The value is validated against the config-declared bounds and, when the
        ParameterManager holds a YAML spec for ``<default namespace>.<name>``,
        against that spec (bounds and dtype).

        Parameters:
            name : str
                Parameter name
            value : Union[float, int, Tensor]
                New parameter value

        Raises:
            ValueError: If value is outside the declared bounds or violates the
                YAML spec
        """
        # Check bounds if specified
        if name in self._parameter_bounds:
            min_val, max_val = self._parameter_bounds[name]
            float_val = float(value)
            if not (min_val <= float_val <= max_val):
                raise ValueError(f"Parameter {name} value {float_val} outside bounds [{min_val}, {max_val}]")

        param_manager = self.get_parameter_manager()
        if param_manager is not None:
            spec = param_manager.get_parameter(f"{self._default_parameter_namespace()}.{name}")
            if spec is not None:
                spec_value = float(value.detach().cpu().item()) if isinstance(value, Tensor) else value
                errors = spec.validate_value(spec_value)
                if errors:
                    raise ValueError(
                        f"Parameter {name} value {spec_value!r} violates its YAML spec "
                        f"{spec.key}: {'; '.join(errors)}"
                    )

        # Update parameter value
        self.parameter_values[name] = value

        # The config is the instance-level source of get_parameter_value: a
        # root-level entry is updated in place, any other name is written into
        # the parameters section (created when absent).
        if not isinstance(self._original_config, dict):
            raise TypeError(f"{self.name}: the Functor config must be a dict to hold instance parameters")
        config_value = float(value.detach().cpu().item()) if isinstance(value, Tensor) else value
        params_section = self._original_config.get("parameters")
        if name in self._original_config and not (isinstance(params_section, dict) and name in params_section):
            self._original_config[name] = config_value
        else:
            if params_section is None:
                params_section = self._original_config["parameters"] = {}
            if not isinstance(params_section, dict):
                raise TypeError(f"{self.name}: config 'parameters' must be a dict, got {type(params_section).__name__}")
            params_section[name] = config_value

        # Update PyTorch parameter if registered
        param_attr_name = f"param_{name}"
        if hasattr(self, param_attr_name):
            param_tensor = getattr(self, param_attr_name)
            if isinstance(param_tensor, nn.Parameter):
                with torch.no_grad():
                    param_tensor.data.fill_(float(value))

    def register_parameters_to_manager(self) -> bool:
        """
        Register this functor's parameters with the global parameter manager.

        This method creates parameter specifications for all parameters defined
        by this functor and registers them with the parameter manager for
        unified access and management.

        Returns:
            bool: True once every parameter is registered.

        Raises:
            RuntimeError: If no parameter manager is available. Registration
                errors of the manager propagate.
        """
        param_manager = self.get_parameter_manager()
        if param_manager is None:
            raise RuntimeError(
                f"register_parameters_to_manager for {self.name} requires a parameter manager"
            )
        from ...parameters import ParameterSpec, ParameterCategory

        # Determine parameter category based on functor category
        if self.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS:
            param_category = ParameterCategory.BIOLOGICAL_PARAMETER
        elif self.metadata.category == ProcessCategory.PHYSICAL_FIELD:
            param_category = ParameterCategory.PHYSICAL_CONSTANT
        else:
            param_category = ParameterCategory.MODEL_PARAMETER

        # Register each parameter the config declares in its 'parameters'
        # section; root-level config options (for example the
        # auto_register_parameters_to_manager flag) are not parameters.
        category_name = self.metadata.category.value
        declared_parameters = self._original_config.get('parameters', {}) or {}
        for param_name in declared_parameters:
            param_value = self.parameter_values[param_name]

            # Construct full parameter key
            param_key = f"{category_name}.{self.name}.{param_name}"

            # Do not override existing database specs to prevent unit drift
            existing_spec = param_manager.get_parameter(param_key)
            if existing_spec is not None:
                continue

            # Declared metadata only: a parameter registered here needs its
            # units from the config entry. Its bounds come from the config
            # entry or the database spec (_load_configuration) and follow the
            # single rule parameter_bounds_error applied by ParameterSpec: a
            # numeric dtype declares a (lower, upper) interval (inf for an
            # open side), None only for a non-numeric dtype, so a numeric
            # parameter without a declared interval raises here.
            if param_name not in self._parameter_units:
                raise ValueError(
                    f"{self.name}: config parameter '{param_name}' has no declared units "
                    "(config entry or parameter database); declare them to register it"
                )
            param_units = self._parameter_units[param_name]
            # dtype has no universal default (ParameterSpec); it is declared by
            # the config entry or the parameter's database specification.
            entry = declared_parameters[param_name]
            param_dtype = entry.get("dtype") if isinstance(entry, dict) else None
            if param_dtype is None:
                database_spec = self._database_spec(param_name)
                param_dtype = getattr(database_spec, "dtype", None) if database_spec is not None else None
            if not param_dtype:
                raise ValueError(
                    f"{self.name}: config parameter '{param_name}' has no declared dtype "
                    "(config entry or parameter database); declare it to register it"
                )
            param_bounds = self._parameter_bounds.get(param_name)
            param_requires_grad = self._parameter_requires_grad.get(param_name, False)

            # Prepare tags
            tags = {"functor_parameter"}
            if param_requires_grad:
                tags.add("learnable")

            # Create parameter specification
            param_spec = ParameterSpec(
                key=param_key,
                value=param_value,
                units=param_units,
                dtype=str(param_dtype),
                bounds=param_bounds,
                category=param_category,
                description=f"Parameter {param_name} for {self.name} functor",
                tags=tags,
                requires_grad=param_requires_grad,
                provenance={
                    'source': 'functor_registration',
                    'functor_name': self.name,
                    'functor_category': category_name,
                    'schema_version': '1.0.0'
                }
            )

            # Register with manager
            param_manager.register_parameter(param_spec)

        logger.info(f"Registered {len(declared_parameters)} parameters for functor {self.name}")
        return True


    def get_required_inputs(self) -> list[str]:
        """Return the declared inputs that every call of this Functor must provide.

        This is the single required-input rule: the declared
        ``metadata.input_variables`` minus ``optional_input_variables``, in
        declaration order. ``validate_state``, the validation pipeline and
        ``NodeRegistry.register_functor_instance`` read requiredness only from
        this method. A Functor declares an optional input by listing it in
        ``optional_input_variables`` (in ``__init__``, or where a configured
        value such as a bound velocity field replaces the state input) and does
        not override this method.
        """
        raw_optional = getattr(self, "optional_input_variables", ()) or ()
        optional = {raw_optional} if isinstance(raw_optional, str) else set(raw_optional)
        return [name for name in self.metadata.input_variables if name not in optional]

    def validate_state(self, state: StateDict) -> ValidationErrors:
        """
        Validate state dictionary for required variables and constraints.

        Parameters:
            state : StateDict
                State dictionary to validate

        Returns:
            ValidationErrors: List of validation error messages
        """
        errors = []

        # Check the inputs of the single required-input rule
        for var_name in self.get_required_inputs():
            if var_name not in state:
                errors.append(f"Required state variable '{var_name}' not found")
            elif not isinstance(state[var_name], (Tensor, float, int)):
                errors.append(f"State variable '{var_name}' must be numeric tensor")

        # Check for physical constraints
        for var_name, value in state.items():
            if var_name.startswith("diagnostics.") or var_name.startswith("__"):
                continue
            if isinstance(value, (Tensor, float, int)):
                # Handle tensor values - only convert to float if single element
                if isinstance(value, Tensor):
                    if value.numel() == 1:
                        numeric_value = float(value.detach().item())
                    else:
                        # For multi-element tensors, skip constraint checking for now
                        continue
                else:
                    numeric_value = float(value) if hasattr(value, '__float__') else value

                # Temperature constraints (must be positive in Kelvin)
                # Absolute temperatures in kelvin are non-negative; a declared
                # anomaly or difference is a signed quantity and is exempt.
                lowered_name = var_name.lower()
                if (
                    'temperature' in lowered_name
                    and 'anomaly' not in lowered_name
                    and 'difference' not in lowered_name
                    and numeric_value < 0
                ):
                    errors.append(f"Temperature '{var_name}' cannot be negative: {numeric_value} K")

                # Concentration constraints (non-negative)
                if any(keyword in var_name.lower() for keyword in ['concentration', 'pool', 'mass']):
                    if numeric_value < 0:
                        errors.append(f"Concentration/mass '{var_name}' cannot be negative: {numeric_value}")

        return errors

    def check_conservation_laws(self,
                               input_state: StateDict,
                               output_updates: FluxDict) -> Dict[str, float]:
        """
        Base conservation check with metadata-first variable identification.

        Tier 1:
            When metadata.output_variables is declared, aggregate only declared
            outputs. This path is deterministic and does not use keyword
            inference.
        Tier 2:
            Not implemented. There is no keyword inference and no
            enable_heuristic_conservation_fallback parameter; when
            output_variables is empty this method reports no local violation
            and the StateManager ledger remains the authoritative check.

        Subclasses should still override this method for domain-specific
        accounting. StateManager remains the authoritative global conservation
        ledger for cross-functor mass balance.

        Parameters
        ----------
        input_state : StateDict
            State before process execution
        output_updates : FluxDict
            Updates produced by this process

        Returns
        -------
        Dict[str, float]
            Conservation violation magnitudes (absolute values)
        """
        violations: Dict[str, float] = {}

        def _to_scalar(val: Any) -> float:
            # Convert tensors or numeric values to scalar for aggregation
            from ..state.types import UpdateItem  # Local import to avoid circular dependency
            if isinstance(val, UpdateItem):
                val = val.value
            if isinstance(val, Tensor):
                return float(val.detach().sum().item())
            if isinstance(val, (float, int)):
                return float(val)
            raise TypeError(
                f"Local conservation check of {self.name} needs numeric declared outputs, "
                f"got {type(val).__name__}"
            )

        declared_outputs = tuple(self.metadata.output_variables or ())
        if declared_outputs:
            matched_vars = [key for key in declared_outputs if key in output_updates]
            if not matched_vars:
                return violations
            total = sum(_to_scalar(output_updates[key]) for key in matched_vars)
            magnitude = abs(total)
            for law in self.metadata.conservation_laws:
                violations[law] = magnitude
            return violations

        return violations

    def _enforce_forward_conservation(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> None:
        """
        Enforce uniform forward-level conservation policy when forward() is called directly.

        References:
            - Schlesinger and Bernhardt (2013), biogeochemical mass-balance framing.
            - Manzoni and Porporato (2009), process-level conservation diagnostics.
        Mathematical expression:
            - For each declared law: violation = |sum(local conservation terms)|
            - Trigger when violation > conservation_tolerance.
        Parameter meanings:
            - validation.conservation.conservation_tolerance: local threshold [dimensionless]
            - validation.conservation.strict_conservation: blocking toggle [dimensionless/bool]
        """
        if self._inside_canonical_call:
            return
        if not self.metadata.conservation_laws:
            return

        tolerance = self.get_parameter_value(
            "conservation_tolerance",
            namespace="validation.conservation",
            required=True,
            validate=True,
        )
        strict = self.get_parameter_value(
            "strict_conservation",
            namespace="validation.conservation",
            required=True,
            validate=True,
        )

        def _as_float(value: Any) -> float:
            if isinstance(value, Tensor):
                if value.numel() == 1:
                    return float(value.detach().item())
                return float(value.detach().mean().item())
            return float(value)

        def _as_bool(value: Any) -> bool:
            # The strict flag is a declared bool (dtype bool in
            # validation/conservation.yaml); a single-element bool tensor is
            # accepted. Any other value is a type error, never "not strict".
            if isinstance(value, Tensor) and value.dtype == torch.bool and value.numel() == 1:
                return bool(value.item())
            if isinstance(value, bool):
                return value
            raise TypeError(
                f"{self.name}: validation.conservation.strict_conservation must be a bool, "
                f"got {type(value).__name__}"
            )

        tolerance_value = _as_float(tolerance)
        strict_enabled = _as_bool(strict)
        from ..state.types import partition_outputs

        filtered_updates, _, intents = partition_outputs(
            output_updates, declared_outputs=self.metadata.output_variables or (),
        )
        scientific_updates = (
            {key: value for key, value in output_updates.items() if key not in intents}
            if intents else output_updates
        )
        
        has_state_or_delta_term = any(
            key.startswith("delta_") or (
                (key[len("delta_"):] if key.startswith("delta_") else key) in input_state
            )
            for key in filtered_updates.keys()
        )

        def _is_forward_conservation_term(key: str) -> bool:
            base_key = key[len("delta_"):] if key.startswith("delta_") else key
            if key.startswith("delta_"):
                return True
            lower_key = base_key.lower()
            pool_like_suffixes = (
                "_carbon",
                "_nitrogen",
                "_phosphorus",
                "_water",
                "_mass",
                "_c",
                "_n",
                "_p",
            )
            pool_like_tokens = ("som", "litter", "biomass", "nh4", "no3", "po4")
            is_pool_like = lower_key.endswith(pool_like_suffixes) or any(
                token in lower_key for token in pool_like_tokens
            )
            # Include produced/consumed pool-like terms even when callers provide
            # a minimal input state (e.g., only one substrate variable).
            if is_pool_like and (base_key in input_state or has_state_or_delta_term):
                return True
            if has_state_or_delta_term and lower_key.endswith("_flux"):
                return True
            if has_state_or_delta_term and (
                lower_key.endswith("_export") or lower_key.endswith("_production")
            ):
                return True
            if has_state_or_delta_term and "respiration" in lower_key:
                return True
            return False

        using_base_checker = (
            self.__class__.check_conservation_laws is FunctorBase.check_conservation_laws
        )
        conservation_updates = (
            {
                key: value
                for key, value in filtered_updates.items()
                if _is_forward_conservation_term(key)
            }
            if using_base_checker
            else scientific_updates
        )
        if not conservation_updates:
            return

        violations = self.check_conservation_laws(input_state, conservation_updates)
        for law, violation in violations.items():
            violation_value = abs(float(violation))
            if violation_value <= tolerance_value:
                continue
            message = (
                f"{self.name}: Conservation violation in '{law}': "
                f"{violation_value:.2e} exceeds tolerance {tolerance_value:.2e}"
            )
            logger.error(message)
            if strict_enabled:
                raise ValueError(message)

    def _extract_validation_value(self, value: Any) -> Any:
        """Convert tensors to detached values for validation context."""
        if isinstance(value, Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach()
        return value

    def _strict_validation_enabled(self) -> bool:
        """Return the framework validation strictness of this Functor's config.

        ``strict_validation`` (a bool, default True) is owned by the config:
        True fails a governed call on an ERROR/CRITICAL finding; False is the
        explicit opt-in that downgrades those findings into the caller's
        ``validation_report``. Another type raises TypeError; the removed key
        ``strict_validation_errors`` raises ConfigurationError.
        """
        if "strict_validation_errors" in self._original_config:
            raise ConfigurationError(
                f"{self.name}: config key 'strict_validation_errors' was removed; "
                "framework validation is strict by default and "
                "'strict_validation: false' is the explicit opt-in to downgrade errors"
            )
        strict_validation = self._original_config.get("strict_validation", True)
        if not isinstance(strict_validation, bool):
            raise TypeError(
                f"{self.name}: config 'strict_validation' must be a bool, "
                f"got {type(strict_validation).__name__}"
            )
        return strict_validation

    def _run_validation_pipeline(self,
                                 state: StateDict,
                                 updates: FluxDict,
                                 time: float,
                                 dt: float,
                                 validation_report: Optional[list] = None) -> Optional[Any]:
        """Execute L1 validations derived from functor metadata.

        Every call with findings appends one record to ``validation_report``
        when the caller supplies it (the Functor itself keeps no findings).
        """
        ValidationEngine = cast(
            Callable[..., Any],
            _load_runtime_symbol("relic.validation.framework.engine", "ValidationEngine"),
        )
        get_global_registry = cast(
            Callable[[], Any],
            _load_runtime_symbol("relic.validation.framework.registry", "get_global_registry"),
        )
        ValidationContext = _load_runtime_symbol("relic.validation.framework.models", "ValidationContext")
        ValidationLevel = _load_runtime_symbol("relic.validation.framework.models", "ValidationLevel")

        # Handle cases where metadata is minimal or validation engine previously failed
        if self.metadata is None:
            return None
        if self._validation_engine is None:
            self._validation_engine = ValidationEngine(
                registry=get_global_registry()
            )

        parameter_snapshot = {
            key: self._extract_validation_value(val)
            for key, val in self.parameter_values.items()
        }
        state_snapshot = dict(state) if isinstance(state, dict) else state
        # ValidationContext.output holds named outputs (Optional[Dict]); a bare
        # tensor returned for the torch.nn.Module hook contract names no
        # variables, so the context carries None for it.
        output_snapshot = dict(updates) if isinstance(updates, dict) else None
        context = ValidationContext(
            state_manager=None,
            config=self._original_config,
            execution_layer=None,
            metadata={
                'functor_name': self.name,
                'functor_category': self.metadata.category.value if self.metadata.category else None,
                'required_input_variables': self.get_required_inputs(),
            },
            state=state_snapshot,
            parameters=parameter_snapshot,
            output=output_snapshot,
            functor_name=self.name,
            functor_metadata=self.metadata.to_dict(),
            time=time,
            dt=dt
        )

        report = self._validation_engine.run(
            layers="L1",
            tags=[self.name],
            context=context,
            enabled_only=True,
            fail_fast=False,
            include_metrics=False
        )

        if report is None:
            return None

        warning_results = report.get_results_by_level(ValidationLevel.WARNING)
        for warning in warning_results:
            logger.warning("Validation warning for %s: %s", self.name, warning.message)

        error_results = (
            report.get_results_by_level(ValidationLevel.ERROR)
            + report.get_results_by_level(ValidationLevel.CRITICAL)
        )
        if not error_results and not warning_results:
            return report

        def _finding_entry(result: Any) -> Dict[str, Any]:
            return {
                "level": result.level.value,
                "code": result.code.value,
                "message": result.message,
                "scope": result.scope,
            }

        # A call with findings adds one record to the caller's report (the
        # simulator passes its run-level report), so a downgraded error is never
        # only a log line.
        strict_validation = self._strict_validation_enabled()
        record: Dict[str, Any] = {
            "functor": self.name,
            "time": float(time),
            "dt": float(dt),
            "strict_validation": strict_validation,
            "errors": [_finding_entry(result) for result in error_results],
            "warnings": [_finding_entry(result) for result in warning_results],
            "disposition": "passed_with_warnings",
        }
        if validation_report is not None:
            validation_report.append(record)

        if error_results:
            error_messages = [result.message for result in error_results]
            messages = "; ".join(error_messages)
            if not strict_validation:
                if validation_report is None:
                    record["disposition"] = "raised"
                    raise ExecutionError(
                        f"Validation failed for functor {self.name}: {messages} "
                        "(strict_validation=False downgrades errors only into a "
                        "caller-supplied validation_report; this call supplied none)"
                    )
                record["disposition"] = "downgraded"
                logger.warning(
                    "Validation reported errors for %s; strict_validation=False downgrades them "
                    "into the caller's validation report: %s",
                    self.name,
                    messages,
                )
                return report

            # Strict validation raises every ERROR/CRITICAL finding; no message
            # class is exempt.
            record["disposition"] = "raised"
            raise ExecutionError(f"Validation failed for functor {self.name}: {messages}")

        return report

    def get_execution_stats(self) -> Dict[str, Union[int, float, str]]:
        """
        Get performance and execution statistics.

        Returns:
            Dict[str, Union[int, float, str]]: Execution statistics
        """
        return {
            'execution_count': self._execution_count,
            'total_execution_time': self._total_execution_time,
            'last_execution_time': self._last_execution_time,
            'min_execution_time': self._performance_stats['min_execution_time'],
            'max_execution_time': self._performance_stats['max_execution_time'],
            'mean_execution_time': self._performance_stats['mean_execution_time'],
            'error_count': self._error_count,
            'state': self.state.value
        }

    def reset_stats(self) -> None:
        """Reset execution statistics and error tracking."""
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = 0.0
        self._performance_stats = {
            'min_execution_time': float('inf'),
            'max_execution_time': 0.0,
            'mean_execution_time': 0.0
        }
        self._error_count = 0
        self._last_error = None
        self._last_pipeline_timings_s = _empty_pipeline_timings()
        self._pipeline_timing_totals_s = _empty_pipeline_timings()
        self._pipeline_timing_count = 0

    def preprocess_state(self, state: StateDict) -> StateDict:
        """
        Hook to normalize input state before validation and forward execution.

        Subclasses may override to expand aliases or inject canonical keys.
        """
        return state

    def postprocess_updates(self, original_state: StateDict, updates: FluxDict) -> FluxDict:
        """
        Hook to normalize updates before they are returned to the caller.

        Subclasses may override to canonicalize output variable names.
        """
        return updates

    def _resolve_call_arguments(
        self,
        args: Tuple[Any, ...],
        time: Optional[float],
        dt: Optional[float],
        kwargs: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Resolve the call into time, dt and the forward keyword arguments.

        time and dt are either the two positional arguments after the state or
        the keywords; scientific inputs come only from the state, so any other
        positional form (for example an environment mapping) raises TypeError,
        as do an ``environment`` keyword and a value given both positionally
        and by keyword.
        """
        if "environment" in kwargs:
            raise TypeError(
                f"{self.name}: the governed call takes no environment argument; "
                "scientific inputs belong in the state"
            )
        if not args:
            return time, dt, dict(kwargs)
        if len(args) == 2 and time is None and dt is None:
            return args[0], args[1], dict(kwargs)
        raise TypeError(
            f"{self.name}: call the Functor as functor(state, time, dt) or "
            f"functor(state, time=..., dt=...); got {len(args)} positional argument(s) "
            f"after the state{' and a keyword time/dt' if time is not None or dt is not None else ''}. "
            "Scientific inputs belong in the state."
        )

    def _require_time_and_dt(
        self,
        time: Optional[float],
        dt: Optional[float],
    ) -> Tuple[float, float]:
        """Validate time and dt existence and positivity."""
        if time is None:
            raise ValueError("time parameter is required")
        if dt is None:
            raise ValueError("dt parameter is required")
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got: {dt}")
        return time, dt

    def get_kernel_identity(self) -> dict[str, Any]:
        """Describe the actual, source-traceable Python forward implementation.

        This narrowly binds the current class, Python cache tag, executable
        code, source content, defaults, and immutable closure cells. It does not
        certify globals, helper implementations, native ABI, configuration, or
        scientific state. Those contracts retain their existing owners.
        Actual class source also binds constructor/helper source changes; it
        does not certify their complete dynamically bound execution closure.
        Source locations, line tables, addresses, and repr fallbacks are not
        identity fields. The result is recomputed so code replacement is seen.
        """
        kernel = self.forward
        binding = "function"
        if isinstance(kernel, MethodType):
            if kernel.__self__ is not self:
                raise ValueError("Python kernel must bind to this Functor instance.")
            binding = "instance_method"
            kernel = kernel.__func__
        if not isinstance(kernel, FunctionType):
            raise ValueError("Kernel identity requires an actual Python function.")

        def constant_identity(value: Any, path: str) -> Any:
            if isinstance(value, CodeType):
                return {
                    "type": "code",
                    "argcount": value.co_argcount,
                    "posonlyargcount": value.co_posonlyargcount,
                    "kwonlyargcount": value.co_kwonlyargcount,
                    "nlocals": value.co_nlocals,
                    "stacksize": value.co_stacksize,
                    "flags": value.co_flags,
                    "bytecode": value.co_code.hex(),
                    "exceptiontable": value.co_exceptiontable.hex(),
                    "name": value.co_name,
                    "qualname": value.co_qualname,
                    "names": list(value.co_names),
                    "varnames": list(value.co_varnames),
                    "freevars": list(value.co_freevars),
                    "cellvars": list(value.co_cellvars),
                    "constants": [
                        constant_identity(item, f"{path}.constants[{index}]")
                        for index, item in enumerate(value.co_consts)
                    ],
                }
            if value is None or type(value) in (bool, int, str):
                return {"type": type(value).__name__, "value": value}
            if value is Ellipsis:
                return {"type": "ellipsis"}
            if type(value) is bytes:
                return {"type": "bytes", "value": value.hex()}
            if type(value) is float:
                return {"type": "float", "value": struct.pack("!d", value).hex()}
            if type(value) is complex:
                return {
                    "type": "complex",
                    "value": struct.pack("!dd", value.real, value.imag).hex(),
                }
            if type(value) in (tuple, frozenset):
                items = [constant_identity(item, f"{path}[]") for item in value]
                if type(value) is frozenset:
                    items.sort(key=lambda item: json.dumps(item, sort_keys=True))
                return {"type": type(value).__name__, "items": items}
            if type(value) is slice:
                # CPython 3.14 folds constant subscripts such as x[1::2] into
                # code.co_consts as slice objects; bind each bound through the
                # same constant rules so mutable bounds are still rejected.
                return {
                    "type": "slice",
                    "start": constant_identity(value.start, f"{path}.start"),
                    "stop": constant_identity(value.stop, f"{path}.stop"),
                    "step": constant_identity(value.step, f"{path}.step"),
                }
            raise ValueError(
                f"Kernel identity rejects mutable or unsupported {path} "
                f"value of type {type(value).__name__}."
            )

        closure: dict[str, Any] = {}
        for name, cell in zip(kernel.__code__.co_freevars, kernel.__closure__ or (), strict=True):
            try:
                value = cell.cell_contents
            except ValueError as error:
                raise ValueError(f"Kernel identity has an empty closure cell: {name}.") from error
            if name == "__class__" and isinstance(value, type) and value in type(self).__mro__:
                closure[name] = {
                    "type": "class_cell",
                    "module": value.__module__,
                    "qualname": value.__qualname__,
                }
            else:
                closure[name] = constant_identity(value, f"closure.{name}")
        try:
            # Inspect the code object so functools.__wrapped__ cannot replace
            # the source of the callable that will actually execute.
            source = inspect.getsource(kernel.__code__)
        except (OSError, TypeError) as error:
            raise ValueError("Kernel identity requires inspectable Python source.") from error
        try:
            # Read the actual class block without following __wrapped__.
            class_lines, class_start = inspect.findsource(type(self))
            class_source = "".join(inspect.getblock(class_lines[class_start:]))
        except (OSError, TypeError) as error:
            raise ValueError("Kernel identity requires inspectable Python class source.") from error
        code_payload = constant_identity(kernel.__code__, "code")
        code_bytes = json.dumps(code_payload, sort_keys=True, ensure_ascii=True).encode("ascii")
        return {
            "schema_version": 1,
            "scope": "bound_python_kernel",
            "class_module": type(self).__module__,
            "class_qualname": type(self).__qualname__,
            "class_source_sha256": hashlib.sha256(class_source.encode("utf-8")).hexdigest(),
            "python_implementation": sys.implementation.name,
            "python_cache_tag": sys.implementation.cache_tag,
            "binding": binding,
            "function_module": kernel.__module__,
            "function_qualname": kernel.__qualname__,
            "code_sha256": hashlib.sha256(code_bytes).hexdigest(),
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "defaults": constant_identity(kernel.__defaults__, "defaults"),
            "keyword_defaults": {
                key: constant_identity(value, f"keyword_defaults.{key}")
                for key, value in sorted((kernel.__kwdefaults__ or {}).items())
            },
            "closure": closure,
        }

    def set_call_governor(
        self,
        governor: Callable[..., FluxDict] | None,
    ) -> None:
        """Bind one runtime governor around the standard PyTorch module call.

        The governor receives ``(module_call, state, time=..., dt=..., **kwargs)``
        and returns raw updates. Input validation precedes it; postprocessing,
        conservation, and output validation still run on every returned result.
        It must call the supplied continuation when execution is required and
        must not replace ``forward`` or call the Functor object recursively.
        Passing ``None`` removes the binding. This callback is runtime policy,
        not a scientific child module or checkpoint state.
        """
        if governor is not None and not callable(governor):
            raise TypeError("Call governor must be callable or None.")
        object.__setattr__(self, "_call_governor", governor)

    def _execute_forward_with_timing(
        self,
        state: StateDict,
        original_state: StateDict,
        time: float,
        dt: float,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[FluxDict, float]:
        """Execute the governed PyTorch call with timing and postprocessing."""
        import time as time_module

        start_time = time_module.perf_counter()
        self._inside_canonical_call = True
        try:
            module_call = super().__call__
            governor = getattr(self, "_call_governor", None)
            if governor is None:
                updates = module_call(state, time=time, dt=dt, **forward_kwargs)
            else:
                updates = governor(
                    module_call, state, time=time, dt=dt, **forward_kwargs,
                )
        finally:
            self._inside_canonical_call = False
        if isinstance(updates, dict):
            from ..state.types import partition_outputs

            _, _, intents = partition_outputs(
                updates, declared_outputs=self.metadata.output_variables or (),
            )
            scientific_updates = (
                {key: value for key, value in updates.items() if key not in intents}
                if intents else updates
            )
            processed = self.postprocess_updates(
                original_state if isinstance(original_state, dict) else {},
                scientific_updates,
            )
            if intents:
                if not isinstance(processed, dict):
                    raise TypeError("Postprocessing must retain a mapping for external intents")
                if intents.keys() & processed.keys():
                    raise ValueError("Postprocessing cannot replace an external intent output")
                updates = {**intents, **processed}
            else:
                updates = processed
        execution_time = time_module.perf_counter() - start_time
        return updates, execution_time

    def _update_performance_stats(self, execution_time: float) -> None:
        """Update execution counters and rolling timing statistics."""
        self._execution_count += 1
        self._total_execution_time += execution_time
        self._last_execution_time = execution_time

        if execution_time < self._performance_stats["min_execution_time"]:
            self._performance_stats["min_execution_time"] = execution_time
        if execution_time > self._performance_stats["max_execution_time"]:
            self._performance_stats["max_execution_time"] = execution_time
        self._performance_stats["mean_execution_time"] = (
            self._total_execution_time / self._execution_count
        )

    def _to_scalar_float(self, value: Any) -> float:
        """Convert tensor or scalar to float for tolerance comparisons."""
        if isinstance(value, Tensor):
            if value.numel() == 1:
                return float(value.detach().item())
            return float(value.detach().mean().item())
        return float(value)

    def _check_local_conservation(
        self,
        state: StateDict,
        updates: FluxDict,
    ) -> None:
        """Run per-functor conservation checks for declared conservation laws."""
        if not self.metadata.conservation_laws:
            return

        # References: StateManager ledger-based conservation accounting.
        # Mathematical: element_balance = sum(delta_stock) + sum(boundary_flux * dt) == 0
        # Parameter meanings: functor_tolerance [dimensionless], functor_severe_threshold [dimensionless].
        # ---- Conservation Check Architecture ----
        # Tier 1 (here): Per-Functor local check.
        #   Runs for all functors declaring metadata.conservation_laws.
        #   Catches process-level violations inside one Functor.
        #   Tolerance source: validation.conservation.functor_tolerance and
        #   validation.conservation.functor_severe_threshold.
        # Tier 2 (StateManager.apply_updates): Global ledger check.
        #   Runs after all updates are aggregated by StateManager.
        #   Catches system-level imbalance across pools and domains.
        #   Tolerance source: StateManager.conservation_tolerance.
        # Tier 1 and Tier 2 are complementary, not redundant.
        tol = self.get_parameter_value(
            "functor_tolerance",
            namespace="validation.conservation",
            source_order=("manager",),
            required=True,
            validate=True,
        )
        severe = self.get_parameter_value(
            "functor_severe_threshold",
            namespace="validation.conservation",
            source_order=("manager",),
            required=True,
            validate=True,
        )
        tol_value = self._to_scalar_float(tol)
        severe_value = self._to_scalar_float(severe)
        using_base_checker = (
            self.__class__.check_conservation_laws is FunctorBase.check_conservation_laws
        )
        declared_outputs = tuple(self.metadata.output_variables or ())

        from ..state.types import partition_outputs

        filtered_updates, _, intents = partition_outputs(
            updates, declared_outputs=self.metadata.output_variables or (),
        )
        scientific_updates = (
            {key: value for key, value in updates.items() if key not in intents}
            if intents else updates
        )
        if using_base_checker and not declared_outputs:
            violations = {}
        else:
            conservation_updates = filtered_updates if using_base_checker else scientific_updates
            violations = self.check_conservation_laws(state, conservation_updates)
        for law, violation in violations.items():
            if violation > tol_value:
                warning_msg = (
                    f"Conservation law '{law}' violated in {self.name}: "
                    f"violation magnitude = {violation}"
                )
                warnings.warn(warning_msg)
                if violation > severe_value and not using_base_checker:
                    raise ConservationError(warning_msg)

    def _raise_execution_error(self, error: Exception) -> NoReturn:
        """Mark functor as failed and raise standardized execution error."""
        self.state = FunctorState.FAILED
        self._error_count += 1
        self._last_error = error
        error_msg = f"Execution failed for functor {self.name}: {str(error)}"
        raise ExecutionError(error_msg) from error

    def __call__(
        self,
        state: StateDict,
        *args: Any,
        time: Optional[float] = None,
        dt: Optional[float] = None,
        validation_report: Optional[list] = None,
        **kwargs: Any,
    ) -> FluxDict:
        """
        Callable interface with validation, timing, and error handling.

        Wraps the forward method with comprehensive validation, performance
        monitoring, conservation checking, and error handling. This is the
        primary execution interface used by the simulator.

        Calling styles (scientific inputs come only from the state):
        - functor(state, time, dt)
        - functor(state, time=..., dt=...)

        ``validation_report`` (a list, not passed to ``forward``) receives one
        record per call whose framework validation reported findings; the
        simulator passes its run-level report.
        """
        if validation_report is not None and not isinstance(validation_report, list):
            raise TypeError(
                f"{self.name}: validation_report must be a list, "
                f"got {type(validation_report).__name__}"
            )
        time, dt, call_kwargs = self._resolve_call_arguments(args, time, dt, kwargs)
        profiling_enabled = bool(self._performance_profiling_enabled)
        pipeline_started = time_module.perf_counter() if profiling_enabled else None

        try:
            call_time, call_dt = self._require_time_and_dt(time, dt)

            original_state = state
            state = self.preprocess_state(state)
            self.state = FunctorState.EXECUTING

            state_validation_started = (
                time_module.perf_counter() if profiling_enabled else None
            )
            validation_errors = self.validate_state(state)
            state_validation_time = (
                max(0.0, time_module.perf_counter() - state_validation_started)
                if state_validation_started is not None
                else 0.0
            )
            if validation_errors:
                raise ValueError(f"State validation failed for {self.name}: {validation_errors}")

            updates, execution_time = self._execute_forward_with_timing(
                state=state,
                original_state=original_state,
                time=call_time,
                dt=call_dt,
                forward_kwargs=call_kwargs,
            )
            self._update_performance_stats(execution_time)

            scientific_updates = updates
            if isinstance(updates, dict):
                from ..state.types import partition_outputs

                _, _, intents = partition_outputs(
                    updates, declared_outputs=self.metadata.output_variables or (),
                )
                if intents:
                    scientific_updates = {
                        key: value for key, value in updates.items() if key not in intents
                    }

            conservation_started = (
                time_module.perf_counter() if profiling_enabled else None
            )
            self._check_local_conservation(state, scientific_updates)
            conservation_time = (
                max(0.0, time_module.perf_counter() - conservation_started)
                if conservation_started is not None
                else 0.0
            )

            validation_started = (
                time_module.perf_counter() if profiling_enabled else None
            )
            self._run_validation_pipeline(
                state, scientific_updates, call_time, call_dt, validation_report,
            )
            framework_validation_time = (
                max(0.0, time_module.perf_counter() - validation_started)
                if validation_started is not None
                else 0.0
            )
            if profiling_enabled and pipeline_started is not None:
                self._last_pipeline_timings_s = {
                    "state_validation": state_validation_time,
                    "forward": max(0.0, float(execution_time)),
                    "local_conservation": conservation_time,
                    "framework_validation": framework_validation_time,
                    "total": max(
                        0.0,
                        time_module.perf_counter() - pipeline_started,
                    ),
                }
                self._pipeline_timing_count += 1
                for stage_name, duration in self._last_pipeline_timings_s.items():
                    self._pipeline_timing_totals_s[stage_name] += duration
            self.state = FunctorState.COMPLETED
            return updates

        except ExecutionError:
            raise
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as error:
            self._raise_execution_error(error)

    def to_device(self, device: Union[str, Device]) -> "FunctorBase":
        """
        Move functor to specified computation device.

        Parameters:
            device : Union[str, Device]
                Target device ("cpu", "cuda", "cuda:0", etc.)

        Returns:
            FunctorBase: Self reference for method chaining
        """
        target_device = torch.device(device)
        super().to(target_device)
        self.device = target_device
        return self

    def manager_binding_key(self, value: Any) -> Optional[str]:
        """Key of the ParameterManager learnable binding that ``value`` is.

        A learnable parameter has one owner, the ParameterManager, which keeps
        one ``nn.Parameter`` per key in ``nn_parameters`` so that every consumer
        and the calibration optimizer share one theta
        (docs/source/architecture_overview.md Section 4.2).

        Returns:
            Optional[str]: The key under which this Functor's own manager holds
            ``value`` (object identity; neither the attribute name nor the value
            decides), or None when ``value`` is no binding of that manager. A
            Functor that has not resolved a manager, or whose manager keeps no
            ``nn_parameters``, holds no binding.
        """
        if not isinstance(value, nn.Parameter):
            return None
        manager = self.__dict__.get("_parameter_manager")
        bindings = getattr(manager, "nn_parameters", None) if manager is not None else None
        if not bindings:
            return None
        for key, binding in bindings.items():
            if binding is value:
                return key
        return None

    def held_manager_parameters(self) -> Dict[str, nn.Parameter]:
        """ParameterManager learnable bindings this Functor holds by reference.

        Returns:
            Dict[str, nn.Parameter]: Manager key to binding for every attribute
            of this Functor, including nested mappings and standard containers,
            that is a binding of its own manager. These are not module
            parameters: they are absent from ``parameters()`` and
            ``state_dict()``, and ``.to()`` / ``.double()`` / ``to_device`` of
            the Functor leave them unchanged; their device follows the
            registration device policy and their value is saved and restored
            through the manager.
        """
        held: Dict[str, nn.Parameter] = {}
        pending = list(reversed(tuple(vars(self).values())))
        visited_containers: set[int] = set()
        while pending:
            value = pending.pop()
            key = self.manager_binding_key(value)
            if key is not None:
                held[key] = value
            elif isinstance(value, (Mapping, list, tuple, set, frozenset)):
                identity = id(value)
                if identity in visited_containers:
                    continue
                visited_containers.add(identity)
                # Only traverse held containers, never another manager or module.
                # Shared containers and cycles must not duplicate live bindings.
                if isinstance(value, Mapping):
                    members = [member for pair in value.items() for member in pair]
                else:
                    members = list(value)
                pending.extend(reversed(members))
        return held

    def __setattr__(self, name: str, value: Any) -> None:
        """Hold a ParameterManager learnable binding by reference.

        An attribute whose value is, by object identity, a binding of this
        Functor's own ParameterManager is stored as a plain instance attribute
        and never registered as a module parameter: the manager stays its one
        owner, module conversions of this Functor cannot convert it in place,
        and every Functor holding it reads the same theta (gradients of all
        holders accumulate in that one binding). Any previous registration of
        the name is dropped. Every other value follows ``nn.Module``.
        """
        if self.manager_binding_key(value) is not None:
            for registry in ("_parameters", "_buffers", "_modules"):
                members = self.__dict__.get(registry)
                if members is not None:
                    members.pop(name, None)
            non_persistent = self.__dict__.get("_non_persistent_buffers_set")
            if non_persistent is not None:
                non_persistent.discard(name)
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Concise string representation for debugging."""
        deps_str = ", ".join(self._dependencies[:3])
        if len(self._dependencies) > 3:
            deps_str += f", ... ({len(self._dependencies)} total)"

        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"category={self.metadata.category.value}, "
                f"dependencies=[{deps_str}], "
                f"executions={self._execution_count}, "
                f"state={self.state.value})")

    def __str__(self) -> str:
        """Human-readable string representation for users."""
        lines = [
            f"Functor: {self.name}",
            f"  Category: {self.metadata.category.value}",
            f"  Version: {self.metadata.version}",
            f"  Description: {self.metadata.description}",
            f"  Execution Phase: {self.metadata.execution_phase.value}",
            f"  Dependencies: {', '.join(self._dependencies) if self._dependencies else 'none'}",
            f"  Parameters: {len(self.parameter_values)} configured",
            f"  Executions: {self._execution_count}",
            f"  State: {self.state.value}",
        ]

        if self.metadata.references:
            lines.append(f"  References: {len(self.metadata.references)} citations")

        if self.metadata.mathematical_expression:
            expr_preview = self.metadata.mathematical_expression[:80]
            if len(self.metadata.mathematical_expression) > 80:
                expr_preview += "..."
            lines.append(f"  Math: {expr_preview}")

        if self._error_count > 0:
            lines.append(f"  Errors: {self._error_count} execution failures")

        return "\n".join(lines)


# Export main classes and types
__all__ = [
    'FunctorBase',
    'FunctorMetadata',
    'ProcessCategory',
    'ExecutionPhase',
    'FunctorState',
    'FunctorException',
    'ConfigurationError',
    'DependencyError',
    'ExecutionError',
    'ConservationError',
    'ConfigDict',
    'ParameterDict',
    'MetadataDict',
    'ValidationErrors',
    'DependencyList'
]
