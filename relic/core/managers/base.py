"""
Base Manager Classes

Defines the foundational classes for the RELIC manager unification framework.
Provides the abstract base class with standard lifecycle management and the
device-awareness mixin.

Key Classes:
    - BaseManager: Core abstract base class for all managers
    - DeviceAwareManagerMixin: Mixin for GPU/CPU device management

Mathematical Foundation:
    Manager State Space: S = {UNINITIALIZED, INITIALIZING, READY, ERROR, CLEANUP}
    State Transitions: T  S x S with valid paths
    Lifecycle Function: L: S_0 -> S_1 -> S_2 -> S_3 -> S_4
    
    Device Operations: D = {device, tensors, operations}
    Device Migration: M: (D_1, device_2) -> D_2
    
References:
    - Gamma, E. et al. (1994). Design Patterns - Template Method, State
    - Fowler, M. (2002). Patterns of Enterprise Application Architecture
    - PyTorch device semantics and accelerator best practices
"""

import abc
import threading
from types import TracebackType
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime, timezone
import logging

from relic.core import device_policy
from torch import device as torch_device

TORCH_AVAILABLE = True

from .exceptions import (
    ManagerError,
    ManagerInitializationError,
    ManagerStateError,
    ManagerDeviceError,
    invalid_state_transition,
    initialization_failed
)
from .metadata import ManagerMetadata

logger = logging.getLogger(__name__)


class ManagerState(Enum):
    """
    Manager lifecycle states with well-defined transition rules.
    
    State Transition Matrix:
        UNINITIALIZED -> INITIALIZING -> READY -> CLEANUP -> UNINITIALIZED
        Any State -> ERROR -> UNINITIALIZED (recovery path)
    
    Mathematical Expression:
        Valid transitions: T = {
            (UNINITIALIZED, INITIALIZING),
            (INITIALIZING, READY), (INITIALIZING, ERROR),
            (READY, CLEANUP), (READY, ERROR),
            (CLEANUP, UNINITIALIZED), (CLEANUP, ERROR),
            (ERROR, UNINITIALIZED)
        }
    """
    UNINITIALIZED = "uninitialized"  # Initial state, not yet configured
    INITIALIZING = "initializing"    # In process of initialization
    READY = "ready"                  # Fully operational state
    ERROR = "error"                  # Error state requiring intervention
    CLEANUP = "cleanup"              # Cleaning up resources before shutdown


class BaseManager(abc.ABC):
    """
    Abstract base class for all RELIC managers.
    
    Provides standardized lifecycle management, state tracking, metadata
    handling, and error recovery patterns. All concrete managers should
    inherit from this class or its specialized variants.
    
    Lifecycle Pattern:
        1. __init__() - Basic object creation
        2. initialize() - Resource allocation and setup
        3. configure() - Parameter and dependency configuration  
        4. validate() - Validation of configuration and state
        5. [Runtime operations via execute() or specific methods]
        6. cleanup() - Resource cleanup and shutdown
        
    Thread Safety:
        - All state transitions are protected by _state_lock
        - Subclasses should use _state_lock for critical sections
        - Lifecycle methods are automatically synchronized
    
    Mathematical Expression:
        Lifecycle: L = initialize  configure  validate  execute*  cleanup
        State invariant: t: state(t) in valid_states
        Error recovery: error(state) -> reset() -> initialize()
    """
    
    def __init__(self, metadata: Optional[ManagerMetadata] = None):
        """
        Initialize base manager with metadata and state tracking.
        
        Parameters:
            metadata : Optional[ManagerMetadata]
                Manager metadata (auto-generated if not provided)
        """
        # Core state management
        self._state = ManagerState.UNINITIALIZED
        self._state_lock = threading.RLock()  # Reentrant lock for nested calls
        self._error_info: Optional[ManagerError] = None
        
        # Metadata and identification
        self._metadata = metadata or self._create_default_metadata()
        self._instance_id = id(self)  # Unique instance identifier
        self._created_at = datetime.now(timezone.utc)
        self._initialized_at: Optional[datetime] = None
        
        # Configuration and dependencies
        self._config: Dict[str, Any] = {}
        self._dependencies: Dict[str, 'BaseManager'] = {}
        
        # Performance tracking
        self._operation_count = 0
        self._last_operation_time: Optional[datetime] = None
        
        logger.debug(f"Created {self.__class__.__name__} instance {self._instance_id}")
    
    # Abstract methods that subclasses must implement
    
    @abc.abstractmethod
    def _do_initialize(self) -> None:
        """
        Perform manager-specific initialization.
        
        This method should:
        - Allocate required resources
        - Set up internal data structures
        - Prepare for configuration phase
        
        Raises:
            ManagerInitializationError: If initialization fails
        """
    
    @abc.abstractmethod
    def _do_configure(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration to the manager.
        
        This method should:
        - Apply configuration parameters
        - Resolve dependencies
        - Prepare for validation
        
        Parameters:
            config : Dict[str, Any]
                The settings returned by _validate_config for this call
                
        Raises:
            Any error raised here is a lifecycle failure: configure() moves the
            manager to ERROR and raises ManagerInitializationError (phase
            'configuration'); reset() recovers.
        """

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the settings of one configure() call before anything is applied.

        configure() calls this before it stores or applies any setting and outside
        its lifecycle-failure handling: an error raised here (a rejected input, for
        example a ValueError naming the setting) reaches the caller unchanged, and
        the lifecycle state, the stored configuration and every setting stay as
        they were, so the next valid configure() proceeds. Subclasses that declare
        settings override this; the base class declares none and returns the
        call's settings unchanged.

        Parameters:
            config : Dict[str, Any]
                The settings of this configure() call (a copy)

        Returns:
            Dict[str, Any]: The settings that _do_configure applies
        """
        return config
    
    @abc.abstractmethod
    def _do_validate(self) -> List[str]:
        """
        Validate manager state and configuration.
        
        This method should:
        - Check configuration validity
        - Verify dependencies are satisfied
        - Validate internal state consistency
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
    
    @abc.abstractmethod
    def _do_cleanup(self) -> None:
        """
        Perform manager-specific cleanup.
        
        This method should:
        - Release allocated resources
        - Close connections and handles
        - Reset internal state
        """
    
    # Concrete lifecycle methods with state management
    
    def initialize(self) -> None:
        """
        Initialize the manager with thread-safe state management.
        
        Raises:
            ManagerStateError: If manager is not in UNINITIALIZED state
            ManagerInitializationError: If initialization fails
        """
        with self._state_lock:
            if self._state != ManagerState.UNINITIALIZED:
                raise invalid_state_transition(
                    self._metadata.name, 
                    self._state.value, 
                    ManagerState.INITIALIZING.value
                )
            
            self._state = ManagerState.INITIALIZING
            
            try:
                logger.info(f"Initializing manager: {self._metadata.name}")
                
                # Perform manager-specific initialization
                self._do_initialize()
                
                self._initialized_at = datetime.now(timezone.utc)
                logger.info(f"Manager {self._metadata.name} initialized successfully")
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                self._state = ManagerState.ERROR
                self._error_info = initialization_failed(
                    self._metadata.name, 
                    "initialization", 
                    cause=e
                )
                logger.error(f"Manager {self._metadata.name} initialization failed: {e}")
                raise self._error_info
    
    def configure(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Configure the manager with provided parameters.
        
        Parameters:
            config : Optional[Dict[str, Any]]
                Configuration parameters
                
        Raises:
            ManagerStateError: If manager is not in INITIALIZING state
            ValueError: (or another error of _validate_config) If the call's
                settings are rejected; nothing is stored or applied, the state
                is unchanged, and a later valid call proceeds
            ManagerInitializationError: If applying valid settings fails; the
                manager is in ERROR until reset()
        """
        with self._state_lock:
            if self._state != ManagerState.INITIALIZING:
                raise invalid_state_transition(
                    self._metadata.name,
                    self._state.value,
                    "configure"
                )
            # Input validation precedes every change and is not a lifecycle failure.
            config = dict(config or {})
            settings = self._validate_config(config)
            
            try:
                self._config.update(config)
                
                logger.debug(f"Configuring manager: {self._metadata.name}")
                
                # Perform manager-specific configuration
                self._do_configure(settings)
                
                logger.debug(f"Manager {self._metadata.name} configured successfully")
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                self._state = ManagerState.ERROR
                self._error_info = initialization_failed(
                    self._metadata.name,
                    "configuration", 
                    cause=e
                )
                logger.error(f"Manager {self._metadata.name} configuration failed: {e}")
                raise self._error_info
    
    def validate(self) -> None:
        """
        Validate manager configuration and state.
        
        Raises:
            ManagerStateError: If manager is not in INITIALIZING state
            ManagerInitializationError: If validation fails
        """
        with self._state_lock:
            if self._state != ManagerState.INITIALIZING:
                raise invalid_state_transition(
                    self._metadata.name,
                    self._state.value,
                    "validate"
                )
            
            try:
                logger.debug(f"Validating manager: {self._metadata.name}")
                
                # Perform manager-specific validation
                validation_errors = self._do_validate()
                
                if validation_errors:
                    error_msg = f"Validation failed for {self._metadata.name}: {'; '.join(validation_errors)}"
                    raise ManagerInitializationError(
                        error_msg,
                        manager_name=self._metadata.name,
                        failed_phase="validation"
                    )
                
                # Transition to READY state
                self._state = ManagerState.READY
                logger.info(f"Manager {self._metadata.name} is ready")
                
            except ManagerInitializationError:
                self._state = ManagerState.ERROR
                raise
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                self._state = ManagerState.ERROR
                self._error_info = initialization_failed(
                    self._metadata.name,
                    "validation",
                    cause=e
                )
                logger.error(f"Manager {self._metadata.name} validation failed: {e}")
                raise self._error_info
    
    def cleanup(self) -> None:
        """
        Clean up manager resources and reset state.

        This method may be called from any state. When _do_cleanup() succeeds
        the manager returns to UNINITIALIZED. When it fails the manager stays in
        ERROR, error_info holds a ManagerError whose cause is the original
        error, and that ManagerError is raised. When the cleanup is interrupted
        (KeyboardInterrupt, SystemExit, task cancellation) the manager is also
        left in ERROR and the interruption propagates unchanged. In both cases a
        later cleanup() or reset() retries the cleanup instead of returning at
        the CLEANUP re-entry guard.

        Raises:
            ManagerError: If _do_cleanup() fails
        """
        with self._state_lock:
            if self._state == ManagerState.CLEANUP:
                return  # Re-entrant call during this cleanup

            self._state = ManagerState.CLEANUP
            logger.info(f"Cleaning up manager: {self._metadata.name}")

            try:
                self._do_cleanup()
            except Exception as exc:
                self._state = ManagerState.ERROR
                self._error_info = ManagerError(
                    f"Manager '{self._metadata.name}' cleanup failed: {exc}",
                    context={"manager_name": self._metadata.name, "failed_phase": "cleanup"},
                    recovery_hints=["Fix the failing resource release, then call reset() to retry the cleanup"],
                    cause=exc,
                )
                logger.error(self._error_info.message)
                raise self._error_info
            except BaseException:
                # The cleanup did not finish; ERROR (not CLEANUP) lets the next
                # cleanup() run again. The interruption itself propagates.
                self._state = ManagerState.ERROR
                raise

            self._state = ManagerState.UNINITIALIZED
            self._error_info = None
            self._initialized_at = None
            logger.info(f"Manager {self._metadata.name} cleaned up successfully")
    
    def reset(self) -> None:
        """
        Reset manager to initial state, performing cleanup if necessary.

        This is the primary recovery method for error conditions. A cleanup
        failure propagates (ManagerError) and the manager stays in ERROR with
        its configuration kept, so a later reset() retries the cleanup.
        """
        with self._state_lock:
            if self._state != ManagerState.UNINITIALIZED:
                self.cleanup()
            
            # Ensure we're in uninitialized state
            self._state = ManagerState.UNINITIALIZED
            self._error_info = None
            self._config.clear()
            self._dependencies.clear()
            
            logger.info(f"Manager {self._metadata.name} reset to initial state")
    
    # State and metadata access methods
    
    @property
    def state(self) -> ManagerState:
        """Get current manager state."""
        with self._state_lock:
            return self._state
    
    @property
    def is_ready(self) -> bool:
        """Check if manager is in READY state."""
        return self.state == ManagerState.READY
    
    @property
    def is_error(self) -> bool:
        """Check if manager is in ERROR state."""
        return self.state == ManagerState.ERROR
    
    @property
    def error_info(self) -> Optional[ManagerError]:
        """Get error information if manager is in ERROR state."""
        with self._state_lock:
            return self._error_info
    
    @property
    def metadata(self) -> ManagerMetadata:
        """Get manager metadata."""
        return self._metadata
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration (read-only)."""
        with self._state_lock:
            return self._config.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive manager status information.
        
        Returns:
            Dict[str, Any]: Status information including state, metadata, performance
        """
        with self._state_lock:
            status = {
                'name': self._metadata.name,
                'version': self._metadata.version,
                'state': self._state.value,
                'is_ready': self.is_ready,
                'is_error': self.is_error,
                'instance_id': self._instance_id,
                'created_at': self._created_at.isoformat(),
                'initialized_at': self._initialized_at.isoformat() if self._initialized_at else None,
                'operation_count': self._operation_count,
                'last_operation_time': self._last_operation_time.isoformat() if self._last_operation_time else None,
                'config_keys': list(self._config.keys()),
                'dependency_count': len(self._dependencies),
                'capabilities': [cap.type.value for cap in self._metadata.capabilities],
                'error_info': self._error_info.to_dict() if self._error_info else None
            }
            return status
    
    # Dependency management
    
    def add_dependency(self, name: str, manager: 'BaseManager') -> None:
        """
        Add a dependency manager.
        
        Parameters:
            name : str
                Dependency name
            manager : BaseManager
                Manager instance to depend on
        """
        with self._state_lock:
            self._dependencies[name] = manager
            logger.debug(f"Added dependency {name} to manager {self._metadata.name}")
    
    def get_dependency(self, name: str) -> Optional['BaseManager']:
        """
        Get a dependency manager by name.
        
        Parameters:
            name : str
                Dependency name
                
        Returns:
            Optional[BaseManager]: Dependency manager or None
        """
        with self._state_lock:
            return self._dependencies.get(name)
    
    def remove_dependency(self, name: str) -> None:
        """
        Remove a dependency manager.
        
        Parameters:
            name : str
                Dependency name to remove
        """
        with self._state_lock:
            if name in self._dependencies:
                del self._dependencies[name]
                logger.debug(f"Removed dependency {name} from manager {self._metadata.name}")
    
    # Template methods for common patterns
    
    def _record_operation(self) -> None:
        """Record operation for performance tracking."""
        with self._state_lock:
            self._operation_count += 1
            self._last_operation_time = datetime.now(timezone.utc)
    
    def _ensure_ready(self) -> None:
        """
        Ensure manager is in READY state for operations.
        
        Raises:
            ManagerStateError: If manager is not ready
        """
        if not self.is_ready:
            raise ManagerStateError(
                f"Manager {self._metadata.name} is not ready (current state: {self._state.value})",
                manager_name=self._metadata.name,
                current_state=self._state.value,
                requested_state="READY"
            )
    
    def _create_default_metadata(self) -> ManagerMetadata:
        """Create default metadata for managers without explicit metadata."""
        from .metadata import create_basic_metadata
        
        return create_basic_metadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description=f"Auto-generated metadata for {self.__class__.__name__}"
        )
    
    # Context manager support for automatic cleanup
    
    def __enter__(self) -> 'BaseManager':
        """Enter context manager - ensure manager is ready."""
        if not self.is_ready:
            self.initialize()
            self.configure()
            self.validate()
        return self
    
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        """Exit context manager - perform cleanup."""
        self.cleanup()
    
    # String representation
    
    def __repr__(self) -> str:
        """String representation of manager."""
        return f"{self.__class__.__name__}(name={self._metadata.name}, state={self._state.value})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self._metadata.name} ({self._state.value})"


class DeviceAwareManagerMixin:
    """
    Mixin class for GPU/CPU device management in managers.
    
    Provides device detection, migration, and memory management
    capabilities integrated with PyTorch device semantics.
    
    Mathematical Expression:
        Device Space: D = {cpu} union {cuda:i | i in [0, n_gpus)}
        Migration: migrate: (tensors, D_1) -> (tensors', D_2)
        Memory Constraint: memory_used(D) <= memory_available(D)
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch_device]] = None,
        *args: Any,
        **kwargs: Any
    ):
        """
        Initialize device-aware manager.
        
        Parameters:
            device : Optional[Union[str, torch.device]]
                Target device ('cpu', 'cuda', 'cuda:0', etc.)
        """
        super().__init__(*args, **kwargs)
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - device management disabled")
            self._device = None
            self._device_available = False
            return
        
        # Device management
        self._device_available = True
        self._device = self._resolve_device(device)
        self._device_lock = threading.Lock()
        
        logger.debug(f"DeviceAwareManager initialized with device: {self._device}")
    
    def _resolve_device(self, device: Optional[Union[str, torch_device]]) -> Optional[torch_device]:
        """
        Resolve device specification to torch.device.
        
        Parameters:
            device : Optional[Union[str, torch.device]]
                Device specification
                
        Returns:
            torch.device: Resolved device
        """
        if not TORCH_AVAILABLE:
            return None
        
        return device_policy.resolve_device(
            explicit_device=device,
            module_name=self.__class__.__module__,
            default="auto",
        )
    
    @property
    def device(self) -> Optional[torch_device]:
        """Get current device."""
        return self._device
    
    @property
    def device_type(self) -> Optional[str]:
        """Get device type string."""
        return self._device.type if self._device else None
    
    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self._device is not None and self._device.type == 'cuda'
    
    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self._device is not None and self._device.type == 'cpu'
    
    def to_device(self, device: Union[str, torch_device]) -> None:
        """
        Migrate manager to different device.
        
        Parameters:
            device : Union[str, torch.device]
                Target device
                
        Raises:
            ManagerDeviceError: If device migration fails
        """
        if not self._device_available:
            raise ManagerDeviceError("Device management not available (PyTorch not found)")
        
        with self._device_lock:
            try:
                new_device = self._resolve_device(device)
                
                if new_device == self._device:
                    return  # Already on target device
                
                logger.info(f"Migrating manager from {self._device} to {new_device}")
                
                old_device = self._device
                self._device = new_device
                
                # Call device migration hook for subclasses
                if hasattr(self, '_on_device_changed'):
                    self._on_device_changed(old_device, new_device)
                
                logger.info(f"Successfully migrated to {new_device}")
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                error = ManagerDeviceError(
                    f"Device migration failed: {e}",
                    device=str(device),
                    operation="migration"
                )
                logger.error(str(error))
                raise error


# Export all base classes
__all__ = [
    'ManagerState',
    'BaseManager', 
    'DeviceAwareManagerMixin'
]
