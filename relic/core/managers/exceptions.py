"""
Manager Framework Exceptions

Defines the exception hierarchy for the RELIC manager unification framework.
Provides structured error handling with context preservation and recovery hints.

Exception Hierarchy:
    ManagerError
     ManagerInitializationError
     ManagerRegistrationError  
     ManagerNotFoundError
     ManagerDependencyError
     ManagerStateError
     ManagerConfigurationError
     ManagerDeviceError
     ManagerVersionError

Mathematical Expression:
    Error Context: E = {code, message, context, recovery_hints, timestamp}
    Error Propagation: E_1 -> E_2 where E_2 = transform(E_1) + additional_context
    
References:
    - McConnell, S. (2004). Code Complete - Error handling best practices
    - Hunt, A. & Thomas, D. (1999). The Pragmatic Programmer - Exception design
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum


class ManagerErrorCode(Enum):
    """Standard error codes for manager operations."""
    
    # Initialization errors (1000-1099)
    INITIALIZATION_FAILED = 1001
    CONFIGURATION_INVALID = 1002
    DEPENDENCIES_UNRESOLVED = 1003
    DEVICE_UNAVAILABLE = 1004
    RESOURCE_ALLOCATION_FAILED = 1005
    
    # Registration errors (1100-1199)
    REGISTRATION_FAILED = 1101
    DUPLICATE_REGISTRATION = 1102
    INVALID_MANAGER_CLASS = 1103
    VERSION_CONFLICT = 1104
    
    # Runtime errors (1200-1299)  
    MANAGER_NOT_FOUND = 1201
    INVALID_STATE_TRANSITION = 1202
    OPERATION_NOT_SUPPORTED = 1203
    DEPENDENCY_CYCLE = 1204
    TIMEOUT_EXCEEDED = 1205
    
    # Device errors (1300-1399)
    DEVICE_MIGRATION_FAILED = 1301
    CUDA_OUT_OF_MEMORY = 1302
    DEVICE_SYNCHRONIZATION_ERROR = 1303
    
    # Version errors (1400-1499)
    inconsistent_VERSION = 1401
    VERSION_NOT_SUPPORTED = 1403


class ManagerError(Exception):
    """
    Base exception for all manager framework errors.
    
    Provides structured error information with context preservation
    and recovery hints for robust error handling.
    
    Attributes:
        message : str
            Human-readable error message
        error_code : ManagerErrorCode
            Standardized error code
        context : Dict[str, Any]
            Additional context information
        recovery_hints : List[str]
            Suggested recovery actions
        cause : Optional[Exception]
            Original exception that caused this error
        timestamp : datetime
            When the error occurred
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ManagerErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_hints: Optional[List[str]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.recovery_hints = recovery_hints or []
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
        
        # Chain the original exception if provided
        if cause is not None:
            self.__cause__ = cause
    
    def add_context(self, key: str, value: Any) -> 'ManagerError':
        """Add context information to the error."""
        self.context[key] = value
        return self
    
    def add_recovery_hint(self, hint: str) -> 'ManagerError':
        """Add recovery hint to the error."""
        self.recovery_hints.append(hint)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'message': self.message,
            'error_code': self.error_code.value if self.error_code else None,
            'error_code_name': self.error_code.name if self.error_code else None,
            'context': self.context,
            'recovery_hints': self.recovery_hints,
            'timestamp': self.timestamp.isoformat(),
            'exception_type': self.__class__.__name__,
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """Enhanced string representation with context."""
        parts = [self.message]
        
        if self.error_code:
            parts.append(f"Error Code: {self.error_code.name} ({self.error_code.value})")
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        if self.recovery_hints:
            hints_str = "; ".join(self.recovery_hints)
            parts.append(f"Recovery Hints: {hints_str}")
        
        return " | ".join(parts)


class ManagerInitializationError(ManagerError):
    """
    Exception raised when manager initialization fails.
    
    This includes configuration errors, dependency resolution failures,
    and resource allocation problems during the initialization phase.
    
    Mathematical Expression:
        Initialization failure: I: S_0 -> ERROR where S_0 = UNINITIALIZED
        Recovery: ERROR -> S_0 -> S_1 (retry with different parameters)
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        failed_phase: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.INITIALIZATION_FAILED,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if failed_phase:
            self.add_context("failed_phase", failed_phase)
            
        # Add common recovery hints
        self.add_recovery_hint("Check manager configuration parameters")
        self.add_recovery_hint("Verify all dependencies are available")
        self.add_recovery_hint("Ensure sufficient system resources")


class ManagerRegistrationError(ManagerError):
    """
    Exception raised when manager registration fails.
    
    This includes duplicate registrations, invalid manager classes,
    and version conflicts in the registry.
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.REGISTRATION_FAILED,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if version:
            self.add_context("version", version)
            
        # Add recovery hints
        self.add_recovery_hint("Check if manager is already registered")
        self.add_recovery_hint("Verify manager class implements BaseManager")
        self.add_recovery_hint("Use different version string or unregister existing")


class ManagerNotFoundError(ManagerError):
    """
    Exception raised when a requested manager cannot be found.
    
    This includes missing registrations and failed resolution
    in the manager registry.
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        version: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.MANAGER_NOT_FOUND,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if version:
            self.add_context("version", version)
        if namespace:
            self.add_context("namespace", namespace)
            
        # Add recovery hints
        self.add_recovery_hint("Register the manager class first")
        self.add_recovery_hint("Check manager name spelling")
        self.add_recovery_hint("Verify version and namespace are correct")


class ManagerDependencyError(ManagerError):
    """
    Exception raised when manager dependencies cannot be resolved.
    
    This includes circular dependencies, missing dependencies,
    and version inconsistencies between managers.
    
    Mathematical Expression:
        Dependency graph: G = (V, E) where V = managers, E = dependencies
        Cycle detection:  path P = (v_1, v_2, ..., v, v_1) in G
        Resolution order: topological_sort(G) if acyclic else ERROR
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        missing_dependencies: Optional[List[str]] = None,
        cycle_path: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.DEPENDENCIES_UNRESOLVED,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if missing_dependencies:
            self.add_context("missing_dependencies", missing_dependencies)
        if cycle_path:
            self.add_context("cycle_path", cycle_path)
            
        # Add specific recovery hints
        if missing_dependencies:
            for dep in missing_dependencies:
                self.add_recovery_hint(f"Register missing dependency: {dep}")
        
        if cycle_path:
            self.add_recovery_hint("Break dependency cycle by refactoring manager interfaces")


class ManagerStateError(ManagerError):
    """
    Exception raised when manager state transitions are invalid.
    
    This includes operations on managers in invalid states and
    illegal state transitions.
    
    Mathematical Expression:
        State machine: S = {UNINITIALIZED, INITIALIZING, READY, ERROR, CLEANUP}
        Valid transitions: T  S x S
        Invalid transition: (s_1, s_2) not in T
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        current_state: Optional[str] = None,
        requested_state: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.INVALID_STATE_TRANSITION,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if current_state:
            self.add_context("current_state", current_state)
        if requested_state:
            self.add_context("requested_state", requested_state)
            
        # Add recovery hints
        self.add_recovery_hint("Check manager current state before operation")
        self.add_recovery_hint("Initialize manager if in UNINITIALIZED state")
        self.add_recovery_hint("Reset manager if in ERROR state")


class ManagerConfigurationError(ManagerError):
    """
    Exception raised when manager configuration is invalid.
    
    This includes missing required parameters, invalid parameter values,
    and configuration validation failures.
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        invalid_parameters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.CONFIGURATION_INVALID,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if invalid_parameters:
            self.add_context("invalid_parameters", invalid_parameters)
            
        # Add recovery hints
        self.add_recovery_hint("Validate configuration parameters against schema")
        self.add_recovery_hint("Check required parameters are provided")
        self.add_recovery_hint("Verify parameter types and value ranges")


class ManagerDeviceError(ManagerError):
    """
    Exception raised when device operations fail.
    
    This includes CUDA out of memory, device migration failures,
    and device synchronization errors.
    """
    
    def __init__(
        self,
        message: str,
        device: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.DEVICE_MIGRATION_FAILED,
            **kwargs
        )
        
        if device:
            self.add_context("device", device)
        if operation:
            self.add_context("operation", operation)
            
        # Add recovery hints
        self.add_recovery_hint("Check device availability and memory")
        self.add_recovery_hint("Try default to CPU if GPU operation failed")
        self.add_recovery_hint("Reduce batch size or model complexity")


class ManagerVersionError(ManagerError):
    """
    Exception raised when version-related operations fail.
    
    This includes inconsistent or unsupported version requests.
    """
    
    def __init__(
        self,
        message: str,
        requested_version: Optional[str] = None,
        available_versions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.inconsistent_VERSION,
            **kwargs
        )
        
        if requested_version:
            self.add_context("requested_version", requested_version)
        if available_versions:
            self.add_context("available_versions", available_versions)
            
        # Add recovery hints
        self.add_recovery_hint("Use supported version from available list")
        self.add_recovery_hint("Update client code for newer API version")
        self.add_recovery_hint("Use the default version configured in the registry")


class ManagerCreationError(ManagerError):
    """
    Exception raised when manager creation fails.
    
    This includes factory creation failures, dependency injection problems,
    and resource allocation issues during manager instantiation.
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.INITIALIZATION_FAILED,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if version:
            self.add_context("version", version)
            
        # Add recovery hints
        self.add_recovery_hint("Check manager registration in registry")
        self.add_recovery_hint("Verify all dependencies are satisfied")
        self.add_recovery_hint("Ensure sufficient resources for creation")


class ManagerValidationError(ManagerError):
    """
    Exception raised when manager validation fails.
    
    This includes state validation failures, schema violations,
    and constraint violations during operation.
    """
    
    def __init__(
        self,
        message: str,
        manager_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=ManagerErrorCode.OPERATION_NOT_SUPPORTED,
            **kwargs
        )
        
        if manager_name:
            self.add_context("manager_name", manager_name)
        if validation_errors:
            self.add_context("validation_errors", validation_errors)
            
        # Add recovery hints
        self.add_recovery_hint("Check state values against schema constraints")
        self.add_recovery_hint("Verify physical bounds are satisfied")
        self.add_recovery_hint("Ensure conservation laws are maintained")


# Convenience functions for common error patterns
def manager_not_found(
    name: str,
    version: Optional[str] = None,
    namespace: Optional[str] = None,
) -> ManagerNotFoundError:
    """Create a standard 'manager not found' error."""
    parts = [f"Manager '{name}' not found"]
    if version:
        parts.append(f"version '{version}'")
    if namespace:
        parts.append(f"in namespace '{namespace}'")
    
    message = " ".join(parts)
    return ManagerNotFoundError(message, name, version, namespace)


def initialization_failed(
    name: str,
    phase: str,
    cause: Optional[Exception] = None,
) -> ManagerInitializationError:
    """Create a standard initialization failure error."""
    message = f"Manager '{name}' initialization failed in phase '{phase}'"
    if cause:
        message += f": {cause}"
    
    return ManagerInitializationError(message, name, phase, cause=cause)


def invalid_state_transition(name: str, current: str, requested: str) -> ManagerStateError:
    """Create a standard invalid state transition error."""
    message = f"Manager '{name}' cannot transition from '{current}' to '{requested}'"
    return ManagerStateError(message, name, current, requested)


def dependency_cycle(cycle_path: List[str]) -> ManagerDependencyError:
    """Create a dependency cycle error."""
    path_str = " -> ".join(cycle_path + [cycle_path[0]])
    message = f"Circular dependency detected: {path_str}"
    return ManagerDependencyError(message, cycle_path=cycle_path)


def manager_creation_failed(
    name: str,
    version: Optional[str] = None,
    reason: Optional[str] = None,
    cause: Optional[Exception] = None,
) -> ManagerCreationError:
    """Create a standard manager creation failure error."""
    parts = [f"Failed to create manager '{name}'"]
    if version:
        parts.append(f"version '{version}'")
    if reason:
        parts.append(f": {reason}")
    
    message = " ".join(parts)
    return ManagerCreationError(message, name, version, cause=cause)


def configuration_invalid(
    name: str,
    invalid_params: Optional[List[str]] = None,
    cause: Optional[Exception] = None,
) -> ManagerConfigurationError:
    """Create a standard configuration validation error."""
    message = f"Invalid configuration for manager '{name}'"
    if invalid_params:
        message += f" (invalid parameters: {', '.join(invalid_params)})"
    
    return ManagerConfigurationError(message, name, invalid_params, cause=cause)


# Export all exceptions
__all__ = [
    'ManagerErrorCode',
    'ManagerError',
    'ManagerInitializationError',
    'ManagerRegistrationError',
    'ManagerNotFoundError',
    'ManagerDependencyError', 
    'ManagerStateError',
    'ManagerConfigurationError',
    'ManagerDeviceError',
    'ManagerVersionError',
    'ManagerCreationError',
    'ManagerValidationError',
    
    # Convenience functions
    'manager_not_found',
    'initialization_failed', 
    'invalid_state_transition',
    'dependency_cycle',
    'manager_creation_failed',
    'configuration_invalid'
]

