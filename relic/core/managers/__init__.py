"""Shared lifecycle contracts for explicit RELIC manager owners.

This package contains base lifecycle, metadata, and exception types. It does
not provide a generic manager registry, factory, health taxonomy, global
instance router, or string-dispatch creation path. Concrete owners are
constructed or injected through their domain and simulator contracts.
"""

from .base import (
    BaseManager,
    DeviceAwareManagerMixin,
    ManagerState,
)
from .exceptions import (
    ManagerDependencyError,
    ManagerError,
    ManagerInitializationError,
    ManagerNotFoundError,
    ManagerRegistrationError,
    ManagerStateError,
)
from .metadata import (
    CapabilityType,
    DependencyType,
    ManagerCapability,
    ManagerDependency,
    ManagerMetadata,
    ManagerResource,
    ResourceType,
)


__all__ = [
    "BaseManager",
    "DeviceAwareManagerMixin",
    "ManagerState",
    "ManagerMetadata",
    "ManagerCapability",
    "ManagerResource",
    "ManagerDependency",
    "CapabilityType",
    "ResourceType",
    "DependencyType",
    "ManagerError",
    "ManagerInitializationError",
    "ManagerRegistrationError",
    "ManagerNotFoundError",
    "ManagerDependencyError",
    "ManagerStateError",
]
