"""
Manager Metadata System

Defines comprehensive metadata structures for RELIC managers including
capabilities, dependencies, resources, and interoperability information.
Supports introspection, validation, and automated dependency resolution.

Mathematical Foundation:
    Metadata Space: M = {name, version, capabilities, dependencies, resources}
    Capability Function: C: Manager -> P(Capability) where P is power set
    Dependency Graph: G = (V, E) where V = managers, E = dependencies  
    Interoperability Matrix: A[i,j] = interoperable(manager_i, manager_j)

References:
    - Fowler, M. (2002). Patterns of Enterprise Application Architecture
    - Evans, E. (2003). Domain-Driven Design - Metadata patterns
    - Vernon, V. (2013). Implementing Domain-Driven Design - Context mapping
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime, timezone
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
import logging

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of capabilities that managers can provide."""
    
    # Core capabilities
    LIFECYCLE_MANAGEMENT = "lifecycle_management"
    STATE_MANAGEMENT = "state_management"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    
    # Device and compute capabilities  
    DEVICE_MANAGEMENT = "device_management"
    GPU_ACCELERATION = "gpu_acceleration"
    DISTRIBUTED_COMPUTING = "distributed_computing"
    PARALLEL_PROCESSING = "parallel_processing"
    
    # Data and I/O capabilities
    DATA_STREAMING = "data_streaming" 
    FILE_IO = "file_io"
    NETWORK_IO = "network_io"
    DATABASE_ACCESS = "database_access"
    
    # Scientific capabilities
    NUMERICAL_COMPUTATION = "numerical_computation"
    SPATIAL_OPERATIONS = "spatial_operations"
    TEMPORAL_PROCESSING = "temporal_processing"
    PARAMETER_MANAGEMENT = "parameter_management"
    
    # Specialized capabilities
    VERSION_CONTROL = "version_control"
    QUALITY_CONTROL = "quality_control"
    CONSTRAINT_ENFORCEMENT = "constraint_enforcement"
    BOUNDARY_CONDITIONS = "boundary_conditions"
    
    # Integration capabilities
    PYTORCH_INTEGRATION = "pytorch_integration"
    NUMPY_INTEGRATION = "numpy_integration"
    XARRAY_INTEGRATION = "xarray_integration"
    CF_CONVENTIONS = "cf_conventions"


class ResourceType(Enum):
    """Types of resources that managers handle."""
    
    # Memory resources
    SYSTEM_MEMORY = "system_memory"
    GPU_MEMORY = "gpu_memory" 
    SHARED_MEMORY = "shared_memory"
    
    # Compute resources
    CPU_CORES = "cpu_cores"
    GPU_DEVICES = "gpu_devices"
    DISTRIBUTED_PROCESSES = "distributed_processes"
    
    # Storage resources
    TEMPORARY_FILES = "temporary_files"
    PERSISTENT_STORAGE = "persistent_storage"
    CACHE_STORAGE = "cache_storage"
    
    # Network resources
    NETWORK_CONNECTIONS = "network_connections"
    MESSAGE_QUEUES = "message_queues"
    
    # Domain-specific resources
    PARAMETER_DATABASES = "parameter_databases"
    STATE_VARIABLES = "state_variables"
    SPATIAL_GRIDS = "spatial_grids"
    TEMPORAL_WINDOWS = "temporal_windows"


class DependencyType(Enum):
    """Types of dependencies between managers."""
    
    # Structural dependencies
    REQUIRED = "required"          # Hard dependency - must be present
    OPTIONAL = "optional"          # Soft dependency - enhances functionality
    ALTERNATIVE = "alternative"    # One of several alternatives needed
    
    # Temporal dependencies  
    INITIALIZATION_ORDER = "initialization_order"  # Must initialize before this
    RUNTIME_COUPLING = "runtime_coupling"         # Interact during execution
    CLEANUP_ORDER = "cleanup_order"              # Cleanup order dependency
    
    # Functional dependencies
    SERVICE_PROVIDER = "service_provider"        # Provides services to this manager
    EVENT_SOURCE = "event_source"              # Source of events/notifications
    DATA_PRODUCER = "data_producer"            # Produces data consumed by this


@dataclass
class ManagerCapability:
    """
    Represents a specific capability provided by a manager.
    
    Capabilities define what functionality a manager provides,
    including supported operations, data types, and quality levels.
    
    Attributes:
        type : CapabilityType
            The type of capability
        level : str
            Quality level ('basic', 'standard', 'advanced', 'expert')
        description : str
            Human-readable description
        parameters : Dict[str, Any]
            Capability-specific parameters
        version : str
            Version of the capability implementation
        quality_metrics : Dict[str, float]
            Performance and quality metrics
        
    Mathematical Expression:
        Capability: c = (type, level, params) in CapabilitySpace
        Capability matching: match(c_1, c_2) = interoperable(c_1.type, c_2.type)  level(c_1) >= level(c_2)
    """
    type: CapabilityType
    level: str = "standard"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate capability after creation."""
        valid_levels = {'basic', 'standard', 'advanced', 'expert'}
        if self.level not in valid_levels:
            raise ValueError(f"Invalid capability level: {self.level}. Must be one of {valid_levels}")
        try:
            Version(self.version)
        except InvalidVersion as exc:
            raise ValueError(f"Invalid capability version format '{self.version}': {exc}") from exc

    def matches_requirement(self, requirement: 'ManagerCapability') -> bool:
        """
        Check if this capability matches a requirement.
        
        Parameters:
            requirement : ManagerCapability
                Required capability to match against
                
        Returns:
            bool: True if this capability satisfies the requirement
        """
        # Type must match exactly
        if self.type != requirement.type:
            return False
        
        # Version must be interoperable (same major version). Versions are
        # validated at construction; an unparseable version raises here.
        if Version(self.version).major != Version(requirement.version).major:
            return False

        # Level interoperability (higher levels satisfy lower requirements)
        level_order = {'basic': 0, 'standard': 1, 'advanced': 2, 'expert': 3}
        self_level = level_order.get(self.level, 1)
        req_level = level_order.get(requirement.level, 1)
        
        return self_level >= req_level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type.value,
            'level': self.level,
            'description': self.description,
            'parameters': self.parameters,
            'version': self.version,
            'quality_metrics': self.quality_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagerCapability':
        """Create from dictionary."""
        data = data.copy()
        data['type'] = CapabilityType(data['type'])
        return cls(**data)


@dataclass
class ManagerResource:
    """
    Represents a resource managed by a manager.
    
    Resources define what system resources (memory, compute, storage, etc.)
    a manager uses or provides, enabling resource planning and conflict detection.
    
    Attributes:
        type : ResourceType
            Type of resource
        mode : str
            Access mode ('read', 'write', 'readwrite', 'exclusive')
        quantity : Optional[Union[int, float]]
            Amount of resource needed/provided
        unit : str
            Unit of measurement for quantity (e.g. 'bytes', 'MB'); required,
            there is no universal resource unit
        constraints : Dict[str, Any]
            Resource constraints and limits
        sharing_policy : str
            How resource can be shared ('exclusive', 'shared', 'pooled')
    """
    type: ResourceType
    unit: str
    mode: str = "readwrite"
    quantity: Optional[Union[int, float]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    sharing_policy: str = "shared"
    
    def __post_init__(self) -> None:
        """Validate resource specification."""
        valid_modes = {'read', 'write', 'readwrite', 'exclusive'}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid resource mode: {self.mode}. Must be one of {valid_modes}")
        
        valid_policies = {'exclusive', 'shared', 'pooled'}
        if self.sharing_policy not in valid_policies:
            raise ValueError(f"Invalid sharing policy: {self.sharing_policy}. Must be one of {valid_policies}")
    
    def conflicts_with(self, other: 'ManagerResource') -> bool:
        """
        Check if this resource conflicts with another.
        
        Parameters:
            other : ManagerResource
                Other resource to check conflict with
                
        Returns:
            bool: True if resources conflict
        """
        # Different resource types don't conflict
        if self.type != other.type:
            return False
        
        # Check sharing policy interoperability
        if self.sharing_policy == 'exclusive' or other.sharing_policy == 'exclusive':
            return True
        
        # Write operations may conflict
        if 'write' in self.mode and 'write' in other.mode:
            if self.sharing_policy != 'pooled' or other.sharing_policy != 'pooled':
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type.value,
            'mode': self.mode,
            'quantity': self.quantity,
            'unit': self.unit,
            'constraints': self.constraints,
            'sharing_policy': self.sharing_policy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagerResource':
        """Create from dictionary."""
        data = data.copy()
        data['type'] = ResourceType(data['type'])
        return cls(**data)


@dataclass
class ManagerDependency:
    """
    Represents a dependency between managers.
    
    Dependencies define the relationships between managers, including
    initialization order, runtime interactions, and service requirements.
    
    Attributes:
        target : str
            Name of the target manager this depends on
        type : DependencyType
            Type of dependency relationship
        version_constraint : Optional[str]
            Version constraint for the target manager
        optional : bool
            Whether dependency is optional
        runtime_coupling : bool
            Whether managers interact during runtime
        description : str
            Human-readable dependency description
    """
    target: str
    type: DependencyType
    version_constraint: Optional[str] = None
    optional: bool = False
    runtime_coupling: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the declared version constraint."""
        if self.version_constraint:
            self._version_specifier()

    def _version_specifier(self) -> SpecifierSet:
        """Parse version_constraint as a PEP 440 specifier set (e.g. '>=1.0.0', '>=1.0,<2.0').

        An unparseable constraint raises ValueError naming the dependency; it is
        never treated as satisfied.
        """
        try:
            return SpecifierSet(self.version_constraint)
        except InvalidSpecifier as exc:
            raise ValueError(
                f"Dependency on '{self.target}' has an invalid version constraint "
                f"'{self.version_constraint}': {exc}"
            ) from exc

    def is_satisfied_by(self, available_managers: Dict[str, 'ManagerMetadata']) -> bool:
        """
        Check if dependency is satisfied by available managers.

        Parameters:
            available_managers : Dict[str, ManagerMetadata]
                Available manager metadata by name

        Returns:
            bool: True if dependency is satisfied
        """
        # Check if target manager is available
        if self.target not in available_managers:
            return self.optional

        target_metadata = available_managers[self.target]

        # Check version constraint if specified. Pre-releases are compared by
        # their version order, as with the plain >= / == comparison.
        if self.version_constraint:
            return self._version_specifier().contains(
                Version(target_metadata.version), prereleases=True
            )

        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'target': self.target,
            'type': self.type.value,
            'version_constraint': self.version_constraint,
            'optional': self.optional,
            'runtime_coupling': self.runtime_coupling,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagerDependency':
        """Create from dictionary."""
        data = data.copy()
        data['type'] = DependencyType(data['type'])
        return cls(**data)


@dataclass
class ManagerMetadata:
    """
    Complete metadata for a manager including capabilities, dependencies, and resources.
    
    This class provides comprehensive information about a manager's functionality,
    requirements, and characteristics for registry management and dependency resolution.
    
    Attributes:
        name : str
            Unique manager name
        version : str
            Semantic version string
        description : str
            Human-readable description
        capabilities : List[ManagerCapability]
            Capabilities provided by this manager
        dependencies : List[ManagerDependency]
            Dependencies on other managers
        resources : List[ManagerResource]
            Resources managed by this manager
        tags : Set[str]
            Classification tags
        namespace : Optional[str]
            Namespace for organization
        author : str
            Author information
        documentation_url : Optional[str]
            URL to documentation
        created_at : datetime
            Creation timestamp
        updated_at : datetime  
            Last update timestamp
        custom_attributes : Dict[str, Any]
            Extension point for custom metadata
            
    Mathematical Expression:
        Manager Metadata: m = (name, version, C, D, R) where
        C = capabilities, D = dependencies, R = resources
        
        Interoperability: interoperable(m_1, m_2) = d in m_1.D: satisfied(d, m_2.C)
    """
    name: str
    version: str
    description: str = ""
    capabilities: List[ManagerCapability] = field(default_factory=list)
    dependencies: List[ManagerDependency] = field(default_factory=list)
    resources: List[ManagerResource] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    namespace: Optional[str] = None
    author: str = "Yuchong Long"
    documentation_url: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metadata after creation."""
        if not self.name:
            raise ValueError("Manager name cannot be empty")
        
        if not self.version:
            raise ValueError("Manager version cannot be empty")
        
        # Validate version format
        try:
            Version(self.version)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            raise ValueError(f"Invalid version format '{self.version}': {e}")
    
    def has_capability(self, capability_type: CapabilityType, level: Optional[str] = None) -> bool:
        """
        Check if manager has a specific capability.
        
        Parameters:
            capability_type : CapabilityType
                Type of capability to check
            level : Optional[str]
                Minimum capability level required
                
        Returns:
            bool: True if manager has the capability
        """
        for capability in self.capabilities:
            if capability.type == capability_type:
                if level is None:
                    return True
                
                # Check level interoperability
                level_order = {'basic': 0, 'standard': 1, 'advanced': 2, 'expert': 3}
                capability_level = level_order.get(capability.level, 1)
                required_level = level_order.get(level, 1)
                
                if capability_level >= required_level:
                    return True
        
        return False
    
    def get_capability(self, capability_type: CapabilityType) -> Optional[ManagerCapability]:
        """Get specific capability by type."""
        for capability in self.capabilities:
            if capability.type == capability_type:
                return capability
        return None
    
    def get_dependencies(
        self,
        dependency_type: Optional[DependencyType] = None,
    ) -> List[ManagerDependency]:
        """
        Get dependencies, optionally filtered by type.
        
        Parameters:
            dependency_type : Optional[DependencyType]
                Type of dependencies to retrieve
                
        Returns:
            List[ManagerDependency]: Matching dependencies
        """
        if dependency_type is None:
            return self.dependencies.copy()
        
        return [dep for dep in self.dependencies if dep.type == dependency_type]
    
    def get_resources(
        self,
        resource_type: Optional[ResourceType] = None,
    ) -> List[ManagerResource]:
        """
        Get resources, optionally filtered by type.
        
        Parameters:
            resource_type : Optional[ResourceType]
                Type of resources to retrieve
                
        Returns:
            List[ManagerResource]: Matching resources
        """
        if resource_type is None:
            return self.resources.copy()
        
        return [res for res in self.resources if res.type == resource_type]
    
    def check_interoperability(self, other: 'ManagerMetadata') -> Tuple[bool, List[str]]:
        """
        Check interoperability with another manager.
        
        Parameters:
            other : ManagerMetadata
                Other manager to check interoperability with
                
        Returns:
            Tuple[bool, List[str]]: (interoperable, list_of_issues)
        """
        issues = []
        
        # Check resource conflicts
        for my_resource in self.resources:
            for other_resource in other.resources:
                if my_resource.conflicts_with(other_resource):
                    issues.append(f"Resource conflict: {my_resource.type.value}")
        
        # Check if our dependencies are satisfied by other manager's capabilities
        {cap.type for cap in other.capabilities}
        for dependency in self.dependencies:
            if dependency.target == other.name:
                # This dependency targets the other manager
                # Check if other manager provides required capabilities
                # (This is a simplified check - more sophisticated matching could be added)
                pass
        
        # Check version interoperability (major version). Versions are validated
        # at construction; an unparseable version raises here.
        if Version(self.version).major != Version(other.version).major:
            issues.append(f"Major version interoperability mismatch: {self.version} vs {other.version}")

        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'capabilities': [cap.to_dict() for cap in self.capabilities],
            'dependencies': [dep.to_dict() for dep in self.dependencies],
            'resources': [res.to_dict() for res in self.resources],
            'tags': list(self.tags),
            'namespace': self.namespace,
            'author': self.author,
            'documentation_url': self.documentation_url,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'custom_attributes': self.custom_attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagerMetadata':
        """Create from dictionary."""
        data = data.copy()
        
        # Convert capabilities
        if 'capabilities' in data:
            data['capabilities'] = [
                ManagerCapability.from_dict(cap) for cap in data['capabilities']
            ]
        
        # Convert dependencies
        if 'dependencies' in data:
            data['dependencies'] = [
                ManagerDependency.from_dict(dep) for dep in data['dependencies']
            ]
        
        # Convert resources
        if 'resources' in data:
            data['resources'] = [
                ManagerResource.from_dict(res) for res in data['resources']
            ]
        
        # Convert datetime fields
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert tags to set
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = set(data['tags'])
        
        return cls(**data)


# Convenience functions for creating common metadata patterns
def create_basic_metadata(
    name: str,
    version: str,
    description: str = "",
    capabilities: Optional[List[CapabilityType]] = None,
    dependencies: Optional[List[str]] = None,
) -> ManagerMetadata:
    """
    Create basic manager metadata with common defaults.
    
    Parameters:
        name : str
            Manager name
        version : str
            Manager version
        description : str
            Manager description
        capabilities : List[CapabilityType]
            List of capability types this manager provides
        dependencies : List[str]
            List of manager names this manager depends on
            
    Returns:
        ManagerMetadata: Configured metadata instance
    """
    metadata = ManagerMetadata(name=name, version=version, description=description)
    
    if capabilities:
        for cap_type in capabilities:
            capability = ManagerCapability(type=cap_type, level="standard")
            metadata.capabilities.append(capability)
    
    if dependencies:
        for dep_name in dependencies:
            dependency = ManagerDependency(target=dep_name, type=DependencyType.REQUIRED)
            metadata.dependencies.append(dependency)
    
    return metadata


# Export all metadata components
__all__ = [
    'CapabilityType',
    'ResourceType',
    'DependencyType',
    'ManagerCapability',
    'ManagerResource',
    'ManagerDependency',
    'ManagerMetadata',
    'create_basic_metadata'
]
