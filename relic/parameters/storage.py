"""
Parameter Storage Component for RELIC

This module provides centralized parameter storage with CRUD operations,
namespace management, indexing, and snapshot/diff capabilities.

It serves as the data layer for the unified parameter management system,
handling the physical storage and retrieval of parameter values while
delegating validation and serialization to specialized components.

Key Features:
    - CRUD operations with namespace support
    - Hierarchical parameter organization
    - Fast key-based lookups with indexing
    - Snapshot and diff capabilities
    - Thread-safe operations
    - Memory-efficient storage

Design Patterns:
    - Repository Pattern for data access
    - Memento Pattern for snapshots
    - Strategy Pattern for storage backends

References:
    - Fowler, M. (2002). Patterns of Enterprise Application Architecture
    - Evans, E. (2003). Domain-Driven Design
"""

import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import logging
import types

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Type aliases
ParameterValue = Union[float, int, str, bool, Tensor]
ParameterDict = Dict[str, ParameterValue]
NamespaceDict = Dict[str, ParameterDict]


@dataclass(frozen=True)
class ParameterSnapshot:
    """
    Immutable snapshot of parameter state.
    
    Captures the complete parameter state at a point in time for
    debugging, rollback, and diff operations.
    
    Attributes:
        timestamp : datetime
            When the snapshot was taken
        _parameters : ParameterDict
            Frozen copy of all parameters
        metadata : Dict[str, Any]
            Additional snapshot metadata
        description : str
            Human-readable description
    """
    timestamp: datetime
    _parameters: ParameterDict = field(repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def __post_init__(self):
        """Ensure immutability by deep copying - use object.__setattr__ for frozen dataclass."""
        object.__setattr__(self, '_parameters', deepcopy(self._parameters))
        object.__setattr__(self, 'metadata', deepcopy(self.metadata))
    
    @property
    def parameters(self) -> types.MappingProxyType:
        """Return a read-only projection whose nested values cannot alter the snapshot."""
        return types.MappingProxyType(deepcopy(self._parameters))
    



class ParameterStorage:
    """
    Centralized parameter storage with namespace management.
    
    Provides thread-safe CRUD operations, namespace organization,
    efficient indexing, and snapshot capabilities for parameter management.
    
    This class serves as the data layer, focusing solely on storage
    and retrieval without concerning itself with validation or serialization.
    
    Architecture:
        - Hierarchical namespace organization
        - Key-based indexing for O(1) lookups
        - Thread-safe operations with fine-grained locking
        - Copy-on-write for snapshots
        
    Mathematical Model:
        Storage space: S = {(k, v) | k in Keys, v in Values}
        Namespace mapping: N : Keys -> Namespaces
        Lookup complexity: O(1) average, O(n) worst case
        
    Examples:
        >>> storage = ParameterStorage()
        >>> storage.set("biosphere.photosynthesis.vcmax25", 60.0)
        >>> value = storage.get("biosphere.photosynthesis.vcmax25")
        >>> params = storage.list(namespace="biosphere.photosynthesis")
    """
    
    def __init__(self):
        """Initialize parameter storage."""
        # Main storage dictionary
        self._parameters: ParameterDict = {}
        
        # Namespace index for fast filtering: every ancestor namespace of a key
        # ("a" and "a.b" for "a.b.c") maps to all keys below it.
        self._namespace_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Snapshot management
        self._snapshots: List[ParameterSnapshot] = []

        # Restore callbacks for cache/unit sync
        self._restore_callbacks: List = []
        
        # Statistics
        self._access_count: Dict[str, int] = defaultdict(int)
        
        logger.debug("ParameterStorage initialized")
    
    def get(self, key: str, default: Optional[ParameterValue] = None) -> Optional[ParameterValue]:
        """
        Retrieve parameter value by key.
        
        Parameters:
            key : str
                Parameter key in dot notation
            default : Optional[ParameterValue]
                Default value if key not found
                
        Returns:
            Optional[ParameterValue]: Parameter value or default
            
        Examples:
            >>> storage.get("biosphere.photosynthesis.vcmax25", default=50.0)
            60.0
        """
        with self._lock:
            self._access_count[key] += 1
            return self._parameters.get(key, default)
    
    def set(self, key: str, value: ParameterValue) -> None:
        """
        Store parameter value with key.
        
        Parameters:
            key : str
                Parameter key in dot notation
            value : ParameterValue
                Parameter value to store
                
        Examples:
            >>> storage.set("biosphere.photosynthesis.jmax25", 120.0)
        """
        with self._lock:
            # Store value
            self._parameters[key] = value
            
            # Update namespace index
            for namespace in extract_namespace_hierarchy(key):
                self._namespace_index[namespace].add(key)

            logger.debug(f"Set parameter {key} = {value}")
    
    def delete(self, key: str) -> bool:
        """
        Remove parameter from storage.
        
        Parameters:
            key : str
                Parameter key to remove
                
        Returns:
            bool: True if deleted, False if not found
            
        Examples:
            >>> storage.delete("obsolete.parameter")
            True
        """
        with self._lock:
            if key in self._parameters:
                del self._parameters[key]
                
                # Update namespace index
                for namespace in extract_namespace_hierarchy(key):
                    members = self._namespace_index.get(namespace)
                    if members is not None:
                        members.discard(key)
                        if not members:
                            del self._namespace_index[namespace]
                
                logger.debug(f"Deleted parameter {key}")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if parameter exists.
        
        Parameters:
            key : str
                Parameter key to check
                
        Returns:
            bool: True if exists
            
        Examples:
            >>> storage.exists("biosphere.photosynthesis.vcmax25")
            True
        """
        with self._lock:
            return key in self._parameters
    
    def list(self, 
             namespace: Optional[str] = None,
             pattern: Optional[str] = None) -> List[str]:
        """
        List parameter keys with optional filtering.
        
        Parameters:
            namespace : Optional[str]
                Filter by namespace prefix
            pattern : Optional[str]
                Filter by pattern (supports wildcards)
                
        Returns:
            List[str]: Matching parameter keys
            
        Examples:
            >>> storage.list(namespace="biosphere.photosynthesis")
            ['biosphere.photosynthesis.vcmax25', 'biosphere.photosynthesis.jmax25']
            
            >>> storage.list(pattern="*vcmax*")
            ['biosphere.photosynthesis.vcmax25', 'atmosphere.chemistry.vcmax_ref']
        """
        with self._lock:
            keys = set(self._parameters.keys())
            
            # Filter by namespace: every key below "<namespace>." (one rule for
            # every namespace; the index holds each key under all its ancestors).
            if namespace:
                keys &= self._namespace_index.get(namespace, set())
            
            # Filter by pattern
            if pattern:
                import fnmatch
                keys = {k for k in keys if fnmatch.fnmatch(k, pattern)}
            
            return sorted(list(keys))
    
    def get_all(self) -> ParameterDict:
        """
        Get all parameters as dictionary.
        
        Returns:
            ParameterDict: Copy of all parameters
            
        Examples:
            >>> all_params = storage.get_all()
            >>> len(all_params)
            245
        """
        with self._lock:
            return self._parameters.copy()
    
    def clear(self) -> None:
        """
        Remove all parameters.
        
        Examples:
            >>> storage.clear()
            >>> storage.list()
            []
        """
        with self._lock:
            self._parameters.clear()
            self._namespace_index.clear()
            logger.info("Cleared all parameters from storage")
    
    def update(self, parameters: ParameterDict) -> None:
        """
        Bulk update multiple parameters.
        
        Parameters:
            parameters : ParameterDict
                Dictionary of parameters to update
                
        Examples:
            >>> storage.update({
            ...     "param1": 1.0,
            ...     "param2": 2.0
            ... })
        """
        with self._lock:
            for key, value in parameters.items():
                self.set(key, value)
    
    def snapshot(
        self,
        description: str = "",
        *,
        parameters: Optional[ParameterDict] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParameterSnapshot:
        """
        Create an immutable snapshot of values, each kept as it is (type and dtype).

        Parameters:
            description : str
                Description of snapshot
            parameters : Optional[ParameterDict]
                Explicit values to capture instead of the stored values, without
                changing live storage. None captures storage; an empty mapping
                captures an empty state.
            metadata : Optional[Dict[str, Any]]
                Extra metadata recorded with the snapshot (deep-copied), for example
                the live learnable values the parameter manager captures. The keys
                ``parameter_count`` and ``namespace_count`` are set by the storage.

        Returns:
            ParameterSnapshot: Immutable snapshot

        Examples:
            >>> snapshot = storage.snapshot("Before optimization")
            >>> snapshot.timestamp
            datetime.datetime(2024, 1, 15, 10, 30, 0)
        """
        with self._lock:
            captured = self._parameters if parameters is None else parameters
            namespaces = {self._extract_namespace(key) for key in captured}
            namespaces.discard(None)
            snapshot = ParameterSnapshot(
                timestamp=datetime.now(),
                _parameters=dict(captured),
                metadata={
                    **(metadata or {}),
                    "parameter_count": len(captured),
                    "namespace_count": len(namespaces),
                },
                description=description
            )
            self._snapshots.append(snapshot)
            logger.info(f"Created snapshot: {description}")
            return snapshot

    def register_restore_callback(self, callback) -> None:
        """
        Register a callback to be invoked after restore operations.

        The callback signature should be callback(snapshot: ParameterSnapshot) -> None.
        A callback validates the snapshot before it changes its own owner and
        raises to reject it; the restore then fails with that error and the
        storage takes back its previous values.
        """
        if callback not in self._restore_callbacks:
            self._restore_callbacks.append(callback)

    def restore(self, snapshot: ParameterSnapshot) -> None:
        """
        Restore parameters from snapshot.

        The stored values are replaced by the snapshot values, then every
        restore callback runs. When a callback rejects the snapshot, its error
        propagates unchanged and the storage takes back the values it held
        before the restore.

        Parameters:
            snapshot : ParameterSnapshot
                Snapshot to restore from

        Examples:
            >>> storage.restore(snapshot)
        """
        with self._lock:
            previous_parameters = dict(self._parameters)
            restored_parameters = dict(snapshot.parameters)
            self.clear()
            self.update(restored_parameters)
        try:
            for callback in self._restore_callbacks:
                callback(snapshot)
        except BaseException:
            with self._lock:
                self.clear()
                self.update(previous_parameters)
            raise
        logger.info(f"Restored from snapshot: {snapshot.description}")
    
    def diff(self, 
             snapshot1: Optional[ParameterSnapshot] = None,
             snapshot2: Optional[ParameterSnapshot] = None) -> Dict[str, Tuple[Any, Any]]:
        """
        Compare two snapshots or current state with snapshot.
        
        Parameters:
            snapshot1 : Optional[ParameterSnapshot]
                First snapshot (None for current state)
            snapshot2 : Optional[ParameterSnapshot]
                Second snapshot (None for current state)
                
        Returns:
            Dict[str, Tuple[Any, Any]]: Changed parameters {key: (old_value, new_value)}
            
        Examples:
            >>> diff = storage.diff(old_snapshot, None)  # Compare with current
            >>> diff
            {'param1': (1.0, 2.0), 'param2': (None, 3.0)}
        """
        with self._lock:
            params1 = snapshot1._parameters if snapshot1 else self._parameters.copy()
            params2 = snapshot2._parameters if snapshot2 else self._parameters.copy()
            
            all_keys = set(params1.keys()) | set(params2.keys())
            diff = {}
            
            for key in all_keys:
                val1 = params1.get(key)
                val2 = params2.get(key)
                if not self._values_equal(val1, val2):
                    diff[key] = (val1, val2)
            
            return diff
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dict[str, Any]: Statistics including counts and access patterns
            
        Examples:
            >>> stats = storage.get_statistics()
            >>> stats['total_parameters']
            245
        """
        with self._lock:
            return {
                "total_parameters": len(self._parameters),
                "total_namespaces": len(self._namespace_index),
                "total_snapshots": len(self._snapshots),
                "most_accessed": sorted(
                    self._access_count.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

    def get_access_log(self) -> Dict[str, int]:
        """Return a copy of parameter access counts."""

        with self._lock:
            return dict(self._access_count)
    
    @staticmethod
    def _extract_namespace(key: str) -> Optional[str]:
        """
        Extract namespace from parameter key.
        
        Parameters:
            key : str
                Parameter key
                
        Returns:
            Optional[str]: Namespace or None
            
        Examples:
            >>> ParameterStorage._extract_namespace("a.b.c")
            'a.b'
        """
        parts = key.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else None
    
    @staticmethod
    def _values_equal(val1: Any, val2: Any) -> bool:
        """
        Compare two parameter values for equality.
        
        Handles special cases like Tensors and NaN values.
        
        Parameters:
            val1 : Any
                First value
            val2 : Any
                Second value
                
        Returns:
            bool: True if values are equal
        """
        if val1 is val2:
            return True
        if val1 is None or val2 is None:
            return False
        if isinstance(val1, Tensor) and isinstance(val2, Tensor):
            return torch.allclose(val1, val2, rtol=1e-7, atol=1e-9, equal_nan=True)
        if isinstance(val1, float) and isinstance(val2, float):
            import math
            if math.isnan(val1) and math.isnan(val2):
                return True
        return bool(val1 == val2)


# Helper functions for namespace operations
def extract_namespace_hierarchy(key: str) -> List[str]:
    """
    Extract all namespace levels from a key.
    
    Parameters:
        key : str
            Parameter key
            
    Returns:
        List[str]: Namespace hierarchy
        
    Examples:
        >>> extract_namespace_hierarchy("a.b.c.d")
        ['a', 'a.b', 'a.b.c']
    """
    parts = key.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def merge_namespaces(params1: ParameterDict, 
                    params2: ParameterDict,
                    precedence: str = "second") -> ParameterDict:
    """
    Merge two parameter dictionaries with precedence.
    
    Parameters:
        params1 : ParameterDict
            First parameter dictionary
        params2 : ParameterDict
            Second parameter dictionary
        precedence : str
            Which dict takes precedence ("first" or "second")
            
    Returns:
        ParameterDict: Merged parameters
        
    Examples:
        >>> merged = merge_namespaces(
        ...     {"a": 1, "b": 2},
        ...     {"b": 3, "c": 4},
        ...     precedence="second"
        ... )
        >>> merged
        {'a': 1, 'b': 3, 'c': 4}
    """
    if precedence == "first":
        result = params2.copy()
        result.update(params1)
    else:
        result = params1.copy()
        result.update(params2)
    return result
