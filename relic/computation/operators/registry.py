"""
Operator Registry for RELIC Unified Computation Layer

This module provides a centralized registration system for all numerical operators
in the RELIC framework, enabling dynamic discovery, type-safe access, and
metadata-driven documentation.

Design Philosophy:
    - Registry Pattern: Central repository for operator classes
    - Decorator Registration: Simple @register_operator decorator for automatic registration
    - Type Safety: Validates that registered classes inherit from OperatorBase
    - Metadata Access: Provides query capabilities for operator discovery
    - Namespace Organization: Supports hierarchical categorization

Usage Example:
    >>> from relic.computation.operators.registry import register_operator, OperatorRegistry
    >>> from relic.computation.operators.base import OperatorBase
    >>>
    >>> @register_operator("gradient.central_difference_1d")
    >>> class CentralDifferenceGradient1D(OperatorBase):
    ...     operator_type = "gradient"
    ...     def apply(self, field, dx):
    ...         # Implementation
    ...         return gradient
    >>>
    >>> # Later retrieval
    >>> op_class = OperatorRegistry.get("gradient.central_difference_1d")
    >>> op = op_class(name="grad_x", config={})

References:
    - Gamma, E. et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley
"""

from typing import Dict, Type, List, Optional, Any
from dataclasses import dataclass
import logging

# Import base class for type checking
from .base import OperatorBase, OperatorMetadata
BASE_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class OperatorRegistryEntry:
    """
    Registry entry for an operator class.

    Attributes
    ----------
    name : str
        Fully qualified operator name (e.g., "gradient.central_difference_1d")
    operator_class : Type[OperatorBase]
        The operator class itself
    operator_type : str
        Type classification (gradient, divergence, laplacian, etc.)
    category : str
        High-level category (spatial, temporal, hybrid)
    metadata : Optional[OperatorMetadata]
        Operator metadata if available
    """
    name: str
    operator_class: Type['OperatorBase']  # type: ignore
    operator_type: str
    category: str = "general"
    metadata: Optional['OperatorMetadata'] = None  # type: ignore


class OperatorRegistry:
    """
    Central registry for all numerical operators in RELIC.

    This class maintains a global registry of operator classes, enabling:
    - Dynamic operator discovery
    - Type-safe operator instantiation
    - Metadata-driven documentation generation
    - Hierarchical organization by type and category

    The registry is a singleton pattern - all registration operations
    affect the global registry state.

    Class Methods
    -------------
    register(name, operator_class, category="general")
        Register an operator class
    get(name)
        Retrieve operator class by name
    list_operators(operator_type=None, category=None)
        List all registered operators with optional filtering
    get_metadata(name)
        Get metadata for a specific operator
    clear()
        Clear all registrations (primarily for testing)

    Examples
    --------
    >>> # Direct registration
    >>> OperatorRegistry.register("my_operator", MyOperatorClass)
    >>>
    >>> # Retrieve and instantiate
    >>> op_class = OperatorRegistry.get("my_operator")
    >>> op = op_class(name="instance1", config={})
    >>>
    >>> # Query all gradient operators
    >>> gradient_ops = OperatorRegistry.list_operators(operator_type="gradient")
    """

    # Class-level registry storage
    _registry: Dict[str, OperatorRegistryEntry] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        operator_class: Type['OperatorBase'],  # type: ignore
        category: str = "general",
        override: bool = False
    ) -> None:
        """
        Register an operator class in the global registry.

        Parameters
        ----------
        name : str
            Fully qualified operator name (e.g., "gradient.central_difference_1d")
        operator_class : Type[OperatorBase]
            The operator class to register (must inherit from OperatorBase)
        category : str, optional
            High-level category (spatial, temporal, hybrid, etc.)
        override : bool, optional
            If True, allows overwriting existing registrations

        Raises
        ------
        TypeError
            If operator_class does not inherit from OperatorBase
        ValueError
            If name is already registered and override=False

        Notes
        -----
        - Operator names should follow dotted namespace convention
        - The operator_type is extracted from the class's operator_type attribute
        """
        # Type validation
        if not issubclass(operator_class, OperatorBase):
            raise TypeError(
                f"Operator class {operator_class.__name__} must inherit from OperatorBase"
            )

        # Check for duplicate registration
        if name in cls._registry and not override:
            raise ValueError(
                f"Operator '{name}' is already registered. "
                f"Use override=True to replace existing registration."
            )

        # Extract operator type from class attribute
        operator_type = getattr(operator_class, 'operator_type', 'generic')

        # Record the metadata of one instance built from the YAML defaults.
        # A construction failure (missing YAML key, invalid default) is a
        # registration error; the operator is not registered without it.
        temp_instance = operator_class(name="__temp__", config={})
        metadata = temp_instance.get_metadata()

        # Create registry entry
        entry = OperatorRegistryEntry(
            name=name,
            operator_class=operator_class,
            operator_type=operator_type,
            category=category,
            metadata=metadata
        )

        cls._registry[name] = entry
        logger.info(f"Registered operator: {name} (type={operator_type}, category={category})")

    @classmethod
    def get(cls, name: str) -> Type['OperatorBase']:  # type: ignore
        """
        Retrieve operator class by name.

        Parameters
        ----------
        name : str
            Fully qualified operator name

        Returns
        -------
        Type[OperatorBase]
            The registered operator class

        Raises
        ------
        KeyError
            If operator name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Operator '{name}' not found in registry. "
                f"Available operators: {available}"
            )

        return cls._registry[name].operator_class

    @classmethod
    def list_operators(
        cls,
        operator_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[str]:
        """
        List all registered operators with optional filtering.

        Parameters
        ----------
        operator_type : Optional[str]
            Filter by operator type (gradient, divergence, etc.)
        category : Optional[str]
            Filter by category (spatial, temporal, etc.)

        Returns
        -------
        List[str]
            List of operator names matching the filters

        Examples
        --------
        >>> # List all operators
        >>> all_ops = OperatorRegistry.list_operators()
        >>>
        >>> # List only gradient operators
        >>> gradients = OperatorRegistry.list_operators(operator_type="gradient")
        >>>
        >>> # List spatial operators
        >>> spatial = OperatorRegistry.list_operators(category="spatial")
        """
        results = []

        for name, entry in cls._registry.items():
            # Apply filters
            if operator_type is not None and entry.operator_type != operator_type:
                continue
            if category is not None and entry.category != category:
                continue

            results.append(name)

        return sorted(results)

    @classmethod
    def get_metadata(cls, name: str) -> Optional['OperatorMetadata']:  # type: ignore
        """
        Get metadata for a specific operator.

        Parameters
        ----------
        name : str
            Operator name

        Returns
        -------
        Optional[OperatorMetadata]
            Operator metadata if available, None otherwise
        """
        if name not in cls._registry:
            raise KeyError(f"Operator '{name}' not found in registry")

        return cls._registry[name].metadata

    @classmethod
    def get_entry(cls, name: str) -> OperatorRegistryEntry:
        """
        Get full registry entry for an operator.

        Parameters
        ----------
        name : str
            Operator name

        Returns
        -------
        OperatorRegistryEntry
            Complete registry entry with class, type, and metadata
        """
        if name not in cls._registry:
            raise KeyError(f"Operator '{name}' not found in registry")

        return cls._registry[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an operator is registered.

        Parameters
        ----------
        name : str
            Operator name to check

        Returns
        -------
        bool
            True if operator is registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registrations.

        Warning
        -------
        This method is primarily intended for testing purposes.
        Use with caution in production code.
        """
        cls._registry.clear()
        logger.warning("Operator registry cleared")

    @classmethod
    def get_registry_info(cls) -> Dict[str, Any]:
        """
        Get summary information about the registry.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing registry statistics and summaries
        """
        type_counts = {}
        category_counts = {}

        for entry in cls._registry.values():
            # Count by type
            type_counts[entry.operator_type] = type_counts.get(entry.operator_type, 0) + 1
            # Count by category
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1

        return {
            'total_operators': len(cls._registry),
            'operator_names': sorted(cls._registry.keys()),
            'types': type_counts,
            'categories': category_counts
        }


def register_operator(
    name: str,
    category: str = "general",
    override: bool = False
):
    """
    Decorator for automatic operator registration.

    This decorator simplifies operator registration by automatically
    calling OperatorRegistry.register() when the class is defined.

    Parameters
    ----------
    name : str
        Fully qualified operator name
    category : str, optional
        High-level category classification
    override : bool, optional
        Allow overwriting existing registrations

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    >>> @register_operator("gradient.central_difference_1d", category="spatial")
    >>> class CentralDifferenceGradient1D(OperatorBase):
    ...     operator_type = "gradient"
    ...     def apply(self, field, dx):
    ...         return gradient

    Notes
    -----
    - This decorator should be used on all operator implementations
    - The decorated class is returned unmodified (decorator only has side effects)
    - Registration occurs at import time, not instantiation time
    """
    def decorator(operator_class: Type['OperatorBase']) -> Type['OperatorBase']:  # type: ignore
        # Register the class
        OperatorRegistry.register(
            name=name,
            operator_class=operator_class,
            category=category,
            override=override
        )

        # Return unmodified class
        return operator_class

    return decorator


# Export main classes and functions
__all__ = [
    'OperatorRegistry',
    'OperatorRegistryEntry',
    'register_operator'
]
