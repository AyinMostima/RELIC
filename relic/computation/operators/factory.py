"""
Operator Factory for RELIC Unified Computation Layer

This module provides factory methods for creating numerical operators with
parameter validation, configuration management, and integration with the
unified parameter system.

Design Philosophy:
    - Abstract Factory Pattern: Unified interface for operator instantiation
    - Configuration-Driven: Support both direct config dicts and Hydra/Pydantic models
    - Parameter Integration: Automatic ParameterManager injection
    - Type Safety: Validates operator types and parameters before instantiation
    - Device Management: Automatic device allocation and dtype handling

Usage Example:
    >>> from relic.computation.operators.factory import OperatorFactory
    >>>
    >>> # Create operator from name with config
    >>> op = OperatorFactory.create(
    ...     operator_name="gradient.central_difference_1d",
    ...     name="grad_x",
    ...     config={"order": 2, "dx": 0.1}
    ... )
    >>>
    >>> # Create with Hydra config
    >>> op = OperatorFactory.create_from_config(hydra_config)
    >>>
    >>> # Batch creation
    >>> operators = OperatorFactory.create_batch([
    ...     {"operator_name": "gradient.central_difference_1d", "name": "grad_x"},
    ...     {"operator_name": "divergence.finite_volume_2d", "name": "div_flux"}
    ... ])

References:
    - Gamma, E. et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley
"""

from collections.abc import Mapping
from typing import Dict, Any, Optional, List, Union
import copy
import logging
import torch

# Import registry
from .registry import OperatorRegistry

# Import base classes
from .base import OperatorBase, OperatorMetadata
BASE_AVAILABLE = True

# Import parameter management
from ...parameters import get_global_manager
PARAMETER_MANAGER_AVAILABLE = True

# Import configuration validation (Pydantic)
from pydantic import BaseModel, Field, field_validator
PYDANTIC_AVAILABLE = True

logger = logging.getLogger(__name__)

# Type aliases
ConfigDict = Dict[str, Any]


class OperatorConfig(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore
    """
    Pydantic model for operator configuration validation.

    This model ensures type safety and validates operator configurations
    before instantiation.

    Attributes
    ----------
    operator_name : str
        Fully qualified operator name from registry
    name : str
        Instance name for this operator
    config : Dict[str, Any]
        Operator-specific parameters
    device : str
        Computation device ("cpu", "cuda", "cuda:0", etc.)
    dtype : str
        Data type ("float32", "float64", etc.)
    namespace : Optional[str]
        Parameter namespace for ParameterManager lookup
    """
    if PYDANTIC_AVAILABLE:
        operator_name: str = Field(..., description="Operator type from registry")
        name: str = Field(..., description="Instance name")
        config: Dict[str, Any] = Field(default_factory=dict, description="Operator parameters")
        device: str = Field(default="cpu", description="Computation device")
        dtype: str = Field(default="float32", description="Data type")
        namespace: Optional[str] = Field(default=None, description="Parameter namespace")

        @field_validator('device')
        def validate_device(cls, v: str):
            """Validate device string."""
            if not v.startswith('cpu') and not v.startswith('cuda'):
                raise ValueError(f"Invalid device: {v}. Must be 'cpu' or 'cuda[:N]'")
            return v

        @field_validator('dtype')
        def validate_dtype(cls, v: str):
            """Validate dtype string."""
            valid_dtypes = ['float16', 'float32', 'float64', 'int32', 'int64']
            if v not in valid_dtypes:
                raise ValueError(f"Invalid dtype: {v}. Must be one of {valid_dtypes}")
            return v


class OperatorFactory:
    """
    Factory for creating numerical operators with validation and parameter injection.

    This class provides multiple creation methods:
    - create(): Direct creation with name and config
    - create_from_config(): Create from Pydantic/Hydra config
    - create_batch(): Create multiple operators at once
    - create_with_validation(): Create with explicit parameter validation

    All creation methods integrate with ParameterManager for parameter lookup
    and validation.

    Examples
    --------
    >>> # Basic creation
    >>> op = OperatorFactory.create(
    ...     operator_name="gradient.central_difference_1d",
    ...     name="grad_x",
    ...     config={"order": 2}
    ... )
    >>>
    >>> # With device specification
    >>> op = OperatorFactory.create(
    ...     operator_name="laplacian.finite_difference_1d",
    ...     name="laplacian",
    ...     device=torch.device("cuda:0"),
    ...     dtype=torch.float64
    ... )
    >>>
    >>> # From validated config
    >>> config = OperatorConfig(
    ...     operator_name="divergence.finite_volume_2d",
    ...     name="div_flux",
    ...     config={"boundary": "periodic"}
    ... )
    >>> op = OperatorFactory.create_from_config(config)
    """

    @staticmethod
    def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
        """
        Deep-merge two config dictionaries where override takes precedence.
        """
        merged: ConfigDict = {}
        for key, value in base.items():
            merged[key] = value
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = OperatorFactory._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def create(
        operator_name: str,
        name: str,
        config: Optional[ConfigDict] = None,
        metadata: Optional['OperatorMetadata'] = None,  # type: ignore
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ) -> 'OperatorBase':  # type: ignore
        """
        Create an operator instance from registry.

        Parameters
        ----------
        operator_name : str
            Operator type name from registry (e.g., "gradient.central_difference_1d")
        name : str
            Instance name for this operator
        config : Optional[ConfigDict]
            Operator-specific configuration parameters
        metadata : Optional[OperatorMetadata]
            Custom metadata (if None, uses class default)
        device : Optional[torch.device]
            Computation device (CPU/GPU)
        dtype : torch.dtype
            Default data type for computations
        namespace : Optional[str]
            Explicit parameter namespace. With validate_parameters, its
            values are merged under config (config wins) and lookup errors
            propagate; it is also passed to the operator. When None, the
            operator resolves its parameters from its own namespace.
        validate_parameters : bool
            Whether to merge the explicit namespace (no effect when namespace
            is None) and whether the operator validates parameter lookups

        Returns
        -------
        OperatorBase
            Instantiated operator

        Raises
        ------
        KeyError
            If operator_name is not registered
        ValueError
            If parameter validation fails
        TypeError
            If config contains invalid types

        Examples
        --------
        >>> op = OperatorFactory.create(
        ...     operator_name="gradient.central_difference_1d",
        ...     name="grad_x",
        ...     config={"order": 2, "dx": 0.1},
        ...     device=torch.device("cuda"),
        ...     dtype=torch.float64
        ... )
        """
        # Retrieve operator class from registry
        try:
            operator_class = OperatorRegistry.get(operator_name)
        except KeyError:
            logger.error(f"Operator '{operator_name}' not found in registry")
            raise

        # Prepare config
        if config is None:
            config = {}

        # Parameter validation via ParameterManager
        if validate_parameters and PARAMETER_MANAGER_AVAILABLE:
            config = OperatorFactory._validate_and_enrich_config(
                operator_name=operator_name,
                config=config,
                namespace=namespace
            )

        # Set device
        if device is None:
            device = torch.device("cpu")

        # Instantiate operator
        try:
            operator = operator_class(
                name=name,
                config=config,
                metadata=metadata,
                device=device,
                dtype=dtype,
                namespace=namespace,
                validate_parameters=validate_parameters
            )

            logger.info(
                f"Created operator '{name}' of type '{operator_name}' "
                f"on device '{device}' with dtype '{dtype}'"
            )

            return operator

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(
                f"Failed to instantiate operator '{operator_name}': {e}",
                exc_info=True
            )
            raise

    @staticmethod
    def create_from_config(
        config: Union['OperatorConfig', ConfigDict]  # type: ignore
    ) -> 'OperatorBase':  # type: ignore
        """
        Create operator from configuration object.

        Parameters
        ----------
        config : Union[OperatorConfig, ConfigDict]
            Configuration (Pydantic model or dict)

        Returns
        -------
        OperatorBase
            Instantiated operator

        Examples
        --------
        >>> # From Pydantic model
        >>> config = OperatorConfig(
        ...     operator_name="gradient.central_difference_1d",
        ...     name="grad_x",
        ...     config={"order": 2}
        ... )
        >>> op = OperatorFactory.create_from_config(config)
        >>>
        >>> # From dict (auto-validated if Pydantic available)
        >>> config_dict = {
        ...     "operator_name": "divergence.finite_volume_2d",
        ...     "name": "div_flux",
        ...     "device": "cuda",
        ...     "dtype": "float64"
        ... }
        >>> op = OperatorFactory.create_from_config(config_dict)
        """
        # Every mapping is validated through OperatorConfig; a validation error
        # (pydantic.ValidationError, a ValueError) propagates. There is no
        # unvalidated raw-dict path.
        if isinstance(config, Mapping):
            config = OperatorConfig(**dict(config))
        if not isinstance(config, OperatorConfig):
            raise TypeError(
                "config must be an OperatorConfig or a mapping, got "
                f"{type(config).__name__}"
            )

        operator_name = config.operator_name
        name = config.name
        op_config = config.config
        device = torch.device(config.device)
        dtype = getattr(torch, config.dtype)
        namespace = config.namespace

        return OperatorFactory.create(
            operator_name=operator_name,
            name=name,
            config=op_config,
            device=device,
            dtype=dtype,
            namespace=namespace
        )

    @staticmethod
    def create_batch(
        configs: List[Union['OperatorConfig', ConfigDict]]  # type: ignore
    ) -> List['OperatorBase']:  # type: ignore
        """
        Create multiple operators from a list of configurations.

        Parameters
        ----------
        configs : List[Union[OperatorConfig, ConfigDict]]
            List of configurations

        Returns
        -------
        List[OperatorBase]
            List of instantiated operators

        Examples
        --------
        >>> configs = [
        ...     {"operator_name": "gradient.central_difference_1d", "name": "grad_x"},
        ...     {"operator_name": "gradient.central_difference_1d", "name": "grad_y"},
        ...     {"operator_name": "divergence.finite_volume_2d", "name": "div_flux"}
        ... ]
        >>> operators = OperatorFactory.create_batch(configs)
        """
        operators = []

        for i, config in enumerate(configs):
            try:
                op = OperatorFactory.create_from_config(config)
                operators.append(op)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                logger.error(f"Failed to create operator {i} from batch: {e}")
                raise

        logger.info(f"Created batch of {len(operators)} operators")
        return operators

    @staticmethod
    def _validate_and_enrich_config(
        operator_name: str,
        config: ConfigDict,
        namespace: Optional[str] = None
    ) -> ConfigDict:
        """
        Merge the parameters of an explicit namespace under the configuration.

        Parameters
        ----------
        operator_name : str
            Registered operator name (used for diagnostics only)
        config : ConfigDict
            Original configuration (not modified; its keys take precedence)
        namespace : Optional[str]
            Explicit namespace. When None the configuration is returned
            unchanged: each operator resolves its parameters from its own
            namespace (for example computation.operators.gradient) through
            OperatorBase.get_parameter. (A namespace derived from the registry
            name, such as computation.operators.gradient_central_difference_1d,
            matches no YAML namespace.)

        Returns
        -------
        ConfigDict
            Copy of config merged over the namespace values: first the cached
            namespace dictionary, then the parameter database values

        Raises
        ------
        Exception
            Parameter manager, lookup and validation errors propagate
            unchanged; nothing is logged and skipped.
        """
        enriched_config = dict(config)
        if namespace is None:
            return enriched_config

        manager = get_global_manager()

        # 1) Cached namespace config (from Hydra ingest). Deep-copied so the
        #    nested writes below never mutate the manager's cached dictionary.
        namespace_defaults: ConfigDict = copy.deepcopy(manager.get_namespace_dict(namespace))

        # 2) Parameter database values under the namespace as nested defaults.
        namespace_parameters = manager.get_namespace_parameters(
            namespace=namespace,
            include_subnamespaces=True,
            return_specs=False,
        )
        prefix = f"{namespace}."
        for full_key, value in namespace_parameters.items():
            short_key = full_key[len(prefix):] if full_key.startswith(prefix) else full_key
            parts = [part for part in short_key.split(".") if part]
            if not parts:
                continue
            cursor = namespace_defaults
            for part in parts[:-1]:
                existing = cursor.get(part)
                if not isinstance(existing, dict):
                    cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = value

        enriched_config = OperatorFactory._deep_merge(namespace_defaults, enriched_config)

        logger.debug(
            f"Validated config for operator '{operator_name}' "
            f"in namespace '{namespace}'"
        )

        return enriched_config

    @staticmethod
    def list_available_operators(
        operator_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[str]:
        """
        List all available operator types from registry.

        Parameters
        ----------
        operator_type : Optional[str]
            Filter by operator type
        category : Optional[str]
            Filter by category

        Returns
        -------
        List[str]
            List of available operator names

        Examples
        --------
        >>> # All operators
        >>> all_ops = OperatorFactory.list_available_operators()
        >>>
        >>> # Only gradient operators
        >>> gradients = OperatorFactory.list_available_operators(operator_type="gradient")
        """
        return OperatorRegistry.list_operators(
            operator_type=operator_type,
            category=category
        )

    @staticmethod
    def get_operator_info(operator_name: str) -> Dict[str, Any]:
        """
        Get information about a registered operator.

        Parameters
        ----------
        operator_name : str
            Operator name

        Returns
        -------
        Dict[str, Any]
            Dictionary with operator class, type, metadata

        Examples
        --------
        >>> info = OperatorFactory.get_operator_info("gradient.central_difference_1d")
        >>> print(info['operator_type'])
        'gradient'
        >>> print(info['metadata'].mathematical_expression)
        'df/dx ~= (f(x+h) - f(x-h)) / (2h)'
        """
        entry = OperatorRegistry.get_entry(operator_name)

        return {
            'name': entry.name,
            'operator_class': entry.operator_class.__name__,
            'operator_type': entry.operator_type,
            'category': entry.category,
            'metadata': entry.metadata
        }

    def get_operator(
        self,
        operator_name: str,
        *,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Convenience operator retrieval for registered operators.

        The keywords are explicit: an unknown or misspelled keyword raises
        TypeError instead of being dropped.

        Parameters
        ----------
        operator_name : str
            Registered operator name.
        name : Optional[str]
            Instance name; defaults to operator_name.
        config : Optional[Dict[str, Any]]
            Operator configuration.
        device : Optional[torch.device]
            Target device.
        dtype : Optional[torch.dtype]
            Desired dtype.

        Returns
        -------
        OperatorBase
            Instantiated operator from the registry.
        """
        return OperatorFactory.create(
            operator_name=operator_name,
            name=operator_name if name is None else name,
            config={} if config is None else config,
            device=device,
            dtype=dtype,
        )


# Export main classes
__all__ = [
    'OperatorFactory',
    'OperatorConfig'
]
