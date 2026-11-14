"""
Solver Factory for RELIC Unified Computation Layer

This module provides factory methods for creating numerical solvers with
parameter validation, operator injection, configuration management, and
integration with the unified parameter system.

Design Philosophy:
    - Abstract Factory Pattern: Unified interface for solver instantiation
    - Configuration-Driven: Support both direct config dicts and Hydra/Pydantic models
    - Operator Injection: Automatic operator assignment for operator-based solvers
    - Parameter Integration: Automatic ParameterManager injection
    - Type Safety: Validates solver types and parameters before instantiation

Usage Example:
    >>> from relic.computation.solvers.factory import SolverFactory
    >>>
    >>> # Create solver from name with config
    >>> solver = SolverFactory.create(
    ...     solver_name="backward_euler",
    ...     name="be_solver",
    ...     config={"max_iterations": 100, "rtol": 1e-6}
    ... )
    >>>
    >>> # Create with operator injection
    >>> from relic.computation.operators.factory import OperatorFactory
    >>> laplacian = OperatorFactory.create("laplacian.finite_difference_1d", "laplacian")
    >>> solver = SolverFactory.create(
    ...     solver_name="crank_nicolson",
    ...     name="cn_solver",
    ...     operator=laplacian
    ... )
    >>>
    >>> # Batch creation
    >>> solvers = SolverFactory.create_batch([
    ...     {"solver_name": "thomas_algorithm", "name": "thomas"},
    ...     {"solver_name": "backward_euler", "name": "be"}
    ... ])

References:
    - Gamma, E. et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley
"""

from collections.abc import Mapping
from typing import Dict, Any, Optional, List, Union
import logging
import torch

# Import registry
from .registry import SolverRegistry

# Import base classes
from .base import SolverBase, SolverMetadata, SolverFamily, SOLVER_BASE_CONFIG_KEYS
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


class SolverConfig(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore
    """
    Pydantic model for solver configuration validation.

    This model ensures type safety and validates solver configurations
    before instantiation.

    Attributes
    ----------
    solver_name : str
        Solver identifier from registry
    name : str
        Instance name for this solver
    config : Dict[str, Any]
        Solver-specific parameters (max_iterations, tolerances, etc.)
    operator_name : Optional[str]
        Name of associated operator (for operator-based solvers)
    device : str
        Computation device ("cpu", "cuda", "cuda:0", etc.)
    dtype : str
        Data type ("float32", "float64", etc.)
    namespace : Optional[str]
        Parameter namespace for ParameterManager lookup
    """
    if PYDANTIC_AVAILABLE:
        solver_name: str = Field(..., description="Solver type from registry")
        name: str = Field(..., description="Instance name")
        config: Dict[str, Any] = Field(default_factory=dict, description="Solver parameters")
        operator_name: Optional[str] = Field(default=None, description="Associated operator")
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


class SolverFactory:
    """
    Factory for creating numerical solvers with validation and operator injection.

    This class provides multiple creation methods:
    - create(): Direct creation with name and config
    - create_from_config(): Create from Pydantic/Hydra config
    - create_batch(): Create multiple solvers at once
    - create_with_operator(): Create with explicit operator injection

    All creation methods integrate with ParameterManager for parameter lookup
    and validation.

    Examples
    --------
    >>> # Basic creation
    >>> solver = SolverFactory.create(
    ...     solver_name="backward_euler",
    ...     name="be_solver",
    ...     config={"max_iterations": 100}
    ... )
    >>>
    >>> # With operator injection
    >>> laplacian = OperatorFactory.create("laplacian.finite_difference_1d", "laplacian")
    >>> solver = SolverFactory.create(
    ...     solver_name="crank_nicolson",
    ...     name="cn_solver",
    ...     operator=laplacian
    ... )
    >>>
    >>> # From validated config
    >>> config = SolverConfig(
    ...     solver_name="thomas_algorithm",
    ...     name="thomas",
    ...     config={"rtol": 1e-8}
    ... )
    >>> solver = SolverFactory.create_from_config(config)
    """

    @staticmethod
    def create(
        solver_name: str,
        name: str,
        operator: Optional[Any] = None,
        config: Optional[ConfigDict] = None,
        metadata: Optional['SolverMetadata'] = None,  # type: ignore
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ) -> 'SolverBase':  # type: ignore
        """
        Create a solver instance from registry.

        Parameters
        ----------
        solver_name : str
            Solver type name from registry (e.g., "backward_euler")
        name : str
            Instance name for this solver
        operator : Optional[Any]
            Associated operator for operator-based solvers
        config : Optional[ConfigDict]
            Solver-specific configuration parameters
        metadata : Optional[SolverMetadata]
            Custom metadata (if None, uses class default)
        device : Optional[torch.device]
            Computation device (CPU/GPU)
        dtype : torch.dtype
            Default data type for computations
        namespace : Optional[str]
            Explicit parameter namespace from which max_iterations, rtol and
            atol are read as required parameters when they are not in
            config. When None, the solver resolves them from its own
            namespace (SolverBase.__init__).
        validate_parameters : bool
            Whether to read the explicit namespace (ignored when namespace is
            None)

        Returns
        -------
        SolverBase
            Instantiated solver

        Raises
        ------
        KeyError
            If solver_name is not registered, or a required parameter is
            missing from the explicit namespace
        ValueError
            If parameter validation fails
        TypeError
            If config contains invalid types

        Examples
        --------
        >>> solver = SolverFactory.create(
        ...     solver_name="backward_euler",
        ...     name="be_solver",
        ...     config={"max_iterations": 100, "rtol": 1e-6},
        ...     device=torch.device("cuda"),
        ...     dtype=torch.float64
        ... )
        """
        # Retrieve solver class from registry
        try:
            solver_class = SolverRegistry.get(solver_name)
        except KeyError:
            logger.error(f"Solver '{solver_name}' not found in registry")
            raise

        # Prepare config
        if config is None:
            config = {}

        # Parameter validation via ParameterManager
        if validate_parameters and PARAMETER_MANAGER_AVAILABLE:
            config = SolverFactory._validate_and_enrich_config(
                solver_name=solver_name,
                config=config,
                namespace=namespace
            )

        # Set device
        if device is None:
            device = torch.device("cpu")

        # Instantiate solver
        try:
            solver = solver_class(
                name=name,
                operator=operator,
                config=config,
                metadata=metadata,
                device=device,
                dtype=dtype
            )

            logger.info(
                f"Created solver '{name}' of type '{solver_name}' "
                f"on device '{device}' with dtype '{dtype}'"
            )

            return solver

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(
                f"Failed to instantiate solver '{solver_name}': {e}",
                exc_info=True
            )
            raise

    @staticmethod
    def create_from_config(
        config: Union['SolverConfig', ConfigDict],  # type: ignore
        operator: Optional[Any] = None
    ) -> 'SolverBase':  # type: ignore
        """
        Create solver from configuration object.

        Parameters
        ----------
        config : Union[SolverConfig, ConfigDict]
            Configuration (Pydantic model or dict)
        operator : Optional[Any]
            Override operator (if None, uses operator_name from config)

        Returns
        -------
        SolverBase
            Instantiated solver

        Examples
        --------
        >>> # From Pydantic model
        >>> config = SolverConfig(
        ...     solver_name="backward_euler",
        ...     name="be_solver",
        ...     config={"max_iterations": 100}
        ... )
        >>> solver = SolverFactory.create_from_config(config)
        >>>
        >>> # From dict (auto-validated if Pydantic available)
        >>> config_dict = {
        ...     "solver_name": "thomas_algorithm",
        ...     "name": "thomas",
        ...     "device": "cuda",
        ...     "dtype": "float64"
        ... }
        >>> solver = SolverFactory.create_from_config(config_dict)
        """
        # Every mapping is validated through SolverConfig; a validation error
        # (pydantic.ValidationError, a ValueError) propagates. There is no
        # unvalidated raw-dict path.
        if isinstance(config, Mapping):
            config = SolverConfig(**dict(config))
        if not isinstance(config, SolverConfig):
            raise TypeError(
                "config must be a SolverConfig or a mapping, got "
                f"{type(config).__name__}"
            )

        solver_name = config.solver_name
        name = config.name
        solver_config = config.config
        operator_name = config.operator_name
        device = torch.device(config.device)
        dtype = getattr(torch, config.dtype)
        namespace = config.namespace

        # Create the requested operator when none was passed in. A failure to
        # build it (for example an unregistered operator_name) propagates; the
        # solver is never created silently without the operator it asked for.
        if operator is None and operator_name is not None:
            from ..operators.factory import OperatorFactory
            operator = OperatorFactory.create(
                operator_name=operator_name,
                name=f"{name}_operator",
                device=device,
                dtype=dtype
            )
            logger.info(f"Created operator '{operator_name}' for solver '{name}'")

        return SolverFactory.create(
            solver_name=solver_name,
            name=name,
            operator=operator,
            config=solver_config,
            device=device,
            dtype=dtype,
            namespace=namespace
        )

    @staticmethod
    def create_batch(
        configs: List[Union['SolverConfig', ConfigDict]]  # type: ignore
    ) -> List['SolverBase']:  # type: ignore
        """
        Create multiple solvers from a list of configurations.

        Parameters
        ----------
        configs : List[Union[SolverConfig, ConfigDict]]
            List of configurations

        Returns
        -------
        List[SolverBase]
            List of instantiated solvers

        Examples
        --------
        >>> configs = [
        ...     {"solver_name": "backward_euler", "name": "be1"},
        ...     {"solver_name": "crank_nicolson", "name": "cn1"},
        ...     {"solver_name": "thomas_algorithm", "name": "thomas1"}
        ... ]
        >>> solvers = SolverFactory.create_batch(configs)
        """
        solvers = []

        for i, config in enumerate(configs):
            try:
                solver = SolverFactory.create_from_config(config)
                solvers.append(solver)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                logger.error(f"Failed to create solver {i} from batch: {e}")
                raise

        logger.info(f"Created batch of {len(solvers)} solvers")
        return solvers

    @staticmethod
    def create_with_operator(
        solver_name: str,
        name: str,
        operator: Any,
        config: Optional[ConfigDict] = None,
        **kwargs
    ) -> 'SolverBase':  # type: ignore
        """
        Create solver with explicit operator injection.

        This is a convenience method for operator-based solvers that
        simplifies the API when the operator is already instantiated.

        Parameters
        ----------
        solver_name : str
            Solver type from registry
        name : str
            Instance name
        operator : Any
            Operator instance (e.g., LaplacianOperator)
        config : Optional[ConfigDict]
            Solver configuration
        **kwargs
            Additional arguments passed to create()

        Returns
        -------
        SolverBase
            Instantiated solver with operator

        Examples
        --------
        >>> from relic.computation.operators.factory import OperatorFactory
        >>> laplacian = OperatorFactory.create("laplacian.finite_difference_1d", "laplacian")
        >>> solver = SolverFactory.create_with_operator(
        ...     solver_name="crank_nicolson",
        ...     name="cn_solver",
        ...     operator=laplacian,
        ...     config={"max_iterations": 50}
        ... )
        """
        return SolverFactory.create(
            solver_name=solver_name,
            name=name,
            operator=operator,
            config=config,
            **kwargs
        )

    @staticmethod
    def _validate_and_enrich_config(
        solver_name: str,
        config: ConfigDict,
        namespace: Optional[str] = None
    ) -> ConfigDict:
        """
        Read the SolverBase keys from an explicit parameter namespace.

        Parameters
        ----------
        solver_name : str
            Registered solver name (used for diagnostics only)
        config : ConfigDict
            Original configuration (not modified)
        namespace : Optional[str]
            Explicit namespace. When None the configuration is returned
            unchanged: the solver resolves max_iterations, rtol and atol from
            its own namespace in SolverBase.__init__, which is the single
            owner of that resolution. (Deriving the namespace from the
            registry name is wrong in general: thomas_algorithm uses
            computation.solvers.thomas.)

        Returns
        -------
        ConfigDict
            Copy of config with every SolverBase key that config does not
            set read from ``<namespace>.<key>``

        Raises
        ------
        KeyError
            If a required parameter is missing from the explicit namespace.
            Parameter manager and validation errors propagate unchanged.
        """
        enriched_config = dict(config)
        if namespace is None:
            return enriched_config

        param_manager = get_global_manager()
        for key in SOLVER_BASE_CONFIG_KEYS:
            if key not in enriched_config:
                enriched_config[key] = param_manager.get(
                    f"{namespace}.{key}",
                    required=True,
                )

        logger.debug(
            f"Validated config for solver '{solver_name}' "
            f"in namespace '{namespace}'"
        )

        return enriched_config

    @staticmethod
    def list_available_solvers(
        family: Optional['SolverFamily'] = None,  # type: ignore
        implicit_only: bool = False
    ) -> List[str]:
        """
        List all available solver types from registry.

        Parameters
        ----------
        family : Optional[SolverFamily]
            Filter by solver family
        implicit_only : bool
            Filter for implicit solvers only

        Returns
        -------
        List[str]
            List of available solver names

        Examples
        --------
        >>> # All solvers
        >>> all_solvers = SolverFactory.list_available_solvers()
        >>>
        >>> # Only implicit ODE solvers
        >>> implicit_ode = SolverFactory.list_available_solvers(
        ...     family=SolverFamily.ODE_IMPLICIT,
        ...     implicit_only=True
        ... )
        """
        return SolverRegistry.list_solvers(
            family=family,
            implicit_only=implicit_only
        )

    @staticmethod
    def get_solver_info(solver_name: str) -> Dict[str, Any]:
        """
        Get information about a registered solver.

        Parameters
        ----------
        solver_name : str
            Solver name

        Returns
        -------
        Dict[str, Any]
            Dictionary with solver class, family, metadata

        Examples
        --------
        >>> info = SolverFactory.get_solver_info("backward_euler")
        >>> print(info['solver_family'])
        SolverFamily.ODE_IMPLICIT
        >>> print(info['is_implicit'])
        True
        """
        entry = SolverRegistry.get_entry(solver_name)

        return {
            'name': entry.name,
            'solver_class': entry.solver_class.__name__,
            'solver_family': entry.solver_family,
            'is_implicit': entry.is_implicit,
            'requires_jacobian': entry.requires_jacobian,
            'metadata': entry.metadata
        }

    @staticmethod
    def get_solvers_for_problem(
        problem_type: str,
        implicit: Optional[bool] = None
    ) -> List[str]:
        """
        Get suitable solvers for a specific problem type.

        Parameters
        ----------
        problem_type : str
            Problem type: "linear", "ode", "nonlinear", "optimization"
        implicit : Optional[bool]
            Filter by implicit/explicit

        Returns
        -------
        List[str]
            List of suitable solver names

        Examples
        --------
        >>> # Get all ODE solvers
        >>> ode_solvers = SolverFactory.get_solvers_for_problem("ode")
        >>>
        >>> # Get implicit ODE solvers only
        >>> implicit_ode = SolverFactory.get_solvers_for_problem("ode", implicit=True)
        """
        return SolverRegistry.get_solvers_for_problem(
            problem_type=problem_type,
            implicit=implicit
        )


# Export main classes
__all__ = [
    'SolverFactory',
    'SolverConfig'
]
