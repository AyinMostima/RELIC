"""
Operator Base Classes for RELIC Unified Computation Layer

This module defines the abstract base classes for all numerical operators used
across the RELIC framework. These abstractions enable configurable, composable,
and testable operator implementations with automatic differentiation support.

Design Philosophy:
    - Strategy Pattern: OperatorBase defines apply() interface, implementations provide strategies
    - Factory Creation: Operators instantiated via OperatorFactory based on config/name
    - Parameter Integration: Full integration with UnifiedParameterManager for unit-safe operations
    - PyTorch Native: All operators are nn.Module for automatic differentiation and device management

Mathematical Foundation:
    Operators are mathematical transformations T: X -> Y where:
    - X is input field/tensor space
    - Y is output field/tensor space
    - T preserves physical units and conservation properties

References:
    - Strang, G. (2007). Computational Science and Engineering. Wellesley-Cambridge Press
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM
    - Trefethen, L.N. & Bau, D. (1997). Numerical Linear Algebra. SIAM
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import logging

# Import parameter management system
from ...parameters import get_global_manager
from ...parameters.ops.unit_helpers import convert_units
PARAMETER_MANAGER_AVAILABLE = True

logger = logging.getLogger(__name__)

# Type aliases
Tensor = torch.Tensor
TensorOrTuple = Union[Tensor, Tuple[Tensor, ...]]
ConfigDict = Dict[str, Any]


@dataclass
class SpatialRequirements:
    """Spatial operator requirements for operator-mesh co-design.

    References:
        - Bauer et al. (2015), Geosci. Model Dev. - stencil and mesh interplay.

    Parameters:
        stencil_radius: maximum stencil extent in cells per direction (required)
        halo_depth: required halo depth for partitioned execution (required)
        locality: "compact" (nearest-neighbour stencil), "wide", or "global"
            (required)
        min_resolution: minimum grid spacing requirement [m]
        preferred_layout: memory layout hint, "aos" or "soa"
    """

    stencil_radius: int
    halo_depth: int
    locality: str
    min_resolution: Optional[float] = None
    preferred_layout: str = "soa"


@dataclass
class OperatorMetadata:
    """
    Metadata for operator documentation and validation.

    Attributes:
        name : str
            Human-readable operator name
        operator_type : str
            Type classification (gradient, divergence, laplacian, etc.)
        description : str
            Brief description of operator functionality
        mathematical_expression : str
            LaTeX or text mathematical formula
        input_units : str
            Expected input units (e.g., "kg m-3"; "[field_units]" for an
            operator that accepts a field in the caller's units); required
        output_units : str
            Output units (e.g., "[field_units] m-1" for a gradient); required
        spatial_dimensions : int
            Number of spatial dimensions supported (1, 2, 3, or 0 for any);
            required
        requires_grad : bool
            Whether operator requires gradient computation
        is_linear : bool
            Whether operator is linear
        is_conservative : bool
            Whether operator conserves quantities
        references : list
            Scientific references (DOI, papers, books)
        parameters : dict
            Required/optional parameters with descriptions
    """
    name: str
    operator_type: str
    description: str = ""
    mathematical_expression: str = ""
    input_units: str = field(kw_only=True)
    output_units: str = field(kw_only=True)
    spatial_dimensions: int = field(kw_only=True)  # 0 means any
    requires_grad: bool = False
    is_linear: bool = True
    is_conservative: bool = True
    references: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    spatial_requirements: Optional[SpatialRequirements] = None


def resolve_operator_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Resolve the computation dtype of an operator that casts its inputs.

    ``None`` selects the OperatorBase default ``torch.float32``. An integer or
    boolean dtype is rejected: the finite-difference kernels allocate their
    result like the input, so an integer computation dtype would truncate
    every difference quotient without any error.

    Raises
    ------
    TypeError
        If ``dtype`` is not a ``torch.dtype`` or is neither floating point
        nor complex.
    """
    resolved = torch.float32 if dtype is None else dtype
    if not isinstance(resolved, torch.dtype):
        raise TypeError("Operator dtype must be a torch.dtype")
    if not (resolved.is_floating_point or resolved.is_complex):
        raise TypeError("Operator dtype must be floating point or complex")
    return resolved


def reject_unexpected_runtime_parameters(
    kwargs: Dict[str, Any],
    allowed: Iterable[str],
) -> None:
    """Reject misspelled or unsupported runtime overrides of an apply() call.

    An unknown keyword is an error; it is never ignored in favour of the
    configured value, which would hide the misspelling.

    Raises
    ------
    TypeError
        If ``kwargs`` holds a name outside ``allowed``.
    """
    unexpected = sorted(set(kwargs) - set(allowed))
    if unexpected:
        names = ", ".join(unexpected)
        raise TypeError(f"unexpected runtime parameter(s): {names}")


class OperatorBase(ABC, nn.Module):
    """
    Abstract base class for all numerical operators in RELIC.

    This class defines the unified interface that all operators must implement,
    providing consistency, composability, and integration with the parameter
    management system.

    Design Principles:
        1. All operators inherit from both ABC and nn.Module
        2. apply() method is the core interface
        3. Configuration via constructor or ParameterManager
        4. Metadata for documentation and validation
        5. Device-agnostic operation

    Parameters
    ----------
    name : str
        Unique identifier for this operator instance
    config : ConfigDict
        Configuration dictionary with operator-specific parameters
    metadata : Optional[OperatorMetadata]
        Operator metadata for documentation; None means the subclass declares
        it in its own constructor (units and spatial dimensions are never
        declared for it by the base class)
    device : Optional[torch.device]
        Computation device (CPU/GPU)
    dtype : torch.dtype
        Default data type for computations

    Attributes
    ----------
    operator_type : str
        Class-level type identifier (override in subclasses)
    name : str
        Instance name
    config : ConfigDict
        Configuration dictionary
    metadata : OperatorMetadata
        Operator metadata

    Examples
    --------
    >>> class GradientOperator(OperatorBase):
    ...     operator_type = "gradient"
    ...     def apply(self, field, dx, **kwargs):
    ...         # Implementation
    ...         return gradient_field
    >>> op = GradientOperator(name="central_difference", config={})
    >>> result = op.apply(input_field, dx=0.1)
    """

    # Class-level type identifier (override in subclasses)
    operator_type: str = "generic"

    def __init__(
        self,
        name: str,
        config: Optional[ConfigDict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ) -> None:
        """Initialize operator base."""
        super().__init__()

        self.name = name
        self.config = config if config is not None else {}
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        # Store namespace and validation flag for parameter access
        self.namespace = namespace
        self.validate_parameters = validate_parameters

        # Metadata: a subclass that receives None declares its own (with its
        # units and spatial dimensions) after reading its parameters.
        self.metadata: Optional[OperatorMetadata] = metadata

        # No diagnostic counter: an operator call changes no operator state
        # (invariant I1).

        # Parameter manager: a bootstrap failure propagates; an operator is
        # never built without its parameter owner.
        self._param_manager = get_global_manager()

    @abstractmethod
    def apply(self, *args, **kwargs) -> TensorOrTuple:
        """
        Apply operator to input tensor(s).

        This is the core method that all operators must implement. It defines
        the transformation performed by the operator.

        Parameters
        ----------
        *args
            Positional arguments (typically input tensors)
        **kwargs
            Keyword arguments (operator-specific parameters)

        Returns
        -------
        TensorOrTuple
            Output tensor(s) after applying operator

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses

        Notes
        -----
        - Input/output units must match metadata specifications
        - Implementation should be differentiable for gradient-based optimization
        - Use self.get_parameter() to retrieve parameters with unit validation
        """
        pass

    def forward(self, *args, **kwargs) -> TensorOrTuple:
        """
        PyTorch nn.Module forward pass - delegates to apply().

        This method is called when using the operator as a callable.
        It updates diagnostics and delegates to the apply() method.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to apply()

        Returns
        -------
        TensorOrTuple
            Result from apply()
        """
        return self.apply(*args, **kwargs)

    def get_parameter(
        self,
        param_name: str,
        namespace: Optional[str] = None,
        validate: bool = True,
        description: Optional[str] = None
    ) -> Any:
        """
        Get parameter value from configuration or ParameterManager.

        Search order:
        1. self.config[param_name]
        2. ParameterManager.get(<namespace>.<param_name>, required=True)

        The YAML entry of <namespace>.<param_name> is the single source of
        truth for units, bounds and allowed_values; the key must exist. The
        value is returned in the YAML unit: a plain config value is taken in
        that unit, a config value given as {"value": v, "units": u} is
        converted from u to it (ParameterManager conversion contract), and
        with ``validate`` the result is checked against the YAML bounds and,
        for a choice parameter, the YAML allowed_values (no code copy of a
        choice list is consulted).

        Parameters
        ----------
        param_name : str
            Name of parameter
        namespace : Optional[str]
            Parameter namespace (e.g., "computation.operators.gradient")
        validate : bool
            Whether to check the value against the YAML bounds and
            allowed_values

        Returns
        -------
        Any
            Parameter value in the YAML unit

        Raises
        ------
        KeyError
            If the key is missing from the YAML parameter database
        ValueError
            If the operator has no namespace, config units cannot be
            converted to the YAML unit, the value lies outside the YAML
            bounds, or a choice value is not in the YAML allowed_values
        """
        def _coerce_scalar(candidate: Any) -> Optional[float]:
            if isinstance(candidate, torch.Tensor):
                if candidate.numel() != 1:
                    return None
                return float(candidate.detach().cpu().item())
            if isinstance(candidate, (int, float)):
                return float(candidate)
            return None

        resolved_namespace = namespace or self.namespace
        if not resolved_namespace:
            raise ValueError(
                f"Operator '{self.name}' has no parameter namespace for "
                f"'{param_name}'"
            )
        full_key = f"{resolved_namespace}.{param_name}"
        spec = self._param_manager.get_parameter(full_key)
        if spec is None:
            raise KeyError(f"Required parameter '{full_key}' not found")
        yaml_units = str(spec.units).strip()

        if param_name in self.config:
            value = self.config[param_name]
            if isinstance(value, dict) and "value" in value:
                config_units = value.get("units")
                value = value["value"]
                if config_units is not None and str(config_units).strip() != yaml_units:
                    # References:
                    #   CF Conventions 1.8 (units from UDUNITS-2); ParameterManager
                    #   unit contract, relic/parameters/ops/unit_validation.py
                    #   (convert_parameter_units_impl).
                    # Mathematical expression:
                    #   v_yaml = convert(v_config, u_config -> u_yaml)
                    # Parameter meanings:
                    #   v_config, u_config: value and units given in the config;
                    #   u_yaml: units of the YAML entry; v_yaml: returned value.
                    # Units:
                    #   v_config in u_config; v_yaml in u_yaml.
                    # Domain/constraints:
                    #   u_config must be convertible to u_yaml; otherwise
                    #   ValueError (units mismatch), never an unconverted value.
                    # Sign convention:
                    #   None; a pure change of units.
                    try:
                        value = convert_units(value, str(config_units).strip(), yaml_units)
                    except ValueError as exc:
                        raise ValueError(
                            f"Config parameter {param_name} units mismatch: cannot "
                            f"convert {config_units} to the YAML unit {yaml_units} "
                            f"of {full_key}"
                        ) from exc
        else:
            value = self._param_manager.get(
                full_key,
                expected_type=None,
                required=True,
            )

        if validate:
            bounds = spec.bounds
            numeric_value = _coerce_scalar(value)
            if bounds is not None and numeric_value is not None and not (
                float(bounds[0]) <= numeric_value <= float(bounds[1])
            ):
                raise ValueError(
                    f"Parameter {full_key}={numeric_value} outside bounds "
                    f"({float(bounds[0])}, {float(bounds[1])}) declared in the YAML database"
                )
            allowed_values = spec.allowed_values
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(
                    f"Parameter {full_key}={value!r} not in the allowed values "
                    f"({', '.join(str(item) for item in allowed_values)}) "
                    "declared in the YAML database"
                )
        return value

    def get_metadata(self) -> OperatorMetadata:
        """
        Get operator metadata.

        Returns
        -------
        OperatorMetadata
            Metadata object with operator documentation

        Raises
        ------
        TypeError
            If the operator declared no OperatorMetadata.
        """
        if self.metadata is None:
            raise TypeError(
                f"Operator '{self.name}' ({type(self).__name__}) declares no "
                "OperatorMetadata; its units and spatial dimensions must be "
                "declared by the operator"
            )
        return self.metadata

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get operator diagnostic information.

        Returns
        -------
        Dict[str, Any]
            Static operator description (name, type, device, dtype); an
            operator keeps no per-call counters (invariant I1)
        """
        return {
            'name': self.name,
            'operator_type': self.operator_type,
            'device': str(self.device),
            'dtype': str(self.dtype)
        }

    def to(self, *args: Any, **kwargs: Any) -> 'OperatorBase':
        """
        Move or cast the operator using the standard nn.Module.to contract.

        Parameters
        ----------
        *args, **kwargs
            Device, dtype, tensor, and transfer options accepted by
            ``torch.nn.Module.to``.

        Returns
        -------
        OperatorBase
            Self (for chaining)
        """
        current_dtype = (
            self.dtype if isinstance(self.dtype, torch.dtype) else torch.get_default_dtype()
        )
        probe = torch.empty(
            (0, 0, 0, 0),
            device=self.device,
            dtype=current_dtype,
        ).to(*args, **kwargs)
        super().to(*args, **kwargs)
        self.device = probe.device
        self.dtype = probe.dtype
        return self

    def __repr__(self) -> str:
        """String representation of operator."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"type='{self.operator_type}', "
            f"device='{self.device}'"
            f")"
        )


# Export main classes
__all__ = [
    'OperatorBase',
    'OperatorMetadata',
    'TensorOrTuple',
    'ConfigDict',
    'reject_unexpected_runtime_parameters',
    'resolve_operator_dtype',
]
