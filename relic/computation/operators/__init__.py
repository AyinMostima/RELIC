"""
Operator System for RELIC Unified Computation Layer

This package provides the complete operator system including:
- OperatorBase: Abstract base class for all operators
- OperatorRegistry: Central registration system
- OperatorFactory: Factory methods for creating operators
- Operator implementations (gradients, divergence, laplacian, etc.)

Usage Example:
    >>> from relic.computation.operators import OperatorFactory, register_operator
    >>>
    >>> # List available operators
    >>> operators = OperatorFactory.list_available_operators()
    >>>
    >>> # Create an operator
    >>> gradient = OperatorFactory.create(
    ...     operator_name="gradient.central_difference_1d",
    ...     name="grad_x",
    ...     config={"order": 2}
    ... )
    >>>
    >>> # Register a new operator
    >>> @register_operator("my_operator", category="spatial")
    >>> class MyOperator(OperatorBase):
    ...     operator_type = "custom"
    ...     def apply(self, field):
    ...         return transformed_field
"""

# Import base classes
from .base import (
    OperatorBase,
    OperatorMetadata,
    SpatialRequirements,
    TensorOrTuple,
    ConfigDict
)

# Import registry
from .registry import (
    OperatorRegistry,
    OperatorRegistryEntry,
    register_operator
)

# Import factory
from .factory import (
    OperatorFactory,
    OperatorConfig
)

# Import operator implementations (auto-registers via decorators)
# Importing these modules triggers @register_operator decorators
from . import gradient
from . import laplacian
from . import divergence
from . import transport
from . import diagnostics
from . import iphs
from .iphs_certificate import (
    IPHS_LEVEL_DEFINITIONS,
    IPHSEquivalenceCertificate,
    load_iphs_equivalence_certificate,
)
from .stochastic_sdae import (
    ConservativeNoiseFluxSpec,
    DescriptorConstraintSpec,
    JumpResetSpec,
    NoiseCovarianceSpec,
    StabilizedAdvectionSpec,
    StochasticTermSpec,
)

# Function-based transport operators (single source of truth in computation layer)
from .functional import (
    apply_diffusion,
    apply_advection,
    advance_advection_diffusion,
    check_conservation,
    apply_non_negative_constraint,
    harmonic_mean,
    compute_conservative_divergence_2d,
    compute_conservative_divergence_3d,
)

# Version info
__version__ = "0.1.0"

# Public API
__all__ = [
    # Base classes
    'OperatorBase',
    'OperatorMetadata',
    'SpatialRequirements',
    'TensorOrTuple',
    'ConfigDict',

    # Registry
    'OperatorRegistry',
    'OperatorRegistryEntry',
    'register_operator',

    # Factory
    'OperatorFactory',
    'OperatorConfig',

    # Function-based transport operators
    'apply_diffusion',
    'apply_advection',
    'advance_advection_diffusion',
    'check_conservation',
    'apply_non_negative_constraint',
    'harmonic_mean',
    'compute_conservative_divergence_2d',
    'compute_conservative_divergence_3d',
    'IPHS_LEVEL_DEFINITIONS',
    'IPHSEquivalenceCertificate',
    'load_iphs_equivalence_certificate',
    'ConservativeNoiseFluxSpec',
    'DescriptorConstraintSpec',
    'JumpResetSpec',
    'NoiseCovarianceSpec',
    'StabilizedAdvectionSpec',
    'StochasticTermSpec',

    # Operator implementation modules (for direct class access if needed)
    'gradient',
    'laplacian',
    'divergence',
    'transport',
    'diagnostics',
    'iphs'
]
