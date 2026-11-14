"""
Core coupling utilities and policies.

Provides centralized management for:
- Precision and tolerance policies
- Tensor context and device management
- Audit records and reporting
- Unified error handling
"""

from .policy import (
    PrecisionPolicy,
    get_precision_policy,
    reset_precision_policy_cache,
    ScalingMode,
)

from .context import (
    TensorContext,
    get_tensor_context,
    apply_context,
)

from .conservation_checker import ConservationDiagnostics, ConservationFixer
from .coupling_operators import (
    TemporalLagOperator,
    ConstraintProjector,
    FluxKernel,
)

from .audit import (
    AuditRecord,
    AuditCollector,
    merge_audits,
)

from .errors import (
    ConservationError,
    CFSyncError,
    ScalingPolicyError,
    ScaleHierarchyError,
    CouplingError,
)

__all__ = [
    # Policy
    'PrecisionPolicy',
    'get_precision_policy',
    'reset_precision_policy_cache',
    'ScalingMode',
    
    # Context
    'TensorContext',
    'get_tensor_context',
    'apply_context',
    
    # Audit
    'AuditRecord',
    'AuditCollector',
    'ConservationDiagnostics',
    'ConservationFixer',
    'TemporalLagOperator',
    'ConstraintProjector',
    'FluxKernel',
    'merge_audits',
    
    # Errors
    'ConservationError',
    'CFSyncError', 
    'ScalingPolicyError',
    'ScaleHierarchyError',
    'CouplingError',
]
