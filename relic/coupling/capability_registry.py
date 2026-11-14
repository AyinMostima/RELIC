"""
Capability registry and error codes for coupling layer.

This module defines standardized error codes and capability declarations
for features that are not implemented or have limited support.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Type
from dataclasses import dataclass, field
from threading import RLock
import logging

from relic.parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.parameters import get_global_manager
from .core.conservation_checker import ConservationFixer
from .core.coupling_operators import (
    ConstraintProjector,
    FluxKernel,
    TemporalLagOperator,
)

if TYPE_CHECKING:
    from relic.parameters import UnifiedParameterManager

_CAPABILITY_OVERRIDE_WARNED_NO_PM: bool = False
_EXTERNAL_AVAILABILITY: Optional[bool] = None
_CAPABILITY_LOCK = RLock()
_VALID_CAPABILITY_STATUSES = frozenset(
    {"implemented", "partial", "not_implemented"}
)
logger = logging.getLogger(__name__)


def _get_param_manager(
    parameter_manager: Optional["UnifiedParameterManager"] = None,
) -> Optional["UnifiedParameterManager"]:
    """Resolve the parameter manager lazily to avoid import-time initialization."""
    if parameter_manager is not None:
        return parameter_manager
    return get_global_manager()

class CouplingErrorCode(Enum):
    """Standardized error codes for coupling layer."""

    # Spatial regridding errors (1000-1099)
    IRREGULAR_GRID_NOT_SUPPORTED = "COUPLING_E1001"
    UNSTRUCTURED_MESH_NOT_SUPPORTED = "COUPLING_E1002"
    ADAPTIVE_GRID_NOT_SUPPORTED = "COUPLING_E1003"
    CURVILINEAR_GRID_NOT_SUPPORTED = "COUPLING_E1004"

    # Temporal coupling errors (1100-1199)
    VARIABLE_TIMESTEP_NOT_SUPPORTED = "COUPLING_E1101"
    ADAPTIVE_TIMESTEPPING_NOT_SUPPORTED = "COUPLING_E1102"

    # External model coupling errors (1200-1299)
    EXTERNAL_MODEL_NOT_AVAILABLE = "COUPLING_E1201"
    PROTOCOL_VERSION_MISMATCH = "COUPLING_E1202"
    COUPLING_CONTRACT_VIOLATION = "COUPLING_E1203"

    # Scale coupling errors (1300-1399)
    HIERARCHICAL_COUPLING_LIMITED = "COUPLING_E1301"
    CROSS_SCALE_CONSERVATION_ERROR = "COUPLING_E1302"


@dataclass
class CapabilityDeclaration:
    """
    Declaration of coupling layer capabilities and limitations.

    Attributes:
        feature: Feature name
        status: Implementation status (implemented, partial, not_implemented)
        limitations: List of current limitations
        alternatives: Suggested alternatives or workarounds
        error_code: Associated error code
        references: Scientific/technical references
    """
    feature: str
    status: str
    limitations: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    error_code: Optional[CouplingErrorCode] = None
    references: List[str] = field(default_factory=list)


def _clone_capability(
    declaration: CapabilityDeclaration,
) -> CapabilityDeclaration:
    """Return a detached declaration owned by the registry or one query."""
    return CapabilityDeclaration(
        feature=declaration.feature,
        status=declaration.status,
        limitations=list(declaration.limitations),
        alternatives=list(declaration.alternatives),
        error_code=declaration.error_code,
        references=list(declaration.references),
    )


class CouplingCapabilityRegistry:
    """Registry of coupling layer capabilities and limitations."""

    CAPABILITIES: Dict[str, CapabilityDeclaration] = {
        "regular_grid_bilinear": CapabilityDeclaration(
            feature="Bilinear interpolation on regular grids",
            status="implemented",
            limitations=[],
            alternatives=[],
            references=["Press et al. (2007) Numerical Recipes"],
        ),
        "irregular_grid_bilinear": CapabilityDeclaration(
            feature="Bilinear interpolation on irregular grids",
            status="implemented",
            limitations=[
                "Uses inverse-distance weighting on k-nearest neighbors",
                "Performance considerations for large grids without KDTree",
            ],
            alternatives=[
                "Use regular grids for exact bilinear stencils",
                "Use nearest neighbor for categorical data",
            ],
            references=[
                "Shepard (1968) Two-dimensional interpolation",
                "ESMF Regridding Guide",
            ],
        ),
        "conservative_regridding_regular": CapabilityDeclaration(
            feature="Conservative regridding on regular grids",
            status="implemented",
            limitations=["2D only currently"],
            alternatives=[],
            references=["Jones (1999) Conservative remapping"],
        ),
        "conservative_regridding_irregular": CapabilityDeclaration(
            feature="Conservative regridding on irregular grids",
            status="not_implemented",
            limitations=[
                "Requires polygon intersection algorithms",
                "Complex for unstructured meshes",
                "High computational cost",
            ],
            alternatives=[
                "Use optional xESMF library",
                "Regrid categorical fields by nearest neighbor (the rule of their categorical basis)",
                "Pre-compute weights offline",
            ],
            error_code=CouplingErrorCode.IRREGULAR_GRID_NOT_SUPPORTED,
            references=["SCRIP documentation", "ESMF User Guide", "xESMF documentation"],
        ),
        "nearest_neighbor_irregular": CapabilityDeclaration(
            feature="Nearest neighbor on irregular grids",
            status="implemented",
            limitations=[
                "KDTree optional; brute-force used when unavailable",
                "Memory overhead for large grids",
            ],
            alternatives=[
                "Enable SciPy KDTree for large grids",
                "Use regular grid approximation",
            ],
            references=["Bentley (1975) KD-trees"],
        ),
        "adaptive_timestep": CapabilityDeclaration(
            feature="Error-controlled adaptive timestep",
            status="implemented",
            limitations=[
                "Internal substepping only; external coupling period unchanged",
            ],
            alternatives=[
                "Use fixed timestep with conservative stability limits",
            ],
            references=[
                "Hairer and Wanner (1996) Solving Ordinary Differential Equations II",
                "Gustafsson (1991) step size control",
            ],
        ),
        "coupling_operator_conservation_fixer": CapabilityDeclaration(
            feature="Conservation fixer coupling operator",
            status="implemented",
            limitations=[],
            alternatives=[],
            references=[
                "Jones (1999) Conservative remapping",
                "Lauritzen et al. (2010)",
            ],
        ),
        "coupling_operator_temporal_lag": CapabilityDeclaration(
            feature="Temporal lag coupling operator",
            status="implemented",
            limitations=[],
            alternatives=[],
            references=[
                "Balaji et al. (2006) coupling interval effects",
            ],
        ),
        "coupling_operator_constraint_projector": CapabilityDeclaration(
            feature="Constraint projector coupling operator",
            status="implemented",
            limitations=[],
            alternatives=[],
            references=[
                "Lauritzen et al. (2011) shape-preserving transport",
            ],
        ),
        "coupling_operator_flux_kernel": CapabilityDeclaration(
            feature="Flux kernel coupling operator",
            status="implemented",
            limitations=[],
            alternatives=[],
            references=[
                "Clark et al. (2015) inter-compartment flux",
            ],
        ),
    }

    COUPLING_OPERATOR_TYPES: Dict[str, Type[Any]] = {
        "conservation_fixer": ConservationFixer,
        "temporal_lag": TemporalLagOperator,
        "constraint_projector": ConstraintProjector,
        "flux_kernel": FluxKernel,
    }

    @classmethod
    def register_feature(cls, declaration: CapabilityDeclaration, name: Optional[str] = None) -> None:
        """Register or update a capability declaration."""
        if not declaration.feature.strip():
            raise ValueError("Capability feature must be a non-empty string")
        if declaration.status not in _VALID_CAPABILITY_STATUSES:
            raise ValueError(
                "Capability status must be implemented, partial, or not_implemented"
            )
        key = name if name is not None else declaration.feature
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Capability registry key must be a non-empty string")
        with _CAPABILITY_LOCK:
            cls.CAPABILITIES[key] = _clone_capability(declaration)

    @classmethod
    def get_capability(cls, feature: str) -> Optional[CapabilityDeclaration]:
        """Get capability declaration."""
        return _resolve_capability_snapshot().get(feature)

    @classmethod
    def list_capabilities(cls, status: Optional[str] = None) -> List[str]:
        """List capability names, optionally filtered by status."""
        capabilities = _resolve_capability_snapshot()
        if status is None:
            return list(capabilities.keys())
        return [
            name
            for name, capability in capabilities.items()
            if capability.status == status
        ]

    @classmethod
    def is_implemented(cls, feature: str) -> bool:
        """Check if a feature is implemented."""
        cap = cls.get_capability(feature)
        return cap.status == "implemented" if cap else False

    @classmethod
    def get_error_code(cls, feature: str) -> Optional[str]:
        """Get error code for a feature."""
        cap = cls.get_capability(feature)
        return cap.error_code.value if cap and cap.error_code else None

    @classmethod
    def get_alternatives(cls, feature: str) -> List[str]:
        """Return suggested alternatives for a given feature."""
        cap = cls.get_capability(feature)
        return list(cap.alternatives) if cap else []

    @classmethod
    def get_implementation_status(cls) -> Dict[str, List[str]]:
        """Get summary of implementation status."""
        capabilities = _resolve_capability_snapshot()
        status = {
            "implemented": [],
            "partial": [],
            "not_implemented": [],
        }
        for name, declaration in capabilities.items():
            if declaration.status in status:
                status[declaration.status].append(name)
        return status

    @classmethod
    def get_coupling_operator_type(cls, operator_name: str) -> Optional[Type[Any]]:
        """Get coupling operator class by registry name."""
        with _CAPABILITY_LOCK:
            return cls.COUPLING_OPERATOR_TYPES.get(operator_name)

    @classmethod
    def list_coupling_operator_types(cls) -> List[str]:
        """List registered coupling operator names."""
        with _CAPABILITY_LOCK:
            return list(cls.COUPLING_OPERATOR_TYPES.keys())

    @classmethod
    def register_coupling_operator_type(cls, name: str, operator_cls: Type[Any]) -> None:
        """Register or override a coupling operator class."""
        with _CAPABILITY_LOCK:
            cls.COUPLING_OPERATOR_TYPES[name] = operator_cls


def _base_capability_snapshot() -> Dict[str, CapabilityDeclaration]:
    """Copy canonical implementation facts under the registry lock."""
    with _CAPABILITY_LOCK:
        return {
            name: _clone_capability(declaration)
            for name, declaration in CouplingCapabilityRegistry.CAPABILITIES.items()
        }


def _resolve_capability_snapshot(
    parameter_manager: Optional["UnifiedParameterManager"] = None,
) -> Dict[str, CapabilityDeclaration]:
    """Resolve policy disablement on a detached copy of implementation facts."""
    global _CAPABILITY_OVERRIDE_WARNED_NO_PM
    capabilities = _base_capability_snapshot()
    pm = _get_param_manager(parameter_manager)
    if pm is None:
        with _CAPABILITY_LOCK:
            if not _CAPABILITY_OVERRIDE_WARNED_NO_PM:
                logger.warning(
                    "Capability overrides skipped because ParameterManager is not initialized yet. "
                    "Returning canonical implementation facts."
                )
                _CAPABILITY_OVERRIDE_WARNED_NO_PM = True
        return capabilities
    namespace = resolve_namespace(ParameterNamespace.COUPLING_CAPABILITIES)
    for name, capability in capabilities.items():
        parameter_key = f"{namespace}.{name}.enabled"
        # An override is optional (an absent entry keeps the implementation fact); a failed lookup
        # and an override that is not a bool raise.
        enabled = pm.get(
            parameter_key,
            units="dimensionless",
            expected_type=bool,
            required=False,
        )
        if enabled is None:
            continue
        if not isinstance(enabled, bool):
            raise TypeError(
                f"Capability override {parameter_key} must be a bool, got {type(enabled).__name__}."
            )
        if enabled is False:
            capability.status = "not_implemented"
            limitation = f"Disabled by ParameterManager: {parameter_key}=false"
            if limitation not in capability.limitations:
                capability.limitations.append(limitation)
    return capabilities


def create_not_implemented_error(
    feature: str,
    additional_info: str = ""
) -> NotImplementedError:
    """
    Create a standardized NotImplementedError with capability information.

    Args:
        feature: Feature name from capability registry
        additional_info: Additional context-specific information

    Returns:
        NotImplementedError with detailed message
    """
    cap = CouplingCapabilityRegistry.get_capability(feature)

    if not cap:
        return NotImplementedError(f"Feature '{feature}' not in capability registry")

    # Build error message
    msg_parts = [f"Feature not implemented: {cap.feature}"]

    if cap.error_code:
        msg_parts.append(f"Error code: {cap.error_code.value}")

    if cap.limitations:
        msg_parts.append(f"Limitations: {', '.join(cap.limitations)}")

    if cap.alternatives:
        msg_parts.append("Alternatives:")
        for alt in cap.alternatives:
            msg_parts.append(f"  - {alt}")

    if additional_info:
        msg_parts.append(f"Additional info: {additional_info}")

    return NotImplementedError("\n".join(msg_parts))


def register_external_availability(is_available: bool, reason: str = "") -> None:
    """Record external adapter availability and synchronize capability registry."""
    global _EXTERNAL_AVAILABILITY
    with _CAPABILITY_LOCK:
        _EXTERNAL_AVAILABILITY = bool(is_available)
        status = "implemented" if is_available else "not_implemented"
        limitations = (
            []
            if is_available
            else [reason or "External coupling dependencies missing"]
        )
        CouplingCapabilityRegistry.register_feature(
            CapabilityDeclaration(
                feature="external_coupling_runtime",
                status=status,
                limitations=limitations,
                alternatives=(
                    ["Install relic[external] extras"]
                    if not is_available
                    else []
                ),
                error_code=CouplingErrorCode.EXTERNAL_MODEL_NOT_AVAILABLE,
                references=["Coupling bootstrap availability check"],
            ),
            name="external_coupling_runtime",
        )


def is_external_available() -> bool:
    """Return cached external coupling availability state."""
    with _CAPABILITY_LOCK:
        return bool(_EXTERNAL_AVAILABILITY)


# Export key items
__all__ = [
    "CouplingErrorCode",
    "CapabilityDeclaration",
    "CouplingCapabilityRegistry",
    "ConservationFixer",
    "TemporalLagOperator",
    "ConstraintProjector",
    "FluxKernel",
    "create_not_implemented_error",
    "register_external_availability",
    "is_external_available",
]



