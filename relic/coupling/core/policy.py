"""
Centralized precision and tolerance policy management.

Provides unified management of precision modes, tolerances, and dtypes
across all scaling operations to ensure consistency and maintainability.

Mathematical Expression:
    Policy mapping: mode -> {tolerance, dtype, conservation}
    where tolerance in [1e-12, 1e-3], dtype in {float32, float64}
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch

from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace
from .conservation_checker import require_relative_tolerance

logger = logging.getLogger(__name__)

PRECISION_POLICY_NAMESPACE = ParameterNamespace.COUPLING_DEFAULTS.value + ".precision"


class ScalingMode(Enum):
    """Precision modes for scaling operations."""
    STRICT_CONSERVATIVE = "strict_conservative"  # epsilon <= 1e-12, float64
    CONSERVATIVE = "conservative"                 # epsilon <= 1e-9, float64
    BALANCED = "balanced"                        # epsilon <= 1e-6, float32
    FAST_APPROX = "fast_approx"                  # epsilon <= 1e-3, float32


@dataclass
class PrecisionPolicy:
    """
    Precision policy of one scaling mode; every value is read from
    ``coupling.scales.defaults.precision.<mode>.*`` (no field has a default).

    Attributes:
        tolerance : float
            Relative conservation tolerance of the mode (dimensionless, finite, in (0, 1)):
            the ScaleManager facade ledger of every call in this mode compares with it
        dtype : torch.dtype
            Data type for computations
        enforce_conservation : bool
            Whether to enforce strict conservation
        max_iterations : int
            Maximum iterations for iterative solvers
        relative_tolerance : float
            Relative tolerance for convergence
        absolute_tolerance : float
            Absolute tolerance for convergence
    """
    tolerance: float
    dtype: torch.dtype
    enforce_conservation: bool
    max_iterations: int
    relative_tolerance: float
    absolute_tolerance: float


def _fetch_parameter_value(
    manager: Any,
    key: str,
    *,
    namespace: str,
    units: str = "dimensionless",
    expected_type: Optional[type] = None,
) -> Any:
    """Fetch a policy parameter from ParameterManager."""
    value = manager.get(
        key,
        namespace=namespace,
        units=units,
        expected_type=expected_type,
        required=True,
    )
    if expected_type is not None and not isinstance(value, expected_type):
        raise TypeError(
            f"Precision policy parameter {namespace}.{key} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def reset_precision_policy_cache() -> None:
    """Clear cached precision policies (useful when parameter manager changes)."""
    _POLICY_CACHE.clear()


def _load_policy_from_params(
    mode: ScalingMode,
    parameter_manager: Optional[Any] = None,
) -> PrecisionPolicy:
    """Load precision policy values from ParameterManager."""
    ns = f"{PRECISION_POLICY_NAMESPACE}.{mode.value}"
    pm = parameter_manager or get_global_manager()

    try:
        tolerance = float(_fetch_parameter_value(pm, "tolerance", namespace=ns, expected_type=float))
        # References: docs/source/theory/scale_coupling.md Section 3 (the precision tolerance is
        #   the one source of the ScaleManager facade ledger tolerance).
        # Mathematical expression: a ledger is accepted when e <= max(tol |L|, rounding bound),
        #   which only has a meaning for a finite 0 < tol < 1.
        # Parameter meanings: tol the relative tolerance of this mode; e the ledger residual; L the
        #   source amount of the ledger.
        # Units: tol dimensionless.
        # Domain/constraints: the parameter store checks bounds, but a NaN passes every bound
        #   comparison; a non-finite value or one outside (0, 1) is an invalid state and raises
        #   here (require_relative_tolerance), naming the key. No other key replaces it.
        # Sign convention: tol > 0.
        tolerance = require_relative_tolerance(tolerance, f"{ns}.tolerance")
        relative_tolerance = float(
            _fetch_parameter_value(pm, "relative_tolerance", namespace=ns, expected_type=float)
        )
        absolute_tolerance = float(
            _fetch_parameter_value(pm, "absolute_tolerance", namespace=ns, expected_type=float)
        )
        max_iterations = int(
            _fetch_parameter_value(pm, "max_iterations", namespace=ns, units="dimensionless", expected_type=int)
        )
        enforce_conservation = bool(
            _fetch_parameter_value(
                pm,
                "enforce_conservation",
                namespace=ns,
                units="dimensionless",
                expected_type=bool,
            )
        )
        dtype_str = str(
            _fetch_parameter_value(
                pm,
                "dtype",
                namespace=ns,
                units="dimensionless",
                expected_type=str,
            )
        ).strip()
        if not hasattr(torch, dtype_str):
            raise ValueError(f"Unsupported dtype '{dtype_str}' for precision policy namespace '{ns}'")
        dtype = getattr(torch, dtype_str)
        return PrecisionPolicy(
            tolerance=tolerance,
            dtype=dtype,
            enforce_conservation=enforce_conservation,
            max_iterations=max_iterations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Missing or invalid precision policy parameters under namespace '{ns}': {exc}"
        ) from exc


_POLICY_CACHE: Dict[Tuple[ScalingMode, int], PrecisionPolicy] = {}


def _make_cache_key(mode: ScalingMode, parameter_manager: Optional[Any]) -> Tuple[ScalingMode, Any]:
    """Build a stable cache key using the database path when available."""
    if parameter_manager is None:
        return (mode, "NO_PARAMETER_MANAGER")
    if hasattr(parameter_manager, "database_path"):
        # One key per database: the resolved path (a path that cannot be resolved raises).
        database_path = parameter_manager.database_path
        if isinstance(database_path, (str, Path)):
            database_path = Path(database_path)
        return (mode, str(database_path.resolve()))
    return (mode, id(parameter_manager))


def get_precision_policy(
    mode: ScalingMode,
    parameter_manager: Optional[Any] = None,
) -> PrecisionPolicy:
    """
    Get precision policy for a given mode.
    
    Parameters:
        mode : ScalingMode
            The scaling mode
            
    Returns:
        PrecisionPolicy: The precision policy configuration
        
    Raises:
        ValueError: If mode is not recognized or parameters are missing
    """
    pm = parameter_manager or get_global_manager()
    cache_key = _make_cache_key(mode, pm)
    if cache_key not in _POLICY_CACHE:
        policy = _load_policy_from_params(mode, parameter_manager=pm)
        logger.info(
            "Loaded precision policy for %s (tol=%s, rtol=%s, atol=%s, max_iter=%s, dtype=%s)",
            mode.value,
            policy.tolerance,
            policy.relative_tolerance,
            policy.absolute_tolerance,
            policy.max_iterations,
            policy.dtype,
        )
        _POLICY_CACHE[cache_key] = policy

    return _POLICY_CACHE[cache_key]
