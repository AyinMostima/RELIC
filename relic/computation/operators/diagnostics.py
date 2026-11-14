"""
Diagnostic Operators for RELIC Unified Computation Layer

This module provides diagnostic operator implementations that wrap the
original function-based diagnostic functions with a class-based OperatorBase
interface. These operators are used for stability checking, conservation
validation, and numerical diagnostics.

Mathematical Expressions:
    CFL Number: CFL = |v|dt/dx (stability constraint: CFL <= 1.0)
    Diffusion Stability: sigma = Ddt/dx2 (stability constraint: sigma <= 0.5)
    Conservation: int(phi_new - phi_old)dV = dtintS dV

Design Pattern:
    These classes wrap the original compute_* and check_* functions,
    providing diagnostic capabilities through the OperatorBase interface.

References:
    - Courant et al. (1928). Partial difference equations of mathematical physics
    - Strikwerda, J.C. (2004). Finite Difference Schemes and PDEs
"""

import math
from typing import Any, Dict, Optional, Union, Tuple
import torch

from .base import (
    OperatorBase,
    OperatorMetadata,
    reject_unexpected_runtime_parameters,
    resolve_operator_dtype,
)
from .registry import register_operator

# Import original implementation functions
from .functional import (
    compute_cfl_number,
    compute_diffusion_stability,
    check_conservation,
    apply_non_negative_constraint
)

Tensor = torch.Tensor


@register_operator("diagnostics.cfl_calculator", category="diagnostics")
class CFLCalculator(OperatorBase):
    """
    CFL (Courant-Friedrichs-Lewy) number calculator.
    
    This class wraps the original compute_cfl_number function,
    providing stability diagnostics for advection schemes.
    
    Mathematical Expression:
        CFL = |v|dt/dx  (1D)
        CFL = dt(|u|/dx + |v|/dy + |w|/dz)  (multi-D)
        
        Stability constraint: CFL <= 1.0 for explicit schemes
    
    Parameters (from ParameterManager or config):
        max_cfl_allowed : float
            Maximum allowed CFL number; required key
            computation.operators.diagnostics.max_cfl_allowed
        warn_threshold : float
            Warning threshold as a fraction of max_cfl_allowed; required key
            computation.operators.diagnostics.warn_threshold
    
    Usage Example:
        >>> from relic.computation.operators import OperatorFactory
        >>> calc = OperatorFactory.create(
        ...     "diagnostics.cfl_calculator",
        ...     config={"max_cfl_allowed": 1.0}
        ... )
        >>> cfl = calc.apply(
        ...     velocity=u,
        ...     grid_spacing=100.0,
        ...     time_step=1800.0
        ... )
    
    References:
        - Courant, R. et al. (1928). On the partial difference equations. IBM
    """
    
    operator_type = "diagnostics"
    
    def __init__(
        self,
        name: str = "cfl_calculator",
        config: Optional[dict] = None,
        metadata: Optional['OperatorMetadata'] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize CFL calculator."""
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.diagnostics",
            validate_parameters=validate_parameters
        )
        
        # Get parameters
        self.max_cfl_allowed = self.get_parameter(
            'max_cfl_allowed',
            validate=validate_parameters,
            description="Maximum allowed CFL number"
        )
        
        self.warn_threshold = self.get_parameter(
            'warn_threshold',
            validate=validate_parameters,
            description="Warning threshold for CFL"
        )
        
        # Create custom metadata if not provided
        if metadata is None:
            self.metadata = OperatorMetadata(
                name=f"{self.name}",
                operator_type="diagnostics",
                description="CFL number calculator for stability diagnostics",
                mathematical_expression="CFL = |v|dt/dx",
                input_units="m s-1",
                output_units="dimensionless",
                spatial_dimensions=0,
                references=[
                    "Courant, R. et al. (1928). IBM Journal",
                    "Holton, J.R. (2012). Dynamic Meteorology"
                ],
                parameters={
                    "max_cfl_allowed": f"{self.max_cfl_allowed}",
                    "warn_threshold": f"{self.warn_threshold}"
                }
            )
    
    def apply(
        self,
        velocity: Union[Tensor, Tuple[Tensor, ...]],
        grid_spacing: Union[float, Tuple[float, ...]],
        time_step: float,
    ) -> Tensor:
        """
        Calculate CFL number for given flow conditions.
        
        Parameters
        ----------
        velocity : Union[Tensor, Tuple[Tensor, ...]]
            Velocity field(s) [m s-1]
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing(s) [m]
        time_step : float
            Time step [s]
        
        Returns
        -------
        cfl_number : Tensor
            CFL number field [dimensionless]
        """
        # Move velocity to correct device if needed
        if isinstance(velocity, torch.Tensor):
            if velocity.device != self.device:
                velocity = velocity.to(self.device)
            if velocity.dtype != self.dtype:
                velocity = velocity.to(self.dtype)
        elif isinstance(velocity, (tuple, list)):
            velocity = tuple(
                v.to(self.device).to(self.dtype) if isinstance(v, torch.Tensor) else v
                for v in velocity
            )
        
        # Call original implementation
        cfl_number = compute_cfl_number(
            velocity=velocity,
            grid_spacing=grid_spacing,
            time_step=float(time_step)
        )
        
        # Check against threshold
        max_cfl = torch.max(cfl_number)
        if max_cfl > self.max_cfl_allowed:
            import warnings
            warnings.warn(
                f"CFL number {max_cfl:.3f} exceeds maximum allowed {self.max_cfl_allowed}"
            )
        elif max_cfl > self.warn_threshold * self.max_cfl_allowed:
            import warnings
            warnings.warn(
                f"CFL number {max_cfl:.3f} exceeds warning threshold "
                f"{self.warn_threshold * self.max_cfl_allowed:.3f}"
            )
        
        return cfl_number


@register_operator("diagnostics.stability_checker", category="diagnostics")
class StabilityChecker(OperatorBase):
    """
    Diffusion stability number checker.
    
    This class wraps the original compute_diffusion_stability function,
    providing stability diagnostics for diffusion schemes.
    
    Mathematical Expression:
        sigma = Ddt/dx2  (1D)
        sigma = Ddt(1/dx2 + 1/dy2 + 1/dz2)  (multi-D)
        
        Stability constraint: sigma <= 0.5 for explicit schemes
    
    Parameters (from ParameterManager or config):
        max_stability_allowed : float
            Maximum allowed stability number; required key
            computation.operators.diagnostics.max_stability_allowed
        warn_threshold : float
            Warning threshold as a fraction of max_stability_allowed; required key
            computation.operators.diagnostics.warn_threshold (shared with CFLCalculator)
    
    References:
        - Strikwerda, J.C. (2004). Finite Difference Schemes and PDEs
    """
    
    operator_type = "diagnostics"
    
    def __init__(
        self,
        name: str = "stability_checker",
        config: Optional[dict] = None,
        metadata: Optional['OperatorMetadata'] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize stability checker."""
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.diagnostics",
            validate_parameters=validate_parameters
        )

        # Get parameters
        self.max_stability_allowed = self.get_parameter(
            'max_stability_allowed',
            validate=validate_parameters,
            description="Maximum allowed diffusion stability number"
        )
        
        self.warn_threshold = self.get_parameter(
            'warn_threshold',
            validate=validate_parameters,
            description="Warning threshold for stability"
        )
        
        # Create custom metadata if not provided
        if metadata is None:
            self.metadata = OperatorMetadata(
                name=f"{self.name}",
                operator_type="diagnostics",
                description="Diffusion stability number checker",
                mathematical_expression="sigma = Ddt/dx2",
                input_units="m2 s-1",
                output_units="dimensionless",
                spatial_dimensions=0,
                references=[
                    "Strikwerda, J.C. (2004). Finite Difference Schemes",
                    "Morton, K.W. & Mayers, D.F. (2005). Numerical PDEs"
                ],
                parameters={
                    "max_stability_allowed": f"{self.max_stability_allowed}",
                    "warn_threshold": f"{self.warn_threshold}"
                }
            )
    
    def apply(
        self,
        diffusivity: Tensor,
        grid_spacing: Union[float, Tuple[float, ...]],
        time_step: float,
    ) -> Tensor:
        """
        Calculate diffusion stability number.
        
        Parameters
        ----------
        diffusivity : Tensor
            Diffusion coefficient field [m2 s-1]
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing(s) [m]
        time_step : float
            Time step [s]
        
        Returns
        -------
        stability_number : Tensor
            Diffusion stability number [dimensionless]
        """
        # Move to correct device/dtype
        if diffusivity.device != self.device:
            diffusivity = diffusivity.to(self.device)
        if diffusivity.dtype != self.dtype:
            diffusivity = diffusivity.to(self.dtype)
        
        # Call original implementation
        stability_number = compute_diffusion_stability(
            diffusivity=diffusivity,
            grid_spacing=grid_spacing,
            time_step=float(time_step)
        )
        
        # Check against threshold
        max_stability = torch.max(stability_number)
        max_stability_value = float(max_stability.item()) if isinstance(max_stability, Tensor) else float(max_stability)
        warn_limit = self.warn_threshold * self.max_stability_allowed
        if max_stability_value > self.max_stability_allowed:
            import warnings
            warnings.warn(
                f"Stability number {max_stability_value:.3f} exceeds maximum allowed {self.max_stability_allowed}"
            )
        elif max_stability_value > warn_limit:
            import warnings
            warnings.warn(
                f"Stability number {max_stability_value:.3f} exceeds warning threshold {warn_limit:.3f}"
            )
        
        return stability_number


@register_operator("diagnostics.conservation_checker", category="diagnostics")
class ConservationChecker(OperatorBase):
    """
    Mass/energy conservation checker.
    
    This class wraps the original check_conservation function,
    providing conservation diagnostics for transport processes.
    
    Mathematical Expression:
        Conservation: int(phi_new - phi_old)dV = dtintS dV
        Error: |dM_computed - dM_expected| / |M_old|
    
    Parameters (from ParameterManager or config):
        tolerance : float
            Relative tolerance for conservation; required ParameterManager key
            computation.operators.diagnostics.tolerance (no Python default)
        strict_mode : bool
            Raise exception on violation; required ParameterManager key
            computation.operators.diagnostics.strict_mode (no Python default)
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods
    """
    
    operator_type = "diagnostics"
    
    def __init__(
        self,
        name: str = "conservation_checker",
        config: Optional[dict] = None,
        metadata: Optional['OperatorMetadata'] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize conservation checker."""
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.diagnostics",
            validate_parameters=validate_parameters
        )
        
        # Get parameters
        self.tolerance = self.get_parameter(
            'tolerance',
            validate=validate_parameters,
            description="Relative conservation tolerance"
        )
        
        self.strict_mode = self.get_parameter(
            'strict_mode',
            validate=validate_parameters,
            description="Raise exception on conservation violation"
        )
        
        # Create custom metadata if not provided
        if metadata is None:
            self.metadata = OperatorMetadata(
                name=f"{self.name}",
                operator_type="diagnostics",
                description="Conservation checker for transport processes",
                mathematical_expression="Error = |dM_computed - dM_expected| / |M_old|",
                input_units="kg m-3",
                output_units="dimensionless",
                spatial_dimensions=0,
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods"
                ],
                parameters={
                    "tolerance": f"{self.tolerance}",
                    "strict_mode": f"{self.strict_mode}"
                }
            )
    
    def apply(
        self,
        field_old: Tensor,
        field_new: Tensor,
        source_term: Optional[Tensor] = None,
        time_step: Optional[float] = None,
        cell_volumes: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Check mass conservation for transport step.
        
        Parameters
        ----------
        field_old : Tensor
            Field values at previous time step
            Units: [kg m-3] for concentration
        field_new : Tensor
            Field values at current time step
            Units: [kg m-3]
        source_term : Optional[Tensor]
            Source/sink term [kg m-3 s-1]
        time_step : Optional[float]
            Time step [s]
        cell_volumes : Optional[Tensor]
            Cell volumes [m3]
        warn_on_violation : bool
            Emit warning when conservation error exceeds tolerance
        **kwargs
            Optional runtime parameter overrides
        
        Returns
        -------
        conservation_error : Tensor
            Relative conservation error [dimensionless]
        """
        # Runtime overrides replace the configured values for this call; an
        # unknown keyword is rejected, never ignored.
        reject_unexpected_runtime_parameters(
            kwargs,
            {"tolerance", "strict_mode", "warn_on_violation"},
        )
        tolerance = kwargs.get('tolerance', self.tolerance)
        strict_mode = kwargs.get('strict_mode', self.strict_mode)
        warn_on_violation = kwargs.get('warn_on_violation', True)
        
        # Move to correct device/dtype
        if field_old.device != self.device:
            field_old = field_old.to(self.device)
            field_new = field_new.to(self.device)
        if field_old.dtype != self.dtype:
            field_old = field_old.to(self.dtype)
            field_new = field_new.to(self.dtype)
        
        # Call original implementation
        conservation_error = check_conservation(
            field_old=field_old,
            field_new=field_new,
            source_term=source_term,
            time_step=time_step,
            cell_volumes=cell_volumes,
            tolerance=float(tolerance),
            warn_on_violation=bool(warn_on_violation)
        )
        
        # Check against tolerance in strict mode
        if strict_mode and conservation_error > tolerance:
            raise ValueError(
                f"Conservation error {conservation_error:.2e} exceeds tolerance {tolerance:.2e}"
            )
        
        return conservation_error


@register_operator("diagnostics.non_negative_constraint", category="diagnostics")
class NonNegativeConstraint(OperatorBase):
    """
    Non-negative constraint enforcer for concentration fields.

    This class wraps the original apply_non_negative_constraint function,
    providing a unified interface for enforcing non-negativity in fields.

    Mathematical Expression:
        c_new = max(c_old, small_value)  (clip method)
        c_new = max(c_old, non_negative_min_value)  (zero method)

    Parameters (from ParameterManager or config):
        method : str
            Constraint method "clip" or "zero"; required key
            computation.operators.diagnostics.method
        small_value : float
            Floor of the clip method in the units of the field; required key
            computation.operators.diagnostics.small_value

    Usage Example:
        >>> from relic.computation.operators import OperatorFactory
        >>> constraint = OperatorFactory.create(
        ...     "diagnostics.non_negative_constraint",
        ...     config={"method": "clip", "small_value": 1e-12}
        ... )
        >>> result = constraint.apply(field=concentration)

    References:
        - Numerical methods for transport equations
    """

    operator_type = "diagnostics"

    def __init__(
        self,
        name: str = "non_negative_constraint",
        config: Optional[dict] = None,
        metadata: Optional['OperatorMetadata'] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize non-negative constraint operator."""
        # namespace and validate_parameters are owned by OperatorBase; the
        # resolved namespace is passed to it (a value assigned before super()
        # was overwritten with None there).
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.diagnostics",
            validate_parameters=validate_parameters,
        )

        # Get parameters
        self.method = self.get_parameter(
            'method',
            namespace=self.namespace,
            validate=self.validate_parameters
        )

        self.small_value = self.get_parameter(
            'small_value',
            namespace=self.namespace,
            validate=self.validate_parameters
        )

        if metadata is None:
            self.metadata = OperatorMetadata(
                name=f"{self.name}",
                operator_type="diagnostics",
                description="Non-negative constraint for concentration fields",
                mathematical_expression=(
                    "c_new = max(c_old, small_value) (clip); "
                    "c_new = max(c_old, non_negative_min_value) (zero)"
                ),
                input_units="[field_units]",
                output_units="[field_units]",
                spatial_dimensions=0,
                parameters={
                    "method": f"{self.method}",
                    "small_value": f"{self.small_value}",
                },
            )

    def apply(
        self,
        field: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Apply non-negative constraint to field.

        Parameters
        ----------
        field : Tensor
            Field values that should be non-negative
        **kwargs
            Optional runtime parameter overrides

        Returns
        -------
        field_constrained : Tensor
            Field with non-negative constraint applied
        """
        # Runtime overrides replace the configured values for this call; an
        # unknown keyword is rejected, never ignored.
        reject_unexpected_runtime_parameters(kwargs, {"method", "small_value"})
        method = kwargs.get('method', self.method)
        small_value = kwargs.get('small_value', self.small_value)

        # Move to correct device/dtype
        if field.device != self.device:
            field = field.to(self.device)
        if field.dtype != self.dtype:
            field = field.to(self.dtype)

        # The clip floor (computation.operators.diagnostics.small_value or a
        # runtime override) is passed only to method 'clip'; method 'zero'
        # reads its own floor, transport.numerical.non_negative_min_value.
        field_constrained = apply_non_negative_constraint(
            field=field,
            method=method,
            small_value=float(small_value) if method == "clip" else None,
        )

        return field_constrained


def compute_mesh_compatibility_score(
    operator_metadata: OperatorMetadata,
    grid_spacing: float,
    partition_size: int,
) -> Dict[str, Any]:
    """Compute mesh compatibility score from operator spatial requirements.

    Mathematical Expression:
        stability_ratio = grid_spacing / min_resolution  (target >= 1.0)
        halo_overhead = 2 * halo_depth / partition_size
        score = min(1.0, stability_ratio) * max(0.0, 1.0 - halo_overhead)

    References:
        - Bauer et al. (2015), Geosci. Model Dev. 8, 3181-3201.

    Parameters:
        operator_metadata: metadata that may define spatial_requirements
        grid_spacing: actual grid spacing [m]
        partition_size: grid cells per partition dimension

    Raises:
        ValueError: grid_spacing is not finite and positive, partition_size
            is not a positive integer, or a declared min_resolution is not
            finite and positive.
    """
    if isinstance(partition_size, bool) or int(partition_size) != partition_size:
        raise ValueError(f"partition_size must be a positive integer, got {partition_size!r}")
    if partition_size < 1:
        raise ValueError(f"partition_size must be a positive integer, got {partition_size!r}")
    spacing = float(grid_spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"grid_spacing must be finite and positive, got {grid_spacing!r}")

    spatial_req = operator_metadata.spatial_requirements
    if spatial_req is None:
        return {
            "score": 1.0,
            "stability_ratio": 1.0,
            "halo_overhead": 0.0,
            "has_spatial_requirements": False,
        }

    # References:
    #   Bauer et al. (2015), Geosci. Model Dev. 8, 3181-3201 (stencil and
    #   mesh interplay; the citation carried by SpatialRequirements).
    # Mathematical expression:
    #   r = dx / dx_min (r = 1 when no dx_min is declared);
    #   h = 2 d / P; score = min(1, r) * max(0, 1 - h)
    # Parameter meanings:
    #   dx: grid spacing; dx_min: declared minimum resolution; d: halo
    #   depth in cells; P: cells per partition dimension; r: resolution
    #   ratio; h: halo overhead fraction.
    # Units:
    #   dx, dx_min [m]; d, P [cells]; r, h, score [1].
    # Domain/constraints:
    #   dx and dx_min finite and positive, P a positive integer; invalid
    #   inputs raise instead of being clamped to a neutral score.
    # Sign convention:
    #   None; all quantities are non-negative ratios.
    min_resolution = spatial_req.min_resolution
    if min_resolution is None:
        stability_ratio = 1.0
    else:
        resolution = float(min_resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError(
                "spatial_requirements.min_resolution must be finite and "
                f"positive when declared, got {min_resolution!r}"
            )
        stability_ratio = spacing / resolution

    halo_depth = int(spatial_req.halo_depth)
    halo_overhead = 2.0 * float(halo_depth) / float(partition_size)

    stability_component = min(1.0, stability_ratio)
    halo_component = max(0.0, 1.0 - halo_overhead)
    score = max(0.0, min(1.0, stability_component * halo_component))

    return {
        "score": score,
        "stability_ratio": stability_ratio,
        "halo_overhead": halo_overhead,
        "stability_component": stability_component,
        "halo_component": halo_component,
        "has_spatial_requirements": True,
        "locality": spatial_req.locality,
        "preferred_layout": spatial_req.preferred_layout,
    }


# Export all classes
__all__ = [
    'CFLCalculator',
    'StabilityChecker',
    'ConservationChecker',
    'NonNegativeConstraint',
    'compute_mesh_compatibility_score',
]
