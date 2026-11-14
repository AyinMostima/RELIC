"""
Gradient Operators for RELIC Unified Computation Layer

This module provides gradient operator implementations that wrap the
original function-based operators from relic.computation.operators.functional
with a class-based OperatorBase interface. This maintains 100% scientific
correctness while enabling factory instantiation, parameter management,
and configuration-driven workflows.

Mathematical Expression:
    Gradient: gradphi = (dphi/dx, dphi/dy, dphi/dz)
    
    Finite Difference Schemes:
    - Central (2nd order): dphi/dx ~= (phi[i+1] - phi[i-1])/(2dx)
    - Forward: dphi/dx ~= (phi[i+1] - phi[i])/dx
    - Backward: dphi/dx ~= (phi[i] - phi[i-1])/dx
    - 4th order: dphi/dx ~= (-phi[i+2] + 8phi[i+1] - 8phi[i-1] + phi[i-2])/(12dx)

Design Pattern:
    These classes act as thin wrappers around the original compute_gradient_*
    functions, providing:
    - OperatorBase interface compliance
    - Factory instantiation support
    - ParameterManager integration
    - Metadata and documentation
    
    The scientific implementation is NOT duplicated - all wrapper classes
    internally call the original functions to preserve correctness.

References:
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM
    - Fornberg, B. (1988). Generation of finite difference formulas
"""

from typing import Any, Optional, Tuple
import torch

from .base import (
    OperatorBase,
    OperatorMetadata,
    SpatialRequirements,
    reject_unexpected_runtime_parameters,
    resolve_operator_dtype,
)
from .registry import register_operator

# Import original implementation functions (single source of truth)
from .functional import (
    compute_gradient_1d,
    compute_gradient_2d,
    compute_gradient_3d
)

Tensor = torch.Tensor


def _normalize_gradient_order(order: Any) -> int:
    """Normalize and validate the supported centered-difference order."""
    if isinstance(order, torch.Tensor):
        if order.numel() != 1:
            raise ValueError("Finite-difference order must be 2 or 4")
        order = order.detach().cpu().item()
    if isinstance(order, bool):
        raise ValueError("Finite-difference order must be 2 or 4")
    try:
        normalized = int(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Finite-difference order must be 2 or 4") from exc
    if normalized not in (2, 4) or float(order) != float(normalized):
        raise ValueError("Finite-difference order must be 2 or 4")
    return normalized


def _prepare_gradient_field(
    field: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Validate a field and move it to the operator computation policy."""
    if not isinstance(field, torch.Tensor):
        raise TypeError("Gradient field must be a torch.Tensor")
    prepared = field.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(prepared).all()):
        raise ValueError("Gradient field values must be finite")
    return prepared


@register_operator("gradient.central_difference_1d", category="spatial")
class CentralDifferenceGradient1D(OperatorBase):
    """
    1D gradient operator using central finite differences.
    
    This class wraps the original compute_gradient_1d function,
    providing OperatorBase interface and ParameterManager integration.
    
    Mathematical Expression:
        dphi/dx ~= (phi[i+1] - phi[i-1])/(2dx)  [2nd order]
        dphi/dx ~= (-phi[i+2] + 8phi[i+1] - 8phi[i-1] + phi[i-2])/(12dx)  [4th order]
    
    Parameters (from ParameterManager or config):
        dx : float
            Grid spacing in x-direction [m]
        boundary_method : str
            Boundary treatment: "zero", "extrapolate", "periodic"
        order : int
            Finite difference order: 2 or 4
    
    Usage Example:
        >>> from relic.computation.operators import OperatorFactory
        >>> op = OperatorFactory.create(
        ...     "gradient.central_difference_1d",
        ...     config={"dx": 100.0, "boundary_method": "periodic", "order": 2}
        ... )
        >>> gradient = op.apply(field)
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 1
    """
    
    operator_type = "gradient"
    
    def __init__(
        self,
        name: str = "gradient_1d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """
        Initialize 1D gradient operator.
        
        Parameters
        ----------
        name : str
            Operator instance name
        config : Optional[dict]
            Configuration dictionary with operator parameters
        device : Optional[torch.device]
            Computation device
        dtype : Optional[torch.dtype]
            Data type for computations
        namespace : Optional[str]
            Parameter namespace override
        validate_parameters : bool
            Whether to validate parameters on initialization
        """
        resolved_dtype = resolve_operator_dtype(dtype)
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolved_dtype,
            namespace=namespace or "computation.operators.gradient",
            validate_parameters=validate_parameters
        )
        
        # Get parameters from ParameterManager or config
        self.dx = self.get_parameter(
            'dx',
            validate=validate_parameters,
            description="Grid spacing in x-direction"
        )
        
        self.boundary_method = self.get_parameter(
            'boundary_method',
            validate=validate_parameters,
            description="Boundary condition type"
        )

        self.order = self.get_parameter(
            'order',
            validate=validate_parameters,
            description="Finite difference order"
        )
        self.order = _normalize_gradient_order(self.order)

        if metadata is None:
            radius = 2 if self.order == 4 else 1
            expression = (
                "dphi/dx ~= "
                "(-phi[i+2] + 8phi[i+1] - 8phi[i-1] + phi[i-2])/(12dx)"
                if self.order == 4
                else "dphi/dx ~= (phi[i+1] - phi[i-1])/(2dx)"
            )
            self.metadata = OperatorMetadata(
                name=f"{self.name}",
                operator_type="gradient",
                description="1D central difference gradient operator",
                mathematical_expression=expression,
                spatial_dimensions=1,
                input_units="[field_units]",
                output_units="[field_units] m-1",
                requires_grad=True,
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM"
                ],
                parameters={
                    "dx": f"{self.dx} m",
                    "boundary_method": self.boundary_method,
                    "order": self.order
                },
                spatial_requirements=SpatialRequirements(
                    stencil_radius=radius,
                    halo_depth=radius,
                    # Nearest-neighbour (radius 1) stencil is compact; the
                    # fourth-order radius-2 stencil is wide.
                    locality="compact" if radius == 1 else "wide",
                ),
            )
    
    def apply(self, field: Tensor, **kwargs) -> Tensor:
        """
        Apply 1D gradient operator to field.
        
        Parameters
        ----------
        field : Tensor
            Input field, shape (..., nx)
            Units: [field_units]
        **kwargs
            Optional runtime parameter overrides:
            - dx : float
            - boundary_method : str
            - order : int
        
        Returns
        -------
        gradient : Tensor
            Spatial gradient dphi/dx, same shape as input
            Units: [field_units m^-1]
        """
        reject_unexpected_runtime_parameters(
            kwargs,
            {"dx", "boundary_method", "order"},
        )

        # Get runtime parameter overrides or use defaults
        dx = kwargs.get('dx', self.dx)
        # The configured choice was checked against the YAML allowed_values
        # of computation.operators.gradient.boundary_method (OperatorBase);
        # the functional kernel owns the implemented closures.
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        order = _normalize_gradient_order(kwargs.get('order', self.order))

        field = _prepare_gradient_field(
            field,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Call original implementation (single source of truth)
        gradient = compute_gradient_1d(
            field=field,
            dx=float(dx),
            boundary_method=str(boundary_method),
            order=int(order)
        )
        
        return gradient


@register_operator("gradient.central_difference_2d", category="spatial")
class CentralDifferenceGradient2D(OperatorBase):
    """
    2D gradient operator using central finite differences.
    
    This class wraps the original compute_gradient_2d function,
    computing (dphi/dx, dphi/dy) for 2D fields.
    
    Mathematical Expression:
        gradphi = (dphi/dx, dphi/dy)
        dphi/dx ~= (phi[i+1,j] - phi[i-1,j])/(2dx)
        dphi/dy ~= (phi[i,j+1] - phi[i,j-1])/(2dy)
    
    Parameters (from ParameterManager or config):
        dx : float
            Grid spacing in x-direction [m]
        dy : float
            Grid spacing in y-direction [m]
        boundary_method : str
            Boundary treatment method
    
    Returns:
        Tuple[Tensor, Tensor]: (grad_x, grad_y)
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 2
    """
    
    operator_type = "gradient"
    
    def __init__(
        self,
        name: str = "gradient_2d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize 2D gradient operator."""
        resolved_dtype = resolve_operator_dtype(dtype)
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolved_dtype,
            namespace=namespace or "computation.operators.gradient",
            validate_parameters=validate_parameters
        )
        
        # Get parameters
        self.dx = self.get_parameter(
            'dx',
            validate=validate_parameters,
            description="Grid spacing in x-direction"
        )
        
        self.dy = self.get_parameter(
            'dy',
            validate=validate_parameters,
            description="Grid spacing in y-direction"
        )
        
        self.boundary_method = self.get_parameter(
            'boundary_method',
            validate=validate_parameters,
            description="Boundary condition type"
        )

        if metadata is None:
            self.metadata = OperatorMetadata(
                name=f"{self.name}",
                operator_type="gradient",
                description="2D central difference gradient operator",
                mathematical_expression="gradphi = (dphi/dx, dphi/dy)",
                spatial_dimensions=2,
                input_units="[field_units]",
                output_units="[field_units] m-1",
                requires_grad=True,
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods"
                ],
                parameters={
                    "dx": f"{self.dx} m",
                    "dy": f"{self.dy} m",
                    "boundary_method": self.boundary_method
                },
                spatial_requirements=SpatialRequirements(
                    stencil_radius=1,
                    halo_depth=1,
                    locality="compact",
                ),
            )
    
    def apply(self, field: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Apply 2D gradient operator to field.
        
        Parameters
        ----------
        field : Tensor
            Input field, shape (..., ny, nx)
            Units: [field_units]
        **kwargs
            Optional runtime parameter overrides
        
        Returns
        -------
        grad_x : Tensor
            Gradient in x-direction, shape (..., ny, nx)
            Units: [field_units m^-1]
        grad_y : Tensor
            Gradient in y-direction, shape (..., ny, nx)
            Units: [field_units m^-1]
        """
        reject_unexpected_runtime_parameters(
            kwargs,
            {"dx", "dy", "boundary_method"},
        )

        # Get runtime parameter overrides
        dx = kwargs.get('dx', self.dx)
        dy = kwargs.get('dy', self.dy)
        # The configured choice was checked against the YAML allowed_values
        # of computation.operators.gradient.boundary_method (OperatorBase);
        # the functional kernel owns the implemented closures.
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        
        field = _prepare_gradient_field(
            field,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Call original implementation
        grad_x, grad_y = compute_gradient_2d(
            field=field,
            dx=float(dx),
            dy=float(dy),
            boundary_method=str(boundary_method)
        )
        
        return grad_x, grad_y


@register_operator("gradient.central_difference_3d", category="spatial")
class CentralDifferenceGradient3D(OperatorBase):
    """
    3D gradient operator using central finite differences.
    
    This class wraps the original compute_gradient_3d function,
    computing (dphi/dx, dphi/dy, dphi/dz) for 3D fields.
    
    Mathematical Expression:
        gradphi = (dphi/dx, dphi/dy, dphi/dz)
        dphi/dx ~= (phi[i+1,j,k] - phi[i-1,j,k])/(2dx)
        dphi/dy ~= (phi[i,j+1,k] - phi[i,j-1,k])/(2dy)
        dphi/dz ~= (phi[i,j,k+1] - phi[i,j,k-1])/(2dz)
    
    Parameters (from ParameterManager or config):
        dx, dy, dz : float
            Grid spacings in x, y, z directions [m]
        boundary_method : str
            Boundary treatment method
    
    Returns:
        Tuple[Tensor, Tensor, Tensor]: (grad_x, grad_y, grad_z)
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 3
    """
    
    operator_type = "gradient"
    
    def __init__(
        self,
        name: str = "gradient_3d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize 3D gradient operator."""
        resolved_dtype = resolve_operator_dtype(dtype)
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolved_dtype,
            namespace=namespace or "computation.operators.gradient",
            validate_parameters=validate_parameters
        )
        
        # Get parameters
        self.dx = self.get_parameter(
            'dx',
            validate=validate_parameters,
            description="Grid spacing in x-direction"
        )
        
        self.dy = self.get_parameter(
            'dy',
            validate=validate_parameters,
            description="Grid spacing in y-direction"
        )
        
        self.dz = self.get_parameter(
            'dz',
            validate=validate_parameters,
            description="Grid spacing in z-direction"
        )
        
        self.boundary_method = self.get_parameter(
            'boundary_method',
            validate=validate_parameters,
            description="Boundary condition type"
        )

        if metadata is None:
            self.metadata = OperatorMetadata(
                name=f"{self.name}",
                operator_type="gradient",
                description="3D central difference gradient operator",
                mathematical_expression="gradphi = (dphi/dx, dphi/dy, dphi/dz)",
                spatial_dimensions=3,
                input_units="[field_units]",
                output_units="[field_units] m-1",
                requires_grad=True,
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods"
                ],
                parameters={
                    "dx": f"{self.dx} m",
                    "dy": f"{self.dy} m",
                    "dz": f"{self.dz} m",
                    "boundary_method": self.boundary_method
                },
                spatial_requirements=SpatialRequirements(
                    stencil_radius=1,
                    halo_depth=1,
                    locality="compact",
                ),
            )
    
    def apply(
        self,
        field: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply 3D gradient operator to field.
        
        Parameters
        ----------
        field : Tensor
            Input field, shape (batch, nz, ny, nx)
            Units: [field_units]
        **kwargs
            Optional runtime parameter overrides
        
        Returns
        -------
        grad_x : Tensor
            Gradient in x-direction, shape (batch, nz, ny, nx)
            Units: [field_units m^-1]
        grad_y : Tensor
            Gradient in y-direction, shape (batch, nz, ny, nx)
            Units: [field_units m^-1]
        grad_z : Tensor
            Gradient in z-direction, shape (batch, nz, ny, nx)
            Units: [field_units m^-1]
        """
        reject_unexpected_runtime_parameters(
            kwargs,
            {"dx", "dy", "dz", "boundary_method"},
        )

        # Get runtime parameter overrides
        dx = kwargs.get('dx', self.dx)
        dy = kwargs.get('dy', self.dy)
        dz = kwargs.get('dz', self.dz)
        # The configured choice was checked against the YAML allowed_values
        # of computation.operators.gradient.boundary_method (OperatorBase);
        # the functional kernel owns the implemented closures.
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        
        field = _prepare_gradient_field(
            field,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Call original implementation
        grad_x, grad_y, grad_z = compute_gradient_3d(
            field=field,
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            boundary_method=str(boundary_method)
        )
        
        return grad_x, grad_y, grad_z


# Export all classes
__all__ = [
    'CentralDifferenceGradient1D',
    'CentralDifferenceGradient2D',
    'CentralDifferenceGradient3D'
]
