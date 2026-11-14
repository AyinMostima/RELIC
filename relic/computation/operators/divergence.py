"""
Divergence Operators for RELIC Unified Computation Layer

This module provides divergence operator implementations that wrap the
original function-based operators from relic.computation.operators.functional
with a class-based OperatorBase interface.

Mathematical Expression:
    Divergence: divJ = dJx/dx + dJy/dy + dJz/dz
    
    For a vector field J = (Jx, Jy, Jz), the divergence measures the
    rate of flux expansion/contraction at each point.
    
    Finite Difference Scheme (2nd order):
    dJx/dx ~= (Jx[i+1] - Jx[i-1])/(2dx)

Design Pattern:
    These classes wrap the original compute_divergence_* functions,
    preserving 100% scientific correctness while providing:
    - OperatorBase interface compliance
    - Factory instantiation support
    - ParameterManager integration
    - Metadata and documentation

References:
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM
    - Hirsch, C. (2007). Numerical Computation of Internal and External Flows
"""

from typing import Optional
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
    compute_divergence_2d,
    compute_divergence_3d
)

Tensor = torch.Tensor


@register_operator("divergence.finite_volume_2d", category="spatial")
class FiniteVolumeDivergence2D(OperatorBase):
    """
    2D divergence operator using finite volume method.
    
    Mathematical Expression:
        divJ = dJx/dx + dJy/dy
        dJx/dx ~= (Jx[i+1,j] - Jx[i-1,j])/(2dx)
        dJy/dy ~= (Jy[i,j+1] - Jy[i,j-1])/(2dy)
    
    Parameters (from ParameterManager or config):
        dx : float
            Grid spacing in x-direction [m]
        dy : float
            Grid spacing in y-direction [m]
        boundary_method : str
            Boundary treatment: "zero", "extrapolate", "periodic"
    
    Usage Example:
        >>> from relic.computation.operators import OperatorFactory
        >>> op = OperatorFactory.create(
        ...     "divergence.finite_volume_2d",
        ...     config={"dx": 100.0, "dy": 100.0, "boundary_method": "periodic"}
        ... )
        >>> divergence = op.apply(flux_x=jx, flux_y=jy)
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 7
        - Hirsch, C. (2007). Numerical Computation. Volume 1
    """
    
    operator_type = "divergence"
    
    def __init__(
        self,
        name: str = "divergence_2d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """
        Initialize 2D divergence operator.
        
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
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.divergence",
            validate_parameters=validate_parameters
        )
        
        # Get parameters from ParameterManager or config
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
        
        # Default metadata unless the caller supplied one.
        if metadata is None:
            self.metadata = OperatorMetadata(
                name=self.name,
                spatial_dimensions=2,
                input_units="[flux_units]",
                output_units="[flux_units] m-1",
                operator_type="divergence",
                parameters={
                    "dx": f"{self.dx} m",
                    "dy": f"{self.dy} m",
                    "boundary_method": self.boundary_method
                },
                mathematical_expression="divJ = dJx/dx + dJy/dy",
                description="2D divergence operator using finite volumes",
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM",
                    "Hirsch, C. (2007). Numerical Computation of Internal and External Flows"
                ]
            )
    
    def apply(
        self,
        flux_x: Tensor,
        flux_y: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Apply 2D divergence operator to flux field.
        
        Parameters
        ----------
        flux_x : Tensor
            Flux component in x-direction, shape (..., ny, nx)
            Units: [flux_units] (e.g., kg m^-2 s^-1)
        flux_y : Tensor
            Flux component in y-direction, shape (..., ny, nx)
            Units: [flux_units]
        **kwargs
            Optional runtime parameter overrides (any other keyword
            raises TypeError):
            - dx : float
            - dy : float
            - boundary_method : str
        
        Returns
        -------
        divergence : Tensor
            Divergence divJ, same shape as input
            Units: [flux_units m^-1] (e.g., kg m^-3 s^-1)
        """
        reject_unexpected_runtime_parameters(
            kwargs,
            {"dx", "dy", "boundary_method"},
        )
        # Runtime overrides replace the configured values for this call only.
        # The configured values were read by OperatorBase.get_parameter from
        # the config or the YAML parameter database (units, bounds and
        # allowed_values checked there); an override is checked by the
        # functional kernel (finite positive spacing, implemented closure).
        # An unknown keyword is rejected, never ignored.
        dx = kwargs.get('dx', self.dx)
        dy = kwargs.get('dy', self.dy)
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        
        # Move to correct device/dtype if needed
        if flux_x.device != self.device:
            flux_x = flux_x.to(self.device)
            flux_y = flux_y.to(self.device)
        if flux_x.dtype != self.dtype:
            flux_x = flux_x.to(self.dtype)
            flux_y = flux_y.to(self.dtype)
        
        # Call original implementation
        divergence = compute_divergence_2d(
            flux_x=flux_x,
            flux_y=flux_y,
            dx=float(dx),
            dy=float(dy),
            boundary_method=str(boundary_method)
        )
        
        return divergence


@register_operator("divergence.finite_volume_3d", category="spatial")
class FiniteVolumeDivergence3D(OperatorBase):
    """
    3D divergence operator using finite volume method.
    
    Mathematical Expression:
        divJ = dJx/dx + dJy/dy + dJz/dz
    
    Parameters (from ParameterManager or config):
        dx, dy, dz : float
            Grid spacings in x, y, z directions [m]
        boundary_method : str
            Boundary treatment method
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 7
        - Hirsch, C. (2007). Numerical Computation. Volume 1
    """
    
    operator_type = "divergence"
    
    def __init__(
        self,
        name: str = "divergence_3d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize 3D divergence operator."""
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.divergence",
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
        
        # Default metadata unless the caller supplied one.
        if metadata is None:
            self.metadata = OperatorMetadata(
                name=self.name,
                spatial_dimensions=3,
                input_units="[flux_units]",
                output_units="[flux_units] m-1",
                operator_type="divergence",
                parameters={
                    "dx": f"{self.dx} m",
                    "dy": f"{self.dy} m",
                    "dz": f"{self.dz} m",
                    "boundary_method": self.boundary_method
                },
                mathematical_expression="divJ = dJx/dx + dJy/dy + dJz/dz",
                description="3D divergence operator using finite volumes",
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods",
                    "Hirsch, C. (2007). Numerical Computation"
                ]
            )
    
    def apply(
        self,
        flux_x: Tensor,
        flux_y: Tensor,
        flux_z: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Apply 3D divergence operator to flux field.
        
        Parameters
        ----------
        flux_x : Tensor
            Flux component in x-direction, shape (batch, nz, ny, nx)
            Units: [flux_units]
        flux_y : Tensor
            Flux component in y-direction, shape (batch, nz, ny, nx)
            Units: [flux_units]
        flux_z : Tensor
            Flux component in z-direction, shape (batch, nz, ny, nx)
            Units: [flux_units]
        **kwargs
            Optional runtime overrides of dx, dy, dz and boundary_method;
            any other keyword raises TypeError
        
        Returns
        -------
        divergence : Tensor
            Divergence divJ, same shape as input
            Units: [flux_units m^-1]
        """
        reject_unexpected_runtime_parameters(
            kwargs,
            {"dx", "dy", "dz", "boundary_method"},
        )
        # Runtime overrides replace the configured values for this call only.
        # The configured values were read by OperatorBase.get_parameter from
        # the config or the YAML parameter database (units, bounds and
        # allowed_values checked there); an override is checked by the
        # functional kernel (finite positive spacing, implemented closure).
        # An unknown keyword is rejected, never ignored.
        dx = kwargs.get('dx', self.dx)
        dy = kwargs.get('dy', self.dy)
        dz = kwargs.get('dz', self.dz)
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        
        # Move to correct device/dtype
        if flux_x.device != self.device:
            flux_x = flux_x.to(self.device)
            flux_y = flux_y.to(self.device)
            flux_z = flux_z.to(self.device)
        if flux_x.dtype != self.dtype:
            flux_x = flux_x.to(self.dtype)
            flux_y = flux_y.to(self.dtype)
            flux_z = flux_z.to(self.dtype)
        
        # Call original implementation
        divergence = compute_divergence_3d(
            flux_x=flux_x,
            flux_y=flux_y,
            flux_z=flux_z,
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            boundary_method=str(boundary_method)
        )
        
        return divergence


# Export all classes
__all__ = [
    'FiniteVolumeDivergence2D',
    'FiniteVolumeDivergence3D'
]
