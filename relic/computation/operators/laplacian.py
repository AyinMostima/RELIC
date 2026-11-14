"""
Laplacian Operators for RELIC Unified Computation Layer

This module provides Laplacian operator implementations that wrap the
original function-based operators from relic.computation.operators.functional
with a class-based OperatorBase interface.

Mathematical Expression:
    Laplacian: laplacianphi = d2phi/dx2 + d2phi/dy2 + d2phi/dz2
    
    Finite Difference Scheme (2nd order):
    d2phi/dx2 ~= (phi[i+1] - 2phi[i] + phi[i-1])/dx2

Design Pattern:
    These classes wrap the original compute_laplacian_* functions,
    preserving 100% scientific correctness while providing:
    - OperatorBase interface compliance
    - Factory instantiation support
    - ParameterManager integration
    - Metadata and documentation

References:
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM
    - Strikwerda, J.C. (2004). Finite Difference Schemes and PDEs. SIAM
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
    compute_laplacian_1d,
    compute_laplacian_2d,
    compute_laplacian_3d
)

Tensor = torch.Tensor


@register_operator("laplacian.finite_difference_1d", category="spatial")
class FiniteDifferenceLaplacian1D(OperatorBase):
    """
    1D Laplacian operator using finite differences.
    
    Mathematical Expression:
        laplacianphi = d2phi/dx2 ~= (phi[i+1] - 2phi[i] + phi[i-1])/dx2
    
    Parameters (from ParameterManager or config):
        dx : float
            Grid spacing in x-direction [m]
        boundary_method : str
            Boundary treatment: "zero", "extrapolate", "periodic"
    
    Usage Example:
        >>> from relic.computation.operators import OperatorFactory
        >>> op = OperatorFactory.create(
        ...     "laplacian.finite_difference_1d",
        ...     config={"dx": 100.0, "boundary_method": "periodic"}
        ... )
        >>> laplacian = op.apply(field)
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 2
    """
    
    operator_type = "laplacian"
    
    def __init__(
        self,
        name: str = "laplacian_1d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """
        Initialize 1D Laplacian operator.
        
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
            namespace=namespace or "computation.operators.laplacian",
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
        
        # Default metadata unless the caller supplied one.
        if metadata is None:
            self.metadata = OperatorMetadata(
                name=self.name,
                spatial_dimensions=1,
                input_units="[field_units]",
                output_units="[field_units] m-2",
                operator_type="laplacian",
                parameters={
                    "dx": f"{self.dx} m",
                    "boundary_method": self.boundary_method
                },
                mathematical_expression="laplacianphi = d2phi/dx2 ~= (phi[i+1] - 2phi[i] + phi[i-1])/dx2",
                description="1D Laplacian operator using finite differences",
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM"
                ]
            )
    
    def apply(self, field: Tensor, **kwargs) -> Tensor:
        """
        Apply 1D Laplacian operator to field.
        
        Parameters
        ----------
        field : Tensor
            Input field, shape (..., nx)
            Units: [field_units]
        **kwargs
            Optional runtime parameter overrides (any other keyword
            raises TypeError):
            - dx : float
            - boundary_method : str
        
        Returns
        -------
        laplacian : Tensor
            Second derivative d2phi/dx2, same shape as input
            Units: [field_units m^-2]
        """
        reject_unexpected_runtime_parameters(
            kwargs,
            {"dx", "boundary_method"},
        )
        # Runtime overrides replace the configured values for this call only.
        # The configured values were read by OperatorBase.get_parameter from
        # the config or the YAML parameter database (units, bounds and
        # allowed_values checked there); an override is checked by the
        # functional kernel (finite positive spacing, implemented closure).
        # An unknown keyword is rejected, never ignored.
        dx = kwargs.get('dx', self.dx)
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        
        # Move to correct device/dtype if needed
        if field.device != self.device:
            field = field.to(self.device)
        if field.dtype != self.dtype:
            field = field.to(self.dtype)
        
        # Call original implementation
        laplacian = compute_laplacian_1d(
            field=field,
            dx=float(dx),
            boundary_method=str(boundary_method)
        )
        
        return laplacian


@register_operator("laplacian.finite_difference_2d", category="spatial")
class FiniteDifferenceLaplacian2D(OperatorBase):
    """
    2D Laplacian operator using finite differences.
    
    Mathematical Expression:
        laplacianphi = d2phi/dx2 + d2phi/dy2
        d2phi/dx2 ~= (phi[i+1,j] - 2phi[i,j] + phi[i-1,j])/dx2
        d2phi/dy2 ~= (phi[i,j+1] - 2phi[i,j] + phi[i,j-1])/dy2
    
    Parameters (from ParameterManager or config):
        dx : float
            Grid spacing in x-direction [m]
        dy : float
            Grid spacing in y-direction [m]
        boundary_method : str
            Boundary treatment method
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 3
    """
    
    operator_type = "laplacian"
    
    def __init__(
        self,
        name: str = "laplacian_2d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize 2D Laplacian operator."""
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.laplacian",
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
        
        # Default metadata unless the caller supplied one.
        if metadata is None:
            self.metadata = OperatorMetadata(
                name=self.name,
                spatial_dimensions=2,
                input_units="[field_units]",
                output_units="[field_units] m-2",
                operator_type="laplacian",
                parameters={
                    "dx": f"{self.dx} m",
                    "dy": f"{self.dy} m",
                    "boundary_method": self.boundary_method
                },
                mathematical_expression="laplacianphi = d2phi/dx2 + d2phi/dy2",
                description="2D Laplacian operator using finite differences",
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods"
                ]
            )
    
    def apply(self, field: Tensor, **kwargs) -> Tensor:
        """
        Apply 2D Laplacian operator to field.
        
        Parameters
        ----------
        field : Tensor
            Input field, shape (..., ny, nx)
            Units: [field_units]
        **kwargs
            Optional runtime overrides of the grid spacings and
            boundary_method; any other keyword raises TypeError
        
        Returns
        -------
        laplacian : Tensor
            2D Laplacian laplacianphi, same shape as input
            Units: [field_units m^-2]
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
        
        # Move to correct device/dtype
        if field.device != self.device:
            field = field.to(self.device)
        if field.dtype != self.dtype:
            field = field.to(self.dtype)
        
        # Call original implementation
        laplacian = compute_laplacian_2d(
            field=field,
            dx=float(dx),
            dy=float(dy),
            boundary_method=str(boundary_method)
        )
        
        return laplacian


@register_operator("laplacian.finite_difference_3d", category="spatial")
class FiniteDifferenceLaplacian3D(OperatorBase):
    """
    3D Laplacian operator using finite differences.
    
    Mathematical Expression:
        laplacianphi = d2phi/dx2 + d2phi/dy2 + d2phi/dz2
    
    Parameters (from ParameterManager or config):
        dx, dy, dz : float
            Grid spacings in x, y, z directions [m]
        boundary_method : str
            Boundary treatment method
    
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. Chapter 4
    """
    
    operator_type = "laplacian"
    
    def __init__(
        self,
        name: str = "laplacian_3d",
        config: Optional[dict] = None,
        metadata: Optional[OperatorMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize 3D Laplacian operator."""
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=resolve_operator_dtype(dtype),
            namespace=namespace or "computation.operators.laplacian",
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
                input_units="[field_units]",
                output_units="[field_units] m-2",
                operator_type="laplacian",
                parameters={
                    "dx": f"{self.dx} m",
                    "dy": f"{self.dy} m",
                    "dz": f"{self.dz} m",
                    "boundary_method": self.boundary_method
                },
                mathematical_expression="laplacianphi = d2phi/dx2 + d2phi/dy2 + d2phi/dz2",
                description="3D Laplacian operator using finite differences",
                references=[
                    "LeVeque, R.J. (2007). Finite Difference Methods"
                ]
            )
    
    def apply(self, field: Tensor, **kwargs) -> Tensor:
        """
        Apply 3D Laplacian operator to field.
        
        Parameters
        ----------
        field : Tensor
            Input field, shape (batch, nz, ny, nx)
            Units: [field_units]
        **kwargs
            Optional runtime overrides of the grid spacings and
            boundary_method; any other keyword raises TypeError
        
        Returns
        -------
        laplacian : Tensor
            3D Laplacian laplacianphi, same shape as input
            Units: [field_units m^-2]
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
        if field.device != self.device:
            field = field.to(self.device)
        if field.dtype != self.dtype:
            field = field.to(self.dtype)
        
        # Call original implementation
        laplacian = compute_laplacian_3d(
            field=field,
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            boundary_method=str(boundary_method)
        )
        
        return laplacian


# Export all classes
__all__ = [
    'FiniteDifferenceLaplacian1D',
    'FiniteDifferenceLaplacian2D',
    'FiniteDifferenceLaplacian3D'
]
