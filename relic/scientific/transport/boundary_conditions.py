"""
Boundary conditions for RELIC transport process operators.

This module provides unified boundary condition implementations for transport
equations including Dirichlet, Neumann, Robin, and periodic boundaries.

Mathematical expressions:
    Dirichlet: phi|partialOmega = g(x, t)
    Neumann: dphi/dn|partialOmega = h(x, t)
    Robin: alpha * phi + beta * dphi/dn|partialOmega = gamma(x, t)
    Periodic: phi(x + L) = phi(x)

Discrete forms:
    Dirichlet: phi[boundary] = g
    Neumann: (phi[i] - phi[i-1]) / Deltax = h
    Robin: alpha * phi[i] + beta * (phi[i] - phi[i-1]) / Deltax = gamma

References:
    - Carslaw, H. S. and Jaeger, J. C. (1959). Conduction of Heat in Solids.
    - Ozisik, M. N. (1993). Heat Conduction.
    - LeVeque, R. J. (2007). Finite Difference Methods for ODEs and PDEs.
    - Strikwerda, J. C. (2004). Finite Difference Schemes and PDEs.
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import torch
from relic.core import device_policy
from enum import Enum
from ...core.managers.base import BaseManager, DeviceAwareManagerMixin
from ...core.managers.metadata import ManagerMetadata, ManagerCapability, ManagerResource, CapabilityType, ResourceType
from relic.parameters import get_global_manager
logger = logging.getLogger(__name__)
Tensor = torch.Tensor
BoundaryValue = Union[float, Tensor, callable]

class BoundaryType(Enum):
    """Enumeration of boundary condition types."""
    DIRICHLET = 'dirichlet'
    NEUMANN = 'neumann'
    ROBIN = 'robin'
    PERIODIC = 'periodic'
    WALL = 'wall'
    OPEN = 'open'

class BoundaryCondition:
    """
    Single boundary condition specification.
    
    This class represents one boundary condition applied to a specific
    face or region of the computational domain.
    """

    def __init__(self, bc_type: Union[BoundaryType, str], value: BoundaryValue=0.0, face: str='x_min', alpha: float=1.0, beta: float=0.0, gamma: BoundaryValue=0.0, time_dependent: bool=False) -> None:
        """
        Initialize boundary condition.
        
        Parameters:
            bc_type : Union[BoundaryType, str]
                Type of boundary condition
            value : BoundaryValue
                Boundary value (for Dirichlet) or flux (for Neumann)
                Units: [field_units] for Dirichlet, [field_units m-1] for Neumann
            face : str
                Boundary face identifier: "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"
            alpha, beta : float
                Robin condition coefficients: alpha*phi + beta*partialphi/partialn = gamma
                Units: [dimensionless] for alpha, [m-1] for beta
            gamma : BoundaryValue
                Robin condition RHS value
                Units: [field_units]
            time_dependent : bool
                Whether boundary condition varies with time
        """
        if isinstance(bc_type, str):
            self.bc_type = BoundaryType(bc_type.lower())
        else:
            self.bc_type = bc_type
        self.value = value
        self.face = face.lower()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.time_dependent = time_dependent
        valid_faces = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
        if self.face not in valid_faces:
            raise ValueError(f"Invalid face '{face}'. Must be one of {valid_faces}")

    def get_value(self, time: float=0.0, coordinates: Optional[Tensor]=None) -> Tensor:
        """
        Get boundary condition value at specified time and coordinates.
        
        Parameters:
            time : float
                Current time [s]
            coordinates : Optional[Tensor]
                Spatial coordinates at boundary, shape (..., spatial_dims)
                Units: [m]
                
        Returns:
            boundary_value : Tensor
                Boundary condition value
                Units: [field_units] or [field_units m-1]
        """
        if callable(self.value):
            if coordinates is not None:
                return self.value(time, coordinates)
            else:
                return self.value(time)
        elif isinstance(self.value, torch.Tensor):
            return self.value
        elif coordinates is not None:
            return torch.full_like(coordinates[..., 0], self.value)
        else:
            return torch.tensor(self.value)

    def get_robin_coefficients(self, time: float=0.0) -> Tuple[float, float, Tensor]:
        """
        Get Robin boundary condition coefficients.
        
        Returns:
            alpha, beta, gamma : Tuple[float, float, Tensor]
                Robin coefficients: alpha*phi + beta*partialphi/partialn = gamma
        """
        if callable(self.gamma):
            gamma_val = self.gamma(time)
        elif isinstance(self.gamma, torch.Tensor):
            gamma_val = self.gamma
        else:
            gamma_val = torch.tensor(self.gamma)
        return (self.alpha, self.beta, gamma_val)

class BoundaryConditionManager(DeviceAwareManagerMixin, BaseManager):
    """
    Manager for multiple boundary conditions applied to transport equations.
    
    This class handles the application of boundary conditions to field tensors
    during transport calculations, ensuring consistent treatment across all
    spatial dimensions and boundary types.
    
    Now inherits from BaseManager for unified lifecycle management.
    """

    def __init__(self, spatial_dims: int=2, device: Optional[torch.device]=None):
        """
        Initialize boundary condition manager.
        
        Parameters:
            spatial_dims : int
                Number of spatial dimensions (1, 2, or 3)
            device : Optional[torch.device]
                Computation device
        """
        self._init_spatial_dims = spatial_dims
        self._init_device = device or (
            torch.device("cpu")
        )
        self._default_boundary_type: BoundaryType = BoundaryType.NEUMANN
        self._default_boundary_value: float = 0.0
        metadata = self._create_metadata()
        super().__init__(device=self._init_device, metadata=metadata)

    def _create_metadata(self) -> ManagerMetadata:
        """Create metadata for BoundaryConditionManager."""
        capabilities = [ManagerCapability(type=CapabilityType.SPATIAL_OPERATIONS, level='standard', description='Boundary condition application for transport equations'), ManagerCapability(type=CapabilityType.CONSTRAINT_ENFORCEMENT, level='standard', description='Physical boundary constraints'), ManagerCapability(type=CapabilityType.DEVICE_MANAGEMENT, level='standard', description='PyTorch device awareness')]
        resources = [ManagerResource(type=ResourceType.SYSTEM_MEMORY, quantity=None, unit='bytes')]
        dependencies = []
        return ManagerMetadata(name='BoundaryConditionManager', version='2.0.0', description='Unified boundary condition manager for transport processes', capabilities=capabilities, resources=resources, dependencies=dependencies, tags={'thread_safe', 'gpu_enabled'}, custom_attributes={'thread_safe': True, 'supports_gpu': True, 'supports_distributed': False, 'mathematical_expression': 'Dirichlet: phi|boundary = g; Neumann: dphi/dn|boundary = h; Robin: alpha*phi + beta*dphi/dn = gamma', 'references': ['Carslaw, H.S., Jaeger, J.C. (1959). Conduction of Heat in Solids.', 'Ozisik, M.N. (1993). Heat Conduction.', 'LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs.', 'Strikwerda, J.C. (2004). Finite Difference Schemes and PDEs.'], 'conservation_laws': [], 'parameters': {}})

    def _load_default_boundary_parameters(self) -> None:
        """Load default boundary condition parameters from ParameterManager."""
        type_value = get_global_manager().get(
            "transport.numerical.default_boundary_type",
            required=True,
            expected_type=None,
        )
        value_value = get_global_manager().get(
            "transport.numerical.default_boundary_value",
            required=True,
            expected_type=None,
        )
        if isinstance(type_value, BoundaryType):
            resolved_type = type_value
        elif isinstance(type_value, str):
            try:
                resolved_type = BoundaryType(type_value.lower())
            except ValueError:
                raise ValueError(
                    f"Unsupported transport.numerical.default_boundary_type: {type_value}"
                ) from None
        else:
            raise TypeError(
                f"Boundary type {type_value!r} is not recognized."
            )
        if isinstance(value_value, torch.Tensor):
            resolved_value = float(value_value.detach())
        elif isinstance(value_value, (int, float)):
            resolved_value = float(value_value)
        else:
            raise TypeError(f"Boundary value {value_value!r} is not numeric.")
        self._default_boundary_type = resolved_type
        self._default_boundary_value = resolved_value

    def _do_initialize(self) -> None:
        """Perform BoundaryConditionManager-specific initialization."""
        self.spatial_dims = self._init_spatial_dims
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        self._load_default_boundary_parameters()
        self._set_default_boundaries()
        logger.info(f'BoundaryConditionManager initialized: spatial_dims={self.spatial_dims}, device={self.device}')

    def _do_configure(self, config: Dict[str, Any]) -> None:
        """Configure BoundaryConditionManager with runtime settings."""
        if 'device' in config:
            device_value = config.get('device')
            if device_value is not None:
                self.to_device(device_value)
        if 'spatial_dims' in config:
            spatial_dims = config.get('spatial_dims')
            if spatial_dims is not None:
                self.spatial_dims = int(spatial_dims)
                self.boundary_conditions.clear()
                self._load_default_boundary_parameters()
                self._set_default_boundaries()
        if 'boundary_conditions' in config:
            boundary_configs = config.get('boundary_conditions') or {}
            for face, bc_config in boundary_configs.items():
                bc_type = bc_config.get('type')
                if bc_type is None:
                    bc_type = self._default_boundary_type.value
                boundary_value = bc_config.get('value')
                if boundary_value is None:
                    boundary_value = self._default_boundary_value
                params = bc_config.get('params') or {}
                self.set_boundary_condition(face=face, bc_type=bc_type, value=boundary_value, **params)
        logger.debug(f'BoundaryConditionManager configured with: {config}')

    def _do_validate(self) -> List[str]:
        """Validate BoundaryConditionManager configuration and state.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        if self.spatial_dims not in [1, 2, 3]:
            errors.append(f'Invalid spatial_dims {self.spatial_dims}. Must be 1, 2, or 3')
        if self.device.type == 'cuda' and (not device_policy.is_cuda_available()):
            errors.append(f'CUDA device {self.device} requested but not available')
        required_faces = []
        if self.spatial_dims >= 1:
            required_faces.extend(['x_min', 'x_max'])
        if self.spatial_dims >= 2:
            required_faces.extend(['y_min', 'y_max'])
        if self.spatial_dims >= 3:
            required_faces.extend(['z_min', 'z_max'])
        for face in required_faces:
            if face not in self.boundary_conditions:
                errors.append(f"No boundary condition set for required face '{face}'")
        for dim in ['x', 'y', 'z']:
            if f'{dim}_min' in self.boundary_conditions and f'{dim}_max' in self.boundary_conditions:
                bc_min = self.boundary_conditions[f'{dim}_min']
                bc_max = self.boundary_conditions[f'{dim}_max']
                if bc_min.bc_type == BoundaryType.PERIODIC and bc_max.bc_type != BoundaryType.PERIODIC:
                    errors.append(f'Inconsistent periodic BC in {dim}-direction')
                elif bc_max.bc_type == BoundaryType.PERIODIC and bc_min.bc_type != BoundaryType.PERIODIC:
                    errors.append(f'Inconsistent periodic BC in {dim}-direction')
        if not errors:
            logger.debug('BoundaryConditionManager validation completed successfully')
        else:
            logger.warning(f'BoundaryConditionManager validation found {len(errors)} issues')
        return errors

    def _do_cleanup(self) -> None:
        """Clean up BoundaryConditionManager resources."""
        self.boundary_conditions.clear()
        logger.info('BoundaryConditionManager cleanup completed')

    def _on_device_changed(
        self,
        _old_device: Optional[torch.device],
        new_device: Optional[torch.device],
    ) -> None:
        """Migrate tensor-valued boundary data owned by this manager."""
        if new_device is None or not hasattr(self, 'boundary_conditions'):
            return
        for boundary_condition in self.boundary_conditions.values():
            if isinstance(boundary_condition.value, torch.Tensor):
                boundary_condition.value = boundary_condition.value.to(new_device)
            if isinstance(boundary_condition.gamma, torch.Tensor):
                boundary_condition.gamma = boundary_condition.gamma.to(new_device)

    def _set_default_boundaries(self) -> None:
        """Set default boundary conditions using ParameterManager defaults."""
        if self.spatial_dims >= 1:
            self.boundary_conditions['x_min'] = BoundaryCondition(self._default_boundary_type, self._default_boundary_value, 'x_min')
            self.boundary_conditions['x_max'] = BoundaryCondition(self._default_boundary_type, self._default_boundary_value, 'x_max')
        if self.spatial_dims >= 2:
            self.boundary_conditions['y_min'] = BoundaryCondition(self._default_boundary_type, self._default_boundary_value, 'y_min')
            self.boundary_conditions['y_max'] = BoundaryCondition(self._default_boundary_type, self._default_boundary_value, 'y_max')
        if self.spatial_dims >= 3:
            self.boundary_conditions['z_min'] = BoundaryCondition(self._default_boundary_type, self._default_boundary_value, 'z_min')
            self.boundary_conditions['z_max'] = BoundaryCondition(self._default_boundary_type, self._default_boundary_value, 'z_max')

    def set_boundary_condition(self, face: str, bc_type: Union[BoundaryType, str], value: BoundaryValue=0.0, **kwargs) -> None:
        """
        Set boundary condition for a specific face.
        
        Parameters:
            face : str
                Boundary face identifier
            bc_type : Union[BoundaryType, str]
                Type of boundary condition
            value : BoundaryValue
                Boundary value or flux
                For Robin conditions: can be tuple (alpha, beta, gamma)
            **kwargs : Additional parameters for Robin conditions
        """
        if isinstance(bc_type, str):
            bc_type_enum = BoundaryType(bc_type.lower())
        else:
            bc_type_enum = bc_type
        if bc_type_enum == BoundaryType.ROBIN and isinstance(value, tuple):
            alpha, beta, gamma = value
            if isinstance(gamma, torch.Tensor) and self.device is not None:
                gamma = gamma.to(self.device)
            bc = BoundaryCondition(bc_type, 0.0, face, alpha=alpha, beta=beta, gamma=gamma, **kwargs)
        else:
            if isinstance(value, torch.Tensor) and self.device is not None:
                value = value.to(self.device)
            bc = BoundaryCondition(bc_type, value, face, **kwargs)
        self.boundary_conditions[face.lower()] = bc

    def apply_boundary_conditions(self, field: Tensor, grid_spacing: Union[float, Tuple[float, ...]], time: float=0.0) -> Tensor:
        """
        Apply all boundary conditions to a field tensor.
        
        This is the main interface for applying boundary conditions during
        transport calculations.
        
        Mathematical Expression:
            For each boundary face, apply the appropriate discretization
            of the boundary condition to update ghost cells or boundary values.
        
        Parameters:
            field : Tensor
                Field tensor to apply boundary conditions to
                Shape: (batch, nx) for 1D, (batch, ny, nx) for 2D, (batch, nz, ny, nx) for 3D
                Units: [field_units]
            grid_spacing : Union[float, Tuple[float, ...]]
                Grid spacing(s) [m]
            time : float
                Current time [s]
                
        Returns:
            field_bc : Tensor
                Field with boundary conditions applied
                Same shape and units as input field
                
        Raises:
            ValueError: If field dimensions don't match spatial_dims
            RuntimeError: If boundary condition cannot be applied
        """
        expected_dims = self.spatial_dims + 1
        if field.dim() != expected_dims:
            raise ValueError(f'Field must have {expected_dims} dimensions for {self.spatial_dims}D')
        field_bc = field.clone()
        if isinstance(grid_spacing, (int, float)):
            spacing = [float(grid_spacing)] * self.spatial_dims
        else:
            spacing = list(grid_spacing)
            if len(spacing) != self.spatial_dims:
                raise ValueError(f'Grid spacing must have {self.spatial_dims} components')
        for face_name, bc in self.boundary_conditions.items():
            if face_name.startswith('x'):
                field_bc = self._apply_x_boundary(field_bc, bc, spacing[0], time)
            elif face_name.startswith('y') and self.spatial_dims >= 2:
                field_bc = self._apply_y_boundary(field_bc, bc, spacing[1], time)
            elif face_name.startswith('z') and self.spatial_dims >= 3:
                field_bc = self._apply_z_boundary(field_bc, bc, spacing[2], time)
        return field_bc

    def _apply_x_boundary(self, field: Tensor, bc: BoundaryCondition, dx: float, time: float) -> Tensor:
        """Apply boundary condition on x-direction faces."""
        if bc.face == 'x_min':
            if bc.bc_type == BoundaryType.DIRICHLET:
                bc_value = bc.get_value(time)
                field[..., 0] = bc_value
            elif bc.bc_type == BoundaryType.NEUMANN:
                flux = bc.get_value(time)
                field[..., 0] = field[..., 1] - flux * dx
            elif bc.bc_type == BoundaryType.ROBIN:
                alpha, beta, gamma = bc.get_robin_coefficients(time)
                field[..., 0] = (gamma + beta * field[..., 1] / dx) / (alpha + beta / dx)
            elif bc.bc_type == BoundaryType.PERIODIC:
                field[..., 0] = field[..., -1]
            elif bc.bc_type == BoundaryType.WALL:
                field[..., 0] = 0.0
            elif bc.bc_type == BoundaryType.OPEN:
                field[..., 0] = field[..., 1]
        elif bc.face == 'x_max':
            if bc.bc_type == BoundaryType.DIRICHLET:
                bc_value = bc.get_value(time)
                field[..., -1] = bc_value
            elif bc.bc_type == BoundaryType.NEUMANN:
                flux = bc.get_value(time)
                field[..., -1] = field[..., -2] + flux * dx
            elif bc.bc_type == BoundaryType.ROBIN:
                alpha, beta, gamma = bc.get_robin_coefficients(time)
                field[..., -1] = (gamma + beta * field[..., -2] / dx) / (alpha + beta / dx)
            elif bc.bc_type == BoundaryType.PERIODIC:
                field[..., -1] = field[..., 0]
            elif bc.bc_type == BoundaryType.WALL:
                field[..., -1] = 0.0
            elif bc.bc_type == BoundaryType.OPEN:
                field[..., -1] = field[..., -2]
        return field

    def _apply_y_boundary(self, field: Tensor, bc: BoundaryCondition, dy: float, time: float) -> Tensor:
        """Apply boundary condition on y-direction faces."""
        if bc.face == 'y_min':
            if bc.bc_type == BoundaryType.DIRICHLET:
                bc_value = bc.get_value(time)
                field[..., 0, :] = bc_value
            elif bc.bc_type == BoundaryType.NEUMANN:
                flux = bc.get_value(time)
                field[..., 0, :] = field[..., 1, :] - flux * dy
            elif bc.bc_type == BoundaryType.ROBIN:
                alpha, beta, gamma = bc.get_robin_coefficients(time)
                field[..., 0, :] = (gamma + beta * field[..., 1, :] / dy) / (alpha + beta / dy)
            elif bc.bc_type == BoundaryType.PERIODIC:
                field[..., 0, :] = field[..., -1, :]
            elif bc.bc_type == BoundaryType.WALL:
                field[..., 0, :] = 0.0
            elif bc.bc_type == BoundaryType.OPEN:
                field[..., 0, :] = field[..., 1, :]
        elif bc.face == 'y_max':
            if bc.bc_type == BoundaryType.DIRICHLET:
                bc_value = bc.get_value(time)
                field[..., -1, :] = bc_value
            elif bc.bc_type == BoundaryType.NEUMANN:
                flux = bc.get_value(time)
                field[..., -1, :] = field[..., -2, :] + flux * dy
            elif bc.bc_type == BoundaryType.ROBIN:
                alpha, beta, gamma = bc.get_robin_coefficients(time)
                field[..., -1, :] = (gamma + beta * field[..., -2, :] / dy) / (alpha + beta / dy)
            elif bc.bc_type == BoundaryType.PERIODIC:
                field[..., -1, :] = field[..., 0, :]
            elif bc.bc_type == BoundaryType.WALL:
                field[..., -1, :] = 0.0
            elif bc.bc_type == BoundaryType.OPEN:
                field[..., -1, :] = field[..., -2, :]
        return field

    def _apply_z_boundary(self, field: Tensor, bc: BoundaryCondition, dz: float, time: float) -> Tensor:
        """Apply boundary condition on z-direction faces."""
        if bc.face == 'z_min':
            if bc.bc_type == BoundaryType.DIRICHLET:
                bc_value = bc.get_value(time)
                field[..., 0, :, :] = bc_value
            elif bc.bc_type == BoundaryType.NEUMANN:
                flux = bc.get_value(time)
                field[..., 0, :, :] = field[..., 1, :, :] - flux * dz
            elif bc.bc_type == BoundaryType.ROBIN:
                alpha, beta, gamma = bc.get_robin_coefficients(time)
                field[..., 0, :, :] = (gamma + beta * field[..., 1, :, :] / dz) / (alpha + beta / dz)
            elif bc.bc_type == BoundaryType.PERIODIC:
                field[..., 0, :, :] = field[..., -1, :, :]
            elif bc.bc_type == BoundaryType.WALL:
                field[..., 0, :, :] = 0.0
            elif bc.bc_type == BoundaryType.OPEN:
                field[..., 0, :, :] = field[..., 1, :, :]
        elif bc.face == 'z_max':
            if bc.bc_type == BoundaryType.DIRICHLET:
                bc_value = bc.get_value(time)
                field[..., -1, :, :] = bc_value
            elif bc.bc_type == BoundaryType.NEUMANN:
                flux = bc.get_value(time)
                field[..., -1, :, :] = field[..., -2, :, :] + flux * dz
            elif bc.bc_type == BoundaryType.ROBIN:
                alpha, beta, gamma = bc.get_robin_coefficients(time)
                field[..., -1, :, :] = (gamma + beta * field[..., -2, :, :] / dz) / (alpha + beta / dz)
            elif bc.bc_type == BoundaryType.PERIODIC:
                field[..., -1, :, :] = field[..., 0, :, :]
            elif bc.bc_type == BoundaryType.WALL:
                field[..., -1, :, :] = 0.0
            elif bc.bc_type == BoundaryType.OPEN:
                field[..., -1, :, :] = field[..., -2, :, :]
        return field

    def get_boundary_flux(self, field: Tensor, face: str, grid_spacing: Union[float, Tuple[float, ...]], diffusivity: Union[float, Tensor], time: float=0.0) -> Tensor:
        """Return the outward-positive diffusive flux through one boundary face.

        References:
          - Patankar (1980), Numerical Heat Transfer and Fluid Flow.
          - LeVeque (2007), one-sided boundary differences for finite differences.
        Mathematical expression:
          J_out = -D * dphi/dn with n the outward unit normal of the face;
          dphi/dn = -dphi/ds at a min face and +dphi/ds at a max face, where s is
          the face-normal axis coordinate. Non-Neumann faces evaluate the axis
          gradient from the supplied field by the one-sided difference
          dphi/ds = (phi[1]-phi[0])/h (min face) or (phi[-1]-phi[-2])/h (max face);
          a Neumann face uses its prescribed axis gradient g_s = dphi/ds directly.
        Parameter meanings:
          field is the field with boundary values already applied; D is the
          diffusivity; h is the physical cell width along the face-normal axis;
          g_s is the Neumann datum stored on the face descriptor.
        Units:
          field [state_unit]; h [m]; D [m2 s-1]; g_s [state_unit m-1];
          J_out [state_unit m s-1].
        Domain/constraints:
          diffusivity is required because a gradient alone is not a flux;
          grid_spacing follows the manager's physical (x, y, z) order and must
          cover the face axis; the face must be configured; the one-sided
          difference is first order on the cell-center spacing h and is not the
          half-cell face law of the finite-volume transport operator.
        Sign convention:
          positive J_out exports the quantity through the face, negative J_out
          imports it; the same convention holds for every face.
        """
        face_lower = face.lower()
        if face_lower not in self.boundary_conditions:
            raise ValueError(f"No boundary condition set for face '{face}'")
        if diffusivity is None:
            raise ValueError(
                "get_boundary_flux requires an explicit diffusivity; a gradient alone is not a flux"
            )
        bc = self.boundary_conditions[face_lower]
        if isinstance(grid_spacing, (int, float)):
            spacing = [float(grid_spacing)] * self.spatial_dims
        else:
            spacing = [float(value) for value in grid_spacing]
        axis_index = {'x': 0, 'y': 1, 'z': 2}[face_lower[0]]
        if axis_index >= len(spacing):
            raise ValueError(
                f"grid_spacing provides {len(spacing)} physical widths; face '{face}' needs axis index {axis_index}"
            )
        width = spacing[axis_index]
        if width <= 0.0:
            raise ValueError("grid_spacing widths must be positive")
        is_min_face = face_lower.endswith('_min')
        # Storage order is (..., z, y, x): the axis offset from the last dimension
        # is 0 for x, 1 for y and 2 for z.
        storage_axis = field.dim() - 1 - axis_index
        if bc.bc_type == BoundaryType.NEUMANN:
            axis_gradient = torch.as_tensor(
                bc.get_value(time), device=field.device, dtype=field.dtype
            )
        else:
            if field.shape[storage_axis] < 2:
                raise ValueError("One-sided boundary gradients require at least two cells along the face axis")
            if is_min_face:
                axis_gradient = (
                    field.select(storage_axis, 1) - field.select(storage_axis, 0)
                ) / width
            else:
                axis_gradient = (
                    field.select(storage_axis, -1) - field.select(storage_axis, -2)
                ) / width
        outward_sign = -1.0 if is_min_face else 1.0
        diffusivity_tensor = torch.as_tensor(diffusivity, device=field.device, dtype=field.dtype)
        return -diffusivity_tensor * (outward_sign * axis_gradient)

    def validate_boundary_conditions(self) -> List[str]:
        """
        Validate all boundary conditions for consistency and completeness.
        
        Returns:
            warnings : List[str]
                List of validation warnings
        """
        return self._do_validate()

def create_dirichlet_walls(spatial_dims: int, wall_value: float=0.0, device: Optional[torch.device]=None) -> BoundaryConditionManager:
    """
    Create boundary condition manager with Dirichlet walls (zero concentration).
    
    Parameters:
        spatial_dims : int
            Number of spatial dimensions
        wall_value : float
            Value at walls [field_units]
        device : Optional[torch.device]
            Computation device
            
    Returns:
        bc_manager : BoundaryConditionManager
            Configured boundary condition manager
    """
    bc_manager = BoundaryConditionManager(spatial_dims, device)
    bc_manager.initialize()
    if spatial_dims >= 1:
        bc_manager.set_boundary_condition('x_min', BoundaryType.DIRICHLET, wall_value)
        bc_manager.set_boundary_condition('x_max', BoundaryType.DIRICHLET, wall_value)
    if spatial_dims >= 2:
        bc_manager.set_boundary_condition('y_min', BoundaryType.DIRICHLET, wall_value)
        bc_manager.set_boundary_condition('y_max', BoundaryType.DIRICHLET, wall_value)
    if spatial_dims >= 3:
        bc_manager.set_boundary_condition('z_min', BoundaryType.DIRICHLET, wall_value)
        bc_manager.set_boundary_condition('z_max', BoundaryType.DIRICHLET, wall_value)
    bc_manager.validate()
    return bc_manager

def create_periodic_boundaries(spatial_dims: int, device: Optional[torch.device]=None) -> BoundaryConditionManager:
    """
    Create boundary condition manager with periodic boundaries in all directions.
    
    Parameters:
        spatial_dims : int
            Number of spatial dimensions
        device : Optional[torch.device]
            Computation device
            
    Returns:
        bc_manager : BoundaryConditionManager
            Configured boundary condition manager
    """
    bc_manager = BoundaryConditionManager(spatial_dims, device)
    bc_manager.initialize()
    if spatial_dims >= 1:
        bc_manager.set_boundary_condition('x_min', BoundaryType.PERIODIC)
        bc_manager.set_boundary_condition('x_max', BoundaryType.PERIODIC)
    if spatial_dims >= 2:
        bc_manager.set_boundary_condition('y_min', BoundaryType.PERIODIC)
        bc_manager.set_boundary_condition('y_max', BoundaryType.PERIODIC)
    if spatial_dims >= 3:
        bc_manager.set_boundary_condition('z_min', BoundaryType.PERIODIC)
        bc_manager.set_boundary_condition('z_max', BoundaryType.PERIODIC)
    bc_manager.validate()
    return bc_manager

def create_open_boundaries(spatial_dims: int, device: Optional[torch.device]=None) -> BoundaryConditionManager:
    """
    Create boundary condition manager with open boundaries (zero gradient).
    
    Parameters:
        spatial_dims : int
            Number of spatial dimensions
        device : Optional[torch.device]
            Computation device
            
    Returns:
        bc_manager : BoundaryConditionManager
            Configured boundary condition manager
    """
    bc_manager = BoundaryConditionManager(spatial_dims, device)
    bc_manager.initialize()
    if spatial_dims >= 1:
        bc_manager.set_boundary_condition('x_min', BoundaryType.OPEN)
        bc_manager.set_boundary_condition('x_max', BoundaryType.OPEN)
    if spatial_dims >= 2:
        bc_manager.set_boundary_condition('y_min', BoundaryType.OPEN)
        bc_manager.set_boundary_condition('y_max', BoundaryType.OPEN)
    if spatial_dims >= 3:
        bc_manager.set_boundary_condition('z_min', BoundaryType.OPEN)
        bc_manager.set_boundary_condition('z_max', BoundaryType.OPEN)
    bc_manager.validate()
    return bc_manager
__all__ = ['BoundaryType', 'BoundaryCondition', 'BoundaryConditionManager', 'create_dirichlet_walls', 'create_periodic_boundaries', 'create_open_boundaries']
