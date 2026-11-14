"""
Core Abstractions for RELIC Framework

This module defines the fundamental abstract base classes that form the
architectural foundation of the RELIC (Recomposable Ecosystem Library for
Integrated Coupling) system. These four pillars provide the theoretical and
computational foundation for all scientific processes in the ecosystem model.

The design follows category theory principles where abstract base classes
define objects and their morphisms (processes) that preserve structure
and composability across different spatiotemporal scales and physical domains.

Architecture:
    Object Categories: PhysicalField, BiologicalProcess, TransportMechanism,
        ScaleCoupling
    Morphisms: Abstract methods defining transformations between objects
    Functors: Concrete implementations that preserve category structure
    Natural Transformations: Coupling mechanisms between categories

References:
    - Mac Lane, S. (1998). Categories for the Working Mathematician. Springer-Verlag
    - Spivak, D.I. (2014). Category Theory for the Sciences. MIT Press
    - Goodfellow, I. et al. (2016). Deep Learning. MIT Press
    - PyTorch Team (2019). PyTorch: An Imperative Style, High-Performance Deep Learning
      Library
    - Farquhar, G.D. et al. (1980). A biochemical model of photosynthetic CO2 assimilation
    - Lloyd, J. & Taylor, J.A. (1994). On the temperature dependence of soil respiration
    - Guenther, A.B. et al. (2012). The Model of Emissions of Gases and Aerosols from
      Nature
"""

import logging
from abc import ABC, abstractmethod
import math
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, TypeAlias, Union, cast

import torch

if TYPE_CHECKING:
    from .functor import FunctorMetadata
    from .state import ExternalIntent, UpdateItem

logger = logging.getLogger(__name__)

# Type aliases for clarity and consistency
Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = torch.device
StateDict: TypeAlias = Dict[str, Tensor]
EnvironmentDict: TypeAlias = Dict[str, Tensor]
FluxValue: TypeAlias = Union[Tensor, float, int, "UpdateItem", "ExternalIntent", Dict[str, Any]]
FluxDict: TypeAlias = Dict[str, FluxValue]
DiagnosticsDict: TypeAlias = Dict[str, Tensor]  # Standard type for module diagnostics output
CoordinateTensor: TypeAlias = Tensor  # Shape: [batch_size, n_dims] or [batch_size, spatial_dims]
FieldTensor: TypeAlias = Tensor  # Shape: [batch_size] or [spatial_shape]
GradientTensor: TypeAlias = Tensor   # Shape: [batch_size, n_dims]


# Shared abstract-layer parameter resolution path.
def _resolve_parameter_with_manager(
    owner: Any,
    *,
    param_name: str,
    namespace_candidates: Optional[Tuple[Optional[str], ...]] = None,
) -> Any:
    """
    Resolve a parameter from local config, then manager namespaces.

    A parameter found in neither raises KeyError; there is no caller default.

    Notes:
        This path exists for abstract classes used outside FunctorBase.
        Bridge functors should rely on FunctorBase.get_parameter_value() through MRO.
    """
    config = getattr(owner, "config", None)
    if isinstance(config, dict) and param_name in config:
        return config[param_name]

    manager: Any = None
    get_manager = getattr(owner, "_get_parameter_manager", None)
    if callable(get_manager):
        manager = get_manager()

    if manager is not None:
        if isinstance(param_name, str) and "." in param_name:
            if manager.get_parameter(param_name) is not None:
                value = manager.get(param_name)
                if value is not None:
                    return value

        for namespace in namespace_candidates or ():
            if not namespace:
                continue
            full_key = f"{namespace}.{param_name}"
            if manager.get_parameter(full_key) is None:
                continue
            value = manager.get(full_key)
            if value is not None:
                return value

    raise KeyError(f"Parameter not found: {param_name}")


# Unified Tensor Processing Utilities for RELIC
def ensure_tensor(
    value: Union[float, int, Tensor],
    reference_tensor: Optional[Tensor] = None,
    device: Optional[Device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Convert value to tensor with consistent device and dtype.

    Parameters:
        value: Value to convert to tensor
        reference_tensor: Reference tensor for device and dtype
        device: Target device (overrides reference_tensor.device)
        dtype: Target dtype (overrides reference_tensor.dtype)

    Returns:
        Tensor with consistent device and dtype
    """
    if reference_tensor is not None:
        target_device = device if device is not None else reference_tensor.device
        target_dtype = dtype if dtype is not None else reference_tensor.dtype
    else:
        target_device = device if device is not None else torch.device("cpu")
        target_dtype = dtype if dtype is not None else torch.float32

    if torch.is_tensor(value):
        tensor_value = cast(Tensor, value)
    else:
        tensor_value = torch.tensor(float(value), dtype=target_dtype)

    if target_device is not None and tensor_value.device != target_device:
        tensor_value = tensor_value.to(target_device)

    if target_dtype is not None and tensor_value.dtype != target_dtype:
        tensor_value = tensor_value.to(target_dtype)

    return tensor_value


def safe_item(tensor: Tensor) -> float:
    """
    Safely extract scalar value from tensor for logging/reporting.

    This function ensures the tensor is detached from computation graph
    before calling .item() to prevent gradient tracking issues.

    Parameters:
        tensor: Input tensor (should be scalar or single-element)

    Returns:
        Float value extracted from tensor
    """
    if not torch.is_tensor(tensor):
        return float(tensor)

    if tensor.numel() != 1:
        raise ValueError(f"Tensor must be scalar, got shape {tensor.shape}")

    return float(cast(float, tensor.detach().item()))


def safe_mean_item(tensor: Tensor) -> float:
    """
    Safely extract mean value from tensor for logging/reporting.

    Parameters:
        tensor: Input tensor

    Returns:
        Float mean value extracted from tensor
    """
    if not torch.is_tensor(tensor):
        return float(tensor)

    return float(cast(float, tensor.mean().detach().item()))


def ensure_device_consistency(
    *tensors: Any,
    target_device: Optional[Device] = None,
) -> List[Tensor]:
    """
    Ensure all tensors are on the same device.

    Parameters:
        tensors: Variable number of tensors to check
        target_device: Target device (uses first tensor's device if None)

    Returns:
        List of tensors on consistent device
    """
    if not tensors:
        return []

    if target_device is None:
        first_tensor = next((t for t in tensors if torch.is_tensor(t)), None)
        if first_tensor is not None:
            target_device = cast(Tensor, first_tensor).device
        else:
            target_device = torch.device("cpu")

    result = []
    for tensor in tensors:
        if torch.is_tensor(tensor):
            result.append(tensor.to(device=target_device))
        else:
            result.append(torch.tensor(tensor, device=target_device))

    return result


def ensure_dtype_consistency(
    *tensors: Any,
    target_dtype: Optional[torch.dtype] = None,
) -> List[Tensor]:
    """
    Ensure all tensors have the same dtype.

    Parameters:
        tensors: Variable number of tensors to check
        target_dtype: Target dtype (uses first tensor's dtype if None)

    Returns:
        List of tensors with consistent dtype
    """
    if not tensors:
        return []

    if target_dtype is None:
        first_tensor = next((t for t in tensors if torch.is_tensor(t)), None)
        if first_tensor is not None:
            target_dtype = cast(Tensor, first_tensor).dtype
        else:
            target_dtype = torch.float32

    result = []
    for tensor in tensors:
        if torch.is_tensor(tensor):
            result.append(tensor.to(dtype=target_dtype))
        else:
            result.append(torch.tensor(tensor, dtype=target_dtype))

    return result


class PhysicalField(ABC):
    """
    Physical Field Abstract Base Class

    Defines spatially and temporally distributed physical quantities including
    temperature fields, moisture distributions, concentration gradients,
    electromagnetic fields, pressure surfaces, and other continuously varying
    environmental properties that govern ecosystem processes.

    This class provides the fundamental interface for all field-based computations
    including spatial interpolation, gradient calculation, field evolution, and
    boundary condition enforcement.

    Mathematical Expression:
        Field representation: phi(x, t) : R^d x R -> R^n
        where x in R^d is spatial coordinate, t in R is time, phi in R^n is field
        value

        Gradient operator: grad(phi) = (dphi/dx_1, dphi/dx_2, ..., dphi/dx_d)

        Field evolution: dphi/dt = f(phi, grad(phi), laplacian(phi), external_forcing)

        Interpolation: phi_interp(x) = Sum_i w_i(x) phi_i where Sum_i w_i(x) = 1

        Conservation: Integral_Omega dphi/dt dOmega + Integral_boundary phi . n dS
        = Integral_Omega S_phi dOmega
        where S_phi is source/sink term, n is outward normal

    Class attribute (required, declared by every concrete field):
        SPATIAL_DIMS : Optional[int]
            Number of coordinate dimensions of the field's spatial query (1, 2
            or 3), or None for a point model, which has no coordinate query
            (its state is read from StateManager).

    The field's identity is its declared outputs (FunctorMetadata.output_variables)
    and their units are the ContractRegistry units; the abstraction declares no
    second name, unit, initial value or time flag.

    Parameters:
        device : Optional[torch.device]
            Computation device (CPU/GPU)

    References:
        - Aris, R. (1962). Vectors, Tensors and the Basic Equations of Fluid
          Mechanics
        - Whitaker, S. (1999). The Method of Volume Averaging. Springer
        - Cushman-Roisin, B. (2011). Introduction to Geophysical Fluid Dynamics
        - LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and
          Partial Differential Equations
    """

    def __init__(self, device: Optional[Device] = None) -> None:
        """
        Initialize the physical field and check its declared SPATIAL_DIMS.

        Mathematical Expression:
            Domain definition: Omega subset R^SPATIAL_DIMS (no domain for a
            point model, SPATIAL_DIMS = None)
        """
        ABC.__init__(self)

        if not hasattr(type(self), "SPATIAL_DIMS"):
            raise TypeError(
                f"{type(self).__name__} must declare the class attribute SPATIAL_DIMS "
                "(1, 2 or 3 coordinate dimensions, or None for a point model)"
            )
        dims = type(self).SPATIAL_DIMS
        if dims is not None:
            if isinstance(dims, bool) or not isinstance(dims, int):
                raise TypeError(
                    f"{type(self).__name__}.SPATIAL_DIMS must be an int (1, 2 or 3) or None, "
                    f"got {dims!r}"
                )
            if dims not in (1, 2, 3):
                raise ValueError(
                    f"{type(self).__name__}.SPATIAL_DIMS must be 1, 2 or 3 (or None), got {dims}"
                )
        self.device = device or torch.device("cpu")

        # Field metadata for computational efficiency
        self.grid_resolution: Optional[Tuple[int, ...]] = None
        self.boundary_conditions: Dict[str, str] = {}
        self.interpolation_method: str = "linear"

    @abstractmethod
    def get_value_at(self, coordinates: CoordinateTensor) -> FieldTensor:
        """
        Get field values at specified spatial coordinates.

        Evaluates the field phi(x) at arbitrary spatial locations using
        interpolation or analytical expressions. Critical for coupling
        between different spatial discretizations and model components.

        Parameters:
            coordinates : CoordinateTensor
                Spatial coordinates where field is evaluated
                Shape: [batch_size, spatial_dims]
                Units: [m] for spatial coordinates

        Returns:
            values : FieldTensor
                Field values at specified coordinates
                Shape: [batch_size]
                Units: the ContractRegistry units of the field's declared output

        Raises:
            ValueError: If coordinates have wrong spatial dimensions
            RuntimeError: If field is not initialized

        Mathematical Expression:
            Evaluation operator: E[phi](x) = phi(x)
            For discrete fields: E[phi](x) = Sum_i w_i(x) phi_i
            where w_i(x) are interpolation weights

            Interpolation methods:
            - Linear: w_i(x) = max(0, 1 - |x - x_i|/h)  [piecewise linear]
            - Cubic: w_i(x) = cubic spline basis functions
            - Spectral: w_i(x) = Fourier/Chebyshev basis functions

        References:
            - Press, W.H. et al. (2007). Numerical Recipes. Cambridge University Press
            - Davis, P.J. (1975). Interpolation and Approximation. Dover Publications
        """
        pass

    @abstractmethod
    def compute_gradient(self, coordinates: CoordinateTensor) -> GradientTensor:
        """
        Compute spatial gradient of field at specified coordinates.

        Calculates gradphi using finite differences, spectral methods, or analytical
        derivatives. Essential for transport processes, force calculations,
        and coupling with momentum equations.

        Parameters:
            coordinates : CoordinateTensor
                Spatial coordinates where gradient is evaluated
                Shape: [batch_size, spatial_dims]
                Units: [m] for spatial coordinates

        Returns:
            gradient : GradientTensor
                Field gradient vector at each coordinate
                Shape: [batch_size, spatial_dims]
                Units: [field_units m^-^1]

        Raises:
            ValueError: If coordinates have wrong spatial dimensions
            RuntimeError: If field gradients are not computable

        Mathematical Expression:
            Gradient operator: grad(phi) = (dphi/dx_1, dphi/dx_2, ..., dphi/dx_d)

            Finite difference approximations:
            - Forward: dphi/dx approx (phi(x+h) - phi(x))/h
            - Central: dphi/dx approx (phi(x+h) - phi(x-h))/(2h)
            - Backward: dphi/dx approx (phi(x) - phi(x-h))/h

            Higher-order schemes:
            - Fourth-order central: dphi/dx approx (-phi(x+2h) + 8phi(x+h)
              - 8phi(x-h) + phi(x-2h))/(12h)

            Spectral methods:
            - Fourier: dphi/dx = F^-^1[ik F[phi]] where F is Fourier transform

        References:
            - Fornberg, B. (1988). Generation of finite difference formulas
            - Canuto, C. et al. (2006). Spectral Methods: Fundamentals in Single Domains
        """
        pass

    @abstractmethod
    def interpolate(
        self,
        source_coordinates: CoordinateTensor,
        target_coordinates: CoordinateTensor,
    ) -> FieldTensor:
        """
        Interpolate field values from source to target coordinate system.

        Performs spatial interpolation between different computational grids,
        enabling coupling between models with different spatial discretizations.
        Critical for multi-scale modeling and adaptive mesh refinement.

        Parameters:
            source_coordinates : CoordinateTensor
                Source grid coordinates
                Shape: [n_source, spatial_dims]
                Units: [m]
            target_coordinates : CoordinateTensor
                Target interpolation coordinates
                Shape: [n_target, spatial_dims]
                Units: [m]

        Returns:
            interpolated_values : FieldTensor
                Interpolated field values at target coordinates
                Shape: [n_target]
                Units: [field units]

        The concrete field owns its one interpolation scheme; there is no
        method selection.

        Raises:
            RuntimeError: If interpolation fails

        Mathematical Expression:
            Interpolation operator: I[phi](x_target) = Sum_i w_i(x_target, x_source_i)
            phi(x_source_i)

            Weight functions:
            - Linear: w_i = Product_j max(0, 1 - |x_target_j - x_source_i_j|
              /h_j)
            - Cubic spline: w_i = cubic polynomial basis functions
            - RBF: w_i = R(||x_target - x_source_i||) where R is radial basis function
            - Kriging: w_i minimizes E[(phi_true - phi_interp)^2] subject to
              unbiasedness

            Conservation property: Integral I[phi] dx = Integral phi dx (for conservative schemes)

        References:
            - Fasshauer, G.E. (2007). Meshfree Approximation Methods with MATLAB
            - Cressie, N. (1993). Statistics for Spatial Data. Wiley
        """
        pass

    @abstractmethod
    def update_field(
        self,
        time_step: float,
        source_terms: EnvironmentDict,
        boundary_conditions: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        """
        Update field values according to governing equations and source terms.

        Advances the field in time using numerical integration schemes while
        enforcing conservation laws, boundary conditions, and physical constraints.
        Forms the core of field evolution in ecosystem models.

        Parameters:
            time_step : float
                Time integration step size
                Units: [s]
            source_terms : EnvironmentDict
                Source/sink terms for field evolution
                Keys: source term names, Values: source rates [field_units s^-^1]
            boundary_conditions : Optional[Dict[str, Tensor]]
                Boundary condition specifications
                Keys: boundary names ("north", "south", etc.)
                Values: boundary values or fluxes

        Raises:
            ValueError: If time_step is non-positive
            RuntimeError: If integration becomes unstable

        Mathematical Expression:
            General evolution equation: dphi/dt = F[phi, gradphi, grad^2phi]
            + S(x,t)
            where F is the field operator and S represents source terms

            Time integration schemes:
            - Explicit Euler: phi^(n+1) = phi^n + Deltat F[phi^n]
            - Implicit Euler: phi^(n+1) = phi^n + Deltat F[phi^(n+1)]
            - Runge-Kutta 4: phi^(n+1) = phi^n + Deltat/6
              (k_1 + 2k_2 + 2k_3 + k_4)
            - Crank-Nicolson: phi^(n+1) = phi^n + Deltat/2
              (F[phi^n] + F[phi^(n+1)])

            Boundary conditions:
            - Dirichlet: phi|_dOmega = g(x,t)
            - Neumann: gradphi.n|_dOmega = h(x,t)
            - Robin: alphaphi + betagradphi.n|_dOmega = gamma(x,t)

            Stability constraint (CFL condition): Deltat <= min(Deltax^2/(2D),
            Deltax/|v|)
            where D is diffusivity and v is advection velocity

        References:
            - Strikwerda, J.C. (2004). Finite Difference Schemes and Partial
              Differential Equations
            - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of
              Time-Dependent Advection-Diffusion-Reaction Equations
        """
        pass

    def validate_coordinates(self, coordinates: CoordinateTensor) -> bool:
        """
        Validate coordinate tensor dimensions and values.

        Parameters:
            coordinates : CoordinateTensor
                Coordinates to validate

        Returns:
            bool: True if coordinates are valid

        Raises:
            RuntimeError: For a point model (SPATIAL_DIMS = None), which has no
                coordinate query.
        """
        dims = type(self).SPATIAL_DIMS
        if dims is None:
            raise RuntimeError(
                f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
                "coordinate query; its state is read from StateManager"
            )
        if not isinstance(coordinates, torch.Tensor):
            return False
        if coordinates.dim() != 2:
            return False
        if coordinates.size(-1) != dims:
            return False
        return True

    # Internal interpolation tool suite
    def _linear_interpolation_1d(
        self,
        source_values: Tensor,
        source_coords: Tensor,
        target_coords: Tensor,
        extrapolate: bool = False,
        monotonic: bool = False
    ) -> Tensor:
        """
        1D linear interpolation with optional monotonicity preservation.

        Parameters:
            source_values : Tensor
                Source field values, shape [n_source]
            source_coords : Tensor
                Source coordinates, shape [n_source, 1]
            target_coords : Tensor
                Target coordinates, shape [n_target, 1]
            extrapolate : bool
                If True, extrapolate beyond source bounds
            monotonic : bool
                If True, enforce monotonicity constraints

        Returns:
            Tensor: Interpolated values, shape [n_target]
        """
        # Flatten coordinates
        x_source = source_coords.squeeze(-1)
        x_target = target_coords.squeeze(-1)

        # Sort source points
        sorted_idx = torch.argsort(x_source)
        x_source_sorted = x_source[sorted_idx]
        y_source_sorted = source_values[sorted_idx]

        # Find interpolation indices
        indices = torch.searchsorted(x_source_sorted, x_target)

        # Clamp indices to valid range
        indices = torch.clamp(indices, 1, len(x_source_sorted) - 1)

        # Get surrounding points
        x0 = x_source_sorted[indices - 1]
        x1 = x_source_sorted[indices]
        y0 = y_source_sorted[indices - 1]
        y1 = y_source_sorted[indices]

        # Compute interpolation weights
        weights = (x_target - x0) / (x1 - x0 + 1e-10)

        # Linear interpolation
        y_target = y0 * (1 - weights) + y1 * weights

        # Handle extrapolation
        if not extrapolate:
            # Clamp to nearest values outside bounds
            below_mask = x_target < x_source_sorted[0]
            above_mask = x_target > x_source_sorted[-1]
            y_target[below_mask] = y_source_sorted[0]
            y_target[above_mask] = y_source_sorted[-1]

        # Apply monotonicity constraint if requested
        if monotonic:
            # Ensure no negative values for positive fields
            y_target = torch.clamp(y_target, min=0)

        return y_target

    def _bilinear_interpolation_2d(
        self,
        source_values: Tensor,
        source_coords: Tensor,
        target_coords: Tensor,
        grid_shape: Optional[Tuple[int, int]] = None,
        extrapolate: bool = False
    ) -> Tensor:
        """
        2D bilinear interpolation on regular or irregular grids.

        Parameters:
            source_values : Tensor
                Source field values, shape [n_source] or [ny, nx]
            source_coords : Tensor
                Source coordinates, shape [n_source, 2]
            target_coords : Tensor
                Target coordinates, shape [n_target, 2]
            grid_shape : Optional[Tuple[int, int]]
                Shape of regular grid if applicable
            extrapolate : bool
                If True, extrapolate beyond source bounds

        Returns:
            Tensor: Interpolated values, shape [n_target]
        """
        # For regular grids, use efficient grid interpolation
        if grid_shape is not None and source_values.dim() == 2:
            ny, nx = grid_shape
            # Normalize target coordinates to grid indices
            x_min, x_max = source_coords[:, 0].min(), source_coords[:, 0].max()
            y_min, y_max = source_coords[:, 1].min(), source_coords[:, 1].max()

            # Map to grid indices
            x_idx = (target_coords[:, 0] - x_min) / (x_max - x_min) * (nx - 1)
            y_idx = (target_coords[:, 1] - y_min) / (y_max - y_min) * (ny - 1)

            # Get integer indices and weights
            x0 = torch.floor(x_idx).long()
            y0 = torch.floor(y_idx).long()
            x1 = x0 + 1
            y1 = y0 + 1

            # Compute weights
            wx = x_idx - x0.float()
            wy = y_idx - y0.float()

            # Clamp indices
            x0 = torch.clamp(x0, 0, nx - 1)
            x1 = torch.clamp(x1, 0, nx - 1)
            y0 = torch.clamp(y0, 0, ny - 1)
            y1 = torch.clamp(y1, 0, ny - 1)

            # Bilinear interpolation
            v00 = source_values[y0, x0]
            v01 = source_values[y0, x1]
            v10 = source_values[y1, x0]
            v11 = source_values[y1, x1]

            v0 = v00 * (1 - wx) + v01 * wx
            v1 = v10 * (1 - wx) + v11 * wx
            result = v0 * (1 - wy) + v1 * wy

            # Handle extrapolation
            if not extrapolate:
                out_of_bounds = (
                    (target_coords[:, 0] < x_min) | (target_coords[:, 0] > x_max) |
                    (target_coords[:, 1] < y_min) | (target_coords[:, 1] > y_max)
                )
                result[out_of_bounds] = 0.0

            return result
        else:
            # Default to scattered interpolation
            # This is simplified - actual implementation would use
            # Delaunay triangulation or similar
            logger.warning("Irregular grid interpolation not fully implemented")
            return torch.zeros(len(target_coords), device=self.device)

    def _trilinear_interpolation_3d(
        self,
        source_values: Tensor,
        source_coords: Tensor,
        target_coords: Tensor,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        extrapolate: bool = False
    ) -> Tensor:
        """
        3D trilinear interpolation on regular grids.

        Parameters:
            source_values : Tensor
                Source field values, shape [nz, ny, nx]
            source_coords : Tensor
                Source coordinates, shape [n_source, 3]
            target_coords : Tensor
                Target coordinates, shape [n_target, 3]
            grid_shape : Optional[Tuple[int, int, int]]
                Shape of regular grid
            extrapolate : bool
                If True, extrapolate beyond source bounds

        Returns:
            Tensor: Interpolated values, shape [n_target]
        """
        if grid_shape is None or source_values.dim() != 3:
            logger.warning("3D interpolation requires regular grid")
            return torch.zeros(len(target_coords), device=self.device)

        nz, ny, nx = grid_shape

        # Get grid bounds
        x_min, x_max = source_coords[:, 0].min(), source_coords[:, 0].max()
        y_min, y_max = source_coords[:, 1].min(), source_coords[:, 1].max()
        z_min, z_max = source_coords[:, 2].min(), source_coords[:, 2].max()

        # Map to grid indices
        x_idx = (target_coords[:, 0] - x_min) / (x_max - x_min) * (nx - 1)
        y_idx = (target_coords[:, 1] - y_min) / (y_max - y_min) * (ny - 1)
        z_idx = (target_coords[:, 2] - z_min) / (z_max - z_min) * (nz - 1)

        # Get integer indices
        x0 = torch.floor(x_idx).long()
        y0 = torch.floor(y_idx).long()
        z0 = torch.floor(z_idx).long()
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Compute weights
        wx = x_idx - x0.float()
        wy = y_idx - y0.float()
        wz = z_idx - z0.float()

        # Clamp indices
        x0 = torch.clamp(x0, 0, nx - 1)
        x1 = torch.clamp(x1, 0, nx - 1)
        y0 = torch.clamp(y0, 0, ny - 1)
        y1 = torch.clamp(y1, 0, ny - 1)
        z0 = torch.clamp(z0, 0, nz - 1)
        z1 = torch.clamp(z1, 0, nz - 1)

        # Trilinear interpolation
        v000 = source_values[z0, y0, x0]
        v001 = source_values[z0, y0, x1]
        v010 = source_values[z0, y1, x0]
        v011 = source_values[z0, y1, x1]
        v100 = source_values[z1, y0, x0]
        v101 = source_values[z1, y0, x1]
        v110 = source_values[z1, y1, x0]
        v111 = source_values[z1, y1, x1]

        # Interpolate along x
        v00 = v000 * (1 - wx) + v001 * wx
        v01 = v010 * (1 - wx) + v011 * wx
        v10 = v100 * (1 - wx) + v101 * wx
        v11 = v110 * (1 - wx) + v111 * wx

        # Interpolate along y
        v0 = v00 * (1 - wy) + v01 * wy
        v1 = v10 * (1 - wy) + v11 * wy

        # Interpolate along z
        result = v0 * (1 - wz) + v1 * wz

        # Handle extrapolation
        if not extrapolate:
            out_of_bounds = (
                (target_coords[:, 0] < x_min) | (target_coords[:, 0] > x_max) |
                (target_coords[:, 1] < y_min) | (target_coords[:, 1] > y_max) |
                (target_coords[:, 2] < z_min) | (target_coords[:, 2] > z_max)
            )
            result[out_of_bounds] = 0.0

        return result

    def _nearest_neighbor_interpolation(
        self,
        source_values: Tensor,
        source_coords: Tensor,
        target_coords: Tensor
    ) -> Tensor:
        """
        Nearest neighbor interpolation for any dimensionality.

        Parameters:
            source_values : Tensor
                Source field values, shape [n_source]
            source_coords : Tensor
                Source coordinates, shape [n_source, d]
            target_coords : Tensor
                Target coordinates, shape [n_target, d]

        Returns:
            Tensor: Interpolated values, shape [n_target]
        """
        # Compute pairwise distances
        # Expand dimensions for broadcasting
        source_expanded = source_coords.unsqueeze(0)  # [1, n_source, d]
        target_expanded = target_coords.unsqueeze(1)  # [n_target, 1, d]

        # Compute squared distances
        distances = torch.sum((target_expanded - source_expanded) ** 2, dim=-1)

        # Find nearest neighbors
        _, nearest_idx = torch.min(distances, dim=1)

        # Get nearest values
        result = source_values[nearest_idx]

        return result

    def _compute_conservative_weights(
        self,
        source_coords: Tensor,
        target_coords: Tensor,
        source_areas: Optional[Tensor] = None,
        target_areas: Optional[Tensor] = None,
        spatial_dims: int = 2
    ) -> Tensor:
        """
        Compute conservative remapping weights based on cell overlap.

        This ensures that extensive quantities (e.g., total mass) are conserved
        during interpolation by computing area/volume-weighted transfers.

        Parameters:
            source_coords : Tensor
                Source cell centers, shape [n_source, spatial_dims]
            target_coords : Tensor
                Target cell centers, shape [n_target, spatial_dims]
            source_areas : Optional[Tensor]
                Source cell areas/volumes, shape [n_source]
            target_areas : Optional[Tensor]
                Target cell areas/volumes, shape [n_target]
            spatial_dims : int
                Number of spatial dimensions (1, 2, or 3)

        Returns:
            Tensor: Weight matrix, shape [n_target, n_source]
                    where weights[i, j] is transfer from source j to target i
        """
        n_source = len(source_coords)
        n_target = len(target_coords)

        # Initialize weight matrix
        weights = torch.zeros(n_target, n_source, device=self.device)

        # Simplified implementation: inverse distance weighting
        # Full implementation would compute actual cell overlaps
        for i in range(n_target):
            # Compute distances from target i to all sources
            distances = torch.norm(source_coords - target_coords[i], dim=-1)

            # Avoid division by zero
            distances = torch.clamp(distances, min=1e-10)

            # Inverse distance weights
            inv_distances = 1.0 / distances

            # Normalize to sum to 1 (conservative)
            weights[i] = inv_distances / inv_distances.sum()

            # Apply area weighting if provided
            if source_areas is not None:
                weights[i] *= source_areas
                weights[i] /= weights[i].sum()

        return weights

    def _apply_monotonicity_limiter(
        self,
        interpolated: Tensor,
        source_min: float,
        source_max: float,
        preserve_positivity: bool = True
    ) -> Tensor:
        """
        Apply monotonicity constraints to prevent overshoots and undershoots.

        Parameters:
            interpolated : Tensor
                Interpolated values
            source_min : float
                Minimum value in source data
            source_max : float
                Maximum value in source data
            preserve_positivity : bool
                If True, enforce non-negative values

        Returns:
            Tensor: Limited values
        """
        # Clamp to source range
        limited = torch.clamp(interpolated, min=source_min, max=source_max)

        # Preserve positivity if requested
        if preserve_positivity:
            limited = torch.clamp(limited, min=0)

        return limited


class BiologicalProcess(ABC):
    """
    Biological Process Abstract Base Class

    Defines biochemical reactions and physiological processes within organisms
    including photosynthesis, respiration, decomposition, nitrogen fixation,
    BVOC synthesis, root uptake, stomatal regulation, and other metabolic
    processes with complex environmental dependencies and regulatory mechanisms.

    This class provides the interface for all biological rate calculations,
    enzyme kinetics, allometric scaling, phenological controls, and stress
    response functions that govern ecosystem carbon, nitrogen, and water cycles.

    Mathematical Expression:
        Rate equation: r = k(T,pH,moisture) * f(substrate) * g(inhibitor)
        * h(enzyme)

        Enzyme kinetics: r = Vmax * [S] / (Km + [S])  [Michaelis-Menten]
        Competitive inhibition: r = Vmax * [S] / (Km(1 + [I]/Ki) + [S])

        Temperature response: k(T) = k_25 * Q_10^((T-298.15)/10) [Q10 model]
        Arrhenius form: k(T) = A * exp(-Ea/(R*T))

        Optimal temperature: k(T) = kopt * exp(-((T-Topt)/(sigmaT))^2) [Gaussian]

        Moisture response: f(theta) = (theta - theta_wilt)/(theta_fc - theta_wilt)
        for theta_wilt <= theta <= theta_fc

        Light response: f(I) = alpha*I / (1 + alpha*I/Isat)  [Michaelis-Menten light curve]

        Nutrient limitation: f(N) = N / (N + Khalf)  [Monod kinetics]

    Parameters:
        process_name : str
            Unique identifier for the biological process
        allometric_scaling : Optional[float]
            Allometric scaling exponent (default: None) [dimensionless]
        device : Optional[torch.device]
            Computation device

    References:
        - Farquhar, G.D. et al. (1980). A biochemical model of photosynthetic CO2
          assimilation in leaves of C3 species
        - Collatz, G.J. et al. (1991). Physiological and environmental regulation of
          stomatal conductance, photosynthesis and transpiration
        - Lloyd, J. & Taylor, J.A. (1994). On the temperature dependence of soil respiration
        - Ball, J.T. et al. (1987). A model predicting stomatal conductance and its
          contribution to the control of photosynthesis under different
          environmental conditions
        - von Caemmerer, S. (2000). Biochemical Models of Leaf Photosynthesis. CSIRO Publishing
        - Guenther, A.B. et al. (2012). The Model of Emissions of Gases and Aerosols
          from Nature version 2.1 (MEGAN2.1)
    """

    def __init__(
        self,
        process_name: str,
        allometric_scaling: Optional[float] = None,
        device: Optional[Device] = None,
    ) -> None:
        """
        Initialize the biological process contract.

        The contract carries no temperature or pH law label: each concrete
        process evaluates its own temperature and pH responses.

        Mathematical Expression:
            Process initialization: P_0 = {process_name, scaling}
        """
        ABC.__init__(self)

        if not process_name or not isinstance(process_name, str):
            raise ValueError("Process name must be a non-empty string")

        self.process_name = process_name
        self.allometric_scaling = allometric_scaling
        self.device = device or torch.device("cpu")
        self.config: Dict[str, Any] = {}  # Initialize config for parameter storage

        # Process parameters (to be set by subclasses or resolved from parameter system)
        self.base_rate: float = 1.0  # Base process rate [process_units]
        self.temperature_optimum: Optional[float] = None
        self.q10_value: Optional[float] = None
        self.ph_optimum: Optional[float] = None

        from ..utils import physical_constants

        # The reference temperature is a physical-constant SSOT entry; a failed
        # lookup propagates.
        self.temperature_optimum = float(
            physical_constants.get_physical_constant("reference_temperature")
        )
        # q10_value and ph_optimum stay None here: at construction the contract
        # has no configuration and no parameter manager, so a lookup could only
        # fail. The executable bridge (BiologicalProcessFunctor) resolves them
        # through FunctorBase.get_parameter_value.

    def get_parameter_value(
        self,
        param_name: str,
    ) -> Any:
        """
        Get parameter value from config or parameter manager.

        This method provides a unified interface for parameter access
        across all biological processes, supporting both local config
        and integration with the global parameter manager.

        Parameters:
            param_name : str
                Parameter name

        Returns:
            Any: Parameter value

        Raises:
            KeyError: If the parameter is found in neither source (there is
                no caller default).
        """
        explicit_namespace = getattr(self, "parameter_namespace", None)
        process_namespace = f"pedosphere.{self.process_name}"
        return _resolve_parameter_with_manager(
            self,
            param_name=param_name,
            namespace_candidates=(explicit_namespace, process_namespace),
        )

    @abstractmethod
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """
        Execute one biological-process step and return process outputs.

        This is the single public computation entry point for biological
        process contracts. Concrete implementations should place all
        scientific logic in ``forward``.
        """
        pass

    def parameter_sensitivity(
        self,
        parameter_name: str,
        state: StateDict,
        *,
        time: float,
        dt: float,
        target_output: Optional[str] = None,
        use_central_difference: bool = True,
        absolute_eps: Optional[float] = None,
        relative_eps: Optional[float] = None,
        aggregation_method: Optional[str] = None,
    ) -> Tensor:
        """
        Compute parameter sensitivity of process rates using finite differences.

        This base implementation provides central and forward difference methods
        with adaptive step size selection and multiple aggregation strategies. A
        stochastic Functor (a ``FunctorSubstreamMixin`` substream owner) is
        differenced with common random numbers: every evaluation starts from the
        substream position saved before the first one, and that position is
        restored after the last one, so the call leaves the stream unchanged.

        Calculates dr/dtheta where r is process rate and theta is parameter.
        Critical for parameter optimization, uncertainty quantification,
        and model calibration procedures.

        The parameter is a tensor member of this module or of one of its
        submodules (an ``nn.Parameter``, a registered buffer or a tensor
        attribute). Each evaluation replaces it through
        ``torch.func.functional_call`` and runs the governed call
        ``self(state, time=time, dt=dt)``; the module itself is never mutated,
        so the original member object and value remain in place, also when an
        evaluation raises.

        Parameters:
            parameter_name : str
                Qualified name of the tensor member (e.g. ``"vcmax25"`` or
                ``"kinetics.k"``)
            state : StateDict
                Current state variables; the only source of scientific inputs
            time : float
                Model time of the evaluation [s]
            dt : float
                Step length of the evaluation [s]
            target_output : Optional[str]
                Output whose sensitivity is returned by the 'target' aggregation
            use_central_difference : bool
                If True, use central difference; if False, use forward
                Default: True
            absolute_eps : Optional[float]
                Absolute bound of an explicit step [parameter units]
                Default: None (no absolute bound)
            relative_eps : Optional[float]
                Relative bound of an explicit step, times x_c [dimensionless]
                Default: None (no relative bound); with neither bound the
                dtype-aware default step is used
            aggregation_method : Optional[str]
                How to aggregate multi-output to scalar ('sum', 'L2', 'target', None)
                Default: None ('target' when target_output is given, else 'sum')

        Returns:
            sensitivity : Tensor
                Parameter sensitivity dr/dtheta
                Shape: [batch_size] or scalar
                Units: [rate_units / parameter_units]

        Raises:
            TypeError: If the process is not an ``nn.Module`` or the named member
                is not a tensor
            KeyError: If parameter_name is not found
            ValueError: If the aggregation or the target is invalid
            RuntimeError: If sensitivity calculation fails

        Mathematical Expression:
            Central difference (preferred for accuracy):
            dr/dtheta approx (r(theta + epsilon) - r(theta - epsilon)) / (2epsilon)

            Forward difference (forward difference for boundary parameters):
            dr/dtheta approx (r(theta + epsilon) - r(theta)) / epsilon

            Default step (Press et al. 1992, section 5.7):
            forward h = eps_m^(1/2) x_c, central h = eps_m^(1/3) x_c,
            x_c = |theta| (theta != 0) or the YAML floor (theta = 0),
            eps_m = torch.finfo(theta.dtype).eps
            Explicit step (each bound optional; the largest given bound):
            epsilon = max(epsilon_abs, epsilon_rel * x_c)

            Aggregation for multi-output:
            - target: Use target_output as key
            - sum: Sum_i r_i (default for FluxDict)
            - L2: sqrt(Sum_i r_i^2)

        References:
            - Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer
            - Sobol, I.M. (2001). Global sensitivity indices for nonlinear models
            - Press, W.H. et al. (2007). Numerical Recipes
            - Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P.
              (1992). Numerical Recipes in C, 2nd ed., section 5.7
            - Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering,
              section 7.1 (finite differences with common random numbers)
            - L'Ecuyer, P. and Perron, G. (1994). On the convergence rates of IPA
              and FDC derivative estimators. Operations Research 42(4), 643-656
        """
        from torch import nn
        from torch.func import functional_call

        from .functor.stochastic_substream import FunctorSubstreamMixin
        from .state import StateManager
        from .state.types import partition_outputs

        if not isinstance(self, nn.Module):
            raise TypeError(
                f"{type(self).__name__}.parameter_sensitivity replaces the parameter through "
                "torch.func.functional_call and needs an nn.Module (a BiologicalProcessFunctor)"
            )
        if aggregation_method not in (None, "target", "sum", "L2"):
            raise ValueError("Unsupported sensitivity aggregation method")
        target_key = target_output
        if target_key is not None and (not isinstance(target_key, str) or not target_key):
            raise ValueError("Sensitivity target must be a nonempty string")
        use_target = aggregation_method == "target" or (aggregation_method is None and target_key is not None)
        if use_target and target_key is None:
            raise ValueError("Target aggregation requires target_output")

        # Resolve the member once: a tensor of this module or of a submodule.
        owner_path, _, leaf = parameter_name.rpartition(".")
        parameter_owner: Any = self
        if owner_path:
            try:
                parameter_owner = self.get_submodule(owner_path)
            except AttributeError as exc:
                raise KeyError(
                    f"Parameter '{parameter_name}' not found in {self.process_name}: "
                    f"'{owner_path}' is not a submodule"
                ) from exc
        # Only real members are replaced: a registered parameter, a registered
        # buffer, or a tensor stored in the owner's __dict__. A value provided
        # by __getattr__ (for example a lazily loaded ParameterManager tensor)
        # is no member functional_call can swap, so it is rejected here.
        members: Dict[str, Any] = dict(parameter_owner.named_parameters(prefix="", recurse=False))
        members.update(parameter_owner.named_buffers(prefix="", recurse=False))
        instance_attribute = vars(parameter_owner).get(leaf)
        if leaf in members:
            param_tensor = members[leaf]
        elif isinstance(instance_attribute, Tensor):
            param_tensor = instance_attribute
        elif leaf and leaf in vars(parameter_owner):
            raise TypeError(
                f"Parameter '{parameter_name}' of {self.process_name} is a "
                f"{type(instance_attribute).__name__}; sensitivity replaces a tensor member "
                "(nn.Parameter, buffer or tensor attribute) through functional_call"
            )
        elif type(parameter_owner).__getattr__ is not nn.Module.__getattr__:
            raise TypeError(
                f"Parameter '{parameter_name}' of {self.process_name} is not a real tensor "
                "member of the module: it is neither a registered parameter or buffer nor a "
                f"tensor in the module's __dict__, and {type(parameter_owner).__name__} provides "
                "attributes through __getattr__, which functional_call cannot replace"
            )
        else:
            raise KeyError(f"Parameter '{parameter_name}' not found in {self.process_name}")
        # A ParameterManager learnable binding held by reference has one owner,
        # the manager: its sensitivity is taken on the manager calibration path
        # (get_trainable_parameters), never by replacing it inside one Functor.
        from .functor.functor_base import FunctorBase

        if isinstance(parameter_owner, FunctorBase):
            held_key = parameter_owner.manager_binding_key(param_tensor)
            if held_key is not None:
                raise TypeError(
                    f"Parameter '{parameter_name}' of {self.process_name} is the "
                    f"ParameterManager learnable binding '{held_key}', held by reference; "
                    "its sensitivity goes through the manager calibration path "
                    "(get_trainable_parameters), not parameter_sensitivity"
                )

        # References: Press, Teukolsky, Vetterling and Flannery (1992),
        #   Numerical Recipes in C, 2nd ed., section 5.7; the zero-value floor
        #   follows the mixed-scale principle of Hairer, Norsett and Wanner
        #   (1993), Solving ODE I, section II.4.
        # Mathematical expression: default h = eps_m^(1/2) x_c (forward
        #   difference) or h = eps_m^(1/3) x_c (central difference); explicit
        #   h = max of the given bounds {eps_abs, eps_rel x_c}; x_c = |theta| where
        #   theta != 0 and x_floor where theta = 0; then h <- (theta + h) - theta
        #   so that theta + h and theta differ by an exactly representable step.
        # Parameter meanings: eps_m is the machine epsilon of theta's dtype
        #   (torch.finfo), x_c the curvature scale, x_floor the declared scale
        #   optimization.sensitivity.finite_difference.zero_parameter_scale,
        #   eps_abs and eps_rel the caller's optional explicit bounds.
        # Units: h, x_c, x_floor and eps_abs in theta units; eps_m and eps_rel
        #   dimensionless.
        # Domain/constraints: floating-point theta; an explicit bound takes
        #   precedence over the default; x_floor is read only when x_c is used
        #   and an element of theta is zero.
        # Sign convention: h > 0 for positive bounds.
        theta = param_tensor.detach()
        explicit_step = absolute_eps is not None or relative_eps is not None

        def curvature_scale() -> Tensor:
            magnitude = theta.abs()
            if not bool((magnitude == 0).any()):
                return magnitude
            from ..parameters import get_global_manager

            zero_scale = get_global_manager().get(
                "optimization.sensitivity.finite_difference.zero_parameter_scale",
                required=True,
            )
            return torch.where(
                magnitude > 0,
                magnitude,
                torch.as_tensor(float(zero_scale), device=theta.device, dtype=theta.dtype),
            )

        if explicit_step:
            bounds: List[Tensor] = []
            if absolute_eps is not None:
                bounds.append(
                    torch.as_tensor(absolute_eps, device=theta.device, dtype=theta.dtype).expand_as(theta)
                )
            if relative_eps is not None:
                bounds.append(
                    torch.as_tensor(relative_eps, device=theta.device, dtype=theta.dtype) * curvature_scale()
                )
            step = bounds[0] if len(bounds) == 1 else torch.maximum(bounds[0], bounds[1])
        else:
            machine_eps = torch.finfo(theta.dtype).eps
            order = 1.0 / 3.0 if use_central_difference else 0.5
            step = machine_eps ** order * curvature_scale()
        eps = (theta + step) - theta

        def numeric_output(value: Any) -> Tensor:
            item = StateManager._coerce_update_item(value)
            value = item.value if item is not None else None
            if isinstance(value, Tensor):
                return value
            if isinstance(value, (float, int)):
                return ensure_tensor(value, reference_tensor=param_tensor)
            raise TypeError("Sensitivity requires a numeric output or numeric UpdateItem")

        # References: Glasserman (2004), Monte Carlo Methods in Financial
        #   Engineering, section 7.1; L'Ecuyer and Perron (1994), Oper. Res. 42(4).
        # Mathematical expression: S = (r(theta + eps; U) - r(theta - eps; U)) / (2 eps),
        #   U the random numbers of the substream from its position omega_0 (the
        #   same U in every evaluation); after the call the position is omega_0.
        # Parameter meanings: omega_0 is the substream position (generator state,
        #   draw count, pending Wiener increments) before the call; U the draws.
        # Units: omega_0 and U are dimensionless; S as below.
        # Domain/constraints: only the Functor-owned substream (FunctorSubstreamMixin
        #   extra state) is restored; a deterministic Functor has no stream.
        # Sign convention: not applicable to the stream position.
        stream_position = self.get_extra_state() if isinstance(self, FunctorSubstreamMixin) else None

        # Invariant I1 (no side effects): a forward must not change the
        # Functor's parameters or buffers in place; state changes are committed
        # through StateManager. The version counter of every member is compared
        # around each evaluation. A change is reported, never undone, so the
        # defect stays visible.
        def member_versions() -> Dict[str, int]:
            members = dict(self.named_parameters(remove_duplicate=False))
            members.update(self.named_buffers(remove_duplicate=False))
            return {name: tensor._version for name, tensor in members.items() if tensor is not None}

        def evaluate_rate(param_val: Tensor) -> Tensor:
            """Evaluate the governed call with the member replaced for this call only."""
            if stream_position is not None:
                self.set_extra_state(stream_position)
            versions_before = member_versions()
            rates = functional_call(
                self,
                {parameter_name: param_val},
                args=(dict(state),),
                kwargs={"time": time, "dt": dt},
            )
            versions_after = member_versions()
            changed = sorted(
                name for name, version in versions_before.items() if versions_after.get(name) != version
            )
            if changed:
                raise RuntimeError(
                    f"{self.process_name}.forward changed the member(s) {changed} in place during "
                    "a sensitivity evaluation; this violates I1 (no side effects): commit such "
                    "changes as updates through StateManager"
                )
            scientific, diagnostics, _ = partition_outputs(rates)
            if use_target:
                if target_key in scientific:
                    return numeric_output(scientific[target_key])
                if target_key in diagnostics:
                    return numeric_output(diagnostics[target_key])
                raise KeyError(f"Sensitivity target '{target_key}' is not a numeric output")
            values = [numeric_output(value) for value in scientific.values()]
            if not values:
                raise ValueError("Sensitivity has no numeric scientific outputs")
            totals = [(value.square() if aggregation_method == "L2" else value).sum() for value in values]
            result = totals[0]
            for total in totals[1:]:
                result = result + total
            return result.sqrt() if aggregation_method == "L2" else result

        # References: Press et al. (2007), Numerical Recipes, section 5.7.
        # Mathematical expression: central S = (r(theta + eps) - r(theta - eps)) / (2 eps);
        #   forward S = (r(theta + eps) - r(theta)) / eps.
        # Parameter meanings: r is the aggregated output of the governed call,
        #   theta the selected member, eps the step of the expression above.
        # Units: S in output units per parameter unit.
        # Domain/constraints: the member is replaced only inside functional_call,
        #   so theta is unchanged after every evaluation (also on failure).
        # Sign convention: positive S means the output increases with theta.
        try:
            if use_central_difference:
                # Evaluate the explicitly requested central difference
                rate_plus = evaluate_rate(param_tensor + eps)
                rate_minus = evaluate_rate(param_tensor - eps)
                sensitivity = (rate_plus - rate_minus) / (2 * eps)

            else:
                # Use forward difference
                rate_current = evaluate_rate(param_tensor)
                rate_plus = evaluate_rate(param_tensor + eps)
                sensitivity = (rate_plus - rate_current) / eps

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            raise RuntimeError(f"Sensitivity calculation failed for {parameter_name}: {exc}") from exc
        finally:
            if stream_position is not None:
                self.set_extra_state(stream_position)

        return sensitivity

    def temperature_function(self, temperature: Tensor) -> Tensor:
        """
        Require the concrete biological Functor to own its response law.

        Parameters:
            temperature : Tensor
                Temperature values [K]

        Raises:
            RuntimeError: Always. The concrete BiologicalProcessFunctor must
                select and parameterize its response through the canonical
                bridge and ParameterManager lifecycle.

        References:
            van't Hoff (1898); Arrhenius (1889); Lloyd and Taylor (1994).
        Mathematical expressions:
            Q10: f(T) = q10^((T - T_ref) / delta_T).
            Arrhenius: f(T) = exp(Ea / R * (1 / T_ref - 1 / T)).
        Parameter meanings:
            q10 is dimensionless, T and T_ref are [K], delta_T is [K],
            Ea is [J mol-1], and R is [J mol-1 K-1].
        """
        del temperature
        raise RuntimeError(
            "Temperature response must be implemented by the concrete "
            "BiologicalProcessFunctor; BiologicalProcess exposes no generic "
            "scientific runtime."
        )

    def get_process_metadata(self) -> Dict[str, Any]:
        """
        Get process metadata for documentation and parameter tracking.

        Returns:
            Dict[str, Any]: Process metadata
        """
        return {
            'process_name': self.process_name,
            'allometric_scaling': self.allometric_scaling,
            'base_rate': self.base_rate,
            'temperature_optimum': self.temperature_optimum,
            'q10_value': self.q10_value,
            'ph_optimum': self.ph_optimum
        }


class TransportMechanism(ABC):
    """
    Transport Mechanism Abstract Base Class

    Defines spatial transport of matter, energy, and momentum including
    molecular diffusion, turbulent mixing, advective transport, root water
    and nutrient uptake, hydraulic flow, gas exchange, and mass transfer
        processes and managed boundary applications with conservation law
        enforcement and flux calculations.

    This class provides the fundamental interface for all transport processes
    that move substances between spatial locations, across interfaces, and
    between different phases or compartments in ecosystem models.

    Mathematical Expression:
        General transport equation: dC/dt + div(J) = S - R
        where C is concentration, J is flux vector, S is source, R is reaction

        Flux components:
        J = J_diffusive + J_advective + J_dispersive
        J_diffusive = -D grad(C)  [Fick's law]
        J_advective = v C    [convective transport]
        J_dispersive = -K grad(C) [turbulent/mechanical dispersion]

        Diffusion equation: D laplacian(C) where D is diffusivity tensor
        Advection equation: v . grad(C) where v is velocity field

        Boundary conditions:
        - Dirichlet: C|_Gamma = g(x,t)
        - Neumann: -D grad(C) . n|_Gamma = q(x,t)
        - Robin: alpha*C + beta*(-D grad(C) . n)|_Gamma = gamma(x,t)
        - Cauchy: C and -D grad(C) . n specified at inlet

        Conservation: d/dt Integral_V C dV + Integral_boundary J . n dA = Integral_V S dV

    Parameters:
        transport_type : str
            Descriptive label of the transport mechanism, one of
            ``TRANSPORT_TYPES``. ``managed_application`` represents an
            externally scheduled matter transfer across a domain boundary; it
            does not own downstream reaction kinetics. A label implies no
            coefficient.
        peclet_number : Optional[float]
            Peclet number Pe = vL/D characterizing transport regime [dimensionless]
        device : Optional[torch.device]
            Computation device

    References:
        - Bird, R.B. et al. (2007). Transport Phenomena. Wiley
        - Crank, J. (1975). The Mathematics of Diffusion. Oxford University Press
        - Bear, J. (1972). Dynamics of Fluids in Porous Media. Elsevier
        - Dagan, G. (1989). Flow and Transport in Porous Formations. Springer-Verlag
        - Raupach, M.R. & Thom, A.S. (1981). Turbulence in and above plant canopies
        - Campbell, G.S. & Norman, J.M. (1998). An Introduction to Environmental Biophysics
    """

    # Descriptive mechanism labels. A label implies no coefficient: the scalar
    # diffusivity is set only by a Functor whose law uses a constant D.
    TRANSPORT_TYPES: ClassVar[frozenset[str]] = frozenset({
        "diffusion",
        "advection",
        "advection_diffusion",
        "turbulent",
        "root_uptake",
        "gas_exchange",
        "boundary_layer_diffusion",
        "dry_deposition",
        "vertical_percolation",
        "capillary_rise",
        "solute_diffusion",
        "managed_application",
        "interphase_mass_transfer",
    })

    def __init__(
        self,
        transport_type: str,
        peclet_number: Optional[float] = None,
        device: Optional[Device] = None,
    ) -> None:
        """
        Initialize transport mechanism with its descriptive label.

        The scalar diffusivity is absent (None) until the concrete process,
        after reading its parameters, sets a constant D through the one
        validated setter ``set_scalar_diffusivity``; a process whose law has no
        constant D never sets it.

        Mathematical Expression:
            Dimensionless groups:
            Pe = vL/D (Peclet number - advection/diffusion)
            Re = _vL/_ (Reynolds number - inertial/viscous)
            Sc = _/D (Schmidt number - momentum/mass diffusion)
        """
        ABC.__init__(self)

        if not transport_type or not isinstance(transport_type, str):
            raise ValueError("Transport type must be a non-empty string")
        if transport_type not in self.TRANSPORT_TYPES:
            raise ValueError(f"Invalid transport type: {transport_type}")

        self.transport_type = transport_type
        self.diffusivity: Optional[Union[float, Tensor]] = None
        self.peclet_number = peclet_number
        self.device = device or torch.device("cpu")

        # Transport parameters. Medium properties such as tortuosity or porosity
        # are owned, validated and exported by the concrete process that uses them.
        self.velocity_field: Optional[Tensor] = None

    def set_scalar_diffusivity(self, value: Union[float, Tensor]) -> None:
        """The one validated setter of the constant scalar diffusivity.

        References: Crank (1975), The Mathematics of Diffusion, Ch. 1 (Fick's
            first law J = -D grad(c)).
        Mathematical expression: D is the constant coefficient of
            J = -D grad(c); the setter stores D unchanged after the checks.
        Parameter meanings: D the scalar diffusivity of this process.
        Units: D [m2 s-1].
        Domain/constraints: a real number or a one-element floating tensor,
            finite and > 0. A bool, None, a string or an integer tensor raises
            TypeError; a non-finite, non-positive or multi-element value raises
            ValueError. A tensor, including a live ParameterManager-owned
            nn.Parameter, is stored as the same object, so autograd and later
            parameter updates reach ``self.diffusivity`` and the law together.
        Sign convention: D > 0; the flux direction comes from -grad(c).
        """
        if isinstance(value, Tensor):
            if not (value.is_floating_point() and not value.is_complex()):
                raise TypeError(
                    f"diffusivity must be a real floating tensor, got dtype {value.dtype}"
                )
            if value.numel() != 1:
                raise ValueError(
                    f"diffusivity must be a scalar, got a tensor of shape {tuple(value.shape)}"
                )
            scalar = float(value.detach().reshape(()).item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            scalar = float(value)
        else:
            raise TypeError(
                f"diffusivity must be a real number or a one-element floating tensor, "
                f"got {type(value).__name__}"
            )
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"diffusivity must be finite and > 0 [m2 s-1], got {scalar}")
        self.diffusivity = value if isinstance(value, Tensor) else scalar

    @abstractmethod
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """
        Execute one transport-process step and return process outputs.

        This is the single public computation entry point for transport
        contracts. Concrete implementations should place all transport
        logic in ``forward``.
        """
        pass

    def _resolve_diffusivity_tensor(self, field_values: Tensor) -> Tensor:
        """Create diffusivity tensor aligned with boundary field tensor."""
        if self.diffusivity is None:
            raise RuntimeError(
                f"Transport type '{self.transport_type}' has no diffusivity"
            )
        return ensure_tensor(self.diffusivity, reference_tensor=field_values)

    def boundary_conditions(
        self,
        boundary_name: str,
        field_values: Tensor,
        normal_vector: Tensor,
        boundary_type: Optional[str] = None,
        boundary_value: Optional[float] = None,
        use_bc_manager: bool = True,
    ) -> Tensor:
        """
        Require the concrete transport process to own its boundary law.

        A boundary flux depends on the discretization of the concrete process
        (the normal distance from the adjacent cell centre to the face, the face
        coefficient, and the boundary types it supports), so the abstract
        contract provides no generic numerical law, following the
        ``compute_diffusivity_tensor`` precedent. A concrete Functor that
        exposes boundary fluxes overrides this method (for example
        LinearAdvection, FickianDiffusion, DryDepositionAggregator and
        EddyDiffusionBLMFunctor) and rejects unknown types and missing values.

        Parameters:
            boundary_name : str
                Boundary face ("x_min", ..., "z_max") or a name the concrete
                process maps to a face.
            field_values : Tensor
                Field values in the cells adjacent to the boundary [field units].
            normal_vector : Tensor
                Outward unit normal at the boundary [dimensionless].
            boundary_type : Optional[str]
                A boundary condition type the concrete process supports.
            boundary_value : Optional[float]
                Prescribed value or flux; never assumed to be zero.
            use_bc_manager : bool
                Concrete-process option; core resolves no boundary manager.

        Raises:
            RuntimeError: Always.

        References:
            Carslaw and Jaeger (1959), Conduction of Heat in Solids;
            LeVeque (2002), Finite Volume Methods for Hyperbolic Problems,
            boundary fluxes of a finite-volume cell.
        Mathematical expression (what a concrete override implements):
            Dirichlet: J_b = -D_b (g - C_P) / delta_n
            Neumann:   J_b = q
            Robin:     alpha C_b + beta J_b = gamma with
                       C_b = C_P - J_b delta_n / D_b, i.e.
                       J_b = (gamma - alpha C_P) / (beta - alpha delta_n / D_b)
        Parameter meanings:
            J_b outward boundary flux; D_b face diffusivity; g prescribed
            boundary value; C_P value in the adjacent cell; C_b boundary value;
            delta_n normal distance from the cell centre to the face; q
            prescribed outward flux; alpha, beta, gamma Robin coefficients
            declared by the process.
        Units: C_P, C_b, g [field units]; D_b [m2 s-1]; delta_n [m];
            J_b, q [field units m s-1]; alpha [1]; beta [s m-1];
            gamma [field units].
        Domain/constraints: D_b > 0 and delta_n > 0 where they appear; the
            boundary type and value are declared, never defaulted.
        Sign convention: J_b is positive along the outward normal (out of the
            domain).
        """
        del boundary_name, field_values, normal_vector
        del boundary_type, boundary_value, use_bc_manager
        raise RuntimeError(
            "Boundary fluxes must be implemented by the concrete transport "
            f"Functor ({type(self).__name__}); TransportMechanism exposes no "
            "generic boundary law."
        )

    def compute_diffusivity_tensor(
        self,
        temperature: Tensor,
        pressure: Tensor,
        medium_properties: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Require the concrete transport process to own its diffusivity law.

        Parameters:
            temperature : Tensor
                Temperature field [K]
            pressure : Tensor
                Pressure field [Pa]
            medium_properties : Optional[Dict[str, Tensor]]
                Medium properties (porosity, tortuosity, etc.)

        Raises:
            RuntimeError: Always. A concrete transport implementation must
                declare its correlation, parameters, units, and validity range.

        References:
            Massman (1998); Fuller, Schettler, and Giddings (1966).
        Mathematical expression:
            D_AB = D_ref * (T / T_ref)^n * (P_ref / P), where the
            temperature exponent n and porous-medium correction are specific
            to the gas pair and transport closure.
        Parameter meanings:
            D_ref is reference diffusivity [m2 s-1], T and T_ref are
            temperatures [K], P and P_ref are pressures [Pa], and n is a
            correlation-specific dimensionless exponent.
        """
        del temperature, pressure, medium_properties
        raise RuntimeError(
            "Diffusivity scaling must be implemented by the concrete transport "
            "Functor; TransportMechanism exposes no generic transport runtime."
        )

    def get_transport_metadata(self) -> Dict[str, Any]:
        """
        Get transport mechanism metadata.

        Returns:
            Dict[str, Any]: Transport metadata
        """
        return {
            'transport_type': self.transport_type,
            'diffusivity': self.diffusivity,
            'peclet_number': self.peclet_number,
            'has_velocity_field': self.velocity_field is not None
        }


class ScaleCoupling(ABC):
    """
    Scale Coupling Abstract Base Class (Simplified)

    Provides the minimal common interface for all scale coupling operations.
    Defines basic upscale/downscale transformations while delegating specialized
    functionality to dedicated subclasses.

    This simplified version removes the previous job overload and focuses solely
    on the core scale transformation interface, promoting better separation of
    concerns and maintainability.

    Mathematical Expression:
        Scale transformation: F: Omega_source -> Omega_target
        Upscaling: phi_coarse = _phi_fine_ (aggregation operator)
        Downscaling: phi_fine = D[phi_coarse] (disaggregation operator)

        Conservation constraint: Integral_Omega_target F[phi] dOmega approx
        Integral_Omega_source phi dOmega

    Parameters:
        source_scale : str
            Source scale identifier
        target_scale : str
            Target scale identifier
        coupling_type : str
            Type of scale coupling ("spatial", "temporal", "hierarchical")
        conservation_properties : List[str]
            List of conserved quantities
        device : Optional[torch.device]
            Computation device

    References:
        - Cushman, J.H. (1997). The Physics of Fluids in Hierarchical Porous Media
        - Whitaker, S. (1999). The Method of Volume Averaging
    """

    def __init__(
        self,
        source_scale: str,
        target_scale: str,
        coupling_type: str,
        conservation_properties: Optional[List[str]] = None,
        device: Optional[Device] = None,
    ) -> None:
        """
        Initialize scale coupling with source/target scales.

        Mathematical Expression:
            Scale mapping: F: Omega_source -> Omega_target
            Scale ratio: _ = L_target / L_source
        """
        ABC.__init__(self)

        if not source_scale or not isinstance(source_scale, str):
            raise ValueError("Source scale must be a non-empty string")
        if not target_scale or not isinstance(target_scale, str):
            raise ValueError("Target scale must be a non-empty string")

        valid_coupling_types = ["spatial", "temporal", "hierarchical", "stochastic"]
        if coupling_type not in valid_coupling_types:
            raise ValueError(f"Invalid coupling type: {coupling_type}")

        self.source_scale = source_scale
        self.target_scale = target_scale
        self.coupling_type = coupling_type
        self.conservation_properties = conservation_properties or ["mass"]
        self.device = device or torch.device("cpu")

    @abstractmethod
    def upscale(self, fine_data: Tensor) -> Tensor:
        """
        Transform data from fine to coarse scale.

        Parameters:
            fine_data : Tensor
                Fine-scale data to be upscaled
                Shape: [n_fine_elements] or [batch_size, n_fine_elements]

        Returns:
            Tensor: Coarse-scale data
                Shape: [n_coarse_elements] or [batch_size, n_coarse_elements]

        Mathematical Expression:
            phi_coarse = U[phi_fine] where U is upscaling operator
            Conservation: Integral phi_coarse dOmega approx Integral phi_fine dOmega
        """
        raise NotImplementedError(
            "ScaleCoupling requires an explicit upscale operator."
        )

    @abstractmethod
    def downscale(self, coarse_data: Tensor) -> Tensor:
        """
        Transform data from coarse to fine scale.

        Parameters:
            coarse_data : Tensor
                Coarse-scale data to be downscaled
                Shape: [n_coarse_elements] or [batch_size, n_coarse_elements]

        Returns:
            Tensor: Fine-scale data
                Shape: [n_fine_elements] or [batch_size, n_fine_elements]

        Mathematical Expression:
            phi_fine = D[phi_coarse] where D is downscaling operator
            Conservation: Integral phi_fine dOmega approx Integral phi_coarse dOmega
        """
        raise NotImplementedError(
            "ScaleCoupling requires an explicit downscale operator."
        )

    def get_coupling_metadata(self) -> Dict[str, Any]:
        """
        Get scale coupling metadata.

        Returns:
            Dict[str, Any]: Coupling metadata
        """
        return {
            'source_scale': self.source_scale,
            'target_scale': self.target_scale,
            'coupling_type': self.coupling_type,
            'conservation_properties': self.conservation_properties
        }

    def get_parameter_value(
        self,
        param_name: str,
    ) -> Any:
        """
        Get parameter value from config or parameter manager.

        This method provides a unified interface for parameter access
        across all scale coupling processes, supporting both local config
        and integration with the global parameter manager.

        Parameters:
            param_name : str
                Parameter name

        Returns:
            Any: Parameter value

        Raises:
            KeyError: If the parameter is found in neither source (there is
                no caller default).
        """
        explicit_namespace = getattr(self, "parameter_namespace", None)
        coupling_namespace = f"coupling.{self.coupling_type}"
        return _resolve_parameter_with_manager(
            self,
            param_name=param_name,
            namespace_candidates=(explicit_namespace, coupling_namespace),
        )


class SpatialScalingBase(ScaleCoupling):
    """
    Spatial Scaling Abstract Base Class

    Defines spatial scale transformation operations including upscaling,
    downscaling, and arbitrary grid regridding with conservation enforcement.
    Specialized for spatial resolution conversion between different grid systems.

    Mathematical Expression:
        Spatial averaging: _phi_(x_coarse) = Integral_V phi(x_fine) W(x_fine,
        x_coarse) dx_fine
        where W is spatial weighting function

        Regridding: phi_target = R[phi_source] where R preserves mass/energy
        Conservative remapping: Integral_Omega_target phi_target dOmega
        = Integral_Omega_source phi_source dOmega

        Grid transformation: x_target = f(x_source, grid_mapping)

    Parameters:
        source_grid : GridDescriptor
            Source spatial grid specification
        target_grid : GridDescriptor
            Target spatial grid specification
        method : str
            Default regridding method ("conservative", "bilinear", "nearest")
        preserve : str
            Conservation property to preserve ("mass", "energy", "none")

    References:
        - Jones, P.W. (1999). Conservative remapping algorithms
        - Ullrich, P.A. et al. (2009). Conservative transport on the sphere
    """

    def __init__(
        self,
        source_scale: str,
        target_scale: str,
        method: str,
        preserve: str,
        device: Optional[Device] = None,
    ) -> None:
        """Initialize spatial scaling with grid transformation parameters."""
        super().__init__(
            source_scale=source_scale,
            target_scale=target_scale,
            coupling_type="spatial",
            conservation_properties=[preserve] if preserve != "none" else [],
            device=device,
        )
        self.method = method
        self.preserve = preserve

    @abstractmethod
    def regrid(
        self,
        data: Tensor,
        source_grid: Any,  # GridDescriptor type
        target_grid: Any,  # GridDescriptor type
        method: str,
        preserve: str,
    ) -> Tensor:
        """
        Regrid data between arbitrary spatial grids.

        Performs conservative or interpolative remapping between different
        spatial discretizations while preserving specified quantities.

        Parameters:
            data : Tensor
                Data to regrid [n_source_cells]
            source_grid : GridDescriptor
                Source grid specification
            target_grid : GridDescriptor
                Target grid specification
            method : str
                Regridding method ("conservative", "bilinear", "nearest",
                "trilinear")
            preserve : str
                Quantity to preserve ("mass", "energy", "none")

        Returns:
            Tensor: Regridded data [n_target_cells]

        Mathematical Expression:
            Conservative: phi_target = Sum_source w_ij phi_source where Sum_i
            w_ij = A_j/A_total
            Bilinear: phi(x,y) = a_0_0 + a_1_0x + a_0_1y + a_1_1xy

            Conservation constraint: Integral_Omega_target phi_target dOmega
            = Integral_Omega_source phi_source dOmega
        """
        pass


class TemporalScalingBase(ScaleCoupling):
    """Abstract contract for bounded temporal scale transformations.

    References:
        - CF Conventions 1.13, Sections 4.4, 7.1, 7.3, and Appendix E.
        - Jones, P.W. (1999), Mon. Weather Rev. 127:2204-2210.

    Mathematical expression:
        ``O_ij = max(0, min(a_i+, b_j+) - max(a_i-, b_j-))``.
        ``delta_i = a_i+ - a_i-`` and ``Delta_j = b_j+ - b_j-``.
        For ``interval_mean``, the observed-support target value is the
        overlap-duration weighted mean. For ``rate``, overlap duration times
        rate yields an observed amount. For ``interval_accumulation``, source
        amounts are partitioned by ``O_ij / delta_i``. ``state_at_boundary``
        uses an explicit exact, left, or right boundary policy and no sum.

    Parameter meanings:
        ``a_i`` and ``b_j`` are explicit half-open source and target bounds;
        ``O_ij`` is overlap duration. ``source_dt`` and ``target_dt`` remain
        public resolution descriptors, not substitutes for explicit bounds.
        A profile or driver declares a reconstruction beyond the default
        piecewise-constant reconstruction.

    Units:
        Bounds and overlap use one convertible time unit. Mean/state outputs
        retain value units; rate integration and interval accumulation use an
        explicit mapped-measure unit contract.

    Domain/constraints:
        Bounds are finite, positive-width, ordered, non-overlapping half-open
        intervals with one declared calendar. Mapping residual, gap or
        imputation contribution, and physical ledger residual are distinct.
        Relative mapping residual uses a positive finite ``Q_ref`` from the
        parameter owner. Reconstruction/profile/RNG behavior is conditional on
        an explicit executable contract.

    Sign convention:
        Overlap durations are nonnegative. Mapping residual is target measure
        minus source measure on the same accepted support; gap is reported
        separately and must not be repaired into the output.
    """

    def __init__(
        self,
        source_scale: str,
        target_scale: str,
        method: str,
        pattern_type: str,
        device: Optional[Device] = None,
    ) -> None:
        """Initialize temporal scaling with time transformation parameters."""
        super().__init__(
            source_scale=source_scale,
            target_scale=target_scale,
            coupling_type="temporal",
            conservation_properties=["mass"] if method == "sum" else [],
            device=device
        )
        self.method = method
        self.pattern_type = pattern_type

    @abstractmethod
    def resample(
        self,
        data: Tensor,
        source_dt: float,
        target_dt: float,
        method: str,
        pattern: Optional[Tensor] = None
    ) -> Tensor:
        """Resample while honoring the bounded temporal contract.

        References:
            CF Conventions 1.13 and Jones (1999) conservative remapping.
        Mathematical expression:
            ``O_ij`` is the duration overlap of explicit half-open bounds;
            ``interval_mean``, ``rate``, and ``interval_accumulation`` use the
            conditional laws documented by :class:`TemporalScalingBase`.
            ``state_at_boundary`` uses an explicit boundary selection policy.
        Parameter meanings:
            ``data`` is source support; ``source_dt`` and ``target_dt`` are
            resolution descriptors; ``method`` selects a declared temporal
            semantics; ``pattern`` is an optional reconstruction/profile input.
        Units:
            Resolution descriptors use seconds. Executable value and mapped
            measure units are declared by the concrete runtime contract.
        Domain/constraints:
            Concrete implementations require explicit bounds/calendar for
            numerical execution and default to piecewise-constant
            reconstruction. Mapping residual, gap, and physical ledger checks
            remain separate and relative residual requires parameter-owned
            ``Q_ref``. Missing reconstruction or profile contracts fail closed.
        Sign convention:
            Positive overlap transfers source support to target support;
            mapping residual is target minus source and is never repaired.

        Returns:
            Tensor: Resampled values on the declared target support.
        """
        pass


class HierarchicalScalingBase(ScaleCoupling):
    """
    Hierarchical Scaling Abstract Base Class

    Defines hierarchical scale transformation operations between nested
    organizational levels (leaf -> plant -> canopy -> stand -> landscape).
    Specialized for multi-level aggregation with area/volume weighting.

    Mathematical Expression:
        Hierarchical aggregation: phi_level_j = Sum_i in level_j w_ij phi_level_i
        where w_ij are hierarchical weights (area, biomass, etc.)

        Multi-level conservation: Sum_leaf phi_leaf = Sum_landscape phi_landscape
        Weight constraint: Sum_i w_ij = 1 for each parent j

        Tree structure: parent(i) -> children(i) with weight mapping

    Concrete classes implement the level transfer through the ScaleCoupling contract
    (upscale/downscale) with an explicit spatial basis per variable; a basis-free
    multi-level API is not part of this contract.

    Parameters:
        hierarchy_levels : List[str]
            Ordered list of hierarchy levels
        weight_type : str
            Weighting scheme ("area", "biomass", "uniform", "custom")
        rule : str
            Aggregation rule ("conservative", "area_weighted", "mass_weighted")

    References:
        - O'Neill, R.V. et al. (1986). A Hierarchical Concept of Ecosystems
        - Urban, D.L. (2005). Modeling ecological processes across scales
    """

    def __init__(
        self,
        source_scale: str,
        target_scale: str,
        hierarchy_levels: List[str],
        weight_type: str,
        rule: str,
        device: Optional[Device] = None,
        name: Optional[str] = None,
        metadata: Optional['FunctorMetadata'] = None,
    ) -> None:
        """Initialize hierarchical scaling with level structure."""
        # Use direct ScaleCoupling initialization to avoid MRO re-entering
        # ScaleCouplingFunctor in multiple-inheritance bridges.
        ScaleCoupling.__init__(
            self,
            source_scale=source_scale,
            target_scale=target_scale,
            coupling_type="hierarchical",
            conservation_properties=["mass", "energy"],
            device=device,
        )
        self.hierarchy_levels = hierarchy_levels
        self.weight_type = weight_type
        self.rule = rule
        self.name = name or f"hierarchical_{source_scale}_to_{target_scale}"
        self.metadata = metadata


# Export all abstract base classes
__all__ = [
    "PhysicalField",
    "BiologicalProcess",
    "TransportMechanism",
    "ScaleCoupling",
    "SpatialScalingBase",
    "TemporalScalingBase",
    "HierarchicalScalingBase",
    "Tensor",
    "Device",
    "StateDict",
    "EnvironmentDict",
    "FluxDict",
    "DiagnosticsDict",
    # Tensor processing utilities
    "ensure_tensor",
    "safe_item",
    "safe_mean_item",
    "ensure_device_consistency",
    "ensure_dtype_consistency"
]
