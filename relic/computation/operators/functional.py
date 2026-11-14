"""
Transport Operators Module for RELIC

This module provides fundamental differential operators, grid utilities,
and numerical computation tools for transport processes including diffusion,
advection, and advection-diffusion equations. All operators use consistent
tensor shapes and units across the RELIC framework.

Mathematical Expressions:
    Gradient: nablaphi = (partialphi/partialx, partialphi/partialy, partialphi/partialz)
    Divergence: nabla*J = partialJx/partialx + partialJy/partialy + partialJz/partialz  
    Laplacian: nabla^2phi = partial^2phi/partialx^2 + partial^2phi/partialy^2 + partial^2phi/partialz^2
    
    Finite Difference Schemes:
    - Central: partialphi/partialx ~= (phi[i+1] - phi[i-1])/(2Deltax)
    - Forward: partialphi/partialx ~= (phi[i+1] - phi[i])/Deltax
    - Backward: partialphi/partialx ~= (phi[i] - phi[i-1])/Deltax
    
    Second derivatives: partial^2phi/partialx^2 ~= (phi[i+1] - 2phi[i] + phi[i-1])/Deltax^2

References:
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM
    - Strikwerda, J.C. (2004). Finite Difference Schemes and PDEs. SIAM  
    - Hirsch, C. (2007). Numerical Computation of Internal and External Flows
    - Morton, K.W. & Mayers, D.F. (2005). Numerical Solution of PDEs
"""

import logging
import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ...parameters import get_global_manager

logger = logging.getLogger(__name__)

# Type aliases for clarity
Tensor = torch.Tensor
GridTensor = Tensor  # Shape: (batch, nz, ny, nx) or reduced dimensions
FluxTensor = Tensor  # Shape: (batch, nz, ny, nx, ndim) for vector fields
GridSpacing = Union[float, Tuple[float, ...]]

# Stencil order of the 2-D/3-D central-difference gradient and divergence
# kernels. These kernels are defined as second-order schemes (their operator
# classes declare stencil radius 1 and read no order parameter), so the order
# is a property of the scheme, not a model parameter; it is passed explicitly
# to compute_gradient_1d, whose order argument has no default.
_CENTRAL_DIFFERENCE_ORDER = 2


def compute_gradient_1d(
    field: Tensor,
    dx: float,
    boundary_method: str,
    order: int
) -> Tensor:
    """
    Compute 1D spatial gradient using finite differences.
    
    Mathematical Expression:
        Central difference (2nd order): partialphi/partialx ~= (phi[i+1] - phi[i-1])/(2Deltax)
        Forward difference: partialphi/partialx ~= (phi[i+1] - phi[i])/Deltax
        Backward difference: partialphi/partialx ~= (phi[i] - phi[i-1])/Deltax
    
    Parameters:
        field : Tensor
            Input field tensor, shape (batch, nx) or (batch, nz, ny, nx)
            Units: [field_units] (e.g., kg m^-3 for concentration)
        dx : float
            Grid spacing in x-direction
            Units: [m]
        boundary_method : str
            Boundary treatment: "zero", "extrapolate", "periodic"; required
            (no default; CentralDifferenceGradient1D passes
            computation.operators.gradient.boundary_method)
        order : int
            Finite difference order (2, 4); required (no default;
            CentralDifferenceGradient1D passes computation.operators.gradient.order)
            
    Returns:
        gradient : Tensor
            Spatial gradient partialphi/partialx
            Shape: same as input field
            Units: [field_units m-1]
            
    References:
        - Fornberg, B. (1988). Finite difference formulas
        - LeVeque, R.J. (2007). Finite Difference Methods
    """
    if order not in (2, 4):
        raise ValueError("Finite-difference order must be 2 or 4")
    boundary_method = _normalize_transport_boundary_method(boundary_method)
    if boundary_method not in ("zero", "extrapolate", "periodic"):
        raise ValueError(
            "Gradient boundary method must be zero, extrapolate, or periodic"
        )
    dx = _normalize_transport_grid_spacing(dx, 1)[0]
    if field.dim() < 2:
        raise ValueError("Field must have at least 2 dimensions (batch, spatial)")
    
    # Get spatial dimension (last dimension)
    nx = field.size(-1)
    if nx < 3:
        raise ValueError("Need at least 3 grid points for gradient calculation")
    
    # Initialize gradient tensor
    grad = torch.zeros_like(field)
    
    if order == 2:
        # Central differences for interior points
        grad[..., 1:-1] = (field[..., 2:] - field[..., :-2]) / (2.0 * dx)
        
        # Handle boundaries
        if boundary_method == "zero":
            # Zero gradient at boundaries
            grad[..., 0] = 0.0
            grad[..., -1] = 0.0
        elif boundary_method == "extrapolate":
            # Forward/backward differences at boundaries
            grad[..., 0] = (field[..., 1] - field[..., 0]) / dx
            grad[..., -1] = (field[..., -1] - field[..., -2]) / dx
        elif boundary_method == "periodic":
            # Periodic boundary conditions
            grad[..., 0] = (field[..., 1] - field[..., -1]) / (2.0 * dx)
            grad[..., -1] = (field[..., 0] - field[..., -2]) / (2.0 * dx)
        
    elif order == 4:
        # Fourth-order central differences for interior points
        if nx < 5:
            raise ValueError("Need at least 5 grid points for 4th order gradient")
        if boundary_method == "periodic":
            # References: Fornberg (1988); LeVeque (2007).
            # Mathematical expression: f'_i = (-f_{i+2}+8f_{i+1}-8f_{i-1}+f_{i-2})/(12 dx).
            # Parameter meanings: f is periodic field data and dx is grid spacing [m].
            return (
                -torch.roll(field, shifts=-2, dims=-1)
                + 8.0 * torch.roll(field, shifts=-1, dims=-1)
                - 8.0 * torch.roll(field, shifts=1, dims=-1)
                + torch.roll(field, shifts=2, dims=-1)
            ) / (12.0 * dx)
        grad[..., 2:-2] = (-field[..., 4:] + 8*field[..., 3:-1] - 8*field[..., 1:-3] + field[..., :-4]) / (12.0 * dx)
        
        # Second-order for near-boundary points
        grad[..., 1] = (field[..., 2] - field[..., 0]) / (2.0 * dx)
        grad[..., -2] = (field[..., -1] - field[..., -3]) / (2.0 * dx)
        
        # Handle boundaries same as 2nd order
        if boundary_method == "extrapolate":
            grad[..., 0] = (field[..., 1] - field[..., 0]) / dx
            grad[..., -1] = (field[..., -1] - field[..., -2]) / dx
    
    return grad


def compute_gradient_2d(
    field: Tensor,
    dx: float,
    dy: float,
    boundary_method: str
) -> Tuple[Tensor, Tensor]:
    """
    Compute 2D spatial gradients (partialphi/partialx, partialphi/partialy) using finite differences.
    
    Mathematical Expression:
        nablaphi = (partialphi/partialx, partialphi/partialy)
        Central differences: partialphi/partialx ~= (phi[i+1,j] - phi[i-1,j])/(2Deltax)
                           partialphi/partialy ~= (phi[i,j+1] - phi[i,j-1])/(2Deltay)
    
    Parameters:
        field : Tensor
            Input field tensor, shape (batch, ny, nx) or (batch, nz, ny, nx)
            Units: [field_units]
        dx : float
            Grid spacing in x-direction [m]
        dy : float
            Grid spacing in y-direction [m] 
        boundary_method : str
            Boundary treatment method
            
    Returns:
        grad_x, grad_y : Tuple[Tensor, Tensor]
            Spatial gradients partialphi/partialx and partialphi/partialy
            Shape: same as input field
            Units: [field_units m-1]
    """
    if field.dim() < 3:
        raise ValueError("Field must have at least 3 dimensions for 2D gradients")
    
    # Compute x-gradient (last dimension)
    grad_x = compute_gradient_1d(field, dx, boundary_method, _CENTRAL_DIFFERENCE_ORDER)
    
    # Compute y-gradient (second-to-last dimension)
    # Transpose to put y-dimension last, compute gradient, transpose back
    field_t = field.transpose(-2, -1)  # swap ny and nx
    grad_y_t = compute_gradient_1d(field_t, dy, boundary_method, _CENTRAL_DIFFERENCE_ORDER)
    grad_y = grad_y_t.transpose(-2, -1)  # swap back
    
    return grad_x, grad_y


def compute_gradient_3d(
    field: Tensor,
    dx: float,
    dy: float, 
    dz: float,
    boundary_method: str
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute 3D spatial gradients (partialphi/partialx, partialphi/partialy, partialphi/partialz) using finite differences.
    
    Mathematical Expression:
        nablaphi = (partialphi/partialx, partialphi/partialy, partialphi/partialz)
    
    Parameters:
        field : Tensor
            Input field tensor, shape (batch, nz, ny, nx)
            Units: [field_units]
        dx, dy, dz : float
            Grid spacings [m]
        boundary_method : str
            Boundary treatment method
            
    Returns:
        grad_x, grad_y, grad_z : Tuple[Tensor, Tensor, Tensor]
            Spatial gradients
            Units: [field_units m-1]
    """
    if field.dim() != 4:
        raise ValueError("Field must have 4 dimensions (batch, nz, ny, nx) for 3D gradients")
    
    # x-gradient (last dimension)
    grad_x = compute_gradient_1d(field, dx, boundary_method, _CENTRAL_DIFFERENCE_ORDER)
    
    # y-gradient (second-to-last dimension)
    field_t = field.transpose(-2, -1)
    grad_y_t = compute_gradient_1d(field_t, dy, boundary_method, _CENTRAL_DIFFERENCE_ORDER)
    grad_y = grad_y_t.transpose(-2, -1)
    
    # z-gradient (third-to-last dimension)
    field_t = field.transpose(-3, -1)
    grad_z_t = compute_gradient_1d(field_t, dz, boundary_method, _CENTRAL_DIFFERENCE_ORDER)
    grad_z = grad_z_t.transpose(-3, -1)
    
    return grad_x, grad_y, grad_z


def compute_divergence_2d(
    flux_x: Tensor,
    flux_y: Tensor,
    dx: float,
    dy: float,
    boundary_method: str
) -> Tensor:
    """
    Compute 2D divergence nabla*J = partialJx/partialx + partialJy/partialy using finite differences.
    
    Mathematical Expression:
        nabla*J = partialJx/partialx + partialJy/partialy
        Central differences: partialJx/partialx ~= (Jx[i+1,j] - Jx[i-1,j])/(2Deltax)
    
    Parameters:
        flux_x : Tensor
            Flux component in x-direction, shape (batch, ny, nx) or (batch, nz, ny, nx)
            Units: [flux_units] (e.g., kg m-2 s^-1)
        flux_y : Tensor
            Flux component in y-direction, same shape as flux_x
            Units: [flux_units]
        dx, dy : float
            Grid spacings [m]
        boundary_method : str
            Boundary treatment method
            
    Returns:
        divergence : Tensor
            Divergence nabla*J
            Shape: same as input fluxes
            Units: [flux_units m-1] = [kg m^-3 s^-1] for mass flux
    """
    # Compute partial derivatives
    dJx_dx = compute_gradient_1d(flux_x, dx, boundary_method, _CENTRAL_DIFFERENCE_ORDER)
    dJy_dy_t = compute_gradient_1d(flux_y.transpose(-2, -1), dy, boundary_method, _CENTRAL_DIFFERENCE_ORDER)
    dJy_dy = dJy_dy_t.transpose(-2, -1)
    
    # Sum for divergence
    divergence = dJx_dx + dJy_dy
    
    return divergence


def compute_divergence_3d(
    flux_x: Tensor,
    flux_y: Tensor,
    flux_z: Tensor,
    dx: float,
    dy: float,
    dz: float,
    boundary_method: str
) -> Tensor:
    """
    Compute 3D divergence nabla*J = partialJx/partialx + partialJy/partialy + partialJz/partialz.
    
    Mathematical Expression:
        nabla*J = partialJx/partialx + partialJy/partialy + partialJz/partialz
    
    Parameters:
        flux_x, flux_y, flux_z : Tensor
            Flux components, shape (batch, nz, ny, nx)
            Units: [flux_units] 
        dx, dy, dz : float
            Grid spacings [m]
        boundary_method : str
            Boundary treatment method
            
    Returns:
        divergence : Tensor
            Divergence nabla*J
            Units: [flux_units m-1]
    """
    # Compute partial derivatives along each axis
    # For x-direction: operate along last dimension
    dJx_dx = compute_gradient_1d(flux_x, dx, boundary_method, order=_CENTRAL_DIFFERENCE_ORDER)
    
    # For y-direction: operate along second-to-last dimension 
    # Transpose to make y the last dimension, compute gradient, transpose back
    flux_y_transposed = flux_y.transpose(-2, -1)
    dJy_dy = compute_gradient_1d(flux_y_transposed, dy, boundary_method, order=_CENTRAL_DIFFERENCE_ORDER).transpose(-2, -1)
    
    # For z-direction: operate along third-to-last dimension
    # Transpose to make z the last dimension, compute gradient, transpose back
    flux_z_transposed = flux_z.transpose(-3, -1)
    dJz_dz = compute_gradient_1d(flux_z_transposed, dz, boundary_method, order=_CENTRAL_DIFFERENCE_ORDER).transpose(-3, -1)
    
    # Sum for divergence
    divergence = dJx_dx + dJy_dy + dJz_dz
    
    return divergence


def _normalize_transport_boundary_method(boundary_method: str) -> str:
    """
    Normalize boundary method aliases used by transport operators.

    Parameters:
        boundary_method : str
            User-provided boundary method name.

    Returns:
        str
            Normalized boundary method keyword.
    """
    normalized = str(boundary_method).strip().lower()
    return normalized


def _shift_along_axis(
    field: Tensor,
    direction: int,
    *,
    axis: int = -1,
    boundary_method: str,
    boundary_value: float = 0.0,
) -> Tensor:
    """
    Shift tensor values along one axis with explicit boundary treatment.

    Mathematical Expression:
        shift(+1): field[i] <- field[i-1]
        shift(-1): field[i] <- field[i+1]

    Boundary handling:
        periodic   : wrap-around
        extrapolate: linear extrapolation
        neumann/open: zero-gradient ghost cell
        zero/dirichlet: fixed boundary value

    Parameters:
        field : Tensor
            Input tensor.
        direction : int
            +1 for backward shift (i-1), -1 for forward shift (i+1).
        axis : int
            Axis to shift.
        boundary_method : str
            Boundary handling method; required (no default), passed on
            from the caller's governed boundary_method.
        boundary_value : float
            Boundary value for dirichlet/zero modes.
    """
    method = _normalize_transport_boundary_method(boundary_method)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")

    if method == "periodic":
        return torch.roll(field, shifts=direction, dims=axis)

    rank = field.dim()
    axis_resolved = axis if axis >= 0 else rank + axis
    if axis_resolved < 0 or axis_resolved >= rank:
        raise ValueError(f"axis {axis} out of bounds for tensor rank {rank}")

    index_first = [slice(None)] * rank
    index_second = [slice(None)] * rank
    index_last = [slice(None)] * rank
    index_prev_last = [slice(None)] * rank
    index_first[axis_resolved] = slice(0, 1)
    index_second[axis_resolved] = slice(1, 2)
    index_last[axis_resolved] = slice(-1, None)
    index_prev_last[axis_resolved] = slice(-2, -1)

    first = field[tuple(index_first)]
    last = field[tuple(index_last)]

    if method == "extrapolate":
        if field.size(axis_resolved) < 2:
            left_ghost = first
            right_ghost = last
        else:
            second = field[tuple(index_second)]
            prev_last = field[tuple(index_prev_last)]
            left_ghost = 2.0 * first - second
            right_ghost = 2.0 * last - prev_last
    elif method in ("neumann", "open"):
        left_ghost = first
        right_ghost = last
    elif method in ("zero", "dirichlet"):
        left_ghost = torch.full_like(first, float(boundary_value))
        right_ghost = torch.full_like(last, float(boundary_value))
    else:
        raise ValueError(
            f"Unsupported boundary_method '{boundary_method}' for conservative shift"
        )

    if direction == 1:
        left_pad = left_ghost
        if axis_resolved == rank - 1:
            core = field[..., :-1]
            return torch.cat([left_pad, core], dim=axis_resolved)
        idx_core = [slice(None)] * rank
        idx_core[axis_resolved] = slice(0, -1)
        core = field[tuple(idx_core)]
        return torch.cat([left_pad, core], dim=axis_resolved)

    right_pad = right_ghost
    if axis_resolved == rank - 1:
        core = field[..., 1:]
        return torch.cat([core, right_pad], dim=axis_resolved)
    idx_core = [slice(None)] * rank
    idx_core[axis_resolved] = slice(1, None)
    core = field[tuple(idx_core)]
    return torch.cat([core, right_pad], dim=axis_resolved)


def _compute_conservative_divergence_1d(
    flux_face: Tensor,
    dx: float,
    boundary_method: str,
) -> Tensor:
    """
    Compute conservative 1D divergence from face fluxes.

    Mathematical Expression (finite-volume):
        div(F)_i = (F_{i+1/2} - F_{i-1/2}) / dx

    Parameters:
        flux_face : Tensor
            Face-centered flux represented on cell index i as F_{i+1/2}.
        dx : float
            Grid spacing.
        boundary_method : str
            Boundary treatment for F_{i-1/2} reconstruction; required (no
            default), passed on from compute_conservative_divergence_2d/3d.
    """
    flux_im1 = _shift_along_axis(
        flux_face,
        1,
        axis=-1,
        boundary_method=boundary_method,
        boundary_value=0.0,
    )
    return (flux_face - flux_im1) / dx


def compute_conservative_divergence_2d(
    flux_x: Tensor,
    flux_y: Tensor,
    dx: float,
    dy: float,
    boundary_method: str,
) -> Tensor:
    """
    Compute conservative 2D divergence from face fluxes.

    Mathematical Expression:
        div(F) = (F_x(i+1/2,j) - F_x(i-1/2,j))/dx
               + (F_y(i,j+1/2) - F_y(i,j-1/2))/dy

    Conservation Property (periodic domain):
        Sum_ij div(F)_ij * dx * dy = 0
        Interior face fluxes cancel pairwise (telescoping sum).

    Parameters:
        flux_x : Tensor
            Face flux in x-direction, indexed as F_x(i+1/2,j).
        flux_y : Tensor
            Face flux in y-direction, indexed as F_y(i,j+1/2).
        dx, dy : float
            Grid spacings in x and y.
        boundary_method : str
            Boundary treatment.

    References:
        - LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems.
        - Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow.
    """
    div_x = _compute_conservative_divergence_1d(flux_x, dx, boundary_method)
    div_y_t = _compute_conservative_divergence_1d(
        flux_y.transpose(-2, -1),
        dy,
        boundary_method,
    )
    div_y = div_y_t.transpose(-2, -1)
    return div_x + div_y


def compute_conservative_divergence_3d(
    flux_x: Tensor,
    flux_y: Tensor,
    flux_z: Tensor,
    dx: float,
    dy: float,
    dz: float,
    boundary_method: str,
) -> Tensor:
    """
    Compute conservative 3D divergence from face fluxes.

    Mathematical Expression:
        div(F) = dF_x/dx + dF_y/dy + dF_z/dz
        using finite-volume face differences in each direction.

    Parameters:
        flux_x, flux_y, flux_z : Tensor
            Face-centered flux tensors.
        dx, dy, dz : float
            Grid spacings.
        boundary_method : str
            Boundary treatment.

    References:
        - LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems.
        - Thomas, J.W. (1995). Numerical Partial Differential Equations.
    """
    div_x = _compute_conservative_divergence_1d(flux_x, dx, boundary_method)
    div_y = _compute_conservative_divergence_1d(
        flux_y.transpose(-2, -1),
        dy,
        boundary_method,
    ).transpose(-2, -1)
    div_z = _compute_conservative_divergence_1d(
        flux_z.transpose(-3, -1),
        dz,
        boundary_method,
    ).transpose(-3, -1)
    return div_x + div_y + div_z


def compute_laplacian_1d(
    field: Tensor,
    dx: float,
    boundary_method: str
) -> Tensor:
    """
    Compute 1D Laplacian nabla^2phi = partial^2phi/partialx^2 using finite differences.
    
    Mathematical Expression:
        nabla^2phi = partial^2phi/partialx^2 ~= (phi[i+1] - 2phi[i] + phi[i-1])/Deltax^2
    
    Parameters:
        field : Tensor
            Input field tensor
            Units: [field_units]
        dx : float
            Grid spacing [m]
        boundary_method : str
            Boundary treatment method
            
    Returns:
        laplacian : Tensor
            Second derivative partial^2phi/partialx^2
            Units: [field_units m-2]
    """
    boundary_method = _normalize_transport_boundary_method(boundary_method)
    if boundary_method not in ("zero", "extrapolate", "periodic"):
        raise ValueError(
            "Laplacian boundary method must be zero, extrapolate, or periodic"
        )
    dx = _normalize_transport_grid_spacing(dx, 1)[0]
    if field.size(-1) < 3:
        raise ValueError("Need at least 3 grid points for Laplacian calculation")
    
    laplacian = torch.zeros_like(field)
    
    # Central differences for interior points
    laplacian[..., 1:-1] = (field[..., 2:] - 2*field[..., 1:-1] + field[..., :-2]) / (dx * dx)
    
    # Handle boundaries
    if boundary_method == "zero":
        # Zero-gradient (Neumann) boundary using reflected ghost cells:
        # phi[-1] = phi[1], phi[N] = phi[N-2] => laplacian[0] = 2*(phi[1]-phi[0])/dx^2
        laplacian[..., 0] = 2.0 * (field[..., 1] - field[..., 0]) / (dx * dx)
        laplacian[..., -1] = 2.0 * (field[..., -2] - field[..., -1]) / (dx * dx)
    elif boundary_method == "extrapolate":
        # Linear extrapolation (second derivative = 0)
        laplacian[..., 0] = 0.0
        laplacian[..., -1] = 0.0
    elif boundary_method == "periodic":
        # Periodic boundary conditions
        laplacian[..., 0] = (field[..., 1] - 2*field[..., 0] + field[..., -1]) / (dx * dx)
        laplacian[..., -1] = (field[..., 0] - 2*field[..., -1] + field[..., -2]) / (dx * dx)
    
    return laplacian


def compute_second_derivative_1d(
    field: Tensor,
    dx: float,
    axis: int = -1,
    *,
    boundary_method: str
) -> Tensor:
    """
    Compute 1D second derivative partial^2phi/partialx^2 along specified axis using finite differences.
    
    Mathematical Expression:
        partial^2phi/partialx^2 ~= (phi[i+1] - 2phi[i] + phi[i-1])/Deltax^2
    
    Parameters:
        field : Tensor
            Input field tensor
            Units: [field_units]
        dx : float
            Grid spacing [m]
        axis : int
            Axis along which to compute derivative
        boundary_method : str
            Boundary treatment method
            
    Returns:
        second_derivative : Tensor
            Second derivative partial^2phi/partialx^2
            Units: [field_units m-2]
    """
    boundary_method = _normalize_transport_boundary_method(boundary_method)
    if boundary_method not in ("zero", "extrapolate", "periodic"):
        raise ValueError(
            "Second-derivative boundary method must be zero, extrapolate, or periodic"
        )
    dx = _normalize_transport_grid_spacing(dx, 1)[0]
    if field.size(axis) < 3:
        raise ValueError("Need at least 3 grid points for second derivative calculation")
    
    # Move the specified axis to the last position
    field_moved = field.moveaxis(axis, -1)
    second_deriv = torch.zeros_like(field_moved)
    
    # Central differences for interior points
    second_deriv[..., 1:-1] = (field_moved[..., 2:] - 2*field_moved[..., 1:-1] + field_moved[..., :-2]) / (dx * dx)
    
    # Handle boundaries
    if boundary_method == "zero":
        # References: LeVeque (2007) finite differences with zero-gradient boundaries.
        # Mathematical expression: f_pp(0) = (f1 - 2*f0 + f_minus1)/dx^2 with f_minus1=f1.
        # Parameter meanings: f is field, dx is grid spacing.
        second_deriv[..., 0] = (field_moved[..., 1] - 2 * field_moved[..., 0] + field_moved[..., 1]) / (dx * dx)
        second_deriv[..., -1] = (field_moved[..., -2] - 2 * field_moved[..., -1] + field_moved[..., -2]) / (dx * dx)
    elif boundary_method == "extrapolate":
        # Linear extrapolation (second derivative = 0)
        second_deriv[..., 0] = 0.0
        second_deriv[..., -1] = 0.0
    elif boundary_method == "periodic":
        # Periodic boundary conditions
        second_deriv[..., 0] = (field_moved[..., 1] - 2*field_moved[..., 0] + field_moved[..., -1]) / (dx * dx)
        second_deriv[..., -1] = (field_moved[..., 0] - 2*field_moved[..., -1] + field_moved[..., -2]) / (dx * dx)
    
    # Move axis back to original position
    return second_deriv.moveaxis(-1, axis)


def compute_laplacian_2d(
    field: Tensor,
    dx: float,
    dy: float,
    boundary_method: str
) -> Tensor:
    """
    Compute 2D Laplacian nabla^2phi = partial^2phi/partialx^2 + partial^2phi/partialy^2.
    
    Mathematical Expression:
        nabla^2phi = partial^2phi/partialx^2 + partial^2phi/partialy^2
    
    Parameters:
        field : Tensor
            Input field tensor, shape (batch, ny, nx) or (batch, nz, ny, nx)
            Units: [field_units]
        dx, dy : float
            Grid spacings [m]
        boundary_method : str
            Boundary treatment method
            
    Returns:
        laplacian : Tensor
            2D Laplacian
            Units: [field_units m-2]
    """
    # Compute second derivatives in each direction separately
    d2_dx2 = compute_second_derivative_1d(field, dx, axis=-1, boundary_method=boundary_method)  # x-direction
    d2_dy2 = compute_second_derivative_1d(field, dy, axis=-2, boundary_method=boundary_method)  # y-direction
    
    laplacian = d2_dx2 + d2_dy2
    
    return laplacian


def compute_laplacian_3d(
    field: Tensor,
    dx: float,
    dy: float,
    dz: float,
    boundary_method: str
) -> Tensor:
    """
    Compute 3D Laplacian nabla^2phi = partial^2phi/partialx^2 + partial^2phi/partialy^2 + partial^2phi/partialz^2.
    
    Mathematical Expression:
        nabla^2phi = partial^2phi/partialx^2 + partial^2phi/partialy^2 + partial^2phi/partialz^2
    
    Parameters:
        field : Tensor
            Input field tensor, shape (batch, nz, ny, nx)
            Units: [field_units]
        dx, dy, dz : float
            Grid spacings [m]
        boundary_method : str
            Boundary treatment method
            
    Returns:
        laplacian : Tensor
            3D Laplacian
            Units: [field_units m-2]
    """
    if field.dim() != 4:
        raise ValueError("Field must have 4 dimensions for 3D Laplacian")
    
    # Compute second derivatives in each direction separately
    d2_dx2 = compute_second_derivative_1d(field, dx, axis=-1, boundary_method=boundary_method)  # x-direction
    d2_dy2 = compute_second_derivative_1d(field, dy, axis=-2, boundary_method=boundary_method)  # y-direction  
    d2_dz2 = compute_second_derivative_1d(field, dz, axis=-3, boundary_method=boundary_method)  # z-direction
    
    laplacian = d2_dx2 + d2_dy2 + d2_dz2
    
    return laplacian


def compute_grid_metrics(
    spatial_dims: Tuple[int, ...],
    grid_spacing: Union[float, Tuple[float, ...]],
    device: Optional[torch.device] = None
) -> Dict[str, Tensor]:
    """
    Compute grid metrics and spacing tensors for transport calculations.
    
    Parameters:
        spatial_dims : Tuple[int, ...]
            Spatial grid dimensions (nx,) for 1D, (ny, nx) for 2D, (nz, ny, nx) for 3D
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing in each direction [m]
            If float: uniform spacing in all directions
            If tuple: (dx,) for 1D, (dy, dx) for 2D, (dz, dy, dx) for 3D
        device : Optional[torch.device]
            Computation device
            
    Returns:
        metrics : Dict[str, Tensor]
            Grid metrics including:
            - 'dx', 'dy', 'dz': Grid spacings [m]
            - 'volumes': Cell volumes [m^3] for 3D, areas [m^2] for 2D, lengths [m] for 1D
            - 'face_areas': Face areas for flux calculations
            
    References:
        - Ferziger, J.H. & Peric, M. (2002). Computational Methods for Fluid Dynamics
    """
    if device is None:
        raise ValueError("compute_grid_metrics requires an explicit device")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")

    ndim = len(spatial_dims)
    if ndim not in (1, 2, 3) or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in spatial_dims
    ):
        raise ValueError(
            "spatial dimensions must contain one to three strictly positive integers"
        )

    spacing = list(_normalize_transport_grid_spacing(grid_spacing, ndim))
    
    metrics = {}
    
    if ndim == 1:
        nx = spatial_dims[0]
        dx = spacing[0]
        
        metrics['dx'] = torch.as_tensor(dx, device=device)
        metrics['volumes'] = torch.full((nx,), dx, device=device)  # Cell lengths
        metrics['face_areas'] = torch.ones((nx+1,), device=device)  # Unit area for 1D
        
    elif ndim == 2:
        ny, nx = spatial_dims
        dy, dx = spacing
        
        metrics['dx'] = torch.as_tensor(dx, device=device)
        metrics['dy'] = torch.as_tensor(dy, device=device)
        
        # Cell areas
        metrics['volumes'] = torch.full((ny, nx), dx * dy, device=device)
        
        # Face areas
        metrics['face_areas_x'] = torch.full((ny, nx+1), dy, device=device)  # x-faces
        metrics['face_areas_y'] = torch.full((ny+1, nx), dx, device=device)  # y-faces
        
    elif ndim == 3:
        nz, ny, nx = spatial_dims
        dz, dy, dx = spacing
        
        metrics['dx'] = torch.as_tensor(dx, device=device)
        metrics['dy'] = torch.as_tensor(dy, device=device)
        metrics['dz'] = torch.as_tensor(dz, device=device)
        
        # Cell volumes
        metrics['volumes'] = torch.full((nz, ny, nx), dx * dy * dz, device=device)
        
        # Face areas
        metrics['face_areas_x'] = torch.full((nz, ny, nx+1), dy * dz, device=device)  # x-faces
        metrics['face_areas_y'] = torch.full((nz, ny+1, nx), dx * dz, device=device)  # y-faces
        metrics['face_areas_z'] = torch.full((nz+1, ny, nx), dx * dy, device=device)  # z-faces
        
    else:
        raise ValueError(f"Unsupported spatial dimensions: {ndim}")
    
    return metrics


def compute_cfl_number(
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    time_step: float
) -> Tensor:
    """
    Compute Courant-Friedrichs-Lewy (CFL) number for stability analysis.
    
    Mathematical Expression:
        CFL = |v| * Deltat / Deltax
        For multi-dimensional: CFL = Deltat * (|u|/Deltax + |v|/Deltay + |w|/Deltaz)
        Stability requires: CFL <= 1.0 for explicit advection schemes
    
    Parameters:
        velocity : Union[Tensor, Tuple[Tensor, ...]]
            Velocity field(s)
            If Tensor: 1D velocity [m s^-1]
            If Tuple: (u, v) for 2D or (u, v, w) for 3D [m s^-1]
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing(s) [m]
        time_step : float
            Time step [s]
            
    Returns:
        cfl_number : Tensor
            CFL number field [dimensionless]
            
    References:
        - Courant, R. et al. (1928). On the partial difference equations of mathematical physics
        - Holton, J.R. (2012). An Introduction to Dynamic Meteorology
    """
    dt_value = float(time_step)
    if not math.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("time_step must be finite and positive")

    if isinstance(velocity, torch.Tensor):
        if not bool(torch.all(torch.isfinite(velocity))):
            raise ValueError("velocity must contain only finite values")
        if isinstance(grid_spacing, (int, float)):
            spacing = float(grid_spacing)
        else:
            if len(grid_spacing) != 1:
                raise ValueError(
                    "Tensor velocity requires exactly one grid-spacing axis"
                )
            spacing = float(grid_spacing[-1])
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("grid spacing values must be finite and positive")
        cfl = torch.abs(velocity) * dt_value / spacing

    elif isinstance(velocity, (tuple, list)):
        if len(velocity) == 0:
            raise ValueError("velocity must contain at least one axis")
        if isinstance(grid_spacing, (int, float)):
            spacing_by_velocity = (float(grid_spacing),) * len(velocity)
        else:
            if len(grid_spacing) != len(velocity):
                raise ValueError(
                    "velocity and grid_spacing must provide the same number of axes"
                )
            # Velocity is ordered (u, v, w); tensor storage is (..., z, y, x).
            spacing_by_velocity = tuple(
                float(value) for value in reversed(grid_spacing)
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in spacing_by_velocity
        ):
            raise ValueError("grid spacing values must be finite and positive")

        cfl = torch.zeros_like(velocity[0])
        for vel_component, spacing in zip(velocity, spacing_by_velocity):
            if not bool(torch.all(torch.isfinite(vel_component))):
                raise ValueError("velocity must contain only finite values")
            try:
                cfl = cfl + torch.abs(vel_component) * dt_value / spacing
            except RuntimeError as exc:
                raise ValueError(
                    "velocity components must be broadcast-compatible"
                ) from exc

    else:
        raise ValueError("Velocity must be Tensor or tuple/list of Tensors")
    
    return cfl


def compute_diffusion_stability(
    diffusivity: Tensor,
    grid_spacing: Union[float, Tuple[float, ...]],
    time_step: float
) -> Tensor:
    """
    Compute diffusion stability number for explicit diffusion schemes.
    
    Mathematical Expression:
        Stability number: sigma = D * Deltat / Deltax^2
        For multi-dimensional: sigma = D * Deltat * (1/Deltax^2 + 1/Deltay^2 + 1/Deltaz^2)
        Stability requires: sigma <= 0.5 for explicit diffusion schemes
    
    Parameters:
        diffusivity : Tensor
            Diffusion coefficient field [m^2 s^-1]
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing(s) [m]
        time_step : float
            Time step [s]
            
    Returns:
        stability_number : Tensor
            Diffusion stability number [dimensionless]
            
    References:
        - Strikwerda, J.C. (2004). Finite Difference Schemes and PDEs
        - Morton, K.W. & Mayers, D.F. (2005). Numerical Solution of PDEs
    """
    if not isinstance(diffusivity, Tensor) or not diffusivity.is_floating_point():
        raise TypeError("diffusivity must use a real floating-point tensor")
    if not bool(torch.all(torch.isfinite(diffusivity))):
        raise ValueError("diffusivity must contain only finite values")
    if bool(torch.any(diffusivity < 0.0)):
        raise ValueError("diffusivity must be non-negative")

    dt_value = float(time_step)
    if not math.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("time_step must be finite and positive")
    spatial_dims = 1 if isinstance(grid_spacing, (int, float)) else len(grid_spacing)
    if spatial_dims not in (1, 2, 3):
        raise ValueError(f"Unsupported grid spacing dimensions: {spatial_dims}")
    spacing_values = _normalize_transport_grid_spacing(grid_spacing, spatial_dims)
    inverse_spacing_squared = sum(1.0 / (value * value) for value in spacing_values)
    stability = diffusivity * dt_value * inverse_spacing_squared
    if not bool(torch.all(torch.isfinite(stability))):
        raise ValueError("diffusion stability number must contain only finite values")
    return stability


def check_conservation(
    field_old: Tensor,
    field_new: Tensor,
    source_term: Optional[Tensor] = None,
    time_step: Optional[float] = None,
    cell_volumes: Optional[Tensor] = None,
    tolerance: Optional[float] = None,
    warn_on_violation: bool = True
) -> Tensor:
    """
    Check mass conservation for transport step.
    
    References:
        - Byun, D.W. et al. Numerical Transport Algorithms for CMAQ.
    
    Mathematical Expression:
        Conservation: integral(phi_new - phi_old)dV = Deltat integral S dV
        where S is source term
    
    Parameters:
        field_old : Tensor
            Field values at previous time step
            Units: [kg m^-3] for concentration
        field_new : Tensor
            Field values at current time step
            Units: [kg m^-3]
        source_term : Optional[Tensor]
            Source/sink term [kg m^-3 s^-1]
        time_step : Optional[float]
            Time step [s]
        cell_volumes : Optional[Tensor]
            Cell volumes [m^3]
        tolerance : Optional[float]
            Relative tolerance for conservation check; None reads
            ParameterManager key computation.operators.diagnostics.tolerance
        warn_on_violation : bool
            Emit warning when conservation error exceeds tolerance
            
    Returns:
        conservation_error : Tensor
            Relative conservation error [dimensionless]
    """
    if tolerance is None:
        from relic.parameters import get_global_manager

        tolerance = get_global_manager().get(
            "computation.operators.diagnostics.tolerance", required=True
        )
    # Compute total mass change
    if cell_volumes is not None:
        mass_old = torch.sum(field_old * cell_volumes)
        mass_new = torch.sum(field_new * cell_volumes) 
        mass_change = mass_new - mass_old
    else:
        mass_old = torch.sum(field_old)
        mass_new = torch.sum(field_new)
        mass_change = mass_new - mass_old
    
    # Expected mass change from sources
    if source_term is not None and time_step is not None:
        if cell_volumes is not None:
            expected_change = time_step * torch.sum(source_term * cell_volumes)
        else:
            expected_change = time_step * torch.sum(source_term)
    else:
        expected_change = mass_change.new_tensor(0.0)
    
    # References: Higham, N.J. (2002), Accuracy and Stability of Numerical
    #   Algorithms, 2nd ed., SIAM, sec. 4.2 (summation bound n u sum|x_i|).
    # Mathematical expression: error = |dM - E| / max(|M_old| + n eps S, tiny),
    #   S = sum|phi_old dV| + sum|phi_new dV| + |E|.
    # Parameter meanings: dM mass change, E = dt integral(S dV) expected change,
    #   n number of summed cells, eps and tiny the machine epsilon and smallest
    #   normal number of the field dtype.
    # Units: dM, E, M_old, and S in field-integral (mass) units; error is
    #   dimensionless.
    # Domain/constraints: a numerical roundoff guard, not a model parameter;
    #   because |dM - E| <= S, all-zero fields give error 0 and never nan, and
    #   a cancelling signed field with M_old = 0 still reports a large error.
    # Sign convention: error is a nonnegative magnitude.
    if cell_volumes is not None:
        mass_scale = torch.sum(torch.abs(field_old * cell_volumes)) + torch.sum(
            torch.abs(field_new * cell_volumes)
        )
    else:
        mass_scale = torch.sum(torch.abs(field_old)) + torch.sum(torch.abs(field_new))
    mass_scale = mass_scale + torch.abs(expected_change)
    dtype_info = torch.finfo(mass_old.dtype)
    roundoff_guard = dtype_info.eps * field_old.numel() * mass_scale
    denominator = torch.clamp(torch.abs(mass_old) + roundoff_guard, min=dtype_info.tiny)
    conservation_error = torch.abs(mass_change - expected_change) / denominator
    
    # Warning if conservation violated
    if warn_on_violation and conservation_error > tolerance:
        warnings.warn(f"Conservation error {conservation_error:.2e} exceeds tolerance {tolerance}")
    
    return conservation_error


def apply_non_negative_constraint(
    field: Tensor,
    method: str,
    small_value: Optional[float],
) -> Tensor:
    """
    Apply a non-negativity floor to a field.

    Parameters:
        field : Tensor
            Field values that should be non-negative
            Units: [field_units]
        method : str
            "clip" or "zero" (no default; NonNegativeConstraint reads
            computation.operators.diagnostics.method)
        small_value : Optional[float]
            Floor of the "clip" method, in field units; required for "clip"
            (the caller passes computation.operators.diagnostics.small_value)
            and must be None for "zero", whose floor is read here from
            transport.numerical.non_negative_min_value

    Returns:
        constrained_field : Tensor
            Field clamped from below at the method's floor
            Units: [field_units]

    Raises:
        ValueError: for an unknown method, a missing clip floor, or a floor
            passed to the "zero" method
    """
    # References:
    #   Hundsdorfer, W. and Verwer, J.G. (2003). Numerical Solution of
    #   Time-Dependent Advection-Diffusion-Reaction Equations. Springer,
    #   Sect. I.7 (positivity and clipping of transported concentrations).
    # Mathematical expression:
    #   c_out = max(c_in, f),  f = small_value (clip) or
    #   f = transport.numerical.non_negative_min_value (zero)
    # Parameter meanings:
    #   c_in: field before the constraint; c_out: constrained field; f: floor
    #   of the selected method (clip: strictly positive floor; zero: plain
    #   non-negativity bound).
    # Units:
    #   c_in, c_out and f in the units of the field.
    # Domain/constraints:
    #   f >= 0 as bounded in YAML; no Python default for method or floor;
    #   values below f are raised to f (the removed mass is not tracked here).
    # Sign convention:
    #   None; the operation only raises values below the floor.
    if method == "clip":
        if small_value is None:
            raise ValueError(
                "method 'clip' requires small_value "
                "(computation.operators.diagnostics.small_value)"
            )
        return torch.clamp(field, min=float(small_value))
    if method == "zero":
        if small_value is not None:
            raise ValueError(
                "method 'zero' does not take small_value; its floor is "
                "transport.numerical.non_negative_min_value"
            )
        clamp_min = float(
            get_global_manager().get(
                "transport.numerical.non_negative_min_value",
                expected_type=float,
                required=True,
            )
        )
        return torch.clamp(field, min=clamp_min)
    raise ValueError(f"Unknown constraint method: {method}")


def harmonic_mean(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute harmonic mean of two tensors.
    
    Mathematical Expression:
        H = 2 * a * b / (a + b)
    
    Parameters:
        a : Tensor
            First value
        b : Tensor
            Second value
            
    Returns:
        h_mean : Tensor
            Harmonic mean
    """
    if not bool(torch.all(torch.isfinite(a))) or not bool(torch.all(torch.isfinite(b))):
        raise ValueError("harmonic mean inputs must contain only finite values")
    a_value, b_value = torch.broadcast_tensors(a, b)
    denominator = a_value + b_value
    zero_denominator = denominator == 0.0
    undefined = zero_denominator & ((a_value != 0.0) | (b_value != 0.0))
    if bool(torch.any(undefined)):
        raise ValueError(
            "harmonic mean is undefined when nonzero inputs sum to zero"
        )
    safe_denominator = torch.where(
        zero_denominator,
        torch.ones_like(denominator),
        denominator,
    )
    value = 2.0 * a_value * b_value / safe_denominator
    return torch.where(zero_denominator, torch.zeros_like(value), value)


# High-level transport operator suite for unified use across all spheres
# These functions provide the main API that all sphere-specific transport 
# modules should use instead of implementing their own operators


_TIME_INTEGRATION_METHOD_KEY = "transport.numerical.time_integration_method"


def _transport_time_integration_methods() -> Tuple[str, ...]:
    """Return the accepted transport time integration methods.

    The allowed_values of the YAML ParameterSpec of
    transport.numerical.time_integration_method are the single source of the
    accepted names; they are read through the manager's get_parameter spec
    path and never copied into code.

    Raises:
        KeyError: If the key is missing from the parameter database.
        ValueError: If the spec declares no allowed_values.
    """
    spec = get_global_manager().get_parameter(_TIME_INTEGRATION_METHOD_KEY)
    if spec is None:
        raise KeyError(f"Required parameter '{_TIME_INTEGRATION_METHOD_KEY}' not found")
    if not spec.allowed_values:
        raise ValueError(
            f"{_TIME_INTEGRATION_METHOD_KEY} declares no allowed_values in the "
            "YAML parameter database"
        )
    return tuple(str(value).strip().lower() for value in spec.allowed_values)


def _resolve_transport_time_integration(
    method: Optional[str],
    max_iter: Optional[int],
    tolerance: Optional[float],
) -> Tuple[str, int, float]:
    """Resolve transport time integration controls from args or parameter DB.

    The method must be one of the allowed_values of the YAML spec of
    transport.numerical.time_integration_method.
    """
    resolved_method = (
        str(method).strip().lower()
        if method is not None
        else str(
            get_global_manager().get(
                _TIME_INTEGRATION_METHOD_KEY,
                expected_type=str,
                required=True,
            )
        ).strip().lower()
    )
    accepted_methods = _transport_time_integration_methods()
    if resolved_method not in accepted_methods:
        raise ValueError(
            f"Unsupported time integration method '{resolved_method}'. "
            f"Supported methods (allowed_values of {_TIME_INTEGRATION_METHOD_KEY}): "
            f"{', '.join(accepted_methods)}"
        )

    resolved_max_iter = (
        int(max_iter)
        if max_iter is not None
        else int(
            get_global_manager().get(
                "transport.numerical.implicit_midpoint_max_iter",
                expected_type=int,
                required=True,
            )
        )
    )
    resolved_tolerance = (
        float(tolerance)
        if tolerance is not None
        else float(
            get_global_manager().get(
                "transport.numerical.implicit_midpoint_tolerance",
                expected_type=float,
                required=True,
            )
        )
    )
    return resolved_method, resolved_max_iter, resolved_tolerance


def _integrate_rate_system(
    state: Tensor,
    dt: float,
    rate_fn: Any,
    method: str,
    max_iter: int,
    tolerance: float,
) -> Tensor:
    """
    Integrate a rate-form PDE semi-discretization for one time step.

    Mathematical Expressions:
        Euler:
            y_{n+1} = y_n + dt * f(y_n)
        Implicit midpoint:
            y_{n+1} = y_n + dt * f((y_n + y_{n+1}) / 2)
        Crank-Nicolson:
            y_{n+1} = y_n + dt/2 * (f(y_n) + f(y_{n+1}))

    Parameters:
        state : Tensor
            Current state y_n.
        dt : float
            Time step.
        rate_fn : Callable
            Function f(y) returning dy/dt.
        method : str
            Integration method; one of the allowed_values of the YAML spec of
            transport.numerical.time_integration_method. "euler" is evaluated
            here; every other accepted name is the registered temporal solver
            of the same name (SolverFactory), which fails fast if absent.
        max_iter : int
            Maximum fixed-point iterations for implicit methods.
        tolerance : float
            Infinity-norm convergence tolerance.
    """
    accepted_methods = _transport_time_integration_methods()
    if method not in accepted_methods:
        raise ValueError(
            f"Unsupported integration method '{method}'; supported methods "
            f"(allowed_values of {_TIME_INTEGRATION_METHOD_KEY}): "
            f"{', '.join(accepted_methods)}"
        )
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    dt_tensor = torch.tensor(dt, device=state.device, dtype=state.dtype)
    if method == "euler":
        rate = rate_fn(state)
        if not bool(torch.all(torch.isfinite(rate))):
            raise ValueError("Euler rate contains non-finite values")
        solution = state + dt_tensor * rate
        if not bool(torch.all(torch.isfinite(solution))):
            raise ValueError("Euler solution contains non-finite values")
        return solution

    # Reuse the canonical temporal solver registry and preserve SolverResult status.
    from ..solvers.base import SolverStatus
    from ..solvers.factory import SolverFactory

    solver = SolverFactory.create(
        solver_name=method,
        name=f"transport_{method}",
        config={
            "max_iterations": int(max_iter),
            "rtol": float(tolerance),
            "atol": float(tolerance),
        },
        device=state.device,
        dtype=state.dtype,
        validate_parameters=False,
    )

    def _ode_rate(local_state: Tensor, _time: float) -> Tensor:
        local_rate = rate_fn(local_state)
        if not bool(torch.all(torch.isfinite(local_rate))):
            raise ValueError(f"{method} rate contains non-finite values")
        return local_rate

    solver_result = solver.solve(
        _ode_rate,
        state,
        t_current=0.0,
        dt=float(dt),
    )
    if solver_result.status is not SolverStatus.SUCCESS:
        raise RuntimeError(
            f"{method} did not converge after {solver_result.iterations} "
            f"iterations; residual={solver_result.residual_norm}; dt={float(dt):.6g}"
        )
    solution = solver_result.solution
    if not isinstance(solution, Tensor) or not bool(
        torch.all(torch.isfinite(solution))
    ):
        raise RuntimeError(f"{method} returned a non-finite tensor solution")
    return solution


def _compute_diffusion_rate_and_boundary_flux_1d(
    concentration: Tensor,
    diffusivity: Tensor,
    dx: float,
    boundary_method: str,
    boundary_state: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return a finite-volume diffusion rate and outward boundary fluxes.

    References:
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        LeVeque (2002), Finite Volume Methods for Hyperbolic Problems.
    Mathematical expression:
        dc_i/dt = (G_{i+1/2} - G_{i-1/2}) / dx,
        G = K dc/dx, and j_out = -G n_x.
    Parameter meanings:
        concentration is c [state_unit], diffusivity is K [m2 s-1], dx is
        cell width [m], boundary_state is the physical boundary-face value,
        and returned outward fluxes have [state_unit m s-1]. For a uniform
        cell-centered grid, a boundary face is dx/2 from the adjacent center.
    Open/closed system route:
        Dirichlet and extrapolated boundaries return explicit outward fluxes.
        Zero, Neumann, and open boundaries impose zero diffusive flux.
    """

    if concentration.shape != diffusivity.shape:
        raise ValueError("diffusivity must have the same shape as concentration")
    if concentration.size(-1) < 2:
        raise ValueError("Diffusion requires at least two cells along each axis")

    method = _normalize_transport_boundary_method(boundary_method)
    supported = {
        "periodic",
        "zero",
        "neumann",
        "open",
        "extrapolate",
        "dirichlet",
    }
    if method not in supported:
        raise ValueError(
            f"Unsupported boundary_method '{boundary_method}' for diffusion"
        )
    if method == "dirichlet" and boundary_state is None:
        raise ValueError("Dirichlet diffusion requires explicit boundary_state")

    interior_conductivity = harmonic_mean(
        diffusivity[..., :-1], diffusivity[..., 1:]
    )
    interior_gradient_flux = (
        interior_conductivity
        * (concentration[..., 1:] - concentration[..., :-1])
        / dx
    )

    if method == "periodic":
        right_concentration = torch.roll(concentration, shifts=-1, dims=-1)
        right_diffusivity = torch.roll(diffusivity, shifts=-1, dims=-1)
        right_flux = (
            harmonic_mean(diffusivity, right_diffusivity)
            * (right_concentration - concentration)
            / dx
        )
        left_flux = torch.roll(right_flux, shifts=1, dims=-1)
        return (right_flux - left_flux) / dx, {}

    zero_flux_boundary = method in ("zero", "neumann", "open")
    if zero_flux_boundary:
        left_boundary_flux = torch.zeros_like(concentration[..., :1])
        right_boundary_flux = torch.zeros_like(concentration[..., -1:])
    elif method == "extrapolate":
        left_ghost = 2.0 * concentration[..., :1] - concentration[..., 1:2]
        right_ghost = (
            2.0 * concentration[..., -1:] - concentration[..., -2:-1]
        )
        left_boundary_flux = (
            diffusivity[..., :1] * (concentration[..., :1] - left_ghost) / dx
        )
        right_boundary_flux = (
            diffusivity[..., -1:]
            * (right_ghost - concentration[..., -1:])
            / dx
        )
    else:
        explicit_boundary = boundary_state.to(
            device=concentration.device,
            dtype=concentration.dtype,
        )
        try:
            explicit_boundary = torch.broadcast_to(
                explicit_boundary, concentration.shape
            )
        except RuntimeError as exc:
            raise ValueError(
                "boundary_state must be broadcast-compatible with concentration"
            ) from exc
        boundary_distance = 0.5 * dx
        left_boundary_flux = (
            diffusivity[..., :1]
            * (concentration[..., :1] - explicit_boundary[..., :1])
            / boundary_distance
        )
        right_boundary_flux = (
            diffusivity[..., -1:]
            * (explicit_boundary[..., -1:] - concentration[..., -1:])
            / boundary_distance
        )

    right_flux = torch.cat(
        [interior_gradient_flux, right_boundary_flux], dim=-1
    )
    left_flux = torch.cat(
        [left_boundary_flux, interior_gradient_flux], dim=-1
    )
    rate = (right_flux - left_flux) / dx
    return rate, {
        "min": left_boundary_flux.squeeze(-1),
        "max": -right_boundary_flux.squeeze(-1),
    }


DIFFUSIVITY_LAYOUTS = ("scalar_field", "directional")


def _scalar_field_reading(
    diffusivity: Tensor, concentration: Tensor, spatial_dims: int
) -> Optional[Tuple[Tensor, ...]]:
    """K as one scalar coefficient (or cell field) shared by every axis, or None."""
    if diffusivity.numel() == 1:
        coefficient = diffusivity.expand_as(concentration)
    else:
        try:
            coefficient = torch.broadcast_to(diffusivity, concentration.shape)
        except RuntimeError:
            return None
    return tuple(coefficient for _ in range(spatial_dims))


def _directional_reading(
    diffusivity: Tensor, concentration: Tensor, spatial_dims: int
) -> Optional[Tuple[Tensor, ...]]:
    """K as storage-ordered per-axis values (Kz, Ky, Kx) or fields, or None."""
    if diffusivity.ndim == 1 and diffusivity.numel() == spatial_dims:
        return tuple(
            torch.ones_like(concentration) * diffusivity[axis_index]
            for axis_index in range(spatial_dims)
        )
    if diffusivity.ndim >= 1 and diffusivity.shape[0] == spatial_dims:
        fields = []
        for axis_index in range(spatial_dims):
            try:
                fields.append(
                    torch.broadcast_to(diffusivity.select(0, axis_index), concentration.shape)
                )
            except RuntimeError:
                return None
        return tuple(fields)
    return None


def _normalize_directional_diffusivity(
    diffusivity: Tensor,
    concentration: Tensor,
    spatial_dims: int,
    layout: Optional[str] = None,
) -> Tuple[Tensor, ...]:
    """Return one cell-centered coefficient field per storage axis.

    References:
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        FiPy finite-volume diffusion discretization documentation.
    Mathematical expression:
        dc/dt = sum_k d/dx_k(K_k dc/dx_k), with diagonal K_k.
    Parameter meanings:
        diffusivity is a 0-d scalar, a scalar cell field ("scalar_field"), or a
        storage-ordered vector (Kz, Ky, Kx) or storage-ordered directional
        cell fields ("directional") [m2 s-1]. concentration is c [state_unit].
        layout declares which reading applies.
    Domain/constraints:
        A tensor coefficient with ndim >= 1 must declare its layout, even
        when only one reading fits its shape: the shape never selects a
        reading. A 0-d coefficient is a scalar and takes the "scalar_field"
        reading when no layout is given. A declared layout must fit the shape.
    Open/closed system route:
        This normalization changes only the constitutive coefficient; boundary
        exchange remains owned by the oriented face-flux ledger.
    """
    if layout is None:
        if diffusivity.ndim != 0:
            raise ValueError(
                f"diffusivity of shape {tuple(diffusivity.shape)} needs "
                "diffusivity_layout='scalar_field' or 'directional'; the layout of a "
                "tensor coefficient is never inferred from its shape"
            )
        layout = "scalar_field"
    if layout not in DIFFUSIVITY_LAYOUTS:
        raise ValueError(
            f"Unknown diffusivity_layout '{layout}'; expected one of {DIFFUSIVITY_LAYOUTS}"
        )
    if layout == "scalar_field":
        reading = _scalar_field_reading(diffusivity, concentration, spatial_dims)
    else:
        reading = _directional_reading(diffusivity, concentration, spatial_dims)
    if reading is None:
        raise ValueError(
            f"diffusivity of shape {tuple(diffusivity.shape)} is not a valid "
            f"'{layout}' coefficient for concentration of shape "
            f"{tuple(concentration.shape)}"
        )
    return reading


def _prepare_diffusion_components(
    concentration: Tensor,
    diffusivity_by_storage_axis: Tuple[Tensor, ...],
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
    boundary_method: str,
    boundary_state: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Assemble directional finite-volume diffusion from the shared 1D law."""

    if isinstance(grid_spacing, (int, float)):
        spacing = (float(grid_spacing),) * spatial_dims
    else:
        spacing = tuple(float(value) for value in grid_spacing)

    dx = spacing[-1]
    rate_x, boundary_x = _compute_diffusion_rate_and_boundary_flux_1d(
        concentration,
        diffusivity_by_storage_axis[-1],
        dx,
        boundary_method,
        boundary_state,
    )
    total_rate = rate_x
    boundary_fluxes: Dict[str, Tensor] = {}
    if boundary_x:
        boundary_fluxes.update(
            {"x_min": boundary_x["min"], "x_max": boundary_x["max"]}
        )
    if spatial_dims == 1:
        return total_rate, boundary_fluxes

    dy = spacing[-2]
    boundary_state_y = (
        boundary_state.transpose(-2, -1) if boundary_state is not None else None
    )
    rate_y_t, boundary_y = _compute_diffusion_rate_and_boundary_flux_1d(
        concentration.transpose(-2, -1),
        diffusivity_by_storage_axis[-2].transpose(-2, -1),
        dy,
        boundary_method,
        boundary_state_y,
    )
    total_rate = total_rate + rate_y_t.transpose(-2, -1)
    if boundary_y:
        boundary_fluxes.update(
            {"y_min": boundary_y["min"], "y_max": boundary_y["max"]}
        )
    if spatial_dims == 2:
        return total_rate, boundary_fluxes

    dz = spacing[-3]
    concentration_z = torch.movedim(concentration, -3, -1)
    diffusivity_z = torch.movedim(diffusivity_by_storage_axis[-3], -3, -1)
    boundary_state_z = (
        torch.movedim(boundary_state, -3, -1)
        if boundary_state is not None
        else None
    )
    rate_z_t, boundary_z = _compute_diffusion_rate_and_boundary_flux_1d(
        concentration_z,
        diffusivity_z,
        dz,
        boundary_method,
        boundary_state_z,
    )
    total_rate = total_rate + torch.movedim(rate_z_t, -1, -3)
    if boundary_z:
        boundary_fluxes.update(
            {"z_min": boundary_z["min"], "z_max": boundary_z["max"]}
        )
    return total_rate, boundary_fluxes


def _normalize_transport_grid_spacing(
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
) -> Tuple[float, ...]:
    """Return finite positive grid spacing in storage-axis order."""
    if isinstance(grid_spacing, (int, float)):
        spacing_values = (float(grid_spacing),) * spatial_dims
    else:
        if len(grid_spacing) != spatial_dims:
            raise ValueError(
                f"grid spacing must provide {spatial_dims} values for "
                f"{spatial_dims}D transport"
            )
        spacing_values = tuple(float(value) for value in grid_spacing)
    if any(not math.isfinite(value) or value <= 0.0 for value in spacing_values):
        raise ValueError("grid spacing values must be finite and positive")
    return spacing_values


def _combine_boundary_flux_ledgers(
    *weighted_ledgers: Tuple[float, Dict[str, Tensor]],
) -> Dict[str, Tensor]:
    """Combine outward-positive boundary fluxes using time-step weights."""
    face_names = sorted(
        {
            face_name
            for _, ledger in weighted_ledgers
            for face_name in ledger
        }
    )
    combined: Dict[str, Tensor] = {}
    for face_name in face_names:
        sample = next(
            ledger[face_name]
            for _, ledger in weighted_ledgers
            if face_name in ledger
        )
        total = torch.zeros_like(sample)
        for weight, ledger in weighted_ledgers:
            if face_name in ledger:
                total = total + float(weight) * ledger[face_name]
        combined[face_name] = total
    return combined


def pack_outward_boundary_flux_cochain(
    boundary_fluxes: Dict[str, Tensor],
    *,
    squeeze_batch: bool = False,
) -> Tensor:
    """Pack an oriented finite-volume face ledger without merging batches.

    References:
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        LeVeque (2002), Finite Volume Methods for Hyperbolic Problems.
    Mathematical expression:
        q_boundary = [q_xmin, q_xmax, q_ymin, q_ymax, q_zmin, q_zmax].
    Parameter meanings:
        Each q_face is an outward-positive face flux with leading batch axis
        and units [state_unit m s-1]. The returned dual cochain has shape
        [batch, total_face_dofs] and the same units.
    Open/closed system route:
        Every supplied boundary face is retained exactly once; an empty ledger
        is rejected because closed domains must omit the optional port.
    """

    if not boundary_fluxes:
        raise ValueError("boundary_fluxes must contain at least one face")
    face_order = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    unknown_faces = set(boundary_fluxes).difference(face_order)
    if unknown_faces:
        raise ValueError(
            f"Unknown boundary flux faces: {sorted(unknown_faces)}"
        )
    ordered_faces = [face for face in face_order if face in boundary_fluxes]
    first = boundary_fluxes[ordered_faces[0]]
    if not isinstance(first, torch.Tensor) or first.ndim < 1:
        raise ValueError("Boundary face fluxes must have a leading batch axis")
    batch_size = first.shape[0]
    packed_faces = []
    for face in ordered_faces:
        value = boundary_fluxes[face]
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise ValueError("Boundary face fluxes must have a leading batch axis")
        if value.shape[0] != batch_size:
            raise ValueError("Boundary face fluxes must share one batch size")
        packed_faces.append(value.reshape(batch_size, -1))
    packed = torch.cat(packed_faces, dim=-1)
    return packed.squeeze(0) if squeeze_batch else packed


def _finite_volume_transport_conservation_error(
    field_old: Tensor,
    field_new: Tensor,
    source_rate: Tensor,
    time_step: float,
    spacing_values: Tuple[float, ...],
    spatial_dims: int,
    boundary_fluxes: Dict[str, Tensor],
    *,
    warn_on_violation: bool,
) -> Tensor:
    """Evaluate the explicit finite-volume storage, source, and port ledger.

    References:
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        Hundsdorfer and Verwer (2003), Time-Dependent ADR Equations.

    Mathematical Expression:
        Delta M - dt * integral(S dV) + dt * sum_f(j_out,f A_f) = 0.

    Parameter meanings:
        M is stored tracer mass, S is the volumetric source rate, j_out,f is
        the outward-positive face flux, A_f is face area, and dt is time [s].

    Open/closed system route:
        Nonzero boundary_fluxes are explicit boundary exports or imports;
        periodic and zero-flux closures have a zero boundary ledger.
    """
    cell_volume = math.prod(spacing_values)
    spatial_axes = tuple(range(field_old.dim() - spatial_dims, field_old.dim()))
    mass_old = torch.sum(field_old, dim=spatial_axes) * cell_volume
    mass_new = torch.sum(field_new, dim=spatial_axes) * cell_volume
    mass_scale_old = torch.sum(torch.abs(field_old), dim=spatial_axes) * cell_volume
    mass_scale_new = torch.sum(torch.abs(field_new), dim=spatial_axes) * cell_volume
    source_input = torch.sum(source_rate, dim=spatial_axes) * cell_volume

    spacing_by_axis = {
        "x": spacing_values[-1],
        "y": spacing_values[-2] if spatial_dims >= 2 else None,
        "z": spacing_values[-3] if spatial_dims >= 3 else None,
    }
    outward_rate = torch.zeros_like(mass_old)
    for face_name, face_flux in boundary_fluxes.items():
        axis_name = face_name.split("_", maxsplit=1)[0]
        if axis_name not in spacing_by_axis or spacing_by_axis[axis_name] is None:
            raise ValueError(f"Unknown boundary-flux face '{face_name}'")
        face_area = cell_volume / float(spacing_by_axis[axis_name])
        if spatial_dims == 1:
            face_total = face_flux
        else:
            face_axes = tuple(
                range(
                    face_flux.dim() - (spatial_dims - 1),
                    face_flux.dim(),
                )
            )
            face_total = torch.sum(face_flux, dim=face_axes)
        outward_rate = outward_rate + face_total * face_area

    dt_value = float(time_step)
    source_change = dt_value * source_input
    boundary_change = dt_value * outward_rate
    ledger_residual = mass_new - mass_old - source_change + boundary_change
    denominator = torch.maximum(
        torch.maximum(mass_scale_old, mass_scale_new),
        torch.maximum(
            torch.maximum(torch.abs(source_change), torch.abs(boundary_change)),
            torch.full_like(mass_old, torch.finfo(mass_old.dtype).eps),
        ),
    )
    conservation_error = torch.max(torch.abs(ledger_residual) / denominator)
    tolerance = float(
        get_global_manager().get(
            "transport.numerical.conservation_tolerance",
            expected_type=float,
            required=True,
        )
    )
    conservation_error_value = float(conservation_error.detach().item())
    if warn_on_violation and conservation_error_value > tolerance:
        warnings.warn(
            "Conservation error "
            f"{conservation_error_value:.2e} exceeds tolerance {tolerance}"
        )
    return conservation_error


def apply_diffusion(
    concentration: Tensor,
    diffusivity: Union[float, Tensor],
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
    boundary_method: str,
    time_step: Optional[float],
    check_stability: bool,
    enforce_non_negative: bool,
    run_conservation_check: bool = True,
    time_integration_method: Optional[str] = None,
    implicit_midpoint_max_iter: Optional[int] = None,
    implicit_midpoint_tolerance: Optional[float] = None,
    boundary_state: Optional[Tensor] = None,
    source_term: Optional[Tensor] = None,
    diffusivity_layout: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a conservative diffusion operator to a concentration field.

    References:
        Crank (1975), The Mathematics of Diffusion.
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.

    Mathematical Expression:
        dc/dt = div(K*grad(c)) + S
        - Scalar, cell-field, and diagonal tensor K use one conservative
          finite-volume face law.
        - Explicit stability: max(K) * dt * sum_k(1/dx_k^2) <= sigma_max.

    Parameter meanings:
        concentration is the transported state [state_unit], diffusivity is K
        [m2 s-1] in storage-axis order (Kz, Ky, Kx) when directional,
        grid_spacing contains dx_k [m], source_term is S [state_unit s-1],
        and time_step is dt [s]. diffusivity_layout is "scalar_field" or
        "directional"; it is required for a tensor K with ndim >= 1 (the
        shape never selects a reading) and optional for a number or a 0-d
        tensor, which is a scalar.

    Open/closed system route:
        Periodic, zero-flux, Neumann, and open boundaries are closed to
        diffusion. Dirichlet and extrapolated boundaries return explicit
        outward boundary fluxes for the mass ledger.

    Time Integration:
        - euler (default): explicit Euler.
        - implicit_midpoint: fixed-point midpoint solve.
        - crank_nicolson: fixed-point trapezoidal solve.
    """
    if spatial_dims not in (1, 2, 3):
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")
    if concentration.dim() < spatial_dims:
        raise ValueError(
            "Concentration rank is incompatible with spatial_dims: "
            f"rank={concentration.dim()}, spatial_dims={spatial_dims}"
        )
    has_explicit_batch = concentration.dim() > spatial_dims
    if not has_explicit_batch:
        concentration = concentration.unsqueeze(0)
    if not concentration.is_floating_point():
        raise TypeError("concentration must use a floating-point dtype")
    if not bool(torch.all(torch.isfinite(concentration))):
        raise ValueError("concentration must contain only finite values")
    if enforce_non_negative and bool(torch.any(concentration < 0.0)):
        raise ValueError(
            "concentration must be non-negative when enforce_non_negative is enabled"
        )

    spacing_values = _normalize_transport_grid_spacing(grid_spacing, spatial_dims)
    if time_step is not None and (
        not math.isfinite(float(time_step)) or float(time_step) <= 0.0
    ):
        raise ValueError("time_step must be finite and positive")

    boundary_method = _normalize_transport_boundary_method(boundary_method)
    if source_term is None:
        source_term_tensor = torch.zeros_like(concentration)
    else:
        if not isinstance(source_term, Tensor) or not source_term.is_floating_point():
            raise TypeError("source_term must use a real floating-point tensor")
        if source_term.is_complex():
            raise TypeError("source_term must use real floating-point values")
        source_term_tensor = source_term.to(
            device=concentration.device,
            dtype=concentration.dtype,
        )
        try:
            source_term_tensor = torch.broadcast_to(
                source_term_tensor,
                concentration.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "source_term must be broadcast-compatible with concentration"
            ) from exc
        if not bool(torch.all(torch.isfinite(source_term_tensor))):
            raise ValueError("source_term must contain only finite values")

    boundary_state_tensor = None
    if boundary_state is not None:
        if boundary_method != "dirichlet":
            raise ValueError(
                "boundary_state is only supported for Dirichlet diffusion"
            )
        boundary_state_tensor = torch.as_tensor(
            boundary_state,
            device=concentration.device,
        )
        if boundary_state_tensor.is_complex():
            raise TypeError("boundary_state must use real floating-point values")
        boundary_state_tensor = boundary_state_tensor.to(dtype=concentration.dtype)
        try:
            boundary_state_tensor = torch.broadcast_to(
                boundary_state_tensor, concentration.shape
            )
        except RuntimeError as exc:
            raise ValueError(
                "boundary_state must be broadcast-compatible with concentration"
            ) from exc
        if not bool(torch.all(torch.isfinite(boundary_state_tensor))):
            raise ValueError("boundary_state must contain only finite values")
        if enforce_non_negative and bool(torch.any(boundary_state_tensor < 0.0)):
            raise ValueError(
                "boundary_state must be non-negative when enforce_non_negative is enabled"
            )

    if isinstance(diffusivity, complex):
        raise TypeError("diffusivity must use a real floating-point dtype")
    if isinstance(diffusivity, (int, float)):
        diffusivity_tensor = torch.tensor(
            float(diffusivity),
            device=concentration.device,
            dtype=concentration.dtype,
        )
    elif isinstance(diffusivity, Tensor) and diffusivity.numel() == 1:
        if not diffusivity.is_floating_point():
            raise TypeError("diffusivity must use a real floating-point dtype")
        diffusivity_tensor = diffusivity.to(
            concentration.device, dtype=concentration.dtype
        )
    else:
        raw_diffusivity_tensor = (
            diffusivity.to(concentration.device)
            if isinstance(diffusivity, Tensor)
            else torch.as_tensor(diffusivity, device=concentration.device)
        )
        if not raw_diffusivity_tensor.is_floating_point():
            raise TypeError("diffusivity must use a real floating-point dtype")
        diffusivity_tensor = raw_diffusivity_tensor.to(dtype=concentration.dtype)

    if not bool(torch.all(torch.isfinite(diffusivity_tensor))):
        raise ValueError("diffusivity must contain only finite values")
    if bool(torch.any(diffusivity_tensor < 0.0)):
        raise ValueError("diffusivity must be non-negative")

    diffusivity_by_storage_axis = _normalize_directional_diffusivity(
        diffusivity_tensor,
        concentration,
        spatial_dims,
        diffusivity_layout,
    )

    diffusion_rate, boundary_fluxes = _prepare_diffusion_components(
        concentration,
        diffusivity_by_storage_axis,
        spacing_values,
        spatial_dims,
        boundary_method,
        boundary_state_tensor,
    )
    if not bool(torch.all(torch.isfinite(diffusion_rate))):
        raise ValueError("diffusion_rate must contain only finite values")
    if any(
        not bool(torch.all(torch.isfinite(boundary_flux)))
        for boundary_flux in boundary_fluxes.values()
    ):
        raise ValueError("diffusion boundary fluxes must contain only finite values")

    def _rate_and_boundary_fluxes(
        local_concentration: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        return _prepare_diffusion_components(
            local_concentration,
            diffusivity_by_storage_axis,
            spacing_values,
            spatial_dims,
            boundary_method,
            boundary_state_tensor,
        )

    def _rate_fn(local_concentration: Tensor) -> Tensor:
        local_rate, _ = _rate_and_boundary_fluxes(local_concentration)
        return local_rate + source_term_tensor

    stability_number = None
    integration_parameters = None
    if check_stability and time_step is not None:
        stability_number = torch.zeros_like(concentration)
        for axis_diffusivity, spacing in zip(
            diffusivity_by_storage_axis,
            spacing_values,
        ):
            stability_number = stability_number + (
                axis_diffusivity * float(time_step) / (spacing * spacing)
            )
        integration_parameters = _resolve_transport_time_integration(
            time_integration_method,
            implicit_midpoint_max_iter,
            implicit_midpoint_tolerance,
        )
        method = integration_parameters[0]
        if method == "euler":
            max_stability = float(torch.max(stability_number).detach())
            max_allowed = float(
                get_global_manager().get(
                    "transport.numerical.cfl_diffusion_max",
                    expected_type=float,
                    required=True,
                )
            )
            if boundary_method == "dirichlet":
                # References: Patankar (1980), cell-centered boundary conductance.
                # Mathematical expression: K/(dx/2) = 2K/dx at a boundary face,
                # so the worst diagonal is 3K/dx2 instead of 2K/dx2.
                # Parameter meanings: max_allowed is the interior Von Neumann bound.
                max_allowed = max_allowed * (2.0 / 3.0)
            if max_stability > max_allowed:
                raise ValueError(
                    "Diffusion stability violation: "
                    f"max stability {max_stability:.6g} exceeds limit "
                    f"{max_allowed:.6g}. Reduce time_step or use "
                    "simulator-managed substeps."
                )

    result = {
        "diffusion_rate": diffusion_rate,
        "stability_number": stability_number,
        "boundary_fluxes": boundary_fluxes,
    }

    if time_step is not None:
        if integration_parameters is None:
            integration_parameters = _resolve_transport_time_integration(
                time_integration_method,
                implicit_midpoint_max_iter,
                implicit_midpoint_tolerance,
            )
        method, max_iter, tol = integration_parameters
        concentration_new = _integrate_rate_system(
            concentration,
            float(time_step),
            _rate_fn,
            method,
            max_iter,
            tol,
        )

        if not bool(torch.all(torch.isfinite(concentration_new))):
            raise ValueError("diffusion concentration update must contain only finite values")

        if enforce_non_negative and bool(torch.any(concentration_new < 0.0)):
            minimum = float(torch.min(concentration_new))
            raise ValueError(
                "Diffusion step violated non-negative concentration: "
                f"minimum={minimum:.6g}. Reduce time_step or use a "
                "positivity-preserving integration method."
            )

        # References: Hairer, Norsett and Wanner (1993), Solving ODEs I;
        #   Hundsdorfer and Verwer (2003), Time-Dependent ADR Equations.
        # Mathematical expression: the step-averaged outward boundary flux is
        #   the quadrature of the integrator: euler j(c_n); implicit_midpoint
        #   j((c_n + c_{n+1})/2); crank_nicolson (j(c_n) + j(c_{n+1}))/2.
        # Parameter meanings: c_n, c_{n+1} are the concentrations before and
        #   after the step; j(c) is the outward boundary-flux ledger at c.
        # Units: c in [state_unit]; j in [state_unit m s-1].
        # Domain/constraints: only these three integrator/quadrature pairs
        #   are implemented; any other accepted method name raises instead of
        #   falling into a quadrature that does not match its integrator.
        # Sign convention: j is positive outward from the domain.
        if method == "euler":
            step_boundary_fluxes = boundary_fluxes
        elif method == "implicit_midpoint":
            _, step_boundary_fluxes = _rate_and_boundary_fluxes(
                0.5 * (concentration + concentration_new)
            )
        elif method == "crank_nicolson":
            _, final_boundary_fluxes = _rate_and_boundary_fluxes(concentration_new)
            step_boundary_fluxes = {
                name: 0.5 * (boundary_fluxes[name] + final_boundary_fluxes[name])
                for name in boundary_fluxes
            }
        else:
            raise ValueError(
                "No boundary-flux quadrature is implemented for time integration "
                f"method '{method}' in apply_diffusion"
            )
        result["boundary_fluxes"] = step_boundary_fluxes

        conservation_error = None
        if run_conservation_check:
            conservation_error = _finite_volume_transport_conservation_error(
                concentration,
                concentration_new,
                source_term_tensor,
                float(time_step),
                spacing_values,
                spatial_dims,
                step_boundary_fluxes,
                warn_on_violation=True,
            )

        result["concentration"] = concentration_new
        result["conservation_error"] = conservation_error

    if not has_explicit_batch:
        result["diffusion_rate"] = result["diffusion_rate"].squeeze(0)
        if "concentration" in result:
            result["concentration"] = result["concentration"].squeeze(0)
        result["boundary_fluxes"] = {
            name: value.squeeze(0)
            for name, value in result["boundary_fluxes"].items()
        }

    return result


def _expand_velocity_like_concentration(
    velocity_tensor: Tensor,
    concentration_tensor: Tensor,
    error_message: str,
) -> Tensor:
    """Expand velocity tensor to concentration tensor shape using existing advection rules."""
    if velocity_tensor.dim() == 0 or velocity_tensor.numel() == 1:
        return velocity_tensor.expand_as(concentration_tensor)
    if velocity_tensor.shape == concentration_tensor.shape:
        return velocity_tensor

    can_expand = (
        velocity_tensor.dim() == concentration_tensor.dim()
        and all(
            vsz in (1, csz)
            for vsz, csz in zip(velocity_tensor.shape, concentration_tensor.shape)
        )
    )
    if can_expand:
        return velocity_tensor.expand_as(concentration_tensor)

    if (
        velocity_tensor.dim() + 1 == concentration_tensor.dim()
        and velocity_tensor.shape == concentration_tensor.shape[1:]
        and concentration_tensor.shape[0] == 1
    ):
        return velocity_tensor.unsqueeze(0).expand_as(concentration_tensor)

    raise ValueError(
        f"{error_message}; velocity must be scalar or broadcast-compatible"
    )


def _compute_finite_volume_outflow_cfl(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
    boundary_method: str,
    time_step: float,
) -> Tensor:
    """Return the local outward-face Courant sum for FV upwind transport.

    References:
        LeVeque (2002), Finite Volume Methods for Hyperbolic Problems.
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.

    Mathematical expression:
        C_out,i = dt * sum_k(
            max(u_k,i+1/2, 0) - min(u_k,i-1/2, 0)
        ) / dx_k.
        Explicit first-order upwind transport is positivity preserving only
        when C_out,i <= 1 for every control volume.

    Parameter meanings:
        concentration defines the control-volume tensor shape [state_unit],
        velocity contains physical components (u, v, w, ...) [m s-1],
        grid_spacing is in storage-axis order [m], and time_step is dt [s].

    Open/closed system route:
        Boundary velocities contribute to the local outflow coefficient; the
        associated tracer mass remains represented by the boundary flux ledger.
    """
    spacing_values = _normalize_transport_grid_spacing(grid_spacing, spatial_dims)
    velocity_components = (
        tuple(velocity) if isinstance(velocity, (tuple, list)) else (velocity,)
    )
    if len(velocity_components) != spatial_dims:
        raise ValueError(
            f"velocity must provide exactly {spatial_dims} spatial components"
        )

    total_cfl = torch.zeros_like(concentration)
    for component_index, component in enumerate(velocity_components):
        axis = -(component_index + 1)
        spacing = spacing_values[axis]
        component_tensor = _expand_velocity_like_concentration(
            torch.as_tensor(
                component,
                device=concentration.device,
                dtype=concentration.dtype,
            ),
            concentration,
            "Velocity shape is incompatible with concentration shape for CFL computation",
        )
        moved_velocity = torch.movedim(component_tensor, axis, -1)
        if boundary_method == "periodic":
            right_face_velocity = 0.5 * (
                moved_velocity + torch.roll(moved_velocity, shifts=-1, dims=-1)
            )
            left_face_velocity = torch.roll(
                right_face_velocity,
                shifts=1,
                dims=-1,
            )
        else:
            interior_face_velocity = 0.5 * (
                moved_velocity[..., :-1] + moved_velocity[..., 1:]
            )
            right_face_velocity = torch.cat(
                [interior_face_velocity, moved_velocity[..., -1:]],
                dim=-1,
            )
            left_face_velocity = torch.cat(
                [moved_velocity[..., :1], interior_face_velocity],
                dim=-1,
            )
        directional_cfl = float(time_step) * (
            torch.clamp_min(right_face_velocity, 0.0)
            + torch.clamp_min(-left_face_velocity, 0.0)
        ) / spacing
        total_cfl = total_cfl + torch.movedim(directional_cfl, -1, axis)
    return total_cfl


def _compute_max_peclet_number(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    diffusivity: Union[float, Tensor],
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
    diffusivity_layout: Optional[str],
) -> Optional[Tensor]:
    """Return the maximum local directional cell Peclet number.

    References:
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        Hundsdorfer and Verwer (2003), Time-Dependent ADR Equations.

    Mathematical expression:
        Pe_i,k = abs(u_i,k) * dx_k / D_i,k,
        Pe_max = max_i,k(Pe_i,k).

    Parameter meanings:
        velocity component u_k is [m s-1], spacing dx_k is [m], diffusivity
        D_k is the axis-k coefficient [m2 s-1] of the same per-axis reading
        (_normalize_directional_diffusivity with diffusivity_layout) that the
        diffusion step uses, and Pe is dimensionless. A directional (Kz, Ky,
        Kx) coefficient is never broadcast as a cell field.

    Open/closed system route:
        Peclet is a diagnostic and does not alter storage, source, or boundary
        flux terms. A zero diffusivity makes the diagnostic undefined.
    """
    spacing_values = _normalize_transport_grid_spacing(grid_spacing, spatial_dims)
    diffusivity_by_storage_axis = _normalize_directional_diffusivity(
        torch.as_tensor(
            diffusivity,
            device=concentration.device,
            dtype=concentration.dtype,
        ),
        concentration,
        spatial_dims,
        diffusivity_layout,
    )
    if any(bool(torch.any(axis_field == 0.0)) for axis_field in diffusivity_by_storage_axis):
        return None

    velocity_components = (
        tuple(velocity) if isinstance(velocity, (tuple, list)) else (velocity,)
    )
    if len(velocity_components) != spatial_dims:
        raise ValueError(
            f"velocity must provide exactly {spatial_dims} components for "
            "Peclet diagnostics"
        )

    component_maxima = []
    for component_index, component in enumerate(velocity_components):
        velocity_tensor = _expand_velocity_like_concentration(
            torch.as_tensor(
                component,
                device=concentration.device,
                dtype=concentration.dtype,
            ),
            concentration,
            "Velocity shape is incompatible with concentration for Peclet diagnostics",
        )
        # References: Patankar (1980); Hundsdorfer and Verwer (2003).
        # Mathematical expression: Pe_i,k = |u_i,k| dx_k / D_i,k.
        # Parameter meanings: u_k velocity component, dx_k spacing and D_k the
        #   axis-k diffusivity of the declared per-axis reading; velocity
        #   component 0 is the last storage axis (x), as for the spacing.
        # Units: u [m s-1], dx [m], D [m2 s-1], Pe dimensionless.
        # Domain/constraints: D_k > 0 (a zero coefficient returns None above).
        # Sign convention: magnitudes only.
        storage_axis = spatial_dims - 1 - component_index
        spacing = spacing_values[storage_axis]
        local_peclet = (
            torch.abs(velocity_tensor) * spacing / diffusivity_by_storage_axis[storage_axis]
        )
        if not bool(torch.all(torch.isfinite(local_peclet))):
            raise ValueError(
                "Peclet diagnostic must contain only finite values for positive diffusivity"
            )
        component_maxima.append(torch.max(local_peclet))
    return torch.max(torch.stack(component_maxima))


def _resolve_advection_boundary_types(
    boundary_method: str,
    boundary_state: Optional[Tensor] = None,
) -> Tuple[str, str, Union[float, Tensor]]:
    """Resolve boundary laws without introducing a hidden nonzero tracer state."""
    boundary_value: Union[float, Tensor] = (
        boundary_state if boundary_state is not None else 0.0
    )
    if boundary_method == "open":
        return "dirichlet", "open", boundary_value
    if boundary_method == "extrapolate":
        return "extrapolate", "extrapolate", boundary_value
    return boundary_method, boundary_method, boundary_value


def _advection_ghost_from_type(
    field: Tensor,
    side: str,
    bc_type: str,
    boundary_value: Union[float, Tensor],
) -> Tensor:
    if side not in ("left", "right"):
        raise ValueError(f"Invalid boundary side '{side}'")

    boundary_slice = field[..., :1] if side == "left" else field[..., -1:]
    if bc_type == "dirichlet":
        if isinstance(boundary_value, Tensor):
            explicit_state = boundary_value.to(device=field.device, dtype=field.dtype)
            try:
                explicit_state = torch.broadcast_to(explicit_state, field.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "boundary_state must be broadcast-compatible with concentration"
                ) from exc
            return explicit_state[..., :1] if side == "left" else explicit_state[..., -1:]
        return torch.full_like(boundary_slice, float(boundary_value))
    if bc_type in ("neumann", "open"):
        return boundary_slice
    if bc_type == "zero":
        return torch.zeros_like(boundary_slice)
    if bc_type == "extrapolate":
        if field.size(-1) < 2:
            return boundary_slice
        if side == "left":
            return 2.0 * field[..., :1] - field[..., 1:2]
        return 2.0 * field[..., -1:] - field[..., -2:-1]
    raise ValueError(f"Unknown boundary type '{bc_type}'")


def _select_advection_boundary(
    field: Tensor,
    side: str,
    inflow_mask: Tensor,
    inflow_type: str,
    outflow_type: str,
    boundary_value: Union[float, Tensor],
) -> Tensor:
    inflow_ghost = _advection_ghost_from_type(field, side, inflow_type, boundary_value)
    outflow_ghost = _advection_ghost_from_type(field, side, outflow_type, boundary_value)
    return torch.where(inflow_mask, inflow_ghost, outflow_ghost)


def _shift_advection_1d(
    field: Tensor,
    direction: int,
    boundary_method: str,
    left_inflow_mask: Tensor,
    right_inflow_mask: Tensor,
    inflow_type: str,
    outflow_type: str,
    boundary_value: Union[float, Tensor],
) -> Tensor:
    """Shift a stencil by any integer offset with explicit ghost states.

    References:
        LeVeque (2002), Finite Volume Methods for Hyperbolic Problems.
    Mathematical expression:
        S_k(c)_i = c_{i-k}, with each out-of-domain index supplied by the
        configured boundary law.
    Parameter meanings:
        direction is the signed cell offset; boundary types select the ghost
        state used at each successive stencil extension.
    """

    if boundary_method == "periodic":
        return torch.roll(field, shifts=direction, dims=-1)

    shifted = field
    side = "left" if direction > 0 else "right"
    inflow_mask = left_inflow_mask if direction > 0 else right_inflow_mask
    for _ in range(abs(direction)):
        ghost = _select_advection_boundary(
            shifted,
            side,
            inflow_mask,
            inflow_type,
            outflow_type,
            boundary_value,
        )
        if direction > 0:
            shifted = torch.cat([ghost, shifted[..., :-1]], dim=-1)
        else:
            shifted = torch.cat([shifted[..., 1:], ghost], dim=-1)
    return shifted


def _dimensionless_limiter_gradient_ratio(
    delta_up: Tensor,
    delta_down: Tensor,
    relative_epsilon: Tensor,
) -> Tensor:
    """Return a scale-invariant successive-gradient ratio for TVD limiters.

    References:
        Sweby (1984), SIAM Journal on Numerical Analysis 21(5), 995-1011.
        LeVeque (2002), Finite Volume Methods for Hyperbolic Problems.

    Mathematical expression:
        r = (delta_up / s) / regularize(delta_down / s),
        s = max(abs(delta_up), abs(delta_down)).

    Parameter meanings and units:
        delta_up and delta_down are successive tracer differences with identical
        tracer units. relative_epsilon is dimensionless, so r is dimensionless.

    Open/closed system route:
        This ratio changes only face reconstruction. Boundary mass exchange is
        retained in the explicit advection boundary-flux ledger.
    """
    gradient_scale = torch.maximum(torch.abs(delta_up), torch.abs(delta_down))
    has_gradient = gradient_scale > 0.0
    safe_scale = torch.where(
        has_gradient,
        gradient_scale,
        torch.ones_like(gradient_scale),
    )
    normalized_up = delta_up / safe_scale
    normalized_down = delta_down / safe_scale
    denominator_sign = torch.where(
        normalized_down < 0.0,
        -torch.ones_like(normalized_down),
        torch.ones_like(normalized_down),
    )
    safe_normalized_down = torch.where(
        torch.abs(normalized_down) > relative_epsilon,
        normalized_down,
        denominator_sign * relative_epsilon,
    )
    ratio = normalized_up / safe_normalized_down
    return torch.where(has_gradient, ratio, torch.zeros_like(ratio))


def _compute_advection_flux_face_1d(
    local_concentration: Tensor,
    u_face: Tensor,
    c_ip1: Tensor,
    shift_fn: Any,
    scheme: str,
    time_step: Optional[float],
    dx_value: float,
) -> Tensor:
    # References:
    # - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs.
    # - Sweby, P.K. (1984). High Resolution Schemes.
    # Expression:
    #   dc/dt = -(F_{i+1/2} - F_{i-1/2})/dx.
    #   r = delta_up/delta_down is dimensionless; phi(r) limits antidiffusion.
    # Parameters and units:
    #   concentration has tracer units, velocity is [m s-1], dt is [s], and
    #   dx is [m]. limiter_epsilon is a dimensionless relative threshold.
    # Open/closed system route:
    #   Boundary face fluxes are retained by the caller in the mass ledger.
    if scheme == "upwind":
        return torch.where(
            u_face >= 0.0,
            u_face * local_concentration,
            u_face * c_ip1,
        )

    if scheme == "lax_wendroff":
        if time_step is None:
            raise ValueError("time_step is required for scheme 'lax_wendroff'")
        dt = float(time_step)
        c_avg = 0.5 * (local_concentration + c_ip1)
        dc = c_ip1 - local_concentration
        return u_face * c_avg - 0.5 * (u_face * u_face) * (dt / dx_value) * dc

    if scheme in ("superbee", "van_leer"):
        if time_step is None:
            raise ValueError(f"time_step is required for scheme '{scheme}'")
        dt = float(time_step)
        limiter_epsilon = float(
            get_global_manager().get(
                "transport.numerical.limiter_epsilon",
                expected_type=float,
                required=True,
            )
        )
        eps = torch.tensor(
            limiter_epsilon,
            device=local_concentration.device,
            dtype=local_concentration.dtype,
        )
        c_im1 = shift_fn(local_concentration, 1)
        c_ip2 = shift_fn(local_concentration, -2)

        delta_up_pos = local_concentration - c_im1
        delta_down_pos = c_ip1 - local_concentration
        r_pos = _dimensionless_limiter_gradient_ratio(
            delta_up_pos,
            delta_down_pos,
            eps,
        )

        delta_up_neg = c_ip2 - c_ip1
        delta_down_neg = c_ip1 - local_concentration
        r_neg = _dimensionless_limiter_gradient_ratio(
            delta_up_neg,
            delta_down_neg,
            eps,
        )
        r_values = torch.where(u_face >= 0.0, r_pos, r_neg)

        if scheme == "superbee":
            phi = torch.maximum(
                torch.zeros_like(r_values),
                torch.minimum(2.0 * r_values, torch.ones_like(r_values)),
            )
            phi = torch.maximum(
                phi,
                torch.minimum(r_values, 2.0 * torch.ones_like(r_values)),
            )
        else:
            phi = (r_values + torch.abs(r_values)) / (1.0 + torch.abs(r_values))

        cfl_local = torch.abs(u_face) * dt / dx_value
        upwind_flux = torch.where(
            u_face >= 0.0,
            u_face * local_concentration,
            u_face * c_ip1,
        )
        limited_gradient = phi * (c_ip1 - local_concentration)
        return upwind_flux + 0.5 * torch.abs(u_face) * (1.0 - cfl_local) * limited_gradient

    if scheme == "central":
        return u_face * 0.5 * (local_concentration + c_ip1)

    raise ValueError(f"Unknown advection scheme '{scheme}'")


def _compute_advection_rate_and_boundary_flux_1d(
    local_concentration: Tensor,
    local_velocity: Tensor,
    dx_value: float,
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    boundary_state: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return a conservative 1D rate and outward boundary face fluxes.

    References:
        LeVeque (2002), Finite Volume Methods for Hyperbolic Problems.
    Mathematical expression:
        dc_i/dt = -(F_{i+1/2} - F_{i-1/2}) / dx.
        F_out,left = -F_{1/2}; F_out,right = F_{N+1/2}.
    Parameter meanings:
        c is tracer concentration [state_unit], u is velocity [m s-1],
        dx is cell width [m], and F is tracer flux [state_unit m s-1].
    Open-system route:
        Nonperiodic mass exchange is returned through the two outward face fluxes.
    """
    if dx_value <= 0.0:
        raise ValueError("Grid spacing must be positive for advection")

    u = _expand_velocity_like_concentration(
        local_velocity.to(local_concentration.device),
        local_concentration,
        "Velocity shape is incompatible with concentration shape in 1D advection",
    )

    inflow_type, outflow_type, boundary_value = _resolve_advection_boundary_types(
        boundary_method,
        boundary_state,
    )
    left_inflow_mask = u[..., :1] >= 0.0
    right_inflow_mask = u[..., -1:] < 0.0

    def _shift_fn(field: Tensor, direction: int) -> Tensor:
        return _shift_advection_1d(
            field,
            direction,
            boundary_method,
            left_inflow_mask,
            right_inflow_mask,
            inflow_type,
            outflow_type,
            boundary_value,
        )

    c_ip1 = _shift_fn(local_concentration, -1)
    if boundary_method == "periodic":
        u_ip1 = torch.roll(u, shifts=-1, dims=-1)
    else:
        u_ip1 = torch.cat([u[..., 1:], u[..., -1:]], dim=-1)
    u_face = 0.5 * (u + u_ip1)

    flux_face = _compute_advection_flux_face_1d(
        local_concentration,
        u_face,
        c_ip1,
        _shift_fn,
        scheme,
        time_step,
        dx_value,
    )
    if boundary_method == "periodic":
        flux_im1 = torch.roll(flux_face, shifts=1, dims=-1)
        return -(flux_face - flux_im1) / dx_value, {}

    left_boundary_state = _select_advection_boundary(
        local_concentration,
        "left",
        left_inflow_mask,
        inflow_type,
        outflow_type,
        boundary_value,
    )
    right_boundary_state = _select_advection_boundary(
        local_concentration,
        "right",
        right_inflow_mask,
        inflow_type,
        outflow_type,
        boundary_value,
    )
    left_velocity = u[..., :1]
    right_velocity = u[..., -1:]

    # References: LeVeque (2002), conservative first-order boundary fluxes.
    # Mathematical expression: F_b = u_b c_upwind,b.
    # Parameter meanings: u_b [m s-1], c_upwind,b [state_unit].
    # Open-system route: outward fluxes are retained for the boundary ledger.
    left_flux = torch.where(
        left_velocity >= 0.0,
        left_velocity * left_boundary_state,
        left_velocity * local_concentration[..., :1],
    )
    right_flux = torch.where(
        right_velocity >= 0.0,
        right_velocity * local_concentration[..., -1:],
        right_velocity * right_boundary_state,
    )
    flux_face = torch.cat([flux_face[..., :-1], right_flux], dim=-1)
    flux_im1 = torch.cat([left_flux, flux_face[..., :-1]], dim=-1)
    outward_fluxes = {
        "min": -left_flux.squeeze(-1),
        "max": right_flux.squeeze(-1),
    }
    return -(flux_face - flux_im1) / dx_value, outward_fluxes


def _compute_advection_rate_1d(
    local_concentration: Tensor,
    local_velocity: Tensor,
    dx_value: float,
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    boundary_state: Optional[Tensor] = None,
) -> Tensor:
    """Return only the conservative rate for time-integration callbacks."""

    rate, _ = _compute_advection_rate_and_boundary_flux_1d(
        local_concentration,
        local_velocity,
        dx_value,
        scheme,
        boundary_method,
        time_step,
        boundary_state,
    )
    return rate


def _validate_advection_cfl(cfl_number: Tensor) -> None:
    """Reject a requested explicit step that violates the configured CFL bound."""

    max_cfl = torch.max(cfl_number)
    max_allowed_cfl = float(
        get_global_manager().get(
            "transport.numerical.cfl_advection_max",
            expected_type=float,
            required=True,
        )
    )
    if max_cfl > max_allowed_cfl:
        raise ValueError(
            "CFL stability violation: "
            f"max CFL {float(max_cfl):.6g} exceeds limit {max_allowed_cfl:.6g}. "
            "Reduce time_step or use simulator-managed FunctorMetadata.substep_count."
        )


def _prepare_advection_1d(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    check_cfl: bool,
    boundary_state: Optional[Tensor],
    enforce_cfl_limit: bool,
) -> Tuple[Tensor, Optional[Tensor], Any, Dict[str, Tensor]]:
    u = velocity[0] if isinstance(velocity, (tuple, list)) else velocity
    dx = float(grid_spacing if isinstance(grid_spacing, (int, float)) else grid_spacing[0])
    u_tensor = torch.as_tensor(u, device=concentration.device, dtype=concentration.dtype)
    advection_rate, boundary_fluxes = _compute_advection_rate_and_boundary_flux_1d(
        concentration,
        u_tensor,
        dx,
        scheme,
        boundary_method,
        time_step,
        boundary_state,
    )

    cfl_number = None
    if check_cfl and time_step is not None:
        cfl_number = _compute_finite_volume_outflow_cfl(
            concentration,
            u_tensor,
            dx,
            1,
            boundary_method,
            time_step,
        )
        if enforce_cfl_limit:
            _validate_advection_cfl(cfl_number)

    def _rate_fn(local_concentration: Tensor) -> Tensor:
        return _compute_advection_rate_1d(
            local_concentration,
            torch.as_tensor(
                u,
                device=local_concentration.device,
                dtype=local_concentration.dtype,
            ),
            dx,
            scheme,
            boundary_method,
            time_step,
            boundary_state,
        )

    named_boundary_fluxes = (
        {}
        if not boundary_fluxes
        else {
            "x_min": boundary_fluxes["min"],
            "x_max": boundary_fluxes["max"],
        }
    )
    return advection_rate, cfl_number, _rate_fn, named_boundary_fluxes


def _prepare_advection_2d(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    check_cfl: bool,
    boundary_state: Optional[Tensor],
    enforce_cfl_limit: bool,
) -> Tuple[Tensor, Optional[Tensor], Any, Dict[str, Tensor]]:
    if not (isinstance(velocity, (tuple, list)) and len(velocity) >= 2):
        raise ValueError("2D advection requires two velocity components")

    u = _expand_velocity_like_concentration(
        torch.as_tensor(
            velocity[0], device=concentration.device, dtype=concentration.dtype
        ),
        concentration,
        "X velocity shape is incompatible with concentration shape in 2D advection",
    )
    v = _expand_velocity_like_concentration(
        torch.as_tensor(
            velocity[1], device=concentration.device, dtype=concentration.dtype
        ),
        concentration,
        "Y velocity shape is incompatible with concentration shape in 2D advection",
    )

    if isinstance(grid_spacing, (int, float)):
        dx = dy = float(grid_spacing)
    else:
        dy, dx = float(grid_spacing[0]), float(grid_spacing[1])

    rate_x, boundary_x = _compute_advection_rate_and_boundary_flux_1d(
        concentration,
        u,
        dx,
        scheme,
        boundary_method,
        time_step,
        boundary_state,
    )
    boundary_state_y = (
        boundary_state.transpose(-2, -1) if boundary_state is not None else None
    )
    rate_y_t, boundary_y = _compute_advection_rate_and_boundary_flux_1d(
        concentration.transpose(-2, -1),
        v.transpose(-2, -1),
        dy,
        scheme,
        boundary_method,
        time_step,
        boundary_state_y,
    )
    advection_rate = rate_x + rate_y_t.transpose(-2, -1)

    cfl_number = None
    if check_cfl and time_step is not None:
        cfl_number = _compute_finite_volume_outflow_cfl(
            concentration,
            (u, v),
            (dy, dx),
            2,
            boundary_method,
            time_step,
        )
        if enforce_cfl_limit:
            _validate_advection_cfl(cfl_number)

    def _rate_fn(local_concentration: Tensor) -> Tensor:
        local_rate_x = _compute_advection_rate_1d(
            local_concentration,
            u,
            dx,
            scheme,
            boundary_method,
            time_step,
            boundary_state,
        )
        local_rate_y_t = _compute_advection_rate_1d(
            local_concentration.transpose(-2, -1),
            v.transpose(-2, -1),
            dy,
            scheme,
            boundary_method,
            time_step,
            boundary_state_y,
        )
        return local_rate_x + local_rate_y_t.transpose(-2, -1)

    boundary_fluxes = {}
    if boundary_x:
        boundary_fluxes.update(
            {"x_min": boundary_x["min"], "x_max": boundary_x["max"]}
        )
    if boundary_y:
        boundary_fluxes.update(
            {"y_min": boundary_y["min"], "y_max": boundary_y["max"]}
        )
    return advection_rate, cfl_number, _rate_fn, boundary_fluxes


def _prepare_advection_3d(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    check_cfl: bool,
    boundary_state: Optional[Tensor],
    enforce_cfl_limit: bool,
) -> Tuple[Tensor, Optional[Tensor], Any, Dict[str, Tensor]]:
    if not (isinstance(velocity, (tuple, list)) and len(velocity) >= 3):
        raise ValueError("3D advection requires three velocity components")

    u = _expand_velocity_like_concentration(
        torch.as_tensor(
            velocity[0], device=concentration.device, dtype=concentration.dtype
        ),
        concentration,
        "X velocity shape is incompatible with concentration shape in 3D advection",
    )
    v = _expand_velocity_like_concentration(
        torch.as_tensor(
            velocity[1], device=concentration.device, dtype=concentration.dtype
        ),
        concentration,
        "Y velocity shape is incompatible with concentration shape in 3D advection",
    )
    w = _expand_velocity_like_concentration(
        torch.as_tensor(
            velocity[2], device=concentration.device, dtype=concentration.dtype
        ),
        concentration,
        "Z velocity shape is incompatible with concentration shape in 3D advection",
    )

    if isinstance(grid_spacing, (int, float)):
        dx = dy = dz = float(grid_spacing)
    else:
        dz, dy, dx = float(grid_spacing[0]), float(grid_spacing[1]), float(grid_spacing[2])

    rate_x, boundary_x = _compute_advection_rate_and_boundary_flux_1d(
        concentration,
        u,
        dx,
        scheme,
        boundary_method,
        time_step,
        boundary_state,
    )
    boundary_state_y = (
        boundary_state.transpose(-2, -1) if boundary_state is not None else None
    )
    rate_y_t, boundary_y = _compute_advection_rate_and_boundary_flux_1d(
        concentration.transpose(-2, -1),
        v.transpose(-2, -1),
        dy,
        scheme,
        boundary_method,
        time_step,
        boundary_state_y,
    )
    c_perm = torch.movedim(concentration, -3, -1)
    w_perm = torch.movedim(w, -3, -1)
    boundary_state_z = (
        torch.movedim(boundary_state, -3, -1)
        if boundary_state is not None
        else None
    )
    rate_z_perm, boundary_z = _compute_advection_rate_and_boundary_flux_1d(
        c_perm,
        w_perm,
        dz,
        scheme,
        boundary_method,
        time_step,
        boundary_state_z,
    )
    advection_rate = (
        rate_x
        + rate_y_t.transpose(-2, -1)
        + torch.movedim(rate_z_perm, -1, -3)
    )

    cfl_number = None
    if check_cfl and time_step is not None:
        cfl_number = _compute_finite_volume_outflow_cfl(
            concentration,
            (u, v, w),
            (dz, dy, dx),
            3,
            boundary_method,
            time_step,
        )
        if enforce_cfl_limit:
            _validate_advection_cfl(cfl_number)

    def _rate_fn(local_concentration: Tensor) -> Tensor:
        local_x = _compute_advection_rate_1d(
            local_concentration,
            u,
            dx,
            scheme,
            boundary_method,
            time_step,
            boundary_state,
        )
        local_y_t = _compute_advection_rate_1d(
            local_concentration.transpose(-2, -1),
            v.transpose(-2, -1),
            dy,
            scheme,
            boundary_method,
            time_step,
            boundary_state_y,
        )
        local_perm = torch.movedim(local_concentration, -3, -1)
        local_z_perm = _compute_advection_rate_1d(
            local_perm,
            w_perm,
            dz,
            scheme,
            boundary_method,
            time_step,
            boundary_state_z,
        )
        return (
            local_x
            + local_y_t.transpose(-2, -1)
            + torch.movedim(local_z_perm, -1, -3)
        )

    boundary_fluxes = {}
    if boundary_x:
        boundary_fluxes.update(
            {"x_min": boundary_x["min"], "x_max": boundary_x["max"]}
        )
    if boundary_y:
        boundary_fluxes.update(
            {"y_min": boundary_y["min"], "y_max": boundary_y["max"]}
        )
    if boundary_z:
        boundary_fluxes.update(
            {"z_min": boundary_z["min"], "z_max": boundary_z["max"]}
        )
    return advection_rate, cfl_number, _rate_fn, boundary_fluxes


def _prepare_advection_nd(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    check_cfl: bool,
    boundary_state: Optional[Tensor],
    enforce_cfl_limit: bool,
) -> Tuple[Tensor, Optional[Tensor], Any, Dict[str, Tensor]]:
    if not (isinstance(velocity, (tuple, list)) and len(velocity) >= spatial_dims):
        raise ValueError(
            f"{spatial_dims}D advection requires {spatial_dims} velocity components"
        )

    if isinstance(grid_spacing, (int, float)):
        spacing_values = [float(grid_spacing)] * spatial_dims
    else:
        if len(grid_spacing) < spatial_dims:
            raise ValueError(
                f"{spatial_dims}D advection requires {spatial_dims} grid spacing values"
            )
        spacing_values = [float(grid_spacing[i]) for i in range(spatial_dims)]

    # References:
    # - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs.
    # - Morton, K.W. and Mayers, D.F. (2005). Numerical Solution of PDEs.
    # Mathematical expression:
    #   dc/dt = -sum_k d(u_k * c)/dx_k
    # Parameter meanings:
    #   spatial_dims: number of spatial axes.
    #   velocity_components[k]: velocity along axis k (x first, then y, z, ...).
    #   spacing_values: grid spacing tuple ordered as outer-to-inner axes.
    def _compute_nd_rate_and_boundaries(
        local_concentration: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_rate = torch.zeros_like(local_concentration)
        boundary_fluxes: Dict[str, Tensor] = {}
        axis_labels = ("x", "y", "z", "w")
        for axis_component in range(spatial_dims):
            axis = -(axis_component + 1)
            spacing = spacing_values[-(axis_component + 1)]
            local_velocity = torch.as_tensor(
                velocity[axis_component],
                device=local_concentration.device,
                dtype=local_concentration.dtype,
            )
            local_velocity = _expand_velocity_like_concentration(
                local_velocity,
                local_concentration,
                "Velocity shape is incompatible with concentration shape in ND advection",
            )
            local_boundary_state = boundary_state
            if axis == -1:
                directional_rate, directional_boundary = (
                    _compute_advection_rate_and_boundary_flux_1d(
                    local_concentration,
                    local_velocity,
                    spacing,
                    scheme,
                    boundary_method,
                    time_step,
                    local_boundary_state,
                )
                )
            else:
                moved_concentration = torch.movedim(local_concentration, axis, -1)
                moved_velocity = torch.movedim(local_velocity, axis, -1)
                moved_boundary_state = (
                    torch.movedim(local_boundary_state, axis, -1)
                    if local_boundary_state is not None
                    else None
                )
                moved_rate, directional_boundary = (
                    _compute_advection_rate_and_boundary_flux_1d(
                    moved_concentration,
                    moved_velocity,
                    spacing,
                    scheme,
                    boundary_method,
                    time_step,
                    moved_boundary_state,
                )
                )
                directional_rate = torch.movedim(moved_rate, -1, axis)
            total_rate = total_rate + directional_rate
            if directional_boundary:
                axis_label = (
                    axis_labels[axis_component]
                    if axis_component < len(axis_labels)
                    else f"axis_{axis_component}"
                )
                boundary_fluxes[f"{axis_label}_min"] = directional_boundary["min"]
                boundary_fluxes[f"{axis_label}_max"] = directional_boundary["max"]
        return total_rate, boundary_fluxes

    def _compute_nd_rate(local_concentration: Tensor) -> Tensor:
        rate, _ = _compute_nd_rate_and_boundaries(local_concentration)
        return rate

    advection_rate, boundary_fluxes = _compute_nd_rate_and_boundaries(concentration)

    cfl_number = None
    if check_cfl and time_step is not None:
        cfl_number = _compute_finite_volume_outflow_cfl(
            concentration,
            velocity,
            tuple(spacing_values),
            spatial_dims,
            boundary_method,
            time_step,
        )
        if enforce_cfl_limit:
            _validate_advection_cfl(cfl_number)

    def _rate_fn(local_concentration: Tensor) -> Tensor:
        return _compute_nd_rate(local_concentration)

    return advection_rate, cfl_number, _rate_fn, boundary_fluxes


def _prepare_advection_components(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    check_cfl: bool,
    boundary_state: Optional[Tensor],
    enforce_cfl_limit: bool,
) -> Tuple[Tensor, Optional[Tensor], Any, Dict[str, Tensor]]:
    if spatial_dims == 1:
        return _prepare_advection_1d(
            concentration,
            velocity,
            grid_spacing,
            scheme,
            boundary_method,
            time_step,
            check_cfl,
            boundary_state,
            enforce_cfl_limit,
        )
    if spatial_dims == 2:
        return _prepare_advection_2d(
            concentration,
            velocity,
            grid_spacing,
            scheme,
            boundary_method,
            time_step,
            check_cfl,
            boundary_state,
            enforce_cfl_limit,
        )
    if spatial_dims == 3:
        return _prepare_advection_3d(
            concentration,
            velocity,
            grid_spacing,
            scheme,
            boundary_method,
            time_step,
            check_cfl,
            boundary_state,
            enforce_cfl_limit,
        )
    return _prepare_advection_nd(
        concentration,
        velocity,
        grid_spacing,
        spatial_dims,
        scheme,
        boundary_method,
        time_step,
        check_cfl,
        boundary_state,
        enforce_cfl_limit,
    )


def _check_advection_time_integration_pair(scheme: str, method: str) -> None:
    """Reject an advection scheme and time integration method that cannot step together.

    Shared by apply_advection (for every step it takes) and AdvectionOperator
    (at construction, for its configured pair), so the rule has one owner.

    Raises:
        ValueError: With the reason, for central with euler or for a fully
            discrete scheme with an implicit method.
    """
    scheme = str(scheme).strip().lower()
    method = str(method).strip().lower()
    # References:
    #   Holton, J.R. (2004). An Introduction to Dynamic Meteorology, 4th ed.,
    #   Ch. 13 (centered-difference advection); Durran, D.R. (1999). Numerical
    #   Methods for Wave Equations in Geophysical Fluid Dynamics; LeVeque,
    #   R.J. (2002). Finite Volume Methods for Hyperbolic Problems (Lax-Wendroff
    #   and flux-limited methods); Hairer, E. and Wanner, G. (1996). Solving
    #   ODEs II (A-stability of the implicit midpoint and trapezoidal rules).
    # Mathematical expression:
    #   Central semi-discretization dc_j/dt = -u (c_{j+1} - c_{j-1}) / (2 dx)
    #   has the purely imaginary eigenvalues lambda = -i (u/dx) sin(k dx).
    #   Forward Euler (FTCS): g = 1 + dt lambda, |g|^2 = 1 + C^2 sin^2(k dx) > 1
    #   for every C != 0, so the pair is unconditionally unstable. Implicit
    #   midpoint and Crank-Nicolson: g = (1 + dt lambda/2) / (1 - dt lambda/2),
    #   |g| = 1, so central advection is neutrally stable with them.
    #   lax_wendroff, superbee and van_leer fluxes already contain dt
    #   (F = u c_avg - u^2 dt/(2 dx) dc, limited (1 - C) antidiffusion); they
    #   are one-step updates c^{n+1} = c^n - dt/dx (F_{j+1/2} - F_{j-1/2}) and
    #   only the euler update of those fluxes is the scheme.
    # Parameter meanings:
    #   u velocity, dx grid spacing, dt time step, C = |u| dt / dx Courant
    #   number, k wave number, g amplification factor per step.
    # Units:
    #   u [m s-1], dx [m], dt [s], k [m-1]; C, g and lambda dt dimensionless.
    # Domain/constraints:
    #   Accepted: upwind with every method; lax_wendroff, superbee and
    #   van_leer only with euler; central only with an implicit method. The
    #   explicit CFL limit of the accepted euler pairs is checked separately
    #   (transport.numerical.cfl_advection_max).
    # Sign convention:
    #   u > 0 transports toward increasing index; Re(lambda) <= 0 is decay.
    if scheme == "central" and method == "euler":
        raise ValueError(
            "central advection requires an implicit time integrator; "
            "forward Euler central FTCS is unconditionally unstable"
        )
    if scheme in ("lax_wendroff", "superbee", "van_leer") and method != "euler":
        raise ValueError(
            f"fully discrete advection scheme '{scheme}' requires euler "
            "and cannot be reintegrated by an implicit method"
        )


def _check_splitting_time_integration_pair(splitting_method: str, method: str) -> None:
    """Reject a splitting method and time integration method that cannot step together.

    Shared by advance_advection_diffusion (combined branch) and
    AdvectionDiffusionOperator (at construction, for its configured pair).

    Raises:
        ValueError: With the reason, for combined splitting with an implicit
            method.
    """
    splitting_method = str(splitting_method).strip().lower()
    method = str(method).strip().lower()
    # References:
    #   Hundsdorfer, W. and Verwer, J.G. (2003). Numerical Solution of
    #   Time-Dependent Advection-Diffusion-Reaction Equations (method of lines
    #   and operator splitting); Strang, G. (1968), SIAM J. Numer. Anal. 5.
    # Mathematical expression:
    #   combined: c^{n+1} = c^n + dt (A(c^n) + D(c^n) + S), one unsplit
    #   forward Euler step, stable only while C + w sigma <= cfl_advection_max
    #   (w = 2, or 3 for a Dirichlet boundary; checked at apply).
    #   sequential: c^{n+1} = D_dt(A_dt(c^n)); symmetric: D_{dt/2} A_dt D_{dt/2};
    #   each sub-step integrates its own rate with the configured method.
    # Parameter meanings:
    #   A(c) = -div(u c) advection rate, D(c) = div(K grad c) diffusion rate,
    #   S source rate, C = |u| dt / dx, sigma = K dt / dx^2, w diagonal weight.
    # Units:
    #   c [state_unit], A, D, S [state_unit s-1], u [m s-1], K [m2 s-1],
    #   dt [s], dx [m]; C, sigma and w dimensionless.
    # Domain/constraints:
    #   combined is defined only for euler; an implicit method is rejected
    #   instead of being replaced by euler. sequential and symmetric accept
    #   every method; their advection sub-step obeys
    #   _check_advection_time_integration_pair, and the explicit diffusion
    #   bound sigma <= cfl_diffusion_max depends on dt and is checked at apply.
    # Sign convention:
    #   Rates are positive into the cell; u > 0 transports toward increasing
    #   index.
    if splitting_method == "combined" and method != "euler":
        raise ValueError(
            "combined splitting requires euler time integration: it is one "
            "unsplit explicit Euler step of the summed advection and diffusion "
            f"rates, not a '{method}' step"
        )


def apply_advection(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    grid_spacing: Union[float, Tuple[float, ...]],
    spatial_dims: int,
    scheme: str,
    boundary_method: str,
    time_step: Optional[float],
    check_cfl: bool,
    time_integration_method: Optional[str] = None,
    implicit_midpoint_max_iter: Optional[int] = None,
    implicit_midpoint_tolerance: Optional[float] = None,
    boundary_state: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """
    Apply advection operator to concentration field using unified operators.
    
    Mathematical Expression:
        dc/dt = -nabla*(v*c)
        Upwind scheme for stability
        
    Parameters:
        concentration : Tensor
            Concentration field [kg m-3]
        velocity : Union[Tensor, Tuple[Tensor, ...]]
            Velocity field(s) [m s-1]
            Scalar for 1D, tuple for multi-dimensional
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing in each direction [m]
        spatial_dims : int
            Number of spatial dimensions
        scheme : str
            Advection scheme: "upwind", "lax_wendroff", "central", "superbee", "van_leer"
        boundary_method : str
            Boundary treatment: "zero", "extrapolate", "periodic", "dirichlet",
            "neumann", or "open"
        boundary_state : Optional[Tensor]
            Explicit tracer state at nonperiodic boundaries [concentration units].
        time_step : Optional[float]
            Time step for integration [s]
        check_cfl : bool
            Check CFL condition
        
    Returns:
        Dict[str, Tensor]
            {
                'concentration' or 'advection_rate': Updated field or rate [kg m-3 s-1],
                'cfl_number': Maximum CFL number [dimensionless],
                'boundary_fluxes': outward face fluxes [concentration m s-1]
            }
            
    References:
        - LeVeque, R.J. (2007). Finite Difference Methods
        - Morton, K.W. & Mayers, D.F. (2005). Numerical Solution of PDEs
    """
    if spatial_dims < 1:
        raise ValueError("spatial_dims must be at least one")
    if concentration.dim() < spatial_dims:
        raise ValueError(
            "Concentration rank is incompatible with spatial_dims: "
            f"rank={concentration.dim()}, spatial_dims={spatial_dims}"
        )
    if not concentration.is_floating_point():
        raise TypeError("concentration must use a floating-point dtype")
    if not bool(torch.all(torch.isfinite(concentration))):
        raise ValueError("concentration must contain only finite values")
    _normalize_transport_grid_spacing(grid_spacing, spatial_dims)
    if time_step is not None and (
        not math.isfinite(float(time_step)) or float(time_step) <= 0.0
    ):
        raise ValueError("time_step must be finite and positive")

    velocity_components = (
        tuple(velocity) if isinstance(velocity, (tuple, list)) else (velocity,)
    )
    if len(velocity_components) != spatial_dims:
        raise ValueError(
            f"velocity must provide exactly {spatial_dims} spatial components"
        )
    for component in velocity_components:
        component_tensor = torch.as_tensor(component)
        if component_tensor.is_complex():
            raise TypeError("velocity must use a real floating-point dtype")
        if not bool(torch.all(torch.isfinite(component_tensor))):
            raise ValueError("velocity must contain only finite values")

    boundary_method = str(boundary_method).lower()
    allowed_boundaries = {
        "zero",
        "extrapolate",
        "periodic",
        "dirichlet",
        "neumann",
        "open",
    }
    if boundary_method not in allowed_boundaries:
        raise ValueError(f"Unknown advection boundary method '{boundary_method}'")
    if boundary_method in {"dirichlet", "open"} and boundary_state is None:
        raise ValueError(
            f"Advection boundary '{boundary_method}' requires explicit boundary_state"
        )
    if boundary_state is not None and boundary_method not in {"dirichlet", "open"}:
        raise ValueError(
            "boundary_state is only supported for Dirichlet or open advection"
        )

    has_explicit_batch = concentration.dim() > spatial_dims
    working_concentration = concentration if has_explicit_batch else concentration.unsqueeze(0)
    working_boundary_state = None
    if boundary_state is not None:
        working_boundary_state = torch.as_tensor(
            boundary_state,
            device=working_concentration.device,
        )
        if working_boundary_state.is_complex():
            raise TypeError("boundary_state must use real floating-point values")
        working_boundary_state = working_boundary_state.to(
            dtype=working_concentration.dtype
        )
        try:
            working_boundary_state = torch.broadcast_to(
                working_boundary_state,
                working_concentration.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "boundary_state must be broadcast-compatible with concentration"
            ) from exc
        if not bool(torch.all(torch.isfinite(working_boundary_state))):
            raise ValueError("boundary_state must contain only finite values")

    method, max_iter, tol = _resolve_transport_time_integration(
        time_integration_method,
        implicit_midpoint_max_iter,
        implicit_midpoint_tolerance,
    )
    scheme = str(scheme).lower()
    if time_step is not None:
        # A rate evaluation takes no step; a fully discrete scheme without
        # time_step is rejected by its flux (time_step is required).
        _check_advection_time_integration_pair(scheme, method)

    advection_rate, cfl_number, rate_fn, boundary_fluxes = _prepare_advection_components(
        working_concentration,
        velocity,
        grid_spacing,
        spatial_dims,
        scheme,
        boundary_method,
        time_step,
        check_cfl,
        working_boundary_state,
        method == "euler",
    )
    if not bool(torch.all(torch.isfinite(advection_rate))):
        raise ValueError("advection_rate must contain only finite values")
    if cfl_number is not None and not bool(torch.all(torch.isfinite(cfl_number))):
        raise ValueError("cfl_number must contain only finite values")
    if any(
        not bool(torch.all(torch.isfinite(boundary_flux)))
        for boundary_flux in boundary_fluxes.values()
    ):
        raise ValueError("advection boundary fluxes must contain only finite values")

    result: Dict[str, Any] = {
        "advection_rate": advection_rate,
        "cfl_number": cfl_number,
        "boundary_fluxes": boundary_fluxes,
    }

    if time_step is not None:
        concentration_new = _integrate_rate_system(
            working_concentration,
            float(time_step),
            rate_fn,
            method,
            max_iter,
            tol,
        )
        if not bool(torch.all(torch.isfinite(concentration_new))):
            raise ValueError("advection concentration update must contain only finite values")
        result["concentration"] = concentration_new
        if method == "euler":
            step_boundary_fluxes = boundary_fluxes
        else:
            def _boundary_fluxes_at(local_concentration: Tensor) -> Dict[str, Tensor]:
                _, _, _, local_boundary_fluxes = _prepare_advection_components(
                    local_concentration,
                    velocity,
                    grid_spacing,
                    spatial_dims,
                    scheme,
                    boundary_method,
                    time_step,
                    False,
                    working_boundary_state,
                    False,
                )
                return local_boundary_fluxes

            # References: Hairer, Norsett and Wanner (1993), Solving ODEs I;
            #   Hundsdorfer and Verwer (2003), Time-Dependent ADR Equations.
            # Mathematical expression: implicit_midpoint uses
            #   j((c_n + c_{n+1})/2); crank_nicolson uses
            #   (j(c_n) + j(c_{n+1}))/2 (euler, above, uses j(c_n)).
            # Parameter meanings: c_n, c_{n+1} are the concentrations before
            #   and after the step; j(c) is the outward boundary-flux ledger.
            # Units: c in [state_unit]; j in [state_unit m s-1].
            # Domain/constraints: only these integrator/quadrature pairs are
            #   implemented; any other accepted method name raises.
            # Sign convention: j is positive outward from the domain.
            if method == "implicit_midpoint":
                step_boundary_fluxes = _boundary_fluxes_at(
                    0.5 * (working_concentration + concentration_new)
                )
            elif method == "crank_nicolson":
                final_boundary_fluxes = _boundary_fluxes_at(concentration_new)
                step_boundary_fluxes = {
                    name: 0.5 * (
                        boundary_fluxes[name] + final_boundary_fluxes[name]
                    )
                    for name in boundary_fluxes
                }
            else:
                raise ValueError(
                    "No boundary-flux quadrature is implemented for time "
                    f"integration method '{method}' in apply_advection"
                )
        result["boundary_fluxes"] = step_boundary_fluxes
        # References: Patankar (1980); Hundsdorfer and Verwer (2003).
        # Expression: Delta M + dt * sum_f(j_out,f A_f) = 0.
        # Parameters: j_out is outward-positive [state_unit m s-1],
        # grid_spacing is [m], and time_step is [s].
        # Open route: every nonperiodic face contribution is retained in the
        # returned boundary_fluxes ledger; periodic boundaries have no port.
        spacing_values = _normalize_transport_grid_spacing(
            grid_spacing,
            spatial_dims,
        )
        result["conservation_error"] = (
            _finite_volume_transport_conservation_error(
                working_concentration,
                concentration_new,
                torch.zeros_like(working_concentration),
                float(time_step),
                spacing_values,
                spatial_dims,
                step_boundary_fluxes,
                warn_on_violation=True,
            )
        )

    if not has_explicit_batch:
        result["advection_rate"] = result["advection_rate"].squeeze(0)
        if result["cfl_number"] is not None:
            result["cfl_number"] = result["cfl_number"].squeeze(0)
        if "concentration" in result:
            result["concentration"] = result["concentration"].squeeze(0)
        result["boundary_fluxes"] = {
            name: value.squeeze(0)
            for name, value in result["boundary_fluxes"].items()
        }

    return result


def advance_advection_diffusion(
    concentration: Tensor,
    velocity: Union[Tensor, Tuple[Tensor, ...]],
    diffusivity: Union[float, Tensor],
    grid_spacing: Union[float, Tuple[float, ...]],
    time_step: float,
    spatial_dims: int,
    splitting_method: str,
    advection_scheme: str,
    boundary_method: str,
    check_stability: bool,
    enforce_non_negative: bool,
    source_term: Optional[Tensor] = None,
    run_conservation_check: bool = True,
    time_integration_method: Optional[str] = None,
    implicit_midpoint_max_iter: Optional[int] = None,
    implicit_midpoint_tolerance: Optional[float] = None,
    advection_boundary_state: Optional[Tensor] = None,
    diffusion_boundary_state: Optional[Tensor] = None,
    diffusivity_layout: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Advance advection-diffusion equation using operator splitting.
    
    Mathematical Expression:
        dc/dt = -nabla*(v*c) + nabla*(K*nabla*c) + S
        
        Operator splitting methods:
        - Sequential: c* = L_adv(dt, c^n), c^(n+1) = L_diff(dt, c*)
        - Symmetric: c* = L_S(dt/2, c^n), then L_diff(dt/2),
          L_adv(dt), L_diff(dt/2), and L_S(dt/2)
        - Combined: c^(n+1) = c^n + dt*(-nabla*(v*c^n) + nabla*(K*nablac^n) + S^n)
    
    Parameters:
        concentration : Tensor
            Concentration field [kg m-3]
        velocity : Union[Tensor, Tuple[Tensor, ...]]
            Velocity field(s) [m s-1]
        diffusivity : Union[float, Tensor]
            Diffusion coefficient [m2 s-1]
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing in each direction [m]
        time_step : float
            Time step [s]
        spatial_dims : int
            Number of spatial dimensions
        splitting_method : str
            "sequential", "symmetric", or "combined"
        advection_scheme : str
            Advection scheme
        boundary_method : str
            Boundary treatment
        check_stability : bool
            Check stability constraints
        enforce_non_negative : bool
            Enforce non-negativity
        source_term : Optional[Tensor]
            Source/sink term [kg m-3 s-1]
        run_conservation_check : bool
            Whether to compute conservation diagnostics
        advection_boundary_state : Optional[Tensor]
            Explicit external tracer state for advective inflow [kg m-3]
        diffusion_boundary_state : Optional[Tensor]
            Explicit physical face state for Dirichlet diffusion [kg m-3]
        diffusivity_layout : Optional[str]
            "scalar_field" or "directional"; required for a tensor diffusivity
            with ndim >= 1 (never inferred from its shape), optional for a
            number or a 0-d tensor. The diffusion step and the Peclet
            diagnostic use the same per-axis reading.

    Returns:
        Dict[str, Tensor]
            {
                'concentration': Updated concentration [kg m-3],
                'advection_rate': Advection rate [kg m-3 s-1],
                'diffusion_rate': Diffusion rate [kg m-3 s-1],
                'cfl_number': Maximum CFL number [dimensionless],
                'stability_number': Diffusion stability number [dimensionless],
                'peclet_number': Maximum Peclet number [dimensionless],
                'conservation_error': Relative conservation error [dimensionless],
                'boundary_fluxes': Outward-positive face flux ledger [kg m-2 s-1]
            }
            
    References:
        - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations
        - Strang, G. (1968). On the construction and comparison of difference schemes
    """
    if not concentration.is_floating_point():
        raise TypeError("concentration must use a floating-point dtype")
    if not bool(torch.all(torch.isfinite(concentration))):
        raise ValueError("concentration must contain only finite values")
    if enforce_non_negative and bool(torch.any(concentration < 0.0)):
        raise ValueError(
            "concentration must be non-negative when enforce_non_negative is enabled"
        )
    if not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be finite and positive")
    spacing_values = _normalize_transport_grid_spacing(grid_spacing, spatial_dims)
    splitting_method = str(splitting_method).strip().lower()

    if advection_boundary_state is not None:
        boundary_input = torch.as_tensor(advection_boundary_state)
        if boundary_input.is_complex():
            raise TypeError(
                "advection_boundary_state must use a real floating-point dtype"
            )
        if not bool(torch.all(torch.isfinite(boundary_input))):
            raise ValueError(
                "advection_boundary_state must contain only finite values"
            )
        if enforce_non_negative and bool(torch.any(boundary_input < 0.0)):
            raise ValueError(
                "advection_boundary_state must be non-negative when "
                "enforce_non_negative is enabled"
            )

    # Source S is an explicit volumetric rate. It is applied once per full step.
    if source_term is None:
        source_term = torch.zeros_like(concentration)
    else:
        source_input = torch.as_tensor(source_term)
        if source_input.is_complex():
            raise TypeError("source_term must use a real floating-point dtype")
        source_term = torch.as_tensor(
            source_term,
            device=concentration.device,
            dtype=concentration.dtype,
        )
        try:
            source_term = torch.broadcast_to(source_term, concentration.shape)
        except RuntimeError as exc:
            raise ValueError(
                "source_term must be broadcast-compatible with concentration"
            ) from exc
        if not bool(torch.all(torch.isfinite(source_term))):
            raise ValueError("source_term must contain only finite values")

    dt_value = float(time_step)
    dt_tensor = concentration.new_tensor(dt_value)
    boundary_fluxes: Dict[str, Tensor]
    combined_stability_number = None
    
    if splitting_method == "sequential":
        # Sequential splitting: advection then diffusion
        adv_result = apply_advection(
            concentration, velocity, grid_spacing, spatial_dims,
            advection_scheme, boundary_method, time_step, check_stability,
            time_integration_method=time_integration_method,
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            boundary_state=advection_boundary_state,
        )
        c_intermediate = adv_result['concentration']
        
        diff_result = apply_diffusion(
            c_intermediate,
            diffusivity,
            grid_spacing,
            spatial_dims,
            boundary_method,
            time_step,
            check_stability,
            enforce_non_negative,
            run_conservation_check=run_conservation_check,
            time_integration_method=time_integration_method,
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            boundary_state=diffusion_boundary_state,
            diffusivity_layout=diffusivity_layout,
        )
        concentration_new = diff_result['concentration']
        
        concentration_new = concentration_new + dt_tensor * source_term
        
        # Collect diagnostics
        advection_rate = adv_result['advection_rate']
        diffusion_rate = diff_result['diffusion_rate']
        cfl_number = adv_result.get('cfl_number')
        stability_number = diff_result.get('stability_number')
        boundary_fluxes = _combine_boundary_flux_ledgers(
            (1.0, adv_result["boundary_fluxes"]),
            (1.0, diff_result["boundary_fluxes"]),
        )
        
    elif splitting_method == "symmetric":
        # Symmetric source-diffusion-advection composition.
        dt_half = time_step / 2.0
        concentration_source_half = concentration + 0.5 * dt_tensor * source_term
        if not bool(torch.all(torch.isfinite(concentration_source_half))):
            raise ValueError(
                "Symmetric source half-step must produce finite concentration values"
            )
        if enforce_non_negative and bool(torch.any(concentration_source_half < 0.0)):
            minimum = float(torch.min(concentration_source_half))
            raise ValueError(
                "Symmetric source half-step violated non-negative concentration: "
                f"minimum={minimum:.6g}"
            )
        
        # First diffusion half-step
        diff_result1 = apply_diffusion(
            concentration_source_half,
            diffusivity,
            grid_spacing,
            spatial_dims,
            boundary_method,
            dt_half,
            check_stability,
            enforce_non_negative=False,
            run_conservation_check=run_conservation_check,
            time_integration_method=time_integration_method,
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            boundary_state=diffusion_boundary_state,
            diffusivity_layout=diffusivity_layout,
        )
        c_step1 = diff_result1['concentration']
        
        # Advection full step
        adv_result = apply_advection(
            c_step1, velocity, grid_spacing, spatial_dims,
            advection_scheme, boundary_method, time_step, check_stability,
            time_integration_method=time_integration_method,
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            boundary_state=advection_boundary_state,
        )
        c_step2 = adv_result['concentration']
        
        # Second diffusion half-step
        diff_result2 = apply_diffusion(
            c_step2,
            diffusivity,
            grid_spacing,
            spatial_dims,
            boundary_method,
            dt_half,
            check_stability,
            enforce_non_negative,
            run_conservation_check=run_conservation_check,
            time_integration_method=time_integration_method,
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            boundary_state=diffusion_boundary_state,
            diffusivity_layout=diffusivity_layout,
        )
        concentration_new = (
            diff_result2['concentration'] + 0.5 * dt_tensor * source_term
        )
        
        # Collect diagnostics
        advection_rate = adv_result['advection_rate']
        diffusion_rate = (diff_result1['diffusion_rate'] + diff_result2['diffusion_rate']) / 2.0
        cfl_number = adv_result.get('cfl_number')
        first_stability = diff_result1.get('stability_number')
        second_stability = diff_result2.get('stability_number')
        stability_number = (
            None
            if first_stability is None or second_stability is None
            else (first_stability + second_stability) / 2.0
        )
        boundary_fluxes = _combine_boundary_flux_ledgers(
            (0.5, diff_result1["boundary_fluxes"]),
            (1.0, adv_result["boundary_fluxes"]),
            (0.5, diff_result2["boundary_fluxes"]),
        )
        
    elif splitting_method == "combined":
        # Combined is one explicit Euler step and cannot silently select an
        # implicit method from the global transport solver configuration.
        combined_method, _, _ = _resolve_transport_time_integration(
            time_integration_method,
            implicit_midpoint_max_iter,
            implicit_midpoint_tolerance,
        )
        _check_splitting_time_integration_pair(splitting_method, combined_method)

        adv_result = apply_advection(
            concentration, velocity, grid_spacing, spatial_dims,
            advection_scheme, boundary_method, time_step, check_stability,
            time_integration_method="euler",
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            boundary_state=advection_boundary_state,
        )
        diff_result = apply_diffusion(
            concentration,
            diffusivity,
            grid_spacing,
            spatial_dims,
            boundary_method,
            time_step,
            check_stability,
            False,
            run_conservation_check=False,
            time_integration_method="euler",
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            boundary_state=diffusion_boundary_state,
            diffusivity_layout=diffusivity_layout,
        )
        
        advection_rate = adv_result['advection_rate']
        diffusion_rate = diff_result['diffusion_rate']
        
        concentration_new = concentration + dt_tensor * (advection_rate + diffusion_rate + source_term)
        
        # Collect diagnostics
        cfl_number = adv_result.get('cfl_number')
        stability_number = diff_result.get('stability_number')
        if check_stability:
            if cfl_number is None or stability_number is None:
                raise RuntimeError(
                    "Combined explicit stability diagnostics are required"
                )
            diffusion_diagonal_factor = (
                3.0 if str(boundary_method).lower() == "dirichlet" else 2.0
            )
            combined_stability_number = (
                torch.max(cfl_number)
                + diffusion_diagonal_factor * torch.max(stability_number)
            )
            combined_limit = float(
                get_global_manager().get(
                    "transport.numerical.cfl_advection_max",
                    expected_type=float,
                    required=True,
                )
            )
            if float(combined_stability_number) > combined_limit:
                raise ValueError(
                    "Combined explicit stability violation: advection CFL plus "
                    "the diffusion diagonal weight is "
                    f"{float(combined_stability_number):.6g}, exceeding "
                    f"{combined_limit:.6g}. Reduce time_step or use splitting."
                )
        boundary_fluxes = _combine_boundary_flux_ledgers(
            (1.0, adv_result["boundary_fluxes"]),
            (1.0, diff_result["boundary_fluxes"]),
        )
    
    else:
        raise ValueError(f"Unknown splitting method: {splitting_method}")
    
    if not bool(torch.all(torch.isfinite(concentration_new))):
        raise ValueError(
            "Advection-diffusion step must produce finite concentration values"
        )
    if enforce_non_negative and bool(torch.any(concentration_new < 0.0)):
        minimum = float(torch.min(concentration_new))
        raise ValueError(
            "Advection-diffusion step violated non-negative concentration: "
            f"minimum={minimum:.6g}. Reduce time_step, revise the source sink, "
            "or use a positivity-preserving integration method."
        )

    conservation_error = None
    if run_conservation_check:
        conservation_error = _finite_volume_transport_conservation_error(
            concentration,
            concentration_new,
            source_term,
            dt_value,
            spacing_values,
            spatial_dims,
            boundary_fluxes,
            warn_on_violation=True,
        )
    
    peclet_number = _compute_max_peclet_number(
        concentration,
        velocity,
        diffusivity,
        spacing_values,
        spatial_dims,
        diffusivity_layout,
    )
    
    # Prepare result
    result = {
        'concentration': concentration_new,
        'advection_rate': advection_rate,
        'diffusion_rate': diffusion_rate,
        'conservation_error': conservation_error,
        'peclet_number': peclet_number,
        'boundary_fluxes': boundary_fluxes,
    }
    
    if cfl_number is not None:
        result['cfl_number'] = cfl_number
    if stability_number is not None:
        result['stability_number'] = stability_number
    if combined_stability_number is not None:
        result['combined_stability_number'] = combined_stability_number
    
    return result


# Export all functions and classes
__all__ = [
    'compute_gradient_1d',
    'compute_gradient_2d',
    'compute_gradient_3d',
    'compute_divergence_2d',
    'compute_divergence_3d',
    'compute_conservative_divergence_2d',
    'compute_conservative_divergence_3d',
    'compute_laplacian_1d',
    'compute_laplacian_2d',
    'compute_laplacian_3d',
    'compute_grid_metrics',
    'compute_cfl_number',
    'compute_diffusion_stability',
    'check_conservation',
    'apply_non_negative_constraint',
    'apply_diffusion',
    'apply_advection',
    'advance_advection_diffusion'
]
