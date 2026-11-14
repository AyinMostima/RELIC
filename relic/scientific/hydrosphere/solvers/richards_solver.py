"""Richards equation numerical solvers extracted from RichardsSoilMoisture."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..soil_moisture.richards import RichardsSoilMoisture


def on_device_of(value: Any, reference: Any) -> Any:
    """
    Return a held hydraulic value on the device of the tensor it meets.

    A ParameterManager binding is held by reference and has one device,
    that of the latest registration of any Functor holding it, and a value
    derived from a binding at use (m_vg, layer_thickness) has the device
    of that binding. The Richards kernel and this solver compute on the
    device of the state profile, so a held tensor on another device is
    read through a differentiable copy; the binding itself is never moved
    here (docs/source/architecture_overview.md Sections 4.2 and 4.4.2).

    Parameters
    ----------
    value : Any
        Held parameter, derived parameter or plain number.
    reference : Any
        State-side operand the value is combined with.

    Returns
    -------
    Any
        ``value`` itself (identity and dtype kept) unless both arguments are
        tensors on different devices; then its copy on the device of
        ``reference``.
    """
    if (
        isinstance(value, Tensor)
        and isinstance(reference, Tensor)
        and value.device != reference.device
    ):
        return value.to(device=reference.device)
    return value


def harmonic_interface_conductivity(
    conductivity: Tensor, harmonic_mean_factor: Any
) -> Tensor:
    """
    Return the interior interface conductivities of a layer profile.

    Parameters
    ----------
    conductivity : Tensor
        Cell hydraulic conductivities K_1..K_n [m s-1], ordered downward.
    harmonic_mean_factor : Any
        hydrosphere.numerical.harmonic_mean_factor [1]; 2 gives the
        harmonic mean.

    Returns
    -------
    Tensor
        K_{i+1/2} for the n - 1 interior interfaces [m s-1].
    """
    # References: Belfort, B., Younes, A., Fahs, M. and Lehmann, F. (2013),
    #   J. Hydrol. 505:202-217, Table 2 (harmonic interblock mean
    #   K = 2 [K_i^-1 + K_j^-1]^-1; read in the HAL draft hal-02307246);
    #   Celia, M.A., Bouloutas, E.T. and Zarba, R.L. (1990), Water Resour.
    #   Res. 26(7):1483-1496 (mass-conservative mixed-form cell balance
    #   that consumes the interface flux).
    # Mathematical expression:
    #   K_{i+1/2} = f K_i [K_{i+1} / (K_i + K_{i+1})] for K_i + K_{i+1} > 0,
    #   K_{i+1/2} = 0 for K_i = K_{i+1} = 0 (limit when a neighbour vanishes).
    # Parameter meanings: K_i, K_{i+1} conductivities of the cells above and
    #   below the interface; f the harmonic mean factor.
    # Units: K [m s-1]; f [1]; the quotient K_{i+1} / (K_i + K_{i+1}) [1].
    # Domain/constraints: K >= 0. No constant is added to K_i + K_{i+1}: it
    #   would carry m s-1 and turn the mean into K^2 / (K + eps / 2) at dry-end
    #   conductivities. The quotient form returns K exactly for equal K of any
    #   magnitude and never forms K_i K_{i+1}, which underflows first. The
    #   0/0 case uses a unit denominator inside torch.where so that autograd
    #   (exact Newton Jacobian) stays finite.
    # Sign convention: K_{i+1/2} >= 0 is a magnitude; flux direction comes
    #   from the head gradient, q = -K_{i+1/2} (dh/dz - 1), positive downward.
    upper = conductivity[:-1]
    lower = conductivity[1:]
    total = upper + lower
    positive = total > 0
    safe_total = torch.where(positive, total, torch.ones_like(total))
    lower_share = torch.where(
        positive, lower / safe_total, torch.zeros_like(total)
    )
    return harmonic_mean_factor * upper * lower_share


class RichardsNumericalSolver:
    """Numerical solvers for Richards equation delegated from RichardsSoilMoisture."""
    def __init__(self, functor: "RichardsSoilMoisture") -> None:
        self._functor = functor

    def __getattr__(self, name: str):
        return getattr(self._functor, name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_functor":
            object.__setattr__(self, name, value)
        else:
            setattr(self._functor, name, value)

    def _solve_linearized_system_picard(self, 
                                      theta_old: Tensor,
                                      theta_iter: Tensor, 
                                      h_iter: Tensor,
                                      k_iter: Tensor,
                                      dt: float,
                                      surface_flux: float,
                                      bottom_flux: Tensor | None = None,
                                      ) -> tuple[Tensor, bool]:
        """
        Compute one mass-conservative modified-Picard correction.

        The correction is solved in pressure-head space and mapped back to
        water content with the retention capacity. The enclosing Richards
        step owns iteration and convergence.
        
        Mathematical Expression
        -----------------------
        A_K delta_h = -R(theta^m),
        theta^(m+1) = theta^m + C(theta^m) delta_h,

        where C = dtheta/dh and R is the same mixed-form residual used by
        Newton-Raphson.
        """
        n = self.soil_layers
        dz = on_device_of(self.layer_thickness, theta_iter)
        residual = self._compute_newton_residual(
            theta_iter,
            theta_old,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=bottom_flux,
        )

        # References: Celia et al. (1990), modified Picard mixed form.
        # Mathematical expression: C = dtheta/dh = 1 / (dh/dtheta).
        # Parameter meanings: C [m-1], theta [m3 m-3], h [m].
        # Units: C [m-1]; dh/dtheta [m]; saturation_epsilon in the units of dh/dtheta.
        # Domain/constraints: dh/dtheta is floored at saturation_epsilon before the reciprocal.
        # Sign convention: C >= 0, water content rises with pressure head.
        dh_dtheta = self._compute_dh_dtheta(theta_iter)
        capacity = torch.reciprocal(
            torch.clamp(dh_dtheta, min=self.saturation_epsilon)
        )

        # Interface conductivity: see harmonic_interface_conductivity
        # (Belfort et al. 2013, Table 2); identical to the residual path.
        k_interface = k_iter.new_zeros(n + 1)
        k_interface[1:-1] = harmonic_interface_conductivity(
            k_iter, self.harmonic_mean_factor
        )
        k_interface[0] = k_iter[0]
        k_interface[-1] = k_iter[-1]

        a = theta_old.new_zeros(n)
        b = capacity.clone()
        c = theta_old.new_zeros(n)
        rhs = -residual
        top_is_head = (
            self._top_boundary is not None
            and self._top_boundary["type"] == "prescribed_head"
        )
        bottom_is_head = (
            self._bottom_boundary is not None
            and self._bottom_boundary["type"] == "prescribed_head"
        )

        for i in range(n):
            if i == 0:
                if top_is_head:
                    b[i] = 1.0
                    c[i] = 0.0
                    rhs[i] = 0.0
                else:
                    coefficient = dt * k_interface[1] / (dz * dz)
                    b[i] += coefficient
                    c[i] = -coefficient
            elif i == n - 1:
                if bottom_is_head:
                    a[i] = 0.0
                    b[i] = 1.0
                    rhs[i] = 0.0
                else:
                    coefficient = dt * k_interface[i] / (dz * dz)
                    a[i] = -coefficient
                    b[i] += coefficient
            else:
                coefficient_up = dt * k_interface[i] / (dz * dz)
                coefficient_down = dt * k_interface[i + 1] / (dz * dz)
                a[i] = -coefficient_up
                b[i] += coefficient_up + coefficient_down
                c[i] = -coefficient_down

        if self.thomas_solver is None:
            raise RuntimeError("Thomas solver is required for modified Picard.")
        delta_head = self.thomas_solver.solve(a=a, b=b, c=c, d=rhs).solution
        theta_candidate = theta_iter + capacity * delta_head

        if top_is_head:
            top_value = self._top_boundary["value"]
            theta_candidate = theta_candidate.clone()
            theta_candidate[0] = self._ensure_tensor(
                self._compute_theta_from_head(top_value)
            ).to(device=theta_candidate.device, dtype=theta_candidate.dtype)
        if bottom_is_head:
            bottom_value = self._bottom_boundary["value"]
            theta_candidate = theta_candidate.clone()
            theta_candidate[-1] = self._ensure_tensor(
                self._compute_theta_from_head(bottom_value)
            ).to(device=theta_candidate.device, dtype=theta_candidate.dtype)

        # References: Celia et al. (1990); Kelley (2003).
        # Mathematical expression:
        #   delta_theta_res = eps * ||theta||_inf,
        #   R_res = ||J||_inf * delta_theta_res.
        # Parameter meanings: eps is machine precision [1], theta is water
        # content [m3 m-3], J is the exact mixed-residual Jacobian [1], and
        # R is the mixed residual [m3 m-3]. The exact Jacobian is evaluated
        # only when the modified-Picard correction reaches state precision.
        state_scale = torch.max(torch.abs(theta_iter))
        state_resolution = (
            torch.finfo(theta_iter.dtype).eps
            * torch.clamp(state_scale, min=torch.finfo(theta_iter.dtype).tiny)
        )
        if bool(
            torch.max(torch.abs(theta_candidate - theta_iter))
            <= state_resolution
        ):
            jacobian = self._compute_newton_jacobian(
                theta_iter,
                theta_old,
                dt,
                surface_flux,
                bottom_flux=bottom_flux,
            )
            residual_resolution = (
                torch.max(torch.sum(torch.abs(jacobian), dim=1))
                * state_resolution
            )
            if bool(torch.max(torch.abs(residual)) <= residual_resolution):
                return theta_candidate, True
        return theta_candidate, False
    
    def _solve_linearized_system_newton(self,
                                      theta_old: Tensor,
                                      theta_iter: Tensor,
                                      h_iter: Tensor, 
                                      k_iter: Tensor,
                                      dt: float,
                                      surface_flux: float,
                                      bottom_flux: Tensor | None = None,
                                      ) -> tuple[Tensor, bool]:
        """
        Solve linearized Richards equation using Newton-Raphson iteration.
        
        Computes one damped Newton correction for the discretized Richards
        residual. The enclosing Richards step owns the nonlinear iteration
        count and convergence decision.
        
        Parameters
        ----------
        theta_old : Tensor
            Soil moisture at previous time step [m3 m-3]
        theta_iter : Tensor
            Current Newton iterate [m3 m-3]
        h_iter : Tensor
            Pressure head at current iterate [m]
        k_iter : Tensor
            Hydraulic conductivity at current iterate [m s-1]
        dt : float
            Time step [s]
        surface_flux : float
            Surface boundary flux [m s-1]
        Returns
        -------
        tuple[Tensor, bool]
            Updated soil moisture profile [m3 m-3] and whether the Newton
            correction reached the representable precision of the state.
            
        Mathematical Expression
        -----------------------
        Newton-Raphson iteration:
        theta^{k+1} = theta^k - J^{-1} * R(theta^k)
        
        where:
        R(theta) = theta - theta_old - dt * [spatial_discretization_terms]
        J = partialR/partialtheta = I - dt * partial[spatial_terms]/partialtheta
        
        References
        ----------
        - Celia et al. (1990). Water Resources Research, 26(7), 1483-1496
        - Lehmann & Ackerer (1998). Advances in Water Resources, 21(2), 97-107
        """
        residual = self._compute_newton_residual(
            theta_iter,
            theta_old,
            h_iter,
            k_iter,
            dt,
            surface_flux,
            bottom_flux=bottom_flux,
        )
        residual_norm = torch.max(torch.abs(residual))

        # References: Paniconi and Putti (1994); Kelley (2003).
        # Mathematical expression: J(theta_k) delta_theta = -R(theta_k),
        # theta_{k+1} = clamp(theta_k + alpha delta_theta, theta_r, theta_s).
        # Parameter meanings: R is the mixed-form residual [m3 m-3], J is
        # dR/dtheta [1], alpha is the Armijo step [1], and theta bounds are
        # residual and saturated water contents [m3 m-3].
        jacobian = self._compute_newton_jacobian(
            theta_iter,
            theta_old,
            dt,
            surface_flux,
            bottom_flux=bottom_flux,
        )
        try:
            delta_theta = self._solve_newton_linear_system(jacobian, -residual)
        except torch.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Richards Newton linear solve failed; no Picard fallback is allowed."
            ) from exc

        # References: Kelley (2003), step-based Newton convergence.
        # Mathematical expression:
        #   delta_theta_res = eps * ||theta||_inf,
        #   R_res = ||J||_inf * delta_theta_res.
        # Parameter meanings: eps is machine precision [1], theta is water
        # content [m3 m-3], J=dR/dtheta [1], and R is the mixed residual
        # [m3 m-3]. This recognizes a converged state only when both the
        # Newton correction and residual are below the representable floor.
        state_scale = torch.max(torch.abs(theta_iter))
        state_resolution = (
            torch.finfo(theta_iter.dtype).eps
            * torch.clamp(state_scale, min=torch.finfo(theta_iter.dtype).tiny)
        )
        jacobian_sensitivity = torch.max(torch.sum(torch.abs(jacobian), dim=1))
        residual_resolution = jacobian_sensitivity * state_resolution
        if (
            bool(torch.max(torch.abs(delta_theta)) <= state_resolution)
            and bool(residual_norm <= residual_resolution)
        ):
            return theta_iter, True

        alpha = self._newton_line_search(
            theta_iter,
            delta_theta,
            theta_old,
            dt,
            surface_flux,
            residual_norm,
            bottom_flux=bottom_flux,
        )
        return (
            torch.clamp(
                theta_iter + alpha * delta_theta,
                min=on_device_of(self.theta_residual, theta_iter),
                max=on_device_of(self.theta_saturated, theta_iter),
            ),
            False,
        )

    def _compute_newton_residual(self, 
                                theta: Tensor,
                                theta_old: Tensor,
                                h_iter: Tensor,
                                k_iter: Tensor,
                                dt: float,
                                surface_flux: float,
                                bottom_flux: Tensor | None = None,
                                ) -> Tensor:
        """
        Compute residual function R(theta) for Newton-Raphson method.
        
        Mathematical Expression
        -----------------------
        R(theta) = theta - theta_old - dt * [nabla*(K(theta)nablah) - nabla*K(theta)e_z] - dt * S(theta)
        
        where the spatial discretization follows finite difference scheme
        """
        n = self.soil_layers
        dz = on_device_of(self.layer_thickness, theta)
        residual = theta.new_zeros(n)
        
        # Current hydraulic properties
        h = self._compute_pressure_head(theta)
        k = self._compute_hydraulic_conductivity(theta)

        # Interface conductivities: see harmonic_interface_conductivity
        # (Belfort et al. 2013, Table 2); identical to the Picard matrix.
        k_interface = k.new_zeros(n + 1)
        k_interface[1:-1] = harmonic_interface_conductivity(
            k, self.harmonic_mean_factor
        )
        k_interface[0] = k[0]    # Top boundary
        k_interface[-1] = k[-1]  # Bottom boundary
        
        # Time derivative term
        residual = theta - theta_old
        
        # Spatial discretization terms
        for i in range(n):
            if i == 0:  # Top boundary node
                # Apply top boundary condition
                if self._top_boundary is not None:
                    boundary_type = self._top_boundary["type"]
                    if boundary_type == "prescribed_flux":
                        flux_value = self._top_boundary["value"]
                        residual[i] -= dt * flux_value / dz
                    elif boundary_type == "prescribed_head":
                        # For prescribed head, residual becomes algebraic constraint
                        head_value = self._top_boundary["value"]
                        theta_prescribed = self._compute_theta_from_head(head_value)
                        residual[i] = theta[i] - theta_prescribed
                        continue
                    else:
                        # Atmospheric mode consumes the canonical net surface flux.
                        residual[i] -= dt * surface_flux / dz
                else:
                    # Default: prescribed flux
                    residual[i] -= dt * surface_flux / dz
                
                # Flux to next layer
                if i < n - 1:
                    flux_down = -k_interface[i+1] * ((h[i+1] - h[i]) / dz - 1.0)
                    residual[i] += dt * flux_down / dz
                
            elif i == n - 1:  # Bottom boundary node
                # Flux from previous layer
                flux_up = -k_interface[i] * ((h[i] - h[i-1]) / dz - 1.0)
                residual[i] -= dt * flux_up / dz
                
                # Apply bottom boundary condition
                if self._bottom_boundary is not None:
                    boundary_type = self._bottom_boundary["type"]
                    if boundary_type == "prescribed_flux":
                        flux_value = self._bottom_boundary["value"]
                        residual[i] += dt * flux_value / dz
                    elif boundary_type == "capillary_flux":
                        if bottom_flux is None:
                            raise ValueError(
                                "Capillary lower boundary requires an explicit "
                                "downward-positive bottom flux."
                            )
                        residual[i] += dt * bottom_flux / dz
                    elif boundary_type == "prescribed_head":
                        head_value = self._bottom_boundary["value"]
                        theta_prescribed = self._compute_theta_from_head(head_value)
                        residual[i] = theta[i] - theta_prescribed
                        continue
                    else:
                        # Free drainage: dh/dz = 0 gives q_bottom = K_bottom.
                        flux_down = k[i]
                        residual[i] += dt * flux_down / dz
                else:
                    # Default free drainage uses the same downward-positive flux.
                    flux_down = k[i]
                    residual[i] += dt * flux_down / dz
                    
            else:  # Internal nodes
                # Flux from layer above
                flux_up = -k_interface[i] * ((h[i] - h[i-1]) / dz - 1.0)
                residual[i] -= dt * flux_up / dz
                
                # Flux to layer below
                flux_down = -k_interface[i+1] * ((h[i+1] - h[i]) / dz - 1.0)
                residual[i] += dt * flux_down / dz
        
        return residual

    def _compute_newton_jacobian(self,
                                theta: Tensor,
                                theta_old: Tensor,
                                dt: float,
                                surface_flux: float,
                                bottom_flux: Tensor | None = None) -> Tensor:
        """Return the exact Jacobian of the canonical mixed-form residual.

        References:
          - Paniconi and Putti (1994), Newton iteration for Richards flow.
          - Kelley (2003), residual-based Newton methods.
        Mathematical expression:
          J_ij(theta) = partial R_i(theta) / partial theta_j.
        Parameter meanings:
          theta and theta_old are water-content profiles [m3 m-3], dt is [s],
          and surface_flux/bottom_flux are downward-positive [m s-1].
        """

        def residual_function(candidate: Tensor) -> Tensor:
            pressure_head = self._compute_pressure_head(candidate)
            conductivity = self._compute_hydraulic_conductivity(candidate)
            return self._compute_newton_residual(
                candidate,
                theta_old,
                pressure_head,
                conductivity,
                dt,
                surface_flux,
                bottom_flux=bottom_flux,
            )

        theta_for_jacobian = theta
        if not theta_for_jacobian.requires_grad:
            theta_for_jacobian = theta_for_jacobian.clone().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            residual_function,
            theta_for_jacobian,
            create_graph=torch.is_grad_enabled(),
            strict=False,
        )

    def _solve_newton_linear_system(self, jacobian: Tensor, rhs: Tensor) -> Tensor:
        """
        Solve linear system Jx = b using appropriate method.
        
        Uses Thomas algorithm for tridiagonal systems when possible,
        otherwise falls back to LU decomposition.
        """
        n = jacobian.shape[0]
        
        # Check if system is tridiagonal or nearly tridiagonal
        if self._is_nearly_tridiagonal(jacobian):
            # Extract diagonals for Thomas algorithm
            a = rhs.new_zeros(n)  # Lower diagonal
            b = rhs.new_zeros(n)  # Main diagonal
            c = rhs.new_zeros(n)  # Upper diagonal
            
            # Fill diagonals
            b = torch.diag(jacobian)
            if n > 1:
                a[1:] = torch.diag(jacobian, -1)  # Lower diagonal
                c[:-1] = torch.diag(jacobian, 1)   # Upper diagonal
            
            # Use Thomas algorithm using unified solver system
            if self.thomas_solver is not None:
                # Use unified Thomas algorithm solver
                result = self.thomas_solver.solve(a=a, b=b, c=c, d=rhs)
                return result.solution
            else:
                # Unified solver system should always be available
                raise RuntimeError("Thomas solver not available. This should not happen in the unified architecture.")
        else:
            # Use general linear algebra solver
            return torch.linalg.solve(jacobian, rhs)

    def _newton_line_search(self,
                           theta_current: Tensor,
                           delta_theta: Tensor,
                           theta_old: Tensor,
                           dt: float,
                           surface_flux: float,
                           residual_norm_old: float,
                           bottom_flux: Tensor | None = None) -> float:
        """
        Perform line search for Newton method to ensure convergence.
        
        Uses simple backtracking line search with Armijo condition.
        
        Returns
        -------
        float
            Step length alpha in (0, 1]
        """
        alpha = self.line_search_initial_alpha
        max_backtracks = self.line_search_max_backtracks
        armijo_constant = self.line_search_armijo_constant
        
        for i in range(max_backtracks):
            # Trial step
            theta_trial = theta_current + alpha * delta_theta
            theta_trial = torch.clamp(
                theta_trial,
                min=on_device_of(self.theta_residual, theta_trial),
                max=on_device_of(self.theta_saturated, theta_trial),
            )
            
            # Compute residual at trial point
            h_trial = self._compute_pressure_head(theta_trial)
            k_trial = self._compute_hydraulic_conductivity(theta_trial)
            
            residual_trial = self._compute_newton_residual(
                theta_trial,
                theta_old,
                h_trial,
                k_trial,
                dt,
                surface_flux,
                bottom_flux=bottom_flux,
            )
            residual_norm_trial = torch.max(torch.abs(residual_trial))
            
            # Armijo condition: sufficient decrease
            if residual_norm_trial <= (1.0 - armijo_constant * alpha) * residual_norm_old:
                return alpha
            
            # Reduce step size
            alpha *= self.line_search_alpha_reduction
            
            if alpha < self.line_search_min_alpha:
                break
        
        raise RuntimeError(
            "Richards Newton line search failed to satisfy the Armijo condition."
        )

    def _is_nearly_tridiagonal(self, matrix: Tensor, tol: float = None) -> bool:
        """Check if matrix is nearly tridiagonal."""
        if tol is None:
            tol = self.tridiagonal_tolerance
        n = matrix.shape[0]
        if n <= 2:
            return True
        
        # Check if elements outside the tridiagonal band are small
        mask = torch.ones_like(matrix, dtype=torch.bool)
        
        # Set tridiagonal elements to False
        mask.fill_diagonal_(False)  # Main diagonal
        if n > 1:
            mask[1:, :-1].fill_diagonal_(False)  # Lower diagonal
            mask[:-1, 1:].fill_diagonal_(False)  # Upper diagonal
        
        # Check if remaining elements are below tolerance
        return torch.max(torch.abs(matrix[mask])) < tol
