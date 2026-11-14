"""Bracketed nonlinear scalar root solvers for the unified solver registry.

The implementation solves independent scalar equations over arbitrary tensor
batches. It keeps the forward bracket on the input device and exposes a
implicit parameter gradients with support for second-order differentiation.
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
from torch import Tensor

from .base import (
    ConfigDict,
    SolverBase,
    SolverFamily,
    SolverMetadata,
    SolverResult,
    SolverStatus,
)
from .registry import register_solver


class _ImplicitRootGradient(torch.autograd.Function):
    """Attach differentiable implicit gradients to independent simple roots."""

    @staticmethod
    def forward(
        ctx: Any,
        root: Tensor,
        residual_fn: Callable[..., Tensor],
        *parameters: Tensor,
    ) -> Tensor:
        ctx.residual_fn = residual_fn
        ctx.parameter_requires_grad = tuple(
            parameter.requires_grad for parameter in parameters
        )
        output_root = root.clone()
        ctx.save_for_backward(output_root, *parameters)
        return output_root

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        grad_output = grad_outputs[0]
        saved = ctx.saved_tensors
        root = saved[0]
        parameters = saved[1:]

        # References: Blondel et al. (2022), modular implicit differentiation.
        # Mathematical expression: F(root, theta)=0; root_theta=-F_theta/F_root;
        #   theta_bar=F_theta.T*(-root_bar/F_root) for independent scalar roots.
        # Parameter meanings: explicit theta tensors may be scalar or root-shaped;
        #   VJP sums contributions for a shared scalar parameter.
        # Units: the implicit derivative has root units divided by parameter units.
        # Domain/constraints: smooth simple roots, nonzero F_root, independent batches.
        #   Saving the returned root retains its outer graph for second derivatives;
        #   VJP reuses one residual graph without a dense batch Jacobian.
        # Sign convention: the minus sign follows differentiation of F=0.
        residual_graph = torch.func.vjp(ctx.residual_fn, root, *parameters)
        residual, pullback = residual_graph[0], residual_graph[1]
        if not isinstance(residual, Tensor) or residual.shape != root.shape:
            raise RuntimeError("Implicit root residual must return a Tensor with root shape")
        residual_gradient = pullback(torch.ones_like(residual))[0]
        if not bool(torch.all(torch.isfinite(residual_gradient))):
            raise RuntimeError("Implicit root derivative must be finite")
        if bool(torch.any(residual_gradient == 0)):
            raise RuntimeError("Implicit root derivative must be nonzero")
        gradients = pullback(-grad_output / residual_gradient)[1:]
        parameter_gradients = tuple(
            gradient if requires_grad else None
            for gradient, requires_grad in zip(gradients, ctx.parameter_requires_grad)
        )
        return (None, None, *parameter_gradients)


@register_solver("bracketed_bisection", family=SolverFamily.NONLINEAR)
class BracketedBisectionSolver(SolverBase):
    """Solve batched independent scalar roots with a strict sign bracket."""

    solver_family = SolverFamily.NONLINEAR

    def __init__(
        self,
        name: str = "bracketed_bisection",
        operator: Optional[Any] = None,
        config: Optional[ConfigDict] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if metadata is None:
            metadata = SolverMetadata(
                name="Bracketed Bisection",
                solver_family=SolverFamily.NONLINEAR,
                description=(
                    "Device-native bisection for batches of independent scalar "
                    "equations with second-order implicit parameter gradients"
                ),
                mathematical_expression=(
                    "F(x, theta) = 0, F(a)F(b) <= 0; "
                    "dx/dtheta = -(dF/dx)^-1 dF/dtheta"
                ),
                order=1,
                stability_region="Guaranteed bracket contraction for continuous F",
                is_implicit=True,
                requires_jacobian=False,
                references=[
                    "Burden, R.L. and Faires, J.D. (2010). Numerical Analysis.",
                    "Blondel, M. et al. (2022). Efficient and Modular Implicit Differentiation.",
                ],
                parameters={
                    "max_iterations": "Maximum bracket contractions",
                    "rtol": "Relative root interval tolerance",
                    "atol": "Absolute root interval tolerance",
                },
            )
        super().__init__(
            name=name,
            operator=operator,
            config=config,
            metadata=metadata,
            device=device,
            dtype=dtype,
            parameter_namespace="computation.solvers.bracketed_bisection",
        )

    def _validate_inputs(
        self,
        residual_fn: Callable[..., Tensor],
        lower: Tensor,
        upper: Tensor,
        parameters: Sequence[Tensor],
    ) -> tuple[Tensor, ...]:
        if not callable(residual_fn):
            raise TypeError("residual_fn must be callable")
        if not isinstance(lower, Tensor) or not isinstance(upper, Tensor):
            raise TypeError("lower and upper must be torch.Tensor instances")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have identical shape")
        if lower.numel() == 0:
            raise ValueError("root brackets must not be empty")
        if lower.dtype != upper.dtype or lower.dtype != self.dtype:
            raise ValueError("lower and upper dtype must match the solver dtype")
        if not lower.is_floating_point():
            raise ValueError("root brackets must use a floating-point dtype")
        solver_device_matches = (
            lower.device.type == self.device.type
            and (
                self.device.index is None
                or lower.device.index == self.device.index
            )
        )
        if lower.device != upper.device or not solver_device_matches:
            raise ValueError("lower and upper device must match the solver device")
        if not bool(torch.all(torch.isfinite(lower))):
            raise ValueError("lower bracket values must be finite")
        if not bool(torch.all(torch.isfinite(upper))):
            raise ValueError("upper bracket values must be finite")
        if bool(torch.any(lower >= upper)):
            raise ValueError("lower bracket values must be less than upper values")

        parameter_tuple = tuple(parameters)
        for parameter in parameter_tuple:
            if not isinstance(parameter, Tensor):
                raise TypeError("root parameters must be explicit torch.Tensor values")
            if parameter.dtype != lower.dtype:
                raise ValueError("root parameter dtype must match bracket dtype")
            if parameter.device != lower.device:
                raise ValueError("root parameter device must match bracket device")
            if parameter.shape not in {torch.Size([]), lower.shape}:
                raise ValueError(
                    "root parameter shape must be scalar or match bracket shape"
                )
            if not bool(torch.all(torch.isfinite(parameter))):
                raise ValueError("root parameters must contain only finite values")
        return parameter_tuple

    @staticmethod
    def _evaluate_residual(
        residual_fn: Callable[..., Tensor],
        point: Tensor,
        parameters: tuple[Tensor, ...],
    ) -> Tensor:
        residual = residual_fn(point, *parameters)
        if not isinstance(residual, Tensor):
            raise TypeError("residual_fn must return a torch.Tensor")
        if residual.shape != point.shape:
            raise ValueError("residual shape must match bracket shape")
        if residual.dtype != point.dtype:
            raise ValueError("residual dtype must match bracket dtype")
        if residual.device != point.device:
            raise ValueError("residual device must match bracket device")
        if not bool(torch.all(torch.isfinite(residual))):
            raise FloatingPointError("residual must contain only finite values")
        return residual

    # References: Burden and Faires (2010), bisection convergence theorem.
    # Mathematical expression: m_k = (a_k+b_k)/2 and retain the sign-changing half.
    # Parameter meanings: a_k and b_k bracket each independent scalar root;
    # rtol and atol bound the final interval in the units of the solved variable.
    # Open route: this numerical closure creates no physical source or sink.
    def solve(
        self,
        residual_fn: Callable[..., Tensor],
        lower: Tensor,
        upper: Tensor,
        *,
        parameters: Sequence[Tensor] = (),
    ) -> SolverResult:
        """Return one root for every element in a sign-changing bracket."""
        parameter_tuple = self._validate_inputs(
            residual_fn,
            lower,
            upper,
            parameters,
        )
        max_iterations = int(self.max_iterations)
        rtol = float(self.rtol)
        atol = float(self.atol)
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")
        if not torch.isfinite(torch.tensor([rtol, atol])).all() or rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be finite and non-negative")

        with torch.no_grad():
            final_lower = lower.detach().clone()
            final_upper = upper.detach().clone()
            residual_lower = self._evaluate_residual(
                residual_fn,
                final_lower,
                parameter_tuple,
            )
            residual_upper = self._evaluate_residual(
                residual_fn,
                final_upper,
                parameter_tuple,
            )
            lower_root = residual_lower == 0
            upper_root = residual_upper == 0
            sign_change = torch.signbit(residual_lower) != torch.signbit(
                residual_upper
            )
            valid_bracket = lower_root | upper_root | sign_change
            if not bool(torch.all(valid_bracket)):
                raise ValueError(
                    "root bracket must contain an endpoint root or a sign change"
                )

            solution = torch.where(lower_root, final_lower, final_upper)
            converged = lower_root | upper_root
            active = ~converged
            iterations_per_element = torch.zeros_like(
                lower,
                dtype=torch.long,
            )

            for iteration in range(max_iterations):
                midpoint = (final_lower + final_upper) * 0.5
                residual_midpoint = self._evaluate_residual(
                    residual_fn,
                    midpoint,
                    parameter_tuple,
                )
                iterations_per_element = torch.where(
                    active,
                    torch.full_like(iterations_per_element, iteration + 1),
                    iterations_per_element,
                )
                interval_tolerance = atol + rtol * torch.abs(midpoint)
                midpoint_root = residual_midpoint == 0
                interval_closed = (
                    0.5 * torch.abs(final_upper - final_lower)
                    <= interval_tolerance
                )
                converged_now = active & (midpoint_root | interval_closed)
                solution = torch.where(converged_now, midpoint, solution)

                remaining = active & ~converged_now
                same_sign_as_lower = torch.signbit(residual_midpoint) == torch.signbit(
                    residual_lower
                )
                move_lower = remaining & same_sign_as_lower
                move_upper = remaining & ~same_sign_as_lower
                final_lower = torch.where(move_lower, midpoint, final_lower)
                residual_lower = torch.where(
                    move_lower,
                    residual_midpoint,
                    residual_lower,
                )
                final_upper = torch.where(move_upper, midpoint, final_upper)
                residual_upper = torch.where(
                    move_upper,
                    residual_midpoint,
                    residual_upper,
                )
                converged = converged | converged_now
                active = remaining

            final_midpoint = (final_lower + final_upper) * 0.5
            solution = torch.where(active, final_midpoint, solution)
            residual = self._evaluate_residual(
                residual_fn,
                solution,
                parameter_tuple,
            )
            success = bool(torch.all(converged))
            status = (
                SolverStatus.SUCCESS
                if success
                else SolverStatus.MAX_ITERATIONS
            )
            iterations = int(torch.max(iterations_per_element).item())
            residual_norm = float(torch.max(torch.abs(residual)).item())

        if success and any(parameter.requires_grad for parameter in parameter_tuple):
            solution = _ImplicitRootGradient.apply(
                solution,
                residual_fn,
                *parameter_tuple,
            )

        return SolverResult(
            solution=solution,
            status=status,
            iterations=iterations,
            residual_norm=residual_norm,
            metadata={
                "method": "bracketed_bisection",
                "gradient": "second_order_implicit",
                "converged_mask": converged,
                "iterations_per_element": iterations_per_element,
                "final_lower": final_lower,
                "final_upper": final_upper,
                "residual": residual,
            },
        )


__all__ = ["BracketedBisectionSolver"]
