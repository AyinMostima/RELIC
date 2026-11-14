"""
Linear System Solvers for RELIC Unified Computation Layer

Direct and iterative solvers for linear systems Ax = b, with specialized
implementations for tridiagonal and sparse systems commonly encountered in
biogeochemical modeling.

Mathematical Foundation:
    Linear system: Ax = b
    where A is coefficient matrix, b is right-hand side, x is unknown
    
    Solution methods:
    - Direct: LU decomposition, Cholesky, Thomas algorithm
    - Iterative: Jacobi, Gauss-Seidel, CG, GMRES
    
    Tridiagonal systems (Thomas algorithm):
    [b0 c0  0  0 ...]     [x0]     [d0]
    [a1 b1 c1  0 ...]     [x1]     [d1]
    [ 0 a2 b2 c2 ...]  *  [x2]  =  [d2]
    [... ... ... ...]     [...]    [...]

Applications:
    - Richards equation finite difference discretization
    - 1D soil heat/moisture transport
    - Implicit diffusion operators
    - Boundary value problems in biogeochemistry

References:
    - Thomas, L.H. (1949). Elliptic problems in linear difference equations
    - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis. Brooks/Cole
    - Trefethen, L.N. & Bau, D. (1997). Numerical Linear Algebra. SIAM
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM
"""

import torch
from torch import Tensor
from typing import Optional, Tuple, Dict, Any
import logging

from .base import (
    SolverBase,
    SolverMetadata,
    SolverResult,
    SolverFamily,
    SolverStatus
)
from .registry import register_solver
from ...parameters import get_global_manager

logger = logging.getLogger(__name__)


def _thomas_pivot(pivot: Tensor, row_scale: Tensor, row: int) -> Tensor:
    """Return the pivot to divide by, or raise for a numerically singular pivot."""
    # References:
    #   Higham, N.J. (2002). Accuracy and Stability of Numerical Algorithms,
    #   2nd ed. SIAM, Sect. 2.2 (rounding model) and Sect. 9.5 (LU
    #   factorization of tridiagonal matrices). Golub, G.H. & Van Loan, C.F.
    #   (2013). Matrix Computations, 4th ed., Sect. 4.3 (banded systems).
    #   Press, W.H. et al. (1992). Numerical Recipes in C, 2nd ed., Sect. 2.4
    #   (tridag has no pivoting and stops at a zero pivot; a system on which
    #   it fails is solved with the pivoting band solver bandec/banbks).
    # Mathematical expression:
    #   p_0 = b_0,  p_i = b_i - a_i * c'_{i-1}
    #   s_i = |a_i| [i >= 1] + |b_i| + |c_i| [i <= n-2]
    #   numerically singular  <=>  not finite(p_i)  or  |p_i| <= eps_m * s_i
    # Parameter meanings:
    #   p_i: pivot (diagonal of U in A = L U without pivoting); s_i: infinity
    #   norm of row i of A; eps_m: machine epsilon of the pivot dtype
    #   (torch.finfo(dtype).eps).
    # Units:
    #   p_i and s_i in the units of the matrix entries; eps_m dimensionless.
    # Domain/constraints:
    #   Floating dtype. pivot and row_scale are 0-d (one system) or 1-d over
    #   batch elements; the test is elementwise and invariant to scaling A. A
    #   flagged pivot always raises ValueError: no pivot is replaced by a
    #   floor and no solution is fabricated (the former sign(p_i) * eps
    #   substitution had no support in the cited literature).
    # Sign convention:
    #   None; only magnitudes are compared.
    threshold = torch.finfo(pivot.dtype).eps * row_scale
    singular = ~torch.isfinite(pivot) | (torch.abs(pivot) <= threshold)
    if bool(torch.any(singular)):
        if pivot.dim() == 0:
            detail = (
                f"pivot {float(pivot)!r} at row {row} is not above "
                f"machine_eps * row_scale = {float(threshold)!r}"
            )
        else:
            elements = torch.nonzero(singular).flatten().tolist()
            detail = (
                f"pivot at row {row} in batch element(s) {elements} is not "
                "above machine_eps * row_scale"
            )
        raise ValueError(
            f"Matrix is singular: {detail}. The Thomas algorithm does not pivot "
            "(Press et al. 1992, Numerical Recipes Sect. 2.4); solve this system "
            "with a banded LU solver with partial pivoting (e.g. LAPACK gbsv)"
        )
    return pivot


def solve_tridiagonal_thomas(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
) -> Tensor:
    """
    Solve tridiagonal system using Thomas algorithm.

    Solves Ax = d where A is tridiagonal with:
    - Lower diagonal: a[1:n] (a[0] is ignored)
    - Main diagonal: b[0:n]
    - Upper diagonal: c[0:n-1] (c[n-1] is ignored)
    - Right hand side: d[0:n]

    Mathematical Expression
    ----------------------
    Forward elimination:
        c'[0] = c[0] / b[0]
        d'[0] = d[0] / b[0]
        For i = 1..n-1:
            denom = b[i] - a[i] * c'[i-1]
            c'[i] = c[i] / denom
            d'[i] = (d[i] - a[i] * d'[i-1]) / denom

    Back substitution:
        x[n-1] = d'[n-1]
        For i = n-2..0:
            x[i] = d'[i] - c'[i] * x[i+1]

    Parameter Meanings
    ------------------
    a: lower diagonal coefficients [n], a[0] ignored
    b: main diagonal coefficients [n]
    c: upper diagonal coefficients [n], c[n-1] ignored
    d: right hand side vector [n]
    A pivot with |p_i| <= machine_eps(dtype) * row_scale_i raises ValueError
    (see _thomas_pivot); there is no pivoting and no pivot regularization.

    References
    ----------
    Thomas, L.H. (1949). Elliptic problems in linear difference equations.
    Press, W.H. et al. (2007). Numerical Recipes, Section 2.4.

    Parameters
    ----------
    a : Tensor
        Lower diagonal coefficients [n], a[0] ignored
    b : Tensor
        Main diagonal coefficients [n]
    c : Tensor
        Upper diagonal coefficients [n], c[n-1] ignored
    d : Tensor
        Right hand side vector [n]

    Returns
    -------
    Tensor
        Solution vector x [n]

    Raises
    ------
    ValueError
        If a pivot is numerically singular; the message recommends a banded
        LU solver with partial pivoting
    """
    n = b.numel()
    if a.numel() != n or c.numel() != n or d.numel() != n:
        raise RuntimeError("All input tensors must have the same size")

    if n == 0:
        return torch.empty(0, dtype=b.dtype, device=b.device)

    if n == 1:
        pivot = _thomas_pivot(b[0], torch.abs(b[0]), 0)
        # Shape (1,), consistent with the solution shape (n,) for every n.
        return d[0:1] / pivot

    # References: Thomas (1949); Press et al. (2007) Section 2.4; Paszke et al. (2019), reverse-mode autograd.
    # Mathematical expression: c'_i = c_i / (b_i - a_i c'_{i-1}), d'_i = (d_i - a_i d'_{i-1}) / (b_i - a_i c'_{i-1}),
    #   x_{n-1} = d'_{n-1}, x_i = d'_i - c'_i x_{i+1}.
    # Parameter meanings: a, b, c are the sub-, main and super-diagonals, d the right-hand side,
    #   c', d' the eliminated coefficients, x the solution.
    # Units: inherited from the caller's linear system (x carries units of d divided by b).
    # Domain/constraints: every pivot must pass the scaled test of _thomas_pivot, else
    #   ValueError; the sweeps build Python lists of 0-d tensors and stack once, so no tensor
    #   on the autograd graph is modified in place and d(x)/d(a, b, c, d) stays available to
    #   callers such as the Richards Picard correction.
    # Sign convention: none; the solver is sign-agnostic.
    b0 = _thomas_pivot(b[0], torch.abs(b[0]) + torch.abs(c[0]), 0)

    cp = [c[0] / b0]
    dp = [d[0] / b0]

    for i in range(1, n):
        row_scale = torch.abs(a[i]) + torch.abs(b[i])
        if i < n - 1:
            row_scale = row_scale + torch.abs(c[i])
        denom = _thomas_pivot(b[i] - a[i] * cp[i - 1], row_scale, i)
        if i < n - 1:
            cp.append(c[i] / denom)
        dp.append((d[i] - a[i] * dp[i - 1]) / denom)

    x = [dp[n - 1]]
    for i in range(n - 2, -1, -1):
        x.append(dp[i] - cp[i] * x[-1])
    x.reverse()
    return torch.stack(x)


def solve_tridiagonal_batch(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
) -> Tensor:
    """
    Solve batch of tridiagonal systems using vectorized Thomas algorithm.

    Mathematical Expression
    ----------------------
    Same as solve_tridiagonal_thomas, applied per batch element.

    Parameter Meanings
    ------------------
    a: lower diagonal coefficients [batch, n]
    b: main diagonal coefficients [batch, n]
    c: upper diagonal coefficients [batch, n]
    d: right hand side vectors [batch, n]
    A numerically singular pivot in any batch element raises ValueError (see
    _thomas_pivot); there is no pivoting and no pivot regularization.

    References
    ----------
    Thomas, L.H. (1949). Elliptic problems in linear difference equations.
    Press, W.H. et al. (2007). Numerical Recipes, Section 2.4.

    Parameters
    ----------
    a : Tensor
        Lower diagonal coefficients [batch_size, n]
    b : Tensor
        Main diagonal coefficients [batch_size, n]
    c : Tensor
        Upper diagonal coefficients [batch_size, n]
    d : Tensor
        Right hand side vectors [batch_size, n]

    Returns
    -------
    Tensor
        Solution vectors [batch_size, n]

    Raises
    ------
    ValueError
        If a pivot of any batch element is numerically singular; the message
        recommends a banded LU solver with partial pivoting
    """
    batch_size, n = b.shape
    device = b.device
    dtype = b.dtype

    if n == 0:
        return torch.empty(batch_size, 0, dtype=dtype, device=device)

    # Pivot test: see _thomas_pivot (elementwise over the batch, scaled by the
    # row infinity norm; a flagged pivot raises).
    if n == 1:
        pivot = _thomas_pivot(b[:, 0], torch.abs(b[:, 0]), 0)
        return d[:, 0:1] / pivot.unsqueeze(-1)

    cp = torch.zeros(batch_size, n - 1, dtype=dtype, device=device)
    dp = torch.zeros(batch_size, n, dtype=dtype, device=device)

    b0 = _thomas_pivot(b[:, 0], torch.abs(b[:, 0]) + torch.abs(c[:, 0]), 0)
    cp[:, 0] = c[:, 0] / b0
    dp[:, 0] = d[:, 0] / b0

    for i in range(1, n):
        row_scale = torch.abs(a[:, i]) + torch.abs(b[:, i])
        if i < n - 1:
            row_scale = row_scale + torch.abs(c[:, i])
        denom = _thomas_pivot(b[:, i] - a[:, i] * cp[:, i - 1], row_scale, i)
        if i < n - 1:
            cp[:, i] = c[:, i] / denom
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / denom

    x = torch.zeros(batch_size, n, dtype=dtype, device=device)
    x[:, n - 1] = dp[:, n - 1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]

    return x


def check_matrix_conditioning(
    a: Tensor,
    b: Tensor,
    c: Tensor
) -> Tuple[Tensor, bool]:
    """
    Check conditioning of tridiagonal matrix using diagonal dominance and
    Gershgorin bounds.

    Mathematical Expression
    ----------------------
    Diagonal dominance:
        |b[i]| >= |a[i]| + |c[i]|
    Condition estimate (Varah 1975 bound on cond_inf):
        r[i] = |a[i]| + |c[i]|
        estimate = max(|b[i]| + r[i]) / min(|b[i]| - r[i])
        (infinite when the dominance margin is zero to working precision;
        well conditioned when estimate < condition_number_limit from YAML
        and the matrix is diagonally dominant)

    Parameter Meanings
    ------------------
    a: lower diagonal coefficients [n], a[0] ignored
    b: main diagonal coefficients [n]
    c: upper diagonal coefficients [n], c[n-1] ignored

    References
    ----------
    Gershgorin, S. (1931). Uber die Abgrenzung der Eigenwerte einer Matrix.
    Varga, R.S. (1962). Matrix Iterative Analysis.
    """
    # References:
    #   Gershgorin, S. (1931). Izv. Akad. Nauk SSSR 6:749-754. Varah, J.M.
    #   (1975). A lower bound for the smallest singular value of a matrix.
    #   Linear Algebra Appl. 11:3-5. Higham, N.J. (2002). Accuracy and
    #   Stability of Numerical Algorithms, 2nd ed. SIAM, Sect. 2.2.
    # Mathematical expression:
    #   r_i = |a_i| [i >= 1] + |c_i| [i <= n-2]
    #   upper = max_i (|b_i| + r_i) = ||A||_inf
    #   lower = min_i (|b_i| - r_i)          (diagonal-dominance margin)
    #   lower <= eps_m * upper  ->  estimate = inf, not well conditioned
    #   otherwise  estimate = upper / lower >= cond_inf(A)  (Varah bound),
    #   well conditioned  <=>  estimate < L  and  |b_i| >= r_i for all i
    # Parameter meanings:
    #   a, b, c: sub-, main and super-diagonal; r_i: off-diagonal row sum;
    #   eps_m: machine epsilon of the matrix dtype (torch.finfo); L:
    #   computation.solvers.thomas.condition_number_limit (YAML).
    # Units:
    #   a, b, c, r_i, upper and lower in the units of the matrix entries;
    #   estimate, eps_m and L dimensionless.
    # Domain/constraints:
    #   Floating dtype. The bound needs strict diagonal dominance
    #   (lower > 0); a margin that is zero to working precision relative to
    #   ||A||_inf gives an infinite estimate. The estimate is an upper bound,
    #   invariant to the sign of the diagonal and to scaling A.
    # Sign convention:
    #   None; only magnitudes enter.
    n = b.numel()
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)
    abs_c = torch.abs(c)
    radius = torch.zeros_like(abs_b)
    if n > 1:
        radius = radius + torch.cat([abs_a.new_zeros(1), abs_a[1:]])
        radius = radius + torch.cat([abs_c[:-1], abs_c.new_zeros(1)])

    is_diagonally_dominant = bool(torch.all(abs_b >= radius))
    upper = torch.max(abs_b + radius)
    lower = torch.min(abs_b - radius)

    if bool(lower <= torch.finfo(b.dtype).eps * upper):
        return torch.full_like(upper, float("inf")), False

    condition_limit = get_global_manager().get(
        "computation.solvers.thomas.condition_number_limit", required=True
    )
    condition_number = upper / lower
    is_well_conditioned = bool(condition_number < condition_limit) and is_diagonally_dominant
    return condition_number, is_well_conditioned


@register_solver("thomas_algorithm", family=SolverFamily.LINEAR_DIRECT)
class ThomasAlgorithmSolver(SolverBase):
    """
    Thomas algorithm for tridiagonal linear systems.
    
    This solver implements the Thomas algorithm (also known as the Tridiagonal
    Matrix Algorithm or TDMA), which is a specialized direct solver for
    tridiagonal linear systems. It is O(n) in both time and space, making it
    highly efficient for 1D discretizations.
    
    Mathematical Expression:
        Solves Ax = d where A is tridiagonal:
        [b0 c0  0  0 ...]     [x0]     [d0]
        [a1 b1 c1  0 ...]     [x1]     [d1]
        [ 0 a2 b2 c2 ...]  *  [x2]  =  [d2]
        [ 0  0 a3 b3 ...]     [x3]     [d3]
        [... ... ... ...]     [...]    [...]
        
        Forward elimination:
            c'[0] = c[0] / b[0]
            d'[0] = d[0] / b[0]
            For i = 1 to n-1:
                denom = b[i] - a[i] * c'[i-1]
                c'[i] = c[i] / denom
                d'[i] = (d[i] - a[i] * d'[i-1]) / denom
        
        Back substitution:
            x[n-1] = d'[n-1]
            For i = n-2 down to 0:
                x[i] = d'[i] - c'[i] * x[i+1]
        
    Stability:
        - Stable for diagonally dominant matrices: |b[i]| >= |a[i]| + |c[i]|
        - No pivoting: a numerically singular pivot always raises ValueError
          (Press et al. 1992, Sect. 2.4); such a system needs a banded LU
          solver with partial pivoting

    Complexity:
        - Time: O(n) for n unknowns
        - Space: O(n) for working arrays
        - Optimal for tridiagonal systems

    Parameters (YAML namespace computation.solvers.thomas):
        check_stability : bool
            Compute the Varah bound on cond_inf(A) (check_matrix_conditioning)
            for every 1-D solve and record it in the result metadata
            (condition_number, is_well_conditioned); the pivot test does not
            depend on this flag
        report_condition : bool
            Add the condition estimate to the result metadata and log a
            warning when the matrix is not well conditioned
            
    Examples:
        >>> # Solve tridiagonal system from 1D diffusion
        >>> solver = SolverFactory.create(
        ...     "thomas_algorithm",
        ...     name="thomas_solver",
        ...     config={"check_stability": True}
        ... )
        >>> n = 100
        >>> a = torch.ones(n) * (-1.0)      # Lower diagonal
        >>> b = torch.ones(n) * 2.0         # Main diagonal
        >>> c = torch.ones(n) * (-1.0)      # Upper diagonal
        >>> d = torch.randn(n)              # Right-hand side
        >>> result = solver.solve(a, b, c, d)
        >>> x = result.solution  # Solution vector
        
    Applications:
        - 1D Richards equation (soil water flow)
        - 1D heat equation (soil temperature)
        - Vertical diffusion in atmospheric models
        - Boundary layer problems
        
    References:
        - Thomas, L.H. (1949). Elliptic problems in linear difference equations
          over a network. Watson Sci. Comput. Lab Report, Columbia University
        - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Chapter 11
        - Press, W.H. et al. (2007). Numerical Recipes, Section 2.4
    """
    
    solver_family = SolverFamily.LINEAR_DIRECT
    
    def __init__(
        self,
        name: str = "thomas_algorithm",
        operator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        check_stability: Optional[bool] = None,
        report_condition: Optional[bool] = None
    ):
        if config is None:
            config = {}
        if check_stability is not None:
            config["check_stability"] = check_stability
        if report_condition is not None:
            config["report_condition"] = report_condition
        # Create metadata if not provided
        if metadata is None:
            metadata = SolverMetadata(
                name="Thomas Algorithm",
                solver_family=SolverFamily.LINEAR_DIRECT,
                description="Direct O(n) solver for tridiagonal linear systems",
                mathematical_expression=(
                    "Forward elimination:\n"
                    "  c'[i] = c[i] / (b[i] - a[i]*c'[i-1])\n"
                    "  d'[i] = (d[i] - a[i]*d'[i-1]) / (b[i] - a[i]*c'[i-1])\n"
                    "Back substitution:\n"
                    "  x[n-1] = d'[n-1]\n"
                    "  x[i] = d'[i] - c'[i]*x[i+1]"
                ),
                order=1,  # Direct method, no approximation error
                stability_region="Stable for diagonally dominant matrices",
                is_implicit=False,  # Direct solver, not iterative
                requires_jacobian=False,
                references=[
                    "Thomas, L.H. (1949). Elliptic problems in linear difference equations",
                    "Burden, R.L. & Faires, J.D. (2010). Numerical Analysis. Brooks/Cole",
                    "Press, W.H. et al. (2007). Numerical Recipes. Cambridge University Press"
                ],
                parameters={
                    "check_stability": (
                        "Record the Varah condition estimate of every 1-D solve in "
                        "the result metadata (the pivot test is always on)"
                    ),
                    "report_condition": "Report the condition estimate in the result metadata"
                }
            )
        
        # The namespace must be known when SolverBase resolves
        # max_iterations/rtol/atol, so it is passed to SolverBase.__init__.
        super().__init__(
            name,
            operator,
            config,
            metadata,
            device,
            dtype,
            parameter_namespace="computation.solvers.thomas",
        )

        # Get solver-specific parameters
        self.check_stability = self.get_parameter("check_stability")
        self.report_condition = self.get_parameter("report_condition")
        
    def solve(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        **kwargs
    ) -> SolverResult:
        """
        Solve tridiagonal system Ax = d.
        
        Parameters:
            a : Tensor [n]
                Lower diagonal coefficients (a[0] is ignored)
            b : Tensor [n]
                Main diagonal coefficients
            c : Tensor [n]
                Upper diagonal coefficients (c[n-1] is ignored)
            d : Tensor [n]
                Right-hand side vector
            **kwargs : Additional arguments
                Accepted for interface compatibility and not used; 2-D inputs
                [batch, n] are solved row by row
                
        Returns:
            SolverResult
                - solution: Solution vector x [n] or [batch, n]
                - status: SUCCESS (a singular pivot raises instead)
                - iterations: 1 (direct solver)
                - metadata: May include condition number if requested
                
        Raises:
            RuntimeError: If input tensors have mismatched shapes
            ValueError: If a pivot is numerically singular
                (|p_i| <= machine_eps * row_scale_i), whatever check_stability
            Any other exception propagates unchanged.
            
        Notes:
            - Input tensors must have same length n
            - a[0] and c[n-1] are ignored (out of tridiagonal band)
            - For batch processing, use 2D tensors [batch_size, n]
        """
        # Validate input shapes (a solve changes no solver state, invariant I1)
        n = b.numel() if b.dim() == 1 else b.shape[-1]
        if a.shape != b.shape or c.shape != b.shape or d.shape != b.shape:
            raise RuntimeError(
                f"All input tensors must have same shape. "
                f"Got a={a.shape}, b={b.shape}, c={c.shape}, d={d.shape}"
            )
        
        # Check for empty system
        if n == 0:
            return SolverResult(
                solution=torch.empty(0, dtype=self.dtype, device=self.device),
                status=SolverStatus.SUCCESS,
                iterations=0
            )
        
        # Prepare metadata
        metadata = {
            "method": "thomas_algorithm",
            "system_size": n,
            "check_stability": self.check_stability
        }
        
        # Condition estimate if requested (1-D systems), recorded in the
        # result metadata; report_condition also logs a poorly conditioned
        # matrix. Errors propagate.
        if (self.check_stability or self.report_condition) and b.dim() == 1:
            condition_number, is_well_conditioned = check_matrix_conditioning(a, b, c)
            metadata["condition_number"] = float(condition_number.item())
            metadata["is_well_conditioned"] = is_well_conditioned
            if self.report_condition and not is_well_conditioned:
                logger.warning(
                    f"Matrix is poorly conditioned: cond(A) ~= {condition_number:.2e}"
                )

        # A numerically singular pivot raises ValueError from the kernel
        # (_thomas_pivot); every other exception propagates unchanged. The
        # callers use result.solution directly, so no SINGULAR result with a
        # fabricated solution is ever returned.
        if a.dim() == 2:
            x = torch.stack([
                solve_tridiagonal_thomas(a[i], b[i], c[i], d[i])
                for i in range(a.shape[0])
            ])
        else:
            x = solve_tridiagonal_thomas(a, b, c, d)

        return SolverResult(
            solution=x,
            status=SolverStatus.SUCCESS,
            iterations=1,  # Direct solver: one pass
            residual_norm=None,
            metadata=metadata
        )

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        **kwargs
    ) -> Tensor:
        """PyTorch module entry point returning the canonical solve result."""
        return self.solve(a, b, c, d, **kwargs).solution

    def get_diagnostics(self) -> dict:
        """Static solver description plus the check_stability setting."""
        diagnostics = super().get_diagnostics()
        diagnostics["check_stability"] = bool(self.check_stability)
        return diagnostics
    
    def check_diagonal_dominance(self, a: Tensor, b: Tensor, c: Tensor) -> bool:
        """
        Check if matrix is diagonally dominant (stability condition).
        
        Parameters:
            a : Tensor [n]
                Lower diagonal
            b : Tensor [n]
                Main diagonal
            c : Tensor [n]
                Upper diagonal
                
        Returns:
            bool
                True if matrix is strictly diagonally dominant
                
        Notes:
            A tridiagonal matrix is strictly diagonally dominant if:
            - Row 0: |b[0]| > |c[0]|
            - Row i (1 <= i < n-1): |b[i]| > |a[i]| + |c[i]|
            - Row n-1: |b[n-1]| > |a[n-1]|
        """
        n = b.numel()
        
        # Check first row
        if torch.abs(b[0]) <= torch.abs(c[0]):
            return False
        
        # Check middle rows
        for i in range(1, n - 1):
            if torch.abs(b[i]) <= torch.abs(a[i]) + torch.abs(c[i]):
                return False
        
        # Check last row
        if n > 1 and torch.abs(b[n-1]) <= torch.abs(a[n-1]):
            return False
        
        return True


# Export linear solvers
__all__ = [
    'solve_tridiagonal_thomas',
    'solve_tridiagonal_batch',
    'check_matrix_conditioning',
    'ThomasAlgorithmSolver'
]
