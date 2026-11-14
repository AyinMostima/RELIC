"""
Test Suite for Numerical Solvers

This module contains comprehensive tests for the numerical solvers in
relic.computation.solvers, including Thomas algorithm and
ODE solvers. Tests cover numerical accuracy, stability, edge cases,
and performance characteristics.

Test Categories:
    - L1 Unit Tests: Individual solver components
    - Convergence Tests: Numerical accuracy and order verification  
    - Stability Tests: Conditioning and robustness analysis
    - Performance Tests: Batch processing and scalability
    - Integration Tests: Solver interactions and consistency

Mathematical Verification:
    - Thomas algorithm: Known analytical solutions
    - ODE solvers: Manufactured solutions with known exact results
    - Error analysis: Convergence rates and stability regions
"""

import logging

import pytest
import torch
import numpy as np

# Import solvers to test
from relic.computation.solvers import (
    solve_tridiagonal_thomas,
    solve_tridiagonal_batch,
    ThomasAlgorithmSolver,
    check_matrix_conditioning,
    BackwardEulerSolver,
    CrankNicolsonSolver,
    ImplicitMidpointSolver,
    RK4Solver,
    solve_ode_system,
    SolverFactory,
    SolverStatus,
)


class TestThomasAlgorithm:
    """Test suite for Thomas algorithm implementation."""
    
    def test_simple_tridiagonal_system(self):
        """Test Thomas algorithm on simple 3x3 system with known solution."""
        # System: [2 1 0] [x0]   [1]
        #         [1 2 1] [x1] = [2]  
        #         [0 1 2] [x2]   [1]
        # Analytical solution: x = [0.2, 0.6, 0.2]
        
        a = torch.tensor([0.0, 1.0, 1.0])  # Lower diagonal
        b = torch.tensor([2.0, 2.0, 2.0])  # Main diagonal
        c = torch.tensor([1.0, 1.0, 0.0])  # Upper diagonal
        d = torch.tensor([1.0, 2.0, 1.0])  # RHS
        
        x = solve_tridiagonal_thomas(a, b, c, d)
        x_expected = torch.tensor([0.0, 1.0, 0.0])
        
        assert torch.allclose(x, x_expected, atol=1e-8)
    
    def test_diagonally_dominant_system(self):
        """Test on diagonally dominant system for stability."""
        n = 10
        
        # Create diagonally dominant system
        a = torch.ones(n) * -0.5
        b = torch.ones(n) * 3.0
        c = torch.ones(n) * -0.5
        d = torch.ones(n) * 2.0
        
        x = solve_tridiagonal_thomas(a, b, c, d)
        
        # Verify solution by substituting back
        residual = self._compute_tridiagonal_residual(a, b, c, x, d)
        assert torch.max(torch.abs(residual)) < 1e-6
    
    def test_ill_conditioned_system(self):
        """Test behavior with ill-conditioned matrix."""
        # Truly singular system (zero diagonal element)
        a = torch.tensor([0.0, 1.0, 1.0])
        b = torch.tensor([0.0, 2.0, 2.0])  # Zero diagonal element
        c = torch.tensor([1.0, 1.0, 0.0])
        d = torch.tensor([1.0, 2.0, 1.0])
        
        # A singular pivot always raises; the message points to a pivoting
        # band solver (Press et al. 1992, Sect. 2.4: tridag does not pivot).
        with pytest.raises(ValueError, match="Matrix is singular") as excinfo:
            solve_tridiagonal_thomas(a, b, c, d)
        assert "partial pivoting" in str(excinfo.value)
        with pytest.raises(ValueError, match="partial pivoting"):
            solve_tridiagonal_batch(a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), d.unsqueeze(0))

    @pytest.mark.parametrize("option", [{"check_stability": False}, {"check_stability": True}, {"eps": 1.0e-12}])
    @pytest.mark.parametrize("kernel", ["single", "batch"])
    def test_pivot_regularization_options_are_removed(self, option, kernel):
        """The kernels have no switch that replaces a small pivot by sign(p) * eps."""
        a = torch.tensor([0.0, 1.0, 1.0])
        b = torch.tensor([2.0, 2.0, 2.0])
        c = torch.tensor([1.0, 1.0, 0.0])
        d = torch.tensor([1.0, 2.0, 1.0])
        with pytest.raises(TypeError):
            if kernel == "single":
                solve_tridiagonal_thomas(a, b, c, d, **option)
            else:
                solve_tridiagonal_batch(
                    a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), d.unsqueeze(0), **option
                )

    def test_thomas_solver_has_no_regularization_setting(self):
        from pathlib import Path

        import yaml

        import relic

        with pytest.raises(TypeError):
            ThomasAlgorithmSolver(regularization=1.0e-12)
        with pytest.raises(ValueError, match="accepted keys"):
            ThomasAlgorithmSolver(config={"regularization": 1.0e-12})
        database = (
            Path(relic.__file__).parent / "parameters" / "database" / "relic"
            / "computation" / "solvers.yaml"
        )
        entries = yaml.safe_load(database.read_text(encoding="utf-8"))["parameters"]
        assert "computation.solvers.thomas.regularization" not in entries
        solver = ThomasAlgorithmSolver(dtype=torch.float64)
        assert "regularization" not in solver.get_diagnostics()
        assert "regularization" not in solver.metadata.parameters

    @pytest.mark.parametrize("check_stability", [True, False])
    def test_thomas_solver_raises_for_a_singular_pivot_whatever_check_stability(self, check_stability):
        """check_stability only toggles the conditioning estimate; the pivot test is always on."""
        solver = ThomasAlgorithmSolver(check_stability=check_stability, dtype=torch.float64)
        a = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
        b = torch.tensor([0.0, 2.0, 2.0], dtype=torch.float64)
        c = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="partial pivoting"):
            solver.solve(a, b, c, d)
        regular = ThomasAlgorithmSolver(
            check_stability=check_stability, report_condition=False, dtype=torch.float64
        )
        result = regular.solve(a, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64), c, d)
        assert ("condition_number" in result.metadata) is check_stability
    
    def test_batch_processing(self):
        """Test batch processing of multiple tridiagonal systems."""
        batch_size, n = 5, 8
        
        # Create batch of diagonally dominant systems
        a = torch.randn(batch_size, n) * 0.5
        b = torch.randn(batch_size, n) + 3.0  # Ensure diagonal dominance
        c = torch.randn(batch_size, n) * 0.5
        d = torch.randn(batch_size, n)
        
        x_batch = solve_tridiagonal_batch(a, b, c, d)
        assert x_batch.shape == (batch_size, n)
        
        # Verify each system individually
        for i in range(batch_size):
            x_single = solve_tridiagonal_thomas(a[i], b[i], c[i], d[i])
            assert torch.allclose(x_batch[i], x_single, atol=1e-8)
    
    def test_thomas_solution_is_differentiable_with_respect_to_all_coefficients(self):
        """The solver must expose d x / d (a, b, c, d) to autograd without in-place graph mutation."""
        a = torch.tensor([0.0, -1.0, -1.0, -1.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([-1.0, -1.0, -1.0, 0.0], dtype=torch.float64, requires_grad=True)
        d = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
        x = solve_tridiagonal_thomas(a, b, c, d)
        # References: Press et al. (2007) Section 2.4.
        # Mathematical expression: A x = d with A tridiagonal(a, b, c); dx/dd = A^-1.
        # Parameter meanings: a sub-diagonal, b diagonal, c super-diagonal, d right-hand side.
        # Units: dimensionless test system.
        # Domain/constraints: strictly diagonally dominant, so the pivots are safe.
        # Sign convention: none.
        matrix = torch.diag(b) + torch.diag(a[1:], -1) + torch.diag(c[:-1], 1)
        reference = torch.linalg.solve(matrix, d)
        torch.testing.assert_close(x, reference)
        gradients = torch.autograd.grad(x.sum(), (a, b, c, d), allow_unused=False)
        reference_gradients = torch.autograd.grad(reference.sum(), (a, b, c, d), allow_unused=True)
        for got, expected in zip(gradients, reference_gradients):
            assert got is not None and torch.all(torch.isfinite(got))
            torch.testing.assert_close(got, expected)
        assert torch.autograd.gradcheck(
            lambda aa, bb, cc, dd: solve_tridiagonal_thomas(aa, bb, cc, dd),
            (a.detach().requires_grad_(True), b.detach().requires_grad_(True),
             c.detach().requires_grad_(True), d.detach().requires_grad_(True)),
        )

    def test_thomas_solver_module(self):
        """Test ThomasAlgorithmSolver PyTorch module."""
        solver = ThomasAlgorithmSolver(check_stability=True)
        
        # Test forward pass
        a = torch.tensor([0.0, 1.0, 1.0])
        b = torch.tensor([2.0, 2.0, 2.0])
        c = torch.tensor([1.0, 1.0, 0.0])
        d = torch.tensor([1.0, 2.0, 1.0])
        
        x = solver(a, b, c, d)
        x_expected = torch.tensor([0.0, 1.0, 0.0])

        assert torch.allclose(x, x_expected, atol=1e-8)

        # The conditioning estimate is part of the solve result, not solver state
        result = solver.solve(a, b, c, d)
        assert 'condition_number' in result.metadata
        assert 'is_well_conditioned' in result.metadata
    
    def test_thomas_resolves_base_parameters_in_its_own_namespace(self, monkeypatch):
        """max_iterations/rtol/atol come from computation.solvers.thomas.*.

        SolverBase resolves these keys during __init__; the namespace must be
        known at that moment, not assigned after super().__init__().
        """
        import relic.computation.solvers.base as base_module

        real_get_global_manager = base_module.get_global_manager
        requested_keys = []

        class _RecordingManager:
            def __init__(self, inner):
                self._inner = inner

            def get(self, key, *args, **kwargs):
                requested_keys.append(key)
                return self._inner.get(key, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        monkeypatch.setattr(
            base_module,
            "get_global_manager",
            lambda *args, **kwargs: _RecordingManager(real_get_global_manager(*args, **kwargs)),
        )
        solver = ThomasAlgorithmSolver()
        manager = real_get_global_manager(preload_all=True, warmup_cache=False)

        assert solver.parameter_namespace == "computation.solvers.thomas"
        for key in ("max_iterations", "rtol", "atol"):
            full_key = f"computation.solvers.thomas.{key}"
            assert full_key in requested_keys
            assert getattr(solver, key) == manager.get(full_key, required=True)
        # A direct solver performs one elimination and back-substitution pass.
        assert solver.max_iterations == 1

    def test_thomas_solver_raises_on_a_numerically_singular_pivot(self):
        """A zero pivot relative to its row raises; no fabricated zero solution is returned.

        Criterion: |p_i| <= eps(dtype) * (|a_i| + |b_i| + |c_i|).
        """
        solver = ThomasAlgorithmSolver(check_stability=True, dtype=torch.float64)
        # [[1, 1], [1, 1]] is exactly singular: p_1 = 1 - 1 * 1 = 0.
        a = torch.tensor([0.0, 1.0], dtype=torch.float64)
        b = torch.tensor([1.0, 1.0], dtype=torch.float64)
        c = torch.tensor([1.0, 0.0], dtype=torch.float64)
        d = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="Matrix is singular.*row 1"):
            solver.solve(a, b, c, d)

        # Row scale 2e6 with pivot one ulp(1e6) ~ 1.2e-10 <= eps * 2e6 ~ 4.4e-10:
        # numerically singular although above the old absolute 1e-12 floor.
        b_near = torch.tensor(
            [1.0e6, float(torch.nextafter(torch.tensor(1.0e6, dtype=torch.float64),
                                          torch.tensor(2.0e6, dtype=torch.float64)))],
            dtype=torch.float64,
        )
        big = torch.tensor([0.0, 1.0e6], dtype=torch.float64)
        with pytest.raises(ValueError, match="Matrix is singular"):
            solver.solve(big, b_near, torch.tensor([1.0e6, 0.0], dtype=torch.float64), d)

    def test_thomas_solver_pivot_test_is_scale_invariant(self):
        """A well-conditioned system scaled by 1e-14 is solved, not flagged singular."""
        solver = ThomasAlgorithmSolver(check_stability=True, dtype=torch.float64)
        a = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float64)
        b = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64)
        c = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float64)
        d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        scale = 1.0e-14
        reference = solver.solve(a, b, c, d)
        scaled = solver.solve(a * scale, b * scale, c * scale, d)
        assert reference.status is SolverStatus.SUCCESS
        assert scaled.status is SolverStatus.SUCCESS
        torch.testing.assert_close(scaled.solution * scale, reference.solution)

    def test_thomas_solver_propagates_genuine_errors(self, monkeypatch):
        """Errors that are not a detected singular pivot propagate unchanged."""
        import relic.computation.solvers.linear as linear_module

        def failing_kernel(*args, **kwargs):
            raise RuntimeError("kernel failure")

        monkeypatch.setattr(linear_module, "solve_tridiagonal_thomas", failing_kernel)
        solver = ThomasAlgorithmSolver(check_stability=True, dtype=torch.float64)
        ones = torch.ones(3, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="kernel failure"):
            solver.solve(ones * 0.0, ones * 4.0, ones * 0.0, ones)

    def test_thomas_has_no_batch_processing_switch(self):
        """The unused batch_processing switch is removed; 2-D input is solved row by row."""
        with pytest.raises(TypeError):
            ThomasAlgorithmSolver(batch_processing=True)
        with pytest.raises(ValueError, match="batch_processing"):
            ThomasAlgorithmSolver(config={"batch_processing": True})

        solver = ThomasAlgorithmSolver(check_stability=True)
        assert "batch_processing" not in solver.get_diagnostics()
        a = torch.tensor([[0.0, -1.0, -1.0], [0.0, -0.5, -0.5]], dtype=torch.float64)
        b = torch.tensor([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0]], dtype=torch.float64)
        c = torch.tensor([[-1.0, -1.0, 0.0], [-0.5, -0.5, 0.0]], dtype=torch.float64)
        d = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
        result = solver.solve(a, b, c, d)
        rows = torch.stack([solve_tridiagonal_thomas(a[i], b[i], c[i], d[i]) for i in range(2)])
        assert result.status is SolverStatus.SUCCESS
        assert torch.equal(result.solution, rows)

    def test_conditioning_check(self):
        """Test matrix conditioning analysis."""
        # Well-conditioned diagonally dominant matrix
        a = torch.tensor([0.0, 1.0, 1.0])
        b = torch.tensor([3.0, 3.0, 3.0])
        c = torch.tensor([1.0, 1.0, 0.0])
        
        cond_num, is_well_conditioned = check_matrix_conditioning(a, b, c)
        assert is_well_conditioned
        assert cond_num < 100  # Reasonable condition number
        
        # Ill-conditioned matrix
        a_ill = torch.tensor([0.0, 1.0, 1.0])
        b_ill = torch.tensor([1.001, 2.0, 2.0])  # Nearly singular
        c_ill = torch.tensor([1.0, 1.0, 0.0])
        
        cond_num_ill, is_well_conditioned_ill = check_matrix_conditioning(a_ill, b_ill, c_ill)
        assert cond_num_ill > 10  # Large condition number
    
    def test_batch_kernel_uses_the_scaled_sign_preserving_pivot_test(self):
        """solve_tridiagonal_batch shares the scaled pivot test of _thomas_pivot."""
        f64 = torch.float64
        # Negative-diagonal, diagonally dominant systems match the per-row kernel.
        a = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.5, 0.5]], dtype=f64)
        b = torch.tensor([[-4.0, -4.0, -4.0], [-3.0, -3.0, -3.0]], dtype=f64)
        c = torch.tensor([[1.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=f64)
        d = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=f64)
        rows = torch.stack([solve_tridiagonal_thomas(a[i], b[i], c[i], d[i]) for i in range(2)])
        torch.testing.assert_close(solve_tridiagonal_batch(a, b, c, d), rows)

        # n == 1 with a negative diagonal: x = d / b, sign preserved.
        single = solve_tridiagonal_batch(
            torch.zeros(2, 1, dtype=f64),
            torch.tensor([[-2.0], [4.0]], dtype=f64),
            torch.zeros(2, 1, dtype=f64),
            torch.ones(2, 1, dtype=f64),
        )
        torch.testing.assert_close(single, torch.tensor([[-0.5], [0.25]], dtype=f64))

        # Scale invariance: a system scaled by 1e-14 is solved, not flagged.
        scaled = solve_tridiagonal_batch(a * 1.0e-14, b * 1.0e-14, c * 1.0e-14, d)
        torch.testing.assert_close(scaled * 1.0e-14, rows)

        # Numerically singular pivot in batch element 1 raises, naming row and element.
        a_s = torch.tensor([[0.0, -1.0], [0.0, 1.0]], dtype=f64)
        b_s = torch.tensor([[4.0, 4.0], [1.0, 1.0]], dtype=f64)
        c_s = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=f64)
        d_s = torch.ones(2, 2, dtype=f64)
        with pytest.raises(ValueError, match=r"Matrix is singular.*row 1.*batch element\(s\) \[1\]"):
            solve_tridiagonal_batch(a_s, b_s, c_s, d_s)

    def test_single_equation_systems_return_shape_one(self):
        """n == 1 returns a tensor of shape (1,) from the kernel and the solver."""
        f64 = torch.float64
        zero = torch.zeros(1, dtype=f64)
        x = solve_tridiagonal_thomas(zero, torch.tensor([-2.0], dtype=f64), zero,
                                     torch.tensor([1.0], dtype=f64))
        assert x.shape == (1,)
        torch.testing.assert_close(x, torch.tensor([-0.5], dtype=f64))
        solver = ThomasAlgorithmSolver(check_stability=True, dtype=f64)
        result = solver.solve(zero, torch.tensor([4.0], dtype=f64), zero,
                              torch.tensor([1.0], dtype=f64))
        assert result.solution.shape == (1,)
        torch.testing.assert_close(result.solution, torch.tensor([0.25], dtype=f64))

    def test_conditioning_estimate_is_the_varah_bound_for_any_diagonal_sign(self):
        """cond_inf(A) <= max_i(|b_i| + r_i) / min_i(|b_i| - r_i), sign and scale invariant."""
        ones = torch.ones(3, dtype=torch.float64)
        a = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
        c = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)

        positive, positive_ok = check_matrix_conditioning(a, 3.0 * ones, c)
        negative, negative_ok = check_matrix_conditioning(a, -3.0 * ones, c)
        tiny, tiny_ok = check_matrix_conditioning(a * 1.0e-14, 3.0e-14 * ones, c * 1.0e-14)

        assert positive.dtype is torch.float64
        # max(|b|+r) = 3 + 2 = 5, min(|b|-r) = 3 - 2 = 1.
        assert float(positive) == 5.0
        assert float(negative) == 5.0
        assert float(tiny) == pytest.approx(5.0, rel=1.0e-12)
        assert positive_ok is True and negative_ok is True and tiny_ok is True

    def test_condition_number_limit_is_a_yaml_parameter(self):
        """The well-conditioned limit is read from computation.solvers.thomas.condition_number_limit."""
        from relic.parameters import get_global_manager

        limit = get_global_manager().get(
            "computation.solvers.thomas.condition_number_limit", required=True
        )
        a = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
        c = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        # Interior row: |b| - r = delta, |b| + r = 4 + delta, so the bound is
        # ~4 / delta = factor * limit.
        for factor, expected in ((0.5, True), (8.0, False)):
            delta = 4.0 / (factor * float(limit))
            b = torch.full((3,), 2.0 + delta, dtype=torch.float64)
            estimate, well_conditioned = check_matrix_conditioning(a, b, c)
            assert well_conditioned is expected
            assert (float(estimate) < float(limit)) is expected

    def test_convergence_order(self):
        """Test convergence order for discretized PDE."""
        # Test on discretized 1D diffusion equation: d虏u/dx虏 = f(x)
        # with u(0) = u(1) = 0, f(x) = 蟺虏sin(蟺x)
        # Analytical solution: u(x) = sin(蟺x)
        
        errors = []
        grid_sizes = [16, 32, 64, 128]
        
        for n in grid_sizes:
            h = 1.0 / (n + 1)  # Grid spacing
            x = torch.linspace(h, 1-h, n)
            
            # Discretized system: -u[i-1] + 2u[i] - u[i+1] = h虏f[i]
            a = torch.ones(n) * (-1)
            b = torch.ones(n) * 2
            c = torch.ones(n) * (-1)
            d = h**2 * (np.pi**2) * torch.sin(np.pi * x)
            
            # Boundary conditions (u(0) = u(1) = 0 already incorporated)
            u_numerical = solve_tridiagonal_thomas(a, b, c, d)
            u_analytical = torch.sin(np.pi * x)
            
            error = torch.max(torch.abs(u_numerical - u_analytical))
            errors.append(error.item())
        
        # Check second-order convergence (error ~ h虏)
        for i in range(1, len(errors)):
            convergence_rate = np.log(errors[i-1] / errors[i]) / np.log(2.0)
            assert 1.8 < convergence_rate < 2.2  # Should be close to 2
    
    def _compute_tridiagonal_residual(self, a: torch.Tensor, b: torch.Tensor, 
                                    c: torch.Tensor, x: torch.Tensor, 
                                    d: torch.Tensor) -> torch.Tensor:
        """Compute residual Ax - d for verification."""
        n = x.numel()
        Ax = torch.zeros_like(x)
        
        # First row
        Ax[0] = b[0] * x[0] + c[0] * x[1]
        
        # Middle rows
        for i in range(1, n-1):
            Ax[i] = a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1]
        
        # Last row
        Ax[n-1] = a[n-1] * x[n-2] + b[n-1] * x[n-1]
        
        return Ax - d


class TestSolverBaseConvergence:
    """Test the canonical successive-iterate convergence predicate."""

    @staticmethod
    def _solver(atol=1e-8, rtol=1e-6):
        # These tests exercise the SolverBase.check_convergence arithmetic with
        # thresholds outside the solver YAML bounds, which a configured value
        # may not take; the thresholds are set on a default-constructed solver.
        solver = BackwardEulerSolver()
        solver.atol = atol
        solver.rtol = rtol
        return solver

    @pytest.mark.parametrize(
        ("residual", "reference", "message"),
        [
            (torch.empty(0), torch.ones(1), "residual must contain"),
            (torch.ones(1), torch.empty(0), "reference must contain"),
            (
                torch.tensor([float("nan")]),
                torch.ones(1),
                "residual must contain only finite",
            ),
            (
                torch.tensor([float("inf")]),
                torch.ones(1),
                "residual must contain only finite",
            ),
            (
                torch.ones(1),
                torch.tensor([float("nan")]),
                "reference must contain only finite",
            ),
            (
                torch.ones(1),
                torch.tensor([float("inf")]),
                "reference must contain only finite",
            ),
        ],
    )
    def test_rejects_empty_or_nonfinite_inputs(
        self,
        residual,
        reference,
        message,
    ):
        solver = self._solver()

        with pytest.raises(ValueError, match=message):
            solver.check_convergence(residual, reference)

    def test_rejects_non_tensor_and_non_real_floating_inputs(self):
        solver = self._solver()

        with pytest.raises(TypeError, match="residual must be a torch.Tensor"):
            solver.check_convergence(1.0, torch.ones(1))
        with pytest.raises(TypeError, match="reference must be a torch.Tensor"):
            solver.check_convergence(torch.ones(1), 1.0)
        with pytest.raises(ValueError, match="real floating strided tensor"):
            solver.check_convergence(torch.ones(1, dtype=torch.int64))
        with pytest.raises(ValueError, match="real floating strided tensor"):
            solver.check_convergence(torch.ones(1, dtype=torch.complex64))

    @pytest.mark.parametrize(
        ("residual", "reference", "message"),
        [
            (torch.ones(2), torch.ones(1, 2), "same shape"),
            (
                torch.ones(2, dtype=torch.float32),
                torch.ones(2, dtype=torch.float64),
                "same dtype",
            ),
            (
                torch.ones(2),
                torch.empty(2, device="meta"),
                "same device",
            ),
        ],
    )
    def test_rejects_incompatible_reference_contracts(
        self,
        residual,
        reference,
        message,
    ):
        solver = self._solver()

        with pytest.raises(ValueError, match=message):
            solver.check_convergence(residual, reference)

    @pytest.mark.parametrize(
        ("attribute", "value", "message"),
        [
            ("atol", 0.0, "atol must be finite and greater than zero"),
            ("atol", -1.0, "atol must be finite and greater than zero"),
            ("atol", float("nan"), "atol must be finite and greater than zero"),
            ("atol", float("inf"), "atol must be finite and greater than zero"),
            ("rtol", -1.0, "rtol must be finite and nonnegative"),
            ("rtol", float("nan"), "rtol must be finite and nonnegative"),
            ("rtol", float("inf"), "rtol must be finite and nonnegative"),
        ],
    )
    def test_rejects_invalid_tolerance_domain(self, attribute, value, message):
        solver = self._solver()
        setattr(solver, attribute, value)

        with pytest.raises(ValueError, match=message):
            solver.check_convergence(torch.zeros(1), torch.ones(1))

    def test_zero_reference_has_no_artificial_relative_scale(self):
        solver = self._solver(atol=1e-20, rtol=1e-3)
        residual = torch.tensor([5e-16], dtype=torch.float64)
        reference = torch.zeros(1, dtype=torch.float64)

        assert solver.check_convergence(residual, reference) is False

    def test_strict_absolute_and_relative_threshold_boundaries(self):
        absolute_solver = self._solver(atol=1.0, rtol=0.0)
        assert absolute_solver.check_convergence(
            torch.tensor(1.0, dtype=torch.float64),
        ) is False

        relative_solver = self._solver(atol=0.25, rtol=0.125)
        reference = torch.tensor(8.0, dtype=torch.float64)
        exact = torch.tensor(1.0, dtype=torch.float64)
        below = torch.nextafter(exact, torch.tensor(0.0, dtype=torch.float64))
        assert relative_solver.check_convergence(exact, reference) is False
        assert relative_solver.check_convergence(below, reference) is True

    def test_reference_none_uses_only_absolute_threshold(self):
        solver = self._solver(atol=0.5, rtol=1e6)

        assert solver.check_convergence(torch.tensor(0.25)) is True
        assert solver.check_convergence(torch.tensor(0.5)) is False

    def test_rejects_finite_operands_whose_relative_threshold_overflows(self):
        solver = self._solver(atol=1.0, rtol=1e308)
        residual = torch.tensor([1.0], dtype=torch.float64)
        reference = torch.tensor([1e308], dtype=torch.float64)

        with pytest.raises(ValueError, match="relative threshold must be finite"):
            solver.check_convergence(residual, reference)

    def test_temporal_solver_rejects_nonfinite_iterate_increment(self):
        solver = self._solver()

        def nonfinite_rhs(y, time):
            _ = time
            return torch.full_like(y, float("nan"))

        with pytest.raises(ValueError, match="residual must contain only finite"):
            solver.solve(
                nonfinite_rhs,
                torch.ones(1),
                t_current=0.0,
                dt=0.1,
            )


class TestSolverBaseConfigKeys:
    """SolverBase rejects configuration keys that the solver does not declare."""

    _BASE_KEYS = ("max_iterations", "rtol", "atol")

    @pytest.mark.parametrize(
        ("solver_name", "stale_key"),
        [
            ("backward_euler", "relaxation_factor"),
            ("implicit_midpoint", "theta"),
            ("crank_nicolson", "thetta"),
            ("imex_euler", "substeps"),
            ("rk4", "adaptive"),
            ("thomas_algorithm", "check_diagonal_dominance"),
            ("thomas_algorithm", "regularization"),
            ("bracketed_bisection", "tolerance"),
        ],
    )
    def test_unknown_config_key_is_rejected_with_accepted_keys(self, solver_name, stale_key):
        with pytest.raises(ValueError) as excinfo:
            SolverFactory.create(
                solver_name=solver_name,
                name=f"{solver_name}_stale_key",
                config={stale_key: 1.0},
                dtype=torch.float64,
            )
        message = str(excinfo.value)
        assert repr(stale_key) in message
        assert "accepted keys" in message
        for key in self._BASE_KEYS:
            assert key in message

    @pytest.mark.parametrize(
        ("solver_name", "config"),
        [
            ("backward_euler", {"damping_factor": 1.0}),
            ("implicit_midpoint", {}),
            ("crank_nicolson", {"theta": 0.5}),
            ("imex_euler", {"explicit_substeps": 2, "damping_factor": 1.0}),
            ("rk4", {}),
            (
                "thomas_algorithm",
                {
                    "check_stability": True,
                    "report_condition": False,
                    # computation.solvers.thomas.max_iterations is bounded to [1, 1].
                    "max_iterations": 1,
                },
            ),
            ("bracketed_bisection", {}),
        ],
    )
    def test_declared_config_keys_are_accepted(self, solver_name, config):
        settings = {"max_iterations": 40, "rtol": 1.0e-8, "atol": 1.0e-10, **config}
        solver = SolverFactory.create(
            solver_name=solver_name,
            name=f"{solver_name}_declared_keys",
            config=settings,
            dtype=torch.float64,
        )
        declared = set(self._BASE_KEYS) | set(solver.metadata.parameters)
        assert set(settings) <= declared

    @pytest.mark.parametrize(
        ("solver_name", "namespace"),
        [
            ("backward_euler", "computation.solvers.backward_euler"),
            ("crank_nicolson", "computation.solvers.crank_nicolson"),
            ("imex_euler", "computation.solvers.imex_euler"),
            ("thomas_algorithm", "computation.solvers.thomas"),
        ],
    )
    def test_declared_solver_parameters_exist_in_the_yaml_database(self, solver_name, namespace):
        """Every declared solver-specific key is a YAML parameter of its namespace."""
        from relic.parameters import get_global_manager

        manager = get_global_manager(preload_all=True, warmup_cache=False)
        solver = SolverFactory.create(
            solver_name=solver_name, name=f"{solver_name}_yaml_keys", dtype=torch.float64
        )
        solver_specific = set(solver.metadata.parameters) - set(self._BASE_KEYS)
        assert solver_specific
        for key in sorted(solver_specific):
            assert manager.get(f"{namespace}.{key}", required=True) is not None



class TestSolverParameterResolution:
    """max_iterations/rtol/atol come from YAML; lookup errors are never swallowed."""

    _BASE_KEYS = ("max_iterations", "rtol", "atol")

    def test_factory_without_namespace_uses_the_solver_namespace(self):
        from relic.parameters import get_global_manager

        manager = get_global_manager(preload_all=True, warmup_cache=False)
        thomas = SolverFactory.create("thomas_algorithm", "thomas_resolution")
        for key in self._BASE_KEYS:
            assert getattr(thomas, key) == manager.get(
                f"computation.solvers.thomas.{key}", required=True
            )
        via_factory = SolverFactory.create(
            "implicit_midpoint", "midpoint_resolution", dtype=torch.float64
        )
        direct = ImplicitMidpointSolver(dtype=torch.float64)
        for key in self._BASE_KEYS:
            assert getattr(via_factory, key) == getattr(direct, key)

    @pytest.mark.parametrize(
        "solver_name",
        [
            "backward_euler",
            "bracketed_bisection",
            "crank_nicolson",
            "imex_euler",
            "implicit_midpoint",
            "rk4",
            "thomas_algorithm",
        ],
    )
    def test_solver_created_without_config_reads_its_own_namespace(self, solver_name):
        """Without configuration a solver takes max_iterations/rtol/atol from its namespace.

        The shared relic.physical.constants.numerical defaults are not a
        silent substitute for a registered solver's own YAML entries.
        """
        from relic.computation.solvers.registry import SolverRegistry
        from relic.parameters import get_global_manager

        assert solver_name in SolverRegistry.list_solvers()
        manager = get_global_manager(preload_all=True, warmup_cache=False)
        solver = SolverFactory.create(
            solver_name, f"{solver_name}_no_config", dtype=torch.float64
        )
        for key in self._BASE_KEYS:
            assert getattr(solver, key) == manager.get(
                f"{solver.parameter_namespace}.{key}", required=True
            )

    def test_every_registered_solver_is_covered_by_the_namespace_test(self):
        from relic.computation.solvers.registry import SolverRegistry

        assert set(SolverRegistry.list_solvers()) == {
            "backward_euler",
            "bracketed_bisection",
            "crank_nicolson",
            "imex_euler",
            "implicit_midpoint",
            "rk4",
            "thomas_algorithm",
        }

    def test_implicit_midpoint_parameters_carry_full_metadata(self):
        from pathlib import Path

        import yaml

        import relic

        database = (
            Path(relic.__file__).parent / "parameters" / "database" / "relic"
            / "computation" / "solvers.yaml"
        )
        entries = yaml.safe_load(database.read_text(encoding="utf-8"))["parameters"]
        for key in self._BASE_KEYS:
            entry = entries[f"computation.solvers.implicit_midpoint.{key}"]
            for field_name in (
                "value", "units", "dtype", "bounds", "category", "description",
                "references", "tags", "requires_grad", "provenance",
            ):
                assert field_name in entry, (key, field_name)

    def test_factory_explicit_namespace_reads_required_parameters(self):
        from relic.parameters import get_global_manager

        manager = get_global_manager(preload_all=True, warmup_cache=False)
        enriched = SolverFactory._validate_and_enrich_config(
            "backward_euler", {}, "computation.solvers.backward_euler"
        )
        for key in self._BASE_KEYS:
            assert enriched[key] == manager.get(
                f"computation.solvers.backward_euler.{key}", required=True
            )
        with pytest.raises(KeyError, match="computation.solvers.no_such_solver.max_iterations"):
            SolverFactory.create(
                "backward_euler", "missing_namespace", namespace="computation.solvers.no_such_solver"
            )

    def test_create_from_config_raises_on_validation_failure(self):
        """An invalid mapping raises; there is no unvalidated raw-dict fallback."""
        with pytest.raises(ValueError, match="name"):
            SolverFactory.create_from_config({"solver_name": "rk4", "name": 123})
        with pytest.raises(TypeError, match="SolverConfig or a mapping"):
            SolverFactory.create_from_config(42)
        solver = SolverFactory.create_from_config(
            {"solver_name": "rk4", "name": "validated_rk4", "dtype": "float64"}
        )
        assert solver.name == "validated_rk4"
        assert solver.dtype is torch.float64

    def test_create_from_config_raises_when_the_operator_cannot_be_built(self):
        """A requested operator that cannot be built is an error, not a silent None."""
        with pytest.raises(KeyError, match="laplacian.finite_difference"):
            SolverFactory.create_from_config(
                {
                    "solver_name": "crank_nicolson",
                    "name": "cn_unregistered_operator",
                    "operator_name": "laplacian.finite_difference",
                }
            )
        solver = SolverFactory.create_from_config(
            {
                "solver_name": "crank_nicolson",
                "name": "cn_with_laplacian",
                "operator_name": "laplacian.finite_difference_1d",
                "dtype": "float64",
            }
        )
        assert solver.operator is not None
        assert solver.operator.name == "cn_with_laplacian_operator"

    def test_factory_propagates_parameter_manager_errors(self, monkeypatch):
        import relic.computation.solvers.factory as factory_module

        def unavailable_manager(*args, **kwargs):
            raise RuntimeError("parameter manager unavailable")

        monkeypatch.setattr(factory_module, "get_global_manager", unavailable_manager)
        with pytest.raises(RuntimeError, match="parameter manager unavailable"):
            SolverFactory.create("rk4", "rk4_manager_failure", namespace="computation.solvers.rk4")

    def test_solver_base_propagates_namespace_lookup_errors(self, monkeypatch):
        import relic.computation.solvers.base as base_module

        real_get_global_manager = base_module.get_global_manager

        class _InvalidNamespaceValueManager:
            def __init__(self, inner):
                self._inner = inner

            def get(self, key, *args, **kwargs):
                if key == "computation.solvers.backward_euler.rtol":
                    raise ValueError("rtol out of bounds")
                return self._inner.get(key, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        monkeypatch.setattr(
            base_module,
            "get_global_manager",
            lambda *args, **kwargs: _InvalidNamespaceValueManager(
                real_get_global_manager(*args, **kwargs)
            ),
        )
        with pytest.raises(ValueError, match="rtol out of bounds"):
            BackwardEulerSolver()

    def test_config_values_are_checked_against_the_yaml_bounds(self):
        """A configured value outside the YAML bounds is rejected, as a manager value is."""
        with pytest.raises(ValueError, match="max_iterations"):
            BackwardEulerSolver(config={"max_iterations": 0})
        with pytest.raises(ValueError, match="damping_factor"):
            BackwardEulerSolver(config={"damping_factor": 2.0})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize(
        ("solver_class", "setting", "key"),
        [
            (BackwardEulerSolver, "damping_factor", "computation.solvers.backward_euler.damping_factor"),
            (CrankNicolsonSolver, "theta", "computation.solvers.crank_nicolson.theta"),
        ],
    )
    def test_non_finite_config_values_are_rejected_by_the_yaml_entry(self, solver_class, setting, key, value):
        """A NaN or +/-inf configured value is rejected when it is resolved against its YAML entry.

        ParameterSpec.validate_value rejects a non-finite number with finite or
        open bounds, so the parameter resolution rejects the value (the
        "Configured <key>=<value>" message) before the solver's own range check.
        """
        with pytest.raises(ValueError, match=f"Configured {key}={value!r}"):
            solver_class(config={setting: value})

    def test_missing_parameter_is_a_key_error(self):
        solver = BackwardEulerSolver(dtype=torch.float64)
        with pytest.raises(KeyError, match="no_such_parameter"):
            solver.get_parameter("no_such_parameter")

    def test_other_manager_errors_keep_their_type(self, monkeypatch):
        """Only a missing key is reported as missing; other errors propagate unchanged."""
        solver = BackwardEulerSolver(dtype=torch.float64)

        class _FailingManager:
            def get_parameter(self, key):
                raise TypeError("corrupt parameter entry")

        monkeypatch.setattr(solver, "_get_parameter_manager", lambda: _FailingManager())
        with pytest.raises(TypeError, match="corrupt parameter entry"):
            solver.get_parameter("damping_factor")


class TestODESolvers:
    """Test suite for ODE solvers."""
    
    def test_backward_euler_linear(self):
        """Test backward Euler on linear ODE with known solution."""
        # dy/dt = -位y, y(0) = y0
        # Analytical solution: y(t) = y0 * exp(-位t)
        
        lam = 2.0
        y0 = torch.tensor([1.0])
        
        def ode_func(y, t):
            return -lam * y
        
        solver = BackwardEulerSolver(rtol=1e-6, atol=1e-8)
        t_points, y_traj = solver.forward(ode_func, y0, (0.0, 2.0), dt=0.1)
        
        # Compare with analytical solution
        t_analytical = t_points
        y_analytical = y0 * torch.exp(-lam * t_analytical)
        
        # Backward Euler should be reasonably accurate for linear ODEs
        error = torch.max(torch.abs(y_traj.squeeze() - y_analytical))
        assert error < 0.05  # Backward Euler with dt=0.1 has reasonable accuracy
    
    def test_crank_nicolson_linear(self):
        """Test Crank-Nicolson on linear ODE."""
        lam = 1.0
        y0 = torch.tensor([2.0])
        
        def ode_func(y, t):
            return -lam * y
        
        solver = CrankNicolsonSolver(rtol=1e-6, atol=1e-8)
        t_points, y_traj = solver.forward(ode_func, y0, (0.0, 1.0), dt=0.05)
        
        # Compare with analytical solution
        y_analytical = y0 * torch.exp(-lam * t_points)
        error = torch.max(torch.abs(y_traj.squeeze() - y_analytical))
        
        # Should be accurate for linear problems (Crank-Nicolson is 2nd order)
        assert error < 1e-3
    
    def test_backward_euler_has_no_unimplemented_newton_switch(self):
        """solve() only implements fixed-point iteration, so use_newton is not a key."""
        from relic.parameters import get_global_manager

        manager = get_global_manager(preload_all=True, warmup_cache=False)
        with pytest.raises(KeyError, match="use_newton"):
            manager.get("computation.solvers.backward_euler.use_newton", required=True)
        with pytest.raises(ValueError, match="use_newton"):
            BackwardEulerSolver(config={"use_newton": True})
        with pytest.raises(TypeError):
            BackwardEulerSolver(use_newton=True)
        solver = BackwardEulerSolver(dtype=torch.float64)
        assert "use_newton" not in solver.metadata.parameters
        assert not hasattr(solver, "use_newton")

    def test_crank_nicolson_theta_is_a_dimensionless_weight(self):
        """theta weights the new-time slope of the theta-method and has no units.

        y^{n+1} = y^n + dt * [theta * f(y^{n+1}) + (1 - theta) * f(y^n)];
        f and (y^{n+1} - y^n) / dt share units, so theta is dimensionless.
        """
        from relic.parameters import get_global_manager

        manager = get_global_manager(preload_all=True, warmup_cache=False)
        spec = manager.get_parameter("computation.solvers.crank_nicolson.theta")
        assert spec.units == "dimensionless"
        assert spec.dimension == "dimensionless"
        assert "weight" in spec.description
        assert [float(bound) for bound in spec.bounds] == [0.0, 1.0]
        assert CrankNicolsonSolver(dtype=torch.float64).theta == 0.5

    @pytest.mark.parametrize("damping", [0.0, -0.5, 1.5, float("nan")])
    def test_backward_euler_rejects_damping_outside_the_unit_interval(self, damping):
        """damping 0 would return y^n unchanged as a converged step."""
        with pytest.raises(ValueError, match="damping_factor"):
            BackwardEulerSolver(damping_factor=damping)

    def test_backward_euler_convergence_uses_the_undamped_fixed_point_residual(self):
        """A damped increment understates ||G(y) - y|| by the damping factor.

        Backward Euler: y^{n+1} = G(y^{n+1}) = y^n + dt f(y^{n+1}); for
        f = -lambda y the step is y^n / (1 + lambda dt).
        """
        lam, dt = 1.0, 0.5
        y0 = torch.tensor([1.0], dtype=torch.float64)
        solver = BackwardEulerSolver(
            dtype=torch.float64,
            damping_factor=0.01,
            atol=1.0e-8,
            rtol=0.0,
            max_iterations=5000,
        )

        result = solver.solve(lambda y, t: -lam * y, y0, 0.0, dt)

        assert result.status == SolverStatus.SUCCESS
        exact = y0 / (1.0 + lam * dt)
        # |y - y*| = |G(y) - y| / (1 + lambda dt) <= atol / 1.5 on the undamped
        # residual; the damped increment stopped near |G(y) - y| = atol / omega.
        assert float(torch.max(torch.abs(result.solution - exact))) <= 1.0e-7

    @pytest.mark.parametrize("theta", [-0.1, 1.5, float("nan")])
    def test_crank_nicolson_rejects_theta_outside_the_unit_interval(self, theta):
        with pytest.raises(ValueError, match="theta"):
            CrankNicolsonSolver(theta=theta)

    def test_stiff_ode_stability(self):
        """Test solver stability on stiff ODE."""
        # Stiff system: dy/dt = -1000y + 1000t, y(0) = 0
        # Analytical solution: y(t) = t - 0.001 + 0.001*exp(-1000t)
        
        def stiff_ode(y, t):
            return -1000.0 * y + 1000.0 * t

        y0 = torch.tensor([0.0])
        t_span = (0.0, 0.01)  # Short time interval for stiff problem

        # The implicit equation is solved by fixed-point iteration, whose map
        # y_{k+1} = y_n + dt * (-1000 y_k + 1000 t) contracts only for
        # dt * 1000 < 1. dt = 5e-4 (factor 0.5) converges at every step; at
        # dt = 1e-3 every step ends with MAX_ITERATIONS and forward raises
        # (test_forward_fails_fast_on_non_converged_implicit_step).
        solver_be = BackwardEulerSolver(max_iter=100)
        t_points, y_traj_be = solver_be.forward(stiff_ode, y0, t_span, dt=0.0005)

        exact = torch.tensor(
            [t - 0.001 + 0.001 * np.exp(-1000.0 * t) for t in t_points.tolist()],
            dtype=y_traj_be.dtype,
        )
        # forward raises SolverStepError on any failed step, so completing the
        # trajectory means every step converged.
        assert torch.all(torch.isfinite(y_traj_be))
        assert torch.max(torch.abs(y_traj_be.squeeze() - exact)) < 1.0e-4
    
    def test_nonlinear_ode(self):
        """Test solver on nonlinear ODE."""
        # Logistic equation: dy/dt = ry(1 - y/K), y(0) = y0
        # Analytical solution: y(t) = K / (1 + ((K-y0)/y0) * exp(-rt))
        
        r, K = 0.5, 10.0
        y0 = torch.tensor([1.0])
        
        def logistic_ode(y, t):
            return r * y * (1 - y / K)
        
        solver = BackwardEulerSolver(rtol=1e-6, max_iter=20)
        t_points, y_traj = solver.forward(logistic_ode, y0, (0.0, 10.0), dt=0.2)
        
        # Check final value approaches carrying capacity
        y_final = y_traj[-1].item()
        assert abs(y_final - K) < 0.6  # Should approach K = 10 (backward Euler with dt=0.2)
        
        # Check monotonic growth
        y_vals = y_traj.squeeze()
        assert torch.all(y_vals[1:] >= y_vals[:-1])  # Non-decreasing
    
    def test_system_of_odes(self):
        """Test solver on system of coupled ODEs."""
        # Predator-prey system:
        # dx/dt = ax - bxy
        # dy/dt = -cy + dxy
        
        a, b, c, d = 1.0, 0.1, 1.5, 0.075
        y0 = torch.tensor([10.0, 5.0])  # [prey, predator]
        
        def predator_prey(y, t):
            x, y_pred = y[0], y[1]
            dxdt = a * x - b * x * y_pred
            dydt = -c * y_pred + d * x * y_pred
            return torch.stack([dxdt, dydt])
        
        solver = CrankNicolsonSolver(rtol=1e-6, max_iter=50)
        t_points, y_traj = solver.forward(predator_prey, y0, (0.0, 5.0), dt=0.1)
        
        # Check that populations remain positive
        assert torch.all(y_traj >= 0)
        
        # Check conservation of energy-like quantity (approximate)
        # H(x,y) = d*x - c*ln(x) + b*y - a*ln(y) should be approximately constant
        x_traj, y_traj_pred = y_traj[:, 0], y_traj[:, 1]
        H = d * x_traj - c * torch.log(x_traj) + b * y_traj_pred - a * torch.log(y_traj_pred)
        H_variation = torch.std(H) / torch.mean(H)
        assert H_variation < 0.1  # Should be relatively constant
    
    def test_convergence_order_verification(self):
        """Test numerical convergence order of solvers."""
        # Test on: dy/dt = y + t, y(0) = 1
        # Analytical: y(t) = 2*exp(t) - t - 1
        
        def test_ode(y, t):
            return y + t
        
        y0 = torch.tensor([1.0])
        t_final = 1.0
        
        # Test Backward Euler (1st order)
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        errors_be = []
        
        for dt in dt_values:
            solver = BackwardEulerSolver()
            t_points, y_traj = solver.forward(test_ode, y0, (0.0, t_final), dt)
            
            y_numerical = y_traj[-1].item()
            y_analytical = 2 * np.exp(t_final) - t_final - 1
            error = abs(y_numerical - y_analytical)
            errors_be.append(error)
        
        # Check first-order convergence
        for i in range(1, len(errors_be)):
            convergence_rate = np.log(errors_be[i-1] / errors_be[i]) / np.log(2.0)
            assert 0.8 < convergence_rate < 1.2  # Should be close to 1
        
        # Test Crank-Nicolson (2nd order)
        errors_cn = []
        for dt in dt_values:
            solver = CrankNicolsonSolver()
            t_points, y_traj = solver.forward(test_ode, y0, (0.0, t_final), dt)
            
            y_numerical = y_traj[-1].item()
            y_analytical = 2 * np.exp(t_final) - t_final - 1
            error = abs(y_numerical - y_analytical)
            errors_cn.append(error)
        
        # Check second-order convergence
        for i in range(1, len(errors_cn)):
            convergence_rate = np.log(errors_cn[i-1] / errors_cn[i]) / np.log(2.0)
            assert 1.8 < convergence_rate < 2.2  # Should be close to 2
    
    def test_solve_ode_system_convenience(self):
        """Test convenience function for ODE solving."""
        def simple_ode(y, t):
            return -y
        
        y0 = torch.tensor([1.0])
        t_points, y_traj = solve_ode_system(
            simple_ode, y0, (0.0, 2.0), 
            method='backward_euler', dt=0.1
        )
        
        # Check basic properties
        assert len(t_points) == len(y_traj)
        assert torch.allclose(t_points[0], torch.tensor(0.0))
        assert torch.allclose(t_points[-1], torch.tensor(2.0))
        assert torch.allclose(y_traj[0], y0)

    _FORWARD_SOLVER_FACTORIES = pytest.mark.parametrize(
        "make_solver",
        [
            lambda: BackwardEulerSolver(dtype=torch.float64),
            lambda: ImplicitMidpointSolver(dtype=torch.float64),
            lambda: CrankNicolsonSolver(dtype=torch.float64),
            lambda: RK4Solver(name="rk4_forward_contract", dtype=torch.float64),
        ],
        ids=["backward_euler", "implicit_midpoint", "crank_nicolson", "rk4"],
    )

    @_FORWARD_SOLVER_FACTORIES
    def test_forward_rejects_removed_adaptive_option(self, make_solver):
        """The step-doubling forward(..., adaptive=True) option was removed.

        The Simulator update-magnitude heuristic is the only step-size
        controller (architecture_overview.md 4.4.1, D83). A request for the
        removed option must fail at the call boundary: it must neither run a
        second controller nor be forwarded to, and ignored by, the RHS.
        """
        solver = make_solver()
        rhs_calls = []

        def rhs_accepting_keywords(y, t, **kwargs):
            rhs_calls.append(kwargs)
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(TypeError, match="adaptive"):
            solver.forward(rhs_accepting_keywords, y0, (0.0, 1.0), 0.1, adaptive=True)
        with pytest.raises(TypeError, match="positional"):
            solver.forward(rhs_accepting_keywords, y0, (0.0, 1.0), 0.1, True)
        assert rhs_calls == []

    def test_solve_ode_system_rejects_removed_adaptive_option(self):
        """The convenience wrapper must not accept the removed adaptive option."""
        y0 = torch.tensor([1.0], dtype=torch.float64)
        for method in ("backward_euler", "implicit_midpoint", "crank_nicolson", "rk4"):
            with pytest.raises(TypeError, match="adaptive"):
                solve_ode_system(
                    lambda y, t: -y, y0, (0.0, 1.0),
                    method=method, dt=0.1, adaptive=True,
                )

    @_FORWARD_SOLVER_FACTORIES
    def test_forward_time_points_match_the_steps_taken(self, make_solver):
        """Every returned state belongs to its returned time point.

        References: Hairer, E. & Wanner, G. (1996), Solving ODEs II;
        Burden, R.L. & Faires, J.D. (2010), Numerical Analysis, Ch. 5.
        Mathematical expression: h_n = min(dt, t_end - t_n),
        t_{n+1} = t_n + h_n, y_{n+1} = Phi_{h_n}(y_n, t_n).
        Parameter meanings: dt requested fixed step, h_n step taken,
        Phi_h one step of the solver, t_end end of the interval.
        Units: t and dt in the time unit of the RHS; y dimensionless here.
        Domain/constraints: dt > 0; 0 < h_n <= dt; last time point equals t_end.
        Sign convention: time advances forward; the RHS -y is a decay.
        """
        solver = make_solver()
        replay = make_solver()

        def rhs(y, t):
            return -y

        dt = 0.1
        t_end = 1.05
        y0 = torch.tensor([1.0], dtype=torch.float64)
        t_points, y_traj = solver.forward(rhs, y0, (0.0, t_end), dt)

        assert t_points.dtype is torch.float64
        assert t_points.shape[0] == y_traj.shape[0]
        assert float(t_points[0]) == 0.0
        assert float(t_points[-1]) == t_end
        steps = torch.diff(t_points)
        assert bool(torch.all(steps > 0.0))
        assert bool(torch.all(steps <= dt * (1.0 + 1.0e-12)))
        for index in range(steps.shape[0]):
            expected = replay.step(
                rhs, y_traj[index], float(t_points[index]), float(steps[index])
            )
            torch.testing.assert_close(
                y_traj[index + 1], expected, rtol=1.0e-12, atol=1.0e-14
            )

    @_FORWARD_SOLVER_FACTORIES
    @pytest.mark.parametrize(
        ("t_span", "dt", "expected_steps"),
        [
            ((0.0, 1.0), 0.1, 10),
            ((0.0, 2.0), 0.1, 20),
            ((0.0, 1.0), 0.0125, 80),
            ((0.0, 10.0), 0.2, 50),
            ((1.0e9, 1.0e9 + 3.6), 0.06, 60),
            ((0.0, 1.0), 0.3, 4),
            ((0.0, 1.05), 0.1, 11),
        ],
    )
    def test_forward_step_grid_has_no_rounding_residual_step(
        self, make_solver, t_span, dt, expected_steps
    ):
        """Accumulated time rounding must not create an extra near-zero step.

        An interval that is an integer multiple of dt gives exactly that many
        steps; a genuine remainder gives one shorter final step. Interior
        points keep the accumulated grid t_{n+1} = t_n + dt, and the last
        point equals t_end exactly.
        """
        solver = make_solver()
        y0 = torch.tensor([1.0], dtype=torch.float64)
        t_points, y_traj = solver.forward(lambda y, t: -0.5 * y, y0, t_span, dt)
        t_start, t_end = t_span

        assert t_points.numel() == expected_steps + 1 == y_traj.shape[0]
        assert float(t_points[0]) == t_start
        assert float(t_points[-1]) == t_end
        steps = torch.diff(t_points)
        assert bool(torch.all(steps > 0.25 * dt))
        for index in range(expected_steps - 1):
            assert float(t_points[index + 1]) == float(t_points[index]) + dt

    @_FORWARD_SOLVER_FACTORIES
    def test_forward_time_grid_is_float64_for_any_state_dtype(self, make_solver):
        """The time grid is computed in binary64 and returned in float64.

        With float32 states at |t| ~ 1e9 (float32 spacing 64) a state-dtype
        time vector would collapse every time point to the same value.
        """
        solver = make_solver()
        t_start, t_end, dt = 1.0e9, 1.0e9 + 3.6, 0.06
        y0 = torch.tensor([1.0], dtype=torch.float32)
        t_points, y_traj = solver.forward(lambda y, t: -0.5 * y, y0, (t_start, t_end), dt)

        assert t_points.dtype is torch.float64
        assert y_traj.dtype is torch.float32
        assert t_points.device == y0.device
        assert t_points.numel() == 61 == y_traj.shape[0]
        assert float(t_points[0]) == t_start
        assert float(t_points[-1]) == t_end
        assert bool(torch.all(torch.diff(t_points) > 0.25 * dt))

    @pytest.mark.parametrize(
        ("t_span", "dt"),
        [
            ((0.0, 1.0), float("inf")),
            ((0.0, 1.0), float("nan")),
            ((float("nan"), 1.0), 0.1),
            ((0.0, float("nan")), 0.1),
        ],
    )
    def test_forward_rejects_nonfinite_time_arguments(self, t_span, dt):
        """Non-finite t_span or dt cannot define a step grid and must fail."""
        solver = BackwardEulerSolver(dtype=torch.float64)
        y0 = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="finite"):
            solver.forward(lambda y, t: -y, y0, t_span, dt)

    def test_rk4_has_no_adaptive_switch(self, caplog):
        """RK4 is fixed-step only: the warning-only switch and its YAML key are removed."""
        from relic.parameters import get_global_manager

        manager = get_global_manager(preload_all=True, warmup_cache=False)
        with pytest.raises(KeyError, match="computation.solvers.rk4.adaptive"):
            manager.get("computation.solvers.rk4.adaptive", required=True)
        for key in ("max_iterations", "rtol", "atol"):
            description = manager.get_parameter(
                f"computation.solvers.rk4.{key}"
            ).description
            assert "adaptive" not in description.lower()
            assert "fixed-step" in description

        with caplog.at_level(logging.WARNING, logger="relic.computation.solvers.temporal"):
            solver = RK4Solver(name="rk4_fixed_step_only", dtype=torch.float64)
        assert "adaptive" not in solver.metadata.parameters
        assert not hasattr(solver, "adaptive")
        assert [r for r in caplog.records if "daptive" in r.getMessage()] == []

    def test_solver_step_error_is_part_of_the_public_package_api(self):
        """SolverStepError is importable from the solvers package and listed in __all__."""
        import relic.computation.solvers as solvers_pkg
        from relic.computation.solvers.base import SolverStepError as base_error

        assert "SolverStepError" in solvers_pkg.__all__
        assert solvers_pkg.SolverStepError is base_error
        assert issubclass(base_error, RuntimeError)

    def test_forward_fails_fast_on_non_converged_implicit_step(self):
        """A MAX_ITERATIONS step must stop the trajectory with its diagnostics.

        For y' = -1000 y and dt = 1e-3 the Backward Euler fixed-point map
        y_{k+1} = y_n + dt * (-1000 y_k) has contraction factor dt * 1000 = 1,
        so the iteration cannot converge.
        """
        from relic.computation.solvers.base import SolverStepError

        solver = BackwardEulerSolver(dtype=torch.float64, max_iterations=5)
        y0 = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(SolverStepError) as excinfo:
            solver.forward(lambda y, t: -1000.0 * y, y0, (0.0, 0.01), 0.001)
        error = excinfo.value
        assert error.status is SolverStatus.MAX_ITERATIONS
        assert error.step_index == 0
        assert error.t_current == 0.0
        assert error.dt == 0.001
        assert error.iterations == 5
        assert error.residual_norm is not None and error.residual_norm > 0.0
        assert "max_iterations" in str(error) and "step 0" in str(error)

    def test_forward_fails_fast_on_diverged_step_with_its_index_and_time(self):
        """A DIVERGED step later in the run reports the step index and start time."""
        from relic.computation.solvers.base import SolverStepError

        solver = RK4Solver(name="rk4_step_status", dtype=torch.float64)

        def rhs_blowing_up_at_quarter(y, t):
            if t >= 0.25:
                return torch.full_like(y, float("inf"))
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(SolverStepError) as excinfo:
            solver.forward(rhs_blowing_up_at_quarter, y0, (0.0, 1.0), 0.1)
        error = excinfo.value
        assert error.status is SolverStatus.DIVERGED
        assert error.step_index == 2
        assert error.t_current == 0.1 + 0.1
        assert error.metadata["divergence_reason"] == "non_finite_solution"

        with pytest.raises(SolverStepError) as direct:
            solver.step(rhs_blowing_up_at_quarter, y0, 0.3, 0.1)
        assert direct.value.step_index is None
        assert direct.value.status is SolverStatus.DIVERGED

    def test_rk4_solve_propagates_rhs_exceptions(self):
        """Genuine errors in the RHS must propagate (fail fast), not become DIVERGED."""
        solver = RK4Solver(name="rk4_fail_fast", dtype=torch.float64)
        y0 = torch.ones(1, dtype=torch.float64)

        def failing_rhs(y, t):
            raise RuntimeError("rhs evaluation failed")

        def wrong_signature_rhs(y):
            return -y

        with pytest.raises(RuntimeError, match="rhs evaluation failed"):
            solver.solve(failing_rhs, y0, 0.0, 0.1)
        with pytest.raises(TypeError):
            solver.solve(wrong_signature_rhs, y0, 0.0, 0.1)

    def test_rk4_solve_reports_only_nonfinite_results_as_diverged(self):
        """A numerically non-finite RK4 result is DIVERGED, explicitly and visibly."""
        solver = RK4Solver(name="rk4_nonfinite", dtype=torch.float64)
        y0 = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def overflowing_rhs(y, t):
            return torch.full_like(y, torch.finfo(y.dtype).max)

        diverged = solver.solve(overflowing_rhs, y0, 0.0, 10.0)
        assert diverged.status is SolverStatus.DIVERGED
        assert not bool(torch.any(torch.isfinite(diverged.solution)))
        assert diverged.metadata["divergence_reason"] == "non_finite_solution"
        assert diverged.metadata["non_finite_elements"] == 2
        assert diverged.iterations == 4

        # References: Kutta (1901); Butcher (2008), classical RK4.
        # Mathematical expression: for y' = -y one RK4 step multiplies y by
        #   R(h) = 1 - h + h^2/2 - h^3/6 + h^4/24.
        # Parameter meanings: h step size, R stability function of RK4.
        # Units: dimensionless test problem.
        # Domain/constraints: h = 0.1 inside the RK4 stability interval.
        # Sign convention: -y is a decay.
        step = 0.1
        finite = solver.solve(lambda y, t: -y, y0, 0.0, step)
        assert finite.status is SolverStatus.SUCCESS
        amplification = 1.0 - step + step**2 / 2.0 - step**3 / 6.0 + step**4 / 24.0
        torch.testing.assert_close(finite.solution, y0 * amplification)

    @pytest.mark.parametrize("method", ["forward_euler", "euler", "rk45"])
    def test_solve_ode_system_rejects_unsupported_methods(self, method):
        """The method name must match the method run; unsupported names fail.

        No explicit Euler ODE solver is registered, so "forward_euler" must not
        silently run backward Euler.
        """
        y0 = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError) as excinfo:
            solve_ode_system(lambda y, t: -y, y0, (0.0, 1.0), method=method, dt=0.1)
        message = str(excinfo.value)
        assert repr(method) in message
        supported_part = message.split("supported methods:", 1)[1]
        for supported in ("backward_euler", "implicit_midpoint", "crank_nicolson", "rk4"):
            assert supported in supported_part
        assert "forward_euler" not in supported_part


class TestBracketedBisection:
    """Test the registered batched nonlinear scalar root solver."""

    @staticmethod
    def _create_solver(**config):
        settings = {
            "max_iterations": 80,
            "rtol": 1.0e-12,
            "atol": 1.0e-12,
        }
        settings.update(config)
        return SolverFactory.create(
            solver_name="bracketed_bisection",
            name="root_solver",
            config=settings,
            dtype=torch.float64,
        )

    def test_registry_and_scalar_root(self):
        assert SolverFactory.get_solvers_for_problem("nonlinear") == [
            "bracketed_bisection"
        ]
        solver = self._create_solver()
        target = torch.tensor(2.0, dtype=torch.float64)
        result = solver(
            lambda x, value: x.square() - value,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
            parameters=(target,),
        )

        assert result.status is SolverStatus.SUCCESS
        assert torch.allclose(
            result.solution,
            torch.sqrt(target),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_batch_dtype_and_diagnostics(self):
        solver = self._create_solver()
        target = torch.tensor([2.0, 3.0], dtype=torch.float64)
        lower = torch.zeros_like(target)
        upper = torch.full_like(target, 2.0)
        result = solver(
            lambda x, value: x.square() - value,
            lower,
            upper,
            parameters=(target,),
        )

        assert result.status is SolverStatus.SUCCESS
        assert result.solution.dtype is torch.float64
        assert result.solution.device == target.device
        assert torch.allclose(result.solution, torch.sqrt(target), rtol=1e-10)
        for key in (
            "converged_mask",
            "iterations_per_element",
            "final_lower",
            "final_upper",
            "residual",
        ):
            assert result.metadata[key].device == target.device
        assert result.metadata["converged_mask"].dtype is torch.bool
        assert result.metadata["iterations_per_element"].dtype is torch.long

    def test_exact_endpoint_roots(self):
        solver = self._create_solver()
        target = torch.tensor([0.0, 4.0], dtype=torch.float64)
        result = solver(
            lambda x, value: x.square() - value,
            torch.tensor([0.0, 0.0], dtype=torch.float64),
            torch.tensor([2.0, 2.0], dtype=torch.float64),
            parameters=(target,),
        )

        assert result.status is SolverStatus.SUCCESS
        assert torch.equal(result.solution, torch.tensor([0.0, 2.0], dtype=torch.float64))
        assert torch.equal(
            result.metadata["iterations_per_element"],
            torch.zeros(2, dtype=torch.long),
        )

    @pytest.mark.parametrize(
        ("lower", "upper", "error"),
        [
            (torch.tensor(1.0), torch.tensor(1.0), ValueError),
            (torch.tensor(2.0), torch.tensor(1.0), ValueError),
            (torch.tensor(float("nan")), torch.tensor(1.0), ValueError),
            (torch.tensor(0.0), torch.tensor(float("inf")), ValueError),
        ],
    )
    def test_invalid_bounds_fail_fast(self, lower, upper, error):
        solver = self._create_solver()
        with pytest.raises(error):
            solver(
                lambda x, value: x - value,
                lower.to(torch.float64),
                upper.to(torch.float64),
                parameters=(torch.tensor(0.5, dtype=torch.float64),),
            )

    def test_invalid_shapes_dtypes_and_bracket_fail_fast(self):
        solver = self._create_solver()
        with pytest.raises(ValueError, match="shape"):
            solver(
                lambda x, value: x - value,
                torch.zeros(2, dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
                parameters=(torch.ones(2, dtype=torch.float64),),
            )
        with pytest.raises(ValueError, match="dtype"):
            solver(
                lambda x, value: x - value,
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
                parameters=(torch.tensor(0.5, dtype=torch.float32),),
            )
        with pytest.raises(ValueError, match="bracket"):
            solver(
                lambda x: x.square() + 1.0,
                torch.tensor(-1.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            )

    def test_nonfinite_residual_fails_fast(self):
        solver = self._create_solver()
        with pytest.raises(FloatingPointError, match="finite"):
            solver(
                lambda x: torch.full_like(x, float("nan")),
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            )

    def test_max_iterations_is_explicit_failure(self):
        solver = self._create_solver(max_iterations=1, rtol=0.0, atol=0.0)
        result = solver(
            lambda x: x.square() - 2.0,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
        )

        assert result.status is SolverStatus.MAX_ITERATIONS
        assert not bool(torch.all(result.metadata["converged_mask"]))

    def test_implicit_parameter_gradient_and_gradcheck(self):
        solver = self._create_solver()
        lower = torch.zeros(2, dtype=torch.float64)
        upper = torch.full((2,), 2.0, dtype=torch.float64)
        target = torch.tensor(
            [2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def solve_target(value):
            return solver(
                lambda x, parameter: x.square() - parameter,
                lower,
                upper,
                parameters=(value,),
            ).solution

        root = solve_target(target)
        root.sum().backward()
        assert torch.allclose(
            target.grad,
            0.5 / torch.sqrt(target.detach()),
            rtol=1e-9,
            atol=1e-10,
        )
        assert torch.autograd.gradcheck(
            solve_target,
            (target.detach().clone().requires_grad_(True),),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        assert torch.autograd.gradgradcheck(
            solve_target,
            (target.detach().clone().requires_grad_(True),),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )

    def test_log_root_retains_the_implicit_second_derivative(self):
        # References: Blondel et al. (2022), modular implicit differentiation.
        # Mathematical expression: F(T, theta)=T^2-theta=0;
        #   d log(T)/d theta=1/(2 theta), d2 log(T)/d theta2=-1/(2 theta^2).
        # Parameter meanings: theta is the explicit residual parameter; T is the root.
        # Units: normalized scalar values for the derivative oracle.
        # Domain/constraints: theta>0, positive simple root, smooth residual.
        # Sign convention: increasing theta increases T and decreases its sensitivity.
        solver = self._create_solver()
        theta = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
        result = solver(
            lambda root, parameter: root.square() - parameter,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(3.0, dtype=torch.float64),
            parameters=(theta,),
        )
        first = torch.autograd.grad(torch.log(result.solution), theta, create_graph=True)[0]
        assert first.requires_grad
        second = torch.autograd.grad(first, theta)[0]
        torch.testing.assert_close(first, 0.5 / theta)
        torch.testing.assert_close(second, -0.5 / theta.square())
        assert result.metadata["gradient"] == "second_order_implicit"

    @pytest.mark.parametrize("shared", ["coefficient", "target", "neither"])
    def test_batched_implicit_second_derivatives_with_shared_parameters(self, shared):
        solver = self._create_solver()
        coefficient = torch.tensor(
            1.25 if shared == "coefficient" else [1.25, 0.75],
            dtype=torch.float64, requires_grad=True,
        )
        target = torch.tensor(
            2.0 if shared == "target" else [2.0, 3.0],
            dtype=torch.float64, requires_grad=True,
        )
        lower = torch.zeros(2, dtype=torch.float64)
        upper = torch.full((2,), 3.0, dtype=torch.float64)

        def solve_parameters(a, theta):
            return solver(
                lambda root, scale, value: scale * root.square() - value,
                lower, upper, parameters=(a, theta),
            ).solution

        actual = solve_parameters(coefficient, target)
        expected = torch.sqrt(target / coefficient)
        weights = torch.tensor([0.5, -2.0], dtype=torch.float64)
        actual_gradients = torch.autograd.grad((actual * weights).sum(), (coefficient, target))
        expected_gradients = torch.autograd.grad((expected * weights).sum(), (coefficient, target))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.autograd.gradgradcheck(
            solve_parameters, (coefficient, target), eps=1e-6, atol=1e-5, rtol=1e-4,
        )

    def test_cuda_device_is_preserved_when_available(self):
        if not torch.cuda.is_available():
            return
        solver = SolverFactory.create(
            solver_name="bracketed_bisection",
            name="cuda_root_solver",
            config={"max_iterations": 80, "rtol": 1e-12, "atol": 1e-12},
            device=torch.device("cuda"),
            dtype=torch.float64,
        )
        target = torch.tensor([2.0, 3.0], device="cuda", dtype=torch.float64)
        result = solver(
            lambda x, value: x.square() - value,
            torch.zeros_like(target),
            torch.full_like(target, 2.0),
            parameters=(target,),
        )
        assert result.status is SolverStatus.SUCCESS
        assert result.solution.device.type == "cuda"

    def test_implicit_second_derivatives_reach_packed_state_views(self):
        solver = self._create_solver()
        state = torch.tensor([1.25, 2.0], dtype=torch.float64, requires_grad=True)

        def solve_state(values):
            return solver(
                lambda root, a, theta: a * root.square() - theta,
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(3.0, dtype=torch.float64),
                parameters=(values[0], values[1]),
            ).solution

        assert torch.autograd.gradgradcheck(solve_state, (state,), eps=1e-6, atol=1e-5, rtol=1e-4)

    def test_implicit_gradient_rejects_a_singular_root_without_regularization(self):
        solver = self._create_solver()
        target = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        root = solver(
            lambda value, parameter: value.square() - parameter,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64), parameters=(target,),
        ).solution
        with pytest.raises(RuntimeError, match="derivative must be nonzero"):
            torch.autograd.grad(root, target)


class TestNumericalAccuracy:
    """Test numerical accuracy and precision."""
    
    def test_machine_precision_limits(self):
        """Test behavior at machine precision limits."""
        # Very small system
        a = torch.tensor([0.0, 1e-15])
        b = torch.tensor([1.0, 1.0])
        c = torch.tensor([1e-15, 0.0])
        d = torch.tensor([1.0, 1.0])
        
        x = solve_tridiagonal_thomas(a, b, c, d)
        assert torch.all(torch.isfinite(x))
    
    def test_different_dtypes(self):
        """Test solvers with different tensor dtypes."""
        # Test float32 vs float64
        a64 = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
        b64 = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        c64 = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        d64 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        
        x64 = solve_tridiagonal_thomas(a64, b64, c64, d64)
        
        # Convert to float32
        a32 = a64.to(torch.float32)
        b32 = b64.to(torch.float32)
        c32 = c64.to(torch.float32)
        d32 = d64.to(torch.float32)
        
        x32 = solve_tridiagonal_thomas(a32, b32, c32, d32)
        
        # Results should be close (within float32 precision)
        assert torch.allclose(x64.to(torch.float32), x32, rtol=1e-6)
    
    def test_gpu_consistency(self):
        """Test GPU consistency if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Move to GPU
        device = torch.device('cuda')
        a = torch.tensor([0.0, 1.0, 1.0], device=device)
        b = torch.tensor([2.0, 2.0, 2.0], device=device)
        c = torch.tensor([1.0, 1.0, 0.0], device=device)
        d = torch.tensor([1.0, 2.0, 1.0], device=device)
        
        x_gpu = solve_tridiagonal_thomas(a, b, c, d)
        
        # Compare with CPU result
        x_cpu = solve_tridiagonal_thomas(
            a.cpu(), b.cpu(), c.cpu(), d.cpu()
        )
        
        assert torch.allclose(x_gpu.cpu(), x_cpu, atol=1e-10)


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    def test_batch_performance(self):
        """Test batch processing performance."""
        batch_sizes = [1, 10, 100]
        n = 100
        
        for batch_size in batch_sizes:
            a = torch.randn(batch_size, n) * 0.1
            b = torch.ones(batch_size, n) * 2.0
            c = torch.randn(batch_size, n) * 0.1
            d = torch.randn(batch_size, n)
            
            # Should complete without error
            x = solve_tridiagonal_batch(a, b, c, d)
            assert x.shape == (batch_size, n)
    
    def test_large_system_stability(self):
        """Test stability with large systems."""
        n = 1000
        
        # Large well-conditioned system
        a = torch.ones(n) * -1.0
        b = torch.ones(n) * 3.0
        c = torch.ones(n) * -1.0
        d = torch.ones(n)
        
        x = solve_tridiagonal_thomas(a, b, c, d)
        
        # Verify solution
        residual = self._compute_large_residual(a, b, c, x, d)
        assert torch.max(torch.abs(residual)) < 1e-6
    
    def _compute_large_residual(self, a, b, c, x, d):
        """Efficient residual computation for large systems."""
        n = x.numel()
        Ax = torch.zeros_like(x)
        
        Ax[0] = b[0] * x[0] + c[0] * x[1]
        Ax[1:n-1] = a[1:n-1] * x[0:n-2] + b[1:n-1] * x[1:n-1] + c[1:n-1] * x[2:n]
        Ax[n-1] = a[n-1] * x[n-2] + b[n-1] * x[n-1]
        
        return Ax - d


class TestRound5SolverDeclarations:
    """Round 5: order, iteration count and solver metadata are declared."""

    def test_solver_metadata_declares_the_order(self):
        from relic.computation.solvers.base import SolverFamily, SolverMetadata

        with pytest.raises(TypeError, match="order"):
            SolverMetadata(name="probe", solver_family=SolverFamily.NONLINEAR)
        assert SolverMetadata(name="probe", solver_family=SolverFamily.NONLINEAR, order=2).order == 2

    def test_solver_result_declares_the_iteration_count(self):
        from relic.computation.solvers.base import SolverResult

        with pytest.raises(TypeError, match="iterations"):
            SolverResult(solution=torch.zeros(1), status=SolverStatus.SUCCESS)
        result = SolverResult(solution=torch.zeros(1), status=SolverStatus.SUCCESS, iterations=0)
        assert result.iterations == 0

    def test_solver_without_declared_metadata_is_refused(self):
        from relic.computation.solvers.base import SolverBase, SolverFamily

        class _UndeclaredSolver(SolverBase):
            solver_family = SolverFamily.NONLINEAR

            def solve(self, *args, **kwargs):
                raise NotImplementedError

        with pytest.raises(TypeError, match="SolverMetadata"):
            _UndeclaredSolver(name="undeclared_probe")


class TestSolverCallHasNoSideEffects:
    """I1: calling a solver changes neither its tensors nor its attributes."""

    @staticmethod
    def _snapshot(solver):
        tensors = {key: value.clone() for key, value in solver.state_dict().items()}
        attributes = {
            key: value for key, value in vars(solver).items()
            if not key.startswith("_") and not isinstance(value, torch.Tensor)
        }
        return tensors, attributes

    def _assert_unchanged(self, solver, before):
        tensors, attributes = self._snapshot(solver)
        assert tensors.keys() == before[0].keys()
        for key, value in tensors.items():
            assert torch.equal(value, before[0][key]), key
        assert attributes == before[1]

    def test_thomas_solve_and_forward_leave_the_solver_unchanged(self):
        solver = ThomasAlgorithmSolver(check_stability=True, dtype=torch.float64)
        a = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
        b = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        c = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        before = self._snapshot(solver)
        solver(a, b, c, d)
        solver.solve(a, b, c, d)
        self._assert_unchanged(solver, before)
        for counter in ("solve_count", "total_iterations", "failed_solves", "last_condition_number"):
            assert not hasattr(solver, counter), counter

    def test_ode_trajectory_leaves_the_solver_unchanged(self):
        solver = BackwardEulerSolver(dtype=torch.float64)
        y0 = torch.tensor([1.0], dtype=torch.float64)
        before = self._snapshot(solver)
        solver.forward(lambda y, t: -y, y0, (0.0, 0.3), 0.1)
        self._assert_unchanged(solver, before)
        for counter in ("solve_count", "total_iterations", "failed_solves"):
            assert not hasattr(solver, counter), counter


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


