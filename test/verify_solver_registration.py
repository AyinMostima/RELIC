"""
Verification script for solver registration in RELIC unified computation layer.

This script tests that all newly migrated solvers are properly registered
in the SolverRegistry and can be instantiated via SolverFactory.

Graft can report zero direct in-degree for registry and factory symbols because
decorators execute at import time and lookups are string/config driven. Static
zero in-degree is therefore not evidence that these canonical owners are dead.

Expected registered solvers:
- backward_euler (ODE_IMPLICIT)
- implicit_midpoint (ODE_IMPLICIT)
- crank_nicolson (ODE_IMPLICIT)
- rk4 (ODE_EXPLICIT)
- thomas_algorithm (LINEAR_DIRECT)
- bracketed_bisection (NONLINEAR)
"""

import sys
import torch
import pytest
from pathlib import Path

# Add relic to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relic.computation.solvers import (
    BackwardEulerSolver,
    BracketedBisectionSolver,
    CrankNicolsonSolver,
    IMEXEulerSolver,
    ImplicitMidpointSolver,
    RK4Solver,
    SolverRegistry,
    SolverFactory,
    SolverFamily,
    SolverStatus,
    ThomasAlgorithmSolver,
)


EXPECTED_SOLVER_CLASSES = {
    "backward_euler": BackwardEulerSolver,
    "implicit_midpoint": ImplicitMidpointSolver,
    "crank_nicolson": CrankNicolsonSolver,
    "imex_euler": IMEXEulerSolver,
    "rk4": RK4Solver,
    "thomas_algorithm": ThomasAlgorithmSolver,
    "bracketed_bisection": BracketedBisectionSolver,
}

def print_separator(title=""):
    """Print formatted separator."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")
    else:
        print(f"{'=' * 80}")


def _check_registry_info():
    """Check registry information and return True/False for CLI."""
    print_separator("TEST 1: Registry Information")
    
    info = SolverRegistry.get_registry_info()
    print(f"[INFO] Total solvers registered: {info['total_solvers']}")
    print(f"[INFO] Solver names: {', '.join(info['solver_names'])}")
    print(f"[INFO] Families: {info['families']}")
    print(f"[INFO] Implicit solvers: {info['implicit_solvers']}")
    print(f"[INFO] Jacobian-required solvers: {info['jacobian_required_solvers']}")
    
    # Check expected solvers
    expected_solvers = [
        "backward_euler",
        "implicit_midpoint",
        "crank_nicolson",
        "rk4",
        "imex_euler",
        "thomas_algorithm",
        "bracketed_bisection",
    ]
    missing = [s for s in expected_solvers if s not in info['solver_names']]
    
    if missing:
        print(f"[ERROR] Missing expected solvers: {missing}")
        return False
    else:
        print(f"[OK] All expected solvers are registered")
        return True


def test_registry_info():
    """Test 1: Print registry information."""
    assert _check_registry_info()


def test_import_time_registry_exactly_matches_decorated_implementations():
    """Import-time registration must expose every and only real solver class."""
    info = SolverRegistry.get_registry_info()

    assert info["total_solvers"] == len(EXPECTED_SOLVER_CLASSES)
    assert set(info["solver_names"]) == set(EXPECTED_SOLVER_CLASSES)
    for solver_name, solver_class in EXPECTED_SOLVER_CLASSES.items():
        assert SolverRegistry.get(solver_name) is solver_class


def test_config_driven_string_lookup_instantiates_real_solver():
    """A validated config must reach the canonical registry-backed factory."""
    solver = SolverFactory.create_from_config(
        {
            "solver_name": "bracketed_bisection",
            "name": "config_driven_root_solver",
            "config": {},
            "device": "cpu",
            "dtype": "float64",
        }
    )

    assert type(solver) is BracketedBisectionSolver
    assert solver.name == "config_driven_root_solver"
    assert solver.max_iterations == 80


@pytest.mark.parametrize("entrypoint", ("registry", "factory"))
def test_unknown_solver_key_fails_explicitly(entrypoint):
    """Unknown string keys cannot fall back to a default solver."""
    unknown = "__q1_unknown_solver__"

    with pytest.raises(KeyError, match=unknown):
        if entrypoint == "registry":
            SolverRegistry.get(unknown)
        else:
            SolverFactory.create(unknown, "unknown_solver_instance")


def _check_solver_metadata():
    """Check solver metadata and return True/False for CLI."""
    print_separator("TEST 2: Solver Metadata")
    
    solvers = [
        "backward_euler",
        "implicit_midpoint",
        "crank_nicolson",
        "rk4",
        "imex_euler",
        "thomas_algorithm",
        "bracketed_bisection",
    ]
    all_ok = True
    
    for solver_name in solvers:
        print(f"\n[INFO] Checking metadata for: {solver_name}")
        try:
            metadata = SolverRegistry.get_metadata(solver_name)
            if metadata is None:
                print(f"  [WARN] No metadata available for {solver_name}")
                continue
            
            print(f"  Name: {metadata.name}")
            print(f"  Family: {metadata.solver_family}")
            print(f"  Order: {metadata.order}")
            print(f"  Implicit: {metadata.is_implicit}")
            print(f"  Stability: {metadata.stability_region}")
            print(f"  References: {len(metadata.references)} citations")
            print(f"  [OK] Metadata complete")
            
        except Exception as e:
            print(f"  [ERROR] Failed to get metadata: {e}")
            all_ok = False
    
    return all_ok


def test_solver_metadata():
    """Test 2: Check solver metadata."""
    assert _check_solver_metadata()


def _check_factory_creation():
    """Create solver instances via factory and return True/False for CLI."""
    print_separator("TEST 3: Factory Creation")
    
    test_cases = [
        ("backward_euler", {"max_iterations": 50, "rtol": 1e-6}),
        ("implicit_midpoint", {"max_iterations": 50, "rtol": 1e-6}),
        ("crank_nicolson", {"theta": 0.5, "max_iterations": 50}),
        ("rk4", {}),
        ("imex_euler", {"explicit_substeps": 2, "damping_factor": 1.0, "max_iterations": 50, "rtol": 1e-8, "atol": 1e-10}),
        ("thomas_algorithm", {"check_stability": True}),
        (
            "bracketed_bisection",
            {"max_iterations": 80, "rtol": 1e-10, "atol": 1e-10},
        ),
    ]
    
    all_ok = True
    
    for solver_name, config in test_cases:
        print(f"\n[INFO] Creating solver: {solver_name}")
        try:
            solver = SolverFactory.create(
                solver_name=solver_name,
                name=f"{solver_name}_instance",
                config=config
            )
            print(f"  Created: {solver}")
            print(f"  Type: {type(solver).__name__}")
            print(f"  Device: {solver.device}")
            print(f"  [OK] Successfully created")
            
        except Exception as e:
            print(f"  [ERROR] Failed to create: {e}")
            all_ok = False
    
    return all_ok


def test_factory_creation():
    """Test 3: Create solver instances via factory."""
    assert _check_factory_creation()


def _check_backward_euler_solve():
    """Test BackwardEuler solver with simple ODE and return True/False for CLI."""
    print_separator("TEST 4: BackwardEuler Functional Test")
    
    print("[INFO] Testing dy/dt = -lambda*y (exponential decay)")
    
    try:
        solver = SolverFactory.create(
            solver_name="backward_euler",
            name="be_test",
            config={"max_iterations": 100, "rtol": 1e-8, "atol": 1e-10}
        )
        
        # Define ODE: dy/dt = -lambda*y, solution: y(t) = y0*exp(-lambda*t)
        lambda_val = 1.0
        def ode_func(y, t):
            return -lambda_val * y
        
        # Initial condition
        y0 = torch.tensor([1.0])
        t0 = 0.0
        dt = 0.1
        
        # Solve one step
        result = solver.solve(ode_func, y0, t0, dt)
        
        print(f"  Initial: y0 = {y0.item():.6f}")
        print(f"  Time step: dt = {dt}")
        print(f"  Status: {result.status}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Solution: y({dt}) = {result.solution.item():.6f}")
        
        # Exact solution: y(0.1) = exp(-0.1) ~ 0.904837
        exact = torch.exp(torch.tensor(-lambda_val * dt))
        error = torch.abs(result.solution - exact).item()
        print(f"  Exact: y({dt}) = {exact.item():.6f}")
        print(f"  Error: {error:.6e}")
        
        if result.status == SolverStatus.SUCCESS and error < 1e-2:
            print(f"  [OK] Backward Euler solved correctly")
            return True
        else:
            print(f"  [ERROR] Solution not accurate enough")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_euler_solve():
    """Test 4: Test BackwardEuler solver with simple ODE."""
    assert _check_backward_euler_solve()


def _check_rk4_solve():
    """Test RK4 solver with simple ODE and return True/False for CLI."""
    print_separator("TEST 5: RK4 Functional Test")
    
    print("[INFO] Testing dy/dt = -y (exponential decay)")
    
    try:
        solver = SolverFactory.create(
            solver_name="rk4",
            name="rk4_test",
            config={}
        )
        
        def ode_func(y, t):
            return -y
        
        y0 = torch.tensor([1.0])
        t0 = 0.0
        dt = 0.1
        
        result = solver.solve(ode_func, y0, t0, dt)
        
        print(f"  Initial: y0 = {y0.item():.6f}")
        print(f"  Status: {result.status}")
        print(f"  Function evals: {result.iterations}")
        print(f"  Solution: y({dt}) = {result.solution.item():.6f}")
        
        exact = torch.exp(torch.tensor(-dt))
        error = torch.abs(result.solution - exact).item()
        print(f"  Exact: y({dt}) = {exact.item():.6f}")
        print(f"  Error: {error:.6e}")
        
        # RK4 should be very accurate (4th order)
        if result.status == SolverStatus.SUCCESS and error < 1e-6:
            print(f"  [OK] RK4 solved correctly with high accuracy")
            return True
        else:
            print(f"  [ERROR] Solution not accurate enough for RK4")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rk4_solve():
    """Test 5: Test RK4 solver with simple ODE."""
    assert _check_rk4_solve()


def _check_thomas_algorithm_solve():
    """Test Thomas algorithm with tridiagonal system and return True/False for CLI."""
    print_separator("TEST 6: Thomas Algorithm Functional Test")
    
    print("[INFO] Testing tridiagonal system from 1D diffusion")
    
    try:
        solver = SolverFactory.create(
            solver_name="thomas_algorithm",
            name="thomas_test",
            config={"check_stability": True}
        )
        
        # Create a simple tridiagonal system: Ax = d
        # [2 -1  0  0]   [x0]   [1]
        # [-1 2 -1  0] * [x1] = [0]
        # [0 -1  2 -1]   [x2]   [0]
        # [0  0 -1  2]   [x3]   [1]
        
        n = 4
        a = torch.tensor([-1.0, -1.0, -1.0, -1.0])  # Lower diagonal
        b = torch.tensor([2.0, 2.0, 2.0, 2.0])      # Main diagonal
        c = torch.tensor([-1.0, -1.0, -1.0, -1.0])  # Upper diagonal
        d = torch.tensor([1.0, 0.0, 0.0, 1.0])      # RHS
        
        result = solver.solve(a, b, c, d)
        
        print(f"  System size: n = {n}")
        print(f"  Status: {result.status}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Solution: x = {result.solution.numpy()}")
        
        # Verify: compute residual r = Ax - d
        x = result.solution
        residual = torch.zeros_like(d)
        residual[0] = b[0]*x[0] + c[0]*x[1] - d[0]
        for i in range(1, n-1):
            residual[i] = a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] - d[i]
        residual[n-1] = a[n-1]*x[n-2] + b[n-1]*x[n-1] - d[n-1]
        
        residual_norm = torch.max(torch.abs(residual)).item()
        print(f"  Residual norm: ||Ax - d|| = {residual_norm:.6e}")
        
        if result.status == SolverStatus.SUCCESS and residual_norm < 1e-6:
            print(f"  [OK] Thomas algorithm solved correctly")
            return True
        else:
            print(f"  [ERROR] Solution residual too large")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thomas_algorithm_solve():
    """Test 6: Test Thomas algorithm with tridiagonal system."""
    assert _check_thomas_algorithm_solve()


def test_bracketed_bisection_solve_and_parameter_defaults():
    """Verify the nonlinear registry family, YAML defaults, and scalar root."""
    solver = SolverFactory.create(
        solver_name="bracketed_bisection",
        name="bracketed_bisection_test",
        config={},
        dtype=torch.float64,
    )
    assert solver.solver_family is SolverFamily.NONLINEAR
    assert solver.max_iterations == 80
    assert solver.rtol == 1.0e-10
    assert solver.atol == 1.0e-10

    target = torch.tensor(2.0, dtype=torch.float64)
    result = solver(
        lambda x, value: x.square() - value,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(2.0, dtype=torch.float64),
        parameters=(target,),
    )
    assert result.status is SolverStatus.SUCCESS
    assert torch.allclose(result.solution, torch.sqrt(target), rtol=1e-9)


def main():
    """Run all verification tests."""
    print_separator("RELIC Solver Registration Verification")
    print("Testing solver registration in unified architecture")
    
    tests = [
        ("Registry Info", _check_registry_info),
        ("Solver Metadata", _check_solver_metadata),
        ("Factory Creation", _check_factory_creation),
        ("BackwardEuler Solve", _check_backward_euler_solve),
        ("RK4 Solve", _check_rk4_solve),
        ("Thomas Algorithm Solve", _check_thomas_algorithm_solve)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[FATAL ERROR] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print_separator("Test Summary")
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

