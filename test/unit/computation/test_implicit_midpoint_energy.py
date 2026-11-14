import torch

from relic.computation.solvers.temporal import ImplicitMidpointSolver


def _oscillator_rhs(y: torch.Tensor, _t: float) -> torch.Tensor:
    x = y[..., 0]
    v = y[..., 1]
    return torch.stack([v, -x], dim=-1)


def _energy(y: torch.Tensor) -> torch.Tensor:
    x = y[..., 0]
    v = y[..., 1]
    return 0.5 * (x * x + v * v)


def test_implicit_midpoint_preserves_harmonic_oscillator_energy():
    dtype = torch.float64
    dt = 0.1
    n_steps = 10000

    y_midpoint = torch.tensor([[1.0, 0.0]], dtype=dtype)
    y_euler = y_midpoint.clone()

    solver = ImplicitMidpointSolver(
        config={"max_iterations": 20, "rtol": 1.0e-12, "atol": 1.0e-12},
        dtype=dtype,
    )

    e0_mid = _energy(y_midpoint).item()
    e0_euler = _energy(y_euler).item()

    t = 0.0
    for _ in range(n_steps):
        result = solver.solve(_oscillator_rhs, y_midpoint, t_current=t, dt=dt)
        y_midpoint = result.solution
        y_euler = y_euler + dt * _oscillator_rhs(y_euler, t)
        t += dt

    e_mid = _energy(y_midpoint).item()
    e_euler = _energy(y_euler).item()

    drift_mid = abs(e_mid - e0_mid) / max(abs(e0_mid), 1.0e-30)
    drift_euler = abs(e_euler - e0_euler) / max(abs(e0_euler), 1.0e-30)

    assert drift_mid < 1.0e-10
    assert drift_euler > 1.0e-2
