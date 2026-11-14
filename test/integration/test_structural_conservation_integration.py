import torch

from relic.computation.operators.functional import apply_advection
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
)
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateMode


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'tracer': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class _ConservativeAdvection2D(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str, dx: float, dy: float, u_value: float, v_value: float):
        metadata = FunctorMetadata(
            name=name,
            version="0.0.0",
            description="2D conservative advection for structural conservation integration test.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="dC/dt = -d(uC)/dx - d(vC)/dy",
            references=[
                "LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems.",
            ],
            input_variables=["tracer"],
            output_variables=["tracer"],
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.dx = float(dx)
        self.dy = float(dy)
        self.u_value = float(u_value)
        self.v_value = float(v_value)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        tracer = state["tracer"]
        u = torch.full_like(tracer, self.u_value)
        v = torch.full_like(tracer, self.v_value)
        dt_value = float(dt.item()) if isinstance(dt, torch.Tensor) else float(dt)

        adv_result = apply_advection(
            concentration=tracer,
            velocity=(u, v),
            grid_spacing=(self.dy, self.dx),
            spatial_dims=2,
            scheme="upwind",
            boundary_method="periodic",
            time_step=dt_value,
            check_cfl=True,
            time_integration_method=kwargs.get("time_integration_method"),
            implicit_midpoint_max_iter=kwargs.get("implicit_midpoint_max_iter"),
            implicit_midpoint_tolerance=kwargs.get("implicit_midpoint_tolerance"),
        )
        delta = adv_result["concentration"] - tracer
        return {"tracer": self.wrap_update(delta, mode=UpdateMode.ACCUMULATE)}


def _gaussian_2d(nx: int, ny: int, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    y = torch.linspace(0.0, 1.0, ny, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sigma2 = 0.01
    center_x, center_y = 0.5, 0.5
    return torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma2))


def test_structural_conservation_integration_with_simulator_time_method():
    dtype = torch.float64
    nx = ny = 32
    dx = dy = 1.0 / nx
    dt = 0.005
    n_steps = 200

    initial_tracer = _gaussian_2d(nx, ny, dtype=dtype).unsqueeze(0)
    initial_mass = initial_tracer.sum().item()

    sim = _declared_simulator(
        {
            "time_step": dt,
            "max_time": n_steps * dt,
            "conservation_checks": False,
            "strict_scale_detection": False,
            "time_integration_method": "implicit_midpoint",
            "implicit_midpoint_max_iter": 8,
            "implicit_midpoint_tolerance": 1.0e-12,
        }
    )
    sim.initialize_state({"tracer": initial_tracer.clone()})
    sim.register_functor(
        _ConservativeAdvection2D(
            name="conservative_advection_2d",
            dx=dx,
            dy=dy,
            u_value=1.0,
            v_value=0.5,
        ),
        execution_group="fast",
    )

    for _ in range(n_steps):
        sim.step(dt=dt)

    final_tracer = sim.state_manager.get_state("tracer")
    final_mass = final_tracer.sum().item()
    relative_error = abs(final_mass - initial_mass) / max(abs(initial_mass), 1.0e-30)

    assert relative_error < 1.0e-10
