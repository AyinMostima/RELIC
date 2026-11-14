"""
Core smoke test for RELICSimulator and Functor bridges.

Policy:
- ASCII-only comments and prints.
- Windows path assumptions for developer instructions.

This test verifies that:
1) Core modules import without error.
2) A minimal BiologicalProcessFunctor can be registered and executed.
3) The simulator main loop applies functor increments through StateManager.
"""

import torch

from relic.core import (
    RELICSimulator,
    BiologicalProcessFunctor,
    FunctorMetadata,
    ProcessCategory,
)


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'x': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class AddOneFunctor(BiologicalProcessFunctor):
    """Simple functor that increments state variable 'x' by +1 per step.

    Notes
    -----
    - Returns increments (not rates). StateManager applies u directly.
    - No scientific process here; pure smoke to validate wiring.
    """

    def __init__(self, name: str = "add_one") -> None:
        meta = FunctorMetadata(
            name=name,
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            description="Smoke test functor that adds +1 per step to x.",
            mathematical_expression="x_{n+1} = x_n + 1",
            references=["N/A - smoke test"],
            parameters={},
            input_variables=["x"],
            output_variables=["x"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale="second",
            coupling_interfaces=[],
        )
        super().__init__(name=name, metadata=meta, config={})

    def get_dependencies(self):
        return []

    def forward(self, state, time: float, dt: float, **kwargs):  # noqa: D401
        # Return increment of +1.0 for variable 'x'.
        val = state.get("x", torch.tensor(0.0))
        return {"x": torch.ones_like(val)}


def test_core_smoke_step_and_run():
    # Create simulator with small time step
    sim = _declared_simulator({"time_step": 1.0, "max_time": 10.0, "output_frequency": 999999})

    # Initialize state with a simple scalar variable
    sim.initialize_state({"x": torch.tensor(0.0)})

    # Register the simple functor in fast group (exec each step)
    sim.register_functor(AddOneFunctor("add_one"), execution_group="fast")

    # Run for 5 steps and validate value increased by 5.0
    results = sim.run(start_time=0.0, end_time=5.0, dt=1.0)
    # Simulator.run returns current state dict
    assert "x" in results
    x_val = float(results["x"].item())
    assert abs(x_val - 5.0) < 1e-8

    # Do not step after run; simulator transitions to COMPLETED state.
    # Validate that final state meets expected value and run completed.
    assert sim.state.name == "COMPLETED"

