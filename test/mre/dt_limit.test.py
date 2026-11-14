import pytest
import torch

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def test_dt_required_for_per_second_increment(dt_limit_cfg) -> None:
    var_name = str(dt_limit_cfg.get("var_name", "mre_state"))
    units = str(dt_limit_cfg.get("units", "kg"))
    value = float(dt_limit_cfg.get("value", 1.0))
    basis = str(dt_limit_cfg.get("basis", "per_second"))
    dt = dt_limit_cfg.get("dt", None)

    spec = StateVariableSpec(name=var_name, units=units, conserved=False)
    sm = StateManager(schema={var_name: spec})
    sm.set_state({var_name: torch.tensor(0.0)})

    increment = UpdateItem(value=value, mode=UpdateMode.ACCUMULATE, basis=basis)

    # References: explicit time integration in mass balance models.
    # Mathematical expression: delta = rate * dt
    # Parameters: rate (value), dt (time step)
    with pytest.raises(ValueError, match="dt is required for an unintegrated state rate"):
        sm.apply_increment(var_name, increment, dt=dt)
