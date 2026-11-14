import torch

import torch

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def test_update_mode_accumulate_then_replace(update_mode_cfg) -> None:
    var_name = str(update_mode_cfg.get("var_name", "mode_state"))
    units = str(update_mode_cfg.get("units", "kg"))
    initial_value = float(update_mode_cfg.get("initial_value", 10.0))
    accumulate_value = float(update_mode_cfg.get("accumulate_value", 5.0))
    replace_value = float(update_mode_cfg.get("replace_value", 7.0))

    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name=var_name,
            units=units,
            conserved=False,
            default_value=initial_value,
        )
    )

    # References: mass balance update rules for state variables.
    # Mathematical expression: accumulate -> x_new = x_old + delta; replace -> x_new = value
    # Parameters: x_old (state), delta (accumulate_value), value (replace_value)
    sm.apply_updates(
        {var_name: UpdateItem(value=accumulate_value, mode=UpdateMode.ACCUMULATE)},
        dt=None,
        check_conservation=False,
    )
    assert torch.isclose(sm.state_vars[var_name], torch.tensor(initial_value + accumulate_value))

    sm.apply_updates(
        {var_name: UpdateItem(value=replace_value, mode=UpdateMode.REPLACE)},
        dt=None,
        check_conservation=False,
    )
    assert torch.isclose(sm.state_vars[var_name], torch.tensor(replace_value))
