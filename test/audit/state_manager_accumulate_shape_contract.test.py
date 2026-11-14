import pytest
import torch

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def test_accumulate_rejects_shape_broadcast() -> None:
    schema = {
        "test_accumulated_state": StateVariableSpec(
            name="test_accumulated_state",
            units="dimensionless",
            required=False,
            conserved=False,
        )
    }
    manager = StateManager(schema=schema)
    manager.set_state(
        {
            "test_accumulated_state": torch.tensor(
                [0.4, 0.35, 0.25], dtype=torch.float32
            )
        }
    )

    with pytest.raises(ValueError, match="shape contract violation"):
        manager.apply_updates(
            {
                "test_accumulated_state": UpdateItem(
                    value=torch.tensor(0.1, dtype=torch.float32),
                    mode=UpdateMode.ACCUMULATE,
                )
            },
            dt=60.0,
            check_conservation=False,
        )
