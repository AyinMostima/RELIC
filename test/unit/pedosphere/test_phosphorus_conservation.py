"""
Canonical phosphorus conservation tests through StateManager and SCM.
"""

import pytest
import torch

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def _build_phosphorus_manager() -> StateManager:
    schema = {
        "available_phosphorus": StateVariableSpec(
            name="available_phosphorus",
            units="g P m-2",
            conserved=True,
            required=False,
        ),
        "plant_p_storage": StateVariableSpec(
            name="plant_p_storage",
            units="g P m-2",
            conserved=True,
            required=False,
        ),
        "fertilizer_p_flux": StateVariableSpec(
            name="fertilizer_p_flux",
            units="g P m-2 s-1",
            conserved=True,
            required=False,
        ),
    }
    manager = StateManager(schema=schema, device="cpu")
    manager.set_state(
        {
            "available_phosphorus": torch.tensor(1.0, dtype=torch.float64),
            "plant_p_storage": torch.tensor(0.0, dtype=torch.float64),
            "fertilizer_p_flux": torch.tensor(0.0, dtype=torch.float64),
        },
        validate=False,
    )
    return manager


def test_open_phosphorus_transaction_uses_scm_and_dt_once() -> None:
    """A fertilizer rate must balance one integrated storage increment."""
    manager = _build_phosphorus_manager()
    dt = 86400.0
    phosphorus_input = torch.tensor(0.872, dtype=torch.float64)
    phosphorus_flux = phosphorus_input / dt
    updates = {
        "available_phosphorus": UpdateItem(
            value=phosphorus_input,
            mode=UpdateMode.ACCUMULATE,
            unit="g P m-2",
            per_timestep=True,
            element="P",
            source_functor="fertilization_management",
        ),
        "fertilizer_p_flux": UpdateItem(
            value=phosphorus_flux,
            mode=UpdateMode.REPLACE,
            unit="g P m-2 s-1",
            per_timestep=False,
            element=None,
            source_functor="fertilization_management",
        ),
    }

    report = manager.generate_conservation_report(
        updates,
        dt=dt,
        store_history=False,
    )
    assert report.phosphorus_report is not None
    assert report.phosphorus_report.balance_error == pytest.approx(0.0, abs=1.0e-12)
    guardian = report.custom_checks["conservation_guardian"]["phosphorus.total"]
    assert guardian["passed"] is True
    assert guardian["residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert guardian["sum_flux_in"] == pytest.approx(
        float(phosphorus_flux),
        abs=1.0e-15,
    )
    assert guardian["delta_storage"] == pytest.approx(0.872, abs=1.0e-12)

    manager.apply_updates(
        updates,
        dt=dt,
        check_conservation=True,
        rollback_on_failure=True,
        update_sources={
            "available_phosphorus": "fertilization_management",
            "fertilizer_p_flux": "fertilization_management",
        },
        update_increments={
            "available_phosphorus": {
                "fertilization_management": phosphorus_input,
            }
        },
    )

    torch.testing.assert_close(
        manager.get_state("available_phosphorus"),
        torch.tensor(1.872, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_unbalanced_phosphorus_update_rolls_back() -> None:
    """The canonical owner must reject and roll back an unbalanced P update."""
    manager = _build_phosphorus_manager()
    increment = torch.tensor(0.25, dtype=torch.float64)

    with pytest.raises(ValueError, match="Conservation violation for phosphorus"):
        manager.apply_updates(
            {
                "available_phosphorus": UpdateItem(
                    value=increment,
                    mode=UpdateMode.ACCUMULATE,
                    unit="g P m-2",
                    per_timestep=True,
                    element="P",
                    source_functor="fertilization_management",
                )
            },
            dt=1.0,
            check_conservation=True,
            rollback_on_failure=True,
            update_sources={
                "available_phosphorus": "fertilization_management",
            },
            update_increments={
                "available_phosphorus": {
                    "fertilization_management": increment,
                }
            },
        )

    torch.testing.assert_close(
        manager.get_state("available_phosphorus"),
        torch.tensor(1.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
