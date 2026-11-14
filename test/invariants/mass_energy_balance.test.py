import torch

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
from relic.validation.framework.models import ValidationCode, ValidationContext
from relic.validation.framework.rules.conservation import (
    validate_energy_balance_conservation,
)


def test_mass_and_energy_balance_stepwise() -> None:
    carbon_unit = "g C m-2"
    sm = StateManager(autoload_contract_registry=False)
    sm.register_variable(
        StateVariableSpec(
            name="carbon_pool",
            units=carbon_unit,
            conserved=True,
            default_value=100.0,
            accumulates="carbon.total",
        )
    )
    sm.register_variable(
        StateVariableSpec(
            name="carbon_reserve",
            units=carbon_unit,
            conserved=True,
            default_value=0.0,
            accumulates="carbon.total",
        )
    )
    sm.set_conservation_ledger(
        {"carbon": ["carbon_pool", "carbon_reserve"]}
    )

    # References: E3SM/ELM discrete elemental budget diagnostics.
    # Mathematical expression: residual_C = delta_pool + delta_reserve = 1 - 1 = 0.
    # Parameters: both terms are integrated carbon storage changes in g C m-2.
    carbon_updates = {
        "carbon_pool": UpdateItem(
            value=1.0,
            mode=UpdateMode.ACCUMULATE,
            unit=carbon_unit,
            per_timestep=True,
            element="C",
            source_functor="mass_energy_balance_test",
        ),
        "carbon_reserve": UpdateItem(
            value=-1.0,
            mode=UpdateMode.ACCUMULATE,
            unit=carbon_unit,
            per_timestep=True,
            element="C",
            source_functor="mass_energy_balance_test",
        ),
    }

    summary = sm.generate_conservation_report(
        updates=carbon_updates,
        current_time=0.0,
        store_history=False,
    )
    assert abs(summary.balance_error.get("C", 0.0)) <= 1e-12

    # Reference: surface energy balance, Rn = H + LE + G.
    # Expression: residual_E = Rn - H - LE - G = 100 - 30 - 50 - 20 = 0.
    # Parameters: Rn, H, LE, and G are surface energy fluxes in W m-2.
    energy_context = ValidationContext(
        metadata={
            "energy_fluxes": {
                "net_radiation": torch.tensor(100.0),
                "sensible_heat_flux": torch.tensor(30.0),
                "latent_heat_flux": torch.tensor(50.0),
                "ground_heat_flux": torch.tensor(20.0),
            }
        },
        tolerance_overrides={},
    )
    energy_results = validate_energy_balance_conservation(energy_context)
    energy_balance = next(
        result
        for result in energy_results
        if result.code == ValidationCode.ENERGY_BALANCE_OK
    )
    assert energy_balance.metrics["energy_balance_residual_max"] == 0.0
