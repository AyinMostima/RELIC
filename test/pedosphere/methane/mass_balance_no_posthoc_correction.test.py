import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.methane.budget_coupler import MethaneBudgetCoupler
from relic.utils.physical_constants import get_physical_constant


CANONICAL_INPUTS = {
    "ch4_production_rate",
    "ch4_oxidation_rate",
    "ch4_transport_loss_rate",
    "ch4_storage_tendency_prod",
    "ch4_storage_tendency_oxid",
    "ch4_storage_tendency_trans",
    "ch4_soil_storage",
}

CANONICAL_OUTPUTS = {
    "ch4_atmospheric_flux",
    "ch4_soil_storage",
    "diagnostics.methane_budget.atmospheric_ch4_mass_flux",
    "diagnostics.methane_budget.mass_balance_residual",
    "diagnostics.methane_budget.storage_tendency_net",
    "diagnostics.methane_budget.transport_limiter_residual",
}


def _state(**overrides):
    state = {
        "ch4_production_rate": torch.tensor(0.0),
        "ch4_oxidation_rate": torch.tensor(0.005),
        "ch4_transport_loss_rate": torch.tensor(0.01),
        "ch4_storage_tendency_prod": torch.tensor(0.0),
        "ch4_storage_tendency_oxid": torch.tensor(-0.005),
        "ch4_storage_tendency_trans": torch.tensor(-0.01),
        "ch4_soil_storage": torch.tensor(1.0),
    }
    state.update(overrides)
    return state


def test_budget_declares_exact_canonical_interface() -> None:
    budget = MethaneBudgetCoupler()

    assert set(budget.metadata.input_variables) == CANONICAL_INPUTS
    assert set(budget.metadata.output_variables) == CANONICAL_OUTPUTS
    assert budget.get_dependencies() == [
        "pedosphere_methane_production",
        "pedosphere_methane_oxidation",
        "pedosphere_methane_transport",
    ]


def test_budget_applies_transport_last_storage_limiter_with_exact_closure() -> None:
    budget = MethaneBudgetCoupler()
    state = _state()
    original = {key: value.clone() for key, value in state.items()}

    updates = budget.forward(state, time=0.0, dt=torch.tensor(100.0))

    assert set(updates) == CANONICAL_OUTPUTS
    assert all(isinstance(item, UpdateItem) for item in updates.values())
    for key, value in original.items():
        torch.testing.assert_close(state[key], value, rtol=0.0, atol=0.0)

    storage = updates["ch4_soil_storage"]
    assert storage.mode == UpdateMode.ACCUMULATE
    assert storage.per_timestep is True
    assert storage.unit == "g C m-2"
    assert storage.element == "C"
    assert storage.source_functor == "pedosphere_methane_budget"
    torch.testing.assert_close(storage.value, torch.tensor(-1.0), rtol=0.0, atol=0.0)

    atmospheric = updates["ch4_atmospheric_flux"]
    assert atmospheric.mode == UpdateMode.REPLACE
    assert atmospheric.per_timestep is False
    assert atmospheric.unit == "g C m-2 s-1"
    assert atmospheric.element == "C"
    assert atmospheric.source_functor == "pedosphere_methane_budget"
    torch.testing.assert_close(
        atmospheric.value,
        torch.tensor(0.005),
        rtol=1.0e-6,
        atol=1.0e-9,
    )

    limiter = updates[
        "diagnostics.methane_budget.transport_limiter_residual"
    ]
    torch.testing.assert_close(
        limiter.value,
        torch.tensor(0.005),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    storage_rate = updates[
        "diagnostics.methane_budget.storage_tendency_net"
    ].value
    torch.testing.assert_close(
        state["ch4_production_rate"]
        - state["ch4_oxidation_rate"]
        - atmospheric.value,
        storage_rate,
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        updates["diagnostics.methane_budget.mass_balance_residual"].value,
        torch.tensor(0.0),
        rtol=0.0,
        atol=1.0e-9,
    )

    methane_mass = float(
        get_physical_constant("relic.physical.constants.molecular_weights.ch4")
    )
    carbon_mass = float(
        get_physical_constant("relic.physical.constants.molecular_weights.carbon")
    )
    torch.testing.assert_close(
        updates["diagnostics.methane_budget.atmospheric_ch4_mass_flux"].value,
        atmospheric.value * (methane_mass / carbon_mass),
        rtol=1.0e-6,
        atol=1.0e-9,
    )


def test_budget_residual_does_not_posthoc_correct_storage() -> None:
    budget = MethaneBudgetCoupler()
    state = _state(
        ch4_production_rate=torch.tensor(1.0),
        ch4_oxidation_rate=torch.tensor(0.0),
        ch4_transport_loss_rate=torch.tensor(0.0),
        ch4_storage_tendency_prod=torch.tensor(0.0),
        ch4_storage_tendency_oxid=torch.tensor(0.0),
        ch4_storage_tendency_trans=torch.tensor(0.0),
        ch4_soil_storage=torch.tensor(2.0),
    )

    updates = budget.forward(state, time=0.0, dt=torch.tensor(600.0))

    torch.testing.assert_close(
        updates["ch4_soil_storage"].value,
        torch.tensor(0.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        updates["diagnostics.methane_budget.mass_balance_residual"].value,
        torch.tensor(1.0),
        rtol=0.0,
        atol=0.0,
    )


def test_budget_rejects_limiter_that_cannot_route_to_transport() -> None:
    budget = MethaneBudgetCoupler()
    state = _state(
        ch4_oxidation_rate=torch.tensor(0.0),
        ch4_transport_loss_rate=torch.tensor(0.001),
        ch4_storage_tendency_oxid=torch.tensor(0.0),
        ch4_storage_tendency_trans=torch.tensor(-0.02),
    )

    with pytest.raises(ValueError, match="cannot be routed entirely"):
        budget.forward(state, time=0.0, dt=torch.tensor(100.0))


@pytest.mark.parametrize(
    ("key", "value", "error_type"),
    [
        ("ch4_production_rate", None, KeyError),
        ("ch4_transport_loss_rate", 0.0, TypeError),
        ("ch4_oxidation_rate", torch.tensor(float("nan")), ValueError),
        ("ch4_soil_storage", torch.tensor(-1.0), ValueError),
    ],
)
def test_budget_required_inputs_fail_fast(key, value, error_type) -> None:
    budget = MethaneBudgetCoupler()
    state = _state()
    if value is None:
        state.pop(key)
    else:
        state[key] = value

    with pytest.raises(error_type):
        budget.forward(state, time=0.0, dt=torch.tensor(100.0))


@pytest.mark.parametrize(
    "dt",
    [torch.tensor(0.0), torch.tensor(float("nan")), torch.tensor([1.0, 2.0])],
)
def test_budget_invalid_dt_fails_fast(dt) -> None:
    budget = MethaneBudgetCoupler()

    with pytest.raises(ValueError):
        budget.forward(_state(), time=0.0, dt=dt)


def test_budget_conservation_check_uses_full_vector() -> None:
    budget = MethaneBudgetCoupler()
    state = _state(
        ch4_production_rate=torch.tensor([1.0, 1.0]),
        ch4_oxidation_rate=torch.tensor([0.0, 0.0]),
        ch4_transport_loss_rate=torch.tensor([0.0, 0.0]),
        ch4_storage_tendency_prod=torch.tensor([1.0, 0.0]),
        ch4_storage_tendency_oxid=torch.tensor([0.0, 0.0]),
        ch4_storage_tendency_trans=torch.tensor([0.0, 0.0]),
        ch4_soil_storage=torch.tensor([2.0, 2.0]),
    )

    updates = budget.forward(state, time=0.0, dt=torch.tensor(1.0))

    assert budget.check_conservation_laws(state, updates) == {"mass": 1.0}
