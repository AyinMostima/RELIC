"""L4 reference trajectories for the canonical fertilization boundary."""

from __future__ import annotations

import pytest
import torch

from relic.core.state import UpdateItem
from relic.scientific.anthroposphere.management.fertilization import (
    FertilizationManagement,
    FertilizerEvent,
    FertilizerType,
)

EVENT_DAY = 100
EVENT_DURATION = 3600.0
STEP_SECONDS = 900.0
KG_TO_G = 1000.0


def _value(item: UpdateItem | torch.Tensor) -> torch.Tensor:
    return item.value if isinstance(item, UpdateItem) else item


def _manager(
    fertilizer_type: FertilizerType,
    product_mass: float | torch.Tensor,
) -> FertilizationManagement:
    return FertilizationManagement(
        config={"calendar_type": "noleap"},
        events=(
            FertilizerEvent(
                fertilizer_type=fertilizer_type,
                product_mass=product_mass,
                application_date=EVENT_DAY,
                application_duration=EVENT_DURATION,
            ),
        )
    )


# References:
#   - FAO (2000), Fertilizers and Their Use, 4th edition.
# Mathematical expression:
#   F_N = A f_N omega / (tau dt).
#   dF_N/dA = f_N omega / (tau dt).
# Parameter meanings:
#   A: fertilizer product mass [kg product m-2].
#   f_N: elemental nitrogen mass fraction [kg N kg-product-1].
#   omega: step/event overlap [s].
#   tau: event duration [s].
#   dt: timestep [s].
def test_nitrogen_flux_matches_analytical_product_mass_gradient() -> None:
    product_mass = torch.tensor(0.012, dtype=torch.float64, requires_grad=True)
    manager = _manager(FertilizerType.UREA, product_mass)
    nitrogen_fraction = manager.get_fertilizer_properties(
        FertilizerType.UREA
    )["nitrogen_mass_fraction"]

    output = manager.forward(
        {},
        time=STEP_SECONDS,
        dt=STEP_SECONDS,
        day_of_year=EVENT_DAY,
    )
    nitrogen_flux = _value(output["fertilizer_n_flux"])
    gradient = torch.autograd.grad(nitrogen_flux, product_mass)[0]

    expected_flux = product_mass.detach() * nitrogen_fraction / EVENT_DURATION
    expected_gradient = nitrogen_fraction / EVENT_DURATION
    assert nitrogen_flux.dtype == torch.float64
    assert nitrogen_flux.detach().item() == pytest.approx(
        expected_flux.item(),
        rel=1.0e-12,
    )
    assert gradient.detach().item() == pytest.approx(
        expected_gradient,
        rel=1.0e-12,
    )


# References:
#   - FAO (2000), Fertilizers and Their Use, 4th edition.
#   - BIPM (2019), SI Brochure, 9th edition.
# Mathematical expression:
#   Delta P = 1000 A f_P omega / tau.
#   d(Delta P)/dA = 1000 f_P omega / tau.
# Parameter meanings:
#   f_P: elemental phosphorus mass fraction [kg P kg-product-1].
#   1000: exact conversion from kg to g [g kg-1].
def test_phosphorus_increment_matches_analytical_product_mass_gradient() -> None:
    product_mass = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    manager = _manager(FertilizerType.SUPERPHOSPHATE, product_mass)
    phosphorus_fraction = manager.get_fertilizer_properties(
        FertilizerType.SUPERPHOSPHATE
    )["elemental_phosphorus_mass_fraction"]

    output = manager.forward(
        {},
        time=0.0,
        dt=STEP_SECONDS,
        day_of_year=EVENT_DAY,
    )
    phosphorus_increment = _value(output["available_phosphorus"])
    gradient = torch.autograd.grad(phosphorus_increment, product_mass)[0]

    active_fraction = STEP_SECONDS / EVENT_DURATION
    expected_increment = (
        product_mass.detach() * phosphorus_fraction * active_fraction * KG_TO_G
    )
    expected_gradient = phosphorus_fraction * active_fraction * KG_TO_G
    assert phosphorus_increment.dtype == torch.float64
    assert phosphorus_increment.detach().item() == pytest.approx(
        expected_increment.item(),
        rel=1.0e-12,
    )
    assert gradient.detach().item() == pytest.approx(
        expected_gradient,
        rel=1.0e-12,
    )


# References:
#   - CLM/CTSM FERT_TO_SMINN and NFERTILIZATION boundary conventions.
# Mathematical expression:
#   sum_k(F_X,k dt_k) = A f_X and r_X = Delta M_X - F_X dt = 0.
# Parameter meanings:
#   F_X,k: step-mean elemental boundary flux.
#   dt_k: timestep duration.
#   r_X: local open-boundary closure residual.
def test_partitioned_event_integrates_to_exact_elemental_mass_and_closes() -> None:
    nitrogen_manager = _manager(FertilizerType.UREA, 0.012)
    phosphorus_manager = _manager(FertilizerType.SUPERPHOSPHATE, 0.02)
    nitrogen_fraction = nitrogen_manager.get_fertilizer_properties(
        FertilizerType.UREA
    )["nitrogen_mass_fraction"]
    phosphorus_fraction = phosphorus_manager.get_fertilizer_properties(
        FertilizerType.SUPERPHOSPHATE
    )["elemental_phosphorus_mass_fraction"]

    integrated_nitrogen = torch.zeros((), dtype=torch.get_default_dtype())
    integrated_phosphorus = torch.zeros((), dtype=torch.get_default_dtype())
    for step in range(4):
        time = step * STEP_SECONDS
        nitrogen_output = nitrogen_manager.forward(
            {},
            time=time,
            dt=STEP_SECONDS,
            day_of_year=EVENT_DAY,
        )
        phosphorus_output = phosphorus_manager.forward(
            {},
            time=time,
            dt=STEP_SECONDS,
            day_of_year=EVENT_DAY,
        )
        integrated_nitrogen = integrated_nitrogen + (
            _value(nitrogen_output["fertilizer_n_flux"]) * STEP_SECONDS
        )
        integrated_phosphorus = integrated_phosphorus + _value(
            phosphorus_output["available_phosphorus"]
        )
        assert nitrogen_output[
            "diagnostics.fertilization.nitrogen_boundary_residual"
        ] == pytest.approx(0.0, abs=1.0e-15)
        assert phosphorus_output[
            "diagnostics.fertilization.phosphorus_boundary_residual"
        ] == pytest.approx(0.0, abs=1.0e-12)

    assert integrated_nitrogen == pytest.approx(
        0.012 * nitrogen_fraction,
        rel=1.0e-7,
    )
    assert integrated_phosphorus == pytest.approx(
        0.02 * phosphorus_fraction * KG_TO_G,
        rel=1.0e-7,
    )


# References:
#   - BIPM (2019), SI Brochure, 9th edition.
# Mathematical expression:
#   sum_{k=0}^{3} Delta M_k =
#   sum_{k=0}^{1} Delta M_k + sum_{k=2}^{3} Delta M_k.
# Parameter meanings:
#   Delta M_k: applied elemental mass in timestep k.
def test_mid_event_reconstruction_preserves_remaining_reference_trajectory() -> None:
    uninterrupted = _manager(FertilizerType.UREA, 0.012)
    before_restart = _manager(FertilizerType.UREA, 0.012)

    uninterrupted_fluxes = []
    restarted_fluxes = []
    for step in range(4):
        output = uninterrupted.forward(
            {},
            time=step * STEP_SECONDS,
            dt=STEP_SECONDS,
            day_of_year=EVENT_DAY,
        )
        uninterrupted_fluxes.append(_value(output["fertilizer_n_flux"]))

    for step in range(2):
        output = before_restart.forward(
            {},
            time=step * STEP_SECONDS,
            dt=STEP_SECONDS,
            day_of_year=EVENT_DAY,
        )
        restarted_fluxes.append(_value(output["fertilizer_n_flux"]))

    checkpoint = before_restart.state_dict()
    after_restart = _manager(FertilizerType.UREA, 0.012)
    after_restart.load_state_dict(checkpoint)
    for step in range(2, 4):
        output = after_restart.forward(
            {},
            time=step * STEP_SECONDS,
            dt=STEP_SECONDS,
            day_of_year=EVENT_DAY,
        )
        restarted_fluxes.append(_value(output["fertilizer_n_flux"]))

    assert torch.stack(restarted_fluxes) == pytest.approx(
        torch.stack(uninterrupted_fluxes),
        rel=1.0e-7,
        abs=0.0,
    )
