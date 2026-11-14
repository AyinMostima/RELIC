"""I3 conservation invariant tests with PBT.

References:
- Olson (1963) mass-balance framing for ecosystem pools.
- RELIC StateManager transactional updates and conservation checks.

Mathematical expression:
- For a closed transfer between two conserved pools A and B:
  Delta(A) + Delta(B) = 0

Parameter meanings:
- initial_a, initial_b: initial mass of two conserved pools.
- transfer: amount transferred from A to B.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator
from relic.scientific.anthroposphere.management.fertilization import FertilizationManagement
from relic.scientific.anthroposphere.management.grazing import GrazingManagement
from relic.scientific.anthroposphere.management.harvest import HarvestManagement
from relic.scientific.anthroposphere.management.tillage import TillageManagement


def _make_manager(initial_a: float, initial_b: float) -> StateManager:
    """Create a two-pool conserved state manager."""
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="carbon_pool_a",
            units="kg C m-2",
            conserved=True,
            default_value=initial_a,
        )
    )
    sm.register_variable(
        StateVariableSpec(
            name="carbon_pool_b",
            units="kg C m-2",
            conserved=True,
            default_value=initial_b,
        )
    )
    return sm


@pytest.mark.property
class TestConservationPBT:
    """I3 zero-net-change invariants for closed transfers."""

    @given(
        initial_a=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        initial_b=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ratio=st.floats(min_value=1e-6, max_value=0.9, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_zero_sum_transfer(self, initial_a: float, initial_b: float, ratio: float) -> None:
        """Closed transfer must preserve total carbon."""
        transfer = initial_a * ratio
        sm = _make_manager(initial_a, initial_b)
        before_state = sm.get_state()
        before_a = float(before_state["carbon_pool_a"].item())
        before_b = float(before_state["carbon_pool_b"].item())

        updates = {
            "carbon_pool_a": UpdateItem(
                value=-transfer,
                mode=UpdateMode.ACCUMULATE,
                unit="kg C m-2",
                basis="per_step",
                element="C",
            ),
            "carbon_pool_b": UpdateItem(
                value=transfer,
                mode=UpdateMode.ACCUMULATE,
                unit="kg C m-2",
                basis="per_step",
                element="C",
            ),
        }
        sm.apply_updates(updates, check_conservation=True)

        after_state = sm.get_state()
        after_a = float(after_state["carbon_pool_a"].item())
        after_b = float(after_state["carbon_pool_b"].item())
        delta_total = (after_a + after_b) - (before_a + before_b)
        tol = max(1e-10, 1e-10 * abs(before_a + before_b))
        assert abs(delta_total) <= tol


def _assert_non_negative_finite(error_tensor: object) -> None:
    if isinstance(error_tensor, UpdateItem):
        error_tensor = error_tensor.value
    assert isinstance(error_tensor, torch.Tensor)
    assert torch.all(torch.isfinite(error_tensor))
    assert torch.all(error_tensor >= 0)


def test_management_tillage_emits_conservation_diagnostics() -> None:
    manager = TillageManagement()
    state = {
        "soil_moisture": torch.tensor(0.25),
        # Tilled-layer soil temperature [K], the required tillage input (no default).
        "soil_temperature": torch.tensor(293.15),
        "soil_organic_carbon": torch.tensor(5.0),
        "days_since_tillage": torch.tensor(20.0),
    }
    outputs = manager.forward(
        state=state,
        time=0.0,
        dt=3600.0,
        calendar_type="noleap",
        day_of_year=120,
    )
    assert "diagnostics.tillage_mass_balance_error" in outputs
    _assert_non_negative_finite(outputs["diagnostics.tillage_mass_balance_error"])


def test_management_fertilization_emits_conservation_diagnostics() -> None:
    manager = FertilizationManagement()
    state = {
        "soil_nitrogen": torch.tensor(0.2),
        "soil_phosphorus": torch.tensor(0.05),
        "soil_ph": torch.tensor(6.8),
        "soil_organic_carbon": torch.tensor(0.02),
        "soil_moisture": torch.tensor(0.3),
        "soil_no3": torch.tensor(0.01),
        "soil_temperature": torch.tensor(293.15),
        "fertilizer_application_rate": torch.tensor(2.0e-4),
    }
    outputs = manager.forward(
        state=state,
        time=0.0,
        dt=3600.0,
        calendar_type="noleap",
        day_of_year=120,
    )
    diagnostic_names = (
        "diagnostics.fertilization.nitrogen_boundary_residual",
        "diagnostics.fertilization.phosphorus_boundary_residual",
    )
    for name in diagnostic_names:
        assert name in outputs
        _assert_non_negative_finite(outputs[name])


def test_management_harvest_emits_conservation_diagnostics() -> None:
    manager = HarvestManagement()
    outputs = manager.forward(
        state={
            "aboveground_biomass": torch.tensor(2.5),
            "belowground_biomass": torch.tensor(0.2),
            "residue_cover": torch.tensor(0.0),
        },
        time=0.0,
        dt=3600.0,
        calendar_type="noleap",
        day_of_year=250,
    )
    diagnostic_names = (
        "diagnostics.harvest.biomass_partition_residual",
        "diagnostics.harvest.carbon_ledger_residual",
        "diagnostics.harvest.nitrogen_ledger_residual",
    )
    for name in diagnostic_names:
        assert name in outputs
        _assert_non_negative_finite(outputs[name])


def test_management_grazing_closes_open_matter_boundary() -> None:
    manager = GrazingManagement()
    state = {
        "aboveground_biomass": torch.tensor(1.8),
        "vegetation_quality": torch.tensor(0.9),
        "soil_moisture": torch.tensor(0.3),
    }
    outputs = manager.forward(
        state=state,
        time=0.0,
        dt=3600.0,
        calendar_type="noleap",
        day_of_year=180,
    )
    assert manager.check_conservation_laws(state, outputs)[
        "grazing_matter_boundary"
    ] <= torch.finfo(torch.float32).eps
    assert outputs["aboveground_biomass"].mode is UpdateMode.ACCUMULATE
    assert torch.allclose(
        outputs["aboveground_biomass"].value
        + outputs["cumulative_intake"].value,
        torch.zeros_like(outputs["aboveground_biomass"].value),
    )


def test_management_coordinator_emits_conservation_diagnostics() -> None:
    manager = ManagementCoordinator()
    state = {
        "soil_moisture": torch.tensor([0.25]),
        "soil_nitrogen": torch.tensor([0.01]),
        "soil_organic_carbon": torch.tensor([5.0]),
        "crop_maturity": torch.tensor([0.7]),
        "aboveground_biomass": torch.tensor([2.0]),
        "air_temperature": torch.tensor([293.15]),
        "day_of_year": torch.tensor(150.0),
        "irrigation_flux": torch.tensor([0.0]),
        "harvest_flux": torch.tensor([0.0]),
        "grazing_flux": torch.tensor([0.0]),
        "co2_emission": torch.tensor([0.0]),
        "field_capacity": torch.tensor([0.35]),
        "wilting_point": torch.tensor([0.15]),
        "evapotranspiration_flux": torch.tensor([3.0e-6]),
        "precipitation_flux": torch.tensor([0.0]),
    }
    outputs = manager.forward(
        state=state,
        time=86400.0,
        dt=3600.0,
        calendar_type="noleap",
    )
    assert "diagnostics.coordinator_mass_balance_error" in outputs
    item = outputs["diagnostics.coordinator_mass_balance_error"]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    _assert_non_negative_finite(item.value)
