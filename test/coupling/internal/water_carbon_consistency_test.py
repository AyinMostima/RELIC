"""Owner tests for the WaterCarbonCoupling modifier-consistency contract (ledger D66/D70).

The coupling emits dimensionless multiplicative modifiers only; it advances no mass. Its local
check therefore verifies the definitional consistency between emitted outputs and their
declared bounds instead of comparing a tensor with itself, and the two temperature usages
(capped f_T for the bounded heterotrophic modifier, uncapped f_T for the Q10 decomposition
multiplier) are exposed so that the check is falsifiable.
"""

from __future__ import annotations

import pytest
import torch

from relic.coupling.internal.water_carbon import WaterCarbonCoupling


def _build_functor() -> WaterCarbonCoupling:
    functor = WaterCarbonCoupling(name="water-carbon-consistency", config={})
    functor._get_cache()
    return functor


def _state(soil_moisture: float, soil_temperature_k: float) -> dict:
    return {
        "soil_moisture": torch.tensor(soil_moisture, dtype=torch.float32),
        "soil_temperature": torch.tensor(soil_temperature_k, dtype=torch.float32),
        "vapor_pressure_deficit": torch.tensor(800.0, dtype=torch.float32),
        "heterotrophic_respiration": torch.tensor(2.0e-7, dtype=torch.float32),
    }


def test_metadata_declares_no_mass_law_for_a_modifier_only_coupling() -> None:
    functor = _build_functor()
    # The bridge normalizes the declared list into a mapping; an empty declaration stays empty.
    assert not functor.metadata.conservation_laws


def test_outputs_expose_both_temperature_usages_and_satisfy_the_consistency_contract() -> None:
    functor = _build_functor()
    # 303.15 K sits well above the 283.15 K Q10 reference, so the uncapped Q10 factor
    # exceeds the unit cap that bounds the heterotrophic modifier.
    outputs = functor.forward(_state(0.30, 303.15), time=0.0, dt=3600.0)
    f_t_uncapped = outputs["diagnostics.water_carbon.temperature_response_factor_uncapped"]
    f_t_capped = outputs["diagnostics.water_carbon.temperature_response_factor"]
    f_theta = outputs["diagnostics.water_carbon.moisture_response_factor"]
    cap = functor.temperature_response_cap.to(f_t_uncapped.dtype)
    assert torch.all(f_t_uncapped > cap)
    torch.testing.assert_close(f_t_capped, torch.minimum(f_t_uncapped, cap))
    # Bounded heterotrophic modifier uses the capped factor.
    expected_rh = torch.clamp(
        f_theta * f_t_capped,
        min=functor.minimum_moisture_factor.to(f_theta.dtype),
        max=functor.maximum_moisture_factor.to(f_theta.dtype),
    )
    torch.testing.assert_close(outputs["heterotrophic_respiration_modifier"], expected_rh)
    # Q10 decomposition multiplier uses the uncapped factor.
    torch.testing.assert_close(outputs["decomposition_rate_modifier"], f_theta * f_t_uncapped)
    torch.testing.assert_close(outputs["soil_carbon_turnover_factor"], outputs["decomposition_rate_modifier"])

    diagnostics = functor.check_conservation_laws(_state(0.30, 303.15), outputs)
    assert set(diagnostics) >= {"modifier_consistency", "bound_violations"}
    assert diagnostics["modifier_consistency"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["bound_violations"] == 0.0


def test_consistency_check_is_falsifiable() -> None:
    functor = _build_functor()
    state = _state(0.30, 293.15)
    outputs = functor.forward(state, time=0.0, dt=3600.0)
    tampered = dict(outputs)
    tampered["heterotrophic_respiration_modifier"] = outputs["heterotrophic_respiration_modifier"] * 0.5
    diagnostics = functor.check_conservation_laws(state, tampered)
    assert diagnostics["modifier_consistency"] > 1e-3
    tampered = dict(outputs)
    tampered["gpp_water_stress_factor"] = torch.tensor([1.5], dtype=torch.float32)
    diagnostics = functor.check_conservation_laws(state, tampered)
    assert diagnostics["bound_violations"] >= 1.0


def test_diagnose_conservation_reports_the_consistency_residual() -> None:
    functor = _build_functor()
    state = _state(0.25, 288.15)
    outputs = functor.forward(state, time=0.0, dt=3600.0)
    diagnostics = functor.diagnose_conservation(state, outputs)
    assert "modifier_consistency_residual" in diagnostics
    assert float(diagnostics["modifier_consistency_residual"]) == pytest.approx(0.0, abs=1e-6)
    assert "carbon_balance_residual" not in diagnostics


def test_modifier_only_coupling_builds_no_conservation_enforcer() -> None:
    """No mass law is declared, so no conservation enforcer or tolerance is resolved."""
    import inspect

    import relic.coupling.internal.water_carbon as water_carbon_module

    functor = _build_functor()
    assert not hasattr(functor, "_conservation_enforcer")
    assert not hasattr(functor, "conservation_tolerance")
    assert not hasattr(water_carbon_module, "ConservationEnforcer")
    assert "conservation_tolerance" not in inspect.getsource(WaterCarbonCoupling.__init__)


# Round 35: the change callback does not keep the coupling alive (the zero stress-weight
# sum is covered in test_water_carbon_stress_weights.py). Round 37 retired the
# reference rate k_ref together with the undeclared decomposition_rate output.


def test_teardown_unregisters_the_change_callback():
    import gc

    manager = _build_functor().get_parameter_manager()
    gc.collect()
    before = len(getattr(manager, "change_callbacks", []))
    functor = _build_functor()
    assert len(manager.change_callbacks) == before + 1
    del functor
    gc.collect()
    assert len(manager.change_callbacks) == before


# Round 37: the coupling emits dimensionless modifiers only. Its undeclared output
# decomposition_rate (k_ref k_dec) was never committed by the simulator and has no
# reader; the pedosphere owns the decomposition rate (carbon_decomposition_rate).
def test_the_undeclared_decomposition_rate_is_retired():
    functor = _build_functor()
    outputs = functor.forward(_state(0.30, 293.15), time=0.0, dt=3600.0)
    assert "decomposition_rate" not in outputs
    assert set(functor.metadata.output_variables) >= {"decomposition_rate_modifier"}
    diagnostics = functor.check_conservation_laws(_state(0.30, 293.15), outputs)
    assert "rate_scaling" not in diagnostics
    manager = functor.get_parameter_manager()
    assert manager.get_parameter("pedosphere.nutrient_cycling.reference_carbon_decomposition_rate") is None


# Round 38: of the formerly undeclared outputs, the three response factors are read by the
# coupling's own consistency check and are declared diagnostics.water_carbon.* with
# contracts; the four without any reader are removed. respiration_flux (= Rh f_T) had no
# reader: the soil respiration flux is the pedosphere's heterotrophic_respiration, so no
# second respiration flux is emitted and none can be counted twice.
RESPONSE_DIAGNOSTICS = (
    "diagnostics.water_carbon.temperature_response_factor",
    "diagnostics.water_carbon.temperature_response_factor_uncapped",
    "diagnostics.water_carbon.moisture_response_factor",
)
REMOVED_OUTPUTS = (
    "water_limited_photosynthesis",
    "temperature_limited_respiration",
    "respiration_flux",
    "temperature_factor",
    "temperature_response_factor",
    "temperature_response_factor_uncapped",
    "moisture_response_factor",
)


def test_every_output_is_declared_and_response_factors_are_diagnostics():
    from relic.contracts.contract_loader import load_contract_registry

    functor = _build_functor()
    outputs = functor.forward(_state(0.30, 293.15), time=0.0, dt=3600.0)
    declared = set(functor.metadata.output_variables)
    assert set(RESPONSE_DIAGNOSTICS) <= declared
    assert set(outputs) <= declared
    assert not set(REMOVED_OUTPUTS) & set(outputs)
    registry = load_contract_registry()
    for name in RESPONSE_DIAGNOSTICS:
        contract = registry.get_contract(name)
        assert contract.kind == "diagnostic"
        assert contract.udunits == "1"
    # GPP, R_a and R_h only fed the removed outputs; they are no inputs any more.
    assert not {
        "gross_primary_productivity",
        "autotrophic_respiration",
        "heterotrophic_respiration",
    } & set(functor.metadata.input_variables)


# Round 39: soil_matric_potential was read but no law used it; it is no input. The
# diagnostic extreme_temperature_response counted capped temperature responses outside
# [0.1, 10], the very clamp bounds applied before the cap, so it was always zero; it is
# removed with its two literals.
def test_unused_matric_potential_and_always_zero_diagnostic_are_removed():
    functor = _build_functor()
    assert "soil_matric_potential" not in functor.metadata.input_variables
    state = _state(0.30, 293.15)
    outputs = functor.forward(state, time=0.0, dt=3600.0)
    diagnostics = functor.diagnose_conservation(state, outputs)
    assert "extreme_temperature_response" not in diagnostics
