import math

import torch

import pytest

import relic.scientific.pedosphere.methane.production as production_module
from relic.scientific.pedosphere.methane import (
    MethaneProductionFunctor,
    MethaneScaleCouplingFunctor,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager


def apply_updates(state, updates):
    for key, value in updates.items():
        mode = UpdateMode.ACCUMULATE
        raw_value = value
        if isinstance(value, UpdateItem):
            raw_value = value.value
            mode = value.mode
        tensor = raw_value if isinstance(raw_value, torch.Tensor) else torch.tensor(float(raw_value))
        tensor = tensor.to(torch.float32)
        if key in state and mode != UpdateMode.REPLACE:
            state[key] = state[key] + tensor
        else:
            state[key] = tensor.clone()
    return state


def run_functor(functor, state, dt=1.0):
    updates = functor(state, time=0.0, dt=torch.tensor(float(dt)))
    return apply_updates(state, updates)


def base_state():
    return {
        "soil_temperature": torch.tensor(293.15),
        "soil_ph": torch.tensor(7.0),
        "heterotrophic_respiration": torch.tensor(4.0),
        "redox_potential": torch.tensor(-150.0),
        "soil_co2_concentration": torch.tensor(0.1),
        "microbial_activity_index": torch.tensor(0.8),
        "ch4_soil_storage": torch.tensor(5.0),
        "ch4_soil_storage_delta": torch.tensor(0.0),
    }


def canonical_production_state():
    state = base_state()
    state.pop("soil_co2_concentration")
    state["land_surface.wetland_fraction"] = torch.tensor(0.5)
    state["methane_gating_weight"] = torch.tensor(0.5)
    return state


def canonical_scale_state():
    return {
        "land_surface.wetland_fraction": torch.tensor(0.65, dtype=torch.float64),
        "water_filled_pore_space": torch.tensor(0.70, dtype=torch.float64),
        "methane_data_score": torch.tensor(0.60, dtype=torch.float64),
    }


def test_production_uses_physical_constants_ssot_for_q10_interval(monkeypatch):
    calls = []

    def fake_get_physical_constant(name):
        calls.append(name)
        return 10.0

    monkeypatch.setattr(
        production_module,
        "get_physical_constant",
        fake_get_physical_constant,
    )

    production = MethaneProductionFunctor()

    assert calls == ["relic.physical.constants.temperature.q10_interval"]
    assert production.q10_interval == 10.0
    assert all(
        not key.startswith("relic.physical.constants.")
        for key in production.metadata.parameters
    )


def test_production_outputs_use_canonical_flux_and_diagnostic_contracts():
    production = MethaneProductionFunctor()
    state = base_state()
    state["methane_gating_weight"] = torch.tensor(0.5)
    state["land_surface.wetland_fraction"] = torch.tensor(0.5)

    updates = production(state, time=0.0, dt=torch.tensor(1.0))
    expected_outputs = {
        "ch4_production_rate",
        "ch4_storage_tendency_prod",
        "diagnostics.methane_production.imaple_rate",
        "diagnostics.methane_production.calibrated_rate",
    }

    assert set(updates) == expected_outputs
    assert set(production.metadata.output_variables) == expected_outputs
    for update in updates.values():
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.unit == "g C m-2 s-1"
        assert update.source_functor == production.name

    assert updates["ch4_production_rate"].element == "C"
    assert updates["ch4_storage_tendency_prod"].element is None
    assert "ch4_production_imaple_rate" not in updates
    assert "ch4_production_dndc_rate" not in updates
    assert "diagnostics.methane_production.dndc_rate" not in updates


def test_imaple_temperature_response_matches_yue_equations_35_to_39():
    """Yue et al. (2024) Eqs. 35-39 with the Zhu et al. (2014) Eq. 5 exponent.

    Q10(Ts) = clip(r_b * Q_b^((Ts - T_base)/10), Q10_min, Q10_max), r_b = 3.0,
    Q_b = 1.33, T_base = 25 C (Yue et al. 2024 Eq. 39); the clip range 1.7-16 is the
    observed production Q10 range of Zhu et al. (2014), applied in the released
    iMAPLE code. f_T = v^x exp(x (1 - v)), v = (Tmax - Ts)/(Tmax - Topt),
    x = W^2 (1 + sqrt(1 + 40/W))^2 / 400, W = ln(Q10) (Tmax - Topt).
    Oracle values were computed independently in double precision.
    """
    production = MethaneProductionFunctor()
    manager = get_global_manager()
    for key, expected in (
        ("q10", 3.0),
        ("q10_temperature_sensitivity", 1.33),
        ("q10_base_temperature", 298.15),
        ("q10_min", 1.7),
        ("q10_max", 16.0),
    ):
        spec = manager.get_parameter(f"pedosphere.methane.production.{key}")
        assert float(spec.value) == pytest.approx(expected, rel=1.0e-6), key
        assert f"pedosphere.methane.production.{key}" in production.metadata.parameters
    celsius = torch.tensor([0.0, 10.0, 25.0, 35.0], dtype=torch.float64)
    temperatures = celsius + 273.15
    q10 = production._temperature_dependent_q10(temperatures)
    assert q10.tolist() == pytest.approx([1.7, 1.9558871493025263, 3.0, 3.99], rel=1.0e-6)
    actual = production._optimal_temperature_response(temperatures)
    assert actual.tolist() == pytest.approx(
        [0.28559450063252734, 0.4636115138541493, 1.0, 0.08795135183226152], rel=1.0e-5
    )
    outside = production._optimal_temperature_response(
        torch.tensor([273.15 - 1.0, 318.15 + 1.0], dtype=torch.float64)
    )
    assert outside.tolist() == [0.0, 0.0]
    rising = production._optimal_temperature_response(
        torch.linspace(273.15, 298.15, 501, dtype=torch.float64)
    )
    assert bool(torch.all(rising[1:] >= rising[:-1]))
    falling = production._optimal_temperature_response(
        torch.linspace(298.15, 318.15, 401, dtype=torch.float64)
    )
    assert bool(torch.all(falling[1:] <= falling[:-1]))
    assert production.temperature_shape_at_coefficient == pytest.approx(40.0)
    assert production.temperature_shape_normalization == pytest.approx(400.0)


def test_imaple_ph_response_matches_zhu_equation_6():
    production = MethaneProductionFunctor()
    soil_ph = torch.tensor([3.9, 4.0, 5.5, 7.0, 9.0, 9.1])
    expected = torch.zeros_like(soil_ph)
    active_ph = soil_ph[1:5]
    lower_ratio = (active_ph - 4.0) / (7.0 - 4.0)
    upper_ratio = (9.0 - active_ph) / (9.0 - 7.0)
    exponent = (9.0 - 7.0) / (7.0 - 4.0)
    expected[1:5] = lower_ratio * upper_ratio**exponent

    actual = production._ph_factor(soil_ph)

    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)
    assert production.ph_lower_limit == pytest.approx(4.0)
    assert production.ph_upper_limit == pytest.approx(9.0)
    assert not hasattr(production, "ph_sigma")


def test_imaple_fixed_reference_case_matches_the_published_response_chain():
    # References: Yue et al. (2024), Eqs. 34-39; Zhu et al. (2014), Eqs. 3-6.
    # Equation: P = Rh * r * f_T * f_pH * f_wet (Yue et al. 2024, Eq. 34; the former
    # unsourced closure factor 6.45 was removed in round 50); f_T uses Q10(20 C) =
    # 3.0 * 1.33^(-0.5) = 2.60133 (Eq. 39, item 50.1).
    # Parameters: Rh=4, r=0.2 (iMAPLE v1 release value), f_wet=0.95, T=293.15 K, pH=7.
    production = MethaneProductionFunctor()
    state = {
        "soil_temperature": torch.tensor(293.15, dtype=torch.float64),
        "soil_ph": torch.tensor(7.0, dtype=torch.float64),
        "heterotrophic_respiration": torch.tensor(4.0, dtype=torch.float64),
        "redox_potential": torch.tensor(-150.0, dtype=torch.float64),
        "microbial_activity_index": torch.tensor(0.8, dtype=torch.float64),
        "land_surface.wetland_fraction": torch.tensor(0.95, dtype=torch.float64),
        "methane_gating_weight": torch.tensor(0.0, dtype=torch.float64),
    }

    updates = production.forward(state, time=0.0, dt=torch.tensor(1.0))

    expected_temperature_factor = 0.8296362679609771
    expected_raw_rate = 0.6305235636503426
    assert expected_raw_rate == pytest.approx(
        4.0 * 0.2 * expected_temperature_factor * 0.95,
        rel=1.0e-14,
    )
    assert updates["diagnostics.methane_production.imaple_rate"].value.item() == (
        pytest.approx(expected_raw_rate, rel=1.0e-6, abs=1.0e-8)
    )
    assert updates["ch4_production_rate"].value.item() == pytest.approx(
        expected_raw_rate,
        rel=1.0e-6,
        abs=1.0e-8,
    )
    torch.testing.assert_close(
        updates["ch4_storage_tendency_prod"].value,
        updates["ch4_production_rate"].value,
        rtol=0.0,
        atol=0.0,
    )


def test_production_has_no_retired_or_unused_runtime_parameters():
    production = MethaneProductionFunctor()

    assert not hasattr(production, "respiration_time_scale")
    assert not hasattr(production, "maximum_efficiency")
    assert not hasattr(production, "dt_minimum")
    # Round 49: the unsourced substrate enhancement (1 + k_h * H2 + k_a * DOC)
    # and the unsourced wetland activation floor are removed.
    assert not hasattr(production, "hydrogen_rate_constant")
    assert not hasattr(production, "acetate_rate_constant")
    assert not hasattr(production, "wetland_floor")
    # Round 50: the closure factor, the logistic redox slope and the dndc
    # key names are removed.
    assert not hasattr(production, "production_scaling")
    assert not hasattr(production, "eh_slope")
    assert not hasattr(production, "eh_threshold")
    # Round 51.1: the unsourced Q10 reference temperature and the unsourced
    # microbial-activity reference are removed.
    assert not hasattr(production, "temperature_reference")
    assert not hasattr(production, "microbe_reference")
    removed_keys = {
        "pedosphere.methane.production.dndc.hydrogenotrophic_rate_constant",
        "pedosphere.methane.production.dndc.acetoclastic_rate_constant",
        "pedosphere.methane.production.wetland_fraction_floor",
        "pedosphere.methane.production.scaling_factor",
        "pedosphere.methane.production.dndc.eh_threshold",
        "pedosphere.methane.production.dndc.eh_slope",
        "pedosphere.methane.production.dndc.microbe_activity_reference",
        "pedosphere.methane.production.temperature_reference",
        "pedosphere.methane.production.calibrated.microbe_activity_reference",
    }
    stale_keys = {
        "pedosphere.methane.production.respiration_time_scale",
        "pedosphere.methane.production.maximum_substrate_enhancement",
        "pedosphere.numerical.minimum_timestep",
        *removed_keys,
    }
    assert stale_keys.isdisjoint(production.metadata.parameters)
    manager = get_global_manager()
    for key in sorted(removed_keys):
        assert manager.get(key, required=False) is None, key


def test_production_declares_only_inputs_used_by_the_runtime():
    production = MethaneProductionFunctor()
    expected_inputs = {
        "soil_temperature",
        "soil_ph",
        "heterotrophic_respiration",
        "redox_potential",
        "land_surface.wetland_fraction",
        "microbial_activity_index",
        "methane_gating_weight",
    }

    assert set(production.metadata.input_variables) == expected_inputs
    assert "soil_co2_concentration" not in production.metadata.input_variables
    assert "dissolved_organic_carbon" not in production.metadata.input_variables
    assert "hydrogen_concentration" not in production.metadata.input_variables


def test_production_metadata_does_not_claim_the_dndc_substrate_model():
    # The calibrated branch is an Rh-based empirical expression with DNDC-type
    # Q10 and redox controls; the DNDC substrate kinetics (Fumoto et al. 2008;
    # van Bodegom and Scholten 2001) are not implemented, and Li et al. (1992)
    # has no methane module, so neither may be cited as the source of the law.
    production = MethaneProductionFunctor()
    metadata = production.metadata

    assert not any("Li et al. (1992)" in ref for ref in metadata.references)
    for symbol in ("k_h", "k_a", "H2", "DOC"):
        assert symbol not in metadata.mathematical_expression, symbol
    assert "DNDC pathways" not in metadata.description
    assert "not the DNDC substrate" in (MethaneProductionFunctor.__doc__ or "")
    assert "s_P" not in metadata.mathematical_expression
    assert "Q10(Ts)" not in metadata.mathematical_expression
    assert "P_cal = Rh * r * f_T(Ts) * f_Eh(Eh) * A" in (MethaneProductionFunctor.__doc__ or "")
    assert not any(
        key.startswith("pedosphere.methane.production.dndc.") for key in metadata.parameters
    )


def test_scale_metadata_does_not_cite_li_1992_for_the_endpoint_branches():
    # Li et al. (1992) has no methane module; the gate routes between the iMAPLE
    # branch and the RELIC calibrated branch of MethaneProductionFunctor.
    metadata = MethaneScaleCouplingFunctor().metadata

    assert not any("Li et al. (1992)" in ref for ref in metadata.references)
    assert "Li et al. (1992)" not in (MethaneScaleCouplingFunctor.__doc__ or "")
    assert "w_dndc" not in metadata.mathematical_expression


@pytest.mark.parametrize(
    "key,expected_reference",
    [
        ("pedosphere.methane.production.release_ratio", "Wania et al. (2010)"),
        ("pedosphere.methane.production.release_ratio", "Zhu et al. (2014)"),
        ("pedosphere.methane.production.q10", "Zhu et al. (2014)"),
        ("pedosphere.methane.production.calibrated.eh_full_activity", "Zhu et al. (2014)"),
        ("pedosphere.methane.production.calibrated.eh_zero_activity", "Zhu et al. (2014)"),
        ("pedosphere.methane.production.calibrated.eh_full_activity", "Zhang et al. (2002)"),
    ],
)
def test_sourced_production_parameters_cite_the_checked_primary_source(key, expected_reference):
    spec = get_global_manager().get_parameter(key)

    assert spec.provenance["source"] == "literature"
    assert spec.provenance.get("literature_check")
    assert any(ref.startswith(expected_reference) for ref in spec.references)


def test_release_ratio_is_the_value_of_the_yue_2024_model_release():
    # Yue et al. (2024) Eq. 34 names the CH4:CO2 release ratio r after Wania et al.
    # (2010) without printing a value; the iMAPLE v1 code released with the paper
    # (ch4flux_wetland.f, parameter r) uses r = 0.2, inside the Wania et al. (2010)
    # site fits (0.1-0.3) and the Zhu et al. (2014) calibrated range (0.1-0.7).
    spec = get_global_manager().get_parameter("pedosphere.methane.production.release_ratio")

    assert float(spec.value) == pytest.approx(0.2)
    assert spec.provenance["source"] == "literature"
    assert any("ch4flux_wetland.f" in ref for ref in spec.references)
    assert "r = 0.2" in spec.provenance["literature_check"]


@pytest.mark.parametrize(
    "missing_key",
    [
        "soil_temperature",
        "soil_ph",
        "heterotrophic_respiration",
        "redox_potential",
        "land_surface.wetland_fraction",
        "microbial_activity_index",
        "methane_gating_weight",
    ],
)
def test_production_forward_fails_fast_when_required_input_is_missing(missing_key):
    production = MethaneProductionFunctor()
    state = canonical_production_state()
    del state[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        production.forward(state, time=0.0, dt=torch.tensor(1.0))


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("soil_temperature", float("nan")),
        ("soil_ph", float("inf")),
        ("heterotrophic_respiration", -1.0),
        ("microbial_activity_index", -1.0),
        ("microbial_activity_index", 1.1),
        ("land_surface.wetland_fraction", 1.1),
        ("methane_gating_weight", -0.1),
    ],
)
def test_production_forward_rejects_invalid_scientific_inputs(key, bad_value):
    production = MethaneProductionFunctor()
    state = canonical_production_state()
    state[key] = torch.tensor(bad_value)

    with pytest.raises(ValueError, match=key):
        production.forward(state, time=0.0, dt=torch.tensor(1.0))


def test_production_forward_rejects_non_tensor_input():
    production = MethaneProductionFunctor()
    state = canonical_production_state()
    state["soil_temperature"] = 293.15

    with pytest.raises(TypeError, match="soil_temperature"):
        production.forward(state, time=0.0, dt=torch.tensor(1.0))


def test_imaple_wetland_factor_is_the_continuous_inundation_fraction():
    # References: Yue et al. (2024), Eq. 34: f_wetland is the wetland inundation
    # fraction; no activation floor is part of the published law.
    # Equation: P_iMAPLE(f) = f * P_iMAPLE(1) for every f in [0, 1].
    production = MethaneProductionFunctor()
    state = canonical_production_state()
    state = {key: value.to(torch.float64) for key, value in state.items()}
    state["methane_gating_weight"] = torch.tensor(0.0, dtype=torch.float64)
    state["land_surface.wetland_fraction"] = torch.tensor(1.0, dtype=torch.float64)
    full_rate = production.forward(state, time=0.0, dt=torch.tensor(1.0))[
        "diagnostics.methane_production.imaple_rate"
    ].value
    assert full_rate.item() > 0.0

    for fraction in (0.0, 0.01, 0.03, 0.049, 0.05, 0.5):
        state["land_surface.wetland_fraction"] = torch.tensor(fraction, dtype=torch.float64)
        updates = production.forward(state, time=0.0, dt=torch.tensor(1.0))
        torch.testing.assert_close(
            updates["diagnostics.methane_production.imaple_rate"].value,
            fraction * full_rate,
            rtol=1.0e-12,
            atol=0.0,
        )
        torch.testing.assert_close(
            updates["ch4_production_rate"].value,
            fraction * full_rate,
            rtol=1.0e-12,
            atol=0.0,
        )


def test_calibrated_branch_is_the_rh_based_product_without_substrate_terms():
    # References: RELIC convention (Rh-based empirical branch with DNDC-type
    # controls); Zhu et al. (2014) Eqs. 3-5 peak temperature factor (the same
    # f_T as the iMAPLE branch); DNDC95 microbial activity index A in [0, 1].
    # Equation: P_cal = Rh * r * f_T(Ts) * f_Eh(Eh) * A,
    #   f_Eh = clip((Eh_0 - Eh) / (Eh_0 - Eh_1), 0, 1) (Zhu et al. 2014 after Zhang
    #   et al. 2002); no DOC or H2 factor and no closure factor.
    production = MethaneProductionFunctor()
    state = canonical_production_state()
    state = {key: value.to(torch.float64) for key, value in state.items()}
    state["methane_gating_weight"] = torch.tensor(1.0, dtype=torch.float64)

    updates = production.forward(state, time=0.0, dt=torch.tensor(1.0))

    as_float = production._scalar_from_value
    temperature_factor = production._optimal_temperature_response(
        torch.tensor(293.15, dtype=torch.float64)
    ).item()
    assert 0.0 < temperature_factor <= 1.0
    eh_full = as_float(production.eh_full_activity)
    eh_zero = as_float(production.eh_zero_activity)
    assert (eh_full, eh_zero) == (-200.0, -100.0)
    redox_factor = min(max((eh_zero - (-150.0)) / (eh_zero - eh_full), 0.0), 1.0)
    assert redox_factor == pytest.approx(0.5)
    expected = (
        4.0
        * as_float(production.release_ratio)
        * temperature_factor
        * redox_factor
        * 0.8
    )
    assert updates["diagnostics.methane_production.calibrated_rate"].value.item() == (
        pytest.approx(expected, rel=1.0e-12)
    )
    assert updates["ch4_production_rate"].value.item() == pytest.approx(expected, rel=1.0e-12)


def test_scale_uses_canonical_inputs_outputs_and_explicit_updates():
    scale = MethaneScaleCouplingFunctor()
    state = canonical_scale_state()
    state_before = {key: value.clone() for key, value in state.items()}

    updates = scale.forward(
        state,
        time=0.0,
        dt=torch.tensor(1800.0, dtype=torch.float64),
    )

    assert set(scale.metadata.input_variables) == {
        "land_surface.wetland_fraction",
        "water_filled_pore_space",
        "methane_data_score",
    }
    assert set(scale.metadata.output_variables) == {
        "methane_gating_weight",
        "methane_gating_score",
    }
    assert set(updates) == set(scale.metadata.output_variables)
    assert scale.get_dependencies() == []
    for key, update in updates.items():
        assert isinstance(update, UpdateItem), key
        assert update.mode == UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.unit == "1"
        assert update.source_functor == scale.name

    torch.testing.assert_close(
        updates["methane_gating_score"].value,
        torch.tensor(0.655, dtype=torch.float64),
        rtol=1.0e-7,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        updates["methane_gating_weight"].value,
        torch.tensor(0.155, dtype=torch.float64),
        rtol=1.0e-7,
        atol=1.0e-12,
    )
    for key, original in state_before.items():
        torch.testing.assert_close(state[key], original, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "missing_key",
    [
        "land_surface.wetland_fraction",
        "water_filled_pore_space",
        "methane_data_score",
    ],
)
def test_scale_fails_fast_when_required_input_is_missing(missing_key):
    scale = MethaneScaleCouplingFunctor()
    state = canonical_scale_state()
    del state[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        scale.forward(state, time=0.0, dt=torch.tensor(1.0))


@pytest.mark.parametrize(
    "key",
    [
        "land_surface.wetland_fraction",
        "water_filled_pore_space",
        "methane_data_score",
    ],
)
def test_scale_rejects_non_tensor_inputs(key):
    scale = MethaneScaleCouplingFunctor()
    state = canonical_scale_state()
    state[key] = 0.5

    with pytest.raises(TypeError, match=key):
        scale.forward(state, time=0.0, dt=torch.tensor(1.0))


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("land_surface.wetland_fraction", float("nan")),
        ("water_filled_pore_space", float("inf")),
        ("methane_data_score", -0.1),
        ("land_surface.wetland_fraction", 1.1),
        ("water_filled_pore_space", -0.1),
        ("methane_data_score", 1.1),
    ],
)
def test_scale_rejects_non_finite_or_out_of_range_inputs(key, bad_value):
    scale = MethaneScaleCouplingFunctor()
    state = canonical_scale_state()
    state[key] = torch.tensor(bad_value, dtype=torch.float64)

    with pytest.raises(ValueError, match=key):
        scale.forward(state, time=0.0, dt=torch.tensor(1.0))


def test_gating_prefers_imaple_in_dry_conditions():
    state = base_state()
    state.update(
        {
            "land_surface.wetland_fraction": torch.tensor(0.05),
            "water_filled_pore_space": torch.tensor(0.3),
            "methane_data_score": torch.tensor(0.0),
        }
    )
    scale = MethaneScaleCouplingFunctor()
    production = MethaneProductionFunctor()

    run_functor(scale, state)
    gating_weight = state["methane_gating_weight"].item()
    run_functor(production, state)

    imaple_rate = state["diagnostics.methane_production.imaple_rate"].item()
    calibrated_rate = state["diagnostics.methane_production.calibrated_rate"].item()
    production_rate = state["ch4_production_rate"].item()

    assert calibrated_rate >= 0.0
    assert gating_weight < 0.2
    assert imaple_rate > 0.0
    assert abs(production_rate - imaple_rate) < 1e-6


def test_gating_prefers_the_calibrated_branch_under_saturated_conditions():
    state = base_state()
    state.update(
        {
            "land_surface.wetland_fraction": torch.tensor(0.95),
            "water_filled_pore_space": torch.tensor(0.93),
            "methane_data_score": torch.tensor(1.0),
            "redox_potential": torch.tensor(-250.0),
            "microbial_activity_index": torch.tensor(1.0),
        }
    )
    scale = MethaneScaleCouplingFunctor()
    production = MethaneProductionFunctor()

    run_functor(scale, state)
    gating_weight = state["methane_gating_weight"].item()
    run_functor(production, state)

    imaple_rate = state["diagnostics.methane_production.imaple_rate"].item()
    calibrated_rate = state["diagnostics.methane_production.calibrated_rate"].item()
    production_rate = state["ch4_production_rate"].item()

    expected_rate = (1.0 - gating_weight) * imaple_rate + gating_weight * calibrated_rate
    assert gating_weight > 0.3
    assert abs(production_rate - expected_rate) < 1e-6


@pytest.mark.parametrize(
    "redox_mv,expected_factor",
    [(-300.0, 1.0), (-200.0, 1.0), (-175.0, 0.75), (-150.0, 0.5), (-125.0, 0.25), (-100.0, 0.0), (0.0, 0.0)],
)
def test_calibrated_redox_factor_is_the_zhu_2014_piecewise_linear_form(redox_mv, expected_factor):
    # References: Zhu et al. (2014), Sect. 2.2.1, after Zhang et al. (2002): the
    # factor is 1 below -200 mV, falls linearly to 0 between -200 and -100 mV,
    # and is 0 above -100 mV.
    production = MethaneProductionFunctor()
    factor = production._redox_gate(torch.tensor(redox_mv, dtype=torch.float64))

    assert factor.item() == pytest.approx(expected_factor, abs=1.0e-12)


@pytest.mark.parametrize(
    "key",
    [
        "pedosphere.methane.production.calibrated.eh_full_activity",
        "pedosphere.methane.production.calibrated.eh_zero_activity",
    ],
)
def test_calibrated_production_parameters_pass_their_spec_validation(key):
    # Negative redox potentials are environmental responses, not biological rates.
    spec = get_global_manager().get_parameter(key)

    assert spec.validate_value(spec.value) == []
