"""Owner tests of the WaterCarbonCoupling stress combination (rounds 35-40).

overall_water_stress = min(f_soil, f_vpd) (Landsberg and Waring 1997, 3-PG: the soil-water and
VPD modifiers are not multiplicative relative to each other; the lower one limits), while the
photosynthetic factor keeps the Jarvis (1976) product f_gpp = f_soil f_vpd. The former weights
gpp_stress_weight / vpd_stress_weight (a weighted arithmetic mean without a source) are gone.
"""

import pytest
import torch

from relic.coupling.internal.water_carbon import WaterCarbonCoupling


def _build_functor() -> WaterCarbonCoupling:
    functor = WaterCarbonCoupling(name="water-carbon-stress-combination", config={})
    functor._get_cache()  # Warm parameter cache for direct attribute updates
    return functor


def _state(**overrides) -> dict:
    state = {
        "soil_moisture": torch.tensor(0.30, dtype=torch.float32),
        "soil_temperature": torch.tensor(293.15, dtype=torch.float32),
        "vapor_pressure_deficit": torch.tensor(800.0, dtype=torch.float32),
    }
    state.update(overrides)
    return state


def test_vapor_pressure_deficit_is_a_required_input() -> None:
    """The stress law f_gpp = f_soil f_vpd and the combined stress both use D; D is an
    always-present atmospheric state, so a missing D raises (no f_vpd = 1 stand-in)."""
    functor = _build_functor()
    assert "vapor_pressure_deficit" in functor.get_required_inputs()
    state = _state()
    del state["vapor_pressure_deficit"]
    with pytest.raises(ValueError, match="vapor_pressure_deficit"):
        functor.forward(state, time=0.0, dt=3600.0)


def test_negative_vapor_pressure_deficit_is_rejected_not_clamped() -> None:
    """D = e_s - e_a >= 0 (contract lower bound): a negative D raises."""
    functor = _build_functor()
    with pytest.raises(ValueError, match="vapor_pressure_deficit"):
        functor.forward(_state(vapor_pressure_deficit=torch.tensor(-10.0)), time=0.0, dt=3600.0)


# References: Landsberg and Waring (1997), For. Ecol. Manage. 95, 209-228 (3-PG uses the
#   lower of the VPD and soil-water modifiers); Jarvis (1976) (multiplicative f_gpp).
# Expected values: piecewise f_soil = (theta - theta_onset) / (theta_opt - theta_onset) on the
#   rising segment and 1 above theta_opt; f_vpd = 1 / (1 + D / D_50).
def _piecewise_soil_factor(functor: WaterCarbonCoupling, theta: float) -> float:
    onset = float(functor.stress_onset_threshold)
    optimum = float(functor.optimal_moisture)
    if theta >= optimum:
        return 1.0
    return (theta - onset) / (optimum - onset)


@pytest.mark.parametrize(
    ("f_soil_target", "f_vpd_target"),
    [
        # Distinguishes the three forms: min 0.5, Jarvis product 0.45, weighted mean 0.62.
        (0.5, 0.9),
        # Wet soil and strong VPD stress: min 0.25.
        (1.0, 0.25),
    ],
)
def test_overall_stress_is_the_lower_of_the_soil_and_vpd_factors(f_soil_target, f_vpd_target) -> None:
    functor = _build_functor()
    functor._wc_parameter_cache.stress_function_type = "piecewise"
    onset = float(functor.stress_onset_threshold)
    optimum = float(functor.optimal_moisture)
    theta = 0.40 if f_soil_target == 1.0 else onset + f_soil_target * (optimum - onset)
    d_50 = float(functor.vpd_half_saturation)
    vpd = d_50 * (1.0 / f_vpd_target - 1.0)
    f_soil = _piecewise_soil_factor(functor, theta)
    f_vpd = 1.0 / (1.0 + vpd / d_50)
    assert f_soil == pytest.approx(f_soil_target, rel=1e-5)
    assert f_vpd == pytest.approx(f_vpd_target, rel=1e-5)

    outputs = functor.forward(
        _state(
            soil_moisture=torch.tensor(theta, dtype=torch.float32),
            vapor_pressure_deficit=torch.tensor(vpd, dtype=torch.float32),
        ),
        time=0.0,
        dt=3600.0,
    )
    assert float(outputs["overall_water_stress"]) == pytest.approx(min(f_soil, f_vpd), rel=1e-5)
    # The photosynthetic factor keeps the Jarvis product.
    assert float(outputs["gpp_water_stress_factor"]) == pytest.approx(f_soil * f_vpd, rel=1e-5)


def test_the_unsourced_weights_are_removed() -> None:
    functor = _build_functor()
    manager = functor.get_parameter_manager()
    for key in ("gpp_stress_weight", "vpd_stress_weight"):
        assert not hasattr(functor, key)
        assert key not in functor.metadata.parameters
        assert manager.get_parameter(f"coupling.scales.water_carbon.{key}") is None


def test_the_unread_vpd_parameters_are_removed() -> None:
    """vpd_stress_slope and maximum_vpd_tolerance were loaded but no calculation used them."""
    functor = _build_functor()
    manager = functor.get_parameter_manager()
    for key in ("vpd_stress_slope", "maximum_vpd_tolerance"):
        assert not hasattr(functor, key)
        assert key not in functor.metadata.parameters
        assert manager.get_parameter(f"coupling.scales.water_carbon.{key}") is None
