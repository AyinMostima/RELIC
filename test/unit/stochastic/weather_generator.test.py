"""SSOT unit tests for stochastic weather precipitation units."""

from __future__ import annotations

import inspect

import pytest
import torch

import relic.stochastic.weather_generator as weather_module
from relic.stochastic.weather_generator import (
    MarkovGammaPrecipitation,
    PrecipitationParameters,
)
from relic.utils.time_constants import SECONDS_PER_DAY


def _parameters() -> PrecipitationParameters:
    return PrecipitationParameters(
        p_wet_wet=1.0,
        p_wet_dry=1.0,
        gamma_shape=2.5,
        gamma_scale=5.0,
        initial_state=True,
        wet_day_threshold=1.0e-8,
    )


def _sample_flux(seed: int) -> torch.Tensor:
    generator = MarkovGammaPrecipitation(_parameters())
    output = generator.sample(
        torch.tensor(0.0),
        num_samples=1,
        seed=seed,
        num_steps=1,
    )
    return output["precipitation_flux"][0, 0].detach().cpu()


def test_wet_day_sample_matches_daily_amount_to_si_flux_oracle() -> None:
    seed = 24680
    parameters = _parameters()
    torch.manual_seed(seed)
    expected_amount = torch.distributions.Gamma(
        parameters.gamma_shape,
        rate=1.0 / parameters.gamma_scale,
    ).sample()

    actual_flux = _sample_flux(seed)

    torch.testing.assert_close(
        actual_flux,
        expected_amount / float(SECONDS_PER_DAY),
    )


def test_sample_log_probability_and_moments_bind_seconds_per_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seconds_per_day = float(SECONDS_PER_DAY)
    monkeypatch.setattr(
        weather_module,
        "SECONDS_PER_DAY",
        seconds_per_day,
        raising=False,
    )
    baseline_flux = _sample_flux(13579)
    baseline_process = MarkovGammaPrecipitation(_parameters())
    baseline_moments = baseline_process.moments()
    observations = torch.tensor([5.0 / seconds_per_day])
    conditioning_state = torch.ones_like(observations)
    baseline_log_probability = baseline_process.log_probability(
        observations,
        conditioning_state=conditioning_state,
    )

    monkeypatch.setattr(weather_module, "SECONDS_PER_DAY", seconds_per_day / 2.0)
    scaled_flux = _sample_flux(13579)
    scaled_process = MarkovGammaPrecipitation(_parameters())
    scaled_moments = scaled_process.moments()
    scaled_log_probability = scaled_process.log_probability(
        observations,
        conditioning_state=conditioning_state,
    )

    torch.testing.assert_close(scaled_flux, 2.0 * baseline_flux)
    torch.testing.assert_close(
        scaled_moments["mean_precipitation"],
        2.0 * baseline_moments["mean_precipitation"],
    )
    torch.testing.assert_close(
        scaled_moments["variance_precipitation"],
        4.0 * baseline_moments["variance_precipitation"],
    )
    assert not torch.equal(scaled_log_probability, baseline_log_probability)


def test_precipitation_process_has_no_local_seconds_per_day_literal() -> None:
    source = inspect.getsource(MarkovGammaPrecipitation)

    assert "24 * 3600" not in source
    assert "24*3600" not in source
    assert "SECONDS_PER_DAY" in source


# ---------------------------------------------------------------------------
# Seeded draws and the derived vapour pressure (MA-21 C7).
#
# References: Tetens (1930); Allen et al. (1998) FAO-56, Eq. 11.
# Mathematical expression: e_s = e_0 exp(a T_c / (T_c + b)), T_c = T - 273.15.
# Parameter meanings: e_0, a, b = Tetens constants from the physical-constant
#     SSOT, T = air temperature [K], T_c = air temperature [degC].
#
# A seeded sample uses its own generator state: the caller's global CPU and
# CUDA streams are unchanged, and the component seeds of the coupled generator
# are derived from one seed sequence rather than offset by a fixed amount.
# ---------------------------------------------------------------------------

import math

from relic.parameters import get_global_manager
from relic.stochastic.weather_generator import (
    OrnsteinUhlenbeckTemperature,
    TemperatureParameters,
    WeatherGenerator,
)
from relic.utils.physical_constants import get_physical_constant


def _temperature(**overrides) -> TemperatureParameters:
    parameters = TemperatureParameters.from_parameter_manager()
    for name, value in overrides.items():
        setattr(parameters, name, value)
    return parameters


def _precipitation(**overrides) -> PrecipitationParameters:
    parameters = PrecipitationParameters.from_parameter_manager()
    for name, value in overrides.items():
        setattr(parameters, name, value)
    return parameters


def _global_rng_states():
    cuda_state = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    return torch.random.get_rng_state().clone(), cuda_state


@pytest.mark.parametrize(
    "process",
    [
        lambda: OrnsteinUhlenbeckTemperature(_temperature()),
        lambda: MarkovGammaPrecipitation(_precipitation()),
        lambda: WeatherGenerator(_temperature(), _precipitation()),
    ],
    ids=["ou_temperature", "markov_gamma", "weather_generator"],
)
def test_seeded_sample_leaves_the_global_generators_untouched(process) -> None:
    generator = process()
    cpu_before, cuda_before = _global_rng_states()

    first = generator.sample(torch.tensor(288.15), num_samples=2, seed=31, num_steps=6)
    cpu_after, cuda_after = _global_rng_states()
    second = generator.sample(torch.tensor(288.15), num_samples=2, seed=31, num_steps=6)

    assert torch.equal(cpu_before, cpu_after)
    if cuda_before is not None:
        assert torch.equal(cuda_before, cuda_after)
    for key, value in first.items():
        assert torch.equal(value, second[key]), key


def test_component_seeds_are_not_fixed_offsets_of_the_member_seed() -> None:
    """Member seed s must not reuse the precipitation stream of member s + 1000."""
    weather = WeatherGenerator(_temperature(), _precipitation())
    coupled = weather.sample(torch.tensor(288.15), num_samples=1, seed=5, num_steps=60)
    offset = MarkovGammaPrecipitation(_precipitation()).sample(
        torch.tensor(288.15), num_samples=1, seed=1005, num_steps=60
    )
    assert not torch.equal(coupled["precipitation_state"], offset["precipitation_state"])


def test_saturation_vapor_pressure_uses_the_celsius_temperature() -> None:
    weather = WeatherGenerator(_temperature(), _precipitation())
    output = weather.sample(
        torch.tensor(293.15, dtype=torch.float64), num_samples=1, seed=3, num_steps=1
    )

    prefix = "relic.physical.constants.psychrometrics."
    base = float(get_physical_constant(prefix + "tetens_base_pressure"))
    coeff_a = float(get_physical_constant(prefix + "tetens_coefficient_a"))
    coeff_b = float(get_physical_constant(prefix + "tetens_coefficient_b"))
    temp_c = 293.15 - float(get_physical_constant("celsius_to_kelvin"))
    expected = base * math.exp(coeff_a * temp_c / (temp_c + coeff_b))

    assert float(output["saturation_vapor_pressure"][0, 0]) == pytest.approx(expected, rel=1e-9)


def test_joint_log_probability_splits_the_conditioning_state_per_component() -> None:
    """Temperature conditions the OU term, the wet/dry state the Markov term."""
    weather = WeatherGenerator(_temperature(), _precipitation())
    observations = torch.tensor([[289.0, 0.0]], dtype=torch.float64)
    conditioning = torch.tensor([[288.0, 1.0]], dtype=torch.float64)

    joint = weather.log_probability(observations, conditioning_state=conditioning)

    expected = weather.temp_gen.log_probability(
        observations[..., 0], conditioning[..., 0]
    ) + weather.precip_gen.log_probability(observations[..., 1], conditioning[..., 1])
    torch.testing.assert_close(joint, expected)


def test_joint_log_probability_needs_both_components() -> None:
    weather = WeatherGenerator(_temperature(), _precipitation())
    with pytest.raises(ValueError, match="air_temperature, precipitation_flux"):
        weather.log_probability(torch.tensor([289.0]))


def test_joint_log_probability_is_not_the_independent_product_under_an_offset() -> None:
    weather = WeatherGenerator(_temperature(wet_day_temperature_offset=-2.0), _precipitation())
    with pytest.raises(NotImplementedError, match="wet_day_temperature_offset"):
        weather.log_probability(torch.tensor([[289.0, 0.0]]))


@pytest.mark.parametrize(
    "p_wet_wet, p_wet_dry, expected",
    [(0.75, 0.25, 0.5), (0.6, 0.2, 1.0 / 3.0), (1.0, 1.0, 1.0)],
)
def test_stationary_wet_probability_of_the_two_state_chain(p_wet_wet, p_wet_dry, expected) -> None:
    """pi_wet = p01 / (1 + p01 - p11) (Wilks 2011, Sect. 9.2.2)."""
    process = MarkovGammaPrecipitation(_precipitation(p_wet_wet=p_wet_wet, p_wet_dry=p_wet_dry))
    assert float(process.moments()["wet_probability"]) == pytest.approx(expected)


def test_chain_without_a_unique_stationary_state_has_no_stationary_moments() -> None:
    """p01 = 0 and p11 = 1 make both states absorbing; no value is substituted."""
    process = MarkovGammaPrecipitation(_precipitation(p_wet_wet=1.0, p_wet_dry=0.0))
    with pytest.raises(ValueError, match="stationary"):
        process.moments()


# ---------------------------------------------------------------------------
# Declared parameters and outputs (MA-21 C7 round 2).
#
# References: Richardson (1981), Water Resour. Res. 17, 182-190 (WGEN:
# temperature conditioned on the wet/dry state); Wilks (2011), Sect. 9.2.
# Mathematical expression: T = T_OU + delta_wet * S, S in {0, 1} the wet state.
# Parameter meanings: delta_wet = wet_day_temperature_offset [K]; T_OU the
# Ornstein-Uhlenbeck temperature (mean_temp is the dry-day mean).
#
# Parameter values come from stochastic/weather_parameters.yaml; the dataclasses
# carry no code defaults. The wet/dry temperature offset is a K offset, never a
# 'correlation'; humidity and VPD had no literature model and are not produced.
# ---------------------------------------------------------------------------


def test_parameter_dataclasses_have_no_code_defaults() -> None:
    with pytest.raises(TypeError):
        TemperatureParameters()
    with pytest.raises(TypeError):
        PrecipitationParameters()


def test_parameters_are_read_from_the_weather_yaml() -> None:
    manager = get_global_manager()
    temperature = TemperatureParameters.from_parameter_manager()
    precipitation = PrecipitationParameters.from_parameter_manager()

    assert temperature.mean_temp == pytest.approx(
        float(manager.get("stochastic.weather.temperature.mean_temp", required=True))
    )
    assert temperature.wet_day_temperature_offset == pytest.approx(
        float(manager.get("stochastic.weather.temperature.wet_day_temperature_offset", required=True))
    )
    assert precipitation.p_wet_wet == pytest.approx(
        float(manager.get("stochastic.weather.precipitation.p_wet_wet", required=True))
    )
    assert precipitation.wet_day_threshold == pytest.approx(
        float(manager.get("stochastic.weather.precipitation.wet_day_threshold", required=True))
    )


def test_correlation_argument_is_gone() -> None:
    with pytest.raises(TypeError):
        WeatherGenerator(_temperature(), _precipitation(), correlation=0.25)


def test_wet_day_offset_shifts_only_wet_day_temperatures() -> None:
    plain = WeatherGenerator(_temperature(wet_day_temperature_offset=0.0), _precipitation())
    shifted = WeatherGenerator(_temperature(wet_day_temperature_offset=-2.0), _precipitation())
    base = plain.sample(torch.tensor(288.15, dtype=torch.float64), num_samples=2, seed=4, num_steps=30)
    moved = shifted.sample(torch.tensor(288.15, dtype=torch.float64), num_samples=2, seed=4, num_steps=30)

    difference = moved["air_temperature"] - base["air_temperature"]
    torch.testing.assert_close(difference, -2.0 * base["precipitation_state"].to(difference.dtype))


def test_moments_report_the_offset_not_a_correlation() -> None:
    temperature = _temperature(wet_day_temperature_offset=-2.0)
    precipitation = _precipitation(p_wet_wet=0.75, p_wet_dry=0.25)
    moments = WeatherGenerator(temperature, precipitation).moments()

    assert "temp_precip_correlation" not in moments
    pi_wet = 0.5
    stationary_var = temperature.volatility**2 / (2.0 * temperature.reversion_rate)
    assert float(moments["temperature_mean"]) == pytest.approx(temperature.mean_temp - 2.0 * pi_wet)
    assert float(moments["temperature_variance"]) == pytest.approx(
        stationary_var + 4.0 * pi_wet * (1.0 - pi_wet)
    )
    assert float(moments["wet_day_temperature_offset"]) == pytest.approx(-2.0)


def test_no_humidity_or_vpd_without_a_literature_model() -> None:
    output = WeatherGenerator(_temperature(), _precipitation()).sample(
        torch.tensor(288.15), num_samples=1, seed=2, num_steps=3
    )
    assert "relative_humidity" not in output
    assert "vapor_pressure_deficit" not in output
    assert "saturation_vapor_pressure" in output


def test_wet_day_threshold_is_the_declared_parameter() -> None:
    observation = torch.tensor([5.0e-9])
    loose = MarkovGammaPrecipitation(_precipitation(wet_day_threshold=1.0e-8))
    strict = MarkovGammaPrecipitation(_precipitation(wet_day_threshold=1.0e-9))
    dry_value = torch.log(torch.tensor(1.0 - loose._require_stationary_wet_probability()))
    torch.testing.assert_close(loose.log_probability(observation)[0], dry_value.to(observation.dtype))
    assert not torch.isclose(strict.log_probability(observation)[0], dry_value.to(observation.dtype))


def test_sample_needs_the_step_count_and_rejects_unknown_options() -> None:
    weather = WeatherGenerator(_temperature(), _precipitation())
    with pytest.raises(TypeError):
        weather.sample(torch.tensor(288.15), num_samples=1, seed=1)
    with pytest.raises(TypeError, match="num_step"):
        weather.sample(torch.tensor(288.15), num_samples=1, seed=1, num_steps=2, num_step=3)


def test_coupled_generator_needs_the_daily_step_of_the_markov_chain() -> None:
    with pytest.raises(ValueError, match="one day"):
        WeatherGenerator(_temperature(dt=0.5), _precipitation())
