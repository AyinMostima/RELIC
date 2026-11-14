"""No-fallback tests for the leaf boundary-layer coupling."""

import torch

from relic.core.feature_flags import feature_manager
from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.parameters import get_global_manager


def _state() -> dict[str, torch.Tensor]:
    return {
        "friction_velocity": torch.tensor([0.4, 0.8], dtype=torch.float64),
        "air_temperature": torch.tensor([295.0, 305.0], dtype=torch.float64),
        "atmospheric_pressure": torch.tensor([95000.0, 102000.0], dtype=torch.float64),
    }


def _value(
    coupling: BiosphereAtmosphereCoupling,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    return coupling.forward(state, time=0.0, dt=60.0)[
        "boundary_layer_conductance"
    ].value


def test_unrelated_parameter_defaults_cannot_supply_required_state() -> None:
    """Old environmental defaults must not alter the explicit boundary inputs."""
    manager = get_global_manager()
    key = "coupling.scales.defaults.net_radiation_diagnostic_default"
    original = manager.get_parameter(key)
    assert original is not None
    original_value = manager.get(key, required=True)

    try:
        coupling = BiosphereAtmosphereCoupling(
            name="no_forcing_defaults",
            config={},
            parameter_manager=manager,
        )
        baseline = _value(coupling, _state())
        manager.set_value(
            key,
            999.0,
            validate=False,
            notify=False,
        )
        actual = _value(coupling, _state())

        torch.testing.assert_close(actual, baseline, rtol=0.0, atol=0.0)
        assert not hasattr(coupling, "_state_default_factories")
    finally:
        manager.set_value(
            key,
            original_value,
            validate=False,
            notify=False,
        )


def test_old_stomatal_et_and_energy_drivers_do_not_enter_the_equation() -> None:
    coupling = BiosphereAtmosphereCoupling(name="boundary_scope", config={})
    canonical = _state()
    extended = {
        **canonical,
        "net_assimilation_rate": torch.full_like(canonical["friction_velocity"], 99.0),
        "vapor_pressure_deficit": torch.full_like(canonical["friction_velocity"], 5000.0),
        "net_radiation": torch.full_like(canonical["friction_velocity"], 900.0),
        "aerodynamic_resistance": torch.full_like(
            canonical["friction_velocity"], 10.0
        ),
        "co2_concentration": torch.full_like(canonical["friction_velocity"], 80.0),
        "relative_humidity": torch.full_like(canonical["friction_velocity"], 0.2),
    }

    baseline = _value(coupling, canonical)
    actual = _value(coupling, extended)

    torch.testing.assert_close(actual, baseline, rtol=0.0, atol=0.0)


def test_ozone_feature_flag_cannot_change_boundary_layer_conductance() -> None:
    coupling = BiosphereAtmosphereCoupling(name="boundary_no_ozone", config={})
    original_flag = feature_manager.is_enabled("o3_stress_enabled")
    state = {
        **_state(),
        "ozone_stress_factor": torch.tensor([0.1, 0.9], dtype=torch.float64),
    }

    try:
        feature_manager.disable("o3_stress_enabled")
        disabled = _value(coupling, state)
        feature_manager.enable("o3_stress_enabled")
        enabled = _value(coupling, state)
        torch.testing.assert_close(enabled, disabled, rtol=0.0, atol=0.0)
    finally:
        if original_flag:
            feature_manager.enable("o3_stress_enabled")
        else:
            feature_manager.disable("o3_stress_enabled")
