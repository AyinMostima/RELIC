"""ParameterManager integration for leaf boundary-layer coupling."""

import pytest
import torch

from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.parameters import get_global_manager


NAMESPACE = "coupling.scales.biosphere_atmosphere"
LEAF_KEY = f"{NAMESPACE}.leaf_characteristic_dimension"
TRANSFER_KEY = f"{NAMESPACE}.boundary_layer_transfer_coefficient"


def _state() -> dict[str, torch.Tensor]:
    return {
        "friction_velocity": torch.tensor(0.5, dtype=torch.float64),
        "air_temperature": torch.tensor(298.15, dtype=torch.float64),
        "atmospheric_pressure": torch.tensor(101325.0, dtype=torch.float64),
    }


def _conductance(module: BiosphereAtmosphereCoupling) -> torch.Tensor:
    return module.forward(_state(), time=0.0, dt=60.0)[
        "boundary_layer_conductance"
    ].value


def test_leaf_dimension_is_loaded_from_parameter_manager_at_construction() -> None:
    """The CTSM inverse-square-root leaf-size response must use ParameterManager."""
    manager = get_global_manager()
    snapshot = manager.snapshot()

    try:
        manager.set_value(
            LEAF_KEY,
            0.05,
            units="m",
            validate=True,
            notify=False,
        )
        small_leaf = BiosphereAtmosphereCoupling(
            name="small_leaf_boundary",
            config={},
            parameter_manager=manager,
        )
        small_value = _conductance(small_leaf)

        manager.set_value(
            LEAF_KEY,
            0.2,
            units="m",
            validate=True,
            notify=False,
        )
        large_leaf = BiosphereAtmosphereCoupling(
            name="large_leaf_boundary",
            config={},
            parameter_manager=manager,
        )
        large_value = _conductance(large_leaf)

        torch.testing.assert_close(
            large_value,
            0.5 * small_value,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert small_leaf.leaf_characteristic_dimension.item() == 0.05
        assert large_leaf.leaf_characteristic_dimension.item() == 0.2
    finally:
        manager.restore(snapshot)


def test_constructed_module_has_a_deterministic_parameter_snapshot() -> None:
    """Manager changes affect new modules without mutating an active module."""
    manager = get_global_manager()
    snapshot = manager.snapshot()

    try:
        module = BiosphereAtmosphereCoupling(
            name="boundary_snapshot",
            config={},
            parameter_manager=manager,
        )
        before = _conductance(module)

        manager.set_value(
            LEAF_KEY,
            0.2,
            units="m",
            validate=True,
            notify=False,
        )
        after = _conductance(module)

        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    finally:
        manager.restore(snapshot)


def test_transfer_coefficient_spec_is_complete_and_scientifically_bounded() -> None:
    manager = get_global_manager()
    spec = manager.get_parameter(TRANSFER_KEY)

    assert spec is not None
    assert spec.value == 0.01
    assert spec.units == "m s-0.5"
    assert spec.dtype == "float64"
    assert spec.bounds == (0.01, 0.01)
    assert spec.requires_grad is False
    assert spec.description
    assert "CTSM Technical Note Eqs. 2.5.128 and 2.5.135" in spec.references
    assert {"boundary_layer", "leaf_gas_exchange"} <= spec.tags
    assert spec.provenance["source"] == "CTSM_5_technical_note"


def test_friction_velocity_is_the_only_turbulence_wind_port() -> None:
    """The CTSM bulk closure consumes canonical u_star without a copied alias."""
    module = BiosphereAtmosphereCoupling(config={})

    assert module.metadata.input_variables == [
        "friction_velocity",
        "air_temperature",
        "atmospheric_pressure",
    ]
    assert module.get_dependencies() == ["atmospheric_turbulence"]

    legacy_state = _state()
    legacy_state["canopy_wind_speed"] = legacy_state.pop("friction_velocity")
    with pytest.raises(ValueError, match="friction_velocity"):
        module.forward(legacy_state, time=0.0, dt=60.0)
