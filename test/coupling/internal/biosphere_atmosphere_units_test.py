import pytest
import torch

from relic.core.functor import ScaleCouplingFunctor
from relic.core.state.types import UpdateItem, UpdateMode
from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.utils.physical_constants import get_physical_constant


INPUT_KEYS = {
    "friction_velocity",
    "air_temperature",
    "atmospheric_pressure",
}
OUTPUT_KEY = "boundary_layer_conductance"


def create_coupling(
    name: str = "biosphere_atmosphere",
    config: dict | None = None,
) -> BiosphereAtmosphereCoupling:
    return BiosphereAtmosphereCoupling(name=name, config=config or {})


def canonical_state(
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        "friction_velocity": torch.tensor(0.5, dtype=dtype, device=device),
        "air_temperature": torch.tensor(298.15, dtype=dtype, device=device),
        "atmospheric_pressure": torch.tensor(101325.0, dtype=dtype, device=device),
    }


def expected_molar_conductance(
    wind_speed: torch.Tensor,
    temperature: torch.Tensor,
    pressure: torch.Tensor,
) -> torch.Tensor:
    transfer_coefficient = wind_speed.new_tensor(0.01)
    leaf_dimension = wind_speed.new_tensor(0.05)
    gas_constant = wind_speed.new_tensor(float(get_physical_constant("gas_constant")))
    conductance_ms = transfer_coefficient * torch.sqrt(wind_speed / leaf_dimension)
    return conductance_ms * pressure / (gas_constant * temperature)


def test_metadata_and_dependency_define_one_boundary_layer_port() -> None:
    coupling = create_coupling()

    assert isinstance(coupling, ScaleCouplingFunctor)
    assert isinstance(coupling, torch.nn.Module)
    assert set(coupling.metadata.input_variables) == INPUT_KEYS
    assert coupling.metadata.output_variables == [OUTPUT_KEY]
    assert set(coupling.metadata.parameters) == {
        "leaf_characteristic_dimension",
        "boundary_layer_transfer_coefficient",
    }
    assert coupling.get_dependencies() == ["atmospheric_turbulence"]


def test_fixed_ctsm_boundary_layer_numerics_and_update_semantics() -> None:
    coupling = create_coupling(name="boundary_exchange")
    state = canonical_state()

    result = coupling.forward(state, time=0.0, dt=60.0)

    assert set(result) == {OUTPUT_KEY}
    update = result[OUTPUT_KEY]
    assert isinstance(update, UpdateItem)
    assert update.mode is UpdateMode.REPLACE
    assert update.per_timestep is False
    assert update.unit == "mol m-2 s-1"
    assert update.source_functor == "boundary_exchange"
    expected = expected_molar_conductance(*state.values())
    torch.testing.assert_close(update.value, expected, rtol=1.0e-12, atol=1.0e-12)
    assert update.value.item() == pytest.approx(1.2925507788001467, rel=1.0e-12)


def test_vectorized_wind_response_preserves_shape_and_square_root_scaling() -> None:
    coupling = create_coupling()
    wind = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
    temperature = torch.full_like(wind, 298.15)
    pressure = torch.full_like(wind, 101325.0)

    result = coupling.forward(
        {
            "friction_velocity": wind,
            "air_temperature": temperature,
            "atmospheric_pressure": pressure,
        },
        time=0.0,
        dt=30.0,
    )[OUTPUT_KEY].value

    expected = expected_molar_conductance(wind, temperature, pressure)
    torch.testing.assert_close(result, expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(result[2:], 2.0 * result[:2], rtol=1.0e-12, atol=1.0e-12)


def test_dtype_device_and_autograd_are_preserved_elementwise() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        coupling = create_coupling()
        wind = torch.tensor([0.4, 0.8], dtype=torch.float64, device=device, requires_grad=True)
        temperature = torch.tensor(
            [290.0, 305.0], dtype=torch.float64, device=device, requires_grad=True
        )
        pressure = torch.tensor(
            [90000.0, 102000.0], dtype=torch.float64, device=device, requires_grad=True
        )
        value = coupling.forward(
            {
                "friction_velocity": wind,
                "air_temperature": temperature,
                "atmospheric_pressure": pressure,
            },
            time=0.0,
            dt=1.0,
        )[OUTPUT_KEY].value

        assert value.dtype is torch.float64
        assert value.device == wind.device
        value.sum().backward()
        assert wind.grad is not None and torch.all(wind.grad > 0.0)
        assert temperature.grad is not None and torch.all(temperature.grad < 0.0)
        assert pressure.grad is not None and torch.all(pressure.grad > 0.0)


@pytest.mark.parametrize("missing_key", sorted(INPUT_KEYS))
def test_missing_required_input_fails_fast(missing_key: str) -> None:
    coupling = create_coupling()
    state = canonical_state()
    state.pop(missing_key)

    with pytest.raises(ValueError, match="Missing required"):
        coupling.forward(state, time=0.0, dt=60.0)


@pytest.mark.parametrize("input_key", sorted(INPUT_KEYS))
@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_inputs_fail_fast(input_key: str, bad_value: float) -> None:
    coupling = create_coupling()
    state = canonical_state()
    state[input_key] = torch.tensor(bad_value, dtype=torch.float64)

    with pytest.raises(ValueError):
        coupling.forward(state, time=0.0, dt=60.0)


def test_non_tensor_nonfloating_and_nonbroadcast_inputs_fail_fast() -> None:
    coupling = create_coupling()
    non_tensor = canonical_state()
    non_tensor["friction_velocity"] = 0.5
    with pytest.raises(TypeError):
        coupling.forward(non_tensor, time=0.0, dt=60.0)

    nonfloating = canonical_state()
    nonfloating["friction_velocity"] = torch.tensor(1, dtype=torch.int64)
    with pytest.raises(TypeError):
        coupling.forward(nonfloating, time=0.0, dt=60.0)

    nonbroadcast = canonical_state()
    nonbroadcast["friction_velocity"] = torch.ones(2, dtype=torch.float64)
    nonbroadcast["air_temperature"] = torch.ones(3, dtype=torch.float64) * 298.15
    with pytest.raises(ValueError, match="broadcast"):
        coupling.forward(nonbroadcast, time=0.0, dt=60.0)


def test_forward_does_not_mutate_state_and_is_dt_invariant() -> None:
    coupling = create_coupling()
    state = canonical_state()
    before = {key: value.clone() for key, value in state.items()}

    one_second = coupling.forward(state, time=0.0, dt=1.0)[OUTPUT_KEY].value
    one_hour = coupling.forward(state, time=0.0, dt=3600.0)[OUTPUT_KEY].value

    torch.testing.assert_close(one_second, one_hour)
    for key, value in state.items():
        torch.testing.assert_close(value, before[key])


def test_only_turbulence_identifier_override_is_supported() -> None:
    coupling = create_coupling(
        config={"upstream_turbulence_functor": "site_turbulence"}
    )
    assert coupling.get_dependencies() == ["site_turbulence"]

    with pytest.raises(ValueError, match="ParameterManager"):
        create_coupling(config={"parameters": {"leaf_characteristic_dimension": 0.1}})


def test_removed_stomatal_et_and_alias_surface_is_absent() -> None:
    coupling = create_coupling()
    for removed_method in (
        "compute_stomatal_conductance",
        "compute_penman_monteith_et",
        "convert_conductance_units",
        "diagnose_conservation",
    ):
        assert not hasattr(coupling, removed_method)

    old_outputs = {
        "stomatal_conductance",
        "stomatal_resistance",
        "canopy_conductance",
        "evapotranspiration_flux",
        "transpiration_rate",
        "latent_heat_flux",
        "sensible_heat_flux",
    }
    result = coupling.forward(canonical_state(), time=0.0, dt=60.0)
    assert old_outputs.isdisjoint(result)
