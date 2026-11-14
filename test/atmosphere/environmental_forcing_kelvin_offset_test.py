import torch

from relic.data.standards.unit_converter import ConversionContext, unit_converter
from relic.scientific.atmosphere.environmental_forcing import EnvironmentalForcingFunctor
from relic.utils.physical_constants import get_physical_constant


def test_vpd_uses_parameterized_kelvin_offset():
    functor = EnvironmentalForcingFunctor()
    air_temperature = torch.tensor(298.15)
    relative_humidity = torch.tensor(0.5)

    output = functor.forward(
        state={
            "air_temperature": air_temperature,
            "relative_humidity": relative_humidity,
        },
        time=0.0,
        dt=1.0,
    )
    kelvin_offset = torch.as_tensor(get_physical_constant("celsius_to_kelvin"))
    expected_es = functor.es_coefficient * torch.exp(
        functor.vpd_a
        * (air_temperature - kelvin_offset)
        / (air_temperature - kelvin_offset + functor.vpd_b)
    )
    expected_vpd_pa = unit_converter.kpa_to_pa(
        expected_es * (1.0 - relative_humidity),
        ConversionContext(),
    )

    output_vpd = output["vapor_pressure_deficit"].value.to(
        dtype=expected_vpd_pa.dtype
    )
    assert torch.allclose(output_vpd, expected_vpd_pa, atol=1e-5)

