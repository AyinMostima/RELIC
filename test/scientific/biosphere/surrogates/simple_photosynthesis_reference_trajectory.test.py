"""L4 analytical benchmarks for simple canopy carbon exchange.

References:
- Baly and Morgan (1934), Nature 133:414.
- Mahecha et al. (2010), Science 329:838-840.
- Chapin et al. (2006), Ecosystems 9:1041-1050.

Mathematical expressions:
- GPP = Gmax PPFD/(PPFD + K) f_T f_w LAI.
- Reco = Rbase Q10^((T - Tref)/10) LAI.
- NEE = Reco - GPP.

Parameter meanings:
- GPP, Reco, and NEE are canopy carbon fluxes [kg C m-2 s-1].
- PPFD and K are photon flux and half-saturation [umol m-2 s-1].
- T, Tref, and Topt are temperatures [K].
- LAI is leaf area index [m2 leaf m-2 ground].
"""

from __future__ import annotations

import math

import pytest
import torch

from relic.core.state import UpdateItem
from relic.scientific.biosphere.surrogates.simple_photosynthesis import (
    SimpleCanopyPhotosynthesis,
)

pytestmark = pytest.mark.scientific

BUFFER_NAMES = (
    "max_gpp_rate",
    "ppfd_half_saturation",
    "temperature_optimum",
    "temperature_width",
    "base_respiration_rate",
    "respiration_q10",
    "soil_moisture_optimum",
    "soil_moisture_width",
    "respiration_reference_temperature",
)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().reshape(-1)[0].cpu().item())


def _reference_state(
    functor: SimpleCanopyPhotosynthesis,
    *,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "ppfd": torch.tensor(
            _scalar(functor.ppfd_half_saturation),
            dtype=dtype,
            requires_grad=requires_grad,
        ),
        "air_temperature": torch.tensor(
            _scalar(functor.temperature_optimum),
            dtype=dtype,
            requires_grad=requires_grad,
        ),
        "leaf_area_index": torch.tensor(
            2.0,
            dtype=dtype,
            requires_grad=requires_grad,
        ),
        "soil_moisture": torch.tensor(
            _scalar(functor.soil_moisture_optimum),
            dtype=dtype,
            requires_grad=requires_grad,
        ),
    }


def _value(updates, key: str) -> torch.Tensor:
    item = updates[key]
    assert isinstance(item, UpdateItem)
    return item.value


def test_scientific_constants_are_persistent_buffers_not_parameters() -> None:
    functor = SimpleCanopyPhotosynthesis(name="simple_canopy_l4_buffers")
    buffers = dict(functor.named_buffers())
    parameters = dict(functor.named_parameters())
    checkpoint = functor.state_dict()

    assert set(BUFFER_NAMES) <= set(buffers)
    assert set(BUFFER_NAMES).isdisjoint(parameters)
    assert set(BUFFER_NAMES) <= set(checkpoint)
    assert all(buffers[name].numel() == 1 for name in BUFFER_NAMES)


def test_dtype_migration_and_checkpoint_roundtrip_preserve_reference_fluxes() -> None:
    source = SimpleCanopyPhotosynthesis(
        name="simple_canopy_l4_checkpoint_source"
    ).to(dtype=torch.float64)
    assert all(
        getattr(source, name).dtype == torch.float64
        for name in BUFFER_NAMES
    )

    with torch.no_grad():
        source.max_gpp_rate.mul_(1.25)
        source.base_respiration_rate.mul_(0.75)
    state = _reference_state(source)
    source_updates = source.forward(state, time=0.0, dt=1.0)
    checkpoint = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
    }

    restarted = SimpleCanopyPhotosynthesis(
        name="simple_canopy_l4_checkpoint_restarted"
    ).to(dtype=torch.float64)
    restarted.load_state_dict(checkpoint)
    restarted_updates = restarted.forward(state, time=0.0, dt=1.0)

    for name in BUFFER_NAMES:
        assert torch.equal(getattr(restarted, name), getattr(source, name))
    for key in (
        "gross_primary_productivity",
        "ecosystem_respiration",
        "net_ecosystem_exchange",
    ):
        assert torch.equal(_value(restarted_updates, key), _value(source_updates, key))


# References:
#   - Baly and Morgan (1934); Mahecha et al. (2010).
# Mathematical expressions at PPFD=K, T=Topt, theta=thetaopt, LAI=2:
#   GPP = Gmax and Reco = 2 Rbase Q10^((Topt-Tref)/10).
#   NEE = Reco - GPP.
# Parameter meanings:
#   K is ppfd_half_saturation and thetaopt is soil_moisture_optimum.
def test_reference_environment_matches_closed_form_carbon_fluxes() -> None:
    functor = SimpleCanopyPhotosynthesis(name="simple_canopy_l4_closed_form").to(
        dtype=torch.float64
    )
    state = _reference_state(functor)
    updates = functor.forward(state, time=0.0, dt=1.0)

    expected_gpp = functor.max_gpp_rate
    expected_respiration = (
        2.0
        * functor.base_respiration_rate
        * torch.pow(
            functor.respiration_q10,
            (
                functor.temperature_optimum
                - functor.respiration_reference_temperature
            )
            / 10.0,
        )
    )
    expected_nee = expected_respiration - expected_gpp

    assert torch.allclose(
        _value(updates, "gross_primary_productivity"),
        expected_gpp,
        rtol=2.0e-12,
        atol=0.0,
    )
    assert torch.allclose(
        _value(updates, "ecosystem_respiration"),
        expected_respiration,
        rtol=2.0e-12,
        atol=0.0,
    )
    assert torch.allclose(
        _value(updates, "net_ecosystem_exchange"),
        expected_nee,
        rtol=2.0e-12,
        atol=0.0,
    )
    assert functor.check_conservation_laws(state, updates)["carbon"] == 0.0


# References:
#   - Baly and Morgan (1934); Mahecha et al. (2010).
# Mathematical expressions:
#   dGPP/dPPFD = Gmax K LAI/(PPFD+K)^2 at f_T=f_w=1.
#   dGPP/dLAI = GPP/LAI and dGPP/dT = 0 at T=Topt.
#   dReco/dT = Reco ln(Q10)/10 and dNEE = dReco - dGPP.
# Parameter meanings:
#   Derivatives retain the units of each flux per driver unit.
def test_autograd_matches_analytical_flux_sensitivities() -> None:
    functor = SimpleCanopyPhotosynthesis(name="simple_canopy_l4_gradient").to(
        dtype=torch.float64
    )
    state = _reference_state(functor, requires_grad=True)
    updates = functor.forward(state, time=0.0, dt=1.0)
    gpp = _value(updates, "gross_primary_productivity")
    respiration = _value(updates, "ecosystem_respiration")
    nee = _value(updates, "net_ecosystem_exchange")
    drivers = (
        state["ppfd"],
        state["leaf_area_index"],
        state["air_temperature"],
    )

    gpp_gradient = torch.autograd.grad(
        gpp,
        drivers,
        retain_graph=True,
    )
    respiration_gradient = torch.autograd.grad(
        respiration,
        drivers,
        retain_graph=True,
        allow_unused=True,
    )
    nee_gradient = torch.autograd.grad(nee, drivers)

    ppfd = state["ppfd"]
    lai = state["leaf_area_index"]
    half_saturation = functor.ppfd_half_saturation
    expected_gpp_ppfd = (
        functor.max_gpp_rate
        * half_saturation
        * lai
        / torch.square(ppfd + half_saturation)
    )
    expected_gpp_lai = gpp / lai
    expected_respiration_lai = respiration / lai
    expected_respiration_temperature = (
        respiration * math.log(_scalar(functor.respiration_q10)) / 10.0
    )

    assert torch.allclose(
        gpp_gradient[0],
        expected_gpp_ppfd,
        rtol=2.0e-10,
        atol=1.0e-18,
    )
    assert torch.allclose(
        gpp_gradient[1],
        expected_gpp_lai,
        rtol=2.0e-10,
        atol=1.0e-18,
    )
    assert gpp_gradient[2].item() == pytest.approx(0.0, abs=1.0e-18)
    assert respiration_gradient[0].item() == pytest.approx(0.0, abs=1.0e-18)
    assert torch.allclose(
        respiration_gradient[1],
        expected_respiration_lai,
        rtol=2.0e-10,
        atol=1.0e-18,
    )
    assert torch.allclose(
        respiration_gradient[2],
        expected_respiration_temperature,
        rtol=2.0e-10,
        atol=1.0e-18,
    )
    for actual, expected in zip(
        nee_gradient,
        (
            -gpp_gradient[0],
            respiration_gradient[1] - gpp_gradient[1],
            respiration_gradient[2] - gpp_gradient[2],
        ),
    ):
        assert torch.allclose(actual, expected, rtol=2.0e-10, atol=1.0e-18)
