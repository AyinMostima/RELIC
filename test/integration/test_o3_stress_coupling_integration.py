"""Integration contract for the canonical ozone-stress photosynthesis route.

BiosphereAtmosphereCoupling owns leaf boundary-layer conductance only. Ozone
stress reaches photosynthesis through the declared ozone_stress_factor state
port rather than a second stomatal or evapotranspiration coupling formula.
"""

import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.photosynthesis.farquhar import (
    FarquharPhotosynthesis,
)
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress


def _canonical_photosynthesis_state() -> dict[str, torch.Tensor]:
    """Build the required canonical Farquhar forcing state."""
    return {
        "air_temperature": torch.tensor(298.15),
        "atmospheric_co2_ppm": torch.tensor(400.0),
        "ppfd": torch.tensor(1000.0),
        "leaf_area_index": torch.tensor(1.0),
        "vapor_pressure_deficit": torch.tensor(1200.0),
    }


def _update_value(output: dict[str, UpdateItem], name: str) -> torch.Tensor:
    """Return one explicit Farquhar output payload."""
    item = output[name]
    assert isinstance(item, UpdateItem)
    return item.value


def test_ozone_stress_factor_drives_canonical_farquhar_port() -> None:
    """Connect the OzoneStress control output to the Farquhar input port."""
    ozone_stress = OzoneStress(name="ozone_stress")
    photosynthesis = FarquharPhotosynthesis(name="farquhar_photosynthesis")

    # References: Emberson et al. (2000); Mills et al. (2011).
    # Expression: F_O3 = eta_uptake * [O3] * g_s / k_O3 and 0 < f_O3 < 1.
    # Parameters: g_s is stomatal conductance and [O3] is ozone mole fraction.
    # Units: g_s [mol m-2 s-1], [O3] [mol mol-1], and f_O3 [1].
    # Route: OzoneStress writes the control consumed by Farquhar.
    ozone_output = ozone_stress.forward(
        {
            "o3_mixing_ratio": torch.tensor(5.0e-9),
            "stomatal_conductance": torch.tensor(0.15),
        },
        time=0.0,
        dt=3600.0,
    )
    ozone_item = ozone_output["ozone_stress_factor"]
    assert isinstance(ozone_item, UpdateItem)
    assert ozone_item.mode is UpdateMode.REPLACE and not ozone_item.per_timestep
    ozone_factor = ozone_item.value
    assert bool(torch.isfinite(ozone_factor).all())
    assert bool(torch.all((ozone_factor > 0.0) & (ozone_factor < 1.0)))
    assert "ozone_stress_factor" in photosynthesis.metadata.input_variables

    unstressed_state = _canonical_photosynthesis_state()
    unstressed_state["ozone_stress_factor"] = torch.tensor(1.0)
    stressed_state = _canonical_photosynthesis_state()
    stressed_state["ozone_stress_factor"] = ozone_factor

    # References: Farquhar et al. (1980); Leuning (1995).
    # Expression: A_gross,stressed = A_gross,potential * f_O3;
    #             A_net = A_gross - R_d.
    # Parameters: f_O3 is the dimensionless external ozone control and R_d is
    #             dark respiration [umol CO2 m-2 s-1].
    # Units: all assimilation and respiration fluxes [umol CO2 m-2 s-1].
    # Route: atmospheric CO2 uptake is an open boundary flux.
    unstressed = photosynthesis.forward(unstressed_state, time=0.0, dt=3600.0)
    stressed = photosynthesis.forward(stressed_state, time=0.0, dt=3600.0)

    gross_unstressed = _update_value(unstressed, "gross_photosynthesis")
    gross_stressed = _update_value(stressed, "gross_photosynthesis")
    net_stressed = _update_value(stressed, "net_photosynthesis")
    respiration_stressed = _update_value(
        stressed,
        "diagnostics.photosynthesis.dark_respiration",
    )

    torch.testing.assert_close(
        gross_stressed,
        gross_unstressed * ozone_factor,
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        net_stressed,
        gross_stressed - respiration_stressed,
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    assert bool(torch.all(gross_stressed < gross_unstressed))
    assert stressed["gross_photosynthesis"].unit == "umol CO2 m-2 s-1"
    assert (
        stressed["gross_photosynthesis"].source_functor
        == "farquhar_photosynthesis"
    )
