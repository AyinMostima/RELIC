"""
Integration contract for surrogate photosynthesis opt-in behavior.
"""

from __future__ import annotations

import logging
import warnings
from unittest.mock import patch

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.core.simulator import RELICSimulator
from relic.parameters import get_global_manager
from relic.scientific.atmosphere.environmental_forcing import (
    EnvironmentalForcingFunctor,
)
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.biosphere import (
    FarquharPhotosynthesis,
    create_standard_biosphere_functors,
)
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
from relic.scientific.biosphere.bvoc.synthesis.isoprene import IsopreneSynthesis
from relic.scientific.biosphere.bvoc.synthesis.monoterpene import (
    MonoterpeneSynthesis,
)
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.phenology import (
    CropPhenology,
    GDDPhenology,
    GSIPhenology,
    PhenologyFunctor,
    StaticPhenology,
)
from relic.scientific.biosphere.surrogates.simple_photosynthesis import (
    SimpleCanopyPhotosynthesis,
)
from relic.utils.physical_constants import get_physical_constant


def _canonical_simple_state(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the canonical tensor-only input state for the surrogate closure."""
    state = {
        "ppfd": torch.tensor(1000.0),
        "air_temperature": torch.tensor(298.15),
        "leaf_area_index": torch.tensor(3.0),
        "soil_moisture": torch.tensor(0.30),
    }
    state.update(overrides)
    return state


def test_default_biosphere_bundle_uses_farquhar_photosynthesis() -> None:
    """
    References:
    - Farquhar et al. (1980), biochemical photosynthesis model.

    Mathematical expression:
    - A = min(Wc, Wj) - Rd.

    Parameter meanings:
    - Wc: Rubisco-limited assimilation.
    - Wj: electron-transport-limited assimilation.
    - Rd: dark respiration.
    """
    functors = create_standard_biosphere_functors()
    assert isinstance(functors["photosynthesis"], FarquharPhotosynthesis)
    assert not isinstance(functors["photosynthesis"], SimpleCanopyPhotosynthesis)
    assert set(functors) >= {
        "maintenance_respiration",
        "growth_respiration",
        "autotrophic_respiration",
    }
    assert functors["growth_respiration"].get_dependencies() == [
        "farquhar_photosynthesis",
        "maintenance_respiration",
    ]
    assert functors["autotrophic_respiration"].get_dependencies() == [
        "maintenance_respiration",
        "growth_respiration",
    ]
    assert isinstance(functors["gsi_phenology"], GSIPhenology)
    assert isinstance(functors["phenology_diagnostics"], PhenologyFunctor)
    assert "phenology_growth" not in functors
    assert functors["phenology_diagnostics"].get_dependencies() == {
        "required": ["gsi_phenology"],
        "optional": [],
    }
    assert functors["photosynthesis"].get_dependencies() == [
        "environmental_forcing",
        "atmospheric_radiation",
        "gsi_phenology",
    ]


@pytest.mark.parametrize(
    ("selected_owner", "expected_type"),
    [
        ("static_phenology", StaticPhenology),
        ("gdd_phenology", GDDPhenology),
        ("gsi_phenology", GSIPhenology),
        ("crop_phenology", CropPhenology),
    ],
)
def test_biosphere_bundle_selects_exactly_one_canonical_phenology_owner(
    selected_owner: str,
    expected_type: type,
) -> None:
    """Compile-time selection must not construct alternative phenology models."""

    functors = create_standard_biosphere_functors(
        phenology_model_owner=selected_owner,
        include_growth=True,
        include_phenology_growth_diagnostic=True,
    )

    canonical_owners = {
        "static_phenology",
        "gdd_phenology",
        "gsi_phenology",
        "crop_phenology",
    }
    assert set(functors).intersection(canonical_owners) == {selected_owner}
    assert isinstance(functors[selected_owner], expected_type)
    assert functors["phenology_diagnostics"].canonical_model_owner == selected_owner
    assert functors["phenology_diagnostics"].get_dependencies() == {
        "required": [selected_owner],
        "optional": [],
    }
    activity = functors["phenology_growth"]
    assert isinstance(activity, PhenologyGrowth)
    assert activity.get_dependencies() == [selected_owner]
    assert activity.metadata.input_variables == ["phenological_factor"]
    assert activity.metadata.output_variables == [
        "diagnostics.phenology.growth_active"
    ]
    allocation_dependencies = set(
        functors["biomass_allocation"].get_dependencies()
    )
    assert allocation_dependencies.intersection(canonical_owners) == {
        selected_owner
    }
    for functor in functors.values():
        outputs = set(functor.metadata.output_variables)
        assert "biosphere.storage_carbon" not in outputs
        assert "biosphere.litter_carbon" not in outputs


def test_biosphere_bundle_rejects_unknown_owner_before_model_construction() -> None:
    """Invalid graph configuration must fail before scientific initialization."""

    with patch(
        "relic.scientific.biosphere.FarquharPhotosynthesis"
    ) as photosynthesis_constructor:
        with pytest.raises(ValueError, match="Unsupported canonical"):
            create_standard_biosphere_functors(
                phenology_model_owner="legacy_runtime_switch"
            )

    photosynthesis_constructor.assert_not_called()


def test_surrogate_photosynthesis_requires_explicit_opt_in() -> None:
    """
    References:
    - RELIC architecture rule for experimental surrogate process isolation.

    Mathematical expression:
    - N/A (bundle composition gate).

    Parameter meanings:
    - use_surrogate_photosynthesis: explicit switch for experimental surrogate.
    """
    functors = create_standard_biosphere_functors(use_surrogate_photosynthesis=True)
    assert isinstance(functors["photosynthesis"], SimpleCanopyPhotosynthesis)
    assert functors["photosynthesis"].name == "simple_photosynthesis"
    assert {
        "maintenance_respiration",
        "growth_respiration",
        "autotrophic_respiration",
        "biomass_allocation",
    }.isdisjoint(functors)
    assert functors["photosynthesis"].get_dependencies() == [
        "atmospheric_radiation",
        "gsi_phenology",
    ]


def test_surrogate_photosynthesis_rejects_standard_growth_closure() -> None:
    """A self-contained surrogate cannot feed the standard NPP growth chain."""

    with patch(
        "relic.scientific.biosphere.surrogates.simple_photosynthesis.SimpleCanopyPhotosynthesis"
    ) as surrogate_constructor:
        with pytest.raises(
            ValueError,
            match="surrogate photosynthesis.*standard growth closure",
        ):
            create_standard_biosphere_functors(
                use_surrogate_photosynthesis=True,
                include_growth=True,
            )

    surrogate_constructor.assert_not_called()


def test_surrogate_photosynthesis_rejects_default_bvoc_closure() -> None:
    """The default MEGAN graph requires Farquhar-only photosynthesis ports."""

    with patch(
        "relic.scientific.biosphere.surrogates.simple_photosynthesis.SimpleCanopyPhotosynthesis"
    ) as surrogate_constructor:
        with pytest.raises(
            ValueError,
            match="surrogate photosynthesis.*default BVOC closure",
        ):
            create_standard_biosphere_functors(
                use_surrogate_photosynthesis=True,
                include_bvoc=True,
            )

    surrogate_constructor.assert_not_called()


def test_farquhar_bundle_constructs_default_bvoc_closure() -> None:
    """The default MEGAN closure must bind to the selected Farquhar owner."""

    functors = create_standard_biosphere_functors(include_bvoc=True)

    assert isinstance(functors["megan_bvoc_model"], MEGANModel)
    assert "bvoc_megan" not in functors
    assert functors["photosynthesis"].name == "farquhar_photosynthesis"
    assert "farquhar_photosynthesis" in functors[
        "megan_bvoc_model"
    ].get_dependencies()


def test_farquhar_bundle_selects_mechanistic_monoterpene_closure() -> None:
    """Compile-time selection must construct exactly one BVOC model owner."""

    functors = create_standard_biosphere_functors(
        include_bvoc=True,
        bvoc_model_owner="mechanistic_monoterpene_synthesis",
    )

    candidate_owners = {
        "megan_bvoc_model",
        "mechanistic_monoterpene_synthesis",
    }
    assert set(functors).intersection(candidate_owners) == {
        "mechanistic_monoterpene_synthesis"
    }
    selected = functors["mechanistic_monoterpene_synthesis"]
    assert isinstance(selected, MonoterpeneSynthesis)
    assert selected.get_dependencies() == ["farquhar_photosynthesis"]
    assert functors["photosynthesis"].name == "farquhar_photosynthesis"


def test_farquhar_bundle_selects_mechanistic_isoprene_closure() -> None:
    """Compile-time selection must construct only the requested isoprene owner."""

    functors = create_standard_biosphere_functors(
        include_bvoc=True,
        bvoc_model_owner="mechanistic_isoprene_synthesis",
    )

    candidate_owners = {
        "megan_bvoc_model",
        "mechanistic_isoprene_synthesis",
        "mechanistic_monoterpene_synthesis",
    }
    assert set(functors).intersection(candidate_owners) == {
        "mechanistic_isoprene_synthesis"
    }
    selected = functors["mechanistic_isoprene_synthesis"]
    assert isinstance(selected, IsopreneSynthesis)
    assert selected.get_dependencies() == ["farquhar_photosynthesis"]
    assert functors["photosynthesis"].name == "farquhar_photosynthesis"


@pytest.mark.parametrize(
    "selected_owner",
    [
        "megan_bvoc_model",
        "mechanistic_isoprene_synthesis",
        "mechanistic_monoterpene_synthesis",
    ],
)
def test_growth_bundle_consumes_the_selected_bvoc_carbon_owner(
    selected_owner: str,
) -> None:
    """One compile-time BVOC owner must feed the unique plant-carbon ledger."""
    functors = create_standard_biosphere_functors(
        include_growth=True,
        include_bvoc=True,
        bvoc_model_owner=selected_owner,
    )
    allocation = functors["biomass_allocation"]

    assert allocation.bvoc_carbon_allocation_owner == selected_owner
    assert selected_owner in allocation.get_dependencies()
    assert "bvoc_carbon_allocation_flux" in allocation.get_required_inputs()
    assert "bvoc_carbon_allocation_flux" in allocation.metadata.input_variables


def test_biosphere_bundle_rejects_unknown_bvoc_owner_before_construction() -> None:
    """Invalid BVOC graph selection must fail before scientific construction."""

    with patch(
        "relic.scientific.biosphere.FarquharPhotosynthesis"
    ) as photosynthesis_constructor:
        with patch(
            "relic.scientific.biosphere.bvoc.megan_model.MEGANModel"
        ) as megan_constructor:
            with patch(
                "relic.scientific.biosphere.bvoc.synthesis.monoterpene."
                "MonoterpeneSynthesis"
            ) as monoterpene_constructor:
                with pytest.raises(ValueError, match="Unsupported BVOC model owner"):
                    create_standard_biosphere_functors(
                        include_bvoc=True,
                        bvoc_model_owner="runtime_fallback",
                    )

    photosynthesis_constructor.assert_not_called()
    megan_constructor.assert_not_called()
    monoterpene_constructor.assert_not_called()


def test_simple_canopy_exposes_only_mass_based_carbon_fluxes() -> None:
    """The canopy surrogate must not masquerade mass GPP as molar assimilation."""
    functor = SimpleCanopyPhotosynthesis()
    output = functor.forward(
        _canonical_simple_state(),
        time=0.0,
        dt=60.0,
    )

    state_outputs = {key for key in output if not key.startswith("diagnostics.")}
    assert state_outputs == {
        "gross_primary_productivity",
        "ecosystem_respiration",
        "net_ecosystem_exchange",
    }
    assert {"gross_photosynthesis", "net_photosynthesis"}.isdisjoint(output)
    assert {
        "diagnostics.photosynthesis.gpp",
        "diagnostics.photosynthesis.respiration",
    }.isdisjoint(output)
    assert set(output) == set(functor.metadata.output_variables)
    assert {
        "diagnostics.photosynthesis.water_stress",
        "diagnostics.photosynthesis.temperature_factor",
    }.issubset(output)

    gpp = output["gross_primary_productivity"].value
    respiration = output["ecosystem_respiration"].value
    nee = output["net_ecosystem_exchange"].value
    # References: Baly and Morgan (1934); Mahecha et al. (2010).
    # Mathematical expression: GPP = G_max Q/(Q+K) exp(-((theta-theta_opt)/w)^2) LAI
    #   and Reco = R_base LAI at T = T_opt = T_ref; NEE = Reco - GPP.
    # Parameter meanings: G_max = 3.0e-7 and R_base = 4.8e-8 are the default
    #   rates per unit leaf area index; Q = 1000, K = 600, theta = 0.30,
    #   theta_opt = 0.35, w = 0.15, LAI = 3.
    # Units: rates [kg C m-2 s-1]; Q, K [umol photons m-2 s-1]; theta [m3 m-3].
    # Domain/constraints: 5.0335e-7 kg C m-2 s-1 is 41.9 umol CO2 m-2 s-1.
    # Sign convention: GPP and Reco positive; NEE positive toward the atmosphere.
    assert torch.allclose(gpp, torch.tensor(5.0334713e-07), rtol=1.0e-6, atol=0.0)
    assert torch.allclose(respiration, torch.tensor(1.44e-07), rtol=1.0e-6, atol=0.0)
    assert torch.allclose(nee, respiration - gpp, rtol=1.0e-6, atol=0.0)
    assert torch.allclose(nee, torch.tensor(-3.5934713e-07), rtol=1.0e-6, atol=0.0)

    for key in state_outputs:
        item = output[key]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == "kg C m-2 s-1"

    assert functor.metadata.input_variables == [
        "ppfd",
        "air_temperature",
        "leaf_area_index",
        "soil_moisture",
    ]
    parameters = functor.metadata.parameters
    assert "biosphere.photosynthesis.surrogate.ppfd_half_saturation" in parameters
    assert not any(
        key.rsplit(".", 1)[-1] == "par_half_saturation" for key in parameters
    )
    assert functor.get_dependencies() == ["atmospheric_radiation"]


# References: relic/parameters/database/relic/validation/conservation.yaml entry
#   validation.conservation.plausibility.gross_primary_productivity_max (owner
#   plausibility limit of the instantaneous canopy GPP, about 83 umol CO2 m-2 s-1).
# Mathematical expression: 0 < GPP <= GPP_max at the canonical daytime state, and
#   0 < lower < default < upper <= GPP_max for G_max and R_base.
# Parameter meanings: GPP_max is the owner limit; G_max = max_gpp_rate and
#   R_base = base_respiration_rate are rates per unit leaf area index; lower and
#   upper are their owner bounds.
# Units: GPP_max [g C m-2 s-1], converted with 1e-3 kg g-1; rates and bounds
#   [kg C m-2 s-1].
# Domain/constraints: default parameters; Q = 1000 umol photons m-2 s-1,
#   T = 298.15 K, LAI = 3, theta = 0.30 m3 m-3.
# Sign convention: GPP and Reco are positive magnitudes.
def test_simple_canopy_default_rates_respect_owner_plausibility_limit() -> None:
    """Default rates and their bounds must stay within the canopy GPP limit."""
    manager = get_global_manager()
    limit_key = "validation.conservation.plausibility.gross_primary_productivity_max"
    assert manager.get_parameter(limit_key).units == "g C m-2 s-1"
    gpp_limit = float(manager.get(limit_key, required=True)) * 1.0e-3

    functor = SimpleCanopyPhotosynthesis()
    output = functor.forward(
        _canonical_simple_state(),
        time=0.0,
        dt=60.0,
    )
    gpp = output["gross_primary_productivity"].value
    assert torch.all(gpp > 0.0)
    assert torch.all(gpp <= gpp_limit)

    for name in ("max_gpp_rate", "base_respiration_rate"):
        spec = manager.get_parameter(f"biosphere.photosynthesis.surrogate.{name}")
        lower, upper = spec.bounds
        assert spec.units == "kg C m-2 s-1"
        assert 0.0 < lower < float(getattr(functor, name)) < upper <= gpp_limit


# References: Mahecha et al. (2010), Science 329:838-840 (Q10 of ecosystem
#   respiration against air temperature 1.4 +/- 0.1 across 60 FLUXNET sites).
# Mathematical expression: Reco(T_ref + 10 K) / Reco(T_ref) = Q10 = 1.4, and
#   1.0 <= lower <= 1.3 and 1.5 <= upper for the owner bounds of Q10.
# Parameter meanings: Q10 = respiration_q10; T_ref is the physical-constant
#   reference temperature; lower and upper are the owner bounds.
# Units: Reco [kg C m-2 s-1]; temperatures [K]; Q10 and bounds [1].
# Domain/constraints: default parameters; Q10 = 1 is no temperature response.
# Sign convention: Reco is a positive magnitude that does not fall with T.
def test_simple_canopy_respiration_q10_matches_cited_ecosystem_value() -> None:
    """The default Q10 must be the value of the source cited for it."""
    functor = SimpleCanopyPhotosynthesis()
    reference_temperature = float(get_physical_constant("reference_temperature"))

    def respiration(air_temperature: float) -> torch.Tensor:
        output = functor.forward(
            _canonical_simple_state(air_temperature=torch.tensor(air_temperature)),
            time=0.0,
            dt=60.0,
        )
        return output["ecosystem_respiration"].value

    ratio = respiration(reference_temperature + 10.0) / respiration(
        reference_temperature
    )
    assert float(ratio) == pytest.approx(1.4, rel=1.0e-6)

    lower, upper = get_global_manager().get_parameter(
        "biosphere.photosynthesis.surrogate.respiration_q10"
    ).bounds
    assert 1.0 <= lower <= 1.3
    assert upper >= 1.5


def test_simple_canopy_rejects_retired_par_port() -> None:
    """The retired energy/light alias must not satisfy the required PPFD port."""
    functor = SimpleCanopyPhotosynthesis()
    state = _canonical_simple_state()
    state.pop("ppfd")
    state["par"] = torch.tensor(1000.0)

    with pytest.raises(KeyError, match="ppfd"):
        functor.forward(state, time=0.0, dt=60.0)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("ppfd", torch.tensor(-1.0)),
        ("air_temperature", torch.tensor(0.0)),
        ("leaf_area_index", torch.tensor(-0.01)),
        ("soil_moisture", torch.tensor(-0.01)),
        ("soil_moisture", torch.tensor(1.01)),
        ("ppfd", torch.tensor(float("nan"))),
        ("air_temperature", torch.tensor(float("inf"))),
        ("leaf_area_index", torch.tensor(float("nan"))),
        ("soil_moisture", torch.tensor(float("nan"))),
    ],
)
def test_simple_canopy_rejects_invalid_scientific_inputs(
    key: str,
    bad_value: torch.Tensor,
) -> None:
    """Invalid domains must fail instead of being clamped or propagated."""
    functor = SimpleCanopyPhotosynthesis()

    with pytest.raises(ValueError, match=key):
        functor.forward(
            _canonical_simple_state(**{key: bad_value}),
            time=0.0,
            dt=60.0,
        )


def test_simple_canopy_requires_floating_tensor_inputs() -> None:
    """Scientific state inputs must remain tensor-valued and differentiable."""
    functor = SimpleCanopyPhotosynthesis()

    with pytest.raises(TypeError, match="ppfd"):
        functor.forward(
            _canonical_simple_state(ppfd=torch.tensor(1000)),
            time=0.0,
            dt=60.0,
        )


def test_simple_canopy_zero_lai_has_exact_zero_canopy_fluxes() -> None:
    """A canopy with zero leaf area cannot assimilate or respire canopy carbon."""
    functor = SimpleCanopyPhotosynthesis()
    output = functor.forward(
        _canonical_simple_state(leaf_area_index=torch.tensor(0.0)),
        time=0.0,
        dt=60.0,
    )

    for key in (
        "gross_primary_productivity",
        "ecosystem_respiration",
        "net_ecosystem_exchange",
    ):
        assert torch.equal(output[key].value, torch.zeros_like(output[key].value))


def test_simple_canopy_uses_physical_constant_ssot() -> None:
    """Reference temperature must come through the physical-constant API."""
    target = (
        "relic.scientific.biosphere.surrogates.simple_photosynthesis."
        "get_physical_constant"
    )
    with patch(target, return_value=torch.tensor(301.15)) as getter:
        functor = SimpleCanopyPhotosynthesis()

    getter.assert_called_once()
    assert getter.call_args.args == ("reference_temperature",)
    assert getter.call_args.kwargs["as_tensor"] is True
    assert getter.call_args.kwargs["device"] == functor.device
    assert torch.equal(
        functor.respiration_reference_temperature,
        getter.return_value.to(device=functor.device, dtype=torch.float32),
    )


def test_simple_canopy_local_carbon_ledger_uses_nee_orientation() -> None:
    """Positive-to-atmosphere NEE must close exactly as Reco minus GPP."""
    functor = SimpleCanopyPhotosynthesis()
    output = functor.forward(
        _canonical_simple_state(),
        time=0.0,
        dt=60.0,
    )

    assert functor.check_conservation_laws(_canonical_simple_state(), output) == {
        "carbon": 0.0
    }


def test_radiation_to_simple_canopy_real_simulator_chain() -> None:
    """The canonical PPFD writer must drive one surrogate carbon closure."""
    dt = 60.0
    simulator = RELICSimulator(
        {
            "max_time": dt,
            "time_step": dt,
            "device": "cpu",
            "conservation_checks": True,
            "strict_dependencies": True,
            "strict_scale_detection": False,
        }
    )
    simulator.initialize_state(
        {
            "air_temperature": torch.tensor(298.15),
            "incoming_shortwave_potential": torch.tensor(1000.0),
            "cos_zenith": torch.tensor(0.8),
            "surface_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.6),
            "surface_albedo": torch.tensor(0.2),
            "surface_emissivity": torch.tensor(0.97),
            "cloud_fraction": torch.tensor(0.2),
            "leaf_area_index": torch.tensor(3.0),
            "soil_moisture": torch.tensor(0.30),
        }
    )
    simulator.register_functor(
        RadiationFunctor(name="atmospheric_radiation", device=torch.device("cpu")),
        execution_group="fast",
    )
    simulator.register_functor(
        SimpleCanopyPhotosynthesis(device=torch.device("cpu")),
        execution_group="fast",
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)

    assert captured == []
    ppfd = simulator.state_manager.get_state("ppfd")
    gpp = simulator.state_manager.get_state("gross_primary_productivity")
    respiration = simulator.state_manager.get_state("ecosystem_respiration")
    nee = simulator.state_manager.get_state("net_ecosystem_exchange")
    assert torch.allclose(
        ppfd,
        torch.tensor(1043.2799, dtype=ppfd.dtype, device=ppfd.device),
        rtol=1.0e-6,
        atol=0.0,
    )
    assert torch.allclose(
        gpp,
        torch.tensor(5.1130129e-07, dtype=gpp.dtype, device=gpp.device),
        rtol=1.0e-6,
        atol=0.0,
    )
    carbon_residual = torch.abs(nee - (respiration - gpp))
    rounding_bound = torch.finfo(nee.dtype).eps * torch.maximum(
        torch.abs(gpp),
        torch.abs(respiration),
    )
    assert torch.all(carbon_residual <= rounding_bound)
    applied_gpp = updates["gross_primary_productivity"]
    if isinstance(applied_gpp, UpdateItem):
        applied_gpp = applied_gpp.value
    assert torch.allclose(applied_gpp, gpp, rtol=1.0e-6, atol=0.0)

    ppfd_writers = {
        item["functor"]
        for item in simulator.get_variable_provenance("ppfd")
        if item.get("role") == "writer"
    }
    gpp_writers = {
        item["functor"]
        for item in simulator.get_variable_provenance("gross_primary_productivity")
        if item.get("role") == "writer"
    }
    assert ppfd_writers == {"atmospheric_radiation"}
    assert gpp_writers == {"simple_photosynthesis"}
    assert simulator.get_effective_device("atmospheric_radiation").type == "cpu"
    assert simulator.get_effective_device("simple_photosynthesis").type == "cpu"


def test_farquhar_to_mechanistic_monoterpene_real_simulator_chain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The selected phenology and BVOC owners must form one routed real DAG.

    References:
    - Farquhar et al. (1980), biochemical photosynthesis.
    - Thornton et al. (2002), vegetation turnover and litter export.
    - Niinemets and Reichstein (2002), stored monoterpene emissions.

    Mathematical expressions:
    - delta C_veg + dt * F_litter,C = 0.
    - delta N_veg + delta N_store + dt * F_litter,N = 0.
    - delta S = dt * (J_syn - J_emit - J_deg).

    Parameter meanings:
    - dt: integration interval [s].
    - F_litter: explicit carbon or nitrogen boundary rate [kg element m-2 s-1].
    - S: monoterpene storage pool [umol m-2].
    """
    dt = 60.0
    caplog.set_level(logging.WARNING)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        bundle = create_standard_biosphere_functors(
            include_bvoc=True,
            include_phenology_growth_diagnostic=True,
            bvoc_model_owner="mechanistic_monoterpene_synthesis",
            phenology_model_owner="static_phenology",
        )
        monoterpene = bundle["mechanistic_monoterpene_synthesis"]
        simulator = RELICSimulator(
            {
                "max_time": dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": True,
                "strict_dependencies": True,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize_state(
            {
                "air_temperature": torch.tensor(298.15),
                "relative_humidity": torch.tensor(0.6),
                "incoming_shortwave_potential": torch.tensor(1000.0),
                "cos_zenith": torch.tensor(0.8),
                "surface_temperature": torch.tensor(298.15),
                "surface_albedo": torch.tensor(0.2),
                "surface_emissivity": torch.tensor(0.97),
                "cloud_fraction": torch.tensor(0.2),
                "day_of_year": torch.tensor(180.0),
                "leaf_carbon": torch.tensor(0.30),
                "froot_carbon": torch.tensor(0.25),
                "leaf_nitrogen": torch.tensor(0.012),
                "froot_nitrogen": torch.tensor(0.009),
                "plant_nitrogen_pool": torch.tensor(0.0),
                "atmospheric_co2_ppm": torch.tensor(400.0),
                "monoterpene_pool_concentration": monoterpene.reset_pools(
                    torch.tensor(50.0)
                ),
            }
        )
        for functor in (
            EnvironmentalForcingFunctor(device=torch.device("cpu")),
            RadiationFunctor(device=torch.device("cpu")),
            bundle["static_phenology"],
            bundle["phenology_growth"],
            bundle["photosynthesis"],
            monoterpene,
        ):
            simulator.register_functor(functor, execution_group="fast")
        updates = simulator.step(dt=dt)

    assert captured == []
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []

    state = simulator.state_manager.get_state()
    expected_values = {
        "vapor_pressure_deficit": 1267.1112060546875,
        "ppfd": 1043.2799072265625,
        "leaf_area_index": 6.0,
        "diagnostics.phenology.growth_active": True,
    }
    for key, expected in expected_values.items():
        actual = updates[key] if key.startswith("diagnostics.") else state[key]
        if actual.dtype is torch.bool:
            assert bool(actual.item()) is expected
            continue
        assert torch.allclose(
            actual,
            torch.as_tensor(expected, dtype=actual.dtype, device=actual.device),
            rtol=1.0e-6,
            atol=0.0,
        ), f"{key}: {actual.detach().cpu().tolist()} != {expected}"

    gross_photosynthesis = state["gross_photosynthesis"]
    gpp = state["gross_primary_productivity"]
    carbon_molar_mass = torch.as_tensor(
        get_physical_constant(
            "relic.physical.constants.molecular_weights.carbon"
        ),
        dtype=gpp.dtype,
        device=gpp.device,
    )
    torch.testing.assert_close(
        gpp,
        gross_photosynthesis * carbon_molar_mass * 1.0e-9,
        rtol=1.0e-6,
        atol=0.0,
    )
    assert torch.isfinite(gross_photosynthesis)
    assert gross_photosynthesis > 0.0
    assert state["monoterpenes_emission"] > 0.0
    assert state["monoterpene_pool_concentration"] > 50.0
    assert 0.0 < state["bvoc_carbon_allocation_flux"] < gpp
    assert simulator.state_manager._current_diagnostics[
        "phenology_growth.phenology.growth_active"
    ] == 1.0

    carbon_residual = (
        state["leaf_carbon"]
        - torch.as_tensor(0.30, dtype=state["leaf_carbon"].dtype)
        + state["froot_carbon"]
        - torch.as_tensor(0.25, dtype=state["froot_carbon"].dtype)
        + dt
        * (
            state["leaf_litterfall_carbon"]
            + state["froot_litterfall_carbon"]
        )
    )
    nitrogen_residual = (
        state["leaf_nitrogen"]
        - torch.as_tensor(0.012, dtype=state["leaf_nitrogen"].dtype)
        + state["froot_nitrogen"]
        - torch.as_tensor(0.009, dtype=state["froot_nitrogen"].dtype)
        + state["plant_nitrogen_pool"]
        + dt
        * (
            state["leaf_litterfall_nitrogen"]
            + state["froot_litterfall_nitrogen"]
        )
    )
    carbon_scale = torch.stack(
        (
            torch.abs(state["leaf_carbon"] - 0.30),
            torch.abs(state["froot_carbon"] - 0.25),
            torch.abs(
                dt
                * (
                    state["leaf_litterfall_carbon"]
                    + state["froot_litterfall_carbon"]
                )
            ),
        )
    ).max()
    nitrogen_scale = torch.stack(
        (
            torch.abs(state["leaf_nitrogen"] - 0.012),
            torch.abs(state["froot_nitrogen"] - 0.009),
            torch.abs(state["plant_nitrogen_pool"]),
            torch.abs(
                dt
                * (
                    state["leaf_litterfall_nitrogen"]
                    + state["froot_litterfall_nitrogen"]
                )
            ),
        )
    ).max()
    float32_rounding = torch.finfo(torch.float32).eps
    assert torch.abs(carbon_residual) <= 8.0 * float32_rounding * carbon_scale
    assert torch.abs(nitrogen_residual) <= 8.0 * float32_rounding * nitrogen_scale

    expected_units = {
        "vapor_pressure_deficit": "Pa",
        "ppfd": "umol photons m-2 s-1",
        "diagnostics.phenology.growth_active": "dimensionless",
        "gross_photosynthesis": "umol CO2 m-2 s-1",
        "gross_primary_productivity": "kg C m-2 s-1",
        "monoterpenes_emission": "mol m-2 s-1",
        "monoterpene_pool_concentration": "umol m-2",
        "bvoc_carbon_allocation_flux": "kg C m-2 s-1",
    }
    registry = simulator.state_manager._contract_registry
    assert registry is not None
    for key, unit in expected_units.items():
        assert isinstance(updates[key], torch.Tensor)
        contract = registry.get_contract(key)
        assert contract is not None
        assert contract.udunits == unit

    expected_writers = {
        "vapor_pressure_deficit": "environmental_forcing",
        "ppfd": "atmospheric_radiation",
        "leaf_area_index": "static_phenology",
        "gross_photosynthesis": "farquhar_photosynthesis",
        "monoterpenes_emission": "mechanistic_monoterpene_synthesis",
        "monoterpene_pool_concentration": "mechanistic_monoterpene_synthesis",
        "bvoc_carbon_allocation_flux": "mechanistic_monoterpene_synthesis",
    }
    for key, expected_writer in expected_writers.items():
        writers = {
            record["functor"]
            for record in simulator.get_variable_provenance(key)
            if record.get("role") in {"writer", "final_writer"}
        }
        assert writers == {expected_writer}, f"{key}: {writers}"

    for functor_name in (
        "environmental_forcing",
        "atmospheric_radiation",
        "static_phenology",
        "phenology_growth",
        "farquhar_photosynthesis",
        "mechanistic_monoterpene_synthesis",
    ):
        assert simulator.get_effective_device(functor_name).type == "cpu"


def test_farquhar_bvoc_growth_real_simulator_closes_structural_carbon_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run the selected BVOC closure through the canonical biomass owner.

    References:
    - Song and Song (2022), 3-PG2Py biomass partitioning.
    - Kesselmeier et al. (2002), BVOC loss from fixed plant carbon.

    Mathematical expressions:
    - NPP = GPP - R_a.
    - NPP = F_BVOC,C + F_leaf + F_stem + F_root.
    - Delta B_i = F_i * dt.

    Parameter meanings:
    - F_BVOC,C: selected BVOC carbon allocation flux [kg C m-2 s-1].
    - F_i: structural organ growth flux [kg C m-2 s-1].
    - B_i: structural organ carbon stock [kg C m-2].
    - dt: simulator step [s].

    Open-system route:
    - The three BiomassAllocation growth ports are the explicit carbon inputs
      to the tracked structural biomass stocks.
    """
    dt = 60.0

    def f32(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32)

    caplog.set_level(logging.WARNING)
    bundle = create_standard_biosphere_functors(
        include_growth=True,
        include_bvoc=True,
        bvoc_model_owner="mechanistic_monoterpene_synthesis",
        phenology_model_owner="static_phenology",
    )
    monoterpene = bundle["mechanistic_monoterpene_synthesis"]
    simulator = RELICSimulator(
        {
            "max_time": dt,
            "time_step": dt,
            "device": "cpu",
            "conservation_checks": True,
            "strict_conservation_enabled": True,
            "strict_dependencies": True,
            "strict_scale_detection": False,
        }
    )
    simulator.initialize_state(
        {
            "air_temperature": f32(298.15),
            "relative_humidity": f32(0.6),
            "incoming_shortwave_potential": f32(1000.0),
            "cos_zenith": f32(0.8),
            "surface_temperature": f32(298.15),
            "surface_albedo": f32(0.2),
            "surface_emissivity": f32(0.97),
            "cloud_fraction": f32(0.2),
            "day_of_year": f32(180.0),
            "leaf_carbon": f32(0.30),
            "froot_carbon": f32(0.25),
            "leaf_nitrogen": f32(0.012),
            "froot_nitrogen": f32(0.009),
            "plant_nitrogen_pool": f32(0.0),
            "atmospheric_co2_ppm": f32(400.0),
            "monoterpene_pool_concentration": monoterpene.reset_pools(
                f32(50.0)
            ),
            "soil_water_potential": f32(-0.5),
            "biomass_leaf": f32(200.0),
            "biomass_stem": f32(500.0),
            "biomass_root": f32(150.0),
            "soluble_sugar": f32(50.0),
            "plant_nitrogen": f32(2.0),
            "plant_phosphorus": f32(0.2),
            "leaf_biomass_c": f32(0.20),
            "wood_biomass_c": f32(0.50),
            "root_biomass_c": f32(0.15),
        }
    )
    initial_biomass = {
        name: simulator.state_manager.get_state(name).clone()
        for name in ("leaf_biomass_c", "wood_biomass_c", "root_biomass_c")
    }
    selected_functors = (
        EnvironmentalForcingFunctor(device=torch.device("cpu")),
        RadiationFunctor(device=torch.device("cpu")),
        bundle["static_phenology"],
        bundle["water_stress"],
        bundle["maintenance_respiration"],
        bundle["photosynthesis"],
        bundle["growth_respiration"],
        monoterpene,
        bundle["autotrophic_respiration"],
        bundle["biomass_allocation"],
    )
    for functor in selected_functors:
        simulator.register_functor(functor, execution_group="fast")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)

    assert captured == []
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert simulator._get_execution_order() == [
        "environmental_forcing",
        "atmospheric_radiation",
        "static_phenology",
        "water_stress",
        "maintenance_respiration",
        "farquhar_photosynthesis",
        "growth_respiration",
        "mechanistic_monoterpene_synthesis",
        "autotrophic_respiration",
        "biomass_allocation",
    ]

    state = simulator.state_manager.get_state()
    rate_keys = (
        "gross_primary_productivity",
        "maintenance_respiration_total",
        "growth_respiration_total",
        "total_autotrophic_respiration",
        "net_primary_productivity",
        "bvoc_carbon_allocation_flux",
        "growth_rate_leaf",
        "growth_rate_stem",
        "growth_rate_root",
    )
    for key in rate_keys:
        assert torch.isfinite(state[key]), key
        assert state[key] >= 0.0, key
    torch.testing.assert_close(
        state["total_autotrophic_respiration"],
        state["maintenance_respiration_total"]
        + state["growth_respiration_total"],
        rtol=2.0 * torch.finfo(torch.float32).eps,
        atol=1.0e-15,
    )

    npp_residual = (
        state["gross_primary_productivity"]
        - state["total_autotrophic_respiration"]
        - state["net_primary_productivity"]
    )
    allocation_residual = (
        state["net_primary_productivity"]
        - state["bvoc_carbon_allocation_flux"]
        - state["growth_rate_leaf"]
        - state["growth_rate_stem"]
        - state["growth_rate_root"]
    )
    structural_delta = sum(
        state[name] - initial_biomass[name]
        for name in ("leaf_biomass_c", "wood_biomass_c", "root_biomass_c")
    )
    structural_input = dt * (
        state["growth_rate_leaf"]
        + state["growth_rate_stem"]
        + state["growth_rate_root"]
    )
    assert torch.abs(npp_residual) <= 2.0 * torch.finfo(torch.float32).eps * torch.abs(
        state["gross_primary_productivity"]
    )
    assert torch.abs(allocation_residual) <= 2.0 * torch.finfo(
        torch.float32
    ).eps * torch.abs(state["net_primary_productivity"])
    assert torch.allclose(
        structural_delta,
        structural_input,
        rtol=0.0,
        atol=3.0e-8,
    )

    registry = simulator.state_manager._contract_registry
    assert registry is not None
    for key in ("growth_rate_leaf", "growth_rate_stem", "growth_rate_root"):
        contract = registry.get_contract(key)
        assert contract is not None
        assert contract.udunits == "kg C m-2 s-1"
        assert contract.accumulates == "carbon.total"
        assert contract.flux_in == ["biomass_allocation"]
        assert key in updates

    expected_writers = {
        "gross_primary_productivity": "farquhar_photosynthesis",
        "total_autotrophic_respiration": "autotrophic_respiration",
        "net_primary_productivity": "autotrophic_respiration",
        "bvoc_carbon_allocation_flux": "mechanistic_monoterpene_synthesis",
        "growth_rate_leaf": "biomass_allocation",
        "growth_rate_stem": "biomass_allocation",
        "growth_rate_root": "biomass_allocation",
        "leaf_biomass_c": "biomass_allocation",
        "wood_biomass_c": "biomass_allocation",
        "root_biomass_c": "biomass_allocation",
    }
    for key, expected_writer in expected_writers.items():
        writers = {
            record["functor"]
            for record in simulator.get_variable_provenance(key)
            if record.get("role") in {"writer", "final_writer"}
        }
        assert writers == {expected_writer}, f"{key}: {writers}"

    for functor in selected_functors:
        assert simulator.get_effective_device(functor.name).type == "cpu"
