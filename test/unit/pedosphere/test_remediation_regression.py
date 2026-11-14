"""Focused regression tests for pedosphere remediation items."""

from __future__ import annotations

from typing import Any

import torch
import warnings

from relic.core.state import UpdateItem
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import (
    SesquiterpeneStressModel,
    SesquiterpeneSynthesis,
)
from relic.scientific.pedosphere import CarbonPoolsFunctor
from relic.scientific.pedosphere.methane import (
    MethaneBudgetCoupler,
    MethaneOxidationFunctor,
    MethaneProductionFunctor,
    MethaneTransportFunctor,
)


def _update_value(value: Any) -> Any:
    """Return the tensor payload from a StateManager update item."""
    return value.value if isinstance(value, UpdateItem) else value


def test_methane_storage_single_writer_is_budget_coupler() -> None:
    """Only MethaneBudgetCoupler should declare CH4 storage outputs."""
    functors = [
        MethaneProductionFunctor(),
        MethaneOxidationFunctor(),
        MethaneTransportFunctor(),
        MethaneBudgetCoupler(),
    ]
    writers = [
        f.metadata.name
        for f in functors
        if "ch4_soil_storage" in getattr(f.metadata, "output_variables", [])
    ]
    assert writers == ["pedosphere_methane_budget"]
    assert all(
        "ch4_soil_storage_delta" not in f.metadata.output_variables
        for f in functors
    )


def test_methane_transport_diffusion_sign_is_bidirectional() -> None:
    """Diffusion must support both emission (positive) and uptake (negative)."""
    transport = MethaneTransportFunctor()
    base_state = {
        "ch4_soil_storage": torch.tensor(1.0),
        "water_filled_pore_space": torch.tensor(0.4),
        "porosity": torch.tensor(0.45),
    }
    dt = torch.tensor(600.0)

    emission_state = dict(base_state)
    emission_state["atmospheric_ch4_concentration"] = torch.tensor(0.0)
    emission = transport.forward(emission_state, time=0.0, dt=dt)

    uptake_state = dict(base_state)
    uptake_state["atmospheric_ch4_concentration"] = torch.tensor(100.0)
    uptake = transport.forward(uptake_state, time=0.0, dt=dt)

    assert _update_value(
        emission["diagnostics.methane_transport.diffusion_rate"]
    ).item() > 0.0
    assert _update_value(
        uptake["diagnostics.methane_transport.diffusion_rate"]
    ).item() < 0.0


def test_sesquiterpene_placeholder_warning_and_forward_contract() -> None:
    """Sesquiterpene stress model should initialize without placeholder warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SesquiterpeneStressModel()
    assert not any("placeholder implementation" in str(w.message) for w in caught)
    result = model.forward({"stress_level": torch.tensor(0.8)}, time=0.0, dt=1.0)
    item = result["stress_response_factor"]
    assert isinstance(item, UpdateItem)
    assert torch.is_tensor(item.value)


def test_sesquiterpene_synthesis_placeholder_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SesquiterpeneSynthesis()
    assert not any("placeholder implementation" in str(w.message) for w in caught)


def test_pedosphere_top_level_exports_carbon_pools_functor() -> None:
    """Top-level pedosphere package should expose CarbonPoolsFunctor."""
    assert CarbonPoolsFunctor is not None


RETIRED_LEACHING_HELPER_PARAMETERS = (
    "pedosphere.transport.water_flux_representation",
    "pedosphere.transport.leaching_max_nitrate_concentration",
    "pedosphere.transport.leaching_max_ammonium_concentration",
    "pedosphere.transport.leaching_min_concentration_threshold",
    "pedosphere.transport.leaching_max_flux_rate",
    "pedosphere.transport.leaching_max_flux_gradient",
    "pedosphere.transport.leaching_max_mobile_fraction",
    "pedosphere.transport.leaching_max_depletion_fraction",
    "pedosphere.transport.leaching_min_mobile_fraction",
    "pedosphere.transport.leaching_max_retardation_factor",
    "pedosphere.transport.leaching_max_layer_concentration_ratio",
    "pedosphere.transport.leaching_flux_consistency_tolerance",
    "pedosphere.transport.leaching_mass_balance_tolerance",
    "pedosphere.transport.leaching_enable_strict_conservation",
    "pedosphere.transport.leaching_soil_layers",
    "pedosphere.transport.leaching_layer_depths",
    "pedosphere.nutrient_cycling.mobile_fraction",
    "pedosphere.nutrient_cycling.no3_mobile_fraction",
    "pedosphere.nutrient_cycling.nh4_mobile_fraction",
)


def test_retired_leaching_boundary_helper_and_parameters_are_removed() -> None:
    """The layered leaching boundary helper was not a Functor, was absent from
    FUNCTOR_REGISTRY, could not be built as a GraphIR node and had no caller in
    relic; its parallel J = q C f_mobile law duplicated the production owner
    NitrogenLeachingProcess. The module and the parameters only it read are removed."""
    import importlib.util

    from relic.parameters import get_global_manager
    from relic.scientific import FUNCTOR_REGISTRY

    assert importlib.util.find_spec("relic.scientific.pedosphere.leaching_boundary_control") is None
    assert "NitrogenLeachingProcess" in FUNCTOR_REGISTRY
    manager = get_global_manager()
    present = [
        key for key in RETIRED_LEACHING_HELPER_PARAMETERS
        if manager.get(key, required=False) is not None
    ]
    assert present == []


def test_transport_forward_does_not_mutate_input_state() -> None:
    """Methane transport forward pass should not mutate input state in place."""
    transport = MethaneTransportFunctor()
    state = {
        "ch4_soil_storage": torch.tensor(1.0),
        "water_filled_pore_space": torch.tensor(0.4),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.5),
    }
    baseline = {k: v.clone() for k, v in state.items()}

    _ = transport.forward(state, time=0.0, dt=torch.tensor(300.0))

    assert set(state.keys()) == set(baseline.keys())
    for key in baseline:
        assert torch.equal(state[key], baseline[key])
