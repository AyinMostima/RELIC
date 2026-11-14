"""Unit contract for the dependency-injected phenology observation aggregator."""

from __future__ import annotations

import inspect

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.phenology.phenology_functor import PhenologyFunctor


def _owner_output(*, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    phenological_factor = torch.tensor(0.7, requires_grad=requires_grad)
    return {
        "leaf_area_index": phenological_factor * 4.0,
        "phenological_factor": phenological_factor,
        "phenology_senescence_factor": phenological_factor * 0.1,
        "phenology_phase": phenological_factor * 0.0 + 2.0,
    }


def _functor(owner: str = "gsi_phenology") -> PhenologyFunctor:
    return PhenologyFunctor(config={"canonical_model_owner": owner})


@pytest.mark.parametrize(
    "owner",
    ["static_phenology", "gdd_phenology", "gsi_phenology", "crop_phenology"],
)
def test_compile_configuration_selects_one_canonical_owner(owner: str) -> None:
    functor = _functor(owner)

    assert functor.canonical_model_owner == owner
    assert functor.get_dependencies() == {"required": [owner], "optional": []}
    assert not hasattr(functor, "primary_model")


def test_missing_or_invalid_compile_selection_fails_fast() -> None:
    with pytest.raises(ValueError, match="config.canonical_model_owner is required"):
        PhenologyFunctor(config={})
    with pytest.raises(ValueError, match="Unsupported canonical phenology model owner"):
        _functor("unknown_phenology")


def test_missing_dependency_and_required_owner_observations_fail_fast() -> None:
    functor = _functor()

    with pytest.raises(KeyError, match="gsi_phenology"):
        functor.forward({}, 0.0, 86400.0, subprocess_outputs={})
    with pytest.raises(KeyError, match="phenology_senescence_factor"):
        functor.forward(
            {},
            0.0,
            86400.0,
            subprocess_outputs={
                "gsi_phenology": {
                    key: value
                    for key, value in _owner_output().items()
                    if key != "phenology_senescence_factor"
                }
            },
        )

    empty_observation = _owner_output()
    empty_observation["leaf_area_index"] = torch.empty(0)
    with pytest.raises(ValueError, match="leaf_area_index.*must not be empty"):
        functor.forward(
            {},
            0.0,
            86400.0,
            subprocess_outputs={"gsi_phenology": empty_observation},
        )


def test_forward_aggregates_only_selected_owner_observations() -> None:
    functor = _functor()
    selected = _owner_output()
    other = {key: value + 10.0 for key, value in _owner_output().items()}

    output = functor.forward(
        {},
        0.0,
        86400.0,
        subprocess_outputs={"gsi_phenology": selected, "gdd_phenology": other},
    )

    assert set(output) == {
        "diagnostics.phenology_state",
        "diagnostics.growth_stage",
        "diagnostics.leaf_area_index",
        "diagnostics.phenology.senescence_factor",
    }
    expected = {
        "diagnostics.phenology_state": ("phenological_factor", "dimensionless"),
        "diagnostics.growth_stage": ("phenology_phase", "dimensionless"),
        "diagnostics.leaf_area_index": ("leaf_area_index", "m2 m-2"),
        "diagnostics.phenology.senescence_factor": (
            "phenology_senescence_factor",
            "dimensionless",
        ),
    }
    for diagnostic_name, (owner_name, unit) in expected.items():
        item = output[diagnostic_name]
        assert isinstance(item, UpdateItem)
        assert item.value is selected[owner_name]
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is True
        assert item.unit == unit
        assert item.source_functor == "biosphere_phenology_functor"


def test_aggregated_observations_preserve_autograd() -> None:
    functor = _functor()
    selected = _owner_output(requires_grad=True)

    output = functor.forward(
        {},
        0.0,
        86400.0,
        subprocess_outputs={"gsi_phenology": selected},
    )
    target = output["diagnostics.leaf_area_index"].value
    target.backward()

    assert selected["phenological_factor"].grad is not None
    assert torch.isfinite(selected["phenological_factor"].grad)


def test_no_legacy_formula_or_runtime_switch_surface_remains() -> None:
    source = inspect.getsource(PhenologyFunctor)
    parameters = inspect.signature(PhenologyFunctor.forward).parameters

    assert list(parameters) == ["self", "state", "time", "dt", "subprocess_outputs"]
    for forbidden in (
        "StaticPhenology(",
        "GDDPhenology(",
        "GSIPhenology(",
        "CropPhenology(",
        "_static_phenology",
        "_gdd_phenology",
        "_gsi_phenology",
        "_crop_phenology",
        "switch_model",
        "adaptive_switching",
    ):
        assert forbidden not in source
