"""L1 checks for the parameter-free PhenologyFunctor aggregation shell."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from relic.core.functor import ProcessCategory, VariableKind
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.phenology.phenology_functor import PhenologyFunctor


def _output(**overrides):
    output = {
        "leaf_area_index": torch.tensor(2.5),
        "phenological_factor": torch.tensor(0.6),
        "phenology_senescence_factor": torch.tensor(0.1),
        "phenology_phase": torch.tensor(2.0),
    }
    output.update(overrides)
    return output


def _functor() -> PhenologyFunctor:
    return PhenologyFunctor(
        config={"canonical_model_owner": "static_phenology"},
        device=torch.device("cpu"),
    )


def test_initialization_is_parameter_free_and_does_not_mutate_config() -> None:
    config = {"canonical_model_owner": "static_phenology"}

    with patch.object(
        PhenologyFunctor,
        "get_parameter_value",
        side_effect=AssertionError("aggregator must not read scientific parameters"),
    ):
        functor = PhenologyFunctor(config=config)

    assert config == {"canonical_model_owner": "static_phenology"}
    assert functor.canonical_model_owner == "static_phenology"


def test_metadata_describes_observation_projection_only() -> None:
    functor = _functor()

    assert functor.metadata.category is ProcessCategory.BIOLOGICAL_PROCESS
    assert functor.metadata.variable_kind is VariableKind.DIAGNOSTIC
    assert set(functor.metadata.input_variables) == {
        "leaf_area_index",
        "phenological_factor",
        "phenology_senescence_factor",
        "phenology_phase",
    }
    assert set(functor.metadata.output_variables) == {
        "diagnostics.phenology_state",
        "diagnostics.growth_stage",
        "diagnostics.leaf_area_index",
        "diagnostics.phenology.senescence_factor",
    }
    assert not functor.metadata.conservation_laws


def test_update_items_are_unwrapped_without_changing_values() -> None:
    functor = _functor()
    leaf_area = torch.tensor(3.0)
    owner_output = _output(
        leaf_area_index=UpdateItem(leaf_area, mode=UpdateMode.REPLACE),
    )

    result = functor.forward(
        {},
        0.0,
        86400.0,
        subprocess_outputs={"static_phenology": owner_output},
    )

    projected = result["diagnostics.leaf_area_index"]
    assert isinstance(projected, UpdateItem)
    assert projected.value is leaf_area
    assert projected.mode is UpdateMode.REPLACE
    assert projected.per_timestep is True
    assert projected.unit == "m2 m-2"


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), torch.ones(2)])
def test_invalid_timestep_is_rejected(dt) -> None:
    functor = _functor()

    with pytest.raises(ValueError, match="dt must be"):
        functor.forward(
            {},
            0.0,
            dt,
            subprocess_outputs={"static_phenology": _output()},
        )


@pytest.mark.parametrize(
    "bad_value, error_type",
    [("invalid", TypeError), (torch.tensor(float("inf")), ValueError)],
)
def test_invalid_required_observation_is_rejected(bad_value, error_type) -> None:
    functor = _functor()

    with pytest.raises(error_type, match="leaf_area_index"):
        functor.forward(
            {},
            0.0,
            86400.0,
            subprocess_outputs={
                "static_phenology": _output(leaf_area_index=bad_value)
            },
        )


def test_nonselected_owner_cannot_satisfy_required_dependency() -> None:
    functor = _functor()

    with pytest.raises(KeyError, match="static_phenology"):
        functor.forward(
            {},
            0.0,
            86400.0,
            subprocess_outputs={"gsi_phenology": _output()},
        )
