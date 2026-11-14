import torch

from relic.scientific.anthroposphere.land_use.mapping import LandUseParameterMapping, LandUseType


def test_land_use_mapping_metadata_outputs_match_forward_keys() -> None:
    functor = LandUseParameterMapping()
    n_types = len(list(LandUseType))
    state = {"land_use_fraction": torch.full((n_types,), 1.0 / n_types)}
    outputs = functor.forward(state=state, time=0.0, dt=86400.0)
    assert set(functor.metadata.output_variables) == set(outputs.keys())

