"""
Regression tests for phenology bounds validation semantics.
"""

import pytest
import torch

from relic.core.functor import ConservationError
from relic.scientific.biosphere.phenology.base import PhenologyOutput, PhenologyPhase
from relic.scientific.biosphere.phenology.models import StaticPhenology


def test_validate_output_bounds_raises_value_error_not_conservation_error():
    model = StaticPhenology()
    bad_output = PhenologyOutput(
        phenology_phase=PhenologyPhase.FULL_LEAF,
        growing_season_index=torch.tensor(1.0),
        thermal_time=torch.tensor(100.0),
        leaf_area_index=torch.tensor(-1.0),
        growth_factor=torch.tensor(0.5),
        senescence_rate=torch.tensor(0.1),
    )

    with pytest.raises(ValueError, match="Negative LAI detected") as exc_info:
        model._validate_output_bounds(bad_output)

    assert not isinstance(exc_info.value, ConservationError)

