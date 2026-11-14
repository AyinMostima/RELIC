import pytest

from relic.coupling.scales.scale_manager import ScaleManager
from relic.core.state import StateManager
from relic.parameters import get_global_manager


def test_facade_topology_parent_map_length_check():
    facade = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    facade.initialize()
    topology = {
        "n_source_units": 2,
        "n_target_units": 1,
        "parent_map": {0: 0},  # length 1, should be 2
        "source_areas": [1.0, 1.0],
        "target_areas": [1.0],
    }
    with pytest.raises(ValueError):
        facade._validate_topology(topology, "aggregate")  # pylint: disable=protected-access


def test_facade_topology_parent_bounds_check():
    facade = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    facade.initialize()
    topology = {
        "n_source_units": 1,
        "n_target_units": 1,
        "parent_map": {0: 1},  # target index out of range
        "source_areas": [1.0],
        "target_areas": [1.0],
    }
    with pytest.raises(ValueError):
        facade._validate_topology(topology, "aggregate")  # pylint: disable=protected-access
