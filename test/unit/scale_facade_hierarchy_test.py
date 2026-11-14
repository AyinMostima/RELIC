import pytest

from relic.coupling.core.policy import get_precision_policy, ScalingMode
from relic.coupling.scales.scale_manager import ScaleManager
from relic.core.state import StateManager
from relic.parameters import get_global_manager


def test_topology_from_parameters_matches_config():
    manager = ScaleManager(
        default_mode=ScalingMode.CONSERVATIVE,
        state_manager=StateManager(),
    )
    manager.initialize()
    topology = manager._build_topology_from_parameters(
        source_scale="patch",
        target_scale="hillslope",
        n_source_units=6,
    )
    assert topology is not None
    assert topology["n_target_units"] == 2
    assert topology["parent_map"][0] == 0
    assert topology["parent_map"][3] == 1

    pm = get_global_manager()
    patch_area = pm.get(
        "coupling.scales.hierarchy.patch.default_area",
        units="m2",
    )
    assert patch_area is not None
    assert pytest.approx(sum(topology["source_areas"]), rel=1e-6) == pytest.approx(
        patch_area * 6.0, rel=1e-6
    )


def test_hierarchical_components_use_precision_policy():
    manager = ScaleManager(
        default_mode=ScalingMode.CONSERVATIVE,
        state_manager=StateManager(),
    )
    manager.initialize()
    manager._init_hierarchical_components()
    policy = get_precision_policy(manager.default_mode)
    assert pytest.approx(
        manager.hierarchical_aggregator.conservation_tolerance, rel=1e-12
    ) == policy.tolerance
    assert manager.hierarchy_index.hierarchy_tolerance <= policy.tolerance

