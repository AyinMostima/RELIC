import torch

import pytest

from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.core import ScalingMode
from relic.core.state import StateManager
from relic.parameters import get_global_manager


def _simple_topology(n_source: int, n_target: int) -> dict:
    return {
        "n_source_units": n_source,
        "n_target_units": n_target,
        "parent_map": {i: 0 for i in range(n_source)},
        "source_areas": [1.0 for _ in range(n_source)],
        "target_areas": [float(max(n_source, 1)) for _ in range(n_target)],
    }


def test_aggregate_applies_physical_bounds():
    manager = ScaleManager(
        device=torch.device("cpu"),
        enforce_conservation=False,
        default_mode=ScalingMode.CONSERVATIVE,
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    manager.initialize()
    topology = _simple_topology(2, 1)
    data = {"air_temperature": torch.tensor([5.0, 6.0])}
    aggregated, audit = manager.aggregate(
        data=data,
        source_scale="patch",
        target_scale="hillslope",
        variables=["air_temperature"],
        metadata={"air_temperature": {"physical_bounds": (0.0, 1.0)}},
        topology=topology,
    )

    assert torch.all(aggregated["air_temperature"] <= 1.0 + 1e-6)
    assert torch.all(aggregated["air_temperature"] >= 0.0)
    assert audit.physical_constraints_applied is True


def test_disaggregate_applies_bounds_and_propagates_cf_metadata():
    manager = ScaleManager(
        device=torch.device("cpu"),
        enforce_conservation=False,
        default_mode=ScalingMode.CONSERVATIVE,
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    manager.initialize()
    topology = _simple_topology(2, 1)
    aggregated = {"air_temperature": torch.tensor([5.0])}
    original = aggregated["air_temperature"].clone()

    result, audit = manager.disaggregate(
        data=aggregated,
        source_scale="hillslope",
        target_scale="patch",
        variables=["air_temperature"],
        metadata={"air_temperature": {"physical_bounds": (0.0, 1.0)}},
        topology=topology,
    )

    # The broadcast value 5.0 is clipped to the declared bounds; the input is not mutated.
    torch.testing.assert_close(result["air_temperature"], torch.tensor([1.0, 1.0]))
    assert audit.physical_constraints_applied is True
    assert "Disaggregated area from hillslope to patch (area: mean)" in audit.metadata["cf_metadata"]["history"]
    torch.testing.assert_close(aggregated["air_temperature"], original)

