import math

import pytest
import torch

from relic.coupling.core.policy import ScalingMode
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.types import ScalingAudit, ScalingStrategy
from relic.core.state import StateManager
from relic.parameters import get_global_manager


def test_hierarchy_round_trip_respects_parameter_counts():
    pm = get_global_manager()
    units_per_hillslope = pm.get(
        "coupling.scales.hierarchy.patch.units_per_hillslope",
        units="dimensionless",
    )
    assert units_per_hillslope is not None
    units_per_hillslope = int(units_per_hillslope)

    class DummyAgg:
        def aggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
            tensor = data if isinstance(data, torch.Tensor) else next(iter(data.values()))
            chunks = torch.split(tensor, units_per_hillslope)
            agg = torch.stack([c.sum() for c in chunks])
            audit = ScalingAudit(
                operation="aggregate",
                source_scale=source_scale,
                target_scale=target_scale,
                variables=variables or ["flux"],
                mode=mode or ScalingMode.CONSERVATIVE,
                strategy=strategy or ScalingStrategy.AREA_WEIGHTED_SUM,
                mass_balance_error=0.0,
                relative_error=0.0,
                cf_compliant=False,
                conservation_enforced=True,
                physical_constraints_applied=False,
                warnings=[],
                metadata={},
            )
            return agg, audit

        def disaggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
            tensor = data if isinstance(data, torch.Tensor) else next(iter(data.values()))
            base = tensor / float(units_per_hillslope)
            expanded = base.repeat_interleave(units_per_hillslope)
            audit = ScalingAudit(
                operation="disaggregate",
                source_scale=source_scale,
                target_scale=target_scale,
                variables=variables or ["flux"],
                mode=mode or ScalingMode.CONSERVATIVE,
                strategy=strategy or ScalingStrategy.AREA_WEIGHTED_SUM,
                mass_balance_error=0.0,
                relative_error=0.0,
                cf_compliant=False,
                conservation_enforced=True,
                physical_constraints_applied=False,
                warnings=[],
                metadata={},
            )
            return expanded, audit

    manager = ScaleManager(
        default_mode=ScalingMode.CONSERVATIVE,
        state_manager=StateManager(),
    )
    manager.initialize()
    manager.hierarchical_strategy = DummyAgg()
    manager._validate_scale_transition = lambda *args, **kwargs: True  # noqa: SLF001

    patch_count = int(units_per_hillslope) * 2
    patch_flux = torch.arange(patch_count, dtype=torch.float32)

    aggregated_flux, _ = manager.aggregate(
        patch_flux,
        source_scale="patch",
        target_scale="hillslope",
        variables=["flux"],
        mode=ScalingMode.CONSERVATIVE,
    )
    expected_targets = math.ceil(patch_count / float(units_per_hillslope))
    assert aggregated_flux.shape[0] == expected_targets

    disaggregated_flux, _ = manager.disaggregate(
        aggregated_flux,
        source_scale="hillslope",
        target_scale="patch",
        variables=["flux"],
        mode=ScalingMode.CONSERVATIVE,
    )
    assert disaggregated_flux.shape[0] == patch_flux.shape[0]
    assert torch.isfinite(disaggregated_flux).all()
    assert disaggregated_flux.sum().item() == pytest.approx(
        patch_flux.sum().item(), rel=0.05
    )

