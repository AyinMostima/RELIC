"""Unit tests for strict conservative mode using ScaleManager."""

import torch

from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.types import ScalingStrategy, ScalingAudit
from relic.coupling.core.policy import ScalingMode
from relic.core.state import StateManager
from relic.parameters import get_global_manager


class MockAggregator:
    """Lightweight aggregator to mimic strict conservative behaviour."""

    def aggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
        variables = variables or (["value"] if isinstance(data, torch.Tensor) else list(data.keys()))
        if isinstance(data, torch.Tensor):
            result = torch.stack([data.sum().to(dtype=torch.float64)])
        else:
            result = {k: v.to(dtype=torch.float64).sum().unsqueeze(0) for k, v in data.items()}
        audit = ScalingAudit(
            operation="aggregate",
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            mode=mode or ScalingMode.STRICT_CONSERVATIVE,
            strategy=strategy or ScalingStrategy.AREA_WEIGHTED_MEAN,
            mass_balance_error=0.0,
            relative_error=0.0,
            cf_compliant=False,
            conservation_enforced=True,
            physical_constraints_applied=False,
            warnings=[],
            metadata={},
        )
        return result, audit

    def disaggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
        variables = variables or (["value"] if isinstance(data, torch.Tensor) else list(data.keys()))
        if isinstance(data, torch.Tensor):
            result = data.repeat(3).to(dtype=torch.float64)
        else:
            result = {k: v.repeat_interleave(3).to(dtype=torch.float64) for k, v in data.items()}
        audit = ScalingAudit(
            operation="disaggregate",
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            mode=mode or ScalingMode.STRICT_CONSERVATIVE,
            strategy=strategy or ScalingStrategy.AREA_WEIGHTED_MEAN,
            mass_balance_error=0.0,
            relative_error=0.0,
            cf_compliant=False,
            conservation_enforced=True,
            physical_constraints_applied=False,
            warnings=[],
            metadata={},
        )
        return result, audit


def _strict_manager():
    pm = get_global_manager()
    manager = ScaleManager(
        default_mode=ScalingMode.STRICT_CONSERVATIVE,
        enforce_conservation=True,
        parameter_manager=pm,
        cf_strict_mode=True,
        state_manager=StateManager(),
    )
    manager.initialize()
    manager.hierarchical_strategy = MockAggregator()
    manager._validate_scale_transition = lambda *args, **kwargs: True  # noqa: SLF001
    return manager


def test_precision_policy_strict():
    """Strict mode should use float64 and tight tolerance."""
    manager = _strict_manager()
    assert manager.precision_policy.tolerance <= 1e-12
    assert manager.precision_policy.dtype == torch.float64
    assert manager.enforce_conservation is True


def test_aggregate_preserves_float64():
    """Aggregate should promote to float64 under strict mode."""
    manager = _strict_manager()
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result, audit = manager.aggregate(data=data, source_scale="patch", target_scale="hillslope")
    assert result.dtype == torch.float64
    assert audit.mode == ScalingMode.STRICT_CONSERVATIVE


def test_disaggregate_preserves_float64():
    """Disaggregate should promote to float64 under strict mode."""
    manager = _strict_manager()
    data = torch.tensor([10.0, 20.0], dtype=torch.float32)
    result, audit = manager.disaggregate(data=data, source_scale="hillslope", target_scale="patch")
    assert result.dtype == torch.float64
    assert audit.mode == ScalingMode.STRICT_CONSERVATIVE
    assert result.numel() >= data.numel()


def test_conservation_enforcement_present():
    """Verify conservation enforcement path available."""
    manager = _strict_manager()
    report = manager.verify_conservation()
    assert isinstance(report, dict)
