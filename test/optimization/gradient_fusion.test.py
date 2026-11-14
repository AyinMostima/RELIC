import math

import pytest
import torch

from relic.optimization.assimilation.gradient_fusion import DualTrackGradientFusion
from relic.parameters import get_global_manager


@pytest.fixture
def parameter_snapshot():
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        yield manager
    finally:
        manager.restore(snapshot)


def test_inverse_variance_weighting(parameter_snapshot) -> None:
    manager = parameter_snapshot
    manager.set_value("optimization.gradient_fusion.physics_weight_mode", "inverse_variance")
    manager.set_value("optimization.gradient_fusion.fixed_physics_weight", 0.5)

    fusion = DualTrackGradientFusion()
    fused, _ = fusion.fuse(
        grad_physics={"x": torch.tensor([2.0])},
        grad_statistical={"x": torch.tensor([0.0])},
        physics_uncertainty={"x": torch.tensor(1.0)},
        statistical_uncertainty={"x": torch.tensor(4.0)},
    )

    expected = (1.0 * 2.0 + 0.25 * 0.0) / (1.0 + 0.25)
    assert fused["x"].item() == pytest.approx(expected)


def test_confidence_interval(parameter_snapshot) -> None:
    manager = parameter_snapshot
    manager.set_value("optimization.gradient_fusion.physics_weight_mode", "inverse_variance")
    manager.set_value("optimization.gradient_fusion.fixed_physics_weight", 0.5)

    fusion = DualTrackGradientFusion()
    _, confidence = fusion.fuse(
        grad_physics={"x": torch.tensor([2.0])},
        grad_statistical={"x": torch.tensor([0.0])},
        physics_uncertainty={"x": torch.tensor(1.0)},
        statistical_uncertainty={"x": torch.tensor(4.0)},
    )

    expected_sigma = math.sqrt(1.0 / (1.0 + 0.25))
    assert confidence["x"].mean().item() == pytest.approx(expected_sigma)


def test_degenerate_cases(parameter_snapshot) -> None:
    manager = parameter_snapshot
    manager.set_value("optimization.gradient_fusion.physics_weight_mode", "inverse_variance")

    fusion = DualTrackGradientFusion()
    fused, confidence = fusion.fuse(
        grad_physics={"x": torch.tensor([1.0, -1.0])},
        grad_statistical={"x": torch.tensor([1.0, 1.0])},
        physics_uncertainty={"x": torch.tensor(0.0)},
        statistical_uncertainty={"x": torch.tensor(1.0e12)},
    )

    assert torch.isfinite(fused["x"]).all()
    assert torch.isfinite(confidence["x"]).all()


class _SparseManager:
    """Manager view in which the listed keys are absent or overridden."""

    def __init__(self, inner, overrides):
        self._inner = inner
        self._overrides = overrides

    def get(self, key, *args, **kwargs):
        if key in self._overrides:
            value = self._overrides[key]
            if value is None:
                if kwargs.get("required"):
                    raise KeyError(f"Required parameter '{key}' not found")
                return None
            return value
        return self._inner.get(key, *args, **kwargs)


def _patch_manager(monkeypatch, overrides) -> None:
    import relic.optimization.assimilation.gradient_fusion as fusion_module

    inner = get_global_manager()
    monkeypatch.setattr(
        fusion_module, "get_global_manager", lambda: _SparseManager(inner, overrides)
    )


def test_missing_fusion_parameter_is_an_error_not_a_code_default(monkeypatch) -> None:
    _patch_manager(
        monkeypatch, {"optimization.gradient_fusion.fixed_physics_weight": None}
    )

    with pytest.raises(KeyError, match="fixed_physics_weight"):
        DualTrackGradientFusion()


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_fixed_weight_outside_unit_interval_is_rejected_not_clamped(monkeypatch, weight) -> None:
    _patch_manager(
        monkeypatch, {"optimization.gradient_fusion.fixed_physics_weight": weight}
    )

    with pytest.raises(ValueError, match="fixed_physics_weight"):
        DualTrackGradientFusion()


def test_scalar_gradient_without_uncertainty_is_not_given_unit_variance(parameter_snapshot) -> None:
    fusion = DualTrackGradientFusion()

    with pytest.raises(ValueError, match="uncertainty"):
        fusion.fuse(
            grad_physics={"x": torch.tensor([2.0])},
            grad_statistical={"x": torch.tensor([1.0])},
            physics_uncertainty={},
            statistical_uncertainty={"x": torch.tensor(1.0)},
        )


def test_unused_ensemble_size_setting_is_not_kept(parameter_snapshot) -> None:
    """No code reads optimization.gradient_fusion.min_ensemble_size (directly or
    through a composed key or a namespace-wide read): neither an attribute nor a
    parameter database entry is kept."""
    from relic.parameters import get_global_manager

    assert not hasattr(DualTrackGradientFusion(), "min_ensemble_size")
    manager = get_global_manager()
    assert manager.get("optimization.gradient_fusion.min_ensemble_size", required=False) is None
    assert sorted(manager.get_namespace_parameters(namespace="optimization.gradient_fusion")) == [
        "optimization.gradient_fusion.fixed_physics_weight",
        "optimization.gradient_fusion.physics_weight_mode",
    ]
