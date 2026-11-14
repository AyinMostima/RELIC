from __future__ import annotations

from pathlib import Path

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.utils.unified_response_functions import beer_lambert_effective_leaf_area
from tutorial.case.functors.stomatal_proxy import StomatalProxyFunctor
from tutorial.case.run_case import register_case_parameters


@pytest.fixture(autouse=True)
def _drop_case_parameter_sources(fresh_global_parameter_manager):
    """register_case_parameters adds the case parameter file as a source of the
    process-wide manager; the manager that was global before is put back."""
    yield


def _case_parameter_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tutorial" / "case" / "config" / "case_parameters.yaml"


def test_stomatal_proxy_replaces_conductance_with_the_lagged_target() -> None:
    """The proxy writes the absolute conductance g_s = clip(g0 + m max(A_leaf, 0),
    g_min, g_max) as a REPLACE update, whatever the current conductance is
    (stomatal_conductance is not incrementable; a delta would be committed as the
    new value)."""
    register_case_parameters(_case_parameter_path())
    manager = get_global_manager()
    g0 = float(manager.get("tutorial.case.stomatal_proxy.g0", required=True))
    slope = float(manager.get("tutorial.case.stomatal_proxy.slope", required=True))
    g_min = float(manager.get("tutorial.case.stomatal_proxy.min", required=True))
    g_max = float(manager.get("tutorial.case.stomatal_proxy.max", required=True))

    proxy = StomatalProxyFunctor(name="stomatal_proxy")
    leaf_photosynthesis = torch.tensor([[-2.0], [10.0], [1.0e4]], dtype=torch.float32)
    lai = torch.tensor([[1.0], [3.0], [6.0]], dtype=torch.float32)
    extinction = manager.get("biosphere.photosynthesis.canopy_light_extinction", required=True)
    net_photosynthesis = leaf_photosynthesis * beer_lambert_effective_leaf_area(lai, extinction)
    expected = torch.clamp(g0 + slope * torch.relu(leaf_photosynthesis), min=g_min, max=g_max)

    for current in (0.3, 0.0, 1.0):
        outputs = proxy.forward(
            {
                "net_photosynthesis": net_photosynthesis,
                "leaf_area_index": lai,
                "stomatal_conductance": torch.full_like(net_photosynthesis, current),
            },
            0.0,
            3600.0,
        )
        update = outputs["stomatal_conductance"]
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.REPLACE
        assert torch.allclose(update.value, expected)


def test_representative_leaf_conductance_is_invariant_to_canopy_support_and_differentiable():
    register_case_parameters(_case_parameter_path())
    manager = get_global_manager()
    proxy = StomatalProxyFunctor()
    leaf = torch.full((3,), 10.0, dtype=torch.float64, requires_grad=True)
    lai = torch.tensor([0.5, 2.0, 6.0], dtype=torch.float64, requires_grad=True)
    extinction = manager.get("biosphere.photosynthesis.canopy_light_extinction", required=True)
    ground = leaf * beer_lambert_effective_leaf_area(lai, extinction)
    result = proxy({"net_photosynthesis": ground, "leaf_area_index": lai}, 0.0, 1800.0)["stomatal_conductance"]
    slope = float(manager.get("tutorial.case.stomatal_proxy.slope", required=True))
    expected = float(manager.get("tutorial.case.stomatal_proxy.g0", required=True)) + slope * leaf
    torch.testing.assert_close(result.value, expected)
    dleaf, dlai = torch.autograd.grad(result.value.sum(), (leaf, lai))
    torch.testing.assert_close(dleaf, torch.full_like(leaf, slope))
    torch.testing.assert_close(dlai, torch.zeros_like(lai), rtol=0, atol=1e-14)
    assert result.unit == "mol m-2 s-1" and result.per_timestep is False
    assert "leaf_area_index" in proxy.metadata.input_variables


@pytest.mark.parametrize("lai", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_or_absent_leaf_support_is_not_assigned_a_default(lai):
    register_case_parameters(_case_parameter_path())
    proxy = StomatalProxyFunctor()
    with pytest.raises(ValueError, match="leaf_area_index|LAI"):
        proxy.forward({"net_photosynthesis": torch.tensor(0.0), "leaf_area_index": torch.tensor(lai)}, 0.0, 1800.0)
    with pytest.raises(KeyError, match="leaf_area_index"):
        proxy.forward({"net_photosynthesis": torch.tensor(10.0)}, 0.0, 1800.0)


def test_extinction_binding_keeps_gradient_and_ground_assimilation_is_not_leaf_rate(monkeypatch):
    register_case_parameters(_case_parameter_path())
    proxy = StomatalProxyFunctor()
    original = proxy.get_parameter_value
    extinction = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    baseline = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    slope = torch.tensor(0.005, dtype=torch.float64, requires_grad=True)

    def get_parameter(key, **kwargs):
        if key == "biosphere.photosynthesis.canopy_light_extinction":
            assert kwargs["required"] is True
            return extinction
        if key in {"g0", "slope"}:
            assert kwargs["required"] is True and kwargs["namespace"] == "tutorial.case.stomatal_proxy"
            return baseline if key == "g0" else slope
        return original(key, **kwargs)

    monkeypatch.setattr(proxy, "get_parameter_value", get_parameter)
    ground = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    lai = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    result = proxy.forward({"net_photosynthesis": ground, "leaf_area_index": lai}, 0.0, 1800.0)["stomatal_conductance"].value
    gradients = torch.autograd.grad(result, (ground, lai, extinction, baseline, slope))
    assert gradients[0] > 0 and gradients[1] < 0 and gradients[2] > 0
    torch.testing.assert_close(gradients[3], torch.ones_like(baseline))
    torch.testing.assert_close(gradients[4], ground / beer_lambert_effective_leaf_area(lai, extinction))
    with pytest.raises(ValueError, match="net_photosynthesis"):
        proxy.forward({"net_photosynthesis": torch.tensor(float("nan")), "leaf_area_index": lai}, 0.0, 1800.0)
