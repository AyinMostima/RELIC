
import pytest
import torch

from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling


def test_spatial_scaling_conservative_failure_raises(monkeypatch):
    scaling = SpatialScaling(
        source_scale="fine",
        target_scale="coarse",
        source_resolution=(1.0, 1.0),
        target_resolution=(2.0, 2.0),
        method="conservative",
        preserve="mass",
        enforce_conservation=True,
        input_variables=["data"],
        output_variables=["data_scaled"],
    )

    def _raise(*_args, **_kwargs):
        raise ValueError("forced failure")

    monkeypatch.setattr(scaling, "_get_regridder", _raise)
    data = torch.ones(4, 4)

    with pytest.raises(ValueError):
        scaling._apply_regridding(data, operation="downscale")
