import torch

from relic.coupling.internal.water_carbon import WaterCarbonCoupling


def test_water_carbon_buffer_batch_loaded():
    module = WaterCarbonCoupling(name="water_carbon_test", config={}, metadata=None)
    # ensure key buffers exist and are tensors on correct device
    for key in [
        "wilting_point_volumetric",
        "field_capacity",
        "optimal_moisture",
        "vpd_half_saturation",
    ]:
        assert hasattr(module, key)
        buf = getattr(module, key)
        assert isinstance(buf, torch.Tensor)

