import torch

from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.parameters import get_global_manager


def test_forward_aligns_root_distribution_with_layer_bottoms() -> None:
    coupling = SoilVegetationCoupling(
        name="sv-root-dist-l2",
        config={},
        parameter_manager=get_global_manager(),
    )

    depth_layers = torch.tensor([1.0], dtype=torch.float32)
    state = {
        "soil_depth_layers": depth_layers,
        "soil_moisture_profile": torch.tensor([0.25], dtype=torch.float32),
        "soil_matric_potential": torch.tensor([-1.5e5], dtype=torch.float32),
        "potential_transpiration": torch.tensor(0.001, dtype=torch.float32),
    }

    outputs = coupling.forward(state=state, time=0.0, dt=1800.0)

    boundaries = torch.cat([torch.zeros(1, dtype=depth_layers.dtype), depth_layers])
    thickness = torch.diff(boundaries)

    root_distribution = outputs[
        "diagnostics.soil_vegetation.root_distribution"
    ].value
    assert root_distribution.shape[0] == depth_layers.shape[0]
    assert torch.isclose(
        torch.sum(root_distribution * thickness),
        torch.tensor(1.0, dtype=thickness.dtype),
        atol=1e-6,
    )

    assert outputs["root_uptake_profile"].value.shape[0] == depth_layers.shape[0]
    assert torch.all(torch.isfinite(outputs["root_uptake_profile"].value))
    residual = outputs[
        "diagnostics.soil_vegetation.water_balance_residual"
    ].value
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-10)
