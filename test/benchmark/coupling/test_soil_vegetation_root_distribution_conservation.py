import torch

from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.parameters import get_global_manager


def test_root_distribution_conservation_respects_depth_semantics() -> None:
    coupling = SoilVegetationCoupling(
        name="sv-root-dist-l3",
        config={},
        parameter_manager=get_global_manager(),
    )

    depth_layers = torch.tensor([0.3, 1.0], dtype=torch.float32)
    state = {
        "soil_depth_layers": depth_layers,
        "soil_moisture_profile": torch.tensor([0.22, 0.20], dtype=torch.float32),
        "soil_matric_potential": torch.tensor([-1.2e5, -1.4e5], dtype=torch.float32),
        "potential_transpiration": torch.tensor(0.0008, dtype=torch.float32),
    }

    outputs = coupling.forward(state=state, time=0.0, dt=900.0)

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

    diagnostics = coupling.check_conservation_laws(state, outputs)
    assert "water" in diagnostics
    assert diagnostics["water"] <= coupling.conservation_tolerance + 1e-6
    assert torch.all(torch.isfinite(outputs["root_uptake_profile"].value))
    assert torch.all(torch.isfinite(outputs["effective_transpiration"].value))
