import torch

from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.parameters import get_global_manager


def _build_coupling() -> SoilVegetationCoupling:
    return SoilVegetationCoupling(
        name="sv-root-dist-l1",
        config={},
        parameter_manager=get_global_manager(),
    )


def test_root_distribution_single_layer_normalizes_to_unity() -> None:
    coupling = _build_coupling()
    depth_profile = torch.tensor([1.0], dtype=torch.float32)

    beta = coupling.compute_root_distribution(depth_profile=depth_profile)

    boundaries = torch.cat([torch.zeros(1, dtype=depth_profile.dtype), depth_profile])
    thickness = torch.diff(boundaries)
    integral = torch.sum(beta * thickness)

    assert beta.shape[0] == thickness.shape[0] == 1
    assert torch.isclose(integral, torch.tensor(1.0, dtype=integral.dtype), atol=1e-6)
    assert torch.all(torch.isfinite(beta))


def test_root_distribution_respects_all_layer_bottoms() -> None:
    coupling = _build_coupling()
    depth_profile = torch.tensor([0.3, 1.0], dtype=torch.float32)

    beta = coupling.compute_root_distribution(depth_profile=depth_profile)

    boundaries = torch.cat([torch.zeros(1, dtype=depth_profile.dtype), depth_profile])
    thickness = torch.diff(boundaries)

    assert beta.shape[0] == depth_profile.shape[0] == thickness.shape[0]
    assert torch.isclose(
        torch.sum(beta * thickness), torch.tensor(1.0, dtype=thickness.dtype), atol=1e-6
    )

    root_density = torch.ones_like(depth_profile)
    beta_density = coupling.compute_root_distribution(
        depth_profile=depth_profile, root_density=root_density
    )
    integral_density = torch.sum(beta_density * thickness)
    assert torch.isclose(
        integral_density, torch.tensor(1.0, dtype=integral_density.dtype), atol=1e-6
    )
