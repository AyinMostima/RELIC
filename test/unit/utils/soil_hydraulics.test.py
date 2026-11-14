
import torch

from relic.utils.soil_hydraulics import (
    effective_saturation,
    van_genuchten_mualem_conductivity,
    van_genuchten_water_retention,
)


def test_effective_saturation_basic() -> None:
    theta = torch.tensor(0.3)
    theta_r = 0.1
    theta_s = 0.5
    expected = 0.5
    result = effective_saturation(theta, theta_r, theta_s, epsilon=1e-12)
    assert torch.allclose(result, torch.tensor(expected), atol=1e-12, rtol=1e-6)


def test_van_genuchten_water_retention_known_value() -> None:
    pressure_head = torch.tensor(-1.0)
    alpha = 2.0
    n_vg = 2.0
    theta_r = 0.1
    theta_s = 0.4
    m_vg = 1.0 - 1.0 / n_vg
    denominator = (1.0 + (alpha * abs(float(pressure_head))) ** n_vg) ** m_vg
    expected = theta_r + (theta_s - theta_r) / denominator
    result = van_genuchten_water_retention(
        pressure_head=pressure_head,
        alpha_vg=alpha,
        n_vg=n_vg,
        theta_residual=theta_r,
        theta_saturated=theta_s,
    )
    assert torch.allclose(result, torch.tensor(expected), atol=1e-10, rtol=1e-6)


def test_van_genuchten_mualem_conductivity_known_value() -> None:
    Se = torch.tensor(0.5)
    Ks = 1.0e-5
    n_vg = 2.0
    tortuosity = 0.5
    m_vg = 1.0 - 1.0 / n_vg
    Se_power = float(Se) ** (1.0 / m_vg)
    inner_term = 1.0 - (1.0 - Se_power) ** m_vg
    expected = Ks * (float(Se) ** tortuosity) * (inner_term ** 2.0)
    result = van_genuchten_mualem_conductivity(
        effective_saturation=Se,
        k_saturated=Ks,
        n_vg=n_vg,
        tortuosity=tortuosity,
        epsilon=1e-12,
    )
    assert torch.allclose(result, torch.tensor(expected), atol=1e-10, rtol=1e-6)
