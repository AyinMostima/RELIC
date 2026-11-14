"""
Test MOST von Karman scaling without mutating the global parameter manager.

- References: Monin and Obukhov (1954); Businger et al. (1971)
- Expression: u_star = k * U / (ln((z-d)/z0m) - psi_m), here psi_m(0) = 0
- Parameters: k (von Karman, dimensionless), U (m s-1), z, z0m, d (m)
"""

import math

import torch

from relic.scientific.atmosphere.most.stability_parameters import friction_velocity


def test_friction_velocity_scales_with_von_karman():
    U = torch.tensor(5.0)
    z = torch.tensor(10.0)
    z0m = torch.tensor(0.1)
    d = torch.tensor(0.0)
    zeta = torch.tensor(0.0)

    k1 = 0.41
    k2 = 0.5
    u1 = friction_velocity(U, z, z0m, d, zeta, k=k1)
    u2 = friction_velocity(U, z, z0m, d, zeta, k=k2)

    ratio = (u2 / u1).item()
    expected = k2 / k1
    assert math.isclose(ratio, expected, rel_tol=1e-3, abs_tol=1e-6)

