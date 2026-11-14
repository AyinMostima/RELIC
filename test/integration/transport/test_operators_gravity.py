# -*- coding: ascii -*-
"""
Integration test: canonical MOST length uses physical-constant SSOT.

- References: Monin & Obukhov (1954)
- Expression: L = -u_star^3 * rho * cp * T / (k * g * H)
- Parameters: g (m s-2), k (dimensionless), rho (kg m-3), cp (J kg-1 K-1)
"""

import torch

from relic.scientific.atmosphere.most.stability_parameters import (
    monin_obukhov_length,
)
from relic.utils.physical_constants import get_physical_constant


def test_monin_obukhov_length_uses_gravity_ssot():
    u_star = torch.tensor(0.5)
    rho = 1.225
    cp = 1004.64
    T = torch.tensor(300.0)
    H = torch.tensor(50.0)
    k = get_physical_constant("von_karman_constant")
    g = get_physical_constant("gravity_acceleration")

    actual = monin_obukhov_length(u_star, rho, cp, T, H)
    expected = -u_star**3 * rho * cp * T / (k * g * H)

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


