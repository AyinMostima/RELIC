import torch

# ASCII-only comments for cross-platform safety.
# Test canonical MOST sensitivity to explicit physical constants.

from relic.scientific.atmosphere.most.stability_parameters import (
    friction_velocity,
    monin_obukhov_length,
)


def test_friction_velocity_responds_to_von_karman_constant():
    # Setup simple neutral case: zeta = 0.
    U = torch.tensor(10.0)
    z = torch.tensor(10.0)
    z0m = torch.tensor(0.1)
    d = torch.tensor(0.0)
    zeta = torch.tensor(0.0)

    u_star_041 = friction_velocity(U, z, z0m, d, zeta, k=0.41).item()
    u_star_039 = friction_velocity(U, z, z0m, d, zeta, k=0.39).item()

    # Ratio should equal 0.39/0.41 within tolerance
    ratio = u_star_039 / u_star_041
    assert abs(ratio - (0.39 / 0.41)) < 1e-6


def test_monin_obukhov_length_responds_to_gravity_constant():
    # Setup values so L is finite and negative for positive upward H.
    u_star = torch.tensor(0.5)
    rho = 1.225
    cp = 1004.64
    T = torch.tensor(300.0)
    H = torch.tensor(100.0)

    L_g_std = monin_obukhov_length(
        u_star, rho, cp, T, H, k=0.41, g=9.80665
    ).item()
    L_g_small = monin_obukhov_length(
        u_star, rho, cp, T, H, k=0.41, g=9.5
    ).item()

    # |L| should increase when gravity is smaller
    assert abs(L_g_small) > abs(L_g_std)

