"""Retention, bucket and water stress read one soil frame.

The bucket stores water between its wilting point theta_wp and field capacity
theta_fc; SoilWaterRetention maps the stored water to a potential psi; WaterStress
scales emission by f = clamp((psi - psi_wilt) / (psi_fc - psi_wilt), 0, 1). The
chain only responds to drying when the van Genuchten parameters map theta_wp to
psi_wilt. With the shipped generic retention (alpha 7.5 m-1, n 1.89, theta_r
0.045) and the ECMWF fine class (theta_s 0.52, theta_wp 0.279) psi(theta_wp) is
about -0.0025 MPa, so f stays 1 over the whole plant-available range: this is the
mismatch found in the RELIC paper-01 C11/C12 members (2026-11-05 diagnostics).

The consistent frames below are the van Genuchten parameters that reproduce the
frame's own water contents at -0.10 bar and -15 bar (ECMWF TM563 Table 2
definitions; Balsamo et al. 2008), derived in RELICPAPER
shared/research_status_review/20261105_d0_coupling_gradient_diagnostics_01.
"""

from __future__ import annotations

import pytest
import torch

from relic.scientific.biosphere.stress.water_stress import WaterStress
from relic.scientific.hydrosphere.soil_moisture.retention import SoilWaterRetention

_NS = "hydrosphere.soil_moisture."

# frame: (theta_s, theta_fc, theta_wp, alpha_vg [m-1], n_vg [1], theta_r [m3 m-3])
CONSISTENT_FRAMES = {
    # ECMWF TM563 fine class; n at the owner lower bound 1.1, alpha and theta_r solved.
    "ecmwf_fine": (0.520, 0.448, 0.279, 3.6631138204703304, 1.1, 0.006105925148495822),
    # Generic bucket frame of the paper-01 plan; theta_r = 0, alpha and n solved.
    "generic": (0.45, 0.30, 0.10, 5.480483872596276, 1.223436354414435, 0.0),
}


@pytest.fixture(autouse=True)
def _restore(restore_global_parameter_manager):
    yield restore_global_parameter_manager


def _write(manager, values):
    # Writes go through the owner's error bounds policy: an out-of-bounds value
    # raises instead of being clamped; the previous policy is put back.
    policy = manager.bounds_policy
    manager.set_bounds_policy("error")
    try:
        for key, value in values.items():
            units = getattr(manager.get_parameter(key), "units", None)
            manager.set_value(key, value, validate=True, notify=False, **({"units": units} if units else {}))
    finally:
        manager.set_bounds_policy(policy)


def _potential(theta: float) -> float:
    out = SoilWaterRetention().forward({"soil_moisture": torch.tensor([theta], dtype=torch.float64)}, 0.0, 3600.0)
    return float(out["soil_water_potential"].value.reshape(-1)[0])  # MPa


def _factor(psi: float) -> float:
    out = WaterStress().forward({"soil_water_potential": torch.tensor([psi], dtype=torch.float64)}, 0.0, 3600.0)
    return float(out["water_stress_factor"].value.reshape(-1)[0])  # 1


@pytest.mark.parametrize("frame", sorted(CONSISTENT_FRAMES))
def test_consistent_frame_maps_bucket_bounds_to_stress_window(restore_global_parameter_manager, frame):
    """psi(theta_wp) = -1.5 MPa and psi(theta_fc) = -0.01 MPa; the stress factor is 0
    at the wilting point, 1 at field capacity and between 0 and 1 in between."""
    # References: van Genuchten (1980) Soil Sci. Soc. Am. J. 44, 892-898; Balsamo et
    #   al. (2008) ECMWF Tech. Memo. 563, Table 2; water_stress.py (linear factor).
    # Mathematical expression: psi = rho_w g h, h = -(Se^(-1/m) - 1)^(1/n) / alpha,
    #   Se = (theta - theta_r)/(theta_s - theta_r), m = 1 - 1/n;
    #   f = clamp((psi - psi_wilt)/(psi_fc - psi_wilt), 0, 1).
    # Units: theta [m3 m-3]; alpha [m-1]; n, f [1]; psi [MPa].
    # Domain/constraints: every value inside the owner bounds (error policy); the
    #   WaterStress thresholds keep their shipped values (-1.5 and -0.033 MPa).
    manager = restore_global_parameter_manager
    theta_s, theta_fc, theta_wp, alpha, n, theta_r = CONSISTENT_FRAMES[frame]
    _write(manager, {
        _NS + "porosity": theta_s,
        _NS + "alpha_vg": alpha,
        _NS + "n_vg": n,
        _NS + "residual_water_content": theta_r,
    })
    psi_wp, psi_fc = _potential(theta_wp), _potential(theta_fc)
    # float32 owner values: a relative tolerance of 1e-3 on the potential.
    assert psi_wp == pytest.approx(-1.5, rel=1.0e-3)
    assert psi_fc == pytest.approx(-0.01, rel=1.0e-3)
    assert _factor(psi_wp) == pytest.approx(0.0, abs=2.0e-3)
    assert _factor(psi_fc) == 1.0
    middle = 0.5 * (theta_wp + theta_fc)
    assert 0.0 < _factor(_potential(middle)) < 1.0


def test_shipped_generic_retention_leaves_the_fine_bucket_unstressed(restore_global_parameter_manager):
    """Characterizes the mismatch: shipped retention defaults with the fine bucket
    class give psi(theta_wp) far above psi_fc, so f = 1 over the bucket's range."""
    manager = restore_global_parameter_manager
    _write(manager, {_NS + "porosity": 0.520})
    psi_wp = _potential(0.279)
    assert -0.01 < psi_wp < 0.0
    assert _factor(psi_wp) == 1.0
