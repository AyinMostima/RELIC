"""
Boundary-port tests for SoilWaterTransportFunctor.
"""

import torch

from relic.scientific.pedosphere.transport.SoilWaterTransportFunctor import (
    SoilWaterTransportFunctor,
)


def test_soil_water_transport_converts_required_mass_fluxes():
    """Canonical mass fluxes must enter the Richards ledger as volume flux."""

    functor = SoilWaterTransportFunctor()

    theta = torch.tensor([0.3, 0.3], dtype=torch.float32)
    captured = {}

    def _fake_update(theta, pressure_head, source_term, dt):
        del pressure_head
        captured["source_term"] = source_term.clone()
        return {
            "theta_new": theta,
            "water_flux": torch.zeros(theta.numel() + 1),
            "water_balance_residual": torch.zeros(()),
            "clipping_residual": torch.zeros_like(theta),
            "richards_iterations": 1,
            "dt": dt,
        }

    functor._compute_iphs_richards_state_update = _fake_update
    infiltration_flux = theta.new_tensor(1.0e-3)
    evapotranspiration_flux = theta.new_tensor(2.0e-4)
    state = {
        "soil_moisture": theta.clone(),
        "infiltration_flux": infiltration_flux,
        "evapotranspiration_flux": evapotranspiration_flux,
    }

    _ = functor.forward(state=state, time=0.0, dt=3600.0)

    expected_flux = (
        infiltration_flux - evapotranspiration_flux
    ) / functor.water_density
    assert "source_term" in captured
    assert torch.allclose(captured["source_term"], expected_flux)
    assert not hasattr(functor, "infiltration_rate_default")
    assert not hasattr(functor, "evapotranspiration_rate_default")


