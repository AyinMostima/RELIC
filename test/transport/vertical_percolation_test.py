import torch

# ASCII-only comments.
# Test that VerticalPercolationTransport returns soil moisture increment consistent
# with drainage mass flux and that diagnostics are namespaced.

from relic.core.state import UpdateItem, UpdateMode, StateManager
from relic.scientific.hydrosphere.transport.vertical_percolation import VerticalPercolationTransport


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _state(theta: float) -> dict[str, torch.Tensor]:
    return {
        "soil_moisture": torch.tensor(theta, dtype=torch.float32),
        "soil_temperature": torch.tensor(298.15, dtype=torch.float32),
    }


def test_vertical_percolation_returns_boundary_flux_without_advancing_storage():
    vp = VerticalPercolationTransport()
    state = _state(0.40)
    theta_before = state["soil_moisture"].clone()

    out = vp.forward(state, time=0.0, dt=60.0)

    assert "drainage_flux" in out
    assert "soil_moisture" not in out
    assert "volume_fraction_of_water_in_soil" not in out
    assert torch.equal(state["soil_moisture"], theta_before)
    drainage = out["drainage_flux"]
    assert isinstance(drainage, UpdateItem)
    assert drainage.mode == UpdateMode.REPLACE
    assert drainage.per_timestep is False
    assert drainage.unit == "kg m-2 s-1"
    assert torch.all(drainage.value > 0.0)


def test_vertical_percolation_updates_state_manager():
    vp = VerticalPercolationTransport()

    state_manager = StateManager()
    state_manager.set_state(_state(0.40), validate=True)
    initial_moisture = state_manager.get_state()["soil_moisture"].clone()

    updates = vp(state_manager.get_state(), time=0.0, dt=60.0)
    state_manager.apply_updates(updates, dt=60.0, check_conservation=False)

    new_state = state_manager.get_state()
    assert torch.equal(new_state["soil_moisture"], initial_moisture)
    assert torch.equal(new_state["drainage_flux"], updates["drainage_flux"].value)
    assert all(
        key.startswith("diagnostics.vertical_percolation.")
        for key in new_state
        if key.startswith("diagnostics.")
    )


def test_vertical_percolation_no_drainage_below_field_capacity():
    vp = VerticalPercolationTransport()

    theta_fc = vp.theta_fc
    theta = torch.clamp(theta_fc - theta_fc.new_tensor(0.01), min=0.0)
    state = _state(float(theta.detach()))

    out = vp.forward(state, time=0.0, dt=60.0)
    drainage_flux = _unwrap_update(out["drainage_flux"])
    assert torch.allclose(drainage_flux, torch.zeros_like(drainage_flux))


def test_vertical_percolation_has_no_old_output_aliases():
    vp = VerticalPercolationTransport()
    out = vp.forward(_state(0.40), time=0.0, dt=60.0)

    assert {
        "percolation_drainage_flux",
        "soil_drainage",
        "subsurface_runoff_flux",
        "volume_fraction_of_water_in_soil",
        "diagnostics.percolation",
    }.isdisjoint(out)

