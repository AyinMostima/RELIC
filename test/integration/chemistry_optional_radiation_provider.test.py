"""The open-mechanism ChemistryFunctor declares atmospheric_radiation as an OPTIONAL provider.

shortwave_down and cos_zenith are required inputs validated in ChemistryFunctor.forward
(J_NO2 = J_max (SW_down / SW_ref) max(0, cos theta_z), MCM J(O1D)/J(NO2) ratio), so the
radiation Functor orders the graph when it is registered but is not a semantic requirement:
the same quantities may be external forcing (precedent: AtmosphericAdvectionDiffusionFunctor
declares atmospheric_turbulence as optional).
"""

import torch

from relic.core.dependency_graph import normalize_dependency_declaration
from relic.core.simulator import RELICSimulator
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor

FORCING = {
    "air_temperature": 298.15,  # K
    "atmospheric_pressure": 101325.0,  # Pa
    "shortwave_down": 800.0,  # W m-2, external forcing
    "cos_zenith": 0.9,  # 1, external forcing
    "specific_humidity": 0.01,  # kg kg-1
    "no_mixing_ratio": 5.0e-9,  # mol mol-1
    "no2_mixing_ratio": 10.0e-9,
    "o3_mixing_ratio": 40.0e-9,
    "co_mixing_ratio": 150.0e-9,
    "ch4_mixing_ratio": 1.8e-6,
    "isoprene_mixing_ratio": 2.0e-9,
}


def test_open_mechanism_declares_radiation_optional_and_closed_box_declares_nothing():
    chem = ChemistryFunctor(name="atmospheric_chemistry", physical_boundary="open_mechanism")
    declared = normalize_dependency_declaration(chem.get_dependencies(), owner=chem.name)
    assert declared.optional == ("atmospheric_radiation",)
    assert declared.required == () and declared.lagged == ()
    box = ChemistryFunctor(name="box_chemistry", physical_boundary="closed_constant_volume")
    assert normalize_dependency_declaration(box.get_dependencies(), owner=box.name).same_step == ()


def test_strict_simulator_runs_two_governed_steps_with_external_radiation_forcing():
    dt = 60.0
    simulator = RELICSimulator({"max_time": 2 * dt, "time_step": dt, "device": "cpu",
                                "conservation_checks": False, "strict_dependencies": True,
                                "strict_scale_detection": False})
    state = {key: torch.tensor(value, dtype=torch.float64) for key, value in FORCING.items()}
    simulator.initialize_state({key: value.clone() for key, value in state.items()})
    simulator.register_functor(ChemistryFunctor(name="atmospheric_chemistry", physical_boundary="open_mechanism"),
                               execution_group="fast")
    simulator.step(dt=dt)
    simulator.step(dt=dt)
    assert simulator.step_count == 2
    o3 = simulator.state_manager.get_state("o3_mixing_ratio")
    assert torch.isfinite(o3).all() and float(o3) != FORCING["o3_mixing_ratio"]
    torch.testing.assert_close(simulator.state_manager.get_state("shortwave_down"), state["shortwave_down"])


def test_registered_radiation_still_runs_before_chemistry():
    simulator = RELICSimulator({"time_step": 60.0, "device": "cpu", "strict_scale_detection": False})
    # Consumer registered first: only the declared optional edge orders the producer first.
    simulator.register_functor(ChemistryFunctor(name="atmospheric_chemistry", physical_boundary="open_mechanism"))
    simulator.register_functor(RadiationFunctor(device=torch.device("cpu")))
    order = simulator.execution_order
    assert order.index("atmospheric_radiation") < order.index("atmospheric_chemistry")
