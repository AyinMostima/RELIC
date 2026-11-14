"""Leaf energy balance in the strictly compiled daily soil-water / BVOC graph.

Graph (every node an existing process class registered through the public NodeRegistry,
no adapters, dt = 86400 s): forcing -> atmospheric_radiation; soil_water_retention reads the
bucket soil_moisture as STATE -> water_stress -> farquhar_photosynthesis ->
medlyn_stomatal_conductance -> canopy_conductance -> penman_monteith_et (daily) -> bucket;
canopy_conductance + net_radiation -> leaf_energy_balance (daily) -> leaf_temperature ->
megan_bvoc_model (temperature_source leaf_temperature, temperature_history with the leaf
T24/T240 states read as STATE) and leaf_temperature_max -> thermal_stress (heat) ->
megan_bvoc_model; thermal_stress_cold on air_temperature_min; water_stress -> MEGAN.

Cells: wet / dry soil x 20 / 30 degC daily mean, plus a no-canopy cell.
Checked here: strict compile and order; drying lowers g_c and raises T_leaf - T_air; the
latent heat of the balance equals the Penman-Monteith latent heat of the same graph; the
day-0 soil-water x temperature interaction of gamma_T and of the heat factor is non-zero
(it is exactly zero with air-temperature inputs); no-canopy cells keep T_air; a restart
from the carried state is bitwise equal to the continuous run.
"""

from __future__ import annotations

import importlib
from copy import deepcopy

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.parameters import get_global_manager

DAY = 86400.0
N_DAYS = 8
SPLIT = 4
P = "diagnostics.leaf_energy_balance."
LEAF_HISTORY = ["leaf_temperature_24h_mean", "leaf_temperature_240h_mean"]
_THERMAL = "relic.scientific.biosphere.stress.thermal_stress.ThermalStress"
_RADIATION_FORCING = ("air_temperature", "incoming_shortwave_potential", "cos_zenith", "surface_temperature",
                      "relative_humidity", "surface_albedo", "surface_emissivity", "cloud_fraction")
_MEDLYN_CONFIG = {"surface_conditions": "leaf_surface_equals_canopy_air",
                  "canopy_conductance_upscaling": "effective_leaf_area_index", "nonpositive_leaf_area": "no_canopy"}
_MEGAN_CONFIG = {"enabled_compounds": ["isoprene"], "emission_pathway": "standard", "use_stress_functors": True,
                 "stress_factors": ["water", "heat", "cold"], "temperature_history": True,
                 "temperature_source": "leaf_temperature"}
_GRAPH = {
    "atmospheric_radiation": ("relic.scientific.atmosphere.radiation.RadiationFunctor", {},
                              {port: "forcing" for port in _RADIATION_FORCING}, ["net_radiation", "ppfd"]),
    "soil_water_retention": ("relic.scientific.hydrosphere.soil_moisture.retention.SoilWaterRetention", {},
                             {"soil_moisture": "state"}, ["soil_water_potential"]),
    "water_stress": ("relic.scientific.biosphere.stress.water_stress.WaterStress", {},
                     {"soil_water_potential": "soil_water_retention"}, ["water_stress_factor"]),
    "farquhar_photosynthesis": ("relic.scientific.biosphere.photosynthesis.farquhar.FarquharPhotosynthesis", {},
                                {"air_temperature": "forcing", "atmospheric_co2_ppm": "forcing",
                                 "ppfd": "atmospheric_radiation", "vapor_pressure_deficit": "forcing",
                                 "leaf_area_index": "forcing", "water_stress_factor": "water_stress"},
                                ["gross_photosynthesis", "net_photosynthesis", "gross_primary_productivity",
                                 "electron_transport_rate"]),
    "medlyn_stomatal_conductance": ("relic.scientific.biosphere.photosynthesis.stomatal.MedlynStomatalConductance",
                                    _MEDLYN_CONFIG,
                                    {"net_photosynthesis": "farquhar_photosynthesis", "leaf_area_index": "forcing",
                                     "ppfd": "atmospheric_radiation", "atmospheric_co2_ppm": "forcing",
                                     "vapor_pressure_deficit": "forcing", "air_temperature": "forcing",
                                     "atmospheric_pressure": "forcing"},
                                    ["stomatal_conductance", "canopy_conductance"]),
    "penman_monteith_et": ("relic.scientific.hydrosphere.evapotranspiration.penman_monteith.PenmanMonteithET",
                           {"canopy_resistance_source": "state", "temporal_support": "daily"},
                           {"net_radiation": "atmospheric_radiation", "air_temperature": "forcing",
                            "wind_speed": "forcing", "atmospheric_pressure": "forcing",
                            "vapor_pressure_deficit": "forcing", "air_temperature_max": "forcing",
                            "air_temperature_min": "forcing", "canopy_conductance": "medlyn_stomatal_conductance"},
                           ["evapotranspiration_flux", "latent_heat_flux"]),
    "bucket_soil_moisture": ("relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture", {},
                             {"soil_moisture": "state", "precipitation_flux": "forcing",
                              "evapotranspiration_flux": "penman_monteith_et"},
                             ["soil_moisture", "surface_runoff_flux", "drainage_flux"]),
    "leaf_energy_balance": ("relic.scientific.biosphere.photosynthesis.leaf_energy_balance.LeafEnergyBalance",
                            {"temporal_support": "daily"},
                            {"air_temperature": "forcing", "air_temperature_max": "forcing",
                             "air_temperature_min": "forcing", "vapor_pressure_deficit": "forcing",
                             "wind_speed": "forcing", "atmospheric_pressure": "forcing",
                             "net_radiation": "atmospheric_radiation",
                             "canopy_conductance": "medlyn_stomatal_conductance", "leaf_area_index": "forcing"},
                            ["leaf_temperature", "leaf_temperature_max"]),
    "thermal_stress": (_THERMAL, {"air_temperature_source": "leaf_temperature_max", "stress_outputs": ["heat_stress_factor"]},
                       {"leaf_temperature_max": "leaf_energy_balance"}, ["heat_stress_factor"]),
    "thermal_stress_cold": (_THERMAL, {"air_temperature_source": "air_temperature_min", "stress_outputs": ["cold_stress_factor"]},
                            {"air_temperature_min": "forcing"}, ["cold_stress_factor"]),
    "megan_bvoc_model": ("relic.scientific.biosphere.bvoc.megan_model.MEGANModel", _MEGAN_CONFIG,
                         {"leaf_temperature": "leaf_energy_balance", "ppfd": "atmospheric_radiation",
                          "leaf_area_index": "forcing", "atmospheric_co2_ppm": "forcing",
                          "water_stress_factor": "water_stress", "heat_stress_factor": "thermal_stress",
                          "cold_stress_factor": "thermal_stress_cold",
                          **{name: "state" for name in LEAF_HISTORY}},
                         ["isoprene_emission", "total_bvoc_emission", "bvoc_carbon_allocation_flux", *LEAF_HISTORY]),
}
_SECOND_INSTANCE_TYPES = {"thermal_stress_cold": "Functor:" + _THERMAL + "@thermal_stress_cold"}
_ONE_SOIL = {"hydrosphere.soil_moisture.alpha_vg": 0.757805, "hydrosphere.soil_moisture.n_vg": 1.420004}
_T = [293.15, 293.15, 303.15, 303.15, 293.15]
_FORCING = {"air_temperature": _T, "air_temperature_max": [t + 7.0 for t in _T], "air_temperature_min": [t - 7.0 for t in _T],
            "surface_temperature": [t + 2.0 for t in _T], "leaf_area_index": [3.0, 3.0, 3.0, 3.0, 0.0],
            "incoming_shortwave_potential": 400.0, "cos_zenith": 0.6, "relative_humidity": 0.6, "surface_albedo": 0.15,
            "surface_emissivity": 0.97, "cloud_fraction": 0.3, "atmospheric_co2_ppm": 400.0, "atmospheric_pressure": 101325.0,
            "precipitation_flux": 0.0, "wind_speed": 2.0, "vapor_pressure_deficit": 1000.0}
WET_C, DRY_C, WET_W, DRY_W, NOCAN = range(5)
_GRID = {"grid_type": "regular", "coord_system": "cartesian", "shape": [1, 5], "spacing": [1.0, 1.0], "origin": [0.0, 0.0]}


@pytest.fixture(scope="module")
def one_soil():
    manager = get_global_manager()
    previous = {key: manager.get(key) for key in _ONE_SOIL}
    policy = manager.bounds_policy
    manager.set_bounds_policy("error")
    try:
        for key, value in _ONE_SOIL.items():
            units = getattr(manager.get_parameter(key), "units", None)
            manager.set_value(key, value, validate=True, **({"units": units} if units else {}))
    finally:
        manager.set_bounds_policy(policy)
    yield manager
    for key, value in previous.items():
        manager.set_value(key, value)


def _compiled():
    registry = NodeRegistry(ssot_catalog=SSOTCatalog.from_parameter_manager(get_global_manager()))
    forcing, nodes = [], {}
    for name, (dotted, selection, inputs, outputs) in _GRAPH.items():
        module, class_name = dotted.rsplit(".", 1)
        arguments = deepcopy(selection)
        pathway = arguments.pop("emission_pathway", None)
        keywords = {"config": arguments} if arguments else {}
        if pathway is not None:
            keywords["emission_pathway"] = importlib.import_module(module).EmissionPathway(pathway)
        if name in _SECOND_INSTANCE_TYPES:
            keywords["name"] = name
        functor = getattr(importlib.import_module(module), class_name)(**keywords)
        assert functor.name == name
        class_type = _SECOND_INSTANCE_TYPES.get(name, "Functor:" + dotted)
        registry.register_functor_instance(functor, node_type_id=class_type)
        node_inputs = {}
        for port, source in inputs.items():
            if source == "state":
                node_inputs[port] = {"state": port}
                continue
            if source == "forcing" and port not in forcing:
                forcing.append(port)
            node_inputs[port] = {"link": [source, port]}
        nodes[name] = {"class_type": class_type, "inputs": node_inputs, "outputs": list(outputs), "config": deepcopy(selection)}
    graph = GraphIR.from_dict({"schema_version": "1.0.0",
                               "nodes": {"forcing": {"class_type": "State:ExternalForcing", "outputs": forcing}, **nodes},
                               "metadata": {"profile_id": "LEAF-ENERGY-BALANCE-DAILY"}})
    return compile_graph(graph, registry, strict=True, auto_adapt=False), registry


def _initial(required, manager):
    wp = float(manager.get("hydrosphere.soil_moisture.wilting_point"))
    fc = float(manager.get("hydrosphere.soil_moisture.field_capacity"))
    dry = wp + 0.15 * (fc - wp)
    state = {}
    for key in required:
        if key == "soil_moisture":
            state[key] = torch.tensor([[fc, dry, fc, dry, fc]], dtype=torch.float32)
        elif key in LEAF_HISTORY:
            state[key] = torch.tensor([_T], dtype=torch.float32)  # declared spin-up: first-day air temperature
        else:
            value = _FORCING[key]
            state[key] = torch.tensor([value], dtype=torch.float32) if isinstance(value, list) else torch.full((1, 5), value, dtype=torch.float32)
    return state


def _run(plan, registry, initial, n_steps):
    executor = GraphExecutor(plan, registry=registry, simulator_config={
        "device": "cpu", "auto_report": False, "enable_auto_scale": True, "strict_scale_detection": True,
        "auto_scale": {"source_grid": deepcopy(_GRID), "target_grid": deepcopy(_GRID)}, "output_frequency": 1})
    with torch.no_grad():
        final = executor.execute(initial, n_steps=n_steps, dt=DAY)
    series = {k: torch.stack([v.reshape(-1) for v in values]) for k, values in executor.time_series.items()
              if values and isinstance(values[0], torch.Tensor)}
    return final, series


@pytest.fixture(scope="module")
def run(one_soil):
    plan, registry = _compiled()
    required = set(plan.report.metadata["state_boundary_lowering"]["required_initial_state_keys"])
    initial = _initial(required, one_soil)
    final, series = _run(plan, registry, {k: v.clone() for k, v in initial.items()}, N_DAYS)
    return plan, registry, required, initial, final, series


def _interaction(x):
    x = x.double()
    return torch.log(x[:, DRY_W]) - torch.log(x[:, DRY_C]) - torch.log(x[:, WET_W]) + torch.log(x[:, WET_C])


def test_graph_compiles_strictly_with_the_leaf_balance_between_conductance_and_megan(run):
    plan, _, required, *_ = run
    assert plan.report.adapter_insertions == []
    assert plan.report.metadata["compile_run_witness"]["passed"] is True
    order = [step.node_id for step in plan.steps]
    for before, after in (("medlyn_stomatal_conductance", "leaf_energy_balance"), ("atmospheric_radiation", "leaf_energy_balance"),
                          ("leaf_energy_balance", "megan_bvoc_model"), ("leaf_energy_balance", "thermal_stress"),
                          ("thermal_stress", "megan_bvoc_model"), ("penman_monteith_et", "bucket_soil_moisture")):
        assert order.index(before) < order.index(after), (before, after)
    steps = {step.node_id: step for step in plan.steps}
    assert steps["megan_bvoc_model"].inputs["leaf_temperature"] == {"link": ["leaf_energy_balance", "leaf_temperature"]}
    assert "air_temperature" not in steps["megan_bvoc_model"].inputs
    assert steps["thermal_stress"].inputs == {"leaf_temperature_max": {"link": ["leaf_energy_balance", "leaf_temperature_max"]}}
    assert steps["soil_water_retention"].inputs["soil_moisture"] == {"state": "soil_moisture"}
    assert set(LEAF_HISTORY) <= required and "leaf_temperature" not in required


def test_drying_lowers_canopy_conductance_and_warms_the_leaf(run):
    *_, series = run
    gc, dT = series["canopy_conductance"], series[P + "leaf_air_temperature_difference"]
    for wet, dry in ((WET_C, DRY_C), (WET_W, DRY_W)):
        assert bool(torch.all(gc[:, dry] < gc[:, wet]))
        assert bool(torch.all(dT[:, dry] > dT[:, wet]))


def test_leaf_latent_heat_is_the_penman_monteith_latent_heat_of_the_graph(run):
    *_, series = run
    canopy = slice(0, 4)
    le_leaf = series[P + "latent_heat_flux"][:, canopy].double()
    le_pm = series["latent_heat_flux"][:, canopy].double()
    assert float((le_leaf - le_pm).abs().max()) < 1e-3  # W m-2, float32 graph
    closure = series[P + "available_energy"].double() - series[P + "sensible_heat_flux"].double() - series[P + "latent_heat_flux"].double()
    assert float(closure[:, canopy].abs().max()) < 1e-4


def test_soil_water_by_temperature_interaction_is_mechanistic_on_day_0(run):
    *_, series = run
    gamma_t = _interaction(series["diagnostics.bvoc.megan.isoprene.gamma_temperature"])
    heat = _interaction(series["heat_stress_factor"])
    emission = _interaction(series["isoprene_emission"])
    # Day 0: both temperature levels start from the same soil water, so this is not the
    # covariance of soil-water trajectories. With air-temperature inputs gamma_T and the
    # heat factor do not depend on soil water and both interactions are exactly zero.
    assert abs(float(gamma_t[0])) > 1e-3
    assert abs(float(heat[0])) > 1e-3
    water = _interaction(series["water_stress_factor"])
    cold = _interaction(series["cold_stress_factor"])
    assert float(water[0]) == 0.0 and float(cold[0]) == 0.0
    factors = gamma_t + heat + water + cold + _interaction(series["diagnostics.bvoc.megan.isoprene.gamma_light"]) \
        + _interaction(series["diagnostics.bvoc.megan.co2_inhibition_factor"])
    assert torch.allclose(emission, factors, rtol=0, atol=1e-5)


def test_no_canopy_cell_keeps_the_air_temperature(run):
    *_, series = run
    assert torch.equal(series["leaf_temperature"][:, NOCAN], torch.full((N_DAYS,), _T[NOCAN], dtype=torch.float32))
    assert torch.equal(series["leaf_temperature_max"][:, NOCAN], torch.full((N_DAYS,), _T[NOCAN] + 7.0, dtype=torch.float32))


def test_leaf_history_lags_one_day(run):
    *_, series = run
    # The T24 state written at day k is the leaf temperature of day k (weight 1 at
    # dt = tau = 86400 s); MEGAN reads it at day k + 1 (state read at the start of the call).
    assert torch.equal(series["leaf_temperature_24h_mean"], series["leaf_temperature"])
    w = 0.1  # dt / tau240 at daily steps
    t240 = torch.tensor(_T, dtype=torch.float64)
    for k in range(N_DAYS):
        t240 = (1 - w) * t240 + w * series["leaf_temperature"][k].double()
        assert torch.allclose(series["leaf_temperature_240h_mean"][k].double(), t240, rtol=0, atol=1e-4)


def test_restart_with_carried_state_is_bitwise(run):
    plan, registry, required, initial, final, series = run
    final_a, seg_a = _run(plan, registry, {k: v.clone() for k, v in initial.items()}, SPLIT)
    carried = {k: final_a[k].clone() for k in required}
    final_b, seg_b = _run(plan, registry, carried, N_DAYS - SPLIT)
    compared = sorted(k for k in series if k in seg_b)
    assert len(compared) > 50
    for key in compared:
        assert torch.equal(series[key][SPLIT:], seg_b[key]), key
    for key in required:
        assert torch.equal(final[key], final_b[key]), key
    reset = dict(carried)
    for key in LEAF_HISTORY:
        reset[key] = initial[key].clone()
    _, seg_r = _run(plan, registry, reset, N_DAYS - SPLIT)
    assert not torch.equal(seg_r["isoprene_emission"][:, :4], series["isoprene_emission"][SPLIT:, :4])
