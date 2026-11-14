"""Governed composition in which soil water feeds back on isoprene emission.

bucket_soil_moisture -> soil_water_retention -> water_stress -> megan_bvoc_model,
with thermal_stress as the second declared stress control. Every node is an
existing process class registered through the public NodeRegistry; the graph is
compiled strictly without adapters and the compiled plan is executed, because a
strict compile alone does not show that the StateManager contracts admit the
run. Expected values are identities between ports of the same step, so the
cases hold for any retention parameters of the owner.
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

_RADIATION_FORCING = (
    "air_temperature",
    "incoming_shortwave_potential",
    "cos_zenith",
    "surface_temperature",
    "relative_humidity",
    "surface_albedo",
    "surface_emissivity",
    "cloud_fraction",
)
_MEGAN = "relic.scientific.biosphere.bvoc.megan_model.MEGANModel"
_MEGAN_INPUTS = {
    "air_temperature": "forcing",
    "ppfd": "atmospheric_radiation",
    "leaf_area_index": "forcing",
    "atmospheric_co2_ppm": "forcing",
}
_MEGAN_OUTPUTS = ["isoprene_emission", "total_bvoc_emission", "bvoc_carbon_allocation_flux"]
_MEGAN_CONFIG = {"enabled_compounds": ["isoprene"], "emission_pathway": "standard"}

# name -> (class path, constructor selection, {input port: source}, [outputs])
# with source "forcing", "state" or the name of the producing node.
_WATER_BALANCE = {
    "atmospheric_radiation": (
        "relic.scientific.atmosphere.radiation.RadiationFunctor",
        {},
        {port: "forcing" for port in _RADIATION_FORCING},
        ["net_radiation", "ppfd"],
    ),
    "priestley_taylor_et": (
        "relic.scientific.hydrosphere.evapotranspiration.priestley_taylor."
        "PriestleyTaylorET",
        {},
        {
            "net_radiation": "atmospheric_radiation",
            "air_temperature": "forcing",
            "atmospheric_pressure": "forcing",
        },
        ["evapotranspiration_flux", "latent_heat_flux"],
    ),
    "bucket_soil_moisture": (
        "relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture",
        {},
        {
            "soil_moisture": "state",
            "precipitation_flux": "forcing",
            "evapotranspiration_flux": "priestley_taylor_et",
        },
        ["soil_moisture", "surface_runoff_flux", "drainage_flux"],
    ),
}
_GRAPHS = {
    "unstressed": {
        **_WATER_BALANCE,
        "megan_bvoc_model": (_MEGAN, _MEGAN_CONFIG, _MEGAN_INPUTS, _MEGAN_OUTPUTS),
    },
    "soil_water_feedback": {
        **_WATER_BALANCE,
        "soil_water_retention": (
            "relic.scientific.hydrosphere.soil_moisture.retention.SoilWaterRetention",
            {},
            {"soil_moisture": "bucket_soil_moisture"},
            ["soil_water_potential"],
        ),
        "water_stress": (
            "relic.scientific.biosphere.stress.water_stress.WaterStress",
            {},
            {"soil_water_potential": "soil_water_retention"},
            ["water_stress_factor"],
        ),
        "thermal_stress": (
            "relic.scientific.biosphere.stress.thermal_stress.ThermalStress",
            {},
            {"air_temperature": "forcing"},
            ["temperature_stress_factor"],
        ),
        "megan_bvoc_model": (
            _MEGAN,
            {
                **_MEGAN_CONFIG,
                "use_stress_functors": True,
                "stress_factors": ["water", "temperature"],
            },
            {
                **_MEGAN_INPUTS,
                "water_stress_factor": "water_stress",
                "temperature_stress_factor": "thermal_stress",
            },
            _MEGAN_OUTPUTS,
        ),
    },
}

# Synthetic one-cell drivers in the SSOT unit of their port: temperatures [K],
# incoming_shortwave_potential [W m-2], atmospheric_pressure [Pa],
# precipitation_flux [kg m-2 s-1], leaf_area_index [m2 m-2],
# atmospheric_co2_ppm [umol mol-1], the remaining drivers [1].
_FORCING = {
    "air_temperature": 303.15,
    "incoming_shortwave_potential": 600.0,
    "cos_zenith": 0.7,
    "surface_temperature": 305.15,
    "relative_humidity": 0.6,
    "surface_albedo": 0.2,
    "surface_emissivity": 0.97,
    "cloud_fraction": 0.3,
    "leaf_area_index": 3.0,
    "atmospheric_co2_ppm": 400.0,
    "atmospheric_pressure": 101325.0,
    "precipitation_flux": 0.0,
}
_ONE_CELL_GRID = {
    "grid_type": "regular",
    "coord_system": "cartesian",
    "shape": [1, 1],
    "spacing": [1.0, 1.0],
    "origin": [0.0, 0.0],
}
_STEP_SECONDS = 1800.0


def _compiled(graph_name):
    registry = NodeRegistry(
        ssot_catalog=SSOTCatalog.from_parameter_manager(get_global_manager())
    )
    forcing, nodes = [], {}
    for name, (dotted, selection, inputs, outputs) in _GRAPHS[graph_name].items():
        module, class_name = dotted.rsplit(".", 1)
        arguments = deepcopy(selection)
        pathway = arguments.pop("emission_pathway", None)
        keywords = {"config": arguments} if arguments else {}
        if pathway is not None:
            megan = importlib.import_module(module)
            keywords["emission_pathway"] = megan.EmissionPathway(pathway)
        functor = getattr(importlib.import_module(module), class_name)(**keywords)
        assert functor.name == name
        registry.register_functor_instance(functor)
        node_inputs = {}
        for port, source in inputs.items():
            if source == "state":
                node_inputs[port] = {"state": port}
                continue
            if source == "forcing" and port not in forcing:
                forcing.append(port)
            node_inputs[port] = {"link": [source, port]}
        nodes[name] = {
            "class_type": "Functor:" + dotted,
            "inputs": node_inputs,
            "outputs": list(outputs),
            "config": deepcopy(selection),
        }
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {"class_type": "State:ExternalForcing", "outputs": forcing},
                **nodes,
            },
            "metadata": {"profile_id": "COMPOSITION-" + graph_name},
        }
    )
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    return plan, registry


def _initial_state():
    # theta_0 is the midpoint of the plant-available range of the bucket read
    # from the owner, so evapotranspiration draws the store down in each step.
    manager = get_global_manager()
    theta_0 = 0.5 * (
        float(manager.get("hydrosphere.soil_moisture.wilting_point"))
        + float(manager.get("hydrosphere.soil_moisture.field_capacity"))
    )
    return {
        name: torch.full((1, 1), value, dtype=torch.float32)
        for name, value in {**_FORCING, "soil_moisture": theta_0}.items()
    }


def _executor(plan, registry):
    return GraphExecutor(
        plan,
        registry=registry,
        simulator_config={
            "device": "cpu",
            "auto_report": False,
            "enable_auto_scale": True,
            "strict_scale_detection": True,
            "auto_scale": {
                "source_grid": deepcopy(_ONE_CELL_GRID),
                "target_grid": deepcopy(_ONE_CELL_GRID),
            },
        },
    )


def _run(plan, registry, n_steps):
    with torch.no_grad():
        state = _executor(plan, registry).execute(
            _initial_state(), n_steps=n_steps, dt=_STEP_SECONDS
        )
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if torch.is_tensor(value)
    }


def test_soil_water_feedback_graph_compiles_strictly_without_adapter():
    plan, _ = _compiled("soil_water_feedback")

    assert plan.report.adapter_insertions == []
    assert plan.report.metadata["compile_run_witness"]["passed"] is True
    order = [step.node_id for step in plan.steps]
    assert sorted(order) == sorted(_GRAPHS["soil_water_feedback"])
    chain = [
        "priestley_taylor_et",
        "bucket_soil_moisture",
        "soil_water_retention",
        "water_stress",
        "megan_bvoc_model",
    ]
    assert [order.index(name) for name in chain] == sorted(
        order.index(name) for name in chain
    )
    assert order.index("thermal_stress") < order.index("megan_bvoc_model")
    required = plan.report.metadata["state_boundary_lowering"][
        "required_initial_state_keys"
    ]
    assert sorted(required) == sorted([*_FORCING, "soil_moisture"])


def test_soil_water_state_reaches_the_emission_in_every_governed_step():
    # References: Manabe (1969) bucket balance; van Genuchten (1980) retention;
    #   water_stress.py and megan_model.py (Guenther et al. 2012 reduced closure).
    # Mathematical expression, for the committed ports of step k:
    #   psi_k = R(theta_k); f_w,k = W(psi_k); E_k = E0_k f_w,k f_T,k,
    #   and theta_2 < theta_1, psi_2 < psi_1 without precipitation.
    # Parameter meanings: theta_k the bucket water content after step k, R the
    #   retention process, W the water stress process, f_T the thermal control,
    #   E0_k the emission of the same graph without stress controls.
    # Units: theta [m3 m-3]; psi [MPa]; f_w, f_T [1]; E, E0 [mol m-2 s-1].
    # Domain/constraints: one cell, constant drivers, fixed step, CPU, float32.
    #   R and W are re-evaluated by a governed call of a fresh instance on the
    #   committed input, so their identities are exact; the emission product is
    #   compared to float32 rounding of three multiplications.
    # Sign convention: psi <= 0; emission positive upward; evapotranspiration
    #   removes water from the bucket.
    plan, registry = _compiled("soil_water_feedback")
    reference_plan, reference_registry = _compiled("unstressed")
    retention = importlib.import_module(
        "relic.scientific.hydrosphere.soil_moisture.retention"
    ).SoilWaterRetention()
    water_stress = importlib.import_module(
        "relic.scientific.biosphere.stress.water_stress"
    ).WaterStress()

    states = [_run(plan, registry, n_steps) for n_steps in (1, 2)]
    references = [
        _run(reference_plan, reference_registry, n_steps) for n_steps in (1, 2)
    ]

    for state, reference in zip(states, references):
        assert torch.equal(state["soil_moisture"], reference["soil_moisture"])
        potential = retention(
            {"soil_moisture": state["soil_moisture"]}, 0.0, _STEP_SECONDS
        )["soil_water_potential"].value
        assert torch.equal(state["soil_water_potential"], potential)
        assert bool((state["soil_water_potential"] <= 0).all())
        factor = water_stress(
            {"soil_water_potential": state["soil_water_potential"]},
            0.0,
            _STEP_SECONDS,
        )["water_stress_factor"].value
        assert torch.equal(state["water_stress_factor"], factor)
        expected = (
            reference["isoprene_emission"]
            * state["water_stress_factor"]
            * state["temperature_stress_factor"]
        )
        assert torch.isfinite(state["isoprene_emission"]).all()
        assert torch.allclose(
            state["isoprene_emission"], expected, rtol=1.0e-6, atol=0.0
        )
    assert bool((states[1]["soil_moisture"] < states[0]["soil_moisture"]).all())
    assert bool(
        (states[1]["soil_water_potential"] < states[0]["soil_water_potential"]).all()
    )


def test_drier_soil_lowers_the_emission_through_the_governed_chain(
    restore_global_parameter_manager,
):
    # References: van Genuchten (1980) retention; water_stress.py; megan_model.py.
    # Mathematical expression: with a retention shape for which the
    #   plant-available range of the bucket spans the stress window,
    #   0 < f_w < 1 and E = E0 f_w f_T < E0 f_T.
    # Parameter meanings: alpha and n are synthetic retention values of this
    #   test (the pair listed for the medium-fine class in
    #   richards_solver.test.py); f_w, f_T the stress controls; E0 the emission
    #   of the graph without stress controls.
    # Units: alpha [m-1]; n [1]; f_w, f_T [1]; E, E0 [mol m-2 s-1].
    # Domain/constraints: the values must lie inside the owner bounds and are
    #   written through the owner with validation; the manager is restored by
    #   the fixture. One cell, one governed step, CPU, float32.
    # Sign convention: emission positive upward; a drier soil has a more
    #   negative potential and a smaller f_w.
    manager = restore_global_parameter_manager
    for key, value, units in (
        ("hydrosphere.soil_moisture.alpha_vg", 0.83, "m-1"),
        ("hydrosphere.soil_moisture.n_vg", 1.25, "dimensionless"),
    ):
        lower, upper = manager.get_parameter(key).bounds
        assert lower <= value <= upper, (key, value, (lower, upper))
        manager.set_value(key, value, units=units, validate=True, notify=False)
        assert float(manager.get(key)) == pytest.approx(value, rel=1.0e-6)

    state = _run(*_compiled("soil_water_feedback"), 1)
    reference = _run(*_compiled("unstressed"), 1)

    water_factor = state["water_stress_factor"]
    assert bool(((water_factor > 0) & (water_factor < 1)).all())
    expected = (
        reference["isoprene_emission"]
        * water_factor
        * state["temperature_stress_factor"]
    )
    assert torch.allclose(state["isoprene_emission"], expected, rtol=1.0e-6, atol=0.0)
    assert bool(
        (
            state["isoprene_emission"]
            < reference["isoprene_emission"] * state["temperature_stress_factor"]
        ).all()
    )


def test_soil_water_feedback_restarts_bitwise_from_a_checkpoint(tmp_path):
    plan, registry = _compiled("soil_water_feedback")
    uninterrupted = _run(plan, registry, 4)

    first = _executor(plan, registry)
    with torch.no_grad():
        first.execute(_initial_state(), n_steps=2, dt=_STEP_SECONDS)
    checkpoint = tmp_path / "after_two_steps.pt"
    first.create_checkpoint(checkpoint)
    with torch.no_grad():
        resumed = _executor(plan, registry).execute_from_checkpoint(
            _initial_state(), checkpoint, n_steps=2, dt=_STEP_SECONDS
        )

    resumed = {
        name: value.detach().cpu()
        for name, value in resumed.items()
        if torch.is_tensor(value)
    }
    assert sorted(resumed) == sorted(uninterrupted)
    for name, value in uninterrupted.items():
        assert torch.equal(resumed[name], value), name


def test_three_factor_declaration_without_an_ozone_producer_is_rejected():
    # The undeclared default keeps the three controls: a graph that supplies no
    # ozone_stress_factor must not compile, so an absent control is never
    # replaced by a silent default.
    graphs = deepcopy(_GRAPHS["soil_water_feedback"])
    dotted, selection, inputs, outputs = graphs["megan_bvoc_model"]
    selection = {key: value for key, value in selection.items() if key != "stress_factors"}
    _GRAPHS["three_factor_probe"] = {
        **graphs,
        "megan_bvoc_model": (dotted, selection, inputs, outputs),
    }
    try:
        with pytest.raises(ValueError, match="ozone_stress_factor"):
            _compiled("three_factor_probe")
    finally:
        del _GRAPHS["three_factor_probe"]


def test_governed_steps_carry_gradients_to_the_manager_bindings(
    restore_global_parameter_manager,
):
    # References: architecture SSOT Sections 4.2 and 4.4.2 (one manager-owned binding per
    #   learnable key, held by reference); Guenther et al. (2012) Eq. (2) for the linear
    #   emission factor; van Genuchten (1980) for the retention shape parameters.
    # Mathematical expression: after two governed steps E_2 = eps g(theta_2, ...), so
    #   dE_2/d eps = E_2 / eps; dE_2/d alpha and dE_2/d n pass through
    #   psi = R(theta; alpha, n) and f_w = W(psi) into the emission law.
    # Parameter meanings: eps the isoprene dry-mass emission factor binding; alpha and n
    #   the retention bindings; C_T1 and alpha_light the MEGAN temperature and light
    #   coefficient bindings.
    # Units: E [mol m-2 s-1]; eps [ug g-1 DW h-1]; alpha [m-1]; n [1]; C_T1 [J mol-1];
    #   alpha_light [m2 s umol-1].
    # Domain/constraints: the retention shape of the drier-soil test so that 0 < f_w < 1;
    #   values written through the owner with validation; the manager is restored by the
    #   fixture. One cell, two governed steps, CPU, float32, autograd enabled.
    # Sign convention: emission positive upward.
    manager = restore_global_parameter_manager
    for key, value, units in (
        ("hydrosphere.soil_moisture.alpha_vg", 0.83, "m-1"),
        ("hydrosphere.soil_moisture.n_vg", 1.25, "dimensionless"),
    ):
        manager.set_value(key, value, units=units, validate=True, notify=False)
    plan, registry = _compiled("soil_water_feedback")
    state = _executor(plan, registry).execute(_initial_state(), n_steps=2, dt=_STEP_SECONDS)
    emission = state["isoprene_emission"].sum()
    assert emission.requires_grad
    keys = (
        "biosphere.bvoc.emission_factors.isoprene",
        "biosphere.bvoc.megan.ct1_isoprene",
        "biosphere.bvoc.megan.alpha_light",
        "hydrosphere.soil_moisture.alpha_vg",
        "hydrosphere.soil_moisture.n_vg",
    )
    bindings = [manager.nn_parameters[key] for key in keys]
    gradients = torch.autograd.grad(emission, bindings, allow_unused=True)
    for key, gradient in zip(keys, gradients):
        assert gradient is not None, key
        assert torch.isfinite(gradient).all() and bool((gradient != 0).all()), key
    epsilon = bindings[0]
    torch.testing.assert_close(
        gradients[0], (emission / epsilon).detach(), rtol=1.0e-5, atol=0.0, check_dtype=False
    )
