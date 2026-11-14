"""Strict publication compilation of single existing process Functors.

A graph that carries a publication profile id must declare shape and support
for every port it names (validator code publication_port_support_incomplete).
The declarations are projected from the SCM variable contracts through
SSOTCatalog and NodeRegistry, so each case registers the real process class,
compiles a minimal graph without any adapter and reads the projected ports.
The compiled plan is then executed through GraphExecutor, because a process
whose outputs the StateManager contracts refuse compiles but cannot run.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.parameters import get_global_manager

# Support of one port: (spatial_kind, area_basis, temporal_basis,
# vertical_basis, shape dimensions).
_CELL_INTENSIVE = ("grid_cell", None, "instantaneous", "none", ["cell"])
_GROUND_AREA_FLUX = (
    "grid_cell",
    "ground_area",
    "instantaneous",
    "none",
    ["cell"],
)

# One case per process: the nodes of the minimal graph as
# name -> (class path, {input port: source}, [output ports]) with source
# "forcing", "state" or the name of the producing node, the support that
# the contracts of the process's own ports must declare, and one synthetic
# one-cell value per external input in the SSOT unit of its port
# (air_temperature and surface_temperature [K], soil_water_potential [MPa],
# incoming_shortwave_potential [W m-2], leaf_area_index [m2 m-2],
# soil_moisture [m3 m-3], the remaining inputs [1]).
_CASES = {
    "thermal_stress": {
        "nodes": {
            # air_temperature is external forcing (no relic Functor writes it)
            # and ThermalStress declares no Functor dependency, so the forcing
            # node and thermal_stress alone are its minimal strict graph.
            "thermal_stress": (
                "relic.scientific.biosphere.stress.thermal_stress.ThermalStress",
                {"air_temperature": "forcing"},
                [
                    "temperature_stress_factor",
                    "heat_stress_factor",
                    "cold_stress_factor",
                ],
            ),
        },
        "support": {
            "temperature_stress_factor": _CELL_INTENSIVE,
            "heat_stress_factor": _CELL_INTENSIVE,
            "cold_stress_factor": _CELL_INTENSIVE,
        },
        "state": {"air_temperature": 308.15},
    },
    "water_stress": {
        "nodes": {
            "water_stress": (
                "relic.scientific.biosphere.stress.water_stress.WaterStress",
                {"soil_water_potential": "forcing"},
                ["water_stress_factor"],
            ),
        },
        "support": {
            "soil_water_potential": _CELL_INTENSIVE,
            "water_stress_factor": _CELL_INTENSIVE,
        },
        "state": {"soil_water_potential": -0.5},
    },
    "soil_water_retention": {
        "nodes": {
            # soil_moisture is supplied externally here (no soil-water process
            # is in the graph), so the retention process alone is its minimal
            # strict graph; 0.25 lies inside [residual_water_content, porosity]
            # for every value the owner bounds admit.
            "soil_water_retention": (
                "relic.scientific.hydrosphere.soil_moisture.retention."
                "SoilWaterRetention",
                {"soil_moisture": "forcing"},
                ["soil_water_potential"],
            ),
        },
        "support": {
            "soil_moisture": _CELL_INTENSIVE,
            "soil_water_potential": _CELL_INTENSIVE,
        },
        "state": {"soil_moisture": 0.25},
    },
    "simple_photosynthesis": {
        "nodes": {
            # SimpleCanopyPhotosynthesis declares atmospheric_radiation as its
            # PPFD provider, so the minimal strict graph links ppfd from it.
            "atmospheric_radiation": (
                "relic.scientific.atmosphere.radiation.RadiationFunctor",
                {
                    "air_temperature": "forcing",
                    "incoming_shortwave_potential": "forcing",
                    "cos_zenith": "forcing",
                    "surface_temperature": "forcing",
                    "relative_humidity": "forcing",
                    "surface_albedo": "forcing",
                    "surface_emissivity": "forcing",
                    "cloud_fraction": "forcing",
                },
                ["ppfd"],
            ),
            "simple_photosynthesis": (
                "relic.scientific.biosphere.surrogates.simple_photosynthesis."
                "SimpleCanopyPhotosynthesis",
                {
                    "ppfd": "atmospheric_radiation",
                    "air_temperature": "forcing",
                    "leaf_area_index": "forcing",
                    "soil_moisture": "state",
                },
                [
                    "gross_primary_productivity",
                    "ecosystem_respiration",
                    "net_ecosystem_exchange",
                ],
            ),
        },
        "support": {
            "ecosystem_respiration": _GROUND_AREA_FLUX,
            "net_ecosystem_exchange": _GROUND_AREA_FLUX,
        },
        "state": {
            "air_temperature": 293.15,
            "incoming_shortwave_potential": 600.0,
            "cos_zenith": 0.7,
            "surface_temperature": 295.15,
            "relative_humidity": 0.6,
            "surface_albedo": 0.2,
            "surface_emissivity": 0.97,
            "cloud_fraction": 0.3,
            "leaf_area_index": 3.0,
            "soil_moisture": 0.25,
        },
    },
}


def _publication_graph(process, registry):
    """Register the case's process classes and build its one-profile graph."""
    forcing, nodes = [], {}
    for name, (dotted, inputs, outputs) in _CASES[process]["nodes"].items():
        module, class_name = dotted.rsplit(".", 1)
        functor = getattr(importlib.import_module(module), class_name)()
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
        }
    return GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {
                    "class_type": "State:ExternalForcing",
                    "outputs": forcing,
                },
                **nodes,
            },
            "metadata": {"profile_id": "PROCESS-COMPILE-" + process},
        }
    )


@pytest.mark.parametrize("process", list(_CASES))
def test_process_publication_graph_compiles_strictly_without_adapter(process):
    case = _CASES[process]
    registry = NodeRegistry(
        ssot_catalog=SSOTCatalog.from_parameter_manager(get_global_manager())
    )
    graph = _publication_graph(process, registry)

    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)

    assert plan.report.adapter_insertions == []
    assert sorted(step.node_id for step in plan.steps) == sorted(case["nodes"])
    assert plan.report.metadata["compile_run_witness"]["passed"] is True
    definition = registry.get_definition(
        "Functor:" + case["nodes"][process][0]
    )
    for variable, expected in case["support"].items():
        ports = [
            table[variable]
            for table in (definition.inputs, definition.outputs)
            if variable in table
        ]
        assert ports, variable
        for port in ports:
            assert port.missing_support_declarations() == (), variable
            assert (
                port.spatial_kind,
                port.area_basis,
                port.temporal_basis,
                port.vertical_basis,
                port.shape["dimensions"],
            ) == expected, variable


@pytest.mark.parametrize("process", list(_CASES))
def test_process_publication_graph_replaces_declared_outputs_each_step(process):
    # References: StateManager update modes (architecture_overview.md Section
    #   4.2); SCM contracts of the declared outputs (incrementable: false).
    # Mathematical expression: y_k = O(x) for every governed step k with
    #   constant external inputs x, so y_1 = y_2 (REPLACE). An ACCUMULATE
    #   commit would give y_2 = y_1 + O(x) * h or be refused by the contract.
    # Parameter meanings: O the process law, x the synthetic one-cell inputs,
    #   y_k a declared output after k steps, h the fixed step.
    # Units: h [s]; x and y keep the SSOT units of their ports.
    # Domain/constraints: one cell, fixed step, no state writer in the graph,
    #   CPU; equality is exact.
    # Sign convention: unchanged from the ports.
    case = _CASES[process]
    registry = NodeRegistry(
        ssot_catalog=SSOTCatalog.from_parameter_manager(get_global_manager())
    )
    plan = compile_graph(
        _publication_graph(process, registry),
        registry,
        strict=True,
        auto_adapt=False,
    )
    required = plan.report.metadata["state_boundary_lowering"][
        "required_initial_state_keys"
    ]
    assert sorted(required) == sorted(case["state"])

    final_states = []
    for n_steps in (1, 2):
        executor = GraphExecutor(
            plan,
            registry=registry,
            simulator_config={"device": "cpu", "auto_report": False},
        )
        initial = {
            name: torch.full((1, 1), value, dtype=torch.float32)
            for name, value in case["state"].items()
        }
        with torch.no_grad():
            final_states.append(
                executor.execute(initial, n_steps=n_steps, dt=1800.0)
            )

    for name in case["nodes"][process][2]:
        after_one, after_two = final_states[0][name], final_states[1][name]
        assert torch.isfinite(after_one).all(), name
        assert torch.equal(after_one, after_two), name
