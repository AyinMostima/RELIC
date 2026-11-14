"""Declared input projection of strictly compiled graph plans.

A GraphExecutor node kernel receives the state keys its plan step binds (a
link or a state binding) and nothing else. An optional input that the graph
holds without a binding of that node, another node's output and the runtime
bookkeeping keys of the shared state are therefore not data dependencies of
the node. The compile report states the projection of every node and names
the optional inputs it leaves unbound together with the graph members that
hold them. A simulator run without a compiled plan keeps name-based reads
(test/scientific/hydrosphere/simulator_optional_inputs.test.py).
"""

from __future__ import annotations

import pytest
import torch

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.orchestration.compiler import compile_graph, plan_to_dict
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.parameters import get_global_manager

READER_CLASS_TYPE = "Functor:test.declared_input_projection.reader"
WRITER_CLASS_TYPE = "Functor:test.declared_input_projection.writer"
HOLDER_CLASS_TYPE = "Functor:test.declared_input_projection.holder"
PENMAN_MONTEITH = (
    "relic.scientific.hydrosphere.evapotranspiration.penman_monteith."
    "PenmanMonteithET"
)


def _dimensionless_state_port(name: str) -> PortSpec:
    return PortSpec(
        name=name,
        standard_name="declared_input_projection_" + name,
        units="1",
        semantics="state",
        spatial_kind="grid_cell",
        area_basis="ground_area",
        temporal_basis="instantaneous",
        vertical_basis="none",
    )


class _PointFunctor(PhysicalFieldFunctor):
    """Point model base of the test kernels (no coordinate query)."""

    SPATIAL_DIMS = None
    INPUTS: tuple[str, ...] = ()
    OPTIONAL: tuple[str, ...] = ()
    OUTPUTS: tuple[str, ...] = ()
    EXPRESSION = ""

    def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None)"
        )

    def __init__(self, name: str, config: dict | None = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            description="Kernel of the declared input projection tests.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=self.EXPRESSION,
            references=["RELIC internal declared-input verification"],
            parameters={},
            input_variables=list(self.INPUTS),
            output_variables=list(self.OUTPUTS),
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)
        self.optional_input_variables = list(self.OPTIONAL)

    def get_dependencies(self) -> list[str]:
        return []

    def _replace(self, value: torch.Tensor) -> UpdateItem:
        return UpdateItem(
            value,
            mode=UpdateMode.REPLACE,
            per_timestep=False,
            unit="1",
            source_functor=self.name,
        )


class _OptionalReader(_PointFunctor):
    """Reads x and adds the optional inputs y and q when its state holds them."""

    INPUTS = ("x", "y", "q")
    OPTIONAL = ("y", "q")
    OUTPUTS = ("z",)
    EXPRESSION = "z = x + y + q, an absent optional term is 0"
    seen: list[tuple[str, ...]] = []

    def __init__(self, name: str = "reader", config: dict | None = None) -> None:
        super().__init__(name=name, config=config)

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC internal declared-input verification.
        # Mathematical expression: z = x + y + q; an absent optional term is 0.
        # Parameter meanings: x required input; y and q optional inputs.
        # Units: x, y, q, z [1].
        # Domain/constraints: scalar tensors; the keys received are recorded.
        # Sign convention: all terms add with their own sign.
        type(self).seen.append(tuple(sorted(state)))
        total = state["x"]
        for name in self.OPTIONAL:
            if name in state:
                total = total + state[name]
        return {"z": self._replace(total)}


class _Writer(_PointFunctor):
    """Writes the name the reader lists as an optional input."""

    INPUTS = ("x",)
    OUTPUTS = ("y",)
    EXPRESSION = "y = 2 * x"

    def __init__(self, name: str = "writer", config: dict | None = None) -> None:
        super().__init__(name=name, config=config)

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC internal declared-input verification.
        # Mathematical expression: y = 2 * x.
        # Parameter meanings: x required input; y output.
        # Units: x, y [1].
        # Domain/constraints: scalar tensors.
        # Sign convention: y has the sign of x.
        return {"y": self._replace(2.0 * state["x"])}


class _Holder(_PointFunctor):
    """Binds y as a declared state, so the graph holds that name."""

    INPUTS = ("y",)
    OUTPUTS = ("w",)
    EXPRESSION = "w = y"

    def __init__(self, name: str = "holder", config: dict | None = None) -> None:
        super().__init__(name=name, config=config)

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC internal declared-input verification.
        # Mathematical expression: w = y.
        # Parameter meanings: y required input; w output.
        # Units: y, w [1].
        # Domain/constraints: scalar tensors.
        # Sign convention: w has the sign of y.
        return {"w": self._replace(state["y"] * 1.0)}


def _registry() -> NodeRegistry:
    registry = NodeRegistry(SSOTCatalog())
    ports = {name: _dimensionless_state_port(name) for name in "xyqzw"}
    for functor, node_type in (
        (_OptionalReader(), READER_CLASS_TYPE),
        (_Writer(), WRITER_CLASS_TYPE),
        (_Holder(), HOLDER_CLASS_TYPE),
    ):
        definition = registry.register_functor_instance(
            functor, node_type_id=node_type
        )
        required = set(functor.get_required_inputs())
        for name in definition.inputs:
            definition.inputs[name] = _dimensionless_state_port(name)
            definition.inputs[name].required = name in required
        for name in definition.outputs:
            definition.outputs[name] = ports[name]
    return registry


def _graph(reader_inputs: dict, *, companion: str | None) -> GraphIR:
    nodes = {
        "reader": {
            "class_type": READER_CLASS_TYPE,
            "inputs": reader_inputs,
            "outputs": ["z"],
            "config": {"execution_group": "fast"},
        },
    }
    if companion == "writer":
        nodes["writer"] = {
            "class_type": WRITER_CLASS_TYPE,
            "inputs": {"x": {"state": "x"}},
            "outputs": ["y"],
            "config": {"execution_group": "fast"},
        }
    if companion == "holder":
        nodes["holder"] = {
            "class_type": HOLDER_CLASS_TYPE,
            "inputs": {"y": {"state": "y"}},
            "outputs": ["w"],
            "config": {"execution_group": "fast"},
        }
    return GraphIR.from_dict(
        {"schema_version": "1.0.0", "nodes": nodes, "metadata": {}}
    )


def _run(graph: GraphIR, initial: dict[str, float], n_steps: int):
    registry = _registry()
    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(
        plan, registry=registry, simulator_config={"device": "cpu"}
    )
    _OptionalReader.seen = []
    state = {name: torch.tensor(value) for name, value in initial.items()}
    with torch.no_grad():
        return executor.execute(state, n_steps=n_steps, dt=1.0)


def test_optional_input_held_as_another_nodes_state_is_not_read():
    graph = _graph({"x": {"state": "x"}}, companion="holder")

    final = _run(graph, {"x": 3.0, "y": 10.0}, n_steps=1)

    assert float(final["w"]) == 10.0
    assert float(final["z"]) == 3.0


def test_optional_input_written_by_an_unlinked_node_is_not_read():
    # The reader runs first (no link orders the two nodes), so the writer's
    # output of step 1 is in the shared state when the reader runs in step 2.
    graph = _graph({"x": {"state": "x"}}, companion="writer")

    final = _run(graph, {"x": 3.0}, n_steps=2)

    assert float(final["y"]) == 6.0
    assert float(final["z"]) == 3.0


@pytest.mark.parametrize(
    ("binding", "companion", "initial", "expected"),
    [
        ({"link": ["writer", "y"]}, "writer", {"x": 3.0}, 9.0),
        ({"state": "y"}, "holder", {"x": 3.0, "y": 10.0}, 13.0),
    ],
)
def test_bound_optional_input_is_read(binding, companion, initial, expected):
    graph = _graph({"x": {"state": "x"}, "y": binding}, companion=companion)

    final = _run(graph, initial, n_steps=1)

    assert float(final["z"]) == expected


def test_kernel_receives_only_the_state_keys_its_step_binds():
    unbound = _graph({"x": {"state": "x"}}, companion="holder")
    _run(unbound, {"x": 3.0, "y": 10.0}, n_steps=2)
    assert _OptionalReader.seen == [("x",), ("x",)]

    bound = _graph(
        {"x": {"state": "x"}, "y": {"state": "y"}}, companion="holder"
    )
    _run(bound, {"x": 3.0, "y": 10.0}, n_steps=1)
    assert _OptionalReader.seen == [("x", "y")]


@pytest.mark.parametrize("strict", [True, False])
def test_compile_report_states_the_input_projection(strict):
    held_as_state = compile_graph(
        _graph({"x": {"state": "x"}}, companion="holder"),
        _registry(),
        strict=strict,
    )
    assert held_as_state.report.metadata["input_projection"] == {
        "schema_version": "1.0.0",
        "policy": "declared_inputs_only",
        "nodes": {
            "holder": {
                "visible_state_keys": ["y"],
                "unbound_optional_inputs": {},
            },
            "reader": {
                "visible_state_keys": ["x"],
                "unbound_optional_inputs": {
                    "q": {"external_state": False, "writers": []},
                    "y": {"external_state": True, "writers": []},
                },
            },
        },
    }

    written = compile_graph(
        _graph({"x": {"state": "x"}}, companion="writer"),
        _registry(),
        strict=strict,
    )
    reader = written.report.metadata["input_projection"]["nodes"]["reader"]
    assert reader["unbound_optional_inputs"]["y"] == {
        "external_state": False,
        "writers": ["writer"],
    }

    linked = compile_graph(
        _graph(
            {"x": {"state": "x"}, "y": {"link": ["writer", "y"]}},
            companion="writer",
        ),
        _registry(),
        strict=strict,
    )
    reader = linked.report.metadata["input_projection"]["nodes"]["reader"]
    assert reader == {
        "visible_state_keys": ["x", "y"],
        "unbound_optional_inputs": {
            "q": {"external_state": False, "writers": []},
        },
    }


def test_non_strict_compile_adds_no_issue_for_an_unbound_optional_input():
    plan = compile_graph(
        _graph({"x": {"state": "x"}}, companion="holder"),
        _registry(),
        strict=False,
    )

    assert plan.report.validation.errors == []
    assert plan.report.validation.warnings == []


def test_serialized_compile_report_carries_the_input_projection():
    plan = compile_graph(
        _graph({"x": {"state": "x"}}, companion="writer"),
        _registry(),
        strict=True,
    )

    serialized = plan_to_dict(plan)["report"]["metadata"]["input_projection"]

    assert serialized == plan.report.metadata["input_projection"]
    assert sorted(serialized["nodes"]) == ["reader", "writer"]


# One synthetic cell of external inputs in the SSOT units of the ports:
# net_radiation [W m-2], air_temperature [K], wind_speed [m s-1],
# atmospheric_pressure [Pa], vapor_pressure_deficit [Pa],
# relative_humidity [1]. The two humidity inputs are deliberately not
# equivalent (RH 0.3 at 298.15 K is a deficit near 2.2 kPa, not 0.5 kPa),
# so the humidity route of the result is identifiable.
_PENMAN_MONTEITH_INPUTS = {
    "net_radiation": 400.0,
    "air_temperature": 298.15,
    "wind_speed": 2.0,
    "atmospheric_pressure": 101325.0,
    "vapor_pressure_deficit": 500.0,
    "relative_humidity": 0.3,
}
# FAO-56 reference surface: the canopy resistance is a parameter, not a state.
_PENMAN_MONTEITH_CONFIG = {"canopy_resistance_source": "fao56_reference_surface"}


def _penman_monteith_flux(forcing: list[str], bound: list[str]) -> torch.Tensor:
    """Governed evapotranspiration [kg m-2 s-1] after one step of 3600 s."""
    import importlib

    module, class_name = PENMAN_MONTEITH.rsplit(".", 1)
    registry = NodeRegistry(
        ssot_catalog=SSOTCatalog.from_parameter_manager(get_global_manager())
    )
    registry.register_functor_instance(
        getattr(importlib.import_module(module), class_name)(
            config=dict(_PENMAN_MONTEITH_CONFIG)
        )
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {
                    "class_type": "State:ExternalForcing",
                    "outputs": list(forcing),
                },
                "penman_monteith_et": {
                    "class_type": "Functor:" + PENMAN_MONTEITH,
                    "inputs": {
                        name: {"link": ["forcing", name]} for name in bound
                    },
                    "outputs": ["evapotranspiration_flux", "latent_heat_flux"],
                    "config": {
                        **_PENMAN_MONTEITH_CONFIG,
                        "execution_group": "fast",
                    },
                },
            },
            "metadata": {},
        }
    )
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    executor = GraphExecutor(
        plan,
        registry=registry,
        simulator_config={"device": "cpu", "auto_report": False},
    )
    initial = {
        name: torch.full(
            (1, 1), _PENMAN_MONTEITH_INPUTS[name], dtype=torch.float32
        )
        for name in forcing
    }
    with torch.no_grad():
        final = executor.execute(initial, n_steps=1, dt=3600.0)
    return final["evapotranspiration_flux"]


def test_penman_monteith_uses_the_humidity_input_its_step_binds():
    # References: Allen et al. (1998), FAO-56 Eq. 3 (Penman-Monteith);
    #   PenmanMonteithET humidity contract (relative humidity is used when
    #   the kernel receives it, the vapour pressure deficit otherwise).
    # Mathematical expression: E = E_PM(R_n, T, u, p, D) with
    #   D = e_s(T) (1 - RH) on the relative-humidity route and D = VPD on
    #   the deficit route; the governed result must equal the route of the
    #   bound input: E(graph holds RH, binds VPD) = E(graph holds only VPD).
    # Parameter meanings: E evapotranspiration flux; R_n net radiation;
    #   T air temperature; u wind speed; p air pressure; D vapour pressure
    #   deficit; RH relative humidity; e_s saturation vapour pressure.
    # Units: E [kg m-2 s-1]; R_n [W m-2]; T [K]; u [m s-1]; p, D, e_s [Pa];
    #   RH [1].
    # Domain/constraints: one cell, one step of 3600 s, CPU, default
    #   parameters; governed runs are compared with governed runs, so the
    #   equalities are exact.
    # Sign convention: E positive from the surface to the atmosphere.
    common = [
        "net_radiation",
        "air_temperature",
        "wind_speed",
        "atmospheric_pressure",
    ]
    deficit_route = _penman_monteith_flux(
        common + ["vapor_pressure_deficit"], common + ["vapor_pressure_deficit"]
    )
    humidity_route = _penman_monteith_flux(
        common + ["relative_humidity"], common + ["relative_humidity"]
    )
    held_not_bound = _penman_monteith_flux(
        common + ["vapor_pressure_deficit", "relative_humidity"],
        common + ["vapor_pressure_deficit"],
    )
    both_bound = _penman_monteith_flux(
        common + ["vapor_pressure_deficit", "relative_humidity"],
        common + ["vapor_pressure_deficit", "relative_humidity"],
    )

    assert torch.isfinite(deficit_route).all()
    assert bool((humidity_route > 1.5 * deficit_route).all())
    assert torch.equal(held_not_bound, deficit_route)
    assert torch.equal(both_bound, humidity_route)
