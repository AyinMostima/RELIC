"""Unit tests for orchestration GraphExecutor."""

import json
from types import SimpleNamespace

import pytest
import torch
from relic.core.state import ExternalIntent, StateWriteMode, UpdateItem

from relic.core.functor import (
    ExecutionError,
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
)
from relic.orchestration.adapters import (
    AdapterRegistry,
    ExecutableAdapterFunctor,
)
from relic.orchestration.cache import NodeCache
from relic.orchestration.compiler import (
    ExecutionStep,
    compile_and_execute,
    compile_graph,
)
from relic.orchestration.executor import ExecutionStrategy, GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway
from test.fixtures.dummy_functor import DummyIncrementFunctor

CLASS_TYPE = "Functor:test.fixtures.dummy_functor.DummyIncrementFunctor"
BIOMASS_CLASS_TYPE = (
    "Functor:relic.scientific.biosphere.growth.biomass_allocation."
    "BiomassAllocation"
)
MEGAN_CLASS_TYPE = (
    "Functor:relic.scientific.biosphere.bvoc.megan_model.MEGANModel"
)
FARQUHAR_CLASS_TYPE = (
    "Functor:relic.scientific.biosphere.photosynthesis.farquhar."
    "FarquharPhotosynthesis"
)
RADIATION_CLASS_TYPE = (
    "Functor:relic.scientific.atmosphere.radiation.RadiationField"
)
GRADIENT_SOURCE_CLASS_TYPE = "Functor:test.gradient_source"
GRADIENT_SINK_CLASS_TYPE = "Functor:test.gradient_sink"
RNG_CACHE_CLAIM_CLASS_TYPE = "Functor:test.rng_cache_claim"
HIDDEN_STATE_CACHE_CLAIM_CLASS_TYPE = "Functor:test.hidden_state_cache_claim"
LOCAL_RNG_CACHE_CLAIM_CLASS_TYPE = "Functor:test.local_rng_cache_claim"
CALLABLE_STATE_CACHE_CLAIM_CLASS_TYPE = "Functor:test.callable_state_cache_claim"


def _dimensionless_state_port(
    name: str,
    *,
    standard_name: str | None = None,
) -> PortSpec:
    return PortSpec(
        name=name,
        standard_name=standard_name or name,
        units="1",
        semantics="state",
        spatial_kind="grid_cell",
        area_basis="ground_area",
        temporal_basis="instantaneous",
        vertical_basis="none",
    )


class _GradientSourceFunctor(PhysicalFieldFunctor):
    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "gradient_source",
        config: dict | None = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            description="Differentiable source for raw-link execution tests.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="y = 2 * x",
            references=["RELIC internal chain-rule verification"],
            parameters={},
            input_variables=["x"],
            output_variables=["y"],
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC internal chain-rule verification.
        # Mathematical expression: y = 2 * x.
        # Parameter meanings: x is the differentiable test input.
        return {"y": 2.0 * state["x"]}


class _GradientSinkFunctor(PhysicalFieldFunctor):
    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "gradient_sink",
        config: dict | None = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            description="Differentiable sink for raw-link execution tests.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="z = y ** 2",
            references=["RELIC internal chain-rule verification"],
            parameters={},
            input_variables=["y"],
            output_variables=["z"],
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC internal chain-rule verification.
        # Mathematical expression: z = y ** 2.
        # Parameter meanings: y is the same-step raw source output.
        return {"z": state["y"] * state["y"]}


def _replacement_source_kernel(self, state, time, dt, **kwargs):
    """A traceable alternate kernel for same-class implementation drift."""
    return {"y": 3.0 * state["x"]}


def _intent_source_kernel(self, state, time, dt, **kwargs):
    return {
        "y": 2.0 * state["x"],
        "export": ExternalIntent("diagnostic", "result.nc", "sample", {}),
    }


def _wrapped_gradient_source_kernel(self, state, time, dt, **kwargs):
    leaf = state["x"].clone().requires_grad_()
    return {
        "y": UpdateItem(leaf.square(), write_mode=StateWriteMode.MUTABLE),
        "diagnostics.source": UpdateItem(leaf, write_mode=StateWriteMode.DERIVED),
    }


def _wrapped_value_source_kernel(self, state, time, dt, **kwargs):
    from relic.core.state import UpdateMode

    return {"y": UpdateItem(
        2.0 * state["x"], mode=UpdateMode.REPLACE,
        write_mode=StateWriteMode.MUTABLE, unit="1", per_timestep=True,
        source_functor=self.name,
    )}


def _nested_intent_source_kernel(self, state, time, dt, **kwargs):
    return {
        "y": 2.0 * state["x"],
        "diagnostics": {"export": ExternalIntent("diagnostic", "result.nc", "sample")},
    }


def _kwargs_source_kernel(self, state, time, dt, **kwargs):
    return {"y": state["x"] * kwargs["boundary"].value}


class _RandomCacheClaimFunctor(PhysicalFieldFunctor):
    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "random_cache_claim",
        config: dict | None = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            description="Invalid deterministic-pure RNG test claim.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="y = U(0, 1)",
            references=["RELIC cache contract verification"],
            parameters={},
            input_variables=[],
            output_variables=["random_value"],
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC cache contract verification.
        # Mathematical expression: y is sampled from U(0, 1).
        # Parameter meanings: no model parameter is used.
        return {"random_value": torch.rand(())}


class _HiddenStateCacheClaimFunctor(PhysicalFieldFunctor):
    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="hidden_state_cache_claim",
            description="Invalid deterministic-pure hidden-state claim.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="y_k = k",
            references=["RELIC cache contract verification"],
            parameters={},
            input_variables=[],
            output_variables=["hidden_state_value"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)
        self.hidden_counter = 0

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC cache contract verification.
        # Mathematical expression: y_k = k after one hidden-state increment.
        # Parameter meanings: k is replay-critical hidden state [1].
        self.hidden_counter += 1
        return {"hidden_state_value": torch.tensor(float(self.hidden_counter))}


class _LocalRNGCacheClaimFunctor(PhysicalFieldFunctor):
    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="local_rng_cache_claim",
            description="Invalid deterministic-pure local-RNG claim.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="y = U_g(0, 1)",
            references=["RELIC cache contract verification"],
            parameters={},
            input_variables=[],
            output_variables=["local_rng_value"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)
        self.local_generator = torch.Generator().manual_seed(73)

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC cache contract verification.
        # Mathematical expression: y is sampled from U_g(0, 1).
        # Parameter meanings: g is a replay-critical local RNG stream.
        return {
            "local_rng_value": torch.rand((), generator=self.local_generator)
        }


class _MutableCallableCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> int:
        self.count += 1
        return self.count


class _CallableStateCacheClaimFunctor(PhysicalFieldFunctor):
    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="callable_state_cache_claim",
            description="Invalid deterministic-pure callable-state claim.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="y_k = c_k",
            references=["RELIC cache contract verification"],
            parameters={},
            input_variables=[],
            output_variables=["callable_state_value"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)
        self.hidden_callable = _MutableCallableCounter()

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC cache contract verification.
        # Mathematical expression: y_k = c_k after a callable-state increment.
        # Parameter meanings: c_k is replay-critical hidden state [1].
        return {"callable_state_value": torch.tensor(float(self.hidden_callable()))}


class _AlternateDummyIncrementFunctor(DummyIncrementFunctor):
    """Runtime-class drift fixture with the same scientific port contract."""


def _build_registry() -> NodeRegistry:
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.build_from_node_spec(CLASS_TYPE, [], ["x"])
    definition.outputs["x"] = _dimensionless_state_port("x")
    registry.register_definition(CLASS_TYPE, definition)
    return registry


def _build_graph(*, config: dict | None = None) -> GraphIR:
    nodes = {
        "n1": NodeSpec(
            class_type=CLASS_TYPE,
            inputs={},
            outputs=["x"],
            config=dict(config or {}),
        )
    }
    return GraphIR(
        schema_version="1.0.0",
        nodes=nodes,
        links=[],
        metadata={},
        assertion={
            "phi": "x is finite and nonnegative",
            "op": "state_domain_check",
            "scope": "state",
            "context": {
                "state": {"x": 0.0},
                "state_domain_constraints": {"x": {"nonnegative": True}},
            },
            "audit_key": "state_domain_audit",
        },
    )


def _build_pure_cache_registry_graph(
    *,
    config: dict | None = None,
) -> tuple[NodeRegistry, GraphIR]:
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.register_functor_instance(
        _GradientSourceFunctor(),
        node_type_id=GRADIENT_SOURCE_CLASS_TYPE,
    )
    definition.inputs["x"] = _dimensionless_state_port("x")
    definition.outputs["y"] = _dimensionless_state_port("y")
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "pure_source": NodeSpec(
                class_type=GRADIENT_SOURCE_CLASS_TYPE,
                inputs={"x": {"state": "x"}},
                outputs=["y"],
                config=dict(config or {}),
            )
        },
    )
    return registry, graph


def _build_executor() -> GraphExecutor:
    return GraphExecutor(SimpleNamespace(steps=[]))


def _build_external_state_plan(*state_keys: str):
    registry = NodeRegistry(SSOTCatalog())
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "forcing": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=list(state_keys),
            )
        },
    )
    return compile_graph(graph, registry), registry


def _build_empty_plan():
    return compile_graph(
        GraphIR(schema_version="1.0.0", nodes={}),
        NodeRegistry(SSOTCatalog()),
    )


def _build_step(class_type: str, config: dict, node_id: str = "node") -> ExecutionStep:
    return ExecutionStep(
        node_id=node_id,
        class_type=class_type,
        inputs={},
        outputs=[],
        config=config,
    )


def test_dependency_binding_failure_is_fail_closed(monkeypatch) -> None:
    functor = DummyIncrementFunctor()
    step = SimpleNamespace(
        node_id="sink",
        inputs={"x": {"link": ["source", "x"]}},
    )

    def _reject_dependencies(_dependencies):
        raise ValueError("invalid dependency contract")

    monkeypatch.setattr(functor, "set_dependencies", _reject_dependencies)

    with pytest.raises(ValueError, match="invalid dependency contract"):
        _build_executor()._attach_dependencies(functor, step)


@pytest.mark.parametrize(
    ("class_type", "field_name", "value", "expected"),
    [
        (
            BIOMASS_CLASS_TYPE,
            "bvoc_carbon_allocation_owner",
            "megan_bvoc_model",
            "megan_bvoc_model",
        ),
        (
            MEGAN_CLASS_TYPE,
            "emission_pathway",
            "standard",
            EmissionPathway.STANDARD,
        ),
    ],
)
def test_build_functor_binds_explicit_constructor_controls(
    class_type,
    field_name,
    value,
    expected,
):
    instance = _build_executor()._build_functor(
        _build_step(class_type, {field_name: value})
    )

    assert getattr(instance, field_name) == expected


def test_build_functor_coerces_enum_and_preserves_residual_config():
    instance = _build_executor()._build_functor(
        _build_step(
            MEGAN_CLASS_TYPE,
            {
                "emission_pathway": "standard",
                "enabled_compounds": ["isoprene"],
            },
        )
    )

    assert instance.emission_pathway is EmissionPathway.STANDARD
    assert instance.enabled_compounds == ["isoprene"]
    assert instance._original_config == {"enabled_compounds": ["isoprene"]}


@pytest.mark.parametrize(
    ("class_type", "config", "message"),
    [
        (
            MEGAN_CLASS_TYPE,
            {"emission_pathway": "legacy"},
            "emission_pathway",
        ),
        (
            FARQUHAR_CLASS_TYPE,
            {"vcmax25": 100.0},
            "vcmax25",
        ),
        (
            MEGAN_CLASS_TYPE,
            {"device": "cpu"},
            "protected",
        ),
        (
            MEGAN_CLASS_TYPE,
            {"parameters": {"emission_factor": 1.0}},
            "protected",
        ),
    ],
)
def test_build_functor_rejects_invalid_or_unsafe_constructor_config(
    class_type,
    config,
    message,
):
    with pytest.raises(ValueError, match=message):
        _build_executor()._build_functor(_build_step(class_type, config))


def test_build_functor_does_not_use_var_kwargs_as_acceptance_contract():
    instance = _build_executor()._build_functor(
        _build_step(RADIATION_CLASS_TYPE, {})
    )

    assert instance.__class__.__name__ == "RadiationField"

    with pytest.raises(TypeError, match="explicit config"):
        _build_executor()._build_functor(
            _build_step(RADIATION_CLASS_TYPE, {"unknown_control": True})
        )


def test_build_functor_strips_execution_metadata():
    instance = _build_executor()._build_functor(
        _build_step(
            CLASS_TYPE,
            {
                "execution_group": "fast",
                "iphs_contribution": {"state_vars": ["x"]},
            },
            node_id="n1",
        )
    )

    assert instance._original_config == {}


def test_build_functor_resolves_registered_runtime_alias():
    alias = "Functor:adapter.test.increment"
    registry = NodeRegistry(SSOTCatalog())
    registry.register_functor_instance(
        DummyIncrementFunctor(),
        node_type_id=alias,
    )
    executor = GraphExecutor(SimpleNamespace(steps=[]), registry=registry)

    instance = executor._build_functor(_build_step(alias, {}))

    assert isinstance(instance, DummyIncrementFunctor)


def test_graph_executor_runs_plan(tmp_path) -> None:
    registry = _build_registry()
    graph = _build_graph()
    plan = compile_graph(graph, registry)
    cache_dir = tmp_path / "cache"
    cache = NodeCache(cache_dir)
    executor = GraphExecutor(plan, cache=cache, registry=registry)
    final_state = executor.execute({"x": torch.tensor(0.0)}, n_steps=2, dt=1.0)
    assertion_audit = plan.report.validation.assertion_audit
    provenance = executor._simulator.get_variable_provenance("x")

    assert plan.report.validation.errors == []
    assert assertion_audit["present"] is True
    assert assertion_audit["valid"] is True
    assert assertion_audit["op_resolved"] is True
    assert assertion_audit["structural_result"]["passed"] is True
    assert plan.report.metadata["audit_summary"]["state_domain_audit"]["passed"] is True
    assert float(final_state["x"].item()) == 2.0
    assert "n1" in executor._simulator.get_functor_summary()
    assert (
        executor._simulator.get_effective_device("n1") == executor._functor_instances["n1"].device
    )
    assert any(record["functor"] == "n1" for record in provenance)
    assert list(cache_dir.iterdir()) == []


def test_graph_executor_initial_recorder_consumes_simulator_identity(tmp_path):
    executor = GraphExecutor(_build_empty_plan(), output_dir=tmp_path)
    assert executor._recorder.run_id == executor._simulator.run_id
    assert executor.state_manager._provenance_run_id == executor._simulator.run_id


def test_graph_executor_does_not_hide_recorder_attachment_failure(tmp_path, monkeypatch):
    from relic.core.state import StateManager

    def fail_attachment(self, recorder, **kwargs):
        raise OSError("recorder attachment failed")

    monkeypatch.setattr(StateManager, "set_provenance_recorder", fail_attachment)
    with pytest.raises(OSError, match="attachment failed"):
        GraphExecutor(_build_empty_plan(), output_dir=tmp_path)


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_graph_rebuild_confirms_old_checkpoint_before_replacing_simulator(
    tmp_path, monkeypatch, failure,
):
    from pathlib import Path
    from relic.core.simulator import checkpoint as checkpoint_module

    registry = _build_registry()
    executor = GraphExecutor(
        compile_graph(_build_graph(), registry), registry=registry,
        simulator_config={
            "device": "cpu", "enable_auto_scale": False,
            "enable_checkpointing": True, "checkpoint_frequency": 1.0,
            "checkpoint_directory": str(tmp_path / "checkpoints"),
        },
    )
    replace = checkpoint_module.os.replace

    def fail_checkpoint(source, target):
        if Path(target).suffix == ".pt":
            raise failure("checkpoint destination unavailable")
        return replace(source, target)

    with monkeypatch.context() as local:
        local.setattr(checkpoint_module.os, "replace", fail_checkpoint)
        with pytest.raises(failure, match="destination unavailable"):
            executor.execute({"x": torch.tensor(0.0)}, n_steps=1, dt=1.0)
        previous = executor._simulator
        job = previous._pending_external_deliveries[0]
        with pytest.raises(failure, match="destination unavailable"):
            executor.execute({"x": torch.tensor(9.0)}, n_steps=0, dt=1.0)
        assert executor._simulator is previous
        assert executor._execution_prepared
        assert previous.step_count == 1
        assert float(executor.state_manager.get_state("x")) == 1.0
        assert previous._pending_external_deliveries[0] is job

    result = executor.execute({"x": torch.tensor(9.0)}, n_steps=0, dt=1.0)
    saved = torch.load(job["resource"], map_location="cpu", weights_only=True)
    assert saved["runtime"]["run_id"] == previous.run_id
    assert saved["runtime"]["step_count"] == 1
    assert float(saved["state_manager"]["state"]["x"]) == 1.0
    assert previous._pending_external_deliveries == []
    assert previous._last_external_delivery_receipts[0]["key"] == job["key"]
    assert executor._simulator.run_id != previous.run_id
    assert float(result["x"]) == 9.0


def test_graph_rebuild_keeps_old_state_when_reset_fails(monkeypatch):
    executor = GraphExecutor(_build_empty_plan())
    executor.execute({"x": torch.tensor(2.0)}, n_steps=0, dt=1.0)
    previous = executor._simulator

    def fail_reset():
        raise RuntimeError("state reset failed")

    monkeypatch.setattr(executor.state_manager, "reset", fail_reset)
    with pytest.raises(RuntimeError, match="state reset failed"):
        executor.execute({"x": torch.tensor(9.0)}, n_steps=0, dt=1.0)
    assert executor._simulator is previous
    assert executor._execution_prepared
    assert float(executor.state_manager.get_state("x")) == 2.0


def test_public_preflight_rejects_unsealed_plan_before_functor_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = GraphExecutor(
        SimpleNamespace(steps=[_build_step(CLASS_TYPE, {}, node_id="n1")]),
        registry=_build_registry(),
    )

    def fail_if_constructed(step):
        raise AssertionError(f"Functor construction must not run: {step.node_id}")

    monkeypatch.setattr(executor, "_build_functor", fail_if_constructed)

    with pytest.raises(TypeError, match="canonical ExecutionPlan"):
        executor.preflight()


def test_zero_step_execute_rejects_unsealed_plan_before_state_reset() -> None:
    class FailIfReset:
        def reset(self):
            raise AssertionError("StateManager reset must not run")

    executor = GraphExecutor(
        SimpleNamespace(steps=[]),
        state_manager=FailIfReset(),
    )

    with pytest.raises(TypeError, match="canonical ExecutionPlan"):
        executor.execute({}, n_steps=0, dt=1.0)


def test_execute_rejects_non_strict_compile_witness_before_state_reset() -> None:
    class FailIfReset:
        def reset(self):
            raise AssertionError("StateManager reset must not run")

    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry, strict=False)
    executor = GraphExecutor(
        plan,
        registry=registry,
        state_manager=FailIfReset(),
    )

    with pytest.raises(ValueError, match="strict compile-run witness"):
        executor.execute({"x": torch.tensor(0.0)}, n_steps=0, dt=1.0)


def test_execute_rejects_plan_drift_before_state_reset() -> None:
    class FailIfReset:
        def reset(self):
            raise AssertionError("StateManager reset must not run")

    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    plan.steps[0].config["contract_drift"] = True
    executor = GraphExecutor(
        plan,
        registry=registry,
        state_manager=FailIfReset(),
    )

    with pytest.raises(ValueError, match="does not match its compiler seal"):
        executor.execute({"x": torch.tensor(0.0)}, n_steps=0, dt=1.0)


def test_execute_rejects_runtime_class_drift_with_valid_registry_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    executor = GraphExecutor(plan, registry=registry)
    monkeypatch.setattr(
        registry,
        "resolve_class_type",
        lambda class_type: _AlternateDummyIncrementFunctor,
    )

    with pytest.raises(ValueError, match="runtime witness"):
        executor.execute({"x": torch.tensor(0.0)}, n_steps=0, dt=1.0)

    assert executor.execution_summary["step_count"] == 0
    assert executor.execution_summary["execution_count"] == 0


def test_execute_rechecks_seal_after_runtime_construction_before_state_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailIfReset:
        def reset(self):
            raise AssertionError("StateManager reset must not run")

    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    executor = GraphExecutor(
        plan,
        registry=registry,
        state_manager=FailIfReset(),
    )
    real_build = executor._build_functor

    def mutating_build(step):
        functor = real_build(step)
        plan.report.metadata["compile_run_witness"]["passed"] = False
        return functor

    monkeypatch.setattr(executor, "_build_functor", mutating_build)

    with pytest.raises(ValueError, match="does not match its compiler seal"):
        executor.execute({"x": torch.tensor(0.0)}, n_steps=0, dt=1.0)


def test_graph_executor_caches_only_dual_declared_deterministic_pure_node(
    tmp_path,
) -> None:
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    cache_dir = tmp_path / "cache"
    executor = GraphExecutor(
        plan,
        cache=NodeCache(cache_dir),
        registry=registry,
    )

    executor.execute({"x": torch.tensor(1.0)}, n_steps=1, dt=1.0)

    assert len(list(cache_dir.glob("*.pkl"))) == 1


@pytest.mark.parametrize("policy", [None, "deterministic_pure"])
def test_prepared_graph_leaves_raw_forward_outside_execution_policy(tmp_path, policy):
    registry, graph = _build_pure_cache_registry_graph(
        config={} if policy is None else {"cache_policy": policy}
    )
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(
        plan, registry=registry, cache=NodeCache(tmp_path / "cache"),
        output_dir=tmp_path / "provenance", simulator_config={"device": "cpu"},
    )
    executor.execute({"x": torch.tensor(3.0)}, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]

    assert "forward" not in vars(functor)
    assert functor.forward.__func__ is _GradientSourceFunctor.forward
    for _ in range(2):
        result = functor.forward({"x": torch.tensor(3.0)}, time=0.0, dt=1.0)
        assert float(result["y"]) == 6.0
    assert list((tmp_path / "cache").iterdir()) == []
    assert not (tmp_path / "provenance" / "lineage.jsonl").exists()
    assert functor.get_execution_stats()["execution_count"] == 0


def test_raw_rng_kernel_remains_separate_from_governed_rng_audit(tmp_path):
    functor = _RandomCacheClaimFunctor()
    step = _build_step(RNG_CACHE_CLAIM_CLASS_TYPE, {"cache_policy": "deterministic_pure"})
    executor = GraphExecutor(
        SimpleNamespace(steps=[step]), cache=NodeCache(tmp_path / "cache"),
        simulator_config={"device": "cpu"},
    )
    executor._wrap_functor(functor, step)

    with torch.random.fork_rng(devices=[]):
        raw_result = functor.forward({}, time=0.0, dt=1.0)
        assert bool(torch.isfinite(raw_result["random_value"]))
        assert list((tmp_path / "cache").iterdir()) == []
        with pytest.raises(ExecutionError, match="consumed runtime RNG"):
            functor({}, time=0.0, dt=1.0)
    assert list((tmp_path / "cache").iterdir()) == []


def test_cache_reuse_and_invalidation_keep_governance_and_lineage(tmp_path, monkeypatch):
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    calls = {"forward": 0}
    original_module_call = torch.nn.Module._call_impl

    def observe_module_call(self, *args, **kwargs):
        if isinstance(self, _GradientSourceFunctor):
            calls["forward"] += 1
        return original_module_call(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "_call_impl", observe_module_call)

    def observe_method(name):
        original = getattr(_GradientSourceFunctor, name)
        calls[name] = 0

        def observed(self, *args, **kwargs):
            calls[name] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(_GradientSourceFunctor, name, observed)

    for name in (
        "validate_state", "postprocess_updates",
        "_check_local_conservation", "_run_validation_pipeline",
    ):
        observe_method(name)
    cache = NodeCache(tmp_path / "cache")
    executor = GraphExecutor(
        plan, registry=registry, cache=cache, output_dir=tmp_path / "provenance",
        simulator_config={"device": "cpu"},
    )
    run_ids = []
    for _ in range(2):
        result = executor.execute({"x": torch.tensor(3.0)}, n_steps=1, dt=1.0)
        run_ids.append(executor._simulator.run_id)
        assert executor._recorder.run_id == run_ids[-1]
        assert float(result["y"]) == 6.0
        assert executor._functor_instances["pure_source"].get_execution_stats()["execution_count"] == 1
    assert calls["forward"] == 1
    assert all(count == 2 for name, count in calls.items() if name != "forward")
    events = [
        json.loads(line)
        for line in (tmp_path / "provenance" / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in events] == ["START", "COMPLETE", "CACHE_HIT"]
    assert run_ids[0] != run_ids[1]
    assert [event["run_id"] for event in events] == [run_ids[0], run_ids[0], run_ids[1]]
    assert {event["node_id"] for event in events} == {"pure_source"}
    assert len({event["cache_key"] for event in events}) == 1
    assert cache.invalidate("pure_source") == 1
    result = executor.execute({"x": torch.tensor(3.0)}, n_steps=1, dt=1.0)
    assert float(result["y"]) == 6.0
    assert calls["forward"] == 2


@pytest.mark.parametrize(
    "hook_name",
    [
        "register_forward_pre_hook", "register_forward_hook",
        "register_full_backward_pre_hook", "register_full_backward_hook",
        "register_module_forward_pre_hook", "register_module_forward_hook",
        "register_module_full_backward_pre_hook", "register_module_full_backward_hook",
    ],
)
def test_cache_eligibility_excludes_active_torch_hooks(hook_name):
    from torch.nn.modules import module as torch_module

    functor = _GradientSourceFunctor()
    step = _build_step(GRADIENT_SOURCE_CLASS_TYPE, {"cache_policy": "deterministic_pure"})
    executor = _build_executor()
    owner = torch_module if hook_name.startswith("register_module_") else functor
    handle = getattr(owner, hook_name)(lambda *args: None)
    try:
        assert not executor._is_cache_eligible(functor, step, {}, 1.0, {})
    finally:
        handle.remove()
    assert executor._is_cache_eligible(functor, step, {}, 1.0, {})


@pytest.mark.parametrize("strategy", [ExecutionStrategy.PUSH, ExecutionStrategy.DIRTY])
@pytest.mark.parametrize("global_hook", [False, True])
def test_cached_and_dirty_execution_run_kernel_while_hooks_are_active(
    tmp_path, monkeypatch, strategy, global_hook,
):
    from torch.nn.modules import module as torch_module

    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    kernel_calls = []
    original_module_call = torch.nn.Module._call_impl

    def observed_module_call(self, *args, **kwargs):
        if isinstance(self, _GradientSourceFunctor):
            kernel_calls.append(self.name)
        return original_module_call(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "_call_impl", observed_module_call)
    executor = GraphExecutor(
        plan, registry=registry, cache=NodeCache(tmp_path / "cache"),
        strategy=strategy, simulator_config={"device": "cpu"},
    )
    state = {"x": torch.tensor(3.0)}
    executor.execute(state, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]
    assert float(functor(state, time=0.0, dt=1.0)["y"]) == 6.0
    assert float(functor(state, time=0.0, dt=1.0)["y"]) == 6.0
    assert len(kernel_calls) == 1
    hook_calls = []

    def after(module, args, output):
        if module is functor:
            hook_calls.append(module.name)
            return {"y": output["y"] + len(hook_calls)}

    register = (
        torch_module.register_module_forward_hook if global_hook
        else functor.register_forward_hook
    )
    handle = register(after)
    try:
        assert float(functor(state, time=0.0, dt=1.0)["y"]) == 7.0
        assert float(functor(state, time=0.0, dt=1.0)["y"]) == 8.0
        assert len(kernel_calls) == 3
    finally:
        handle.remove()
    assert len(hook_calls) == 2
    assert float(functor(state, time=0.0, dt=1.0)["y"]) == 6.0
    assert len(kernel_calls) == 3


@pytest.mark.parametrize("strategy", [ExecutionStrategy.PUSH, ExecutionStrategy.DIRTY])
@pytest.mark.parametrize("kernel", [_intent_source_kernel, _wrapped_gradient_source_kernel])
def test_effect_and_wrapped_gradient_outputs_are_never_reused(
    tmp_path, monkeypatch, strategy, kernel,
):
    from unittest.mock import Mock

    monkeypatch.setattr(_GradientSourceFunctor, "forward", kernel)
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    cache = NodeCache(tmp_path / "cache")
    put = Mock(wraps=cache.put)
    monkeypatch.setattr(cache, "put", put)
    executor = GraphExecutor(
        plan, registry=registry, cache=cache, strategy=strategy,
        simulator_config={"device": "cpu"},
    )
    state = {"x": torch.tensor(3.0)}
    executor.execute(state, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]
    assert executor._is_cache_eligible(functor, plan.steps[0], state, 1.0, {})
    assert executor._compute_cache_key(functor, plan.steps[0], state, 0.0, 1.0, {})
    first = functor(state, time=0.0, dt=1.0)
    second = functor(state, time=0.0, dt=1.0)
    put.assert_not_called()
    assert "pure_source" not in executor._last_outputs
    assert "pure_source" not in executor._last_input_hashes
    if strategy is ExecutionStrategy.DIRTY:
        assert "pure_source" in executor._dirty_nodes
    if kernel is _intent_source_kernel:
        assert first["export"] == second["export"]
        assert first["export"] is not second["export"]
        assert float(second["y"]) == 6.0
    else:
        assert first["y"].value is not second["y"].value
        for result in (first, second):
            assert result["y"].value.grad_fn is not None
            gradient = torch.autograd.grad(
                result["y"].value, result["diagnostics.source"].value,
            )[0]
            assert float(result["y"].value.detach()) == 9.0
            assert float(gradient) == 6.0


@pytest.mark.parametrize("source", ["disk", "dirty"])
@pytest.mark.parametrize("kernel", [
    _intent_source_kernel, _wrapped_gradient_source_kernel, _nested_intent_source_kernel,
])
def test_ineligible_existing_output_is_invalidated_before_recomputation(
    tmp_path, monkeypatch, source, kernel,
):
    from unittest.mock import Mock

    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    cache = NodeCache(tmp_path / "cache")
    executor = GraphExecutor(
        plan, registry=registry, cache=cache, strategy=ExecutionStrategy.DIRTY,
        simulator_config={"device": "cpu"},
    )
    state = {"x": torch.tensor(3.0)}
    executor.execute(state, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]
    step = plan.steps[0]
    stale = kernel(functor, state, time=0.0, dt=1.0)
    key = executor._compute_cache_key(functor, step, state, 0.0, 1.0, {})
    assert key is not None
    if source == "disk":
        cache.put(key, {"y": torch.tensor(-99.0)})
        monkeypatch.setattr(cache, "get", lambda requested: stale)
    else:
        executor._last_outputs["pure_source"] = stale
        executor._update_dirty_tracking("pure_source", step, state, 0.0, 1.0)
    invalidate = Mock(wraps=cache.invalidate)
    monkeypatch.setattr(cache, "invalidate", invalidate)
    result = functor(state, time=0.0, dt=1.0)
    assert set(result) == {"y"}
    assert float(result["y"]) == 6.0
    assert set(executor._last_outputs["pure_source"]) == {"y"}
    if source == "disk":
        invalidate.assert_called_once_with("pure_source")
        assert float(NodeCache.get(cache, key)["y"]) == 6.0


def _prepared_pure_cache_executor(tmp_path) -> tuple[GraphExecutor, NodeCache, _GradientSourceFunctor]:
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    cache = NodeCache(tmp_path / "cache")
    executor = GraphExecutor(
        plan, registry=registry, cache=cache, strategy=ExecutionStrategy.PUSH,
        simulator_config={"device": "cpu"},
    )
    executor.execute({"x": torch.tensor(3.0)}, n_steps=0, dt=1.0)
    return executor, cache, executor._functor_instances["pure_source"]


def test_cache_write_failure_of_an_eligible_node_propagates(tmp_path, monkeypatch):
    """An opted-in node whose output cannot be persisted must not run silently uncached."""
    executor, cache, functor = _prepared_pure_cache_executor(tmp_path)

    def _cache_device_full(key, value):
        raise OSError("no space left on the node cache device")

    monkeypatch.setattr(cache, "put", _cache_device_full)

    with pytest.raises(ExecutionError, match="no space left") as excinfo:
        functor({"x": torch.tensor(3.0)}, time=0.0, dt=1.0)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_cache_key_failure_of_an_eligible_node_propagates(tmp_path, monkeypatch):
    """An opted-in node whose cache contract cannot be hashed must not disable caching silently."""
    executor, cache, functor = _prepared_pure_cache_executor(tmp_path)

    def _unhashable(payload):
        raise TypeError("cannot pickle '_thread.lock' object")

    monkeypatch.setattr(cache, "hash_object", _unhashable)

    with pytest.raises(ExecutionError, match="cannot pickle") as excinfo:
        functor({"x": torch.tensor(3.0)}, time=0.0, dt=1.0)
    assert isinstance(excinfo.value.__cause__, TypeError)


@pytest.mark.parametrize("strategy", [ExecutionStrategy.PUSH, ExecutionStrategy.DIRTY])
def test_cache_eligibility_checks_gradients_inside_updateitem_kwargs(
    tmp_path, monkeypatch, strategy,
):
    from unittest.mock import Mock

    monkeypatch.setattr(_GradientSourceFunctor, "forward", _kwargs_source_kernel)
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    cache = NodeCache(tmp_path / "cache")
    get = Mock(wraps=cache.get)
    put = Mock(wraps=cache.put)
    monkeypatch.setattr(cache, "get", get)
    monkeypatch.setattr(cache, "put", put)
    executor = GraphExecutor(
        compile_graph(graph, registry), registry=registry, cache=cache,
        strategy=strategy, simulator_config={"device": "cpu"},
    )
    state = {"x": torch.tensor(3.0)}
    executor.execute(state, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]
    leaf = torch.tensor(2.0, requires_grad=True)
    for _ in range(2):
        result = functor(state, time=0.0, dt=1.0, boundary=UpdateItem(leaf))
        gradient = torch.autograd.grad(result["y"], leaf)[0]
        assert float(result["y"].detach()) == 6.0
        assert float(gradient) == 3.0
    get.assert_not_called()
    put.assert_not_called()
    assert "pure_source" not in executor._last_outputs


@pytest.mark.parametrize("strategy", [ExecutionStrategy.PUSH, ExecutionStrategy.DIRTY])
def test_malformed_nested_intent_fails_without_cache_side_effects(
    tmp_path, monkeypatch, strategy,
):
    from unittest.mock import Mock

    monkeypatch.setattr(_GradientSourceFunctor, "forward", _nested_intent_source_kernel)
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    cache = NodeCache(tmp_path / "cache")
    put = Mock(wraps=cache.put)
    monkeypatch.setattr(cache, "put", put)
    executor = GraphExecutor(
        compile_graph(graph, registry), registry=registry, cache=cache,
        strategy=strategy, simulator_config={"device": "cpu"},
    )
    state = {"x": torch.tensor(3.0)}
    executor.execute(state, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]
    with pytest.raises(ExecutionError, match="top-level"):
        functor(state, time=0.0, dt=1.0)
    put.assert_not_called()
    assert "pure_source" not in executor._last_outputs


@pytest.mark.parametrize("strategy", [ExecutionStrategy.PUSH, ExecutionStrategy.DIRTY])
def test_non_gradient_updateitem_remains_cacheable(tmp_path, monkeypatch, strategy):
    from unittest.mock import Mock

    monkeypatch.setattr(_GradientSourceFunctor, "forward", _wrapped_value_source_kernel)
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    cache = NodeCache(tmp_path / "cache")
    put = Mock(wraps=cache.put)
    monkeypatch.setattr(cache, "put", put)
    executor = GraphExecutor(
        compile_graph(graph, registry), registry=registry, cache=cache,
        strategy=strategy, simulator_config={"device": "cpu"},
    )
    state = {"x": torch.tensor(3.0)}
    executor.execute(state, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]
    first = functor(state, time=0.0, dt=1.0)
    second = functor(state, time=0.0, dt=1.0)
    put.assert_called_once()
    assert float(second["y"].value) == 6.0
    assert vars(first["y"]) == vars(second["y"])


def test_instance_forward_override_is_not_pure_runtime_shell_state(monkeypatch):
    functor = _GradientSourceFunctor()
    step = _build_step(GRADIENT_SOURCE_CLASS_TYPE, {"cache_policy": "deterministic_pure"})
    monkeypatch.setattr(functor, "forward", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="replay-critical instance state: forward"):
        _build_executor()._is_cache_eligible(functor, step, {}, 1.0, {})


def test_disk_cache_invalidates_same_class_in_place_kernel_replacement(tmp_path, monkeypatch):
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(
        plan, registry=registry, cache=NodeCache(tmp_path / "cache"),
        simulator_config={"device": "cpu"},
    )
    first = executor.execute({"x": torch.tensor(3.0)}, n_steps=1, dt=1.0)
    assert float(first["y"]) == 6.0
    kernel = _GradientSourceFunctor.forward
    monkeypatch.setattr(kernel, "__code__", _replacement_source_kernel.__code__)
    second = executor.execute({"x": torch.tensor(3.0)}, n_steps=1, dt=1.0)
    assert _GradientSourceFunctor.forward is kernel
    assert float(second["y"]) == 9.0
    # The stale entry of the replaced provider is evicted by the closure
    # invalidation; only the new identity's entry remains.
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 1


def test_dirty_reuse_invalidates_same_class_in_place_kernel_replacement(tmp_path, monkeypatch):
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(
        plan, registry=registry, strategy=ExecutionStrategy.DIRTY,
        simulator_config={"device": "cpu"},
    )
    state = {"x": torch.tensor(3.0)}
    executor.execute(state, n_steps=0, dt=1.0)
    functor = executor._functor_instances["pure_source"]
    calls = []
    original_module_call = torch.nn.Module._call_impl

    def observe_module_call(self, *args, **kwargs):
        if self is functor:
            calls.append(self.name)
        return original_module_call(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "_call_impl", observe_module_call)
    for _ in range(2):
        assert float(functor(state, time=0.0, dt=1.0)["y"]) == 6.0
    assert len(calls) == 1
    monkeypatch.setattr(_GradientSourceFunctor.forward, "__code__", _replacement_source_kernel.__code__)
    for _ in range(2):
        assert float(functor(state, time=0.0, dt=1.0)["y"]) == 9.0
    assert len(calls) == 2
    assert executor.cache is None


@pytest.mark.parametrize("strategy", [ExecutionStrategy.PUSH, ExecutionStrategy.DIRTY])
def test_cache_rejects_untraceable_kernel_before_execution(tmp_path, monkeypatch, strategy):
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(
        plan, registry=registry, strategy=strategy,
        cache=NodeCache(tmp_path / "cache") if strategy == ExecutionStrategy.PUSH else None,
        simulator_config={"device": "cpu"},
    )
    kernel = _GradientSourceFunctor.forward
    monkeypatch.setattr(kernel, "__code__", kernel.__code__.replace(co_filename="<untraceable-cache-kernel>"))
    calls = []
    original_module_call = torch.nn.Module._call_impl

    def observe_module_call(self, *args, **kwargs):
        if isinstance(self, _GradientSourceFunctor):
            calls.append(self.name)
        return original_module_call(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "_call_impl", observe_module_call)
    with pytest.raises(ExecutionError, match="inspectable Python source"):
        executor.execute({"x": torch.tensor(3.0)}, n_steps=1, dt=1.0)
    assert calls == []
    assert executor.execution_summary["step_count"] == 0
    assert float(executor.state_manager.get_state()["x"]) == 3.0


def test_cache_key_binds_node_config_and_execution_plan_contract(tmp_path) -> None:
    cache = NodeCache(tmp_path / "cache")
    for variant in ("a", "b"):
        registry, graph = _build_pure_cache_registry_graph(
            config={
                "cache_policy": "deterministic_pure",
                "contract_variant": variant,
            }
        )
        plan = compile_graph(graph, registry)
        GraphExecutor(plan, cache=cache, registry=registry).execute(
            {"x": torch.tensor(0.0)},
            n_steps=1,
            dt=1.0,
        )

    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 2


def test_cache_key_and_eligibility_use_only_declared_node_inputs(tmp_path) -> None:
    registry, graph = _build_pure_cache_registry_graph(
        config={"cache_policy": "deterministic_pure"}
    )
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(
        plan,
        cache=NodeCache(tmp_path / "cache"),
        registry=registry,
    )
    step = plan.steps[0]
    functor = executor._build_functor(step)
    dt = torch.tensor(1.0, device=executor._simulator.device)
    declared = torch.tensor(2.0, device=executor._simulator.device)
    unrelated_a = torch.ones(
        4096,
        device=executor._simulator.device,
        requires_grad=True,
    )
    unrelated_b = torch.full(
        (4096,),
        3.0,
        device=executor._simulator.device,
        requires_grad=True,
    )

    key_a = executor._compute_cache_key(
        functor,
        step,
        {"x": declared, "unrelated": unrelated_a},
        0.0,
        dt,
        {},
    )
    key_b = executor._compute_cache_key(
        functor,
        step,
        {"x": declared, "unrelated": unrelated_b},
        0.0,
        dt,
        {},
    )

    assert key_a == key_b
    assert executor._is_cache_eligible(
        functor,
        step,
        {"x": declared, "unrelated": unrelated_a},
        dt,
        {},
    )


def test_deterministic_pure_claim_rejects_runtime_rng_consumption(tmp_path) -> None:
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.register_functor_instance(
        _RandomCacheClaimFunctor(),
        node_type_id=RNG_CACHE_CLAIM_CLASS_TYPE,
    )
    definition.outputs["random_value"] = _dimensionless_state_port(
        "random_value"
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "random_node": {
                    "class_type": RNG_CACHE_CLAIM_CLASS_TYPE,
                    "inputs": {},
                    "outputs": ["random_value"],
                    "config": {"cache_policy": "deterministic_pure"},
                }
            },
            "metadata": {},
        }
    )
    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(
        plan,
        cache=NodeCache(tmp_path / "cache"),
        registry=registry,
    )

    with pytest.raises(ExecutionError, match="deterministic_pure"):
        executor.execute({}, n_steps=1, dt=1.0)

    assert list((tmp_path / "cache").iterdir()) == []


@pytest.mark.parametrize(
    ("functor", "class_type", "output_name"),
    [
        (
            _HiddenStateCacheClaimFunctor(),
            HIDDEN_STATE_CACHE_CLAIM_CLASS_TYPE,
            "hidden_state_value",
        ),
        (
            _LocalRNGCacheClaimFunctor(),
            LOCAL_RNG_CACHE_CLAIM_CLASS_TYPE,
            "local_rng_value",
        ),
        (
            _CallableStateCacheClaimFunctor(),
            CALLABLE_STATE_CACHE_CLAIM_CLASS_TYPE,
            "callable_state_value",
        ),
    ],
)
def test_deterministic_pure_claim_rejects_replay_critical_instance_state(
    tmp_path,
    functor,
    class_type,
    output_name,
) -> None:
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.register_functor_instance(functor, node_type_id=class_type)
    definition.outputs[output_name] = _dimensionless_state_port(output_name)
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "invalid_pure_node": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": [output_name],
                    "config": {"cache_policy": "deterministic_pure"},
                }
            },
            "metadata": {},
        }
    )
    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(
        plan,
        cache=NodeCache(tmp_path / "cache"),
        registry=registry,
    )

    with pytest.raises(ExecutionError, match="replay-critical instance state"):
        executor.execute({}, n_steps=1, dt=1.0)

    assert list((tmp_path / "cache").iterdir()) == []


def test_graph_executor_preserves_simulator_config_across_reset() -> None:
    config = {
        "time_step": 900.0,
        "adaptive_dt_enabled": False,
        "enable_auto_scale": False,
    }
    executor = GraphExecutor(
        _build_empty_plan(),
        simulator_config=config,
    )
    config["time_step"] = -1.0

    assert executor.simulator_config["time_step"] == 900.0
    snapshot = executor.simulator_config
    snapshot["time_step"] = -2.0
    assert executor.simulator_config["time_step"] == 900.0

    executor.execute({}, n_steps=0, dt=900.0)

    assert executor.simulator_config["time_step"] == 900.0
    assert executor._simulator.config["time_step"] == 900.0


def test_graph_executor_moves_state_before_cpu_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = GraphExecutor(
        _build_empty_plan(),
        simulator_config={"device": "cpu"},
    )
    events: list[tuple[str, str]] = []
    real_to_device = executor.state_manager.to_device
    real_initialize = executor.state_manager.initialize

    def tracked_to_device(device: torch.device | str) -> None:
        events.append(("to_device", str(device)))
        real_to_device(device)

    def tracked_initialize(
        initial_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        events.append(("initialize", str(executor.state_manager.device)))
        real_initialize(initial_state)

    monkeypatch.setattr(
        executor.state_manager,
        "to_device",
        tracked_to_device,
    )
    monkeypatch.setattr(
        executor.state_manager,
        "initialize",
        tracked_initialize,
    )

    final_state = executor.execute(
        {"x": torch.tensor(1.0)},
        n_steps=0,
        dt=1.0,
    )

    assert events[:2] == [
        ("to_device", "cpu"),
        ("initialize", "cpu"),
    ]
    assert executor.state_manager.device.type == "cpu"
    assert final_state["x"].device.type == "cpu"
    assert executor.execution_summary["device"] == "cpu"


def test_graph_executor_exposes_detached_time_series_snapshot() -> None:
    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    executor = GraphExecutor(plan, registry=registry)

    executor.execute({"x": torch.tensor(0.0)}, n_steps=2, dt=1.0)
    time_series = executor.time_series

    assert len(time_series["x"]) == 2
    time_series["x"][0].fill_(99.0)
    assert float(executor.time_series["x"][0].item()) != 99.0
    assert executor.execution_summary["current_time"] == pytest.approx(2.0)
    assert executor.execution_summary["step_count"] == 2
    assert executor.execution_summary["execution_count"] == 1
    assert executor.execution_summary["execution_order"] == ["n1"]


def test_graph_executor_rejects_non_mapping_simulator_config() -> None:
    with pytest.raises(TypeError, match="simulator_config"):
        GraphExecutor(
            SimpleNamespace(steps=[]),
            simulator_config=["invalid"],
        )


def test_compile_and_preflight_defer_device_preparation_to_execution(monkeypatch):
    from relic.core.simulator import registration

    calls = []
    original = registration._maybe_migrate_functor_device

    def observe(simulator, functor):
        calls.append(functor.name)
        return original(simulator, functor)

    monkeypatch.setattr(registration, "_maybe_migrate_functor_device", observe)
    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    executor = GraphExecutor(plan, registry=registry, simulator_config={"device": "cpu"})
    executor.preflight()
    assert calls == []
    state = executor.execute({"x": torch.tensor(0.0)}, n_steps=1, dt=1.0)
    assert calls == ["n1"]
    assert state["x"].device.type == "cpu"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU/CUDA binding migration requires CUDA")
@pytest.mark.parametrize("original_device", ["cpu", "cuda:0"])
def test_compile_and_preflight_preserve_live_parameter_gradient_edge(original_device, fresh_global_parameter_manager):
    from relic.parameters import get_global_manager
    from relic.parameters.ops.nn_params import relocate_binding
    from relic.parameters.spec import ParameterSpec
    from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import PriestleyTaylorET

    manager = get_global_manager()
    key = "hydrosphere.evapotranspiration.alpha_pt"
    specification = manager.get_parameter(key).to_dict()
    specification.update(dtype="float64", requires_grad=True)
    manager.register_parameter(ParameterSpec.from_dict(specification))
    relocate_binding(manager, key, torch.device(original_device))
    binding = manager.nn_parameters[key]
    old_value = binding.square()
    old_edge = torch.autograd.graph.get_gradient_edge(binding)
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    functor = PriestleyTaylorET()
    class_type = f"Functor:{PriestleyTaylorET.__module__}.{PriestleyTaylorET.__name__}"
    registry.register_functor_instance(functor, node_type_id=class_type)
    inputs = ("net_radiation", "air_temperature", "atmospheric_pressure")
    graph = GraphIR(schema_version="1.0.0", nodes={
        "forcing": NodeSpec(class_type="State:ExternalForcing", outputs=list(inputs)),
        "priestley_taylor_et": NodeSpec(class_type=class_type, config={"execution_group": "fast"},
                                       inputs={name: {"state": name} for name in inputs},
                                       outputs=["evapotranspiration_flux", "latent_heat_flux", "diagnostics.evapotranspiration"]),
    })
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    assert binding.device == torch.device(original_device)
    executor = GraphExecutor(plan, registry=registry, simulator_config={"device": "cpu"})
    executor.preflight()
    assert binding is manager.nn_parameters[key]
    assert binding.device == torch.device(original_device)
    assert torch.autograd.graph.get_gradient_edge(binding).node is old_edge.node
    gradient, = torch.autograd.grad(old_value, binding)
    torch.testing.assert_close(gradient, 2 * binding)
    state = executor.execute({"net_radiation": torch.tensor(400.0), "air_temperature": torch.tensor(298.15),
                              "atmospheric_pressure": torch.tensor(101325.0)}, n_steps=1, dt=1800.0)
    assert binding.device.type == "cpu"
    assert state["evapotranspiration_flux"].device.type == "cpu"
    assert torch.isfinite(torch.autograd.grad(state["evapotranspiration_flux"].sum(), binding)[0]).all()


def test_graph_executor_checkpoint_resume_matches_uninterrupted(tmp_path) -> None:
    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    config = {
        "device": "cpu",
        "time_step": 1.0,
        "adaptive_dt_enabled": False,
        "enable_checkpointing": False,
    }
    initial_state = {"x": torch.tensor(0.0)}

    uninterrupted = GraphExecutor(
        plan,
        registry=registry,
        simulator_config=config,
    )
    uninterrupted_state = uninterrupted.execute(
        initial_state,
        n_steps=2,
        dt=1.0,
    )

    first_leg = GraphExecutor(
        plan,
        registry=registry,
        simulator_config=config,
        output_dir=tmp_path / "provenance",
    )
    first_leg.execute(initial_state, n_steps=1, dt=1.0)
    checkpoint_path = tmp_path / "graph_executor.pt"
    receipt = first_leg.create_checkpoint(checkpoint_path)

    restarted = GraphExecutor(
        plan,
        registry=registry,
        simulator_config=config,
        output_dir=tmp_path / "provenance",
    )
    restarted_state = restarted.execute_from_checkpoint(
        initial_state,
        checkpoint_path,
        n_steps=1,
        dt=1.0,
    )

    assert receipt["path"] == str(checkpoint_path.resolve())
    assert set(restarted_state) == set(uninterrupted_state)
    assert all(
        torch.equal(
            restarted_state[name].detach().cpu(),
            uninterrupted_state[name].detach().cpu(),
        )
        for name in uninterrupted_state
    )
    assert set(restarted.time_series) == set(uninterrupted.time_series)
    for name, uninterrupted_values in uninterrupted.time_series.items():
        restarted_values = restarted.time_series[name]
        assert len(restarted_values) == len(uninterrupted_values)
        assert all(
            torch.equal(
                restarted_value.detach().cpu(),
                uninterrupted_value.detach().cpu(),
            )
            for restarted_value, uninterrupted_value in zip(
                restarted_values,
                uninterrupted_values,
                strict=True,
            )
        )
    assert restarted.execution_summary["current_time"] == pytest.approx(2.0)
    assert restarted.execution_summary["step_count"] == 2
    assert restarted.execution_summary["execution_order"] == ["n1"]
    assert restarted._simulator.run_id == first_leg._simulator.run_id
    assert restarted._recorder.run_id == first_leg._simulator.run_id
    records = [json.loads(line) for line in
               (tmp_path / "provenance" / "runtime.log.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert {record["run_id"] for record in records} == {first_leg._simulator.run_id}
    assert len({record["notification_id"] for record in records}) == 2


def test_graph_executor_rejects_checkpoint_before_preparation(tmp_path) -> None:
    executor = _build_executor()

    with pytest.raises(RuntimeError, match="prepared execution"):
        executor.create_checkpoint(tmp_path / "premature.pt")


def test_checkpoint_rejects_prepared_plan_drift_without_writing(tmp_path) -> None:
    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    executor = GraphExecutor(plan, registry=registry)
    executor.execute({"x": torch.tensor(0.0)}, n_steps=0, dt=1.0)
    plan.steps[0].config["checkpoint_drift"] = True
    checkpoint_path = tmp_path / "must_not_exist.pt"

    with pytest.raises(ValueError, match="does not match its compiler seal"):
        executor.create_checkpoint(checkpoint_path)

    assert not checkpoint_path.exists()


def test_graph_executor_restore_failure_executes_no_steps(tmp_path) -> None:
    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    executor = GraphExecutor(
        plan,
        registry=registry,
        simulator_config={"device": "cpu"},
    )

    with pytest.raises(FileNotFoundError, match="missing or empty"):
        executor.execute_from_checkpoint(
            {"x": torch.tensor(0.0)},
            tmp_path / "missing.pt",
            n_steps=1,
            dt=1.0,
        )

    assert executor.execution_summary["step_count"] == 0
    assert executor.execution_summary["execution_count"] == 0


def test_restart_rejects_plan_drift_before_state_reset(tmp_path) -> None:
    class FailIfReset:
        def reset(self):
            raise AssertionError("StateManager reset must not run")

    registry = _build_registry()
    plan = compile_graph(_build_graph(), registry)
    plan.steps[0].config["restart_drift"] = True
    executor = GraphExecutor(
        plan,
        registry=registry,
        state_manager=FailIfReset(),
    )

    with pytest.raises(ValueError, match="does not match its compiler seal"):
        executor.execute_from_checkpoint(
            {"x": torch.tensor(0.0)},
            tmp_path / "missing.pt",
            n_steps=0,
            dt=1.0,
        )


def test_graph_executor_rejects_missing_lowered_external_state_before_reset():
    class FailIfReset:
        def reset(self):
            raise AssertionError("StateManager reset must not run before preflight")

    plan, registry = _build_external_state_plan("z", "x")
    executor = GraphExecutor(
        plan,
        state_manager=FailIfReset(),
        registry=registry,
    )

    with pytest.raises(
        ValueError,
        match="Missing required initial state keys: x, z",
    ):
        executor.execute({}, n_steps=0, dt=1.0)


def test_graph_executor_preserves_provided_lowered_external_state():
    plan, registry = _build_external_state_plan("x")
    executor = GraphExecutor(plan, registry=registry)
    initial_x = torch.tensor(3.5)

    final_state = executor.execute({"x": initial_x}, n_steps=1, dt=1.0)

    assert plan.steps == []
    assert torch.equal(
        final_state["x"].detach().cpu(),
        initial_x.detach().cpu(),
    )


def test_graph_executor_executes_finite_external_forcing_sequence():
    plan, registry = _build_external_state_plan("forcing")
    executor = GraphExecutor(
        plan,
        registry=registry,
        simulator_config={
            "time_step": 30.0,
            "adaptive_dt_enabled": False,
        },
    )

    final_state = executor.execute_forcing_sequence(
        {"forcing": torch.tensor([0.0, 0.0])},
        [
            {"forcing": torch.tensor([1.0, 2.0])},
            {"forcing": torch.tensor([3.0, 4.0])},
        ],
        dt=30.0,
    )

    torch.testing.assert_close(
        final_state["forcing"].detach().cpu(),
        torch.tensor([3.0, 4.0]),
    )
    assert executor.execution_summary["current_time"] == pytest.approx(60.0)
    assert executor.execution_summary["step_count"] == 2
    assert executor.execution_summary["execution_count"] == 1


@pytest.mark.parametrize(
    ("forcing_sequence", "message"),
    [
        ([], "non-empty"),
        ([{"unknown": torch.tensor([1.0, 2.0])}], "external forcing"),
        ([{"forcing": torch.tensor([1.0])}], "shape"),
        (
            [{"forcing": torch.tensor([1.0, float("nan")])}],
            "finite",
        ),
    ],
)
def test_graph_executor_rejects_invalid_forcing_sequence_before_execution(
    forcing_sequence,
    message,
):
    plan, registry = _build_external_state_plan("forcing")
    executor = GraphExecutor(plan, registry=registry)

    with pytest.raises((TypeError, ValueError), match=message):
        executor.execute_forcing_sequence(
            {"forcing": torch.tensor([0.0, 0.0])},
            forcing_sequence,
            dt=30.0,
        )

    assert executor.execution_summary["step_count"] == 0
    assert executor.execution_summary["execution_count"] == 0


def test_forcing_rejects_unsealed_plan_before_reading_boundary_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = GraphExecutor(SimpleNamespace(steps=[]))

    def fail_if_boundary_is_read(initial_state):
        raise AssertionError(
            f"Boundary metadata must not be read for {sorted(initial_state)}"
        )

    monkeypatch.setattr(
        executor,
        "_validate_required_initial_state",
        fail_if_boundary_is_read,
    )

    with pytest.raises(TypeError, match="canonical ExecutionPlan"):
        executor.execute_forcing_sequence(
            {"forcing": torch.tensor([0.0])},
            [{"forcing": torch.tensor([1.0])}],
            dt=1.0,
        )


def test_explicit_adapter_functor_executes_through_registered_alias():
    registry = NodeRegistry(SSOTCatalog())
    source_class_type = (
        "Functor:test.fixtures.dummy_functor.DummyIncrementFunctor"
    )
    source_definition = registry.register_functor_instance(
        DummyIncrementFunctor(),
        node_type_id=source_class_type,
    )
    source_port = PortSpec(
        name="x",
        standard_name="test_mass_flux",
        units="kg s-1",
        semantics="flux",
        spatial_kind="grid_cell",
        area_basis="ground_area",
        temporal_basis="instantaneous",
        vertical_basis="none",
    )
    source_definition.outputs["x"] = source_port

    adapter_config = {
        "adapter_type": "flux_to_increment",
        "source_node": "source",
        "source_port": "x",
        "target_port": "x_increment",
    }
    adapter = ExecutableAdapterFunctor(
        name="flux_to_increment",
        **adapter_config,
    )
    adapter_class_type = "Functor:adapter.flux_to_increment.x"
    adapter_registry = AdapterRegistry()
    adapter_registry.register_executable_node(
        node_registry=registry,
        node_type_id=adapter_class_type,
        functor=adapter,
        source_port=source_port,
        target_port=PortSpec(
            name="x_increment",
            standard_name="test_mass_increment",
            units="kg",
            semantics="increment",
            write_mode="mutable",
            spatial_kind="grid_cell",
            area_basis="ground_area",
            temporal_basis="interval_integral",
            vertical_basis="none",
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": source_class_type,
                    "inputs": {},
                    "outputs": ["x"],
                    "config": {},
                },
                "adapter_node": {
                    "class_type": adapter_class_type,
                    "inputs": {"x": {"link": ["source", "x"]}},
                    "outputs": ["x_increment"],
                    "config": {
                        **adapter_config,
                        "execution_group": "fast",
                    },
                },
            },
            "links": [
                {
                    "from": {"node": "source", "port": "x"},
                    "to": {"node": "adapter_node", "port": "x"},
                }
            ],
            "metadata": {},
        }
    )

    plan = compile_graph(
        graph,
        registry,
        adapter_registry=adapter_registry,
        strict=True,
    )
    executor = GraphExecutor(plan, registry=registry)
    final_state = executor.execute(
        {
            "x": torch.tensor(0.0),
            "x_increment": torch.tensor(0.0),
        },
        n_steps=1,
        dt=2.0,
    )

    assert torch.allclose(
        final_state["x_increment"],
        torch.full_like(final_state["x_increment"], 2.0),
    )
    assert "x_increment" not in executor.state_manager._current_diagnostics
    assert float(
        executor._functor_instances["adapter_node"]
        .last_adapter_metadata["dt"]
        .detach()
        .cpu()
        .item()
    ) == 2.0
    assert "adapter_node" in executor._simulator.get_functor_summary()
    assert any(
        record["functor"] == "adapter_node"
        for record in executor._simulator.get_variable_provenance("x_increment")
    )


def test_explicit_cache_policy_preserves_autograd_by_bypassing_cache(tmp_path):
    registry = NodeRegistry(SSOTCatalog())
    source_definition = registry.register_functor_instance(
        _GradientSourceFunctor(),
        node_type_id=GRADIENT_SOURCE_CLASS_TYPE,
    )
    sink_definition = registry.register_functor_instance(
        _GradientSinkFunctor(),
        node_type_id=GRADIENT_SINK_CLASS_TYPE,
    )
    raw_link_port = _dimensionless_state_port(
        "y",
        standard_name="gradient_test_signal",
    )
    source_definition.inputs["x"] = _dimensionless_state_port("x")
    source_definition.outputs["y"] = raw_link_port
    sink_definition.inputs["y"] = raw_link_port
    sink_definition.outputs["z"] = _dimensionless_state_port("z")
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": GRADIENT_SOURCE_CLASS_TYPE,
                    "inputs": {"x": {"state": "x"}},
                    "outputs": ["y"],
                    "config": {
                        "execution_group": "fast",
                        "cache_policy": "deterministic_pure",
                    },
                },
                "sink": {
                    "class_type": GRADIENT_SINK_CLASS_TYPE,
                    "inputs": {"y": {"link": ["source", "y"]}},
                    "outputs": ["z"],
                    "config": {
                        "execution_group": "fast",
                        "cache_policy": "deterministic_pure",
                    },
                },
            },
            "metadata": {},
        }
    )

    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(
        plan,
        registry=registry,
        cache=NodeCache(tmp_path / "cache"),
        strategy=ExecutionStrategy.PUSH,
    )
    input_x = torch.tensor(
        3.0,
        device=executor._simulator.device,
        requires_grad=True,
    )

    final_state = executor.execute({"x": input_x}, n_steps=1, dt=1.0)
    gradient = torch.autograd.grad(final_state["z"], input_x)[0]

    assert [step.node_id for step in plan.steps] == ["source", "sink"]
    assert executor._strategy is ExecutionStrategy.PUSH
    assert executor.cache is not None
    assert list((tmp_path / "cache").iterdir()) == []
    assert torch.allclose(final_state["z"], torch.full_like(final_state["z"], 36.0))
    assert torch.allclose(gradient, torch.full_like(gradient, 24.0))
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.ne(gradient, 0).all())


def test_compile_and_execute_smoke() -> None:
    registry = _build_registry()
    graph = _build_graph()
    result = compile_and_execute(
        graph,
        registry,
        {"x": torch.tensor(0.0)},
        n_steps=1,
        dt=1.0,
    )
    assert float(result["x"].item()) == 1.0


def test_compile_and_execute_accepts_canonical_simulator_config() -> None:
    registry = _build_registry()
    graph = _build_graph()

    result = compile_and_execute(
        graph,
        registry,
        {"x": torch.tensor(0.0)},
        n_steps=2,
        dt=1.0,
        simulator_config={
            "time_step": 1.0,
            "adaptive_dt_enabled": False,
            "enable_auto_scale": False,
        },
    )

    assert float(result["x"].item()) == 2.0


# ---------------------------------------------------------------------------
# D03: cache dependency closure, provider-identity key binding, and epoch
# consistency consumed from the simulator's committed provider view.
# ---------------------------------------------------------------------------

UNRELATED_PURE_CLASS_TYPE = "Functor:test.unrelated_pure"


class _UnrelatedPureFunctor(PhysicalFieldFunctor):
    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self, name: str = "unrelated_pure", config: dict | None = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            description="Independent pure node for closure invalidation tests.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="v = u + 1",
            references=["RELIC internal cache closure verification"],
            parameters={},
            input_variables=["u"],
            output_variables=["v"],
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC internal cache closure verification.
        # Mathematical expression: v = u + 1.
        # Parameter meanings: u is the independent test input.
        return {"v": state["u"] + 1.0}


def _build_closure_registry_graph(*, sink_input: dict) -> tuple[NodeRegistry, GraphIR]:
    registry = NodeRegistry(SSOTCatalog())
    source_definition = registry.register_functor_instance(
        _GradientSourceFunctor(), node_type_id=GRADIENT_SOURCE_CLASS_TYPE,
    )
    sink_definition = registry.register_functor_instance(
        _GradientSinkFunctor(), node_type_id=GRADIENT_SINK_CLASS_TYPE,
    )
    unrelated_definition = registry.register_functor_instance(
        _UnrelatedPureFunctor(), node_type_id=UNRELATED_PURE_CLASS_TYPE,
    )
    signal = _dimensionless_state_port("y", standard_name="gradient_test_signal")
    source_definition.inputs["x"] = _dimensionless_state_port("x")
    source_definition.outputs["y"] = signal
    sink_definition.inputs["y"] = signal
    sink_definition.outputs["z"] = _dimensionless_state_port("z")
    unrelated_definition.inputs["u"] = _dimensionless_state_port("u")
    unrelated_definition.outputs["v"] = _dimensionless_state_port("v")
    config = {"execution_group": "fast", "cache_policy": "deterministic_pure"}
    graph = GraphIR.from_dict({
        "schema_version": "1.0.0",
        "nodes": {
            "source": {
                "class_type": GRADIENT_SOURCE_CLASS_TYPE,
                "inputs": {"x": {"state": "x"}},
                "outputs": ["y"],
                "config": dict(config),
            },
            "sink": {
                "class_type": GRADIENT_SINK_CLASS_TYPE,
                "inputs": {"y": dict(sink_input)},
                "outputs": ["z"],
                "config": dict(config),
            },
            "unrelated": {
                "class_type": UNRELATED_PURE_CLASS_TYPE,
                "inputs": {"u": {"state": "u"}},
                "outputs": ["v"],
                "config": dict(config),
            },
        },
        "metadata": {},
    })
    return registry, graph


def _observe_kernel_calls(monkeypatch) -> list[str]:
    calls: list[str] = []
    original_module_call = torch.nn.Module._call_impl

    def observe_module_call(self, *args, **kwargs):
        if isinstance(self, (_GradientSourceFunctor, _GradientSinkFunctor, _UnrelatedPureFunctor)):
            calls.append(self.name)
        return original_module_call(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "_call_impl", observe_module_call)
    return calls


def test_upstream_provider_change_invalidates_consumer_closure_and_keeps_unrelated_cache(
    tmp_path, monkeypatch,
):
    registry, graph = _build_closure_registry_graph(sink_input={"link": ["source", "y"]})
    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(
        plan, registry=registry, cache=NodeCache(tmp_path / "cache"),
        simulator_config={"device": "cpu"},
    )
    initial = {"x": torch.tensor(3.0), "u": torch.tensor(1.0)}
    calls = _observe_kernel_calls(monkeypatch)

    first = executor.execute(dict(initial), n_steps=1, dt=1.0)
    assert float(first["z"]) == 36.0 and float(first["v"]) == 2.0
    assert sorted(calls) == ["sink", "source", "unrelated"]
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 3
    digests_before = dict(executor._provider_digests)
    epoch_key_before = executor._provider_view_key

    # Same identities under a new simulator epoch: every node hits the cache.
    calls.clear()
    second = executor.execute(dict(initial), n_steps=1, dt=1.0)
    assert float(second["z"]) == 36.0 and calls == []
    assert executor._provider_digests == digests_before
    assert executor._provider_view_key != epoch_key_before
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 3

    # Upstream implementation change: source and its consumer closure miss,
    # the stale entries are evicted, and the unrelated node keeps its hit.
    kernel = _GradientSourceFunctor.forward
    monkeypatch.setattr(kernel, "__code__", _replacement_source_kernel.__code__)
    calls.clear()
    third = executor.execute(dict(initial), n_steps=1, dt=1.0)
    assert float(third["z"]) == 81.0 and float(third["v"]) == 2.0
    assert sorted(calls) == ["sink", "source"]
    assert executor._provider_digests["source"] != digests_before["source"]
    assert executor._provider_digests["sink"] == digests_before["sink"]
    assert executor._provider_digests["unrelated"] == digests_before["unrelated"]
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 3


def test_invalidate_node_closure_reaches_shared_state_readers_and_dirty_tracking(tmp_path):
    registry, graph = _build_closure_registry_graph(sink_input={"state": "y"})
    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(
        plan, registry=registry, strategy=ExecutionStrategy.DIRTY,
        cache=NodeCache(tmp_path / "cache"), simulator_config={"device": "cpu"},
    )
    initial = {"x": torch.tensor(3.0), "u": torch.tensor(1.0), "y": torch.tensor(0.0)}
    executor.execute(dict(initial), n_steps=1, dt=1.0)
    assert executor._dirty_nodes == set()
    assert set(executor._last_outputs) == {"source", "sink", "unrelated"}
    entries_before = {path.name for path in (tmp_path / "cache").glob("*.pkl")}
    assert len(entries_before) == 3

    closure = executor.invalidate_node_closure("source")
    assert closure == ("sink", "source")
    assert executor._dirty_nodes == {"sink", "source"}
    assert set(executor._last_outputs) == {"unrelated"}
    assert set(executor._last_input_hashes) == {"unrelated"}
    entries_after = {path.name for path in (tmp_path / "cache").glob("*.pkl")}
    assert len(entries_after) == 1 and entries_after < entries_before

    assert executor.invalidate_node_closure("unrelated") == ("unrelated",)
    assert executor._dirty_nodes == {"sink", "source", "unrelated"}
    with pytest.raises(KeyError):
        executor.invalidate_node_closure("ghost")


def test_cache_key_binds_provider_binding_and_upstream_identity(tmp_path):
    registry, graph = _build_closure_registry_graph(sink_input={"link": ["source", "y"]})
    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(
        plan, registry=registry, cache=NodeCache(tmp_path / "cache"),
        simulator_config={"device": "cpu"},
    )
    executor.execute({"x": torch.tensor(3.0), "u": torch.tensor(1.0)}, n_steps=0, dt=1.0)
    executor._reconcile_provider_view()
    sink_contract = executor._provider_binding_contract("sink")
    assert set(sink_contract["upstream_providers"]) == {"source"}
    assert sink_contract["provider_identity"] == executor._provider_digests["sink"]
    assert sink_contract["upstream_providers"]["source"] == executor._provider_digests["source"]
    assert sink_contract["rng_contract"] == "deterministic_pure_no_runtime_rng"
    assert executor._provider_binding_contract("unrelated")["upstream_providers"] == {}
    view = executor._simulator.get_committed_provider_view()
    provider = view.provider("source")
    assert executor._provider_digests["source"] == (
        provider.implementation_digest, provider.contract_digest, provider.binding_digest,
    )


def test_resolve_functor_class_never_scans_the_registry_on_a_miss(monkeypatch):
    import relic.orchestration.executor as executor_module

    def _forbidden_scan():
        raise AssertionError("a lookup miss triggered a functor registry scan")

    monkeypatch.setattr(executor_module, "register_all_functors", _forbidden_scan, raising=False)
    executor = _build_executor()

    with pytest.raises(KeyError, match="no_such_functor_probe"):
        executor._resolve_functor_class("no_such_functor_probe")
    assert executor._resolve_functor_class("bucket_soil_moisture").__name__ == "BucketSoilMoisture"


@pytest.mark.parametrize("parameter", ["n_steps", "dt"])
def test_graph_execution_states_step_count_and_time_step(parameter):
    """No silent 3600 s step or single-step run: callers state both."""
    import inspect

    from relic.orchestration.compiler import compile_and_execute
    from relic.orchestration.executor import GraphExecutor

    for function in (GraphExecutor.execute, compile_and_execute):
        signature = inspect.signature(function)
        assert signature.parameters[parameter].default is inspect.Parameter.empty, function.__qualname__
