"""Compatible restart under an explicit certificate (F03).

Exact restart stays the default and rejects every topology or contract
difference. A compatible restart requires a ``RestartCompatibilityCertificate``
that binds the real source and target topology digests, certifies the
contract leaf paths that may differ, bounds numeric parameter drift, and maps
renamed state variables with identical units, shape, and dtype.
Implementation identity, graph structure, runtime environment, and parameter
specifications can never be certified away.
"""

from __future__ import annotations

import pytest
import torch

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
)
from relic.core.simulator import RELICSimulator
from relic.core.simulator.checkpoint import (
    RestartCompatibilityCertificate,
    compute_restart_topology_digests,
)
from relic.core.state.types import StateVariableSpec
from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog
from test.fixtures.dummy_functor import DummyIncrementFunctor


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'writer_out': 'dimensionless',
    'writer_out_v2': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class _Writer(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name="writer", output="writer_out", gain=1.0, config=None):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="compatible restart fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=[output],
        )
        super().__init__(name=name, config=dict(config or {}), metadata=metadata)
        self._output = output
        self.parameter_values["gain"] = float(gain)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {self._output: torch.tensor(2.5) * float(self.parameter_values["gain"])}


class _OtherWriter(_Writer):
    def forward(self, state, time, dt, **kwargs):
        return {self._output: torch.tensor(2.5) * float(self.parameter_values["gain"]) * 1.0}


def _simulator(functor, *, state_name="writer_out", units=None, shape=()):
    sim = _declared_simulator({
        "device": "cpu", "conservation_checks": False,
        "strict_dependencies": False, "adaptive_dt_enabled": False,
    })
    sim.initialize_state({state_name: torch.zeros(shape)})
    if units is not None:
        sim.state_manager.register_variable(StateVariableSpec(name=state_name, units=units))
    sim.register_functor(functor, execution_group="fast")
    return sim


def _certificate(target, path, **overrides):
    digests = compute_restart_topology_digests(target, path)
    fields = dict(
        source_topology_sha256=digests["source_topology_sha256"],
        target_topology_sha256=digests["target_topology_sha256"],
        forcing_cursor_policy="exact",
    )
    fields.update(overrides)
    return RestartCompatibilityCertificate(**fields)


GAIN_PATH = "scientific_contract.functors.writer.parameter_values.gain"


class TestParameterDriftCertificate:
    def test_exact_rejects_drift_and_certificate_accepts_bounded_drift(self, tmp_path):
        source = _simulator(_Writer(gain=1.0))
        source.step(dt=1.0)
        path = tmp_path / "source.pt"
        source.create_checkpoint(path)

        target = _simulator(_Writer(gain=1.05))
        with pytest.raises(ValueError, match="scientific_contract"):
            target.restore_checkpoint(path)
        with pytest.raises(ValueError, match="uncertified"):
            target.restore_checkpoint(
                path, compatibility_certificate=_certificate(target, path),
            )
        certificate = _certificate(
            target, path, allowed_contract_paths=(GAIN_PATH,), parameter_relative_tolerance=0.1,
        )
        epoch_before = target.provider_epoch
        receipt = target.restore_checkpoint(path, compatibility_certificate=certificate)
        assert receipt["restart_mode"] == "compatible"
        assert receipt["provider_epoch"] == epoch_before + 1
        audit = receipt["compatibility"]
        assert audit["certificate_sha256"] == certificate.sha256()
        accepted = {item["path"]: item["reason"] for item in audit["accepted_differences"]}
        assert accepted[GAIN_PATH] == "certified"
        assert accepted["scientific_contract.functors.writer.provider_identity.binding_digest"] == "derived_binding_digest"
        assert audit["mapped_state_variables"] == []
        assert target.step_count == 1
        assert float(target.get_state()["writer_out"]) == 2.5
        assert float(target.functors["writer"].parameter_values["gain"]) == 1.05
        target.step(dt=1.0)
        # Plain tensor outputs accumulate: 2.5 (restored) + 2.5 * 1.05.
        assert float(target.get_state()["writer_out"]) == pytest.approx(5.125)

    def test_drift_beyond_tolerance_is_rejected_before_mutation(self, tmp_path):
        source = _simulator(_Writer(gain=1.0))
        source.step(dt=1.0)
        path = tmp_path / "source.pt"
        source.create_checkpoint(path)
        target = _simulator(_Writer(gain=1.5))
        certificate = _certificate(
            target, path, allowed_contract_paths=(GAIN_PATH,), parameter_relative_tolerance=0.1,
        )
        with pytest.raises(ValueError, match="exceeds the certified relative tolerance"):
            target.restore_checkpoint(path, compatibility_certificate=certificate)
        assert target.step_count == 0 and float(target.get_state()["writer_out"]) == 0.0

    def test_certificate_must_name_the_real_source_and_target(self, tmp_path):
        source = _simulator(_Writer(gain=1.0))
        source.step(dt=1.0)
        path = tmp_path / "source.pt"
        source.create_checkpoint(path)
        target = _simulator(_Writer(gain=1.05))
        digests = compute_restart_topology_digests(target, path)
        assert digests["source_topology_sha256"] != digests["target_topology_sha256"]
        wrong_source = RestartCompatibilityCertificate(
            source_topology_sha256="0" * 64,
            target_topology_sha256=digests["target_topology_sha256"],
            allowed_contract_paths=(GAIN_PATH,), parameter_relative_tolerance=1.0, forcing_cursor_policy="exact",
        )
        with pytest.raises(ValueError, match="source topology digest"):
            target.restore_checkpoint(path, compatibility_certificate=wrong_source)
        wrong_target = RestartCompatibilityCertificate(
            source_topology_sha256=digests["source_topology_sha256"],
            target_topology_sha256="0" * 64,
            allowed_contract_paths=(GAIN_PATH,), parameter_relative_tolerance=1.0, forcing_cursor_policy="exact",
        )
        with pytest.raises(ValueError, match="target topology digest"):
            target.restore_checkpoint(path, compatibility_certificate=wrong_target)
        with pytest.raises(TypeError, match="RestartCompatibilityCertificate"):
            target.restore_checkpoint(path, compatibility_certificate={"migration_verified": True})
        with pytest.raises(TypeError):
            RestartCompatibilityCertificate(
                source_topology_sha256=digests["source_topology_sha256"],
                target_topology_sha256=digests["target_topology_sha256"],
                migration_verified=True,  # type: ignore[call-arg], forcing_cursor_policy="exact"
            )

    def test_certificate_on_an_identical_topology_records_no_differences(self, tmp_path):
        source = _simulator(_Writer(gain=1.0))
        source.step(dt=1.0)
        path = tmp_path / "source.pt"
        source.create_checkpoint(path)
        target = _simulator(_Writer(gain=1.0))
        receipt = target.restore_checkpoint(
            path, compatibility_certificate=_certificate(target, path),
        )
        assert receipt["restart_mode"] == "compatible"
        assert receipt["compatibility"]["accepted_differences"] == []
        assert target.step_count == 1


class TestUnwaivableDifferences:
    def test_implementation_change_cannot_be_certified(self, tmp_path):
        source = _simulator(_Writer(gain=1.0))
        source.step(dt=1.0)
        path = tmp_path / "source.pt"
        source.create_checkpoint(path)
        target = _simulator(_OtherWriter(gain=1.0))
        certificate = _certificate(
            target, path, allowed_contract_paths=("scientific_contract",), parameter_relative_tolerance=1.0,
        )
        with pytest.raises(ValueError, match="unwaivable") as info:
            target.restore_checkpoint(path, compatibility_certificate=certificate)
        message = str(info.value)
        assert "kernel_identity" in message and "implementation_digest" in message
        assert "functor_types" in message
        assert target.step_count == 0

    def test_parameter_specs_and_config_without_certification_are_rejected(self, tmp_path):
        source = _simulator(_Writer(gain=1.0, config={"mode": "a"}))
        source.step(dt=1.0)
        path = tmp_path / "source.pt"
        source.create_checkpoint(path)
        target = _simulator(_Writer(gain=1.0, config={"mode": "b"}))
        with pytest.raises(ValueError, match="uncertified"):
            target.restore_checkpoint(
                path, compatibility_certificate=_certificate(target, path),
            )
        # Config entries are mirrored into parameter_values by FunctorBase, so
        # both leaves change; the non-numeric value needs its exact path.
        prefix_only = _certificate(
            target, path, allowed_contract_paths=("scientific_contract.functors.writer",),
        )
        with pytest.raises(ValueError, match="non-numeric parameter change"):
            target.restore_checkpoint(path, compatibility_certificate=prefix_only)
        certificate = _certificate(
            target, path,
            allowed_contract_paths=(
                "scientific_contract.functors.writer.config.mode",
                "scientific_contract.functors.writer.parameter_values.mode",
            ),
        )
        receipt = target.restore_checkpoint(path, compatibility_certificate=certificate)
        reasons = {item["path"]: item["reason"] for item in receipt["compatibility"]["accepted_differences"]}
        assert reasons["scientific_contract.functors.writer.config.mode"] == "certified"
        assert reasons["scientific_contract.functors.writer.parameter_values.mode"] == "certified"
        assert reasons["scientific_contract.functors.writer.provider_identity.binding_digest"] == "derived_binding_digest"
        assert reasons["scientific_contract.functors.writer.provider_identity.contract_digest"] == "derived_contract_digest"


class TestStateVariableMap:
    def _source_checkpoint(self, tmp_path, **kwargs):
        source = _simulator(_Writer(output="writer_out"), **kwargs)
        source.step(dt=1.0)
        path = tmp_path / "source.pt"
        source.create_checkpoint(path)
        return path

    def _rename_paths(self):
        return (
            "scientific_contract.functors.writer.metadata",
            "scientific_contract.state_manager.schema",
        )

    def test_certified_rename_restores_the_saved_value_under_the_new_name(self, tmp_path):
        path = self._source_checkpoint(tmp_path, units="kg m-2")
        target = _simulator(_Writer(output="writer_out_v2"), state_name="writer_out_v2", units="kg m-2")
        with pytest.raises(ValueError, match="uncertified"):
            target.restore_checkpoint(path, compatibility_certificate=_certificate(target, path))
        certificate = _certificate(
            target, path, allowed_contract_paths=self._rename_paths(),
            state_variable_map={"writer_out": "writer_out_v2"},
        )
        receipt = target.restore_checkpoint(path, compatibility_certificate=certificate)
        assert receipt["compatibility"]["mapped_state_variables"] == [
            {"saved": "writer_out", "target": "writer_out_v2", "units": "kg m-2"},
        ]
        state = target.get_state()
        assert "writer_out" not in state
        assert float(state["writer_out_v2"]) == 2.5
        target.step(dt=1.0)
        assert target.step_count == 2

    def test_units_shape_and_missing_mappings_are_rejected(self, tmp_path):
        path = self._source_checkpoint(tmp_path, units="kg m-2")
        wrong_units = _simulator(_Writer(output="writer_out_v2"), state_name="writer_out_v2", units="mol m-2")
        with pytest.raises(ValueError, match="units differ"):
            wrong_units.restore_checkpoint(
                path, compatibility_certificate=_certificate(
                    wrong_units, path, allowed_contract_paths=self._rename_paths(),
                    state_variable_map={"writer_out": "writer_out_v2"},
                ),
            )
        wrong_shape = _simulator(
            _Writer(output="writer_out_v2"), state_name="writer_out_v2", units="kg m-2", shape=(2,),
        )
        with pytest.raises(ValueError, match="shape or dtype"):
            wrong_shape.restore_checkpoint(
                path, compatibility_certificate=_certificate(
                    wrong_shape, path, allowed_contract_paths=self._rename_paths(),
                    state_variable_map={"writer_out": "writer_out_v2"},
                ),
            )
        target = _simulator(_Writer(output="writer_out_v2"), state_name="writer_out_v2", units="kg m-2")
        with pytest.raises(ValueError, match="does not carry"):
            target.restore_checkpoint(
                path, compatibility_certificate=_certificate(
                    target, path, allowed_contract_paths=self._rename_paths(),
                    state_variable_map={"ghost": "writer_out_v2"},
                ),
            )
        with pytest.raises(ValueError, match="two saved names to one target"):
            RestartCompatibilityCertificate(
                source_topology_sha256="a" * 64, target_topology_sha256="b" * 64,
                state_variable_map={"x": "z", "y": "z"}, forcing_cursor_policy="exact",
            )
        assert target.step_count == 0


class TestForcingCursorPolicy:
    def test_clear_policy_drops_the_saved_cursor_and_exact_keeps_it(self, tmp_path):
        source = _simulator(_Writer(gain=1.0))
        source.bind_forcing_cursor(sequence_sha256="c" * 64, length=4)
        source.step(dt=1.0)
        path = tmp_path / "cursor.pt"
        source.create_checkpoint(path)
        exact_target = _simulator(_Writer(gain=1.0))
        exact_target.restore_checkpoint(path, compatibility_certificate=_certificate(exact_target, path))
        assert exact_target.get_forcing_cursor()["consumed"] == 1
        cleared_target = _simulator(_Writer(gain=1.0))
        cleared_target.restore_checkpoint(
            path, compatibility_certificate=_certificate(cleared_target, path, forcing_cursor_policy="clear"),
        )
        assert cleared_target.get_forcing_cursor() is None
        with pytest.raises(ValueError, match="forcing_cursor_policy"):
            RestartCompatibilityCertificate(
                source_topology_sha256="a" * 64, target_topology_sha256="b" * 64,
                forcing_cursor_policy="reseed",
            )


CLASS_TYPE = "Functor:test.fixtures.dummy_functor.DummyIncrementFunctor"


def _executor(output_frequency: int) -> GraphExecutor:
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.build_from_node_spec(CLASS_TYPE, [], ["x"])
    definition.outputs["x"] = PortSpec(
        name="x", standard_name="x", units="1", semantics="state", spatial_kind="grid_cell",
        area_basis="ground_area", temporal_basis="instantaneous", vertical_basis="none",
    )
    registry.register_definition(CLASS_TYPE, definition)
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"n1": NodeSpec(class_type=CLASS_TYPE, inputs={}, outputs=["x"], config={})},
        links=[], metadata={},
    )
    plan = compile_graph(graph, registry)
    return GraphExecutor(
        plan, registry=registry,
        simulator_config={"device": "cpu", "time_step": 1.0, "adaptive_dt_enabled": False,
                          "output_frequency": output_frequency},
    )


def test_graph_executor_passes_the_certificate_to_the_checkpoint_owner(tmp_path):
    producer = _executor(output_frequency=1)
    producer.execute({"x": torch.tensor(0.0)}, n_steps=2, dt=1.0)
    path = tmp_path / "graph.pt"
    producer.create_checkpoint(path)

    consumer = _executor(output_frequency=2)
    with pytest.raises(ValueError, match="scientific_contract"):
        consumer.execute_from_checkpoint({"x": torch.tensor(0.0)}, path, n_steps=1, dt=1.0)
    consumer.execute({"x": torch.tensor(0.0)}, n_steps=0, dt=1.0)
    certificate = _certificate(
        consumer._simulator, path,
        allowed_contract_paths=("scientific_contract.simulator_config.output_frequency",),
    )
    final = consumer.execute_from_checkpoint(
        {"x": torch.tensor(0.0)}, path, n_steps=1, dt=1.0,
        compatibility_certificate=certificate,
    )
    assert float(final["x"]) == 3.0
    assert consumer._simulator.step_count == 3
    assert isinstance(DummyIncrementFunctor, type)

def test_parameter_contract_does_not_drop_a_failing_manager():
    """A manager that cannot be obtained fails the contract instead of being skipped."""
    from types import SimpleNamespace

    from relic.core.simulator.checkpoint import _collect_parameter_managers

    def _broken():
        raise RuntimeError("parameter database unavailable")

    functor = SimpleNamespace(_parameter_manager=None, get_parameter_manager=_broken)
    with pytest.raises(RuntimeError, match="parameter database unavailable"):
        _collect_parameter_managers({"probe": functor})
