"""Cross-graph migration replay under an explicit migration certificate (F04).

A saved graph is mapped onto a target graph inside a restricted family:
mapped Functors keep type, group, implementation identity, and pairwise
dependencies on the common domain; retired and new Functors and dropped state
variables must be declared; history is preserved. The migration validates
first and commits atomically; identity and composition laws hold on the
common domain; equal, approximate, and incomparable observations are stated
explicitly by the tests.
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
    GraphMigrationCertificate,
    compute_restart_topology_digests,
)
from relic.core.state.types import StateVariableSpec
from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'a_out': 'dimensionless',
    'a_total': 'dimensionless',
    'c_out': 'dimensionless',
    'x_out': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class _Accumulator(TransportMechanismFunctor):
    """out += rate * dt each step (ACCUMULATE of a plain increment)."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name, output, rate=1.0, dependencies=()):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="migration fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=[output],
        )
        super().__init__(name=name, config={"rate": float(rate)}, metadata=metadata)
        self._output = output
        self._deps = list(dependencies)

    def get_dependencies(self):
        return list(self._deps)

    def forward(self, state, time, dt, **kwargs):
        return {self._output: torch.tensor(self._original_config["rate"]) * float(dt)}


def _simulator(functors, state, *, units=None):
    sim = _declared_simulator({
        "device": "cpu", "conservation_checks": False,
        "strict_dependencies": False, "adaptive_dt_enabled": False, "output_frequency": 1,
    })
    sim.initialize_state({name: torch.tensor(float(value)) for name, value in state.items()})
    for name, unit in (units or {}).items():
        sim.state_manager.register_variable(StateVariableSpec(name=name, units=unit))
    for functor in functors:
        sim.register_functor(functor, execution_group="fast")
    return sim


def test_certificate_policies_are_declared_by_the_certificate():
    """replay_mode, new_functor_rng and forcing_cursor_policy select replay
    semantics; each certificate names them (no default policy)."""
    import dataclasses

    from relic.core.simulator.checkpoint import RestartCompatibilityCertificate

    for cls, names in (
        (GraphMigrationCertificate, ("replay_mode", "new_functor_rng", "forcing_cursor_policy")),
        (RestartCompatibilityCertificate, ("forcing_cursor_policy",)),
    ):
        fields = {field.name: field for field in dataclasses.fields(cls)}
        for name in names:
            assert fields[name].default is dataclasses.MISSING, (cls.__name__, name)
            assert fields[name].kw_only, (cls.__name__, name)


def _certificate(target, path, **overrides):
    digests = compute_restart_topology_digests(target, path)
    fields = dict(
        source_topology_sha256=digests["source_topology_sha256"],
        target_topology_sha256=digests["target_topology_sha256"],
        functor_map={},
        replay_mode="switch",
        new_functor_rng="fresh_substream",
        forcing_cursor_policy="exact",
    )
    fields.update(overrides)
    return GraphMigrationCertificate(**fields)


def _graph_a():
    return _simulator(
        [_Accumulator("wa", "a_out", rate=1.0), _Accumulator("wx", "x_out", rate=3.0)],
        {"a_out": 0.0, "x_out": 0.0},
    )


def _graph_b():
    return _simulator(
        [_Accumulator("wb", "a_out", rate=1.0), _Accumulator("wc", "c_out", rate=5.0)],
        {"a_out": 0.0, "c_out": 100.0},
    )


def _series(sim, name):
    return [float(t) for t in sim.time_series.get(name, [])]


class TestMigrationLaws:
    def test_identity_migration_equals_exact_restore(self, tmp_path):
        source = _graph_a()
        source.step(dt=1.0)
        source.step(dt=1.0)
        path = tmp_path / "a.pt"
        source.create_checkpoint(path)

        exact = _graph_a()
        exact.restore_checkpoint(path)
        exact.step(dt=1.0)

        identity = _graph_a()
        receipt = identity.restore_checkpoint(
            path, migration_certificate=_certificate(identity, path, functor_map={"wa": "wa", "wx": "wx"}),
        )
        assert receipt["restart_mode"] == "migration"
        assert receipt["compatibility"]["accepted_differences"] == []
        assert receipt["migration"]["mapped_state_variables"] == []
        identity.step(dt=1.0)
        for name in ("a_out", "x_out"):
            assert float(identity.get_state()[name]) == float(exact.get_state()[name])
            assert _series(identity, name) == _series(exact, name)
        assert identity.step_count == exact.step_count == 3

    def test_composition_of_migrations_equals_the_direct_migration(self, tmp_path):
        source = _graph_a()
        source.step(dt=1.0)
        path_a = tmp_path / "a.pt"
        source.create_checkpoint(path_a)

        def graph_b():
            return _simulator([_Accumulator("wb", "a_out"), _Accumulator("wx", "x_out", rate=3.0)],
                              {"a_out": 0.0, "x_out": 0.0})

        def graph_c():
            return _simulator([_Accumulator("wd", "a_out"), _Accumulator("wx", "x_out", rate=3.0)],
                              {"a_out": 0.0, "x_out": 0.0})

        via_b = graph_b()
        via_b.restore_checkpoint(
            path_a, migration_certificate=_certificate(via_b, path_a, functor_map={"wa": "wb", "wx": "wx"}),
        )
        path_b = tmp_path / "b.pt"
        via_b.create_checkpoint(path_b)
        composed = graph_c()
        composed.restore_checkpoint(
            path_b, migration_certificate=_certificate(composed, path_b, functor_map={"wb": "wd", "wx": "wx"}),
        )

        direct = graph_c()
        direct.restore_checkpoint(
            path_a, migration_certificate=_certificate(direct, path_a, functor_map={"wa": "wd", "wx": "wx"}),
        )
        for sim in (composed, direct):
            sim.step(dt=1.0)
            sim.step(dt=1.0)
        for name in ("a_out", "x_out"):
            assert float(composed.get_state()[name]) == float(direct.get_state()[name])
            assert _series(composed, name) == _series(direct, name)
        assert composed.step_count == direct.step_count == 3


class TestSwitchMigration:
    def test_rename_add_and_retire_with_history_preserved(self, tmp_path):
        source = _graph_a()
        for _ in range(3):
            source.step(dt=1.0)
        path = tmp_path / "a.pt"
        source.create_checkpoint(path)
        # The time series holds the emitted increments, one per accepted step.
        assert _series(source, "x_out") == [3.0, 3.0, 3.0]

        target = _graph_b()
        with pytest.raises(ValueError, match="does not match the active simulator"):
            target.restore_checkpoint(path)
        certificate = _certificate(
            target, path,
            functor_map={"wa": "wb"}, retired_functors=("wx",), new_functors=("wc",),
            retired_state_variables=("x_out",),
        )
        receipt = target.restore_checkpoint(path, migration_certificate=certificate)
        migration = receipt["migration"]
        assert migration["certificate_sha256"] == certificate.sha256()
        assert migration["retired_functors"] == ["wx"] and migration["new_functors"] == ["wc"]
        assert migration["preserved_history_variables"] == ["x_out"]
        assert migration["new_functor_rng"] == "fresh_substream"
        state = target.get_state()
        assert float(state["a_out"]) == 3.0
        assert float(state["c_out"]) == 100.0  # new Functor starts from live initialization
        assert "x_out" not in state
        assert _series(target, "x_out") == [3.0, 3.0, 3.0]  # history is never disposed
        assert target.step_count == 3 and target.current_time == 3.0

        target.step(dt=1.0)
        # Reference: the target graph started from the mapped state at the
        # same time gives the same next step (equal observation).
        reference = _graph_b()
        reference.state_manager.set_state({"a_out": torch.tensor(3.0), "c_out": torch.tensor(100.0)}, validate=False)
        reference.current_time = 3.0
        reference.step(dt=1.0)
        assert float(target.get_state()["a_out"]) == float(reference.get_state()["a_out"]) == 4.0
        assert float(target.get_state()["c_out"]) == float(reference.get_state()["c_out"]) == 105.0
        # Incomparable observation: the retired variable has no successor.
        assert "x_out" not in target.get_state() and "x_out" not in reference.get_state()

    def test_state_rename_requires_same_units_and_declared_targets(self, tmp_path):
        source = _simulator([_Accumulator("wa", "a_out")], {"a_out": 0.0}, units={"a_out": "kg m-2"})
        source.config["max_provenance_records_per_variable"] = 2
        for _ in range(5):
            source.step(dt=1.0)
        path = tmp_path / "a.pt"
        source.create_checkpoint(path)
        renamed = _simulator([_Accumulator("wa", "a_total")], {"a_total": 0.0}, units={"a_total": "kg m-2"})
        renamed.config["max_provenance_records_per_variable"] = 2
        with pytest.raises(ValueError, match="silently drop saved state variables"):
            renamed.restore_checkpoint(
                path, migration_certificate=_certificate(renamed, path, functor_map={"wa": "wa"}),
            )
        certificate = _certificate(
            renamed, path, functor_map={"wa": "wa"}, state_variable_map={"a_out": "a_total"},
            allowed_contract_paths=("scientific_contract.functors.wa.metadata",),
        )
        receipt = renamed.restore_checkpoint(path, migration_certificate=certificate)
        assert receipt["migration"]["mapped_state_variables"] == [{"saved": "a_out", "target": "a_total"}]
        assert float(renamed.get_state()["a_total"]) == 5.0
        assert _series(renamed, "a_total") == [1.0] * 5
        assert renamed._provenance_evicted_counts == {"a_total": 3}
        assert renamed.get_variable_provenance("a_total", include_retention=True)["evicted_records"] == 3

        wrong_units = _simulator([_Accumulator("wa", "a_total")], {"a_total": 0.0}, units={"a_total": "mol m-2"})
        wrong_units.config["max_provenance_records_per_variable"] = 2
        with pytest.raises(ValueError, match="units differ"):
            wrong_units.restore_checkpoint(
                path, migration_certificate=_certificate(
                    wrong_units, path, functor_map={"wa": "wa"}, state_variable_map={"a_out": "a_total"},
                    allowed_contract_paths=("scientific_contract.functors.wa.metadata",),
                ),
            )


class TestMigrationRejections:
    def _checkpoint(self, tmp_path, steps=2):
        source = _graph_a()
        for _ in range(steps):
            source.step(dt=1.0)
        path = tmp_path / "a.pt"
        source.create_checkpoint(path)
        return path

    def test_undeclared_retired_new_and_double_writer_are_rejected(self, tmp_path):
        path = self._checkpoint(tmp_path)
        target = _graph_b()
        with pytest.raises(ValueError, match="without a target or a retirement"):
            target.restore_checkpoint(path, migration_certificate=_certificate(
                target, path, functor_map={"wa": "wb"}, new_functors=("wc",), retired_state_variables=("x_out",),
            ))
        with pytest.raises(ValueError, match="neither maps nor declares new"):
            target.restore_checkpoint(path, migration_certificate=_certificate(
                target, path, functor_map={"wa": "wb"}, retired_functors=("wx",), retired_state_variables=("x_out",),
            ))
        with pytest.raises(ValueError, match="double writer"):
            GraphMigrationCertificate(
                source_topology_sha256="a" * 64, target_topology_sha256="b" * 64,
                functor_map={"wa": "wb", "wx": "wb"}, replay_mode="switch", new_functor_rng="fresh_substream", forcing_cursor_policy="exact",
            )
        with pytest.raises(ValueError, match="does not carry"):
            target.restore_checkpoint(path, migration_certificate=_certificate(
                target, path, functor_map={"wa": "wb", "ghost": "wc"}, retired_functors=("wx",),
                retired_state_variables=("x_out",),
            ))
        assert target.step_count == 0 and float(target.get_state()["c_out"]) == 100.0

    def test_family_type_and_dependency_changes_are_rejected(self, tmp_path):
        path = self._checkpoint(tmp_path)

        class _OtherType(_Accumulator):
            pass

        other = _simulator([_OtherType("wb", "a_out"), _Accumulator("wx", "x_out", rate=3.0)],
                           {"a_out": 0.0, "x_out": 0.0})
        with pytest.raises(ValueError, match="graph family"):
            other.restore_checkpoint(path, migration_certificate=_certificate(
                other, path, functor_map={"wa": "wb", "wx": "wx"},
            ))
        dependent = _simulator(
            [_Accumulator("wb", "a_out", dependencies=("wx",)), _Accumulator("wx", "x_out", rate=3.0)],
            {"a_out": 0.0, "x_out": 0.0},
        )
        with pytest.raises(ValueError, match="dependency relation"):
            dependent.restore_checkpoint(path, migration_certificate=_certificate(
                dependent, path, functor_map={"wa": "wb", "wx": "wx"},
            ))

    def test_certificate_digests_and_exclusivity(self, tmp_path):
        path = self._checkpoint(tmp_path)
        target = _graph_a()
        digests = compute_restart_topology_digests(target, path)
        with pytest.raises(ValueError, match="source topology digest"):
            target.restore_checkpoint(path, migration_certificate=GraphMigrationCertificate(
                source_topology_sha256="0" * 64, target_topology_sha256=digests["target_topology_sha256"],
                functor_map={"wa": "wa", "wx": "wx"}, replay_mode="switch", new_functor_rng="fresh_substream", forcing_cursor_policy="exact",
            ))
        with pytest.raises(TypeError, match="GraphMigrationCertificate"):
            target.restore_checkpoint(path, migration_certificate={"functor_map": {}})
        from relic.core.simulator.checkpoint import RestartCompatibilityCertificate
        with pytest.raises(ValueError, match="only one certificate"):
            target.restore_checkpoint(
                path,
                compatibility_certificate=RestartCompatibilityCertificate(
                    source_topology_sha256=digests["source_topology_sha256"],
                    target_topology_sha256=digests["target_topology_sha256"], forcing_cursor_policy="exact",
                ),
                migration_certificate=_certificate(target, path, functor_map={"wa": "wa", "wx": "wx"}),
            )
        with pytest.raises(ValueError, match="new_functor_rng"):
            GraphMigrationCertificate(
                source_topology_sha256="a" * 64, target_topology_sha256="b" * 64,
                functor_map={}, new_functor_rng="shared", replay_mode="switch", forcing_cursor_policy="exact",
            )

    def test_counterfactual_replay_requires_complete_history(self, tmp_path):
        source = _graph_a()
        path = tmp_path / "fresh.pt"
        source.create_checkpoint(path)
        target = _graph_a()
        with pytest.raises(ValueError, match="complete counterfactual replay"):
            target.restore_checkpoint(path, migration_certificate=_certificate(
                target, path, functor_map={"wa": "wa", "wx": "wx"}, replay_mode="counterfactual",
            ))
        stepped = self._checkpoint(tmp_path, steps=2)
        replay = _graph_a()
        receipt = replay.restore_checkpoint(stepped, migration_certificate=_certificate(
            replay, stepped, functor_map={"wa": "wa", "wx": "wx"}, replay_mode="counterfactual",
        ))
        assert receipt["migration"]["replay_mode"] == "counterfactual"
        assert _series(replay, "a_out") == [1.0, 1.0]

    @pytest.mark.parametrize("fail_compensation", [False, True])
    def test_partial_apply_failure_rolls_back_the_live_target(self, tmp_path, monkeypatch, fail_compensation):
        path = self._checkpoint(tmp_path)
        target = _graph_b()
        target.step(dt=1.0)
        live_before = {name: float(value) for name, value in target.get_state().items()}
        original = target.functors["wc"].load_state_dict
        load_calls = []

        def failing_load(*args, **kwargs):
            load_calls.append(True)
            if len(load_calls) == 1 or fail_compensation:
                raise RuntimeError("partial apply")
            return original(*args, **kwargs)

        monkeypatch.setattr(target.functors["wc"], "load_state_dict", failing_load)
        error_type = ExceptionGroup if fail_compensation else RuntimeError
        error_message = "compensation was incomplete" if fail_compensation else "partial apply"
        with pytest.raises(error_type, match=error_message) as caught:
            target.restore_checkpoint(path, migration_certificate=_certificate(
                target, path, functor_map={"wa": "wb"}, retired_functors=("wx",), new_functors=("wc",),
                retired_state_variables=("x_out",),
            ))
        assert len(load_calls) == 2
        if fail_compensation:
            assert [str(error) for error in caught.value.exceptions] == ["partial apply", "partial apply"]
        monkeypatch.setattr(target.functors["wc"], "load_state_dict", original)
        assert {name: float(value) for name, value in target.get_state().items()} == live_before
        assert target.step_count == 1
        target.step(dt=1.0)
        assert target.step_count == 2


CLASS_TYPE = "Functor:test.migration_accumulator"


class _PlanAccumulator(_Accumulator):
    """Plan-constructible accumulator with a fixed output port."""

    def __init__(self, name: str = "template", config: dict | None = None) -> None:
        super().__init__(name, "a_out", rate=float((config or {}).get("rate", 1.0)))


def _executor(node_id: str, output: str, initial: float) -> tuple[GraphExecutor, dict]:
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.register_functor_instance(
        _PlanAccumulator(), node_type_id=CLASS_TYPE,
    )
    definition.outputs[output] = PortSpec(
        name=output, standard_name=output, units="1", semantics="state", spatial_kind="grid_cell",
        area_basis="ground_area", temporal_basis="instantaneous", vertical_basis="none",
    )
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={node_id: NodeSpec(class_type=CLASS_TYPE, inputs={}, outputs=[output],
                                 config={"execution_group": "fast"})},
        links=[], metadata={},
    )
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(
        plan, registry=registry,
        simulator_config={"device": "cpu", "time_step": 1.0, "adaptive_dt_enabled": False, "output_frequency": 1},
    )
    return executor, {output: torch.tensor(initial)}


def test_graph_executor_migrates_between_plans(tmp_path):
    producer, initial = _executor("old_node", "a_out", 0.0)
    producer.execute(initial, n_steps=2, dt=1.0)
    path = tmp_path / "old.pt"
    producer.create_checkpoint(path)

    consumer, new_initial = _executor("new_node", "a_out", 0.0)
    consumer.execute(new_initial, n_steps=0, dt=1.0)
    certificate = _certificate(consumer._simulator, path, functor_map={"old_node": "new_node"})
    final = consumer.migrate_from_checkpoint(new_initial, path, certificate, n_steps=1, dt=1.0)
    assert float(final["a_out"]) == 3.0
    assert consumer._simulator.step_count == 3
    with pytest.raises(ValueError, match="does not match the active simulator"):
        consumer.execute_from_checkpoint(new_initial, path, n_steps=1, dt=1.0)
