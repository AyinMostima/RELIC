import torch

from relic.core.simulator.execution_loop import compute_splitting_sensitivity
from relic.core.state import StateManager, UpdateItem, UpdateMode


class _DummyStateManager:
    def __init__(self, initial_state):
        self._state_variables = initial_state


class _DummySimulator:
    def __init__(self, commutative: bool):
        self.state_manager = _DummyStateManager({"x": torch.tensor(1.0)})
        self.commutative = commutative
        self.current_time = 7200.0
        self.start_times = []
        self._step_update_sources = {}
        self._step_functor_increments = {}
        self._step_provenance_detail = {}

    def _get_current_state(self):
        return self.state_manager._state_variables

    def _execute_group(self, order, dt, *, start_time):
        self.start_times.append(start_time)
        x_value = self.state_manager._state_variables["x"]
        for name in order:
            if name == "A":
                x_value = x_value + 1.0
            elif name == "B":
                if self.commutative:
                    x_value = x_value + 2.0
                else:
                    x_value = x_value * 2.0
        self.state_manager._state_variables["x"] = x_value
        return {"x": x_value}


class _Recorder:
    def __init__(self) -> None:
        self.records = []

    def record_state_update(self, *args, **kwargs) -> None:
        self.records.append((args, kwargs))


class _TransactionalSimulator:
    def __init__(self) -> None:
        self.state_manager = StateManager()
        self.state_manager.initialize_state({"x": torch.tensor(1.0)})
        self.recorder = _Recorder()
        self.state_manager.set_provenance_recorder(self.recorder)
        self.current_time = 0.0
        self._step_update_sources = {"seed": ("existing", UpdateMode.REPLACE)}
        self._step_functor_increments = {"seed": {"existing": torch.tensor(0.0)}}
        self._step_provenance_detail = {
            "seed": [("existing", UpdateMode.REPLACE, 0)]
        }

    def _get_current_state(self):
        return self.state_manager.get_state(clone=True)

    def _execute_group(self, order, dt, *, start_time):
        del dt, start_time
        pre_value = self.state_manager.get_state("x")
        post_value = pre_value
        for name in order:
            if name == "A":
                post_value = post_value + 1.0
            elif name == "B":
                post_value = post_value * 2.0
        writer = "_".join(order)
        self._step_update_sources["x"] = (writer, UpdateMode.REPLACE)
        self._step_functor_increments["x"] = {writer: post_value - pre_value}
        self._step_provenance_detail["x"] = [(writer, UpdateMode.REPLACE, 0)]
        self.state_manager.ingest_module_diagnostics(writer, {"trial": 1.0})
        self.state_manager.apply_updates(
            {
                "x": UpdateItem(
                    value=post_value,
                    mode=UpdateMode.REPLACE,
                    per_timestep=True,
                    source_functor=writer,
                )
            },
            update_sources={"x": writer},
            update_increments={"x": {writer: post_value - pre_value}},
        )
        return {"x": post_value}


def test_permutation_count_defaults_to_the_parameter_database(restore_global_parameter_manager) -> None:
    """Without n_permutations the sample size is
    computation.diagnostics.splitting_sensitivity.max_permutations."""
    import inspect

    assert inspect.signature(compute_splitting_sensitivity).parameters["n_permutations"].default is None
    restore_global_parameter_manager.set_batch(
        {"computation.diagnostics.splitting_sensitivity.max_permutations": 1}, validate=True, notify=False,
    )
    simulator = _DummySimulator(commutative=False)
    result = compute_splitting_sensitivity(
        simulator=simulator, state={"x": torch.tensor(1.0)}, dt=1.0, group_order=["A", "B"],
    )
    assert result["permutations_tested"] == 1


def test_commutative_operators_zero_sensitivity() -> None:
    simulator = _DummySimulator(commutative=True)
    state = {"x": torch.tensor(1.0)}
    result = compute_splitting_sensitivity(
        simulator=simulator,
        state=state,
        dt=1.0,
        group_order=["A", "B"],
        n_permutations=2,
    )
    assert result["commutativity_violation"] <= 1e-12


def test_noncommutative_detection() -> None:
    simulator = _DummySimulator(commutative=False)
    state = {"x": torch.tensor(1.0)}
    result = compute_splitting_sensitivity(
        simulator=simulator,
        state=state,
        dt=1.0,
        group_order=["A", "B"],
        n_permutations=2,
    )
    assert result["commutativity_violation"] > 0.0
    assert result["permutations_tested"] == 2
    # Every permutation trial accounts for [current_time, current_time + dt].
    assert simulator.start_times == [7200.0, 7200.0]


def test_state_not_modified() -> None:
    simulator = _DummySimulator(commutative=False)
    original_state = simulator._get_current_state()["x"].clone()
    result = compute_splitting_sensitivity(
        simulator=simulator,
        state={"x": original_state.clone()},
        dt=1.0,
        group_order=["A", "B"],
        n_permutations=2,
    )
    assert result["permutations_tested"] == 2
    assert torch.isclose(simulator._get_current_state()["x"], original_state)


def test_integer_state_tensors_are_ignored() -> None:
    simulator = _DummySimulator(commutative=False)
    simulator.state_manager._state_variables["step"] = torch.tensor(1, dtype=torch.int64)
    result = compute_splitting_sensitivity(
        simulator=simulator,
        state={
            "x": torch.tensor(1.0),
            "step": torch.tensor(1, dtype=torch.int64),
        },
        dt=1.0,
        group_order=["A", "B"],
        n_permutations=2,
    )
    assert result["permutations_tested"] == 2
    assert "x" in result["per_variable_sensitivity"]
    assert "step" not in result["per_variable_sensitivity"]


def test_trials_restore_all_state_manager_and_provenance_side_effects() -> None:
    simulator = _TransactionalSimulator()
    manager_before = simulator.state_manager.checkpoint()
    sources_before = dict(simulator._step_update_sources)
    increments_before = {
        key: dict(value) for key, value in simulator._step_functor_increments.items()
    }
    provenance_before = {
        key: list(value) for key, value in simulator._step_provenance_detail.items()
    }
    provenance_owner = simulator._step_provenance_detail
    seed_records = provenance_owner["seed"]

    result = compute_splitting_sensitivity(
        simulator=simulator,
        state={"x": torch.tensor(1.0)},
        dt=1.0,
        group_order=["A", "B"],
        n_permutations=2,
    )

    assert result["permutations_tested"] == 2
    assert simulator.state_manager.get_diagnostics()["update_counter"] == 0
    assert simulator.state_manager.conservation_history == manager_before["conservation_history"]
    assert simulator.state_manager._current_diagnostics == manager_before["diagnostics"]
    assert simulator.recorder.records == []
    assert simulator._step_update_sources == sources_before
    assert simulator._step_functor_increments == increments_before
    assert simulator._step_provenance_detail == provenance_before
    # The recorded history is restored as the same objects, not as a clone of it.
    assert simulator._step_provenance_detail is provenance_owner
    assert provenance_owner["seed"] is seed_records
