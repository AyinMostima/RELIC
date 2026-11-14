import torch

from relic.core.functor import ExecutionPhase, FunctorMetadata, ProcessCategory
from relic.core.functor import TransportMechanismFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateMode


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'test_moisture': 'kg m-2',
    'test_rate': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class _ReplaceSoilMoistureFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str = "replace_soil_moisture"):
        metadata = FunctorMetadata(
            name=name,
            version="0.0.0",
            description="Test functor that returns an absolute test state.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="test_moisture_new = 0.1",
            references=["test"],
            input_variables=["test_moisture"],
            output_variables=["test_moisture"],
        )
        super().__init__(name=name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {
            "test_moisture": self.wrap_update(
                torch.tensor(0.1, dtype=torch.float32), mode=UpdateMode.REPLACE
            )
        }


class _DiagnosticSoilMoistureFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str = "diagnostic_soil_moisture"):
        metadata = FunctorMetadata(
            name=name,
            version="0.0.0",
            description="Test functor that emits a diagnostic metric alongside a test state update.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="test_moisture_new = test_moisture + 0.05",
            references=["test"],
            input_variables=["test_moisture"],
            output_variables=["test_moisture"],
        )
        super().__init__(name=name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {
            "test_moisture": self.wrap_update(
                torch.tensor(0.05, dtype=torch.float32), mode=UpdateMode.ACCUMULATE
            ),
            "diagnostics.test_metric": torch.tensor(1.0, dtype=torch.float32),
        }


class _RawSoilMoistureFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str = "raw_soil_moisture"):
        metadata = FunctorMetadata(
            name=name,
            version="0.0.0",
            description="Test functor that returns a raw test state value.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="test_moisture_new = 0.1",
            references=["test"],
            input_variables=["test_moisture"],
            output_variables=["test_moisture"],
        )
        super().__init__(name=name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {
            "test_moisture": torch.tensor(0.1, dtype=torch.float32)
        }


class _RawRateLikeOutputFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str = "raw_rate_like_output"):
        metadata = FunctorMetadata(
            name=name,
            version="0.0.0",
            description="Test functor that returns a raw rate-like test value.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="test_rate = 2.0",
            references=["test"],
            input_variables=[],
            output_variables=["test_rate"],
        )
        super().__init__(name=name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {
            "test_rate": torch.tensor(2.0, dtype=torch.float32)
        }


def test_simulator_applies_replace_mode_for_absolute_state_updates():
    """
    Regression: absolute state outputs must not be treated as ACCUMULATE deltas.

    If a functor returns an absolute state value without REPLACE mode,
    the simulator would treat it as a delta and incorrectly add it to the state.
    """
    sim = _declared_simulator({"max_time": 2.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"test_moisture": torch.tensor(0.3, dtype=torch.float32)})

    sim.register_functor(_ReplaceSoilMoistureFunctor())
    sim.step(dt=1.0)

    test_moisture = sim.state_manager.get_state("test_moisture")
    assert torch.isclose(test_moisture, torch.tensor(0.1, dtype=torch.float32))


def test_simulator_ignores_non_state_updates():
    """
    Regression: non-state outputs should not be persisted into StateManager.
    """
    sim = _declared_simulator({"max_time": 2.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"test_moisture": torch.tensor(0.3, dtype=torch.float32)})

    sim.register_functor(_DiagnosticSoilMoistureFunctor())
    updates = sim.step(dt=1.0)

    test_moisture = sim.state_manager.get_state("test_moisture")
    assert torch.isclose(test_moisture, torch.tensor(0.35, dtype=torch.float32))
    assert "diagnostics.test_metric" in updates
    assert sim.state_manager.get_state("diagnostics.test_metric") is None


def test_simulator_run_specs_default_replace_for_raw_updates(tmp_path):
    irf_path = tmp_path / "irf.yaml"
    irf_path.write_text(
        "schema_version: \"1.0.0\"\nrun_specs:\n  default_mode: \"replace\"\n",
        encoding="utf-8",
    )
    sim = _declared_simulator({
        "max_time": 2.0,
        "time_step": 1.0,
        "conservation_checks": False,
        "irf_specs_path": str(irf_path),
    })
    sim.initialize_state({"test_moisture": torch.tensor(0.3, dtype=torch.float32)})
    sim.register_functor(_RawSoilMoistureFunctor())
    sim.step(dt=1.0)

    test_moisture = sim.state_manager.get_state("test_moisture")
    assert torch.isclose(test_moisture, torch.tensor(0.1, dtype=torch.float32))


def test_simulator_defaults_replace_for_rate_like_outputs():
    sim = _declared_simulator({"max_time": 3.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"test_rate": torch.tensor(1.0, dtype=torch.float32)})
    sim.register_functor(_RawRateLikeOutputFunctor())

    sim.step(dt=1.0)
    first_rate = sim.state_manager.get_state("test_rate")
    assert torch.isclose(first_rate, torch.tensor(2.0, dtype=torch.float32))

    sim.step(dt=1.0)
    second_rate = sim.state_manager.get_state("test_rate")
    assert torch.isclose(second_rate, torch.tensor(2.0, dtype=torch.float32))

def test_unreadable_irf_specs_fail_even_without_strict_mode(tmp_path):
    """irf_strict governs validation findings, not whether configured specs are read."""
    import pytest

    sim = _declared_simulator({
        "max_time": 2.0,
        "time_step": 1.0,
        "conservation_checks": False,
        "irf_specs_path": str(tmp_path / "missing_irf.yaml"),
        "irf_strict": False,
    })
    with pytest.raises((OSError, ValueError)):
        sim.initialize_state({"test_moisture": torch.tensor(0.3, dtype=torch.float32)})


def test_run_spec_resolution_failure_is_not_replaced_by_replace(tmp_path):
    import pytest

    irf_path = tmp_path / "irf.yaml"
    irf_path.write_text(
        "schema_version: \"1.0.0\"\nrun_specs:\n  default_mode: \"replace\"\n",
        encoding="utf-8",
    )
    sim = _declared_simulator({
        "max_time": 2.0,
        "time_step": 1.0,
        "conservation_checks": False,
        "irf_specs_path": str(irf_path),
    })

    def _broken(name, mode):
        raise KeyError(f"no run spec for {name}")

    from types import SimpleNamespace

    sim.run_specs = SimpleNamespace(resolve_update_mode=_broken, strict=False)
    with pytest.raises(KeyError, match="no run spec"):
        sim._resolve_update_mode("test_moisture", UpdateMode.ACCUMULATE)
