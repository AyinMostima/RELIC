"""
Integration tests for Simulator <-> StateManager transactional updates.

ASCII-only comments and prints.
"""
import warnings

import pytest
import torch

from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionError,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
)
from relic.core.simulator import RELICSimulator
import relic.scientific as scientific_package
from relic.scientific import register_functor
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.hydrosphere.soil_moisture.richards import (
    RichardsSoilMoisture,
)
from relic.scientific.hydrosphere.transport.capillary_rise import (
    CapillaryRiseTransport,
)
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport,
)
from relic.scientific.pedosphere.transport.SoilWaterTransportFunctor import (
    SoilWaterTransportFunctor,
)


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'biosphere.extra_carbon_pool': 'kg C m-2',
    'medium_marker': 'dimensionless',
    'pool': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class PlusOneFunctor(BiologicalProcessFunctor):
    """
    Simple additive functor for testing.

    References: No physical model (algebraic test process).
    Formula: pool(t+dt) = pool(t) + 1.0
    Parameters: None.
    """
    def __init__(self):
        meta = FunctorMetadata(
            name="plus_one",
            description="Add 1.0 to pool",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="pool += 1.0",
            references=["None (algebraic test)"],
            input_variables=["pool"],
            output_variables=["pool"],
        )
        super().__init__(name="plus_one", config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {"pool": torch.tensor(1.0)}


class MinusTwoFunctor(BiologicalProcessFunctor):
    """
    Simple subtractive functor for rollback testing.

    References: No physical model (algebraic test process).
    Formula: pool(t+dt) = pool(t) - 2.0 (may break non-negative bound)
    Parameters: None.
    """
    def __init__(self):
        meta = FunctorMetadata(
            name="minus_two",
            description="Subtract 2.0 from pool",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="pool -= 2.0",
            references=["None (algebraic test)"],
            input_variables=["pool"],
            output_variables=["pool"],
        )
        super().__init__(name="minus_two", config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {"pool": torch.tensor(-2.0)}


class CarbonLedgerFunctor(BiologicalProcessFunctor):
    """Functor that declares carbon conservation metadata for ledger propagation."""

    def __init__(self):
        meta = FunctorMetadata(
            name="carbon_ledgers",
            description="Emit carbon diagnostics for ledger tracking",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="dC/dt = 0",
            references=["Ledger smoke test"],
            conservation_laws=["carbon"],
            output_variables=["biosphere.extra_carbon_pool"],
        )
        super().__init__(name="carbon_ledgers", config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {"biosphere.extra_carbon_pool": torch.tensor(0.0)}


class DeltaRateFunctor(BiologicalProcessFunctor):
    """
    Functor that emits delta_ updates to verify delta mapping and dt integration.

    References: No physical model (algebraic test process).
    Formula: delta_pool_rate = 1.5 (per second), pool += delta_pool_rate * dt
    Parameters: None.
    """
    def __init__(self):
        meta = FunctorMetadata(
            name="delta_rate",
            description="Emit delta_ updates for pool",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="pool += rate * dt",
            references=["None (algebraic test)"],
            output_variables=["pool"],
        )
        super().__init__(name="delta_rate", config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {"delta_pool": torch.tensor(1.5)}


class DummydefaultFunctor(BiologicalProcessFunctor):
    """Minimal default functor for registry resolution tests."""
    FUNCTOR_NAME = "dummy_default"

    def __init__(self, name: str = "dummy_default") -> None:
        meta = FunctorMetadata(
            name=name,
            description="Registry default test",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="no-op",
            references=["None (test stub)"],
            input_variables=["pool"],
        )
        super().__init__(name=name, config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {}


class RequestedNameProbe(BiologicalProcessFunctor):
    """Bootstrap probe without FUNCTOR_NAME, requested through a registry alias."""

    def __init__(self, name: str) -> None:
        meta = FunctorMetadata(
            name=name,
            description="Bootstrap requested-name probe",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="no-op",
            references=["None (test stub)"],
            input_variables=["pool"],
        )
        super().__init__(name=name, config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {}


class FailingMediumFunctor(BiologicalProcessFunctor):
    """Fail after a fast group has committed its candidate state."""

    def __init__(self) -> None:
        meta = FunctorMetadata(
            name="failing_medium",
            description="Raise during medium-group execution",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="raise RuntimeError before state update",
            references=["None (transaction test)"],
            input_variables=["medium_marker"],
            output_variables=["medium_marker"],
        )
        super().__init__(name="failing_medium", config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        raise RuntimeError("intentional medium-group failure")


class InMemoryProvenanceRecorder:
    """Capture recorder calls without external file I/O."""

    def __init__(self) -> None:
        self.records = []

    def record_state_update(self, *args, **kwargs) -> None:
        self.records.append((args, kwargs))


def test_transaction_success_and_state_update():
    sim = _declared_simulator({"time_step": 1.0, "max_time": 10.0})
    sim.initialize_state({"pool": torch.tensor(0.0)})
    sim.register_functor(PlusOneFunctor(), execution_group="fast")

    # Run one step
    sim.run(start_time=0.0, end_time=1.0, dt=1.0)
    # After one step, pool should be 1.0 via StateManager updates
    current_state = sim.state_manager.get_state()
    assert float(current_state["pool"]) == pytest.approx(1.0, rel=1e-6)


def test_transaction_clamps_to_bounds_when_functor_exceeds_limits():
    from relic.core.dependency_graph import DependencyGraphResolver
    from relic.core.state import StateManager
    from relic.core.state.types import StateVariableSpec

    sim = _declared_simulator({"time_step": 1.0, "max_time": 10.0})
    # The test declares the bounds it exercises; no bound is inferred from a name.
    sim.initialize(
        StateManager({"pool": StateVariableSpec(name="pool", units="dimensionless", bounds=(0.0, 1000.0))}),
        DependencyGraphResolver(),
    )
    # Initial pool near lower bound so minus_two attempts to push it negative
    sim.initialize_state({"pool": torch.tensor(0.5)})
    sim.register_functor(MinusTwoFunctor(), execution_group="fast")

    # Stepping should complete and clamp pool to its lower bound (0.0)
    sim.step(dt=1.0)

    current_state = sim.state_manager.get_state()
    assert float(current_state["pool"]) == pytest.approx(0.0, rel=1e-6)


def test_metadata_law_does_not_create_uncontracted_ledger_member():
    sim = _declared_simulator({"time_step": 1.0, "max_time": 10.0})
    sim.initialize_state({"biosphere.extra_carbon_pool": torch.tensor(0.0)})
    sim.register_functor(CarbonLedgerFunctor(), execution_group="fast")

    ledger = sim.state_manager.get_conservation_ledger()
    assert "carbon" in ledger
    assert "biosphere.extra_carbon_pool" not in ledger["carbon"]


def test_delta_updates_flow_to_state_manager_with_dt():
    sim = _declared_simulator({"time_step": 2.0, "max_time": 10.0})
    sim.initialize_state({"pool": torch.tensor(0.0)})
    sim.register_functor(DeltaRateFunctor(), execution_group="fast")

    sim.step(dt=2.0)

    current_state = sim.state_manager.get_state()
    assert float(current_state["pool"]) == pytest.approx(3.0, rel=1e-6)


class LeafGrowthBoundaryFluxFunctor(BiologicalProcessFunctor):
    """
    Writes the declared boundary flux growth_rate_leaf and the leaf carbon it feeds.

    References: RELIC SCM contracts growth_rate_leaf.yaml (kind flux, allowed_in
    biomass_allocation) and leaf_biomass_c.yaml (carbon.total member).
    Formula: growth_rate_leaf = F_leaf (REPLACE rate, per_timestep=False);
    delta leaf_biomass_c = F_leaf * dt (ACCUMULATE amount), so the carbon budget closes.
    Parameters: F_leaf = 2e-8 kg C m-2 s-1 (algebraic test value), non-negative into
    the plant; dt [s] is the call interval.
    """
    def __init__(self):
        meta = FunctorMetadata(
            name="biomass_allocation",
            description="Write one leaf growth boundary rate and its leaf carbon increment",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="growth_rate_leaf = F_leaf; delta leaf_biomass_c = F_leaf * dt",
            references=[
                "relic/contracts/variables/biosphere/growth_rate_leaf.yaml",
                "relic/contracts/variables/pedosphere/leaf_biomass_c.yaml",
            ],
            input_variables=["leaf_biomass_c"],
            output_variables=["growth_rate_leaf", "leaf_biomass_c"],
        )
        super().__init__(name="biomass_allocation", config={}, metadata=meta)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        from relic.core.state import UpdateItem, UpdateMode

        rate = torch.tensor([2.0e-8])
        return {
            "growth_rate_leaf": UpdateItem(
                rate,
                UpdateMode.REPLACE,
                unit="kg C m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "leaf_biomass_c": UpdateItem(
                rate * float(dt),
                UpdateMode.ACCUMULATE,
                unit="kg C m-2",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            ),
        }


@pytest.mark.parametrize("conservation_checks", [True, False], ids=["checks_on", "checks_off"])
def test_accepted_step_with_a_boundary_flux_commits_with_conservation_checks(conservation_checks):
    """A step writing a declared boundary rate commits whether or not the checks run.

    The step loop builds no conservation report from its returned mapping of
    payloads: that mapping carries no UpdateItems and no commit interval, which
    the report requires for a boundary flux (PROFILE-001 regression).
    """
    sim = _declared_simulator(
        {"time_step": 1.0, "max_time": 10.0, "conservation_checks": conservation_checks}
    )
    sim.initialize_state(
        {"growth_rate_leaf": torch.tensor([0.0]), "leaf_biomass_c": torch.tensor([0.1])}
    )
    sim.register_functor(LeafGrowthBoundaryFluxFunctor(), execution_group="fast")

    sim.step(dt=1.0)

    assert sim.state_manager.get_state("growth_rate_leaf").tolist() == pytest.approx([2.0e-8])
    assert sim.state_manager.get_state("leaf_biomass_c").tolist() == pytest.approx([0.1 + 2.0e-8])
    assert sim.step_count == 1


def test_default_functor_registry_resolution() -> None:
    register_functor(DummydefaultFunctor)

    sim = _declared_simulator(
        {
            "time_step": 1.0,
            "enable_bootstrap_modules": True,
            "bootstrap_functor_name": "dummy_default",
        }
    )
    sim.initialize_state({"pool": torch.tensor(0.0)})

    assert "dummy_default" in sim.functors


def _bootstrap_simulator(**settings) -> RELICSimulator:
    config = {"time_step": 1.0, "enable_bootstrap_modules": True}
    config.update(settings)
    return _declared_simulator(config)


def test_default_bootstrap_name_is_a_declared_parameter() -> None:
    from relic.parameters import get_global_manager

    key = "computation.simulator_bootstrap.default_functor_name"
    spec = get_global_manager().get_parameter(key)
    assert spec is not None and spec.description
    name = get_global_manager().get(key, expected_type=str, required=True)
    assert scientific_package.get_functor_path(name)


def test_default_bootstrap_name_is_read_from_the_parameter_database(monkeypatch) -> None:
    from relic.parameters import get_global_manager

    probe_path = f"{RequestedNameProbe.__module__}:{RequestedNameProbe.__name__}"
    monkeypatch.setitem(scientific_package.FUNCTOR_REGISTRY, "requested_name_probe", probe_path)
    manager = get_global_manager()
    original_get = manager.get

    def _get(key, *args, **kwargs):
        if key == "computation.simulator_bootstrap.default_functor_name":
            return "requested_name_probe"
        return original_get(key, *args, **kwargs)

    monkeypatch.setattr(manager, "get", _get)
    sim = _bootstrap_simulator()

    sim.initialize_state({"pool": torch.tensor(0.0)})

    assert list(sim.functors) == ["requested_name_probe"]


def test_unresolvable_default_bootstrap_name_raises_with_the_name() -> None:
    sim = _bootstrap_simulator(bootstrap_functor_name="no_such_bootstrap_probe")

    with pytest.raises(KeyError, match="no_such_bootstrap_probe"):
        sim.initialize_state({"pool": torch.tensor(0.0)})


def test_explicitly_empty_bootstrap_request_registers_nothing(monkeypatch) -> None:
    probe_path = f"{RequestedNameProbe.__module__}:{RequestedNameProbe.__name__}"
    monkeypatch.setitem(scientific_package.FUNCTOR_REGISTRY, "requested_name_probe", probe_path)
    sim = _bootstrap_simulator(bootstrap_functors=[], bootstrap_functor_name="requested_name_probe")

    sim.initialize_state({"pool": torch.tensor(0.0)})

    assert len(sim.functors) == 0


def test_default_bootstrap_registers_under_the_requested_name(monkeypatch) -> None:
    probe_path = f"{RequestedNameProbe.__module__}:{RequestedNameProbe.__name__}"
    monkeypatch.setitem(scientific_package.FUNCTOR_REGISTRY, "requested_name_probe", probe_path)
    sim = _bootstrap_simulator(bootstrap_functor_name="requested_name_probe")

    sim.initialize_state({"pool": torch.tensor(0.0)})

    assert list(sim.functors) == ["requested_name_probe"]


def test_late_group_failure_rolls_back_the_entire_step() -> None:
    sim = _declared_simulator(
        {
            "time_step": 1.0,
            "max_time": 10.0,
            "medium_interval_seconds": 1.0,
        }
    )
    sim.initialize_state(
        {
            "pool": torch.tensor(0.0),
            "medium_marker": torch.tensor(0.0),
        }
    )
    recorder = InMemoryProvenanceRecorder()
    sim.state_manager.set_provenance_recorder(recorder)
    sim.register_functor(PlusOneFunctor(), execution_group="fast")
    sim.register_functor(FailingMediumFunctor(), execution_group="medium")

    state_before = sim.state_manager.checkpoint()
    runtime_before = {
        "current_time": sim.current_time,
        "step_count": sim.step_count,
        "previous_medium_index": sim._previous_medium_index,
        "last_medium_execution_time": sim._last_medium_execution_time,
        "bootstrap_groups": set(sim._bootstrap_groups),
        "time_series": dict(sim.time_series),
    }

    with pytest.raises(ExecutionError, match="failing_medium"):
        sim.step(dt=1.0)

    restored_pool = sim.state_manager.get_state("pool")
    assert torch.equal(restored_pool, torch.zeros_like(restored_pool))
    assert sim.state_manager.get_diagnostics()["update_counter"] == 0
    assert sim.state_manager.conservation_history == state_before["conservation_history"]
    assert sim.state_manager._current_diagnostics == state_before["diagnostics"]
    assert sim.state_manager._enable_auto_report is state_before["auto_report_enabled"]
    assert recorder.records == []
    assert sim.current_time == runtime_before["current_time"]
    assert sim.step_count == runtime_before["step_count"]
    assert sim._previous_medium_index == runtime_before["previous_medium_index"]
    assert sim._last_medium_execution_time == runtime_before["last_medium_execution_time"]
    assert sim._bootstrap_groups == runtime_before["bootstrap_groups"]
    assert sim.time_series == runtime_before["time_series"]


def test_bucket_state_manager_checkpoint_restart_is_numerically_identical() -> None:
    """Bucket restart must reproduce an uninterrupted two-step water balance.

    References: Manabe (1969), single-layer bucket water balance.
    Mathematical expression: Delta(theta) = dt * (P - ET - Q - D) / (rho_w Z).
    Parameter meanings: theta is volumetric storage, P/ET/Q/D are boundary
    mass fluxes, rho_w is water density, and Z is bucket depth.
    """
    dt = 1800.0
    initial_state = {
        "soil_moisture": torch.tensor(0.30, dtype=torch.float64),
        "precipitation_flux": torch.tensor(0.001, dtype=torch.float64),
        "evapotranspiration_flux": torch.tensor(0.0001, dtype=torch.float64),
    }

    def build_simulator() -> RELICSimulator:
        simulator = _declared_simulator(
            {
                "max_time": 2.0 * dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": True,
                "strict_dependencies": True,
            }
        )
        simulator.initialize_state(
            {key: value.clone() for key, value in initial_state.items()}
        )
        simulator.register_functor(
            BucketSoilMoisture(name="bucket_soil_moisture"),
            execution_group="fast",
        )
        return simulator

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        continuous = build_simulator()
        continuous.step(dt=dt)
        continuous.step(dt=dt)

        first_leg = build_simulator()
        first_leg.step(dt=dt)
        checkpoint = first_leg.state_manager.checkpoint()

        restarted = build_simulator()
        restarted.state_manager.restore(checkpoint)
        restarted.current_time = first_leg.current_time
        restarted.step_count = first_leg.step_count
        restarted.step(dt=dt)

    torch.testing.assert_close(
        restarted.state_manager.get_state("soil_moisture"),
        continuous.state_manager.get_state("soil_moisture"),
    )
    assert not hasattr(restarted.functors["bucket_soil_moisture"], "_drainage_factor")
    assert caught == []


def test_soil_water_transport_checkpoint_restart_is_numerically_identical() -> None:
    """Alternative soil-water transport restart reproduces its water ledger.

    References: Richards (1931); Celia et al. (1990).
    Mathematical expression: R_w = sum_i(Delta theta_i dz)
        - q_surface dt + q_bottom dt.
    Parameter meanings: theta is volumetric water content, dz is layer spacing,
        and q_surface/q_bottom are open water-boundary fluxes.
    """

    dt = 60.0
    probe = SoilWaterTransportFunctor(device=torch.device("cpu"))
    lower = float(torch.as_tensor(probe.wilting_point))
    upper = float(torch.as_tensor(probe.porosity))
    initial_theta = torch.full(
        (5,),
        lower + 0.5 * (upper - lower),
        dtype=torch.float64,
    )
    initial_state = {
        "soil_moisture": initial_theta,
        "infiltration_flux": torch.tensor(0.0, dtype=torch.float64),
        "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float64),
    }

    def build_simulator() -> RELICSimulator:
        simulator = _declared_simulator(
            {
                "max_time": 2.0 * dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": True,
                "strict_dependencies": True,
            }
        )
        simulator.initialize_state(
            {key: value.clone() for key, value in initial_state.items()}
        )
        simulator.register_functor(
            SoilWaterTransportFunctor(device=torch.device("cpu")),
            execution_group="fast",
        )
        return simulator

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        continuous = build_simulator()
        continuous.step(dt=dt)
        continuous.step(dt=dt)

        first_leg = build_simulator()
        first_leg.step(dt=dt)
        checkpoint = first_leg.state_manager.checkpoint()

        restarted = build_simulator()
        restarted.state_manager.restore(checkpoint)
        restarted.current_time = first_leg.current_time
        restarted.step_count = first_leg.step_count
        restarted.step(dt=dt)

    torch.testing.assert_close(
        restarted.state_manager.get_state("soil_moisture"),
        continuous.state_manager.get_state("soil_moisture"),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restarted.state_manager.get_state(
            "diagnostics.soil_water_transport.water_balance_residual"
        ),
        continuous.state_manager.get_state(
            "diagnostics.soil_water_transport.water_balance_residual"
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert restarted.current_time == continuous.current_time == 2.0 * dt
    assert restarted.step_count == continuous.step_count == 2
    provenance = restarted.get_variable_provenance("soil_moisture")
    assert {entry["functor"] for entry in provenance} == {
        "soil_water_transport"
    }
    assert {entry["role"] for entry in provenance} == {
        "writer",
        "final_writer",
    }
    assert {entry["mode"] for entry in provenance} == {"UpdateMode.REPLACE"}
    assert restarted.get_effective_device("soil_water_transport").type == "cpu"
    assert "_last_richards_iterations" not in vars(
        restarted.functors["soil_water_transport"]
    )
    assert caught == []


def test_richards_checkpoint_restart_and_writer_provenance_are_exact() -> None:
    """Richards restart reproduces state, boundary fluxes, and writer evidence.

    References: Richards (1931); Celia et al. (1990).
    Mathematical expression: Delta W = (q_top - q_bottom) dt, with
        q_bottom = K_bottom for the downward-positive free-drainage boundary.
    Parameter meanings: W is column water depth, q_top is the net atmospheric
        input, q_bottom is drainage, and dt is the model step.
    """

    dt = 60.0
    initial_state = {
        "soil_moisture": torch.full((10,), 0.30, dtype=torch.float64),
        "infiltration_flux": torch.tensor(0.0, dtype=torch.float64),
        "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float64),
    }

    def build_simulator() -> RELICSimulator:
        simulator = _declared_simulator(
            {
                "max_time": 2.0 * dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": True,
                "strict_dependencies": False,
            }
        )
        simulator.initialize_state(
            {key: value.clone() for key, value in initial_state.items()}
        )
        model = RichardsSoilMoisture(name="richards_soil_moisture")
        model.set_boundary_conditions(
            top={"type": "atmospheric"},
            bottom={"type": "free_drainage"},
        )
        simulator.register_functor(model, execution_group="fast")
        return simulator

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        continuous = build_simulator()
        continuous.step(dt=dt)
        continuous_updates = continuous.step(dt=dt)

        first_leg = build_simulator()
        first_leg.step(dt=dt)
        checkpoint = first_leg.state_manager.checkpoint()

        restarted = build_simulator()
        restarted.state_manager.restore(checkpoint)
        restarted.current_time = first_leg.current_time
        restarted.step_count = first_leg.step_count
        restarted_updates = restarted.step(dt=dt)

    for key in (
        "soil_moisture",
        "soil_matric_potential",
        "drainage_flux",
    ):
        torch.testing.assert_close(
            restarted.state_manager.get_state(key),
            continuous.state_manager.get_state(key),
            rtol=0.0,
            atol=0.0,
        )

    assert restarted.current_time == continuous.current_time == 2.0 * dt
    assert restarted.step_count == continuous.step_count == 2
    flux_profile_key = (
        "diagnostics.soil_moisture_richards.water_flux_profile"
    )
    torch.testing.assert_close(
        restarted_updates[flux_profile_key],
        continuous_updates[flux_profile_key],
        rtol=0.0,
        atol=0.0,
    )
    mass_error = restarted_updates[
        "diagnostics.soil_moisture_richards.mass_conservation_error"
    ]
    assert mass_error <= (
        restarted.functors["richards_soil_moisture"].mass_conservation_tolerance
    )
    for key in ("soil_moisture", "soil_matric_potential", "drainage_flux"):
        provenance = restarted.get_variable_provenance(key)
        assert {entry["functor"] for entry in provenance} == {
            "richards_soil_moisture"
        }
        assert {entry["role"] for entry in provenance} == {
            "writer",
            "final_writer",
        }
        assert {entry["mode"] for entry in provenance} == {
            "UpdateMode.REPLACE"
        }
    assert restarted.get_effective_device("richards_soil_moisture").type == "cpu"
    assert caught == []


def test_capillary_rise_to_richards_restart_and_provenance_are_exact() -> None:
    """Close the capillary, Richards, and groundwater state chain.

    References: Gardner (1958); Richards (1931); Celia et al. (1990).
    Mathematical expression: dM_soil/dt = F_cap - F_drain and
        dM_gw/dt = F_drain - F_cap - F_base for zero external forcing.
    Parameter meanings: M_soil and M_gw are water mass storages; F_cap is the
        upward internal transfer, F_drain is the downward internal transfer,
        and F_base is the open groundwater boundary export.
    """

    dt = 60.0
    initial_state = {
        "water_table_depth": torch.tensor(1.5, dtype=torch.float64),
        "groundwater_storage": torch.tensor(200.0, dtype=torch.float64),
        "soil_moisture": torch.full((10,), 0.25, dtype=torch.float64),
        "infiltration_flux": torch.tensor(0.0, dtype=torch.float64),
        "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float64),
        "lateral_inflow": torch.tensor(0.0, dtype=torch.float64),
        "groundwater_pumping": torch.tensor(0.0, dtype=torch.float64),
    }

    def build_simulator() -> RELICSimulator:
        simulator = _declared_simulator(
            {
                "max_time": 2.0 * dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": True,
                "strict_dependencies": True,
            }
        )
        simulator.initialize_state(
            {key: value.clone() for key, value in initial_state.items()}
        )
        capillary = CapillaryRiseTransport(name="capillary_rise_transport")
        richards = RichardsSoilMoisture(name="richards_soil_moisture")
        groundwater = GroundwaterDynamics(name="groundwater_dynamics")
        richards.set_boundary_conditions(
            top={"type": "atmospheric"},
            bottom={"type": "capillary_flux"},
        )
        simulator.register_functor(capillary, execution_group="fast")
        simulator.register_functor(richards, execution_group="fast")
        simulator.register_functor(groundwater, execution_group="fast")
        simulator._compute_group_execution_orders()
        assert simulator._order_fast.index("capillary_rise_transport") < (
            simulator._order_fast.index("richards_soil_moisture")
        )
        assert simulator._order_fast.index("richards_soil_moisture") < (
            simulator._order_fast.index("groundwater_dynamics")
        )
        return simulator

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        continuous = build_simulator()
        assert "capillary_rise_flux" not in continuous.state_manager.get_state()
        continuous.step(dt=dt)
        continuous_updates = continuous.step(dt=dt)

        first_leg = build_simulator()
        first_leg.step(dt=dt)
        checkpoint = first_leg.state_manager.checkpoint()

        restarted = build_simulator()
        restarted.state_manager.restore(checkpoint)
        restarted.current_time = first_leg.current_time
        restarted.step_count = first_leg.step_count
        moisture_before = restarted.state_manager.get_state("soil_moisture").clone()
        groundwater_before = restarted.state_manager.get_state(
            "groundwater_storage"
        ).clone()
        water_table_before = restarted.state_manager.get_state(
            "water_table_depth"
        ).clone()
        restarted_updates = restarted.step(dt=dt)

    for key in (
        "soil_moisture",
        "soil_matric_potential",
        "drainage_flux",
        "capillary_rise_flux",
        "groundwater_storage",
        "water_table_depth",
        "baseflow_flux",
    ):
        torch.testing.assert_close(
            restarted.state_manager.get_state(key),
            continuous.state_manager.get_state(key),
            rtol=0.0,
            atol=0.0,
        )

    assert restarted.current_time == continuous.current_time == 2.0 * dt
    assert restarted.step_count == continuous.step_count == 2
    capillary_flux = restarted.state_manager.get_state("capillary_rise_flux")
    assert float(capillary_flux.detach()) > 0.0
    torch.testing.assert_close(
        restarted.state_manager.get_state("drainage_flux"),
        torch.zeros_like(restarted.state_manager.get_state("drainage_flux")),
    )
    expected_capillary = CapillaryRiseTransport(name="expected_capillary")
    expected_capillary_flux = expected_capillary.forward(
        {"water_table_depth": water_table_before},
        time=dt,
        dt=dt,
    )["capillary_rise_flux"].value
    torch.testing.assert_close(
        capillary_flux,
        expected_capillary_flux.to(
            device=capillary_flux.device,
            dtype=capillary_flux.dtype,
        ),
    )
    profile_key = "diagnostics.soil_moisture_richards.water_flux_profile"
    bottom_mass_flux = restarted_updates[profile_key][-1]
    capillary_flux_for_richards = capillary_flux.to(
        device=bottom_mass_flux.device,
        dtype=bottom_mass_flux.dtype,
    )
    torch.testing.assert_close(bottom_mass_flux, -capillary_flux_for_richards)
    torch.testing.assert_close(
        restarted_updates[profile_key],
        continuous_updates[profile_key],
        rtol=0.0,
        atol=0.0,
    )

    richards = restarted.functors["richards_soil_moisture"]
    moisture_after = restarted.state_manager.get_state("soil_moisture")
    storage_rate = (
        torch.sum(moisture_after - moisture_before)
        * richards.layer_thickness
        * richards.water_density
        / dt
    )
    torch.testing.assert_close(
        storage_rate,
        capillary_flux.to(device=storage_rate.device, dtype=storage_rate.dtype),
        rtol=1.0e-3,
        atol=1.0e-8,
    )
    mass_error = restarted_updates[
        "diagnostics.soil_moisture_richards.mass_conservation_error"
    ]
    assert mass_error <= richards.mass_conservation_tolerance

    groundwater = restarted.functors["groundwater_dynamics"]
    groundwater_after = restarted.state_manager.get_state("groundwater_storage")
    groundwater_mass_change = (
        groundwater_after - groundwater_before
    ) * groundwater.water_density_reference / 1000.0
    drainage_flux = restarted.state_manager.get_state("drainage_flux").to(
        dtype=groundwater_mass_change.dtype
    )
    baseflow_flux = restarted.state_manager.get_state("baseflow_flux").to(
        dtype=groundwater_mass_change.dtype
    )
    capillary_for_groundwater = capillary_flux.to(dtype=groundwater_mass_change.dtype)
    expected_groundwater_mass_change = (
        drainage_flux - capillary_for_groundwater - baseflow_flux
    ) * dt
    torch.testing.assert_close(
        groundwater_mass_change,
        expected_groundwater_mass_change,
        rtol=1.0e-8,
        atol=1.0e-10,
    )

    soil_mass_change = (
        torch.sum(moisture_after - moisture_before)
        * richards.layer_thickness
        * richards.water_density
    )
    combined_mass_change = soil_mass_change + groundwater_mass_change
    torch.testing.assert_close(
        combined_mass_change,
        -baseflow_flux * dt,
        rtol=1.0e-3,
        atol=1.0e-8,
    )

    expected_writers = {
        "capillary_rise_flux": "capillary_rise_transport",
        "soil_moisture": "richards_soil_moisture",
        "soil_matric_potential": "richards_soil_moisture",
        "drainage_flux": "richards_soil_moisture",
        "groundwater_storage": "groundwater_dynamics",
        "water_table_depth": "groundwater_dynamics",
        "baseflow_flux": "groundwater_dynamics",
    }
    for key, writer in expected_writers.items():
        provenance = restarted.get_variable_provenance(key)
        assert {entry["functor"] for entry in provenance} == {writer}
        assert {entry["role"] for entry in provenance} == {
            "writer",
            "final_writer",
        }
        assert {entry["mode"] for entry in provenance} == {
            "UpdateMode.REPLACE"
        }
    assert restarted.get_effective_device("capillary_rise_transport").type == "cpu"
    assert restarted.get_effective_device("richards_soil_moisture").type == "cpu"
    assert restarted.get_effective_device("groundwater_dynamics").type == "cpu"
    assert caught == [], [str(item.message) for item in caught]


def test_vertical_groundwater_float32_restart_has_bounded_mass_residual() -> None:
    """Vertical drainage restart must preserve the open groundwater ledger.

    References: van Genuchten (1980); Mualem (1976); Wigmosta et al. (1994).
    Mathematical expression:
        Delta S = (F_drain + F_lat - F_pump - F_cap - F_base)
            * 1000 / rho_w * dt.
    Parameter meanings:
        S is groundwater storage [mm]; F_drain, F_lat, F_pump, F_cap, and
        F_base are mass fluxes [kg m-2 s-1]; rho_w is water density
        [kg m-3]; and dt is time [s].
    """
    dt = 3600.0
    initial_state = {
        "soil_moisture": torch.tensor(0.40, dtype=torch.float32),
        "soil_temperature": torch.tensor(298.15, dtype=torch.float32),
        "groundwater_storage": torch.tensor(100.0, dtype=torch.float32),
        "water_table_depth": torch.tensor(2.0, dtype=torch.float32),
        "capillary_rise_flux": torch.tensor(5.0e-7, dtype=torch.float32),
        "lateral_inflow": torch.tensor(2.0e-6, dtype=torch.float32),
        "groundwater_pumping": torch.tensor(1.0e-6, dtype=torch.float32),
    }

    def build_simulator() -> RELICSimulator:
        simulator = _declared_simulator(
            {
                "max_time": 2.0 * dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": True,
                "strict_dependencies": True,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize_state(
            {key: value.clone() for key, value in initial_state.items()}
        )
        simulator.register_functor(
            VerticalPercolationTransport(
                name="vertical_percolation_transport"
            ),
            execution_group="fast",
        )
        simulator.register_functor(
            GroundwaterDynamics(name="groundwater_dynamics"),
            execution_group="fast",
        )
        return simulator

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        continuous = build_simulator()
        continuous.step(dt=dt)
        continuous_second = continuous.step(dt=dt)

        first_leg = build_simulator()
        first_leg.step(dt=dt)
        checkpoint = first_leg.state_manager.checkpoint()

        restarted = build_simulator()
        restarted.state_manager.restore(checkpoint)
        restarted.current_time = first_leg.current_time
        restarted.step_count = first_leg.step_count
        restarted_input = restarted.state_manager.get_state(clone=True)
        direct_second = restarted.functors["groundwater_dynamics"].forward(
            restarted_input,
            time=restarted.current_time,
            dt=dt,
        )
        restarted_second = restarted.step(dt=dt)

    for key in (
        "groundwater_storage",
        "water_table_depth",
        "drainage_flux",
        "baseflow_flux",
    ):
        torch.testing.assert_close(
            restarted.state_manager.get_state(key),
            continuous.state_manager.get_state(key),
            rtol=0.0,
            atol=0.0,
        )
    for key in ("groundwater_storage", "water_table_depth", "baseflow_flux"):
        torch.testing.assert_close(
            restarted_second[key],
            continuous_second[key],
            rtol=0.0,
            atol=0.0,
        )

    diagnostics = direct_second["diagnostics.groundwater"]
    storage_before = restarted_input["groundwater_storage"]
    storage_after = direct_second["groundwater_storage"].value
    baseflow_flux = direct_second["baseflow_flux"].value
    water_density = torch.as_tensor(
        restarted.functors["groundwater_dynamics"].water_density_reference,
        dtype=storage_before.dtype,
        device=storage_before.device,
    )
    expected_port_increment = (
        restarted_input["drainage_flux"]
        + restarted_input["lateral_inflow"]
        - restarted_input["groundwater_pumping"]
        - restarted_input["capillary_rise_flux"]
        - baseflow_flux
    ) * (1000.0 / water_density) * dt
    actual_port_increment = diagnostics["net_groundwater_flux"] * dt
    machine_epsilon = torch.finfo(storage_before.dtype).eps
    gamma_8 = 8.0 * machine_epsilon / (1.0 - 8.0 * machine_epsilon)
    expected_roundoff_bound = gamma_8 * (
        torch.abs(storage_after)
        + torch.abs(storage_before)
        + torch.abs(actual_port_increment)
    )

    for port_value in (
        restarted_input["drainage_flux"],
        restarted_input["lateral_inflow"],
        restarted_input["groundwater_pumping"],
        restarted_input["capillary_rise_flux"],
        baseflow_flux,
    ):
        assert bool(port_value > 0.0)
    torch.testing.assert_close(
        actual_port_increment,
        expected_port_increment,
        rtol=gamma_8,
        atol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["storage_balance_roundoff_bound"],
        expected_roundoff_bound,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.abs(
        diagnostics["storage_balance_residual"]
    ) <= diagnostics["storage_balance_roundoff_bound"]
    assert restarted.functors["groundwater_dynamics"].check_conservation_laws(
        restarted_input,
        direct_second,
    ) == {"mass": 0.0}

    expected_writers = {
        "drainage_flux": "vertical_percolation_transport",
        "groundwater_storage": "groundwater_dynamics",
        "water_table_depth": "groundwater_dynamics",
        "baseflow_flux": "groundwater_dynamics",
    }
    for key, writer in expected_writers.items():
        provenance = restarted.get_variable_provenance(key)
        assert {entry["functor"] for entry in provenance} == {writer}
        assert {entry["role"] for entry in provenance} == {
            "writer",
            "final_writer",
        }
    assert restarted.get_effective_device(
        "vertical_percolation_transport"
    ).type == "cpu"
    assert restarted.get_effective_device("groundwater_dynamics").type == "cpu"
    assert not [
        item
        for item in caught
        if issubclass(item.category, UserWarning)
    ]


