import asyncio
import copy
import os
import random
import warnings

import numpy as np
import pytest
import torch

import relic.core.simulator.dependency_ops as simulator_dependency_ops
import relic.core.simulator.registration as simulator_registration
from relic.core.functor import (
    ConflictResolutionPolicy,
    ExecutionError,
    ExecutionPhase,
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
)
from relic.core.simulator import RELICSimulator, SimulationState
from relic.core.dependency_graph import DependencyGraphResolver, MissingDependencyError
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import (
    PriestleyTaylorET,
)
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff
from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport,
)
from relic.scientific.pedosphere.transport.SoilWaterTransportFunctor import (
    SoilWaterTransportFunctor,
)
from relic.scientific.pedosphere.carbon_pools.carbon_pools_functor import (
    CarbonPoolsFunctor,
)
from relic.scientific.pedosphere.carbon_pools.pools_matrix import PoolsMatrix
from relic.scientific.pedosphere.decomposition.first_order import (
    FirstOrderDecomposition,
)
from relic.scientific.pedosphere.decomposition.microbial_enzyme import (
    MicrobialEnzymeDecomposition,
)
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.anthroposphere.management.coordinator import (
    ManagementCoordinator,
)
from relic.scientific.anthroposphere.management.event_scheduler import (
    EventPriority,
    EventType,
    ManagementEvent,
)
from relic.core.state import ConservationSummary, ExternalIntent, StateManager, UpdateItem, UpdateMode
from relic.core.simulator.checkpoint import _capture_rng_state
from relic.core.functor.functor_types import SpatialScale, TemporalScale, VariableKind
from relic.core.state import StateVariableSpec


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'dependency_flux': 'dimensionless',
    'dependency_total': 'dimensionless',
    'misdeclared_total': 'dimensionless',
    'random_state': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class _ExactIntervalTransport(TransportMechanismFunctor):
    """Integrate a constant unit rate to expose scheduler interval precision."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self, substeps):
        metadata = FunctorMetadata(
            name="exact_interval_transport",
            description="Exact elapsed-interval transport fixture.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            mathematical_expression="d clock_integral / dt = 1",
            references=["Analytical solution of a constant-rate ODE"],
            input_variables=["clock_integral"],
            output_variables=["clock_integral"],
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.STATE,
            substep_count=substeps,
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: exact integration of a constant unit-rate ODE.
        # Mathematical expression: Delta x = dt for dx/dt = 1.
        # Parameter meanings: x records the integrated elapsed interval.
        # Units: x and dt [s]; the unit rate is dimensionless.
        # Domain/constraints: positive finite scalar intervals.
        # Sign convention: positive dt increases the integrated clock.
        return {"clock_integral": UpdateItem(
            torch.as_tensor(dt, dtype=state["clock_integral"].dtype,
                            device=state["clock_integral"].device),
            mode=UpdateMode.ACCUMULATE, unit="s", per_timestep=True,
            source_functor=self.name,
        )}


@pytest.mark.parametrize("substeps", [1, 4])
def test_float64_interval_matches_clock_through_governed_substeps(substeps):
    interval = 1.0 + 2.0 ** -30
    manager = StateManager({"clock_integral": StateVariableSpec(
        name="clock_integral", units="s", conserved=False,
    )}, device="cpu")
    manager.initialize({"clock_integral": torch.tensor(0.0, dtype=torch.float64)})
    simulator = _declared_simulator({"device": "cpu", "time_step": interval})
    simulator.initialize(manager, DependencyGraphResolver())
    simulator.register_functor(_ExactIntervalTransport(substeps), execution_group="fast")
    simulator.step(interval)
    assert simulator.get_state()["clock_integral"].item() == interval
    assert simulator.current_time == interval
    assert simulator.step_count == 1


class _DummyETFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="dummy et overlap checker",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["evapotranspiration_flux"],
            conflict_resolution_policy=ConflictResolutionPolicy.INHERIT,
        )
        super().__init__(name=name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"evapotranspiration_flux": torch.tensor(0.0)}


class _DummyPolicyFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str, policy: ConflictResolutionPolicy):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="dummy policy overlap checker",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["dummy_overlap_flux"],
            conflict_resolution_policy=policy,
        )
        super().__init__(name=name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"dummy_overlap_flux": torch.tensor(0.0)}


class _DummyBaseflowFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="dummy baseflow overlap checker",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["baseflow_flux"],
            conflict_resolution_policy=ConflictResolutionPolicy.INHERIT,
        )
        super().__init__(name=name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"baseflow_flux": torch.tensor(0.0)}


class _DummyMultiElementFunctor(TransportMechanismFunctor):
    """Fixture proving that process laws do not classify output elements."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self):
        metadata = FunctorMetadata(
            name="dummy_multi_element_functor",
            version="1.0.0",
            description="multi-element conservation ledger fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            conservation_laws=["carbon", "nitrogen"],
            output_variables=["phenology_phase", "leaf_carbon", "leaf_nitrogen"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {}


class _PrimaryWriterOverlapBase(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str, value: float):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="legacy shared overlap test fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["ozone_stress_factor"],
            conflict_resolution_policy=ConflictResolutionPolicy.INHERIT,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.value = torch.tensor(value, dtype=torch.float32)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {
            "ozone_stress_factor": self.wrap_update(self.value, mode=UpdateMode.REPLACE),
        }


class OzoneStress(_PrimaryWriterOverlapBase):
    pass


class FarquharPhotosynthesis(_PrimaryWriterOverlapBase):
    pass


class _PlainForwardObject:
    """Non-Functor object used to verify the registration boundary."""

    name = "plain_forward_object"

    def forward(self, state, time, dt):
        return {}


class _UnbridgedScientificFunctor(FunctorBase):
    """Scientific fixture that deliberately bypasses the four bridges."""

    def __init__(self):
        metadata = FunctorMetadata(
            name="unbridged_scientific_functor",
            version="1.0.0",
            description="unbridged scientific registration fixture",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["unbridged_output"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"unbridged_output": torch.tensor(0.0)}


class _MismatchedBridgeFunctor(TransportMechanismFunctor):
    """Bridge fixture whose declared scientific category is inconsistent."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self):
        metadata = FunctorMetadata(
            name="mismatched_bridge_functor",
            version="1.0.0",
            description="bridge and category mismatch fixture",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["mismatched_bridge_output"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"mismatched_bridge_output": torch.tensor(0.0)}


class _DependencyOutputProducer(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, requests=(), substeps=1, accumulate=False):
        metadata = FunctorMetadata(
            name="dependency_output_producer",
            version="1.0.0",
            description="dependency output fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["dependency_flux"],
            substep_count=substeps,
        )
        super().__init__(name=metadata.name, config={"external_requests": list(requests), "accumulate": accumulate}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        value = torch.tensor(2.5, dtype=torch.float32)
        if self._original_config["accumulate"]:
            value = UpdateItem(value, mode=UpdateMode.ACCUMULATE, per_timestep=True)
        return {
            "dependency_flux": value,
            **{f"delivery_{index}": ExternalIntent(**request)
               for index, request in enumerate(self._original_config["external_requests"])},
        }


class _DependencyOutputConsumer(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self):
        metadata = FunctorMetadata(
            name="dependency_output_consumer",
            version="1.0.0",
            description="dependency interconnection fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["dependency_total"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return ["dependency_output_producer"]

    def forward(self, state, time, dt, subprocess_outputs, **kwargs):
        producer_output = subprocess_outputs["dependency_output_producer"]
        return {"dependency_total": producer_output["dependency_flux"]}


class _OptionalDependencyConsumer(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self):
        metadata = FunctorMetadata(
            name="optional_dependency_consumer",
            version="1.0.0",
            description="optional dependency rebinding fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["optional_dependency_total"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return {"optional": ["dependency_output_producer"]}

    def forward(self, state, time, dt, **kwargs):
        return {"optional_dependency_total": torch.tensor(1.0)}


class _RandomDrawFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, *, fail: bool):
        metadata = FunctorMetadata(
            name="random_draw_functor",
            version="1.0.0",
            description="RNG transaction fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["random_state"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)
        self.fail = fail
        self.draws = []

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        draw = (
            random.random(),
            float(np.random.random()),
            float(torch.rand((), device=self.device).cpu()),
        )
        self.draws.append(draw)
        if self.fail:
            raise RuntimeError("injected stochastic step failure")
        return {"random_state": torch.tensor(10.0, device=self.device)}


def _assert_rng_state_equal(left, right):
    assert left["python"] == right["python"]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    assert len(left["torch_cuda"]) == len(right["torch_cuda"])
    for left_state, right_state in zip(left["torch_cuda"], right["torch_cuda"]):
        assert torch.equal(left_state, right_state)
    assert (left["numpy"] is None) == (right["numpy"] is None)
    if left["numpy"] is not None:
        assert left["numpy"]["algorithm"] == right["numpy"]["algorithm"]
        assert torch.equal(left["numpy"]["keys"], right["numpy"]["keys"])
        assert left["numpy"]["position"] == right["numpy"]["position"]
        assert left["numpy"]["has_gauss"] == right["numpy"]["has_gauss"]
        assert left["numpy"]["cached_gaussian"] == right["numpy"]["cached_gaussian"]


def _assert_nested_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert list(left) == list(right)
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_state_equal(left_item, right_item)
        return
    assert left == right


class _VarKwargsRecorder(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self):
        metadata = FunctorMetadata(
            name="var_kwargs_recorder",
            version="1.0.0",
            description="generic keyword routing fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["var_kwargs_result"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)
        self.received_kwargs = None

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        self.received_kwargs = dict(kwargs)
        return {"var_kwargs_result": torch.tensor(1.0)}


class _ExplicitIntegrationControlRecorder(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self):
        metadata = FunctorMetadata(
            name="explicit_integration_control_recorder",
            version="1.0.0",
            description="explicit integration control routing fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["explicit_integration_result"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)
        self.received_controls = None

    def get_dependencies(self):
        return []

    def forward(
        self,
        state,
        time,
        dt,
        time_integration_method,
        implicit_midpoint_max_iter,
        implicit_midpoint_tolerance,
        calendar_type,
    ):
        self.received_controls = (
            time_integration_method,
            implicit_midpoint_max_iter,
            implicit_midpoint_tolerance,
            calendar_type,
        )
        return {"explicit_integration_result": torch.tensor(1.0)}


def test_step_does_not_crash_on_cross_group_dependency_scan():
    sim = _declared_simulator(
        {
            "max_time": 10.0,
            "time_step": 60.0,
            "conservation_checks": False,
        }
    )
    sim.initialize_state({"soil_moisture": torch.tensor(0.30, dtype=torch.float32)})
    sim.register_functor(BucketSoilMoisture(name="bucket_soil_moisture"), execution_group="slow")

    updates = sim.step(dt=60.0)
    assert isinstance(updates, dict)
    assert sim.current_time > 0.0


def test_bucket_step_commits_replace_state_with_provenance() -> None:
    """The real simulator must commit the canonical Bucket result exactly once."""
    dt = 3600.0
    initial_state = {
        "soil_moisture": torch.tensor(0.30, dtype=torch.float64),
        "precipitation_flux": torch.tensor(0.001, dtype=torch.float64),
        "evapotranspiration_flux": torch.tensor(0.0001, dtype=torch.float64),
    }
    direct_functor = BucketSoilMoisture(name="bucket_soil_moisture")
    direct = direct_functor.forward(initial_state, time=0.0, dt=dt)
    expected = direct["soil_moisture"].value

    simulator = _declared_simulator(
        {
            "max_time": dt,
            "time_step": dt,
            "device": "cpu",
            "conservation_checks": True,
            "strict_dependencies": True,
        }
    )
    simulator.initialize_state(
        {key: value.clone() for key, value in initial_state.items()}
    )
    runtime_functor = BucketSoilMoisture(name="bucket_soil_moisture")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulator.register_functor(runtime_functor, execution_group="fast")
        updates = simulator.step(dt=dt)

    torch.testing.assert_close(updates["soil_moisture"], expected)
    torch.testing.assert_close(
        simulator.state_manager.get_state("soil_moisture"),
        expected,
    )
    assert "soil_moisture_change" not in updates
    assert "soil_water_storage" not in updates
    provenance = simulator.get_variable_provenance("soil_moisture")
    assert {entry["functor"] for entry in provenance} == {
        "bucket_soil_moisture"
    }
    assert {entry["role"] for entry in provenance} == {
        "writer",
        "final_writer",
    }
    assert {entry["mode"] for entry in provenance} == {
        "UpdateMode.REPLACE"
    }
    assert simulator.get_effective_device("bucket_soil_moisture").type == "cpu"
    assert caught == []


def test_registration_rejects_objects_outside_functor_lifecycle() -> None:
    """Every executable process must reach the simulator through FunctorBase."""
    sim = _declared_simulator({"conservation_checks": False})

    with pytest.raises(TypeError, match="must inherit from FunctorBase"):
        sim.register_functor(_PlainForwardObject())


def test_registration_rejects_unbridged_scientific_functor_before_mutation() -> None:
    """A scientific category cannot enter runtime through bare FunctorBase."""
    sim = _declared_simulator({"conservation_checks": False})
    candidate = _UnbridgedScientificFunctor()

    with pytest.raises(TypeError, match="requires BiologicalProcessFunctor"):
        sim.register_functor(candidate)

    assert candidate.name not in sim.functors
    assert all(candidate.name not in members for members in sim._groups.values())
    assert sim.dependency_graph is None


def test_registration_rejects_bridge_category_mismatch_before_mutation() -> None:
    """The bridge class must be the owner selected by the declared category."""
    sim = _declared_simulator({"conservation_checks": False})
    candidate = _MismatchedBridgeFunctor()

    with pytest.raises(TypeError, match="requires BiologicalProcessFunctor"):
        sim.register_functor(candidate)

    assert candidate.name not in sim.functors
    assert all(candidate.name not in members for members in sim._groups.values())
    assert sim.dependency_graph is None


def _registration_footprint(simulator) -> tuple:
    return (
        sorted(simulator.functors),
        {name: list(members) for name, members in simulator._groups.items()},
        simulator.dependency_graph,
        simulator._provider_lifecycle.epoch,
        dict(simulator._provider_lifecycle.provider_uids),
    )


def test_registration_consumes_canonical_execution_group_predicate(monkeypatch) -> None:
    """The explicit argument is judged by the one predicate dependency_ops._is_valid_execution_group."""
    simulator = _declared_simulator()
    functor = _DummyPolicyFunctor(
        "canonical_execution_group_predicate",
        ConflictResolutionPolicy.INHERIT,
    )
    monkeypatch.setattr(
        simulator_dependency_ops,
        "_is_valid_execution_group",
        lambda value: False,
    )
    before = _registration_footprint(simulator)

    with pytest.raises(ValueError, match="execution_group 'medium'"):
        simulator.register_functor(functor, execution_group="medium")

    assert _registration_footprint(simulator) == before


def test_invalid_explicit_execution_group_is_rejected_before_any_registration_state() -> None:
    simulator = _declared_simulator()
    functor = _DummyPolicyFunctor(
        "invalid_execution_group",
        ConflictResolutionPolicy.INHERIT,
    )
    before = _registration_footprint(simulator)

    with pytest.raises(ValueError, match="execution_group 'fas'"):
        simulator.register_functor(functor, execution_group="fas")

    assert _registration_footprint(simulator) == before
    assert simulator.dependency_graph is None


def test_resolved_group_outside_the_simulator_groups_is_an_invariant_error(monkeypatch) -> None:
    simulator = _declared_simulator()
    functor = _DummyPolicyFunctor("invariant_group", ConflictResolutionPolicy.INHERIT)
    monkeypatch.setattr(
        simulator,
        "_determine_execution_group",
        lambda candidate, execution_group=None: "turbo",
    )
    before = _registration_footprint(simulator)

    with pytest.raises(RuntimeError, match="turbo"):
        simulator.register_functor(functor)

    assert _registration_footprint(simulator) == before


def test_registration_argument_overrides_a_conflicting_process_groups_entry() -> None:
    simulator = _declared_simulator({"process_groups": {"argument_override": "slow"}})
    functor = _DummyPolicyFunctor("argument_override", ConflictResolutionPolicy.INHERIT)
    functor.metadata.temporal_scale = TemporalScale.SECOND

    simulator.register_functor(functor, execution_group="medium")

    assert functor.name in simulator._groups["medium"]


def test_valid_explicit_execution_group_is_preserved(caplog) -> None:
    simulator = _declared_simulator()
    functor = _DummyPolicyFunctor(
        "valid_explicit_execution_group",
        ConflictResolutionPolicy.INHERIT,
    )

    with caplog.at_level("WARNING", logger=simulator_registration.__name__):
        simulator.register_functor(functor, execution_group="medium")

    assert "Invalid execution group" not in caplog.text
    assert functor.name in simulator._groups["medium"]


def test_dependency_outputs_are_injected_from_existing_execution_order():
    """A consumer receives raw outputs from its declared dependency only."""
    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "strict_dependencies": True,
            "time_step": 1.0,
        }
    )
    sim.initialize_state(
        {
            "dependency_flux": torch.tensor(0.0),
            "dependency_total": torch.tensor(0.0),
        }
    )
    sim.register_functor(_DependencyOutputProducer())
    sim.register_functor(_DependencyOutputConsumer())

    updates = sim.step(dt=1.0)

    dependency_total = updates["dependency_total"]
    if hasattr(dependency_total, "value"):
        dependency_total = dependency_total.value
    assert torch.allclose(dependency_total.cpu(), torch.tensor(2.5))


def test_simulator_routes_only_explicitly_declared_integration_controls():
    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "time_step": 1.0,
            "time_integration_method": "implicit_midpoint",
            "implicit_midpoint_max_iter": 7,
            "implicit_midpoint_tolerance": 1.0e-8,
        }
    )
    sim.initialize_state({})
    generic = _VarKwargsRecorder()
    explicit = _ExplicitIntegrationControlRecorder()
    sim.register_functor(generic)
    sim.register_functor(explicit)

    sim.step(dt=1.0)

    assert generic.received_kwargs == {}
    assert explicit.received_controls == (
        "implicit_midpoint",
        7,
        1.0e-8,
        "noleap",
    )


def test_unconfigured_simulator_takes_integration_controls_from_the_parameter_database():
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    sim = _declared_simulator({"conservation_checks": False, "time_step": 1.0})
    sim.initialize_state({})
    explicit = _ExplicitIntegrationControlRecorder()
    sim.register_functor(explicit)

    sim.step(dt=1.0)

    method, max_iter, tolerance, _calendar = explicit.received_controls
    assert method == manager.get("transport.numerical.time_integration_method", required=True)
    assert max_iter == manager.get("transport.numerical.implicit_midpoint_max_iter", required=True) == 50
    assert tolerance == float(manager.get("transport.numerical.implicit_midpoint_tolerance", required=True))


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("time_integration_method", "rk4"),
        ("implicit_midpoint_max_iter", 0),
        ("implicit_midpoint_max_iter", 101),
        ("implicit_midpoint_max_iter", 2.5),
        ("implicit_midpoint_tolerance", 1.0e-3),
    ],
)
def test_configured_integration_controls_are_validated_against_the_parameter_database(setting, value):
    with pytest.raises(ValueError, match=setting):
        _declared_simulator({"conservation_checks": False, "time_step": 1.0, setting: value})


def test_management_handler_failure_rolls_back_the_accepted_step_transaction():
    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "strict_dependencies": False,
            "time_step": 3600.0,
            "device": "cpu",
        }
    )
    sim.initialize_state(
        {
            "soil_moisture": torch.tensor([0.3]),
            "soil_nitrogen": torch.tensor([0.01]),
            "soil_organic_carbon": torch.tensor([5.0]),
            "crop_maturity": torch.tensor([0.7]),
            "aboveground_biomass": torch.tensor([1.0]),
            "air_temperature": torch.tensor([298.15]),
            "irrigation_flux": torch.tensor([0.0]),
            "harvest_flux": torch.tensor([0.0]),
            "grazing_flux": torch.tensor([0.0]),
            "co2_emission": torch.tensor([0.0]),
            "field_capacity": torch.tensor([0.35]),
            "wilting_point": torch.tensor([0.15]),
            "evapotranspiration_flux": torch.tensor([3.0e-6]),
            "precipitation_flux": torch.tensor([0.0]),
        }
    )
    coordinator = ManagementCoordinator()
    coordinator.last_coordination_time = 0.0

    def fail_handler(_state, _parameters):
        raise RuntimeError("injected coordinator handler failure")

    coordinator.scheduler.register_event_handler(EventType.IRRIGATION, fail_handler)
    event = ManagementEvent(
        event_id="failing_irrigation",
        event_type=EventType.IRRIGATION,
        scheduled_time=0.0,
        priority=EventPriority.NORMAL,
        duration=60.0,
        parameters={"water_amount": 1.0},
        resource_requirements={"water": 1.0},
    )
    assert coordinator.scheduler.schedule_event(event) is True
    sim.register_functor(coordinator, execution_group="fast")
    state_before = {
        key: value.detach().clone() for key, value in sim.get_state().items()
    }
    functor_before = copy.deepcopy(coordinator.state_dict())
    time_before = sim.current_time
    step_before = sim.step_count

    with pytest.raises(ExecutionError, match="injected coordinator handler failure"):
        sim.step(dt=3600.0)

    assert sim.current_time == time_before
    assert sim.step_count == step_before
    for key, value in state_before.items():
        assert torch.equal(sim.get_state()[key], value)
    _assert_nested_state_equal(coordinator.state_dict(), functor_before)
    assert [queued.event_id for queued in coordinator.scheduler.events] == [
        "failing_irrigation"
    ]
    assert coordinator.scheduler.resource_quotas["water"].current_used == pytest.approx(
        0.0
    )


def test_penman_and_priestley_are_mutually_exclusive():
    sim = _declared_simulator({"conservation_checks": False})
    sim.register_functor(PenmanMonteithET(name="penman_monteith_et", config={"canopy_resistance_source": "state"}))

    with pytest.raises(ValueError, match="mutually exclusive process model"):
        sim.register_functor(PriestleyTaylorET(name="priestley_taylor_et"))


@pytest.mark.parametrize(
    ("selected_factory", "rejected_factory"),
    [
        (
            lambda: BucketSoilMoisture(name="bucket_soil_moisture"),
            lambda: RichardsSoilMoisture(name="richards_soil_moisture"),
        ),
        (
            lambda: RichardsSoilMoisture(name="richards_soil_moisture"),
            lambda: BucketSoilMoisture(name="bucket_soil_moisture"),
        ),
        (
            lambda: BucketSoilMoisture(name="bucket_soil_moisture"),
            SoilWaterTransportFunctor,
        ),
        (
            SoilWaterTransportFunctor,
            lambda: BucketSoilMoisture(name="bucket_soil_moisture"),
        ),
        (
            lambda: RichardsSoilMoisture(name="richards_soil_moisture"),
            SoilWaterTransportFunctor,
        ),
        (
            SoilWaterTransportFunctor,
            lambda: RichardsSoilMoisture(name="richards_soil_moisture"),
        ),
    ],
    ids=[
        "bucket-richards",
        "richards-bucket",
        "bucket-pedosphere-richards",
        "pedosphere-richards-bucket",
        "hydrosphere-richards-pedosphere-richards",
        "pedosphere-richards-hydrosphere-richards",
    ],
)
def test_soil_moisture_full_state_closures_are_transactionally_exclusive(
    selected_factory,
    rejected_factory,
) -> None:
    """Every selected soil-moisture closure must exclude both alternatives."""

    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False}
    )
    selected = selected_factory()
    rejected = rejected_factory()
    sim.register_functor(selected)

    with pytest.raises(
        ValueError,
        match="mutually exclusive process model|Mutually exclusive overlap variable",
    ):
        sim.register_functor(rejected)

    assert selected.name in sim.functors
    _assert_registration_absent(sim, rejected.name)


def test_bucket_and_surface_runoff_are_mutually_exclusive():
    sim = _declared_simulator({"conservation_checks": False})
    sim.register_functor(BucketSoilMoisture(name="bucket_soil_moisture"))

    with pytest.raises(ValueError, match="mutually exclusive process model"):
        sim.register_functor(SurfaceRunoff(name="surface_runoff"))


def _assert_registration_absent(simulator: RELICSimulator, functor_name: str) -> None:
    """A rejected registration must not mutate any runtime registry."""
    assert functor_name not in simulator.functors
    assert all(functor_name not in members for members in simulator._groups.values())
    assert functor_name not in simulator._bootstrap_groups
    if simulator.dependency_graph is not None:
        assert functor_name not in simulator.dependency_graph.nodes
    if simulator.execution_order is not None:
        assert functor_name not in simulator.execution_order


@pytest.mark.parametrize(
    ("selected_factory", "rejected_factory"),
    [
        (
            lambda: GroundwaterDynamics(name="groundwater_storage_owner"),
            lambda: SubsurfaceRunoff(name="subsurface_storage_alternative"),
        ),
    ],
    ids=["groundwater-storage"],
)
def test_hydrology_full_state_closures_are_transactionally_exclusive(
    selected_factory,
    rejected_factory,
) -> None:
    """Alternative groundwater/runoff closures must have one active writer."""
    simulator = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False}
    )
    simulator.register_functor(selected_factory())
    rejected = rejected_factory()

    with pytest.raises(ValueError, match="Mutually exclusive overlap variable"):
        simulator.register_functor(rejected)

    _assert_registration_absent(simulator, rejected.name)


def test_bucket_and_vertical_percolation_are_transactionally_exclusive() -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    sim.register_functor(BucketSoilMoisture(name="bucket_overlap_vertical"))
    vertical = VerticalPercolationTransport(name="vertical_percolation_overlap")

    with pytest.raises(ValueError, match="Mutually exclusive overlap variable"):
        sim.register_functor(vertical)

    assert "bucket_overlap_vertical" in sim.functors
    _assert_registration_absent(sim, vertical.name)


def test_critical_output_overlap_is_blocked_even_when_not_strict():
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    sim.register_functor(_DummyETFunctor(name="dummy_et_a"))

    with pytest.raises(ValueError, match="Critical single-writer outputs|Mutually exclusive overlap variable"):
        sim.register_functor(_DummyETFunctor(name="dummy_et_b"))


def test_metadata_error_policy_blocks_overlap_when_not_strict():
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    sim.register_functor(
        _DummyPolicyFunctor(
            name="dummy_policy_a",
            policy=ConflictResolutionPolicy.ERROR,
        )
    )

    with pytest.raises(ValueError, match="Output variable overlap detected"):
        sim.register_functor(
            _DummyPolicyFunctor(
                name="dummy_policy_b",
                policy=ConflictResolutionPolicy.INHERIT,
            )
        )


def test_transport_models_are_mutually_exclusive_when_strict() -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": True})
    sim.register_functor(LinearAdvection(name="linear_advection_guard", spatial_dims=1, grid_spacing=[0.1]))

    with pytest.raises(ValueError, match="mutually exclusive process model"):
        sim.register_functor(FickianDiffusion(name="fickian_diffusion_guard", spatial_dims=1, grid_spacing=[0.1]))


def test_single_transport_model_can_register_and_step_when_strict() -> None:
    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "strict_dependencies": True,
        }
    )
    advection = LinearAdvection(
        name="linear_advection_step",
        spatial_dims=1,
        velocity_field=[torch.zeros(1, 16, dtype=torch.float32)],
        grid_spacing=[0.1],
    )
    sim.initialize_state({"tracer_concentration": torch.linspace(0.0, 1.0, 16).unsqueeze(0)})
    sim.register_functor(advection)

    updates = advection.forward(
        state={"tracer_concentration": torch.linspace(0.0, 1.0, 16).unsqueeze(0)},
        time=0.0,
        dt=0.01,
    )
    assert isinstance(updates, dict)
    assert "tracer_concentration" in updates


def test_advection_diffusion_blocks_linear_advection_in_strict_mode() -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": True})
    sim.register_functor(AdvectionDiffusion(name="advdiff_guard", spatial_dims=1, diffusivity=1.0e-5, grid_spacing=(0.1,)))

    with pytest.raises(ValueError, match="mutually exclusive process model"):
        sim.register_functor(LinearAdvection(name="linear_advection_after_advdiff", spatial_dims=1, grid_spacing=[0.1]))


def test_unmanaged_baseflow_overlap_does_not_create_a_fake_policy(caplog) -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    sim.register_functor(_DummyBaseflowFunctor(name="dummy_baseflow_a"))

    with caplog.at_level("WARNING"):
        sim.register_functor(_DummyBaseflowFunctor(name="dummy_baseflow_b"))

    assert "dummy_baseflow_b" in sim.functors
    assert "Output variable overlap detected" in caplog.text


@pytest.mark.parametrize(
    ("first_model", "second_model"),
    [
        (
            lambda: PoolsMatrix(name="pools_matrix_variant", config={}),
            lambda: FirstOrderDecomposition(name="first_order_variant", config={}),
        ),
        (
            lambda: PoolsMatrix(name="pools_matrix_variant", config={}),
            lambda: MicrobialEnzymeDecomposition(name="microbial_variant", config={}),
        ),
        (
            lambda: FirstOrderDecomposition(name="first_order_variant", config={}),
            lambda: MicrobialEnzymeDecomposition(name="microbial_variant", config={}),
        ),
    ],
    ids=["matrix-vs-first-order", "matrix-vs-microbial", "first-order-vs-microbial"],
)
def test_carbon_decomposition_model_variants_are_mutually_exclusive(
    first_model,
    second_model,
) -> None:
    """Exactly one carbon decomposition closure may own pool tendencies."""
    simulator = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    simulator.register_functor(first_model())
    rejected_model = second_model()

    with pytest.raises(ValueError, match="Mutually exclusive overlap variable"):
        simulator.register_functor(rejected_model)

    _assert_registration_absent(simulator, rejected_model.name)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: PoolsMatrix(name="pools_matrix_variant", config={}),
        lambda: FirstOrderDecomposition(name="first_order_variant", config={}),
        lambda: MicrobialEnzymeDecomposition(name="microbial_variant", config={}),
    ],
    ids=["matrix", "first-order", "microbial"],
)
def test_carbon_pool_observer_registers_with_one_selected_model(model_factory) -> None:
    """The state-neutral pool observer can compose with one selected closure."""
    simulator = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    simulator.register_functor(CarbonPoolsFunctor())
    simulator.register_functor(model_factory())

    assert "carbon_pools_observer" in simulator.functors


def test_competing_replace_writers_are_rejected() -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False, "time_step": 60.0})
    sim.initialize_state({"ozone_stress_factor": torch.tensor(1.0, dtype=torch.float32)})
    sim.register_functor(FarquharPhotosynthesis(name="farquhar_overlap_writer", value=0.9), execution_group="fast")
    sim.register_functor(OzoneStress(name="ozone_overlap_writer", value=0.4), execution_group="fast")

    with pytest.raises(RuntimeError, match="Group execution failed") as exc_info:
        sim.step(dt=60.0)

    assert exc_info.value.__cause__ is not None
    assert "Conflicting REPLACE updates for 'ozone_stress_factor'" in str(
        exc_info.value.__cause__
    )


def test_unregister_functor_removes_registered_functor() -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    functor_name = "dummy_unregister_target"
    sim.register_functor(
        _DummyPolicyFunctor(
            name=functor_name,
            policy=ConflictResolutionPolicy.INHERIT,
        ),
        execution_group="fast",
    )

    removed = sim.unregister_functor(functor_name)

    assert removed is True
    assert functor_name not in sim.functors
    assert all(functor_name not in members for members in sim._groups.values())
    assert functor_name not in sim.execution_order
    assert functor_name not in sim._order_fast
    assert functor_name not in sim._order_medium
    assert functor_name not in sim._order_slow


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_registration_rolls_back_every_registry_after_late_failure(monkeypatch, failure_type) -> None:
    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "strict_dependencies": False,
            "device": "cpu",
        }
    )
    sim.register_functor(_DependencyOutputProducer(), execution_group="fast")
    baseline_groups = {name: list(members) for name, members in sim._groups.items()}
    baseline_nodes = set(sim.dependency_graph.nodes)
    baseline_order = list(sim.execution_order)
    baseline_bootstrap = set(sim._bootstrap_groups)
    candidate = _OptionalDependencyConsumer()
    candidate.force_initial_execution = True
    candidate.device = torch.device("cuda")
    original_candidate_device = candidate.device

    def _fail_ledger_refresh():
        raise failure_type("injected late registration failure")

    monkeypatch.setattr(sim, "_refresh_conservation_ledger", _fail_ledger_refresh)

    with pytest.raises(failure_type, match="injected late registration failure"):
        sim.register_functor(candidate, execution_group="medium")

    _assert_registration_absent(sim, candidate.name)
    assert {name: list(members) for name, members in sim._groups.items()} == baseline_groups
    assert set(sim.dependency_graph.nodes) == baseline_nodes
    assert sim.execution_order == baseline_order
    assert sim._bootstrap_groups == baseline_bootstrap
    assert candidate.device == original_candidate_device


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_registration_rolls_back_after_partial_device_migration_failure(monkeypatch, failure_type) -> None:
    class _NestedDeviceOwner:
        def __init__(self) -> None:
            self.device = torch.device("cpu")

        def to_device(self, device) -> None:
            self.device = torch.device(device)

    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "strict_dependencies": False,
            "device": "cpu",
        }
    )
    sim.register_functor(_DependencyOutputProducer(), execution_group="fast")
    baseline_groups = {name: list(members) for name, members in sim._groups.items()}
    baseline_nodes = set(sim.dependency_graph.nodes)
    baseline_order = list(sim.execution_order)
    baseline_bootstrap = set(sim._bootstrap_groups)
    candidate = _OptionalDependencyConsumer()
    candidate.boundary_manager = _NestedDeviceOwner()
    original_candidate_device = candidate.device
    original_nested_device = candidate.boundary_manager.device
    monkeypatch.setattr(
        sim,
        "get_effective_device",
        lambda _functor_name: torch.device("meta"),
    )

    def _fail_device_migration(_target_device):
        candidate.device = torch.device("meta")
        candidate.boundary_manager.to_device("meta")
        raise failure_type("injected device migration failure")

    monkeypatch.setattr(candidate, "to_device", _fail_device_migration)

    with pytest.raises(failure_type, match="injected device migration failure"):
        sim.register_functor(candidate, execution_group="medium")

    _assert_registration_absent(sim, candidate.name)
    assert {name: list(members) for name, members in sim._groups.items()} == baseline_groups
    assert set(sim.dependency_graph.nodes) == baseline_nodes
    assert sim.execution_order == baseline_order
    assert sim._bootstrap_groups == baseline_bootstrap
    assert candidate.device == original_candidate_device
    assert candidate.boundary_manager.device == original_nested_device


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_unregistration_rolls_back_every_registry_after_late_failure(monkeypatch, failure_type) -> None:
    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "strict_dependencies": False,
            "device": "cpu",
        }
    )
    sim.set_state_manager(StateManager(config={}, device="cpu"))
    provider = _DependencyOutputProducer()
    sim.register_functor(provider, execution_group="fast")
    sim.register_functor(_OptionalDependencyConsumer(), execution_group="medium")
    baseline_groups = {name: list(members) for name, members in sim._groups.items()}
    baseline_nodes = set(sim.dependency_graph.nodes)
    baseline_edges = list(sim.dependency_graph.edges)
    baseline_order = list(sim.execution_order)
    baseline_group_orders = {
        "fast": list(sim._order_fast),
        "medium": list(sim._order_medium),
        "slow": list(sim._order_slow),
    }
    baseline_groups_dirty = sim._groups_dirty
    baseline_ledger = {
        name: list(members)
        for name, members in sim.state_manager.get_conservation_ledger().items()
    }

    def _fail_ledger_refresh():
        sim.state_manager.set_conservation_ledger({"carbon": []})
        raise failure_type("injected late unregistration failure")

    monkeypatch.setattr(sim, "_refresh_conservation_ledger", _fail_ledger_refresh)

    with pytest.raises(failure_type, match="injected late unregistration failure"):
        sim.unregister_functor(provider.name)

    assert sim.functors[provider.name] is provider
    assert {name: list(members) for name, members in sim._groups.items()} == baseline_groups
    assert set(sim.dependency_graph.nodes) == baseline_nodes
    assert list(sim.dependency_graph.edges) == baseline_edges
    assert sim.execution_order == baseline_order
    assert sim._order_fast == baseline_group_orders["fast"]
    assert sim._order_medium == baseline_group_orders["medium"]
    assert sim._order_slow == baseline_group_orders["slow"]
    assert sim._groups_dirty == baseline_groups_dirty
    assert sim.state_manager.get_conservation_ledger() == baseline_ledger


def test_device_rollback_attempts_all_owners_after_compensation_cancel(monkeypatch):
    attempted = []

    class DeviceOwner:
        def __init__(self, name):
            self.name = name
            self.device = torch.device("cpu")

        def to_device(self, device):
            attempted.append(self.name)
            if self.name == "first":
                raise asyncio.CancelledError("first compensation cancelled")
            self.device = torch.device(device)

    sim = _declared_simulator({"conservation_checks": False, "device": "cpu"})
    candidate = _DependencyOutputProducer()
    candidate.first_owner = DeviceOwner("first")
    candidate.second_owner = DeviceOwner("second")
    monkeypatch.setattr(sim, "get_effective_device", lambda _name: torch.device("meta"))

    def fail_migration(_device):
        candidate.first_owner.device = torch.device("meta")
        candidate.second_owner.device = torch.device("meta")
        raise RuntimeError("original migration failed")

    monkeypatch.setattr(candidate, "to_device", fail_migration)
    with pytest.raises(BaseExceptionGroup) as caught:
        sim.register_functor(candidate)
    assert attempted == ["first", "second"]
    assert candidate.second_owner.device == torch.device("cpu")
    assert isinstance(caught.value.exceptions[0], RuntimeError)
    assert "original migration failed" in str(caught.value.exceptions[0])
    _assert_registration_absent(sim, candidate.name)


@pytest.mark.parametrize("operation", ["register", "unregister"])
def test_registration_compensation_failure_preserves_cause_and_restores_ledger(monkeypatch, operation):
    sim = _declared_simulator({"conservation_checks": False, "device": "cpu"})
    sim.set_state_manager(StateManager(config={}, device="cpu"))
    provider = _DependencyOutputProducer()
    sim.register_functor(provider)
    baseline_ledger = copy.deepcopy(sim.state_manager.get_conservation_ledger())
    attempted = []

    def fail_commit():
        sim.state_manager.set_conservation_ledger({"carbon": ["corrupted"]})
        raise asyncio.CancelledError("original commit cancelled")

    def fail_graph_restore(_snapshot):
        attempted.append("graph")
        raise RuntimeError("graph compensation failed")

    original_ledger_setter = sim.state_manager.set_conservation_ledger

    def record_ledger_restore(ledger):
        attempted.append("ledger")
        original_ledger_setter(ledger)

    monkeypatch.setattr(sim, "_refresh_conservation_ledger", fail_commit)
    monkeypatch.setattr(sim.dependency_graph, "restore_mutation_state", fail_graph_restore)
    monkeypatch.setattr(sim.state_manager, "set_conservation_ledger", record_ledger_restore)
    with pytest.raises(BaseExceptionGroup) as caught:
        if operation == "register":
            sim.register_functor(_OptionalDependencyConsumer())
        else:
            sim.unregister_functor(provider.name)
    assert attempted[-2:] == ["graph", "ledger"]
    assert sim.state_manager.get_conservation_ledger() == baseline_ledger
    assert isinstance(caught.value.exceptions[0], asyncio.CancelledError)
    assert "original commit cancelled" in str(caught.value.exceptions[0])
    assert sim.functors[provider.name] is provider
    assert "optional_dependency_consumer" not in sim.functors


def test_strict_registration_rejects_missing_required_dependency_transactionally() -> None:
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": True}
    )
    candidate = _DependencyOutputConsumer()

    with pytest.raises(RuntimeError, match="Missing required dependencies"):
        sim.register_functor(candidate, execution_group="fast")

    _assert_registration_absent(sim, candidate.name)


def test_consumer_first_required_dependency_rebinds_without_optional_downgrade() -> None:
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False}
    )
    consumer = _DependencyOutputConsumer()
    provider = _DependencyOutputProducer()

    sim.register_functor(consumer, execution_group="fast")

    assert sim.execution_order is None
    assert provider.name in sim.dependency_graph.nodes[consumer.name].dependencies
    assert provider.name not in sim.dependency_graph._optional_deps.get(
        consumer.name,
        set(),
    )

    sim.register_functor(provider, execution_group="fast")

    matching_edges = [
        edge
        for edge in sim.dependency_graph.edges
        if edge.source == consumer.name and edge.target == provider.name
    ]
    assert len(matching_edges) == 1
    assert matching_edges[0].required is True
    assert sim.execution_order.index(provider.name) < sim.execution_order.index(
        consumer.name
    )

    with pytest.raises(MissingDependencyError):
        sim.unregister_functor(provider.name)

    assert provider.name in sim.functors
    assert consumer.name in sim.functors


def test_unregister_required_provider_fails_before_simulator_mutation() -> None:
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False}
    )
    sim.register_functor(_DependencyOutputProducer(), execution_group="fast")
    sim.register_functor(_DependencyOutputConsumer(), execution_group="fast")
    baseline_groups = {name: list(members) for name, members in sim._groups.items()}
    baseline_order = list(sim.execution_order)

    with pytest.raises(MissingDependencyError):
        sim.unregister_functor("dependency_output_producer")

    assert "dependency_output_producer" in sim.functors
    assert "dependency_output_consumer" in sim.functors
    assert {name: list(members) for name, members in sim._groups.items()} == baseline_groups
    assert sim.execution_order == baseline_order


def test_unregister_rejects_registry_graph_inconsistency_without_mutation() -> None:
    """A stale graph must not bypass required-dependency removal guards."""
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False}
    )
    sim.set_state_manager(StateManager(config={}, device="cpu"))
    provider = _DependencyOutputProducer()
    consumer = _DependencyOutputConsumer()
    sim.register_functor(provider, execution_group="fast")
    sim.register_functor(consumer, execution_group="medium")

    inconsistent_graph = DependencyGraphResolver(use_networkx=True)
    sim.dependency_graph = inconsistent_graph
    baseline_functors = tuple(
        (name, id(functor)) for name, functor in sim.functors.items()
    )
    baseline_groups = {name: list(members) for name, members in sim._groups.items()}
    baseline_bootstrap = set(sim._bootstrap_groups)
    baseline_groups_dirty = sim._groups_dirty
    baseline_order = list(sim.execution_order)
    baseline_group_orders = {
        "fast": list(sim._order_fast),
        "medium": list(sim._order_medium),
        "slow": list(sim._order_slow),
    }
    baseline_ledger = {
        name: list(members)
        for name, members in sim.state_manager.get_conservation_ledger().items()
    }

    with pytest.raises(RuntimeError, match="dependency graph is inconsistent"):
        sim.unregister_functor(provider.name)

    assert tuple((name, id(functor)) for name, functor in sim.functors.items()) == (
        baseline_functors
    )
    assert {name: list(members) for name, members in sim._groups.items()} == (
        baseline_groups
    )
    assert sim._bootstrap_groups == baseline_bootstrap
    assert sim._groups_dirty == baseline_groups_dirty
    assert sim.execution_order == baseline_order
    assert sim._order_fast == baseline_group_orders["fast"]
    assert sim._order_medium == baseline_group_orders["medium"]
    assert sim._order_slow == baseline_group_orders["slow"]
    assert sim.dependency_graph is inconsistent_graph
    assert not sim.dependency_graph.nodes
    assert not sim.dependency_graph.edges
    assert sim.dependency_graph.execution_order is None
    assert sim.state_manager.get_conservation_ledger() == baseline_ledger
    assert provider.name in consumer.get_dependencies()


def test_optional_provider_removal_preserves_declaration_and_rebinds() -> None:
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False}
    )
    sim.register_functor(_DependencyOutputProducer(), execution_group="fast")
    sim.register_functor(_OptionalDependencyConsumer(), execution_group="fast")

    assert sim.unregister_functor("dependency_output_producer") is True
    assert "dependency_output_producer" in sim.dependency_graph.nodes[
        "optional_dependency_consumer"
    ].dependencies

    sim.register_functor(_DependencyOutputProducer(), execution_group="fast")

    assert sim.execution_order.index("dependency_output_producer") < sim.execution_order.index(
        "optional_dependency_consumer"
    )


def test_unregister_functor_missing_returns_false_when_non_strict() -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    assert sim.unregister_functor("missing_functor", strict=False) is False


def test_unregister_functor_missing_raises_when_strict() -> None:
    sim = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    with pytest.raises(KeyError, match="is not registered"):
        sim.unregister_functor("missing_functor")


def test_functor_registration_does_not_infer_element_membership_from_outputs() -> None:
    """SCM accumulates declarations remain the only runtime membership owner."""
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    sim.set_state_manager(StateManager(config={}, device="cpu"))
    baseline = sim.state_manager.get_conservation_ledger()
    assert "leaf_carbon" in baseline["carbon"]
    assert "leaf_nitrogen" in baseline["nitrogen"]
    assert all("phenology_phase" not in members for members in baseline.values())

    sim.register_functor(_DummyMultiElementFunctor())

    assert sim.state_manager.get_conservation_ledger() == baseline

    sim.unregister_functor("dummy_multi_element_functor")
    assert sim.state_manager.get_conservation_ledger() == baseline


def test_conservation_refresh_rebuilds_instead_of_preserving_stale_members() -> None:
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    sim.set_state_manager(StateManager(config={}, device="cpu"))
    canonical = sim.state_manager.get_conservation_ledger()
    sim.state_manager.set_conservation_ledger(
        {
            "carbon": ["leaf_carbon", "phenology_phase", "leaf_nitrogen"],
            "nitrogen": ["leaf_nitrogen", "phenology_phase", "leaf_carbon"],
        }
    )

    sim._refresh_conservation_ledger()

    assert sim.state_manager.get_conservation_ledger() == canonical


def test_failed_step_restores_python_numpy_and_torch_rng_state() -> None:
    random.seed(401)
    np.random.seed(402)
    torch.manual_seed(403)
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    sim.initialize_state({"random_state": torch.tensor(1.0)})
    sim.register_functor(_RandomDrawFunctor(fail=True), execution_group="fast")
    state_before = {
        name: value.detach().clone() for name, value in sim.get_state().items()
    }
    time_before = sim.current_time
    step_before = sim.step_count
    rng_before = _capture_rng_state(sim)

    assert sim.state is SimulationState.INITIALIZED

    with pytest.raises(ExecutionError, match="injected stochastic step failure"):
        sim.step(dt=1.0)

    assert sim.state is SimulationState.INITIALIZED
    assert sim.current_time == time_before
    assert sim.step_count == step_before
    for name, value in state_before.items():
        assert torch.equal(sim.get_state()[name], value)
    _assert_rng_state_equal(rng_before, _capture_rng_state(sim))


def test_adaptive_rejection_restores_and_replays_rng_substream() -> None:
    random.seed(501)
    np.random.seed(502)
    torch.manual_seed(503)
    sim = _declared_simulator(
        {
            "conservation_checks": False,
            "strict_dependencies": False,
            "device": "cpu",
            "adaptive_dt_enabled": True,
            "adaptive_atol": 1.0e-12,
            "adaptive_rtol": 1.0e-12,
            "adaptive_dt_min": 0.1,
        }
    )
    sim.initialize_state({"random_state": torch.tensor(1.0)})
    functor = _RandomDrawFunctor(fail=False)
    sim.register_functor(functor, execution_group="fast")

    sim.step(dt=10.0)

    assert len(functor.draws) == 2
    assert functor.draws[0] == functor.draws[1]
    assert sim._adaptive_next_dt < 10.0


@pytest.mark.parametrize(
    "invalid_dt",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "infinite"],
)
def test_step_rejects_invalid_dt_without_runtime_mutation(invalid_dt: float) -> None:
    """A simulator step requires a finite, strictly positive interval."""
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    sim.initialize_state({})
    state_before = sim.state
    time_before = sim.current_time
    step_before = sim.step_count

    with pytest.raises(ValueError, match="finite and positive"):
        sim.step(dt=invalid_dt)

    assert sim.state is state_before
    assert sim.current_time == time_before
    assert sim.step_count == step_before


@pytest.mark.parametrize(
    "invalid_dt",
    [True, "3600", torch.tensor(1.0), torch.tensor([1.0])],
    ids=["boolean", "numeric-string", "scalar-tensor", "vector-tensor"],
)
def test_step_rejects_non_real_dt_without_runtime_mutation(invalid_dt) -> None:
    """The step interval must be a Python numeric real, not a coercible object."""
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    sim.initialize_state({})
    state_before = sim.state
    time_before = sim.current_time
    step_before = sim.step_count

    with pytest.raises(TypeError, match="real scalar"):
        sim.step(dt=invalid_dt)

    assert sim.state is state_before
    assert sim.current_time == time_before
    assert sim.step_count == step_before


@pytest.mark.parametrize("failure_stage", ["none", "provenance", "export"])
def test_diagnostic_export_observes_only_accepted_steps(
    failure_stage, tmp_path, monkeypatch,
) -> None:
    """External output follows commit and cannot undo accepted science."""
    sim = _declared_simulator(
        {"conservation_checks": False, "strict_dependencies": False, "device": "cpu"}
    )
    sim.initialize_state({"dependency_flux": torch.tensor(0.0)})
    sim.register_functor(_DependencyOutputProducer(), execution_group="fast")
    sim.enable_diagnostic_export = True
    sim.diagnostic_output_frequency = 1
    output = tmp_path / "accepted_step.txt"
    sim.diagnostic_output_path = output
    sim.state_manager._conservation_history.append(ConservationSummary(timestamp=0.0, total_violations=0, tolerance_value=1e-6))
    events = []
    original_commit = sim.state_manager.commit_provenance_transaction

    def commit():
        events.append("commit")
        if failure_stage == "provenance":
            raise RuntimeError("injected provenance failure")
        original_commit()

    def export(*, snapshot=None, delivery_key=None):
        from relic.data.io.writer import WriteResult
        events.append("export")
        output.write_text(str(float(sim.get_state()["dependency_flux"])), encoding="ascii")
        if failure_stage == "export":
            raise OSError("injected diagnostic export failure")
        return WriteResult(output_path=output, delivery_key=delivery_key, sha256="observed")

    monkeypatch.setattr(sim.state_manager, "commit_provenance_transaction", commit)
    monkeypatch.setattr(sim, "_export_diagnostics", export)
    if failure_stage == "provenance":
        with pytest.raises(RuntimeError, match="injected provenance failure"):
            sim.step(dt=1.0)
        assert events == ["commit"]
        assert not output.exists()
        assert sim.current_time == 0.0
        assert sim.step_count == 0
        assert float(sim.get_state()["dependency_flux"]) == 0.0
    else:
        if failure_stage == "export":
            with pytest.raises(OSError, match="injected diagnostic export failure"):
                sim.step(dt=1.0)
        else:
            sim.step(dt=1.0)
        assert events == ["commit", "export"]
        assert sim.current_time == 1.0
        assert sim.step_count == 1
        assert float(sim.get_state()["dependency_flux"]) == 2.5
        assert output.read_text(encoding="ascii") == "2.5"
    assert sim.state_manager._provenance_transaction_active is False


def _delivery_simulator(requests=(), *, substeps=1, accumulate=False, **config):
    sim = _declared_simulator({
        "device": "cpu", "conservation_checks": False,
        "strict_dependencies": False, "adaptive_dt_enabled": False, **config,
    })
    sim.initialize_state({"dependency_flux": torch.tensor(0.0)})
    sim.register_functor(_DependencyOutputProducer(requests, substeps, accumulate), execution_group="fast")
    if config.get("enable_diagnostic_export") or any(request["operation"] == "diagnostic" for request in requests):
        sim.state_manager._conservation_history.append(ConservationSummary(timestamp=0.0, total_violations=0, tolerance_value=1e-6))
    return sim


def test_explicit_checkpoint_is_delivered_only_after_accepted_commit(tmp_path, monkeypatch):
    path = tmp_path / "accepted.pt"
    sim = _delivery_simulator([{"operation": "checkpoint", "resource": str(path), "request_id": "sample"}])
    save = torch.save
    observed = []
    def observe_save(*args, **kwargs):
        assert sim.state_manager._provenance_transaction_active is False
        assert sim.step_count == 1
        assert sim.get_state()["dependency_flux"].item() == 2.5
        observed.append(sim.current_time)
        return save(*args, **kwargs)
    monkeypatch.setattr(torch, "save", observe_save)
    updates = sim.step(dt=1.0)
    assert observed == [1.0]
    assert set(updates) == {"dependency_flux"}
    assert "delivery_0" not in sim.get_state()
    payload = torch.load(path, weights_only=True)
    assert set(payload["state_manager"]["state"]) == {"dependency_flux"}
    assert payload["state_manager"]["state"]["dependency_flux"].item() == 2.5
    assert payload["runtime"]["run_id"] == sim.run_id
    assert sim._pending_external_deliveries == []
    assert len(sim._last_external_delivery_receipts) == 1
    assert not hasattr(sim, "_active_step_external_intents")


def test_failed_commit_discards_control_requests_and_preserves_run_identity(tmp_path, monkeypatch):
    path = tmp_path / "rejected.pt"
    sim = _delivery_simulator([{"operation": "checkpoint", "resource": str(path), "request_id": "sample"}])
    run_id = sim.run_id
    def fail_commit():
        raise RuntimeError("rejected provenance")
    monkeypatch.setattr(sim.state_manager, "commit_provenance_transaction", fail_commit)
    with pytest.raises(RuntimeError, match="rejected provenance"):
        sim.step(dt=1.0)
    assert not path.exists()
    assert sim.step_count == 0 and sim.current_time == 0.0
    assert sim.get_state()["dependency_flux"].item() == 0.0
    assert sim.run_id == run_id
    assert sim._pending_external_deliveries == []
    assert not hasattr(sim, "_active_step_external_intents")


def test_delivery_failure_retains_accepted_snapshot_and_retries_without_advancing(tmp_path, monkeypatch):
    from relic.core.state import UpdateItem
    path = tmp_path / "retry.pt"
    sim = _delivery_simulator([{"operation": "checkpoint", "resource": str(path), "request_id": "sample"}])
    save = torch.save
    def fail_save(*args, **kwargs):
        raise OSError("temporary disk failure")
    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="temporary disk failure"):
        sim.step(dt=1.0)
    assert sim.step_count == 1 and sim.current_time == 1.0
    assert sim.get_state()["dependency_flux"].item() == 2.5
    assert len(sim._pending_external_deliveries) == 1
    assert sim._pending_external_deliveries[0]["attempts"] == 1
    with pytest.raises(OSError, match="temporary disk failure"):
        sim.step(dt=1.0)
    assert sim.step_count == 1
    sim.state_manager.apply_updates({
        "dependency_flux": UpdateItem(torch.tensor(100.0), mode=UpdateMode.REPLACE),
    })
    monkeypatch.setattr(torch, "save", save)
    receipts = sim.retry_external_deliveries()
    assert len(receipts) == 1
    assert sim._pending_external_deliveries == []
    assert sim.current_time == 1.0 and sim.step_count == 1
    assert sim.get_state()["dependency_flux"].item() == 100.0
    assert torch.load(path, weights_only=True)["state_manager"]["state"]["dependency_flux"].item() == 2.5
    assert sim.retry_external_deliveries() == []


@pytest.mark.parametrize("operation", ["checkpoint", "diagnostic"])
def test_automatic_and_explicit_requests_share_one_delivery(tmp_path, monkeypatch, operation):
    sim = _delivery_simulator(
        enable_checkpointing=operation == "checkpoint", checkpoint_frequency=1.0,
        checkpoint_directory=str(tmp_path), enable_diagnostic_export=operation == "diagnostic",
        diagnostic_output_path=str(tmp_path / "diagnostics.nc"), diagnostic_output_frequency=1,
    )
    path = (tmp_path / sim.run_id / "checkpoint_step_000000000001.pt") if operation == "checkpoint" else tmp_path / "diagnostics.nc"
    sim.functors["dependency_output_producer"]._original_config["external_requests"] = [
        {"operation": operation, "resource": str(path), "request_id": "explicit"},
    ]
    sim.step(dt=1.0)
    assert path.exists()
    assert len(sim._last_external_delivery_receipts) == 1
    assert len(sim._last_external_delivery_receipts[0]["sources"]) == 2
    assert sim._pending_external_deliveries == []
    if operation == "checkpoint":
        assert sim.last_checkpoint_time == 1.0
        assert torch.load(path, weights_only=True)["runtime"]["last_checkpoint_time"] == 1.0


def test_adaptive_retry_discards_rejected_substep_requests(tmp_path, monkeypatch):
    from relic.core.simulator import execution_loop
    path = tmp_path / "adaptive.pt"
    sim = _delivery_simulator(
        [{"operation": "checkpoint", "resource": str(path), "request_id": "substep"}],
        substeps=2, adaptive_dt_enabled=True, adaptive_dt_min=0.1,
    )
    monkeypatch.setattr(execution_loop, "_compute_error_norm", lambda *args: 2.0)
    sim.step(dt=10.0)
    assert sim.current_time < 10.0
    sources = sim._last_external_delivery_receipts[0]["sources"]
    assert [source["substep"] for source in sources] == [0, 1]
    assert sources[1]["time"] == pytest.approx(sim.current_time / 2.0)
    assert torch.load(path, weights_only=True)["state_manager"]["state"]["dependency_flux"].item() == 2.5


def test_conflicting_delivery_parameters_reject_the_scientific_step(tmp_path):
    path = tmp_path / "conflict.nc"
    sim = _delivery_simulator([
        {"operation": "diagnostic", "resource": str(path), "request_id": "one", "payload": {"reference_date": "2000-01-01"}},
        {"operation": "diagnostic", "resource": str(path), "request_id": "two", "payload": {"reference_date": "2001-01-01"}},
    ])
    with pytest.raises(ValueError, match="conflicting|conflict"):
        sim.step(dt=1.0)
    assert sim.step_count == 0 and sim.current_time == 0.0
    assert sim.get_state()["dependency_flux"].item() == 0.0
    assert sim._pending_external_deliveries == []
    assert not path.exists()


def test_different_output_formats_cannot_share_one_resource(tmp_path):
    path = tmp_path / "shared.bin"
    sim = _delivery_simulator([
        {"operation": "checkpoint", "resource": str(path), "request_id": "checkpoint"},
        {"operation": "diagnostic", "resource": str(path), "request_id": "diagnostic"},
    ])
    with pytest.raises(ValueError, match="conflict"):
        sim.step(dt=1.0)
    assert sim.step_count == 0
    assert sim._pending_external_deliveries == []
    assert not path.exists()


def test_pending_periodic_output_is_retried_before_the_next_scientific_step(tmp_path, monkeypatch):
    from relic.data.io.writer import DataWriter
    path = tmp_path / "periodic.nc"
    sim = _delivery_simulator(enable_diagnostic_export=True, diagnostic_output_path=str(path), diagnostic_output_frequency=1)
    write = DataWriter._write_with_format
    with monkeypatch.context() as local:
        def fail_write(*args, **kwargs):
            raise OSError("temporary diagnostic failure")
        local.setattr(DataWriter, "_write_with_format", fail_write)
        with pytest.raises(OSError, match="temporary diagnostic failure"):
            sim.step(dt=1.0)
    observed_steps = []
    def observe_write(*args, **kwargs):
        observed_steps.append(sim.step_count)
        return write(*args, **kwargs)
    monkeypatch.setattr(DataWriter, "_write_with_format", observe_write)
    sim.step(dt=1.0)
    assert observed_steps == [1, 2]
    assert sim.step_count == 2
    assert sim._pending_external_deliveries == []


@pytest.mark.parametrize("phase", ["finalize", "prepare", "prepare_second", "before_commit"])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_unaccepted_delivery_batch_rolls_back_all_state_and_rng(tmp_path, monkeypatch, phase, failure_type):
    from relic.core.simulator import checkpoint, execution_loop

    paths = [tmp_path / "first.pt", tmp_path / "second.pt"]
    sim = _delivery_simulator([
        {"operation": "checkpoint", "resource": str(path), "request_id": path.stem}
        for path in paths
    ])
    before_state = sim.state_manager.checkpoint()
    before_runtime = execution_loop._snapshot_step_runtime(sim)
    before_rng = checkpoint._capture_rng_state(sim)
    finalize = execution_loop._finalize_step
    capture = execution_loop.capture_checkpoint_payload
    project = execution_loop.checkpoint_delivery_snapshot
    capture_count = 0
    projection_count = 0

    def finalize_with_draws(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(())
        finalize(*args, **kwargs)
        if phase == "finalize":
            raise failure_type("injected preparation interruption")

    def capture_with_failure(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        if phase == "prepare":
            raise failure_type("injected preparation interruption")
        return capture(*args, **kwargs)

    def project_with_failure(*args, **kwargs):
        nonlocal projection_count
        projection_count += 1
        if phase == "prepare_second" and projection_count == 2:
            raise failure_type("injected preparation interruption")
        return project(*args, **kwargs)

    monkeypatch.setattr(execution_loop, "_finalize_step", finalize_with_draws)
    monkeypatch.setattr(execution_loop, "capture_checkpoint_payload", capture_with_failure)
    monkeypatch.setattr(execution_loop, "checkpoint_delivery_snapshot", project_with_failure)
    if phase == "before_commit":
        def reject_commit():
            raise failure_type("injected preparation interruption")
        monkeypatch.setattr(sim.state_manager, "commit_provenance_transaction", reject_commit)
    with pytest.raises(failure_type, match="injected preparation interruption"):
        sim.step(dt=1.0)
    _assert_nested_state_equal(before_state, sim.state_manager.checkpoint())
    for name in ("run_id", "current_time", "step_count", "total_execution_time", "_step_update_sources"):
        if name in before_runtime:
            assert getattr(sim, name) == before_runtime[name]
        else:
            assert not hasattr(sim, name)
    _assert_rng_state_equal(before_rng, checkpoint._capture_rng_state(sim))
    assert sim._pending_external_deliveries == []
    assert not any(path.exists() for path in paths)
    assert not hasattr(sim, "_active_step_external_intents")
    if phase == "prepare_second":
        assert capture_count == 1 and projection_count == 2


@pytest.mark.parametrize("phase", ["after_commit", "cleanup", "delivery"])
def test_accepted_cancellation_keeps_complete_delivery_for_retry(tmp_path, monkeypatch, phase):
    from relic.core.simulator import checkpoint, execution_loop

    path = tmp_path / "accepted_cancel.pt"
    sim = _delivery_simulator(
        [{"operation": "checkpoint", "resource": str(path), "request_id": "cancel"}],
        enable_profiling=True,
    )
    commit = sim.state_manager.commit_provenance_transaction
    set_report = sim.state_manager.set_auto_report_enabled
    accepted_rng = None

    def commit_then_cancel():
        nonlocal accepted_rng
        commit()
        accepted_rng = checkpoint._capture_rng_state(sim)
        if phase == "after_commit":
            raise KeyboardInterrupt("accepted interruption")

    def restore_report_then_cancel(enabled):
        set_report(enabled)
        if phase == "cleanup" and enabled:
            raise KeyboardInterrupt("accepted interruption")

    with monkeypatch.context() as local:
        local.setattr(sim.state_manager, "commit_provenance_transaction", commit_then_cancel)
        local.setattr(sim.state_manager, "set_auto_report_enabled", restore_report_then_cancel)
        if phase == "delivery":
            def cancel_save(*args, **kwargs):
                raise KeyboardInterrupt("accepted interruption")
            local.setattr(torch, "save", cancel_save)
        with pytest.raises(KeyboardInterrupt, match="accepted interruption"):
            sim.step(dt=1.0)
    assert sim.step_count == 1 and sim.current_time == 1.0
    assert sim.get_state()["dependency_flux"].item() == 2.5
    _assert_rng_state_equal(accepted_rng, checkpoint._capture_rng_state(sim))
    assert sim.state_manager._provenance_transaction_active is False
    assert sim.state_manager._enable_auto_report is True
    assert sim._step_update_sources is None
    assert not hasattr(sim, "_active_step_stage_timings")
    assert sim.functors["dependency_output_producer"]._performance_profiling_enabled is False
    assert sim._performance_diagnostics is not None
    assert sim._performance_diagnostics["accepted_steps"] == 1
    assert len(sim._pending_external_deliveries) == 1
    assert sim._pending_external_deliveries[0]["snapshot"] is not None
    assert not path.exists()
    execution_loop.retry_external_deliveries(sim)
    assert sim._pending_external_deliveries == []
    assert torch.load(path, weights_only=True)["state_manager"]["state"]["dependency_flux"].item() == 2.5
    assert sim.step_count == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows resource identity")
@pytest.mark.parametrize("conflict", [True, False])
def test_windows_path_aliases_have_one_delivery_identity(tmp_path, conflict):
    path = tmp_path / "sharedcase.nc"
    sim = _delivery_simulator([
        {"operation": "diagnostic", "resource": str(path), "request_id": "first", "payload": {"reference_date": "2000-01-01"}},
        {"operation": "diagnostic", "resource": str(path.with_name("SharedCase.nc")), "request_id": "second",
         "payload": {"reference_date": "2001-01-01" if conflict else "2000-01-01"}},
    ])
    if conflict:
        with pytest.raises(ValueError, match="conflict"):
            sim.step(dt=1.0)
        assert sim.step_count == 0 and not path.exists()
    else:
        sim.step(dt=1.0)
        assert len(sim._last_external_delivery_receipts) == 1
        assert len(sim._last_external_delivery_receipts[0]["sources"]) == 2
        assert path.exists()
    assert sim._pending_external_deliveries == []


def test_existing_checkpoint_target_rejects_new_step_before_acceptance(tmp_path):
    path = tmp_path / "fixed.pt"
    sim = _delivery_simulator([{"operation": "checkpoint", "resource": str(path), "request_id": "fixed"}])
    sim.step(dt=1.0)
    content = path.read_bytes()
    with pytest.raises(FileExistsError, match="checkpoint|Checkpoint"):
        sim.step(dt=1.0)
    assert sim.step_count == 1 and sim.current_time == 1.0
    assert sim.get_state()["dependency_flux"].item() == 2.5
    assert sim._pending_external_deliveries == []
    assert path.read_bytes() == content


def test_invalid_diagnostic_epoch_rejects_step_without_pending_work(tmp_path):
    from relic.coupling.scales.temporal.core import ValidationError
    path = tmp_path / "bad_epoch.nc"
    sim = _delivery_simulator([{"operation": "diagnostic", "resource": str(path), "request_id": "epoch",
                               "payload": {"reference_date": "2001-02-29"}}])
    with pytest.raises(ValidationError, match="reference time"):
        sim.step(dt=1.0)
    assert sim.step_count == 0 and sim.current_time == 0.0
    assert sim.get_state()["dependency_flux"].item() == 0.0
    assert sim._pending_external_deliveries == [] and not path.exists()


@pytest.mark.parametrize("invalid", ["nonscalar", "collision"])
def test_invalid_diagnostic_data_rejects_science_before_delivery(tmp_path, invalid):
    from xarray import MergeError
    path = tmp_path / "uncreated" / "bad_data.nc"
    sim = _delivery_simulator([{"operation": "diagnostic", "resource": str(path), "request_id": "bad-data"}])
    sim.state_manager._conservation_history.clear()
    if invalid == "nonscalar":
        summaries = [ConservationSummary(timestamp=0.0, custom_checks={"vector": torch.tensor([1.0, 2.0])}, total_violations=0, tolerance_value=1e-6)]
    else:
        summaries = [
            ConservationSummary(timestamp=0.0, custom_checks={"a.b": 1.0}, total_violations=0, tolerance_value=1e-6),
            ConservationSummary(timestamp=1.0, custom_checks={"a_b": 2.0}, total_violations=0, tolerance_value=1e-6),
        ]
    sim.state_manager._conservation_history.extend(summaries)
    before_rng = _capture_rng_state(sim)
    with pytest.raises((TypeError, ValueError, MergeError)):
        sim.step(dt=1.0)
    assert sim.step_count == 0 and sim.current_time == 0.0
    assert sim.get_state()["dependency_flux"].item() == 0.0
    _assert_rng_state_equal(before_rng, _capture_rng_state(sim))
    assert sim._pending_external_deliveries == []
    assert not path.parent.exists()


@pytest.mark.parametrize("accumulate", [False, True])
def test_delivery_retry_confirms_accepted_notifications_before_writing(tmp_path, monkeypatch, accumulate):
    import json
    from relic.core.provenance import ProvenanceRecorder
    from relic.core.simulator.execution_loop import retry_external_deliveries
    path = tmp_path / "after_notifications.pt"
    sim = _delivery_simulator(
        [{"operation": "checkpoint", "resource": str(path), "request_id": "notify"}], accumulate=accumulate,
    )
    events = []
    available = False
    recorder = ProvenanceRecorder(tmp_path / "provenance")
    record_update = recorder.record_state_update

    def notify(*args, **kwargs):
        events.append("notification")
        if not available:
            raise OSError("notification not confirmed")
        return record_update(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_state_update", notify)
    sim.state_manager.set_provenance_recorder(recorder)
    save = torch.save
    def write_after_notification(*args, **kwargs):
        assert sim.state_manager._pending_provenance_events == []
        events.append("write")
        return save(*args, **kwargs)
    monkeypatch.setattr(torch, "save", write_after_notification)
    with pytest.raises(RuntimeError, match="notification"):
        sim.step(dt=1.0)
    assert sim.step_count == 1 and sim.current_time == 1.0
    assert len(sim._pending_external_deliveries) == 1
    with pytest.raises(RuntimeError, match="notification"):
        retry_external_deliveries(sim)
    assert not path.exists()
    assert sim._pending_external_deliveries[0]["attempts"] == 0
    available = True
    retry_external_deliveries(sim)
    assert events == ["notification", "notification", "notification", "write"]
    assert sim.step_count == 1 and sim.current_time == 1.0
    assert sim.get_state()["dependency_flux"].item() == 2.5
    assert sim._pending_external_deliveries == []
    assert path.exists()
    records = [json.loads(line) for line in (tmp_path / "provenance" / "runtime.log.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["run_id"] == sim.run_id


@pytest.mark.parametrize("accumulate", [False, True])
def test_old_notification_prefix_is_confirmed_before_new_step_savepoint(monkeypatch, accumulate):
    from types import SimpleNamespace
    from relic.core.simulator.execution_loop import retry_external_deliveries
    sim = _delivery_simulator(accumulate=accumulate)
    sim.register_functor(_DependencyOutputConsumer(), execution_group="fast")
    phase = "first"
    completed = []

    def notify(step, variable, *args):
        if phase == "first" or (phase == "tail" and variable == "dependency_total"):
            raise OSError("notification still unavailable")
        completed.append(variable)

    sim.state_manager.set_provenance_recorder(SimpleNamespace(record_state_update=notify))
    with pytest.raises(RuntimeError, match="notification"):
        sim.step(dt=1.0)
    assert len(sim.state_manager._pending_provenance_events) == 2
    phase = "tail"
    snapshot = sim.state_manager.create_transaction_savepoint
    snapshots = []
    def count_snapshot():
        snapshots.append(True)
        return snapshot()
    monkeypatch.setattr(sim.state_manager, "create_transaction_savepoint", count_snapshot)
    with pytest.raises(RuntimeError, match="notification"):
        sim.step(dt=1.0)
    assert snapshots == []
    assert completed == ["dependency_flux"]
    assert [event["variable"] for event in sim.state_manager._pending_provenance_events] == ["dependency_total"]
    assert sim.step_count == 1 and sim.current_time == 1.0
    phase = "ready"
    retry_external_deliveries(sim)
    assert completed == ["dependency_flux", "dependency_total"]
    assert sim.state_manager._pending_provenance_events == []
    assert sim.step_count == 1


def test_delivery_retry_cannot_accept_an_active_scientific_transaction():
    from relic.core.simulator.execution_loop import retry_external_deliveries
    sim = _delivery_simulator()
    sim.state_manager.begin_provenance_transaction()
    with pytest.raises(RuntimeError, match="accepted|active"):
        retry_external_deliveries(sim)
    assert sim.state_manager._provenance_transaction_active is True
    assert sim.step_count == 0


def test_reset_preserves_accepted_runtime_until_notifications_are_confirmed(tmp_path, monkeypatch):
    import json
    from relic.core.provenance import ProvenanceRecorder
    path = tmp_path / "before_reset.pt"
    sim = _delivery_simulator(
        [{"operation": "checkpoint", "resource": str(path), "request_id": "reset"}],
        output_frequency=1, enable_profiling=True,
    )
    available = False
    recorder = ProvenanceRecorder(tmp_path / "provenance")
    record_update = recorder.record_state_update
    def notify(*args, **kwargs):
        if not available:
            raise OSError("reset notification unavailable")
        return record_update(*args, **kwargs)
    monkeypatch.setattr(recorder, "record_state_update", notify)
    sim.state_manager.set_provenance_recorder(recorder)
    with pytest.raises(RuntimeError, match="notification"):
        sim.step(dt=1.0)
    before = {name: getattr(sim, name) for name in (
        "run_id", "state", "current_time", "step_count", "total_execution_time", "last_checkpoint_time",
    )}
    series = copy.deepcopy(sim.time_series)
    performance = copy.deepcopy(sim._performance_diagnostics)
    with pytest.raises(RuntimeError, match="notification"):
        sim.reset()
    assert {name: getattr(sim, name) for name in before} == before
    _assert_nested_state_equal(series, sim.time_series)
    _assert_nested_state_equal(performance, sim._performance_diagnostics)
    assert sim.get_state()["dependency_flux"].item() == 2.5
    assert len(sim._pending_external_deliveries) == 1
    available = True
    sim.retry_external_deliveries()
    sim.reset()
    assert sim.run_id != before["run_id"]
    assert sim.current_time == 0.0 and sim.step_count == 0
    assert sim.get_state()["dependency_flux"].item() == 0.0
    assert path.exists()
    records = [json.loads(line) for line in (tmp_path / "provenance" / "runtime.log.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["run_id"] == before["run_id"]


@pytest.mark.parametrize("auto_report", [True, False])
def test_prepared_checkpoint_matches_the_actual_accepted_boundary(tmp_path, auto_report):
    from relic.core.simulator import checkpoint
    path = tmp_path / "equivalent.pt"
    sim = _delivery_simulator([{"operation": "checkpoint", "resource": str(path), "request_id": "equivalent"}])
    sim.state_manager.set_auto_report_enabled(auto_report)
    sim.step(dt=1.0)
    prepared = torch.load(path, weights_only=True)
    actual = checkpoint.capture_checkpoint_payload(sim)
    assert len(prepared["delivery_state"]["pending"]) == 1
    assert actual["delivery_state"]["pending"] == []
    key = prepared["delivery_state"]["pending"][0]["key"]
    assert actual["delivery_state"]["receipts"][0]["key"] == key
    for payload in (prepared, actual):
        payload.pop("created_utc")
        payload.pop("integrity")
        payload.pop("delivery_state")
    _assert_nested_state_equal(prepared, actual)


def test_accepted_checkpoint_contains_all_same_step_deliveries_for_restart(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys
    from relic.data.io.writer import DataWriter

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    diagnostic = tmp_path / "diagnostic.nc"
    requests = [
        {"operation": "checkpoint", "resource": str(first), "request_id": "first"},
        {"operation": "diagnostic", "resource": str(diagnostic), "request_id": "diagnostic"},
        {"operation": "checkpoint", "resource": str(second), "request_id": "second"},
    ]
    source = _delivery_simulator(requests)

    def fail_diagnostic(*args, **kwargs):
        raise OSError("diagnostic delivery unavailable")

    with monkeypatch.context() as local:
        local.setattr(DataWriter, "write_dataset", fail_diagnostic)
        with pytest.raises(OSError, match="diagnostic delivery unavailable"):
            source.step(dt=1.0)
    assert first.exists() and not second.exists() and not diagnostic.exists()
    assert source.step_count == 1
    assert len(source._pending_external_deliveries) == 2
    expected_second = source._pending_external_deliveries[-1]["snapshot"]["integrity"]

    script = '''
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[3])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
simulator = module._delivery_simulator(json.loads(sys.argv[4]))
simulator.restore_checkpoint(sys.argv[1])
simulator.retry_external_deliveries()
print(json.dumps({"time": simulator.current_time, "step": simulator.step_count, "value": float(simulator.get_state()["dependency_flux"]), "pending": len(simulator._pending_external_deliveries), "receipts": len(simulator._last_external_delivery_receipts)}))
'''
    for checkpoint_path in (first, second):
        result = subprocess.run(
            [sys.executable, "-c", script, str(checkpoint_path), __name__, __file__, json.dumps(requests)],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=120,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
        assert json.loads(result.stdout.strip().splitlines()[-1]) == {
            "time": 1.0, "step": 1, "value": 2.5, "pending": 0, "receipts": 3,
        }
        assert torch.load(second, weights_only=True)["integrity"] == expected_second
        assert diagnostic.exists()


class _MisdeclaredDependencyConsumer(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, declaration):
        metadata = FunctorMetadata(
            name="misdeclared_dependency_consumer",
            version="1.0.0",
            description="dependency declaration contract fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["misdeclared_total"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)
        self.declaration = declaration

    def get_dependencies(self):
        return self.declaration

    def forward(self, state, time, dt, **kwargs):
        return {"misdeclared_total": torch.tensor(1.0)}


@pytest.mark.parametrize(
    ("declaration", "error", "match"),
    [
        ({"dependency_output_producer": "optional"}, ValueError, "unknown dependency categories"),
        ({"required": ["dependency_output_producer"], "lagged": ["dependency_output_producer"]},
         ValueError, "more than one category"),
        ({"required": [7]}, TypeError, "non-string"),
    ],
)
def test_registration_rejects_misdeclared_dependencies_before_mutation(declaration, error, match):
    simulator = _declared_simulator({"device": "cpu", "conservation_checks": False})
    simulator.initialize_state({"misdeclared_total": torch.tensor(0.0)})
    epoch = simulator.provider_epoch

    with pytest.raises(error, match=match):
        simulator.register_functor(_MisdeclaredDependencyConsumer(declaration))

    assert "misdeclared_dependency_consumer" not in simulator.functors
    assert simulator.provider_epoch == epoch
    assert all("misdeclared_dependency_consumer" not in members for members in simulator._groups.values())

def test_sort_failure_is_not_a_warning_in_non_strict_mode(monkeypatch):
    """Non-strict mode tolerates dependencies registered later, not a failing sort."""
    simulator = _declared_simulator({"conservation_checks": False, "strict_dependencies": False})
    simulator.set_state_manager(StateManager(config={}, device="cpu"))
    simulator.register_functor(_DependencyOutputProducer(), execution_group="fast")

    def _broken(*args, **kwargs):
        raise ValueError("graph bookkeeping is inconsistent")

    monkeypatch.setattr(simulator.dependency_graph, "topological_sort", _broken)
    with pytest.raises(ValueError, match="bookkeeping is inconsistent"):
        simulator.register_functor(_DependencyOutputConsumer(), execution_group="fast")
