"""CR-A-04 T1: governed admissibility of the fast-group splitting configuration.

Design 20261019_cr_a04_design_01, sections 2-4: the scheduling controls are
validated when the simulator is built (D1: no silent Lie fallback), and every
provider-view publication resolves the admissibility of the configured
composition before the view becomes visible:

- S1: under Strang an SCM flux contract that carries a transfer from one fast
  member that holds state to another (directly, or through fast members without
  persistent state) is refused. In the palindrome the provider releases
  (q1 + q2) h / 2 over its two half-steps while the consumer integrates the value
  it read, so the transfer leaks (q1 - q2) h / 2 per step (D3; snow_pack ->
  bucket via snow_melt_flux and rain_flux). Couplings through shared state
  (nitrification and denitrification on no3_nitrogen) are admissible.
- S3: splitting_policy require_declared refuses Strang while a fast member that
  holds (or is not shown not to hold) state declares no sub_flow_accuracy.
- S5: fast_order reverse is admissible only without same-step dependency edges
  among the fast members.
- A refused publication leaves the committed provider view unchanged.

References: Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and
Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.5; McLachlan and
Quispel (2002) Acta Numerica 11:341-434; Gear and Wells (1984) BIT 24(4):484-502.
Mathematical expression (fixtures): x(t + dt) = x(t) exp(-k dt), the exact
relaxation sub-flow of one variable (EXACT sub-flow accuracy when declared).
Parameter meanings: x a dimensionless test state; k = _RELAXATION_RATE [s-1].
Units: x dimensionless; t, dt [s]; k [s-1].
Domain/constraints: k > 0, dt > 0; test-only uncontracted variables declared through
the state_units setting.
Sign convention: relaxation decreases |x|.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pytest
import torch

from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorMetadata,
    SubFlowAccuracy,
    TemporalScale,
    VariableKind,
)
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager

_SCHEDULER_LOGGER = "relic.core.simulator.execution_loop"
_RELAXATION_RATE = 1.0e-3  # k [s-1], test fixture constant
_FIXTURE_UNITS = {name: "dimensionless" for name in ("relax_a", "relax_b", "relax_c", "relax_flux")}


class _Relaxation(BiologicalProcessFunctor):
    """Exact relaxation x <- x exp(-k dt) of one test variable, or a constant output.

    With ``reads_own_state`` the Functor reads and rewrites ``variable`` (a
    recurrence, i.e. persistent state); without it the Functor returns the
    constant 1 for ``variable`` (no persistent state).
    """

    def __init__(
        self,
        name: str,
        variable: str,
        *,
        variable_kind: VariableKind = VariableKind.STATE,
        accuracy: SubFlowAccuracy = SubFlowAccuracy.UNDECLARED,
        reads_own_state: bool = True,
        dependencies: Optional[Dict[str, List[str]]] = None,
        call_log: Optional[List[tuple]] = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            description=f"Relaxation test fixture {name}",
            references=["Hairer, Lubich and Wanner (2006) Geometric Numerical Integration, Sect. II.5"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=[variable] if reads_own_state else [],
            output_variables=[variable],
            variable_kind=variable_kind,
            sub_flow_accuracy=accuracy,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.execution_group = "fast"
        self.variable = variable
        self.reads_own_state = reads_own_state
        self._declared_dependencies = dependencies or {"required": [], "optional": [], "lagged": []}
        self.call_log = call_log if call_log is not None else []

    def get_dependencies(self) -> Dict[str, List[str]]:
        return {key: list(value) for key, value in self._declared_dependencies.items()}

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        # References: Hairer, Lubich and Wanner (2006), Sect. II.5 (exact sub-flows).
        # Mathematical expression: x(t + dt) = x(t) exp(-k dt); constant output 1 otherwise.
        # Parameter meanings: x the fixture state; k = _RELAXATION_RATE.
        # Units: x dimensionless; dt [s]; k [s-1].
        # Domain/constraints: dt > 0.
        # Sign convention: relaxation decreases |x|.
        self.call_log.append((self.name, float(time), float(dt.item())))
        if self.reads_own_state:
            value = state[self.variable] * math.exp(-_RELAXATION_RATE * float(dt.item()))
        else:
            value = torch.tensor(1.0, dtype=torch.float64)
        return {
            self.variable: UpdateItem(
                value=torch.as_tensor(value, dtype=torch.float64),
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                source_functor=self.name,
            )
        }


def _fixture_simulator(**config) -> RELICSimulator:
    settings = {"time_step": 60.0, "device": "cpu", "state_units": dict(_FIXTURE_UNITS)}
    settings.update(config)
    simulator = RELICSimulator(settings)
    simulator.initialize_state({"relax_a": 1.0, "relax_b": 2.0, "relax_c": 3.0, "relax_flux": 0.0})
    return simulator


def _view_record(simulator: RELICSimulator) -> Dict[str, Any]:
    return {
        "view": simulator.get_committed_provider_view(),
        "epoch": simulator.provider_epoch,
        "functors": tuple(simulator.functors.keys()),
        "groups": {name: tuple(members) for name, members in simulator._groups.items()},
        "order_fast": tuple(simulator._order_fast),
        "receipts": len(simulator.get_lifecycle_receipts()),
    }


def _assert_publication_rolled_back(simulator: RELICSimulator, before: Dict[str, Any]) -> None:
    """A refused publication leaves the committed view and the registry unchanged."""
    assert simulator.get_committed_provider_view() is before["view"]
    assert simulator.provider_epoch == before["epoch"]
    assert tuple(simulator.functors.keys()) == before["functors"]
    assert {name: tuple(members) for name, members in simulator._groups.items()} == before["groups"]
    assert tuple(simulator._order_fast) == before["order_fast"]
    receipts = simulator.get_lifecycle_receipts()
    assert len(receipts) == before["receipts"] + 1
    assert receipts[-1].status == "rolled_back"


# --------------------------------------------------------------------------
# D1 and configuration validation (main.py).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["palindromic", "strang2", "", 2, True])
def test_unknown_splitting_mode_is_refused_not_run_as_lie(value):
    with pytest.raises(ValueError, match="splitting_mode"):
        RELICSimulator({"splitting_mode": value})


def test_splitting_controls_default_to_the_parameter_database_and_are_normalized():
    manager = get_global_manager()
    simulator = RELICSimulator({})
    assert simulator._splitting_mode == manager.get("computation.scheduling.splitting_mode") == "lie"
    assert simulator._splitting_policy == manager.get("computation.scheduling.splitting_policy") == "warn"
    assert simulator._fast_order_mode == manager.get("computation.scheduling.fast_order") == "forward"
    assert simulator._substep_overrides == {}
    assert simulator._multirate_exchanges == ()
    configured = RELICSimulator(
        {"splitting_mode": " Strang ", "splitting_policy": "REQUIRE_DECLARED", "fast_order": "Reverse"}
    )
    assert configured._splitting_mode == "strang"
    assert configured._splitting_policy == "require_declared"
    assert configured._fast_order_mode == "reverse"


@pytest.mark.parametrize(
    "setting, value",
    [("splitting_policy", "strict"), ("splitting_policy", 1), ("fast_order", "backward"), ("fast_order", "")],
)
def test_unknown_policy_or_fast_order_is_refused(setting, value):
    with pytest.raises(ValueError, match=setting):
        RELICSimulator({setting: value})


@pytest.mark.parametrize(
    "value",
    [
        [("bucket_soil_moisture", 2)],
        {"": {"count": 2}},
        {"bucket_soil_moisture": 2},
        {"bucket_soil_moisture": {}},
        {"bucket_soil_moisture": {"count": 0}},
        {"bucket_soil_moisture": {"count": 2.5}},
        {"bucket_soil_moisture": {"count": True}},
        {"bucket_soil_moisture": {"max_seconds": -1.0}},
        {"bucket_soil_moisture": {"max_seconds": float("inf")}},
        {"bucket_soil_moisture": {"max_seconds": float("nan")}},
        {"bucket_soil_moisture": {"count": 2, "max_seconds": 60.0}},
        {"bucket_soil_moisture": {"steps": 2}},
    ],
)
def test_malformed_substep_overrides_are_refused(value):
    with pytest.raises(ValueError, match="substep_overrides"):
        RELICSimulator({"substep_overrides": value})


_EXCHANGE = {"id": "bucket_water", "variable": "soil_moisture", "consumer_group": "medium", "mode": "interval_mean"}


@pytest.mark.parametrize(
    "value",
    [
        {"id": "x"},
        [42],
        [{key: value for key, value in _EXCHANGE.items() if key != "mode"}],
        [dict(_EXCHANGE, extra=1)],
        [dict(_EXCHANGE, id="")],
        [dict(_EXCHANGE, variable="")],
        [dict(_EXCHANGE, consumer_group="fast")],
        [dict(_EXCHANGE, mode="average")],
        [_EXCHANGE, dict(_EXCHANGE, variable="snow_water_equivalent")],
        [_EXCHANGE, dict(_EXCHANGE, id="bucket_water_2")],
        # The id names the StateManager mean variable and its owner identity.
        [dict(_EXCHANGE, id="bucket water")],
        [dict(_EXCHANGE, id="bucket.water")],
        [dict(_EXCHANGE, id="2bucket")],
    ],
)
def test_malformed_multirate_exchanges_are_refused(value):
    with pytest.raises(ValueError, match="multirate_exchanges"):
        RELICSimulator({"multirate_exchanges": value})


def test_interpolating_multirate_exchange_is_refused():
    with pytest.raises(ValueError, match="interpolat"):
        RELICSimulator({"multirate_exchanges": [dict(_EXCHANGE, mode="interpolate")]})


def test_well_formed_overrides_and_exchanges_are_accepted_in_canonical_form():
    """Valid overrides and exchanges are kept in the canonical form the simulator writes.

    Sub-step overrides are executed since CR-A-04 step 4 (registration binding and
    execution_group; test/core/splitting_substep_overrides.test.py) and multirate
    exchanges since step 6 (execution_loop and execution_group;
    test/core/splitting_multirate_exchanges.test.py).
    """
    count = RELICSimulator({"substep_overrides": {"bucket_soil_moisture": {"count": 2}}})
    assert count._substep_overrides == {"bucket_soil_moisture": {"count": 2}}
    seconds = RELICSimulator({"substep_overrides": {"bucket_soil_moisture": {"max_seconds": 600}}})
    assert seconds._substep_overrides == {"bucket_soil_moisture": {"max_seconds": 600.0}}
    assert type(seconds._substep_overrides["bucket_soil_moisture"]["max_seconds"]) is float
    hold = dict(_EXCHANGE, id="bucket_water_slow", mode="hold", consumer_group="slow")
    exchanges = RELICSimulator({"multirate_exchanges": [dict(_EXCHANGE), hold]})
    assert exchanges._multirate_exchanges == (dict(_EXCHANGE), hold)
    assert type(exchanges._multirate_exchanges) is tuple
    empty = RELICSimulator({"substep_overrides": {}, "multirate_exchanges": []})
    assert empty._substep_overrides == {} and empty._multirate_exchanges == ()


# --------------------------------------------------------------------------
# S1: flux transfers between fast members that hold state (real Functors).
# --------------------------------------------------------------------------

_RAIN_AND_MELT = {"water_input_source": "rain_flux_plus_snow_melt_flux"}


def _snow_bucket_simulator(splitting_mode: str) -> RELICSimulator:
    simulator = RELICSimulator(
        {
            "time_step": 3600.0,
            "device": "cpu",
            "splitting_mode": splitting_mode,
            "state_units": {"sublimation_flux": "kg m-2 s-1"},
        }
    )
    simulator.initialize_state(
        {
            "precipitation_flux": torch.tensor([3.0e-5]),
            "air_temperature": torch.tensor([278.15]),
            "snow_water_equivalent": torch.tensor([10.0]),
            "sublimation_flux": torch.tensor([0.0]),
            "soil_moisture": torch.tensor([0.25]),
            "evapotranspiration_flux": torch.tensor([0.0]),
            "rain_flux": torch.tensor([0.0]),
            "snow_melt_flux": torch.tensor([0.0]),
        }
    )
    return simulator


@pytest.mark.parametrize("first", ["snow_pack", "bucket_soil_moisture"])
def test_strang_refuses_a_flux_transfer_between_two_stateful_fast_members(first):
    from relic.scientific.hydrosphere.snow.snowpack import SnowPack
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    simulator = _snow_bucket_simulator("strang")
    members = {
        "snow_pack": SnowPack(),
        "bucket_soil_moisture": BucketSoilMoisture(config=dict(_RAIN_AND_MELT)),
    }
    second = next(name for name in members if name != first)
    simulator.register_functor(members[first], execution_group="fast")
    before = _view_record(simulator)
    with pytest.raises(RuntimeError, match=r"S1") as refusal:
        simulator.register_functor(members[second], execution_group="fast")
    message = str(refusal.value)
    for token in ("snow_pack", "bucket_soil_moisture", "snow_melt_flux", "rain_flux"):
        assert token in message
    _assert_publication_rolled_back(simulator, before)


def test_lie_admits_and_flags_the_same_flux_transfer():
    from relic.scientific.hydrosphere.snow.snowpack import SnowPack
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    simulator = _snow_bucket_simulator("lie")
    simulator.register_functor(SnowPack(), execution_group="fast")
    simulator.register_functor(BucketSoilMoisture(config=dict(_RAIN_AND_MELT)), execution_group="fast")
    report = simulator.get_splitting_order_report()
    assert report["member_class"] == {"snow_pack": "stateful", "bucket_soil_moisture": "stateful"}
    flagged = report["refused_or_flagged_edges"]
    assert len(flagged) == 1
    entry = flagged[0]
    assert entry["rule"] == "S1" and entry["status"] == "flagged"
    assert entry["provider"] == "snow_pack" and entry["consumer"] == "bucket_soil_moisture"
    assert set(entry["variables"]) == {"rain_flux", "snow_melt_flux"}


def test_strang_admits_the_transfer_when_the_provider_runs_in_a_slower_group():
    """Only the fast members form the palindrome; a medium snow pack is a held provider."""
    from relic.scientific.hydrosphere.snow.snowpack import SnowPack
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    simulator = _snow_bucket_simulator("strang")
    simulator.register_functor(SnowPack(), execution_group="medium")
    simulator.register_functor(BucketSoilMoisture(config=dict(_RAIN_AND_MELT)), execution_group="fast")
    assert simulator.get_splitting_order_report()["refused_or_flagged_edges"] == ()


def test_strang_refuses_a_transfer_relayed_by_a_member_without_state():
    """Stateful phenology -> litterfall mapper (no state) -> stateful decomposition."""
    from relic.scientific.biosphere.phenology.models import StaticPhenology
    from relic.scientific.pedosphere.decomposition import FirstOrderDecomposition
    from relic.scientific.pedosphere.interface_mapping import LitterfallInterfaceMapper

    simulator = RELICSimulator({"device": "cpu", "splitting_mode": "strang"})
    simulator.initialize_state({"leaf_carbon": torch.tensor(2.0)})
    simulator.register_functor(StaticPhenology(name="static_phenology"), execution_group="fast")
    simulator.register_functor(
        LitterfallInterfaceMapper(
            name="litterfall_interface_mapper", config={"dependencies": ["static_phenology"]}
        ),
        execution_group="fast",
    )
    assert simulator.get_splitting_order_report()["member_class"]["litterfall_interface_mapper"] == "stateless"
    before = _view_record(simulator)
    with pytest.raises(RuntimeError, match=r"S1") as refusal:
        simulator.register_functor(
            FirstOrderDecomposition(
                name="first_order_decomposition", config={"dependencies": ["litterfall_interface_mapper"]}
            ),
            execution_group="fast",
        )
    message = str(refusal.value)
    assert "static_phenology" in message and "first_order_decomposition" in message
    assert "litterfall_interface_mapper" in message
    _assert_publication_rolled_back(simulator, before)


def test_strang_admits_a_state_coupled_transfer():
    """Nitrification moves NH4 to NO3 inside one call; denitrification reads the NO3 state."""
    from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess
    from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess

    simulator = RELICSimulator({"device": "cpu", "splitting_mode": "strang"})
    simulator.initialize_state({"nh4_nitrogen": torch.tensor(0.1), "no3_nitrogen": torch.tensor(0.2)})
    simulator.register_functor(NitrificationProcess(), execution_group="fast")
    simulator.register_functor(DenitrificationProcess(), execution_group="fast")
    report = simulator.get_splitting_order_report()
    assert report["member_class"] == {"nitrification": "stateful", "denitrification": "stateful"}
    assert report["refused_or_flagged_edges"] == ()


def test_strang_flags_but_admits_a_flux_link_between_unclassified_members():
    """S1 refuses only proven stateful endpoints; an unclassified link is flagged.

    Reuses the boundary-stage fixtures of test_time_space_enhancements.py: a writer
    of the n2o_flux boundary rate with an uncontracted probe store and a reader of
    that rate, neither declaring variable_kind (so neither is shown to hold state).
    """
    import importlib.util
    import sys
    from pathlib import Path

    alias = "_cr_a04_time_space_fixtures"
    module = sys.modules.get(alias)
    if module is None:
        path = Path(__file__).resolve().parents[1] / "unit" / "test_time_space_enhancements.py"
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    simulator = module._boundary_stage_simulator("strang", initial_rate=7.0)
    writer = module._BoundaryAndStoreWriter()
    simulator.register_functor(writer)
    simulator.register_functor(module._BoundaryStageReader([writer.name]))
    report = simulator.get_splitting_order_report()
    assert report["member_class"] == {"nitrification": "unclassified", "rate_reader": "unclassified"}
    (edge,) = report["refused_or_flagged_edges"]
    assert edge["status"] == "flagged" and edge["variables"] == ("n2o_flux",)
    assert (edge["provider"], edge["consumer"]) == ("nitrification", "rate_reader")


def test_strang_admits_a_member_without_state_feeding_a_stateful_member():
    """The C7-like composition: Priestley-Taylor ET (no state) feeds the bucket."""
    from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import PriestleyTaylorET
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    simulator = RELICSimulator({"time_step": 1800.0, "device": "cpu", "splitting_mode": "strang"})
    simulator.initialize_state({"soil_moisture": torch.tensor([0.25])})
    simulator.register_functor(PriestleyTaylorET(), execution_group="fast")
    bucket = BucketSoilMoisture()
    simulator.register_functor(bucket, execution_group="fast")
    report = simulator.get_splitting_order_report()
    assert report["member_class"] == {"priestley_taylor_et": "stateless", "bucket_soil_moisture": "stateful"}
    assert report["refused_or_flagged_edges"] == ()
    assert report["splitting_policy"] == "warn"
    assert report["fast_order_mode"] == "forward"
    assert report["time_step_seconds"] == 1800.0
    # Strang: the first member runs two half-steps, the last one full step.
    assert report["effective_substep_seconds"] == {
        "priestley_taylor_et": 900.0 / PriestleyTaylorET().metadata.substep_count,
        "bucket_soil_moisture": 1800.0 / bucket.metadata.substep_count,
    }
    assert report["exchanges"] == ()
    assert report["cross_group_composition"] == "lie_fast_medium_slow"


# --------------------------------------------------------------------------
# S3: splitting_policy require_declared.
# --------------------------------------------------------------------------


def test_require_declared_refuses_strang_with_an_undeclared_stateful_member():
    simulator = _fixture_simulator(splitting_mode="strang", splitting_policy="require_declared")
    simulator.register_functor(_Relaxation("relax_a", "relax_a", accuracy=SubFlowAccuracy.EXACT))
    before = _view_record(simulator)
    with pytest.raises(RuntimeError, match=r"S3.*relax_b"):
        simulator.register_functor(_Relaxation("relax_b", "relax_b"))
    _assert_publication_rolled_back(simulator, before)


def test_require_declared_refuses_an_undeclared_member_not_shown_to_be_stateless():
    """Unknown variable kind, uncontracted output, no self-read: not proven stateless."""
    simulator = _fixture_simulator(splitting_mode="strang", splitting_policy="require_declared")
    simulator.register_functor(_Relaxation("relax_a", "relax_a", accuracy=SubFlowAccuracy.EXACT))
    unknown = _Relaxation(
        "relax_unknown", "relax_flux", variable_kind=VariableKind.UNKNOWN, reads_own_state=False
    )
    before = _view_record(simulator)
    with pytest.raises(RuntimeError, match=r"S3.*relax_unknown"):
        simulator.register_functor(unknown)
    _assert_publication_rolled_back(simulator, before)


def test_require_declared_admits_declared_stateful_and_undeclared_stateless_members():
    simulator = _fixture_simulator(splitting_mode="strang", splitting_policy="require_declared")
    simulator.register_functor(_Relaxation("relax_a", "relax_a", accuracy=SubFlowAccuracy.EXACT))
    simulator.register_functor(_Relaxation("relax_b", "relax_b", accuracy=SubFlowAccuracy.ORDER_1))
    simulator.register_functor(
        _Relaxation("relax_flux", "relax_flux", variable_kind=VariableKind.FLUX, reads_own_state=False)
    )
    report = simulator.get_splitting_order_report()
    assert report["member_class"] == {"relax_a": "stateful", "relax_b": "stateful", "relax_flux": "stateless"}
    assert report["splitting_policy"] == "require_declared"
    simulator.step()


def test_require_declared_applies_only_to_a_split_strang_composition():
    single = _fixture_simulator(splitting_mode="strang", splitting_policy="require_declared")
    single.register_functor(_Relaxation("relax_a", "relax_a"))
    lie = _fixture_simulator(splitting_mode="lie", splitting_policy="require_declared")
    lie.register_functor(_Relaxation("relax_a", "relax_a"))
    lie.register_functor(_Relaxation("relax_b", "relax_b"))
    assert lie.get_splitting_order_report()["effective_order"] == "unverified"


def test_warn_policy_admits_an_undeclared_stateful_member_with_the_order_warning(caplog):
    simulator = _fixture_simulator(splitting_mode="strang")
    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        simulator.register_functor(_Relaxation("relax_a", "relax_a", accuracy=SubFlowAccuracy.EXACT))
        simulator.register_functor(_Relaxation("relax_b", "relax_b"))
    messages = [record.getMessage() for record in caplog.records if "Strang splitting" in record.getMessage()]
    assert len(messages) == 1 and "unverified" in messages[0]


# --------------------------------------------------------------------------
# S5: fast_order reverse.
# --------------------------------------------------------------------------


def test_reverse_fast_order_runs_independent_members_in_reverse():
    calls: List[tuple] = []
    simulator = _fixture_simulator(fast_order="reverse")
    simulator.register_functor(_Relaxation("relax_a", "relax_a", call_log=calls))
    simulator.register_functor(_Relaxation("relax_b", "relax_b", call_log=calls))
    simulator.register_functor(_Relaxation("relax_c", "relax_c", call_log=calls))
    assert simulator.get_committed_provider_view().order_fast == ("relax_c", "relax_b", "relax_a")
    report = simulator.get_splitting_order_report()
    assert report["fast_order_mode"] == "reverse"
    assert report["fast_order"] == ("relax_c", "relax_b", "relax_a")
    simulator.step()
    assert [name for name, _, _ in calls] == ["relax_c", "relax_b", "relax_a"]

    forward_calls: List[tuple] = []
    forward = _fixture_simulator()
    for name in ("relax_a", "relax_b", "relax_c"):
        forward.register_functor(_Relaxation(name, name, call_log=forward_calls))
    forward.step()
    assert [name for name, _, _ in forward_calls] == ["relax_a", "relax_b", "relax_c"]


def test_reverse_fast_order_composes_the_strang_palindrome_over_the_reversed_order():
    calls: List[tuple] = []
    simulator = _fixture_simulator(fast_order="reverse", splitting_mode="strang")
    simulator.register_functor(_Relaxation("relax_a", "relax_a", call_log=calls))
    simulator.register_functor(_Relaxation("relax_b", "relax_b", call_log=calls))
    simulator.step()
    assert calls == [("relax_b", 0.0, 30.0), ("relax_a", 0.0, 60.0), ("relax_b", 30.0, 30.0)]


@pytest.mark.parametrize("category", ["required", "optional"])
def test_reverse_fast_order_is_refused_with_a_same_step_edge(category):
    simulator = _fixture_simulator(fast_order="reverse")
    simulator.register_functor(_Relaxation("relax_a", "relax_a"))
    dependencies = {"required": [], "optional": [], "lagged": []}
    dependencies[category] = ["relax_a"]
    before = _view_record(simulator)
    with pytest.raises(RuntimeError, match=r"S5.*relax_b.*relax_a"):
        simulator.register_functor(_Relaxation("relax_b", "relax_b", dependencies=dependencies))
    _assert_publication_rolled_back(simulator, before)


def test_reverse_fast_order_admits_lagged_edges_and_forward_order_keeps_edges():
    lagged = _fixture_simulator(fast_order="reverse")
    lagged.register_functor(_Relaxation("relax_a", "relax_a"))
    lagged.register_functor(
        _Relaxation("relax_b", "relax_b", dependencies={"required": [], "optional": [], "lagged": ["relax_a"]})
    )
    assert lagged.get_committed_provider_view().order_fast == ("relax_b", "relax_a")

    forward = _fixture_simulator()
    forward.register_functor(_Relaxation("relax_a", "relax_a"))
    forward.register_functor(
        _Relaxation("relax_b", "relax_b", dependencies={"required": ["relax_a"], "optional": [], "lagged": []})
    )
    assert forward.get_committed_provider_view().order_fast == ("relax_a", "relax_b")
