"""L1 tests for the canonical fertilization management boundary."""

from pathlib import Path

import pytest
import torch

from relic.core.abstractions import TransportMechanism
from relic.core.functor import ProcessCategory, TransportMechanismFunctor
from relic.core.state import (
    StateManager,
    StateVariableSpec,
    UpdateItem,
    UpdateMode,
)
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.scientific.anthroposphere.management.fertilization import (
    ApplicationMethod,
    FertilizationManagement,
    FertilizerEvent,
    FertilizerType,
)
from relic.utils.physical_constants import get_physical_constant


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))
KG_TO_G = float(
    get_physical_constant("relic.physical.constants.conversions.kg_to_g")
)


def _value(item: UpdateItem | torch.Tensor) -> torch.Tensor:
    return item.value if isinstance(item, UpdateItem) else item


def _p2o5_elemental_p_fraction() -> float:
    phosphorus = float(
        get_physical_constant(
            "relic.physical.constants.molecular_weights.phosphorus"
        )
    )
    oxygen = float(
        get_physical_constant("relic.physical.constants.molecular_weights.o2")
    )
    return (2.0 * phosphorus) / (2.0 * phosphorus + 2.5 * oxygen)


@pytest.fixture
def manager() -> FertilizationManagement:
    return FertilizationManagement(config={"calendar_type": "noleap"})


def test_fertilization_fulfils_managed_transport_contract(manager):
    assert isinstance(manager, TransportMechanismFunctor)
    assert isinstance(manager, TransportMechanism)
    assert manager.transport_type == "managed_application"
    assert manager.name == "fertilization_management"
    assert manager.runtime_path == "canonical_iphs_fertilization_boundary_port"


def test_metadata_declares_exact_boundary_interface(manager):
    metadata = manager.metadata
    assert metadata.category == ProcessCategory.TRANSPORT_MECHANISM
    assert metadata.input_variables == ["day_of_year"]
    assert set(manager.optional_input_variables) == {"day_of_year"}
    assert metadata.output_variables == [
        "fertilizer_n_flux",
        "fertilizer_p_flux",
        "available_phosphorus",
        "diagnostics.fertilization.nitrogen_boundary_residual",
        "diagnostics.fertilization.phosphorus_boundary_residual",
        "diagnostics.fertilization.canonical_iphs_runtime",
    ]
    assert "Canonical IPHS:" in metadata.mathematical_expression
    assert "Delta M_N" in " ".join(metadata.iphs_terms)
    assert "Delta P_available" in " ".join(metadata.iphs_terms)
    assert metadata.structural_role == ("managed_matter_boundary",)


def test_default_schedule_uses_explicit_product_mass_basis(manager):
    events = manager.fertilizer_events
    assert len(events) == 3
    assert {event.fertilizer_type for event in events} == {
        FertilizerType.UREA,
        FertilizerType.AMMONIUM_NITRATE,
        FertilizerType.SUPERPHOSPHATE,
    }
    assert all(event.product_mass > 0.0 for event in events)
    assert all(
        CalendarHandler.validate_simulator_day(
            event.application_date,
            "noleap",
        )
        == event.application_date
        for event in events
    )
    assert all(event.application_duration is None for event in events)


def test_explicit_invalid_calendar_is_not_silently_replaced():
    with pytest.raises(ValueError, match="calendar"):
        FertilizationManagement(config={"calendar_type": ""}, events=())


def test_default_schedule_rejects_fractional_parameter_day(monkeypatch):
    manager = FertilizationManagement(config={"calendar_type": "noleap"})

    def fractional_spring_day(name):
        return 120.5 if name == "spring_day" else 100.0

    monkeypatch.setattr(manager, "_parameter_scalar", fractional_spring_day)
    with pytest.raises(ValueError, match="spring_day"):
        manager._build_default_schedule()


@pytest.mark.parametrize(
    "application_date",
    (
        True,
        1.5,
        float("nan"),
        torch.tensor([1.0, 2.0]),
    ),
    ids=("boolean", "fractional", "nan", "non_scalar_tensor"),
)
def test_schedule_rejects_non_integral_application_date(application_date):
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.UREA,
        product_mass=0.01,
        application_date=application_date,
    )

    with pytest.raises(ValueError, match="application_date"):
        FertilizationManagement(
            config={"calendar_type": "noleap"},
            events=(event,),
        )


def test_domain_parameters_and_physical_constants_are_distinguished(manager):
    required = set(manager.required_parameters())
    assert required == {
        "anthroposphere.management.fertilization.spring_urea_product_mass",
        "anthroposphere.management.fertilization.spring_day",
        "anthroposphere.management.fertilization.side_dress_ammonium_nitrate_product_mass",
        "anthroposphere.management.fertilization.side_dress_day",
        "anthroposphere.management.fertilization.spring_superphosphate_product_mass",
        "anthroposphere.management.fertilization.phosphorus_day",
        "anthroposphere.management.fertilization.urea.n_content",
        "anthroposphere.management.fertilization.ammonium_nitrate.n_content",
        "anthroposphere.management.fertilization.ammonium_sulfate.n_content",
        "anthroposphere.management.fertilization.superphosphate.p2o5_mass_fraction",
    }
    assert set(manager.physical_constant_keys()) == {
        "relic.physical.constants.time.seconds_per_day",
        "relic.physical.constants.conversions.kg_to_g",
        "relic.physical.constants.molecular_weights.phosphorus",
        "relic.physical.constants.molecular_weights.o2",
    }
    assert not (required & set(manager.physical_constant_keys()))


def test_fertilizer_composition_uses_elemental_bases(manager):
    urea = manager.get_fertilizer_properties(FertilizerType.UREA)
    superphosphate = manager.get_fertilizer_properties(
        FertilizerType.SUPERPHOSPHATE
    )

    assert urea == {
        "nitrogen_mass_fraction": pytest.approx(0.46),
        "elemental_phosphorus_mass_fraction": pytest.approx(0.0),
    }
    assert superphosphate["nitrogen_mass_fraction"] == pytest.approx(0.0)
    p2o5_fraction = float(
        manager.get_parameter_value(
            "superphosphate.p2o5_mass_fraction",
            namespace="anthroposphere.management.fertilization",
            source_order=("manager",),
            required=True,
        )
    )
    assert p2o5_fraction == pytest.approx(0.20, rel=1.0e-7)
    assert superphosphate["elemental_phosphorus_mass_fraction"] == pytest.approx(
        p2o5_fraction * _p2o5_elemental_p_fraction(),
        rel=1.0e-12,
    )


def test_forward_returns_exact_update_contract_for_n_and_p():
    events = (
        FertilizerEvent(
            fertilizer_type=FertilizerType.UREA,
            product_mass=0.01,
            application_date=100,
            method=ApplicationMethod.BROADCAST,
        ),
        FertilizerEvent(
            fertilizer_type=FertilizerType.SUPERPHOSPHATE,
            product_mass=0.01,
            application_date=100,
            method=ApplicationMethod.INCORPORATION,
        ),
    )
    manager = FertilizationManagement(events=events)

    output = manager.forward(
        {},
        time=0.0,
        dt=SECONDS_PER_DAY,
        day_of_year=100,
        calendar_type="noleap",
    )

    assert set(output) == set(manager.metadata.output_variables)
    nitrogen = 0.01 * 0.46
    phosphorus_g = 0.01 * 0.20 * _p2o5_elemental_p_fraction() * KG_TO_G

    n_flux = output["fertilizer_n_flux"]
    p_flux = output["fertilizer_p_flux"]
    p_increment = output["available_phosphorus"]
    assert isinstance(n_flux, UpdateItem)
    assert isinstance(p_flux, UpdateItem)
    assert isinstance(p_increment, UpdateItem)

    assert n_flux.mode is UpdateMode.REPLACE
    assert n_flux.per_timestep is False
    assert n_flux.unit == "kg N m-2 s-1"
    assert n_flux.element is None
    assert n_flux.source_functor == "fertilization_management"
    assert _value(n_flux).item() == pytest.approx(nitrogen / SECONDS_PER_DAY)

    assert p_flux.mode is UpdateMode.REPLACE
    assert p_flux.per_timestep is False
    assert p_flux.unit == "g P m-2 s-1"
    assert p_flux.element is None
    assert p_flux.source_functor == "fertilization_management"
    assert _value(p_flux).item() == pytest.approx(phosphorus_g / SECONDS_PER_DAY)

    assert p_increment.mode is UpdateMode.ACCUMULATE
    assert p_increment.per_timestep is True
    assert p_increment.unit == "g P m-2"
    assert p_increment.element == "P"
    assert p_increment.source_functor == "fertilization_management"
    assert _value(p_increment).item() == pytest.approx(phosphorus_g)


def test_real_state_manager_applies_one_open_phosphorus_transaction():
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.SUPERPHOSPHATE,
        product_mass=0.01,
        application_date=100,
    )
    manager = FertilizationManagement(events=(event,))
    dt = SECONDS_PER_DAY
    output = manager.forward(
        {},
        time=0.0,
        dt=dt,
        day_of_year=100,
        calendar_type="noleap",
    )
    initial_p = torch.tensor(1.0)
    state_manager = StateManager(
        schema={
            "available_phosphorus": StateVariableSpec(
                name="available_phosphorus",
                units="g P m-2",
                conserved=True,
                required=False,
            ),
            "fertilizer_p_flux": StateVariableSpec(
                name="fertilizer_p_flux",
                units="g P m-2 s-1",
                conserved=True,
                required=False,
            ),
        },
        device="cpu",
    )
    state_manager.set_state(
        {
            "available_phosphorus": initial_p,
            "fertilizer_p_flux": torch.tensor(0.0),
        },
        validate=False,
    )
    p_increment = output["available_phosphorus"]
    p_flux = output["fertilizer_p_flux"]

    state_manager.apply_updates(
        {
            "available_phosphorus": p_increment,
            "fertilizer_p_flux": p_flux,
        },
        dt=dt,
        check_conservation=True,
        rollback_on_failure=True,
        update_sources={
            "available_phosphorus": "fertilization_management",
            "fertilizer_p_flux": "fertilization_management",
        },
        update_increments={
            "available_phosphorus": {
                "fertilization_management": _value(p_increment),
            }
        },
    )

    actual_delta = state_manager.get_state("available_phosphorus") - initial_p
    assert torch.allclose(actual_delta, _value(p_increment))
    assert torch.allclose(_value(p_flux) * dt, _value(p_increment))
    assert state_manager.get_state("fertilizer_p_flux").item() == pytest.approx(
        _value(p_flux).item()
    )


def test_event_overlap_is_invariant_to_timestep_partitioning():
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.SUPERPHOSPHATE,
        product_mass=0.01,
        application_date=100,
    )
    full = FertilizationManagement(events=(event,)).forward(
        {},
        time=0.0,
        dt=SECONDS_PER_DAY,
        day_of_year=100,
        calendar_type="noleap",
    )
    half_manager = FertilizationManagement(events=(event,))
    first = half_manager.forward(
        {},
        time=0.0,
        dt=SECONDS_PER_DAY / 2.0,
        day_of_year=100,
        calendar_type="noleap",
    )
    second = half_manager.forward(
        {},
        time=SECONDS_PER_DAY / 2.0,
        dt=SECONDS_PER_DAY / 2.0,
        day_of_year=100,
        calendar_type="noleap",
    )

    partitioned = _value(first["available_phosphorus"]) + _value(
        second["available_phosphorus"]
    )
    assert torch.allclose(
        partitioned,
        _value(full["available_phosphorus"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert _value(first["fertilizer_p_flux"]).item() == pytest.approx(
        _value(second["fertilizer_p_flux"]).item()
    )


def test_time_calendar_path_matches_explicit_day():
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.UREA,
        product_mass=0.01,
        application_date=100,
    )
    manager = FertilizationManagement(events=(event,))
    absolute_time = 99.0 * SECONDS_PER_DAY

    derived = manager.forward(
        {},
        time=absolute_time,
        dt=3600.0,
        calendar_type="noleap",
    )
    explicit = manager.forward(
        {},
        time=0.0,
        dt=3600.0,
        day_of_year=100,
        calendar_type="noleap",
    )

    assert torch.allclose(
        _value(derived["fertilizer_n_flux"]),
        _value(explicit["fertilizer_n_flux"]),
    )


def test_360_day_calendar_repeats_event_at_exact_year_boundary():
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.UREA,
        product_mass=0.01,
        application_date=1,
    )
    manager = FertilizationManagement(
        config={"calendar_type": "360_day"},
        events=(event,),
    )

    output = manager.forward(
        {},
        time=360.0 * SECONDS_PER_DAY,
        dt=3600.0,
        calendar_type="360_day",
    )

    assert _value(output["fertilizer_n_flux"]).item() > 0.0


def test_360_day_calendar_rejects_day_361_schedule_at_runtime():
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.UREA,
        product_mass=0.01,
        application_date=361,
    )
    manager = FertilizationManagement(events=(event,))

    with pytest.raises(ValueError, match=r"application_date must be in \[1, 360\]"):
        manager.forward(
            {},
            time=0.0,
            dt=3600.0,
            calendar_type="360_day",
        )


def test_inactive_day_is_explicit_zero_boundary():
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.UREA,
        product_mass=0.01,
        application_date=100,
    )
    manager = FertilizationManagement(events=(event,))

    output = manager.forward(
        {},
        time=0.0,
        dt=3600.0,
        day_of_year=101,
        calendar_type="noleap",
    )

    assert _value(output["fertilizer_n_flux"]).item() == 0.0
    assert _value(output["fertilizer_p_flux"]).item() == 0.0
    assert _value(output["available_phosphorus"]).item() == 0.0
    assert output[
        "diagnostics.fertilization.nitrogen_boundary_residual"
    ].item() == 0.0
    assert output[
        "diagnostics.fertilization.phosphorus_boundary_residual"
    ].item() == 0.0


def test_boundary_residuals_close_in_tensor_arithmetic():
    event = FertilizerEvent(
        fertilizer_type=FertilizerType.SUPERPHOSPHATE,
        product_mass=0.01,
        application_date=100,
    )
    manager = FertilizationManagement(events=(event,))
    output = manager.forward(
        {},
        time=0.0,
        dt=3600.0,
        day_of_year=100,
        calendar_type="noleap",
    )

    assert output[
        "diagnostics.fertilization.nitrogen_boundary_residual"
    ].item() == 0.0
    assert output[
        "diagnostics.fertilization.phosphorus_boundary_residual"
    ].item() == 0.0
    assert manager.check_conservation_laws({}, output) == {
        "nitrogen_boundary": 0.0,
        "phosphorus_boundary": 0.0,
    }


def test_conservation_checker_returns_exact_residual_magnitudes(manager):
    """The hook must return numerical maxima, not merely avoid errors."""
    output = {
        "diagnostics.fertilization.nitrogen_boundary_residual": torch.tensor(
            [-0.25, 0.1], dtype=torch.float64
        ),
        "diagnostics.fertilization.phosphorus_boundary_residual": torch.tensor(
            [0.03, -0.02], dtype=torch.float64
        ),
    }

    assert manager.check_conservation_laws({}, output) == {
        "nitrogen_boundary": pytest.approx(0.25, rel=0.0, abs=0.0),
        "phosphorus_boundary": pytest.approx(0.03, rel=0.0, abs=0.0),
    }


@pytest.mark.parametrize(
    ("diagnostic", "value", "message"),
    (
        (
            "diagnostics.fertilization.nitrogen_boundary_residual",
            torch.tensor([float("nan")]),
            "finite",
        ),
        (
            "diagnostics.fertilization.phosphorus_boundary_residual",
            torch.empty(0),
            "non-empty",
        ),
    ),
)
def test_conservation_checker_rejects_malformed_diagnostics(
    manager,
    diagnostic,
    value,
    message,
):
    """Malformed residual diagnostics cannot count as conservation evidence."""
    output = {
        "diagnostics.fertilization.nitrogen_boundary_residual": torch.tensor(0.0),
        "diagnostics.fertilization.phosphorus_boundary_residual": torch.tensor(0.0),
    }
    output[diagnostic] = value

    with pytest.raises(ValueError, match=message):
        manager.check_conservation_laws({}, output)


@pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("nan"), float("inf")])
def test_forward_rejects_invalid_timestep(bad_dt):
    manager = FertilizationManagement(events=())
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        manager.forward({}, time=0.0, dt=bad_dt, day_of_year=100)


@pytest.mark.parametrize(
    "event, message",
    [
        (
            FertilizerEvent(FertilizerType.UREA, 0.0, 100),
            "product_mass must be finite and positive",
        ),
        (
            FertilizerEvent(FertilizerType.UREA, float("nan"), 100),
            "product_mass must be finite and positive",
        ),
        (
            FertilizerEvent(FertilizerType.UREA, 0.01, 0),
            "application_date must be a finite positive integer",
        ),
    ],
)
def test_schedule_rejects_invalid_events(event, message):
    with pytest.raises(ValueError, match=message):
        FertilizationManagement(events=(event,))


@pytest.mark.parametrize(
    "fertilizer_type",
    [
        FertilizerType.POTASH,
        FertilizerType.NPK_COMPOUND,
        FertilizerType.ORGANIC,
        FertilizerType.SLOW_RELEASE,
    ],
)
def test_products_without_complete_canonical_routes_fail_fast(fertilizer_type):
    event = FertilizerEvent(fertilizer_type, 0.01, 100)
    with pytest.raises(ValueError, match="has no complete canonical elemental route"):
        FertilizationManagement(events=(event,))


def test_structure_declares_two_typed_matter_ports(manager):
    forms = manager.declared_forms()
    assert {form.semantic_name for form in forms} == {
        "fertilization_nitrogen_boundary_flow",
        "fertilization_phosphorus_boundary_flow",
    }
    ports = manager.ports()
    assert {port.name for port in ports} == {
        "fertilization_nitrogen_input_port",
        "fertilization_phosphorus_input_port",
    }
    assert all(port.port_kind == "matter" for port in ports)
    assert {port.flow_var for port in ports} == {
        "fertilizer_n_flux",
        "fertilizer_p_flux",
    }


def test_compute_uses_the_single_forward_path(manager):
    direct = manager.forward({}, time=0.0, dt=3600.0, day_of_year=1)
    computed = manager.compute(
        {"day_of_year": torch.tensor(1.0)},
        time=0.0,
        dt=3600.0,
    )
    for key in manager.metadata.output_variables:
        assert torch.equal(_value(direct[key]), _value(computed[key]))


def test_dependencies_are_empty_functor_edges(manager):
    assert manager.get_dependencies() == []


def test_source_contains_no_soil_reaction_or_hidden_pool_runtime():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "management"
        / "fertilization.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "advance_reaction_network(",
        "_compute_rate_core",
        "_calculate_dndc_denitrification",
        "_calculate_nutrient_release",
        "slow_release_n",
        "soluble_n",
        "volatilization_flux",
        "nitrification_rate",
        "denitrification_rate",
        "fertilizer_k_flux",
        "state.get(",
    )
    for term in forbidden:
        assert term not in source


def test_outputs_are_finite_and_on_functor_device(manager):
    output = manager.forward({}, time=0.0, dt=3600.0, day_of_year=1)
    for item in output.values():
        tensor = _value(item)
        assert torch.isfinite(tensor).all()
        assert tensor.device == manager.device


def test_real_simulator_applies_each_annual_event_exactly_once_over_two_years():
    """Deterministic calendar event through the governed simulator (G07).

    References:
        CF-1.8 calendar coordinates; RELIC explicit-calendar contract.
    Mathematical expression:
        sum_steps F_N dt = M_product f_N per calendar year, with F_N nonzero
        only on the steps overlapping the recurring event window.
    Parameter meanings:
        M_product applied urea mass [kg m-2], f_N its nitrogen fraction, dt the
        daily simulator step [s], day 100 the application date.
    Units: F_N kg N m-2 s-1; dt s; masses kg N m-2.
    Domain/constraints: noleap calendar, 730 daily steps, no explicit
        day_of_year state so the time-derived calendar path is exercised.
    Sign convention: nitrogen leaves the boundary as a nonnegative flux.
    """
    from relic.core.simulator import RELICSimulator

    event = FertilizerEvent(
        fertilizer_type=FertilizerType.UREA,
        product_mass=0.01,
        application_date=100,
    )
    manager = FertilizationManagement(config={"calendar_type": "noleap"}, events=(event,))
    # The nitrogen boundary export is closed by the nitrogen-cycling owner
    # in coupled runs; this single-Functor run verifies event timing and
    # exactly-once application, so the domain ledger check is not requested.
    simulator = RELICSimulator(
        {
            "time_step": SECONDS_PER_DAY,
            "calendar_type": "noleap",
            "conservation_checks": False,
            "device": "cpu",
        }
    )
    simulator.initialize_state(
        {
            "available_phosphorus": torch.tensor([0.0]),
            "fertilizer_n_flux": torch.tensor([0.0]),
            "fertilizer_p_flux": torch.tensor([0.0]),
        }
    )
    simulator.register_functor(manager, execution_group="fast")

    applied_per_step = []
    for _ in range(2 * 365):
        outputs = simulator.step(dt=SECONDS_PER_DAY)
        applied_per_step.append(float(_value(outputs["fertilizer_n_flux"]).reshape(-1)[0]) * SECONDS_PER_DAY)

    active_steps = [index for index, mass in enumerate(applied_per_step) if mass > 0.0]
    # Day-of-year 100 is the 100th daily step (index 99) of each year: once
    # per year, never duplicated, never lost, and never shifted by the year
    # boundary.
    assert active_steps == [99, 99 + 365]
    nitrogen_per_event = 0.01 * 0.46
    for index in active_steps:
        assert applied_per_step[index] == pytest.approx(nitrogen_per_event, rel=1e-6)
    assert sum(applied_per_step) == pytest.approx(2.0 * nitrogen_per_event, rel=1e-6)
    assert simulator.step_count == 730
    assert float(simulator.state_manager.get_state("available_phosphorus").reshape(-1)[0]) == 0.0
