import warnings
from pathlib import Path

import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.management.grazing import GrazingManagement


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _value(item: UpdateItem | torch.Tensor) -> torch.Tensor:
    return item.value if isinstance(item, UpdateItem) else item


def _state() -> dict[str, torch.Tensor]:
    return {
        "aboveground_biomass": torch.tensor([[3.0]], dtype=torch.float32),
        "vegetation_quality": torch.tensor([[0.8]], dtype=torch.float32),
        "soil_moisture": torch.tensor([[0.3]], dtype=torch.float32),
    }


def test_grazing_declares_canonical_iphs_biomass_return_port():
    manager = GrazingManagement(config={"calendar_type": "noleap"})

    assert manager.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert manager.runtime_path == "canonical_iphs_grazing_biomass_return_port"

    forms = manager.declared_forms()
    assert {form.semantic_name for form in forms} >= {
        "grazing_biomass_pool",
        "grazing_return_flux",
        "grazing_stocking_port",
    }

    hodge = manager.build_hodge_contract(strict=True)
    audit = hodge.validate(passive=True)
    assert audit["psd"] is True
    assert torch.all(torch.diag(hodge.matrix) >= 0.0)

    ports = manager.ports()
    assert len(ports) == 1
    assert ports[0].name == "grazing_biomass_return_port"
    assert ports[0].port_kind == "matter"
    assert ports[0].positive_power_convention == "out_of_system"


def test_grazing_forward_reports_canonical_runtime():
    manager = GrazingManagement(config={"calendar_type": "noleap"})

    state = _state()
    outputs = manager.forward(state, time=0.0, dt=86400.0, day_of_year=150)

    assert set(outputs) == set(manager.metadata.output_variables)
    assert outputs["diagnostics.grazing_management.canonical_iphs_runtime"].item() == 1.0
    expected_pressure = manager.grazing_events[0].stocking_rate * state["vegetation_quality"]
    assert torch.allclose(outputs["diagnostics.grazing_pressure"], expected_pressure)
    assert torch.isfinite(_value(outputs["grazing_flux"])).all()
    assert manager.check_conservation_laws(state, outputs) == {
        "grazing_matter_boundary": 0.0
    }

    replace_units = {
        "grazing_flux": "kg m-2 s-1",
        "dung_c_flux": "kg C m-2 s-1",
        "dung_n_flux": "kg N m-2 s-1",
        "urine_n_flux": "kg N m-2 s-1",
        "livestock_carbon_intake_flux": "kg C m-2 s-1",
        "livestock_carbon_export_flux": "kg C m-2 s-1",
        "livestock_nitrogen_intake_flux": "kg N m-2 s-1",
        "livestock_nitrogen_export_flux": "kg N m-2 s-1",
        "trampling_damage": "1",
        "grazing_active": "1",
        "current_stocking_rate": "animals ha-1",
        "utilization_rate": "1",
    }
    for name, unit in replace_units.items():
        item = outputs[name]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == unit
        assert item.source_functor == manager.name

    accumulate_units = {
        "aboveground_biomass": "kg m-2",
        "cumulative_intake": "kg m-2",
        "days_grazed": "day",
    }
    for name, unit in accumulate_units.items():
        item = outputs[name]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.ACCUMULATE
        assert item.per_timestep is True
        assert item.unit == unit
        assert item.source_functor == manager.name


def test_grazing_forward_is_the_single_calculation_entry():
    """forward is the only calculation entry (item 113): the uncalled
    _compute_rate_core wrapper, which forwarded an environment mapping as keywords,
    is removed."""
    assert not hasattr(GrazingManagement, "_compute_rate_core")


def test_grazing_governed_call_matches_the_forward_kernel():
    """The governed call and the forward kernel give the same grazing flux for the
    complete declared state (day_of_year included). Day 150 lies in the default spring
    event (days 120-270), so the flux is positive; day 10 lies outside every default
    event, so it is zero."""
    for day, grazing in ((10, False), (150, True)):
        state = dict(
            _state(),
            day_of_year=torch.tensor(float(day)),
        )
        governed = GrazingManagement(config={"calendar_type": "noleap"})
        kernel = GrazingManagement(config={"calendar_type": "noleap"})
        governed_flux = _value(governed(state, time=0.0, dt=86400.0)["grazing_flux"])
        kernel_flux = _value(kernel.forward(state, time=0.0, dt=86400.0)["grazing_flux"])
        assert torch.equal(governed_flux, kernel_flux)
        assert bool((governed_flux > 0.0).all()) is grazing


def test_grazing_simulator_commits_biomass_with_provenance_without_warning():
    simulator = RELICSimulator(
        {
            "max_time": 3600.0,
            "time_step": 3600.0,
            "conservation_checks": False,
            "device": "cpu",
        }
    )
    simulator.initialize_state(
        {
            "aboveground_biomass": torch.tensor([3.0]),
            "vegetation_quality": torch.tensor([0.8]),
            "soil_moisture": torch.tensor([0.3]),
            "day_of_year": torch.tensor(150.0),
            "cumulative_intake": torch.tensor([0.0]),
            "days_grazed": torch.tensor([0.0]),
        }
    )
    simulator.register_functor(GrazingManagement(), execution_group="fast")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        updates = simulator.step(dt=3600.0)

    # Item 116 (source correction): D is the spring event stocking rate 1.5 ha-1,
    # formerly the state animal_density 2.0; P_g = D Q = 1.5 * 0.8.
    assert torch.allclose(
        updates["diagnostics.grazing_pressure"],
        torch.tensor([1.2]),
    )
    provenance = simulator.get_variable_provenance("grazing_flux")
    assert {item["role"] for item in provenance} == {"writer", "final_writer"}
    assert {item["mode"] for item in provenance} == {"UpdateMode.REPLACE"}
    state = simulator.state_manager.get_state(clone=True)
    assert state["aboveground_biomass"].item() < 3.0
    assert state["cumulative_intake"].item() > 0.0
    assert state["days_grazed"].item() == torch.tensor(1.0 / 24.0).item()
    biomass_provenance = simulator.get_variable_provenance("aboveground_biomass")
    assert {item["role"] for item in biomass_provenance} == {
        "writer",
        "final_writer",
        "increment_source",
    }
    assert {item["mode"] for item in biomass_provenance} == {
        "UpdateMode.ACCUMULATE"
    }
    assert str(simulator.get_effective_device("grazing_management")) == "cpu"


def test_grazing_checkpoint_restart_is_numerically_identical():
    """Checkpoint restart must preserve every grazing prognostic state.

    References:
    - Milchunas and Lauenroth (1993), grazing effects on vegetation.
    Mathematical expressions:
    - B_ag(t+dt) = B_ag(t) - F_grazing dt.
    - I_cum(t+dt) = I_cum(t) + F_grazing dt.
    - D_grazed(t+dt) = D_grazed(t) + dt / seconds_per_day.
    Parameter meanings:
    - B_ag is aboveground biomass [kg m-2].
    - F_grazing is grazing removal [kg m-2 s-1].
    - I_cum is cumulative intake [kg m-2].
    - D_grazed is accumulated grazing duration [day].
    """
    dt = 3600.0
    initial_state = {
        "aboveground_biomass": torch.tensor([3.0], dtype=torch.float64),
        "vegetation_quality": torch.tensor([0.8], dtype=torch.float64),
        "soil_moisture": torch.tensor([0.3], dtype=torch.float64),
        "day_of_year": torch.tensor(150.0, dtype=torch.float64),
        "cumulative_intake": torch.tensor([0.0], dtype=torch.float64),
        "days_grazed": torch.tensor([0.0], dtype=torch.float64),
    }

    def build_simulator() -> RELICSimulator:
        simulator = RELICSimulator(
            {
                "max_time": 2.0 * dt,
                "time_step": dt,
                "conservation_checks": False,
                "device": "cpu",
            }
        )
        simulator.initialize_state(
            {key: value.clone() for key, value in initial_state.items()}
        )
        simulator.register_functor(
            GrazingManagement(),
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
        restarted_second = restarted.step(dt=dt)

    for key in ("aboveground_biomass", "cumulative_intake", "days_grazed"):
        torch.testing.assert_close(
            restarted.state_manager.get_state(key),
            continuous.state_manager.get_state(key),
            rtol=0.0,
            atol=0.0,
        )
    for key in (
        "grazing_flux",
        "dung_c_flux",
        "dung_n_flux",
        "urine_n_flux",
    ):
        torch.testing.assert_close(
            _value(restarted_second[key]),
            _value(continuous_second[key]),
            rtol=0.0,
            atol=0.0,
        )
    assert {
        item["role"]
        for item in restarted.get_variable_provenance("aboveground_biomass")
    } == {"writer", "final_writer", "increment_source"}
    assert str(restarted.get_effective_device("grazing_management")) == "cpu"
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]


def test_grazing_source_has_no_parallel_runtime_terms():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "management"
        / "grazing.py"
    ).read_text(encoding="utf-8")

    forbidden_terms = (
        "fallback",
        "legacy_mode",
        "use_legacy",
        "enable_legacy",
        "compute_mode",
        "zero_flux_fallback",
    )
    for term in forbidden_terms:
        assert term not in source

    assert "advance_reaction_network(" in source
    assert "_assemble_grazing_reaction_outputs" in source
    assert "stoichiometric_N=" in source


def test_grazing_phase8_certificate_is_valid():
    manager = GrazingManagement(config={"calendar_type": "noleap"})
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_management_grazing_grazing_management.yaml"
    )

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert "N_h r_h" in certificate.state_advancement_map["state_advancer"]
    assert certificate.dependency_interconnection_map["direct_functor_calls"] == (
        "none; forward reads declared biomass, forage, soil, schedule, and "
        "stocking state and calls only local grazing helpers"
    )
    assert certificate.input_variables == tuple(manager.metadata.input_variables)
    assert "forage_quality" not in certificate.input_variables
    assert certificate.unit_map["diagnostics.grazing_pressure"] == "animals ha-1"


def test_grazing_stocking_density_comes_from_the_active_event():
    """Item 116: D in I = a W^b D Q (1 - exp(-k B)) is the stocking rate of the active
    grazing event, the herd whose body weight and livestock type the intake uses (DNDC
    grazing(): animal heads belong to the grazing event). animal_density is no declared
    input and has no contract; a state entry of that name is not read."""
    manager = GrazingManagement(config={"calendar_type": "noleap"})
    assert "animal_density" not in manager.metadata.input_variables
    assert not (
        PROJECT_ROOT / "relic" / "contracts" / "variables" / "anthroposphere" / "animal_density.yaml"
    ).exists()

    state = dict(_state(), day_of_year=torch.tensor(150.0))
    reference = GrazingManagement(config={"calendar_type": "noleap"}).forward(
        state, time=0.0, dt=86400.0
    )
    with_density = GrazingManagement(config={"calendar_type": "noleap"}).forward(
        dict(state, animal_density=torch.tensor([[5.0]], dtype=torch.float32)),
        time=0.0,
        dt=86400.0,
    )
    for name in ("grazing_flux", "current_stocking_rate", "diagnostics.grazing_pressure"):
        assert torch.equal(_value(with_density[name]), _value(reference[name])), name

    doubled = GrazingManagement(config={"calendar_type": "noleap"})
    event_rate = doubled.grazing_events[0].stocking_rate
    doubled.grazing_events[0].stocking_rate = 2.0 * event_rate
    doubled_out = doubled.forward(state, time=0.0, dt=86400.0)
    assert torch.allclose(
        _value(doubled_out["current_stocking_rate"]),
        torch.full_like(_value(reference["current_stocking_rate"]), 2.0 * event_rate),
    )
    assert torch.all(_value(doubled_out["grazing_flux"]) > _value(reference["grazing_flux"]))


def test_grazing_governed_call_needs_no_animal_density():
    """The governed call validates the declared inputs; the stocking density is not
    one of them."""
    state = dict(_state(), day_of_year=torch.tensor(150.0))
    governed = GrazingManagement(config={"calendar_type": "noleap"})(
        state, time=0.0, dt=86400.0
    )
    assert torch.all(_value(governed["grazing_flux"]) > 0.0)


def test_grazing_sets_no_unused_enzyme_kinetics_label():
    """Item 121: no grazing algorithm reads the bridge enzyme_kinetics label (it is
    only validated and copied into metadata), so GrazingManagement does not set one."""
    import inspect

    assert "enzyme_kinetics" not in inspect.getsource(GrazingManagement.__init__)


def test_grazing_sets_no_unused_bridge_labels():
    """Item 122: temperature_response and ph_dependence are bridge labels no grazing
    algorithm reads (the intake has no temperature or pH law), so GrazingManagement
    does not set them."""
    import inspect

    source = inspect.getsource(GrazingManagement.__init__)
    for label in ("temperature_response", "ph_dependence", "enzyme_kinetics"):
        assert label not in source, label
