from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.state import UpdateItem
from relic.scientific.anthroposphere.management.fire_management import FireManagement


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _state() -> dict[str, torch.Tensor]:
    return {
        "aboveground_biomass": torch.tensor([[2.0]], dtype=torch.float32),
        "belowground_biomass": torch.tensor([[1.0]], dtype=torch.float32),
        "litter_mass": torch.tensor([[0.5]], dtype=torch.float32),
        "fuel_moisture": torch.tensor([[0.10]], dtype=torch.float32),
        "days_since_fire": torch.tensor([[30.0]], dtype=torch.float32),
        "temperature": torch.tensor([[303.15]], dtype=torch.float32),
    }


def _value(item: UpdateItem | torch.Tensor) -> torch.Tensor:
    return item.value if isinstance(item, UpdateItem) else item


def test_fire_management_declares_canonical_iphs_combustion_port():
    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})

    assert manager.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert manager.runtime_path == "canonical_iphs_fire_combustion_port"

    forms = manager.declared_forms()
    assert {form.semantic_name for form in forms} >= {
        "fire_fuel_pool",
        "fire_combustion_flux",
        "fire_management_port",
    }

    hodge = manager.build_hodge_contract(strict=True)
    audit = hodge.validate(passive=True)
    assert audit["psd"] is True
    assert torch.all(torch.diag(hodge.matrix) >= 0.0)

    ports = manager.ports()
    assert len(ports) == 1
    assert ports[0].name == "fire_combustion_management_port"
    assert ports[0].port_kind == "matter"


def test_fire_inputs_are_checked_against_their_physical_domain():
    """The fire inputs are checked against their physical domains (the contract
    bounds: non-negative masses and fuel moisture, non-negative absolute
    temperature): forest biomass above 5 kg m-2, litter above 3 kg DM m-2,
    live-fuel moisture above 1 and a boreal winter temperature are read as
    given, and a negative value raises instead of being clipped."""
    import pytest

    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    state = _state()
    state.update({
        "aboveground_biomass": torch.tensor([[20.0]], dtype=torch.float32),
        "litter_mass": torch.tensor([[4.0]], dtype=torch.float32),
        "fuel_moisture": torch.tensor([[1.2]], dtype=torch.float32),
        "days_since_fire": torch.tensor([[400.0]], dtype=torch.float32),
        "temperature": torch.tensor([[240.0]], dtype=torch.float32),
    })
    outputs = manager.forward(state, time=0.0, dt=86400.0, day_of_year=100)
    assert torch.isfinite(_value(outputs["co2_emission"])).all()
    for key in ("aboveground_biomass", "litter_mass", "fuel_moisture", "days_since_fire", "temperature"):
        invalid = _state()
        invalid[key] = torch.tensor([[-1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match=key):
            manager.forward(invalid, time=0.0, dt=86400.0, day_of_year=100)


def test_fire_inputs_have_no_parameter_default():
    """Aboveground biomass, litter, fuel moisture and temperature are declared fire
    inputs without a ParameterManager fallback: a direct forward call without one
    raises like the governed call. Fire no longer reads the former substitutes
    (aboveground_biomass_default, litter_mass_default and the ignition threshold
    fuel_moisture_threshold used as a moisture state); the reference temperature
    anthroposphere.management.temperature of the moisture factor stays a parameter
    but is no default of the temperature input."""
    import pytest

    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    for key in ("aboveground_biomass", "litter_mass", "fuel_moisture", "days_since_fire", "temperature"):
        assert key in manager.get_required_inputs()
        state = _state()
        state.pop(key)
        with pytest.raises(KeyError, match=key):
            manager.forward(state, time=0.0, dt=86400.0, day_of_year=100)
    former_defaults = {
        "anthroposphere.management.aboveground_biomass_default",
        "anthroposphere.management.litter_mass_default",
        "anthroposphere.management.fire.fuel_moisture_threshold",
    }
    assert not former_defaults & set(manager.metadata.parameters)


def _seeded(seed: int = 7) -> FireManagement:
    return FireManagement(config={"calendar_type": "noleap", "seed": seed})


def _wildfire_day_state(day: int) -> dict[str, torch.Tensor]:
    """Fuel state on a day of the default wildfire window (event day 240)."""
    return {
        "aboveground_biomass": torch.tensor([[2.0]], dtype=torch.float32),
        "litter_mass": torch.tensor([[0.5]], dtype=torch.float32),
        "fuel_moisture": torch.tensor([[0.10]], dtype=torch.float32),
        "days_since_fire": torch.tensor([[30.0]], dtype=torch.float32),
        "temperature": torch.tensor([[303.15]], dtype=torch.float32),
        "day_of_year": torch.tensor(float(day)),
    }


def _run_days(manager: FireManagement, days) -> list[dict[str, torch.Tensor]]:
    outputs = []
    for day in days:
        result = manager.forward(
            _wildfire_day_state(day), time=(day - 1) * 86400.0, dt=86400.0
        )
        outputs.append({name: _value(item).clone() for name, item in result.items()})
    return outputs


def test_fire_ignition_draw_comes_from_the_functor_substream():
    """Item 115: the wildfire ignition draw u ~ U(0, 1) comes from the Functor-owned
    substream (FunctorSubstreamMixin); the private generator and its separate
    rng_state extra state are removed, so the substream owner is the only RNG state."""
    from relic.core.functor import BiologicalProcessFunctor, FunctorSubstreamMixin

    manager = _seeded()
    assert isinstance(manager, FunctorSubstreamMixin)
    mro = FireManagement.__mro__
    assert mro.index(FunctorSubstreamMixin) < mro.index(BiologicalProcessFunctor)
    assert not hasattr(manager, "_rng")
    extra = manager.get_extra_state()
    assert "rng_state" not in extra
    assert set(extra) == {
        "format",
        "version",
        "seed",
        "draw_count",
        "generator_state",
        "wiener_pending",
        "fire_events",
        "executed_schedule_occurrences",
    }
    assert extra["seed"] == 7


def test_fire_seed_follows_the_substream_convention():
    """config ``seed`` is required: a non-negative integer below 2**63
    (validate_substream_seed). The former key random_seed is rejected with a hint."""
    for config in ({}, {"seed": None}):
        with pytest.raises(ValueError, match="seed"):
            FireManagement(config={"calendar_type": "noleap", **config})
    for invalid in ("abc", 1.5, True, -1, 2**63):
        with pytest.raises(ValueError, match="seed"):
            FireManagement(config={"calendar_type": "noleap", "seed": invalid})
    with pytest.raises(ValueError, match="use 'seed'"):
        FireManagement(config={"calendar_type": "noleap", "random_seed": 7})
    assert _seeded(2**63 - 1).get_extra_state()["seed"] == 2**63 - 1


def test_fire_ignition_draws_are_bitwise_reproducible():
    """Equal seeds give bitwise equal ignition decisions, outputs and stream
    positions over the default wildfire window; a different seed gives a different
    stream position."""
    days = range(228, 253)
    first, second, other = _seeded(7), _seeded(7), _seeded(8)
    first_out, second_out = _run_days(first, days), _run_days(second, days)
    _run_days(other, days)
    for left, right in zip(first_out, second_out):
        assert left.keys() == right.keys()
        for name in left:
            assert torch.equal(left[name], right[name]), name
    assert first.draw_count == second.draw_count > 0
    assert torch.equal(first.generator.get_state(), second.generator.get_state())
    assert not torch.equal(first.generator.get_state(), other.generator.get_state())


def test_fire_substream_checkpoint_continues_the_sequence():
    """A checkpoint (state_dict with the substream extra state) restored into a fresh
    FireManagement continues the ignition sequence of the uninterrupted run."""
    import copy

    uninterrupted = _seeded(11)
    reference = _run_days(uninterrupted, range(228, 253))

    interrupted = _seeded(11)
    _run_days(interrupted, range(228, 240))
    checkpoint = copy.deepcopy(interrupted.state_dict())
    restored = _seeded(11)
    restored.load_state_dict(checkpoint)
    continued = _run_days(restored, range(240, 253))

    for expected, actual in zip(reference[12:], continued):
        for name in expected:
            assert torch.equal(expected[name], actual[name]), name
    assert restored.draw_count == uninterrupted.draw_count
    assert torch.equal(restored.generator.get_state(), uninterrupted.generator.get_state())


def test_fire_parameter_sensitivity_uses_common_random_numbers():
    """The base central difference restores the substream position around its
    evaluations (common random numbers): the stream position, draw count and
    executed occurrences are unchanged, and the next forward equals that of a twin
    that made no sensitivity call. FireManagement holds no tensor member (item 118),
    so a probe buffer that forward does not read is perturbed: both evaluations draw
    the ignition from the same stream position and the derivative is exactly 0."""
    manager, twin = _seeded(3), _seeded(3)
    manager.register_buffer("sensitivity_probe", torch.tensor(1.0))
    state = _wildfire_day_state(240)
    before = manager.get_extra_state()
    sensitivity = manager.parameter_sensitivity(
        "sensitivity_probe",
        state,
        time=239 * 86400.0,
        dt=86400.0,
        target_output="days_since_fire",
        relative_eps=1.0e-2,
        absolute_eps=1.0e-2,
    )
    after = manager.get_extra_state()
    assert torch.equal(after["generator_state"], before["generator_state"])
    assert after["draw_count"] == before["draw_count"]
    assert after["executed_schedule_occurrences"] == before["executed_schedule_occurrences"]
    assert float(sensitivity) == 0.0
    next_out = manager.forward(state, time=239 * 86400.0, dt=86400.0)
    twin_out = twin.forward(state, time=239 * 86400.0, dt=86400.0)
    for name in ("combustion_flux", "burned_fraction", "diagnostics.fire.events_today"):
        assert torch.equal(_value(next_out[name]), _value(twin_out[name])), name


def test_fire_management_forward_reports_canonical_runtime():
    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})

    outputs = manager.forward(_state(), time=0.0, dt=86400.0, day_of_year=100)

    assert _value(
        outputs["diagnostics.fire_management.canonical_iphs_runtime"]
    ).item() == 1.0
    assert torch.isfinite(_value(outputs["co2_emission"])).all()
    assert torch.isfinite(_value(outputs["burned_fraction"])).all()


def _burn_state() -> dict[str, torch.Tensor]:
    return {
        "fuel_load": torch.tensor([[5.0]], dtype=torch.float32),
        "fuel_moisture": torch.tensor([[0.1]], dtype=torch.float32),
        "days_since_fire": torch.tensor([[30.0]], dtype=torch.float32),
        "aboveground_biomass": torch.tensor([[0.3]], dtype=torch.float32),
        "litter_mass": torch.tensor([[0.3]], dtype=torch.float32),
        "temperature": torch.tensor([[288.15]], dtype=torch.float32),
    }


def test_fire_management_forward_is_the_single_calculation_entry():
    """forward is the only calculation entry (item 113): the uncalled
    _compute_rate_core wrapper, which took calendar_type from an environment mapping,
    is removed; the calendar reaches forward as the simulator keyword."""
    assert not hasattr(FireManagement, "_compute_rate_core")


def test_fire_management_governed_call_matches_the_forward_kernel():
    """The governed call and the forward kernel give the same combustion, CO2 and
    carbon-loss fluxes. Day 1 has no scheduled occurrence (all zero); day 100 executes
    the default prescribed burn, so the comparison is not a zero-equals-zero check.
    (The removed rate-core test compared day-1 zeros: its force_fire entry sat in the
    environment mapping, which the wrapper never read.)"""
    for day, burning in ((1, False), (100, True)):
        governed = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        kernel = FireManagement(config={"calendar_type": "noleap", "seed": 0})
        governed_out = governed(_burn_state(), time=0.0, dt=86400.0, day_of_year=day)
        kernel_out = kernel.forward(_burn_state(), time=0.0, dt=86400.0, day_of_year=day)
        for name in ("combustion_flux", "co2_emission", "carbon_loss"):
            assert torch.equal(_value(governed_out[name]), _value(kernel_out[name])), name
        assert bool((_value(governed_out["combustion_flux"]) > 0.0).all()) is burning


def test_fire_management_calendar_is_the_forward_keyword():
    """Without a configured calendar, forward takes calendar_type as its keyword (the
    simulator passes it because forward names it); without either it raises."""
    configured = FireManagement(config={"calendar_type": "noleap", "seed": 0}).forward(
        _burn_state(), time=0.0, dt=86400.0, day_of_year=100
    )
    keyword = FireManagement(config={"seed": 0}).forward(
        _burn_state(), time=0.0, dt=86400.0, calendar_type="noleap", day_of_year=100
    )
    assert torch.equal(
        _value(keyword["combustion_flux"]), _value(configured["combustion_flux"])
    )
    with pytest.raises(ValueError, match="calendar_type"):
        FireManagement(config={"seed": 0}).forward(
            _burn_state(), time=0.0, dt=86400.0, day_of_year=100
        )


def test_fire_management_source_has_no_parallel_runtime_terms():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "management"
        / "fire_management.py"
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
    assert "_assemble_fire_reaction_outputs" in source
    assert "stoichiometric_N=" in source


def test_fire_management_phase8_certificate_is_valid():
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_management_fire_management_fire_management.yaml"
    )

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert "N_h r_h" in certificate.state_advancement_map["state_advancer"]
    assert certificate.dependency_interconnection_map["direct_functor_calls"] == (
        "none; forward reads declared fuel, biomass, litter, moisture, "
        "temperature, schedule, and suppression state and calls only local "
        "fire helpers"
    )



@pytest.mark.parametrize(
    "key, value",
    [
        ("burn_fraction", 1.5),
        ("burn_fraction", -0.1),
        ("burn_fraction", float("nan")),
        ("fuel_load", -1.0),
        ("fuel_load", float("inf")),
    ],
)
def test_fire_overrides_outside_their_domain_raise(key, value):
    """burn_fraction [1] must lie in [0, 1] and fuel_load [kg DM m-2] must be finite
    and non-negative (their contract bounds); a value outside is an input error and
    raises instead of being clipped."""
    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    state = _state()
    state[key] = torch.tensor([[value]], dtype=torch.float32)
    with pytest.raises(ValueError, match=key):
        manager.forward(state, time=0.0, dt=86400.0, day_of_year=100)


def test_fire_burn_fraction_in_its_domain_is_used_unchanged():
    """An admissible burn_fraction replaces the event fraction as supplied."""
    import inspect

    source = inspect.getsource(FireManagement.forward)
    assert "effective_burn_fraction = torch.clamp(" not in source
    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    state = _state()
    state["burn_fraction"] = torch.tensor([[0.5]], dtype=torch.float32)
    outputs = manager.forward(state, time=0.0, dt=86400.0, day_of_year=100)
    assert torch.isfinite(_value(outputs["co2_emission"])).all()


def _fire_state(day: int, days_since_fire: float = 30.0) -> dict[str, torch.Tensor]:
    state = dict(_burn_state())
    state["day_of_year"] = torch.tensor(float(day))
    state["days_since_fire"] = torch.tensor([[days_since_fire]], dtype=torch.float32)
    return state


def _tensor_state(manager: FireManagement) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in manager.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def test_fire_forward_mutates_no_module_tensor():
    """Item 118 (invariant I1): forward changes no module tensor. The former
    cumulative buffers were advanced in place on every call (a retried or
    sensitivity-evaluated step counted twice); they, the unread suppression buffer,
    last_fire_intensity and the uncalled get_fire_statistics are removed."""
    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    for name in (
        "cumulative_burned_area",
        "cumulative_emissions",
        "suppression_resources",
        "days_since_fire",
        "last_fire_intensity",
        "get_fire_statistics",
    ):
        assert not hasattr(manager, name), name
    before = _tensor_state(manager)
    first = manager.forward(_fire_state(1), time=0.0, dt=86400.0)
    second = manager.forward(_fire_state(1), time=0.0, dt=86400.0)
    for name in first:
        assert torch.equal(_value(first[name]), _value(second[name])), name
    manager.forward(_fire_state(100), time=99 * 86400.0, dt=86400.0)
    after = _tensor_state(manager)
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name


def test_days_since_fire_is_the_owned_state_input():
    """days_since_fire [day] is read from the state and returned as a REPLACE update:
    D' = D + dt / t_day without a fire, D' = 0 on a burn day. A missing value raises
    (no parameter initial value stands in for it)."""
    from relic.core.functor.functor_exceptions import ExecutionError
    from relic.core.state import UpdateMode

    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    assert "days_since_fire" in manager.metadata.input_variables
    quiet = manager.forward(_fire_state(1, 30.0), time=0.0, dt=43200.0)
    item = quiet["days_since_fire"]
    assert item.mode is UpdateMode.REPLACE and item.per_timestep is False
    assert torch.allclose(item.value, torch.tensor([[30.5]], dtype=item.value.dtype))
    burn = manager.forward(_fire_state(100, 30.0), time=99 * 86400.0, dt=86400.0)
    assert torch.equal(
        burn["days_since_fire"].value, torch.zeros((1, 1), dtype=item.value.dtype)
    )
    missing = _fire_state(1)
    missing.pop("days_since_fire")
    with pytest.raises(ExecutionError, match="days_since_fire"):
        FireManagement(config={"calendar_type": "noleap", "seed": 0})(
            missing, time=0.0, dt=86400.0
        )


def test_fire_retry_and_checkpoint_do_not_double_count():
    """An adaptive retry re-evaluates the step from the rolled-back state: the output
    depends only on (state, dt), never on a previous evaluation. A checkpoint carries
    no fire tensor besides the substream extra state."""
    import copy

    manager = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    manager.forward(_fire_state(1, 30.0), time=0.0, dt=86400.0)  # rejected attempt
    retry = manager.forward(_fire_state(1, 30.0), time=0.0, dt=43200.0)
    fresh = FireManagement(config={"calendar_type": "noleap", "seed": 0}).forward(
        _fire_state(1, 30.0), time=0.0, dt=43200.0
    )
    assert torch.equal(retry["days_since_fire"].value, fresh["days_since_fire"].value)

    checkpoint = copy.deepcopy(manager.state_dict())
    assert set(checkpoint) == {"_extra_state"}
    restored = FireManagement(config={"calendar_type": "noleap", "seed": 0})
    restored.load_state_dict(checkpoint)
    left = manager.forward(_fire_state(2, 31.0), time=86400.0, dt=86400.0)
    right = restored.forward(_fire_state(2, 31.0), time=86400.0, dt=86400.0)
    for name in left:
        assert torch.equal(_value(left[name]), _value(right[name])), name
