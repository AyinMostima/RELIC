"""
Unit tests for hydrosphere.snow.albedo SnowAlbedo functor.

Focus on parameter loading via ParameterManager and default-driven
forward behaviour when forcing data are absent.
"""

import inspect
from pathlib import Path

import pytest
import math
import torch
import warnings

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.contracts.contract_loader import ContractRegistry
from relic.core.functor import IPHSDECFunctor
from relic.core.state import UpdateItem
from relic.parameters import get_global_manager
from relic.scientific.hydrosphere.snow import albedo as snow_albedo_module
from relic.scientific.hydrosphere.snow.albedo import SnowAlbedo


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def call_forward(functor, state, time=0.0, dt=86400.0):
    """Call forward and expose unwrapped primary outputs for legacy tests.

    The state is the only input channel of the Functor.
    """
    result = functor.forward(dict(state), time=time, dt=dt)
    payload = dict(result)
    if "surface_albedo" in payload:
        payload["surface_albedo"] = _unwrap_update(payload["surface_albedo"])
        payload["rate"] = payload["surface_albedo"]
    return payload


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


def test_surface_albedo_contract_matches_runtime_writer() -> None:
    """The SCM must describe the replace-only SnowAlbedo diagnostic."""
    contract_path = (
        Path(__file__).resolve().parents[4]
        / "relic"
        / "contracts"
        / "variables"
        / "hydrosphere"
        / "surface_albedo.yaml"
    )
    source_text = contract_path.read_text(encoding="utf-8")
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(contract_path)
    contract = registry.get_contract("surface_albedo")

    assert "REVIEW_NEEDED" not in source_text
    assert contract is not None
    assert contract.kind == "diagnostic"
    assert contract.cf_standard_name == "surface_albedo"
    assert contract.owner == "snow_albedo"
    assert contract.incrementable is False
    assert contract.tendency_writers == []
    assert registry.can_write("surface_albedo", "snow_albedo")
    assert not registry.can_increment("surface_albedo", "snow_albedo")
    assert contract.raw["numerics"]["lower_bound"] == 0.0
    assert contract.raw["numerics"]["upper_bound"] == 1.0


@pytest.fixture(scope="module")
def parameter_manager():
    """Return global parameter manager for assertions."""
    manager = get_global_manager()
    assert manager is not None
    return manager


def test_snow_albedo_parameters_loaded(parameter_manager):
    """SnowAlbedo should pull all required parameters from YAML namespaces."""
    functor = SnowAlbedo()

    # Verify categorical selection
    assert functor.albedo_scheme in {"bats", "age_based"}

    # Compare numeric parameters with ParameterManager values
    expected_fresh = parameter_manager.get(
        "hydrosphere.snow.fresh_snow_albedo",
        units="dimensionless",
    )
    expected_reset = parameter_manager.get(
        "hydrosphere.snow.snow_reset_threshold",
        units="kg m-2 s-1",
    )
    expected_decay_cold = parameter_manager.get(
        "hydrosphere.snow.albedo_decay_rate_cold",
        units="s-1",
    )
    expected_decay_warm = parameter_manager.get(
        "hydrosphere.snow.albedo_decay_rate_warm",
        units="s-1",
    )
    expected_temp_threshold = parameter_manager.get(
        "hydrosphere.snow.temperature_threshold_warm",
        units="K",
    )
    expected_grain_factor = parameter_manager.get(
        "hydrosphere.snow.bats_grain_growth_temperature_factor",
        units="K",
    )

    assert pytest.approx(functor.fresh_snow_albedo) == float(expected_fresh)
    assert pytest.approx(functor.snow_reset_threshold) == float(expected_reset)
    assert pytest.approx(functor.albedo_decay_rate_cold) == float(expected_decay_cold)
    assert pytest.approx(functor.albedo_decay_rate_warm) == float(expected_decay_warm)
    assert pytest.approx(functor.temperature_threshold_warm) == float(expected_temp_threshold)
    assert pytest.approx(functor.bats_grain_growth_temperature_factor) == float(expected_grain_factor)
    # D55: the BATS r1 exponent is b (1/T_ref - 1/T) with b = 5000 K; no dimensionless
    # multiplier exists in BATS 1e or in the RHESSys reference implementation.
    assert not hasattr(functor, "grain_growth_coefficient")


def test_physical_constants_are_loaded_through_ssot_api(monkeypatch):
    """seconds_per_day is the only physical constant; no temperature constant stands in for a
    missing input any more."""
    calls = []
    constants = {"seconds_per_day": 90000.0}

    def fake_get_physical_constant(name):
        calls.append(name)
        return constants[name]

    monkeypatch.setattr(
        snow_albedo_module,
        "get_physical_constant",
        fake_get_physical_constant,
        raising=False,
    )

    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value(
            "hydrosphere.snow.albedo_scheme",
            "age_based",
            units="dimensionless",
            validate=True,
            notify=False,
        )
        functor = SnowAlbedo()
    finally:
        manager.restore(snapshot)
    assert functor.age_state_key == "snow_surface_age"
    result = functor.forward(
        state={
            "snowfall": torch.tensor(0.0),
            "air_temperature": torch.tensor(268.15),
            "snow_surface_temperature": torch.tensor(268.15),
            "solar_zenith_angle": torch.tensor(0.0),
            "snow_depth": torch.tensor(0.2),
            "snow_surface_age": torch.tensor(0.0),
        },
        time=0.0,
        dt=constants["seconds_per_day"],
    )

    assert calls == ["seconds_per_day"]
    assert not hasattr(functor, "standard_temperature")
    snow_surface_age = _unwrap_update(result["snow_surface_age"])
    assert torch.isclose(snow_surface_age, snow_surface_age.new_tensor(1.0))
    assert not any(
        key.startswith("relic.physical.constants.")
        for key in functor.required_parameters()
    )


_SNOW_STATE_DEFAULT_KEYS = (
    "hydrosphere.state_defaults.snowfall_flux",
    "hydrosphere.state_defaults.solar_zenith_angle",
    "hydrosphere.state_defaults.snow_depth",
    "hydrosphere.state_defaults.snow_surface_age",
    "hydrosphere.state_defaults.snow_nondimensional_age",
)


def test_snow_albedo_requires_every_forcing_input():
    """A missing input raises; no state default or constant temperature is substituted."""
    functor = SnowAlbedo()
    required = {
        "snowfall",
        "snow_surface_temperature",
        "air_temperature",
        "solar_zenith_angle",
        "snow_depth",
        functor.age_state_key,
    }
    assert required <= set(functor.get_required_inputs())

    with pytest.raises(ValueError) as error:
        functor.forward(state={}, time=0.0, dt=86400.0)
    for name in required:
        assert name in str(error.value)

    partial = _snow_age_probe_state(snow_nondimensional_age=torch.tensor(0.5))
    del partial["snow_surface_temperature"]
    with pytest.raises(ValueError, match="snow_surface_temperature"):
        functor.forward(state=partial, time=0.0, dt=86400.0)


def test_snow_albedo_reads_no_state_default_keys():
    functor = SnowAlbedo()
    manager = get_global_manager()

    assert not hasattr(functor, "_state_defaults")
    assert not hasattr(functor, "_apply_state_defaults")
    for key in _SNOW_STATE_DEFAULT_KEYS:
        assert key not in functor.required_parameters()
        assert manager.get_parameter(key) is None, key


def test_snow_albedo_compute_rate_matches_forward():
    """compute_rate output must mirror forward surface albedo."""
    functor = SnowAlbedo()
    rate_payload = call_forward(
        functor, state=_snow_age_probe_state(snow_nondimensional_age=torch.tensor(0.5))
    )

    assert torch.isclose(rate_payload["rate"], rate_payload["surface_albedo"], atol=1e-6)


def test_snow_albedo_rejects_invalid_state_instead_of_restarting_from_defaults():
    """Invalid supplied observations must not be discarded silently."""
    functor = SnowAlbedo()
    with pytest.raises((TypeError, ValueError)):
        functor.forward(
            state={"snow_depth": {"invalid": "value"}},
            time=0.0,
            dt=86400.0,
        )


class TestSnowAlbedoPhase8:
    """Test Phase 8 IPHS-DEC contract for snow albedo."""

    def test_phase8_iphs_protocol(self):
        """SnowAlbedo should expose observation/radiation IPHS hooks."""
        functor = SnowAlbedo()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.snow.fresh_snow_albedo" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "surface_albedo",
            "snow_surface_age",
            "snow_nondimensional_age",
            "absorbed_shortwave_fraction",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {"snow_albedo_radiation_port"}

    def test_phase8_certificate_is_valid(self):
        """SnowAlbedo Phase 8 certificate should be complete."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_snow_albedo_snow_albedo.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == "relic.scientific.hydrosphere.snow.albedo.SnowAlbedo"
        assert certificate.canonical_role == "observation"

    def test_radiation_residual_diagnostics_are_routed(self):
        """Projection and shortwave partition diagnostics should be exposed."""
        functor = SnowAlbedo()
        result = functor.forward(
            state={
                "snowfall": torch.tensor(0.0),
                "air_temperature": torch.tensor(268.15),
                "snow_surface_temperature": torch.tensor(268.15),
                "solar_zenith_angle": torch.tensor(0.6),
                "snow_depth": torch.tensor(0.2),
                functor.age_state_key: torch.tensor(2.0),
            },
            time=0.0,
            dt=86400.0,
        )

        diagnostics = result["diagnostics.snow_albedo"]
        assert "albedo_clipping_residual" in diagnostics
        assert "snow_absorbed_shortwave_fraction" in diagnostics
        assert torch.isfinite(diagnostics["albedo_clipping_residual"]).all()
        assert torch.isfinite(diagnostics["snow_absorbed_shortwave_fraction"]).all()
        assert torch.all(diagnostics["albedo_clipping_residual"] >= 0.0)

    def test_radiation_port_power_diagnostic_is_routed(self):
        """Incoming shortwave effort should be routed to an auditable port power."""
        functor = SnowAlbedo()
        incoming_shortwave = torch.tensor(300.0)
        result = functor.forward(
            state={
                "snowfall": torch.tensor(0.0),
                "air_temperature": torch.tensor(268.15),
                "snow_surface_temperature": torch.tensor(268.15),
                "solar_zenith_angle": torch.tensor(0.6),
                "snow_depth": torch.tensor(0.2),
                functor.age_state_key: torch.tensor(2.0),
                "shortwave_down": incoming_shortwave,
            },
            time=0.0,
            dt=86400.0,
        )

        diagnostics = result["diagnostics.snow_albedo"]
        absorbed_fraction = diagnostics["snow_absorbed_shortwave_fraction"]
        port_power = diagnostics["snow_albedo_radiation_port_power"]
        assert "snow_albedo_radiation_port_power" in diagnostics
        assert torch.isclose(
            port_power,
            incoming_shortwave.to(port_power.dtype) * absorbed_fraction.to(port_power.dtype),
            atol=1e-6,
        )
        assert functor.age_tendency_key in diagnostics


def _snow_age_probe_state(**overrides):
    state = {
        "snowfall": torch.tensor(0.0),
        "snow_surface_temperature": torch.tensor(263.15),
        "air_temperature": torch.tensor(263.15),
        "solar_zenith_angle": torch.tensor(0.3),
        "snow_depth": torch.tensor(0.5),
    }
    state.update(overrides)
    return state


def _bats_oracle(functor, temperature, zenith, tau_start, dt):
    # References: Dickinson et al. (1993), BATS Version 1e; RHESSys snowpack_daily_F.c
    #   reference implementation (r1 = exp(5000*(1/273.16 - 1/(Tss+KELVIN))), r2 = min(r1^10, 1),
    #   r3 = bats_r3); Yang et al. (1997) J. Climate 10:353-373.
    # Mathematical expression: r1 = exp(b*(1/T_ref - 1/(T+eps))), r2 = min(r1^10, 1),
    #   tau = tau_0 + dt*k_decay(T)*(r1+r2+r3), F = tau/(1+tau), alpha = w*vis + (1-w)*nir.
    # Parameter meanings: b BATS grain growth temperature factor (5000 K), T_ref reference
    #   temperature, eps denominator guard, k_decay cold/warm rate, r3 dust and soot, w spectral weight.
    # Units: tau, F, r1, r2, r3, alpha [1]; b, T_ref, eps, T [K]; k_decay [s-1]; dt [s]; zenith [rad].
    # Domain/constraints: T inside the configured BATS clamp range; no fresh snow.
    # Sign convention: aging increases tau and lowers albedo.
    t_clamped = min(max(temperature, functor.temperature_clamp_min), functor.temperature_clamp_max)
    r1 = math.exp(
        functor.bats_grain_growth_temperature_factor
        * (1.0 / functor.bats_reference_temperature - 1.0 / (t_clamped + functor.bats_temperature_epsilon))
    )
    r2 = min(r1 ** 10, 1.0)
    r3 = functor.dust_soot_factor
    decay = functor.albedo_decay_rate_warm if t_clamped >= functor.temperature_threshold_warm else functor.albedo_decay_rate_cold
    tau = tau_start + dt * decay * (r1 + r2 + r3)
    fage = tau / (1.0 + tau)
    cos_z = math.cos(zenith)
    b = functor.bats_b_parameter
    fz = (1.0 / b) * ((1.0 + b) / (1.0 + 2.0 * b * cos_z) - 1.0) if cos_z < functor.bats_cos_zenith_threshold else 0.0
    vis = functor.bats_albedo_vis_base * (1.0 - functor.bats_albedo_vis_age_factor * fage)
    vis = vis + functor.bats_albedo_zenith_factor * fz * (1.0 - vis)
    nir = functor.bats_albedo_nir_base * (1.0 - functor.bats_albedo_nir_age_factor * fage)
    nir = nir + functor.bats_albedo_zenith_factor * fz * (1.0 - nir)
    w = functor.bats_albedo_average_weight
    return tau, w * vis + (1.0 - w) * nir


def test_bats_scheme_owns_the_nondimensional_age_state():
    functor = SnowAlbedo(name="snow_albedo_bats_age_state")
    assert functor.albedo_scheme == "bats"
    assert functor.age_state_key == "snow_nondimensional_age"
    dt = 86400.0
    result = functor(
        _snow_age_probe_state(snow_nondimensional_age=torch.tensor(0.5), snow_surface_age=torch.tensor(30.0)),
        time=0.0,
        dt=dt,
    )
    assert "snow_nondimensional_age" in result
    assert "snow_surface_age" not in result
    expected_tau, expected_albedo = _bats_oracle(functor, 263.15, 0.3, 0.5, dt)
    tau = float(_unwrap_update(result["snow_nondimensional_age"]))
    assert abs(tau - expected_tau) <= 1.0e-5 * expected_tau
    albedo = float(_unwrap_update(result["surface_albedo"]))
    assert functor.albedo_clamp_min < expected_albedo < functor.albedo_clamp_max
    assert abs(albedo - expected_albedo) <= 1.0e-5
    diagnostics = result["diagnostics.snow_albedo"]
    assert "snow_nondimensional_age_tendency" in diagnostics
    assert "snow_age_tendency" not in diagnostics
    tendency = float(diagnostics["snow_nondimensional_age_tendency"])
    assert abs(tendency - (expected_tau - 0.5) / dt) <= 1.0e-5 * abs((expected_tau - 0.5) / dt)


def test_fresh_snowfall_resets_only_the_active_age_state():
    functor = SnowAlbedo(name="snow_albedo_bats_reset")
    dt = 3600.0
    result = functor(
        _snow_age_probe_state(snowfall=torch.tensor(0.01), snow_nondimensional_age=torch.tensor(4.0)),
        time=0.0,
        dt=dt,
    )
    expected_tau, _ = _bats_oracle(functor, 263.15, 0.3, 0.0, dt)
    tau = float(_unwrap_update(result["snow_nondimensional_age"]))
    assert bool(result["diagnostics.snow_albedo"]["fresh_snow_reset"])
    assert abs(tau - expected_tau) <= 1.0e-5 * expected_tau
    assert tau < 4.0


def test_age_based_scheme_owns_the_day_age_with_kelvin_inverse_sensitivity():
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value(
            "hydrosphere.snow.albedo_scheme",
            "age_based",
            units="dimensionless",
            validate=True,
            notify=False,
        )
        functor = SnowAlbedo(name="snow_albedo_day_age_state")
        assert functor.albedo_scheme == "age_based"
        assert functor.age_state_key == "snow_surface_age"
        dt = 86400.0
        result = functor(
            _snow_age_probe_state(
                snow_surface_age=torch.tensor(2.0),
                snow_nondimensional_age=torch.tensor(9.0),
                air_temperature=torch.tensor(278.15),
                snow_surface_temperature=torch.tensor(273.15),
            ),
            time=0.0,
            dt=dt,
        )
        assert "snow_surface_age" in result
        assert "snow_nondimensional_age" not in result
        # References: Warren and Wiscombe (1980).
        # Mathematical expression: t_age = 2 + 1 day; f_T = exp(kappa*(278.15 - T_warm));
        #   alpha = alpha_old + (alpha_fresh - alpha_old)*exp(-t_age*f_T/tau).
        # Parameter meanings: kappa [K-1] is the dedicated age-based sensitivity.
        # Units: t_age, tau [day]; kappa [K-1]; T [K]; alpha [1].
        # Domain/constraints: warm branch active because T > T_warm.
        # Sign convention: warmth accelerates the albedo decay.
        age = float(_unwrap_update(result["snow_surface_age"]))
        assert abs(age - 3.0) <= 1.0e-6
        kappa = functor.age_based_temperature_sensitivity
        factor = math.exp((278.15 - functor.temperature_threshold_warm) * kappa)
        expected_albedo = functor.old_snow_albedo + (
            functor.fresh_snow_albedo - functor.old_snow_albedo
        ) * math.exp(-3.0 * factor / functor.aging_time_scale)
        albedo = float(_unwrap_update(result["surface_albedo"]))
        assert abs(albedo - expected_albedo) <= 1.0e-5
        diagnostics = result["diagnostics.snow_albedo"]
        assert "snow_age_tendency" in diagnostics
        assert "snow_nondimensional_age_tendency" not in diagnostics
        assert abs(float(diagnostics["snow_age_tendency"]) - 1.0 / dt) <= 1.0e-12
    finally:
        manager.restore(snapshot)


def test_snow_age_state_contracts_are_distinct_and_owned():
    base = Path(__file__).resolve().parents[4] / "relic" / "contracts" / "variables" / "hydrosphere"
    registry = ContractRegistry(strict=True)
    for name in ("snow_surface_age", "snow_nondimensional_age"):
        path = base / f"{name}.yaml"
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)
    day = registry.get_contract("snow_surface_age")
    nondimensional = registry.get_contract("snow_nondimensional_age")
    assert day is not None and nondimensional is not None
    assert day.kind == "state" and nondimensional.kind == "state"
    assert day.udunits == "day"
    assert nondimensional.udunits == "dimensionless"
    for contract, name in ((day, "snow_surface_age"), (nondimensional, "snow_nondimensional_age")):
        assert contract.owner == "snow_albedo"
        assert contract.incrementable is False
        assert registry.can_write(name, "snow_albedo")
        assert not registry.can_increment(name, "snow_albedo")
        assert contract.raw["numerics"]["lower_bound"] == 0.0
    functor = SnowAlbedo(name="snow_albedo_contract_probe")
    assert {"snow_surface_age", "snow_nondimensional_age"} <= set(functor.get_provides())
    assert "hydrosphere.snow.age_based_temperature_sensitivity" in functor.required_parameters()
    assert "hydrosphere.state_defaults.snow_nondimensional_age" not in functor.required_parameters()
    assert {form.semantic_name for form in functor.declared_forms()} >= {
        "snow_surface_age",
        "snow_nondimensional_age",
    }


def test_governed_call_reports_shortwave_partition_closure_without_false_energy_warning():
    functor = SnowAlbedo(name="snow_albedo_energy_partition")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = functor(
            _snow_age_probe_state(
                snow_nondimensional_age=torch.tensor([0.2, 3.0]),
                snow_depth=torch.tensor([0.5, 0.0]),
                snow_surface_temperature=torch.tensor([263.15, 275.15]),
                air_temperature=torch.tensor([263.15, 275.15]),
                solar_zenith_angle=torch.tensor([0.3, 1.2]),
                shortwave_down=torch.tensor([300.0, 500.0]),
            ),
            time=0.0,
            dt=3600.0,
        )
    violations = functor.check_conservation_laws({}, result)
    assert set(violations) == {"energy"}
    assert violations["energy"] <= 1.0e-6
    albedo = _unwrap_update(result["surface_albedo"])
    absorbed = result["diagnostics.snow_albedo"]["snow_absorbed_shortwave_fraction"]
    assert torch.allclose(albedo + absorbed, torch.ones_like(albedo), atol=1.0e-6)


def test_energy_partition_check_requires_the_routed_outputs():
    functor = SnowAlbedo(name="snow_albedo_energy_partition_guard")
    with pytest.raises(ValueError, match="surface_albedo"):
        functor.check_conservation_laws({}, {"surface_albedo": torch.tensor(0.8)})


# Round 35: the remaining default-shaped values of SnowAlbedo. The optional shortwave input
# is an explicit branch without a zero substitute, the step length is checked instead of
# clamped by a dimensionless epsilon, the field interface does not return a parameter in
# place of the committed albedo, and the structural Hodge albedo is named as a reference.


def test_absent_shortwave_omits_the_radiation_port_power():
    functor = SnowAlbedo()
    result = functor.forward(
        state=_snow_age_probe_state(snow_nondimensional_age=torch.tensor(0.5)),
        time=0.0,
        dt=86400.0,
    )
    diagnostics = result["diagnostics.snow_albedo"]
    assert "snow_absorbed_shortwave_fraction" in diagnostics
    assert "snow_albedo_radiation_port_power" not in diagnostics


@pytest.mark.parametrize("dt", [0.0, -60.0])
def test_non_positive_step_is_rejected(dt):
    functor = SnowAlbedo()
    with pytest.raises(ValueError, match="dt"):
        functor.forward(
            state=_snow_age_probe_state(snow_nondimensional_age=torch.tensor(0.5)),
            time=0.0,
            dt=dt,
        )


def test_seconds_per_day_is_not_clamped_by_a_dimensionless_epsilon():
    source = inspect.getsource(SnowAlbedo.forward)
    assert "torch.clamp(seconds_per_day" not in source


def test_no_duration_is_clamped_by_a_dimensionless_epsilon():
    # forward checks dt > 0, so the aging rates divide by the step itself.
    source = inspect.getsource(SnowAlbedo)
    assert "dt_safe" not in source
    assert "torch.clamp(dt" not in source


def test_field_queries_require_the_committed_albedo():
    functor = SnowAlbedo()
    coordinates = torch.zeros((1, 2))
    with pytest.raises(RuntimeError, match="surface_albedo"):
        functor.get_value_at(coordinates)
    with pytest.raises(RuntimeError, match="surface_albedo"):
        functor.interpolate(coordinates, torch.zeros((3, 2)))


def test_hodge_reference_albedo_is_named_as_a_reference_state():
    functor = SnowAlbedo()
    manager = get_global_manager()
    reference_key = "hydrosphere.snow.hodge_reference_surface_albedo"
    hodge = functor.build_hodge_contract(n_dofs=1)
    assert reference_key in hodge.parameter_keys
    assert reference_key in functor.metadata.parameters
    assert reference_key in functor.required_parameters()
    assert manager.get_parameter(reference_key) is not None
    assert manager.get_parameter("hydrosphere.snow.surface_albedo_default") is None
    assert [key for key in functor.required_parameters() if key.endswith("_default")] == []


def test_incident_shortwave_is_the_surface_downwelling_shortwave_variable():
    """The shortwave effort of the snow albedo port is the surface downwelling shortwave flux
    (global radiation at the lower boundary of the atmosphere, CF
    surface_downwelling_shortwave_flux_in_air), the variable shortwave_down: one CF identity has
    one variable, so the albedo model reads shortwave_down and no second name carries it."""
    from relic.contracts import load_contract_registry
    from relic.orchestration.ssot_catalog import SSOTCatalog
    from relic.scientific.hydrosphere import interface_mapping

    functor = SnowAlbedo()
    assert "shortwave_down" in functor.metadata.input_variables
    assert "incoming_shortwave_radiation" not in functor.metadata.input_variables
    assert functor.optional_input_variables == {"shortwave_down"}
    (port,) = functor.ports()
    assert port.effort_var == "shortwave_down"
    cf_name = "surface_downwelling_shortwave_flux_in_air"
    holders = [m.standard_name for m in interface_mapping.ALL_MAPPINGS if m.cf_standard_name == cf_name]
    assert holders == ["shortwave_down"]
    assert "incoming_shortwave_radiation" not in {m.standard_name for m in interface_mapping.ALL_MAPPINGS}
    assert load_contract_registry().get_contract("shortwave_down").cf_standard_name == cf_name
    assert SSOTCatalog.from_parameter_manager().get_by_standard_name(cf_name).name == "shortwave_down"
    power = functor.forward(
        state={
            "snowfall": torch.tensor(0.0),
            "air_temperature": torch.tensor(268.15),
            "snow_surface_temperature": torch.tensor(268.15),
            "solar_zenith_angle": torch.tensor(0.6),
            "snow_depth": torch.tensor(0.2),
            functor.age_state_key: torch.tensor(2.0),
            "shortwave_down": torch.tensor(300.0),
        },
        time=0.0,
        dt=86400.0,
    )["diagnostics.snow_albedo"]["snow_albedo_radiation_port_power"]
    assert float(power) > 0.0
