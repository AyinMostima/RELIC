"""
Unit tests for hydrosphere.snow.snowpack SnowPack functor.

Focus on parameter loading, precipitation partitioning, and SWE mass balance.
"""

import pytest
import torch
from pathlib import Path

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.parameters import get_global_manager
import relic.scientific.hydrosphere.snow.snowpack as snowpack_module
from relic.scientific.hydrosphere.snow.snowpack import SnowPack
from relic.core.state import UpdateItem
from relic.utils.physical_constants import get_physical_constant


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


@pytest.fixture(scope="module")
def parameter_manager():
    manager = get_global_manager()
    assert manager is not None
    return manager


def test_snowpack_parameters_loaded(parameter_manager):
    functor = SnowPack()

    expected_ddf = parameter_manager.get(
        "hydrosphere.snow.degree_day_factor",
        units="mm K-1 day-1",
    )
    expected_melt_threshold = parameter_manager.get(
        "hydrosphere.snow.temperature_threshold_warm",
        units="K",
    )
    expected_partition_threshold = parameter_manager.get(
        "hydrosphere.snow.rain_snow_temperature_threshold",
        units="K",
    )
    expected_density = parameter_manager.get(
        "hydrosphere.snow.snow_density",
        units="kg m-3",
    )

    assert pytest.approx(_to_float(expected_ddf)) == _to_float(functor.degree_day_factor)
    assert pytest.approx(_to_float(expected_melt_threshold)) == _to_float(functor.melt_temperature_threshold)
    assert pytest.approx(_to_float(expected_partition_threshold)) == _to_float(functor.rain_snow_temperature_threshold)
    assert pytest.approx(_to_float(expected_density)) == _to_float(functor.snow_density)


def test_physical_constants_are_loaded_through_ssot_api(monkeypatch):
    """Physical constants must bypass the scientific parameter accessor."""
    calls = []
    constants = {
        "seconds_per_day": 43200.0,
        "water_density": 800.0,
    }

    def fake_get_physical_constant(name):
        calls.append(name)
        return constants[name]

    monkeypatch.setattr(
        snowpack_module,
        "get_physical_constant",
        fake_get_physical_constant,
        raising=False,
    )

    functor = SnowPack()
    dt = 3600.0
    temperature_excess = 2.0
    swe_initial = 20.0
    result = functor.forward(
        state={
            "precipitation_flux": torch.tensor(0.0),
            "air_temperature": torch.tensor(
                _to_float(functor.melt_temperature_threshold) + temperature_excess
            ),
            "snow_water_equivalent": torch.tensor(swe_initial),
            "sublimation_flux": torch.tensor(0.0),
        },
        time=0.0,
        dt=dt,
    )

    expected_melt_mm_s = (
        _to_float(functor.degree_day_factor)
        * temperature_excess
        / constants["seconds_per_day"]
    )
    expected_swe = swe_initial - expected_melt_mm_s * dt
    # The melt depth rate becomes a mass flux through UnitConverter's single CF
    # liquid-water-equivalent rule, which reads water.density_liquid from the
    # physical-constant SSOT itself (not the module attribute patched above).
    expected_melt_flux = (
        expected_melt_mm_s
        * get_physical_constant("relic.physical.constants.water.density_liquid")
        / 1000.0
    )
    expected_snow_depth = (
        expected_swe
        / 1000.0
        * constants["water_density"]
        / _to_float(functor.snow_density)
    )
    hodge = functor.build_hodge_contract()

    assert calls == ["seconds_per_day", "water_density"]
    assert _to_float(functor.seconds_per_day) == pytest.approx(
        constants["seconds_per_day"]
    )
    assert _to_float(functor.water_density) == pytest.approx(
        constants["water_density"]
    )
    assert _to_float(_unwrap_update(result["snow_water_equivalent"])) == pytest.approx(
        expected_swe
    )
    assert _to_float(_unwrap_update(result["snow_melt_flux"])) == pytest.approx(
        expected_melt_flux
    )
    assert _to_float(_unwrap_update(result["snow_depth"])) == pytest.approx(
        expected_snow_depth
    )
    assert torch.allclose(
        hodge.matrix,
        torch.eye(
            hodge.matrix.shape[0],
            dtype=hodge.matrix.dtype,
            device=hodge.matrix.device,
        )
        # The Hodge mass-to-SWE factor comes from UnitConverter's single
        # liquid-water-equivalent rule (SSOT water.density_liquid), not from the
        # module attribute patched above.
        * (1000.0 / get_physical_constant("relic.physical.constants.water.density_liquid")),
    )
    assert not any(
        key.startswith("relic.physical.constants.")
        for key in functor.required_parameters()
    )
    assert hodge.parameter_keys == (
        "relic.physical.constants.water.density_liquid",
    )
    assert "seconds_per_day" not in functor.metadata.parameters
    import inspect

    assert "1000.0 / water_density" not in inspect.getsource(SnowPack.build_hodge_contract)
    assert "density_liquid" not in functor.metadata.parameters


def test_snowpack_partitioning_cold(parameter_manager):
    functor = SnowPack()
    rho_w = get_physical_constant("water_density")
    threshold = parameter_manager.get(
        "hydrosphere.snow.rain_snow_temperature_threshold",
        units="K",
    )

    precipitation_flux = torch.tensor(0.001, dtype=torch.float32)
    state = {
        "precipitation_flux": precipitation_flux,
        "air_temperature": torch.tensor(_to_float(threshold) - 2.0),
        "snow_water_equivalent": torch.tensor(0.0),
        "sublimation_flux": torch.tensor(0.0),
    }
    dt = 3600.0
    result = functor(state, time=0.0, dt=dt)

    snow_melt_flux = _unwrap_update(result["snow_melt_flux"])
    rain_flux = _unwrap_update(result["rain_flux"])
    snowfall = _unwrap_update(result["snowfall"])
    swe_new = _unwrap_update(result["snow_water_equivalent"])

    assert torch.isclose(snow_melt_flux, torch.zeros_like(snow_melt_flux), atol=1e-8)
    assert torch.isclose(rain_flux, torch.zeros_like(rain_flux), atol=1e-8)
    assert torch.isclose(snowfall, precipitation_flux.to(snowfall.dtype), atol=1e-8)

    expected_swe = precipitation_flux * dt * 1000.0 / _to_float(rho_w)
    expected_swe = expected_swe.to(swe_new.dtype)
    assert torch.isclose(swe_new, expected_swe, atol=1e-6)


def test_snowpack_melt_warm(parameter_manager):
    functor = SnowPack()
    ddf = _to_float(
        parameter_manager.get(
            "hydrosphere.snow.degree_day_factor",
            units="mm K-1 day-1",
        )
    )
    melt_threshold = _to_float(
        parameter_manager.get(
            "hydrosphere.snow.temperature_threshold_warm",
            units="K",
        )
    )
    seconds_per_day = _to_float(get_physical_constant("seconds_per_day"))
    rho_w = _to_float(get_physical_constant("water_density"))

    swe_initial = torch.tensor(50.0, dtype=torch.float32)
    state = {
        "precipitation_flux": torch.tensor(0.0),
        "air_temperature": torch.tensor(melt_threshold + 3.0),
        "snow_water_equivalent": swe_initial,
        "sublimation_flux": torch.tensor(0.0),
    }
    dt = 3600.0
    result = functor(state, time=0.0, dt=dt)

    snow_melt_flux = _unwrap_update(result["snow_melt_flux"])
    swe_new = _unwrap_update(result["snow_water_equivalent"])

    melt_mm_s = (ddf * 3.0) / seconds_per_day
    available_melt = _to_float(swe_initial) / dt
    melt_mm_s = min(melt_mm_s, available_melt)
    expected_swe = _to_float(swe_initial) - melt_mm_s * dt
    expected_melt_flux = melt_mm_s * rho_w / 1000.0

    assert torch.isclose(
        snow_melt_flux,
        torch.tensor(expected_melt_flux, dtype=snow_melt_flux.dtype),
        atol=1e-6,
    )
    assert torch.isclose(swe_new, torch.tensor(expected_swe, dtype=swe_new.dtype), atol=1e-6)


_SNOWPACK_STATE_DEFAULT_KEYS = (
    "hydrosphere.state_defaults.precipitation_flux",
    "hydrosphere.state_defaults.snow_water_equivalent",
    "hydrosphere.state_defaults.sublimation_flux",
)


def test_snowpack_requires_every_input(parameter_manager):
    """A missing input raises; no state default is substituted."""
    functor = SnowPack()
    assert set(functor.get_required_inputs()) == {
        "precipitation_flux",
        "air_temperature",
        "snow_water_equivalent",
        "sublimation_flux",
    }

    with pytest.raises(ValueError) as error:
        functor.forward({"air_temperature": torch.tensor(270.0)}, time=0.0, dt=3600.0)
    for name in ("precipitation_flux", "snow_water_equivalent", "sublimation_flux"):
        assert name in str(error.value)


def test_snowpack_reads_no_state_default_keys(parameter_manager):
    functor = SnowPack()

    assert not hasattr(functor, "_state_defaults")
    for key in _SNOWPACK_STATE_DEFAULT_KEYS + ("hydrosphere.state_defaults.air_temperature",):
        assert key not in functor.required_parameters()
    for key in _SNOWPACK_STATE_DEFAULT_KEYS:
        assert parameter_manager.get_parameter(key) is None, key


def test_outputs_are_contract_replace_items(parameter_manager):
    """Each state output is one REPLACE UpdateItem in its SCM contract unit written by
    snow_pack, so StateManager can check the unit against the contract."""
    from relic.contracts import load_contract_registry
    from relic.core.state import UpdateMode

    registry = load_contract_registry()
    functor = SnowPack()
    state = {
        "precipitation_flux": torch.tensor(0.001),
        "air_temperature": torch.tensor(271.0),
        "snow_water_equivalent": torch.tensor(20.0),
        "sublimation_flux": torch.tensor(0.0),
    }
    result = functor(state, time=0.0, dt=3600.0)
    for name in ("snow_water_equivalent", "snow_depth", "snow_melt_flux", "rain_flux", "snowfall"):
        item = result[name]
        assert isinstance(item, UpdateItem), name
        assert item.mode == UpdateMode.REPLACE, name
        assert item.per_timestep is False, name
        assert item.unit == registry.get_contract(name).udunits, name
        assert item.source_functor == functor.name, name



# Variable -> the hydrosphere.state_defaults key its reader substituted before the
# state defaults were retired.
_RETIRED_SNOW_STATE_DEFAULTS = {
    "snow_water_equivalent": "hydrosphere.state_defaults.snow_water_equivalent",
    "snowfall": "hydrosphere.state_defaults.snowfall_flux",
    "snow_depth": "hydrosphere.state_defaults.snow_depth",
}


def test_snow_contracts_describe_required_inputs_without_state_defaults(parameter_manager):
    """snow_pack reads snow_water_equivalent and snow_albedo reads snowfall and snow_depth
    as required inputs: a missing value raises ValueError naming it and no state default
    is substituted (snow_albedo: test_snow_albedo_requires_every_forcing_input). Their
    SCM contracts say so and name no retired state-default key."""
    from relic.contracts import load_contract_registry

    functor = SnowPack()
    assert "snow_water_equivalent" in functor.get_required_inputs()
    with pytest.raises(ValueError, match="snow_water_equivalent"):
        functor.forward({}, time=0.0, dt=3600.0)

    registry = load_contract_registry()
    for variable, retired_key in _RETIRED_SNOW_STATE_DEFAULTS.items():
        assert parameter_manager.get_parameter(retired_key) is None, retired_key
        raw = registry.get_contract(variable).raw
        text = f"{raw['description']} {raw['provenance']['note']}"
        assert "state_defaults" not in text, variable
        assert "required input" in text and "raises ValueError" in text, variable


def test_snow_depth_takes_the_mm_to_m_step_from_the_unit_ssot(parameter_manager):
    """d = SWE[m] * rho_w / rho_snow, with the SWE mm -> m step from the unit SSOT
    (UnitConverter) instead of a literal 1000 in the forward pass."""
    import inspect

    from relic.data.standards.unit_converter import unit_converter

    assert "1000.0" not in inspect.getsource(SnowPack.forward)
    functor = SnowPack()
    swe = torch.tensor(40.0)
    state = {
        "precipitation_flux": torch.tensor(0.0),
        "air_temperature": torch.tensor(250.0),
        "snow_water_equivalent": swe,
        "sublimation_flux": torch.tensor(0.0),
    }
    result = functor(state, time=0.0, dt=3600.0)
    rho_w = _to_float(get_physical_constant("water_density"))
    rho_snow = _to_float(
        parameter_manager.get("hydrosphere.snow.snow_density", units="kg m-3")
    )
    expected = unit_converter.convert(swe, "snow_water_equivalent", "mm", "m") * rho_w / rho_snow
    depth = _unwrap_update(result["snow_depth"])
    assert torch.isclose(depth, expected.to(depth.dtype), rtol=1e-6)


class TestSnowPackPhase8:
    """Test Phase 8 IPHS-DEC contract for SnowPack."""

    def test_phase8_iphs_protocol(self):
        """SnowPack should expose mass-balance field IPHS hooks."""
        functor = SnowPack()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.snow.degree_day_factor" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "snow_water_equivalent",
            "snowfall_mass_flux",
            "snow_melt_mass_flux",
            "sublimation_mass_flux",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "snowpack_accumulation_port",
            "snowpack_melt_release_port",
        }

    def test_phase8_certificate_is_valid(self):
        """SnowPack Phase 8 certificate should be complete."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_snow_snowpack_snow_pack.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == "relic.scientific.hydrosphere.snow.snowpack.SnowPack"
        assert certificate.canonical_role == "field"

    def test_projection_residual_diagnostics_are_routed(self):
        """Melt and sublimation capacity residuals should be exposed."""
        functor = SnowPack()
        result = functor.forward(
            state={
                "precipitation_flux": torch.tensor(0.0),
                "air_temperature": torch.tensor(float(functor.melt_temperature_threshold) + 20.0),
                "snow_water_equivalent": torch.tensor(1.0),
                "sublimation_flux": torch.tensor(0.01),
            },
            time=0.0,
            dt=86400.0,
        )

        diagnostics = result["diagnostics.snow_pack"]
        assert "melt_capacity_residual_mm_s" in diagnostics
        assert "sublimation_capacity_residual_mm_s" in diagnostics
        assert "swe_nonnegative_projection_residual_mm" in diagnostics
        assert torch.isfinite(diagnostics["melt_capacity_residual_mm_s"]).all()
        assert torch.isfinite(diagnostics["sublimation_capacity_residual_mm_s"]).all()
        assert torch.all(diagnostics["melt_capacity_residual_mm_s"] >= 0.0)
        assert torch.all(diagnostics["sublimation_capacity_residual_mm_s"] >= 0.0)
