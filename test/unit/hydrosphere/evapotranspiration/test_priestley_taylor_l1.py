"""
L1 Unit Tests for Priestley-Taylor Evapotranspiration Module

This test suite validates the current RELIC Priestley-Taylor ET functor
contract (TransportMechanismFunctor returning flux diagnostics in SI units).

References:
- Priestley, C.H.B. and Taylor, R.J. (1972). Monthly Weather Review, 100(2), 81-92.
- Allen, R.G. et al. (1998). FAO-56 Crop Evapotranspiration.

Notes:
- All tests use ASCII-only text in strings and comments for cross-platform stability.
"""

import pytest
import torch
import warnings
from pathlib import Path

import relic.scientific.hydrosphere.evapotranspiration.priestley_taylor as priestley_taylor_module
from relic.core.abstractions import TransportMechanism
from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import FunctorBase, IPHSDECFunctor
from relic.core.functor.functor_exceptions import ExecutionError
from relic.core.state import UpdateItem
from relic.parameters import get_global_manager
from relic.parameters.spec import ParameterSpec
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import PriestleyTaylorET


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def restore_parameter_owner(restore_global_parameter_manager):
    yield


@pytest.fixture
def pt_model(device: torch.device) -> PriestleyTaylorET:
    return PriestleyTaylorET(name="test_priestley_taylor", device=device)


@pytest.fixture
def base_state() -> dict:
    return {
        "net_radiation": torch.tensor(400.0, dtype=torch.float32),
        "air_temperature": torch.tensor(298.15, dtype=torch.float32),
        "atmospheric_pressure": torch.tensor(101325.0, dtype=torch.float32),
    }


class TestPriestleyTaylorET:
    def test_physical_constants_are_loaded_through_ssot_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        constants = {
            "latent_heat_vaporization": 2_400_000.0,
            "air_specific_heat_cp": 1_000.0,
            "water_molecular_weight": 18.0,
            "air_molecular_weight": 30.0,
            "relic.physical.constants.psychrometrics.tetens_base_pressure": 610.0,
            "relic.physical.constants.psychrometrics.tetens_coefficient_a": 17.0,
            "relic.physical.constants.psychrometrics.tetens_coefficient_b": 237.0,
            "relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff": 4098.0,
            "celsius_to_kelvin": 273.0,
        }

        def fake_get_physical_constant(name: str) -> float:
            calls.append(name)
            return constants[name]

        monkeypatch.setattr(
            priestley_taylor_module,
            "get_physical_constant",
            fake_get_physical_constant,
            raising=False,
        )

        model = PriestleyTaylorET()
        hodge = model.build_hodge_contract()

        assert calls == list(constants)
        assert float(model.latent_heat_vaporization) == pytest.approx(2_400_000.0)
        assert float(model.specific_heat_air) == pytest.approx(1_000.0)
        assert float(model.molecular_weight_ratio) == pytest.approx(18.0 / 30.0)
        assert float(model.tetens_base_pressure) == pytest.approx(610.0)
        assert float(model.tetens_coeff_a) == pytest.approx(17.0)
        assert float(model.tetens_coeff_b) == pytest.approx(237.0)
        assert float(model.slope_coefficient) == pytest.approx(4098.0)
        assert float(model.celsius_to_kelvin) == pytest.approx(273.0)
        assert not any(
            key.startswith("relic.physical.constants.")
            or key.startswith(
                (
                    "hydrosphere.physical.latent_heat_vaporization_reference",
                    "hydrosphere.physical.specific_heat_air_constant_pressure",
                    "hydrosphere.physical.molecular_weight_ratio_water_air",
                )
            )
            for key in model.required_parameters()
        )
        assert "hydrosphere.physical.latent_heat_vaporization_reference" not in (
            hodge.parameter_keys
        )
        assert {
            "relic.physical.constants.water.latent_heat_vaporization",
            "relic.physical.constants.air.specific_heat_cp",
            "relic.physical.constants.water.molecular_weight",
            "relic.physical.constants.air.molecular_weight_dry",
        }.issubset(set(hodge.parameter_keys))

    def test_dynamic_alpha_uses_canonical_pressure_unit_converter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        class RecordingConverter:
            def convert(
                self,
                data: torch.Tensor,
                variable_name: str,
                source_units: str,
                target_units: str,
            ) -> torch.Tensor:
                calls.append((variable_name, source_units, target_units))
                return data / 1000.0

        monkeypatch.setattr(
            priestley_taylor_module,
            "unit_converter",
            RecordingConverter(),
            raising=False,
        )
        model = PriestleyTaylorET()

        alpha = model._compute_dynamic_alpha(
            {"air_temperature": torch.tensor(298.15)},
            vapor_pressure_deficit=torch.tensor(2000.0),
        )

        assert calls == [("vapor_pressure_deficit", "Pa", "kPa")]
        assert torch.isfinite(alpha).all()

    def test_obsolete_hydrosphere_physical_constant_duplicates_are_absent(
        self,
    ) -> None:
        manager = get_global_manager()
        duplicate_keys = (
            "hydrosphere.physical.latent_heat_vaporization_reference",
            "hydrosphere.physical.specific_heat_air_constant_pressure",
            "hydrosphere.physical.molecular_weight_ratio_water_air",
        )

        assert {
            key: manager.get_parameter(key)
            for key in duplicate_keys
            if manager.get_parameter(key) is not None
        } == {}

    @pytest.mark.parametrize("pressure_alias", ["air_pressure", "surface_pressure"])
    def test_atmospheric_pressure_aliases_are_rejected(
        self,
        pt_model: PriestleyTaylorET,
        base_state: dict,
        pressure_alias: str,
    ) -> None:
        state = dict(base_state)
        state[pressure_alias] = state.pop("atmospheric_pressure")

        with pytest.raises(ValueError, match="atmospheric_pressure"):
            pt_model.forward(state, time=0.0, dt=3600.0)

    def test_latent_heat_conversion_uses_process_specific_conservation_hook(
        self,
        pt_model: PriestleyTaylorET,
        base_state: dict,
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pt_model(base_state, time=0.0, dt=3600.0)

        assert "evapotranspiration_flux" in result
        assert "latent_heat_flux" in result
        assert set(pt_model.metadata.conservation_laws) == {"energy"}
        assert pt_model.check_conservation_laws(base_state, result)["energy"] == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
        assert [item for item in caught if issubclass(item.category, UserWarning)] == []

        perturbed = dict(result)
        perturbed["latent_heat_flux"] = (
            _unwrap_update(result["latent_heat_flux"]) + 1.0
        )
        assert pt_model.check_conservation_laws(base_state, perturbed)["energy"] > 0.0

    @pytest.mark.parametrize(
        ("key", "value", "error_type"),
        [
            ("evapotranspiration_flux", "invalid", TypeError),
            ("latent_heat_flux", torch.empty(0), ValueError),
            ("latent_heat_flux", torch.tensor(float("nan")), ValueError),
            ("latent_heat_flux", torch.tensor(float("inf")), ValueError),
        ],
    )
    def test_latent_heat_conversion_conservation_hook_fails_fast(
        self,
        pt_model: PriestleyTaylorET,
        base_state: dict,
        key: str,
        value,
        error_type: type[Exception],
    ) -> None:
        result = pt_model.forward(base_state, time=0.0, dt=3600.0)
        invalid = dict(result)
        invalid[key] = value

        with pytest.raises(error_type):
            pt_model.check_conservation_laws(base_state, invalid)

    @pytest.mark.parametrize(
        "missing_key",
        ["evapotranspiration_flux", "latent_heat_flux"],
    )
    def test_latent_heat_conversion_conservation_hook_requires_both_ports(
        self,
        pt_model: PriestleyTaylorET,
        base_state: dict,
        missing_key: str,
    ) -> None:
        result = pt_model.forward(base_state, time=0.0, dt=3600.0)
        result.pop(missing_key)

        with pytest.raises(KeyError, match=missing_key):
            pt_model.check_conservation_laws(base_state, result)

    def test_class_inheritance(self, pt_model: PriestleyTaylorET) -> None:
        assert isinstance(pt_model, FunctorBase)
        assert isinstance(pt_model, TransportMechanism)
        assert hasattr(pt_model, "forward")
        assert hasattr(pt_model, "forward")
        assert hasattr(pt_model, "forward")

    def test_model_initialization_device(self, device: torch.device) -> None:
        model = PriestleyTaylorET(name="test_model", device=device)
        assert model.name == "test_model"
        assert getattr(model, "device") == device

    def test_basic_et_calculation(self, pt_model: PriestleyTaylorET, base_state: dict) -> None:
        result = pt_model.forward(base_state, time=43200.0, dt=3600.0)
        assert isinstance(result, dict)
        assert "evapotranspiration_flux" in result
        et = _unwrap_update(result["evapotranspiration_flux"])
        assert isinstance(et, torch.Tensor)
        assert torch.all(torch.isfinite(et))
        assert torch.all(et >= 0.0)

    def test_radiation_dependency(self, pt_model: PriestleyTaylorET, base_state: dict) -> None:
        low = dict(base_state)
        low["net_radiation"] = torch.tensor(200.0)
        high = dict(base_state)
        high["net_radiation"] = torch.tensor(600.0)

        et_low = _unwrap_update(
            pt_model.forward(low, time=43200.0, dt=3600.0)["evapotranspiration_flux"]
        )
        et_high = _unwrap_update(
            pt_model.forward(high, time=43200.0, dt=3600.0)["evapotranspiration_flux"]
        )
        assert torch.all(et_high >= et_low)

    def test_temperature_dependency(self, pt_model: PriestleyTaylorET, base_state: dict) -> None:
        cool = dict(base_state)
        cool["air_temperature"] = torch.tensor(288.15)
        warm = dict(base_state)
        warm["air_temperature"] = torch.tensor(308.15)

        et_cool = _unwrap_update(
            pt_model.forward(cool, time=43200.0, dt=3600.0)["evapotranspiration_flux"]
        )
        et_warm = _unwrap_update(
            pt_model.forward(warm, time=43200.0, dt=3600.0)["evapotranspiration_flux"]
        )
        assert torch.all(et_warm >= et_cool)

    def test_output_keys_are_stable(self, pt_model: PriestleyTaylorET, base_state: dict) -> None:
        result = pt_model.forward(base_state, time=43200.0, dt=3600.0)
        assert "evapotranspiration_flux" in result
        assert "diagnostics.evapotranspiration" in result
        diagnostics = result["diagnostics.evapotranspiration"]
        expected_keys = {
            "potential_evapotranspiration",
            "equilibrium_evapotranspiration",
            "alpha_coefficient",
            "equilibrium_fraction",
            "advection_factor",
            "available_energy",
            "slope_saturation_vapor_pressure",
            "psychrometric_constant",
        }
        assert expected_keys.issubset(set(diagnostics.keys()))

    def test_forward_input_dict_immutability(self, pt_model: PriestleyTaylorET) -> None:
        """forward must not mutate the caller-provided state dictionary."""
        state = {
            "net_radiation": torch.tensor(320.0),
            "air_temperature": torch.tensor(299.15),
            "atmospheric_pressure": torch.tensor(101325.0),
        }

        state_before = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }

        result = pt_model.forward(state, time=0.0, dt=3600.0)
        assert isinstance(result, dict)

        assert set(state.keys()) == set(state_before.keys())
        for key, value in state_before.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(state[key], value)
            else:
                assert state[key] == value


class TestPriestleyTaylorPhase8:
    """Test Phase 8 IPHS contract."""

    def test_phase8_iphs_protocol(self) -> None:
        """Test structural IPHS hooks."""
        functor = PriestleyTaylorET()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.evapotranspiration.alpha_pt" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "equilibrium_evapotranspiration_flux",
            "latent_heat_flux",
            "available_energy",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "priestley_taylor_equilibrium_flux",
            "priestley_taylor_latent_heat_flux",
        }

    def test_phase8_certificate_is_valid(self) -> None:
        """Test Phase 8 certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_evapotranspiration_priestley_taylor_priestley_taylor_et.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == (
            "relic.scientific.hydrosphere.evapotranspiration.priestley_taylor.PriestleyTaylorET"
        )
        assert certificate.canonical_role == "transport"

    def test_energy_residual_diagnostics_are_routed(self, base_state: dict) -> None:
        """Test residual diagnostics are exposed."""
        functor = PriestleyTaylorET()
        result = functor.forward(base_state, time=43200.0, dt=3600.0)

        diagnostics = result["diagnostics.evapotranspiration"]
        assert "available_energy_residual" in diagnostics
        assert "negative_et_projection_residual" in diagnostics
        assert torch.isfinite(diagnostics["available_energy_residual"]).all()
        assert torch.all(diagnostics["negative_et_projection_residual"] >= 0.0)

    @pytest.mark.parametrize("missing", ["air_temperature", "atmospheric_pressure"])
    def test_forward_has_no_input_fallback(self, base_state: dict, missing: str) -> None:
        """The Hodge reference state is never substituted for a missing forcing input."""
        functor = PriestleyTaylorET()
        state = {key: value for key, value in base_state.items() if key != missing}

        with pytest.raises((KeyError, ValueError), match=missing):
            functor.forward(state, time=43200.0, dt=3600.0)



# Round 35: the static Hodge contract is assembled at a named reference state, proves its
# denominators positive instead of clamping them with the dimensionless epsilon_divide, and
# the Functor reads no transport attribute that no computation uses.
_PT_REFERENCE_KEYS = (
    "hydrosphere.evapotranspiration.hodge_reference_air_temperature",
    "hydrosphere.evapotranspiration.hodge_reference_atmospheric_pressure",
)


def test_hodge_reference_state_is_named_as_a_reference() -> None:
    functor = PriestleyTaylorET()
    manager = get_global_manager()
    required = set(functor.required_parameters())
    assert set(_PT_REFERENCE_KEYS) <= required
    assert set(_PT_REFERENCE_KEYS) <= set(functor.metadata.parameters)
    assert tuple(functor.build_hodge_contract(n_dofs=1).metric_keys) == _PT_REFERENCE_KEYS
    for key in _PT_REFERENCE_KEYS:
        assert manager.get_parameter(key) is not None
    for retired in (
        "hydrosphere.state_defaults.air_temperature",
        "hydrosphere.evapotranspiration.atmospheric_pressure_diagnostic_default",
    ):
        assert retired not in required
        assert manager.get_parameter(retired) is None
    assert [key for key in required if key.endswith("_default")] == []


def test_hodge_contract_has_no_dimensionless_division_guard(monkeypatch) -> None:
    functor = PriestleyTaylorET()
    assert "validation.numerical.epsilon_divide" not in functor.required_parameters()
    assert "validation.numerical.epsilon_divide" not in functor.metadata.parameters
    monkeypatch.setattr(
        priestley_taylor_module,
        "compute_psychrometric_constant",
        lambda *args, **kwargs: torch.tensor(-1.0e6),
    )
    with pytest.raises(ValueError, match="denominator"):
        functor.build_hodge_contract(n_dofs=1)


def test_no_unused_transport_attribute_is_read() -> None:
    functor = PriestleyTaylorET()
    required = set(functor.required_parameters())
    for key in (
        "hydrosphere.transport.water_vapor_diffusivity",
        "hydrosphere.transport.gas_tortuosity",
        "hydrosphere.transport.gas_porosity",
    ):
        assert key not in required
        assert key not in functor.metadata.parameters
    # Energy-limited equilibrium evaporation has no scalar diffusivity law.
    assert functor.diffusivity is None


def test_moisture_stress_reads_the_formal_soil_parameters() -> None:
    """theta_s and theta_r of the moisture stress are the formal soil parameters
    hydrosphere.soil_moisture.porosity and residual_water_content (the pair Richards
    reads); no representative-loam default stands in for the soil."""
    model = PriestleyTaylorET()
    manager = model.get_parameter_manager()
    for key in (
        "hydrosphere.physical.default_saturated_water_content",
        "hydrosphere.physical.default_residual_water_content",
    ):
        assert key not in model.metadata.parameters
        assert manager.get_parameter(key) is None
    assert {
        "hydrosphere.soil_moisture.porosity",
        "hydrosphere.soil_moisture.residual_water_content",
    } <= set(model.metadata.parameters)
    assert not hasattr(model, "default_saturated_water_content")
    assert float(model.saturated_water_content) == pytest.approx(
        float(manager.get("hydrosphere.soil_moisture.porosity", required=True))
    )


def test_observed_ground_heat_is_a_required_signed_state_port(base_state):
    model = PriestleyTaylorET(config={"parameters": {"soil_heat_flux_source": "state"}})
    assert "soil_heat_flux" in model.get_required_inputs()
    ground = torch.tensor(-25.0, dtype=torch.float32, requires_grad=True)
    result = model({**base_state, "soil_heat_flux": ground}, time=0.0, dt=1800.0)
    diagnostic = result["diagnostics.evapotranspiration"]
    assert diagnostic["soil_heat_flux"].dtype == torch.float32
    torch.testing.assert_close(diagnostic["soil_heat_flux"], ground.to(torch.float32))
    torch.testing.assert_close(diagnostic["available_energy"], base_state["net_radiation"].to(torch.float32) - ground.to(torch.float32))
    et = _unwrap_update(result["evapotranspiration_flux"])
    gradient, = torch.autograd.grad(et, ground)
    expected = -diagnostic["alpha_coefficient"] * diagnostic["equilibrium_fraction"] / model.latent_heat_vaporization
    torch.testing.assert_close(gradient, expected.to(gradient.dtype), rtol=1.0e-6, atol=0.0)
    model.soil_heat_fraction = torch.tensor(0.9)
    again = model({**base_state, "soil_heat_flux": ground}, time=0.0, dt=1800.0)
    torch.testing.assert_close(_unwrap_update(again["evapotranspiration_flux"]), et)


@pytest.mark.parametrize("shape", [(), (1, 1)])
def test_float64_driver_and_alpha_derivatives_keep_input_precision(base_state, shape, fresh_global_parameter_manager):
    manager = get_global_manager()
    key = "hydrosphere.evapotranspiration.alpha_pt"
    specification = manager.get_parameter(key).to_dict()
    specification.update(dtype="float64", requires_grad=True, value=1.26)
    manager.register_parameter(ParameterSpec.from_dict(specification))
    alpha = manager.nn_parameters[key]
    model = PriestleyTaylorET(config={"soil_heat_flux_source": "state"})
    assert model.alpha_pt is alpha
    state = {key: value.double().reshape(shape) for key, value in base_state.items()}
    state["soil_heat_flux"] = torch.full(shape, 40.0, dtype=torch.float64, requires_grad=True)
    output = model(state, time=0.0, dt=1800.0)
    latent = _unwrap_update(output["latent_heat_flux"])
    diagnostic = output["diagnostics.evapotranspiration"]
    assert latent.dtype == torch.float64
    assert diagnostic["soil_heat_flux"].dtype == torch.float64
    assert diagnostic["alpha_coefficient"].dtype == torch.float64
    alpha_gradient, ground_gradient = torch.autograd.grad(latent.sum(), (alpha, state["soil_heat_flux"]))
    # References: Priestley and Taylor (1972), fixed-alpha positive available-energy branch.
    # Mathematical expression: LE=alpha*f*(Rn-G); dLE/dalpha=f*(Rn-G); dLE/dG=-alpha*f.
    # Parameter meanings: f is the equilibrium fraction, Rn net radiation, G ground heat, alpha coefficient.
    # Units: LE,Rn,G [W m-2], alpha,f [1]; alpha derivative [W m-2], G derivative [1].
    # Domain/constraints: double inputs, positive available energy, alpha perturbations inside YAML bounds.
    # Sign convention: positive LE is outward; increasing G reduces outward latent heat.
    expected_alpha = (diagnostic["equilibrium_fraction"] * diagnostic["available_energy"]).sum()
    expected_ground = -alpha * diagnostic["equilibrium_fraction"]
    torch.testing.assert_close(alpha_gradient, expected_alpha, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ground_gradient, expected_ground, rtol=1e-12, atol=1e-12)
    for relative_step in (1e-3, 1e-4, 1e-5, 1e-6):
        h = 1.26 * relative_step
        values = []
        for offset in (h, -h):
            with torch.no_grad():
                alpha.fill_(1.26 + offset)
            values.append(_unwrap_update(model(state, time=0.0, dt=1800.0)["latent_heat_flux"]).detach())
        numeric = ((values[0] - values[1]) / (2 * h)).sum()
        torch.testing.assert_close(numeric, alpha_gradient, rtol=1e-8, atol=1e-10)
    with torch.no_grad():
        alpha.fill_(1.26)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64])
def test_tensor_conversion_keeps_float32_floor_and_double_precision(pt_model, dtype):
    value = torch.tensor(1, dtype=dtype)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    converted = pt_model._ensure_tensor(value)
    assert converted.dtype == expected_dtype
    assert converted.device == pt_model.device
    assert converted.item() == 1
    assert pt_model._ensure_tensor(1.0).dtype == torch.float32


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_observed_ground_heat_never_falls_back_to_a_fraction(base_state, value):
    model = PriestleyTaylorET(config={"parameters": {"soil_heat_flux_source": "state"}})
    state = dict(base_state)
    if value is not None:
        state["soil_heat_flux"] = torch.tensor(value)
    with pytest.raises(ExecutionError, match="soil_heat_flux") as failure:
        model(state, time=0.0, dt=1800.0)
    assert isinstance(failure.value.__cause__, ValueError)


def test_fraction_ground_heat_is_explicit_and_keeps_the_original_formula(base_state):
    model = PriestleyTaylorET()
    assert model.soil_heat_flux_source == "fraction"
    assert "soil_heat_flux" in model.metadata.input_variables
    assert "soil_heat_flux" not in model.get_required_inputs()
    result = model(base_state, time=0.0, dt=1800.0)
    diagnostic = result["diagnostics.evapotranspiration"]
    torch.testing.assert_close(diagnostic["soil_heat_flux"], model.soil_heat_fraction * base_state["net_radiation"].to(torch.float32))
    extra = model({**base_state, "soil_heat_flux": torch.tensor(300.0)}, time=0.0, dt=1800.0)
    torch.testing.assert_close(_unwrap_update(result["evapotranspiration_flux"]), _unwrap_update(extra["evapotranspiration_flux"]))


def test_unknown_ground_heat_source_is_rejected():
    with pytest.raises(ValueError, match="soil_heat_flux_source"):
        PriestleyTaylorET(config={"parameters": {"soil_heat_flux_source": "guess"}})


def test_ground_heat_above_radiation_preserves_energy_and_projection_diagnostics(base_state):
    model = PriestleyTaylorET(config={"parameters": {"soil_heat_flux_source": "state"}})
    result = model({**base_state, "soil_heat_flux": torch.tensor(500.0)}, time=0.0, dt=1800.0)
    diagnostic = result["diagnostics.evapotranspiration"]
    assert float(diagnostic["available_energy"]) == pytest.approx(-100.0)
    assert float(diagnostic["available_energy_residual"]) == 0.0
    assert float(diagnostic["negative_et_projection_residual"].detach()) > 0.0


@pytest.mark.parametrize("source,include_ground", [("fraction", False), ("state", False), ("state", True)])
def test_ground_heat_source_requiredness_survives_strict_graph_execution(base_state, tmp_path, source, include_ground):
    from relic.orchestration.compiler import compile_graph
    from relic.orchestration.executor import GraphExecutor
    from relic.orchestration.graph_ir import GraphIR, NodeSpec
    from relic.orchestration.node_registry import NodeRegistry
    from relic.orchestration.ssot_catalog import SSOTCatalog

    config = {"soil_heat_flux_source": source, "execution_group": "fast"}
    functor = PriestleyTaylorET(config=config)
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    class_type = f"Functor:{PriestleyTaylorET.__module__}.{PriestleyTaylorET.__name__}"
    definition = registry.register_functor_instance(functor, node_type_id=class_type)
    assert definition.inputs["soil_heat_flux"].required is (source == "state")
    initial = dict(base_state)
    if include_ground:
        initial["soil_heat_flux"] = torch.tensor(-25.0)
    graph = GraphIR(schema_version="1.0.0", nodes={
        "forcing": NodeSpec(class_type="State:ExternalForcing", outputs=list(initial)),
        "priestley_taylor_et": NodeSpec(
            class_type=class_type, config=config,
            inputs={name: {"state": name} for name in initial},
            outputs=["evapotranspiration_flux", "latent_heat_flux", "diagnostics.evapotranspiration"],
        ),
    })
    if source == "state" and not include_ground:
        with pytest.raises(ValueError, match="soil_heat_flux"):
            compile_graph(graph, registry, strict=True, auto_adapt=False)
        return
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    executor = GraphExecutor(plan, registry=registry, output_dir=tmp_path,
                             simulator_config={"device": "cpu", "auto_report": False})
    final = executor.execute(initial, n_steps=1, dt=1800.0)
    assert torch.isfinite(final["latent_heat_flux"]).all()
    torch.testing.assert_close(final["latent_heat_flux"], final["evapotranspiration_flux"] * functor.latent_heat_vaporization)


@pytest.mark.parametrize("candidate_kind", ["leaf_tensor", "calibrator_parameter"])
def test_explicit_alpha_candidates_keep_independent_gradients_and_manager_owner(
    base_state, candidate_kind,
):
    manager = get_global_manager()
    key = "hydrosphere.evapotranspiration.alpha_pt"
    binding = manager.get(key, expected_type=torch.Tensor, required=True)
    original = binding.detach().clone()
    candidates = [torch.tensor(value, dtype=torch.float32, requires_grad=True)
                  for value in (0.9, 1.1)]
    if candidate_kind == "calibrator_parameter":
        candidates = [torch.nn.Parameter(value) for value in candidates]
    models = [PriestleyTaylorET(config={
        "parameters": {"alpha_pt": {"value": candidate, "units": "dimensionless"}},
        "soil_heat_flux_source": "state",
    }) for candidate in candidates]
    for model, candidate in zip(models, candidates):
        assert model.alpha_pt is candidate
        assert model.manager_binding_key(model.alpha_pt) is None
        assert key not in model.held_manager_parameters()
        if candidate_kind == "calibrator_parameter":
            assert dict(model.named_parameters())["alpha_pt"] is candidate
        else:
            assert "alpha_pt" not in dict(model.named_parameters())
    state = {**base_state, "soil_heat_flux": torch.tensor(40.0, dtype=torch.float32)}
    outputs = [model(state, time=0.0, dt=1800.0) for model in models]
    # References: Priestley and Taylor (1972); canonical isolated config-candidate contract.
    # Mathematical expression: LE=alpha*Delta/(Delta+gamma)*(Rn-G),
    #   dLE/dalpha=Delta/(Delta+gamma)*(Rn-G) for the positive unprojected case.
    # Parameter meanings: alpha is the independent candidate, Delta/gamma the
    #   production psychrometric diagnostics, and Rn-G the available energy.
    # Units: LE/Rn/G [W m-2], alpha and equilibrium fraction [1].
    # Domain/constraints: fixed alpha, positive available energy, unchanged manager
    #   binding; each objective has a separately constructed candidate Functor.
    # Sign convention: positive LE is outward; candidate derivatives are positive here.
    for index, (model, output) in enumerate(zip(models, outputs)):
        latent = _unwrap_update(output["latent_heat_flux"])
        own_gradient, other_gradient = torch.autograd.grad(
            latent, (candidates[index], candidates[1 - index]), allow_unused=True,
        )
        diagnostic = output["diagnostics.evapotranspiration"]
        expected_gradient = diagnostic["equilibrium_fraction"] * diagnostic["available_energy"]
        assert own_gradient is not None
        assert other_gradient is None
        torch.testing.assert_close(own_gradient, expected_gradient, rtol=2.0e-6, atol=0.0)
        torch.testing.assert_close(diagnostic["alpha_coefficient"], candidates[index])
        torch.testing.assert_close(
            diagnostic["potential_evapotranspiration"],
            _unwrap_update(output["evapotranspiration_flux"]),
        )
    assert float(_unwrap_update(outputs[1]["latent_heat_flux"]).detach()) > float(
        _unwrap_update(outputs[0]["latent_heat_flux"]).detach()
    )
    assert manager.get(key, expected_type=torch.Tensor, required=True) is binding
    torch.testing.assert_close(binding.detach(), original, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("candidate", [0.59, 1.61, float("nan"), float("inf")])
def test_explicit_alpha_candidate_obeys_original_yaml_bounds(candidate):
    with pytest.raises(ValueError, match="alpha_pt.*outside bounds"):
        PriestleyTaylorET(config={"parameters": {"alpha_pt": torch.tensor(candidate, dtype=torch.float32)}})


def test_explicit_alpha_candidate_rejects_incompatible_units():
    manager = get_global_manager()
    key = "hydrosphere.evapotranspiration.alpha_pt"
    original_units = manager.get_parameter(key).units
    with pytest.raises(ValueError):
        PriestleyTaylorET(config={"parameters": {"alpha_pt": {"value": 0.9, "units": "m"}}})
    assert manager.get_parameter(key).units == original_units


def test_alpha_without_override_keeps_same_device_shared_manager_binding():
    manager = get_global_manager()
    key = "hydrosphere.evapotranspiration.alpha_pt"
    binding = manager.get(key, expected_type=torch.Tensor, required=True)
    # Match the existing binding's device. A preceding strict-graph test may
    # have migrated it; cross-device placeholder conversion is a separate core contract.
    first = PriestleyTaylorET(device=binding.device)
    second = PriestleyTaylorET(device=binding.device)
    assert first.alpha_pt is binding and second.alpha_pt is binding
    assert first.held_manager_parameters()[key] is binding
    assert second.held_manager_parameters()[key] is binding
    assert "alpha_pt" not in dict(first.named_parameters())
    assert "alpha_pt" not in first.state_dict()


# Sub-freezing option (2026-10-10). The admitted air-temperature domain is the manager value
# of hydrosphere.numerical.temperature_bounds, exactly as before. Inside a domain that extends
# below 273.15 K the default keeps the single liquid Tetens form; `ice` selects the Murray
# (1967) ice coefficients and the latent heat of sublimation cell by cell. The default domain
# still rejects such temperatures, so default behaviour is unchanged.
_ICE_PHASE_CONFIG = {"parameters": {"sub_freezing_phase": "ice"}}
_TEMPERATURE_DOMAIN_KEY = "hydrosphere.numerical.temperature_bounds"


@pytest.fixture
def sub_freezing_domain():
    """Lower the shared manager domain to its declared minimum for one test and restore it."""
    manager = get_global_manager()
    original = list(manager.get(_TEMPERATURE_DOMAIN_KEY))
    manager.set_value(_TEMPERATURE_DOMAIN_KEY, [223.15, 323.15], units="K", validate=True)
    try:
        yield
    finally:
        manager.set_value(_TEMPERATURE_DOMAIN_KEY, original, units="K", validate=True)


def _scalar(value):
    value = _unwrap_update(value)
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _independent_reference(model, net_radiation, air_temperature, pressure, phase):
    """Float64 evaluation of the documented expression with literature constants."""
    import math

    # References: Priestley and Taylor (1972); Murray (1967) ice coefficients; Allen et al.
    #   (1998), FAO-56, Eqs. 8, 11 and 13 for the liquid form.
    # Mathematical expression: e = e0 exp(a Tc / (Tc + b)); D = c e / (Tc + b)^2 with
    #   (a, b, c, L) = (17.27, 237.3, 4098, L_v) for liquid and
    #   (21.8745584, 265.5, a*b, L_s = L_v(0 degC) + L_f) for ice;
    #   gamma = c_p P / (eps L); E_raw = alpha D / (D + gamma) (Rn - G) / L; G = f_G Rn;
    #   E = max(E_raw, 0); LE = L E; projection = max(-E_raw, 0).
    # Parameter meanings: e saturation vapor pressure; D its slope; L latent heat of the
    #   phase; gamma psychrometric constant; alpha Priestley-Taylor coefficient; f_G
    #   soil-heat fraction; eps molar-mass ratio of water to dry air.
    # Units: e, P [Pa]; D, gamma [Pa K-1]; L [J kg-1]; Rn, G, LE [W m-2];
    #   E, projection [kg m-2 s-1]; alpha, f_G, eps [1]; Tc, b [degC].
    # Domain/constraints: 223.15 K <= T <= 323.15 K; literature constants written here are
    #   independent of the parameter database, whose values the Functor reads.
    # Sign convention: Rn positive into the surface; E and LE positive outward.
    celsius = air_temperature - 273.15
    if phase == "ice":
        a, b, latent = 21.8745584, 265.5, 2.501e6 + 3.34e5
        slope_coefficient = a * b
    else:
        a, b, latent, slope_coefficient = 17.27, 237.3, 2.45e6, 4098.0
    saturation = 610.78 * math.exp(a * celsius / (celsius + b))
    slope = slope_coefficient * saturation / (celsius + b) ** 2
    gamma = float(model.specific_heat_air) * pressure / (float(model.molecular_weight_ratio) * latent)
    available = net_radiation * (1.0 - _scalar(model.soil_heat_fraction))
    raw = _scalar(model.alpha_pt) * slope / (slope + gamma) * available / latent
    return {"evaporation": max(raw, 0.0), "latent_heat": max(raw, 0.0) * latent, "projection": max(-raw, 0.0),
            "slope": slope, "gamma": gamma, "fraction": slope / (slope + gamma), "latent": latent}


def _cold_state(temperature, net_radiation=150.0, pressure=96000.0):
    return {"net_radiation": torch.tensor(net_radiation), "air_temperature": torch.tensor(temperature),
            "atmospheric_pressure": torch.tensor(pressure)}


def test_default_sub_freezing_phase_is_liquid_and_default_domain_still_rejects():
    model = PriestleyTaylorET()
    assert model.sub_freezing_phase == "liquid"
    assert "hydrosphere.evapotranspiration.sub_freezing_phase" in model.required_parameters()
    assert "hydrosphere.evapotranspiration.sub_freezing_phase" in model.metadata.parameters
    assert not hasattr(model, "latent_heat_sublimation")
    with pytest.raises(ValueError, match="Temperature outside valid range"):
        model.forward(_cold_state(263.15), time=0.0, dt=1800.0)


def test_ice_option_alone_does_not_extend_the_domain_and_no_config_candidate_is_read():
    # The admitted domain stays the manager value; a domain written into the Functor
    # configuration is not a second source, exactly as before the option existed.
    model = PriestleyTaylorET(config={"parameters": {"sub_freezing_phase": "ice", "temperature_bounds": [223.15, 323.15]}})
    assert model.sub_freezing_phase == "ice"
    with pytest.raises(ValueError, match="Temperature outside valid range"):
        model.forward(_cold_state(263.15), time=0.0, dt=1800.0)


@pytest.mark.parametrize("config", [{"parameters": {"sub_freezing_phase": "ice"}}, {"sub_freezing_phase": "ice"}])
def test_sub_freezing_phase_is_read_from_both_configuration_forms(config):
    # Graph node configurations use the root-level key; direct construction also uses `parameters`.
    assert PriestleyTaylorET(config=config).sub_freezing_phase == "ice"


def test_unknown_sub_freezing_phase_is_rejected():
    with pytest.raises(ValueError, match="sub_freezing_phase"):
        PriestleyTaylorET(config={"parameters": {"sub_freezing_phase": "snow"}})


@pytest.mark.parametrize("temperature", [273.14, 272.15, 263.15, 243.15, 223.15])
def test_ice_phase_matches_independent_expression_below_freezing(sub_freezing_domain, temperature):
    model = PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    result = model.forward(_cold_state(temperature), time=0.0, dt=1800.0)
    reference = _independent_reference(model, 150.0, temperature, 96000.0, "ice")
    diagnostics = result["diagnostics.evapotranspiration"]
    assert _scalar(result["evapotranspiration_flux"]) == pytest.approx(reference["evaporation"], rel=5.0e-6)
    assert _scalar(result["latent_heat_flux"]) == pytest.approx(reference["latent_heat"], rel=5.0e-6)
    assert _scalar(diagnostics["slope_saturation_vapor_pressure"]) == pytest.approx(reference["slope"], rel=5.0e-6)
    assert _scalar(diagnostics["psychrometric_constant"]) == pytest.approx(reference["gamma"], rel=5.0e-6)
    assert _scalar(diagnostics["equilibrium_fraction"]) == pytest.approx(reference["fraction"], rel=5.0e-6)
    assert _scalar(diagnostics["equilibrium_evapotranspiration"]) == pytest.approx(
        reference["evaporation"] / _scalar(model.alpha_pt), rel=5.0e-6)
    assert _scalar(diagnostics["potential_evapotranspiration"]) == pytest.approx(reference["evaporation"], rel=5.0e-6)


def test_liquid_phase_inside_an_extended_domain_is_the_single_tetens_form(sub_freezing_domain):
    liquid, ice = PriestleyTaylorET(), PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    cold = _cold_state(263.15)
    liquid_result, ice_result = liquid.forward(cold, time=0.0, dt=1800.0), ice.forward(cold, time=0.0, dt=1800.0)
    reference = _independent_reference(liquid, 150.0, 263.15, 96000.0, "liquid")
    assert _scalar(liquid_result["evapotranspiration_flux"]) == pytest.approx(reference["evaporation"], rel=5.0e-6)
    assert _scalar(liquid_result["latent_heat_flux"]) == pytest.approx(reference["latent_heat"], rel=5.0e-6)
    # Measured at 263.15 K and 96 kPa: the ice branch carries 12.6 percent more latent heat
    # and 2.7 percent less water mass than the liquid form for the same available energy.
    assert _scalar(ice_result["latent_heat_flux"]) / _scalar(liquid_result["latent_heat_flux"]) == pytest.approx(1.1257, abs=2.0e-3)
    assert _scalar(ice_result["evapotranspiration_flux"]) / _scalar(liquid_result["evapotranspiration_flux"]) == pytest.approx(0.9728, abs=2.0e-3)


def test_ice_phase_leaves_results_at_and_above_freezing_unchanged(base_state):
    liquid = PriestleyTaylorET()
    ice = PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    for temperature in (273.15, 283.15, 298.15):
        state = {**base_state, "air_temperature": torch.tensor(temperature)}
        expected, actual = liquid.forward(state, time=0.0, dt=1800.0), ice.forward(state, time=0.0, dt=1800.0)
        for name in ("evapotranspiration_flux", "latent_heat_flux"):
            assert torch.equal(_unwrap_update(expected[name]), _unwrap_update(actual[name]))
        for name, value in expected["diagnostics.evapotranspiration"].items():
            assert torch.equal(value, actual["diagnostics.evapotranspiration"][name])


def test_ice_phase_selects_the_phase_cell_by_cell_including_the_freezing_point(sub_freezing_domain):
    liquid, ice = PriestleyTaylorET(), PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    temperatures = (253.15, 273.14, 273.15, 278.15)
    state = {"net_radiation": torch.tensor([150.0, 150.0, 150.0, 150.0]), "air_temperature": torch.tensor(temperatures),
             "atmospheric_pressure": torch.tensor(96000.0)}
    together = ice.forward(state, time=0.0, dt=1800.0)
    for index, temperature in enumerate(temperatures):
        phase = "ice" if temperature < 273.15 else "liquid"
        reference = _independent_reference(ice, 150.0, temperature, 96000.0, phase)
        assert _scalar(_unwrap_update(together["latent_heat_flux"]).reshape(-1)[index]) == pytest.approx(reference["latent_heat"], rel=5.0e-6)
        assert _scalar(_unwrap_update(together["evapotranspiration_flux"]).reshape(-1)[index]) == pytest.approx(reference["evaporation"], rel=5.0e-6)
    # Exactly at the freezing point the cell is liquid and equals the default Functor bit for bit.
    at_freezing = liquid.forward(_cold_state(273.15), time=0.0, dt=1800.0)
    assert _unwrap_update(together["latent_heat_flux"]).reshape(-1)[2] == _unwrap_update(at_freezing["latent_heat_flux"]).reshape(())
    # With a scalar pressure the psychrometric constant becomes one value per cell in ice mode.
    assert together["diagnostics.evapotranspiration"]["psychrometric_constant"].shape == (4,)


def test_ice_phase_gradient_is_finite_on_both_sides_of_the_freezing_point(sub_freezing_domain):
    model = PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    temperature = torch.tensor([263.15, 283.15], requires_grad=True)
    state = {"net_radiation": torch.tensor([150.0, 150.0]), "air_temperature": temperature,
             "atmospheric_pressure": torch.tensor([96000.0, 96000.0])}
    flux = _unwrap_update(model.forward(state, time=0.0, dt=1800.0)["latent_heat_flux"])
    (gradient,) = torch.autograd.grad(flux.sum(), temperature)
    assert torch.isfinite(gradient).all() and bool((gradient > 0).all())


def test_ice_phase_negative_available_energy_is_projected_like_the_liquid_form(sub_freezing_domain):
    model = PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    result = model.forward(_cold_state(263.15, net_radiation=-50.0), time=0.0, dt=1800.0)
    reference = _independent_reference(model, -50.0, 263.15, 96000.0, "ice")
    assert _scalar(result["evapotranspiration_flux"]) == 0.0 and _scalar(result["latent_heat_flux"]) == 0.0
    assert _scalar(result["diagnostics.evapotranspiration"]["negative_et_projection_residual"]) == pytest.approx(reference["projection"], rel=5.0e-6)


@pytest.mark.parametrize("temperature", [263.15, 272.15, 283.15])
def test_ice_phase_water_energy_port_pair_passes_the_local_hook_through_the_governed_call(sub_freezing_domain, temperature):
    # The governed call runs the per-Functor conservation hook, which forward() alone does not.
    model = PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    state = _cold_state(temperature)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Conservation law")
        result = model(state, time=0.0, dt=1800.0)
    assert model.check_conservation_laws(state, result)["energy"] <= 1.0e-6
    reference = _independent_reference(model, 150.0, temperature, 96000.0, "ice" if temperature < 273.15 else "liquid")
    assert _scalar(result["latent_heat_flux"]) / _scalar(result["evapotranspiration_flux"]) == pytest.approx(reference["latent"], rel=5.0e-6)


def test_ice_phase_constants_are_read_through_the_ssot_api_only_in_ice_mode(monkeypatch):
    calls = []
    original = priestley_taylor_module.get_physical_constant

    def recording_get_physical_constant(name: str):
        calls.append(name)
        return original(name)

    monkeypatch.setattr(priestley_taylor_module, "get_physical_constant", recording_get_physical_constant)
    ice_keys = [
        "relic.physical.constants.psychrometrics.tetens_ice_coefficient_a",
        "relic.physical.constants.psychrometrics.tetens_ice_coefficient_b",
        "relic.physical.constants.water.latent_heat_vaporization_intercept",
        "relic.physical.constants.water.latent_heat_fusion",
    ]
    PriestleyTaylorET()
    assert not set(ice_keys) & set(calls)
    calls.clear()
    model = PriestleyTaylorET(config=_ICE_PHASE_CONFIG)
    assert calls[-4:] == ice_keys
    # L_s = L_v(0 degC) + L_f from the two existing SSOT constants [J kg-1].
    assert float(model.latent_heat_sublimation) == pytest.approx(2.501e6 + 3.34e5)
    assert float(model.tetens_ice_coeff_a) == 21.8745584 and float(model.tetens_ice_coeff_b) == 265.5
