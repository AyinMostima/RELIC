"""
Unit Tests for Penman-Monteith Evapotranspiration Model

Tests the PenmanMonteithET functor for:
- Initialization
- Forward method
- Parameter loading
- Physical constraints
- Units consistency
"""

import pytest
import torch
from pathlib import Path
from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.parameters import get_global_manager
from relic.scientific.hydrosphere.evapotranspiration import PenmanMonteithET
from relic.core.functor.functor_exceptions import ExecutionError
from relic.core.state import UpdateItem


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


class TestPenmanMonteithInitialization:
    """Test initialization of Penman-Monteith ET model."""
    
    def test_basic_initialization(self):
        """Test that PenmanMonteithET can be initialized with defaults."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        assert functor.name == "penman_monteith_et"
        assert functor is not None
    
    def test_custom_name(self):
        """Test initialization with custom name."""
        functor = PenmanMonteithET(name="custom_pm_et", config={"canopy_resistance_source": "state"})
        assert functor.name == "custom_pm_et"


class TestPenmanMonteithForward:
    """Test forward method computation."""
    
    def test_forward_basic(self, et_state, time_params):
        """Test forward method with valid inputs."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(
            et_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        # Check primary output exists
        assert "evapotranspiration_flux" in result
        
        # Check ET is a tensor
        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        assert isinstance(et_flux, torch.Tensor)
    
    def test_forward_output_aliases(self, et_state, time_params):
        """Test that outputs are namespaced and aliases are removed."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time_params["time"], time_params["dt"])

        assert "evapotranspiration_flux" in result
        assert "diagnostics.evapotranspiration" in result
        diagnostics = result["diagnostics.evapotranspiration"]
        assert isinstance(diagnostics, dict)

        removed_aliases = [
            "et",
            "evapotranspiration",
            "et_flux",
            "vapor_flux",
            "water_vapor_flux",
        ]
        for alias in removed_aliases:
            assert alias not in result, f"Unexpected alias in output: {alias}"
    
    def test_forward_diagnostics(self, et_state, time_params):
        """Test that diagnostic outputs are present."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time_params["time"], time_params["dt"])
        
        diagnostics = result["diagnostics.evapotranspiration"]
        expected_diagnostics = [
            "vapor_pressure_deficit",
            "aerodynamic_conductance",
            "canopy_conductance",
            "slope_saturation_vapor_pressure",
            "psychrometric_constant",
        ]

        for diagnostic in expected_diagnostics:
            assert diagnostic in diagnostics, f"Missing diagnostic: {diagnostic}"


class TestPenmanMonteithPhysicalConstraints:
    """Test physical constraints and validation."""
    
    def test_et_non_negative(self, et_state, time_params):
        """Test that ET flux is always non-negative."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time_params["time"], time_params["dt"])
        
        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        assert torch.all(et_flux >= 0), "ET flux must be non-negative"
    
    def test_et_reasonable_range(self, et_state, time_params):
        """Test that ET flux is within reasonable physical range."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time_params["time"], time_params["dt"])
        
        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        # Maximum physical ET: ~1 mm/s = 1 kg/m2/s
        assert torch.all(et_flux < 0.001), "ET flux exceeds physical maximum"
    
    def test_vpd_non_negative(self, et_state, time_params):
        """Test that vapor pressure deficit is non-negative."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time_params["time"], time_params["dt"])
        
        vpd = result["diagnostics.evapotranspiration"]["vapor_pressure_deficit"]
        assert torch.all(vpd >= 0), "VPD must be non-negative"
    
    def test_conductances_positive(self, et_state, time_params):
        """Test that conductances are positive."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time_params["time"], time_params["dt"])
        
        diagnostics = result["diagnostics.evapotranspiration"]
        assert torch.all(diagnostics["aerodynamic_conductance"] > 0)
        assert torch.all(diagnostics["canopy_conductance"] > 0)


class TestPenmanMonteithParameterLoading:
    """Test parameter loading from parameter manager."""
    
    def test_physical_constants_loaded(self):
        """Test that physical constants are loaded correctly."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        # Check that physical constants exist and are reasonable
        assert functor.latent_heat_vaporization > 2e6  # ~2.45e6 J/kg
        assert functor.latent_heat_vaporization < 3e6
        
        assert functor.specific_heat_air > 1000  # ~1013 J/kg/K
        assert functor.specific_heat_air < 1100
        
        # The air density is not a constant: forward() computes it from T, p and e_a
        # (TestPenmanMonteithMoistAirDensityAndSoilHeatFlux).
        assert not hasattr(functor, "air_density")
    
    def test_canopy_parameters_exist(self):
        """Test that canopy parameters are set."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        assert hasattr(functor, 'roughness_length')
        assert hasattr(functor, 'displacement_height')
        assert hasattr(functor, 'minimum_conductance')

    def test_aerodynamic_conductance_default_is_described_as_the_hodge_reference(self):
        """The key parameterizes only the static Hodge contract, never a runtime fallback,
        and is named as the reference state it is."""
        manager = get_global_manager()
        spec = manager.get_parameter("hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance")
        assert manager.get_parameter("hydrosphere.evapotranspiration.aerodynamic_conductance_default") is None
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        hodge = functor.build_hodge_contract(n_dofs=1)
        assert "hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance" in hodge.parameter_keys

        assert "build_hodge_contract" in spec.description
        assert "unavailable" not in spec.description
        assert not {"default_value", "error_handling"} & set(spec.tags)


class TestPenmanMonteithInputContract:
    """Each input is read once by its canonical name; absence raises or has a declared meaning."""

    _DELETED_KEYS = (
        "hydrosphere.evapotranspiration.net_radiation_diagnostic_default",
        "hydrosphere.evapotranspiration.wind_speed_diagnostic_default",
        "hydrosphere.evapotranspiration.surface_resistance_diagnostic_default",
        "hydrosphere.evapotranspiration.relative_humidity_diagnostic_default",
        "hydrosphere.evapotranspiration.air_temperature_diagnostic_default",
        "hydrosphere.evapotranspiration.vapor_pressure_deficit_diagnostic_default",
        "hydrosphere.evapotranspiration.boundary_layer_resistance_default",
        "hydrosphere.state_defaults.canopy_storage",
        "hydrosphere.state_defaults.leaf_area_index",
    )

    @staticmethod
    def _et(result):
        return _unwrap_update(result["evapotranspiration_flux"])

    def test_no_state_or_diagnostic_defaults_are_loaded(self):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        for attribute in (
            "_state_defaults",
            "_diagnostic_state_values",
            "_vpd_default",
            "_boundary_resistance_default",
            "_apply_state_defaults",
        ):
            assert not hasattr(functor, attribute)
        declared = set(functor.required_parameters())
        for key in self._DELETED_KEYS:
            assert key not in declared
            assert get_global_manager().get(key, required=False) is None

    def test_canopy_conductance_is_the_inverse_surface_resistance(self, et_state, time_params):
        """r_s = 1 / g_c: g_c = 0.02 m s-1 gives the r_s = 50 s m-1 result (not the old 100 default)."""
        base = {k: v for k, v in et_state.items() if k != "surface_resistance"}
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        from_conductance = functor.forward(
            {**base, "canopy_conductance": torch.tensor(0.02)}, time_params["time"], time_params["dt"]
        )
        from_resistance = functor.forward(
            {**base, "surface_resistance": torch.tensor(50.0)}, time_params["time"], time_params["dt"]
        )

        assert torch.allclose(self._et(from_conductance), self._et(from_resistance), rtol=1e-6)

    @pytest.mark.parametrize(
        "extra",
        [{}, {"stomatal_conductance": torch.tensor(0.02)}],
    )
    def test_missing_canopy_resistance_raises(self, et_state, time_params, extra):
        state = {k: v for k, v in et_state.items() if k != "surface_resistance"}
        state.update(extra)

        with pytest.raises(ValueError, match="surface_resistance"):
            PenmanMonteithET(config={"canopy_resistance_source": "state"}).forward(state, time_params["time"], time_params["dt"])

    def test_humidity_alias_is_not_read(self, et_state, time_params):
        state = {k: v for k, v in et_state.items() if k != "relative_humidity"}
        state["atmospheric_vapor_pressure_deficit"] = torch.tensor(700.0)

        with pytest.raises(ValueError, match="relative_humidity"):
            PenmanMonteithET(config={"canopy_resistance_source": "state"}).forward(state, time_params["time"], time_params["dt"])

    def test_without_interception_the_canopy_store_is_neither_read_nor_invented(
        self, et_state, time_params
    ):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        assert functor.enable_canopy_interception is False

        without_store = functor.forward(dict(et_state), time_params["time"], time_params["dt"])
        with_store = functor.forward(
            {**et_state, "canopy_storage": torch.tensor(0.4), "precipitation_flux": torch.tensor(1.0e-4)},
            time_params["time"],
            time_params["dt"],
        )

        assert "canopy_storage" not in without_store
        assert "throughfall_flux" not in without_store["diagnostics.evapotranspiration"]
        assert torch.equal(self._et(without_store), self._et(with_store))
        assert _unwrap_update(with_store["canopy_storage"]).item() == pytest.approx(0.4)
        assert with_store["diagnostics.evapotranspiration"]["throughfall_flux"].item() == pytest.approx(1.0e-4)

    @pytest.mark.parametrize("missing", ["precipitation_flux", "canopy_storage", "leaf_area_index"])
    def test_interception_requires_its_canopy_inputs(self, et_state, time_params, missing):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        functor.enable_canopy_interception = True
        state = {
            **et_state,
            "precipitation_flux": torch.tensor(1.0e-4),
            "canopy_storage": torch.tensor(0.2),
            "leaf_area_index": torch.tensor(3.0),
        }
        del state[missing]

        with pytest.raises(ValueError, match=missing):
            functor.forward(state, time_params["time"], time_params["dt"])

    def test_metadata_declares_the_inputs_actually_read(self):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        inputs = set(functor.metadata.input_variables)

        assert "stomatal_conductance" not in inputs
        for name in ("precipitation_flux", "canopy_storage", "leaf_area_index"):
            assert name in inputs
            assert name in functor.optional_input_variables
        assert {"relative_humidity", "vapor_pressure_deficit"} <= functor.optional_input_variables
        assert "atmospheric_pressure" not in functor.optional_input_variables


class TestPenmanMonteithMoistAirDensityAndSoilHeatFlux:
    """rho_a from T, p and e_a (not the 0 degC dry density); G with the FAO-56 hourly ratios."""

    @staticmethod
    def _state(functor, net_radiation, actual_vapor_pressure):
        saturation = functor._es_tetens_pa(torch.tensor(298.15))
        return {
            "net_radiation": torch.tensor(net_radiation),
            "air_temperature": torch.tensor(298.15),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "vapor_pressure_deficit": saturation - actual_vapor_pressure,
            "aerodynamic_resistance": torch.tensor(50.0),
            "surface_resistance": torch.tensor(100.0),
        }

    @staticmethod
    def _hand_density(temperature, pressure, vapor_pressure):
        from relic.utils.physical_constants import get_physical_constant

        gas_constant = float(get_physical_constant("gas_constant"))
        dry = float(get_physical_constant("air_molecular_weight")) * 1.0e-3
        water = float(get_physical_constant("water_molecular_weight")) * 1.0e-3
        return (pressure - (1.0 - water / dry) * vapor_pressure) / (gas_constant / dry * temperature)

    def test_aerodynamic_term_is_proportional_to_the_moist_air_density(self):
        """Rn = 0 isolates the aerodynamic term of ET and of the reference ET."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(self._state(functor, 0.0, 1500.0), time=0.0, dt=3600.0)
        diagnostics = _unwrap_update(result["diagnostics.evapotranspiration"])
        rho = self._hand_density(298.15, 101325.0, 1500.0)
        slope = float(diagnostics["slope_saturation_vapor_pressure"])
        gamma = float(diagnostics["psychrometric_constant"])
        ga = float(diagnostics["aerodynamic_conductance"])
        gc = float(diagnostics["canopy_conductance"])
        vpd = float(diagnostics["vapor_pressure_deficit"])
        g_ref = 1.0 / float(get_global_manager().get("hydrosphere.physical.reference_surface_resistance"))
        aerodynamic = rho * functor.specific_heat_air * vpd * ga
        lam = functor.latent_heat_vaporization

        et = float(_unwrap_update(result["evapotranspiration_flux"]))
        reference_et = float(diagnostics["reference_evapotranspiration"])

        assert et == pytest.approx(aerodynamic / (lam * (slope + gamma * (1.0 + ga / gc))), rel=1e-5)
        assert reference_et == pytest.approx(
            aerodynamic / (lam * (slope + gamma * (1.0 + ga / g_ref))), rel=1e-5
        )
        assert not hasattr(functor, "air_density")

    @pytest.mark.parametrize(
        "net_radiation, ratio_key",
        [
            (400.0, "hydrosphere.physical.soil_heat_flux_ratio_daytime"),
            (-80.0, "hydrosphere.physical.soil_heat_flux_ratio_nighttime"),
        ],
    )
    def test_soil_heat_flux_uses_the_fao56_daytime_and_nighttime_ratios(self, net_radiation, ratio_key):
        """FAO-56 Eqs. 45/46: G = 0.1 Rn when Rn > 0, G = 0.5 Rn otherwise (hourly steps).

        e_a = e_s (VPD = 0) removes the aerodynamic term; at night the negative raw ET
        is reported as negative_et_projection_residual.
        """
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        saturation = float(functor._es_tetens_pa(torch.tensor(298.15)))
        result = functor.forward(self._state(functor, net_radiation, saturation), time=0.0, dt=3600.0)
        diagnostics = _unwrap_update(result["diagnostics.evapotranspiration"])
        ratio = float(get_global_manager().get(ratio_key))
        slope = float(diagnostics["slope_saturation_vapor_pressure"])
        gamma = float(diagnostics["psychrometric_constant"])
        ga = float(diagnostics["aerodynamic_conductance"])
        gc = float(diagnostics["canopy_conductance"])
        raw_et = slope * (1.0 - ratio) * net_radiation / (
            functor.latent_heat_vaporization * (slope + gamma * (1.0 + ga / gc))
        )

        if net_radiation > 0.0:
            assert float(_unwrap_update(result["evapotranspiration_flux"])) == pytest.approx(raw_et, rel=1e-5)
        else:
            assert float(diagnostics["negative_et_projection_residual"]) == pytest.approx(-raw_et, rel=1e-5)
        assert ratio == pytest.approx(0.1 if net_radiation > 0.0 else 0.5)


class TestPenmanMonteithRequiredInputs:
    """Optional PM inputs (humidity, resistances, canopy store) are optional ports."""

    @pytest.mark.parametrize("source", ["state", "fao56_reference_surface"])
    def test_node_definition_requires_only_the_always_read_inputs(self, source):
        from relic.orchestration.node_registry import NodeRegistry
        from relic.orchestration.ssot_catalog import SSOTCatalog

        functor = PenmanMonteithET(config={"canopy_resistance_source": source})
        definition = NodeRegistry(SSOTCatalog()).register_functor_instance(functor)

        required = [name for name, port in definition.inputs.items() if port.required]
        assert required == ["net_radiation", "air_temperature", "wind_speed", "atmospheric_pressure"]
        assert required == functor.get_required_inputs()


class TestPenmanMonteithHourlySoilHeatFluxDomain:
    """The FAO-56 hourly G fractions (Eqs. 45-46) hold only for steps of one hour or less."""

    @staticmethod
    def _state():
        return {
            "net_radiation": torch.tensor(400.0),
            "air_temperature": torch.tensor(298.15),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "relative_humidity": torch.tensor(0.5),
            "aerodynamic_resistance": torch.tensor(50.0),
            "surface_resistance": torch.tensor(100.0),
        }

    def test_the_step_limit_is_a_declared_parameter_of_one_hour(self):
        spec = get_global_manager().get_parameter("hydrosphere.physical.soil_heat_flux_hourly_max_step")

        assert spec is not None and spec.units == "s"
        assert float(spec.value) == pytest.approx(3600.0)

    def test_a_governed_call_longer_than_one_hour_raises(self):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        # The governed call reports the forward() ValueError as its ExecutionError cause.
        with pytest.raises(ExecutionError, match=r"dt=7200.*3600.*FAO-56 hourly soil heat flux fractions") as excinfo:
            functor(self._state(), time=0.0, dt=7200.0)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.parametrize("dt", [3600.0, 1800.0])
    def test_governed_calls_of_one_hour_or_less_run(self, dt):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        result = functor(self._state(), time=0.0, dt=dt)

        assert float(_unwrap_update(result["evapotranspiration_flux"])) > 0.0


class TestPenmanMonteithCanopyResistanceSource:
    """The canopy resistance source is a declared configuration choice, never implicit."""

    @pytest.mark.parametrize("config", [None, {}, {"canopy_resistance_source": "default"}])
    def test_constructor_requires_a_declared_source(self, config):
        with pytest.raises(ValueError, match="canopy_resistance_source"):
            PenmanMonteithET(config=config)

    def test_state_source_reads_surface_resistance_or_canopy_conductance(self):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        assert {"surface_resistance", "canopy_conductance"} <= set(functor.metadata.input_variables)
        assert functor.canopy_resistance_source == "state"

    def test_fao56_reference_surface_uses_the_declared_70_s_m_parameter(self, et_state, time_params):
        """FAO-56 reference grass: r_s = hydrosphere.physical.reference_surface_resistance (70 s m-1)."""
        reference = PenmanMonteithET(config={"canopy_resistance_source": "fao56_reference_surface"})
        from_state = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        assert "surface_resistance" not in reference.metadata.input_variables
        assert "canopy_conductance" not in reference.metadata.input_variables
        assert get_global_manager().get("hydrosphere.physical.reference_surface_resistance") == pytest.approx(70.0)

        ignored_state = {**et_state, "surface_resistance": torch.tensor(500.0)}
        et_reference = _unwrap_update(
            reference.forward(ignored_state, time_params["time"], time_params["dt"])["evapotranspiration_flux"]
        )
        et_state_70 = _unwrap_update(
            from_state.forward(
                {**et_state, "surface_resistance": torch.tensor(70.0)}, time_params["time"], time_params["dt"]
            )["evapotranspiration_flux"]
        )
        assert torch.equal(et_reference, et_state_70)

    @pytest.mark.parametrize(
        "variable, udunits, kind",
        [("surface_resistance", "s m-1", "forcing"), ("canopy_conductance", "m s-1", "state")],
    )
    def test_state_source_inputs_have_scm_contracts(self, variable, udunits, kind):
        """Both state-source inputs are positive SCM variables; surface_resistance is
        externally supplied, canopy_conductance externally or by the one Medlyn provider
        configured with canopy_conductance_upscaling (B1, 2026-11-06)."""
        from relic.contracts.contract_loader import load_contract_registry

        contract = load_contract_registry().get_contract(variable)
        assert contract is not None
        assert contract.udunits == udunits
        assert contract.kind == kind
        expected_writers = {"surface_resistance": [], "canopy_conductance": ["medlyn_stomatal_conductance"]}
        assert contract.writers == expected_writers[variable]
        assert contract.raw["numerics"]["lower_bound"] == 0.0
        assert contract.raw["incrementable"] is False


class TestPenmanMonteithAtmosphericPressure:
    """The canonical atmospheric_pressure is read; no alias or default substitution."""

    @pytest.mark.parametrize(
        "substitute",
        [
            {"air_pressure": torch.tensor(90000.0)},
            {"surface_pressure": torch.tensor(90000.0)},
            {},
        ],
    )
    def test_missing_atmospheric_pressure_raises(self, et_state, time_params, substitute):
        state = {key: value for key, value in et_state.items() if key != "atmospheric_pressure"}
        state.update(substitute)

        with pytest.raises(ValueError, match="atmospheric_pressure"):
            PenmanMonteithET(config={"canopy_resistance_source": "state"}).forward(state, time_params["time"], time_params["dt"])

    def test_atmospheric_pressure_is_a_required_input(self):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        assert "atmospheric_pressure" not in functor.optional_input_variables
        assert "atmospheric_pressure" in functor.metadata.input_variables

    def test_an_alias_named_pressure_does_not_override_the_canonical_value(
        self, et_state, time_params
    ):
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        canonical = functor.forward(dict(et_state), time_params["time"], time_params["dt"])
        with_alias = functor.forward(
            {**et_state, "air_pressure": torch.tensor(50000.0)},
            time_params["time"],
            time_params["dt"],
        )

        assert torch.equal(
            _unwrap_update(canonical["evapotranspiration_flux"]),
            _unwrap_update(with_alias["evapotranspiration_flux"]),
        )


class TestPenmanMonteithComputeRate:
    """Test compute_rate interface."""
    
    def test_compute_rate_with_defaults(self, et_state):
        """Test compute_rate with default time parameters."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time=0.0, dt=1.0)
        
        assert "evapotranspiration_flux" in result
        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        assert isinstance(et_flux, torch.Tensor)
    
    def test_compute_rate_with_explicit_time(self, et_state):
        """Test compute_rate with explicit time and dt."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time=100.0, dt=1800.0)
        
        assert "evapotranspiration_flux" in result
    
    def test_forward_reads_its_drivers_from_the_state_only(self, basic_state):
        """The state is the one input channel: without the drivers the required
        inputs are missing, the same drivers in the state are read; no secondary
        _compute_rate_core entry exists."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        state = {"soil_moisture": torch.tensor(0.3)}
        drivers = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(293.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(100.0),
        }
        with pytest.raises(ValueError, match="Required state variable missing"):
            functor.forward(dict(state), time=0.0, dt=1.0)
        result = functor.forward({**state, **drivers}, time=0.0, dt=1.0)
        assert "evapotranspiration_flux" in result
        assert not hasattr(PenmanMonteithET, "_compute_rate_core")


class TestPenmanMonteithEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_required_variable(self):
        """Test that missing required variables raise ValueError."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        incomplete_state = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(293.15),
            # Missing other required variables
        }
        
        with pytest.raises(ValueError, match="Required state variable missing"):
            functor.forward(incomplete_state, time=0.0, dt=1.0)
    
    def test_extreme_temperature_low(self, et_state):
        """Test that extremely low temperature is caught."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        et_state["air_temperature"] = torch.tensor(200.0)  # -73 C, too low

        with pytest.raises(ValueError):
            functor.forward(et_state, time=0.0, dt=1.0)
    
    def test_extreme_temperature_high(self, et_state):
        """Test that extremely high temperature is caught."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        et_state["air_temperature"] = torch.tensor(350.0)  # 77 C, too high

        with pytest.raises(ValueError):
            functor.forward(et_state, time=0.0, dt=1.0)
    
    def test_invalid_humidity(self, et_state):
        """Test that invalid humidity is caught."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        et_state["relative_humidity"] = torch.tensor(1.5)  # > 1.0, invalid
        
        with pytest.raises(ValueError, match="Relative humidity"):
            functor.forward(et_state, time=0.0, dt=1.0)


class TestPenmanMonteithUnitsConsistency:
    """Test units consistency across calculations."""
    
    def test_output_units_consistency(self, et_state, time_params):
        """Test that output units are consistent."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(et_state, time_params["time"], time_params["dt"])
        
        # ET flux should be in kg m-2 s-1
        # For typical conditions, ET should be in range 0 to 0.001 kg/m2/s
        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        assert et_flux >= 0
        assert et_flux < 0.001
        
        # VPD should be in Pa (typically 0-5000 Pa)
        vpd = result["diagnostics.evapotranspiration"]["vapor_pressure_deficit"]
        assert vpd >= 0
        assert vpd < 10000
        
        # Conductances in m/s (typically 0.001-1.0 m/s)
        ga = result["diagnostics.evapotranspiration"]["aerodynamic_conductance"]
        assert ga > 0
        assert ga < 10.0


class TestPenmanMonteithPhase8:
    """Test Phase 8 IPHS contract."""

    def test_phase8_iphs_protocol(self):
        """Test structural IPHS hooks."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.evapotranspiration.roughness_length" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "evapotranspiration_flux",
            "latent_heat_flux",
            "vapor_pressure_deficit",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "penman_monteith_vapor_flux",
            "penman_monteith_latent_heat_flux",
        }

    def test_phase8_certificate_is_valid(self):
        """Test Phase 8 certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_evapotranspiration_penman_monteith_penman_monteith_et.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == (
            "relic.scientific.hydrosphere.evapotranspiration.penman_monteith.PenmanMonteithET"
        )
        assert certificate.canonical_role == "transport"

    def test_energy_and_storage_residual_diagnostics_are_routed(self, et_state, time_params):
        """Test residual diagnostics are exposed (the canopy store residual needs a store)."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = functor.forward(
            {**et_state, "canopy_storage": torch.tensor(0.0)}, time_params["time"], time_params["dt"]
        )

        diagnostics = result["diagnostics.evapotranspiration"]
        assert "available_energy_residual" in diagnostics
        assert "canopy_storage_residual" in diagnostics
        assert "negative_et_projection_residual" in diagnostics
        assert torch.isfinite(diagnostics["available_energy_residual"]).all()
        assert torch.all(diagnostics["canopy_storage_residual"] >= 0.0)
        assert torch.all(diagnostics["negative_et_projection_residual"] >= 0.0)


class TestPenmanMonteithDenominatorAndUnits:
    """The ET denominator is positive on the validated domain; no epsilon clamp hides a sign error.

    References: Monteith (1965); Allen et al. (1998), FAO-56.
    Mathematical expression: D = lambda (Delta + gamma (1 + g_a / g_c)) with lambda > 0,
      Delta > 0 (Tetens slope), gamma = c_p p / (epsilon lambda) > 0 for p > 0,
      g_a >= g_a,min > 0 and g_c >= g_c,min > 0, so D >= lambda (Delta + gamma) > 0.
    Units: D [J kg-1 Pa K-1]; a dimensionless epsilon cannot bound it.
    """

    def test_denominator_is_not_clamped_by_a_dimensionless_epsilon(
        self, et_state, time_params, monkeypatch
    ):
        import relic.scientific.hydrosphere.evapotranspiration.penman_monteith as module

        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        assert "validation.numerical.epsilon_divide" not in functor.required_parameters()

        def negative_gamma(*args, **kwargs):
            return torch.tensor(-1.0e6)

        monkeypatch.setattr(module, "compute_psychrometric_constant", negative_gamma)
        with pytest.raises(ValueError, match="denominator"):
            functor.forward(dict(et_state), time_params["time"], time_params["dt"])

    def test_interception_threshold_uses_the_liquid_water_equivalent_rule(self, caplog):
        """2e-5 mm in one hour lies below 1e-8 kg m-2 s-1 * 3600 s = 3.6e-5 mm (not 1e-5)."""
        import logging

        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        caplog.set_level(logging.DEBUG)
        functor._log_canopy_diagnostics(
            precipitation_mm=2.0e-5,
            interception_mm=1.0e-5,
            evap_from_interception_mm=0.0,
            throughfall_mm=1.0e-5,
            total_et_mm=0.0,
            transpiration_mm=0.0,
            budget_residual=0.0,
            et_balance=0.0,
            dt=3600.0,
            canopy_storage=0.0,
        )

        assert "Interception efficiency: 0.0%" in caplog.text

    def test_hodge_mobility_has_no_literal_conductance_floor(self, monkeypatch):
        """g_eff = g_a g_c / (g_a + g_c) needs g_a > 0 and g_c > 0 [m s-1]; a zero
        conductance raises instead of being lifted to a dimensionless 1e-12."""
        import inspect

        source = inspect.getsource(PenmanMonteithET.build_hodge_contract)
        assert "1.0e-12" not in source
        assert "max(float(self.latent_heat_vaporization), 1.0)" not in source

        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        assert functor.build_hodge_contract(n_dofs=1).matrix.item() > 0.0
        monkeypatch.setattr(functor, "minimum_conductance", 0.0)
        with pytest.raises(ValueError, match="positive"):
            functor.build_hodge_contract(n_dofs=1)



class TestPenmanMonteithCanopyStorageOutput:
    """canopy_storage leaves PM only when a canopy store exists (D-T1-4)."""

    def test_without_interception_the_governed_call_warns_about_no_missing_output(
        self, et_state, time_params, caplog
    ):
        import logging

        state = {
            key: value
            for key, value in et_state.items()
            if key not in {"canopy_storage", "precipitation_flux", "leaf_area_index"}
        }
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        assert functor.enable_canopy_interception is False
        assert "canopy_storage" in functor.metadata.optional_output_variables

        caplog.set_level(logging.WARNING)
        updates = functor(state, time_params["time"], time_params["dt"])

        assert "canopy_storage" not in updates
        assert "Output variable 'canopy_storage' not found" not in caplog.text

    def test_with_interception_canopy_storage_is_a_required_output(self):
        manager = get_global_manager()
        snapshot = manager.snapshot()
        try:
            manager.set_value(
                "hydrosphere.evapotranspiration.enable_canopy_interception",
                True,
                units="dimensionless",
                validate=True,
                notify=False,
            )
            functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        finally:
            manager.restore(snapshot)

        assert functor.enable_canopy_interception is True
        assert "canopy_storage" in functor.metadata.output_variables
        assert "canopy_storage" not in functor.metadata.optional_output_variables


class TestPenmanMonteithReadKeysEnterTheComputation:
    """Round 35: every key PM reads enters a computation, a branch or a check."""

    _DEAD_KEYS = (
        "hydrosphere.transport.water_vapor_diffusivity",
        "hydrosphere.evapotranspiration.canopy_height",
        "hydrosphere.evapotranspiration.drip_coefficient",
    )

    def test_dead_keys_are_not_read(self):
        """canopy_height (d and z0m are their own keys), the Rutter drainage coefficient
        (the executed store drains by capacity overflow) and the water-vapour diffusivity
        (gas exchange has no scalar diffusivity law) enter no calculation."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        for key in self._DEAD_KEYS:
            assert key not in functor.required_parameters()
            assert key not in functor.metadata.parameters
        assert not hasattr(functor, "canopy_height")
        assert not hasattr(functor, "drip_coefficient")
        assert functor.transport_type == "gas_exchange"
        assert functor.diffusivity is None

    def test_interception_is_documented_as_the_executed_overflow_store(self):
        """The executed canopy store fills to C_max = c_LAI LAI and drains only the
        excess above C_max; no text claims an exponential Rutter drainage law."""
        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        switch_text = functor.metadata.parameters["hydrosphere.evapotranspiration.enable_canopy_interception"]
        assert "Rutter" not in switch_text
        assert "overflow" in switch_text
        doc = PenmanMonteithET._process_canopy_interception.__doc__
        assert "overflow" in doc

    def test_aerodynamic_resistance_is_a_documented_optional_input(self):
        """Without aerodynamic_resistance, g_a is the FAO-56 Eq. 4 neutral log-profile
        value from wind_speed: a documented optional branch, not a fallback."""
        import inspect

        functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        assert "aerodynamic_resistance" in functor.optional_input_variables
        source = inspect.getsource(PenmanMonteithET.forward).lower()
        assert "fall back" not in source
        assert "fao-56 eq. 4" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


