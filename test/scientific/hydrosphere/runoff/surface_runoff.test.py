"""
Unit Tests for Surface Runoff Model

Tests the SurfaceRunoff functor for basic functionality.
"""

from pathlib import Path

import pytest
import torch
from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.hydrosphere.runoff import SurfaceRunoff
from relic.utils.physical_constants import get_physical_constant


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


class TestSurfaceRunoffInitialization:
    """Test initialization."""
    
    def test_basic_initialization(self):
        """Test that SurfaceRunoff can be initialized."""
        functor = SurfaceRunoff()
        assert functor.name == "surface_runoff"


class TestSurfaceRunoffForward:
    """Test forward method."""
    
    def test_forward_basic(self, runoff_state, time_params):
        """Test forward method with valid inputs."""
        functor = SurfaceRunoff()
        result = functor.forward(
            runoff_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        assert set(result) == {
            "surface_runoff_flux",
            "infiltration_flux",
            "diagnostics.surface_runoff.runoff_coefficient",
            "diagnostics.surface_runoff.precipitation_rate_mm_h",
            "diagnostics.surface_runoff.runoff_rate_mm_h",
            "diagnostics.surface_runoff.scs_retention",
            "diagnostics.surface_runoff.initial_abstraction",
            "diagnostics.surface_runoff.runoff_nonnegative_residual",
            "diagnostics.surface_runoff.precipitation_limit_residual",
            "diagnostics.surface_runoff.water_balance_residual",
        }

    def test_flux_update_contract_and_open_boundary_balance(self):
        """Test explicit flux writes and the open precipitation partition."""
        functor = SurfaceRunoff(name="surface_runoff_contract")
        precipitation = torch.tensor(5.0e-3)
        result = functor.forward(
            {
                "precipitation_flux": precipitation,
                "soil_moisture": torch.tensor(0.4),
            },
            time=0.0,
            dt=3600.0,
        )

        for key in ("surface_runoff_flux", "infiltration_flux"):
            update = result[key]
            assert isinstance(update, UpdateItem)
            assert update.mode == UpdateMode.REPLACE
            assert update.per_timestep is False
            assert update.unit == "kg m-2 s-1"
            assert update.source_functor == "surface_runoff_contract"

        runoff = _unwrap_update(result["surface_runoff_flux"])
        infiltration = _unwrap_update(result["infiltration_flux"])
        residual = result["diagnostics.surface_runoff.water_balance_residual"]
        assert torch.equal(precipitation, runoff + infiltration)
        assert torch.equal(residual, precipitation - runoff - infiltration)
        assert functor.check_conservation_laws({}, result) == {"mass": 0.0}

        perturbed_result = dict(result)
        perturbed_result[
            "diagnostics.surface_runoff.water_balance_residual"
        ] = torch.tensor(-2.5e-4)
        assert functor.check_conservation_laws({}, perturbed_result) == {
            "mass": pytest.approx(2.5e-4)
        }

        missing_residual = dict(result)
        missing_residual.pop(
            "diagnostics.surface_runoff.water_balance_residual"
        )
        with pytest.raises(
            KeyError,
            match="Missing conservation diagnostic",
        ):
            functor.check_conservation_laws({}, missing_residual)

        nonfinite_residual = dict(result)
        nonfinite_residual[
            "diagnostics.surface_runoff.water_balance_residual"
        ] = torch.tensor(float("nan"))
        with pytest.raises(
            ValueError,
            match="Surface-runoff mass residual must be finite",
        ):
            functor.check_conservation_laws({}, nonfinite_residual)

    def test_profile_input_uses_surface_layer_and_emits_scalar_boundary_fluxes(self):
        """Test that a vertical profile drives one scalar column boundary flux."""
        functor = SurfaceRunoff(name="surface_runoff_profile")
        precipitation = torch.tensor(5.0e-3)
        profile_result = functor.forward(
            {
                "precipitation_flux": precipitation,
                "soil_moisture": torch.tensor([0.2, 0.8]),
            },
            time=0.0,
            dt=3600.0,
        )
        surface_result = functor.forward(
            {
                "precipitation_flux": precipitation,
                "soil_moisture": torch.tensor(0.2),
            },
            time=0.0,
            dt=3600.0,
        )

        for key in ("surface_runoff_flux", "infiltration_flux"):
            profile_flux = _unwrap_update(profile_result[key])
            surface_flux = _unwrap_update(surface_result[key])
            assert profile_flux.numel() == 1
            torch.testing.assert_close(profile_flux, surface_flux)

    def test_scs_diagnostics_report_the_effective_retention(self):
        """Test that SCS diagnostics are the values used by the runoff law."""
        functor = SurfaceRunoff()
        soil_moisture = torch.tensor(0.4)
        result = functor.forward(
            {
                "precipitation_flux": torch.tensor(5.0e-3),
                "soil_moisture": soil_moisture,
            },
            time=0.0,
            dt=3600.0,
        )

        def parameter_tensor(value):
            return torch.as_tensor(value).to(dtype=soil_moisture.dtype)

        base_retention = torch.clamp(
            parameter_tensor(functor.scs_numerator_constant)
            / parameter_tensor(functor.curve_number)
            - parameter_tensor(functor.scs_offset_constant),
            min=parameter_tensor(functor.scs_storage_minimum),
        )
        effective_retention = (
            base_retention
            * parameter_tensor(functor.antecedent_moisture_factor)
            * (
                1.0
                - parameter_tensor(functor.moisture_adjustment_factor)
                * soil_moisture
            )
        )
        effective_abstraction = (
            parameter_tensor(functor.initial_abstraction_ratio)
            * effective_retention
        )

        torch.testing.assert_close(
            result["diagnostics.surface_runoff.scs_retention"],
            effective_retention,
        )
        torch.testing.assert_close(
            result["diagnostics.surface_runoff.initial_abstraction"],
            effective_abstraction,
        )

    def test_parameter_and_diagnostic_ports_are_not_public_state_outputs(self):
        """Test that configuration and observations do not become state owners."""
        functor = SurfaceRunoff()
        result = functor.forward(
            {
                "precipitation_flux": torch.tensor(5.0e-3),
                "soil_moisture": torch.tensor(0.4),
            },
            time=0.0,
            dt=3600.0,
        )

        old_public_outputs = {
            "runoff_coefficient",
            "precipitation_rate_mm_h",
            "runoff_rate_mm_h",
            "scs_retention",
            "initial_abstraction",
            "diagnostics.surface_runoff",
        }
        assert old_public_outputs.isdisjoint(result)
        assert set(functor.metadata.output_variables) == set(result)

    def test_curve_number_is_parameter_manager_owned(self):
        """Test that state aliases cannot mutate the SCS configuration."""
        functor = SurfaceRunoff()
        base_state = {
            "precipitation_flux": torch.tensor(5.0e-3),
            "soil_moisture": torch.tensor(0.4),
        }
        baseline = _unwrap_update(
            functor.forward(base_state, time=0.0, dt=3600.0)["surface_runoff_flux"]
        )
        with_aliases = _unwrap_update(
            functor.forward(
                {
                    **base_state,
                    "curve_number": torch.tensor(100.0),
                    "curve_number_dynamic": torch.tensor(100.0),
                },
                time=0.0,
                dt=3600.0,
            )["surface_runoff_flux"]
        )

        assert torch.equal(baseline, with_aliases)
        assert "curve_number" not in functor.metadata.input_variables
        assert "curve_number_dynamic" not in functor.metadata.input_variables
        assert not hasattr(functor, "update_curve_number")
        assert not hasattr(functor, "_compute_rate_core")
        assert not hasattr(functor, "surface_flux_default_soil_moisture")
        assert not hasattr(functor, "surface_flux_default_dt")

    def test_dependencies_do_not_create_a_soil_state_feedback_edge(self):
        """Test that the partition reads committed state without a writer edge."""
        dependencies = SurfaceRunoff().get_dependencies()

        assert dependencies == {"required": [], "optional": []}

    def test_scalar_configuration_rejects_spatial_averaging_fallback(self):
        """Test that scalar parameter validation never hides a vector input."""
        with pytest.raises(ValueError, match="scalar"):
            SurfaceRunoff()._to_scalar(torch.tensor([1.0, 2.0]))

    @pytest.mark.parametrize(
        "state, message",
        [
            (
                {
                    "rain_flux": torch.tensor(1.0e-4),
                    "soil_moisture": torch.tensor(0.4),
                },
                "requires precipitation_flux",
            ),
            (
                {
                    "precipitation_flux": torch.tensor(float("nan")),
                    "soil_moisture": torch.tensor(0.4),
                },
                "finite",
            ),
            (
                {
                    "precipitation_flux": torch.tensor(-1.0e-4),
                    "soil_moisture": torch.tensor(0.4),
                },
                "non-negative",
            ),
        ],
    )
    def test_forcing_contract_fails_fast(self, state, message):
        """Test canonical precipitation forcing and finite-domain validation."""
        with pytest.raises(ValueError, match=message):
            SurfaceRunoff().forward(state, time=0.0, dt=3600.0)


RAIN_AND_MELT = {"surface_water_input_source": "rain_flux_plus_snow_melt_flux"}


class TestSurfaceWaterInputSource:
    """The declared source selects exactly one surface water input representation.

    CLM/ELM form the soil-surface input as qflx_top_soil = qflx_rain_grnd +
    qflx_snomelt below a snowpack and use total precipitation only without one;
    the source is a declaration (YAML SSOT, per-instance config), never inferred
    from which variables happen to be present in the state."""

    def test_default_source_is_the_declared_precipitation_flux(self):
        functor = SurfaceRunoff()
        assert functor.surface_water_input_source == "precipitation_flux"
        assert functor.get_dependencies() == {"required": [], "optional": []}
        assert "hydrosphere.runoff.surface_water_input_source" in functor.required_parameters()

    def test_rain_and_melt_source_depends_on_the_snowpack_owner(self):
        from relic.contracts import load_contract_registry

        registry = load_contract_registry()
        owner = registry.get_owner("rain_flux")
        assert owner == registry.get_owner("snow_melt_flux") == "snow_pack"
        functor = SurfaceRunoff(config=dict(RAIN_AND_MELT))
        assert functor.surface_water_input_source == "rain_flux_plus_snow_melt_flux"
        assert functor.get_dependencies()["required"] == [owner]

    def test_unknown_source_fails_fast(self):
        with pytest.raises(ValueError, match="surface_water_input_source"):
            SurfaceRunoff(config={"surface_water_input_source": "throughfall"})

    @pytest.mark.parametrize(
        "config, expected_input",
        [({}, 2.0e-4), (RAIN_AND_MELT, 5.0e-5 + 1.0e-4)],
    )
    def test_only_the_declared_representation_is_read(self, config, expected_input):
        """No double counting: with every representation present, the partition
        closes on the declared input only."""
        state = {
            "precipitation_flux": torch.tensor(2.0e-4),
            "rain_flux": torch.tensor(5.0e-5),
            "snow_melt_flux": torch.tensor(1.0e-4),
            "soil_moisture": torch.tensor(0.2),
        }
        result = SurfaceRunoff(config=dict(config)).forward(state, time=0.0, dt=3600.0)
        total = _unwrap_update(result["surface_runoff_flux"]) + _unwrap_update(
            result["infiltration_flux"]
        )
        assert torch.isclose(total, torch.tensor(expected_input), rtol=1e-6, atol=0.0)

    def test_missing_declared_representation_fails_fast(self):
        with pytest.raises(ValueError, match="snow_melt_flux"):
            SurfaceRunoff(config=dict(RAIN_AND_MELT)).forward(
                {"rain_flux": torch.tensor(1.0e-4), "soil_moisture": torch.tensor(0.2)},
                time=0.0,
                dt=3600.0,
            )
        with pytest.raises(ValueError, match="precipitation_flux"):
            SurfaceRunoff().forward(
                {
                    "rain_flux": torch.tensor(1.0e-4),
                    "snow_melt_flux": torch.tensor(0.0),
                    "soil_moisture": torch.tensor(0.2),
                },
                time=0.0,
                dt=3600.0,
            )

    def test_snowpack_and_surface_runoff_share_one_governed_state(self):
        """snow_pack reads precipitation_flux and surface_runoff reads its rain and
        melt in the same governed step (the declared dependency orders them even
        when surface_runoff is registered first); water is conserved:
        R + I = P_rain + M and P dt = dSWE + (R + I) dt without sublimation."""
        from relic.core.simulator import RELICSimulator
        from relic.data.standards.unit_converter import unit_converter
        from relic.scientific.hydrosphere.snow.snowpack import SnowPack

        dt = 3600.0
        precipitation = torch.tensor([3.0e-5, 3.0e-5])
        swe_before = torch.tensor([10.0, 10.0])
        simulator = RELICSimulator({
            "time_step": dt,
            "max_time": 10.0 * dt,
            "device": "cpu",
            "state_units": {"sublimation_flux": "kg m-2 s-1"},
        })
        simulator.initialize_state({
            "precipitation_flux": precipitation,
            "air_temperature": torch.tensor([268.15, 278.15]),
            "snow_water_equivalent": swe_before,
            "sublimation_flux": torch.tensor([0.0, 0.0]),
            "soil_moisture": torch.tensor([0.25, 0.25]),
        })
        simulator.register_functor(SurfaceRunoff(config=dict(RAIN_AND_MELT)), execution_group="fast")
        simulator.register_functor(SnowPack(), execution_group="fast")
        simulator.step(dt=dt)

        state = simulator.state_manager.get_state()
        rain = state["rain_flux"].reshape(-1)
        melt = state["snow_melt_flux"].reshape(-1)
        runoff = state["surface_runoff_flux"].reshape(-1)
        infiltration = state["infiltration_flux"].reshape(-1)
        assert (rain + melt)[1] > 0.0
        assert torch.allclose(runoff + infiltration, rain + melt, rtol=1e-6, atol=1e-12)
        swe_change = unit_converter.convert(
            state["snow_water_equivalent"].reshape(-1) - swe_before,
            "lwe_thickness_of_surface_snow_amount",
            "mm",
            "kg m-2",
        )
        assert torch.allclose(
            precipitation * dt,
            swe_change + (runoff + infiltration) * dt,
            rtol=1e-5,
            atol=1e-9,
        )


class TestSurfaceRunoffPhysicalConstraints:
    """Test physical constraints."""
    
    def test_runoff_non_negative(self, runoff_state, time_params):
        """Test that runoff is non-negative."""
        functor = SurfaceRunoff()
        result = functor.forward(runoff_state, time_params["time"], time_params["dt"])
        
        if "surface_runoff_flux" in result:
            runoff_flux = _unwrap_update(result["surface_runoff_flux"])
            assert torch.all(runoff_flux >= 0)
        elif "runoff_flux" in result:
            assert torch.all(result["runoff_flux"] >= 0)


class TestSurfaceRunoffPhase8:
    """Test Phase 8 IPHS contract."""

    def test_phase8_iphs_protocol(self):
        """Test structural IPHS hooks."""
        functor = SurfaceRunoff()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.runoff.curve_number" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "precipitation_flux",
            "soil_moisture",
            "surface_runoff_flux",
        }
        assert len(hodge_terms) == 1
        assert hodge_terms[0].domain.units == "m-1"
        assert hodge_terms[0].domain.semantic_name == "surface_routing_gradient"
        assert hodge_terms[0].codomain.units == "kg m-2 s-1"
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "surface_runoff_boundary_flux",
            "soil_infiltration_boundary",
        }
        infiltration_port = next(
            port for port in ports if port.flow_var == "infiltration_flux"
        )
        assert infiltration_port.positive_power_convention == "out_of_system"
        assert infiltration_port.port_kind == "matter"
        assert infiltration_port.flow_space.units == "kg m-2 s-1"
        assert infiltration_port.flow_space.complex_id == "surface_soil_water_interface"
        assert infiltration_port.flow_space.physical_domain == "surface_soil_interface"
        assert infiltration_port.power_units == "kg m-1 s-1"
        assert set(functor.metadata.conservation_laws) == {"mass"}
        assert "relic.physical.constants.water.density_liquid" not in functor.required_parameters()
        assert "relic.physical.constants.time.seconds_per_hour" not in functor.required_parameters()
        assert "relic.physical.constants.water.density_liquid" not in hodge_terms[0].parameter_keys
        assert functor.water_density == pytest.approx(get_physical_constant("water_density"))
        assert functor.time_conversion == pytest.approx(
            get_physical_constant("seconds_per_hour")
        )

    def test_phase8_certificate_is_valid(self):
        """Test Phase 8 certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_runoff_surface_surface_runoff.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == "relic.scientific.hydrosphere.runoff.surface.SurfaceRunoff"
        assert certificate.canonical_role == "transport"

    def test_residual_diagnostics_are_routed(self, runoff_state, time_params):
        """Test projection residual diagnostics are exposed."""
        functor = SurfaceRunoff()
        result = functor.forward(
            runoff_state,
            time=time_params["time"],
            dt=time_params["dt"],
        )

        assert torch.all(
            result["diagnostics.surface_runoff.runoff_nonnegative_residual"] >= 0.0
        )
        assert torch.all(
            result["diagnostics.surface_runoff.precipitation_limit_residual"] >= 0.0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

