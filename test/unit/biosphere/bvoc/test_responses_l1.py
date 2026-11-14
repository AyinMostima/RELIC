"""
L1 Unit Tests for BVOC Response Functions Module

Comprehensive testing of BVOC-specific response functions including light responses,
temperature responses, environmental stress factors, and integrated activity factors.

Test Coverage:
- BVOCLightResponse: MEGAN2.1 light activity composed from the shared megan_model law
- BVOCTemperatureResponse: MEGAN2.1 Eq. (8) and Eq. (11) temperature activity
- BVOCEnvironmentalResponse: soil-moisture, water-potential and combined stress factors
- BVOCResponseIntegrator: Integration of all response factors
- Parameter validation and boundary conditions
- Retired helpers without literature support or production callers

Scientific Validation:
- Light activity: Guenther et al. (2012) Eq. (4) for isoprene (LDF = 1) and
  Eq. (3) gamma_L = (1 - LDF) + LDF gamma_P_LDF for sesquiterpenes
- Temperature activity: Eq. (8) for isoprene and Eq. (11)
  gamma_T = exp(beta (T - T_s)) for light-independent classes
- Stress: minimum of explicit factors in [0, 1]
- Total activity: gamma_total = gamma_L gamma_T gamma_stress

References:
- Guenther, A. B., et al. (2012): MEGAN2.1, Eqs. (3), (4), (8), (11) and Table 4
- Guenther, A. B., et al. (2006): MEGAN soil-moisture activity factor
- Brilli, F., et al. (2007): Drought response of isoprene emission
"""

import inspect
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
import warnings

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from relic.scientific.biosphere.bvoc.responses import (
    BVOCLightResponse,
    BVOCTemperatureResponse,
    BVOCEnvironmentalResponse,
    BVOCResponseIntegrator
)


def test_import_preserves_native_torch_exp_and_forward_differentiation():
    script = """
import torch
original_exp = torch.exp
import relic.scientific.biosphere.bvoc.responses
assert torch.exp is original_exp, 'BVOC import must not replace torch.exp'
value, tangent = torch.func.jvp(
    torch.exp, (torch.tensor(0.0),), (torch.tensor(1.0),),
)
assert value.item() == 1.0 and tangent.item() == 1.0
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("value", [None, torch.tensor([]), torch.tensor([1.0, 2.0])])
def test_required_scalar_parameter_rejects_missing_or_multiple_values(value):
    from relic.scientific.biosphere.bvoc.responses import _require_scalar

    manager = SimpleNamespace(get=lambda *args, **kwargs: value)
    with pytest.raises(ValueError, match="scalar parameter"):
        _require_scalar(manager, "beta", "test", "K-1")


def test_scalar_parameter_preserves_single_value_and_optional_absence():
    from relic.scientific.biosphere.bvoc.responses import _require_scalar

    scalar = SimpleNamespace(get=lambda *args, **kwargs: torch.tensor([1.25], dtype=torch.float64))
    absent = SimpleNamespace(get=lambda *args, **kwargs: None)
    assert _require_scalar(scalar, "beta", "test", "K-1") == 1.25
    assert _require_scalar(absent, "optional", "test", "1", required=False) is None


def _manager_scalar(key, units):
    """Read one registered scalar from the global ParameterManager."""
    from relic.parameters import get_global_manager

    value = get_global_manager().get(key, units=units, required=True)
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _light_parameters():
    return (
        _manager_scalar("biosphere.bvoc.megan.alpha_light", "m2 s umol-1"),
        _manager_scalar("biosphere.bvoc.megan.cl_light", "dimensionless"),
    )


class TestBVOCLightResponseL1:
    """MEGAN2.1 light activity composed from the shared megan_model law."""

    def test_isoprene_light_activity_is_the_shared_megan_law(self):
        """Isoprene is fully light dependent (Guenther et al. 2012, Table 4), so
        Eq. (3) reduces to the Eq. (4) light-dependent activity owned by
        megan_model; no clamp is applied."""
        from relic.scientific.biosphere.bvoc.megan_model import (
            MEGANModel,
            megan_light_dependent_activity,
        )

        ppfd = torch.tensor([0.0, 200.0, 500.0, 1000.0, 2000.0], dtype=torch.float32)
        alpha, cl = _light_parameters()
        gamma = BVOCLightResponse.get_compound_light_response("isoprene", ppfd)
        manual = torch.tensor(
            [alpha * cl * q / math.sqrt(1.0 + (alpha * q) ** 2) for q in ppfd.tolist()],
            dtype=gamma.dtype,
        )
        assert torch.allclose(gamma, manual, rtol=1e-5, atol=1e-9)
        shared = megan_light_dependent_activity(
            ppfd,
            torch.tensor(alpha, dtype=ppfd.dtype),
            torch.tensor(cl, dtype=ppfd.dtype),
        )
        assert torch.allclose(gamma, shared, rtol=1e-6, atol=1e-9)
        megan = MEGANModel()._get_light_factor("isoprene", ppfd)
        assert torch.allclose(gamma, megan.to(gamma.dtype), rtol=1e-6, atol=1e-9)
        assert gamma[0].item() == 0.0
        assert torch.all(gamma[1:] > gamma[:-1])

    def test_sesquiterpene_light_activity_follows_megan21_eq3(self):
        """gamma_L = (1 - LDF) + LDF gamma_P_LDF with the registered SQT LDF."""
        ppfd = torch.tensor([0.0, 200.0, 500.0, 1000.0, 2000.0], dtype=torch.float32)
        alpha, cl = _light_parameters()
        ldf = _manager_scalar("biosphere.bvoc.megan.ldf_sesquiterpenes", "dimensionless")
        assert ldf == pytest.approx(0.5)
        gamma = BVOCLightResponse.get_compound_light_response("sesquiterpenes", ppfd)
        manual = torch.tensor(
            [
                (1.0 - ldf) + ldf * alpha * cl * q / math.sqrt(1.0 + (alpha * q) ** 2)
                for q in ppfd.tolist()
            ],
            dtype=gamma.dtype,
        )
        assert torch.allclose(gamma, manual, rtol=1e-5, atol=1e-9)
        # The light-independent share alone remains in darkness; the retired
        # weak-light curve was confined to [0.8, 1.2].
        assert gamma[0].item() == pytest.approx(1.0 - ldf)
        assert gamma[0].item() < 0.8
        assert torch.all(gamma[1:] > gamma[:-1])

    def test_monoterpene_light_activity_follows_megan21_eq3(self):
        """The lumped monoterpene class uses Eq. (3) with the representative
        'Other Monoterpenes' LDF of Guenther et al. (2012) Table 4."""
        ppfd = torch.tensor([0.0, 200.0, 500.0, 1000.0, 2000.0], dtype=torch.float32)
        alpha, cl = _light_parameters()
        ldf = _manager_scalar("biosphere.bvoc.megan.ldf_monoterpenes", "dimensionless")
        assert ldf == pytest.approx(0.4)
        gamma = BVOCLightResponse.get_compound_light_response("monoterpenes", ppfd)
        manual = torch.tensor(
            [
                (1.0 - ldf) + ldf * alpha * cl * q / math.sqrt(1.0 + (alpha * q) ** 2)
                for q in ppfd.tolist()
            ],
            dtype=gamma.dtype,
        )
        assert torch.allclose(gamma, manual, rtol=1e-5, atol=1e-9)
        assert gamma[0].item() == pytest.approx(1.0 - ldf)

    def test_ovoc_light_activity_is_the_emission_weighted_component_mixture(self):
        """The lumped OVOC light activity is sum_j w_j [(1 - LDF_j) + LDF_j gamma_P_LDF]
        over methanol, acetone and acetaldehyde with their Table 4 LDF and the Table 6
        emission shares, equal to MEGANModel."""
        from relic.scientific.biosphere.bvoc.megan_model import MEGANModel

        ppfd = torch.tensor([0.0, 500.0, 1500.0], dtype=torch.float32)
        alpha, cl = _light_parameters()
        names = ("methanol", "acetone", "acetaldehyde")
        emissions = {
            name: _manager_scalar(f"biosphere.bvoc.megan.ovoc.{name}.global_emission", "g yr-1")
            for name in names
        }
        total = sum(emissions.values())
        manual = []
        for q in ppfd.tolist():
            light_dependent = alpha * cl * q / math.sqrt(1.0 + (alpha * q) ** 2)
            manual.append(
                sum(
                    emissions[name]
                    / total
                    * (
                        1.0
                        - _manager_scalar(f"biosphere.bvoc.megan.ovoc.{name}.ldf", "dimensionless")
                        * (1.0 - light_dependent)
                    )
                    for name in names
                )
            )
        gamma = BVOCLightResponse.get_compound_light_response("ovocs", ppfd)
        assert torch.allclose(gamma, torch.tensor(manual, dtype=gamma.dtype), rtol=1e-5, atol=1e-9)
        megan = MEGANModel()._get_light_factor("ovocs", ppfd)
        assert torch.allclose(gamma, megan.to(gamma.dtype), rtol=1e-6)

    @pytest.mark.parametrize("compound", ["unknown"])
    def test_light_activity_requires_a_registered_light_dependent_fraction(self, compound):
        with pytest.raises(ValueError, match="light-dependent fraction"):
            BVOCLightResponse.get_compound_light_response(compound, torch.tensor([500.0]))

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_light_activity_rejects_invalid_ppfd(self, value):
        with pytest.raises(ValueError, match="ppfd"):
            BVOCLightResponse.get_compound_light_response(
                "sesquiterpenes", torch.tensor([value])
            )

    def test_calculate_factor_uses_the_same_dispatch(self):
        ppfd = torch.tensor([300.0, 1500.0], dtype=torch.float32)
        helper = BVOCLightResponse()
        assert torch.allclose(
            helper.calculate_factor(ppfd, compound="sesquiterpenes"),
            BVOCLightResponse.get_compound_light_response("sesquiterpenes", ppfd),
        )

    def test_retired_light_helpers_are_absent(self):
        assert not hasattr(BVOCLightResponse, "isoprene_light_response")
        assert not hasattr(BVOCLightResponse, "monoterpene_light_response")


class TestBVOCTemperatureResponseL1:
    """MEGAN2.1 temperature activity composed from the shared megan_model laws."""

    def test_isoprene_temperature_activity_is_the_shared_megan_eq8(self):
        """Isoprene uses Guenther et al. (2012) Eq. (8) at the registered fixed
        standard history; the retired geometric-mean law and [0.01, 5] clamp
        are gone."""
        from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
        from relic.utils.physical_constants import get_physical_constant

        temperatures = torch.tensor([283.15, 298.15, 313.15, 328.15], dtype=torch.float32)
        gamma = BVOCTemperatureResponse.get_compound_temperature_response(
            "isoprene", temperatures
        )
        eopt = _manager_scalar("biosphere.bvoc.megan.eopt_isoprene", "dimensionless")
        topt = _manager_scalar("biosphere.bvoc.megan.topt", "K")
        ct1 = _manager_scalar("biosphere.bvoc.megan.ct1_isoprene", "J mol-1")
        ct2 = _manager_scalar("biosphere.bvoc.megan.ct2", "J mol-1")
        r_gas = float(get_physical_constant("gas_constant"))
        manual = []
        for temperature in temperatures.tolist():
            x = (1.0 / topt - 1.0 / temperature) / r_gas
            manual.append(
                eopt * ct2 * math.exp(ct1 * x) / (ct2 - ct1 * (1.0 - math.exp(ct2 * x)))
            )
        assert torch.allclose(gamma, torch.tensor(manual, dtype=gamma.dtype), rtol=1e-4)
        megan = MEGANModel()._get_temperature_factor("isoprene", temperatures)
        assert torch.allclose(gamma, megan.to(gamma.dtype), rtol=1e-6)
        # Eq. (8) returns E_opt exactly at T = T_opt (G12 Eq. 9: 313 K at the standard
        # history); 313.15 K lies 0.15 K above the optimum.
        at_topt = BVOCTemperatureResponse.get_compound_temperature_response(
            "isoprene", torch.tensor([topt], dtype=torch.float32)
        )
        assert at_topt.item() == pytest.approx(eopt, rel=1e-5)
        assert gamma[0] < gamma[1] < gamma[2]
        assert gamma[2] > gamma[3]

    @pytest.mark.parametrize("compound", ["monoterpenes", "sesquiterpenes"])
    def test_partly_light_dependent_temperature_activity_is_the_shared_megan_eq7(
        self, compound
    ):
        """gamma_T = (1 - LDF) exp(beta (T - T_s)) + LDF gamma_T,LDF (Guenther et al.
        2012, Eq. 7) with the class's Table 4 row, equal to MEGANModel."""
        from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
        from relic.utils.physical_constants import get_physical_constant

        temperatures = torch.tensor([273.15, 293.15, 303.15, 313.15], dtype=torch.float32)
        ldf = _manager_scalar(f"biosphere.bvoc.megan.ldf_{compound}", "dimensionless")
        beta = _manager_scalar(f"biosphere.bvoc.megan.beta_{compound}", "K-1")
        eopt = _manager_scalar(f"biosphere.bvoc.megan.eopt_{compound}", "dimensionless")
        ct1 = _manager_scalar(f"biosphere.bvoc.megan.ct1_{compound}", "J mol-1")
        topt = _manager_scalar("biosphere.bvoc.megan.topt", "K")
        ct2 = _manager_scalar("biosphere.bvoc.megan.ct2", "J mol-1")
        standard_temperature = _manager_scalar("biosphere.bvoc.t_standard", "K")
        r_gas = float(get_physical_constant("gas_constant"))
        manual = []
        for temperature in temperatures.tolist():
            x = (1.0 / topt - 1.0 / temperature) / r_gas
            light_dependent = (
                eopt * ct2 * math.exp(ct1 * x) / (ct2 - ct1 * (1.0 - math.exp(ct2 * x)))
            )
            manual.append(
                (1.0 - ldf) * math.exp(beta * (temperature - standard_temperature))
                + ldf * light_dependent
            )
        gamma = BVOCTemperatureResponse.get_compound_temperature_response(
            compound, temperatures
        )
        assert torch.allclose(gamma, torch.tensor(manual, dtype=gamma.dtype), rtol=1e-5)
        megan = MEGANModel()._get_temperature_factor(compound, temperatures)
        assert torch.allclose(gamma, megan.to(gamma.dtype), rtol=1e-6)

    def test_ovoc_temperature_activity_is_the_emission_weighted_component_mixture(self):
        """gamma_T,OVOC = sum_j w_j gamma_T,j with each component's Eq. (7) activity from
        its Table 4 row and w_j the Table 6 emission shares, equal to MEGANModel."""
        from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
        from relic.utils.physical_constants import get_physical_constant

        temperatures = torch.tensor([273.15, 293.15, 303.15, 313.15], dtype=torch.float32)
        topt = _manager_scalar("biosphere.bvoc.megan.topt", "K")
        ct2 = _manager_scalar("biosphere.bvoc.megan.ct2", "J mol-1")
        standard_temperature = _manager_scalar("biosphere.bvoc.t_standard", "K")
        r_gas = float(get_physical_constant("gas_constant"))
        names = ("methanol", "acetone", "acetaldehyde")

        def component(name, field, units):
            return _manager_scalar(f"biosphere.bvoc.megan.ovoc.{name}.{field}", units)

        total = sum(component(name, "global_emission", "g yr-1") for name in names)
        manual = []
        for temperature in temperatures.tolist():
            x = (1.0 / topt - 1.0 / temperature) / r_gas
            value = 0.0
            for name in names:
                ldf = component(name, "ldf", "dimensionless")
                ct1 = component(name, "ct1", "J mol-1")
                light_dependent = (
                    component(name, "eopt", "dimensionless")
                    * ct2
                    * math.exp(ct1 * x)
                    / (ct2 - ct1 * (1.0 - math.exp(ct2 * x)))
                )
                light_independent = math.exp(
                    component(name, "beta", "K-1") * (temperature - standard_temperature)
                )
                value += component(name, "global_emission", "g yr-1") / total * (
                    (1.0 - ldf) * light_independent + ldf * light_dependent
                )
            manual.append(value)
        gamma = BVOCTemperatureResponse.get_compound_temperature_response("ovocs", temperatures)
        assert torch.allclose(gamma, torch.tensor(manual, dtype=gamma.dtype), rtol=1e-5)
        megan = MEGANModel()._get_temperature_factor("ovocs", temperatures)
        assert torch.allclose(gamma, megan.to(gamma.dtype), rtol=1e-6)

    def test_temperature_activity_rejects_unregistered_classes_and_invalid_temperature(self):
        with pytest.raises(ValueError, match="compound class"):
            BVOCTemperatureResponse.get_compound_temperature_response(
                "unknown", torch.tensor([300.0])
            )
        with pytest.raises(ValueError, match="temperature"):
            BVOCTemperatureResponse.get_compound_temperature_response(
                "sesquiterpenes", torch.tensor([0.0])
            )
        with pytest.raises(ValueError, match="temperature"):
            BVOCTemperatureResponse.get_compound_temperature_response(
                "sesquiterpenes", torch.tensor([float("nan")])
            )

    def test_parameter_overrides_bypassing_the_manager_are_not_accepted(self):
        with pytest.raises(TypeError):
            BVOCTemperatureResponse.get_compound_temperature_response(
                "monoterpenes", torch.tensor([300.0]), beta=0.12
            )

    def test_retired_temperature_helpers_are_absent(self):
        assert not hasattr(BVOCTemperatureResponse, "isoprene_temperature_response")
        assert not hasattr(BVOCTemperatureResponse, "exponential_temperature_response")


class TestBVOCEnvironmentalResponseL1:
    """Water and combined stress factors without unsupported floors or clamps."""

    def test_soil_moisture_factor_is_the_relative_moisture_without_floor(self):
        soil_moisture = torch.tensor([0.0, 0.05, 0.5, 1.0])
        gamma = BVOCEnvironmentalResponse.drought_stress_response(
            soil_moisture=soil_moisture
        )
        assert torch.equal(gamma, soil_moisture)

    @pytest.mark.parametrize("value", [-0.1, 1.2, float("nan")])
    def test_soil_moisture_outside_unit_interval_is_rejected(self, value):
        with pytest.raises(ValueError, match="soil_moisture"):
            BVOCEnvironmentalResponse.drought_stress_response(
                soil_moisture=torch.tensor([value])
            )

    def test_water_potential_factor_is_the_linear_ramp(self):
        water_potential = torch.tensor([-2.0, -1.5, -0.8, -0.1, 0.0])
        gamma = BVOCEnvironmentalResponse.drought_stress_response(
            water_potential=water_potential,
            wilting_point=-1.5,
            field_capacity=-0.1,
        )
        expected = torch.clamp((water_potential + 1.5) / 1.4, 0.0, 1.0)
        assert torch.allclose(gamma, expected)

    def test_drought_factor_requires_an_input(self):
        with pytest.raises(ValueError, match="soil_moisture or water_potential"):
            BVOCEnvironmentalResponse.drought_stress_response()

    def test_combined_stress_is_the_minimum_of_explicit_factors(self):
        state = {
            'water_stress_factor': torch.tensor(0.8),
            'temperature_stress_factor': torch.tensor(0.9),
            'nutrient_stress_factor': torch.tensor(0.7),
        }
        gamma = BVOCEnvironmentalResponse.comprehensive_stress_response(state)
        assert gamma.item() == pytest.approx(0.7)
        severe = BVOCEnvironmentalResponse.comprehensive_stress_response(
            {'water_stress_factor': torch.tensor(0.02)}
        )
        assert severe.item() == pytest.approx(0.02)

    def test_temperature_alone_adds_no_stress_factor(self):
        """Temperature enters only through gamma_T; the retired parabolic
        temperature-stress default is not applied a second time."""
        gamma = BVOCEnvironmentalResponse.comprehensive_stress_response(
            {'soil_moisture': torch.tensor(0.6), 'temperature': torch.tensor(313.15)},
        )
        assert gamma.item() == pytest.approx(0.6)
        only_temperature = BVOCEnvironmentalResponse.comprehensive_stress_response(
            {'temperature': torch.tensor(313.15)}
        )
        assert only_temperature.item() == 1.0

    @pytest.mark.parametrize("key", ["water_stress_factor", "temperature_stress_factor", "nutrient_stress_factor"])
    @pytest.mark.parametrize("value", [-0.1, 1.5, float("nan")])
    def test_stress_factors_outside_unit_interval_are_rejected(self, key, value):
        with pytest.raises(ValueError, match=key):
            BVOCEnvironmentalResponse.comprehensive_stress_response(
                {key: torch.tensor(value)}
            )

    def test_retired_environmental_helpers_are_absent(self):
        """The auxiliary CO2 law (production CO2 inhibition is
        MEGANModel._compute_co2_inhibition_factor) and the leaf-age power law had
        no production caller and no literature support; both are retired. The
        uncalled second aggregator calculate_all_factors, which substituted a
        neutral factor for a missing temperature or PAR driver, is retired too;
        BVOCResponseIntegrator.calculate_all_activity_factors is the one aggregator."""
        assert not hasattr(BVOCEnvironmentalResponse, "co2_inhibition_response")
        assert not hasattr(BVOCEnvironmentalResponse, "leaf_age_response")
        assert not hasattr(BVOCEnvironmentalResponse, "calculate_all_factors")

    def test_stress_factors_are_the_contracted_factor_outputs(self):
        """12.2b: the explicit factors are read under the contracted names of their
        writers (water_stress_factor, temperature_stress_factor, nutrient_stress_factor);
        the former plain names are not read (neutral 1 when only they are given)."""
        gamma = BVOCEnvironmentalResponse.comprehensive_stress_response(
            {
                "water_stress_factor": torch.tensor(0.8),
                "temperature_stress_factor": torch.tensor(0.9),
                "nutrient_stress_factor": torch.tensor(0.7),
            }
        )
        assert gamma.item() == pytest.approx(0.7)
        plain = BVOCEnvironmentalResponse.comprehensive_stress_response(
            {"water_stress": torch.tensor(0.2), "nutrient_stress": torch.tensor(0.3)}
        )
        assert plain.item() == 1.0

    def test_stress_drivers_come_from_one_state_mapping(self):
        """The explicit factors and the soil_moisture driver are read from the same
        state mapping; there is no separate environment channel."""
        params = inspect.signature(
            BVOCEnvironmentalResponse.comprehensive_stress_response
        ).parameters
        assert list(params) == ["state"]
        drought = BVOCEnvironmentalResponse.drought_stress_response(
            soil_moisture=torch.tensor(0.35)
        )
        gamma = BVOCEnvironmentalResponse.comprehensive_stress_response(
            {"soil_moisture": torch.tensor(0.35), "nutrient_stress_factor": torch.tensor(0.9)}
        )
        torch.testing.assert_close(gamma, drought)
        explicit = BVOCEnvironmentalResponse.comprehensive_stress_response(
            {"water_stress_factor": torch.tensor(0.7), "soil_moisture": torch.tensor(0.2)}
        )
        assert explicit.item() == pytest.approx(0.7)


class TestBVOCResponseIntegratorL1:
    """L1 unit tests for BVOC response integrator."""

    @pytest.fixture
    def integrator(self):
        """Create response integrator for testing."""
        return BVOCResponseIntegrator()

    @pytest.fixture
    def standard_conditions(self):
        """Standard environmental conditions for testing."""
        return {
            'temperature': torch.tensor(298.15),  # 25 degC
            'ppfd': torch.tensor(1000.0),         # 1000 umol m-2 s-1
            'state': {
                'water_stress_factor': torch.tensor(1.0),
                'temperature_stress_factor': torch.tensor(1.0)
            },
        }

    def test_integrator_instantiation(self, integrator):
        """Test L1.1: Response integrator instantiation."""
        assert isinstance(integrator, BVOCResponseIntegrator)
        assert hasattr(integrator, 'light_response')
        assert hasattr(integrator, 'temp_response')
        assert hasattr(integrator, 'env_response')
        assert hasattr(integrator, 'device')

    def test_integrator_carries_no_functor_metadata(self):
        """The response helper is composed inside the synthesis Functors; it is not
        a Functor, declares no FunctorMetadata and publishes no gamma_* outputs."""
        import relic.scientific.biosphere.bvoc.responses as responses
        from relic.core.functor import FunctorBase

        assert not issubclass(BVOCResponseIntegrator, FunctorBase)
        assert not hasattr(BVOCResponseIntegrator, "metadata")
        assert not hasattr(responses, "BVOC_RESPONSE_METADATA")

    def test_helper_docstrings_cite_supporting_registered_works(self):
        """Helper docstrings cite registered works whose content matches the helper:
        Penuelas and Llusia (2003) address thermal protection rather than drought,
        and Brilli et al. (2007) report a drought response only; the leaf-age
        helper whose power law had no literature basis is retired."""
        from relic.scientific.biosphere.bvoc.responses import BVOCEnvironmentalResponse

        drought_doc = BVOCEnvironmentalResponse.drought_stress_response.__doc__
        integrated_doc = BVOCResponseIntegrator.calculate_integrated_response.__doc__
        assert not hasattr(BVOCEnvironmentalResponse, "leaf_age_response")
        assert "Penuelas" not in drought_doc
        assert "Guenther et al. (2006)" in drought_doc
        assert "drought/temperature stress interactions" not in integrated_doc

    def test_calculate_all_activity_factors_isoprene(self, integrator, standard_conditions):
        """Test L1.2: Calculate all activity factors for isoprene."""
        results = integrator.calculate_all_activity_factors(
            compound='isoprene',
            temperature=standard_conditions['temperature'],
            ppfd=standard_conditions['ppfd'],
            state=standard_conditions['state']
        )

        # Check all expected factors are present
        expected_factors = [
            'gamma_light', 'gamma_temperature', 'gamma_stress', 'gamma_total'
        ]

        for factor in expected_factors:
            assert factor in results
            assert isinstance(results[factor], torch.Tensor)
            assert torch.isfinite(results[factor]).all()

        # Check value ranges
        assert results['gamma_light'] >= 0.0
        assert results['gamma_temperature'] > 0.0
        assert results['gamma_stress'].item() == pytest.approx(1.0)
        assert results['gamma_total'] >= 0.0
        assert 'gamma_co2' not in results

        # Verify total is product of components
        expected_total = (results['gamma_light'] *
                         results['gamma_temperature'] *
                         results['gamma_stress'])

        assert torch.allclose(results['gamma_total'], expected_total, rtol=0.01)

    def test_unregistered_activity_requires_a_registered_light_fraction(
        self, integrator, standard_conditions
    ):
        """A compound class without a registered light-dependent fraction is
        refused instead of receiving an invented one."""
        with pytest.raises(ValueError, match="light-dependent fraction"):
            integrator.calculate_all_activity_factors(
                compound='unknown',
                temperature=standard_conditions['temperature'],
                ppfd=standard_conditions['ppfd'],
                state=standard_conditions['state']
            )

    def test_leaf_age_factor_is_not_an_activity_input(self, integrator, standard_conditions):
        """The retired leaf-age power law is not applied to a supplied factor."""
        state_with_age = dict(standard_conditions['state'])
        state_with_age['leaf_age_factor'] = torch.tensor(0.2)
        results = integrator.calculate_all_activity_factors(
            compound='sesquiterpenes',
            temperature=standard_conditions['temperature'],
            ppfd=standard_conditions['ppfd'],
            state=state_with_age
        )
        assert 'gamma_age' not in results
        expected_total = (results['gamma_light'] *
                         results['gamma_temperature'] *
                         results['gamma_stress'])
        assert torch.allclose(results['gamma_total'], expected_total)

    def test_activity_factors_temperature_sensitivity(self, integrator):
        """Test L1.5: Temperature sensitivity of activity factors."""
        temperatures = torch.tensor([283.15, 298.15, 313.15])  # 10, 25, 40 degC
        ppfd = torch.tensor(1000.0)
        state = {'water_stress_factor': torch.tensor(1.0), 'temperature_stress_factor': torch.tensor(1.0)}

        # Isoprene follows the peaked Eq. (8) response up to T_opt
        results_iso = []
        for temp in temperatures:
            result = integrator.calculate_all_activity_factors(
                'isoprene', temp, ppfd, state)
            results_iso.append(result['gamma_total'].item())
        assert results_iso[0] < results_iso[1] < results_iso[2]

        # Sesquiterpenes follow the monotonic Eq. (11) response
        results_sqt = []
        for temp in temperatures:
            result = integrator.calculate_all_activity_factors(
                'sesquiterpenes', temp, ppfd, state)
            results_sqt.append(result['gamma_total'].item())
        assert results_sqt[1] > results_sqt[0]
        assert results_sqt[2] > results_sqt[1]

    def test_activity_factors_light_sensitivity(self, integrator):
        """Test L1.6: Light sensitivity of activity factors."""
        temperature = torch.tensor(298.15)
        ppfd_values = torch.tensor([0.0, 500.0, 1000.0, 2000.0])
        state = {'water_stress_factor': torch.tensor(1.0), 'temperature_stress_factor': torch.tensor(1.0)}

        # Isoprene is fully light dependent
        results_iso = []
        for ppfd in ppfd_values:
            result = integrator.calculate_all_activity_factors(
                'isoprene', temperature, ppfd, state)
            results_iso.append(result['gamma_total'].item())
        assert results_iso[0] == 0.0
        assert results_iso[1] > results_iso[0]
        assert results_iso[2] > results_iso[1]
        assert results_iso[3] > results_iso[2]

        # Sesquiterpenes keep the light-independent share (1 - LDF) in darkness
        alpha, cl = _light_parameters()
        ldf = _manager_scalar("biosphere.bvoc.megan.ldf_sesquiterpenes", "dimensionless")
        results_sqt = []
        for ppfd in ppfd_values:
            result = integrator.calculate_all_activity_factors(
                'sesquiterpenes', temperature, ppfd, state)
            results_sqt.append(result['gamma_total'].item())
        for ppfd, total in zip(ppfd_values.tolist(), results_sqt):
            gamma_l = (1.0 - ldf) + ldf * alpha * cl * ppfd / math.sqrt(1.0 + (alpha * ppfd) ** 2)
            assert total / results_sqt[0] == pytest.approx(gamma_l / (1.0 - ldf), rel=1e-5)

    def test_activity_factors_stress_integration(self, integrator, standard_conditions):
        """Test L1.7: Stress factor integration."""
        # No stress
        results_no_stress = integrator.calculate_all_activity_factors(
            compound='isoprene',
            temperature=standard_conditions['temperature'],
            ppfd=standard_conditions['ppfd'],
            state={'water_stress_factor': torch.tensor(1.0), 'temperature_stress_factor': torch.tensor(1.0)}
        )

        # With stress
        results_with_stress = integrator.calculate_all_activity_factors(
            compound='isoprene',
            temperature=standard_conditions['temperature'],
            ppfd=standard_conditions['ppfd'],
            state={'water_stress_factor': torch.tensor(0.6), 'temperature_stress_factor': torch.tensor(0.8)}
        )

        # Stress should reduce total activity
        assert results_with_stress['gamma_total'] < results_no_stress['gamma_total']
        assert results_with_stress['gamma_stress'].item() == pytest.approx(0.6, rel=0.01)  # Minimum of stresses

    def test_integrator_reads_one_state_mapping(self, integrator):
        """calculate_all_activity_factors and calculate_integrated_response take no
        environment argument; soil_moisture reaches gamma_stress through the state."""
        factors = inspect.signature(integrator.calculate_all_activity_factors).parameters
        assert list(factors) == ["compound", "temperature", "ppfd", "state"]
        integrated = inspect.signature(integrator.calculate_integrated_response).parameters
        assert list(integrated) == ["state", "compound"]
        drought = BVOCEnvironmentalResponse.drought_stress_response(
            soil_moisture=torch.tensor(0.4)
        )
        responses = integrator.calculate_integrated_response(
            {
                "temperature": torch.tensor(308.15),
                "ppfd": torch.tensor(1200.0),
                "soil_moisture": torch.tensor(0.4),
            },
            compound="isoprene",
        )
        torch.testing.assert_close(responses["stress_response"], drought, check_dtype=False)
        torch.testing.assert_close(
            responses["integrated_response"],
            responses["temperature_response"]
            * responses["light_response"]
            * responses["stress_response"],
        )

    def test_integrator_device_handling(self):
        """Test L1.8: Device handling in integrator."""
        # Test CPU device
        cpu_integrator = BVOCResponseIntegrator(device=torch.device('cpu'))
        assert cpu_integrator.device == torch.device('cpu')

        temperature = torch.tensor(298.15, device='cpu')
        ppfd = torch.tensor(1000.0, device='cpu')
        state = {'water_stress_factor': torch.tensor(1.0, device='cpu')}

        results = cpu_integrator.calculate_all_activity_factors(
            'isoprene', temperature, ppfd, state)

        # All results should be on CPU
        for value in results.values():
            if isinstance(value, torch.Tensor):
                assert value.device == torch.device('cpu')


class TestBVOCResponsesModuleExportsL1:
    """L1 unit tests for module exports and integration."""

    def test_module_exports(self):
        """Test L1.1: Module __all__ exports."""
        from relic.scientific.biosphere.bvoc import responses

        # Check that __all__ is defined
        assert hasattr(responses, '__all__')

        expected_exports = [
            'BVOCLightResponse',
            'BVOCTemperatureResponse',
            'BVOCEnvironmentalResponse',
            'BVOCResponseIntegrator'
        ]

        for item in expected_exports:
            assert item in responses.__all__
            assert hasattr(responses, item)

    def test_activity_laws_are_the_shared_megan_functions(self):
        """The helper composes the megan_model module-level laws instead of
        keeping a second implementation."""
        from relic.scientific.biosphere.bvoc import megan_model, responses

        # The laws the helpers call are the megan_model objects themselves.
        for name in ("megan_class_light_activity", "megan_class_temperature_activity"):
            assert getattr(responses, name) is getattr(megan_model, name)
        # No other MEGAN law or table is re-implemented in responses: a name that
        # exists there must be the megan_model object.
        for name in (
            "megan_light_dependent_activity",
            "megan_light_activity",
            "megan_ldf_temperature_activity",
            "megan_lif_temperature_activity",
            "megan_temperature_activity",
            "MEGAN_TABLE4_ROWS",
        ):
            shared = getattr(megan_model, name)
            assert getattr(responses, name, shared) is shared
        assert not hasattr(responses, "_LIGHT_DEPENDENT_FRACTION_KEYS")

    def test_extreme_temperature_is_evaluated_without_clamp(self):
        """Eq. (8) stays finite far above T_opt and is not lifted to a floor."""
        response = BVOCTemperatureResponse.get_compound_temperature_response(
            "isoprene", torch.tensor([400.15])
        )
        assert torch.isfinite(response).all()
        assert 0.0 < response.item() < 0.01


if __name__ == "__main__":
    pytest.main([__file__])

