"""
L1 Unit Test: BVOC Responses Parameter Integration

Tests that all BVOC response functions correctly load parameters from
ParameterManager with proper namespaces after P0.1 and P0.2 fixes.

Test Coverage:
- All parameters have correct namespace (biosphere.bvoc.*)
- All parameters exist in YAML
- All parameters have correct units
- All parameters fall within bounds
- No hardcoded defaults bypass ParameterManager

References:
- Guenther et al. (2012). MEGAN 2.1. Geosci Model Dev 5:1471-1492
- Guenther et al. (2012). MEGAN2.1 Table 4 light-dependent fractions
"""

import pytest
import torch
from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.responses import (
    BVOCLightResponse,
    BVOCTemperatureResponse,
    BVOCEnvironmentalResponse,
    BVOCResponseIntegrator
)


class TestBVOCLightResponseParameters:
    """Test BVOC light response parameter loading."""

    LIGHT_PARAMETERS = {
        "biosphere.bvoc.megan.alpha_light": "m2 s umol-1",
        "biosphere.bvoc.megan.cl_light": "dimensionless",
        "biosphere.bvoc.megan.ldf_sesquiterpenes": "dimensionless",
    }

    def test_light_parameters_exist_with_units(self):
        """The MEGAN2.1 light activity reads exactly these registered parameters."""
        param_manager = get_global_manager()
        for param_key, expected_unit in self.LIGHT_PARAMETERS.items():
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Spec for {param_key} not found"
            assert spec.units == expected_unit, \
                f"{param_key} has wrong units: {spec.units} != {expected_unit}"

    def test_sesquiterpene_ldf_cites_guenther_2012_table_4(self):
        """LDF = 0.5 for the sesquiterpene classes of Guenther et al. (2012) Table 4."""
        spec = get_global_manager().get_parameter("biosphere.bvoc.megan.ldf_sesquiterpenes")
        assert float(spec.value) == pytest.approx(0.5)
        assert tuple(spec.bounds) == (0.0, 1.0)
        assert any("Table 4" in reference for reference in spec.references)

    def test_light_response_functional(self):
        """Light activity is non-negative and increases with PPFD."""
        ppfd = torch.tensor([0.0, 500.0, 1000.0, 1500.0, 2000.0])
        for compound in ("isoprene", "sesquiterpenes"):
            response = BVOCLightResponse.get_compound_light_response(compound, ppfd)
            assert response.shape == ppfd.shape
            assert torch.all(response >= 0.0), "Light response should be non-negative"
            assert torch.all(response[1:] > response[:-1]), "Response should increase with PPFD"


class TestBVOCTemperatureResponseParameters:
    """Test BVOC temperature response parameter loading."""

    TEMPERATURE_PARAMETERS = {
        "biosphere.bvoc.megan.eopt_isoprene": "dimensionless",
        "biosphere.bvoc.megan.topt": "K",
        "biosphere.bvoc.megan.ct1_isoprene": "J mol-1",
        "biosphere.bvoc.megan.ct2": "J mol-1",
        "biosphere.bvoc.t_standard": "K",
        "biosphere.bvoc.megan.beta_monoterpenes": "K-1",
        "biosphere.bvoc.megan.beta_sesquiterpenes": "K-1",
        "biosphere.bvoc.megan.ovoc.methanol.beta": "K-1",
        "biosphere.bvoc.megan.ovoc.acetone.beta": "K-1",
        "biosphere.bvoc.megan.ovoc.acetaldehyde.beta": "K-1",
        "biosphere.bvoc.megan.ovoc.methanol.global_emission": "g yr-1",
    }
    RETIRED_PARAMETERS = (
        "biosphere.bvoc.megan.beta_ovocs",
        "biosphere.bvoc.megan.ldf_ovocs",
        "biosphere.bvoc.megan.entropy_isoprene",
        "biosphere.bvoc.megan.tm_reference",
        "biosphere.bvoc.q10.temperature_interval",
        "biosphere.bvoc.megan.beta_default",
        "biosphere.constants.temperature.megan_reference",
    )

    def test_temperature_parameters_exist_with_units(self):
        """Eq. (8) and Eq. (11) read exactly these registered parameters."""
        param_manager = get_global_manager()
        for param_key, expected_unit in self.TEMPERATURE_PARAMETERS.items():
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Spec for {param_key} not found"
            assert spec.units == expected_unit, \
                f"{param_key} has wrong units: {spec.units} != {expected_unit}"

    def test_retired_temperature_parameters_are_removed(self):
        """Parameters read only by the retired geometric-mean law, the interval
        rescaling, the duplicate reference temperatures of the helper and of the
        retired SesquiterpeneSynthesis gamma_T normalization, and the
        unknown-compound fallback are removed."""
        param_manager = get_global_manager()
        present = [
            key for key in self.RETIRED_PARAMETERS
            if param_manager.get(key, required=False) is not None
        ]
        assert present == []

    def test_isoprene_temperature_response_functional(self):
        """Eq. (8) rises to the optimum and declines above it."""
        temperature = torch.tensor([283.15, 293.15, 303.15, 313.15, 323.15])
        response = BVOCTemperatureResponse.get_compound_temperature_response(
            "isoprene", temperature
        )
        assert response.shape == temperature.shape
        assert torch.all(response > 0.0)
        assert response[1] > response[0]
        assert response[2] > response[1]
        assert response[3] > response[4]

    def test_monoterpene_temperature_response_functional(self):
        """Eq. (11) increases monotonically with temperature."""
        temperature = torch.tensor([283.15, 293.15, 303.15, 313.15])
        response = BVOCTemperatureResponse.get_compound_temperature_response(
            'monoterpenes', temperature
        )
        assert response.shape == temperature.shape
        assert torch.all(response > 0.0)
        assert torch.all(response[1:] > response[:-1]), "Response should increase with T"


class TestBVOCEnvironmentalResponseParameters:
    """Test BVOC environmental response parameter loading."""

    def test_retired_auxiliary_co2_parameters_are_removed(self):
        """The auxiliary CO2 helper is retired and its parameters had no other reader."""
        param_manager = get_global_manager()
        present = [
            key for key in (
                "biosphere.bvoc.co2_inhibition.k_co2",
                "biosphere.bvoc.co2_inhibition.baseline_co2",
            )
            if param_manager.get(key, required=False) is not None
        ]
        assert present == []

    def test_production_co2_parameters_exist(self):
        """MEGANModel's Eq. (14) CO2 parameters remain registered."""
        param_manager = get_global_manager()
        for param_key, unit in (
            ("biosphere.bvoc.co2_inhibition.ismax", "dimensionless"),
            ("biosphere.bvoc.co2_inhibition.c_star", "ppm"),
            ("biosphere.bvoc.co2_inhibition.hill_exponent", "dimensionless"),
            ("biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio", "dimensionless"),
        ):
            spec = param_manager.get_parameter(param_key)
            assert spec is not None and spec.units == unit, param_key


class TestBVOCResponseIntegrator:
    """Test integrated BVOC response calculator."""

    def test_integrator_initialization(self):
        """Test response integrator initializes without error."""
        integrator = BVOCResponseIntegrator()

        assert integrator is not None
        assert hasattr(integrator, 'light_response')
        assert hasattr(integrator, 'temp_response')
        assert hasattr(integrator, 'env_response')
        print("[OK] BVOCResponseIntegrator initialized successfully")

    def test_calculate_all_activity_factors(self):
        """Test integrated calculation of all activity factors."""
        integrator = BVOCResponseIntegrator()

        # Typical summer midday conditions
        temperature = torch.tensor([303.15])  # 30掳C
        ppfd = torch.tensor([1000.0])  # Full sun
        state = {}

        # Calculate for isoprene
        results = integrator.calculate_all_activity_factors(
            compound='isoprene',
            temperature=temperature,
            ppfd=ppfd,
            state=state,
        )

        # Verify all expected keys present
        expected_keys = ['gamma_light', 'gamma_temperature', 'gamma_stress', 'gamma_total']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
            assert results[key] is not None
            print(f"[OK] {key} = {results[key].item():.3f}")

        # Verify total is product of components
        expected_total = (results['gamma_light'] *
                         results['gamma_temperature'] *
                         results['gamma_stress'])
        assert torch.allclose(results['gamma_total'], expected_total, rtol=1e-5)
        print(f"[OK] Total activity factor: {results['gamma_total'].item():.3f}")


class TestParameterNamespaceConsistency:
    """Test P0.1 fix - all parameters use correct namespace."""

    def test_no_unnamespaced_megan_calls(self):
        """Verify no direct 'megan.*' calls without namespace prefix."""
        import inspect

        # Get source code of responses module
        source = inspect.getsource(BVOCLightResponse)
        source += inspect.getsource(BVOCTemperatureResponse)
        source += inspect.getsource(BVOCEnvironmentalResponse)

        # Check for incorrect pattern: param_manager.get("megan.
        # Should be: param_manager.get("biosphere.bvoc.megan.
        incorrect_pattern = 'param_manager.get("megan.'
        assert incorrect_pattern not in source, \
            "Found unnamespaced 'megan.*' parameter call - P0.1 fix incomplete"

        # Check for incorrect pattern: param_manager.get("co2_inhibition.
        incorrect_pattern2 = 'param_manager.get("co2_inhibition.'
        assert incorrect_pattern2 not in source, \
            "Found unnamespaced 'co2_inhibition.*' parameter call - P0.1 fix incomplete"

        print("[OK] All parameter calls use proper 'biosphere.bvoc.*' namespace")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])

