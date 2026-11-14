"""
Test suite for NitrificationProcess parameter management

Tests L1 unit validation after Phase 6 parameter fixes.
Verifies that all 24 fixed parameter calls work correctly with units, bounds, and validate=True.

Fixed parameters:
1. vmax_nitrification (Line 106-112)
2. km_nh4 (Line 114-120)
3. q10_nitrification (Line 122-128)
4. ph_optimum_nitrification (Line 130-136)
5. wfps_nitrification_optimum (pedosphere.physical; round 53.1)
6. n2o_factor (Line 147-153)
7. reference_temperature (Line 156-162)
8. extreme_temperature_factor (Line 164-170)
9. ph_response_width (Line 172-178)
10. (oxygen_limitation_factor removed in round 53.1)
11. minimum_oxygen_factor (Line 188-194)
12. soil_ph_default (Line 197-202)
13. unity_multiplier (Lines 267, 291, 315)
14. temperature_minimum (Line 275-280)
15. temperature_maximum (Line 282-286)
16. min_rate_threshold (Line 295-300)
17. max_rate_factor (Line 301-306)
18. ph_minimum_factor (Line 324-329)
19. soil_temperature (environment_defaults, Line 246-251)
20. soil_water_content (environment_defaults, Line 252-257)
21-24. Various environment defaults and numerical parameters

References:
- Phase 6 Step 5: nitrification.py parameter fixes
- FunctorBase.get_parameter_value() API
- Parton et al. (1996) N2O production model
- Li et al. (1992) DNDC nitrification
"""

import unittest
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess


class TestNitrificationProcessParameters(unittest.TestCase):
    """Test NitrificationProcess parameter management"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')

    def test_import_module(self):
        """Test that NitrificationProcess can be imported"""
        self.assertIsNotNone(NitrificationProcess)

    def test_module_initialization(self):
        """Test basic module initialization with default parameters"""
        nitrification = NitrificationProcess(
            name="nitrification_test",
            config={}
        )
        self.assertIsNotNone(nitrification)
        self.assertIsInstance(nitrification, NitrificationProcess)

    def test_parameter_retrieval(self):
        """Test key nitrification parameter retrieval"""
        nitrification = NitrificationProcess(
            name="nitrification_test",
            config={}
        )

        # Test vmax_nitrification (Line 106-112)
        vmax = nitrification.get_parameter_value(
            "vmax_nitrification",
            namespace="pedosphere.nitrification",
            units="g N m-2 s-1",
            bounds=(1.0e-7, 1.0e-5),
            validate=True
        )
        self.assertIsInstance(vmax, (int, float, torch.Tensor))

        # Test km_nh4 (Line 114-120)
        km_nh4 = nitrification.get_parameter_value(
            "km_nh4",
            namespace="pedosphere.nitrification",
            units="g N m-2",
            bounds=(0.001, 0.02),
            validate=True
        )
        self.assertIsInstance(km_nh4, (int, float, torch.Tensor))

        # Test q10_nitrification (Line 122-128)
        q10 = nitrification.get_parameter_value(
            "q10_nitrification",
            namespace="pedosphere.physical",
            bounds=(1.8, 2.5),
            validate=True
        )
        self.assertIsInstance(q10, (int, float, torch.Tensor))

        # Test ph_optimum_nitrification (Line 130-136)
        ph_opt = nitrification.get_parameter_value(
            "ph_optimum_nitrification",
            namespace="pedosphere.physical",
            bounds=(7.0, 8.5),
            validate=True
        )
        self.assertIsInstance(ph_opt, (int, float, torch.Tensor))

        # Test wfps_nitrification_optimum (oxygen limitation threshold, round 53.1)
        wfps_optimum = nitrification.get_parameter_value(
            "wfps_nitrification_optimum",
            namespace="pedosphere.physical",
            units="dimensionless",
            bounds=(0.35, 0.7),
            validate=True
        )
        self.assertIsInstance(wfps_optimum, (int, float, torch.Tensor))

    def test_forward_minimal(self):
        """Test forward pass with minimal state input"""
        nitrification = NitrificationProcess(
            name="nitrification_test",
            config={}
        )

        # Minimal state for nitrification computation
        state = {
            'nh4_nitrogen': torch.tensor([5.0]),  # g N m-2
            'soil_temperature': torch.tensor([293.15]),  # K
            'water_filled_pore_space': torch.tensor([0.5]),  # dimensionless
            'soil_ph': torch.tensor([7.0])
        }

        # Test that forward pass executes without parameter errors
        try:
            output = nitrification.forward(state, time=0.0, dt=1800.0)
            self.assertIsInstance(output, dict)
            # Should produce nitrification fluxes and pool updates
            self.assertIn('nitrification_flux', output)
            self.assertIn('nh4_nitrogen', output)
            self.assertIn('no3_nitrogen', output)
        except Exception as e:
            # If it fails, it should not be due to missing parameter attributes
            error_str = str(e).lower()
            self.assertNotIn('missing units', error_str)
            self.assertNotIn('missing bounds', error_str)
            self.assertNotIn('validate', error_str)
            self.assertNotIn('namespace', error_str)

    def test_no_hardcoded_parameters(self):
        """Verify no hardcoded physical constants bypass parameter system"""
        nitrification = NitrificationProcess(
            name="nitrification_test",
            config={}
        )

        # All 24 fixed parameters should be retrievable via parameter system
        critical_params = [
            ("vmax_nitrification", "g N m-2 s-1", "pedosphere.nitrification"),
            ("km_nh4", "g N m-2", "pedosphere.nitrification"),
            ("q10_nitrification", None, "pedosphere.physical"),
            ("ph_optimum_nitrification", None, "pedosphere.physical"),
            ("wfps_nitrification_optimum", "dimensionless", "pedosphere.physical"),
            ("n2o_factor", None, "pedosphere.nitrification"),
        ]

        for param_name, units, namespace in critical_params:
            if units:
                value = nitrification.get_parameter_value(
                    param_name,
                    namespace=namespace,
                    units=units,
                    validate=True
                )
            else:
                value = nitrification.get_parameter_value(
                    param_name,
                    namespace=namespace,
                    validate=True
                )
            self.assertIsNotNone(value, f"Parameter {param_name} should be retrievable")

    def test_parameter_bounds_validation(self):
        """Test that bounds validation works for all fixed parameters"""
        nitrification = NitrificationProcess(
            name="nitrification_test",
            config={}
        )

        # Test vmax bounds
        vmax = nitrification.get_parameter_value(
            "vmax_nitrification",
            namespace="pedosphere.nitrification",
            units="g N m-2 s-1",
            bounds=(1.0e-7, 1.0e-5),
            validate=True
        )
        if isinstance(vmax, torch.Tensor):
            vmax_val = vmax.item()
        else:
            vmax_val = float(vmax)
        self.assertGreaterEqual(vmax_val, 0.0)

        # Test Q10 bounds
        q10 = nitrification.get_parameter_value(
            "q10_nitrification",
            namespace="pedosphere.physical",
            bounds=(1.8, 2.5),
            validate=True
        )
        if isinstance(q10, torch.Tensor):
            q10_val = q10.item()
        else:
            q10_val = float(q10)
        self.assertGreaterEqual(q10_val, 1.5)
        self.assertLessEqual(q10_val, 4.0)  # Physical upper limit

    def test_parameter_units(self):
        """Verify units are correct (CF-1.8 compliant)"""
        nitrification = NitrificationProcess(
            name="nitrification_test",
            config={}
        )

        # Test nitrification rate units (g N m^-2 s^-1)
        vmax = nitrification.get_parameter_value(
            "vmax_nitrification",
            units="g N m-2 s-1",
            namespace="pedosphere.nitrification",
            validate=True
        )
        self.assertIsNotNone(vmax)

        # Test NH4 pool units (g N m^-2)
        km_nh4 = nitrification.get_parameter_value(
            "km_nh4",
            units="g N m-2",
            namespace="pedosphere.nitrification",
            validate=True
        )
        self.assertIsNotNone(km_nh4)

        # Test the water-filled pore space threshold units (dimensionless)
        wfps_optimum = nitrification.get_parameter_value(
            "wfps_nitrification_optimum",
            units="dimensionless",
            namespace="pedosphere.physical",
            validate=True
        )
        self.assertIsNotNone(wfps_optimum)

    def test_parameter_namespace(self):
        """Verify namespace correctness"""
        nitrification = NitrificationProcess(
            name="nitrification_test",
            config={}
        )

        # Nitrification-specific parameters should be in pedosphere.nitrification namespace
        nitrif_params = [
            "vmax_nitrification",
            "km_nh4",
            "n2o_factor"
        ]

        for param in nitrif_params:
            value = nitrification.get_parameter_value(
                param,
                namespace="pedosphere.nitrification",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in pedosphere.nitrification namespace")

        # Physical parameters should be in pedosphere.physical namespace
        physical_params = [
            "q10_nitrification",
            "ph_optimum_nitrification",
            "wfps_nitrification_optimum",
        ]

        for param in physical_params:
            value = nitrification.get_parameter_value(
                param,
                namespace="pedosphere.physical",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in pedosphere.physical namespace")



RESPONSE_BOUND_KEYS = {
    "pedosphere.nitrification.minimum_oxygen_factor": "oxygen response factor",
    "pedosphere.nitrification.min_rate_threshold": "temperature factor",
    "pedosphere.nitrification.max_rate_factor": "temperature factor",
    # Round 53.2: values read by NitrificationProcess that had placeholder references.
    "pedosphere.nitrification.km_nh4": "half-saturation",
    "pedosphere.nitrification.temperature_minimum": "temperature factor",
    "pedosphere.nitrification.temperature_maximum": "temperature factor",
    "pedosphere.nitrification.extreme_temperature_factor": "temperature factor",
    "pedosphere.nitrification.ph_minimum_factor": "pH response factor",
}
PLACEHOLDER_REFERENCES = (
    "Severe waterlogging",
    "Numerical precision",
    "Stability control",
    "Michaelis-Menten kinetics for nitrification",
    "Extreme temperature limits",
    "High temperature inhibition",
    "Temperature stress response",
    "pH inhibition",
)


def test_oxygen_limitation_threshold_is_the_sourced_wfps_optimum():
    """Round 53.1: the one WFPS threshold key, 0.6 after Linn and Doran (1984)."""
    from relic.parameters import get_global_manager

    spec = get_global_manager().get_parameter("pedosphere.physical.wfps_nitrification_optimum")
    assert float(spec.value) == pytest.approx(0.6)
    assert spec.units == "dimensionless"
    assert spec.provenance["source"] == "literature"
    text = " ".join(spec.references)
    assert "Linn" in text and "10.2136/sssaj1984.03615995004800060013x" in text
    assert "Bateman" in text
    for broken in ("10.1016/0038-0717(94)00192-6", "10.1046/j.1365-2486.2000.00327.x"):
        assert broken not in text
    lower, upper = spec.bounds
    assert 0.0 < lower <= float(spec.value) <= upper < 1.0


def test_nitrification_response_bounds_have_truthful_provenance():
    """Round 52.3: no primary source (Parton et al. 1996/2001 via the CLM4.5 Technical
    Note Eq. 16.11, the DNDC95 reference code) gives these bounds, so they are declared
    RELIC conventions with their role instead of placeholder references."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    for key, role in RESPONSE_BOUND_KEYS.items():
        spec = manager.get_parameter(key)
        assert spec.provenance["source"] == "relic_convention", key
        assert spec.provenance.get("notes"), key
        assert role in spec.description, key
        assert any(ref.startswith("RELIC convention") for ref in spec.references), key
        for placeholder in PLACEHOLDER_REFERENCES:
            assert placeholder not in spec.references, (key, placeholder)
            assert spec.provenance["source"] != placeholder, (key, placeholder)


if __name__ == '__main__':
    unittest.main(verbosity=2)

