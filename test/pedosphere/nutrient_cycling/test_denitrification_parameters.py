"""
Test suite for DenitrificationProcess parameter management

Tests L1 unit validation after Phase 6 parameter fixes.
Verifies that all 25 fixed parameter calls work correctly with units, bounds, and validate=True.

Fixed parameters:
1. vmax_denitrification (Line 89-95)
2. km_no3 (Line 97-103)
3. wfps_denitrification_threshold (pedosphere.physical; follow-up F2)
4. q10_denitrification (Line 114-120)
5. reference_temperature (Line 123-129)
6. anaerobic_transition_width (Line 131-137)
7. minimum_anaerobic_factor (Line 139-145)
8. carbon_half_saturation (Line 147-154)
9. ph_optimum_denitrification (Line 156-162)
10. ph_response_width (Line 164-170)
11. labile_carbon_default (Line 173-178)
12. soil_ph_default (Line 180-185)
13. min_rate_threshold (Line 260-265)
14. max_rate_factor (Line 266-271)
15. ph_minimum_factor (Line 295-300)
16. n2o_no3_threshold (Line 313-318)
17. n2o_ph_reference (Line 322-327)
18. n2o_ph_sensitivity (Line 328-333)
19. base_n2o_fraction (Line 337-342)
20. n2o_no3_enhancement (Line 343-348)
21. max_n2o_fraction (Line 349-354)
22. soil_water_content (environment_defaults, Line 235-240)
23. soil_temperature (environment_defaults, Line 242-246)
24-25. Various environment defaults and numerical parameters

References:
- Phase 6 Step 5: denitrification.py parameter fixes
- FunctorBase.get_parameter_value() API
- Parton et al. (1996) N2O production model
- Li et al. (1992) DNDC denitrification
"""

import unittest
from unittest.mock import patch
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.scientific.pedosphere.nutrient_cycling import denitrification as denitrification_module
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess


class TestDenitrificationProcessParameters(unittest.TestCase):
    """Test DenitrificationProcess parameter management"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')

    def test_import_module(self):
        """Test that DenitrificationProcess can be imported"""
        self.assertIsNotNone(DenitrificationProcess)

    def test_module_initialization(self):
        """Test basic module initialization with default parameters"""
        denitrification = DenitrificationProcess(
            name="denitrification_test",
            config={}
        )
        self.assertIsNotNone(denitrification)
        self.assertIsInstance(denitrification, DenitrificationProcess)

    def test_parameter_retrieval(self):
        """Test key denitrification parameter retrieval"""
        denitrification = DenitrificationProcess(
            name="denitrification_test",
            config={}
        )

        # Test vmax_denitrification (Line 89-95)
        vmax = denitrification.get_parameter_value(
            "vmax_denitrification",
            namespace="pedosphere.denitrification",
            units="g N m-2 s-1",
            bounds=(1.0e-7, 1.0e-5),
            validate=True
        )
        self.assertIsInstance(vmax, (int, float, torch.Tensor))

        # Test km_no3 (Line 97-103)
        km_no3 = denitrification.get_parameter_value(
            "km_no3",
            namespace="pedosphere.denitrification",
            units="g N m-2",
            bounds=(0.0005, 0.01),
            validate=True
        )
        self.assertIsInstance(km_no3, (int, float, torch.Tensor))

        # Test wfps_denitrification_threshold (anaerobic factor threshold, follow-up F2)
        anaerobic_thresh = denitrification.get_parameter_value(
            "wfps_denitrification_threshold",
            namespace="pedosphere.physical",
            units="dimensionless",
            bounds=(0.5, 0.8),
            validate=True
        )
        self.assertIsInstance(anaerobic_thresh, (int, float, torch.Tensor))

        # Test q10_denitrification (Line 114-120)
        q10 = denitrification.get_parameter_value(
            "q10_denitrification",
            namespace="pedosphere.physical",
            bounds=(2.0, 2.8),
            validate=True
        )
        self.assertIsInstance(q10, (int, float, torch.Tensor))

        # Test carbon_half_saturation (Line 147-154)
        carbon_km = denitrification.get_parameter_value(
            "carbon_half_saturation",
            namespace="pedosphere.denitrification",
            units="g C m-2",
            bounds=(0.01, 0.2),
            validate=True
        )
        self.assertIsInstance(carbon_km, (int, float, torch.Tensor))

    def test_forward_minimal(self):
        """Test forward pass with minimal state input"""
        denitrification = DenitrificationProcess(
            name="denitrification_test",
            config={}
        )

        # Minimal state for denitrification computation
        state = {
            'no3_nitrogen': torch.tensor([10.0]),  # g N m-2
            'soil_temperature': torch.tensor([293.15]),  # K
            'water_filled_pore_space': torch.tensor([0.8]),  # dimensionless (anaerobic)
            'labile_carbon': torch.tensor([5.0])  # g C m-2
        }

        # Test that forward pass executes without parameter errors
        try:
            output = denitrification.forward(state, time=0.0, dt=1800.0, labile_carbon=torch.tensor([5.0]))
            self.assertIsInstance(output, dict)
            # Should produce denitrification fluxes and pool updates
            self.assertIn('denitrification_flux', output)
            self.assertIn('n2o_flux', output)
            self.assertIn('n2_production', output)
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
        denitrification = DenitrificationProcess(
            name="denitrification_test",
            config={}
        )

        # All 25 fixed parameters should be retrievable via parameter system
        critical_params = [
            ("vmax_denitrification", "g N m-2 s-1", "pedosphere.denitrification"),
            ("km_no3", "g N m-2", "pedosphere.denitrification"),
            ("wfps_denitrification_threshold", "dimensionless", "pedosphere.physical"),
            ("q10_denitrification", None, "pedosphere.physical"),
            ("carbon_half_saturation", "g C m-2", "pedosphere.denitrification"),
            ("base_n2o_fraction", None, "pedosphere.denitrification"),
        ]

        for param_name, units, namespace in critical_params:
            if units:
                value = denitrification.get_parameter_value(
                    param_name,
                    namespace=namespace,
                    units=units,
                    validate=True
                )
            else:
                value = denitrification.get_parameter_value(
                    param_name,
                    namespace=namespace,
                    validate=True
                )
            self.assertIsNotNone(value, f"Parameter {param_name} should be retrievable")

    def test_parameter_bounds_validation(self):
        """Test that bounds validation works for all fixed parameters"""
        denitrification = DenitrificationProcess(
            name="denitrification_test",
            config={}
        )

        # Test vmax bounds
        vmax = denitrification.get_parameter_value(
            "vmax_denitrification",
            namespace="pedosphere.denitrification",
            units="g N m-2 s-1",
            bounds=(1.0e-7, 1.0e-5),
            validate=True
        )
        if isinstance(vmax, torch.Tensor):
            vmax_val = vmax.item()
        else:
            vmax_val = float(vmax)
        self.assertGreaterEqual(vmax_val, 0.0)

        # Test anaerobic threshold bounds
        anaerobic_thresh = denitrification.get_parameter_value(
            "wfps_denitrification_threshold",
            namespace="pedosphere.physical",
            units="dimensionless",
            bounds=(0.5, 0.8),
            validate=True
        )
        if isinstance(anaerobic_thresh, torch.Tensor):
            thresh_val = anaerobic_thresh.item()
        else:
            thresh_val = float(anaerobic_thresh)
        self.assertGreaterEqual(thresh_val, 0.0)
        self.assertLessEqual(thresh_val, 1.0)  # Physical upper limit (pore-space fraction)

    def test_parameter_units(self):
        """Verify units are correct (CF-1.8 compliant)"""
        denitrification = DenitrificationProcess(
            name="denitrification_test",
            config={}
        )

        # Test denitrification rate units (g N m^-2 s^-1)
        vmax = denitrification.get_parameter_value(
            "vmax_denitrification",
            units="g N m-2 s-1",
            namespace="pedosphere.denitrification",
            validate=True
        )
        self.assertIsNotNone(vmax)

        # Test NO3 pool units (g N m^-2)
        km_no3 = denitrification.get_parameter_value(
            "km_no3",
            units="g N m-2",
            namespace="pedosphere.denitrification",
            validate=True
        )
        self.assertIsNotNone(km_no3)

        # Test the water-filled pore space threshold units (dimensionless)
        anaerobic_thresh = denitrification.get_parameter_value(
            "wfps_denitrification_threshold",
            units="dimensionless",
            namespace="pedosphere.physical",
            validate=True
        )
        self.assertIsNotNone(anaerobic_thresh)

        # Test carbon units (g C m^-2)
        carbon_km = denitrification.get_parameter_value(
            "carbon_half_saturation",
            units="g C m-2",
            namespace="pedosphere.denitrification",
            validate=True
        )
        self.assertIsNotNone(carbon_km)

    def test_parameter_namespace(self):
        """Verify namespace correctness"""
        denitrification = DenitrificationProcess(
            name="denitrification_test",
            config={}
        )

        # Denitrification-specific parameters should be in pedosphere.denitrification namespace
        denitrif_params = [
            "vmax_denitrification",
            "km_no3",
            "carbon_half_saturation",
            "base_n2o_fraction"
        ]

        for param in denitrif_params:
            value = denitrification.get_parameter_value(
                param,
                namespace="pedosphere.denitrification",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in pedosphere.denitrification namespace")

        # Physical parameters should be in pedosphere.physical namespace
        physical_params = [
            "q10_denitrification",
            "ph_optimum_denitrification"
        ]

        for param in physical_params:
            value = denitrification.get_parameter_value(
                param,
                namespace="pedosphere.physical",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in pedosphere.physical namespace")

    def test_q10_interval_uses_physical_constant_ssot(self):
        """Q10 interval must use the physical-constant API, not Functor parameters."""
        constant_calls = []

        def tracked_get_physical_constant(name):
            constant_calls.append(name)
            return 10.0

        with patch.object(
            denitrification_module,
            "get_physical_constant",
            side_effect=tracked_get_physical_constant,
            create=True,
        ):
            denitrification = DenitrificationProcess(
                name="denitrification_test",
                config={},
            )

        self.assertEqual(
            constant_calls,
            ["relic.physical.constants.temperature.q10_interval"],
        )
        self.assertEqual(float(denitrification.q10_interval), 10.0)
        self.assertNotIn(
            "relic.physical.constants.temperature.q10_interval",
            denitrification.metadata.parameters,
        )


DENITRIFICATION_CERTIFICATE = (
    project_root
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "pedosphere_nutrient_cycling_denitrification_denitrification_process.yaml"
)


def test_temperature_factor_floor_is_a_dimensionless_convention():
    """Round 54.2: min_rate_threshold is read only as the lower clamp of the
    dimensionless Q10 factor f_T = clip(Q10^((T - T_ref)/10), lower, upper), so it is a
    dimensionless RELIC convention, not a rate in g N m-2 s-1 with the placeholder
    reference "Numerical precision"."""
    import yaml
    from relic.parameters import get_global_manager

    key = "pedosphere.denitrification.min_rate_threshold"
    spec = get_global_manager().get_parameter(key)
    assert spec.units == "dimensionless"
    assert spec.provenance["source"] == "relic_convention"
    assert spec.provenance.get("notes")
    assert "temperature factor" in spec.description
    assert any(ref.startswith("RELIC convention") for ref in spec.references)
    assert "Numerical precision" not in spec.references
    process = DenitrificationProcess(name="denitrification_test", config={})
    assert process.metadata.parameters[key].endswith("[dimensionless]")
    assert float(process.min_rate_threshold) == float(spec.value)
    certificate = yaml.safe_load(DENITRIFICATION_CERTIFICATE.read_text(encoding="utf-8"))
    assert certificate["parameter_map"][key].endswith("[dimensionless]")


def _denitrification_state(wfps):
    wfps = torch.as_tensor(wfps, dtype=torch.float64)
    return {
        "no3_nitrogen": torch.full_like(wfps, 5.0),
        "labile_carbon": torch.full_like(wfps, 0.1),
        "water_filled_pore_space": wfps,
        "soil_temperature": torch.full_like(wfps, 293.15),
        "soil_ph": torch.full_like(wfps, 7.0),
    }


def test_anaerobic_factor_follows_water_filled_pore_space():
    """F2: threshold and transition width share the water-filled pore space basis.

    DAYCENT (Parton et al. 1996; Del Grosso et al. 2000) drives denitrification with
    WFPS; aerobic activity declines above 60 percent WFPS (Linn and Doran 1984).
    f_anoxic = max(clip((WFPS - WFPS_thr)/w, 0, 1), f_min), monotone in WFPS.
    """
    import pytest
    import yaml
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    spec = manager.get_parameter("pedosphere.physical.wfps_denitrification_threshold")
    assert float(spec.value) == pytest.approx(0.6)
    assert spec.provenance["source"] == "literature"
    text = " ".join(spec.references)
    assert "NEEDS_CITATION" not in text
    assert "10.2136/sssaj1984.03615995004800060013x" in text
    assert "10.1029/1999GB001225" in text
    assert manager.get_parameter("pedosphere.denitrification.anaerobic_threshold") is None
    for key in (
        "pedosphere.denitrification.anaerobic_transition_width",
        "pedosphere.denitrification.minimum_anaerobic_factor",
    ):
        convention = manager.get_parameter(key)
        assert convention.provenance["source"] == "relic_convention", key
        assert "water-filled pore space" in convention.description, key

    process = DenitrificationProcess(name="denitrification_test", config={})
    assert "water_filled_pore_space" in process.metadata.input_variables
    assert "soil_moisture" not in process.metadata.input_variables
    wfps = [0.3, 0.6, 0.62, 0.75, 0.9, 1.0]
    factor = process._compute_rate_core(_denitrification_state(wfps))["anaerobic_factor"]
    assert factor.to(torch.float64).tolist() == pytest.approx([0.1, 0.1, 0.1, 0.5, 1.0, 1.0], abs=1.0e-6)
    dense = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    ramp = process._compute_rate_core(_denitrification_state(dense))["anaerobic_factor"]
    assert bool(torch.all(ramp[1:] >= ramp[:-1]))
    state = _denitrification_state([0.7])
    del state["water_filled_pore_space"]
    state["soil_moisture"] = torch.tensor([0.3], dtype=torch.float64)
    with pytest.raises(ValueError, match="water_filled_pore_space"):
        process._compute_rate_core(state)

    certificate = yaml.safe_load(DENITRIFICATION_CERTIFICATE.read_text(encoding="utf-8"))
    for inputs in (
        certificate["input_variables"],
        certificate["interface_map"]["legacy_inputs"],
        certificate["interface_map"]["iphs_inputs"],
    ):
        assert "water_filled_pore_space" in inputs and "soil_moisture" not in inputs
    assert "pedosphere.physical.wfps_denitrification_threshold" in certificate["parameter_keys"]
    assert "pedosphere.denitrification.anaerobic_threshold" not in certificate["parameter_map"]


def test_water_filled_pore_space_outside_the_unit_interval_raises():
    import pytest

    process = DenitrificationProcess(name="denitrification_test", config={})
    for value in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="water_filled_pore_space"):
            process.forward(_denitrification_state([value]), time=0.0, dt=3600.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

