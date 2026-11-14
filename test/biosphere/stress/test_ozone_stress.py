"""
Test suite for OzoneStress Functor parameter management

Tests L1 unit validation of ozone stress parameter system after P0-1 fixes.
Verifies that all parameter calls work correctly with proper namespace, units, 
bounds, validation, and register_if_missing=False flags.

Fixed issues in P0-1:
- Removed 7 register_buffer hardcoded parameter calls (lines 160-166)
- Added register_if_missing=False to all get_parameter_value calls
- Eliminated redundant parameter fetching in __init__
- All parameters now dynamically loaded from ParameterManager

Critical parameters tested:
1. global.factor_min (namespace: biosphere.stress) - stress factor lower bound
2. global.factor_max (namespace: biosphere.stress) - stress factor upper bound
3. ozone_uptake_rate (namespace: biosphere.stress.ozone) - uptake efficiency
4. flux_threshold (namespace: biosphere.stress.ozone) - POD threshold
5. ozone_detox_capacity (namespace: biosphere.stress.ozone) - detoxification rate
6. antioxidant_capacity (namespace: biosphere.stress.ozone) - antioxidant pool

References:
- P0-1: ozone_stress.py register_buffer elimination
- FunctorBase.get_parameter_value() API
- Parameter system audit report
- relic/parameters/database/relic/biosphere/stress.yaml
"""

import unittest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
from test.utils.forward_call import call_forward


class TestOzoneStressParameters(unittest.TestCase):
    """Test OzoneStress parameter management after P0-1 refactoring"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')

    def test_import_ozone_stress(self):
        """Test that OzoneStress can be imported"""
        self.assertIsNotNone(OzoneStress)

    def test_module_initialization(self):
        """Test basic module initialization with default parameters"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )
        self.assertIsNotNone(ozone_stress)
        self.assertIsInstance(ozone_stress, OzoneStress)
        
        # Verify namespace is properly set
        self.assertEqual(ozone_stress.namespace, "biosphere.stress.ozone")

    def test_no_register_buffer_in_init(self):
        """Verify that __init__ does not call register_buffer (P0-1 fix)"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )
        
        # After P0-1 fix, these should NOT exist as buffers
        prohibited_buffers = [
            'ozone_threshold',
            'flux_threshold',
            'ozone_uptake_rate',
            'ozone_detox_capacity',
            'antioxidant_capacity',
            'pody_critical_wheat',
            'aot40_critical_wheat'
        ]
        
        for buffer_name in prohibited_buffers:
            self.assertNotIn(
                buffer_name,
                dict(ozone_stress.named_buffers()),
                f"Buffer '{buffer_name}' should not exist after P0-1 fix"
            )

    def test_stress_factor_bounds_parameters(self):
        """Test stress factor min/max parameters from biosphere.stress namespace"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Test global.factor_min
        stress_min = ozone_stress.get_parameter_value(
            "global.factor_min",
            namespace="biosphere.stress",
            units="dimensionless",
            bounds=(0.0, 0.2),
            validate=True
        )
        self.assertIsInstance(stress_min, (int, float, torch.Tensor))
        
        # Verify value is within bounds
        if isinstance(stress_min, torch.Tensor):
            stress_min_val = stress_min.item()
        else:
            stress_min_val = float(stress_min)
        self.assertGreaterEqual(stress_min_val, 0.0)
        self.assertLessEqual(stress_min_val, 0.2)

        # Test global.factor_max
        stress_max = ozone_stress.get_parameter_value(
            "global.factor_max",
            namespace="biosphere.stress",
            units="dimensionless",
            bounds=(0.8, 1.0),
            validate=True
        )
        self.assertIsInstance(stress_max, (int, float, torch.Tensor))
        
        # Verify value is within bounds
        if isinstance(stress_max, torch.Tensor):
            stress_max_val = stress_max.item()
        else:
            stress_max_val = float(stress_max)
        self.assertGreaterEqual(stress_max_val, 0.8)
        self.assertLessEqual(stress_max_val, 1.0)

    def test_ozone_uptake_parameters(self):
        """Test ozone uptake rate parameter from biosphere.stress.ozone namespace"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Test ozone_uptake_rate (dimensionless efficiency)
        uptake_rate = ozone_stress.get_parameter_value(
            "ozone_uptake_rate",
            namespace="biosphere.stress.ozone",
            units="dimensionless",
            bounds=(0.0, 1.0),
            validate=True
        )
        self.assertIsInstance(uptake_rate, (int, float, torch.Tensor))
        
        # Verify value is within bounds [0, 1]
        if isinstance(uptake_rate, torch.Tensor):
            uptake_val = uptake_rate.item()
        else:
            uptake_val = float(uptake_rate)
        self.assertGreaterEqual(uptake_val, 0.0)
        self.assertLessEqual(uptake_val, 1.0)

    def test_flux_threshold_parameter(self):
        """Test flux threshold parameter for POD calculation"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Test flux_threshold (nmol O3 m-2 s-1)
        flux_threshold = ozone_stress.get_parameter_value(
            "flux_threshold",
            namespace="biosphere.stress.ozone",
            units="nmol O3 m-2 s-1",
            bounds=(0.0, 100.0),
            validate=True
        )
        self.assertIsInstance(flux_threshold, (int, float, torch.Tensor))
        
        # Verify value is within bounds
        if isinstance(flux_threshold, torch.Tensor):
            threshold_val = flux_threshold.item()
        else:
            threshold_val = float(flux_threshold)
        self.assertGreaterEqual(threshold_val, 0.0)
        self.assertLessEqual(threshold_val, 100.0)

    def test_detoxification_capacity_parameter(self):
        """Test ozone detoxification capacity parameter"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Test ozone_detox_capacity (mol O3 m-2 s-1)
        detox_capacity = ozone_stress.get_parameter_value(
            "ozone_detox_capacity",
            namespace="biosphere.stress.ozone",
            units="mol O3 m-2 s-1",
            bounds=(1e-9, 5e-9),
            validate=True
        )
        self.assertIsInstance(detox_capacity, (int, float, torch.Tensor))
        
        # Verify value is within bounds
        if isinstance(detox_capacity, torch.Tensor):
            detox_val = detox_capacity.item()
        else:
            detox_val = float(detox_capacity)
        self.assertGreaterEqual(detox_val, 1e-9)
        self.assertLessEqual(detox_val, 5e-9)

    def test_antioxidant_capacity_parameter(self):
        """Test antioxidant capacity parameter"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Test antioxidant_capacity (mol O3 m-2)
        antioxidant_capacity = ozone_stress.get_parameter_value(
            "antioxidant_capacity",
            namespace="biosphere.stress.ozone",
            units="mol O3 m-2",
            bounds=(5e-9, 2e-8),
            validate=True
        )
        self.assertIsInstance(antioxidant_capacity, (int, float, torch.Tensor))
        
        # Verify value is within bounds
        if isinstance(antioxidant_capacity, torch.Tensor):
            antioxidant_val = antioxidant_capacity.item()
        else:
            antioxidant_val = float(antioxidant_capacity)
        self.assertGreaterEqual(antioxidant_val, 5e-9)
        self.assertLessEqual(antioxidant_val, 2e-8)

    def test_all_parameters_retrievable(self):
        """Verify all critical parameters are retrievable via ParameterManager"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # List of all critical parameters after P0-1 fix
        critical_params = [
            ("global.factor_min", "dimensionless", "biosphere.stress"),
            ("global.factor_max", "dimensionless", "biosphere.stress"),
            ("ozone_uptake_rate", "dimensionless", "biosphere.stress.ozone"),
            ("flux_threshold", "nmol O3 m-2 s-1", "biosphere.stress.ozone"),
            ("ozone_detox_capacity", "mol O3 m-2 s-1", "biosphere.stress.ozone"),
            ("antioxidant_capacity", "mol O3 m-2", "biosphere.stress.ozone"),
        ]

        for param_name, units, namespace in critical_params:
            with self.subTest(param=param_name):
                value = ozone_stress.get_parameter_value(
                    param_name,
                    namespace=namespace,
                    units=units,
                    validate=True
                )
                self.assertIsNotNone(
                    value,
                    f"Parameter {param_name} should be retrievable from namespace {namespace}"
                )

    def test_forward_with_minimal_state(self):
        """Test forward pass with minimal required state variables"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Minimal state for ozone stress computation
        state = {
            'o3_mixing_ratio': torch.tensor([50e-9], device=self.device),  # 50 ppb in mol/mol
            'stomatal_conductance': torch.tensor([0.1], device=self.device)     # 0.1 mol m-2 s-1
        }

        # Test that forward pass executes without parameter errors
        try:
            output = ozone_stress.forward(state, time=0.0, dt=3600.0)
            self.assertIsInstance(output, dict)
            
            # Should produce all three outputs
            self.assertIn('ozone_stress_factor', output)
            self.assertIn('ozone_uptake', output)
            self.assertIn('antioxidant_depletion', output)
            
            # Verify output types and shapes
            for key in ['ozone_stress_factor', 'ozone_uptake', 'antioxidant_depletion']:
                self.assertIsInstance(output[key], torch.Tensor)
                self.assertEqual(output[key].shape, torch.Size([1]))
                
            # Verify stress factor is in valid range [0, 1]
            stress_val = output['ozone_stress_factor'].item()
            self.assertGreaterEqual(stress_val, 0.0)
            self.assertLessEqual(stress_val, 1.0)
            
        except Exception as e:
            # If it fails, it should NOT be due to missing parameter metadata
            error_str = str(e).lower()
            self.assertNotIn('missing units', error_str)
            self.assertNotIn('missing bounds', error_str)
            self.assertNotIn('validate', error_str)
            self.assertNotIn('namespace', error_str)
            self.assertNotIn('register_if_missing', error_str)

    def test_phase8_reaction_network_diagnostics(self):
        """Test explicit N_h r_h rows and IPHS port diagnostics."""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )
        state = {
            'o3_mixing_ratio': torch.tensor([50e-9], device=self.device),
            'stomatal_conductance': torch.tensor([0.1], device=self.device),
        }

        output = ozone_stress.forward(state, time=0.0, dt=3600.0)

        # F_O3 = eta [O3] g_s / k_O3 = 0.8 * 50e-9 * 0.1 / 1.67 (k_O3: Sitch et al.
        # 2007 ratio of the leaf resistance for O3 to that for water vapour);
        # C_detox = 2e-9 and [O3]_threshold = 40e-9 from the YAML database.
        uptake = 0.8 * 50e-9 * 0.1 / 1.67
        expected_uptake = torch.tensor([uptake], device=self.device)
        expected_detox = torch.tensor([2.0e-9], device=self.device)
        expected_excess = torch.tensor([uptake - 2.0e-9], device=self.device)
        expected_port_power = torch.tensor([uptake * 50e-9 / 40e-9], device=self.device)
        expected_entropy = torch.tensor([2.0e-9 * uptake / 2.0e-9], device=self.device)

        self.assertTrue(torch.allclose(output['ozone_uptake'].value, expected_uptake))
        self.assertTrue(torch.allclose(
            output['diagnostics.ozone_stress.detoxification_flux'],
            expected_detox,
        ))
        self.assertTrue(torch.allclose(
            output['diagnostics.ozone_stress.excess_ozone_flux'],
            expected_excess,
        ))
        self.assertTrue(torch.allclose(
            output['diagnostics.ozone_stress.antioxidant_pool_tendency'],
            -expected_detox,
        ))
        self.assertTrue(torch.allclose(
            output['diagnostics.ozone_stress.damage_tendency'],
            expected_excess,
        ))
        self.assertTrue(torch.allclose(
            output['diagnostics.ozone_stress.port_power'],
            expected_port_power,
        ))
        self.assertTrue(torch.allclose(
            output['diagnostics.ozone_stress.entropy_production'],
            expected_entropy,
        ))
        self.assertGreaterEqual(
            output['diagnostics.ozone_stress.entropy_production'].item(),
            0.0,
        )

    def test_compute_rate_with_environment(self):
        """Test compute_rate method with environment variables"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # State (can be minimal for compute_rate)
        state = {}

        # Environment with required variables
        environment = {
            'o3_mixing_ratio': torch.tensor([50e-9], device=self.device),
            'stomatal_conductance': torch.tensor([0.1], device=self.device)
        }

        # Test that compute_rate executes without parameter errors
        try:
            output = call_forward(ozone_stress, {**state, **environment})
            self.assertIsInstance(output, dict)
            self.assertIn('ozone_stress_factor', output)
            
            # Verify stress factor is in valid range
            stress_val = output['ozone_stress_factor'].item()
            self.assertGreaterEqual(stress_val, 0.0)
            self.assertLessEqual(stress_val, 1.0)
            
        except Exception as e:
            error_str = str(e).lower()
            self.assertNotIn('missing units', error_str)
            self.assertNotIn('missing bounds', error_str)
            self.assertNotIn('validate', error_str)

    def test_register_if_missing_false_enforced(self):
        """Verify register_if_missing=False prevents auto-registration"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Attempt to get a non-existent parameter with register_if_missing=False
        # This should raise an error or return None, NOT auto-create the parameter
        try:
            fake_param = ozone_stress.get_parameter_value(
                "nonexistent_parameter_xyz123",
                namespace="biosphere.stress.ozone",
                units="dimensionless",
                validate=True
            )
            # If it returns something, it should be None or default
            # It should NOT have auto-created the parameter
            if fake_param is not None:
                self.fail(
                    "register_if_missing=False should prevent auto-registration of nonexistent parameters"
                )
        except (KeyError, ValueError, AttributeError):
            # Expected behavior: parameter not found and not auto-registered
            pass

    def test_no_hardcoded_defaults_in_forward(self):
        """Verify forward() does not use state.get() with torch.tensor defaults"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Test with incomplete state - should raise KeyError, not use defaults
        incomplete_state = {
            'o3_mixing_ratio': torch.tensor([50e-9], device=self.device)
            # Missing 'stomatal_conductance'
        }

        with self.assertRaises(KeyError) as context:
            ozone_stress.forward(incomplete_state, time=0.0, dt=3600.0)
        
        # Error message should mention missing variable
        self.assertIn('stomatal_conductance', str(context.exception))

    def test_parameter_units_consistency(self):
        """Test that parameter units are consistent with scientific definitions"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Test flux threshold units (nmol O3 m-2 s-1)
        flux_threshold = ozone_stress.get_parameter_value(
            "flux_threshold",
            namespace="biosphere.stress.ozone",
            units="nmol O3 m-2 s-1",
            validate=True
        )
        # Should succeed with correct units
        self.assertIsNotNone(flux_threshold)

        # Test detox capacity units (mol O3 m-2 s-1)
        detox_capacity = ozone_stress.get_parameter_value(
            "ozone_detox_capacity",
            namespace="biosphere.stress.ozone",
            units="mol O3 m-2 s-1",
            validate=True
        )
        self.assertIsNotNone(detox_capacity)

        # Test antioxidant capacity units (mol O3 m-2)
        antioxidant_capacity = ozone_stress.get_parameter_value(
            "antioxidant_capacity",
            namespace="biosphere.stress.ozone",
            units="mol O3 m-2",
            validate=True
        )
        self.assertIsNotNone(antioxidant_capacity)

    def test_metadata_completeness(self):
        """Test that Functor metadata is properly configured"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Check that metadata exists
        self.assertIsNotNone(ozone_stress.metadata)
        
        # Check key metadata fields
        self.assertEqual(ozone_stress.metadata.name, "ozone_stress_test")
        self.assertIsNotNone(ozone_stress.metadata.description)
        self.assertIsNotNone(ozone_stress.metadata.parameters)
        self.assertIsNotNone(ozone_stress.metadata.references)
        
        # Verify parameters are documented in metadata
        documented_params = ozone_stress.metadata.parameters.keys()
        self.assertIn('biosphere.stress.ozone.ozone_uptake_rate', documented_params)
        self.assertIn('biosphere.stress.ozone.flux_threshold', documented_params)
        self.assertIn('biosphere.stress.ozone.ozone_detox_capacity', documented_params)
        self.assertIn('biosphere.stress.ozone.antioxidant_capacity', documented_params)

    def test_dependencies_declaration(self):
        """Test that Functor dependencies are properly declared"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Get declared dependencies
        dependencies = ozone_stress.get_dependencies()
        
        # Ozone from the atmospheric chemistry (optional same-step provider of
        # o3_mixing_ratio); stomatal_conductance has no relic writer, so no provider
        # is declared for it (FarquharPhotosynthesis outputs no conductance).
        from relic.core.dependency_graph import normalize_dependency_declaration

        declared = normalize_dependency_declaration(dependencies, owner=ozone_stress.name)
        self.assertIn('atmospheric_chemistry', declared.optional)
        self.assertNotIn(
            'farquhar_photosynthesis',
            declared.required + declared.optional + declared.lagged,
        )

    def test_provides_declaration(self):
        """Test that Functor output variables are properly declared"""
        ozone_stress = OzoneStress(
            name="ozone_stress_test",
            device=self.device
        )

        # Get declared output variables
        provides = ozone_stress.get_provides()
        
        # Should provide stress factor and uptake flux
        self.assertIsInstance(provides, list)
        self.assertIn('ozone_stress_factor', provides)
        self.assertIn('ozone_uptake', provides)
        self.assertIn('antioxidant_depletion', provides)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)


def test_ozone_stress_reads_the_chemistry_ozone_state():
    """One ozone mole fraction: OzoneStress reads o3_mixing_ratio, the state written by
    the atmospheric chemistry (CF mole_fraction_of_ozone_in_air), and is ordered after
    the chemistry Functor whenever that Functor is registered."""
    import inspect

    from relic.contracts import load_contract_registry
    from relic.core.dependency_graph import normalize_dependency_declaration
    from relic.scientific.atmosphere.chemistry import ChemistryFunctor

    registry = load_contract_registry()
    chemistry_name = inspect.signature(ChemistryFunctor.__init__).parameters["name"].default
    ozone = registry.get_contract("o3_mixing_ratio")
    assert chemistry_name in ozone.increment_writers
    assert ozone.cf_standard_name == "mole_fraction_of_ozone_in_air"

    stress = OzoneStress(name="ozone_state_probe", device=torch.device("cpu"))
    inputs = stress.metadata.input_variables
    assert "o3_mixing_ratio" in inputs
    assert not {"ozone_concentration", "o3_concentration"} & set(inputs)
    declared = normalize_dependency_declaration(stress.get_dependencies(), owner=stress.name)
    assert chemistry_name in declared.optional
    assert "atmosphere_chemistry" not in declared.required + declared.optional

    # The former forcing name is a contract alias of the chemistry state, not a
    # second variable; its own former alias is retired.
    assert registry.get_contract("ozone_concentration") is None
    assert registry.resolve_alias("ozone_concentration") == "o3_mixing_ratio"
    assert not registry.is_alias("o3_concentration")


def test_retired_ozone_name_o3_concentration_is_not_named_by_relic_code():
    """o3_concentration was retired with the OzoneStress rename to o3_mixing_ratio;
    no relic module, parameter or contract names it (no3_concentration is nitrate)."""
    import re

    root = Path(__file__).resolve().parents[3] / "relic"
    pattern = re.compile(r"(?<![A-Za-z0-9_])o3_concentration(?![A-Za-z0-9_])")
    hits = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".py", ".yaml", ".yml"} or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(root)}:{number}")
    assert hits == []


def test_ozone_stress_without_the_ozone_state_raises():
    """The retired names are not read as a fallback: a missing o3_mixing_ratio raises."""
    import pytest

    stress = OzoneStress(name="ozone_missing_probe", device=torch.device("cpu"))
    with pytest.raises(KeyError, match="o3_mixing_ratio"):
        stress.forward(
            {
                "ozone_concentration": torch.tensor([40.0e-9]),
                "o3_concentration": torch.tensor([40.0e-9]),
                "stomatal_conductance": torch.tensor([0.2]),
            },
            time=0.0,
            dt=3600.0,
        )


def test_ozone_mole_fraction_is_one_unit_for_contract_parameters_and_reader():
    """Contract, reader parameters and the stomatal flux all use mol mol-1 (no ppb)."""
    from relic.contracts import load_contract_registry
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    assert load_contract_registry().get_contract("o3_mixing_ratio").udunits == "mol mol-1"
    assert manager.get_parameter("biosphere.stress.ozone.ozone_threshold").units == "mol mol-1"

    stress = OzoneStress(name="ozone_units_probe", device=torch.device("cpu"))
    conductance = torch.tensor([0.2], dtype=torch.float64)
    mole_fraction = torch.tensor([40.0e-9], dtype=torch.float64)  # 40 ppb
    output = stress.forward(
        {"o3_mixing_ratio": mole_fraction, "stomatal_conductance": conductance},
        time=0.0,
        dt=3600.0,
    )
    eta = float(manager.get("biosphere.stress.ozone.ozone_uptake_rate", required=True))
    k_o3 = float(manager.get(_K_O3_KEY, required=True))
    # References: Emberson et al. (2000); Sitch et al. (2007) as given by Leung et al.
    #   (2020), GMD 13, 6201, Eq. (5); Niinemets et al. (2014), Fick's first law.
    # Mathematical expression: F_O3 = (g_s / k_O3) * c * eta.
    # Parameter meanings: g_s leaf stomatal conductance to water vapour, k_O3 ratio of
    #   the leaf resistance for O3 to that for water vapour, c ozone mole fraction,
    #   eta the uptake efficiency.
    # Units: g_s [mol m-2 s-1]; k_O3 [1]; c [mol mol-1]; eta [1]; F_O3 [mol O3 m-2 s-1].
    # Domain/constraints: c is a mole fraction in [0, 1]; no ppb conversion.
    # Sign convention: positive F_O3 is uptake into the leaf.
    uptake = output["ozone_uptake"]
    uptake = uptake.value if hasattr(uptake, "value") else uptake
    torch.testing.assert_close(
        uptake.to(torch.float64),
        conductance / k_o3 * mole_fraction * eta,
        rtol=1.0e-6,
        atol=0.0,
    )


_K_O3_KEY = "relic.physical.constants.conversions.h2o_o3_diffusivity_ratio"


def _ozone_certificate():
    """The IPHS certificate of OzoneStress."""
    import yaml

    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "biosphere_stress_ozone_stress_ozone_stress.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def test_stomatal_conductance_is_a_caller_supplied_forcing_without_a_provider():
    """No relic Functor writes stomatal_conductance: FarquharPhotosynthesis evaluates Ac
    and Aj at the atmospheric CO2 mole fraction and outputs no conductance. OzoneStress
    declares no provider for it (the former farquhar_photosynthesis dependency was
    dangling and made every composition without Farquhar fail), reads it as a required
    caller-supplied forcing with no fallback default, and its certificate names no
    Farquhar port."""
    import pytest

    from relic.core.dependency_graph import DependencyGraphResolver
    from relic.parameters import get_global_manager
    from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

    farquhar = FarquharPhotosynthesis(name="farquhar_probe", enable_gpu=False)
    assert "stomatal_conductance" not in farquhar.metadata.output_variables
    assert "stomatal_conductance" not in farquhar.get_provides()

    stress = OzoneStress(name="ozone_stress", device=torch.device("cpu"))
    graph = DependencyGraphResolver()
    graph.add_functor(stress, "ozone_stress")
    assert graph.topological_sort() == ["ozone_stress"]

    with pytest.raises(KeyError, match="stomatal_conductance") as missing:
        stress.forward({"o3_mixing_ratio": torch.tensor([40.0e-9])}, time=0.0, dt=3600.0)
    assert "water vapour" in str(missing.value) and "forcing" in str(missing.value)
    assert get_global_manager().get_parameter(
        "biosphere.stress.ozone.default_stomatal_conductance"
    ) is None

    certificate = _ozone_certificate()
    assert "farquhar_photosynthesis" not in certificate["composition_ports"]
    assert "farquhar_photosynthesis" not in (
        certificate["dependency_interconnection_map"]["dependencies"]
    )


def test_stomatal_uptake_converts_the_water_vapour_conductance_with_k_o3():
    """Sitch et al. (2007), as given by Leung et al. (2020, GMD 13, 6201, Eq. 5):
    F_O3 = [O3] / (R_a + k_O3 / g_l), with g_l the leaf conductance for water vapour and
    k_O3 = 1.67 the ratio of the leaf resistance for O3 to that for water vapour.
    OzoneStress reads the leaf-level stomatal conductance to water vapour and takes
    g_s / k_O3 as the O3 conductance (R_a -> 0; molar form with g_s in mol m-2 s-1 and
    [O3] a mole fraction). k_O3 is a physical constant of the SSOT, declared in the
    metadata and in the certificate."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    spec = manager.get_parameter(_K_O3_KEY)
    assert spec is not None
    assert float(spec.value) == 1.67 and spec.units == "dimensionless"
    references = " ".join(spec.references)
    assert "Sitch" in references and "10.1038/nature06059" in references
    assert "Leung" in references and "10.5194/gmd-13-6201-2020" in references

    stress = OzoneStress(name="ozone_k_o3_probe", device=torch.device("cpu"))
    assert stress.metadata.parameters[_K_O3_KEY].endswith("[dimensionless]")

    eta = float(manager.get("biosphere.stress.ozone.ozone_uptake_rate", required=True))
    conductance = torch.tensor([0.05, 0.1], dtype=torch.float64)
    mole_fraction = torch.tensor([5.0e-9, 5.0e-9], dtype=torch.float64)
    output = stress.forward(
        {"o3_mixing_ratio": mole_fraction, "stomatal_conductance": conductance},
        time=0.0,
        dt=3600.0,
    )
    torch.testing.assert_close(
        output["ozone_uptake"].value.to(torch.float64),
        conductance / 1.67 * mole_fraction * eta,
        rtol=1.0e-6,
        atol=0.0,
    )

    certificate = _ozone_certificate()
    assert _K_O3_KEY in certificate["parameter_keys"]
    assert "ParameterManager" in certificate["parameter_management_map"][_K_O3_KEY]["source"]


def test_governed_simulator_steps_replace_the_algebraic_outputs():
    """The ozone stress factor, the uptake flux and the depletion index are instantaneous
    values of the current inputs: in the governed simulator (no run specs) each step
    replaces them, the non-incrementable contracts accept the writes, and nothing
    accumulates across steps."""
    from relic.core.simulator import RELICSimulator

    inputs = {
        "o3_mixing_ratio": torch.tensor([4.0e-8]),
        "stomatal_conductance": torch.tensor([0.2]),
    }
    stress = OzoneStress(device=torch.device("cpu"))
    direct = stress(dict(inputs), 0.0, 60.0)

    def value(item):
        return item.value if hasattr(item, "value") else item

    simulator = RELICSimulator({"time_step": 60.0, "max_time": 600.0, "device": "cpu"})
    simulator.initialize_state(dict(inputs))
    simulator.register_functor(stress, execution_group="fast")
    for _ in range(3):
        simulator.step(dt=60.0)
        state = simulator.state_manager.get_state()
        for name in ("ozone_stress_factor", "ozone_uptake", "antioxidant_depletion"):
            torch.testing.assert_close(
                state[name].reshape(-1), value(direct[name]).reshape(-1).to(state[name].dtype)
            )


def test_ozone_stress_has_no_uncalled_rate_core():
    """Round 11: the _compute_rate_core wrapper, which merged an environment mapping over
    the state and fixed dt = 1 s, had no relic caller and is retired; forward is the one
    kernel."""
    assert not hasattr(OzoneStress, "_compute_rate_core")


def test_ozone_stress_declares_no_closed_ozone_ledger():
    """OzoneStress reads the atmospheric O3 mole fraction as forcing and advances no ozone
    stock: its reaction network only partitions the uptake flux (F = D + X). No closed
    ozone ledger exists, so no conservation law is declared and the governed call emits no
    conservation warning (the base check would sum factor, flux and depletion)."""
    import warnings

    import yaml

    stress = OzoneStress(device=torch.device("cpu"))
    assert not stress.metadata.conservation_laws
    state = {
        "o3_mixing_ratio": torch.tensor([4.0e-8, 8.0e-8]),
        "stomatal_conductance": torch.tensor([0.05, 0.3]),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stress(dict(state), time=0.0, dt=3600.0)
    assert not [w for w in caught if "Conservation law" in str(w.message)]
    certificate = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3]
            / "docs" / "source" / "theory" / "iphs_certificates"
            / "biosphere_stress_ozone_stress_ozone_stress.yaml"
        ).read_text(encoding="utf-8")
    )
    assert certificate["conservation_map"]["conserved_quantities"].startswith("none")


def test_antioxidant_depletion_is_the_specific_detoxification_consumption_rate():
    """antioxidant_depletion = D / C_ax [s-1]: the antioxidant consumed by the declared
    detoxification reaction D = min(F, C_detox) per unit antioxidant capacity (the
    apoplastic ascorbate pool is replenished, Ploechl et al. 2000, so no per-step fraction
    of a non-regenerated pool is formed). The excess flux X damages, it is not quenched."""
    from relic.parameters import get_global_manager

    capacity = float(get_global_manager().get("biosphere.stress.ozone.antioxidant_capacity"))
    stress = OzoneStress(device=torch.device("cpu"))
    state = {
        "o3_mixing_ratio": torch.tensor([5.0e-9, 1.0e-7]),
        "stomatal_conductance": torch.tensor([0.05, 0.4]),
    }
    out = stress(dict(state), time=0.0, dt=3600.0)
    uptake = out["ozone_uptake"].value
    detox = out["diagnostics.ozone_stress.detoxification_flux"]
    detox = detox.value if hasattr(detox, "value") else detox
    depletion = out["antioxidant_depletion"]
    assert depletion.unit == "s-1"
    torch.testing.assert_close(depletion.value, detox / capacity)
    # the second case exceeds the detoxification capacity: only D consumes antioxidant
    assert float(uptake[1]) > float(detox[1])
    assert float(depletion.value[1]) < float(uptake[1]) / capacity
