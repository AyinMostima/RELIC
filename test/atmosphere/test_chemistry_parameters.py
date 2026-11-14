"""
Test suite for ChemistryFunctor parameter management

Tests L1 unit validation for active gas-phase chemistry parameters.
Verifies canonical parameter calls with units, bounds, and validate=True.

Fixed parameters:
1. j_no2_irradiance_linear_coefficient (Trebs et al. 2009 B1)
2. uv_a_surface_albedo (Trebs et al. 2009 alpha)
3. k_no_o3_298 (Line 396)
4. k_no_o3_activation_energy (Line 398)
5. k_isoprene_oh_298 (Line 405)
6. k_isoprene_o3_298 (Line 414)
7. k_isoprene_oh_activation_energy (Line 407)
8. k_isoprene_o3_activation_energy (Line 416)
9. k_oh_co_298 (Line 502)
10. k_oh_ch4_298 (Line 507)
11. k_oh_ch4_activation_energy (Line 508)
12. k_oh_no2_298 (Line 512)
13. minimum_concentration

References:
- Phase 6 Step 5: chemistry.py parameter fixes
- FunctorBase.get_parameter_value() API
"""

import unittest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.atmosphere.chemistry import ChemistryFunctor


CERTIFICATE_PATH = (
    project_root
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "atmosphere_chemistry_chemistry_functor.yaml"
)

EXPECTED_INPUT_VARIABLES = (
    "air_temperature",
    "atmospheric_pressure",
    "shortwave_down",
    "cos_zenith",
    "no_mixing_ratio",
    "no2_mixing_ratio",
    "o3_mixing_ratio",
    "isoprene_mixing_ratio",
    "co_mixing_ratio",
    "ch4_mixing_ratio",
    "specific_humidity",
)

EXPECTED_OUTPUT_VARIABLES = (
    "no_mixing_ratio",
    "no2_mixing_ratio",
    "o3_mixing_ratio",
    "isoprene_mixing_ratio",
    "diagnostics.chemistry.photolysis_rate_no2",
    "diagnostics.chemistry.isoprene_oxidation_rate",
    "diagnostics.chemistry.reaction_no_o3_rate",
    "diagnostics.chemistry.photolysis_no2_rate",
    "diagnostics.chemistry.rate_constant_no_o3",
    "diagnostics.chemistry.shortwave_radiation_used",
    "diagnostics.chemistry.oh_concentration",
    "diagnostics.chemistry.air_number_density",
    "diagnostics.chemistry.ro2_no_conversion_rate",
    "diagnostics.chemistry.ro2_no_branching_fraction",
    "diagnostics.chemistry.ox_net_chemical_tendency",
    "diagnostics.chemistry.carbon_oxidized_rate",
)

EXPECTED_UNIT_MAP = {
    "air_temperature": "K",
    "atmospheric_pressure": "Pa",
    "shortwave_down": "W m-2",
    "cos_zenith": "dimensionless",
    "no_mixing_ratio": "mol mol-1",
    "no2_mixing_ratio": "mol mol-1",
    "o3_mixing_ratio": "mol mol-1",
    "isoprene_mixing_ratio": "mol mol-1",
    "co_mixing_ratio": "mol mol-1",
    "ch4_mixing_ratio": "mol mol-1",
    "specific_humidity": "kg kg-1",
    "diagnostics.chemistry.photolysis_rate_no2": "s-1",
    "diagnostics.chemistry.isoprene_oxidation_rate": "mol mol-1 s-1",
    "diagnostics.chemistry.reaction_no_o3_rate": "mol mol-1 s-1",
    "diagnostics.chemistry.photolysis_no2_rate": "mol mol-1 s-1",
    "diagnostics.chemistry.rate_constant_no_o3": "cm3 molecule-1 s-1",
    "diagnostics.chemistry.shortwave_radiation_used": "W m-2",
    "diagnostics.chemistry.oh_concentration": "molecule cm-3",
    "diagnostics.chemistry.air_number_density": "molecule cm-3",
    "diagnostics.chemistry.ro2_no_conversion_rate": "mol mol-1 s-1",
    "diagnostics.chemistry.ro2_no_branching_fraction": "dimensionless",
    "diagnostics.chemistry.ox_net_chemical_tendency": "mol mol-1 s-1",
    "diagnostics.chemistry.carbon_oxidized_rate": "mol mol-1 s-1",
}


class TestChemistryFunctorParameters(unittest.TestCase):
    """Test ChemistryFunctor parameter management"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')

    def test_import_module(self):
        """Test that ChemistryFunctor can be imported"""
        self.assertIsNotNone(ChemistryFunctor)

    def test_module_initialization(self):
        """Test basic module initialization with default parameters"""
        chemistry = ChemistryFunctor(
            name="chemistry_test",
            device=self.device, physical_boundary="open_mechanism"
        )
        self.assertIsNotNone(chemistry)
        self.assertIsInstance(chemistry, ChemistryFunctor)

    def test_parameter_retrieval(self):
        """Test photolysis and reaction rate parameter retrieval"""
        chemistry = ChemistryFunctor(
            name="chemistry_test",
            device=self.device, physical_boundary="open_mechanism"
        )

        # Trebs et al. (2009) B1 of the J_NO2 irradiance relation
        j_no2 = chemistry.get_parameter_value(
            "j_no2_irradiance_linear_coefficient",
            namespace="atmosphere.chemistry",
            units="W-1 m2 s-1",
            bounds=(1e-6, 1e-4),
            validate=True
        )
        self.assertIsInstance(j_no2, (int, float, torch.Tensor))

        # Test k_no_o3_298 (Line 396 fix)
        k_no_o3 = chemistry.get_parameter_value(
            "k_no_o3_298",
            namespace="atmosphere.chemistry",
            units="cm3 molecule-1 s-1",
            bounds=(1e-15, 1e-13),
            validate=True
        )
        self.assertIsInstance(k_no_o3, (int, float, torch.Tensor))

        # Test k_isoprene_oh_298 (Line 405 fix)
        k_isop_oh = chemistry.get_parameter_value(
            "k_isoprene_oh_298",
            namespace="atmosphere.chemistry",
            units="cm3 molecule-1 s-1",
            bounds=(1e-11, 1e-9),
            validate=True
        )
        self.assertIsInstance(k_isop_oh, (int, float, torch.Tensor))

    def test_forward_minimal(self):
        """Test forward pass with minimal state input"""
        chemistry = ChemistryFunctor(
            name="chemistry_test",
            device=self.device, physical_boundary="open_mechanism"
        )

        # Minimal state for chemistry computation
        state = {
            'air_temperature': torch.tensor([298.0]),
            'no_mixing_ratio': torch.tensor([1e-9]),
            'no2_mixing_ratio': torch.tensor([1e-8]),
            'o3_mixing_ratio': torch.tensor([4e-8]),
            'atmospheric_pressure': torch.tensor([101325.0]),  # Pa
            'co_mixing_ratio': torch.tensor([150e-9]),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor([1.8e-6]),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor([0.01]),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor([1.0]),  # overhead sun
            'shortwave_down': torch.tensor([300.0]),
        }

        # The instantaneous explicit-Euler positivity limit is about 359 s.
        output = chemistry.forward(state, time=0.0, dt=300.0)
        self.assertIsInstance(output, dict)
        for key in ('no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio'):
            item = output[key]
            self.assertIsInstance(item, UpdateItem)
            self.assertEqual(item.mode, UpdateMode.ACCUMULATE)
            self.assertEqual(item.unit, 'mol mol-1')
            self.assertTrue(item.per_timestep)
            self.assertEqual(item.source_functor, chemistry.name)
            self.assertTrue(torch.isfinite(item.value).all())
            self.assertTrue(torch.all(state[key] + item.value >= 0.0))
        self.assertNotIn('delta_no', output)
        self.assertNotIn('delta_no2', output)
        self.assertNotIn('delta_o3', output)

    def test_shortwave_down_is_the_only_required_radiation_proxy(self):
        """Reject obsolete radiation alternatives instead of using fallbacks."""
        chemistry = ChemistryFunctor(name="chemistry_test", device=self.device, physical_boundary="open_mechanism")
        base_state = {
            "air_temperature": torch.tensor(298.0),
            "no_mixing_ratio": torch.tensor(1.0e-9),
            "no2_mixing_ratio": torch.tensor(1.0e-8),
            "o3_mixing_ratio": torch.tensor(4.0e-8),
            "atmospheric_pressure": torch.tensor(101325.0),  # Pa
            "co_mixing_ratio": torch.tensor(150e-9),  # OH sink, 150 ppb
            "ch4_mixing_ratio": torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            "specific_humidity": torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            "cos_zenith": torch.tensor(1.0),  # overhead sun
        }

        for obsolete_name, obsolete_value in (
            ("net_radiation", torch.tensor(300.0)),
            ("photosynthetically_active_radiation", torch.tensor(1380.0)),
        ):
            with self.subTest(obsolete_name=obsolete_name):
                state = dict(base_state)
                state[obsolete_name] = obsolete_value
                with self.assertRaisesRegex(ValueError, "shortwave_down is required"):
                    chemistry.forward(state, time=0.0, dt=60.0)

        self.assertNotIn("shortwave_down", chemistry.optional_input_variables)
        self.assertNotIn("net_radiation", chemistry.metadata.input_variables)
        self.assertNotIn(
            "photosynthetically_active_radiation",
            chemistry.metadata.input_variables,
        )

    def test_shortwave_diagnostic_preserves_the_canonical_input(self):
        """Expose the exact canonical radiation proxy used by chemistry."""
        chemistry = ChemistryFunctor(name="chemistry_test", device=self.device, physical_boundary="open_mechanism")
        shortwave_down = torch.tensor(300.0)
        state = {
            "air_temperature": torch.tensor(298.0),
            "shortwave_down": shortwave_down,
            "no_mixing_ratio": torch.tensor(1.0e-9),
            "no2_mixing_ratio": torch.tensor(1.0e-8),
            "o3_mixing_ratio": torch.tensor(4.0e-8),
            "atmospheric_pressure": torch.tensor(101325.0),  # Pa
            "co_mixing_ratio": torch.tensor(150e-9),  # OH sink, 150 ppb
            "ch4_mixing_ratio": torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            "specific_humidity": torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            "cos_zenith": torch.tensor(1.0),  # overhead sun
        }

        output = chemistry.forward(state, time=0.0, dt=60.0)

        diagnostic = output["diagnostics.chemistry.shortwave_radiation_used"]
        self.assertIsInstance(diagnostic, UpdateItem)
        self.assertEqual(diagnostic.mode, UpdateMode.REPLACE)
        self.assertEqual(diagnostic.unit, "W m-2")
        self.assertTrue(diagnostic.per_timestep)
        self.assertEqual(diagnostic.source_functor, chemistry.name)
        self.assertTrue(torch.equal(diagnostic.value, shortwave_down))

    def test_no_hardcoded_parameters(self):
        """Verify no hardcoded physical constants bypass parameter system"""
        chemistry = ChemistryFunctor(
            name="chemistry_test",
            device=self.device, physical_boundary="open_mechanism"
        )

        # Active gas-phase parameters should be retrievable via the parameter system.
        critical_params = [
            ("j_no2_irradiance_linear_coefficient", "W-1 m2 s-1", "atmosphere.chemistry"),
            ("k_no_o3_298", "cm3 molecule-1 s-1", "atmosphere.chemistry"),
            ("k_isoprene_oh_298", "cm3 molecule-1 s-1", "atmosphere.chemistry"),
            ("k_isoprene_o3_298", "cm3 molecule-1 s-1", "atmosphere.chemistry"),
            ("k_oh_co_298", "cm3 molecule-1 s-1", "atmosphere.chemistry"),
        ]

        for param_name, units, namespace in critical_params:
            value = chemistry.get_parameter_value(
                param_name,
                namespace=namespace,
                units=units,
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param_name} should be retrievable")

    def test_parameter_bounds_validation(self):
        """Test that bounds validation works for all fixed parameters"""
        chemistry = ChemistryFunctor(
            name="chemistry_test",
            device=self.device, physical_boundary="open_mechanism"
        )

        # Test the J_NO2 irradiance coefficient bounds
        j_no2 = chemistry.get_parameter_value(
            "j_no2_irradiance_linear_coefficient",
            namespace="atmosphere.chemistry",
            units="W-1 m2 s-1",
            bounds=(1e-6, 1e-4),
            validate=True
        )
        if isinstance(j_no2, torch.Tensor):
            j_no2_val = j_no2.item()
        else:
            j_no2_val = float(j_no2)
        self.assertGreaterEqual(j_no2_val, 0.0)
        self.assertLessEqual(j_no2_val, 1.0)  # Physical upper limit

        # Test reaction rate constant bounds
        k_no_o3 = chemistry.get_parameter_value(
            "k_no_o3_298",
            namespace="atmosphere.chemistry",
            units="cm3 molecule-1 s-1",
            bounds=(1e-15, 1e-13),
            validate=True
        )
        if isinstance(k_no_o3, torch.Tensor):
            k_val = k_no_o3.item()
        else:
            k_val = float(k_no_o3)
        self.assertGreaterEqual(k_val, 0.0)  # Physical constraint

    def test_parameter_units(self):
        """Verify units are correct (CF-1.8 compliant)"""
        chemistry = ChemistryFunctor(
            name="chemistry_test",
            device=self.device, physical_boundary="open_mechanism"
        )

        # Test the J_NO2 irradiance coefficient units (W-1 m2 s-1)
        j_no2 = chemistry.get_parameter_value(
            "j_no2_irradiance_linear_coefficient",
            units="W-1 m2 s-1",
            namespace="atmosphere.chemistry",
            validate=True
        )
        self.assertIsNotNone(j_no2)

        # Test activation energy units (J mol^-1)
        ea_no_o3 = chemistry.get_parameter_value(
            "k_no_o3_activation_energy",
            units="J mol-1",
            namespace="atmosphere.chemistry",
            validate=True
        )
        self.assertIsNotNone(ea_no_o3)

    def test_parameter_namespace(self):
        """Verify namespace correctness"""
        chemistry = ChemistryFunctor(
            name="chemistry_test",
            device=self.device, physical_boundary="open_mechanism"
        )

        # All chemistry parameters should be in atmosphere.chemistry namespace
        params_to_test = [
            "j_no2_irradiance_linear_coefficient",
            "k_no_o3_298",
            "k_isoprene_oh_298",
            "minimum_concentration",
        ]

        for param in params_to_test:
            value = chemistry.get_parameter_value(
                param,
                namespace="atmosphere.chemistry",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in atmosphere.chemistry namespace")

    def test_phase8_iphs_structure_hooks_are_declared(self):
        """Verify chemistry exposes Phase 8 field structure hooks."""
        chemistry = ChemistryFunctor(name="chemistry_test", device=self.device, physical_boundary="open_mechanism")

        self.assertIsInstance(chemistry, IPHSDECFunctor)
        forms = chemistry.declared_forms()
        hodge_terms = chemistry.build_hodge_terms()
        dissipative = chemistry.dissipative_structure({})
        ports = chemistry.ports()

        self.assertEqual(
            {form.semantic_name for form in forms},
            {
                "atmospheric_species_mixing_ratio",
                "atmospheric_chemistry_tendency",
            },
        )
        self.assertIn(
            "atmosphere.chemistry.j_no2_irradiance_linear_coefficient",
            chemistry.required_parameters(),
        )
        self.assertEqual(len(hodge_terms), 1)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0))
        self.assertEqual(
            {port.name for port in ports},
            {"atmospheric_radiation_photolysis_port"},
        )
        self.assertEqual(
            ports[0].flow_var,
            "diagnostics.chemistry.photolysis_rate_no2",
        )
        self.assertNotIn(
            "atmosphere.radiation.canopy.ppfd_conversion",
            chemistry.required_parameters(),
        )

    def test_phase8_certificate_level3_evidence_is_valid(self):
        """Verify chemistry remains level 3 with bounded evidence."""
        certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
        audit = certificate.validate()

        self.assertTrue(audit["passed"])
        self.assertEqual(certificate.canonical_role, "field")
        self.assertEqual(certificate.iphs_level, 3)
        self.assertIn("level_3_limit", certificate.iphs_level_evidence)
        self.assertNotIn("storage_functional", certificate.iphs_level_evidence)

    def test_phase8_certificate_matches_runtime_contract_exactly(self):
        """Verify the certificate has no stale chemistry aliases."""
        chemistry = ChemistryFunctor(name="chemistry_test", device=self.device, physical_boundary="open_mechanism")
        certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)

        self.assertEqual(certificate.input_variables, EXPECTED_INPUT_VARIABLES)
        self.assertEqual(
            certificate.input_variables,
            tuple(chemistry.metadata.input_variables),
        )
        self.assertEqual(certificate.output_variables, EXPECTED_OUTPUT_VARIABLES)
        self.assertEqual(
            certificate.output_variables,
            tuple(chemistry.metadata.output_variables),
        )
        self.assertEqual(certificate.parameter_keys, chemistry.required_parameters())
        self.assertEqual(dict(certificate.unit_map), EXPECTED_UNIT_MAP)
        self.assertEqual(
            tuple(certificate.interface_map["legacy_inputs"]),
            EXPECTED_INPUT_VARIABLES,
        )
        self.assertEqual(
            tuple(certificate.interface_map["iphs_inputs"]),
            EXPECTED_INPUT_VARIABLES,
        )
        self.assertEqual(
            tuple(certificate.interface_map["legacy_outputs"]),
            EXPECTED_OUTPUT_VARIABLES,
        )
        self.assertEqual(
            tuple(certificate.interface_map["iphs_outputs"]),
            EXPECTED_OUTPUT_VARIABLES,
        )
        self.assertFalse(certificate.runtime_replacement_map["runtime_uses_iphs"])
        self.assertFalse(certificate.runtime_replacement_map["legacy_removed"])
        self.assertTrue(certificate.runtime_replacement_map["no_legacy_mode"])
        self.assertEqual(
            certificate.theory_document,
            "docs\\source\\theory\\chemistry.md",
        )

    def test_closed_constant_volume_certificate_matches_the_selection_contract(self):
        """The closed-box Level 5 certificate binds to a governed, committed step."""
        import torch as _torch
        from relic.core.simulator import RELICSimulator

        closed = ChemistryFunctor(
            name="atmospheric_chemistry",
            physical_boundary="closed_constant_volume",
            device=self.device,
        )
        certificate = load_iphs_equivalence_certificate(
            CERTIFICATE_PATH.with_name(
                "atmosphere_chemistry_chemistry_functor_closed_constant_volume.yaml"
            )
        )
        self.assertEqual(certificate.iphs_level, 5)
        self.assertEqual(certificate.formulation, "generic")
        self.assertEqual(
            certificate.process_id,
            "relic.scientific.atmosphere.chemistry.ChemistryFunctor",
        )
        self.assertEqual(
            certificate.interface_map["constructor_selection"],
            'ChemistryFunctor(physical_boundary="closed_constant_volume")',
        )
        self.assertEqual(certificate.input_variables, tuple(closed.metadata.input_variables))
        self.assertEqual(certificate.output_variables, tuple(closed.metadata.output_variables))
        self.assertEqual(certificate.parameter_keys, closed.required_parameters())
        self.assertEqual(tuple(certificate.interface_map["iphs_inputs"]), certificate.input_variables)
        self.assertEqual(tuple(certificate.interface_map["iphs_outputs"]), certificate.output_variables)
        self.assertEqual(set(certificate.unit_map), set(certificate.input_variables) | set(certificate.output_variables))
        self.assertEqual(certificate.composition_ports, ())
        self.assertEqual(closed.ports(), ())
        self.assertEqual(closed.get_dependencies(), [])
        self.assertEqual(
            certificate.strict_tolerance_parameter_keys(),
            ("validation.iphs.runtime_point_tolerance",),
        )

        # Descriptive validation alone must stay process-unbound.
        descriptive = certificate.validate()
        self.assertFalse(descriptive["passed"])
        self.assertEqual(descriptive["runtime_binding"], "descriptive")
        self.assertTrue(
            any("process_bound_runtime_unavailable" in issue for issue in descriptive["issues"]),
            descriptive["issues"],
        )

        # One governed, committed step of the real process supplies the witness.
        amounts = _torch.tensor([1.0e-8, 1.0e-8, 1.0e-8, 8.0e-3], dtype=_torch.float64)
        energy = (amounts * closed.standard_molar_properties(
            _torch.tensor(298.15, dtype=_torch.float64)
        )["u"]).sum().detach()
        state = {
            "box_no_amount": amounts[0].clone(),
            "box_no2_amount": amounts[1].clone(),
            "box_o3_amount": amounts[2].clone(),
            "box_o2_amount": amounts[3].clone(),
            "box_internal_energy": energy.clone(),
            "box_volume": _torch.tensor(1.0e-3, dtype=_torch.float64),
        }
        simulator = RELICSimulator({
            "max_time": 1.0, "time_step": 1.0, "device": "cpu",
            "conservation_checks": True, "strict_dependencies": True,
        })
        simulator.initialize_state(state)
        simulator.register_functor(closed, execution_group="fast")
        simulator.step(dt=1.0)
        runtime_point_audit = closed.last_strict_iphs_audit()
        self.assertIsNotNone(runtime_point_audit)
        committed = [
            name for name in certificate.output_variables
            if not name.startswith("diagnostics.")
            and simulator.get_variable_provenance(name)[-1]["functor"] == closed.name
        ]
        runtime_point_audit["governed_execution"] = {
            "entry": "FunctorBase.__call__",
            "functor": closed.name,
            "committed_variables": committed,
            "step": simulator.step_count,
        }
        verified = certificate.validate(runtime_point_audit=runtime_point_audit)
        self.assertEqual(verified["issues"], [])
        self.assertTrue(verified["passed"])
        self.assertEqual(verified["runtime_verified_level"], 5)
        self.assertEqual(verified["runtime_evidence_status"], "verified")
        self.assertEqual(verified["runtime_binding"], "process_bound")
        self.assertTrue(certificate.runtime_replacement_map["runtime_uses_iphs"])
        self.assertTrue(certificate.runtime_replacement_map["legacy_removed"])
        self.assertTrue(certificate.runtime_replacement_map["legacy_never_existed"])

        # The open-mechanism certificate is untouched by the selection.
        open_certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
        self.assertEqual(open_certificate.input_variables, EXPECTED_INPUT_VARIABLES)
        self.assertEqual(open_certificate.iphs_level, 3)

    def test_invalid_species_inputs_fail_fast_without_hidden_floor(self):
        """Reject invalid species values and allow an exact zero state."""
        chemistry = ChemistryFunctor(name="chemistry_test", device=self.device, physical_boundary="open_mechanism")
        base_state = {
            "air_temperature": torch.tensor(298.0),
            "shortwave_down": torch.tensor(300.0),
            "no_mixing_ratio": torch.tensor(1.0e-9),
            "no2_mixing_ratio": torch.tensor(1.0e-8),
            "o3_mixing_ratio": torch.tensor(4.0e-8),
            "atmospheric_pressure": torch.tensor(101325.0),  # Pa
            "co_mixing_ratio": torch.tensor(150e-9),  # OH sink, 150 ppb
            "ch4_mixing_ratio": torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            "specific_humidity": torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            "cos_zenith": torch.tensor(1.0),  # overhead sun
        }

        for invalid_value in (-1.0e-9, float("nan"), float("inf")):
            with self.subTest(invalid_value=invalid_value):
                state = dict(base_state)
                state["no_mixing_ratio"] = torch.tensor(invalid_value)
                with self.assertRaisesRegex(ValueError, "no_mixing_ratio"):
                    chemistry.forward(state, time=0.0, dt=60.0)

        zero_state = dict(base_state)
        zero_state.update(
            {
                "no_mixing_ratio": torch.tensor(0.0),
                "no2_mixing_ratio": torch.tensor(0.0),
                "o3_mixing_ratio": torch.tensor(0.0),
                "atmospheric_pressure": torch.tensor(101325.0),  # Pa
                "co_mixing_ratio": torch.tensor(150e-9),  # OH sink, 150 ppb
                "ch4_mixing_ratio": torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
                "specific_humidity": torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
                "cos_zenith": torch.tensor(1.0),  # overhead sun
            }
        )
        output = chemistry.forward(zero_state, time=0.0, dt=60.0)
        for key in ("no_mixing_ratio", "no2_mixing_ratio", "o3_mixing_ratio"):
            self.assertEqual(output[key].value.item(), 0.0)


class TestChemistryDeclaredSelectionsAndLayout(unittest.TestCase):
    """Round 47: no literal default stands in for a selection or a layout."""

    def test_physical_boundary_is_a_required_selection(self):
        import inspect

        parameter = inspect.signature(ChemistryFunctor.__init__).parameters['physical_boundary']
        self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            ChemistryFunctor(**{})

    def test_no_literal_default_duplicates_the_dependency_or_the_selection(self):
        import inspect

        source = inspect.getsource(ChemistryFunctor)
        self.assertNotIn('"dependencies", ["atmospheric_radiation"]', source)
        self.assertNotIn('getattr(self, "physical_boundary"', source)
        chemistry = ChemistryFunctor(physical_boundary='open_mechanism')
        # Since 2026-10-17 the radiation provider is optional (shortwave_down and cos_zenith
        # may be external forcing); the declaration is still made once, in get_dependencies.
        self.assertEqual(chemistry.get_dependencies(), {'optional': ['atmospheric_radiation']})

    def test_hodge_degrees_of_freedom_come_from_the_declared_species(self):
        import inspect

        self.assertEqual(
            ChemistryFunctor.OPEN_MECHANISM_SPECIES,
            ('no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio', 'isoprene_mixing_ratio'),
        )
        parameter = inspect.signature(ChemistryFunctor.build_hodge_contract).parameters['n_dofs']
        self.assertIs(parameter.default, inspect.Parameter.empty)
        chemistry = ChemistryFunctor(physical_boundary='open_mechanism')
        n_species = len(ChemistryFunctor.OPEN_MECHANISM_SPECIES)
        (hodge,) = chemistry.build_hodge_terms()
        self.assertEqual(tuple(hodge.matrix.shape), (n_species, n_species))
        self.assertEqual(chemistry.declared_forms()[0].n_dofs, n_species)
        dissipative = chemistry.dissipative_structure({})
        self.assertEqual(dissipative['state_vars'], ChemistryFunctor.OPEN_MECHANISM_SPECIES)
        self.assertEqual(tuple(dissipative['M'].shape), (n_species, n_species))

    # IUPAC preferred values (Atkinson et al. 2004, ACP 4:1461, Vol. I; Atkinson et al.
    # 2006, ACP 6:3625, Vol. II) and MCM v3.3.1: datasheet, A [cm3 molecule-1 s-1],
    # E/R [K], k298 [cm3 molecule-1 s-1], evaluated temperature range [K], Delta(E/R) [K].
    ARRHENIUS_EVALUATIONS = {
        "k_no_o3": ("I.A3.54", 1.4e-12, 1310.0, 1.8e-14, (195.0, 308.0), 200.0),
        "k_isoprene_oh": ("II.A2.10", 2.7e-11, -390.0, 1.0e-10, (240.0, 430.0), 100.0),
        "k_isoprene_o3": ("II.A1.157", 1.03e-14, 1995.0, 1.27e-17, (240.0, 360.0), 200.0),
        "k_oh_ch4": ("II.A2.3", 1.85e-12, 1690.0, 6.4e-15, (200.0, 300.0), 100.0),
    }

    def test_arrhenius_parameters_reproduce_the_iupac_mcm_evaluations(self):
        """Each Arrhenius rate is the IUPAC k298-anchored law with the evaluated E/R.

        The kernel evaluates k(T) = k_ref exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K. With
        Ea = (E/R) R and k_ref = the IUPAC preferred k298, k(T) / (A exp(-(E/R)/T)) is the
        constant k298 / (A exp(-(E/R)/298)) over the evaluated range (IUPAC propagates its
        reliability from 298 K with Delta(E/R)); the constant is 1.043 for NO + O3, whose
        k298 and Arrhenius fit IUPAC evaluates separately, and within 0.4 % of 1 otherwise.
        Ea is positive for OH + CH4, NO + O3 and O3 + isoprene and negative for OH + isoprene.
        """
        import math

        from relic.utils.physical_constants import get_physical_constant
        from relic.utils.unified_response_functions import arrhenius_temperature_response

        chemistry = ChemistryFunctor(
            name="chemistry_test", device=torch.device("cpu"), physical_boundary="open_mechanism"
        )
        gas_constant = float(get_physical_constant("gas_constant"))
        for name, (_, pre_exponential, e_over_r, k298, (t_low, t_high), _) in self.ARRHENIUS_EVALUATIONS.items():
            k_reference = float(chemistry.get_parameter_value(f"{name}_298", required=True))
            activation = float(
                chemistry.get_parameter_value(f"{name}_activation_energy", required=True)
            )
            self.assertAlmostEqual(activation / gas_constant, e_over_r, delta=0.5, msg=name)
            self.assertAlmostEqual(k_reference / k298, 1.0, places=12, msg=name)
            temperature = torch.tensor([t_low, 298.0, t_high], dtype=torch.float64)
            modeled = arrhenius_temperature_response(
                torch.tensor(k_reference, dtype=torch.float64),
                temperature,
                torch.tensor(activation, dtype=torch.float64),
            )
            evaluated = pre_exponential * torch.exp(-e_over_r / temperature)
            anchor = k298 / (pre_exponential * math.exp(-e_over_r / 298.0))
            relative = (modeled / evaluated / anchor - 1.0).abs().max().item()
            self.assertLess(relative, 5.0e-3, msg=name)
        rates = chemistry._calculate_temperature_dependent_rates(
            torch.tensor([273.0, 313.0], dtype=torch.float64),
            reactions=("k_no_o3", "k_isoprene_oh", "k_isoprene_o3"),
        )
        # OH + isoprene slows with warming; NO + O3 and O3 + isoprene speed up.
        self.assertLess(float(rates["k_isoprene_oh"][1]), float(rates["k_isoprene_oh"][0]))
        self.assertGreater(float(rates["k_isoprene_o3"][1]), float(rates["k_isoprene_o3"][0]))
        self.assertGreater(float(rates["k_no_o3"][1]), float(rates["k_no_o3"][0]))

    def test_arrhenius_entries_cite_the_evaluation_they_use(self):
        import yaml
        from relic.utils.physical_constants import get_physical_constant

        gas_constant = float(get_physical_constant("gas_constant"))
        path = project_root / "relic" / "parameters" / "database" / "relic" / "atmosphere" / "chemistry.yaml"
        entries = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["parameters"]
        for name, (datasheet, _, e_over_r, _, _, reliability) in self.ARRHENIUS_EVALUATIONS.items():
            for key in (f"{name}_298", f"{name}_activation_energy"):
                entry = entries[f"atmosphere.chemistry.{key}"]
                text = " ".join(str(item) for item in entry["references"])
                self.assertIn(datasheet, text, msg=key)
                self.assertIn("MCM v3.3.1", text, msg=key)
                self.assertEqual(entry["provenance"]["source"], "literature", msg=key)
            entry = entries[f"atmosphere.chemistry.{name}_activation_energy"]
            text = " ".join(str(item) for item in entry["references"])
            self.assertIn(str(int(abs(e_over_r))), text, msg=name)
            self.assertNotIn("negative temperature dependence typical", str(entry.get("note", "")))
            # The bounds cover the IUPAC reliability interval and never change the sign of Ea.
            lower, upper = entry["bounds"]
            self.assertLessEqual(lower, (e_over_r - reliability) * gas_constant + 0.1, msg=name)
            self.assertGreaterEqual(upper, (e_over_r + reliability) * gas_constant - 0.1, msg=name)
            self.assertGreater(lower * upper, 0.0, msg=name)
            self.assertLessEqual(lower, entry["value"])
            self.assertLessEqual(entry["value"], upper)
        # OH + CO is temperature independent over 200-300 K (IUPAC II.A2.12); the OH + NO2
        # falloff (IUPAC I.A3.43) is not implemented, so its Ea = 0 is a labelled convention.
        oh_co = entries["atmosphere.chemistry.k_oh_co_activation_energy"]
        self.assertEqual(oh_co["value"], 0.0)
        self.assertIn("II.A2.12", " ".join(str(item) for item in oh_co["references"]))
        self.assertEqual(oh_co["provenance"]["source"], "literature")
        oh_no2 = entries["atmosphere.chemistry.k_oh_no2_activation_energy"]
        self.assertEqual(oh_no2["value"], 0.0)
        self.assertIn("I.A3.43", " ".join(str(item) for item in oh_no2["references"]))
        self.assertEqual(oh_no2["provenance"]["source"], "relic_convention")
        self.assertIn("falloff", oh_no2["description"])

    def test_q10_factors_and_oh_no2_rate_state_their_actual_source(self):
        """F3: the audit-only chemistry Q10 has no literature source (RELIC convention with
        a role); the photolysis Q10 is removed (IUPAC/Troe 2000: the NO2 quantum yield is 1.00
        at 300-398 nm at 248 and 298 K). k_oh_no2_298 is the IUPAC I.A3.43 fall-off at
        1 atm, 298.15 K: k = k0 kinf/(k0 + kinf) Fc^(1/(1 + (log10(k0/kinf)/N)^2)),
        N = 0.75 - 1.27 log10 Fc, k0 = 3.3e-30 (T/300)^-3.0 [N2], kinf = 4.1e-11, Fc = 0.4
        (Atkinson et al. 2004). Mollner et al. (2010) measured 9.2e-12 for the HONO2 channel
        in air at 760 Torr."""
        import math
        import yaml
        from relic.utils.physical_constants import get_physical_constant

        path = project_root / "relic" / "parameters" / "database" / "relic" / "atmosphere" / "chemistry.yaml"
        entries = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["parameters"]
        self.assertNotIn("atmosphere.chemistry.photolysis_temperature_q10", entries)
        entry = entries["atmosphere.chemistry.chemistry_temperature_q10"]
        self.assertEqual(entry["provenance"]["source"], "relic_convention")
        self.assertIn("Role:", entry["provenance"]["notes"])
        self.assertTrue(any(str(ref).startswith("RELIC convention") for ref in entry["references"]))
        temperature = 298.15
        boltzmann = float(get_physical_constant("boltzmann_constant"))
        number_density = 101325.0 / (boltzmann * temperature) * 1.0e-6  # molecule cm-3
        k0 = 3.3e-30 * (temperature / 300.0) ** -3.0 * number_density
        kinf, fc = 4.1e-11, 0.4
        width = 0.75 - 1.27 * math.log10(fc)
        falloff = k0 * kinf / (k0 + kinf) * fc ** (1.0 / (1.0 + (math.log10(k0 / kinf) / width) ** 2))
        entry = entries["atmosphere.chemistry.k_oh_no2_298"]
        self.assertAlmostEqual(entry["value"] / falloff, 1.0, delta=5.0e-3)
        text = " ".join(str(item) for item in entry["references"])
        self.assertIn("I.A3.43", text)
        self.assertIn("9.2e-12", text)
        self.assertEqual(entry["provenance"]["source"], "literature")

    def test_o1d_kinetics_and_mcm_photolysis_entries_cite_their_evaluations(self):
        """O(1D) quenching and water reaction (Warneck and Williams 2012 Table 7.1 from
        Sander et al. 2006, k = A exp(-B/T)) and the MCM v3.3.1 J1/J4 parameters."""
        import yaml
        from relic.utils.physical_constants import get_physical_constant

        gas_constant = float(get_physical_constant("gas_constant"))
        path = project_root / "relic" / "parameters" / "database" / "relic" / "atmosphere" / "chemistry.yaml"
        entries = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["parameters"]
        for partner, (k298, e_over_r) in {
            "h2o": (2.0e-10, -60.0), "n2": (3.1e-11, -110.0), "o2": (4.0e-11, -67.0),
        }.items():
            rate = entries[f"atmosphere.chemistry.k_o1d_{partner}_298"]
            energy = entries[f"atmosphere.chemistry.k_o1d_{partner}_activation_energy"]
            self.assertEqual(rate["value"], k298, msg=partner)
            self.assertAlmostEqual(energy["value"] / gas_constant, e_over_r, delta=0.01, msg=partner)
            for entry in (rate, energy):
                text = " ".join(str(item) for item in entry["references"])
                self.assertIn("Table 7.1", text, msg=partner)
                self.assertIn("Sander", text, msg=partner)
                self.assertEqual(entry["provenance"]["source"], "literature", msg=partner)
        for name, value in {
            "mcm_o1d_photolysis_l": 6.073e-5, "mcm_o1d_photolysis_m": 1.743, "mcm_o1d_photolysis_n": 0.474,
            "mcm_no2_photolysis_l": 1.165e-2, "mcm_no2_photolysis_m": 0.244, "mcm_no2_photolysis_n": 0.267,
        }.items():
            entry = entries[f"atmosphere.chemistry.{name}"]
            self.assertEqual(entry["value"], value, msg=name)
            self.assertIn("Saunders", " ".join(str(item) for item in entry["references"]), msg=name)
        oh_yield = entries["atmosphere.chemistry.o1d_water_oh_yield"]
        self.assertEqual((oh_yield["value"], list(oh_yield["bounds"])), (2.0, [2.0, 2.0]))

    def test_config_first_source_order_is_the_mixin_class_declaration(self):
        from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

        self.assertTrue(issubclass(ChemistryFunctor, NamespaceDefaultMixin))
        self.assertEqual(ChemistryFunctor.parameter_lookup_source_order, ('config', 'manager'))

    # Isoprene + NO3 (CR-A-03): MCM v3.3.1 NO3 + C5H8 = NISOPO2, k = 3.15e-12 exp(-450/T)
    # [cm3 molecule-1 s-1] (Jenkin et al. 2015 supplement MCM_C5H8_subset.fac), the IUPAC 2006
    # datasheet II.A3.69 expression over 250-390 K with Delta log k = +-0.2 at 298 K and
    # Delta(E/R) = +-200 K. The IUPAC 2015 sheet NO3_VOC8 (2.95e-12 exp(-450/T)) is the
    # sensitivity value, Delta log10 k = log10(3.15/2.95) = 0.0285 at every temperature.
    NO3_PRE_EXPONENTIAL = 3.15e-12
    NO3_E_OVER_R = 450.0

    def test_isoprene_no3_parameters_are_registered_with_units_and_sources(self):
        import math

        import yaml
        from relic.parameters import get_global_manager
        from relic.utils.physical_constants import get_physical_constant

        gas_constant = float(get_physical_constant("gas_constant"))
        manager = get_global_manager()
        rate_spec = manager.get_parameter("atmosphere.chemistry.k_isoprene_no3_298")
        energy_spec = manager.get_parameter("atmosphere.chemistry.k_isoprene_no3_activation_energy")
        self.assertIsNotNone(rate_spec)
        self.assertIsNotNone(energy_spec)
        self.assertEqual(rate_spec.units, "cm3 molecule-1 s-1")
        self.assertEqual(energy_spec.units, "J mol-1")

        k298 = self.NO3_PRE_EXPONENTIAL * math.exp(-self.NO3_E_OVER_R / 298.15)
        self.assertAlmostEqual(float(rate_spec.value) / k298, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(float(energy_spec.value) / gas_constant, self.NO3_E_OVER_R, delta=1.0e-3)
        # Bounds cover the IUPAC 2006 reliability (Delta log k = +-0.2 at 298 K and
        # Delta(E/R) = +-200 K) and never change the sign of Ea.
        rate_lower, rate_upper = (float(value) for value in rate_spec.bounds)
        self.assertLessEqual(rate_lower, k298 * 10.0 ** -0.2)
        self.assertGreaterEqual(rate_upper, k298 * 10.0 ** 0.2)
        energy_lower, energy_upper = (float(value) for value in energy_spec.bounds)
        self.assertLessEqual(energy_lower, (self.NO3_E_OVER_R - 200.0) * gas_constant + 0.1)
        self.assertGreaterEqual(energy_upper, (self.NO3_E_OVER_R + 200.0) * gas_constant - 0.1)
        self.assertGreater(energy_lower, 0.0)

        path = project_root / "relic" / "parameters" / "database" / "relic" / "atmosphere" / "chemistry.yaml"
        entries = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["parameters"]
        for key, dimension in (
            ("k_isoprene_no3_298", "second_order_rate_constant"),
            ("k_isoprene_no3_activation_energy", "energy_per_amount"),
        ):
            entry = entries[f"atmosphere.chemistry.{key}"]
            text = " ".join(str(item) for item in entry["references"])
            self.assertIn("MCM v3.3.1", text, msg=key)
            self.assertIn("II.A3.69", text, msg=key)
            self.assertIn("NO3_VOC8", text, msg=key)
            self.assertIn("2.95e-12", text, msg=key)
            self.assertEqual(entry["provenance"]["source"], "literature", msg=key)
            self.assertEqual(entry["dtype"], "float64", msg=key)
            self.assertEqual(entry["dimension"], dimension, msg=key)
            self.assertFalse(entry["requires_grad"], msg=key)

        chemistry = ChemistryFunctor(
            name="chemistry_test", device=torch.device("cpu"), physical_boundary="open_mechanism"
        )
        for key, units in (
            ("k_isoprene_no3_298", "cm3 molecule-1 s-1"),
            ("k_isoprene_no3_activation_energy", "J mol-1"),
        ):
            self.assertIn(key, ChemistryFunctor.OPEN_MECHANISM_PARAMETERS)
            self.assertIn(f"atmosphere.chemistry.{key}", chemistry.required_parameters())
            value = chemistry.get_parameter_value(
                key, namespace="atmosphere.chemistry", units=units, validate=True
            )
            self.assertIsNotNone(value, msg=key)

    def test_isoprene_no3_rate_follows_the_mcm_expression(self):
        chemistry = ChemistryFunctor(
            name="chemistry_test", device=torch.device("cpu"), physical_boundary="open_mechanism"
        )
        temperature = torch.tensor([250.0, 298.15, 390.0], dtype=torch.float64)
        rates = chemistry._calculate_temperature_dependent_rates(
            temperature, reactions=("k_isoprene_no3",)
        )
        modeled = rates["k_isoprene_no3"]
        expected = self.NO3_PRE_EXPONENTIAL * torch.exp(-self.NO3_E_OVER_R / temperature)
        self.assertLess(abs(float(modeled[1] / expected[1]) - 1.0), 1.0e-6)
        self.assertLess(float((modeled / expected - 1.0).abs().max()), 1.0e-5)
        # E/R = +450 K: NO3 addition speeds up with warming over the evaluated 250-390 K.
        self.assertLess(float(modeled[0]), float(modeled[1]))
        self.assertLess(float(modeled[1]), float(modeled[2]))
        # The forward mechanism has no NO3 species: the default selection is unchanged.
        default_rates = chemistry._calculate_temperature_dependent_rates(temperature)
        self.assertNotIn("k_isoprene_no3", default_rates)


if __name__ == '__main__':
    unittest.main(verbosity=2)

