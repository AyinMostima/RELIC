"""
Test suite for TurbulenceFunctor parameter management

Tests L1 unit validation after Phase 6 parameter fixes.
Verifies that the fixed parameter call works correctly with units, bounds, and validate=True.

Validated parameters:
1. neutral_obukhov_length finite neutral-limit representation

References:
- Phase 6 Step 5: turbulence.py parameter fixes
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
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor


CERTIFICATE_PATH = (
    project_root
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "atmosphere_turbulence_turbulence_functor.yaml"
)


class TestTurbulenceFunctorParameters(unittest.TestCase):
    """Test TurbulenceFunctor parameter management"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')

    def test_import_module(self):
        """Test that TurbulenceFunctor can be imported"""
        self.assertIsNotNone(TurbulenceFunctor)

    def test_module_initialization(self):
        """Test basic module initialization with default parameters"""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )
        self.assertIsNotNone(turbulence)
        self.assertIsInstance(turbulence, TurbulenceFunctor)

    def test_parameter_retrieval(self):
        """Test neutral-limit parameter retrieval."""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )

        neutral_length = turbulence.get_parameter_value(
            "neutral_obukhov_length",
            namespace="atmosphere.turbulence",
            units="m",
            bounds=(100.0, 10000.0),
            validate=True
        )
        self.assertIsInstance(neutral_length, (int, float, torch.Tensor))

    def test_forward_minimal(self):
        """Test forward pass with minimal state input"""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )

        state = {
            'wind_speed': torch.tensor([5.0]),
            'air_temperature': torch.tensor([298.0]),
            'surface_temperature': torch.tensor([300.0]),
            'reference_height': torch.tensor([10.0]),
            'roughness_length_momentum': torch.tensor([0.1]),
            'displacement_height': torch.tensor([0.0]),
            'atmospheric_pressure': torch.tensor([101325.0]),
            'relative_humidity': torch.tensor([0.5]),
        }

        output = turbulence.forward(state, time=0.0, dt=1800.0)
        self.assertIsInstance(output, dict)
        for key in ('friction_velocity', 'aerodynamic_resistance'):
            self.assertIsInstance(output[key], UpdateItem)
            self.assertEqual(output[key].mode, UpdateMode.REPLACE)
            self.assertTrue(output[key].per_timestep)

    def test_no_hardcoded_parameters(self):
        """Verify no hardcoded physical constants bypass parameter system"""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )

        critical_params = [
            ("neutral_obukhov_length", "m", "atmosphere.turbulence"),
        ]

        for param_name, units, namespace in critical_params:
            value = turbulence.get_parameter_value(
                param_name,
                namespace=namespace,
                units=units,
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param_name} should be retrievable")

    def test_parameter_bounds_validation(self):
        """Test that bounds validation works for the fixed parameter"""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )

        neutral_length = turbulence.get_parameter_value(
            "neutral_obukhov_length",
            namespace="atmosphere.turbulence",
            units="m",
            bounds=(100.0, 10000.0),
            validate=True
        )
        if isinstance(neutral_length, torch.Tensor):
            length_value = neutral_length.item()
        else:
            length_value = float(neutral_length)
        self.assertGreaterEqual(length_value, 100.0)
        self.assertLessEqual(length_value, 10000.0)

    def test_parameter_units(self):
        """Verify units are correct (CF-1.8 compliant)"""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )

        neutral_length = turbulence.get_parameter_value(
            "neutral_obukhov_length",
            units="m",
            namespace="atmosphere.turbulence",
            validate=True
        )
        self.assertIsNotNone(neutral_length)

    def test_parameter_namespace(self):
        """Verify namespace correctness"""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )

        neutral_length = turbulence.get_parameter_value(
            "neutral_obukhov_length",
            namespace="atmosphere.turbulence",
            validate=True
        )
        self.assertIsNotNone(neutral_length)

    def test_missing_geometric_inputs_fail_fast(self):
        """Ensure missing MOST inputs cannot enter a hidden default path."""
        turbulence = TurbulenceFunctor(
            name="turbulence_test",
            device=self.device
        )
        state = {
            'wind_speed': torch.tensor([5.0], dtype=torch.float32),
            'air_temperature': torch.tensor([298.0], dtype=torch.float32),
        }
        with self.assertRaisesRegex(ValueError, "Missing required turbulence state inputs"):
            turbulence.forward(state, time=0.0, dt=600.0)

    def test_phase8_iphs_structure_hooks_are_declared(self):
        """Verify turbulence exposes Phase 8 transport structure hooks."""
        turbulence = TurbulenceFunctor(name="turbulence_test", device=self.device)

        self.assertIsInstance(turbulence, IPHSDECFunctor)
        forms = turbulence.declared_forms()
        hodge_terms = turbulence.build_hodge_terms()
        dissipative = turbulence.dissipative_structure({})
        ports = turbulence.ports()

        self.assertEqual(
            {form.semantic_name for form in forms},
            {
                "surface_layer_wind_speed",
                "surface_layer_momentum_gradient",
                "surface_layer_momentum_flux",
            },
        )
        expected_parameter_keys = {
            "atmosphere.turbulence.default_reference_height",
            "atmosphere.turbulence.roughness_heat_ratio",
            "atmosphere.turbulence.minimum_wind_speed",
            "atmosphere.turbulence.minimum_friction_velocity",
            "atmosphere.turbulence.max_iterations",
            "atmosphere.turbulence.convergence_tolerance",
            "atmosphere.turbulence.neutral_obukhov_length",
            "atmosphere.most.beta_stable",
            "atmosphere.most.gamma_unstable",
            "atmosphere.most.stable_correction_limit",
            "atmosphere.most.unstable_correction_limit",
            "atmosphere.most.h_min_safeguard",
            "atmosphere.most.monin_obukhov_length_clamp_min",
            "atmosphere.most.monin_obukhov_length_clamp_max",
        }
        self.assertEqual(set(turbulence.required_parameters()), expected_parameter_keys)
        self.assertTrue(
            {
                "atmosphere.most.beta_stable",
                "atmosphere.most.gamma_unstable",
                "atmosphere.most.stable_correction_limit",
                "atmosphere.most.unstable_correction_limit",
            }.issubset(turbulence.metadata.parameters)
        )
        self.assertFalse(
            any(
                key.startswith("relic.physical.constants.")
                for key in turbulence.required_parameters()
            )
        )
        self.assertEqual(len(hodge_terms), 1)
        self.assertEqual(
            hodge_terms[0].parameter_keys,
            ("atmosphere.turbulence.minimum_friction_velocity",),
        )
        self.assertEqual(
            hodge_terms[0].metric_keys,
            ("atmosphere.turbulence.default_reference_height",),
        )
        self.assertTrue(torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0))
        self.assertEqual(
            {port.name for port in ports},
            {"surface_layer_momentum_exchange_port"},
        )

    def test_momentum_flux_uses_the_moist_air_density_of_the_inputs(self):
        """tau = rho u*^2 with rho = (p - (1 - eps) e) / (R_d T_a) and e = RH e_s(T_a)."""
        from relic.scientific.hydrosphere.utils.psychrometrics import (
            compute_moist_air_density,
            compute_saturation_vapor_pressure,
        )
        from relic.utils.physical_constants import get_physical_constant

        tetens = {
            "tetens_base_pressure": float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_base_pressure")),
            "tetens_coeff_a": float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_a")),
            "tetens_coeff_b": float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_b")),
            "kelvin_offset": float(get_physical_constant("celsius_to_kelvin")),
        }
        turbulence = TurbulenceFunctor(name="turbulence_density", device=self.device)
        densities = []
        for air_temperature in (298.15, 273.15):
            state = {
                'wind_speed': torch.tensor([5.0], dtype=torch.float64),
                'air_temperature': torch.tensor([air_temperature], dtype=torch.float64),
                'surface_temperature': torch.tensor([air_temperature + 2.0], dtype=torch.float64),
                'reference_height': torch.tensor([10.0], dtype=torch.float64),
                'roughness_length_momentum': torch.tensor([0.1], dtype=torch.float64),
                'displacement_height': torch.tensor([0.0], dtype=torch.float64),
                'atmospheric_pressure': torch.tensor([101325.0], dtype=torch.float64),
                'relative_humidity': torch.tensor([0.5], dtype=torch.float64),
            }
            output = turbulence.forward(state, time=0.0, dt=60.0)
            u_star = output['friction_velocity'].value
            tau = output['momentum_flux'].value
            vapor_pressure = state['relative_humidity'] * compute_saturation_vapor_pressure(
                state['air_temperature'], **tetens
            )
            # References: Stull (1988); Holton (2012) Eq. 2.63-2.64; FAO-56 Eq. 11, 19.
            # Mathematical expression: tau = rho u*^2, rho = rho(T_a, p, RH e_s(T_a)).
            # Parameter meanings: T_a air temperature, p pressure, RH relative
            #   humidity, e_s saturation vapour pressure, u* friction velocity.
            # Units: tau [kg m-1 s-2]; rho [kg m-3]; u* [m s-1]; T_a [K]; p, e [Pa].
            # Domain/constraints: T_a > 0, p > 0, 0 <= RH e_s < p.
            # Sign convention: tau is a positive magnitude.
            rho = compute_moist_air_density(
                state['air_temperature'], state['atmospheric_pressure'], vapor_pressure
            )
            torch.testing.assert_close(tau / u_star**2, rho, rtol=1.0e-6, atol=0.0)
            densities.append(float(rho))
        # About 1.18 kg m-3 at 25 degC and 1.29 kg m-3 at 0 degC for RH = 0.5.
        self.assertGreater(densities[1] - densities[0], 0.1)
        self.assertEqual(
            {'atmospheric_pressure', 'relative_humidity'}
            - set(turbulence.metadata.input_variables),
            set(),
        )
        missing = dict(state)
        missing.pop('atmospheric_pressure')
        with self.assertRaises(ValueError):
            turbulence.forward(missing, time=0.0, dt=60.0)

    def test_phase8_certificate_level3_evidence_is_valid(self):
        """Verify turbulence remains level 3 with bounded evidence."""
        certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
        audit = certificate.validate()

        self.assertTrue(audit["passed"])
        self.assertEqual(certificate.canonical_role, "transport")
        self.assertEqual(certificate.iphs_level, 3)
        self.assertIn("level_3_limit", certificate.iphs_level_evidence)
        self.assertNotIn("storage_functional", certificate.iphs_level_evidence)
        self.assertFalse(
            any(
                key.startswith("relic.physical.constants.")
                for key in certificate.parameter_keys
            )
        )
        self.assertEqual(
            set(certificate.parameter_management_map),
            set(certificate.parameter_keys),
        )
        self.assertEqual(
            set(certificate.parameter_keys),
            set(TurbulenceFunctor().required_parameters()),
        )
        self.assertIn("rho*cp", certificate.literature_formula.replace(" ", ""))
        self.assertIn("rah=fh/(k*u*)", certificate.literature_formula.replace(" ", ""))
        self.assertNotIn("minimum_aerodynamic_resistance", certificate.parameter_keys)


class TestTurbulenceDeclaredLayout(unittest.TestCase):
    """Round 47: the Hodge degrees of freedom come from the declared layout."""

    def test_hodge_degrees_of_freedom_are_explicit(self):
        import inspect

        parameter = inspect.signature(TurbulenceFunctor.build_hodge_contract).parameters['n_dofs']
        self.assertIs(parameter.default, inspect.Parameter.empty)
        turbulence = TurbulenceFunctor(name='turbulence_layout', device=torch.device('cpu'))
        n_state = turbulence.declared_forms()[0].n_dofs
        (hodge,) = turbulence.build_hodge_terms()
        self.assertEqual(tuple(hodge.matrix.shape), (n_state, n_state))
        dissipative = turbulence.dissipative_structure({})
        self.assertEqual(tuple(dissipative['M'].shape), (n_state, n_state))


if __name__ == '__main__':
    unittest.main(verbosity=2)

