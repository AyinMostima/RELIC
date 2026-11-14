"""
Test suite for EddyDiffusionBLMFunctor parameter management

Tests the active ParameterManager bindings used by the single K-theory runtime.
Removed diagnostic/dilution modes and fallback meteorology are not part of the
public execution contract.

References:
- Phase 6 Step 5: eddy_diffusion_blm.py parameter fixes
- FunctorBase.get_parameter_value() API
"""

import unittest
from unittest.mock import patch
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import EddyDiffusionBLMFunctor


CERTIFICATE_PATH = (
    project_root
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "atmosphere_transport_eddy_diffusion_blm_eddy_diffusion_blm_functor.yaml"
)


class TestEddyDiffusionBLMFunctorParameters(unittest.TestCase):
    """Test EddyDiffusionBLMFunctor parameter management"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')

    def test_import_module(self):
        """Test that EddyDiffusionBLMFunctor can be imported"""
        self.assertIsNotNone(EddyDiffusionBLMFunctor)

    def test_module_initialization(self):
        """Test basic module initialization with default parameters"""
        eddy_diff = EddyDiffusionBLMFunctor(
            name="eddy_diff_test"
        )
        self.assertIsNotNone(eddy_diff)
        self.assertIsInstance(eddy_diff, EddyDiffusionBLMFunctor)

    def test_parameter_retrieval(self):
        """Test active K-theory parameters through the Functor access path."""
        eddy_diff = EddyDiffusionBLMFunctor(
            name="eddy_diff_test"
        )

        for name, units in (
            ("k_min", "m2 s-1"),
            ("k_max", "m2 s-1"),
            ("pbl_height_min", "m"),
            ("pbl_height_max", "m"),
            ("background_no_mixing_ratio", "mol mol-1"),
            ("background_o3_mixing_ratio", "mol mol-1"),
        ):
            value = eddy_diff.get_parameter_value(
                name,
                namespace="atmosphere.transport",
                units=units,
                validate=True,
            )
            self.assertIsInstance(value, (int, float, torch.Tensor))

        # Test the shared stable Businger-Dyer coefficient.
        beta_h = eddy_diff.get_parameter_value(
            "beta_stable",
            namespace="atmosphere.most",
            units="dimensionless",
            bounds=(4.0, 6.0),
            validate=True
        )
        self.assertIsInstance(beta_h, (int, float, torch.Tensor))

        # Test the shared unstable Businger-Dyer coefficient.
        gamma_h = eddy_diff.get_parameter_value(
            "gamma_unstable",
            namespace="atmosphere.most",
            units="dimensionless",
            bounds=(12.0, 20.0),
            validate=True
        )
        self.assertIsInstance(gamma_h, (int, float, torch.Tensor))

    def test_forward_minimal(self):
        """Disabled runtime emits one numeric diagnostic and no state writer."""
        eddy_diff = EddyDiffusionBLMFunctor(
            name="eddy_diff_test"
        )

        output = eddy_diff.forward({}, time=0.0, dt=1800.0)
        self.assertEqual(set(output), {"diagnostics.eddy_diffusion.enabled"})
        self.assertEqual(output["diagnostics.eddy_diffusion.enabled"].item(), 0.0)

    def test_no_hardcoded_parameters(self):
        """Verify the declared active parameter inventory is exact."""
        eddy_diff = EddyDiffusionBLMFunctor(
            name="eddy_diff_test"
        )

        self.assertEqual(
            eddy_diff.required_parameters(),
            (
                "atmosphere.transport.enabled",
                "atmosphere.transport.target_species",
                "atmosphere.transport.k_min",
                "atmosphere.transport.k_max",
                "atmosphere.transport.z_over_L_limit",
                "atmosphere.transport.ustar_min",
                "atmosphere.transport.ustar_max",
                "atmosphere.transport.pbl_height_min",
                "atmosphere.transport.pbl_height_max",
                "atmosphere.transport.background_no_mixing_ratio",
                "atmosphere.transport.background_no2_mixing_ratio",
                "atmosphere.transport.background_o3_mixing_ratio",
                "atmosphere.transport.background_isoprene_mixing_ratio",
                "transport.numerical.cfl_safety_factor",
                "transport.numerical.max_diffusion_stability",
                "atmosphere.most.beta_stable",
                "atmosphere.most.gamma_unstable",
                "atmosphere.most.stable_correction_limit",
                "atmosphere.most.unstable_correction_limit",
            ),
        )
        self.assertEqual(
            eddy_diff.physical_constant_keys(),
            (
                "relic.physical.constants.turbulence.von_karman_constant",
            ),
        )
        self.assertTrue(
            set(eddy_diff.required_parameters()).isdisjoint(
                eddy_diff.physical_constant_keys()
            )
        )
        self.assertNotIn("von_karman_constant", eddy_diff.metadata.parameters)

    def test_process_limit_respects_shared_most_validity_domain(self):
        """The process clamp must remain inside the shared MOST envelope."""
        eddy_diff = EddyDiffusionBLMFunctor(name="eddy_diff_test")
        runtime_parameters = dict(eddy_diff._runtime_parameters)
        runtime_parameters["z_over_L_limit"] = 5.0
        eddy_diff._runtime_parameters = runtime_parameters
        received_zeta = []

        def record_phi_h(zeta):
            received_zeta.append(zeta.detach().clone())
            return torch.ones_like(zeta)

        # References: Businger et al. (1971); Foken (2006).
        # Expression: zeta_eff = clip(z/L, -min(l_p,l_u), min(l_p,l_s)).
        # Parameters: l_p is the process cap; l_s and l_u are shared MOST limits.
        shared_limits = {
            "stable_correction_limit": 2.0,
            "unstable_correction_limit": 1.5,
        }
        with patch.object(
            eddy_diff,
            "get_parameter_value",
            side_effect=lambda name, **kwargs: shared_limits[name],
        ), patch(
            "relic.scientific.atmosphere.transport.eddy_diffusion_blm.most_phi_h",
            side_effect=record_phi_h,
        ):
            (
                _,
                stability_parameter,
                diffusivity_clamped,
                clipping_residual,
            ) = eddy_diff._calculate_eddy_diffusivity(
                    torch.tensor([800.0, 600.0]),
                    {
                        "friction_velocity": torch.tensor([0.3, 0.3]),
                        "monin_obukhov_length": torch.tensor([100.0, -100.0]),
                    },
                )

        expected = torch.tensor([2.0, -1.5])
        self.assertTrue(torch.allclose(stability_parameter, expected))
        self.assertTrue(torch.allclose(received_zeta[0], expected))
        self.assertEqual(diffusivity_clamped.item(), 0.0)
        self.assertTrue(
            torch.allclose(clipping_residual, torch.tensor([2.0, -1.5]))
        )
        self.assertIn(
            "diagnostics.eddy_diffusion.stability_parameter_clipping_residual",
            eddy_diff.metadata.output_variables,
        )

    def test_parameter_bounds_validation(self):
        """Test configured eddy diffusivity and MOST coefficient bounds."""
        eddy_diff = EddyDiffusionBLMFunctor(
            name="eddy_diff_test"
        )

        snapshot = eddy_diff.get_transport_metadata()["parameter_snapshot"]
        self.assertGreater(snapshot["k_bounds"][0], 0.0)
        self.assertGreater(
            snapshot["k_bounds"][1],
            snapshot["k_bounds"][0],
        )
        self.assertGreater(snapshot["cfl_factor"], 0.0)
        self.assertLessEqual(snapshot["cfl_factor"], 1.0)
        self.assertFalse(hasattr(eddy_diff.config_transport, "k_min"))

        # beta_h should be around 5.0
        beta_h = eddy_diff.get_parameter_value(
            "beta_stable",
            namespace="atmosphere.most",
            units="dimensionless",
            bounds=(4.0, 6.0),
            validate=True
        )
        if isinstance(beta_h, torch.Tensor):
            beta_val = beta_h.item()
        else:
            beta_val = float(beta_h)
        self.assertGreaterEqual(beta_val, 4.0)
        self.assertLessEqual(beta_val, 6.0)

    def test_parameter_units(self):
        """Verify active transport and numerical parameter units."""
        eddy_diff = EddyDiffusionBLMFunctor(
            name="eddy_diff_test"
        )

        background = eddy_diff.get_parameter_value(
            "background_no2_mixing_ratio",
            units="mol mol-1",
            namespace="atmosphere.transport",
            validate=True
        )
        self.assertIsNotNone(background)

        diffusivity = eddy_diff.get_parameter_value(
            "k_max",
            units="m2 s-1",
            namespace="atmosphere.transport",
            validate=True
        )
        self.assertIsNotNone(diffusivity)

        # Test MOST stability parameters (dimensionless)
        beta_h = eddy_diff.get_parameter_value(
            "beta_stable",
            units="dimensionless",
            namespace="atmosphere.most",
            validate=True
        )
        self.assertIsNotNone(beta_h)

    def test_parameter_namespace(self):
        """Verify namespace correctness"""
        eddy_diff = EddyDiffusionBLMFunctor(
            name="eddy_diff_test"
        )

        # Test atmosphere.transport namespace.
        transport_params = [
            "k_min",
            "k_max",
            "pbl_height_min",
            "background_isoprene_mixing_ratio",
        ]

        for param in transport_params:
            value = eddy_diff.get_parameter_value(
                param,
                namespace="atmosphere.transport",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in atmosphere.transport namespace")

        # Test atmosphere.most namespace
        most_params = ["beta_stable", "gamma_unstable"]
        for param in most_params:
            value = eddy_diff.get_parameter_value(
                param,
                namespace="atmosphere.most",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in atmosphere.most namespace")

    def test_cached_transport_values_are_process_parameters(self):
        """PBL validity bounds and background mixing ratios are process parameters of
        the relaxation law r = K / h^2 (C_bg - C); no cached value replaces an input."""
        eddy_diff = EddyDiffusionBLMFunctor(name="eddy_diff_parameters")
        self.assertFalse(hasattr(eddy_diff, "_transport_defaults"))
        self.assertFalse(hasattr(eddy_diff, "_load_transport_defaults"))
        self.assertEqual(
            set(eddy_diff._transport_parameters),
            {
                "pbl_height_min",
                "pbl_height_max",
                "background_no_mixing_ratio",
                "background_no2_mixing_ratio",
                "background_o3_mixing_ratio",
                "background_isoprene_mixing_ratio",
            },
        )

    def test_disabled_forward_has_no_hidden_state_writer(self):
        """Disabled transport returns no fake zero state updates."""
        eddy_diff = EddyDiffusionBLMFunctor(name="eddy_diff_test")
        output = eddy_diff.forward(
            {
                "no_mixing_ratio": torch.tensor([1.0e-8]),
            },
            time=0.0,
            dt=60.0,
        )

        self.assertEqual(set(output), {"diagnostics.eddy_diffusion.enabled"})
        self.assertEqual(output["diagnostics.eddy_diffusion.enabled"].item(), 0.0)
        self.assertFalse(any(not key.startswith("diagnostics.") for key in output))

    def test_phase8_iphs_structure_hooks_are_declared(self):
        """Verify eddy diffusion exposes Phase 8 transport structure hooks."""
        eddy_diff = EddyDiffusionBLMFunctor(name="eddy_diff_test")

        self.assertIsInstance(eddy_diff, IPHSDECFunctor)
        forms = eddy_diff.declared_forms()
        hodge_terms = eddy_diff.build_hodge_terms()
        dissipative = eddy_diff.dissipative_structure({})
        ports = eddy_diff.ports()

        self.assertEqual(
            {form.semantic_name for form in forms},
            {
                "boundary_layer_species_mixing_ratio",
                "boundary_layer_species_gradient",
                "boundary_layer_turbulent_flux",
            },
        )
        self.assertIn("atmosphere.transport.k_min", eddy_diff.required_parameters())
        self.assertEqual(len(hodge_terms), 1)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0))
        self.assertEqual(
            {port.name for port in ports},
            {"boundary_layer_exchange_port"},
        )

    def test_phase8_certificate_level3_evidence_is_valid(self):
        """Verify eddy diffusion remains level 3 with bounded evidence."""
        certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
        audit = certificate.validate()

        self.assertTrue(audit["passed"])
        self.assertEqual(certificate.canonical_role, "transport")
        self.assertEqual(certificate.iphs_level, 3)
        self.assertIn("level_3_limit", certificate.iphs_level_evidence)
        self.assertNotIn("storage_functional", certificate.iphs_level_evidence)
        eddy_diff = EddyDiffusionBLMFunctor(name="eddy_diff_test")
        self.assertEqual(
            list(certificate.input_variables),
            list(eddy_diff.metadata.input_variables),
        )
        self.assertEqual(
            list(certificate.output_variables),
            list(eddy_diff.metadata.output_variables),
        )
        self.assertEqual(
            list(certificate.parameter_keys),
            list(eddy_diff.required_parameters()),
        )
        physical_key = (
            "relic.physical.constants.turbulence.von_karman_constant"
        )
        self.assertNotIn(physical_key, certificate.parameter_map)
        self.assertEqual(
            certificate.parameter_management_map[physical_key]["source"],
            "physical constants SSOT via get_physical_constant",
        )
        self.assertFalse(certificate.runtime_replacement_map["runtime_uses_iphs"])
        self.assertFalse(certificate.runtime_replacement_map["legacy_removed"])
        self.assertTrue(certificate.runtime_replacement_map["no_legacy_mode"])


if __name__ == '__main__':
    unittest.main(verbosity=2)

