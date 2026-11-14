"""L1 tests for the canonical dry-deposition resistance closure."""

import sys
import unittest
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.computation.operators.iphs_certificate import (
    load_iphs_equivalence_certificate,
)
from relic.core.functor import IPHSDECFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionAggregator,
    DryDepositionConfig,
)
from test.audit.scan_output_contract import analyze_source


CERTIFICATE_PATH = (
    project_root
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "atmosphere_transport_dry_deposition_aggregator_dry_deposition_aggregator.yaml"
)
MODULE_PATH = (
    project_root
    / "relic"
    / "scientific"
    / "atmosphere"
    / "transport"
    / "dry_deposition_aggregator.py"
)


class TestDryDepositionAggregatorParameters(unittest.TestCase):
    """Validate parameter ownership, numerics, and Phase 8 declarations."""

    @staticmethod
    def _state(value_scale: float = 1.0) -> dict[str, torch.Tensor]:
        return {
            "no_mixing_ratio": torch.tensor([1.0e-8 * value_scale]),
            "no2_mixing_ratio": torch.tensor([2.0e-8 * value_scale]),
            "o3_mixing_ratio": torch.tensor([5.0e-8 * value_scale]),
            "so2_mixing_ratio": torch.tensor([1.0e-9 * value_scale]),
            "pbl_height": torch.tensor([1000.0]),
            "aerodynamic_resistance": torch.tensor([50.0]),
        }

    def test_import_and_initialization(self):
        """The canonical class initializes with immutable species semantics."""
        deposition = DryDepositionAggregator(name="deposition_test")

        self.assertIsInstance(deposition, DryDepositionAggregator)
        self.assertEqual(
            deposition.target_species,
            ("no", "no2", "o3", "so2"),
        )
        self.assertEqual(
            tuple(deposition.species_map.values()),
            (
                "no_mixing_ratio",
                "no2_mixing_ratio",
                "o3_mixing_ratio",
                "so2_mixing_ratio",
            ),
        )
        self.assertIsNone(deposition.diffusivity)
        self.assertEqual(
            deposition.get_transport_metadata()["closure"],
            "resistance_law",
        )
        self.assertEqual(set(deposition.metadata.conservation_laws), {"mass"})

    def test_surface_resistance_parameters_use_yaml_ssot(self):
        """Every runtime numerical closure coefficient is manager-backed."""
        deposition = DryDepositionAggregator(name="deposition_test")

        for species in deposition.target_species:
            namespace = f"atmosphere.dry_deposition.species.{species}"
            value = deposition.get_parameter_value(
                "effective_non_aerodynamic_resistance",
                namespace=namespace,
                units="s m-1",
                validate=True,
                required=True,
            )
            scalar = value.item() if isinstance(value, torch.Tensor) else float(value)
            self.assertGreater(scalar, 0.0)
            self.assertEqual(
                deposition.species_parameters[species].effective_resistance,
                scalar,
            )

    def test_forward_matches_resistance_law_exactly(self):
        """Runtime output equals the single declared formula for every species."""
        deposition = DryDepositionAggregator(name="deposition_test")
        state = self._state()
        dt = 600.0

        output = deposition.forward(state, time=0.0, dt=dt)
        self.assertEqual(set(output), set(deposition.metadata.output_variables))

        expected_total = torch.zeros_like(state["no_mixing_ratio"])
        for species in deposition.target_species:
            key = deposition.species_map[species]
            resistance = deposition.species_parameters[species].effective_resistance
            expected = (
                -state[key]
                * dt
                / (
                    (state["aerodynamic_resistance"] + resistance)
                    * state["pbl_height"]
                )
            )
            item = output[key]
            self.assertIsInstance(item, UpdateItem)
            self.assertEqual(item.mode, UpdateMode.ACCUMULATE)
            self.assertTrue(item.per_timestep)
            self.assertEqual(item.unit, "mol mol-1")
            self.assertTrue(torch.allclose(item.value, expected, rtol=1.0e-6))
            expected_total = expected_total + expected

        self.assertTrue(
            torch.allclose(
                output[
                    "diagnostics.dry_deposition.total_mixing_ratio_increment"
                ],
                expected_total,
                rtol=1.0e-6,
            )
        )
        self.assertEqual(
            output[
                "diagnostics.dry_deposition.total_limiter_residual"
            ].item(),
            0.0,
        )
        self.assertTrue(
            torch.equal(
                output[
                    "diagnostics.dry_deposition.normalized_mass_balance_residual"
                ],
                torch.zeros(
                    len(deposition.target_species),
                    *state["no_mixing_ratio"].shape,
                ),
            )
        )
        self.assertEqual(
            deposition.check_conservation_laws(state, output)["mass"],
            0.0,
        )

    def test_static_output_contract_matches_metadata_exactly(self):
        """Every canonical runtime output must be statically auditable."""
        records = analyze_source(
            MODULE_PATH.read_text(encoding="utf-8-sig"),
            MODULE_PATH,
        )
        record = next(
            item for item in records if item["class"] == "DryDepositionAggregator"
        )

        self.assertEqual(record["unresolved_dynamic_keys"], [])
        self.assertEqual(record["missing_in_forward_literal"], [])
        self.assertEqual(record["extra_not_in_metadata"], [])
        self.assertEqual(
            set(record["forward_literal_keys"]),
            set(record["metadata_outputs"]),
        )

    def test_missing_canonical_inputs_fail_fast(self):
        """No missing state may be replaced by a default or alias."""
        required = [
            "no_mixing_ratio",
            "no2_mixing_ratio",
            "o3_mixing_ratio",
            "so2_mixing_ratio",
            "pbl_height",
            "aerodynamic_resistance",
        ]
        for key in required:
            with self.subTest(key=key):
                deposition = DryDepositionAggregator(name=f"missing_{key}")
                state = self._state()
                del state[key]
                with self.assertRaises(KeyError):
                    deposition.forward(state, time=0.0, dt=60.0)

    def test_invalid_state_and_timestep_fail_fast(self):
        """Non-physical states and timesteps are rejected without clipping."""
        deposition = DryDepositionAggregator(name="deposition_test")

        state = self._state()
        state["aerodynamic_resistance"] = torch.tensor([0.0])
        with self.assertRaises(ValueError):
            deposition.forward(state, time=0.0, dt=60.0)

        state = self._state()
        state["pbl_height"] = torch.tensor([-1.0])
        with self.assertRaises(ValueError):
            deposition.forward(state, time=0.0, dt=60.0)

        state = self._state()
        state["no_mixing_ratio"] = torch.tensor([-1.0e-9])
        with self.assertRaises(ValueError):
            deposition.forward(state, time=0.0, dt=60.0)

        with self.assertRaises(ValueError):
            deposition.forward(self._state(), time=0.0, dt=0.0)

    def test_availability_limiter_routes_residual(self):
        """A long timestep cannot remove more inventory than is available."""
        deposition = DryDepositionAggregator(name="deposition_test")
        state = self._state()

        output = deposition.forward(state, time=0.0, dt=1.0e12)

        for species in deposition.target_species:
            key = deposition.species_map[species]
            self.assertTrue(torch.equal(output[key].value, -state[key]))
        self.assertGreater(
            output[
                "diagnostics.dry_deposition.total_limiter_residual"
            ].item(),
            0.0,
        )
        self.assertEqual(
            deposition.check_conservation_laws(state, output)["mass"],
            0.0,
        )

    def test_normalized_mass_balance_checker_detects_tampering(self):
        """The per-species open-boundary residual cannot cancel by species."""
        deposition = DryDepositionAggregator(name="deposition_test")
        state = self._state()
        output = deposition.forward(state, time=0.0, dt=600.0)
        tampered = dict(output)
        tampered[
            "diagnostics.dry_deposition.normalized_mass_balance_residual"
        ] = torch.tensor([[0.0], [0.0], [1.0e-6], [0.0]])

        self.assertGreater(
            deposition.check_conservation_laws(state, tampered)["mass"],
            0.0,
        )

    def test_normalized_mass_balance_checker_fails_fast(self):
        """Malformed per-species residuals are never silently accepted."""
        deposition = DryDepositionAggregator(name="deposition_test")
        state = self._state()
        output = deposition.forward(state, time=0.0, dt=600.0)
        residual_key = (
            "diagnostics.dry_deposition.normalized_mass_balance_residual"
        )

        missing = dict(output)
        missing.pop(residual_key)
        with self.assertRaises(KeyError):
            deposition.check_conservation_laws(state, missing)

        for invalid, error_type in (
            ("invalid", TypeError),
            (torch.empty(0), ValueError),
            (torch.tensor(float("nan")), ValueError),
            (torch.tensor(float("inf")), ValueError),
        ):
            malformed = dict(output)
            malformed[residual_key] = invalid
            with self.assertRaises(error_type):
                deposition.check_conservation_laws(state, malformed)

    def test_disabled_runtime_has_no_writer(self):
        """Explicit structural deactivation returns only its diagnostic."""
        deposition = DryDepositionAggregator(
            name="deposition_test",
            deposition_config=DryDepositionConfig(enabled=False),
        )

        output = deposition.forward({}, time=0.0, dt=60.0)

        self.assertEqual(
            set(output),
            {
                "diagnostics.dry_deposition.enabled",
                "diagnostics.dry_deposition.normalized_mass_balance_residual",
            },
        )
        self.assertEqual(
            output["diagnostics.dry_deposition.enabled"].item(),
            0.0,
        )
        self.assertEqual(
            deposition.check_conservation_laws({}, output)["mass"],
            0.0,
        )

    def test_numerical_config_override_is_rejected(self):
        """General config cannot bypass the ParameterManager numerical SSOT."""
        with self.assertRaises(ValueError):
            DryDepositionAggregator(
                config={"parameters": {"reference_height": 10.0}}
            )
        with self.assertRaises(TypeError):
            DryDepositionAggregator(
                deposition_config=DryDepositionConfig(enabled="false")  # type: ignore[arg-type]
            )

    def test_inherited_diffusion_paths_fail_fast(self):
        """Dry deposition exposes no second constant-diffusivity runtime."""
        deposition = DryDepositionAggregator(name="deposition_test")
        field = torch.ones(2)

        with self.assertRaises(RuntimeError):
            deposition._resolve_diffusivity_tensor(field)
        with self.assertRaises(RuntimeError):
            deposition.boundary_conditions(
                "surface",
                field,
                torch.ones_like(field),
                "dirichlet",
            )
        with self.assertRaises(RuntimeError):
            deposition.compute_diffusivity_tensor(field, field)

    def test_phase8_iphs_structure_hooks_are_declared(self):
        """The non-strict level-3 transport declarations remain valid."""
        deposition = DryDepositionAggregator(name="deposition_test")

        self.assertIsInstance(deposition, IPHSDECFunctor)
        forms = deposition.declared_forms()
        hodge_terms = deposition.build_hodge_terms()
        dissipative = deposition.dissipative_structure({})
        ports = deposition.ports()

        self.assertEqual(
            {form.semantic_name for form in forms},
            {
                "near_surface_species_mixing_ratio",
                "dry_deposition_normalized_boundary_sink",
            },
        )
        self.assertEqual(len(hodge_terms), 1)
        self.assertTrue(
            torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        )
        self.assertEqual(
            {port.name for port in ports},
            {"dry_deposition_surface_export"},
        )

    def test_phase8_certificate_level3_evidence_is_valid(self):
        """The certificate remains explicitly non-strict level 3."""
        certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
        audit = certificate.validate()

        self.assertTrue(audit["passed"])
        self.assertEqual(certificate.canonical_role, "transport")
        self.assertEqual(certificate.iphs_level, 3)
        self.assertIn("level_3_limit", certificate.iphs_level_evidence)
        self.assertNotIn("storage_functional", certificate.iphs_level_evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
