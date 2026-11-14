"""Unit tests for SoilWaterTransportFunctor parameter wiring."""

import inspect
import unittest
from typing import Any

import torch

from relic.parameters import get_global_manager
from relic.scientific.pedosphere.transport.SoilWaterTransportFunctor import (
    SoilWaterTransportFunctor,
)
from relic.utils.physical_constants import get_physical_constant


# The declared metadata order, which is also the certificate parameter_keys order.
EXPECTED_PARAMETER_KEYS = (
    "pedosphere.grid.dz",
    "pedosphere.hydraulic.saturated_conductivity",
    "pedosphere.moisture.wilting_point",
    "pedosphere.soil.porosity",
    "pedosphere.van_genuchten.alpha_vg",
    "pedosphere.van_genuchten.n_vg",
    "pedosphere.transport.gravity_head_gradient",
    "transport.numerical.min_timestep",
    "transport.numerical.richards_tolerance",
    "transport.numerical.max_richards_iterations",
    "pedosphere.numerical.cfl_number",
    "pedosphere.numerical.epsilon_length",
    "pedosphere.numerical.epsilon_diffusivity",
    "pedosphere.numerical.water_capacity_epsilon",
    "pedosphere.numerical.harmonic_mean_factor",
    "pedosphere.numerical.linear_solver_tolerance",
    "pedosphere.numerical.under_relaxation_factor",
)


class TestSoilWaterTransportParameters(unittest.TestCase):
    """Verify that pedosphere transport functor pulls parameters from YAML."""

    def setUp(self) -> None:
        self.functor = SoilWaterTransportFunctor(
            device=torch.device('cpu'),
        )
        self.manager = get_global_manager()
        if self.manager is None:
            self.fail("Global ParameterManager must be available for this test")

    def _to_float(self, value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def test_hydraulic_properties_from_manager(self) -> None:
        ks_expected = self.manager.get('pedosphere.hydraulic.saturated_conductivity')
        wp_expected = self.manager.get('pedosphere.moisture.wilting_point')
        phi_expected = self.manager.get('pedosphere.soil.porosity')
        alpha_expected = self.manager.get('pedosphere.van_genuchten.alpha_vg')
        n_expected = self.manager.get('pedosphere.van_genuchten.n_vg')

        self.assertAlmostEqual(
            self._to_float(self.functor.saturated_conductivity),
            self._to_float(ks_expected),
        )
        self.assertAlmostEqual(
            self._to_float(self.functor.wilting_point),
            self._to_float(wp_expected),
        )
        self.assertAlmostEqual(
            self._to_float(self.functor.porosity),
            self._to_float(phi_expected),
        )
        self.assertAlmostEqual(
            self._to_float(self.functor.alpha_vg),
            self._to_float(alpha_expected),
        )
        self.assertAlmostEqual(
            self._to_float(self.functor.n_vg),
            self._to_float(n_expected),
        )

    def test_alpha_default_is_the_loam_class_value_in_the_owner_unit(self) -> None:
        # References: Carsel, R.F. and Parrish, R.S. (1988), Water Resour. Res.
        #   24:755-769, class means as tabulated by van Genuchten, M.Th., Leij, F.J.
        #   and Yates, S.R. (1991), The RETC Code, EPA/600/2-91/065, Table 4 (loam).
        # Mathematical expression: alpha [m-1] = 100 * alpha [cm-1].
        # Parameter meanings: alpha and n of the van Genuchten retention curve of
        #   the loam texture class named by the owner description.
        # Units: alpha [cm-1] in the source table and [m-1] in the owner; n [1].
        # Domain/constraints: one texture class; the owner unit of alpha is the
        #   inverse of the pressure-head unit [m] of the retention functions.
        # Sign convention: not applicable.
        loam_alpha_per_cm = 0.036
        loam_n = 1.56
        expected = 100.0 * loam_alpha_per_cm
        spec = self.manager.get_parameter('pedosphere.van_genuchten.alpha_vg')
        lower, upper = spec.bounds

        self.assertEqual(spec.units, 'm-1')
        self.assertAlmostEqual(
            self._to_float(self.manager.get('pedosphere.van_genuchten.n_vg')),
            loam_n,
            places=6,
        )
        self.assertAlmostEqual(
            self._to_float(self.manager.get('pedosphere.van_genuchten.alpha_vg')),
            expected,
            places=5,
        )
        self.assertAlmostEqual(
            self._to_float(self.functor.alpha_vg), expected, places=5
        )
        self.assertLessEqual(lower, expected)
        self.assertLessEqual(expected, upper)

    def test_grid_and_solver_settings_from_manager(self) -> None:
        dz_expected = self.manager.get('pedosphere.grid.dz')
        tol_expected = self.manager.get('transport.numerical.richards_tolerance')
        max_iter_expected = self.manager.get('transport.numerical.max_richards_iterations')
        gravity_expected = self.manager.get('pedosphere.transport.gravity_head_gradient')

        self.assertAlmostEqual(
            self._to_float(self.functor.grid_spacing[0]),
            self._to_float(dz_expected),
        )
        self.assertAlmostEqual(
            self._to_float(self.functor.richards_tolerance),
            self._to_float(tol_expected),
        )
        self.assertEqual(
            int(self.functor.max_richards_iterations),
            int(max_iter_expected),
        )
        self.assertAlmostEqual(
            self._to_float(self.functor.gravity_head_gradient),
            self._to_float(gravity_expected),
        )
        self.assertEqual(self.functor.required_parameters(), EXPECTED_PARAMETER_KEYS)
        self.assertNotIn("field_capacity", self.functor.metadata.parameters)
        self.assertNotIn("domain_depth", self.functor.metadata.parameters)
        self.assertIn("pedosphere.grid.dz", self.functor.metadata.parameters)
        self.assertFalse(hasattr(self.functor, "field_capacity"))
        self.assertFalse(hasattr(self.functor, "domain_depth"))

        signature = inspect.signature(SoilWaterTransportFunctor.__init__)
        self.assertNotIn("grid_spacing", signature.parameters)
        self.assertNotIn("domain_depth", signature.parameters)

    def test_boundary_conversion_uses_physical_constant_ssot(self) -> None:
        water_density = get_physical_constant("water_density")

        self.assertAlmostEqual(
            self._to_float(self.functor.water_density),
            self._to_float(water_density),
        )
        self.assertNotIn(
            "pedosphere.state_defaults.infiltration_rate",
            self.functor.required_parameters(),
        )
        self.assertNotIn(
            "pedosphere.state_defaults.evapotranspiration_rate",
            self.functor.required_parameters(),
        )


if __name__ == '__main__':
    unittest.main()

