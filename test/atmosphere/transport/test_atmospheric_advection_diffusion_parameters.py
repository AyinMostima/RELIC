"""
Test suite for AtmosphericAdvectionDiffusionFunctor parameter management

Tests L1 unit validation after Phase 6 parameter fixes.
Verifies that all 11 fixed parameter calls work correctly with units, bounds, and validate=True.

Fixed parameters:
1. dx (Line 140)
2. dy (Line 141)
3. dx (Line 144)
4. dy (Line 145)
5. dz (Line 146)
6. boundary_layer_height (Line 152)
7. eddy_diffusivity_max (Line 160)
8. molecular_diffusivity (Line 168)
9. min_timestep (Line 177)
10. cfl_safety_factor (Line 184)
11. conservation_tolerance (Line 191)
12. friction_velocity (Line 358)
13. potential_temperature_gradient (Line 376)

References:
- Phase 6 Step 5: AtmosphericAdvectionDiffusionFunctor.py parameter fixes
- FunctorBase.get_parameter_value() API
"""

import copy
import importlib
import math
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import ExecutionError, IPHSDECFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.transport.boundary_conditions import BoundaryType
from relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor import AtmosphericAdvectionDiffusionFunctor

transport_module = importlib.import_module(
    "relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor"
)


CERTIFICATE_PATH = (
    project_root
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "atmosphere_transport_atmospheric_advection_diffusion_functor_atmospheric_advection_diffusion_functor.yaml"
)


class TestAtmosphericAdvectionDiffusionFunctorParameters(unittest.TestCase):
    """Test AtmosphericAdvectionDiffusionFunctor parameter management"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')

    def _build_with_tracked_physical_constants(self, stability_scheme):
        """Build the Functor with distinct values from the physical SSOT API."""
        values = {
            "gravity_acceleration": 5.0,
            "reference_temperature": 250.0,
            "von_karman_constant": 0.5,
            "air_specific_heat_cp": 1000.0,
            "relic.physical.constants.psychrometrics.tetens_base_pressure": 600.0,
            "relic.physical.constants.psychrometrics.tetens_coefficient_a": 17.0,
            "relic.physical.constants.psychrometrics.tetens_coefficient_b": 240.0,
            "celsius_to_kelvin": 273.0,
        }
        calls = []

        def tracked_get_physical_constant(name):
            calls.append(name)
            return values[name]

        with patch.object(
            transport_module,
            "get_physical_constant",
            side_effect=tracked_get_physical_constant,
            create=True,
        ):
            transport = AtmosphericAdvectionDiffusionFunctor(
                spatial_dims=2,
                stability_scheme=stability_scheme,
                boundary_layer_height=1000.0,
                device=self.device,
            )
        return transport, values, calls

    def test_import_module(self):
        """Test that AtmosphericAdvectionDiffusionFunctor can be imported"""
        self.assertIsNotNone(AtmosphericAdvectionDiffusionFunctor)

    def test_module_initialization(self):
        """Test basic module initialization with default parameters"""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )
        self.assertIsNotNone(transport)
        self.assertIsInstance(transport, AtmosphericAdvectionDiffusionFunctor)

    def test_device_migration_includes_nested_boundary_manager(self):
        """Keep nested boundary state on the Functor's effective device."""
        transport = AtmosphericAdvectionDiffusionFunctor(spatial_dims=2, stability_scheme="monin-obukhov")
        transport.boundary_manager.set_boundary_condition(
            "x_min",
            "dirichlet",
            torch.tensor(2.0),
        )

        result = transport.to_device("meta")

        self.assertIs(result, transport)
        self.assertEqual(transport.device.type, "meta")
        self.assertEqual(transport.boundary_manager.device.type, "meta")
        self.assertEqual(
            transport.boundary_manager.boundary_conditions["x_min"].value.device.type,
            "meta",
        )

    def test_parameter_retrieval(self):
        """Test grid and diffusivity parameter retrieval"""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )

        # Test dx (Line 140/144 fix)
        dx = transport.get_parameter_value(
            "dx",
            namespace="atmosphere.grid",
            units="m",
            bounds=(1.0, 10000.0),
            validate=True
        )
        self.assertIsInstance(dx, (int, float, torch.Tensor))

        # Test boundary_layer_height (Line 152 fix)
        blh = transport.get_parameter_value(
            "boundary_layer_height",
            namespace="atmosphere.boundary_layer",
            units="m",
            bounds=(50.0, 3000.0),
            validate=True
        )
        self.assertIsInstance(blh, (int, float, torch.Tensor))

        # Test eddy_diffusivity_max (Line 160 fix)
        K_max = transport.get_parameter_value(
            "eddy_diffusivity_max",
            namespace="atmosphere.turbulence",
            units="m2 s-1",
            bounds=(0.1, 100.0),
            validate=True
        )
        self.assertIsInstance(K_max, (int, float, torch.Tensor))

    def test_forward_minimal(self):
        """Test forward pass with minimal state input"""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )

        # Create minimal spatial state; the default monin-obukhov scheme needs
        # the surface sensible heat flux as forcing when no producer runs, and
        # the pressure and relative humidity of the moist-air density.
        spatial_size = (10, 10)
        state = {
            'tracer_concentration': torch.ones(spatial_size) * 1e-9,
            'wind_u': torch.full(spatial_size, 2.0),
            'wind_v': torch.zeros(spatial_size),
            'temperature': torch.full(spatial_size, 298.15),
            'sensible_heat_flux': torch.full(spatial_size, 50.0),
            'atmospheric_pressure': torch.full(spatial_size, 101325.0),
            'relative_humidity': torch.full(spatial_size, 0.5),
        }

        output = transport.forward(state, time=0.0, dt=100.0)
        self.assertIsInstance(output, dict)
        self.assertIn('tracer_concentration', output)
        self.assertIn('diagnostics.atmospheric_advection_diffusion.cfl_number', output)
        self.assertIn('diagnostics.stability_number', output)

    def test_no_hardcoded_parameters(self):
        """Verify no hardcoded physical constants bypass parameter system"""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )

        # All 11 fixed parameters should be retrievable via parameter system
        critical_params = [
            ("dx", "m", "atmosphere.grid"),
            ("dy", "m", "atmosphere.grid"),
            ("boundary_layer_height", "m", "atmosphere.boundary_layer"),
            ("eddy_diffusivity_max", "m2 s-1", "atmosphere.turbulence"),
            ("molecular_diffusivity", "m2 s-1", "atmosphere.molecular_diffusion"),
            ("cfl_safety_factor", "dimensionless", "transport.numerical"),
        ]

        for param_name, units, namespace in critical_params:
            value = transport.get_parameter_value(
                param_name,
                namespace=namespace,
                units=units,
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param_name} should be retrievable")

    def test_physical_constants_use_dedicated_ssot(self):
        """Require immutable constants to bypass the Functor parameter accessor."""
        transport, values, calls = self._build_with_tracked_physical_constants(
            "monin-obukhov"
        )

        self.assertEqual(
            calls,
            [
                "gravity_acceleration",
                "reference_temperature",
                "von_karman_constant",
                "air_specific_heat_cp",
                "relic.physical.constants.psychrometrics.tetens_base_pressure",
                "relic.physical.constants.psychrometrics.tetens_coefficient_a",
                "relic.physical.constants.psychrometrics.tetens_coefficient_b",
                "celsius_to_kelvin",
            ],
        )
        self.assertFalse(hasattr(transport, "air_density"))
        self.assertEqual(transport.gravity_acceleration, values["gravity_acceleration"])
        self.assertEqual(transport.reference_temperature, values["reference_temperature"])
        self.assertEqual(transport.von_karman_constant, values["von_karman_constant"])

        required = set(transport.required_parameters())
        self.assertIn(
            "atmosphere.turbulence.minimum_friction_velocity",
            required,
        )
        self.assertIn(
            "atmosphere.stability.potential_temperature_gradient",
            required,
        )
        self.assertFalse(any(key.startswith("relic.physical.constants") for key in required))

    def test_monin_obukhov_stability_uses_ssot_values_exactly(self):
        """Energy heat flux requires rho*cp and the actual input temperature."""
        transport, values, _ = self._build_with_tracked_physical_constants(
            "monin-obukhov"
        )
        original_lookup = transport.get_parameter_value

        def reject_physical_parameter_lookup(name, *args, **kwargs):
            namespace = str(kwargs.get("namespace", ""))
            if namespace.startswith("relic.physical.constants"):
                raise AssertionError(
                    "Physical constants must not use Functor parameter lookup."
                )
            return original_lookup(name, *args, **kwargs)

        keys = ("temperature", "sensible_heat_flux", "atmospheric_pressure", "relative_humidity")
        state = {
            key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for key, value in zip(keys, (280.0, 0.001, 95000.0, 0.6))
        }
        with patch.object(
            transport,
            "get_parameter_value",
            side_effect=reject_physical_parameter_lookup,
        ):
            actual = transport._compute_atmospheric_stability(state)

        u_star = float(
            original_lookup(
                "minimum_friction_velocity",
                namespace="atmosphere.turbulence",
                source_order=("manager",),
                required=True,
            )
        )
        from relic.scientific.hydrosphere.utils.psychrometrics import compute_moist_air_density

        def expected_zeta(temperature, heat, pressure, humidity):
            # References: Monin and Obukhov (1954); Tetens (1930); Holton and
            #   Hakim (2012) Eqs. 2.63-2.64.
            # Mathematical expression: zeta = z / L, L = -rho cp T u*^3 / (kappa g H),
            #   rho = (p - (1 - eps) RH e_s(T)) / (R_d T), e_s = e_0 exp(a Tc / (Tc + b)).
            # Parameter meanings: u* friction velocity, T air temperature, H heat
            #   flux, p pressure, RH relative humidity, e_0/a/b/K_0 tracked Tetens values.
            # Units: L, z [m]; rho [kg m-3]; cp [J kg-1 K-1]; H [W m-2]; p, e_s [Pa].
            # Domain/constraints: positive u*, T, p; 0 <= RH <= 1; nonzero H; the
            #   clamp stays inactive (asserted below).
            # Sign convention: upward heat flux gives negative L and zeta.
            celsius = temperature - values["celsius_to_kelvin"]
            saturation = values["relic.physical.constants.psychrometrics.tetens_base_pressure"] * torch.exp(
                values["relic.physical.constants.psychrometrics.tetens_coefficient_a"] * celsius
                / (celsius + values["relic.physical.constants.psychrometrics.tetens_coefficient_b"])
            )
            density = compute_moist_air_density(temperature, pressure, humidity * saturation)
            length = -(u_star**3 * density * values["air_specific_heat_cp"] * temperature) / (
                values["von_karman_constant"] * values["gravity_acceleration"] * heat
            )
            return torch.clamp(500.0 / length, min=-10.0, max=10.0)

        leaves = tuple(state[key].detach().clone().requires_grad_(True) for key in keys)
        expected = expected_zeta(*leaves)
        self.assertLess(float(expected.detach().abs()), 10.0)
        self.assertTrue(torch.allclose(actual, expected, rtol=1.0e-12, atol=0.0))
        actual_gradients = torch.autograd.grad(actual, tuple(state[key] for key in keys))
        expected_gradients = torch.autograd.grad(expected, leaves)
        for key, got, want in zip(keys, actual_gradients, expected_gradients):
            torch.testing.assert_close(got, want, rtol=1.0e-10, atol=0.0, msg=key)
        # The density follows the pressure and humidity: no fixed STP value.
        self.assertNotEqual(float(actual_gradients[2]), 0.0)
        self.assertNotEqual(float(actual_gradients[3]), 0.0)

    def test_unstable_requested_interval_is_rejected_before_field_mutation(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(0.01, 0.01), stability_scheme="constant", device=self.device,
        )
        field = torch.zeros((4, 4), dtype=torch.float64)
        state = {
            "tracer_concentration": field,
            "wind_u": torch.zeros_like(field),
            "wind_v": torch.zeros_like(field),
            "source_term": torch.full_like(field, 1.0e-7),
        }
        before = {name: value.clone() for name, value in state.items()}
        with self.assertRaisesRegex(ValueError, "stability"):
            transport.forward(state, time=0.0, dt=1000.0)
        for name, value in before.items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    def test_stable_interval_integrates_the_complete_source_duration(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        field = torch.zeros((4, 4), dtype=torch.float64)
        state = {
            "tracer_concentration": field,
            "wind_u": torch.zeros_like(field),
            "wind_v": torch.zeros_like(field),
            "wind_w": torch.zeros_like(field),
            "temperature": torch.full_like(field, 280.0),
            "sensible_heat_flux": torch.zeros_like(field),
            "source_term": torch.full_like(field, 1.0e-7),
        }
        output = transport(state, time=0.0, dt=1000.0)
        # References: LeVeque (2002), uniform-source transport oracle.
        # Mathematical expression: delta_c=S*dt for uniform c and zero velocity.
        # Parameter meanings: S is the external concentration source rate.
        # Units: c [kg m-3], S [kg m-3 s-1], dt [s].
        # Domain/constraints: spatially uniform field and stable explicit interval.
        # Sign convention: positive S adds tracer mass.
        increment = output["tracer_concentration"].value
        self.assertEqual(increment.dtype, torch.float64)
        torch.testing.assert_close(increment, torch.full_like(field, 1.0e-4), rtol=1.0e-12, atol=0.0)

    def test_eddy_diffusivity_is_elementwise_with_finite_branch_gradients(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, stability_scheme="richardson", device=self.device,
        )
        stability = torch.tensor([[-0.1, 0.0, 0.1]], dtype=torch.float64, requires_grad=True)
        actual = transport._compute_eddy_diffusivity({}, stability)
        # References: RELIC atmospheric transport's existing empirical K closure.
        # Mathematical expression: K=K0*(1+2*a/(1+a)) or K0/(1+10*zeta).
        # Parameter meanings: K0=0.1*Kmax; a=-zeta on the unstable branch.
        # Units: K,K0,Kmax [m2 s-1]; stability parameter [1].
        # Domain/constraints: each formula only on its own half-line.
        # Sign convention: negative stability enhances mixing; positive suppresses it.
        base = transport.eddy_diffusivity_max * 0.1
        expected = torch.tensor([[base * (1 + 0.2 / 1.1), base, base / 2]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=0.0)
        gradient = torch.autograd.grad(actual.sum(), stability)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        torch.testing.assert_close(gradient[0, 0], stability.new_tensor(-2 * base / 1.1**2))
        torch.testing.assert_close(gradient[0, 2], stability.new_tensor(-10 * base / 2**2))

    def test_scalar_forcing_broadcast_preserves_the_gradient(self):
        transport = AtmosphericAdvectionDiffusionFunctor(spatial_dims=2, device=self.device, stability_scheme="monin-obukhov")
        source = torch.tensor(0.125, dtype=torch.float64, requires_grad=True)
        field = torch.zeros((2, 3), dtype=torch.float64)
        broadcast = transport._broadcast_field(source, field)
        self.assertEqual(broadcast.dtype, field.dtype)
        derivative = torch.autograd.grad(broadcast.sum(), source)[0]
        self.assertEqual(float(derivative), 6.0)

    def test_richardson_stability_uses_ssot_values_exactly(self):
        """The existing mid-layer bulk closure must use shear, not wind speed alone."""
        transport, values, _ = self._build_with_tracked_physical_constants(
            "richardson"
        )
        original_lookup = transport.get_parameter_value

        def reject_physical_parameter_lookup(name, *args, **kwargs):
            namespace = str(kwargs.get("namespace", ""))
            if namespace.startswith("relic.physical.constants"):
                raise AssertionError(
                    "Physical constants must not use Functor parameter lookup."
                )
            return original_lookup(name, *args, **kwargs)

        state = {
            "tracer_concentration": torch.ones((2, 2), dtype=torch.float64),
            "wind_u": torch.full((2, 2), 20.0, dtype=torch.float64),
            "wind_v": torch.zeros((2, 2)),
        }
        with patch.object(
            transport,
            "get_parameter_value",
            side_effect=reject_physical_parameter_lookup,
        ):
            actual = transport._compute_atmospheric_stability(state)

        gradient = float(
            original_lookup(
                "potential_temperature_gradient",
                namespace="atmosphere.stability",
                source_order=("manager",),
                required=True,
            )
        )
        # References: Richardson (1920); Stull (1988), bulk Richardson number.
        # Mathematical expression: Ri=(g/theta)*theta_z*z**2/U**2.
        # Parameter meanings: z=BLH/2; U/z is the declared bulk-shear proxy.
        # Units: theta_z [K m-1], z [m], U [m s-1]; Ri [1].
        # Domain/constraints: linear theta profile; nonzero wind difference.
        # Sign convention: positive theta_z is stable stratification.
        expected_value = (
            gradient * values["gravity_acceleration"] * (transport.boundary_layer_height / 2)**2
        ) / (
            values["reference_temperature"] * 20.0**2
        )
        expected = torch.full_like(actual, expected_value)
        self.assertTrue(torch.allclose(actual, expected, rtol=1.0e-12, atol=0.0))

    def test_richardson_rejects_undefined_zero_shear(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, stability_scheme="richardson", device=self.device,
        )
        field = torch.zeros((2, 2), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "delta_u.*nonzero"):
            transport._compute_atmospheric_stability({
                "tracer_concentration": field, "wind_u": field, "wind_v": field,
            })

    def test_governed_richardson_grid_has_dimensionless_stability_and_finite_updates(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), boundary_layer_height=400.0,
            stability_scheme="richardson", device=self.device,
        )
        field = torch.full((3, 3), 1.0e-9, dtype=torch.float64)
        wind = torch.tensor([[20.0, 30.0, 40.0]] * 3, dtype=torch.float64)
        state = {
            "tracer_concentration": field, "wind_u": wind, "wind_v": torch.zeros_like(wind),
            "wind_w": torch.zeros_like(wind), "temperature": torch.full_like(wind, 280.0),
            "sensible_heat_flux": torch.zeros_like(wind),
        }
        result = transport(state, time=0.0, dt=0.01)
        gradient = float(transport.get_parameter_value(
            "potential_temperature_gradient", namespace="atmosphere.stability", required=True,
        ))
        expected = transport.gravity_acceleration * gradient * 200.0**2 / (transport.reference_temperature * wind**2)
        torch.testing.assert_close(result["diagnostics.stability_parameter"], expected, rtol=1.0e-12, atol=0)
        increment = result["tracer_concentration"].value
        self.assertEqual(increment.shape, field.shape)
        self.assertTrue(bool(torch.isfinite(increment).all()))

    def test_structural_dissipation_proxy_accepts_floating_state_and_states_its_actual_basis(self):
        transport = AtmosphericAdvectionDiffusionFunctor(spatial_dims=2, device=self.device, stability_scheme="monin-obukhov")
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                concentration = torch.tensor([1.0, 3.0, 5.0], dtype=dtype, requires_grad=True)
                result = transport.entropy_terms({"tracer_concentration": concentration})
                residual = concentration[:2].double() - concentration.double().mean()
                expected = residual.square().sum() * (transport.molecular_diffusivity + transport.eddy_diffusivity_max)
                torch.testing.assert_close(result["thermodynamic_force"], residual, rtol=0, atol=0)
                torch.testing.assert_close(result["dissipation_rate"], expected, rtol=1.0e-7, atol=0)
                self.assertTrue(bool(torch.isfinite(torch.autograd.grad(result["dissipation_rate"], concentration)[0]).all()))

    def test_parameter_bounds_validation(self):
        """Test that bounds validation works for all fixed parameters"""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )

        # Grid spacing should be positive
        dx = transport.get_parameter_value(
            "dx",
            namespace="atmosphere.grid",
            units="m",
            bounds=(1.0, 10000.0),
            validate=True
        )
        if isinstance(dx, torch.Tensor):
            dx_val = dx.item()
        else:
            dx_val = float(dx)
        self.assertGreaterEqual(dx_val, 1.0)
        self.assertLessEqual(dx_val, 10000.0)

        # CFL safety factor should be < 1
        cfl = transport.get_parameter_value(
            "cfl_safety_factor",
            namespace="transport.numerical",
            units="dimensionless",
            bounds=(0.1, 1.0),
            validate=True
        )
        if isinstance(cfl, torch.Tensor):
            cfl_val = cfl.item()
        else:
            cfl_val = float(cfl)
        self.assertGreaterEqual(cfl_val, 0.1)
        self.assertLessEqual(cfl_val, 1.0)

    def test_closed_boundary_update_preserves_inventory_and_nonnegative_cells(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0),
            stability_scheme="constant", device=self.device,
        )
        for size, location in ((5, (2, 2)), (4, (1, 1)), (4, (0, 0))):
            with self.subTest(size=size, location=location):
                field = torch.zeros((size, size), dtype=torch.float64)
                field[location] = 1.0
                state = {
                    "tracer_concentration": field, "wind_u": torch.zeros_like(field),
                    "wind_v": torch.zeros_like(field), "wind_w": torch.zeros_like(field),
                    "temperature": torch.full_like(field, 280.0), "sensible_heat_flux": torch.zeros_like(field),
                }
                before = field.clone()
                result = transport(state, time=0.0, dt=1.0)
                accepted = before + result["tracer_concentration"].value
                # References: LeVeque (2002), cell-centered finite-volume balance.
                # Mathematical expression: sum(delta_c*dx*dy)=0 for closed faces.
                # Parameter meanings: delta_c is relative to the caller's actual cells.
                # Units: 2D integrated inventory [kg m-1], per unit omitted depth.
                # Domain/constraints: zero source/wind, stable positive diffusion.
                # Sign convention: positive inventory change adds tracer.
                residual = (accepted - before).sum() * 1000.0**2
                self.assertLess(abs(float(residual)), 1.0e-9)
                self.assertTrue(bool(torch.all(accepted >= 0)))
                torch.testing.assert_close(field, before, rtol=0, atol=0)
                self.assertLess(abs(float(result["diagnostics.conservation_error"])), 1.0e-12)

    def test_physical_xyz_spacing_maps_to_storage_axes_for_advection_and_cfl(self):
        spacing = (2.0, 10.0)
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=spacing, stability_scheme="constant", device=self.device,
        )
        yy, xx = torch.meshgrid(torch.arange(5, dtype=torch.float64), torch.arange(5, dtype=torch.float64), indexing="ij")
        for axis, coordinates in enumerate((xx, yy)):
            with self.subTest(axis=axis):
                field = (1.0 + coordinates) * 1.0e-9
                state = {
                    "tracer_concentration": field,
                    "wind_u": torch.full_like(field, float(axis == 0)),
                    "wind_v": torch.full_like(field, float(axis == 1)),
                    "wind_w": torch.zeros_like(field), "temperature": torch.full_like(field, 280.0),
                    "sensible_heat_flux": torch.zeros_like(field),
                }
                result = transport(state, time=0.0, dt=0.001)
                # References: LeVeque (2002), upwind advection of a linear field.
                # Mathematical expression: delta_c=-u*dc/dx*dt; CFL=u*dt/dx.
                # Parameter meanings: spacing is physical (dx,dy), tensor axes (y,x).
                # Units: delta_c [kg m-3], spacing [m], dt [s], velocity [m s-1].
                # Domain/constraints: interior linear field has zero diffusion Laplacian.
                # Sign convention: positive velocity reduces an increasing profile locally.
                expected = -1.0e-9 * 0.001 / spacing[axis]
                self.assertAlmostEqual(float(result["tracer_concentration"].value[2, 2]), expected, delta=1.0e-21)
                self.assertAlmostEqual(
                    float(result["diagnostics.atmospheric_advection_diffusion.cfl_number"]),
                    0.001 / spacing[axis], delta=1.0e-15,
                )

    def test_unequal_spacing_quadratic_diffusion_uses_physical_dx(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(2.0, 10.0), stability_scheme="constant", device=self.device,
        )
        x = torch.arange(5, dtype=torch.float64)
        field = ((1.0 + x.square()) * 1.0e-9).repeat(5, 1)
        state = {
            "tracer_concentration": field, "wind_u": torch.zeros_like(field),
            "wind_v": torch.zeros_like(field), "wind_w": torch.zeros_like(field),
            "temperature": torch.full_like(field, 280.0), "sensible_heat_flux": torch.zeros_like(field),
        }
        result = transport(state, time=0.0, dt=0.001)
        # References: LeVeque (2007), central second difference of a quadratic.
        # Mathematical expression: delta_c=2*K*a*dt/dx**2 for c_i=a*(1+i**2).
        # Parameter meanings: a=1e-9 is concentration per squared grid index.
        # Units: K [m2 s-1], dx [m], dt [s], delta_c [kg m-3].
        # Domain/constraints: interior cell and a constant neutral diffusion coefficient.
        # Sign convention: a convex profile gains concentration at the central cell.
        diffusivity = transport.molecular_diffusivity + 0.1 * transport.eddy_diffusivity_max
        expected = 2.0 * diffusivity * 1.0e-9 * 0.001 / 2.0**2
        self.assertAlmostEqual(float(result["tracer_concentration"].value[2, 2]), expected, delta=1.0e-21)

    def test_default_three_dimensional_grid_uses_xyz_wind_components(self):
        transport = AtmosphericAdvectionDiffusionFunctor(spatial_dims=3, stability_scheme="constant", device=self.device)
        field = torch.ones((3, 3, 3), dtype=torch.float64) * 1.0e-9
        state = {
            "tracer_concentration": field, "wind_u": torch.ones_like(field),
            "wind_v": torch.full_like(field, 2.0), "wind_w": torch.full_like(field, 3.0),
            "temperature": torch.full_like(field, 280.0), "sensible_heat_flux": torch.zeros_like(field),
        }
        result = transport(state, time=0.0, dt=0.01)
        expected_cfl = 0.01 * sum(speed / width for speed, width in zip((1.0, 2.0, 3.0), transport.grid_spacing))
        self.assertAlmostEqual(
            float(result["diagnostics.atmospheric_advection_diffusion.cfl_number"]),
            expected_cfl, delta=1.0e-15,
        )

    def test_parameter_units(self):
        """Verify units are correct (CF-1.8 compliant)"""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )

        # Test grid spacing units (m)
        dx = transport.get_parameter_value(
            "dx",
            units="m",
            namespace="atmosphere.grid",
            validate=True
        )
        self.assertIsNotNone(dx)

        # Test diffusivity units (m^2 s^-1)
        K_max = transport.get_parameter_value(
            "eddy_diffusivity_max",
            units="m2 s-1",
            namespace="atmosphere.turbulence",
            validate=True
        )
        self.assertIsNotNone(K_max)

        # Test boundary layer height units (m)
        blh = transport.get_parameter_value(
            "boundary_layer_height",
            units="m",
            namespace="atmosphere.boundary_layer",
            validate=True
        )
        self.assertIsNotNone(blh)

    def test_parameter_namespace(self):
        """Verify namespace correctness"""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )

        # Test atmosphere.grid namespace
        grid_params = ["dx", "dy"]
        for param in grid_params:
            value = transport.get_parameter_value(
                param,
                namespace="atmosphere.grid",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in atmosphere.grid namespace")

        # Test atmosphere.turbulence namespace
        turbulence_params = ["eddy_diffusivity_max"]
        for param in turbulence_params:
            value = transport.get_parameter_value(
                param,
                namespace="atmosphere.turbulence",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in atmosphere.turbulence namespace")

        # Test transport.numerical namespace
        numerical_params = ["cfl_safety_factor"]
        for param in numerical_params:
            value = transport.get_parameter_value(
                param,
                namespace="transport.numerical",
                validate=True
            )
            self.assertIsNotNone(value, f"Parameter {param} should exist in transport.numerical namespace")

    def test_monin_obukhov_length_uses_the_moist_air_density(self):
        """L = -rho cp T u*^3 / (kappa g H) with the moist-air density of T, p and RH."""
        from relic.scientific.hydrosphere.utils.psychrometrics import (
            compute_moist_air_density,
            compute_saturation_vapor_pressure,
        )
        from relic.utils.physical_constants import get_physical_constant

        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, stability_scheme="monin-obukhov", boundary_layer_height=1000.0,
            device=self.device,
        )
        u_star = float(transport.get_parameter_value(
            "minimum_friction_velocity", namespace="atmosphere.turbulence",
            source_order=("manager",), required=True,
        ))
        zetas = []
        for temperature in (298.15, 273.15):
            state = {
                "temperature": torch.tensor([temperature], dtype=torch.float64),
                "sensible_heat_flux": torch.tensor([1.0e-3], dtype=torch.float64),
                "atmospheric_pressure": torch.tensor([101325.0], dtype=torch.float64),
                "relative_humidity": torch.tensor([0.5], dtype=torch.float64),
            }
            actual = transport._compute_atmospheric_stability(state)
            vapor_pressure = state["relative_humidity"] * compute_saturation_vapor_pressure(
                state["temperature"],
                tetens_base_pressure=float(get_physical_constant(
                    "relic.physical.constants.psychrometrics.tetens_base_pressure")),
                tetens_coeff_a=float(get_physical_constant(
                    "relic.physical.constants.psychrometrics.tetens_coefficient_a")),
                tetens_coeff_b=float(get_physical_constant(
                    "relic.physical.constants.psychrometrics.tetens_coefficient_b")),
                kelvin_offset=float(get_physical_constant("celsius_to_kelvin")),
            )
            rho = compute_moist_air_density(
                state["temperature"], state["atmospheric_pressure"], vapor_pressure
            )
            # References: Monin and Obukhov (1954); Holton (2012) Eq. 2.63-2.64.
            # Mathematical expression: zeta = z / L, L = -rho cp T u*^3 / (kappa g H).
            # Parameter meanings: rho moist-air density, cp heat capacity, T air
            #   temperature, u* minimum friction velocity, H sensible heat flux, z = h/2.
            # Units: rho [kg m-3]; cp [J kg-1 K-1]; T [K]; H [W m-2]; z, L [m]; zeta [1].
            # Domain/constraints: |zeta| below the transport stability limit here.
            # Sign convention: upward H > 0 gives L < 0 and zeta < 0 (unstable).
            length = -rho * transport.air_specific_heat * state["temperature"] * u_star**3 / (
                transport.von_karman_constant * transport.gravity_acceleration * 1.0e-3
            )
            torch.testing.assert_close(actual, 500.0 / length, rtol=1.0e-9, atol=0.0)
            zetas.append(float(actual))
        self.assertFalse(hasattr(transport, "air_density"))
        self.assertTrue(all(abs(value) < 10.0 for value in zetas))

    def test_missing_wind_components_raise_instead_of_state_defaults(self):
        """The wind components have no ParameterManager fallback: a missing one raises;
        a 2-D field reads wind_u and wind_v only."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant",
            device=self.device,
        )
        self.assertFalse(hasattr(transport, "_state_defaults"))
        self.assertFalse(hasattr(transport, "_apply_state_defaults"))
        self.assertFalse(any(
            key.startswith("atmosphere.state_defaults.") for key in transport.required_parameters()
        ))
        self.assertIn("wind_w", transport.optional_input_variables)
        field = torch.full((3, 3), 1.0e-9, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "wind_v"):
            transport.forward(
                {"tracer_concentration": field, "wind_u": torch.zeros_like(field)},
                time=0.0, dt=1.0,
            )
        transport.forward(
            {
                "tracer_concentration": field,
                "wind_u": torch.zeros_like(field),
                "wind_v": torch.zeros_like(field),
            },
            time=0.0, dt=1.0,
        )

    def test_numerical_parameters_are_the_ones_the_explicit_scheme_uses(self):
        """The explicit Euler step is bounded before the update by the advective
        Courant number and the diffusion number (cfl_advection_max, cfl_diffusion_max,
        a violation raises); min_timestep, cfl_safety_factor and conservation_tolerance
        were read into attributes that nothing used and are no longer read."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant",
            device=self.device,
        )
        required = set(transport.required_parameters())
        for name in ("min_timestep", "cfl_safety_factor", "conservation_tolerance"):
            self.assertFalse(hasattr(transport, name))
            self.assertNotIn(f"transport.numerical.{name}", required)
            self.assertNotIn(f"transport.numerical.{name}", transport.metadata.parameters)
        for name in ("cfl_advection_max", "cfl_diffusion_max"):
            self.assertIn(f"transport.numerical.{name}", required)
            self.assertIn(f"transport.numerical.{name}", transport.metadata.parameters)

    def test_absent_source_term_is_the_declared_no_source_case(self):
        """source_term [kg m-3 s-1] is a declared optional input: its absence is the
        closed no-source case S = 0 (AdvectionDiffusion._prepare_source_term), not a
        ParameterManager default; atmosphere.transport.source_term_default is removed."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant",
            device=self.device,
        )
        self.assertFalse(hasattr(transport, "default_source_term"))
        self.assertIn("source_term", transport.metadata.input_variables)
        self.assertIn("source_term", transport.optional_input_variables)
        self.assertNotIn(
            "atmosphere.transport.source_term_default", transport.required_parameters()
        )
        field = torch.full((3, 3), 1.0e-9, dtype=torch.float64)
        field[1, 1] = 2.0e-9
        state = {
            "tracer_concentration": field,
            "wind_u": torch.full_like(field, 1.0),
            "wind_v": torch.zeros_like(field),
        }
        without = transport.forward(dict(state), time=0.0, dt=1.0)
        explicit = transport.forward(
            {**state, "source_term": torch.zeros_like(field)}, time=0.0, dt=1.0
        )
        torch.testing.assert_close(
            getattr(without["tracer_concentration"], "value", without["tracer_concentration"]),
            getattr(explicit["tracer_concentration"], "value", explicit["tracer_concentration"]),
        )

    def test_monin_obukhov_reads_the_sensible_heat_flux_written_by_turbulence(self):
        """The Obukhov-length heat flux is the atmospheric_turbulence output, not a default."""
        from relic.core.simulator import RELICSimulator
        from relic.scientific.atmosphere.turbulence import TurbulenceFunctor

        # The wind components carry no SCM contract; this caller declares their units.
        simulator = RELICSimulator({
            "time_step": 1.0,
            "state_units": {"wind_u": "m s-1", "wind_v": "m s-1", "wind_w": "m s-1"},
        })
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="monin-obukhov",
            device=self.device,
        )
        # Registered consumer first: only the declared dependency orders the producer first.
        simulator.register_functor(transport)
        simulator.register_functor(TurbulenceFunctor())
        order = simulator.execution_order
        self.assertIsNotNone(order)
        self.assertLess(
            order.index("atmospheric_turbulence"), order.index("atmospheric_advection_diffusion")
        )
        shape = (3, 3)
        simulator.initialize_state({
            "wind_speed": torch.full(shape, 3.0),
            "air_temperature": torch.full(shape, 290.0),
            "surface_temperature": torch.full(shape, 295.0),
            "reference_height": torch.full(shape, 10.0),
            "roughness_length_momentum": torch.full(shape, 0.1),
            "displacement_height": torch.zeros(shape),
            "tracer_concentration": torch.full(shape, 1.0e-9),
            "wind_u": torch.full(shape, 2.0),
            "wind_v": torch.zeros(shape),
            "wind_w": torch.zeros(shape),
            "temperature": torch.full(shape, 290.0),
            "atmospheric_pressure": torch.full(shape, 101325.0),
            "relative_humidity": torch.full(shape, 0.5),
        })
        with patch.object(
            transport,
            "_compute_atmospheric_stability",
            wraps=transport._compute_atmospheric_stability,
        ) as stability:
            simulator.step(1.0)
        read_state = stability.call_args.args[0]
        self.assertNotIn("surface_heat_flux", read_state)
        produced = simulator.state_manager.get_state("sensible_heat_flux")
        torch.testing.assert_close(
            torch.as_tensor(read_state["sensible_heat_flux"]).to(produced.dtype), produced,
            rtol=0.0, atol=0.0,
        )
        # A warmer surface gives an upward (positive) flux that is not the former
        # 20 W m-2 state default.
        self.assertTrue(bool(torch.all(produced > 0.0)))
        self.assertFalse(bool(torch.allclose(produced, torch.full_like(produced, 20.0))))

    def test_per_cell_diagnostics_share_the_caller_tracer_shape(self):
        """Unbatched and batched fields: increment and per-cell rates have one shape."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant",
            device=self.device,
        )
        for shape in ((4, 5), (2, 4, 5)):
            with self.subTest(shape=shape):
                field = torch.full(shape, 1.0e-9, dtype=torch.float64)
                field[..., 1, 2] = 2.0e-9
                state = {
                    "tracer_concentration": field, "wind_u": torch.full_like(field, 1.0),
                    "wind_v": torch.zeros_like(field), "wind_w": torch.zeros_like(field),
                }
                result = transport(state, time=0.0, dt=1.0)
                self.assertEqual(tuple(result["tracer_concentration"].value.shape), shape)
                self.assertEqual(tuple(result["diagnostics.advection_rate"].shape), shape)
                self.assertEqual(tuple(result["diagnostics.diffusion_rate"].shape), shape)
                # References: Hundsdorfer and Verwer (2003), explicit Euler step.
                # Mathematical expression: delta_c = dt * (A(c) + D(c) + S).
                # Parameter meanings: A, D the returned advection and diffusion rates,
                #   S = 0 the closed no-source case (no source_term supplied), dt the step.
                # Units: delta_c [kg m-3]; A, D, S [kg m-3 s-1]; dt [s].
                # Domain/constraints: combined single Euler step of the operator.
                # Sign convention: positive rates add tracer to the cell.
                source = torch.zeros_like(field)
                torch.testing.assert_close(
                    result["tracer_concentration"].value,
                    1.0 * (result["diagnostics.advection_rate"]
                           + result["diagnostics.diffusion_rate"] + source),
                    rtol=1.0e-9, atol=1.0e-24,
                )

    def test_cell_courant_sum_has_its_own_diagnostic_name(self):
        """The pre-step cell Courant sum is not published as the generic diagnostics.cfl_number."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(100.0, 50.0), stability_scheme="constant",
            device=self.device,
        )
        field = torch.full((4, 5), 1.0e-9, dtype=torch.float64)
        x = torch.arange(5, dtype=torch.float64)
        wind_u = (1.0 + x).repeat(4, 1)
        wind_v = torch.full_like(field, -2.0)
        state = {
            "tracer_concentration": field, "wind_u": wind_u, "wind_v": wind_v,
            "wind_w": torch.zeros_like(field),
        }
        result = transport(state, time=0.0, dt=1.0)
        # References: Courant, Friedrichs and Lewy (1928).
        # Mathematical expression: C = max over cells of dt * (|u|/dx + |v|/dy).
        # Parameter meanings: u, v the cell-centred wind components along x, y;
        #   dx, dy the physical cell widths; dt the requested step.
        # Units: u, v [m s-1]; dx, dy [m]; dt [s]; C [1].
        # Domain/constraints: finite winds and positive widths.
        # Sign convention: C is a magnitude, independent of the wind direction.
        expected = float(torch.max(1.0 * (wind_u.abs() / 100.0 + wind_v.abs() / 50.0)))
        self.assertAlmostEqual(
            float(result["diagnostics.atmospheric_advection_diffusion.cfl_number"]),
            expected, delta=1.0e-15,
        )
        self.assertNotIn("diagnostics.cfl_number", result)
        self.assertIn(
            "diagnostics.atmospheric_advection_diffusion.cfl_number",
            transport.metadata.output_variables,
        )
        self.assertNotIn("diagnostics.cfl_number", transport.metadata.output_variables)

    def test_monin_obukhov_without_a_sensible_heat_flux_fails(self):
        """No producer and no forcing: the step fails instead of using a default."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="monin-obukhov",
            device=self.device,
        )
        field = torch.full((3, 3), 1.0e-9, dtype=torch.float64)
        state = {
            "tracer_concentration": field, "wind_u": torch.zeros_like(field),
            "wind_v": torch.zeros_like(field), "wind_w": torch.zeros_like(field),
            "temperature": torch.full_like(field, 290.0),
        }
        with self.assertRaisesRegex(ValueError, "sensible_heat_flux"):
            transport.forward(state, time=0.0, dt=1.0)

    def test_monin_obukhov_without_air_temperature_fails(self):
        """The Obukhov length reads T; a missing temperature raises instead of
        taking the former atmosphere.state_defaults.temperature value."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="monin-obukhov",
            device=self.device,
        )
        field = torch.full((3, 3), 1.0e-9, dtype=torch.float64)
        state = {
            "tracer_concentration": field, "wind_u": torch.zeros_like(field),
            "wind_v": torch.zeros_like(field), "wind_w": torch.zeros_like(field),
            "sensible_heat_flux": torch.full_like(field, 50.0),
            "atmospheric_pressure": torch.full_like(field, 101325.0),
            "relative_humidity": torch.full_like(field, 0.5),
        }
        with self.assertRaisesRegex(ValueError, "temperature"):
            transport.forward(state, time=0.0, dt=1.0)
        self.assertFalse(hasattr(transport, "_state_defaults"))
        self.assertNotIn(
            "atmosphere.state_defaults.temperature", transport.required_parameters()
        )

    def test_heat_flux_and_temperature_are_inputs_of_the_monin_obukhov_scheme_only(self):
        """Declared inputs are what each stability scheme reads; the producer is a declared dependency."""
        from relic.core.dependency_graph import normalize_dependency_declaration

        for scheme, reads_heat_flux in (
            ("monin-obukhov", True), ("richardson", False), ("constant", False),
        ):
            with self.subTest(scheme=scheme):
                transport = AtmosphericAdvectionDiffusionFunctor(
                    spatial_dims=2, stability_scheme=scheme, device=self.device,
                )
                inputs = set(transport.metadata.input_variables)
                optional = set(transport.optional_input_variables)
                stability_inputs = {
                    "temperature", "sensible_heat_flux",
                    "atmospheric_pressure", "relative_humidity",
                }
                self.assertTrue(stability_inputs <= inputs)
                self.assertNotIn("surface_heat_flux", inputs)
                self.assertEqual(
                    optional & stability_inputs, set() if reads_heat_flux else stability_inputs
                )
                if not reads_heat_flux:
                    field = torch.full((3, 3), 1.0e-9, dtype=torch.float64)
                    output = transport(
                        {"tracer_concentration": field, "wind_u": torch.zeros_like(field),
                         "wind_v": torch.full_like(field, 1.0), "wind_w": torch.zeros_like(field)},
                        time=0.0, dt=1.0,
                    )
                    self.assertIn("tracer_concentration", output)
                declared = normalize_dependency_declaration(
                    transport.get_dependencies(), owner=transport.name
                )
                self.assertEqual(list(declared.optional), ["atmospheric_turbulence"])
                self.assertEqual(list(declared.required), [])
        self.assertNotIn(
            "atmosphere.state_defaults.surface_heat_flux", transport.required_parameters()
        )
        from relic.parameters import get_global_manager

        self.assertIsNone(
            get_global_manager().get("atmosphere.state_defaults.surface_heat_flux", required=False)
        )

    def test_phase8_iphs_structure_hooks_are_declared(self):
        """Verify Phase 8 structural IPHS declarations for atmospheric transport."""
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2,
            device=self.device, stability_scheme="monin-obukhov"
        )
        state = {
            "tracer_concentration": torch.ones((4, 4), device=self.device),
            "wind_u": torch.full((4, 4), 1.0, device=self.device),
            "wind_v": torch.full((4, 4), 0.5, device=self.device),
            "temperature": torch.full((4, 4), 290.0, device=self.device),
            "sensible_heat_flux": torch.full((4, 4), 50.0, device=self.device),
        }

        forms = transport.declared_forms()
        hodge_terms = transport.build_hodge_terms()
        dissipative = transport.dissipative_structure(state)
        ports = transport.ports()

        self.assertIsInstance(transport, IPHSDECFunctor)
        self.assertEqual(
            transport.PHASE8_RUNTIME_REPLACEMENT_STATUS,
            "canonical_iphs_runtime"
        )
        self.assertIn("atmosphere.turbulence.eddy_diffusivity_max", transport.required_parameters())
        self.assertEqual(
            {form.semantic_name for form in forms},
            {
                "tracer_concentration",
                "atmospheric_tracer_gradient",
                "atmospheric_tracer_flux",
            }
        )
        self.assertEqual(len(hodge_terms), 1)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0))
        self.assertEqual(
            {port.name for port in ports},
            {
                "atmosphere_tracer_boundary_export",
                "atmosphere_tracer_source_port",
            }
        )

    def test_phase8_certificate_level3_evidence_is_valid(self):
        """Verify the atmospheric transport certificate remains level 3."""
        certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
        audit = certificate.validate()

        self.assertTrue(audit["passed"])
        self.assertEqual(
            certificate.process_id,
            "relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor.AtmosphericAdvectionDiffusionFunctor",
        )
        self.assertEqual(certificate.canonical_role, "transport")
        self.assertEqual(certificate.iphs_level, 3)
        self.assertNotIn("storage_functional", certificate.iphs_level_evidence)

    @staticmethod
    def _closed_probe_state(field):
        return {
            "tracer_concentration": field, "wind_u": torch.zeros_like(field),
            "wind_v": torch.zeros_like(field), "wind_w": torch.zeros_like(field),
            "temperature": torch.full_like(field, 280.0), "sensible_heat_flux": torch.zeros_like(field),
        }

    @staticmethod
    def _face_ledger_2d(cochain, ny, nx):
        """Split the packed outward ledger into (x_min, x_max, y_min, y_max) faces."""
        parts = torch.split(cochain, (ny, ny, nx, nx))
        return dict(zip(("x_min", "x_max", "y_min", "y_max"), parts))

    @staticmethod
    def _boundary_snapshot(transport):
        snapshot = {}
        for face, bc in transport.boundary_manager.boundary_conditions.items():
            value = bc.value.clone() if isinstance(bc.value, torch.Tensor) else bc.value
            snapshot[face] = (bc.bc_type, value, bc.time_dependent)
        return snapshot

    def test_returned_increment_is_an_explicit_integrated_update_item(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        field = torch.full((3, 3), 1.0e-9, dtype=torch.float64)
        result = transport(self._closed_probe_state(field), time=0.0, dt=1.0)
        item = result["tracer_concentration"]
        self.assertIsInstance(item, UpdateItem)
        self.assertEqual(item.mode, UpdateMode.ACCUMULATE)
        self.assertTrue(item.per_timestep)
        self.assertEqual(item.unit, "kg m-3")
        self.assertEqual(item.source_functor, "atmospheric_advection_diffusion")
        self.assertEqual(item.value.shape, field.shape)
        self.assertEqual(item.value.dtype, torch.float64)
        self.assertNotIn("atmospheric_transport_flux_tracer_concentration", result)

    def test_uniform_source_budget_emits_no_false_mass_warning(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        field = torch.zeros((4, 4), dtype=torch.float64)
        state = self._closed_probe_state(field)
        state["source_term"] = torch.full_like(field, 1.0e-7)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            output = transport(state, time=0.0, dt=1000.0)
        # References: Patankar (1980); Hundsdorfer and Verwer (2003).
        # Mathematical expression: sum(delta_c)dV = dt*sum(S)dV with zero faces.
        # Parameter meanings: S is an explicit external source, not a closure error.
        # Units: delta_c [kg m-3], S [kg m-3 s-1], dt [s].
        # Domain/constraints: uniform field, zero wind, zero-gradient faces.
        # Sign convention: the open ledger reports zero residual for this source.
        mass_warnings = [w for w in captured if "Conservation law 'mass'" in str(w.message)]
        self.assertEqual(mass_warnings, [])
        torch.testing.assert_close(
            output["tracer_concentration"].value, torch.full_like(field, 1.0e-4), rtol=1.0e-12, atol=0.0,
        )
        self.assertLess(abs(float(output["diagnostics.conservation_error"])), 1.0e-12)
        ledger = output["diagnostics.boundary_flux_cochain"]
        self.assertEqual(tuple(ledger.shape), (16,))
        torch.testing.assert_close(ledger, torch.zeros_like(ledger), rtol=0.0, atol=0.0)

    def test_zero_neumann_with_wind_reports_outward_ledger_and_closes_open_budget(self):
        spacing = (1000.0, 2000.0)
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=spacing, stability_scheme="constant", device=self.device,
        )
        ny, nx = 3, 5
        field = ((1.0 + torch.arange(nx, dtype=torch.float64)) * 1.0e-9).repeat(ny, 1)
        u, dt = 1.0, 0.5
        state = self._closed_probe_state(field)
        state["wind_u"] = torch.full_like(field, u)
        result = transport(state, time=0.0, dt=dt)
        delta = result["tracer_concentration"].value
        ledger = self._face_ledger_2d(result["diagnostics.boundary_flux_cochain"], ny, nx)
        # References: LeVeque (2002), upwind zero-gradient ghost states; Patankar (1980).
        # Mathematical expression: j_out(x_min)=-u*c[:,0], j_out(x_max)=u*c[:,-1],
        #   diffusive face flux 0; sum(delta_c)dx*dy + dt*sum_f(j_out,f*A_f)=0.
        # Parameter meanings: A_f is dy for x faces and dx for y faces per unit depth.
        # Units: j_out [kg m-2 s-1], delta_c [kg m-3], spacing [m], dt [s].
        # Domain/constraints: zero-gradient faces are open to advective exchange.
        # Sign convention: negative j_out at x_min is an import through that face.
        torch.testing.assert_close(ledger["x_min"], -u * field[:, 0], rtol=1.0e-12, atol=0.0)
        torch.testing.assert_close(ledger["x_max"], u * field[:, -1], rtol=1.0e-12, atol=0.0)
        torch.testing.assert_close(ledger["y_min"], torch.zeros(nx, dtype=torch.float64), rtol=0.0, atol=0.0)
        torch.testing.assert_close(ledger["y_max"], torch.zeros(nx, dtype=torch.float64), rtol=0.0, atol=0.0)
        dx, dy = spacing
        inventory_change = float(delta.sum()) * dx * dy
        boundary_export = dt * (
            float(ledger["x_min"].sum()) * dy + float(ledger["x_max"].sum()) * dy
            + float(ledger["y_min"].sum()) * dx + float(ledger["y_max"].sum()) * dx
        )
        scale = dt * u * float(field[:, -1].sum()) * dy
        self.assertLess(abs(inventory_change + boundary_export), 1.0e-12 * scale)
        self.assertLess(abs(float(result["diagnostics.conservation_error"])), 1.0e-12)
        self.assertAlmostEqual(
            float(result["diagnostics.atmospheric_advection_diffusion.cfl_number"]),
            u * dt / dx, delta=1.0e-15,
        )

    def test_periodic_faces_conserve_inventory_without_exterior_ledger(self):
        spacing = (1000.0, 1000.0)
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=spacing, stability_scheme="constant", device=self.device,
        )
        for face in ("x_min", "x_max", "y_min", "y_max"):
            transport.boundary_manager.set_boundary_condition(face, "periodic", 0.0)
        field = torch.zeros((5, 5), dtype=torch.float64)
        pulse = 1.0e-6
        field[2, 4] = pulse
        u, dt = 1.0, 1.0
        state = self._closed_probe_state(field)
        state["wind_u"] = torch.full_like(field, u)
        result = transport(state, time=0.0, dt=dt)
        delta = result["tracer_concentration"].value
        # References: LeVeque (2002), periodic upwind wrap; Patankar (1980).
        # Mathematical expression: delta_c[2,0]=dt*c[2,4]*(u/dx+K/dx**2); sum(delta_c)=0.
        # Parameter meanings: the x_max cell feeds the wrapped x_min cell by
        #   upwind advection and diffusion; K is the constant total diffusivity.
        # Units: c [kg m-3], u [m s-1], K [m2 s-1], dx [m], dt [s].
        # Domain/constraints: fully paired periodic axes, no exterior faces.
        # Sign convention: a periodic domain has no boundary export.
        diffusivity = transport.molecular_diffusivity + 0.1 * transport.eddy_diffusivity_max
        dx = spacing[0]
        expected_wrap = dt * pulse * (u / dx + diffusivity / dx**2)
        self.assertAlmostEqual(float(delta[2, 0]), expected_wrap, delta=expected_wrap * 1.0e-12)
        self.assertNotIn("diagnostics.boundary_flux_cochain", result)
        self.assertLess(abs(float(delta.sum())), 1.0e-12 * pulse)
        self.assertTrue(bool(torch.all(field + delta >= 0)))
        self.assertLess(abs(float(result["diagnostics.conservation_error"])), 1.0e-12)

    def test_uniform_scalar_dirichlet_uses_half_cell_face_law_and_preserves_gradient(self):
        spacing = (1000.0, 500.0)
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=spacing, stability_scheme="constant", device=self.device,
        )
        boundary_value = torch.tensor(2.0e-9, dtype=torch.float64, requires_grad=True)
        for face in ("x_min", "x_max", "y_min", "y_max"):
            transport.boundary_manager.set_boundary_condition(face, "dirichlet", boundary_value)
        field = torch.zeros((5, 5), dtype=torch.float64)
        dt = 0.001
        result = transport(self._closed_probe_state(field), time=0.0, dt=dt)
        delta = result["tracer_concentration"].value
        detached = delta.detach()
        # References: Patankar (1980), half-cell boundary distance; LeVeque (2002).
        # Mathematical expression: edge cell delta_c=K*(g-0)/(h/2)/h*dt=2*K*g*dt/h**2
        #   per exterior face; interior cells of a zero field stay zero;
        #   outward face flux j_out=-K*(g-0)/(h/2).
        # Parameter meanings: g boundary concentration, K total diffusivity, h face width.
        # Units: g [kg m-3], K [m2 s-1], h [m], dt [s], j_out [kg m-2 s-1].
        # Domain/constraints: one static scalar g on every face, zero wind.
        # Sign convention: negative outward flux imports tracer from the boundary.
        diffusivity = transport.molecular_diffusivity + 0.1 * transport.eddy_diffusivity_max
        g = float(boundary_value.detach())
        dx, dy = spacing
        edge_x = 2.0 * diffusivity * g * dt / dx**2
        edge_y = 2.0 * diffusivity * g * dt / dy**2
        self.assertEqual(float(detached[2, 2]), 0.0)
        self.assertAlmostEqual(float(detached[2, 0]), edge_x, delta=edge_x * 1.0e-12)
        self.assertAlmostEqual(float(detached[0, 2]), edge_y, delta=edge_y * 1.0e-12)
        self.assertAlmostEqual(float(detached[0, 0]), edge_x + edge_y, delta=(edge_x + edge_y) * 1.0e-12)
        ledger = self._face_ledger_2d(result["diagnostics.boundary_flux_cochain"], 5, 5)
        torch.testing.assert_close(
            ledger["x_min"], torch.full((5,), -diffusivity * g / (dx / 2.0), dtype=torch.float64),
            rtol=1.0e-12, atol=0.0,
        )
        torch.testing.assert_close(
            ledger["y_max"], torch.full((5,), -diffusivity * g / (dy / 2.0), dtype=torch.float64),
            rtol=1.0e-12, atol=0.0,
        )
        inventory_change = float(detached.sum()) * dx * dy
        boundary_export = dt * (
            float(ledger["x_min"].sum()) * dy + float(ledger["x_max"].sum()) * dy
            + float(ledger["y_min"].sum()) * dx + float(ledger["y_max"].sum()) * dx
        )
        self.assertLess(abs(inventory_change + boundary_export), 1.0e-12 * abs(boundary_export))
        gradient = torch.autograd.grad(delta.sum(), boundary_value)[0]
        self.assertTrue(math.isfinite(float(gradient)))
        self.assertAlmostEqual(
            float(gradient), 2.0 * diffusivity * dt * (5 * 2 / dx**2 + 5 * 2 / dy**2),
            delta=1.0e-12 * float(gradient),
        )

    def test_unsupported_boundary_descriptors_fail_before_any_mutation(self):
        faces = ("x_min", "x_max", "y_min", "y_max")

        def all_dirichlet(manager, value):
            for face in faces:
                manager.set_boundary_condition(face, "dirichlet", value)

        cases = {
            "nonzero_neumann": (lambda m: m.set_boundary_condition("x_min", "neumann", 1.0e-12), "nonzero Neumann"),
            "robin": (lambda m: m.set_boundary_condition("x_min", "robin", (1.0, 1.0, 0.0)), "Mixed boundary laws"),
            "wall": (lambda m: m.set_boundary_condition("y_max", "wall", 0.0), "Mixed boundary laws"),
            "open": (lambda m: m.set_boundary_condition("y_max", "open", 0.0), "Mixed boundary laws"),
            "all_wall": (lambda m: [m.set_boundary_condition(f, "wall", 0.0) for f in faces], "no canonical finite-volume contract"),
            "all_open": (lambda m: [m.set_boundary_condition(f, "open", 0.0) for f in faces], "no canonical finite-volume contract"),
            "mixed_periodic": (
                lambda m: [m.set_boundary_condition(f, "periodic", 0.0) for f in ("x_min", "x_max")],
                "Mixed boundary laws",
            ),
            "callable": (lambda m: all_dirichlet(m, lambda t: 1.0e-9), "callable or time-dependent"),
            "time_dependent": (
                lambda m: [m.set_boundary_condition(f, "dirichlet", 1.0e-9, time_dependent=True) for f in faces],
                "callable or time-dependent",
            ),
            "tensor_field_dirichlet": (
                lambda m: all_dirichlet(m, torch.full((5,), 1.0e-9, dtype=torch.float64)),
                "exactly one element",
            ),
            "unequal_dirichlet": (
                lambda m: (all_dirichlet(m, 1.0e-9), m.set_boundary_condition("x_max", "dirichlet", 2.0e-9)),
                "differs from face",
            ),
            "extra_inactive_face": (lambda m: m.set_boundary_condition("z_min", "neumann", 0.0), "exactly the active faces"),
        }
        for name, (configure, message) in cases.items():
            with self.subTest(case=name):
                transport = AtmosphericAdvectionDiffusionFunctor(
                    spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
                )
                configure(transport.boundary_manager)
                field = torch.zeros((5, 5), dtype=torch.float64)
                field[2, 2] = 1.0
                state = self._closed_probe_state(field)
                before_state = {key: value.clone() for key, value in state.items()}
                before_boundary = self._boundary_snapshot(transport)
                with self.assertRaises(ExecutionError) as context:
                    transport(state, time=0.0, dt=1.0)
                self.assertIsInstance(context.exception.__cause__, (ValueError, TypeError))
                self.assertIn(message, str(context.exception.__cause__))
                for key, value in before_state.items():
                    torch.testing.assert_close(state[key], value, rtol=0.0, atol=0.0)
                after_boundary = self._boundary_snapshot(transport)
                self.assertEqual(set(after_boundary), set(before_boundary))
                for face in before_boundary:
                    self.assertEqual(after_boundary[face][0], before_boundary[face][0])
                    self.assertEqual(after_boundary[face][2], before_boundary[face][2])

    def test_boundary_descriptors_round_trip_through_functor_state_dict(self):
        faces = ("x_min", "x_max", "y_min", "y_max")
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        boundary_value = torch.tensor(7.0e-9, dtype=torch.float64)
        for face in faces:
            transport.boundary_manager.set_boundary_condition(face, "dirichlet", boundary_value)
        saved = copy.deepcopy(transport.state_dict())
        self.assertIn("_extra_state", saved)
        payload = saved["_extra_state"]
        self.assertEqual(payload["format"], AtmosphericAdvectionDiffusionFunctor.EXTRA_STATE_FORMAT)
        self.assertEqual(payload["version"], AtmosphericAdvectionDiffusionFunctor.EXTRA_STATE_VERSION)
        self.assertEqual(payload["spatial_dims"], 2)
        self.assertEqual(set(payload["boundary_conditions"]), set(faces))
        self.assertEqual(payload["boundary_conditions"]["x_min"]["type"], "dirichlet")
        self.assertIsNot(payload["boundary_conditions"]["x_min"]["value"], boundary_value)

        for face in faces:
            transport.boundary_manager.set_boundary_condition(face, "neumann", 0.0)
        template = torch.zeros((5, 5), dtype=torch.float64)
        self.assertEqual(transport._resolve_canonical_boundary_law(template)[0], "neumann")

        transport.load_state_dict(saved, strict=True)
        law, advective_state, diffusive_state = transport._resolve_canonical_boundary_law(template)
        self.assertEqual(law, "dirichlet")
        self.assertEqual(float(advective_state), 7.0e-9)
        self.assertEqual(advective_state.dtype, torch.float64)
        self.assertIs(advective_state, diffusive_state)
        for face in faces:
            restored = transport.boundary_manager.boundary_conditions[face]
            self.assertEqual(restored.bc_type, BoundaryType.DIRICHLET)
            self.assertIsInstance(restored.value, torch.Tensor)
            self.assertEqual(restored.value.dtype, torch.float64)
        result = transport(self._closed_probe_state(template), time=0.0, dt=0.001)
        self.assertGreater(float(result["tracer_concentration"].value[2, 0]), 0.0)
        self.assertEqual(float(result["tracer_concentration"].value[2, 2]), 0.0)

    def test_invalid_extra_state_is_rejected_without_partial_mutation(self):
        faces = ("x_min", "x_max", "y_min", "y_max")
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        saved = copy.deepcopy(transport.state_dict())["_extra_state"]

        def assert_defaults_untouched():
            for face in faces:
                bc = transport.boundary_manager.boundary_conditions[face]
                self.assertEqual(bc.bc_type, BoundaryType.NEUMANN)
                self.assertEqual(float(bc.value), 0.0)

        bad_law = copy.deepcopy(saved)
        bad_law["boundary_conditions"]["y_max"] = {"type": "dirichlet", "value": 3.0e-9}
        bad_law["boundary_conditions"]["x_min"] = {"type": "robin", "value": 0.0}
        with self.assertRaisesRegex(ValueError, "not restorable"):
            transport.set_extra_state(bad_law)
        assert_defaults_untouched()

        bad_dims = copy.deepcopy(saved)
        bad_dims["spatial_dims"] = 3
        with self.assertRaisesRegex(ValueError, "spatial_dims"):
            transport.set_extra_state(bad_dims)
        assert_defaults_untouched()

        missing_face = copy.deepcopy(saved)
        del missing_face["boundary_conditions"]["x_min"]
        with self.assertRaisesRegex(ValueError, "boundary faces"):
            transport.set_extra_state(missing_face)
        assert_defaults_untouched()

        bad_value = copy.deepcopy(saved)
        bad_value["boundary_conditions"]["x_min"] = {"type": "dirichlet", "value": float("nan")}
        with self.assertRaisesRegex(ValueError, "finite"):
            transport.set_extra_state(bad_value)
        assert_defaults_untouched()

        with self.assertRaisesRegex(TypeError, "mapping"):
            transport.set_extra_state([saved])
        assert_defaults_untouched()

        transport.boundary_manager.set_boundary_condition("x_min", "dirichlet", lambda t: 1.0e-9)
        with self.assertRaisesRegex(ValueError, "callable or time-dependent"):
            transport.state_dict()
        transport.boundary_manager.set_boundary_condition("x_min", "robin", (1.0, 1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "cannot be checkpointed"):
            transport.state_dict()

    def test_conservation_hook_reports_canonical_open_ledger_residual(self):
        transport = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        increment = torch.zeros((2, 2), dtype=torch.float64)
        reported = transport.check_conservation_laws(
            {},
            {
                "tracer_concentration": increment,
                "diagnostics.conservation_error": torch.tensor(0.25, dtype=torch.float64),
            },
        )
        self.assertEqual(reported, {"mass": 0.25})
        with self.assertRaisesRegex(ValueError, "diagnostics.conservation_error"):
            transport.check_conservation_laws({}, {"tracer_concentration": increment})
        with self.assertRaisesRegex(ValueError, "finite real scalar"):
            transport.check_conservation_laws(
                {},
                {
                    "tracer_concentration": increment,
                    "diagnostics.conservation_error": torch.tensor(float("nan")),
                },
            )
        with self.assertRaisesRegex(ValueError, "tracer_concentration"):
            transport.check_conservation_laws({}, {"diagnostics.conservation_error": torch.tensor(0.0)})

    def test_three_dimensional_closed_hotspot_preserves_inventory(self):
        transport = AtmosphericAdvectionDiffusionFunctor(spatial_dims=3, stability_scheme="constant", device=self.device)
        field = torch.zeros((3, 3, 3), dtype=torch.float64)
        field[1, 1, 1] = 1.0
        result = transport(self._closed_probe_state(field), time=0.0, dt=1.0)
        delta = result["tracer_concentration"].value
        dx, dy, dz = transport.grid_spacing
        # References: LeVeque (2002), cell-centered finite-volume balance in 3D.
        # Mathematical expression: sum(delta_c*dx*dy*dz)=0 with zero exterior faces,
        #   checked relative to the stored inventory sum(c*dx*dy*dz).
        # Parameter meanings: delta_c is relative to the caller's actual cells.
        # Units: inventory [kg]; spacing [m]; delta_c [kg m-3].
        # Domain/constraints: zero wind, zero source, zero-gradient faces, center pulse.
        # Sign convention: positive delta_c adds tracer.
        self.assertEqual(delta.shape, field.shape)
        self.assertLess(abs(float(delta.sum())), 1.0e-12 * float(field.sum()))
        self.assertTrue(bool(torch.all(field + delta >= 0)))
        ledger = result["diagnostics.boundary_flux_cochain"]
        self.assertEqual(tuple(ledger.shape), (6 * 9,))
        torch.testing.assert_close(ledger, torch.zeros_like(ledger), rtol=0.0, atol=0.0)
        self.assertLess(abs(float(result["diagnostics.conservation_error"])), 1.0e-12)

    def test_grid_spacing_must_match_spatial_dims_and_be_positive(self):
        with self.assertRaisesRegex(ValueError, "exactly one physical width"):
            AtmosphericAdvectionDiffusionFunctor(spatial_dims=2, grid_spacing=(1000.0,), device=self.device, stability_scheme="monin-obukhov")
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            AtmosphericAdvectionDiffusionFunctor(spatial_dims=2, grid_spacing=(1000.0, 0.0), device=self.device, stability_scheme="monin-obukhov")

    def test_boundary_descriptors_pass_the_canonical_checkpoint_grammar(self):
        from relic.core.simulator.checkpoint import _capture_functor_state_dict

        faces = ("x_min", "x_max", "y_min", "y_max")
        source = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        for face in faces:
            source.boundary_manager.set_boundary_condition(
                face, "dirichlet", torch.tensor(4.0e-9, dtype=torch.float64),
            )
        # References: RELIC simulator checkpoint owner, strict weights-only grammar.
        # Mathematical expression: decode(encode(state_dict)) == state_dict, then
        #   load_state_dict(strict=True) reproduces the configured face law.
        # Parameter meanings: the encoded payload is the durable restart record
        #   of the boundary descriptors; no second persistence channel exists.
        # Units: Dirichlet datum [kg m-3]; the descriptor law is dimensionless.
        # Domain/constraints: only the supported static scalar grammar is encoded.
        # Sign convention: none; descriptors are configuration, not fluxes.
        encoded, validated = _capture_functor_state_dict("atmospheric_advection_diffusion", source)
        self.assertIn("_extra_state", encoded)
        self.assertEqual(
            encoded["_extra_state"]["boundary_conditions"]["y_min"]["type"], "dirichlet",
        )
        restored = AtmosphericAdvectionDiffusionFunctor(
            spatial_dims=2, grid_spacing=(1000.0, 1000.0), stability_scheme="constant", device=self.device,
        )
        self.assertEqual(
            restored._resolve_canonical_boundary_law(torch.zeros((2, 2), dtype=torch.float64))[0],
            "neumann",
        )
        restored.load_state_dict(dict(validated), strict=True)
        law, boundary_state, _ = restored._resolve_canonical_boundary_law(
            torch.zeros((2, 2), dtype=torch.float64),
        )
        self.assertEqual(law, "dirichlet")
        self.assertEqual(float(boundary_state), 4.0e-9)
        self.assertEqual(
            restored.get_extra_state()["boundary_conditions"], source.get_extra_state()["boundary_conditions"],
        )


class TestAtmosphericTransportRequiredSelections(unittest.TestCase):
    """Round 47: the domain dimension and the stability scheme are required."""

    def test_dimension_and_stability_scheme_have_no_default(self):
        import inspect

        parameters = inspect.signature(AtmosphericAdvectionDiffusionFunctor.__init__).parameters
        self.assertIs(parameters['spatial_dims'].default, inspect.Parameter.empty)
        self.assertIs(parameters['stability_scheme'].default, inspect.Parameter.empty)
        self.assertIs(parameters['stability_scheme'].kind, inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(TypeError):
            AtmosphericAdvectionDiffusionFunctor(**{'spatial_dims': 3})


if __name__ == '__main__':
    unittest.main(verbosity=2)

