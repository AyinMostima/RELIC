"""
L1 Unit Tests for MOST Standard Library

Tests Monin-Obukhov Similarity Theory functions including stability functions,
friction velocity, Monin-Obukhov length, and aerodynamic resistance.

Mathematical Validation:
    - Stability functions psi_m and psi_h (Businger-Dyer formulation)
    - Friction velocity u* calculation
    - Monin-Obukhov length L
    - Aerodynamic resistance r_a
    - Eddy diffusivity K

Test Coverage:
    1. Neutral conditions (z/L = 0)
    2. Stable conditions (z/L > 0)
    3. Unstable conditions (z/L < 0)
    4. Extreme stability limits
    5. Consistency between functions
"""

import pytest
import torch

import relic.scientific.atmosphere.most as most_module
import relic.scientific.atmosphere.most.similarity_functions as similarity_functions
import relic.scientific.atmosphere.most.stability_parameters as stability_parameters_module

from relic.scientific.atmosphere.most import (
    psi_m, psi_h, phi_m, phi_h,
    friction_velocity,
    monin_obukhov_length,
    temperature_scale,
    aerodynamic_resistance,
    aerodynamic_resistance_momentum,
    eddy_diffusivity,
    richardson_number,
    zeta_from_richardson,
    iterative_most_solution,
    VON_KARMAN_CONSTANT,
    GRAVITY_ACCELERATION
)


class TestStabilityFunctions:
    """Test MOST stability functions psi_m and psi_h."""
    
    def test_psi_neutral(self):
        """Test stability functions at neutral conditions (z/L = 0)."""
        zeta = torch.tensor(0.0)
        
        # At neutral conditions, psi = 0
        assert abs(psi_m(zeta).item()) < 1e-10
        assert abs(psi_h(zeta).item()) < 1e-10
    
    def test_psi_stable(self):
        """Test stability functions for stable conditions (z/L > 0)."""
        zeta_values = torch.tensor([0.1, 0.5, 1.0, 1.5])
        
        # For stable conditions: psi = -5 * z/L (Businger-Dyer)
        psi_m_vals = psi_m(zeta_values)
        psi_h_vals = psi_h(zeta_values)
        
        # Check that psi < 0 for stable conditions
        assert torch.all(psi_m_vals < 0)
        assert torch.all(psi_h_vals < 0)
        
        # Check approximate linear relationship for moderate stability
        expected_psi = -5.0 * zeta_values[:2]  # Only for small zeta
        assert torch.allclose(psi_m_vals[:2], expected_psi, rtol=0.1)
        assert torch.allclose(psi_h_vals[:2], expected_psi, rtol=0.1)
    
    def test_psi_unstable(self):
        """Test stability functions for unstable conditions (z/L < 0)."""
        zeta_values = torch.tensor([-0.1, -0.5, -1.0, -1.5])
        
        psi_m_vals = psi_m(zeta_values)
        psi_h_vals = psi_h(zeta_values)
        
        # For unstable conditions, psi > 0
        assert torch.all(psi_m_vals > 0)
        assert torch.all(psi_h_vals > 0)
        
        # The momentum and heat corrections differ under unstable conditions.
    
    def test_psi_limits(self):
        """Test exact functions at the configured Businger-Dyer domain bounds."""
        zeta_stable = torch.tensor(2.0)
        psi_m_stable = psi_m(zeta_stable)
        psi_h_stable = psi_h(zeta_stable)
        
        assert psi_m_stable.item() == pytest.approx(-10.0)
        assert psi_h_stable.item() == pytest.approx(-10.0)
        
        zeta_unstable = torch.tensor(-2.0)
        psi_m_unstable = psi_m(zeta_unstable)
        psi_h_unstable = psi_h(zeta_unstable)
        
        assert psi_m_unstable.item() > 0
        assert psi_h_unstable.item() > 0
        assert torch.isfinite(psi_m_unstable)
        assert torch.isfinite(psi_h_unstable)

    @pytest.mark.parametrize("function", [psi_m, psi_h, phi_m, phi_h])
    @pytest.mark.parametrize("zeta_value", [-2.01, 2.01])
    def test_similarity_functions_reject_out_of_domain_zeta(
        self,
        function,
        zeta_value,
    ):
        """Configured Businger-Dyer bounds must fail fast without clipping."""
        with pytest.raises(ValueError, match="Businger-Dyer validity domain"):
            function(torch.tensor(zeta_value))

    @pytest.mark.parametrize("function", [psi_m, psi_h, phi_m, phi_h])
    @pytest.mark.parametrize("zeta_value", [float("nan"), float("inf")])
    def test_similarity_functions_reject_nonfinite_zeta(
        self,
        function,
        zeta_value,
    ):
        """Non-finite stability forcing must not be hidden by a clamp."""
        with pytest.raises(ValueError, match="finite"):
            function(torch.tensor(zeta_value))

    def test_phi_h_has_no_hidden_output_clamp(self):
        """The gradient function must preserve its exact in-domain value."""
        assert phi_h(torch.tensor(2.0)).item() == pytest.approx(11.0)
    
    def test_psi_vectorized(self):
        """Test that stability functions work with vector inputs."""
        zeta = torch.linspace(-1, 1, 100)
        
        psi_m_vals = psi_m(zeta)
        psi_h_vals = psi_h(zeta)
        
        assert psi_m_vals.shape == zeta.shape
        assert psi_h_vals.shape == zeta.shape
        assert torch.all(torch.isfinite(psi_m_vals))
        assert torch.all(torch.isfinite(psi_h_vals))

    def test_single_similarity_api_has_no_wrapper_aliases(self):
        """The public package must expose only the canonical psi functions."""
        assert not hasattr(similarity_functions, "stability_correction_momentum")
        assert not hasattr(similarity_functions, "stability_correction_heat")
        assert not hasattr(most_module, "stability_correction_momentum")
        assert not hasattr(most_module, "stability_correction_heat")

    def test_coefficients_use_only_canonical_shared_parameter_keys(self, monkeypatch):
        """The Businger-Dyer closure must not use exception-driven aliases."""
        requested_keys = []
        values = {
            "atmosphere.most.beta_stable": 5.0,
            "atmosphere.most.gamma_unstable": 16.0,
            "atmosphere.most.stable_correction_limit": 2.0,
            "atmosphere.most.unstable_correction_limit": 2.0,
        }

        class RecordingManager:
            def get(self, key, **kwargs):
                requested_keys.append(key)
                return values[key]

        monkeypatch.setattr(
            similarity_functions,
            "get_global_manager",
            lambda: RecordingManager(),
        )
        coefficients = similarity_functions._most_coefficients()

        assert requested_keys == [
            "atmosphere.most.beta_stable",
            "atmosphere.most.gamma_unstable",
            "atmosphere.most.stable_correction_limit",
            "atmosphere.most.unstable_correction_limit",
        ]
        assert coefficients == {
            "beta_stable": 5.0,
            "gamma_unstable": 16.0,
            "stable_limit": 2.0,
            "unstable_limit": 2.0,
        }

    def test_parameter_manager_updates_are_not_hidden_by_local_cache(self, monkeypatch):
        """The utility must rely on ParameterManager as the only cache owner."""
        values = {
            "atmosphere.most.beta_stable": 5.0,
            "atmosphere.most.gamma_unstable": 16.0,
            "atmosphere.most.stable_correction_limit": 2.0,
            "atmosphere.most.unstable_correction_limit": 2.0,
        }

        class MutableManager:
            def get(self, key, **kwargs):
                return values[key]

        monkeypatch.setattr(
            similarity_functions,
            "get_global_manager",
            lambda: MutableManager(),
        )
        assert not hasattr(similarity_functions._most_coefficients, "cache_clear")
        first = similarity_functions._most_coefficients()
        values["atmosphere.most.beta_stable"] = 4.5
        second = similarity_functions._most_coefficients()

        assert first["beta_stable"] == 5.0
        assert second["beta_stable"] == 4.5


class TestFrictionVelocity:
    """Test friction velocity calculations."""
    
    def test_friction_velocity_neutral(self):
        """Test u* calculation for neutral conditions."""
        U = torch.tensor(10.0)  # Wind speed [m/s]
        z = torch.tensor(10.0)   # Height [m]
        z0m = torch.tensor(0.1)  # Roughness length [m]
        d = torch.tensor(0.0)    # Displacement height [m]
        zeta = torch.tensor(0.0)  # Neutral stability
        
        u_star = friction_velocity(U, z, z0m, d, zeta)
        
        # For neutral: u* = k * U / ln(z/z0m)
        expected = VON_KARMAN_CONSTANT * U / torch.log(z / z0m)
        
        assert torch.allclose(u_star, expected, rtol=1e-5)
    
    def test_friction_velocity_stable(self):
        """Test u* calculation for stable conditions."""
        U = torch.tensor(5.0)
        z = torch.tensor(10.0)
        z0m = torch.tensor(0.01)
        d = torch.tensor(0.0)
        zeta = torch.tensor(0.5)  # Stable
        
        u_star = friction_velocity(U, z, z0m, d, zeta)
        
        # u* should be less than neutral case due to stability
        u_star_neutral = friction_velocity(U, z, z0m, d, torch.tensor(0.0))
        assert u_star < u_star_neutral
        assert u_star > 0
    
    def test_friction_velocity_with_displacement(self):
        """Test u* calculation with displacement height."""
        U = torch.tensor(10.0)
        z = torch.tensor(15.0)
        z0m = torch.tensor(0.1)
        d = torch.tensor(5.0)  # Displacement height
        zeta = torch.tensor(0.0)
        
        u_star = friction_velocity(U, z, z0m, d, zeta)
        
        # Should use (z - d) in calculations
        effective_z = z - d
        expected = VON_KARMAN_CONSTANT * U / torch.log(effective_z / z0m)
        
        assert torch.allclose(u_star, expected, rtol=1e-5)

    def test_friction_velocity_includes_roughness_level_correction(self):
        """Test the complete integrated MOST momentum profile."""
        U = torch.tensor(5.0, dtype=torch.float64)
        z = torch.tensor(10.0, dtype=torch.float64)
        z0m = torch.tensor(0.1, dtype=torch.float64)
        d = torch.tensor(0.0, dtype=torch.float64)
        zeta_ref = torch.tensor(0.5, dtype=torch.float64)

        zeta_roughness = zeta_ref * z0m / (z - d)
        profile_factor = (
            torch.log((z - d) / z0m)
            - psi_m(zeta_ref)
            + psi_m(zeta_roughness)
        )
        expected = VON_KARMAN_CONSTANT * U / profile_factor

        actual = friction_velocity(U, z, z0m, d, zeta_ref)

        assert actual.dtype == torch.float64
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_friction_velocity_has_no_hidden_floor(self):
        """Test that the primitive returns the exact low-wind MOST value."""
        U = torch.tensor(0.01, dtype=torch.float64)
        z = torch.tensor(10.0, dtype=torch.float64)
        z0m = torch.tensor(0.1, dtype=torch.float64)
        d = torch.tensor(0.0, dtype=torch.float64)
        zeta_ref = torch.tensor(0.0, dtype=torch.float64)
        expected = VON_KARMAN_CONSTANT * U / torch.log((z - d) / z0m)

        actual = friction_velocity(U, z, z0m, d, zeta_ref)

        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        ("field", "replacement"),
        [
            ("U", torch.tensor(float("nan"))),
            ("z0m", torch.tensor(0.0)),
            ("z", torch.tensor(0.05)),
            ("zeta", torch.tensor(float("inf"))),
        ],
    )
    def test_friction_velocity_rejects_invalid_inputs(self, field, replacement):
        """Test fail-fast validation for the momentum profile."""
        inputs = {
            "U": torch.tensor(5.0),
            "z": torch.tensor(10.0),
            "z0m": torch.tensor(0.1),
            "d": torch.tensor(0.0),
            "zeta": torch.tensor(0.0),
        }
        inputs[field] = replacement

        with pytest.raises((TypeError, ValueError)):
            friction_velocity(**inputs)


class TestTemperatureScale:
    """Test MOST temperature-scale calculations."""

    def test_temperature_scale_includes_roughness_level_correction(self):
        """Test the complete integrated MOST scalar profile."""
        theta_surface = torch.tensor(298.15, dtype=torch.float64)
        theta_air = torch.tensor(293.15, dtype=torch.float64)
        z = torch.tensor(10.0, dtype=torch.float64)
        z0h = torch.tensor(0.01, dtype=torch.float64)
        d = torch.tensor(0.0, dtype=torch.float64)
        zeta_ref = torch.tensor(-0.5, dtype=torch.float64)
        zeta_roughness = zeta_ref * z0h / (z - d)
        profile_factor = (
            torch.log((z - d) / z0h)
            - psi_h(zeta_ref)
            + psi_h(zeta_roughness)
        )
        expected = (
            VON_KARMAN_CONSTANT
            * (theta_surface - theta_air)
            / profile_factor
        )

        actual = temperature_scale(
            theta_surface,
            theta_air,
            z,
            z0h,
            d,
            zeta_ref,
        )

        assert actual.dtype == torch.float64
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TestMoninObukhovLength:
    """Test Monin-Obukhov length calculations."""
    
    def test_L_unstable(self):
        """Test L calculation for unstable conditions."""
        u_star = torch.tensor(0.5)
        rho = 1.225
        cp = 1004.0
        T = torch.tensor(293.15)
        H = torch.tensor(100.0)  # Positive heat flux (unstable)
        
        L = monin_obukhov_length(u_star, rho, cp, T, H)
        
        # L should be negative for unstable conditions
        assert L < 0
        
        # Check calculation
        expected = -u_star**3 * rho * cp * T / (VON_KARMAN_CONSTANT * GRAVITY_ACCELERATION * H)
        assert torch.allclose(L, expected, rtol=1e-5)

    def test_L_exact_reference_value(self):
        """Test the complete rho-cp Obukhov-length expression."""
        actual = monin_obukhov_length(
            torch.tensor(0.5, dtype=torch.float64),
            1.225,
            1004.0,
            torch.tensor(293.15, dtype=torch.float64),
            torch.tensor(100.0, dtype=torch.float64),
        )

        assert actual.item() == pytest.approx(
            -112.0895642242764,
            rel=1e-12,
            abs=1e-12,
        )
    
    def test_L_stable(self):
        """Test L calculation for stable conditions."""
        u_star = torch.tensor(0.3)
        rho = 1.225
        cp = 1004.0
        T = torch.tensor(273.15)
        H = torch.tensor(-50.0)  # Negative heat flux (stable)
        
        L = monin_obukhov_length(u_star, rho, cp, T, H)
        
        # L should be positive for stable conditions
        assert L > 0
    
    def test_L_neutral(self):
        """Test L calculation for near-neutral conditions."""
        u_star = torch.tensor(0.4)
        rho = 1.225
        cp = 1004.0
        T = torch.tensor(288.15)
        H = torch.tensor(1e-6)  # Very small heat flux
        
        L = monin_obukhov_length(u_star, rho, cp, T, H)
        
        # L should be very large for neutral conditions
        assert abs(L) > 100


class TestAerodynamicResistance:
    """Test aerodynamic resistance calculations."""
    
    def test_ra_neutral(self):
        """Test r_a calculation for neutral conditions."""
        u_star = torch.tensor(0.4)
        z = torch.tensor(10.0)
        z0h = torch.tensor(0.01)
        zeta = torch.tensor(0.0)
        
        ra = aerodynamic_resistance(u_star, z, z0h, zeta)
        
        # For neutral scalar transport: rah = ln(z/z0h) / (k * u*).
        ln_h = torch.log(z / z0h)
        expected = ln_h / (VON_KARMAN_CONSTANT * u_star)
        
        assert torch.allclose(ra, expected, rtol=1e-12, atol=1e-12)
        assert ra.item() == pytest.approx(42.12045901818376, rel=1e-12)
        assert ra > 0

    def test_ram_neutral(self):
        """Test neutral momentum resistance and the wind-profile identity."""
        u_star = torch.tensor(0.4)
        z = torch.tensor(10.0)
        z0m = torch.tensor(0.1)
        zeta = torch.tensor(0.0)

        ram = aerodynamic_resistance_momentum(u_star, z, z0m, zeta)

        expected = torch.log(z / z0m) / (VON_KARMAN_CONSTANT * u_star)
        assert torch.allclose(ram, expected, rtol=1e-12, atol=1e-12)
        assert ram.item() == pytest.approx(28.08030601212251, rel=1e-12)
    
    def test_ra_stable(self):
        """Test r_a increases with stability."""
        u_star = torch.tensor(0.3)
        z = torch.tensor(10.0)
        z0h = torch.tensor(0.01)
        
        ra_neutral = aerodynamic_resistance(u_star, z, z0h, torch.tensor(0.0))
        ra_stable = aerodynamic_resistance(u_star, z, z0h, torch.tensor(0.5))
        
        # Resistance should increase with stability
        assert ra_stable > ra_neutral
    
    def test_ra_unstable(self):
        """Test r_a decreases with instability."""
        u_star = torch.tensor(0.5)
        z = torch.tensor(10.0)
        z0h = torch.tensor(0.01)
        
        ra_neutral = aerodynamic_resistance(u_star, z, z0h, torch.tensor(0.0))
        ra_unstable = aerodynamic_resistance(u_star, z, z0h, torch.tensor(-0.5))
        
        # Resistance should decrease with instability
        assert ra_unstable < ra_neutral

    def test_resistance_uses_full_roughness_level_correction(self):
        """Test the integrated MOST profile at both measurement and roughness levels."""
        u_star = torch.tensor(0.4)
        z = torch.tensor(10.0)
        z0h = torch.tensor(0.1)
        z0m = torch.tensor(0.1)
        zeta_value = torch.tensor(0.5)

        roughness_zeta_h = zeta_value * z0h / z
        roughness_zeta_m = zeta_value * z0m / z
        expected_h = (
            torch.log(z / z0h)
            - psi_h(zeta_value)
            + psi_h(roughness_zeta_h)
        ) / (VON_KARMAN_CONSTANT * u_star)
        expected_m = (
            torch.log(z / z0m)
            - psi_m(zeta_value)
            + psi_m(roughness_zeta_m)
        ) / (VON_KARMAN_CONSTANT * u_star)

        actual_h = aerodynamic_resistance(u_star, z, z0h, zeta_value)
        actual_m = aerodynamic_resistance_momentum(
            u_star, z, z0m, zeta_value
        )

        assert torch.allclose(actual_h, expected_h, rtol=1e-12, atol=1e-12)
        assert torch.allclose(actual_m, expected_m, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        ("u_star", "z", "z0h", "zeta_value"),
        [
            (0.0, 10.0, 0.01, 0.0),
            (0.4, 0.01, 0.01, 0.0),
            (0.4, 10.0, 0.0, 0.0),
            (0.4, 10.0, 0.01, float("nan")),
        ],
    )
    def test_scalar_resistance_rejects_invalid_inputs(
        self,
        u_star,
        z,
        z0h,
        zeta_value,
    ):
        """Test fail-fast validation without hidden floors or geometry fallback."""
        with pytest.raises(ValueError):
            aerodynamic_resistance(
                torch.tensor(u_star),
                torch.tensor(z),
                torch.tensor(z0h),
                torch.tensor(zeta_value),
            )


class TestEddyDiffusivity:
    """Test eddy diffusivity calculations."""
    
    def test_K_neutral(self):
        """Test K calculation for neutral conditions."""
        u_star = torch.tensor(0.4)
        z = torch.tensor(5.0)
        psi_h_val = torch.tensor(0.0)  # Neutral
        
        K = eddy_diffusivity(u_star, z, psi_h_val)
        
        # For neutral: K = k * u* * z
        expected = VON_KARMAN_CONSTANT * u_star * z
        
        assert torch.allclose(K, expected, rtol=1e-5)
        assert K > 0
    
    def test_K_stable(self):
        """Test K reduction in stable conditions."""
        u_star = torch.tensor(0.3)
        z = torch.tensor(5.0)
        
        K_neutral = eddy_diffusivity(u_star, z, torch.tensor(0.0))  # zeta = 0 for neutral
        K_stable = eddy_diffusivity(u_star, z, torch.tensor(0.5))  # zeta > 0 for stable
        
        # K should be reduced in stable conditions
        assert K_stable < K_neutral
    
    def test_K_unstable(self):
        """Test K enhancement in unstable conditions."""
        u_star = torch.tensor(0.5)
        z = torch.tensor(5.0)
        
        K_neutral = eddy_diffusivity(u_star, z, torch.tensor(0.0))  # zeta = 0 for neutral
        K_unstable = eddy_diffusivity(u_star, z, torch.tensor(-0.5))  # zeta < 0 for unstable
        
        # K should be enhanced in unstable conditions
        assert K_unstable > K_neutral


class TestRichardsonNumber:
    """Test Richardson number calculations."""
    
    def test_bulk_richardson(self):
        """Test bulk Richardson number calculation."""
        delta_z = torch.tensor(10.0)  # Height difference
        theta_mean = torch.tensor(300.0)  # Mean potential temperature
        delta_theta = torch.tensor(2.0)  # Temperature difference
        delta_U = torch.tensor(5.0)  # Wind speed difference
        
        Ri_b = richardson_number(delta_theta, delta_U, delta_z, theta_mean)
        
        # Ri_b = (g / theta_mean) * (delta_theta / delta_z) / (delta_U / delta_z)^2
        expected = (GRAVITY_ACCELERATION / theta_mean) * (delta_theta / delta_z) / ((delta_U / delta_z)**2)
        
        assert torch.allclose(Ri_b, expected, rtol=1e-5)
    
    def test_zeta_from_richardson_neutral(self):
        """Test z/L from Richardson number for neutral conditions."""
        Ri_b = torch.tensor(0.0)  # Neutral
        
        zeta_val = zeta_from_richardson(Ri_b)
        
        # For Ri_b = 0, expect zeta approximately 0.
        assert abs(zeta_val.item()) < 1e-3
    
    def test_zeta_from_richardson_stable(self):
        """Test z/L from Richardson number for stable conditions."""
        Ri_b = torch.tensor(0.1)  # Stable
        
        zeta_val = zeta_from_richardson(Ri_b)
        
        # For stable conditions, zeta > 0
        assert zeta_val > 0
    
    def test_zeta_from_richardson_unstable(self):
        """Test z/L from Richardson number for unstable conditions."""
        Ri_b = torch.tensor(-0.1)  # Unstable
        
        zeta_val = zeta_from_richardson(Ri_b)
        
        # For unstable conditions, zeta < 0
        assert zeta_val < 0


def _moist_rho(air_temperature):
    """Moist-air density [kg m-3] of the air temperature at p = 101325 Pa and e = 1000 Pa."""
    from relic.scientific.hydrosphere.utils.psychrometrics import compute_moist_air_density

    temperature = torch.as_tensor(air_temperature)
    return compute_moist_air_density(
        temperature,
        torch.full_like(temperature, 101325.0),
        torch.full_like(temperature, 1000.0),
    )


class TestIterativeSolution:
    """Test iterative MOST solution."""
    
    def test_iterative_solution_convergence(self):
        """Test that iterative solution converges."""
        U = torch.tensor(10.0)
        T_air = torch.tensor(293.15)
        T_surface = torch.tensor(298.15)  # Warmer surface (unstable)
        z = torch.tensor(10.0)
        z0m = torch.tensor(0.1)
        z0h = torch.tensor(0.01)
        d = torch.tensor(0.0)  # Zero displacement height
        
        u_star, L, H = iterative_most_solution(
            U, T_air, T_surface, z, z0m, z0h, d,
            max_iterations=20,
            tolerance=1e-3,
            rho=_moist_rho(T_air),
        )
        
        # Check physical consistency
        assert u_star > 0
        assert torch.isfinite(L)
        assert torch.isfinite(H)
        
        # For unstable conditions (surface warmer)
        assert H > 0  # Positive heat flux
        assert L < 0  # Negative L
    
    def test_iterative_solution_neutral(self):
        """Test iterative solution for neutral conditions."""
        U = torch.tensor(10.0)
        T_air = torch.tensor(293.15)
        T_surface = torch.tensor(293.15)  # Same temperature (neutral)
        z = torch.tensor(10.0)
        z0m = torch.tensor(0.1)
        z0h = torch.tensor(0.01)
        d = torch.tensor(0.0)  # Zero displacement height
        
        u_star, L, H = iterative_most_solution(
            U, T_air, T_surface, z, z0m, z0h, d, rho=_moist_rho(T_air)
        )

        # Near-neutral conditions
        assert abs(H) < 10  # Small heat flux
        assert abs(L) > 100  # Large L
    
    def test_iterative_solution_stable(self):
        """Test iterative solution for stable conditions."""
        U = torch.tensor(5.0)
        T_air = torch.tensor(293.15)
        T_surface = torch.tensor(288.15)  # Cooler surface (stable)
        z = torch.tensor(10.0)
        z0m = torch.tensor(0.01)
        z0h = torch.tensor(0.001)
        d = torch.tensor(0.0)  # Zero displacement height
        
        u_star, L, H = iterative_most_solution(
            U, T_air, T_surface, z, z0m, z0h, d, rho=_moist_rho(T_air)
        )

        # For stable conditions (surface cooler)
        assert H < 0  # Negative heat flux
        assert L > 0  # Positive L

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_iterations": 0},
            {"max_iterations": -1},
            {"tolerance": 0.0},
            {"tolerance": -1.0},
            {"tolerance": float("nan")},
            {"tolerance": float("inf")},
        ],
    )
    def test_iterative_solution_rejects_invalid_controls(self, kwargs):
        """Test validation of nonlinear iteration controls."""
        args = (
            torch.tensor(10.0),
            torch.tensor(293.15),
            torch.tensor(298.15),
            torch.tensor(10.0),
            torch.tensor(0.1),
            torch.tensor(0.01),
            torch.tensor(0.0),
        )

        with pytest.raises(ValueError):
            iterative_most_solution(*args, rho=_moist_rho(args[1]), **kwargs)

    def test_iterative_solution_rejects_nonfinite_forcing(self):
        """Test that invalid forcing cannot enter the nonlinear solve."""
        with pytest.raises(ValueError, match="finite"):
            iterative_most_solution(
                torch.tensor(float("nan")),
                torch.tensor(293.15),
                torch.tensor(298.15),
                torch.tensor(10.0),
                torch.tensor(0.1),
                torch.tensor(0.01),
                torch.tensor(0.0),
                rho=_moist_rho(torch.tensor(293.15)),
            )

    def test_iterative_solution_rejects_invalid_geometry(self):
        """Test that the reference height must exceed displacement and roughness."""
        with pytest.raises(ValueError, match="roughness"):
            iterative_most_solution(
                torch.tensor(10.0),
                torch.tensor(293.15),
                torch.tensor(298.15),
                torch.tensor(0.05),
                torch.tensor(0.1),
                torch.tensor(0.01),
                torch.tensor(0.0),
                rho=_moist_rho(torch.tensor(293.15)),
            )

    def test_iterative_solution_raises_when_budget_is_exhausted(self):
        """Test that an unconverged iterate is never returned as a solution."""
        with pytest.raises(RuntimeError, match="MOST iteration failed to converge"):
            iterative_most_solution(
                torch.tensor(10.0),
                torch.tensor(293.15),
                torch.tensor(298.15),
                torch.tensor(10.0),
                torch.tensor(0.1),
                torch.tensor(0.01),
                torch.tensor(0.0),
                max_iterations=1,
                tolerance=1e-12,
                rho=_moist_rho(torch.tensor(293.15)),
            )

    def test_iterative_solution_checks_u_star_and_L_convergence(self, monkeypatch):
        """Test that convergence requires both nonlinear unknowns."""
        fixed_u_star = torch.tensor(0.5)
        sequence = iter(
            [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(1.0)]
        )

        monkeypatch.setattr(
            stability_parameters_module,
            "friction_velocity",
            lambda *args, **kwargs: fixed_u_star,
        )
        monkeypatch.setattr(
            stability_parameters_module,
            "temperature_scale",
            lambda *args, **kwargs: torch.tensor(1.0),
        )
        monkeypatch.setattr(
            stability_parameters_module,
            "monin_obukhov_length",
            lambda *args, **kwargs: next(sequence),
        )

        with pytest.raises(RuntimeError, match="MOST iteration failed to converge"):
            iterative_most_solution(
                torch.tensor(10.0),
                torch.tensor(293.15),
                torch.tensor(298.15),
                torch.tensor(10.0),
                torch.tensor(0.1),
                torch.tensor(0.01),
                torch.tensor(0.0),
                max_iterations=3,
                tolerance=1e-6,
                rho=_moist_rho(torch.tensor(293.15)),
            )

    def test_iterative_solution_rejects_partially_converged_batch(self, monkeypatch):
        """Test that every batch element must converge."""
        U = torch.tensor([10.0, 10.0])
        z = torch.tensor([10.0, 10.0])
        z0m = torch.tensor([0.1, 0.1])
        neutral_u = VON_KARMAN_CONSTANT * U / torch.log(z / z0m)
        friction_sequence = iter(
            [
                neutral_u,
                torch.stack((neutral_u[0], neutral_u[1] * 1.5)),
            ]
        )

        monkeypatch.setattr(
            stability_parameters_module,
            "friction_velocity",
            lambda *args, **kwargs: next(friction_sequence),
        )
        monkeypatch.setattr(
            stability_parameters_module,
            "temperature_scale",
            lambda *args, **kwargs: torch.zeros_like(U),
        )
        monkeypatch.setattr(
            stability_parameters_module,
            "monin_obukhov_length",
            lambda *args, **kwargs: torch.full_like(U, 1000.0),
        )

        with pytest.raises(RuntimeError, match="MOST iteration failed to converge"):
            iterative_most_solution(
                U,
                torch.full_like(U, 293.15),
                torch.full_like(U, 298.15),
                z,
                z0m,
                torch.full_like(U, 0.01),
                torch.zeros_like(U),
                max_iterations=1,
                tolerance=1e-6,
                rho=_moist_rho(torch.full_like(U, 293.15)),
            )


class TestConsistency:
    """Test consistency between different MOST functions."""
    
    def test_stability_function_consistency(self):
        """Test that psi_m and psi_h are consistent."""
        zeta = torch.linspace(-1, 1, 50)
        
        psi_m_vals = psi_m(zeta)
        psi_h_vals = psi_h(zeta)
        
        # For stable conditions, they should be equal
        stable_mask = zeta > 0
        assert torch.allclose(psi_m_vals[stable_mask], 
                            psi_h_vals[stable_mask], 
                            rtol=1e-5)
        
        # For unstable conditions, the relationship is more complex
        # Both integrated corrections are positive under unstable conditions.
        unstable_mask = zeta < 0
        # Both should be positive for unstable conditions
        assert torch.all(psi_m_vals[unstable_mask] > 0)
        assert torch.all(psi_h_vals[unstable_mask] > 0)
    
    def test_reciprocal_relationships(self):
        """Test reciprocal relationships in MOST."""
        # Generate test conditions
        U = torch.tensor(10.0)
        z = torch.tensor(10.0)
        z0m = torch.tensor(0.1)
        z0h = torch.tensor(0.01)
        d = torch.tensor(0.0)
        T = torch.tensor(293.15)
        
        # Calculate u* for neutral conditions
        zeta_neutral = torch.tensor(0.0)
        u_star = friction_velocity(U, z, z0m, d, zeta_neutral)
        
        # Calculate ra
        ra = aerodynamic_resistance(u_star, z, z0h, zeta_neutral)
        
        # Check the neutral wind-profile identity.
        ln_m = torch.log(z / z0m)
        u_star_check = U * VON_KARMAN_CONSTANT / ln_m
        ram = aerodynamic_resistance_momentum(
            u_star, z, z0m, zeta_neutral
        )
        
        assert torch.allclose(u_star, u_star_check, rtol=1e-5)
        assert torch.allclose(ram, U / u_star**2, rtol=1e-12, atol=1e-12)
        assert ra > 0.0


def _hand_moist_air_density(temperature, pressure, vapor_pressure):
    """rho = (p - (1 - eps) e) / (R_d T), eps = M_w / M_d, R_d = R / M_d (Holton 2012 Eq. 2.63-2.64)."""
    from relic.data.standards.unit_converter import unit_converter
    from relic.utils.physical_constants import get_physical_constant

    gas_constant = float(get_physical_constant("relic.physical.constants.universal.gas_constant"))
    dry = float(get_physical_constant("relic.physical.constants.air.molecular_weight_dry"))
    water = float(get_physical_constant("relic.physical.constants.water.molecular_weight"))
    # The SSOT molar masses are in g mol-1; the equation of state uses kg mol-1.
    dry = float(unit_converter.convert(torch.tensor(dry), "molar_mass", "g mol-1", "kg mol-1"))
    water = float(unit_converter.convert(torch.tensor(water), "molar_mass", "g mol-1", "kg mol-1"))
    epsilon = water / dry
    return (pressure - (1.0 - epsilon) * vapor_pressure) / (gas_constant / dry * temperature)


class TestActualAirDensity:
    """rho is the moist-air density of the caller, never an STP default."""

    def test_most_density_argument_has_no_default(self):
        import inspect

        import relic.scientific.atmosphere.most.surface_fluxes as surface_fluxes_module

        for function in (
            surface_fluxes_module.momentum_flux,
            surface_fluxes_module.sensible_heat_flux,
            surface_fluxes_module.latent_heat_flux,
            stability_parameters_module.iterative_most_solution,
        ):
            parameter = inspect.signature(function).parameters["rho"]
            assert parameter.default is inspect.Parameter.empty, function.__name__
        with pytest.raises(TypeError):
            surface_fluxes_module.sensible_heat_flux(torch.tensor([0.3]), torch.tensor([-0.5]))
        for invalid in (0.0, -1.2, float("nan"), torch.tensor([1.2, 1.2, 1.2])):
            with pytest.raises(ValueError):
                surface_fluxes_module.momentum_flux(torch.tensor([0.3, 0.4]), invalid)
        assert not hasattr(surface_fluxes_module, "AIR_DENSITY_STANDARD")
        assert not hasattr(stability_parameters_module, "AIR_DENSITY_STANDARD")
        assert not hasattr(most_module, "AIR_DENSITY_STANDARD")

    def test_sensible_heat_flux_follows_the_moist_air_density(self):
        """H = rho cp u* theta*: 25 degC and 0 degC give different rho, H proportional to it."""
        from relic.scientific.atmosphere.most.surface_fluxes import sensible_heat_flux
        from relic.scientific.hydrosphere.utils.psychrometrics import compute_moist_air_density
        from relic.utils.physical_constants import get_physical_constant

        pressure = torch.tensor([101325.0, 101325.0], dtype=torch.float64)
        vapor_pressure = torch.tensor([1500.0, 500.0], dtype=torch.float64)
        temperature = torch.tensor([298.15, 273.15], dtype=torch.float64)
        rho = compute_moist_air_density(temperature, pressure, vapor_pressure)
        torch.testing.assert_close(
            rho, _hand_moist_air_density(temperature, pressure, vapor_pressure),
            rtol=1.0e-12, atol=0.0,
        )
        assert float(rho[0]) < float(rho[1])
        u_star = torch.tensor([0.3, 0.3], dtype=torch.float64)
        theta_star = torch.tensor([-0.5, -0.5], dtype=torch.float64)
        cp = float(get_physical_constant("air_specific_heat_cp"))
        # References: Stull (1988); Garratt (1992).
        # Mathematical expression: H = rho cp u* theta*.
        # Parameter meanings: rho the moist-air density, cp dry-air heat
        #   capacity, u* friction velocity, theta* temperature scale.
        # Units: rho [kg m-3]; cp [J kg-1 K-1]; u* [m s-1]; theta* [K]; H [W m-2].
        # Domain/constraints: rho > 0 from T > 0, p > 0, 0 <= e < p.
        # Sign convention: theta* = k (theta_s - theta_a) / fh < 0 (air warmer
        #   than the surface) gives a downward, negative H.
        heat = sensible_heat_flux(u_star, theta_star, rho, cp)
        torch.testing.assert_close(heat, rho * cp * u_star * theta_star, rtol=1.0e-12, atol=0.0)
        torch.testing.assert_close(heat[0] / heat[1], rho[0] / rho[1], rtol=1.0e-12, atol=0.0)
        length = monin_obukhov_length(
            u_star, rho, cp, temperature, torch.tensor([100.0, 100.0], dtype=torch.float64)
        )
        expected = -u_star**3 * rho * cp * temperature / (
            VON_KARMAN_CONSTANT * GRAVITY_ACCELERATION * 100.0
        )
        torch.testing.assert_close(length, expected, rtol=1.0e-12, atol=0.0)


def test_k_theory_scalar_type_is_a_required_selection():
    """Round 47: momentum, heat or scalar is chosen explicitly; an unknown type raises."""
    import inspect

    from relic.scientific.atmosphere.most import k_theory_K

    parameter = inspect.signature(k_theory_K).parameters['scalar_type']
    assert parameter.default is inspect.Parameter.empty
    # Unstable surface layer (zeta < 0), where phi_m and phi_h differ.
    u_star, z, zeta = torch.tensor(0.4), torch.tensor(10.0), torch.tensor(-0.5)
    heat = k_theory_K(u_star, z, zeta, scalar_type='heat')
    torch.testing.assert_close(k_theory_K(u_star, z, zeta, scalar_type='scalar'), heat)
    assert not torch.equal(k_theory_K(u_star, z, zeta, scalar_type='momentum'), heat)
    with pytest.raises(ValueError, match="scalar_type"):
        k_theory_K(u_star, z, zeta, scalar_type='hat')


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

