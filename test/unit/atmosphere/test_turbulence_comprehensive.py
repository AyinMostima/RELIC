"""
Comprehensive L1 Unit Tests for Atmospheric Turbulence Model

This test suite provides exhaustive testing of the TurbulenceFunctor class,
covering all Monin-Obukhov Similarity Theory (MOST) calculations, surface layer turbulence processes,
and scientific accuracy. Tests ensure zero-defect operation and validate against
established literature values for atmospheric turbulence physics.

Test Categories:
1. Basic Functionality Tests
2. MOST Parameter Calculation Tests  
3. Stability Function Tests
4. Friction Velocity and Iterative Solution Tests
5. Aerodynamic Resistance Tests
6. Scientific Accuracy Validation
7. Edge Case and Boundary Tests
8. Error Handling and Robustness Tests

References for validation values:
- Monin & Obukhov (1954). Basic laws of turbulent mixing
- Businger et al. (1971). Flux-profile relationships in atmospheric surface layer
- Garratt (1992). The Atmospheric Boundary Layer
- Stull (1988). An Introduction to Boundary Layer Meteorology
"""

import pytest
import torch
from typing import Dict
import math

import relic.scientific.atmosphere.turbulence as turbulence_module
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor
from relic.scientific.atmosphere.most import psi_h, psi_m
from relic.utils.physical_constants import get_physical_constant
from relic.core.state import UpdateItem, UpdateMode


def _output_tensor(result: Dict[str, UpdateItem], key: str) -> torch.Tensor:
    """Return a tensor from an explicit turbulence state update."""
    item = result[key]
    assert isinstance(item, UpdateItem)
    assert item.mode == UpdateMode.REPLACE
    assert item.per_timestep is True
    assert isinstance(item.value, torch.Tensor)
    return item.value


def _moist_air_density_of(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Moist-air density [kg m-3] of a turbulence input state from the shared owners."""
    from relic.scientific.hydrosphere.utils.psychrometrics import (
        compute_moist_air_density,
        compute_saturation_vapor_pressure,
    )

    # References: Tetens (1930); Allen et al. (1998) FAO-56 Eqs. 11 and 19;
    #   Holton and Hakim (2012) Eqs. 2.63-2.64.
    # Mathematical expression: rho = (p - (1 - eps) RH e_s(Ta)) / (R_d Ta).
    # Parameter meanings: Ta air temperature, p atmospheric pressure, RH relative
    #   humidity, e_s the Tetens saturation vapour pressure of the SSOT constants.
    # Units: Ta [K]; p, e_s [Pa]; RH [1]; rho [kg m-3].
    # Domain/constraints: Ta > 0, p > 0, 0 <= RH e_s < p.
    # Sign convention: rho is a positive state property.
    saturation = compute_saturation_vapor_pressure(
        state['air_temperature'],
        tetens_base_pressure=float(get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_base_pressure")),
        tetens_coeff_a=float(get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_a")),
        tetens_coeff_b=float(get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_b")),
        kelvin_offset=float(get_physical_constant("celsius_to_kelvin")),
    )
    return compute_moist_air_density(
        state['air_temperature'],
        state['atmospheric_pressure'],
        state['relative_humidity'] * saturation,
    )


class TestTurbulenceBasicFunctionality:
    """Test basic functionality and interface compliance for TurbulenceFunctor."""
    
    @pytest.fixture
    def device(self):
        """Test device (CPU or CUDA if available)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_initialization_default_parameters(self, device):
        """Test turbulence functor initialization with default parameters."""
        turbulence = TurbulenceFunctor(device=device)
        
        # Check basic attributes
        assert turbulence.name == "atmospheric_turbulence"
        assert hasattr(turbulence, 'metadata')
        assert hasattr(turbulence, 'config')
        
        # Verify MOST constants use SSOT physical constants
        expected_k = get_physical_constant('von_karman_constant')
        assert turbulence.von_karman_constant == expected_k
        assert turbulence.get_parameter_value('default_reference_height') is not None
        assert turbulence.get_parameter_value('minimum_wind_speed') is not None
        
        # Test parameter bounds
        k = turbulence.von_karman_constant
        assert 0.35 <= k <= 0.45
        
        ref_height = turbulence.get_parameter_value('default_reference_height')
        assert 1.0 <= ref_height <= 100.0
        
    def test_initialization_custom_config(self, device):
        """Test that config cannot bypass ParameterManager YAML."""
        config = {
            'parameters': {
                'von_karman_constant': {'value': 0.41, 'units': 'dimensionless'},
                'default_reference_height': {'value': 15.0, 'units': 'm'},
                'roughness_heat_ratio': {'value': 0.2, 'units': 'dimensionless'}
            }
        }
        
        with pytest.raises(ValueError, match="ParameterManager YAML"):
            TurbulenceFunctor(config=config, device=device)
        
    def test_device_consistency(self, device):
        """Test CPU/GPU device consistency."""
        turbulence = TurbulenceFunctor(device=device)
        
        # Check device is handled properly (may not be stored as attribute)
        # Test works by running calculations successfully on the device
        state = {
            'wind_speed': torch.tensor(5.0, device=device),
            'air_temperature': torch.tensor(298.15, device=device),
            'surface_temperature': torch.tensor(300.15, device=device),
            'reference_height': torch.tensor(10.0, device=device),
            'roughness_length_momentum': torch.tensor(0.1, device=device),
            'displacement_height': torch.tensor(0.0, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),
            'relative_humidity': torch.tensor(0.5, device=device),
        }
        result = turbulence.forward(state, time=43200.0, dt=60.0)
        
        # Check outputs are on correct device - handle device index differences
        assert _output_tensor(result, 'friction_velocity').device.type == device.type
        assert _output_tensor(result, 'aerodynamic_resistance').device.type == device.type
            
    def test_dependencies_interface(self, device):
        """Test get_dependencies interface compliance."""
        turbulence = TurbulenceFunctor(device=device)
        dependencies = turbulence.get_dependencies()
        
        assert isinstance(dependencies, list)
        # Turbulence may have dependencies on radiation for heat flux calculations
        
    def test_functor_metadata(self, device):
        """Test functor metadata is properly set."""
        turbulence = TurbulenceFunctor(device=device)
        metadata = turbulence.metadata
        
        assert metadata is not None
        assert metadata.name == "atmospheric_turbulence"
        assert "monin-obukhov" in metadata.description.lower()
        assert "friction_velocity" in metadata.output_variables
        assert "aerodynamic_resistance" in metadata.output_variables
        assert set(metadata.conservation_laws) == {"momentum_boundary"}


class TestTurbulenceFunctorIntegration:
    """Test full TurbulenceFunctor forward integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.turbulence = TurbulenceFunctor()
        
        # Standard test conditions
        self.standard_state = {
            'wind_speed': torch.tensor(5.0),
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
    def test_forward_basic_functionality(self):
        """Test basic forward method functionality."""
        result = self.turbulence.forward(self.standard_state, time=43200.0, dt=60.0)
        
        # Check all expected outputs
        required_outputs = [
            'friction_velocity', 'aerodynamic_resistance',
            'diagnostics.turbulence.stability_parameter',
            'monin_obukhov_length', 'sensible_heat_flux', 'momentum_flux',
            'diagnostics.turbulence.drag_coefficient',
        ]

        assert set(result) == set(required_outputs)
        for output in required_outputs:
            assert torch.all(torch.isfinite(_output_tensor(result, output)))
        assert result['friction_velocity'].unit == 'm s-1'
        assert result['aerodynamic_resistance'].unit == 's m-1'
        assert result['momentum_flux'].unit == 'kg m-1 s-2'

    @staticmethod
    def _replace_update_value(result, key, value):
        """Replace one UpdateItem value while preserving its runtime contract."""
        item = result[key]
        result[key] = UpdateItem(
            value=value,
            mode=item.mode,
            unit=item.unit,
            per_timestep=item.per_timestep,
            source_functor=item.source_functor,
        )

    def test_open_momentum_boundary_conservation(self):
        """Surface stress must equal air density times friction velocity squared."""
        result = self.turbulence.forward(
            self.standard_state,
            time=43200.0,
            dt=60.0,
        )

        assert self.turbulence.check_conservation_laws(
            self.standard_state,
            result,
        )["momentum_boundary"] == pytest.approx(0.0, abs=1.0e-12)

        tampered = dict(result)
        momentum_flux = _output_tensor(tampered, "momentum_flux")
        self._replace_update_value(
            tampered,
            "momentum_flux",
            momentum_flux * momentum_flux.new_tensor(1.1),
        )
        assert self.turbulence.check_conservation_laws(
            self.standard_state,
            tampered,
        )["momentum_boundary"] == pytest.approx(
            1.0 / 11.0,
            rel=1.0e-6,
        )

    @pytest.mark.parametrize(
        ("key", "invalid_value", "error_type", "message"),
        [
            (
                "friction_velocity",
                1.0,
                TypeError,
                "must be a torch.Tensor",
            ),
            (
                "friction_velocity",
                torch.empty(0),
                ValueError,
                "must be non-empty",
            ),
            (
                "friction_velocity",
                torch.tensor(1, dtype=torch.int64),
                TypeError,
                "must be floating",
            ),
            (
                "friction_velocity",
                torch.tensor(float("nan")),
                ValueError,
                "must be finite",
            ),
            (
                "momentum_flux",
                torch.tensor(float("inf")),
                ValueError,
                "must be finite",
            ),
        ],
    )
    def test_momentum_boundary_rejects_invalid_outputs(
        self,
        key,
        invalid_value,
        error_type,
        message,
    ):
        """Invalid momentum evidence must fail before residual evaluation."""
        result = self.turbulence.forward(
            self.standard_state,
            time=43200.0,
            dt=60.0,
        )
        self._replace_update_value(result, key, invalid_value)

        with pytest.raises(error_type, match=message):
            self.turbulence.check_conservation_laws(
                self.standard_state,
                result,
            )

    @pytest.mark.parametrize(
        "missing_key",
        ["friction_velocity", "momentum_flux"],
    )
    def test_momentum_boundary_requires_both_open_ports(self, missing_key):
        """Both sides of the open momentum relation are mandatory."""
        result = self.turbulence.forward(
            self.standard_state,
            time=43200.0,
            dt=60.0,
        )
        result.pop(missing_key)

        with pytest.raises(KeyError, match=missing_key):
            self.turbulence.check_conservation_laws(
                self.standard_state,
                result,
            )

    def test_momentum_boundary_requires_matching_shapes(self):
        """Friction velocity and surface stress must have identical shapes."""
        result = self.turbulence.forward(
            self.standard_state,
            time=43200.0,
            dt=60.0,
        )
        momentum_flux = _output_tensor(result, "momentum_flux")
        self._replace_update_value(
            result,
            "momentum_flux",
            momentum_flux.reshape(1).expand(2).clone(),
        )

        with pytest.raises(ValueError, match="identical shapes"):
            self.turbulence.check_conservation_laws(
                self.standard_state,
                result,
            )
            
    def test_forward_with_missing_surface_temperature(self):
        """Test missing surface temperature fails fast."""
        state = self.standard_state.copy()
        del state['surface_temperature']

        with pytest.raises(ValueError, match="surface_temperature"):
            self.turbulence.forward(state, time=43200.0, dt=60.0)
        
    def test_forward_with_missing_geometry(self):
        """Test missing geometry cannot use hidden defaults."""
        state = {
            'wind_speed': torch.tensor(4.0),
            'air_temperature': torch.tensor(295.15),
            'surface_temperature': torch.tensor(297.15)
        }
        
        with pytest.raises(ValueError, match="reference_height"):
            self.turbulence.forward(state, time=43200.0, dt=60.0)
        
    def test_forward_diagnostics_output(self):
        """Test forward method with diagnostic outputs."""
        result = self.turbulence.forward(
            self.standard_state, time=43200.0, dt=60.0, return_diagnostics=True
        )

        assert 'diagnostics.turbulence.stability_parameter' in result
        assert 'diagnostics.turbulence.drag_coefficient' in result
        assert 'stability_function_momentum' not in result
        assert 'heat_flux' not in result

    def test_forward_rejects_mixed_input_dtype(self):
        """Test that the Functor does not silently cast scientific forcing."""
        state = self.standard_state.copy()
        base_dtype = state['wind_speed'].dtype
        mixed_dtype = torch.float32 if base_dtype == torch.float64 else torch.float64
        state['surface_temperature'] = torch.tensor(300.15, dtype=mixed_dtype)

        with pytest.raises(ValueError, match="dtype and device"):
            self.turbulence.forward(state, time=43200.0, dt=60.0)


class TestScientificAccuracyValidation:
    """Test against established literature values and MOST theory."""
    
    def setup_method(self):
        """Set up standard test conditions."""
        self.turbulence = TurbulenceFunctor()
        
        # Standard neutral conditions for validation
        self.neutral_conditions = {
            'wind_speed': torch.tensor(5.0),          # 5 m/s
            'air_temperature': torch.tensor(298.15),   # 25 C
            'surface_temperature': torch.tensor(298.15), # Same as air (neutral)
            'reference_height': torch.tensor(10.0),     # 10 m
            'roughness_length_momentum': torch.tensor(0.1), # 0.1 m
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        # Unstable conditions
        self.unstable_conditions = {
            'wind_speed': torch.tensor(3.0),
            'air_temperature': torch.tensor(295.15),   # Cooler air
            'surface_temperature': torch.tensor(300.15), # Warmer surface
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
    def test_neutral_condition_validation(self):
        """Test MOST calculations under neutral conditions."""
        result = self.turbulence.forward(self.neutral_conditions, time=43200.0, dt=60.0)
        
        u_star = _output_tensor(result, 'friction_velocity').item()
        ra = _output_tensor(result, 'aerodynamic_resistance').item()
        L = _output_tensor(result, 'monin_obukhov_length').item()
        
        # References: Paulson (1970); Businger et al. (1971).
        # Expressions: u*=kU/fm and rah=fh/(k u*) with roughness corrections.
        # Parameters: z/z0/L [1], U/u* [m s-1], and rah [s m-1].
        k = get_physical_constant("von_karman_constant")
        zeta_ref = 10.0 / L
        z0m_over_l = 0.1 / L
        z0h_over_l = 0.01 / L
        zeta_tensor = torch.tensor(zeta_ref)
        fm = (
            math.log(10.0 / 0.1)
            - psi_m(zeta_tensor).item()
            + psi_m(torch.tensor(z0m_over_l)).item()
        )
        fh = (
            math.log(10.0 / 0.01)
            - psi_h(zeta_tensor).item()
            + psi_h(torch.tensor(z0h_over_l)).item()
        )
        expected_u_star = k * 5.0 / fm
        expected_ra = fh / (k * u_star)
        assert u_star == pytest.approx(expected_u_star, rel=1e-6)
        assert ra == pytest.approx(expected_ra, rel=1e-6)
        
        # L should be large for neutral conditions
        assert abs(L) > 100.0
        
    def test_unstable_condition_validation(self):
        """Test MOST calculations under unstable conditions."""
        result = self.turbulence.forward(self.unstable_conditions, time=43200.0, dt=60.0)
        
        L = _output_tensor(result, 'monin_obukhov_length').item()
        H = _output_tensor(result, 'sensible_heat_flux').item()
        stability_param = _output_tensor(
            result, 'diagnostics.turbulence.stability_parameter'
        ).item()
        
        # For unstable conditions
        assert L < 0, "L should be negative for unstable conditions"
        assert H > 0, "Heat flux should be positive (upward) for unstable conditions"
        assert stability_param < 0, "z/L should be negative for unstable conditions"
        
        u_star = _output_tensor(result, 'friction_velocity').item()
        rho = float(_moist_air_density_of(self.unstable_conditions))
        cp = get_physical_constant("air_specific_heat_cp")
        k = get_physical_constant("von_karman_constant")
        g = get_physical_constant("gravity_acceleration")
        expected_l = -(rho * cp * 295.15 * u_star**3) / (k * g * H)
        assert L == pytest.approx(expected_l, rel=1e-6)
        assert 0.0 <= H <= 250.0  # Reasonable heat flux range for moderate instability
        
    def test_drag_coefficient_validation(self):
        """Test neutral drag coefficient calculation."""
        result = self.turbulence.forward(self.neutral_conditions, time=43200.0, dt=60.0)
        
        CD = _output_tensor(
            result, 'diagnostics.turbulence.drag_coefficient'
        ).item()
        
        # Neutral drag coefficient: CD = (k/ln(z/z0))^2.
        # CD is approximately 0.0038 for this case.
        k = 0.4
        ln_term = math.log(10.0 / 0.1)
        expected_CD = (k / ln_term)**2
        
        assert 0.8 * expected_CD <= CD <= 1.2 * expected_CD
        
    def test_von_karman_constant_consistency(self):
        """Test consistency with Von Karman constant."""
        # Config overrides must not bypass SSOT physical constants.
        config_low_k = {
            'parameters': {
                'von_karman_constant': {'value': 0.35}
            }
        }
        
        config_high_k = {
            'parameters': {
                'von_karman_constant': {'value': 0.45}
            }
        }
        
        with pytest.raises(ValueError, match="ParameterManager YAML"):
            TurbulenceFunctor(config=config_low_k)
        with pytest.raises(ValueError, match="ParameterManager YAML"):
            TurbulenceFunctor(config=config_high_k)
        
    def test_roughness_length_scaling(self):
        """Test proper scaling with roughness length changes."""
        # Smooth surface
        smooth_conditions = self.neutral_conditions.copy()
        smooth_conditions['roughness_length_momentum'] = torch.tensor(0.001)
        
        # Rough surface  
        rough_conditions = self.neutral_conditions.copy()
        rough_conditions['roughness_length_momentum'] = torch.tensor(1.0)
        
        result_smooth = self.turbulence.forward(smooth_conditions, time=43200.0, dt=60.0)
        result_rough = self.turbulence.forward(rough_conditions, time=43200.0, dt=60.0)
        
        # Smoother surface should have higher resistance
        assert _output_tensor(
            result_smooth, 'aerodynamic_resistance'
        ) > _output_tensor(result_rough, 'aerodynamic_resistance')
        
        # Rougher surface should have higher friction velocity for same wind
        assert _output_tensor(
            result_rough, 'friction_velocity'
        ) > _output_tensor(result_smooth, 'friction_velocity')


class TestEdgeCasesAndBoundaryConditions:
    """Test model behavior under extreme and boundary conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.turbulence = TurbulenceFunctor()
        
    def test_very_low_wind_speed(self):
        """Test wind below the declared MOST floor fails fast."""
        low_wind_state = {
            'wind_speed': torch.tensor(0.01),        # Very low wind
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(299.15),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }

        with pytest.raises(ValueError, match="minimum_wind_speed"):
            self.turbulence.forward(low_wind_state, time=43200.0, dt=60.0)
        
    def test_extreme_temperature_differences(self):
        """Reject forcing outside the configured Businger-Dyer domain."""
        very_unstable_state = {
            'wind_speed': torch.tensor(2.0),
            'air_temperature': torch.tensor(283.15),
            'surface_temperature': torch.tensor(313.15),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        with pytest.raises(
            ValueError,
            match="outside the configured Businger-Dyer validity domain",
        ):
            self.turbulence.forward(very_unstable_state, time=43200.0, dt=60.0)
        
    def test_very_rough_surface(self):
        """Test behavior with very rough surfaces."""
        rough_surface_state = {
            'wind_speed': torch.tensor(8.0),
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'reference_height': torch.tensor(15.0),      # Higher to avoid z < z0
            'roughness_length_momentum': torch.tensor(2.0), # Very rough
            'displacement_height': torch.tensor(5.0),      # With displacement
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        result = self.turbulence.forward(rough_surface_state, time=43200.0, dt=60.0)
        
        # Should produce finite results even for rough surfaces
        assert torch.isfinite(_output_tensor(result, 'friction_velocity'))
        assert torch.isfinite(_output_tensor(result, 'aerodynamic_resistance'))
        assert _output_tensor(result, 'friction_velocity').item() > 0
        
    def test_height_roughness_constraints(self):
        """Test handling when reference height is too close to roughness."""
        constraint_state = {
            'wind_speed': torch.tensor(5.0),
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'reference_height': torch.tensor(0.4),
            'roughness_length_momentum': torch.tensor(0.4),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        with pytest.raises(ValueError, match="must exceed"):
            self.turbulence.forward(constraint_state, time=43200.0, dt=60.0)


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness under various conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.turbulence = TurbulenceFunctor()
        
    def test_missing_required_inputs(self):
        """Test error handling for missing required inputs."""
        incomplete_state = {
            'air_temperature': torch.tensor(298.15)
            # Missing wind_speed
        }
        
        with pytest.raises(ValueError, match="Missing required turbulence state inputs"):
            self.turbulence.forward(incomplete_state, time=43200.0, dt=60.0)
            
        incomplete_state2 = {
            'wind_speed': torch.tensor(5.0)
            # Missing air_temperature
        }
        
        with pytest.raises(ValueError, match="Missing required turbulence state inputs"):
            self.turbulence.forward(incomplete_state2, time=43200.0, dt=60.0)
            
    def test_invalid_parameter_ranges(self):
        """Test handling of invalid parameter values."""
        # Test with configuration containing invalid values
        invalid_config = {
            'parameters': {
                'von_karman_constant': {'value': 1.5}  # Too high
            }
        }
        
        with pytest.raises(ValueError, match="ParameterManager YAML"):
            TurbulenceFunctor(config=invalid_config)
            
    def test_negative_input_handling(self):
        """Test handling of negative input values."""
        negative_state = {
            'wind_speed': torch.tensor(-2.0),        # Negative wind speed
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }

        with pytest.raises(ValueError):
            self.turbulence.forward(negative_state, time=43200.0, dt=60.0)
            
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        nan_state = {
            'wind_speed': torch.tensor(float('nan')),
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }

        with pytest.raises(ValueError, match="finite"):
            self.turbulence.forward(nan_state, time=43200.0, dt=60.0)
            
    def test_zero_temperature_difference_handling(self):
        """Test handling when air and surface temperatures are identical."""
        neutral_state = {
            'wind_speed': torch.tensor(5.0),
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(298.15),  # Exactly same
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        result = self.turbulence.forward(neutral_state, time=43200.0, dt=60.0)
        
        # Should handle neutral conditions gracefully
        assert torch.isfinite(_output_tensor(result, 'monin_obukhov_length'))
        assert torch.isfinite(_output_tensor(result, 'sensible_heat_flux'))
        assert abs(_output_tensor(result, 'sensible_heat_flux').item()) < 10.0


class TestCanonicalConstruction:
    """Test the single canonical TurbulenceFunctor construction path."""
    
    def test_heat_transfer_transport_creation(self):
        """Test default construction uses the canonical writer name."""
        turbulence = TurbulenceFunctor()
        assert turbulence.name == "atmospheric_turbulence"
        
    def test_momentum_transport_creation(self):
        """Test the removed transport-mode factory is not exported."""
        assert not hasattr(turbulence_module, "create_turbulent_transport")
        
    def test_scalar_transport_creation(self):
        """Test the module exports only the canonical class."""
        assert turbulence_module.__all__ == ["TurbulenceFunctor"]
        
    def test_custom_parameters_in_factory(self):
        """Test parameter overrides are rejected on the canonical constructor."""
        with pytest.raises(ValueError, match="ParameterManager YAML"):
            TurbulenceFunctor(config={"parameters": {"max_iterations": 3}})
        
    def test_invalid_transport_mode(self):
        """Test declared dependencies remain the only accepted config content."""
        turbulence = TurbulenceFunctor(config={"dependencies": ["meteorology"]})
        assert turbulence.get_dependencies() == ["meteorology"]

    def test_private_most_formula_methods_are_removed(self):
        """Test that TurbulenceFunctor is only a canonical MOST lifecycle shell."""
        turbulence = TurbulenceFunctor()
        assert not hasattr(turbulence, "_calculate_stability_functions")
        assert not hasattr(turbulence, "_calculate_friction_velocity_iterative")
        assert not hasattr(turbulence, "_calculate_aerodynamic_resistance")

    def test_forward_delegates_once_to_canonical_most(self, monkeypatch):
        """Test exact delegation without a second formula or output floor."""
        calls = {"solve": 0, "zeta": 0, "resistance": 0, "momentum": 0}

        def fake_solver(*args, **kwargs):
            calls["solve"] += 1
            assert torch.allclose(args[5], args[4] * 0.1)
            assert kwargs["max_iterations"] == 10
            assert kwargs["tolerance"] == pytest.approx(0.01)
            torch.testing.assert_close(kwargs["rho"], _moist_air_density_of(state))
            return (
                args[0].new_tensor(0.005),
                args[0].new_tensor(-20.0),
                args[0].new_tensor(25.0),
            )

        def fake_zeta(*args, **kwargs):
            calls["zeta"] += 1
            return args[0].new_tensor(-0.5)

        def fake_resistance(*args, **kwargs):
            calls["resistance"] += 1
            return args[0].new_tensor(12.5)

        def fake_momentum(*args, **kwargs):
            calls["momentum"] += 1
            torch.testing.assert_close(args[1], _moist_air_density_of(state))
            return args[0].new_tensor(0.25)

        monkeypatch.setattr(
            turbulence_module,
            "iterative_most_solution",
            fake_solver,
            raising=False,
        )
        monkeypatch.setattr(
            turbulence_module,
            "most_zeta",
            fake_zeta,
            raising=False,
        )
        monkeypatch.setattr(
            turbulence_module,
            "most_aerodynamic_resistance",
            fake_resistance,
            raising=False,
        )
        monkeypatch.setattr(
            turbulence_module,
            "most_momentum_flux",
            fake_momentum,
            raising=False,
        )

        state = {
            'wind_speed': torch.tensor(5.0),
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        result = TurbulenceFunctor().forward(state, time=0.0, dt=60.0)

        assert calls == {"solve": 1, "zeta": 1, "resistance": 1, "momentum": 1}
        assert _output_tensor(result, 'friction_velocity').item() == pytest.approx(0.005)
        assert _output_tensor(result, 'aerodynamic_resistance').item() == pytest.approx(12.5)
        assert _output_tensor(result, 'momentum_flux').item() == pytest.approx(0.25)


# Integration with pytest fixtures and parameterized tests
@pytest.fixture(scope="module")
def turbulence_model():
    """Pytest fixture for Turbulence model."""
    return TurbulenceFunctor()


@pytest.fixture(scope="module") 
def standard_atmospheric_conditions():
    """Pytest fixture for standard atmospheric test conditions."""
    return {
        'wind_speed': torch.tensor(5.0),
        'air_temperature': torch.tensor(298.15),
        'surface_temperature': torch.tensor(300.15),
        'reference_height': torch.tensor(10.0),
        'roughness_length_momentum': torch.tensor(0.1),
        'displacement_height': torch.tensor(0.0),
        'atmospheric_pressure': torch.tensor(101325.0),
        'relative_humidity': torch.tensor(0.5),
    }


@pytest.mark.parametrize(
    "wind_speed,within_domain",
    [(1.0, False), (3.0, True), (5.0, True), (8.0, True)],
)
def test_wind_speed_sweep(
    turbulence_model,
    standard_atmospheric_conditions,
    wind_speed,
    within_domain,
):
    """Resolve valid winds and reject winds outside the selected closure domain."""
    conditions = standard_atmospheric_conditions.copy()
    conditions['wind_speed'] = torch.tensor(wind_speed)

    if not within_domain:
        with pytest.raises(
            ValueError,
            match="outside the configured Businger-Dyer validity domain",
        ):
            turbulence_model.forward(conditions, time=43200.0, dt=60.0)
        return

    result = turbulence_model.forward(conditions, time=43200.0, dt=60.0)
    
    # All outputs should be finite and physically reasonable
    assert torch.isfinite(_output_tensor(result, 'friction_velocity'))
    assert torch.isfinite(_output_tensor(result, 'aerodynamic_resistance'))
    assert _output_tensor(result, 'friction_velocity').item() > 0
    assert _output_tensor(result, 'aerodynamic_resistance').item() > 0


@pytest.mark.parametrize("stability_type,air_temp,surf_temp", [
    ("stable", 298.15, 293.15),      # Surface 5K cooler
    ("neutral", 298.15, 298.15),     # Same temperature
    ("unstable", 298.15, 303.15),    # Surface 5K warmer
    ("very_unstable", 298.15, 308.15) # Surface 10K warmer
])
def test_stability_regime_sweep(turbulence_model, standard_atmospheric_conditions, stability_type, air_temp, surf_temp):
    """Parametrized test across atmospheric stability regimes."""
    conditions = standard_atmospheric_conditions.copy()
    conditions['air_temperature'] = torch.tensor(air_temp)
    conditions['surface_temperature'] = torch.tensor(surf_temp)
    
    result = turbulence_model.forward(conditions, time=43200.0, dt=60.0)
    
    # Check stability-dependent behavior
    L = _output_tensor(result, 'monin_obukhov_length').item()
    H = _output_tensor(result, 'sensible_heat_flux').item()
    
    if stability_type == "stable":
        assert L > 0, "L should be positive for stable conditions"
        assert H < 0, "Heat flux should be negative (downward) for stable conditions"
    elif stability_type == "neutral":
        assert abs(L) > 50.0, "L should be large magnitude for neutral conditions"
        assert abs(H) < 50.0, "Heat flux should be small for neutral conditions"
    elif stability_type in ["unstable", "very_unstable"]:
        assert L < 0, "L should be negative for unstable conditions"
        assert H > 0, "Heat flux should be positive (upward) for unstable conditions"
    
    # All results should be finite
    assert torch.isfinite(_output_tensor(result, 'friction_velocity'))
    assert torch.isfinite(_output_tensor(result, 'aerodynamic_resistance'))


@pytest.mark.parametrize("roughness_length", [0.001, 0.01, 0.1, 0.5])
def test_roughness_length_sweep(turbulence_model, standard_atmospheric_conditions, roughness_length):
    """Parametrized test across surface roughness range."""
    conditions = standard_atmospheric_conditions.copy()
    conditions['roughness_length_momentum'] = torch.tensor(roughness_length)
    
    result = turbulence_model.forward(conditions, time=43200.0, dt=60.0)
    
    assert torch.isfinite(_output_tensor(result, 'friction_velocity'))
    assert torch.isfinite(_output_tensor(result, 'aerodynamic_resistance'))
    assert _output_tensor(
        result, 'diagnostics.turbulence.drag_coefficient'
    ).item() > 0
    
    # Higher roughness should give higher friction velocity (for same wind)
    u_star = _output_tensor(result, 'friction_velocity').item()
    assert u_star > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])


