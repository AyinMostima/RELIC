#!/usr/bin/env python3
"""
L1 Unit Tests for Hydrosphere Transport Mechanisms

This module provides comprehensive L1 (unit-level) testing for hydrosphere transport
functors including vertical percolation and soil water transport mechanisms.

Test Coverage:
    - VerticalPercolationTransport: Vertical soil water drainage and percolation
    - van Genuchten-Mualem hydraulic functions validation
    - Darcy-Buckingham equation implementation
    - Parameter validation and boundary conditions
    - Numerical stability and conservation checks
    - Configuration and operational modes testing

Scientific Validation:
    - Hydraulic conductivity functions verification
    - Water retention curve validation
    - Physical parameter bounds validation
    - Units consistency verification (kg m-2 s-1)
    - Mass conservation verification
    - Field capacity and saturation thresholds

Test Structure:
    Level 1: Individual component validation
    - Parameter initialization and bounds
    - Core calculation methods (van Genuchten, Mualem)
    - Error handling and edge cases
    - Diagnostic output validation
    - Hydraulic model comparisons

References:
    - van Genuchten, M.T. (1980). Closed-form equation for hydraulic conductivity
    - Mualem, Y. (1976). New model for predicting hydraulic conductivity
    - Darcy, H. (1856). Les Fontaines publiques de la ville de Dijon
    - FAO-56: Allen, R.G. et al. (1998). Crop evapotranspiration
"""

import sys
import os
import pytest
import torch
from typing import Dict, Any, Tuple, Optional

# Add relic to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the modules to test
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport, HydraulicModel
)
from relic.core.state import UpdateItem, UpdateMode


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestVerticalPercolationTransport:
    """Test suite for vertical percolation soil water transport."""

    EXPECTED_OUTPUTS = {
        "drainage_flux",
        "diagnostics.vertical_percolation.effective_saturation",
        "diagnostics.vertical_percolation.hydraulic_conductivity",
        "diagnostics.vertical_percolation.relative_hydraulic_conductivity",
        "diagnostics.vertical_percolation.field_capacity_excess",
        "diagnostics.vertical_percolation.water_balance_residual",
    }

    @pytest.fixture
    def parameter_override(self):
        """Helper to override ParameterManager values and restore after tests."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        original_specs: Dict[str, Tuple[Any, Optional[str]]] = {}

        def _apply(overrides: Dict[str, Tuple[Any, Optional[str], Optional[bool]]]) -> None:
            for key, payload in overrides.items():
                if len(payload) == 3:
                    value, units, validate = payload
                else:
                    value, units = payload
                    validate = True
                spec = manager.get_parameter(key)
                if spec is None:
                    raise KeyError(f"Missing parameter: {key}")
                if key not in original_specs:
                    original_specs[key] = (spec.value, spec.units)
                manager.set_value(key, value, units=units, validate=bool(validate), notify=False)

        yield _apply

        for key, (value, units) in original_specs.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)

    @pytest.fixture
    def percolation_overrides(self, parameter_override):
        """Apply a deterministic set of overrides for percolation tests."""
        overrides = {
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (1e-6, "m s-1", True),
            "hydrosphere.soil_moisture.porosity": (0.43, "m3 m-3", True),
            "hydrosphere.soil_moisture.residual_water_content": (0.08, "m3 m-3", True),
            "hydrosphere.soil_moisture.field_capacity": (0.22, "m3 m-3", True),
            "hydrosphere.soil_moisture.wilting_point": (0.10, "m3 m-3", True),
            "hydrosphere.soil_moisture.alpha_vg": (1.0, "m-1", True),
            "hydrosphere.soil_moisture.n_vg": (1.56, "dimensionless", True),
            "hydrosphere.transport.soil_layer_thickness": (1.0, "m", True),
        }
        parameter_override(overrides)
        return overrides
    
    @pytest.fixture 
    def percolation_functor(self, percolation_overrides):
        """Create vertical percolation functor instance."""
        return VerticalPercolationTransport(
            name="test_vertical_percolation"
        )
    
    @pytest.fixture
    def soil_state(self):
        """Basic soil water state for testing."""
        return {
            "soil_moisture": torch.tensor(0.25),
            "soil_temperature": torch.tensor(293.15),
        }
    
    @pytest.fixture
    def dry_soil_state(self):
        """Dry soil state (below field capacity)."""
        return {
            "soil_moisture": torch.tensor(0.15),
            "soil_temperature": torch.tensor(293.15),
        }
    
    @pytest.fixture
    def saturated_soil_state(self):
        """Nearly saturated soil state."""
        return {
            "soil_moisture": torch.tensor(0.40),
            "soil_temperature": torch.tensor(298.15),
        }

    def test_initialization_default(self):
        """Test default initialization of vertical percolation functor."""
        functor = VerticalPercolationTransport()

        assert functor.name == "vertical_percolation_transport"
        assert functor.hydraulic_model == HydraulicModel.VAN_GENUCHTEN_MUALEM.value
        assert not hasattr(functor, "enabled")
        assert 0.0 < functor.Ks_value < 1.0
        assert 0.0 < functor.theta_s_value <= 1.0
        assert functor.theta_r_value >= 0.0
        assert functor.theta_fc_value > functor.theta_r_value

    def test_initialization_custom_config(self, percolation_overrides):
        """Test initialization with overridden parameters."""
        functor = VerticalPercolationTransport()

        assert functor.hydraulic_model == HydraulicModel.VAN_GENUCHTEN_MUALEM.value
        assert functor.vg_alpha_value == pytest.approx(1.0, rel=1e-6)
        assert functor.vg_n_value == pytest.approx(1.56, rel=1e-6)
        assert functor.soil_layer_thickness == pytest.approx(1.0, rel=1e-6)

    def test_parameter_validation_water_content_hierarchy(self, parameter_override):
        """Test water content hierarchy validation."""
        # Valid hierarchy: theta_r <= theta_wp <= theta_fc <= theta_s <= 1.0
        parameter_override({
            "hydrosphere.soil_moisture.residual_water_content": (0.05, "m3 m-3", True),
            "hydrosphere.soil_moisture.wilting_point": (0.10, "m3 m-3", True),
            "hydrosphere.soil_moisture.field_capacity": (0.22, "m3 m-3", True),
            "hydrosphere.soil_moisture.porosity": (0.43, "m3 m-3", True),
        })
        VerticalPercolationTransport()

        # Invalid hierarchy: field capacity > saturation
        with pytest.raises(ValueError, match="Invalid water content hierarchy"):
            parameter_override({
                "hydrosphere.soil_moisture.residual_water_content": (0.05, "m3 m-3", True),
                "hydrosphere.soil_moisture.wilting_point": (0.10, "m3 m-3", True),
                "hydrosphere.soil_moisture.field_capacity": (0.50, "m3 m-3", True),
                "hydrosphere.soil_moisture.porosity": (0.43, "m3 m-3", True),
            })
            VerticalPercolationTransport()
            
        # Invalid: wilting point < residual
        with pytest.raises(ValueError, match="Invalid water content hierarchy"):
            parameter_override({
                "hydrosphere.soil_moisture.residual_water_content": (0.15, "m3 m-3", True),
                "hydrosphere.soil_moisture.wilting_point": (0.10, "m3 m-3", True),
                "hydrosphere.soil_moisture.field_capacity": (0.22, "m3 m-3", True),
                "hydrosphere.soil_moisture.porosity": (0.43, "m3 m-3", True),
            })
            VerticalPercolationTransport()
    
    def test_van_genuchten_parameters_validation(self, parameter_override):
        """Test van Genuchten parameter validation."""
        # Valid parameters
        parameter_override({
            "hydrosphere.soil_moisture.alpha_vg": (1.0, "m-1", True),
            "hydrosphere.soil_moisture.n_vg": (1.56, "dimensionless", True),
        })
        functor = VerticalPercolationTransport()
        assert functor.vg_m_value == pytest.approx(1.0 - 1.0 / 1.56, rel=1e-6)
        
        # Invalid alpha (negative)
        with pytest.raises(ValueError):
            parameter_override({
                "hydrosphere.soil_moisture.alpha_vg": (-1.0, "m-1", False),
            })
            VerticalPercolationTransport()
            
        # Invalid n (<= 1)
        with pytest.raises(ValueError):
            parameter_override({
                "hydrosphere.soil_moisture.n_vg": (0.8, "dimensionless", False),
            })
            VerticalPercolationTransport()
    
    def test_drainage_flux_bounds_validation(self, parameter_override):
        """Test positive drainage time and geometry validation."""
        parameter_override({
            "hydrosphere.transport.drainage_time_constant": (3600.0, "s", True),
            "hydrosphere.transport.soil_layer_thickness": (1.0, "m", True),
        })
        functor = VerticalPercolationTransport()
        assert functor.drainage_time_constant > 0.0
        assert functor.soil_layer_thickness > 0.0

        parameter_override({
            "hydrosphere.transport.drainage_time_constant": (-1.0, "s", False),
        })
        with pytest.raises(ValueError, match="drainage_time_constant|Drainage time constant"):
            VerticalPercolationTransport()
    
    def test_dependencies(self, percolation_functor):
        """External-state drainage must not depend on alternative closures."""
        deps = percolation_functor.get_dependencies()

        assert deps == {"required": [], "optional": []}
        
    def test_forward_disabled(self, percolation_functor, soil_state):
        """Test that no disabled zero-output mode or input alias remains."""
        assert not hasattr(percolation_functor, "enable_transport")
        with pytest.raises(ValueError, match="soil_moisture"):
            percolation_functor.forward(
                {
                    "volume_fraction_of_water_in_soil": soil_state["soil_moisture"],
                    "soil_temperature": soil_state["soil_temperature"],
                },
                time=0.0,
                dt=3600.0,
            )

    def test_forward_above_field_capacity(self, percolation_functor, soil_state):
        """Test forward pass with soil water above field capacity."""
        output = percolation_functor.forward(soil_state, time=0.0, dt=3600.0)
        drainage = _unwrap_update(output["drainage_flux"])

        assert set(output) == self.EXPECTED_OUTPUTS
        assert drainage > 0
        assert torch.isfinite(drainage)

    def test_forward_below_field_capacity(self, percolation_functor, dry_soil_state):
        """Test forward pass with soil water below field capacity."""
        output = percolation_functor.forward(dry_soil_state, time=0.0, dt=3600.0)
        drainage = _unwrap_update(output["drainage_flux"])

        assert torch.equal(drainage, torch.zeros_like(drainage))

    def test_forward_near_saturation(self, percolation_functor, saturated_soil_state):
        """Test forward pass with soil near saturation."""
        wet = _unwrap_update(
            percolation_functor.forward(saturated_soil_state, 0.0, 1800.0)["drainage_flux"]
        )
        moderate = _unwrap_update(
            percolation_functor.forward(
                {"soil_moisture": torch.tensor(0.25), "soil_temperature": torch.tensor(298.15)},
                0.0,
                1800.0,
            )["drainage_flux"]
        )

        assert torch.isfinite(wet)
        assert wet > moderate > 0.0

    def test_effective_saturation_calculation(self, percolation_functor):
        """Test effective saturation calculation."""
        # Test with known values
        theta = torch.tensor(0.25)  # Water content
        Se = percolation_functor._calculate_effective_saturation(theta)
        
        # Se = (theta - theta_r) / (theta_s - theta_r)
        # Se = (0.25 - 0.08) / (0.43 - 0.08) = 0.17 / 0.35 ~= 0.486
        expected_Se = (0.25 - 0.08) / (0.43 - 0.08)
        assert Se == pytest.approx(expected_Se, rel=1e-3)
        assert 0 <= Se <= 1
        
        # Test boundary conditions
        theta_min = torch.tensor(0.05)  # Below residual
        Se_min = percolation_functor._calculate_effective_saturation(theta_min)
        assert float(Se_min) == pytest.approx(1e-6, rel=1e-3, abs=1e-12)  # Should be clamped to small positive value
        
        theta_max = torch.tensor(0.45)  # Above saturation
        Se_max = percolation_functor._calculate_effective_saturation(theta_max)
        assert Se_max <= 1.0  # Should be clamped to 1.0
    
    def test_van_genuchten_mualem_conductivity(self, percolation_functor):
        """Test van Genuchten-Mualem hydraulic conductivity calculation."""
        # Test with different effective saturations
        Se_values = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        
        for Se in Se_values:
            K = percolation_functor._van_genuchten_mualem_conductivity(Se)
            
            # K should be between 0 and K_s
            assert K >= 0
            assert K <= percolation_functor.Ks
            assert torch.isfinite(K)
            
        # Higher saturation should give higher conductivity
        Se_low = torch.tensor(0.2)
        Se_high = torch.tensor(0.8)
        K_low = percolation_functor._van_genuchten_mualem_conductivity(Se_low)
        K_high = percolation_functor._van_genuchten_mualem_conductivity(Se_high)
        
        assert K_high > K_low, "Higher saturation should give higher conductivity"
    
    def test_hydraulic_model_comparison(self, parameter_override, percolation_overrides):
        """Test different hydraulic models give different results."""
        Se = torch.tensor(0.5)
        temperature = torch.tensor(293.15)

        conductivities = []
        for model in HydraulicModel:
            parameter_override({
                "hydrosphere.transport.vertical_percolation.hydraulic_model": (
                    model.value,
                    "dimensionless",
                    True,
                )
            })
            functor = VerticalPercolationTransport()
            conductivity = functor._calculate_hydraulic_conductivity(Se, temperature)
            conductivities.append(conductivity)

        for conductivity in conductivities:
            K = conductivity
            assert torch.isfinite(K)
            assert K > 0
        values = [round(float(value.detach()), 12) for value in conductivities]
        assert len(set(values)) > 1

    def test_temperature_correction(self, parameter_override, percolation_overrides):
        """Test temperature effects on hydraulic conductivity."""
        Se = torch.tensor(0.6)
        T_cold = torch.tensor(278.15)
        T_warm = torch.tensor(308.15)

        enabled = VerticalPercolationTransport()
        K_cold = enabled._calculate_hydraulic_conductivity(Se, T_cold)
        K_warm = enabled._calculate_hydraulic_conductivity(Se, T_warm)
        assert K_warm > K_cold

        parameter_override({
            "hydrosphere.transport.vertical_percolation.enable_temperature_effects": (
                False,
                "dimensionless",
                True,
            )
        })
        disabled = VerticalPercolationTransport()
        K_cold_no_temp = disabled._calculate_hydraulic_conductivity(Se, T_cold)
        K_warm_no_temp = disabled._calculate_hydraulic_conductivity(Se, T_warm)
        assert torch.allclose(K_cold_no_temp, K_warm_no_temp, rtol=1e-5)

    def test_drainage_flux_calculation(self, percolation_functor):
        """Test drainage flux calculation methodology."""
        theta_dry = torch.tensor(0.15)
        theta_moist = torch.tensor(0.25)
        theta_wet = torch.tensor(0.35)
        K = torch.tensor(1e-6)

        flux_dry = percolation_functor._calculate_drainage_flux(theta_dry, K)
        flux_moist = percolation_functor._calculate_drainage_flux(theta_moist, K)
        flux_wet = percolation_functor._calculate_drainage_flux(theta_wet, K)

        assert flux_dry <= flux_moist <= flux_wet
        assert flux_dry >= 0
        for flux in [flux_dry, flux_moist, flux_wet]:
            assert torch.isfinite(flux)

    def test_subsurface_runoff_generation(self, percolation_functor, saturated_soil_state):
        """Test that lateral runoff is not duplicated by vertical drainage."""
        output = percolation_functor.forward(saturated_soil_state, 0.0, 1800.0)

        assert "drainage_flux" in output
        assert "subsurface_runoff_flux" not in output
        assert not hasattr(percolation_functor, "_calculate_subsurface_runoff")

    def test_diagnostics_output(self, percolation_functor, soil_state):
        """Test diagnostic information output."""
        output = percolation_functor.forward(soil_state, 0.0, 1800.0)

        assert set(output) == self.EXPECTED_OUTPUTS
        for name in self.EXPECTED_OUTPUTS - {"drainage_flux"}:
            assert isinstance(output[name], UpdateItem)
            assert output[name].mode == UpdateMode.REPLACE
            assert output[name].per_timestep is False
            assert torch.isfinite(output[name].value).all()

    def test_boundary_conditions_extreme_water_content(self, percolation_functor):
        """Test physical-domain endpoints and fail-fast behavior."""
        for theta in [percolation_functor.theta_r_value, percolation_functor.theta_s_value]:
            output = percolation_functor.forward(
                {"soil_moisture": torch.tensor(theta), "soil_temperature": torch.tensor(298.15)},
                0.0,
                3600.0,
            )
            assert torch.isfinite(_unwrap_update(output["drainage_flux"]))

        with pytest.raises(ValueError, match="physical range"):
            percolation_functor.forward(
                {"soil_moisture": torch.tensor(0.99), "soil_temperature": torch.tensor(298.15)},
                0.0,
                3600.0,
            )

    def test_mass_flux_units_consistency(self, percolation_functor, soil_state):
        """Test that output fluxes have consistent units (kg m-2 s-1)."""
        output = percolation_functor.forward(soil_state, 0.0, 3600.0)
        update = output["drainage_flux"]

        assert update.unit == "kg m-2 s-1"
        assert update.mode == UpdateMode.REPLACE
        assert update.per_timestep is False
        assert float(update.value.detach()) > 0.0

    def test_parameter_modification_methods(self, percolation_functor):
        """Test that runtime parameter mutation side channels are absent."""
        assert not hasattr(percolation_functor, "enable_transport")
        assert not hasattr(percolation_functor, "set_hydraulic_model")
        assert not hasattr(percolation_functor, "set_soil_parameters")

    def test_forward_interface(self, percolation_functor):
        """Test forward interface for TransportMechanism consistency."""
        state = {
            "soil_moisture": torch.tensor(0.3),
            "soil_temperature": torch.tensor(298.15),
        }
        fluxes = percolation_functor.forward(state, time=0.0, dt=3600.0)

        assert isinstance(fluxes, dict)
        assert set(fluxes) == self.EXPECTED_OUTPUTS
        assert "soil_moisture" not in fluxes
        assert all(isinstance(value, UpdateItem) for value in fluxes.values())


def run_l1_hydrosphere_tests():
    """Run all L1 unit tests for hydrosphere transport."""
    test_files = [__file__]
    
    print("Running L1 Unit Tests for Hydrosphere Transport...")
    print("=" * 60)
    
    # Configure pytest to be verbose and show detailed output
    pytest_args = [
        "-v",  # Verbose
        "-s",  # Don't capture output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
    ] + test_files
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("ALL L1 HYDROSPHERE TRANSPORT TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME L1 TESTS FAILED - Check output above")
        print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    # Run tests when called directly
    exit_code = run_l1_hydrosphere_tests()
    sys.exit(exit_code)
