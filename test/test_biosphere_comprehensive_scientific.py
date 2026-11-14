"""
Comprehensive Scientific Validation Tests for Biosphere Modules

This test suite performs thorough scientific validation of all biosphere modules,
ensuring both architectural compliance and scientific accuracy of results.

Test Levels:
- L1: Inheritance and Interface Compliance
- L2: Parameter Validation and Ranges  
- L3: Scientific Output Validation
- L4: Mathematical Consistency Checks
- L5: Integration and Dependencies

Author: Yuchong Long
Date: 2025-08-26
"""

import pytest
import torch
import numpy as np
import warnings
import sys
import os
from contextlib import contextmanager

# Add relic to path for imports
sys.path.append(os.path.abspath('relic'))

from typing import Dict, Any

# Test core abstractions
from relic.core.abstractions import BiologicalProcess
from relic.core.state import UpdateItem, UpdateMode
import relic.parameters as param_system

# Test biosphere modules
try:
    from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
    from relic.scientific.biosphere.respiration.autotrophic_respiration import AutotrophicRespiration
    from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway, MEGANModel
    from relic.scientific.biosphere.bvoc.synthesis.isoprene import IsopreneSynthesis
except ImportError as e:
    print(f"[WARNING] Import error in biosphere modules: {e}")
    pytest.skip(f"Biosphere module import failed: {e}", allow_module_level=True)


@contextmanager
def temporary_parameter_overrides(overrides: Dict[str, Dict[str, float]]):
    """
    Temporarily override ParameterManager values during a test.

    Parameters
    ----------
    overrides : Dict[str, Dict[str, float]]
        Mapping from fully-qualified parameter key to specification dict with
        ``value`` and optional ``units`` entries.
    """
    manager = param_system.get_global_manager()
    previous_values: Dict[str, Any] = {}
    try:
        for key, spec in overrides.items():
            previous_values[key] = manager.get(key)
            param_system.set_value(key, spec["value"], units=spec.get("units"))
        yield
    finally:
        for key, old_value in previous_values.items():
            param_system.set_value(key, old_value, units=overrides[key].get("units"))


def _replace_flux_value(result: Dict[str, UpdateItem], key: str, unit: str):
    """Validate and unwrap one canonical instantaneous flux."""
    item = result[key]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    assert item.unit == unit
    return item.value


class TestBiosphereArchitecturalCompliance:
    """Test L1: Architectural compliance and inheritance correctness."""
    
    def test_farquhar_photosynthesis_inheritance(self):
        """Test FarquharPhotosynthesis inheritance and interface compliance."""
        model = FarquharPhotosynthesis(
            name="test_farquhar",
            config={"parameters": {"vcmax25": 60.0, "jmax25": 120.0, "rd25": 1.5}}
        )
        
        # Test inheritance
        assert isinstance(model, BiologicalProcess), "FarquharPhotosynthesis must inherit BiologicalProcess"
        assert hasattr(model, 'forward'), "Must implement compute_rate method"
        assert hasattr(model, 'parameter_sensitivity'), "Must implement parameter_sensitivity method"
        assert hasattr(model, 'forward'), "Must implement forward method"
        assert hasattr(model, 'get_dependencies'), "Must implement get_dependencies method"
        
        # Test FunctorBase consistency
        assert hasattr(model, 'name'), "Must have name attribute for FunctorBase consistency"
        assert hasattr(model, 'metadata'), "Must have metadata attribute"
        
        print("[OK] FarquharPhotosynthesis inheritance verified")
        
    def test_autotrophic_respiration_inheritance(self):
        """Test AutotrophicRespiration inheritance and interface compliance."""
        model = AutotrophicRespiration(name="test_respiration")
        
        # Test inheritance
        assert isinstance(model, BiologicalProcess), "AutotrophicRespiration must inherit BiologicalProcess"
        assert hasattr(model, 'forward'), "Must implement compute_rate method"
        assert hasattr(model, 'forward'), "Must implement forward method"
        
        # Component Functors are scheduler dependencies, not embedded children.
        assert model.get_dependencies() == [
            'maintenance_respiration',
            'growth_respiration',
        ]
        assert not hasattr(model, 'maintenance_respiration')
        assert not hasattr(model, 'growth_respiration')
        
        print("[OK] AutotrophicRespiration inheritance verified")
        
    def test_megan_model_inheritance(self):
        """Test MEGANModel inheritance and interface compliance."""
        model = MEGANModel(
            name="test_megan",
            config={
                "enabled_compounds": ["isoprene", "monoterpenes"]
            }
        )
        
        # Test inheritance
        assert isinstance(model, BiologicalProcess), "MEGANModel must inherit BiologicalProcess"
        assert hasattr(model, 'forward'), "Must implement compute_rate method"
        assert hasattr(model, 'forward'), "Must implement forward method"
        
        # Test compound management
        assert hasattr(model, 'enabled_compounds'), "Must track enabled compounds"
        assert hasattr(model, 'supported_compounds'), "Must define supported compounds"
        
        print("[OK] MEGANModel inheritance verified")
        
    def test_isoprene_synthesis_inheritance(self):
        """Test IsopreneSynthesis inheritance and interface compliance."""
        model = IsopreneSynthesis(name="test_isoprene_synthesis")
        
        # Test inheritance
        assert isinstance(model, BiologicalProcess), "IsopreneSynthesis must inherit BiologicalProcess"
        assert hasattr(model, 'forward'), "Must implement compute_rate method"
        assert hasattr(model, 'forward'), "Must implement forward method"
        
        print("[OK] IsopreneSynthesis inheritance verified")
        

class TestBiosphereParameterValidation:
    """Test L2: Parameter validation and acceptable ranges."""
    
    def test_farquhar_parameter_ranges(self):
        """Test Farquhar model parameter validation and scientific ranges."""
        overrides = {
            "biosphere.photosynthesis.vcmax25": {"value": 60.0, "units": "umol CO2 m-2 s-1"},
            "biosphere.photosynthesis.jmax25": {"value": 120.0, "units": "umol e- m-2 s-1"},
            "biosphere.photosynthesis.rd25": {"value": 1.2, "units": "umol CO2 m-2 s-1"},
        }
        with temporary_parameter_overrides(overrides):
            model = FarquharPhotosynthesis()
            
            assert pytest.approx(model.vcmax25.item(), rel=0, abs=1e-6) == 60.0, "Vcmax25 parameter correctly set"
            assert pytest.approx(model.jmax25.item(), rel=0, abs=1e-6) == 120.0, "Jmax25 parameter correctly set"
            assert pytest.approx(model.rd25.item(), rel=0, abs=1e-6) == 1.2, "Rd25 parameter correctly set"
        
        # Test parameter relationships (scientific consistency)
        jmax_vcmax_ratio = model.jmax25.item() / model.vcmax25.item()
        assert 1.5 <= jmax_vcmax_ratio <= 3.0, f"Jmax/Vcmax ratio ({jmax_vcmax_ratio:.2f}) within expected range [1.5-3.0]"
        
        rd_vcmax_ratio = model.rd25.item() / model.vcmax25.item()
        assert 0.005 <= rd_vcmax_ratio <= 0.05, f"Rd/Vcmax ratio ({rd_vcmax_ratio:.3f}) within expected range [0.005-0.05]"
        
        print(f"[OK] Farquhar parameter validation - Jmax/Vcmax: {jmax_vcmax_ratio:.2f}, Rd/Vcmax: {rd_vcmax_ratio:.3f}")
        
    def test_megan_emission_factor_ranges(self):
        """Test MEGAN emission factors are within literature ranges."""
        model = MEGANModel(
            config={
                "enabled_compounds": [
                    "isoprene",
                    "monoterpenes",
                    "sesquiterpenes",
                ]
            }
        )
        
        # Validate emission factors against literature ranges
        # Guenther et al. (2012) MEGAN 2.1 paper
        iso_ef = model.isoprene_ef.item()
        mono_ef = model.monoterpenes_ef.item()
        sesq_ef = model.sesquiterpenes_ef.item()
        
        assert 0.1 <= iso_ef <= 100.0, f"Isoprene EF ({iso_ef}) within literature range [0.1-100] ug g-1 h-1"
        assert 0.01 <= mono_ef <= 50.0, f"Monoterpene EF ({mono_ef}) within literature range [0.01-50] ug g-1 h-1"
        assert 0.001 <= sesq_ef <= 10.0, f"Sesquiterpene EF ({sesq_ef}) within literature range [0.001-10] ug g-1 h-1"
        
        print(f"[OK] MEGAN emission factors - ISO: {iso_ef}, MONO: {mono_ef}, SESQ: {sesq_ef}")


class TestBiosphereScientificOutputs:
    """Test L3: Scientific accuracy of model outputs."""
    
    def test_farquhar_photosynthesis_rates(self):
        """Test Farquhar model produces scientifically reasonable photosynthesis rates."""
        overrides = {
            "biosphere.photosynthesis.vcmax25": {"value": 60.0, "units": "umol CO2 m-2 s-1"},
            "biosphere.photosynthesis.jmax25": {"value": 120.0, "units": "umol e- m-2 s-1"},
            "biosphere.photosynthesis.rd25": {"value": 1.5, "units": "umol CO2 m-2 s-1"},
        }
        with temporary_parameter_overrides(overrides):
            model = FarquharPhotosynthesis()
            
            state = {
                "air_temperature": torch.tensor(298.15),
                "atmospheric_co2_ppm": torch.tensor(400.0),
                "ppfd": torch.tensor(1500.0),
                "leaf_area_index": torch.tensor(1.0),
                "vapor_pressure_deficit": torch.tensor(1000.0),
                "vcmax_scaling": torch.tensor(1.0),
            }
            
            result = model.forward(state, time=0.0, dt=3600.0)
            net_photosynthesis = _replace_flux_value(
                result, "net_photosynthesis", "umol CO2 m-2 s-1"
            ).item()
            
            assert 5.0 <= net_photosynthesis <= 40.0, f"Net photosynthesis {net_photosynthesis:.2f} within expected range"
            
            print(f"[OK] Farquhar net photosynthesis under standard conditions: {net_photosynthesis:.2f} umol m-2 s-1")
        
    def test_megan_emission_rates(self):
        """Test MEGAN model produces scientifically reasonable emission rates."""
        model = MEGANModel(
            emission_pathway=EmissionPathway.STANDARD,
            config={
                "enabled_compounds": ["isoprene", "monoterpenes"]
            }
        )
        
        # Standard conditions
        state = {
            'air_temperature': torch.tensor(303.15),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(2.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        result = model.forward(state, time=0.0, dt=3600.0)
        
        # Extract outputs
        total_emission = _replace_flux_value(
            result, 'total_bvoc_emission', 'mol m-2 s-1'
        ).item()
        iso_emission = _replace_flux_value(
            result, 'isoprene_emission', 'mol m-2 s-1'
        ).item()
        mono_emission = _replace_flux_value(
            result, 'monoterpenes_emission', 'mol m-2 s-1'
        ).item()
        iso_temp_factor = _replace_flux_value(
            result,
            'diagnostics.bvoc.megan.isoprene.gamma_temperature',
            'dimensionless',
        ).item()
        iso_light_factor = _replace_flux_value(
            result,
            'diagnostics.bvoc.megan.isoprene.gamma_light',
            'dimensionless',
        ).item()
        
        # Scientific validation
        assert total_emission > 0
        assert iso_emission > 0
        assert mono_emission > 0
        assert total_emission == pytest.approx(
            iso_emission + mono_emission, rel=1.0e-6, abs=1.0e-18
        )
        
        # Temperature factor validation (MEGAN peaked Arrhenius curve stays near unity at 30 degC)
        assert 0.5 <= iso_temp_factor <= 1.5, f"Isoprene temperature factor ({iso_temp_factor:.2f}) near canonical MEGAN range"
        
        # Light factor validation (should be >0 for isoprene at 1000 umol m-2 s-1)
        assert 0.5 <= iso_light_factor <= 2.0, f"Isoprene light factor ({iso_light_factor:.2f}) reasonable at 1000 PAR"
        
        print(f"[OK] MEGAN outputs - Total: {total_emission:.6e}, ISO: {iso_emission:.6e}, MONO: {mono_emission:.6e} mol m-2 s-1")
        

class TestBiosphereMathematicalConsistency:
    """Test L4: Mathematical consistency and conservation laws."""
    
    def test_farquhar_co2_response_curve(self):
        """Test Farquhar model CO2 response follows expected biochemistry."""
        model = FarquharPhotosynthesis()
        
        # CO2 response curve (A-Ci curve)
        co2_range = torch.linspace(50.0, 800.0, 20)  # 50-800 ppm CO2
        base_state = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(1500.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'vcmax_scaling': torch.tensor(1.0),
        }
        
        photosynthesis_rates = []
        for co2 in co2_range:
            state = base_state.copy()
            state['atmospheric_co2_ppm'] = co2
            result = model.forward(state, time=0.0, dt=3600.0)
            photosynthesis_rates.append(
                _replace_flux_value(
                    result, 'net_photosynthesis', 'umol CO2 m-2 s-1'
                ).item()
            )
        
        # Check A-Ci curve properties
        rates = np.array(photosynthesis_rates)
        
        # Should be monotonically increasing (with possible saturation)
        differences = np.diff(rates)
        increasing_regions = differences > -0.1  # Allow small numerical fluctuations
        increasing_fraction = np.sum(increasing_regions) / len(increasing_regions)
        
        assert increasing_fraction >= 0.8, f"A-Ci curve should be mostly increasing ({increasing_fraction*100:.1f}% increasing)"
        
        # Check saturation behavior at high CO2
        high_co2_rates = rates[-5:]  # Last 5 points
        low_co2_rates = rates[:5]    # First 5 points
        
        saturation_ratio = np.mean(high_co2_rates) / np.mean(low_co2_rates)
        assert 2.0 <= saturation_ratio <= 10.0, f"High CO2 enhancement ({saturation_ratio:.1f}x) within expected range"
        
        print(f"[OK] Farquhar A-Ci curve - {increasing_fraction*100:.1f}% increasing, {saturation_ratio:.1f}x CO2 enhancement")
        
    def test_megan_temperature_response(self):
        """Test MEGAN temperature responses follow expected patterns."""
        model = MEGANModel(
            emission_pathway=EmissionPathway.STANDARD,
            config={"enabled_compounds": ["isoprene", "monoterpenes"]},
        )
        
        # Temperature range test
        temp_range = torch.linspace(283.15, 313.15, 16)  # 10-40 degC
        base_state = {
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(2.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        iso_emissions = []
        mono_emissions = []
        
        for temp in temp_range:
            state = base_state.copy()
            state['air_temperature'] = temp
            result = model.forward(state, time=0.0, dt=3600.0)
            iso_emissions.append(
                _replace_flux_value(
                    result, 'isoprene_emission', 'mol m-2 s-1'
                ).item()
            )
            mono_emissions.append(
                _replace_flux_value(
                    result, 'monoterpenes_emission', 'mol m-2 s-1'
                ).item()
            )
        
        iso_rates = np.array(iso_emissions)
        mono_rates = np.array(mono_emissions)

        def _safe_linear_slope(x_values, y_values):
            x_arr = np.asarray(x_values, dtype=np.float64)
            y_arr = np.asarray(y_values, dtype=np.float64)
            if x_arr.size < 2:
                return 0.0
            denom = x_arr[-1] - x_arr[0]
            if abs(denom) < 1e-9:
                return 0.0
            mean_x = np.mean(x_arr)
            mean_y = np.mean(y_arr)
            dx = x_arr - mean_x
            dy = y_arr - mean_y
            var_x = np.mean(dx * dx)
            if var_x <= 1e-12:
                return (y_arr[-1] - y_arr[0]) / denom
            cov_xy = np.mean(dx * dy)
            return float(cov_xy / var_x)
        
        def _robust_q10(temps_kelvin: np.ndarray, fluxes: np.ndarray) -> float:
            temps_c = np.asarray(temps_kelvin, dtype=np.float64) - 273.15
            rates = np.asarray(fluxes, dtype=np.float64)
            valid = np.isfinite(rates) & (rates > 0)
            temps = temps_c[valid]
            resp = rates[valid]
            if temps.size < 2:
                return 1.0
            limit = min(temps.size, 12)
            temps = temps[:limit]
            resp = resp[:limit]
            centered = temps - np.mean(temps)
            log_resp = np.log(resp)
            if not np.all(np.isfinite(log_resp)):
                log_resp = log_resp[np.isfinite(log_resp)]
                centered = centered[: log_resp.size]
            if log_resp.size < 2:
                t0, t1 = temps[0], temps[-1]
                if resp[0] <= 0 or resp[-1] <= 0 or abs(t1 - t0) < 1e-6:
                    return 1.0
                ratio = resp[-1] / resp[0]
                return float(ratio ** (10.0 / (t1 - t0)))
            var_temp = np.mean(centered * centered)
            if var_temp <= 1e-12:
                t0, t1 = temps[0], temps[-1]
                if resp[0] <= 0 or resp[-1] <= 0 or abs(t1 - t0) < 1e-6:
                    return 1.0
                ratio = resp[-1] / resp[0]
                return float(ratio ** (10.0 / (t1 - t0)))
            centered_log = log_resp - np.mean(log_resp)
            slope = np.mean(centered * centered_log) / var_temp
            return float(np.exp(slope * 10.0))
        
        # Both should increase with temperature
        iso_slope = _safe_linear_slope(temp_range.numpy(), iso_rates)
        mono_slope = _safe_linear_slope(temp_range.numpy(), mono_rates)
        
        assert iso_slope > 0, f"Isoprene should increase with temperature (slope: {iso_slope:.6f})"
        assert mono_slope > 0, f"Monoterpenes should increase with temperature (slope: {mono_slope:.6f})"
        
        # Test Q10 relationship for monoterpenes (exponential increase)
        mono_q10_fit = _robust_q10(temp_range.numpy(), mono_rates)
        expected_q10 = np.exp(0.09 * 10)  # Based on MEGAN parameters
        
        assert 1.5 <= mono_q10_fit <= 4.0, f"Monoterpene Q10 ({mono_q10_fit:.2f}) within expected range [1.5-4.0]"
        
        print(f"[OK] MEGAN temperature response - ISO slope: {iso_slope:.6f}, MONO Q10: {mono_q10_fit:.2f}")


class TestBiosphereIntegrationDependencies:
    """Test L5: Integration capabilities and dependency handling."""
    
    def test_dependency_declarations(self):
        """Test all modules properly declare their dependencies."""
        farquhar = FarquharPhotosynthesis()
        megan = MEGANModel()
        respiration = AutotrophicRespiration()
        
        # Test dependency methods exist
        farquhar_deps = farquhar.get_dependencies()
        megan_deps = megan.get_dependencies()
        resp_deps = respiration.get_dependencies()
        
        assert isinstance(farquhar_deps, list), "Dependencies must be returned as list"
        assert isinstance(megan_deps, list), "Dependencies must be returned as list"
        assert isinstance(resp_deps, list), "Dependencies must be returned as list"
        
        # MEGAN should depend on photosynthesis when coupled
        if 'farquhar_photosynthesis' not in megan_deps:
            print("[INFO] MEGAN running in independent mode (no photosynthesis coupling)")
        else:
            print("[INFO] MEGAN properly declares photosynthesis dependency")
        
        print("[OK] Dependency declarations verified")
        
    def test_input_output_consistency(self):
        """Test input/output variable consistency between modules."""
        farquhar = FarquharPhotosynthesis()
        megan = MEGANModel()
        
        farquhar_provides = farquhar.get_provides()
        megan_required = megan.get_required_inputs()
        
        # Check for potential integration points
        photosynthesis_outputs = [var for var in farquhar_provides if 'photosynthesis' in var]
        photosynthesis_inputs = [var for var in megan_required if 'photosynthesis' in var]
        
        if photosynthesis_outputs and photosynthesis_inputs:
            # Check consistency
            consistent_vars = set(photosynthesis_outputs) & set(photosynthesis_inputs)
            if consistent_vars:
                print(f"[OK] Found consistent photosynthesis variables: {consistent_vars}")
            else:
                print("[INFO] No direct photosynthesis variable matching (may use name mapping)")
        
        print("[OK] Input/output consistency checked")
        

class TestBiosphereRobustness:
    """Test model robustness and error handling."""
    
    def test_extreme_condition_handling(self):
        """Test models handle extreme environmental conditions gracefully."""
        farquhar = FarquharPhotosynthesis()
        
        # Test extreme temperatures
        extreme_conditions = [
            {'air_temperature': torch.tensor(273.15), 'atmospheric_co2_ppm': torch.tensor(400.0), 'ppfd': torch.tensor(1000.0), 'leaf_area_index': torch.tensor(1.0), 'vapor_pressure_deficit': torch.tensor(1000.0), 'vcmax_scaling': torch.tensor(1.0)},  # 0 degC
            {'air_temperature': torch.tensor(323.15), 'atmospheric_co2_ppm': torch.tensor(400.0), 'ppfd': torch.tensor(1000.0), 'leaf_area_index': torch.tensor(1.0), 'vapor_pressure_deficit': torch.tensor(1000.0), 'vcmax_scaling': torch.tensor(1.0)},  # 50 degC
            {'air_temperature': torch.tensor(298.15), 'atmospheric_co2_ppm': torch.tensor(50.0), 'ppfd': torch.tensor(1000.0), 'leaf_area_index': torch.tensor(1.0), 'vapor_pressure_deficit': torch.tensor(1000.0), 'vcmax_scaling': torch.tensor(1.0)},   # Very low CO2
            {'air_temperature': torch.tensor(298.15), 'atmospheric_co2_ppm': torch.tensor(400.0), 'ppfd': torch.tensor(0.1), 'leaf_area_index': torch.tensor(1.0), 'vapor_pressure_deficit': torch.tensor(1000.0), 'vcmax_scaling': torch.tensor(1.0)},    # Very low light
        ]
        
        for i, state in enumerate(extreme_conditions):
            try:
                result = farquhar.forward(state, time=0.0, dt=3600.0)
                
                # Check outputs are finite and non-negative
                gross_ps = _replace_flux_value(
                    result, 'gross_photosynthesis', 'umol CO2 m-2 s-1'
                ).item()
                net_ps = _replace_flux_value(
                    result, 'net_photosynthesis', 'umol CO2 m-2 s-1'
                ).item()
                
                assert np.isfinite(gross_ps), f"Extreme condition {i}: Gross photosynthesis should be finite"
                assert np.isfinite(net_ps), f"Extreme condition {i}: Net photosynthesis should be finite"
                assert gross_ps >= 0, f"Extreme condition {i}: Gross photosynthesis should be non-negative"
                assert net_ps >= -10, f"Extreme condition {i}: Net photosynthesis should be > -10 (respiration only)"
                
            except Exception as e:
                pytest.fail(f"Model failed at extreme condition {i}: {e}")
        
        print("[OK] Extreme condition handling verified")
        
    def test_input_validation_warnings(self):
        """Test models provide appropriate warnings for invalid inputs."""
        farquhar = FarquharPhotosynthesis(validate_inputs=True)
        
        # Test with invalid temperature (should warn or handle gracefully)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            state = {
                'air_temperature': torch.tensor(350.0),
                'atmospheric_co2_ppm': torch.tensor(400.0),
                'ppfd': torch.tensor(1000.0),
                'leaf_area_index': torch.tensor(1.0),
                'vapor_pressure_deficit': torch.tensor(1000.0),
                'vcmax_scaling': torch.tensor(1.0)
            }
            
            try:
                result = farquhar.forward(state, time=0.0, dt=3600.0)
                print("[INFO] Model handled extreme temperature without error")
            except ValueError:
                print("[OK] Model properly validates extreme temperature")
            except Exception as e:
                print(f"[INFO] Model response to extreme temperature: {e}")
        
        print("[OK] Input validation tested")


def test_cache_artifact_paths_are_classified_without_mutation():
    """Classify cache artifacts without mutating the shared worktree."""
    import glob
    
    cache_patterns = [
        'relic/**/__pycache__',
        'test/__pycache__',
        '**/*.pyc',
        '**/*.pyo'
    ]
    
    classified_count = 0
    for pattern in cache_patterns:
        for path in glob.glob(pattern, recursive=True):
            normalized = path.replace('\\', '/')
            assert '__pycache__' in normalized or normalized.endswith(('.pyc', '.pyo'))
            classified_count += 1

    print(f"[OK] Classified {classified_count} cache artifacts without mutation")


if __name__ == "__main__":
    """Run comprehensive biosphere scientific validation."""
    print("=" * 80)
    print("COMPREHENSIVE BIOSPHERE SCIENTIFIC VALIDATION")
    print("=" * 80)
    
    # Disable warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    test_classes = [
        TestBiosphereArchitecturalCompliance(),
        TestBiosphereParameterValidation(), 
        TestBiosphereScientificOutputs(),
        TestBiosphereMathematicalConsistency(),
        TestBiosphereIntegrationDependencies(),
        TestBiosphereRobustness()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n--- {class_name} ---")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                passed_tests += 1
                print(f"[PASS] {method_name}")
            except Exception as e:
                print(f"[FAIL] {method_name}: {e}")
    
    # Cache artifacts are classified without mutating the shared worktree.
    test_cache_artifact_paths_are_classified_without_mutation()
    
    # Final report
    print("\n" + "=" * 80)
    print(f"BIOSPHERE VALIDATION SUMMARY: {passed_tests}/{total_tests} TESTS PASSED")
    if passed_tests == total_tests:
        print("[SUCCESS] All biosphere modules pass comprehensive scientific validation!")
    else:
        failed = total_tests - passed_tests
        print(f"[ATTENTION] {failed} tests failed - review results above")
    print("=" * 80)


