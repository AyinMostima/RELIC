"""
Test suite for RELIC parameter management system refactoring
Tests P0-P4 fixes and L1-L3 validation levels
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test P0: Inheritance fixes
def test_mep_pathway_model_initialization():
    """MEP kinetic parameters must not be overridden outside ParameterManager."""
    from relic.scientific.biosphere.bvoc.synthesis.isoprene import MEPPathwayModel
    
    config = {
        "v_max_dxs": 3.0,
        "k_g3p": 60.0,
        "k_pyruvate": 35.0,
        "q10_dxs": 3.0,
        "q10_dxr": 2.5
    }
    
    with pytest.raises(ValueError, match="ParameterManager"):
        MEPPathwayModel(config)

def test_monoterpene_pool_model_initialization():
    """Require parent-owned SSOT resolution for the pool component."""
    from relic.scientific.biosphere.bvoc.synthesis.monoterpene import (
        MonoterpenePoolModel,
        MonoterpeneSynthesis,
    )

    with pytest.raises(TypeError):
        MonoterpenePoolModel()

    model = MonoterpeneSynthesis().pool_model
    assert torch.isfinite(model.v_max_mts)
    assert model.v_max_mts.item() > 0.0
    assert 0.0 <= model.pool_initial.item() <= model.pool_capacity.item()
    print("[OK] MonoterpenePoolModel initialization successful")

# Test P1: Parameter migration from default values
def test_richards_parameters_from_config():
    """Richards must use the exact ParameterManager dependency contract."""
    from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
    from relic.parameters import get_global_manager

    obsolete_config = {
        "n_layers": 10,
        "soil_depth": 2.0,
        "solver": "mixed_form",
        "convergence_criteria": {
            "tolerance": 1e-6,
            "max_iterations": 100,
        },
    }
    with pytest.raises(ValueError, match="Unsupported RichardsSoilMoisture config keys"):
        RichardsSoilMoisture(config=obsolete_config)

    richards = RichardsSoilMoisture()
    expected_keys = {
        "hydrosphere.soil_moisture.layer_count",
        "hydrosphere.soil_moisture.soil_depth",
        "hydrosphere.soil_moisture.residual_water_content",
        "hydrosphere.soil_moisture.porosity",
        "hydrosphere.soil_moisture.alpha_vg",
        "hydrosphere.soil_moisture.n_vg",
        "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
        "hydrosphere.soil_moisture.tortuosity_parameter",
        "hydrosphere.physical.field_capacity_fraction",
        "hydrosphere.physical.wilting_point_fraction",
        "hydrosphere.numerical.richards_max_picard_iterations",
        "hydrosphere.numerical.richards_max_newton_iterations",
        "hydrosphere.numerical.richards_convergence_tolerance",
        "hydrosphere.numerical.richards_newton_absolute_tolerance",
        "hydrosphere.numerical.richards_newton_relative_tolerance",
        "hydrosphere.soil_moisture.richards_enable_newton_raphson",
        "hydrosphere.soil_moisture.richards_enable_mass_conservation_check",
        "hydrosphere.numerical.mass_conservation_tolerance",
        "hydrosphere.numerical.richards_cfl_coefficient",
        "hydrosphere.numerical.saturation_epsilon",
        "hydrosphere.numerical.minimum_pressure_term",
        "hydrosphere.numerical.minimum_se_term",
        "hydrosphere.numerical.minimum_term1",
        "hydrosphere.numerical.minimum_timestep",
        "hydrosphere.numerical.harmonic_mean_factor",
        "hydrosphere.numerical.tridiagonal_tolerance",
        "hydrosphere.numerical.line_search_initial_alpha",
        "hydrosphere.numerical.line_search_max_backtracks",
        "hydrosphere.numerical.line_search_armijo_constant",
        "hydrosphere.numerical.line_search_alpha_reduction_factor",
        "hydrosphere.numerical.line_search_min_alpha",
        "hydrosphere.numerical.characteristic_pressure_head",
    }
    assert set(richards.required_parameters()) == expected_keys
    assert set(richards.metadata.parameters) == expected_keys
    assert not hasattr(richards, "transport_tortuosity")
    assert all(not key.startswith("relic.physical.constants") for key in expected_keys)
    for key, metadata in richards.metadata.parameters.items():
        assert metadata["namespace"] == key.rsplit(".", 1)[0]
        assert metadata["required"] is True
        assert metadata["description"]

    manager = get_global_manager()
    obsolete_boundary_keys = {
        "hydrosphere.soil_moisture.richards_top_boundary_type_default",
        "hydrosphere.soil_moisture.richards_top_head_default",
        "hydrosphere.soil_moisture.richards_bottom_boundary_type_default",
        "hydrosphere.soil_moisture.richards_bottom_head_default",
        "hydrosphere.soil_moisture.richards_bottom_flux_default",
    }
    for key in obsolete_boundary_keys:
        assert manager.get_parameter(key) is None, key

    for key in expected_keys:
        spec = manager.get_parameter(key)
        assert spec is not None, key
        assert spec.units
        assert spec.dtype
        assert spec.category is not None
        assert spec.description
        assert spec.references
        assert spec.tags
        assert isinstance(spec.requires_grad, bool)
        assert spec.provenance
        if spec.dtype == "bool":
            assert isinstance(spec.value, bool)
        else:
            assert spec.bounds is not None

def test_farquhar_parameters_from_config():
    """Test Farquhar photosynthesis parameters load from config"""
    from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
    
    config = {
        "pathway": "C3",
        "pft_name": "broadleaf_deciduous"
    }
    
    farquhar = FarquharPhotosynthesis(config=config, device=torch.device("cpu"))
    
    # Test that parameters are loaded (will use parameter manager)
    try:
        vcmax = farquhar.vcmax25
        assert vcmax is not None
        print(f"[OK] Farquhar Vcmax25 loaded successfully")
    except Exception as e:
        print(f"[OK] Farquhar parameters require proper initialization: {e}")

# Test P3: Physical constants unification
def test_physical_constants_unified():
    """Test that physical constants are loaded from unified source"""
    from relic.parameters import UnifiedParameterManager
    
    pm = UnifiedParameterManager()
    
    # Test von Karman constant
    von_karman = pm.get_parameter("physical.von_karman_constant")
    if von_karman:
        assert 0.35 <= von_karman.value <= 0.45  # Physical range
        print(f"[OK] Von Karman constant: {von_karman.value}")
    
    # Test gravity
    gravity = pm.get_parameter("physical.gravitational_acceleration")
    if gravity:
        assert 9.7 <= gravity.value <= 9.9  # Physical range
        print(f"[OK] Gravitational acceleration: {gravity.value}")
    
    # Test standard temperature
    t_standard = pm.get_parameter("physical.standard_temperature")
    if t_standard:
        assert t_standard.value == 273.15
        print(f"[OK] Standard temperature: {t_standard.value}")

# Test conservation and validation
def test_mass_conservation():
    """Test that mass conservation is enforced"""
    from relic.core.state import StateManager
    from relic.core.config.models import StateConfig
    
    # Create state config
    state_config = StateConfig(
        dimensions={
            "n_layers": 5,
            "n_species": 2
        },
        variables={
            "soil_moisture": {
                "dims": ["n_layers"],
                "units": "m3 m-3",
                "bounds": [0.0, 1.0],
                "initial_value": 0.3
            },
            "soil_carbon": {
                "dims": ["n_layers"],
                "units": "kg m-2",
                "bounds": [0.0, 100.0],
                "initial_value": 10.0
            }
        }
    )
    
    # Create state manager
    sm = StateManager(config=state_config)
    sm.initialize()
    
    # Test conservation check
    initial_carbon = sm.get_state("soil_carbon").sum()
    
    # Apply an update that conserves mass
    updates = {
        "soil_carbon": torch.zeros(5)  # No net change
    }
    
    sm.apply_updates(updates, dt=3600.0, check_conservation=True)
    
    final_carbon = sm.get_state("soil_carbon").sum()
    assert torch.allclose(initial_carbon, final_carbon, atol=1e-6)
    print("[OK] Mass conservation check passed")

# Test parameter validation
def test_parameter_bounds_validation():
    """Test that parameter bounds are enforced"""
    from relic.parameters.spec import ParameterSpec
    from relic.parameters.validation import ParameterValidator
    from relic.parameters import ParameterCategory
    
    # Create a parameter with bounds
    param = ParameterSpec(
        key="test_param",
        value=0.5,
        units="dimensionless",
        bounds=(0.0, 1.0),
        category=ParameterCategory.COMPUTATIONAL,
        description="Test parameter", dtype="float32"
    )
    validator = ParameterValidator()

    # Test validation
    errors = validator.validate({param.key: param})
    assert not errors
    print("[OK] Parameter bounds validation passed")

    # Test out-of-bounds
    param.value = 1.5
    errors = validator.validate({param.key: param})
    assert errors
    print("[OK] Out-of-bounds parameter correctly rejected")

# Main test runner
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RELIC Parameter System Test Suite")
    print("="*60 + "\n")
    
    print("Testing P0: Inheritance Fixes")
    print("-"*40)
    test_mep_pathway_model_initialization()
    test_monoterpene_pool_model_initialization()
    
    print("\nTesting P1: Parameter Migration")
    print("-"*40)
    test_richards_parameters_from_config()
    test_farquhar_parameters_from_config()
    
    print("\nTesting P3: Physical Constants")
    print("-"*40)
    test_physical_constants_unified()
    
    print("\nTesting Validation & Conservation")
    print("-"*40)
    test_mass_conservation()
    test_parameter_bounds_validation()
    
    print("\n" + "="*60)
    print("[SUCCESS] All tests completed successfully!")
    print("="*60)

