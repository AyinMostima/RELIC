"""
Test phenology module parameter system integration
Tests that all phenology models properly use ParameterManager
"""

import sys
import torch
from pathlib import Path

# Add relic to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from relic.scientific.biosphere.phenology.models import (
    StaticPhenology, GDDPhenology, GSIPhenology, CropPhenology
)
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant


def test_static_phenology():
    """Test StaticPhenology model initialization and execution"""
    print("Testing StaticPhenology...")
    
    # Create model
    model = StaticPhenology(name="test_static")
    
    # Create test inputs
    day_of_year = torch.tensor(150.0)  # Day 150 (late May)
    temperature = torch.tensor(288.15)  # 15C in Kelvin
    environment = {
        'day_of_year': day_of_year,
        'air_temperature': temperature
    }
    
    # Compute phenological state
    output = model.compute_phenological_state(
        {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
        dt=get_physical_constant("seconds_per_day"),
    )
    
    # Check output
    assert output.phenology_phase is not None
    assert output.leaf_area_index is not None
    assert output.growth_factor is not None
    assert output.senescence_rate is not None
    
    print(f"  Phase: {output.phenology_phase}")
    print(f"  LAI: {output.leaf_area_index.item():.3f}")
    print(f"  Growth factor: {output.growth_factor.item():.3f}")
    print("  [OK] StaticPhenology test passed")
    
def test_gdd_phenology():
    """Test GDDPhenology model initialization and execution"""
    print("\nTesting GDDPhenology...")
    
    # Create model
    model = GDDPhenology(name="test_gdd")
    
    # Create test inputs
    day_of_year = torch.tensor(150.0)
    temperature = torch.tensor(293.15)  # 20C in Kelvin
    environment = {
        'day_of_year': day_of_year,
        'air_temperature': temperature,
        'air_temperature_max': torch.tensor(298.15),  # 25C
        'air_temperature_min': torch.tensor(288.15),  # 15C
        'phenology_gdd_accumulated': torch.tensor(0.0),
        'phenology_gdd_previous_day_of_year': torch.tensor(149.0),
    }
    
    # Compute phenological state
    output = model.compute_phenological_state(
        {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
        dt=torch.tensor(
            get_physical_constant("seconds_per_day")
        ),
    )
    
    # Check output
    assert output.phenology_phase is not None
    assert output.thermal_time is not None
    assert output.thermal_time.item() == 15.0
    assert output.leaf_area_index is not None
    assert output.growth_factor is not None
    
    print(f"  Phase: {output.phenology_phase}")
    print(f"  Thermal time: {output.thermal_time.item():.1f}")
    print(f"  LAI: {output.leaf_area_index.item():.3f}")
    print("  [OK] GDDPhenology test passed")
    
def test_gsi_phenology():
    """Test GSIPhenology model initialization and execution"""
    print("\nTesting GSIPhenology...")
    
    # Create model
    model = GSIPhenology(name="test_gsi")
    
    # Create test inputs
    day_of_year = torch.tensor(150.0)
    temperature = torch.tensor(293.15)  # 20C in Kelvin
    environment = {
        'day_of_year': day_of_year,
        'air_temperature': temperature,
        'air_temperature_min': torch.tensor(288.15),  # 15C in Kelvin
        'vapor_pressure_deficit': torch.tensor(1500.0),  # Pa
        'daylength': torch.tensor(50000.0),  # ~14 hours
        'phenology_gsi_history': torch.zeros(model.memory_days),
        'phenology_gsi_history_index': torch.tensor(0.0),
        'phenology_gsi_history_count': torch.tensor(0.0),
        'phenology_gsi_in_growing_season': torch.tensor(0.0),
    }
    
    # Compute phenological state
    output = model.compute_phenological_state(
        {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
        dt=torch.tensor(
            get_physical_constant("seconds_per_day")
        ),
    )
    
    # Check output
    assert output.phenology_phase is not None
    assert output.growing_season_index is not None
    assert output.leaf_area_index is not None
    assert output.growth_factor is not None
    
    print(f"  Phase: {output.phenology_phase}")
    print(f"  GSI: {output.growing_season_index.item():.3f}")
    print(f"  LAI: {output.leaf_area_index.item():.3f}")
    print("  [OK] GSIPhenology test passed")
    
def test_crop_phenology():
    """Test CropPhenology model initialization and execution"""
    print("\nTesting CropPhenology...")
    
    # Create model
    model = CropPhenology(name="test_crop")
    
    # Create test inputs - sowing day
    day_of_year = torch.tensor(120.0)  # Sowing day
    temperature = torch.tensor(288.15)  # 15C in Kelvin
    environment = {
        'day_of_year': day_of_year,
        'air_temperature': temperature,
        'phenology_crop_accumulated_phu': torch.tensor(0.0),
        'phenology_crop_is_sown': torch.tensor(0.0),
        'phenology_crop_is_harvested': torch.tensor(0.0),
        'phenology_crop_vernalization_days': torch.tensor(0.0),
    }
    
    # Compute phenological state at sowing
    output = model.compute_phenological_state(
        {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
        dt=torch.tensor(
            get_physical_constant('seconds_per_day')
        ),
    )
    
    # Check output
    assert output.phenology_phase is not None
    assert output.thermal_time is not None
    assert 'development_index' in output.diagnostic_variables
    assert 'harvest_index' in output.diagnostic_variables
    assert output.additional_outputs[
        'phenology_crop_accumulated_phu'
    ].item() == 10.0
    assert output.additional_outputs['phenology_crop_is_sown'].item() == 1.0
    
    print(f"  Phase: {output.phenology_phase}")
    print(f"  Development index: {output.diagnostic_variables['development_index'].item():.3f}")
    print(f"  Is sown: {output.diagnostic_variables['is_sown'].item()}")
    print("  [OK] CropPhenology test passed")
    
def test_parameter_loading():
    """Test that parameters are properly loaded from ParameterManager"""
    print("\nTesting parameter loading from ParameterManager...")
    
    param_manager = get_global_manager()
    
    # Check some key parameters exist
    test_params = [
        'biosphere.phenology.static.leaf_onset_doy',
        'biosphere.phenology.gdd.base_temp',
        'biosphere.phenology.gsi.tmin_threshold',
        'biosphere.phenology.crop.phu_total_maize',
        'biosphere.phenology.lai.max_lai_deciduous',
        'biosphere.phenology.turnover.leaf_turnover_rate'
    ]
    
    for param_key in test_params:
        value = param_manager.get(param_key, required=True)
        assert value is not None
        print(f"  {param_key}: {value} [OK]")
    
    print("  [OK] All parameters loaded successfully")


def main():
    """Run all tests"""
    print("=" * 60)
    print("PHENOLOGY PARAMETER SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    test_parameter_loading()
    test_static_phenology()
    test_gdd_phenology()
    test_gsi_phenology()
    test_crop_phenology()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - Phenology module is working correctly")
    print("Parameter system integration is successful")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

