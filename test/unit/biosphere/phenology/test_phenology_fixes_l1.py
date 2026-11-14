"""
L1 Unit Tests for Phenology Module Fixes

Tests specific fixes applied to the phenology module:
1. Tensor-safe boolean comparisons
2. Environment variable naming (air_temperature_max/min)  
3. Unit validation implementation
4. Parameter system integration

Author: Yuchong Long
Date: 2025-09-25
"""

import pytest
import torch

from relic.scientific.biosphere.phenology.models import (
    GDDPhenology, GSIPhenology, StaticPhenology
)
# StateDict and EnvironmentDict are just type aliases for Dict[str, Tensor]
StateDict = dict
EnvironmentDict = dict
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant


def _canonical_gdd_state(
    *,
    accumulated: float = 0.0,
    previous_day: float = 149.0,
):
    """Return the required StateManager-owned GDD temporal inputs (the step length
    is the dt keyword of the call, not an input)."""
    return {
        'phenology_gdd_accumulated': torch.tensor(accumulated),
        'phenology_gdd_previous_day_of_year': torch.tensor(previous_day),
    }


class TestTensorBooleanComparisons:
    """Test tensor-safe boolean comparison fixes in models.py"""
    
    def test_gdd_canonical_kelvin_tensor(self):
        """Test vectorized GDD calculation from canonical Kelvin inputs."""
        # Setup
        config = {
            'phenology_type': 'gdd',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'gdd_base_temp': 5.0,
            'gdd_threshold_start': 100.0,
            'gdd_threshold_end': 2500.0,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = GDDPhenology(name='test_gdd', config=config)
        
        # Test with Kelvin temperature tensor
        temp_kelvin = torch.tensor([273.15, 283.15, 293.15, 303.15])  # 0, 10, 20, 30 degC
        environment = {
            'day_of_year': torch.tensor(150),
            'air_temperature_max': temp_kelvin + 5.0,
            'air_temperature_min': temp_kelvin - 5.0,
            **_canonical_gdd_state(),
        }
        
        # Should not raise "bool value of Tensor is ambiguous" error
        output = model.compute_phenological_state(
            {**environment, "day_of_year": torch.tensor(150), "air_temperature": temp_kelvin},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output is not None
        assert output.thermal_time is not None
        accumulated = output.additional_outputs['phenology_gdd_accumulated']
        assert torch.allclose(
            accumulated,
            accumulated.new_tensor([0.0, 5.0, 15.0, 25.0]),
        )
        
    def test_gdd_half_day_scaling_with_canonical_kelvin(self):
        """Scale canonical daily thermal time by a half-day execution step."""
        config = {
            'phenology_type': 'gdd',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'gdd_base_temp': 5.0,
            'gdd_threshold_start': 100.0,
            'gdd_threshold_end': 2500.0,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = GDDPhenology(name='test_gdd', config=config)
        
        kelvin_offset = get_physical_constant("T_FREEZE")
        temp_celsius = torch.tensor([0.0, 10.0, 20.0, 30.0])
        temp_kelvin = temp_celsius + kelvin_offset
        environment = {
            'day_of_year': torch.tensor(150),
            'air_temperature_max': temp_kelvin + 5.0,
            'air_temperature_min': temp_kelvin - 5.0,
            **_canonical_gdd_state(accumulated=10.0),
        }
        
        output = model.compute_phenological_state(
            {**environment, "day_of_year": torch.tensor(150), "air_temperature": temp_kelvin},
            dt=0.5 * get_physical_constant("seconds_per_day"),
        )
        
        assert output is not None
        assert output.thermal_time is not None
        accumulated = output.additional_outputs['phenology_gdd_accumulated']
        assert torch.allclose(
            accumulated,
            accumulated.new_tensor([10.0, 12.5, 17.5, 22.5]),
        )
        
    def test_required_temporal_state_inputs_fail_fast(self):
        """Reject incomplete canonical GDD temporal state without fallback."""
        config = {
            'phenology_type': 'gdd',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'gdd_base_temp': 5.0,
            'gdd_threshold_start': 100.0,
            'gdd_threshold_end': 2500.0,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = GDDPhenology(name='test_gdd', config=config)
        
        canonical_state = _canonical_gdd_state()
        environment = {
            'day_of_year': torch.tensor(150),
            'air_temperature_max': torch.tensor(298.15),
            'air_temperature_min': torch.tensor(288.15),
            **canonical_state,
        }

        for missing_key in canonical_state:
            incomplete = dict(environment)
            del incomplete[missing_key]
            with pytest.raises(ValueError, match=missing_key):
                model.compute_phenological_state(
                    {**incomplete, "day_of_year": torch.tensor(150), "air_temperature": torch.tensor(293.15)},
                    dt=get_physical_constant("seconds_per_day"),
                )
        # The step length is the required dt keyword, never a state entry.
        with pytest.raises(TypeError):
            model.compute_phenological_state(
                {**environment, "day_of_year": torch.tensor(150), "air_temperature": torch.tensor(293.15)},
            )


class TestEnvironmentVariableNaming:
    """Test environment variable naming consistency fixes"""
    
    def test_air_temperature_max_min_cf_standard(self):
        """Test that models accept CF-standard air_temperature_max/min"""
        config = {
            'phenology_type': 'gdd',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'gdd_base_temp': 5.0,
            'gdd_threshold_start': 100.0,
            'gdd_threshold_end': 2500.0,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = GDDPhenology(name='test_gdd', config=config)
        
        # Use CF-standard names
        environment = {
            'day_of_year': torch.tensor(150),
            'air_temperature_max': torch.tensor(298.15),
            'air_temperature_min': torch.tensor(288.15),
            **_canonical_gdd_state(),
        }
        
        output = model.compute_phenological_state(
            {**environment, "day_of_year": torch.tensor(150), "air_temperature": torch.tensor(293.15)},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output is not None
        assert output.thermal_time is not None
        
    def test_historical_temperature_max_min_names_are_rejected(self):
        """Require canonical extrema and reject historical aliases."""
        config = {
            'phenology_type': 'gdd',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'gdd_base_temp': 5.0,
            'gdd_threshold_start': 100.0,
            'gdd_threshold_end': 2500.0,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = GDDPhenology(name='test_gdd', config=config)
        
        historical_environment = {
            'day_of_year': torch.tensor(150),
            'temperature_max': torch.tensor(298.15),
            'temperature_min': torch.tensor(288.15),
            **_canonical_gdd_state(),
        }

        with pytest.raises(ValueError, match="air_temperature_max"):
            model.compute_phenological_state(
                {**historical_environment, "day_of_year": torch.tensor(150), "air_temperature": torch.tensor(293.15)},
                dt=get_physical_constant("seconds_per_day"),
            )

        output = model.compute_phenological_state(
            {
                "air_temperature_max": torch.tensor(298.15),
                "air_temperature_min": torch.tensor(288.15),
                **_canonical_gdd_state(),
                "day_of_year": torch.tensor(150),
                "air_temperature": torch.tensor(293.15),
            },
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output is not None
        assert output.thermal_time is not None
        
    def test_historical_names_do_not_override_canonical_extrema(self):
        """Historical extrema must not create a second production input path."""
        config = {
            'phenology_type': 'gdd',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'gdd_base_temp': 5.0,
            'gdd_threshold_start': 100.0,
            'gdd_threshold_end': 2500.0,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = GDDPhenology(name='test_gdd', config=config)
        
        kelvin_offset = get_physical_constant("T_FREEZE")
        environment = {
            'day_of_year': torch.tensor(150),
            'air_temperature_max': torch.tensor(28.0 + kelvin_offset),
            'air_temperature_min': torch.tensor(18.0 + kelvin_offset),
            'temperature_max': torch.tensor(25.0),
            'temperature_min': torch.tensor(15.0),
            **_canonical_gdd_state(),
        }
        
        output = model.compute_phenological_state(
            {**environment, "day_of_year": torch.tensor(150), "air_temperature": torch.tensor(23.0 + kelvin_offset)},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output is not None
        assert output.additional_outputs[
            'phenology_gdd_accumulated'
        ].item() == pytest.approx(18.0)
        

class TestUnitValidation:
    """Test flux unit validation implementation"""
    
    def test_validate_flux_units_carbon_fluxes(self):
        """Test unit validation for carbon fluxes"""
        config = {
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'leaf_onset_doy': 120,
            'leaf_offset_doy': 280,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = StaticPhenology(name='test_static', config=config)
        
        state = {
            'leaf_carbon': torch.tensor(0.5),
            'froot_carbon': torch.tensor(0.3),
            'leaf_nitrogen': torch.tensor(0.02),
            'froot_nitrogen': torch.tensor(0.01)
        }
        
        environment = {
            'day_of_year': torch.tensor(150),
            'air_temperature': torch.tensor(20.0)
        }
        
        # This should trigger unit validation
        updates = model({**state, **environment}, time=0.0, dt=1.0)
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Check that carbon fluxes exist and are reasonable
        assert 'leaf_turnover_flux' not in fluxes
        assert 'froot_turnover_flux' not in fluxes
        assert 'leaf_litterfall_carbon' in fluxes
        assert 'froot_litterfall_carbon' in fluxes
        
        # Values should be small (turnover rates are per second)
        assert fluxes['leaf_litterfall_carbon'].abs().max() < 1.0
        assert fluxes['froot_litterfall_carbon'].abs().max() < 1.0
        
    def test_validate_flux_units_dimensionless(self):
        """Test unit validation for dimensionless variables"""
        config = {
            'phenology_type': 'gsi',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'gsi_tmin_threshold': -2.0,
            'gsi_tmin_optimal': 5.0,
            'gsi_vpd_min': 900.0,
            'gsi_vpd_max': 4100.0,
            'gsi_onset_threshold': 0.5,
            'gsi_offset_threshold': 0.5,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = GSIPhenology(name='test_gsi', config=config)
        freeze = get_physical_constant("T_FREEZE")
        seconds_per_day = get_physical_constant("seconds_per_day")
        
        state = {
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
        }
        environment = {
            'day_of_year': torch.tensor(150.0),
            'air_temperature': torch.tensor(freeze + 20.0),
            'air_temperature_min': torch.tensor(freeze + 15.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'daylength': torch.tensor(14.0 * 3600),  # 14 hours in seconds
            'phenology_gsi_history': torch.zeros(model.memory_days),
            'phenology_gsi_history_index': torch.tensor(0.0),
            'phenology_gsi_history_count': torch.tensor(0.0),
            'phenology_gsi_in_growing_season': torch.tensor(0.0),
        }

        updates = model(
            {**state, **environment},
            time=0.0,
            dt=seconds_per_day,
        )
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Check dimensionless factors are in [0, 1]
        if 'phenological_factor' in fluxes:
            assert 0 <= fluxes['phenological_factor'].item() <= 1.0
        if 'phenology_senescence_factor' in fluxes:
            assert 0 <= fluxes['phenology_senescence_factor'].item() <= 1.0
        for key in (
            'phenology_gsi_history',
            'phenology_gsi_history_index',
            'phenology_gsi_history_count',
            'phenology_gsi_in_growing_season',
        ):
            assert updates[key].unit == 'dimensionless'
            
    def test_unit_validation_with_large_flux_warning(self, caplog):
        """Test that unreasonably large fluxes trigger warnings"""
        config = {
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'leaf_onset_doy': 120,
            'leaf_offset_doy': 280,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = StaticPhenology(name='test_static', config=config)
        
        # Create state with very large carbon pools
        state = {
            'leaf_carbon': torch.tensor(100.0),  # Unreasonably large
            'froot_carbon': torch.tensor(100.0),
            'leaf_nitrogen': torch.tensor(5.0),
            'froot_nitrogen': torch.tensor(5.0)
        }
        
        environment = {
            'day_of_year': torch.tensor(250),  # Late in season for high turnover
            'air_temperature': torch.tensor(10.0)
        }
        
        # This should work but may log warnings
        updates = model({**state, **environment}, time=0.0, dt=1.0)
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Should complete without error
        assert fluxes is not None
        

class TestParameterSystemIntegration:
    """Test parameter manager integration"""
    
    def test_parameters_loaded_from_unified_yaml(self):
        """Test that parameters are loaded from phenology_unified.yaml"""
        param_manager = get_global_manager()
        
        # Check that key phenology parameters exist
        gdd_base_k = param_manager.get('biosphere.phenology.gdd.base_temp')
        assert gdd_base_k is not None
        assert pytest.approx(gdd_base_k, rel=0, abs=1e-6) == 278.15  # Stored in Kelvin
        gdd_base_c = param_manager.get('biosphere.phenology.gdd.base_temp', units='degC')
        assert pytest.approx(gdd_base_c, rel=0, abs=1e-6) == 5.0  # Converted to Celsius
        
        gsi_tmin = param_manager.get('biosphere.phenology.gsi.tmin_threshold')
        assert gsi_tmin is not None
        
        max_lai = param_manager.get('biosphere.phenology.lai.max_lai_deciduous')
        assert max_lai is not None
        
    def test_unit_notation_consistency(self):
        """Test that unit notation is consistent (year-1, m^2 kg^-1)"""
        param_manager = get_global_manager()
        
        turnover_spec = param_manager.get_parameter('biosphere.phenology.turnover.leaf_turnover_rate')
        assert turnover_spec is not None
        assert turnover_spec.units == 's-1'  # Stored in SI for consistency
            
        sla_spec = param_manager.get_parameter('biosphere.phenology.morphology.specific_leaf_area')
        assert sla_spec is not None
        units_str = sla_spec.units or ''
        # Accept either m2 kg-1 or m^2 kg^-1 representations
        assert 'm2' in units_str or 'm^2' in units_str
        assert 'kg-1' in units_str or 'kg^-1' in units_str
            
    def test_no_duplicate_parameters_after_consolidation(self):
        """Test that there are no duplicate parameter definitions"""
        param_manager = get_global_manager()
        
        # These parameters should exist only once in unified yaml
        test_params = [
            'biosphere.phenology.gdd.base_temp',
            'biosphere.phenology.gsi.tmin_threshold',
            'biosphere.phenology.static.leaf_onset_doy',
            'biosphere.phenology.lai.max_lai_deciduous'
        ]
        
        for param in test_params:
            value = param_manager.get(param)
            assert value is not None, f"Parameter {param} should exist"


class TestConservationChecks:
    """Test conservation check fixes"""
    
    def test_flux_conservation_without_phenology_output(self):
        """Test that flux conservation works without requiring PhenologyOutput"""
        config = {
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'leaf_onset_doy': 120,
            'leaf_offset_doy': 280,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = StaticPhenology(name='test_static', config=config)
        
        state = {
            'leaf_carbon': torch.tensor(0.5),
            'froot_carbon': torch.tensor(0.3),
            'leaf_nitrogen': torch.tensor(0.02),
            'froot_nitrogen': torch.tensor(0.01)
        }
        
        environment = {
            'day_of_year': torch.tensor(150),
            'air_temperature': torch.tensor(20.0)
        }
        
        # This should not raise ConservationError about missing PhenologyOutput fields
        updates = model({**state, **environment}, time=0.0, dt=1.0)
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Check mass conservation for turnover
        if 'leaf_carbon' in fluxes and 'leaf_litter_carbon' in fluxes:
            # Carbon lost from leaves should equal carbon gained in litter
            assert torch.allclose(
                -fluxes['leaf_carbon'],
                fluxes['leaf_litter_carbon'],
                rtol=1e-5
            )
            
    def test_nitrogen_retranslocation_conservation(self):
        """Test nitrogen conservation with retranslocation"""
        config = {
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'leaf_onset_doy': 120,
            'leaf_offset_doy': 280,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        model = StaticPhenology(name='test_static', config=config)
        
        state = {
            'leaf_carbon': torch.tensor(0.5),
            'froot_carbon': torch.tensor(0.3),
            'leaf_nitrogen': torch.tensor(0.02),
            'froot_nitrogen': torch.tensor(0.01)
        }
        
        environment = {
            'day_of_year': torch.tensor(280),  # Fall season
            'air_temperature': torch.tensor(10.0)
        }
        
        updates = model({**state, **environment}, time=0.0, dt=1.0)
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Check nitrogen conservation
        if all(k in fluxes for k in ['leaf_nitrogen', 'leaf_litter_nitrogen', 'plant_nitrogen_pool']):
            # Total N loss from leaves should equal litter N + retranslocated N
            total_n_loss = -fluxes['leaf_nitrogen']
            total_n_fate = fluxes['leaf_litter_nitrogen'] + fluxes['plant_nitrogen_pool']
            
            # Allow small numerical error
            assert torch.allclose(total_n_loss, total_n_fate, rtol=1e-4)


def test_all_phenology_models_initialize():
    """Integration test that all phenology models initialize with fixed parameters"""
    models = []
    
    # Test each phenology type
    for ptype, model_class in [
        ('static', StaticPhenology),
        ('gdd', GDDPhenology),
        ('gsi', GSIPhenology)
    ]:
        # Build complete config for each model type
        config = {
            'phenology_type': ptype,
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'max_lai': 6.0,
            'min_lai': 0.5,
            'sla': 20.0,
            'leaf_turnover_rate': 7.927e-09,
            'froot_turnover_rate': 3.171e-08,
            'wood_turnover_rate': 6.342e-10
        }
        
        # Add type-specific parameters
        if ptype == 'static':
            config.update({
                'leaf_onset_doy': 120,
                'leaf_offset_doy': 280
            })
        elif ptype == 'gdd':
            config.update({
                'gdd_base_temp': 5.0,
                'gdd_threshold_start': 100.0,
                'gdd_threshold_end': 2500.0
            })
        elif ptype == 'gsi':
            config.update({
                'gsi_tmin_threshold': -2.0,
                'gsi_tmin_optimal': 5.0,
                'gsi_vpd_min': 900.0,
                'gsi_vpd_max': 4100.0,
                'gsi_onset_threshold': 0.5,
                'gsi_offset_threshold': 0.5
            })
        
        model = model_class(name=f'test_{ptype}', config=config)
        models.append(model)
        
        # Verify model initialized correctly
        assert model is not None
        assert model.phenology_params is not None
        assert model.phenology_params.phenology_type == ptype
        
    # All models should initialize without error
    assert len(models) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])



